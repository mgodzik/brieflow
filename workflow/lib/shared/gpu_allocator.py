"""GPU allocation utilities for coordinating Snakemake jobs."""

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
import fcntl

LEDGER_PATH = Path("/tmp/gpu_allocator.json")


def _query_free_memory():
    """Return list of free GPU memory in MB using nvidia-smi.

    Returns an empty list if nvidia-smi is unavailable or no GPUs detected.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    return [int(l) for l in lines]


def _read_ledger(file_obj):
    file_obj.seek(0)
    try:
        return json.load(file_obj)
    except Exception:
        return {}


def _write_ledger(file_obj, data):
    file_obj.seek(0)
    file_obj.truncate()
    json.dump(data, file_obj)
    file_obj.flush()


@contextmanager
def reserve_gpu(required_mem_mb, poll_interval=5):
    """Reserve a GPU with at least ``required_mem_mb`` free memory.

    This function blocks until a GPU with sufficient free memory is found.
    The chosen GPU ID is exported via ``CUDA_VISIBLE_DEVICES`` so that
    downstream libraries automatically use the correct device.

    Args:
        required_mem_mb (int): Amount of memory in MB required for the job.
        poll_interval (int, optional): Seconds to wait between checks. Defaults to 5.
    """
    required_mem_mb = int(required_mem_mb)
    if required_mem_mb <= 0:
        # No reservation required
        yield None
        return

    while True:
        free_mem = _query_free_memory()
        if not free_mem:
            raise RuntimeError("No GPUs detected or nvidia-smi not found")

        for idx, mem in enumerate(free_mem):
            LEDGER_PATH.touch(exist_ok=True)
            with open(LEDGER_PATH, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                ledger = _read_ledger(f)
                used = ledger.get(str(idx), 0)
                available = mem - used
                if available >= required_mem_mb:
                    ledger[str(idx)] = used + required_mem_mb
                    _write_ledger(f, ledger)
                    fcntl.flock(f, fcntl.LOCK_UN)

                    prev_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
                    try:
                        yield idx
                    finally:
                        with open(LEDGER_PATH, "r+") as f2:
                            fcntl.flock(f2, fcntl.LOCK_EX)
                            ledger2 = _read_ledger(f2)
                            ledger2[str(idx)] = max(
                                0, ledger2.get(str(idx), 0) - required_mem_mb
                            )
                            _write_ledger(f2, ledger2)
                            fcntl.flock(f2, fcntl.LOCK_UN)
                        if prev_visible is not None:
                            os.environ["CUDA_VISIBLE_DEVICES"] = prev_visible
                        else:
                            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    return
                else:
                    fcntl.flock(f, fcntl.LOCK_UN)
        time.sleep(poll_interval)