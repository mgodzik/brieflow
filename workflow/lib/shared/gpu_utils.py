"""Utility functions for reserving a GPU device.

The original implementation reserved a GPU by setting the
``CUDA_VISIBLE_DEVICES`` environment variable.  This would hide all other
devices from the current process.  In order to cooperate better with code
that expects all devices to be visible, the active device is now selected
directly via :func:`torch.cuda.set_device` while leaving the visibility
of the remaining devices untouched.

The function :func:`_reserve_gpu` keeps the lock acquisition logic of the
previous version.  Each GPU device is protected by a lock file which
prevents concurrent reservation from multiple processes.  When a device
with sufficient free memory is found, it is set as the active device and
the lock is held until the interpreter exits.  Devices are considered in
random order to spread allocations across GPUs.
"""

from __future__ import annotations

import atexit
import random
import json
import tempfile
import time
from pathlib import Path
from typing import Dict
import fcntl
import torch

# from filelock import FileLock, Timeout

# keep references to locks for the lifetime of the process so that they
# are not garbage collected and released prematurely
# _GPU_LOCKS: Dict[int, FileLock] = {}

#     The lock files are stored inside the system temporary directory.
#     """
#     lock_dir = Path(tempfile.gettempdir()) / "gpu_locks"
#     lock_dir.mkdir(parents=True, exist_ok=True)
#     return lock_dir / f"gpu{dev_id}.lock"

# def _release_gpu(dev_id: int) -> None:
# """Release the lock for ``dev_id`` if it is held."""
# lock = _GPU_LOCKS.pop(dev_id, None)
# if lock is not None and lock.is_locked:
#     try:
#         lock.release()
#     except Exception:
#         pass
# """Release the ledger entry for ``dev_id`` if it exists."""
# reserved = _GPU_RESERVATIONS.pop(dev_id, None)
# if reserved is None:
#     return

# Track memory reservations (in MB) made by this process.
_GPU_RESERVATIONS: Dict[int, int] = {}

# Path to the ledger file coordinating reservations across processes.
LEDGER_PATH = Path(tempfile.gettempdir()) / "gpu_ledger.json"


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


def _release_gpu(dev_id: int) -> None:
    reserved = _GPU_RESERVATIONS.pop(dev_id, 0)
    if reserved == 0:
        return
    LEDGER_PATH.touch(exist_ok=True)
    with open(LEDGER_PATH, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        ledger = _read_ledger(f)
        current = ledger.get(str(dev_id), 0)
        new_val = max(0, current - reserved)
        if new_val:
            ledger[str(dev_id)] = new_val
        else:
            ledger.pop(str(dev_id), None)
        _write_ledger(f, ledger)
        fcntl.flock(f, fcntl.LOCK_UN)


def _reserve_gpu(
    min_free_mem: int = 0,
    lock_timeout: int | float = 3600,
    check_interval: int | float = 1,
    recheck_interval: int | float = 60,
) -> int:
    """Reserve a GPU with at least ``min_free_mem`` MB of free memory.

    This function iterates over all visible CUDA devices in random order,
    checks the free memory reported by ``torch.cuda.mem_get_info`` and the
    current reservations from the ledger. When a device is found that
    satisfies the free memory constraint, the reservation is recorded in
    the ledger and the device is set as the active one via
    :func:`torch.cuda.set_device`. The reservation is released when the
    process exits.

    Parameters
    ----------
    min_free_mem:
        The minimum amount of free memory in **MB** required on the device.
    lock_timeout:
        Maximum time in seconds to wait for a free GPU.  A value of ``0``
        means to retry indefinitely.
    check_interval:
        Time in seconds to wait between successive scans of all devices.
    recheck_interval:
        Unused placeholder kept for backwards compatibility.

    Returns
    -------
    int
        The device id of the reserved GPU.

    Raises
    ------
    RuntimeError
        If no suitable GPU is found before ``lock_timeout`` expires.

    """
    import importlib

    start = time.time()
    deadline = None if lock_timeout == 0 else start + lock_timeout

    required_mem = int(min_free_mem)

    while True:
        dev_ids = list(range(torch.cuda.device_count()))
        random.shuffle(dev_ids)
        for dev_id in dev_ids:
            free_mem_bytes, _ = torch.cuda.mem_get_info(dev_id)
            free_mem_mb = free_mem_bytes // (1024**2)

            LEDGER_PATH.touch(exist_ok=True)
            with open(LEDGER_PATH, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                ledger = _read_ledger(f)
                used_mb = ledger.get(str(dev_id), 0)
                available_mb = free_mem_mb - used_mb
                if available_mb >= required_mem:
                    ledger[str(dev_id)] = used_mb + required_mem
                    _write_ledger(f, ledger)
                    fcntl.flock(f, fcntl.LOCK_UN)

                    torch.cuda.set_device(dev_id)
                    _GPU_RESERVATIONS[dev_id] = required_mem
                    atexit.register(_release_gpu, dev_id)
                    return dev_id
                fcntl.flock(f, fcntl.LOCK_UN)

        if deadline is not None and time.time() > deadline:
            raise RuntimeError("No GPU with sufficient memory available")

        time.sleep(check_interval)


__all__ = ["_reserve_gpu", "_release_gpu"]

# import os
# import random
# import atexit
# import tempfile
# from pathlib import Path

# import torch

# LOCK_DIR = Path(tempfile.gettempdir()) / "brieflow-gpu-locks"
# LOCK_DIR.mkdir(parents=True, exist_ok=True)


# def _lock_path(dev_id: int) -> Path:
#     """Return the lock file path for a given device id."""
#     return LOCK_DIR / f"gpu_{dev_id}.lock"


# def _acquire_lock(dev_id: int):
#     """Try to acquire a lock for the given GPU device.

#     Returns a file descriptor if the lock was acquired, otherwise ``None``.
#     """
#     lock = _lock_path(dev_id)
#     try:
#         fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
#     except FileExistsError:
#         return None
#     os.write(fd, str(os.getpid()).encode())
#     return fd


# def _release_lock(dev_id: int, fd: int) -> None:
#     """Release the lock for ``dev_id``."""
#     os.close(fd)
#     try:
#         os.remove(_lock_path(dev_id))
#     except FileNotFoundError:
#         pass


# def _register_cleanup(dev_id: int, fd: int) -> None:
#     """Register cleanup handler for the lock."""
#     def _cleanup():
#         _release_lock(dev_id, fd)
#     atexit.register(_cleanup)


# def _reserve_gpu(min_free_mem: int = 0) -> int:
#     """Reserve a GPU with at least ``min_free_mem`` free memory.

#     Returns the device id and sets ``CUDA_VISIBLE_DEVICES`` so the process
#     sees only the selected device.
#     """
#     if torch.cuda.device_count() == 0:
#         raise RuntimeError("No CUDA devices available")

#     device_ids = list(range(torch.cuda.device_count()))
#     random.shuffle(device_ids)

#     for dev_id in device_ids:
#         fd = _acquire_lock(dev_id)
#         if fd is None:
#             continue
#         try:
#             try:
#                 torch.cuda.set_device(dev_id)
#                 free_mem, _ = torch.cuda.mem_get_info()
#             except Exception:
#                 free_mem = 0
#             if free_mem >= min_free_mem:
#                 os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_id)
#                 _register_cleanup(dev_id, fd)
#                 return dev_id
#         finally:
#             # If we didn't return, release the lock immediately
#             if _lock_path(dev_id).exists() and os.environ.get("CUDA_VISIBLE_DEVICES") != str(dev_id):
#                 _release_lock(dev_id, fd)
#     raise RuntimeError("No GPU with enough free memory available")


# # Public aliases for compatibility with potential callers
# reserve_gpu = _reserve_gpu
# acquire_gpu = _reserve_gpu
# get_free_gpu = _reserve_gpu
