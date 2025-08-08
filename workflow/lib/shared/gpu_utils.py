# GPU utility functions for reserving CUDA devices based on free memory.

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

import torch


@contextmanager
def reserve_gpu(required_mem_mb: int, lock_dir: str = "/tmp/gpu_locks"):
    """Reserve a GPU device with at least ``required_mem_mb`` free memory.

    This function enumerates all visible CUDA devices and selects the first one
    that satisfies the requested free memory. A simple file based lock is used
    to avoid multiple processes reserving the same device concurrently.

    Args:
        required_mem_mb: Minimum amount of free memory required (in megabytes).
        lock_dir: Directory where lock files will be created.

    Yields:
        Optional[int]: The id of the reserved CUDA device. ``None`` if no
        device satisfies ``required_mem_mb``.
    """

    os.makedirs(lock_dir, exist_ok=True)
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    for dev_id in range(torch.cuda.device_count()):
        lock_path = os.path.join(lock_dir, f"gpu{dev_id}.lock")
        try:
            # try to exclusively create the lock file
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            # another process holds the lock
            continue

        try:
            with torch.cuda.device(dev_id):
                free_bytes, _ = torch.cuda.mem_get_info()
            free_mb = free_bytes // (1024 * 1024)
            if free_mb < required_mem_mb:
                # not enough memory, release lock and continue
                os.close(fd)
                os.remove(lock_path)
                continue

            # Reserve this device by setting the environment variable so that
            # downstream libraries only see the selected GPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_id)
            try:
                yield dev_id
            finally:
                os.close(fd)
                os.remove(lock_path)
                if original_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
            return
        except Exception:
            # in case something goes wrong ensure we clean up
            os.close(fd)
            os.remove(lock_path)
            raise

    # no device satisfied the requirement
    yield None