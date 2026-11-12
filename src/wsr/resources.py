"""How much more memory this machine can take before it hurts.

On Windows, every allocation (GPU memory included: WDDM backs video memory with system commit) counts against
the commit limit (RAM + page file). Past the limit, a system-managed page file grows onto the system drive. On a
nearly full disk that growth can fill the drive and make other programs fail. Measured on a 16 GB machine: 21 GB
committed at idle by other software; loading a 1B model in PyTorch added 6 GB.

`headroom()` is the extra memory a step can commit before the page file would have to eat into the last
`DISK_FLOOR` of the system drive. Optional heavy steps (starting llama-server, converting to GGUF) check it and
are skipped, with the reason, when they do not fit. Elsewhere it is free RAM plus swap (Linux / macOS do not
grow swap onto the disk this way).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

DISK_FLOOR = 3 * 2**30


def _windows_commit() -> Optional[tuple[int, int]]:
    """(committed bytes, commit limit) from GetPerformanceInfo, or None."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class PI(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                        ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                        ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                        ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                        ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                        ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD),
                        ("ThreadCount", wintypes.DWORD)]
        p = PI()
        p.cb = ctypes.sizeof(PI)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb):
            return None
        return p.CommitTotal * p.PageSize, p.CommitLimit * p.PageSize
    except (OSError, AttributeError, ValueError):
        return None


def system_drive() -> Path:
    return Path(os.environ.get("SystemDrive", "") + "\\") if sys.platform == "win32" else Path("/")


def headroom() -> Optional[int]:
    """Bytes a step may still commit safely; None when it cannot be measured."""
    try:
        disk = shutil.disk_usage(system_drive()).free
    except OSError:
        disk = None
    wc = _windows_commit()
    if wc is not None:
        used, limit = wc
        spare_disk = max(0, (disk or 0) - DISK_FLOOR)
        return max(0, limit - used) + spare_disk
    try:
        import psutil  # type: ignore
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        return int(vm.available + sw.free)
    except ImportError:
        return None


def check(step: str, need: int) -> Optional[str]:
    """None when `need` bytes fit; else the reason in words."""
    h = headroom()
    if h is None or h >= need:
        return None
    return (f"{step} would need about {need / 2**30:.1f} GB more memory, and only {h / 2**30:.1f} GB can be committed "
            f"before the page file would fill the system drive (keep {DISK_FLOOR >> 30} GB free)")
