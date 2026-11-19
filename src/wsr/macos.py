"""Facts about the Mac wsr runs on, with no dependencies: whether it is Apple Silicon, the macOS version, the chip,
and how much of the unified memory the GPU (Metal) may use.

On Apple Silicon the CPU and the GPU share one pool of memory, but Metal may only wire part of it for the GPU: the
recommended working set, about two thirds of the RAM on machines up to 36 GB and three quarters above (macOS's
default; `sudo sysctl iogpu.wired_limit_mb=N` changes it). A model larger than that does not run on the GPU even
when the RAM could hold it, so memory checks on a Mac compare against this limit, not the whole RAM.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from functools import lru_cache
from typing import Optional

DEFAULT_SHARE_SMALL = 2 / 3      # of the RAM, up to SMALL_RAM (macOS's default GPU working set, approximately)
DEFAULT_SHARE_LARGE = 3 / 4      # above it
SMALL_RAM = 36 * 2**30


def is_mac() -> bool:
    return sys.platform == "darwin"


def apple_silicon() -> bool:
    return is_mac() and platform.machine() == "arm64"


@lru_cache(maxsize=32)
def sysctl(name: str) -> Optional[str]:
    """`sysctl -n NAME`, or None (not a Mac, the key does not exist, or sysctl failed)."""
    if not is_mac():
        return None
    try:
        r = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    out = (r.stdout or "").strip()
    return out if r.returncode == 0 and out else None


def macos_version() -> Optional[tuple[int, ...]]:
    """(14, 5) for macOS 14.5; None elsewhere."""
    if not is_mac():
        return None
    v = platform.mac_ver()[0]
    try:
        return tuple(int(x) for x in v.split(".") if x) or None
    except ValueError:
        return None


def chip() -> str:
    """"Apple M2 Pro", or the best name macOS gives."""
    return sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine() or "unknown"


def ram_bytes() -> Optional[int]:
    v = sysctl("hw.memsize")
    try:
        return int(v) if v else None
    except ValueError:
        return None


def gpu_limit_bytes(ram: Optional[int] = None) -> Optional[int]:
    """The memory Metal lets the GPU use: `iogpu.wired_limit_mb` when it is set (0 means the default), else
    macOS's default share of the RAM. None when this is not Apple Silicon or the RAM cannot be read. An estimate
    for the default: PyTorch's torch.mps.recommended_max_memory() reports the exact value where it exists."""
    if ram is None:
        if not apple_silicon():
            return None
        ram = ram_bytes()
    if not ram:
        return None
    v = sysctl("iogpu.wired_limit_mb") if apple_silicon() else None
    try:
        mb = int(v) if v else 0
    except ValueError:
        mb = 0
    if mb > 0:
        return mb * 2**20
    return int(ram * (DEFAULT_SHARE_SMALL if ram <= SMALL_RAM else DEFAULT_SHARE_LARGE))


def describe() -> str:
    """"Apple M2 Pro, macOS 14.5" (for cache keys and reports)."""
    v = macos_version()
    return f"{chip()}, macOS {'.'.join(map(str, v)) if v else '?'}"
