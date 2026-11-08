"""Results that depend only on the model, kept between runs.

Some steps give the same answer for a model whatever the prompt:
* the regression baseline: the 17 suite replies and their teacher-forced log-probabilities;
* the known-direction library: refusal, language and repetition directions from wsr's fixed prompts;
* the statistics of a weight matrix (SVD, kurtosis).

They are stored under `~/.cache/wsr/models/<key>/` (or `WSR_CACHE_DIR`). The key holds:
* the model's files: names, sizes, modification times, and the config / tokenizer files' bytes;
* the weights as they are in memory: a full-precision, position-weighted checksum of every tensor,
  so a model patched in memory never matches the cache of the file it came from;
* the precision, the device and its name, and the torch / transformers / wsr versions (numerics differ
  across them);
* whether batching is on.

Each entry also hashes everything else it depends on (prompts, template settings, token cap). A
cached result is exactly the result the same computation produced before. When any part of the key
differs, the result is computed again. `WSR_CACHE=off` turns caching off.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch

FORMAT = 1


def enabled() -> bool:
    return os.environ.get("WSR_CACHE", "on").strip().lower() not in ("off", "0", "no", "false")


def root() -> Path:
    return Path(os.environ.get("WSR_CACHE_DIR") or Path.home() / ".cache" / "wsr") / "models"


def _weights_checksum(lm: Any) -> str:
    """Every tensor's float64 sum, sum of squares and position-weighted sum: any edited entry changes it."""
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in lm.model.named_parameters():
            t = p.detach().reshape(-1)
            s = s2 = sw = 0.0
            for i in range(0, t.numel(), 1 << 22):
                c = t[i:i + (1 << 22)].double()
                w = torch.arange(i + 1, i + 1 + c.numel(), device=c.device, dtype=torch.float64)
                s += float(c.sum())
                s2 += float((c * c).sum())
                sw += float((c * w).sum())
            h.update(f"{name}|{tuple(p.shape)}|{p.dtype}|{s!r}|{s2!r}|{sw!r}\n".encode())
    return h.hexdigest()


def model_key(lm: Any) -> Optional[str]:
    """None when the model has no directory (built in memory) or caching is off."""
    if not enabled() or getattr(lm, "path", None) is None:
        return None
    memo = getattr(lm, "_cache_key", None)
    if memo is not None and memo[0] == getattr(lm, "_wver", 0):
        return memo[1]
    import transformers

    from . import __version__
    from .deep import batching as BT

    h = hashlib.sha256()
    d = Path(lm.path)
    for f in sorted(d.iterdir()):
        if f.is_file():
            st = f.stat()
            h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
            if f.suffix == ".json" and st.st_size < 50_000_000:
                h.update(f.read_bytes())
    dev = lm.device
    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    p = next(lm.model.parameters())
    h.update(f"{FORMAT}|{p.dtype}|{dev.type}|{dev_name}|{torch.__version__}|{transformers.__version__}|"
             f"{__version__}|batch={BT.enabled()}\n".encode())
    h.update(_weights_checksum(lm).encode())
    key = f"{d.name}-{h.hexdigest()[:20]}"
    lm._cache_key = (getattr(lm, "_wver", 0), key)
    return key


def _entry(lm: Any, kind: str, depends: Any) -> Optional[Path]:
    key = model_key(lm)
    if key is None:
        return None
    dh = hashlib.sha256(json.dumps(depends, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return root() / key / f"{kind}-{dh}"


def load(lm: Any, kind: str, depends: Any) -> Optional[tuple[dict, dict[str, torch.Tensor]]]:
    """(meta, tensors) of a stored result, or None."""
    e = _entry(lm, kind, depends)
    if e is None:
        return None
    meta_p, ten_p = e.with_suffix(".json"), e.with_suffix(".safetensors")
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        tensors: dict[str, torch.Tensor] = {}
        if ten_p.exists():
            from safetensors.torch import load as st_load
            tensors = st_load(ten_p.read_bytes())
    except (OSError, ValueError):
        return None
    if meta.get("format") != FORMAT:
        return None
    return meta, tensors


def save(lm: Any, kind: str, depends: Any, meta: dict, tensors: Optional[dict[str, torch.Tensor]] = None) -> None:
    e = _entry(lm, kind, depends)
    if e is None:
        return
    e.parent.mkdir(parents=True, exist_ok=True)
    if tensors:
        from safetensors.torch import save as st_save
        tmp = e.with_suffix(".safetensors.tmp")
        tmp.write_bytes(st_save({k: v.detach().cpu().contiguous() for k, v in tensors.items()}))
        tmp.replace(e.with_suffix(".safetensors"))
    tmp = e.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({**meta, "format": FORMAT}, default=str, ensure_ascii=False), encoding="utf-8")
    tmp.replace(e.with_suffix(".json"))
