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
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:                  # torch is imported only where a model is at hand: `wsr cache` works without it
    import torch

FORMAT = 1


def enabled() -> bool:
    return os.environ.get("WSR_CACHE", "on").strip().lower() not in ("off", "0", "no", "false")


def base() -> Path:
    """The one folder everything wsr keeps lives under (`WSR_CACHE_DIR`, else ~/.cache/wsr)."""
    return Path(os.environ.get("WSR_CACHE_DIR") or Path.home() / ".cache" / "wsr")


def root() -> Path:
    return base() / "models"


def _weights_checksum(lm: Any) -> str:
    """Every tensor's float64 sum, sum of squares and position-weighted sum: any edited entry changes it."""
    import torch

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
    import torch
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
    from .output import DETECTOR_VERSION

    h.update(f"{FORMAT}|{p.dtype}|{dev.type}|{dev_name}|{torch.__version__}|{transformers.__version__}|"
             f"{__version__}|batch={BT.enabled()}|detectors={DETECTOR_VERSION}\n".encode())
    h.update(_weights_checksum(lm).encode())
    key = f"{d.name}-{h.hexdigest()[:20]}"
    lm._cache_key = (getattr(lm, "_wver", 0), key)
    return key


def depends_hash(depends: Any) -> str:
    return hashlib.sha256(json.dumps(depends, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _entry(lm: Any, kind: str, depends: Any) -> Optional[Path]:
    key = model_key(lm)
    if key is None:
        return None
    return root() / key / f"{kind}-{depends_hash(depends)}"


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


# -- showing and clearing (wsr cache) ----------------------------------------------------------------------

GROUPS = {"models": "per-model results: regression baselines, original replies, diagnoses, direction libraries, "
                    "calibrated probes",
          "gguf": "GGUF conversions of original models (for the llama-server judge)",
          "datasets": "converted public datasets (wsr data fetch)",
          "converter": "the downloaded llama.cpp converter source",
          "identity": "exact tensor hashes of model checkpoints (computed again when needed)",
          "legacy": "probe caches written by older wsr versions (no longer read)"}

_MODEL_ENTRY = re.compile(r".+-[0-9a-f]{20}$")                    # <model folder name>-<key>, as model_key writes it
_GGUF_ENTRY = re.compile(r".+-[0-9a-f]{12}\.[A-Za-z0-9_]+\.gguf$")  # <model>-<key>.<quant>.gguf (program/judge.py)


def _group_paths(group: str) -> list[Path]:
    """The entries of a group that wsr wrote, recognised by name. `wsr cache clear` deletes these and nothing
    else, so a WSR_CACHE_DIR that also holds the user's own `models/` or `datasets/` folders loses nothing."""
    b = base()
    if group == "models":
        d = b / "models"
        return sorted(p for p in d.iterdir() if p.is_dir() and _MODEL_ENTRY.match(p.name)) if d.is_dir() else []
    if group == "gguf":
        d = b / "gguf"
        return sorted(p for p in d.iterdir() if p.is_file() and _GGUF_ENTRY.match(p.name)) if d.is_dir() else []
    if group == "datasets":
        from .program.datasets import SOURCES

        d = b / "datasets"
        return sorted(p for p in (d / f"{n}.jsonl" for n in SOURCES) if p.is_file())
    if group == "converter":
        return sorted(p for p in b.glob("llama.cpp-*") if (p / "convert_hf_to_gguf.py").is_file())
    if group == "identity":
        d = b / "identity"
        if not d.is_dir():
            return []
        return sorted(p for p in d.iterdir() if p.is_file() and re.match(r".+-[0-9a-f]{20}\.json$", p.name))
    if group == "legacy":
        return [p for p in (b / "langprobe", b / "copyheads") if p.is_dir()]
    return []


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def usage() -> dict[str, Any]:
    """Size of each group, and of each model's entry under models/."""
    out: dict[str, Any] = {"base": str(base()), "groups": {}, "models": []}
    for g in GROUPS:
        paths = _group_paths(g)
        out["groups"][g] = {"bytes": sum(_size(p) for p in paths), "paths": [str(p) for p in paths]}
    for d in _group_paths("models"):
        kinds: dict[str, int] = {}
        for f in d.iterdir():
            if f.suffix == ".json":
                k = f.stem.rsplit("-", 1)[0]
                kinds[k] = kinds.get(k, 0) + 1
        out["models"].append({"key": d.name, "bytes": _size(d), "entries": kinds})
    return out


def targets(groups: list[str], model: Optional[str] = None) -> list[Path]:
    """What `wsr cache clear` would delete."""
    if model:
        return [d for d in _group_paths("models") if d.name.startswith(model)]
    out: list[Path] = []
    for g in (list(GROUPS) if "all" in groups else groups):
        if g not in GROUPS:
            raise ValueError(f"unknown cache group {g!r}: {', '.join(GROUPS)} or all")
        out += _group_paths(g)
    return out


def remove(paths: list[Path]) -> int:
    """Delete them (only paths under the cache base); returns the bytes freed."""
    import shutil

    b = base().resolve()
    freed = 0
    for p in paths:
        rp = p.resolve()
        if b not in rp.parents:
            raise ValueError(f"{p} is not inside the wsr cache ({b})")
        freed += _size(rp)
        shutil.rmtree(rp) if rp.is_dir() else rp.unlink()
    return freed
