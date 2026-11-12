"""The deep diagnosis, kept across runs (wsr/cache.py).

Same weights (the model key covers the files, a weight checksum, dtype, device and library versions), same prompt,
kind, contrast prompts, token limit and template give the same diagnosis: every step is greedy and seeded. A later
run on the same goal (a rerun, another agent, a resumed program) gets the stored result instead of ~30 minutes of
work on a small GPU. Runs that read a GGUF or a second model are not stored.

Loading is safe: the result is written with torch.save, and read back with torch.load(weights_only=True), which
only builds the classes allowed for that entry. The allowed list is stored next to it and may only name wsr's
own classes and a few standard types; anything else in the result means it is not stored at all.
"""
from __future__ import annotations

import dataclasses
import enum
import importlib
import io
import time
from typing import Any, Optional

import torch

from .. import cache

KIND = "deep-diagnosis"
STD = {"collections.OrderedDict", "collections.defaultdict", "builtins.set", "builtins.frozenset", "builtins.slice",
       "builtins.complex"}


def _name(t: type) -> str:
    return f"{t.__module__}.{t.__qualname__}"


def _classes(obj: Any, seen: set[int], out: set[str]) -> None:
    if isinstance(obj, enum.Enum):                 # before the str / int check: str- and int-based enums
        out.add(_name(type(obj)))
        return
    if id(obj) in seen or obj is None or isinstance(obj, (str, bytes, int, float, bool, torch.Tensor, torch.dtype,
                                                         torch.device, torch.Size)):
        return
    seen.add(id(obj))
    t = type(obj)
    if isinstance(obj, (list, tuple)):
        if t not in (list, tuple):
            out.add(_name(t))
        for x in obj:
            _classes(x, seen, out)
        return
    if isinstance(obj, dict):
        if t is not dict:
            out.add(_name(t))
        for k, v in obj.items():
            _classes(k, seen, out)
            _classes(v, seen, out)
        return
    if isinstance(obj, (set, frozenset)):
        out.add(_name(t))
        for x in obj:
            _classes(x, seen, out)
        return
    out.add(_name(t))
    if isinstance(obj, enum.Enum):
        return
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            _classes(getattr(obj, f.name), seen, out)
        return
    d = getattr(obj, "__dict__", None)
    if d is not None:
        _classes(d, seen, out)


def _allowed(names: set[str]) -> bool:
    return all(n.startswith("wsr.") or n in STD for n in names)


def _resolve(names: list[str]) -> list:
    out = []
    for n in names:
        if not (n.startswith("wsr.") or n in STD):
            raise ValueError(n)
        mod, _, qual = n.rpartition(".")
        if mod not in ("builtins", "collections") and not mod.startswith("wsr."):
            raise ValueError(n)
        obj = importlib.import_module(mod)
        for part in qual.split("."):
            obj = getattr(obj, part)
        out.append(obj)
    return out


def depends(messages: list[dict], kw: dict) -> Optional[dict]:
    """The inputs the diagnosis depends on, or None when this run is not stored (a GGUF or a second model)."""
    if kw.get("base") is not None or kw.get("gguf") or kw.get("reference_dir"):
        return None
    keys = ("kind", "failing", "passing", "expect_lang", "max_new", "template_kwargs", "known", "probes")
    return {"messages": messages, **{k: kw.get(k) for k in keys}}


def load(lm: Any, dep: dict) -> Optional[Any]:
    hit = cache.load(lm, KIND, dep)
    if hit is None:
        return None
    meta, _ = hit
    blob = cache.root() / (cache.model_key(lm) or "") / f"{KIND}-{cache.depends_hash(dep)}.pt"
    try:
        allow = _resolve(meta["classes"])
        with torch.serialization.safe_globals(allow):
            res = torch.load(io.BytesIO(blob.read_bytes()), weights_only=True)
    except (OSError, ValueError, AttributeError, ImportError, RuntimeError, KeyError, TypeError,
            __import__("pickle").UnpicklingError):
        return None
    res.lm = lm
    res.report.context["from_cache"] = {"stored": meta.get("stored"), "seconds_then": meta.get("seconds"),
                                        "entry": blob.name}
    return res


def save(lm: Any, dep: dict, res: Any, seconds: float, say=None) -> None:
    key = cache.model_key(lm)
    if key is None:
        return
    keep_lm, res.lm = res.lm, None
    try:
        names: set[str] = set()
        _classes(res, set(), names)
        if not _allowed(names):
            if say:
                say("  diagnosis not cached: it holds " + ", ".join(sorted(n for n in names
                                                                         if not (n.startswith("wsr.") or n in STD))))
            return
        buf = io.BytesIO()
        torch.save(res, buf)
    finally:
        res.lm = keep_lm
    blob = cache.root() / key / f"{KIND}-{cache.depends_hash(dep)}.pt"
    blob.parent.mkdir(parents=True, exist_ok=True)
    tmp = blob.with_suffix(".pt.tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(blob)
    cache.save(lm, KIND, dep, {"classes": sorted(names), "seconds": round(seconds, 1),
                               "stored": time.strftime("%Y-%m-%dT%H:%M:%S")})
