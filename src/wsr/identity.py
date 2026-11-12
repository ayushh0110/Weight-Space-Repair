"""Exact identity of a model's weights, so a patch is never applied to weights it was not made for.

A patch is a difference from exact weights (a gate or set_columns stores new minus old): on other weights it
writes something else, and the checks it passed were run on different weights. So a patch records the identity of
the model it was made on, and every command that applies it (apply, evaluate --patch, repair --check, export)
compares it first and refuses on any difference unless the user explicitly allows it.

Three parts, as in the patch's `identity.json`:

* the **tensor manifest**: a SHA-256 of every tensor (its name, shape, dtype and contiguous bytes), in float32,
  bfloat16 and float16. The three are the supported precisions a model is loaded in; storing all of them for every
  tensor lets a model loaded in another precision match exactly (a bf16 file loaded as float32 is exact; loaded as
  float16 it is the same conversion every loader makes). This decides; there is no tolerance.
* the **model metadata**: architecture fields of config.json, and hashes of the config, tokenizer, chat template
  and generation files. A different architecture refuses; a different tokenizer or template is reported.
* the **file hashes**: SHA-256 of every file of the checkpoint, recorded for provenance only, so re-saving or
  re-sharding an identical model does not make it a different one.

The manifest is computed from the safetensors files (the exact stored values), cached under the wsr cache keyed by
the files' names, sizes and modification times, so it is computed once per model. A model with no safetensors
files (built in memory, or a .bin checkpoint wsr will not unpickle) gets it from the weights in memory, for the
precisions those weights determine exactly.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

FORMAT = 1
PRECISIONS = ("float32", "bfloat16", "float16")
_CHUNK = 1 << 24
META_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "tokenizer.model", "vocab.json",
              "merges.txt")
ARCH_KEYS = ("model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "intermediate_size", "vocab_size", "tie_word_embeddings")


def _dt(name: str):
    import torch
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _dtname(t) -> str:
    return str(t.dtype).replace("torch.", "")


def _buffer(c):
    """A contiguous CPU tensor's bytes, as a numpy array (bfloat16 has no numpy type: viewed as int16)."""
    import torch
    if c.dtype in (torch.bfloat16, torch.float16):
        return c.view(torch.int16).numpy()
    if c.dtype == torch.float32:
        return c.view(torch.int32).numpy()
    return c.numpy()


def tensor_hash(name: str, t, dtype: Optional[str] = None) -> str:
    """SHA-256 of a tensor's name, shape, dtype and bytes, after converting it to `dtype` (None: as it is)."""
    dname = dtype or _dtname(t)
    h = hashlib.sha256(f"{name}|{list(t.shape)}|{dname}\n".encode())
    flat = t.detach().reshape(-1)
    for i in range(0, flat.numel(), _CHUNK):
        c = flat[i:i + _CHUNK]
        if dtype is not None and _dtname(c) != dtype:
            c = c.to(_dt(dtype))
        h.update(memoryview(_buffer(c.contiguous().cpu())))
    return h.hexdigest()


def _tensor_entry(name: str, t, exact: tuple[str, ...]) -> dict[str, Any]:
    if not t.is_floating_point():
        return {"shape": list(t.shape), "stored": _dtname(t), "hashes": {_dtname(t): tensor_hash(name, t)}}
    return {"shape": list(t.shape), "stored": _dtname(t), "hashes": {d: tensor_hash(name, t, d) for d in exact}}


def _digests(tensors: dict[str, dict]) -> dict[str, str]:
    """One hash per precision over every tensor's hash: the logical model, whatever its files look like."""
    out = {}
    for d in PRECISIONS:
        h = hashlib.sha256()
        ok = True
        for name in sorted(tensors):
            e = tensors[name]
            v = e["hashes"].get(d) if e["stored"] in PRECISIONS else next(iter(e["hashes"].values()))
            if v is None:                         # a float tensor with no exact hash in this precision
                ok = False
                break
            h.update(f"{name}={v}\n".encode())
        if ok:
            out[d] = h.hexdigest()
    return out


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def metadata(model_dir: Optional[Path]) -> dict[str, Any]:
    if model_dir is None:
        return {}
    out: dict[str, Any] = {"files": {}}
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
        out["architecture"] = {k: cfg.get(k, tc.get(k)) for k in ARCH_KEYS if k in cfg or k in tc}
    except (OSError, ValueError):
        out["architecture"] = {}
    for n in META_FILES:
        p = model_dir / n
        if p.is_file():
            out["files"][n] = _file_sha(p)
    m = re.search(r"snapshots[\\/]([0-9a-f]{40})", str(model_dir.resolve()))
    if m:
        out["revision"] = m.group(1)
    return out


def _stat_key(model_dir: Path) -> str:
    h = hashlib.sha256(str(FORMAT).encode())
    for f in sorted(model_dir.iterdir()):
        if f.is_file():
            st = f.stat()
            h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return f"{model_dir.name}-{h.hexdigest()[:20]}"


def from_files(model_dir: Path, *, cache: bool = True) -> Optional[dict[str, Any]]:
    """The manifest of a checkpoint's safetensors files (every tensor, every precision), or None without them."""
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return None
    from . import cache as C

    key = _stat_key(model_dir)
    cpath = C.base() / "identity" / f"{key}.json"
    if cache and C.enabled() and cpath.exists():
        try:
            return json.loads(cpath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    from safetensors import safe_open

    tensors: dict[str, dict] = {}
    for f in files:
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                tensors[k] = _tensor_entry(k, h.get_tensor(k), PRECISIONS)
    man = {"format": FORMAT, "source": "files", "tensors": tensors, "digests": _digests(tensors),
           "metadata": metadata(model_dir),
           "files": {f.name: {"bytes": f.stat().st_size, "sha256": _file_sha(f)} for f in sorted(model_dir.iterdir())
                     if f.is_file()}}
    if cache and C.enabled():
        cpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(man), encoding="utf-8")
        tmp.replace(cpath)
    return man


def _params(lm: Any) -> dict[str, Any]:
    """Every parameter once (a tied output layer is the input embedding's tensor, listed under its first name)."""
    return dict(lm.model.named_parameters())


def from_model(lm: Any) -> dict[str, Any]:
    """The manifest of the weights in memory, for the precisions they determine exactly: float32 weights (loaded
    from a 16-bit or 32-bit file) determine all three; 16-bit weights determine only their own precision."""
    exact_all = all(_dtname(p) == "float32" for p in lm.model.parameters())
    tensors = {}
    for name, p in _params(lm).items():
        exact = PRECISIONS if exact_all else (_dtname(p),)
        tensors[name] = _tensor_entry(name, p, exact)
    return {"format": FORMAT, "source": f"memory ({'float32' if exact_all else 'loaded precision only'})",
            "tensors": tensors, "digests": _digests(tensors), "metadata": metadata(getattr(lm, "path", None)),
            "files": {}}


def manifest(lm: Any) -> dict[str, Any]:
    """The unedited model's manifest: from its files when they hold every parameter, else from memory. Kept on the
    model object for the run (the files do not change under it)."""
    memo = getattr(lm, "_identity", None)
    if memo is not None:
        return memo
    man = None
    path = getattr(lm, "path", None)
    if path is not None:
        man = from_files(Path(path))
        if man is not None and any(n not in man["tensors"] for n in _params(lm)):
            man = None                              # names differ from the files: fall back to memory
    if man is None:
        man = from_model(lm)
    lm._identity = man
    return man


def check(lm: Any, man: dict[str, Any], *, patched: Optional[dict[str, dict]] = None) -> dict[str, Any]:
    """Compare the weights in memory with a manifest, tensor by tensor, in the precision each is loaded in.
    `patched`: hashes that replace the manifest's for the tensors a patch edited (to check a patched model).
    Returns {verdict: same | different, mismatched, missing, unverifiable, metadata_differs}."""
    tensors = dict(man.get("tensors") or {})
    for n, v in (patched or {}).items():
        tensors[n] = v
    mismatched, unverifiable = [], []
    params = _params(lm)
    # Float32 weights loaded from a 16-bit file hold exactly the 16-bit values: compare them in that precision
    # (the conversion back is exact), so loading in more precision than the file never counts as a difference.
    ck = str(getattr(lm, "checkpoint_dtype", "") or "").replace("torch.", "")
    for name, p in params.items():
        exp = tensors.get(name)
        if exp is None:
            continue
        d = _dtname(p) if p.is_floating_point() else None
        if d == "float32" and ck in ("bfloat16", "float16"):
            d = ck
        want = exp["hashes"].get(d) if d else next(iter(exp["hashes"].values()))
        if want is None:
            unverifiable.append(f"{name} ({d})")
            continue
        if tensor_hash(name, p, d) != want:
            mismatched.append(name)
    # A parameter the manifest does not know is a difference. Tensors in the files that are not parameters
    # (buffers some checkpoints carry, a tied output layer stored twice) take no part in the patch.
    missing = sorted(n for n in params if n not in tensors)
    extra: list[str] = []
    meta_now = metadata(getattr(lm, "path", None))
    arch_then = (man.get("metadata") or {}).get("architecture") or {}
    arch_now = meta_now.get("architecture") or {}
    arch_diff = sorted(k for k in arch_then if k in arch_now and arch_then[k] != arch_now[k])
    files_then = (man.get("metadata") or {}).get("files") or {}
    meta_diff = sorted(n for n in files_then if n in (meta_now.get("files") or {}) and files_then[n] !=
                       meta_now["files"][n])
    bad = mismatched or missing or extra or unverifiable or arch_diff
    out = {"verdict": "different" if bad else "same", "mismatched": mismatched[:20], "n_mismatched": len(mismatched),
           "missing": (missing + extra)[:20], "unverifiable": unverifiable[:20], "architecture_differs": arch_diff,
           "metadata_differs": meta_diff}
    if bad:
        what = (f"{len(mismatched)} tensor(s) differ (first: {mismatched[0]})" if mismatched else
                f"tensor names differ ({(missing + extra)[0]})" if missing or extra else
                f"architecture differs ({', '.join(arch_diff)})" if arch_diff else
                f"loaded in a precision the patch has no exact hash for ({unverifiable[0]})")
        out["why"] = what
    return out


def patched_hashes(man: dict[str, Any], lm: Any, patch: Any) -> dict[str, dict]:
    """Hashes of the edited tensors as `wsr patch apply` writes them: the stored values plus the patch, rounded once
    to the stored dtype, then in every precision. A patched model can then be checked exactly (for revert)."""
    import torch

    from .surgery import edits as E

    out = {}
    path = getattr(lm, "path", None)
    by_name = {}
    for d in patch.deltas:
        by_name.setdefault(d.name, []).append(d)
    stored_vals = _stored_values(Path(path), list(by_name)) if path is not None and man.get("source") == "files" else {}
    for name, ds in by_name.items():
        base = stored_vals.get(name)
        if base is None:
            base = E._param(lm, name).detach().float().cpu()
        stored = (man["tensors"].get(name) or {}).get("stored") or _dtname(base)
        acc = base.float()
        for d in ds:
            delta = d.dense().float()
            acc = acc + (delta[:, 0] if acc.dim() == 1 else delta)
        new = acc.to(_dt(stored) if stored in PRECISIONS else torch.float32)
        out[name] = _tensor_entry(name, new, PRECISIONS)
    return out


def _stored_values(model_dir: Path, names: list[str]) -> dict[str, Any]:
    from safetensors import safe_open

    out = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if k in names:
                    out[k] = h.get_tensor(k)
    return out


def for_patch(lm: Any, patch: Any) -> dict[str, Any]:
    """The identity a patch is saved with: the unedited model's manifest and the edited tensors after patching."""
    man = manifest(lm)
    return {"format": FORMAT, "model": {"name": getattr(lm, "name", ""), "source": man["source"],
                                        "digests": man["digests"], "metadata": man.get("metadata") or {},
                                        "files": man.get("files") or {}},
            "tensors": man["tensors"], "patched": patched_hashes(man, lm, patch)}


def check_files(model_dir: Path, ident: dict[str, Any]) -> dict[str, Any]:
    """Compare a checkpoint's files (not a loaded model) with a patch's identity, in each tensor's stored dtype:
    for the GGUF export, which patches the files directly."""
    man = from_files(model_dir)
    if man is None:
        return {"verdict": "different", "why": f"{model_dir} has no safetensors files to check"}
    want = ident.get("tensors") or {}
    mism = []
    for name, v in man["tensors"].items():
        w = want.get(name)
        if w is None:
            mism.append(name)
            continue
        d = v["stored"]
        if w["hashes"].get(d) != v["hashes"].get(d):
            mism.append(name)
    missing = sorted(n for n in want if n not in man["tensors"])
    bad = mism + missing
    out = {"verdict": "different" if bad else "same", "mismatched": bad[:20], "n_mismatched": len(bad)}
    if bad:
        out["why"] = f"{len(bad)} tensor(s) of {model_dir.name} differ from the model the patch was made on (first: {bad[0]})"
    return out
