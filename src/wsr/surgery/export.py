"""`wsr patch export --gguf`: ship a verified patch into the file people serve.

1. The patch is applied to the full-precision model and saved in its checkpoint precision, as
   `wsr patch apply` does.
2. llama.cpp's own converter (`convert_hf_to_gguf.py` with its `gguf-py`) writes an F16 / BF16 GGUF.
   wsr uses a copy the user points at (`--llama-cpp DIR` or `WSR_LLAMA_CPP`), or one it fetched before.
   Otherwise it asks before downloading the llama.cpp source at the tag that matches the user's
   llama-server build (or a pinned tag) and keeps only the converter files.
3. `llama-quantize`, found next to llama-server, makes the requested type (Q8_0, Q4_K_M, ...).

Every step that downloads, converts or writes gigabytes asks first (consent kinds `download` and
`convert`). A "no" stops with what was needed; nothing is half-written. The served file is then
checked with `wsr evaluate --backend llama-server --gguf <original> --patched-gguf <patched>`, which shows how
much of the gain survives quantisation.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from ..consent import Consent, NeedsConsent

PINNED_TAG = "b10244"
FLOAT_TYPES = {"F16": "f16", "BF16": "bf16", "F32": "f32", "Q8_0": "q8_0"}   # the converter writes these directly


def cache_dir() -> Path:
    from ..cache import base

    return base()


def _has_converter(d: Path) -> bool:
    return (d / "convert_hf_to_gguf.py").exists() and (d / "gguf-py" / "gguf").is_dir()


def server_tag(server: Optional[Path]) -> Optional[str]:
    """The llama.cpp tag of a llama-server binary (its --version build number)."""
    if server is None:
        return None
    try:
        out = subprocess.run([str(server), "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                             cwd=str(server.parent))
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"version:\s*(\d+)", (out.stdout or "") + (out.stderr or ""))
    return f"b{m.group(1)}" if m else None


def find_converter(explicit: Optional[str] = None, tag: Optional[str] = None) -> Optional[Path]:
    for c in (explicit, os.environ.get("WSR_LLAMA_CPP")):
        if c and _has_converter(Path(c).expanduser()):
            return Path(c).expanduser()
    for t in ([tag] if tag else []) + [PINNED_TAG]:
        d = cache_dir() / f"llama.cpp-{t}"
        if _has_converter(d):
            return d
    return None


def fetch_converter(consent: Consent, tag: str) -> Path:
    """Download llama.cpp's source at `tag` (after asking) and keep the converter files only."""
    url = f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/{tag}"
    dest = cache_dir() / f"llama.cpp-{tag}"
    consent.require("download", f"download llama.cpp {tag} source from github.com/ggml-org/llama.cpp (about 30 MB; "
                                "only the GGUF converter files are kept)", what="llama.cpp's GGUF converter",
                    why="it writes the GGUF the patch is shipped in",
                    detail=f"kept under {dest}; or point --llama-cpp at a llama.cpp checkout you already have")
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tgz = Path(td) / "src.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as r, open(tgz, "wb") as f:
            shutil.copyfileobj(r, f)
        with tarfile.open(tgz) as tf:
            keep = []
            for m in tf.getmembers():
                parts = m.name.split("/", 1)
                if len(parts) < 2:
                    continue
                rel = parts[1]
                if (rel == "convert_hf_to_gguf.py" or rel.startswith("gguf-py/") or rel.startswith("conversion/")) \
                        and ".." not in rel and not rel.startswith("/") and (m.isfile() or m.isdir()):
                    m.name = rel
                    keep.append(m)
            try:                                                    # members filtered above; "data" refuses
                tf.extractall(dest, members=keep, filter="data")    # links, devices and paths outside dest too
            except TypeError:                                       # a Python without extraction filters
                tf.extractall(dest, members=keep)                   # noqa: S202
    if not _has_converter(dest):
        raise ValueError(f"the llama.cpp {tag} source did not contain the converter where expected")
    return dest


def _quantize_bin(server: Optional[Path]) -> Optional[Path]:
    cands = []
    if server is not None:
        cands += [server.parent / "llama-quantize.exe", server.parent / "llama-quantize"]
    w = shutil.which("llama-quantize") or shutil.which("llama-quantize.exe")
    if w:
        cands.append(Path(w))
    return next((c for c in cands if c.exists()), None)


def _model_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.glob("*.safetensors")) or sum(f.stat().st_size for f in d.glob("*.bin"))


def verify_source(model_dir: Path, patch_dir: Path, *, allow_base_mismatch: bool = False,
                  allow_unverified_legacy: bool = False, log=None) -> dict:
    """Before a patch is written into files: are these the files of the model it was made for? Exact, in each
    tensor's stored dtype (wsr/identity.py). Refuses (ValueError) unless the user allowed the difference."""
    from .. import identity as ID
    from . import edits as E

    patch = E.Patch.load(patch_dir)
    if not patch.identity:
        if not allow_unverified_legacy:
            raise ValueError("cannot verify the source checkpoint: the patch records no exact identity of its model "
                             "(--allow-unverified-legacy to export it anyway)")
        r = {"verdict": "unverified"}
    else:
        r = ID.check_files(model_dir, patch.identity)
        if r["verdict"] != "same" and not allow_base_mismatch:
            raise ValueError(f"{r['why']}: the patch was not made for these files, so the GGUF would not hold the edit "
                             "that was checked (--allow-base-mismatch to export it anyway)")
    if r["verdict"] != "same" and log:
        log(f"  WARNING: exporting a patch whose source could not be verified ({r['verdict']})")
    return r


def patch_files(model_dir: Path, patch_dir: Path, dst: Path) -> bool:
    """Write the patched checkpoint by editing the safetensors files directly: tensors the patch does not touch
    are copied bit for bit; a touched tensor becomes round(w + sum of its deltas), added in float32 and rounded
    once to the file's own type - the same arithmetic as applying the patch to a float32 model and saving it in
    the checkpoint's precision. Needs one shard in memory, not the model. False (nothing written) when the patch
    cannot be expressed on the files (it unties the embeddings, or names a tensor the files do not hold)."""
    import json as _json

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from . import edits as E

    patch = E.Patch.load(patch_dir)
    files = sorted(model_dir.glob("*.safetensors"))
    if patch.untie or not files:
        return False
    where: dict[str, Path] = {}
    meta: dict[Path, dict] = {}
    for f in files:
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                where[k] = f
            meta[f] = h.metadata() or {}
    if any(d.name not in where for d in patch.deltas):
        return False
    by_file: dict[Path, dict[str, list]] = {}
    for d in patch.deltas:
        by_file.setdefault(where[d.name], {}).setdefault(d.name, []).append(d)
    dst.mkdir(parents=True, exist_ok=True)
    for f in model_dir.iterdir():
        if f.is_file() and f not in by_file:
            shutil.copy2(f, dst / f.name)
    for f, names in by_file.items():
        t = load_file(str(f))
        for name, ds in names.items():
            w = t[name]
            acc = w.float()
            for d in ds:
                delta = d.dense().float()
                acc = acc + (delta[:, 0] if w.dim() == 1 else delta)
            t[name] = acc.to(w.dtype).contiguous()
        try:
            save_file(t, str(dst / f.name), metadata={**meta[f], "format": "pt"})
        except Exception as e:                          # noqa: BLE001 - safetensors raises its own error type
            msg = str(e)
            if "space" in msg.lower() or "StorageFull" in msg:
                raise OSError(f"the disk filled up while writing the patched checkpoint ({dst}); free some space "
                              "and run again") from e
            raise OSError(f"could not write the patched checkpoint: {msg[:200]}") from e
        del t
    (dst / "wsr_patch.json").write_text(_json.dumps({"patch": patch.label, "tensors": sorted(where[d.name].name
                                                    for d in patch.deltas)}), encoding="utf-8")
    return True


def _tool_error(r: subprocess.CompletedProcess) -> str:
    """The tool's own message, without progress bars (tqdm lines carry '%|' or start with a counter)."""
    text = (r.stderr or "").replace("\r", "\n") + "\n" + (r.stdout or "")
    lines = [ln for ln in text.splitlines()
             if ln.strip() and "%|" not in ln and not ln.lstrip().startswith(("Writing:", "Loading"))]
    code = r.returncode
    killed = code < 0 or code in (137, 3221225477, 3221226505)
    head = f"exit code {code}" + (" (killed: most likely out of memory)" if killed else "")
    return head + ("\n" + "\n".join(lines[-20:]) if lines else "")


def _direct_ok(model_dir: Path, patch_dir: Path) -> bool:
    try:
        import json as _json
        meta = _json.loads((patch_dir / "patch.json").read_text(encoding="utf-8"))
        if meta.get("untie"):
            return False
        from safetensors import safe_open
        keys = set()
        for f in model_dir.glob("*.safetensors"):
            with safe_open(str(f), "pt") as h:
                keys.update(h.keys())
        return bool(keys) and all(n in keys for n in meta.get("params", []))
    except (OSError, ValueError):
        return False


def torch_float32():
    import torch
    return torch.float32


def _ram_available() -> Optional[int]:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except ImportError:
        return None


def export(model_dir: Path, patch_dir: Optional[Path], out: Path, *, quant: str = "Q8_0", consent: Consent,
           llama_cpp: Optional[str] = None, server: Optional[Path] = None, log=print, lm=None,
           allow_base_mismatch: bool = False, allow_unverified_legacy: bool = False) -> dict:
    """Write `out` (a GGUF of the patched model, or of the model itself when patch_dir is None). `lm`: the model
    already in memory (unpatched), used to write the patched copy instead of loading a second one."""
    say = log or (lambda s: None)
    quant = quant.upper()
    source_check = (verify_source(model_dir, patch_dir, allow_base_mismatch=allow_base_mismatch,
                                  allow_unverified_legacy=allow_unverified_legacy, log=say)
                    if patch_dir is not None else None)
    size = _model_bytes(model_dir)
    # The in-memory model is reused only in float32: a 16-bit checkpoint round-trips exactly through float32,
    # but not through another 16-bit type (bf16 <-> fp16 loses small weights), and the candidate must be
    # exactly the original plus the patch.
    if lm is not None and next(lm.model.parameters()).dtype != torch_float32():
        lm = None
    from .. import resources

    why = resources.check("converting to GGUF", int(1.2 * size))
    if why:
        raise ValueError(why)
    ram = _ram_available()
    # The steps run one after another, so the peak is the largest of them, with a 20 % margin: llama.cpp's
    # converter (measured: 0.93x the checkpoint, Llama-3.2-1B), one shard when the files are patched directly,
    # or a float32 copy of the model (2x a 16-bit checkpoint) on the fallback path.
    own_copy = patch_dir is not None and lm is None and not _direct_ok(model_dir, patch_dir)
    need = 1.2 * size * (2.0 if own_copy else 1.0)
    if ram is not None and size and ram < need:
        raise ValueError(f"not enough free RAM to convert: about {need / 1e9:.1f} GB needed, {ram / 1e9:.1f} GB "
                         "available (close other programs, or convert separately with wsr patch export)")
    free = shutil.disk_usage(out.parent if out.parent.exists() else Path.cwd()).free
    # On disk at once: the patched checkpoint (one copy of the files; none for the unpatched model) and the
    # GGUF being written (at most the checkpoint's size for F16, about half for Q8_0), plus a 10 % margin.
    need = 1.1 * size * (2.0 if patch_dir is not None else 1.0)
    if size and free < need:
        raise ValueError(f"not enough disk space: about {need / 1e9:.1f} GB needed for the intermediate files, "
                         f"{free / 1e9:.1f} GB free")
    tag = server_tag(server) or PINNED_TAG
    conv = find_converter(llama_cpp, tag)
    if conv is None:
        conv = fetch_converter(consent, tag)
    qbin = _quantize_bin(server) if quant not in FLOAT_TYPES else None
    if quant not in FLOAT_TYPES and qbin is None:
        raise ValueError(f"{quant} needs llama-quantize (next to llama-server, or on PATH); the converter alone writes "
                         f"{', '.join(FLOAT_TYPES)}")
    what = f"the patched model ({patch_dir.name})" if patch_dir else "the model"
    consent.require("convert", f"convert {what} to a {quant} GGUF ({size / 1e9:.1f} GB of weights; runs llama.cpp's "
                               "converter locally)", what=f"converting {what} to GGUF",
                    detail=f"writes {out}; intermediate files are removed afterwards")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="wsr-export-", dir=str(out.parent)))
    try:
        src = model_dir
        if patch_dir is not None and patch_files(model_dir, patch_dir, work / "patched"):
            src = work / "patched"
            say("  applied the patch to the checkpoint files (untouched tensors copied unchanged)")
        elif patch_dir is not None:
            from ..deep.lm import LM
            from . import edits as E
            from .playbook import save_patched_model, serve_dtype

            own = lm is None
            if own:
                lm = LM.load(model_dir, dtype="float32", device="cpu")
            patch = E.Patch.load(patch_dir)
            ck = lm.checkpoint_dtype if lm.checkpoint_dtype in ("bfloat16", "float16") else None
            src = work / "patched"
            save_patched_model(lm, patch, src, serve_dtype(lm, ck) if ck else None)
            for f in model_dir.iterdir():                     # tokenizer and config files the converter reads
                if f.is_file() and not (src / f.name).exists() and f.suffix not in (".safetensors", ".bin", ".pt"):
                    shutil.copy2(f, src / f.name)
            if own:
                del lm
            say(f"  applied {patch.label}")
        outtype = FLOAT_TYPES.get(quant, "f16")
        first = out if quant in FLOAT_TYPES else work / "model-f16.gguf"
        env = {**os.environ, "PYTHONPATH": str(conv / "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", ""),
               "NO_LOCAL_GGUF": ""}
        r = subprocess.run([sys.executable, str(conv / "convert_hf_to_gguf.py"), str(src), "--outfile", str(first),
                            "--outtype", outtype], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if r.returncode != 0:
            raise ValueError("llama.cpp's converter failed: " + _tool_error(r))
        say(f"  converted -> {first.name} ({outtype})")
        if quant not in FLOAT_TYPES:
            assert qbin is not None
            r = subprocess.run([str(qbin), str(first), str(out), quant], capture_output=True, text=True, encoding="utf-8", errors="replace",
                               cwd=str(qbin.parent))
            if r.returncode != 0:
                raise ValueError("llama-quantize failed: " + _tool_error(r))
            say(f"  quantised -> {out.name} ({quant})")
    except BaseException:
        out.unlink(missing_ok=True)                 # never leave a half-written GGUF that looks like a real one
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    from ..identity import _file_sha

    return {"gguf": str(out), "quant": quant, "converter": str(conv), "tag": tag,
            "bytes": out.stat().st_size if out.exists() else 0,
            "sha256": _file_sha(out) if out.exists() else None,
            "source_check": (source_check or {}).get("verdict")}


__all__ = ["export", "find_converter", "fetch_converter", "server_tag", "NeedsConsent"]
