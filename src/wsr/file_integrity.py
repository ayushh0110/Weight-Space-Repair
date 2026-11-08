"""Layer 1: file integrity.

Structural validation of safetensors and GGUF before anything is loaded
(the ModelVet technique, reimplemented in pure Python, and ModelVet itself is
consulted when installed), plus checks ModelVet leaves out: truncated or
partial downloads, shard-index consistency, NaN/Inf/all-zero tensors, and
size / sha256 against the upstream Hugging Face file (LLMDoctor's
``paths-info`` technique).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

from .findings import Cause, Finding, Layer, Severity
from .gguf import GGML_TYPES, GGUFError, read_gguf

MAX_ST_HEADER = 100_000_000  # the safetensors reference implementation's cap

# safetensors dtype → bits per element
ST_DTYPES = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8, "I16": 16, "U16": 16,
    "F16": 16, "BF16": 16, "I32": 32, "U32": 32, "F32": 32, "F64": 64, "I64": 64, "U64": 64,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, "C64": 64,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _f(code: str, sev: Severity, msg: str, path: Path, fix: Optional[str] = None, source: str = "ModelVet-style",
       cause: Cause = Cause.FILE, **ev: Any) -> Finding:
    return Finding(code=code, layer=Layer.FILE, severity=sev, message=f"{path.name}: {msg}", cause=cause,
                   fix=fix, source=source, evidence=dict(ev, path=str(path)))


REDOWNLOAD = "re-download the file and verify its sha256 against the Hub"


# --------------------------------------------------------------------------
# safetensors
# --------------------------------------------------------------------------

def _no_dupes(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Returns (header, data_start). Raises ValueError with a precise reason."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError(f"file is {size} bytes; too small for a safetensors header")
        (n,) = struct.unpack("<Q", head)
        if n > MAX_ST_HEADER:
            raise ValueError(f"header length {n} exceeds the {MAX_ST_HEADER}-byte cap")
        if 8 + n > size:
            raise ValueError(f"header length {n} runs past end of file ({size} bytes)")
        raw = f.read(n)
    if not raw.startswith(b"{"):
        raise ValueError("header does not start with '{'")
    header = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dupes)
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return header, 8 + n


def check_safetensors(path: Path, scan_values: bool = False) -> list[Finding]:
    out: list[Finding] = []
    size = path.stat().st_size
    try:
        header, start = read_safetensors_header(path)
    except (ValueError, UnicodeDecodeError) as e:
        return [_f("ST_HEADER_INVALID", Severity.FAIL, str(e), path, REDOWNLOAD)]

    meta = header.pop("__metadata__", None)
    if meta is not None and (not isinstance(meta, dict) or
                             not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())):
        out.append(_f("ST_METADATA_INVALID", Severity.WARN, "__metadata__ must map strings to strings", path))

    spans: list[tuple[int, int, str]] = []
    problems: list[str] = []
    for name, info in header.items():
        if not isinstance(info, dict):
            problems.append(f"{name}: entry is not an object")
            continue
        extra = set(info) - {"dtype", "shape", "data_offsets"}
        if extra:
            problems.append(f"{name}: unknown fields {sorted(extra)}")
        dt, shape, offs = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dt not in ST_DTYPES:
            problems.append(f"{name}: unknown dtype {dt!r}")
            continue
        if not isinstance(shape, list) or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0
                                                  for d in shape):
            problems.append(f"{name}: bad shape {shape!r}")
            continue
        if (not isinstance(offs, list) or len(offs) != 2 or
                not all(isinstance(o, int) and not isinstance(o, bool) and o >= 0 for o in offs)):
            problems.append(f"{name}: bad data_offsets {offs!r}")
            continue
        b, e = offs
        if b > e:
            problems.append(f"{name}: data_offsets begin {b} > end {e}")
            continue
        n = 1
        for d in shape:
            n *= d
        bits = n * ST_DTYPES[dt]
        if bits % 8 or bits // 8 != e - b:
            problems.append(f"{name}: {dt}{shape} needs {bits / 8:g} bytes, offsets span {e - b}")
        spans.append((b, e, name))

    if problems:
        out.append(_f("ST_TENSOR_INVALID", Severity.FAIL, f"{len(problems)} bad tensor entries; first: {problems[0]}",
                      path, REDOWNLOAD, problems=problems[:20]))

    spans.sort()
    cursor = 0
    for b, e, name in spans:
        if b != cursor:
            kind = "gap" if b > cursor else "overlap"
            out.append(_f("ST_OFFSETS_NOT_TILED", Severity.FAIL, f"{kind} before tensor {name!r} at offset {b}",
                          path, REDOWNLOAD))
            break
        cursor = e
    data_len = size - start
    if spans and cursor != data_len:
        if cursor > data_len:
            out.append(_f("TRUNCATED_FILE", Severity.FAIL,
                          f"tensors need {cursor} data bytes but the file holds {data_len}: incomplete download",
                          path, REDOWNLOAD, missing_bytes=cursor - data_len))
        else:
            out.append(_f("ST_TRAILING_BYTES", Severity.WARN, f"{data_len - cursor} bytes after the last tensor", path))

    if not out:
        out.append(_f("ST_STRUCTURE_OK", Severity.PASS, f"{len(spans)} tensors, structure valid", path))

    if scan_values and not any(x.severity == Severity.FAIL for x in out):
        out.extend(_scan_values(path, header, start))
    return out


def _scan_values(path: Path, header: dict, start: int) -> list[Finding]:
    """NaN / Inf / all-zero tensors (bad merges, broken conversions)."""
    try:
        import numpy as np
    except ImportError:
        return [_f("VALUE_SCAN_SKIPPED", Severity.SKIP, "numpy not installed", path)]
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    bad_nan, bad_zero = [], []
    for name, info in header.items():
        dt = info["dtype"]
        b, e = info["data_offsets"]
        if e == b:
            continue
        raw = mm[start + b:start + e]
        if dt == "F32":
            arr = raw.view(np.float32)
        elif dt == "F16":
            arr = raw.view(np.float16)
        elif dt == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            continue
        if not np.isfinite(arr).all():
            bad_nan.append(name)
        elif not arr.any() and "bias" not in name:
            bad_zero.append(name)
    out = []
    if bad_nan:
        out.append(_f("NAN_OR_INF_WEIGHTS", Severity.FAIL, f"{len(bad_nan)} tensors contain NaN/Inf: {bad_nan[:5]}",
                      path, "the conversion or merge that produced this file is broken; rebuild it from the source",
                      source="wsr", tensors=bad_nan[:50]))
    if bad_zero:
        out.append(_f("ALL_ZERO_TENSORS", Severity.WARN, f"{len(bad_zero)} weight tensors are all zeros: {bad_zero[:5]}",
                      path, source="wsr", tensors=bad_zero[:50]))
    if not out:
        out.append(_f("VALUES_OK", Severity.PASS, "no NaN/Inf or all-zero weight tensors", path, source="wsr"))
    return out


def check_shard_index(index_path: Path) -> list[Finding]:
    try:
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = idx["weight_map"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [_f("SHARD_INDEX_INVALID", Severity.FAIL, f"cannot read index: {e}", index_path, REDOWNLOAD)]
    out: list[Finding] = []
    by_file: dict[str, set[str]] = {}
    for t, fname in weight_map.items():
        by_file.setdefault(fname, set()).add(t)
    for fname, names in sorted(by_file.items()):
        p = index_path.parent / fname
        if not p.exists():
            out.append(_f("SHARD_MISSING", Severity.FAIL, f"shard {fname} listed in index but absent", index_path,
                          REDOWNLOAD))
            continue
        try:
            header, _ = read_safetensors_header(p)
        except (ValueError, UnicodeDecodeError):
            continue  # reported by the per-file check
        missing = names - set(header)
        if missing:
            out.append(_f("SHARD_TENSORS_MISSING", Severity.FAIL,
                          f"{fname} lacks {len(missing)} tensors the index assigns to it", index_path, REDOWNLOAD,
                          tensors=sorted(missing)[:20]))
    if not out:
        out.append(_f("SHARD_INDEX_OK", Severity.PASS, f"{len(by_file)} shards consistent with index", index_path))
    return out


# --------------------------------------------------------------------------
# GGUF
# --------------------------------------------------------------------------

def check_gguf(path: Path) -> list[Finding]:
    try:
        g = read_gguf(str(path))
    except GGUFError as e:
        msg = str(e)
        trunc = "past end of file" in msg
        return [_f("TRUNCATED_FILE" if trunc else "GGUF_HEADER_INVALID", Severity.FAIL, msg, path, REDOWNLOAD)]
    out: list[Finding] = []
    data_len = g.file_size - g.data_offset
    spans, unknown = [], []
    for t in g.tensors:
        if t.offset % g.alignment:
            out.append(_f("GGUF_MISALIGNED", Severity.FAIL, f"tensor {t.name!r} offset {t.offset} not "
                          f"{g.alignment}-aligned", path, REDOWNLOAD))
            break
        if any(d == 0 for d in t.shape):
            out.append(_f("GGUF_ZERO_DIM", Severity.FAIL, f"tensor {t.name!r} has a zero dimension {t.shape}", path,
                          REDOWNLOAD))
            break
        nb = t.nbytes()
        if nb is None:
            unknown.append(t.name)
            continue
        spans.append((t.offset, t.offset + nb, t.name))
    spans.sort()
    for (b1, e1, n1), (b2, _, n2) in zip(spans, spans[1:]):
        if b2 < e1:
            out.append(_f("GGUF_OVERLAP", Severity.FAIL, f"tensors {n1!r} and {n2!r} overlap", path, REDOWNLOAD))
            break
    need = max((e for _, e, _ in spans), default=0)
    if need > data_len:
        out.append(_f("TRUNCATED_FILE", Severity.FAIL,
                      f"tensor data needs {need} bytes but the file holds {data_len}: incomplete download",
                      path, REDOWNLOAD, missing_bytes=need - data_len))
    if unknown:
        out.append(_f("GGUF_UNKNOWN_TYPES", Severity.INFO, f"{len(unknown)} tensors use ggml types this checker "
                      "does not size; skipped", path))
    if not any(x.severity == Severity.FAIL for x in out):
        q = g.quant or "?"
        out.append(_f("GGUF_STRUCTURE_OK", Severity.PASS,
                      f"GGUF v{g.version}, {g.arch}, {q}, {len(g.tensors)} tensors, structure valid", path,
                      arch=g.arch, quant=q))
    return out


# --------------------------------------------------------------------------
# ModelVet (optional second opinion)
# --------------------------------------------------------------------------

def check_with_modelvet(path: Path) -> list[Finding]:
    try:
        import modelvet  # type: ignore
    except ImportError:
        return []
    try:
        r = modelvet.verify_path(str(path))
    except Exception as e:  # noqa: BLE001
        return [_f("MODELVET_ERROR", Severity.INFO, f"modelvet raised {e!r}", path, source="ModelVet")]
    if getattr(r, "accepted", False):
        return [_f("MODELVET_ACCEPT", Severity.PASS, "ModelVet ACCEPT", path, source="ModelVet")]
    return [_f("MODELVET_REJECT", Severity.FAIL, f"ModelVet REJECT: {getattr(r, 'violation_name', '?')}", path,
               REDOWNLOAD, source="ModelVet")]


# --------------------------------------------------------------------------
# Provenance: size / sha256 vs upstream
# --------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hf_paths_info(repo: str, paths: Iterable[str], revision: str = "main", timeout: float = 20) -> dict[str, dict]:
    url = f"https://huggingface.co/api/models/{repo}/paths-info/{urllib.parse.quote(revision, safe='')}"
    body = urllib.parse.urlencode([("paths", p) for p in paths]).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded",
                                          "User-Agent": "wsr-triage"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {e["path"]: e for e in json.load(resp)}


def check_provenance(files: list[Path], root: Path, upstream: Optional[str], verify_hash: bool) -> list[Finding]:
    out: list[Finding] = []
    # HF cache blobs are named by their sha256: that is a free upstream reference.
    for p in files:
        real = Path(os.path.realpath(p))
        if verify_hash and _HEX64.match(real.name):
            got = sha256_file(real)
            if got != real.name:
                out.append(_f("CHECKSUM_MISMATCH", Severity.FAIL, "contents do not match the HF cache blob name",
                              p, REDOWNLOAD, source="LLMDoctor", expected=real.name, actual=got))
            else:
                out.append(_f("CHECKSUM_OK", Severity.PASS, "sha256 matches HF cache blob", p, source="LLMDoctor"))
    if not upstream:
        return out
    repo, _, rev = upstream.partition("@")
    rels = [p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name for p in files]
    try:
        info = hf_paths_info(repo, rels, rev or "main")
    except Exception as e:  # noqa: BLE001
        return out + [_f("UPSTREAM_UNREACHABLE", Severity.SKIP, f"could not query {repo}: {e}", root,
                         source="LLMDoctor")]
    for p, rel in zip(files, rels):
        up = info.get(rel)
        if up is None:
            out.append(_f("UPSTREAM_MISSING", Severity.INFO, f"{rel} not in {repo}", p, source="LLMDoctor"))
            continue
        lfs = up.get("lfs") or {}
        want_size = lfs.get("size", up.get("size"))
        have = p.stat().st_size
        if want_size is not None and want_size != have:
            out.append(_f("SIZE_MISMATCH", Severity.FAIL, f"{have} bytes locally, {want_size} upstream", p, REDOWNLOAD,
                          source="LLMDoctor"))
            continue
        if verify_hash and lfs.get("oid"):
            got = sha256_file(p)
            if got != lfs["oid"]:
                out.append(_f("CHECKSUM_MISMATCH", Severity.FAIL, "sha256 differs from upstream (outdated or corrupt)",
                              p, REDOWNLOAD, source="LLMDoctor", expected=lfs["oid"], actual=got))
                continue
        out.append(_f("UPSTREAM_MATCH", Severity.PASS, "matches upstream" + (" sha256" if verify_hash else " size"),
                      p, source="LLMDoctor"))
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def model_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(list(target.glob("*.safetensors")) + list(target.glob("*.gguf")))


def check_files(target: Path, *, scan_values: bool = False, upstream: Optional[str] = None,
                verify_hash: bool = False) -> list[Finding]:
    out: list[Finding] = []
    root = target if target.is_dir() else target.parent
    files = model_files(target)
    if target.is_dir():
        for partial in list(root.glob("*.incomplete")) + list(root.glob("*-partial")) + list(root.glob("*.part")):
            out.append(_f("PARTIAL_DOWNLOAD", Severity.FAIL, "unfinished download present", partial, REDOWNLOAD,
                          source="LLMDoctor"))
        idx = root / "model.safetensors.index.json"
        if idx.exists():
            out.extend(check_shard_index(idx))
    if not files:
        out.append(_f("NO_WEIGHTS", Severity.FAIL, "no .safetensors or .gguf files found", target,
                      "point at the model directory or a weights file"))
        return out
    for p in files:
        if p.suffix == ".gguf":
            out.extend(check_gguf(p))
        else:
            out.extend(check_safetensors(p, scan_values=scan_values))
        out.extend(check_with_modelvet(p))
    out.extend(check_provenance(files, root, upstream, verify_hash))
    return out


__all__ = ["check_files", "check_safetensors", "check_gguf", "check_shard_index", "read_safetensors_header",
           "GGML_TYPES"]
