"""Quantised-tensor health (GGUF) and quantisation error against full precision.

Separates "the quant broke it" from "the weights are like that" (Gap 1):

* per-tensor statistics, as in SecAI-Hub/gguf-guard: mean, std, kurtosis,
  outlier ratio, sparsity, NaN/Inf, block-scale entropy, zero and repeated
  blocks, bin saturation; each stat is compared with the same tensor role in
  the other layers (median / MAD), so a single collapsed layer stands out;
* relative quantisation error ||W - W_q||_F / ||W||_F per tensor against the
  original safetensors, which gguf-guard does not do;
* super weights (Yu et al. 2024): the few huge entries of down_proj whose loss
  breaks a model. Their quantisation error is reported on its own.

Only numpy. Formats: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K,
Q6_K. Large tensors are row-sampled so a 5 GB per-layer embedding costs MBs.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..gguf import GGML_TYPES, GGUFFile, TensorInfo, read_gguf

QK_K = 256
MAX_SAMPLE_ROWS = 512


# -- dequantisation (ggml-quants.c reference layouts) -------------------------------

def _f16(b: np.ndarray) -> np.ndarray:
    return b.copy().view(np.float16).astype(np.float32)


def _deq_q4_0(blk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = _f16(blk[:, :2]).reshape(-1, 1)
    qs = blk[:, 2:18]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.int8) - 8
    return q * d, d.ravel(), q + 8


def _deq_q4_1(blk):
    d = _f16(blk[:, :2]).reshape(-1, 1)
    m = _f16(blk[:, 2:4]).reshape(-1, 1)
    qs = blk[:, 4:20]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32)
    return q * d + m, d.ravel(), q


def _qh_bits(qh: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(qh, axis=1, bitorder="little")  # [n, 32]: bit j of the little-endian uint32
    return bits


def _deq_q5_0(blk):
    d = _f16(blk[:, :2]).reshape(-1, 1)
    bits = _qh_bits(blk[:, 2:6])
    qs = blk[:, 6:22]
    lo = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.int16)
    q = (lo | (bits.astype(np.int16) << 4)) - 16
    return q * d, d.ravel(), q + 16


def _deq_q5_1(blk):
    d = _f16(blk[:, :2]).reshape(-1, 1)
    m = _f16(blk[:, 2:4]).reshape(-1, 1)
    bits = _qh_bits(blk[:, 4:8])
    qs = blk[:, 8:24]
    lo = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.int16)
    q = lo | (bits.astype(np.int16) << 4)
    return q * d + m, d.ravel(), q


def _deq_q8_0(blk):
    d = _f16(blk[:, :2]).reshape(-1, 1)
    q = blk[:, 2:34].copy().view(np.int8).astype(np.float32)
    return q * d, d.ravel(), q + 128


def _scale_min_k4(sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[n, 12] packed 6-bit scales/mins -> two [n, 8] arrays."""
    s = np.zeros((sc.shape[0], 8), np.float32)
    m = np.zeros_like(s)
    for j in range(8):
        if j < 4:
            s[:, j] = sc[:, j] & 63
            m[:, j] = sc[:, j + 4] & 63
        else:
            s[:, j] = (sc[:, j + 4] & 0x0F) | ((sc[:, j - 4] >> 6) << 4)
            m[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    return s, m


def _deq_q4_k(blk):
    d = _f16(blk[:, 0:2]).reshape(-1, 1)
    dmin = _f16(blk[:, 2:4]).reshape(-1, 1)
    sc, mn = _scale_min_k4(blk[:, 4:16])
    qs = blk[:, 16:144]
    out = np.empty((blk.shape[0], QK_K), np.float32)
    qv = np.empty_like(out)
    for g in range(4):
        q = qs[:, 32 * g:32 * g + 32]
        for half, nib in ((0, q & 0x0F), (1, q >> 4)):
            j = 2 * g + half
            sl = slice(64 * g + 32 * half, 64 * g + 32 * half + 32)
            out[:, sl] = d * sc[:, j:j + 1] * nib - dmin * mn[:, j:j + 1]
            qv[:, sl] = nib
    return out, (d * sc).ravel(), qv


def _deq_q5_k(blk):
    d = _f16(blk[:, 0:2]).reshape(-1, 1)
    dmin = _f16(blk[:, 2:4]).reshape(-1, 1)
    sc, mn = _scale_min_k4(blk[:, 4:16])
    qh = blk[:, 16:48]
    qs = blk[:, 48:176]
    out = np.empty((blk.shape[0], QK_K), np.float32)
    qv = np.empty_like(out)
    for g in range(4):
        q = qs[:, 32 * g:32 * g + 32]
        for half, nib in ((0, q & 0x0F), (1, q >> 4)):
            j = 2 * g + half
            hb = ((qh >> j) & 1).astype(np.int16) << 4
            v = nib.astype(np.int16) | hb
            sl = slice(64 * g + 32 * half, 64 * g + 32 * half + 32)
            out[:, sl] = d * sc[:, j:j + 1] * v - dmin * mn[:, j:j + 1]
            qv[:, sl] = v
    return out, (d * sc).ravel(), qv


def _deq_q6_k(blk):
    ql = blk[:, 0:128]
    qh = blk[:, 128:192]
    sc = blk[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = _f16(blk[:, 208:210]).reshape(-1, 1)
    out = np.empty((blk.shape[0], QK_K), np.float32)
    qv = np.empty_like(out)
    for n in range(2):
        l_ = ql[:, 64 * n:64 * n + 64]
        h_ = qh[:, 32 * n:32 * n + 32]
        s_ = sc[:, 8 * n:8 * n + 8]
        lo = [l_[:, :32] & 0x0F, l_[:, 32:] & 0x0F, l_[:, :32] >> 4, l_[:, 32:] >> 4]
        for k in range(4):
            q = (lo[k].astype(np.int16) | (((h_ >> (2 * k)) & 3).astype(np.int16) << 4)) - 32
            base = 128 * n + 32 * k
            for half in range(2):  # 16-element sub-blocks
                sl = slice(base + 16 * half, base + 16 * half + 16)
                out[:, sl] = d * s_[:, 2 * k + half:2 * k + half + 1] * q[:, 16 * half:16 * half + 16]
                qv[:, sl] = q[:, 16 * half:16 * half + 16] + 32
    return out, (d * sc).ravel(), qv


_DEQ = {"Q4_0": _deq_q4_0, "Q4_1": _deq_q4_1, "Q5_0": _deq_q5_0, "Q5_1": _deq_q5_1, "Q8_0": _deq_q8_0,
        "Q4_K": _deq_q4_k, "Q5_K": _deq_q5_k, "Q6_K": _deq_q6_k}
_QMAX = {"Q4_0": 15, "Q4_1": 15, "Q5_0": 31, "Q5_1": 31, "Q8_0": 255, "Q4_K": 15, "Q5_K": 31, "Q6_K": 63}
SUPPORTED = {"F32", "F16", "BF16", *_DEQ}


@dataclass
class Tensor:
    """Dequantised rows of one GGUF tensor (all rows, or a sample)."""
    name: str
    qtype: str
    shape: tuple[int, int]          # (rows, cols), numpy order
    rows: np.ndarray                 # row indices held
    values: np.ndarray               # [len(rows), cols] float32
    scales: Optional[np.ndarray] = None
    codes: Optional[np.ndarray] = None
    raw_blocks: Optional[np.ndarray] = None


def _tensor_view(path: Path, g: GGUFFile, t: TensorInfo) -> tuple[np.memmap, int, int, int, int]:
    name, block, size = GGML_TYPES[t.ggml_type]
    cols = t.shape[0]
    rows = 1
    for x in t.shape[1:]:
        rows *= x
    row_bytes = cols // block * size
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=g.data_offset + t.offset, shape=(rows * row_bytes,))
    return mm, rows, cols, block, size


def load_tensor(path: Path, g: GGUFFile, t: TensorInfo, rows: Optional[np.ndarray] = None,
                max_rows: int = MAX_SAMPLE_ROWS, seed: int = 0) -> Tensor:
    """Dequantise a tensor, or `max_rows` evenly spread rows of it."""
    qtype = t.type_name
    if qtype not in SUPPORTED:
        raise ValueError(f"{t.name}: {qtype} not supported")
    mm, n_rows, cols, block, size = _tensor_view(path, g, t)
    if rows is None:
        rows = np.arange(n_rows) if n_rows <= max_rows else np.unique(np.linspace(0, n_rows - 1, max_rows).astype(int))
    row_bytes = cols // block * size
    raw = np.stack([np.asarray(mm[r * row_bytes:(r + 1) * row_bytes]) for r in rows])
    if qtype == "F32":
        vals = raw.view(np.float32).reshape(len(rows), cols)
        return Tensor(t.name, qtype, (n_rows, cols), rows, vals.astype(np.float32))
    if qtype == "F16":
        return Tensor(t.name, qtype, (n_rows, cols), rows, raw.view(np.float16).astype(np.float32).reshape(len(rows), cols))
    if qtype == "BF16":
        u = raw.view(np.uint16).astype(np.uint32) << 16
        return Tensor(t.name, qtype, (n_rows, cols), rows, u.view(np.float32).reshape(len(rows), cols))
    blocks = raw.reshape(-1, size)
    vals, scales, codes = _DEQ[qtype](blocks)
    return Tensor(t.name, qtype, (n_rows, cols), rows, vals.reshape(len(rows), cols).astype(np.float32),
                  scales=scales, codes=codes, raw_blocks=blocks)


# -- statistics ---------------------------------------------------------------------

def tensor_stats(x: Tensor) -> dict[str, Any]:
    v = x.values.astype(np.float64).ravel()
    finite = np.isfinite(v)
    out: dict[str, Any] = {"name": x.name, "type": x.qtype, "shape": list(x.shape),
                           "sampled_rows": int(len(x.rows)), "nan_inf": int((~finite).sum())}
    v = v[finite]
    if not v.size:
        return out
    mu, sd = float(v.mean()), float(v.std())
    out.update(mean=mu, std=sd, max_abs=float(np.abs(v).max()), sparsity=float((v == 0).mean()))
    if sd > 0:
        z = (v - mu) / sd
        out["kurtosis"] = float((z ** 4).mean() - 3.0)          # excess kurtosis; Gaussian = 0
        out["outlier_ratio"] = float((np.abs(z) > 6).mean())      # beyond 6 sigma
    if x.scales is not None and x.scales.size:
        s = np.abs(x.scales[np.isfinite(x.scales)])
        out["zero_scale_blocks"] = float((s == 0).mean())
        nz = s[s > 0]
        if nz.size > 1:
            hist, _ = np.histogram(np.log2(nz), bins=64)
            p = hist[hist > 0] / hist.sum()
            out["scale_entropy"] = float(-(p * np.log2(p)).sum() / math.log2(64))  # 0 = all one scale
    if x.codes is not None:
        q = x.codes
        qmax = _QMAX[x.qtype]
        out["saturation"] = float(((q <= 0) | (q >= qmax)).mean())
    if x.raw_blocks is not None and len(x.raw_blocks) > 1:
        b = np.ascontiguousarray(x.raw_blocks)
        keys = b.view(np.dtype((np.void, b.shape[1]))).ravel()
        nonzero = b.any(axis=1)
        _, counts = np.unique(keys[nonzero], return_counts=True)
        out["repeated_blocks"] = float((counts[counts > 1].sum()) / max(1, nonzero.sum()))
    return out


_LAYER = re.compile(r"^blk\.(\d+)\.(.+)$")
_ROBUST_KEYS = ("std", "kurtosis", "outlier_ratio", "max_abs", "scale_entropy", "saturation", "zero_scale_blocks")


def robust_anomalies(stats: list[dict], z_cut: float = 6.0) -> list[dict]:
    """Per role (attn_q, ffn_down...), flag a layer whose stat is far from the other layers (median / MAD)."""
    roles: dict[str, list[dict]] = {}
    for s in stats:
        m = _LAYER.match(s["name"])
        # Matrices only: norm gains differ by design from layer to layer (Gemma 4's
        # first and KV-sharing layers carry gains of 20-100), so they are no evidence.
        if m and len(s.get("shape", [])) == 2 and min(s["shape"]) > 1:
            roles.setdefault(m.group(2), []).append(s)
    out = []
    for role, group in roles.items():
        if len(group) < 4:
            continue
        for key in _ROBUST_KEYS:
            vals = np.array([g.get(key, np.nan) for g in group], float)
            ok = np.isfinite(vals)
            if ok.sum() < 4:
                continue
            med = np.median(vals[ok])
            mad = np.median(np.abs(vals[ok] - med)) * 1.4826
            if mad <= 1e-12:
                continue
            for g, v in zip(group, vals):
                if np.isfinite(v) and abs(v - med) / mad > z_cut:
                    out.append({"tensor": g["name"], "stat": key, "value": float(v), "role_median": float(med),
                                "robust_z": float((v - med) / mad)})
    return sorted(out, key=lambda a: -abs(a["robust_z"]))


# -- reference comparison ---------------------------------------------------------------

_HF_MAP = [
    (r"^token_embd\.weight$", "model.embed_tokens.weight"),
    (r"^output\.weight$", "lm_head.weight"),
    (r"^output_norm\.weight$", "model.norm.weight"),
    (r"^blk\.(\d+)\.attn_q\.weight$", r"model.layers.\1.self_attn.q_proj.weight"),
    (r"^blk\.(\d+)\.attn_k\.weight$", r"model.layers.\1.self_attn.k_proj.weight"),
    (r"^blk\.(\d+)\.attn_v\.weight$", r"model.layers.\1.self_attn.v_proj.weight"),
    (r"^blk\.(\d+)\.attn_output\.weight$", r"model.layers.\1.self_attn.o_proj.weight"),
    (r"^blk\.(\d+)\.attn_q_norm\.weight$", r"model.layers.\1.self_attn.q_norm.weight"),
    (r"^blk\.(\d+)\.attn_k_norm\.weight$", r"model.layers.\1.self_attn.k_norm.weight"),
    (r"^blk\.(\d+)\.ffn_gate\.weight$", r"model.layers.\1.mlp.gate_proj.weight"),
    (r"^blk\.(\d+)\.ffn_up\.weight$", r"model.layers.\1.mlp.up_proj.weight"),
    (r"^blk\.(\d+)\.ffn_down\.weight$", r"model.layers.\1.mlp.down_proj.weight"),
    (r"^blk\.(\d+)\.attn_norm\.weight$", r"model.layers.\1.input_layernorm.weight"),
    (r"^blk\.(\d+)\.ffn_norm\.weight$", r"model.layers.\1.post_attention_layernorm.weight"),
    (r"^blk\.(\d+)\.post_attention_norm\.weight$", r"model.layers.\1.post_attention_layernorm.weight"),
    (r"^blk\.(\d+)\.post_ffw_norm\.weight$", r"model.layers.\1.post_feedforward_layernorm.weight"),
]


def hf_name(gguf_name: str, available: set[str]) -> Optional[str]:
    """The safetensors name for a GGUF tensor; tries the common prefixes (model., model.language_model.)."""
    for pat, rep in _HF_MAP:
        if re.match(pat, gguf_name):
            base = re.sub(pat, rep, gguf_name)
            tail = base[len("model."):] if base.startswith("model.") else base
            for cand in (base, "model.language_model." + tail, "language_model.model." + tail, tail):
                if cand in available:
                    return cand
    return None


class SafetensorsDir:
    """Lazy row access to every tensor in a HF model directory (one or many shards)."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self.path = Path(path)
        files = sorted(self.path.glob("*.safetensors")) if self.path.is_dir() else [self.path]
        if not files:
            raise FileNotFoundError(f"no .safetensors in {self.path}")
        self._handles = {}
        self.where: dict[str, Path] = {}
        for f in files:
            h = safe_open(str(f), framework="numpy")
            self._handles[f] = h
            for k in h.keys():
                self.where[k] = f

    def names(self) -> set[str]:
        return set(self.where)

    def rows(self, name: str, rows: np.ndarray) -> np.ndarray:
        h = self._handles[self.where[name]]
        sl = h.get_slice(name)
        shape = sl.get_shape()
        if len(shape) == 1:
            return np.asarray(sl[:], dtype=np.float32).reshape(1, -1)
        out = [np.asarray(sl[int(r):int(r) + 1], dtype=np.float32) for r in rows]
        return np.concatenate(out, axis=0).reshape(len(rows), -1)


def _bf16_safe_rows(st: SafetensorsDir, name: str, rows: np.ndarray) -> np.ndarray:
    try:
        return st.rows(name, rows)
    except (TypeError, ValueError):
        # numpy has no bfloat16: read through torch (safetensors' pt framework) when the shard is BF16.
        from safetensors import safe_open

        with safe_open(str(st.where[name]), framework="pt") as h:
            sl = h.get_slice(name)
            if len(sl.get_shape()) == 1:
                return sl[:].float().numpy().reshape(1, -1)
            return np.concatenate([sl[int(r):int(r) + 1].float().numpy() for r in rows]).reshape(len(rows), -1)


def _unpermute_rope(w: np.ndarray, n_head: int) -> np.ndarray:
    """Undo llama.cpp's q/k row permutation for llama-arch rope (HF -> GGUF reorders rows)."""
    r, c = w.shape
    return w.reshape(n_head, r // n_head // 2, 2, c).swapaxes(1, 2).reshape(r, c)


def _rescale_fit(q: np.ndarray, w: np.ndarray, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Row and column scales r, c minimising ||diag(r) q diag(c) - w|| (alternating least squares).

    AWQ / SmoothQuant-style conversions divide per-channel scales into a norm
    gain (or the previous matrix's rows) and multiply them into the next
    matrix's columns: the function is unchanged, but each tensor alone no longer
    matches the original. Only rows + cols free parameters, so block-level
    damage (Q4_0 blocks run along a row) is not absorbed.
    """
    r = np.ones(q.shape[0])
    c = np.ones(q.shape[1])
    for _ in range(iters):
        qc = q * c
        r = (qc * w).sum(1) / np.maximum((qc * qc).sum(1), 1e-30)
        qr = q * r[:, None]
        c = (qr * w).sum(0) / np.maximum((qr * qr).sum(0), 1e-30)
    return r, c


def quant_error(path: Path, g: GGUFFile, t: TensorInfo, ref: SafetensorsDir, ref_name: str,
                n_head: Optional[int] = None, n_head_kv: Optional[int] = None,
                col_weight: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Relative error of the GGUF tensor against the reference, on the same sampled rows.

    col_weight: the original norm gain in front of this matrix (attn_q/k/v, ffn_gate/up).
    Error is then measured on W·diag(gain), what the model actually multiplies the
    normalised residual by, so columns the norm silences do not dominate.
    """
    x = load_tensor(path, g, t)
    full = x.shape[0] <= MAX_SAMPLE_ROWS
    rows = x.rows
    if not full and ("attn_q" in t.name or "attn_k" in t.name):
        # A rope permutation moves rows: compare whole tensors for q/k.
        x = load_tensor(path, g, t, max_rows=10 ** 9)
        rows, full = x.rows, True
    w = _bf16_safe_rows(ref, ref_name, rows)
    q = x.values
    if w.shape != q.shape:
        if w.size == q.size:
            w = w.reshape(q.shape)
        else:
            return {"tensor": t.name, "reference": ref_name, "error": f"shape {q.shape} vs reference {w.shape}"}
    cands = {"plain": w}
    if full and q.shape[0] > 1:
        for nh in {n_head, n_head_kv} - {None}:
            if q.shape[0] % (2 * nh) == 0:
                cands[f"rope_permuted_{nh}"] = _unpermute_rope(w, nh)
    if q.shape[0] == 1:
        cands["plus_one"] = w + 1.0                    # Gemma norms: GGUF stores 1 + w
    cw = None
    if col_weight is not None and q.ndim == 2 and q.shape[0] > 1 and col_weight.size == q.shape[1]:
        cw = np.abs(col_weight.astype(np.float64)).ravel()

    def rel_of(qq: np.ndarray, ww: np.ndarray) -> float:
        if cw is not None:
            qq, ww = qq * cw, ww * cw
        return float(np.linalg.norm(qq - ww)) / (float(np.linalg.norm(ww)) or 1.0)

    best = None
    for label, ww in cands.items():
        rel = rel_of(q, ww)
        if best is None or rel < best[1]:
            best = (label, rel, ww)
    label, rel, ww = best
    out: dict[str, Any] = {}
    if q.ndim == 2 and min(q.shape) > 1:
        # Function-preserving rescale (AWQ-style)? Accept it only when it removes most of the error.
        qq, w2 = (q * cw, ww * cw) if cw is not None else (q, ww)
        r, c = _rescale_fit(qq.astype(np.float64), w2.astype(np.float64))
        rel_s = float(np.linalg.norm(qq * r[:, None] * c - w2)) / (float(np.linalg.norm(w2)) or 1.0)
        if rel_s < 0.7 * rel:
            out.update(plain_rel_error=rel, row_scale_spread=float(np.std(np.log(np.abs(r) + 1e-30))),
                       col_scale_spread=float(np.std(np.log(np.abs(c) + 1e-30))))
            label, rel = label + "+rescaled", rel_s
            q = q * r[:, None] * c
    out = {"tensor": t.name, "reference": ref_name, "type": x.qtype, "rel_error": rel, "layout": label,
           "rows_compared": int(len(rows)), "norm_weighted": cw is not None,
           "cosine": float((q * ww).sum() / ((np.linalg.norm(q) * np.linalg.norm(ww)) or 1)), **out}
    if "ffn_down" in t.name:
        flat = np.abs(ww).ravel()
        k = min(4, flat.size)
        top = np.argpartition(flat, -k)[-k:]
        sw = [{"row": int(rows[i // ww.shape[1]]), "col": int(i % ww.shape[1]), "ref": float(ww.ravel()[i]),
               "quant": float(q.ravel()[i])} for i in top]
        mx = float(flat.max())
        med = float(np.median(flat)) or 1e-12
        out["super_weight_ratio"] = mx / med
        out["largest_weights"] = sorted(sw, key=lambda s: -abs(s["ref"]))
    return out


# -- entry point -----------------------------------------------------------------------

# Expected relative error of a healthy tensor, by type (round-to-nearest on
# roughly Gaussian weights). Several times above this marks a damaged tensor.
EXPECTED_REL_ERROR = {"F32": 1e-6, "F16": 1e-3, "BF16": 5e-3, "Q8_0": 0.01, "Q6_K": 0.02, "Q5_1": 0.03,
                      "Q5_0": 0.035, "Q5_K": 0.035, "Q4_1": 0.06, "Q4_0": 0.07, "Q4_K": 0.06}


@dataclass
class QuantReport:
    gguf: str
    stats: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def processed(self) -> bool:
        """Scales were folded in at conversion (AWQ-style): such tools also move weights beyond rounding."""
        return sum(str(e.get("layout", "")).endswith("+rescaled") for e in self.errors) >= 3

    def worst_errors(self, k: int = 10) -> list[dict]:
        ok = [e for e in self.errors if "rel_error" in e and not e.get("folded_scales")]
        peers: dict[tuple[str, str], list[float]] = {}
        if self.processed:
            # Baseline from this file: the same matrix role and type in the other layers.
            for e in ok:
                m = _LAYER.match(e["tensor"])
                if m:
                    peers.setdefault((m.group(2), e["type"]), []).append(e["rel_error"])
        scored = []
        for e in ok:
            exp = EXPECTED_REL_ERROR.get(e["type"], 0.05)
            m = _LAYER.match(e["tensor"])
            grp = peers.get((m.group(2), e["type"]), []) if m else []
            base = "type"
            if len(grp) >= 4:
                exp, base = max(exp, float(np.median(grp))), "same role in this file"
            scored.append(dict(e, ratio_to_expected=e["rel_error"] / exp, expected=exp, baseline=base))
        return sorted(scored, key=lambda e: -e["ratio_to_expected"])[:k]

    def to_json(self) -> str:
        return json.dumps({"gguf": self.gguf, "anomalies": self.anomalies, "worst_errors": self.worst_errors(20),
                           "stats": self.stats, "errors": self.errors, "skipped": self.skipped}, indent=1)


def analyse(gguf_path: str | Path, reference: Optional[str | Path] = None, *, only: Optional[str] = None,
            max_tensors: Optional[int] = None) -> QuantReport:
    path = Path(gguf_path)
    g = read_gguf(str(path))
    rep = QuantReport(str(path))
    ref = SafetensorsDir(Path(reference)) if reference else None
    names = ref.names() if ref else set()
    n_head = g.arch_key("attention.head_count")
    n_head_kv = g.arch_key("attention.head_count_kv")
    n_head = n_head if isinstance(n_head, int) else None
    n_head_kv = n_head_kv if isinstance(n_head_kv, int) else None
    todo = [t for t in g.tensors if not only or re.search(only, t.name)]
    for t in todo[:max_tensors]:
        if t.type_name not in SUPPORTED:
            rep.skipped.append(f"{t.name} ({t.type_name})")
            continue
        try:
            rep.stats.append(tensor_stats(load_tensor(path, g, t)))
        except (ValueError, OSError) as e:
            rep.skipped.append(f"{t.name}: {e}")
            continue
        if ref:
            rn = hf_name(t.name, names)
            if rn:
                try:
                    rep.errors.append(quant_error(path, g, t, ref, rn, n_head, n_head_kv,
                                                  col_weight=_norm_gain(ref, names, t.name)))
                except (ValueError, KeyError, OSError) as e:
                    rep.errors.append({"tensor": t.name, "reference": rn, "error": str(e)})
    _mark_folded_norms(rep.errors)
    rep.anomalies = robust_anomalies(rep.stats)
    return rep


_READS_NORM = {"attn_q": "attn_norm", "attn_k": "attn_norm", "attn_v": "attn_norm",
               "ffn_gate": "ffn_norm", "ffn_up": "ffn_norm"}


def _norm_gain(ref: SafetensorsDir, names: set[str], gguf_name: str) -> Optional[np.ndarray]:
    """The original norm gain this matrix reads through (input / post-attention layernorm), if any."""
    m = re.match(r"^blk\.(\d+)\.(\w+)\.weight$", gguf_name)
    if not m or m.group(2) not in _READS_NORM:
        return None
    rn = hf_name(f"blk.{m.group(1)}.{_READS_NORM[m.group(2)]}.weight", names)
    if not rn:
        return None
    try:
        return _bf16_safe_rows(ref, rn, np.arange(1)).ravel()
    except (ValueError, KeyError, OSError):
        return None


def _mark_folded_norms(errors: list[dict]) -> None:
    """When the matrices carry folded scales, the norm gains carry the inverse: not damage."""
    folded = [e for e in errors if str(e.get("layout", "")).endswith("+rescaled")]
    if len(folded) < 3:
        return
    for e in errors:
        if e.get("rel_error", 0) > 0.01 and e.get("type") in ("F32", "F16", "BF16") and "norm" in e["tensor"] \
                and not str(e.get("layout", "")).endswith("+rescaled"):
            e["layout"] = "folded_scales"
            e["folded_scales"] = True
