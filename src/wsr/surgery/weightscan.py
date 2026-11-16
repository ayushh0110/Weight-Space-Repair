"""Damaged weights: rows or columns of a transformer layer's weight matrices far larger than the rest of the matrix.

A healthy model has no such entries. Measured on Qwen2.5-0.5B and -1.5B and Llama-3.2-1B: in every 2-D weight of
every layer, the largest row or column norm is at most 13.4x the median of that axis (Llama's MLP neuron 7258 in
layer 15, a known outlier, in both gate and up projections; everything else is under 11x). Corruption that scales
part of a matrix (a bad merge or conversion, a fine-tune that blew up, a wrong scale) leaves entries tens to
hundreds of times larger: the planted test (16 columns of layer 12's MLP output x200 on Qwen2.5-0.5B) gives 279x.

`scan` finds such entries; `repairs` turns a finding into edits that bring them back to the matrix's own scale.
A finding is evidence, not proof (an unusual model could have a real outlier this large), so every repair is a
candidate like any other: verified, then judged. Biases and norm gains are not scanned: healthy models have
entries hundreds of times their median there (Qwen's k_proj bias: 410x), so a large value says nothing. The
embedding and the output head are not scanned either (rare tokens have unusual norms).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ..deep.lm import LM
from . import edits as E

THRESHOLD = 50.0        # a row / column this many times its matrix's median norm is damage (healthy max: 13.4)
RELATIVE = 20.0         # or this many times the median, AND RELATIVE_TO_OTHERS x the same kind in every other layer
RELATIVE_TO_OTHERS = 10.0
GROUP_MIN = 20.0        # entries of a flagged matrix taken as damaged: above GROUP_MIN x the median, cut at a gap
GROUP_GAP = 3.0         # ... of at least this factor between consecutive sorted ratios
MAX_ENTRIES = 256


@dataclass
class Finding:
    param: str                      # parameter name (model.named_parameters)
    layer: int
    kind: str                       # e.g. "mlp.down_proj.weight" (the name inside the layer)
    axis: str                       # "col" (an input dimension: a column of the [out, in] matrix) or "row"
    index: list[int]                # the damaged rows / columns
    ratios: list[float]             # their norm over the axis median
    median: float                   # the axis's median norm
    others: float                   # the largest ratio of the same kind and axis in any other layer
    norms: list[float] = field(default_factory=list)

    def where(self) -> str:
        return f"{self.param} {self.axis}s"

    def as_dict(self, show: int = 32) -> dict[str, Any]:
        return {"param": self.param, "layer": self.layer, "axis": self.axis, "count": len(self.index),
                "index": self.index[:show], "max_ratio": round(max(self.ratios), 1),
                "min_ratio": round(min(self.ratios), 1), "median_norm": round(self.median, 4),
                "same_kind_other_layers_max_ratio": round(self.others, 1)}


def _axis_norms(w: torch.Tensor) -> dict[str, torch.Tensor]:
    """Column norms (one per input dimension) and row norms, accumulated in float32 without a float32 copy."""
    return {"col": torch.linalg.vector_norm(w, dim=0, dtype=torch.float32).cpu(),
            "row": torch.linalg.vector_norm(w, dim=1, dtype=torch.float32).cpu()}


def _group(r: torch.Tensor) -> list[int]:
    """The damaged entries of a flagged axis: the top entries down to the deepest gap of GROUP_GAP x between
    consecutive sorted ratios that is still above GROUP_MIN (a second, smaller damaged group is kept with the
    first); without such a gap, every entry above GROUP_MIN."""
    vals, idx = torch.sort(r, descending=True)
    vals, idx = vals[:MAX_ENTRIES + 1], idx[:MAX_ENTRIES + 1]
    cut = None
    for i in range(len(vals) - 1):
        if vals[i] < GROUP_MIN:
            break
        if vals[i] / max(float(vals[i + 1]), 1e-12) >= GROUP_GAP:
            cut = i + 1
    if cut is None:
        cut = int((vals >= GROUP_MIN).sum())
    return sorted(int(j) for j in idx[:cut].tolist())


def scan(lm: LM, *, threshold: float = THRESHOLD) -> list[Finding]:
    """Every damaged-looking row / column group in the layers' 2-D weights, largest first. Cached per weight
    version (an applied edit changes it)."""
    key = (getattr(lm, "_wver", 0), float(threshold))
    hit = getattr(lm, "_weightscan", None)
    if hit is not None and hit[0] == key:
        return list(hit[1])
    names = E._names(lm)
    stats: dict[tuple[str, str], dict[int, tuple]] = {}      # (kind, axis) -> layer -> (max, median, r, name, norms)
    for li, layer in enumerate(lm.layers):
        for sub, p in layer.named_parameters():
            if p.dim() != 2 or id(p) not in names:
                continue
            full = names[id(p)]
            with torch.no_grad():
                for axis, nrm in _axis_norms(p.detach()).items():
                    med = float(nrm.median())
                    if med <= 0:
                        continue
                    r = nrm / med
                    stats.setdefault((sub, axis), {})[li] = (float(r.max()), med, r, full, nrm)
    out = []
    for (sub, axis), by_layer in stats.items():
        for li, (mx, med, r, full, nrm) in by_layer.items():
            others = max((v[0] for k, v in by_layer.items() if k != li), default=0.0)
            if not (mx >= threshold or (mx >= RELATIVE and mx >= RELATIVE_TO_OTHERS * max(others, 1.0))):
                continue
            idx = _group(r)
            if not idx:
                continue
            out.append(Finding(full, li, sub, axis, idx, [round(float(r[j]), 2) for j in idx], med, others,
                               [float(nrm[j]) for j in idx]))
    out.sort(key=lambda f: -max(f.ratios))
    lm._weightscan = (key, list(out))
    return out


def rescale(lm: LM, param: str, axis: str, index: list[int], factors: list[float], *, label: str = "",
            source: str = "") -> E.Patch:
    """Multiply these rows / columns of `param` by these factors (one each): a low-rank delta like every edit."""
    p = E._param(lm, param)
    W = p.detach().float().cpu()
    f = torch.tensor([float(x) for x in factors]) - 1.0
    k = len(index)
    if axis == "col":
        V = torch.zeros(W.shape[1], k)
        for c, j in enumerate(index):
            V[j, c] = 1.0
        d = E.Delta(param, W[:, index] * f[None, :], V)
    elif axis == "row":
        U = torch.zeros(W.shape[0], k)
        for c, j in enumerate(index):
            U[j, c] = 1.0
        d = E.Delta(param, U, (W[index, :] * f[:, None]).T.contiguous())
    else:
        raise ValueError(f"axis must be col or row, not {axis!r}")
    return E.Patch(label or f"rescale {k} {axis}(s) of {param}", "rescale",
                   {"param": param, "axis": axis, "index": [int(j) for j in index],
                    "factors": [float(x) for x in factors], "source": source}, [d])


def repairs(lm: LM, f: Finding) -> list[tuple[str, Any]]:
    """(label, builder) for a finding, the most faithful first:
    - one common factor: every damaged entry divided by (their median norm / the axis median), which undoes a
      single scale exactly when the damaged entries were typical before (it keeps their norms relative to each
      other);
    - each to the median: every damaged entry set to the axis's median norm (direction kept);
    - zeroed: the entries removed (when what they hold is lost anyway)."""
    common = float(torch.tensor(f.norms).median()) / f.median
    base = f"damaged weights: {len(f.index)} {f.axis}(s) of {f.param}"
    out = [(f"divided by {common:.3g}",
            lambda: rescale(lm, f.param, f.axis, f.index, [1.0 / common] * len(f.index),
                            label=f"{base} divided by {common:.3g}", source=base)),
           ("each to the median norm",
            lambda: rescale(lm, f.param, f.axis, f.index, [f.median / n for n in f.norms],
                            label=f"{base} each to the median norm", source=base)),
           ("zeroed", lambda: rescale(lm, f.param, f.axis, f.index, [0.0] * len(f.index),
                                      label=f"{base} zeroed", source=base))]
    return out


def control(lm: LM, patch: E.Patch, seed: int = 0) -> E.Patch:
    """The random control of a rescale: the same factors on as many random other rows / columns of the same
    matrix. If it fixes the failure as well, the fix is not specific to the entries the scan found."""
    sp = patch.spec
    g = torch.Generator().manual_seed(seed)
    W = E._param(lm, sp["param"])
    width = W.shape[1] if sp["axis"] == "col" else W.shape[0]
    chosen = set(sp["index"])
    pool = [j for j in range(width) if j not in chosen]
    pick = sorted(pool[j] for j in torch.randperm(len(pool), generator=g)[:len(chosen)].tolist())
    return rescale(lm, sp["param"], sp["axis"], pick, sp["factors"], source="random control")


def evidence(f: Finding) -> str:
    return (f"{len(f.index)} {f.axis}(s) of {f.param} are {min(f.ratios):.0f}-{max(f.ratios):.0f}x the matrix's median "
            f"norm (the same kind in other layers: at most {f.others:.1f}x; healthy models measured: at most 13.4x)")


def summary(found: Optional[list[Finding]]) -> str:
    if not found:
        return "no damaged-looking weights: every row and column is within the range healthy models show"
    return "; ".join(evidence(f) for f in found[:3]) + (f"; and {len(found) - 3} more" if len(found) > 3 else "")
