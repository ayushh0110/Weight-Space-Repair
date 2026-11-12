"""Weight edits as low-rank deltas: W += U @ V.T on named parameters.

Each edit is the weight-space form of an intervention the deep layer already
tested with hooks, so a patch can be checked against the hook it came from:

* orthogonalise(d)   - remove direction d from every write to the residual stream
                       (token embeddings, every attention output and MLP output
                       matrix and bias). On a pre-norm model this is exactly
                       `lm.steer(ablate=d)` (Arditi et al. 2024 directional
                       ablation, as used by abliteration tools). alpha < 1
                       removes only that fraction.
* project(sites)     - the local form: take a direction out of chosen layers' output
                       projections only, optionally only chosen MLP neurons, at any
                       strength (alpha > 1 pushes against it); `lm.steer(project_sites=...)`.
* scale_neurons(ns)  - scale MLP neurons' output columns (0 = `zero_neurons`).
* scale_heads(hs)    - scale attention heads' output columns (0 = `zero_heads`).
* gate(...)          - rewrite one unused MLP neuron into a conditional circuit: it fires only
                       where a trigger direction is present and writes against the failure
                       there (adds behaviour instead of removing it everywhere).
* control(patch)     - a random edit of the same kind and size, for the specificity check.

A patch is small (rank 1 per matrix for a direction), reversible, and saved as
safetensors U / V pairs plus a JSON description. Tied input / output embeddings
are untied first when the embeddings are edited: the hooks never touch the
unembedding, so the patch must not either.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import torch
from torch import nn

from ..deep.lm import LM


@dataclass
class Delta:
    name: str                 # parameter name in model.named_parameters()
    U: torch.Tensor           # [out, r]  (CPU float32)
    V: torch.Tensor           # [in, r]; a bias (1-D parameter) uses V = [[1]] * r

    def dense(self) -> torch.Tensor:
        return self.U @ self.V.T

    def norm(self) -> float:
        """Frobenius norm of U V^T without forming it (the embedding's delta is vocab x d)."""
        return float(torch.sqrt(((self.U.T @ self.U) * (self.V.T @ self.V)).sum().clamp(min=0)))


@dataclass
class Site:
    """One output projection a direction edit touches: layer, "mlp" (down_proj) or "attn" (o_proj),
    the unit direction to take out of its writes, and optionally only these input columns (MLP
    neurons); None = the whole matrix."""
    layer: int
    part: str
    u: torch.Tensor
    cols: Optional[list[int]] = None


@dataclass
class Patch:
    label: str
    kind: str                                  # orthogonalise | project | scale_neurons | scale_heads
    spec: dict[str, Any] = field(default_factory=dict)
    deltas: list[Delta] = field(default_factory=list)
    untie: bool = False                        # untie lm_head from the embeddings before applying
    checks: dict[str, Any] = field(default_factory=dict)    # what verified it (mode, results); saved
    identity: dict[str, Any] = field(default_factory=dict)  # the model it was made on (wsr/identity.py); saved
    legacy_base: bool = False                  # saved by an older wsr with two aggregate sums instead of hashes

    @property
    def n_params_touched(self) -> int:
        return sum(d.U.shape[0] * (d.V.shape[0] if d.V.shape[0] > 1 else 1) for d in self.deltas)

    @property
    def n_free_params(self) -> int:
        """Numbers the edit is free to choose: rank x (rows + input columns it touches) per tensor, at most the
        entries themselves. A rank-1 projection of one direction out of every layer's writes is a few hundred
        thousand, though it touches every entry of those matrices; this, not the touched entries, is what can
        fit (or overfit) the examples."""
        n = 0
        for d in self.deltas:
            cols = int((d.V.abs().sum(1) > 0).sum()) if d.V.shape[0] > 1 else 1
            rows = int((d.U.abs().sum(1) > 0).sum())
            n += min(int(d.U.shape[1]) * (rows + cols), rows * cols)
        return n

    @property
    def n_weights_changed(self) -> int:
        """Entries the patch can change: rows x the input columns it reads (a column mask counts its columns)."""
        n = 0
        for d in self.deltas:
            cols = int((d.V.abs().sum(1) > 0).sum()) if d.V.shape[0] > 1 else 1
            rows = int((d.U.abs().sum(1) > 0).sum())
            n += rows * cols
        return n

    @property
    def size_floats(self) -> int:
        return sum(d.U.numel() + d.V.numel() for d in self.deltas)

    # -- persistence -----------------------------------------------------------------------------
    def save(self, out: Path, lm: Optional["LM"] = None) -> None:
        """`lm`: the unedited model the patch was made on. Its exact identity (wsr/identity.py) is written with the
        patch as identity.json, so applying the patch to other weights is refused."""
        from safetensors.torch import save_file

        from .. import identity as ID

        if lm is not None and not self.identity:
            self.identity = ID.for_patch(lm, self)
        out.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for k, d in enumerate(self.deltas):
            tensors[f"{k}.U"] = d.U.contiguous()
            tensors[f"{k}.V"] = d.V.contiguous()
        # Tensor-valued spec entries (the direction) are kept, so a reloaded patch can be checked
        # again, random control included.
        for k, v in self.spec.items():
            if isinstance(v, torch.Tensor):
                tensors[f"spec.{k}"] = v.detach().float().cpu().clone().contiguous()   # own memory
        save_file(tensors, str(out / "patch.safetensors"))
        meta = {"label": self.label, "kind": self.kind, "untie": self.untie,
                "params": [d.name for d in self.deltas], "spec": _jsonable(self.spec),
                "checks": _jsonable(self.checks)}
        (out / "patch.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        if self.identity:
            (out / "identity.json").write_text(json.dumps(self.identity), encoding="utf-8")
        else:
            (out / "identity.json").unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "Patch":
        from safetensors.torch import load_file

        meta = json.loads((path / "patch.json").read_text(encoding="utf-8"))
        t = load_file(str(path / "patch.safetensors"))
        deltas = [Delta(n, t[f"{k}.U"], t[f"{k}.V"]) for k, n in enumerate(meta["params"])]
        spec = dict(meta.get("spec", {}))
        for k in list(spec):
            if f"spec.{k}" in t:
                spec[k] = t[f"spec.{k}"]
        ident = json.loads((path / "identity.json").read_text(encoding="utf-8")) \
            if (path / "identity.json").exists() else {}
        return cls(meta["label"], meta["kind"], spec, deltas, meta.get("untie", False), meta.get("checks", {}),
                   ident, legacy_base=bool(meta.get("base")) and not ident)


def _jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return {"tensor": list(x.shape)}
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# -- parameters ------------------------------------------------------------------------------------

def _names(lm: LM) -> dict[int, str]:
    return {id(p): n for n, p in lm.model.named_parameters(remove_duplicate=False)}


def _embedding(lm: LM) -> nn.Embedding:
    return lm.model.get_input_embeddings()


def _embedding_name(lm: LM) -> str:
    """The input embedding's own parameter name. With tied embeddings the same tensor is also
    `lm_head.weight`, and after untying that name belongs to the unembedding: the patch must name
    the embedding, or it edits the output layer instead."""
    emb = _embedding(lm)
    for n, mod in lm.model.named_modules():
        if mod is emb:
            return f"{n}.weight" if n else "weight"
    raise ValueError("input embedding module not found")


def tied(lm: LM) -> bool:
    out = lm.model.get_output_embeddings()
    return out is not None and out.weight is _embedding(lm).weight


def untie(lm: LM) -> None:
    """Give lm_head its own copy of the embedding matrix (no-op when already separate)."""
    if not tied(lm):
        return
    out = lm.model.get_output_embeddings()
    out.weight = nn.Parameter(_embedding(lm).weight.detach().clone())
    lm.model.config.tie_word_embeddings = False
    if hasattr(lm.model.config, "get_text_config"):
        try:
            lm.model.config.get_text_config().tie_word_embeddings = False
        except AttributeError:
            pass


def _param(lm: LM, name: str) -> nn.Parameter:
    p = dict(lm.model.named_parameters(remove_duplicate=False)).get(name)
    if p is None:
        raise ValueError(f"the patch changes {name}, which this model does not have: it was made for another model")
    return p


def identity_check(lm: LM, patch: "Patch", *, patched: bool = False) -> dict[str, Any]:
    """Is this the model the patch was made on (or, with `patched`, that model with this patch applied)? Exact, by
    tensor hashes (wsr/identity.py): {verdict: same | different | unverified, why, ...}. "unverified": the patch
    records no identity (saved by an older wsr, or without the model)."""
    from .. import identity as ID

    if not patch.identity:
        why = ("the patch records only two aggregate sums of its base tensors (an older wsr), which cannot identify "
               "the weights" if patch.legacy_base else "the patch records no identity of the model it was made on")
        return {"verdict": "unverified", "why": why}
    return ID.check(lm, patch.identity, patched=patch.identity.get("patched") if patched else None)


def fits(lm: LM, patch: "Patch") -> None:
    """Raise ValueError unless every delta names a parameter of this model with the right shape. Run before
    anything is changed, so a patch for another model is refused whole instead of applied in part."""
    for d in patch.deltas:
        p = _param(lm, d.name)
        want = (d.U.shape[0], d.V.shape[0]) if p.dim() == 2 else (d.U.shape[0],)
        if tuple(p.shape) != want or (p.dim() == 1 and d.V.shape[0] != 1) or p.dim() > 2:
            raise ValueError(f"the patch's change to {d.name} has shape {want}, the model's tensor is "
                             f"{tuple(p.shape)}: it was made for another model")


# -- builders --------------------------------------------------------------------------------------

def orthogonalise(lm: LM, d: torch.Tensor, alpha: float = 1.0, *, layers: Optional[Sequence[int]] = None,
                  embed: bool = True, label: str = "", source: str = "") -> Patch:
    """Remove alpha of direction d from every residual write: W <- W - alpha d (d^T W), b <- b - alpha d (d.b)."""
    names = _names(lm)
    u = (d / d.norm()).float().cpu()
    deltas: list[Delta] = []
    if embed:
        E = _embedding(lm).weight.detach().float().cpu()
        deltas.append(Delta(_embedding_name(lm), -alpha * (E @ u)[:, None], u[:, None]))
    for i in (range(lm.n_layers) if layers is None else layers):
        for mod in (lm.o_proj(i), lm.down_proj(i)):
            if mod is None:
                raise ValueError(f"layer {i}: no attention or MLP output projection found")
            W = mod.weight.detach().float().cpu()              # [d_model, d_in]
            deltas.append(Delta(names[id(mod.weight)], -alpha * u[:, None], (u @ W)[:, None]))
            if mod.bias is not None:
                b = mod.bias.detach().float().cpu()
                deltas.append(Delta(names[id(mod.bias)], (-alpha * float(u @ b)) * u[:, None], torch.ones(1, 1)))
    return Patch(label or f"remove direction (alpha {alpha:g})", "orthogonalise",
                 {"alpha": alpha, "layers": "all" if layers is None else list(layers), "embed": embed,
                  "source": source, "direction": u}, deltas, untie=embed and tied(lm))


def _scale_columns(lm: LM, mod: nn.Linear, cols: list[int], s: float) -> Delta:
    W = mod.weight.detach().float().cpu()
    V = torch.zeros(W.shape[1], len(cols))
    for k, c in enumerate(cols):
        V[c, k] = 1.0
    return Delta(_names(lm)[id(mod.weight)], (s - 1.0) * W[:, cols], V)


OVERWRITE_KINDS = ("gate", "agent_set_columns", "agent_set_rows")


def _footprint(d: "Delta") -> tuple[set, set]:
    rows = set(torch.nonzero(d.U.abs().sum(1) > 0).flatten().tolist())
    cols = set(torch.nonzero(d.V.abs().sum(1) > 0).flatten().tolist()) if d.V.shape[0] > 1 else {0}
    return rows, cols


def overwrite_clash(parts: Sequence["Patch"]) -> str:
    """Why these patches cannot be added up, or ''. An overwrite (a gate taking over a neuron, set_columns,
    set_rows) is stored as new - old; a second edit of the same entries computed from the same old weights
    makes the sum neither of them. Additive edits (projections, scales, free deltas) compose."""
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if a.kind not in OVERWRITE_KINDS and b.kind not in OVERWRITE_KINDS:
                continue
            for da in a.deltas:
                ra, ca = _footprint(da)
                for db in b.deltas:
                    if da.name != db.name:
                        continue
                    rb, cb = _footprint(db)
                    if ra & rb and ca & cb:
                        return (f"cannot combine {a.label!r} and {b.label!r}: both change the same entries of {da.name} "
                                "and at least one overwrites them, so their sum is neither edit (two gates at one "
                                "layer take the same quiet neuron: give one a \"neuron\")")
    return ""


def scale_neurons(lm: LM, neurons: Sequence[tuple[int, int]], s: float = 0.0, *, label: str = "",
                  source: str = "") -> Patch:
    """Scale MLP neurons' contribution: column j of layer i's down projection times s."""
    by_layer: dict[int, list[int]] = {}
    for i, j in neurons:
        by_layer.setdefault(int(i), []).append(int(j))
    deltas = [_scale_columns(lm, lm.down_proj(i), sorted(set(js)), s) for i, js in sorted(by_layer.items())]
    return Patch(label or f"scale {len(neurons)} neuron(s) x{s:g}", "scale_neurons",
                 {"neurons": [list(n) for n in neurons], "scale": s, "source": source}, deltas)


def scale_heads(lm: LM, heads: Sequence[tuple[int, int]], s: float = 0.0, *, label: str = "",
                source: str = "") -> Patch:
    """Scale attention heads' contribution: their block of the output projection's input columns times s."""
    by_layer: dict[int, list[int]] = {}
    for i, h in heads:
        hd = lm.head_dim(int(i))
        by_layer.setdefault(int(i), []).extend(range(int(h) * hd, (int(h) + 1) * hd))
    deltas = [_scale_columns(lm, lm.o_proj(i), sorted(set(cs)), s) for i, cs in sorted(by_layer.items())]
    return Patch(label or f"scale {len(heads)} head(s) x{s:g}", "scale_heads",
                 {"heads": [list(h) for h in heads], "scale": s, "source": source}, deltas)


def project(lm: LM, sites: Sequence[Site], alpha: float = 1.0, *, label: str = "", source: str = "") -> Patch:
    """Take alpha of each site's direction out of that site's writes only:
    W <- W - alpha u (m * (u^T W)), m the column mask (every column when the site has none).

    The local form of `orthogonalise`: only the layers (and, for an MLP, the neurons) that write
    the direction are edited. alpha 1 removes the site's write along u; alpha > 1 pushes against
    the direction the rest of the network still writes (the edit that fixed Qwen3.5-0.8B's
    reasoning loops was this, on 64 neurons in each of two MLPs, at alpha 6). Hook form:
    `lm.run(..., project_sites=(sites, alpha))`."""
    names = _names(lm)
    deltas: list[Delta] = []
    for s in sites:
        mod = lm.down_proj(s.layer) if s.part == "mlp" else lm.o_proj(s.layer)
        if mod is None:
            raise ValueError(f"layer {s.layer}: no {s.part} output projection found")
        u = (s.u / s.u.norm()).float().cpu()
        W = mod.weight.detach().float().cpu()
        wu = u @ W
        if s.cols is not None:
            m = torch.zeros_like(wu)
            m[list(s.cols)] = 1.0
            wu = wu * m
        deltas.append(Delta(names[id(mod.weight)], -alpha * u[:, None], wu[:, None]))
        if s.cols is None and mod.bias is not None:
            b = mod.bias.detach().float().cpu()
            deltas.append(Delta(names[id(mod.bias)], (-alpha * float(u @ b)) * u[:, None], torch.ones(1, 1)))
    where = ", ".join(f"layer {s.layer} {s.part}" + (f" ({len(s.cols)} neurons)" if s.cols is not None else "")
                      for s in sites)
    return Patch(label or f"push direction out of {where} (alpha {alpha:g})", "project",
                 {"alpha": alpha, "source": source,
                  "sites": [{"layer": s.layer, "part": s.part, "cols": None if s.cols is None else list(s.cols)}
                            for s in sites],
                  "directions": torch.stack([(s.u / s.u.norm()).float().cpu() for s in sites])}, deltas)


def _row_delta(names: dict, W: torch.Tensor, param: nn.Parameter, j: int, new_row: torch.Tensor) -> Delta:
    """Replace row j of a [out, in] weight: U = e_j, V = new - old."""
    U = torch.zeros(W.shape[0], 1)
    U[j, 0] = 1.0
    return Delta(names[id(param)], U, (new_row - W[j])[:, None])


def _col_delta(names: dict, W: torch.Tensor, param: nn.Parameter, j: int, new_col: torch.Tensor) -> Delta:
    """Replace column j of a [out, in] weight: U = new - old, V = e_j."""
    V = torch.zeros(W.shape[1], 1)
    V[j, 0] = 1.0
    return Delta(names[id(param)], (new_col - W[:, j])[:, None], V)


def gate(lm: LM, layer: int, neuron: int, read: torch.Tensor, write: torch.Tensor, *, threshold: float = 0.0,
         up_scale: float = 1.0, mean_dir: Optional[torch.Tensor] = None, mean_proj: Optional[float] = None,
         sharpness: float = 8.0, strength: float = 1.0, label: str = "", source: str = "") -> Patch:
    """Rewrite one MLP neuron into a conditional circuit: it fires only where the normed input's
    projection p on `read` exceeds `threshold`, and then writes about -strength * `write`.

    The removal edits act everywhere, including where the behaviour is wanted (a refusal direction
    removed for the over-refused prompt is removed for harmful ones too). A gated neuron is silent
    elsewhere. SwiGLU MLPs have no bias, so there are two forms:

    * threshold 0 (every position that must stay unchanged projects below 0): gate row
      sharpness * r, up row r / up_scale (up_scale = the mean p where it fires), so the neuron's
      activation is about sharpness * p^2 / up_scale above 0 and ~0 below;
    * threshold > 0: the bias comes from the residual stream's near-constant mean direction m
      (mean projection mean_proj): gate row sharpness * (r - threshold * m / mean_proj), up row
      m / mean_proj (~1). Only valid where the input's projection on m really is near constant.

    The neuron overwritten should be one the model barely uses (`surgery.sites.quiet_neurons`).
    Needs a gate_proj / up_proj / down_proj MLP.
    """
    mlp = lm.mlp(layer)
    g, up, down = getattr(mlp, "gate_proj", None), getattr(mlp, "up_proj", None), lm.down_proj(layer)
    if g is None or up is None or down is None:
        raise ValueError(f"layer {layer}: a gated edit needs a gate_proj / up_proj / down_proj MLP")
    names = _names(lm)
    r = (read / read.norm()).float().cpu()
    w = (write / write.norm()).float().cpu()
    Wg, Wu, Wd = (x.weight.detach().float().cpu() for x in (g, up, down))
    if mean_dir is None:
        if threshold != 0.0:
            raise ValueError("a gate with a non-zero threshold needs the mean direction for its bias")
        grow, urow, m = sharpness * r, r / up_scale, None
        out_scale = strength * up_scale / sharpness       # activation ~ sharpness * p^2 / up_scale
    else:
        m = (mean_dir / mean_dir.norm()).float().cpu()
        grow, urow = sharpness * (r - threshold * m / mean_proj), m / mean_proj
        out_scale = strength / sharpness                 # activation ~ sharpness * (p - threshold)
    deltas = [_row_delta(names, Wg, g.weight, neuron, grow), _row_delta(names, Wu, up.weight, neuron, urow),
              _col_delta(names, Wd, down.weight, neuron, -out_scale * w)]
    spec = {"layer": layer, "neuron": neuron, "threshold": threshold, "up_scale": up_scale,
            "mean_proj": mean_proj, "sharpness": sharpness, "strength": strength, "source": source,
            "read": r, "write": w}
    if m is not None:
        spec["mean_dir"] = m
    return Patch(label or f"gated neuron {layer}.{neuron} (threshold {threshold:.3g}, strength {strength:g})", "gate",
                 spec, deltas)


def _rand_orth(u: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    u = u / u.norm()
    r = torch.randn(u.shape[0], generator=g)
    r = r - (r @ u) * u
    return r / r.norm()


def control(lm: LM, patch: Patch, seed: int = 0) -> Patch:
    """A random edit of the same kind and size: a random direction orthogonal to the patch's (same
    layers, neurons and strength, each matrix's change rescaled to the same norm), or the same
    number of random other neurons / heads in the same layers. If it fixes the failure as well,
    the fix is not specific to what the diagnosis found."""
    g = torch.Generator().manual_seed(seed)
    sp = patch.spec
    if patch.kind == "orthogonalise":
        layers = None if sp.get("layers") in (None, "all") else sp["layers"]
        c = orthogonalise(lm, _rand_orth(sp["direction"], g), sp["alpha"], layers=layers, embed=sp.get("embed", True),
                          source="random control")
    elif patch.kind == "project":
        sites = [Site(int(s["layer"]), s["part"], _rand_orth(u, g), s.get("cols"))
                 for s, u in zip(sp["sites"], sp["directions"])]
        c = project(lm, sites, sp["alpha"], source="random control")
    elif patch.kind in ("scale_neurons", "scale_heads"):
        key = "neurons" if patch.kind == "scale_neurons" else "heads"
        chosen = {tuple(int(x) for x in n) for n in sp[key]}
        by_layer: dict[int, int] = {}
        for i, _ in chosen:
            by_layer[i] = by_layer.get(i, 0) + 1
        picked = []
        for i, k in sorted(by_layer.items()):
            width = lm.down_proj(i).weight.shape[1] if key == "neurons" else lm.n_heads
            pool = [j for j in range(width) if (i, j) not in chosen]
            order = torch.randperm(len(pool), generator=g)[:k].tolist()
            picked += [(i, pool[j]) for j in order]
        build = scale_neurons if key == "neurons" else scale_heads
        c = build(lm, picked, sp["scale"], source="random control")
    elif patch.kind == "gate":
        c = gate(lm, int(sp["layer"]), int(sp["neuron"]), _rand_orth(sp["read"], g), sp["write"],
                 threshold=sp["threshold"], up_scale=sp.get("up_scale", 1.0), mean_dir=sp.get("mean_dir"),
                 mean_proj=sp.get("mean_proj"), sharpness=sp["sharpness"], strength=sp["strength"],
                 source="random control")
    else:
        raise ValueError(f"no random control for a {patch.kind} patch")
    if patch.kind in ("orthogonalise", "project"):
        for cd, rd in zip(c.deltas, patch.deltas):
            n = cd.norm()
            if n > 0:
                cd.U = cd.U * (rd.norm() / n)
    c.label = f"random control for: {patch.label}"
    return c


# -- applying --------------------------------------------------------------------------------------

def _bump(lm: LM) -> None:
    lm._wver = getattr(lm, "_wver", 0) + 1         # clean-pass memo keys on the weight version


@torch.no_grad()
def apply(lm: LM, patch: Patch, sign: float = 1.0) -> None:
    """Add (sign=+1) or subtract (-1) the patch in place. Prefer `applied()` for trials: it restores exactly."""
    fits(lm, patch)
    if patch.untie:
        untie(lm)
    for d in patch.deltas:
        p = _param(lm, d.name)
        delta = d.dense()
        if p.dim() == 1:
            delta = delta[:, 0]
        p.add_(sign * delta.to(p.device, p.dtype))
    _bump(lm)


@contextlib.contextmanager
def applied(lm: LM, patch: Patch) -> Iterator[None]:
    """The patch applied for the duration; the touched parameters are restored bit for bit after."""
    fits(lm, patch)
    if patch.untie:
        untie(lm)
    saved = {d.name: _param(lm, d.name).detach().to("cpu", copy=True) for d in patch.deltas}
    try:
        apply(lm, patch)
        yield
    finally:
        with torch.no_grad():
            for n, t in saved.items():
                _param(lm, n).copy_(t.to(_param(lm, n).device))
        _bump(lm)
