"""Where a failure is written: the layers, and the MLP neurons in them, that write the
failing-minus-working difference into the residual stream.

Input is a set of teacher-forced states: token ids, the positions to read, and whether they
are the failing or the working side. Two sources:

* the contrast set (failing vs passing prompts, read at the decision token), and
* matched continuations of the same prompt (pairs.py): the failing continuation vs a working
  one, read over the continuation's tokens, so only the failure differs.

For each depth the direction is mean(failing) - mean(working) of the residual. A layer's
share is how much of that difference its attention / MLP output writes along the direction
of the residual right after it; a neuron's share is its activation difference times its
down-projection column along that direction. The local edits (`edits.project`) take the
direction out of the few sites that write most of it, and in the sparse form only out of the
neurons that write most of each MLP site's share (the shape of the edit that fixed
Qwen3.5-0.8B's reasoning loops: 64 of 3,584 neurons in each of two MLPs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from ..deep.lm import LM
from . import edits as E


@dataclass
class State:
    name: str
    ids: torch.Tensor              # [1, T]
    positions: list[int]           # positions to read
    bad: bool                      # failing side


@dataclass
class SiteMap:
    source: str                                     # "contrast" | "continuations"
    dirs: dict[int, torch.Tensor]                   # depth -> unit direction (failing - working)
    sites: list[dict[str, Any]]                     # chosen, ranked: layer, part, score, share
    ranked: list[dict[str, Any]]                    # every site with a positive score, ranked
    neurons: dict[int, list[tuple[int, float]]]     # chosen MLP layer -> chosen (neuron, score)
    neuron_share: dict[int, float]                  # chosen MLP layer -> share of its score they carry
    n_bad: int = 0
    n_good: int = 0
    notes: list[str] = field(default_factory=list)
    nscore: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)   # layer -> per-neuron score

    def edit_sites(self, sparse: bool) -> list[E.Site]:
        """The chosen sites as edit targets; sparse: MLP sites restricted to their chosen neurons
        (attention sites are left out: they have no neuron mask)."""
        out = []
        for s in self.sites:
            u = self.dirs[s["layer"] + 1]
            if sparse:
                if s["part"] == "mlp" and s["layer"] in self.neurons:
                    out.append(E.Site(s["layer"], "mlp", u, [j for j, _ in self.neurons[s["layer"]]]))
            else:
                out.append(E.Site(s["layer"], s["part"], u))
        return out

    def describe(self, sparse: bool) -> str:
        parts = []
        for s in self.sites:
            if sparse and (s["part"] != "mlp" or s["layer"] not in self.neurons):
                continue
            t = f"layer {s['layer']} {'MLP' if s['part'] == 'mlp' else 'attention'} ({s['share']:.0%})"
            if sparse:
                t += (f", {len(self.neurons[s['layer']])} neurons carrying "
                      f"{self.neuron_share[s['layer']]:.0%} of it")
            parts.append(t)
        side = ("the failing continuation vs a working continuation of the same prompt"
                if self.source == "continuations" else "failing vs working prompts")
        return f"write most of the difference between {side}: " + "; ".join(parts)


def from_contrast(c: Any) -> list[State]:
    out = []
    for e in list(c.failing) + list(c.passing):
        if e.decision_ids is not None:
            out.append(State(e.name, e.decision_ids, [e.decision_ids.shape[1] - 1], e in c.failing))
    return out


def _accumulate(lm: LM, states: Sequence[State]) -> dict[str, Any]:
    L = lm.n_layers
    acc: dict[bool, dict[str, Any]] = {}
    for st in states:
        got = lm.seq_capture(st.ids, depths=range(1, L + 1), mlp_layers=range(L), attn_layers=range(L),
                             positions=st.positions)
        a = acc.setdefault(st.bad, {"n": 0, "resid": {}, "attn": {}, "mlp": {}, "act": {}})
        a["n"] += 1
        for key, src in (("resid", got["resid"]), ("attn", got["attn"]), ("mlp", got["mlp"]), ("act", got["mlp_act"])):
            for k, v in src.items():
                m = v.mean(0)                                      # mean over the state's positions
                a[key][k] = a[key].get(k, 0) + m
    return acc


def attribute(lm: LM, texts: Sequence[tuple[torch.Tensor, int, float]], layers: Optional[Sequence[int]] = None,
              first_only: bool = False) -> torch.Tensor:
    """Causal neuron scores, [n_layers, d_ff]: attribution patching (activation x gradient) of
    sum_k weight_k * mean log p(continuation_k), summed over the continuation's positions.

    texts: (ids = prefix + continuation, prefix length, weight); weight +1 for the failing
    continuation and -1 for the working one, so a neuron scores high when it makes the failure
    more likely than the working text. Unlike the projection shares of `locate`, this counts a
    neuron by its effect on the output, so a late layer that merely carries which token comes next
    does not outrank the earlier neuron that causes the repeat."""
    layers = list(range(lm.n_layers)) if layers is None else list(layers)
    d_ff = lm.down_proj(layers[0]).weight.shape[1]
    out = torch.zeros(lm.n_layers, d_ff)
    for ids, P, wgt in texts:
        acts: dict[int, torch.Tensor] = {}
        regs = [(lm.model.get_input_embeddings(), "post", lambda m, a, o: o.detach().requires_grad_(True))]
        for i in layers:
            regs.append((lm.down_proj(i), "pre", lambda m, a, i=i: acts.__setitem__(i, a[0])))
        x = ids.to(lm.device)
        with torch.enable_grad(), lm._hooks(regs):
            k = x.shape[1] - P + 1
            extra = {"logits_to_keep": k} if lm._keeps_logits else {}
            lg = lm.model(input_ids=x, attention_mask=torch.ones_like(x), **extra).logits[0][-k:].float()
            lp = lg[:1 if first_only else -1].log_softmax(-1)
            tgt = x[0, P:P + 1] if first_only else x[0, P:]
            metric = lp.gather(1, tgt[:, None]).mean() * wgt
            grads = torch.autograd.grad(metric, [acts[i] for i in layers], allow_unused=True)
        for i, g in zip(layers, grads):
            if g is not None:
                out[i] += (acts[i].detach().float() * g.float())[0, P - 1:].sum(0).cpu()
    return out


def attribute_heads(lm: LM, texts: Sequence[tuple[torch.Tensor, int, float]],
                    first_only: bool = False) -> torch.Tensor:
    """Causal head scores, [n_layers, n_heads] (0 for layers without attention heads): activation x gradient of
    each head's slice of the attention output projection's input, with the same texts and metric as
    `attribute`. A head scores high when its output makes the failing continuation more likely than the
    working one."""
    layers = [i for i in range(lm.n_layers) if lm.o_proj(i) is not None]
    heads = max((lm.o_proj(i).weight.shape[1] // lm.head_dim(i) for i in layers), default=0)
    out = torch.zeros(lm.n_layers, heads)
    for ids, P, wgt in texts:
        acts: dict[int, torch.Tensor] = {}
        regs = [(lm.model.get_input_embeddings(), "post", lambda m, a, o: o.detach().requires_grad_(True))]
        for i in layers:
            regs.append((lm.o_proj(i), "pre", lambda m, a, i=i: acts.__setitem__(i, a[0])))
        x = ids.to(lm.device)
        with torch.enable_grad(), lm._hooks(regs):
            k = x.shape[1] - P + 1
            extra = {"logits_to_keep": k} if lm._keeps_logits else {}
            lg = lm.model(input_ids=x, attention_mask=torch.ones_like(x), **extra).logits[0][-k:].float()
            lp = lg[:1 if first_only else -1].log_softmax(-1)
            tgt = x[0, P:P + 1] if first_only else x[0, P:]
            metric = lp.gather(1, tgt[:, None]).mean() * wgt
            grads = torch.autograd.grad(metric, [acts[i] for i in layers], allow_unused=True)
        for i, g in zip(layers, grads):
            if g is not None:
                hd = lm.head_dim(i)
                s = (acts[i].detach().float() * g.float())[0, P - 1:].sum(0).cpu()
                out[i, : s.shape[0] // hd] += s.reshape(-1, hd).sum(-1)
    return out


def causal_neurons(lm: LM, pairs: Sequence[Any], top: int = 64, first_only: bool = True) -> dict[str, Any]:
    """The neurons that most raise the repeat's likelihood over the working continuation, from
    matched pairs (pairs.Pair), with the layers they sit in. first_only: score the step where the
    two continuations part (the loop's decision) instead of every continuation token, most of
    which (list numbers, newlines) both share."""
    texts = []
    for p in pairs:
        for cont, w in ((p.bad, 1.0), (p.good, -1.0)):
            texts.append((torch.cat([p.prefix, torch.tensor([cont], dtype=torch.long)], 1), p.prefix.shape[1], w))
    A = attribute(lm, texts, first_only=first_only) / max(1, len(pairs))
    flat = A.flatten()
    idx = torch.argsort(flat, descending=True)[:top].tolist()
    d_ff = A.shape[1]
    ranked = [(i // d_ff, i % d_ff, float(flat[i])) for i in idx]
    by_layer = A.clamp(min=0).sum(1)
    return {"neurons": ranked, "layer_share": (by_layer / (by_layer.sum() + 1e-12)).tolist(), "scores": A}


@torch.no_grad()
def mlp_inputs(lm: LM, layer: int, ids: torch.Tensor, positions: Optional[Sequence[int]] = None) -> torch.Tensor:
    """The (normed) input the MLP of `layer` reads, [T, d] (or the given positions)."""
    got: dict[str, torch.Tensor] = {}

    def pre(m, args):
        x = args[0][0]
        got["x"] = (x if positions is None else x[list(positions)]).detach().float().cpu()
    with lm._hooks([(lm.mlp(layer), "pre", pre)]):
        lm.model(input_ids=ids.to(lm.device), attention_mask=torch.ones_like(ids).to(lm.device))
    return got["x"]


@torch.no_grad()
def quiet_neurons(lm: LM, layer: int, texts: Sequence[torch.Tensor], k: int = 8) -> list[int]:
    """The k neurons of `layer` with the smallest mean |activation x output-column norm| over the
    texts: the ones the model barely uses, safest to overwrite."""
    tot = None
    for ids in texts:
        a = lm.seq_capture(ids, mlp_layers=[layer])["mlp_act"][layer].abs().mean(0)
        tot = a if tot is None else tot + a
    w = lm.down_proj(layer).weight.detach().float().cpu().norm(dim=0)
    return torch.argsort(tot * w)[:k].tolist()


@torch.no_grad()
def gate_plan(lm: LM, layer: int, bad: Sequence[State], good: Sequence[State], silent: Sequence[torch.Tensor],
              ref: Sequence[torch.Tensor], quantile: float = 0.995, min_fire: float = 0.5,
              max_off: float = 0.02) -> dict[str, Any]:
    """What a gated neuron at `layer` should read and where it should fire (see `edits.gate`).

    read: mean(bad) - mean(good) of the MLP input. The neuron must stay off at the good states
    and at every position of the `silent` texts: its threshold is the `quantile` of the read
    projection there. When that is below 0 the gate needs no bias (threshold 0); otherwise the
    bias comes from the mean input direction over the `ref` texts, which is only approximately
    constant, so the plan is judged on the gate's real pre-activation: it must fire on at least
    `min_fire` of the failing positions and at most `max_off` of the ones that must stay
    unchanged. Returns {"ok": False, "why": ...} when no gate does."""
    def xs(states):
        return torch.cat([mlp_inputs(lm, layer, s.ids, s.positions) for s in states]) if states else None
    B, G = xs(bad), xs(good)
    if B is None or G is None:
        return {"ok": False, "why": "no failing or no working states"}
    r = B.mean(0) - G.mean(0)
    r = r / (r.norm() + 1e-12)
    X_off = torch.cat([G] + [mlp_inputs(lm, layer, t) for t in silent])
    off = X_off @ r
    q = float(torch.quantile(off, quantile))
    pb = B @ r
    out: dict[str, Any] = {"layer": layer, "read": r, "quantile_off": q}
    if q <= 0:
        thr, mean_dir, mean_proj, spread = 0.0, None, None, None
        z_bad, z_off = pb, off
    else:
        R = torch.cat([mlp_inputs(lm, layer, t) for t in ref])
        m = R.mean(0)
        m = m / (m.norm() + 1e-12)
        proj = R @ m
        mean_proj = float(proj.mean())
        spread = float(proj.std() / (abs(mean_proj) + 1e-12))
        if mean_proj <= 0:
            return {**out, "ok": False, "why": "the input has no positive mean direction to build a bias from"}
        thr, mean_dir = q, m
        # The gate's real pre-activation, with the approximate bias: p - thr * (x.m) / mean_proj.
        z_bad = pb - thr * (B @ m) / mean_proj
        z_off = off - thr * (X_off @ m) / mean_proj
    fire = z_bad > 0
    fire_bad, fire_off = float(fire.float().mean()), float((z_off > 0).float().mean())
    out.update(threshold=thr, mean_dir=mean_dir, mean_proj=mean_proj, spread=spread, fire_bad=fire_bad,
               fire_off=fire_off)
    if fire_bad < min_fire:
        return {**out, "ok": False, "why": f"it fires on only {fire_bad:.0%} of the failing positions"}
    if fire_off > max_off:
        return {**out, "ok": False, "why": f"it would fire on {fire_off:.1%} of the positions that must stay "
                                           "unchanged" + (f" (the bias is approximate: spread {spread:.0%})"
                                                          if spread is not None else "")}
    return {**out, "ok": True, "up_scale": float(pb[fire].mean()) if thr == 0 else 1.0,
            "margin": float(z_bad[fire].mean())}


@torch.no_grad()
def locate(lm: LM, states: Sequence[State], *, source: str = "contrast", max_sites: int = 3, cover: float = 0.6,
           neuron_cover: float = 0.9, max_neuron_frac: float = 1 / 32) -> Optional[SiteMap]:
    """The sites (and neurons) writing most of the failing-minus-working difference, or None
    when there are not states on both sides."""
    if not any(s.bad for s in states) or all(s.bad for s in states):
        return None
    acc = _accumulate(lm, states)
    B, G = acc[True], acc[False]
    mean = lambda a, key, k: a[key][k] / a["n"]                    # noqa: E731
    L = lm.n_layers
    dirs: dict[int, torch.Tensor] = {}
    for depth in range(1, L + 1):
        v = mean(B, "resid", depth) - mean(G, "resid", depth)
        dirs[depth] = v / (v.norm() + 1e-12)
    scores = []
    nscore: dict[int, torch.Tensor] = {}
    for i in range(L):
        u = dirs[i + 1]
        for part in ("attn", "mlp"):
            sc = float((mean(B, part, i) - mean(G, part, i)) @ u)
            scores.append({"layer": i, "part": part, "score": sc})
        W = lm.down_proj(i).weight.detach().float().cpu()
        nscore[i] = (mean(B, "act", i) - mean(G, "act", i)) * (u @ W)
    pos = sorted((s for s in scores if s["score"] > 0), key=lambda s: -s["score"])
    total = sum(s["score"] for s in pos)
    if not pos or total <= 0:
        return None
    for s in pos:
        s["share"] = s["score"] / total
    chosen, got = [], 0.0
    for s in pos[:max_sites]:
        chosen.append(s)
        got += s["share"]
        if got >= cover:
            break
    neurons: dict[int, list[tuple[int, float]]] = {}
    nshare: dict[int, float] = {}
    for s in chosen:
        if s["part"] != "mlp":
            continue
        ns = nscore[s["layer"]]
        tot = float(ns.clamp(min=0).sum())
        if tot <= 0:
            continue
        cap = max(8, int(ns.numel() * max_neuron_frac))
        order = torch.argsort(ns, descending=True)
        picked, run = [], 0.0
        for j in order[:cap].tolist():
            if float(ns[j]) <= 0:
                break
            picked.append((j, float(ns[j])))
            run += float(ns[j])
            if run >= neuron_cover * tot:
                break
        neurons[s["layer"]] = picked
        nshare[s["layer"]] = run / tot
    return SiteMap(source, dirs, chosen, pos, neurons, nshare, n_bad=B["n"], n_good=G["n"], nscore=nscore)
