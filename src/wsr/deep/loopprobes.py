"""Loops: a battery of mechanism probes, run side by side on the same failing loop.

A repetition loop has several distinct root causes, and each needs its own test.
All of them run on every loop failure (none gates another) and each reports a
verdict with its own control, so the report can say which mechanisms the
evidence supports:

1. attractor      - the residual stream collapses onto a cycle: participation
                    ratio (effective rank) of the states inside the loop vs before
                    it and vs a working run; cosine of states one cycle apart; and
                    the defining test, escape: force the alternative token at the
                    repeat and see whether generation falls back into the loop.
2. mlp            - MLPs, not attention, write the repeated tokens: direct logit
                    attribution along every position of the cycle (attention vs
                    MLP vs embeddings), the neurons that fire on token X and write
                    X, and removing them against random neurons of the same layers.
3. copy heads     - copyheads.py: induction-style heads copying the earlier copy.
4. positional     - RoPE scores depend only on relative position, so shifting
                    every position id by D changes nothing in exact arithmetic;
                    a failure that moves with D is positional (aliasing, lost
                    precision at large angles, or a context past the trained length).
5. decoding       - the loop under greedy decoding vs sampled decoding and a mild
                    repetition penalty: a loop that only greedy decoding keeps is a
                    runtime cause, not a weight-level one.
Quantisation (scale collapse in v_proj / down_proj) is `wsr quant` and
`wsr deep --gguf`; it needs the GGUF file, so it is reported as checked or not.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .contrast import Example, list_items
from .lm import LM


# -- where the loop is ------------------------------------------------------------------------

@dataclass
class LoopSpan:
    ex: Example
    ids: torch.Tensor          # prompt + reply body, [1, T]
    prompt_len: int
    onset: int                 # reply-token index where the first copy of the repeating block starts
    second: int                # reply-token index where its second copy starts
    period: int                # tokens per cycle


def loop_span(lm: LM, ex: Example) -> Optional[LoopSpan]:
    body = ex.gen[:-1] if ex.gen and ex.gen[-1] in lm.eot_ids else list(ex.gen)
    m = ex.meta
    if "loop_item" in m:
        items = list_items(lm, ex.gen)
        q, p = m["loop_item"], m["loop_period"]
        if q - p < 0 or q >= len(items):
            return None
        onset, second = items[q - p][0], items[q][0]
    elif "loop_onset" in m:
        onset, second = m["loop_onset"], m["loop_onset"] + m["loop_period"]
    else:
        return None
    if second <= onset or second >= len(body):
        return None
    ids = torch.cat([ex.prompt_ids, torch.tensor([body], dtype=torch.long)], dim=1)
    return LoopSpan(ex, ids, ex.prompt_ids.shape[1], onset, second, second - onset)


def _pr(X: torch.Tensor) -> float:
    """Participation ratio of the centred states: (sum s^2)^2 / sum s^4 (1 = rank one)."""
    if X.shape[0] < 3:
        return float("nan")
    s = torch.linalg.svdvals(X - X.mean(0)) ** 2
    return float(s.sum() ** 2 / (s.pow(2).sum() + 1e-12))


def _cos_lag(X: torch.Tensor, lag: int) -> float:
    if X.shape[0] <= lag:
        return float("nan")
    a, b = X[lag:], X[:-lag]
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


# -- 1. attractor --------------------------------------------------------------------------------

def attractor(lm: LM, spans: list[LoopSpan], passing: list[Example], clf, window: int = 24) -> dict[str, Any]:
    L = lm.n_layers
    depths = sorted({max(1, L // 4), L // 2, (3 * L) // 4, L})
    sp = spans[0]
    cap = lm.seq_capture(sp.ids, depths=depths)["resid"]
    P = sp.prompt_len
    loop_lo, loop_hi = P + sp.onset, sp.ids.shape[1]
    pre_lo = max(P, loop_lo - window)
    ref = None
    if passing:
        pe = passing[0]
        pb = pe.gen[:-1] if pe.gen and pe.gen[-1] in lm.eot_ids else list(pe.gen)
        if pb:
            pids = torch.cat([pe.prompt_ids, torch.tensor([pb], dtype=torch.long)], dim=1)
            ref = (lm.seq_capture(pids, depths=depths)["resid"], pe.prompt_ids.shape[1], pids.shape[1])
    rows = []
    for d in depths:
        X = cap[d]
        loop = X[max(loop_lo, loop_hi - window):loop_hi]
        pre = X[pre_lo:loop_lo]
        row = {"layer": d - 1, "pr_loop": _pr(loop), "pr_before": _pr(pre),
               "cos_cycle_loop": _cos_lag(X[loop_lo:loop_hi], sp.period),
               "cos_cycle_before": _cos_lag(X[P:loop_lo], sp.period)}
        if ref is not None:
            R, rp, rT = ref
            seg = R[d][max(rp, min(rT, loop_hi) - window):min(rT, loop_hi)]
            row["pr_working"] = _pr(seg)
            row["cos_cycle_working"] = _cos_lag(R[d][rp:rT], sp.period)
        # contraction: are successive cycles getting closer?
        cyc = [X[loop_lo + k * sp.period: loop_lo + (k + 1) * sp.period] for k in range((loop_hi - loop_lo) // sp.period)]
        dist = [float((cyc[k + 1] - cyc[k]).norm() / (cyc[k].norm() + 1e-9)) for k in range(len(cyc) - 1)]
        row["cycle_distance"] = [round(x, 3) for x in dist]
        row["contracting"] = len(dist) >= 2 and dist[-1] < 0.7 * dist[0]
        rows.append(row)
    # escape: force the alternative at the repeat, keep generating; does it fall back in?
    esc = []
    for s in spans[:3]:
        alt = s.ex.meta.get("alt_token")
        if alt is None:
            continue
        body = s.ids[0, s.prompt_len:].tolist()
        prefix = torch.cat([s.ex.prompt_ids, torch.tensor([body[:s.second] + [alt]], dtype=torch.long)], dim=1)
        cont = lm.generate(prefix, max(48, 3 * s.period))
        text = lm.tok.decode(body[:s.second] + [alt] + [t for t in cont if t not in lm.eot_ids],
                             skip_special_tokens=True)
        groups = clf.groups_of(text, [alt] + cont)
        rep = s.ex.meta.get("repeated")
        back = "loop" in groups or (rep is not None and lm.tok.decode(cont).lower().count(rep[:20]) >= 2)
        esc.append({"example": s.ex.name, "returns_to_loop": bool(back), "tail": text[-100:]})
    collapsed = [r for r in rows if r["cos_cycle_loop"] >= 0.9]
    returns = sum(e["returns_to_loop"] for e in esc)
    # An attractor: pushed out of the loop, generation falls back in (>= 2/3), and the states repeat.
    supported = bool(esc) and returns * 3 >= 2 * len(esc) and bool(collapsed)
    return {"by_layer": rows, "escape": esc, "escape_returns": returns, "escape_total": len(esc),
            "collapse_from_layer": collapsed[0]["layer"] if collapsed else None, "supported": supported}


# -- 2. MLP self-reinforcement --------------------------------------------------------------------

def _loop_positions(sp: LoopSpan) -> tuple[list[int], list[int]]:
    """Absolute positions inside the loop and the token each one predicts."""
    T = sp.ids.shape[1]
    a0 = sp.prompt_len + sp.onset - 1
    pos = list(range(max(a0, 0), T - 1))
    return pos, [int(sp.ids[0, a + 1]) for a in pos]


def _metric_grads(lm: LM, finals: torch.Tensor, ys: list[int]) -> torch.Tensor:
    """d(logit[y] - mean logit)/d(final residual) at each position (exact through the final norm)."""
    x = finals.clone().float().requires_grad_(True)
    with torch.enable_grad():
        lg = lm.logits_of(x)
        m = lg[torch.arange(len(ys)), torch.tensor(ys)] - lg.mean(-1)
        m.sum().backward()
    return x.grad.detach().float()


def mlp_reinforcement(lm: LM, spans: list[LoopSpan], ts: list, c, clf, k_neurons=(4, 16), n_null: int = 3,
                      seed: int = 0) -> dict[str, Any]:
    L = lm.n_layers
    sp = spans[0]
    pos, ys = _loop_positions(sp)
    if len(pos) < 2:
        return {"supported": False, "note": "loop too short"}
    cap = lm.seq_capture(sp.ids, depths=[0, L], mlp_layers=list(range(L)), attn_layers=list(range(L)))
    U = _metric_grads(lm, cap["resid"][L][pos], ys)                       # [n, d]
    emb = float((U * cap["resid"][0][pos]).sum())
    attn = [float((U * cap["attn"][i][pos]).sum()) for i in range(L)]
    mlp = [float((U * cap["mlp"][i][pos]).sum()) for i in range(L)]
    pos_tot = max(emb, 0) + sum(max(x, 0) for x in attn) + sum(max(x, 0) for x in mlp) or 1.0
    share = {"embeddings": max(emb, 0) / pos_tot, "attention": sum(max(x, 0) for x in attn) / pos_tot,
             "mlp": sum(max(x, 0) for x in mlp) / pos_tot}
    # Neurons by attribution patching at the decision point(s): activation x gradient of the failure's
    # lead, summed over every position and through the whole network. A neuron that writes at an
    # earlier copy and reaches the decision through attention counts (direct attribution along the
    # cycle missed exactly that: a planted "Penguin -> Penguin" neuron in layer 18 of Qwen2.5-0.5B).
    attr = None
    for t in ts[:3]:
        fail, alt = list(t.fail_ids), int(t.alt)

        def metric(lg, fail=fail, alt=alt):
            return torch.logsumexp(lg[fail], 0) - lg[alt]
        a = lm.neuron_attribution(t.ex.decision_ids, metric)
        attr = a if attr is None else attr + a
    flat = attr.flatten()
    d_ff = attr.shape[1]
    neurons = []
    for idx in torch.argsort(flat, descending=True)[:8].tolist():
        i, j = idx // d_ff, idx % d_ff
        W = lm.down_proj(i).weight.detach().float().cpu()
        acts = cap["mlp_act"][i][:, j]
        hot = torch.argsort(acts, descending=True)[:5].tolist()
        fires_on = [lm.tok.decode([int(sp.ids[0, a])]) for a in hot]
        promotes = torch.argsort(lm.logits_of(W[:, j]), descending=True)[:3].tolist()
        neurons.append({"layer": i, "neuron": j, "contribution": round(float(flat[idx]), 3),
                        "fires_on": fires_on, "promotes": [lm.tok.decode([t]) for t in promotes],
                        "self_reinforcing": any(lm.tok.decode([t]).strip().lower() in
                                                {f.strip().lower() for f in fires_on} for t in promotes)})
    tot = float(attr.clamp(min=0).sum()) or 1.0
    for n in neurons:
        n["share_of_positive_attribution"] = round(n["contribution"] / tot, 3)
    neurons.sort(key=lambda n: -n["contribution"])
    out: dict[str, Any] = {"share": {k: round(v, 3) for k, v in share.items()},
                           "attn_by_layer": [round(x, 3) for x in attn], "mlp_by_layer": [round(x, 3) for x in mlp],
                           "top_neurons": neurons[:8], "positions": len(pos)}
    # causal: remove the top neurons at every position, vs random neurons of the same layers
    rng = random.Random(seed)
    ranked = [(n["layer"], n["neuron"]) for n in neurons]
    base = {t.ex.name: t.metric(lm.run(t.ex.decision_ids, capture=()).logits) for t in ts}

    def removed(zs) -> float:
        dr = [(base[t.ex.name] - t.metric(lm.run(t.ex.decision_ids, capture=(), zero_neurons=zs).logits)) /
              (abs(base[t.ex.name]) + 1e-6) for t in ts]
        return sum(dr) / len(dr)

    d_ff = lm.down_proj(0).weight.shape[1]
    best = None
    out["by_k"] = []
    for k in k_neurons:
        zs = ranked[:k]
        if len(zs) < k:
            zs = ranked
        r = removed(zs)
        nulls = [removed([(i, rng.randrange(d_ff)) for i, _ in zs]) for _ in range(n_null)]
        row = {"k": len(zs), "removed": round(r, 3), "null_removed_max": round(max(nulls), 3),
               "specific": r >= max(nulls) + 0.2 and r > 0}
        out["by_k"].append(row)
        if row["specific"] and (best is None or r > best["removed"]):
            best = row
    out["chosen_k"] = best["k"] if best else None
    if best:
        zs = ranked[:best["k"]]
        jf = [(e.name + "+no_neurons", e.messages) for e in c.failing[:3]]
        jp = [(e.name + "+no_neurons", e.messages) for e in c.passing[:3]]
        fixed = sum("loop" not in e.groups for e in clf.run_many(jf, hooks=lambda: lm.steer(zero_neurons=zs)))
        nz = [(i, rng.randrange(d_ff)) for i, _ in zs]
        null_fixed = sum("loop" not in e.groups for e in clf.run_many(jf, hooks=lambda: lm.steer(zero_neurons=nz)))
        kept = sum(not e.groups for e in clf.run_many(jp, hooks=lambda: lm.steer(zero_neurons=zs)))
        out.update(gen_fixed=fixed, gen_total=min(3, len(c.failing)), null_gen_fixed=null_fixed,
                   passing_kept=kept, passing_total=min(3, len(c.passing)))
    causal = best is not None and out.get("gen_fixed", 0) > out.get("null_gen_fixed", 0)
    # Self-reinforcement proper: the deciding neurons fire on the repeated token and promote it.
    # Neurons that only tip a near-tie (Qwen2.5-0.5B animal list: top neuron promotes "sea") are
    # causal but are not this mechanism.
    selfr = any(n["self_reinforcing"] for n in neurons[:3])
    out["decides_tie"] = bool(causal and not selfr)
    out["supported"] = bool(causal and selfr)
    return out


# -- 4. positional ---------------------------------------------------------------------------------

def positional(lm: LM, ts: list, offsets=(1001, 4099, 16411)) -> dict[str, Any]:
    """Odd offsets on purpose: a defect that aliases positions in steps of 2^k is invisible to
    offsets that are multiples of the step."""
    cfg = lm.cfg
    max_pos = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    theta = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
    T = max(t.ex.decision_ids.shape[1] for t in ts)
    offs = [o for o in offsets if not max_pos or o + T < max_pos]
    if max_pos and max_pos - T - 1 > (offs[-1] if offs else 0):
        offs.append(max_pos - T - 1)
    base = {t.ex.name: lm.run(t.ex.decision_ids, capture=()).logits for t in ts}
    rows = []
    for o in offs:
        drift, flips, kl = [], 0, []
        for t in ts:
            lg = lm.run(t.ex.decision_ids, capture=(), pos_offset=o).logits
            m0, m = t.metric(base[t.ex.name]), t.metric(lg)
            drift.append(abs(m - m0) / (abs(m0) + 1e-6))
            flips += (m < 0) != (m0 < 0)
            lp0, lp = base[t.ex.name].log_softmax(-1), lg.log_softmax(-1)
            kl.append(float((lp0.exp() * (lp0 - lp)).sum()))
        rows.append({"offset": o, "drift": round(sum(drift) / len(drift), 3), "flipped": flips, "of": len(ts),
                     "kl": round(sum(kl) / len(kl), 4)})
    past = bool(max_pos) and T > max_pos
    worst = max(rows, key=lambda r: r["drift"]) if rows else None
    # Two questions. A defect: a pure shift changes the next-token distribution at all (healthy
    # Qwen2.5 in fp32: KL exactly 0; with positions aliased to multiples of 8: KL 0.56-1.43).
    # A cause of THIS failure: the shift moves the failure's own lead by half or flips it.
    defect = bool(rows) and max(r["kl"] for r in rows) >= 0.1
    causes = bool(worst) and (worst["flipped"] > 0 or worst["drift"] >= 0.5)
    return {"rope_theta": theta, "max_position_embeddings": max_pos, "decision_position": T,
            "past_trained_length": past, "by_offset": rows, "positional_defect": defect,
            "shift_sensitive": causes, "supported": causes or past,
            "dtype": str(next(lm.model.parameters()).dtype)}


# -- 5. decoding ------------------------------------------------------------------------------------

def decoding(lm: LM, c, clf, n: int = 3, seed: int = 0) -> dict[str, Any]:
    ex = next((e for e in c.failing if e.name == "original"), c.failing[0])
    res = {}
    for name, kw in (("sampled_t0.7", {"do_sample": True, "temperature": 0.7, "top_p": 0.9}),
                     ("penalty_1.1", {"repetition_penalty": 1.1})):
        loops = 0
        runs = n if kw.get("do_sample") else 1
        for r in range(runs):
            torch.manual_seed(seed + r)
            ids = lm.encode(ex.messages, **clf.template_kwargs)
            gen = lm.generate(ids, clf.max_new, **kw)
            body = [t for t in gen if t not in lm.eot_ids]
            loops += "loop" in clf.groups_of(lm.tok.decode(body, skip_special_tokens=True), gen,
                                             request=None)
        res[name] = {"loops": loops, "of": runs}
    # Supported when sampling escapes every time: the loop lives in greedy argmax, not in the weights.
    escapes = res["sampled_t0.7"]["loops"] == 0
    return {**res, "sampling_escapes": escapes, "penalty_escapes": res["penalty_1.1"]["loops"] == 0,
            "supported": escapes}


# -- the battery ----------------------------------------------------------------------------------

def run_all(lm: LM, c, ts: list, clf, copy: Optional[dict] = None, quant: Optional[list] = None) -> dict[str, Any]:
    spans = [s for s in (loop_span(lm, t.ex) for t in ts) if s]
    for t in ts:
        t.ex.meta["alt_token"] = t.alt
    out: dict[str, Any] = {"spans": len(spans)}
    if spans:
        out["attractor"] = attractor(lm, spans, c.passing, clf)
        out["mlp"] = mlp_reinforcement(lm, spans, ts, c, clf)
    out["positional"] = positional(lm, ts)
    out["decoding"] = decoding(lm, c, clf)
    if copy is not None:
        out["copy_heads"] = {"supported": bool(copy.get("specific")), "summary": copy.get("summary")}
    out["quantisation"] = ({"checked": True, "damaged": [q["tensor"] for q in quant if q.get("ratio_to_expected", 0) >= 3]}
                           if quant is not None else {"checked": False})
    return out


def summary_rows(b: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """(mechanism, supported, one line of evidence) for each probe that ran."""
    rows = []
    a = b.get("attractor")
    if a:
        best = min(a["by_layer"], key=lambda r: r["pr_loop"] / ((r.get("pr_working") or r["pr_before"] or 1e9)))
        ref = best.get("pr_working") or best["pr_before"]
        rows.append(("attractor", a["supported"],
                     f"forcing the alternative at the repeat: back in the loop {a['escape_returns']}/{a['escape_total']}; "
                     f"after layer {best['layer']} the loop's states have effective rank {best['pr_loop']:.1f} "
                     f"(working run {ref:.1f}), cosine one cycle apart {best['cos_cycle_loop']:.2f}"))
    m = b.get("mlp")
    if m and "share" in m:
        n0 = m["top_neurons"][0] if m["top_neurons"] else None
        line = (f"along the cycle the repeated tokens' logits come {m['share']['mlp']:.0%} from MLPs, "
                f"{m['share']['attention']:.0%} from attention, {m['share']['embeddings']:.0%} from embeddings")
        if n0:
            line += (f"; top neuron layer {n0['layer']} #{n0['neuron']} fires on {n0['fires_on'][0].strip()!r}, "
                     f"promotes {', '.join(repr(p.strip()) for p in n0['promotes'])}")
        if m.get("chosen_k"):
            line += (f"; removing the top {m['chosen_k']} neurons stops the loop in {m['gen_fixed']}/{m['gen_total']} "
                     f"(random neurons {m['null_gen_fixed']}/{m['gen_total']})")
            if m.get("decides_tie"):
                line += ("; they tip a near-tie but do not fire on and promote the repeated token: causal, "
                         "not self-reinforcement")
        else:
            br = max(m.get("by_k", [{"removed": 0, "null_removed_max": 0, "k": 0}]), key=lambda r: r["removed"])
            line += (f"; removing the top {br['k']} neurons takes away {br['removed']:.0%} of the lead "
                     f"(random {br['null_removed_max']:.0%})")
        rows.append(("MLP self-reinforcement", m["supported"], line))
    ch = b.get("copy_heads")
    if ch:
        rows.append(("copy heads", ch["supported"], ch.get("summary") or ""))
    p = b.get("positional")
    if p:
        w = max(p["by_offset"], key=lambda r: r["drift"]) if p["by_offset"] else None
        line = (f"decision at position {p['decision_position']} of {p['max_position_embeddings'] or '?'} trained; ")
        if w:
            kl = max(r["kl"] for r in p["by_offset"])
            line += (f"shifting all positions by up to {w['offset']} moves the failure's lead by {w['drift']:.0%} "
                     f"(flips {w['flipped']}/{w['of']}); next-token KL under the shift {kl:.3f}")
            if p.get("positional_defect"):
                line += (" - a positional defect is present (correct RoPE is shift-invariant)" +
                         ("" if p["supported"] else ", but it does not decide this failure"))
        else:
            line += "no shift tested"
        rows.append(("positional (RoPE)", p["supported"], line))
    d = b.get("decoding")
    if d:
        rows.append(("greedy decoding", d["supported"],
                     f"greedy loops; sampled (T 0.7) loops {d['sampled_t0.7']['loops']}/{d['sampled_t0.7']['of']}; "
                     f"with a 1.1 repetition penalty it loops {d['penalty_1.1']['loops']}/{d['penalty_1.1']['of']}"))
    q = b.get("quantisation")
    if q:
        rows.append(("quantisation", bool(q.get("damaged")),
                     ("damaged tensors: " + ", ".join(q["damaged"][:4])) if q.get("damaged") else
                     ("no tensor beyond its normal error" if q.get("checked") else
                      "not checked here: pass --gguf with the served file")))
    return rows
