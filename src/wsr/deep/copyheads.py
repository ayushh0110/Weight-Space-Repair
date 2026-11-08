"""Loops: which attention heads copy the repeated item forward (a parallel probe to the contrast set).

Repetition in transformers runs through induction-style copy heads (Olsson et al.
2022): at the start of the next item, a head attends back to an earlier copy and
writes that copy's token forward. Two measurements, neither needing passing prompts:

* calibrated once per model: each head's induction score on a random token
  sequence repeated twice (attention from each token of the second copy to the
  token after its earlier occurrence), cached under ~/.cache/wsr/copyheads/;
* at the loop's decision point: each head's attention mass on earlier positions
  holding the token it is about to repeat, times its direct effect on the
  failure metric (the logit difference falls by this much when the head's
  output is taken out of the final residual).

Causal test: remove the top heads (zero their output at every position) and
see whether the decision flips and the generated list stops looping, against
removing the same number of random heads (the null), and whether lists that did
not loop still come out fine.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .lm import LM

CACHE = Path(os.environ.get("WSR_CACHE", Path.home() / ".cache" / "wsr")) / "copyheads"


def induction_scores(lm: LM, seq_len: int = 40, n_seqs: int = 4, seed: int = 0, cache: bool = True) -> torch.Tensor:
    """[n_layers, n_heads] mean attention to the induction target on repeated random tokens."""
    from .langprobe import _key

    path = CACHE / f"{lm.name}-{_key(lm)}-L{seq_len}.pt"
    if cache and path.exists():
        return torch.load(path)
    g = torch.Generator().manual_seed(seed)
    V = lm.unembed.weight.shape[0]
    lo, hi = min(1000, V // 4), min(V - 1, 30000)
    total = None
    for _ in range(n_seqs):
        r = torch.randint(lo, hi, (seq_len,), generator=g)
        ids = torch.cat([r, r])[None]
        ids = ids.to(lm.device)
        with lm.eager_attention(), torch.no_grad():
            out = lm.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_attentions=True)
        A = torch.stack([a[0].float().cpu() for a in lm.attentions_by_layer(out.attentions)])   # [L, H, T, T]
        q = torch.arange(seq_len + 1, 2 * seq_len)                          # second copy (from its 2nd token)
        k = q - seq_len + 1                                                 # token after the earlier occurrence
        s = A[:, :, q, k].mean(-1)
        total = s if total is None else total + s
    sc = total / n_seqs
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sc, path)
    return sc


@dataclass
class CopyHeads:
    heads: list[dict]          # ranked: layer, head, copy_attention, direct_effect, induction, score
    copy_positions: int
    token: str


def copy_heads(lm: LM, target, induction: torch.Tensor, top: int = 8) -> CopyHeads:
    """Rank heads at a loop's decision point. target: analysis.Target (failing example + metric)."""
    ex = target.ex
    ids = ex.decision_ids
    tok = int(ex.decision_tok)
    seq = ids[0].tolist()
    copies = [j for j, t in enumerate(seq[:-1]) if t == tok]
    att = lm.attention(ids)                                               # [L, H, T]
    mass = att[:, :, copies].sum(-1) if copies else torch.zeros(att.shape[:2])
    L, H = lm.n_layers, lm.n_heads
    r = lm.run(ids, capture=("resid",), heads_at=list(lm.head_layers))
    final = r.resid[-1]
    m0 = target.metric(lm.logits_of(final))
    writes = []
    for i in range(L):
        if i not in lm.head_layers:                                       # linear-attention layer: no heads
            writes.extend(torch.zeros(lm.d_model) for _ in range(H))
            continue
        W = lm.o_proj(i).weight.detach().float().cpu()                    # [d, H*hd]
        hd = W.shape[1] // H
        for h in range(H):
            writes.append(W[:, h * hd:(h + 1) * hd] @ r.heads[i][h])
    Wr = torch.stack(writes)                                              # [L*H, d]
    lg = lm.logits_of(final[None] - Wr)                                   # head's output taken out
    effect = torch.tensor([m0 - target.metric(row) for row in lg]).view(L, H)
    score = mass * effect.clamp(min=0)
    ok = [i * H + h for i in lm.head_layers for h in range(H)]
    order = sorted(ok, key=lambda k: -float(score.flatten()[k]))[:top]
    heads = [{"layer": i // H, "head": i % H, "copy_attention": round(float(mass.flatten()[i]), 3),
              "direct_effect": round(float(effect.flatten()[i]), 3),
              "induction": round(float(induction.flatten()[i]), 3), "score": round(float(score.flatten()[i]), 3)}
             for i in order]
    return CopyHeads(heads, len(copies), lm.token_str(tok))


def ablation_tests(lm: LM, c, ts: list, ch: CopyHeads, clf, ks=(1, 2, 4, 8), n_null: int = 3,
                   seed: int = 0) -> dict[str, Any]:
    """Remove the top-k copy heads: does the decision flip, does generation stop looping, vs random heads."""
    rng = random.Random(seed)
    H = lm.n_heads
    ranked = [(h["layer"], h["head"]) for h in ch.heads]
    out: dict[str, Any] = {"by_k": []}

    base = {t.ex.name: t.metric(lm.run(t.ex.decision_ids, capture=()).logits) for t in ts}

    def effect(heads) -> tuple[float, int]:
        """(mean share of the failure's logit lead removed, prompts where the failure token loses)."""
        drop, fl = [], 0
        for t in ts:
            m = t.metric(lm.run(t.ex.decision_ids, capture=(), zero_heads=heads).logits)
            m0 = base[t.ex.name]
            drop.append((m0 - m) / (abs(m0) + 1e-6))
            fl += m < 0
        return sum(drop) / len(drop), fl

    best = None
    for k in ks:
        hs = ranked[:k]
        if len(hs) < k:
            break
        d, f = effect(hs)
        pool = [(i, h) for i in lm.head_layers for h in range(H) if (i, h) not in ranked]
        nulls = [effect(rng.sample(pool, k)) for _ in range(n_null)]
        nd = max(x[0] for x in nulls)
        row = {"k": k, "removed": round(d, 3), "flipped": f, "of": len(ts),
               "null_removed_max": round(nd, 3), "null_flipped": [x[1] for x in nulls]}
        row["specific"] = d >= nd + 0.2 and d > 0
        out["by_k"].append(row)
        if row["specific"] and (best is None or d > best["removed"]):
            best = row
    chosen = best["k"] if best else None
    out["chosen_k"] = chosen
    if chosen is None:
        return out
    hs = ranked[:chosen]
    out["heads"] = hs

    def gen(examples, heads, tag):
        res = []
        runs = clf.run_many([(ex.name + tag, ex.messages) for ex in examples],
                            hooks=lambda: lm.steer(zero_heads=heads))
        for ex, e2 in zip(examples, runs):
            res.append({"example": ex.name, "loops": "loop" in e2.groups, "groups": sorted(e2.groups),
                        "tail": e2.text[-100:]})
        return res

    fl, pl = c.failing[:3], c.passing[:3]
    g_f = gen(fl, hs, "+no_copy_heads")
    g_p = gen(pl, hs, "+no_copy_heads")
    pool = [(i, h) for i in lm.head_layers for h in range(H) if (i, h) not in ranked]
    g_n = gen(fl, rng.sample(pool, chosen), "+no_random_heads")
    out.update(gen_fixed=sum(not r["loops"] for r in g_f), gen_total=len(g_f),
               passing_kept=sum(not r["groups"] for r in g_p), passing_total=len(g_p),
               null_gen_fixed=sum(not r["loops"] for r in g_n), samples=g_f + g_p + g_n)
    return out


def summary(ch: CopyHeads, ab: dict, induction: torch.Tensor) -> str:
    h0 = ch.heads[0] if ch.heads else None
    if not h0:
        return "no head attends to an earlier copy"
    thr = float(induction.flatten().quantile(0.95))
    ind = [f"{h['layer']}.{h['head']}" for h in ch.heads[:4] if h["induction"] >= thr]
    s = (f"at the repeat, head {h0['layer']}.{h0['head']} puts {h0['copy_attention']:.0%} of its attention on "
         f"earlier copies of {ch.token.strip()!r} and its output carries {h0['direct_effect']:+.2f} of the logit "
         f"difference")
    if ind:
        s += f"; {', '.join(ind)} {'is an induction head' if len(ind) == 1 else 'are induction heads'} (calibrated)"
    k = ab.get("chosen_k")
    row = next((r for r in ab.get("by_k", []) if r["k"] == k), None)
    if k and row:
        s += (f"; removing the top {k} copy head(s) takes away {row['removed']:.0%} of the failure's logit lead "
              f"(random heads at most {row['null_removed_max']:.0%}) and stops the loop in "
              f"{ab['gen_fixed']}/{ab['gen_total']} on generation (random heads: {ab['null_gen_fixed']}/"
              f"{ab['gen_total']}), keeping {ab['passing_kept']}/{ab['passing_total']} working lists fine")
    elif ab.get("by_k"):
        r = max(ab["by_k"], key=lambda r: r["removed"])
        s += (f"; removing them takes away at most {r['removed']:.0%} of the failure's logit lead (top {r['k']}; "
              f"random heads {r['null_removed_max']:.0%}): copying is not where this loop is decided")
    return s


def as_evidence(ch: CopyHeads, ab: dict, induction: torch.Tensor) -> dict[str, Any]:
    flat = induction.flatten()
    top_ind = torch.argsort(flat, descending=True)[:5].tolist()
    H = induction.shape[1]
    return {"copy_heads": ch.heads, "copy_positions": ch.copy_positions, "token": ch.token, "ablation": ab,
            "top_induction_heads": [{"layer": i // H, "head": i % H, "induction": round(float(flat[i]), 3)}
                                    for i in top_ind]}

