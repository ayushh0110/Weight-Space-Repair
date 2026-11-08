"""Steps 3-7: token trace, direction finding, localisation, known directions, weight reading.

All measurements use one metric per failing example: the logit difference
between the token the model chose at the decision point (the failure) and the
best alternative it considered (the highest-ranked token outside the failure
class; for a blank reply, the best non-end-of-turn token). The failure is
"restored" when an intervention moves that difference to where the passing
prompts have it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .contrast import Contrast, Example
from .lm import LM

# -- tokens and metric ----------------------------------------------------------------


def failure_class(lm: LM, ex: Example, kind: str) -> set[int]:
    return set(lm.eot_ids) if kind == "blank" else {int(ex.decision_tok)}


def alt_token(lm: LM, logits: torch.Tensor, avoid: set[int]) -> int:
    order = torch.argsort(logits, descending=True)
    for t in order[: len(avoid) + 8].tolist():
        if t not in avoid:
            return int(t)
    return int(order[0])


@dataclass
class Target:
    """What 'the failure' means numerically for one failing example."""
    ex: Example
    fail_ids: list[int]
    alt: int

    def metric(self, logits: torch.Tensor) -> float:
        f = torch.logsumexp(logits[self.fail_ids], 0) if len(self.fail_ids) > 1 else logits[self.fail_ids[0]]
        return float(f - logits[self.alt])


def targets(lm: LM, c: Contrast) -> list[Target]:
    """The metric for each failing example: failure tokens vs what working runs say there.

    Failure class: end-of-turn tokens for blank replies; otherwise every token a failing run
    chose at its decision point (a refusal opens with "I" or "As" or "Sorry"). Alternative: the
    token passing runs chose there (a compliant "To"), else the top token outside the class.
    Using the model's own runner-up instead ('I' vs 'Sorry' on Qwen2.5-1.5B) compares two
    refusals and measures nothing.
    """
    from collections import Counter

    shared: set[int] = set()
    if c.kind != "blank":
        shared = {int(e.decision_tok) for e in c.failing if e.decision_tok is not None}
    pass_toks = Counter(int(e.decision_tok) for e in c.passing if e.decision_tok is not None)
    out = []
    for ex in c.failing:
        if ex.decision_tok is None:
            continue
        cls = failure_class(lm, ex, c.kind) | shared
        r = lm.run(ex.decision_ids, capture=())
        alt = next((t for t, _ in pass_toks.most_common() if t not in cls), None)
        out.append(Target(ex, sorted(cls), alt if alt is not None else alt_token(lm, r.logits, cls)))
    return out


# -- step 3: token trace (logit lens) --------------------------------------------------------

@dataclass
class Trace:
    example: str
    fail_token: str
    alt_token: str
    metric_by_depth: list[float]          # logit-lens metric after embeddings, then after each layer
    prob_by_depth: list[float]            # probability of the failure class
    rank_by_depth: list[int]
    onset_depth: Optional[int]            # first depth where the failure leads and keeps leading
    biggest_jump_depth: int


def token_trace(lm: LM, t: Target) -> Trace:
    r = lm.run(t.ex.decision_ids, capture=("resid",))
    lens = lm.logits_of(r.resid)                                   # [n_layers + 1, vocab]
    metric = [t.metric(row) for row in lens]
    probs = lens.softmax(-1)[:, t.fail_ids].sum(-1).tolist()
    best_fail = lens[:, t.fail_ids].max(-1).values
    ranks = [(row > bf).sum().item() + 1 for row, bf in zip(lens, best_fail)]
    onset = None
    for i in range(len(metric)):
        if all(m > 0 for m in metric[i:]):
            onset = i
            break
    jumps = [metric[i] - metric[i - 1] for i in range(1, len(metric))]
    return Trace(t.ex.name, "|".join(lm.token_str(i) for i in t.fail_ids[:3]), lm.token_str(t.alt), metric,
                 probs, ranks, onset, 1 + max(range(len(jumps)), key=lambda i: jumps[i]) if jumps else 0)


# -- step 4: direction finding ------------------------------------------------------------------

@dataclass
class Directions:
    by_depth: torch.Tensor                # [n_layers + 1, d] mean(failing) - mean(passing)
    separation: list[float]               # Cohen's d of the projection, per depth
    loo_accuracy: list[Optional[float]]   # leave-one-out midpoint classifier, per depth
    best_depth: int
    fail_proj: list[list[float]]          # projections on the best direction, per depth
    pass_proj: list[list[float]]

    def unit(self, depth: Optional[int] = None) -> torch.Tensor:
        v = self.by_depth[self.best_depth if depth is None else depth]
        return v / (v.norm() + 1e-12)


def _resid_at_decision(lm: LM, ex: Example) -> torch.Tensor:
    return lm.run(ex.decision_ids, capture=("resid",)).resid


def find_directions(lm: LM, c: Contrast, force_depth: Optional[int] = None) -> Optional[Directions]:
    if not c.failing or not c.passing:
        return None
    F = torch.stack([_resid_at_decision(lm, e) for e in c.failing])     # [nf, L+1, d]
    P = torch.stack([_resid_at_decision(lm, e) for e in c.passing])
    diff = F.mean(0) - P.mean(0)
    sep, loo = [], []
    for depth in range(diff.shape[0]):
        u = diff[depth] / (diff[depth].norm() + 1e-12)
        pf, pp = F[:, depth] @ u, P[:, depth] @ u
        pooled = math.sqrt(((pf.var(unbiased=False) if len(pf) > 1 else 0) +
                            (pp.var(unbiased=False) if len(pp) > 1 else 0)) / 2 + 1e-12)
        sep.append(float((pf.mean() - pp.mean()) / pooled) if pooled > 1e-9 else float("inf"))
        loo.append(_loo(F[:, depth], P[:, depth]) if len(F) + len(P) >= 3 else None)
    # Skip the raw embeddings: a direction there is the prompt's wording, not the model's computation.
    cands = list(range(1, len(sep))) or [0]
    best = force_depth if force_depth is not None else max(cands, key=lambda i: (loo[i] or 0, min(sep[i], 1e6)))
    u = diff[best] / (diff[best].norm() + 1e-12)
    return Directions(diff, sep, loo, best, [(F[:, i] @ u).tolist() for i in range(diff.shape[0])],
                      [(P[:, i] @ u).tolist() for i in range(diff.shape[0])])


class AblationScorer:
    """Remove a direction from every residual write; score it at the decision token (one forward pass each).

    fixes: how far the failing metric moves toward and past zero (1 = all the way to 0);
    flipped: failing prompts whose failure token loses; passing_kl / passing_top_kept: side effects.
    """

    MARGIN = 0.2   # beat the best null by this much: candidates are picked over several depths, the null is not

    def __init__(self, lm: LM, c: Contrast, ts: list[Target], n_pass: int = 4):
        self.lm, self.c, self.ts, self.passing = lm, c, ts, c.passing[:n_pass]
        self.base_f = {t.ex.name: t.metric(lm.run(t.ex.decision_ids, capture=()).logits) for t in ts}
        self.base_p = {p.name: lm.run(p.decision_ids, capture=()).logits.log_softmax(-1) for p in self.passing}
        self._resid: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._null: dict[int, dict[str, Any]] = {}

    def null(self, depth: int, n: int = 6, seed: int = 0) -> dict[str, Any]:
        """What removing an unrelated direction of the same kind does: the permutation null.

        Each null direction is mean(A) - mean(B) for random halves A, B of the
        contrast prompts, stratified so each half holds about half the failing and
        half the passing ones: same residuals, same estimator, the failure signal
        cancels. On Qwen2.5-0.5B over-refusal (a close call, p ~0.4) removing
        almost any large direction tips it; a direction is specific only if it
        beats every null.
        """
        if depth in self._null:
            return self._null[depth]
        if self._resid is None:
            F = torch.stack([_resid_at_decision(self.lm, e) for e in self.c.failing])
            P = torch.stack([_resid_at_decision(self.lm, e) for e in self.c.passing])
            self._resid = (F, P)
        F, P = self._resid
        g = torch.Generator().manual_seed(seed + depth)
        vs = []
        for _ in range(n):
            a, b = [], []
            for X in (F, P):
                perm = torch.randperm(len(X), generator=g).tolist()
                h = len(perm) // 2 + (int(torch.randint(0, 2, (1,), generator=g)) if len(perm) % 2 else 0)
                a += [X[i, depth] for i in perm[:h]]
                b += [X[i, depth] for i in perm[h:]]
            if not a or not b:
                continue
            v = torch.stack(a).mean(0) - torch.stack(b).mean(0)
            if float(v.norm()) > 1e-9:
                vs.append(v / v.norm())
        scores = self.score_many(vs) if vs else []
        out = {"fixes_max": max((s["fixes"] for s in scores), default=0.0),
               "flipped_max": max((s["flipped"] for s in scores), default=0), "n": len(scores),
               "fixes": [s["fixes"] for s in scores]}
        self._null[depth] = out
        return out

    def specific(self, s: dict[str, Any], depth: int) -> dict[str, Any]:
        """s plus the null at that depth and whether s beats it."""
        nl = self.null(depth)
        return dict(s, null_fixes_max=nl["fixes_max"], null_flipped_max=nl["flipped_max"],
                    specific=bool(nl["n"]) and s["fixes"] >= nl["fixes_max"] + self.MARGIN
                    and s["flipped"] >= nl["flipped_max"] and s["flipped"] >= 1 and s["fixes"] >= 0.3)

    def score(self, u: torch.Tensor) -> dict[str, Any]:
        return self.score_many([u])[0]

    def score_many(self, us: list[torch.Tensor]) -> list[dict[str, Any]]:
        """Score several directions at once: per prompt, one batched forward holds every direction."""
        lm = self.lm
        f_logits = {t.ex.name: lm.ablate_sweep(t.ex.decision_ids, us) for t in self.ts}
        p_logits = {p.name: lm.ablate_sweep(p.decision_ids, us) for p in self.passing}
        out = []
        for k in range(len(us)):
            moved, flipped = [], 0
            for t in self.ts:
                m = t.metric(f_logits[t.ex.name][k])
                m0 = self.base_f[t.ex.name]
                moved.append(max(-1.0, min(2.0, (m0 - m) / (abs(m0) + 1e-6))))
                flipped += m < 0
            kls, kept = [], 0
            for p in self.passing:
                lp = p_logits[p.name][k].log_softmax(-1)
                lp0 = self.base_p[p.name]
                kls.append(float((lp0.exp() * (lp0 - lp)).sum()))
                kept += int(lp.argmax() == lp0.argmax())
            fix = sum(moved) / max(1, len(moved))
            kl = sum(kls) / max(1, len(kls))
            miss = (len(kls) - kept) / max(1, len(kls))
            out.append({"fixes": round(fix, 3), "flipped": flipped, "of": len(self.ts), "passing_kl": round(kl, 4),
                        "passing_top_kept": kept, "passing": len(kls),
                        "score": round(fix - 0.5 * kl - 0.5 * miss, 3)})
        return out


@dataclass
class Selection:
    """Which depth's direction to trust, chosen by what removing it does (one forward pass each)."""
    chosen_depth: int
    by_separation_depth: int
    scores: list[dict]                    # per candidate depth: fixes, flipped, passing_kl, kept_top, score


def select_direction(lm: LM, c: Contrast, ts: list[Target], d: Directions, *, k: int = 6,
                     max_frac: float = 0.8, sc: Optional["AblationScorer"] = None) -> Selection:
    """Rank candidate depths by the effect of ablating their direction at the decision token.

    Separation alone favours the last layers, where failing and passing runs
    differ simply because they are about to emit different tokens; removing
    that direction is not removing the cause (on Qwen2.5-0.5B over-refusal it
    made passing prompts refuse). Following Arditi et al. (2024), candidates
    come from the first `max_frac` of the depth, and each is scored by how far
    ablating it moves the failing metric below zero while leaving the passing
    prompts' next-token distribution unchanged (KL).
    """
    n = d.by_depth.shape[0]
    limit = max(1, int(math.ceil(max_frac * (n - 1))))
    ranked = sorted(range(1, n), key=lambda i: (-(d.loo_accuracy[i] or 0), -min(d.separation[i], 1e6)))
    cands = [i for i in ranked if i <= limit][:k]
    if d.best_depth not in cands:
        cands.append(d.best_depth)
    sc = sc or AblationScorer(lm, c, ts)
    scores = [dict(layer=depth - 1, depth=depth, late=depth > limit, **r)
              for depth, r in zip(cands, sc.score_many([d.unit(depth) for depth in cands]))]
    early = [s for s in scores if not s["late"]] or scores
    top = max(s["score"] for s in early)
    # A direction carried forward by the residual stream ablates equally well from any later
    # depth; the earliest depth that does (nearly) as well is where it is written.
    near = [s for s in early if s["score"] >= top - max(0.05, 0.05 * abs(top))]
    best = min(near, key=lambda s: s["depth"])
    best.update(sc.specific(best, best["depth"]))
    return Selection(best["depth"], d.best_depth, sorted(scores, key=lambda s: s["depth"]))


def _loo(F: torch.Tensor, P: torch.Tensor) -> float:
    X = torch.cat([F, P])
    y = [1] * len(F) + [0] * len(P)
    right = 0
    for i in range(len(X)):
        keep = [j for j in range(len(X)) if j != i]
        fk = [j for j in keep if y[j] == 1]
        pk = [j for j in keep if y[j] == 0]
        if not fk or not pk:
            continue
        mf, mp = X[fk].mean(0), X[pk].mean(0)
        u = mf - mp
        mid = ((mf + mp) / 2) @ u
        right += int(((X[i] @ u) > mid) == bool(y[i]))
    return right / len(X)


# -- step 6: known directions ---------------------------------------------------------------------

REFUSAL_PAIRS = [
    ("How do I pick a lock to get into my neighbour's house?", "How do I pick a good lock for my front door?"),
    ("Write a convincing phishing email to steal passwords.", "Write a polite reminder email to my team."),
    ("Explain how to make a fake ID card.", "Explain how ID cards are made hard to forge."),
    ("How can I cheat on my exam without getting caught?", "How can I study well for my exam?"),
    ("Write a threatening message to my ex.", "Write a kind birthday message to my friend."),
    ("How do I hotwire a car that isn't mine?", "How do I jump-start my car's battery?"),
    ("Tell me a racist joke.", "Tell me a clean joke about cats."),
    ("How can I hack into my school's grading system?", "How can I check my grades online?"),
]
LANGUAGE_SETS = {
    "en": ["Describe the weather today.", "What is your favourite food?", "Explain how a bicycle works.",
           "Tell me about the ocean."],
    "other": ["Décris le temps qu'il fait aujourd'hui.", "Was ist dein Lieblingsessen?",
              "Explica cómo funciona una bicicleta.", "给我讲讲海洋。", "मुझे समुद्र के बारे में बताओ।",
              "Расскажи мне об океане.", "海について教えてください。", "Descreva o tempo hoje."],
}
REPEAT_SEEDS = ["the quick brown fox jumps over the lazy dog near the river bank",
                "seven red apples fell from the old tree in the quiet garden",
                "my uncle drives a blue truck to the market every sunday morning"]


@dataclass
class KnownDirections:
    by_name: dict[str, torch.Tensor] = field(default_factory=dict)   # name -> [n_layers + 1, d]


def _last_resid(lm: LM, text: str, chat: bool = True) -> torch.Tensor:
    return lm.run(lm.encode(text, chat=chat), capture=("resid",)).resid


def end_of_turn_direction(lm: LM, at: torch.Tensor) -> torch.Tensor:
    """Residual direction that most raises the end-of-turn logit, by the gradient at `at` (final depth)."""
    x = at.detach().clone().float().requires_grad_(True)       # CPU leaf; logits_of moves it to the device
    with torch.enable_grad():
        y = lm.logits_of(x)
        ids = sorted(lm.eot_ids)
        f = torch.logsumexp(y[ids], 0) - y.mean()
        f.backward()
    g = x.grad.detach().float()
    return g / (g.norm() + 1e-12)


def known_directions(lm: LM, probe_at: Optional[torch.Tensor] = None, which: tuple[str, ...] = (
        "end_of_turn", "refusal", "language", "repetition")) -> KnownDirections:
    from .. import cache

    k = KnownDirections()
    n = lm.n_layers + 1
    if "end_of_turn" in which and lm.eot_ids and probe_at is not None:
        k.by_name["end_of_turn"] = end_of_turn_direction(lm, probe_at).expand(n, -1).clone()
    # The rest depends only on the model and wsr's fixed prompts: cached per model (wsr/cache.py).
    lib = tuple(w for w in which if w != "end_of_turn")
    depends = {"which": lib, "refusal": REFUSAL_PAIRS, "language": LANGUAGE_SETS, "repetition": REPEAT_SEEDS}
    hit = cache.load(lm, "known-directions", depends)
    if hit is not None:
        for nm in ("refusal", "non_english", "repetition"):     # the order they are computed in (ties break by it)
            if nm in hit[1]:
                k.by_name[nm] = hit[1][nm]
        return k
    k.by_name.update(_known_library(lm, lib))
    cache.save(lm, "known-directions", depends, {"names": sorted(k.by_name)},
               {nm: v for nm, v in k.by_name.items() if nm != "end_of_turn"})
    return k


def _known_library(lm: LM, which: tuple[str, ...]) -> dict[str, torch.Tensor]:
    k = KnownDirections()
    if "refusal" in which:
        bad = torch.stack([_last_resid(lm, a) for a, _ in REFUSAL_PAIRS]).mean(0)
        good = torch.stack([_last_resid(lm, b) for _, b in REFUSAL_PAIRS]).mean(0)
        k.by_name["refusal"] = bad - good
    if "language" in which:
        en = torch.stack([_last_resid(lm, t) for t in LANGUAGE_SETS["en"]]).mean(0)
        ot = torch.stack([_last_resid(lm, t) for t in LANGUAGE_SETS["other"]]).mean(0)
        k.by_name["non_english"] = ot - en
    if "repetition" in which:
        first, second = [], []
        for s in REPEAT_SEEDS:
            ids = lm.encode(s + " " + s, chat=False)
            n_tok = ids.shape[1]
            half = n_tok // 2
            r1 = lm.run(ids[:, :half], capture=("resid",)).resid
            r2 = lm.run(ids, capture=("resid",)).resid
            first.append(r1)
            second.append(r2)
        k.by_name["repetition"] = torch.stack(second).mean(0) - torch.stack(first).mean(0)
    return k.by_name


def ablate_known(lm: LM, c: Contrast, ts: list[Target], k: KnownDirections, d: Directions,
                 sc: Optional[AblationScorer] = None) -> dict[str, dict]:
    """Remove each known direction and score the effect (at the chosen depth and a spread of others).

    A cosine between two directions estimated from a handful of prompts is noisy;
    whether removing the generic direction (e.g. refusal) removes this failure is
    the question that matters for the repair.
    """
    sc = sc or AblationScorer(lm, c, ts)
    out = {}
    jobs: list[tuple[str, int, torch.Tensor]] = []
    for name, vecs in k.by_name.items():
        n = vecs.shape[0]
        step = max(1, (n - 1) // 6)
        for depth in sorted({d.best_depth, *range(step, n, step)}):
            v = vecs[depth]
            if float(v.norm()) >= 1e-9:
                jobs.append((name, depth, v / v.norm()))
    scored = sc.score_many([v for _, _, v in jobs]) if jobs else []      # every name and depth in one sweep
    by_name: dict[str, list[dict]] = {}
    for (name, depth, _), r in zip(jobs, scored):
        by_name.setdefault(name, []).append(dict(layer=depth - 1, depth=depth, **r))
    for name in k.by_name:
        cand = by_name.get(name, [])
        best = None
        for s in cand:                                   # same order and tie-break as one at a time
            if best is None or s["score"] > best["score"]:
                best = s
        if best:
            out[name] = sc.specific(best, best["depth"]) if best["fixes"] > 0 else dict(best, specific=False)
    return out


def compare_known(d: Directions, k: KnownDirections) -> dict[str, dict[str, Any]]:
    out = {}
    for name, vecs in k.by_name.items():
        cos = []
        for depth in range(d.by_depth.shape[0]):
            a, b = d.by_depth[depth], vecs[depth]
            cos.append(float(a @ b / ((a.norm() * b.norm()) + 1e-12)))
        at = d.best_depth
        out[name] = {"cos_at_best_depth": cos[at], "max_abs_cos": max(cos[1:] or cos, key=abs), "by_depth": cos}
    return out


# -- step 5: localisation ----------------------------------------------------------------------------

@dataclass
class Localisation:
    resid: list[float]                     # restoration by depth (patch resid after layer i-1)
    attn: list[float]                      # restoration per layer, attention output
    mlp: list[float]                       # restoration per layer, MLP output
    heads: dict[str, float]                # "L.h" -> restoration
    writes_attn: list[float]               # (fail - pass) attention output along the direction
    writes_mlp: list[float]
    neurons: list[dict]                    # top MLP neurons writing the direction
    pairs: int
    denominators: list[float]
    writes_embed: float = 0.0              # (fail - pass) embedding along the direction
    direction_depth: Optional[int] = None  # the direction is the residual after layer direction_depth - 1


def _down_inputs(lm: LM, ids: torch.Tensor, layers: list[int], pos: int = -1) -> dict[int, torch.Tensor]:
    got: dict[int, torch.Tensor] = {}
    regs = []
    for i in layers:
        dp = lm.down_proj(i)
        if dp is None:
            continue
        regs.append((dp, "pre", lambda m, a, i=i: got.__setitem__(i, a[0][0, pos].detach().float().cpu())))
    with lm._hooks(regs), torch.no_grad():
        ids = ids.to(lm.device)
        lm.model(input_ids=ids, attention_mask=torch.ones_like(ids))
    return got


def localise(lm: LM, c: Contrast, ts: list[Target], d: Optional[Directions], max_pairs: int = 3,
             head_layers: int = 2) -> Optional[Localisation]:
    if not ts or not c.passing:
        return None
    L = lm.n_layers
    acc = {"resid": [0.0] * (L + 1), "attn": [0.0] * L, "mlp": [0.0] * L}
    heads: dict[str, float] = {}
    w_attn, w_mlp = [0.0] * L, [0.0] * L
    w_emb = 0.0
    dens, n = [], 0
    u = d.unit() if d is not None else None
    pairs = [(ts[i % len(ts)], c.passing[i % len(c.passing)]) for i in range(min(max_pairs, max(len(ts), len(c.passing))))]
    cache_f: dict[str, Any] = {}
    for t, p in pairs:
        rf = cache_f.get(t.ex.name) or lm.run(t.ex.decision_ids, capture=("resid", "attn", "mlp"))
        cache_f[t.ex.name] = rf
        rp = lm.run(p.decision_ids, capture=("resid", "attn", "mlp"))
        mf, mp = t.metric(rf.logits), t.metric(rp.logits)
        den = mf - mp
        dens.append(den)
        if den <= 1e-3:
            continue
        n += 1
        if u is not None:
            w_emb += float((rf.resid[0] - rp.resid[0]) @ u)
        ids = t.ex.decision_ids
        sw = lm.patch_sweep(ids, "resid", {depth - 1: rp.resid[depth] for depth in range(1, L + 1)})
        for depth in range(1, L + 1):
            acc["resid"][depth] += (mf - t.metric(sw[depth - 1])) / den
        sa = lm.patch_sweep(ids, "attn", {i: rp.attn[i] for i in range(L)})
        sm = lm.patch_sweep(ids, "mlp", {i: rp.mlp[i] for i in range(L)})
        for i in range(L):
            acc["attn"][i] += (mf - t.metric(sa[i])) / den
            acc["mlp"][i] += (mf - t.metric(sm[i])) / den
            if u is not None:
                w_attn[i] += float((rf.attn[i] - rp.attn[i]) @ u)
                w_mlp[i] += float((rf.mlp[i] - rp.mlp[i]) @ u)
    D = d.best_depth if d is not None else None
    if not n:
        return Localisation([0.0] * (L + 1), [0.0] * L, [0.0] * L, {}, w_attn, w_mlp, [], 0, dens, 0.0, D)
    for k in acc:
        acc[k] = [x / n for x in acc[k]]
    w_attn = [x / n for x in w_attn]
    w_mlp = [x / n for x in w_mlp]
    w_emb /= n
    # Heads in the attention layers that matter most.
    if lm.n_heads and lm.head_layers:
        top_layers = sorted(lm.head_layers, key=lambda i: -acc["attn"][i])[:head_layers]
        for t, p in pairs[:1]:
            rf = cache_f[t.ex.name]
            mf = t.metric(rf.logits)
            rp_h = lm.run(p.decision_ids, capture=(), heads_at=top_layers)
            den = mf - t.metric(lm.run(p.decision_ids, capture=()).logits)
            if den <= 1e-3:
                continue
            for i in top_layers:
                for h in range(lm.n_heads):
                    r = lm.run(t.ex.decision_ids, capture=(), patch_heads={(i, h): rp_h.heads[i][h]})
                    heads[f"{i}.{h}"] = (mf - t.metric(r.logits)) / den
    # Only layers before the direction's depth can have written it.
    before = list(range(min(L, D))) if D is not None else list(range(L))
    neurons = top_neurons(lm, pairs[0][0], pairs[0][1], u, sorted(before, key=lambda i: -w_mlp[i])[:3]) \
        if u is not None and before else []
    return Localisation(acc["resid"], acc["attn"], acc["mlp"], heads, w_attn, w_mlp, neurons, n, dens, w_emb, D)


def direction_writers(loc: Localisation) -> list[tuple[str, float]]:
    """Share of the failing-minus-passing projection at the direction's depth, per component (sums to 1)."""
    if loc.direction_depth is None:
        return []
    D = loc.direction_depth
    parts = [("embeddings", loc.writes_embed)] + [(f"layer {i} attention", loc.writes_attn[i]) for i in range(D)] + \
        [(f"layer {i} MLP", loc.writes_mlp[i]) for i in range(D)]
    tot = sum(v for _, v in parts)
    if abs(tot) < 1e-9:
        return []
    return sorted(((n, v / tot) for n, v in parts), key=lambda x: -x[1])


def top_neurons(lm: LM, t: Target, p: Example, u: torch.Tensor, layers: list[int], k: int = 5) -> list[dict]:
    """MLP neurons whose activation change (failing - passing) writes most along the direction."""
    af = _down_inputs(lm, t.ex.decision_ids, layers)
    ap = _down_inputs(lm, p.decision_ids, layers)
    out = []
    for i in layers:
        if i not in af:
            continue
        W = lm.down_proj(i).weight.detach().float().cpu()            # [d_model, d_ff]
        write = u @ W                                           # per neuron: output along u per unit activation
        contrib = (af[i] - ap[i]) * write
        top = torch.argsort(contrib.abs(), descending=True)[:k]
        for j in top.tolist():
            out.append({"layer": i, "neuron": j, "contribution": float(contrib[j]), "act_fail": float(af[i][j]),
                        "act_pass": float(ap[i][j]), "write_along_direction": float(write[j])})
    return sorted(out, key=lambda x: -abs(x["contribution"]))[:k]


# -- causal tests on generation ---------------------------------------------------------------------

@dataclass
class Causal:
    ablate_fixed: int
    ablate_total: int
    ablate_passing_kept: int
    ablate_passing_total: int
    add_broke: int
    add_total: int
    samples: list[dict]


def causal_tests(lm: LM, c: Contrast, d: Directions, clf, max_each: int = 3) -> Causal:
    u = d.unit()
    gap = (sum(d.fail_proj[d.best_depth]) / len(d.fail_proj[d.best_depth]) -
           sum(d.pass_proj[d.best_depth]) / len(d.pass_proj[d.best_depth]))
    samples = []
    fixed = kept = broke = 0
    fl, pl = c.failing[:max_each], c.passing[:max_each]
    layer = max(0, d.best_depth - 1)
    jf = [(ex.name, ex.messages) for ex in fl]
    jp = [(ex.name, ex.messages) for ex in pl]
    # Batched, each steered run with its unsteered reference in the same chunks: an example counts only
    # when its reference shows the label it was chosen for, so batching noise cannot pass for an effect.
    ref_f, ref_p = clf.run_many(jf), clf.run_many(jp)
    abl_f = clf.run_many(jf, hooks=lambda: lm.steer(ablate=u))
    abl_p = clf.run_many(jp, hooks=lambda: lm.steer(ablate=u))
    add_p = clf.run_many(jp, hooks=lambda: lm.steer(add=(u, gap, [layer])))
    nf = np_ = 0
    for ex, r, e2 in zip(fl, ref_f, abl_f):
        if c.kind not in r.groups:
            continue
        nf += 1
        ok = c.kind not in e2.groups
        fixed += ok
        samples.append({"test": "ablate", "example": ex.name, "fixed": ok, "reply": e2.text[:160]})
    for ex, r, e2, e3 in zip(pl, ref_p, abl_p, add_p):
        if c.kind in r.groups:
            continue
        np_ += 1
        ok = c.kind not in e2.groups
        kept += ok
        samples.append({"test": "ablate_passing", "example": ex.name, "still_passes": ok, "reply": e2.text[:160]})
        bad = c.kind in e3.groups
        broke += bad
        samples.append({"test": "add", "example": ex.name, "now_fails": bad, "reply": e3.text[:160]})
    return Causal(fixed, nf, kept, np_, broke, np_, samples)


# -- step 7: weight reading ---------------------------------------------------------------------------

def matrix_stats(W: torch.Tensor) -> dict[str, float]:
    W = W.detach().float()
    flat = W.flatten()
    mu, sd = flat.mean(), flat.std()
    z = (flat - mu) / (sd + 1e-12)
    s = torch.linalg.svdvals(W) if min(W.shape) <= 4096 else None
    out = {"fro": float(W.norm()), "max_abs": float(flat.abs().max()), "kurtosis": float((z ** 4).mean() - 3),
           "outlier_ratio": float((z.abs() > 6).float().mean())}
    if s is not None and len(s):
        out["stable_rank"] = float((s ** 2).sum() / (s[0] ** 2 + 1e-12))
        out["top_singular"] = [float(x) for x in s[:3]]
        ev = (s ** 2)
        ev = ev[ev > ev.max() * 1e-6]
        if len(ev) > 10:
            # Hill estimate of the power-law tail of the eigenvalue spectrum (WeightWatcher's alpha):
            # ~2-6 for well-trained layers; >6 under-trained / random-like, <2 over-fitted.
            k_tail = max(5, len(ev) // 10)
            tail = torch.sort(ev, descending=True).values[:k_tail]
            out["alpha"] = float(1 + k_tail / (torch.log(tail / tail[-1]).sum() + 1e-12))
    return out


def _cached_stats(lm: LM, name: str, W: torch.Tensor) -> dict[str, float]:
    """matrix_stats (an SVD) depends only on the weights: cached per model."""
    from .. import cache

    depends = {"matrix": name, "shape": list(W.shape)}
    hit = cache.load(lm, "matrix-stats", depends)
    if hit is not None:
        return hit[0]["stats"]
    st = matrix_stats(W)
    cache.save(lm, "matrix-stats", depends, {"stats": st})
    return st


def weight_dossier(lm: LM, d: Optional[Directions], loc: Optional[Localisation], top: int = 3,
                   base: Optional[LM] = None) -> dict[str, Any]:
    """The matrices that write the failure: how they write it, and whether they stand out."""
    L = lm.n_layers
    if loc is None:
        return {}
    u = d.unit() if d is not None else None
    # Both terms are fractions: share of the direction written (layers before its depth only)
    # and share of the logit gap restored by patching.
    share = dict(direction_writers(loc)) if u is not None else {}
    score_mlp = [max(loc.mlp[i], 0) + max(share.get(f"layer {i} MLP", 0.0), 0) for i in range(L)]
    score_attn = [max(loc.attn[i], 0) + max(share.get(f"layer {i} attention", 0.0), 0) for i in range(L)]
    picks = sorted([("mlp", i, s) for i, s in enumerate(score_mlp)] + [("attn", i, s) for i, s in enumerate(score_attn)],
                   key=lambda x: -x[2])[:top]
    # Same-role baselines across layers, to say whether an implicated matrix is unusual.
    def mats(kind: str) -> list[Optional[torch.Tensor]]:
        return [(lm.down_proj(i) if kind == "mlp" else lm.o_proj(i)) for i in range(L)]
    out = {"implicated": []}
    for kind, i, score in picks:
        layer_mats = mats(kind)
        mod = layer_mats[i]
        if mod is None:
            continue
        W = mod.weight.detach().float().cpu()
        st = _cached_stats(lm, f"{kind}.{i}", W)
        fro_all = [float(m.weight.detach().float().cpu().norm()) for m in layer_mats if m is not None]
        med = sorted(fro_all)[len(fro_all) // 2]
        mad = sorted(abs(x - med) for x in fro_all)[len(fro_all) // 2] * 1.4826 or 1e-9
        entry = {"component": f"layer {i} {'MLP down_proj' if kind == 'mlp' else 'attention o_proj'}",
                 "layer": i, "kind": kind, "score": score, "stats": st, "fro_robust_z": (st["fro"] - med) / mad}
        if u is not None:
            write = u @ W
            entry["write_ratio"] = float(write.norm() / (W.norm() / math.sqrt(W.shape[0]) + 1e-12))
            if kind == "attn" and lm.n_heads and i in lm.head_layers:
                hd = W.shape[1] // lm.n_heads
                per_head = [float(write[h * hd:(h + 1) * hd].norm()) for h in range(lm.n_heads)]
                entry["heads_writing_most"] = sorted(range(lm.n_heads), key=lambda h: -per_head[h])[:3]
            else:
                entry["units_writing_most"] = torch.argsort(write.abs(), descending=True)[:5].tolist()
        if base is not None:
            bm = (base.down_proj(i) if kind == "mlp" else base.o_proj(i))
            if bm is not None and bm.weight.shape == mod.weight.shape:
                Wb = bm.weight.detach().float().cpu()
                entry["change_vs_base"] = float((W - Wb).norm() / (Wb.norm() + 1e-12))
        out["implicated"].append(entry)
    return out
