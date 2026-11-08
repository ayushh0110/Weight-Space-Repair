"""Matched continuations of the same prompt, for loops: the failing continuation (the second
copy the model wrote) and a working one (the model's own continuation from the same point,
sampled with repeated 3-grams forbidden), so only the failure differs between them.

Two uses:

* a direction and sites from what differs between them (sites.py), instead of failing vs
  passing prompts, which also differ in what the prompts are about;
* a cheap screen for a candidate edit, one teacher-forced forward per text instead of
  generating until the loop starts: the edit should make the repeated continuation less
  likely and leave the working one alone, and must not make the working one end early
  (the premature-stop control). Measured the same way on Qwen3.5-0.8B's reasoning loops
  (Takehi et al. 2026, "Sparks of weight-space programming"): -0.024 / -0.049 / -0.104
  nats per token on the repeated text at strengths 1 / 2 / 4, the working text unchanged.

Built for loops only. A blank, refusal or wrong-language failure is chosen at one token, and
both continuations share the prompt up to it, so there is nothing to compare before that token.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..deep.contrast import _list_loop_item, _token_loop_onset, list_items
from ..deep.lm import LM
from .sites import State


@dataclass
class Pair:
    name: str
    prefix: torch.Tensor      # [1, P]: prompt + reply up to where the loop's second copy starts
    bad: list[int]            # the failing continuation (the repeat)
    good: list[int]           # a working continuation from the same point


def _loops(lm: LM, gen: list[int]) -> bool:
    body = gen[:-1] if gen and gen[-1] in lm.eot_ids else gen
    return _token_loop_onset(body) is not None or _list_loop_item(list_items(lm, body)) is not None


@torch.no_grad()
def build(lm: LM, c: Any, *, n_tokens: int = 24, tries: int = 4, seed: int = 0, max_pairs: int = 6) -> list[Pair]:
    if c is None or c.kind != "loop":
        return []
    out: list[Pair] = []
    for ex in c.failing:
        if len(out) >= max_pairs:
            break
        if ex.decision_ids is None:
            continue
        P0 = ex.prompt_ids.shape[1]
        tok = ex.decision_ids.shape[1] - P0
        if tok <= 0:
            continue
        bad = [t for t in ex.gen[tok:tok + n_tokens] if t not in lm.eot_ids]
        if len(bad) < 4:
            continue
        for k in range(tries):
            torch.manual_seed(seed + 7919 * k)
            # The working continuation must not take the step the loop takes: its first repeated token
            # is forbidden (a sampled " Penguin Fish" after "11. Penguin" still repeats the item).
            g = lm.generate(ex.decision_ids, n_tokens, do_sample=True, temperature=0.9, top_p=0.95,
                            no_repeat_ngram_size=3, bad_words_ids=[[int(bad[0])]])
            body = [t for t in g if t not in lm.eot_ids]
            if len(body) < 4 or body[0] == bad[0] or _loops(lm, ex.gen[:tok] + body):
                continue
            out.append(Pair(ex.name, ex.decision_ids, bad, body))
            break
    return out


def states(pairs: list[Pair]) -> list[State]:
    out = []
    for p in pairs:
        P = p.prefix.shape[1]
        for side, cont in ((True, p.bad), (False, p.good)):
            ids = torch.cat([p.prefix, torch.tensor([cont], dtype=torch.long)], 1)
            out.append(State(f"{p.name}:{'repeat' if side else 'working'}", ids, list(range(P, P + len(cont))), side))
    return out


@torch.no_grad()
def _score(lm: LM, prefix: torch.Tensor, cont: list[int]) -> tuple[float, float]:
    """(mean log p of the continuation's tokens, mean probability of ending the turn at them)."""
    ids = torch.cat([prefix, torch.tensor([cont], dtype=torch.long)], 1).to(lm.device)
    k = len(cont) + 1          # only the positions that predict the continuation (vocab x T would not fit)
    extra = {"logits_to_keep": k} if lm._keeps_logits else {}
    lg = lm.model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra).logits[0]
    rows = lg[-k:-1].float().log_softmax(-1)
    tgt = torch.tensor(cont, device=rows.device)
    eot = sorted(lm.eot_ids)
    p_end = rows[:, eot].exp().sum(-1) if eot else torch.zeros(len(cont), device=rows.device)
    return float(rows.gather(1, tgt[:, None]).mean()), float(p_end.mean())


def selectivity(lm: LM, pairs: list[Pair]) -> Optional[dict[str, float]]:
    """Mean over pairs: log p per token of the repeated and of the working continuation, and the
    probability of ending the turn inside the working one."""
    if not pairs:
        return None
    bad = [_score(lm, p.prefix, p.bad) for p in pairs]
    good = [_score(lm, p.prefix, p.good) for p in pairs]
    n = len(pairs)
    return {"logp_repeat": sum(b[0] for b in bad) / n, "logp_working": sum(g[0] for g in good) / n,
            "p_end_working": sum(g[1] for g in good) / n}


def compare(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    d_rep = after["logp_repeat"] - before["logp_repeat"]
    d_work = after["logp_working"] - before["logp_working"]
    early = after["p_end_working"] > max(0.05, 3 * before["p_end_working"])
    return {"repeat_delta": round(d_rep, 4), "working_delta": round(d_work, 4),
            "p_end_working_before": round(before["p_end_working"], 5), "p_end_working_after":
            round(after["p_end_working"], 5), "selective": d_rep < 0 and d_rep < d_work,
            "early_stop_risk": early}
