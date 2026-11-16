"""DRY ("don't repeat yourself"), the repetition sampler llama.cpp ships (`--dry-multiplier`), for PyTorch generation.

A global repetition penalty (transformers' `repetition_penalty`) divides the logit of EVERY token already in the
context, the prompt included: it pushes away from the request's own words and from the model's usual phrasing,
which can change what the model does (a loop goal's agent saw it stop refusing harmful requests). DRY penalises
only a token that would EXTEND a sequence that already occurred: if the context ends with tokens that also appear
earlier, followed there by token t, then t is penalised by

    multiplier * base ** (length - allowed_length)        when length >= allowed_length

where length is how many tokens of that earlier occurrence match the end of the context. A sequence breaker
(newline, ':', '"', '*' by default) ends a match, so ordinary structure (list numbering, headings, quotes) is not
treated as repetition. Parameters and defaults follow llama.cpp (base 1.75, allowed length 2, the whole context);
a breaker here is a token whose text contains a breaker string (llama.cpp also matches breakers spanning tokens).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

BREAKERS = ("\n", ":", '"', "*")
MAX_EXPONENT = 40           # base ** 40 already removes the token; a bound keeps the arithmetic finite


def penalties(seq: Sequence[int], is_breaker, multiplier: float, base: float, allowed_length: int,
              max_match: int = 64) -> dict[int, float]:
    """{token: amount to subtract from its logit} for the next token after `seq`."""
    n = len(seq)
    if n < 2 or is_breaker(seq[-1]):
        return {}
    room = 0                                     # tokens since the last breaker: a match cannot reach past it
    for t in reversed(seq):
        if is_breaker(t):
            break
        room += 1
    room = min(room, max_match)
    last, best = seq[-1], {}
    for i in range(n - 2, -1, -1):
        if seq[i] != last:
            continue
        length = 1
        while length < room and i - length >= 0 and seq[i - length] == seq[n - 1 - length] \
                and not is_breaker(seq[i - length]):
            length += 1
        nxt = seq[i + 1]
        if length >= allowed_length and length > best.get(nxt, 0):
            best[nxt] = length
    return {t: multiplier * base ** min(L - allowed_length, MAX_EXPONENT) for t, L in best.items()}


class DRYProcessor:
    """A transformers LogitsProcessor. `pad`: the left-padding id of a batch (leading runs of it are not context)."""

    def __init__(self, tok: Any, multiplier: float, base: float = 1.75, allowed_length: int = 2,
                 breakers: Sequence[str] = BREAKERS, pad: Optional[int] = None):
        self.tok, self.multiplier, self.base = tok, float(multiplier), float(base)
        self.allowed_length, self.breakers, self.pad = int(allowed_length), tuple(breakers), pad
        self._breaker: dict[int, bool] = {}

    def is_breaker(self, t: int) -> bool:
        b = self._breaker.get(t)
        if b is None:
            text = self.tok.decode([t])
            b = self._breaker[t] = any(s in text for s in self.breakers)
        return b

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.multiplier <= 0:
            return scores
        for r, row in enumerate(input_ids.tolist()):
            k = 0
            if self.pad is not None:
                while k < len(row) and row[k] == self.pad:
                    k += 1
            pen = penalties(row[k:], self.is_breaker, self.multiplier, self.base, self.allowed_length)
            if pen:
                idx = torch.tensor(list(pen), device=scores.device)
                scores[r, idx] -= torch.tensor(list(pen.values()), device=scores.device, dtype=scores.dtype)
        return scores


def processor(tok: Any, spec: dict, pad: Optional[int] = None) -> Any:
    """`spec` as Protocol.gen_kwargs puts it: {multiplier, base, allowed_length}."""
    from transformers import LogitsProcessorList

    return LogitsProcessorList([DRYProcessor(tok, spec["multiplier"], spec.get("base", 1.75),
                                             spec.get("allowed_length", 2), pad=pad)])
