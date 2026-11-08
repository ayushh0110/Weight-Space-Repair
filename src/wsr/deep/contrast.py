"""Step 2: a contrast set of failing prompts and near-identical passing ones.

Variants are small, meaning-preserving edits of the failing prompt (drop a
polite word, trim the last word, drop the system prompt, rephrase the opening,
add or remove final punctuation...). Each is run greedily on the model and
classified with the same detectors as the triage layer, so "passing" means the
same thing everywhere. User-supplied failing / passing prompts are added as is.

Each example also gets its decision point: the position where the failure is
chosen (the first answer token for blank / refusal / language failures; the
first token of the second copy for loops) and the token the model picked there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..output import _LIST_MARK, check_output
from ..verdict import _groups_of
from .lm import LM

_POLITE = re.compile(r"\b(please|kindly|thanks|thank you|could you|can you|would you)\b[,!.]?\s*", re.I)
_OPENERS = ("", "Please ", "Could you ", "I need you to ", "Quick question: ")


@dataclass
class Example:
    name: str
    messages: list[dict]
    prompt_ids: torch.Tensor
    gen: list[int]
    text: str
    groups: set[str]
    failing: bool
    decision_ids: Optional[torch.Tensor] = None    # tokens fed up to the decision point
    decision_tok: Optional[int] = None             # what the model chose there
    meta: dict = field(default_factory=dict)

    def brief(self) -> dict:
        return {"name": self.name, "prompt": _last_user(self.messages)[:200], "reply": self.text[:200],
                "failing": self.failing, "groups": sorted(self.groups)}


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _with_last_user(messages: list[dict], text: str) -> list[dict]:
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = text
            break
    return out


def variants(messages: list[dict]) -> list[tuple[str, list[dict]]]:
    """Near-identical rewrites of the last user turn (and of the conversation around it)."""
    u = _last_user(messages).strip()
    out: list[tuple[str, str]] = []
    words = u.split()
    if len(words) > 2:
        out.append(("drop_last_word", " ".join(words[:-1])))
        out.append(("drop_first_word", " ".join(words[1:])))
    p = _POLITE.sub("", u).strip()
    if p and p != u:
        out.append(("no_polite_words", p[0].upper() + p[1:]))
    if u and u[-1] in ".?!":
        out.append(("no_final_punct", u[:-1]))
    else:
        out.append(("final_period", u + "."))
    out.append(("lowercase", u.lower()))
    for op in _OPENERS[1:3]:
        out.append((f"opener:{op.strip()}", op + u[0].lower() + u[1:] if u else op))
    out.append(("brief", u + " Answer briefly."))
    sentences = re.split(r"(?<=[.?!])\s+", u)
    if len(sentences) > 1:
        for i in range(len(sentences)):
            out.append((f"drop_sentence_{i}", " ".join(s for j, s in enumerate(sentences) if j != i)))
    seen, res = {u}, []
    for name, text in out:
        if text and text not in seen:
            seen.add(text)
            res.append((name, _with_last_user(messages, text)))
    if any(m.get("role") == "system" for m in messages):
        res.append(("no_system_prompt", [m for m in messages if m.get("role") != "system"]))
    return res


_FRESH_OPENERS = ("I need you to ", "Quick question: ", "Hi! ", "Hello, ", "Hey, can you help? ")
_FRESH_SUFFIXES = (" Thanks!", " Thank you.", " Please answer.", " Be precise.")


def fresh_variants(messages: list[dict], exclude: set[str] = frozenset()) -> list[tuple[str, list[dict]]]:
    """Meaning-preserving rewrites built differently from `variants()` (other openers, closing
    words, a default system prompt), so a patch chosen on the contrast set can be confirmed on
    prompts that took no part in choosing it."""
    u = _last_user(messages).strip()
    if not u:
        return []
    low = u[0].lower() + u[1:]
    out: list[tuple[str, str]] = [(f"fresh_opener:{op.strip()}", op + low) for op in _FRESH_OPENERS]
    out += [(f"fresh_suffix:{s.strip()}", u + s) for s in _FRESH_SUFFIXES]
    out.append(("fresh_newline", u + "\n"))
    out.append(("fresh_quoted", f"My request: {u}"))
    seen = {u, *(_last_user(m).strip() for _, m in variants(messages))} | set(exclude)
    res = []
    for name, text in out:
        if text.strip() and text not in seen:
            seen.add(text)
            res.append((name, _with_last_user(messages, text)))
    if not any(m.get("role") == "system" for m in messages):
        res.append(("fresh_system_prompt", [{"role": "system", "content": "You are a helpful assistant."}]
                    + [dict(m) for m in messages]))
    return res


def _token_loop_onset(gen: list[int], max_period: int = 40, min_repeats: int = 3) -> Optional[tuple[int, int]]:
    """(onset, period) of a repeating tail of token ids, or None."""
    best = None
    n = len(gen)
    for p in range(1, min(max_period, n // min_repeats) + 1):
        k = 0
        while k + p < n and gen[n - 1 - k] == gen[n - 1 - k - p]:
            k += 1
        covered = k + p
        if covered >= p * min_repeats and (best is None or covered > best[2]):
            best = (n - covered, p, covered)
    return (best[0], best[1]) if best else None


class Classifier:
    def __init__(self, lm: LM, *, expect_lang: Optional[str] = None, max_new: int = 48,
                 template_kwargs: Optional[dict] = None):
        self.lm, self.expect_lang, self.max_new = lm, expect_lang, max_new
        self.template_kwargs = template_kwargs or {}

    def run(self, name: str, messages: list[dict], hooks=None) -> Example:
        ids = self.lm.encode(messages, **self.template_kwargs)
        gen = self.lm.generate(ids, self.max_new, hooks=hooks)
        return self._example(name, messages, ids, gen)

    def run_many(self, items: list[tuple[str, list[dict]]], hooks=None) -> list[Example]:
        """`run` on many (name, messages), batched as memory allows; same order. `hooks` (e.g.
        `lambda: lm.steer(ablate=u)`) applies to every row. The same list runs in the same chunks every
        time (LM.generate_batch), so a steered run and its unsteered reference are numerically paired."""
        ids = [self.lm.encode(m, **self.template_kwargs) for _, m in items]
        gens = self.lm.generate_batch(ids, self.max_new, hooks=hooks)
        return [self._example(n, m, i, g) for (n, m), i, g in zip(items, ids, gens)]

    def _example(self, name: str, messages: list[dict], ids, gen: list[int]) -> Example:
        ended = bool(gen) and gen[-1] in self.lm.eot_ids
        body = gen[:-1] if ended else gen
        text = self.lm.tok.decode(body, skip_special_tokens=True)
        groups = self.groups_of(text, gen, request=_last_user(messages))
        return Example(name, messages, ids, gen, text, groups, bool(groups))

    def groups_of(self, text: str, gen: list[int], request: Optional[str] = None) -> set[str]:
        fs = check_output(text, finish_reason=None, expect_lang=self.expect_lang, check_refusal=True,
                          request=request)
        codes = {f.code for f in fs if f.failed}
        codes.discard("TRUNCATED")     # max_new is ours, not the user's cap
        if not gen or gen[0] in self.lm.eot_ids:
            codes.add("EMPTY")
        return _groups_of(codes)


def list_items(lm: LM, gen: list[int]) -> list[tuple[int, str]]:
    """(token index where the item's text starts, normalised item text) for each list line."""
    body = gen[:-1] if gen and gen[-1] in lm.eot_ids else gen
    if not body:
        return []
    ends = [len(lm.tok.decode(body[:i + 1], skip_special_tokens=True)) for i in range(len(body))]
    text = lm.tok.decode(body, skip_special_tokens=True)
    items, start = [], 0
    complete = text.endswith("\n")
    lines = text.split("\n")
    for n, line in enumerate(lines):
        m = _LIST_MARK.match(line)
        content = line[m.end():].strip() if m else ""
        last = n == len(lines) - 1
        if m and content and (complete or not last):
            cs = start + m.end()
            tok = next((i for i, e in enumerate(ends) if e > cs), None)
            if tok is not None:
                items.append((tok, content.lower()))
        start += len(line) + 1
    return items


def _list_loop_item(items: list[tuple[int, str]], max_period: int = 10) -> Optional[tuple[int, int]]:
    """(index of the first item of the second copy, period) of a stuck list, or None."""
    xs = [t for _, t in items]
    best = None
    for p in range(1, min(max_period, len(xs) // 3) + 1):
        reps, cur = 1, len(xs) - p
        while cur - p >= 0 and xs[cur - p:cur] == xs[len(xs) - p:]:
            reps += 1
            cur -= p
        if reps >= 3 and (best is None or reps * p > best[2]):
            best = (cur + p, p, reps * p)
    return (best[0], best[1]) if best else None


def _at(ex: Example, tok: int) -> None:
    ex.decision_ids = torch.cat([ex.prompt_ids, torch.tensor([ex.gen[:tok]], dtype=torch.long)], dim=1)
    ex.decision_tok = ex.gen[tok]


def decision_point(lm: LM, ex: Example, kind: str, loop_item: Optional[int] = None,
                   at_token: Optional[int] = None) -> None:
    """Where the failure is chosen, and the token chosen there.

    Loops: the first token of the second copy. A stuck list ("16. Penguin / 17. Penguin")
    is not token-periodic, because the item numbers change, so list items are compared
    with their markers stripped. Passing runs of a loop failure are read at the same
    list position (loop_item), so failing and passing are compared at the same step.
    """
    gen = ex.gen
    if kind == "blank" and gen:
        # A blank / cut-short reply is chosen where the end-of-turn token comes, which is not
        # always the first token ("C" then end-of-turn). Passing runs are read at the same step.
        if ex.failing:
            j = next((i for i, t in enumerate(gen) if t in lm.eot_ids), None)
            if j:
                _at(ex, j)
                ex.meta.update(eot_at=j)
                return
        elif at_token and len(gen) > at_token:
            _at(ex, at_token)
            ex.meta.update(eot_at=at_token, aligned_to_failing=True)
            return
    if kind == "loop" and gen:
        items = list_items(lm, gen)
        if ex.failing:
            li = _list_loop_item(items)
            if li:
                q, period = li
                _at(ex, items[q][0])
                ex.meta.update(loop_item=q, loop_period=period, loop_kind="list", repeated=items[q][1][:60])
                return
            lo = _token_loop_onset(gen[:-1] if gen[-1] in lm.eot_ids else gen)
            if lo:
                onset, period = lo
                _at(ex, onset + period)
                ex.meta.update(loop_onset=onset, loop_period=period, loop_kind="tokens")
                return
        elif loop_item is not None and items:
            q = min(loop_item, len(items) - 1)
            _at(ex, items[q][0])
            ex.meta.update(loop_item=q, aligned_to_failing=True)
            return
    ex.decision_ids = ex.prompt_ids
    ex.decision_tok = gen[0] if gen else None


@dataclass
class Contrast:
    kind: str
    failing: list[Example]
    passing: list[Example]
    tried: list[Example]


def primary_kind(groups: set[str]) -> str:
    for k in ("blank", "loop", "refusal", "language", "template_leak", "garbage", "structure"):
        if k in groups:
            return k
    return "other"


def build(lm: LM, messages: list[dict], *, kind: str = "auto", extra_failing: tuple[str, ...] = (),
          extra_passing: tuple[str, ...] = (), expect_lang: Optional[str] = None, max_new: int = 48,
          template_kwargs: Optional[dict] = None, max_variants: int = 12) -> Contrast:
    clf = Classifier(lm, expect_lang=expect_lang, max_new=max_new, template_kwargs=template_kwargs)
    jobs = ([("original", messages)]
            + [(f"user_failing_{i}", _with_last_user(messages, t)) for i, t in enumerate(extra_failing)]
            + [(f"user_passing_{i}", _with_last_user(messages, t)) for i, t in enumerate(extra_passing)]
            + list(variants(messages)[:max_variants]))
    tried = clf.run_many(jobs)
    orig = tried[0]
    k = primary_kind(orig.groups) if kind == "auto" else kind
    # An example fails when it shows the original's failure kind.
    for ex in tried:
        ex.failing = k in ex.groups if k != "other" else bool(ex.groups)
    failing = [e for e in tried if e.failing]
    passing = [e for e in tried if not e.failing and not e.groups]
    for e in failing:
        decision_point(lm, e, k)
    qs = sorted(e.meta["loop_item"] for e in failing if "loop_item" in e.meta)
    q = qs[len(qs) // 2] if qs else None
    js = sorted(e.meta.get("eot_at", 0) for e in failing)
    j = js[len(js) // 2] if js else None
    for e in passing:
        decision_point(lm, e, k, loop_item=q, at_token=j or None)
    return Contrast(k, failing, passing, tried)
