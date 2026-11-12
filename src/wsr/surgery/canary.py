"""semantic_regression_canary: does a fix lose simple things the model could already do?

The detectors and the regression suite see the form of a reply (blank, loop, language, refusal) and how far the
model drifted; they cannot tell "the answer is 42" from "the answer is 17". This canary checks answers: small
deterministic tasks in several categories, each with an answer that is computed by code (arithmetic, units,
sequences, comparisons) or fixed and hand-checked (a few facts and instruction formats).

It is relative to the model: only items the unedited model gets right count (an item it already fails cannot be
lost), replies are greedy, and the fix may lose at most `max_lost` of them (1 by default). It detects specific
capability regressions; it does not establish that replies are correct in general.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Optional

NAME = "semantic_regression_canary"
MAX_NEW = 48
MAX_LOST = 1
VERSION = 1

# Fixed facts, each checked by hand and unambiguous in its one-word answer.
FACTS = [
    ("What is the capital of France? Answer with one word.", "Paris"),
    ("What is the capital of Japan? Answer with one word.", "Tokyo"),
    ("What is the capital of Italy? Answer with one word.", "Rome"),
    ("What is the capital of Germany? Answer with one word.", "Berlin"),
    ("What is the capital of Egypt? Answer with one word.", "Cairo"),
    ("Which planet do we live on? Answer with one word.", "Earth"),
]
FIXED_NUMBERS = [
    ("How many days are in a week? Answer with the number only.", "7"),
    ("How many legs does a spider have? Answer with the number only.", "8"),
    ("How many months are in a year? Answer with the number only.", "12"),
]
INSTRUCTIONS = [
    ("Reply with only the word YES, in capital letters, and nothing else.", r"^\W*YES\W*$"),
    ("Write the numbers from 1 to 5, separated by commas, and nothing else.", r"1\s*,\s*2\s*,\s*3\s*,\s*4\s*,\s*5"),
    ("Answer with exactly one word: which animal says 'meow'?", r"(?i)\bcats?\b"),
]


def items(seed: int = 0) -> list[dict[str, Any]]:
    """The canary items: {id, category, messages, grader, answer | pattern}. Deterministic for a seed."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []

    def add(cat: str, q: str, grader: str, **ans: Any) -> None:
        out.append({"id": f"canary:{cat}:{len([o for o in out if o['category'] == cat])}", "category": cat,
                    "messages": [{"role": "user", "content": q}], "grader": grader, **ans})
    for _ in range(3):
        a, b = rng.randint(12, 89), rng.randint(12, 89)
        add("arithmetic", f"What is {a} + {b}? Answer with the number only.", "math", answer=str(a + b))
    for _ in range(2):
        a, b = rng.randint(50, 99), rng.randint(11, 49)
        add("arithmetic", f"What is {a} - {b}? Answer with the number only.", "math", answer=str(a - b))
    for _ in range(2):
        a, b = rng.randint(3, 12), rng.randint(3, 12)
        add("arithmetic", f"What is {a} times {b}? Answer with the number only.", "math", answer=str(a * b))
    h, m, kg = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
    add("units", f"How many minutes are in {h} hours? Answer with the number only.", "math", answer=str(60 * h))
    add("units", f"How many centimetres are in {m} metres? Answer with the number only.", "math", answer=str(100 * m))
    add("units", f"How many grams are in {kg} kilograms? Answer with the number only.", "math", answer=str(1000 * kg))
    for _ in range(2):
        n = rng.randint(100, 989)
        add("sequence", f"What number comes right after {n}? Answer with the number only.", "math", answer=str(n + 1))
    s, d = rng.randint(2, 9), rng.randint(2, 6)
    seq = ", ".join(str(s + d * k) for k in range(4))
    add("sequence", f"What is the next number in the sequence {seq}? Answer with the number only.", "math",
        answer=str(s + 4 * d))
    for _ in range(3):
        a, b = rng.sample(range(100, 999), 2)
        add("comparison", f"Which is larger, {a} or {b}? Answer with the number only.", "math", answer=str(max(a, b)))
    for q, a in FACTS:
        add("facts", q, "contains", answer=a)
    for q, a in FIXED_NUMBERS:
        add("facts", q, "math", answer=a)
    for q, p in INSTRUCTIONS:
        add("instructions", q, "regex", pattern=p)
    return out


def digest(its: Optional[list[dict]] = None) -> str:
    return hashlib.sha256(json.dumps(its or items(), sort_keys=True).encode()).hexdigest()[:16]


def protocol(template_kwargs: Optional[dict] = None):
    """Greedy, a short reply, the model's own template settings, no system prompt: deterministic."""
    from ..program.objective import Protocol

    return Protocol(template_kwargs=dict(template_kwargs or {}), sampling={}, max_new_tokens=MAX_NEW, system=None)


def run(source: Any, template_kwargs: Optional[dict] = None, *, original: bool) -> list[dict]:
    """The canary on a source (an LM, a TorchBackend or a llama-server backend). `original`: the unedited model
    (its rows are cached with their batch plan, so the edited run is paired chunk for chunk)."""
    from ..program import evaluate as EV
    from ..program.profile import Runner

    r = Runner(source, protocol(template_kwargs), "item")
    fn = EV.original_set if original else EV._run_set
    return fn(r, items(), samples=1, seed=0)


def compare(before: list[dict], after: list[dict], max_lost: int = MAX_LOST) -> dict[str, Any]:
    """Items the original got right (eligible) and lost after. holds: lost <= max_lost."""
    cats: dict[str, list[int]] = {}
    lost, gained = [], []
    eligible = 0
    by = {it["id"]: it["category"] for it in items()}
    for b, a in zip(before, after):
        was = bool(b["correct"] and b["correct"][0])
        now = bool(a["correct"] and a["correct"][0])
        c = cats.setdefault(by.get(b["id"], "?"), [0, 0])
        if was:
            eligible += 1
            c[0] += 1
            if not now:
                lost.append(b["id"])
                c[1] += 1
        elif now:
            gained.append(b["id"])
    out = {"name": NAME, "kind": "canary", "items": len(before), "eligible": eligible, "lost": len(lost),
           "lost_items": lost, "gained": len(gained), "max_lost": max_lost, "version": VERSION,
           "by_category": {k: {"eligible": v[0], "lost": v[1]} for k, v in sorted(cats.items())},
           "holds": len(lost) <= max_lost}
    if not out["holds"]:
        out["problems"] = [f"loses {len(lost)} of {eligible} canary items the original got right ("
                           + ", ".join(lost[:6]) + ")"]
    return out
