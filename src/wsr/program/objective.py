"""The goal file: what to improve, on which data, how it is served, how it is judged, what must not break.

```yaml
name: llama-math
kind: capability                 # repair | capability
model: ./Llama-3.2-1B-Instruct
protocol:
  template_kwargs: {}
  sampling: {}                   # empty = greedy; else temperature / top_p / top_k / repetition_penalty
  max_new_tokens: 512
  system: null
practice: data/math_practice.jsonl    # the agent may read these
heldout:  data/math_heldout.jsonl     # validation: judges the candidates; every look is logged ("validation" works too)
confirmation: data/math_confirm.jsonl # optional: one evaluation of one frozen candidate, then it is used up
grader: math                     # exact | contains | regex:<p> | math | detector:<group>
judge: {alpha: 0.05, min_gain: 0.05}  # exact sign test below alpha, and at least this gain (5 points)
guardrails:
  - {suite: wsr_regression}      # the 17 built-in prompts, safety included (drift, new failures, refusals)
  - {suite: semantic_regression_canary}   # deterministic tasks the original gets right must stay right
  - {data: data/general.jsonl, grader: contains, max_drop: 0.02}
budget: {practice_evals: 10, verifications: 6, usd: 5}   # the agent's limits; the --agent-* flags override them
```

Data files are JSON Lines. Each row has the request as `messages` (a chat array), or `prompt` /
`problem` / `question` / `input` (one user turn); and optionally `answer` (else `\\boxed{}` in
`solution` is used), `pattern` for regex grading, and `id`. Paths are relative to the goal file.
JSON goal files work everywhere; YAML needs PyYAML.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import graders as G

PROMPT_KEYS = ("prompt", "problem", "question", "input", "instruction")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
KINDS = ("repair", "capability")
SUITES = ("wsr_regression", "semantic_regression_canary")


@dataclass
class Protocol:
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 512
    system: Optional[str] = None

    @property
    def sampled(self) -> bool:
        return float(self.sampling.get("temperature", 0) or 0) > 0

    def gen_kwargs(self) -> dict[str, Any]:
        """For transformers' generate. presence_penalty has no transformers equivalent: it is reported, not
        silently applied as something else."""
        if not self.sampled:
            kw: dict[str, Any] = {}
        else:
            kw = {"do_sample": True, **{k: self.sampling[k] for k in ("temperature", "top_p", "top_k", "min_p")
                                        if k in self.sampling}}
        if "repetition_penalty" in self.sampling:
            kw["repetition_penalty"] = self.sampling["repetition_penalty"]
        return kw

    def describe(self) -> str:
        bits = []
        if self.template_kwargs:
            bits.append(", ".join(f"{k}={v}" for k, v in self.template_kwargs.items()))
        bits.append("greedy" if not self.sampled else
                    "sampled " + " ".join(f"{k} {v}" for k, v in self.sampling.items()))
        bits.append(f"max {self.max_new_tokens}")
        return ", ".join(bits)


@dataclass
class Guardrail:
    suite: Optional[str] = None           # "wsr_regression"
    data: Optional[str] = None
    grader: Optional[str] = None
    max_drop: float = 0.02
    limit: Optional[int] = None
    max_drift: float = 0.15               # wsr_regression only
    min_agreement: float = 0.85
    allow_safety_loss: bool = False       # wsr_regression only: a patch may stop refusing the suite's harmful requests
    max_lost: int = 1                     # semantic_regression_canary only: items the original got right that may fail

    @property
    def name(self) -> str:
        return self.suite or Path(self.data or "?").stem


@dataclass
class Objective:
    name: str
    model: Optional[str] = None
    kind: str = "capability"
    protocol: Protocol = field(default_factory=Protocol)
    practice: Optional[str] = None
    heldout: Optional[str] = None             # the validation set
    grader: Optional[str] = None
    guardrails: list[Guardrail] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    confirmation: Optional[str] = None        # used once, for one frozen candidate (program/confirm.py)
    judge: dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None           # where it was loaded from (paths resolve against its folder)

    def resolve(self, p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        q = Path(p).expanduser()
        return q if q.is_absolute() or self.path is None else (self.path.parent / q)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("path")
        d["guardrails"] = [{k: v for k, v in asdict(g).items() if v is not None} for g in self.guardrails]
        return d


BUDGET_KEYS = ("practice_evals", "verifications", "usd")


def _budget(raw: Any) -> dict[str, Any]:
    """The agent's limits from the goal file: practice_eval calls, full verifications, API spend in USD.
    `passes` is the older name of practice_evals."""
    b = dict(raw or {})
    if "passes" in b:
        b.setdefault("practice_evals", b["passes"])
        del b["passes"]
    bad = set(b) - set(BUDGET_KEYS)
    if bad:
        raise ValueError(f"unknown budget field(s): {', '.join(sorted(bad))} (known: {', '.join(BUDGET_KEYS)})")
    for k, v in b.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"budget.{k} must be a number that is not negative")
    return b


def _judge(raw: Any) -> dict[str, Any]:
    """The acceptance policy's settings: alpha for the exact sign test, min_gain for the smallest gain worth
    shipping (a share of the metric, 0.05 = 5 points)."""
    j = dict(raw or {})
    bad = set(j) - {"alpha", "min_gain"}
    if bad:
        raise ValueError(f"unknown judge field(s): {', '.join(sorted(bad))} (known: alpha, min_gain)")
    if "alpha" in j and not (isinstance(j["alpha"], (int, float)) and 0 < j["alpha"] <= 0.1):
        raise ValueError("judge.alpha must be a number in (0, 0.1]")
    if "min_gain" in j and not (isinstance(j["min_gain"], (int, float)) and 0 <= j["min_gain"] <= 1):
        raise ValueError("judge.min_gain must be a number in [0, 1] (a share of the metric: 0.05 = 5 points)")
    return j


def _read_text_obj(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ValueError(f"{path} is YAML: pip install pyyaml, or write the goal file as JSON") from e
        d = yaml.safe_load(text)
    else:
        try:
            d = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not valid JSON ({e.msg}, line {e.lineno})") from None
    if not isinstance(d, dict):
        raise ValueError(f"{path}: the goal file must be a mapping")
    return d


def from_dict(d: dict[str, Any], path: Optional[Path] = None) -> Objective:
    d = dict(d)
    if "validation" in d:                          # the three-set name for the held-out set
        if d.get("heldout") and d["heldout"] != d["validation"]:
            raise ValueError("give the held-out set once: heldout or validation (they are the same set)")
        d["heldout"] = d.pop("validation")
    unknown = set(d) - {"name", "model", "kind", "protocol", "practice", "heldout", "confirmation", "grader",
                        "guardrails", "budget", "judge"}
    if unknown:
        raise ValueError(f"unknown field(s) in the goal file: {', '.join(sorted(unknown))}")
    if not d.get("name"):
        raise ValueError("the goal file needs a name")
    kind = d.get("kind", "capability")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    p = d.get("protocol") or {}
    bad = set(p) - {"template_kwargs", "sampling", "max_new_tokens", "system"}
    if bad:
        raise ValueError(f"unknown protocol field(s): {', '.join(sorted(bad))}")
    samp = p.get("sampling") or {}
    bad = set(samp) - set(SAMPLING_KEYS)
    if bad:
        raise ValueError(f"unknown sampling field(s): {', '.join(sorted(bad))} (known: {', '.join(SAMPLING_KEYS)})")
    proto = Protocol(template_kwargs=dict(p.get("template_kwargs") or {}), sampling=dict(samp),
                     max_new_tokens=int(p.get("max_new_tokens", 512)), system=p.get("system"))
    G.check_name(d.get("grader"))
    rails = []
    for g in d.get("guardrails") or []:
        if not isinstance(g, dict):
            raise ValueError("each guardrail is a mapping: {suite: wsr_regression} or {data: ..., grader: ...}")
        bad = set(g) - {"suite", "data", "grader", "max_drop", "limit", "max_drift", "min_agreement",
                        "allow_safety_loss", "max_lost"}
        if bad:
            raise ValueError(f"unknown guardrail field(s): {', '.join(sorted(bad))}")
        if g.get("suite") and g["suite"] not in SUITES:
            raise ValueError(f"unknown guardrail suite {g['suite']!r}: the built-in ones are {', '.join(SUITES)}")
        if not g.get("suite") and not g.get("data"):
            raise ValueError("a guardrail needs suite: wsr_regression or data: <file>")
        if g.get("data"):
            G.check_name(g.get("grader") or "detector:any")
        rails.append(Guardrail(**g))
    return Objective(name=str(d["name"]), model=d.get("model"), kind=kind, protocol=proto, practice=d.get("practice"),
                     heldout=d.get("heldout"), grader=d.get("grader"), guardrails=rails,
                     budget=_budget(d.get("budget")), confirmation=d.get("confirmation"), judge=_judge(d.get("judge")),
                     path=path)


def load(path: str | Path) -> Objective:
    path = Path(path)
    return from_dict(_read_text_obj(path), path=path.resolve())


# -- data ------------------------------------------------------------------------------------------------

def _row_messages(row: dict[str, Any], system: Optional[str]) -> list[dict]:
    if isinstance(row.get("messages"), list):
        msgs = list(row["messages"])
    else:
        text = next((row[k] for k in PROMPT_KEYS if isinstance(row.get(k), str) and row[k].strip()), None)
        if text is None:
            raise ValueError(f"a row needs messages or one of {', '.join(PROMPT_KEYS)}")
        msgs = [{"role": "user", "content": text}]
    if system and not any(m.get("role") == "system" for m in msgs):
        msgs = [{"role": "system", "content": system}] + msgs
    return msgs


def load_items(path: str | Path, *, system: Optional[str] = None, limit: Optional[int] = None) -> list[dict]:
    """Rows as {id, messages, answer?, pattern?}. A row with only `solution` gets its \\boxed{} answer."""
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not JSON ({e.msg})") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{n}: each line must be a JSON object")
            try:
                msgs = _row_messages(row, system)
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
            ans = row.get("answer", row.get("target"))
            if ans is None and isinstance(row.get("solution"), str):
                ans = G.boxed(row["solution"])
            out.append({"id": str(row.get("id", row.get("unique_id", f"{Path(path).stem}:{n}"))), "messages": msgs,
                        **({"answer": ans} if ans is not None else {}),
                        **({"pattern": row["pattern"]} if "pattern" in row else {}),
                        **{k: row[k] for k in ("expect", "lang") if k in row}})
            if limit and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


def _key(item: dict) -> str:
    return json.dumps([m.get("content") for m in item["messages"] if m.get("role") == "user"], ensure_ascii=False)


def _control_char(t: str) -> str:
    """The first control character other than newline, tab and carriage return, or ''."""
    return next((ch for ch in t if ord(ch) < 32 and ch not in "\n\t\r"), "")


def check(obj: Objective) -> dict[str, Any]:
    """What `wsr objective check` reports: files, row counts, gradable rows, practice/held-out overlap."""
    problems, info = [], {}
    sets = {}
    for split in ("practice", "heldout", "confirmation"):
        p = obj.resolve(getattr(obj, split))
        if p is None:
            info[split] = None
            continue
        if not p.exists():
            problems.append(f"{split}: {p} does not exist")
            continue
        try:
            items = load_items(p, system=obj.protocol.system)
        except ValueError as e:
            problems.append(str(e))
            continue
        sets[split] = items
        gradable = sum(G.grade(obj.grader, "", it, set()) is not None for it in items) if obj.grader else 0
        info[split] = {"file": str(p), "rows": len(items), "gradable": gradable}
        if obj.grader and not obj.grader.startswith(("detector", "regex")) and gradable < len(items):
            problems.append(f"{split}: {len(items) - gradable}/{len(items)} rows have no answer for grader "
                            f"{obj.grader}")
    if "confirmation" in sets:
        cp = obj.resolve(obj.confirmation)
        from .confirm import used
        u = used(cp) if cp is not None else None
        if u is not None:
            problems.append(f"the confirmation set was already used ({u.get('time')}, goal {u.get('goal')}): it is "
                            "evaluated once; give fresh data for a new confirmation")
        for other in ("practice", "heldout"):
            if other in sets:
                both = {_key(i) for i in sets[other]} & {_key(i) for i in sets["confirmation"]}
                if both:
                    problems.append(f"{len(both)} prompt(s) are in both {other} and confirmation")
    if "practice" in sets and "heldout" in sets:
        shared = {_key(i) for i in sets["practice"]} & {_key(i) for i in sets["heldout"]}
        info["overlap"] = len(shared)
        if shared:
            problems.append(f"{len(shared)} prompt(s) are in both practice and held-out: the held-out score would "
                            "not be held out")
        pp, hp = obj.resolve(obj.practice), obj.resolve(obj.heldout)
        if pp and hp and pp.parent.resolve() == hp.parent.resolve():
            info["note"] = ("practice and held-out sit in the same folder: when an agent gets read access to the "
                            "practice folder (wsr program), keep the held-out file elsewhere")
    if not obj.heldout:
        problems.append("no held-out file: wsr evaluate can only score the practice set, which an agent may "
                        "have fitted")
    # wsr's harm probe is calibrated on its built-in refusal pools: a validation or confirmation set drawn from them
    # would judge probe-gated steering on prompts it was fitted on.
    from ..deep import refusal_pools as RP
    pool = RP.texts()
    for split in ("heldout", "confirmation"):
        if split in sets:
            n = sum(RP.norm(it["messages"][-1].get("content", "")) in pool for it in sets[split])
            if n:
                problems.append(f"{n} {split} prompt(s) are in wsr's built-in refusal pools (wsr-refusal), which the "
                                "harm probe is calibrated on: use them for practice, not for judging")
    # A control character in a prompt is almost always a JSON escape by accident: a backslash-b in a JSON
    # string (as in a LaTeX boxed command written with one backslash) is a backspace.
    texts = [("protocol.system", obj.protocol.system or "")]
    for split, items in sets.items():
        texts += [(f"{split} {it['id']}", str(m.get("content") or "")) for it in items[:2000] for m in it["messages"]]
    bad = [(w, t) for w, t in texts if _control_char(t)]
    if bad:
        w, t = bad[0]
        problems.append(f"{len(bad)} prompt(s) contain control characters (first: {w}, {_control_char(t)!r}); in "
                        "JSON a backslash must be doubled, e.g. " + json.dumps("\\boxed{}"))
    for g in obj.guardrails:
        if g.data:
            gp = obj.resolve(g.data)
            if not gp or not gp.exists():
                problems.append(f"guardrail {g.name}: {gp} does not exist")
    return {"name": obj.name, "ok": not problems, "problems": problems, **info,
            "grader": obj.grader, "protocol": obj.protocol.describe(),
            "guardrails": [g.name for g in obj.guardrails]}


TEMPLATE = {
    "name": "my-objective",
    "kind": "capability",
    "model": "./model",
    "protocol": {"template_kwargs": {}, "sampling": {}, "max_new_tokens": 512, "system": None},
    "practice": "data/practice.jsonl",
    "heldout": "data/heldout.jsonl",
    "grader": "math",
    "confirmation": None,
    "judge": {"alpha": 0.05, "min_gain": 0.05},
    "guardrails": [{"suite": "wsr_regression"}, {"suite": "semantic_regression_canary"}],
    "budget": {"practice_evals": 10, "verifications": 6, "usd": 5},
}


def write_template(out: Path, **over: Any) -> Path:
    d = json.loads(json.dumps(TEMPLATE))
    for k, v in over.items():
        if v is not None:
            d[k] = v
    from_dict(d)                                  # the template must load
    if out.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ValueError("pip install pyyaml, or name the goal file .json") from e
        out.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        out.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def split_file(src: Path, practice: Path, heldout: Path, *, heldout_share: float = 0.5, seed: int = 0,
               limit: Optional[int] = None) -> dict[str, int]:
    """Shuffle a JSONL file (seeded) and split it; rows are copied unchanged."""
    import random

    rows = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    k = int(round(len(rows) * heldout_share))
    heldout.parent.mkdir(parents=True, exist_ok=True)
    practice.parent.mkdir(parents=True, exist_ok=True)
    heldout.write_text("\n".join(rows[:k]) + "\n", encoding="utf-8")
    practice.write_text("\n".join(rows[k:]) + "\n", encoding="utf-8")
    return {"practice": len(rows) - k, "heldout": k}
