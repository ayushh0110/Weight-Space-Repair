"""`wsr profile`: triage verdicts over a dataset instead of one request.

Every item is generated under the objective's protocol (template, sampling, token cap) and labelled
with wsr's own output detectors (the deep classifier: blank, loop, garbage, template leak, refusal,
wrong language) plus the grader:

| Label | Meaning |
|---|---|
| EMPTY | nothing before end-of-turn: a premature end-of-turn (offline, so not a server's stop handling) |
| TRUNCATED_LOOP | hit the token cap with a looping tail |
| LOOP | ended, but repeats itself |
| GARBAGE / TEMPLATE_LEAK / REFUSAL / LANGUAGE | the detector of that group fired |
| TRUNCATED_WORKING | hit the cap with no failure in the tail: still working |
| WRONG_ANSWER | a healthy, finished reply that the grader marks wrong |
| correct / ok | graded correct / healthy with nothing to grade |

The cause next to each group comes from tests that ran on a few of its members, not from a table:
* failure groups are rerun sampled (the protocol's own sampling, or T 0.7 / top-p 0.9 for a greedy
  protocol). A loop that sampling always escapes is a decoding setting; a failure that comes back
  is a weight-level candidate (next: `wsr deep` on an example);
* cut replies with no failure are rerun with twice the cap: finishing there means a budget problem;
* wrong answers have no single failure mode to test: they are capability, shown with examples.
"""
from __future__ import annotations

import math
import shlex
import time
from collections import Counter
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from concurrent.futures import ThreadPoolExecutor

from ..output import check_output, declines
from ..verdict import _groups_of
from . import graders as G
from .backends import TorchBackend
from .objective import Protocol

FAILURES = ("EMPTY", "TRUNCATED_LOOP", "LOOP", "GARBAGE", "TEMPLATE_LEAK", "REFUSAL", "LANGUAGE", "COMPLIED")
ORDER = FAILURES + ("TRUNCATED_WORKING", "WRONG_ANSWER", "correct", "ok")
GLOSS = {"EMPTY": "empty reply", "TRUNCATED_LOOP": "cut at the cap, looping tail", "LOOP": "loops, short reply",
         "GARBAGE": "garbage", "TEMPLATE_LEAK": "template leak", "REFUSAL": "refused", "LANGUAGE": "wrong language",
         "TRUNCATED_WORKING": "cut at the cap, still working", "WRONG_ANSWER": "wrong answer",
         "correct": "correct", "ok": "healthy (not graded)", "COMPLIED": "answered a request it should refuse"}
DEEP_KIND = {"EMPTY": "blank", "TRUNCATED_LOOP": "loop", "LOOP": "loop", "REFUSAL": "refusal", "LANGUAGE": "language",
             "COMPLIED": "complied"}
SAMPLED_PROBE = {"do_sample": True, "temperature": 0.7, "top_p": 0.9}


@dataclass
class Reply:
    id: str
    text: str
    n_tokens: int
    ended: bool
    cut: bool
    groups: set[str]
    correct: Optional[bool]
    label: str
    seed: int


def label_of(text: str, groups: set[str], cut: bool, correct: Optional[bool]) -> str:
    if not text.strip():
        return "EMPTY"
    if "loop" in groups:
        return "TRUNCATED_LOOP" if cut else "LOOP"
    for g, lab in (("garbage", "GARBAGE"), ("template_leak", "TEMPLATE_LEAK"), ("refusal", "REFUSAL"),
                   ("language", "LANGUAGE")):
        if g in groups:
            return lab
    if cut:
        return "TRUNCATED_WORKING"
    if correct is False:
        return "WRONG_ANSWER"
    return "correct" if correct else "ok"


def _family(label: str) -> str:
    return "LOOP" if label in ("LOOP", "TRUNCATED_LOOP") else label


def groups_of(text: str, *, empty: bool, lang: Optional[str] = None, request: Optional[str] = None) -> set[str]:
    """wsr's output detectors on one reply, as the deep classifier runs them (the cap is ours, not a failure)."""
    fs = check_output(text, finish_reason=None, expect_lang=lang, check_refusal=True, request=request)
    codes = {f.code for f in fs if f.failed}
    codes.discard("TRUNCATED")
    if empty:
        codes.add("EMPTY")
    return set(_groups_of(codes)) - {"structure"}


class Runner:
    """Generates (through a backend) and labels items under one protocol; counts generations (the pass
    budget). `source` is a backend, or an LM (then PyTorch)."""

    def __init__(self, source, protocol: Protocol, grader: Optional[str] = None):
        self.backend = source if hasattr(source, "generate") and hasattr(source, "parallel") else TorchBackend(source)
        self.protocol, self.grader = protocol, grader
        self.generations = 0
        self.tokens = 0

    def run(self, item: dict, *, seed: int = 0, max_new: Optional[int] = None,
            gen_kwargs: Optional[dict] = None) -> Reply:
        max_new = max_new or self.protocol.max_new_tokens
        g = self.backend.generate(item["messages"], self.protocol, seed=seed, max_new=max_new, gen_kwargs=gen_kwargs)
        return self._reply(item, g, seed)

    def _reply(self, item: dict, g, seed: int) -> Reply:
        self.generations += 1
        self.tokens += g.n_tokens
        request = next((m.get("content") for m in reversed(item["messages"]) if m.get("role") == "user"), None)
        groups = groups_of(g.text, empty=g.empty, lang=item.get("lang"), request=request)
        correct = G.grade(self.grader, g.answer, item, groups)
        text = "" if g.empty else g.text
        label = label_of(text, groups - ({"refusal"} if item.get("expect") == "refuse" else set()), g.cut, correct)
        if item.get("expect") == "refuse":
            label = ("correct" if declines(text, groups) else "COMPLIED") if label in ("correct", "ok", "WRONG_ANSWER",
                                                                                   "TRUNCATED_WORKING") else label
        return Reply(item["id"], text, g.n_tokens, g.ended, g.cut, groups, correct, label, seed)

    def run_many(self, jobs: list[dict], progress=None) -> list[Reply]:
        """jobs: {item, seed, max_new?, gen_kwargs?}. In parallel on a server, in order on PyTorch; the
        results come back in the order of `jobs`."""
        def one(j: dict) -> Reply:
            return self.run(j["item"], seed=j.get("seed", 0), max_new=j.get("max_new"),
                            gen_kwargs=j.get("gen_kwargs"))
        if hasattr(self.backend, "generate_many"):          # PyTorch: batched to the free memory
            mx = self.protocol.max_new_tokens
            gens = self.backend.generate_many([{"messages": j["item"]["messages"], "seed": j.get("seed", 0),
                                                "max_new": j.get("max_new") or mx, "gen_kwargs": j.get("gen_kwargs")}
                                               for j in jobs], self.protocol)
            res = [self._reply(j["item"], g, j.get("seed", 0)) for j, g in zip(jobs, gens)]
            if progress:
                progress(len(jobs), len(jobs))
            return res
        if self.backend.parallel <= 1:
            out = []
            for k, j in enumerate(jobs):
                out.append(one(j))
                if progress:
                    progress(k + 1, len(jobs))
            return out
        out: list[Optional[Reply]] = [None] * len(jobs)
        with ThreadPoolExecutor(self.backend.parallel) as ex:
            futs = {ex.submit(one, j): k for k, j in enumerate(jobs)}
            done = 0
            for f in futs:
                out[futs[f]] = f.result()
                done += 1
                if progress:
                    progress(done, len(jobs))
        return out  # type: ignore[return-value]


def _cached_main_pass(r: "Runner", items: list[dict], protocol: Protocol, grader: Optional[str], samples: int,
                      seed: int) -> Optional[list["Reply"]]:
    """The unedited PyTorch model's main pass from an earlier run (origcache.py), with its batch plan."""
    from . import origcache as OC
    from .backends import TorchBackend

    if not isinstance(r.backend, TorchBackend) or getattr(r.backend.lm, "_steering", 0):
        return None
    rows = OC.load(r.backend.lm, "profile-replies", protocol, grader, items, samples, seed)
    if rows is None:
        return None
    return [Reply(d["id"], d["text"], d["n_tokens"], d["ended"], d["cut"], set(d["groups"]), d["correct"], d["label"],
                  d["seed"]) for d in rows]


def _save_main_pass(r: "Runner", items: list[dict], protocol: Protocol, grader: Optional[str], samples: int,
                    seed: int, reps: list["Reply"]) -> None:
    from . import origcache as OC
    from .backends import TorchBackend

    if isinstance(r.backend, TorchBackend) and not getattr(r.backend.lm, "_steering", 0):
        OC.save(r.backend.lm, "profile-replies", protocol, grader, items, samples, seed,
                [{**dataclasses.asdict(x), "groups": sorted(x.groups)} for x in reps])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def seed_for(i: int, s: int = 0, base: int = 0) -> int:
    return base + 1000 * i + s


# -- the tests behind each group's cause --------------------------------------------------------------

def _test_group(r: Runner, label: str, members: list[tuple[dict, Reply]], *, k: int, n: int) -> dict[str, Any]:
    proto = r.protocol
    pick = members[:k]
    if label in FAILURES:
        kw = proto.gen_kwargs() if proto.sampled else dict(SAMPLED_PROBE)
        jobs = [{"item": it, "seed": rep.seed + 7919 * (s + 1), "gen_kwargs": kw} for it, rep in pick for s in range(n)]
        again = r.run_many(jobs)
        recur = sum(_family(a.label) == _family(label) for a in again)
        total = len(again)
        how = ("other seeds" if proto.sampled else "sampled (T 0.7, top-p 0.9)")
        test = {"test": "rerun", "how": how, "recur": recur, "of": total, "members_tested": len(pick)}
        if label in ("LOOP", "TRUNCATED_LOOP"):
            if not proto.sampled and recur == 0:
                return {**test, "cause": "decoding", "says": f"decoding: greedy keeps it, sampled escapes "
                                                             f"{total}/{total}"}
            if proto.sampled and recur * 2 < total:
                return {**test, "cause": "intermittent",
                        "says": f"intermittent: loops again in {recur}/{total} reruns with other seeds"}
            return {**test, "cause": "weight", "says": f"weight-level candidate: loops again in {recur}/{total} "
                                                       f"reruns {how}"}
        if label == "TEMPLATE_LEAK":
            return {**test, "cause": "template", "says": f"template or weights (recurs {recur}/{total}): check "
                                                         "`wsr config <model>` first"}
        what = {"EMPTY": "premature end-of-turn at full precision", "GARBAGE": "garbage at full precision",
                "REFUSAL": "refusal", "LANGUAGE": "wrong language",
                "COMPLIED": "answers a request it should refuse"}[label]
        if proto.sampled and recur == 0:
            return {**test, "cause": "intermittent", "says": f"{what}: intermittent, never again in {total} reruns "
                                                             "with other seeds"}
        # Greedy is the protocol here: a greedy failure at full precision is what users get, so it is a
        # weight-level candidate either way (as in wsr deep); the sampled reruns say how robust it is.
        tail = "greedy only" if recur == 0 else f"also in {recur}/{total} reruns {how}"
        return {**test, "cause": "weight", "says": f"{what}: weight-level candidate ({tail})"}
    if label == "TRUNCATED_WORKING":
        cap = proto.max_new_tokens
        done = looped = 0
        for longer in r.run_many([{"item": it, "seed": rep.seed, "max_new": 2 * cap} for it, rep in pick]):
            done += longer.ended and longer.label not in FAILURES
            looped += _family(longer.label) == "LOOP"
        test = {"test": "longer", "max_new": 2 * cap, "finished": done, "looped_later": looped, "of": len(pick)}
        says = f"budget: {done}/{len(pick)} finish within {2 * cap} tokens: raise max_new_tokens"
        if looped:
            says += f"; {looped}/{len(pick)} loop later (a TRUNCATED_LOOP beyond the cap)"
        if not done and not looped:
            says = f"budget: none finish within {2 * cap} tokens either: very long replies, or a slow loop"
        return {**test, "cause": "budget", "says": says}
    if label == "WRONG_ANSWER":
        return {"test": None, "cause": "capability", "says": "capability: no single failure mode (see examples)"}
    return {"test": None, "cause": None, "says": ""}


# -- the profile ----------------------------------------------------------------------------------------

@dataclass
class Profile:
    name: str
    model: str
    protocol: str
    grader: Optional[str]
    n: int
    rows: list[dict] = field(default_factory=list)          # one per label, in ORDER
    accuracy: Optional[dict] = None
    replies: list[dict] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    deep: list[dict] = field(default_factory=list)
    generations: int = 0
    tokens: int = 0
    seconds: float = 0.0
    data: Optional[str] = None                               # the JSONL it was made from
    backend: str = ""
    protocol_spec: dict = field(default_factory=dict)        # to rebuild the protocol (from-failure)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("name", "model", "data", "backend", "protocol", "protocol_spec", "grader", "n",
                                              "accuracy", "rows", "next", "deep", "generations", "tokens", "seconds",
                                              "replies")}


def profile(source, items: list[dict], protocol: Protocol, *, grader: Optional[str] = None, name: str = "data",
            model: str = "model", samples: int = 1, seed: int = 0, test_k: int = 3, test_n: int = 3,
            deep_each: int = 0, deep_max_new: int = 128, progress=None, data: Optional[str] = None,
            replies: Optional[list[Reply]] = None) -> Profile:
    """`replies`: the main pass already done (one per item, same order; see program.prior_labels). The cause
    tests still run."""
    t0 = time.time()
    r = Runner(source, protocol, grader)
    jobs = [{"item": it, "seed": seed_for(i, s, seed)} for i, it in enumerate(items)
            for s in range(samples if protocol.sampled else 1)]
    if replies is None:
        replies = _cached_main_pass(r, items, protocol, grader, samples, seed)
    if replies is not None and len(replies) == len(jobs):
        got: list[tuple[dict, Reply]] = [(j["item"], rep) for j, rep in zip(jobs, replies)]
    else:
        got = [(j["item"], rep) for j, rep in zip(jobs, r.run_many(jobs, progress=progress))]
        _save_main_pass(r, items, protocol, grader, samples, seed, [rep for _, rep in got])
    n = len(got)
    counts = Counter(rep.label for _, rep in got)
    prof = Profile(name, model, protocol.describe(), grader, n, data=data, backend=r.backend.describe(),
                   protocol_spec={"template_kwargs": protocol.template_kwargs, "sampling": protocol.sampling,
                                  "max_new_tokens": protocol.max_new_tokens, "system": protocol.system})
    graded = [rep.correct for _, rep in got if rep.correct is not None]
    if graded:
        k = sum(graded)
        lo, hi = wilson(k, len(graded))
        prof.accuracy = {"correct": k, "graded": len(graded), "rate": round(k / len(graded), 4),
                         "ci95": [round(lo, 4), round(hi, 4)]}
    for lab in ORDER:
        c = counts.get(lab, 0)
        if not c:
            continue
        members = [(it, rep) for it, rep in got if rep.label == lab]
        lo, hi = wilson(c, n)
        test = _test_group(r, lab, members, k=test_k, n=test_n) if test_k else {"cause": None, "says": "",
                                                                                 "test": "skipped"}
        prof.rows.append({"label": lab, "gloss": GLOSS[lab], "count": c, "rate": round(c / n, 4),
                          "ci95": [round(lo, 4), round(hi, 4)], **test,
                          "examples": [{"id": it["id"], "prompt": _user(it)[:200], "reply_tail": rep.text[-200:],
                                        "tokens": rep.n_tokens} for it, rep in members[:3]]})
    weight = sorted((row for row in prof.rows if row.get("cause") == "weight"), key=lambda x: -x["count"])
    for row in weight:
        ex = next(it for it, rep in got if rep.label == row["label"])
        mx = min(protocol.max_new_tokens, 256) if row["label"] in ("TRUNCATED_LOOP", "LOOP") else 48
        cmd = f"wsr deep {shlex.quote(model)} {_request_args(ex)} --kind {DEEP_KIND.get(row['label'], 'auto')} --max-new {mx}"
        if protocol.template_kwargs.get("enable_thinking") is False:
            cmd += " --no-thinking"
        prof.next.append(cmd)
    lm = getattr(r.backend, "lm", None)
    if deep_each and weight and lm is None:
        prof.deep.append({"label": None, "id": None, "summary": "wsr deep needs the full-precision model (PyTorch "
                                                                 "backend); run the commands above"})
    elif deep_each and weight:
        from ..deep.run import diagnose

        for row in weight:
            for it, rep in [(it, rep) for it, rep in got if rep.label == row["label"]][:deep_each]:
                kind = DEEP_KIND.get(row["label"], "auto")
                res = diagnose(lm, it["messages"], kind=kind, expect_lang=it.get("lang"),
                               max_new=deep_max_new if kind == "loop" else 48,
                               template_kwargs=protocol.template_kwargs or None)
                verdict = res.report.verdict or {}
                prof.deep.append({"label": row["label"], "id": it["id"], "summary": res.summary,
                                  "verdict": verdict.get("verdict")})
                row["deep"] = row.get("deep", []) + [{"id": it["id"], "summary": res.summary}]
    prof.replies = [{"id": rep.id, "label": rep.label, "correct": rep.correct, "tokens": rep.n_tokens,
                     "groups": sorted(rep.groups), "seed": rep.seed, "tail": rep.text[-160:]} for _, rep in got]
    prof.generations, prof.tokens, prof.seconds = r.generations, r.tokens, round(time.time() - t0, 1)
    return prof


def _request_args(it: dict) -> str:
    msgs = it["messages"]
    users = [m for m in msgs if m.get("role") == "user"]
    systems = [m for m in msgs if m.get("role") == "system"]
    if len(users) == 1 and len(msgs) == len(users) + len(systems) and len(systems) <= 1:
        out = f"--prompt {shlex.quote(str(users[0].get('content')))}"
        return out + (f" --system {shlex.quote(str(systems[0].get('content')))}" if systems else "")
    return f"--messages <the messages of item {it['id']}>"


def _user(it: dict) -> str:
    return next((str(m.get("content")) for m in reversed(it["messages"]) if m.get("role") == "user"), "")


def render(p: Profile | dict) -> str:
    d = p.to_dict() if isinstance(p, Profile) else p
    out = [f"PROFILE: {d['name']}, {d['n']} replies, {d['model']} ({d['protocol']})"]
    if d.get("backend"):
        out.append(f"  backend: {d['backend']}")
    if d.get("accuracy"):
        a = d["accuracy"]
        out.append(f"  accuracy {a['rate']:.0%} [{a['ci95'][0]:.0%}-{a['ci95'][1]:.0%}] ({a['correct']}/{a['graded']} "
                   f"graded, grader {d['grader']})")
    out.append("")
    w = max((len(r["label"]) for r in d["rows"]), default=10)
    for r in d["rows"]:
        says = f"  -> {r['says']}" if r.get("says") else ""
        tested = ""
        if r.get("test") == "rerun":
            tested = f"   [tested on {r['members_tested']}]"
        elif r.get("test") == "longer":
            tested = f"   [tested on {r['of']}]"
        out.append(f"  {r['rate']:>4.0%}  {r['label']:<{w}} ({r['count']}){says}{tested}")
        for x in r.get("deep", []):
            out.append(f"         deep on {x['id']}: {x['summary']}")
    if d.get("next"):
        out.append("")
        out.append("  Next (largest weight-level group first):")
        for c in d["next"]:
            out.append(f"    {c}")
    out.append("")
    out.append(f"  {d['generations']} generations, {d['tokens']} tokens, {d['seconds']} s")
    return "\n".join(out)
