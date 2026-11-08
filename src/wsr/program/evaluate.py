"""`wsr evaluate`: the judge for an objective.

The original and the patched model answer the same items under the objective's protocol, with the
same seed per item and sample (common random numbers), so the per-item difference measures the
patch and not the sampling noise. Reported:

* the score of each (accuracy with the grader; without one, the share of replies with no failure);
* the paired difference, its standard error, wins / losses / ties per item;
* the failure-group rates of the profile before and after;
* every guardrail: the built-in regression suite (new failures, drift, top-token agreement, harmful
  requests still refused) and data guardrails (score drop within `max_drop`);
* a verdict: **improved** only when the gain is above 2 standard errors and every guardrail holds.

Every evaluation on the held-out split is appended to `<goal>.heldout_log.jsonl` next to the goal
file, and the report says how many times the held-out set has been looked at: a score chosen after
many looks is optimistic.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from collections import Counter
from typing import Any, Optional

from .objective import Guardrail, Objective, Protocol, load_items
from .profile import FAILURES, Runner, seed_for

Z = 2.0


def _score(rep_correct: Optional[bool], label: str, graded: bool) -> float:
    if graded:
        return float(bool(rep_correct))
    return float(label not in FAILURES and label != "TRUNCATED_WORKING")


def _run_set(r: Runner, items: list[dict], *, samples: int, seed: int, progress=None, tag: str = "") -> list[dict]:
    k = samples if r.protocol.sampled else 1
    jobs = [{"item": it, "seed": seed_for(i, s, seed)} for i, it in enumerate(items) for s in range(k)]
    reps = r.run_many(jobs, progress=(lambda i, n: progress(tag, i, n)) if progress else None)
    out = []
    for i, it in enumerate(items):
        rs = reps[i * k:(i + 1) * k]
        out.append({"id": it["id"], "labels": [x.label for x in rs], "correct": [x.correct for x in rs],
                    "tail": rs[0].text[-160:]})
    return out


@contextlib.contextmanager
def _patched(source, patch, patched_source):
    """The patched model as a source: a second backend (its own GGUF), or the PyTorch model with the patch
    applied in memory for the duration."""
    if patched_source is not None:
        yield patched_source
        return
    from ..surgery.edits import applied

    lm = _torch_lm(source)
    if lm is None:
        raise ValueError("this backend cannot apply a patch: give the patched model's own backend (--patched-gguf), "
                         "or use --backend torch")
    with applied(lm, patch):
        yield source


def _torch_lm(source):
    """The PyTorch model behind a source (an LM, or a TorchBackend), else None (a server)."""
    from .backends import ServerBackend, TorchBackend

    if isinstance(source, TorchBackend):
        return source.lm
    if isinstance(source, ServerBackend):
        return None
    return source


def _metric(rows: list[dict]) -> tuple[list[float], bool]:
    graded = any(c is not None for r in rows for c in r["correct"])
    vals = []
    for r in rows:
        ss = [_score(c, lab, graded) for c, lab in zip(r["correct"], r["labels"]) if not graded or c is not None]
        vals.append(sum(ss) / len(ss) if ss else float("nan"))
    return vals, graded


def _paired(a: list[float], b: list[float]) -> dict[str, Any]:
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    n = len(pairs)
    if not n:
        return {"n": 0}
    d = [y - x for x, y in pairs]
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "before": round(sum(x for x, _ in pairs) / n, 4), "after": round(sum(y for _, y in pairs) / n, 4),
            "diff": round(mean, 4), "se": round(se, 4), "z": round(mean / se, 2) if se > 0 else None,
            "wins": sum(v > 0 for v in d), "losses": sum(v < 0 for v in d), "ties": sum(v == 0 for v in d)}


def _rates(rows: list[dict]) -> dict[str, float]:
    c = Counter(lab for r in rows for lab in r["labels"])
    n = sum(c.values()) or 1
    return {k: round(v / n, 4) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}


def _verdict(p: dict[str, Any], rails_ok: bool) -> str:
    if not p.get("n"):
        return "nothing to compare"
    diff, se = p["diff"], p["se"]
    clear_gain = diff > Z * se and diff > 0
    clear_loss = diff < -Z * se and diff < 0
    if clear_gain and rails_ok:
        return "improved"
    if clear_gain:
        return "improved on the task, but a guardrail failed"
    if clear_loss:
        return "worse"
    return "no clear change"


class HeldoutLog:
    def __init__(self, obj: Objective):
        self.path = (obj.path.parent / f"{obj.path.stem}.heldout_log.jsonl") if obj.path else None

    def count(self) -> int:
        if not self.path or not self.path.exists():
            return 0
        return sum(1 for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip())

    def add(self, entry: dict[str, Any]) -> int:
        if not self.path:
            return 0
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return self.count()


def _guard_data(source, obj: Objective, g: Guardrail, patch, *, samples: int, seed: int,
                patched_source=None) -> dict[str, Any]:
    items = load_items(obj.resolve(g.data), system=obj.protocol.system, limit=g.limit)
    r = Runner(source, obj.protocol, g.grader or "detector:any")
    before = _run_set(r, items, samples=samples, seed=seed)
    with _patched(source, patch, patched_source) as ps:
        r2 = Runner(ps, obj.protocol, g.grader or "detector:any")
        after = _run_set(r2, items, samples=samples, seed=seed)
    a, _ = _metric(before)
    b, _ = _metric(after)
    p = _paired(a, b)
    drop = -p.get("diff", 0.0)
    return {"name": g.name, "kind": "data", "grader": g.grader or "detector:any", **p, "max_drop": g.max_drop,
            "holds": drop <= g.max_drop, "generations": r.generations + r2.generations}


def _guard_suite(source, obj: Objective, g: Guardrail, patch, known: Optional[dict] = None) -> dict[str, Any]:
    """The regression suite under the patch. `known`: the same suite already measured on the same patch with
    the same template (the playbook's / agent's verification), reused as is: the computation is deterministic,
    so running it again would give the same numbers."""
    from ..surgery import regress as R
    from ..surgery.edits import applied

    need = ("prompts", "unchanged_replies", "mean_delta_nll", "top1_agreement", "safety_refused_before",
            "new_failures", "safety_lost")
    if known is not None and all(k in known for k in need):
        probs = R.Budget(max_delta_nll=g.max_drift, min_top1_agreement=g.min_agreement).problems(known)
        return {"name": g.name, "kind": "suite", "holds": not probs, "problems": probs,
                **{k: known[k] for k in need[:5]}, "new_failures": known["new_failures"],
                "safety_lost": known["safety_lost"], "reused": "the verification's run of the same suite and patch"}
    lm = _torch_lm(source)
    if lm is None:
        return {"name": g.name, "kind": "suite", "holds": None,
                "problems": ["not run: the regression suite measures drift on the full-precision model (PyTorch "
                             "backend)"]}
    tk = obj.protocol.template_kwargs or None
    base = R.baseline(lm, template_kwargs=tk)
    with applied(lm, patch):
        reg = R.compare(lm, base, template_kwargs=tk)
    probs = R.Budget(max_delta_nll=g.max_drift, min_top1_agreement=g.min_agreement).problems(reg)
    return {"name": g.name, "kind": "suite", "holds": not probs, "problems": probs,
            **{k: reg[k] for k in ("prompts", "unchanged_replies", "mean_delta_nll", "top1_agreement",
                                   "safety_refused_before")},
            "new_failures": reg["new_failures"], "safety_lost": reg["safety_lost"]}


def evaluate(source, obj: Objective, *, patch=None, patch_name: str = "", split: str = "heldout",
             limit: Optional[int] = None, samples: int = 1, seed: int = 0, guardrails: bool = True,
             log: bool = True, progress=None, patched_source=None, guard_source=None,
             suite_result: Optional[dict] = None) -> dict[str, Any]:
    """source: an LM / backend for the original model; patched_source: the patched model's own backend (else
    the patch is applied to the PyTorch model in memory)."""
    t0 = time.time()
    path = obj.resolve(obj.heldout if split == "heldout" else obj.practice)
    if path is None:
        raise ValueError(f"the goal file has no {split} file")
    items = load_items(path, system=obj.protocol.system, limit=limit)
    proto: Protocol = obj.protocol
    r = Runner(source, proto, obj.grader)
    res: dict[str, Any] = {"objective": obj.name, "split": split, "items": len(items), "backend": r.backend.describe(),
                           "samples": samples if proto.sampled else 1, "protocol": proto.describe(),
                           "grader": obj.grader, "patch": patch_name or None}
    before = _run_set(r, items, samples=samples, seed=seed, progress=progress, tag="original")
    a, graded = _metric(before)
    res["metric"] = "accuracy" if graded else "share of replies with no failure"
    res["rates_before"] = _rates(before)
    if patch is None and patched_source is None:
        n = sum(not math.isnan(x) for x in a)
        res["score"] = round(sum(x for x in a if not math.isnan(x)) / n, 4) if n else None
        res["verdict"] = "scored (no patch)"
    else:
        with _patched(source, patch, patched_source) as ps:
            rp = Runner(ps, proto, obj.grader)
            after = _run_set(rp, items, samples=samples, seed=seed, progress=progress, tag="patched")
        r.generations += rp.generations
        b, _ = _metric(after)
        res["paired"] = _paired(a, b)
        res["rates_after"] = _rates(after)
        res["changed"] = [{"id": x["id"], "before": x["labels"], "after": y["labels"], "tail_after": y["tail"]}
                          for x, y in zip(before, after) if x["labels"] != y["labels"]][:20]
        rails = []
        if guardrails:
            for g in obj.guardrails:
                rails.append(_guard_suite(guard_source or source, obj, g, patch, known=suite_result) if g.suite else
                             _guard_data(source, obj, g, patch, samples=samples, seed=seed,
                                         patched_source=patched_source))
        res["guardrails"] = rails
        res["verdict"] = _verdict(res["paired"], all(g["holds"] is not False for g in rails))
        skipped = [g["name"] for g in rails if g["holds"] is None]
        if skipped and res["verdict"] == "improved":
            res["verdict"] = "improved, but guardrail(s) not run: " + ", ".join(skipped)
    res["generations"] = r.generations
    res["seconds"] = round(time.time() - t0, 1)
    if split == "heldout" and log:
        hl = HeldoutLog(obj)
        res["heldout_looks"] = hl.add({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "patch": patch_name or None,
                                       "items": len(items), "verdict": res["verdict"]})
    return res


def render(res: dict[str, Any]) -> str:
    out = [f"EVALUATE: {res['objective']} ({res['split']}, {res['items']} items x {res['samples']}, "
           f"{res['protocol']}, {res['metric']})"]
    if res.get("patch"):
        out.append(f"  patch: {res['patch']}")
    out.append(f"  backend: {res['backend']}")
    out.append("")
    if "paired" in res:
        p = res["paired"]
        if p.get("n"):
            z = f", z {p['z']}" if p.get("z") is not None else ""
            out.append(f"  {res['verdict'].upper()}: {p['before']:.1%} -> {p['after']:.1%} "
                       f"({p['diff']:+.1%} +- {p['se']:.1%} SE{z}); items better {p['wins']}, worse {p['losses']}, "
                       f"same {p['ties']}")
        else:
            out.append("  nothing to compare: no item could be scored")
        out.append("")
        labs = sorted(set(res["rates_before"]) | set(res["rates_after"]), key=lambda k: -res["rates_before"].get(k, 0))
        out.append("  label                before  after")
        for k in labs:
            out.append(f"  {k:<20} {res['rates_before'].get(k, 0):>6.0%} {res['rates_after'].get(k, 0):>6.0%}")
        if res.get("guardrails"):
            out.append("")
            out.append("  guardrails:")
            for g in res["guardrails"]:
                mark = "holds" if g["holds"] else ("NOT RUN" if g["holds"] is None else "FAILS")
                if g["kind"] == "suite" and g["holds"] is None:
                    det = g["problems"][0]
                elif g["kind"] == "suite":
                    det = "; ".join(g["problems"]) or (f"{g['unchanged_replies']}/{g['prompts']} replies unchanged, "
                                                       f"drift {g['mean_delta_nll']}, agreement {g['top1_agreement']}")
                else:
                    det = (f"{g.get('before', 0):.1%} -> {g.get('after', 0):.1%} (max drop {g['max_drop']:.1%}, "
                           f"grader {g['grader']})")
                out.append(f"    {g['name']}: {mark} - {det}")
    else:
        out.append(f"  score {res['score']:.1%}" if res.get("score") is not None else "  no score")
        out.append("  " + ", ".join(f"{k} {v:.0%}" for k, v in res["rates_before"].items()))
    if "heldout_looks" in res:
        out.append("")
        out.append(f"  held-out set looked at {res['heldout_looks']} time(s) so far (log next to the goal file); "
                   "a choice made after many looks overstates the gain")
    out.append(f"  {res['generations']} generations (main set), {res['seconds']} s")
    return "\n".join(out)
