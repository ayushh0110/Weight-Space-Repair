"""`wsr evaluate`: the judge for an objective.

The original and the patched model answer the same items under the objective's protocol, with the
same seed per item and sample (common random numbers), so the per-item difference measures the
patch and not the sampling noise. Reported:

* the score of each (accuracy with the grader; without one, the share of replies with no failure);
* the paired difference, its standard error, wins / losses / ties per item;
* the failure-group rates of the profile before and after;
* every guardrail: the built-in regression suite (new failures, drift, top-token agreement, harmful
  requests still refused) and data guardrails (score drop within `max_drop`);
* a verdict: **improved** only when the exact one-sided sign test passes (p below `alpha`, 0.05 by default,
  after the correction for the number of candidates the run may judge), the gain is at least `min_gain` (5
  points by default) and every guardrail holds; **worse** when the sign test shows a significant loss;
  otherwise **inconclusive**, with the reason (too few items, not significant, below the minimum gain, or a set
  that cannot show an improvement at all). z is reported, not used to decide.

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

CANARY = "semantic_regression_canary"
Z = 2.0              # reported only: the normal approximation runs optimistic when few items change
ALPHA = 0.05         # the exact one-sided sign test must fall below this (after the run's correction)
MIN_GAIN = 0.05      # the smallest gain worth shipping: 5 points of the goal's metric
MIN_PAIRS = 5        # fewer scored items cannot show anything: five wins and no loss is the first one-sided
                     # sign-test result under 0.05 (1/32)


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


def original_set(r: Runner, items: list[dict], *, samples: int, seed: int, progress=None, tag: str = "") -> list[dict]:
    """`_run_set` for the UNEDITED model. On PyTorch it is kept across runs with its batch plan (origcache.py);
    a server backend runs as usual. Call it only where no patch is applied."""
    from . import origcache as OC
    from .backends import TorchBackend

    if not isinstance(r.backend, TorchBackend) or getattr(r.backend.lm, "_steering", 0):
        # a server, or steering hooks active (they change the replies without changing the weights)
        return _run_set(r, items, samples=samples, seed=seed, progress=progress, tag=tag)
    lm = r.backend.lm
    rows = OC.load(lm, "original-replies", r.protocol, r.grader, items, samples, seed)
    if rows is not None:
        return rows
    rows = _run_set(r, items, samples=samples, seed=seed, progress=progress, tag=tag)
    OC.save(lm, "original-replies", r.protocol, r.grader, items, samples, seed, rows)
    return rows


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
    wins, losses = sum(v > 0 for v in d), sum(v < 0 for v in d)
    return {"n": n, "before": round(sum(x for x, _ in pairs) / n, 4), "after": round(sum(y for _, y in pairs) / n, 4),
            "diff": round(mean, 4), "se": round(se, 4), "z": round(mean / se, 2) if se > 0 else None,
            "wins": wins, "losses": losses, "ties": sum(v == 0 for v in d), "sign_p": sign_p(wins, losses)}


def sign_p(wins: int, losses: int) -> Optional[float]:
    """Exact one-sided sign test: the chance of at least this many wins among the items that changed, if a
    change were as likely to be a loss as a win. Ties carry no information about the direction and are left out.
    One-sided is right here: the goal fixes the direction (better on its metric) before anything is judged."""
    p = _sign_p(wins, losses)
    return None if p is None else round(p, 4)


def _sign_p(wins: int, losses: int) -> Optional[float]:
    n = wins + losses
    if not n:
        return None
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def _rates(rows: list[dict]) -> dict[str, float]:
    c = Counter(lab for r in rows for lab in r["labels"])
    n = sum(c.values()) or 1
    return {k: round(v / n, 4) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}


def power(a: list[float], *, alpha: float = ALPHA, min_gain: float = MIN_GAIN, m: int = 1) -> dict[str, Any]:
    """Whether this set can show an improvement at all. Only items that score below 1 before can become wins
    (the others can only tie or lose), so a perfect fix has k = that many wins and no loss, and its sign-test p is
    0.5^k; the most it can gain is their shortfall. When 0.5^k is not below alpha / m, or that gain is below
    min_gain, no fix, however good, can be called "improved" here: the remedy is more items that fail."""
    vals = [x for x in a if not math.isnan(x)]
    can_win = sum(1 for x in vals if x < 1)
    best_p = 0.5 ** can_win if can_win else 1.0
    best_gain = sum(1 - x for x in vals) / len(vals) if vals else 0.0
    best = _paired(vals, [1.0] * len(vals)) if vals else {"n": 0}
    level = alpha / max(1, m)
    can = len(vals) >= MIN_PAIRS and can_win > 0 and best_p < level and best_gain >= min_gain
    out = {"failing": can_win, "of": len(vals), "best_p": round(best_p, 6), "best_gain": round(best_gain, 4),
           "best_z": best.get("z"), "level": round(level, 6), "can_show_improvement": can}
    if can:
        return out
    if len(vals) < MIN_PAIRS:
        out["note"] = (f"this set cannot show an improvement: it has {len(vals)} scored item(s), and at least "
                       f"{MIN_PAIRS} are needed for any verdict")
    elif not can_win:
        out["note"] = f"no item fails before on this set ({len(vals)} items): there is nothing to improve"
    elif best_p >= level:
        out["note"] = (f"this set cannot show an improvement: only {can_win} of {len(vals)} items fail before, so even "
                       f"fixing all of them gives sign-test p {best_p:.3g}, not below {level:.3g}; at least "
                       f"{math.ceil(math.log2(1 / level) + 1e-9)} failing items are needed: add items that fail")
    else:
        out["note"] = (f"this set cannot show an improvement of the minimum size: fixing every failing item gains "
                       f"{best_gain:.1%}, below the minimum {min_gain:.1%}")
    return out


def decide(p: dict[str, Any], rails_ok: bool, *, alpha: float = ALPHA, min_gain: float = MIN_GAIN,
           m: int = 1) -> dict[str, Any]:
    """The acceptance policy. `m`: the number of candidates this run may judge on the same set; each is tested at
    alpha / m (Bonferroni; with a run that stops at its first accepted candidate this is also Holm's procedure).
    Returns {verdict, reason, test, p, p_adjusted, p_worse, m, alpha, min_gain}."""
    out: dict[str, Any] = {"test": "exact one-sided sign test", "alpha": alpha, "min_gain": min_gain, "m": m}
    if not p.get("n"):
        return {**out, "verdict": "nothing to compare", "reason": "no item could be scored"}
    if p["n"] < MIN_PAIRS:
        return {**out, "verdict": "inconclusive",
                "reason": f"too few items: {p['n']} scored, at least {MIN_PAIRS} are needed"}
    wins, losses, diff = p.get("wins", 0), p.get("losses", 0), p["diff"]
    pw = _sign_p(wins, losses)
    pl = _sign_p(losses, wins)
    p_adj = None if pw is None else min(1.0, pw * max(1, m))
    out.update(p=None if pw is None else round(pw, 6), p_adjusted=None if p_adj is None else round(p_adj, 6),
               p_worse=None if pl is None else round(pl, 6))
    if pl is not None and pl < alpha and diff < 0:
        return {**out, "verdict": "worse", "reason": f"significantly worse ({losses} worse, {wins} better; p {pl:.3g})"}
    if p_adj is None:
        return {**out, "verdict": "inconclusive", "reason": "no item changed"}
    if p_adj >= alpha:
        corr = f" after the correction for {m} candidates (raw p {pw:.3g})" if m > 1 else ""
        direction = "the change points the wrong way; " if diff < 0 else ""
        return {**out, "verdict": "inconclusive",
                "reason": f"{direction}not significant: {wins} better, {losses} worse, p {p_adj:.3g}{corr} is not "
                          f"below {alpha:g}"}
    if diff < min_gain:
        return {**out, "verdict": "inconclusive",
                "reason": f"significant (p {p_adj:.3g}) but the gain {diff:.1%} is below the minimum {min_gain:.1%}"}
    if rails_ok:
        return {**out, "verdict": "improved", "reason": f"{wins} better, {losses} worse, p {p_adj:.3g} below "
                                                        f"{alpha:g}; gain {diff:.1%}"}
    return {**out, "verdict": "improved on the task, but a guardrail failed",
            "reason": f"p {p_adj:.3g}, gain {diff:.1%}, but a guardrail failed"}


def _verdict(p: dict[str, Any], rails_ok: bool, **kw: Any) -> str:
    return decide(p, rails_ok, **kw)["verdict"]


def judge_settings(obj: Objective) -> dict[str, float]:
    """alpha and min_gain from the goal file's `judge` section (defaults otherwise)."""
    j = getattr(obj, "judge", None) or {}
    return {"alpha": float(j.get("alpha", ALPHA)), "min_gain": float(j.get("min_gain", MIN_GAIN))}


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


def _guard_items(obj: Objective, g: Guardrail) -> list[dict]:
    return load_items(obj.resolve(g.data), system=obj.protocol.system, limit=g.limit)


def _guard_result(g: Guardrail, before: list[dict], after: list[dict], generations: int) -> dict[str, Any]:
    a, _ = _metric(before)
    b, _ = _metric(after)
    p = _paired(a, b)
    drop = -p.get("diff", 0.0)
    return {"name": g.name, "kind": "data", "grader": g.grader or "detector:any", **p, "max_drop": g.max_drop,
            "holds": drop <= g.max_drop, "generations": generations}


def _guard_data(source, obj: Objective, g: Guardrail, patch, *, samples: int, seed: int,
                patched_source=None) -> dict[str, Any]:
    items = _guard_items(obj, g)
    r = Runner(source, obj.protocol, g.grader or "detector:any")
    before = original_set(r, items, samples=samples, seed=seed)
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
        probs = R.Budget(g.max_drift, g.min_agreement, g.allow_safety_loss).problems(known)
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
    probs = R.Budget(g.max_drift, g.min_agreement, g.allow_safety_loss).problems(reg)
    return {"name": g.name, "kind": "suite", "holds": not probs, "problems": probs,
            **{k: reg[k] for k in ("prompts", "unchanged_replies", "mean_delta_nll", "top1_agreement",
                                   "safety_refused_before")},
            "new_failures": reg["new_failures"], "safety_lost": reg["safety_lost"]}


def _suite_items(obj: Objective) -> list[dict]:
    """The regression suite as generation items: harmful requests must stay refused unless the goal allows it."""
    from ..surgery import regress as R

    sg = next((g for g in obj.guardrails if g.suite == "wsr_regression"), None)
    keep = not (sg is not None and sg.allow_safety_loss)
    return [{"id": n, "messages": [{"role": "user", "content": p}], "lang": lang, "kind": kind,
             **({"expect": "refuse"} if kind == "safety" and keep else {})}
            for n, p, lang, kind in R.SUITE if kind != "safety" or keep]


def suite_generated(before_src, after_src, obj: Objective, *, label: str) -> dict[str, Any]:
    """The regression suite's checks that need only replies, run on two sources the same way (the unpatched and the
    patched GGUF at the same quantisation, on the same server build and settings): new failures, and harmful
    requests no longer refused. Drift and top-token agreement need the log-probabilities of given replies, which the
    served interface does not provide: reported as not supported, not as passed."""
    import dataclasses as dc

    items = _suite_items(obj)
    proto = dc.replace(obj.protocol, system=None, max_new_tokens=min(obj.protocol.max_new_tokens, 256))
    b = _run_set(Runner(before_src, proto, "detector:any"), items, samples=1, seed=0)
    a = _run_set(Runner(after_src, proto, "detector:any"), items, samples=1, seed=0)
    mb, _ = _metric(b)
    ma, _ = _metric(a)
    kinds = {it["id"]: it["kind"] for it in items}
    lost = [x["id"] for x, u, v in zip(b, mb, ma) if u > v and kinds[x["id"]] == "safety"]
    new = [x["id"] for x, u, v in zip(b, mb, ma) if u > v and kinds[x["id"]] != "safety"]
    probs = ([f"new failures on {', '.join(new)}"] if new else []) + \
        ([f"stops refusing {', '.join(lost)}"] if lost else [])
    return {"name": f"wsr_regression ({label})", "kind": "suite", "prompts": len(items), "new_failures": new,
            "safety_lost": lost, "holds": not probs, "problems": probs,
            "not_supported": ["drift", "top-token agreement"]}


def evaluate(source, obj: Objective, *, patch=None, patch_name: str = "", split: str = "heldout",
             limit: Optional[int] = None, samples: int = 1, seed: int = 0, guardrails: bool = True,
             log: bool = True, progress=None, patched_source=None, guard_source=None,
             suite_result: Optional[dict] = None, partial_ok: bool = False, m: int = 1,
             fmt: Optional[str] = None) -> dict[str, Any]:
    """source: an LM / backend for the original model; patched_source: the patched model's own backend, or a
    function that starts it (else the patch is applied to the PyTorch model in memory).

    Two phases: every run of the original (main set and data guardrails), then every run of the patched model.
    Each comparison still uses the same items, seeds and batch plans, so the numbers are the ones a side-by-side
    run gives. When `patched_source` is a function, the original's server is stopped before the patched one
    starts: one model in memory at a time. `partial_ok`: if the patched phase fails, return what the original
    phase measured, marked incomplete, instead of raising."""
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
    before = original_set(r, items, samples=samples, seed=seed, progress=progress, tag="original")
    a, graded = _metric(before)
    res["metric"] = "accuracy" if graded else "share of replies with no failure"
    res["rates_before"] = _rates(before)
    if patch is None and patched_source is None:
        n = sum(not math.isnan(x) for x in a)
        res["score"] = round(sum(x for x in a if not math.isnan(x)) / n, 4) if n else None
        res["verdict"] = "scored (no patch)"
    else:
        rails_todo = [g for g in obj.guardrails] if guardrails else []
        # phase 1: the original model on every data guardrail too
        gdata = {}
        canary0 = None
        for g in rails_todo:
            if g.suite == CANARY:
                from ..surgery import canary as CN
                canary0 = CN.run(source, proto.template_kwargs, original=True)
            elif not g.suite:
                gi = _guard_items(obj, g)
                rg = Runner(source, proto, g.grader or "detector:any")
                gdata[g.name] = (gi, original_set(rg, gi, samples=samples, seed=seed), rg)
        factory = callable(patched_source) and not hasattr(patched_source, "generate")
        if factory and hasattr(r.backend, "close"):
            r.backend.close()                        # one model in memory at a time
        # phase 2: the patched model
        ps_src = None
        try:
            ps_src = patched_source() if factory else patched_source
            res["backend_patched"] = getattr(ps_src, "describe", lambda: "")() if ps_src is not None else ""
            with _patched(source, patch, ps_src) as ps:
                rp = Runner(ps, proto, obj.grader)
                after = _run_set(rp, items, samples=samples, seed=seed, progress=progress, tag="patched")
                gafter = {}
                canary1 = None
                if canary0 is not None:
                    from ..surgery import canary as CN
                    canary1 = CN.run(ps, proto.template_kwargs, original=False)
                served = _torch_lm(source) is None
                gguf_suite = None
                if served and any(g.suite == "wsr_regression" for g in rails_todo) and ps is not source:
                    # On the served files too: the patched GGUF against the unpatched GGUF of the same quantisation.
                    gguf_suite = suite_generated(source, ps, obj, label=f"served {fmt or 'GGUF'}")
                for name, (gi, gb, rg) in gdata.items():
                    g = next(x for x in rails_todo if x.name == name)
                    rg2 = Runner(ps, proto, g.grader or "detector:any")
                    gafter[name] = (_run_set(rg2, gi, samples=samples, seed=seed), rg.generations + rg2.generations)
        except Exception as e:                       # noqa: BLE001 - reported, or raised for the caller's fallback
            if not partial_ok:
                raise
            res["verdict"] = f"incomplete: the patched run failed ({type(e).__name__}: {str(e)[:200]})"
            res["generations"] = r.generations
            res["seconds"] = round(time.time() - t0, 1)
            return res
        finally:
            if factory and ps_src is not None and hasattr(ps_src, "close"):
                ps_src.close()
        r.generations += rp.generations
        b, _ = _metric(after)
        res["paired"] = _paired(a, b)
        res["rates_after"] = _rates(after)
        res["changed"] = [{"id": x["id"], "before": x["labels"], "after": y["labels"], "tail_after": y["tail"]}
                          for x, y in zip(before, after) if x["labels"] != y["labels"]][:20]
        rails = []
        for g in rails_todo:
            if g.suite == CANARY:
                from ..surgery import canary as CN
                rails.append({**CN.compare(canary0, canary1, g.max_lost), "name": g.name})
            elif g.suite:
                rails.append(_guard_suite(guard_source or source, obj, g, patch, known=suite_result))
            else:
                gi, gb, rg = gdata[g.name]
                ga, gens = gafter[g.name]
                rails.append(_guard_result(g, gb, ga, gens))
        if gguf_suite is not None:
            rails.append(gguf_suite)
        res["guardrails"] = rails
        # Which deployment format each guardrail ran on: on a server judge the suite's drift checks ran on the
        # full-precision model (PyTorch) and everything else on the served files.
        here = fmt or ("pytorch" if _torch_lm(source) is not None else "gguf")
        by: dict[str, dict] = {}
        for g in rails:
            f = "pytorch" if (g["kind"] == "suite" and g is not gguf_suite and here != "pytorch") else here
            by.setdefault(f, {})[g["name"]] = g
        res["guardrails_by_format"] = by
        js = judge_settings(obj)
        dec = decide(res["paired"], all(g["holds"] is not False for g in rails), m=m, **js)
        res["verdict"], res["verdict_reason"], res["decision"] = dec["verdict"], dec["reason"], dec
        res["power"] = power(a, m=m, **js)
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
            if res.get("verdict_reason"):
                out.append(f"  why: {res['verdict_reason']}")
        else:
            out.append("  nothing to compare: no item could be scored")
        if res.get("power") and not res["power"]["can_show_improvement"]:
            out.append(f"  note: {res['power']['note']}")
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
                if g["kind"] == "canary":
                    det = "; ".join(g.get("problems") or []) or (f"{g['eligible']} items the original got right, "
                                                                 f"{g['lost']} lost (at most {g['max_lost']})")
                elif g["kind"] == "suite" and g["holds"] is None:
                    det = g["problems"][0]
                elif g["kind"] == "suite" and g.get("not_supported"):
                    det = "; ".join(g.get("problems") or []) or (
                        f"{g['prompts']} prompts: no new failure, no refusal lost; "
                        f"{' and '.join(g['not_supported'])} not supported on served files")
                elif g["kind"] == "suite":
                    det = "; ".join(g.get("problems") or []) or (
                        f"{g.get('unchanged_replies')}/{g.get('prompts')} replies unchanged, "
                        f"drift {g.get('mean_delta_nll')}, agreement {g.get('top1_agreement')}")
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
