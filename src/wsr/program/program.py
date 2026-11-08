"""`wsr program`: work towards an objective, judged on data the worker never sees.

1. **Profile** the practice set: which failure groups, how often, and which point at the weights.
2. **Diagnose** the largest weight-level group. Its first members are the failing prompts and
   working practice items are the contrast, as `wsr deep` takes them.
3. **Playbook**: the rule-based recipes, with the deep checks.
4. **Agent** (with `--agent`, when the playbook cannot fix it): the repair agent's tools plus
   three for the objective:
   - `practice_eval`: an edit against the original on practice items, same seeds, paired;
   - `checkpoint`: a leaderboard of edits worth keeping;
   - `final_choice`: ends the run with one edit.

   The held-out file is never one of its tools, and its path is not in the brief.
5. **Judge**: `wsr evaluate` on the held-out split, with the guardrails. The patch ships
   (`<out>/patch`) only when the verdict is "improved". Otherwise it is kept as an unverified
   candidate (`<out>/candidate`), with the numbers.
6. **Report**: `PROGRAM.md` holds every stage (WSP_PLAN phase 5); `program.json` holds the same
   as data.

A capability objective with no weight-level failure group (only wrong answers, say) gives the edit
tools nothing to start from. The run then stops after the profile and says so.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from . import evaluate as EV
from . import profile as PF
from .backends import TorchBackend
from .objective import Objective, load_items


class ObjectiveTools:
    """The agent's extra tools for an objective; practice data only."""

    system = ("This run has an OBJECTIVE, not only one failing prompt. The failing and working prompts you see "
              "come from the objective's practice set. Besides verify_edit (one prompt's checks), measure an edit "
              "on the practice set with practice_eval: it reports the paired score change against the original "
              "model and the failure-group rates. Keep promising edits with checkpoint. End with final_choice on "
              "the edit you want judged: a held-out set you cannot see, plus guardrails, decides whether it ships. "
              "An edit that passes verify_edit is not enough on its own; one that helps the practice set without "
              "harming the guardrails is what wins.")

    def __init__(self, lm, obj: Objective, items: list[dict], *, max_evals: int = 10, default_n: int = 20):
        self.lm, self.obj, self.items = lm, obj, items
        self.max_evals, self.default_n = max_evals, default_n
        self.evals = 0
        self.base: dict[int, list[dict]] = {}
        self.scores: dict[str, dict] = {}
        self.board: list[dict] = []
        self.wb = None
        self.generations = 0
        self.tools = {"practice_eval": self._practice_eval, "checkpoint": self._checkpoint,
                      "final_choice": self._final_choice}
        self.schemas = [
            {"name": "practice_eval", "description": "Score an edit on the first n practice items against the "
             "original model (same seeds): score before / after, paired difference with its standard error, items "
             "better / worse, failure-group rates. Counts against a small budget.",
             "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                               "n": {"type": "integer", "minimum": 1}},
                              "required": ["edit_id"]}},
            {"name": "checkpoint", "description": "Keep an edit on the leaderboard with a note (its last "
             "practice_eval is attached). Returns the leaderboard.",
             "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                               "note": {"type": "string"}},
                              "required": ["edit_id", "note"]}},
            {"name": "final_choice", "description": "End the run with this edit; the held-out judge decides whether "
             "it ships. The edit must have a practice_eval.",
             "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                               "rationale": {"type": "string"}},
                              "required": ["edit_id", "rationale"]}},
        ]

    def brief(self, wb) -> str:
        o = self.obj
        return (f"## Objective: {o.name} ({o.kind})\n- protocol: {o.protocol.describe()}\n- grader: {o.grader}\n"
                f"- practice set: {len(self.items)} items (practice_eval uses the first n; default {self.default_n})\n"
                f"- guardrails: {', '.join(g.name for g in o.guardrails) or 'none'}\n"
                f"- practice_eval budget: {self.max_evals} calls\n- the held-out set is not available to you")

    def _base(self, n: int) -> list[dict]:
        if n not in self.base:
            r = PF.Runner(TorchBackend(self.lm), self.obj.protocol, self.obj.grader)
            self.base[n] = EV._run_set(r, self.items[:n], samples=1, seed=0)
            self.generations += r.generations
        return self.base[n]

    def _practice_eval(self, wb, edit_id: str, n: Optional[int] = None) -> dict[str, Any]:
        from ..surgery.edits import applied

        if self.evals >= self.max_evals:
            return {"error": f"practice_eval budget used ({self.max_evals}); choose with final_choice"}
        n = max(1, min(int(n or self.default_n), len(self.items)))
        patch = wb._edit(edit_id)
        base = self._base(n)
        r = PF.Runner(TorchBackend(self.lm), self.obj.protocol, self.obj.grader)
        with applied(self.lm, patch):
            after = EV._run_set(r, self.items[:n], samples=1, seed=0)
        self.generations += r.generations
        self.evals += 1
        a, graded = EV._metric(base)
        b, _ = EV._metric(after)
        res = {"edit_id": edit_id, "n": n, "metric": "accuracy" if graded else "share with no failure",
               **EV._paired(a, b), "rates_before": EV._rates(base), "rates_after": EV._rates(after),
               "practice_evals_left": self.max_evals - self.evals}
        self.scores[edit_id] = res
        return res

    def _checkpoint(self, wb, edit_id: str, note: str) -> dict[str, Any]:
        wb._edit(edit_id)
        self.board = [b for b in self.board if b["edit_id"] != edit_id]
        sc = self.scores.get(edit_id)
        self.board.append({"edit_id": edit_id, "label": wb.edits[edit_id].label, "note": note,
                           "practice": {k: sc[k] for k in ("n", "before", "after", "diff", "se")} if sc else None})
        self.board.sort(key=lambda b: -(b["practice"] or {}).get("diff", -9))
        return {"leaderboard": self.board}

    def _final_choice(self, wb, edit_id: str, rationale: str) -> dict[str, Any]:
        wb._edit(edit_id)
        if edit_id not in self.scores:
            return {"error": "run practice_eval on this edit first"}
        wb.done = {"final": edit_id, "rationale": rationale}
        return {"accepted": True}


def prior_labels(lm, obj: Objective, items: list[dict]) -> Optional[list]:
    """The practice replies `wsr objective from-failure` labelled (practice/labels.json), if they are exactly what
    profiling would compute now: same weights (cache key), protocol and grader, every item's messages the same,
    labelled on the full-precision model, greedy. Otherwise None (profile as usual)."""
    from .. import cache
    from .fromfailure import msg_hash

    p = obj.resolve(obj.practice)
    f = p.parent / "labels.json" if p else None
    if f is None or not f.exists() or obj.protocol.sampled:
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    key = cache.model_key(lm)
    proto = {"template_kwargs": obj.protocol.template_kwargs, "sampling": obj.protocol.sampling,
             "max_new_tokens": obj.protocol.max_new_tokens, "system": obj.protocol.system}
    if not key or d.get("model_key") != key or d.get("protocol") != proto or d.get("grader") != obj.grader \
            or not str(d.get("backend", "")).startswith("pytorch"):
        return None
    by_id = {r["id"]: r for r in d.get("replies", [])}
    out = []
    for i, it in enumerate(items):
        r = by_id.get(it["id"])
        if r is None or r["msg"] != msg_hash(it["messages"]):
            return None
        out.append(PF.Reply(it["id"], r["text"], r["tokens"], r["ended"], r["cut"], set(r["groups"]), r["correct"],
                            r["label"], PF.seed_for(i, 0, 0)))
    return out


def _user_text(it: dict) -> str:
    return next((str(m.get("content")) for m in reversed(it["messages"]) if m.get("role") == "user"), "")


def run_program(lm, obj: Objective, *, out: Path, profile_source=None, provider=None, playbook: bool = True,
                profile_limit: int = 40, agent_budget=None, agent_limits=None, practice_evals: int = 10,
                eval_limit: Optional[int] = None, log=print, decisions: Optional[list] = None,
                mode: str = "deep", judge: str = "torch", judge_quant: str = "Q8_0", consent=None,
                model_dir: Optional[Path] = None, server_bin: Optional[str] = None) -> dict[str, Any]:
    from ..deep.run import diagnose
    from ..surgery import playbook as P

    say = log or (lambda s: None)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    proto = obj.protocol
    practice = load_items(obj.resolve(obj.practice), system=proto.system) if obj.practice else []
    if not practice:
        raise ValueError("wsr program needs a practice set in the goal file")
    rec: dict[str, Any] = {"objective": obj.to_dict(), "stages": {}}

    # 1. profile
    src = profile_source or lm
    prior = prior_labels(lm, obj, practice[:profile_limit])
    if prior is not None:
        say(f"practice replies from wsr objective from-failure reused ({len(prior)} items: same weights, protocol "
            "and messages)")
    prof = PF.profile(prior and lm or src, practice[:profile_limit], proto, grader=obj.grader, name=obj.name,
                      model=str(obj.model or ""), test_k=2, test_n=2, replies=prior)
    rec["stages"]["profile_reused"] = prior is not None
    rec["stages"]["profile"] = prof.to_dict()
    say(PF.render(prof))
    weight = sorted((r for r in prof.rows if r.get("cause") == "weight"), key=lambda r: -r["count"])
    if not weight:
        rec["outcome"] = "nothing to edit"
        rec["says"] = ("no failure group in the practice profile points at the weights (decoding, budget and capability "
                       "groups are not edited); nothing for the edit tools to start from")
        return _finish(rec, out, t0)
    group = weight[0]
    label = group["label"]
    fail = [it for it, rp in zip(practice[:profile_limit], prof.replies) if rp["label"] == label]
    work = [it for it, rp in zip(practice[:profile_limit], prof.replies) if rp["label"] in ("correct", "ok")]
    if profile_source is not None and fail and prior is None:
        # The profile ran on the served (quantised) file; the diagnosis and edits use the full-precision model.
        # Keep the members that fail the same way there too.
        r = PF.Runner(TorchBackend(lm), proto, obj.grader)
        again = r.run_many([{"item": it, "seed": 0} for it in fail[:8]])
        confirmed = [it for it, rp in zip(fail[:8], again) if PF._family(rp.label) == PF._family(label)]
        rec["stages"]["full_precision_check"] = {"tested": len(again), "still_failing": len(confirmed)}
        say(f"full-precision check: {len(confirmed)}/{len(again)} of the group's prompts fail the same way in PyTorch")
        if not confirmed:
            rec["outcome"] = "nothing to edit"
            rec["says"] = (f"the {label} replies seen on the served file do not reproduce on the full-precision model "
                           "(a quantisation effect: see wsr qbisect), so there is nothing in the weights to edit")
            return _finish(rec, out, t0)
        fail = confirmed + [it for it in fail if it not in confirmed and it not in fail[:8]]
    kind = PF.DEEP_KIND.get(label, "auto")
    max_new = min(proto.max_new_tokens, 256) if kind == "loop" else min(proto.max_new_tokens, 48)
    rec["stages"]["group"] = {"label": label, "count": group["count"], "says": group["says"], "kind": kind,
                              "failing_used": [it["id"] for it in fail[:4]], "working_used": [it["id"] for it in work[:3]]}
    say(f"\nworking on {label} ({group['count']} of {prof.n} practice replies): {group['says']}")

    # 2. diagnose, 3. playbook
    tk = proto.template_kwargs or None
    res = diagnose(lm, fail[0]["messages"], kind=kind, failing=tuple(_user_text(i) for i in fail[1:4]),
                   passing=tuple(_user_text(i) for i in work[:3]), max_new=max_new, template_kwargs=tk,
                   out_dir=out / "deep")
    rec["stages"]["diagnosis"] = {"summary": res.summary, "confidence": res.confidence}
    say(f"diagnosis: {res.summary}")
    rep = P.repair(lm, res, max_new=max_new, template_kwargs=tk, mode=mode, playbook=playbook, log=say)
    rec["stages"]["playbook"] = {"status": rep.status, "summary": rep.summary,
                                 "patch": rep.patch.label if rep.patch is not None else None}
    candidate, by = (rep.patch, "playbook") if rep.status == "fixed" else (None, None)
    # The verification already ran the regression suite on this patch with this template; the judge's
    # guardrail reuses it (same computation, same numbers) instead of running it again.
    suite_result = rep.regression if rep.status == "fixed" else None

    # 4. agent
    if candidate is None and provider is not None and rep.status == "could_not_fix":
        from ..agent import run as agent_run

        tools = ObjectiveTools(lm, obj, practice, max_evals=practice_evals)
        say(f"\nthe playbook could not fix it; handing over to the agent ({provider.name}) with the objective")
        rep2 = agent_run(lm, res, rep, provider, budget=agent_budget, limits=agent_limits, out=out, log=say,
                         program=tools)
        wb = tools.wb
        rec["stages"]["agent"] = {**(rep2.checks or {}).get("agent", {}), "leaderboard": tools.board,
                                  "practice_evals": tools.scores}
        if wb is not None and wb.done and "final" in wb.done:
            candidate, by = wb.edits[wb.done["final"]], f"agent (final choice: {wb.done['rationale'][:200]})"
            suite_result = (wb.results.get(wb.done["final"]) or {}).get("regression")
        elif rep2.status == "fixed" and rep2.patch is not None:
            candidate, by = rep2.patch, "agent (submitted after verify_edit)"
            suite_result = rep2.regression
    if candidate is None:
        rec["outcome"] = "no candidate"
        rec["says"] = rep.summary
        return _finish(rec, out, t0)

    # 5. the judge
    say(f"\njudging {candidate.label} ({by}) on the held-out set")
    from ..consent import Consent
    from . import judge as JG

    jd = JG.choose(judge, lm=lm, model_dir=model_dir, patch=candidate, out=out, consent=consent or Consent(),
                   quant=judge_quant, server_bin=server_bin, max_new=proto.max_new_tokens, log=say)
    try:
        if jd.kind == "llama-server":
            ev = EV.evaluate(jd.source, obj, patched_source=jd.patched, guard_source=lm, patch=candidate,
                             patch_name=candidate.label, split="heldout", limit=eval_limit, suite_result=suite_result)
        else:
            ev = EV.evaluate(lm, obj, patch=candidate, patch_name=candidate.label, split="heldout", limit=eval_limit,
                             suite_result=suite_result)
    finally:
        jd.close()
    ev["judge"] = {"kind": jd.kind, "notes": jd.notes, "files": jd.files}
    rec["stages"]["judge"] = ev
    say(EV.render(ev))
    shipped = ev["verdict"] == "improved"
    candidate.checks["objective"] = {"name": obj.name, "verdict": ev["verdict"], "paired": ev.get("paired"),
                                     "by": by}
    candidate.save(out / ("patch" if shipped else "candidate"))
    rec["outcome"] = "shipped" if shipped else "not shipped"
    rec["patch"] = {"label": candidate.label, "by": by, "dir": str(out / ("patch" if shipped else "candidate"))}
    rec["says"] = ev["verdict"]
    if consent is not None and consent.decisions:
        rec["decisions"] = consent.decisions
    elif decisions:
        rec["decisions"] = decisions
    return _finish(rec, out, t0)


def _finish(rec: dict, out: Path, t0: float) -> dict:
    rec["seconds"] = round(time.time() - t0, 1)
    (out / "program.json").write_text(json.dumps(rec, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    (out / "PROGRAM.md").write_text(report(rec), encoding="utf-8")
    return rec


def report(rec: dict) -> str:
    o = rec["objective"]
    st = rec["stages"]
    lines = [f"# Program: {o['name']}", "", f"**{rec['outcome']}** - {rec.get('says', '')}", "",
             "## Objective", f"- kind: {o['kind']}; grader: {o['grader']}; practice: {o['practice']}; held-out: "
             f"{o['heldout']} (not given to the agent)",
             f"- guardrails: {', '.join(g.get('suite') or g.get('data', '?') for g in o['guardrails']) or 'none'}", ""]
    if "profile" in st:
        lines += ["## Practice profile", "", "```", PF.render(st["profile"]), "```", ""]
    if "group" in st:
        g = st["group"]
        lines += ["## Worked on", f"{g['label']} ({g['count']} replies): {g['says']}", ""]
    if "diagnosis" in st:
        lines += ["## Diagnosis", st["diagnosis"]["summary"], ""]
    if "playbook" in st:
        p = st["playbook"]
        lines += ["## Playbook", f"{p['status']}: {p['summary']}", ""]
    if "agent" in st:
        a = st["agent"]
        lines += ["## Agent", f"- {a.get('provider')}, {a.get('turns')} turns, ${a.get('cost_usd')}, stopped: "
                  f"{a.get('stopped')}"]
        for b in a.get("leaderboard") or []:
            lines.append(f"- checkpoint {b['edit_id']} ({b['label']}): {b['note']} - practice {b['practice']}")
        for n in a.get("notes") or []:
            lines.append(f"- note: {n}")
        lines.append("")
    if "judge" in st:
        j = st["judge"].get("judge") or {}
        lines += ["## Held-out judge", f"Judge: {j.get('kind', 'pytorch')}"
                  + (f" ({'; '.join(j['notes'])})" if j.get("notes") else ""), "", "```", EV.render(st["judge"]),
                  "```", ""]
        if j.get("files", {}).get("candidate"):
            lines += [f"Served file of the candidate: `{j['files']['candidate']}`", ""]
    if rec.get("patch"):
        p = rec["patch"]
        lines += ["## Patch", f"{p['label']} (found by {p['by']}) in `{p['dir']}`."]
        if rec["outcome"] == "shipped":
            lines.append(f"Apply: `wsr patch apply <model-dir> {p['dir']} --out <dir>`; revert with `wsr patch revert`.")
        else:
            lines.append("Not shipped: the held-out judge did not call it an improvement. Kept for inspection only.")
        lines.append("")
    if rec.get("decisions"):
        lines += ["## Questions asked", *[f"- {d['question']}: {'yes' if d['allowed'] else 'no'} ({d['how']})"
                                          for d in rec["decisions"]], ""]
    lines.append(f"{rec['seconds']} s")
    return "\n".join(lines) + "\n"
