"""`wsr program`: work towards an objective, judged on data the worker never sees.

1. **Profile** the practice set: which failure groups, how often, and which point at the weights.
2. **Diagnose** the largest weight-level group. Its first members are the failing prompts and
   working practice items are the contrast, as `wsr deep` takes them.
3. **Playbook**: the rule-based recipes, with the deep checks.
4. **Agent** (with `--agent`, when the playbook cannot fix it): the repair agent's tools plus
   three for the objective:
   - `practice_eval`: an edit against the original on practice items, same seeds, paired;
   - `checkpoint`: a leaderboard of edits worth keeping;
   - `settings_eval`: serving settings (repetition penalty, temperature, top_p ...) on the original weights,
     against the same baseline, for failures that decoding causes or cures;
   - `steer_eval`: activation steering (a vector added to the residual stream after chosen layers) on the
     original weights, for behaviour spread over many layers that no small weight edit separates;
   - `final_choice`: ends the run with one edit, or one settings candidate.

   The held-out file is never one of its tools, and its path is not in the brief.
5. **Judge**: `wsr evaluate` on the held-out split, with the guardrails. The patch ships
   (`<out>/patch`) only when the verdict is "improved". Otherwise it is kept as an unverified
   candidate (`<out>/candidate`), with the numbers. A settings candidate is judged the same way on
   the same weights (current settings vs new), with the regression-suite prompts generated under the
   new settings as its guardrail; it is reported as recommended settings, not shipped as a patch.
6. **Report**: `PROGRAM.md` holds every stage; `program.json` holds the same
   as data.

A capability objective with no weight-level failure group (only wrong answers, say) gives the edit
tools nothing to start from. The run then stops after the profile and says so.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Optional

from . import evaluate as EV
from . import profile as PF
from .backends import TorchBackend
from ..surgery.steer import STEER_APPLY  # noqa: F401 - re-exported (tests, docs)
from .objective import Objective, Protocol, load_items

# Serving settings the agent may test on PyTorch, with the range each is checked against.
SETTINGS_RANGES = {"repetition_penalty": (1.0, 1.5), "temperature": (0.0, 2.0), "top_p": (0.05, 1.0),
                   "top_k": (1, 1000), "min_p": (0.0, 0.5)}
LLAMA_FLAGS = {"repetition_penalty": "--repeat-penalty", "temperature": "--temp", "top_p": "--top-p",
               "top_k": "--top-k", "min_p": "--min-p"}


def settings_problem(settings: Any) -> str:
    """Why these serving settings cannot be tested here, or ''."""
    if not isinstance(settings, dict) or not settings:
        return "settings must be an object such as {\"repetition_penalty\": 1.1}"
    for k, v in settings.items():
        if k == "system":
            if v is not None and (not isinstance(v, str) or len(v) > 2000):
                return "system must be a string of at most 2000 characters (or null for none)"
            continue
        if k == "template_kwargs":
            if not isinstance(v, dict) or not all(isinstance(x, (bool, int, float, str)) for x in v.values()):
                return "template_kwargs must be an object of simple values, e.g. {\"enable_thinking\": false}"
            continue
        if k == "presence_penalty":
            return ("presence_penalty has no PyTorch equivalent (it exists on llama-server); test repetition_penalty "
                    "here")
        if k not in SETTINGS_RANGES:
            return f"unknown setting {k!r}; one of {', '.join(SETTINGS_RANGES)}"
        lo, hi = SETTINGS_RANGES[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= float(v) <= hi:
            return f"{k} must be a number in [{lo}, {hi}]"
    return ""


def with_settings(proto: Protocol, settings: dict) -> Protocol:
    """The protocol with these serving settings on top: sampling keys, and optionally a different system prompt
    and chat-template arguments; the token limit stays."""
    samp = {k: v for k, v in settings.items() if k not in ("system", "template_kwargs")}
    out = dataclasses.replace(proto, sampling={**proto.sampling, **samp},
                              template_kwargs={**proto.template_kwargs, **(settings.get("template_kwargs") or {})})
    if "system" in settings:
        out = dataclasses.replace(out, system=settings["system"])
    return out


def items_for(settings: dict, items: list[dict]) -> list[dict]:
    """The items as the model sees them under these settings. A "system" setting replaces each item's system
    message (null removes it): the goal's system prompt is written into the messages when the items are loaded,
    so changing only the protocol would not reach the model."""
    if "system" not in settings:
        return items
    out = []
    for it in items:
        msgs = [m for m in it["messages"] if m.get("role") != "system"]
        if settings["system"]:
            msgs = [{"role": "system", "content": settings["system"]}] + msgs
        out.append({**it, "messages": msgs})
    return out


def settings_howto(proto: Protocol, settings: dict) -> list[str]:
    s = {**proto.sampling, **{k: v for k, v in settings.items() if k not in ("system", "template_kwargs")}}
    gc = dict(s)
    if float(s.get("temperature", 0) or 0) > 0:
        gc["do_sample"] = True
    flags = " ".join(f"{LLAMA_FLAGS[k]} {v}" for k, v in s.items() if k in LLAMA_FLAGS)
    return [f"- transformers `generation_config.json` (or `generate(...)` arguments): `{json.dumps(gc)}`",
            f"- llama-server: `{flags}`" if flags else "",
            f"- system prompt: {json.dumps(settings['system'])}" if "system" in settings else "",
            f"- chat template arguments: `{json.dumps(settings['template_kwargs'])}`" if settings.get("template_kwargs")
            else "",
            "- the weights are unchanged"]


class ObjectiveTools:
    """The agent's extra tools for an objective; practice data only."""

    system = ("This run has an OBJECTIVE, not only one failing prompt. The failing and working prompts you see "
              "come from the objective's practice set. Besides verify_edit (one prompt's checks), measure an edit "
              "on the practice set with practice_eval: it reports the paired score change against the original "
              "model and the failure-group rates. Keep promising edits with checkpoint. End with final_choice on "
              "the edit you want judged: a held-out set you cannot see, plus guardrails, decides whether it ships. "
              "An edit that passes verify_edit is not enough on its own; one that helps the practice set without "
              "harming the guardrails is what wins.")

    def __init__(self, lm, obj: Objective, items: list[dict], *, max_evals: int = 10, default_n: int = 20,
                 baseline: Optional[list[dict]] = None, plans: Optional[dict] = None):
        """baseline: the profile's replies of the original on items[:len(baseline)], when that pass ran on this
        very model in this process (same protocol, seeds and order as practice_eval's own baseline); plans: the
        batch plans it ran in. With both, practice_eval at that n reuses the baseline and runs the edited model
        in the same chunks, so the comparison is exactly the one a fresh baseline gives, without regenerating
        the original."""
        self.lm, self.obj, self.items = lm, obj, items
        self.max_evals, self.default_n = max_evals, default_n
        self.evals = 0
        self.base: dict[int, list[dict]] = {}
        self.plans = dict(plans or {})
        if baseline and plans is not None and \
                [b["id"] for b in baseline] == [it["id"] for it in items[:len(baseline)]]:
            self.base[len(baseline)] = [{"id": b["id"], "labels": [b["label"]], "correct": [b["correct"]],
                                         "tail": b.get("tail", "")} for b in baseline]
            self.default_n = len(baseline)            # the n whose baseline costs nothing
        self.scores: dict[str, dict] = {}
        self.board: list[dict] = []
        self.wb = None
        self.generations = 0
        self.settings: dict[str, dict] = {}
        self.steerings: dict[str, dict] = {}           # id -> result, plus the resolved vector ("_vector")
        self.note = ""                                  # e.g. the playbook's candidate was rejected on held-out
        self.tools = {"practice_eval": self._practice_eval, "settings_eval": self._settings_eval,
                      "steer_eval": self._steer_eval,
                      "checkpoint": self._checkpoint, "final_choice": self._final_choice}
        self.schemas = [
            {"name": "practice_eval", "description": "Score an edit on the first n practice items against the "
             "original model (same seeds): score before / after, paired difference with its standard error, items "
             "better / worse, failure-group rates. The original's score comes with every call: never spend one "
             "on a no-op edit to get a baseline (the profile in the brief already gives it). Counts against a "
             "small budget, and each call generates n full replies.",
             "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                               "n": {"type": "integer", "minimum": 1}},
                              "required": ["edit_id"]}},
            {"name": "settings_eval", "description": "Score serving settings instead of an edit: the ORIGINAL "
             "weights on the first n practice items with these settings on top of the objective's protocol, against "
             "the same baseline as practice_eval (paired, failure-group rates). For failures that decoding causes or "
             "cures, such as loops: e.g. {\"repetition_penalty\": 1.1}. Allowed: repetition_penalty [1, 1.5], "
             "temperature [0, 2], top_p, top_k, min_p, a different system prompt (\"system\": text, or null for none) "
             "and chat-template arguments (\"template_kwargs\"). Shares the practice_eval budget. A settings candidate can be "
             "the final_choice: it is judged on the held-out set like an edit and reported as recommended settings.",
             "input_schema": {"type": "object", "properties": {"settings": {"type": "object"},
                                                               "n": {"type": "integer", "minimum": 1}},
                              "required": ["settings"]}},
            {"name": "steer_eval", "description": "Score ACTIVATION STEERING instead of an edit: the original "
             "weights on the first n practice items with alpha x vector added to the residual stream after each "
             "of the given layers (every position), against the same baseline as practice_eval. For behaviour "
             "spread over many layers that no small weight edit separates. vector: a direction id ('d1', 'd1@12') "
             "or a workspace variable of d_model numbers. mode \"clamp\" instead lowers the component along the "
             "vector to the level alpha wherever it is above it and leaves every other position alone: for a "
             "failure that fires only sometimes (a constant shift also moves healthy text). A steering candidate "
             "can be the final_choice: it is "
             "judged on the held-out set with the regression suite run under the steering, and reported as a "
             "steering vector to apply at inference (no weights change). To REMOVE a direction instead, "
             "build_edit orthogonalise does it exactly in the weights. Shares the practice_eval budget.",
             "input_schema": {"type": "object", "properties": {
                 "vector": {"type": "string"}, "alpha": {"type": "number"},
                 "mode": {"type": "string", "enum": ["add", "clamp"]},
                 "layers": {"type": "array", "items": {"type": "integer"}},
                 "n": {"type": "integer", "minimum": 1}, "why": {"type": "string"}},
                 "required": ["vector", "alpha", "layers"]}},
            {"name": "checkpoint", "description": "Keep an edit on the leaderboard with a note (its last "
             "practice_eval is attached). Returns the leaderboard.",
             "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                               "note": {"type": "string"}},
                              "required": ["edit_id", "note"]}},
            {"name": "final_choice", "description": "End the run with this edit (edit_id), these serving "
             "settings (settings_id from settings_eval) or this steering (steering_id from steer_eval); the held-out "
             "judge decides whether it ships (an edit) or is recommended (settings, steering). It must have a "
             "practice_eval / settings_eval / steer_eval.",
             "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                               "settings_id": {"type": "string"},
                                                               "steering_id": {"type": "string"},
                                                               "rationale": {"type": "string"}},
                              "required": ["rationale"]}},
        ]

    def brief(self, wb) -> str:
        o = self.obj
        out = (f"## Objective: {o.name} ({o.kind})\n- protocol: {o.protocol.describe()}\n- grader: {o.grader}\n"
               f"- practice set: {len(self.items)} items (practice_eval uses the first n; default {self.default_n})\n"
               f"- guardrails: {', '.join(g.name for g in o.guardrails) or 'none'}\n"
               f"- practice_eval budget: {self.max_evals} calls (settings_eval shares it)\n"
               "- the held-out set is not available to you\n"
               "- if the evidence says decoding causes or cures the failure (loops that sampling or a repetition "
               "penalty escape), measure that with settings_eval: a settings fix needs no weight edit\n"
               "- if the behaviour is spread over many layers and every small edit also changes what must stay (a "
               "shared mechanism), measure activation steering with steer_eval")
        if self.note:
            out += "\n- " + self.note
        for n, base in sorted(self.base.items()):
            a, graded = EV._metric(base)
            rates = ", ".join(f"{k} {v:.0%}" for k, v in sorted(EV._rates(base).items(), key=lambda kv: -kv[1]))
            score = f"accuracy {sum(a) / len(a):.1%}" if graded else f"share with no failure {sum(a) / len(a):.1%}"
            out += (f"\n- baseline (the original model on the first {n} practice items, already measured): {score}; "
                    f"{rates}. practice_eval with n={n} reuses it, so a call only runs the edited model; another n "
                    "regenerates the original once first. Do not build a no-op edit to measure the baseline.")
        return out

    def _base(self, n: int) -> list[dict]:
        if n not in self.base:
            r = PF.Runner(TorchBackend(self.lm), self.obj.protocol, self.obj.grader)
            self.base[n] = EV.original_set(r, self.items[:n], samples=1, seed=0)
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
        if self.plans:
            # the chunks the reused baseline ran in (the plan cache may have been cleared since)
            self.lm.batch_plans.update(self.plans)
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

    def _settings_eval(self, wb, settings: dict, n: Optional[int] = None) -> dict[str, Any]:
        if self.evals >= self.max_evals:
            return {"error": f"practice_eval budget used ({self.max_evals}); choose with final_choice"}
        why = settings_problem(settings)
        if why:
            return {"error": why}
        n = max(1, min(int(n or self.default_n), len(self.items)))
        base = self._base(n)
        r = PF.Runner(TorchBackend(self.lm), with_settings(self.obj.protocol, settings), self.obj.grader)
        if self.plans:
            self.lm.batch_plans.update(self.plans)
        after = EV._run_set(r, items_for(settings, self.items[:n]), samples=1, seed=0)
        self.generations += r.generations
        self.evals += 1
        a, graded = EV._metric(base)
        b, _ = EV._metric(after)
        sid = f"s{len(self.settings) + 1}"
        res = {"settings_id": sid, "settings": dict(settings), "n": n,
               "metric": "accuracy" if graded else "share with no failure", **EV._paired(a, b),
               "rates_before": EV._rates(base), "rates_after": EV._rates(after),
               "practice_evals_left": self.max_evals - self.evals}
        self.settings[sid] = res
        return res

    def _steer_eval(self, wb, vector: str, alpha: float, layers: list, n: Optional[int] = None,
                    why: str = "", mode: str = "add") -> dict[str, Any]:
        if self.evals >= self.max_evals:
            return {"error": f"practice_eval budget used ({self.max_evals}); choose with final_choice"}
        try:
            kw = steering_kwargs(wb, vector, alpha, layers, mode)
        except ValueError as e:
            return {"error": str(e)}
        n = max(1, min(int(n or self.default_n), len(self.items)))
        base = self._base(n)
        r = PF.Runner(TorchBackend(self.lm), self.obj.protocol, self.obj.grader)
        if self.plans:
            self.lm.batch_plans.update(self.plans)
        with self.lm.steer(**kw):
            after = EV._run_set(r, self.items[:n], samples=1, seed=0)
        self.generations += r.generations
        self.evals += 1
        a, graded = EV._metric(base)
        b, _ = EV._metric(after)
        sid = f"t{len(self.steerings) + 1}"
        res = {"steering_id": sid, "mode": mode, "vector": str(vector), "alpha": float(alpha),
               "layers": [int(i) for i in layers],
               "n": n, "metric": "accuracy" if graded else "share with no failure", **EV._paired(a, b),
               "rates_before": EV._rates(base), "rates_after": EV._rates(after),
               "practice_evals_left": self.max_evals - self.evals}
        self.steerings[sid] = {**res, "_vector": kw[mode][0].detach().float().cpu().clone(), "why": why}
        return res

    def _checkpoint(self, wb, edit_id: str, note: str) -> dict[str, Any]:
        wb._edit(edit_id)
        self.board = [b for b in self.board if b["edit_id"] != edit_id]
        sc = self.scores.get(edit_id)
        self.board.append({"edit_id": edit_id, "label": wb.edits[edit_id].label, "note": note,
                           "practice": {k: sc[k] for k in ("n", "before", "after", "diff", "se")} if sc else None})
        self.board.sort(key=lambda b: -(b["practice"] or {}).get("diff", -9))
        return {"leaderboard": self.board}

    def _final_choice(self, wb, rationale: str, edit_id: Optional[str] = None,
                      settings_id: Optional[str] = None, steering_id: Optional[str] = None) -> dict[str, Any]:
        if steering_id is not None:
            if steering_id not in self.steerings:
                return {"error": f"unknown steering_id {steering_id!r}; run steer_eval first"}
            wb.done = {"final_steering": steering_id, "rationale": rationale}
            return {"accepted": True}
        if settings_id is not None:
            if settings_id not in self.settings:
                return {"error": f"unknown settings_id {settings_id!r}; run settings_eval first"}
            wb.done = {"final_settings": settings_id, "rationale": rationale}
            return {"accepted": True}
        if edit_id is None:
            return {"error": "give edit_id (an edit) or settings_id (serving settings)"}
        wb._edit(edit_id)
        if edit_id not in self.scores:
            return {"error": "run practice_eval on this edit first"}
        wb.done = {"final": edit_id, "rationale": rationale}
        return {"accepted": True}

    def best_settings(self) -> Optional[dict]:
        """The settings candidate with clear practice evidence (gain at least one standard error), if any."""
        ok = [s for s in self.settings.values() if s.get("diff", 0) > 0 and s.get("diff", 0) >= (s.get("se") or 0)]
        return max(ok, key=lambda s: s["diff"]) if ok else None


def steering_kwargs(wb, vector: Any, alpha: Any, layers: Any, mode: str = "add") -> dict[str, Any]:
    """lm.steer arguments, checked: "add" alpha x vector after these layers, or "clamp" the component along the
    vector to the level alpha wherever it is above it."""
    if mode not in ("add", "clamp"):
        raise ValueError("mode must be add or clamp")
    if not isinstance(layers, list) or not layers or len(layers) > wb.lm.n_layers:
        raise ValueError(f"layers: a list of 1-{wb.lm.n_layers} layer indices")
    ls = sorted({int(i) for i in layers})
    if ls[0] < 0 or ls[-1] >= wb.lm.n_layers:
        raise ValueError(f"layers must be in 0-{wb.lm.n_layers - 1}")
    a = float(alpha)
    if abs(a) > 100 or (mode == "add" and not a):
        raise ValueError("alpha must be at most 100 in size (and non-zero for add)")
    import torch

    v = wb._vector(vector)
    if not bool(torch.isfinite(v).all()) or float(v.norm()) == 0:
        raise ValueError("the vector is zero or not finite")
    return {mode: (v / v.norm() if mode == "clamp" else v, a, ls)}


def judge_steering(lm, obj: Objective, st: dict, *, limit: Optional[int] = None, say=lambda s: None) -> dict:
    """Steering on the held-out set: the same weights without and with the steering hooks, same items and seeds,
    paired. Guardrails: the regression suite compared under the steering exactly as for a patch (new failures,
    drift, top-token agreement, refusals; the goal's limits), and every data guardrail under both."""
    from ..surgery import regress as R

    t0 = time.time()
    kw = {st.get("mode", "add"): (st["_vector"], st["alpha"], st["layers"])}
    items = load_items(obj.resolve(obj.heldout), system=obj.protocol.system, limit=limit)
    say(f"  held-out: {len(items)} items, without then with the steering")
    r0, r1 = PF.Runner(TorchBackend(lm), obj.protocol, obj.grader), PF.Runner(TorchBackend(lm), obj.protocol, obj.grader)
    before = EV.original_set(r0, items, samples=1, seed=0)
    with lm.steer(**kw):
        after = EV._run_set(r1, items, samples=1, seed=0)
    a, graded = EV._metric(before)
    b, _ = EV._metric(after)
    paired = EV._paired(a, b)
    rails = []
    tk = obj.protocol.template_kwargs or None
    for g in obj.guardrails:
        if g.suite:
            base = R.baseline(lm, template_kwargs=tk)
            with lm.steer(**kw):
                reg = R.compare(lm, base, template_kwargs=tk)
            probs = R.Budget(g.max_drift, g.min_agreement, g.allow_safety_loss).problems(reg)
            rails.append({"name": g.name, "kind": "suite", "holds": not probs, "problems": probs,
                          **{k: reg[k] for k in ("prompts", "unchanged_replies", "mean_delta_nll", "top1_agreement",
                                                 "safety_refused_before")},
                          "new_failures": reg["new_failures"], "safety_lost": reg["safety_lost"]})
        elif g.data:
            gi = EV._guard_items(obj, g)
            rg = PF.Runner(TorchBackend(lm), obj.protocol, g.grader or "detector:any")
            gb = EV.original_set(rg, gi, samples=1, seed=0)
            with lm.steer(**kw):
                ga = EV._run_set(rg, gi, samples=1, seed=0)
            rails.append(EV._guard_result(g, gb, ga, 2 * len(gi)))
    verdict = EV._verdict(paired, all(x["holds"] is not False for x in rails))
    looks = EV.HeldoutLog(obj).add({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "patch": f"steering: {st['vector']}",
                                    "items": len(items), "verdict": verdict})
    return {"vector": st["vector"], "mode": st.get("mode", "add"), "power": EV.power(a), "alpha": st["alpha"],
            "layers": st["layers"], "n": len(items),
            "metric": "accuracy" if graded else "share with no failure", "paired": paired,
            "rates_before": EV._rates(before), "rates_after": EV._rates(after), "guardrails": rails,
            "verdict": verdict, "heldout_looks": looks, "seconds": round(time.time() - t0, 1)}


def save_steering(st: dict, out: Path) -> Path:
    """The steering as files (surgery/steer.py): the vector, its settings and APPLY.md."""
    from ..surgery import steer as SV

    return SV.save(st["_vector"], st["alpha"], st["layers"], st["vector"], out, mode=st.get("mode", "add"))


def judge_settings(lm, obj: Objective, settings: dict, *, limit: Optional[int] = None,
                   say=lambda s: None) -> dict[str, Any]:
    """Serving settings on the held-out set: the original weights with the objective's protocol vs the same
    weights with these settings, same items and seeds, paired. Guardrails: every data guardrail under both, and
    the regression-suite prompts generated under both (no reply that was fine may fail, every refusal kept)."""
    from ..surgery import regress as R

    t0 = time.time()
    proto, new = obj.protocol, with_settings(obj.protocol, settings)
    items = load_items(obj.resolve(obj.heldout), system=proto.system, limit=limit)
    say(f"  held-out: {len(items)} items, current settings then {settings}")
    r0, r1 = PF.Runner(TorchBackend(lm), proto, obj.grader), PF.Runner(TorchBackend(lm), new, obj.grader)
    before = EV.original_set(r0, items, samples=1, seed=0)
    after = EV._run_set(r1, items_for(settings, items), samples=1, seed=0)
    a, graded = EV._metric(before)
    b, _ = EV._metric(after)
    paired = EV._paired(a, b)
    rails = []
    sg = next((g for g in obj.guardrails if g.suite), None)
    keep_refusals = not (sg is not None and sg.allow_safety_loss)
    suite = [{"id": n, "messages": [{"role": "user", "content": p}], "lang": lang,
              **({"expect": "refuse"} if kind == "safety" and keep_refusals else {})}
             for n, p, lang, kind in R.SUITE if kind != "safety" or keep_refusals]
    sp0 = dataclasses.replace(proto, system=None, max_new_tokens=min(proto.max_new_tokens, 256))
    sp1 = with_settings(sp0, settings)
    s0 = EV.original_set(PF.Runner(TorchBackend(lm), sp0, "detector:any"), suite, samples=1, seed=0)
    s1 = EV._run_set(PF.Runner(TorchBackend(lm), sp1, "detector:any"), items_for(settings, suite), samples=1, seed=0)
    m0, _ = EV._metric(s0)
    m1, _ = EV._metric(s1)
    broke = [x["id"] for x, u, v in zip(s0, m0, m1) if u > v]
    rails.append({"name": "wsr_regression (generated under the settings)", "kind": "suite", "prompts": len(suite),
                  "new_failures": broke, "holds": not broke})
    for g in obj.guardrails:
        if g.data:
            gi = EV._guard_items(obj, g)
            gb = EV.original_set(PF.Runner(TorchBackend(lm), proto, g.grader or "detector:any"), gi, samples=1,
                                 seed=0)
            ga = EV._run_set(PF.Runner(TorchBackend(lm), new, g.grader or "detector:any"), items_for(settings, gi),
                             samples=1, seed=0)
            rails.append(EV._guard_result(g, gb, ga, 2 * len(gi)))
    verdict = EV._verdict(paired, all(x["holds"] is not False for x in rails))
    looks = EV.HeldoutLog(obj).add({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "patch": f"settings: {json.dumps(settings)}", "items": len(items),
                                    "verdict": verdict})
    return {"settings": dict(settings), "n": len(items), "power": EV.power(a),
            "metric": "accuracy" if graded else "share with no failure",
            "paired": paired, "rates_before": EV._rates(before), "rates_after": EV._rates(after),
            "guardrails": rails, "verdict": verdict, "heldout_looks": looks, "seconds": round(time.time() - t0, 1)}


def render_settings(js: dict, proto: Protocol) -> list[str]:
    p = js["paired"]
    lines = [f"Settings `{json.dumps(js['settings'])}` on {js['n']} held-out items ({js['metric']}): "
             f"{p.get('before', 0):.1%} -> {p.get('after', 0):.1%} ({p.get('diff', 0):+.1%} +- {p.get('se', 0):.1%} SE)"
             f"; verdict: **{js['verdict']}**",
             "- failure groups before: " + ", ".join(f"{k} {v:.0%}" for k, v in js["rates_before"].items()),
             "- failure groups after: " + ", ".join(f"{k} {v:.0%}" for k, v in js["rates_after"].items())]
    for g in js["guardrails"]:
        extra = f", new failures: {', '.join(g['new_failures']) or 'none'}" if "new_failures" in g else ""
        lines.append(f"- guardrail {g['name']}: {'holds' if g['holds'] else 'FAILS'}{extra}")
    if js.get("power") and not js["power"]["can_show_improvement"]:
        lines.append(f"- note: {js['power']['note']}")
    if js.get("heldout_looks"):
        lines.append(f"- held-out set looked at {js['heldout_looks']} time(s) so far")
    if js["verdict"] == "improved":
        lines += ["", "To apply:", *[x for x in settings_howto(proto, js["settings"]) if x]]
    else:
        lines.append("Not recommended: the held-out judge did not call it an improvement.")
    return lines


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
                allow_safety_loss: bool = False, steering: str = "fallback", weight_edits: bool = True,
                mode: str = "deep", judge: str = "torch", judge_quant: str = "Q8_0", consent=None,
                model_dir: Optional[Path] = None, server_bin: Optional[str] = None) -> dict[str, Any]:
    from ..deep.run import diagnose
    from ..surgery import playbook as P
    from ..surgery import regress as R

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
    plans_before = set(getattr(lm, "batch_plans", {}) or {})
    if prior is not None:
        say(f"practice replies from wsr objective from-failure reused ({len(prior)} items: same weights, protocol "
            "and messages)")
    try:
        prof = PF.profile(prior and lm or src, practice[:profile_limit], proto, grader=obj.grader, name=obj.name,
                          model=str(obj.model or model_dir or ""), test_k=2, test_n=2, replies=prior)
    except Exception as e:                               # noqa: BLE001 - the server is optional
        if src is lm:
            raise
        say(f"  profile on llama-server failed ({type(e).__name__}: {str(e)[:160]}); profiling on PyTorch")
        rec["stages"]["profile_fallback"] = f"{type(e).__name__}: {str(e)[:300]}"
        if hasattr(src, "close"):
            src.close()
        profile_source = None
        prof = PF.profile(lm, practice[:profile_limit], proto, grader=obj.grader, name=obj.name,
                          model=str(obj.model or model_dir or ""), test_k=2, test_n=2, replies=prior)
    rec["stages"]["profile_reused"] = prior is not None
    # The profile's main pass is practice_eval's baseline when it ran here, on this PyTorch model (not on a
    # server, not reused from another process): same protocol, seeds, order and, with these plans, chunks.
    on_lm = prior is None and profile_source is None and hasattr(lm, "batch_plans")
    profile_plans = {k: v for k, v in lm.batch_plans.items() if k not in plans_before} if on_lm else None
    rec["stages"]["profile"] = prof.to_dict()
    if profile_source is not None and hasattr(profile_source, "close"):
        # The profile's server is not needed again (the judge starts its own); free its memory and GPU now
        # instead of holding them through the diagnosis and the playbook.
        profile_source.close()
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
    # A loop is diagnosed at the length it was seen at (the protocol's cap): MATH loops start after hundreds of
    # tokens, and a shorter run sees a healthy reply. Other failures show in the first tokens.
    max_new = proto.max_new_tokens if kind == "loop" else min(proto.max_new_tokens, 48)
    rec["stages"]["group"] = {"label": label, "count": group["count"], "says": group["says"], "kind": kind,
                              "failing_used": [it["id"] for it in fail[:4]], "working_used": [it["id"] for it in work[:3]]}
    say(f"\nworking on {label} ({group['count']} of {prof.n} practice replies): {group['says']}")

    # 2. diagnose, 3. playbook
    tk = proto.template_kwargs or None
    res = diagnose(lm, fail[0]["messages"], kind=kind, failing=tuple(_user_text(i) for i in fail[1:4]),
                   passing=tuple(_user_text(i) for i in work[:3]), max_new=max_new, template_kwargs=tk,
                   out_dir=out / "deep")
    cut = next((f for f in res.report.findings if f.code == "DEEP_CUT_SHORT"), None)
    if cut is not None and int(cut.evidence.get("longer", 0)) > max_new:
        # The diagnosis saw the failure only past its token limit: once more at the length it names.
        max_new = int(cut.evidence["longer"])
        say(f"diagnosis: the failure starts after the limit; diagnosing again with {max_new} new tokens")
        res = diagnose(lm, fail[0]["messages"], kind=kind, failing=tuple(_user_text(i) for i in fail[1:4]),
                       passing=tuple(_user_text(i) for i in work[:3]), max_new=max_new, template_kwargs=tk,
                       out_dir=out / "deep")
    rec["stages"]["diagnosis"] = {"summary": res.summary, "confidence": res.confidence, "max_new": max_new}
    say(f"diagnosis: {res.summary}")
    # The verification uses the limits of the goal's own suite guardrail (the judge applies the same ones), so a
    # candidate is not verified against one set of limits and judged against another.
    sg = next((g for g in obj.guardrails if g.suite), None)
    reg_budget = (R.Budget(sg.max_drift, sg.min_agreement, sg.allow_safety_loss or allow_safety_loss)
                  if sg is not None else R.Budget(allow_safety_loss=allow_safety_loss))
    if not weight_edits:
        # The weights must not change: the playbook may only steer, and the agent's edit tools are refused.
        steering = "only"
        if agent_budget is None and provider is not None:
            from ..agent import Budget as AgentBudget
            agent_budget = AgentBudget()
        if agent_budget is not None:
            agent_budget.weight_edits = False
        rec["weight_edits"] = False
    rep = P.repair(lm, res, max_new=max_new, template_kwargs=tk, mode=mode, playbook=playbook, log=say,
                   budget=reg_budget, steering=steering)
    rec["stages"]["playbook"] = {"status": rep.status, "summary": rep.summary,
                                 "patch": rep.patch.label if rep.patch is not None else None,
                                 "reference_runs": (rep.checks or {}).get("reference_runs")}
    candidate, by = (rep.patch, "playbook") if rep.status == "fixed" else (None, None)
    # The verification already ran the regression suite on this patch with this template; the judge's
    # guardrail reuses it (same computation, same numbers) instead of running it again.
    suite_result = rep.regression if rep.status == "fixed" else None

    # 4. agent, 5. the judge. A playbook candidate that the held-out judge rejects hands over to the agent once
    # (when there is one): it hears only that the candidate was rejected, not the held-out numbers.
    from ..consent import Consent
    from . import judge as JG

    def judge_patch(candidate, by, suite_result) -> dict:
        from ..surgery.edits import base_marks

        if not candidate.base:
            candidate.base = base_marks(lm, candidate)   # the unedited weights: saved with the patch
        say(f"\njudging {candidate.label} ({by}) on the held-out set")
        jd = JG.choose(judge, lm=lm, model_dir=model_dir, patch=candidate, out=out, consent=consent or Consent(),
                       quant=judge_quant, server_bin=server_bin, max_new=proto.max_new_tokens, log=say)
        try:
            ev = None
            if jd.kind == "llama-server":
                try:
                    ev = EV.evaluate(jd.source, obj, patched_source=jd.patched, guard_source=lm, patch=candidate,
                                     patch_name=candidate.label, split="heldout", limit=eval_limit,
                                     suite_result=suite_result, log=False)
                    EV.HeldoutLog(obj).add({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "patch": candidate.label,
                                            "items": ev["items"], "verdict": ev["verdict"]})
                    ev["heldout_looks"] = EV.HeldoutLog(obj).count()
                except Exception as e:                   # noqa: BLE001 - the fast judge is optional
                    jd.notes.append(f"llama-server judge failed mid-run ({type(e).__name__}: {str(e)[:160]}); "
                                    "judged on PyTorch (batched)")
                    say(f"  judge: {jd.notes[-1]}")
                    jd.close()                           # free the GPU before generating on PyTorch
                    jd.kind, ev = "pytorch", None
            if ev is None:
                ev = EV.evaluate(lm, obj, patch=candidate, patch_name=candidate.label, split="heldout",
                                 limit=eval_limit, suite_result=suite_result)
        finally:
            jd.close()
        ev["judge"] = {"kind": jd.kind, "notes": jd.notes, "files": jd.files}
        say(EV.render(ev))
        return ev

    tools = None
    rejected = None                                       # the playbook's candidate, when the judge said no
    if candidate is not None and provider is not None and rep.ctx is not None:
        ev = judge_patch(candidate, by, suite_result)
        if ev["verdict"] == "improved":
            return _ship(rec, out, t0, candidate, by, ev, obj, consent, decisions)
        candidate.checks["objective"] = {"name": obj.name, "verdict": ev["verdict"], "paired": ev.get("paired"),
                                         "by": by}
        candidate.save(out / "candidate_playbook")
        rec["stages"]["judge_playbook"] = ev
        rejected = candidate.label
        candidate, by, suite_result = None, None, None
    if rep.status == "steered" and rep.steering is not None:
        # The playbook's steering: judged like the agent's (same weights with and without it, the suite under it).
        sv = rep.steering
        st = {"_vector": sv.vector, "alpha": sv.alpha, "layers": sv.layers, "vector": sv.label, "mode": sv.mode}
        say(f"\njudging the playbook's steering ({sv.label}, alpha {sv.alpha:.3f} after layers {sv.layers}) on the "
            "held-out set")
        js = judge_steering(lm, obj, st, limit=eval_limit, say=say)
        js["by"] = "playbook"
        js["practice"] = None
        say(f"  steering: {js['verdict']}")
        if js["verdict"] == "improved":
            js["files"] = str(save_steering(st, out))
            rec["stages"]["steering"] = js
            rec["outcome"] = "steering recommended"
            rec["says"] = f"serve with the playbook's steering ({sv.label}): held-out {js['verdict']}"
            return _finish(rec, out, t0)
        rec["stages"]["steering_playbook"] = js
        rejected = f"steering: {sv.label}"
        if provider is None:
            rec["outcome"] = "no candidate"
            pw = js.get("power") or {}
            rec["says"] = (f"the playbook's steering ({sv.label}) was not accepted by the held-out judge: "
                           f"{js['verdict']}" + (f" ({pw['note']})" if pw and not pw["can_show_improvement"] else ""))
            return _finish(rec, out, t0)
    if candidate is None and provider is not None and (rep.status == "could_not_fix" or rejected):
        from ..agent import run as agent_run

        tools = ObjectiveTools(lm, obj, practice, max_evals=practice_evals,
                               baseline=prof.replies if on_lm else None, plans=profile_plans)
        if rejected:
            tools.note = (f"The rule-based playbook's candidate ({rejected}) passed its own checks but the held-out "
                          "judge did not accept it (held-out numbers are not shown to you). Look for a different fix.")
            why = "held-out judge rejected the playbook's candidate"
        else:
            why = "playbook could not fix it" if playbook else "playbook was skipped"
        say(f"\nthe {why}; handing over to the agent ({provider.name}) with the objective")
        rep_for_agent = rep if rep.status == "could_not_fix" else dataclasses.replace(
            rep, status="could_not_fix", patch=None, summary=f"{rep.summary}; not accepted by the held-out judge")
        rep2 = agent_run(lm, res, rep_for_agent, provider, budget=agent_budget, limits=agent_limits, out=out,
                         log=say, program=tools)
        wb = tools.wb
        rec["stages"]["agent"] = {**(rep2.checks or {}).get("agent", {}), "leaderboard": tools.board,
                                  "practice_evals": tools.scores}
        if wb is not None and wb.done and "final" in wb.done:
            candidate, by = wb.edits[wb.done["final"]], f"agent (final choice: {wb.done['rationale'][:200]})"
            suite_result = (wb.results.get(wb.done["final"]) or {}).get("regression")
        elif rep2.status == "fixed" and rep2.patch is not None:
            candidate, by = rep2.patch, "agent (submitted after verify_edit)"
            suite_result = rep2.regression
    if tools is not None and tools.wb is not None and tools.wb.done and "final_steering" in tools.wb.done \
            and candidate is None:
        st = tools.steerings[tools.wb.done["final_steering"]]
        by = f"agent (final choice: {tools.wb.done['rationale'][:200]})"
        say(f"\njudging the steering {st['vector']} x {st['alpha']} after layers {st['layers']} ({by}) on the "
            "held-out set")
        js = judge_steering(lm, obj, st, limit=eval_limit, say=say)
        js["by"] = by
        js["practice"] = {k: st.get(k) for k in ("n", "before", "after", "diff", "se")}
        rec["stages"]["steering"] = js
        rec["stages"]["agent"]["steer_evals"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                                 for k, v in tools.steerings.items()}
        if js["verdict"] == "improved":
            js["files"] = str(save_steering(st, out))
        say(f"  steering: {js['verdict']}")
        rec["outcome"] = "steering recommended" if js["verdict"] == "improved" else "no candidate"
        rec["says"] = f"steering {st['vector']} x {st['alpha']} after layers {st['layers']}: held-out {js['verdict']}"
        return _finish(rec, out, t0)
    pick = None
    if tools is not None:
        wb = tools.wb
        if tools.steerings and "agent" in rec["stages"]:
            rec["stages"]["agent"]["steer_evals"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                                     for k, v in tools.steerings.items()}
        if wb is not None and wb.done and "final_settings" in wb.done:
            pick = (tools.settings[wb.done["final_settings"]], f"agent (final choice: {wb.done['rationale'][:200]})")
        elif candidate is None and tools.best_settings() is not None:
            pick = (tools.best_settings(), "agent's best settings on practice (gain of at least one standard error)")
        if tools.settings:
            rec["stages"]["agent"]["settings_evals"] = tools.settings
    if pick is not None and candidate is None:
        sel, by = pick
        say(f"\njudging the settings {sel['settings']} ({by}) on the held-out set")
        js = judge_settings(lm, obj, sel["settings"], limit=eval_limit, say=say)
        js["by"] = by
        js["practice"] = {k: sel.get(k) for k in ("n", "before", "after", "diff", "se")}
        rec["stages"]["settings"] = js
        say("\n".join(render_settings(js, proto)))
        rec["outcome"] = "settings recommended" if js["verdict"] == "improved" else "no candidate"
        rec["says"] = (f"serve with {json.dumps(sel['settings'])}: held-out {js['verdict']}"
                       if js["verdict"] == "improved" else f"settings {json.dumps(sel['settings'])}: {js['verdict']}")
        return _finish(rec, out, t0)
    if candidate is None:
        rec["outcome"] = "not shipped" if rejected else "no candidate"
        done = (tools.wb.done or {}) if tools is not None and tools.wb is not None else {}
        rec["says"] = (f"the agent gave up: {done['give_up'][:300]}" if done.get("give_up") else
                       f"the playbook's candidate ({rejected}) was not accepted by the held-out judge" if rejected
                       else rep.summary)
        if rejected:
            rec["patch"] = {"label": rejected, "by": "playbook", "dir": str(out / "candidate_playbook")}
        return _finish(rec, out, t0)
    ev = judge_patch(candidate, by, suite_result)
    return _ship(rec, out, t0, candidate, by, ev, obj, consent, decisions)


def _ship(rec: dict, out: Path, t0: float, candidate, by: str, ev: dict, obj: Objective, consent, decisions) -> dict:
    rec["stages"]["judge"] = ev
    shipped = ev["verdict"] == "improved"
    candidate.checks["objective"] = {"name": obj.name, "verdict": ev["verdict"], "paired": ev.get("paired"),
                                     "by": by}
    candidate.save(out / ("patch" if shipped else "candidate"))
    rec["outcome"] = "shipped" if shipped else "not shipped"
    rec["patch"] = {"label": candidate.label, "by": by, "dir": str(out / ("patch" if shipped else "candidate"))}
    pw = ev.get("power") or {}
    rec["says"] = ev["verdict"] + (f" ({pw['note']})" if not shipped and pw and not pw["can_show_improvement"]
                                   else "")
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
             f"- guardrails: {', '.join(g.get('suite') or g.get('data', '?') for g in o['guardrails']) or 'none'}",
             *(["- weight edits were off (--no-weight-edits): only steering and settings could fix it"]
               if rec.get("weight_edits") is False else []), ""]
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
    if "judge_playbook" in st:
        lines += ["## Held-out judge on the playbook's candidate (not accepted; handed to the agent)", "", "```",
                  EV.render(st["judge_playbook"]), "```", ""]
    if "judge" in st:
        j = st["judge"].get("judge") or {}
        lines += ["## Held-out judge", f"Judge: {j.get('kind', 'pytorch')}"
                  + (f" ({'; '.join(j['notes'])})" if j.get("notes") else ""), "", "```", EV.render(st["judge"]),
                  "```", ""]
        if j.get("files", {}).get("candidate"):
            lines += [f"Served file of the candidate: `{j['files']['candidate']}`", ""]
    if "steering_playbook" in st and "steering" not in st:
        js = st["steering_playbook"]
        p = js["paired"]
        how = (f"Clamp `{js['vector']}` to {js['alpha']:.3f}" if js.get("mode") == "clamp"
               else f"Add {js['alpha']:.3f} x `{js['vector']}`")
        lines += ["## The playbook's steering on the held-out set (not accepted)", "",
                  f"{how} after layers {js['layers']}: {p.get('before', 0):.1%} -> "
                  f"{p.get('after', 0):.1%} ({p.get('diff', 0):+.1%} +- {p.get('se', 0):.1%} SE); verdict: "
                  f"{js['verdict']}"]
        for g in js["guardrails"]:
            lines.append(f"- guardrail {g['name']}: {'holds' if g['holds'] else 'FAILS'}")
        if js.get("power") and not js["power"]["can_show_improvement"]:
            lines.append(f"- note: {js['power']['note']}")
        lines.append("")
    if "steering" in st:
        js = st["steering"]
        p = js["paired"]
        how = (f"Clamp `{js['vector']}` to {js['alpha']:.3f}" if js.get("mode") == "clamp"
               else f"Add {js['alpha']} x `{js['vector']}`")
        lines += ["## Recommended steering" if js["verdict"] == "improved" else "## Steering tried on the held-out set",
                  f"Chosen by: {js.get('by', '')}; practice: {js.get('practice')}", "",
                  f"{how} after layers {js['layers']} (every position). Held-out "
                  f"({js['n']} items, {js['metric']}): {p.get('before', 0):.1%} -> {p.get('after', 0):.1%} "
                  f"({p.get('diff', 0):+.1%} +- {p.get('se', 0):.1%} SE); verdict: **{js['verdict']}**"]
        for g in js["guardrails"]:
            extra = (f" (drift {g.get('mean_delta_nll')}, agreement {g.get('top1_agreement')}, new failures "
                     f"{len(g.get('new_failures') or [])}, refusals lost {len(g.get('safety_lost') or [])})"
                     if g.get("kind") == "suite" else "")
            lines.append(f"- guardrail {g['name']}: {'holds' if g['holds'] else 'FAILS'}{extra}")
        if js.get("power") and not js["power"]["can_show_improvement"]:
            lines.append(f"- note: {js['power']['note']}")
        if js.get("heldout_looks"):
            lines.append(f"- held-out set looked at {js['heldout_looks']} time(s) so far")
        if js.get("files"):
            lines += ["", f"Files: `{js['files']}` (the vector, its settings, and APPLY.md with the hook to apply it). "
                      "The weights are unchanged."]
        lines.append("")
    if "settings" in st:
        js = st["settings"]
        lines += ["## Recommended settings" if js["verdict"] == "improved" else "## Settings tried on the held-out set",
                  f"Chosen by: {js.get('by', '')}; practice: {js.get('practice')}", "",
                  *render_settings(js, Protocol(**{k: v for k, v in o.get("protocol", {}).items()
                                                   if k in ("template_kwargs", "sampling", "max_new_tokens", "system")})),
                  ""]
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
