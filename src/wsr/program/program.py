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

import contextlib
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Optional

from . import attempts as AT
from . import evaluate as EV
from . import profile as PF
from .backends import TorchBackend
from ..surgery.steer import STEER_APPLY  # noqa: F401 - re-exported (tests, docs)
from .objective import Objective, Protocol, load_items

# Serving settings the agent may test on PyTorch, with the range each is checked against.
SETTINGS_RANGES = {"repetition_penalty": (1.0, 1.5), "temperature": (0.0, 2.0), "top_p": (0.05, 1.0),
                   "top_k": (1, 1000), "min_p": (0.0, 0.5),
                   "dry_multiplier": (0.0, 5.0), "dry_base": (1.0, 4.0), "dry_allowed_length": (1, 20)}
LLAMA_FLAGS = {"repetition_penalty": "--repeat-penalty", "temperature": "--temp", "top_p": "--top-p",
               "top_k": "--top-k", "min_p": "--min-p", "dry_multiplier": "--dry-multiplier",
               "dry_base": "--dry-base", "dry_allowed_length": "--dry-allowed-length"}


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
        if k == "dry_allowed_length" and int(v) != v:
            return "dry_allowed_length must be a whole number of tokens"
    if any(k in settings for k in ("dry_base", "dry_allowed_length")) and "dry_multiplier" not in settings:
        return "dry_base and dry_allowed_length shape the DRY penalty: give dry_multiplier (> 0) to turn it on"
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


STAGE = 30          # early exit: a measuring call with n >= 2 x STAGE practice items runs the first STAGE first
STOP_P = 0.05       # ... and stops when they are worse at this level of the exact one-sided sign test,
STOP_BROKEN = 3     # or newly break at least this many replies the original answered (more than it mends)
BROKEN = EV.BROKEN                                      # replies with no behaviour (garbage, empty, a loop)


def _wb_original(wb):
    """The workbench's original weights for the duration (multi-stage surgery: agent/workbench.py), or nothing to do."""
    f = getattr(wb, "original", None)
    return f() if callable(f) else contextlib.nullcontext()


def _wb_candidate(wb, edit_id: str) -> tuple:
    """(id, patch) of what is measured and judged for `edit_id`: with a committed stack, the stack plus the edit as one
    patch on the original weights."""
    f = getattr(wb, "candidate", None)
    return f(edit_id) if callable(f) else (edit_id, wb._edit(edit_id))


def _wb_stack(wb) -> tuple:
    """(id, patch) of the workbench's committed stack, or (None, None)."""
    f = getattr(wb, "stack_candidate", None)
    got = f() if callable(f) else None
    return got if got is not None else (None, None)


@contextlib.contextmanager
def _under(lm, base=None, kw: Optional[dict] = None):
    """A candidate's "after" side: the weight patch it sits on (a committed stack), then its steering hooks."""
    from ..surgery.edits import applied

    with (applied(lm, base) if base is not None else contextlib.nullcontext()):
        with (lm.steer(**kw) if kw is not None else contextlib.nullcontext()):
            yield


def _base_note(base) -> dict:
    return {"base_patch": {"label": base.label, "parts": (base.spec or {}).get("parts"),
                           "weights_changed": base.n_weights_changed}} if base is not None else {}


def stage_problem(before: list[dict], after: list[dict]) -> Optional[str]:
    """Why the first stage of a measuring call says the candidate is not worth the rest, or None. Practice-side
    only: it saves generations and never decides anything the judge decides."""
    a, _ = EV._metric(before)
    b, _ = EV._metric(after)
    pr = EV._paired(a, b)
    wins, losses = pr.get("wins", 0), pr.get("losses", 0)
    p = EV._sign_p(losses, wins) if losses > wins else None
    if p is not None and p <= STOP_P:
        return (f"clearly worse on the first {len(before)} items: {losses} worse, {wins} better (exact sign test "
                f"p={p:.3f})")
    broke = mended = 0
    for x, y in zip(before, after):
        was, now = bool(BROKEN & set(x["labels"])), bool(BROKEN & set(y["labels"]))
        broke += now and not was
        mended += was and not now
    if broke >= STOP_BROKEN and broke > mended:
        return (f"breaks {broke} replies of the first {len(before)} that the original answered (garbage, a loop or "
                f"an empty reply), and mends {mended}")
    return None


class ObjectiveTools:
    """The agent's extra tools for an objective; practice data only."""

    system = ("This run has an OBJECTIVE, not only one failing prompt. The failing and working prompts you see "
              "come from the objective's practice set. Besides verify_edit (one prompt's checks), measure an edit "
              "on the practice set with practice_eval: it reports the paired score change against the original "
              "model and the failure-group rates. Keep promising edits with checkpoint. End with final_choice on "
              "the candidate you want judged: a validation set you cannot see, plus the guardrails listed in the "
              "brief, decides at once. A rejected candidate is answered with abstract symptoms only (the gain "
              "rounded, which guardrail failed and what kind of thing it protects, counts in buckets; never prompts) "
              "and, while attempts remain, you may choose a DIFFERENT candidate; every attempt makes the bar for the "
              "next one stricter, so measure before you choose. A candidate that passes verify_edit or helps practice "
              "is not enough on its own: it must also keep what every guardrail protects. Check that on practice "
              "before final_choice: every measuring tool reports regression_suite (the judge's own check on wsr's "
              "regression suite, run the same way: holds false means the judge would reject the candidate) and "
              "harmful_kept (wsr's harmful requests, still refused?). By default a measuring call stops early when the "
              "candidate cannot pass or is clearly worse on its first 30 items (stopped_early says why); "
              "stop_early: false measures every item. With committed edits (commit_edit), every measuring tool and "
              "final_choice take the committed stack plus the candidate (an edit, a steering or settings) together, "
              "against the ORIGINAL model: what the judge sees and what would ship.")

    EARLY = (" stop_early (default true): the regression suite and wsr's harmful requests are checked first and, "
             "for n >= 60, the first 30 items before the rest; the call stops when the suite fails (the judge would "
             "reject the candidate), harmful refusals are lost, or the first 30 are clearly worse (exact sign test "
             "at 5%, or 3+ replies newly broken). A call stopped before any practice item does not use the budget.")

    def __init__(self, lm, obj: Objective, items: list[dict], *, max_evals: int = 10, default_n: int = 20,
                 baseline: Optional[list[dict]] = None, plans: Optional[dict] = None, attempts: int = 1,
                 judge=None):
        """baseline: the profile's replies of the original on items[:len(baseline)], when that pass ran on this
        very model in this process (same protocol, seeds and order as practice_eval's own baseline); plans: the
        batch plans it ran in. With both, practice_eval at that n reuses the baseline and runs the edited model
        in the same chunks, so the comparison is exactly the one a fresh baseline gives, without regenerating
        the original.

        attempts / judge: final_choice is judged at once by `judge(kind, id, wb)` (a validation result); a rejected
        candidate is answered with abstract symptoms (attempts.py) while attempts remain. Without a judge,
        final_choice only records the choice (the caller judges it)."""
        self.lm, self.obj, self.items = lm, obj, items
        self.max_evals, self.default_n = max_evals, default_n
        self.attempts_max, self.judge = max(1, int(attempts)), judge
        self.attempts: list[dict] = []                 # judged attempts: kind, id, verdict, feedback (no items)
        self.judged: dict[str, dict] = {}              # fingerprint -> validation result
        self.accepted: Optional[tuple] = None          # (kind, id, validation result) of the accepted attempt
        self.results: list[dict] = []                  # the full validation result of each attempt (for the report)
        self._mirror_items = AT.harm_mirror() if AT.harm_check_on(obj) else []
        self._mirror_base: Optional[list[dict]] = None
        self.evals = 0
        self.base: dict[int, list[dict]] = {}
        self.base_seg: dict[tuple[int, int], list[dict]] = {}     # the original on an early-exit stage
        self.suite_known: dict[str, dict] = {}                    # edit id -> its regression-suite check (judge reuses)
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
        EARLY = self.EARLY
        self.schemas = [
            {"name": "practice_eval", "description": "Score an edit on the first n practice items against the "
             "original model (same seeds): score before / after, paired difference with its standard error, items "
             "better / worse, failure-group rates. The original's score comes with every call: never spend one "
             "on a no-op edit to get a baseline (the profile in the brief already gives it). It "
             "also reports harmful_kept: wsr's own harmful requests (not the guardrail's data), refused before vs "
             "under the edit, a practice-side check of the safety guardrail. regression_suite: the judge's own check "
             "on wsr's regression suite for this edit (new failures, drift, top-token agreement, refusals; a "
             "verify_edit run of the same edit is reused). Counts against a small budget, and each call generates up "
             "to n full replies." + EARLY,
             "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                               "n": {"type": "integer", "minimum": 1},
                                                               "stop_early": {"type": "boolean"}},
                              "required": ["edit_id"]}},
            {"name": "settings_eval", "description": "Score serving settings instead of an edit: the ORIGINAL "
             "weights on the first n practice items with these settings on top of the objective's protocol, against "
             "the same baseline as practice_eval (paired, failure-group rates). For failures that decoding causes or "
             "cures, such as loops: e.g. {\"repetition_penalty\": 1.1}. Allowed: repetition_penalty [1, 1.5], "
             "dry_multiplier [0, 5] with dry_base [1, 4] (default 1.75) and dry_allowed_length [1, 20] (default 2), "
             "temperature [0, 2], top_p, top_k, min_p, a different system prompt (\"system\": text, or null for none) "
             "and chat-template arguments (\"template_kwargs\"). Shares the practice_eval budget. A settings candidate can be "
             "the final_choice: it is judged on the held-out set like an edit and reported as recommended settings. "
             "repetition_penalty divides the logit of every token already in the context, the PROMPT's words "
             "included, so it can change more than repetition (e.g. how the model refuses); DRY (llama.cpp's "
             "--dry-multiplier) penalises only a token that would extend a sequence already repeated, and stops at "
             "newlines, ':', quotes and '*', so list structure is not penalised. "
             "steering_id (from steer_eval) measures these settings TOGETHER with that steering, as one runtime "
             "bundle served with both: for a fix that needs both, e.g. a penalty that ends loops while a harm_probe "
             "gate keeps harmful requests refused. Its settings_id is then the bundle, judged and shipped as one. "
             "The judge does NOT measure drift for settings alone (they change what the model is asked and how it "
             "decodes, not the model): it generates wsr's regression suite under the settings and requires that no "
             "reply that was fine fails and every refusal is kept. regression_suite reports exactly that check (with a "
             "steering_id, also the steering's drift, agreement and refusals); no need to estimate it by hand." + EARLY,
             "input_schema": {"type": "object", "properties": {"settings": {"type": "object"},
                                                               "n": {"type": "integer", "minimum": 1},
                                                               "steering_id": {"type": "string"},
                                                               "stop_early": {"type": "boolean"}},
                              "required": ["settings"]}},
            {"name": "steer_eval", "description": "Score ACTIVATION STEERING instead of an edit: the original "
             "weights on the first n practice items with alpha x vector added to the residual stream after each "
             "of the given layers (every position), against the same baseline as practice_eval. For behaviour "
             "spread over many layers that no small weight edit separates. vector: a direction id ('d1', 'd1@12') "
             "or a workspace variable of d_model numbers. mode \"clamp\" instead lowers the component along the "
             "vector to the level alpha wherever it is above it and leaves every other position alone: for a "
             "failure that fires only sometimes (a constant shift also moves healthy text). mode \"gated\" takes a "
             "gate id from harm_probe as the vector (alpha and layers are ignored): the refusal tied to the model's "
             "own reading of harm, for over- and under-refusal. The result also has harmful_kept "
             "(see practice_eval). A steering candidate can be the final_choice: it is "
             "judged on the held-out set with the regression suite run under the steering, and reported as a "
             "steering vector to apply at inference (no weights change). To REMOVE a direction instead, "
             "build_edit orthogonalise does it exactly in the weights. Shares the practice_eval budget. "
             "regression_suite: the judge's own check of the steering on wsr's regression suite (drift, agreement, "
             "new failures, refusals)." + EARLY,
             "input_schema": {"type": "object", "properties": {
                 "vector": {"type": "string"}, "alpha": {"type": "number"},
                 "mode": {"type": "string", "enum": ["add", "clamp", "gated"]},
                 "layers": {"type": "array", "items": {"type": "integer"}},
                 "n": {"type": "integer", "minimum": 1}, "why": {"type": "string"},
                 "stop_early": {"type": "boolean"}},
                 "required": ["vector"]}},
            {"name": "checkpoint", "description": "Keep a measured candidate on the leaderboard with a note: an "
             "edit (edit_id), a steering (steering_id) or serving settings (settings_id); its last practice result is "
             "attached. Returns the leaderboard.",
             "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                               "steering_id": {"type": "string"},
                                                               "settings_id": {"type": "string"},
                                                               "note": {"type": "string"}},
                              "required": ["note"]}},
            {"name": "final_choice", "description": "Submit this edit (edit_id), these serving settings "
             "(settings_id) or this steering (steering_id) to the judge, which runs at once: the validation set you "
             "cannot see (exact sign test, minimum gain) and every guardrail in the brief. Accepted: the run ends, and "
             "the confirmation set, if the goal has one, decides once. Rejected: you get abstract symptoms (the gain "
             "rounded to 5 points, whether it was significant, which guardrail failed and what kind of thing it "
             "protects, counts in buckets; never prompts or items) and, while attempts remain, you may submit a "
             "DIFFERENT candidate; an identical one is refused, and each attempt makes the next one's bar stricter. "
             "The candidate needs a practice_eval / settings_eval / steer_eval first. Before submitting, check on "
             "practice what each guardrail protects (harmful_kept in the measuring tools).",
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
               "- guardrails (the judge checks every one; you cannot see their data):\n"
               + "".join(f"  - {x}\n" for x in AT.guardrail_brief(o) or ["none"])
               + f"- attempts: {self.attempts_max} judged final_choice(s) at most; a rejection returns abstract "
               "symptoms only\n"
               f"- practice_eval budget: {self.max_evals} calls (settings_eval shares it)\n"
               "- the held-out set is not available to you\n"
               "- if the evidence says decoding causes or cures the failure (loops that sampling or a repetition "
               "penalty escape), measure that with settings_eval: a settings fix needs no weight edit\n"
               "- if the failure is conditional (it must change only on some prompts or positions, as a loop does), "
               "gate_sweep runs the playbook's gated-neuron search: a plan at every layer and edits at a ladder of "
               "strengths sized from the failure's gap, so no layer or strength has to be guessed\n"
               "- if the behaviour is spread over many layers and every small edit also changes what must stay (a "
               "shared mechanism), measure activation steering with steer_eval"
               "\n- if most replies are garbage (or the model may have been damaged by a merge, conversion or "
               "fine-tune), weight_scan lists rows / columns many times larger than their matrix and builds restoring "
               "edits"
               "\n- for over- or under-refusal, harm_probe says whether the model itself reads harm and builds a gate "
               "on that reading; steer_eval {vector: <gate id>, mode: gated} measures it"
               "\n- if one fix reveals another fault (restored weights answer harmful requests the damaged model only "
               "garbled; a loop fix leaves a wrong language), commit_edit the first fix: every tool then diagnoses, "
               "probes and calibrates on the repaired model, and the measuring tools and final_choice take the committed "
               "stack plus the next candidate (an edit, a steering or settings) against the original weights")
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

    def _mirror(self, ctx=None, proto=None, transform=None) -> Optional[dict]:
        """Refusal goals: wsr's held-back harmful requests, refused before vs under the candidate (attempts.py)."""
        if not self._mirror_items:
            return None
        if self._mirror_base is None:
            r0 = PF.Runner(TorchBackend(self.lm), self.obj.protocol, "detector:refusal")
            self._mirror_base = EV.original_set(r0, self._mirror_items, samples=1, seed=0)
            self.generations += r0.generations
        r1 = PF.Runner(TorchBackend(self.lm), proto or self.obj.protocol, "detector:refusal")
        items = transform(self._mirror_items) if transform else self._mirror_items
        with (ctx() if ctx else contextlib.nullcontext()):
            after = EV._run_set(r1, items, samples=1, seed=0)
        self.generations += r1.generations
        return AT.mirror_result(self._mirror_base, after)

    def _base(self, n: int) -> list[dict]:
        if n not in self.base:
            r = PF.Runner(TorchBackend(self.lm), self.obj.protocol, self.obj.grader)
            self.base[n] = EV.original_set(r, self.items[:n], samples=1, seed=0)
            self.generations += r.generations
        return self.base[n]

    def _base_seg(self, s: int, e: int) -> list[dict]:
        """The original on practice items s..e-1: a whole call (s = 0) or an early-exit stage, each its own prompt
        list with its own batches, so the candidate's run of the same list is numerically paired with it."""
        if s == 0:
            return self._base(e)
        if (s, e) not in self.base_seg:
            r = PF.Runner(TorchBackend(self.lm), self.obj.protocol, self.obj.grader)
            self.base_seg[(s, e)] = EV.original_set(r, self.items[s:e], samples=1, seed=0)
            self.generations += r.generations
        return self.base_seg[(s, e)]

    def _suite_guard(self):
        return next((g for g in self.obj.guardrails if g.suite == "wsr_regression"), None)

    def _measure(self, n: int, run_after, *, suite=None, mirror=None, stop_early: bool = True) -> dict[str, Any]:
        """The body shared by practice_eval, settings_eval and steer_eval. `run_after(items)`: the candidate's rows on
        these practice items; `suite()`: the judge's own regression-suite checks for the candidate (rails);
        `mirror()`: harmful_kept, or None.

        stop_early runs the cheap checks first and the practice items in two stages (n >= 2 x STAGE), and stops as
        soon as the candidate is not worth the rest:
        - the regression suite fails: the judge runs this same deterministic check, so it would reject it;
        - fewer of wsr's harmful requests are refused (net): the judge's harm check allows no loss;
        - on the first STAGE items the candidate is worse at the exact sign test's STOP_P level, or newly breaks
          (garbage, a loop, an empty reply) STOP_BROKEN or more replies the original answered, more than it mends.
        Each stage is its own prompt list against its own baseline, so before and after still run in the same
        batches. A call stopped before any practice item does not count against the budget (it saw none)."""
        out: dict[str, Any] = {"n": n}
        stop: Optional[dict] = None
        if suite is not None:
            rails = suite()
            if rails:
                holds = all(r.get("holds") is not False for r in rails)
                out["regression_suite"] = {"holds": holds, "checks": rails,
                                           "what": "the judge's own check on wsr's regression suite, run the same way"}
                if stop_early and not holds:
                    stop = {"why": "the regression suite fails: the judge runs this same check, so it would reject "
                                   "this candidate", "practice_items_run": 0}
        if stop is None and stop_early and mirror is not None:
            mk = mirror()
            if mk is not None:
                out["harmful_kept"] = mk
                if mk["refused_after"] < mk["refused_before"]:
                    stop = {"why": f"fewer of wsr's harmful requests refused ({mk['refused_before']} -> "
                                   f"{mk['refused_after']} of {mk['of']}): the judge's harm check allows no loss",
                            "practice_items_run": 0}
        before: list[dict] = []
        after: list[dict] = []
        if stop is None:
            segs = [(0, STAGE), (STAGE, n)] if stop_early and n >= 2 * STAGE else [(0, n)]
            for k, (s, e) in enumerate(segs):
                before += self._base_seg(s, e)
                if self.plans:
                    # the chunks the reused baseline ran in (the plan cache may have been cleared since)
                    self.lm.batch_plans.update(self.plans)
                after += run_after(self.items[s:e])
                if k < len(segs) - 1:
                    why = stage_problem(before, after)
                    if why:
                        stop = {"why": why, "practice_items_run": e}
                        break
            if mirror is not None and "harmful_kept" not in out:
                mk = mirror()
                if mk is not None:
                    out["harmful_kept"] = mk
        if before:
            a, graded = EV._metric(before)
            b, _ = EV._metric(after)
            out.update({"metric": "accuracy" if graded else "share with no failure", **EV._paired(a, b),
                        "rates_before": EV._rates(before), "rates_after": EV._rates(after)})
            self.evals += 1
        if stop is not None:
            out["stopped_early"] = {**stop, "of": n,
                                    "note": "the rest was not generated; stop_early: false measures every item"}
        out["practice_evals_left"] = self.max_evals - self.evals
        return out

    def _patch_suite(self, wb, edit_id: str, patch) -> Optional[dict]:
        """The judge's suite check for an edit (EV._guard_suite): the verification's run is reused when there is one,
        and this run is kept for the judge in turn (the same deterministic computation)."""
        sg = self._suite_guard()
        if sg is None:
            return None
        known = (((getattr(wb, "results", None) or {}).get(edit_id) or {}).get("regression")
                 or self.suite_known.get(edit_id))
        rail = EV._guard_suite(self.lm, self.obj, sg, patch, known=known)
        if rail.get("holds") is not None:
            self.suite_known[edit_id] = rail
        return rail

    def _practice_eval(self, wb, edit_id: str, n: Optional[int] = None, stop_early: bool = True) -> dict[str, Any]:
        from ..surgery.edits import applied

        if self.evals >= self.max_evals:
            return {"error": f"practice_eval budget used ({self.max_evals}); choose with final_choice"}
        n = max(1, min(int(n or self.default_n), len(self.items)))
        key, patch = _wb_candidate(wb, edit_id)          # with a committed stack: the stack plus the edit, one patch
        r = PF.Runner(TorchBackend(self.lm), self.obj.protocol, self.obj.grader)

        def run_after(items: list[dict]) -> list[dict]:
            with applied(self.lm, patch):
                return EV._run_set(r, items, samples=1, seed=0)
        def suite() -> list[dict]:
            rail = self._patch_suite(wb, key, patch)
            return [rail] if rail is not None else []
        with _wb_original(wb):                           # against the original weights, as the judge measures
            m = self._measure(n, run_after, suite=suite, mirror=lambda: self._mirror(lambda: applied(self.lm, patch)),
                              stop_early=stop_early)
        res = {"edit_id": key, **({"measured": f"{key}: the committed stack plus {edit_id} as one patch, against the "
                                               "original model"} if key != edit_id else {}), **m}
        self.generations += r.generations
        self.scores[key] = res
        return res

    def _settings_eval(self, wb, settings: dict, n: Optional[int] = None,
                       steering_id: Optional[str] = None, stop_early: bool = True) -> dict[str, Any]:
        if self.evals >= self.max_evals:
            return {"error": f"practice_eval budget used ({self.max_evals}); choose with final_choice"}
        why = settings_problem(settings)
        if why:
            return {"error": why}
        bid, base = _wb_stack(wb)                        # the committed stack the settings sit on, if any
        st = None
        if steering_id is not None:
            if steering_id not in self.steerings:
                return {"error": f"unknown steering_id {steering_id!r}; run steer_eval first"}
            st = self.steerings[steering_id]
            if st.get("on_stack") != bid:
                return {"error": f"steering {steering_id} was measured on {st.get('on_stack') or 'the original weights'}, "
                                 f"but the committed stack is now {bid or 'empty'}: measure the steering again "
                                 "(steer_eval) and bundle that one"}
        kw = steer_kw(st) if st is not None else None
        ctx = (lambda: _under(self.lm, base, kw)) if (kw is not None or base is not None) else None
        n = max(1, min(int(n or self.default_n), len(self.items)))
        proto = with_settings(self.obj.protocol, settings)
        r = PF.Runner(TorchBackend(self.lm), proto, self.obj.grader)

        def run_after(items: list[dict]) -> list[dict]:
            with (ctx() if ctx else contextlib.nullcontext()):
                return EV._run_set(r, items_for(settings, items), samples=1, seed=0)
        with _wb_original(wb):
            m = self._measure(n, run_after, suite=lambda: settings_suite_rails(self.lm, self.obj, settings, st,
                                                                               base=base),
                              mirror=lambda: self._mirror(ctx, proto=proto,
                                                          transform=lambda its: items_for(settings, its)),
                              stop_early=stop_early)
        self.generations += r.generations
        sid = f"s{len(self.settings) + 1}"
        res = {"settings_id": sid, "settings": dict(settings), **({"steering_id": steering_id} if st is not None else {}),
               **({"on_stack": bid, "measured": f"these settings on top of the committed stack {bid}, against the "
                                                "original model"} if bid else {}), **m}
        self.settings[sid] = {**res, "_base": base}
        return res

    def _steer_eval(self, wb, vector: str, alpha: Optional[float] = None, layers: Optional[list] = None,
                    n: Optional[int] = None, why: str = "", mode: str = "add", stop_early: bool = True) -> dict[str, Any]:
        if self.evals >= self.max_evals:
            return {"error": f"practice_eval budget used ({self.max_evals}); choose with final_choice"}
        try:
            kw = steering_kwargs(wb, vector, alpha, layers, mode)
        except ValueError as e:
            return {"error": str(e)}
        bid, base = _wb_stack(wb)                        # the committed stack the steering sits on, if any
        if mode == "gated":
            on = tuple((getattr(wb, "gate_built_on", None) or {}).get(str(vector), ()))
            if on != tuple(getattr(wb, "stack", None) or ()):
                return {"error": f"gate {vector} was calibrated on {'+'.join(on) or 'the original weights'}, but the "
                                 f"committed stack is now {bid or 'empty'}: the gate reads activations of the model it "
                                 "was calibrated on. Call harm_probe again on this model"}
        n = max(1, min(int(n or self.default_n), len(self.items)))
        r = PF.Runner(TorchBackend(self.lm), self.obj.protocol, self.obj.grader)

        def run_after(items: list[dict]) -> list[dict]:
            with _under(self.lm, base, kw):
                return EV._run_set(r, items, samples=1, seed=0)
        sg = self._suite_guard()
        suite = (lambda: [steering_suite_rail(self.lm, self.obj, {}, sg, kw=kw, base=base)]) if sg is not None \
            else None
        with _wb_original(wb):
            m = self._measure(n, run_after, suite=suite, mirror=lambda: self._mirror(lambda: _under(self.lm, base, kw)),
                              stop_early=stop_early)
        if bid:
            m = {"on_stack": bid, "measured": f"this steering on top of the committed stack {bid}, against the "
                                              "original model", **m}
        self.generations += r.generations
        sid = f"t{len(self.steerings) + 1}"
        g = kw.get("gated")
        if g is not None:
            alpha = g.up if g.up is not None else g.down
            layers = g.layers
        res = {"steering_id": sid, "mode": mode, "vector": str(vector), "alpha": float(alpha),
               "layers": [int(i) for i in layers], **m}
        vec = g.r if g is not None else kw[mode][0]
        self.steerings[sid] = {**res, "_vector": vec.detach().float().cpu().clone(), "why": why, "_base": base,
                               **({"_gate": g, "gate": g.as_dict(), "vector": g.label} if g is not None else {})}
        return res

    def _checkpoint(self, wb, note: str, edit_id: Optional[str] = None, steering_id: Optional[str] = None,
                    settings_id: Optional[str] = None) -> dict[str, Any]:
        """Any measured candidate on one leaderboard: an edit, a steering or serving settings."""
        if steering_id is not None:
            if steering_id not in self.steerings:
                return {"error": f"unknown steering_id {steering_id!r}; run steer_eval first"}
            cid, label, sc = steering_id, self.steerings[steering_id].get("vector"), self.steerings[steering_id]
        elif settings_id is not None:
            if settings_id not in self.settings:
                return {"error": f"unknown settings_id {settings_id!r}; run settings_eval first"}
            cid, label, sc = settings_id, json.dumps(self.settings[settings_id]["settings"]), self.settings[settings_id]
        elif edit_id is not None:
            key, patch = _wb_candidate(wb, edit_id)
            cid, label, sc = key, patch.label, self.scores.get(key)
        else:
            return {"error": "give edit_id, steering_id or settings_id"}
        self.board = [b for b in self.board if b["edit_id"] != cid]
        self.board.append({"edit_id": cid, "label": label, "note": note,
                           "practice": ({k: sc.get(k) for k in ("n", "before", "after", "diff", "se")}
                                        if sc and "diff" in sc else None),
                           **({"harmful_kept": sc["harmful_kept"]} if sc and sc.get("harmful_kept") else {})})
        self.board.sort(key=lambda b: -(b["practice"] or {}).get("diff", -9))
        return {"leaderboard": self.board}

    def _final_choice(self, wb, rationale: str, edit_id: Optional[str] = None,
                      settings_id: Optional[str] = None, steering_id: Optional[str] = None) -> dict[str, Any]:
        if steering_id is not None:
            if steering_id not in self.steerings:
                return {"error": f"unknown steering_id {steering_id!r}; run steer_eval first"}
            kind, ref, key, item = "steering", steering_id, "final_steering", self.steerings[steering_id]
        elif settings_id is not None:
            if settings_id not in self.settings:
                return {"error": f"unknown settings_id {settings_id!r}; run settings_eval first"}
            kind, ref, key, item = "settings", settings_id, "final_settings", self.settings[settings_id]
        elif edit_id is None:
            return {"error": "give edit_id (an edit), steering_id (a steering) or settings_id (serving settings)"}
        else:
            cand, item = _wb_candidate(wb, edit_id)        # with a committed stack: the stack plus the edit
            if cand not in self.scores:
                return {"error": "run practice_eval on this edit first"
                                 + (f" (with the committed stack, the candidate is {cand})" if cand != edit_id else "")}
            kind, ref, key = "edit", cand, "final"
        last = item if kind != "edit" else self.scores.get(ref) or {}
        if (last.get("regression_suite") or {}).get("holds") is False:
            return {"error": "this candidate fails the regression suite (regression_suite in its measurement): the "
                             "judge runs the same check and would reject it, so no attempt is spent on it"}
        done = {key: ref, "rationale": rationale}
        if self.judge is None:                       # the caller judges the choice
            wb.done = done
            return {"accepted": True}
        fp = AT.fingerprint(kind, item)
        if kind == "settings" and item.get("steering_id"):          # a bundle: the settings and the steering
            fp = AT.fingerprint("bundle", {"parts": [fp, AT.fingerprint("steering",
                                                                        self.steerings[item["steering_id"]])]})
        if kind != "edit" and item.get("_base") is not None:          # on a committed stack: judged with it
            fp = AT.fingerprint("bundle", {"parts": [fp, AT.fingerprint("edit", item["_base"])]})
        if fp in self.judged:
            return {"error": "this candidate (or an identical one) was already judged; choose a different one"}
        with _wb_original(wb):                       # the judge compares with the original weights
            res = self.judge(kind, ref, wb)
        self.judged[fp] = res
        self.results.append(res)
        n = len(self.attempts) + 1
        fb = AT.feedback(res, self.obj, n, self.attempts_max)
        self.attempts.append({"attempt": n, "kind": kind, "id": ref, "verdict": res.get("verdict"),
                              "feedback": fb, "rationale": rationale[:500]})
        if res.get("verdict") == "improved":
            self.accepted = (kind, ref, res)
            wb.done = done
            return {"accepted": True, "note": "accepted on validation; the confirmation set (if the goal has one) "
                                              "decides once, after this run ends"}
        left = self.attempts_max - n
        if left <= 0:
            wb.done = {"rejected": True, "rationale": rationale}
            return {"accepted": False, "feedback": fb, "attempts_left": 0, "note": "no attempts left: the run ends"}
        self.max_evals += 2                          # room to measure a different candidate
        return {"accepted": False, "feedback": fb, "attempts_left": left,
                "practice_evals_left": self.max_evals - self.evals,
                "note": "rejected on validation. Use the symptoms to choose a DIFFERENT candidate (an identical one is "
                        "refused), measure it on practice, check what the failed guardrail protects, then final_choice "
                        "again; each attempt raises the bar for the next"}


def steering_kwargs(wb, vector: Any, alpha: Any, layers: Any, mode: str = "add") -> dict[str, Any]:
    """lm.steer arguments, checked: "add" alpha x vector after these layers, "clamp" the component along the
    vector to the level alpha wherever it is above it, or "gated" a gate from harm_probe (vector = its id)."""
    if mode == "gated":
        g = getattr(wb, "gates", {}).get(str(vector))
        if g is None:
            raise ValueError(f"mode gated needs a gate id from harm_probe as the vector, not {vector!r}")
        return {"gated": g}
    if mode not in ("add", "clamp"):
        raise ValueError("mode must be add, clamp or gated")
    if alpha is None or layers is None:
        raise ValueError("add and clamp need alpha and layers")
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


def steer_kw(st: dict) -> dict[str, Any]:
    """lm.steer arguments for a steering record: {"mode", "_vector", "alpha", "layers"}, or a gated one with "_gate"
    (deep/harmprobe.py)."""
    if st.get("mode") == "gated":
        return {"gated": st["_gate"]}
    return {st.get("mode", "add"): (st["_vector"], st["alpha"], st["layers"])}


def steering_suite_rail(lm, obj: Objective, st: dict, g, name: Optional[str] = None,
                        kw: Optional[dict] = None, base=None) -> dict[str, Any]:
    """The judge's regression-suite check for a steering: the suite compared under its hooks exactly as for a patch
    (new failures, drift, top-token agreement, refusals; the limits of guardrail `g`). Shared by judge_steering,
    judge_settings (a bundle) and the agent's measuring tools, so the practice-side check is the judge's own.
    `base` (or the steering's own "_base"): the weight patch it sits on (a committed stack), under the hooks too;
    `lm` holds the original weights."""
    from ..surgery import regress as R

    tk = obj.protocol.template_kwargs or None
    ref = R.baseline(lm, template_kwargs=tk)
    with _under(lm, base if base is not None else st.get("_base"), kw if kw is not None else steer_kw(st)):
        reg = R.compare(lm, ref, template_kwargs=tk)
    probs = R.Budget(g.max_drift, g.min_agreement, g.allow_safety_loss).problems(reg)
    return {"name": name or g.name, "kind": "suite", "holds": not probs, "problems": probs,
            **{k: reg[k] for k in ("prompts", "unchanged_replies", "mean_delta_nll", "top1_agreement",
                                   "safety_refused_before")},
            "new_failures": reg["new_failures"], "safety_lost": reg["safety_lost"],
            **({"drift_not_measured": reg["drift_not_measured"]} if reg.get("drift_not_measured") else {})}


def settings_suite_rails(lm, obj: Objective, settings: dict, st: Optional[dict] = None,
                         base=None) -> list[dict[str, Any]]:
    """The judge's regression-suite checks for serving settings: the suite's prompts generated under the settings
    (no reply that was fine may fail, every refusal kept). Decoding settings and a system prompt change what the
    model is asked and how it decodes, not the model, so drift is not measured for them; with a steering served
    together (a bundle) its own check (steering_suite_rail) is added. Shared by judge_settings and settings_eval.

    `base`: a weight patch the settings sit on (a committed stack). It does change the model, so everything "after"
    runs on it, and its own suite check (drift, agreement, refusals) is added (with a steering, under its hooks)."""
    from ..surgery import regress as R

    kw = steer_kw(st) if st is not None else None
    proto = obj.protocol
    sg = next((g for g in obj.guardrails if g.suite == "wsr_regression"), None)
    keep_refusals = not (sg is not None and sg.allow_safety_loss)
    suite = [{"id": n, "messages": [{"role": "user", "content": p}], "lang": lang,
              **({"expect": "refuse"} if kind == "safety" and keep_refusals else {})}
             for n, p, lang, kind in R.SUITE if kind != "safety" or keep_refusals]
    sp0 = dataclasses.replace(proto, system=None, max_new_tokens=min(proto.max_new_tokens, 256))
    sp1 = with_settings(sp0, settings)
    s0 = EV.original_set(PF.Runner(TorchBackend(lm), sp0, "detector:any"), suite, samples=1, seed=0)
    with _under(lm, base, kw):
        s1 = EV._run_set(PF.Runner(TorchBackend(lm), sp1, "detector:any"), items_for(settings, suite), samples=1,
                         seed=0)
    m0, _ = EV._metric(s0)
    m1, _ = EV._metric(s1)
    broke = [x["id"] for x, u, v in zip(s0, m0, m1) if u > v]
    under = " and the steering" if st else ""
    under += " on the committed stack" if base is not None else ""
    rails = [{"name": f"wsr_regression (generated under the settings{under})",
              "kind": "suite", "prompts": len(suite), "new_failures": broke, "holds": not broke}]
    if st is not None and sg is not None:
        rails.append(steering_suite_rail(lm, obj, st, sg, base=base,
                                         name="wsr_regression (the steering's drift and refusals"
                                              + (" on the committed stack)" if base is not None else ")")))
    elif base is not None and sg is not None:
        rails.append(dict(EV._guard_suite(lm, obj, sg, base),
                          name="wsr_regression (the committed stack's drift and refusals)"))
    return rails


def _fresh_under(lm, base, kw, r, items: list[dict]) -> list[dict]:
    with _under(lm, base, kw):
        return EV._run_set(r, items, samples=1, seed=0)


def _data_rail(g, gi: list[dict], items: list[dict], rg0, rg1, raw0: list, raw1: list, *, fresh_before, fresh_after,
               after_items: Optional[list] = None) -> dict:
    """A data guardrail for the steering and settings judges. On the very same prompts as the main set
    (EV.same_prompts: never a safety set) it grades the main set's replies with its own grader, before (when they were
    generated here) and after, instead of generating them again; otherwise it generates its own."""
    same = EV.same_prompts(g, gi, items)
    where = []

    def made() -> int:
        return rg0.generations + (rg1.generations if rg1 is not rg0 else 0)
    n0 = made()
    if same and raw0:
        gb = EV._regrade(rg0, gi, raw0, samples=1, seed=0)
        where.append("original")
    else:
        gb = fresh_before()
    if same:
        ga = EV._regrade(rg1, after_items if after_items is not None else gi, raw1, samples=1, seed=0)
        where.append("candidate")
    else:
        ga = fresh_after()
    rail = EV._guard_result(g, gb, ga, made() - n0, gi)
    if where:
        rail["replies"] = EV.reused_note(" and ".join(where))
    return rail


def judge_steering(lm, obj: Objective, st: dict, *, limit: Optional[int] = None, say=lambda s: None,
                   m: int = 1) -> dict:
    """Steering on the held-out set: the same weights without and with the steering hooks, same items and seeds,
    paired. Guardrails: the regression suite compared under the steering exactly as for a patch (new failures,
    drift, top-token agreement, refusals; the goal's limits), and every data guardrail under both."""
    t0 = time.time()
    kw = steer_kw(st)
    base = st.get("_base")                       # a weight patch the steering sits on (a committed stack), or None
    items = load_items(obj.resolve(obj.heldout), system=obj.protocol.system, limit=limit)
    say(f"  held-out: {len(items)} items, without then with the steering"
        + (f" on the weight patch {base.label}" if base is not None else ""))
    r0, r1 = PF.Runner(TorchBackend(lm), obj.protocol, obj.grader), PF.Runner(TorchBackend(lm), obj.protocol, obj.grader)
    raw0: list = []                              # the replies' generations, for same-prompt guardrails
    raw1: list = []
    before = EV.original_set(r0, items, samples=1, seed=0, keep=raw0)
    with _under(lm, base, kw):
        after = EV._run_set(r1, items, samples=1, seed=0, keep=raw1)
    a, graded = EV._metric(before)
    b, _ = EV._metric(after)
    paired = EV._paired(a, b)
    rails = []
    for g in obj.guardrails:
        if g.suite == EV.CANARY:
            from ..surgery import canary as CN
            c0 = CN.run(lm, obj.protocol.template_kwargs, original=True)
            with _under(lm, base, kw):
                c1 = CN.run(lm, obj.protocol.template_kwargs, original=False)
            rails.append({**CN.compare(c0, c1, g.max_lost), "name": g.name})
        elif g.suite:
            rails.append(steering_suite_rail(lm, obj, st, g))
        elif g.data:
            gi = EV._guard_items(obj, g)
            rg = PF.Runner(TorchBackend(lm), obj.protocol, g.grader or "detector:any")
            rails.append(_data_rail(g, gi, items, rg, rg, raw0, raw1,
                                    fresh_before=lambda: EV.original_set(rg, gi, samples=1, seed=0),
                                    fresh_after=lambda: _fresh_under(lm, base, kw, rg, gi)))
    dec = EV.decide(paired, all(x["holds"] is not False for x in rails), m=m, **EV.judge_settings(obj))
    verdict = dec["verdict"]
    looks = EV.HeldoutLog(obj).add({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "patch": f"steering: {st['vector']}"
                                    + (f" on the weight patch: {base.label}" if base is not None else ""),
                                    "items": len(items), "verdict": verdict})
    return {"vector": st["vector"], "mode": st.get("mode", "add"), **_base_note(base), "power": EV.power(a, m=m, **EV.judge_settings(obj)),
            "decision": dec, "verdict_reason": dec["reason"], "alpha": st["alpha"],
            "layers": st["layers"], "n": len(items),
            "metric": "accuracy" if graded else "share with no failure", "paired": paired,
            "rates_before": EV._rates(before), "rates_after": EV._rates(after), "guardrails": rails,
            "verdict": verdict, "heldout_looks": looks, "seconds": round(time.time() - t0, 1)}


def save_steering(st: dict, out: Path, dirname: str = "steering", lm=None, with_base: bool = True) -> Path:
    """The steering as files (surgery/steer.py): the vector, its settings and APPLY.md; with the weight patch it was
    judged on (a committed stack) in patch/, unless `with_base` is False (a bundle writes it once, at its top)."""
    from ..surgery import steer as SV

    d = SV.save(st["_vector"], st["alpha"], st["layers"], st["vector"], out, mode=st.get("mode", "add"),
                dirname=dirname, gate=st.get("_gate"))
    if with_base and st.get("_base") is not None:
        _save_base(st["_base"], Path(d), lm, "this steering")
    return d


def _save_base(base, d: Path, lm, what: str) -> None:
    """The weight patch a runtime fix was judged on top of (multi-stage surgery), in d/patch, and a note in APPLY.md."""
    base.save(d / "patch", lm)
    note = (f"\n\n## The weight patch underneath\n\n{what[0].upper() + what[1:]} was judged on top of a weight patch "
            f"(`patch/`: {base.label}). Apply the patch to the weights first (`wsr patch apply MODEL_DIR patch --out "
            "NEW_DIR`), then serve the patched model with it. Neither part alone was what the judge accepted.\n")
    ap = d / "APPLY.md"
    ap.write_text((ap.read_text(encoding="utf-8") if ap.exists() else "") + note, encoding="utf-8")


def _settings_files(settings: dict, proto: Protocol, d: Path, steering: bool = False, base=None, lm=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "settings.json").write_text(json.dumps(settings, indent=1, sort_keys=True), encoding="utf-8")
    both = ("\n\nThese settings were judged TOGETHER with the steering in `steering/` (see steering/APPLY.md): "
            "serve both, or neither. Either one alone was not what the judge accepted.\n") if steering else "\n"
    (d / "APPLY.md").write_text("# Applying these settings\n\n" + "\n".join(x for x in settings_howto(proto, settings)
                                                                          if x) + both, encoding="utf-8")
    if base is not None:
        _save_base(base, d, lm, "these settings" + (" and the steering" if steering else ""))
    return d


def _accept(lm, obj: Objective, rec: dict, *, kind: str, label: str, by: str, val: dict, stage: Path,
            write_files, run_after, evidence: Optional[dict], say) -> bool:
    """After the validation judge: the confirmation stage (when the goal has a confirmation set), then the fix's
    MANIFEST.json. Returns whether the fix ships.

    * rejected on validation: not shipped;
    * no confirmation set: shipped, as "validated, not confirmed";
    * a confirmation set: shipped only when the frozen candidate, run once on it, passes the same policy (exact
      sign test at alpha, the minimum gain); a set already used, or too small to show anything, does not ship."""
    from .. import evidence as EVD
    from .. import manifest as MF
    from . import confirm as CF

    write_files(stage)
    model, evaluation = MF.model_record(lm), MF.goal_record(obj)
    cand = MF.candidate_hash(model, MF.artifact_hash(stage), evaluation)
    conf = None
    if val.get("verdict") != "improved":
        ship, status, why = False, "rejected on validation", val.get("verdict_reason") or val.get("verdict")
    elif not obj.confirmation:
        ship, status, why = True, "validated, not confirmed", "the goal has no confirmation set"
    else:
        say(f"\nconfirming {label} on the confirmation set (once; the candidate is frozen)")
        conf = CF.run(lm, obj, run_after=run_after, say=say,
                      record={"goal": obj.name, "goal_version": evaluation["goal_version"], "candidate": cand,
                              "model": (model.get("digests") or {}).get("float32"), "kind": kind, "label": label})
        ship = conf["verdict"] == "improved"
        status = ("confirmed" if ship else "inconclusive" if conf["verdict"] == "inconclusive" else "not confirmed")
        why = conf["reason"]
        say(f"  confirmation: {status} ({why})")
    passed = val.get("verdict") == "improved"
    ev = EVD.merge(evidence or EVD.make(intervention=True),
                   EVD.make(generalization=(["validation_set"] if passed else []) + (["confirmation_set"] if ship and conf
                                                                                      else [])))
    fmt = "gguf:" + val["judge"]["quant"] if (val.get("judge") or {}).get("kind") == "llama-server" and \
        (val.get("judge") or {}).get("quant") else "pytorch"
    by_fmt = val.get("guardrails_by_format") or {}
    rails = {f: MF.guardrails({"guardrails": list(rs.values())}, f)[f] for f, rs in by_fmt.items()} if by_fmt \
        else MF.guardrails(val, "pytorch")
    # Deployable in a format only when every guardrail that ran on that format held (the target was judged on
    # `fmt`); a format nothing ran on is not claimed.
    deploy = {}
    for f, rs in (rails or {"pytorch": {}}).items():
        deploy[f] = ship and all(g.get("holds") is not False for g in rs.values())
    if not deploy:
        deploy = {"pytorch": ship}
    judge = val.get("judge") or {}
    MF.write(stage, {"kind": kind, "artifact": {"label": label, "by": by}, "model": model, "evaluation": evaluation,
                     "statistics": {"validation": MF.statistics(val), "confirmation": conf},
                     "judged_on": fmt, "server_version": judge.get("server_version"), "guardrails": rails,
                     "evidence": ev,
                     "decision": {"status": status, "shipped": ship, "why": why, "deployable": deploy}})
    rec["status"], rec["status_why"] = status, why
    if conf is not None:
        rec["stages"]["confirmation"] = conf
    return ship


def _final_dir(stage: Path, final: Path) -> Path:
    import shutil

    if final.resolve() == stage.resolve():
        return final
    if final.exists():
        shutil.rmtree(final)
    stage.replace(final)
    return final


def judge_settings(lm, obj: Objective, settings: dict, *, limit: Optional[int] = None,
                   say=lambda s: None, m: int = 1, st: Optional[dict] = None, base=None) -> dict[str, Any]:
    """Serving settings on the held-out set: the original weights with the objective's protocol vs the same
    weights with these settings, same items and seeds, paired. Guardrails: every data guardrail under both, and
    the regression-suite prompts generated under both (no reply that was fine may fail, every refusal kept).

    `st`: a steering served together with the settings (a runtime bundle). Everything "after" then runs under its
    hooks too, and the regression suite also gets the steering's own check (drift, top-token agreement and
    refusals, teacher-forced under the hooks, as judge_steering does): the hooks change the model's internals,
    which decoding settings alone never do.

    `base`: a weight patch the settings sit on (a committed stack): everything "after" runs on it, and the suite
    checks add its drift and refusals."""
    t0 = time.time()
    kw = steer_kw(st) if st is not None else None

    def hooked():
        return _under(lm, base, kw)
    proto, new = obj.protocol, with_settings(obj.protocol, settings)
    items = load_items(obj.resolve(obj.heldout), system=proto.system, limit=limit)
    say(f"  held-out: {len(items)} items, current settings then {settings}"
        + (f" with the steering {st['vector']}" if st is not None else "")
        + (f" on the weight patch {base.label}" if base is not None else ""))
    r0, r1 = PF.Runner(TorchBackend(lm), proto, obj.grader), PF.Runner(TorchBackend(lm), new, obj.grader)
    raw0: list = []                              # the replies' generations, for same-prompt guardrails
    raw1: list = []
    before = EV.original_set(r0, items, samples=1, seed=0, keep=raw0)
    with hooked():
        after = EV._run_set(r1, items_for(settings, items), samples=1, seed=0, keep=raw1)
    a, graded = EV._metric(before)
    b, _ = EV._metric(after)
    paired = EV._paired(a, b)
    rails = settings_suite_rails(lm, obj, settings, st, base=base)
    cg = next((g for g in obj.guardrails if g.suite == EV.CANARY), None)
    if cg is not None:
        from ..surgery import canary as CN
        c0 = CN.run(lm, proto.template_kwargs, original=True)
        with hooked():
            c1 = EV._run_set(PF.Runner(TorchBackend(lm), with_settings(CN.protocol(proto.template_kwargs), settings),
                                       "item"), items_for(settings, CN.items()), samples=1, seed=0)
        rails.append({**CN.compare(c0, c1, cg.max_lost), "name": cg.name})
    for g in obj.guardrails:
        if g.data:
            gi = EV._guard_items(obj, g)
            rg0 = PF.Runner(TorchBackend(lm), proto, g.grader or "detector:any")
            rg1 = PF.Runner(TorchBackend(lm), new, g.grader or "detector:any")
            rails.append(_data_rail(g, gi, items, rg0, rg1, raw0, raw1,
                                    fresh_before=lambda: EV.original_set(rg0, gi, samples=1, seed=0),
                                    fresh_after=lambda: _fresh_under(lm, base, kw, rg1, items_for(settings, gi)),
                                    after_items=items_for(settings, gi)))
    dec = EV.decide(paired, all(x["holds"] is not False for x in rails), m=m, **EV.judge_settings(obj))
    verdict = dec["verdict"]
    looks = EV.HeldoutLog(obj).add({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "patch": f"settings: {json.dumps(settings)}"
                                             + (f" + steering: {st['vector']}" if st is not None else "")
                                             + (f" on the weight patch: {base.label}" if base is not None else ""),
                                    "items": len(items), "verdict": verdict})
    return {"settings": dict(settings), "n": len(items), "power": EV.power(a, m=m, **EV.judge_settings(obj)),
            **_base_note(base),
            **({"steering": {k: st.get(k) for k in ("vector", "mode", "alpha", "layers")}} if st is not None else {}),
            "decision": dec, "verdict_reason": dec["reason"],
            "metric": "accuracy" if graded else "share with no failure",
            "paired": paired, "rates_before": EV._rates(before), "rates_after": EV._rates(after),
            "guardrails": rails, "verdict": verdict, "heldout_looks": looks, "seconds": round(time.time() - t0, 1)}


def render_settings(js: dict, proto: Protocol) -> list[str]:
    p = js["paired"]
    stg = js.get("steering")
    with_st = (f" together with the steering {stg['vector']} ({stg.get('mode', 'add')}, layers {stg.get('layers')})"
               if stg else "")
    lines = [f"Settings `{json.dumps(js['settings'])}`{with_st} on {js['n']} held-out items ({js['metric']}): "
             f"{p.get('before', 0):.1%} -> {p.get('after', 0):.1%} ({p.get('diff', 0):+.1%} +- {p.get('se', 0):.1%} SE)"
             f"; verdict: **{js['verdict']}** ({js.get('verdict_reason', '')})",
             "- failure groups before: " + ", ".join(f"{k} {v:.0%}" for k, v in js["rates_before"].items()),
             "- failure groups after: " + ", ".join(f"{k} {v:.0%}" for k, v in js["rates_after"].items())]
    for g in js["guardrails"]:
        extra = f", new failures: {', '.join(g['new_failures']) or 'none'}" if "new_failures" in g else ""
        lines.append(f"- guardrail {g['name']}: {'holds' if g['holds'] else 'FAILS'}{extra}{EV.guard_scope_note(g)}")
    if js.get("power") and not js["power"]["can_show_improvement"]:
        lines.append(f"- note: {js['power']['note']}")
    if js.get("heldout_looks"):
        lines.append(f"- held-out set looked at {js['heldout_looks']} time(s) so far")
    if js.get("base_patch"):
        lines.append(f"- on top of the weight patch {js['base_patch']['label']} (a committed stack): judged together")
    if js["verdict"] == "improved":
        lines += ["", "To apply:", *[x for x in settings_howto(proto, js["settings"]) if x]]
        if js.get("base_patch"):
            lines.append("- AND the weight patch in `patch/` applied to the weights first (`wsr patch apply`): the "
                         "settings were judged on the patched model")
        if stg:
            lines.append("- AND the steering in `steering/` (its APPLY.md has the hook): the two were judged "
                         "together and are recommended only together")
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


def _validation_power(lm, obj: Objective, m: int, limit: Optional[int]) -> dict[str, Any]:
    """EV.power on the validation set at alpha / m, from the original model's replies on PyTorch: the same items,
    seeds and cache entry as judge_steering's and the PyTorch patch judge's original side."""
    items = load_items(obj.resolve(obj.heldout), system=obj.protocol.system, limit=limit)
    r0 = PF.Runner(TorchBackend(lm), obj.protocol, obj.grader)
    a, _ = EV._metric(EV.original_set(r0, items, samples=1, seed=0))
    return {**EV.power(a, m=m, **EV.judge_settings(obj)), "m": m, "measured_on": "pytorch (original model)"}


def _user_text(it: dict) -> str:
    return next((str(m.get("content")) for m in reversed(it["messages"]) if m.get("role") == "user"), "")


def run_program(lm, obj: Objective, *, out: Path, profile_source=None, provider=None, playbook: bool = True,
                profile_limit: int = 40, agent_budget=None, agent_limits=None, practice_evals: int = 10,
                eval_limit: Optional[int] = None, log=print, decisions: Optional[list] = None,
                allow_safety_loss: bool = False, steering: str = "fallback", weight_edits: bool = True,
                mode: str = "deep", judge: str = "torch", judge_quant: str = "Q8_0", consent=None,
                model_dir: Optional[Path] = None, server_bin: Optional[str] = None,
                agent_attempts: int = AT.ATTEMPTS) -> dict[str, Any]:
    from ..deep.run import diagnose
    from ..surgery import playbook as P
    from ..surgery import regress as R

    say = log or (lambda s: None)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    if not allow_safety_loss:
        obj = AT.with_harm_check(obj)      # held-back harmful requests the agent never sees: every judge checks them
    proto = obj.protocol
    practice = load_items(obj.resolve(obj.practice), system=proto.system) if obj.practice else []
    rec: dict[str, Any] = {"objective": obj.to_dict(), "stages": {}}
    # wsr's held-back harmful requests are seen by its own harm check only (the judge's half, the agent's mirror):
    # a copy in the goal's practice set would let the run see, or tune to, what the judge checks. Left out here,
    # whatever the goal file says (wsr objective check asks for them to be removed).
    from ..deep import refusal_pools as RP
    held = [it["id"] for it in practice if RP.is_held_back(it)]
    if held:
        practice = [it for it in practice if not RP.is_held_back(it)]
        say(f"practice: {len(held)} prompt(s) are wsr's held-back harmful requests (its own harm check's); left out "
            "of this run (remove them from the practice file)")
        rec["stages"]["held_back_left_out"] = {"practice": held}
    if not practice:
        raise ValueError("wsr program needs a practice set in the goal file")

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
    # Work on what the goal grades. A loop goal (detector:loop) on list requests also profiles the model's
    # refusals; they can outnumber the loops, but the grader counts them as passing, so they are not the target.
    graded_fail = {}
    for rp in prof.replies:
        if rp.get("correct") is False:
            graded_fail[rp["label"]] = graded_fail.get(rp["label"], 0) + 1
    if any(rp.get("correct") is not None for rp in prof.replies):
        off = [r["label"] for r in weight if not graded_fail.get(r["label"])]
        if off:
            say(f"not working on {', '.join(off)}: the goal's grader ({obj.grader}) counts those replies as passing")
            rec["stages"]["not_graded_failures"] = off
        weight = sorted((r for r in weight if graded_fail.get(r["label"])), key=lambda r: -graded_fail[r["label"]])
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
    # The language the replies must be in (an item's "lang", as the profile grades it). Without it the diagnosis and
    # the playbook's checks cannot see a wrong-language reply at all. Only items in the first failing item's language
    # are used, so one expected language is right for every prompt the diagnosis runs.
    lang = fail[0].get("lang") or None
    fail_used = fail[:1] + [it for it in fail[1:] if (it.get("lang") or None) == lang][:3]
    work_used = [it for it in work if (it.get("lang") or None) == lang][:3]
    rec["stages"]["group"] = {"label": label, "count": group["count"], "says": group["says"], "kind": kind,
                              "failing_used": [it["id"] for it in fail_used],
                              "working_used": [it["id"] for it in work_used], "expect_lang": lang}
    say(f"\nworking on {label} ({group['count']} of {prof.n} practice replies): {group['says']}")

    # The harm probe (the diagnosis's reading, the playbook's gated steering and the agent's harm_probe: one
    # probe everywhere) is calibrated on the goal's practice and contrast prompts too (the user's own kind of request,
    # labelled by `expect`), for any goal, so a gate reads harm the way this goal means it. Only with both kinds: one-
    # sided data teaches the probe the goal's style instead of harm (DESIGN.md section 22). Practice data only:
    # validation stays unseen. A goal with prompts of one kind (a repair goal's answerable requests) can add a
    # `contrast` file with the other.
    from .objective import harm_extra_problem
    harm_extra = None
    pool = [it for it in practice if it.get("messages")]
    if obj.contrast:
        con = load_items(obj.resolve(obj.contrast), system=proto.system)
        held_c = [it["id"] for it in con if RP.is_held_back(it)]
        if held_c:
            say(f"contrast: {len(held_c)} prompt(s) are wsr's held-back harmful requests; left out of the harm probe's "
                "data (remove them from the contrast file)")
            rec["stages"].setdefault("held_back_left_out", {})["contrast"] = held_c
        pool += [it for it in con if not RP.is_held_back(it)]
    why_not = harm_extra_problem(pool)
    if not why_not:
        harm_extra = pool
        rec["stages"]["harm_probe_data"] = {"used": "built-in pools + the goal's practice and contrast prompts",
                                            "items": len(pool)}
    elif kind in ("refusal", "complied"):
        say(f"harm probe: {why_not}")
        rec["stages"]["harm_probe_data"] = {"used": "built-in pools only", "why": why_not}
    # 2. diagnose, 3. playbook
    tk = proto.template_kwargs or None
    res = diagnose(lm, fail[0]["messages"], kind=kind, failing=tuple(_user_text(i) for i in fail_used[1:]),
                   passing=tuple(_user_text(i) for i in work_used), expect_lang=lang, max_new=max_new,
                   template_kwargs=tk, out_dir=out / "deep", harm_extra=harm_extra)
    cut = next((f for f in res.report.findings if f.code == "DEEP_CUT_SHORT"), None)
    if cut is not None and int(cut.evidence.get("longer", 0)) > max_new:
        # The diagnosis saw the failure only past its token limit: once more at the length it names.
        max_new = int(cut.evidence["longer"])
        say(f"diagnosis: the failure starts after the limit; diagnosing again with {max_new} new tokens")
        res = diagnose(lm, fail[0]["messages"], kind=kind, failing=tuple(_user_text(i) for i in fail_used[1:]),
                       passing=tuple(_user_text(i) for i in work_used), expect_lang=lang, max_new=max_new,
                       template_kwargs=tk, out_dir=out / "deep", harm_extra=harm_extra)
    rec["stages"]["diagnosis"] = {"summary": res.summary, "confidence": res.confidence, "max_new": max_new}
    say(f"diagnosis: {res.summary}")
    # The verification uses the limits of the goal's own suite guardrail (the judge applies the same ones), so a
    # candidate is not verified against one set of limits and judged against another.
    sg = next((g for g in obj.guardrails if g.suite == "wsr_regression"), None)
    cg = next((g for g in obj.guardrails if g.suite == EV.CANARY), None)
    reg_budget = (R.Budget(sg.max_drift, sg.min_agreement, sg.allow_safety_loss or allow_safety_loss)
                  if sg is not None else R.Budget(allow_safety_loss=allow_safety_loss))
    reg_budget.canary_max_lost = cg.max_lost if cg is not None else None
    if not weight_edits:
        # The weights must not change: the playbook may only steer, and the agent's edit tools are refused.
        steering = "only"
        if agent_budget is None and provider is not None:
            from ..agent import Budget as AgentBudget
            agent_budget = AgentBudget()
        if agent_budget is not None:
            agent_budget.weight_edits = False
        rec["weight_edits"] = False
    rep = P.repair(lm, res, expect_lang=lang, max_new=max_new, template_kwargs=tk, mode=mode, playbook=playbook,
                   log=say, budget=reg_budget, steering=steering, harm_extra=harm_extra)
    rec["stages"]["playbook"] = {"status": rep.status, "summary": rep.summary,
                                 "patch": rep.patch.label if rep.patch is not None else None,
                                 "reference_runs": (rep.checks or {}).get("reference_runs")}
    candidate, by = (rep.patch, "playbook") if rep.status == "fixed" else (None, None)
    # The verification already ran the regression suite on this patch with this template; the judge's
    # guardrail reuses it (same computation, same numbers) instead of running it again.
    suite_result = rep.regression if rep.status == "fixed" else None
    # Refusal goals: when the playbook's candidate is a weight patch, its steering (probe-gated first: the refusal
    # tied to the model's own harm reading) never ran, because steering is the fallback when no weight edit passes.
    # If the validation set rejects that patch, the playbook gets one steering pass. Its slot is declared here, before
    # anything is judged, so it is part of the correction whether it is used or not.
    steer_pass = kind in ("refusal", "complied") and steering != "off" and rep.status == "fixed"
    # The held-out (validation) set may judge one candidate from the playbook, for a refusal goal one steering from
    # the playbook, and with an agent up to `agent_attempts` (each judged at once, a rejection answered with abstract
    # symptoms only: attempts.py). Every one is tested at alpha / slots, declared here before anything is judged, so
    # several looks at the same set cannot multiply the chance of a false acceptance.
    agent_attempts = max(1, int(agent_attempts))
    slots = EV.candidate_slots(playbook=playbook, steer_pass=steer_pass,
                               agent_attempts=agent_attempts if provider is not None else None)
    rec["stages"]["candidate_slots"] = slots

    # 4. agent, 5. the judge. A playbook candidate that the held-out judge rejects hands over to the agent once
    # (when there is one): it hears only that the candidate was rejected, not the held-out numbers.
    from ..consent import Consent
    from . import judge as JG

    def judge_patch(candidate, by, suite_result) -> dict:
        from .. import identity as ID

        if not candidate.identity:
            candidate.identity = ID.for_patch(lm, candidate)   # the unedited weights: saved with the patch
        say(f"\njudging {candidate.label} ({by}) on the held-out set")
        jd = JG.choose(judge, lm=lm, model_dir=model_dir, patch=candidate, out=out, consent=consent or Consent(),
                       quant=judge_quant, server_bin=server_bin, max_new=proto.max_new_tokens, log=say)
        try:
            ev = None
            if jd.kind == "llama-server":
                try:
                    ev = EV.evaluate(jd.source, obj, patched_source=jd.patched, guard_source=lm, patch=candidate,
                                     patch_name=candidate.label, split="heldout", limit=eval_limit,
                                     suite_result=suite_result, log=False, m=slots, fmt=f"gguf:{judge_quant}")
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
                                 limit=eval_limit, suite_result=suite_result, m=slots)
        finally:
            jd.close()
        ev["judge"] = {"kind": jd.kind, "notes": jd.notes, "files": jd.files,
                       **({"quant": judge_quant, "server_version": jd.server_version} if jd.kind == "llama-server"
                          else {})}
        say(EV.render(ev))
        return ev

    tools = None
    rejected = None                                       # the playbook's candidate, when the judge said no
    if candidate is not None and (provider is not None or steer_pass) and rep.ctx is not None:
        ev = judge_patch(candidate, by, suite_result)
        if ev["verdict"] == "improved":
            return _ship(rec, out, t0, candidate, by, ev, obj, consent, decisions, lm=lm, say=say)
        candidate.checks["objective"] = {"name": obj.name, "verdict": ev["verdict"], "paired": ev.get("paired"),
                                         "by": by}
        candidate.save(out / "candidate_playbook")
        rec["stages"]["judge_playbook"] = ev
        rejected = candidate.label
        candidate, by, suite_result = None, None, None
        if steer_pass:
            say("\nthe held-out judge did not accept the playbook's weight patch; refusal goal: the playbook's "
                "steering pass (probe-gated first), in the slot declared for it")
            rep = P.repair(lm, res, expect_lang=lang, max_new=max_new, template_kwargs=tk, mode=mode,
                           playbook=playbook, log=say, budget=reg_budget, steering="only", harm_extra=harm_extra)
            rec["stages"]["playbook_steering"] = {"status": rep.status, "summary": rep.summary,
                                                  "steering": rep.steering.as_dict() if rep.steering else None}
    if rep.status == "steered" and rep.steering is not None:
        # The playbook's steering: judged like the agent's (same weights with and without it, the suite under it).
        sv = rep.steering
        st = {"_vector": sv.vector, "alpha": sv.alpha, "layers": sv.layers, "vector": sv.label, "mode": sv.mode,
              "_gate": sv.gate}
        say(f"\njudging the playbook's steering ({sv.label}, alpha {sv.alpha:.3f} after layers {sv.layers}) on the "
            "held-out set")
        js = judge_steering(lm, obj, st, limit=eval_limit, say=say, m=slots)
        js["by"] = "playbook"
        js["practice"] = None
        say(f"  steering: {js['verdict']}")
        if js["verdict"] == "improved":
            ok = _accept(lm, obj, rec, kind="steering", label=sv.label, by="playbook", val=js,
                         stage=out / "candidate_steering", write_files=lambda d: save_steering(st, d.parent, d.name),
                         run_after=_steer_runner(lm, obj, st), evidence=(rep.checks or {}).get("evidence"), say=say)
            js["files"] = str(_final_dir(out / "candidate_steering", out / ("steering" if ok else "candidate_steering")))
            rec["stages"]["steering"] = js
            rec["outcome"] = "steering recommended" if ok else "not shipped"
            rec["says"] = (f"serve with the playbook's steering ({sv.label}): {rec['status']}" if ok else
                           f"the playbook's steering ({sv.label}) passed validation, then {rec['status']}: "
                           f"{rec['status_why']}")
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

        # Before the agent spends its budget: can the validation set show an improvement at all, at the level every
        # attempt is judged at? It depends only on the original model's replies (no candidate is looked at), the
        # ones the PyTorch judges compare with (kept across runs, so a PyTorch judge does not generate them again).
        vp = _validation_power(lm, obj, slots, eval_limit)
        rec["stages"]["validation_power"] = vp
        if not vp["can_show_improvement"]:
            say(f"\nthe agent is not started: {vp['note']} (each candidate is judged at alpha / {slots}, the run's "
                f"{slots} candidate slots)")
            rec["outcome"] = "no candidate"
            rec["says"] = ((f"the playbook's candidate ({rejected}) was not accepted, and " if rejected else "")
                           + f"the agent was not started: on the full-precision model, {vp['note']} (alpha / {slots}: "
                           f"{slots} candidate slots). Add items that fail to the validation set, or allow fewer "
                           "attempts (--agent-attempts)")
            return _finish(rec, out, t0)

        def judge_attempt(kind: str, ref: str, wb) -> dict:
            n = len(tools.attempts) + 1
            by_ = f"agent (attempt {n} of {agent_attempts})"
            if kind == "edit":
                return judge_patch(wb.edits[ref], by_, (wb.results.get(ref) or {}).get("regression")
                                   or tools.suite_known.get(ref))
            if kind == "steering":
                st_ = tools.steerings[ref]
                say(f"\njudging the steering {st_['vector']} ({by_}) on the held-out set")
                js_ = judge_steering(lm, obj, st_, limit=eval_limit, say=say, m=slots)
                say(f"  steering: {js_['verdict']}")
                return js_
            sel_ = tools.settings[ref]
            st_ = tools.steerings.get(sel_.get("steering_id") or "")
            say(f"\njudging the settings {sel_['settings']}" + (f" with the steering {st_['vector']}" if st_ else "")
                + f" ({by_}) on the held-out set")
            js_ = judge_settings(lm, obj, sel_["settings"], limit=eval_limit, say=say, m=slots, st=st_,
                                 base=sel_.get("_base"))
            say("\n".join(render_settings(js_, proto)))
            return js_

        tools = ObjectiveTools(lm, obj, practice, max_evals=practice_evals,
                               baseline=prof.replies if on_lm else None, plans=profile_plans,
                               attempts=agent_attempts, judge=judge_attempt)
        hpd = rec["stages"].get("harm_probe_data") or {}
        if hpd.get("why"):
            tools.note = "Harm probe: " + hpd["why"]
        if rejected:
            tools.note = ((tools.note + " ") if tools.note else "") + (
                f"The rule-based playbook's candidate ({rejected}) passed its own checks but the held-out "
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
                                  "practice_evals": tools.scores, "attempts": tools.attempts,
                                  "attempts_max": agent_attempts}
        if wb is not None and wb.done and "final" in wb.done:
            candidate, by = wb.edits[wb.done["final"]], f"agent (final choice: {wb.done['rationale'][:200]})"
            suite_result = (wb.results.get(wb.done["final"]) or {}).get("regression")
        elif rep2.status == "fixed" and rep2.patch is not None and len(tools.attempts) < agent_attempts:
            candidate, by = rep2.patch, "agent (submitted after verify_edit)"
            suite_result = rep2.regression
    if tools is not None and tools.wb is not None and tools.wb.done and "final_steering" in tools.wb.done \
            and candidate is None:
        st = tools.steerings[tools.wb.done["final_steering"]]
        by = f"agent (final choice: {tools.wb.done['rationale'][:200]})"
        if tools.accepted is not None and tools.accepted[1] == tools.wb.done["final_steering"]:
            js = tools.accepted[2]                       # judged when it was chosen: not looked at again
        else:
            say(f"\njudging the steering {st['vector']} x {st['alpha']} after layers {st['layers']} ({by}) on the "
                "held-out set")
            js = judge_steering(lm, obj, st, limit=eval_limit, say=say, m=slots)
        js["by"] = by
        js["practice"] = {k: st.get(k) for k in ("n", "before", "after", "diff", "se")}
        rec["stages"]["steering"] = js
        rec["stages"]["agent"]["steer_evals"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                                 for k, v in tools.steerings.items()}
        say(f"  steering: {js['verdict']}")
        sbase = st.get("_base")
        ok = _accept(lm, obj, rec, kind="patch+steering" if sbase is not None else "steering",
                     label=st["vector"] + (f" on the weight patch {sbase.label}" if sbase is not None else ""), by=by,
                     val=js, stage=out / "candidate_steering",
                     write_files=lambda d: save_steering(st, d.parent, d.name, lm=lm),
                     run_after=_steer_runner(lm, obj, st), evidence=None, say=say)
        js["files"] = str(_final_dir(out / "candidate_steering", out / ("steering" if ok else "candidate_steering")))
        rec["outcome"] = ("steering recommended" if ok else "not shipped" if js["verdict"] == "improved"
                          else "no candidate")
        rec["says"] = (f"steering {st['vector']} x {st['alpha']} after layers {st['layers']}"
                       + (f" on the weight patch {sbase.label} (apply both)" if sbase is not None else "")
                       + f": {rec['status']}" + (f" ({rec['status_why']})" if not ok else ""))
        return _finish(rec, out, t0)
    pick = None
    if tools is not None:
        wb = tools.wb
        if tools.steerings and "agent" in rec["stages"]:
            rec["stages"]["agent"]["steer_evals"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                                     for k, v in tools.steerings.items()}
        # Only what the agent chose is judged. Settings it measured but did not submit are not judged on its
        # behalf: it can have reasons the practice gain does not show (a loop goal's agent measured a repetition
        # penalty fixing every loop, wrote that it lost harmful refusals and was not safe to submit, and an
        # earlier fallback judged it anyway).
        if wb is not None and wb.done and "final_settings" in wb.done:
            pick = (tools.settings[wb.done["final_settings"]], f"agent (final choice: {wb.done['rationale'][:200]})")
        if tools.settings:
            rec["stages"]["agent"]["settings_evals"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                                        for k, v in tools.settings.items()}
    if pick is not None and candidate is None:
        sel, by = pick
        bst = tools.steerings.get(sel.get("steering_id") or "")       # a runtime bundle: settings + steering
        sbase = sel.get("_base")                                      # the committed stack it sits on, if any
        if tools.accepted is not None and tools.accepted[0] == "settings" and \
                tools.settings.get(tools.accepted[1]) is sel:
            js = tools.accepted[2]                       # judged when it was chosen: not looked at again
            shown = True                                 # ... and its report was printed then
        else:
            say(f"\njudging the settings {sel['settings']} ({by}) on the held-out set")
            js = judge_settings(lm, obj, sel["settings"], limit=eval_limit, say=say, m=slots, st=bst, base=sbase)
            shown = False
        js["by"] = by
        js["practice"] = {k: sel.get(k) for k in ("n", "before", "after", "diff", "se")}
        rec["stages"]["settings"] = js
        if not shown:
            say("\n".join(render_settings(js, proto)))
        what = (json.dumps(sel["settings"], sort_keys=True) + (f" + steering {bst['vector']}" if bst else "")
                + (f" on the weight patch {sbase.label}" if sbase is not None else ""))

        def files(d: Path) -> Path:
            _settings_files(sel["settings"], proto, d, steering=bst is not None, base=sbase, lm=lm)
            if bst is not None:
                save_steering(bst, d, lm=lm, with_base=False)
            return d
        kind_ = ("patch+" if sbase is not None else "") + ("settings+steering" if bst else "settings")
        ok = _accept(lm, obj, rec, kind=kind_, label=what, by=by, val=js,
                     stage=out / "candidate_settings", write_files=files,
                     run_after=_settings_runner(lm, obj, sel["settings"], st=bst, base=sbase), evidence=None, say=say)
        js["files"] = str(_final_dir(out / "candidate_settings", out / ("settings" if ok else "candidate_settings")))
        rec["outcome"] = ("settings recommended" if ok else "not shipped" if js["verdict"] == "improved"
                          else "no candidate")
        rec["says"] = (f"serve with {what}: {rec['status']}" if ok else
                       f"settings {what}: {rec['status']} ({rec['status_why']})")
        return _finish(rec, out, t0)
    if candidate is None:
        rec["outcome"] = "not shipped" if rejected else "no candidate"
        done = (tools.wb.done or {}) if tools is not None and tools.wb is not None else {}
        att = tools.attempts if tools is not None else []
        # An agent that ran out of turns, budget or provider says so (not the playbook's summary it started from).
        stopped = (rec["stages"].get("agent") or {}).get("stopped")
        why_none = "; ".join(x for x in (
            f"the playbook's candidate ({rejected}) was not accepted by the held-out judge" if rejected else "",
            f"the agent stopped: {stopped}" if stopped else "") if x)
        rec["says"] = (f"the agent gave up: {done['give_up'][:300]}" if done.get("give_up") else
                       f"none of the agent's {len(att)} attempt(s) was accepted on validation: "
                       + "; ".join(f"{a['kind']} {a['id']}: {a['verdict']}" for a in att) if att else
                       why_none or rep.summary)
        if att:
            rec["outcome"] = "no candidate"            # as for a judged steering or settings the judge rejected
            # the last attempt's full validation result goes in the report (the agent saw only its symptoms)
            rec["stages"][{"edit": "judge", "steering": "steering", "settings": "settings"}[att[-1]["kind"]]] =                 tools.results[-1]
        if rejected:
            rec["patch"] = {"label": rejected, "by": "playbook", "dir": str(out / "candidate_playbook")}
        return _finish(rec, out, t0)
    if tools is not None and tools.accepted is not None and tools.accepted[0] == "edit" and \
            tools.wb is not None and tools.wb.edits.get(tools.accepted[1]) is candidate:
        ev = tools.accepted[2]                           # judged when it was chosen: not looked at again
    else:
        ev = judge_patch(candidate, by, suite_result)
    return _ship(rec, out, t0, candidate, by, ev, obj, consent, decisions, lm=lm, say=say)


def _steer_runner(lm, obj: Objective, st: dict):
    def run(items: list[dict]) -> list[dict]:
        kw = steer_kw(st)
        r = PF.Runner(TorchBackend(lm), obj.protocol, obj.grader)
        with _under(lm, st.get("_base"), kw):
            return EV._run_set(r, items, samples=1, seed=0)
    return run


def _settings_runner(lm, obj: Objective, settings: dict, st: Optional[dict] = None, base=None):
    def run(items: list[dict]) -> list[dict]:
        r = PF.Runner(TorchBackend(lm), with_settings(obj.protocol, settings), obj.grader)
        with _under(lm, base, steer_kw(st) if st is not None else None):
            return EV._run_set(r, items_for(settings, items), samples=1, seed=0)
    return run


def _ship(rec: dict, out: Path, t0: float, candidate, by: str, ev: dict, obj: Objective, consent, decisions, *,
          lm=None, say=lambda s: None) -> dict:
    from ..surgery.edits import applied

    rec["stages"]["judge"] = ev
    candidate.checks["objective"] = {"name": obj.name, "verdict": ev["verdict"], "paired": ev.get("paired"),
                                     "by": by}

    def run_after(items: list[dict]) -> list[dict]:
        r = PF.Runner(TorchBackend(lm), obj.protocol, obj.grader)
        with applied(lm, candidate):
            return EV._run_set(r, items, samples=1, seed=0)
    shipped = _accept(lm, obj, rec, kind="patch", label=candidate.label, by=by, val=ev, stage=out / "candidate",
                      write_files=lambda d: candidate.save(d, lm), run_after=run_after,
                      evidence=(candidate.checks or {}).get("evidence"), say=say)
    final = _final_dir(out / "candidate", out / ("patch" if shipped else "candidate"))
    rec["outcome"] = "shipped" if shipped else "not shipped"
    rec["patch"] = {"label": candidate.label, "by": by, "dir": str(final)}
    pw = ev.get("power") or {}
    rec["says"] = (rec["status"] if shipped else
                   f"{ev['verdict']}: {rec['status_why']}" if ev["verdict"] != "improved" else
                   f"passed validation, then {rec['status']}: {rec['status_why']}")
    if not shipped and pw and not pw.get("can_show_improvement") and ev["verdict"] != "improved":
        rec["says"] += f" ({pw['note']})"
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
    if rec.get("status"):
        lines += ["## Decision", f"**{rec['status']}** - {rec.get('status_why', '')}"]
        cf = st.get("confirmation")
        if cf:
            p = cf.get("paired") or {}
            lines.append(f"- confirmation set ({cf.get('items')} items, evaluated once, per-item results not shown): "
                         f"{cf['verdict']}" + (f", {p.get('before', 0):.1%} -> {p.get('after', 0):.1%} "
                                               f"({p.get('wins')} better, {p.get('losses')} worse)" if p else "")
                         + f"; {cf.get('reason', '')}")
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
            lines.append(f"- guardrail {g['name']}: {'holds' if g['holds'] else 'FAILS'}{EV.guard_scope_note(g)}")
        if js.get("power") and not js["power"]["can_show_improvement"]:
            lines.append(f"- note: {js['power']['note']}")
        lines.append("")
    if "steering" in st:
        js = st["steering"]
        p = js["paired"]
        how = (f"Clamp `{js['vector']}` to {js['alpha']:.3f}" if js.get("mode") == "clamp"
               else f"Add {js['alpha']} x `{js['vector']}`")
        lines += ["## Recommended steering" if rec["outcome"] == "steering recommended"
                  else "## Steering tried on the held-out set",
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
        if js.get("base_patch"):
            lines.append(f"- on top of the weight patch {js['base_patch']['label']} (a committed stack): judged "
                         "together")
        if js.get("files"):
            lines += ["", f"Files: `{js['files']}` (the vector, its settings, and APPLY.md with the hook to apply it). "
                      + ("The weight patch it was judged on is in `patch/`: apply it to the weights first, then the "
                         "hook." if js.get("base_patch") else "The weights are unchanged.")]
        lines.append("")
    if "settings" in st:
        js = st["settings"]
        lines += ["## Recommended settings" if rec["outcome"] == "settings recommended"
                  else "## Settings tried on the held-out set",
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
