"""Deep triage: the eight steps of the product report, in order.

1. Trigger      reproduce the failure on the full-precision model, greedy, its own template
2. Contrast     failing prompts + near-identical passing variants (contrast.py)
3. Token trace  logit lens on the decision token: at which depth does the failure take over
4. Directions   mean(failing) - mean(passing) residual per depth, separation, leave-one-out accuracy
5. Localise     activation patching per depth / attention / MLP / head; neurons that write it;
                then causal tests: ablating the direction must fix the failure, adding it must cause it
6. Known        cosine with known directions (end-of-turn, refusal, non-English, repetition)
7. Weights      the implicated matrices: how strongly they write the direction, stats vs other layers,
                change vs a base model; with --gguf, the quantisation error of those tensors
8. Report       one sentence ("this failure comes from ..."), confidence, evidence, hand-off files
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from ..findings import Cause, Finding, Layer, Report, Severity
from . import analysis as A
from .contrast import Classifier, Contrast, build
from .lm import LM

SRC = "deep triage (product report, Milestone 3)"
MIN_SIDE = 3          # failing and passing runs needed on each side for a trustworthy contrast


def _f(code: str, sev: Severity, msg: str, *, cause: Cause = Cause.NONE, fix: Optional[str] = None,
       **ev: Any) -> Finding:
    return Finding(code, Layer.DEEP, sev, msg, cause=cause, fix=fix, source=SRC,
                   evidence={k: v for k, v in ev.items() if v is not None})


@dataclass
class DeepResult:
    report: Report
    contrast: Optional[Contrast] = None
    traces: list[A.Trace] = field(default_factory=list)
    directions: Optional[A.Directions] = None
    known: dict[str, Any] = field(default_factory=dict)
    loc: Optional[A.Localisation] = None
    causal: Optional[A.Causal] = None
    weights: dict[str, Any] = field(default_factory=dict)
    quant: list[dict] = field(default_factory=list)
    selection: Optional[A.Selection] = None
    known_ablation: dict[str, Any] = field(default_factory=dict)
    lang: dict[str, Any] = field(default_factory=dict)
    copy: dict[str, Any] = field(default_factory=dict)
    loop: dict[str, Any] = field(default_factory=dict)
    qbisect: dict[str, Any] = field(default_factory=dict)
    known_vectors: Any = None                       # A.KnownDirections, for surgery
    lm: Any = None
    summary: str = ""
    confidence: str = "low"


def diagnose(lm: LM, messages: list[dict], **kw: Any) -> DeepResult:
    """Deep triage of one failing conversation (see the module docstring for the eight steps).

    Clean forward passes are reused within one run (lm.memo()): the same prompt's clean pass
    feeds several steps and recomputing it gives the same numbers.
    """
    from . import diagcache as DC

    dep = DC.depends(messages, kw)
    if dep is not None:
        res = DC.load(lm, dep)
        if res is not None:
            if kw.get("out_dir"):
                write_handoff(res, Path(kw["out_dir"]))
            return res
    t0 = time.perf_counter()
    with lm.memo():
        res = _diagnose(lm, messages, **kw)
    if dep is not None:
        DC.save(lm, dep, res, time.perf_counter() - t0)
    return res


def _diagnose(lm: LM, messages: list[dict], *, kind: str = "auto", failing: tuple[str, ...] = (),
             passing: tuple[str, ...] = (), expect_lang: Optional[str] = None, max_new: int = 48,
             template_kwargs: Optional[dict] = None, base: Optional[LM] = None, gguf: Optional[str] = None,
             reference_dir: Optional[str] = None, known: bool = True, out_dir: Optional[Path] = None,
             probes: bool = True) -> DeepResult:
    """probes: run the calibrated probe for the failure kind (language probe, copy-head probe)."""
    rep = Report()
    res = DeepResult(rep)
    res.lm = lm
    lm.stats.clear()
    lm.stage("contrast")
    rep.context.update(model=lm.name, layers=lm.n_layers, d_model=lm.d_model, heads=lm.n_heads)

    # 1-2. reproduce, then the contrast set
    c = build(lm, messages, kind=kind, extra_failing=failing, extra_passing=passing, expect_lang=expect_lang,
              max_new=max_new, template_kwargs=template_kwargs)
    res.contrast = c
    orig = c.tried[0]
    rep.context["original_reply"] = orig.text[:300]
    if not orig.failing:
        rep.findings.append(_f("DEEP_NOT_REPRODUCED", Severity.FAIL,
                               f"the full-precision model answers this prompt fine ({orig.text[:80]!r}): the failure "
                               "is not in the weights", cause=Cause.QUANTISATION,
                               fix="the served copy differs: compare quantisation (`wsr quant <gguf> --reference "
                                   "<hf dir>`), backend and template; run `wsr triage --differential`",
                               reply=orig.text[:200]))
        if gguf and reference_dir:
            res.quant = _quant_check(gguf, reference_dir, None)
            _quant_findings(rep, res.quant)
            # Which quantised tensor causes it: rebuild the served model from the GGUF and bisect.
            from . import qbisect as QB
            qm = QB.build(lm, gguf)
            qclf = Classifier(lm, expect_lang=expect_lang, max_new=max_new, template_kwargs=template_kwargs,
                              expect_refuse=kind == "complied")
            qb = QB.bisect(qm, qclf, messages)
            res.qbisect = qb
            o = qb.get("outcome")
            rep.findings.append(_f("DEEP_QBISECT", Severity.FAIL if o in ("localised", "necessary", "diffuse") else
                                   Severity.INFO,
                                   QB.summary(qb), cause=Cause.QUANTISATION if o in ("localised", "necessary", "diffuse",
                                                                                       "unconfirmed") else
                                   Cause.RUNTIME if o == "quantised_passes" else Cause.NONE,
                                   fix=("requantise with these tensors at a higher-bit type (llama-quantize "
                                        "--tensor-type <name>=q8_0) or with an importance matrix: " +
                                        ", ".join(qb.get("minimal_set", []))) if o in ("localised", "necessary")
                                   else None,
                                   **{k: v for k, v in qb.items() if k != "skipped"}))
        res.summary = "does not reproduce on full precision: not weight-level"
        longer = max(128, 4 * max_new)
        late = None
        if orig.gen and orig.gen[-1] not in lm.eot_ids and longer > max_new:
            # The reply was cut at max_new while still healthy. A failure that starts later (a loop
            # often needs 50-100 tokens) would be missed, so look once further before saying
            # "not weight-level".
            late = Classifier(lm, expect_lang=expect_lang, max_new=longer, template_kwargs=template_kwargs,
                              expect_refuse=kind == "complied").run(
                "original_longer", messages)
        if late is not None and late.groups:
            res.summary = (f"does not fail within {max_new} new tokens on full precision, but fails later "
                           f"({', '.join(sorted(late.groups))} within {longer}): rerun with --max-new {longer}")
            rep.findings = [f for f in rep.findings if f.code != "DEEP_NOT_REPRODUCED"]
            rep.findings.append(_f("DEEP_CUT_SHORT", Severity.WARN,
                                   f"the full-precision reply was healthy at {max_new} new tokens but shows "
                                   f"{', '.join(sorted(late.groups))} within {longer}: {late.text[-120:]!r}",
                                   fix=f"rerun with --max-new {longer} so the diagnosis sees the failure",
                                   max_new=max_new, longer=longer, groups=sorted(late.groups)))
        rep.verdict = _verdict(res)
        return _finish(res, out_dir)
    rep.findings.append(_f("DEEP_REPRODUCED", Severity.INFO,
                           f"reproduces on full precision ({c.kind}): {orig.text[:80]!r}", kind=c.kind))
    thin = len(c.passing) < MIN_SIDE or len(c.failing) < MIN_SIDE
    note = ""
    if not c.passing:
        note = ": no passing variant, so nothing to contrast with"
    elif thin:
        note = (f": fewer than {MIN_SIDE} on one side, so directions, separation and localisation rest on very "
                "few runs")
    rep.findings.append(_f("DEEP_CONTRAST", Severity.WARN if thin else Severity.INFO,
                           f"{len(c.failing)} failing and {len(c.passing)} passing prompts out of {len(c.tried)} "
                           "near-identical variants" + note,
                           fix=("pass more prompts with --passing (that work) and --failing (that fail)"
                                if thin else None),
                           examples=[e.brief() for e in c.tried]))

    lm.stage("trace")
    # 3. token trace
    ts = A.targets(lm, c)
    res.traces = [A.token_trace(lm, t) for t in ts[:3]]
    if res.traces:
        tr = res.traces[0]
        where = f"layer {tr.onset_depth - 1}" if tr.onset_depth else "the embeddings"
        at = ""
        m0 = ts[0].ex.meta
        if "loop_item" in m0:
            at = f"at list item {m0['loop_item'] + 1}, where {m0.get('repeated', '')!r} starts repeating: "
        elif m0.get("eot_at"):
            at = f"at token {m0['eot_at'] + 1} of the reply, where it ends its turn: "
        elif "loop_onset" in m0:
            at = f"at token {m0['loop_onset'] + m0['loop_period']} of the reply, where the repeat starts: "
        rep.findings.append(_f("DEEP_TRACE", Severity.INFO,
                               at + f"'{tr.fail_token}' beats the best alternative '{tr.alt_token}' from {where} "
                               f"onward; the biggest jump is at layer {tr.biggest_jump_depth - 1} "
                               f"(p={tr.prob_by_depth[-1]:.2f} at the output)",
                               onset_layer=None if tr.onset_depth is None else tr.onset_depth - 1,
                               jump_layer=tr.biggest_jump_depth - 1,
                               metric_by_depth=[round(x, 3) for x in tr.metric_by_depth]))

    lm.stage("probes")
    # 6b. calibrated, kind-specific probes (independent of the small contrast set)
    if probes and c.kind == "language" and expect_lang:
        res.lang = _language_probe(lm, c, expect_lang, max_new, template_kwargs, rep)
    if probes and c.kind == "loop" and ts:
        res.copy = _copy_probe(lm, c, ts, expect_lang, max_new, template_kwargs, rep)

    if not c.passing:
        res.summary = f"{c.kind} reproduces on full precision; no passing variant to localise against"
        if res.lang and res.lang.get("summary"):
            res.summary += "; " + res.lang["summary"]
        if probes and c.kind == "loop" and ts:
            # The loop battery needs no passing prompts: run it even when every variant loops.
            res.loop = _loop_battery(lm, c, ts, expect_lang, max_new, template_kwargs, res, rep)
            yes = res.loop.get("supported") or []
            res.summary += "; loop mechanisms with their own test: " + (", ".join(yes) if yes else "none")
            if yes:
                cause = (Cause.QUANTISATION if "quantisation" in yes else
                         Cause.SAMPLING if yes == ["greedy decoding"] else Cause.WEIGHT_LEVEL)
                res.confidence = "medium"
                rep.findings.append(_f("DEEP_VERDICT", Severity.FAIL, res.summary + " (confidence: medium)",
                                       cause=cause, fix="see DEEP_LOOP_MECHANISMS for the components each test "
                                                        "names; with no passing variant, direction finding was skipped",
                                       confidence="medium", kind=c.kind))
        rep.verdict = _verdict(res)
        return _finish(res, out_dir)

    lm.stage("directions")
    # 4. directions
    d = A.find_directions(lm, c)
    sel = None
    scorer = A.AblationScorer(lm, c, ts) if d is not None and ts else None
    if d is not None and ts:
        sel = A.select_direction(lm, c, ts, d, sc=scorer)
        if sel.chosen_depth != d.best_depth:
            d = A.find_directions(lm, c, force_depth=sel.chosen_depth)
    res.directions = d
    res.selection = sel
    if d is not None:
        acc = d.loo_accuracy[d.best_depth]
        chosen = next((s for s in sel.scores if s["depth"] == d.best_depth), None) if sel else None
        how = ""
        if sel and chosen:
            how = (f"; chosen by ablation effect at the decision token: moves the failure "
                   f"{chosen['fixes']:.0%} of the way, flips {chosen['flipped']}/{chosen['of']}, passing KL "
                   f"{chosen['passing_kl']:.3f}")
            if "specific" in chosen:
                how += (f"; {'beats' if chosen['specific'] else 'does not beat'} shuffled-label directions "
                        f"(best {chosen['null_fixes_max']:.0%}, {chosen['null_flipped_max']}/{chosen['of']})")
            if sel.by_separation_depth != d.best_depth:
                how += f" (best separation was after layer {sel.by_separation_depth - 1})"
        rep.findings.append(_f("DEEP_DIRECTION", Severity.INFO,
                               f"failing and passing runs separate after layer {d.best_depth - 1} "
                               f"(Cohen's d {min(d.separation[d.best_depth], 99):.1f}"
                               + (f", leave-one-out accuracy {acc:.0%}" if acc is not None else "") + ")" + how,
                               best_layer=d.best_depth - 1,
                               separation=[round(min(x, 99), 2) for x in d.separation],
                               loo=[None if a is None else round(a, 2) for a in d.loo_accuracy],
                               selection=sel.scores if sel else None))

    lm.stage("known")
    # 6. known directions
    if known and d is not None:
        # The end-of-turn direction is a gradient; take it at a passing state. At a failing state
        # the residual already points along it, and RMSNorm is blind to scaling along its input.
        probe_at = lm.run(c.passing[0].decision_ids, capture=("resid",)).resid[-1]
        kd = A.known_directions(lm, probe_at)
        res.known_vectors = kd
        res.known = A.compare_known(d, kd)
        res.known_ablation = A.ablate_known(lm, c, ts, kd, d, sc=scorer) if ts else {}
        best = max(res.known.items(), key=lambda kv: abs(kv[1]["cos_at_best_depth"]), default=None)
        if best:
            name, v = best
            msg = (f"closest known direction: {name} (cosine {v['cos_at_best_depth']:+.2f} at "
                   f"layer {d.best_depth - 1}); others: " +
                   ", ".join(f"{k} {x['cos_at_best_depth']:+.2f}" for k, x in res.known.items() if k != name))
            ka = res.known_ablation
            spec = sorted(((n, v) for n, v in ka.items() if v.get("specific")), key=lambda kv: -kv[1]["score"])
            if spec:
                kn, kv = spec[0]
                msg += (f". Removing the known {kn} direction (layer {kv['layer']}) moves the failure "
                        f"{kv['fixes']:.0%} of the way and flips {kv['flipped']}/{kv['of']}, beyond shuffled-label "
                        f"directions (best {kv['null_fixes_max']:.0%}); passing KL {kv['passing_kl']:.3f}")
            elif ka:
                msg += (". Removing a known direction does no better than removing shuffled-label directions: "
                        "no known direction is specific to this failure")
            rep.findings.append(_f("DEEP_KNOWN_DIRECTION", Severity.INFO, msg,
                                   known={k: round(x["cos_at_best_depth"], 3) for k, x in res.known.items()},
                                   known_ablation=res.known_ablation or None))

    lm.stage("localise")
    # 5. localisation + causal tests
    loc = A.localise(lm, c, ts, d)
    res.loc = loc
    if loc and loc.pairs:
        L = lm.n_layers
        comps = [(f"layer {i} MLP", loc.mlp[i]) for i in range(L)] + [(f"layer {i} attention", loc.attn[i])
                                                                        for i in range(L)]
        comps.sort(key=lambda x: -x[1])
        heads = sorted(loc.heads.items(), key=lambda kv: -kv[1])[:3]
        writers = A.direction_writers(loc)
        msg = ""
        if writers and d is not None:
            msg = (f"the direction after layer {d.best_depth - 1} is written by " +
                   ", ".join(f"{n} {v:.0%}" for n, v in writers[:3] if v > 0) + "; ")
        rep.findings.append(_f("DEEP_LOCALISED", Severity.INFO,
                               msg + "the output logit is built by (patching from passing runs restores) " +
                               ", ".join(f"{n} {_pct(v)}" for n, v in comps[:3]) +
                               (("; heads " + ", ".join(f"{h} {_pct(v)}" for h, v in heads)) if heads else ""),
                               top=[{"component": n, "restored": round(v, 3)} for n, v in comps[:6]],
                               direction_writers=[{"component": n, "share": round(v, 3)} for n, v in writers[:8]],
                               resid=[round(x, 3) for x in loc.resid], heads=dict(heads),
                               neurons=loc.neurons[:5]))
    if d is not None:
        lm.stage("causal")
        clf = Classifier(lm, expect_lang=expect_lang, max_new=max_new, template_kwargs=template_kwargs,
                         expect_refuse=c.kind == "complied")
        cz = A.causal_tests(lm, c, d, clf)
        res.causal = cz
        rep.findings.append(_f("DEEP_CAUSAL", Severity.INFO,
                               f"removing the direction fixes {cz.ablate_fixed}/{cz.ablate_total} failing prompts "
                               f"and keeps {cz.ablate_passing_kept}/{cz.ablate_passing_total} passing ones passing; "
                               f"adding it breaks {cz.add_broke}/{cz.add_total} passing prompts",
                               samples=cz.samples))

    lm.stage("weights")
    # 7. weights (+ quantisation error of the implicated tensors)
    res.weights = A.weight_dossier(lm, d, loc, base=base)
    if res.weights.get("implicated"):
        w0 = res.weights["implicated"][0]
        extra = ""
        if loc and loc.neurons:
            tot = sum(abs(n["contribution"]) for n in loc.neurons) or 1.0
            n0 = loc.neurons[0]
            extra = (f"; neuron {n0['neuron']} of layer {n0['layer']} carries {abs(n0['contribution']) / tot:.0%} of "
                     f"what the top neurons write (activation {n0['act_fail']:.3g} failing vs {n0['act_pass']:.3g} "
                     "passing)")
        elif "heads_writing_most" in w0:
            extra = f"; heads writing most: {w0['heads_writing_most']}"
        rep.findings.append(_f("DEEP_WEIGHTS", Severity.INFO,
                               f"{w0['component']} (norm robust-z {w0['fro_robust_z']:+.1f} against the same matrix in "
                               f"other layers, kurtosis {w0['stats']['kurtosis']:.1f}){extra}",
                               implicated=res.weights["implicated"]))
    if gguf and reference_dir and res.weights.get("implicated"):
        res.quant = _quant_check(gguf, reference_dir, [w["layer"] for w in res.weights["implicated"]])
        _quant_findings(rep, res.quant)

    lm.stage("loop_battery")
    # 7b. loops: the mechanism battery (attractor, MLP, copy heads, positional, decoding, quantisation)
    if probes and c.kind == "loop" and ts:
        res.loop = _loop_battery(lm, c, ts, expect_lang, max_new, template_kwargs, res, rep)

    lm.stage("summary")
    # 8. summary
    res.summary, res.confidence = _summarise(lm, res)
    cause, fix = Cause.WEIGHT_LEVEL, f"hand the package in the output directory to the surgery playbook ({_playbook(c.kind)})"
    sup = set(res.loop.get("supported") or [])
    weight_mech = sup & {"attractor", "MLP self-reinforcement", "copy heads", "positional (RoPE)"}
    if "quantisation" in sup:
        cause, fix = Cause.QUANTISATION, "requantise the damaged tensors (see the quantisation findings)"
    elif "greedy decoding" in sup and not weight_mech:
        # Mechanism 5: a near-tie caught by greedy argmax; sampling escapes it. A runtime setting, not a weight bug.
        cause = Cause.SAMPLING
        fix = ("decode with sampling (temperature 0.6-0.8, top_p 0.9) or a DRY / repetition penalty; a weight repair "
               "is not indicated: no weight-level loop mechanism passed its test")
        d = res.loop.get("decoding", {})
        s = d.get("sampled_t0.7", {})
        p = res.traces[0].prob_by_depth[-1] if res.traces else None
        res.summary = (f"greedy decoding keeps this loop: sampling escapes it ({s.get('of', 0) - s.get('loops', 0)}/"
                       f"{s.get('of', 0)} sampled runs do not loop) and no weight-level loop mechanism (attractor, MLP, "
                       "copy heads, positional) passed its test"
                       + (f"; the repeated token is a near-tie (p={p:.2f})" if p is not None and p < 0.5 else ""))
    rep.findings.append(_f("DEEP_VERDICT", Severity.FAIL, res.summary + f" (confidence: {res.confidence})",
                           cause=cause, fix=fix, confidence=res.confidence, kind=c.kind))
    rep.verdict = _verdict(res)
    return _finish(res, out_dir)


def _language_probe(lm: LM, c: Contrast, expect: str, max_new: int, template_kwargs: Optional[dict],
                    rep: Report) -> dict[str, Any]:
    from . import langprobe as LP

    ex = next((e for e in c.failing if e.name == "original"), c.failing[0] if c.failing else None)
    if ex is None:
        return {}
    answered = LP.reply_language(ex.text)
    probe = LP.calibrate(lm, log=None)
    if expect not in probe.langs or answered not in probe.langs:
        rep.findings.append(_f("DEEP_LANG_PROBE", Severity.INFO,
                               f"language probe skipped: {expect!r} or the reply's language {answered!r} is not among "
                               f"its calibrated languages ({', '.join(probe.langs)})"))
        return {}
    tr = LP.trace(probe, lm, ex.decision_ids, expect, answered)
    late = range(probe.best_depth(), len(tr.p_expected))
    pe = sum(tr.p_expected[i] for i in late) / len(late)
    pa = sum(tr.p_answered[i] for i in late) / len(late)
    if pa > pe:
        mech = "plan"
        what = (f"wrong plan: from layer {tr.switch_depth - 1 if tr.switch_depth else '?'} the residual stream "
                f"itself encodes {LP.NAMES[answered]} (p={pa:.2f} over the late layers)")
    else:
        mech = "readout"
        what = (f"right plan, wrong token: the residual stream mostly encodes {LP.NAMES[expect]} (p={pe:.2f} over the "
                f"late layers), yet the first token is {LP.NAMES[answered]}: a small {LP.NAMES[answered]} component "
                "decides the readout")
    msg = LP.summary_line(tr, probe) + ". " + what[0].upper() + what[1:]
    mis = probe.meta.get("mismatch_accuracy", [])
    bd = probe.best_depth()
    ev = dict(expected=expect, answered=answered, mechanism=mech, best_layer=bd - 1,
              p_expected=[round(x, 3) for x in tr.p_expected], p_answered=[round(x, 3) for x in tr.p_answered],
              top_by_depth=tr.top, switch_layer=None if tr.switch_depth is None else tr.switch_depth - 1,
              probe_accuracy=[round(x, 3) for x in probe.loo_accuracy],
              probe_accuracy_when_request_ignored=mis[bd] if mis else None, calibration=probe.meta)
    rep.findings.append(_f("DEEP_LANG_PROBE", Severity.INFO, msg, **ev))
    # Causal: the calibrated direction, at the best depth (and the switch depth when it is earlier).
    from .contrast import Classifier
    clf = Classifier(lm, expect_lang=expect, max_new=max_new, template_kwargs=template_kwargs)
    depth = tr.switch_depth if (tr.switch_depth and tr.switch_depth < bd and mech == "plan") else bd
    st = LP.steer_tests(probe, lm, clf, c.failing[:3], expect, answered, depth)
    n = st["of"]
    smsg = (f"with the calibrated {LP.NAMES[answered]}-minus-{LP.NAMES[expect]} direction after layer {st['layer']}: "
            f"removing it gives {LP.NAMES[expect]} in {st['ablate_fixed']}/{n}, steering toward "
            f"{LP.NAMES[expect]} in {st['steer_fixed']}/{n}")
    if "null_fixed" in st:
        smsg += f"; steering along an unrelated pair ({st['null_pair']}) in {st['null_fixed']}/{n}"
    rep.findings.append(_f("DEEP_LANG_STEER", Severity.INFO, smsg, **st))
    fixed = max(st["ablate_fixed"], st["steer_fixed"])
    specific = fixed > st.get("null_fixed", 0)
    summ = what
    if fixed and specific:
        how = "removing" if st["ablate_fixed"] >= st["steer_fixed"] else "steering along"
        summ += (f"; {how} the calibrated {LP.NAMES[answered]}-minus-{LP.NAMES[expect]} direction after layer "
                 f"{st['layer']} restores {LP.NAMES[expect]} ({fixed}/{n}; an unrelated language pair: "
                 f"{st.get('null_fixed', 0)}/{n})")
    elif fixed:
        summ += "; steering restores it, but so does an unrelated language pair, so the direction is not specific"
    return dict(ev, steer=st, summary=summ, fixed=fixed, specific=specific)


def _copy_probe(lm: LM, c: Contrast, ts: list, expect_lang: Optional[str], max_new: int,
                template_kwargs: Optional[dict], rep: Report) -> dict[str, Any]:
    from . import copyheads as CH
    from .contrast import Classifier

    ind = CH.induction_scores(lm)
    ch = CH.copy_heads(lm, ts[0], ind)
    clf = Classifier(lm, expect_lang=expect_lang, max_new=max_new, template_kwargs=template_kwargs)
    ab = CH.ablation_tests(lm, c, ts, ch, clf)
    msg = CH.summary(ch, ab, ind)
    ev = CH.as_evidence(ch, ab, ind)
    rep.findings.append(_f("DEEP_COPY_HEADS", Severity.INFO, msg[0].upper() + msg[1:], **ev))
    ok = bool(ab.get("chosen_k")) and ab.get("gen_fixed", 0) > ab.get("null_gen_fixed", 0)
    return dict(ev, summary=msg, fixed=ab.get("gen_fixed", 0), specific=ok)


def _loop_battery(lm: LM, c: Contrast, ts: list, expect_lang: Optional[str], max_new: int,
                  template_kwargs: Optional[dict], res: "DeepResult", rep: Report) -> dict[str, Any]:
    from . import loopprobes as LB
    from .contrast import Classifier

    clf = Classifier(lm, expect_lang=expect_lang, max_new=max_new, template_kwargs=template_kwargs)
    b = LB.run_all(lm, c, ts, clf, copy=res.copy or None, quant=res.quant if res.quant else None)
    rows = LB.summary_rows(b)
    yes = [n for n, ok, _ in rows if ok]
    msg = ("loop mechanisms supported by their own test: " + (", ".join(yes) if yes else "none") + ". " +
           " | ".join(f"{n}: {'YES' if ok else 'no'} ({line})" for n, ok, line in rows))
    rep.findings.append(_f("DEEP_LOOP_MECHANISMS", Severity.INFO, msg, battery=b,
                           table=[{"mechanism": n, "supported": ok, "evidence": line} for n, ok, line in rows]))
    pz = b.get("positional") or {}
    if pz.get("positional_defect"):
        kl = max(r["kl"] for r in pz["by_offset"])
        rep.findings.append(_f("DEEP_POSITIONAL_DEFECT", Severity.WARN,
                               f"shifting every position id changes the next-token distribution (KL up to {kl:.3f}, "
                               f"{pz['dtype']}); correct RoPE is shift-invariant, so positions are aliased or lose "
                               "precision at large angles", cause=Cause.RUNTIME,
                               fix="check the rotary implementation and its dtype (compute RoPE in float32), and "
                                   "the context length against max_position_embeddings", by_offset=pz["by_offset"]))
    return dict(b, supported=yes)


def _pct(v: float) -> str:
    """Restoration share; patching can overshoot (the passing run's value is further from the failure)."""
    return ">100%" if v > 1.05 else f"{max(v, 0):.0%}"


def _playbook(kind: str) -> str:
    return {"blank": "premature end-of-turn", "loop": "repetition and token collapse", "refusal": "over-refusal",
            "complied": "under-refusal", "language": "wrong language"}.get(kind, "free-form exploration")


def _summarise(lm: LM, res: DeepResult) -> tuple[str, str]:
    c, d, loc, cz = res.contrast, res.directions, res.loc, res.causal
    parts = []
    causal_ok = bool(cz and cz.ablate_total and cz.ablate_fixed / cz.ablate_total >= 0.5)
    if d is not None and causal_ok:
        parts.append(f"this {c.kind} failure comes from a direction that separates failing from passing runs "
                     f"after layer {d.best_depth - 1}")
    elif d is not None:
        parts.append(f"this {c.kind} failure goes with a direction after layer {d.best_depth - 1} (separates the "
                     f"runs, but removing it fixes {cz.ablate_fixed if cz else 0}/{cz.ablate_total if cz else 0}: "
                     "correlated, not shown to cause it)")
    else:
        parts.append(f"this {c.kind} failure")
    if d is not None and res.known:
        name, v = max(res.known.items(), key=lambda kv: abs(kv[1]["cos_at_best_depth"]))
        spec_known = res.known_ablation.get(name, {}).get("specific")
        if abs(v["cos_at_best_depth"]) >= 0.5 or (abs(v["cos_at_best_depth"]) >= 0.3 and spec_known):
            parts.append(f"resembling the known {name} direction (cos {v['cos_at_best_depth']:+.2f})")
    ka = {n: v for n, v in res.known_ablation.items() if v.get("specific")}
    if ka:
        kn, kv = max(ka.items(), key=lambda kv: kv[1]["score"])
        if kv["flipped"] and kv["fixes"] >= 0.5:
            parts.append(f"and removing the generic {kn} direction (layer {kv['layer']}) also removes it "
                         f"({kv['flipped']}/{kv['of']} at the decision token)")
    writers = []
    if loc and loc.pairs:
        dw = [(n, v) for n, v in A.direction_writers(loc) if v >= 0.15 and n != "embeddings"]
        writers = [n for n, _ in dw[:2]]
        if writers:
            w = "written by " + " and ".join(writers)
            if loc.neurons:
                n0 = loc.neurons[0]
                w += f" (top neuron: layer {n0['layer']} #{n0['neuron']})"
            parts.append(w)
        L = lm.n_layers
        comps = sorted([(f"layer {i} MLP", loc.mlp[i]) for i in range(L)] +
                       [(f"layer {i} attention", loc.attn[i]) for i in range(L)], key=lambda x: -x[1])
        # Only components at or after the direction's layer can turn it into the output token.
        D = d.best_depth if d is not None else 0
        late = [n for n, v in comps if v >= 0.2 and n not in writers and int(n.split()[1]) >= D - 1][:2]
        if late:
            parts.append("turned into the output token by " + " and ".join(late))
    if res.traces and res.traces[0].prob_by_depth[-1] < 0.5:
        parts.append(f"and the decision is a close call (p={res.traces[0].prob_by_depth[-1]:.2f} for the failure "
                     "token): small changes to the prompt or the activations tip it")
    if res.loop.get("supported") is not None:
        yes = res.loop["supported"]
        parts.append("loop mechanisms with their own causal or control test: " + (", ".join(yes) if yes else
                     "none of attractor / MLP / copy heads / positional / greedy decoding"))
    probe = res.lang if res.lang.get("summary") else res.copy if res.copy.get("summary") else {}
    label = "language probe" if probe is res.lang else "copy heads"
    if probe and probe.get("specific") and probe.get("fixed"):
        # A calibrated probe confirmed by intervention is the strongest evidence: lead with it.
        lead = f"this {c.kind} failure: {probe['summary']}"
        rest = "; ".join(parts[1:]) if len(parts) > 1 else ""
        summary = lead + (f". Contrast set: {parts[0]}" + (f", {rest}" if rest else "") if parts else "")
    else:
        if probe:
            parts.append(f"{label}: {probe['summary']}")
        summary = ", ".join(parts)
    score = 0
    if cz and cz.ablate_total:
        score += 2 if cz.ablate_fixed / cz.ablate_total >= 0.67 else (1 if cz.ablate_fixed else 0)
    if cz and cz.add_total and cz.add_broke:
        score += 1
    if loc and loc.pairs and writers:
        score += 1
    if d is not None and (d.loo_accuracy[d.best_depth] or 0) >= 0.99:
        score += 1
    if cz and cz.ablate_passing_total and cz.ablate_passing_kept < cz.ablate_passing_total / 2:
        score -= 1                     # removing it breaks healthy prompts: not a clean, separable cause
    chosen = next((s for s in res.selection.scores if d is not None and s["depth"] == d.best_depth), None) \
        if res.selection else None
    if chosen is not None and chosen.get("specific") is False:
        score -= 1                     # shuffled-label directions do as well: the effect is not specific
        summary += "; removing shuffled-label directions works about as well, so the direction is not specific"
    if len(c.passing) < MIN_SIDE or len(c.failing) < MIN_SIDE:
        score -= 1                     # thin contrast set
    if (res.lang.get("fixed") and res.lang.get("specific")) or res.copy.get("specific"):
        score += 2                     # a calibrated probe, causally confirmed
    conf = "high" if score >= 4 else "medium" if score >= 2 else "low"
    probe_ok = bool(probe and probe.get("specific") and probe.get("fixed"))
    if not causal_ok and not probe_ok and conf == "high":
        conf = "medium"
    if not probe_ok and chosen is not None and chosen.get("specific") is False and conf == "high":
        conf = "medium"                # nothing beats the null: a high rating would overstate it
    if probe_ok:
        n_fix, n_all = probe.get("fixed", 0), (probe.get("steer") or {}).get("of") or probe.get("ablation", {}).get(
            "gen_total") or 3
        conf = "high" if n_fix * 3 >= 2 * n_all else max(conf, "medium", key=["low", "medium", "high"].index)
    return summary, conf


def _quant_check(gguf: str, reference_dir: str, layers: Optional[list[int]]) -> list[dict]:
    from .quant import analyse

    if layers is None:
        return analyse(gguf, reference_dir).worst_errors(12)
    from ..gguf import read_gguf

    n = read_gguf(gguf).arch_key("block_count")
    spread = set(layers)
    if isinstance(n, int) and n > 0:
        # Peers for the per-file baseline (AWQ-processed files): a spread of other layers.
        spread |= {round(i * (n - 1) / 5) for i in range(6)}
    only = r"^blk\.(" + "|".join(str(i) for i in sorted(spread)) + r")\."
    q = analyse(gguf, reference_dir, only=only)
    keep = re.compile(r"^blk\.(" + "|".join(str(i) for i in sorted(set(layers))) + r")\.")
    return [e for e in q.worst_errors(10 ** 6) if keep.match(e["tensor"])][:12]


def _quant_findings(rep: Report, worst: list[dict]) -> None:
    bad = [w for w in worst if w.get("ratio_to_expected", 0) >= 3]
    if bad:
        rep.findings.append(_f("DEEP_QUANT_DAMAGE", Severity.FAIL,
                               f"{len(bad)} tensor(s) are quantised far worse than their type allows: " +
                               ", ".join(f"{w['tensor']} {w['rel_error']:.1%} ({w['ratio_to_expected']:.0f}x)"
                                         for w in bad[:4]), cause=Cause.QUANTISATION,
                               fix="requantise from the original (or a higher-bit type for these tensors / an "
                                   "imatrix quant)", worst=bad))
    elif worst:
        rep.findings.append(_f("DEEP_QUANT_OK", Severity.PASS,
                               f"quantisation error within the type's normal range (worst {worst[0]['tensor']} "
                               f"{worst[0].get('rel_error', 0):.1%})"))


def _verdict(res: DeepResult) -> dict[str, Any]:
    from ..verdict import _brief

    fs = res.report.findings
    primary = next((f for f in fs if f.code in ("DEEP_VERDICT", "DEEP_NOT_REPRODUCED")), None)
    q = next((f for f in fs if f.code == "DEEP_QUANT_DAMAGE"), None)
    cut = next((f for f in fs if f.code == "DEEP_CUT_SHORT"), None)
    if primary is None and cut is not None:
        return {"verdict": "inconclusive", "summary": res.summary, "primary": _brief(cut), "next_steps": [cut.fix]}
    if primary is None:
        return {"verdict": "unexplained", "summary": res.summary, "primary": None, "next_steps": []}
    verdict = primary.cause.value
    if primary.code == "DEEP_NOT_REPRODUCED" and q is None:
        verdict = "not_weight_level"
    return {"verdict": verdict, "summary": res.summary, "confidence": res.confidence, "primary": _brief(primary),
            "contributing": [_brief(q)] if q else [], "next_steps": [primary.fix] if primary.fix else []}


def _finish(res: DeepResult, out_dir: Optional[Path]) -> DeepResult:
    lm = res.lm
    if lm is not None:
        lm.stage(None)
        t = {k: {kk: (round(vv, 2) if kk == "s" else int(vv)) for kk, vv in v.items()}
             for k, v in lm.stats.items() if k not in ("load", "other") and v["s"] > 0.005}
        res.report.context["timings"] = t
        tot = sum(v["s"] for v in t.values())
        res.report.findings.append(_f("DEEP_TIMINGS", Severity.INFO,
                                      f"{tot:.0f} s in all: " + ", ".join(
                                          f"{k} {v['s']:.0f} s ({v['fwd']} forwards, {v['gen']} generations)"
                                          for k, v in sorted(t.items(), key=lambda kv: -kv[1]["s"])),
                                      timings=t, total_s=round(tot, 2)))
    if out_dir:
        write_handoff(res, Path(out_dir))
    return res


def write_handoff(res: DeepResult, out: Path) -> None:
    """The surgery package: prompts, traces, directions, components, weights."""
    out.mkdir(parents=True, exist_ok=True)
    c = res.contrast
    (out / "report.json").write_text(json.dumps(res.report.as_dict(), indent=1, default=str), encoding="utf-8")
    if c:
        (out / "contrast.json").write_text(json.dumps({"kind": c.kind, "examples": [e.brief() for e in c.tried]},
                                                      indent=1, ensure_ascii=False), encoding="utf-8")
    if res.traces:
        (out / "traces.json").write_text(json.dumps([t.__dict__ for t in res.traces], indent=1, default=str),
                                         encoding="utf-8")
    if res.directions is not None:
        torch.save({"by_depth": res.directions.by_depth, "best_depth": res.directions.best_depth}, out / "direction.pt")
    if res.loc:
        (out / "localisation.json").write_text(json.dumps(res.loc.__dict__, indent=1, default=str), encoding="utf-8")
    if res.weights:
        (out / "weights.json").write_text(json.dumps(res.weights, indent=1, default=str), encoding="utf-8")
    (out / "SUMMARY.md").write_text(f"# Deep triage\n\n{res.summary}\n\nConfidence: {res.confidence}\n",
                                    encoding="utf-8")
