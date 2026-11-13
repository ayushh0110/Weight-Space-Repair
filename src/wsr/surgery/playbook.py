"""Surgery playbook: from a deep diagnosis to the smallest weight edit that fixes it.

Each entry turns the evidence the deep layer produced into candidate edits,
and each candidate into a ladder of strengths, weakest first. A step is screened
cheaply (the decision token; for loops also the matched continuations), then
verified on generation and against the regression suite; the first step that
passes every check of the mode is the patch. When none does, the result says why
and what to do instead: an honest "could not fix".

Two modes (`wsr repair`, `--deep`):

  fast  the diagnosis's own candidates; checks: the failing / working prompts, the
        regression suite, and the patch again at serving precision (bf16 / fp16).
  deep  also the local edits (sites.py): the direction taken out of only the layers,
        or only the MLP neurons, that write it, at strengths up to 6 - the shape of
        the edit that fixed Qwen3.5-0.8B's reasoning loops (Takehi et al. 2026); for
        loops the direction comes from a failing and a working continuation of the
        same prompt (pairs.py). Candidates go smallest edit first. Extra checks:
        fresh prompt variants that took no part in choosing the patch, and a random
        edit of the same size as a control (reported, not used to reject); with
        `--sampled`, paired sampled runs at the model's own sampling settings.

Scope (product report): surgery removes, suppresses or redirects behaviour the
model already has. It cannot add knowledge or a skill.

Entries built:
  over-refusal           generic refusal direction (known library), contrast direction
  wrong language         calibrated answered-minus-expected language direction, contrast direction
  premature end-of-turn  the localised neurons, contrast direction, end-of-turn direction
  repetition / collapse  the loop battery's components: MLP neurons, copy heads, attractor direction
  anything else          the contrast direction from the user's failing / passing prompts
  (deep, every entry)    sparse and local edits on the sites writing the failure
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from ..deep import analysis as A
from ..deep.contrast import Classifier, _with_last_user, fresh_variants, Reference
from ..deep.lm import LM
from . import edits as E
from . import pairs as PR
from . import regress as R
from . import sites as S


@dataclass
class Candidate:
    label: str
    evidence: str
    ladder: list[tuple[str, Callable[[], E.Patch]]]      # (strength label, builder), weakest first
    check: Optional[Callable[[], tuple[float, float]]] = None   # (KL weight edit vs hook, KL hook vs clean), full strength
    scope: str = "global"                                # component | sparse | local | global (smallest first in deep)


@dataclass
class Entry:
    name: str
    could_not_fix: str


ENTRIES = {
    "refusal": Entry("over-refusal",
                     "no single small edit removes this refusal without breaking other prompts: fine-tune on the "
                     "over-refused requests (a few hundred LoRA pairs), or use a larger model; at runtime, a system "
                     "prompt that allows the topic"),
    "language": Entry("wrong language",
                      "fine-tune on replies in the requested language, or use a larger model; at runtime, a system "
                      "prompt written in the target language, or prefill the first words in it"),
    "blank": Entry("premature end-of-turn",
                   "fine-tune on complete answers to these prompts; at runtime, forbid end-of-turn for the first "
                   "tokens (llama-server logit_bias / min_tokens)"),
    "loop": Entry("repetition and token collapse",
                  "decode with sampling, a DRY sampler or a mild repetition penalty; fine-tune on long lists"),
}
GENERIC = Entry("contrast direction",
                "pass more failing and passing prompts (--failing / --passing) so the direction is better estimated, "
                "or fine-tune on the failing prompts")

DIRECTION_LADDER = (0.25, 0.5, 0.75, 1.0)
COMPONENT_LADDER = (0.5, 0.0)          # scale: halve, then remove
# Local edits: the rest of the network still writes the direction, so the edited sites may push
# against it. The Qwen3.5-0.8B loop fix was tested at 1, 2, 3, 4 and 6 and kept 6.
LOCAL_LADDER = (1.0, 2.0, 3.0, 4.0, 6.0)
_SCOPE_ORDER = {"component": 0, "sparse": 1, "local": 2, "gated": 3, "global": 4}


@dataclass
class Repair:
    status: str                                    # fixed | could_not_fix | not_indicated | verified | not_verified
    entry: str
    summary: str
    patch: Optional[E.Patch] = None
    trials: list[dict] = field(default_factory=list)
    regression: Optional[dict] = None
    target: Optional[dict] = None
    next_step: Optional[str] = None
    seconds: float = 0.0
    mode: str = "fast"                             # fast | deep (+ "sampled")
    checks: dict = field(default_factory=dict)     # serving, fresh, control, sampled, continuations, sites
    ctx: Any = field(default=None, repr=False)     # the verification context (for the agent); not saved
    sitemap: Any = field(default=None, repr=False)
    steering: Optional["Steering"] = None          # status "steered": an inference-time fix, not a patch


@dataclass
class Steering:
    """An inference-time intervention after each of `layers`, at every position; the weights do not change.
    mode "add": alpha x vector added (LM.steer(add=...)). mode "clamp": the component along the unit vector is
    lowered to `alpha` wherever it is above it, and left alone elsewhere (LM.steer(clamp=...)): it acts only where
    the failure's direction is present, so healthy text passes through unchanged. mode "gated": probe-gated steering
    (deep/harmprobe.py Gate): the refusal component is moved where the reply starts, by the model's own reading of
    harm (vector: the refusal direction; alpha: the raised level, or the lowered one when only that side acts)."""
    vector: torch.Tensor                           # [d_model], unit length
    alpha: float                                   # add: per layer; clamp: the level; gated: see above
    layers: list[int]
    label: str
    source: str = ""
    mode: str = "add"
    gate: Any = None                               # mode "gated": the harmprobe.Gate

    def kwargs(self) -> dict[str, Any]:
        if self.mode == "gated":
            return {"gated": self.gate}
        return {self.mode: (self.vector, self.alpha, self.layers)}

    def control_kwargs(self, r: torch.Tensor) -> dict[str, Any]:
        """The same intervention along another (random) direction."""
        if self.mode == "gated":
            return {"gated": self.gate.with_direction(r)}
        return {self.mode: (r, self.alpha, self.layers)}

    def as_dict(self) -> dict[str, Any]:
        out = {"label": self.label, "mode": self.mode, "alpha": round(self.alpha, 4), "layers": self.layers,
               "source": self.source, "vector_norm": round(float(self.vector.norm()), 6)}
        if self.gate is not None:
            out["gate"] = self.gate.as_dict()
        return out


STEER_SCALES = (0.5, 1.0, 1.5, 2.0)                # x the failing-minus-working gap at the chosen depth


# -- candidates ------------------------------------------------------------------------------------

def _kl(a: torch.Tensor, b: torch.Tensor) -> float:
    pa = a.float().log_softmax(-1)
    return float((pa.exp() * (pa - b.float().log_softmax(-1))).sum())


def _dir_candidate(lm: LM, label: str, evidence: str, v: torch.Tensor, probe_ids: torch.Tensor) -> Candidate:
    v = v.detach().float().cpu()

    def check() -> float:
        clean = lm.run(probe_ids, capture=()).logits
        hook = lm.run(probe_ids, capture=(), ablate=v).logits
        with E.applied(lm, E.orthogonalise(lm, v, 1.0)):
            w = lm.run(probe_ids, capture=()).logits
        return _kl(hook, w), _kl(clean, hook)
    ladder = [(f"alpha {a:g}", lambda a=a: E.orthogonalise(lm, v, a, label=f"{label}, alpha {a:g}", source=label))
              for a in DIRECTION_LADDER]
    return Candidate(label, evidence, ladder, check)


def _neuron_candidate(lm: LM, label: str, evidence: str, ns: list[tuple[int, int]],
                      probe_ids: torch.Tensor) -> Candidate:
    def check() -> float:
        clean = lm.run(probe_ids, capture=()).logits
        hook = lm.run(probe_ids, capture=(), zero_neurons=ns).logits
        with E.applied(lm, E.scale_neurons(lm, ns, 0.0)):
            w = lm.run(probe_ids, capture=()).logits
        return _kl(hook, w), _kl(clean, hook)
    ladder = [(f"x{s:g}", lambda s=s: E.scale_neurons(lm, ns, s, label=f"{label} x{s:g}", source=label))
              for s in COMPONENT_LADDER]
    return Candidate(label, evidence, ladder, check, scope="component")


def _head_candidate(lm: LM, label: str, evidence: str, hs: list[tuple[int, int]],
                    probe_ids: torch.Tensor) -> Candidate:
    def check() -> float:
        clean = lm.run(probe_ids, capture=()).logits
        hook = lm.run(probe_ids, capture=(), zero_heads=hs).logits
        with E.applied(lm, E.scale_heads(lm, hs, 0.0)):
            w = lm.run(probe_ids, capture=()).logits
        return _kl(hook, w), _kl(clean, hook)
    ladder = [(f"x{s:g}", lambda s=s: E.scale_heads(lm, hs, s, label=f"{label} x{s:g}", source=label))
              for s in COMPONENT_LADDER]
    return Candidate(label, evidence, ladder, check, scope="component")


def _local_candidates(lm: LM, sm: S.SiteMap, probe_ids: torch.Tensor) -> list[Candidate]:
    """Sparse (chosen MLP neurons only) and local (the chosen sites' whole matrices) edits."""
    out = []
    for sparse in (True, False):
        sites = sm.edit_sites(sparse)
        if not sites:
            continue
        what = ("neurons" if sparse else "layers")
        label = (f"{'sparse' if sparse else 'local'} {'continuation' if sm.source == 'continuations' else 'contrast'}"
                 f" direction on the {what} writing it")
        n_cols = sum(len(s.cols) for s in sites if s.cols is not None)

        def check(sites=sites) -> tuple[float, float]:
            clean = lm.run(probe_ids, capture=()).logits
            hook = lm.run(probe_ids, capture=(), project_sites=(sites, 1.0)).logits
            with E.applied(lm, E.project(lm, sites, 1.0)):
                w = lm.run(probe_ids, capture=()).logits
            return _kl(hook, w), _kl(clean, hook)
        ladder = [(f"alpha {a:g}", lambda a=a, sites=sites, label=label:
                   E.project(lm, sites, a, label=f"{label}, alpha {a:g}", source=label)) for a in LOCAL_LADDER]
        ev = ("the " + (f"{n_cols} neurons that " if sparse else f"{len(sites)} site(s) that ") + sm.describe(sparse))
        out.append(Candidate(label, ev, ladder, check, scope="sparse" if sparse else "local"))
    return out


def screened_candidate(lm: LM, prs: list, *, alpha: float = 3.0, keep: int = 2, min_ratio: float = 3.0,
                       max_neurons: int = 64) -> tuple[Optional[Candidate], dict[str, Any]]:
    """Deep mode, continuation failures with >= 4 matched pairs: choose the edit layers by a
    selectivity screen instead of by projection share, and confirm the choice on held-out pairs.

    Half the pairs (SELECT) give the direction and neurons; every MLP layer except the last gets a
    single-layer sparse edit at `alpha`, scored by how much it lowers the repeat against the working
    text. The `keep` layers with the best ratio (>= `min_ratio`) are combined, and the combined edit
    must also be selective on the other half (HELD), which played no part in the choice.

    Why (Qwen3.5-0.8B, the looping MATH-500 traces of Takehi et al.): projection share and the raw
    screen both pick the last layer, which writes the next token directly; edits there look selective
    on the pairs they were fitted to and are not on new ones. Middle layers chosen this way were
    selective on held-out pairs, as were the paper's own layers 15 and 19."""
    info: dict[str, Any] = {"pairs": len(prs), "alpha": alpha}
    if len(prs) < 4:
        info["why"] = "needs at least 4 matched pairs (half to choose, half to confirm)"
        return None, info
    sel, held = prs[::2], prs[1::2]
    sm = S.locate(lm, PR.states(sel), source="continuations", max_sites=2 * lm.n_layers, cover=1.0)
    if sm is None:
        info["why"] = "no direction separates the repeat from the working continuation"
        return None, info
    s0, h0 = PR.selectivity(lm, sel), PR.selectivity(lm, held)

    def layer_site(i: int) -> Optional[E.Site]:
        ns = sm.nscore.get(i)
        if ns is None or i + 1 not in sm.dirs:
            return None
        k = min(max_neurons, max(8, ns.numel() // 32))
        cols = [j for j in torch.argsort(ns, descending=True)[:k].tolist() if float(ns[j]) > 0]
        return E.Site(i, "mlp", sm.dirs[i + 1], cols) if cols else None

    rows = []
    for i in range(lm.n_layers - 1):                    # the last layer writes the token itself: see above
        site = layer_site(i)
        if site is None:
            continue
        with E.applied(lm, E.project(lm, [site], alpha)):
            r = PR.compare(s0, PR.selectivity(lm, sel))
        ratio = (-r["repeat_delta"] / max(1e-4, -r["working_delta"])) if r["repeat_delta"] < 0 else 0.0
        rows.append({"layer": i, "repeat_delta": r["repeat_delta"], "working_delta": r["working_delta"],
                     "ratio": round(ratio, 2), "site": site})
    ok = sorted((r for r in rows if r["repeat_delta"] < 0 and r["ratio"] >= min_ratio),
                key=lambda r: (-r["ratio"], r["repeat_delta"]))[:keep]
    info["screen"] = [{k: v for k, v in r.items() if k != "site"} for r in rows]
    if not ok:
        info["why"] = f"no single layer lowers the repeat at least {min_ratio:g}x more than the working text"
        return None, info
    sites = [r["site"] for r in ok]
    layers = [r["layer"] for r in ok]
    info["layers"] = layers
    with E.applied(lm, E.project(lm, sites, alpha)):
        hc = PR.compare(h0, PR.selectivity(lm, held))
    info["held"] = hc
    if not (hc["selective"] and hc["repeat_delta"] < 0 and hc["working_delta"] > hc["repeat_delta"] / min_ratio):
        info["why"] = (f"layers {layers} were selective on the pairs that chose them but not on the held-out "
                       f"pairs (repeat {hc['repeat_delta']:+.3f}, working {hc['working_delta']:+.3f} nats/token)")
        return None, info
    label = "screened sparse continuation direction on layers " + ", ".join(str(i) for i in layers)
    ev = (f"{len(sites)} layer(s) chosen by a selectivity screen on {len(sel)} pairs and confirmed on {len(held)} "
          f"held-out pairs: at alpha {alpha:g} the repeat falls {hc['repeat_delta']:+.3f} and the working text "
          f"{hc['working_delta']:+.3f} nats/token; {sum(len(s.cols) for s in sites)} neurons")
    ladder = [(f"alpha {a:g}", lambda a=a: E.project(lm, sites, a, label=f"{label}, alpha {a:g}", source=label))
              for a in LOCAL_LADDER]
    return Candidate(label, ev, ladder, None, scope="sparse"), info


GATE_LADDER = (0.5, 1.0, 2.0, 4.0)       # x the strength that cancels the failure's gap at the failing positions


def _states_of(replies: list, bad: bool) -> list[S.State]:
    return [S.State(r.name, r.ids, [r.ids.shape[1] - 1], bad) for r in replies]


def _full_texts(replies: list) -> list[torch.Tensor]:
    return [torch.cat([r.ids, torch.tensor([r.gen], dtype=torch.long)], 1) if r.gen else r.ids for r in replies]


def gate_candidates(lm: LM, res: Any, sm: Optional[S.SiteMap], base: R.Baseline, prs: list) -> list[Candidate]:
    """Deep mode: one gated neuron (edits.gate) in the MLP that writes most of the failure. It reads
    what separates the failing states from the ones that must not change, fires only above every
    silent text, and writes against the failure direction.

    Over-refusal: the states that must not change are the harmful requests the model rightly
    refuses, so the neuron reads the false alarm (over-refused minus rightly refused) instead of
    the refusal itself, and stays off on real harm."""
    c = res.contrast
    if sm is None:
        return []
    mlps = [s for s in sm.sites if s["part"] == "mlp"] or sm.sites
    top = mlps[0]["layer"]
    if not all(getattr(lm.mlp(top), n, None) is not None for n in ("gate_proj", "up_proj")):
        return []
    ref = _full_texts(base.replies)[:8]                 # typical inputs, for the bias direction
    if c.kind == "loop" and len(prs) >= 2:
        sts = PR.states(prs)
        bad, good = [s for s in sts if s.bad], [s for s in sts if not s.bad]
    else:
        bad = [st for st in S.from_contrast(c) if st.bad]
        good = [st for st in S.from_contrast(c) if not st.bad]
    # Silent: every position of the healthy regression replies, and the working prompts where they
    # decide (the good states themselves, inside gate_plan).
    silent = _full_texts([r for r in base.replies if r.kind != "safety"])
    how = "failing vs working, silent on the working prompts and every regression reply"
    if c.kind == "refusal":
        harm = [r for r in base.replies if r.kind == "safety" and r.refused]
        if harm:
            good = _states_of(harm, False)
            silent = _full_texts(harm)
            how = ("the over-refused prompt vs the harmful requests it rightly refuses (the false alarm), silent on "
                   "those requests")
    if not bad or not good or not ref:
        return []
    # The writing site first, then earlier layers: the trigger is often readable before the layer
    # that writes the failure (Qwen2.5-0.5B over-refusal: layer 20 writes it, layer 14 reads it cleanly).
    plan, layer = {"ok": False}, top
    for layer in dict.fromkeys(max(0, top - k) for k in (0, 6, 12)):
        plan = S.gate_plan(lm, layer, bad, good, silent, ref)
        if plan["ok"]:
            break
    if not plan["ok"]:
        return []
    neuron = S.quiet_neurons(lm, layer, ref[:6], k=1)[0]
    w = sm.dirs[layer + 1]

    def gap() -> float:
        def mean(states):
            v = [lm.seq_capture(s.ids, depths=[layer + 1], positions=s.positions)["resid"][layer + 1].mean(0)
                 for s in states]
            return torch.stack(v).mean(0)
        g = [st for st in (S.from_contrast(c) if c.kind != "loop" or len(prs) < 2 else PR.states(prs)) if not st.bad]
        return float((mean(bad) - mean(g)) @ w)
    s0 = max(gap(), 1e-3) / max(plan["margin"], 1e-6)       # cancels the gap at the mean firing margin
    label = f"gated neuron {layer}.{neuron}"
    ev = (f"reads {how}; fires on {plan['fire_bad']:.0%} of the failing positions and "
          f"{plan['fire_off']:.1%} of the silent ones; writes against the failure direction after layer {layer} "
          + ("(threshold 0, no bias needed)" if plan["mean_dir"] is None else
             f"(bias from the mean input direction, spread {plan['spread']:.0%})"))
    ladder = [(f"x{k:g}", lambda k=k: E.gate(lm, layer, neuron, plan["read"], w, threshold=plan["threshold"],
                                             up_scale=plan["up_scale"], mean_dir=plan["mean_dir"],
                                             mean_proj=plan["mean_proj"], strength=k * s0,
                                             label=f"{label} x{k:g}", source=label))
              for k in GATE_LADDER]
    return [Candidate(label, ev, ladder, None, scope="gated")]


def _known_vector(res: Any, name: str) -> Optional[tuple[torch.Tensor, dict]]:
    kv = getattr(res, "known_vectors", None)
    ka = (res.known_ablation or {}).get(name)
    if kv is None or not ka or name not in kv.by_name or ka.get("fixes", 0) <= 0:
        return None
    return kv.by_name[name][ka["depth"]], ka


def candidates(lm: LM, res: Any, *, mode: str = "fast", sitemap: Optional[S.SiteMap] = None) -> list[Candidate]:
    """The diagnosis's candidates (both modes); in deep mode also the local ones on `sitemap`,
    and every candidate ordered smallest edit first (components, sparse, local, global)."""
    out = _diagnosis_candidates(lm, res)
    if mode == "deep" and sitemap is not None:
        out += _local_candidates(lm, sitemap, res.contrast.failing[0].decision_ids)
        out.sort(key=lambda c: _SCOPE_ORDER.get(c.scope, 9))
    return out


def _diagnosis_candidates(lm: LM, res: Any) -> list[Candidate]:
    c = res.contrast
    probe_ids = c.failing[0].decision_ids
    out: list[Candidate] = []
    kind = c.kind

    def contrast() -> None:
        d = res.directions
        if d is None:
            return
        chosen = next((s for s in (res.selection.scores if res.selection else []) if s["depth"] == d.best_depth), {})
        cz = res.causal
        ev = (f"separates failing from passing runs after layer {d.best_depth - 1}; removing it moves the decision "
              f"{chosen.get('fixes', 0):.0%} of the way, generation fixes {cz.ablate_fixed if cz else 0}/"
              f"{cz.ablate_total if cz else 0}")
        out.append(_dir_candidate(lm, f"contrast direction after layer {d.best_depth - 1}", ev, d.unit(), probe_ids))

    def known(name: str, what: str) -> None:
        got = _known_vector(res, name)
        if got:
            v, ka = got
            out.append(_dir_candidate(lm, f"generic {what} direction after layer {ka['layer']}",
                                      f"removing it moves the decision {ka['fixes']:.0%} of the way, flips "
                                      f"{ka['flipped']}/{ka['of']}" + (" (beats shuffled-label directions)"
                                                                     if ka.get("specific") else ""), v, probe_ids))

    if kind == "refusal":
        known("refusal", "refusal")
        contrast()
    elif kind == "language":
        lp = res.lang or {}
        st = lp.get("steer") or {}
        if lp.get("expected") and lp.get("answered") and st.get("depth"):
            from ..deep import langprobe as LP
            probe = LP.calibrate(lm, log=None)
            v = probe.direction(st["depth"], lp["answered"], lp["expected"])
            out.append(_dir_candidate(lm, f"calibrated {LP.NAMES.get(lp['answered'], lp['answered'])}-minus-"
                                      f"{LP.NAMES.get(lp['expected'], lp['expected'])} direction after layer "
                                      f"{st['layer']}",
                                      f"removing it restores the language in {st.get('ablate_fixed', 0)}/"
                                      f"{st.get('of', 0)} (unrelated pair {st.get('null_fixed', 0)}/{st.get('of', 0)})",
                                      v, probe_ids))
        contrast()
    elif kind == "blank":
        loc = res.loc
        if loc and loc.neurons:
            top = [(n["layer"], n["neuron"]) for n in loc.neurons]
            out.append(_neuron_candidate(lm, f"neuron {top[0][0]}.{top[0][1]}",
                                         "the top neuron writing the end-of-turn direction", top[:1], probe_ids))
            if len(top) >= 3:
                out.append(_neuron_candidate(lm, "top 3 neurons " + ", ".join(f"{i}.{j}" for i, j in top[:3]),
                                             "the neurons writing most of the direction", top[:3], probe_ids))
        contrast()
        known("end_of_turn", "end-of-turn")
    elif kind == "loop":
        lb = res.loop or {}
        m = lb.get("mlp") or {}
        if m.get("chosen_k") and (m.get("supported") or m.get("decides_tie")):
            ns = [(n["layer"], n["neuron"]) for n in m["top_neurons"][:m["chosen_k"]]]
            out.append(_neuron_candidate(lm, f"{len(ns)} loop neurons", "attribution-ranked neurons whose removal "
                                         f"stops the loop ({m.get('gen_fixed', 0)}/{m.get('gen_total', 0)}, random "
                                         f"{m.get('null_gen_fixed', 0)})", ns, probe_ids))
        ab = (res.copy or {}).get("ablation") or {}
        if (res.copy or {}).get("specific") and ab.get("heads"):
            hs = [tuple(h) for h in ab["heads"]]
            out.append(_head_candidate(lm, f"{len(hs)} copy heads", "calibrated copy heads whose removal stops the "
                                       "loop beyond random heads", hs, probe_ids))
        if (lb.get("attractor") or {}).get("supported"):
            contrast()
            known("repetition", "repetition")
    else:
        contrast()
    return out


def not_indicated(res: Any) -> Optional[tuple[str, str]]:
    """(why, what to do) when the diagnosis says the weights are not where to fix it."""
    v = (res.report.verdict or {}).get("verdict")
    cut = next((f for f in res.report.findings if f.code == "DEEP_CUT_SHORT"), None)
    if cut is not None:
        return ("the failure starts after the token limit: the diagnosis saw a healthy reply at --max-new",
                cut.fix or "rerun with a larger --max-new (128 or more: loops often start after 50-100 tokens)")
    if v in ("not_weight_level", "quantisation") or any(f.code == "DEEP_NOT_REPRODUCED" for f in res.report.findings):
        return ("the failure does not reproduce on full precision, or comes from quantisation",
                "fix the served copy: requantise the named tensors (`wsr qbisect`) or the whole file at a higher "
                "precision")
    sup = set((res.loop or {}).get("supported") or [])
    if "positional (RoPE)" in sup:
        return ("positions are encoded wrongly (RoPE): a runtime or config defect, not a learned behaviour",
                "compute RoPE in float32 and check the rope settings against the model card")
    if res.contrast and res.contrast.kind == "loop" and sup and sup <= {"greedy decoding"}:
        return ("greedy decoding keeps this loop and sampling escapes it: a decoding setting",
                "decode with temperature 0.6-0.8 and top_p 0.9, or a DRY / mild repetition penalty")
    return None


# -- search ----------------------------------------------------------------------------------------

def _screen(lm: LM, ts: list) -> dict[str, Any]:
    ms = [t.metric(lm.run(t.ex.decision_ids, capture=()).logits) for t in ts]
    return {"flipped": sum(m < 0 for m in ms), "of": len(ms), "metrics": [round(m, 3) for m in ms]}


def _judge_jobs(before_fail: list, before_pass: list) -> list[tuple[str, list[dict]]]:
    return [(e.name + "+patch", e.messages) for e in list(before_fail) + list(before_pass)]


def reference(clf: Classifier, before_fail: list, before_pass: list) -> list:
    """The unpatched model on exactly the job list `_judge` will run under each patch. The same list
    runs in the same batch chunks, so a reply can differ between the two only because of the patch."""
    return clf.run_many(_judge_jobs(before_fail, before_pass))


def _judge(kind: str, before_fail: list, before_pass: list, clf: Classifier, usual: set[str],
           ref: Optional[list] = None) -> dict[str, Any]:
    """Run the (patched) model on failing and working examples. A failing one is fixed when it loses
    the failure and gains nothing the working ones do not already show; a working one breaks when
    it gains a failure. With `ref` (the unpatched run of the same list, see `reference`), "before" is
    that run: an example whose reference does not fail cannot count as fixed, and a working example
    breaks only relative to its reference."""
    after = clf.run_many(_judge_jobs(before_fail, before_pass))
    nf = len(before_fail)
    rf = ref[:nf] if ref is not None else list(before_fail)
    rp = ref[nf:] if ref is not None else list(before_pass)
    fail = [(e.name, a) for e, a, r in zip(before_fail, after[:nf], rf) if (kind in r.groups or kind == "other")]
    keep = [(r, a) for r, a in zip(rp, after[nf:])]
    fixed = [n for n, e in fail if kind not in e.groups and e.groups <= usual]
    broke = [e0.name.removesuffix("+patch") for e0, e in keep if e.groups - e0.groups]
    return {"fixed": len(fixed), "of": len(fail), "fixed_names": fixed, "failing_names": [n for n, _ in fail],
            "passing_kept": len(keep) - len(broke),
            "passing": len(keep), "broke": broke,
            "samples": [{"prompt": n, "after": e.text[:120], "groups": sorted(e.groups)} for n, e in fail[:4]]}


def _agreement(kind: str, fail: list, keep: list, ref: list) -> dict[str, int]:
    """How many reference replies keep the label they had when the set was built (expected: all)."""
    nf = len(fail)
    same_f = sum((kind in r.groups or kind == "other") for r in ref[:nf])
    same_p = sum(not r.groups for r in ref[nf:])
    orig = [(kind in r.groups or kind == "other") for e, r in zip(fail, ref[:nf]) if getattr(e, "name", "") == "original"]
    return {"failing": same_f, "of_failing": nf, "passing": same_p, "of_passing": len(keep),
            **({"original_fails": orig[0]} if orig else {})}


LONG_KINDS = ("loop",)          # failures that build up over a reply: a later start can pass the token limit
MAX_REF_NEW = 4096


def _longer_reference(kind: str, clf: Classifier, fail: list, keep: list, agrees: dict[str, int],
                      say: Callable[[str], None], what: str):
    """When fewer than 2/3 of the failing examples fail again in their unedited reference run, and the failure
    builds up over the reply (a loop), the cause is usually the token limit: greedy replies diverge a little
    between batch layouts and the loop starts later, past the cap. Run the reference once more at twice the
    limit; keep it (and judge every edit at that length) only if more examples fail there. Returns
    (classifier, reference, agreement at the cap, agreement at the longer length) or None."""
    nf = agrees.get("of_failing", 0)
    if kind not in LONG_KINDS or not nf or _share_ok(agrees.get("failing", 0), nf) or clf.max_new >= MAX_REF_NEW:
        return None
    long_clf = Classifier(clf.lm, expect_lang=clf.expect_lang, max_new=min(2 * clf.max_new, MAX_REF_NEW),
                          template_kwargs=clf.template_kwargs, expect_refuse=getattr(clf, "expect_refuse", False))
    ref = reference(long_clf, fail, keep)
    got = _agreement(kind, fail, keep, ref)
    say(f"reference runs ({what}): {agrees.get('failing', 0)}/{nf} failing examples fail again at "
        f"{clf.max_new} tokens, {got['failing']}/{nf} at {long_clf.max_new}"
        + ("; judging at the longer length" if got["failing"] > agrees.get("failing", 0) else "; keeping the cap"))
    if got["failing"] <= agrees.get("failing", 0):
        return None
    return long_clf, ref, {**agrees, "max_new": clf.max_new}, {**got, "max_new": long_clf.max_new}


def _prepared_reference(kind: str, clf: Classifier, fail: list, keep: list, say: Callable[[str], None],
                        what: str) -> Reference:
    """The reference run at the cap, redone at a longer length when the cap hides a loop (see below)."""
    replies = reference(clf, fail, keep)
    ref = Reference(replies, clf, _agreement(kind, fail, keep, replies))
    longer = _longer_reference(kind, clf, fail, keep, ref.agrees, say, what)
    if longer is not None:
        long_clf, long_replies, at_cap, agrees = longer
        ref = Reference(long_replies, long_clf, agrees, agrees_at_cap=at_cap)
    return ref


def _usual(c: Any) -> set[str]:
    return set().union(*(e.groups for e in c.passing)) - {c.kind} if c.passing else set()


def _target(lm: LM, c: Any, clf: Classifier) -> dict[str, Any]:
    # Judged against the model's own behaviour (see _judge), at the length its reference was run with.
    ref = c.reference
    t = _judge(c.kind, c.failing, c.passing, ref.clf if ref else clf, _usual(c), ref=ref.replies if ref else None)
    t["original_fixed"] = "original" in t.pop("fixed_names")
    # Only an example whose unedited reference run fails can count as fixed. When the original's own reference
    # run does not show the failure (long greedy replies can diverge between batch layouts), its fix cannot be
    # measured here: say so instead of reporting it as still failing, which no edit could ever change.
    t["original_measured"] = "original" in t.pop("failing_names")
    return t


def _drift(reg: dict[str, Any]) -> str:
    if reg.get("mean_delta_nll") is None:
        return "drift not measured (no healthy original reply)"
    return f"drift {reg['mean_delta_nll']:.3f} nats/token, top-token agreement {reg['top1_agreement']:.0%}"


def _share_ok(got: int, of: int) -> bool:
    return got * 3 >= 2 * of


def _target_ok(t: dict[str, Any]) -> list[str]:
    out = []
    if not t["original_fixed"] and t.get("original_measured", True):
        out.append("the original prompt still fails")
    if not t["of"]:
        out.append("no failing prompt shows the failure in its unedited reference run, so no fix can be measured")
    elif not _share_ok(t["fixed"], t["of"]):
        out.append(f"fixes only {t['fixed']}/{t['of']} failing prompts")
    if t["passing"] and not _share_ok(t["passing_kept"], t["passing"]):
        out.append(f"breaks {t['passing'] - t['passing_kept']}/{t['passing']} working prompts")
    return out


# -- serving precision -----------------------------------------------------------------------------

_DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def serve_dtype(lm: LM, want: str = "auto") -> Optional[torch.dtype]:
    """The precision the patched model will be served in: the checkpoint's 16-bit type, else bf16
    (its 8-bit mantissa rounds hardest); None = do not check."""
    if want == "none":
        return None
    if want != "auto":
        return _DT[want]
    ck = _DT.get(str(getattr(lm, "checkpoint_dtype", "") or "").replace("torch.", ""))
    return ck if ck in (torch.bfloat16, torch.float16) else torch.bfloat16


@torch.no_grad()
def _round_touched(lm: LM, patch: E.Patch, dt: torch.dtype, chunk: int = 4096) -> float:
    """Inside `applied(patch)`: round every touched tensor to `dt` (as saving the patched model in
    that precision does) and return how much of the patch's change is lost: the norm of
    (round(W0 + delta) - round(W0) - delta) over the norm of delta."""
    err = tot = 0.0
    for d in patch.deltas:
        p = E._param(lm, d.name)
        U, V = d.U.to(p.device), d.V.to(p.device)
        for r0 in range(0, p.shape[0], chunk):
            dense = U[r0:r0 + chunk] @ V.T
            if p.dim() == 1:
                dense = dense[:, 0]
            cur = p[r0:r0 + chunk].float()
            w0 = cur - dense
            served = cur.to(dt).float()
            err += float(((served - w0.to(dt).float()) - dense).pow(2).sum())
            tot += float(dense.pow(2).sum())
            p[r0:r0 + chunk].copy_(served.to(p.dtype))
    E._bump(lm)
    return math.sqrt(err / tot) if tot > 0 else 0.0


# -- sampled runs ----------------------------------------------------------------------------------

def sampling_params(lm: LM) -> dict[str, Any]:
    """The model's own sampling settings (generation_config), else temperature 0.7, top_p 0.9."""
    gc = getattr(lm.model, "generation_config", None)
    t = getattr(gc, "temperature", None)
    if not getattr(gc, "do_sample", False) or not t:
        return {"temperature": 0.7, "top_p": 0.9}
    out = {"temperature": float(t), "top_p": float(getattr(gc, "top_p", None) or 1.0)}
    if getattr(gc, "top_k", None):
        out["top_k"] = int(gc.top_k)
    return out


def _sampled(lm: LM, c: Any, clf: Classifier, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """n samples per prompt (failing first, then working), seeded per (prompt, sample) so the
    original and the patched model draw from the same random stream."""
    rows = []
    prompts = ([(e, True) for e in c.failing[:cfg["prompts"]]] +
               [(e, False) for e in c.passing[:max(1, cfg["prompts"] // 2)]])
    for k, (e, failing) in enumerate(prompts):
        ids = lm.encode(e.messages, **clf.template_kwargs)
        for s in range(cfg["n"]):
            torch.manual_seed(cfg["seed"] + 1000 * k + s)
            gen = lm.generate(ids, cfg["max_new"], do_sample=True, **cfg["params"])
            ended = bool(gen) and gen[-1] in lm.eot_ids
            text = lm.tok.decode(gen[:-1] if ended else gen, skip_special_tokens=True)
            groups = clf.groups_of(text, gen, request=e.messages[-1].get("content") if e.messages else None)
            fail = (c.kind in groups) if failing else bool(groups - _usual(c))
            rows.append({"prompt": e.name, "failing_prompt": failing, "sample": s, "fails": fail,
                         "tokens": len(gen), "tail": text[-80:]})
    return rows


def _paired(before: list[dict], after: list[dict]) -> dict[str, Any]:
    fb = [(b, a) for b, a in zip(before, after) if b["failing_prompt"]]
    pb = [(b, a) for b, a in zip(before, after) if not b["failing_prompt"]]
    d = [int(b["fails"]) - int(a["fails"]) for b, a in fb]
    n = len(d)
    gain = sum(d) / n if n else 0.0
    sd = math.sqrt(sum((x - gain) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {"samples": n, "failure_rate_before": round(sum(b["fails"] for b, _ in fb) / n, 3) if n else None,
            "failure_rate_after": round(sum(a["fails"] for _, a in fb) / n, 3) if n else None,
            "gain": round(gain, 3), "se": round(se, 3), "rescues": sum(x > 0 for x in d),
            "losses": sum(x < 0 for x in d), "working_samples": len(pb),
            "working_broken": sum(a["fails"] and not b["fails"] for b, a in pb)}


# -- verification ----------------------------------------------------------------------------------

@dataclass
class _Ctx:
    lm: LM
    res: Any
    clf: Classifier
    base: R.Baseline
    budget: R.Budget
    template_kwargs: Optional[dict]
    mode: str = "fast"
    serve: Optional[torch.dtype] = None
    fresh: Optional[dict] = None            # {"failing": [...], "passing": [...], "consulted": int}
    sampled: Optional[dict] = None          # {"cfg": ..., "before": [...]}
    pairs: list = field(default_factory=list)
    sel0: Optional[dict] = None
    canary: Optional[list] = None           # the unedited model's canary rows (surgery/canary.py)
    harm_extra: Optional[list] = None       # labelled practice items that calibrate the harm probe (wsr program)


def _verify(x: _Ctx, patch: Optional[E.Patch], step: dict[str, Any]) -> list[str]:
    """Every check of the mode, cheapest first, with the patch (or steering hooks) applied by the caller. Fills
    `step`; returns the problems (empty: it passes). patch None = steering: no weights to round."""
    lm, c = x.lm, x.res.contrast
    tg = _target(lm, c, x.clf)
    step["target"] = tg
    bad = _target_ok(tg)
    if bad:
        return bad
    reg = R.compare(lm, x.base, template_kwargs=x.template_kwargs)
    step["regression"] = {k: v for k, v in reg.items() if k != "rows"}
    step["_regression_rows"] = reg["rows"]
    probs = x.budget.problems(reg)
    if probs:
        return ["fixes it, but " + "; ".join(probs)]
    if x.canary is not None and getattr(x.budget, "canary_max_lost", None) is not None:
        from . import canary as CN
        cn = CN.compare(x.canary, CN.run(lm, x.template_kwargs, original=False), x.budget.canary_max_lost)
        step["canary"] = cn
        if not cn["holds"]:
            return ["fixes it, but " + "; ".join(cn["problems"])]
    if x.mode == "deep" and x.fresh is not None:
        if x.fresh["failing"]:
            x.fresh["consulted"] += 1
            fref = x.fresh.get("reference")
            fr = _judge(c.kind, x.fresh["failing"], x.fresh["passing"], fref.clf if fref else x.clf, _usual(c),
                        ref=fref.replies if fref else None)
            fr.pop("fixed_names")
            fr.pop("failing_names", None)
            step["fresh"] = fr
            fb = []
            if not _share_ok(fr["fixed"], fr["of"]):
                fb.append(f"fixes only {fr['fixed']}/{fr['of']} fresh failing variants")
            if fr["passing"] and not _share_ok(fr["passing_kept"], fr["passing"]):
                fb.append(f"breaks {fr['passing'] - fr['passing_kept']}/{fr['passing']} fresh working variants")
            if fb:
                return ["fixes the contrast prompts, but " + "; ".join(fb)]
        else:
            step["fresh"] = {"note": "no fresh variant reproduces the failure, so there was nothing to confirm on"}
    if x.sampled is not None:
        after = _sampled(lm, c, x.clf, x.sampled["cfg"])
        pr = _paired(x.sampled["before"], after)
        step["sampled"] = pr
        sp = []
        if pr["samples"] and not (pr["gain"] > 0 and pr["gain"] > 2 * pr["se"]):
            sp.append(f"sampled failure rate {pr['failure_rate_before']:.0%} -> {pr['failure_rate_after']:.0%} "
                      f"(gain {pr['gain']:+.2f} +- {pr['se']:.2f}): not clearly lower")
        if pr["working_samples"] and pr["working_broken"] * 3 > pr["working_samples"]:
            sp.append(f"working prompts fail in {pr['working_broken']}/{pr['working_samples']} sampled runs")
        if sp:
            return ["passes the greedy checks, but " + "; ".join(sp)]
    # Serving precision last: it rounds the touched tensors in place (restored by `applied`).
    if patch is None:
        step["serving"] = {"note": "steering adds a vector at inference; no weights change, nothing to round"}
    elif x.serve is not None:
        loaded = next(lm.model.parameters()).dtype
        name = str(x.serve).replace("torch.", "")
        if loaded == x.serve:
            step["serving"] = {"dtype": name, "note": "the model was loaded in this precision: every check above "
                                                      "already ran on the rounded weights"}
        else:
            lost = _round_touched(lm, patch, x.serve)
            ts = _target(lm, c, x.clf)
            step["serving"] = {"dtype": name, "delta_lost": round(lost, 4), "target": ts}
            sb = _target_ok(ts)
            if sb:
                return [f"in {name} the patch loses {lost:.0%} of its change to rounding and " + "; ".join(sb)]
    return []


def _control(x: _Ctx, patch: E.Patch, tg: dict[str, Any]) -> dict[str, Any]:
    """A random edit of the same kind and size: reported, never used to reject."""
    try:
        ctrl = E.control(x.lm, patch)
    except (ValueError, KeyError) as e:
        return {"note": f"no control for this patch: {e}"}
    with E.applied(x.lm, ctrl):
        t = _target(x.lm, x.res.contrast, x.clf)
        sel = PR.selectivity(x.lm, x.pairs) if x.pairs else None
    out = {"label": ctrl.label, "fixed": t["fixed"], "of": t["of"], "passing_kept": t["passing_kept"],
           "passing": t["passing"], "samples": t["samples"][:2]}
    if sel and x.sel0:
        out["continuations"] = PR.compare(x.sel0, sel)
    if t["fixed"] * 2 < max(1, tg["fixed"]):
        out["specific"] = True
        out["verdict"] = (f"specific: the random edit fixes {t['fixed']}/{t['of']} against the patch's "
                          f"{tg['fixed']}/{tg['of']}")
    else:
        out["specific"] = False
        out["verdict"] = (f"not specific: a random edit of the same size also fixes {t['fixed']}/{t['of']} - the "
                          "failure is fragile here, and the patch works without being tied to what the diagnosis "
                          "found")
    return out


def _prepare(lm: LM, res: Any, *, mode: str, clf: Classifier, confirm: tuple[str, ...], sampled: Optional[dict],
             say: Callable[[str], None]) -> tuple:
    """On the unpatched model: the fresh variants (deep), the sampled baseline, the matched
    continuations (loops) and, in deep mode, where the failure is written."""
    c = res.contrast
    fresh = None
    if mode == "deep":
        msgs = c.tried[0].messages
        used = {e.messages[-1].get("content", "") for e in c.tried if e.messages}
        cand = fresh_variants(msgs, used)[:10] + [(f"user_confirm_{i}", _with_last_user(msgs, t))
                                                  for i, t in enumerate(confirm)]
        fresh = {"failing": [], "passing": [], "consulted": 0, "tried": len(cand)}
        for e in clf.run_many(cand):
            failing = c.kind in e.groups if c.kind != "other" else bool(e.groups)
            if failing:
                fresh["failing"].append(e)
            elif not e.groups:
                fresh["passing"].append(e)
        say(f"fresh variants: {len(fresh['failing'])} failing, {len(fresh['passing'])} working of {len(cand)}")
        fresh["reference"] = _prepared_reference(c.kind, clf, fresh["failing"], fresh["passing"], say,
                                                 "fresh variants")
    # The unpatched reference for the target check (see `reference`): same list, same chunks as the
    # patched runs, so noise from batching cannot count as a fix.
    c.reference = _prepared_reference(c.kind, clf, c.failing, c.passing, say, "target prompts")
    sb = None
    if sampled:
        cfg = {"n": sampled.get("n", 4), "max_new": sampled.get("max_new", 512), "seed": sampled.get("seed", 0),
               "prompts": sampled.get("prompts", 6), "params": sampled.get("params") or sampling_params(lm)}
        say(f"sampled baseline: {cfg['n']} samples per prompt, {cfg['params']}, up to {cfg['max_new']} tokens")
        sb = {"cfg": cfg, "before": _sampled(lm, c, clf, cfg)}
    prs = PR.build(lm, c)
    sel0 = PR.selectivity(lm, prs) if prs else None
    if prs:
        say(f"matched continuations: {len(prs)} pair(s) (the repeat vs a working continuation of the same prompt)")
    sm = None
    if mode == "deep":
        use_pairs = len(prs) >= 2
        sm = S.locate(lm, PR.states(prs) if use_pairs else S.from_contrast(c),
                      source="continuations" if use_pairs else "contrast")
        if sm:
            say("sites: " + sm.describe(False))
    return fresh, sb, prs, sel0, sm


def _canary0(lm: LM, budget: R.Budget, template_kwargs: Optional[dict]) -> Optional[list]:
    """The unedited model on the canary (cached with its batch plan), unless the budget turns it off."""
    if getattr(budget, "canary_max_lost", None) is None:
        return None
    from . import canary as CN

    return CN.run(lm, template_kwargs, original=True)


def _mode_name(mode: str, sampled: Optional[dict]) -> str:
    return mode + ("+sampled" if sampled and mode == "deep" else "")


def _site_summary(sm: Optional[S.SiteMap]) -> Optional[dict]:
    if sm is None:
        return None
    return {"source": sm.source, "chosen": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
                                            for s in sm.sites],
            "neurons": {str(i): len(v) for i, v in sm.neurons.items()},
            "neuron_share": {str(i): round(v, 3) for i, v in sm.neuron_share.items()}}


def repair(lm: LM, res: Any, *, expect_lang: Optional[str] = None, max_new: int = 48,
           template_kwargs: Optional[dict] = None, budget: Optional[R.Budget] = None, keep: tuple[str, ...] = (),
           mode: str = "fast", confirm: tuple[str, ...] = (), sampled: Optional[dict] = None,
           serve: str = "auto", playbook: bool = True, steering: str = "fallback",
           log: Optional[Callable[[str], None]] = print, harm_extra: Optional[list] = None) -> Repair:
    """Search the candidates for the first step passing every check of `mode` ("fast" | "deep");
    `sampled` (deep only): {"n", "max_new", "prompts", "params"} for paired sampled runs.

    steering: "fallback" (default) tries activation steering against the diagnosed direction when no weight edit
    passes; "only" tries steering and no weight edit (for comparing the two); "off" never steers. A steering that
    passes the same checks returns status "steered" with `Repair.steering` set (no patch). For refusal failures the
    first steering tried is probe-gated (harmprobe.py), when the diagnosis found the model's harm reading usable;
    `harm_extra` (labelled practice items) calibrates that probe on the user's own kind of request as well."""
    if steering not in ("fallback", "only", "off"):
        raise ValueError(f"steering must be fallback, only or off, not {steering!r}")
    t0 = time.perf_counter()
    say = log or (lambda s: None)
    budget = budget or R.Budget()
    c = res.contrast
    entry = ENTRIES.get(c.kind, GENERIC) if c else GENERIC
    mname = _mode_name(mode, sampled)
    why = not_indicated(res)
    if why:
        return Repair("not_indicated", entry.name, f"surgery not indicated: {why[0]}", next_step=why[1],
                      seconds=time.perf_counter() - t0, mode=mname)
    clf = Classifier(lm, expect_lang=expect_lang, max_new=max_new, template_kwargs=template_kwargs,
                     expect_refuse=bool(res.contrast) and res.contrast.kind == "complied")
    fresh, sb, prs, sel0, sm = _prepare(lm, res, mode=mode, clf=clf, confirm=confirm,
                                        sampled=sampled if mode == "deep" else None, say=say)
    checks: dict[str, Any] = {}
    # How often the unpatched reference run kept each example's label (expected: always). A difference
    # is batching-level noise on a near tie; the checks count against the reference either way.
    tref, fref = res.contrast.reference, (fresh or {}).get("reference")
    ra = {"target": tref.describe() if tref else None, "fresh": fref.describe() if fref else None}
    checks["reference_runs"] = {k: v for k, v in ra.items() if v is not None}
    if prs:
        checks["continuations"] = {"pairs": len(prs), "before": {k: round(v, 4) for k, v in sel0.items()}}
    if sm is not None:
        checks["sites"] = _site_summary(sm)
    cands = candidates(lm, res, mode=mode, sitemap=sm) if playbook and steering != "only" else []
    if mode == "deep" and len(prs) >= 4 and playbook and steering != "only":
        sc, sinfo = screened_candidate(lm, prs)
        checks["layer_screen"] = sinfo
        say("layer screen: " + (f"layers {sinfo['layers']} confirmed on held-out pairs" if sc else sinfo["why"]))
        if sc:
            cands = sorted([sc] + cands, key=lambda cd: _SCOPE_ORDER.get(cd.scope, 9))
    if not cands:
        base = R.baseline(lm, keep=keep, keep_lang=expect_lang or "en", template_kwargs=template_kwargs)
        x = _Ctx(lm, res, clf, base, budget, template_kwargs, mode, serve_dtype(lm, serve), fresh, sb, prs, sel0,
             canary=_canary0(lm, budget, template_kwargs), harm_extra=harm_extra)
        trials: list[dict] = []
        if playbook and steering != "off":
            got = _try_steering(x, A.targets(lm, c), trials, say)
            if got is not None:
                return _steered(entry, *got, trials, checks, mname, x, time.perf_counter() - t0)
        return Repair("could_not_fix", entry.name, "the diagnosis named no direction or component to edit"
                      + ("; steering against the diagnosed direction did not pass either" if trials else "")
                      if playbook else "the rule-based playbook was skipped (--agent-only)",
                      trials=trials, next_step=entry.could_not_fix, seconds=time.perf_counter() - t0, mode=mname,
                      checks=checks, ctx=x, sitemap=sm)
    ts = A.targets(lm, c)
    say(f"surgery ({entry.name}, {mname} mode): {len(cands)} candidate(s); regression baseline on "
        f"{len(R.SUITE) + len(keep)} prompts")
    base = R.baseline(lm, keep=keep, keep_lang=expect_lang or "en", template_kwargs=template_kwargs)
    if mode == "deep":
        # The gated neuron needs the regression replies (where it must stay silent).
        g = gate_candidates(lm, res, sm, base, prs)
        if g:
            cands = sorted(cands + g, key=lambda cd: _SCOPE_ORDER.get(cd.scope, 9))
            say(f"  + {g[0].label}: {g[0].evidence}")
    x = _Ctx(lm, res, clf, base, budget, template_kwargs, mode, serve_dtype(lm, serve), fresh, sb, prs, sel0,
             canary=_canary0(lm, budget, template_kwargs), harm_extra=harm_extra)
    trials: list[dict] = []
    for cand in cands:
        row: dict[str, Any] = {"candidate": cand.label, "scope": cand.scope, "evidence": cand.evidence, "steps": []}
        trials.append(row)
        if cand.check:
            kl, effect = cand.check()
            row["matches_hook_kl"], row["hook_effect_kl"] = round(kl, 6), round(effect, 6)
            # An edit that is not equivalent (post-norm layers) misses by the size of the effect itself.
            if kl > max(1e-3, 0.05 * effect):
                row["skipped"] = (f"the weight edit does not reproduce the tested intervention (KL {kl:.4f} vs the "
                                  f"intervention's own effect {effect:.4f}): this "
                                  "architecture does not add these writes straight to the residual stream")
                say(f"  {cand.label}: skipped, {row['skipped']}")
                continue
        for k, (strength, build) in enumerate(cand.ladder):
            patch = build()
            step: dict[str, Any] = {"strength": strength, "weights_changed": patch.n_weights_changed}
            row["steps"].append(step)
            with E.applied(lm, patch):
                sc = _screen(lm, ts)
                step["screen"] = sc
                cont = PR.compare(sel0, PR.selectivity(lm, prs)) if prs else None
                if cont:
                    step["continuations"] = cont
                # The decision token can be shared by the failing and the wanted reply ("Небо" opens a
                # Russian and a Ukrainian answer alike), so the screen only skips the weaker steps; the
                # full-strength step is always judged on generation. For loops a step that does not make
                # the repeat less likely (matched continuations) is not a reason to go on either.
                last = k == len(cand.ladder) - 1
                if not last and sc["flipped"] == 0 and (cont is None or not cont["selective"]):
                    step["result"] = "no failing prompt flips at the decision token" + (
                        "; the repeated continuation does not become less likely" if cont else "")
                    say(f"  {cand.label} {strength}: {step['result']}")
                    continue
                probs = _verify(x, patch, step)
                rows = step.pop("_regression_rows", None)
                if probs:
                    step["problems"] = probs
                    step["result"] = "; ".join(probs)
                    say(f"  {cand.label} {strength}: {step['result']}")
                    continue
            step["result"] = "fixed"
            tg, reg = step["target"], dict(step["regression"], rows=rows)
            if mode == "deep":
                step["control"] = _control(x, patch, tg)
            say(f"  {cand.label} {strength}: fixed {tg['fixed']}/{tg['of']}, every {mname} check passed "
                f"({_drift(reg)})")
            return _fixed(entry, patch, tg, reg, step, trials, checks, c, mname, x, time.perf_counter() - t0)
    # No weight edit passed: try steering against the diagnosed direction (an inference-time fix).
    if steering != "off":
        got = _try_steering(x, ts, trials, say)
        if got is not None:
            return _steered(entry, *got, trials, checks, mname, x, time.perf_counter() - t0)
    # Nothing passed: say what came closest and why it failed.
    # Closest: fixed the target with the fewest regression problems, else fixed the most prompts.
    def rank(st: dict) -> tuple:
        return ("regression" in st, -len(st.get("problems", [])), st["target"]["fixed"])
    closest = None
    for row in trials:
        for st in row["steps"]:
            if "target" in st and (closest is None or rank(st) > rank(closest[1])):
                closest = (row["candidate"], st)
    reason = (f"closest: {closest[0]} {closest[1]['strength']}, {closest[1]['result']}" if closest else
              "no candidate flipped a failing prompt at its decision token")
    if fresh is not None:
        checks["fresh"] = {"failing": len(fresh["failing"]), "passing": len(fresh["passing"]),
                           "tried": fresh["tried"], "consulted": fresh["consulted"]}
    return Repair("could_not_fix", entry.name, f"{entry.name}: could not fix inside the regression budget ({reason})",
                  trials=trials, next_step=entry.could_not_fix, seconds=time.perf_counter() - t0, mode=mname,
                  checks=checks, ctx=x, sitemap=sm)


def causal_layers(res: Any, top: int = 2) -> list[tuple[int, float]]:
    """(layer, impact) of the layers that cause the failure most, from the diagnosis's own activation patching
    (deep/analysis.localise): how much of the failure goes away when that layer's attention and MLP outputs are
    replaced by a working run's. Already computed; no extra forward pass."""
    loc = getattr(res, "loc", None)
    if loc is None or not loc.attn:
        return []
    impact = [max(0.0, a) + max(0.0, m) for a, m in zip(loc.attn, loc.mlp)]
    order = sorted(range(len(impact)), key=lambda i: -impact[i])
    return [(i, round(impact[i], 4)) for i in order[:top] if impact[i] > 0]


def steering_candidates(res: Any, n_layers: int, keep_refusals: bool = False) -> list[dict[str, Any]]:
    """What to steer with, most selective first. Each: {source, placement, layers, u (unit), per (gap per layer),
    sign (-1: against the direction, +1: toward it)}.

    0. At the layers the diagnosis measured as most causal (activation patching, `causal_layers`): the most
       causal layer alone, the second alone, then both; the failing-minus-working direction at that depth.
       Steering one layer moves the least of what must stay, so it comes first.
    1. The diagnosed direction at its chosen depth, after that layer or over the three layers up to it.
    2. For refusal failures, the known refusal direction (harmful minus harmless prompts): at the causal layers,
       then over three layers around the middle and two thirds of the depth; against it for over-refusal,
       toward it for under-refusal ("complied").

    keep_refusals (the safety rule is on) for over-refusal: first, the diagnosed direction with its component along
    the known refusal direction removed, at the causal layers. Steering along what is left cannot push the harmful-
    request refusal one way or the other, so real refusals stay (found by the agent: held-out 81% -> 92%, no refusal
    lost)."""
    out: list[dict[str, Any]] = []
    d = getattr(res, "directions", None)
    kind = res.contrast.kind if getattr(res, "contrast", None) else None
    kv = getattr(res, "known_vectors", None)
    ref = (getattr(kv, "by_name", {}) or {}).get("refusal") if kv is not None else None
    cl = causal_layers(res)

    def at(vec_by_depth, layers, source, sign):
        v = vec_by_depth[layers[-1] + 1].float().cpu()
        if float(v.norm()) == 0:
            return None
        placement = (f"after layer {layers[0]}" if len(layers) == 1 else
                     f"after layers {', '.join(map(str, layers))}")
        return {"source": source, "placement": placement + " (most causal)" if layers == [cl[0][0]] else placement,
                "layers": layers, "u": v / v.norm(), "per": float(v.norm()) / len(layers), "sign": sign}
    sets = [[cl[0][0]]] if cl else []
    if len(cl) > 1:
        sets += [[cl[1][0]], sorted({cl[0][0], cl[1][0]})]
    if keep_refusals and kind == "refusal" and ref is not None and d is not None:
        orth = d.by_depth.clone().float()
        for k in range(orth.shape[0]):
            r = ref[k].float()
            if float(r.norm()) > 0:
                r = r / r.norm()
                orth[k] = orth[k] - (orth[k] @ r) * r
        for ls in sets:
            c = at(orth, ls, "the diagnosed direction without its refusal part", -1)
            if c:
                out.append(c)
    if d is not None:
        for ls in sets:
            c = at(d.by_depth, ls, "the diagnosed direction", -1)
            if c:
                out.append(c)
    if d is not None and d.best_depth >= 1 and float(d.by_depth[int(d.best_depth)].norm()) > 0:
        depth = int(d.best_depth)
        gap = float(d.by_depth[depth].norm())
        u = d.unit(depth).float().cpu()
        last = depth - 1                            # depth k is the output of layer k - 1
        out.append({"source": "the diagnosed direction", "placement": f"after layer {last}", "layers": [last],
                    "u": u, "per": gap, "sign": -1})
        if last >= 2:
            out.append({"source": "the diagnosed direction", "placement": f"over layers {last - 2}-{last}",
                        "layers": [last - 2, last - 1, last], "u": u, "per": gap / 3, "sign": -1})
    if kind in ("refusal", "complied") and ref is not None:
        for ls in sets:
            c = at(ref, ls, "the known refusal direction", -1 if kind == "refusal" else 1)
            if c:
                out.append(c)
        for frac in (0.5, 2 / 3):
            depth = max(2, min(n_layers - 1, round(frac * n_layers)))
            v = ref[depth].float().cpu()
            if float(v.norm()) == 0:
                continue
            layers = [depth - 2, depth - 1, depth]
            out.append({"source": "the known refusal direction", "placement": f"over layers {layers[0]}-{layers[-1]}",
                        "layers": layers, "u": v / v.norm(), "per": float(v.norm()) / 3,
                        "sign": -1 if kind == "refusal" else 1})
    # The most causal layer is often the diagnosis's chosen depth too: the same direction at the same layers is
    # tried once (each trial costs a full verification).
    seen: set[tuple] = set()
    uniq = []
    for cd in out:
        key = (cd["source"], tuple(cd["layers"]), cd["sign"])
        if key not in seen:
            seen.add(key)
            uniq.append(cd)
    return uniq


def _working_levels(lm: LM, res: Any, u: torch.Tensor, layers: list[int]) -> list[float]:
    """The working examples' component along u after each layer, at their decision positions: where a clamp
    lowers the failing ones to (the largest of them, so no working example is touched at its decision)."""
    vals = []
    for e in res.contrast.passing[:8]:
        ids = getattr(e, "decision_ids", None)
        if ids is None:
            continue
        r = lm.run(ids, capture=("resid",)).resid           # [n_layers + 1, d]
        vals.append([float(r[i + 1] @ u) for i in layers])
    return [max(v[j] for v in vals) for j in range(len(layers))] if vals else []


def _try_steering(x: _Ctx, ts: list, trials: list, say: Callable[[str], None]):
    """Steering candidates, weakest first, through the same screen and checks as a weight edit (no rounding
    check: steering changes no weights). Returns (Steering, step) for the first that passes, or None."""
    lm = x.lm
    cl = causal_layers(x.res)
    if cl:
        say("  steering starts at the most causal layers (diagnosis patching): " +
            ", ".join(f"layer {i} (patching it removes {v:.2f} of the failure's lead" +
                      (", more than all of it: the layer drives it" if v > 1 else "") + ")" for i, v in cl))
    got = _try_gated(x, ts, trials, say)
    if got is not None:
        return got
    for cd in steering_candidates(x.res, lm.n_layers, keep_refusals=not x.budget.allow_safety_loss):
        u, layers, per, sign, src, placement = cd["u"], cd["layers"], cd["per"], cd["sign"], cd["source"], \
            cd["placement"]
        # 1. clamp (conditional), only when steering against: the component is lowered to the working level
        lv = _working_levels(lm, x.res, u, layers) if sign < 0 else []
        if lv:
            row = {"candidate": f"clamp {src} to the working level, {placement}", "scope": "steering",
                   "kind": "steering", "evidence": f"working examples' component along it at most {max(lv):.2f}; "
                                                   "lowered only where above that", "steps": []}
            trials.append(row)
            for margin in (0.0, -0.5):
                level = max(lv) + margin * per * len(layers)
                st = Steering(u, level, layers, f"clamp {src} to {level:.2f} {placement}", src, mode="clamp")
                got = _steer_step(x, ts, row, st, f"level {level:.2f}", last=margin == -0.5, say=say)
                if got is not None:
                    return got
        # 2. add: a constant shift against (or toward) the direction
        toward = "toward" if sign > 0 else "against"
        row = {"candidate": f"steering {toward} {src}, {placement}", "scope": "steering", "kind": "steering",
               "evidence": f"gap {per * len(layers):.2f} at depth {layers[-1] + 1}; added at every position",
               "steps": []}
        trials.append(row)
        for k, sc in enumerate(STEER_SCALES):
            st = Steering(u, sign * sc * per, layers, f"steer {'+' if sign > 0 else '-'}{sc:g} x gap {toward} {src} "
                          f"{placement}", src)
            got = _steer_step(x, ts, row, st, f"{sc:g} x gap", last=k == len(STEER_SCALES) - 1, say=say)
            if got is not None:
                return got
    return None


# Probe-gated steering, weakest first: (depth: None = the probe's chosen depth, "two_thirds"; raise by sd; margin).
GATED_LADDER = ((None, 2.0, 0.5), (None, 3.0, 0.5), ("two_thirds", 2.0, 0.5), ("two_thirds", 3.0, 0.5))


def gated_candidates(probe: Any, n_layers: int) -> list[tuple[int, float, float]]:
    """(depth, up_sd, margin_up) to try, the probe's chosen depth first, then two thirds of the way down (where a
    change still has layers to act through), each depth once."""
    out, seen = [], set()
    for d, sd, mu in GATED_LADDER:
        depth = probe.depth if d is None else max(1, min(n_layers - 1, round(2 * n_layers / 3)))
        if depth not in probe.depths:
            depth = min(probe.depths, key=lambda k: abs(k - depth))
        if (depth, sd, mu) not in seen:
            seen.add((depth, sd, mu))
            out.append((depth, sd, mu))
    return out


def _try_gated(x: _Ctx, ts: list, trials: list, say: Callable[[str], None]):
    """Refusal failures: steering tied to the model's own reading of harm (deep/harmprobe.py). Skipped, with the
    reason in the trials, when the diagnosis found that reading unusable. Returns (Steering, step) or None."""
    c = getattr(x.res, "contrast", None)
    if c is None or c.kind not in ("refusal", "complied"):
        return None
    from ..deep import harmprobe as HP

    try:
        probe = HP.calibrate(x.lm, template_kwargs=x.template_kwargs, extra=x.harm_extra or None, log=say)
    except ValueError as e:
        trials.append({"candidate": "probe-gated steering", "scope": "steering", "kind": "steering",
                       "evidence": f"not tried: {e}", "steps": []})
        return None
    row = {"candidate": "probe-gated steering (the refusal tied to the model's own harm reading)", "scope": "steering",
           "kind": "steering", "evidence": probe.summary(), "steps": [], "harm_probe": probe.as_dict()}
    trials.append(row)
    if not probe.viable:
        row["result"] = "not tried: " + probe.why
        say(f"  probe-gated steering not tried: {probe.why}")
        return None
    say(f"  harm probe: {probe.summary()}")
    cands = gated_candidates(probe, x.lm.n_layers)
    for k, (depth, sd, mu) in enumerate(cands):
        try:
            g = probe.gate(up_sd=sd, margin_up=mu, depth=depth)
        except ValueError as e:
            row["result"] = f"not tried: {e}"
            return None
        st = Steering(g.r, float(g.up if g.up is not None else g.down), g.layers, g.label, "the harm probe",
                      mode="gated", gate=g)
        got = _steer_step(x, ts, row, st, f"after layer {depth - 1}, +{sd:g} sd, margin {mu:g}",
                          last=k == len(cands) - 1, say=say)
        if got is not None:
            return got
    return None


def _steer_step(x: _Ctx, ts: list, row: dict, st: "Steering", strength: str, *, last: bool,
                say: Callable[[str], None]):
    """One steering trial: the screen (a step that flips nothing is skipped unless it is the last), then the same
    checks as a weight edit. Returns (Steering, step) when it passes."""
    lm = x.lm
    step: dict[str, Any] = {"strength": strength, "steering": st.as_dict()}
    row["steps"].append(step)
    with lm.steer(**st.kwargs()):
        sc = _screen(lm, ts)
        step["screen"] = sc
        if not last and sc["flipped"] == 0:
            step["result"] = "no failing prompt flips at the decision token"
            say(f"  {row['candidate']} {strength}: {step['result']}")
            return None
        probs = _verify(x, None, step)
        rows = step.pop("_regression_rows", None)
    if probs:
        step["problems"] = probs
        step["result"] = "; ".join(probs)
        say(f"  {row['candidate']} {strength}: {step['result']}")
        return None
    step["result"] = "fixed"
    step["control"] = _steer_control(x, st, step["target"])
    say(f"  {row['candidate']} {strength}: fixed {step['target']['fixed']}/{step['target']['of']}, every check "
        f"passed ({_drift(step['regression'])})")
    step["_rows"] = rows
    return st, step


def _steer_control(x: _Ctx, st: Steering, tg: dict[str, Any]) -> dict[str, Any]:
    """A random direction of the same size at the same layers: reported, never used to reject."""
    g = torch.Generator().manual_seed(0)
    r = torch.randn(st.vector.shape[0], generator=g)
    r = r - (r @ st.vector) * st.vector
    r = r / r.norm()
    with x.lm.steer(**st.control_kwargs(r)):
        t = _target(x.lm, x.res.contrast, x.clf)
    out = {"label": "random direction, same size and layers", "fixed": t["fixed"], "of": t["of"],
           "passing_kept": t["passing_kept"], "passing": t["passing"]}
    out["specific"] = t["fixed"] * 2 < max(1, tg["fixed"])
    out["verdict"] = (f"specific: a random direction fixes {t['fixed']}/{t['of']} against {tg['fixed']}/{tg['of']}"
                      if out["specific"] else f"not specific: a random direction also fixes {t['fixed']}/{t['of']}")
    return out


def _steered(entry: Entry, st: Steering, step: dict, trials: list, checks: dict, mname: str, x: _Ctx,
             seconds: float) -> Repair:
    tg, reg = step["target"], dict(step["regression"], rows=step.pop("_rows", None) or [])
    summ = (f"{entry.name} fixed by steering ({st.label}; {mname} checks, no weights changed): {tg['fixed']}/"
            f"{tg['of']} failing prompts pass, {tg['passing_kept']}/{tg['passing']} working prompts still pass; on "
            f"{reg['prompts']} regression prompts {_drift(reg)}, harmful requests still refused "
            f"{reg['safety_refused_before'] - len(reg['safety_lost'])}/{reg['safety_refused_before']}")
    for key in ("fresh", "sampled", "control", "canary"):
        if key in step:
            checks[key] = step[key]
    checks["steering"] = st.as_dict()
    checks["evidence"] = repair_evidence(checks)
    return Repair("steered", entry.name, summ, trials=trials, regression=reg, target=tg, seconds=seconds, mode=mname,
                  checks=checks, ctx=x, steering=st)


def repair_evidence(checks: dict) -> dict:
    """What an accepted fix rests on (wsr/evidence.py): the edit or steering is an intervention; a random control
    that the fix beat is a control; fresh variants that took no part in the choice are generalization."""
    from ..evidence import make

    co = checks.get("control") or {}
    fr = checks.get("fresh") or {}
    control = "random_direction_same_size" if "steering" in checks else "random_edit_same_size"
    return make(intervention=True, controls=[control] if co.get("specific") else [],
                generalization=["fresh_variants"] if "fixed" in fr and fr.get("of") else [])


def _fixed(entry: Entry, patch: E.Patch, tg: dict, reg: dict, step: dict, trials: list, checks: dict, c: Any,
           mname: str, x: _Ctx, seconds: float, status: str = "fixed") -> Repair:
    verb = "fixed by" if status == "fixed" else "verified for"
    summ = (f"{entry.name} {verb} {patch.label} ({mname} checks): {tg['fixed']}/{tg['of']} failing prompts pass, "
            f"{tg['passing_kept']}/{tg['passing']} working prompts still pass; on {reg['prompts']} "
            f"regression prompts no new failures, {reg['unchanged_replies']} replies unchanged, "
            f"{_drift(reg)}, "
            f"harmful requests still refused {reg['safety_refused_before'] - len(reg['safety_lost'])}/"
            f"{reg['safety_refused_before']}"
            + (f"; it also fixes {len(reg['also_fixed'])} regression prompt(s) that failed before ("
               + ", ".join(r["prompt"] for r in reg["also_fixed"]) + ")" if reg.get("also_fixed") else ""))
    for key in ("serving", "fresh", "sampled", "control", "continuations", "canary"):
        if key in step:
            checks[key] = step[key]
    if x.fresh is not None and isinstance(checks.get("fresh"), dict):
        checks["fresh"] = dict(checks["fresh"], consulted=x.fresh["consulted"], tried=x.fresh["tried"])
        if x.fresh["consulted"] > 1:
            checks["fresh"]["note"] = (f"the fresh variants also rejected {x.fresh['consulted'] - 1} earlier "
                                       "candidate(s), so they are not fully independent of the choice")
    sv = checks.get("serving") or {}
    if "delta_lost" in sv:
        summ += f"; still fixes in {sv['dtype']} ({sv['delta_lost']:.0%} of the change lost to rounding)"
    fr = checks.get("fresh") or {}
    if "fixed" in fr:
        summ += f"; fresh variants fixed {fr['fixed']}/{fr['of']}"
    elif "note" in fr and x.fresh is not None and not x.fresh["failing"]:
        summ += "; no fresh variant reproduced the failure, so the fresh confirmation did not run"
    if "sampled" in checks:
        s = checks["sampled"]
        summ += (f"; sampled failure rate {s['failure_rate_before']:.0%} -> {s['failure_rate_after']:.0%} "
                 f"({s['rescues']} rescued, {s['losses']} lost)")
    if "verdict" in (checks.get("control") or {}):
        summ += f"; random control: {checks['control']['verdict']}"
    if c.kind == "language":
        # Measured on Qwen2.5-0.5B: removing the Russian direction gave Ukrainian words that made
        # no sense, and the model's own likelihood ranks a correct Ukrainian answer lower still.
        summ += ("; note: the detectors check the reply's language, not its meaning - read the "
                 "samples in REPAIR.md before shipping the patch")
    checks["evidence"] = repair_evidence(checks)
    patch.checks = {"mode": mname, "status": status, **{k: v for k, v in checks.items() if k != "sites"}}
    return Repair(status, entry.name, summ, patch=patch, trials=trials, regression=reg, target=tg,
                  seconds=seconds, mode=mname, checks=checks, ctx=x)


def check(lm: LM, res: Any, patch: E.Patch, *, expect_lang: Optional[str] = None, max_new: int = 48,
          template_kwargs: Optional[dict] = None, budget: Optional[R.Budget] = None, keep: tuple[str, ...] = (),
          mode: str = "deep", confirm: tuple[str, ...] = (), sampled: Optional[dict] = None, serve: str = "auto",
          log: Optional[Callable[[str], None]] = print) -> Repair:
    """Check an existing patch (e.g. one a fast repair found) against every check of `mode`,
    without searching: status "verified" or "not_verified"."""
    t0 = time.perf_counter()
    say = log or (lambda s: None)
    budget = budget or R.Budget()
    c = res.contrast
    entry = ENTRIES.get(c.kind, GENERIC) if c else GENERIC
    mname = _mode_name(mode, sampled)
    clf = Classifier(lm, expect_lang=expect_lang, max_new=max_new, template_kwargs=template_kwargs,
                     expect_refuse=bool(res.contrast) and res.contrast.kind == "complied")
    fresh, sb, prs, sel0, _ = _prepare(lm, res, mode=mode, clf=clf, confirm=confirm,
                                       sampled=sampled if mode == "deep" else None, say=say)
    base = R.baseline(lm, keep=keep, keep_lang=expect_lang or "en", template_kwargs=template_kwargs)
    x = _Ctx(lm, res, clf, base, budget, template_kwargs, mode, serve_dtype(lm, serve), fresh, sb, prs, sel0,
             canary=_canary0(lm, budget, template_kwargs))
    checks: dict[str, Any] = {}
    step: dict[str, Any] = {"strength": "as saved", "weights_changed": patch.n_weights_changed}
    trials = [{"candidate": patch.label, "scope": "given", "evidence": "the patch given to --check", "steps": [step]}]
    with E.applied(lm, patch):
        step["screen"] = _screen(lm, A.targets(lm, c))
        if prs:
            step["continuations"] = PR.compare(sel0, PR.selectivity(lm, prs))
        probs = _verify(x, patch, step)
        rows = step.pop("_regression_rows", None)
    if probs:
        step["problems"] = probs
        step["result"] = "; ".join(probs)
        return Repair("not_verified", entry.name, f"{patch.label} does not pass the {mname} checks: {step['result']}",
                      trials=trials, target=step.get("target"), seconds=time.perf_counter() - t0, mode=mname,
                      checks=checks, next_step="search again: `wsr repair` (with --deep for the local edits)")
    step["result"] = "passes"
    if mode == "deep":
        step["control"] = _control(x, patch, step["target"])
    reg = dict(step["regression"], rows=rows)
    return _fixed(entry, patch, step["target"], reg, step, trials, checks, c, mname, x, time.perf_counter() - t0,
                  status="verified")


# -- output ----------------------------------------------------------------------------------------

def _checks_md(rep: Repair) -> list[str]:
    ck = rep.checks or {}
    out = [f"## Checks ({rep.mode} mode)", ""]
    sv = ck.get("serving")
    if sv:
        out.append(f"- **Serving precision ({sv['dtype']})**: " + (sv.get("note") or
                   f"{sv['delta_lost']:.1%} of the change lost to rounding; failing prompts fixed "
                   f"{sv['target']['fixed']}/{sv['target']['of']}"))
    fr = ck.get("fresh")
    if fr:
        out.append("- **Fresh variants**: " + (f"{fr['fixed']}/{fr['of']} failing fixed, {fr['passing_kept']}/"
                                                 f"{fr['passing']} working kept" if "fixed" in fr else fr.get("note", "")
                                                 or f"{fr.get('failing', 0)} failing of {fr.get('tried', 0)} tried")
                   + (f" ({fr['note']})" if "fixed" in fr and fr.get("note") else ""))
    co = ck.get("control")
    if co:
        out.append(f"- **Random control** ({co.get('label', '')}): " + (co.get("verdict") or co.get("note", "")))
    sp = ck.get("sampled")
    if sp:
        out.append(f"- **Sampled runs**: failure rate {sp['failure_rate_before']:.0%} -> {sp['failure_rate_after']:.0%}"
                   f" over {sp['samples']} paired samples (gain {sp['gain']:+.2f} +- {sp['se']:.2f}; {sp['rescues']} "
                   f"rescued, {sp['losses']} lost); working prompts broken in {sp['working_broken']}/"
                   f"{sp['working_samples']}")
    cn = ck.get("continuations")
    if cn and "repeat_delta" in cn:
        out.append(f"- **Matched continuations**: repeat {cn['repeat_delta']:+.3f} nats/token, working "
                   f"{cn['working_delta']:+.3f}; end-of-turn probability in the working text "
                   f"{cn['p_end_working_before']:.4f} -> {cn['p_end_working_after']:.4f}"
                   + (" (early-stop risk)" if cn.get("early_stop_risk") else ""))
    st = ck.get("sites")
    if st:
        out.append("- **Where it is written** (" + st["source"] + "): " + ", ".join(
            f"layer {s['layer']} {s['part']} {s['share']:.0%}" for s in st["chosen"]))
    ls = ck.get("layer_screen")
    if ls:
        if ls.get("layers") and "why" not in ls:
            h = ls["held"]
            out.append(f"- **Layer screen**: layers {ls['layers']}, confirmed on held-out pairs (repeat "
                       f"{h['repeat_delta']:+.3f}, working {h['working_delta']:+.3f} nats/token at alpha "
                       f"{ls['alpha']:g})")
        else:
            out.append(f"- **Layer screen**: no candidate; {ls['why']}")
    ag = ck.get("agent")
    if ag:
        out.append(f"- **Agent**: {ag.get('provider')}, {ag.get('turns')} turns, ${ag.get('cost_usd')}, stopped: "
                   f"{ag.get('stopped')}" + (f"; unlocked by the user: {', '.join(ag['unlocked'])}"
                                             if ag.get("unlocked") else ""))
        for n in ag.get("notes") or []:
            out.append(f"  - note: {n}")
        if ag.get("resumed_from"):
            out.append(f"  - resumed from `{ag['resumed_from']}` ({len(ag.get('earlier_runs') or [])} earlier run(s))")
        if ag.get("denied"):
            out.append("  - refused in the workspace: " + ", ".join(f"{k} x{v}" for k, v in
                                                                     list(ag["denied"].items())[:12]))
    reqs = (ag or {}).get("capability_requests") or []
    if reqs:
        out += ["", "### What the agent needed but could not do", ""]
        if (ag or {}).get("rerun"):
            out += [f"Rerun with: `{ag['rerun']}`", ""]
        for r in reqs:
            out.append(f"- **{r['capability']}** ({r['category']}): {r['what_it_would_do']}")
            out.append(f"  - why: {r['why_needed']}")
            if r.get("evidence"):
                out.append(f"  - evidence: {r['evidence']}")
            if r.get("unlock"):
                out.append(f"  - how: {r['unlock']}")
            if r.get("proposal"):
                pr = r["proposal"]
                out.append("  - the proposal is tried first when the run is resumed with the unlock")
                out.append(f"  - unverified proposal saved in `agent_proposals/{pr['id']}` ({pr['label']}, "
                           f"{pr['weights_changed']:,} weights): check it with `wsr repair <model> ... --check "
                           f"agent_proposals/{pr['id']}` before using it")
    return out + [""] if len(out) > 2 else []


def save_patched_model(lm: LM, patch: E.Patch, out: Path, dtype: Optional[torch.dtype] = None) -> None:
    """The patched model, saved in `dtype` (the checkpoint's precision; the patch was checked there)."""
    with E.applied(lm, patch):
        sd = None
        if dtype is not None and next(lm.model.parameters()).dtype != dtype:
            # Tied weights (embedding = unembedding) share storage: keep one key per storage, or the
            # converted copies are written as two tensors and the model loads (and converts) untied.
            seen: set[int] = set()
            sd = {}
            for k, v in lm.model.state_dict().items():
                ptr = v.data_ptr()
                if ptr in seen and v.numel():
                    continue
                seen.add(ptr)
                sd[k] = v.to(dtype) if v.is_floating_point() else v
        prev = getattr(lm.model.config, "dtype", None)
        if dtype is not None:
            lm.model.config.dtype = dtype
        try:
            lm.model.save_pretrained(str(out), state_dict=sd)
        except Exception as e:                           # noqa: BLE001 - safetensors raises its own error type
            if "space" in str(e).lower() or "StorageFull" in str(e):
                raise OSError(f"the disk filled up while writing the patched model to {out}: free some space, "
                              "delete that folder and run again") from e
            raise
        finally:
            if dtype is not None:
                lm.model.config.dtype = prev
    lm.tok.save_pretrained(str(out))


def _repair_manifest(rep: Repair, d: Path, lm: LM, kind: str, label: str) -> None:
    """MANIFEST.json for a fix from `wsr repair`: the checks it passed on its own prompts and the regression suite.
    There is no held-out judgement here (wsr program gives one), and the manifest says so."""
    from .. import manifest as MF

    ok = rep.status in ("fixed", "verified", "steered")
    reg = rep.regression or {}
    rails = {"wsr_regression": {"kind": "suite", "holds": ok,
                                **{k: reg.get(k) for k in ("new_failures", "safety_lost", "mean_delta_nll",
                                                           "top1_agreement") if k in reg}}}
    cn = (rep.checks or {}).get("canary")
    if cn:
        rails["semantic_regression_canary"] = {"kind": "canary", **{k: cn.get(k) for k in ("holds", "eligible", "lost")}}
    MF.write(d, {"kind": kind, "artifact": {"label": label, "by": "wsr repair (" + rep.mode + ")"},
                 "model": MF.model_record(lm),
                 "statistics": {"target": rep.target and {k: rep.target.get(k) for k in ("fixed", "of", "passing_kept",
                                                                                         "passing")}},
                 "guardrails": {"pytorch": rails}, "evidence": (rep.checks or {}).get("evidence"),
                 "decision": {"status": rep.status, "shipped": ok,
                              "why": "checked on its failing and working prompts and the regression suite; not judged "
                                     "on held-out data (wsr program does that)",
                              "deployable": {"pytorch": ok}}})


def write(rep: Repair, out: Path, lm: Optional[LM] = None, save_model: bool = False) -> None:
    import json

    out.mkdir(parents=True, exist_ok=True)
    doc = {"status": rep.status, "mode": rep.mode, "entry": rep.entry, "summary": rep.summary,
           "next_step": rep.next_step, "seconds": round(rep.seconds, 1),
           "patch": rep.patch.label if rep.patch else None,
           "weights_changed": rep.patch.n_weights_changed if rep.patch else None,
           "checks": rep.checks, "target": rep.target, "regression": rep.regression, "trials": rep.trials}
    (out / "repair.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    lines = [f"# Surgery: {rep.entry}", "", f"**{rep.status.replace('_', ' ')}** ({rep.mode} mode) - {rep.summary}", ""]
    if rep.next_step:
        lines += [f"Next step: {rep.next_step}", ""]
    if rep.steering is not None:
        from . import steer as SV
        d = SV.save(rep.steering.vector, rep.steering.alpha, rep.steering.layers, rep.steering.label, out,
                    mode=rep.steering.mode, gate=rep.steering.gate)
        if lm is not None:
            _repair_manifest(rep, d, lm, "steering", rep.steering.label)
        lines += [f"Steering: {rep.steering.label} ({rep.steering.mode}, alpha {rep.steering.alpha:.3f}, layers "
                  f"{rep.steering.layers}); no weights change. Files in `{d}` (vector, settings, APPLY.md).", ""]
    if rep.patch:
        lines += [f"Patch: `{rep.patch.label}` ({len(rep.patch.deltas)} matrices, {rep.patch.n_weights_changed:,} "
                  f"weights changed, stored as {rep.patch.size_floats:,} floats). "
                  "Apply with `wsr patch apply <model-dir> <this dir>/patch --out <dir>`.", ""]
    lines += _checks_md(rep)
    if rep.regression:
        lines += ["| Regression prompt | Same reply | Before | After |", "|---|---|---|---|"]
        for r in rep.regression["rows"]:
            lines.append(f"| {r['prompt']} | {'yes' if r['same_reply'] else 'no'} | "
                         f"{r['before'][:60].replace('|', '/').replace(chr(10), ' ')} | "
                         f"{r['after'][:60].replace('|', '/').replace(chr(10), ' ')} |")
        lines.append("")
    lines += ["## Trials", ""]
    for row in rep.trials:
        lines.append(f"- **{row['candidate']}** ({row['evidence']})" +
                     (f": skipped, {row['skipped']}" if row.get("skipped") else ""))
        for st in row["steps"]:
            cn = st.get("continuations")
            extra = (f" [repeat {cn['repeat_delta']:+.3f}, working {cn['working_delta']:+.3f} nats/token]"
                     if cn else "")
            lines.append(f"  - {st['strength']}: {st.get('result', '')}{extra}")
    (out / "REPAIR.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if rep.patch:
        rep.patch.save(out / "patch", lm)
        if lm is not None:
            _repair_manifest(rep, out / "patch", lm, "patch", rep.patch.label)
        if save_model and lm is not None:
            save_patched_model(lm, rep.patch, out / "model", serve_dtype(lm, "auto"))
