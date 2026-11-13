"""What kind of evidence a claim rests on, as separate dimensions rather than one ladder.

An experiment can be interventional and controlled without being replicated, and a held-out evaluation shows
that an effect generalises, not that an independent run reproduced it. So every finding and every accepted fix
carries:

    {"observation": bool,        # a correlation was measured (a direction separates, a probe reads, a score)
     "intervention": bool,       # changing activations or weights changed the outcome
     "controls": [...],          # what ruled out chance or a generic effect (null directions, random edits ...)
     "generalization": [...],    # out-of-sample checks (fresh variants, validation and confirmation sets)
     "replication": [...]}       # independent reruns that reproduced it (only when they were actually run)

The lists name what ran, so a reader (or a results generator) can weigh the claim. Nothing is inferred: a
held-out pass adds to `generalization`, never to `replication`.
"""
from __future__ import annotations

from typing import Any, Iterable

DIMENSIONS = ("observation", "intervention", "controls", "generalization", "replication")


def make(*, observation: bool = False, intervention: bool = False, controls: Iterable[str] = (),
         generalization: Iterable[str] = (), replication: Iterable[str] = ()) -> dict[str, Any]:
    return {"observation": bool(observation), "intervention": bool(intervention),
            "controls": sorted(set(controls)), "generalization": sorted(set(generalization)),
            "replication": sorted(set(replication))}


def merge(*parts: dict[str, Any]) -> dict[str, Any]:
    out = make()
    for p in parts:
        if not p:
            continue
        out["observation"] |= bool(p.get("observation"))
        out["intervention"] |= bool(p.get("intervention"))
        for k in ("controls", "generalization", "replication"):
            out[k] = sorted(set(out[k]) | set(p.get(k) or ()))
    return out


def describe(ev: dict[str, Any]) -> str:
    """One line for reports: what the claim rests on."""
    if not ev:
        return "no evidence recorded"
    bits = []
    if ev.get("observation"):
        bits.append("observed")
    if ev.get("intervention"):
        bits.append("intervened")
    for k, word in (("controls", "controls"), ("generalization", "generalised to"), ("replication", "replicated by")):
        if ev.get(k):
            bits.append(f"{word}: {', '.join(ev[k])}")
    if not ev.get("replication"):
        bits.append("not replicated")
    return "; ".join(bits)


# What each deep-diagnosis finding measured. Controls are added only when the finding's own evidence shows the
# control ran (see for_finding).
_DEEP = {
    "DEEP_REPRODUCED": dict(observation=True), "DEEP_NOT_REPRODUCED": dict(observation=True),
    "DEEP_CUT_SHORT": dict(observation=True), "DEEP_CONTRAST": dict(observation=True),
    "DEEP_TRACE": dict(observation=True), "DEEP_WEIGHTS": dict(observation=True),
    "DEEP_QUANT_DAMAGE": dict(observation=True), "DEEP_QUANT_OK": dict(observation=True),
    "DEEP_LANG_PROBE": dict(observation=True),
    "DEEP_DIRECTION": dict(observation=True, intervention=True),
    "DEEP_KNOWN_DIRECTION": dict(observation=True),
    "DEEP_LOCALISED": dict(intervention=True),
    "DEEP_CAUSAL": dict(intervention=True, controls=["working_prompts"]),
    "DEEP_LANG_STEER": dict(intervention=True),
    "DEEP_COPY_HEADS": dict(observation=True, intervention=True),
    "DEEP_LOOP_MECHANISMS": dict(observation=True, intervention=True),
    "DEEP_POSITIONAL_DEFECT": dict(intervention=True, controls=["position_shift_invariance"]),
    "DEEP_QBISECT": dict(intervention=True),
    # Fitted on two families of calibration prompts, scored on the other two: a held-out reading, no intervention.
    "DEEP_HARM_PROBE": dict(observation=True, generalization=["held_out_prompt_families"]),
}


def for_finding(code: str, ev: dict[str, Any]) -> dict[str, Any] | None:
    """The evidence dimensions of a deep finding, from what its evidence says ran. None for summary findings."""
    base = _DEEP.get(code)
    if base is None:
        return None
    controls = list(base.get("controls", ()))
    if code == "DEEP_DIRECTION" and any("specific" in s for s in (ev.get("selection") or [])):
        controls.append("shuffled_label_null")
    if code == "DEEP_KNOWN_DIRECTION" and ev.get("known_ablation"):
        base = dict(base, intervention=True)
        controls.append("shuffled_label_null")
    if code == "DEEP_LANG_STEER" and "null_fixed" in ev:
        controls.append("unrelated_language_pair")
    if code == "DEEP_COPY_HEADS":
        controls.append("random_heads")
    if code == "DEEP_LOOP_MECHANISMS":
        controls += ["random_heads", "random_neurons", "position_shift", "sampling"]
    if code == "DEEP_QBISECT":
        controls.append("random_tensors")
    return make(observation=base.get("observation", False), intervention=base.get("intervention", False),
                controls=controls, generalization=base.get("generalization", ()))
