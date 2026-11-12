"""MANIFEST.json: the one record of why a fix was accepted, written next to every patch, steering and settings.

It says, in one place a reader or a program can check: which model the fix was made for (its exact identity, see
identity.py), what the fix is (a hash of its files), what it was judged on (the goal, its data and the grader,
each with a version), the statistics of each judgement, the guardrails per deployment format, what the evidence
rests on (evidence.py), and the decision: shipped or not, confirmed or only validated, deployable in which formats.

`wsr patch show DIR` prints it; `wsr patch check DIR [MODEL_DIR]` recomputes the fix's hash and, with a model,
its identity. The CLI, PROGRAM.md and REPAIR.md read the same record, so they cannot disagree about a fix.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

FORMAT = 1
NAME = "MANIFEST.json"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def artifact_hash(d: Path) -> Optional[str]:
    """SHA-256 of the fix itself: a patch's deltas and description (not its check results, which are appended
    later), a steering's vector and settings, or a settings file."""
    d = Path(d)
    h = hashlib.sha256()
    if (d / "patch.safetensors").exists():
        meta = json.loads((d / "patch.json").read_text(encoding="utf-8"))
        h.update(json.dumps({k: meta.get(k) for k in ("label", "kind", "untie", "params")}, sort_keys=True).encode())
        h.update(_sha_file(d / "patch.safetensors").encode())
    elif (d / "steer.safetensors").exists():
        h.update(_sha_file(d / "steer.safetensors").encode())
        spec = json.loads((d / "steering.json").read_text(encoding="utf-8"))
        h.update(json.dumps({k: spec.get(k) for k in ("mode", "alpha", "layers")}, sort_keys=True).encode())
    elif (d / "settings.json").exists():
        h.update(json.dumps(json.loads((d / "settings.json").read_text(encoding="utf-8")), sort_keys=True).encode())
    else:
        return None
    return h.hexdigest()


def model_record(lm: Any) -> dict[str, Any]:
    """The model a fix was made for: its name, revision and exact identity digests (identity.py)."""
    from . import identity as ID

    man = ID.manifest(lm)
    meta = man.get("metadata") or {}
    return {"name": getattr(lm, "name", ""), "revision": meta.get("revision"), "identity_source": man["source"],
            "digests": man["digests"], "architecture": meta.get("architecture") or {},
            "files": {k: v.get("sha256") for k, v in (man.get("files") or {}).items()}}


def goal_record(obj: Any) -> dict[str, Any]:
    """What a fix was judged on: the goal and each of its data files, by content hash, and the graders' versions.
    The goal version covers the goal file and every data file, so any change makes a new version."""
    from .output import DETECTOR_VERSION
    from .program.graders import GRADER_VERSION

    files = {}
    for role in ("practice", "heldout", "confirmation"):
        p = obj.resolve(getattr(obj, role, None))
        if p is not None and p.exists():
            files[role] = _sha_file(p)
    for g in obj.guardrails:
        if g.data:
            p = obj.resolve(g.data)
            if p is not None and p.exists():
                files[f"guardrail:{g.name}"] = _sha_file(p)
    h = hashlib.sha256()
    if obj.path is not None and Path(obj.path).exists():
        h.update(_sha_file(Path(obj.path)).encode())
    for k in sorted(files):
        h.update(f"{k}={files[k]}".encode())
    proto = obj.protocol
    return {"goal": obj.name, "goal_version": h.hexdigest(), "datasets": files, "grader": obj.grader,
            "grader_version": GRADER_VERSION, "detector_version": DETECTOR_VERSION,
            "protocol": {"template_kwargs": proto.template_kwargs, "sampling": proto.sampling,
                         "max_new_tokens": proto.max_new_tokens, "system": proto.system},
            "judge": {"alpha": (obj.judge or {}).get("alpha"), "min_gain": (obj.judge or {}).get("min_gain")}}


def candidate_hash(model: dict, artifact: Optional[str], evaluation: dict) -> str:
    """Identifies one frozen candidate under one evaluation: the base model, the fix, the protocol, the grader and
    detector versions and the datasets. Confirmation records are kept against it (program/confirm.py)."""
    key = {"model": (model.get("digests") or {}).get("float32") or model.get("digests"), "artifact": artifact,
           "protocol": evaluation.get("protocol"), "grader": evaluation.get("grader"),
           "grader_version": evaluation.get("grader_version"), "detector_version": evaluation.get("detector_version"),
           "datasets": evaluation.get("datasets")}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def statistics(ev: dict) -> dict[str, Any]:
    """The judgement of one set, from evaluate() or judge_steering / judge_settings."""
    p = ev.get("paired") or {}
    dec = ev.get("decision") or {}
    return {"items": p.get("n"), "before": p.get("before"), "after": p.get("after"), "gain": p.get("diff"),
            "wins": p.get("wins"), "losses": p.get("losses"), "ties": p.get("ties"), "z": p.get("z"),
            "test": dec.get("test"), "p": dec.get("p"), "p_adjusted": dec.get("p_adjusted"),
            "candidates_on_this_set": dec.get("m"), "alpha": dec.get("alpha"), "min_gain": dec.get("min_gain"),
            "verdict": ev.get("verdict"), "reason": ev.get("verdict_reason"),
            "power": ev.get("power"), "heldout_looks": ev.get("heldout_looks")}


def guardrails(ev: dict, fmt: str = "pytorch") -> dict[str, Any]:
    out = {}
    for g in ev.get("guardrails") or []:
        out[g["name"]] = {k: g.get(k) for k in ("kind", "holds", "problems", "new_failures", "safety_lost",
                                                  "mean_delta_nll", "top1_agreement", "eligible", "lost",
                                                  "before", "after", "max_drop", "not_supported")
                          if g.get(k) is not None}
    return {fmt: out}


def write(d: Path, record: dict[str, Any]) -> Path:
    """Write MANIFEST.json into the fix's directory, with the fix's hash computed from what is on disk."""
    from . import __version__

    d = Path(d)
    rec = {"format": FORMAT, "wsr_version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           **record}
    rec.setdefault("artifact", {})["sha256"] = artifact_hash(d)
    if rec.get("evaluation") and rec.get("model"):
        rec["candidate"] = candidate_hash(rec["model"], rec["artifact"]["sha256"], rec["evaluation"])
    tmp = d / (NAME + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / NAME)
    return d / NAME


def read(d: Path) -> dict[str, Any]:
    p = Path(d) / NAME
    if not p.exists():
        raise ValueError(f"{d} has no {NAME} (a fix from an older wsr, or not a fix directory)")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("format") != FORMAT:
        raise ValueError(f"{p}: manifest format {rec.get('format')} is not supported (this wsr reads {FORMAT})")
    return rec


def update(d: Path, path: list[str], value: Any) -> None:
    """Set one entry (e.g. the GGUF guardrails of a format, added after an export) and rewrite the file."""
    rec = read(d)
    node = rec
    for k in path[:-1]:
        node = node.setdefault(k, {})
    node[path[-1]] = value
    (Path(d) / NAME).write_text(json.dumps(rec, indent=1, default=str, ensure_ascii=False), encoding="utf-8")


def check(d: Path, lm: Any = None) -> dict[str, Any]:
    """Recompute what can be recomputed: the fix's hash, the identity file's digests, and (with a model) whether
    that model is the one the fix was made for. {ok, problems, identity}."""
    rec = read(d)
    problems = []
    now = artifact_hash(Path(d))
    if now != (rec.get("artifact") or {}).get("sha256"):
        problems.append("the fix's files changed since the manifest was written (its hash differs)")
    ident_p = Path(d) / "identity.json"
    if ident_p.exists():
        ident = json.loads(ident_p.read_text(encoding="utf-8"))
        if (ident.get("model") or {}).get("digests") != (rec.get("model") or {}).get("digests"):
            problems.append("identity.json does not match the manifest's model")
    out: dict[str, Any] = {"problems": problems}
    if lm is not None:
        from . import identity as ID

        if ident_p.exists():
            r = ID.check(lm, json.loads(ident_p.read_text(encoding="utf-8")))
        else:
            mine = ID.manifest(lm)["digests"]
            want = (rec.get("model") or {}).get("digests") or {}
            same = any(mine.get(k) == v for k, v in want.items())
            r = {"verdict": "same" if same else "different",
                 **({} if same else {"why": "the model's digests differ from the manifest's"})}
        out["identity"] = r
        if r["verdict"] != "same":
            problems.append(f"not the model this fix was made for: {r.get('why', r['verdict'])}")
    out["ok"] = not problems
    return out


def render(rec: dict[str, Any]) -> str:
    from .evidence import describe

    a, m, dec = rec.get("artifact") or {}, rec.get("model") or {}, rec.get("decision") or {}
    lines = [f"{rec.get('kind', 'fix')}: {a.get('label', '')}",
             f"  model: {m.get('name')} (revision {m.get('revision') or 'unknown'}; identity "
             f"{(m.get('digests') or {}).get('float32', '?')[:16]}..., from {m.get('identity_source')})",
             f"  fix sha256: {a.get('sha256')}"]
    if rec.get("candidate"):
        lines.append(f"  candidate: {rec['candidate'][:16]}...")
    lines.append(f"  decision: {dec.get('status', '?')}" + (f" ({dec['why']})" if dec.get("why") else ""))
    for fmt, ok in (dec.get("deployable") or {}).items():
        lines.append(f"    deployable as {fmt}: {ok}")
    ev = rec.get("evaluation") or {}
    if ev:
        lines.append(f"  judged on: goal {ev.get('goal')} (version {str(ev.get('goal_version'))[:12]}...), grader "
                     f"{ev.get('grader')} v{ev.get('grader_version')}, detectors v{ev.get('detector_version')}")
    for stage, st in (rec.get("statistics") or {}).items():
        if not st:
            continue
        lines.append(f"  {stage}: {st.get('verdict')}: {st.get('before')} -> {st.get('after')} on {st.get('items')} "
                     f"items ({st.get('wins')} better, {st.get('losses')} worse; p {st.get('p_adjusted')}, "
                     f"{st.get('test')})" + (f"; {st['reason']}" if st.get("reason") else ""))
    for fmt, rails in (rec.get("guardrails") or {}).items():
        for name, g in rails.items():
            lines.append(f"  guardrail {name} on {fmt}: {'holds' if g.get('holds') else 'FAILS' if g.get('holds') is False else 'not run'}"
                         + (f" ({'; '.join(g['problems'])})" if g.get("problems") else ""))
    if rec.get("evidence"):
        lines.append(f"  evidence: {describe(rec['evidence'])}")
    return "\n".join(lines)
