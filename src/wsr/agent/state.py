"""The agent's state on disk: what `--resume` (and, later, long-run restarts) continue from.

Saved after every turn to `<out>/agent_state/`:
- state.json: the model fingerprint, wsr version, budget used, notes, capability requests, pair
  sets, verify results, the registries' metadata, and the workspace variables with every tensor
  replaced by a reference;
- tensors.safetensors: those tensors (workspace variables, direction vectors);
- edits/<id>/ and proposals/<id>/: each patch in the normal patch format.

No pickle anywhere: a state file can be read without running code from it. Workspace values that
are neither tensors nor plain data are skipped and listed (`skipped`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..surgery import edits as E
from ..surgery import pairs as PR

VERSION = 1
MAX_STATE_BYTES = 1_000_000_000          # workspace tensors above this are not saved (listed as skipped)


@torch.no_grad()
def fingerprint(lm: Any) -> str:
    """Identifies the weights: name, shape, dtype, sum and sum of squares of every tensor (computed in
    row chunks, so the embedding needs no full float copy). Any changed neuron changes it."""
    h = hashlib.sha256()
    for name, p in lm.model.named_parameters():
        t = p.detach()
        s = s2 = 0.0
        rows = t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(1, -1)
        for i in range(0, rows.shape[0], 4096):
            c = rows[i:i + 4096].float()
            s += float(c.sum())
            s2 += float((c * c).sum())
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|{s:.6e}|{s2:.6e}\n".encode())
    return h.hexdigest()[:24]


def _enc(v: Any, tens: dict[str, torch.Tensor], skipped: list[str], where: str) -> Any:
    if isinstance(v, torch.Tensor):
        k = f"t{len(tens)}"
        tens[k] = v.detach().to("cpu").contiguous().clone()
        return {"__t__": k}
    if isinstance(v, dict):
        return {"__d__": [[_enc(k, tens, skipped, where), _enc(x, tens, skipped, f"{where}.{k}")] for k, x in v.items()]}
    if isinstance(v, tuple):
        return {"__tuple__": [_enc(x, tens, skipped, where) for x in v]}
    if isinstance(v, list):
        return [_enc(x, tens, skipped, where) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    skipped.append(f"{where} ({type(v).__name__})")
    return {"__skipped__": type(v).__name__}


def _dec(v: Any, tens: dict[str, torch.Tensor]) -> Any:
    if isinstance(v, list):
        return [_dec(x, tens) for x in v]
    if isinstance(v, dict):
        if "__t__" in v:
            return tens[v["__t__"]]
        if "__d__" in v:
            return {_dec(k, tens): _dec(x, tens) for k, x in v["__d__"]}
        if "__tuple__" in v:
            return tuple(_dec(x, tens) for x in v["__tuple__"])
        if "__skipped__" in v:
            return None
        return {k: _dec(x, tens) for k, x in v.items()}
    return v


def _size(v: Any) -> int:
    if isinstance(v, torch.Tensor):
        return v.numel() * v.element_size()
    if isinstance(v, dict):
        return sum(_size(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return sum(_size(x) for x in v)
    return 0


def save(wb: Any, out: Path, *, fp: str, history: list[dict], wsr_version: str = "") -> Path:
    """Write the workbench's state; returns the state directory."""
    from safetensors.torch import save_file

    d = Path(out) / "agent_state"
    d.mkdir(parents=True, exist_ok=True)
    tens: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    lab_vars, total = {}, 0
    for k, v in wb.lab.vars.items():
        n = _size(v)
        if total + n > MAX_STATE_BYTES:
            skipped.append(f"{k} (too large to save: {n / 1e6:.0f} MB)")
            continue
        total += n
        lab_vars[k] = _enc(v, tens, skipped, k)
    dirs = {did: {"by_depth": _enc(x.get("by_depth"), tens, skipped, did), "vector": _enc(x["vector"], tens, skipped,
                                                                                          did),
                  "source": _enc(x.get("source"), tens, skipped, did)} for did, x in wb.directions.items()}
    pairs = {pid: [{"name": p.name, "prefix": p.prefix[0].tolist(), "bad": list(p.bad), "good": list(p.good)}
                   for p in ps] for pid, ps in wb.pairsets.items()}
    for kind, reg in (("edits", wb.edits), ("proposals", wb.proposals)):
        for eid, patch in reg.items():
            pd = d / kind / eid
            if not (pd / "patch.json").exists():          # patches never change once built
                patch.save(pd)
    state = {"version": VERSION, "wsr_version": wsr_version, "fingerprint": fp, "used": wb.used,
             "notes": wb.notes, "requests": wb.requests, "done": wb.done,
             "results": _enc({k: {kk: vv for kk, vv in r.items() if kk != "_rows"} for k, r in wb.results.items()},
                             tens, skipped, "results"),
             "verified": sorted(wb.verified), "edits": sorted(wb.edits), "proposals": sorted(wb.proposals),
             "directions": dirs, "pairsets": pairs, "lab": lab_vars, "lab_denied": wb.lab.denied,
             "skipped": skipped, "history": history}
    if tens:
        save_file(tens, str(d / "tensors.safetensors.tmp"))
        (d / "tensors.safetensors.tmp").replace(d / "tensors.safetensors")
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(state, default=str, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / "state.json")                         # atomic: a crash never leaves half a state
    return d


def load(wb: Any, src: Path, *, fp: str) -> dict[str, Any]:
    """Restore a saved state into a fresh workbench. Refuses a different model. Returns the raw state."""
    from safetensors.torch import load as load_bytes

    d = Path(src)
    if (d / "agent_state").is_dir():
        d = d / "agent_state"
    if not (d / "state.json").exists():
        raise ValueError(f"no saved agent state in {src} (expected agent_state/state.json)")
    st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    if st.get("version") != VERSION:
        raise ValueError(f"agent state version {st.get('version')} is not supported (this wsr reads {VERSION})")
    if st["fingerprint"] != fp:
        raise ValueError("the saved state belongs to different weights (model fingerprint differs): resume only "
                         "with the model the run was made on")
    # Read into memory, not memory-mapped: the same file is rewritten after the next turn (Windows
    # cannot replace a file that is still mapped).
    tens = load_bytes((d / "tensors.safetensors").read_bytes()) if (d / "tensors.safetensors").exists() else {}
    wb.lab.vars.update({k: _dec(v, tens) for k, v in st["lab"].items()})
    for k, n in (st.get("lab_denied") or {}).items():
        wb.lab.denied[k] = wb.lab.denied.get(k, 0) + n
    for did, x in st["directions"].items():
        wb.directions[did] = {"by_depth": _dec(x["by_depth"], tens), "vector": _dec(x["vector"], tens),
                              "source": _dec(x["source"], tens)}
    for pid, ps in st["pairsets"].items():
        wb.pairsets[pid] = [PR.Pair(p["name"], torch.tensor([p["prefix"]], dtype=torch.long), p["bad"], p["good"])
                            for p in ps]
    for eid in st["edits"]:
        wb.edits[eid] = E.Patch.load(d / "edits" / eid)
    for pid in st["proposals"]:
        wb.proposals[pid] = E.Patch.load(d / "proposals" / pid)
    results = _dec(st["results"], tens)
    wb.results.update(results)
    # Passing a check in an earlier run does not carry over: a resumed run verifies again before submitting.
    wb.notes[:0] = st["notes"]
    wb.requests[:0] = st["requests"]
    return st


def report(wb: Any, st: dict[str, Any], newly_allowed: list[str]) -> str:
    """The state report a resumed (or restarted) conversation starts from."""
    lines = ["", "## Resumed run: what the earlier run(s) did", ""]
    prev = st.get("history") or []
    if prev:
        lines.append("Earlier runs: " + "; ".join(f"{h.get('provider')}: {h.get('turns')} turns, stopped: "
                                                  f"{h.get('stopped')}" for h in prev))
    if st.get("notes"):
        lines.append("Notes:")
        lines += [f"- {n}" for n in st["notes"]]
    if wb.edits:
        lines.append("Edits built (ids still valid):")
        for eid, p in wb.edits.items():
            r = wb.results.get(eid)
            res = ("not verified" if r is None else ("passed verify_edit" if not r.get("problems") else
                                                     "failed: " + "; ".join(r["problems"])[:300]))
            lines.append(f"- {eid}: {p.label} ({p.n_weights_changed:,} weights) - {res}")
    if wb.directions:
        lines.append("Directions: " + ", ".join(wb.directions))
    if wb.pairsets:
        lines.append("Pair sets: " + ", ".join(f"{k} ({len(v)} pairs)" for k, v in wb.pairsets.items()))
    if wb.lab.vars:
        lines.append("Workspace variables: " + ", ".join(f"{k} = {v}" for k, v in wb.lab.summary().items())[:3000])
    if st.get("skipped"):
        lines.append("Not restored: " + ", ".join(st["skipped"])[:1000])
    if wb.requests:
        lines.append("Capability requests:")
        lines += [f"- {r['capability']} ({r['category']})" + (f", proposal {r['proposal']['id']}" if r.get("proposal")
                                                             else "") for r in wb.requests]
    if newly_allowed:
        lines.append("Now unlocked by the user: " + ", ".join(newly_allowed) + ". Proposals that need them were "
                     "built and verified first (see below).")
    lines.append("Budgets start fresh in this run. Edits that passed before must pass verify_edit again before "
                 "submit_patch.")
    return "\n".join(lines)
