"""The confirmation set: one evaluation of one frozen candidate, after which the set is used up.

Three sets keep the evaluation honest:
* practice: the agent optimises against it, as often as it likes;
* validation (the goal's `heldout`): candidates are judged on it, each at alpha / (candidates the run may judge);
  every look is logged;
* confirmation (optional): the one candidate that passed validation is evaluated once, frozen. Its per-item
  results are not shown and nothing is tuned against them.

A confirmation set is consumed by its content, not by the goal file that names it: the record is kept under the
SHA-256 of the data file's bytes in wsr's cache (and a note is written next to the file), so a new goal version,
a renamed copy or another candidate cannot reuse it. A new experiment that needs confirmation needs fresh data.
This guards against accidental reuse on this machine; nothing local can stop someone who copies the data
elsewhere, and the report says how the set was used.

Before a candidate is run on it, the unedited model's results on the set decide whether it can show an
improvement at all (sign-test p of a perfect fix, and its largest possible gain): when it cannot, the verdict is
"inconclusive: insufficient confirmation sample size" and the set is not consumed (no candidate saw it).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional


def data_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _registry() -> Path:
    from .. import cache

    return cache.base() / "confirmations.json"


def _read() -> dict[str, Any]:
    try:
        return json.loads(_registry().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def used(path: Path) -> Optional[dict[str, Any]]:
    """The record of an earlier use of this confirmation data (by content), or None."""
    rec = _read().get(data_hash(path))
    if rec is None:
        side = Path(str(path) + ".used.json")
        if side.exists():
            try:
                rec = json.loads(side.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                rec = {"time": "unknown", "note": f"{side.name} exists but cannot be read"}
    return rec


def consume(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": str(Path(path).resolve()), **record}
    reg = _read()
    reg[data_hash(path)] = rec
    p = _registry()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    try:
        Path(str(path) + ".used.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass                                  # a read-only data folder: the registry still holds the record
    return rec


def run(lm: Any, obj: Any, *, run_after: Callable[[list[dict]], list[dict]], record: dict[str, Any],
        say: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """Confirm one frozen candidate. `run_after(items)` runs the candidate on the items (rows as evaluate's
    _run_set returns them); `record` is stored with the consumption (goal, goal version, candidate, model).
    Returns {verdict, reason, consumed, items, paired?, decision?, power}. Per-item results are not returned."""
    from . import evaluate as EV
    from . import profile as PF
    from .backends import TorchBackend
    from .objective import load_items

    path = obj.resolve(obj.confirmation)
    if path is None or not path.exists():
        return {"verdict": "inconclusive", "reason": f"the confirmation file {obj.confirmation} does not exist",
                "consumed": False}
    earlier = used(path)
    if earlier is not None:
        return {"verdict": "not confirmed", "consumed": False, "earlier_use": earlier,
                "reason": f"this confirmation data was already used ({earlier.get('time')}, goal "
                          f"{earlier.get('goal')}): a confirmation set is evaluated once; give fresh data"}
    items = load_items(path, system=obj.protocol.system)
    js = EV.judge_settings(obj)
    r0 = PF.Runner(TorchBackend(lm), obj.protocol, obj.grader)
    before = EV.original_set(r0, items, samples=1, seed=0)
    a, _ = EV._metric(before)
    pw = EV.power(a, m=1, **js)
    if not pw["can_show_improvement"]:
        say(f"  confirmation: {pw['note']} (not used: no candidate was run on it)")
        return {"verdict": "inconclusive", "consumed": False, "items": len(items), "power": pw,
                "reason": f"insufficient confirmation sample size: {pw['note']}"}
    say(f"  confirmation: {len(items)} items, the frozen candidate once")
    after = run_after(items)
    b, _ = EV._metric(after)
    paired = EV._paired(a, b)
    dec = EV.decide(paired, True, m=1, **js)
    consume(path, {**record, "verdict": dec["verdict"], "items": len(items)})
    return {"verdict": dec["verdict"], "reason": dec["reason"], "decision": dec, "consumed": True,
            "items": len(items), "power": pw,
            "paired": {k: paired.get(k) for k in ("n", "before", "after", "diff", "wins", "losses", "ties", "z")}}
