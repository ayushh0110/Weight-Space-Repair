"""Replies of the unpatched PyTorch model, kept across runs (wsr/cache.py), with the batch plan they ran in.

The original model's replies on a fixed set (the practice profile, the held-out baseline, a guardrail set) do
not change between runs: same weights (the cache key includes a weight checksum, dtype, device and library
versions), same protocol, grader, items and seeds. Re-running `wsr program` or `wsr evaluate` on the same goal
then skips them. The batch plan is stored too and installed on a hit, so the edited model's run on the same items
uses the same chunks and the before / after comparison stays exactly paired, as within one run.

Only for the PyTorch backend and an unedited model: an applied edit changes the weight version, so the cache key
is recomputed from the edited weights and a stored original can never be returned for it. WSR_CACHE=off turns
it off (as for every wsr cache).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Optional

from .. import cache
from .objective import Protocol


def _items_hash(items: list[dict]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def _depends(protocol: Protocol, grader: Optional[str], items: list[dict], samples: int, seed: int) -> dict:
    return {"protocol": dataclasses.asdict(protocol), "grader": grader, "items": _items_hash(items),
            "n": len(items), "samples": samples if protocol.sampled else 1, "seed": seed}


def plan_key(lm: Any, protocol: Protocol, items: list[dict], samples: int) -> tuple:
    """The key LM.generate_batch files this job list's chunks under (see TorchBackend.generate_many)."""
    k = samples if protocol.sampled else 1
    lens = tuple(int(lm.encode(it["messages"], **protocol.template_kwargs).shape[1]) for it in items for _ in range(k))
    return (lens, int(protocol.max_new_tokens), bool(protocol.gen_kwargs().get("do_sample")))


def load(lm: Any, kind: str, protocol: Protocol, grader: Optional[str], items: list[dict], samples: int,
         seed: int) -> Optional[list]:
    hit = cache.load(lm, kind, _depends(protocol, grader, items, samples, seed))
    if hit is None:
        return None
    meta, _ = hit
    plan = meta.get("plan")
    if plan:
        lm.batch_plans[plan_key(lm, protocol, items, samples)] = [list(c) for c in plan]
    return meta["rows"]


def save(lm: Any, kind: str, protocol: Protocol, grader: Optional[str], items: list[dict], samples: int, seed: int,
         rows: list) -> None:
    plan = getattr(lm, "batch_plans", {}).get(plan_key(lm, protocol, items, samples))
    cache.save(lm, kind, _depends(protocol, grader, items, samples, seed), {"rows": rows, "plan": plan})
