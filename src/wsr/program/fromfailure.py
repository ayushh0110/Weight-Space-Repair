"""`wsr objective from-failure`: from "my model does X wrong" to a goal file with a scored held-out set.

1. **Confirm the failure.** The prompt is run under the protocol and labelled as in `wsr profile`;
   the same test decides what it is. A loop that sampling always escapes is a decoding setting, and
   a prompt that does not fail is not a failure: both stop here with that answer, as `wsr repair`
   does. `--deep` also runs the full deep diagnosis and stops when it says "not weight-level".
2. **Find items where that failure occurs.** A dataset the user gives (local JSONL), or a profile
   group (the profile's own items). With no dataset, the items are wsr's rewrites of the failing and
   working prompts (`variants` and `fresh_variants`). wsr does not download datasets here. For
   `--dataset auto` it names the public source that fits the failure and uses the rewrites meanwhile.
3. **Measure the hit rate.** Items that fail the same way are the failing set; items with no failure
   are the working set. Below 5 % the failure does not live in that dataset. The command then falls
   back to the rewrites, and says so.
4. **Split, stratified.** Failing and working items both go into practice and held-out, so "working
   stays working" is scored too. The held-out file is written in its own folder (`heldout/`).
5. **Write the goal file.** The grader is the failure's detector (`detector:<group>`), or the task
   grader for wrong answers. The guardrail is wsr's regression suite, which includes the
   harmful-request refusals that an over-refusal fix must keep.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from ..deep.contrast import fresh_variants, variants
from . import objective as O
from .profile import FAILURES, Runner, _family, _test_group

GROUP_OF = {"EMPTY": "blank", "LOOP": "loop", "TRUNCATED_LOOP": "loop", "REFUSAL": "refusal", "LANGUAGE": "language",
            "GARBAGE": "garbage", "TEMPLATE_LEAK": "template_leak"}
MIN_HIT = 0.05


def _item(i: int, messages: list[dict], source: str) -> dict:
    return {"id": f"{source}{i}", "messages": messages}


def rewrite_items(failing: list[list[dict]], passing: list[list[dict]]) -> list[dict]:
    seen, out = set(), []
    for msgs in failing + passing:
        for m in [msgs] + [v for _, v in variants(msgs)] + [v for _, v in fresh_variants(msgs)]:
            # Keyed by the user text, as `wsr objective check` keys its overlap test: a rewrite that only adds
            # or drops a system prompt must not land in both practice and held-out.
            key = O._key({"messages": m})
            if key not in seen:
                seen.add(key)
                out.append(_item(len(out), m, "v"))
    return out


def _split(items: list[dict], share: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    items = list(items)
    rng.shuffle(items)
    k = int(round(len(items) * share))
    if len(items) >= 2:
        k = min(max(k, 1), len(items) - 1)      # both sides get at least one
    return items[k:], items[:k]


# Failing items a set needs before any fix can pass the exact sign test there: 0.5^k below alpha. Validation is
# judged at alpha / 2 when an agent may add a second candidate (6 items); confirmation once at alpha (5 items).
MIN_FAILING_VALIDATION = 6
MIN_FAILING_CONFIRMATION = 5


def _three_way(fail: list[dict], work: list[dict], heldout_share: float, confirm_share: float,
               rng: random.Random) -> tuple[list[dict], list[dict], Optional[str]]:
    """Take the confirmation set out first, when the failing items allow it without shrinking validation below what
    it needs. Returns (confirmation failing, confirmation working, why there is none)."""
    if confirm_share <= 0:
        return [], [], "no confirmation set requested (--confirm-share 0)"
    n = len(fail)
    n_conf = max(MIN_FAILING_CONFIRMATION, int(round(confirm_share * n)))
    rest = n - n_conf
    n_val = int(round(heldout_share * rest))
    need = MIN_FAILING_CONFIRMATION + MIN_FAILING_VALIDATION + 1
    if n_val < MIN_FAILING_VALIDATION or rest - n_val < 1:
        return [], [], (f"no confirmation set: {n} failing items, and a confirmation set needs "
                        f"{MIN_FAILING_CONFIRMATION} failing items next to {MIN_FAILING_VALIDATION} for validation and "
                        f"at least 1 for practice ({need} in all); validation is not reduced to make room. Add items "
                        "that fail (a dataset, more --failing prompts) to get one")
    fail[:] = rng.sample(fail, len(fail))
    work[:] = rng.sample(work, len(work)) if work else []
    cf, fail[:] = fail[:n_conf], fail[n_conf:]
    k = int(round(len(work) * n_conf / max(1, n)))
    cw, work[:] = work[:k], work[k:]
    return cf, cw, None


def msg_hash(messages: list[dict]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def save_labels(path: Path, items: list[dict], labels: dict, protocol: O.Protocol, r, lm) -> None:
    """The practice items' replies as labelled here, so `wsr program` need not generate them again. Reused only
    when the model weights (cache key), the protocol, the grader and every item's exact messages match."""
    from .. import cache

    tlm = getattr(r.backend, "lm", None)
    rows = [{"id": it["id"], "msg": msg_hash(it["messages"]), "label": labels[it["id"]].label,
             "correct": labels[it["id"]].correct, "tokens": labels[it["id"]].n_tokens,
             "groups": sorted(labels[it["id"]].groups), "text": labels[it["id"]].text,
             "ended": labels[it["id"]].ended, "cut": labels[it["id"]].cut}
            for it in items if it["id"] in labels]
    path.write_text(json.dumps({
        "model_key": cache.model_key(tlm) if tlm is not None else None, "backend": r.backend.describe(),
        "grader": r.grader, "protocol": {"template_kwargs": protocol.template_kwargs, "sampling": protocol.sampling,
                                         "max_new_tokens": protocol.max_new_tokens, "system": protocol.system},
        "replies": rows}, ensure_ascii=False), encoding="utf-8")


def _write(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            row = {"id": it["id"], "messages": it["messages"]}
            for k in ("answer", "pattern", "lang", "expect"):
                if k in it:
                    row[k] = it[k]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def from_failure(lm, *, out_dir: Path, protocol: O.Protocol, model: str,
                 messages: Optional[list[dict]] = None, failing: tuple[str, ...] = (), passing: tuple[str, ...] = (),
                 dataset: Optional[Path] = None, auto: bool = False, profile: Optional[dict] = None,
                 group: Optional[str] = None, grader: Optional[str] = None, samples: int = 200,
                 heldout_share: float = 0.5, seed: int = 0, deep: bool = False, name: Optional[str] = None,
                 expect_lang: Optional[str] = None, consent=None, confirm_share: float = 0.25) -> dict[str, Any]:
    rng = random.Random(seed)
    res: dict[str, Any] = {"steps": []}
    step = res["steps"].append
    r = Runner(lm, protocol, grader) if lm is not None else None

    # 1. what the failure is
    if profile is not None:
        if not group:
            raise ValueError("--profile needs --group (a label from the profile, e.g. TRUNCATED_LOOP)")
        row = next((x for x in profile.get("rows", []) if x["label"] == group), None)
        if row is None:
            raise ValueError(f"no group {group} in the profile; it has "
                             f"{', '.join(x['label'] for x in profile.get('rows', []))}")
        if row.get("cause") in ("decoding", "budget"):
            return {**res, "stopped": row["cause"], "says": row["says"]}
        label = group
        step(f"failure: {group} from the profile ({row['count']} replies; {row.get('says') or 'not tested'})")
    else:
        if not messages:
            raise ValueError("give the failing prompt (--prompt / --messages), or --profile with --group")
        assert r is not None
        item = {"id": "original", "messages": messages, **({"lang": expect_lang} if expect_lang else {})}
        rep = r.run(item)
        label = rep.label
        if label not in FAILURES and label != "WRONG_ANSWER":
            return {**res, "stopped": "not_reproduced",
                    "says": f"the prompt does not fail under this protocol ({protocol.describe()}): {label}; "
                            f"reply {rep.text[:120]!r}"}
        t = _test_group(r, label, [(item, rep)], k=1, n=3)
        step(f"failure: {label} ({t.get('says')})")
        if t.get("cause") == "decoding":
            return {**res, "stopped": "decoding", "says": t["says"] + ": a sampling setting, not a weight objective"}
        if deep and getattr(r.backend, "lm", None) is None:
            step("deep: skipped - wsr deep needs the full-precision model (PyTorch backend)")
        elif deep:
            from ..deep.run import diagnose
            kind = {"EMPTY": "blank", "LOOP": "loop", "TRUNCATED_LOOP": "loop", "REFUSAL": "refusal",
                    "LANGUAGE": "language"}.get(label, "auto")
            d = diagnose(r.backend.lm, messages, kind=kind, expect_lang=expect_lang,
                         max_new=min(protocol.max_new_tokens, 256) if kind == "loop" else 48,
                         template_kwargs=protocol.template_kwargs or None)
            v = (d.report.verdict or {}).get("verdict")
            step(f"deep: {d.summary}")
            if v in ("not_weight_level", "decoding", "runtime", "template"):
                return {**res, "stopped": v, "says": d.summary}
    fam = _family(label)
    want_group = GROUP_OF.get(label)
    if grader is None:
        grader = f"detector:{want_group}" if want_group else None
    if grader is None:
        raise ValueError("a wrong-answer objective needs a task grader (--grader exact | contains | math | regex:...) "
                         "and a dataset with answers")
    if r is not None:
        r.grader = grader          # items are labelled with the goal's grader: "working" means it passes it

    # 2-3. items and hit rate
    labels: dict = {}                                  # id -> Reply, saved for wsr program (see save_labels)
    if auto and profile is None:
        from . import datasets as D

        name_ = D.for_failure(label, expect_lang)
        if name_ is None:
            step(f"--dataset auto: no public source fits {label}"
                 + (f" in {expect_lang}" if expect_lang else "") + "; using the prompt's rewrites")
        else:
            got = D.fetch(name_, consent, required=False) if consent is not None else None
            if got is None:
                step(f"--dataset auto: {name_} fits {label} but was not downloaded (answered no); using the "
                     "prompt's rewrites")
            else:
                dataset = got
                step(f"--dataset auto: {name_} ({D.SOURCES[name_].about})")
    fail_items: list[dict] = []
    work_items: list[dict] = []
    source = ""
    if profile is not None:
        data = profile.get("data")
        if not data or not Path(data).exists():
            raise ValueError("the profile does not name its data file (profiles written by this wsr do); pass "
                             "--dataset with the file it was made from")
        items = {it["id"]: it for it in O.load_items(Path(dataset or data), system=protocol.system)}
        for rp in profile.get("replies", []):
            it = items.get(rp["id"])
            if it is None:
                continue
            if _family(rp["label"]) == fam:
                fail_items.append(it)
            elif rp["label"] in ("correct", "ok") or (rp["label"] == "WRONG_ANSWER" and fam != "WRONG_ANSWER"):
                work_items.append(it)
        source = f"profile of {data}"
    elif dataset is not None:
        assert r is not None
        items = O.load_items(dataset, system=protocol.system, limit=samples)
        for it in items:
            if expect_lang and "lang" not in it:
                it["lang"] = expect_lang
        for it, rp in zip(items, r.run_many([{"item": it, "seed": 0} for it in items])):
            labels[it["id"]] = rp
            if _family(rp.label) == fam:
                fail_items.append(it)
            elif rp.label in ("correct", "ok") or (rp.label == "WRONG_ANSWER" and fam != "WRONG_ANSWER"):
                work_items.append(it)
        rate = len(fail_items) / max(1, len(items))
        step(f"hit rate in {dataset.name}: {len(fail_items)}/{len(items)} ({rate:.0%}) fail as {label}")
        source = str(dataset)
        if rate < MIN_HIT:
            step(f"below {MIN_HIT:.0%}: this failure does not live in {dataset.name}; using wsr's rewrites of the "
                 "prompt instead")
            fail_items, work_items, source = [], [], ""
    if not source:
        if not messages:
            raise ValueError("no items show this failure, and there is no prompt to rewrite")
        assert r is not None
        fl = [messages] + [[{"role": "user", "content": p}] for p in failing]
        pl = [[{"role": "user", "content": p}] for p in passing]
        rewrites = rewrite_items(fl, pl)
        for it in rewrites:
            if expect_lang:
                it["lang"] = expect_lang
        for it, rp in zip(rewrites, r.run_many([{"item": it, "seed": 0} for it in rewrites])):
            labels[it["id"]] = rp
            if _family(rp.label) == fam:
                fail_items.append(it)
            elif rp.label in ("correct", "ok"):
                work_items.append(it)
        source = "rewrites of the prompt(s)"
        step(f"rewrites: {len(fail_items)} fail as {label}, {len(work_items)} work")
    if len(fail_items) < 2:
        return {**res, "stopped": "too_few", "says": f"only {len(fail_items)} item(s) fail as {label}: too few to "
                                                     "split into practice and held-out; add --failing prompts or a "
                                                     "dataset"}

    # 4. stratified split (confirmation first, when there are enough failing items); 5. the goal file
    cf, cw, no_conf = _three_way(fail_items, work_items, heldout_share, confirm_share, rng)
    if no_conf:
        step(no_conf)
    fp, fh = _split(fail_items, heldout_share, rng)
    wp, wh = _split(work_items, heldout_share, rng) if work_items else ([], [])
    out_dir.mkdir(parents=True, exist_ok=True)
    nm = name or f"fix-{label.lower()}"
    _write(out_dir / "practice" / "practice.jsonl", fp + wp)
    if r is not None and labels:
        save_labels(out_dir / "practice" / "labels.json", fp + wp, labels, protocol, r, lm)
    _write(out_dir / "heldout" / "heldout.jsonl", fh + wh)
    if cf:
        _write(out_dir / "confirmation" / "confirmation.jsonl", cf + cw)
    goal = {"name": nm, "kind": "repair" if label != "WRONG_ANSWER" else "capability", "model": model,
            "protocol": {"template_kwargs": protocol.template_kwargs, "sampling": protocol.sampling,
                         "max_new_tokens": protocol.max_new_tokens, "system": protocol.system},
            "practice": "practice/practice.jsonl", "heldout": "heldout/heldout.jsonl", "grader": grader,
            "guardrails": [{"suite": "wsr_regression"}, {"suite": "semantic_regression_canary"}],
            "budget": {"practice_evals": 10, "verifications": 6, "usd": 5}}
    if cf:
        goal["confirmation"] = "confirmation/confirmation.jsonl"
    O.from_dict(goal)
    path = out_dir / "goal.json"
    path.write_text(json.dumps(goal, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    step(f"split: practice {len(fp)} failing + {len(wp)} working; validation (held-out) {len(fh)} failing + "
         f"{len(wh)} working" + (f"; confirmation {len(cf)} failing + {len(cw)} working (used once)" if cf else ""))
    return {**res, "stopped": None, "goal": str(path), "label": label, "grader": grader, "source": source,
            "practice": {"failing": len(fp), "working": len(wp)}, "heldout": {"failing": len(fh), "working": len(wh)},
            "confirmation": {"failing": len(cf), "working": len(cw)} if cf else {"none": no_conf}}


def render(res: dict[str, Any]) -> str:
    out = [f"  {s}" for s in res["steps"]]
    if res.get("stopped"):
        out.append(f"STOPPED ({res['stopped']}): {res['says']}")
    else:
        out.append(f"WROTE {res['goal']}: grader {res['grader']}, items from {res['source']}")
        out.append(f"  next: wsr objective check {res['goal']}")
        out.append(f"        wsr evaluate --objective {res['goal']}            (score the unedited model)")
    return "\n".join(out)
