"""The evaluation protocol around a fix: the semantic canary, the three data sets with a confirmation set used
once, MANIFEST.json as the record of a fix, guardrails per deployment format, and evidence dimensions."""
import json
import re
import shutil

import pytest

from wsr import evidence as EVD
from wsr.program import graders as G
from wsr.program import objective as O

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deep_fixture import planted_model  # noqa: E402

from wsr import manifest as MF  # noqa: E402
from wsr.deep.lm import LM  # noqa: E402
from wsr.program import confirm as CF  # noqa: E402
from wsr.program import evaluate as EV  # noqa: E402
from wsr.program import fromfailure as FF  # noqa: E402
from wsr.program.program import run_program  # noqa: E402
from wsr.surgery import canary as CN  # noqa: E402
from wsr.surgery import regress as R  # noqa: E402

TOY_SUITE = [("story", "write a story about the dog", "en", "general"), ("sun", "what is the sun", "en", "general")]
PRACTICE = ["tell me about the cat please", "write a story please", "what is the moon please", "list numbers please",
            "write a poem about the moon", "what is a tree", "how is the park"]
VALIDATION = ["tell me about the dog please", "what is the sun please", "write a poem please", "how is the park please",
              "where is the big red mat please", "why is the moon small please", "list numbers", "what is the sun"]
CONFIRMATION = ["who sat on the mat please", "tell me about the tree please", "write a story about the cat please",
                "what is the blue moon please", "how was the world please", "where ran the dog please", "hello world",
                "what is a tree"]


@pytest.fixture()
def lm():
    m, t = planted_model()
    return LM(m, t, name="planted")


def _jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps({"id": f"{p.stem}{i}", "prompt": r}) for i, r in enumerate(rows)) + "\n",
                 encoding="utf-8")


def _goal(tmp_path, *, confirmation=CONFIRMATION):
    _jsonl(tmp_path / "practice.jsonl", PRACTICE)
    _jsonl(tmp_path / "val" / "v.jsonl", VALIDATION)
    d = {"name": "toy-blank", "kind": "repair", "practice": "practice.jsonl", "validation": "val/v.jsonl",
         "grader": "detector:blank", "protocol": {"max_new_tokens": 12},
         "guardrails": [{"suite": "wsr_regression"}, {"suite": "semantic_regression_canary"}]}
    if confirmation is not None:
        _jsonl(tmp_path / "conf" / "c.jsonl", confirmation)
        d["confirmation"] = "conf/c.jsonl"
    (tmp_path / "goal.json").write_text(json.dumps(d), encoding="utf-8")
    return O.load(tmp_path / "goal.json")


# -- the canary --------------------------------------------------------------------------------------------

def test_canary_items_are_deterministic_categorised_and_correct():
    a, b = CN.items(), CN.items()
    assert a == b and len(a) >= 25 and len({it["category"] for it in a}) >= 5
    for it in a:
        q = it["messages"][0]["content"]
        m = re.match(r"What is (\d+) (\+|-|times) (\d+)\?", q)
        if m:
            x, op, y = int(m.group(1)), m.group(2), int(m.group(3))
            assert it["answer"] == str(x + y if op == "+" else x - y if op == "-" else x * y), q
        m = re.match(r"Which is larger, (\d+) or (\d+)\?", q)
        if m:
            assert it["answer"] == str(max(int(m.group(1)), int(m.group(2))))
        assert it.get("answer") or it.get("pattern")
    assert G.grade("item", "The answer is 113.", {"grader": "math", "answer": "113"})
    assert G.grade("item", "Paris.", {"grader": "contains", "answer": "Paris"})
    assert G.grade("item", "YES", {"grader": "regex", "pattern": r"^\W*YES\W*$"})
    assert G.grade("item", "yes, sure", {"grader": "regex", "pattern": r"^\W*YES\W*$"}) is False


def test_canary_counts_only_items_the_original_got_right():
    def rows(correct):
        return [{"id": it["id"], "labels": ["ok"], "correct": [c], "tail": ""} for it, c in zip(CN.items(), correct)]
    n = len(CN.items())
    before = rows([True] * 10 + [False] * (n - 10))
    same = CN.compare(before, before)
    assert same["eligible"] == 10 and same["lost"] == 0 and same["holds"]
    one = CN.compare(before, rows([False] + [True] * 9 + [False] * (n - 10)))
    assert one["lost"] == 1 and one["holds"]                                    # max_lost 1 by default
    two = CN.compare(before, rows([False, False] + [True] * 8 + [True] * (n - 10)))
    assert two["lost"] == 2 and two["gained"] == n - 10 and not two["holds"] and "loses 2 of 10" in two["problems"][0]
    assert CN.compare(rows([False] * n), rows([True] * n))["eligible"] == 0     # nothing to lose


def test_goal_files_take_the_canary_validation_and_confirmation():
    base = {"name": "g", "practice": "p.jsonl"}
    obj = O.from_dict({**base, "validation": "v.jsonl", "confirmation": "c.jsonl",
                       "guardrails": [{"suite": "semantic_regression_canary", "max_lost": 0}]})
    assert obj.heldout == "v.jsonl" and obj.confirmation == "c.jsonl" and obj.guardrails[0].max_lost == 0
    with pytest.raises(ValueError, match="once"):
        O.from_dict({**base, "validation": "v.jsonl", "heldout": "h.jsonl"})
    with pytest.raises(ValueError, match="built-in ones"):
        O.from_dict({**base, "heldout": "h.jsonl", "guardrails": [{"suite": "nope"}]})
    O.from_dict(O.TEMPLATE)                                                       # the template loads


# -- the three sets --------------------------------------------------------------------------------------

def test_the_split_keeps_a_confirmation_set_only_when_validation_stays_whole():
    import random
    fail = [{"id": f"f{i}"} for i in range(20)]
    work = [{"id": f"w{i}"} for i in range(8)]
    cf, cw, why = FF._three_way(fail, work, 0.5, 0.25, random.Random(0))
    assert why is None and len(cf) == 5 and len(fail) == 15 and len(cw) == 2 and len(work) == 6
    few = [{"id": f"f{i}"} for i in range(11)]
    cf, cw, why = FF._three_way(few, [], 0.5, 0.25, random.Random(0))
    assert not cf and "12 in all" in why and len(few) == 11                      # nothing taken
    assert FF._three_way(fail, work, 0.5, 0.0, random.Random(0))[2].startswith("no confirmation set requested")


def test_a_confirmation_set_is_consumed_by_its_content(tmp_path):
    a = tmp_path / "a.jsonl"
    _jsonl(a, ["x", "y"])
    assert CF.used(a) is None
    CF.consume(a, {"goal": "g1", "candidate": "c1"})
    b = tmp_path / "elsewhere" / "renamed.jsonl"
    b.parent.mkdir()
    shutil.copy(a, b)
    rec = CF.used(b)                                                              # a copy under another name
    assert rec is not None and rec["goal"] == "g1"
    _jsonl(tmp_path / "c.jsonl", ["x", "z"])
    assert CF.used(tmp_path / "c.jsonl") is None                                  # different data: unused


def test_program_confirms_once_then_refuses_to_reuse_the_set(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    obj = _goal(tmp_path)
    rec = run_program(lm, obj, out=tmp_path / "run1", log=None, mode="fast", steering="off")
    assert rec["outcome"] == "shipped" and rec["status"] == "confirmed", (rec.get("status"), rec.get("status_why"))
    conf = rec["stages"]["confirmation"]
    assert conf["consumed"] and conf["verdict"] == "improved" and conf["paired"]["wins"] >= 5
    man = MF.read(tmp_path / "run1" / "patch")
    assert man["decision"] == {**man["decision"], "status": "confirmed", "shipped": True}
    assert man["statistics"]["confirmation"]["verdict"] == "improved" and man["candidate"]
    assert "confirmation_set" in man["evidence"]["generalization"] and not man["evidence"]["replication"]
    assert "semantic_regression_canary" in man["guardrails"]["pytorch"]
    assert "## Decision" in (tmp_path / "run1" / "PROGRAM.md").read_text(encoding="utf-8")
    # the same goal again: validation still passes, but the confirmation data is used up
    rec2 = run_program(lm, O.load(tmp_path / "goal.json"), out=tmp_path / "run2", log=None, mode="fast",
                       steering="off")
    assert rec2["outcome"] == "not shipped" and rec2["status"] == "not confirmed"
    assert "already used" in rec2["status_why"] and not (tmp_path / "run2" / "patch").exists()
    assert MF.read(tmp_path / "run2" / "candidate")["decision"]["shipped"] is False


def test_a_confirmation_set_that_cannot_show_a_gain_is_not_used(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    obj = _goal(tmp_path, confirmation=["who sat on the mat please", "tell me about the tree please",
                                        "hello world", "what is a tree"])           # 2 failing: p 0.25 at best
    rec = run_program(lm, obj, out=tmp_path / "run", log=None, mode="fast", steering="off")
    assert rec["outcome"] == "not shipped" and rec["status"] == "inconclusive"
    assert "insufficient confirmation sample size" in rec["status_why"]
    assert not rec["stages"]["confirmation"]["consumed"] and CF.used(tmp_path / "conf" / "c.jsonl") is None


def test_without_a_confirmation_set_a_fix_ships_as_validated_only(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    rec = run_program(lm, _goal(tmp_path, confirmation=None), out=tmp_path / "run", log=None, mode="fast",
                      steering="off")
    assert rec["outcome"] == "shipped" and rec["status"] == "validated, not confirmed"


# -- the manifest ------------------------------------------------------------------------------------------

def test_patch_check_notices_a_changed_fix_and_a_wrong_model(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    run_program(lm, _goal(tmp_path, confirmation=None), out=tmp_path / "run", log=None, mode="fast", steering="off")
    d = tmp_path / "run" / "patch"
    assert MF.check(d, lm)["ok"]
    other = LM(*planted_model(seed=1), name="other")
    r = MF.check(d, other)
    assert not r["ok"] and r["identity"]["verdict"] == "different"
    with open(d / "patch.safetensors", "ab") as f:
        f.write(b"\0")
    assert any("hash differs" in p for p in MF.check(d)["problems"])
    assert "validated, not confirmed" in MF.render(MF.read(d))


# -- formats and evidence ----------------------------------------------------------------------------------

def test_the_served_suite_check_compares_like_with_like(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    obj = _goal(tmp_path, confirmation=None)
    g = EV.suite_generated(lm, lm, obj, label="same model")
    assert g["holds"] and g["not_supported"] == ["drift", "top-token agreement"]
    # the report renders a served-file suite check (it has no drift fields): a real run crashed here
    res = {"objective": "toy", "split": "heldout", "items": 1, "samples": 1, "protocol": "greedy", "metric": "m",
           "backend": "llama-server", "paired": {"n": 1, "before": 0.0, "after": 1.0, "diff": 1.0, "se": 0.0, "z": None,
                                                 "wins": 1, "losses": 0, "ties": 0},
           "verdict": "inconclusive", "rates_before": {}, "rates_after": {}, "guardrails": [g], "generations": 2,
           "seconds": 0.1}
    assert "not supported on served files" in EV.render(res)
    ev = EV.evaluate(lm, obj, patch=None, log=False)
    assert ev["verdict"] == "scored (no patch)"


def test_deep_findings_and_fixes_carry_evidence_dimensions(lm):
    from wsr.deep.run import diagnose
    res = diagnose(lm, [{"role": "user", "content": "tell me about the cat please"}], max_new=12)
    by = {f.code: f.evidence.get("evidence") for f in res.report.findings}
    assert by["DEEP_REPRODUCED"] == EVD.make(observation=True)
    assert by["DEEP_LOCALISED"]["intervention"] and not by["DEEP_LOCALISED"]["replication"]
    assert "working_prompts" in by["DEEP_CAUSAL"]["controls"]
    ev = EVD.merge(EVD.make(intervention=True, controls=["random_edit_same_size"]),
                   EVD.make(generalization=["validation_set"]))
    assert ev["generalization"] == ["validation_set"] and ev["replication"] == []
    assert "not replicated" in EVD.describe(ev)
