"""Phase 1 of weight-space programming: goal files, graders, dataset profiles and the paired judge."""
import json

import pytest

from wsr.program import graders as G
from wsr.program import objective as O

# -- graders ---------------------------------------------------------------------------------------------


def test_boxed_takes_the_last_balanced_box():
    assert G.boxed(r"first \boxed{1} then \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert G.boxed("no box") is None


@pytest.mark.parametrize("pred,gold,ok", [
    (r"\dfrac{1}{2}", r"\frac12", True),
    ("0.5", r"\frac{1}{2}", True),
    ("1,000", "1000", True),
    (r"90^\circ", "90", True),
    (r"\text{(B)}", "(B)", True),
    ("3", "4", False),
    (None, "4", False),
])
def test_math_equal(pred, gold, ok):
    assert G.math_equal(pred, gold) is ok


def test_math_grader_reads_box_then_answer_phrase_then_last_number():
    item = {"answer": "42"}
    assert G.grade("math", r"so \boxed{42}.", item)
    assert G.grade("math", "The answer is 42.", item)
    assert G.grade("math", "7 times 6 makes 42", item)
    assert not G.grade("math", "it is 41", item)
    assert G.grade("math", "anything", {}) is None            # nothing to grade against


def test_exact_contains_regex_and_detector_graders():
    assert G.grade("exact", "Thinking...\nParis.", {"answer": "paris"})
    assert not G.grade("exact", "Paris is nice", {"answer": "Paris"})
    assert G.grade("contains", "I think Paris is the answer", {"answer": "paris"})
    assert G.grade(r"regex:\d{3}", "code 123", {})
    assert G.grade("regex", "abc", {"pattern": "b"})
    assert G.grade("detector:loop", "", {}, {"blank"})
    assert not G.grade("detector:blank", "", {}, {"blank"})
    assert not G.grade("detector:any", "x", {}, {"refusal"})


def test_unknown_and_unbuilt_graders_are_named():
    with pytest.raises(ValueError, match="not built yet"):
        G.check_name("ifeval")
    with pytest.raises(ValueError, match="unknown grader"):
        G.check_name("bleu")
    with pytest.raises(ValueError, match="needs a group"):
        G.check_name("detector:nonsense")

# -- goal files ------------------------------------------------------------------------------------------


def _write_jsonl(p, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_goal_file_validation_names_the_problem():
    with pytest.raises(ValueError, match="needs a name"):
        O.from_dict({})
    with pytest.raises(ValueError, match="unknown field"):
        O.from_dict({"name": "x", "heldot": "a"})
    with pytest.raises(ValueError, match="unknown sampling"):
        O.from_dict({"name": "x", "protocol": {"sampling": {"temprature": 1}}})
    with pytest.raises(ValueError, match="guardrail suite"):
        O.from_dict({"name": "x", "guardrails": [{"suite": "mmlu"}]})
    with pytest.raises(ValueError, match="kind"):
        O.from_dict({"name": "x", "kind": "vibes"})
    o = O.from_dict({"name": "x", "protocol": {"sampling": {"temperature": 0.6, "top_p": 0.95,
                                                            "presence_penalty": 1.5}}})
    assert o.protocol.sampled and o.protocol.gen_kwargs() == {"do_sample": True, "temperature": 0.6, "top_p": 0.95}


def test_items_from_math_style_rows_and_chat_rows(tmp_path):
    p = _write_jsonl(tmp_path / "d.jsonl", [
        {"problem": "1+1?", "solution": r"It is \boxed{2}.", "unique_id": "a"},
        {"messages": [{"role": "user", "content": "hi"}], "answer": "hello"},
        {"question": "q", "pattern": "x"},
    ])
    items = O.load_items(p, system="be brief")
    assert items[0]["id"] == "a" and items[0]["answer"] == "2"
    assert items[0]["messages"][0] == {"role": "system", "content": "be brief"}
    assert items[1]["answer"] == "hello" and items[2]["pattern"] == "x"
    bad = _write_jsonl(tmp_path / "bad.jsonl", [{"text": "no prompt key"}])
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        O.load_items(bad)


def test_check_finds_overlap_missing_answers_and_missing_heldout(tmp_path):
    _write_jsonl(tmp_path / "p.jsonl", [{"prompt": "a", "answer": "1"}, {"prompt": "b"}])
    _write_jsonl(tmp_path / "h.jsonl", [{"prompt": "a", "answer": "1"}])
    g = tmp_path / "goal.json"
    g.write_text(json.dumps({"name": "t", "practice": "p.jsonl", "heldout": "h.jsonl", "grader": "math"}))
    res = O.check(O.load(g))
    assert not res["ok"] and res["overlap"] == 1
    assert any("both practice and held-out" in x for x in res["problems"])
    assert any("1/2 rows have no answer" in x for x in res["problems"])
    g.write_text(json.dumps({"name": "t", "practice": "p.jsonl"}))
    assert any("no held-out file" in x for x in O.check(O.load(g))["problems"])


def test_template_and_split_round_trip(tmp_path):
    out = O.write_template(tmp_path / "goal.json", name="mine", grader="exact")
    o = O.load(out)
    assert o.name == "mine" and o.grader == "exact" and o.guardrails[0].suite == "wsr_regression"
    src = _write_jsonl(tmp_path / "all.jsonl", [{"prompt": str(i)} for i in range(10)])
    n = O.split_file(src, tmp_path / "pr.jsonl", tmp_path / "ho.jsonl", heldout_share=0.3, seed=1)
    assert n == {"practice": 7, "heldout": 3}
    a = {i["messages"][0]["content"] for i in O.load_items(tmp_path / "pr.jsonl")}
    b = {i["messages"][0]["content"] for i in O.load_items(tmp_path / "ho.jsonl")}
    assert not a & b and len(a | b) == 10


def test_cli_objective_init_and_check(tmp_path, capsys):
    from wsr.cli import main

    g = tmp_path / "goal.json"
    assert main(["objective", "init", str(g), "--name", "t"]) == 0
    assert main(["objective", "init", str(g)]) == 2                 # exists
    assert main(["objective", "check", str(g)]) == 1                 # template data files do not exist
    assert "does not exist" in capsys.readouterr().out

# -- profile and evaluate on the planted toy model -------------------------------------------------------


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deep_fixture import PLANT_LAYER, PLANT_NEURON, planted_model  # noqa: E402

from wsr.deep.lm import LM  # noqa: E402
from wsr.program import evaluate as EV  # noqa: E402
from wsr.program import profile as P  # noqa: E402
from wsr.surgery import edits as E  # noqa: E402

PLEASE = ["tell me about the cat please", "write a story please", "what is the sun please"]
OTHER = ["write a poem about the moon", "list numbers", "what is a tree", "how is the park"]


@pytest.fixture()
def lm():
    m, t = planted_model()
    return LM(m, t, name="planted")


def _items(prompts):
    return [{"id": f"i{k}", "messages": [{"role": "user", "content": c}]} for k, c in enumerate(prompts)]


def test_labels_follow_the_detectors():
    assert P.label_of("", set(), False, None) == "EMPTY"
    assert P.label_of("a a a", {"loop"}, True, None) == "TRUNCATED_LOOP"
    assert P.label_of("a a a", {"loop"}, False, None) == "LOOP"
    assert P.label_of("long", set(), True, None) == "TRUNCATED_WORKING"
    assert P.label_of("4", set(), False, False) == "WRONG_ANSWER"
    assert P.label_of("4", set(), False, True) == "correct"
    assert P.label_of("ok", set(), False, None) == "ok"
    lo, hi = P.wilson(5, 10)
    assert lo < 0.5 < hi


def test_profile_groups_the_planted_blank_and_tests_each_group(lm):
    prof = P.profile(lm, _items(PLEASE + OTHER), O.Protocol(max_new_tokens=24), grader="detector:any",
                     name="toy", model="toy-model")
    rows = {r["label"]: r for r in prof.rows}
    assert rows["EMPTY"]["count"] >= 3
    # Greedy is the protocol: a greedy blank at full precision is a weight-level candidate (as in wsr deep).
    assert rows["EMPTY"]["cause"] == "weight" and rows["EMPTY"]["test"] == "rerun"
    for lab in ("TRUNCATED_LOOP", "LOOP"):
        if lab in rows:                                        # a loop always gets the decoding test
            assert rows[lab]["cause"] in ("decoding", "weight")
    assert any(c.startswith("wsr deep toy-model --prompt") and "--kind blank" in c for c in prof.next)
    assert prof.accuracy["graded"] == len(PLEASE + OTHER)
    text = P.render(prof)
    assert "PROFILE: toy" in text and "EMPTY" in text and "Next" in text
    assert prof.generations > len(PLEASE + OTHER)               # the tests ran


def test_budget_test_reruns_cut_replies_with_twice_the_cap(lm):
    r = P.Runner(lm, O.Protocol(max_new_tokens=2))
    items = _items(["write a poem about the moon"])
    rep = r.run(items[0])
    assert rep.cut
    rep.label = "TRUNCATED_WORKING"
    t = P._test_group(r, "TRUNCATED_WORKING", [(items[0], rep)], k=1, n=1)
    assert t["test"] == "longer" and t["max_new"] == 4 and t["cause"] == "budget"


def _goal(tmp_path, heldout, *, rails=()):
    _write_jsonl(tmp_path / "h.jsonl", [{"prompt": c} for c in heldout])
    _write_jsonl(tmp_path / "g.jsonl", [{"prompt": c} for c in OTHER])
    d = {"name": "toy", "heldout": "h.jsonl", "grader": "detector:blank", "protocol": {"max_new_tokens": 12},
         "guardrails": list(rails)}
    (tmp_path / "goal.json").write_text(json.dumps(d))
    return O.load(tmp_path / "goal.json")


def test_evaluate_the_fix_is_improved_paired_and_logged(lm, tmp_path):
    obj = _goal(tmp_path, PLEASE + OTHER[:1], rails=[{"data": "g.jsonl", "grader": "detector:any", "max_drop": 0.5}])
    fix = E.scale_neurons(lm, [(PLANT_LAYER, PLANT_NEURON)], 0.0)
    res = EV.evaluate(lm, obj, patch=fix, patch_name="fix")
    p = res["paired"]
    assert p["after"] > p["before"] and p["wins"] >= 3 and p["losses"] == 0
    assert res["verdict"] == "improved", res
    assert res["rates_before"].get("EMPTY", 0) > res["rates_after"].get("EMPTY", 0)
    assert res["guardrails"][0]["holds"]
    assert res["heldout_looks"] == 1
    assert EV.evaluate(lm, obj)["heldout_looks"] == 2               # every look is counted
    assert (tmp_path / "goal.heldout_log.jsonl").exists()
    assert "IMPROVED" in EV.render(res)


def test_evaluate_a_harmless_nothing_is_no_clear_change_and_a_guardrail_can_fail(lm, tmp_path):
    obj = _goal(tmp_path, PLEASE, rails=[{"data": "g.jsonl", "grader": "detector:any", "max_drop": -0.01}])
    noop = E.scale_neurons(lm, [(0, 1)], 1.0)                      # scale 1: changes nothing
    res = EV.evaluate(lm, obj, patch=noop, log=False)
    assert res["paired"]["diff"] == 0 and res["verdict"] == "no clear change"
    assert not res["guardrails"][0]["holds"]                        # a negative max_drop cannot hold
    assert "heldout_looks" not in res


def test_sampled_evaluation_uses_the_same_seeds_for_both(lm, tmp_path):
    obj = _goal(tmp_path, OTHER)
    obj.protocol.sampling = {"temperature": 1.0}
    noop = E.scale_neurons(lm, [(0, 1)], 1.0)
    res = EV.evaluate(lm, obj, patch=noop, samples=2, log=False)
    # Same weights and same seeds: every item scores the same, so the difference is exactly 0.
    assert res["paired"]["diff"] == 0 and res["paired"]["ties"] == len(OTHER) and res["samples"] == 2


# -- from-failure ----------------------------------------------------------------------------------------

from wsr.program import fromfailure as F  # noqa: E402


def test_from_failure_builds_a_stratified_goal_from_rewrites(lm, tmp_path):
    res = F.from_failure(lm, out_dir=tmp_path / "goal", protocol=O.Protocol(max_new_tokens=12), model="toy",
                         messages=[{"role": "user", "content": PLEASE[0]}], failing=tuple(PLEASE[1:]),
                         passing=("what is a tree", "write a poem about the moon"))
    assert res["stopped"] is None, res
    assert res["label"] == "EMPTY" and res["grader"] == "detector:blank"
    assert res["heldout"]["failing"] >= 1 and res["practice"]["failing"] >= 1
    obj = O.load(res["goal"])
    assert O.check(obj)["overlap"] == 0
    assert obj.resolve(obj.heldout).parent.name == "heldout"        # its own folder
    # The goal is scorable: the planted fix improves it.
    fix = E.scale_neurons(lm, [(PLANT_LAYER, PLANT_NEURON)], 0.0)
    ev = EV.evaluate(lm, obj, patch=fix, guardrails=False, log=False)
    assert ev["paired"]["diff"] > 0


def test_from_failure_stops_when_nothing_fails(lm, tmp_path):
    res = F.from_failure(lm, out_dir=tmp_path / "g", protocol=O.Protocol(max_new_tokens=12), model="toy",
                         messages=[{"role": "user", "content": "write a poem about the moon"}])
    assert res["stopped"] == "not_reproduced" and not (tmp_path / "g" / "goal.json").exists()


def test_from_failure_from_a_profile_group(lm, tmp_path):
    data = _write_jsonl(tmp_path / "d.jsonl", [{"id": f"i{k}", "prompt": c} for k, c in enumerate(PLEASE + OTHER)])
    prof = P.profile(lm, O.load_items(data), O.Protocol(max_new_tokens=12), name="toy", model="toy", test_k=1,
                     test_n=1, data=str(data)).to_dict()
    res = F.from_failure(None, out_dir=tmp_path / "g", protocol=O.Protocol(max_new_tokens=12), model="toy",
                         profile=prof, group="EMPTY")
    assert res["stopped"] is None and res["source"].startswith("profile of")
    with pytest.raises(ValueError, match="no group"):
        F.from_failure(None, out_dir=tmp_path / "g2", protocol=O.Protocol(), model="toy", profile=prof, group="X")


# -- wsr program -----------------------------------------------------------------------------------------

from wsr.program.program import run_program  # noqa: E402
from wsr.surgery import regress as R  # noqa: E402
from wsr.surgery import playbook as PB  # noqa: E402

TOY_SUITE = [("story", "write a story about the dog", "en", "general"), ("sun", "what is the sun", "en", "general")]


def _program_goal(tmp_path):
    prac = ["tell me about the cat please", "write a story please", "what is the moon please",
            "list numbers please", "write a poem about the moon", "what is a tree", "how is the park"]
    held = ["tell me about the dog please", "what is the sun please", "write a poem please", "list numbers",
            "what is the sun"]
    _write_jsonl(tmp_path / "practice.jsonl", [{"id": f"p{i}", "prompt": c} for i, c in enumerate(prac)])
    (tmp_path / "held").mkdir()
    _write_jsonl(tmp_path / "held" / "h.jsonl", [{"id": f"h{i}", "prompt": c} for i, c in enumerate(held)])
    d = {"name": "toy-blank", "kind": "repair", "practice": "practice.jsonl", "heldout": "held/h.jsonl",
         "grader": "detector:blank", "protocol": {"max_new_tokens": 12}, "guardrails": [{"suite": "wsr_regression"}]}
    (tmp_path / "goal.json").write_text(json.dumps(d))
    return O.load(tmp_path / "goal.json")


def test_program_playbook_fix_is_judged_on_heldout_and_shipped(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    obj = _program_goal(tmp_path)
    rec = run_program(lm, obj, out=tmp_path / "run", log=None, mode="fast")
    assert rec["stages"]["group"]["label"] == "EMPTY"
    assert rec["stages"]["playbook"]["status"] == "fixed", rec["stages"]["playbook"]
    assert rec["outcome"] == "shipped", rec["stages"]["judge"]["verdict"]
    assert (tmp_path / "run" / "patch" / "patch.json").exists()
    md = (tmp_path / "run" / "PROGRAM.md").read_text(encoding="utf-8")
    assert "Held-out judge" in md and "wsr patch apply" in md and "not given to the agent" in md


def test_program_agent_uses_practice_tools_and_final_choice(lm, tmp_path, monkeypatch):
    from wsr.agent import AgentLimits, Budget, ScriptedProvider

    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    monkeypatch.setattr(PB, "candidates", lambda *a, **k: [])          # the playbook finds nothing
    obj = _program_goal(tmp_path)
    script = [
        [("build_edit", {"spec": {"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0}})],
        [("final_choice", {"edit_id": "e1", "rationale": "too early"})],         # refused: no practice_eval yet
        [("practice_eval", {"edit_id": "e1", "n": 7})],
        [("checkpoint", {"edit_id": "e1", "note": "zeroes the planted neuron"})],
        [("final_choice", {"edit_id": "e1", "rationale": "best on practice"})],
    ]
    rec = run_program(lm, obj, out=tmp_path / "run", provider=ScriptedProvider(script), playbook=True,
                      agent_budget=Budget(), agent_limits=AgentLimits(max_turns=10), log=None)
    ag = rec["stages"]["agent"]
    assert ag["stopped"] == "final choice" and ag["leaderboard"][0]["edit_id"] == "e1"
    assert ag["practice_evals"]["e1"]["diff"] > 0
    assert rec["patch"]["by"].startswith("agent (final choice")
    assert rec["outcome"] == "shipped"
    tr = json.loads((tmp_path / "run" / "agent_transcript.json").read_text(encoding="utf-8"))
    assert "h.jsonl" not in json.dumps(tr)                               # the held-out path never reaches the agent
    errs = [m for m in tr["transcript"] if m.get("role") == "tool" and m.get("is_error")]
    assert errs and "practice_eval" in json.dumps(errs[0]["result"])


def test_program_stops_when_nothing_points_at_the_weights(lm, tmp_path):
    _write_jsonl(tmp_path / "p.jsonl", [{"prompt": "what is a tree"}])
    _write_jsonl(tmp_path / "h.jsonl", [{"prompt": "how is the park"}])
    (tmp_path / "g.json").write_text(json.dumps({"name": "x", "practice": "p.jsonl", "heldout": "h.jsonl",
                                                 "grader": "detector:blank", "protocol": {"max_new_tokens": 6}}))
    rec = run_program(lm, O.load(tmp_path / "g.json"), out=tmp_path / "r", log=None)
    assert rec["outcome"] == "nothing to edit" and (tmp_path / "r" / "PROGRAM.md").exists()


def test_program_reuses_from_failure_labels_only_when_everything_matches(tmp_path, monkeypatch):
    import json as _json

    from wsr.program.backends import TorchBackend
    from wsr.program.program import prior_labels

    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    m, t = planted_model()
    m.save_pretrained(tmp_path / "toy")
    t.save_pretrained(tmp_path / "toy")
    lm2 = LM.load(tmp_path / "toy", dtype="float32", device="cpu")
    res = F.from_failure(TorchBackend(lm2), out_dir=tmp_path / "goal", protocol=O.Protocol(max_new_tokens=12),
                         model=str(tmp_path / "toy"), messages=[{"role": "user", "content": PLEASE[0]}],
                         failing=tuple(PLEASE[1:]), passing=("what is a tree", "write a poem about the moon"))
    obj = O.load(res["goal"])
    items = O.load_items(obj.resolve(obj.practice))
    prior = prior_labels(lm2, obj, items)
    assert prior is not None and len(prior) == len(items)
    fresh = P.profile(lm2, items, obj.protocol, grader=obj.grader, test_k=0)
    assert [r.label for r in prior] == [r["label"] for r in fresh.replies]       # what profiling gives now
    obj.protocol.max_new_tokens = 13                                             # another protocol: not reused
    assert prior_labels(lm2, obj, items) is None
    obj.protocol.max_new_tokens = 12
    items[0]["messages"] = items[0]["messages"] + [{"role": "user", "content": "x"}]   # other messages: not reused
    assert prior_labels(lm2, obj, items) is None
    lab = _json.loads((obj.resolve(obj.practice).parent / "labels.json").read_text(encoding="utf-8"))
    assert lab["model_key"] and lab["backend"].startswith("pytorch")
