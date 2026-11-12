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
# More prompts the planted neuron blanks: a held-out set needs at least 5 failing items before any fix can pass the
# exact sign test (0.5^5 < 0.05), and 6 when a run judges two candidates (each at 0.025).
PLEASE_MORE = ["how is the park please", "write a poem please", "tell me about the dog please",
               "where is the big red mat please", "why is the moon small please"]
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
    obj = _goal(tmp_path, PLEASE + PLEASE_MORE[:3] + OTHER[:2],
                rails=[{"data": "g.jsonl", "grader": "detector:any", "max_drop": 0.5}])
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
    obj = _goal(tmp_path, PLEASE + OTHER[:2], rails=[{"data": "g.jsonl", "grader": "detector:any", "max_drop": -0.01}])
    noop = E.scale_neurons(lm, [(0, 1)], 1.0)                      # scale 1: changes nothing
    res = EV.evaluate(lm, obj, patch=noop, log=False)
    assert res["paired"]["diff"] == 0 and res["verdict"] == "inconclusive" and "no item changed" in res["verdict_reason"]
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
    held = ["tell me about the dog please", "what is the sun please", "write a poem please", "how is the park please",
            "where is the big red mat please", "why is the moon small please", "who ran in the park please",
            "list numbers", "what is the sun"]
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


def test_practice_eval_reuses_the_profile_baseline_exactly(lm, tmp_path):
    """The profile's main pass, run on this model, is practice_eval's baseline: the result is the one a fresh
    baseline gives, and the original is not regenerated."""
    from types import SimpleNamespace as NS

    from wsr.program.program import ObjectiveTools

    obj = _program_goal(tmp_path)
    items = O.load_items(obj.resolve(obj.practice), system=obj.protocol.system)
    before = set(lm.batch_plans)
    prof = P.profile(lm, items, obj.protocol, grader=obj.grader, test_k=0)
    plans = {k: v for k, v in lm.batch_plans.items() if k not in before}
    patch = E.scale_neurons(lm, [(PLANT_LAYER, PLANT_NEURON)], 0.0)
    wb = NS(_edit=lambda eid: patch)
    lm.batch_plans.clear()                                   # as if the diagnosis had filled the cache since
    reused = ObjectiveTools(lm, obj, items, baseline=prof.replies, plans=plans)
    assert reused.default_n == len(items) and "already measured" in reused.brief(None)
    got = reused._practice_eval(wb, "e1", n=len(items))
    fresh = ObjectiveTools(lm, obj, items)
    want = fresh._practice_eval(wb, "e1", n=len(items))
    assert got == want and got["diff"] > 0
    assert reused.generations == len(items) and fresh.generations == 2 * len(items)
    # a profile that did not run here (no plans), or on other items, is not reused
    assert ObjectiveTools(lm, obj, items, baseline=prof.replies).base == {}
    assert ObjectiveTools(lm, obj, items[1:], baseline=prof.replies, plans=plans).base == {}


def test_settings_are_validated_before_anything_runs():
    from wsr.program.program import settings_problem, settings_howto, with_settings
    assert settings_problem({"repetition_penalty": 1.1}) == ""
    assert "no PyTorch equivalent" in settings_problem({"presence_penalty": 0.5})
    assert "unknown setting" in settings_problem({"dry_multiplier": 0.8})
    assert "[1.0, 1.5]" in settings_problem({"repetition_penalty": 3})
    assert "object" in settings_problem({})
    p = O.Protocol(max_new_tokens=64, system="s")
    q = with_settings(p, {"repetition_penalty": 1.1})
    assert q.gen_kwargs() == {"repetition_penalty": 1.1} and q.system == "s" and q.max_new_tokens == 64 and not p.sampling
    how = "\n".join(settings_howto(p, {"repetition_penalty": 1.1, "temperature": 0.7}))
    assert "--repeat-penalty 1.1" in how and "--temp 0.7" in how and '"do_sample": true' in how


def test_program_agent_settings_choice_is_judged_on_heldout(lm, tmp_path, monkeypatch):
    """The agent measures serving settings on practice, chooses them, and the held-out judge compares current and
    new settings on the same weights, with the suite generated under both as the guardrail; nothing is patched."""
    from wsr.agent import AgentLimits, Budget, ScriptedProvider

    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    monkeypatch.setattr(PB, "candidates", lambda *a, **k: [])
    obj = _program_goal(tmp_path)
    script = [
        [("settings_eval", {"settings": {"presence_penalty": 1.0}})],                 # refused: not on PyTorch
        [("settings_eval", {"settings": {"repetition_penalty": 1.2}, "n": 7})],
        [("final_choice", {"settings_id": "s9", "rationale": "typo"})],                # refused: unknown id
        [("final_choice", {"settings_id": "s1", "rationale": "penalty"})],
    ]
    rec = run_program(lm, obj, out=tmp_path / "run", provider=ScriptedProvider(script), playbook=True,
                      agent_budget=Budget(), agent_limits=AgentLimits(max_turns=10), log=None)
    ag = rec["stages"]["agent"]
    assert ag["stopped"] == "final choice (settings)" and ag["settings_evals"]["s1"]["n"] == 7
    js = rec["stages"]["settings"]
    assert js["settings"] == {"repetition_penalty": 1.2} and js["n"] == 9 and js["verdict"]
    assert js["guardrails"][0]["prompts"] == len(TOY_SUITE)
    assert rec["outcome"] in ("settings recommended", "no candidate") and "patch" not in rec
    assert not (tmp_path / "run" / "patch").exists()
    md = (tmp_path / "run" / "PROGRAM.md").read_text(encoding="utf-8")
    assert "Settings" in md and "repetition_penalty" in md
    tr = json.loads((tmp_path / "run" / "agent_transcript.json").read_text(encoding="utf-8"))
    errs = [m for m in tr["transcript"] if m.get("role") == "tool" and m.get("is_error")]
    assert len(errs) == 2


def test_original_replies_are_kept_across_runs_but_never_for_an_edited_model(tmp_path, monkeypatch):
    """The unedited model's replies on a fixed set are stored with their batch plan; a second process gets them
    without generating, the plan is installed, and the edited model (other weights, other key) is never served
    the stored original."""
    monkeypatch.setenv("WSR_CACHE_DIR", str(tmp_path / "cache"))
    m, t = planted_model()
    m.save_pretrained(tmp_path / "m")
    t.save_pretrained(tmp_path / "m")
    lm = LM.load(tmp_path / "m", dtype="float32", device="cpu")
    obj = _program_goal(tmp_path)
    items = O.load_items(obj.resolve(obj.practice), system=obj.protocol.system)
    r1 = P.Runner(lm, obj.protocol, obj.grader)
    rows1 = EV.original_set(r1, items, samples=1, seed=0)
    assert r1.generations == len(items)
    lm2 = LM.load(tmp_path / "m", dtype="float32", device="cpu")             # a later run
    r2 = P.Runner(lm2, obj.protocol, obj.grader)
    assert EV.original_set(r2, items, samples=1, seed=0) == rows1 and r2.generations == 0
    from wsr.program import origcache as OC
    assert OC.plan_key(lm2, obj.protocol, items, 1) in lm2.batch_plans
    patch = E.scale_neurons(lm2, [(PLANT_LAYER, PLANT_NEURON)], 0.0)
    with E.applied(lm2, patch):
        r3 = P.Runner(lm2, obj.protocol, obj.grader)
        rows3 = EV.original_set(r3, items, samples=1, seed=0)
    assert r3.generations == len(items) and rows3 != rows1                   # generated, not served the original
    prof = P.profile(lm2, items, obj.protocol, grader=obj.grader, test_k=0)  # the profile's main pass: cached too
    lm3 = LM.load(tmp_path / "m", dtype="float32", device="cpu")
    again = P.profile(lm3, items, obj.protocol, grader=obj.grader, test_k=0)
    assert [x["label"] for x in again.replies] == [x["label"] for x in prof.replies] and again.generations == 0


def test_a_settings_system_prompt_reaches_the_model():
    """The goal's system prompt is written into the items when they are loaded; a "system" setting must replace
    it in the messages (or remove it with null), not only in the protocol (live run: two opposite system prompts
    gave identical replies)."""
    from wsr.program.program import items_for
    items = [{"id": "a", "messages": [{"role": "system", "content": "old"}, {"role": "user", "content": "q"}]},
             {"id": "b", "messages": [{"role": "user", "content": "q2"}]}]
    new = items_for({"system": "new"}, items)
    assert [m["messages"][0] for m in new] == [{"role": "system", "content": "new"}] * 2
    assert [m["messages"][-1]["content"] for m in new] == ["q", "q2"] and items[0]["messages"][0]["content"] == "old"
    assert all(m["role"] != "system" for it in items_for({"system": None}, items) for m in it["messages"])
    assert items_for({"repetition_penalty": 1.1}, items) is items


def test_a_rejected_playbook_candidate_hands_over_to_the_agent(lm, tmp_path, monkeypatch):
    """The playbook's candidate passes its own checks but the held-out judge says no: the agent gets the objective
    (told only that the candidate was rejected), and its choice is judged and can ship."""
    from wsr.agent import AgentLimits, Budget, ScriptedProvider

    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    real = EV.evaluate
    calls = []

    def first_says_no(*a, **k):
        ev = real(*a, **k)
        calls.append(k.get("patch_name"))
        if len(calls) == 1:
            ev = {**ev, "verdict": "inconclusive"}
        return ev
    monkeypatch.setattr(EV, "evaluate", first_says_no)
    obj = _program_goal(tmp_path)
    script = [
        [("build_edit", {"spec": {"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0}})],
        [("practice_eval", {"edit_id": "e1", "n": 7})],
        [("final_choice", {"edit_id": "e1", "rationale": "zero the planted neuron"})],
    ]
    rec = run_program(lm, obj, out=tmp_path / "run", provider=ScriptedProvider(script), playbook=True, mode="fast",
                      agent_budget=Budget(), agent_limits=AgentLimits(max_turns=10), log=None)
    assert len(calls) == 2 and rec["stages"]["judge_playbook"]["verdict"] == "inconclusive"
    assert rec["stages"]["agent"]["stopped"] == "final choice" and rec["outcome"] == "shipped"
    assert (tmp_path / "run" / "candidate_playbook" / "patch.json").exists()
    tr = json.loads((tmp_path / "run" / "agent_transcript.json").read_text(encoding="utf-8"))
    brief = tr["transcript"][0]["content"]
    assert "did not accept it" in brief and "h.jsonl" not in json.dumps(tr)
    md = (tmp_path / "run" / "PROGRAM.md").read_text(encoding="utf-8")
    assert "playbook's candidate (not accepted" in md


def test_refusal_guardrail_is_strict_by_default_and_a_goal_can_turn_it_off(tmp_path):
    """By default a patch that stops refusing the suite's harmful requests fails the guardrail; the goal (or
    --allow-safety-loss) can allow it. The drift limits and this switch reach the verification budget too."""
    from wsr.surgery.regress import Budget
    d = {"name": "g", "practice": "p.jsonl", "heldout": "h.jsonl", "grader": "detector:any",
         "guardrails": [{"suite": "wsr_regression", "max_drift": 0.3, "allow_safety_loss": True}]}
    g = O.from_dict(d).guardrails[0]
    assert g.allow_safety_loss is True and g.max_drift == 0.3
    assert O.from_dict({**d, "guardrails": [{"suite": "wsr_regression"}]}).guardrails[0].allow_safety_loss is False
    with pytest.raises(ValueError):
        O.from_dict({**d, "guardrails": [{"suite": "wsr_regression", "allow_refusal_loss": True}]})
    reg = {"new_failures": [], "mean_delta_nll": 0.01, "top1_agreement": 0.99, "safety_refused_before": 3,
           "safety_lost": [{"prompt": "harm_weapon"}]}
    assert any("stops refusing" in p for p in Budget().problems(reg))
    assert Budget(g.max_drift, g.min_agreement, g.allow_safety_loss).problems(reg) == []


def test_deep_diagnosis_is_kept_across_runs_and_loaded_safely(tmp_path, monkeypatch):
    """A second diagnosis of the same prompts on the same weights comes from the cache, with the same report;
    the loader builds only the classes recorded for the entry, and refuses a list that names anything else."""
    import json as J
    from wsr.deep.run import diagnose
    monkeypatch.setenv("WSR_CACHE_DIR", str(tmp_path / "cache"))
    m, t = planted_model()
    m.save_pretrained(tmp_path / "m")
    t.save_pretrained(tmp_path / "m")
    msgs = [{"role": "user", "content": "tell me about the cat please"}]
    lm = LM.load(tmp_path / "m", dtype="float32", device="cpu")
    first = diagnose(lm, msgs, max_new=12, failing=("write a story please",), passing=("what is a tree",))
    assert "from_cache" not in first.report.context
    lm2 = LM.load(tmp_path / "m", dtype="float32", device="cpu")
    again = diagnose(lm2, msgs, max_new=12, failing=("write a story please",), passing=("what is a tree",),
                     out_dir=tmp_path / "out")
    assert again.report.context.get("from_cache") and again.lm is lm2
    strip = lambda r: [f for f in J.loads(J.dumps(r.report.as_dict(), default=str))["findings"]  # noqa: E731
                       if f["code"] != "DEEP_TIMINGS"]
    assert strip(again) == strip(first) and again.summary == first.summary
    assert (tmp_path / "out" / "report.json").exists()
    other = diagnose(lm2, msgs, max_new=10, failing=("write a story please",), passing=("what is a tree",))
    assert "from_cache" not in other.report.context                        # another setting: not served
    from wsr import cache as CA
    from wsr.deep import diagcache as DC
    dep = DC.depends(msgs, {"max_new": 12, "failing": ("write a story please",), "passing": ("what is a tree",)})
    meta = next((tmp_path / "cache" / "models").rglob(f"deep-diagnosis-{CA.depends_hash(dep)}.json"))
    d = J.loads(meta.read_text(encoding="utf-8"))
    assert all(c.startswith("wsr.") or c.startswith(("builtins.", "collections.")) for c in d["classes"])
    d["classes"].append("os.system")
    meta.write_text(J.dumps(d), encoding="utf-8")
    assert DC.load(lm2, dep) is None


def test_program_agent_steering_choice_is_judged_and_exported(lm, tmp_path, monkeypatch):
    """The agent measures a steering vector on practice and chooses it; the held-out judge compares the same
    weights with and without it (suite under the steering as the guardrail); nothing is patched, and when it is
    recommended the vector, its settings and how to apply it are written."""
    from wsr.agent import AgentLimits, Budget, ScriptedProvider

    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    monkeypatch.setattr(PB, "candidates", lambda *a, **k: [])
    obj = _program_goal(tmp_path)
    script = [
        [("make_direction", {"source": {"kind": "contrast"}})],
        [("steer_eval", {"vector": "d1", "alpha": 0, "layers": [2]})],                      # refused: alpha 0
        [("steer_eval", {"vector": "d1", "alpha": -4.0, "layers": [2, 3], "n": 7})],
        [("final_choice", {"steering_id": "t1", "rationale": "push against the failure direction"})],
    ]
    rec = run_program(lm, obj, out=tmp_path / "run", provider=ScriptedProvider(script), playbook=True,
                      agent_budget=Budget(), agent_limits=AgentLimits(max_turns=10), log=None)
    ag = rec["stages"]["agent"]
    assert ag["stopped"] == "final choice (steering)" and ag["steer_evals"]["t1"]["n"] == 7
    js = rec["stages"]["steering"]
    assert js["layers"] == [2, 3] and js["alpha"] == -4.0 and js["guardrails"][0]["kind"] == "suite"
    assert "patch" not in rec and not (tmp_path / "run" / "patch").exists()
    if rec["outcome"] == "steering recommended":
        d = tmp_path / "run" / "steering"
        assert (d / "steer.safetensors").exists() and (d / "APPLY.md").exists()
    md = (tmp_path / "run" / "PROGRAM.md").read_text(encoding="utf-8")
    assert "steering" in md.lower()


@pytest.mark.parametrize("mode,alpha", [("add", 2.5), ("clamp", 0.1)])
def test_exported_steering_hook_matches_wsr_steering(lm, mode, alpha):
    """The forward hook written to APPLY.md gives the same logits as LM.steer, in both modes."""
    import torch
    from wsr.program.program import STEER_APPLY
    v = torch.randn(lm.d_model)
    spec = {"alpha": alpha, "layers": [1, 3], "mode": mode}
    ids = lm.encode([{"role": "user", "content": "tell me about the cat please"}])
    with torch.no_grad(), lm.steer(**{mode: (v, spec["alpha"], spec["layers"])}):
        want = lm.model(input_ids=ids).logits
    code = STEER_APPLY.split("```python", 1)[1].split("```", 1)[0]
    body = code.split("def add_steering", 1)[1]
    ns = {"spec": spec, "vec": v}
    exec("def add_steering" + body.split("handles =", 1)[0], ns)
    hs = [lm.model.model.layers[i].register_forward_hook(ns["add_steering"]) for i in spec["layers"]]
    try:
        with torch.no_grad():
            got = lm.model(input_ids=ids).logits
    finally:
        for h in hs:
            h.remove()
    assert torch.allclose(got, want, atol=1e-5)


def test_replies_cache_is_never_used_while_steering(lm, monkeypatch, tmp_path):
    import torch
    from wsr.program import origcache as OC
    monkeypatch.setattr(OC, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache read")))
    monkeypatch.setattr(OC, "save", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache write")))
    obj = _program_goal(tmp_path)
    items = O.load_items(obj.resolve(obj.practice), system=obj.protocol.system)[:2]
    r = P.Runner(lm, obj.protocol, obj.grader)
    with lm.steer(add=(torch.randn(lm.d_model), 1.0, [1])):
        rows = EV.original_set(r, items, samples=1, seed=0)
    assert len(rows) == 2 and r.generations == 2


def test_cache_command_shows_and_clears_only_what_is_named(tmp_path, monkeypatch, capsys):
    import io
    from wsr import cache as C
    from wsr.cli import main
    monkeypatch.setenv("WSR_CACHE_DIR", str(tmp_path / "c"))
    m1, m2 = "m1-" + "a" * 20, "m2-" + "b" * 20              # as cache.model_key names them
    gg = "x-" + "c" * 12 + ".Q8_0.gguf"
    mine = ("models/my-own-model/weights.bin", "datasets/mine.jsonl", "gguf/mine.gguf")   # not wsr's: must survive
    ident = "identity/m-" + "d" * 20 + ".json"
    for rel in (f"models/{m1}/regression-baseline-1.json", f"models/{m2}/profile-replies-2.json",
                f"gguf/{gg}", "datasets/xstest.jsonl", ident, "langprobe/old.pt", "confirmations.json", *mine):
        f = tmp_path / "c" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x" * 100)
    (tmp_path / "c" / "consent.json").write_text("{}")
    shown = capsys.readouterr() and main(["cache", "show"]) == 0 and capsys.readouterr().out
    assert shown and m1 in shown and "my-own-model" not in shown
    monkeypatch.setattr("sys.stdin", io.StringIO(""))            # not a terminal, no --yes: nothing deleted
    assert main(["cache", "clear", "gguf"]) == 3 and (tmp_path / "c" / "gguf" / gg).exists()
    assert main(["cache", "clear", "--model", "m1", "--yes"]) == 0
    assert not (tmp_path / "c" / "models" / m1).exists() and (tmp_path / "c" / "models" / m2).exists()
    assert main(["cache", "clear", "all", "--yes"]) == 0
    assert not (tmp_path / "c" / "gguf" / gg).exists() and not (tmp_path / "c" / "datasets" / "xstest.jsonl").exists()
    assert not (tmp_path / "c" / "models" / m2).exists()
    # only what wsr wrote is deleted: a cache folder shared with the user's own models and data loses nothing
    assert all((tmp_path / "c" / rel).exists() for rel in mine)
    assert (tmp_path / "c" / "consent.json").exists()                # remembered answers are not cache
    assert not (tmp_path / "c" / ident).exists() and not (tmp_path / "c" / "langprobe").exists()
    assert (tmp_path / "c" / "confirmations.json").exists()          # the record of used confirmation sets is kept
    with pytest.raises(ValueError):
        C.remove([tmp_path / "elsewhere.txt"])


def test_judge_says_when_a_set_is_too_small_to_show_an_improvement():
    """2 failing items of 30: even a perfect fix has sign-test p 0.25, so the set is the limit (the first over-refusal
    goal: 93.3% -> 96.7% could never ship). 5 failing items are the least that can pass at 0.05, 6 at 0.025."""
    pw = EV.power([0.0, 0.0] + [1.0] * 28)
    assert pw["failing"] == 2 and not pw["can_show_improvement"] and pw["best_p"] == 0.25 and "add items" in pw["note"]
    assert EV.power([0.0] * 5 + [1.0] * 15)["can_show_improvement"]
    assert not EV.power([0.0] * 5 + [1.0] * 15, m=2)["can_show_improvement"]       # two candidates: 0.025 each
    assert EV.power([0.0] * 6 + [1.0] * 15, m=2)["can_show_improvement"]
    assert EV.power([0.0] * 30)["can_show_improvement"]                  # all failing: a fix would be clear
    assert not EV.power([1.0] * 30)["can_show_improvement"]
    small = EV.power([0.0] * 6 + [1.0] * 194)                            # 6 of 200: significant, but at most +3 points
    assert not small["can_show_improvement"] and "minimum" in small["note"]


def test_program_judges_the_playbooks_steering_on_heldout(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    obj = _program_goal(tmp_path)
    rec = run_program(lm, obj, out=tmp_path / "run", log=None, mode="fast", steering="only")
    pb = rec["stages"]["playbook"]
    if pb["status"] == "steered":
        js = rec["stages"].get("steering") or rec["stages"].get("steering_playbook")
        assert js["by"] == "playbook" and js["guardrails"][0]["kind"] == "suite"
        assert rec["outcome"] in ("steering recommended", "no candidate") and "patch" not in rec
        if rec["outcome"] == "steering recommended":
            assert (tmp_path / "run" / "steering" / "APPLY.md").exists()
        assert "steering" in (tmp_path / "run" / "PROGRAM.md").read_text(encoding="utf-8").lower()
    else:
        assert rec["outcome"] == "no candidate"


def test_no_weight_edits_leaves_only_steering_and_settings(lm, tmp_path, monkeypatch):
    """--no-weight-edits: the playbook only steers, the agent's build_edit and fit_edit are refused with the
    reason, steer_eval works, and the report says weight edits were off."""
    from wsr.agent import AgentLimits, Budget, ScriptedProvider

    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    monkeypatch.setattr(PB, "_try_steering", lambda *a, **k: None)          # the playbook's steering finds nothing
    obj = _program_goal(tmp_path)
    script = [
        [("build_edit", {"spec": {"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0}})],
        [("fit_edit", {"params": [f"model.layers.{PLANT_LAYER}.mlp.down_proj.weight"],
                       "targets": [{"item": "original", "weight": -1}]})],
        [("make_direction", {"source": {"kind": "contrast"}})],
        [("steer_eval", {"vector": "d1", "alpha": 0.0, "layers": [PLANT_LAYER], "mode": "clamp", "n": 7})],
        [("final_choice", {"steering_id": "t1", "rationale": "clamp at the planted layer"})],
    ]
    rec = run_program(lm, obj, out=tmp_path / "run", provider=ScriptedProvider(script), playbook=True,
                      agent_budget=Budget(), agent_limits=AgentLimits(max_turns=10), log=None, weight_edits=False)
    assert rec["weight_edits"] is False and rec["stages"]["playbook"]["status"] == "could_not_fix"
    tr = json.loads((tmp_path / "run" / "agent_transcript.json").read_text(encoding="utf-8"))
    errs = [m for m in tr["transcript"] if m.get("role") == "tool" and m.get("is_error")]
    assert {m["name"] for m in errs} >= {"build_edit", "fit_edit"}
    assert all("--no-weight-edits" in json.dumps(m["result"]) for m in errs if m["name"] in ("build_edit", "fit_edit"))
    assert "Weight edits are off" in tr["transcript"][0]["content"]
    assert rec["stages"]["agent"]["stopped"] == "final choice (steering)" and "patch" not in rec
    assert "weight edits were off" in (tmp_path / "run" / "PROGRAM.md").read_text(encoding="utf-8")


# -- audit fixes -----------------------------------------------------------------------------------------

def test_a_verdict_needs_five_scored_items_and_reports_the_exact_sign_test():
    """One or two fixed items gave "improved" (their standard error is 0): too few items now give no verdict."""
    for n in (1, 2, 4):
        p = EV._paired([0.0] * n, [1.0] * n)
        d = EV.decide(p, True)
        assert d["verdict"] == "inconclusive" and d["reason"].startswith("too few items"), n
        assert not EV.power([0.0] * n)["can_show_improvement"] and "at least 5" in EV.power([0.0] * n)["note"]
    p = EV._paired([0.0] * 5, [1.0] * 5)
    assert EV._verdict(p, True) == "improved" and p["sign_p"] == round(1 / 32, 4)
    q = EV._paired([0.0] * 4 + [1.0] * 33, [1.0] * 37)        # the over-refusal run: 4 wins, no loss, z 2.09
    d = EV.decide(q, True)
    assert q["wins"] == 4 and q["sign_p"] == 0.0625 and q["z"] > 2
    assert d["verdict"] == "inconclusive" and "not significant" in d["reason"]   # z > 2 alone no longer ships it
    assert EV._paired([1.0] * 6, [1.0] * 6)["sign_p"] is None  # nothing changed: no test to report
    assert EV.sign_p(3, 1) == round(5 / 16, 4)


def test_steering_and_settings_judges_count_as_looks_at_the_heldout_set(lm, tmp_path, monkeypatch):
    from wsr.program.program import judge_settings, judge_steering
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    obj = _program_goal(tmp_path)
    js = judge_settings(lm, obj, {"repetition_penalty": 1.1})
    assert js["heldout_looks"] == 1
    v = torch.zeros(lm.d_model)
    v[0] = 1.0
    jt = judge_steering(lm, obj, {"_vector": v, "alpha": 0.01, "layers": [0], "vector": "test", "mode": "add"})
    assert jt["heldout_looks"] == 2 and EV.HeldoutLog(obj).count() == 2
    log = (tmp_path / "goal.heldout_log.jsonl").read_text(encoding="utf-8")
    assert "settings: " in log and "steering: test" in log


def test_goal_budget_is_checked_and_the_flags_override_it(tmp_path):
    import argparse

    from wsr.cli import _agent_limits
    base = {"name": "g", "practice": "p.jsonl", "heldout": "h.jsonl"}
    obj = O.from_dict({**base, "budget": {"passes": 50, "verifications": 10, "usd": 20}})
    assert obj.budget == {"practice_evals": 50, "verifications": 10, "usd": 20}     # passes: the older name
    with pytest.raises(ValueError, match="unknown budget"):
        O.from_dict({**base, "budget": {"dollars": 5}})
    with pytest.raises(ValueError, match="not negative"):
        O.from_dict({**base, "budget": {"usd": -1}})
    ns = dict(agent_allow="", agent_turns=80, agent_max_tokens=1000)
    none = argparse.Namespace(agent_verifications=None, agent_budget_usd=None, practice_evals=None, **ns)
    b, lim, pe = _agent_limits(none, obj.budget)
    assert (b.verifications, lim.max_cost_usd, pe) == (10, 20.0, 50)               # the goal file's
    b, lim, pe = _agent_limits(none, {})
    assert (b.verifications, lim.max_cost_usd, pe) == (6, 5.0, 10)                 # the built-in defaults
    flags = argparse.Namespace(agent_verifications=2, agent_budget_usd=1.5, practice_evals=3, **ns)
    b, lim, pe = _agent_limits(flags, obj.budget)
    assert (b.verifications, lim.max_cost_usd, pe) == (2, 1.5, 3)                  # a flag wins
    with pytest.raises(ValueError, match="--agent-allow"):
        _agent_limits(argparse.Namespace(**{**vars(none), "agent_allow": "embed,everything"}))
    assert O.TEMPLATE["budget"] == {"practice_evals": 10, "verifications": 6, "usd": 5}


def test_a_bad_regex_grader_is_a_clear_error_not_a_traceback():
    from wsr.program import graders as G
    with pytest.raises(ValueError, match="not a valid regular expression"):
        G.check_name("regex:(unclosed")
    with pytest.raises(ValueError, match="not a valid regular expression"):
        G.grade("regex", "text", {"pattern": "["})
    assert G.grade("regex:^ok", "ok then", {}) is True


def test_cli_usage_errors_exit_2_and_recommended_fixes_exit_0(tmp_path, capsys):
    from wsr.cli import FIX_OUTCOMES, main
    assert main(["triage"]) == 2 and "wsr: give at least one of" in capsys.readouterr().err
    assert main(["triage", "--endpoint", "http://127.0.0.1:9"]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{bad")
    assert main(["response", str(bad)]) == 2 and "bad.json: not valid JSON" in capsys.readouterr().err
    assert main(["objective", "check", str(bad)]) == 2 and "bad.json" in capsys.readouterr().err
    assert set(FIX_OUTCOMES) == {"shipped", "steering recommended", "settings recommended"}


def test_wsr_cache_works_without_torch(tmp_path):
    """`pip install wsr-triage` has no dependencies: the cache command must not import torch."""
    import os
    import subprocess
    import sys
    code = ("import sys; sys.modules['torch'] = None; sys.modules['safetensors'] = None\n"
            "from wsr.cli import main; sys.exit(main(['cache', 'show']))")
    env = {**os.environ, "WSR_CACHE_DIR": str(tmp_path), "PYTHONPATH": "src"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "wsr cache:" in r.stdout, r.stderr[-400:]


def test_probe_caches_use_the_model_cache_not_a_folder_named_after_wsr_cache(lm, tmp_path, monkeypatch):
    """WSR_CACHE is the on/off switch: the probes once took its value as a folder name ("off/langprobe")."""
    from wsr.deep import copyheads as CH
    from wsr.deep import langprobe as LP
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WSR_CACHE", "off")
    CH.induction_scores(lm, seq_len=8, n_seqs=1)
    assert not (tmp_path / "off").exists() and not hasattr(LP, "CACHE") and not hasattr(CH, "CACHE")
    probe = LP.LangProbe(["en", "fr"], torch.randn(3, 2, 4), torch.rand(3, 4) + 0.1, [0.5, 0.9, 0.8], [1.0, 1.0, 1.0],
                         {"en": 5, "fr": 5}, 1, {"mismatch_accuracy": [float("nan"), 1.0, 1.0]})
    meta, tensors = probe.to_cache()
    back = LP.LangProbe.from_cache(json.loads(json.dumps(meta)), tensors)       # plain JSON + tensors, no pickle
    assert back.langs == probe.langs and torch.equal(back.means, probe.means) and back.counts == probe.counts


# -- judge policy (exact test, minimum gain, correction, inconclusive) ----------------------------------------

def test_the_exact_test_decides_and_inconclusive_is_not_worse():
    win = EV._paired([0.0] * 8 + [1.0] * 12, [1.0] * 20)               # 8 wins: p 0.0039, gain 40 points
    assert EV.decide(win, True)["verdict"] == "improved"
    assert EV.decide(win, False)["verdict"] == "improved on the task, but a guardrail failed"
    assert EV.decide(win, True, m=2)["p_adjusted"] == round(2 * 0.5 ** 8, 6)
    tiny = EV._paired([0.0] * 30 + [1.0] * 970, [1.0] * 1000)          # significant, but only +3 points
    t = EV.decide(tiny, True)
    assert t["verdict"] == "inconclusive" and "below the minimum" in t["reason"]
    assert EV.decide(tiny, True, min_gain=0.01)["verdict"] == "improved"
    lose = EV._paired([1.0] * 8 + [0.0] * 12, [0.0] * 20)              # 8 losses, no win
    assert EV.decide(lose, True)["verdict"] == "worse"
    mixed = EV._paired([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])   # 3 better, 2 worse
    m = EV.decide(mixed, True)
    assert m["verdict"] == "inconclusive" and "not significant" in m["reason"]
    five = EV._paired([0.0] * 5 + [1.0] * 10, [1.0] * 15)
    assert EV.decide(five, True)["verdict"] == "improved"
    assert EV.decide(five, True, m=2)["verdict"] == "inconclusive"     # p 0.031 x 2 candidates = 0.0625


def test_goal_judge_section_sets_alpha_and_min_gain():
    base = {"name": "g", "practice": "p.jsonl", "heldout": "h.jsonl"}
    obj = O.from_dict({**base, "judge": {"min_gain": 0.1, "alpha": 0.01}})
    assert EV.judge_settings(obj) == {"alpha": 0.01, "min_gain": 0.1}
    assert EV.judge_settings(O.from_dict(base)) == {"alpha": EV.ALPHA, "min_gain": EV.MIN_GAIN}
    with pytest.raises(ValueError, match="judge"):
        O.from_dict({**base, "judge": {"z": 2}})
    with pytest.raises(ValueError, match="judge"):
        O.from_dict({**base, "judge": {"alpha": 0.5}})
