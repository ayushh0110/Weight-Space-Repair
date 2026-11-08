"""Repetition-penalty damage: fluent text that never reuses a word, or loses its shape.

Samples are real replies from a live llama-server (see data/penalty_samples.json).
"""
import json
from pathlib import Path

import pytest
from conftest import completion

from wsr.output import check_output, local_recurrence, prose_tokens, repetition_avoidance_score
from wsr.pipeline import TriageInput, triage
from wsr.probes import Endpoint, is_penalised, without_penalties

S = json.loads((Path(__file__).parent / "data" / "penalty_samples.json").read_text(encoding="utf-8"))


def ep(server):
    e = Endpoint(server.url, "m", budget_s=30, timeout=10)
    e.reasoning = False
    return e


@pytest.mark.parametrize("name", ["compare_greedy", "story_greedy", "explain_greedy", "hi_greedy", "de_greedy",
                                  "product_t08", "list30_greedy"])
def test_healthy_replies_pass_even_when_a_penalty_is_set(name):
    for pen in (False, True):
        s, _ = repetition_avoidance_score(S[name], pen)
        assert not s or s <= 0.5, (name, pen, s)


@pytest.mark.parametrize("name", ["compare_rp20", "story_rp15", "explain_rp20", "history_rp20"])
def test_penalised_prose_is_caught_without_knowing_the_request(name):
    s, ev = repetition_avoidance_score(S[name])
    assert s > 0.5, (name, ev)


def test_hindi_penalised_prose_is_caught_when_the_penalty_is_known():
    assert repetition_avoidance_score(S["hi_rp20"], True)[0] > 0.5
    assert local_recurrence(prose_tokens(S["hi_greedy"])) > 0.3  # Devanagari marks stay inside the word


def test_inflected_scripts_are_not_judged():
    # Russian sits near 0.06 even when healthy: no baseline, so no verdict.
    assert repetition_avoidance_score(S["ru_greedy"], True)[0] is None


def test_list_items_are_not_prose():
    assert repetition_avoidance_score(S["list30_greedy"], True)[0] is None


def test_check_output_reports_it_in_the_garbage_group():
    fs = {f.code: f for f in check_output(S["compare_rp20"])}
    assert fs["REPETITION_AVOIDANCE"].failed
    from wsr.verdict import _groups_of
    assert "garbage" in _groups_of({"REPETITION_AVOIDANCE"})


def test_penalty_spellings():
    assert is_penalised({"repeat_penalty": 1.3})
    assert is_penalised({"options": {"repeat_penalty": 1.5}})
    assert is_penalised({"frequency_penalty": 0.8})
    assert not is_penalised({"repeat_penalty": 1.1, "presence_penalty": 0.2})
    assert without_penalties({"options": {"repeat_penalty": 2, "num_ctx": 8}, "temperature": 0}) == \
        {"options": {"num_ctx": 8}, "temperature": 0}


def _pen_server(pen_text, ref_text):
    def h(b):
        return completion(pen_text if "repeat_penalty" in b else ref_text, "stop")
    return h


def test_fluent_but_off_task_list_is_blamed_on_the_penalty(fake_server):
    # The live case: 'list 1 to 30' with repeat_penalty 2.0 gives four lines of fluent filler.
    srv = fake_server(_pen_server(S["list30_rp20"].replace("```", ""), S["list30_greedy"]))
    rep = triage(TriageInput(endpoint=ep(srv), messages=[{"role": "user", "content": "List 1 to 30, one per line"}],
                             params={"temperature": 0, "repeat_penalty": 2.0, "max_tokens": 200}, probes=[],
                             run_differential=True))
    codes = {f.code for f in rep.findings}
    assert "PENALTY_SHAPE_LOSS" in codes and "DIFF_FLIP:penalties_neutral" in codes
    assert rep.verdict["verdict"] == "sampling"
    assert rep.verdict["primary"]["code"] == "DIFF_FLIP:penalties_neutral"


def test_penalty_that_does_no_harm_is_not_blamed(fake_server):
    srv = fake_server(_pen_server(S["compare_greedy"], S["compare_greedy"]))
    rep = triage(TriageInput(endpoint=ep(srv), messages=[{"role": "user", "content": "Compare TCP and UDP"}],
                             params={"temperature": 0, "repeat_penalty": 1.3, "max_tokens": 300}, probes=[],
                             run_differential=True))
    codes = {f.code for f in rep.findings}
    assert "PENALTY_AB_OK" in codes and "PENALTY_SHAPE_LOSS" not in codes
    assert rep.verdict["verdict"] == "no_failure_observed"


def test_penalised_prose_flips_in_the_differential(fake_server):
    srv = fake_server(_pen_server(S["compare_rp20"], S["compare_greedy"]))
    rep = triage(TriageInput(endpoint=ep(srv), messages=[{"role": "user", "content": "Compare TCP and UDP"}],
                             params={"temperature": 0, "repeat_penalty": 2.0, "max_tokens": 300}, probes=[],
                             run_differential=True))
    assert rep.verdict["verdict"] == "sampling"
    assert rep.verdict["primary"]["code"] == "DIFF_FLIP:penalties_neutral"


def test_partial_flip_credits_what_the_variant_fixed(fake_server):
    # Live Hindi case: penalties off cures the word avoidance, the reply still hits the cap.
    def h(b):
        text = S["compare_rp20"] if "repeat_penalty" in b else S["compare_greedy"]
        return completion(text.rsplit(" ", 3)[0], "length")
    rep = triage(TriageInput(endpoint=ep(fake_server(h)), messages=[{"role": "user", "content": "Compare TCP and UDP"}],
                             params={"temperature": 0, "repeat_penalty": 2.0, "max_tokens": 120}, probes=[],
                             run_differential=True))
    flip = next(f for f in rep.findings if f.code == "DIFF_FLIP:penalties_neutral")
    assert flip.evidence["explains"] == ["garbage"] and "structure remains" in flip.message
    assert rep.verdict["verdict"] == "sampling"
