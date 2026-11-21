"""REASONING_LOOP: loops inside the reasoning channel, which the answer detectors never see.

Real traces (tests/data/reasoning_traces.json.gz, gzipped JSON): Qwen3.5-0.8B Q8_0 on llama-server, thinking on,
top-p 0.95, top-k 20, presence penalty 1.5. Eight are the Sparks setting on MATH-500 (temperature 0.6, 32,768-token
cap); the rest are GSM8K, IFEval, XSTest (safe prompts) and chat prompts at temperature 1.0 and 0.6 with an
8,192-token cap. Each is labelled by reading it: a loop, healthy (finished, or hit the cap still working) or
escaped (looped, then answered). Each keeps its last 150 lines, which is what the detector judges.
"""
import gzip
import json
import time
from pathlib import Path

import pytest
from conftest import completion

from wsr.findings import Cause, Finding, Layer, Severity
from wsr.output import RL_THRESHOLD, check_output, check_reasoning, reasoning_loop_detail
from wsr.pipeline import REPAIR_MAX_NEW, TriageInput, triage
from wsr.probes import Endpoint, differential, probe_bos
from wsr.response import check_response
from wsr.verdict import EXPLAINS, GROUPS, REASONING_LOOP_CAUSES, decide

TRACES = json.loads(gzip.decompress((Path(__file__).parent / "data" / "reasoning_traces.json.gz").read_bytes()))
LOOPS = [t for t in TRACES if t["label"] == "loop"]
HEALTHY = [t for t in TRACES if t["label"] == "healthy"]
ESCAPED = [t for t in TRACES if t["label"] == "escaped"]


def _unique_lines(n, start=0):
    return [f"Step {i}: consider the case where x equals {i * 7 % 13} and check the remainder." for i in
            range(start, start + n)]


def _looping_reasoning(head=100, block=6, turns=30):
    cyc = [f"Wait, let me re-check equation {j} because the sign might be wrong." for j in range(block)]
    return "\n".join(_unique_lines(head) + cyc * turns) + "\n"


# -- the detector -----------------------------------------------------------------------------------------

def test_fixture_has_both_kinds():
    assert len(LOOPS) >= 15 and len(HEALTHY) >= 25 and ESCAPED
    assert {t["label"] for t in TRACES} == {"loop", "healthy", "escaped"}


def test_the_late_loop_is_caught():
    # c005: the loop began about 100 lines before the cap, too late to fill the 120-line window.
    t = next(t for t in LOOPS if t["name"].endswith("c005"))
    assert reasoning_loop_detail(t["reasoning"]).lines == 40
    assert check_reasoning(t["reasoning"], cut=True, answered=False)[0].severity == Severity.FAIL


@pytest.mark.parametrize("t", ESCAPED, ids=lambda t: t["name"])
def test_a_real_loop_that_escaped_never_fails(t):
    assert not check_reasoning(t["reasoning"], cut=False, answered=True)[0].failed


@pytest.mark.parametrize("t", LOOPS, ids=lambda t: t["name"])
def test_real_reasoning_loops_fail(t):
    f, = check_reasoning(t["reasoning"], cut=True, answered=False)
    assert f.severity == Severity.FAIL and f.code == "REASONING_LOOP", (f.score, f.message)
    assert f.layer == Layer.OUTPUT and f.cause == Cause.SYMPTOM
    assert "more tokens will not end a loop" in f.fix


@pytest.mark.parametrize("t", HEALTHY, ids=lambda t: t["name"])
def test_real_healthy_reasoning_passes_even_at_the_cap(t):
    f, = check_reasoning(t["reasoning"], cut=True, answered=False)
    assert f.severity == Severity.PASS, (f.score, f.evidence)
    assert f.score <= RL_THRESHOLD


def test_margin_between_loops_and_healthy_traces():
    worst_healthy = max(reasoning_loop_detail(t["reasoning"]).score for t in HEALTHY)
    best_loop = min(reasoning_loop_detail(t["reasoning"]).score for t in LOOPS)
    assert worst_healthy < RL_THRESHOLD < best_loop


def test_a_loop_the_model_escaped_is_information_only():
    f, = check_reasoning(_looping_reasoning(), cut=False, answered=True)
    assert f.severity == Severity.INFO and not f.failed


@pytest.mark.parametrize("cut,answered", [(True, False), (False, False), (True, True)])
def test_a_loop_that_cost_the_answer_fails(cut, answered):
    f, = check_reasoning(_looping_reasoning(), cut=cut, answered=answered)
    assert f.severity == Severity.FAIL


def test_too_little_reasoning_is_not_judged():
    assert check_reasoning("Thinking Process: the answer is short.", cut=True, answered=False) == []
    assert check_reasoning("\n".join(["Wait, check that again carefully now."] * 20), cut=True, answered=False) == []
    assert check_reasoning("", cut=True, answered=False) == []


def test_short_filler_lines_do_not_count_as_recurrence():
    # "Wait." and "Hmm." recur in healthy reasoning; only long lines are compared.
    lines = []
    for i, ln in enumerate(_unique_lines(150)):
        lines += [ln, "Wait.", "Hmm, ok."] if i % 2 else [ln]
    f, = check_reasoning("\n".join(lines) + "\n", cut=True, answered=False)
    assert f.severity == Severity.PASS


def test_a_cut_last_line_is_ignored():
    from wsr.output import _rl_lines
    assert [s for _, s in _rl_lines("first line\nsecond line\nthird li")] == ["first line", "second line"]
    assert len(_rl_lines("first line\nsecond line\nthird line\n")) == 3
    assert [o for o, _ in _rl_lines("ab\n\n  - cd\n")] == [0, 4]     # offsets survive blank lines and markers


def test_onset_and_the_suggested_max_new():
    text = _looping_reasoning(head=100, block=6, turns=30)
    rl = reasoning_loop_detail(text)
    assert rl.mode == "cycle" and rl.period == 6
    head_chars = len("\n".join(_unique_lines(100))) + 1
    # Measured over 40-line stretches, so the onset is right to within a few lines of where the loop begins.
    line = 80
    assert head_chars - 10 * line <= rl.onset_char <= head_chars + 6 * line
    f, = check_reasoning(text, cut=True, answered=False, reasoning_tokens=len(text) // 4)
    ev = f.evidence
    assert ev["tokens_estimated"] is False
    assert ev["onset_tokens"] == round(rl.onset_char / (len(text) / (len(text) // 4)))
    assert ev["suggest_max_new"] % 1000 == 0 and ev["suggest_max_new"] >= ev["onset_tokens"] + 1000


def test_real_loop_onsets_are_reported():
    for t in LOOPS:
        f, = check_reasoning(t["reasoning"], cut=True, answered=False)
        assert "onset_tokens" in f.evidence and f.evidence["suggest_max_new"] >= 1000, t["name"]


def test_check_output_is_unchanged_by_the_reasoning_check():
    # The deep / program / agent classifiers use check_output: it must never emit the new code.
    codes = {f.code for f in check_output(_looping_reasoning())}
    assert "REASONING_LOOP" not in codes


# -- the response layer, every provider shape --------------------------------------------------------------

def _shapes(reasoning, content="", finish="length"):
    yield "openai", completion(content, finish, completion_tokens=8000, reasoning_content=reasoning)
    yield "openai-reasoning", completion(content, finish, completion_tokens=8000, reasoning=reasoning)
    yield "ollama", {"model": "q", "message": {"role": "assistant", "content": content, "thinking": reasoning},
                     "done": True, "done_reason": finish, "eval_count": 8000}
    yield "anthropic", {"type": "message", "role": "assistant", "stop_reason": "max_tokens" if finish == "length"
                        else "end_turn", "content": [{"type": "thinking", "thinking": reasoning}] +
                        ([{"type": "text", "text": content}] if content else []),
                        "usage": {"input_tokens": 10, "output_tokens": 8000}}


@pytest.mark.parametrize("shape", ["openai", "openai-reasoning", "ollama", "anthropic"])
def test_check_response_judges_the_reasoning_in_every_shape(shape):
    raw = dict(_shapes(_looping_reasoning()))[shape]
    _, fs = check_response(raw, max_tokens=8000)
    codes = {f.code: f for f in fs}
    assert codes["REASONING_LOOP"].severity == Severity.FAIL, shape
    assert codes["REASONING_EXHAUSTED"].failed, shape


def test_a_healthy_thinking_reply_gains_no_failure():
    raw = completion("The answer is 42.", "stop", completion_tokens=900,
                     reasoning_content="\n".join(_unique_lines(200)) + "\n")
    _, fs = check_response(raw, max_tokens=8000)
    assert not [f for f in fs if f.failed]
    assert [f.severity for f in fs if f.code == "REASONING_LOOP"] == [Severity.PASS]


def test_a_reply_without_reasoning_is_untouched():
    _, fs = check_response(completion("Hello there.", "stop"), max_tokens=100)
    assert "REASONING_LOOP" not in {f.code for f in fs}


# -- the verdict -----------------------------------------------------------------------------------------

def _loop_finding():
    return Finding("REASONING_LOOP", Layer.OUTPUT, Severity.FAIL, "loop", cause=Cause.SYMPTOM)


def test_reasoning_loop_is_its_own_group():
    assert GROUPS["reasoning_loop"] == {"REASONING_LOOP"}
    assert "REASONING_LOOP" not in GROUPS["loop"]


def test_reasoning_loop_causes_are_answer_loop_causes():
    for code in REASONING_LOOP_CAUSES:
        assert {"loop", "reasoning_loop"} <= EXPLAINS[code], code


@pytest.mark.parametrize("code,layer", [("EOS_MISMATCH", Layer.CONFIG), ("NO_STOP_TOKENS", Layer.CONFIG),
                                        ("STOP_ID_OUT_OF_RANGE", Layer.CONFIG), ("SERVER_NO_BOS", Layer.PROBE),
                                        ("STOP_PROBE_RUNON", Layer.PROBE)])
def test_end_of_turn_causes_do_not_explain_a_reasoning_loop(code, layer):
    v = decide([_loop_finding(), Finding(code, layer, Severity.WARN, "x", cause=Cause.CONFIG)])
    assert v["verdict"] == "unexplained" and v["primary"] is None
    # ...while they still explain an answer loop, as before.
    v = decide([Finding("TAIL_LOOP", Layer.OUTPUT, Severity.FAIL, "loop", cause=Cause.SYMPTOM),
                Finding(code, layer, Severity.WARN, "x", cause=Cause.CONFIG)])
    assert v["primary"]["code"] == code


def test_reasoning_runaway_explains_a_reasoning_loop():
    v = decide([_loop_finding(), Finding("REASONING_RUNAWAY", Layer.PROBE, Severity.FAIL, "x", cause=Cause.REQUEST)])
    assert v["verdict"] == "request" and v["primary"]["code"] == "REASONING_RUNAWAY"


def test_a_thinking_off_flip_explains_a_reasoning_loop():
    flip = Finding("DIFF_FLIP:thinking_off", Layer.PROBE, Severity.FAIL, "x", cause=Cause.REQUEST,
                   evidence={"explains": ["blank", "reasoning_loop"]})
    v = decide([_loop_finding(), Finding("REASONING_EXHAUSTED", Layer.RESPONSE, Severity.FAIL, "x",
                                         cause=Cause.REQUEST), flip])
    assert v["primary"]["code"] == "DIFF_FLIP:thinking_off" and not v["unexplained_groups"]


# -- the pipeline ----------------------------------------------------------------------------------------

def _offline(reasoning, model_name=""):
    n = max(1, len(reasoning) // 3)          # a reply cut at the cap: every completion token is reasoning
    raw = completion("", "length", completion_tokens=n, reasoning_content=reasoning)
    return triage(TriageInput(messages=[{"role": "user", "content": "Solve it."}],
                              params={"max_tokens": n, "chat_template_kwargs": {"enable_thinking": True}},
                              response=raw, model_name=model_name))


def test_an_implausible_token_count_falls_back_to_an_estimate():
    text = _looping_reasoning()
    f, = check_reasoning(text, cut=True, answered=False, reasoning_tokens=len(text) * 3)   # 0.33 chars/token
    g, = check_reasoning(text, cut=True, answered=False)
    assert f.evidence["tokens_estimated"] is True
    assert f.evidence["onset_tokens"] == g.evidence["onset_tokens"]


def test_offline_loop_verdict_rewords_the_cap_fix_and_adds_the_deep_step():
    rep = _offline(LOOPS[0]["reasoning"])
    assert rep.verdict["verdict"] == "request"
    assert rep.verdict["primary"]["code"] == "REASONING_EXHAUSTED"
    ex = next(f for f in rep.findings if f.code == "REASONING_EXHAUSTED")
    assert "will not help" in ex.fix and "see REASONING_LOOP" in ex.message
    assert "reasoning_loop" in rep.verdict["symptom_groups"]
    steps = rep.verdict["next_steps"]
    assert any("--differential" in s and "thinking off" in s for s in steps)
    deep = steps[-1]
    assert "wsr deep" in deep and "--thinking" in deep and "--kind loop" in deep and "--max-new" in deep


def test_a_late_loop_says_the_repair_stage_cannot_judge_it():
    late = "\n".join(_unique_lines(6000)) + "\n" + _looping_reasoning(head=0, turns=40)
    rep = _offline(late)
    deep = rep.verdict["next_steps"][-1]
    n = next(f for f in rep.findings if f.code == "REASONING_LOOP").evidence["suggest_max_new"]
    assert n > REPAIR_MAX_NEW and f"--max-new {n}" in deep and f"at most {REPAIR_MAX_NEW}" in deep


def test_an_early_loop_does_not_mention_the_repair_limit():
    rep = _offline(_looping_reasoning(head=40))
    assert "at most" not in rep.verdict["next_steps"][-1]


def test_qwen35_08b_known_issue_explains_the_loop():
    rep = _offline(LOOPS[0]["reasoning"], model_name="Qwen3.5-0.8B")
    codes = {f.code for f in rep.findings}
    assert "KNOWN_ISSUE:qwen35-0.8b-thinking-loops" in codes
    assert not rep.verdict["unexplained_groups"]
    assert "KNOWN_ISSUE:qwen35-0.8b-thinking-loops" not in {f.code for f in _offline(
        LOOPS[0]["reasoning"], model_name="Qwen3.5-9B").findings}


def test_a_cap_without_a_loop_keeps_the_old_advice():
    rep = _offline("\n".join(_unique_lines(300)) + "\n")
    ex = next(f for f in rep.findings if f.code == "REASONING_EXHAUSTED")
    assert ex.fix.startswith("raise max_tokens") and "see REASONING_LOOP" not in ex.message
    assert not any("wsr deep" in s for s in rep.verdict["next_steps"])


def test_repair_limit_matches_the_playbook():
    pytest.importorskip("torch")
    from wsr.surgery.playbook import MAX_REF_NEW
    assert REPAIR_MAX_NEW == MAX_REF_NEW


# -- the differential ------------------------------------------------------------------------------------

def loop_server(delay=0.0):
    """Loops in its reasoning until the cap unless thinking is switched off."""
    def h(b):
        time.sleep(delay)
        if (b.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
            return completion("The answer is 9.", "stop", completion_tokens=40)
        return completion("", "length", completion_tokens=b.get("max_tokens") or 8000,
                          reasoning_content=_looping_reasoning())
    return h


def _ep(server, **kw):
    return Endpoint(server.url, "m", **{"budget_s": 30, "timeout": 10, **kw})


PARAMS = {"temperature": 0.6, "max_tokens": 8000, "chat_template_kwargs": {"enable_thinking": True}}
MSGS = [{"role": "user", "content": "How many divisors does 196 have?"}]


def test_thinking_off_runs_first_and_explains_the_loop(fake_server):
    s = fake_server(loop_server())
    out = differential(_ep(s), MSGS, PARAMS, {"REASONING_LOOP", "REASONING_EXHAUSTED", "EMPTY"})
    assert s.calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    flip = next(f for f in out if f.code == "DIFF_FLIP:thinking_off")
    assert set(flip.evidence["explains"]) == {"blank", "reasoning_loop"}


def test_thinking_off_keeps_its_place_for_other_failures(fake_server):
    s = fake_server(lambda b: completion("x " * 300, "stop"))
    differential(_ep(s), MSGS, PARAMS, {"TAIL_LOOP"})
    assert s.calls[-1].get("chat_template_kwargs") == {"enable_thinking": False}
    assert s.calls[0].get("chat_template_kwargs") == {"enable_thinking": True}


def test_thinking_off_is_skipped_when_thinking_is_known_off_or_already_off(fake_server):
    s = fake_server(loop_server())
    e = _ep(s)
    e.reasoning = False
    out = differential(e, MSGS, PARAMS, {"REASONING_LOOP"})
    assert not any(f.code.endswith(":thinking_off") for f in out)
    s2 = fake_server(loop_server())
    out = differential(_ep(s2), MSGS, {**PARAMS, "chat_template_kwargs": {"enable_thinking": False}},
                       {"REASONING_LOOP"})
    assert not any(f.code.endswith(":thinking_off") for f in out)


def test_live_triage_blames_thinking_for_the_loop(fake_server):
    s = fake_server(loop_server())
    rep = triage(TriageInput(endpoint=_ep(s), messages=MSGS, params=PARAMS, probes=[], run_differential=True))
    assert rep.verdict["verdict"] == "request"
    assert rep.verdict["primary"]["code"] == "DIFF_FLIP:thinking_off"
    assert not rep.verdict["unexplained_groups"]
    # Found live: the flip's stock fix ended "or raise max_tokens", which a loop makes wrong.
    assert "raise max_tokens" not in rep.verdict["primary"]["fix"]
    assert "more tokens will not help" in rep.verdict["next_steps"][0]


def test_a_thinking_off_flip_for_a_plain_blank_keeps_its_fix(fake_server):
    from conftest import completion as c

    def h(b):   # thinks past the cap without looping, answers with thinking off
        if (b.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
            return c("196 has nine positive divisors.", "stop", completion_tokens=12)
        return c("", "length", completion_tokens=b.get("max_tokens") or 500,
                 reasoning_content="\n".join(_unique_lines(200)) + "\n")
    rep = triage(TriageInput(endpoint=_ep(fake_server(h)), messages=MSGS, params=PARAMS, probes=[],
                             run_differential=True))
    assert rep.verdict["primary"]["code"] == "DIFF_FLIP:thinking_off"
    assert "raise max_tokens" in rep.verdict["primary"]["fix"]


def test_a_budget_that_runs_out_is_reported(fake_server):
    s = fake_server(loop_server(delay=0.6))
    e = _ep(s, budget_s=1.0, timeout=10)
    out = differential(e, MSGS, PARAMS, {"REASONING_LOOP", "REASONING_EXHAUSTED"})
    inc = [f for f in out if f.code == "DIFF_INCOMPLETE"]
    assert len(inc) == 1 and inc[0].severity == Severity.WARN and inc[0].cause == Cause.NONE
    assert "1 s" in inc[0].message and inc[0].evidence["not_run"]
    assert "--budget" in inc[0].fix


def test_no_budget_line_when_everything_ran(fake_server):
    out = differential(_ep(fake_server(loop_server())), MSGS, PARAMS, {"REASONING_LOOP"})
    assert not any(f.code == "DIFF_INCOMPLETE" for f in out)


# -- the BOS probe ---------------------------------------------------------------------------------------

class _BosEndpoint(Endpoint):
    def __init__(self, model, lead, path=""):
        super().__init__("http://127.0.0.1:1", model)
        self._lead, self._path = lead, path

    def detect_backend(self):
        return "llama-server"

    def props(self):
        return {"bos_token": ",", "model_path": self._path}

    def _req(self, method, path, body=None, timeout=None):
        if path == "/apply-template":
            return {"prompt": "<|im_start|>user\nSay hello.<|im_end|>\n"}
        return {"tokens": [{"id": 11, "piece": ","}] * self._lead + [{"id": 1, "piece": "<|im_start|>"}]}


@pytest.mark.parametrize("model,path", [("Qwen3.5-0.8B", ""), ("local", "/models/qwen2.5-1.5b-instruct-q4.gguf"),
                                        ("QwQ-32B", "")])
def test_no_bos_is_expected_on_qwen(model, path):
    f, = probe_bos(_BosEndpoint(model, 0, path))
    assert f.code == "SERVER_NO_BOS" and f.severity == Severity.INFO and not f.failed


def test_no_bos_still_warns_on_other_models():
    f, = probe_bos(_BosEndpoint("llama-3.2-1b", 0, "/models/Llama-3.2-1B.gguf"))
    assert f.code == "SERVER_NO_BOS" and f.severity == Severity.WARN and f.cause == Cause.CONFIG


def test_double_bos_is_unchanged_on_qwen():
    f, = probe_bos(_BosEndpoint("Qwen3.5-0.8B", 2))
    assert f.code == "SERVER_DOUBLE_BOS" and f.severity == Severity.FAIL


# -- deep --thinking, and the detector grader ------------------------------------------------------------

class _Tok:
    def __init__(self, honours=True):
        self.chat_template = "x"
        self.honours = honours

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        think = kw.get("enable_thinking") if self.honours else None
        return "<think>\n" if think else "<think>\n\n</think>\n\n"


class _LM:
    def __init__(self, tok):
        self.tok = tok


def test_thinking_flags_map_to_template_kwargs():
    from argparse import Namespace

    from wsr.cli import _thinking_kwargs
    assert _thinking_kwargs(Namespace(thinking=True, no_thinking=False)) == {"enable_thinking": True}
    assert _thinking_kwargs(Namespace(thinking=False, no_thinking=True)) == {"enable_thinking": False}
    assert _thinking_kwargs(Namespace(thinking=False, no_thinking=False)) is None
    assert _thinking_kwargs(Namespace()) is None


def test_thinking_and_no_thinking_cannot_be_combined():
    from wsr.cli import build_parser
    for cmd in ("deep", "repair"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([cmd, "m", "--thinking", "--no-thinking", "--out", "o"])
        assert build_parser().parse_args([cmd, "m", "--thinking", "--out", "o"]).thinking


def test_a_template_that_ignores_the_switch_is_reported(capsys):
    from wsr.cli import _warn_if_switch_ignored
    msgs = [{"role": "user", "content": "hi"}]
    assert not _warn_if_switch_ignored(_LM(_Tok(honours=True)), msgs, {"enable_thinking": True})
    assert _warn_if_switch_ignored(_LM(_Tok(honours=False)), msgs, {"enable_thinking": True})
    assert "--thinking changes nothing" in capsys.readouterr().err
    assert _warn_if_switch_ignored(_LM(_Tok(honours=False)), msgs, {"enable_thinking": False})
    assert "--no-thinking changes nothing" in capsys.readouterr().err
    assert not _warn_if_switch_ignored(_LM(_Tok(honours=False)), msgs, None)


def test_the_real_qwen35_template_honours_thinking():
    tf = pytest.importorskip("transformers")
    p = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
         "2fc06364715b967f1860aea9cf38778875588b17")
    if not (p / "chat_template.jinja").exists():
        pytest.skip("Qwen3.5-0.8B is not in the local Hugging Face cache")
    from wsr.cli import _warn_if_switch_ignored
    tok = tf.AutoTokenizer.from_pretrained(str(p))
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    assert tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True).endswith("<think>\n")
    assert tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True).endswith("</think>\n\n")
    assert not _warn_if_switch_ignored(_LM(tok), msgs, {"enable_thinking": True})


def test_detector_grader_rejects_reasoning_loop():
    from wsr.program.graders import check_name
    check_name("detector:loop")
    check_name("detector:any")
    with pytest.raises(ValueError, match="detector:any or one of") as e:
        check_name("detector:reasoning_loop")
    assert "reasoning_loop" not in str(e.value)
