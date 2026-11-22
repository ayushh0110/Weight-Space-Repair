"""Malformed input: replies, requests and files that are not what a server or a user should send.

check_response promises never to raise, and triage reads files people write by hand, so a wrong type in any
field must give a finding or a clear error (exit 2), never a traceback.
"""
import json

import pytest

from wsr.cli import main
from wsr.pipeline import TriageInput, triage
from wsr.response import _int, check_response

LOOP = "\n".join([f"Step {i}: unique line number {i} here." for i in range(60)]
                 + ["Wait, let me re-check the sign of the term."] * 80)
ODD = [None, 5, 1.5, True, ["a", 5, None, {"type": "text", "text": 3}], {"x": 1}, "", LOOP]


def _shapes(v):
    return [
        {"choices": [{"index": 0, "finish_reason": "length", "message": {
            "role": "assistant", "content": v, "reasoning_content": v, "tool_calls": [v, {"function": v}],
            "function_call": v}}], "usage": {"completion_tokens": v, "completion_tokens_details": v}},
        {"model": "m", "message": {"role": "assistant", "content": v, "thinking": v, "tool_calls": [v]},
         "done": True, "done_reason": "length", "eval_count": v},
        {"model": "m", "response": v, "thinking": v, "done": True},
        {"type": "message", "stop_reason": "max_tokens", "usage": v,
         "content": [v, {"type": "thinking", "thinking": v}, {"type": "text", "text": v}]},
        {"candidates": [{"content": {"parts": [v, {"thought": True, "text": v}, {"text": v}, {"functionCall": v}]}}],
         "usageMetadata": v, "promptFeedback": v},
        {"candidates": v}, {"candidates": [v]},
    ]


@pytest.mark.parametrize("v", ODD, ids=repr)
def test_check_response_never_raises_on_odd_fields(v):
    for shape in _shapes(v):
        for cap in (None, 8000, "8000", float("nan")):
            n, _ = check_response(shape, max_tokens=cap)
            assert isinstance(n.text, str) and isinstance(n.reasoning_text, str)


def test_a_reasoning_loop_in_an_ollama_thinking_list_is_still_read():
    n, fs = check_response({"model": "m", "message": {"role": "assistant", "content": "", "thinking": [LOOP + "\n"]},
                            "done": True, "done_reason": "length", "eval_count": 8000})
    assert n.reasoning_text and any(f.code == "REASONING_LOOP" and f.failed for f in fs)


def test_int_takes_only_whole_numbers():
    assert _int(8000.0) == 8000 and _int(7) == 7
    assert _int(float("nan")) is None and _int(float("inf")) is None and _int(True) is None and _int("8") is None


@pytest.mark.parametrize("key", ["temperature", "options", "max_tokens", "stop", "seed", "chat_template_kwargs"])
@pytest.mark.parametrize("v", ["0.6", "x", [], {}, None, float("nan"), True], ids=repr)
def test_offline_triage_takes_odd_request_values(key, v):
    raw = {"choices": [{"index": 0, "finish_reason": "length",
                        "message": {"role": "assistant", "content": "", "reasoning_content": LOOP + "\n"}}]}
    rep = triage(TriageInput(response=raw, params={key: v}, messages=[{"role": "user", "content": "hi"}], probes=[]))
    assert rep.verdict["verdict"]


@pytest.mark.parametrize("content, flag, why", [
    (["hi"], "--messages", "message 0 is not an object with a role"),
    ([{"content": "hi"}], "--messages", "message 0 is not an object with a role"),
    ([1, 2], "--request", "expected a JSON object"),
])
def test_cli_rejects_malformed_files_with_a_clear_error(tmp_path, capsys, content, flag, why):
    f = tmp_path / "x.json"
    f.write_text(json.dumps(content), encoding="utf-8")
    code = main(["triage", "--response", "examples/blank_reply/response.json", flag, str(f)])
    assert code == 2 and why in capsys.readouterr().err


def test_cli_names_a_file_that_is_not_utf8(tmp_path, capsys):
    f = tmp_path / "x.json"
    f.write_bytes(b"\xff\xfe\x00bad")
    assert main(["triage", "--response", str(f)]) == 2
    assert "x.json: not UTF-8 text" in capsys.readouterr().err


def test_an_empty_messages_array_still_means_no_prompt(tmp_path):
    f = tmp_path / "m.json"
    f.write_text("[]", encoding="utf-8")
    assert main(["triage", "--response", "examples/blank_reply/response.json", "--messages", str(f)]) == 1


def test_a_stream_that_keeps_sending_stops_at_the_budget():
    # The socket timeout only fires on silence; a server streaming without pause must still stop at --budget.
    import contextlib
    import time

    from wsr.probes import Endpoint, EndpointError

    ep = Endpoint("http://127.0.0.1:1", "m", budget_s=60)
    ep.reachable = lambda: None

    def lines():
        yield b'data: {"choices": [{"delta": {"content": "x"}}]}\n'
        ep._deadline = time.monotonic() - 1          # the budget runs out while tokens are still arriving
        while True:
            yield b'data: {"choices": [{"delta": {"content": "x"}}]}\n'

    ep._open = lambda *a: contextlib.nullcontext(lines())
    with pytest.raises(EndpointError, match="time budget ran out mid-stream"):
        ep.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8)
