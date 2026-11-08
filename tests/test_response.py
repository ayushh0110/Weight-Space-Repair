from conftest import completion

from wsr.response import check_response, normalize


def codes(raw, **kw):
    return {f.code for f in check_response(raw, **kw)[1] if f.failed}


def test_ok():
    assert codes(completion("Hello there.")) == set()


def test_truncated_and_near_limit():
    assert codes(completion("cut", "length")) == {"TRUNCATED"}
    assert codes(completion("x", "stop", completion_tokens=98), max_tokens=100) == {"NEAR_LIMIT"}


def test_native_finish_reason_beats_normalised_stop():
    raw = completion("cut")
    raw["choices"][0]["native_finish_reason"] = "max_tokens"
    assert codes(raw) == {"TRUNCATED"}


def test_reasoning_exhausted():
    raw = completion("", "length", reasoning_content="thinking...")
    raw["usage"]["completion_tokens_details"] = {"reasoning_tokens": 500}
    assert codes(raw) == {"REASONING_EXHAUSTED"}


def test_empty_and_missing_finish():
    assert codes(completion("  ", "stop")) == {"EMPTY_OUTPUT"}
    assert codes(completion("hi", None)) == {"MISSING_FINISH_REASON"}
    assert codes({"foo": 1}) == {"MALFORMED_RESPONSE"}


def test_filter_and_refusal():
    assert codes(completion("", "content_filter")) == {"CONTENT_FILTERED"}
    assert codes(completion("", "stop", refusal="no")) == {"REFUSAL"}


def test_error_bodies():
    assert codes({"error": {"message": "Rate limit exceeded", "code": 429}}) == {"RATE_LIMITED"}
    assert codes({"error": {"message": "maximum context length exceeded"}}) == {"CONTEXT_OVERFLOW"}


def test_anthropic_shape():
    raw = {"type": "message", "content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens",
           "usage": {"input_tokens": 5, "output_tokens": 100}}
    assert normalize(raw).provider == "anthropic"
    assert codes(raw) == {"TRUNCATED"}


def test_gemini_shapes():
    raw = {"candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": "SAFETY"}]}
    assert codes(raw) == {"CONTENT_FILTERED"}
    assert codes({"promptFeedback": {"blockReason": "SAFETY"}}) == {"CONTENT_FILTERED"}


def test_ollama_native_and_input_truncation():
    raw = {"model": "m", "message": {"role": "assistant", "content": "ok"}, "done": True, "done_reason": "stop",
           "prompt_eval_count": 2048, "eval_count": 3}
    assert normalize(raw).provider == "ollama"
    assert codes(raw, sent_prompt_tokens=6000) == {"INPUT_TRUNCATED"}


def test_tool_calls():
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}}}}]

    def call(name, args):
        return completion(None, "tool_calls", tool_calls=[{"function": {"name": name, "arguments": args}}])

    assert codes(call("get_weather", '{"city": "Paris"}'), tools=tools) == set()
    assert codes(call("get_weather", '{"town": "Paris"}'), tools=tools) == {"TOOL_ARGS_INVALID"}
    assert codes(call("get_weather", '{"city": "Par'), tools=tools) == {"TOOL_ARGS_UNPARSEABLE"}
    assert codes(call("notify:notify", "{}"), tools=tools) == {"TOOL_UNKNOWN"}
    assert codes(completion(None, "length", tool_calls=[{"function": {"name": "x", "arguments": "{"}}]),
                 tools=tools) >= {"TOOL_CALL_TRUNCATED"}
