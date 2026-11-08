"""Regressions found running wsr against a live llama-server (Gemma 4 E2B Q4_0, thinking on)."""
import json

import pytest
from conftest import completion, write_gguf

from wsr.config_checks import kv_cache_bytes, load_gguf
from wsr.known_issues import match
from wsr.output import check_output
from wsr.pipeline import TriageInput, triage
from wsr.probes import Endpoint, run_probes
from wsr.request_checks import check_request
from wsr.stream import assemble, looks_like_stream


def ep(server):
    return Endpoint(server.url, "m", budget_s=30, timeout=10)


def thinking_server(answer="OK", think_tokens=120):
    """Reasons for think_tokens before answering unless thinking is switched off."""
    def h(b):
        off = (b.get("chat_template_kwargs") or {}).get("enable_thinking") is False
        cap = b.get("max_tokens") or 10_000
        if off:
            return completion(answer, "stop", completion_tokens=2)
        if cap <= think_tokens:
            return completion("", "length", completion_tokens=cap, reasoning_content="Thinking Process: ...")
        return completion(answer, "stop", completion_tokens=think_tokens + 2,
                          reasoning_content="Thinking Process: the answer is short.")
    return h


def test_reasoning_probe_finds_starved_answer_and_switch(fake_server):
    fs = {f.code: f for f in run_probes(ep(fake_server(thinking_server())), ["reasoning"])}
    assert "THINKING_ON" in fs
    assert fs["REASONING_STARVES_ANSWER"].failed
    assert fs["THINKING_SWITCH_OK"].severity.value == "pass"


def test_stop_probe_is_not_fooled_by_reasoning(fake_server):
    # Before the fix a reasoning model that spent the cap thinking was called an EOS run-on.
    fs = {f.code for f in run_probes(ep(fake_server(thinking_server())), ["stop"])}
    assert "STOP_PROBE_RUNON" not in fs and "STOP_PROBE_OK" in fs


def test_blank_from_reasoning_flips_with_thinking_off(fake_server):
    s = fake_server(thinking_server(answer='{"name": "Ada Lovelace", "age": 36}', think_tokens=600))
    rep = triage(TriageInput(endpoint=ep(s), messages=[{"role": "user", "content": "Return JSON"}],
                             params={"temperature": 0, "max_tokens": 500}, probes=[], run_differential=True,
                             expect_json=True))
    assert rep.verdict["verdict"] == "request"
    assert rep.verdict["primary"]["code"] == "DIFF_FLIP:thinking_off"


def test_unknown_probe_name_is_an_error(fake_server):
    with pytest.raises(ValueError):
        run_probes(ep(fake_server(thinking_server())), ["nope"])


# -- prefill / JSON ----------------------------------------------------------------

PREFILL = [{"role": "user", "content": "Return JSON"}, {"role": "assistant", "content": "<think>\n</think>\n"}]


def test_prefill_with_foreign_reasoning_markup():
    fs = {f.code: f for f in check_request({}, backend="llama-server", messages=PREFILL,
                                           template="{{ '<|turn>model' }}{% if enable_thinking %}<|think|>{% endif %}")}
    f = fs["PREFILL_REASONING_MARKUP"]
    assert f.failed and f.evidence["foreign_markup"]


def test_json_behind_think_block_is_wrapped_not_invalid():
    fs = {f.code: f for f in check_output('<think>\n</think>\n```json\n{"a": 1}\n```', expect_json=True)}
    assert fs["JSON_WRAPPED"].failed and "INVALID_JSON" not in fs
    fs = {f.code: f for f in check_output('```json\n{"a": 1}\n```', expect_json=True)}
    assert not fs["INVALID_JSON"].failed


def test_prefill_case_is_a_request_verdict_not_a_known_template_bug(tmp_path):
    tpl = ("{{ bos_token }}{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}"
           "{% if add_generation_prompt %}<|turn>model\n{% endif %}")
    g = write_gguf(tmp_path / "gemma-4-e2b.gguf", metadata={"general.architecture": (8, "gemma4"),
                                                           "tokenizer.chat_template": (8, tpl)})
    raw = completion('<think>\n</think>\n```json\n{"a": 1}\n```')
    rep = triage(TriageInput(model_dir=g, backend="llama-server", messages=PREFILL, params={"max_tokens": 500},
                             response=raw, expect_json=True))
    assert rep.verdict["verdict"] == "request"
    assert rep.verdict["primary"]["code"] == "PREFILL_REASONING_MARKUP"
    assert not any(f.code.startswith("KNOWN_ISSUE:gemma4-early") for f in rep.findings)


def test_known_issue_excludes():
    assert match("gemma-4", {"EMPTY"}, conditions={"gguf"})
    assert not match("gemma-4", {"EMPTY"}, conditions={"gguf", "reasoning_exhausted"})


# -- memory: sliding window + shared KV ---------------------------------------------------

def test_kv_cache_respects_swa_and_shared_layers(tmp_path):
    u32 = {"block_count": 10, "attention.head_count": 8, "attention.head_count_kv": 1, "embedding_length": 1536,
           "attention.key_length": 512, "attention.value_length": 512, "attention.key_length_swa": 256,
           "attention.value_length_swa": 256, "attention.sliding_window": 512, "attention.shared_kv_layers": 4}
    md = {"general.architecture": (8, "gemma4"), **{f"gemma4.{k}": (4, v) for k, v in u32.items()},
          "gemma4.attention.sliding_window_pattern": ("arr7", [True, True, True, True, False] * 2)}
    spec = load_gguf(write_gguf(tmp_path / "g.gguf", metadata=md))
    # 6 own-KV layers: indexes 0-3 sliding (256-dim, 512 window), 4 global (512-dim), 5 sliding.
    want = 5 * 1 * 512 * 2 * 512 + 1 * 1 * 1024 * 2 * 32768
    assert kv_cache_bytes(spec, 32768) == want


# -- language -----------------------------------------------------------------------------

@pytest.mark.parametrize("want,text,fails", [
    ("en", "编译器是一种程序，它把用高级语言写的源代码翻译成机器可以执行的代码。它会检查语法错误。", True),
    ("zh", "编译器是一种程序，它把用高级语言写的源代码翻译成机器可以执行的代码。它会检查语法错误。", False),
    ("ja", "编译器是一种程序，它把用高级语言写的源代码翻译成机器可以执行的代码。它会检查语法错误。", True),
    ("ja", "コンパイラは、高水準言語で書かれたソースコードを機械が実行できるコードに変換するプログラムです。", False),
    ("ar", "کامپایلر برنامه‌ای است که کد منبع نوشته شده به زبان سطح بالا را به کد ماشین تبدیل می‌کند.", True),
    ("hi", "कंपाइलर एक प्रोग्राम है जो उच्च स्तरीय भाषा में लिखे गए स्रोत कोड को मशीन कोड में बदलता है।", False),
    ("hi", "कंपायलर हा एक प्रोग्राम आहे जो उच्च स्तरीय भाषेतील कोड मशीन कोडमध्ये बदलतो आणि त्रुटी शोधतो.", True),
    ("ru", "Компілятор це програма, яка перекладає вихідний код у машинний код і знаходить помилки.", True),
    ("ko", "컴파일러는 고급 언어로 작성된 소스 코드를 기계가 실행할 수 있는 코드로 변환하는 프로그램입니다.", False),
])
def test_non_latin_languages(want, text, fails):
    f = next(f for f in check_output(text, expect_lang=want) if f.code == "LANG_MISMATCH")
    assert f.failed == fails, f.message


# -- streaming -----------------------------------------------------------------------------

def sse(*chunks, done=True):
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    return "".join(lines) + ("data: [DONE]\n\n" if done else "")


def delta(**d):
    fin = d.pop("finish", None)
    return {"choices": [{"index": 0, "delta": d, "finish_reason": fin}]}


def test_stream_assembles_text_and_reasoning():
    raw, fs = assemble(sse(delta(reasoning_content="hmm"), delta(content="Hel"), delta(content="lo"),
                           delta(finish="stop")).splitlines(True))
    assert not [f for f in fs if f.failed]
    m = raw["choices"][0]["message"]
    assert m["content"] == "Hello" and m["reasoning_content"] == "hmm"


def test_stream_cut_mid_answer():
    _, fs = assemble(sse(delta(content="The answer is"), done=False).splitlines(True))
    assert any(f.code == "STREAM_CUT" for f in fs)


def test_stream_tool_fragments_that_do_not_assemble():
    tc = lambda **f: delta(tool_calls=[{"index": 0, **f}])  # noqa: E731
    _, fs = assemble(sse(tc(id="a", function={"name": "get_weather", "arguments": '{"city": '}),
                         tc(function={"arguments": '"Par'}), delta(finish="tool_calls")).splitlines(True))
    assert any(f.code == "STREAM_TOOL_ARGS_FRAGMENTED" for f in fs)


def test_stream_error_event_and_malformed_chunk():
    text = sse(delta(content="x")) .replace("data: [DONE]", 'data: {"error": {"message": "Context size has been '
                                                           'exceeded."}}\n\ndata: {broken')
    _, fs = assemble(text.splitlines(True))
    codes = {f.code for f in fs}
    assert {"STREAM_ERROR_EVENT", "STREAM_MALFORMED_CHUNK"} <= codes and "STREAM_CUT" not in codes


def test_ollama_ndjson_stream():
    lines = [json.dumps({"message": {"content": "Hi"}, "done": False}) + "\n",
             json.dumps({"message": {"content": "!"}, "done": True, "done_reason": "stop", "eval_count": 2}) + "\n"]
    assert looks_like_stream("".join(lines))
    raw, fs = assemble(lines)
    assert raw["choices"][0]["message"]["content"] == "Hi!" and not [f for f in fs if f.failed]


def test_stream_probe_detects_streaming_path_bug(fake_server):
    def h(b):
        if b.get("stream"):  # streaming path loses the tool call's closing brace
            if b.get("tools"):
                return sse(delta(tool_calls=[{"index": 0, "id": "x", "function": {
                    "name": "get_weather", "arguments": '{"city": "Paris"'}}]), delta(finish="tool_calls"))
            return sse(delta(content="1, 2, 3"), delta(finish="stop"))
        if b.get("tools"):
            return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {
                    "name": "get_weather", "arguments": '{"city": "Paris"}'}}]}}]}
        return completion("1, 2, 3")

    fs = {f.code: f for f in run_probes(ep(fake_server(h)), ["stream"])}
    assert fs["STREAM_OK:text"].severity.value == "pass"
    assert fs["STREAM_TOOL_ARGS_FRAGMENTED"].failed
    assert fs["STREAM_DIFFERS:tool"].failed


def test_cli_response_accepts_a_cut_stream(tmp_path, capsys):
    from wsr.cli import main

    p = tmp_path / "cut.sse"
    p.write_text(sse(delta(content="The answer is"), done=False), encoding="utf-8")
    assert main(["response", str(p)]) == 1
    out = capsys.readouterr().out
    assert "STREAM_CUT" in out and "VERDICT: RUNTIME" in out


SALAD = ("Fractbusters lever Forgetoxone rarely Gaddherst TURN epoxy Tether prematurely repeal ultim бойول無 "
         "Merryத This challeng आरकषण develops boldnessニーカー ಮಾಡಿ tranch பேசியforgot troughs hôn mg "
         "assumptions đúngvoices произоCStringétatථ粟 examplesublic textesწყვე emoc维निंग nghĩ")


def test_token_salad_from_high_temperature_is_garbage():
    fs = {f.code: f for f in check_output(SALAD)}
    assert fs["TOKEN_SALAD"].failed
    legit = ("Pythonでプログラムを書くときは、まずvirtualenvを作ってからpipでパッケージをインストールします。"
             "Hello in other languages: Spanish hola, Russian привет, Japanese こんにちは, Chinese 你好, "
             "Arabic مرحبا, Hindi नमस्ते, Korean 안녕하세요, Greek γεια, Hebrew שלום, Thai สวัสดี.")
    assert not {f.code: f for f in check_output(legit)}["TOKEN_SALAD"].failed


def test_high_temperature_salad_is_a_sampling_verdict(fake_server):
    s = fake_server(lambda b: completion(SALAD if b.get("temperature", 0) > 1 else
                                         "A gear trades speed for force by changing how far the wheel turns.",
                                         "length" if b.get("temperature", 0) > 1 else "stop"))
    rep = triage(TriageInput(endpoint=ep(s), messages=[{"role": "user", "content": "Explain gears."}],
                             params={"temperature": 3.5, "max_tokens": 200}, probes=[], run_differential=True))
    assert rep.verdict["verdict"] == "sampling", rep.verdict
    assert rep.verdict["primary"]["code"] in ("DIFF_FLIP:greedy", "TEMPERATURE_HIGH")


def test_stop_string_cut_is_linked_to_the_stop_string(fake_server):
    s = fake_server(lambda b: completion("The sky lets loose a gentle pour,") if b.get("stop") else
                    completion("The sky lets loose a gentle pour,\nAnd rooftops drum.\nThe streets run silver.\n"
                               "Then the clouds move on."))
    rep = triage(TriageInput(endpoint=ep(s), messages=[{"role": "user", "content": "Poem, four lines."}],
                             params={"temperature": 0, "max_tokens": 200, "stop": ["\n"]}, probes=[],
                             run_differential=True))
    assert rep.verdict["verdict"] == "request"
    assert rep.verdict["primary"]["code"] in ("DIFF_FLIP:stop_off", "STOP_TOO_BROAD")


def test_server_parser_rejection_is_a_garbage_symptom(fake_server):
    err = {"error": {"code": 500, "message": "The model produced output that does not match the expected "
                                             "peg-gemma4 format", "type": "server_error"}}
    s = fake_server(lambda b: err if b.get("temperature", 0) > 1 else completion("A gear trades speed for force."))
    rep = triage(TriageInput(endpoint=ep(s), messages=[{"role": "user", "content": "Explain gears."}],
                             params={"temperature": 3.5, "max_tokens": 200}, probes=[], run_differential=True))
    assert any(f.code == "OUTPUT_PARSE_REJECTED" for f in rep.findings)
    assert rep.verdict["verdict"] == "sampling", rep.verdict


def test_cache_probe_flags_cache_dependent_reply(fake_server):
    def h(b):
        if b.get("cache_prompt") is False:
            return completion("1, 2, 3")
        r = completion("1, 2, 3", reasoning_content="Thinking...")
        r["timings"] = {"cache_n": 24}
        return r
    srv = fake_server(h)
    e = ep(srv)
    e.backend = "llama-server"
    from wsr.probes import probe_cache
    fs = {f.code for f in probe_cache(e)}
    assert "PROMPT_CACHE_CHANGES_OUTPUT" in fs


def test_flip_that_does_not_reproduce_is_not_a_cause(fake_server):
    sampled = {"n": 0}

    def h(b):  # greedy always loops; the sampled variant escapes once, then loops too
        if b.get("temperature", 0) > 0:
            sampled["n"] += 1
            if sampled["n"] == 1:
                return completion("Open the file and count the words.")
        return completion(LOOPY, "length")

    e = ep(fake_server(h))
    e.reasoning = False
    rep = triage(TriageInput(endpoint=e, messages=[{"role": "user", "content": "Count words"}],
                             params={"temperature": 0, "max_tokens": 200}, probes=[], run_differential=True))
    codes = {f.code for f in rep.findings}
    assert "DIFF_UNSTABLE:sampled" in codes and "DIFF_FLIP:sampled" not in codes


LOOPY = "Step one is to open the file. " + "Then we continue. " * 40


def _load_server(fake_server, under_load):
    """Fillers take a while; the probe request answers differently while others are in flight."""
    import threading
    import time as _t
    state = {"n": 0}
    lock = threading.Lock()

    def h(b):
        with lock:
            state["n"] += 1
            busy = state["n"] > 1
        try:
            if "500" in b["messages"][0]["content"]:
                _t.sleep(1.2)
                return completion("1, 2, 3")
            return completion(under_load if busy else "Red\nYellow\nBlue")
        finally:
            with lock:
                state["n"] -= 1
    e = ep(fake_server(h, threaded=True))
    e.reasoning = False
    return e


def test_concurrency_probe_flags_load_dependent_reply(fake_server):
    from wsr.probes import probe_concurrency
    fs = probe_concurrency(_load_server(fake_server, "Red\nGreen\nBlue"))
    assert [f.code for f in fs] == ["LOAD_DEPENDENT_OUTPUT"] and fs[0].severity.value == "warn"
    fs = probe_concurrency(_load_server(fake_server, "Red\nYellow\nBlue"))
    assert [f.code for f in fs] == ["LOAD_INVARIANT"]


def test_concurrency_probe_failure_that_comes_with_load(fake_server):
    from wsr.probes import probe_concurrency
    e = _load_server(fake_server, "")          # blank only when other requests are decoding
    fs = probe_concurrency(e, [{"role": "user", "content": "Name three primary colours."}], {"max_tokens": 64})
    assert fs[0].code == "LOAD_DEPENDENT_OUTPUT" and fs[0].failed and "depends on load" in fs[0].message
