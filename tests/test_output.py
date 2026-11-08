import pytest

from wsr.output import (check_output, compression_ratio, periodic_detail, repetition_score, tail_loop_detail,
                        truncation_score)

HEALTHY = ("Connection pooling keeps a set of open database connections ready for reuse. Instead of opening a "
           "new socket for every request, the application borrows one from the pool and returns it afterwards, "
           "which cuts latency and load on the server.")


def failing(text, **kw):
    return {f.code for f in check_output(text, **kw) if f.failed}


def test_healthy_prose_passes_everything():
    assert failing(HEALTHY) == set()


@pytest.mark.parametrize("text", ["", "   ", "```\n```", "{}", "..."])
def test_empty(text):
    assert failing(text) == {"EMPTY"}


def test_word_loop():
    text = "Here are the steps. " + "I will now continue with the next step. " * 25
    assert {"REPETITION", "TAIL_LOOP"} <= failing(text)


def test_good_start_then_tail_loop():
    text = HEALTHY + " " + "and so on " * 40
    f = {x.code: x for x in check_output(text)}
    assert f["TAIL_LOOP"].failed
    assert f["TAIL_LOOP"].evidence["mode"] == "word"


def test_cjk_loop_uses_char_mode():
    text = "好的我来回答这个问题" + "我们需要继续下去" * 40
    _, mode = tail_loop_detail(text)
    assert mode == "char"
    assert "TAIL_LOOP" in failing(text)


def test_periodicity_needs_exact_repeats():
    assert periodic_detail(list("abcabcabc"), 10, 3).period == 3
    assert periodic_detail(list("abcabdabc"), 10, 3).score == 0


def test_compression_ratio_orders_redundancy():
    assert compression_ratio("x" * 500) < 0.05
    assert compression_ratio(HEALTHY) > 0.5


def test_symbol_soup():
    assert "SYMBOL_SOUP" in failing("G" * 40)
    assert "SYMBOL_SOUP" in failing("!!!! #### $$$$ %%%% &&&& **** @@@@ ^^^^")
    assert "SYMBOL_SOUP" not in failing("------------------------------\nA table header line follows.")


def test_control_token_leak_and_invented_turn():
    assert {"CONTROL_TOKEN_LEAK", "INVENTED_TURN"} <= failing(
        "Hello! How can I help?<|im_end|>\nuser: tell me a joke please.")


def test_reasoning_leak():
    assert "REASONING_LEAK" in failing("<think>the user wants a number</think>The answer is 4.")


def test_refusal():
    assert "REFUSAL_TEXT" in failing("I'm sorry, but I can't help with that request.")
    assert "REFUSAL_TEXT" not in failing(HEALTHY)


def test_truncation():
    assert truncation_score("text", "length") == 1.0
    assert truncation_score("```python\nprint(1)") == 0.9
    assert truncation_score('{"a": "b { c"}') == 0.0
    assert "TRUNCATED" in failing("The answer has three parts: first", finish_reason="length")


def test_json():
    assert "INVALID_JSON" in failing("{not json", expect_json=True)
    assert "INVALID_JSON" not in failing('{"a": 1}', expect_json=True, preset="strictJson")
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    assert "INVALID_JSON" in failing('{"a": "x"}', json_schema=schema, preset="strictJson")


def test_language_and_script():
    es = ("La conexión a la base de datos se mantiene abierta para que la aplicación pueda reutilizarla en "
          "lugar de abrir una nueva cada vez, lo que reduce la latencia y la carga del servidor de forma notable.")
    assert "LANG_MISMATCH" in failing(es, expect_lang="en")
    assert "LANG_MISMATCH" not in failing(es, expect_lang="es")
    assert "SCRIPT_MISMATCH" in failing("这是一个关于数据库连接池的回答，它可以减少延迟。", expect_script="latin")
    assert "SCRIPT_MISMATCH" not in failing(HEALTHY, expect_script="latin")


def test_prompt_echo():
    prompt = HEALTHY * 2
    assert "PROMPT_ECHO" in failing(HEALTHY + " " + HEALTHY, prompt=prompt)
    assert "PROMPT_ECHO" not in failing(HEALTHY, prompt="Explain how a hash map works in detail, please, "
                                        "with examples of collisions and resizing behaviour over time.")


def test_agent_loop():
    turn = {"role": "assistant", "tool_calls": [{"function": {"name": "read", "arguments": '{"path": "a.py"}'}}]}
    turns = [{"role": "user", "content": "fix it"}] + [turn, {"role": "tool", "content": "..."}] * 4
    assert "AGENT_LOOP" in failing("still working", turns=turns, preset="lenient")


def test_repetition_abstains_on_short_text():
    assert repetition_score("a a a") == 0


def test_refusal_with_a_typographic_apostrophe_is_caught():
    """Llama 3 writes "I can’t" (U+2019); the ASCII-only pattern missed it."""
    from wsr.output import refusal_score
    from wsr.media import image_not_seen_score
    assert refusal_score("I can’t provide instructions on how to hack your own router.") == 1.0
    assert refusal_score("I can't provide instructions on how to hack your own router.") == 1.0
    assert image_not_seen_score("I can’t see the image you attached.") == 1.0
