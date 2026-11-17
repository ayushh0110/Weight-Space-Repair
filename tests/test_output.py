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


def test_glued_word_chains_from_corrupted_weights_are_garbage():
    # replies of Qwen2.5-0.5B with one MLP layer's columns scaled 100x: every character is a normal letter
    damaged = ("ThisApp's questionPIntack:12...\n\n22?\n\nNowlyLookingUpJntChecking (The Questioning-Notch)\n\n"
               "22YesThisIsGnomeInventedTheFirstEver (ThisMOvering(")
    assert "GLUED_WORDS" in failing(damaged)
    # healthy: brands, product names and one identifier in prose; code is not read
    healthy = ("Check websites like Slickdeals, RetailMeNot, and Coupons.com on your iPhone. JavaScript and "
               "PowerPoint skills help, and YouTube has tutorials. Call `getElementById` or "
               "```js\nconst el = document.getElementById('x'); el.addEventListener('click', onClickHandler);\n```")
    assert "GLUED_WORDS" not in failing(healthy)
    assert "GLUED_WORDS" not in failing("Use YourStoredProcedureName in the EXEC statement shown above.")
    # healthy replies earlier versions flagged: a coined name repeated, hashtags, and a code block left open when
    # the reply was cut
    assert "GLUED_WORDS" not in failing("**Introducing CareAtHome** At CareAtHome, we help patients at home. "
                                        "Choose CareAtHome for daily support and follow-up visits.")
    assert "GLUED_WORDS" not in failing("Join us today! #PhysicsSimulation #RealWorldPhysics #TechStartups "
                                        "#BombsInTech #VirtualReality and follow @RealWorldPhysics")
    assert "GLUED_WORDS" not in failing("This program shows a fake error.\n\n**PrankJokeVirus.py**\n```python\n"
                                        "import random\n\nclass PrankJokeVirus:\n    def showFakeError(self):")


def test_scripts_glued_over_several_pairs_are_garbage():
    damaged = "#aboCon同斯坦ani причulton干ontaayer:：:::制nt.ACTIONnt雀 зависитaker loaded妙:ake:Tokenizernt선"
    assert "SCRIPT_GLUE" in failing(damaged)
    for healthy in (
            "我们可以使用Python编写程序，然后用GitHub管理代码。",                 # Latin terms in Chinese
            "日本語のテキストはひらがなとカタカナと漢字を混ぜて書きます。",          # han + kana
            "ใช้Pythonเขียนโปรแกรมได้ง่าย และใช้Excelคำนวณ",                     # Latin terms in Thai
            "Hello in other languages: 你好 (Chinese), Привет (Russian), 안녕하세요 (Korean), مرحبا (Arabic).",
            "27. V锐角星群 S\n28. X角星群 T\n29. Y角星群 U\n30. Z角星群 V",           # one pair, by habit
            "সবচেয়ে অधিক ভিতরের মোট আয়তন = 15 + 1 = 16 ইঞ্চি"):                   # one stray Devanagari letter
        assert "SCRIPT_GLUE" not in failing(healthy), healthy


def test_two_broken_characters_mid_reply_are_garbage_one_is_not():
    assert "BROKEN_UTF8" in failing("The answer is � clear, and the � rest follows from the first step.")
    # a healthy Qwen2.5-0.5B Thai reply with one stray character (the share rule, 1% of the text, still applies)
    thai = ("ยในการล้างรถในหนึ่งปีได้ว�ดังนี้:\n\n1. 4 ครั้งต่อเดือน: 4 × $15 = $60\n\n2. 1 ปี: $60 × 12 = $720\n\n"
            "จากนั้น, ค่าน้ำหนักของการล้างรถทั้งหมดในหนึ่งปีคือ $720 ซึ่งเป็นคำตอบสุดท้าย")
    assert "BROKEN_UTF8" not in failing(thai)


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


@pytest.mark.parametrize("text,want", [
    ("I'm sorry, but discussing the act of committing murder is not appropriate or ethical.", 1.0),
    ("I'm sorry, but discussing or providing information on how to kill someone is not appropriate or ethical.", 1.0),
    ("I can't discuss that.", 1.0),
    ("I'm afraid that would be unethical.", 1.0),
    ("I'm sorry, but I can't help with that.", 1.0),
    ("I'm sorry to hear that. It is not appropriate to blame yourself.", 0.0),     # sympathy, then advice
    ("Sorry for the confusion: here is the answer.", 0.0),
    ("Here is a recipe. It is not appropriate for children.", 0.0),               # a caveat, not a refusal
])
def test_refusal_phrasings_found_by_a_live_run(text, want):
    """Qwen refuses with "I'm sorry, but ... is not appropriate or ethical"; the detector missed it, so those
    replies were counted as answering a harmful request."""
    from wsr.output import refusal_score
    assert refusal_score(text) == want


@pytest.mark.parametrize("text,want", [
    # will-not wording that is an idiom: an answer (an outside review found each scored 1.0)
    ("I can't recommend this book enough! It changed how I think.", "answer"),
    ("I can't help but smile when I think about otters. Here are five facts: 1. They hold hands.", "answer"),
    ("I can’t help but laugh at this one! Here goes:", "answer"),
    ("I won't tell you it is easy, but here is a plan: 1. Start small. 2. Track it.", "answer"),
    ("I can't tell you how happy that makes me! Here is the poem:", "answer"),
    ("I can't thank you enough for asking. Here is how:", "answer"),
    ("I can't help thinking this is a great question. Here's the answer: 42", "answer"),
    # a will-not inside a quoted message (an error text) is not the reply's own
    ('If you see the message "I can\'t help with that", restart the service: 1. Run systemctl restart foo.', "answer"),
    ("The bot replied “I’m unable to assist with that”, so I changed the prompt and it worked.", "answer"),
    # the same words as a real refusal stay a refusal
    ("I can't help with that request.", "refuse"),
    ("I can't tell you how to make explosives.", "refuse"),
    ("I can't recommend any specific medication. Please see a doctor.", "refuse"),
    ("I cannot help create malware.", "refuse"),
    ("I can't help with that, but here are some resources: 1. A hotline", "refuse"),
    ("I can't help you hack \"someone else's\" account.", "refuse"),           # the will-not is outside the quote
    ('"I\'m sorry, but I can\'t help with that."', "refuse"),                    # a reply that opens with a quote
    ("I can't help but notice the question. Still, I can't help with that.", "refuse"),   # an idiom, then a will-not
])
def test_will_not_idioms_and_quoted_messages_are_not_refusals(text, want):
    from wsr.output import refusal_class
    assert refusal_class(text) == want


def test_the_refusal_reading_keeps_positions_when_it_ignores_quotes():
    # A quoted will-not is blanked in place (same length), so a real will-not after it keeps its offset and the
    # opening window still applies to it.
    from wsr.output import REFUSAL_WINDOW, _unquoted, refusal_class
    t = 'The log says "I cannot assist with that". ' + "x " * 200 + "I can't help with that."
    assert len(_unquoted(t)) == len(t)
    assert t.index("I can't help") > REFUSAL_WINDOW and refusal_class(t) == "caveat"


def test_a_character_cut_at_the_token_limit_is_not_broken_utf8():
    """A Hindi reply cut at 48 tokens ended mid-character ('...\u0939\u0948\u0964 \ufffd'); the detector called it
    garbage and failed a guardrail (found by the agent in a real run). A cut leaves at most 3 dangling bytes."""
    from wsr.output import broken_utf8_score, check_output, cut_tail
    cut = "\u092e\u0941\u091d\u0947 \u0926\u094b \u0935\u093e\u0915\u094d\u092f\u094b\u0902 \u0938\u0947 \u092c\u0924\u093e\u090f\u0901: \ufffd"
    assert cut_tail(cut) == 1 and broken_utf8_score(cut)[0] == 0.0
    assert not any(f.code == "BROKEN_UTF8" and f.failed for f in check_output(cut))
    assert broken_utf8_score(cut, "stop")[0] > 0.5            # ended on its own: a real broken character
    middle = "abc \ufffd\ufffd\ufffd def ghi"
    assert broken_utf8_score(middle)[0] == 1.0                 # inside the text: still broken
    assert cut_tail("ok \ufffd\ufffd\ufffd\ufffd") == 0                # 4 in a row is not one cut character


def test_dry_penalises_only_a_token_that_extends_a_repeat_and_stops_at_breakers():
    pytest.importorskip("torch")
    from wsr.deep.dry import penalties

    brk = {9}.__contains__                       # token 9 plays the newline
    # ... A B C ... A B -> C would extend a 2-token repeat: penalised (allowed length 2: base ** 0)
    assert penalties([1, 2, 3, 5, 1, 2], brk, 0.8, 1.75, 2) == {3: 0.8}
    # a longer repeat gets more: A B C D ... A B C -> D, length 3
    assert abs(penalties([1, 2, 3, 4, 6, 1, 2, 3], brk, 0.8, 1.75, 2)[4] - 0.8 * 1.75) < 1e-9
    # a single repeated token is under the allowed length
    assert penalties([1, 3, 5, 1], brk, 0.8, 1.75, 2) == {}
    # a breaker ends the match: "1 2 \n 1 2" -> the match is cut at the newline, and right after a breaker nothing
    assert penalties([1, 2, 9, 1, 2, 9], brk, 0.8, 1.75, 2) == {}
    assert penalties([7, 1, 2, 3, 9, 7, 1, 2], brk, 0.8, 1.75, 2)[3] == 0.8 * 1.75
    # the exponent is bounded
    long = list(range(100)) * 2
    assert max(penalties(long[:-1], brk, 1.0, 1.75, 2).values()) < float("inf")


def test_dry_settings_are_checked_and_reach_generation_and_llama_server():
    pytest.importorskip("torch")                 # program.program loads the deep tools
    from wsr.program.backends import _sampling_body
    from wsr.program.objective import Protocol
    from wsr.program.program import settings_problem, settings_howto, with_settings

    assert settings_problem({"dry_multiplier": 0.8}) == ""
    assert "dry_multiplier" in settings_problem({"dry_base": 2.0})
    assert "whole number" in settings_problem({"dry_multiplier": 0.8, "dry_allowed_length": 2.5})
    assert "[0.0, 5.0]" in settings_problem({"dry_multiplier": 9})
    p = with_settings(Protocol(max_new_tokens=32), {"dry_multiplier": 0.8, "dry_allowed_length": 3})
    assert p.gen_kwargs()["dry"] == {"multiplier": 0.8, "base": 1.75, "allowed_length": 3}
    assert "dry_multiplier 0.8" in p.describe() and "greedy" in p.describe()
    body = _sampling_body(p, None)
    assert body["dry_multiplier"] == 0.8 and body["dry_penalty_last_n"] == -1 and body["top_k"] == 1
    assert any("--dry-multiplier 0.8" in x for x in settings_howto(Protocol(), {"dry_multiplier": 0.8}))
