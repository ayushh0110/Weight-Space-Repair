"""Detectors on real Qwen2.5-0.5B-Instruct replies (tests/data/qwen05_scan.json)."""
import json
from pathlib import Path

from wsr.output import check_output, detect_language, list_loop_detail

ROWS = json.loads((Path(__file__).parent / "data" / "qwen05_scan.json").read_text(encoding="utf-8"))


def row(prefix):
    return next(r for r in ROWS if r["request"].startswith(prefix))


def failed(r, **kw):
    return {f.code for f in check_output(r["reply"], expect_lang=r["lang"], request=r["request"], **kw)
            if f.failed} - {"TRUNCATED", "BROKEN_UTF8"}


def test_urdu_reply_is_urdu_not_persian():
    r = row("آسمان نیلا")
    assert detect_language(r["reply"])[0] == "ur"
    assert "LANG_MISMATCH" not in failed(r)


def test_persian_stays_persian():
    assert detect_language(row("چرا آسمان")["reply"])[0] == "fa"


def test_real_wrong_language_replies():
    for prefix in ("Чому небо", "ทำไมท้องฟ้า", "Answer in French", "मुझे पायथन", "आकाश निळे", "आकाश किन"):
        assert "LANG_MISMATCH" in failed(row(prefix)), prefix


def test_stuck_numbered_list_is_a_loop():
    r = row("List 50 animals")
    ll = list_loop_detail(r["reply"])
    assert ll.period == 1 and ll.repeats >= 5
    assert "TAIL_LOOP" in failed(r)


def test_healthy_lists_and_tables_are_not_loops():
    for prefix in ("Write a long list of fruits", "List all prime numbers", "Write a table of the squares",
                   "Count from 1 to 200", "Generate 30 random"):
        assert "TAIL_LOOP" not in failed(row(prefix)), prefix


def test_requested_repetition_is_not_a_failure_but_unrequested_is():
    r = row("Write the word 'banana'")
    assert not failed(r) & {"TAIL_LOOP", "REPETITION", "LOW_ENTROPY"}
    assert "TAIL_LOOP" in {f.code for f in check_output(r["reply"]) if f.failed}     # without the request


def test_list_cut_right_after_the_number_is_still_a_loop():
    """Generation that stops at "25." used to hide the loop: the bare marker broke the repeat."""
    items = ["1. Elephant", "2. Giraffe", "3. Lion", "4. Tiger", "5. Penguin Penguin"] + \
        [f"{i}. Penguin" for i in range(6, 20)]
    for tail in ("20.", "20. Peng", ""):
        text = "\n".join(items + ([tail] if tail else []))
        ll = list_loop_detail(text)
        assert ll.period == 1 and ll.repeats >= 10, tail
