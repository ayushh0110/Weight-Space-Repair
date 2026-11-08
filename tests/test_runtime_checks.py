"""Field-reported runtime / conversion failures (runtime_checks.py)."""
from wsr.runtime_checks import (check_context_shift, check_gguf_conversion, check_hf_dtype, check_launch,
                                check_slots, ollama_env, parse_server_args)


def codes(fs):
    return {f.code for f in fs}


def test_pretokenizer_missing_or_default_is_flagged():
    assert "PRETOKENIZER_DEFAULT" in codes(check_gguf_conversion({"tokenizer.ggml.model": "gpt2"}, "llama", "Q4_0"))
    assert "PRETOKENIZER_DEFAULT" in codes(check_gguf_conversion(
        {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "default"}, "llama", "Q4_0"))
    assert not check_gguf_conversion({"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2"}, "qwen2", "Q4_0")
    assert not check_gguf_conversion({"tokenizer.ggml.model": "llama"}, "llama", "Q4_0")   # SentencePiece


def test_gemma2_softcap_and_fp16_risk():
    assert "SOFTCAP_MISSING" in codes(check_gguf_conversion({"tokenizer.ggml.model": "llama"}, "gemma2", "Q8_0"))
    ok = {"tokenizer.ggml.model": "llama", "gemma2.attn_logit_softcapping": 50.0,
          "gemma2.final_logit_softcapping": 30.0}
    assert not check_gguf_conversion(ok, "gemma2", "Q8_0")
    assert "FP16_OVERFLOW_RISK" in codes(check_gguf_conversion(ok, "gemma2", "F16"))
    cfg = {"model_type": "gemma3", "torch_dtype": "bfloat16"}
    assert "FP16_OVERFLOW_RISK" in codes(check_hf_dtype(cfg, "half"))
    assert not check_hf_dtype(cfg, "bfloat16")
    assert not check_hf_dtype({"model_type": "qwen2", "torch_dtype": "bfloat16"}, "half")


def test_launch_flags():
    f = parse_server_args("llama-server -m x.gguf -c 8192 -np 4 -ctk q4_0 -ctv q8_0 -fa on --context-shift "
                          "--rope-freq-base 500000")
    assert f["cache_k"] == "q4_0" and f["cache_v"] == "q8_0" and f["parallel"] == "4" and f["ctx"] == "8192"
    fs = {x.code: x for x in check_launch(f, ctx_trained=32768, rope_freq_base_file=1000000.0)}
    assert {"KV_CACHE_QUANTISED", "FLASH_ATTENTION_ON", "CONTEXT_SPLIT_ACROSS_SLOTS", "CONTEXT_SHIFT_ON",
            "ROPE_OVERRIDE_DIFFERS"} <= set(fs)
    assert fs["CONTEXT_SPLIT_ACROSS_SLOTS"].evidence["per_slot"] == 2048
    assert not check_launch(parse_server_args("llama-server -m x.gguf -c 8192 -ctk f16"))
    assert parse_server_args("--flash-attn -c 4096")["flash_attn"] == "on"          # bare flag (older builds)


def test_ollama_env():
    f = ollama_env({"OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NUM_PARALLEL": "2",
                    "OLLAMA_CONTEXT_LENGTH": "4096"})
    assert f == {"cache_k": "q4_0", "cache_v": "q4_0", "flash_attn": "on", "parallel": "2", "ctx": "4096"}


def test_slots_and_context_shift():
    props = {"default_generation_settings": {"n_ctx": 512}, "total_slots": 4}
    s = check_slots(props, ctx_trained=32768)
    assert s and s[0].code == "SERVER_SLOTS" and s[0].evidence["n_ctx_slot"] == 512
    assert check_context_shift(400, 200, 512)[0].code == "CONTEXT_SHIFTED"
    assert not check_context_shift(300, 100, 512)
    assert not check_context_shift(None, 100, 512)


def test_field_causes_can_win_the_verdict():
    from wsr.findings import Cause, Finding, Layer, Severity
    from wsr.verdict import decide
    lang = Finding("LANG_MISMATCH", Layer.OUTPUT, Severity.FAIL, "reads as ru", cause=Cause.SYMPTOM)
    v = decide([lang, *check_context_shift(1900, 300, 2048)])
    assert v["verdict"] == "runtime" and v["primary"]["code"] == "CONTEXT_SHIFTED"
    garbage = Finding("SYMBOL_SOUP", Layer.OUTPUT, Severity.FAIL, "soup", cause=Cause.SYMPTOM)
    v = decide([garbage, *check_launch(parse_server_args("-ctk q4_0"))])
    assert v["primary"]["code"] == "KV_CACHE_QUANTISED"
    # A slot split alone explains nothing: no request reached the slot.
    v = decide([lang, *check_launch(parse_server_args("-c 4096 -np 4"))])
    assert v["verdict"] == "unexplained"


def test_load_dependent_output_explains_only_a_failure_that_changes_with_load():
    from wsr.findings import Cause, Finding, Layer, Severity
    from wsr.verdict import decide
    blank = Finding("EMPTY", Layer.OUTPUT, Severity.FAIL, "empty", cause=Cause.SYMPTOM)
    warn = Finding("LOAD_DEPENDENT_OUTPUT", Layer.PROBE, Severity.WARN, "text differs", cause=Cause.RUNTIME)
    assert decide([blank, warn])["verdict"] == "unexplained"
    fail = Finding("LOAD_DEPENDENT_OUTPUT", Layer.PROBE, Severity.FAIL, "blank only under load", cause=Cause.RUNTIME,
                   evidence={"explains": ["blank"]})
    assert decide([blank, fail])["primary"]["code"] == "LOAD_DEPENDENT_OUTPUT"


def test_zero_completion_tokens_is_a_count_not_unknown():
    from wsr.response import normalize
    n = normalize({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 0}})
    assert n.completion_tokens == 0
