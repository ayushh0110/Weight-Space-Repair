"""Runtime and conversion checks found in the field (llama.cpp / Ollama / vLLM issue trackers).

These failures look like a bad model but come from how it was converted or launched:

* GGUF without a pre-tokenizer: llama.cpp falls back to "default" BPE splitting
  and warns that generation quality will be degraded; files converted by old
  convert scripts (Llama 3 era) are the usual cause.
* Gemma-2-style logit soft-capping missing from a GGUF: the backend cannot
  apply it, and long or confident outputs degrade.
* A bfloat16-trained Gemma stored or run in float16: activations overflow
  (float16 tops out near 65k), become inf / NaN, and the reply comes out empty.
* llama-server splits --ctx-size across --parallel slots: each request gets
  ctx / slots tokens. Past the slot, the prompt is cut, or generation "context
  shifts" (drops the start) and the model loses the instructions.
* Launch flags that change the numbers: quantised KV cache (q4_0 / q8_0),
  flash attention (fp16 accumulators have overflowed in the CUDA kernel), RoPE
  overrides that disagree with the file, disabled or enabled context shift.

Sources: llama.cpp #7219 (pre-tokenizer), #11681 (ctx divided by parallel),
#17284 and #9607 (context shift), #17746 (flash-attention fp16 overflow);
transformers #36822 (Gemma 3 empty in float16); Ollama context-length docs.
"""
from __future__ import annotations

import os
import shlex
from typing import Any, Optional

from .findings import Cause, Finding, Layer, Severity

SRC = "field reports (llama.cpp, Ollama, transformers issues)"


def _f(code: str, layer: Layer, sev: Severity, msg: str, *, cause: Cause, fix: Optional[str] = None,
       **ev: Any) -> Finding:
    return Finding(code, layer, sev, msg, cause=cause, fix=fix, source=SRC,
                   evidence={k: v for k, v in ev.items() if v is not None})


# -- conversion (files) --------------------------------------------------------------------------

def check_gguf_conversion(meta: dict, arch: Optional[str], file_type: Optional[str]) -> list[Finding]:
    out: list[Finding] = []
    model = meta.get("tokenizer.ggml.model")
    pre = meta.get("tokenizer.ggml.pre")
    if model == "gpt2" and (pre is None or pre == "default"):
        out.append(_f("PRETOKENIZER_DEFAULT", Layer.CONFIG, Severity.FAIL,
                      "BPE tokenizer without a pre-tokenizer type: llama.cpp falls back to 'default' splitting "
                      "and warns that generation quality will be degraded",
                      cause=Cause.CONFIG, fix="re-convert with a current llama.cpp (convert_hf_to_gguf.py) or "
                                              "download a GGUF made by a recent build",
                      tokenizer_model=model, pre=pre))
    a = (arch or "").lower()
    if a == "gemma2":
        if meta.get(f"{a}.attn_logit_softcapping") is None or meta.get(f"{a}.final_logit_softcapping") is None:
            out.append(_f("SOFTCAP_MISSING", Layer.CONFIG, Severity.FAIL,
                          "Gemma 2 GGUF without its logit soft-capping keys: the backend cannot apply them",
                          cause=Cause.CONFIG, fix="re-convert with a current llama.cpp"))
    if a.startswith("gemma") and file_type in ("F16", "MOSTLY_F16"):
        out.append(_f("FP16_OVERFLOW_RISK", Layer.CONFIG, Severity.WARN,
                      f"{arch} was trained in bfloat16 and this file is float16: Gemma activations can exceed "
                      "float16's range (about 65k) and turn into inf / NaN, giving empty or garbage replies",
                      cause=Cause.QUANTISATION, fix="use a BF16 or Q8_0 file, or a backend that computes in "
                                                    "bf16 / fp32"))
    return out


def check_hf_dtype(cfg: dict, run_dtype: Optional[str]) -> list[Finding]:
    """A bfloat16-trained Gemma run in float16 (vLLM --dtype half, transformers torch_dtype=float16)."""
    mt = str(cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type") or "").lower()
    trained = str(cfg.get("torch_dtype") or cfg.get("dtype") or "").lower()
    if run_dtype and run_dtype.lower() in ("float16", "half", "fp16") and "bfloat16" in trained and "gemma" in mt:
        return [_f("FP16_OVERFLOW_RISK", Layer.CONFIG, Severity.WARN,
                   f"{mt} was trained in bfloat16 and runs in float16: activations can overflow to inf / NaN "
                   "(transformers #36822: Gemma 3 replies empty in float16)",
                   cause=Cause.RUNTIME, fix="run in bfloat16 (or float32)")]
    return []


# -- server launch -------------------------------------------------------------------------------

_FLAGS = {"-ctk": "cache_k", "--cache-type-k": "cache_k", "-ctv": "cache_v", "--cache-type-v": "cache_v",
          "-fa": "flash_attn", "--flash-attn": "flash_attn", "-np": "parallel", "--parallel": "parallel",
          "-c": "ctx", "--ctx-size": "ctx", "--rope-scaling": "rope_scaling", "--rope-scale": "rope_scale",
          "--rope-freq-base": "rope_freq_base", "--rope-freq-scale": "rope_freq_scale",
          "--yarn-orig-ctx": "yarn_orig_ctx", "--override-kv": "override_kv", "--dtype": "dtype"}
_BOOL = {"--no-context-shift": ("context_shift", False), "--context-shift": ("context_shift", True),
         "-nkvo": ("no_kv_offload", True), "--no-kv-offload": ("no_kv_offload", True)}


def parse_server_args(args: str) -> dict[str, Any]:
    """llama-server / vLLM command-line flags that change behaviour, plus Ollama's environment."""
    out: dict[str, Any] = {}
    toks = shlex.split(args or "", posix=os.name != "nt")
    i = 0
    while i < len(toks):
        t = toks[i]
        key, _, val = t.partition("=")
        if key in _BOOL:
            k, v = _BOOL[key]
            out[k] = v
        elif key in _FLAGS:
            k = _FLAGS[key]
            if not val:
                nxt = toks[i + 1] if i + 1 < len(toks) else ""
                if k == "flash_attn" and (not nxt or nxt.startswith("-")):
                    val = "on"                          # bare -fa (older builds)
                else:
                    val, i = nxt, i + 1
            out[k] = val
        i += 1
    return out


def ollama_env(env: Optional[dict] = None) -> dict[str, Any]:
    e = os.environ if env is None else env
    out: dict[str, Any] = {}
    if e.get("OLLAMA_KV_CACHE_TYPE"):
        out["cache_k"] = out["cache_v"] = e["OLLAMA_KV_CACHE_TYPE"]
    if e.get("OLLAMA_FLASH_ATTENTION"):
        out["flash_attn"] = "on" if e["OLLAMA_FLASH_ATTENTION"] in ("1", "true", "on") else "off"
    if e.get("OLLAMA_NUM_PARALLEL"):
        out["parallel"] = e["OLLAMA_NUM_PARALLEL"]
    if e.get("OLLAMA_CONTEXT_LENGTH"):
        out["ctx"] = e["OLLAMA_CONTEXT_LENGTH"]
    return out


def _int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def check_launch(flags: dict[str, Any], *, ctx_trained: Optional[int] = None,
                 rope_freq_base_file: Optional[float] = None, cfg: Optional[dict] = None) -> list[Finding]:
    out: list[Finding] = []
    for side in ("cache_k", "cache_v"):
        v = str(flags.get(side) or "").lower()
        if v and v not in ("f16", "bf16", "f32"):
            sev = Severity.WARN if v.startswith(("q4", "q5", "iq")) else Severity.INFO
            out.append(_f("KV_CACHE_QUANTISED", Layer.CONFIG, sev,
                          f"the KV cache ({side[-1].upper()}) is stored as {v}: it shifts logits and flips close "
                          "tokens (tool-call errors were reported with quantised caches)",
                          cause=Cause.RUNTIME, fix="rerun with f16 KV cache (drop -ctk/-ctv) and compare: "
                                                   "`wsr triage --alt <second server>`", **{side: v}))
    fa = str(flags.get("flash_attn") or "").lower()
    if fa in ("on", "1", "true", "auto"):
        out.append(_f("FLASH_ATTENTION_ON", Layer.CONFIG, Severity.INFO,
                      f"flash attention is {fa}: a different attention kernel (fp16 accumulators in the CUDA kernel "
                      "have overflowed into garbage before, llama.cpp #17746)",
                      cause=Cause.RUNTIME, fix="if replies are garbage, compare with -fa off on a second server "
                                               "(`--alt`)"))
    par, ctx = _int(flags.get("parallel")), _int(flags.get("ctx"))
    if par and par > 1 and ctx:
        out.append(_f("CONTEXT_SPLIT_ACROSS_SLOTS", Layer.CONFIG, Severity.WARN if ctx // par < 4096 else Severity.INFO,
                      f"--ctx-size {ctx} is split across {par} slots: each request gets {ctx // par} tokens",
                      cause=Cause.RUNTIME, fix=f"raise --ctx-size to {ctx_trained or ctx * par} x slots, or lower "
                                               "--parallel", per_slot=ctx // par))
    if flags.get("context_shift") is True:
        out.append(_f("CONTEXT_SHIFT_ON", Layer.CONFIG, Severity.INFO,
                      "context shift is on: past the slot's context the server drops the oldest tokens "
                      "mid-generation, and the model loses the start of the conversation",
                      cause=Cause.RUNTIME, fix="raise the context, or --no-context-shift to stop instead"))
    rb = flags.get("rope_freq_base")
    if rb and rope_freq_base_file:
        try:
            if abs(float(rb) - float(rope_freq_base_file)) / float(rope_freq_base_file) > 1e-3:
                out.append(_f("ROPE_OVERRIDE_DIFFERS", Layer.CONFIG, Severity.FAIL,
                              f"--rope-freq-base {rb} overrides the file's {rope_freq_base_file}: positions are "
                              "encoded differently from training",
                              cause=Cause.CONFIG, fix="drop --rope-freq-base unless the model card asks for it"))
        except ValueError:
            pass
    if flags.get("rope_scaling") or flags.get("rope_scale") or flags.get("rope_freq_scale"):
        out.append(_f("ROPE_SCALING_FLAG", Layer.CONFIG, Severity.INFO,
                      "RoPE scaling is forced on the command line: static YaRN / linear scaling helps long "
                      "inputs and can hurt short ones (Qwen's guidance: enable it only for long contexts)",
                      cause=Cause.CONFIG, fix="drop it for short prompts; match the model card's factor"))
    if cfg is not None:
        out += check_hf_dtype(cfg, flags.get("dtype"))
    return out


# -- live server -----------------------------------------------------------------------------------

def check_slots(props: dict, *, ctx_trained: Optional[int] = None) -> list[Finding]:
    """llama-server /props: per-slot context against the model's trained context."""
    n_slot = _int((props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx"))
    slots = _int(props.get("total_slots"))
    out: list[Finding] = []
    if n_slot and slots and slots > 1:
        out.append(_f("SERVER_SLOTS", Layer.PROBE, Severity.WARN if n_slot < 4096 else Severity.INFO,
                      f"the server has {slots} slots of {n_slot} tokens each"
                      + (f" (the model was trained on {ctx_trained})" if ctx_trained else ""),
                      cause=Cause.RUNTIME, fix="long prompts or replies past a slot are cut or context-shifted: "
                                               "raise --ctx-size or lower --parallel",
                      slots=slots, n_ctx_slot=n_slot))
    return out


def check_context_shift(prompt_tokens: Optional[int], completion_tokens: Optional[int],
                        n_ctx_slot: Optional[int]) -> list[Finding]:
    """Prompt + reply past the slot: the server dropped the start mid-generation (or will cut it)."""
    if not (prompt_tokens and n_ctx_slot):
        return []
    total = prompt_tokens + (completion_tokens or 0)
    if total > n_ctx_slot:
        return [_f("CONTEXT_SHIFTED", Layer.RESPONSE, Severity.FAIL,
                   f"prompt {prompt_tokens} + reply {completion_tokens or 0} tokens = {total}, past the slot's "
                   f"{n_ctx_slot}: the server either shifted the context (dropping the start of the conversation, "
                   "with --context-shift) or stopped at the slot's end",
                   cause=Cause.RUNTIME, fix="raise --ctx-size (it is divided by --parallel), or shorten the input",
                   total=total, n_ctx_slot=n_ctx_slot)]
    return []
