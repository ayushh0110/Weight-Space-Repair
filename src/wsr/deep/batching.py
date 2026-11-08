"""How many sequences to generate at once without running the machine out of memory.

`plan(lm, prompt_len, max_new)` looks at what is free right now (CUDA: `torch.cuda.mem_get_info`; CPU:
available RAM) and estimates what one sequence costs:
* the KV cache for prompt + reply (layers x KV heads x head dim x 2 x bytes per element);
* the prefill: the MLP and attention activations of the prompt (only the last position's logits are kept);
* the last-step logits in float32, plus sampling buffers.

It uses at most `SHARE` of the free memory, minus a fixed reserve, and caps the batch at `MAX_BATCH`.
The estimate is deliberately pessimistic. On top of it, `generate_batch` catches an out-of-memory
error, frees the cache, halves the batch and tries again, down to one sequence at a time, so a wrong
estimate costs time, not a crash.

Models that cannot be batched safely (linear-attention / recurrent layers, whose state padding may
leak into) and generations with intervention hooks stay one at a time.
`WSR_BATCH=off` turns batching off; `WSR_MAX_BATCH=N` lowers the cap.
"""
from __future__ import annotations

import os
from typing import Any

import torch

SHARE = 0.8                       # of the free memory (after the reserve), at most
RESERVE_CUDA = 400 * 2**20        # bytes always left free on the GPU (other apps use it too)
RESERVE_CPU = 2 * 2**30           # and in RAM
MAX_BATCH = 32
MAX_BATCH_CPU = 8


def enabled() -> bool:
    return os.environ.get("WSR_BATCH", "on").strip().lower() not in ("off", "0", "no", "false")


def cap() -> int:
    try:
        return max(1, int(os.environ.get("WSR_MAX_BATCH", MAX_BATCH)))
    except ValueError:
        return MAX_BATCH


def batchable(lm: Any) -> bool:
    """Plain attention stacks only: padded positions are masked there; a recurrent state is not."""
    if not enabled():
        return False
    cfg = lm.cfg
    kinds = set(getattr(cfg, "layer_types", None) or [])
    if kinds - {"full_attention", "sliding_attention", "attention"}:
        return False
    mt = str(getattr(cfg, "model_type", "")).lower()
    return not any(x in mt for x in ("mamba", "rwkv", "jamba", "next", "falcon_h", "lfm"))


def free_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # Memory PyTorch already holds but is not using (freed tensors in its caching allocator) serves new
        # tensors too; the driver does not count it as free. The reserve for other applications still applies
        # to what the driver reports, so another program never loses memory it could have had.
        cached = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
        return max(0, int(free) - RESERVE_CUDA) + int(cached)
    try:
        import psutil  # type: ignore

        avail = int(psutil.virtual_memory().available)
    except ImportError:                       # be careful without a way to measure
        avail = 4 * 2**30
    return avail - RESERVE_CPU


def per_sequence_bytes(lm: Any, prompt_len: int, max_new: int) -> int:
    cfg = lm.cfg
    p = next(lm.model.parameters())
    el = p.element_size()
    n_layers = lm.n_layers
    heads = int(getattr(cfg, "num_attention_heads", 0) or 1)
    kv = int(getattr(cfg, "num_key_value_heads", 0) or heads)
    hd = int(getattr(cfg, "head_dim", 0) or (lm.d_model // heads))
    vocab = int(getattr(cfg, "vocab_size", 0) or lm.unembed.weight.shape[0])
    inter = int(getattr(cfg, "intermediate_size", 0) or 4 * lm.d_model)
    total = prompt_len + max_new
    kv_cache = 2 * n_layers * kv * hd * total * el
    # generate() keeps only the last position's logits (logits_to_keep), so the prefill costs the MLP and
    # attention activations of the prompt, not vocabulary x prompt. Measured on Llama-3.2-1B fp16 (prompt
    # 362, 128 new): 34 MB per sequence against 35 MB from this formula before the margin.
    prefill = prompt_len * (3 * inter + 4 * lm.d_model) * el
    step = vocab * 4 * 3                       # float32 logits, processed copy, probabilities
    return int(1.25 * (kv_cache + prefill + step)) + 1


def plan(lm: Any, prompt_len: int, max_new: int, n: int) -> int:
    """The batch size for `n` sequences of up to `prompt_len` prompt tokens and `max_new` new ones."""
    if n <= 1 or not batchable(lm):
        return 1
    dev = lm.device
    budget = int(free_bytes(dev) * SHARE)
    if budget <= 0:
        return 1
    b = budget // per_sequence_bytes(lm, prompt_len, max_new)
    limit = cap() if dev.type == "cuda" else min(cap(), MAX_BATCH_CPU)
    return int(max(1, min(b, limit, n)))


def is_oom(e: BaseException) -> bool:
    if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
