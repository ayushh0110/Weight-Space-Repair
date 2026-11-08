"""Deep triage: weight-level diagnosis on a full-precision Hugging Face model.

Read-only. Given a failing prompt, it names the component that causes the
failure: where in depth the failing token takes over (token trace), the
behaviour direction that separates failing from passing runs, the layers,
MLPs and heads that write it (patching and ablation), how it relates to known
directions (refusal, end-of-turn, language, repetition), and what the
implicated weight matrices look like. For GGUF users it also measures how far
each quantised tensor is from the full-precision one (quant.py).

Needs ``torch`` and ``transformers`` (``pip install wsr-triage[deep]``); the
rest of ``wsr`` does not.
"""
