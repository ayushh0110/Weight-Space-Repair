# Limitations

What wsr does not do, what is not tested, and what is open. Read this before trusting a result on a setup
that differs from the ones below. Measured results and how each part was built are in [DESIGN.md](DESIGN.md).

## Scale and models

- **Weight-level stages are tested on models up to 1B parameters:** Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct,
  Qwen3.5-0.8B, and a tiny random Qwen3.5 for the hybrid linear-attention path. Nothing larger has been
  verified. On a 4 GB GPU, models above about 1.5B do not fit for editing.
- **Gemma and Mistral are untested** at the weight level. Writer shares assume a plain pre-norm residual;
  Gemma's extra norms rescale each write, so they are approximate there.
- **Surgery removes, suppresses or redirects behaviour the model already has.** It cannot add knowledge or
  a skill; a goal that needs one ends in "could not fix" or "nothing to edit", and the report says so.

## Triage

Not tested against a live server:

- a server without a vision projector (its error texts are from llama-server's documentation);
- Ollama's native API, vLLM, LM Studio and OpenAI-hosted endpoints (detection and error texts from their
  documentation, and fake-server tests);
- the `--alt` quantisation differential with a second live server;
- the context probe above 2,048 tokens.

Known limits:

- Audio and video inputs are not checked, and image token cost is not estimated.
- Repetition avoidance is not detected for Cyrillic, Arabic and CJK replies (healthy text there already
  reuses few words); only the penalty A/B covers them.
- A penalty set as a server-side default (Ollama, LM Studio) is invisible to wsr.
- A fluent answer that is off task, with no cause in the request, is out of reach for generic detectors.
- Probes cannot run in parallel on llama-server without changing the replies (measured: it is not batch
  invariant).

## Quantisation

- `wsr quant` skips IQ, TQ and MXFP4 types; Q5_K has not been seen in a real file.
- AWQ-processed GGUFs lose more than rounding alone; a per-file baseline handles it, but the exact processing
  is not identified.
- `wsr qbisect` reproduces the served model in PyTorch from dequantised weights, not llama.cpp's integer
  kernels, so a failure only the kernels cause shows as "the quantised model passes". K-quant files and
  architectures other than Qwen and Llama are not bisected yet.

## Deep diagnosis

- Contrast sets are small (3-5 failing and 3-5 working prompts), so directions are noisy; more `--failing`
  and `--passing` prompts help. Close calls (the failing token below p 0.5) often end with no specific
  direction.
- The language probe covers 12 languages; its calibration questions were not written by native speakers.
- A "right plan, wrong token" failure is located only at the readout.
- The copy-head probe has not been tested on a strong copy loop.
- Sparse autoencoder features, attribution patching and ACDC are not used.

## Repair and the judge

- **Meaning is not checked.** Target and regression checks use the triage detectors (blank, loop, garbage,
  language, refusal). A fluent reply in the right language can still be wrong; read the samples in the report.
- **The regression suite is small:** 17 prompts of 48 tokens. Deep mode adds fresh variants of the failing
  prompt; data guardrails in a goal file add more.
- **Held-out sets in the tested runs were small** (20 to 40 items). The bar is a paired gain above 2
  standard errors with at least 5 scored items; with few changed items that is more generous than the exact
  sign test, which the report prints next to it. Looks at the held-out set are logged, not corrected for.
- Matched continuations exist for loops only; blank, refusal and language edits use the contrast direction.
- `--sampled` checks a patch under sampling; it does not search for failures that appear only under sampling.
- No dedicated recipes for endless reasoning, excessive length, formatting habits or persona; those go
  through the generic contrast direction.
- Patches apply to Hugging Face weights; the GGUF path converts the patched model again. The file-level
  export does not compare a patch's base weights (`wsr patch apply` does).

## Steering

- Steering cannot separate a failure that shares its mechanism with wanted behaviour. Over-refusal on
  Llama-3.2-1B with refusals kept was not fixed by steering; a gated weight edit fixed it.
- A steering ships as a transformers hook. A llama.cpp control vector is not written (llama.cpp has no clamp
  mode).

## The agent

- **The Anthropic provider has not been run live.** Live runs used Gemini 3.8 Flash (its OpenAI-compatible
  endpoint), Claude Opus 5.5 through a test-only Kiro CLI driver that is not part of wsr, a local Gemma 4 E2B,
  and a free OpenRouter model.
- Results vary between runs on hard cases: on the same over-refusal goal one run came close and another gave up.
- The workspace is a safe subset of Python, not a kernel; an isolated container kernel is not built.
- A single slow torch call in the workspace runs to its end (the deadline is checked between operations).
- The agent's state is rewritten after every turn, up to 1 GB of workspace tensors.

## Graders

- No IFEval instruction checker and no model-as-judge grader: format goals are graded by the detectors only.

## Open work

- Runs on 3B to 8B models, and on Gemma and Mistral.
- A container kernel for the agent (Docker or WSL, no network, weights read-only).
- Prefilled matched pairs for single-token failures (blank, refusal, language).
- An IFEval checker.
- A known-good reference checkpoint, for models damaged everywhere.
- Float32 compute on float16 weights for GPUs whose half precision is slow.
- A llama.cpp control-vector export for `add` steerings.
- Continuous integration and a changelog.
