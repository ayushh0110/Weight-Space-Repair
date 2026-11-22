# Limitations

What has been tested so far, and where to be careful. Measured results are in [RESULTS.md](RESULTS.md) and how
each part works is in [DESIGN.md](DESIGN.md).

## Models and hardware

- Deep diagnosis and `wsr program` with the agent are tested on models up to 1.5B parameters and weight repair up
  to 1B (Qwen2.5-0.5B and 1.5B, Llama-3.2-1B, Qwen3.5-0.8B), all on a 4 GB GTX 1650. Triage has no size limit, it
  reads files, requests and replies. Larger models, Gemma and Mistral at the weight level are next.
- Apple's MPS is opt-in and checked by a self-test on each Mac before use, with a CPU fallback. It hasn't been
  measured on a real Mac yet, `scripts/check_mps.py` collects reports.
- MLX checkpoints are recognised by triage. The deep tools need the full-precision model they came from.

## Triage

- Tested live on llama-server. Ollama's native API, vLLM, LM Studio and hosted OpenAI endpoints are covered by
  their documented formats and fake-server tests.
- Audio and video inputs aren't checked. A sampling default set inside the server (Ollama, LM Studio) isn't
  visible to wsr, so pass the settings you actually use.
- The language check needs the expected language (`--expect-lang`) and judges replies of 25 words or more.
- `REASONING_LOOP` is tuned on Qwen3.5-0.8B traces (25 loops and 54 healthy replies, no overlap in score). It
  reads the reasoning line by line.
- The warmer rerun judges one prompt on two samples, so it tells you this prompt escapes at temperature 1.0, not
  how often. It runs for loops at temperature 0.8 or below when the request sets a temperature.

## Quantisation

- `wsr quant` skips IQ, TQ and MXFP4 types.
- `wsr qbisect` rebuilds the served model in PyTorch from dequantised weights, so a failure caused only by
  llama.cpp's own kernels shows as "the quantised model passes". It covers Qwen and Llama architectures.

## Deep diagnosis and repair

- Surgery removes, suppresses or redirects behaviour the model already has. It can't add knowledge or a skill,
  and the report says so when a goal needs one.
- Fixes are checked with the triage detectors, a 17-prompt regression suite and a 32-item semantic canary. That
  catches broken answers, not subtly worse open-ended ones, so add a data guardrail with a real grader for what
  matters to you.
- The exact sign test is strict with small sets: with k failing items the best possible p is 0.5^k, and the power
  check says before a run when a set is too small to show a result.
- A thinking-mode loop that starts after thousands of tokens is slow to reproduce in `wsr deep` (it regenerates
  from the start on PyTorch), and the repair stage judges up to 4,096 tokens.
- A patch is tied to the exact weights it was made on, so the same model re-saved in another precision is
  refused unless you pass `--allow-base-mismatch`.

## Steering and the harm probe

- Probe-gated steering is as good as the probe's reading of harm. `wsr program` calibrates it on the goal's own
  prompts as well as wsr's pools, and the held-out judge measures the result.
- Gated steering ships as transformers hooks. There's no llama.cpp form yet.

## The agent

- Live runs used Gemini 3.8 Flash, a local Gemma 4 E2B, an OpenRouter model and Claude Opus 5.5 (through a test
  driver). The built-in Anthropic provider hasn't had a live run yet.
- The agent's workspace is a restricted numeric language inside wsr's own process, with size, time and memory
  limits. A container sandbox is planned.
- `wsr program` can't resume a stopped agent run, `wsr repair --agent --resume` can.

## Next up

- runs on 3B to 8B models, and on Gemma and Mistral
- a container sandbox for the agent
- an IFEval checker
- a llama.cpp control-vector export for steering
