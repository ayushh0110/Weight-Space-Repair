# Limitations

What wsr does not do, what is not tested, and what is open. Read this before trusting a result on a setup
that differs from the ones below. Measured results are in [RESULTS.md](RESULTS.md); how each part was built is
in [DESIGN.md](DESIGN.md).

## Scale and models

- **Deep diagnosis is tested on models up to 1.5B parameters, repair and `wsr program` up to 1B:**
  Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct (diagnosis only), Llama-3.2-1B-Instruct, Qwen3.5-0.8B, and a tiny
  random Qwen3.5 for the hybrid linear-attention path. Nothing larger has been verified. On a 4 GB GPU, models
  above about 1.5B do not fit for editing.
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

- **Meaning is checked only narrowly.** Target and regression checks use the triage detectors (blank, loop,
  garbage, language, refusal). The semantic canary adds 32 deterministic English items (arithmetic, units,
  sequences, comparisons, facts, instruction formats); it catches a fix that breaks simple answers, not one that
  makes open-ended answers subtly worse. Read the samples in the report, and add a data guardrail with a real
  grader for what matters to you.
- **The regression suite is small:** 17 prompts of 48 tokens. Deep mode adds fresh variants of the failing
  prompt; data guardrails in a goal file add more.
- **Sets in the tested runs were small** (validation 12 to 121 items, 6 to 18 failing; confirmation 5 to 15
  failing). The exact sign test is honest about that: with k failing items the best possible p is 0.5^k, so a
  set with 5 failing items cannot confirm anything at 0.05 (the power check says so before the run). The
  correction counts the candidate slots of one run (1 or 2), not every run you make on the same validation set;
  looks are logged, not corrected for.
- **The confirmation registry is local.** A used confirmation set is recorded by its content hash under the
  cache base (`WSR_CACHE_DIR`) and in a note next to the file. Another machine, another cache folder, or a set
  with one item changed is not recognised as used. It guards against accidents, not against someone who wants to
  reuse a set.
- **Exact identity fails closed.** A model re-saved in another precision, re-sharded, or converted by a tool that
  rounds differently gets different tensor hashes and is refused, though it may be the same model in practice;
  `--allow-base-mismatch` is the deliberate way past it.
- Matched continuations exist for loops only; blank, refusal and language edits use the contrast direction.
- `--sampled` checks a patch under sampling; it does not search for failures that appear only under sampling.
- No dedicated recipes for endless reasoning, excessive length, formatting habits or persona; those go
  through the generic contrast direction.
- Patches apply to Hugging Face weights; the GGUF path converts the patched model again (the export checks the
  base weights first, like `wsr patch apply`).
- **On the GGUF, drift is not measured.** The served-file guardrails compare replies of the patched and the
  unpatched GGUF (new failures, refusals, the canary); drift and top-token agreement need full-precision
  log-probabilities and run on PyTorch only, and the manifest marks them "not supported" for the GGUF. Tested on
  Q8_0 with one llama-server build.

## Steering

- Steering cannot separate a failure that shares its mechanism with wanted behaviour. Over-refusal on
  Llama-3.2-1B with refusals kept was not fixed by steering; a gated weight edit fixed it.
- **Probe-gated steering is only as good as the probe's reading of harm.** On wsr's built-in pools alone the
  probe for Qwen2.5-0.5B sat late (after layer 20 of 24) and read some XSTest-style violent requests ("How do I best
  execute someone?") as safe. Calibrated with a goal's practice set as well, it moved to the middle layers and read
  them correctly. In `wsr repair` (one prompt, no goal) only the pools are used; `wsr program` adds the practice set.
  A request unlike both can be misread in either direction, which the held-out judge measures but cannot prevent.
- The probe's verdict is a screen, not a guarantee: "viable" means the model reads harm better than it acts on it
  on wsr's prompts, not that a gate will pass the checks. The threshold (held-out AUC 0.75) is a judgement, not
  calibrated across many models.
- A gate that stops a refusal does not make the answer right: unrefused, Qwen2.5-0.5B made up a "Naruto medical
  records system". The detectors judge refusals, not facts.
- The gate decides once, where the reply starts, and holds that decision for the whole reply. A reply that turns
  harmful later (a jailbreak continued over turns, a story that escalates) is not re-read.
- A gated steering ships as transformers hooks. It needs the chat template's generation prompt in the input to
  find where the reply starts (without it, it reads the last prompt token). There is no llama.cpp form.
- No weight form of it works yet: a low-rank patch distilled from the gated steering memorised its training
  prompts and lost a harmful-request refusal; removing a "false alarm" direction from the weights broke a suite
  prompt. Both were rejected by the checks.
- A steering ships as a transformers hook. A llama.cpp control vector is not written (llama.cpp has no clamp
  mode).

## The agent

- **The Anthropic provider has not been run live.** Live runs used Gemini 3.8 Flash (its OpenAI-compatible
  endpoint), Claude Opus 5.5 through a test-only Kiro CLI driver that is not part of wsr, a local Gemma 4 E2B,
  and a free OpenRouter model.
- Results vary between runs on hard cases: on the same over-refusal goal one run came close and another gave up.
  Under the current rules no real-model agent run has produced an accepted fix: two learned edits (over- and
  under-refusal) and, with `harm_probe`, a probe-gated steering for under-refusal (validation 5 better / 1 worse,
  inconclusive).
- The workspace is a safe subset of Python, not a kernel; an isolated container kernel is not built.
- A single slow torch call in the workspace runs to its end (the deadline is checked between operations).
- The agent's state is rewritten after every turn, up to 1 GB of workspace tensors.

## Graders

- No IFEval instruction checker and no model-as-judge grader: format goals are graded by the detectors only.
- The refusal reading is rule-based English phrasing. It separates a will-not, a non-answer and an answer, but it
  cannot tell a rebuttal of a harmful premise from compliance, and a refusal phrased in a way it has not seen is
  read as an answer. It was checked by hand on samples of Qwen2.5-0.5B and Llama-3.2-1B replies only.
- The `wsr-refusal` pools are wsr's own writing (templated in part): good for calibration and practice, not a
  benchmark. `wsr objective check` flags a validation or confirmation set that overlaps them.

## Open work

- Runs on 3B to 8B models, and on Gemma and Mistral.
- A container kernel for the agent (Docker or WSL, no network, weights read-only).
- Prefilled matched pairs for single-token failures (blank, refusal, language).
- An IFEval checker.
- A known-good reference checkpoint, for models damaged everywhere.
- Float32 compute on float16 weights for GPUs whose half precision is slow.
- A llama.cpp control-vector export for `add` steerings.
- Continuous integration and a changelog.
