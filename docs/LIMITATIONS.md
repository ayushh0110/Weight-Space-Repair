# Limitations

What wsr does not do, what is not tested, and what is open. Read this before trusting a result on a setup
that differs from the ones below. Measured results are in [RESULTS.md](RESULTS.md); how each part was built is
in [DESIGN.md](DESIGN.md).

## Scale and models

- **Deep diagnosis and `wsr program` with the agent are tested on models up to 1.5B parameters, weight repair
  up to 1B:** Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct (diagnosis, and probe-gated steering found by the agent),
  Llama-3.2-1B-Instruct, Qwen3.5-0.8B, and a tiny random Qwen3.5 for the hybrid linear-attention path. Qwen3.5-0.8B
  was too slow on this GPU for a refusal test (its hybrid layers generate one prompt at a time). Nothing larger has been verified. On a 4 GB GPU, models
  above about 1.5B do not fit for editing.
- **Apple's MPS is opt-in and not yet measured on a Mac.** It was built without one: its code paths are tested
  with stand-ins, the self-test's checks on CPU and CUDA, and macOS itself on CPU in CI (GitHub's Macs have no
  usable GPU). Each Mac checks itself before MPS is used (a self-test and a repeat check, falling back to the CPU),
  which catches unsound or non-repeatable results, not every difference a large model could show. `--device auto`
  does not pick MPS until `scripts/check_mps.py` reports from real Macs agree; until then a run on MPS is checked
  but unmeasured. Metal's share of the unified memory is macOS's default estimate unless PyTorch or
  `iogpu.wired_limit_mb` gives the exact figure. The CPU fallback for ops MPS lacks is set only when wsr is
  imported before torch (every `wsr` command; a script that imports torch first must set it itself).
- **MLX checkpoints are recognised, not loaded.** Triage reads them; the deep tools need the full-precision model
  they were converted from.
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
- The language check needs the expected language (`--expect-lang`, or an item's `lang`); it is not inferred from
  the request, and replies under 25 words are not judged. `--expect-script` checks the writing system instead.
- `--scan-values` (NaN / Inf in safetensors weights) needs numpy; without it the scan is reported as skipped.

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
- **Sets are moderate.** The powered refusal tests have 19 to 37 failing items in validation and 16 to 32 in
  confirmation; the planted-bug goals have 6 to 18 and 5 to 15, the failure-type tests 22 to 61 and 18 to 47,
  and the runs that could not show a result are kept
  as a record in RESULTS.md. The exact sign test is honest about that: with k failing items the best possible p is 0.5^k, so a
  set with 4 failing items cannot confirm anything at 0.05 (the power check says so before the run), and one
  with 5 confirms only when all five are fixed and none is lost. The
  correction (Bonferroni) counts the candidate slots one run declares (the playbook's candidate, a refusal goal's
  steering pass, and up to 3 agent attempts), not every run you make on the same validation set: looks are logged
  next to the goal, not corrected for. Several of the goals in RESULTS.md were judged more than once.
- **Splits are by prompt text, not by prompt family.** `wsr objective split` and `from-failure` deduplicate exact
  user text; close paraphrases of one prompt can land in different splits, and then count as separate evidence. The
  test sets in RESULTS.md were written as distinct prompts, but no family ids were recorded. Keeping a family in
  one split (a `family` field) is open work.
- **The confirmation registry is local.** A used confirmation set is recorded by its content hash under the
  cache base (`WSR_CACHE_DIR`) and in a note next to the file. Another machine, another cache folder, or a set
  with one item changed is not recognised as used. It guards against accidents, not against someone who wants to
  reuse a set.
- **Exact identity fails closed.** A model re-saved in another precision, re-sharded, or converted by a tool that
  rounds differently gets different tensor hashes and is refused, though it may be the same model in practice;
  `--allow-base-mismatch` is the deliberate way past it.
- **The damaged-weights recipe finds one kind of damage.** It flags rows or columns whose norm is far above
  their matrix's median. Damage that keeps norms (noise at the same scale, flipped signs, swapped rows) is not
  found. Biases, norm gains, the embedding and the output head are not scanned. Its thresholds rest on three
  healthy models (Qwen2.5-0.5B and 1.5B, Llama-3.2-1B, largest ratio 13.4x), and it was tested on one planted
  pattern (16 columns x200).
- **A repair that restores the original inherits the original's safety.** The harm check compares the candidate
  with the damaged model, and garbage counts as refused for a harmful request. On the planted garbage goal the
  exact restore was not shipped, because the original Qwen2.5-0.5B answers 2 of the 20 held-back harmful
  requests. Accepting that is `allow_safety_loss`, your decision. The agent can now commit the restore and build
  a second fix (a refusal gate, for example) on the restored model (`commit_edit`); the playbook cannot, and stops at
  the restore. With a contrast file for the harm probe the agent shipped the restore + a gate: the strict check passed
  on its 20 held-back requests, but on 511 independent harmful requests the shipped fix still answers 8% (the
  undamaged model: 17%). Twenty requests is a sample, not a guarantee.
- **A gate's estimate is only as good as the calibration prompts.** It counts what a gate would do on wsr's pools and
  the goal's own prompts; on independent harmful requests, margins it showed lowering none still lowered 7 to 10.
- **Guardrails can measure few items on a badly broken model.** Items whose original reply was broken are left
  out, so on the garbage goal the item guardrails measured 5 of 66 items. The report gives the count; the harm
  check and the regression suite are unaffected.
- **Early exit is a practice-side shortcut.** A measuring call stops after its first 30 items when the candidate
  is significantly worse there or newly broke at least 3 replies, more than it mended. A candidate that would
  have recovered on the rest is reported as stopped; `stop_early: false` measures everything. The judge never
  stops early.
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
  Under the current rules seven real-model agent runs produced confirmed fixes, all powered: the five refusal
  tests with `harm_probe` (Qwen2.5-0.5B over- and under-refusal, Llama-3.2-1B under-refusal, Qwen2.5-1.5B over-
  and under-refusal), planted blank replies, and list loops (the DRY sampler). The earlier ones that could not
  show a result are kept as a record in RESULTS.md. The feedback loop over judged attempts is tested with scripted agents
  only; no live run has needed a second attempt yet.
- The judged attempts leak a little about the validation set: a few coarse symptoms per attempt (the gain
  rounded to 5 points, buckets of counts, which guardrail failed). With 3 attempts that cannot reconstruct a set of
  dozens of items, and the correction counts every attempt, but it is not zero; the confirmation set, evaluated once
  after an accepted attempt, is what protects against fitting to validation.
- The safety mirror (`harmful_kept`) uses half of wsr's own held-back harmful requests, not the goal's guardrail
  data: a practice-side signal. The judge checks the other half, which the agent never sees. Twenty requests a
  side is enough to catch a repair that answers several, not to measure a small change.
- **`wsr program` cannot resume an agent run.** Its practice budget and held-out looks are not in the agent's
  state, so a run stopped by a limit or an outage starts again. `wsr repair --agent --resume` can continue.
- **Multi-stage surgery is stricter for settings.** Settings measured on a committed stack are checked by
  generating the regression suite under them on the patched model, so the stack's own change to a suite reply
  counts. Judged alone, a patch's suite check reads drift and agreement instead. A stack plus settings can therefore
  fail where the patch alone would pass. At most 4 edits can be committed at once, and an edit built on one stack
  is never reused on another (it is rebuilt).
- Transient provider errors are retried for about 7.5 minutes (waits of 30, 120 and 300 seconds). A longer
  outage still ends the run.
- **The harm probe reads style when its data is one-sided.** On a benchmark whose harmful and safe prompts are
  written alike (OR-Bench), a probe calibrated with one kind only learned the style, and its gate answered 61
  toxic requests in one run and refused 175 safe prompts in another (both stopped by the guardrails). wsr now
  calibrates on a goal's prompts only when practice plus `contrast` hold both kinds; otherwise it falls back to the
  built-in pools, which may read such a benchmark poorly.
- **A data guardrail is judged on its net change.** A refusal guardrail lets newly refused requests offset
  harmful requests that are newly answered: on Qwen2.5-1.5B over-refusal the harmful guardrail held at 83% → 93%
  while 10 requests the model used to refuse were answered. Only the regression suite's safety check is per item.
  A per-item limit for refusal guardrails is not built yet.
- The workspace is a restricted numeric language with Python-like syntax, not a kernel; an isolated container
  kernel is not built. It runs in wsr's own process: its limits are wsr's checks (sizes estimated before a call,
  lengths of strings and lists, integer size, steps, time), with a memory watch as the backstop (with psutil,
  3 GB of growth per call), not operating-system limits on a separate process.
- A single slow torch call in the workspace runs to its end (the deadline is checked between operations), and a
  single call whose size no estimate covers can allocate before the memory watch sees it.
- The agent's state is rewritten after every turn, up to 1 GB of workspace tensors.

## Graders

- No IFEval instruction checker and no model-as-judge grader: format goals are graded by the detectors only.
- The refusal reading is rule-based English phrasing. It separates a will-not, a non-answer and an answer, but it
  cannot tell a rebuttal of a harmful premise from compliance, and a refusal phrased in a way it has not seen is
  read as an answer. It was checked by hand on samples of Qwen2.5-0.5B and Llama-3.2-1B replies only. Common
  idioms with will-not wording ("I can't help but smile", "I can't recommend it enough") and a will-not inside a
  quoted message are not read as refusals (2,009 recorded replies read the same before and after that change);
  idioms it does not list still are.
- The `wsr-refusal` pools are wsr's own writing (templated in part): good for calibration and practice, not a
  benchmark. `wsr objective check` flags a validation or confirmation set that overlaps them.

## Downloads and caches

- The llama.cpp converter is downloaded (after asking) from the official GitHub tag that matches your llama-server
  build, over HTTPS, with no checksum of its own. Point `--llama-cpp` (or `WSR_LLAMA_CPP`) at a checkout you trust
  to skip the download.
- The GGUF judge's cache of converted original models is keyed by file names, sizes and modification times (and
  the config's content), not a hash of the weights: replacing a weight file in place with one of the same size
  and time would reuse the old conversion. `wsr cache clear gguf` removes them.

## Open work

- Runs on 3B to 8B models, and on Gemma and Mistral.
- A container kernel for the agent (Docker or WSL, no network, weights read-only).
- Prefilled matched pairs for single-token failures (blank, refusal, language).
- An IFEval checker.
- A known-good reference checkpoint, for models damaged everywhere.
- Float32 compute on float16 weights for GPUs whose half precision is slow. On a GTX 1650, fp16 decoding runs at
  21 GB/s against 182 in fp32, and Llama-3.2-1B, which does not fit in fp32, takes about 5 s per reply.
- Multi-stage surgery in the playbook (the agent has it: `commit_edit`).
- Resuming an agent run inside `wsr program`.
- A llama.cpp control-vector export for `add` steerings.
- A changelog. (Continuous integration runs the tests on every push: the core tests on Linux and Windows without
  optional dependencies, and the deep tests on CPU torch. The CUDA comparison test runs only on a machine with a GPU.)
