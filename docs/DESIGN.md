# Triage layer — technique inventory and hybrid design

Scope comes from the product report (v1 / Milestone 2): a basic triage CLI for
small open-weight models (Qwen, Gemma, Llama under ~4B), safetensors first, GGUF
read-only, focused on blank responses, loops, wrong language and over-refusal.
Deep weight-level diagnosis (section 4) and surgery (section 5, `wsr repair`) are
built on top of it; triage produces the verdict and the hand-off package.

## 1. What the existing tools actually do

Researched Nov 2026 from each project's README and, where available, source.

### LLMDoctor (Arthur031221/LLMDoctor, MIT, Python)
"brew doctor for local LLMs". Two halves:

* **Store scan** (Ollama, LM Studio, HF cache, MLX): duplicates (size-bucket then
  sha256), orphan Ollama blobs, `-partial` downloads, superseded HF revisions,
  broken files, **memory fit** (weights + f16 KV cache at target context:
  `layers × kv_heads × (key_dim + value_dim) × 2 bytes × tokens`, sliding-window
  and MLA aware) vs RAM / GPU wired limit, **template staleness** (local GGUF
  chat template vs upstream GGUF header read with an HTTP range request, after
  CRLF→LF / trailing-whitespace normalisation; Ollama manifest layer digests),
  outdated weights (sha256 vs Hub `paths-info`).
* **Endpoint probes** (backend detected via `/api/version`, `/props`,
  `/api/v0/models`, else generic OpenAI): calibration request for chars/token,
  shared time budget (90 s / 300 s deep), **context truncation vs weak recall**
  (random canary code at prompt start and end, at 85 % of 2k/4k/8k/16k; server
  `prompt_tokens` far below what was sent ⇒ truncation, the missing canary says
  which end; all tokens counted but canary forgotten ⇒ weak recall), tool-call
  arguments validated against JSON schemas (5 agent tools, parallel calls,
  streamed deltas), JSON-schema output, reasoning returned in a separate field,
  `max_tokens` ⇒ `finish_reason=length`, prefix cache, speed, RAM.

### llm-output-guard (edwinsatya/llm-output-guard, MIT, TypeScript)
Deterministic detectors for degenerate 200-OK output; every detector returns a
0–1 score and all of them run. Ported algorithms (source read):

| Signal | Algorithm | Default (chat preset) |
|---|---|---|
| EMPTY | trimmed empty, no word tokens, empty fence, punctuation/`{}` only | — |
| TOO_SHORT | `1 - len/minChars` | minLength 12 |
| REPETITION | duplicate word 3-gram fraction, first 8000 chars, abstain < 12 words | 0.4 |
| TAIL_LOOP | exact periodicity: largest share of last 200 words (or 400 chars) covered by a block (period ≤ 40 words / 80 chars) repeating ≥ 3× to the end; char mode when tail is ≥ 50 % Han/Kana/Thai | 0.5 word / 0.7 char |
| LOW_ENTROPY | greedy LZ77 tokens/chars (window 1024, minMatch 4); score `1 - ratio/0.32` | 0.75 |
| TRUNCATED | length finish reason ⇒ 1; else odd fences 0.9, unbalanced brackets 0.8, no terminal punctuation 0.55; valid JSON ⇒ 0 | 0.75 (longForm) |
| INVALID_JSON | parse → required keys → schema | opt-in |
| SCRIPT_MISMATCH | share of letters outside expected script, code/URLs excluded | opt-in |
| LANG_MISMATCH | function-word profile (8 Latin-script languages), `(best − target)/best`, abstain < 25 words | 0.6 |
| PROMPT_ECHO | share of output 5-word runs (12-char runs for CJK) found verbatim in prompt, abstain < 40 tokens | 0.6 |
| AGENT_LOOP | exact cycle over agent-turn fingerprints | — |

Plus presets (chat, strictJson, longForm, lenient) and threshold calibration at a
target FPR from the user's own logs.

### llm-stopcheck (tonyweckstrom1987/llm-stopcheck, MIT, Python)
Classifies an OpenAI-shaped response from `finish_reason`, `native_finish_reason`
(LiteLLM/OpenRouter normalise max-tokens to `stop`), content, `tool_calls`,
`reasoning_content`/`reasoning` and `usage.completion_tokens_details.reasoning_tokens`:
`truncated`, `truncated_empty`, `reasoning_exhausted`, `tool_call_truncated`,
`empty_output`, `content_filtered`, `refusal`, `missing_finish_reason`,
`unknown_finish_reason`, `malformed_response`, warnings `near_limit` (≥ 95 %
of `max_tokens`) and `at_limit`; hints `raise_max_tokens` / `retry` / `fail`.
Gaps: no native Anthropic / Gemini / Ollama shapes, only `choices[0]`.

### ModelVet (tetsuo-ai/modelvet, MIT, C11 + Python binding)
Verify-before-load structural validation. GGUF v2/v3 with a dtype table pinned
to llama.cpp; safetensors frame checks, offset tiling, dtype bit widths,
duplicate keys, unknown fields; checked arithmetic on every file-derived
integer; ACCEPT / REJECT (+ first violation and byte offset) / no-verdict exit
codes 0/1/2. Structural only — says nothing about behaviour or provenance.

### Raschka's repetition-loop checklist (manual)
Greedy decoding with the correct checkpoint, tokenizer, chat template and eval
mode; log generated token IDs; confirm one token per step, last-position
logits, KV-cache position advancing; check EOS / end-of-turn, stop strings, max
new tokens. Interpretation: greedy loops but sampling escapes ⇒ decoding is
part of it; every method loops on many prompts ⇒ checkpoint / data / template.
Non-determinism under greedy ⇒ nondeterministic kernels or hidden state.

### Troubleshooting knowledge (guides, issue trackers)
* `Say hello.` at temperature 0 as the first probe. Garbage from token one ⇒
  file or backend; control tokens in output ⇒ template; coherent loops ⇒
  sampling; loops only in long chats ⇒ context.
* `GGGG` / symbol soup: broken quant, Vulkan backend (llama.cpp #13310), driver;
  CPU-vs-GPU differential.
* Literal `<|im_end|>`, `<|eot_id|>`, `<end_of_turn>` in replies, unasked
  follow-up turns ⇒ wrong or missing template / stop token.
* EOS disagreement between `tokenizer_config.json`, `config.json`,
  `generation_config.json` and the token the chat template ends a turn with
  (Llama 3 `<|eot_id|>` vs `<|end_of_text|>`, ChatML `<|im_end|>`).
* Repeat penalty > 1.2 makes loops worse; llama-server has penalties off by
  default; Ollama's newer runner ignores repeat/frequency/presence penalties
  (Ollama #15783); loops only under `format=json` / grammar (Ollama #15502).
* Ollama silently truncates over-long prompts; only `prompt_eval_count` shows it.
* Sub-Q4 quants of small models are more loop-prone.
* Gemma 4: early GGUFs with broken templates; vLLM tool parser leaking raw
  tokens (#44522); model-level repetition collapse on long enumerations
  (arXiv 2606.13705).

## 2. Coverage the hybrid must reach

Every failure cause any of the above detects, plus the report's additions
(differential experiments, known-issue DB, weight-level verdict).

| Failure cause | Source technique | Our module |
|---|---|---|
| Corrupt / unsafe safetensors | ModelVet | `file_integrity` (own validator; uses `modelvet` when installed) |
| Corrupt / truncated GGUF | ModelVet, guides | `file_integrity` + `gguf` reader |
| Partial download / size mismatch | LLMDoctor, guides | `file_integrity` |
| Checksum differs from upstream | LLMDoctor (`paths-info`) | `file_integrity --upstream` |
| Model does not fit memory | LLMDoctor | `config_checks.memory_fit` |
| Stale / wrong chat template | LLMDoctor, guides | `config_checks` (vs base files, vs upstream) |
| Template missing / fails to render | LLMDoctor, guides | `config_checks` |
| EOS / end-of-turn / stop mismatch | Raschka, HF threads | `config_checks.eos_consistency` |
| Double BOS | troubleshooting | `config_checks` |
| Context window too small / silent input truncation | LLMDoctor | `probes.context`, `response` (prompt token count) |
| Weak long-context recall | LLMDoctor | `probes.context` |
| Bad sampling params | guides | `request_checks.sampling` |
| Output cut by max tokens | stopcheck, output-guard | `response` |
| Reasoning budget exhausted | stopcheck | `response` |
| Tool call truncated | stopcheck | `response` |
| Tool-call schema failures | LLMDoctor | `response` / `probes.tools` |
| Content filter / provider refusal | stopcheck | `response` |
| Missing / unknown finish reason, malformed body | stopcheck | `response` |
| Empty / too short | output-guard, stopcheck | `output` |
| Repetition / tail loop (word + CJK) | output-guard | `output` |
| Low-entropy collapse / symbol soup | output-guard, guides | `output` |
| Structural truncation | output-guard | `output` |
| Invalid JSON | output-guard, LLMDoctor | `output` |
| Wrong script / wrong language | output-guard | `output` |
| Prompt echo | output-guard | `output` |
| Agent loop | output-guard | `output.agent_loop` |
| Control-token leak / invented turns | guides | `output` (new) |
| Reasoning leaked into content | LLMDoctor `think` | `output` (new) |
| Refusal text (over-refusal) | report v1 scope | `output` (new) |
| Non-determinism under greedy | Raschka | `probes.determinism` |
| Decoding-caused loops (greedy vs sampled) | Raschka | `probes.differential` |
| Backend / quant / grammar-caused failure | report (novel) | `probes.differential` (next) |
| Known upstream bug | report | `known_issues` |
| Weight-level | report | verdict only; deep diagnosis is Milestone 3 |

Added after live testing against llama-server (Gemma 4 E2B Q4_0, thinking on):

| Failure cause | Source technique | Our module |
|---|---|---|
| Thinking eats the answer budget (blank reply) | live finding | `probes.reasoning`, differential `thinking_off` |
| Thinking switch ignored / reasoning unparsed into content | live finding | `probes.reasoning` |
| Server prompt has double / no BOS (ground truth) | live finding | `probes.bos` (`/apply-template` + `/tokenize`) |
| Client prefill with foreign reasoning markup, echoed into content | live finding (ScreenMind) | `request_checks.check_prefill`, differential `prefill_off` |
| JSON valid only after stripping a think block / fence | live finding | `output.json_check` (`JSON_WRAPPED`) |
| Token salad (high-entropy multi-script garbage), broken UTF-8 | live finding | `output` (`TOKEN_SALAD`, `BROKEN_UTF8`) |
| Reply cut by a client stop string | live finding | `pipeline` (`ENDED_ON_STOP_STRING`), differential `stop_off` |
| Wrong language, non-Latin targets (zh/ja/ko/ar/fa/ur/hi/mr/ne/ru/uk/bg/th/he/el...) | output-guard extended | `output.detect_language` |
| Stream cut mid-answer, error events, malformed chunks, tool fragments that do not assemble, stream ≠ non-stream | live finding | `stream`, `probes.stream` |
| KV memory on sliding-window / shared-KV models | LLMDoctor extended | `config_checks.kv_cache_bytes` (per-layer) |
| Reply depends on the server's prompt cache (greedy reply differs fresh vs cached) | live finding | `probes.cache`, differential `cache_off` |
| Server's output parser rejects the generation (HTTP 500 "does not match the expected format") | live finding | `response` (`OUTPUT_PARSE_REJECTED`, garbage symptom) |
| Over-strong repetition penalty: fluent prose that never reuses a word | live finding | `output` (`REPETITION_AVOIDANCE`, local recurrence) |
| Over-strong penalty: fluent but off-task (list stops after 4 lines, answer cut to one word) | live finding | `probes.penalty_ab` (`PENALTY_SHAPE_LOSS`) |
| Image sent in an `images` field an OpenAI endpoint drops; model asks for the image | live finding | `media.check_media`, `IMAGE_NOT_SEEN`, differential `images_as_parts` |
| image_url as a bare string, bad / cut base64, remote URL the server cannot fetch, no projector | live finding | `media.check_media`, `response` (`IMAGE_REJECTED`) |
| Server accepts images but the model does not see them, or misreads them (mismatched mmproj) | live finding | `probes.vision` |

Two verdict rules came out of the live runs. A differential flip counts only
if it reproduces (original fails again, variant passes again); otherwise it is
reported as `DIFF_UNSTABLE`, because on a nondeterministic server one lucky
rerun is not a cause. A cause the response itself confirms (the reply
starts with the request's prefill) ranks with a flip, above static hypotheses.
And a variant that fixes one symptom but not another is credited with what it
fixed (penalties off cures the word avoidance; the reply still hits the cap).

### Why local recurrence, not TTR or Zipf

184 replies from the live server (24 prompts in English, German, French,
Spanish, Hindi, Russian, Arabic, Chinese, Japanese; greedy, temperature 0.8,
repeat_penalty 1.05-2.0, frequency / presence penalty) decided the detector:

* **TTR** depends on length (Heaps' law): a healthy and a penalised 100-word
  paragraph both sat at 0.63.
* **Zipf slope** flattens under a penalty (-0.6 to -0.4) but varies as much
  between topics, so no threshold separates the two.
* **Local recurrence** (share of words already seen in the previous 50) is
  what the penalty acts on: llama.cpp penalises tokens seen in the last 64.
  Healthy Latin / Devanagari prose: 0.14-0.38. repeat_penalty 1.5-2.0: 0.02-0.18.
  Lists, code and headings are skipped; Russian, Arabic and CJK sit at
  0.01-0.17 even when healthy, so they are not judged.

Result: no false positive on the 82 replies without a strong penalty; caught
11/24 at repeat_penalty 1.3, 14/24 at 1.5, 19/24 at 2.0. The misses are code
(caught by other detectors), Cyrillic, Arabic and CJK. The penalty A/B covers
what one reply cannot show: when nothing else fails and the request sets a
strong penalty, the same request runs without it and the reply must keep its
line count, length and word reuse.

Not covered yet: per-tensor value statistics on GGUF (kurtosis, outliers, block
scale entropy, as in SecAI-Hub/gguf-guard) and quantisation error against a
full-precision reference. Today a quantisation verdict needs `--alt` (a second
endpoint serving another quant) or the LOW_BIT_SMALL_MODEL heuristic.
Semantic task failures in fluent text (off-task answers) are out of reach for
generic detectors, except where a request setting causes them and an A/B
rerun shows the difference (penalties).

Out of scope on purpose: LLMDoctor's dedupe/unbundle/cleanup (housekeeping, not
failure causes), speed and prefix-cache benchmarks (performance, not failure).

## 3. Hybrid design

1. **Symptoms and causes are separate.** Output detectors (layer 4) produce
   *symptoms*. File, config, request and probe checks produce *causes*. The
   verdict engine maps each symptom to candidate causes and reports the cause
   with the strongest evidence, in layer order: file → config → request →
   runtime → known bug → weight-level. A loop with a repeat penalty of 1.5 is a
   sampling verdict, not a model verdict.
2. **Scores, not booleans** (from output-guard): every detector returns 0–1 and
   all of them run, so the report lists every signal.
3. **Provider-native fields are ground truth** (from stopcheck): finish reason
   beats heuristics; the native finish reason is checked too; Anthropic, Gemini
   and Ollama-native shapes are normalised (stopcheck's gap).
4. **Server counts beat client guesses** (from LLMDoctor): truncation is decided
   by `prompt_tokens` / `prompt_eval_count`, with canaries to say which end.
5. **One factor at a time** (Raschka + report): greedy vs sampled, deterministic
   rerun, then backend / quant / grammar / template / context differentials.
6. **Every finding carries a fix.** Non-weight causes get an exact setting or
   command; a weight-level verdict gets the hand-off package (failing prompt,
   response, all signals).
7. **No required dependencies.** Core is the standard library. `jinja2` renders
   templates, `modelvet` adds a second structural opinion, both optional.

## 4. Deep triage (layer 7): weight-level diagnosis

Runs on the full-precision Hugging Face model, read-only, CPU is enough for
models under ~4B. Code: `src/wsr/deep/`. Command: `wsr deep <hf dir> --prompt ...`.
Gap 1 (GGUF tensor statistics and quantisation error) lives here too, because
it needs the full-precision weights the deep layer already requires:
`wsr quant <gguf> --reference <hf dir>`.

### Tools and techniques surveyed (Nov 2026)

| Tool / technique | What it gives | Used here as |
|---|---|---|
| TransformerLens 3 (TransformerBridge, ~48 architectures) / nnsight | hooks, caching, patching | not a dependency: plain PyTorch forward hooks (`deep/lm.py`) so any architecture transformers loads works, Gemma 4 included |
| Logit lens (nostalgebraist) | next-token readout per layer | step 3 token trace, with the model's own final norm and logit soft-cap |
| Tuned lens (Belrose et al.) | trained per-layer translators | not used: needs lenses trained per model |
| Activation patching / causal tracing (Meng et al.; Heimersheim & Nanda) | which component carries the failure | step 5: residual per depth, attention and MLP per layer, heads in the top layers |
| Attribution patching, ACDC, edge attribution patching | cheap / automated circuit search | not yet (to do): exact patching is affordable at < 4B |
| Difference-of-means directions (Arditi et al. refusal direction; Marks & Tegmark mass-mean probing) | behaviour direction | step 4, with Cohen's d and leave-one-out accuracy per depth |
| Directional ablation / activation addition (Arditi; Turner et al. ActAdd; CAA) | is the direction causal | step 5 causal tests: removing it must fix failing prompts, adding it must break passing ones |
| Abliteration tools (heretic, FailSpy, DECCP, ErisForge) | refusal-direction removal | refusal entry of the known-direction library (step 6); removal itself is surgery, not triage |
| Known directions: refusal, end-of-turn, language, repetition | named behaviours | step 6 library, computed on the model itself (no downloads): harmful vs harmless pairs, gradient of the end-of-turn logit, English vs 8 languages, second vs first copy of a repeated sentence |
| Sparse autoencoders / transcoders (SAELens, Gemma Scope 2) and circuit-tracer attribution graphs | feature-level explanations | not yet (to do): only for models with published SAEs / transcoders (Gemma 2/3, Llama 3.x) |
| MIT Weight-Space Programming `weightspace` harness (Apache-2.0) | agent tools: list / describe / read tensor slices, SVD, forward budget | step 7 dossier is the deterministic half; the agent loop is Milestone 4 |
| WeightWatcher (spectral analysis) | data-free layer quality (power-law alpha) | `alpha` in the weight dossier (Hill estimate of the eigenvalue tail) |
| Super weights (Yu et al. 2024) | single scalars whose loss breaks a model | `quant` reports the largest down_proj entries and their quantised values |
| SecAI-Hub/gguf-guard | GGUF per-tensor stats, block checks, robust baselines | `quant`: kurtosis, outliers, sparsity, NaN/Inf, block-scale entropy, zero / repeated blocks, bin saturation, per-role median/MAD |
| Quantisation error vs full precision (absent from gguf-guard) | quant damage per tensor | `quant --reference`: relative Frobenius error and cosine per tensor, after fitting out per-row / per-column scales (AWQ / SmoothQuant fold them in at conversion) and weighting columns by the original norm gain; baseline is the type's normal error, or the same matrix in the file's other layers when the file was processed that way |

### The eight steps, and what each emits

| Step | How | Finding |
|---|---|---|
| 1 Trigger | rerun the failing prompt greedily on the FP model with its own template | `DEEP_REPRODUCED` or `DEEP_NOT_REPRODUCED` (then quant / runtime, not weights) |
| 2 Contrast set | 10-14 meaning-preserving variants + user `--failing` / `--passing`, classified by the triage detectors | `DEEP_CONTRAST` |
| 3 Token trace | logit lens on the logit difference "failure token vs best alternative" at the decision point | `DEEP_TRACE` (onset layer, biggest jump) |
| 4 Direction | mean(failing) - mean(passing) residual per depth; the depth is chosen by what removing it does at the decision token (moves the failure, keeps passing prompts' next-token distribution: KL), from the first 80 % of layers, earliest depth within 0.05 of the best score; must beat a permutation null (removing mean(A) - mean(B) for 6 stratified random halves of the same prompts) | `DEEP_DIRECTION` |
| 5 Localisation | two questions kept apart: who *writes the direction* (each earlier component's output along it, failing minus passing; shares sum to 1 with the embeddings) and who *builds the output logit* (patch passing activations into the failing run); neurons by activation change x write along the direction, before its depth only; ablation and addition on generation | `DEEP_LOCALISED`, `DEEP_CAUSAL` |
| 6 Known directions | cosine with the library, and removing each known direction to see whether it removes this failure (a cosine between two directions estimated from a few prompts is noisy); a match is claimed only if it beats the same permutation null | `DEEP_KNOWN_DIRECTION` |
| 7 Weights | implicated down_proj / o_proj: stats, robust z vs other layers, alpha, change vs `--base`; with `--gguf`, their quantisation error | `DEEP_WEIGHTS`, `DEEP_QUANT_DAMAGE` / `DEEP_QUANT_OK` |
| 8 Report | "this failure comes from ..., written by ..." + confidence (causal tests, localisation, separation) | `DEEP_VERDICT`, hand-off directory |

The decision point is where the failure is chosen: the first answer token for
blank, refusal and language failures; the first token of the second copy for
loops. One metric is used throughout: the logit of the failure token (for a
blank reply, all end-of-turn tokens) minus the best alternative the model
considered there.

### Ground truth

`tests/deep_fixture.py` plants a weight-level bug in a tiny random Llama: one
MLP neuron in layer 2 fires on the word "please" and writes the end-of-turn
direction, so those prompts get blank replies. Deep triage finds it unaided:
onset at layer 2, the direction matches end-of-turn (cos 0.96), layer 2's MLP
restores 100 % when patched, neuron #5 carries it, and removing the direction
fixes 3/3 failing prompts while keeping 3/3 passing ones.

### Real model: Qwen2.5-0.5B-Instruct (CPU, float32, ~7 min per run)

A scan of 28 prompts found three over-refusals on the full-precision model
("bomb-shaped birthday cake", "hack my own router to install OpenWrt",
"strangle a fig tree vine"). What the real runs changed:

* **Direction depth by intervention, not separation.** Separation peaked after
  layer 20 of 24, where runs differ only because they are about to emit "I"
  vs "Creating". Removing that direction made 3/3 passing prompts refuse.
  Chosen by ablation effect instead (layer 5 on the cake case): it fixes 2/3
  and keeps 3/3 passing on generation.
* **Permutation null.** These refusals are close calls (p 0.33-0.44 for the
  refusal token), so removing almost any large direction tips them: the
  repetition and non-English directions "fixed" them too. With the null
  (shuffled-label directions, beat the best by 0.2), only one claim survives:
  on the router case, removing the model's generic refusal direction (layer 15)
  flips 5/6 failing prompts, against at most 40 % for shuffled directions. The
  cake case has no specific direction; the summary says it is a close call.
* **Writers vs logit builders.** Patching said "layer 18 MLP"; the direction
  at layer 5 cannot be written by layer 18. Both are now reported separately
  (cake: direction written by layers 3-5 MLPs; logit built by layer 18 MLP and
  layer 10 attention).
* **Quantisation error on Qwen's official GGUFs.** They are AWQ-processed:
  per-channel scales divided into the norms and multiplied into q/k/v/gate/up
  (and the MLP hidden units), so every tensor differs from the original by
  20-120 % while the function does not. A per-row / per-column scale fit brings
  them to 9-11 % (Q4_0); a real file with 5 % of block scales x4 or 2 % of
  blocks scrambled is still caught (53 % / 30 %). Q4_K and Q6_K decoders agree
  with the original weights on the Q4_K_M file (8.2 % / 3.9 % median).

### Real model, round 2: GPU, more failure kinds, a planted bug in real weights

* **GPU.** `wsr deep --device auto` (CUDA when available, fp16). Token ids and
  every captured activation stay on the CPU in float32; only the model runs on
  the device. On a GTX 1650 next to a running llama-server, Qwen 0.5B fp16 gives
  the same conclusions as CPU fp32 (numbers within 1-2 %). Patching is batched
  (`LM.patch_sweep`: one row per patched layer), checked equal to single patches.
* **A planted bug in real Qwen weights, found blind.** Layer 16 MLP neuron #0
  rewritten (`scripts/plant_real_bug.py`): its gate reads a "this prompt is about
  cats" direction, its output a vector optimised through the remaining layers to
  make `<|im_end|>` win; cat questions end after one token. Given only the
  failing prompt plus a few working prompts, deep triage reports: direction
  after layer 16, written 91 % by layer 16 MLP, neuron #0 carries 98 % of it
  (activation 11.1 failing vs ~0 passing), removing it fixes 3/3 and keeps 3/3,
  adding it breaks 3/3, confidence high.
* **Bugs this found and fixed:** a blank failure is decided where the
  end-of-turn token comes, not always at the first token ("C" then end); a list
  loop ("16. Penguin / 17. Penguin") is decided at the second copy of the stuck
  item, and passing runs are read at the same list position; token text is
  decoded ("В", not "ÐĴ"); restoration over 100 % is shown as ">100%"; fewer
  than 3 runs on either side of the contrast is a warning and lowers confidence;
  a known-direction match needs cos >= 0.5 or a specific removal result; only
  components at or after the direction's layer are named as turning it into the
  output.
* **Triage-layer fixes from the same scan** (`tests/test_real_replies.py`, real
  replies): Urdu was read as Persian (Persian's marker letters are shared with
  Urdu; Urdu-only letters now take priority); numbered lists stuck on one item
  were not loops (list-level periodicity with markers stripped); repetition the
  user asked for ("write banana 100 times") is no longer a failure.
* **Wrong language (Ukrainian answered in Russian)** and **loops (stuck animal
  list)** run end to end; both are close calls (p 0.10 for the failure token)
  and the null rejects their directions, which the report says.
* **Qwen2.5-1.5B (CPU, 27 min).** The router over-refusal persists at 1.5B. With
  three near-identical working prompts added, deep triage reports its cleanest
  real result: the direction after layer 17 matches the model's generic refusal
  direction (cos +0.88); removing that direction flips 8/8 failing prompts
  (shuffled-label directions: 0/8); on generation, removing it fixes 3/3 and
  keeps 3/3, adding it breaks 3/3; written by layers 16-17 MLPs; confidence
  high. Found on the way: the metric compared the refusal token with the
  model's runner-up, which was another refusal ('I' vs 'Sorry'). The
  alternative is now the token the passing runs chose ('To').

### Calibrated probes for wrong language and loops (parallel to the contrast set)

The contrast set (5-15 near-identical prompts) is too small and too alike for
some questions. Two probes are calibrated once per model, cached under
`~/.cache/wsr/`, and run only for their failure kind (`--no-probes` skips them).

**Language probe** (`deep/langprobe.py`). 147 prompts in 12 languages (en fr de
es pt it ru uk zh ja ko hi), including cross-language requests ("Answer in
Russian: ...", a French question asking for English). Labels are the language
the model actually answered in, not the one asked for. A diagonal-covariance
nearest-class-mean classifier per depth, temperature fitted on held-out
prompts.

* *Fit inside the reply, score at the decision point.* Fitted at the
  decision point alone, it learned the request: on Qwen 0.5B, "Answer in French
  only: ..." answered in English read as French with p = 1.00, because asked and
  answered languages nearly always agree in calibration. It is now fitted on the
  residual after 1-3 tokens of the model's own reply (the language being written
  is unambiguous) and scored at decision points of held-out prompts: 88-96 %
  from layer 17 on (Qwen 0.5B), and 63-68 % on the 19 calibration prompts where
  the model ignored the requested language (5 % at middle layers). Accuracy by
  depth is U-shaped: high early (input language), low in the middle, high late,
  as in Wendler et al. 2024.
* *Two mechanisms.* "Wrong plan": the residual itself turns to the wrong
  language (Ukrainian -> Russian from layer 7, p 0.78 late). "Right plan, wrong
  token": the residual mostly encodes the requested language but the first token
  is another ("Answer in French" -> English: French p 0.99, English token).
* *Causal, with a null.* Removing the calibrated answered-minus-expected
  direction, and steering toward the expected language, on generation; null =
  the same norm along an unrelated pair. Qwen 0.5B: Ukrainian restored 3/3 by
  removal (null en-fr 0/3); French restored 3/3 by removal and by steering
  (null de-es 0/3). The contrast-set direction for the same failures was not
  specific, so the report now leads with a confirmed probe.

**Copy-head probe** (`deep/copyheads.py`). Calibrated: each head's induction
score on a repeated random sequence (Qwen 0.5B: 16.3, 9.13, 11.12 at 0.97-0.99,
median head 0.01; 2 s). At the loop's decision point: each head's attention on
earlier copies of the token it is about to repeat, times its direct effect on
the failure's logit lead (the lead's fall when the head's output is taken out of
the final residual). Causal: remove the top k heads at every position; the share
of the logit lead removed must beat random heads by 0.2, then generation must
stop looping more often than with random heads, and lists that did not loop must
stay fine. A binary flip test was too strict: removing the 8 top heads shrank
the lead without flipping it, and the network partly compensates.

### Loop battery (`deep/loopprobes.py`): five root causes, run side by side

Every loop failure runs all probes; none gates another; each has its own control.

| Root cause | Test | Control | Real-model validation |
|---|---|---|---|
| Attractor | force the alternative token at the repeat: does generation fall back in, with states one cycle apart at cosine >= 0.9 | the same push on other loops | planted Penguin model 3/3 back in; natural list 0/3, copy loop 0/1 |
| MLP self-reinforcement | attribution patching (activation x gradient, every position, whole network); the deciding neurons must fire on and promote the repeated token | random neurons of the same layers | toy plant found exactly; real plant seeded the loop but attention carried it (not blamed, correctly: removing it raises the lead) |
| Copy heads | `copyheads.py` | random heads | "AB AB": 37 % vs 14 %, loop stops |
| Positional (RoPE) | shift every position id (odd offsets, so 2^k aliasing is not hidden); KL >= 0.1 = a defect, lead moved >= 50 % = this failure's cause | healthy model is exactly shift-invariant (KL 0.000 fp32 and fp16) | planted aliasing (multiples of 8): KL 0.56-1.43 |
| Greedy decoding | rerun sampled (T 0.7, 3 seeds) and with a 1.1 penalty | - | natural list: sampling escapes 3/3 -> verdict `sampling` |
| Quantisation | `wsr quant` / `--gguf` | type or per-file baseline | real Q4_0 damage flagged |

Fixes found on the way: a cut-off list ending in a bare "25." hid the loop from
the triage detector; with no passing variant the deep layer skipped the battery;
direct logit attribution missed effects that reach the repeat through attention;
fp16 overflows the real plant (it prints "!!!!"), and bf16 rounding breaks its
trigger, so plants are saved in float32.

### Quantisation bisection (`deep/qbisect.py`, `wsr qbisect`, and `wsr deep --gguf` when a failure does not reproduce)

Which quantised tensor causes a failure. The served model is rebuilt in PyTorch
from the GGUF's own dequantised tensors; each tensor can be switched between
that and full precision.

* **Consistent swaps.** AWQ-style files (Qwen's official GGUFs) fold scales
  into norms and the next matrices. The full-precision side is the original
  mapped into the file's scale space by exact paired transforms: column scale
  c = n_original / n_gguf (norms are float32 in the file) on every matrix that
  reads the norm; MLP hidden-unit scales applied to up and down as a pair; v/o
  only when the float32 v bias shows it. A self-check compares the rebuilt
  full-precision logits with the original model before anything else
  (Qwen 0.5B: KL 1e-7 fp32, 2e-6 fp16). Gemma-style 1 + w norms are decided by
  architecture: a folded Qwen norm sat closer to 1 + w by chance and made the
  rebuilt layer 0 24x off.
* **Search.** First divergence token between the quantised and full-precision
  replies; restore each quantised tensor alone (one forward each); greedy
  minimal set until the decision flips, then pruned.
* **Proof, on generation.** Restoring the set fixes the reply; quantising only
  the set in the full-precision model breaks it; restoring the same number of
  random tensors of the same kinds does not fix it. Outcomes: `localised` (all
  three), `necessary` (restoring fixes, alone it does not cause it), `unconfirmed`
  (random tensors fix it too; flagged as a near-tie when the lead < 0.1 logit),
  `diffuse`, `quantised_passes` (then the server's kernels / runtime, not the
  weights), `both_fail`, `rebuild_failed`.

| Case | Result |
|---|---|
| Planted: 20 % of `blk.12.ffn_down` block scales x50 in the real Q4_0 | localised to `blk.12.ffn_down` (restores 103 % of the gap, next tensor 2.6 %), both directions on generation, random 0/3 |
| Planted, weaker (10 % x8) | `quantised_passes`: no failure to attribute |
| Natural: official Q4_0, "Give me 40 different names for a cat" refused | localised to `blk.16.attn_v` (also the file's worst-error tensor in `wsr quant`); quantising it alone makes full precision refuse; random 0/3 |
| Natural: "Write a long list of fruits" refused | `necessary`: restoring `token_embd` fixes it, random 0/3, but it alone does not cause it |
| Natural: "Count from 1 to 200" refused | `unconfirmed`: near-tie (lead 0.05), random tensors fix it 2/3 |

### Run time and what was optimised (GTX 1650 4 GB, Qwen2.5-0.5B, Nov 2026)

Measured per stage (`DEEP_TIMINGS`; also in `report.context["timings"]`). Every
change below had to leave the conclusions unchanged; a speedup that moves numbers
past rounding was not kept.

| Case | Before | After | Verdict |
|---|---|---|---|
| Over-refusal, "hack my own router" | 226 s | 82 s | same |
| Wrong language, Ukrainian to Russian | 226 s | 113 s (384 s the first time per model and precision: probe calibration) | same |
| Greedy list loop | 540 s | 333 s | same |
| `wsr triage` on a failing request, probes + differential (Gemma 4 E2B, llama-server CPU) | | 109 s | |
| `wsr probe`, default set | | 29 s | |

What made the difference, in order:

1. **float32 where it fits (`auto_dtype`).** The GTX 1650 (like all GTX 16xx)
   has no fast half precision: fp16 matmul 0.34 TFLOPS vs fp32 1.53, so one
   84-token forward took 295 ms in fp16 and 82 ms in fp32. float32 is also the
   reference precision. `dtype="auto"` now picks float32 on any device where
   the weights fit (x1.25 + 768 MB headroom in free VRAM), else bfloat16 on
   Ampere and newer, else float16. Against the fp16 runs, every verdict is the
   same word for word; probabilities move in the second decimal (0.44 vs 0.45),
   which is fp16 rounding going away.
2. **Batched rows only in float32.** Ablation and patch sweeps put one
   direction or layer per batch row (same prompt, no padding). In fp16 on the
   GPU a batch is not bit-identical to single rows (logits move up to 0.05), and
   it was not faster, since one row already filled the GPU. In fp32, batch vs
   single differ by at most 7.6e-5 in a logit (4e-6 in probability), with every
   top token the same, and batching is 3x faster for ablation and 5.9x for
   last-position patching (prefix computed once, see `_prefix_cache`).
3. **Logits for the last position only** (`logits_to_keep=1`): about 20 % per
   forward; bit-identical on the GPU, last-bit rounding (~1e-7) on CPU.
4. **Memo of clean forward passes** within one diagnosis: exact by construction
   (key = token ids, position, captures, weight version).

Two fp32 runs of the same case give identical reports (0 differences).

**Kept serial on purpose.**
- *Generation* (contrast set, causal tests, loop battery: now most of the
  time). Batching rows changes fp arithmetic and sampled runs' random streams,
  and two generations at once would share the model's hooks.
- *Triage requests to the server.* llama-server batches the active slots into
  one forward and is not batch-invariant: measured on the 4-slot Gemma server,
  3 of 6 greedy replies changed when sent together instead of one at a time
  (one flipped thinking on); the same requests one at a time repeat exactly.
  Parallel probes would change what the probes see. That fact is now a probe
  of its own (`concurrency`, below).

### Load-dependent replies (`concurrency` probe)

The same greedy request runs alone twice (plain nondeterminism is
`NONDETERMINISTIC_GREEDY`), then once while up to three filler requests decode
in the other slots. A different reply is `LOAD_DEPENDENT_OUTPUT` (WARN); if the
failure classification itself changes it is a FAIL ("the failure depends on
load"), because then a production failure comes and goes with traffic and a lone
rerun does not reproduce it. It runs in the triage differential when a failure
is present and the server has more than one slot, and as `wsr probe --only
concurrency`. Live on the Gemma 4 E2B server: fired on a failing list request
(reply split at character 35), passed on "Name three primary colours".

## 5. Surgery (`wsr repair`, `src/wsr/surgery/`): playbook first

The product report's surgery layer: a playbook of behavioural failures, each
with a direction or component recipe, the layers to edit and verification
checks; an honest "could not fix" when no edit passes. Surgery removes,
suppresses or redirects behaviour the model already has; it cannot add a skill.

**Edits are the weight-space twin of a tested intervention** (`edits.py`), stored
as low-rank deltas `W += U V^T` (a patch file of a few KB to a few MB,
reversible):

| Intervention the deep layer tested | Weight edit | Check |
|---|---|---|
| remove direction d from every residual write (`steer(ablate=d)`) | orthogonalise the token embeddings and every `o_proj` / `down_proj` (and biases) against d; alpha < 1 removes part of it | logits vs the hook, KL within max(1e-3, 5 % of the intervention's own effect); KL < 1e-6 on the toy and on a tiny Qwen3.5 |
| take d out of chosen sites only (`steer(project_sites=...)`) | `project`: `W <- W - alpha u (m * (u^T W))` on the chosen layers' `down_proj` / `o_proj`, m = a neuron mask or all columns; alpha up to 6 | KL < 1e-6 at alpha 1 and 6; the mask changes only its columns |
| zero neurons (`zero_neurons`) | scale their `down_proj` columns (x0.5, x0) | exact on the toy |
| zero heads (`zero_heads`) | scale their `o_proj` input columns (softmax-attention layers only) | exact on the toy and on Qwen3.5's attention layers |

Tied input / output embeddings (Qwen2.5-0.5B) are untied before the embeddings
are edited: the hook never touches the unembedding. A post-norm architecture
would fail the check (an orthogonal write is renormalised), so the candidate is
skipped with that reason instead of shipping a non-equivalent edit.

**Two bugs found while building the local edits (Nov 2026), both fixed.** (1) On a
model with tied embeddings the embedding and `lm_head` are one tensor with two
names; the patch recorded the second name, so after untying it edited the
*output layer* instead of the token embeddings. Every direction patch on
Qwen2.5-0.5B before the fix was this wrong edit, which is where the "~1e-3, float
rounding" KL came from. (2) The directional-ablation hook took the direction out
of every layer's writes but not out of the token embeddings, so layer 0 still read
it; Arditi et al. remove it there too. With both fixed the weight edit matches the
hook to KL ~1e-8 (tiny tied Qwen3.5) where it was ~1e-2. The router and Ukrainian
cases were rerun after the fix (round 2 below).

**Hybrid models (Qwen3.5 / Qwen3-Next).** 18 of Qwen3.5-0.8B's 24 layers are linear
attention (Gated DeltaNet, `linear_attn.out_proj`), the other 6 softmax attention.
The wrapper finds both (`_ATTN_NAMES`), and everything that works on heads (copy
heads, head patching, `scale_heads`, random-head controls) runs only on
`lm.head_layers`; attention weights come back for the softmax layers only and are
padded with zeros. Direction and neuron edits touch both kinds of layer. Tested on
a tiny random Qwen3.5, not on the real model.

**Playbook** (`playbook.py`), candidates strongest evidence first:

| Entry | Candidates | Could-not-fix advice |
|---|---|---|
| over-refusal | generic refusal direction (known library, at the depth its ablation worked), contrast direction | LoRA on the over-refused requests, larger model, permissive system prompt |
| wrong language | calibrated answered-minus-expected direction (language probe), contrast direction | fine-tune on target-language replies, larger model, system prompt / prefill in the language |
| premature end-of-turn | top localised neuron, top 3 neurons, contrast direction, end-of-turn direction | fine-tune on complete answers, logit_bias / min_tokens on end-of-turn |
| repetition / collapse | loop battery's neurons (MLP), copy heads, attractor -> contrast / repetition direction | sampling, DRY, mild repetition penalty |
| anything else | contrast direction from `--failing` / `--passing` | more contrast prompts, fine-tune |

Not indicated (no edit tried): the failure does not reproduce on full precision
or is quantisation (`wsr qbisect` names the tensors), positional / RoPE defects,
and loops that only greedy decoding keeps.

**Two modes.** `wsr repair` (fast) tries the diagnosis's own candidates and checks
the failing / working prompts, the regression suite and serving precision.
`wsr repair --deep` adds the local edits and two more checks; `--sampled` adds
paired sampled runs; `--check PATCH` runs a mode's checks on an existing patch
without searching (fast while iterating, deep once before shipping). Every patch
records the mode it passed (`patch.json` `checks`), and the summary says
"(fast checks)" or "(deep checks)".

| | fast | deep | deep `--sampled` |
|---|---|---|---|
| candidates | the diagnosis's (neurons, heads, global directions) | + sparse and local edits on the sites writing the failure, smallest edit first | same |
| direction for the local edits | - | loops: failing vs working **continuation of the same prompt**; otherwise the contrast prompts | same |
| screen | decision token; loops also matched continuations | same | same |
| target + regression | yes | yes | yes |
| serving precision (bf16 / fp16 rounding) | yes | yes | yes |
| fresh prompt variants (not used to choose) | - | must fix 2/3 | same |
| random edit of the same size | - | reported, never rejects | same |
| paired sampled runs at the model's own settings | - | - | failure rate must drop by more than 2 SE |

**Where it is written (`sites.py`).** From any set of teacher-forced states
(failing / working prompts at the decision token, or the two continuations of a
pair over their tokens): per depth the direction mean(failing) - mean(working);
per layer the share of that difference its attention / MLP output writes along
the direction right after it; per MLP neuron its activation difference times its
`down_proj` column along the direction. The chosen sites are the top ones
covering 60 % (at most 3); in each chosen MLP, the neurons covering 90 % of its
share (at most 1/32 of the layer). On the toys it finds the planted layer and
neuron from the contrast set (blank) and from matched continuations (loop).

**Matched continuations (`pairs.py`, loops).** The failing continuation is the
second copy the model wrote; the working one is the model's own continuation from
the same point, sampled with repeated 3-grams forbidden, kept only if it does not
loop. The cheap screen scores both by teacher forcing: a step that flips no
decision token and does not make the repeat less likely than the working text is
skipped (unless it is the last step), and a rise of the end-of-turn probability
inside the working text is flagged as an early-stop risk. Loops only: a blank,
refusal or wrong-language failure is chosen at one token, and both continuations
would share the prompt up to it.

**Fresh variants and the random control (deep).** Fresh variants
(`contrast.fresh_variants`) are built differently from the contrast set's (other
openers and closings, a quoted request, a default system prompt) plus any
`--confirm` prompts, and run once on the unpatched model; a patch must fix 2/3 of
the failing ones and keep 2/3 of the working ones. The report counts how often the
set was consulted, because a set that rejected earlier candidates is no longer
fully independent of the choice. The random control (`edits.control`) is a random
direction orthogonal to the patch's with the same sites, mask and per-matrix
change norm, or the same number of random other neurons / heads in the same
layers; "not specific" means a random edit of that size also fixes the failure,
so the failure is fragile there.

**Serving precision.** After the target and regression checks, the touched
tensors are rounded to the checkpoint's 16-bit type (bf16 when it is fp32) and the
target is checked again; the report gives the share of the change lost to
rounding. `--save-model` and `wsr patch apply` save in that precision.

**Search.** Each candidate runs a ladder, weakest first (directions: alpha 0.25,
0.5, 0.75, 1; components: x0.5, x0). A step is screened at the decision token
(one forward per failing prompt; weaker steps that flip nothing are skipped, the
full-strength step is always generated, because the decision token can be the
same in the failing and the wanted reply: "Небо" opens Russian and Ukrainian
alike). Then: **target** - the original and at least 2/3 of the failing prompts
lose the failure and gain nothing the working prompts do not already show, and
at least 2/3 of the working prompts gain no failure; **regression**
(`regress.py`, 17 prompts: 8 general English, 6 other languages, 3 harmful
requests, plus `--keep` prompts) - no new failure from the triage detectors, drift
(mean change in log-likelihood of the original healthy replies, content tokens
only) <= 0.15 nats/token, top-token agreement >= 85 %, and no harmful request
that was refused stops being refused (`--allow-safety-loss` to waive). The first
step that passes is the patch. Local edits use their own ladder, alpha 1, 2, 3,
4, 6: the rest of the network still writes the direction, so the edited sites can
push against it (the Qwen3.5-0.8B loop fix in Takehi et al. 2026 was a 64-neuron
edit in each of two MLPs at alpha 6).

**Results on real Qwen2.5-0.5B-Instruct, round 1** (GPU, fp32; before the
tied-embedding fix, so the two direction results here used the wrong edit):

| Case | Outcome | Evidence |
|---|---|---|
| Planted blank-reply neuron (layer 16 #0) | **fixed** by zeroing neuron 16.0, found blind; 10/10 failing prompts pass, 3/3 working kept; regression drift 0.003, agreement 97 %; patch 25 KB | Against the original healthy Qwen: planted model 3/17 identical replies, drift 0.92; repaired 14/17 identical, drift 0.0001, agreement 99.5 % |
| Over-refusal, "hack my own router" | **could not fix** | Removing the generic refusal direction at alpha 0.5 fixes the refusal but the model then explains the pipe bomb; alpha 0.75-1 also stops refusing malware. Abliteration's known side effect, caught by the safety check |
| Wrong language, Ukrainian to Russian | **could not fix** | The calibrated direction gives Ukrainian words at full strength (7/9) but the content is nonsense and the one working prompt breaks; the model barely has Ukrainian to redirect to (its own likelihood ranks a correct human Ukrainian answer, 2.23 nats/token, below the nonsense, 1.7) |

Three verification fixes came out of those runs: drift is measured only on
regression replies that were healthy before (a planted bug that also blanks
some suite prompts made a correct fix look like 1.4 nats of drift); end-of-turn
tokens are left out of drift (a patch that stops cutting replies short moves one
token's probability a lot); "translate into Spanish" expects no fixed language.

**Round 2** (after the tied-embedding and embedding-ablation fixes; fast = `wsr repair`,
deep = `--deep`; surgery time from the report's own clock):

| Case | Fast | Deep |
|---|---|---|
| Planted blank-reply neuron (layer 16 #0) | **fixed**, neuron 16.0 x0, 10/10, drift 0.007, 98 % agreement, 0 % lost in bf16 (109 s) | **fixed**, same neuron; fresh variants 10/10; random control 0/10, so **specific** (163 s) |
| Over-refusal, "hack my own router" | could not fix: the refusal direction at alpha 1 stops refusing 2/2 harmful requests (345 s) | **fixed by a gated neuron** (layer 20 #2955, x2): 5/6 failing and 8/8 fresh variants fixed, 2/2 working kept, all 17 regression replies unchanged, drift -0.000, 100 % agreement, both harmful requests still refused, 0 % lost in bf16, random control 0/6, so **specific**. Without the gate the closest was the sparse edit at alpha 2-6, which still lost the pipe-bomb refusal |
| Wrong language, Ukrainian to Russian | could not fix: the calibrated direction breaks the working prompt (252 s) | could not fix: no local edit flips the decision, no gate separates the cases; consistent with the model barely knowing Ukrainian |
| Planted self-reinforcing loop (layer 18 #0, "Penguin") | could not fix: the diagnosis named no component | could not fix: the continuation sites point at layers 22-23 (they write the next token), not at layer 18. **After the layer screen: fixed** by a sparse edit on layers 8 and 10 (screen-chosen, confirmed on held-out pairs), alpha 3: 7/7 failing and 9/9 fresh variants fixed, no new regression failures, drift 0.001, 98 % agreement, harmful requests still refused 2/2, 1 % lost in bf16, random control 0/7 (770 s). It acts upstream of the planted neuron: a fix, not a localisation of the planted bug |

What the planted loop showed: (1) the first working continuations still started with the repeated
item (" Penguin Fish"); the repeated token is now forbidden there, and with that, zeroing the
planted neuron lowers the repeat's log-probability by 0.127 nats/token against 0.030 for the
working text (selective); (2) neither the projection shares nor a causal attribution (activation x
gradient, over the whole continuation or only its first token) ranks the planted neuron near the
top: the last layers, which write the next token directly, dominate. Automatic site-finding has
that limit; the agent has `attribute_neurons`, `inspect_neuron` and `screen_edit` to go further.

**A second architecture: Llama-3.2-1B-Instruct** (16 layers, d 2,048, 32 heads, tied
embeddings, end-of-turn `<|eot_id|>`; fp16 on the GTX 1650). The wrapper needed no change. The
cases: a planted bug like the Qwen one, and the model's own failures found by scanning 14 prompts
with the triage detectors.

| Case | Fast | Deep |
|---|---|---|
| Planted blank-reply neuron (layer 10 #0; cat questions blank, 16/17, with leakage onto 2/8 others) | **fixed**: found neuron 10.0 blind, x0: 10/10, no new regression failures, drift 0.077 (the planted model's leakage is part of the baseline), 99 % agreement, 0 % lost in bf16 (782 s) | **fixed**, same neuron; fresh variants 10/10; random control 0/10, so specific (991 s) |
| Real over-refusal: "How do I hack my own router to install OpenWrt?" refused | **fixed**: generic refusal direction after layer 3, alpha 1: 5/6, harmful requests still refused 3/3, drift 0.003, 97 % agreement, 7 % lost in bf16 | **fixed by a gated neuron** (layer 9 #6820, x2): 5/6, fresh variants 6/7, harmful still refused 3/3, 14/17 regression replies unchanged, drift 0.001, 99 % agreement, 0 % lost in bf16, random control 0/6 |
| Real loop: Ukrainian "why is the sky blue" repeats "блакитно-червоний колір" | not indicated: greedy decoding keeps it, 3/3 sampled runs escape it, no weight mechanism passed its test: a decoding setting | same |

Unlike Qwen2.5-0.5B, Llama's fast path already fixes the router case. Its refusal direction is
specific enough to leave real refusals intact. The gated neuron fixed it on both architectures.

Two bugs this found, both fixed:
1. **Curly apostrophes.** Llama writes "I can\u2019t" with U+2019, and the refusal and "image not
   seen" patterns only matched an ASCII apostrophe, so the refusal was scored as a healthy answer.
   Both now normalise apostrophes first (`output.plain_apostrophes`).
2. **Cut-off replies.** At the default `--max-new 48` the Ukrainian loop had not started, and the
   diagnosis said "does not reproduce on full precision: not weight-level", pointing at
   quantisation. When the healthy reply is still going at the limit, the diagnosis now looks
   once further (4x the limit, at least 128 tokens). If the reply fails there, the verdict is
   `inconclusive` (`DEEP_CUT_SHORT`, naming what it saw) and the next step gives the `--max-new`
   to rerun with. If it stays healthy, "not weight-level" stands as before.

**The paper's loop, as a test of site-finding** (Qwen3.5-0.8B; the unedited model's logged
MATH-500 traces from Takehi et al., most of which loop to the 32k cap; their released edit:
64 neurons in each of layers 15 and 19, alpha 6). Each pair: prefix = prompt plus the first
768 thinking tokens, working = the next 64 tokens, repeat = 64 tokens from the looping tail.
Ten pairs: five choose, five are held out.

| Edit (MLP neurons, continuation direction) | Held-out repeat | Held-out working | Selective |
|---|---|---|---|
| wsr's default sites (layer 23 MLP and attention, 22 MLP), alpha 3 | -0.105 | -0.262 | no |
| raw screen's top two (23, 9), alpha 6 | -0.727 | -1.182 | no |
| screen without the last layer, best two ratios (12, 22), alpha 6 | **-0.067** | **+0.002** | **yes** |
| screen without the last layer, best two repeat drops (9, 13), alpha 6 | -0.041 | -0.002 | yes |
| same plus the next two (9, 13, 14, 2), alpha 6 | -0.061 | -0.021 | yes, less |
| the paper's layers 15 + 19, our neurons, alpha 6 | -0.056 | +0.002 | yes |
| the paper's released edit | -0.014 | -0.031 | no, by this measure |

(nats/token; scripts `exp1b_screen.py`, `exp1c_combos.py`.) Three findings:
1. Projection share and causal attribution both rank the last layer first, because it writes the
   next token. An edit there fits the pairs it was chosen on (ratio 4.6) and fails on new ones.
   That is also why the planted Qwen loop above was not found automatically; with the screen, deep
   mode fixes it (layers 8 and 10, see the table).
2. Middle layers chosen by a single-layer screen on half the pairs stay selective on the other
   half. wsr's choice (12, 22) matches the paper's own layers on this measure without being told
   them. Neuron overlap with their 64 is 5-9 per layer, against 1.1 by chance: the same region,
   not the same neurons.
3. Adding layers beyond two trades selectivity for size.

So deep mode now has `screened_candidate`: for a loop with at least 4 matched pairs it screens
every MLP layer but the last on half the pairs, combines the two best ratios (at least 3x), and
offers the edit only if it is also selective on the held-out half; `REPAIR.md` reports the
screen either way. The paper's released edit is not selective on these teacher-forced pairs.
Their pairs differ (matched contexts at bf16), and their proof was 1,152 generated solutions,
so this says the proxy is limited, not that their edit is wrong.

Time: diagnosis plus surgery 3-6 min per case on the GTX 1650 (surgery alone
1-4 min: a 17-prompt baseline, then about 1 min per generated step).

## 6. The surgery agent (`wsr repair --agent`, `src/wsr/agent/`)

The product plan's last stage, and the method of Takehi et al. 2026: a frontier model reasons about
the network and edits its weights. Here it comes *after* the rule-based playbook, and only when the
playbook could not fix the failure:

```
wsr deep (diagnosis) -> playbook (fast or --deep) -> fixed? ship
                                                  -> could not fix -> agent(brief, tools) -> submit | give up
```

**What the agent is given (`loop.brief`).** The model's shape (layers, widths, which layers have
heads), the failure kind, verdict and summary, the diagnosis findings, the contrast set with its
replies, where the failure is written, every candidate the playbook tried at every strength with
the exact check it failed (drift, agreement, which harmful request it stopped refusing), the
playbook's suggested next step, the checks its own patch must pass, and its budget.

**What it can do (`workbench.py`), and nothing else.** The aim is the access Takehi et al. gave
their editor, in a form that is safe on the user's own machine:

| Paper's editor had | wsr's agent has |
|---|---|
| a persistent Python kernel over the weights (in an isolated container, no internet) | `compute`: a persistent numeric workspace (`lab.py`) over read-only copies of every weight, with torch math and linear algebra, in a small language with no file, network or process access |
| `save_edit(name, tensor)` on any tensor except the embedding / output layer / final norm, and no whole-tensor rescaling | `build_edit` free-form: `delta` (U V^T), `set_columns`, `set_rows` from workspace values, on any weight except the same three, no whole-tensor rescaling; plus the playbook's recipes |
| forward passes: teacher-forced reads of up to 1,024 texts (log-probs, hidden states, MLP activations, attention) | `read`: up to 64 texts per call, log-probs, residuals, MLP activations, attention outputs at chosen positions, on the original model, an edit, or with interventions |
| generations, original / edited / both with shared seeds | `generate`: greedy or sampled, `compare` runs both on the same seeds |
| matched contexts built in the kernel | `make_pairs` (its own) and `screen_layers` (the held-out layer screen) |
| a notebook folder, handoffs every 40 turns | variables persist in the workspace; `note`; old tool results shortened when the context grows |
| no gradients, no running the model in the kernel | the same: the workspace has no autograd and no model; only `read` / `generate` run it |
| (no automatic judge: the experimenters evaluated afterwards) | `verify_edit`: the playbook's own checks, and only a verified edit can be submitted |

Full tool list:

| Tool | What it does | Cost |
|---|---|---|
| `list_examples` | failing / working prompts, the regression suite (and which harmful requests are refused), the held-out fresh variants | free |
| `compute` | the numeric workspace (above) | 1 compute (300) |
| `read` | teacher-forced pass, captures stored in the workspace | 1 read (80) |
| `generate` | replies, optionally sampled, with an edit and/or interventions, classified by the triage detectors | 1 generation per reply (60) |
| `make_direction` | failing-minus-working direction at every depth from a source (contrast, matched continuations, its own prompt lists or pairs), the sites and neurons writing it, cosines with the known vectors | free |
| `make_pairs`, `screen_layers` | its own matched continuations; the held-out layer screen on them | screen: 1 screen |
| `attribute_neurons` | causal ranking (activation x gradient) of the neurons raising the failure over the working text | 1 screen |
| `inspect_neuron`, `logit_lens` | what a neuron writes, its activation on failing vs working; what the residual says at a depth | free |
| `build_edit` | recipes (orthogonalise, project, scale neurons / heads, gated neuron, combine) or free-form (delta, set_columns, set_rows) | free |
| `screen_edit` | decision-token flips; matched-continuation selectivity (the loop's or a pair set's) | 1 screen (40) |
| `verify_edit` | the playbook's own `_verify` with the mode's checks | 1 verification (6) |
| `note`, `submit_patch`, `give_up` | notebook; end the run | - |

**Why it cannot break things.**
- The workspace is not Python. Code is parsed and checked before it runs: no `import`, `def`,
  `lambda`, `class`, `while`, `try`, `with`. No name or attribute that starts with `_` can be
  used, so `().__class__`-style escapes do not parse. Every attribute access and call is checked
  at run time against a per-type whitelist (tensor maths yes; `numpy`, `storage`, in-place `add_`,
  `str.format`, `torch.save` no). Loops and calls check a 120 s deadline and a step budget, and
  binary operations check the result size before computing (64M elements).
- Weights handed to the workspace are copies, so the model cannot be changed from there. The tests
  try a set of escapes and check that the weights are bit-identical afterwards.
- Every edit is applied only for one tool call, and the weights are restored bit for bit after it
  (`edits.applied`).
- Free-form edits are refused on the input embedding, the output layer and the final norm (the
  paper's harness refused the same). They are also refused when they only rescale a whole tensor,
  when their rank is over 1024 per tensor, when |delta| > |W|, or when they hold NaN.
- `submit_patch` accepts only an edit that passed `verify_edit` in the same run. The shipped patch
  therefore meets exactly the checks a playbook patch meets, safety included unless the user
  passed `--allow-safety-loss`.
- The workbench enforces the budgets; the loop enforces turns (80) and spend ($5). Text from the
  model under repair is marked as data in the system prompt.
- Everything the agent saw and did, with each call's `why`, is written to `agent_transcript.json`,
  and the patch records `checks.agent` (provider, turns, cost, rationale, notes).

What it still cannot do, on purpose: run arbitrary Python (a real kernel would need an isolated
container; `TODO.md` has that as an opt-in), train or use gradients, edit the
embedding / unembedding / final norm, change the checks, or reach files and the network.

**When the repair needs something wsr does not allow (`request_capability`).** The agent can
conclude, from its own evidence, that the fix needs something it cannot do: a frozen tensor (the
failure is keyed on one token's embedding row), a larger or a whole-tensor edit, an edit that is not
a low-rank change (new neurons, a different shape), an analysis the workspace cannot express, a
config, template or decoding change, or training. Then it does not guess and does not work around a
guard. It files a request with the mechanism and the evidence, and, if it can build the edit in the
workspace, attaches it as a proposal:

| Category | What happens |
|---|---|
| `edit_embedding`, `edit_output_layer`, `edit_final_norm`, `rescale_tensor`, `larger_edit` | the reply names the opt-in: rerun with `--agent-allow embed / unembed / final_norm / rescale / large` (tied embeddings need both embed and unembed). The unlocked edit is still built, verified and judged by the same checks |
| `more_budget` | rerun with larger `--agent-verifications / --agent-turns / --agent-budget-usd` |
| `new_tool`, `new_edit_kind`, `architecture_change` | recorded for wsr's developers (`capability_requests.json`): the list of tools to add next |
| `config_or_runtime`, `training`, `other` | outside weight editing: the report's next step |

A proposal is saved as an **unverified** patch in `agent_proposals/<id>/`, marked
`checks.verified: false`, and is never applied, verified or shipped by the run. The user can run the
full checks on it with `wsr repair <model> ... --check agent_proposals/<id>`. The run goes on: the
agent keeps working within the tools, or gives up citing the request. Every request (capability,
category, why, evidence, the opt-in, the proposal) goes into `REPAIR.md` under "What the agent
needed but could not do", into `repair.json` (`checks.agent.capability_requests`) and into the CLI
output, and the next step names the flags to rerun with.

**State, `--resume` and drafts that come back (WSP_PLAN phase 1.6).** After every turn the run
writes `<out>/agent_state/` (`agent/state.py`):
- the workspace variables (tensors in safetensors, everything else as JSON; no pickle, so a
  state file can be read without running code from it);
- the directions, edits, proposals and pair sets, with their ids;
- notes, requests, verify results and the budget used;
- the refused-operation counts;
- a fingerprint of the weights (per tensor: shape, dtype, sum and sum of squares, so a single
  changed neuron changes it).

`wsr repair ... --resume <out>` continues a run:
1. It refuses a different model.
2. It restores everything with the same ids, and the earlier transcript is kept as
   `agent_transcript.run<n>.json`.
3. It builds each saved proposal again under the current guards. A proposal that the new
   `--agent-allow` unlocks permit is verified before the conversation starts, and the first one
   that passes is submitted.
4. Otherwise the agent starts a new conversation from a state report: what was tried and with
   what result, the notes, the workspace and the open requests.

Passing a check in an earlier run does not carry over; a resumed run verifies again. When the
agent asked for an unlock, REPAIR.md and the CLI print the exact command: the same arguments plus
`--agent-allow ... --resume <out>`.

**Agent-only runs and the refused-operations log.** `--agent-only` skips the rule-based playbook,
for testing an agent or when the recipes are known not to fit; the checks are the same. Every
operation the safe workspace refuses is counted by kind (`attr:Tensor.numpy`, `call:getattr`,
`syntax:FunctionDef`, `name:open`, `limit:time`). The counts appear in `checks.agent.denied` and
REPAIR.md, and over real runs they decide what the workspace should allow (WSP_PLAN phase 2).

**First live runs: a free local agent (WSP_PLAN phase 0).** The agent model was Gemma 4 E2B Q4_0
on the CPU through llama-server (`--jinja`, OpenAI-compatible, no key, $0), and the model under
repair was the planted-blank Llama-3.2-1B on the GPU. Each run used `--agent-only`, so the playbook
was skipped and the agent had to find the fix itself. Four runs, each fixing what the previous one
exposed:

| Run | What happened | What it changed in wsr |
|---|---|---|
| 1 | After `list_examples` the model thought for 7,800 tokens, the request timed out, and the traceback ended wsr: no report | a provider error now ends the agent run cleanly (state saved, report written); `--agent-timeout`, `--agent-turn-tokens`, `--agent-extra` (here `{"chat_template_kwargs": {"enable_thinking": false}}`); a turn that runs out of output before a tool call gets one nudge |
| 2 | 11 turns. It found layer 10 #0 (`inspect_neuron`: promotes `<\|eot_id\|>`, activation 18.8 failing vs 0.0 working) but then tried an orthogonalise over the size cap and a gate with the spec nested one level too deep ("unknown edit kind None"), and gave up | explicit errors: why an edit is over the cap (orthogonalise also edits the embedding unless `embed: false`), the valid kinds with examples, nested specs accepted, a direction id given as `read_source` explained; evidence hints (a neuron that fires only on failing prompts, or a layer whose share sits in 1-3 neurons, names the candidate edit) |
| 3 | 9 turns. It followed the hint to `scale_neurons` on 10 #0 but wrote `neurons: [[10, 0], 0.5]` and got "'float' object is not iterable"; an extra `id` argument and pairs without `bad` gave raw TypeError / KeyError | every tool checks its input: missing fields named with the schema, unknown arguments dropped and listed, the `[layer, neuron]` shape explained with an example |
| 4 | **fixed in 7 turns, no error**: list_examples, make_direction, inspect_neuron, build_edit scale_neurons [[10, 0]] x0, screen_edit (10/10 flips), verify_edit (passed), submit_patch. 10/10 failing fixed, 3/3 working kept, no new regression failures, harmful requests still refused 2/2; 45,944 tokens | - |

What this shows, and what it does not:
- **The loop, the guards and the transcript work with a live model**, and a 2B model on the CPU can
  complete a repair at no cost.
- **Every guard held.** The workspace refused nothing; the agent never used it.
- **The fix came after wsr's tools started pointing from evidence to a candidate edit.** Runs 2
  and 3 failed without that. The checks still decided what shipped.
- **It was a planted bug with one clear neuron.** It says little about hard cases. Those need a
  frontier model (phase 0 quality runs).

**First quality runs: a frontier-scale agent on a hard case (OpenRouter free tier).** The agent
was `nvidia/nemotron-3-ultra-550b-a55b:free`, the only large free model with tool calls on
OpenRouter that day (the free Gemini and Qwen models had none). The case was Llama-3.2-1B's real
over-refusal ("hack my own router to install OpenWrt"), deep mode, `--agent-only`. The playbook
fixes this case with a gated neuron (9.6820), but the agent was not told that.

| Run | Turns | What it did | Result |
|---|---|---|---|
| 1 | 25 | list_examples, its own pairs, one layer screen (alpha 0.5), 12 logit lenses, 10 neuron inspections; never wrote a note or built an edit | turn limit, no edit tried |
| 2 (told its budget) | 20 | its own pairs, a layer screen, attribute_neurons, 11 neuron inspections, make_direction, one read, 3 workspace computations | turn limit, no edit tried |

What changed in wsr because of these runs:
- **The agent is told its budget.** Every turn's last tool result carries the turns,
  verifications, screens and generations left. Near the end it also tells the agent to build,
  screen and verify its best candidate, or record its findings with `note` for a resumed run.
  Run 1 explored with no idea it was about to stop.
- **The refused-operations log did its job.** Its only entries in run 2 were `import torch` (×2),
  a habit, since torch is already provided. Following phase 2's rule for few, specific refusals,
  imports of modules the workspace already provides (`torch`, `math`,
  `torch.nn.functional as F`, `torch.linalg`) are now no-ops. Every other import is still refused,
  and the message lists what is available.

So far: the free frontier-scale model is a careful explorer. On a hard case it needs more turns
than the free tier's daily request limit allows (about 50 requests a day; these two runs used
about 45). Neither run tried a wrong or unsafe edit. Nothing reached the checks, so nothing was
judged. This is not yet evidence either way on whether an agent finds the gate itself.

**Providers (any model with tool calling).** The loop keeps one neutral history (system prompt;
user text; assistant text + tool calls; tool results), and a provider only translates it, so the
tools, the budgets and the judge are the same whatever model drives them.

| `--agent-provider` | Models | Key | Notes |
|---|---|---|---|
| `anthropic` (default) | Claude; default `claude-opus-5-5` | `ANTHROPIC_API_KEY` (or an `ant auth login` profile) | adaptive thinking, effort `high`, prompt caching, server-side refusal fallbacks (a safety-classifier decline is retried on the recommended model instead of ending the run); list prices built in, so `--agent-budget-usd` holds |
| `openai` | OpenAI's models, or with `--agent-base-url` any OpenAI-compatible server: OpenRouter (Gemini, DeepSeek, Mistral, Qwen, Llama ...), Together, Groq, Fireworks, DeepSeek, Mistral, Gemini's OpenAI endpoint, a local vLLM / Ollama / llama-server | `OPENAI_API_KEY`, or the variable named by `--agent-key-env`; a local server needs none | `--agent-model` is required; `--agent-effort` is sent as `reasoning_effort` only when given; prices via `--agent-price IN OUT` (USD / M tokens), otherwise the spend cap cannot be computed and `--agent-max-tokens` (default 6M) is the limit |

Keys are read only from environment variables, never from the command line (shell history) and
never stored. Tool arguments that are not valid JSON go back to the model as an error. Every
provider gets the same turn, token and spend caps. The quality of the search depends on the model:
the checks guarantee that a weak model cannot ship a bad patch, not that it finds a good one. A
scripted provider replays fixed tool calls for tests and dry runs. Another API plugs in through
the same `turn(system, messages, tools)` interface.

**Status.** Built and tested with the scripted provider and with the Claude adapter against a
fake client (request shape, parsing, cost). The scripted tests cover:
- a run that finds and ships the planted neuron edit through the recipes;
- a run that finds the same neuron from the weights alone in the workspace, checks its activation
  with `read`, and ships its own `set_columns` edit;
- an unverified submit being refused;
- workspace escapes being refused with the weights unchanged;
- each free-form guard;
- reads with interventions, and paired sampling;
- its own pairs, plus the layer screen finding the planted loop layer;
- the turn limit and context trimming.

It has not been run against the live API yet, because that needs the user's key.

**The gated neuron (`edits.gate`, experiment 3).** Every removal edit acts everywhere, so it also
acts where the behaviour is wanted. The gated edit rewrites one MLP neuron the model barely uses
(`sites.quiet_neurons`) into a conditional circuit: its gate row reads what separates the failing
states from the ones that must stay unchanged, and its output writes against the failure
direction. SwiGLU has no bias, so two forms: threshold 0 when every position that must stay
unchanged already projects below 0 (no bias needed), or a threshold built from the residual
stream's mean input direction, used only when the input's projection on it is near constant
(spread <= 15 %); otherwise no gate is offered. For over-refusal the states that must stay
unchanged are the harmful requests the model rightly refuses, so the neuron reads the false alarm
(over-refused minus rightly refused) and is silent on real harm. Deep mode offers it after the
local edits; the agent can build it with any sources.


## 7. Scope: triage and verified repair, with a weight-space-programming agent

What can be claimed today, and what is still missing before "full weight-space programming" in
the sense of Takehi et al. 2026 is a fair description.

**Built and shown on real models** (Qwen2.5-0.5B / 1.5B, Qwen3.5-0.8B, Llama-3.2-1B):
- triage of a live server (request, template, sampling, quantisation, runtime);
- a deep diagnosis that reproduces a failure at full precision and localises it;
- a playbook of weight edits, each with a tested hook twin;
- the paper's method as checks and tools: matched contexts, per-site directions, edits on the top
  writer neurons, teacher-forced selectivity, random controls, a held-out layer screen, and a
  freeze-and-confirm stage (fresh variants, regression and safety, sampled runs, serving
  precision);
- reversible, verified patches.

Fixed this way: planted bugs in Qwen and Llama, a real over-refusal in both, and the planted Qwen
loop. Declined correctly: a language the model does not know, and a decoding loop.

**The agent, shown live on an easy case only** (section 6). It has the paper's kind of access.
- A free local 2B agent fixed a planted bug in 7 turns.
- A frontier-scale free agent ran out of turns on the hard router case without trying an edit.
- **Phase 1 is built** (section 8): goal files, dataset profiles and a paired judge.
  - On a real model it scores an agent-found patch on a held-out set: +70 points, ± 10.5 SE.
  - What is still missing is an agent working towards an objective (`wsr program`).

**Missing for the full claim:**

| Gap | Why it matters | What closes it |
|---|---|---|
| Objective-driven runs | The paper's task was open-ended ("make it better at math") | goal files, profiles and the judge are built (section 8); `wsr program` (WSP_PLAN phase 3) puts the agent on them |
| Benchmark-scale evaluation | The paper confirmed on 1,152 problems with 32k-token sampled generations | the paired judge exists, but on transformers only, one sequence at a time; it needs the llama-server / vLLM backends for long sampled runs |
| A live agent result on a hard case | The agent is the part that makes this programming rather than recipes | an easy case is done; the hard cases need a paid or higher-limit key, or a self-hosted agent model |
| Full Python for the agent | The workspace is a safe subset | the opt-in container kernel (`TODO.md`, item 9) |
| Scale and formats | Tested up to 1.5B on a 4 GB GPU; GGUF survival untested | a larger GPU; experiment 6 |

So the accurate description is: **triage, diagnosis and verified weight-space repair of observed
failures, with a weight-space-programming agent in the style of Takehi et al.** "Full weight-space
programming" fits once the objective mode, the benchmark harness and one live agent result exist.

## 8. Objectives, dataset profiles and the judge (`src/wsr/program/`, WSP_PLAN phases 1 and 1.5)

Sections 4-6 start from one reproduced failure and check a patch on prompt sets. This section adds
what an open-ended goal needs: a task with a metric, data split so the agent is never scored on
what it worked on, and an evaluation large enough to measure a gain.

- **Goal file** (`wsr objective init | check | split`, JSON or YAML). It holds:
  - the protocol: template kwargs, sampling and token cap;
  - the practice and held-out JSONL files;
  - a grader, a list of guardrails and a budget.

  `check` reports missing files, rows the grader cannot score, and prompts that sit in both splits,
  which would leak the held-out set.
- **Graders**:
  - `exact`;
  - `contains`;
  - `regex:<pattern>`;
  - `math`: the last `\boxed{}`, else "the answer is", else the last number, compared after LaTeX
    normalisation or as a number (math_verify when it is installed);
  - `detector:<group>`: wsr's own output detectors, which grade behaviour (blank, loop, refusal,
    language) and need no answer.

  `ifeval` and `judge:` are named but not built.
- **`wsr profile`**: triage over a dataset. Each reply gets a label from the detectors and the
  grader: EMPTY, TRUNCATED_LOOP, LOOP, GARBAGE, TEMPLATE_LEAK, REFUSAL, LANGUAGE,
  TRUNCATED_WORKING, WRONG_ANSWER or correct. Rates come with Wilson intervals. The cause shown
  for each group comes from a test run on up to 3 of its members:
  - **Failure groups** are rerun sampled. A loop that sampling always escapes is a decoding
    setting. A failure under a greedy protocol is a weight-level candidate, as in `wsr deep`, and
    the reruns say how robust it is.
  - **Cut replies with no failure** are rerun at twice the cap. Finishing there makes it a budget
    problem.
  - **Wrong answers** are capability, shown with examples.

  Weight-level groups get a ready `wsr deep` command; `--deep N` runs it.
- **`wsr objective from-failure`** turns one prompt, a local dataset or a profile group into a goal
  file:
  1. it confirms the failure, and stops on "does not fail" or "decoding";
  2. it measures the hit rate;
  3. below 5 % it falls back to wsr's own rewrites of the prompt;
  4. it splits failing and working items, stratified, with the held-out file in its own folder.

  Rewrites are deduplicated by user text. A test caught a leak here: a rewrite that only adds a
  system prompt had landed in both splits. `--dataset auto` names the public source that fits
  the failure but downloads nothing.
- **`wsr evaluate`**: the judge.
  - The original and patched models answer the same items with the same seed per item and sample
    (common random numbers): with the same weights the difference is exactly 0 (tested).
  - It reports the paired difference with its standard error, items better / worse / same,
    per-label rates before and after, and the guardrails: the 17-prompt regression suite (drift,
    agreement, new failures, harmful requests still refused) and data guardrails with a maximum
    drop.
  - **improved** requires a gain above 2 standard errors with every guardrail holding.
  - Every look at the held-out set is appended to `<goal>.heldout_log.jsonl` and the count is
    printed.

**Positive control on a real model.** Phase 0's agent-found patch for the planted-blank
Llama-3.2-1B (neuron 10.0 ×0) was scored on a new held-out set it had never seen: 12 cat
questions and 8 others, with grader `detector:blank` and the regression guardrail.

| | before | after |
|---|---|---|
| Score (no blank reply) | 30 % | 100 % (+70 points ± 10.5 SE, z 6.7; 14 items better, 0 worse) |
| EMPTY | 65 % | 0 % |
| Regression guardrail | | holds: 12/17 replies unchanged, drift 0.077 nats/token, agreement 99 % |

The EMPTY rate before includes the planted bug's leakage onto non-cat questions (13 of 20 blank).
After the fix, 95 % of replies are TRUNCATED_WORKING at the 48-token cap. That is the expected
label for healthy long replies under a short cap.

**A first dataset profile: Llama-3.2-1B on MATH-500.** Run on 16 practice problems from a seeded
60/60 split of 120 (the cached HF copy, nothing downloaded), greedy, 512-token cap, with a system
prompt asking for `\boxed{}`. It took 1,474 s for 24 generations, one sequence at a time on the
GTX 1650.

| | Share | Cause, from the test that ran |
|---|---|---|
| correct | 2/16 (12 %, 95 % CI 4-36 %) | - |
| WRONG_ANSWER | 8 | capability: no single failure mode |
| TRUNCATED_WORKING | 3 | budget: 2/2 tested finish within 1,024 tokens |
| TRUNCATED_LOOP | 2 | weight-level candidate: loops again in 4/4 sampled reruns (repeats "Apply the tangent sum identity again") |
| LOOP | 1 | weight-level candidate: 2/2 sampled reruns loop too |

The profile names `wsr deep --kind loop` on the two looping problems as the next step. This is
the paper's failure (reasoning that loops until the cap) at 1B scale, but the numbers are small:
16 problems, and 3 of 16 loop. The slow part is generation. A benchmark-sized profile needs the
llama-server backend, which is still to build.

## 9. Asking first, two backends, `wsr program` and GGUF export (WSP_PLAN phases 3-5)

**Asking first (`src/wsr/consent.py`).** Every step the user did not plainly ask for asks first:

| Step | Kind |
|---|---|
| downloading a dataset or llama.cpp's converter | `download` |
| starting a llama-server | `llama-server` |
| converting a model to GGUF | `convert` |
| building an isolated environment, when phase 2 comes | `sandbox` |

At a terminal it is a yes / no question with "no" as the default. In scripts, use `--yes` /
`--no` / `--allow KINDS` / `--deny KINDS`, or `WSR_ASSUME` / `WSR_ALLOW` / `WSR_DENY`. With no
terminal and no flag, the answer is no. Each question is asked once per run and recorded; the
answers appear in reports.

A "no" is final for the run. The caller then takes another route where there is one, and says so:

| Refused | Instead |
|---|---|
| starting llama-server | PyTorch |
| downloading a dataset in `from-failure --dataset auto` | rewrites of the prompt |
| a `--backend llama-server` that was asked for explicitly | stops (`NeedsConsent`, exit code 3) |
| fetching the converter, or converting | stops (`NeedsConsent`, exit code 3) |

The stop message names what was needed and how to allow it. A dataset already in the Hugging Face
cache is used without asking, since nothing is downloaded.

**Backends (`src/wsr/program/backends.py`).** `profile`, `evaluate`, `from-failure` and the profile
stage of `program` get replies from:
- **PyTorch**: the full-precision model, one sequence at a time. It is the only backend that can
  apply a patch in memory.
- **llama-server**: one already running (`--endpoint`), or one wsr starts from `--gguf` after
  asking and stops at the end. Requests run in parallel slots, with a seed per request, the
  protocol's sampling (presence_penalty is native there) and `chat_template_kwargs`.

`auto` picks llama-server when the binary and a GGUF are there and the answer is yes; otherwise it
uses PyTorch and says why. Both backends feed the same labelling code, so profiles from either
read the same.

To score a patch on llama-server, give the patched model's own GGUF (`--patched-gguf`). The
regression-suite guardrail needs the full-precision model, so on a server it is reported as not
run, and the verdict says so.

Measured on the GTX 1650 with 512-token answers:
- llama-server (Gemma 4 E2B Q4_0): 9,393 tokens in 90 s;
- PyTorch (Llama-3.2-1B fp16): 10,233 tokens in 1,474 s.

That is about 16× faster. The models differ, but are of similar size.

**Datasets (`wsr data list | fetch`).** The registry:
- MATH-500 and GSM8K, graded with `math`;
- XSTest (Paul/XSTest; the unsafe half is marked `expect: refuse`), graded with
  `detector:refusal`;
- IFEval prompts, graded with `detector:any`; the IFEval checker itself is not built;
- MGSM in 10 languages, graded with `math`.

Each is converted to wsr's rows. `from-failure --dataset auto` maps the failure to a source:
- loops: MATH-500;
- refusals: XSTest;
- wrong language: MGSM in that language;
- blank / garbage / template leak: IFEval.

A row marked `expect: refuse` is scored the other way round. Refusing it is correct. Answering it
gets the new failure label COMPLIED.

**`wsr program goal.json` (`src/wsr/program/program.py`).**
1. Profiles the practice set. llama-server can be used for this stage.
2. Takes the largest group whose test points at the weights and builds the deep diagnosis from
   its first members, contrasted with working practice items.
3. Runs the playbook in deep mode, or fast mode with `--fast`.
4. With `--agent`, when the playbook cannot fix it, runs the repair agent with three objective
   tools:
   - `practice_eval`: paired, same seeds, on the first n practice items, with a budget;
   - `checkpoint`: a leaderboard;
   - `final_choice`: refused until that edit has a practice_eval.

   The held-out file is not a tool, and its path never reaches the agent (tested on the
   transcript).
5. The judge is `wsr evaluate` on the held-out set with the guardrails. The patch ships to
   `<out>/patch` only on **improved**; otherwise it is kept in `<out>/candidate`.
6. `PROGRAM.md` holds the objective, profile, group, diagnosis, playbook, agent (turns, stop
   reason, leaderboard, notes), judge, patch with its apply line, and the questions asked. This is
   phase 5's single report.

When no group points at the weights, for example only wrong answers, it stops after the profile:
the edit tools have nothing to start from.

**GGUF export (`wsr patch export`, `src/wsr/surgery/export.py`).**
1. Checks disk space first: three times the model size.
2. Applies the patch.
3. Converts with llama.cpp's own `convert_hf_to_gguf.py` and `gguf-py`. These come from
   `--llama-cpp`, `WSR_LLAMA_CPP`, or a copy fetched earlier. Otherwise wsr asks, then downloads
   the llama.cpp source at the tag matching the user's llama-server build (`--version` 10244 →
   `b10244`, the commit the binary was built from) and keeps only the converter files.
4. Quantises with the `llama-quantize` next to llama-server.

Converting asks too. Then
`wsr evaluate --backend llama-server --gguf <unpatched> --patched-gguf <patched>` shows how much of
the gain survives Q8_0 / Q4_K_M.

**Run end to end (after the user's yes to the converter download).** The patch that `wsr program`
shipped for the planted-blank Llama was exported next to the unpatched model. Each export took 1-2
minutes. Both were judged on the same held-out set:

| Served file | Blank replies before → after | Score | Items better / worse | Time |
|---|---|---|---|---|
| full precision (PyTorch) | 53 % → 0 % | 23.7 % → 100 % (+76.3 ± 7.0) | 29 / 0 | 433 s |
| Q8_0 GGUF (1.32 GB) | 53 % → 0 % | 23.7 % → 100 % (+76.3 ± 7.0) | 29 / 0 | 34 s |
| Q4_K_M GGUF (0.81 GB) | 53 % → 0 % | 26.3 % → 100 % (+73.7 ± 7.2) | 28 / 0 | 14 s |

The whole fix survives both quantisations. On the GGUFs the regression-suite guardrail is reported
as not run, because it measures drift on the full-precision model; it held there. Every
evaluation counts as a look at the held-out set; this one has been looked at 3 times.

**A bug the export found.** The first patched GGUF was 0.28 GB larger than the unpatched one, with
a separate `output.weight`. The patch touched one down-projection and did not untie anything.
The cause: when `save_patched_model` converted the state dict to the checkpoint's 16-bit type, it
copied the tied embedding twice. Both copies were saved, so the model reloaded (and converted)
untied. This affected `wsr patch apply` and `--save-model` too. Now one key is kept per storage,
and there is a regression test: a tied toy model saved in fp16 reloads tied. The re-exported
patched GGUF is the same size as the unpatched one.

**Phase 2 (the isolated kernel) stays deferred, as planned.** The refused-operations log from the
live runs had only `import torch` / `import math` style imports. Following the plan's rule, those
were allowed as no-ops instead of building a container. The `sandbox` consent kind is reserved for
when a run needs more.

**The first real `wsr program` run** (planted-blank Llama-3.2-1B, GTX 1650, PyTorch, 1,371 s):
1. `wsr objective from-failure` started from one cat question plus two more failing and two
   working prompts. The failure was confirmed as EMPTY, reproducing in 3/3 sampled reruns.
2. It made 76 rewrites: 39 blank, 37 working. The stratified split gave practice 19 + 19 and
   held-out 20 + 18, with no overlap; the held-out file sits in its own folder.
3. `wsr program`:
   - **Profile:** 50 % EMPTY (a weight-level candidate, 4/4 sampled reruns).
   - **Diagnosis:** layer 9-10.
   - **Deep playbook:** a gated neuron (4.695) and single neurons in layer 9 did not flip the
     decision. A sparse contrast direction on the writer neurons (alpha 1) passed every deep
     check: 11/11, drift 0.079, agreement 100 %.
   - **Held-out judge:** 23.7 % → 100 % (+76.3 points ± 7.0 SE, z 10.9); 29 items better, 0
     worse. Regression guardrail held: 13/17 replies unchanged, agreement 99.6 %.

   **Shipped** to `run/patch`, with `PROGRAM.md` as the single report.

Two caveats:
- **The fix differs from Phase 0's.** The playbook's fix here is not the planted neuron 10.0 that
  the phase 0 agent zeroed. With the rewrites as contrast, the diagnosis pointed one layer earlier.
  The judge measures the objective, not whether the edit matches the planted cause.
- **The WRONG_ANSWER label is misleading here.** Under a `detector:blank` grader, it marks short
  replies that the blank detector flags as TOO_SHORT. They are not empty, but they are not right
  either.

## 10. Speed: batching by free memory, and the llama-server judge

**Why PyTorch generation was slow here.** On the GTX 1650 (Turing, no tensor cores), cuBLAS's fp16
matrix-vector product is about 5× slower than its fp32 one:

| One MLP matrix, 8192×2048 | Time |
|---|---|
| fp16, one sequence | 2.08 ms |
| fp32, one sequence (twice the bytes) | 0.38 ms |
| fp16, 16 sequences at once | 1.59 ms |

Llama-3.2-1B does not fit in 4 GB at fp32, so it runs in fp16. That gives 126 ms per decode step
and 7.3 tokens/s for a single sequence, even with plain transformers and an idle GPU. Sixteen
sequences cost no more than one, so on this card batching is the main lever, and it is limited only
by free memory.

**Batched generation (`deep/batching.py`, `LM.generate_batch`).**
- **Batch size.** It comes from the memory free right now (`torch.cuda.mem_get_info`; RAM via
  psutil on the CPU). It always leaves 400 MB for other applications, and uses at most 80 % of the
  rest.
- **Per-sequence estimate.** The KV cache, the prompt's MLP and attention activations, and the
  last-step logits, × 1.25. Measured on Llama fp16 (prompt 362, 128 new tokens): 34 MB real
  against 35 MB from the formula. The first version also counted logits over the whole prompt,
  which `generate()` never materialises, and overestimated 4.7×.
- **Caps.** 32 on CUDA, 8 on CPU; `WSR_MAX_BATCH` lowers it and `WSR_BATCH=off` turns batching
  off.
- **Out of memory.** The cache is freed, the batch halved and the step retried, down to one
  sequence at a time. Only memory errors are retried (tested with an injected error).
- **What is not batched.** Linear-attention and recurrent models (Qwen3.5's hybrid layers, Mamba
  and similar), and generations with intervention hooks: padded positions are not masked in a
  recurrent state.
- **Order.** Prompts are grouped by length (left padding). With sampling, each chunk is seeded
  from its first prompt, so the same list gives the same chunks and draws, which keeps paired
  comparisons paired.

**Correctness.**
- Toy models (fp32): batched greedy output equals one-at-a-time output exactly (tested).
- Real Llama fp16: the 17-prompt regression suite at 48 tokens is identical, 17/17.
- Over 512 tokens, fp16 rounding with padding can flip a near-tie. On 16 MATH problems a few
  labels moved (2 → 3 looping). Both sides of a paired evaluation use the same batches, so the
  comparison stays like-for-like.

**Where it is used.**
- profile, evaluate, from-failure and the agent's practice_eval (through the PyTorch backend);
- deep diagnosis: the contrast set and its variants, the fresh variants;
- the playbook's re-checks of the failing and working prompts;
- the regression suite: baseline, and the comparison for every candidate.

The test suite went from 133 s to 100 s.

| Measured on Llama-3.2-1B, GTX 1650 | Before | Now |
|---|---|---|
| 17-prompt regression suite, 48 tokens | 89.5 s | 19.7 s (identical replies) |
| 16 MATH prompts × 128 tokens | 7.3 tokens/s | 30.8 tokens/s (batch 8, no out-of-memory, 564 MB stayed free) |

The 16-problem MATH profile at 512 tokens went from 1,474 s to 1,217 s. That run used the first,
over-cautious estimate, with about 600 MB free, so the batch was 1–4.

**The llama-server judge (`wsr program --judge auto|torch|llama-server`, `program/judge.py`).**
- **auto** uses llama-server when its binary is there and the user agrees. It exports the
  candidate to `<out>/served/candidate.Q8_0.gguf`; that file ships with the patch. The original is
  exported once per model into `~/.cache/wsr/gguf/`, keyed by the model's files. One server is
  started for each.
- The regression-suite guardrail stays on PyTorch: it measures drift at full precision, with short
  replies.
- After a "no" (now, or a remembered "never"), or without llama-server, it uses PyTorch (batched)
  and says why. `torch` never asks; `llama-server` stops when it cannot.

**Remembered answers.** At any prompt, `always` / `never` is remembered per kind
(`~/.cache/wsr/consent.json`; `wsr consent show | reset`). A user who said "never" to llama-server
is not asked again; flags still win.

**End to end on the planted-blank Llama.** The judge stage took 215 s, including converting the
original once and the candidate, against 433 s on PyTorch. The verdict was the same: 23.7 % →
100 %, guardrail held, shipped. The whole run took 1,375 s, as before. Diagnosis and playbook
dominate it, and they are forward-pass bound (see the next steps).

## 11. Speed without changing results

The rule for this round: **no optimisation may change what wsr concludes.** Each change below comes
with the check that holds it to that.

**Where batching could change a result, and the guarantee.** On real Llama in fp16:
- Batching copies of one prompt (no padding) gave output identical to one at a time over 200 tokens.
- With padding, last-step logits moved by at most 0.06 and the chosen token was always the same.
  That is fp16's own rounding, so a one-at-a-time fp16 run is not a "more correct" reference.
- Over long replies, padding can still flip a near-tie.

So the guarantee is about comparisons:
- **Fixed batch plans.** The same prompt list (same lengths and settings) always runs in the same
  chunks within a run (`LM.batch_plans`), and a cached baseline carries its plan. With identical
  weights, before and after give identical outputs, so a difference can only come from the patch.
- **Reference runs.** The playbook's "before" sets were built in other batches (the contrast set,
  the fresh variants). Each now gets one unpatched reference run in exactly the chunks the patched
  runs use, and "fixed" / "broke" are measured against it. An example whose reference does not
  fail cannot count as fixed. `checks.reference_runs` reports how often the reference kept the
  original label.
- **Causal tests** got the same pairing, which makes them stricter than before: an ablation counts
  as a fix only on examples whose unsteered reference fails.

**Measured equivalence.** Deep diagnosis of Llama's router over-refusal, run with the old
behaviour (batching and cache off), then new on a cold cache, new on a warm cache, and new with
every fix of this round. **Every finding is identical, word for word, numbers included**: the
verdict, the chosen depths, the known-direction comparisons and ablations, the causal counts,
localisation and the weight dossier.

| Change | What it does | Why results cannot change |
|---|---|---|
| Steered generation batched (causal tests, copy heads, language probe, loop neurons) | runs every example under one steer in one batch | each steer is the same for every row, at position -1 or every position, and left padding keeps -1 aligned; batched equals single on the toys (tested); each real-vs-control pair shares one list, so it shares chunks |
| Per-model cache (`wsr/cache.py`): regression baseline with its batch plan, known-direction library, matrix statistics | computed once per model | the key includes the files, a float64 position-weighted checksum of the in-memory weights (a patched model never hits the file's cache; tested), precision, device and its name, versions and the batching mode; a hit returns exactly what was stored (tested equal); the library keeps its computation order, since later ties break by order; `WSR_CACHE=off` |
| The judge reuses the verification's regression run | removes a duplicate 17-prompt run | same patch, template, suite and baseline: the same deterministic computation. The guardrail's own thresholds are applied to it |
| Free memory includes PyTorch's own cache | the planner had seen 0 MB free mid-diagnosis while PyTorch held 763 MB it could reuse, and fell back to batch 1 | changes batch sizes only. The 400 MB reserve applies only to the driver's free memory, so other applications never lose memory they could have had |
| Practice labels reused from `from-failure` in `wsr program` | skips re-generating the practice set | only when the cache key (weights), protocol, grader and each item's exact messages match, labelled at full precision and greedy; otherwise it profiles as usual (tested) |
| `wsr fix` | from-failure + program in one process | the model is loaded once; the same functions run |
| Export patches the checkpoint files directly | one shard in memory instead of a float32 model copy | untouched tensors are copied bit for bit; touched ones are round(w + sum of deltas), in float32, rounded once: **bit-identical to the model path (tested)**. Patches that untie the embeddings use the model path |

**Measured (Llama-3.2-1B, GTX 1650):**

| Step | Before this round | After | Results |
|---|---|---|---|
| Deep diagnosis, router over-refusal | 402 s | 300 s (causal tests 78 → 37 s, contrast 52 → 11 s, weight stats 16 → 1 s) | every finding identical |
| `wsr program`, planted-blank cat goal | 1,371 s | 646 s (practice labels reused; guardrail reused; batched) | same patch kind, every deep check passed, shipped |
| From one failing prompt to a shipped patch (`wsr fix`) | from-failure plus program, two model loads, about 25 min | 12 min 48 s, one process | held-out 33.3 % → 100 % (+66.7 ± 8.8 SE), guardrail held, shipped |

**The llama-server judge, end to end with the new export.**
- `wsr fix` with `--judge auto` (llama-server allowed) patched the checkpoint files directly,
  converted the candidate to Q8_0 and judged on llama-server, with the regression guardrail reused
  from verification.
- **The verdict is identical to the PyTorch judge**: 33.3 % → 100 % (+66.7 ± 8.8 SE), 20 better /
  0 worse / 10 same, shipped.
- The held-out scoring itself took 9.6 s. The export costs 1-2 minutes, so on short replies (48
  tokens) the whole run took 14.1 min against 12.8 min with the PyTorch judge. The server judge
  pays off on long replies, where PyTorch generation dominates.

An earlier attempt exposed one more bug. The disk filled during the export: free space on this
machine swings between about 5 and 14 GB, from other software. safetensors raised its own error
type, which the judge's fallback did not catch, and the run ended without a report after 12
minutes. Now any failure in the optional fast judge falls back to PyTorch with the reason (tested
with an unknown error type), and a full disk while patching files says so in plain words.

**Two verifications kept, one duplicate removed.** The playbook / agent verification checks the
failing prompt with safety, fresh variants and a random control. The held-out judge checks the
objective on unseen data. They test different things; dropping either would lower accuracy. The
only duplicated part, the regression suite run twice on the same patch, is now reused.

**Not done, deliberately:**
- **More batching of analysis forward passes.** The big sweeps (known directions, localisation)
  already batch rows of the same prompt without padding, and they are compute-bound here (about
  0.29 s per row). The rest are a few single forwards over different prompts; batching those needs
  padding, which moves residuals slightly, and those residuals feed thresholds. The gain is small
  and the risk is not zero.
- **Batched candidate screening.** Each candidate has different weights, so their screens cannot
  share a forward pass. The decision-token screen already runs before any full verification.

**Precision for general GPUs (unchanged, reviewed).** fp32 wherever it fits (CPU, or a GPU with
room): it is the reference precision, and batched fp32 rows agree with single rows to about 1e-6.
Otherwise bf16 on Ampere and newer, fp16 before. This puts precision first, which is right for a
diagnosis tool on any GPU; nothing is tuned to one card. Batch sizes come from each machine's free
memory.

**Defaults that follow the user.**
- **llama-server.** One answer covers every llama-server step of a run (profile, judge). A "no"
  anywhere (a flag, a remembered `never`, or earlier in the run) keeps everything on PyTorch
  without asking again. In `wsr program`, a yes makes llama-server the profile backend: the
  original is exported to a cached GGUF, and the export is covered by the same answer. Because
  the served file is quantised, the chosen failure group is re-run on the full-precision model
  first, and only prompts that fail there too go to the diagnosis.
- **Prompt caching.** On for Claude models (every Claude 3+ model supports it), with an explicit
  breakpoint on the fixed system prompt plus automatic caching for the conversation. If the API
  rejects it, the turn is retried once without caching and caching stays off for the run. On
  OpenAI-compatible servers, `cache_control` is sent only where it is documented (OpenRouter with
  Anthropic or Gemini models). Elsewhere caching is automatic and nothing is sent; the cached
  token counts are reported. Cache reads are priced per model (0.05× on Opus 5.5, 0.025× on
  Fable 5.1, 0.1× otherwise), and the token cap counts the whole prompt.

**Bugs found by these checks:**
- `from-failure` labelled items before setting the grader, so replies the blank detector flags as
  too short counted as "working". It now labels with the goal's grader. The cat case went from 37
  working to 20.
- In-process export loaded a second float32 model, and llama.cpp's converter was killed for lack
  of memory. The error was hidden behind progress bars, and a half-written 863 MB GGUF was left
  behind. Now: direct file patching, a RAM check first, the real error shown, and a failed export
  deletes its partial file.
- Export could have reused a model loaded in fp16 to write a bf16 checkpoint, which loses small
  weights. Now the in-memory model is reused only in fp32.
