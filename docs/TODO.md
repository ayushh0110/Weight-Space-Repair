# Do later

Things built but not verified live, or known limits. Each line says why it is open.

## Not tested live

- **Server without a vision projector.** `IMAGE_NO_VISION`, `VISION_OFF` and the
  "not supported / mmproj" branch of `IMAGE_REJECTED` rely on llama-server's
  documented error text; the test server always had `--mmproj` loaded.
- **Ollama native API** (`--api ollama`): images in `messages[].images`, NDJSON
  streams, `think=false`, penalty options. Only exercised against a fake server.
- **vLLM, LM Studio, OpenAI-hosted endpoints.** Backend detection and their
  error texts are from docs, not live runs.
- **Audio and video inputs.** The server reports `audio` and `video` modalities;
  `wsr` checks images only.
- **Remote image URL that the server can fetch.** Only the failing case
  (`cannot make GET request`) was seen live.
- **`--alt` quantisation differential** with a second live server on another
  quant. Covered by fake-server tests only.
- **Context probe above 2048 tokens.** The live server had `-c 2048` shared by
  4 slots, so the 4k / 8k / 16k levels were never reached live.

## Known limits

- **Repetition avoidance on Cyrillic, Arabic, CJK.** Healthy text there already
  reuses few words (0.01-0.17), so the single-reply detector skips it; only the
  penalty A/B (line count, length) covers these scripts.
- **Penalty A/B needs a strong penalty in the request.** A server-side default
  penalty (Ollama / LM Studio 1.1+) is invisible to `wsr`.
- **Script substitution under a penalty** (Hindi reply switching to Urdu
  sentence endings `ہے۔`) is seen but not detected.
- **Off-task but fluent answers** with no request-side cause are still out of
  reach for generic detectors.
- **Image token cost** is not estimated: prompt-size checks count text only and
  trust the server's `prompt_tokens` for the rest.
- **Two requests overlapping on a small shared context** ("Context size has
  been exceeded" mid-stream) seen once, not reproduced on purpose.

## Deep layer

- **Real models so far:** Qwen2.5-0.5B-Instruct (over-refusal, wrong language,
  list loop, and a planted blank-reply neuron found blind). Not yet: other
  architectures (Llama, Gemma), natural blank replies.
- **Planted bug leaks.** The real-Qwen plant fires on some non-cat questions
  ("Why do frogs croak?"); a single linear read cannot separate "cats" from
  "other animals" at layer 16. Fine for localisation ground truth, not as a clean trigger.
- **Batched generation and batched ablation scoring** (different prompts need
  padding) are not built; only patching is batched.
- **Close-call failures** (refusal token p < 0.5): many perturbations tip them,
  so the permutation null rejects most directions (the cake case ends with no
  specific direction). A bigger contrast set (`--failing` / `--passing`, or
  generated paraphrases) is the way to a specific one; not built yet.
- **Speed:** ~7 min per run for 0.5B on CPU in float32; mostly generation in
  the contrast set and causal tests. Batching the forward passes would cut it.
- **Small contrast sets.** Near-identical variants give 3-5 failing and 3-5
  passing prompts, so leave-one-out accuracy is coarse (50 % on Qwen) and the
  direction is noisy; the ablation-chosen direction still works causally. More
  `--failing` / `--passing` prompts help.
- **Direction writers assume a plain pre-norm residual** (Llama, Qwen,
  Mistral). Gemma's post-attention / post-MLP norms rescale each write before
  it is added, so the writer shares are approximate there.
- **AWQ-processed GGUFs** (Qwen's official files): beyond folded scales,
  errors are higher than rounding alone (Q8_0 ~3.8 % instead of ~0.4 %), on
  each block's largest weights. Handled by a per-file baseline; the exact
  processing is not identified. Files with folding in only some layers (< 3
  matrices) still use the type's constant baseline.
- **Q5_K** decoder not yet seen in a real file (Q4_K, Q6_K are checked on
  Qwen2.5-0.5B Q4_K_M against the original: 8.2 % / 3.9 % median error).
- **IQ*, TQ*, MXFP4 GGUF types** are skipped by `wsr quant`.
- **Quantisation error on the real Gemma file** needs Gemma 4 E2B's safetensors
  (~10 GB); its GGUF-to-HF name map (per-layer embeddings, KV-shared layers) is untested.
- **Loop failures** pick the decision point at the second copy of the repeat;
  not yet exercised on a real loop.
- **SAE / transcoder features** (SAELens, Gemma Scope 2, circuit-tracer) and
  attribution patching / ACDC: not built.
- **Agent weight reading** (step 7, the reasoning agent): the deterministic
  dossier is built; the agent loop belongs to Milestone 4.
- **GPU**: tested on a 4 GB GTX 1650. float32 is the default where the model
  fits (0.5B); 1.5B drops to fp16 there, where batched rows are not
  bit-identical to single rows (logits move up to 0.05). bfloat16 is not native
  on Turing. Models over ~1.5B do not fit.
- **Generation dominates deep run time** (contrast set, causal tests, loop
  battery). Exact speedups left: none found; batching, `torch.compile` and CUDA
  graphs all change the arithmetic or the sampled streams.
- **Triage cannot run probes in parallel** on llama-server without changing
  replies (not batch-invariant, measured). A server started with `-np 1` gives
  nothing to parallelise either. Ollama / vLLM batching not measured.

## Calibrated probes

- **Language probe covers 12 languages** (en fr de es pt it ru uk zh ja ko hi);
  failures involving others (th, mr, ne, ar ...) skip it. Calibration takes ~8
  min on a GTX 1650 for 0.5B (147 short generations), then it is cached.
  Calibration questions were written by me, not native speakers.
- **"Right plan, wrong token"** (Answer in French -> English) is diagnosed but
  its location is only "the readout"; tracing which late component flips the
  first token is not built.
- **Copy-head probe found no decisive copy circuit** on the one real loop
  (Qwen 0.5B animal list: top 8 copy heads remove 38 % of the lead, random
  heads 30 %). It is untested on a loop that is a strong copy (p >> 0.1), e.g.
  long-context verbatim repetition.
- **Confidence cap when nothing beats the null** is checked by unit logic only,
  not by a fresh end-to-end run.

## Loop battery

- **MLP self-reinforcement on a real model:** no real loop driven by one neuron
  at the repeat position yet (the planted neuron seeds "Penguin Penguin", then
  attention carries the list loop). Next: a token-level self-loop plant.
- **Quantisation bisection** is built (`wsr qbisect`). Open: it reproduces in
  PyTorch with dequantised weights, not llama.cpp's integer kernels, so a
  failure that only the kernels cause shows as `quantised_passes`; K-quant files
  and GGUFs of other architectures are not bisected yet; a loop caused by
  quantisation (as opposed to a refusal or garbage) has not been bisected.
- **Real long-context RoPE failure:** only planted aliasing tested.
- **Attractor rule** uses an escape test on up to 3 failing prompts; a natural
  attractor loop (not planted) has not been seen yet.

## Surgery

- **Loop entries not run on a real model.** MLP-neuron and copy-head edits are
  built and checked against their hooks on the toy; no real loop with a
  weight-level mechanism has been repaired yet (the natural Qwen loop is a
  greedy-decoding case, correctly "not indicated").
- **Playbook entries from the report not built as dedicated recipes:** reasoning
  that never ends, excessive length, formatting habits, persona / tone. They run
  through the generic contrast-direction entry with user-supplied `--failing` /
  `--passing` prompts.
- **Meaning is not checked.** The target and regression checks use the triage
  detectors (blank, loop, garbage, language, refusal); a fluent-looking reply in
  the right language can still be nonsense (seen on Ukrainian). Base-model
  likelihood does not separate them for a language the model barely knows.
- **Regression suite is small** (17 prompts, 48 tokens each). Deep mode adds fresh
  prompt variants of the failing prompt; a bigger general suite would still make
  "fixed" stronger.
- **Matched continuations are built for loops only.** Blank, refusal and
  wrong-language failures are chosen at one token, so their local edits use the
  contrast-prompt direction. A prefilled working continuation (force the wanted
  first token, let the model go on) would extend them; not built.
- **`--sampled` checks, it does not search.** The contrast set is still built
  greedily, so a failure that appears only under sampling (a thinking model's
  long reasoning loop) is not diagnosed in the first place; `--sampled` only
  confirms a patch for one that is.
- **Hybrid models tested on a tiny random Qwen3.5 only**, not on Qwen3.5-0.8B.
- **GGUF patches.** Patches apply to Hugging Face weights; serving the fix from
  llama.cpp means re-converting the patched model.
- **Saving the patched model** needs about the model's size free on disk (it is
  now saved in the checkpoint's 16-bit precision, not fp32).
- **The reasoning agent** (free-form exploration after the playbook) is not built.

## Done since the last round

- **Llama-3.2-1B-Instruct**: the wrapper and the surgery run unchanged (DESIGN.md, "A second
  architecture"). Planted blank fixed in both modes; the model's real over-refusal fixed in both
  (deep: gated neuron); its Ukrainian loop correctly sent to decoding settings. Gemma and Mistral
  are still untested.
- Refusal detection misses Llama 3's typographic apostrophe: fixed.
- A reply cut at `--max-new` before the failure showed was called "not weight-level": now
  `inconclusive` with the fix (raise `--max-new`).
- The agent can file `request_capability` when the fix needs something its tools do not allow;
  opt-in unlocks via `--agent-allow`; unverified proposals saved for `--check`.

## Towards full weight-space programming (DESIGN.md section 7)

1. A live agent run on a real case (Claude or any OpenAI-compatible model).
2. `--objective`: a practice set to work on and a held-out benchmark the agent never sees.
3. A benchmark harness for the final patch (sampled, long replies, on vLLM or llama-server).
4. Reproduce the paper's Qwen3.5-0.8B result on a MATH-500 subset with the agent.
5. The opt-in container kernel (item 9 below).

## Experiments worth running (from building surgery and reading Takehi et al. 2026)

Each has a hypothesis, a test that can fail, and what it would add to wsr.

1. **Find the paper's loop fix automatically.** Takehi et al. released Qwen3.5-0.8B's
   unedited generations (looping traces, 58 files, 4.8 MB) and the exact edit (64
   neurons in each of layers 15 and 19). Teacher-force the logged traces (repeat
   tail vs the productive part before it), run `sites.locate`, and compare the
   chosen layers and neurons with theirs; then `wsr repair --check` their edit
   rebuilt as a patch. Hypothesis: our sites rank layers 15 and 19 MLP first and
   overlap most of their 128 neurons. Adds: a real-model validation of the
   local-edit path without generating 32k-token traces, and the thinking-loop
   entry.
   **Run; the hypothesis was wrong, and the fix is in deep mode.** On 6-10 looping
   traces (prefix = prompt + first 768-1024 thinking tokens; working = the next
   64-96; repeat = from the looping tail): projection share ranks layer 23 MLP
   first (26-31 %), layer 15 MLP 17th and layer 19 MLP 11th of 48 sites; causal
   attribution ranks layer 15 fourth and 19 eighteenth of 24 layers. Neuron overlap with their 64 is 8-9 (layer
   15) and 5-9 (layer 19) against 1.1 by chance: above chance, far from "most".
   On held-out pairs, every edit that includes layer 23 lowers the working text
   more than the repeat (alpha 3: repeat -0.105, working -0.262 nats/token).
   Their layers 15 + 19 with our neurons are selective (alpha 6: -0.056 /
   +0.002). A per-layer screen that skips the last layer and keeps the two best
   ratios picks layers 12 and 22 and is as selective (alpha 6: -0.067 / +0.002).
   Now `playbook.screened_candidate`, deep mode, loops with >= 4 pairs; with it
   the planted Qwen2.5 loop, unfixable before, is fixed (layers 8 + 10, alpha 3,
   every deep check passed, random control 0/7). Their
   released edit on our proxy pairs: -0.014 / -0.031, not selective by this
   measure; their pairs and their 1,152-problem check differ from ours, so this
   does not say their edit is wrong, only that teacher-forced pairs are a proxy.
   Scripts: `exp1_q35.py`, `exp1b_screen.py`, `exp1c_combos.py` (scratch).
2. **Edit the trigger, not the behaviour (over-refusal).** The router case fails
   because removing the refusal direction removes all refusals. Contrast the
   over-refused prompt with harmful prompts the model *should* refuse (both
   refused, so the output side is the same): the difference is the false "this is
   harmful" signal, written earlier in the network. A local edit there should
   leave real refusals alone. Hypothesis: the router case becomes fixable with the
   safety check intact. Adds: a safety-preserving over-refusal recipe.
3. **Gated edits: a new circuit instead of a removal.** Every edit so far is
   unconditional, so it also acts where the behaviour is wanted (Ukrainian broke
   the working prompt; refusal removal broke safety). The paper's car fix added a
   side path that switches on only in the failure state. In an LLM: rewrite one
   unused MLP neuron so its gate reads the trigger (the contrast direction at an
   early layer) and its output writes against the failure direction. Hypothesis:
   it fixes the failing prompts while the regression suite stays near zero drift,
   because the neuron is silent elsewhere. Adds: a fourth edit type
   (`edits.gate`), the first that adds rather than removes.
   **Done (with 2).** `edits.gate` plus `sites.gate_plan`; for over-refusal the gate
   reads over-refused minus rightly-refused, which is experiment 2. Result on
   Qwen2.5-0.5B: the router case went from "could not fix" to fixed (layer 20
   neuron 2955, x2: 5/6 failing and 8/8 fresh variants fixed, all 17 regression
   replies unchanged, both harmful requests still refused, random control 0/6).
   Ukrainian: no layer gives a gate that fires on the failure and stays off
   elsewhere, so it declines.
4. **A cheap fixability predictor.** Before generating anything, the matched
   continuations' dose-response says whether one direction separates the failure
   (the repeat's log-probability falls while the working text stays) or not (both
   fall: no single component, like the paper's Gemma). Hypothesis: the
   selectivity at alpha 1 predicts "fixed" vs "could not fix" on the planted and
   real cases. Adds: an early "unlikely to be fixable by an edit" answer in
   seconds.
   **Partial, from the logged runs (4 cases, not a validated predictor).** In both
   fixed cases some candidate flipped decision tokens before the final strength;
   in both unfixable ones (Ukrainian, planted loop) none did at any screened
   strength. The gate, which fixed the router, is not screened this way, so a
   "no flip" answer should not stop deep mode from trying it.
5. **Prefilled pairs for single-token failures.** Extend matched continuations to
   blank, refusal and language failures: force the wanted first token (prefill)
   and let the model continue, so both sides share the prompt and differ only in
   the reply. Hypothesis: the direction from those pairs fixes with less drift
   than the contrast-prompt one. Adds: same-prompt directions for every entry.
6. **Do patches survive GGUF quantisation?** Apply a fixed patch, convert to GGUF
   (Q8_0, Q4_K_M), rerun the triage on llama-server. Hypothesis: sparse
   high-strength edits survive Q8_0 and degrade at Q4. Adds: a `--serve-dtype q4`
   style check and the GGUF patch path.
   **Not run.** Needs the `gguf` package and llama.cpp's `convert_hf_to_gguf.py`
   (only `llama-quantize` is installed here). `--serve-dtype` covers bf16/fp16
   only.
7. **Weight-only audit.** The planted bugs are visible in the weights alone (a
   neuron whose output column aligns with the end-of-turn unembedding row, or
   writes its own input token). Scan every `down_proj` column for alignment with
   special-token and own-input rows, against the distribution in a clean model.
   Hypothesis: both planted neurons rank first, with few false alarms on clean
   Qwen. Adds: a prompt-free "suspicious neurons" section in `wsr deep`.
   **Run (blank model): fails as a detector.** The planted blank-reply neuron ranks
   107th of 116,736 on end-of-turn alignment (robust z 4.5), while clean Qwen has
   natural outliers at z 28.7. A shortlist for the agent's `inspect_neuron`, not
   an alarm. The loop model was not scored (the self-reading scan timed out).
8. **The reasoning agent (the paper's actual method).** When the playbook says
   could-not-fix, hand an LLM the tools surgery already has (run, capture,
   locate, project, verify) and a budget of verification runs, and let it look
   for an edit itself; `_verify` stays the judge, so nothing ships unchecked.
   Hypothesis: it fixes at least one case the playbook cannot. Adds: Milestone 4.
   **Built** (`wsr repair --agent`, `src/wsr/agent/`; see DESIGN.md section 6),
   tested with a scripted provider. It now has the paper's kind of access: a numeric
   workspace over every weight (`compute`), teacher-forced reads with interventions
   (`read`), sampled paired generation, its own matched pairs and the layer screen,
   and free-form low-rank edits (`delta`, `set_columns`, `set_rows`) under the
   paper's own exclusions. Not yet run live: needs an `ANTHROPIC_API_KEY`.
9. **An opt-in real Python kernel for the agent.** The workspace is a safe subset;
   the paper's editor had full Python in an isolated container. Run a kernel in a
   container with no network and only the weights mounted read-only (Docker or
   WSL), behind `--agent-kernel docker`, with edits still going through
   `build_edit` and the judge. Adds: any analysis the subset cannot express.
