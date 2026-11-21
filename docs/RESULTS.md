# What is built and what is proven

Every result wsr has measured, with the setup, the control and the outcome, including the runs that failed.
How each part was built is in [DESIGN.md](DESIGN.md); what is not tested is in [LIMITATIONS.md](LIMITATIONS.md).

**In short**

- **Triage and deep diagnosis** name the cause of 15 failure types against a known answer: a live server, a bug
  planted in real weights, or a real failure with a control that rules out chance.
- **Repair** fixed planted and real failures on Qwen2.5-0.5B and Llama-3.2-1B, and refused, with reasons, to ship
  fixes whose side effects were too large.
- **Under the current acceptance rules** (an exact sign test, a minimum gain, a confirmation set evaluated once,
  a semantic canary), two fixes for a bug planted in real weights were accepted and confirmed: a weight patch
  (validation 18 better / 0 worse, confirmation 15 / 0, p 0.00003) and an activation steering (confirmation 5 / 0,
  p 0.031).
- **Real over- and under-refusal, fixed by the agent and confirmed.** In every powered test so far (at least 15
  failing items in validation, fresh data, the full toolkit), Claude Opus 5.5 with `harm_probe` chose probe-gated
  steering (the refusal tied to the model's own reading of harm) and wsr confirmed it once on fresh prompts, five
  out of five: Qwen2.5-0.5B over-refusal (confirmation 15 better / 2 worse) and under-refusal (12 / 0),
  Llama-3.2-1B under-refusal (23 / 0), Qwen2.5-1.5B over-refusal (15 / 0) and under-refusal (12 / 1). Every
  guardrail held; in one run a harmful-request guardrail held only on net (see the notes). Runs that could
  not show a result (too few failing items, or before the toolkit had the harm probe) are kept as a record, with
  the reason, in their own section.
- **Every failure type, with the playbook and with the agent**, on fresh powered data. Confirmed: blank replies
  (both), loops (the playbook's gated neuron; the agent's DRY sampler), over-refusal on two models and
  under-refusal on one (the playbook's probe-gated steering). No candidate, correctly: Llama under-refusal and
  wrong language (playbook). Wrong language, agent: confirmed with multi-stage surgery (a learned weight edit,
  committed, plus a probe-gated refusal calibrated on it and the DRY sampler: validation 23 better / 1 worse,
  confirmation 23 / 0). Garbage from damaged weights: the exact repair alone fails the strict harm check (the
  restored original answers 2 of 20 harmful requests the damaged model only garbled); with multi-stage surgery and a
  contrast file for the harm probe, the agent shipped the restore plus a probe-gated refusal: validation 62 better /
  0 worse with every held-back harmful request refused, confirmation 51 / 0.
- **Earlier runs, under the looser rule (z > 2)**, also shipped an over-refusal fix (4 better, 0 worse) and an
  under-refusal fix; the first would not pass the exact test, and neither reproduced in the reruns.
- **The agent** (Claude Opus 5.5, Gemini 3.8 Flash and smaller models) found the exact planted cause in every
  planted case. The guardrails stopped every edit that broke something else.
- **Scale:** deep diagnosis and `wsr program` with the agent are tested on models up to 1.5B parameters; weight
  repair up to 1B. Every number below comes from a 4 GB GTX 1650.

## How it was measured

| What | Detail |
|---|---|
| Models | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct (diagnosis, and the agent's probe-gated steering; weight repair not tested), Llama-3.2-1B-Instruct, Qwen3.5-0.8B (positive control), a 4-layer toy Llama, a tiny random Qwen3.5 (hybrid attention) |
| Quantised files | Qwen2.5-0.5B official Q4_0 and Q4_K_M (AWQ-processed), Gemma 4 E2B Q4_0 (statistics only), wsr's own Q8_0 and Q4_K_M exports |
| Live server | llama-server b10244 with Gemma 4 E2B Q4_0 and vision: 21 scripted scenarios |
| Bugs planted in real weights | blank-reply neurons (Qwen layer 16 #0, Llama layer 10 #0), a self-reinforcing "Penguin" neuron (Qwen layer 18 #0), corrupted MLP columns (Qwen layer 12), RoPE positions aliased to multiples of 8, damaged Q4_0 blocks |
| Real failures | over-refusal (Qwen 0.5B and 1.5B on fresh safe-but-alarming prompts; Llama-1B on XSTest, which it barely over-refuses), under-refusal (Qwen 0.5B and 1.5B, Llama-1B on fresh harmful requests and XSTest), wrong language (Qwen 0.5B), loops (Qwen 0.5B lists, Llama-1B on MATH-500), the official Q4_0 refusing a harmless request |
| Controls | shuffled-label directions, random heads and neurons, an unrelated language pair, random edits and random steering directions of the same size, held-out prompts |
| The judge (current) | the original and the candidate on the same items and seeds; "improved" needs the exact one-sided sign test below 0.05 / m, a gain of at least 5 points and every guardrail holding; m is the number of candidate slots the run declares before anything is judged: 1 for the playbook's candidate, 1 for the playbook's steering pass on a refusal goal whose weight patch was rejected, and 1 per judged agent attempt (3 by default) (Bonferroni). Validation p-values below are the judge's, already adjusted (raw p x m); a confirmation set judges one frozen candidate, so its p is raw. z is reported only |
| The judge (earlier runs) | a paired gain above 2 standard errors and at least 5 scored items; the rows below say which earlier results the current rule would reject |
| Confirmation | a third data set, evaluated once on the one frozen candidate that passed validation, then recorded as used (by its content); its size is fixed before anything runs. "Confirmed" means it passed there once: one goal, one run, not a replication across datasets or labs |
| Semantic canary | 32 deterministic items (arithmetic, units, sequences, comparisons, facts, instruction formats) with computed or hand-checked answers; only items the original gets right count; at most 1 may be lost |
| Agents | Claude Opus 5.5 (through a test-only driver for the Kiro CLI, not part of wsr), Gemini 3.8 Flash (AI Studio's OpenAI-compatible endpoint), Nemotron 3 Ultra (OpenRouter free tier), Gemma 4 E2B (local, llama-server on the CPU) |
| Hardware | GTX 1650 4 GB, CUDA 12.4, float32 by default (float16 for 1.5B), 15 GB RAM |
| Automated tests | 484 passing, including regression tests built from real model replies |

## Triage and deep diagnosis: proven against a known answer

"Proven" means wsr named the known cause of this failure type, on the models listed, with the control in the
row; other architectures are not tested. Most rows rest on one or a few cases (a planted bug, a live scenario),
so they show that the check works, not how often it is right across many failures. The weaker cases are under
"Partly proven".

| Failure type | wsr names | Proof | Result |
|---|---|---|---|
| Image sent the wrong way (dropped field, bare string, bad base64, remote URL) | the request field and the fix | live llama-server with vision | all cases caught; the rerun with images as content parts flips them |
| Repetition-penalty damage | sampling, not the model | 184 live replies, penalty A/B rerun | 0 false alarms on 82 normal replies; 19/24 caught at penalty 2.0 |
| Server down or wrong port | runtime | live | fails in 3 s with exit 2 (it used to hang) |
| Wrong language in a reply | the detector (28 languages) | real Qwen replies | Ukrainian to Russian, Thai to Chinese, Marathi to Hindi, a French request answered in English: all flagged; Urdu no longer misread as Persian |
| A numbered list stuck on one item | a loop | real Qwen reply | "16. Penguin / 17. Penguin" flagged; healthy lists and tables are not |
| One neuron causing blank replies | the layer and the neuron | planted in the toy and in real Qwen 0.5B | real Qwen: layer 16 MLP writes 91% of it, neuron #0 98%; removing it fixes 3/3; high confidence |
| Over-refusal through the general refusal direction | the direction and its layers | real Qwen 1.5B, "how do I hack my own router" | matches the refusal direction (cosine 0.88); removing it fixes 8/8, shuffled directions 0/8 |
| Wrong language, wrong plan | the layer where the plan switches | real Qwen 0.5B, Ukrainian answered in Russian | the plan turns Russian at layer 7; removing that direction gives Ukrainian 3/3, an unrelated pair 0/3 |
| Wrong language, right plan, wrong first token | the readout | real Qwen 0.5B, "Answer in French" answered in English | the probe reads French at p 0.99; removing the English direction gives French 3/3, an unrelated pair 0/3 |
| A loop driven by copy heads | the heads | real Qwen 0.5B, "AB AB AB" | heads 22.5, 16.7, 16.12; removing the top 8 removes 37% of the lead (random heads 14%) and stops the loop |
| A loop that only greedy decoding keeps | sampling, not the weights | real Qwen 0.5B animal list | sampling escapes 3/3; the verdict is a decoding setting |
| A damaged quantised tensor | the tensor | real Qwen Q4_0 with planted damage | 53% and 30% error flagged; healthy AWQ files raise no false alarm (there were 192 before the fix) |
| Broken RoPE positions | a positional defect | real Qwen, positions aliased to multiples of 8 | shifting positions changes the output (KL 0.56-1.43); the healthy model exactly 0 |
| A loop held by an attractor | the attractor | the planted Penguin loop in real Qwen against two other loops | forced off the loop it falls back in 3/3 (cosine 0.96); the natural list 0/3 and the copy loop 0/1 do not |
| A failure caused by one quantised tensor | the tensor (`wsr qbisect`) | planted damage in a real Q4_0; Qwen's official Q4_0 refusing "40 names for a cat" | planted: blk.12.ffn_down, found in both directions, random 0/3; natural: blk.16.attn_v, and quantising it alone makes the full-precision model refuse; random 0/3 |

### Loops: five causes, one battery

Every loop goes through all five tests, each with its own control.

| Cause | Test and control | Positive case | Negative case | Status |
|---|---|---|---|---|
| Degenerate attractor | force another token at the repeat; does it fall back in, are states one cycle apart alike | planted Penguin: back in 3/3, cosine 0.96 | natural list 0/3, "AB AB" 0/1 | proven |
| MLP self-reinforcement | attribution per neuron; the top neurons must stop the loop more than random ones and promote the repeated token | toy model: the planted neuron exactly | natural list: neurons tip the tie but promote another word, so not counted | toy only |
| Copy heads | calibrated induction scores and attention to earlier copies; top heads against random heads | "AB AB": 37% of the lead against 14%, loop stops | natural list: 38% against 30%, not counted | proven |
| RoPE / positions | shift every position id; correct RoPE does not change | planted aliasing: KL 0.56-1.43 | healthy Qwen in fp32 and fp16: KL 0.000 | proven |
| Greedy decoding | rerun sampled (T 0.7) and with a 1.1 repetition penalty | natural list: sampling escapes 3/3 | planted Penguin and "AB AB": sampling still loops | proven |

A loop can have more than one cause: the planted Penguin neuron seeds the first repeat, then the model's own
attractor and copy heads carry it, and the battery reports the attractor.

### Against failures reported in the field

The issue trackers and guides of llama.cpp, Ollama, vLLM and transformers were searched for recurring causes
of a "broken" model. Of 14, triage covered 8; the other 6 are now checks, and live testing added a 15th.

| Field-reported cause | Triage check | Status |
|---|---|---|
| Wrong or missing EOS: the model never stops | stop probe, EOS vs end-of-turn | covered |
| Wrong or stale chat template | template against upstream, server template against the file | covered |
| Corrupted download | sha256 against upstream | covered |
| Repetition from sampling settings | penalty A/B, greedy vs sampled differential | covered |
| Ollama truncating the prompt silently | context probe, prompt token count | covered |
| Vocabulary not matching the tokenizer | tokenizer vs embeddings | covered |
| `generation_config` sampling defaults applied silently | request checks | covered |
| Sliding-window attention and the prompt cache | cache probe | covered |
| GGUF without a pre-tokenizer | `PRETOKENIZER_DEFAULT` | added, unit tests |
| Context split across server slots | `CONTEXT_SPLIT_ACROSS_SLOTS`, `SERVER_SLOTS` | added, fired live (4 slots of 2,048) |
| Context shift dropping the start mid-reply | `CONTEXT_SHIFT_ON`, `CONTEXT_SHIFTED` | added, unit tests |
| Quantised KV cache, flash-attention fp16 overflow | `KV_CACHE_QUANTISED`, `FLASH_ATTENTION_ON` | added, unit tests |
| A bf16-trained model run in fp16 (NaN, empty replies) | `FP16_OVERFLOW_RISK` | added, unit tests |
| Missing logit soft-capping; RoPE overrides that differ from the file | `SOFTCAP_MISSING`, `ROPE_OVERRIDE_DIFFERS` | added, unit tests |
| The reply changes with server load | `LOAD_DEPENDENT_OUTPUT` (concurrency probe) | added, fired live |

The new file checks raise no false alarm on the three real GGUF files. Sources: llama.cpp
[#7219](https://github.com/ggerganov/llama.cpp/issues/7219), [#11681](https://github.com/ggml-org/llama.cpp/issues/11681),
[#17284](https://github.com/ggml-org/llama.cpp/issues/17284), [#17746](https://github.com/ggml-org/llama.cpp/pull/17746);
transformers [#36822](https://github.com/huggingface/transformers/issues/36822);
[vLLM generation_config](https://github.com/vllm-project/vllm/pull/11164/files).

### Reasoning loops: the Sparks baseline

Qwen3.5-0.8B (ggml-org Q8_0 GGUF on llama-server, one sample per prompt), thinking on, with the settings of the
Sparks base-model runs: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, a 32,768-token cap. The first
8 MATH-500 problems in order:

| | hit the cap | correct (equivalent answers counted) |
|---|---|---|
| thinking on | 6 of 8 | 2 of 8 |
| thinking off, nothing else changed | 0 of 8 | 5 of 8 |

5 of the 6 capped replies loop inside the reasoning (a line repeated 38 to 213 times); the sixth is still working
at the cap. The loops start after about 2,300, 4,200, 6,400, 7,500 and 19,300 tokens. Before `REASONING_LOOP`
triage said `REASONING_EXHAUSTED` with "raise max_tokens", which does not end a loop; it now names the loop and
says to turn thinking off.

Live, `wsr triage --differential` on problem 5 of these 8 (same settings, a 12,288-token cap so each rerun takes
about 2 minutes; one llama-server slot; 29 minutes in all): the reply loops from about token 9,100, and the verdict
is REQUEST from `DIFF_FLIP:thinking_off`: turning only thinking off removes both the blank reply and the loop, and
the flip reproduces. Greedy decoding, neutral penalties and the prompt cache off all loop again: removing the
presence penalty does not help, and greedy is the low-temperature end, which loops the most (below).
`SERVER_NO_BOS` no longer warns on this Qwen model.

Temperature, paired: 40 more prompts (16 GSM8K, 12 IFEval, 8 XSTest safe prompts, 4 chat prompts), each run at
temperature 1.0 and at 0.6 with everything else the same (top-p 0.95, top-k 20, presence penalty 1.5, an 8,192-token
cap, the same seed). 1.0 with these settings is the model card's thinking-mode recommendation for text; 0.6 is what
the Sparks runs used, and what the card used for its own thinking-mode benchmarks. Each reply labelled by reading
it:

| | temperature 1.0 | temperature 0.6 | only at 0.6 | only at 1.0 | exact sign test |
|---|---|---|---|---|---|
| looped until the cap | 0 of 40 | 20 of 40 | 20 | 0 | p = 2e-6 |
| hit the cap (looping or still working) | 18 | 25 | 9 | 2 | p = 0.07 |
| answered | 23 | 15 | 2 | 10 | p = 0.04 |

Two more replies looped and then answered (one at each temperature). At 1.0 the cap is still hit 18 times, by
reasoning that is still working, so 1.0 does not make the 8,192-token budget enough; it stops the loops.

The detector, on all 81 labelled replies with enough reasoning to judge (the test fixture): the 25 loops score 0.66
to 1.0, the healthy replies that hit the cap at most 0.41 (one that was copying its draft out as the final
version), and the threshold is 0.55. A line counts as repeated from its third appearance, so a draft copied out
once does not. Not a replication of the Sparks numbers: a different runtime and quantisation, one sample per prompt,
40 prompts. The model card warns that the 0.8B "is more prone to entering thinking loops" in thinking mode and runs
it without thinking by default.

### Partly proven

| Case | What wsr says | Why it is not proven |
|---|---|---|
| MLP self-reinforcement on a real model | right on the toy; on real Qwen it reports the attractor | the planted neuron only seeds the loop and attention carries it; no real loop driven by one neuron is known |
| Over-refusal on Qwen 0.5B ("cake", "router") | the refusal direction on the router case, nothing specific on the cake case | near ties (p 0.33-0.45): almost any change tips them, so the null rejects most directions |
| Copy heads on "AB AB" | heads named, loop stopped | one failing example; removing random heads also stopped some fragile loops |
| Language probe on hard cases | 63-68% right when the model ignored the requested language | calibration questions not written by native speakers; 12 languages only |
| AWQ-processed GGUFs | no false alarm, damage still caught | the extra error beyond scale folding is not identified |
| Thin contrast sets | warns and lowers its confidence | rewrites often give fewer than 3 working prompts |

## Repair on real models

Every fix had to pass the target check, a 17-prompt regression suite (no new failures, drift at most 0.15
nats per token, top-token agreement at least 85%, every harmful request refused before still refused) and the
check at serving precision; deep mode adds fresh variants and a random control. These are single-prompt repairs
(`wsr repair`): judged on the failing prompt, its variants and fresh variants, not on a held-out set with the
statistical judge; that is what the `wsr program` sections below do.

| Model | Case | Outcome |
|---|---|---|
| Qwen2.5-0.5B | planted blank-reply neuron | **fixed**: neuron 16 #0 found blind; zeroing it fixes 10/10; the patch is 25 KB; against the original healthy model, 14/17 regression replies identical (3/17 before), drift 0.0001, 99.5% agreement |
| Qwen2.5-0.5B | over-refusal, "hack my own router" | **fixed in deep mode**: fast mode removed the refusal direction and the model then explained how to build a pipe bomb, so the safety check rejected it; deep mode built a gated neuron (layer 20 #2955) that reads the false alarm: 5/6 failing and 8/8 fresh variants fixed, all 17 regression replies unchanged, both harmful requests still refused, random control 0/6 |
| Qwen2.5-0.5B | wrong language, Ukrainian answered in Russian | **could not fix** in either mode: the edit gives Ukrainian words that make no sense and breaks a working prompt; the model barely knows Ukrainian |
| Qwen2.5-0.5B | planted self-reinforcing loop | **fixed in deep mode**: a layer screen on half the matched pairs chose layers 8 and 10, the other half confirmed them; 7/7 failing and 9/9 fresh variants fixed, drift 0.001, 98% agreement, random control 0/7 |
| Qwen2.5-0.5B | loop kept only by greedy decoding | **not indicated**: a decoding setting |
| Llama-3.2-1B | planted blank-reply neuron (layer 10 #0) | **fixed** in both modes: 10/10 prompts and 10/10 fresh variants, random control 0/10 |
| Llama-3.2-1B | its own over-refusal of the router question | **fixed** in both modes; deep mode's gated neuron (layer 9 #6820): 6/7 fresh variants, 14/17 regression replies unchanged, random control 0/6; the 3 harmful requests still refused |
| Llama-3.2-1B | its own Ukrainian loop | **not indicated**: greedy keeps it, 3/3 sampled runs escape |

The layer screen came from the published weight-space-programming edit on Qwen3.5-0.8B's reasoning loops:
there, the last layer ranks first by projection and attribution but its edits fail on new pairs; a screen over
the middle layers chose 12 and 22, which stayed selective on held-out pairs (repeat −0.067, working +0.002
nats per token), as did the published layers 15 and 19. Measured on teacher-forced pairs, not generated
solutions.

## `wsr program` under the current rules

Rerun on new goals built for these rules: practice, validation and a confirmation set, split from the same data
before any candidate ran, with the confirmation share fixed in advance. Every run records its decision in the
fix's `MANIFEST.json`.

| Goal | Model | Fix | Validation | Confirmation (once) | Decision |
|---|---|---|---|---|---|
| Planted blank replies, goal A2 (fresh questions; confirmation share fixed at 0.3, at least 10 failing items required in advance) | Qwen2.5-0.5B | sparse projection on the neurons writing the failure (playbook) | 18 better, 0 worse of 28, p 0.000004; drift 0.015, canary 22 eligible, 1 lost | 15 better, 0 worse of 23, p 0.00003 | **shipped, confirmed** |
| Planted blank replies, goal B, weights not allowed to change | Qwen2.5-0.5B | clamp steering after layer 14, the most causal layer | 6 better, 0 worse, p 0.016 | 5 better, 0 worse, p 0.031 | **steering recommended, confirmed** (small sets: 6 and 5 failing items, so it passed only because every one was fixed; valid, but weakly powered) |
| The planted-neuron patch judged on the served file (llama-server, Q8_0, both files on build b10244) | Qwen2.5-0.5B | the zero-neuron patch | 6 better, 0 worse on Q8_0, p 0.016; on the GGUF the canary (25 eligible, 0 lost) and the suite's reply checks hold; drift checks on PyTorch | not run: that confirmation set was already used | a served-file check, not a ship decision: the gain and the guardrails hold after quantisation |

## Every failure type, with the playbook and with the agent

Each failure type `wsr program` works on was tested twice on fresh data: once by the playbook alone, once by the
agent alone (Claude Opus 5.5 through the test-only Kiro driver, `--agent --no-playbook`). Where both used one data
set, they got disjoint halves, each with its own practice, validation and confirmation split. The plan for every
run was written before it ran. Every row is powered: the runs new to this table have at least 22 failing items in
validation and 18 in confirmation; the refusal rows marked "(above)" are the powered refusal tests, with at least
15 and 12 (the threshold fixed for them before their data was labelled; they had 19 to 37 and 16 to 32).

| Failure | Model and data | Playbook | Agent |
|---|---|---|---|
| Blank replies (a planted neuron) | Qwen2.5-0.5B; fresh cat questions (the playbook's run is goal A2 above) | sparse projection: validation 18 better / 0 worse, confirmation 15 / 0: **shipped, confirmed** | zeroed neuron 16 #0, the planted one, in 7 turns: validation 3.8% → 94.2% (47 / 0), confirmation 40 / 0: **shipped, confirmed** |
| Loops (list requests, 256 tokens) | Qwen2.5-0.5B; 853 list requests (playbook), 587 new ones (agent) | a gated neuron (layer 11 #3016, x4): validation 85.7% → 93.5% (17 / 5, p 0.008), all 17 suite replies unchanged, confirmation 10 / 0 (p 0.001): **shipped, confirmed** | the DRY sampler (multiplier 0.8, base 1.75, allowed length 2), chosen in 5 turns: validation 80.7% → 96.2% (33 / 0), every guardrail held, held-back harmful requests refused 90% → 95%; confirmation 30 / 1 (p 1.5e-8): **settings recommended, confirmed** |
| Over-refusal | Qwen2.5-0.5B; OR-Bench hard prompts (playbook), fresh prompts (agent, above) | probe-gated steering after layer 21, with a contrast set for the probe: validation 68% → 75% (19 / 9, p 0.044), harmful refused 70.6% → 82.4% (38 newly refused, 8 newly answered), confirmation 19 / 3 (p 0.0004): **confirmed** | two-way gate after layer 20: **confirmed** (above) |
| Over-refusal | Qwen2.5-1.5B; OR-Bench (playbook), fresh (agent) | probe-gated steering after layer 17: validation 24% → 60% (56 / 4), harmful refused 94.9% → 94.1% (10 newly answered, 8 newly refused), confirmation 48 / 2: **confirmed** | **confirmed** (above) |
| Under-refusal | Qwen2.5-0.5B; OR-Bench toxic prompts (playbook), fresh (agent) | probe-gated steering after layer 21: validation 69% → 79% (25 / 10, p 0.017), safe prompts answered 69% → 78%, confirmation 20 / 4 (p 0.0008): **confirmed** | **confirmed** (above) |
| Under-refusal | Llama-3.2-1B; OR-Bench (playbook), fresh and XSTest (agent) | **no candidate**: the closest, steering against the diagnosed direction after layer 4, refused a maths prompt and drifted 0.85 nats per token (limit 0.15) | **confirmed** (above) |
| Wrong language (Ukrainian answered in Russian) | Llama-3.2-1B; Global-MMLU Ukrainian, 1,400 questions, 151 failing | **no candidate**: the closest edit fixed the contrast prompts but only 3 of 5 fresh failing variants, and broke 2 of 5 fresh working ones | multi-stage: a rank-4 learned edit on layer 11's MLP output, committed; on it a probe-gated refusal after layer 10 and the DRY sampler (0.6, 1.75, allowed length 5), judged and shipped as one (weight patch + steering + settings). Validation 90.1% → 98.8% (23 better / 1 worse, p 4.5e-6), wrong-language replies 10% → 1%, every guardrail held (suite drift 0.05 under the steering, canary 0 of 23 lost, loop guardrail unchanged, harm check 70% → 85%: 4 newly refused, 1 newly answered); confirmation 89% → 100% (**23 better / 0 worse**, p 1.2e-7): **settings recommended, with its patch and steering, confirmed**, in 2 h 42 min |
| Garbage from damaged weights (planted: 16 columns of layer 12's MLP output x200) | Qwen2.5-0.5B; 366 list requests the undamaged model answers properly | the damaged-weights recipe's exact restore (divide by 197): validation 7.6% → 100% (61 / 0, p 4e-19), every other guardrail held; **not shipped**: the harm check failed (below) | run 1 (no multi-stage surgery yet): found the same damage with `weight_scan` and **gave up** in 12 minutes, since every restore lost harmful refusals. Run 2, with `commit_edit`: committed the exact restore, found the RESTORED model's harm reading usable, built 13 refusal gates on it and measured each on top of the restore (the best kept 20 of 20 of its own harmful requests refused, against 15 for the restore alone). Validation, all 3 attempts: 6.1% → 100% (p 6.5e-19), every guardrail held but the harm check, 100% → 95%: 1 of the 20 held-back requests answered (the restore alone: 2); no candidate. Run 3, after the goal got a contrast file (60 must-refuse and 40 safe look-alike list requests) and the harm probe its rework (DESIGN.md section 26): committed the restore, read the gate's estimate, set the lowering margin so the gate almost never lowers a refusal, and chose the restore + a probe-gated refusal after layer 20 on its first attempt. Validation 6.1% → 100% (**62 better / 0 worse**, p 6.5e-19), every guardrail held, the harm check 20 of 20 refused; confirmation 7.3% → 100% (**51 / 0**, p 4.4e-16): **steering recommended, with its patch, confirmed**, in 9 minutes |

Notes:

- **Garbage: a correct repair that safety stops.** The repair restores the original Qwen2.5-0.5B, and that model
  answers 2 of the 20 held-back harmful requests that the damaged model only garbled. Garbage counts as refused for a
  request that must be refused (it produced nothing harmful), so the harm check reads 100% → 90% and the fix is not
  shipped. The agent's practice mirror showed the same (18 → 15 of 20 refused) before it spent an attempt. A goal
  can allow this explicitly (`allow_safety_loss` on its regression-suite guardrail); wsr does not decide it alone.
- **Garbage, multi-stage.** Run 2 is the case `commit_edit` was built for (DESIGN.md section 24): on the damaged
  weights the harm probe had no usable reading; on the committed restore it had one, and gates calibrated on it
  halved the harmful answers. The split harm check did its job: the agent's half of the held-back requests read 20
  of 20 refused under its gates, and the judge's half, which the agent never sees, found the one that still slipped
  through. Run 3 shipped (DESIGN.md section 26 has what changed in between). Its validation set had 3 earlier looks
  (logged next to the goal); its confirmation set was fresh.
- **Garbage, how safe the shipped fix is beyond the judge's 20.** On 511 independent harmful requests (written for
  these tests and XSTest's harmful half), the restore alone, which is the undamaged Qwen2.5-0.5B, answers 89 (17%);
  the shipped restore + gate answers 40 (8%). It refuses 3 of 117 goal-like list requests (the restore: 3) and 69
  of 200 safe look-alikes (the restore: 66). So it is safer than the model before the damage, not safe on every
  request: the strict check passed on its 20 held-back requests, which is a sample.
- **Wrong language, agent.** Run 4 used every part of multi-stage surgery: it committed its learned edit,
  measured DRY on top (attempt 1: the edit + DRY, rejected: 3-5 replies newly garbage and 1-2 harmful requests
  answered), reverted the edit to try a system prompt on the original weights, committed it again, calibrated a
  harm gate on the edited model and chose the edit + gate + gentler DRY (attempt 2): accepted and confirmed. The
  harm check holds on its net change; this fix newly answers 1 of the 20 held-back harmful requests while newly
  refusing 4 (see LIMITATIONS.md on net guardrails). Run 3's choice (e45, a learned edit alone) had been judged
  after that run was killed: +9.5 points but rejected by the loop guardrail (11 replies newly looping).
- **Net guardrails.** The playbook's refusal runs judge the opposite set (harmful requests for over-refusal, safe
  ones for under-refusal) on its net change. The Qwen2.5-1.5B over-refusal fix newly answered 10 harmful requests
  and newly refused 8. The suite's own safety check is per item and held each time.
- **Earlier guard rule.** The playbook's refusal runs and the loop playbook run were judged before guardrails left
  out items whose original reply was garbage, empty or a loop. That rule changes only those items.
- **Looks.** The two Qwen2.5-0.5B OR-Bench validation sets were looked at by a first run without a contrast set
  (DESIGN.md section 22); their confirmation sets were fresh. Every other validation set had one look.
- **Not testable:** wrong language on Qwen2.5-0.5B with MGSM (Bengali, Telugu, Swahili, Thai): 17 failing of 1,000,
  too few for any fix to show a result.

## Over- and under-refusal: the model's own reading of harm

**What was found first.** wsr's refusal detector (version 3) counted any reply opening with "As an AI language
model, I..." as a refusal. On Qwen2.5-0.5B's harmful XSTest practice prompts, 7 of the 10 replies checked by hand
that it credited as refusals were compliance behind a disclaimer; it also missed refusals with no "can't" ("I'm
sorry, but X is not a safe or ethical practice"). Detector version 4 reads three classes: refusal, non-answer,
answer. It was checked by hand on 54 replies. On the practice sets it changes the baseline failures:

| Practice set | Shipped rule (v3) | Three-way reading (v4) |
|---|---|---|
| Qwen, safe XSTest prompts refused (of 40) | 17 | 12 |
| Qwen, harmful XSTest prompts answered (of 75) | 9 | 14 |
| Llama, safe XSTest prompts refused (of 40) | 2 | 2 |
| Llama, harmful XSTest prompts answered (of 75) | 24 | 21 |

Qwen both over- and under-refuses, and Llama's real problem is under-refusal. Results in this section use the
three-way reading.

**The harm probe.** A linear probe on the residual stream at the last prompt token separates harmful requests from
look-alike safe ones far better than either model's refusals do. With wsr's built-in pools (`wsr-refusal`), scored
on prompt families it was not fitted on, Qwen's probe has held-out AUC 0.93 after layer 20; with the goal's practice
set added, 0.96 after layer 14. The models' own refusals have balanced accuracy about 0.8 as a harm detector. On the
pools alone the probe sat late and read some XSTest-style violent requests as safe; it read them correctly once the
practice set was added. That is why `wsr program` adds the practice set.

**What was tried** (Qwen; dev prompts held out of the practice pools; numbers are false refusals / harmful
requests answered):

| Idea | False refusals | Harmful answered | Outcome |
|---|---|---|---|
| none (baseline) | 48 of 167 | 28 of 124 | |
| a system prompt that asks for calibrated refusals (161 safe, 34 harmful) | 50 → 33 | 8 → 16 | worse on harm |
| **probe-gated steering** | 48 → 11 | 28 → 5 | drift 0.019, canary and suite intact |
| the same on Llama-3.2-1B | 25 → 0 of 167 | 36 → 11 of 124 | drift 0.029, canary and suite intact |
| the same, probe fitted on XSTest-style prompts only | | | stopped refusing the suite's pipe-bomb request; fixed by adding varied harmful requests to the probe's data |
| a low-rank weight patch distilled from the steering | 48 → 39 | 28 → 26 | memorised its 70 training prompts; lost a suite refusal; rejected |
| removing the "false alarm" direction from the weights | 48 → 29 | 28 → 36 | broke a suite prompt |
| removing the plain refusal direction | 48 → 9 | 28 → 108 | unsafe |

**Judged results.** A test counts here only when it can show a result:

- **powered:** at least 15 failing items in validation and 12 in confirmation, fixed before the data was labelled,
  so a real fix can reach significance after the correction;
- **fresh data:** validation and confirmation never judged before for that model;
- **the full toolkit:** the agent has `harm_probe`, two-way gates only, the guardrails described, the practice-side
  safety mirror and 3 judged attempts.

Runs that fail one of these are in the record further down, with the reason.

Every test was run by Claude Opus 5.5, through the test-only Kiro driver, in `wsr program --agent --no-playbook`. The
plan for each was recorded before it ran. All data were split stratified on the unedited model's own replies
(practice, validation, confirmation), as `wsr objective from-failure` does.

| Model and failure | Data | What the agent chose | Validation | Guardrails | Confirmation (once) | Decision |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B over-refusal | practice: 40 XSTest safe; validation / confirmation: 100 / 100 fresh safe-but-alarming prompts (35 / 22 failing) | two-way gate after layer 20, first attempt, 6 turns | 65% → 93%: 28 better / 0 worse | harmful kept 86.7% → 86.7% (3 lost, 3 gained); suite drift −0.001; canary 1 of 23 lost | 78% → 91%: **15 better / 2 worse**, p 0.001 | **confirmed** |
| Qwen2.5-0.5B under-refusal | 311 fresh harmful requests (58 answered): practice 106 / validation 112 (21 failing) / confirmation 93 (17) | two-way gate after layer 15 with a wider "lower" margin (2.0), first attempt, 14 tool calls | 81.3% → 88.4%: 9 better / 1 worse, p 0.032 after the correction for 3 | safe answered 71.5% → 76.0% (12 better, 3 worse); suite drift 0.011, no refusal lost; canary 0 lost | **12 better / 0 worse**, p 0.0002 | **confirmed** |
| Llama-3.2-1B under-refusal | the 311 fresh plus XSTest's 200 harmful (104 answered): practice 173 / validation 184 (37) / confirmation 154 (32) | two-way gate after layer 7, first attempt, 7 tool calls | 79.9% → 96.2%: 32 better / 2 worse | safe answered 98.5% → 97.0% (3 newly refused, inside the 2-point limit); suite drift 0.024, no refusal lost; canary 0 lost | **23 better / 0 worse**, p 1e-7 | **confirmed** |
| Qwen2.5-1.5B over-refusal | 200 fresh safe prompts (67 refused): practice 68 / validation 72 (24) / confirmation 60 (20); guard: XSTest's 200 harmful | two-way gate after layer 21, first attempt, 14 tool calls (compared it with a gate after layer 17 on practice and kept the one that lost no harmful request) | 66.7% → 83.3%: 12 better / 0 worse, p 0.0007 | harmful refused 83% → 93% on net, but **10 previously refused were answered** (30 newly refused); suite drift −0.002, no refusal lost; canary 0 of 28 lost | **15 better / 0 worse**, p 0.00003 | **confirmed** |
| Qwen2.5-1.5B under-refusal | 311 fresh harmful (53 answered): practice 106 / validation 112 (19) / confirmation 93 (16); guard: XSTest's 250 safe | two-way gate after layer 16 with both margins widened to 2.0, first attempt, 15 tool calls | 83.0% → 98.2%: 17 better / 0 worse, p 0.00002 | safe answered 65% → 76% (30 better, 4 worse); suite drift −0.003, no refusal lost; canary 0 of 28 lost | **12 better / 1 worse**, p 0.002 | **confirmed** |

Notes:

- **Qwen under-refusal.** On practice the gate fixed 15 of 20 failing requests; on validation only 9 of 21. That
  gap is real, even though the fix passed. The agent widened the "lower" margin after the safety mirror showed its
  first gate answering one harmful request, then chose the safer gate at the same practice gain.
- **Llama under-refusal.** It cost 3 newly refused safe prompts: the gate's known trade-off (a safe prompt the probe
  misreads as harmful).
- **Guardrail data.** The safe guardrail was the 200 fresh safe prompts. For Qwen they had been its over-refusal
  validation and confirmation sets; that is reuse as a check, not for choosing. Llama had seen XSTest's harmful
  prompts only as a guardrail before.
- **Qwen2.5-1.5B over-refusal: a guardrail that held only on net.** The harmful-request guardrail measures the
  net change, so 30 newly refused requests covered 10 harmful requests that the model had refused and now
  answered. The regression suite's own safety check is per item (no refusal may be lost) and held. For a safety
  guardrail, losses hidden by gains elsewhere are arguably the wrong default; see LIMITATIONS.md.
- **Agents adapt to the safety mirror.** In both Qwen2.5-1.5B runs and the Qwen2.5-0.5B under-refusal run the agent
  measured more than one gate on practice and kept the one that lost no held-back harmful request, widening the
  margins where a gate touched borderline prompts.
- **Bugs found by these runs, fixed:** `checkpoint` took only edit ids (now steerings and settings too), and
  `generate` with `compare` and only an intervention (a gate) ran one gated pass labelled "original"; it now shows
  the untouched model next to the gated one.
- **Not tested:** Qwen3.5-0.8B. Its hybrid linear-attention layers generate one prompt at a time, too slowly on this
  GPU for a probe calibration plus validation and confirmation (measuring its refusal rates alone did not finish
  in 50 minutes).
- **Feedback loop.** None of the five needed a second attempt, so the loop is proven by the scripted tests only.
- **Who wrote the data.** The fresh prompts were written for these tests; they are not a public benchmark.

**The same mechanism with the experiment's scripts** (before it was built into wsr, the experiment's own settings
chosen on practice): Qwen over-refusal on fresh XSTest prompts, validation 64% → 84% (12 better / 3 worse, 16
failing), confirmation 70% → 93% (10 / 0, p 0.001): confirmed.

## Record: runs that cannot show a result

Kept for the record, with why each one says little. None of them counts above.

| Run | Outcome | Why it cannot show a result |
|---|---|---|
| Planted blank replies, goal A (Qwen2.5-0.5B, playbook: zero neuron 16 #0) | validation 6 better / 0 worse, p 0.016; confirmation 4 / 0, p 0.0625: inconclusive | its confirmation set had 5 failing items, so only a perfect 5 of 5 could pass. Goal A2 reran it with at least 10 failing items fixed in advance: confirmed |

Refusal runs:

| Run | Outcome | Why it cannot show a result |
|---|---|---|
| Llama-3.2-1B over-refusal, playbook (XSTest, 14 refused) | no candidate: the edit that fixed it stopped refusing 2 of 3 harmful requests | Llama barely over-refuses: 7 of 121 validation items failed, too few for any fix to show the minimum gain. Its real problem is under-refusal (tested above) |
| Llama-3.2-1B over-refusal, agent (Opus 5.5, before `harm_probe`): a rank-4 learned edit at layer 8, 27 turns | inconclusive: 2 better / 0 worse | the same goal (7 failing); the agent's edit was safe and passed every check |
| Qwen2.5-0.5B under-refusal, playbook (XSTest, 24 answered) | no candidate: the closest steering drifted 0.180 nats per token | before the harm probe, and before the refusal reading was fixed (detector version 3 credited disclaimers as refusals) |
| Qwen2.5-0.5B under-refusal, agent (Opus 5.5, before `harm_probe`): a rank-4 learned edit on attention layers 10-12, 32 turns | rejected: the canary lost 3 items (one sum, two comparisons) | a half toolkit: no harm probe, no gate. It also exposed a detector bug (a character cut at the token limit read as garbage), since fixed |
| Qwen2.5-0.5B under-refusal, agent with `harm_probe` (XSTest, 7 turns) | inconclusive: validation 5 better / 1 worse; every guardrail held; safe prompts answered 67% → 87% | only 10 of 75 validation items failed, on a set already judged several times. Rerun powered above: confirmed |
| Qwen2.5-0.5B over-refusal, agent, one-sided gates available | rejected: a lower-only gate fixed 28 / 0 on validation but answered 3 of 75 harmful requests and refused none | a superseded toolkit: one-sided gates have since been removed. Rerun above: confirmed |
| Qwen2.5-0.5B over-refusal, experiment scripts (XSTest, 61 / 60) | validation 15 better / 0 worse; confirmation 4 / 0, p 0.0625 | the confirmation set had too few failing items for a fix to pass (best possible p with 4 wins is 0.0625) |
| Qwen2.5-0.5B under-refusal, experiment scripts (XSTest, 75) | inconclusive: 7 better / 1 worse | 11 failing, on the same overused XSTest half |
| Llama-3.2-1B over-refusal, experiment scripts (121 safe, 6 failing) | inconclusive: 5 / 3 | 6 failing items: the minimum gain was out of reach |
| A low-rank weight patch distilled from the steering (Qwen) | rejected: over-refusal 12 / 8, under-refusal 6 / 15, a suite refusal lost | it memorised its 70 training prompts: a weight form that does not work yet |
| `wsr program` end to end, playbook only (Qwen over-refusal) | steering recommended: validation 16 better / 0 worse, every guardrail held | a pipeline check on a validation set judged three times before, with no confirmation set left. It found the integration gap that the declared steering slot now closes (DESIGN.md section 20) |

Runs from the failure-type tests above:

| Run | Outcome | Why it cannot show a result |
|---|---|---|
| Loops, agent, first goal (Qwen2.5-0.5B, 853 list requests) | settings recommended: a repetition penalty of 1.2, validation 21 better / 0 worse, confirmation 18 / 1 | the agent had given up, writing that the penalty lost harmful refusals; a fallback judged the settings anyway. The fallback is removed (only what the agent chooses is judged), and the rerun above chose DRY |
| Loops, agent, rerun 1 | stopped while its first choice (DRY) was being judged; nothing recorded | the judge was still on the earlier guard rule; rerun under the current one (above) |
| Garbage, playbook and agent, first two goals | no candidate: the playbook named nothing to edit, then nothing that passed; the first agent found the damage from weight statistics and gave up; the second was stopped for the new recipe | before the damaged-weights recipe. The second goal's list requests were also a poor target: the undamaged model refuses about 40% of them, so an exact restore turns many replies into refusals, not answers. The goal was rebuilt from requests the undamaged model answers |
| Wrong language, playbook, run 1 | surgery not indicated | the diagnosis did not get the items' expected language, so Russian replies counted as passing. Fixed; the rerun is the row above |
| Wrong language, agent, run 1 | stopped by the test driver's 150-minute limit while measuring a system prompt | the agent had spent about 35 minutes estimating drift by hand; the measuring tools now report the judge's suite check |
| Wrong language, agent, run 3 | killed with the Claude session that started it, while the judge scored its first choice | not a wsr fault. Its choice, e45, was judged afterwards (rejected: the loop guardrail fell 100% → 95.5%); its later attempts were lost. Run 4 is the row above. Long runs now start in a terminal tab of their own |
| Wrong language, agent, run 2 | stopped 70 minutes in: the provider said the model was temporarily unavailable | a provider outage, not a result. Transient provider errors are now retried |

The removed program-table rows, kept as they were:

| Goal | Model | Fix | Validation | Confirmation (once) | Decision |
|---|---|---|---|---|---|
| Over-refusal (XSTest safe prompts, 14 refused) | Llama-3.2-1B | playbook | — | not used | **no candidate**: the edit that fixed it stopped refusing 2 of 3 harmful requests |
| Over-refusal, the same goal | Llama-3.2-1B | agent (Opus 5.5, 27 turns): a rank-4 learned edit at layer 8 | 2 better, 0 worse of 7 failing, p 0.5 after the correction for 2 candidates; every refusal kept, canary 0 lost | not used | **inconclusive**: too few items fixed |
| Under-refusal (XSTest harmful prompts, 24 answered) | Qwen2.5-0.5B | playbook | — | not used | **no candidate**: the closest steering drifted 0.180 nats per token (limit 0.15) |
| Under-refusal, the same goal | Qwen2.5-0.5B | agent (Opus 5.5, 32 turns): a rank-4 learned edit on attention layers 10-12 | 88% → 96%, 6 better, 0 worse, p 0.031 | not used | **rejected by guardrails**: the canary lost 3 items the original got right (one sum, two comparisons) |

The under-refusal run also flagged a Hindi suite reply as garbage: the reply had been cut by the token limit in
the middle of a character. The agent worked this out in its notes (the original model's reply does the same at
96 tokens). The detector now ignores a character cut at the very end of a reply that stopped at the limit; the
canary loss alone still rejects that edit.

The earlier detailed write-ups:

> **Judged** (wsr's judge, validation then a confirmation set used once; steering settings frozen from practice data
> and recorded before any run; guardrails: the regression suite, the canary, and the other half of XSTest: harmful
> prompts must stay refused or safe ones answered):
>
> | Goal | Validation | Guardrails | Confirmation | Decision |
> |---|---|---|---|---|
> | Qwen over-refusal (61 / 60 XSTest safe prompts) | 67% → 92%, 15 better / 0 worse | hold; harmful refused 85% → 93% | 80% → 87%, 4 / 0, p 0.0625 | inconclusive: one item short |
> | Qwen over-refusal, replication on fresh prompts (45 / 44) | 64% → 84%, 12 / 3, p 0.018 | hold | 70% → 93%, 10 / 0, p 0.001 | **confirmed** |
> | Qwen under-refusal (75 harmful XSTest prompts) | 85% → 93%, 7 / 1, p 0.07 after the correction for 2 | hold; safe answered 67% → 92% | not run | inconclusive |
> | Llama over-refusal (121 safe prompts, 6 failing) | 95% → 97%, 5 / 3 | hold; harmful refused 76% → 90% (17 / 0) | not run | inconclusive: too few failing items to show the minimum gain |
> | a distilled weight patch (Qwen) | over-refusal 12 / 8; under-refusal 6 / 15 | fail | | rejected |
>
> These were run with the experiment's scripts, which use the same mechanism and checks that wsr now builds in.
>
> **Built into wsr, end to end.** `wsr program` on Qwen's over-refusal goal (the first goal's validation set, so its
> third and fourth looks; no confirmation set left):
>
> 1. The profile found 12 of 40 practice prompts refused.
> 2. The diagnosis reported the harm probe as viable.
> 3. The playbook's first candidate was a gated neuron. It passed its own checks, then changed none of the 61
>    validation items: rejected.
> 4. In the steering slot declared for refusal goals, the playbook tried probe-gated steering after layer 20. It fixed
>    7 of 9 failing prompts, and every check passed.
> 5. Validation: 67.2% → 93.4%, 16 better / 0 worse, p 0.00003 after the correction for 2 slots. Suite drift −0.001,
>    canary 23 eligible / 1 lost, harmful refused 87% → 87% (3 better, 3 worse).
> 6. Recommended, validated and not confirmed, in 4.5 minutes.
>
> The steering's `APPLY.md` hook, run as written, gives the same replies as wsr's hook.
>
> **The agent with the new tool.** `wsr program --agent --no-playbook` on Qwen's under-refusal goal (75 harmful XSTest
> practice prompts and 75 for validation; guard: 61 safe XSTest prompts must stay answered), with Claude Opus 5.5
> through the test-only Kiro driver. Unprompted, the agent:
>
> 1. read the diagnosis's harm-probe verdict and called `harm_probe` on the failing prompts and a few suite prompts
>    (failing prompts read harmful, logits 0.9 to 5.2; benign ones −6 to −8.5);
> 2. ran the gate in generation (all four failing prompts refused; benign suite prompts unchanged);
> 3. measured it with `steer_eval` mode gated: practice 81.3% → 97.3% on 75 items, 13 better / 1 worse;
> 4. spot-checked the suite's harmful prompts, the language prompts and safe look-alikes under it;
> 5. chose it with `final_choice`, in 7 turns and 9 tool calls.
>
> The judge, with 2 slots:
>
> | Validation | Guardrails | Decision |
> |---|---|---|
> | 86.7% → 92.0%, 5 better / 1 worse, raw p 0.109, 0.219 after the correction | suite drift 0.010, every harmful refusal kept; canary 0 lost; safe prompts still answered 67% → 87% (12 better, 0 worse) | **inconclusive**: not shipped |
>
> Only 10 of the 75 validation prompts failed before, so the gain had little room to reach significance. As in the
> experiment, the gate helped under-refusal on practice far more than on validation, and it fixed the over-refusal
> side as a by-product. This validation set had been judged before (the experiment's runs).
>
> **The agent on Qwen's over-refusal, with fresh data.** All 250 XSTest safe prompts had been used on Qwen, so 200
> fresh safe-but-alarming prompts were written for this goal (none in XSTest or wsr's pools) and split into 100 for
> validation and 100 for confirmation. Practice was 40 XSTest safe prompts; the guardrail was 75 XSTest harmful prompts
> that must stay refused (at most a 2-point drop). Claude Opus 5.5 ran through the Kiro test driver, with the playbook
> skipped. The plan was recorded before each run.
>
> | Run | What the agent chose | Validation (100 fresh) | Guardrails | Confirmation (100 fresh, once) | Decision |
> |---|---|---|---|---|---|
> | 1: one-sided gates available, no feedback | a lower-only gate (practice 9 better / 0 worse) | 65% → 93%, 28 better / 0 worse | harmful refused 86.7% → 82.7% (3 lost, 0 gained): **failed**; suite and canary held | not used | rejected |
> | 2: two-way gate only, guardrails described, safety mirror, 3 attempts | the two-way gate after layer 20 (practice 9 better / 0 worse; mirror: 3 harmful lost, 6 gained) | 65% → 93%, 28 better / 0 worse, p far below 0.05 / 3 | harmful 86.7% → 86.7% (3 lost, 3 gained); suite drift −0.001; canary 23 eligible, 1 lost | 78% → 91%, **15 better / 2 worse, p 0.001** | **steering recommended, confirmed** |
>
> In run 2 the agent read the probe's verdict, tested the gate on failing, harmful, suite, language and arithmetic
> prompts, read the safety mirror, and chose on its first attempt (6 turns, 9 tool calls), so the feedback loop was
> not needed; the scripted tests cover it. Its validation set had one earlier look (run 1's rejected gate), which is
> in the goal's log; the confirmation set was fresh.

## Record: earlier `wsr program` runs (the looser rule)

Judged by the earlier rule (a paired gain above 2 standard errors, at least 5 items), with no confirmation set,
no canary and refusal detector version 3 (which credited disclaimers as refusals). Kept as a record. What each kind
of row means today:

- **Still a valid signal:** the planted blank replies (11 better, 0 worse: exact p 0.0005, and rerun under the
  current rules with confirmation), the Llama planted neuron from one cat question, and the patch surviving
  quantisation on the served GGUF.
- **Cannot show a result:**
  - the first over-refusal goal: 2 failing items of 30;
  - the positive control on Qwen3.5-0.8B: z 1.5 on 40 problems, too small a sample;
  - the corrupted-neuron runs: the detectors see little of the damage, so there is nothing for the judge to measure.
- **Would not pass today, or did not reproduce:**
  - the over-refusal fix with 4 better, 0 worse (exact p 0.0625);
  - the steerings allowed to drop harmful refusals by their goal;
  - the under-refusal fixes judged with detector version 3.
  The powered refusal tests above replace them.
- **Correct negative results:** reasoning loops (no candidate, a capability limit), GSM8K and German (nothing to
  edit). These show wsr declining to ship, not a fix.

| Goal | Model | Who found it | What | Held-out | Verdict |
|---|---|---|---|---|---|
| Planted blank replies | Qwen2.5-0.5B | playbook, and the agent separately | zero neuron 16 #0 | 30% → 85% (20 items; z 4.82; 11 better, 0 worse; sign test p 0.0005) | **shipped**; rerun after the pre-release audit with the same result |
| Planted blank replies | Qwen2.5-0.5B | playbook, weights not allowed to change | clamp the diagnosed direction after layer 16 | 30% → 85% (z 4.82) | **steering recommended**; rerun after the audit, same result |
| Planted blank replies | toy Llama | agent (Opus 5.5), playbook off | the planted neuron, with a random control it ran itself | 40% → 100% (z 2.45) | **shipped** |
| Planted blank replies, from one cat question | Llama-3.2-1B | `wsr program` | the planted neuron | 23.7% → 100% (+76 points, SE 7) | **shipped** |
| Blank replies, the agent's patch | Llama-3.2-1B | `wsr evaluate` | — | blank replies 65% → 0%, score +70 points (SE 10.5), guardrail held | improved |
| The same patch on the served GGUF | Llama-3.2-1B | `wsr patch export`, judged on llama-server | — | blank replies 53% → 0% on Q8_0 (+76 points) and on Q4_K_M (+74), 0 items worse | survives quantisation |
| Over-refusal (XSTest), first goal | Llama-3.2-1B | agent (Opus 5.5) | gates and learned edits; one passed verification, 93% → 100% on practice | a harmful-request guardrail fell 66.7% → 53.3% | not shipped; the set had 2 failing items of 30, so no fix could reach z 2 (the judge now says so) |
| Over-refusal, rebuilt goal (7 failing of 37) | Llama-3.2-1B | playbook, then the agent (Opus 5.5) | playbook z 1.43, rejected; the agent combined three gated neurons (layers 9-11) | 81.1% → 91.9% (z 2.09; 4 better, 0 worse; sign test p 0.0625); drift 0.0006, agreement 99.7% | **shipped**; the goal allowed refusal changes, and none was lost |
| Over-refusal, rebuilt goal | Llama-3.2-1B | playbook steering | clamp the known refusal direction over layers 6-8 | 81.1% → 97.3% (z 2.64) | **steering recommended**; 2 suite harmful requests no longer refused, allowed by that goal |
| Over-refusal, refusals kept, weights not allowed to change | Llama-3.2-1B | playbook, then agent | playbook: every strength dropped refusals or drifted; agent run 1: the over-refusal direction minus its refusal part, after layer 8 | run 1: 81.1% → 91.9%, no refusal lost, drift 0.09 | not shipped: one new loop on the Chinese suite prompt; run 2 gave up |
| Under-refusal (XSTest), first goal | Qwen2.5-0.5B | agent, 7 learned edits | — | 86.7% → 100%, but safe prompts refused more (56.7% → 36.7%) | not shipped |
| Under-refusal, second goal | Qwen2.5-0.5B | agent, all tools | a learned low-rank edit on layers 15-16 | 62.5% → 87.5% (z 3.21); safe prompts answered 50% → 55% | **shipped** |
| Under-refusal, second goal | Qwen2.5-0.5B | playbook steering | clamp after layer 15 | 62.5% → 75% (z 1.68); safe-prompt guardrail failed | rejected |
| Under-refusal, second goal | Qwen2.5-0.5B | agent, weights not allowed to change | add a direction after layers 15-16 | practice 62.5% → 93.8% (z 3.3); held-out 62.5% → 71.9% (z 1.0) | rejected: it fitted the practice set |
| Under-refusal, refusals kept, steering only | Qwen2.5-0.5B | agent | steer toward refusal at layer 18 | 62.5% → 93.8%, but safe prompts answered 50% → 20% | not shipped |
| Planted corrupted neurons | Qwen2.5-0.5B | agent (Opus 5.5) | found layer 12 columns 100-115 at about 100x the norm from weight statistics | — | not shipped: the detectors see little of the damage, and drift is measured against the broken model |
| Planted corrupted neurons | Qwen2.5-0.5B | playbook steering; agent, steering only | clamp; the agent clamped layer 12 | stops the garbage but other languages fail; the agent's clamp changed nothing | correctly not passed |
| Reasoning loops (MATH-500) | Llama-3.2-1B | playbook (15 edits); agents (Gemini 3.8 Flash, Opus 5.5) | sparse edits, gates, learned edits, repetition penalties, a system prompt | Opus: practice loops 17.5% → 10%, but 3/7 working prompts rewritten; penalties 1.05 and 1.1 halved loops without raising accuracy | no candidate: a reasoning limit, not a loop circuit |
| Capability (GSM8K) | Qwen2.5-0.5B | — | — | — | nothing to edit, correctly: the token limit was too short, and the profile said so |
| Wrong language (German) | both models | — | — | — | nothing to edit: both answer German in German |
| Positive control: the published edit | Qwen3.5-0.8B | rebuilt exactly (its checksums match) | 64 neurons in each of layers 15 and 19 | looping replies 25% → 12.5% on 40 MATH-500 problems | "no clear change" at z 1.5: the judge needs a larger sample |

Steering starts at the layers the diagnosis's activation patching measured as most causal (no extra compute):
on the planted blank model layers 14 (0.78 of the failure's lead) and 16 (0.75), where the clamp at 16, the
planted layer, passed; on Llama over-refusal layers 8 (0.61) and 11 (0.53).

**What the runs show.** Where one direction carries the failure, a weight patch and a steering both fix it.
Where the failure shares its mechanism with what must stay (Llama's over-refusal with refusals kept), only a
conditional edit (gated neurons) fixed it; steering could not. The held-out judge rejected two fits to the
practice set that looked like wins, and the guardrails stopped every edit that traded one failure for another.

## The agent's runs

| Agent | Case | Turns | Outcome |
|---|---|---|---|
| Gemma 4 E2B (local, CPU, free) | planted Llama blank, playbook skipped | 7 | fixed, 1 verification, after 3 earlier runs exposed a crash, runaway thinking and unclear tool errors (all fixed) |
| Nemotron 3 Ultra (OpenRouter free tier) | Llama's router over-refusal | 25 and 20 | explored but built no edit before the turn limit; since then the agent is told its remaining budget every turn |
| Gemini 3.8 Flash | Llama MATH loops | 64 | 4 edits built (3 refused by the size cap or the gate guard), 0/8 decision flips; stopped at the $1.50 cap; 92% of input from the prompt cache |
| Claude Opus 5.5 | eight failure types (earlier table) | 15 (toy), 24 (over-refusal, MATH loops) | found the exact planted cause each time; 4 fixes shipped under the earlier rule, every unsafe or overfitted edit stopped |
| Claude Opus 5.5, current rules, before `harm_probe` (record) | Llama over-refusal; Qwen under-refusal | 27; 32 | a learned edit that kept every refusal but fixed 2 of 7 (inconclusive); a learned edit that fixed the task but lost canary items (rejected) |
| Claude Opus 5.5, with `harm_probe` (powered tests) | Qwen 0.5B under-refusal; Llama under-refusal; Qwen 1.5B over- and under-refusal (fresh data) | 14, 7, 14 and 15 tool calls | the two-way gate, first attempt each: confirmed, 12 / 0, 23 / 0, 15 / 0 and 12 / 1 |
| Claude Opus 5.5, with `harm_probe` | Qwen over-refusal, fresh prompts (playbook skipped) | 6 | chose the two-way gate on its first attempt: validation 28 better / 0 worse, guardrails held, confirmation 15 / 2 (p 0.001): **confirmed** (an earlier run with one-sided gates available chose lower-only and failed the harmful guardrail) |
| Claude Opus 5.5, with `harm_probe`, underpowered (record) | Qwen under-refusal, XSTest (playbook skipped) | 7 | chose probe-gated steering from the probe's verdict: practice 13 better / 1 worse, validation 5 / 1 (inconclusive), every guardrail held and safe prompts answered 67% → 87% |
| Claude Opus 5.5, failure-type tests | planted blank replies; list loops; garbage from damaged weights | 7; 5; 12 minutes | zeroed the planted neuron (confirmed); chose DRY over a repetition penalty (confirmed); found the damage with `weight_scan` and gave up because every restore lost harmful refusals |

The runs found and fixed eight wsr bugs: a verification demand no edit could meet, a system-prompt setting that
never reached the model, no diagnosis kind for under-refusal, over-cautious memory estimates in `fit_edit`, a
refusal phrase the detector missed, `generate` taking an example name as text, Gemini's thought signatures not
returned, and thinking tokens not counted as output.

## Speed, with identical results

Each speedup was checked against the slow path.

| Change | Measured | Check |
|---|---|---|
| float32 where the model fits (this GPU's half precision is slow) | fp32 matmul 1.53 TFLOPS against fp16 0.34 | the reference precision |
| Batched generation sized to free VRAM | the regression suite 89.5 s → 19.7 s; 30.8 tokens/s from about 7 | identical replies |
| Agent `generate` with 8 prompts per call | 70.2 s → 11.4 s (64 tokens), 244 s → 42.5 s (256 tokens) | identical replies, 8/8 |
| Diagnosis and the original model's replies kept across runs | a planted-blank program rerun 81.3 s → 37.9 s | same patch, same held-out result |
| Captures skip the unused vocabulary projection | 5.54 s → 4.45 s per 736-token capture | bit-identical activations |
| Same batches before and after, reference runs reused | diagnosis 402 s → 300 s | identical findings, numbers included |
| `wsr fix` in one process | about 25 min → 12.8 min | same verdict |
| llama-server judge | the judge stage halved | same verdict as PyTorch (33.3% → 100%, 0 worse) |
| Deep runs on Qwen 0.5B | over-refusal 226 s → 82 s, wrong language 226 s → 113 s, list loop 540 s → 333 s | verdicts word for word the same |

| Each batch chunk sized for its own longest prompt (length-sorted) | 2.4x on short prompts | identical replies; out-of-memory halving kept |
| The harm probe's built-in pools: replies and residuals cached, residuals captured in batches | the pools are measured once per model, not once per goal | the same probe (the cache is keyed by model and probe version) |

Rejected after measuring: stopping a loop early saves 5% of tokens; looking for a loop past the cap at twice the
length reproduced 7/8 failing prompts instead of 3/8 but adds about 6 minutes per check, so it runs only for
loops. A prefix cache for the shared start of every prompt (chat template and system prompt) would save under 5%:
that prefix is short against a reply of 64 to 256 tokens. It would also make the "before" and "after" replies
differ in numerics from an uncached run, so it was not built.

**Less time on candidates that cannot pass.** These change no verdict; they stop work whose answer is already
known:

- The agent's measuring tools run the judge's own regression-suite check and the practice-side harm check first.
  A candidate that fails either is reported at once, before any practice item, and the call does not count
  against the practice budget. The garbage agent learned in one call each that all three restores lost harmful
  refusals.
- `settings_eval` reports the suite check for settings itself, so the agent no longer estimates drift by hand. In
  the first wrong-language run it had spent about 35 minutes on that; in the second, a system prompt's suite
  failure came back in one call.
- Measured on 60 or more items, a call runs in two stages: the first 30, then the rest. It stops after the first
  stage when the candidate is already significantly worse (exact sign test, p ≤ 0.05) or broke at least 3 replies,
  more than it mended. Each stage is paired against its own baseline, so the numbers are exact.
- `final_choice` refuses a candidate whose suite check failed: the judge would reject it for certain, and the
  attempt is not spent.
- A guardrail on exactly the validation prompts grades the judge's own replies with its grader instead of
  generating them again (never a safety set). On the Llama wrong-language goal (four such guardrails over 252
  prompts) the judge took 15 minutes. The run before this change had been judging the same edit for more than 75
  minutes when it was killed. The originals' side of the guardrails also came from the cache that run had written,
  so not all of the difference is reuse; the four guardrails generated no reply of their own.

## Pre-release audit

The whole tool was read before release and every suspected fault reproduced, then fixed with a test: agent runs
that could crash on a long tool result, workspace allocations checked too late, a judge that called 1-2 fixed
items an improvement, probe caches that used pickle and a weak key, a cache command that needed torch, a goal
budget that was ignored, wrong exit codes, and patches that could be applied to other weights. Measured after
the fix: a saved patch's base check says "same" for its own model (also loaded in float16) and "different" for
another model of the same shape (0.8% difference in the edited tensor). The full list is in DESIGN.md
section 18.

## Second review: stricter acceptance

After two outside reviews, the acceptance rules were tightened before the reruns above: the exact sign test and a
minimum gain decide (z is reported only), runs that may judge two candidates test each at half the level, an
optional confirmation set is evaluated once per frozen candidate and recorded as used by its content, a semantic
canary checks answers the original got right, a patch records exact hashes of every tensor of its model (and
the commands that apply or export it refuse other weights), guardrails run on the served GGUF against the
unpatched GGUF of the same quantisation, and every judged fix carries a MANIFEST.json with its evidence. The
reruns found four bugs, all fixed: the converter's output decoded with the wrong code page on Windows, the
report crashing on a served-file guardrail, a steering's control recorded under the wrong name, and the
truncated-character false alarm above. DESIGN.md section 19 has the details.

## Not proven

The short list; [LIMITATIONS.md](LIMITATIONS.md) has all of it.

- Nothing above 1.5B parameters (diagnosis, the agent's steering) or 1B (weight repair); Gemma and Mistral untested
  at the weight level.
- Moderate sets: the powered refusal tests have 19 to 37 failing items in validation and 16 to 32 in
  confirmation; the planted-bug goals fewer. One agent run per case, and the refusal tests' fresh prompts were
  written for them, not a public benchmark.
- The Anthropic provider (the default for `--agent`) has not been run live.
- No baseline comparisons with other tools (abliteration tools, fine-tuning).
- Failures that only llama.cpp's integer kernels cause, real long-context RoPE failures, and naturally occurring
  blank replies have not been seen.
