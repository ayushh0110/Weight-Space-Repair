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
  p 0.031). The agent's fixes for Llama's over-refusal and Qwen's under-refusal were not accepted: the first
  fixed too few items, the second lost simple arithmetic the original model could do.
- **Probe-gated steering** fixed real over-refusal on Qwen2.5-0.5B. The refusal is tied to the model's own
  reading of harm. It was confirmed once on fresh XSTest prompts (confirmation 10 better / 0 worse, p 0.001) with
  every guardrail holding, harmful requests kept refused. Built into wsr and run end to end, `wsr program` found and
  recommended it on its own (validation 16 better / 0 worse); that validation set had been seen before, so that run
  shows the pipeline, not a new result. Qwen's under-refusal and Llama's over-refusal improved but stayed
  inconclusive.
- **Earlier runs, under the looser rule (z > 2)**, also shipped an over-refusal fix (4 better, 0 worse) and an
  under-refusal fix; the first would not pass the exact test, and neither reproduced in the reruns.
- **The agent** (Claude Opus 5.5, Gemini 3.8 Flash and smaller models) found the exact planted cause in every
  planted case. The guardrails stopped every edit that broke something else.
- **Scale:** deep diagnosis is tested on models up to 1.5B parameters; repair and `wsr program` up to 1B. Every
  number below comes from a 4 GB GTX 1650.

## How it was measured

| What | Detail |
|---|---|
| Models | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct (diagnosis only), Llama-3.2-1B-Instruct, Qwen3.5-0.8B (positive control), a 4-layer toy Llama, a tiny random Qwen3.5 (hybrid attention) |
| Quantised files | Qwen2.5-0.5B official Q4_0 and Q4_K_M (AWQ-processed), Gemma 4 E2B Q4_0 (statistics only), wsr's own Q8_0 and Q4_K_M exports |
| Live server | llama-server b10244 with Gemma 4 E2B Q4_0 and vision: 21 scripted scenarios |
| Bugs planted in real weights | blank-reply neurons (Qwen layer 16 #0, Llama layer 10 #0), a self-reinforcing "Penguin" neuron (Qwen layer 18 #0), corrupted MLP columns (Qwen layer 12), RoPE positions aliased to multiples of 8, damaged Q4_0 blocks |
| Real failures | over-refusal (Qwen 1.5B, Llama-1B on XSTest), under-refusal (Qwen 0.5B on XSTest), wrong language (Qwen 0.5B), loops (Qwen 0.5B lists, Llama-1B on MATH-500), the official Q4_0 refusing a harmless request |
| Controls | shuffled-label directions, random heads and neurons, an unrelated language pair, random edits and random steering directions of the same size, held-out prompts |
| The judge (current) | the original and the candidate on the same items and seeds; "improved" needs the exact one-sided sign test below 0.05 (divided by the number of candidates a run may judge), a gain of at least 5 points and every guardrail holding; z is reported only |
| The judge (earlier runs) | a paired gain above 2 standard errors and at least 5 scored items; the rows below say which earlier results the current rule would reject |
| Confirmation | a third data set, evaluated once on the one frozen candidate that passed validation, then recorded as used (by its content); its size is fixed before anything runs |
| Semantic canary | 32 deterministic items (arithmetic, units, sequences, comparisons, facts, instruction formats) with computed or hand-checked answers; only items the original gets right count; at most 1 may be lost |
| Agents | Claude Opus 5.5 (through a test-only driver for the Kiro CLI, not part of wsr), Gemini 3.8 Flash (AI Studio's OpenAI-compatible endpoint), Nemotron 3 Ultra (OpenRouter free tier), Gemma 4 E2B (local, llama-server on the CPU) |
| Hardware | GTX 1650 4 GB, CUDA 12.4, float32 by default (float16 for 1.5B), 15 GB RAM |
| Automated tests | 384 passing, including regression tests built from real model replies |

## Triage and deep diagnosis: proven against a known answer

"Proven" means this failure type, on the models listed; other architectures are not tested.

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
check at serving precision; deep mode adds fresh variants and a random control.

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
| Planted blank replies, goal A (5 failing items held for confirmation) | Qwen2.5-0.5B | zero neuron 16 #0 (playbook) | 6 better, 0 worse of 12, p 0.016; suite and canary hold | 4 better, 0 worse of 11, p 0.0625 | **inconclusive, not shipped**: one item short of significance on a set this small |
| Planted blank replies, goal A2 (fresh questions; confirmation share fixed at 0.3, at least 10 failing items required in advance) | Qwen2.5-0.5B | sparse projection on the neurons writing the failure (playbook) | 18 better, 0 worse of 28, p 0.000004; drift 0.015, canary 22 eligible, 1 lost | 15 better, 0 worse of 23, p 0.00003 | **shipped, confirmed** |
| Planted blank replies, goal B, weights not allowed to change | Qwen2.5-0.5B | clamp steering after layer 14, the most causal layer | 6 better, 0 worse, p 0.016 | 5 better, 0 worse, p 0.031 | **steering recommended, confirmed** |
| Goal A judged on the served file (llama-server, Q8_0, both files on build b10244) | Qwen2.5-0.5B | the same patch as goal A | 6 better, 0 worse on Q8_0, p 0.016; on the GGUF the canary (25 eligible, 0 lost) and the suite's reply checks hold; drift checks on PyTorch | refused: the set was already used | not shipped (the confirmation set is used once) |
| Over-refusal (XSTest safe prompts, 14 refused) | Llama-3.2-1B | playbook | — | not used | **no candidate**: the edit that fixed it stopped refusing 2 of 3 harmful requests |
| Over-refusal, the same goal | Llama-3.2-1B | agent (Opus 5.5, 27 turns): a rank-4 learned edit at layer 8 | 2 better, 0 worse of 7 failing, p 0.5 after the correction for 2 candidates; every refusal kept, canary 0 lost | not used | **inconclusive**: too few items fixed |
| Under-refusal (XSTest harmful prompts, 24 answered) | Qwen2.5-0.5B | playbook | — | not used | **no candidate**: the closest steering drifted 0.180 nats per token (limit 0.15) |
| Under-refusal, the same goal | Qwen2.5-0.5B | agent (Opus 5.5, 32 turns): a rank-4 learned edit on attention layers 10-12 | 88% → 96%, 6 better, 0 worse, p 0.031 | not used | **rejected by guardrails**: the canary lost 3 items the original got right (one sum, two comparisons) |

The under-refusal run also flagged a Hindi suite reply as garbage: the reply had been cut by the token limit in
the middle of a character. The agent worked this out in its notes (the original model's reply does the same at
96 tokens). The detector now ignores a character cut at the very end of a reply that stopped at the limit; the
canary loss alone still rejects that edit.

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

**Judged** (wsr's judge, validation then a confirmation set used once; steering settings frozen from practice data
and recorded before any run; guardrails: the regression suite, the canary, and the other half of XSTest: harmful
prompts must stay refused or safe ones answered):

| Goal | Validation | Guardrails | Confirmation | Decision |
|---|---|---|---|---|
| Qwen over-refusal (61 / 60 XSTest safe prompts) | 67% → 92%, 15 better / 0 worse | hold; harmful refused 85% → 93% | 80% → 87%, 4 / 0, p 0.0625 | inconclusive: one item short |
| Qwen over-refusal, replication on fresh prompts (45 / 44) | 64% → 84%, 12 / 3, p 0.018 | hold | 70% → 93%, 10 / 0, p 0.001 | **confirmed** |
| Qwen under-refusal (75 harmful XSTest prompts) | 85% → 93%, 7 / 1, p 0.07 after the correction for 2 | hold; safe answered 67% → 92% | not run | inconclusive |
| Llama over-refusal (121 safe prompts, 6 failing) | 95% → 97%, 5 / 3 | hold; harmful refused 76% → 90% (17 / 0) | not run | inconclusive: too few failing items to show the minimum gain |
| a distilled weight patch (Qwen) | over-refusal 12 / 8; under-refusal 6 / 15 | fail | | rejected |

These were run with the experiment's scripts, which use the same mechanism and checks that wsr now builds in.

**Built into wsr, end to end.** `wsr program` on Qwen's over-refusal goal (the first goal's validation set, so its
third and fourth looks; no confirmation set left):

1. The profile found 12 of 40 practice prompts refused.
2. The diagnosis reported the harm probe as viable.
3. The playbook's first candidate was a gated neuron. It passed its own checks, then changed none of the 61
   validation items: rejected.
4. In the steering slot declared for refusal goals, the playbook tried probe-gated steering after layer 20. It fixed
   7 of 9 failing prompts, and every check passed.
5. Validation: 67.2% → 93.4%, 16 better / 0 worse, p 0.00003 after the correction for 2 slots. Suite drift −0.001,
   canary 23 eligible / 1 lost, harmful refused 87% → 87% (3 better, 3 worse).
6. Recommended, validated and not confirmed, in 4.5 minutes.

The steering's `APPLY.md` hook, run as written, gives the same replies as wsr's hook.

**The agent with the new tool.** `wsr program --agent --no-playbook` on Qwen's under-refusal goal (75 harmful XSTest
practice prompts and 75 for validation; guard: 61 safe XSTest prompts must stay answered), with Claude Opus 5.5
through the test-only Kiro driver. Unprompted, the agent:

1. read the diagnosis's harm-probe verdict and called `harm_probe` on the failing prompts and a few suite prompts
   (failing prompts read harmful, logits 0.9 to 5.2; benign ones −6 to −8.5);
2. ran the gate in generation (all four failing prompts refused; benign suite prompts unchanged);
3. measured it with `steer_eval` mode gated: practice 81.3% → 97.3% on 75 items, 13 better / 1 worse;
4. spot-checked the suite's harmful prompts, the language prompts and safe look-alikes under it;
5. chose it with `final_choice`, in 7 turns and 9 tool calls.

The judge, with 2 slots:

| Validation | Guardrails | Decision |
|---|---|---|
| 86.7% → 92.0%, 5 better / 1 worse, raw p 0.109, 0.219 after the correction | suite drift 0.010, every harmful refusal kept; canary 0 lost; safe prompts still answered 67% → 87% (12 better, 0 worse) | **inconclusive**: not shipped |

Only 10 of the 75 validation prompts failed before, so the gain had little room to reach significance. As in the
experiment, the gate helped under-refusal on practice far more than on validation, and it fixed the over-refusal
side as a by-product. This validation set had been judged before (the experiment's runs).

## Earlier `wsr program` runs (before the stricter rules)

Judged by the earlier rule (a paired gain above 2 standard errors, at least 5 items), with no confirmation set
and no canary. The current rule would reject the over-refusal fix below (4 better, 0 worse: exact p 0.0625), and
neither it nor the under-refusal fix reproduced in the reruns above.

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
| Claude Opus 5.5, current rules | Llama over-refusal; Qwen under-refusal | 27; 32 | a learned edit that kept every refusal but fixed 2 of 7 (inconclusive); a learned edit that fixed the task but lost canary items (rejected) |
| Claude Opus 5.5, with `harm_probe` | Qwen under-refusal (playbook skipped) | 7 | chose probe-gated steering from the probe's verdict: practice 13 better / 1 worse, validation 5 / 1 (inconclusive), every guardrail held and safe prompts answered 67% → 87% |

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

Rejected after measuring: stopping a loop early saves 5% of tokens; looking for a loop past the cap at twice the
length reproduced 7/8 failing prompts instead of 3/8 but adds about 6 minutes per check, so it runs only for
loops.

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

- Nothing above 1.5B parameters (diagnosis) or 1B (repair); Gemma and Mistral untested at the weight level.
- Small sets: validation sets of 12 to 121 items with 6 to 18 failing, confirmation sets with 5 to 15 failing;
  one agent run per case.
- The Anthropic provider (the default for `--agent`) has not been run live.
- No baseline comparisons with other tools (abliteration tools, fine-tuning).
- Failures that only llama.cpp's integer kernels cause, real long-context RoPE failures, and naturally occurring
  blank replies have not been seen.
