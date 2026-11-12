# What is built and what is proven

Every result wsr has measured, with the setup, the control and the outcome, including the runs that failed.
How each part was built is in [DESIGN.md](DESIGN.md); what is not tested is in [LIMITATIONS.md](LIMITATIONS.md).

**In short**

- **Triage and deep diagnosis** name the cause of 15 failure types against a known answer: a live server, a bug
  planted in real weights, or a real failure with a control that rules out chance.
- **Repair** fixed planted and real failures on Qwen2.5-0.5B and Llama-3.2-1B, and refused, with reasons, to ship
  fixes whose side effects were too large.
- **`wsr program`** shipped weight patches and steerings that a paired held-out judge accepted: blank replies
  30% → 85% (z 4.82), over-refusal 81.1% → 91.9% with no harmful-request refusal lost, under-refusal 62.5% →
  87.5% (z 3.21). It shipped nothing where the judge or a guardrail said no.
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
| The judge | the original and the candidate on the same held-out items and seeds; "improved" needs a paired gain above 2 standard errors, at least 5 scored items and every guardrail holding; the one-sided sign test is reported beside it |
| Agents | Claude Opus 5.5 (through a test-only driver for the Kiro CLI, not part of wsr), Gemini 3.8 Flash (AI Studio's OpenAI-compatible endpoint), Nemotron 3 Ultra (OpenRouter free tier), Gemma 4 E2B (local, llama-server on the CPU) |
| Hardware | GTX 1650 4 GB, CUDA 12.4, float32 by default (float16 for 1.5B), 15 GB RAM |
| Automated tests | 340 passing, including regression tests built from real model replies |

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

## `wsr program`: judged on held-out data

Each row is a goal with a practice set the tools may use and a held-out set they never see.

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
| Claude Opus 5.5 | eight failure types (table above) | 15 (toy), 24 (over-refusal, MATH loops) | found the exact planted cause each time; 4 fixes shipped, every unsafe or overfitted edit stopped |

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

## Not proven

The short list; [LIMITATIONS.md](LIMITATIONS.md) has all of it.

- Nothing above 1.5B parameters (diagnosis) or 1B (repair); Gemma and Mistral untested at the weight level.
- Held-out sets of 20 to 40 items, and one agent run per case except where noted.
- The Anthropic provider (the default for `--agent`) has not been run live.
- No baseline comparisons with other tools (abliteration tools, fine-tuning).
- Failures that only llama.cpp's integer kernels cause, real long-context RoPE failures, and naturally occurring
  blank replies have not been seen.
