# wsr guide

Everything wsr can do, how each part decides, and what it writes. The [README](../README.md) is the short
version; [RESULTS.md](RESULTS.md) has every measured result; [DESIGN.md](DESIGN.md) records how each part
was built; [LIMITATIONS.md](LIMITATIONS.md) lists what is not tested or not built. The command reference at the end is
generated from the CLI itself.

## Contents

1. [Which command for which question](#which-command-for-which-question)
2. [Triage: what failed, and is it the model?](#triage-what-failed-and-is-it-the-model)
3. [Deep diagnosis: where in the network](#deep-diagnosis-where-in-the-network)
4. [Quantisation](#quantisation)
5. [Repair: the smallest fix that passes the checks](#repair-the-smallest-fix-that-passes-the-checks)
6. [The surgery agent](#the-surgery-agent)
7. [Weight-space programming: goals, profiles and the held-out judge](#weight-space-programming-goals-profiles-and-the-held-out-judge)
8. [What a fix is made for: identity, manifest, evidence](#what-a-fix-is-made-for-identity-manifest-evidence)
9. [Backends: PyTorch and llama-server](#backends-pytorch-and-llama-server)
10. [Safety: what wsr will not do without you](#safety-what-wsr-will-not-do-without-you)
11. [Caching](#caching)
12. [Files wsr writes](#files-wsr-writes)
13. [Environment variables](#environment-variables)
14. [Exit codes](#exit-codes)
15. [Command reference](#command-reference)

## Which command for which question

| Question | Command | Needs |
|---|---|---|
| Why is this reply broken? Is it my setup or the model? | `wsr triage` | nothing (a server, a saved response, or the model folder) |
| Is this file intact? Is the template right? | `wsr file`, `wsr config` | the model folder or file |
| What kind of failure is this saved response or text? | `wsr response`, `wsr output` | the saved JSON or text |
| How does my running server behave? | `wsr probe` | a running server |
| Where in the weights does the failure come from? | `wsr deep` | the full-precision model, `deep` extra |
| Did quantisation break it, and which tensor? | `wsr quant`, `wsr qbisect` | the GGUF (and the original for `qbisect`) |
| Fix one failing prompt | `wsr repair` | `deep` extra |
| Fix it from one prompt, end to end, judged on held-out data | `wsr fix` | `deep` extra |
| Improve a measurable goal over a dataset | `wsr objective`, `wsr profile`, `wsr evaluate`, `wsr program` | `deep` extra, a goal file |
| Ship a fix to the file people serve | `wsr patch apply`, `wsr patch export` | a patch from `repair` or `program` |
| What is this fix, how was it judged, does it match my model? | `wsr patch show`, `wsr patch check` | a fix directory with its `MANIFEST.json` |
| Public datasets, remembered answers, what wsr keeps | `wsr data`, `wsr consent`, `wsr cache` | nothing |

Triage needs no GPU and no dependencies. Everything that touches weights needs the `deep` extra; deep
diagnosis is tested on models up to 1.5B parameters, repair and `wsr program` up to 1B, on a 4 GB GPU.

## Triage: what failed, and is it the model?

`wsr triage` runs the layers in order and stops blaming the model as long as anything outside the weights
explains the failure:

| Layer | What it checks |
|---|---|
| 1 File | safetensors and GGUF structure, truncated or partial downloads, shard index, NaN / Inf values (`--scan-values`), size and sha256 against the Hub (`--hash`, `--upstream`) |
| 2 Config | chat template missing, failing or stale against upstream; EOS vs end-of-turn token; double BOS; tokenizer vs embedding size; low-bit quantisation; memory fit; GGUF pre-tokenizer, soft-capping, fp16 overflow risk; server launch flags (`--server-args`, Ollama's environment): KV-cache quantisation, flash attention, context split across slots, context shift, RoPE overrides |
| 3 Request | penalties, temperature, min_p, stop strings, max_tokens, Ollama's context, constrained decoding; images: a dropped `images` field, malformed image parts, bad or cut base64, remote URLs, a server without vision |
| 3 Response | finish reasons (OpenAI, Anthropic, Gemini, Ollama), reasoning that used up the budget, content filters, refusal fields, input truncation, tool-call schemas; SSE and NDJSON streams cut short or malformed |
| 4 Output | empty, short, repetition and tail loops (word and CJK), low entropy, truncation, symbol soup, token salad, broken UTF-8, repetition avoidance under a penalty, control-token leaks, invented turns, reasoning leaks, refusal text, JSON and schema, script and language, prompt echo, agent loops |
| 5 Probes | live checks of the server (below), and with `--differential` one-factor reruns: the failing request again with one thing changed at a time, each flip confirmed by a second run |
| 6 Known issues | documented model and runtime bugs, matched on model, backend, conditions and symptoms (`src/wsr/known_issues.json`; add your own with `--known-issues`) |

Probes (`--probes` or `wsr probe --only`): `reasoning`, `bos`, `smoke`, `stop`, `finish`, `tools`, `stream`,
`vision`, `determinism`, `cache`, `context`, `concurrency`. The last four are slow and off unless named.

**The verdict** names one cause with its evidence and the exact fix: `file`, `config`, `request`, `sampling`,
`runtime`, `quantisation`, `provider` (a filter, a refusal field, a rate limit), `known_bug`, `weight_level`,
or `unexplained`; `no_failure_observed` when nothing failed. Only a failure that survives every other layer is
called `weight_level`. `--handoff DIR` writes the failing request, the response and the report for the
weight-level stages; `--json` prints the report as JSON.

Output presets for the detectors: `chat` (default), `strictJson`, `longForm`, `lenient`. Expected language:
`--expect-lang` (ISO code) or `--expect-script`.

## Deep diagnosis: where in the network

`wsr deep MODEL_DIR --prompt "..."` works on the full-precision Hugging Face model. Failure kinds: `blank`
(premature end of turn), `loop`, `refusal` (over-refusal), `complied` (a harmful request answered: under-refusal,
used when a refusal is expected), `language`, and `other`; `--kind auto` picks one.

What it does, in order:

1. **Reproduce** the failure greedily at full precision. If it does not reproduce, the cause is outside the
   weights (often quantisation: see `wsr qbisect`). If the reply is cut before the failure shows, it says so
   and names the longer `--max-new` to use.
2. **Contrast set**: near-identical prompts that fail and that work (yours from `--failing` / `--passing`,
   plus generated variants).
3. **Token trace** (logit lens): the depth at which the failing token takes over.
4. **Direction**: the failing-minus-working direction at every depth, chosen by how much removing it moves the
   decision, with a permutation null.
5. **Localisation**: activation patching of every layer's attention and MLP, heads and neurons; how much of the
   failure each one carries.
6. **Known directions**: refusal, end-of-turn, language and repetition directions from wsr's fixed prompts,
   compared with the failure.
7. **Calibrated probes**: a 12-language probe for wrong-language failures; copy heads and an MLP and attractor
   battery for loops; RoPE and greedy-only checks.
8. **Causal tests**: ablation and addition of what was found, on generation, against random controls.
9. **Weight dossier** of the implicated matrices; with `--base`, how much fine-tuning changed them; with
   `--gguf`, their quantisation error.

The result is a verdict (`weight_level`, `not_weight_level`, `quantisation`, or inconclusive) with a confidence.
`--out DIR` writes the hand-off package. A diagnosis is cached: the same weights, prompts and settings give the
stored result (see [Caching](#caching)).

## Quantisation

- `wsr quant FILE.gguf [--reference HF_DIR]`: per-tensor statistics, NaN / Inf, zero-scale and repeated blocks,
  outliers against the same tensor in other layers, and with the original weights the relative error of every
  tensor against what its type normally loses (AWQ-style folded scales are fitted out first).
- `wsr qbisect HF_DIR FILE.gguf --prompt "..."`: rebuilds the served model in PyTorch from the GGUF's
  dequantised weights and swaps tensors back to full precision until the failure goes: the smallest set of
  tensors to requantise at more bits.

## Repair: the smallest fix that passes the checks

`wsr repair MODEL_DIR --prompt "..." --out DIR` diagnoses (as `wsr deep`), then searches for a fix.

**Candidates, smallest first.** The playbook turns the diagnosis into candidate edits per failure kind:

| Entry | Candidates |
|---|---|
| over-refusal | the known refusal direction, the contrast direction, a gated neuron that reads the false alarm (over-refused minus rightly refused) |
| wrong language | the calibrated answered-minus-expected language direction, the contrast direction |
| premature end-of-turn | the localised neurons, the contrast direction, the end-of-turn direction |
| loops | the loop battery's MLP neurons and copy heads, the attractor direction, a layer screen on matched continuations |
| anything else | the contrast direction from your `--failing` / `--passing` prompts |

Edit types: direction removal (orthogonalisation of every write, or only chosen layers or neurons), neuron and
head scaling, a gated neuron (a quiet neuron rewritten to fire only on the failure and write against it), and,
for the agent, free-form and learned low-rank changes. Each candidate is tried as a ladder of strengths,
weakest first; a weight edit is first checked against the intervention it came from.

**Checks a fix must pass.** `fast` mode (default):

- **Target**: the original prompt and at least 2/3 of the failing prompts fixed, at least 2/3 of the working ones
  unchanged. Judged against the unedited model's own reference run of the same list, so batching noise cannot
  count as a fix. For loops that start past the cap, the reference runs again at twice the length.
- **Regression suite**: 17 prompts (general tasks, six languages, three harmful requests) plus your `--keep`
  prompts: no new failures, drift at most 0.15 nats per token on the original replies (`--max-drift`), top-token
  agreement at least 85% (`--min-agreement`), and every harmful request refused before still refused.
- **Semantic canary**: 32 deterministic items in six groups (arithmetic, units, sequences, comparisons, facts,
  instruction formats), answers computed or hand-checked, graded per item, greedy at 48 tokens. Only items the
  original model gets right count, and a fix may lose at most one of them. The suite catches broken text; the
  canary catches a fix that still writes fluent text but gets simple answers wrong.
- **Serving precision**: the patch again after rounding to bf16 or fp16 (`--serve-dtype`).

`--deep` adds local edits (only the layers or MLP neurons that write the failure, strengths up to 6), a
held-out layer screen for loops, gated neurons, fresh prompt variants that took no part in the choice, and a
random edit of the same size as a control. `--sampled` adds paired sampled runs at the model's own sampling
settings. `--check PATCH_DIR` runs the checks on an existing patch without searching.

**Steering.** When no weight edit passes, the playbook tries activation steering (`--steering fallback`,
the default; `only`, `off`): starting at the layers the diagnosis measured as most causal, it either clamps the
failure direction (lowers its component to the working prompts' level only where it is above it: `clamp`) or
adds a constant shift (`add`), through the same checks. A steering changes no weights; it ships as a vector, its
settings and an `APPLY.md` with the transformers hook.

**Outcome.** `fixed` (a patch), `steered`, `verified` / `not_verified` (with `--check`), `could_not_fix` (with
the closest attempt and the practical next step), or `not_indicated` (the weights are not where to fix it).

**Shipping a patch.** `wsr patch apply MODEL_DIR PATCH_DIR --out DIR` writes the patched model in the
checkpoint's precision; `revert` undoes it (and checks that the edited tensors are back to their exact original
hashes); `export --gguf FILE --quant Q4_K_M` writes a GGUF through llama.cpp's converter (asks before fetching
it) and records the GGUF's SHA-256. A patch refuses any weights but the exact ones it was made on (see
[identity](#what-a-fix-is-made-for-identity-manifest-evidence)).

## The surgery agent

When the playbook cannot fix a failure, `--agent` hands everything it knew (the diagnosis, every trial and the
check it failed) to a frontier model that works through tools. The checks stay the judge.

**Providers.** `--agent-provider anthropic` (default; Claude, model `claude-opus-5-5` unless
`--agent-model`; key in `ANTHROPIC_API_KEY`) or `openai` (any OpenAI-compatible server: OpenAI, OpenRouter,
Together, Groq, DeepSeek, Mistral, Gemini's OpenAI endpoint, or a local llama-server / vLLM / Ollama with
`--agent-base-url`). Keys come from environment variables only (`--agent-key-env`), never the command line.

**Tools.**

| Group | Tools |
|---|---|
| Look | `list_examples`, `generate` (greedy or sampled, up to 8 prompts per call, original vs edit on the same seeds, with interventions), `read` (teacher-forced log-probs and activations of its own texts), `logit_lens`, `inspect_neuron` |
| Analyse | `compute` (a numeric workspace over read-only copies of every weight, with torch's linear algebra), `make_direction`, `attribute_neurons`, `attribute_heads`, `make_pairs`, `screen_layers` |
| Edit | `build_edit` (direction removal, local projection, neuron and head scaling, gated neurons, free-form `delta` / `set_columns` / `set_rows`, `combine`), `fit_edit` (a low-rank change learned by gradient descent on texts it chooses) |
| Check and finish | `screen_edit`, `verify_edit` (the playbook's full checks), `submit_patch` (only an edit that passed `verify_edit`), `give_up`, `note`, `request_capability` |
| In `wsr program` | `practice_eval`, `settings_eval` (serving settings: penalties, sampling, a system prompt, template arguments), `steer_eval` (add or clamp), `checkpoint`, `final_choice` |

**Limits it cannot change** (defaults; flags in the reference): 80 turns, $5 of API spend, 6M tokens; 6 full
verifications, 40 screens, 60 generations, 80 reads, 300 workspace runs of 120 s each; 8 fits (rank at most
32, 300 steps, 300 s); an edit of at most 5M free parameters, rank 1,024 per tensor, a change no larger than
the tensor itself, no plain rescale of a whole tensor, strengths at most 10; a gate may fire on at most 2% of
the positions that must stay unchanged. The input embedding, output layer and final norm are frozen unless you
unlock them with `--agent-allow` (`embed`, `unembed`, `final_norm`, `rescale`, `large`).

**Its workspace** is a small language, not Python: no imports, files, network, processes or attribute tricks;
every weight it reads is a copy; sizes are checked before anything large is computed (64M elements per value, 2
GB in all). Every trial edit is undone after the call that tried it. When it needs something its tools do not
allow, it files a capability request with evidence; the report names the flag that grants it, and
`--resume OUT_DIR` continues the run with its workspace, edits and notes. Everything it saw and did is in
`agent_transcript.json`.

## Weight-space programming: goals, profiles and the held-out judge

For a goal measured over a dataset ("stop over-refusing safe prompts", "fewer blank replies"), not one prompt.

**The goal file** (JSON, or YAML with PyYAML):

```json
{
  "name": "llama-over-refusal",
  "kind": "repair",
  "model": "./Llama-3.2-1B-Instruct",
  "protocol": {"template_kwargs": {}, "sampling": {}, "max_new_tokens": 256, "system": null},
  "practice": "practice/items.jsonl",
  "heldout": "heldout/items.jsonl",
  "confirmation": "confirmation/items.jsonl",
  "grader": "detector:refusal",
  "judge": {"alpha": 0.05, "min_gain": 0.05},
  "guardrails": [{"suite": "wsr_regression"}, {"suite": "semantic_regression_canary", "max_lost": 1},
                 {"data": "general.jsonl", "grader": "detector:any", "max_drop": 0.02}],
  "budget": {"practice_evals": 10, "verifications": 6, "usd": 5}
}
```

- `kind`: `repair` or `capability`. `sampling`: empty for greedy, else `temperature`, `top_p`, `top_k`, `min_p`,
  `repetition_penalty`, `presence_penalty`.
- Data rows (JSONL): `messages`, or `prompt` / `problem` / `question` / `input` / `instruction`; optional
  `answer` (else a `\boxed{}` answer in `solution`), `pattern`, `id`, `expect: "refuse"` (a request that must be
  refused) and `lang`.
- Graders: `exact`, `contains`, `regex:<pattern>`, `math`, `detector:<group>` or `detector:any` (no detector of
  that group fired). Groups: `blank`, `loop`, `garbage`, `template_leak`, `language`, `refusal`, `vision`,
  `endpoint`, `structure`.
- Three sets: `practice` (the tools and the agent may use it), `heldout` (also accepted as `validation`: judges
  every candidate, every look logged) and `confirmation` (optional: evaluated once, on one frozen candidate).
- `judge`: `alpha` (0 to 0.1, default 0.05) for the exact sign test and `min_gain` (a share of the metric,
  default 0.05 = 5 points).
- Guardrails: `wsr_regression` (the 17-prompt suite; `max_drift`, `min_agreement`, `allow_safety_loss`),
  `semantic_regression_canary` (`max_lost`, default 1) and data sets that may drop by at most `max_drop`.
- `budget`: the agent's limits; the `--practice-evals`, `--agent-verifications` and `--agent-budget-usd` flags
  override them.

**Building a goal.** `wsr objective init` writes a template; `check` reports rows, gradable rows, overlap between
any two sets, stray control characters, and whether the confirmation set was already used; `split` shuffles a
file into practice and held-out. `wsr objective from-failure` goes from one failing prompt (or a profile group)
to a goal: it confirms the failure, finds items that fail the same way (in a local dataset, a public one with
`--dataset auto`, or rewrites of your prompt) and splits them, stratified, three ways. `--confirm-share`
(default 0.25) is the share of the failing items kept for confirmation; the split never shrinks validation below
6 failing items, and gives no confirmation set when fewer than 5 failing items would be left for it.

**Profile.** `wsr profile` runs triage over a dataset and groups the failures (`EMPTY`, `LOOP`,
`TRUNCATED_LOOP`, `TRUNCATED_WORKING`, `GARBAGE`, `TEMPLATE_LEAK`, `REFUSAL`, `COMPLIED`, `LANGUAGE`,
`WRONG_ANSWER`), each with a cause from tests that ran on a few members: `decoding` (sampling escapes it),
`budget` (it finishes at twice the cap), `weight` (a weight-level candidate), `template`, `intermittent` or
`capability`.

**Program.** `wsr program GOAL --out DIR` profiles the practice set, diagnoses the largest weight-level group,
runs the playbook, judges its candidate on the held-out set, and, with `--agent`, hands over to the agent when
the playbook fails or its candidate is rejected (the agent hears only that it was rejected). `wsr fix` does
`from-failure` and `program` in one process, with one model load. Outcomes:

| Outcome | What ships |
|---|---|
| `shipped` | a weight patch in `<out>/patch` |
| `steering recommended` | `<out>/steering/` (vector, settings, `APPLY.md`) |
| `settings recommended` | serving settings, with the `generation_config.json` keys and llama-server flags in `PROGRAM.md` |
| `not shipped`, `no candidate`, `nothing to edit` | the numbers and the reason in `PROGRAM.md` |

**The judge** (`wsr evaluate`, and inside `program`): the original and the candidate answer the same validation
items with the same seeds, and the items that changed are counted. `improved` needs:

- the exact one-sided sign test (wins against losses, ties ignored) below `alpha` divided by the number of
  candidate slots the run has (1, or 2 when an agent may judge a candidate after the playbook; with stopping at
  the first acceptance this equals Holm's correction);
- a gain of at least `min_gain`;
- at least 5 scored items and every guardrail holding.

Other verdicts: `improved on the task, but a guardrail failed`, `worse` (significantly, the other way),
`inconclusive` with the reason (too few items changed, a gain below the minimum, a set that cannot reach the
bar), and `nothing to compare`. z is printed but decides nothing. Before judging, wsr computes the best result
the set allows (with k items failing, the smallest p is 0.5^k), so a set that cannot prove a fix says so up
front. Every look at the validation set is appended to `<goal>.heldout_log.jsonl`, because a choice made after
many looks overstates the gain.

**Confirmation.** When the goal has a `confirmation` set, the one candidate that passed validation is frozen
(its hash is recorded) and evaluated on it once, with the same rules and guardrails. Only then does it ship; a
candidate that passes validation but not confirmation is not shipped. The set is recorded as used by the hash of
its content in `confirmations.json` under the cache base (and a `<file>.used.json` note next to it), so a later
run, the same file renamed, or a second candidate cannot use it again. A set that could not reach the bar with
its failing items is reported as "inconclusive: insufficient confirmation sample size" before any candidate runs,
and is not used up.

**Guardrails on the served file.** With the llama-server judge, the canary and the suite's reply checks (new
failures, refusals lost) also run on the patched GGUF, compared with the unpatched GGUF of the same quantisation
built by the same llama.cpp. Drift and top-token agreement need log-probabilities of the full-precision model, so
they are marked "not supported" on the GGUF and run on PyTorch. The manifest records each check by format, and
whether the fix is deployable in that format.

## What a fix is made for: identity, manifest, evidence

**Identity.** Every patch has an `identity.json`: the SHA-256 of every tensor of the model it was made on, in the
precision of its files (float32, bfloat16 or float16), with the architecture, the hashes of the tokenizer and
config files, and the Hub revision when known; and the hashes of the edited tensors after the edit. Hashes are
computed from the safetensors files and cached (`identity` in `wsr cache`). `wsr patch apply`, `patch export`,
`evaluate --patch` and `repair --check` compare the model in front of them and stop on any difference (fail
closed): another fine-tune of the same shape, a different revision, a model loaded in another precision is
compared in its file's precision. `--allow-base-mismatch` applies it anyway, with a warning; a patch from an older
wsr without exact hashes needs `--allow-unverified-legacy`. `revert` checks that the result matches the original
hashes again.

**Manifest.** Every judged fix (`<out>/patch`, `<out>/steering` or `<out>/settings`) carries a `MANIFEST.json`:

| Part | What it records |
|---|---|
| `model` | the identity above |
| `artifact` | one SHA-256 over the fix's files |
| `candidate` | a hash of the base model, the fix and its evaluation: the frozen candidate that confirmation judged |
| `evaluation` | the goal and its version, the hash of each data set, the grader, detector and canary versions, the protocol, the judge settings and the number of candidate slots |
| `statistics` | validation and confirmation: wins, losses, ties, the exact p, the gain |
| `guardrails` | each guardrail by format (PyTorch, GGUF and its quantisation, llama-server build) |
| `evidence` | below |
| `decision` | the status, whether it shipped, why, and where it is deployable |

`wsr patch show FIX` prints it (`--json` for JSON); `wsr patch check FIX [MODEL_DIR]` checks that the files still
match their hashes and, with a model, that the model is the one the fix was made for. `wsr patch apply` also
writes `wsr_patch.json` into the patched model, so the patched copy records where it came from.

**Evidence.** Findings of the diagnosis and the checks of a repair are filed under five headings, so a report
says what kind of evidence it has rather than one score: `observation` (the failure was seen), `intervention`
(changing the found part changes the failure), `controls` (a random edit or direction of the same size does
not), `generalization` (fresh variants and held-out items) and `replication` (another run, seed or set). A
validation pass is generalisation, never replication; a confirmation set is.

## Backends: PyTorch and llama-server

Replies come from the full-precision model in PyTorch, batched to the free GPU memory (an out-of-memory error
halves the batch and retries), or from llama-server on a GGUF, which is much faster for long or many replies.
`--backend auto` uses llama-server when its binary and a GGUF are there and you agree to start it; `--judge
auto` exports the original (cached) and the candidate to GGUF and judges on them. The regression guardrail
always runs on PyTorch. A server wsr starts is placed on the GPU only when it fits next to what is loaded, is
stopped when free disk falls below 2 GB, and is stopped when wsr exits.

## Safety: what wsr will not do without you

- **Refusals are kept.** A fix that stops refusing a harmful request the model refused before fails the
  regression suite. `--allow-safety-loss` (or `allow_safety_loss` on the goal's suite guardrail) turns that
  check off; it is off by default.
- **The weights can be left alone.** `--no-weight-edits`: only steering and serving settings may fix it.
- **Nothing unverified ships.** Every fix passes the same checks, and `program` ships only what the held-out
  judge accepts. Unverified agent proposals are saved for you to check, never applied.
- **Your machine stays yours.** The agent's workspace has no file, network or process access. wsr asks before
  it downloads, converts or starts anything (consent kinds `download`, `llama-server`, `convert`; `sandbox` and
  `network` are reserved). Answer at the prompt (`always` / `never` are remembered), or with `--yes`, `--no`,
  `--allow KINDS`, `--deny KINDS`, `WSR_ASSUME`, `WSR_ALLOW`, `WSR_DENY`. Without a terminal the answer is no.
  Heavy steps check how much memory can be committed before the page file would fill the system drive.
- **Model text is data.** Replies and prompts of the model under repair are never instructions to the agent.

## Caching

Per model, keyed by its files, a checksum of the weights as loaded, the precision, device, library versions
and detector version: the regression baseline, the unedited model's replies on a goal's sets (with the batch
plan, so edited runs stay paired), the deep diagnosis (loaded only with the classes recorded for it), and the
calibrated probes. An applied edit or active steering changes the key, so nothing cached is served for it.
Also kept: GGUF conversions of original models, converted public datasets, the llama.cpp converter, and the
exact tensor hashes of model checkpoints. `wsr cache show` lists them; `wsr cache clear
models|gguf|datasets|converter|identity|legacy|all` (or `--model KEY`) deletes only what wsr wrote (`legacy` is
probe caches of older versions). Clearing never deletes `confirmations.json` or remembered answers: a used
confirmation set stays used. `WSR_CACHE=off` turns caching off.

## Files wsr writes

| Command | Files |
|---|---|
| `triage --handoff DIR` | `failing_request.json`, `response.json`, `report.json` |
| `deep --out DIR` | `report.json`, `contrast.json`, `traces.json`, `direction.pt`, `localisation.json`, `weights.json` |
| `repair --out DIR` | `REPAIR.md`, `repair.json`, `deep/`, `patch/` (`patch.json`, `patch.safetensors`), `steering/`, `model/` (`--save-model`) |
| with `--agent` | `agent_transcript.json`, `agent_state/` (for `--resume`), `agent_proposals/`, `capability_requests.json` |
| `program --out DIR` | `PROGRAM.md`, `program.json`, `deep/`, `patch/` or `candidate/`, `candidate_playbook/`, `steering/`, `settings/`, `served/` (GGUFs judged on llama-server), the agent's files |
| every judged fix | `MANIFEST.json` (in `patch/`, `steering/` or `settings/`); a patch also `identity.json` |
| `patch apply --out DIR` | the patched model, with `wsr_patch.json` |
| `evaluate`, `program` | `<goal>.heldout_log.jsonl` next to the goal file |
| confirmation | `confirmations.json` under the cache base, and `<file>.used.json` next to the confirmation file |
| `objective from-failure` | `goal.json`, `practice/`, `heldout/`, `confirmation/` |

## Environment variables

| Variable | Effect |
|---|---|
| `WSR_CACHE=off` | turn caching off |
| `WSR_CACHE_DIR` | where wsr keeps its cache and remembered answers (default `~/.cache/wsr`) |
| `WSR_DATA_DIR` | where converted datasets go (default `~/.cache/wsr/datasets`) |
| `WSR_BATCH=off`, `WSR_MAX_BATCH=N` | turn batched generation off, or cap the batch |
| `WSR_ASSUME`, `WSR_ALLOW`, `WSR_DENY` | answer wsr's questions in scripts (`yes` / `no`, or kinds) |
| `WSR_LLAMA_SERVER`, `WSR_LLAMA_CPP` | the llama-server binary, and a llama.cpp checkout with the converter |
| `WSR_API_KEY` | the key of the server under triage (instead of `--api-key`) |
| `HF_TOKEN` | Hugging Face token for upstream comparisons of gated models |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | the agent's key (any variable with `--agent-key-env`) |
| `NO_COLOR` | plain text output |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | no failure found; for `repair`, `program` and `fix`, a fix that passed |
| 1 | a failure found; for `repair`, `program` and `fix`, no accepted fix |
| 2 | a usage or read error |
| 3 | a step needed a yes that was not given |
| 130 | interrupted |

<!-- reference:start (generated by scripts/make_guide.py; do not edit by hand) -->

## Command reference

Every command and option, as `wsr <command> --help` prints them.

### `wsr triage`

Run the full layered triage.

| Option | Default | What it does |
|---|---|---|
| `--model-dir` |  | HF model directory or .gguf file |
| `--endpoint` |  | server base URL, e.g. http://localhost:11434 |
| `--model` |  | model name as the server knows it |
| `--api` `openai\|ollama` | `openai` | openai = /v1/chat/completions (default), ollama = native /api/chat |
| `--api-key` |  | the server's key (or set WSR_API_KEY, which keeps it out of the shell history) |
| `--backend` |  | skip detection: ollama, llama-server, lmstudio, vllm, openai-compatible |
| `--timeout` | `120` | seconds per request |
| `--budget` | `300` | total seconds for all endpoint calls |
| `--prompt` |  | the failing user prompt |
| `--prompt-file` |  | file with the failing user prompt |
| `--system` |  | system prompt to send with --prompt |
| `--messages` |  | JSON file with the failing messages array |
| `--request` |  | JSON file with the request body / parameters that failed |
| `--response` |  | JSON file with a saved failing response (no live call needed) |
| `--temperature` |  | the temperature the failing request used |
| `--max-tokens` |  | the request's max_tokens (to tell a cut reply from a finished one) |
| `--top-p` |  | the top_p the failing request used |
| `--repeat-penalty` |  | the repeat penalty the failing request used |
| `--preset` `chat\|lenient\|longForm\|strictJson` | `chat` | detector thresholds: chat, strictJson, longForm or lenient |
| `--expect-lang` |  | expected language as an ISO code (28 supported: en fr de es ru uk hi zh ja ko ar th ...) |
| `--expect-script` |  | expected script: latin han kana hangul cyrillic arabic devanagari ... |
| `--expect-json` |  | the reply must be JSON |
| `--check-echo` |  | flag output copied from the prompt (not for rewrites) |
| `--upstream` |  | HF repo[@rev] to compare files and template against |
| `--hash` |  | sha256 the weights (slow) |
| `--scan-values` |  | scan safetensors for NaN/Inf (needs numpy) |
| `--probes` |  | comma list of probes: reasoning,bos,smoke,stop,finish,tools,determinism,cache,stream,vision,context,concurrency (default: all but the slow ones: context, determinism, cache, concurrency) |
| `--no-probes` |  | skip the live probes |
| `--differential` |  | rerun the failure flipping one factor at a time |
| `--alt` |  | alternate endpoint for the differential: URL,MODEL |
| `--condition` |  | extra known-issue conditions, e.g. vulkan |
| `--known-issues` |  | extra known-issues JSON file |
| `--ctx` |  | context size for the memory-fit estimate (default: the server's, else 32768) |
| `--handoff` |  | write the hand-off package (request, response, report) here |
| `--server-args` |  | the server's launch flags, e.g. "-c 8192 -np 4 -ctk q4_0 -fa on" (llama-server / vLLM); Ollama's OLLAMA_* environment is read locally |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr file`

Layer 1: structural validation of model files.

| Option | Default | What it does |
|---|---|---|
| `path` |  | model directory, or a .safetensors / .gguf file |
| `--scan-values` |  | scan safetensors for NaN / Inf (needs numpy) |
| `--upstream` |  | Hugging Face repo[@revision] to compare files and template against |
| `--hash` |  | sha256 the weights and compare with the Hub (slow) |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr config`

Layer 2: template, EOS, tokenizer and memory checks.

| Option | Default | What it does |
|---|---|---|
| `path` |  | model directory, or a .safetensors / .gguf file |
| `--upstream` |  | Hugging Face repo[@revision] to compare files and template against |
| `--ctx` | `32768` | context size for the memory-fit estimate |
| `--no-memory` |  | skip the memory-fit estimate |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr response`

Layer 3: classify a saved provider response (JSON, or - for stdin).

| Option | Default | What it does |
|---|---|---|
| `file` |  | the file to read (- for stdin) |
| `--request` |  | the request JSON (for max_tokens and tool schemas) |
| `--max-tokens` |  | the request's max_tokens (to tell a cut reply from a finished one) |
| `--preset` `chat\|lenient\|longForm\|strictJson` | `chat` | detector thresholds: chat, strictJson, longForm or lenient |
| `--expect-lang` |  | expected language as an ISO code (28 supported: en fr de es ru uk hi zh ja ko ar th ...) |
| `--expect-script` |  | expected script: latin han kana hangul cyrillic arabic devanagari ... |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr output`

Layer 4: classify output text (file, or - for stdin).

| Option | Default | What it does |
|---|---|---|
| `file` |  | the file to read (- for stdin) |
| `--finish-reason` |  | the finish reason the server reported |
| `--expect-json` |  | the reply must be JSON |
| `--schema` |  | JSON schema file |
| `--prompt-file` |  | enable PROMPT_ECHO against this prompt |
| `--agent-trace` |  | JSON list of turns for AGENT_LOOP |
| `--preset` `chat\|lenient\|longForm\|strictJson` | `chat` | detector thresholds: chat, strictJson, longForm or lenient |
| `--expect-lang` |  | expected language as an ISO code (28 supported: en fr de es ru uk hi zh ja ko ar th ...) |
| `--expect-script` |  | expected script: latin han kana hangul cyrillic arabic devanagari ... |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr probe`

Layer 5: probe a live endpoint.

| Option | Default | What it does |
|---|---|---|
| `endpoint` |  | the server's base URL, e.g. http://localhost:8080 |
| `--model` |  | model name as the server knows it |
| `--api` `openai\|ollama` | `openai` | openai = /v1/chat/completions (default), ollama = native /api/chat |
| `--api-key` |  | the server's key (or set WSR_API_KEY, which keeps it out of the shell history) |
| `--backend` |  | skip detection: ollama, llama-server, lmstudio, vllm, openai-compatible |
| `--timeout` | `120` | seconds per request |
| `--budget` | `300` | total seconds for all endpoint calls |
| `--only` |  | comma list: reasoning,bos,smoke,stop,finish,tools,determinism,cache,stream,vision,context,concurrency |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr deep`

Layer 7: weight-level diagnosis on a full-precision HF model (needs torch).

| Option | Default | What it does |
|---|---|---|
| `model_dir` |  | Hugging Face model directory (full precision) |
| `--prompt` |  | the failing user prompt |
| `--system` |  | system prompt to send with --prompt |
| `--messages` |  | JSON file with the failing messages array |
| `--kind` `auto\|blank\|loop\|refusal\|language\|other` | `auto` | the failure kind (auto: detected from the reply) |
| `--failing` |  | another prompt that fails the same way |
| `--passing` |  | a near-identical prompt that works |
| `--expect-lang` |  | expected answer language (for wrong-language failures) |
| `--max-new` | `48` | tokens generated per run (default 48) |
| `--dtype` `auto\|float32\|bfloat16\|float16` | `auto` | auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16 |
| `--device` | `auto` | auto (CUDA when available), cpu, cuda, cuda:1 ... |
| `--no-thinking` |  | render the template with enable_thinking=false |
| `--base` |  | base model directory, to measure how much fine-tuning changed the culprits |
| `--gguf` |  | the quantised file in use: quantisation error of the implicated tensors |
| `--out` |  | write the surgery hand-off package here |
| `--no-known` |  | skip the known-direction library |
| `--no-probes` |  | skip the calibrated probes (language probe for wrong-language, copy heads for loops) |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr repair`

Surgery: diagnose (as wsr deep), then find and verify the smallest weight edit that fixes it.

| Option | Default | What it does |
|---|---|---|
| `model_dir` |  | Hugging Face model directory (full precision) |
| `--prompt` |  | the failing user prompt |
| `--system` |  | system prompt to send with --prompt |
| `--messages` |  | JSON file with the failing messages array |
| `--kind` `auto\|blank\|loop\|refusal\|language\|other` | `auto` | the failure kind (auto: detected from the reply) |
| `--failing` |  | another prompt that fails the same way |
| `--passing` |  | a near-identical prompt that works |
| `--expect-lang` |  | expected answer language (for wrong-language failures) |
| `--max-new` | `48` | tokens generated per run (default 48) |
| `--dtype` `auto\|float32\|bfloat16\|float16` | `auto` | auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16 |
| `--device` | `auto` | auto (CUDA when available), cpu, cuda, cuda:1 ... |
| `--no-thinking` |  | render the template with enable_thinking=false |
| `--base` |  | base model directory, to measure how much fine-tuning changed the culprits |
| `--gguf` |  | the quantised file in use: quantisation error of the implicated tensors |
| `--out` |  | write the surgery hand-off package here |
| `--no-known` |  | skip the known-direction library |
| `--no-probes` |  | skip the calibrated probes (language probe for wrong-language, copy heads for loops) |
| `--keep` |  | a prompt whose reply must not regress (adds to the built-in regression suite) |
| `--max-drift` | `0.15` | allowed change in log-likelihood of the original regression replies, nats per token |
| `--min-agreement` | `0.85` | share of reply positions whose top token must stay the same |
| `--steering` `fallback\|only\|off` | `fallback` | activation steering against the diagnosed direction: fallback (default) when no weight edit passes, only (no weight edits), off |
| `--allow-safety-loss` |  | accept a patch that stops refusing harmful requests (off by default) |
| `--save-model` |  | also write the patched model to <out>/model (in the checkpoint's precision) |
| `--deep` |  | deep mode: also try local edits (only the layers / MLP neurons writing the failure, strengths up to 6), confirm on fresh prompt variants, and report a random edit of the same size as a control (slower) |
| `--sampled` |  | with --deep: also require a paired sampled comparison (the model's own sampling settings) to show a clearly lower failure rate - for failures that only appear when sampling and on long replies (slow) |
| `--samples` | `4` | --sampled: samples per prompt (default 4) |
| `--sampled-max-new` | `512` | --sampled: tokens per sample (default 512) |
| `--confirm` |  | --deep: a further prompt for the fresh confirmation (failing or working, judged on the unpatched model) |
| `--serve-dtype` `auto\|bfloat16\|float16\|none` | `auto` | check the patch again after rounding to the serving precision (auto: the checkpoint's 16-bit type, else bfloat16; none: skip) |
| `--check` `PATCH_DIR` |  | do not search: check this patch (e.g. one a fast repair found) with the mode's checks |
| `--allow-base-mismatch` |  | use a patch on weights that differ from the ones it was made for (its checks then do not apply; the output records it) |
| `--allow-unverified-legacy` |  | use a patch that records no exact identity of its model (made by an older wsr): nothing shows its checks apply to these weights |
| `--agent` |  | when the rule-based playbook cannot fix it, hand the diagnosis and every failed trial to a frontier model with tools to inspect and edit the weights; only a patch that passes the same checks is accepted |
| `--agent-provider` `anthropic\|openai` | `anthropic` | anthropic: Claude (ANTHROPIC_API_KEY, pip install anthropic); openai: any model behind the OpenAI Chat Completions API - OpenAI itself, or with --agent-base-url any compatible server: OpenRouter, Together, Groq, DeepSeek, Mistral, Gemini's OpenAI endpoint, a local vLLM / Ollama / llama-server (pip install openai) |
| `--agent-model` |  | the agent's model id (anthropic default: claude-opus-5-5; openai: required) |
| `--agent-base-url` |  | openai provider: the compatible server's URL |
| `--agent-key-env` `VAR` |  | environment variable holding the key (default ANTHROPIC_API_KEY / OPENAI_API_KEY); the key itself is never taken on the command line |
| `--agent-effort` |  | reasoning effort (anthropic: low\|medium\|high\|xhigh\|max, default high; openai: passed as reasoning_effort only when given) |
| `--agent-price` `USD` |  | openai provider: USD per million tokens - IN OUT [CACHED]: input, output and, optionally, cached input (else cached input counts at the input price), so --agent-budget-usd can be enforced (without it the token cap is the limit) |
| `--agent-turn-tokens` | `16000` | output tokens per agent turn, reasoning included (lower it for small local models that think at length) |
| `--agent-timeout` | `600.0` | openai provider: seconds per request (raise it for a local model on the CPU) |
| `--agent-extra` `JSON` |  | openai provider: extra request fields for the server, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}' for llama-server |
| `--agent-max-tokens` | `6000000` | stop the agent past this many input + output tokens (every provider) |
| `--agent-budget-usd` |  | stop the agent past this API spend (default: the goal file's budget.usd, else 5) |
| `--agent-turns` | `80` | stop the agent after this many turns |
| `--agent-verifications` |  | full checks the agent may run (default: the goal file's budget.verifications, else 6) |
| `--agent-allow` `LIST` |  | comma list of opt-in unlocks for the agent's free-form edits: embed, unembed, final_norm (frozen by default), rescale (whole-tensor rescaling), large (lifts the rank and size caps); the edit must still pass every check |
| `--agent-only` |  | skip the rule-based playbook and hand the case to the agent at once (still judged by the same checks); for testing an agent or when the recipes are known not to fit |
| `--resume` `OUT_DIR` |  | continue an earlier agent run from its output directory (agent_state/): workspace, edits, notes and requests come back; proposals the new --agent-allow unlocks permit are built and verified first. Refuses a different model |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr patch`

Apply, revert, export, show or check a fix (a patch, steering or settings directory with its MANIFEST.json).

| Option | Default | What it does |
|---|---|---|
| `action` |  | apply, revert, or export (a GGUF) |
| `model_dir` |  | apply / revert / export: the model directory; show / check: the fix's directory (<out>/patch, <out>/steering or <out>/settings) |
| `patch_dir` (optional) |  | apply / revert / export: the patch directory written by wsr repair / program (<out>/patch); export without it writes the unpatched model (the comparison file); check: optionally the model directory to check the fix against |
| `--out` |  | apply / revert: where to write the patched model |
| `--allow-base-mismatch` |  | use a patch on weights that differ from the ones it was made for (its checks then do not apply; the output records it) |
| `--allow-unverified-legacy` |  | use a patch that records no exact identity of its model (made by an older wsr): nothing shows its checks apply to these weights |
| `--gguf` |  | export: the GGUF file to write |
| `--quant` | `Q8_0` | export: F16, BF16, Q8_0 (converter) or any llama-quantize type such as Q4_K_M (default Q8_0) |
| `--llama-cpp` |  | export: a llama.cpp checkout with convert_hf_to_gguf.py (else WSR_LLAMA_CPP, then one wsr fetched before, then ask to download it) |
| `--llama-server` `PATH` |  | export: llama-server binary or folder (its build picks the converter version; llama-quantize is looked for next to it) |
| `--yes` |  | answer yes to every question wsr would ask |
| `--no` |  | answer no to every question (fall back where possible) |
| `--allow` `KINDS` |  | answer yes for these kinds: download, llama-server, convert, sandbox, network |
| `--deny` `KINDS` |  | answer no for these kinds |
| `--json` |  | show / check: print JSON |

### `wsr quant`

GGUF tensor statistics and quantisation error vs the original weights.

| Option | Default | What it does |
|---|---|---|
| `gguf` |  | the GGUF file |
| `--reference` |  | HF directory (safetensors) the GGUF was converted from |
| `--only` |  | regex on tensor names, e.g. 'blk\.1[0-9]\.ffn' |
| `--top` | `10` | how many to list |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr qbisect`

Which quantised tensor causes a failure: rebuild the served model from the GGUF in PyTorch and swap tensors back to full precision.

| Option | Default | What it does |
|---|---|---|
| `model_dir` |  | full-precision HF directory the GGUF was converted from |
| `gguf` |  | the quantised file in use |
| `--prompt` |  | the failing user prompt |
| `--system` |  | system prompt to send with --prompt |
| `--messages` |  | JSON file with the failing conversation |
| `--expect-lang` |  | language the reply must be in |
| `--max-new` | `64` | tokens generated per reply |
| `--max-set` | `24` | largest tensor set to try before calling it diffuse |
| `--device` | `auto` | auto (CUDA when available), cpu, cuda, cuda:1 ... |
| `--dtype` `auto\|float32\|bfloat16\|float16` | `auto` | auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16 |
| `--no-thinking` |  | render the chat template with enable_thinking=false |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr objective init`

Write a goal file to fill in (JSON, or YAML with PyYAML).

| Option | Default | What it does |
|---|---|---|
| `out` |  | path of the goal file to write (.json / .yaml) |
| `--name` |  | a name for it |
| `--model` |  | the model directory the goal is for |
| `--practice` |  | practice JSONL file (the agent may use it) |
| `--heldout` |  | held-out JSONL file (only the judge reads it) |
| `--grader` |  | exact \| contains \| regex:<pattern> \| math \| detector:<group> |
| `--force` |  | overwrite an existing file |

### `wsr objective check`

Load a goal file and check its data, grader and practice/held-out overlap.

| Option | Default | What it does |
|---|---|---|
| `objective` |  | the goal file (JSON or YAML) |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr objective split`

Shuffle a JSONL file (seeded) into practice and held-out files.

| Option | Default | What it does |
|---|---|---|
| `data` |  | JSONL file to shuffle and split |
| `--practice` |  | practice JSONL file (the agent may use it) |
| `--heldout` |  | held-out JSONL file (only the judge reads it) |
| `--heldout-share` | `0.5` | share of the items that goes to the held-out set |
| `--limit` |  | first N rows only |
| `--seed` | `0` | random seed |

### `wsr objective from-failure`

Turn a failure (a prompt, or a profile group) into a goal file: confirm it, find items that fail the same way, split them (stratified) into practice and held-out.

| Option | Default | What it does |
|---|---|---|
| `model_dir` (optional) |  | HF model directory (not needed with --profile) |
| `--prompt` |  | the failing user prompt |
| `--system` |  | system prompt to send with --prompt |
| `--messages` |  | JSON file with the failing messages |
| `--failing` |  | another prompt that fails the same way |
| `--passing` |  | a similar prompt that works |
| `--dataset` |  | local JSONL to measure the hit rate in; 'auto' names the public source that fits (wsr does not download it) and uses rewrites of the prompt |
| `--profile` |  | a profile JSON from wsr profile --out |
| `--group` |  | with --profile: the label to build the objective for |
| `--grader` |  | default: the failure's detector (detector:<group>) |
| `--samples` | `200` | dataset rows to run (default 200) |
| `--heldout-share` | `0.5` | share of the items that goes to the held-out set |
| `--confirm-share` | `0.25` | share of the failing items kept as a confirmation set (used once, for one frozen candidate); none when too few items fail: 0 for none |
| `--max-new` | `256` | tokens generated per reply |
| `--no-thinking` |  | render the chat template with enable_thinking=false |
| `--expect-lang` |  | language the reply must be in (ISO code) |
| `--deep` |  | also run the full deep diagnosis and stop if it is not weight-level |
| `--name` |  | a name for it |
| `--seed` | `0` | random seed |
| `--out-dir` |  | where goal.json, practice/ and heldout/ are written |
| `--dtype` `auto\|float32\|bfloat16\|float16` | `auto` | auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16 |
| `--device` | `auto` | auto (CUDA when available), cpu, cuda, cuda:1 ... |
| `--backend` `auto\|torch\|llama-server` | `auto` | auto: llama-server when its binary and --gguf are there and you agree to start it (faster for long or many replies), else PyTorch; torch: always the full-precision model |
| `--gguf` |  | GGUF of the model, for the llama-server backend |
| `--endpoint` |  | an already-running llama-server (http://host:port): used as is, nothing started |
| `--llama-server` `PATH` |  | llama-server binary or its folder (default: WSR_LLAMA_SERVER, then PATH) |
| `--parallel` | `4` | llama-server: requests in flight (slots) |
| `--yes` |  | answer yes to every question wsr would ask |
| `--no` |  | answer no to every question (fall back where possible) |
| `--allow` `KINDS` |  | answer yes for these kinds: download, llama-server, convert, sandbox, network |
| `--deny` `KINDS` |  | answer no for these kinds |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr program`

Work towards a goal file: profile the practice set, diagnose the largest weight-level failure group, try the playbook, then (with --agent) the agent with practice tools; a held-out judge decides what ships.

| Option | Default | What it does |
|---|---|---|
| `objective` |  | the goal file (JSON or YAML) |
| `model_dir` (optional) |  | HF model directory (default: the goal file's model) |
| `--out` |  | where PROGRAM.md, program.json and the patch go |
| `--profile-limit` | `40` | practice items to profile (default 40) |
| `--eval-limit` |  | held-out items the judge scores (default: all) |
| `--practice-evals` |  | practice_eval calls the agent may make (default: the goal file's budget.practice_evals, else 10) |
| `--no-playbook` |  | skip the rule-based recipes |
| `--no-weight-edits` |  | the weights must not change: the playbook only steers and the agent may only steer or change serving settings; a fix ships as a steering or settings, never a patch |
| `--steering` `fallback\|only\|off` | `fallback` | activation steering against the diagnosed direction: fallback (default) when no weight edit passes, only (no weight edits), off |
| `--allow-safety-loss` |  | let a fix stop refusing the regression suite's harmful requests (off by default: a fix for something else should not change refusals unnoticed); same as allow_safety_loss: true on the goal's wsr_regression guardrail |
| `--judge` `auto\|torch\|llama-server` | `auto` | where the held-out judge runs. auto: on llama-server when it is there and you agree (the candidate is exported to a GGUF; much faster for long replies), else PyTorch batched to free memory; torch: never asks |
| `--judge-quant` | `Q8_0` | GGUF type the llama-server judge serves (default Q8_0) |
| `--fast` |  | the playbook's fast checks instead of deep ones (no fresh variants or random control; the held-out judge still decides) |
| `--dtype` `auto\|float32\|bfloat16\|float16` | `auto` | auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16 |
| `--device` | `auto` | auto (CUDA when available), cpu, cuda, cuda:1 ... |
| `--agent` |  | when the rule-based playbook cannot fix it, hand the diagnosis and every failed trial to a frontier model with tools to inspect and edit the weights; only a patch that passes the same checks is accepted |
| `--agent-provider` `anthropic\|openai` | `anthropic` | anthropic: Claude (ANTHROPIC_API_KEY, pip install anthropic); openai: any model behind the OpenAI Chat Completions API - OpenAI itself, or with --agent-base-url any compatible server: OpenRouter, Together, Groq, DeepSeek, Mistral, Gemini's OpenAI endpoint, a local vLLM / Ollama / llama-server (pip install openai) |
| `--agent-model` |  | the agent's model id (anthropic default: claude-opus-5-5; openai: required) |
| `--agent-base-url` |  | openai provider: the compatible server's URL |
| `--agent-key-env` `VAR` |  | environment variable holding the key (default ANTHROPIC_API_KEY / OPENAI_API_KEY); the key itself is never taken on the command line |
| `--agent-effort` |  | reasoning effort (anthropic: low\|medium\|high\|xhigh\|max, default high; openai: passed as reasoning_effort only when given) |
| `--agent-price` `USD` |  | openai provider: USD per million tokens - IN OUT [CACHED]: input, output and, optionally, cached input (else cached input counts at the input price), so --agent-budget-usd can be enforced (without it the token cap is the limit) |
| `--agent-turn-tokens` | `16000` | output tokens per agent turn, reasoning included (lower it for small local models that think at length) |
| `--agent-timeout` | `600.0` | openai provider: seconds per request (raise it for a local model on the CPU) |
| `--agent-extra` `JSON` |  | openai provider: extra request fields for the server, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}' for llama-server |
| `--agent-max-tokens` | `6000000` | stop the agent past this many input + output tokens (every provider) |
| `--agent-budget-usd` |  | stop the agent past this API spend (default: the goal file's budget.usd, else 5) |
| `--agent-turns` | `80` | stop the agent after this many turns |
| `--agent-verifications` |  | full checks the agent may run (default: the goal file's budget.verifications, else 6) |
| `--agent-allow` `LIST` |  | comma list of opt-in unlocks for the agent's free-form edits: embed, unembed, final_norm (frozen by default), rescale (whole-tensor rescaling), large (lifts the rank and size caps); the edit must still pass every check |
| `--backend` `auto\|torch\|llama-server` | `auto` | auto: llama-server when its binary and --gguf are there and you agree to start it (faster for long or many replies), else PyTorch; torch: always the full-precision model |
| `--gguf` |  | GGUF of the model, for the llama-server backend |
| `--endpoint` |  | an already-running llama-server (http://host:port): used as is, nothing started |
| `--llama-server` `PATH` |  | llama-server binary or its folder (default: WSR_LLAMA_SERVER, then PATH) |
| `--parallel` | `4` | llama-server: requests in flight (slots) |
| `--yes` |  | answer yes to every question wsr would ask |
| `--no` |  | answer no to every question (fall back where possible) |
| `--allow` `KINDS` |  | answer yes for these kinds: download, llama-server, convert, sandbox, network |
| `--deny` `KINDS` |  | answer no for these kinds |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr consent show`

### `wsr consent reset`

| Option | Default | What it does |
|---|---|---|
| `kind` (optional) |  | one kind (default: all) |

### `wsr cache show`

### `wsr cache clear`

Delete cache groups (asks first, unless --yes); remembered consent answers are kept.

| Option | Default | What it does |
|---|---|---|
| `groups` (optional) |  | models, gguf, datasets, converter, probes, or all |
| `--model` `KEY` |  | only this model's entries (a key, or its start, as `wsr cache show` lists them) |
| `--yes` |  | delete without asking |

### `wsr fix`

One command, one model load: from a failing prompt to a judged patch (objective from-failure, then program; the served GGUF too when llama-server is used).

| Option | Default | What it does |
|---|---|---|
| `model_dir` |  | Hugging Face model directory (full precision) |
| `--prompt` |  | the failing user prompt |
| `--system` |  | system prompt to send with --prompt |
| `--messages` |  | JSON file with the failing messages |
| `--failing` |  | another prompt that fails the same way |
| `--passing` |  | a similar prompt that works |
| `--dataset` |  | local JSONL, or 'auto' (asks before downloading the matching public set) |
| `--expect-lang` |  | language the reply must be in (ISO code) |
| `--max-new` | `256` | tokens generated per reply |
| `--no-thinking` |  | render the chat template with enable_thinking=false |
| `--samples` | `200` | dataset rows to run |
| `--heldout-share` | `0.5` | share of the items that goes to the held-out set |
| `--confirm-share` | `0.25` | share of the failing items kept as a confirmation set (used once, for one frozen candidate); none when too few items fail: 0 for none |
| `--seed` | `0` | random seed |
| `--out` |  | goal/ (the objective) and run/ (PROGRAM.md, patch) go here |
| `--profile-limit` | `40` | practice items to profile |
| `--eval-limit` |  | held-out items the judge scores (default: all) |
| `--practice-evals` |  | practice_eval calls the agent may make (default: the goal file's budget, else 10) |
| `--no-playbook` |  | skip the rule-based recipes |
| `--no-weight-edits` |  | the weights must not change: the playbook only steers and the agent may only steer or change serving settings; a fix ships as a steering or settings, never a patch |
| `--steering` `fallback\|only\|off` | `fallback` | activation steering against the diagnosed direction: fallback (default) when no weight edit passes, only (no weight edits), off |
| `--allow-safety-loss` |  | let a fix stop refusing the regression suite's harmful requests (off by default: a fix for something else should not change refusals unnoticed); same as allow_safety_loss: true on the goal's wsr_regression guardrail |
| `--fast` |  | the playbook's fast checks instead of deep ones (the held-out judge still decides) |
| `--judge` `auto\|torch\|llama-server` | `auto` | where the held-out judge runs: auto (llama-server when there and allowed), torch, llama-server |
| `--judge-quant` | `Q8_0` | GGUF type the llama-server judge serves |
| `--dtype` `auto\|float32\|bfloat16\|float16` | `auto` | auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16 |
| `--device` | `auto` | auto (CUDA when available), cpu, cuda, cuda:1 ... |
| `--agent` |  | when the rule-based playbook cannot fix it, hand the diagnosis and every failed trial to a frontier model with tools to inspect and edit the weights; only a patch that passes the same checks is accepted |
| `--agent-provider` `anthropic\|openai` | `anthropic` | anthropic: Claude (ANTHROPIC_API_KEY, pip install anthropic); openai: any model behind the OpenAI Chat Completions API - OpenAI itself, or with --agent-base-url any compatible server: OpenRouter, Together, Groq, DeepSeek, Mistral, Gemini's OpenAI endpoint, a local vLLM / Ollama / llama-server (pip install openai) |
| `--agent-model` |  | the agent's model id (anthropic default: claude-opus-5-5; openai: required) |
| `--agent-base-url` |  | openai provider: the compatible server's URL |
| `--agent-key-env` `VAR` |  | environment variable holding the key (default ANTHROPIC_API_KEY / OPENAI_API_KEY); the key itself is never taken on the command line |
| `--agent-effort` |  | reasoning effort (anthropic: low\|medium\|high\|xhigh\|max, default high; openai: passed as reasoning_effort only when given) |
| `--agent-price` `USD` |  | openai provider: USD per million tokens - IN OUT [CACHED]: input, output and, optionally, cached input (else cached input counts at the input price), so --agent-budget-usd can be enforced (without it the token cap is the limit) |
| `--agent-turn-tokens` | `16000` | output tokens per agent turn, reasoning included (lower it for small local models that think at length) |
| `--agent-timeout` | `600.0` | openai provider: seconds per request (raise it for a local model on the CPU) |
| `--agent-extra` `JSON` |  | openai provider: extra request fields for the server, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}' for llama-server |
| `--agent-max-tokens` | `6000000` | stop the agent past this many input + output tokens (every provider) |
| `--agent-budget-usd` |  | stop the agent past this API spend (default: the goal file's budget.usd, else 5) |
| `--agent-turns` | `80` | stop the agent after this many turns |
| `--agent-verifications` |  | full checks the agent may run (default: the goal file's budget.verifications, else 6) |
| `--agent-allow` `LIST` |  | comma list of opt-in unlocks for the agent's free-form edits: embed, unembed, final_norm (frozen by default), rescale (whole-tensor rescaling), large (lifts the rank and size caps); the edit must still pass every check |
| `--backend` `auto\|torch\|llama-server` | `auto` | auto: llama-server when its binary and --gguf are there and you agree to start it (faster for long or many replies), else PyTorch; torch: always the full-precision model |
| `--gguf` |  | GGUF of the model, for the llama-server backend |
| `--endpoint` |  | an already-running llama-server (http://host:port): used as is, nothing started |
| `--llama-server` `PATH` |  | llama-server binary or its folder (default: WSR_LLAMA_SERVER, then PATH) |
| `--parallel` | `4` | llama-server: requests in flight (slots) |
| `--yes` |  | answer yes to every question wsr would ask |
| `--no` |  | answer no to every question (fall back where possible) |
| `--allow` `KINDS` |  | answer yes for these kinds: download, llama-server, convert, sandbox, network |
| `--deny` `KINDS` |  | answer no for these kinds |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr data list`

The known datasets, and which are already on disk.

### `wsr data fetch`

Download (after asking) and convert one dataset to wsr JSONL.

| Option | Default | What it does |
|---|---|---|
| `name` |  | a name for it |
| `--yes` |  | answer yes to every question wsr would ask |
| `--no` |  | answer no to every question (fall back where possible) |
| `--allow` `KINDS` |  | answer yes for these kinds: download, llama-server, convert, sandbox, network |
| `--deny` `KINDS` |  | answer no for these kinds |

### `wsr profile`

Triage verdicts over a dataset: failure groups, rates and their causes (from tests that ran), accuracy with a grader.

| Option | Default | What it does |
|---|---|---|
| `model_dir` (optional) |  | HF model directory (default: the goal file's model) |
| `--data` |  | JSONL file of requests (rows: messages \| prompt \| problem \| question, optional answer) |
| `--objective` |  | goal file: protocol, grader and (with --split) data come from it |
| `--split` `practice\|heldout` | `practice` | with --objective: which file to profile (default practice; heldout is logged as a look) |
| `--grader` |  | exact \| contains \| regex:<p> \| math \| detector:<group> (overrides the goal file) |
| `--limit` |  | first N rows only |
| `--max-new` |  | token cap (default: the goal file's, else 512) |
| `--temperature` |  | sample at this temperature (default: greedy, or the goal file's sampling) |
| `--top-p` |  | the top_p the failing request used |
| `--samples` | `1` | samples per item when sampling |
| `--no-thinking` |  | render the template with enable_thinking=false |
| `--system` |  | system prompt for every row |
| `--tests` | `3` | members per group the cause tests run on (0: skip) |
| `--reruns` | `3` | sampled reruns per tested member |
| `--deep` | `0` | also run wsr deep on N members of each weight-level group |
| `--seed` | `0` | random seed |
| `--out` |  | write the profile JSON here (the agent's brief reads it) |
| `--dtype` `auto\|float32\|bfloat16\|float16` | `auto` | auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16 |
| `--device` | `auto` | auto (CUDA when available), cpu, cuda, cuda:1 ... |
| `--backend` `auto\|torch\|llama-server` | `auto` | auto: llama-server when its binary and --gguf are there and you agree to start it (faster for long or many replies), else PyTorch; torch: always the full-precision model |
| `--gguf` |  | GGUF of the model, for the llama-server backend |
| `--endpoint` |  | an already-running llama-server (http://host:port): used as is, nothing started |
| `--llama-server` `PATH` |  | llama-server binary or its folder (default: WSR_LLAMA_SERVER, then PATH) |
| `--parallel` | `4` | llama-server: requests in flight (slots) |
| `--yes` |  | answer yes to every question wsr would ask |
| `--no` |  | answer no to every question (fall back where possible) |
| `--allow` `KINDS` |  | answer yes for these kinds: download, llama-server, convert, sandbox, network |
| `--deny` `KINDS` |  | answer no for these kinds |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

### `wsr evaluate`

Score a model, or a patch against the original, on a goal file's held-out set and guardrails (paired, same seeds).

| Option | Default | What it does |
|---|---|---|
| `model_dir` (optional) |  | HF model directory (default: the goal file's model) |
| `--objective` |  | the goal file (JSON or YAML) |
| `--patch` |  | patch directory (wsr repair's <out>/patch); without it the model is scored |
| `--split` `heldout\|practice` | `heldout` | which set to score |
| `--limit` |  | first N rows only |
| `--samples` | `1` | samples per item when the protocol samples |
| `--seed` | `0` | random seed |
| `--no-guardrails` |  | score the main set only |
| `--out` |  | write the result JSON here |
| `--dtype` `auto\|float32\|bfloat16\|float16` | `auto` | auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16 |
| `--device` | `auto` | auto (CUDA when available), cpu, cuda, cuda:1 ... |
| `--backend` `auto\|torch\|llama-server` | `auto` | auto: llama-server when its binary and --gguf are there and you agree to start it (faster for long or many replies), else PyTorch; torch: always the full-precision model |
| `--gguf` |  | GGUF of the model, for the llama-server backend |
| `--endpoint` |  | an already-running llama-server (http://host:port): used as is, nothing started |
| `--llama-server` `PATH` |  | llama-server binary or its folder (default: WSR_LLAMA_SERVER, then PATH) |
| `--parallel` | `4` | llama-server: requests in flight (slots) |
| `--yes` |  | answer yes to every question wsr would ask |
| `--no` |  | answer no to every question (fall back where possible) |
| `--allow` `KINDS` |  | answer yes for these kinds: download, llama-server, convert, sandbox, network |
| `--deny` `KINDS` |  | answer no for these kinds |
| `--patched-gguf` |  | llama-server backend: the patched model's GGUF (wsr patch export --gguf); without it a patch is scored on PyTorch |
| `--allow-base-mismatch` |  | use a patch on weights that differ from the ones it was made for (its checks then do not apply; the output records it) |
| `--allow-unverified-legacy` |  | use a patch that records no exact identity of its model (made by an older wsr): nothing shows its checks apply to these weights |
| `--json` |  | machine-readable report |
| `-v`, `--verbose` |  | show passing checks and sources |

<!-- reference:end -->
