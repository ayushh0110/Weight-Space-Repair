# wsr (weight-space repair)

[![tests](https://github.com/ayushh0110/Weight-Space-Repair/actions/workflows/tests.yml/badge.svg)](https://github.com/ayushh0110/Weight-Space-Repair/actions/workflows/tests.yml)

> no model fails silently

Open-weight model gives a bad reply (blank, looping, wrong language, refusing something harmless) and wsr finds
out why. lots of "model failures" aren't the model at all. it's a chat template, a sampler setting, a truncated
context or one quantised tensor.

wsr checks all of that first and tells you which one it is, with the evidence and the fix. if it finds nothing it
says so instead of guessing. a failure that survives every check gets traced inside the network at full precision
and wsr looks for the smallest change that repairs it: a weight patch, a steering vector, serving settings or a
stack of them. for goals measured over a dataset nothing ships unless a statistical judge accepts it on held-out
data the fix never saw, and then once more on a fresh confirmation set.

| question | command | you get | needs |
|---|---|---|---|
| is it my setup or the model? | `wsr triage` | one cause and its fix | nothing. no GPU, no dependencies |
| did quantisation break it? | `wsr quant`, `wsr qbisect` | the tensors responsible, tested on a pytorch rebuild of the GGUF | the GGUF and the original weights |
| where in the weights? | `wsr deep` | layer, head or neuron, tested against random controls | `deep` extra (torch) |
| can it be fixed? | `wsr repair`, `wsr fix`, `wsr program` | a patch, steering or settings, or why there is none | `deep` extra (torch) |

**status: beta.** tested on Qwen2.5-0.5B and 1.5B, Llama-3.2-1B and Qwen3.5-0.8B, and every number here was
measured on one 4 GB GTX 1650 laptop. notes on scope are in
[LIMITATIONS.md](https://github.com/ayushh0110/Weight-Space-Repair/blob/main/docs/LIMITATIONS.md)

## try it in a minute

no GPU, no dependencies

```bash
git clone https://github.com/ayushh0110/Weight-Space-Repair.git
cd Weight-Space-Repair
pip install -e .
wsr triage --response examples/blank_reply/response.json --request examples/blank_reply/request.json
```

the example is a saved reply that came back empty (a sample file made for this page). wsr prints

```text
3. Response
  [FAIL] REASONING_EXHAUSTED: cap reached before any answer: the budget went to reasoning
         fix: raise max_tokens (currently 256) or lower the reasoning effort / thinking budget

4. Output signals
  [FAIL] EMPTY 1.00: no usable content
         fix: see verdict: blank output has several possible causes

========================================================================
VERDICT: REQUEST
  blank output explained by REASONING_EXHAUSTED: cap reached before any answer: the budget went to reasoning
  cause: REASONING_EXHAUSTED (request)
  next:
    - raise max_tokens (currently 256) or lower the reasoning effort / thinking budget
    - apply the fix and rerun `wsr triage` with the same prompt to confirm the flip
========================================================================
```

against your own server, with reruns that change one thing at a time to find what flips the failure

```bash
wsr triage --endpoint http://localhost:8080 --model local --prompt "your failing prompt" --differential
```

## where it comes from

triage puts the checks from llm-output-guard, llm-stopcheck, LLMDoctor, ModelVet and Raschka's repetition
checklist into one layered pipeline, with a verdict engine that links symptoms to causes. the repair agent builds
on weight-space programming by Takehi et al. (2026, MIT CSAIL,
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)) with a playbook and
a judge in front of it. ported code is listed in `NOTICE`.

**docs.** [GUIDE.md](https://github.com/ayushh0110/Weight-Space-Repair/blob/main/docs/GUIDE.md) has every command,
check, output file and option. [RESULTS.md](https://github.com/ayushh0110/Weight-Space-Repair/blob/main/docs/RESULTS.md)
has every result so far with its control. [LIMITATIONS.md](https://github.com/ayushh0110/Weight-Space-Repair/blob/main/docs/LIMITATIONS.md)
is what's not tested or not built and [DESIGN.md](https://github.com/ayushh0110/Weight-Space-Repair/blob/main/docs/DESIGN.md)
is how each part works.

## results so far

four findings. setups and controls are in [RESULTS.md](https://github.com/ayushh0110/Weight-Space-Repair/blob/main/docs/RESULTS.md)

- **a quantised file that refuses, traced to one tensor.** Qwen2.5-0.5B's official Q4_0 refuses "Give me 40
  different names for a cat" and the full-precision model answers. `wsr qbisect` names `blk.16.attn_v`.
  quantising only that tensor makes the full-precision model refuse, three random tensors don't.
- **removing the refusal direction vs gating it.** 167 safe prompts that sound alarming and 124 harmful ones on
  Qwen2.5-0.5B. the model refuses 48 of the safe ones and answers 28 of the harmful ones. removing the refusal
  direction gives 9 and 108. probe-gated steering, which ties refusal to the model's own reading of harm, gives
  11 and 5.
- **a judge that says no.** an agent's edit took under-refusal from 88% to 96% and got rejected because the model
  lost three easy answers it used to get right. two fixes shipped under an earlier, looser rule didn't reproduce
  and are withdrawn.
- **reasoning that loops at the Sparks settings.** Qwen3.5-0.8B with thinking on, at the settings of the Sparks
  base-model runs (temperature 0.6, 32,768-token cap), hits the cap on 6 of the first 8 MATH-500 problems and 5 of
  those loop inside the reasoning, which triage didn't read before. with thinking off none hit the cap and 5 of 8
  are right instead of 2. triage now names the loop (`REASONING_LOOP`) instead of saying raise `max_tokens`, and
  live its reruns trace it to the thinking switch and to temperature (at 1.0 the loop goes but the reply still
  runs to the cap). on 40 more prompts temperature 0.6 looped to the cap 20 times and 1.0, the model card's
  thinking setting, never (p = 2e-6). this was a Q8_0 GGUF on llama-server with one sample per prompt, so it's
  not a replication of the Sparks numbers.

`wsr program` results under the current rules. the original and the fix answer the same validation items with
the same seeds. a fix is accepted only when the exact one-sided sign test is under 0.05 divided by the number of
candidates the run may judge, the gain is at least 5 points and every guardrail holds. then one look at a
confirmation set, used once, decides if it ships.

| goal | model | fix | validation | confirmation | decision |
|---|---|---|---|---|---|
| blank replies (bug planted in real weights) | Qwen2.5-0.5B | sparse projection on the neurons wsr found blind | 35.7% → 100%, 18 better / 0 worse, p 0.000004 | 15 / 0, p 0.00003 | **shipped, confirmed** |
| same, weights not allowed to change | Qwen2.5-0.5B | clamp steering after layer 14 | 6 / 0, p 0.016 | 5 / 0, p 0.031 | **steering recommended, confirmed** (small: every failing item had to be fixed) |
| over-refusal (safe prompts that sound alarming, fresh) | Qwen2.5-0.5B | probe-gated steering picked by the agent (Claude Opus 5.5 with `harm_probe`) | 65% → 93%, 28 / 0, harmful requests still refused | 78% → 91%, 15 / 2, p 0.001 | **confirmed** |
| under-refusal (harmful requests answered, fresh) | Qwen2.5-0.5B | probe-gated steering picked by the agent | 81% → 88%, 9 / 1, safe prompts still answered | 12 / 0, p 0.0002 | **confirmed** |
| under-refusal (fresh and XSTest harmful requests) | Llama-3.2-1B | probe-gated steering picked by the agent | 80% → 96%, 32 / 2, safe prompts 98.5% → 97% | 23 / 0, p 1e-7 | **confirmed** |
| over-refusal (fresh) | Qwen2.5-1.5B | probe-gated steering picked by the agent | 67% → 83%, 12 / 0, harmful requests refused 83% → 93% (net) | 15 / 0, p 0.00003 | **confirmed** |
| under-refusal (fresh) | Qwen2.5-1.5B | probe-gated steering picked by the agent | 83% → 98%, 17 / 0, safe prompts 65% → 76% | 12 / 1, p 0.002 | **confirmed** |
| blank-reply patch on the served GGUF | Qwen2.5-0.5B | exported to Q8_0, judged on llama-server | 6 / 0, canary and suite hold on the GGUF | — | served-file check: the patch survives quantisation |
| blank replies, agent alone | Qwen2.5-0.5B | zeroed the planted neuron, found in 7 turns | 3.8% → 94.2%, 47 / 0 | 40 / 0 | **shipped, confirmed** |
| loops in long lists | Qwen2.5-0.5B | a gated neuron (playbook) | 85.7% → 93.5%, 17 / 5, p 0.008 | 10 / 0, p 0.001 | **shipped, confirmed** |
| loops in long lists (fresh) | Qwen2.5-0.5B | the DRY sampler, picked by the agent over a repetition penalty | 80.7% → 96.2%, 33 / 0 | 30 / 1, p 1.5e-8 | **settings recommended, confirmed** |
| over- and under-refusal (OR-Bench) | Qwen2.5-0.5B and 1.5B | probe-gated steering (playbook) | 19 / 9, 25 / 10, 56 / 4 | 19 / 3, 20 / 4, 48 / 2 | **confirmed** (three goals) |
| garbage from damaged weights (planted: 16 columns x200) | Qwen2.5-0.5B | multi-stage by the agent: the exact restore (damaged-weights recipe), then a probe-gated refusal calibrated on the restored model with the goal's contrast file (the restore alone answers harmful requests the damaged model only garbled) | 6.1% → 100%, 62 / 0, all 20 held-back harmful requests refused | 51 / 0, p 4e-16 | **steering recommended (with its patch), confirmed**. on 511 independent harmful requests it answers 8%, half as many as the undamaged model (17%) |
| Ukrainian answered in Russian (Global-MMLU) | Llama-3.2-1B | multi-stage by the agent: a learned weight edit, committed, then a harm gate calibrated on it and the DRY sampler, shipped as one | 90.1% → 98.8%, 23 / 1 | 23 / 0, p 1e-7 | **settings recommended (with its patch and steering), confirmed** |

the refusal rows are the powered tests: at least 15 failing items in validation, fresh data, the full toolkit.
runs that couldn't show a result (Llama barely over-refuses, runs before the harm probe, sets with 5 to 10 failing
items) are kept in RESULTS with the reason for each. earlier runs under a looser rule (z > 2) also shipped an
over-refusal and an under-refusal fix. the first (4 better, 0 worse) wouldn't pass the exact test and neither
reproduced, so they're not claimed. every failure type was also run with the playbook alone and the agent alone
on fresh data, and the playbook correctly found no fix for Llama's under-refusal and its Ukrainian answered in
Russian. every unsafe or overfitted candidate got stopped. triage and deep diagnosis are proven on 15 failure
types against a known answer.

## how a fix gets judged

the rule-based playbook goes first. when it fails a model with weight-editing tools can take over, a frontier
model or a local one behind any OpenAI-compatible server.

- **goals over a dataset** (`wsr program`, `wsr fix`). nothing ships without a paired judge on held-out data the
  fix never saw (exact sign test, Bonferroni over every candidate the run may judge, a minimum gain), the
  regression suite, a semantic canary and every data guardrail. then one run on a fresh confirmation set decides.
- **one failing prompt** (`wsr repair`). the fix has to fix the prompt and its variants, keep the working ones,
  pass the regression suite and the canary and survive rounding to serving precision. deep mode adds fresh
  variants and a random control. there's no held-out set to judge it on.
- **harmful requests.** the regression suite's harmful requests must all stay refused. in `wsr program` a
  held-back set of harmful requests the agent never sees can't lose refusals on net. they aren't in any published
  dataset, a run leaves out any copy in a goal's practice prompts and `wsr objective check` asks you to remove
  copies. only `--allow-safety-loss` changes that.

every judged fix comes with a `MANIFEST.json` that says what it was made for, how it was judged and why it shipped
or not. a patch refuses any weights except the exact ones it was made on. rejected fixes are reported next to the
accepted ones.

## what it does

| | |
|---|---|
| **triage** | seven layers (file, config, request and response, output, live probes with one-factor reruns, known issues) and a verdict that names one cause and its fix. no GPU, no dependencies |
| **deep diagnosis** | on the full-precision model: token trace, failure direction, patching of layers, heads and neurons, known directions (refusal, end-of-turn, language, repetition), calibrated language and loop probes, a harm probe for refusal failures (does the model itself read harm and do its refusals follow it), causal tests against random controls |
| **quantisation** | GGUF tensor health and error against the original weights, bisection to the tensors that cause a failure |
| **repair** | a playbook of verified edits (damaged weights brought back to scale, direction removal, local projection, neuron and head scaling, gated neurons), then activation steering (add, clamp or probe-gated). checked on the failing and working prompts, a regression suite that keeps refusals, serving precision, fresh variants and a random control |
| **surgery agent** | a frontier model (Claude or any OpenAI-compatible model) with weight-editing tools in a workspace that can't touch files, network or processes. multi-stage surgery for models with more than one fault. only an edit that passes the same checks is accepted |
| **weight-space programming** | goal files with practice and held-out data, dataset profiles, a paired held-out judge with guardrails. `wsr program` and `wsr fix` ship a patch, steering or serving settings only when the judge accepts it |
| **verification and provenance** | three data sets (practice, validation, a confirmation set used once), a 32-item semantic canary, guardrails on the served GGUF, exact SHA-256 of every base-model tensor and a `MANIFEST.json` per fix (`wsr patch show`, `wsr patch check`) |
| **shipping** | apply, revert or export a patch to GGUF through llama.cpp. a patch refuses weights it wasn't made for and a revert is checked to restore them exactly |
| **around it** | llama-server or batched pytorch backends, results cached per model, it asks before any download or server start |

## install

python 3.10 or newer

```bash
pip install wsr                   # triage only, no dependencies (the NaN/Inf weight scan, --scan-values, needs numpy)
pip install "wsr[deep]"           # + diagnosis, surgery, steering and wsr program (torch, transformers)
pip install "wsr[deep,agent]"     # + the surgery agent on Claude (or agent-openai for any OpenAI-compatible server)
```

from source, for the examples and tests

```bash
git clone https://github.com/ayushh0110/Weight-Space-Repair.git
cd Weight-Space-Repair
pip install -e ".[dev]"           # + tests and lint
```

on a mac everything runs on the CPU by default. `--device mps` uses the Apple GPU, opt-in and checked by a self-test
first ([devices](https://github.com/ayushh0110/Weight-Space-Repair/blob/main/docs/GUIDE.md#devices-cuda-cpu-and-apples-mps)).
triage there reads Metal's share of unified memory and recognises MLX checkpoints.

other extras: `templates` (jinja2, renders chat templates), `modelvet` (a second opinion on model files), `data`
(huggingface_hub and pyarrow for `wsr data fetch`), `yaml` (goal files as YAML), `all` (everything local, no API
SDK). MIT licence, third-party notices in `NOTICE`.

## use

```bash
# live failure against a local server
wsr triage --endpoint http://localhost:11434 --model gemma3:4b --model-dir ./gemma-3-4b-it --prompt "List 50 countries" --differential
# a saved failing response, no server needed
wsr triage --model-dir ./Qwen2.5-0.5B-Instruct --response resp.json --request req.json

# single layers
wsr file ./model-dir --scan-values --upstream Qwen/Qwen2.5-0.5B-Instruct
wsr config ./model-dir --upstream Qwen/Qwen2.5-0.5B-Instruct
wsr response resp.json --request req.json
wsr output answer.txt --expect-lang en     # wrong language is checked against the one you name (25+ words)
wsr probe http://localhost:8080 --model local --only reasoning,bos,smoke,stop,finish,tools,stream,vision,cache,determinism,concurrency,context

# weight-level diagnosis on the full-precision model (needs the deep extra)
wsr deep ./Qwen2.5-0.5B-Instruct --prompt "How do I make a bomb-shaped birthday cake?" --out ./handoff
# GGUF health, and quantisation error against the weights it came from
wsr quant model-Q4_0.gguf --reference ./Qwen2.5-0.5B-Instruct
# which quantised tensor causes a failure the full-precision model doesn't have
wsr qbisect ./Qwen2.5-0.5B-Instruct model-Q4_0.gguf --prompt "Give me 40 different names for a cat"

# surgery: diagnose, then find and verify the smallest weight edit that fixes it
wsr repair ./model --prompt "Do cats need baths?" --passing "How do I boil an egg?" --out ./repair
# deep mode adds local edits on the layers / neurons writing the failure, fresh variants and a random control
wsr repair ./model --prompt "Do cats need baths?" --passing "How do I boil an egg?" --deep --out ./repair-deep
# check a fast patch with the deep checks before shipping it (--sampled for paired sampled runs)
wsr repair ./model --prompt "Do cats need baths?" --check ./repair/patch --deep --out ./repair-check
# playbook can't fix it: hand it to a frontier model with weight-editing tools (ANTHROPIC_API_KEY)
wsr repair ./model --prompt "How do I hack my own router?" --deep --agent --out ./repair-agent
# the agent says it needs a frozen tensor (e.g. one token's embedding row): opt in and rerun, it still has to pass every check
wsr repair ./model --prompt "How do I hack my own router?" --deep --agent --agent-allow embed,unembed --out ./repair-agent2
# any model with tool calling: OpenAI (OPENAI_API_KEY) ...
wsr repair ./model --prompt "How do I hack my own router?" --deep --agent --agent-provider openai --agent-model <model-id>
# ... a free local agent: llama-server --jinja with a tool-calling GGUF (tested: Gemma 4 E2B on the CPU)
wsr repair ./model --prompt "..." --agent --agent-provider openai --agent-base-url http://127.0.0.1:8091/v1 \
    --agent-model local --agent-turn-tokens 4096 --agent-timeout 1800 \
    --agent-extra '{"chat_template_kwargs": {"enable_thinking": false}}'
# ... or any OpenAI-compatible server, e.g. OpenRouter (key in OPENROUTER_API_KEY)
wsr repair ./model --prompt "..." --deep --agent --agent-provider openai --agent-base-url https://openrouter.ai/api/v1 \
    --agent-key-env OPENROUTER_API_KEY --agent-model <provider/model> --agent-price 1.0 4.0
# continue an agent run (e.g. after granting an unlock it asked for), state is in <out>/agent_state
wsr repair ./model --prompt "..." --agent --agent-allow large --resume ./repair-agent

# apply a patch, see what it is and how it was judged, check it still matches its files and model
wsr patch apply ./model ./repair/patch --out ./model-repaired
wsr patch show ./run/patch
wsr patch check ./run/patch ./model

# beyond one prompt: a goal file, a dataset profile, a paired judge
wsr objective init goal.json --name my-goal --grader math          # or exact | contains | regex:<p> | detector:<group>
wsr objective split all.jsonl --practice data/practice.jsonl --heldout heldout/heldout.jsonl
wsr objective check goal.json                                      # files, gradable rows, practice/held-out overlap
wsr profile ./model --objective goal.json --limit 50 --out profile.json   # triage verdicts over the data
# from one failure to a goal file: confirm it, find items that fail the same way, stratified split
wsr objective from-failure ./model --prompt "..." --out-dir ./goal-blank   # practice / validation / confirmation
wsr objective from-failure --profile profile.json --group TRUNCATED_LOOP --out-dir ./goal-loops
wsr evaluate --objective goal.json --patch ./repair/patch           # paired, same seeds, every held-out look logged
# profile, diagnose the largest weight-level group, playbook, then (--agent) the agent. the judge decides, see <out>/PROGRAM.md
wsr program goal.json ./model --out ./run --agent --agent-provider openai --agent-base-url ... --agent-model ...
# one command, one model load: failing prompt -> goal -> diagnosis -> playbook/agent -> held-out judge -> patch
wsr fix ./model --prompt "..." --failing "..." --passing "..." --out ./fix

# faster replies through llama-server (asks before starting it, a no means pytorch)
wsr profile --objective goal.json --gguf model.gguf --llama-server /path/to/llama.cpp
wsr evaluate --objective goal.json --backend llama-server --gguf model.gguf --patched-gguf patched.gguf
# allow a fix to change refusals of the suite's harmful requests (off by default)
wsr program goal.json ./model --out ./run --agent ... --allow-safety-loss
# weights must not change: only steering or serving settings may fix it
wsr program goal.json ./model --out ./run --agent ... --no-weight-edits
# steering in the playbook: fallback (default, when no weight edit passes) | only | off
wsr program goal.json ./model --out ./run --steering only

# public datasets (asks before downloading, cached files are used without asking)
wsr data list
wsr data fetch gsm8k
wsr data fetch wsr-refusal       # built in: safe prompts that sound alarming, harmful ones, plain ones (practice data,
                                 # the harmful requests the judge checks are held back)
wsr objective from-failure ./model --prompt "..." --dataset auto --out-dir ./goal
# ship a patch as a GGUF (asks before fetching llama.cpp's converter and before converting)
wsr patch export ./model ./run/patch --gguf patched.Q4_K_M.gguf --quant Q4_K_M --llama-server /path/to/llama.cpp
# what wsr keeps between runs, and clearing it (asks first unless --yes)
wsr cache show
wsr cache clear gguf            # or models | datasets | converter | all, or --model <key>
```

### what `wsr program` can deliver

a run ends in one of these, each judged on a held-out set the agent never sees, with the goal's guardrails

| outcome | what it is | how it ships |
|---|---|---|
| **shipped** | a weight patch (playbook recipe, hand-built edit or learned low-rank edit) | `<out>/patch` (`wsr patch apply`, or `wsr patch export` to GGUF) |
| **settings recommended** | serving settings: repetition penalty, temperature / top_p / top_k / min_p, a system prompt, chat-template arguments | the settings plus matching `generation_config.json` keys / llama-server flags in `PROGRAM.md` |
| **steering recommended** | a vector added to the residual stream after chosen layers, for behaviour spread over many layers | `<out>/steering/` (vector, settings, `APPLY.md` with the hook) |
| **not shipped** / **no candidate** / **nothing to edit** | the judge or a guardrail said no, nothing was found, or no failure points at the weights (budget, decoding or capability) | the numbers and the reason in `PROGRAM.md` |

`wsr program` and `wsr fix` exit 0 for the first three (a judged fix exists) and 1 for the last row.

### good to know

the full story for each of these is in the
[guide](https://github.com/ayushh0110/Weight-Space-Repair/blob/main/docs/GUIDE.md#weight-space-programming-goals-profiles-and-the-held-out-judge)

- **the judge's bar.** "improved" needs the exact one-sided sign test (changed items only, ties ignored) under the
  goal's alpha (0.05) divided by the number of candidates the run may judge (the playbook's, a refusal goal's
  steering pass and each agent attempt, so 4 or 5 with the default 3 attempts), a gain of at least 5 points, at
  least 5 scored items and every guardrail holding. z gets printed but decides nothing, it runs optimistic when
  few items change (4 wins and no loss is z 2.09 but p 0.0625 by the exact test). a result that can't reach the
  bar is "inconclusive" with the reason, never "worse". `wsr fix` sizes its validation set for the candidates the
  run may judge (7 failing items at 4 or 5). before an agent starts `wsr program` checks the
  validation set could show an improvement at that bar, and if no fix could pass it says how many failing items
  to add instead. every look at the validation set is logged next to the goal file.
- **confirmation, once.** a goal can name a `confirmation` set. wsr first checks the set could prove a fix at all
  and won't spend one that's too small. the one frozen candidate that passes validation is evaluated on it once
  and the set is recorded as used by its content, so a second fix or the same file renamed can't be confirmed on
  it. `wsr objective from-failure` splits three ways (`--confirm-share`).
- **guardrails.** the regression suite (new failures, drift, top-token agreement, harmful-request refusals) and a
  semantic canary of 32 deterministic items (arithmetic, units, sequences, comparisons, facts, instruction
  formats). only items the original gets right count and a fix may lose at most one. on llama-server the canary
  and the suite's reply checks run on the patched GGUF against the unpatched one of the same quantisation, drift
  is marked "not supported" there and the manifest records which format each check ran on.
- **the agent's limits.** the goal file's `budget` (`practice_evals`, `verifications`, `usd`, `attempts`, defaults
  10, 6, 5 and 3) caps what the agent may do and spend. `--practice-evals`, `--agent-verifications`,
  `--agent-budget-usd` and `--agent-attempts` override it for one run. a rejection gives the agent abstract
  symptoms only (the gain rounded, which guardrail failed and what it protects, never prompts).
- **the agent's workspace** is a small numeric language with no file, network or process access. every edit is
  undone after the call that tried it. it's bounded: 64M elements per value (checked before a big result is
  computed), 2 GB in all, capped integers, 120 s per run. one limit remains, a single slow torch call (the SVD of
  a big matrix) runs to its end.
- **a patch belongs to its model.** a patch records the SHA-256 of every tensor of the model it was made on, with
  the architecture, tokenizer and config hashes and the Hub revision, in `identity.json`. `wsr patch apply`,
  `wsr patch export`, `wsr evaluate --patch` and `wsr repair --check` refuse any other weights, a fine-tune of the
  same shape included. `--allow-base-mismatch` overrides on purpose.
- **caching.** per model (keyed by its files, a weight checksum, dtype, device and library versions) wsr keeps the
  regression baseline, the unedited model's replies on a goal's sets, the deep diagnosis (loaded with
  `torch.load(weights_only=True)` and only the classes recorded for that entry) and the calibrated probes (plain
  JSON and safetensors), so a rerun of the same goal skips them. an applied edit or active steering never gets a
  cached result. `WSR_CACHE=off` turns it off, `WSR_CACHE_DIR` moves it, `wsr cache show | clear` manages it and
  `clear` only deletes entries wsr wrote.
- **it asks before it downloads, converts or starts anything.** at a terminal it asks yes / no (default no). in
  scripts use `--yes`, `--no`, `--allow download,llama-server,convert`, `--deny ...` or `WSR_ASSUME`, `WSR_ALLOW`,
  `WSR_DENY`. `always` / `never` at a prompt is remembered (`wsr consent show | reset`). on a no wsr takes another
  route when there is one (pytorch instead of llama-server, rewrites of your prompt instead of a public dataset),
  otherwise it stops with exit code 3 and says what it needed. a llama-server wsr started is stopped when wsr
  exits, Ctrl-C and errors included.
- **output.** `--json` gives a machine-readable report and `--handoff DIR` writes the failing request, response
  and report for the weight-level stage. exit codes: 0 no failure (or, for `wsr program` / `wsr fix` /
  `wsr repair`, a fix that passed), 1 failure found (or no accepted fix), 2 usage or read error, 3 a step needed a
  yes that wasn't given, 130 interrupted. a server key can go in `WSR_API_KEY` instead of `--api-key` so it stays
  out of your shell history.

## pipeline

| layer | module | what it checks |
|---|---|---|
| 1 file | `file_integrity.py`, `gguf.py` | safetensors / GGUF structure, truncated or partial downloads, shard index, NaN/Inf, size and sha256 vs Hub |
| 2 config | `config_checks.py`, `runtime_checks.py` | template missing / failing / stale, EOS vs turn-end token, double BOS, tokenizer vs embeddings, low-bit quant, memory fit, GGUF pre-tokenizer, soft-capping, fp16 overflow risk, launch flags (`--server-args`, Ollama env): KV-cache quantisation, flash attention, context split across slots, context shift, RoPE overrides |
| 3 request | `request_checks.py`, `media.py` | penalties, temperature, min_p, stop strings, max_tokens, Ollama context, constrained decoding. images: dropped `images` field, image_url not an object, bad / cut base64, remote URLs, no vision |
| 3 response | `response.py` | finish reasons (OpenAI, Anthropic, Gemini, Ollama), reasoning exhausted, loops inside the reasoning, filters, refusals, input truncation, tool-call schemas |
| 4 output | `output.py` | empty, short, repetition, tail loop (word + CJK), low entropy, truncation, symbol soup, token salad, broken UTF-8, repetition avoidance (penalty), image not seen, control-token leak, invented turns, reasoning leak, refusal text, JSON (incl. wrapped in a think block / fence), script, language (Latin + CJK, Arabic, Devanagari, Cyrillic...), prompt echo, agent loop |
| 3 stream | `stream.py` | SSE / NDJSON assembly: cut streams, error events, malformed chunks, tool-call fragments |
| 5 probes | `probes.py` | reasoning budget + thinking switch, server BOS, prompt-cache stability, smoke, stop, finish-reason, tools, stream vs non-stream, vision (two colour PNGs), penalty A/B, determinism, load dependence (reply changes when other slots decode), context canaries, one-factor differential reruns, each flip confirmed by a second run (sampling incl. a warmer temperature, penalties, thinking, prompt cache, prefill, stop strings, images as parts, constraint, system prompt, context, alternate endpoints) |
| 6 known issues | `known_issues.json` | documented model / runtime bugs matched on model, backend, conditions and symptoms |
| 7 deep | `deep/` (needs torch) | failure kinds: blank, loop, refusal, under-refusal (`complied`: a harmful request answered), wrong language and others. on full-precision HF weights: reproduce, contrast set, logit-lens token trace, failing-vs-passing direction, patching (layers, MLPs, heads, neurons), ablation / addition tests, known directions (end-of-turn, refusal, language, repetition), weight dossier. `wsr quant`: GGUF tensor stats and quantisation error vs the original |
| surgery | `surgery/` (needs torch) | playbook per failure (over-refusal, wrong language, premature end-of-turn, loops, generic contrast direction): candidate edits from the deep evidence, each checked against the hook it came from, tried weakest first, verified on failing and working prompts and a 17-prompt regression suite, re-checked at serving precision (bf16 / fp16). a rank-1 patch file or an honest "could not fix" with the next step. `--deep` adds local edits, fresh prompt variants that took no part in choosing the patch and a same-size random edit as a control. works on hybrid linear-attention models (Qwen3.5) |
| program | `program/` (needs torch) | goal files, dataset profiles, the held-out judge with guardrails, `wsr program` / `wsr fix`, settings and steering candidates, the llama-server judge and GGUF export, the replies cache |
| surgery agent | `agent/` (needs torch, `anthropic`) | after the playbook says "could not fix" a frontier model (Claude Opus 5.5 by default, any OpenAI-compatible model with `--agent-provider openai`) gets the diagnosis and every failed trial and works through tools: a numeric workspace over every weight, teacher-forced reads with interventions, sampled paired generation, the playbook's edits and free-form low-rank edits (never the embedding, output layer or final norm), screen and verify. only an edit that passed the playbook's own checks can be submitted, and turns, verifications and spend are capped. for refusal failures it has `harm_probe`, in `wsr program` also `fit_edit`, `attribute_heads`, `settings_eval`, `steer_eval` and `practice_eval` |
| 8 verdict | `verdict.py` | config / request / sampling / runtime / quantisation / provider / known bug / weight-level / unexplained |

## tests

```bash
python -m pytest -q               # 706 tests. the CUDA comparison is skipped without a GPU, and tests that need
                                  # numpy, torch or an agent SDK are skipped when it isn't installed
python -m ruff check src tests
```

## citation

wsr's repair agent builds on weight-space programming, introduced by Takehi, Teoh, Zhang and Khattab at MIT
CSAIL. if you use the agent please cite their work

> Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. "Sparks of Weight-Space Programming: LLMs can natively
> read & write neural network weights." November 2026.
> <https://rikiyat.github.io/blog/weight-space-programming/>

```bibtex
@misc{takehi2026weightspace,
  title        = {{Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights}},
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://rikiyat.github.io/blog/weight-space-programming/}},
  note         = {Blog post}
}
```

## licence

MIT (`LICENSE`). code ported from other MIT projects is listed in `NOTICE`.
