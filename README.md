# wsr — Weight-Space Repair

[![tests](https://github.com/ayushh0110/Weight-Space-Repair/actions/workflows/tests.yml/badge.svg)](https://github.com/ayushh0110/Weight-Space-Repair/actions/workflows/tests.yml)

> No model fails silently.

When an open-weight model gives a bad reply (blank, looping, the wrong language, a refusal of a harmless
request), wsr finds out why. Many apparent model failures are not the model at all: a chat template, a sampler
setting, a truncated context or one quantised tensor. wsr tests each of these and names the cause it finds, with
its evidence and the fix, or says plainly that it found none. A failure that survives all of those is traced
inside the network at full precision, and wsr looks for the smallest change that repairs it: a weight patch, an
activation steering, serving settings, or several of these stacked. For a goal measured over a dataset
(`wsr program`, `wsr fix`), a fix ships only when a statistical judge accepts it on held-out data the fix never
saw, and then once more on a fresh confirmation set.

| Question | Command | What you get | Needs |
|---|---|---|---|
| Is it my setup or the model? | `wsr triage` | one cause and its fix | nothing: no GPU, no dependencies |
| Did quantisation break it? | `wsr quant`, `wsr qbisect` | the tensors responsible (a pruned set, tested on a PyTorch rebuild of the GGUF) | the GGUF and the original weights |
| Where in the weights? | `wsr deep` | the layer, head or neuron, tested against random controls | the `deep` extra (torch) |
| Can it be fixed? | `wsr repair`, `wsr fix`, `wsr program` | a patch, a steering or settings, or the reason there is none | the `deep` extra (torch) |

**Status: beta, small models only.** Deep diagnosis and `wsr program` with the agent are tested on models up to
1.5B parameters, weight repair up to 1B (Qwen2.5-0.5B and 1.5B, Llama-3.2-1B, Qwen3.5-0.8B). Every number here was measured on
one 4 GB GTX 1650. [docs/LIMITATIONS.md](docs/LIMITATIONS.md) lists what is not tested.

## Try it in a minute

No GPU and no dependencies:

```bash
git clone https://github.com/ayushh0110/Weight-Space-Repair.git
cd Weight-Space-Repair
pip install -e .
wsr triage --response examples/blank_reply/response.json --request examples/blank_reply/request.json
```

The example is a saved reply that came back empty (a sample file, written for this page). What wsr prints:

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

Against your own server, with one-factor reruns that show which setting flips the failure:

```bash
wsr triage --endpoint http://localhost:8080 --model local --prompt "your failing prompt" --differential
```

## How a fix is judged

A rule-based playbook goes first; a model with weight-editing tools (a frontier model, or a local one behind any
OpenAI-compatible server) can take over when it fails.

- **Goals over a dataset** (`wsr program`, `wsr fix`): nothing ships without a paired judge on held-out data the
  fix never saw (an exact sign test with a Bonferroni correction over every candidate the run may judge, and a
  minimum gain), the regression suite, a semantic canary and every data guardrail; then one evaluation on a fresh
  confirmation set decides.
- **One failing prompt** (`wsr repair`): the fix must fix the prompt and its variants, keep the working ones, pass
  the regression suite and the canary, and survive rounding to the serving precision; deep mode adds fresh
  variants and a random control. There is no held-out set to judge it on.
- **Harmful requests:** the regression suite's harmful requests must each stay refused, and in `wsr program` a
  held-back set of harmful requests the agent never sees may not lose refusals on net (they are not in any
  published dataset, a run leaves out any copy in a goal's practice prompts, and `wsr objective check` asks for
  copies to be removed). A fix may change that only
  when you allow it (`--allow-safety-loss`).

Every judged fix carries a `MANIFEST.json` that says what it was made for, how it was judged and why it shipped or
not, and a patch refuses any weights other than the exact ones it was made on.

The fixes the judge rejected are reported next to the ones it accepted.

## Where it comes from

The triage layer combines the techniques of llm-output-guard, llm-stopcheck, LLMDoctor, ModelVet and Raschka's
repetition checklist into one layered pipeline, with a verdict engine that links symptoms to causes. The repair
agent follows the weight-space programming of Takehi et al. (2026, MIT CSAIL:
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)), with the playbook
and the judge in front of it. Ported code is listed in `NOTICE`.

**Documentation:** [docs/GUIDE.md](docs/GUIDE.md) covers every command, check, output file and option;
[docs/RESULTS.md](docs/RESULTS.md) every result measured so far, with its control;
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) what is not tested or not built; [docs/DESIGN.md](docs/DESIGN.md)
how each part was built.

## Results so far

Three findings, each with its setup and control in [docs/RESULTS.md](docs/RESULTS.md):

- **A quantised file that refuses, traced to one tensor.** Qwen2.5-0.5B's official Q4_0 refuses "Give me 40
  different names for a cat"; the full-precision model answers. `wsr qbisect` names `blk.16.attn_v`: quantising
  that tensor alone makes the full-precision model refuse, and three random tensors do not.
- **Removing the refusal direction against gating it.** On 167 safe prompts that sound alarming and 124 harmful
  ones (Qwen2.5-0.5B), the model refuses 48 of the safe ones and answers 28 of the harmful ones. Removing the
  refusal direction gives 9 and 108. Probe-gated steering, which ties the refusal to the model's own reading of
  harm, gives 11 and 5.
- **A judge that rejects.** An agent's edit took under-refusal from 88% to 96% and was rejected because the model
  lost three easy answers it had right before. Two fixes shipped under an earlier, looser rule did not reproduce
  and are withdrawn.

Results from `wsr program` under the current rules: the original and the fix answer the same validation items
with the same seeds; a fix is accepted only when the exact one-sided sign test is below 0.05 (divided by the
number of candidates the run may judge), the gain is at least 5 points and every guardrail holds; then one
evaluation on a confirmation set, used once, decides whether it ships.

| Goal | Model | Fix | Validation | Confirmation | Decision |
|---|---|---|---|---|---|
| Blank replies (a bug planted in real weights) | Qwen2.5-0.5B | sparse projection on the neurons wsr found blind | 35.7% → 100%, 18 better / 0 worse, p 0.000004 | 15 / 0, p 0.00003 | **shipped, confirmed** |
| The same, weights not allowed to change | Qwen2.5-0.5B | clamp steering after layer 14 | 6 / 0, p 0.016 | 5 / 0, p 0.031 | **steering recommended, confirmed** (small: every failing item had to be fixed) |
| Over-refusal (safe prompts that sound alarming; fresh) | Qwen2.5-0.5B | probe-gated steering, chosen by the agent (Claude Opus 5.5 with `harm_probe`) | 65% → 93%, 28 / 0; harmful requests still refused | 78% → 91%, 15 / 2, p 0.001 | **confirmed** |
| Under-refusal (harmful requests answered; fresh) | Qwen2.5-0.5B | probe-gated steering, chosen by the agent | 81% → 88%, 9 / 1; safe prompts still answered | 12 / 0, p 0.0002 | **confirmed** |
| Under-refusal (fresh and XSTest harmful requests) | Llama-3.2-1B | probe-gated steering, chosen by the agent | 80% → 96%, 32 / 2; safe prompts 98.5% → 97% | 23 / 0, p 1e-7 | **confirmed** |
| Over-refusal (fresh) | Qwen2.5-1.5B | probe-gated steering, chosen by the agent | 67% → 83%, 12 / 0; harmful requests refused 83% → 93% (net) | 15 / 0, p 0.00003 | **confirmed** |
| Under-refusal (fresh) | Qwen2.5-1.5B | probe-gated steering, chosen by the agent | 83% → 98%, 17 / 0; safe prompts 65% → 76% | 12 / 1, p 0.002 | **confirmed** |
| The blank-reply patch on the served GGUF | Qwen2.5-0.5B | exported to Q8_0, judged on llama-server | 6 / 0; canary and suite hold on the GGUF | — | a served-file check: the patch survives quantisation |
| Blank replies, the agent alone | Qwen2.5-0.5B | zeroed the planted neuron, found in 7 turns | 3.8% → 94.2%, 47 / 0 | 40 / 0 | **shipped, confirmed** |
| Loops in long lists | Qwen2.5-0.5B | a gated neuron (the playbook) | 85.7% → 93.5%, 17 / 5, p 0.008 | 10 / 0, p 0.001 | **shipped, confirmed** |
| Loops in long lists (fresh) | Qwen2.5-0.5B | the DRY sampler, chosen by the agent over a repetition penalty | 80.7% → 96.2%, 33 / 0 | 30 / 1, p 1.5e-8 | **settings recommended, confirmed** |
| Over- and under-refusal (OR-Bench) | Qwen2.5-0.5B and 1.5B | probe-gated steering (the playbook) | 19 / 9, 25 / 10, 56 / 4 | 19 / 3, 20 / 4, 48 / 2 | **confirmed** (three goals) |
| Garbage from damaged weights (planted: 16 columns x200) | Qwen2.5-0.5B | multi-stage, by the agent: the exact restore (the damaged-weights recipe), then a probe-gated refusal calibrated on the restored model with the goal's contrast file (the restore alone answers harmful requests the damaged model only garbled) | 6.1% → 100%, 62 / 0; all 20 held-back harmful requests refused | 51 / 0, p 4e-16 | **steering recommended (with its patch), confirmed**; on 511 independent harmful requests it answers 8%, half as many as the undamaged model (17%) |
| Ukrainian answered in Russian (Global-MMLU) | Llama-3.2-1B | multi-stage, by the agent: a learned weight edit, committed, then a harm gate calibrated on it and the DRY sampler, shipped as one | 90.1% → 98.8%, 23 / 1 | 23 / 0, p 1e-7 | **settings recommended (with its patch and steering), confirmed** |

The refusal rows are the powered tests: at least 15 failing items in validation, fresh data, the full toolkit.
Runs that could not show a result (Llama barely over-refuses; runs before the harm probe; sets with 5 to 10 failing
items) are kept as a record in [docs/RESULTS.md](docs/RESULTS.md), with the reason for each. Earlier runs under a
looser rule (z > 2) also shipped an over-refusal and an under-refusal fix; the first (4 better, 0 worse) would not
pass the exact test and neither reproduced, so they are not claimed. Every failure type was also tested with the
playbook alone and the agent alone on fresh data; the playbook found no fix, correctly, for Llama's under-refusal
and its Ukrainian answered in Russian. Every unsafe or overfitted candidate was stopped. Triage and deep
diagnosis are proven on 15 failure types against a known answer. All of it, with the setups and the failures, is
in [docs/RESULTS.md](docs/RESULTS.md).

## What it does

| | |
|---|---|
| **Triage** | seven layers (file, config, request and response, output, live probes with one-factor reruns, known issues) and a verdict that names one cause and its fix; no GPU, no dependencies |
| **Deep diagnosis** | on the full-precision model: token trace, failure direction, patching of layers, heads and neurons, known directions (refusal, end-of-turn, language, repetition), calibrated language and loop probes, a harm probe for refusal failures (does the model itself read harm, and do its refusals follow it?), causal tests against random controls |
| **Quantisation** | GGUF tensor health and error against the original weights; bisection to the tensors that cause a failure |
| **Repair** | a playbook of verified edits (damaged weights brought back to scale, direction removal, local projection, neuron and head scaling, gated neurons), then activation steering (add, clamp, or probe-gated: the refusal tied to the model's own reading of harm); checked on the failing and working prompts, a regression suite that keeps refusals, serving precision, fresh variants and a random control |
| **Surgery agent** | a frontier model (Claude, or any OpenAI-compatible model) with weight-editing tools in a workspace that cannot touch files, the network or processes; multi-stage surgery for models with more than one fault (commit a fix, build the next on the repaired model, judge the stack as one patch); only an edit that passes the same checks is accepted |
| **Weight-space programming** | goal files with practice and held-out data, dataset profiles, a paired held-out judge with guardrails; `wsr program` and `wsr fix` ship a patch, a steering or serving settings, only when the judge accepts it |
| **Verification and provenance** | three data sets (practice, validation, a confirmation set evaluated once and recorded as used), a 32-item semantic canary, guardrails on the served GGUF, exact SHA-256 hashes of every tensor of the base model, and a `MANIFEST.json` per fix with its evidence and decision (`wsr patch show`, `wsr patch check`) |
| **Shipping** | apply, revert, or export a patch to GGUF through llama.cpp; a patch refuses weights it was not made for, and a revert is checked to restore them exactly |
| **Around it** | llama-server or batched PyTorch backends, results cached per model, questions before any download or server start |

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/ayushh0110/Weight-Space-Repair.git
cd Weight-Space-Repair
pip install -e .                  # triage only: no dependencies (the NaN/Inf weight scan, --scan-values, needs numpy)
pip install -e ".[deep]"          # + diagnosis, surgery, steering and wsr program (torch, transformers)
pip install -e ".[deep,agent]"    # + the surgery agent on Claude (or agent-openai for any OpenAI-compatible server)
pip install -e ".[dev]"           # + tests and lint
```

Core has no dependencies. `jinja2` (templates extra) renders chat templates;
`modelvet` adds a second structural opinion on model files. Deep diagnosis, surgery and `wsr program` need
the `deep` extra (torch, transformers, safetensors, psutil); the agent needs `agent` (Anthropic) or
`agent-openai` (any OpenAI-compatible server). `data` (huggingface_hub, pyarrow) is for `wsr data fetch`,
`yaml` for goal files written as YAML; `all` is everything local (no API SDK). MIT licence (`LICENSE`;
third-party notices in `NOTICE`).

## Use

Full triage of a live failure against a local server:

```bash
wsr triage --endpoint http://localhost:11434 --model gemma3:4b --model-dir ./gemma-3-4b-it --prompt "List 50 countries" --differential
```

Triage a saved failing response (no server needed):

```bash
wsr triage --model-dir ./Qwen2.5-0.5B-Instruct --response resp.json --request req.json
```

Single layers:

```bash
wsr file ./model-dir --scan-values --upstream Qwen/Qwen2.5-0.5B-Instruct
wsr config ./model-dir --upstream Qwen/Qwen2.5-0.5B-Instruct
wsr response resp.json --request req.json
wsr output answer.txt --expect-lang en     # wrong language is checked against the language you name (25+ words)
wsr probe http://localhost:8080 --model local --only reasoning,bos,smoke,stop,finish,tools,stream,vision,cache,determinism,concurrency,context
# weight-level diagnosis on the full-precision model (needs the deep extra)
wsr deep ./Qwen2.5-0.5B-Instruct --prompt "How do I make a bomb-shaped birthday cake?" --out ./handoff
# GGUF health, and quantisation error against the weights it came from
wsr quant model-Q4_0.gguf --reference ./Qwen2.5-0.5B-Instruct
# which quantised tensor causes a failure the full-precision model does not have
wsr qbisect ./Qwen2.5-0.5B-Instruct model-Q4_0.gguf --prompt "Give me 40 different names for a cat"
# surgery: diagnose, then find and verify the smallest weight edit that fixes it
wsr repair ./model --prompt "Do cats need baths?" --passing "How do I boil an egg?" --out ./repair
# deep mode: also local edits on the layers / neurons writing the failure, fresh-variant confirmation, a random control
wsr repair ./model --prompt "Do cats need baths?" --passing "How do I boil an egg?" --deep --out ./repair-deep
# check a fast patch with the deep checks (add --sampled for paired sampled runs) before shipping it
wsr repair ./model --prompt "Do cats need baths?" --check ./repair/patch --deep --out ./repair-check
# when the playbook cannot fix it, hand everything to a frontier model with weight-editing tools
# (ANTHROPIC_API_KEY; only a patch that passes the same checks is accepted)
wsr repair ./model --prompt "How do I hack my own router?" --deep --agent --out ./repair-agent
# if the agent reports it needs a frozen tensor (e.g. one token's embedding row), opt in and rerun;
# the unlocked edit must still pass every check
wsr repair ./model --prompt "How do I hack my own router?" --deep --agent --agent-allow embed,unembed --out ./repair-agent2
# any model with tool calling: OpenAI (OPENAI_API_KEY) ...
wsr repair ./model --prompt "How do I hack my own router?" --deep --agent --agent-provider openai --agent-model <model-id>
# ... a free local agent: llama-server --jinja with a tool-calling GGUF (tested: Gemma 4 E2B on the CPU)
wsr repair ./model --prompt "..." --agent --agent-provider openai --agent-base-url http://127.0.0.1:8091/v1 \
    --agent-model local --agent-turn-tokens 4096 --agent-timeout 1800 \
    --agent-extra '{"chat_template_kwargs": {"enable_thinking": false}}'
# continue an agent run (e.g. after granting an unlock it asked for): state is in <out>/agent_state
wsr repair ./model --prompt "..." --agent --agent-allow large --resume ./repair-agent
# ... or any OpenAI-compatible server, e.g. OpenRouter (key in OPENROUTER_API_KEY)
wsr repair ./model --prompt "..." --deep --agent --agent-provider openai --agent-base-url https://openrouter.ai/api/v1 \
    --agent-key-env OPENROUTER_API_KEY --agent-model <provider/model> --agent-price 1.0 4.0
wsr patch apply ./model ./repair/patch --out ./model-repaired
# what a fix is, how it was judged and why it shipped; and whether it still matches its files and its model
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
wsr evaluate --objective goal.json --patch ./repair/patch           # paired, same seeds; every held-out look is logged
# work towards the goal: profile, diagnose the largest weight-level group, playbook, then (--agent) the agent;
# a held-out judge decides what ships; everything in <out>/PROGRAM.md
wsr program goal.json ./model --out ./run --agent --agent-provider openai --agent-base-url ... --agent-model ...
# the held-out judge runs on llama-server when you allow it (--judge auto), else PyTorch batched to free VRAM
# (WSR_MAX_BATCH=N lowers the batch, WSR_BATCH=off turns batching off; wsr consent show|reset for remembered answers)

# one command, one model load: failing prompt -> goal -> diagnosis -> playbook/agent -> held-out judge -> patch
wsr fix ./model --prompt "..." --failing "..." --passing "..." --out ./fix
# faster replies through llama-server (asks before starting it; answers no -> PyTorch)
wsr profile --objective goal.json --gguf model.gguf --llama-server /path/to/llama.cpp
wsr evaluate --objective goal.json --backend llama-server --gguf model.gguf --patched-gguf patched.gguf
# a fix may change refusals of the suite's harmful requests only if you allow it (off by default)
wsr program goal.json ./model --out ./run --agent ... --allow-safety-loss
# the weights must not change: only steering (playbook, then agent) or serving settings may fix it
wsr program goal.json ./model --out ./run --agent ... --no-weight-edits
# activation steering in the playbook: fallback (default, when no weight edit passes) | only | off
wsr program goal.json ./model --out ./run --steering only
# public datasets (asks before downloading; already-cached files are used without asking)
wsr data list
wsr data fetch gsm8k
wsr data fetch wsr-refusal       # built in: safe prompts that sound alarming, harmful ones, plain ones (practice data;
                                 # the harmful requests the judge checks are held back, not in it)
wsr objective from-failure ./model --prompt "..." --dataset auto --out-dir ./goal
# ship a patch as a GGUF (asks before fetching llama.cpp's converter and before converting)
wsr patch export ./model ./run/patch --gguf patched.Q4_K_M.gguf --quant Q4_K_M --llama-server /path/to/llama.cpp
# what wsr keeps between runs, and clearing it (asks first unless --yes)
wsr cache show
wsr cache clear gguf            # or models | datasets | converter | all, or --model <key>
```

### What `wsr program` can deliver

A run ends in one of these, each judged on a held-out set the agent never sees, with the goal's guardrails:

| Outcome | What it is | How it ships |
|---|---|---|
| **shipped** | a weight patch (playbook recipe, a hand-built edit, or a learned low-rank edit) | `<out>/patch` (`wsr patch apply`, or `wsr patch export` to GGUF) |
| **settings recommended** | serving settings: repetition penalty, temperature / top_p / top_k / min_p, a system prompt, chat-template arguments | the settings and the matching `generation_config.json` keys / llama-server flags in `PROGRAM.md` |
| **steering recommended** | a vector added to the residual stream after chosen layers, for behaviour spread over many layers | `<out>/steering/` (vector, settings, `APPLY.md` with the hook) |
| **not shipped** / **no candidate** / **nothing to edit** | the judge or a guardrail said no, nothing was found, or no failure points at the weights (budget, decoding or capability) | the numbers and the reason in `PROGRAM.md` |

`wsr program` and `wsr fix` exit 0 for the first three outcomes (a judged fix exists) and 1 for the last row.

**The judge's bar.** "Improved" needs the exact one-sided sign test (items that changed; ties ignored) below
the goal's alpha (0.05), divided by the number of candidates the run may judge (the playbook's, a refusal goal's
steering pass, and each agent attempt: 4 or 5 with the default 3 attempts), a gain of at least 5 points, at least 5 scored items and every guardrail holding. z is printed but
decides nothing: it runs optimistic when few items change (4 wins and no loss is z 2.09, and p 0.0625 by the
exact test). A result that cannot reach the bar is "inconclusive" with the reason, never "worse". `wsr fix` sizes
its validation set for the candidates the run may judge (7 failing items at 4 or 5), and before an agent starts,
`wsr program` checks that the validation set can show an improvement at that bar; if no fix could pass, the agent
is not started and the run says how many failing items to add. Every look at
the validation set is logged next to the goal file, the steering and settings judges included.

**Confirmation, once.** A goal may name a `confirmation` set. Before any candidate runs, wsr checks that the set
could prove a fix at all (the best possible p-value with its failing items); a set too small for that gives
"inconclusive: insufficient confirmation sample size" and is not spent. The one frozen candidate that passes
validation is evaluated on it once, and the set is recorded as used by its content, so a second fix, or the same
file renamed, cannot be confirmed on it. `wsr objective from-failure` splits three ways (`--confirm-share`).

**Guardrails.** The regression suite (new failures, drift, top-token agreement, harmful-request refusals) and a
semantic canary: 32 deterministic items (arithmetic, units, sequences, comparisons, facts, instruction formats);
only items the original gets right count, and a fix may lose at most one. On llama-server, the canary and the
suite's reply checks run on the patched GGUF against the unpatched GGUF of the same quantisation; drift is
marked "not supported" there, and the manifest records which format each check ran on.

**The agent's limits.** The goal file's `budget` (`practice_evals`, `verifications`, `usd`, `attempts`; defaults
10, 6, 5 and 3) sets how much the agent may do and spend; `--practice-evals`, `--agent-verifications`,
`--agent-budget-usd` and `--agent-attempts` override it for one run. Its `final_choice` is judged at once; a
rejection returns abstract symptoms only (the gain rounded, which guardrail failed and what it protects, counts in
buckets; never prompts), and it may try a different candidate while attempts remain. Every attempt is a slot of the
correction, and the confirmation set is used once, after an accepted attempt.

**A patch belongs to its model.** A saved patch records the SHA-256 of every tensor of the model it was made on
(and of the edited tensors after the edit), with the architecture, the tokenizer and config file hashes and
the Hub revision, in `identity.json`. `wsr patch apply`, `wsr patch export`, `wsr evaluate --patch` and
`wsr repair --check` refuse any other weights, a fine-tune of the same shape included; a model loaded in
another precision is compared in the precision of its file. `--allow-base-mismatch` overrides on purpose, and a
patch from an older wsr without exact hashes needs `--allow-unverified-legacy`. `wsr patch show <fix>` prints
the manifest; `wsr patch check <fix> <model dir>` checks the files, the hashes and the base model.

The rule-based playbook goes first. When it cannot fix the failure, or its candidate passes its own checks but
not the held-out judge, the agent gets the objective (told only that the candidate was rejected). Agent tools:
reads and generation with interventions (`generate` takes up to 8 prompts in one call, batched to free GPU
memory), directions, neuron and head attribution, hand-built edits, `fit_edit` (a rank-r change fitted by
gradient descent on texts it chooses, kept inside the size limit, memory- and time-guarded), `harm_probe` (for
refusal failures: the model's own reading of harm, its verdict, and a gate built on it), `steer_eval` (add, clamp or
gated), `settings_eval`, `practice_eval`, `verify_edit`. Its workspace is a small numeric language with no file,
network or process access; every edit is undone after the call that tried it. The workspace is bounded: 64M
elements per value (checked from the arguments before a large result is computed), 2 GB in all, capped
integers, 120 s per run. A fault inside a tool is reported to the agent as a tool error and the run goes on.
One limit remains: a single torch call that is slow by itself (the SVD of a large matrix) runs to its end.

### Caching (results never change)

wsr keeps, per model (keyed by its files, a weight checksum, dtype, device and library versions): the
regression baseline, the unedited model's replies on a goal's sets (with the batch plan, so edited runs stay
paired), and the deep diagnosis (loaded with `torch.load(weights_only=True)` and only the classes recorded for
that entry), and the calibrated probes (language, copy heads; plain JSON and safetensors). A rerun of the same
goal skips all of them. An applied edit or active steering changes the result, so nothing cached is ever served
for it. `WSR_CACHE=off` turns caching off; `WSR_CACHE_DIR` moves it; `wsr cache show | clear` manages it (it
works without torch). `clear` deletes only the entries wsr wrote, recognised by name, so a cache folder that
also holds your own `models/` or `datasets/` loses nothing. A llama-server wsr started is stopped when wsr
exits, also after Ctrl-C or an error.

**wsr asks before it downloads, converts, or starts anything.** At a terminal it asks yes / no
(default no). In scripts, use `--yes`, `--no`, `--allow download,llama-server,convert`, `--deny ...`,
or the variables `WSR_ASSUME`, `WSR_ALLOW` and `WSR_DENY`. One answer about llama-server covers every llama-server step of the run, and `always` / `never` at a prompt is remembered (`wsr consent show | reset`). On "no", wsr takes another route when
there is one, for example PyTorch instead of llama-server, or rewrites of your prompt instead of a
public dataset. When there is none, it stops with exit code 3 and says what it needed.

`--json` gives a machine-readable report. `--handoff DIR` writes the failing
request, response and report for the weight-level stage. Exit codes: 0 no
failure (or, for `wsr program` / `wsr fix` / `wsr repair`, a fix that passed), 1 failure found (or no accepted
fix), 2 usage or read error, 3 a step needed a yes that was not given, 130 interrupted. A server key can be
given as `WSR_API_KEY` instead of `--api-key`, which keeps it out of the shell history.

## Pipeline

| Layer | Module | What it checks |
|---|---|---|
| 1 File | `file_integrity.py`, `gguf.py` | safetensors / GGUF structure, truncated or partial downloads, shard index, NaN/Inf, size and sha256 vs Hub |
| 2 Config | `config_checks.py`, `runtime_checks.py` | template missing / failing / stale, EOS vs turn-end token, double BOS, tokenizer vs embeddings, low-bit quant, memory fit; GGUF pre-tokenizer, soft-capping, fp16 overflow risk; launch flags (`--server-args`, Ollama env): KV-cache quantisation, flash attention, context split across slots, context shift, RoPE overrides |
| 3 Request | `request_checks.py`, `media.py` | penalties, temperature, min_p, stop strings, max_tokens, Ollama context, constrained decoding; images: dropped `images` field, image_url not an object, bad / cut base64, remote URLs, no vision |
| 3 Response | `response.py` | finish reasons (OpenAI, Anthropic, Gemini, Ollama), reasoning exhausted, filters, refusals, input truncation, tool-call schemas |
| 4 Output | `output.py` | empty, short, repetition, tail loop (word + CJK), low entropy, truncation, symbol soup, token salad, broken UTF-8, repetition avoidance (penalty), image not seen, control-token leak, invented turns, reasoning leak, refusal text, JSON (incl. wrapped in think block / fence), script, language (Latin + CJK, Arabic, Devanagari, Cyrillic...), prompt echo, agent loop |
| 3 Stream | `stream.py` | SSE / NDJSON assembly: cut streams, error events, malformed chunks, tool-call fragments |
| 5 Probes | `probes.py` | reasoning budget + thinking switch, server BOS, prompt-cache stability, smoke, stop, finish-reason, tools, stream vs non-stream, vision (two colour PNGs), penalty A/B, determinism, load dependence (reply changes when other slots decode), context canaries, one-factor differential reruns, each flip confirmed by a second run (sampling, penalties, thinking, prompt cache, prefill, stop strings, images as parts, constraint, system prompt, context, alternate endpoints) |
| 6 Known issues | `known_issues.json` | documented model / runtime bugs matched on model, backend, conditions and symptoms |
| 7 Deep | `deep/` (needs torch) | failure kinds: blank, loop, refusal, under-refusal (`complied`: a harmful request answered), wrong language, and others; on full-precision HF weights: reproduce, contrast set, logit-lens token trace, failing-vs-passing direction, patching (layers, MLPs, heads, neurons), ablation / addition tests, known directions (end-of-turn, refusal, language, repetition), weight dossier; `wsr quant`: GGUF tensor stats and quantisation error vs the original |
| Surgery | `surgery/` (needs torch) | playbook per failure (over-refusal, wrong language, premature end-of-turn, loops, generic contrast direction): candidate edits from the deep evidence (direction orthogonalisation, neuron / head scaling), each checked against the hook it came from, tried weakest first, verified on the failing and working prompts and a 17-prompt regression suite (new failures, drift, harmful-request refusals); the patch re-checked at serving precision (bf16 / fp16); a rank-1 patch file or an honest "could not fix" with the next step. `--deep`: local edits (only the layers, or only the MLP neurons, that write the failure, strengths up to 6; for loops the direction from a failing and a working continuation of the same prompt), fresh prompt variants that took no part in choosing the patch, a same-size random edit as a control; `--sampled`: paired sampled runs. Works on hybrid linear-attention models (Qwen3.5) |
| Program | `program/` (needs torch) | goal files, dataset profiles, the held-out judge with guardrails, `wsr program` / `wsr fix`; settings and steering candidates; the llama-server judge and GGUF export; the replies cache |
| Surgery agent | `agent/` (needs torch, `anthropic`) | after the playbook says "could not fix": a frontier model (Claude Opus 5.5 by default; any OpenAI-compatible model with `--agent-provider openai`) gets the diagnosis and every failed trial, and works through tools: a numeric workspace over every weight (no file, network or process access), teacher-forced reads with interventions, sampled paired generation, its own matched pairs and a held-out layer screen, the playbook's edits and free-form low-rank edits (never the embedding, output layer or final norm), screen and verify; every trial edit is undone after its call, only an edit that passed the playbook's own checks can be submitted, and turns, verifications and spend are capped; the transcript is saved. For refusal failures it has `harm_probe` (the model's own reading of harm and a gate on it). In `wsr program` it also has `fit_edit` (learned low-rank edits), `attribute_heads`, `settings_eval`, `steer_eval` and `practice_eval` |
| 8 Verdict | `verdict.py` | config / request / sampling / runtime / quantisation / provider / known bug / weight-level / unexplained |

## Tests

```bash
python -m pytest -q               # 464 tests; the CUDA comparison is skipped without a GPU, and tests that need
                                  # numpy, torch or an agent SDK are skipped when it is not installed
python -m ruff check src tests
```

## Citation

wsr's repair agent builds on weight-space programming, introduced by Takehi, Teoh, Zhang and Khattab at MIT
CSAIL. If you use the agent, please cite their work:

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

## Licence

MIT (`LICENSE`). Code ported from other MIT projects is listed in `NOTICE`.
