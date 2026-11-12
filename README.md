# wsr — Weight-Space Repair

> No model fails silently.

Give it a failing prompt and an open-weight model. It finds out what failed and why. Causes
outside the weights (file, config, request, sampling, runtime) get an exact fix. A failure
that survives all of those is diagnosed inside the network at full precision, and wsr looks
for the smallest fix that passes its checks: a weight patch, an activation steering, or
serving settings. A rule-based playbook goes first; a frontier model with weight-editing
tools can take over when it fails. Nothing ships without a paired judge on data the fix never saw (an exact
sign test and a minimum gain), a regression suite and a semantic canary; refusals of harmful requests are kept
by default. Every judged fix carries a `MANIFEST.json` that says what it was made for, how it was judged and
why it shipped or not, and a patch refuses any weights other than the exact ones it was made on.

The triage layer combines the techniques of llm-output-guard, llm-stopcheck, LLMDoctor,
ModelVet and Raschka's repetition checklist into one layered pipeline, with a verdict engine
that links symptoms to causes.

**Documentation:** [docs/GUIDE.md](docs/GUIDE.md) covers every command, check, output file and option;
[docs/RESULTS.md](docs/RESULTS.md) every result measured so far, with its control;
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) what is not tested or not built; [docs/DESIGN.md](docs/DESIGN.md)
how each part was built.

## Results so far

Results from `wsr program` under the current rules: the original and the fix answer the same validation items
with the same seeds; a fix is accepted only when the exact one-sided sign test is below 0.05 (divided by the
number of candidates the run may judge), the gain is at least 5 points and every guardrail holds; then one
evaluation on a confirmation set, used once, decides whether it ships.

| Goal | Model | Fix | Validation | Confirmation | Decision |
|---|---|---|---|---|---|
| Blank replies (a bug planted in real weights) | Qwen2.5-0.5B | sparse projection on the neurons wsr found blind | 35.7% → 100%, 18 better / 0 worse, p 0.000004 | 15 / 0, p 0.00003 | **shipped, confirmed** |
| The same, weights not allowed to change | Qwen2.5-0.5B | clamp steering after layer 14 | 6 / 0, p 0.016 | 5 / 0, p 0.031 | **steering recommended, confirmed** |
| The same bug, a smaller goal | Qwen2.5-0.5B | zero the planted neuron | 6 / 0, p 0.016 | 4 / 0, p 0.0625 | inconclusive: confirmation one item short |
| Over-refusal of safe prompts (XSTest) | Llama-3.2-1B | a learned edit by the agent (Opus 5.5) | 2 / 0, every refusal kept | not used | inconclusive: too few fixed |
| Under-refusal of harmful prompts (XSTest) | Qwen2.5-0.5B | a learned edit by the agent (Opus 5.5) | 88% → 96%, 6 / 0 | not used | rejected: the canary lost 3 answers |
| The blank-reply patch on the served GGUF | Qwen2.5-0.5B | exported to Q8_0, judged on llama-server | 6 / 0; canary and suite hold on the GGUF | (already used) | the patch survives quantisation |

Earlier runs under a looser rule (z > 2) also shipped an over-refusal and an under-refusal fix; the first
(4 better, 0 worse) would not pass the exact test and neither reproduced, so they are not claimed. Every unsafe
or overfitted candidate was stopped. Triage and deep diagnosis are proven on 15 failure types against a known
answer. All of it, with the setups and the failures, is in [docs/RESULTS.md](docs/RESULTS.md).

## What it does

| | |
|---|---|
| **Triage** | seven layers (file, config, request and response, output, live probes with one-factor reruns, known issues) and a verdict that names one cause and its fix; no GPU, no dependencies |
| **Deep diagnosis** | on the full-precision model: token trace, failure direction, patching of layers, heads and neurons, known directions (refusal, end-of-turn, language, repetition), calibrated language and loop probes, causal tests against random controls |
| **Quantisation** | GGUF tensor health and error against the original weights; bisection to the tensors that cause a failure |
| **Repair** | a playbook of verified edits (direction removal, local projection, neuron and head scaling, gated neurons), then activation steering (add or clamp); checked on the failing and working prompts, a regression suite that keeps refusals, serving precision, fresh variants and a random control |
| **Surgery agent** | a frontier model (Claude, or any OpenAI-compatible model) with weight-editing tools in a workspace that cannot touch files, the network or processes; only an edit that passes the same checks is accepted |
| **Weight-space programming** | goal files with practice and held-out data, dataset profiles, a paired held-out judge with guardrails; `wsr program` and `wsr fix` ship a patch, a steering or serving settings, only when the judge accepts it |
| **Verification and provenance** | three data sets (practice, validation, a confirmation set evaluated once and recorded as used), a 32-item semantic canary, guardrails on the served GGUF, exact SHA-256 hashes of every tensor of the base model, and a `MANIFEST.json` per fix with its evidence and decision (`wsr patch show`, `wsr patch check`) |
| **Shipping** | apply, revert, or export a patch to GGUF through llama.cpp; a patch refuses weights it was not made for, and a revert is checked to restore them exactly |
| **Around it** | llama-server or batched PyTorch backends, results cached per model, questions before any download or server start |

**Status: beta.** Deep diagnosis is tested on models up to 1.5B parameters, repair and `wsr program`
up to 1B (Qwen2.5-0.5B and 1.5B, Llama-3.2-1B, Qwen3.5-0.8B), all on a 4 GB GPU. Triage needs no GPU.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/ayushh0110/Weight-Space-Repair.git
cd Weight-Space-Repair
pip install -e .                  # triage only: no dependencies
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
wsr output answer.txt --expect-lang en
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
the goal's alpha (0.05), divided by the number of candidates the run may judge (2 when an agent may follow the
playbook), a gain of at least 5 points, at least 5 scored items and every guardrail holding. z is printed but
decides nothing: it runs optimistic when few items change (4 wins and no loss is z 2.09, and p 0.0625 by the
exact test). A result that cannot reach the bar is "inconclusive" with the reason, never "worse". Every look at
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

**The agent's limits.** The goal file's `budget` (`practice_evals`, `verifications`, `usd`; defaults 10, 6
and 5) sets how much the agent may do and spend; `--practice-evals`, `--agent-verifications` and
`--agent-budget-usd` override it for one run.

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
gradient descent on texts it chooses, kept inside the size limit, memory- and time-guarded), `steer_eval`,
`settings_eval`, `practice_eval`, `verify_edit`. Its workspace is a small numeric language with no file,
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
| Surgery agent | `agent/` (needs torch, `anthropic`) | after the playbook says "could not fix": a frontier model (Claude Opus 5.5 by default; any OpenAI-compatible model with `--agent-provider openai`) gets the diagnosis and every failed trial, and works through tools: a numeric workspace over every weight (no file, network or process access), teacher-forced reads with interventions, sampled paired generation, its own matched pairs and a held-out layer screen, the playbook's edits and free-form low-rank edits (never the embedding, output layer or final norm), screen and verify; every trial edit is undone after its call, only an edit that passed the playbook's own checks can be submitted, and turns, verifications and spend are capped; the transcript is saved. In `wsr program` it also has `fit_edit` (learned low-rank edits), `attribute_heads`, `settings_eval`, `steer_eval` and `practice_eval` |
| 8 Verdict | `verdict.py` | config / request / sampling / runtime / quantisation / provider / known bug / weight-level / unexplained |

## Tests

```bash
python -m pytest -q               # 363 tests; the CUDA comparison is skipped without a GPU
python -m ruff check src tests
```

## Licence

MIT (`LICENSE`). Code ported from other MIT projects is listed in `NOTICE`.
