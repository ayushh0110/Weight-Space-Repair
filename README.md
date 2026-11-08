# wsr — triage layer of the Weight-Space Repair CLI

> No model fails silently.

Give it a failing prompt and a model. It finds out what failed and why. Causes
outside the weights (file, config, request, sampling, runtime) get an exact fix;
only a failure that survives all of those is called weight-level and packaged
for the deep-diagnosis stage.

It combines the techniques of llm-output-guard, llm-stopcheck, LLMDoctor,
ModelVet and Raschka's repetition checklist into one layered pipeline, and adds
a verdict engine that links symptoms to causes. See
[docs/DESIGN.md](docs/DESIGN.md) for the technique inventory and coverage table.

## Install

```bash
pip install -e ".[dev]"
```

Core has no dependencies. `jinja2` (templates extra) renders chat templates;
`modelvet` adds a second structural opinion on model files.

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
# weight-level diagnosis on the full-precision model (pip install 'wsr-triage[deep]')
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

# beyond one prompt (docs/WSP_PLAN.md phase 1): a goal file, a dataset profile, a paired judge
wsr objective init goal.json --name my-goal --grader math          # or exact | contains | regex:<p> | detector:<group>
wsr objective split all.jsonl --practice data/practice.jsonl --heldout heldout/heldout.jsonl
wsr objective check goal.json                                      # files, gradable rows, practice/held-out overlap
wsr profile ./model --objective goal.json --limit 50 --out profile.json   # triage verdicts over the data
# from one failure to a goal file: confirm it, find items that fail the same way, stratified split
wsr objective from-failure ./model --prompt "..." --out-dir ./goal-blank
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
# public datasets (asks before downloading; already-cached files are used without asking)
wsr data list
wsr data fetch gsm8k
wsr objective from-failure ./model --prompt "..." --dataset auto --out-dir ./goal
# ship a patch as a GGUF (asks before fetching llama.cpp's converter and before converting)
wsr patch export ./model ./run/patch --gguf patched.Q4_K_M.gguf --quant Q4_K_M --llama-server /path/to/llama.cpp
```

**wsr asks before it downloads, converts, or starts anything.** At a terminal it asks yes / no
(default no). In scripts, use `--yes`, `--no`, `--allow download,llama-server,convert`, `--deny ...`,
or the variables `WSR_ASSUME`, `WSR_ALLOW` and `WSR_DENY`. One answer about llama-server covers every llama-server step of the run, and `always` / `never` at a prompt is remembered (`wsr consent show | reset`). On "no", wsr takes another route when
there is one, for example PyTorch instead of llama-server, or rewrites of your prompt instead of a
public dataset. When there is none, it stops with exit code 3 and says what it needed.

`--json` gives a machine-readable report. `--handoff DIR` writes the failing
request, response and report for the weight-level stage. Exit codes: 0 no
failure, 1 failure found, 2 usage or read error.

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
| 7 Deep | `deep/` (needs torch) | on full-precision HF weights: reproduce, contrast set, logit-lens token trace, failing-vs-passing direction, patching (layers, MLPs, heads, neurons), ablation / addition tests, known directions (end-of-turn, refusal, language, repetition), weight dossier; `wsr quant`: GGUF tensor stats and quantisation error vs the original |
| Surgery | `surgery/` (needs torch) | playbook per failure (over-refusal, wrong language, premature end-of-turn, loops, generic contrast direction): candidate edits from the deep evidence (direction orthogonalisation, neuron / head scaling), each checked against the hook it came from, tried weakest first, verified on the failing and working prompts and a 17-prompt regression suite (new failures, drift, harmful-request refusals); the patch re-checked at serving precision (bf16 / fp16); a rank-1 patch file or an honest "could not fix" with the next step. `--deep`: local edits (only the layers, or only the MLP neurons, that write the failure, strengths up to 6; for loops the direction from a failing and a working continuation of the same prompt), fresh prompt variants that took no part in choosing the patch, a same-size random edit as a control; `--sampled`: paired sampled runs. Works on hybrid linear-attention models (Qwen3.5) |
| Surgery agent | `agent/` (needs torch, `anthropic`) | after the playbook says "could not fix": a frontier model (Claude Opus 5.5 by default; any OpenAI-compatible model with `--agent-provider openai`) gets the diagnosis and every failed trial, and works through tools: a numeric workspace over every weight (no file, network or process access), teacher-forced reads with interventions, sampled paired generation, its own matched pairs and a held-out layer screen, the playbook's edits and free-form low-rank edits (never the embedding, output layer or final norm), screen and verify; every trial edit is undone after its call, only an edit that passed the playbook's own checks can be submitted, and turns, verifications and spend are capped; the transcript is saved |
| 8 Verdict | `verdict.py` | config / request / sampling / runtime / quantisation / provider / known bug / weight-level / unexplained |

## Tests

```bash
python -m pytest -q
```
