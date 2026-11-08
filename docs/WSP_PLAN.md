# Plan: wsr as a full weight-space programming tool

Status: revised 2026-11-07 after review (see "Review of the suggestions" at the end). Builds on
DESIGN.md sections 5-7.

Progress:
- Phase 0: the refused-operations log and `--agent-only` are built. The free local smoke test
  ran 4 times (Gemma 4 E2B via llama-server). Runs 1-3 exposed and fixed a crash on provider
  errors, runaway thinking and unclear tool errors; run 4 fixed the planted Llama bug in 7 turns at
  no cost (DESIGN.md section 6). Quality runs with a frontier model are still to do.
- Phase 1.6: state, `--resume` and drafts tried first are built and tested.
- Phase 1 is built (transformers backend; DESIGN.md section 8):
  - goal files (`wsr objective init | check | split`);
  - graders: exact, contains, regex, math, detector;
  - `wsr profile`, with causes from tests that ran;
  - `wsr evaluate`: paired, same seeds, guardrails, held-out looks logged.

  Still to do: the llama-server and vLLM backends, and the ifeval and judge graders.
- Phase 1.5 is built without downloads: `wsr objective from-failure` from a prompt, a local dataset or a profile
  group, with a hit rate, a fallback to rewrites and a stratified split. `--dataset auto` names the
  public source but does not fetch it.
- Phase 0 quality runs have started on OpenRouter's free tier (Nemotron 3 Ultra); results are in DESIGN.md section 6.
- Asking first is built (`--yes / --no / --allow / --deny`, `WSR_*`, prompts at a terminal). The
  llama-server backend is built (profile, evaluate, from-failure; about 16× faster here), and so is the
  dataset registry (`wsr data`).
- Phase 3 (`wsr program`) is built and tested:
  - profile, group, diagnosis, playbook;
  - the agent with practice_eval / checkpoint / final_choice;
  - the held-out judge;
  - `PROGRAM.md`, which is phase 5's report.
- Phase 4 (`wsr patch export --gguf`) is built and was run end to end. A shipped patch survives
  Q8_0 and Q4_K_M on the served file (blank replies 53 % → 0 % on both). Exporting also found and
  fixed a tied-weight save bug.
- Phase 2 stays deferred: the refused-operations log had only imports, now allowed as no-ops.
- Still a plan: a live `wsr program` run with a frontier agent on a hard objective.

## The idea in one picture

wsr already answers the question the paper's agent spent its first turns on: *what goes wrong,
and is it in the weights at all?* The plan keeps that as the front door. It adds three things
behind it:
- objectives, including open-ended ones;
- an agent whose state survives restarts and reruns;
- an evaluation harness that scores the result on data the agent never saw.

Full Python comes last, once real runs show what the agent needs.

```
            served model / HF dir / GGUF
                        |
   wsr triage  --> not the weights? (request, template, sampling, quantisation, runtime)
                        |    -> report the fix, stop
   wsr deep    --> reproduces at full precision, localised          (per prompt)
   wsr profile --> the same verdicts over a dataset                 (NEW: per task)
                        |
   wsr objective from-failure / init  --> goal file                 (NEW)
                        |
   wsr program --> 1. playbook recipes (rule-based, as today)
                   2. agent: safe workspace (later: isolated full Python) + forward-pass tools
                      state saved: --resume continues a run, restarts keep everything
                        |
   wsr evaluate --> paired, sampled, long-reply scoring on the held-out set and guardrails (NEW)
                        |
   verified patch --> wsr patch apply / export --gguf --> re-triage the served model
```

`wsr repair` stays as it is. It becomes the special case "a repair objective built from one
prompt", and keeps its current checks.

## Why open-ended goals are possible

They were never blocked by the editing, only by the judge. Today a patch is judged on the failing
prompts, a 17-prompt regression suite and a few variants: enough to say "this failure is gone and
nothing broke", not enough to say "the model got better at X". An open-ended goal needs three
things wsr does not have yet:
1. a task with a metric;
2. data split so the agent never sees the part it is scored on;
3. an evaluation that is long and sampled enough to measure it.

Phases 1-1.5 and 3 add those. Triage is the advantage over the paper's setup. The paper's editor
characterised the failure by hand ("85 of 96 answers hit the token cap"); `wsr profile` does that
automatically with the same detectors and tests as triage, so the agent starts with a failure
profile, not a blank page.

## Phase 0: live agent runs (no new code)

- **Free smoke test, through llama-server.** It checks the plumbing. The OpenAI-compatible
  provider already talks to a local llama-server started with `--jinja`:

  ```
  wsr repair <model> ... --agent --agent-provider openai --agent-base-url http://127.0.0.1:8080/v1 --agent-model local
  ```

  Limits: the agent model needs a chat template with tool calls (Qwen 2.5/3, Llama 3.x, Gemma 4,
  Mistral). On the 4 GB GPU the agent model and the model under repair cannot both sit in VRAM, so
  the agent runs on the CPU (`-ngl 0`) or the repaired model does. A 2-8B agent will mostly give
  up or call tools wrongly. That is still a useful test of the loop, the guards and the transcript,
  but not of the method.
- **Quality runs, with a frontier model** (any supported provider, or a free tier such as
  OpenRouter's free models or Gemini's OpenAI endpoint). The cases:
  - Llama's Ukrainian loop (expected answer: give up, it is decoding);
  - the Qwen router case (can it find the gate itself?);
  - the planted Qwen loop with the playbook disabled.
- **Log what the safe workspace refused.** Every refused attribute or call is counted per run
  (`checks.agent.denied`). Over 5-10 runs this shows which operations the agent actually needs,
  which decides phase 2.

Acceptance:
- the smoke test runs end to end with no crash and every guard holding;
- at least one fix found by the agent, not the playbook, passes the deep checks;
- the transcripts show no attempt to work around a guard.

## Phase 1: objectives, profiles and the evaluation harness

**Objective file** (`wsr objective init`, YAML or JSON):

```yaml
name: qwen35-math-loops
kind: capability            # repair | capability
model: ./Qwen3.5-0.8B
protocol:                   # how the model is served; from triage when available
  template_kwargs: {enable_thinking: true}
  sampling: {temperature: 0.6, top_p: 0.95, top_k: 20, presence_penalty: 1.5}
  max_new_tokens: 4096
practice: data/math_train.jsonl        # the agent may read these
heldout:  data/math500.jsonl           # never where the agent can read it
grader: math_verify                    # exact | regex | math_verify | ifeval | detector:<group> | judge:<model>
guardrails:                            # must not get worse beyond the margin
  - {suite: wsr_regression}            # today's 17 prompts, safety included
  - {data: data/mmlu_pro_200.jsonl, grader: exact, max_drop: 0.02}
  - {data: data/ifeval_100.jsonl, grader: ifeval, max_drop: 0.03}
budget: {passes: 50, verifications: 10, usd: 20}
```

**`wsr profile <model> --data practice.jsonl`** prints the same report as triage, with the same
renderer, finding codes and verdict wording, for a dataset instead of one request:

```
PROFILE: gsm8k, 200 samples, Qwen3.5-0.8B (thinking on, temp 0.6, max 4096)

  38%  TRUNCATED, looping tail  (76)   -> weight-level loop (DEEP_VERDICT: MLP, layers 12-22)   [deep ran on 6]
   9%  TRUNCATED, still working  (18)  -> budget: raise max_new_tokens (reply was productive)
  12%  WRONG_ANSWER              (24)  -> capability: no single failure mode (see examples)
   4%  EMPTY                      (8)  -> premature end-of-turn at full precision (weight-level candidate)
   3%  LOOP, short reply          (6)  -> decoding: greedy keeps it, sampled escapes 3/3
  34%  correct                   (68)

  Next: wsr objective from-failure --profile gsm8k_profile.json --group TRUNCATED_LOOP
```

- The cause on each line comes from tests that ran, not from a fixed table. A cut reply is split
  into a looping tail (a weight candidate) and still working (a budget problem). EMPTY in an
  offline run means a premature end-of-turn; it is not the server-side STOP_PROBE_RUNON. The
  decoding verdict needs the sampled-escape test. Each weight-level group gets a `wsr deep` on a
  few members.
- The rates come with confidence intervals and three examples per group, and the JSON goes into
  the agent's brief.

**`wsr evaluate <model> [--patch P] --objective obj.yaml`** (the harness):
- backends: transformers (small models), llama-server through triage's own client (GGUF, long
  replies, fast on this GPU), and vLLM when present;
- paired runs: original and patched on the same seeds, with the accuracy difference, its standard
  error, wins / losses, the per-group failure rates from the profile, and the guardrails;
- pass budget: one pass = one batch, counted like the paper's forward passes;
- the held-out set is consulted by the harness only, and the number of consultations is reported.

Acceptance: on Qwen3.5-0.8B with the paper's protocol (shorter cap on this GPU), `wsr profile`
reproduces "most answers hit the cap". The paper's released p50c edit, rebuilt as a patch (done in
experiment 1), must score better on the held-out subset than the unedited model, with a lower
looping-tail share. This positive control tests the harness against a known answer.

## Phase 1.5: `wsr objective from-failure` (from "my model does X wrong" to a scored benchmark)

```
wsr objective from-failure --model ./model --prompt "..." [--dataset auto|gsm8k|ifeval|...] [--samples 200]
wsr objective from-failure --profile profile.json --group TRUNCATED_LOOP
wsr objective from-failure --triage report.json
```

`from-triage` is the same command with a triage report as input, so there is one command, not two.
It works in five steps:
1. **Confirm the failure type** with the deep diagnosis (blank, loop, refusal, language, format),
   including "is it weight-level?". If the answer is decoding or runtime, it stops with that fix,
   as `repair` does.
2. **Pick data where that failure can occur.** `auto` maps the failure to a matching source:

   | Failure | Source |
   |---|---|
   | loop or cut-off reasoning | GSM8K / MATH |
   | over-refusal | XSTest / OR-Bench style safe-but-scary prompts, plus harmful prompts that must stay refused |
   | wrong language | MGSM / Aya in the user's language |
   | format | IFEval |
   | blank | a general mix |

   A dataset the user names wins. The first use of each public dataset asks before downloading.
3. **Measure the hit rate.** Run the samples and keep the ones that show the same failure. If the
   hit rate is below 5 %, the failure does not live in that dataset. The command then builds the
   set from wsr's own prompt variants (`fresh_variants` plus paraphrases) and says so, instead of a
   benchmark that does not test the problem.
4. **Split, stratified.** Failing and working items both go into practice and held-out, so
   "working stays working" is measured too. The held-out file is written where the agent cannot
   read it.
5. **Write the goal file**, with the grader set to the failure's detector (for example
   `detector:loop`) plus the task grader when the dataset has answers. Guardrails come from the
   failure kind: harmful-prompt refusals for over-refusal, and so on.

## Phase 1.6: agent state, `--resume` and drafts that come back

One mechanism serves both reruns and the long-run restarts of phase 3.

- **State, saved to `<out>/agent_state/` after every turn:**
  - the workspace variables (tensors in safetensors, the rest as JSON; no pickle);
  - the registries: directions, edits (as patches), pair sets, verified results, checkpoints;
  - notes, capability requests and proposals;
  - the budget used;
  - the transcript so far;
  - a fingerprint of the model (hash of config and weights index) and the wsr version.
- **`wsr repair ... --resume <out dir>`** (and `wsr program ... --resume`):
  - refuses if the model fingerprint differs;
  - warns if the requested unlock is still not granted;
  - starts a new conversation with a state report (what was tried, a results table, the best
    checkpoint, the notes, the open requests);
  - keeps every id valid;
  - tries pending proposals first: a proposal the new unlocks now allow is built as a normal edit
    and verified before anything else, so "I need embed access" becomes a step the run picks up
    again.
- **REPAIR.md** gives the exact rerun line:
  `Rerun with: wsr repair <same args> --agent-allow embed,unembed --resume <out>`.
- Budgets start fresh on resume, and the report shows spend per run and in total.

## Phase 2: the isolated kernel (deferred until phase 0 has 5-10 runs)

Full Python is the riskiest item, so it waits for evidence. The refused-operations log from
phase 0 decides it:
- If the refusals are few and specific (a missing torch function, a numpy call), widen the safe
  workspace's whitelist and skip the container.
- If the agent needs things the subset cannot express (its own helper functions, scipy
  optimisers, sklearn probes), build the container. The design for then:
  - **`wsr sandbox init [--runtime docker|podman|wsl]`** writes the environment. For Docker or
    Podman: a Dockerfile (python, torch CPU, numpy, scipy, scikit-learn, pandas, sympy,
    safetensors, tokenizers) and the run line `--network none --read-only --cap-drop ALL
    --security-opt no-new-privileges --pids-limit 256 --memory 12g --cpus 4 --user 1000`, with
    `/model` and `/practice` read-only and a writable `/work`. For WSL: the kernel under
    `unshare --net --map-root-user` with read-only bind mounts, labelled as weaker than a
    container. The user installs the runtime when this phase starts.
  - **`wsr sandbox doctor`** probes the environment and blocks the agent if anything leaks:
    network, a writable `/model`, host home / SSH / cloud credentials, the held-out data, root,
    no resource limits.
  - **The kernel**: a persistent Python process over stdin / stdout, with `NAMES`, `load`,
    `load_original`, `save_edit`, `drop_edit`, `tok`; autograd disabled (a scope rule; the
    container is the safety boundary); every block logged.
  - **Forward passes stay on the host.** `read`, `generate`, `screen`, `verify` and `evaluate` run
    in wsr's trusted process.
  - **Edits cross as data only.** The host checks names, shapes, frozen tensors, NaN, rescaling
    and size, then builds a `Patch`, with a new sparse column / row delta form so a 64-column edit
    stays small.
  - `--agent-kernel safe|docker|podman|wsl`, with `safe` the default and fallback.

## Phase 3: `wsr program` (open-ended runs)

- `wsr program obj.yaml --agent-provider ... [--agent-kernel ...]` works in this order:
  1. profile the task;
  2. try the playbook on the top weight-level group;
  3. hand over to the agent with the objective, profile, playbook trials, guardrails and budget.
- New agent tools, all on the host:
  - `practice_eval(edit, n)`;
  - `paired_generate(edit, items, n)` (the paper's "both" generation);
  - `teacher_forced(texts)` (up to 1,024 texts);
  - `checkpoint(edit, note)`;
  - `final_choice(edit)`.

  The held-out set is not a tool.
- **Restarts (decided now).** The paper also restarted: its harness opened a new conversation
  every 40 turns or after a harness restart (p50c had 4), carrying the task, the notebook and the
  current edit, while the Python kernel kept its variables. wsr does the same with phase 1.6's
  state:
  - **Nothing is lost.** The workspace, the registries and every checkpoint survive a restart;
    only the conversation is new. The state report replaces the old turns.
  - **"Current edit" is a leaderboard, not one active patch.** Edits are never left applied
    between calls anyway. Every checkpoint has its practice score, its selectivity and its checks.
    The new conversation starts from the best checkpoint as its reference (option A), with the
    rest still there to go back to.
  - **Exploration stays open.** The state report lists the directions tried and dropped, so the
    agent can branch away from the best (option B's freedom) without repeating work.
    `--restart-from fresh` gives pure option B for users who want it.
- **The judge.** After `final_choice`, `wsr evaluate` scores the frozen edit on the held-out set
  and the guardrails. It ships only if the gain is above 2 standard errors and every guardrail
  holds. Otherwise the report says "no improvement" with the numbers, and the patch is kept as
  unverified.
- `wsr repair` = `wsr program` with a repair objective from one prompt and today's checks.

Acceptance: a live run on Qwen3.5-0.8B with a MATH objective (reduced budget) produces an edit that
beats the unedited model on the held-out subset with the guardrails intact. The edit is compared
with the paper's p50c (site overlap, selectivity, score).

## Phase 4: shipping into served formats

- `wsr patch export --gguf` applies a patch to a GGUF, by conversion or by rewriting and
  requantising the touched tensors.
- The served file is then re-triaged against the same objective on llama-server. The report says
  how much of the gain survives Q8_0 and Q4_K_M.

Needs the `gguf` package and llama.cpp's converter, downloaded with the user's OK.

## Phase 5: one report

`REPORT.md` per run, covering:
- the triage verdict;
- the profile;
- the objective;
- what the playbook tried;
- the agent's notes, requests and resumes;
- the evaluation tables;
- the patch, with how to apply and revert it.

## What each phase needs from the user

| Phase | Needs |
|---|---|
| 0 | nothing for the free smoke test (llama-server and a tool-calling GGUF); an API key in the environment for quality runs |
| 1, 1.5 | OK to download public datasets the first time (GSM8K, MATH, MMLU-Pro, IFEval, XSTest, MGSM) |
| 1.6 | nothing |
| 2 | Docker Desktop, Podman or a WSL distro, installed when this phase starts |
| 3 | an API key and a spend budget |
| 4 | OK to download `gguf` and llama.cpp's converter |

## Limits that stay

- **This GPU** (GTX 1650, 4 GB) holds models up to about 1.5B for editing. A local agent and the
  repaired model compete for it. Long sampled evaluation runs best through llama-server on GGUF.
- **Only verified gains ship.** A held-out set and guardrails reduce overfitting to the practice
  data; they do not remove it, and the report says how often the held-out set was consulted.
- **The agent's skill is the model's.** wsr guarantees the checks, not that a fix is found.

## Order

1. Phase 0 smoke test, plus the refused-operations log (a few lines).
2. Phase 1.6, state and resume: small, and both phase 3 and the capability-request loop need it.
3. Phase 1, then phase 1.5.
4. Phase 0 quality runs, as soon as a key is available.
5. Phase 3.
6. Phase 2, if the runs call for it.
7. Phase 4, alongside.

## Review of the suggestions (2026-11-07)

| Suggestion | Verdict | Change made |
|---|---|---|
| Live agent for free through llama-server | yes | Phase 0 smoke test. The provider already supports it. Caveats: tool-call templates, VRAM sharing, and a small agent tests the loop, not the method |
| `wsr objective from-failure` | yes, with fixes | It is the same command as from-triage. The dataset is matched to the failure kind. The hit rate is measured, with a fallback to the model's own prompt variants when the failure is not in the dataset. The split is stratified so working items are measured too. Downloads ask first |
| `wsr profile` prints a triage-style verdict | yes, with fixes | It uses the same renderer and codes. Each cause comes from a test that ran, not a fixed mapping: a cut reply is split into looping (weight candidate) and productive (budget), and an offline EMPTY is a premature end-of-turn, not the server's STOP_PROBE_RUNON |
| Capability requests resume with `--resume`, draft first | yes, made general | Phase 1.6: full state, a model fingerprint check, and pending proposals built and verified first once the unlock is granted |
| Decide the restart design now | yes, decided | The paper did restart (4 conversations in p50c) with the notebook and current edit while the kernel kept its variables. wsr keeps all state; only the conversation is new; the best checkpoint is the reference; the alternatives stay available; `--restart-from fresh` is optional |
| Hold full Python until 5-10 real runs | agreed | Phase 2 is deferred and gated on a refused-operations log from phase 0, which also says exactly which operations to allow |
