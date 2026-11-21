# Changelog

All notable changes to wsr are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is 0.x, a minor
release may change commands, file formats or defaults; each change is listed under its release.

The release workflow publishes the section whose heading matches the pushed tag as the GitHub Release notes, so
every release needs its own `## [x.y.z] - YYYY-MM-DD` heading here before it is tagged.

## [Unreleased]

### Added

- **`REASONING_LOOP`**: triage now reads the reasoning channel (`reasoning_content`, Ollama `thinking`, Anthropic
  thinking blocks) for loops, which the answer detectors never saw. It catches a block of up to 40 lines repeating
  and a reasoning that keeps restating lines it wrote just before. It fails only when the loop cost the answer
  (cap reached or no answer), and reports roughly where the loop starts. It is its own symptom group, so causes
  about where the turn ends (stop tokens, EOS, BOS) cannot explain it. The detectors used by `wsr deep`,
  `wsr program` and the agent's judge are unchanged.
- **`wsr deep --thinking`** (and `wsr repair --thinking`): render the chat template with `enable_thinking=true`.
  Without it, a model whose template thinks only when asked (Qwen3.5-0.8B) runs without thinking, so a
  thinking-mode failure cannot reproduce. Both `--thinking` and `--no-thinking` now warn when the template renders
  the same prompt with and without the switch.
- A known issue for Qwen3.5-0.8B's thinking loops, from its model card.
- **`DIFF_INCOMPLETE`**: when the time budget (`--budget`) runs out during the differential, triage says which
  reruns did not finish, instead of dropping them silently.

### Changed

- When the reasoning loops, the `REASONING_EXHAUSTED` fix no longer says to raise `max_tokens` (a loop does not end
  with more tokens); it says to turn thinking off or set a reasoning budget. The verdict adds a `wsr deep --thinking`
  step with a `--max-new` sized from where the loop starts, and says when that is past the 4,096 tokens the repair
  stage can judge.
- The differential runs `thinking_off` first when the reasoning itself failed, and credits it, greedy, sampled,
  neutral penalties and a shorter context with fixing a reasoning loop.
- `SERVER_NO_BOS` is information, not a warning, for Qwen-family models, which use no BOS token by design. Before,
  it could be offered as the cause of a loop or blank reply on every Qwen model served by llama-server.
- The `detector:<group>` grader rejects `reasoning_loop`, which it cannot see.

## [0.1.0] - 2026-11-20

The first release.

### Added

- **Triage** (`wsr triage`, and the single layers `file`, `config`, `response`, `output`, `probe`): seven layers
  (file, config, request and response, output, live probes with one-factor reruns, known issues) and a verdict that
  names one cause and its fix. No GPU and no dependencies.
- **Deep diagnosis** (`wsr deep`): token trace, failure direction, patching of layers, heads and neurons, known
  directions (refusal, end-of-turn, language, repetition), calibrated language and loop probes, the harm probe for
  refusal failures, causal tests against random controls.
- **Quantisation** (`wsr quant`, `wsr qbisect`): GGUF tensor health, error against the original weights, and
  bisection to the tensors that cause a failure.
- **Repair** (`wsr repair`): a playbook of verified weight edits (damaged weights brought back to scale, direction
  removal, local projection, neuron and head scaling, gated neurons), then activation steering (add, clamp, or
  probe-gated); checked on the failing and working prompts, a regression suite that keeps refusals, a semantic
  canary and serving precision.
- **Surgery agent** (`--agent`): Claude or any OpenAI-compatible model with weight-editing tools in a restricted
  numeric workspace, multi-stage surgery (`commit_edit`, `revert_edit`), and `--resume`.
- **Weight-space programming** (`wsr objective`, `profile`, `evaluate`, `program`, `fix`): goal files with
  practice, validation and confirmation sets, a paired held-out judge (exact sign test with a Bonferroni
  correction, a minimum gain, guardrails), and a confirmation set used once.
- **Shipping and provenance** (`wsr patch apply | revert | export | show | check`): exact SHA-256 identity of every
  tensor of the base model, a `MANIFEST.json` per judged fix, export to GGUF through llama.cpp.
- llama-server and batched PyTorch backends, per-model caches (`wsr cache`), and questions before any download,
  conversion or server start (`wsr consent`).
- **Apple Silicon** (`--device mps`, opt-in): before a model goes on MPS, wsr checks the Mac (Apple Silicon,
  macOS 14+, a torch build with MPS), runs a self-test once per chip, macOS and library versions (CPU against MPS
  on a tiny model built in code: logits, repeated greedy runs, batching, bit-exact restores, the CPU fallback for
  missing ops), and checks that the real model's greedy reply repeats; on a failure the run uses the CPU and says
  why. `--device auto` picks CUDA, else the CPU, on every machine. `scripts/check_mps.py` measures a real model on
  a Mac, and an "MPS report" issue form collects the results.
- **On a Mac, without MPS:** triage's memory check uses Metal's share of the unified memory; MLX checkpoints are
  recognised (their quantisation reported, not treated as full precision, a clear message from `wsr deep`); a
  llama-server wsr starts is placed against Metal's share, and Homebrew's `llama-server` is found off PATH;
  memory headroom counts swap only as far as the system disk can hold it.
- Continuous integration: core tests on Linux, Windows and macOS with no optional dependencies, deep tests on CPU
  torch on Linux and macOS.

[Unreleased]: https://github.com/ayushh0110/Weight-Space-Repair/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ayushh0110/Weight-Space-Repair/releases/tag/v0.1.0
