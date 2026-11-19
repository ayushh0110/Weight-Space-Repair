# Contributing to wsr

Thanks for helping. The most useful contributions right now are reports from setups wsr has not been tested on,
and fixes that come with a test.

## Reports that help most

- **A verdict that was wrong, or a failure wsr missed.** Open an issue with the "Wrong verdict" form and attach the
  output of `wsr triage ... --json` (remove anything private first). A saved request and response
  (`--handoff DIR` writes them) let anyone reproduce it without your server.
- **Runs on models or servers wsr has not been tested on**: anything above 1.5B parameters, Gemma, Mistral, vLLM,
  LM Studio, Ollama's native API ([docs/LIMITATIONS.md](docs/LIMITATIONS.md) lists them). Say what you ran, on what
  hardware, and what came out, including runs that found nothing.
- **A Mac with Apple Silicon**: `python scripts/check_mps.py` (and with a small model folder, if you have one),
  posted with the "MPS report" form. A few reports decide when `--device auto` may use the Mac's GPU.
- **Agent runs with other models**, local ones especially: which model, which goal, the outcome and the
  `agent_transcript.json` if you can share it.

## Working on the code

```bash
git clone https://github.com/ayushh0110/Weight-Space-Repair.git
cd Weight-Space-Repair
pip install -e ".[dev]"                 # triage and the core tests (no torch)
pip install -e ".[dev,deep,yaml]"       # + diagnosis, surgery and wsr program tests (CPU torch is enough)
python -m pytest -q
python -m ruff check src tests
```

Tests that need torch, numpy or an agent SDK skip themselves when it is not installed. The deep tests build tiny
models themselves; no download is needed.

- **Every behaviour change comes with a test.** A bug fix gets a test that fails before the fix.
- **Triage stays dependency-free.** Code under `src/wsr/` outside `deep/`, `surgery/`, `program/` and `agent/`
  imports only the standard library at module level.
- **Claims match measurements.** A change that affects a number in [docs/RESULTS.md](docs/RESULTS.md) updates it,
  with the setup and the control, or says that the result was not rerun.
- **The command reference is generated.** After changing a command's options, run `python scripts/make_guide.py`
  (`tests/test_docs.py` checks it is up to date, and that every option has help text).
- Lines are wrapped at 120 characters (`ruff` checks the rest).

For a larger change (a new command, a new edit type, a change to how the judge decides), open an issue first to
agree on the design. The judge's rules in particular change only with evidence, since every result depends on
them.

## Security

Please do not report security problems in public issues; see [SECURITY.md](SECURITY.md).

## Licence

By contributing you agree that your contribution is released under the MIT licence of this repository.
