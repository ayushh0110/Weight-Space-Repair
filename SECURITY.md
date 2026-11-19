# Security policy

## Reporting a vulnerability

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/ayushh0110/Weight-Space-Repair/security/advisories/new),
not in a public issue. Include what you ran, what happened, and a minimal reproduction if you have one.

## What is in scope

- **The agent's workspace** (`src/wsr/agent/lab.py`): a restricted numeric language the surgery agent runs. It is
  meant to have no file, network or process access and bounded memory and time. Any way for workspace code to
  reach Python, the file system, the network or a process, or to bypass its size and time limits, is a
  vulnerability. Note the documented limit: it runs inside wsr's own process, not an operating-system sandbox
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md)).
- **Loading untrusted files**: a crafted GGUF, safetensors file, cache entry or patch that makes wsr run code,
  write outside its output and cache folders, or exhaust memory before its checks run.
- **Patch identity**: a way to make `wsr patch apply` or `export` accept weights other than the ones a patch was
  made for without `--allow-base-mismatch`.
- **Consent**: a way for wsr to download, convert or start a server without the user's yes.

## Not a vulnerability

- A model's replies under a fix that the judge accepted (that is a result, and belongs in a normal issue).
- Removing refusals on purpose with `--allow-safety-loss`, which exists for that.

## Supported versions

Only the latest release gets fixes while wsr is at 0.x.
