"""Ask before doing anything the user did not plainly ask for.

Downloading a dataset or a package, starting a local server, building an isolated environment:
wsr asks first, and a "no" is final for the run. The caller then falls back when it can (for example
PyTorch instead of llama-server, or rewrites of the prompt instead of a public dataset) and says so.
When nothing else works, it stops with `NeedsConsent`, which names what was needed and how to allow it.

How an answer is given:
* at a terminal, a yes / no question (default no);
* `--yes` answers yes to everything, `--no` answers no to everything;
* `--allow download,llama-server` answers yes to those kinds only (`--deny` answers no to those kinds);
* the environment variables `WSR_ASSUME=yes|no`, `WSR_ALLOW=...` and `WSR_DENY=...` do the same for
  scripts.

Without a terminal and without any of these, the answer is no, and the message says how to allow it.

At the prompt, `always` / `never` also remember the answer for that kind in
`~/.cache/wsr/consent.json` (`wsr consent show`, `wsr consent reset`), so a user who said "never" to
llama-server is not asked again and wsr takes the other route. Flags and variables still win over a
remembered answer.
Every question and answer is recorded (`decisions`) and goes into the report.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, TextIO

KINDS = {
    "download": "download a dataset, model file or package",
    "llama-server": "start a local llama-server (uses the GPU / CPU while it runs)",
    "convert": "convert a model to GGUF with llama.cpp's converter",
    "sandbox": "build or start an isolated environment (container / WSL)",
    "network": "send requests to a remote service",
}


# One answer covers every question of these kinds in a run: a user who uses (or refuses) llama-server
# for one step does the same for the others. Downloads and conversions are asked per file.
PER_KIND = {"llama-server", "sandbox", "network"}


class NeedsConsent(ValueError):
    """A step that cannot go on without a yes. Exit code 3 from the CLI."""

    def __init__(self, kind: str, what: str, why: str = ""):
        self.kind, self.what, self.why = kind, what, why
        msg = f"cannot proceed: this needs {what}"
        if why:
            msg += f" ({why})"
        msg += f"; allow it by answering yes at the prompt, or rerun with --allow {kind} (or --yes)"
        super().__init__(msg)


def memory_path() -> Path:
    return Path(os.environ.get("WSR_CACHE_DIR") or Path.home() / ".cache" / "wsr") / "consent.json"


def remembered() -> dict[str, bool]:
    try:
        d = json.loads(memory_path().read_text(encoding="utf-8"))
        return {k: bool(v) for k, v in d.items() if k in KINDS}
    except (OSError, ValueError):
        return {}


def remember(kind: str, allowed: Optional[bool]) -> None:
    d = remembered()
    if allowed is None:
        d.pop(kind, None)
    else:
        d[kind] = allowed
    p = memory_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def _split(s: Optional[str]) -> set[str]:
    return {x.strip() for x in (s or "").split(",") if x.strip()}


@dataclass
class Consent:
    assume: Optional[str] = None                 # "yes" | "no" | None (ask)
    allow: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)
    interactive: Optional[bool] = None           # None: a terminal on stdin and stderr
    input_fn: Callable[[str], str] = input
    out: TextIO = sys.stderr
    decisions: list[dict] = field(default_factory=list)
    _cache: dict[str, bool] = field(default_factory=dict)
    _kind: dict[str, bool] = field(default_factory=dict)

    def known(self, kind: str) -> Optional[bool]:
        """The answer for `kind` if it is already settled (a flag, a remembered answer, or an earlier answer
        this run for a per-kind question), without asking. None: it would have to be asked."""
        if kind in self.deny:
            return False
        if kind in self.allow or "all" in self.allow:
            return True
        if self.assume in ("yes", "no"):
            return self.assume == "yes"
        if kind in self._kind:
            return self._kind[kind]
        mem = remembered()
        return mem.get(kind)

    @classmethod
    def from_args(cls, args=None, **kw) -> "Consent":
        env_assume = os.environ.get("WSR_ASSUME", "").strip().lower() or None
        assume = None
        if args is not None and getattr(args, "yes", False):
            assume = "yes"
        elif args is not None and getattr(args, "no", False):
            assume = "no"
        elif env_assume in ("yes", "no"):
            assume = env_assume
        allow = _split(os.environ.get("WSR_ALLOW")) | _split(getattr(args, "allow", None) if args else None)
        deny = _split(os.environ.get("WSR_DENY")) | _split(getattr(args, "deny", None) if args else None)
        bad = (allow | deny) - set(KINDS) - {"all"}
        if bad:
            raise ValueError(f"unknown consent kind(s): {', '.join(sorted(bad))} (known: {', '.join(KINDS)})")
        return cls(assume=assume, allow=allow, deny=deny, **kw)

    def _tty(self) -> bool:
        if self.interactive is not None:
            return self.interactive
        try:
            return sys.stdin.isatty() and self.out.isatty()
        except (AttributeError, ValueError):
            return False

    def ask(self, kind: str, question: str, *, detail: str = "") -> bool:
        """True when the user agrees. The same question is asked once per run."""
        if kind not in KINDS:
            raise ValueError(f"unknown consent kind {kind!r}")
        key = f"{kind}:{question}"
        if key in self._cache:
            return self._cache[key]
        if kind in self.deny:
            ok, how = False, f"--deny {kind}"
        elif kind in self.allow or "all" in self.allow:
            ok, how = True, f"--allow {kind}"
        elif self.assume == "yes":
            ok, how = True, "--yes"
        elif self.assume == "no":
            ok, how = False, "--no"
        elif kind in PER_KIND and kind in self._kind:
            ok, how = self._kind[kind], "same answer as earlier in this run"
        elif kind in (mem := remembered()):
            ok, how = mem[kind], "remembered (wsr consent reset to forget)"
            if not ok:
                print(f"wsr: not allowed to {question} (you answered 'never' before; wsr consent reset {kind})",
                      file=self.out)
        elif self._tty():
            print(f"\nwsr wants to {question}", file=self.out)
            if detail:
                print(f"  {detail}", file=self.out)
            try:
                ans = self.input_fn("  Allow? [y/N, always, never] ").strip().lower()
            except EOFError:
                ans = ""
            ok, how = ans in ("y", "yes", "a", "always"), "asked"
            if ans in ("always", "never"):
                remember(kind, ans == "always")
                how = f"asked ({ans}, remembered)"
        else:
            ok, how = False, "no terminal to ask"
            print(f"wsr: not allowed to {question} (no terminal to ask; rerun with --allow {kind} or --yes)",
                  file=self.out)
        self._cache[key] = ok
        if kind in PER_KIND:
            self._kind[kind] = ok
        self.decisions.append({"kind": kind, "question": question, "allowed": ok, "how": how})
        return ok

    def require(self, kind: str, question: str, *, what: str, why: str = "", detail: str = "") -> None:
        """Ask; on no, stop with NeedsConsent (for steps that have no fallback)."""
        if not self.ask(kind, question, detail=detail):
            raise NeedsConsent(kind, what, why)


def add_args(p) -> None:
    """--yes / --no / --allow / --deny on a subcommand."""
    g = p.add_argument_group("asking first (downloads, servers, isolated environments)")
    g.add_argument("--yes", action="store_true", help="answer yes to every question wsr would ask")
    g.add_argument("--no", action="store_true", help="answer no to every question (fall back where possible)")
    g.add_argument("--allow", metavar="KINDS", help="answer yes for these kinds: " + ", ".join(KINDS))
    g.add_argument("--deny", metavar="KINDS", help="answer no for these kinds")
