"""Where the held-out judge runs: the served file on llama-server, or the full-precision model on PyTorch.

`--judge auto` (the default) uses llama-server when its binary is there and the user agrees: to export
the candidate (and, once, the original) to a GGUF and to start the servers. Otherwise, after a "no" (now,
or remembered from an earlier "never") or without llama-server, it uses PyTorch, batched to the free
memory. `--judge torch` never asks; `--judge llama-server` stops when it cannot.

On llama-server:
* the original model's GGUF is cached under `~/.cache/wsr/gguf/`, keyed by the model's files, so it is
  converted once per model and quantisation;
* the candidate's GGUF is written to `<out>/served/`, and it is the file that ships with the patch;
* the regression-suite guardrail still runs on PyTorch: it measures drift on the full-precision model,
  with short replies, so it is cheap there.

The report says which judge ran. A verdict on Q8_0 is a verdict on the file users serve, which is close
to the full-precision model but not the same thing.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..consent import Consent, NeedsConsent
from .backends import ManagedServer, ServerBackend, find_llama_server


@dataclass
class Judge:
    kind: str                                   # "llama-server" | "pytorch"
    source: Any = None
    patched: Any = None
    notes: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    server_version: Optional[str] = None

    def close(self) -> None:
        for s in (self.source, self.patched):
            if s is not None and hasattr(s, "close"):
                s.close()


def _model_key(model_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(model_dir.iterdir()):
        if f.is_file() and f.suffix in (".json", ".safetensors", ".model", ".bin"):
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
            if f.name == "config.json":
                h.update(f.read_bytes())
    return f"{model_dir.name}-{h.hexdigest()[:12]}"


def original_gguf(model_dir: Path, quant: str, *, consent: Consent, server: Optional[Path], log=print) -> Path:
    from ..surgery.export import cache_dir, export

    p = cache_dir() / "gguf" / f"{_model_key(model_dir)}.{quant}.gguf"
    if p.exists():
        return p
    export(model_dir, None, p, quant=quant, consent=consent, server=server, log=log)
    return p


USE_QUESTION = ("use llama-server for this run (the practice profile and the held-out judge): the model is exported "
                "to a {quant} GGUF once (cached) and served locally")


def _cover_convert(consent: Consent) -> None:
    """A yes to the llama-server question includes the GGUF export it names."""
    if "convert" not in consent.allow and consent.known("convert") is not False:
        consent.allow.add("convert")
        consent.decisions.append({"kind": "convert", "question": "export the model to GGUF for llama-server",
                                  "allowed": True, "how": "covered by the llama-server answer"})


def served_original(*, model_dir: Path, consent: Consent, server_bin: Optional[str], quant: str = "Q8_0",
                    max_new: int = 512, parallel: int = 4, required: bool = False, log=print):
    """llama-server on the original model, as the default profile backend for a user who uses llama-server.
    (backend or None, note). Not asked again when the answer is already known: a "no" anywhere (flag,
    remembered 'never', or earlier this run) means PyTorch without a question."""
    say = log or (lambda s: None)
    server = find_llama_server(server_bin)
    if server is None:
        if required:
            raise ValueError("--backend llama-server: llama-server was not found (--llama-server PATH)")
        return None, "llama-server not found: profile on PyTorch (batched)"
    if not consent.ask("llama-server", USE_QUESTION.format(quant=quant),
                       detail="faster for long or many replies. No: everything runs on PyTorch, batched to free memory"):
        if required:
            raise NeedsConsent("llama-server", "starting llama-server", "--backend llama-server was asked for")
        return None, "llama-server not used (answered no): profile on PyTorch (batched)"
    _cover_convert(consent)
    try:
        orig = original_gguf(model_dir, quant, consent=consent, server=server, log=say)
        srv = ManagedServer(server, orig, ctx_per_slot=max_new + 2048, parallel=parallel)
        srv.wait()
    except Exception as e:                              # noqa: BLE001 - optional: fall back to PyTorch
        if required:
            raise
        return None, f"llama-server could not be used ({str(e)[:160]}): profile on PyTorch (batched)"
    return (ServerBackend(srv.url, parallel=parallel, label=f"{orig.name}", owned=srv),
            f"profile on llama-server ({orig.name})")


def choose(kind: str, *, lm, model_dir: Optional[Path], patch, out: Path, consent: Consent, quant: str = "Q8_0",
           server_bin: Optional[str] = None, max_new: int = 512, parallel: int = 4, log=print) -> Judge:
    say = log or (lambda s: None)
    if kind not in ("auto", "torch", "llama-server"):
        raise ValueError("--judge is auto, torch or llama-server")
    torch_judge = Judge("pytorch", source=lm)
    if kind == "torch":
        return torch_judge
    server = find_llama_server(server_bin)
    why = None
    if server is None:
        why = "llama-server was not found (--llama-server PATH, WSR_LLAMA_SERVER, or on PATH)"
    elif model_dir is None:
        why = "the model directory is needed to convert it"
    elif not consent.ask("llama-server", f"judge on llama-server: export the original and the candidate to {quant} "
                                         "GGUF and serve them",
                         detail="much faster for long replies; the regression guardrail stays on PyTorch. No: the "
                                "judge runs on PyTorch (batched to free memory)"):
        why = "llama-server judge not allowed (answered no)"
    else:
        _cover_convert(consent)
    if why is None:
        assert server is not None and model_dir is not None
        try:
            orig = original_gguf(model_dir, quant, consent=consent, server=server, log=say)
            cand = out / "served" / f"candidate.{quant}.gguf"
            from ..surgery.export import export

            pdir = out / "candidate_tmp"
            patch.save(pdir)
            try:
                export(model_dir, pdir, cand, quant=quant, consent=consent, server=server, log=say, lm=lm)
            finally:
                import shutil
                shutil.rmtree(pdir, ignore_errors=True)
            ctx = max_new + 2048
            s1 = ManagedServer(server, orig, ctx_per_slot=ctx, parallel=parallel)
            s1.wait()
            try:
                s2 = ManagedServer(server, cand, ctx_per_slot=ctx, parallel=parallel)
                s2.wait()
            except Exception:
                s1.stop()
                raise
            from ..surgery.export import server_tag

            return Judge("llama-server", ServerBackend(s1.url, parallel=parallel, label=f"{orig.name}", owned=s1),
                         ServerBackend(s2.url, parallel=parallel, label=f"{cand.name}", owned=s2),
                         notes=[f"judged on {quant} GGUFs via llama-server (both on the same build and settings); "
                                "the suite's drift checks on PyTorch, its reply checks and the canary on the GGUFs"],
                         files={"original": str(orig), "candidate": str(cand)}, server_version=server_tag(server))
        except NeedsConsent as e:
            why = f"llama-server judge not possible: {e}"
        except Exception as e:                          # noqa: BLE001 - the fast judge is optional: never sink the run
            why = f"llama-server judge failed ({type(e).__name__}: {str(e)[:200]})"
    if kind == "llama-server":
        raise ValueError(f"--judge llama-server: {why}")
    torch_judge.notes.append(f"{why}; judged on PyTorch (batched)")
    say(f"  judge: {why}; using PyTorch (batched)")
    return torch_judge
