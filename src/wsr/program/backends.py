"""Where replies come from: PyTorch (transformers) or a llama-server.

Both give the same thing for one request: the reply text, its length in tokens, and whether it ended
by itself or was cut at the cap. Labels and grades are computed from that in one place
(`profile.Runner`), so a profile on llama-server and one on PyTorch are read the same way.

* `TorchBackend` - the full-precision model wsr edits; one sequence at a time; the only backend that
  can score a patch (the patch is applied to the weights in memory).
* `ServerBackend` - an OpenAI-compatible llama-server: an already-running one (`--endpoint`), or one
  wsr starts from a GGUF after asking (`ManagedServer`). Requests run in parallel slots, with a seed
  per request; it serves the quantised file users actually run. A patched model needs its own GGUF
  (`--patched-gguf`, from `wsr patch export --gguf`).

`choose()` picks one. `auto` uses llama-server when its binary and a GGUF are there and the user
agrees to start it, and otherwise PyTorch, saying why.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..consent import Consent, NeedsConsent
from .objective import Protocol


@dataclass
class Gen:
    text: str                      # what is classified (reasoning, if any, then the answer)
    answer: str                    # what is graded (the answer part)
    n_tokens: int
    ended: bool                    # stopped by itself (end-of-turn / stop)
    cut: bool                      # stopped at the token cap
    empty: bool                    # nothing at all before end-of-turn


class TorchBackend:
    name = "pytorch"
    parallel = 1

    def __init__(self, lm):
        self.lm = lm

    def describe(self) -> str:
        return f"pytorch ({self.lm.name}, one sequence at a time)"

    def generate(self, messages: list[dict], protocol: Protocol, *, seed: int, max_new: int,
                 gen_kwargs: Optional[dict] = None) -> Gen:
        import torch

        lm = self.lm
        kw = dict(protocol.gen_kwargs() if gen_kwargs is None else gen_kwargs)
        ids = lm.encode(messages, **protocol.template_kwargs)
        if kw.get("do_sample"):
            torch.manual_seed(seed)
        with torch.no_grad():
            gen = lm.generate(ids, max_new, **kw)
        ended = bool(gen) and gen[-1] in lm.eot_ids
        body = gen[:-1] if ended else gen
        text = lm.tok.decode(body, skip_special_tokens=True)
        return Gen(text, text, len(gen), ended, not ended and len(gen) >= max_new,
                   empty=not gen or gen[0] in lm.eot_ids or not text.strip())

    def generate_many(self, jobs: list[dict], protocol: Protocol) -> list[Gen]:
        """Jobs {messages, seed, max_new, gen_kwargs}: grouped by settings, each group generated in batches
        sized to the free memory (deep/batching.py). Same results as `generate` one by one."""
        lm = self.lm
        out: list[Optional[Gen]] = [None] * len(jobs)
        groups: dict[str, list[int]] = {}
        for k, j in enumerate(jobs):
            kw = protocol.gen_kwargs() if j.get("gen_kwargs") is None else j["gen_kwargs"]
            groups.setdefault(json.dumps([j["max_new"], kw], sort_keys=True, default=str), []).append(k)
        for key, ks in groups.items():
            max_new, kw = json.loads(key)
            ids = [lm.encode(jobs[k]["messages"], **protocol.template_kwargs) for k in ks]
            gens = lm.generate_batch(ids, max_new, seeds=[jobs[k]["seed"] for k in ks], **kw)
            for k, gen in zip(ks, gens):
                out[k] = self._gen(gen, max_new)
        return out  # type: ignore[return-value]

    def _gen(self, gen: list[int], max_new: int) -> Gen:
        lm = self.lm
        ended = bool(gen) and gen[-1] in lm.eot_ids
        body = gen[:-1] if ended else gen
        text = lm.tok.decode(body, skip_special_tokens=True)
        return Gen(text, text, len(gen), ended, not ended and len(gen) >= max_new,
                   empty=not gen or gen[0] in lm.eot_ids or not text.strip())

    def close(self) -> None:
        pass


def _sampling_body(protocol: Protocol, gen_kwargs: Optional[dict]) -> dict[str, Any]:
    """The protocol's sampling as llama-server fields (presence_penalty is native there)."""
    if gen_kwargs is not None:                      # a probe's own settings (e.g. T 0.7 reruns)
        if not gen_kwargs.get("do_sample"):
            return {"temperature": 0.0, "top_k": 1}
        body = {k: gen_kwargs[k] for k in ("temperature", "top_p", "top_k", "min_p") if k in gen_kwargs}
        if "repetition_penalty" in gen_kwargs:
            body["repeat_penalty"] = gen_kwargs["repetition_penalty"]
        return body
    if not protocol.sampled:
        body = {"temperature": 0.0, "top_k": 1}
    else:
        body = {k: protocol.sampling[k] for k in ("temperature", "top_p", "top_k", "min_p") if k in protocol.sampling}
    if "repetition_penalty" in protocol.sampling:
        body["repeat_penalty"] = protocol.sampling["repetition_penalty"]
    if "presence_penalty" in protocol.sampling:
        body["presence_penalty"] = protocol.sampling["presence_penalty"]
    return body


class ServerBackend:
    name = "llama-server"

    def __init__(self, base_url: str, *, model: str = "model", parallel: int = 4, timeout: float = 1800,
                 label: str = "", owned: Optional["ManagedServer"] = None):
        self.base = base_url.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base += "/v1"
        self.model, self.parallel, self.timeout = model, max(1, parallel), timeout
        self.label = label or self.base
        self.owned = owned

    def describe(self) -> str:
        return f"llama-server ({self.label}, {self.parallel} parallel)"

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"llama-server answered {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"llama-server at {self.base} is not reachable: {e.reason}") from e

    def generate(self, messages: list[dict], protocol: Protocol, *, seed: int, max_new: int,
                 gen_kwargs: Optional[dict] = None) -> Gen:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_new, "seed": seed,
                                "stream": False, **_sampling_body(protocol, gen_kwargs)}
        if protocol.template_kwargs:
            body["chat_template_kwargs"] = protocol.template_kwargs
        d = self._post(body)
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        answer = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        text = (reasoning + "\n" + answer) if reasoning else answer
        n = int((d.get("usage") or {}).get("completion_tokens") or 0)
        fin = ch.get("finish_reason")
        cut = fin == "length"
        return Gen(text, answer, n, ended=not cut, cut=cut, empty=not text.strip())

    def close(self) -> None:
        if self.owned is not None:
            self.owned.stop()


# -- starting a llama-server -----------------------------------------------------------------------------

def find_llama_server(explicit: Optional[str] = None) -> Optional[Path]:
    for c in (explicit, os.environ.get("WSR_LLAMA_SERVER")):
        if c:
            p = Path(c).expanduser()
            if p.is_dir():
                p = next((p / n for n in ("llama-server.exe", "llama-server") if (p / n).exists()), p)
            if p.exists():
                return p
    w = shutil.which("llama-server") or shutil.which("llama-server.exe")
    return Path(w) if w else None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ManagedServer:
    """A llama-server wsr started (after asking) and stops when done."""

    def __init__(self, binary: Path, gguf: Path, *, ctx_per_slot: int, parallel: int = 4, gpu_layers: int = 99,
                 log: Optional[Path] = None, extra: tuple[str, ...] = ()):
        self.binary, self.gguf, self.parallel = binary, gguf, parallel
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [str(binary), "-m", str(gguf), "--port", str(self.port), "--host", "127.0.0.1", "-ngl", str(gpu_layers),
               "-c", str(ctx_per_slot * parallel), "-np", str(parallel), "--jinja", *extra]
        self.cmd = cmd
        self._log = open(log, "w", encoding="utf-8") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT,
                                     cwd=str(binary.parent))

    def wait(self, timeout: float = 300) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode} while loading "
                                   f"{self.gguf.name}" + (" (see its log)" if self._log is not subprocess.DEVNULL
                                                          else ""))
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(1)
        self.stop()
        raise RuntimeError(f"llama-server did not become ready in {timeout:.0f} s")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log is not subprocess.DEVNULL:
            self._log.close()


@dataclass
class Choice:
    backend: Any
    notes: list[str]


def choose(kind: str, *, consent: Consent, protocol: Protocol, model_dir: Optional[str] = None,
           gguf: Optional[str] = None, endpoint: Optional[str] = None, server_bin: Optional[str] = None,
           parallel: int = 4, need_patch: bool = False, patched_gguf: Optional[str] = None,
           load_torch=None, log: Optional[Path] = None) -> Choice:
    """kind: auto | torch | llama-server. Returns the backend and what was decided, in words.

    `need_patch`: the caller will score a patch. Only PyTorch can apply it in memory; a server needs the
    patched GGUF (then the caller gets a second backend for it through `choose(..., gguf=patched_gguf)`)."""
    notes: list[str] = []
    want_server = kind in ("auto", "llama-server")
    if kind not in ("auto", "torch", "llama-server"):
        raise ValueError("--backend is auto, torch or llama-server")
    if want_server and need_patch and not patched_gguf and not endpoint:
        msg = ("a patch can only be scored on llama-server with its own GGUF (--patched-gguf, from wsr patch export "
               "--gguf)")
        if kind == "llama-server":
            raise ValueError(msg)
        notes.append(msg + "; using PyTorch")
        want_server = False
    if want_server and endpoint:
        return Choice(ServerBackend(endpoint, parallel=parallel, label=endpoint), notes + [f"llama-server at {endpoint}"])
    if want_server:
        binary = find_llama_server(server_bin)
        missing = []
        if binary is None:
            missing.append("the llama-server binary (--llama-server PATH, WSR_LLAMA_SERVER, or on PATH)")
        if not gguf:
            missing.append("a GGUF of the model (--gguf FILE)")
        elif not Path(gguf).exists():
            missing.append(f"the GGUF {gguf} (not found)")
        if missing:
            if kind == "llama-server":
                raise ValueError("llama-server backend needs " + " and ".join(missing))
            notes.append("llama-server would be faster for long or many replies, but needs " + " and ".join(missing)
                         + "; using PyTorch")
        else:
            assert binary is not None and gguf
            ok = consent.ask("llama-server", f"start llama-server with {Path(gguf).name}",
                             detail=f"{binary} on 127.0.0.1, {parallel} parallel slots, GPU offload; stopped when "
                                    "wsr finishes. Faster than PyTorch for long replies. No: use PyTorch instead.")
            if ok:
                srv = ManagedServer(binary, Path(gguf), ctx_per_slot=protocol.max_new_tokens * 2 + 2048,
                                    parallel=parallel, log=log)
                srv.wait()
                return Choice(ServerBackend(srv.url, parallel=parallel, label=f"{Path(gguf).name}, started by wsr",
                                            owned=srv), notes + [f"llama-server started: {Path(gguf).name}"])
            if kind == "llama-server":
                raise NeedsConsent("llama-server", "starting llama-server", "--backend llama-server was asked for")
            notes.append("llama-server not started (answered no); using PyTorch")
    if load_torch is None or not model_dir:
        raise ValueError("the PyTorch backend needs the full-precision model directory")
    return Choice(TorchBackend(load_torch()), notes)
