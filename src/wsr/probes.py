"""Layers 3b and 5: live endpoint probes and differential reruns.

Client: OpenAI-compatible ``/v1/chat/completions`` (Ollama, llama-server, LM
Studio, vLLM all serve it) or Ollama-native ``/api/chat``. Backend detection
follows LLMDoctor (``/api/version``, ``/props``, ``/api/v0/models``).

Probes, each tied to a source technique:

* reasoning    — does the model think before answering, how many tokens does a
                 one-word answer cost, does a small max_tokens blank the answer,
                 and does the per-request thinking switch work. Every later
                 probe gives the answer room for the reasoning.
* bos          — llama-server only: render + tokenise exactly as the server does
                 and count the BOS tokens at the start (double-BOS ground truth).
* cache        — llama-server only: the same greedy request evaluated fresh
                 (cache_prompt=false) and on top of a reused prompt cache must
                 give the same reply; on sliding-window models it may not.
* smoke        — "Say hello." at temperature 0 (troubleshooting guides): garbage
                 from token one = file/backend, control tokens = template.
* stop         — a one-word answer must end with a stop, not the cap: a model
                 that runs on to max_tokens is not emitting / honouring its
                 end-of-turn token (Raschka EOS check).
* finish       — tiny max_tokens must report a length finish reason, otherwise
                 truncation is invisible on this server (LLMDoctor max-tokens).
* context      — canary codes at the start and end of long prompts; the server's
                 prompt token count decides truncation vs weak recall (LLMDoctor).
* tools        — tool-call arguments validated against the schema (LLMDoctor).
* determinism  — same greedy request twice (Raschka).
* vision       — two solid-colour PNGs, one-word answers: does the model see
                 the image at all, and does it read it right (mismatched mmproj).

Differential: rerun the failing request flipping one factor at a time
(sampling, penalties, constraint, system prompt, context length, alternate
endpoint). The factor that flips the result is the cause (product report).
"""
from __future__ import annotations

import json
import random
import string
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .findings import Cause, Finding, Layer, Severity
from .media import content_text, has_images, images_as_parts
from .output import check_output
from .response import check_response, normalize


class EndpointError(Exception):
    pass


@dataclass
class Endpoint:
    url: str
    model: str
    api: str = "openai"            # "openai" | "ollama"
    api_key: Optional[str] = None
    timeout: float = 120
    backend: Optional[str] = None
    budget_s: float = 300
    _deadline: float = field(default=0.0, repr=False)
    reasoning: Optional[bool] = None        # learned by probe_reasoning
    reasoning_tokens: int = 0               # tokens spent on a one-word answer with thinking on
    thinking_off: Optional[dict] = None     # request params verified to turn thinking off
    _props: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")
        if self.url.endswith("/v1"):
            self.url = self.url[:-3]
        self._deadline = time.monotonic() + self.budget_s

    # -- HTTP -----------------------------------------------------------------
    def reachable(self, timeout: float = 3.0) -> Optional[str]:
        """None if the URL answers HTTP at all (any status), else why not. Checked once, then cached.

        A dead or wrong port must fail fast: on Windows a closed localhost port
        does not refuse, and a port held by a non-HTTP service accepts and then
        never answers. Either way each request would wait out its full timeout,
        and the probes chain dozens of requests.
        """
        if not hasattr(self, "_reach"):
            req = urllib.request.Request(self.url + "/", headers={"User-Agent": "wsr-triage"})
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            try:
                urllib.request.urlopen(req, timeout=timeout).close()
                self._reach = None
            except urllib.error.HTTPError:
                self._reach = None  # 404 / 401 still means a server is there
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                reason = getattr(e, "reason", e)
                self._reach = f"{self.url} does not answer HTTP ({reason})"
        return self._reach

    def _open(self, method: str, path: str, body: Optional[dict], timeout: Optional[float]):
        why = self.reachable()
        if why:
            raise EndpointError(why)
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise EndpointError("probe time budget exhausted")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", "User-Agent": "wsr-triage"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        return urllib.request.urlopen(req, timeout=min(timeout or self.timeout, remaining))

    def chat_stream(self, messages: list[dict], **params: Any) -> list[str]:
        """The raw stream lines (SSE or NDJSON); a dropped connection keeps what arrived."""
        if self.api == "ollama":
            body, path = self._ollama_body(messages, params), "/api/chat"
            body["stream"] = True
        else:
            body = {"model": self.model, "messages": messages, **params, "stream": True,
                    "stream_options": {"include_usage": True}}
            path = "/v1/chat/completions"
        lines: list[str] = []
        try:
            with self._open("POST", path, body, None) as r:
                for raw in r:
                    lines.append(raw.decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            return ["data: " + (e.read().decode("utf-8", "replace") or json.dumps({"error": {"code": e.code}}))]
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if not lines:
                raise EndpointError(f"POST {path} (stream): {e}") from e
        return lines

    def _req(self, method: str, path: str, body: Optional[dict] = None, timeout: Optional[float] = None) -> Any:
        try:
            with self._open(method, path, body, timeout) as r:
                return json.loads(r.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read().decode("utf-8"))
            except ValueError:
                return {"error": {"message": f"HTTP {e.code}", "code": e.code}}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise EndpointError(f"{method} {path}: {e}") from e

    def _ollama_body(self, messages: list[dict], params: dict) -> dict:
        params = dict(params)
        opts = dict(params.pop("options", {}) or {})
        for k in ("temperature", "top_p", "top_k", "min_p", "seed", "repeat_penalty", "num_ctx", "stop"):
            if k in params:
                opts[k] = params.pop(k)
        if "max_tokens" in params:
            opts["num_predict"] = params.pop("max_tokens")
        return {"model": self.model, "messages": messages, "stream": False, "options": opts, **params}

    def detect_backend(self) -> str:
        if self.backend:
            return self.backend
        for path, name, test in (
            ("/api/version", "ollama", lambda j: isinstance(j, dict) and "version" in j and "error" not in j),
            ("/props", "llama-server", lambda j: isinstance(j, dict) and ("default_generation_settings" in j
                                                                           or "chat_template" in j)),
            ("/api/v0/models", "lmstudio", lambda j: isinstance(j, dict) and isinstance(j.get("data"), list)),
            ("/v1/models", "vllm", lambda j: isinstance(j, dict) and any(
                (m.get("owned_by") == "vllm") for m in j.get("data", []) if isinstance(m, dict))),
        ):
            try:
                if test(self._req("GET", path, timeout=5)):
                    self.backend = name
                    return name
            except EndpointError:
                continue
        self.backend = "openai-compatible"
        return self.backend

    def no_think_params(self) -> dict:
        """Per-request switch that turns thinking off (Ollama native: think; others: template kwarg)."""
        if self.api == "ollama":
            return {"think": False}
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def props(self) -> dict:
        if self._props is None:
            self._props = {}
            try:
                if self.detect_backend() == "llama-server":
                    p = self._req("GET", "/props", timeout=5)
                    self._props = p if isinstance(p, dict) else {}
            except EndpointError:
                pass
        return self._props

    def server_ctx(self) -> Optional[int]:
        """Context size the server actually runs with, where it says."""
        p = self.props()
        n = (p.get("default_generation_settings") or {}).get("n_ctx") or p.get("n_ctx")
        return n if isinstance(n, int) and n > 0 else None

    def server_template(self) -> Optional[str]:
        """The template the server actually uses, where it exposes one."""
        try:
            if self.detect_backend() == "llama-server":
                return self._req("GET", "/props", timeout=5).get("chat_template")
            if self.backend == "ollama":
                return self._req("POST", "/api/show", {"model": self.model}, timeout=10).get("template")
        except (EndpointError, AttributeError):
            return None
        return None

    def chat(self, messages: list[dict], **params: Any) -> tuple[dict, float]:
        t0 = time.monotonic()
        if self.api == "ollama":
            raw = self._req("POST", "/api/chat", self._ollama_body(messages, params))
        else:
            body = {"model": self.model, "messages": messages, "stream": False, **params}
            raw = self._req("POST", "/v1/chat/completions", body)
        return raw, time.monotonic() - t0


def _pf(code: str, sev: Severity, msg: str, cause: Cause = Cause.NONE, fix: Optional[str] = None,
        source: str = "wsr probe", **ev: Any) -> Finding:
    return Finding(code=code, layer=Layer.PROBE, severity=sev, message=msg, cause=cause, fix=fix, source=source,
                   evidence={k: v for k, v in ev.items() if v is not None})


def _failing(findings: list[Finding]) -> set[str]:
    return {f.code for f in findings if f.failed}


_PENALTY_KEYS = ("repeat_penalty", "repetition_penalty", "frequency_penalty", "presence_penalty")


def is_penalised(params: dict) -> bool:
    """A repetition penalty strong enough to distort the reply (llama.cpp / OpenAI / Ollama spellings)."""
    p = {**(params.get("options") or {}), **params}
    rp = p.get("repeat_penalty", p.get("repetition_penalty"))
    if isinstance(rp, (int, float)) and rp > 1.15:
        return True
    return any(isinstance(p.get(k), (int, float)) and abs(p[k]) >= 0.5
               for k in ("frequency_penalty", "presence_penalty"))


def without_penalties(params: dict) -> dict:
    out = {k: x for k, x in params.items() if k not in _PENALTY_KEYS}
    if isinstance(params.get("options"), dict):
        out["options"] = {k: x for k, x in params["options"].items() if k not in _PENALTY_KEYS}
    return out


def run_and_classify(ep: Endpoint, messages: list[dict], params: dict, *, expect_lang: Optional[str] = None,
                     tools: Optional[list] = None, preset: str = "chat", echo: bool = False
                     ) -> tuple[Any, list[Finding], str]:
    raw, _ = ep.chat(messages, **dict(params))
    max_tokens = params.get("max_tokens") or (params.get("options") or {}).get("num_predict")
    n, resp = check_response(raw, max_tokens=max_tokens, tools=tools)
    out: list[Finding] = []
    if n.text.strip() or not n.tool_calls:
        prompt_text = "\n".join(content_text(m.get("content")) for m in messages if m.get("role") != "assistant")
        out = check_output(n.text, preset=preset, finish_reason=None, expect_lang=expect_lang,
                           prompt=prompt_text if echo else None, penalised=is_penalised(params),
                           has_images=has_images(messages, params),
                           request=next((content_text(m.get("content")) for m in reversed(messages)
                                         if m.get("role") == "user"), None))
    return raw, resp + out, n.text


# -- reasoning budget -------------------------------------------------------------

def _room(ep: Endpoint, params: dict, *, plain: bool = False) -> dict:
    """Give the answer room when the model thinks first.

    plain=True turns thinking off instead when the switch is known to work, for
    probes that test something other than the reasoning path.
    """
    if not ep.reasoning:
        return params
    if plain and ep.thinking_off:
        return {**params, **ep.thinking_off}
    extra = max(512, 4 * ep.reasoning_tokens)
    ctx = ep.server_ctx()
    if ctx:
        extra = min(extra, max(ctx // 2, 64))
    return dict(params, max_tokens=int(params.get("max_tokens") or 0) + extra)


_REASONING_CODES = {"REASONING_EXHAUSTED"}

# -- probes ---------------------------------------------------------------------

TRIVIAL = [{"role": "user", "content": "Reply with exactly one word: OK"}]
SMALL_CAP = 64


def probe_reasoning(ep: Endpoint) -> list[Finding]:
    from .output import reasoning_leak_score

    raw, _ = ep.chat(TRIVIAL, temperature=0, seed=0, max_tokens=1024)
    n = normalize(raw)
    if n.error:
        return [_pf("REASONING_PROBE_ERROR", Severity.SKIP, str(n.error.get("message", n.error)))]
    out: list[Finding] = []
    if reasoning_leak_score(n.text) >= 0.5:
        out.append(_pf("REASONING_UNPARSED", Severity.FAIL,
                       f"reasoning arrives inside the answer text: {n.text[:80]!r}", Cause.RUNTIME,
                       "turn on the server's reasoning parser (llama-server --reasoning-format auto / deepseek; "
                       "vLLM --reasoning-parser; Ollama think=true) so it goes to a separate field",
                       source="troubleshooting guides"))
    if not (n.reasoning_text.strip() or n.reasoning_tokens):
        ep.reasoning = bool(out)
        return out or [_pf("REASONING_NONE", Severity.PASS, "no separate reasoning output")]
    ep.reasoning = True
    tok = n.completion_tokens or max(1, len(n.reasoning_text) // 4)
    ep.reasoning_tokens = tok
    out.append(_pf("THINKING_ON", Severity.INFO,
                   f"the model reasons before answering: {tok} tokens for a one-word answer "
                   f"({len(n.reasoning_text)} chars of reasoning)", tokens=tok))
    if not n.text.strip() and n.finish_reason in ("length", "max_tokens"):
        out.append(_pf("REASONING_RUNAWAY", Severity.FAIL,
                       "1024 tokens of reasoning and still no answer to 'Reply with exactly one word: OK'",
                       Cause.REQUEST, "set a reasoning budget (llama-server --reasoning-budget, Ollama think=low) "
                       "or turn thinking off for short tasks"))
    raw2, _ = ep.chat(TRIVIAL, temperature=0, seed=0, max_tokens=SMALL_CAP)
    n2 = normalize(raw2)
    if not n2.error and not n2.text.strip() and not n2.tool_calls:
        out.append(_pf("REASONING_STARVES_ANSWER", Severity.WARN,
                       f"with max_tokens={SMALL_CAP} the answer comes back blank: the whole budget goes to "
                       f"reasoning ({tok} tokens for one word). Any caller with a small cap gets empty replies",
                       Cause.REQUEST, f"raise max_tokens well above {tok}, or turn thinking off per request "
                       "(chat_template_kwargs={'enable_thinking': false} on llama-server / vLLM, think=false on "
                       "Ollama; server-wide: llama-server --reasoning-budget 0)",
                       max_tokens=SMALL_CAP, reasoning_tokens=tok))
    off = ep.no_think_params()
    raw3, _ = ep.chat(TRIVIAL, temperature=0, seed=0, max_tokens=SMALL_CAP, **off)
    n3 = normalize(raw3)
    if n3.error:
        out.append(_pf("THINKING_SWITCH_UNSUPPORTED", Severity.INFO,
                       f"server rejected {off}: {n3.error.get('message', n3.error)}"))
    elif not (n3.reasoning_text.strip() or n3.reasoning_tokens) and n3.text.strip():
        ep.thinking_off = off
        out.append(_pf("THINKING_SWITCH_OK", Severity.PASS,
                       f"{off} turns thinking off: one-word answer in {n3.completion_tokens} tokens instead of {tok}",
                       tokens_off=n3.completion_tokens))
    else:
        out.append(_pf("THINKING_SWITCH_IGNORED", Severity.WARN,
                       f"{off} did not turn thinking off", Cause.RUNTIME,
                       "use the server-wide switch (llama-server --reasoning-budget 0) or a non-thinking template"))
    return out


def probe_bos(ep: Endpoint) -> list[Finding]:
    """llama-server: tokenise the rendered prompt exactly as the server does and count leading BOS."""
    if ep.detect_backend() != "llama-server":
        return []
    bos = ep.props().get("bos_token")
    msgs = [{"role": "user", "content": "Say hello."}]
    try:
        r = ep._req("POST", "/apply-template", {"messages": msgs})
        prompt = r.get("prompt") if isinstance(r, dict) else None
        if not prompt or not bos:
            return []
        t = ep._req("POST", "/tokenize", {"content": prompt, "add_special": True, "with_pieces": True})
    except EndpointError:
        return []
    toks = t.get("tokens") if isinstance(t, dict) else None
    if not isinstance(toks, list):
        return []
    lead = 0
    for x in toks:
        if isinstance(x, dict) and x.get("piece") == bos:
            lead += 1
        else:
            break
    if lead > 1:
        return [_pf("SERVER_DOUBLE_BOS", Severity.FAIL, f"the server's prompt starts with {lead} x {bos!r}",
                    Cause.CONFIG, "the template writes BOS and the tokenizer adds another: remove it from the "
                    "template, or set tokenizer.ggml.add_bos_token=false", source="troubleshooting guides")]
    if lead == 0:
        return [_pf("SERVER_NO_BOS", Severity.WARN, f"the server's prompt has no {bos!r} at the start", Cause.CONFIG,
                    "most models need BOS: check tokenizer.ggml.add_bos_token and the template",
                    source="troubleshooting guides")]
    return [_pf("SERVER_BOS_OK", Severity.PASS, f"one {bos!r} at the start of the server's prompt")]

def probe_smoke(ep: Endpoint) -> list[Finding]:
    raw, fs, text = run_and_classify(ep, [{"role": "user", "content": "Say hello."}],
                                     _room(ep, {"temperature": 0, "max_tokens": 64, "seed": 0}))
    bad = _failing(fs)
    if bad & _REASONING_CODES:
        return [_pf("SMOKE_BLANK", Severity.FAIL, "blank answer to 'Say hello.': reasoning used the whole budget",
                    Cause.REQUEST, "raise max_tokens or turn thinking off for this call")]
    if bad & {"SYMBOL_SOUP", "LOW_ENTROPY"}:
        return [_pf("SMOKE_GARBAGE", Severity.FAIL, f"garbage on a trivial greedy prompt: {text[:60]!r}",
                    Cause.RUNTIME, "file or backend: re-download / verify the file, then try another backend "
                    "(CPU vs GPU, CUDA vs Vulkan) and a higher-precision quant", source="troubleshooting guides")]
    if bad & {"CONTROL_TOKEN_LEAK", "INVENTED_TURN", "REASONING_LEAK"}:
        return [_pf("SMOKE_TEMPLATE_LEAK", Severity.FAIL, f"control tokens / extra turns on 'Say hello.': "
                    f"{text[:80]!r}", Cause.CONFIG, "the server's chat template is wrong or stale; compare it with "
                    "upstream (`wsr config --upstream`)", source="troubleshooting guides")]
    if bad & {"EMPTY", "EMPTY_OUTPUT", "TRUNCATED_EMPTY"}:
        return [_pf("SMOKE_BLANK", Severity.FAIL, "blank answer to 'Say hello.'", Cause.CONFIG,
                    "blank on a trivial prompt points at the template, EOS, or reasoning parser")]
    return [_pf("SMOKE_OK", Severity.PASS, f"'Say hello.' -> {text.strip()[:60]!r}")]


def probe_stop(ep: Endpoint) -> list[Finding]:
    raw, fs, text = run_and_classify(ep, TRIVIAL, _room(ep, {"temperature": 0, "max_tokens": 96, "seed": 0}))
    n = normalize(raw)
    if _failing(fs) & _REASONING_CODES:
        return [_pf("STOP_PROBE_INCONCLUSIVE", Severity.SKIP,
                    "reasoning used the whole budget before any answer; stop behaviour not tested")]
    if n.finish_reason in ("length", "max_tokens"):
        return [_pf("STOP_PROBE_RUNON", Severity.FAIL,
                    f"a one-word answer ran to max_tokens ({len(text)} chars): the end-of-turn token is not "
                    "emitted or not treated as a stop", Cause.CONFIG,
                    "make generation stop on the template's end-of-turn token (eos_token_id / stop strings); "
                    "run `wsr config` on the model files", source="Raschka checklist", text=text[:200])]
    return [_pf("STOP_PROBE_OK", Severity.PASS, f"stopped on its own ({n.finish_reason})")]


def probe_finish_reason(ep: Endpoint) -> list[Finding]:
    raw, _ = ep.chat([{"role": "user", "content": "Count from 1 to 200, separated by commas."}],
                     temperature=0, max_tokens=8, seed=0)
    n = normalize(raw)
    if n.finish_reason in ("length", "max_tokens"):
        return [_pf("FINISH_REASON_OK", Severity.PASS, "server reports a length finish reason at the cap",
                    source="LLMDoctor max-tokens")]
    return [_pf("FINISH_REASON_UNRELIABLE", Severity.WARN,
                f"cut at 8 tokens but finish_reason is {n.finish_reason!r}: truncation is invisible on this server",
                Cause.RUNTIME, "rely on output-level truncation heuristics; check the server / proxy version",
                source="LLMDoctor max-tokens")]


def _canary() -> str:
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))


def probe_context(ep: Endpoint, levels: tuple[int, ...] = (2048, 4096, 8192, 16384)) -> list[Finding]:
    filler_unit = ("The archive contains routine maintenance notes about pumps, valves and filters. "
                   "Nothing in this paragraph is important. ")
    # Calibrate chars per token from the server's own count.
    raw, _ = ep.chat([{"role": "user", "content": filler_unit * 20}], temperature=0, max_tokens=1)
    pt = normalize(raw).prompt_tokens
    cpt = (len(filler_unit) * 20 / pt) if pt else 4.0
    out: list[Finding] = []
    for level in levels:
        target_chars = int(level * 0.85 * cpt)
        a, b = _canary(), _canary()
        body = filler_unit * max(1, target_chars // len(filler_unit))
        prompt = (f"Remember code {a}.\n\n{body}\n\nRemember code {b}.\n\n"
                  "What were the two codes? Answer with both codes only.")
        try:
            raw, _ = ep.chat([{"role": "user", "content": prompt}],
                             **_room(ep, {"temperature": 0, "max_tokens": 32}, plain=True))
        except EndpointError as e:
            out.append(_pf("CONTEXT_PROBE_ERROR", Severity.SKIP, f"{level}: {e}"))
            break
        n = normalize(raw)
        sent = int(len(prompt) / cpt)
        seen = n.text + "\n" + n.reasoning_text
        got_a, got_b = a in seen, b in seen
        if n.error:
            out.append(_pf("CONTEXT_LIMIT", Severity.INFO, f"{level}-token prompt rejected: "
                           f"{n.error.get('message', n.error)}", context=level))
            break
        if n.prompt_tokens is not None and n.prompt_tokens < 0.8 * sent:
            lost = "start" if not got_a else ("end" if not got_b else "middle")
            out.append(_pf("CONTEXT_TRUNCATION", Severity.FAIL,
                           f"~{sent}-token prompt: server read {n.prompt_tokens}; the {lost} was dropped silently",
                           Cause.REQUEST, "raise the server context size (num_ctx / -c / --max-model-len)",
                           source="LLMDoctor", level=level, prompt_tokens=n.prompt_tokens, sent=sent))
            break
        if not (got_a and got_b) and not n.text.strip() and n.reasoning_text.strip():
            out.append(_pf("CONTEXT_INCONCLUSIVE", Severity.SKIP,
                           f"~{sent}-token prompt: reasoning used the budget before the answer", level=level))
            continue
        if not (got_a and got_b):
            out.append(_pf("WEAK_RECALL", Severity.WARN,
                           f"~{sent}-token prompt fully read but the model lost the "
                           f"{'start' if not got_a else 'end'} canary", Cause.WEIGHT_LEVEL,
                           source="LLMDoctor", level=level))
            continue
        out.append(_pf("CONTEXT_OK", Severity.PASS, f"~{sent} tokens read and recalled", level=level))
    return out


TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}


def probe_tools(ep: Endpoint) -> list[Finding]:
    raw, fs, text = run_and_classify(ep, [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
                                     _room(ep, {"temperature": 0, "max_tokens": 256, "seed": 0, "tools": [TOOL]}),
                                     tools=[TOOL])
    n = normalize(raw)
    starved = _failing(fs) & _REASONING_CODES or (
        "TOOL_CALL_TRUNCATED" in _failing(fs) and n.reasoning_text.strip() and not n.tool_calls)
    if starved:
        return [_pf("TOOL_PROBE_INCONCLUSIVE", Severity.SKIP, "reasoning used the budget before the tool call")]
    bad = [f for f in fs if f.failed and f.code.startswith(("TOOL_", "CONTROL_TOKEN"))]
    if bad:
        return [_pf("TOOL_PROBE_FAIL", Severity.FAIL, bad[0].message, Cause.RUNTIME,
                    "check the server's tool-call parser and chat template", source="LLMDoctor")]
    if not n.tool_calls:
        return [_pf("TOOL_PROBE_NO_CALL", Severity.WARN, f"no tool call; answered {text[:60]!r}", Cause.CONFIG,
                    "template may not render tools; check --jinja (llama-server) or the model's tool support",
                    source="LLMDoctor")]
    return [_pf("TOOL_PROBE_OK", Severity.PASS, f"valid call {n.tool_calls[0].get('name')}", source="LLMDoctor")]


def probe_determinism(ep: Endpoint, messages: Optional[list[dict]] = None, params: Optional[dict] = None
                      ) -> list[Finding]:
    messages = messages or [{"role": "user", "content": "Name three primary colours, one line each."}]
    p = _room(ep, dict(params or {"max_tokens": 64}, temperature=0, seed=0))
    if ep.detect_backend() == "llama-server":
        # Both evaluated fresh: a reused prompt cache can change the reply on its own (probe_cache),
        # which is a different finding from nondeterministic kernels.
        p["cache_prompt"] = False
    a, b = normalize(ep.chat(messages, **p)[0]), normalize(ep.chat(messages, **p)[0])
    t1, t2 = a.text + a.reasoning_text, b.text + b.reasoning_text
    if t1 != t2:
        return [_pf("NONDETERMINISTIC_GREEDY", Severity.WARN,
                    "two identical greedy requests gave different text: nondeterministic kernels, batching or "
                    "hidden state", Cause.RUNTIME, "disable batching / use a deterministic backend to reproduce",
                    source="Raschka checklist")]
    return [_pf("DETERMINISTIC_GREEDY", Severity.PASS, "greedy output reproducible", source="Raschka checklist")]


def _parsed_calls(calls: list[dict]) -> list:
    res = []
    for c in calls:
        a = c.get("arguments")
        try:
            res.append((c.get("name"), json.loads(a) if isinstance(a, str) else a))
        except ValueError:
            res.append((c.get("name"), a))
    return res


def probe_stream(ep: Endpoint) -> list[Finding]:
    """Same greedy request streamed and not: the stream must be intact and say the same thing."""
    from .stream import assemble

    out: list[Finding] = []
    # Fresh evaluation on both sides, so prompt-cache effects (probe_cache) do not
    # show up as streaming differences.
    fresh = {"cache_prompt": False} if ep.detect_backend() == "llama-server" else {}
    for label, msgs, params in (
        ("text", [{"role": "user", "content": "Count from 1 to 15, separated by commas."}],
         _room(ep, {"temperature": 0, "seed": 0, "max_tokens": 96})),
        ("tool", [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
         _room(ep, {"temperature": 0, "seed": 0, "max_tokens": 256, "tools": [TOOL]})),
    ):
        params = {**params, **fresh}
        lines = ep.chat_stream(msgs, **params)
        s_raw, s_find = assemble(lines)
        s, n = normalize(s_raw), normalize(ep.chat(msgs, **params)[0])
        for f in s_find:
            f.layer, f.evidence = Layer.PROBE, dict(f.evidence, probe=label)
        out += [f for f in s_find if f.severity != Severity.INFO]
        if s.error or n.error:
            out.append(_pf(f"STREAM_PROBE_ERROR:{label}", Severity.SKIP, str(s.error or n.error)))
            continue
        diffs = []
        if s.text.strip() != n.text.strip():
            diffs.append(f"text {s.text.strip()[:40]!r} vs {n.text.strip()[:40]!r}")
        if s.finish_reason != n.finish_reason:
            diffs.append(f"finish {s.finish_reason!r} vs {n.finish_reason!r}")
        if _parsed_calls(s.tool_calls) != _parsed_calls(n.tool_calls):
            diffs.append(f"tool calls {_parsed_calls(s.tool_calls)} vs {_parsed_calls(n.tool_calls)}")
        if bool(s.reasoning_text.strip()) != bool(n.reasoning_text.strip()):
            diffs.append("reasoning present in only one mode")
        if diffs:
            lost = not (s.text.strip() or s.tool_calls) and bool(n.text.strip() or n.tool_calls)
            out.append(_pf(f"STREAM_DIFFERS:{label}", Severity.FAIL if lost else Severity.WARN,
                           f"streamed vs non-streamed greedy {label} reply differ: " + "; ".join(diffs),
                           Cause.RUNTIME, "the server's streaming path (parser / chunking) differs from the "
                           "non-streaming path; reproduce with stream=false and report it upstream",
                           source="wsr stream", lines=len(lines)))
        elif not any(f.failed for f in s_find):
            out.append(_pf(f"STREAM_OK:{label}", Severity.PASS,
                           f"{len(lines)} stream lines assemble into the same {label} reply", source="wsr stream"))
    return out


def _sig(n) -> tuple:
    return (n.text.strip(), bool(n.reasoning_text.strip()), n.finish_reason,
            tuple((c.get("name"), c.get("arguments")) for c in n.tool_calls))


_VISION_COLOURS = {"red": (220, 20, 20), "blue": (20, 40, 220)}


def probe_vision(ep: Endpoint) -> list[Finding]:
    """Does the model actually see an image? Two solid-colour PNGs, one-word answers.

    A server without its projector errors or the model asks for the image; a
    projector that does not match the model (or a broken image pipeline) gives
    fluent but wrong colours. Skipped when the server says it has no vision.
    """
    import base64 as _b64

    from .media import image_not_seen_score, image_part, solid_png

    mods = ep.props().get("modalities") if ep.detect_backend() == "llama-server" else None
    if isinstance(mods, dict) and mods.get("vision") is False:
        return [_pf("VISION_OFF", Severity.INFO, "the server reports no vision support (no --mmproj): image "
                    "requests will fail", fix="llama-server --mmproj <mmproj.gguf> if this model should see images")]
    q = "What colour is this image? Answer with one word."
    answers, wrong = {}, []
    for name, rgb in _VISION_COLOURS.items():
        png = solid_png(rgb)
        if ep.api == "ollama":
            msgs = [{"role": "user", "content": q, "images": [_b64.b64encode(png).decode()]}]
        else:
            msgs = [{"role": "user", "content": [image_part(png), {"type": "text", "text": q}]}]
        raw, _ = ep.chat(msgs, **_room(ep, {"temperature": 0, "max_tokens": 24}, plain=True))
        n, resp = check_response(raw, max_tokens=None)
        errs = [f for f in resp if f.code in ("IMAGE_REJECTED", "PROVIDER_ERROR")]
        if errs:
            return [_pf("VISION_UNSUPPORTED", Severity.WARN, f"the server rejects an inline PNG: {errs[0].message}",
                        Cause.CONFIG, errs[0].fix or "load the model's vision projector")]
        text = n.text.strip()
        answers[name] = text[:60]
        if image_not_seen_score(text) > 0.5:
            return [_pf("VISION_IGNORED", Severity.FAIL, f"the model says it got no image: {text[:100]!r}",
                        Cause.RUNTIME, "the server accepts images but drops them before the model: check that the "
                        "projector is loaded and that this API path forwards image parts")]
        if name not in text.lower():
            wrong.append(name)
    if len(wrong) == len(_VISION_COLOURS):
        return [_pf("VISION_WRONG", Severity.FAIL, f"the model answers but misreads plain colours: {answers}",
                    Cause.CONFIG, "the vision projector probably does not match this model (or its quant breaks "
                    "it): use the mmproj file from the same release as the model", answers=answers)]
    if wrong:
        return [_pf("VISION_WEAK", Severity.WARN, f"one of two plain colours misread: {answers}", answers=answers)]
    return [_pf("VISION_OK", Severity.PASS, f"the model reads images ({answers})")]


def probe_cache(ep: Endpoint) -> list[Finding]:
    """llama-server: a reused prompt cache must not change a greedy reply."""
    if ep.detect_backend() != "llama-server":
        return []
    msgs = [{"role": "user", "content": "Count from 1 to 15, separated by commas."}]
    base = _room(ep, {"temperature": 0, "seed": 0, "max_tokens": 96})
    fresh = [normalize(ep.chat(msgs, **base, cache_prompt=False)[0]) for _ in range(2)]
    if any(n.error for n in fresh):
        return [_pf("CACHE_PROBE_ERROR", Severity.SKIP, str(fresh[0].error or fresh[1].error))]
    if _sig(fresh[0]) != _sig(fresh[1]):
        return [_pf("NONDETERMINISTIC_FRESH", Severity.WARN, "two fresh (uncached) greedy evaluations differ",
                    Cause.RUNTIME, "nondeterministic kernels or batching: pin -np 1 / --kv-unified off to reproduce")]
    cached = []
    for _ in range(2):
        raw, _ = ep.chat(msgs, **base, cache_prompt=True)
        n = normalize(raw)
        cached.append((n, (raw.get("timings") or {}).get("cache_n") if isinstance(raw, dict) else None))
    differs = [(n, c) for n, c in cached if _sig(n) != _sig(fresh[0])]
    if differs:
        n, c = differs[0]
        what = []
        if bool(n.reasoning_text.strip()) != bool(fresh[0].reasoning_text.strip()):
            what.append(f"thinking {'on' if n.reasoning_text.strip() else 'off'} with the cache, "
                        f"{'on' if fresh[0].reasoning_text.strip() else 'off'} without")
        if n.text.strip() != fresh[0].text.strip():
            what.append(f"text {n.text.strip()[:30]!r} vs {fresh[0].text.strip()[:30]!r}")
        if n.finish_reason != fresh[0].finish_reason:
            what.append(f"finish {n.finish_reason!r} vs {fresh[0].finish_reason!r}")
        return [_pf("PROMPT_CACHE_CHANGES_OUTPUT", Severity.WARN,
                    f"the same greedy request gives a different reply when the server reuses its prompt cache "
                    f"({c} cached tokens): " + "; ".join(what or ["tool calls differ"]) +
                    ". Replies depend on what ran before, so failures come and go", Cause.RUNTIME,
                    "send cache_prompt=false for reproducible replies (slower prompt processing), or run "
                    "llama-server with --cache-reuse 0 / -np 1; report it upstream with the model name",
                    source="wsr probe", cache_n=c)]
    return [_pf("PROMPT_CACHE_OK", Severity.PASS, "reused prompt cache gives the same greedy reply")]


def server_slots(ep: Endpoint) -> Optional[int]:
    """How many requests the server decodes together (llama-server -np), where it says."""
    n = ep.props().get("total_slots") if ep.detect_backend() == "llama-server" else None
    return n if isinstance(n, int) and n > 0 else None


_FILLER = [{"role": "user", "content": "Count from 1 to 500, separated by commas."}]


def probe_concurrency(ep: Endpoint, messages: Optional[list[dict]] = None, params: Optional[dict] = None,
                      *, expect_lang: Optional[str] = None) -> list[Finding]:
    """Does a greedy reply change when other requests are decoded alongside it?

    llama-server batches the active slots into one forward pass, and the
    kernels are not batch-invariant: the same greedy request can give a
    different reply under load (measured: 3 of 6 replies on a 4-slot CPU
    server, one flipping thinking on). Then a failure seen in production comes
    and goes with traffic, and a lone rerun does not reproduce it. The request
    runs alone twice (plain nondeterminism is a different finding) and once
    while other slots are busy.
    """
    import threading

    slots = server_slots(ep)
    if slots is not None and slots < 2:
        return []
    k = min((slots or 4) - 1, 3)
    probe_msgs = messages or [{"role": "user", "content": "Name three primary colours, one line each."}]
    p = _room(ep, dict(params or {"max_tokens": 64}, temperature=0, seed=0))
    if ep.detect_backend() == "llama-server":
        p["cache_prompt"] = False
    a = normalize(ep.chat(probe_msgs, **p)[0])
    b = normalize(ep.chat(probe_msgs, **p)[0])
    if a.error or b.error:
        return [_pf("CONCURRENCY_PROBE_ERROR", Severity.SKIP, str(a.error or b.error))]
    if _sig(a) != _sig(b):
        return [_pf("NONDETERMINISTIC_GREEDY", Severity.WARN, "two identical greedy requests, one at a time, gave "
                    "different replies", Cause.RUNTIME, "disable batching / use a deterministic backend to reproduce",
                    source="Raschka checklist")]
    # Fillers long enough to stay in flight for the whole probe reply.
    fill = {"temperature": 0, "seed": 1, "max_tokens": (a.completion_tokens or 64) + 64}
    if "cache_prompt" in p:
        fill["cache_prompt"] = False
    errs: list[Exception] = []

    def _filler() -> None:
        try:
            ep.chat(_FILLER, **fill)
        except EndpointError as e:
            errs.append(e)

    ts = [threading.Thread(target=_filler, daemon=True) for _ in range(k)]
    for t in ts:
        t.start()
    time.sleep(0.5)          # let the fillers reach decoding before the probe request arrives
    c = normalize(ep.chat(probe_msgs, **p)[0])
    for t in ts:
        t.join()
    if c.error:
        return [_pf("CONCURRENCY_PROBE_ERROR", Severity.SKIP, str(c.error))]
    if _sig(c) == _sig(a):
        return [_pf("LOAD_INVARIANT", Severity.PASS, f"the greedy reply is the same alone and with {k} other "
                    "requests decoding alongside it", source="wsr probe", others=k)]
    what = []
    if bool(c.reasoning_text.strip()) != bool(a.reasoning_text.strip()):
        what.append(f"thinking {'on' if c.reasoning_text.strip() else 'off'} under load, "
                    f"{'on' if a.reasoning_text.strip() else 'off'} alone")
    ta, tc = a.text.strip(), c.text.strip()
    if ta != tc:
        i = next((j for j in range(min(len(ta), len(tc))) if ta[j] != tc[j]), min(len(ta), len(tc)))
        what.append(f"text splits at char {i}: {ta[max(0, i - 15):i + 25]!r} vs {tc[max(0, i - 15):i + 25]!r}")
    sev, verdict, changed = Severity.WARN, "", None
    if messages:
        from .verdict import _groups_of

        def _cls(n) -> set[str]:
            fs = check_output(n.text, preset="chat", finish_reason=None, expect_lang=expect_lang,
                              penalised=is_penalised(params or {}), has_images=has_images(probe_msgs, p),
                              request=next((content_text(m.get("content")) for m in reversed(probe_msgs)
                                            if m.get("role") == "user"), None))
            return _groups_of(_failing(fs))
        ga, gc = _cls(a), _cls(c)
        if ga != gc:
            sev, changed = Severity.FAIL, sorted(ga ^ gc)
            verdict = (f". The failure itself depends on load: alone {sorted(ga) or 'passes'}, under load "
                       f"{sorted(gc) or 'passes'}")
    return [_pf("LOAD_DEPENDENT_OUTPUT", sev,
                f"the same greedy request gives a different reply when {k} other requests are decoded alongside "
                f"it ({'; '.join(what or ['tool calls or finish differ'])}){verdict}. Replies in production "
                "depend on traffic, so a failure can come and go and a lone rerun may not reproduce it",
                Cause.RUNTIME, "reproduce with the server's other slots busy, or run llama-server -np 1 where "
                "exact reproducibility matters; report it upstream with the backend and build",
                source="wsr probe", others=k, slots=slots, explains=changed)]


PROBES: dict[str, Callable[[Endpoint], list[Finding]]] = {
    "reasoning": probe_reasoning, "bos": probe_bos, "smoke": probe_smoke, "stop": probe_stop,
    "finish": probe_finish_reason, "tools": probe_tools, "determinism": probe_determinism, "cache": probe_cache,
    "stream": probe_stream, "vision": probe_vision,
    "context": probe_context, "concurrency": probe_concurrency,
}
_OPT_IN = {"context", "determinism", "cache", "concurrency"}  # slow


def run_probes(ep: Endpoint, only: Optional[list[str]] = None) -> list[Finding]:
    unknown = set(only or ()) - set(PROBES)
    if unknown:
        raise ValueError(f"unknown probe(s) {sorted(unknown)}; choose from {sorted(PROBES)}")
    out: list[Finding] = []
    if only and "reasoning" not in only and ep.reasoning is None and set(only) - {"finish", "bos"}:
        try:  # later probes need to know the reasoning budget; its findings matter too
            out.extend(probe_reasoning(ep))
        except EndpointError:
            pass
    for name, fn in PROBES.items():
        if only and name not in only:
            continue
        if not only and name in _OPT_IN:
            continue
        try:
            out.extend(fn(ep))
        except EndpointError as e:
            out.append(_pf(f"{name.upper()}_PROBE_ERROR", Severity.SKIP, str(e)))
    return out


# -- differential -----------------------------------------------------------------

def _variants(messages: list[dict], params: dict) -> list[tuple[str, Cause, list[dict], dict, set[str]]]:
    """(factor, cause if it flips, messages, params, groups it can explain)."""
    v = []
    neutral = without_penalties(params)
    if neutral != params:
        # Penalties also punish closing tokens already seen (``` , }, EOS): unclosed structure,
        # or an answer that stops after one item because every next token was used.
        v.append(("penalties_neutral", Cause.SAMPLING, messages, neutral, {"loop", "garbage", "language",
                                                                          "structure", "blank"}))
    t = params.get("temperature")
    if t is None or t > 0:
        v.append(("greedy", Cause.SAMPLING, messages, dict(params, temperature=0, seed=0),
                  {"loop", "garbage", "language"}))
    else:
        v.append(("sampled", Cause.SAMPLING, messages, dict(params, temperature=0.7, top_p=0.95, seed=1),
                  {"loop"}))
    unconstrained = {k: x for k, x in params.items() if k not in ("response_format", "format", "grammar",
                                                                   "json_schema", "guided_json")}
    if unconstrained != params:
        v.append(("constraint_off", Cause.RUNTIME, messages, unconstrained, {"loop", "garbage", "structure"}))
    if any(m.get("role") == "system" for m in messages):
        v.append(("system_prompt_off", Cause.REQUEST, [m for m in messages if m.get("role") != "system"], params,
                  {"blank", "refusal", "language", "template_leak"}))
    if params.get("stop"):
        v.append(("stop_off", Cause.REQUEST, messages, {k: x for k, x in params.items() if k != "stop"},
                  {"structure", "blank"}))
    if messages and messages[-1].get("role") == "assistant":
        v.append(("prefill_off", Cause.REQUEST, messages[:-1], params, {"template_leak", "structure", "language"}))
    total = sum(len(content_text(m.get("content"))) for m in messages)
    if total > 4000 and isinstance(messages[-1].get("content"), str):
        short = [dict(m) for m in messages]
        last = short[-1]
        c = str(last.get("content", ""))
        last["content"] = c[: len(c) // 4] + "\n...\n" + c[-len(c) // 4:]
        v.append(("shorter_context", Cause.REQUEST, short, params, {"loop", "blank", "language", "structure"}))
    return v


def differential(ep: Endpoint, messages: list[dict], params: dict, failing: set[str], *,
                 alts: Optional[list[Endpoint]] = None, expect_lang: Optional[str] = None) -> list[Finding]:
    """Rerun flipping one factor at a time; report each factor that makes the failure disappear."""
    from .verdict import _groups_of  # local import to avoid a cycle

    want = _groups_of(failing)
    if not want:
        return []
    out: list[Finding] = []
    runs = [(name, cause, ep, m, p, ex) for name, cause, m, p, ex in _variants(messages, params)]
    if ep.detect_backend() == "llama-server" and params.get("cache_prompt") is not False:
        runs.append(("cache_off", Cause.RUNTIME, ep, messages, {**params, "cache_prompt": False}, set(want)))
    off = ep.no_think_params()
    already_off = all(params.get(k) == v for k, v in off.items())
    if ep.reasoning is not False and not already_off:
        runs.append(("thinking_off", Cause.REQUEST, ep, messages, {**params, **ep.no_think_params()},
                     {"blank", "loop", "structure"}))
    moved = images_as_parts(messages, params) if ep.api != "ollama" else None
    if moved:
        runs.append(("images_as_parts", Cause.REQUEST, ep, moved[0], moved[1], {"vision", "blank", "refusal"}))
    for alt in alts or []:
        runs.append((f"alt:{alt.model}@{alt.url}", Cause.QUANTISATION, alt, messages, params, set(want)))
    for name, cause, target, m, p, ex in runs:
        if not (ex & want):
            continue
        try:
            _, fs, text = run_and_classify(target, m, p, expect_lang=expect_lang)
        except EndpointError as e:
            out.append(_pf(f"DIFF_ERROR:{name}", Severity.SKIP, str(e)))
            continue
        still = _groups_of(_failing(fs)) & want
        # A variant can fix one symptom and leave another (penalties off fixes
        # the word salad, the reply still hits the cap): credit what it fixed.
        fixed = (want - still) & ex
        if fixed:
            # Confirm: one flaky run is not a cause. The original must fail again
            # and the variant must pass again.
            try:
                _, fs_orig, _ = run_and_classify(ep, messages, params, expect_lang=expect_lang)
                _, fs_again, _ = run_and_classify(target, m, p, expect_lang=expect_lang)
            except EndpointError as e:
                out.append(_pf(f"DIFF_ERROR:{name}", Severity.SKIP, str(e)))
                continue
            if not (_groups_of(_failing(fs_orig)) & fixed) or _groups_of(_failing(fs_again)) & fixed:
                out.append(_pf(f"DIFF_UNSTABLE:{name}", Severity.INFO,
                               f"'{name}' flipped once but did not reproduce: the failure comes and goes on its own "
                               "(nondeterministic server or prompt cache)", source="differential (product report)"))
                continue
            left = f"; {', '.join(sorted(still))} remains" if still else ""
            out.append(_pf(f"DIFF_FLIP:{name}", Severity.FAIL,
                           f"changing only '{name}' makes the {', '.join(sorted(fixed))} failure disappear{left}",
                           cause, _flip_fix(name), source="differential (product report)",
                           explains=sorted(fixed), sample=text[:160]))
        else:
            out.append(_pf(f"DIFF_SAME:{name}", Severity.INFO, f"'{name}': failure persists ({', '.join(sorted(still))})",
                           source="differential (product report)"))
    return out


def _shape(text: str) -> dict:
    from .output import local_recurrence, prose_tokens, script_profile

    toks = prose_tokens(text)
    prof = script_profile(" ".join(toks))
    ok = len(toks) >= 60 and prof.get("latin", 0) + prof.get("devanagari", 0) >= 0.8
    return {"lines": sum(1 for x in text.splitlines() if x.strip()), "words": len(text.split()),
            "recurrence": round(local_recurrence(toks), 3) if ok else None}


def _shape_lost(pen: dict, ref: dict) -> list[str]:
    """How the penalised reply lost the shape the unpenalised one has."""
    why = []
    if ref["lines"] >= 6 and pen["lines"] < 0.5 * ref["lines"]:
        why.append(f"{ref['lines']} lines -> {pen['lines']}")
    if ref["words"] >= 20 and pen["words"] < 0.4 * ref["words"]:
        why.append(f"{ref['words']} words -> {pen['words']}")
    if (pen["recurrence"] is not None and ref["recurrence"] is not None and ref["recurrence"] >= 0.12
            and pen["recurrence"] < 0.5 * ref["recurrence"]):
        why.append(f"word reuse {ref['recurrence']} -> {pen['recurrence']}")
    return why


def penalty_ab(ep: Endpoint, messages: list[dict], params: dict, text: str, *,
               expect_lang: Optional[str] = None) -> list[Finding]:
    """A/B the request's penalties when the reply looks fine on its own.

    An over-strong penalty can leave fluent text that has drifted off the task:
    a list that stops after four lines, an answer cut to one word, prose that
    never repeats a word. No single-reply detector sees that, but the same
    request without the penalties shows the shape the answer should have. The
    penalised side must lose shape twice (the original reply and a rerun).
    """
    neutral = without_penalties(params)
    if neutral == params:
        return []
    src = "penalty A/B (wsr)"
    try:
        _, fs_ref, ref_text = run_and_classify(ep, messages, neutral, expect_lang=expect_lang)
    except EndpointError as e:
        return [_pf("PENALTY_AB_ERROR", Severity.SKIP, str(e), source=src)]
    from .verdict import _groups_of

    if _groups_of(_failing(fs_ref)) or not ref_text.strip():
        return [_pf("PENALTY_AB_INCONCLUSIVE", Severity.INFO,
                    "the request without penalties fails too, so it is no baseline", source=src)]
    ref = _shape(ref_text)
    why = _shape_lost(_shape(text), ref)
    if not why:
        return [_pf("PENALTY_AB_OK", Severity.PASS, "the reply keeps its shape with the penalties removed",
                    source=src)]
    try:
        _, _, again = run_and_classify(ep, messages, params, expect_lang=expect_lang)
    except EndpointError as e:
        return [_pf("PENALTY_AB_ERROR", Severity.SKIP, str(e), source=src)]
    why2 = _shape_lost(_shape(again), ref)
    if not why2:
        return [_pf("DIFF_UNSTABLE:penalties_neutral", Severity.INFO,
                    f"the penalised reply lost shape once ({'; '.join(why)}) but not on a rerun", source=src)]
    sym = Finding("PENALTY_SHAPE_LOSS", Layer.OUTPUT, Severity.WARN,
                  f"the reply reads fine but has lost the shape the same request has without penalties "
                  f"({'; '.join(why)}): the penalty pushes the model off the task", cause=Cause.SYMPTOM,
                  source=src, evidence={"penalised": why, "rerun": why2, "reference_sample": ref_text[:160]})
    flip = _pf("DIFF_FLIP:penalties_neutral", Severity.FAIL,
               f"removing only the penalties restores the answer's shape ({'; '.join(why)})", Cause.SAMPLING,
               _flip_fix("penalties_neutral"), source=src, explains=["garbage"], sample=ref_text[:160])
    return [sym, flip]


def _flip_fix(name: str) -> str:
    return {
        "penalties_neutral": "drop the repeat / frequency / presence penalties (use 1.0-1.05)",
        "greedy": "sampling settings cause it: lower temperature, check min_p / top_p",
        "sampled": "greedy loops but sampling escapes: use temperature 0.6-0.8 with DRY or a mild penalty",
        "constraint_off": "constrained decoding causes it: relax the grammar / schema and validate afterwards",
        "system_prompt_off": "the system prompt causes it: check what it asks for (language, format, refusals); "
                             "if it asks for nothing like this, the template may mishandle the system role",
        "thinking_off": "thinking uses the token budget or loops: turn it off for this call "
                        "(chat_template_kwargs={'enable_thinking': false}; Ollama think=false) or raise max_tokens",
        "cache_off": "the server's prompt cache changes the reply: send cache_prompt=false (or --cache-reuse 0 / "
                     "-np 1) and report it upstream",
        "images_as_parts": "the endpoint drops the 'images' field: send each image as an image_url content part",
        "stop_off": "a stop string cuts the reply: narrow or remove the stop strings",
        "prefill_off": "the assistant prefill causes it: drop it, or strip it from the reply before parsing",
        "shorter_context": "prompt length causes it: raise the context size or shorten the prompt",
    }.get(name, "this endpoint / quantisation behaves differently: the failure comes from the backend or the "
                "quantisation, not the weights")
