"""Layer 3a: request parameters (sampling, limits, stops, constraints).

Encodes local-LLM troubleshooting knowledge as checks on the request that
produced the failure. Accepts OpenAI-style, Ollama-style (``options``) and
llama.cpp-style parameter names.
"""
from __future__ import annotations

import json
import re

from typing import Any, Optional

from .findings import Cause, Finding, Layer, Severity

SRC = "troubleshooting guides"


def _f(code: str, sev: Severity, msg: str, fix: Optional[str] = None, cause: Cause = Cause.SAMPLING,
       **ev: Any) -> Finding:
    return Finding(code=code, layer=Layer.REQUEST, severity=sev, message=msg, cause=cause, fix=fix, source=SRC,
                   evidence={k: v for k, v in ev.items() if v is not None})


def flatten_params(req: dict) -> dict[str, Any]:
    """Merge top-level, Ollama ``options`` and alias names into one dict."""
    p: dict[str, Any] = {}
    p.update({k: v for k, v in req.items() if not isinstance(v, (list, dict)) or k in ("stop", "response_format",
                                                                                         "format", "tools")})
    if isinstance(req.get("options"), dict):
        p.update(req["options"])
    aliases = {
        "repetition_penalty": "repeat_penalty", "max_completion_tokens": "max_tokens",
        "num_predict": "max_tokens", "n_predict": "max_tokens", "max_new_tokens": "max_tokens",
        "max_output_tokens": "max_tokens", "n_ctx": "num_ctx", "ctx_size": "num_ctx",
    }
    for a, b in aliases.items():
        if a in p and b not in p:
            p[b] = p[a]
    return p


_THINK_MARKUP = re.compile(r"</?think>|</?thinking>|<\|channel\|?>|<channel\|>", re.I)


def check_prefill(messages: Optional[list], *, backend: Optional[str] = None,
                  template: Optional[str] = None) -> list[Finding]:
    """A trailing assistant message is a prefill: the model continues it."""
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        return []
    c = messages[-1].get("content")
    c = c if isinstance(c, str) else json.dumps(c)
    echo = backend == "llama-server"
    out = []
    m = _THINK_MARKUP.search(c)
    if m:
        foreign = template is not None and m.group(0).lower() not in template.lower()
        out.append(_f("PREFILL_REASONING_MARKUP", Severity.WARN,
                      f"the request prefills the answer with {c[:40]!r} to skip thinking"
                      + (f"; this model's template does not use {m.group(0)!r}" if foreign else "")
                      + ("; llama-server returns the prefill at the start of content, so the answer starts with "
                         "that markup" if echo else "; servers that echo prefills put it in the answer"),
                      "turn thinking off with the template switch instead (chat_template_kwargs="
                      "{'enable_thinking': false}), or strip the prefill from the reply before parsing",
                      cause=Cause.REQUEST, prefill=c[:200], foreign_markup=foreign or None))
    elif echo:
        out.append(_f("PREFILL", Severity.INFO, f"the answer is prefilled with {c[:40]!r}; llama-server "
                      "returns it as part of content", cause=Cause.REQUEST))
    return out


def check_request(req: dict, *, backend: Optional[str] = None, prompt_tokens: Optional[int] = None,
                  ctx_trained: Optional[int] = None, messages: Optional[list] = None,
                  template: Optional[str] = None) -> list[Finding]:
    p = flatten_params(req)
    out: list[Finding] = check_prefill(messages or req.get("messages"), backend=backend, template=template)

    rp = p.get("repeat_penalty")
    if isinstance(rp, (int, float)):
        if rp > 1.2:
            out.append(_f("REPEAT_PENALTY_HIGH", Severity.WARN,
                          f"repeat penalty {rp} distorts the distribution and can make loops worse",
                          "use 1.0-1.05 and rely on DRY / lower temperature", repeat_penalty=rp))
        elif rp < 1.0:
            out.append(_f("REPEAT_PENALTY_BELOW_ONE", Severity.WARN,
                          f"repeat penalty {rp} < 1 rewards repetition", "set it to 1.0-1.05", repeat_penalty=rp))
    elif backend == "llama-server":
        out.append(_f("NO_REPEAT_CONTROL", Severity.INFO,
                      "llama-server applies no repeat penalty or DRY by default",
                      "set dry_multiplier 0.8 if loops appear"))

    t = p.get("temperature")
    if isinstance(t, (int, float)):
        if t > 1.5:
            out.append(_f("TEMPERATURE_HIGH", Severity.WARN, f"temperature {t} produces incoherent text on small models",
                          "use 0.6-1.0", temperature=t))
        if t == 0:
            out.append(_f("GREEDY", Severity.INFO, "greedy decoding (temperature 0): loops here are "
                          "deterministic; compare with sampling", temperature=t))

    mp = p.get("min_p")
    if isinstance(mp, (int, float)) and mp > 0.3:
        out.append(_f("MIN_P_HIGH", Severity.WARN, f"min_p {mp} leaves almost no candidates", "use 0.05-0.1"))
    tp = p.get("top_p")
    if isinstance(tp, (int, float)) and 0 < tp < 0.2:
        out.append(_f("TOP_P_LOW", Severity.INFO, f"top_p {tp} is close to greedy"))
    for k in ("frequency_penalty", "presence_penalty"):
        v = p.get(k)
        if isinstance(v, (int, float)) and abs(v) > 1.5:
            out.append(_f(f"{k.upper()}_EXTREME", Severity.WARN, f"{k} {v} pushes the model off-distribution",
                          "keep within ±1"))

    mt = p.get("max_tokens")
    if isinstance(mt, int) and 0 < mt < 32:
        out.append(_f("MAX_TOKENS_TINY", Severity.WARN, f"max_tokens {mt} leaves no room for an answer",
                      "raise max_tokens", cause=Cause.REQUEST, max_tokens=mt))

    stops = p.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    if isinstance(stops, list):
        risky = [s for s in stops if isinstance(s, str) and s.strip() == "" or s in ("\n", ".", "\n\n")]
        if risky:
            out.append(_f("STOP_TOO_BROAD", Severity.WARN,
                          f"stop strings {risky!r} can end the answer at its first line, giving blank or cut output",
                          "remove broad stop strings; stop on the model's end-of-turn token instead",
                          cause=Cause.REQUEST))

    if backend == "ollama":
        if "num_ctx" not in p:
            out.append(_f("OLLAMA_DEFAULT_CONTEXT", Severity.WARN,
                          "num_ctx not set: Ollama uses its default context and silently drops overflow",
                          "pass options.num_ctx sized to your prompt + answer", cause=Cause.REQUEST))
        if rp is not None or "frequency_penalty" in p or "presence_penalty" in p:
            out.append(_f("OLLAMA_PENALTY_MAY_BE_IGNORED", Severity.INFO,
                          "Ollama's newer runner ignores repeat/frequency/presence penalties for some models "
                          "(ollama#15783)", "use temperature / min_p, or test the same model in llama.cpp"))

    nctx = p.get("num_ctx")
    if isinstance(nctx, int) and ctx_trained and nctx > ctx_trained:
        out.append(_f("CONTEXT_ABOVE_TRAINED", Severity.WARN,
                      f"context {nctx} exceeds the model's trained {ctx_trained} without rope scaling",
                      f"use at most {ctx_trained}", cause=Cause.CONFIG))
    if isinstance(nctx, int) and prompt_tokens and prompt_tokens > nctx:
        out.append(_f("PROMPT_EXCEEDS_CONTEXT", Severity.FAIL,
                      f"prompt is ~{prompt_tokens} tokens but the context is {nctx}",
                      "raise num_ctx or shorten the prompt", cause=Cause.REQUEST))

    fmt = p.get("format") or p.get("response_format") or p.get("grammar") or p.get("json_schema")
    if fmt:
        out.append(_f("CONSTRAINED_DECODING", Severity.INFO,
                      "grammar / JSON-constrained decoding is on: some models loop only under constraints "
                      "(ollama#15502); the differential stage reruns without it", cause=Cause.RUNTIME))
    return out
