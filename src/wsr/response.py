"""Layer 3: request / response checks.

Normalises a provider response (OpenAI-compatible, Anthropic Messages, Gemini,
Ollama native chat/generate, error bodies) and classifies it from provider
fields, which are ground truth when present.

The finish-reason taxonomy and verdict codes follow llm-stopcheck
(c) tonyweckstrom1987, MIT licence; see NOTICE. Native Anthropic / Gemini /
Ollama shapes, input-truncation detection from server token counts (the
LLMDoctor technique) and tool-argument schema validation are additions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .findings import Cause, Finding, Layer, Severity
from .jsonschema_lite import validate as validate_schema

SRC_SC = "llm-stopcheck"

TRUNCATED_REASONS = {
    "length", "max_tokens", "max_output_tokens", "model_length", "max_tokens_exceeded",
    "model_context_window_exceeded", "context_length_exceeded",
}
FILTERED_REASONS = {
    "content_filter", "safety", "recitation", "blocklist", "prohibited_content", "spii",
    "image_safety", "language",
}
REFUSAL_REASONS = {"refusal"}
OK_REASONS = {"stop", "end_turn", "stop_sequence", "tool_calls", "tool_use", "function_call", "pause_turn"}


@dataclass
class NormalizedResponse:
    provider: str
    text: str = ""
    reasoning_text: str = ""
    finish_reason: Optional[str] = None
    native_finish_reason: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {name, arguments(raw)}
    refusal: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    error: Optional[dict[str, Any]] = None
    malformed: Optional[str] = None


def _norm(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower().replace("-", "_")
    return s or None


def _int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    return None


def _first_int(*vs: Any) -> Optional[int]:
    """The first value that is an int (0 counts: a reply with no tokens is not an unknown count)."""
    for v in vs:
        if _int(v) is not None:
            return _int(v)
    return None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else (p.get("text") or "") for p in content
                       if isinstance(p, (str, dict)) and (isinstance(p, str) or p.get("type", "text") == "text"))
    return ""


def _as_dict(obj: Any) -> Optional[dict]:
    if isinstance(obj, dict):
        return obj
    for name in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                d = fn()
            except Exception:  # noqa: BLE001
                continue
            if isinstance(d, dict):
                return d
    return None


def normalize(response: Any) -> NormalizedResponse:
    d = _as_dict(response)
    if d is None:
        return NormalizedResponse("unknown", malformed="response is not a JSON object")

    if "error" in d and not d.get("choices"):
        err = d["error"] if isinstance(d["error"], dict) else {"message": str(d["error"])}
        return NormalizedResponse("error", error=err)

    # OpenAI-compatible (OpenAI, vLLM, llama-server, LM Studio, Ollama /v1, LiteLLM, OpenRouter)
    if "choices" in d:
        ch = d["choices"]
        if not isinstance(ch, list) or not ch or not isinstance(ch[0], dict):
            return NormalizedResponse("openai", malformed="response has no choices")
        c = ch[0]
        msg = c.get("message") if isinstance(c.get("message"), dict) else {}
        if not msg and "text" in c:  # legacy /v1/completions
            msg = {"content": c.get("text")}
        native = _norm(c.get("native_finish_reason"))
        if native is None and isinstance(c.get("provider_specific_fields"), dict):
            native = _norm(c["provider_specific_fields"].get("native_finish_reason"))
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        if msg.get("function_call"):
            fc = msg["function_call"]
            calls.append({"name": fc.get("name"), "arguments": fc.get("arguments")})
        usage = d.get("usage") if isinstance(d.get("usage"), dict) else {}
        det = usage.get("completion_tokens_details")
        rt = _int(det.get("reasoning_tokens")) if isinstance(det, dict) else None
        return NormalizedResponse(
            "openai",
            text=_text(msg.get("content")),
            reasoning_text=_text(msg.get("reasoning_content")) or _text(msg.get("reasoning")),
            finish_reason=_norm(c.get("finish_reason")),
            native_finish_reason=native,
            tool_calls=calls,
            refusal=msg.get("refusal") if isinstance(msg.get("refusal"), str) else None,
            prompt_tokens=_int(usage.get("prompt_tokens")),
            completion_tokens=_first_int(usage.get("completion_tokens"), usage.get("output_tokens")),
            reasoning_tokens=rt if rt is not None else _int(usage.get("reasoning_tokens")),
        )

    # Anthropic Messages
    if d.get("type") == "message" or ("stop_reason" in d and isinstance(d.get("content"), list)):
        text, thinking, calls = [], [], []
        for b in d.get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                text.append(b.get("text") or "")
            elif b.get("type") in ("thinking", "redacted_thinking"):
                thinking.append(b.get("thinking") or "")
            elif b.get("type") == "tool_use":
                calls.append({"name": b.get("name"), "arguments": b.get("input")})
        usage = d.get("usage") or {}
        return NormalizedResponse(
            "anthropic", text="".join(text), reasoning_text="".join(thinking),
            finish_reason=_norm(d.get("stop_reason")), tool_calls=calls,
            prompt_tokens=_int(usage.get("input_tokens")),
            completion_tokens=_int(usage.get("output_tokens")),
        )

    # Gemini generateContent
    if "candidates" in d or "promptFeedback" in d:
        cands = d.get("candidates") or []
        usage = d.get("usageMetadata") or {}
        block = (d.get("promptFeedback") or {}).get("blockReason")
        if not cands:
            return NormalizedResponse("gemini", finish_reason="safety" if block else None,
                                      native_finish_reason=_norm(block),
                                      prompt_tokens=_int(usage.get("promptTokenCount")),
                                      malformed=None if block else "no candidates")
        c = cands[0]
        text, thinking, calls = [], [], []
        for part in (c.get("content") or {}).get("parts") or []:
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append({"name": fc.get("name"), "arguments": fc.get("args")})
            elif part.get("thought"):
                thinking.append(part.get("text") or "")
            elif "text" in part:
                text.append(part["text"])
        fr = _norm(c.get("finishReason"))
        return NormalizedResponse(
            "gemini", text="".join(text), reasoning_text="".join(thinking),
            finish_reason="max_tokens" if fr == "max_tokens" else fr, tool_calls=calls,
            prompt_tokens=_int(usage.get("promptTokenCount")),
            completion_tokens=_int(usage.get("candidatesTokenCount")),
            reasoning_tokens=_int(usage.get("thoughtsTokenCount")),
        )

    # Ollama native /api/chat and /api/generate
    if "done" in d or "done_reason" in d or "prompt_eval_count" in d:
        msg = d.get("message") if isinstance(d.get("message"), dict) else {}
        text = msg.get("content") if msg else d.get("response", "")
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        fr = _norm(d.get("done_reason"))
        if fr is None and d.get("done") is True:
            fr = "stop"
        return NormalizedResponse(
            "ollama", text=text or "", reasoning_text=(msg.get("thinking") or d.get("thinking") or ""),
            finish_reason=fr, tool_calls=calls,
            prompt_tokens=_int(d.get("prompt_eval_count")), completion_tokens=_int(d.get("eval_count")),
        )

    return NormalizedResponse("unknown", malformed="unrecognised response shape")


def _finding(code: str, sev: Severity, msg: str, cause: Cause, fix: Optional[str] = None,
             source: str = SRC_SC, **evidence: Any) -> Finding:
    return Finding(code=code, layer=Layer.RESPONSE, severity=sev, message=msg, cause=cause, fix=fix,
                   source=source, evidence={k: v for k, v in evidence.items() if v is not None})


def check_response(response: Any, *, max_tokens: Optional[int] = None, near_limit: float = 0.95,
                   require_text: bool = True, sent_prompt_tokens: Optional[int] = None,
                   context_window: Optional[int] = None, tools: Optional[Sequence[dict]] = None
                   ) -> tuple[NormalizedResponse, list[Finding]]:
    """Classify a response. Never raises."""
    n = normalize(response)
    out: list[Finding] = []
    ev = dict(provider=n.provider, finish_reason=n.finish_reason, native_finish_reason=n.native_finish_reason,
              completion_tokens=n.completion_tokens, reasoning_tokens=n.reasoning_tokens, max_tokens=max_tokens)

    if n.error is not None:
        msg = str(n.error.get("message") or n.error)
        code = str(n.error.get("code") or n.error.get("type") or "")
        low = (msg + code).lower()
        if "rate" in low or "429" in low or "quota" in low:
            out.append(_finding("RATE_LIMITED", Severity.FAIL, f"provider rate limit: {msg}", Cause.PROVIDER,
                                "back off and retry; check your quota", source="Datadog error data"))
        elif "context" in low and ("length" in low or "window" in low or "exceed" in low):
            out.append(_finding("CONTEXT_OVERFLOW", Severity.FAIL, f"prompt exceeds context: {msg}", Cause.REQUEST,
                                "shorten the prompt or raise the server's context size (num_ctx / -c / max_model_len)"))
        elif any(k in low for k in ("image", "base64", "mmproj", "multimodal", "audio file", "get request", "download")):
            out.append(_image_error(msg, low))
        elif "does not match the expected" in low or "failed to parse" in low and "output" in low:
            # llama-server's chat parser rejects what the model generated: the output itself is malformed.
            out.append(_finding("OUTPUT_PARSE_REJECTED", Severity.FAIL,
                                f"the server's output parser rejected the generated text: {msg}", Cause.SYMPTOM,
                                "the raw generation is malformed (garbage, broken tool call or channel markup): "
                                "rerun greedy; if greedy also fails, the template / parser does not match the model"))
        else:
            out.append(_finding("PROVIDER_ERROR", Severity.FAIL, f"error body: {msg}", Cause.PROVIDER, "retry"))
        return n, out

    if n.malformed:
        out.append(_finding("MALFORMED_RESPONSE", Severity.FAIL, n.malformed, Cause.RUNTIME, "retry", **ev))
        return n, out

    f, nat = n.finish_reason, n.native_finish_reason
    raise_fix = (f"raise max_tokens (currently {max_tokens})" if max_tokens else "raise max_tokens")

    if f is None:
        out.append(_finding("MISSING_FINISH_REASON", Severity.FAIL,
                            "no finish reason: the stream may have ended early", Cause.RUNTIME, "retry", **ev))
    elif f in TRUNCATED_REASONS or (nat in TRUNCATED_REASONS):
        mismatch = "" if f in TRUNCATED_REASONS else f" (finish_reason {f!r} but native {nat!r})"
        if f in ("model_context_window_exceeded", "context_length_exceeded"):
            out.append(_finding("CONTEXT_OVERFLOW", Severity.FAIL, "generation hit the context window" + mismatch,
                                Cause.REQUEST, "shorten the prompt or raise the context size", **ev))
        elif n.tool_calls:
            out.append(_finding("TOOL_CALL_TRUNCATED", Severity.FAIL,
                                "output cap reached while writing a tool call" + mismatch, Cause.REQUEST, raise_fix, **ev))
        elif not n.text.strip():
            if n.reasoning_tokens or n.reasoning_text.strip():
                out.append(_finding("REASONING_EXHAUSTED", Severity.FAIL,
                                    "cap reached before any answer: the budget went to reasoning" + mismatch,
                                    Cause.REQUEST, raise_fix + " or lower the reasoning effort / thinking budget", **ev))
            else:
                out.append(_finding("TRUNCATED_EMPTY", Severity.FAIL, "cap reached with empty content" + mismatch,
                                    Cause.REQUEST, raise_fix, **ev))
        else:
            out.append(_finding("TRUNCATED", Severity.FAIL, "output cap reached; answer is cut off" + mismatch,
                                Cause.REQUEST, raise_fix, **ev))
    elif f in FILTERED_REASONS or nat in FILTERED_REASONS:
        out.append(_finding("CONTENT_FILTERED", Severity.FAIL, "blocked by a provider content filter",
                            Cause.PROVIDER, "rephrase, or adjust the provider's safety settings", **ev))
    elif f in REFUSAL_REASONS or (n.refusal and n.refusal.strip()):
        out.append(_finding("REFUSAL", Severity.FAIL, "provider-reported refusal", Cause.PROVIDER, None, **ev))
    elif f not in OK_REASONS:
        out.append(_finding("UNKNOWN_FINISH_REASON", Severity.FAIL, f"unrecognised finish reason {f!r}",
                            Cause.RUNTIME, "retry; check the server version", **ev))
    elif require_text and not n.tool_calls and not n.text.strip():
        hint = None
        if n.reasoning_text.strip() or n.reasoning_tokens:
            hint = "the model produced reasoning but no answer; check the reasoning parser / template"
        out.append(_finding("EMPTY_OUTPUT", Severity.FAIL, f"finish reason {f!r} but content is empty",
                            Cause.SYMPTOM, hint, **ev))
    else:
        out.append(_finding("FINISH_OK", Severity.PASS, f"finished with {f!r}", Cause.NONE, **ev))
        if max_tokens and n.completion_tokens is not None:
            if n.completion_tokens >= max_tokens:
                out.append(_finding("AT_LIMIT", Severity.WARN, "completion used exactly max_tokens", Cause.REQUEST,
                                    raise_fix, **ev))
            elif n.completion_tokens >= near_limit * max_tokens:
                out.append(_finding("NEAR_LIMIT", Severity.WARN,
                                    f"completion used {n.completion_tokens}/{max_tokens} tokens", Cause.REQUEST,
                                    raise_fix, **ev))

    # Input truncation: the server's own count is the signal (LLMDoctor).
    if sent_prompt_tokens and n.prompt_tokens is not None:
        ratio = n.prompt_tokens / sent_prompt_tokens
        if ratio < 0.9:
            caveat = (" Ollama reports only newly evaluated tokens, so a prefix-cache hit also lowers this; "
                      "rerun with a fresh prompt to confirm." if n.provider == "ollama" else "")
            out.append(Finding(
                "INPUT_TRUNCATED", Layer.RESPONSE, Severity.FAIL if n.provider != "ollama" else Severity.WARN,
                f"server read {n.prompt_tokens} of ~{sent_prompt_tokens} prompt tokens ({ratio:.0%}).{caveat}",
                Cause.REQUEST, evidence={"prompt_tokens": n.prompt_tokens, "sent": sent_prompt_tokens},
                fix="raise the server context size (Ollama: options.num_ctx; llama-server: -c; vLLM: --max-model-len)",
                source="LLMDoctor"))
    if context_window and n.prompt_tokens is not None and n.prompt_tokens >= 0.98 * context_window:
        out.append(Finding("PROMPT_FILLS_CONTEXT", Layer.RESPONSE, Severity.WARN,
                           f"prompt uses {n.prompt_tokens}/{context_window} tokens; input was likely cut",
                           Cause.REQUEST, fix="raise the context size or shorten the prompt", source="LLMDoctor"))

    # Tool calls: arguments must parse and match the declared schema.
    if n.tool_calls:
        declared = {}
        for t in tools or []:
            fn = t.get("function", t)
            declared[fn.get("name")] = fn.get("parameters") or fn.get("input_schema") or {}
        for i, call in enumerate(n.tool_calls):
            args = call.get("arguments")
            name = call.get("name")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError as e:
                    out.append(Finding("TOOL_ARGS_UNPARSEABLE", Layer.RESPONSE, Severity.FAIL,
                                       f"tool call {i} ({name}) arguments are not JSON: {e}", Cause.SYMPTOM,
                                       evidence={"arguments": args[:200]}, source="LLMDoctor",
                                       fix="check the server's tool-call parser and chat template"))
                    continue
            if declared and name not in declared:
                out.append(Finding("TOOL_UNKNOWN", Layer.RESPONSE, Severity.FAIL,
                                   f"tool call {i} names {name!r}, not one of {sorted(declared)}", Cause.SYMPTOM,
                                   source="LLMDoctor", fix="stale template or tool parser often causes this"))
                continue
            if name in declared:
                errs = validate_schema(args, declared[name])
                if errs:
                    out.append(Finding("TOOL_ARGS_INVALID", Layer.RESPONSE, Severity.FAIL,
                                       f"tool call {i} ({name}) violates schema: {errs[0]}", Cause.SYMPTOM,
                                       evidence={"errors": errs[:10]}, source="LLMDoctor"))
    return n, out


def _image_error(msg: str, low: str) -> Finding:
    cause, fix = Cause.REQUEST, "check how the request sends the image"
    if "download" in low or "fetch" in low:
        fix = "the server could not fetch the image URL: inline it as a base64 data URI"
    elif "invalid base64" in low:
        fix = "image_url must be an object {'url': 'data:image/png;base64,...'} holding valid base64"
    elif "failed to load" in low or "decode" in low:
        fix = "the bytes are not a decodable image: re-encode the file (PNG / JPEG / WebP)"
    elif "not supported" in low or "mmproj" in low or "multimodal" in low or "not enabled" in low:
        cause, fix = Cause.CONFIG, ("the server has no vision projector: llama-server --mmproj <mmproj.gguf>, "
                                    "or a multimodal model tag on Ollama")
    return _finding("IMAGE_REJECTED", Severity.FAIL, f"the server rejected the image: {msg}", cause, fix)
