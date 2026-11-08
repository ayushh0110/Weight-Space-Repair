"""Streaming validation: assemble SSE / NDJSON streams and check the stream itself.

A streamed reply can fail in ways the final text does not show:

* the connection ends before the terminal chunk (no finish_reason, no [DONE]):
  the answer was cut mid-stream, often with no error anywhere;
* a chunk is not valid JSON, or the server sends an error event mid-stream;
* tool-call fragments do not assemble: argument deltas that never become valid
  JSON, a call with no name, an id that changes between fragments, index gaps;
* content arrives after the finish chunk.

The assembled reply is returned in the non-streaming OpenAI shape so the
response and output layers judge it like any other reply. Accepts OpenAI-style
SSE (``data: {...}`` / ``data: [DONE]``) and Ollama NDJSON.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from .findings import Cause, Finding, Layer, Severity

SRC = "wsr stream"


def _f(code: str, sev: Severity, msg: str, cause: Cause = Cause.RUNTIME, fix: Optional[str] = None,
       **ev: Any) -> Finding:
    return Finding(code=code, layer=Layer.RESPONSE, severity=sev, message=msg, cause=cause, fix=fix, source=SRC,
                   evidence={k: v for k, v in ev.items() if v is not None})


def looks_like_stream(text: str) -> bool:
    head = text.lstrip()[:200]
    if head.startswith(("data:", "event:", ":")):
        return True
    lines = [x for x in text.strip().splitlines()[:3] if x.strip()]
    if len(lines) >= 2:
        try:
            return all(isinstance(json.loads(x), dict) and ("done" in json.loads(x) or "message" in json.loads(x))
                       for x in lines)
        except ValueError:
            return False
    return False


def _events(lines: Iterable[str]) -> Iterable[tuple[str, str]]:
    """(kind, payload) with kind 'data' | 'done' | 'raw'."""
    for line in lines:
        line = line.rstrip("\r\n")
        if not line.strip() or line.startswith(":") or line.startswith(("event:", "id:", "retry:")):
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
            yield ("done", "") if payload == "[DONE]" else ("data", payload)
        else:
            yield "raw", line.strip()  # NDJSON (Ollama) or junk


def assemble(lines: Iterable[str]) -> tuple[dict, list[Finding]]:
    out: list[Finding] = []
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    order: list[int] = []
    finish: Optional[str] = None
    usage: dict = {}
    model = None
    saw_done = ollama = False
    n_chunks = bad = 0
    after_finish = 0
    error: Optional[dict] = None
    for kind, payload in _events(lines):
        if kind == "done":
            saw_done = True
            continue
        try:
            d = json.loads(payload)
        except ValueError:
            bad += 1
            continue
        if not isinstance(d, dict):
            bad += 1
            continue
        n_chunks += 1
        if "error" in d and not d.get("choices"):
            error = d["error"] if isinstance(d["error"], dict) else {"message": str(d["error"])}
            continue
        model = model or d.get("model")
        if kind == "raw" or "message" in d and "choices" not in d:  # Ollama NDJSON
            ollama = True
            m = d.get("message") or {}
            if finish and (m.get("content") or m.get("thinking")):
                after_finish += 1
            content.append(m.get("content") or "")
            reasoning.append(m.get("thinking") or "")
            for i, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                calls[len(calls)] = {"id": tc.get("id"), "name": fn.get("name"),
                                     "args": [args if isinstance(args, str) else json.dumps(args)], "ids": set()}
                order.append(len(calls) - 1)
            if d.get("done"):
                saw_done = True
                finish = d.get("done_reason") or "stop"
                usage = {"prompt_tokens": d.get("prompt_eval_count"), "completion_tokens": d.get("eval_count")}
            continue
        if isinstance(d.get("usage"), dict):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or ch.get("message") or {}
            if finish and (delta.get("content") or delta.get("reasoning_content")):
                after_finish += 1
            if delta.get("content"):
                content.append(delta["content"])
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                reasoning.append(r)
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", len(order) and order[-1])
                c = calls.get(idx)
                if c is None:
                    c = calls[idx] = {"id": None, "name": None, "args": [], "ids": set()}
                    order.append(idx)
                if tc.get("id"):
                    c["ids"].add(tc["id"])
                    c["id"] = c["id"] or tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    same = not c["name"] or c["name"] == fn["name"]
                    c["name"] = fn["name"] if same else c["name"] + fn["name"]  # some servers stream the name
                if fn.get("arguments"):
                    c["args"].append(fn["arguments"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]

    ev = dict(chunks=n_chunks, malformed=bad or None)
    if bad:
        out.append(_f("STREAM_MALFORMED_CHUNK", Severity.FAIL, f"{bad} stream chunk(s) are not valid JSON",
                      Cause.PROVIDER, "check the server / proxy: chunks are being split or corrupted", **ev))
    if error:
        out.append(_f("STREAM_ERROR_EVENT", Severity.FAIL,
                      f"error event mid-stream after {len(''.join(content))} chars: {error.get('message', error)}",
                      Cause.PROVIDER, "the server failed during generation; see its log", **ev))
    if finish is None and not error:
        out.append(_f("STREAM_CUT", Severity.FAIL,
                      f"stream ended after {n_chunks} chunk(s) and {len(''.join(content))} chars with no "
                      "finish_reason" + ("" if saw_done else " and no [DONE]") +
                      ": the connection dropped mid-answer", Cause.RUNTIME,
                      "check client / proxy timeouts and the server log; a cut stream looks like a short answer",
                      **ev))
    elif not saw_done and not ollama:
        out.append(_f("STREAM_NO_DONE", Severity.INFO, "finish_reason received but no [DONE] terminator", **ev))
    if after_finish:
        out.append(_f("STREAM_CONTENT_AFTER_FINISH", Severity.WARN,
                      f"{after_finish} chunk(s) carried text after the finish chunk; clients that stop at "
                      "finish_reason lose it", Cause.RUNTIME, **ev))

    tool_calls = []
    if order and sorted(order) != list(range(min(order), min(order) + len(order))):
        out.append(_f("STREAM_TOOL_INDEX_GAP", Severity.WARN, f"tool-call indexes {sorted(order)} have gaps",
                      Cause.RUNTIME))
    for idx in order:
        c = calls[idx]
        args = "".join(c["args"])
        if not c["name"]:
            out.append(_f("STREAM_TOOL_NO_NAME", Severity.FAIL, f"tool call #{idx} never received a name",
                          Cause.RUNTIME, "the server's streaming tool-call parser is broken; try stream=false"))
        if len(c["ids"]) > 1:
            out.append(_f("STREAM_TOOL_ID_CHANGED", Severity.WARN,
                          f"tool call #{idx} changed id mid-stream ({sorted(c['ids'])})", Cause.RUNTIME))
        if args.strip():
            try:
                json.loads(args)
            except ValueError as e:
                out.append(_f("STREAM_TOOL_ARGS_FRAGMENTED", Severity.FAIL,
                              f"tool call #{idx} ({c['name']}): {len(c['args'])} argument fragments do not "
                              f"assemble into JSON ({e})", Cause.RUNTIME,
                              "the streaming tool-call parser drops or duplicates fragments; compare with "
                              "stream=false", fragments=c["args"][:20]))
        tool_calls.append({"id": c["id"], "type": "function",
                           "function": {"name": c["name"], "arguments": args}})

    msg: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if reasoning and "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    raw: dict[str, Any] = {"object": "chat.completion", "model": model,
                           "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                           "usage": {k: v for k, v in usage.items() if v is not None}}
    if error and not content:
        raw = {"error": error}
    return raw, out
