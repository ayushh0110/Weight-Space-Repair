"""Model providers for the surgery agent: one turn in, tool calls out.

A provider gets the system prompt, the conversation so far and the tool schemas, and returns the
assistant turn: its text, its tool calls, the raw content to append to the history, and token
usage. The loop (loop.py) owns everything else, so a provider can be swapped without touching
the tools or the checks.

* AnthropicProvider - Claude through the official SDK (`pip install anthropic`); credentials from
  ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile), never stored by wsr.
  Default model claude-opus-5-5, adaptive thinking, effort "high", and server-side refusal
  fallbacks on (`fallbacks: "default"`), so a safety-classifier decline is retried on the
  recommended fallback model instead of ending the run.
* OpenAIProvider - any model behind the OpenAI Chat Completions API (`pip install openai`): OpenAI,
  or any OpenAI-compatible server via `base_url` (OpenRouter, Together, Groq, DeepSeek, Mistral,
  Gemini's OpenAI endpoint, local vLLM / Ollama / llama-server); key from an environment variable.
* ScriptedProvider - replays a fixed list of tool calls (tests, demos, dry runs).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

# Claude Opus 5.5 list price, USD per million tokens (input, output), for the spend cap.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
          "claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0), "claude-mythos-5-1": (10.0, 50.0),
          "claude-mythos-5": (10.0, 50.0)}
# A model this table does not know is counted at the dearest price in it, so the spend cap stops early, not late.
UNKNOWN_PRICE = max(PRICES.values())
# Cache reads as a share of the input price (writes cost 1.25x for the 5-minute TTL).
CACHE_READ = {"claude-opus-5-5": 0.05, "claude-fable-5-1": 0.025, "claude-mythos-5-1": 0.025}


def anthropic_caches(model: str) -> bool:
    """Prompt caching on the Claude API: every Claude 3 and later model. A prefix below the model's minimum
    simply is not cached (no error); a model that rejects it anyway turns caching off for the run."""
    m = model.lower()
    return m.startswith("claude-") and not m.startswith(("claude-2", "claude-instant", "claude-1"))


def openai_cache_control(model: str, base_url: Optional[str]) -> bool:
    """Explicit cache_control breakpoints on an OpenAI-compatible server: only where the server documents
    them (OpenRouter, for Anthropic and Gemini models). OpenAI itself, DeepSeek and most others cache
    repeated prefixes automatically and would reject or ignore the field, so it is not sent there."""
    if not base_url or "openrouter.ai" not in base_url:
        return False
    return model.lower().startswith(("anthropic/", "google/gemini"))


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Turn:
    text: str
    calls: list[ToolCall]
    content: Any                     # what to append to the history as the assistant message
    stop: str                        # end_turn | tool_use | max_tokens | refusal | ...
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Provider(Protocol):
    name: str

    def turn(self, system: str, messages: list[dict], tools: list[dict]) -> Turn: ...


class AnthropicProvider:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 16000,
                 api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as e:
            raise ValueError("the surgery agent needs the Anthropic SDK: pip install anthropic") from e
        self._anthropic = anthropic
        # The SDK resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant` profile itself.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.name = f"anthropic:{model}"
        self.caching = anthropic_caches(model)
        self.cached_tokens = 0

    def _create(self, system: str, messages: list[dict], tools: list[dict]):
        kw: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, messages=messages, tools=tools,
                                  thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                                  betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"})
        if self.caching:
            # The system prompt and tools never change within a run: an explicit breakpoint on the system block
            # caches both; automatic caching follows the growing conversation.
            kw["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            kw["cache_control"] = {"type": "ephemeral"}
        else:
            kw["system"] = system
        return self.client.beta.messages.create(**kw)

    def turn(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        a = self._anthropic
        try:
            try:
                r = self._create(system, messages, tools)
            except a.BadRequestError as e:
                if not self.caching or "cache" not in str(getattr(e, "message", e)).lower():
                    raise
                self.caching = False                   # this model or route does not take it: run on without
                r = self._create(system, messages, tools)
        except a.AuthenticationError as e:
            raise ValueError("Anthropic API key rejected: set ANTHROPIC_API_KEY to a valid key") from e
        except a.BadRequestError as e:
            raise ValueError(f"Anthropic API rejected the request: {e.message}") from e
        calls = [ToolCall(b.id, b.name, dict(b.input or {})) for b in r.content if b.type == "tool_use"]
        text = "".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        created = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        self.cached_tokens += read
        tin = (u.input_tokens or 0) + created * 1.25 + read * CACHE_READ.get(self.model, 0.1)
        pin, pout = PRICES.get(self.model, UNKNOWN_PRICE)
        # Tokens for the run's token cap: the whole prompt (uncached, written and read), not only the uncached part.
        return Turn(text, calls, r.content, r.stop_reason or "", int(u.input_tokens or 0) + created + read,
                    int(u.output_tokens or 0), (tin * pin + (u.output_tokens or 0) * pout) / 1e6)


class OpenAIProvider:
    """Any model behind the OpenAI Chat Completions API, through the official SDK (`pip install openai`):
    OpenAI itself, or, with `base_url`, an OpenAI-compatible server (OpenRouter, Together, Groq, Fireworks,
    DeepSeek, Mistral, Gemini's OpenAI endpoint, a local vLLM / Ollama / llama-server).

    The loop's history is kept in one neutral form (system prompt; user text; assistant text + tool_use
    blocks; user tool_result blocks); this adapter translates it to chat messages on every turn, so the
    loop, the tools and the checks do not depend on the provider. The key is read from the environment
    variable `key_env` (OPENAI_API_KEY by default), never from the command line, and never stored.
    Prices (USD per million tokens) are the user's to give; without them the spend cap cannot be
    enforced and the loop's token cap is the limit."""

    def __init__(self, model: str, *, base_url: Optional[str] = None, key_env: str = "OPENAI_API_KEY",
                 effort: Optional[str] = None, max_tokens: int = 16000, price_in: Optional[float] = None,
                 price_out: Optional[float] = None, price_cached: Optional[float] = None, timeout: float = 600.0,
                 extra_body: Optional[dict] = None,
                 client: Any = None):
        if not model:
            raise ValueError("the OpenAI-compatible provider needs --agent-model (the model id on that server)")
        if client is None:
            try:
                import openai
            except ImportError as e:
                raise ValueError("this provider needs the OpenAI SDK: pip install openai") from e
            import os
            key = os.environ.get(key_env)
            if not key and base_url is None:
                raise ValueError(f"set {key_env} to your API key (or pick another variable with --agent-key-env)")
            # Local servers often need no key; the SDK still wants a string.
            client = openai.OpenAI(api_key=key or "not-needed", base_url=base_url, timeout=timeout, max_retries=1)
            self._errors = (openai.AuthenticationError, openai.BadRequestError)
        else:
            self._errors = ()
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens
        self.base_url = base_url
        self.extra_body = dict(extra_body or {})
        self.prices = (price_in, price_out) if price_in is not None and price_out is not None else None
        # cached input tokens bill at the full input price unless the user gives the cache-read price
        self.price_cached = price_cached if self.prices and price_cached is not None else None
        self.name = f"openai-compatible:{model}" + (f"@{base_url}" if base_url else "")
        self.caching = openai_cache_control(model, base_url)
        self.cached_tokens = 0

    @staticmethod
    def tools(schemas: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": s["name"], "description": s.get("description", ""),
                                                  "parameters": s.get("input_schema") or {"type": "object"}}}
                for s in schemas]

    @staticmethod
    def messages(system: str, history: list[dict]) -> list[dict]:
        """The neutral history as Chat Completions messages."""
        out: list[dict] = [{"role": "system", "content": system}]
        for m in history:
            c = m["content"]
            if m["role"] == "user":
                if isinstance(c, str):
                    out.append({"role": "user", "content": c})
                    continue
                for b in c:
                    if b.get("type") == "tool_result":
                        body = b.get("content", "")
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                    "content": ("ERROR: " + body) if b.get("is_error") else body})
                    elif b.get("type") == "text":
                        out.append({"role": "user", "content": b["text"]})
                continue
            blocks = c if isinstance(c, list) else [{"type": "text", "text": str(c)}]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})},
                      # server data the call came with, returned unchanged (Gemini 3's thought signatures)
                      **({"extra_content": b["extra_content"]} if b.get("extra_content") else {})}
                     for b in blocks if b.get("type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        return out

    def turn(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        kw: dict[str, Any] = {"model": self.model, "messages": self.messages(system, messages),
                              "tools": self.tools(tools), "tool_choice": "auto"}
        # api.openai.com takes max_completion_tokens; compatible servers mostly still take max_tokens.
        kw["max_tokens" if self.base_url else "max_completion_tokens"] = self.max_tokens
        if self.effort:
            kw["reasoning_effort"] = self.effort
        if self.extra_body:                               # server-specific options, e.g. chat_template_kwargs
            kw["extra_body"] = self.extra_body
        if self.caching:                                  # a breakpoint on the fixed system prompt
            m0 = kw["messages"][0]
            kw["messages"] = [{"role": "system", "content": [{"type": "text", "text": m0["content"],
                                                             "cache_control": {"type": "ephemeral"}}]},
                              *kw["messages"][1:]]
        try:
            try:
                r = self.client.chat.completions.create(**kw)
            except self._errors as e:                   # type: ignore[misc]
                if not self.caching or "cache" not in str(e).lower():
                    raise
                self.caching = False
                kw["messages"] = self.messages(system, messages)
                r = self.client.chat.completions.create(**kw)
        except self._errors as e:                       # type: ignore[misc]
            raise ValueError(f"{self.name} rejected the request: {e}") from e
        ch = r.choices[0]
        msg = ch.message
        calls, content = [], []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = {"__invalid_json__": (tc.function.arguments or "")[:2000]}
            calls.append(ToolCall(tc.id, tc.function.name, args))
            block = {"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args}
            # Gemini 3 on its OpenAI endpoint attaches a thought signature here and rejects the next turn
            # unless it comes back with the call.
            extra = getattr(tc, "extra_content", None) or (getattr(tc, "model_extra", None) or {}).get("extra_content")
            if extra:
                block["extra_content"] = extra
            content.append(block)
        stop = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens",
                "content_filter": "refusal"}.get(ch.finish_reason or "", ch.finish_reason or "")
        if getattr(msg, "refusal", None):
            stop = "refusal"
        u = getattr(r, "usage", None)
        tin = int(getattr(u, "prompt_tokens", 0) or 0)
        tout = int(getattr(u, "completion_tokens", 0) or 0)
        # Some servers (Gemini's OpenAI endpoint) leave thinking tokens out of completion_tokens but bill them as
        # output; they show up only in total_tokens.
        total = int(getattr(u, "total_tokens", 0) or 0)
        if total > tin + tout:
            tout = total - tin
        det = getattr(u, "prompt_tokens_details", None)
        cached = min(tin, int(getattr(det, "cached_tokens", 0) or 0))   # automatic caching, where the server reports it
        self.cached_tokens += cached
        cost = 0.0
        if self.prices:
            pc = self.prices[0] if self.price_cached is None else self.price_cached
            cost = ((tin - cached) * self.prices[0] + cached * pc + tout * self.prices[1]) / 1e6
        return Turn(msg.content or "", calls, content, stop, tin, tout, cost)


@dataclass
class ScriptedProvider:
    """Plays back turns: each is a list of (tool name, input) pairs, or a str for a final text turn.
    An input may be a callable taking the results so far, for ids that only exist at run time."""
    script: list[Any]
    name: str = "scripted"
    results: list[str] = field(default_factory=list)
    _i: int = 0

    def turn(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        if messages and isinstance(messages[-1].get("content"), list):
            self.results += [c.get("content", "") for c in messages[-1]["content"]
                             if isinstance(c, dict) and c.get("type") == "tool_result"]
        if self._i >= len(self.script):
            return Turn("done", [], [{"type": "text", "text": "done"}], "end_turn")
        step = self.script[self._i]
        self._i += 1
        if isinstance(step, str):
            return Turn(step, [], [{"type": "text", "text": step}], "end_turn")
        calls = []
        for k, (name, inp) in enumerate(step):
            inp = inp(self.results) if callable(inp) else inp
            calls.append(ToolCall(f"t{self._i}_{k}", name, inp))
        content = [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
        return Turn("", calls, content, "tool_use")
