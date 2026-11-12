"""wsr — Weight-Space Repair CLI, triage layer.

    wsr triage   full layered pipeline (files, config, request, response, output, probes, known issues, verdict)
    wsr file     layer 1: structural validation of safetensors / GGUF
    wsr config   layer 2: template, EOS, tokenizer, memory fit
    wsr response layer 3: classify a saved provider response
    wsr output   layer 4: classify output text
    wsr probe    layer 5: probe a live endpoint
    wsr deep     weight-level diagnosis on a full-precision HF model (needs torch)
    wsr quant    GGUF tensor statistics and quantisation error vs the original weights
    wsr qbisect  which quantised tensor causes a failure
    wsr repair   surgery: the smallest verified weight edit for a diagnosed failure (--deep: local
                 edits and stricter checks; --check PATCH: verify an existing patch; --agent: a frontier
                 model with tools when the rule-based playbook cannot fix it)
    wsr patch    apply or revert a surgery patch, or export the patched model as a GGUF
    wsr objective / profile / evaluate / program / fix
                 weight-space programming: a goal file (practice and held-out data, grader, guardrails),
                 failure groups over a dataset, the paired held-out judge, and the whole loop
    wsr data     public datasets as goal-file rows (asks before downloading)
    wsr consent  remembered answers to wsr's questions
    wsr cache    what wsr keeps between runs: show or clear

Exit codes: 0 no failure found / a fix was found and accepted, 1 a failure found / no accepted fix,
2 usage or read error, 3 a step needed a yes that was not given, 130 interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .config_checks import check_config
from .file_integrity import check_files
from .findings import Report, Severity
from .output import PRESETS, check_output
from .pipeline import TriageInput, triage, write_handoff
from .probes import PROBES, Endpoint
from .report import to_json, to_text
from .response import check_response
from .verdict import decide


# `wsr program` / `wsr fix` outcomes that mean a judged fix exists (exit code 0)
FIX_OUTCOMES = ("shipped", "steering recommended", "settings recommended")
AGENT_ALLOWS = ("embed", "unembed", "final_norm", "rescale", "large")


def _loads(text: str, path: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{'stdin' if path == '-' else path}: not valid JSON ({e.msg}, line {e.lineno})") from None


def _read_json(path: str) -> Any:
    return _loads(_read_text(path), path)


def _messages_arg(args: argparse.Namespace, *, required: bool = True) -> Optional[list[dict]]:
    """The failing conversation: --messages FILE (an array, or an object with "messages"), or --prompt with
    --system."""
    if getattr(args, "messages", None):
        m = _read_json(args.messages)
        m = m.get("messages") if isinstance(m, dict) else m
        if not isinstance(m, list) or not m:
            raise ValueError(f"{args.messages}: expected a messages array (or an object with one)")
        return m
    if getattr(args, "prompt", None):
        system = [{"role": "system", "content": args.system}] if getattr(args, "system", None) else []
        return system + [{"role": "user", "content": args.prompt}]
    if required:
        raise ValueError("give the failing prompt: --prompt or --messages")
    return None


def _read_response(path: str) -> tuple[Any, list]:
    """A saved response: one JSON body, or a raw SSE / NDJSON stream (assembled and checked)."""
    from .stream import assemble, looks_like_stream

    text = _read_text(path)
    if looks_like_stream(text):
        return assemble(text.splitlines(True))
    return _loads(text, path), []


def _read_text(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def _emit(rep: Report, args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        print(to_json(rep))
    else:
        print(to_text(rep, verbose=getattr(args, "verbose", False)))
    v = rep.verdict or {}
    if v.get("verdict") not in (None, "no_failure_observed"):
        return 1
    return 1 if rep.worst == Severity.FAIL else 0


def _endpoint(args: argparse.Namespace, url: Optional[str] = None, model: Optional[str] = None) -> Endpoint:
    return Endpoint(url or args.endpoint, model or args.model, api=args.api,
                    api_key=args.api_key or os.environ.get("WSR_API_KEY"),
                    timeout=args.timeout, backend=args.backend, budget_s=args.budget)


def _alt(spec: str, args: argparse.Namespace) -> Endpoint:
    # name=url,model  or  url,model
    _, _, rest = spec.rpartition("=") if "=" in spec.split(",")[0] else ("", "", spec)
    url, _, model = rest.partition(",")
    if not url or not model:
        raise ValueError(f"--alt expects URL,MODEL (got {spec!r})")
    return _endpoint(args, url, model)


# -- subcommands ---------------------------------------------------------------

def cmd_file(args: argparse.Namespace) -> int:
    rep = Report()
    rep.extend(check_files(Path(args.path), scan_values=args.scan_values, upstream=args.upstream,
                           verify_hash=args.hash))
    return _emit(rep, args)


def cmd_config(args: argparse.Namespace) -> int:
    rep = Report()
    _, fs = check_config(Path(args.path), upstream=args.upstream, target_ctx=args.ctx, memory=not args.no_memory)
    rep.extend(fs)
    return _emit(rep, args)


def cmd_response(args: argparse.Namespace) -> int:
    raw, stream_fs = _read_response(args.file)
    req = _read_json(args.request) if args.request else {}
    norm, fs = check_response(raw, max_tokens=args.max_tokens or req.get("max_tokens"), tools=req.get("tools"))
    rep = Report(findings=stream_fs + fs)
    if norm.text.strip() or not norm.tool_calls:
        rep.extend(check_output(norm.text, preset=args.preset, finish_reason=norm.finish_reason,
                                expect_lang=args.expect_lang, expect_script=args.expect_script))
    rep.verdict = decide(rep.findings)
    return _emit(rep, args)


def cmd_output(args: argparse.Namespace) -> int:
    text = _read_text(args.file)
    turns = None
    if args.agent_trace:
        turns = _read_json(args.agent_trace)
    schema = _read_json(args.schema) if args.schema else None
    rep = Report(findings=check_output(
        text, preset=args.preset, finish_reason=args.finish_reason, expect_json=args.expect_json,
        json_schema=schema, expect_script=args.expect_script, expect_lang=args.expect_lang,
        prompt=_read_text(args.prompt_file) if args.prompt_file else None, turns=turns))
    rep.verdict = decide(rep.findings)
    return _emit(rep, args)


def cmd_probe(args: argparse.Namespace) -> int:
    from .probes import run_probes

    ep = _endpoint(args)
    why = ep.reachable()
    if why:
        print(f"wsr: {why}; is the server running?", file=sys.stderr)
        return 2
    rep = Report()
    rep.context["backend"] = ep.detect_backend()
    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else None
    try:
        rep.extend(run_probes(ep, only))
    except ValueError as e:
        print(f"wsr: {e}", file=sys.stderr)
        return 2
    return _emit(rep, args)


def cmd_triage(args: argparse.Namespace) -> int:
    resp_raw, stream_fs = _read_response(args.response) if args.response else (None, [])
    messages = None
    if args.messages:
        messages = _read_json(args.messages)
    elif args.prompt or args.prompt_file:
        prompt = args.prompt if args.prompt else _read_text(args.prompt_file)
        messages = ([{"role": "system", "content": args.system}] if args.system else []) + \
            [{"role": "user", "content": prompt}]
    params: dict[str, Any] = _read_json(args.request) if args.request else {}
    if isinstance(params.get("messages"), list) and messages is None:
        messages = params.pop("messages")
    params.pop("model", None)
    params.pop("stream", None)
    for k in ("temperature", "max_tokens", "top_p", "repeat_penalty"):
        v = getattr(args, k)
        if v is not None:
            params[k] = v

    ep = _endpoint(args) if args.endpoint else None
    if args.endpoint and not args.model:
        raise ValueError("--endpoint needs --model")
    inp = TriageInput(
        model_dir=Path(args.model_dir) if args.model_dir else None,
        endpoint=ep, messages=messages, params=params,
        response=resp_raw, extra_findings=stream_fs,
        model_name=args.model or "", backend=args.backend, upstream=args.upstream,
        expect_lang=args.expect_lang, expect_script=args.expect_script, expect_json=args.expect_json,
        check_echo=args.check_echo, preset=args.preset, scan_values=args.scan_values, verify_hash=args.hash,
        probes=[] if args.no_probes else (args.probes.split(",") if args.probes else None),
        run_differential=args.differential, alts=[_alt(a, args) for a in args.alt or []],
        conditions=args.condition or [], known_issues_file=Path(args.known_issues) if args.known_issues else None,
        target_ctx=args.ctx, server_args=args.server_args)
    if not (inp.model_dir or inp.endpoint or inp.response is not None):
        raise ValueError("give at least one of --model-dir, --endpoint, --response")
    rep = triage(inp)
    if args.handoff:
        write_handoff(rep, inp, Path(args.handoff))
    return _emit(rep, args)


# -- parser --------------------------------------------------------------------

def _agent_opts(p: argparse.ArgumentParser) -> None:
    """The agent's flags, shared by `wsr repair`, `wsr program` and `wsr fix`."""
    p.add_argument("--agent", action="store_true",
                    help="when the rule-based playbook cannot fix it, hand the diagnosis and every failed trial to "
                         "a frontier model with tools to inspect and edit the weights; only a patch that passes the "
                         "same checks is accepted")
    p.add_argument("--agent-provider", default="anthropic", choices=["anthropic", "openai"],
                    help="anthropic: Claude (ANTHROPIC_API_KEY, pip install anthropic); openai: any model behind the "
                         "OpenAI Chat Completions API - OpenAI itself, or with --agent-base-url any compatible "
                         "server: OpenRouter, Together, Groq, DeepSeek, Mistral, Gemini's OpenAI endpoint, a local "
                         "vLLM / Ollama / llama-server (pip install openai)")
    p.add_argument("--agent-model", default=None,
                    help="the agent's model id (anthropic default: claude-opus-5-5; openai: required)")
    p.add_argument("--agent-base-url", default=None, help="openai provider: the compatible server's URL")
    p.add_argument("--agent-key-env", default=None, metavar="VAR",
                    help="environment variable holding the key (default ANTHROPIC_API_KEY / OPENAI_API_KEY); the "
                         "key itself is never taken on the command line")
    p.add_argument("--agent-effort", default=None,
                    help="reasoning effort (anthropic: low|medium|high|xhigh|max, default high; openai: passed as "
                         "reasoning_effort only when given)")
    p.add_argument("--agent-price", nargs="+", type=float, default=None, metavar="USD",
                    help="openai provider: USD per million tokens - IN OUT [CACHED]: input, output and, optionally, "
                         "cached input (else cached input counts at the input price), so --agent-budget-usd can be "
                         "enforced (without it the token cap is the limit)")
    p.add_argument("--agent-turn-tokens", type=int, default=16000,
                    help="output tokens per agent turn, reasoning included (lower it for small local models that "
                         "think at length)")
    p.add_argument("--agent-timeout", type=float, default=600.0,
                    help="openai provider: seconds per request (raise it for a local model on the CPU)")
    p.add_argument("--agent-extra", default=None, metavar="JSON",
                    help="openai provider: extra request fields for the server, e.g. "
                         "'{\"chat_template_kwargs\": {\"enable_thinking\": false}}' for llama-server")
    p.add_argument("--agent-max-tokens", type=int, default=6_000_000,
                    help="stop the agent past this many input + output tokens (every provider)")
    p.add_argument("--agent-budget-usd", type=float, default=None,
                    help="stop the agent past this API spend (default: the goal file's budget.usd, else 5)")
    p.add_argument("--agent-turns", type=int, default=80)
    p.add_argument("--agent-verifications", type=int, default=None,
                    help="full checks the agent may run (default: the goal file's budget.verifications, else 6)")
    p.add_argument("--agent-allow", default="", metavar="LIST",
                    help="comma list of opt-in unlocks for the agent's free-form edits: embed, unembed, final_norm "
                         "(frozen by default), rescale (whole-tensor rescaling), large (lifts the rank and size "
                         "caps); the edit must still pass every check")


def _backend_opts(p: argparse.ArgumentParser) -> None:
    from .consent import add_args

    g = p.add_argument_group("where replies come from")
    g.add_argument("--backend", default="auto", choices=["auto", "torch", "llama-server"],
                   help="auto: llama-server when its binary and --gguf are there and you agree to start it (faster "
                        "for long or many replies), else PyTorch; torch: always the full-precision model")
    g.add_argument("--gguf", help="GGUF of the model, for the llama-server backend")
    g.add_argument("--endpoint", help="an already-running llama-server (http://host:port): used as is, nothing started")
    g.add_argument("--llama-server", dest="llama_server", metavar="PATH",
                   help="llama-server binary or its folder (default: WSR_LLAMA_SERVER, then PATH)")
    g.add_argument("--parallel", type=int, default=4, help="llama-server: requests in flight (slots)")
    add_args(p)


def _source(args: argparse.Namespace, model: Optional[str], proto, *, need_patch: bool = False,
            patched_gguf: Optional[str] = None, consent=None):
    """The backend for these args, asking before starting anything. Returns (backend, notes, consent)."""
    from .consent import Consent
    from .program import backends as B

    consent = consent or Consent.from_args(args)

    def load():
        _need_torch()
        from .deep.lm import LM
        return LM.load(model, dtype=args.dtype, device=args.device)

    ch = B.choose(args.backend, consent=consent, protocol=proto, model_dir=model, gguf=args.gguf,
                  endpoint=args.endpoint, server_bin=args.llama_server, parallel=args.parallel,
                  need_patch=need_patch, patched_gguf=patched_gguf, load_torch=load)
    for n in ch.notes:
        print(f"  backend: {n}", file=sys.stderr)
    return ch.backend, ch.notes, consent


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.add_argument("-v", "--verbose", action="store_true", help="show passing checks and sources")


def _output_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", choices=sorted(PRESETS), default="chat")
    p.add_argument("--expect-lang", help="expected language as an ISO code (28 supported: en fr de es ru uk hi "
                                         "zh ja ko ar th ...)")
    p.add_argument("--expect-script", help="expected script: latin han kana hangul cyrillic arabic devanagari ...")


def _endpoint_opts(p: argparse.ArgumentParser, required: bool = False) -> None:
    p.add_argument("--model", required=required, help="model name as the server knows it")
    p.add_argument("--api", choices=["openai", "ollama"], default="openai",
                   help="openai = /v1/chat/completions (default), ollama = native /api/chat")
    p.add_argument("--api-key", help="the server's key (or set WSR_API_KEY, which keeps it out of the shell history)")
    p.add_argument("--backend", help="skip detection: ollama, llama-server, lmstudio, vllm, openai-compatible")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--budget", type=float, default=300, help="total seconds for all endpoint calls")


def _deep_opts(d: argparse.ArgumentParser) -> None:
    d.add_argument("model_dir", help="Hugging Face model directory (full precision)")
    d.add_argument("--prompt", help="the failing user prompt")
    d.add_argument("--system", help="system prompt to send with --prompt")
    d.add_argument("--messages", help="JSON file with the failing messages array")
    d.add_argument("--kind", default="auto", choices=["auto", "blank", "loop", "refusal", "language", "other"])
    d.add_argument("--failing", action="append", default=[], help="another prompt that fails the same way")
    d.add_argument("--passing", action="append", default=[], help="a near-identical prompt that works")
    d.add_argument("--expect-lang", help="expected answer language (for wrong-language failures)")
    d.add_argument("--max-new", type=int, default=48, help="tokens generated per run (default 48)")
    d.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"],
                   help="auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16")
    d.add_argument("--device", default="auto", help="auto (CUDA when available), cpu, cuda, cuda:1 ...")
    d.add_argument("--no-thinking", action="store_true", help="render the template with enable_thinking=false")
    d.add_argument("--base", help="base model directory, to measure how much fine-tuning changed the culprits")
    d.add_argument("--gguf", help="the quantised file in use: quantisation error of the implicated tensors")
    d.add_argument("--out", help="write the surgery hand-off package here")
    d.add_argument("--no-known", action="store_true", help="skip the known-direction library")
    d.add_argument("--no-probes", action="store_true",
                   help="skip the calibrated probes (language probe for wrong-language, copy heads for loops)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wsr", description="Find out why a model failed, and how to fix it.")
    ap.add_argument("--version", action="version", version=f"wsr {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage", help="run the full layered triage")
    t.add_argument("--model-dir", help="HF model directory or .gguf file")
    t.add_argument("--endpoint", help="server base URL, e.g. http://localhost:11434")
    _endpoint_opts(t)
    t.add_argument("--prompt", help="the failing user prompt")
    t.add_argument("--prompt-file")
    t.add_argument("--system", help="system prompt to send with --prompt")
    t.add_argument("--messages", help="JSON file with the failing messages array")
    t.add_argument("--request", help="JSON file with the request body / parameters that failed")
    t.add_argument("--response", help="JSON file with a saved failing response (no live call needed)")
    t.add_argument("--temperature", type=float)
    t.add_argument("--max-tokens", type=int)
    t.add_argument("--top-p", type=float)
    t.add_argument("--repeat-penalty", type=float)
    _output_opts(t)
    t.add_argument("--expect-json", action="store_true")
    t.add_argument("--check-echo", action="store_true", help="flag output copied from the prompt (not for rewrites)")
    t.add_argument("--upstream", help="HF repo[@rev] to compare files and template against")
    t.add_argument("--hash", action="store_true", help="sha256 the weights (slow)")
    t.add_argument("--scan-values", action="store_true", help="scan safetensors for NaN/Inf (needs numpy)")
    t.add_argument("--probes", help=f"comma list of probes: {','.join(PROBES)} (default: all but the slow ones: "
                        "context, determinism, cache, concurrency)")
    t.add_argument("--no-probes", action="store_true")
    t.add_argument("--differential", action="store_true", help="rerun the failure flipping one factor at a time")
    t.add_argument("--alt", action="append", help="alternate endpoint for the differential: URL,MODEL")
    t.add_argument("--condition", action="append", help="extra known-issue conditions, e.g. vulkan")
    t.add_argument("--known-issues", help="extra known-issues JSON file")
    t.add_argument("--ctx", type=int, default=None,
                   help="context size for the memory-fit estimate (default: the server's, else 32768)")
    t.add_argument("--handoff", help="write the hand-off package (request, response, report) here")
    t.add_argument("--server-args", help="the server's launch flags, e.g. \"-c 8192 -np 4 -ctk q4_0 -fa on\" "
                                         "(llama-server / vLLM); Ollama's OLLAMA_* environment is read locally")
    _common(t)
    t.set_defaults(fn=cmd_triage)

    f = sub.add_parser("file", help="layer 1: structural validation of model files")
    f.add_argument("path")
    f.add_argument("--scan-values", action="store_true")
    f.add_argument("--upstream")
    f.add_argument("--hash", action="store_true")
    _common(f)
    f.set_defaults(fn=cmd_file)

    c = sub.add_parser("config", help="layer 2: template, EOS, tokenizer and memory checks")
    c.add_argument("path")
    c.add_argument("--upstream")
    c.add_argument("--ctx", type=int, default=32768)
    c.add_argument("--no-memory", action="store_true")
    _common(c)
    c.set_defaults(fn=cmd_config)

    r = sub.add_parser("response", help="layer 3: classify a saved provider response (JSON, or - for stdin)")
    r.add_argument("file")
    r.add_argument("--request", help="the request JSON (for max_tokens and tool schemas)")
    r.add_argument("--max-tokens", type=int)
    _output_opts(r)
    _common(r)
    r.set_defaults(fn=cmd_response)

    o = sub.add_parser("output", help="layer 4: classify output text (file, or - for stdin)")
    o.add_argument("file")
    o.add_argument("--finish-reason")
    o.add_argument("--expect-json", action="store_true")
    o.add_argument("--schema", help="JSON schema file")
    o.add_argument("--prompt-file", help="enable PROMPT_ECHO against this prompt")
    o.add_argument("--agent-trace", help="JSON list of turns for AGENT_LOOP")
    _output_opts(o)
    _common(o)
    o.set_defaults(fn=cmd_output)

    p = sub.add_parser("probe", help="layer 5: probe a live endpoint")
    p.add_argument("endpoint")
    _endpoint_opts(p, required=True)
    p.add_argument("--only", help=f"comma list: {','.join(PROBES)}")
    _common(p)
    p.set_defaults(fn=cmd_probe)

    d = sub.add_parser("deep", help="layer 7: weight-level diagnosis on a full-precision HF model (needs torch)")
    _deep_opts(d)
    _common(d)
    d.set_defaults(fn=cmd_deep)

    rp = sub.add_parser("repair", help="surgery: diagnose (as wsr deep), then find and verify the smallest weight "
                                       "edit that fixes it")
    _deep_opts(rp)
    rp.add_argument("--keep", action="append", default=[], help="a prompt whose reply must not regress (adds to the "
                                                                "built-in regression suite)")
    rp.add_argument("--max-drift", type=float, default=0.15,
                    help="allowed change in log-likelihood of the original regression replies, nats per token")
    rp.add_argument("--min-agreement", type=float, default=0.85,
                    help="share of reply positions whose top token must stay the same")
    rp.add_argument("--steering", default="fallback", choices=["fallback", "only", "off"], help="activation steering against the diagnosed direction: fallback (default) when no weight edit passes, only (no weight edits), off")
    rp.add_argument("--allow-safety-loss", action="store_true",
                    help="accept a patch that stops refusing harmful requests (off by default)")
    rp.add_argument("--save-model", action="store_true", help="also write the patched model to <out>/model (in "
                                                              "the checkpoint's precision)")
    rp.add_argument("--deep", action="store_true",
                    help="deep mode: also try local edits (only the layers / MLP neurons writing the failure, "
                         "strengths up to 6), confirm on fresh prompt variants, and report a random edit of the "
                         "same size as a control (slower)")
    rp.add_argument("--sampled", action="store_true",
                    help="with --deep: also require a paired sampled comparison (the model's own sampling "
                         "settings) to show a clearly lower failure rate - for failures that only appear when "
                         "sampling and on long replies (slow)")
    rp.add_argument("--samples", type=int, default=4, help="--sampled: samples per prompt (default 4)")
    rp.add_argument("--sampled-max-new", type=int, default=512, help="--sampled: tokens per sample (default 512)")
    rp.add_argument("--confirm", action="append", default=[],
                    help="--deep: a further prompt for the fresh confirmation (failing or working, judged on the "
                         "unpatched model)")
    rp.add_argument("--serve-dtype", choices=["auto", "bfloat16", "float16", "none"], default="auto",
                    help="check the patch again after rounding to the serving precision (auto: the checkpoint's "
                         "16-bit type, else bfloat16; none: skip)")
    rp.add_argument("--check", metavar="PATCH_DIR",
                    help="do not search: check this patch (e.g. one a fast repair found) with the mode's checks")
    _identity_flags(rp)
    _agent_opts(rp)
    rp.add_argument("--agent-only", action="store_true",
                    help="skip the rule-based playbook and hand the case to the agent at once (still judged by the "
                         "same checks); for testing an agent or when the recipes are known not to fit")
    rp.add_argument("--resume", default=None, metavar="OUT_DIR",
                    help="continue an earlier agent run from its output directory (agent_state/): workspace, "
                         "edits, notes and requests come back; proposals the new --agent-allow unlocks permit are "
                         "built and verified first. Refuses a different model")
    _common(rp)
    rp.set_defaults(fn=cmd_repair)

    pa = sub.add_parser("patch", help="apply, revert, export, show or check a fix (a patch, steering or settings "
                                      "directory with its MANIFEST.json)")
    pa.add_argument("action", choices=["apply", "revert", "export", "show", "check"])
    pa.add_argument("model_dir", help="apply / revert / export: the model directory; show / check: the fix's directory "
                                      "(<out>/patch, <out>/steering or <out>/settings)")
    pa.add_argument("patch_dir", nargs="?", help="apply / revert / export: the patch directory written by wsr repair / "
                                                 "program (<out>/patch); export without it writes the unpatched model "
                                                 "(the comparison file); check: optionally the model directory to "
                                                 "check the fix against")
    pa.add_argument("--out", help="apply / revert: where to write the patched model")
    _identity_flags(pa)
    pa.add_argument("--gguf", help="export: the GGUF file to write")
    pa.add_argument("--quant", default="Q8_0", help="export: F16, BF16, Q8_0 (converter) or any llama-quantize type "
                                                    "such as Q4_K_M (default Q8_0)")
    pa.add_argument("--llama-cpp", help="export: a llama.cpp checkout with convert_hf_to_gguf.py (else WSR_LLAMA_CPP, "
                                        "then one wsr fetched before, then ask to download it)")
    pa.add_argument("--llama-server", dest="llama_server", metavar="PATH",
                    help="export: llama-server binary or folder (its build picks the converter version; "
                         "llama-quantize is looked for next to it)")
    from .consent import add_args as _ca
    _ca(pa)
    pa.add_argument("--json", action="store_true", help="show / check: print JSON")
    pa.set_defaults(fn=cmd_patch)

    q = sub.add_parser("quant", help="GGUF tensor statistics and quantisation error vs the original weights")
    q.add_argument("gguf")
    q.add_argument("--reference", help="HF directory (safetensors) the GGUF was converted from")
    q.add_argument("--only", help=r"regex on tensor names, e.g. 'blk\.1[0-9]\.ffn'")
    q.add_argument("--top", type=int, default=10)
    _common(q)
    q.set_defaults(fn=cmd_quant)

    b = sub.add_parser("qbisect", help="which quantised tensor causes a failure: rebuild the served model from the "
                                       "GGUF in PyTorch and swap tensors back to full precision")
    b.add_argument("model_dir", help="full-precision HF directory the GGUF was converted from")
    b.add_argument("gguf", help="the quantised file in use")
    b.add_argument("--prompt")
    b.add_argument("--system")
    b.add_argument("--messages", help="JSON file with the failing conversation")
    b.add_argument("--expect-lang", help="language the reply must be in")
    b.add_argument("--max-new", type=int, default=64)
    b.add_argument("--max-set", type=int, default=24, help="largest tensor set to try before calling it diffuse")
    b.add_argument("--device", default="auto")
    b.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    b.add_argument("--no-thinking", action="store_true")
    _common(b)
    b.set_defaults(fn=cmd_qbisect)

    ob = sub.add_parser("objective", help="goal files for weight-space programming: init, check, split")
    obs = ob.add_subparsers(dest="action", required=True)
    oi = obs.add_parser("init", help="write a goal file to fill in (JSON, or YAML with PyYAML)")
    oi.add_argument("out", help="path of the goal file to write (.json / .yaml)")
    oi.add_argument("--name")
    oi.add_argument("--model")
    oi.add_argument("--practice")
    oi.add_argument("--heldout")
    oi.add_argument("--grader")
    oi.add_argument("--force", action="store_true", help="overwrite an existing file")
    oi.set_defaults(fn=cmd_objective_init)
    oc = obs.add_parser("check", help="load a goal file and check its data, grader and practice/held-out overlap")
    oc.add_argument("objective")
    _common(oc)
    oc.set_defaults(fn=cmd_objective_check)
    osp = obs.add_parser("split", help="shuffle a JSONL file (seeded) into practice and held-out files")
    osp.add_argument("data")
    osp.add_argument("--practice", required=True)
    osp.add_argument("--heldout", required=True)
    osp.add_argument("--heldout-share", type=float, default=0.5)
    osp.add_argument("--limit", type=int)
    osp.add_argument("--seed", type=int, default=0)
    osp.set_defaults(fn=cmd_objective_split)
    off = obs.add_parser("from-failure", help="turn a failure (a prompt, or a profile group) into a goal file: "
                                              "confirm it, find items that fail the same way, split them "
                                              "(stratified) into practice and held-out")
    off.add_argument("model_dir", nargs="?", help="HF model directory (not needed with --profile)")
    off.add_argument("--prompt", help="the failing user prompt")
    off.add_argument("--system")
    off.add_argument("--messages", help="JSON file with the failing messages")
    off.add_argument("--failing", action="append", default=[], help="another prompt that fails the same way")
    off.add_argument("--passing", action="append", default=[], help="a similar prompt that works")
    off.add_argument("--dataset", help="local JSONL to measure the hit rate in; 'auto' names the public source that "
                                       "fits (wsr does not download it) and uses rewrites of the prompt")
    off.add_argument("--profile", help="a profile JSON from wsr profile --out")
    off.add_argument("--group", help="with --profile: the label to build the objective for")
    off.add_argument("--grader", help="default: the failure's detector (detector:<group>)")
    off.add_argument("--samples", type=int, default=200, help="dataset rows to run (default 200)")
    off.add_argument("--heldout-share", type=float, default=0.5)
    off.add_argument("--confirm-share", type=float, default=0.25,
                     help="share of the failing items kept as a confirmation set (used once, for one frozen "
                          "candidate); none when too few items fail: 0 for none")
    off.add_argument("--max-new", type=int, default=256)
    off.add_argument("--no-thinking", action="store_true")
    off.add_argument("--expect-lang")
    off.add_argument("--deep", action="store_true", help="also run the full deep diagnosis and stop if it is not "
                                                          "weight-level")
    off.add_argument("--name")
    off.add_argument("--seed", type=int, default=0)
    off.add_argument("--out-dir", required=True, help="where goal.json, practice/ and heldout/ are written")
    off.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    off.add_argument("--device", default="auto")
    _backend_opts(off)
    _common(off)
    off.set_defaults(fn=cmd_objective_from_failure)

    pg = sub.add_parser("program", help="work towards a goal file: profile the practice set, diagnose the largest "
                                        "weight-level failure group, try the playbook, then (with --agent) the agent "
                                        "with practice tools; a held-out judge decides what ships")
    pg.add_argument("objective")
    pg.add_argument("model_dir", nargs="?", help="HF model directory (default: the goal file's model)")
    pg.add_argument("--out", required=True, help="where PROGRAM.md, program.json and the patch go")
    pg.add_argument("--profile-limit", type=int, default=40, help="practice items to profile (default 40)")
    pg.add_argument("--eval-limit", type=int, help="held-out items the judge scores (default: all)")
    pg.add_argument("--practice-evals", type=int, default=None,
                    help="practice_eval calls the agent may make (default: the goal file's budget.practice_evals, "
                         "else 10)")
    pg.add_argument("--no-playbook", action="store_true", help="skip the rule-based recipes")
    pg.add_argument("--no-weight-edits", action="store_true",
                    help="the weights must not change: the playbook only steers and the agent may only steer or "
                         "change serving settings; a fix ships as a steering or settings, never a patch")
    pg.add_argument("--steering", default="fallback", choices=["fallback", "only", "off"], help="activation steering against the diagnosed direction: fallback (default) when no weight edit passes, only (no weight edits), off")
    pg.add_argument("--allow-safety-loss", action="store_true",
                    help="let a fix stop refusing the regression suite's harmful requests (off by default: a fix for something else should not change refusals unnoticed); same as allow_safety_loss: true on the goal's wsr_regression guardrail")
    pg.add_argument("--judge", default="auto", choices=["auto", "torch", "llama-server"],
                    help="where the held-out judge runs. auto: on llama-server when it is there and you agree (the "
                         "candidate is exported to a GGUF; much faster for long replies), else PyTorch batched to "
                         "free memory; torch: never asks")
    pg.add_argument("--judge-quant", default="Q8_0", help="GGUF type the llama-server judge serves (default Q8_0)")
    pg.add_argument("--fast", action="store_true", help="the playbook's fast checks instead of deep ones (no fresh "
                                                        "variants or random control; the held-out judge still decides)")
    pg.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    pg.add_argument("--device", default="auto")
    _agent_opts(pg)
    _backend_opts(pg)
    _common(pg)
    pg.set_defaults(fn=cmd_program)

    cs = sub.add_parser("consent", help="answers wsr remembers ('always' / 'never' at a prompt)")
    css = cs.add_subparsers(dest="action", required=True)
    css.add_parser("show").set_defaults(fn=cmd_consent)
    csr = css.add_parser("reset")
    csr.add_argument("kind", nargs="?", help="one kind (default: all)")
    csr.set_defaults(fn=cmd_consent)

    ch = sub.add_parser("cache", help="show or clear what wsr keeps in its cache (~/.cache/wsr)")
    chs = ch.add_subparsers(dest="action", required=True)
    chs.add_parser("show").set_defaults(fn=cmd_cache)
    chc = chs.add_parser("clear", help="delete cache groups (asks first, unless --yes); remembered consent "
                                       "answers are kept")
    chc.add_argument("groups", nargs="*", help="models, gguf, datasets, converter, probes, or all")
    chc.add_argument("--model", default=None, metavar="KEY", help="only this model's entries (a key, or its start, "
                                                                  "as `wsr cache show` lists them)")
    chc.add_argument("--yes", action="store_true", help="delete without asking")
    chc.set_defaults(fn=cmd_cache)

    fx = sub.add_parser("fix", help="one command, one model load: from a failing prompt to a judged patch "
                                    "(objective from-failure, then program; the served GGUF too when llama-server is "
                                    "used)")
    fx.add_argument("model_dir")
    fx.add_argument("--prompt", help="the failing user prompt")
    fx.add_argument("--system")
    fx.add_argument("--messages", help="JSON file with the failing messages")
    fx.add_argument("--failing", action="append", default=[], help="another prompt that fails the same way")
    fx.add_argument("--passing", action="append", default=[], help="a similar prompt that works")
    fx.add_argument("--dataset", help="local JSONL, or 'auto' (asks before downloading the matching public set)")
    fx.add_argument("--expect-lang")
    fx.add_argument("--max-new", type=int, default=256)
    fx.add_argument("--no-thinking", action="store_true")
    fx.add_argument("--samples", type=int, default=200)
    fx.add_argument("--heldout-share", type=float, default=0.5)
    fx.add_argument("--confirm-share", type=float, default=0.25,
                    help="share of the failing items kept as a confirmation set (used once, for one frozen "
                         "candidate); none when too few items fail: 0 for none")
    fx.add_argument("--seed", type=int, default=0)
    fx.add_argument("--out", required=True, help="goal/ (the objective) and run/ (PROGRAM.md, patch) go here")
    fx.add_argument("--profile-limit", type=int, default=40)
    fx.add_argument("--eval-limit", type=int)
    fx.add_argument("--practice-evals", type=int, default=None)
    fx.add_argument("--no-playbook", action="store_true")
    fx.add_argument("--no-weight-edits", action="store_true",
                    help="the weights must not change: the playbook only steers and the agent may only steer or "
                         "change serving settings; a fix ships as a steering or settings, never a patch")
    fx.add_argument("--steering", default="fallback", choices=["fallback", "only", "off"], help="activation steering against the diagnosed direction: fallback (default) when no weight edit passes, only (no weight edits), off")
    fx.add_argument("--allow-safety-loss", action="store_true",
                    help="let a fix stop refusing the regression suite's harmful requests (off by default: a fix for something else should not change refusals unnoticed); same as allow_safety_loss: true on the goal's wsr_regression guardrail")
    fx.add_argument("--fast", action="store_true")
    fx.add_argument("--judge", default="auto", choices=["auto", "torch", "llama-server"])
    fx.add_argument("--judge-quant", default="Q8_0")
    fx.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    fx.add_argument("--device", default="auto")
    _agent_opts(fx)
    _backend_opts(fx)
    _common(fx)
    fx.set_defaults(fn=cmd_fix)

    da = sub.add_parser("data", help="public datasets wsr can turn into goal-file rows (downloads ask first)")
    das = da.add_subparsers(dest="action", required=True)
    dl = das.add_parser("list", help="the known datasets, and which are already on disk")
    dl.set_defaults(fn=cmd_data_list)
    dfch = das.add_parser("fetch", help="download (after asking) and convert one dataset to wsr JSONL")
    dfch.add_argument("name")
    from .consent import add_args as _consent_args
    _consent_args(dfch)
    dfch.set_defaults(fn=cmd_data_fetch)

    pr = sub.add_parser("profile", help="triage verdicts over a dataset: failure groups, rates and their causes "
                                        "(from tests that ran), accuracy with a grader")
    pr.add_argument("model_dir", nargs="?", help="HF model directory (default: the goal file's model)")
    pr.add_argument("--data", help="JSONL file of requests (rows: messages | prompt | problem | question, "
                                   "optional answer)")
    pr.add_argument("--objective", help="goal file: protocol, grader and (with --split) data come from it")
    pr.add_argument("--split", default="practice", choices=["practice", "heldout"],
                    help="with --objective: which file to profile (default practice; heldout is logged as a look)")
    pr.add_argument("--grader", help="exact | contains | regex:<p> | math | detector:<group> (overrides the goal "
                                     "file)")
    pr.add_argument("--limit", type=int, help="first N rows only")
    pr.add_argument("--max-new", type=int, help="token cap (default: the goal file's, else 512)")
    pr.add_argument("--temperature", type=float, help="sample at this temperature (default: greedy, or the goal "
                                                      "file's sampling)")
    pr.add_argument("--top-p", type=float)
    pr.add_argument("--samples", type=int, default=1, help="samples per item when sampling")
    pr.add_argument("--no-thinking", action="store_true", help="render the template with enable_thinking=false")
    pr.add_argument("--system", help="system prompt for every row")
    pr.add_argument("--tests", type=int, default=3, help="members per group the cause tests run on (0: skip)")
    pr.add_argument("--reruns", type=int, default=3, help="sampled reruns per tested member")
    pr.add_argument("--deep", type=int, default=0, help="also run wsr deep on N members of each weight-level group")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", help="write the profile JSON here (the agent's brief reads it)")
    pr.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    pr.add_argument("--device", default="auto")
    _backend_opts(pr)
    _common(pr)
    pr.set_defaults(fn=cmd_profile)

    ev = sub.add_parser("evaluate", help="score a model, or a patch against the original, on a goal file's "
                                         "held-out set and guardrails (paired, same seeds)")
    ev.add_argument("model_dir", nargs="?", help="HF model directory (default: the goal file's model)")
    ev.add_argument("--objective", required=True)
    ev.add_argument("--patch", help="patch directory (wsr repair's <out>/patch); without it the model is scored")
    ev.add_argument("--split", default="heldout", choices=["heldout", "practice"])
    ev.add_argument("--limit", type=int)
    ev.add_argument("--samples", type=int, default=1, help="samples per item when the protocol samples")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--no-guardrails", action="store_true")
    ev.add_argument("--out", help="write the result JSON here")
    ev.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ev.add_argument("--device", default="auto")
    _backend_opts(ev)
    ev.add_argument("--patched-gguf", help="llama-server backend: the patched model's GGUF (wsr patch export --gguf); "
                                           "without it a patch is scored on PyTorch")
    _identity_flags(ev)
    _common(ev)
    ev.set_defaults(fn=cmd_evaluate)
    _fill_help(ap)
    return ap


# Help for options that several commands share and that are added without their own text. An option's own help
# always wins; every option must end up with one (tests/test_docs.py).
HELP = {
    "action": "apply, revert, or export (a GGUF)",
    "agent_turns": "stop the agent after this many turns",
    "ctx": "context size for the memory-fit estimate",
    "data": "JSONL file to shuffle and split",
    "device": "auto (CUDA when available), cpu, cuda, cuda:1 ...",
    "dtype": "auto: float32 where it fits (CPU, or a GPU with room), else bfloat16 / float16",
    "endpoint": "the server's base URL, e.g. http://localhost:8080",
    "eval_limit": "held-out items the judge scores (default: all)",
    "expect_json": "the reply must be JSON",
    "expect_lang": "language the reply must be in (ISO code)",
    "fast": "the playbook's fast checks instead of deep ones (the held-out judge still decides)",
    "file": "the file to read (- for stdin)",
    "finish_reason": "the finish reason the server reported",
    "gguf": "the GGUF file",
    "grader": "exact | contains | regex:<pattern> | math | detector:<group>",
    "hash": "sha256 the weights and compare with the Hub (slow)",
    "heldout": "held-out JSONL file (only the judge reads it)",
    "heldout_share": "share of the items that goes to the held-out set",
    "judge": "where the held-out judge runs: auto (llama-server when there and allowed), torch, llama-server",
    "judge_quant": "GGUF type the llama-server judge serves",
    "kind": "the failure kind (auto: detected from the reply)",
    "limit": "first N rows only",
    "max_new": "tokens generated per reply",
    "max_tokens": "the request's max_tokens (to tell a cut reply from a finished one)",
    "model": "the model directory the goal is for",
    "model_dir": "Hugging Face model directory (full precision)",
    "name": "a name for it",
    "no_guardrails": "score the main set only",
    "no_memory": "skip the memory-fit estimate",
    "no_playbook": "skip the rule-based recipes",
    "no_probes": "skip the live probes",
    "no_thinking": "render the chat template with enable_thinking=false",
    "objective": "the goal file (JSON or YAML)",
    "path": "model directory, or a .safetensors / .gguf file",
    "practice": "practice JSONL file (the agent may use it)",
    "practice_evals": "practice_eval calls the agent may make (default: the goal file's budget, else 10)",
    "preset": "detector thresholds: chat, strictJson, longForm or lenient",
    "profile_limit": "practice items to profile",
    "prompt": "the failing user prompt",
    "prompt_file": "file with the failing user prompt",
    "repeat_penalty": "the repeat penalty the failing request used",
    "samples": "dataset rows to run",
    "scan_values": "scan safetensors for NaN / Inf (needs numpy)",
    "seed": "random seed",
    "split": "which set to score",
    "system": "system prompt to send with --prompt",
    "temperature": "the temperature the failing request used",
    "timeout": "seconds per request",
    "top": "how many to list",
    "top_p": "the top_p the failing request used",
    "upstream": "Hugging Face repo[@revision] to compare files and template against",
}


def _fill_help(p: argparse.ArgumentParser) -> None:
    for a in p._actions:
        if isinstance(a, argparse._SubParsersAction):
            for sub in a.choices.values():
                _fill_help(sub)
        elif not isinstance(a, argparse._HelpAction) and not a.help and a.dest in HELP:
            a.help = HELP[a.dest]


def cmd_objective_init(args: argparse.Namespace) -> int:
    from .program import objective as O

    out = Path(args.out)
    if out.exists() and not args.force:
        raise ValueError(f"{out} exists (use --force to overwrite)")
    O.write_template(out, name=args.name, model=args.model, practice=args.practice, heldout=args.heldout,
                     grader=args.grader)
    print(f"wrote {out}: fill in the data files, grader, protocol and guardrails, then: wsr objective check {out}")
    return 0


def cmd_objective_check(args: argparse.Namespace) -> int:
    from .program import objective as O

    res = O.check(O.load(args.objective))
    if args.json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
    else:
        print(f"{'OK' if res['ok'] else 'PROBLEMS'}: {res['name']} ({res['protocol']}, grader {res['grader']})")
        for split in ("practice", "heldout"):
            v = res.get(split)
            if v:
                print(f"  {split}: {v['rows']} rows, {v['gradable']} gradable ({v['file']})")
        if "overlap" in res:
            print(f"  practice / held-out overlap: {res['overlap']} prompt(s)")
        if res["guardrails"]:
            print(f"  guardrails: {', '.join(res['guardrails'])}")
        for p in res["problems"]:
            print(f"  - {p}")
        if res.get("note"):
            print(f"  note: {res['note']}")
    return 0 if res["ok"] else 1


def cmd_objective_split(args: argparse.Namespace) -> int:
    from .program import objective as O

    n = O.split_file(Path(args.data), Path(args.practice), Path(args.heldout), heldout_share=args.heldout_share,
                     seed=args.seed, limit=args.limit)
    print(f"practice {n['practice']} rows -> {args.practice}; held-out {n['heldout']} rows -> {args.heldout}")
    return 0


def cmd_program(args: argparse.Namespace) -> int:
    _need_torch()
    from .deep.lm import LM
    from .program import backends as B
    from .program import objective as O
    from .program.program import run_program

    obj = O.load(args.objective)
    model = args.model_dir or (str(obj.resolve(obj.model)) if obj.model else None)
    if not model:
        raise ValueError("wsr program edits the full-precision model: give its directory (or set model in the goal "
                         "file)")
    if not obj.practice or not obj.heldout:
        raise ValueError("wsr program needs both a practice and a held-out file in the goal file")
    provider = _agent_provider(args) if args.agent else None
    lm = LM.load(model, dtype=args.dtype, device=args.device)
    from .consent import Consent

    prof_src, consent = None, Consent.from_args(args)
    if args.endpoint or args.gguf:                              # a server the user named: used as given
        prof_src, _n, _c = _source(args, model, obj.protocol, consent=consent)
        if isinstance(prof_src, B.TorchBackend):
            prof_src = None                                     # the model already loaded
    elif args.backend != "torch" and args.judge != "torch":
        # llama-server is the default only for a user who uses it: one question for the run (or none, when a
        # flag or a remembered answer settles it), and a "no" keeps everything on PyTorch.
        from .program import judge as JG
        prof_src, note = JG.served_original(model_dir=Path(model), consent=consent, server_bin=args.llama_server,
                                            quant=args.judge_quant, max_new=obj.protocol.max_new_tokens,
                                            parallel=args.parallel, required=args.backend == "llama-server")
        if not args.json:
            print(f"  backend: {note}")
    budget, limits, practice_evals = _agent_limits(args, obj.budget) if provider is not None else (None, None, 10)
    if args.allow_safety_loss:
        for g in obj.guardrails:
            if g.suite:
                g.allow_safety_loss = True
    try:
        rec = run_program(lm, obj, allow_safety_loss=args.allow_safety_loss, steering=args.steering,
                          weight_edits=not args.no_weight_edits, out=Path(args.out), profile_source=prof_src, provider=provider,
                          playbook=not args.no_playbook, profile_limit=args.profile_limit, agent_budget=budget,
                          agent_limits=limits, practice_evals=practice_evals, eval_limit=args.eval_limit,
                          log=None if args.json else print, mode="fast" if args.fast else "deep",
                          judge=args.judge, judge_quant=args.judge_quant, consent=consent, model_dir=Path(model),
                          server_bin=args.llama_server)
    finally:
        if prof_src is not None:
            prof_src.close()
    if args.json:
        print(json.dumps(rec, indent=1, default=str, ensure_ascii=False))
    else:
        print(f"\n{rec['outcome'].upper()}: {rec.get('says', '')}")
        print(f"report: {Path(args.out) / 'PROGRAM.md'} ({rec['seconds']:.0f} s)")
    return 0 if rec["outcome"] in FIX_OUTCOMES else 1


def cmd_fix(args: argparse.Namespace) -> int:
    """from-failure and program in one process: the model is loaded once, and program reuses the practice
    labels from-failure computed (same weights, protocol and messages)."""
    _need_torch()
    from .consent import Consent
    from .deep.lm import LM
    from .program import fromfailure as F
    from .program import objective as O
    from .program.backends import TorchBackend
    from .program.program import run_program

    out = Path(args.out)
    consent = Consent.from_args(args)
    provider = _agent_provider(args) if args.agent else None        # fails early: no SDK, key or model
    messages = _messages_arg(args)
    lm = LM.load(args.model_dir, dtype=args.dtype, device=args.device)
    proto = O.Protocol(template_kwargs={"enable_thinking": False} if args.no_thinking else {},
                       max_new_tokens=args.max_new, system=args.system)
    auto = args.dataset == "auto"
    ff = F.from_failure(TorchBackend(lm), out_dir=out / "goal", protocol=proto, model=str(args.model_dir),
                        messages=messages, failing=tuple(args.failing), passing=tuple(args.passing),
                        dataset=Path(args.dataset) if args.dataset and not auto else None, auto=auto,
                        samples=args.samples, heldout_share=args.heldout_share, seed=args.seed,
                        confirm_share=args.confirm_share,
                        expect_lang=args.expect_lang, consent=consent)
    if not args.json:
        print(F.render(ff))
    if ff.get("stopped"):
        if args.json:
            print(json.dumps(ff, indent=1, ensure_ascii=False))
        return 1
    obj = O.load(ff["goal"])
    prof_src = None
    if args.endpoint or args.gguf:
        from .program import backends as B
        prof_src, _n, _c = _source(args, args.model_dir, obj.protocol, consent=consent)
        if isinstance(prof_src, B.TorchBackend):
            prof_src = None
    budget, limits, practice_evals = _agent_limits(args, obj.budget) if provider is not None else (None, None, 10)
    if args.allow_safety_loss:
        for g in obj.guardrails:
            if g.suite:
                g.allow_safety_loss = True
    try:
        rec = run_program(lm, obj, allow_safety_loss=args.allow_safety_loss, steering=args.steering,
                          weight_edits=not args.no_weight_edits, out=out / "run", profile_source=prof_src, provider=provider,
                          playbook=not args.no_playbook, profile_limit=args.profile_limit, agent_budget=budget,
                          agent_limits=limits, practice_evals=practice_evals, eval_limit=args.eval_limit,
                          log=None if args.json else print, mode="fast" if args.fast else "deep", judge=args.judge,
                          judge_quant=args.judge_quant, consent=consent, model_dir=Path(args.model_dir),
                          server_bin=args.llama_server)
    finally:
        if prof_src is not None:
            prof_src.close()
    if args.json:
        print(json.dumps({"from_failure": ff, "program": rec}, indent=1, default=str, ensure_ascii=False))
    else:
        print(f"\n{rec['outcome'].upper()}: {rec.get('says', '')}")
        print(f"report: {out / 'run' / 'PROGRAM.md'} ({rec['seconds']:.0f} s)")
    return 0 if rec["outcome"] in FIX_OUTCOMES else 1


def cmd_consent(args: argparse.Namespace) -> int:
    from .consent import KINDS, memory_path, remember, remembered

    if args.action == "reset":
        for k in ([args.kind] if args.kind else list(remembered())):
            if k not in KINDS:
                raise ValueError(f"unknown kind {k!r}: {', '.join(KINDS)}")
            remember(k, None)
    mem = remembered()
    for k, what in KINDS.items():
        v = "always" if mem.get(k) is True else "never" if mem.get(k) is False else "ask"
        print(f"  {k:<13} {v:<7} {what}")
    print(f"  stored in {memory_path()}")
    return 0


def _mb(n: int) -> str:
    return f"{n / 2**20:,.1f} MB" if n < 2**30 else f"{n / 2**30:,.2f} GB"


def cmd_cache(args: argparse.Namespace) -> int:
    from . import cache as C

    if args.action == "show":
        u = C.usage()
        print(f"wsr cache: {u['base']}")
        for g, what in C.GROUPS.items():
            print(f"  {g:<10} {_mb(u['groups'][g]['bytes']):>10}  {what}")
        for m in u["models"]:
            ent = ", ".join(f"{k} {n}" for k, n in sorted(m["entries"].items()))
            print(f"    {m['key']:<44} {_mb(m['bytes']):>10}  {ent}")
        print(f"  total {_mb(sum(g['bytes'] for g in u['groups'].values()))}; clear with `wsr cache clear <group>` "
              "(WSR_CACHE=off turns caching off)")
        return 0
    if not args.groups and not args.model:
        print("name what to clear: models, gguf, datasets, converter, probes, all (or --model KEY)")
        return 2
    paths = C.targets(args.groups, args.model)
    if not paths:
        print("nothing to clear")
        return 0
    total = sum(C._size(p) for p in paths)
    for p in paths:
        print(f"  {p}  ({_mb(C._size(p))})")
    if not args.yes:
        if not sys.stdin.isatty():
            print(f"would delete {_mb(total)}; not a terminal, so nothing was deleted: add --yes")
            return 3
        if input(f"delete these ({_mb(total)})? [y/N] ").strip().lower() not in ("y", "yes"):
            print("nothing deleted")
            return 0
    print(f"freed {_mb(C.remove(paths))}")
    return 0


def cmd_data_list(args: argparse.Namespace) -> int:
    from .program import datasets as D

    for d in D.listing():
        print(f"  {d['name']:<10} {'on disk' if d['local'] else d['size']:>8}  {d['about']} "
              f"(huggingface.co/datasets/{d['repo']}; grader {d['grader']})")
    print(f"  files go to {D.data_dir()}")
    return 0


def cmd_data_fetch(args: argparse.Namespace) -> int:
    from .consent import Consent
    from .program import datasets as D

    p = D.fetch(args.name, Consent.from_args(args))
    n = sum(1 for _ in open(p, encoding="utf-8"))
    print(f"{args.name}: {n} rows -> {p}")
    return 0


def cmd_objective_from_failure(args: argparse.Namespace) -> int:
    from .program import fromfailure as F
    from .program import objective as O

    from .consent import Consent

    consent = Consent.from_args(args)
    prof = _read_json(args.profile) if args.profile else None
    if prof is not None:
        spec = prof.get("protocol_spec") or {}
        proto = O.Protocol(template_kwargs=dict(spec.get("template_kwargs") or {}),
                           sampling=dict(spec.get("sampling") or {}),
                           max_new_tokens=int(spec.get("max_new_tokens", args.max_new)), system=spec.get("system"))
        model = args.model_dir or prof.get("model") or ""
        lm = None
    else:
        if not (args.model_dir or args.gguf or args.endpoint):
            raise ValueError("give the model directory (or --gguf / --endpoint)")
        proto = O.Protocol(template_kwargs={"enable_thinking": False} if args.no_thinking else {},
                           max_new_tokens=args.max_new, system=args.system)
        model = args.model_dir or args.gguf or args.endpoint
        lm, _, _ = _source(args, args.model_dir, proto)
    messages = _messages_arg(args, required=False)
    auto = args.dataset == "auto"
    res = F.from_failure(lm, out_dir=Path(args.out_dir), protocol=proto, model=str(model), messages=messages,
                         failing=tuple(args.failing), passing=tuple(args.passing),
                         dataset=Path(args.dataset) if args.dataset and not auto else None, auto=auto, profile=prof,
                         group=args.group, grader=args.grader, samples=args.samples,
                         heldout_share=args.heldout_share, seed=args.seed, deep=args.deep, name=args.name,
                         confirm_share=args.confirm_share,
                         expect_lang=args.expect_lang, consent=consent)
    print(json.dumps(res, indent=1, ensure_ascii=False) if args.json else F.render(res))
    return 0 if not res.get("stopped") else 1


def _progress(label: str):
    def f(*a):
        if len(a) == 3:
            tag, i, n = a
        else:
            (i, n), tag = a, label
        if i == n or i % 10 == 0:
            print(f"  {tag}: {i}/{n}", file=sys.stderr, flush=True)
    return f


def cmd_profile(args: argparse.Namespace) -> int:
    from .program import objective as O
    from .program import profile as P

    obj = O.load(args.objective) if args.objective else None
    proto = obj.protocol if obj else O.Protocol()
    if args.max_new:
        proto.max_new_tokens = args.max_new
    if args.temperature is not None:
        proto.sampling = {"temperature": args.temperature, **({"top_p": args.top_p} if args.top_p else {})}
    if args.no_thinking:
        proto.template_kwargs = {**proto.template_kwargs, "enable_thinking": False}
    if args.system:
        proto.system = args.system
    grader = args.grader or (obj.grader if obj else None)
    if grader:
        from .program.graders import check_name
        check_name(grader)
    if args.data:
        data = Path(args.data)
    elif obj:
        data = obj.resolve(obj.practice if args.split == "practice" else obj.heldout)
        if data is None:
            raise ValueError(f"the goal file has no {args.split} file")
    else:
        raise ValueError("give --data, or --objective")
    model = args.model_dir or (str(obj.resolve(obj.model)) if obj and obj.model else None)
    if not model and not (args.gguf or args.endpoint):
        raise ValueError("give the model directory (or set model in the goal file), or --gguf / --endpoint")
    items = O.load_items(data, system=proto.system, limit=args.limit)
    src, notes, consent = _source(args, model, proto)
    try:
        prof = P.profile(src, items, proto, grader=grader, name=(obj.name if obj else Path(data).stem),
                         model=model or args.gguf or args.endpoint,
                         samples=args.samples, seed=args.seed, test_k=args.tests, test_n=args.reruns,
                         deep_each=args.deep, progress=_progress("profile"), data=str(Path(data).resolve()))
    finally:
        src.close()
    if obj and args.split == "heldout" and not args.data:
        from .program.evaluate import HeldoutLog
        HeldoutLog(obj).add({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "patch": None, "items": len(items),
                             "verdict": "profiled"})
    d = prof.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(d, indent=1, ensure_ascii=False) if args.json else P.render(prof))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .program import backends as B
    from .program import evaluate as E
    from .program import objective as O

    obj = O.load(args.objective)
    model = args.model_dir or (str(obj.resolve(obj.model)) if obj.model else None)
    if not model and not (args.gguf or args.endpoint):
        raise ValueError("give the model directory (or set model in the goal file), or --gguf / --endpoint")
    patch = None
    if args.patch:
        _need_torch()
        from .surgery.edits import Patch
        patch = Patch.load(Path(args.patch))
    src, notes, consent = _source(args, model, obj.protocol, need_patch=patch is not None,
                                  patched_gguf=args.patched_gguf)
    psrc = None
    if args.patched_gguf and not isinstance(src, B.TorchBackend):
        def psrc():                                   # started when the original's runs are done
            return B.choose("llama-server", consent=consent, protocol=obj.protocol, gguf=args.patched_gguf,
                            server_bin=args.llama_server, parallel=args.parallel).backend
    if patch is not None and isinstance(src, B.TorchBackend):
        _check_identity(src.lm, patch, args)
    try:
        res = E.evaluate(src, obj, patch=patch, patched_source=psrc,
                         fmt=("gguf:" + Path(args.patched_gguf).stem.rsplit(".", 1)[-1]) if args.patched_gguf else None,
                         patch_name=(f"{args.patch} ({patch.label})" if patch else
                                     f"{Path(args.patched_gguf).name} (patched GGUF)" if psrc is not None else ""),
                         partial_ok=True, split=args.split, limit=args.limit, samples=args.samples, seed=args.seed,
                         guardrails=not args.no_guardrails, progress=_progress("evaluate"))
    finally:
        src.close()                                   # the patched server, if any, is closed inside evaluate
    res["backend_notes"] = notes
    res["decisions"] = consent.decisions
    if args.patch and args.patched_gguf and (Path(args.patch) / "MANIFEST.json").exists() and res.get("paired"):
        # The fix's record learns how it did as this served file (format from the GGUF's name, e.g. x.Q4_K_M.gguf).
        from . import manifest as MF
        stem = Path(args.patched_gguf).stem
        fmt = "gguf:" + (stem.rsplit(".", 1)[1] if "." in stem else stem)
        by = res.get("guardrails_by_format") or {}
        rails = {g["name"]: g for f, rs in by.items() if f != "pytorch" for g in rs.values()}
        ok = res["verdict"] == "improved" and all(g.get("holds") is not False for g in rails.values())
        MF.update(Path(args.patch), ["guardrails", fmt], MF.guardrails({"guardrails": list(rails.values())}, fmt)[fmt])
        MF.update(Path(args.patch), ["statistics", fmt], MF.statistics(res))
        MF.update(Path(args.patch), ["decision", "deployable", fmt], ok)
        res["manifest_updated"] = {"format": fmt, "deployable": ok}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False) if args.json else E.render(res))
    return 1 if res["verdict"] == "worse" else 0


def cmd_qbisect(args: argparse.Namespace) -> int:
    _need_torch()
    from .deep.contrast import Classifier
    from .deep.lm import LM
    from .deep import qbisect as QB
    from .findings import Cause, Finding, Layer, Report, Severity

    messages = _messages_arg(args)
    lm = LM.load(args.model_dir, dtype=args.dtype, device=args.device)
    qm = QB.build(lm, args.gguf, log=lambda m: print(m, file=sys.stderr))
    clf = Classifier(lm, expect_lang=args.expect_lang, max_new=args.max_new,
                     template_kwargs={"enable_thinking": False} if args.no_thinking else None)
    res = QB.bisect(qm, clf, messages, max_set=args.max_set, log=lambda m: print(m, file=sys.stderr))
    o = res.get("outcome")
    cause = {"localised": Cause.QUANTISATION, "necessary": Cause.QUANTISATION, "diffuse": Cause.QUANTISATION,
             "unconfirmed": Cause.QUANTISATION, "quantised_passes": Cause.RUNTIME}.get(o, Cause.NONE)
    fix = None
    if o in ("localised", "necessary"):
        fix = ("requantise with these tensors at a higher-bit type (llama-quantize --tensor-type <name>=q8_0) or "
               "with an importance matrix: " + ", ".join(res.get("minimal_set", [])))
    elif o == "diffuse":
        fix = "requantise the whole model at more bits (Q5_K_M / Q6_K / Q8_0), or with an importance matrix"
    elif o == "quantised_passes":
        fix = "compare llama.cpp build, backend and flags; run `wsr triage --differential`"
    sev = Severity.FAIL if cause != Cause.NONE else Severity.INFO
    rep = Report(findings=[Finding("QBISECT", Layer.DEEP, sev, QB.summary(res), cause=cause, fix=fix,
                                   source="quantisation bisection", evidence=res)])
    rep.context.update(model=lm.name, gguf=args.gguf, outcome=o)
    rep.verdict = decide(rep.findings)
    return _emit(rep, args)


def _need_torch() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise ValueError(f"deep triage needs torch and transformers ({e}); pip install 'wsr-triage[deep]'") from e


def cmd_deep(args: argparse.Namespace) -> int:
    _need_torch()
    from .deep.lm import LM
    from .deep.run import diagnose

    messages = _messages_arg(args)
    lm = LM.load(args.model_dir, dtype=args.dtype, device=args.device)
    base = LM.load(args.base, dtype=args.dtype, device=args.device) if args.base else None
    res = diagnose(lm, messages, kind=args.kind, failing=tuple(args.failing), passing=tuple(args.passing),
                   expect_lang=args.expect_lang, max_new=args.max_new,
                   template_kwargs={"enable_thinking": False} if args.no_thinking else None, base=base,
                   gguf=args.gguf, reference_dir=args.model_dir if args.gguf else None, known=not args.no_known,
                   probes=not args.no_probes,
                   out_dir=Path(args.out) if args.out else None)
    return _emit(res.report, args)


def cmd_repair(args: argparse.Namespace) -> int:
    _need_torch()
    from .deep.lm import LM
    from .deep.run import diagnose
    from .surgery import regress as R
    from .surgery.playbook import repair, write

    if not args.out:
        raise ValueError("wsr repair needs --out (the patch and report are written there)")
    out = Path(args.out)
    messages = _messages_arg(args)
    # Everything that can be wrong with the arguments is said before the model is loaded and diagnosed.
    if args.sampled and not args.deep:
        raise ValueError("--sampled is part of deep mode: add --deep")
    if (args.agent_only or args.resume) and not args.agent:
        raise ValueError("--agent-only and --resume are for the agent: add --agent")
    provider = _agent_provider(args) if args.agent and not args.check else None    # no SDK, key or model: now
    tk = {"enable_thinking": False} if args.no_thinking else None
    lm = LM.load(args.model_dir, dtype=args.dtype, device=args.device)
    res = diagnose(lm, messages, kind=args.kind, failing=tuple(args.failing), passing=tuple(args.passing),
                   expect_lang=args.expect_lang, max_new=args.max_new, template_kwargs=tk,
                   known=not args.no_known, probes=not args.no_probes, out_dir=out / "deep")
    print(f"diagnosis: {res.summary} (confidence: {res.confidence})")
    kw = dict(expect_lang=args.expect_lang, max_new=args.max_new, template_kwargs=tk,
              budget=R.Budget(args.max_drift, args.min_agreement, args.allow_safety_loss), keep=tuple(args.keep),
              mode="deep" if args.deep else "fast", confirm=tuple(args.confirm), serve=args.serve_dtype,
              sampled={"n": args.samples, "max_new": args.sampled_max_new} if args.sampled else None,
              log=None if args.json else print)
    if args.check:
        from .surgery import edits as E
        from .surgery.playbook import check
        given = E.Patch.load(Path(args.check))
        _check_identity(lm, given, args)
        rep = check(lm, res, given, **kw)
    else:
        rep = repair(lm, res, **kw, playbook=not args.agent_only, steering=args.steering)
        if provider is not None and rep.status == "could_not_fix":
            from .agent import run
            if not args.json:
                print(f"\nrule-based surgery could not fix it; handing over to the agent ({provider.name})")
            budget, limits, _ = _agent_limits(args)
            rep = run(lm, res, rep, provider, budget=budget, limits=limits, out=out,
                      resume=Path(args.resume) if args.resume else None,
                      log=None if args.json else print)
    hint = _rerun_hint(rep, out)
    if hint:
        rep.checks["agent"]["rerun"] = hint
        rep.next_step = (rep.next_step + "; " if rep.next_step else "") + f"rerun with: {hint}"
    write(rep, out, lm, save_model=args.save_model)
    if args.json:
        print((out / "repair.json").read_text(encoding="utf-8"))
    else:
        print()
        print(f"{rep.status.replace('_', ' ').upper()}: {rep.summary}")
        if rep.next_step:
            print(f"next step: {rep.next_step}")
        for r in (rep.checks or {}).get("agent", {}).get("capability_requests", []):
            print(f"agent needed: {r['capability']} ({r['category']}): {r['why_needed'][:200]}"
                  + (f" -> {r['unlock']}" if r.get("unlock") else "")
                  + (f" [unverified proposal saved: agent_proposals/{r['proposal']['id']}]" if r.get("proposal") else ""))
        print(f"report: {out / 'REPAIR.md'} ({rep.seconds:.0f} s)")
    return 0 if rep.status in ("fixed", "verified", "steered") else 1


def _rerun_hint(rep, out: Path) -> Optional[str]:
    """The exact command that continues this run with the unlocks the agent asked for."""
    import shlex

    from .agent.workbench import UNLOCKS
    ag = (rep.checks or {}).get("agent") or {}
    want = {u for r in ag.get("capability_requests") or [] for u in UNLOCKS.get(r["category"], ((), ""))[0]}
    have = set(ag.get("unlocked") or [])
    if not want - have:
        return None
    argv, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a in ("--agent-allow", "--resume"):
            skip = True
            continue
        if a.startswith(("--agent-allow=", "--resume=")):
            continue
        argv.append(a)
    return shlex.join(["wsr", *argv, "--agent-allow", ",".join(sorted(want | have)), "--resume", str(out)])


def _check_identity(lm, patch, args: argparse.Namespace, *, patched: bool = False) -> dict:
    """Refuse a patch made for other weights, or one that cannot show which weights it was made for, unless the
    user explicitly allows it (--allow-base-mismatch / --allow-unverified-legacy). Fail closed."""
    from .surgery.edits import identity_check

    r = identity_check(lm, patch, patched=patched)
    what = "the patched model this patch produced" if patched else "the model this patch was made for"
    if r["verdict"] == "unverified":
        if not getattr(args, "allow_unverified_legacy", False):
            raise ValueError(f"cannot verify that this is {what}: {r['why']}. Rerun with --allow-unverified-legacy "
                             "to use it anyway (its checks then say nothing about these weights)")
        print(f"wsr: WARNING: {r['why']}; using it anyway (--allow-unverified-legacy). Nothing shows that the checks "
              "it passed apply to these weights", file=sys.stderr)
    elif r["verdict"] == "different":
        if not getattr(args, "allow_base_mismatch", False):
            raise ValueError(f"this is not {what}: {r['why']}. Applied here it is not the edit that was checked "
                             "(--allow-base-mismatch applies it anyway)")
        print(f"wsr: WARNING: this is not {what} ({r['why']}); using it anyway (--allow-base-mismatch)",
              file=sys.stderr)
    if r.get("metadata_differs"):
        print(f"wsr: note: these files differ from the model the patch was made on: "
              f"{', '.join(r['metadata_differs'])}", file=sys.stderr)
    return r


def _identity_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--allow-base-mismatch", action="store_true",
                   help="use a patch on weights that differ from the ones it was made for (its checks then do not "
                        "apply; the output records it)")
    p.add_argument("--allow-unverified-legacy", action="store_true",
                   help="use a patch that records no exact identity of its model (made by an older wsr): nothing "
                        "shows its checks apply to these weights")


def _agent_allow(args: argparse.Namespace) -> tuple[str, ...]:
    allow = tuple(x.strip() for x in (getattr(args, "agent_allow", "") or "").split(",") if x.strip())
    bad = sorted(set(allow) - set(AGENT_ALLOWS))
    if bad:
        raise ValueError(f"--agent-allow: unknown {', '.join(bad)} (known: {', '.join(AGENT_ALLOWS)})")
    return allow


def _agent_limits(args: argparse.Namespace, goal_budget: Optional[dict] = None):
    """(workbench budget, loop limits, practice evals). A flag wins over the goal file's `budget`, which wins over
    the built-in default."""
    from .agent import AgentLimits, Budget

    gb = goal_budget or {}

    def pick(flag: str, key: str, default: float) -> float:
        v = getattr(args, flag, None)
        return v if v is not None else gb.get(key, default)
    return (Budget(verifications=int(pick("agent_verifications", "verifications", 6)), allow=_agent_allow(args)),
            AgentLimits(args.agent_turns, float(pick("agent_budget_usd", "usd", 5.0)), args.agent_max_tokens),
            int(pick("practice_evals", "practice_evals", 10)))


def _agent_provider(args: argparse.Namespace):
    _agent_allow(args)                                  # a mistyped unlock is said now, not after the diagnosis
    if args.agent_provider == "anthropic":
        from .agent import AnthropicProvider
        effort = args.agent_effort or "high"
        if effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError(f"--agent-effort {effort!r}: anthropic takes low, medium, high, xhigh or max")
        key = os.environ.get(args.agent_key_env) if args.agent_key_env else None
        if args.agent_key_env and not key:
            raise ValueError(f"--agent-key-env {args.agent_key_env}: that variable is not set")
        try:
            return AnthropicProvider(args.agent_model or "claude-opus-5-5", effort=effort, api_key=key,
                                     max_tokens=getattr(args, "agent_turn_tokens", 16000))
        except ValueError as e:
            raise ValueError(str(e)) from None
    from .agent import OpenAIProvider
    if args.agent_price is not None and len(args.agent_price) not in (2, 3):
        raise ValueError("--agent-price takes IN OUT or IN OUT CACHED (USD per million tokens)")
    extra = None
    if getattr(args, "agent_extra", None):
        try:
            extra = json.loads(args.agent_extra)
        except ValueError as e:
            raise ValueError(f"--agent-extra is not valid JSON: {e}") from None
        if not isinstance(extra, dict):
            raise ValueError("--agent-extra must be a JSON object")
    try:
        return OpenAIProvider(args.agent_model, base_url=args.agent_base_url,
                              key_env=args.agent_key_env or "OPENAI_API_KEY", effort=args.agent_effort,
                              price_in=args.agent_price[0] if args.agent_price else None,
                              price_out=args.agent_price[1] if args.agent_price else None,
                              price_cached=args.agent_price[2] if args.agent_price and len(args.agent_price) > 2 else None,
                              max_tokens=getattr(args, "agent_turn_tokens", 16000),
                              timeout=getattr(args, "agent_timeout", 600.0), extra_body=extra)
    except ValueError as e:
        raise ValueError(str(e)) from None


def cmd_patch(args: argparse.Namespace) -> int:
    if args.action in ("show", "check"):
        from . import manifest as MF

        d = Path(args.model_dir)
        rec = MF.read(d)
        if args.action == "show":
            print(json.dumps(rec, indent=1, ensure_ascii=False) if getattr(args, "json", False) else MF.render(rec))
            return 0
        lm = None
        if args.patch_dir:
            _need_torch()
            from .deep.lm import LM
            lm = LM.load(args.patch_dir, dtype="float32", device="cpu")
        r = MF.check(d, lm)
        if getattr(args, "json", False):
            print(json.dumps(r, indent=1, default=str))
        else:
            print(("OK: " if r["ok"] else "PROBLEMS: ") + str(d))
            for p in r["problems"]:
                print(f"  - {p}")
            if lm is not None and r["ok"]:
                print(f"  this is the model the fix was made for ({lm.name})")
            print(f"  decision: {(rec.get('decision') or {}).get('status')}")
        return 0 if r["ok"] else 1
    if args.action == "export":
        from .consent import Consent
        from .program.backends import find_llama_server
        from .surgery.export import export

        if not args.gguf:
            raise ValueError("export needs --gguf OUT.gguf")
        info = export(Path(args.model_dir), Path(args.patch_dir) if args.patch_dir else None, Path(args.gguf),
                      quant=args.quant, consent=Consent.from_args(args), llama_cpp=args.llama_cpp,
                      server=find_llama_server(args.llama_server),
                      allow_base_mismatch=args.allow_base_mismatch,
                      allow_unverified_legacy=args.allow_unverified_legacy)
        print(f"wrote {info['gguf']} ({info['quant']}, {info['bytes'] / 1e9:.2f} GB; converter llama.cpp {info['tag']})")
        if args.patch_dir:
            print("check what survives: wsr evaluate --objective <goal> --backend llama-server --gguf <unpatched.gguf> "
                  f"--patched-gguf {info['gguf']}")
        else:
            print(f"the unpatched file, for comparison: wsr evaluate ... --gguf {info['gguf']} --patched-gguf <patched.gguf>")
        return 0
    if not args.patch_dir or not args.out:
        raise ValueError(f"{args.action} needs the patch directory and --out")
    _need_torch()
    from .deep.lm import LM
    from .surgery import edits as E

    from .surgery.playbook import save_patched_model, serve_dtype

    lm = LM.load(args.model_dir, dtype="float32", device="cpu")
    patch = E.Patch.load(Path(args.patch_dir))
    # apply: the model it was made for; revert: that model with this patch applied (identity.json has both).
    ident = _check_identity(lm, patch, args, patched=args.action == "revert")
    if args.action == "revert":
        patch.deltas = [E.Delta(d.name, -d.U, d.V) for d in patch.deltas]
    # Saved in the checkpoint's own precision: the patch was checked after that rounding.
    ck = lm.checkpoint_dtype if lm.checkpoint_dtype in ("bfloat16", "float16") else None
    save_patched_model(lm, patch, Path(args.out), serve_dtype(lm, ck) if ck else None)
    (Path(args.out) / "wsr_patch.json").write_text(json.dumps(
        {"action": args.action, "patch": patch.label, "patch_dir": str(Path(args.patch_dir).resolve()),
         "identity_check": ident.get("verdict"),
         "allowed": [f for f in ("allow_base_mismatch", "allow_unverified_legacy") if getattr(args, f, False)]},
        indent=1), encoding="utf-8")
    checked = (patch.checks or {}).get("mode")
    print(f"{args.action}: {patch.label} ({len(patch.deltas)} tensors) -> {args.out} ({ck or 'float32'})"
          + (f"; checked in {checked} mode" if checked else ""))
    return 0


def cmd_quant(args: argparse.Namespace) -> int:
    from .deep.quant import EXPECTED_REL_ERROR, analyse

    q = analyse(args.gguf, args.reference, only=args.only)
    if getattr(args, "json", False):
        print(q.to_json())
        return 0
    n = len(q.stats)
    print(f"{n} tensors read" + (f", {len(q.skipped)} skipped (unsupported type)" if q.skipped else ""))
    bad = [s for s in q.stats if s.get("nan_inf")]
    for s in bad:
        print(f"  [FAIL] {s['name']}: {s['nan_inf']} NaN/Inf values")
    for s in q.stats:
        if s.get("zero_scale_blocks", 0) > 0.05 or s.get("repeated_blocks", 0) > 0.05:
            print(f"  [WARN] {s['name']}: zero-scale blocks {s.get('zero_scale_blocks', 0):.1%}, "
                  f"repeated blocks {s.get('repeated_blocks', 0):.1%}")
    if q.anomalies:
        print(f"Unusual against the same tensor in other layers (median / MAD), top {args.top}:")
        for a in q.anomalies[:args.top]:
            print(f"  [info] {a['tensor']}: {a['stat']} {a['value']:.4g} (layers' median {a['role_median']:.4g}, "
                  f"robust z {a['robust_z']:+.1f})")
    worst = q.worst_errors(args.top)
    fails = 0
    resc = [e for e in q.errors if str(e.get("layout", "")).endswith("+rescaled")]
    if resc:
        norms = sum(1 for e in q.errors if e.get("folded_scales"))
        print(f"{len(resc)} matrices carry per-channel scales folded in at conversion (AWQ / SmoothQuant style; the "
              f"function is unchanged){f' and {norms} norm gains the inverse' if norms else ''}: their error is "
              "measured after fitting those scales out")
    if worst:
        print(f"Quantisation error vs {args.reference} (relative Frobenius), worst {len(worst)}:")
        for w in worst:
            exp = w.get("expected", EXPECTED_REL_ERROR.get(w["type"], 0.05))
            tag = "FAIL" if w["ratio_to_expected"] >= 3 else "ok  "
            fails += tag == "FAIL"
            where = "the same matrix in other layers" if w.get("baseline", "type") != "type" else "the type"
            print(f"  [{tag}] {w['tensor']} {w['type']}: {w['rel_error']:.2%} (normal for {where} ~{exp:.1%}; "
                  f"cos {w['cosine']:.4f}; layout {w['layout']})")
        errs = [e for e in q.errors if "error" in e]
        for e in errs[:5]:
            print(f"  [skip] {e['tensor']}: {e['error']}")
    elif args.reference:
        print("no tensor matched a reference name")
    else:
        print("pass --reference <hf dir> to measure quantisation error against the original weights")
    return 1 if (bad or fails) else 0


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        except (OSError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    from .consent import NeedsConsent

    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("wsr: interrupted", file=sys.stderr)
        return 130
    except NeedsConsent as e:
        print(f"wsr: {e}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as e:
        print(f"wsr: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
