"""The full layered triage pipeline, usable as a library (integrated mode).

    from wsr.pipeline import TriageInput, triage
    report = triage(TriageInput(model_dir=Path("..."), endpoint=Endpoint(...), messages=[...], params={...}))
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from . import known_issues
from .config_checks import ModelSpec, check_config
from .file_integrity import check_files
from .findings import Cause, Finding, Layer, Report, Severity
from .output import check_output
from .media import check_media, content_text, has_images
from .probes import (Endpoint, EndpointError, differential, is_penalised, penalty_ab, probe_determinism,
                     run_probes)
from .request_checks import check_request
from .response import check_response
from .verdict import decide


@dataclass
class TriageInput:
    model_dir: Optional[Path] = None
    endpoint: Optional[Endpoint] = None
    messages: Optional[list[dict]] = None
    params: dict[str, Any] = field(default_factory=dict)
    response: Optional[Any] = None          # a saved response instead of a live call
    model_name: str = ""
    backend: Optional[str] = None
    upstream: Optional[str] = None
    expect_lang: Optional[str] = None
    expect_script: Optional[str] = None
    expect_json: bool = False
    check_echo: bool = False
    preset: str = "chat"
    scan_values: bool = False
    verify_hash: bool = False
    probes: Optional[list[str]] = None      # None = default set when an endpoint is given; [] = none
    run_differential: bool = False
    alts: list[Endpoint] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    known_issues_file: Optional[Path] = None
    extra_findings: list[Finding] = field(default_factory=list)  # e.g. stream checks on a saved stream
    target_ctx: Optional[int] = None         # None: the server's context size, else 32768
    server_args: Optional[str] = None        # the server's launch command line (llama-server / vLLM flags)


def _is_full_precision(spec: Optional[ModelSpec], model_dir: Optional[Path]) -> bool:
    if spec is None or spec.kind != "hf" or model_dir is None:
        return False
    cfg = {}
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return "quantization_config" not in cfg


def _prompt_tokens_estimate(messages: list[dict]) -> int:
    # Text only: base64 image data is not prompt text (a 30 KB JPEG is ~260 image tokens).
    return sum(len(content_text(m.get("content"))) for m in messages) // 4


def _stop_string_cut(norm, params: dict) -> list[Finding]:
    from .output import truncation_score

    stop = params.get("stop") or (params.get("options") or {}).get("stop")
    if not stop or norm.finish_reason not in ("stop", "end_turn", "STOP") or norm.tool_calls:
        return []
    t = norm.text.strip()
    if t and truncation_score(t, None) < 0.5:
        return []
    return [Finding("ENDED_ON_STOP_STRING", Layer.RESPONSE, Severity.WARN,
                    f"the reply stopped normally but mid-thought ({t[-40:]!r}) with stop strings {stop!r} set: "
                    "a stop string probably cut it", cause=Cause.SYMPTOM,
                    fix="narrow the stop strings; rerun without them to confirm", source="wsr",
                    evidence={"stop": stop})]


def _confirm_prefill(rep: Report, text: str, messages: Optional[list[dict]]) -> None:
    """The reply starts with the prefill's markup: the prefill is the leak, observed, not inferred."""
    if not messages or messages[-1].get("role") != "assistant":
        return
    pre = str(messages[-1].get("content") or "").strip()
    if not pre or not text.lstrip().startswith(pre):
        return
    for f in rep.findings:
        if f.code == "PREFILL_REASONING_MARKUP":
            f.severity = Severity.FAIL
            f.evidence["confirmed"] = True
            f.message += " (confirmed: the reply starts with the prefill)"


def triage(inp: TriageInput) -> Report:
    rep = Report()
    rep.extend(inp.extra_findings)
    spec: Optional[ModelSpec] = None

    # 1-2. static checks on the model files
    if inp.model_dir:
        files = check_files(inp.model_dir, scan_values=inp.scan_values, upstream=inp.upstream,
                            verify_hash=inp.verify_hash)
        if inp.model_dir.is_dir() and (inp.model_dir / "config.json").exists():
            # A config-only directory is a normal way to triage a served model.
            for f in files:
                if f.code == "NO_WEIGHTS":
                    f.severity, f.fix = Severity.SKIP, None
                    f.message = "no weight files here; file-integrity checks skipped"
        rep.extend(files)

    if inp.endpoint and inp.endpoint.reachable():
        rep.findings.append(Finding("ENDPOINT_UNREACHABLE", Layer.RESPONSE, Severity.FAIL,
                                    inp.endpoint.reachable(), cause=Cause.RUNTIME,
                                    fix="start the server, or check --endpoint (host, port, http vs https)"))
        inp = replace(inp, endpoint=None)
    backend = inp.backend
    if inp.endpoint and not backend:
        try:
            backend = inp.endpoint.detect_backend()
        except EndpointError:
            backend = None
    ctx = inp.target_ctx or (inp.endpoint.server_ctx() if inp.endpoint else None) or 32768
    if inp.model_dir:
        spec, cfg = check_config(inp.model_dir, upstream=inp.upstream, target_ctx=ctx)
        rep.extend(cfg)
    rep.context.update(backend=backend, model=inp.model_name or (inp.endpoint.model if inp.endpoint else None))

    # Launch flags and server slots (field reports: KV-cache quantisation, flash attention,
    # context split across slots, context shift, RoPE overrides)
    from .runtime_checks import check_launch, check_slots, ollama_env, parse_server_args
    flags = parse_server_args(inp.server_args) if inp.server_args else {}
    if backend == "ollama" and inp.endpoint and any(h in inp.endpoint.url for h in ("localhost", "127.0.0.1")):
        flags = {**ollama_env(), **flags}
    if flags:
        rb = spec.meta.get(f"{spec.arch}.rope.freq_base") if spec and spec.kind == "gguf" else None
        rep.extend(check_launch(flags, ctx_trained=spec.ctx_trained if spec else None, rope_freq_base_file=rb,
                                cfg=spec.meta if spec and spec.kind == "hf" else None))
    slot_ctx = None
    if inp.endpoint and backend == "llama-server":
        try:
            props = inp.endpoint.props()
            if not any(f.code == "CONTEXT_SPLIT_ACROSS_SLOTS" for f in rep.findings):   # same fact, once
                rep.extend(check_slots(props, ctx_trained=spec.ctx_trained if spec else None))
            slot_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        except EndpointError:
            pass

    # Server-side template vs the model files' template
    if inp.endpoint and spec and spec.template:
        server_tpl = inp.endpoint.server_template()
        if server_tpl and backend == "llama-server":
            from .config_checks import _norm_template

            if _norm_template(server_tpl) != _norm_template(spec.template):
                rep.findings.append(Finding(
                    "SERVER_TEMPLATE_DIFFERS", Layer.CONFIG, Severity.WARN,
                    "the server renders prompts with a different chat template than the model files",
                    cause=Cause.CONFIG,
                    fix="start llama-server with --jinja and the model's template (--chat-template-file)",
                    source="LLMDoctor template-stale"))

    # 3. request
    messages = inp.messages
    if inp.params or messages:
        rep.extend(check_request(inp.params, backend=backend,
                                 prompt_tokens=_prompt_tokens_estimate(messages) if messages else None,
                                 ctx_trained=spec.ctx_trained if spec else None, messages=messages,
                                 template=spec.template if spec else None))

    if messages and has_images(messages, inp.params):
        vision = None
        if inp.endpoint and backend == "llama-server":
            mods = inp.endpoint.props().get("modalities")
            vision = mods.get("vision") if isinstance(mods, dict) else None
        rep.extend(check_media(messages, inp.params, api=inp.endpoint.api if inp.endpoint else None,
                               vision=vision))

    # 3-4. the failing call itself (live or saved)
    raw = inp.response
    if raw is None and inp.endpoint and messages:
        try:
            raw, _ = inp.endpoint.chat(messages, **dict(inp.params))
        except EndpointError as e:
            rep.findings.append(Finding("ENDPOINT_UNREACHABLE", Layer.RESPONSE, Severity.FAIL, str(e),
                                        cause=Cause.RUNTIME))
    text = ""
    if raw is not None:
        max_tokens = inp.params.get("max_tokens") or (inp.params.get("options") or {}).get("num_predict")
        sent = _prompt_tokens_estimate(messages) if messages else None
        norm, resp = check_response(raw, max_tokens=max_tokens, tools=inp.params.get("tools"),
                                    sent_prompt_tokens=sent if sent and sent > 500 else None,
                                    context_window=(inp.params.get("options") or {}).get("num_ctx"))
        rep.extend(resp)
        from .runtime_checks import check_context_shift
        rep.extend(check_context_shift(norm.prompt_tokens, norm.completion_tokens,
                                       slot_ctx if isinstance(slot_ctx, int) else None))
        text = norm.text
        rep.extend(_stop_string_cut(norm, inp.params))
        _confirm_prefill(rep, norm.text, messages)
        rep.context["response_text"] = text[:2000]
        if norm.text.strip() or not norm.tool_calls:
            prompt_text = "\n".join(content_text(m.get("content")) for m in messages or []
                                    if m.get("role") != "assistant")
            # finish_reason is judged by the response layer; the output layer
            # adds only its structural truncation heuristics, so no double count.
            rep.extend(check_output(
                norm.text, preset=inp.preset, finish_reason=None, expect_json=inp.expect_json,
                expect_script=inp.expect_script, expect_lang=inp.expect_lang,
                prompt=prompt_text if inp.check_echo else None, penalised=is_penalised(inp.params),
                has_images=has_images(messages, inp.params),
                request=next((content_text(m.get("content")) for m in reversed(messages or [])
                              if m.get("role") == "user"), None)))

    # 5. probes and differential
    if inp.endpoint and inp.probes != []:
        rep.extend(run_probes(inp.endpoint, inp.probes))
    elif inp.endpoint and inp.run_differential and inp.endpoint.reasoning is None:
        from .probes import probe_reasoning
        try:  # the differential needs to know whether the model thinks
            rep.extend(probe_reasoning(inp.endpoint))
        except EndpointError:
            pass
    if inp.endpoint and all(inp.params.get(k) == v for k, v in inp.endpoint.no_think_params().items()):
        # The probe measured the server default; this request turns thinking off,
        # so a reasoning budget cannot be what broke it.
        for f in rep.findings:
            if f.code in ("REASONING_STARVES_ANSWER", "REASONING_RUNAWAY") and f.failed:
                f.severity = Severity.INFO
                f.message += " (server default only: this request turns thinking off)"
    failing = {f.code for f in rep.findings if f.failed and f.layer in (Layer.OUTPUT, Layer.RESPONSE)}
    if (inp.endpoint and messages and inp.run_differential and not failing and raw is not None
            and is_penalised(inp.params)):
        # Fluent but off-task under a strong penalty: only an A/B shows it.
        rep.extend(penalty_ab(inp.endpoint, messages, inp.params, text, expect_lang=inp.expect_lang))
    if inp.endpoint and messages and inp.run_differential and failing:
        try:
            rep.extend(probe_determinism(inp.endpoint, messages, inp.params))
        except EndpointError:
            pass
        from .probes import probe_concurrency, server_slots
        if (server_slots(inp.endpoint) or 1) > 1:
            try:  # a failure that depends on what else the server is decoding
                rep.extend(probe_concurrency(inp.endpoint, messages, inp.params, expect_lang=inp.expect_lang))
            except EndpointError:
                pass
        rep.extend(differential(inp.endpoint, messages, inp.params, failing, alts=inp.alts,
                                expect_lang=inp.expect_lang))

    # 6. known issues
    conds = set(inp.conditions)
    p = inp.params
    opts = p.get("options") or {}
    if any(k in p for k in ("format", "response_format", "grammar", "json_schema")):
        conds.add("constrained")
    if any(k in p or k in opts for k in ("repeat_penalty", "repetition_penalty", "frequency_penalty",
                                         "presence_penalty")):
        conds.add("penalty_set")
    else:
        conds.add("no_penalty")
    if spec and spec.kind == "gguf" or backend in ("ollama", "llama-server", "lmstudio"):
        conds.add("gguf")
    codes = {f.code for f in rep.findings if f.failed}
    if "PREFILL_REASONING_MARKUP" in codes:
        conds.add("prefill_markup")
    if "REASONING_EXHAUSTED" in codes or any(c.startswith("DIFF_FLIP:thinking_off") for c in codes):
        conds.add("reasoning_exhausted")
    if any(f.code == "TEMPLATE_MATCHES_UPSTREAM" for f in rep.findings):
        conds.add("template_verified")
    model = " ".join(x for x in (inp.model_name, inp.endpoint.model if inp.endpoint else "",
                                 spec.arch if spec else "", str(spec.path) if spec else "") if x)
    sym = {f.code for f in rep.findings if f.failed and f.layer in (Layer.OUTPUT, Layer.RESPONSE)}
    if sym:
        db = known_issues.load_db(inp.known_issues_file)
        rep.extend(known_issues.match(model, sym, backend=backend, conditions=conds, db=db))

    # 7. verdict
    clean_static = not any(f.failed for f in rep.findings if f.layer in (Layer.FILE, Layer.CONFIG, Layer.REQUEST))
    rep.verdict = decide(rep.findings, full_precision_reference=_is_full_precision(spec, inp.model_dir)
                         and clean_static)
    return rep


def write_handoff(rep: Report, inp: TriageInput, out_dir: Path) -> Path:
    """The package the deep weight-level stage consumes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failing_request.json").write_text(json.dumps(
        {"model": rep.context.get("model"), "messages": inp.messages, "params": inp.params}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(rep.as_dict(), indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    if inp.response is not None:
        (out_dir / "response.json").write_text(json.dumps(inp.response, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    return out_dir
