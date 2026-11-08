"""Verdict engine: symptoms + causes -> one verdict with evidence and a fix.

Output detectors report *symptoms*; file / config / request / response /
probe / known-issue checks report *causes*. A cause only wins if it can
explain an observed symptom group. Empirical evidence (a differential rerun
that flipped the result) beats static evidence, then earlier layers beat
later ones. Weight-level is claimed only when every non-weight explanation has
been ruled out on a full-precision model with a valid template.
"""
from __future__ import annotations

from typing import Any

from .findings import CAUSE_PRIORITY, Cause, Finding, Layer, Severity

GROUPS: dict[str, set[str]] = {
    "blank": {"EMPTY", "EMPTY_OUTPUT", "TRUNCATED_EMPTY", "TOO_SHORT", "REASONING_EXHAUSTED"},
    "loop": {"TAIL_LOOP", "REPETITION", "AGENT_LOOP"},
    "garbage": {"SYMBOL_SOUP", "LOW_ENTROPY", "TOKEN_SALAD", "BROKEN_UTF8", "OUTPUT_PARSE_REJECTED",
                "REPETITION_AVOIDANCE", "PENALTY_SHAPE_LOSS"},
    "template_leak": {"CONTROL_TOKEN_LEAK", "INVENTED_TURN", "REASONING_LEAK", "PROMPT_ECHO"},
    "language": {"LANG_MISMATCH", "SCRIPT_MISMATCH"},
    "refusal": {"REFUSAL_TEXT", "REFUSAL", "CONTENT_FILTERED"},
    "vision": {"IMAGE_NOT_SEEN", "IMAGE_REJECTED"},
    "endpoint": {"ENDPOINT_UNREACHABLE"},
    "structure": {"INVALID_JSON", "JSON_WRAPPED", "ENDED_ON_STOP_STRING", "TRUNCATED", "TOOL_CALL_TRUNCATED",
                  "TOOL_ARGS_UNPARSEABLE", "TOOL_UNKNOWN", "TOOL_ARGS_INVALID", "MISSING_FINISH_REASON",
                  "STREAM_CUT", "STREAM_ERROR_EVENT", "STREAM_MALFORMED_CHUNK", "STREAM_TOOL_ARGS_FRAGMENTED",
                  "STREAM_TOOL_NO_NAME"},
}
ALL = set(GROUPS)

EXPLAINS: dict[str, set[str]] = {
    # config
    "EOS_MISMATCH": {"loop", "template_leak", "structure"},
    "NO_STOP_TOKENS": {"loop", "template_leak", "structure"},
    "STOP_ID_OUT_OF_RANGE": {"loop", "template_leak"},
    "TEMPLATE_MISSING": ALL - {"garbage"},
    "TEMPLATE_MISSING_VS_UPSTREAM": ALL - {"garbage"},
    "TEMPLATE_RENDER_ERROR": ALL,
    "NO_GENERATION_PROMPT": {"blank", "template_leak", "language"},
    "TEMPLATE_STALE": {"template_leak", "blank", "loop", "structure", "language"},
    "TEMPLATE_CONFLICT": {"template_leak", "blank", "loop", "structure", "language"},
    "DOUBLE_BOS": {"loop", "garbage", "blank"},
    "TOKENIZER_EXCEEDS_EMBEDDINGS": {"blank", "garbage"},
    "LOW_BIT_SMALL_MODEL": {"loop", "garbage", "language"},
    "MEMORY_NO_FIT": {"blank", "garbage"},
    # request
    "REPEAT_PENALTY_HIGH": {"loop", "garbage", "structure"},
    "REPEAT_PENALTY_BELOW_ONE": {"loop"},
    "TEMPERATURE_HIGH": {"garbage", "language"},
    "MIN_P_HIGH": {"loop"},
    "FREQUENCY_PENALTY_EXTREME": {"garbage", "language"},
    "PRESENCE_PENALTY_EXTREME": {"garbage", "language"},
    "STOP_TOO_BROAD": {"blank", "structure"},
    "MAX_TOKENS_TINY": {"blank", "structure"},
    "PROMPT_EXCEEDS_CONTEXT": {"loop", "blank", "language", "structure", "template_leak"},
    "INPUT_TRUNCATED": {"loop", "blank", "language", "structure", "template_leak"},
    "PROMPT_FILLS_CONTEXT": {"loop", "blank", "language", "structure"},
    "CONTEXT_OVERFLOW": {"blank", "structure"},
    "ENDPOINT_UNREACHABLE": {"endpoint"},
    "IMAGE_FIELD_IGNORED": {"vision"},
    "IMAGE_URL_NOT_OBJECT": {"vision", "blank"},
    "IMAGE_BAD_DATA": {"vision", "blank"},
    "IMAGE_REMOTE_URL": {"vision", "blank"},
    "IMAGE_NO_VISION": {"vision", "blank"},
    "IMAGE_REJECTED": {"vision", "blank"},
    "VISION_IGNORED": {"vision"},
    "VISION_UNSUPPORTED": {"vision", "blank"},
    "OLLAMA_DEFAULT_CONTEXT": {"loop", "language", "structure"},
    "CONTEXT_ABOVE_TRAINED": {"loop", "garbage", "language"},
    # response (only when no loop is present: a loop always ends at the cap)
    "TRUNCATED": {"structure"},
    "TRUNCATED_EMPTY": {"blank"},
    "REASONING_EXHAUSTED": {"blank"},
    "REASONING_STARVES_ANSWER": {"blank"},
    "PROMPT_CACHE_CHANGES_OUTPUT": {"blank", "loop", "garbage", "language", "structure"},
    "STREAM_CUT": {"structure", "blank"},
    "STREAM_ERROR_EVENT": {"structure", "blank"},
    "STREAM_MALFORMED_CHUNK": {"structure", "garbage"},
    "STREAM_TOOL_ARGS_FRAGMENTED": {"structure"},
    "STREAM_TOOL_NO_NAME": {"structure"},
    "PREFILL_REASONING_MARKUP": {"template_leak", "structure"},
    "REASONING_RUNAWAY": {"blank", "loop"},
    "REASONING_UNPARSED": {"template_leak", "structure"},
    "SERVER_DOUBLE_BOS": {"loop", "garbage", "blank"},
    "SERVER_NO_BOS": {"loop", "garbage", "blank"},
    "TOOL_CALL_TRUNCATED": {"structure"},
    "AT_LIMIT": {"structure"},
    "NEAR_LIMIT": {"structure"},
    "CONTENT_FILTERED": {"blank", "refusal"},
    "REFUSAL": {"refusal", "blank"},
    "RATE_LIMITED": {"blank"},
    "PROVIDER_ERROR": {"blank"},
    "MALFORMED_RESPONSE": {"blank", "structure"},
    "MISSING_FINISH_REASON": {"blank", "structure"},
    # probes
    "STOP_PROBE_RUNON": {"loop", "template_leak", "structure"},
    "SMOKE_GARBAGE": {"garbage"},
    "SMOKE_TEMPLATE_LEAK": {"template_leak"},
    "SMOKE_BLANK": {"blank"},
    "TOOL_PROBE_FAIL": {"structure"},
    "VISION_WRONG": {"vision"},
    "SERVER_TEMPLATE_DIFFERS": {"template_leak", "blank", "loop", "structure", "language"},
    "UNKNOWN_FINISH_REASON": {"blank", "structure"},
    "STREAM_CONTENT_AFTER_FINISH": {"structure"},
    "STREAM_TOOL_ID_CHANGED": {"structure"},
    "STREAM_TOOL_INDEX_GAP": {"structure"},
    # conversion and launch (field reports, runtime_checks.py). The slot split itself
    # explains nothing until a request actually reaches the slot: CONTEXT_SHIFTED does.
    "PRETOKENIZER_DEFAULT": {"garbage", "language", "loop"},
    "SOFTCAP_MISSING": {"garbage", "loop"},
    "FP16_OVERFLOW_RISK": {"blank", "garbage"},
    "ROPE_OVERRIDE_DIFFERS": {"garbage", "loop", "language"},
    "KV_CACHE_QUANTISED": {"garbage", "loop", "structure"},
    "CONTEXT_SHIFTED": {"loop", "language", "structure", "garbage", "template_leak", "blank"},
}
_CAP_CODES = {"TRUNCATED", "AT_LIMIT", "NEAR_LIMIT", "TOOL_CALL_TRUNCATED"}
# File findings that say nothing about the bytes the server actually loaded.
_NOT_CORRUPTION = {"NO_WEIGHTS", "UPSTREAM_UNREACHABLE", "UPSTREAM_MISSING"}

_UNEXPLAINED_HINT = {
    "vision": "vision: run `wsr probe --only vision` to check the server reads images at all",
    "template_leak": "template_leak: control tokens / extra turns with no template or EOS cause found",
    "loop": "loop: no static cause; run `--differential` (greedy vs sampled, penalties, constraint, context)",
    "garbage": "garbage: re-download the file, then compare backends (CPU vs GPU) and quants with --alt",
    "blank": "blank: check the reasoning parser and stop strings; run `wsr probe --only smoke,stop`",
    "language": "language: check that the system prompt survives (context size) and the template's system role",
    "refusal": "refusal: run `--differential` (system prompt off); persistent over-refusal is a surgery case",
    "structure": "structure: raise max_tokens, check the tool / JSON parser on the server",
}
_RUNTIME_SUFFIX = (" — the model files are correct, so the server is not using them: check its template and stop "
                   "settings (llama-server --jinja, Ollama TEMPLATE / PARAMETER stop) or run "
                   "`wsr probe --only smoke,stop`")


def _groups_of(codes: set[str]) -> set[str]:
    return {g for g, members in GROUPS.items() if codes & members}


def _explains(f: Finding, observed: set[str], has_loop: bool) -> set[str]:
    if f.code.startswith("DIFF_FLIP"):
        return set(f.evidence.get("explains", [])) & observed or observed
    if f.code == "LOAD_DEPENDENT_OUTPUT":
        # Only when the failure itself comes and goes with load, and only the groups that changed.
        return set(f.evidence.get("explains", [])) & observed if f.severity == Severity.FAIL else set()
    if f.code.startswith("KNOWN_ISSUE:"):
        return _groups_of(set(f.evidence.get("matched_symptoms", []))) & observed
    if f.layer == Layer.FILE and f.severity == Severity.FAIL and f.code not in _NOT_CORRUPTION:
        return observed
    if has_loop and f.code in _CAP_CODES:
        return set()
    return EXPLAINS.get(f.code, set()) & observed


def decide(findings: list[Finding], *, full_precision_reference: bool = False) -> dict[str, Any]:
    symptom_codes = {f.code for f in findings if f.failed and f.cause == Cause.SYMPTOM}
    # Response-layer failures are symptoms too (blank, truncated...) as well as causes.
    symptom_codes |= {f.code for f in findings if f.failed and f.layer == Layer.RESPONSE}
    observed = _groups_of(symptom_codes)
    # A run-on (loop or talking past the turn end) always ends at the cap, so
    # the cap is a consequence there, not a cause.
    # Garbage, too, runs until the cap stops it.
    has_loop = bool(observed & {"loop", "template_leak", "garbage"})

    candidates = [f for f in findings if f.failed and f.cause not in (Cause.SYMPTOM, Cause.NONE)]
    scored = []
    for f in candidates:
        ex = _explains(f, observed, has_loop)
        empirical = f.code.startswith("DIFF_FLIP") or f.layer == Layer.PROBE
        # A differential flip, or a cause the response itself confirms, outranks static hypotheses.
        confirmed = f.code.startswith("DIFF_FLIP") or bool(f.evidence.get("confirmed"))
        rank = (3 if confirmed else 0) + (2 if ex else 0) + (1 if f.severity == Severity.FAIL else 0)
        prio = CAUSE_PRIORITY.index(f.cause) if f.cause in CAUSE_PRIORITY else len(CAUSE_PRIORITY)
        scored.append((rank, len(ex), -prio, empirical, f, ex))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    scored = [(r, p, e, f, ex) for r, _, p, e, f, ex in scored]

    explaining = [(f, ex) for rank, _, _, f, ex in scored if ex]
    other = [f for _, _, _, f, ex in scored if not ex]

    if not observed:
        return {
            "verdict": "no_failure_observed",
            "summary": "the output looks healthy" + (f"; {len(other)} latent issue(s) found" if other else ""),
            "symptoms": [],
            "primary": None,
            "contributing": [],
            "other_issues": [_brief(f) for f in other],
            "next_steps": ["pass the exact failing prompt and request parameters (--prompt / --request)"]
            if not symptom_codes else [],
        }

    if explaining:
        primary, ex = explaining[0]
        contributing = [_brief(f) for f, _ in explaining[1:]]
        verdict = primary.cause.value
        summary = f"{', '.join(sorted(ex))} output explained by {primary.code}: {primary.message}"
        steps = [primary.fix] if primary.fix else []
        if primary.cause in (Cause.CONFIG, Cause.REQUEST, Cause.SAMPLING, Cause.FILE):
            steps.append("apply the fix and rerun `wsr triage` with the same prompt to confirm the flip")
        unexplained = observed - set().union(*(e for _, e in explaining))
        config_checked = any(f.layer == Layer.CONFIG for f in findings)
        for g in sorted(unexplained):
            steps.append(_UNEXPLAINED_HINT.get(g, f"{g}: no cause found yet; run the differential") +
                         (_RUNTIME_SUFFIX if g == "template_leak" and config_checked else ""))
        if unexplained:
            summary += f"; still unexplained: {', '.join(sorted(unexplained))}"
        return {
            "verdict": verdict,
            "summary": summary,
            "symptoms": sorted(symptom_codes),
            "symptom_groups": sorted(observed),
            "unexplained_groups": sorted(unexplained),
            "primary": _brief(primary),
            "contributing": contributing,
            "other_issues": [_brief(f) for f in other],
            "next_steps": steps,
        }

    if "template_leak" in observed and any(f.layer == Layer.CONFIG for f in findings):
        # Control tokens / invented turns with correct model files: the server
        # is rendering or stopping differently. Never a weight-level verdict.
        return {
            "verdict": Cause.RUNTIME.value,
            "summary": ("control tokens / invented turns although the model files' template and stop tokens are "
                        "correct: the serving runtime is not using them"),
            "symptoms": sorted(symptom_codes),
            "symptom_groups": sorted(observed),
            "primary": None,
            "contributing": [],
            "other_issues": [_brief(f) for f in other],
            "next_steps": [
                "check the server's template and stop settings: llama-server --jinja / --chat-template-file, "
                "Ollama TEMPLATE and PARAMETER stop, vLLM --chat-template",
                "run `wsr probe <url> --model <m> --only smoke,stop` against the server",
                "if the model is a GGUF, compare its embedded template: `wsr config <file.gguf> --upstream <repo>`",
            ],
        }

    if full_precision_reference:
        return {
            "verdict": Cause.WEIGHT_LEVEL.value,
            "summary": (f"{', '.join(sorted(observed))} output persists on a full-precision model with a valid "
                        "template and neutral request settings; no file, config, request or runtime cause found"),
            "symptoms": sorted(symptom_codes),
            "symptom_groups": sorted(observed),
            "primary": None,
            "contributing": [],
            "other_issues": [_brief(f) for f in other],
            "next_steps": ["deep weight-level diagnosis: `wsr deep <model-dir> --prompt ...` (or the --handoff "
                           "package)",
                           "check the known-issue database for this model family upstream"],
        }

    return {
        "verdict": "unexplained",
        "summary": f"{', '.join(sorted(observed))} output with no cause found in the static checks",
        "symptoms": sorted(symptom_codes),
        "symptom_groups": sorted(observed),
        "primary": None,
        "contributing": [],
        "other_issues": [_brief(f) for f in other],
        "next_steps": [
            "run the endpoint probes and differential reruns: `wsr triage --endpoint ... --differential`",
            "compare against a full-precision reference: `--alt ref=<url>,<model>`",
            "pass --model-dir so file and template checks can run",
        ],
    }


def _brief(f: Finding) -> dict[str, Any]:
    d = {"code": f.code, "cause": f.cause.value, "severity": f.severity.value, "message": f.message}
    if f.fix:
        d["fix"] = f.fix
    if f.source:
        d["source"] = f.source
    return d
