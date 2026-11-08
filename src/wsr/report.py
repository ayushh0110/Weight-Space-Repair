"""Render a Report as text or JSON."""
from __future__ import annotations

import json
import os
import sys

from .findings import Layer, Report, Severity

_ORDER = [Layer.FILE, Layer.CONFIG, Layer.REQUEST, Layer.RESPONSE, Layer.OUTPUT, Layer.PROBE, Layer.KNOWN_ISSUE, Layer.DEEP]
_TITLE = {
    Layer.FILE: "1. File integrity", Layer.CONFIG: "2. Environment and config", Layer.REQUEST: "3. Request",
    Layer.RESPONSE: "3. Response", Layer.OUTPUT: "4. Output signals", Layer.PROBE: "5. Probes and differential",
    Layer.KNOWN_ISSUE: "6. Known issues", Layer.DEEP: "7. Deep diagnosis (weights)",
}


def _color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _tag(sev: Severity, color: bool) -> str:
    label = {"fail": "FAIL", "warn": "WARN", "info": "info", "pass": " ok ", "skip": "skip"}[sev.value]
    if not color:
        return f"[{label}]"
    code = {"fail": "31;1", "warn": "33", "info": "36", "pass": "32", "skip": "90"}[sev.value]
    return f"\033[{code}m[{label}]\033[0m"


def to_text(report: Report, verbose: bool = False) -> str:
    color = _color()
    lines: list[str] = []
    for layer in _ORDER:
        items = [f for f in report.findings if f.layer == layer]
        shown = [f for f in items if verbose or f.severity not in (Severity.PASS, Severity.SKIP)]
        ok = sum(1 for f in items if f.severity == Severity.PASS)
        if not shown and not ok:
            continue
        lines.append(f"\n{_TITLE[layer]}" + (f"  ({ok} passed)" if ok and not verbose else ""))
        for f in shown:
            score = f" {f.score:.2f}" if f.score is not None and f.severity != Severity.PASS else ""
            lines.append(f"  {_tag(f.severity, color)} {f.code}{score}: {f.message}")
            if f.fix and f.severity in (Severity.FAIL, Severity.WARN):
                lines.append(f"         fix: {f.fix}")
            if verbose and f.source:
                lines.append(f"         via: {f.source}")
    v = report.verdict
    if v:
        bar = "=" * 72
        lines += ["", bar, f"VERDICT: {v['verdict'].upper()}", f"  {v['summary']}"]
        if v.get("primary"):
            p = v["primary"]
            lines.append(f"  cause: {p['code']} ({p['cause']})")
        for c in v.get("contributing", [])[:5]:
            lines.append(f"  also:  {c['code']} ({c['cause']}): {c['message']}")
        if v.get("next_steps"):
            lines.append("  next:")
            lines += [f"    - {s}" for s in v["next_steps"]]
        if v.get("other_issues"):
            lines.append(f"  {len(v['other_issues'])} other issue(s) not linked to this failure; see above")
        lines.append(bar)
    return "\n".join(lines).lstrip("\n")


def to_json(report: Report) -> str:
    return json.dumps(report.as_dict(), indent=2, ensure_ascii=False, default=str)
