"""Shared result types.

A *Finding* is one check's outcome. Findings are either causes (file, config,
request, runtime...) or symptoms (what the output looks like). The verdict
engine in :mod:`wsr.verdict` combines them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"
    PASS = "pass"
    SKIP = "skip"

    @property
    def rank(self) -> int:
        return {"fail": 4, "warn": 3, "info": 2, "pass": 1, "skip": 0}[self.value]


class Layer(str, Enum):
    FILE = "file"
    CONFIG = "config"
    REQUEST = "request"
    RESPONSE = "response"
    OUTPUT = "output"
    PROBE = "probe"
    KNOWN_ISSUE = "known_issue"
    DEEP = "deep"


class Cause(str, Enum):
    """Verdict categories from the product report, plus 'symptom' for output signals."""

    FILE = "file"
    CONFIG = "config"
    REQUEST = "request"
    SAMPLING = "sampling"
    RUNTIME = "runtime"
    QUANTISATION = "quantisation"
    PROVIDER = "provider"          # content filter, refusal field, rate limit
    KNOWN_BUG = "known_bug"
    WEIGHT_LEVEL = "weight_level"
    SYMPTOM = "symptom"            # an output signal, not a cause by itself
    NONE = "none"


# Order in which causes win when several are present. Earlier layers are
# cheaper to fix and, when broken, make every later observation unreliable.
CAUSE_PRIORITY = [
    Cause.FILE,
    Cause.CONFIG,
    Cause.REQUEST,
    Cause.SAMPLING,
    Cause.RUNTIME,
    Cause.QUANTISATION,
    Cause.PROVIDER,
    Cause.KNOWN_BUG,
    Cause.WEIGHT_LEVEL,
]


@dataclass
class Finding:
    code: str
    layer: Layer
    severity: Severity
    message: str
    cause: Cause = Cause.NONE
    score: Optional[float] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    fix: Optional[str] = None
    source: Optional[str] = None  # technique provenance, e.g. "llm-output-guard"

    @property
    def failed(self) -> bool:
        return self.severity in (Severity.FAIL, Severity.WARN)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layer"] = self.layer.value
        d["severity"] = self.severity.value
        d["cause"] = self.cause.value
        return {k: v for k, v in d.items() if v not in (None, {}, [])}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    verdict: Optional[dict[str, Any]] = None
    context: dict[str, Any] = field(default_factory=dict)

    def extend(self, items: list[Finding]) -> None:
        self.findings.extend(items)

    @property
    def worst(self) -> Severity:
        if not self.findings:
            return Severity.PASS
        return max((f.severity for f in self.findings), key=lambda s: s.rank)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "context": self.context,
            "findings": [f.as_dict() for f in self.findings],
        }
