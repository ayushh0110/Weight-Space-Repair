"""Layer 6: match symptoms + model + runtime against documented bugs.

The database is a JSON file shipped with the package (``known_issues.json``);
``--known-issues PATH`` adds a user file. Each entry names a model regex,
optional backends, optional conditions and the symptom codes it explains.
"""
from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional

from .findings import Cause, Finding, Layer, Severity


def load_db(extra: Optional[Path] = None) -> list[dict]:
    db = json.loads(resources.files("wsr").joinpath("known_issues.json").read_text(encoding="utf-8"))
    if extra:
        db += json.loads(Path(extra).read_text(encoding="utf-8"))
    return db


def match(model: str, symptoms: Iterable[str], *, backend: Optional[str] = None,
          conditions: Iterable[str] = (), db: Optional[list[dict]] = None) -> list[Finding]:
    db = db if db is not None else load_db()
    sym, cond = set(symptoms), set(conditions)
    out = []
    for e in db:
        if not re.search(e.get("model", ".*"), model or "", re.I):
            continue
        if e.get("backend") and backend not in e["backend"]:
            continue
        if not set(e.get("requires", [])) <= cond or set(e.get("excludes", [])) & cond:
            continue
        hit = sym & set(e.get("symptoms", []))
        if not hit:
            continue
        out.append(Finding(
            code=f"KNOWN_ISSUE:{e['id']}", layer=Layer.KNOWN_ISSUE, severity=Severity.WARN,
            message=e["title"], cause=Cause(e.get("cause", "known_bug")), fix=e.get("fix"), source=e.get("source"),
            evidence={"matched_symptoms": sorted(hit), "level": e.get("level")}))
    return out
