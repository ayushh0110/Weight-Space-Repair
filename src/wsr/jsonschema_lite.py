"""Minimal JSON Schema validator (the subset tool schemas and response formats use).

Supports: type (incl. lists), properties, required, additionalProperties (bool),
items, enum, const, minLength/maxLength, minimum/maximum, minItems/maxItems,
anyOf/oneOf. Unknown keywords are ignored, never errors. Returns a list of
human-readable violations; empty list means valid.

If the ``jsonschema`` package is installed it is not used on purpose: results
must be identical across installs.
"""
from __future__ import annotations

from typing import Any

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _is_type(value: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    py = _TYPES.get(name)
    return True if py is None else isinstance(value, py)


def validate(value: Any, schema: Any, path: str = "$") -> list[str]:
    if not isinstance(schema, dict):
        return []
    errors: list[str] = []

    t = schema.get("type")
    if t is not None:
        names = t if isinstance(t, list) else [t]
        if not any(_is_type(value, n) for n in names):
            return [f"{path}: expected {'/'.join(names)}, got {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']!r}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")

    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        for key, sub in props.items():
            if key in value:
                errors.extend(validate(value[key], sub, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unexpected property {key!r}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems {schema['maxItems']}")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                errors.extend(validate(item, items, f"{path}[{i}]"))

    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            passing = sum(1 for o in options if not validate(value, o, path))
            if passing == 0 or (key == "oneOf" and passing > 1):
                errors.append(f"{path}: does not match {key} ({passing} options matched)")

    return errors
