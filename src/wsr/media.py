"""Images in requests: where they sit, whether the server will read them, whether the model saw them.

Failures seen against a live llama-server with a vision projector loaded:

* an Ollama-style ``images`` field (top level or per message) sent to an
  OpenAI-compatible endpoint is dropped without an error; the model answers
  "please provide the image";
* ``image_url`` given as a bare string instead of ``{"url": ...}`` -> HTTP 500
  "Invalid base64 value";
* a remote http(s) URL the server cannot fetch -> HTTP 500 "Failed to download image";
* a data URI whose bytes are not an image -> HTTP 400 "Failed to load image";
* a server started without its projector (``--mmproj``) cannot read images at all.

Base64 image data must never be counted as prompt text: a 30 KB JPEG is ~10k
"tokens" of text but ~260 image tokens.
"""
from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Optional

from .findings import Cause, Finding, Layer, Severity

_MAGIC = ((b"\x89PNG", "png"), (b"\xff\xd8\xff", "jpeg"), (b"GIF8", "gif"), (b"BM", "bmp"),
          (b"RIFF", "webp"), (b"II*\x00", "tiff"), (b"MM\x00*", "tiff"))
_DATA_URI = re.compile(r"^data:([\w/+.-]*)(;base64)?,(.*)$", re.S)


def content_text(content: Any) -> str:
    """The text of a message's content: a string, or the text parts of a parts list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                out.append(p["text"])
        return "\n".join(out)
    return "" if content is None else str(content)


def image_refs(messages: Optional[list], params: Optional[dict] = None) -> list[dict]:
    """Every image the request carries, readable or not.

    kind: data (inline data URI), url (remote), field (Ollama-style 'images'
    entry), bad_part (image_url not an object), anthropic (base64 source block).
    """
    refs: list[dict] = []
    for i, m in enumerate(messages or []):
        if not isinstance(m, dict):
            continue
        for img in m.get("images") or []:
            refs.append({"msg": i, "kind": "field", "value": img})
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for p in c:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("image_url", "input_image"):
                v = p.get("image_url")
                if isinstance(v, dict):
                    url = v.get("url") or ""
                    refs.append({"msg": i, "kind": "data" if str(url).startswith("data:") else "url", "value": url})
                else:
                    refs.append({"msg": i, "kind": "bad_part", "value": v})
            elif t == "image" and isinstance(p.get("source"), dict):
                refs.append({"msg": i, "kind": "anthropic", "value": p["source"].get("data") or p["source"].get("url")})
    for img in (params or {}).get("images") or []:
        refs.append({"msg": None, "kind": "field", "value": img})
    return refs


def _sniff(raw: bytes) -> Optional[str]:
    for magic, fmt in _MAGIC:
        if raw.startswith(magic):
            return fmt
    return None


def decode_data_uri(uri: str) -> tuple[Optional[bytes], Optional[str]]:
    """(bytes, problem). problem is None when the URI holds a recognisable image."""
    m = _DATA_URI.match(uri.strip())
    if not m:
        return None, "not a data URI"
    if not m.group(2):
        return None, "data URI is not base64-encoded"
    try:
        raw = base64.b64decode(m.group(3), validate=True)
    except (binascii.Error, ValueError):
        return None, "invalid base64"
    fmt = _sniff(raw)
    if not fmt:
        return raw, f"the bytes are not a known image format (starts {raw[:8]!r})"
    if _cut(raw, fmt):
        return raw, f"the {fmt.upper()} is cut short ({len(raw)} bytes, no end marker)"
    return raw, None


def _cut(raw: bytes, fmt: str) -> bool:
    if fmt == "png":
        return b"IEND" not in raw[-32:]
    if fmt == "jpeg":
        return b"\xff\xd9" not in raw.rstrip(b"\0")[-16:]
    return False


def _f(code: str, sev: Severity, msg: str, fix: Optional[str], cause: Cause = Cause.REQUEST, **ev: Any) -> Finding:
    return Finding(code, Layer.REQUEST, sev, msg, cause=cause, fix=fix, source="wsr media",
                   evidence={k: v for k, v in ev.items() if v is not None})


def check_media(messages: Optional[list], params: Optional[dict] = None, *, api: Optional[str] = None,
                vision: Optional[bool] = None) -> list[Finding]:
    """Static checks on the images a request carries.

    api: 'ollama' for Ollama's native /api/chat, 'openai' for /v1/chat/completions,
    None when unknown. vision: the server's own report (llama-server /props
    modalities), None when unknown.
    """
    refs = image_refs(messages, params)
    if not refs:
        return []
    out: list[Finding] = []
    fields = [r for r in refs if r["kind"] == "field"]
    if fields and api != "ollama":
        where = "top-level 'images'" if any(r["msg"] is None for r in fields) else "messages[].images"
        sev = Severity.FAIL if api == "openai" else Severity.WARN
        out.append(_f("IMAGE_FIELD_IGNORED", sev,
                      f"{len(fields)} image(s) in {where}: only Ollama's native API reads that field; an "
                      "OpenAI-compatible endpoint drops it without an error and the model never sees the image",
                      "send each image as a content part: {'type': 'image_url', 'image_url': {'url': "
                      "'data:image/png;base64,...'}}", count=len(fields), where=where))
    bad = [r for r in refs if r["kind"] == "bad_part"]
    if bad:
        out.append(_f("IMAGE_URL_NOT_OBJECT", Severity.FAIL,
                      f"{len(bad)} image_url part(s) hold a bare {type(bad[0]['value']).__name__} instead of "
                      "{'url': ...}; llama-server answers HTTP 500 'Invalid base64 value'",
                      "wrap it: {'type': 'image_url', 'image_url': {'url': '<data URI or URL>'}}"))
    for r in refs:
        if r["kind"] == "data":
            _, problem = decode_data_uri(str(r["value"]))
            if problem:
                out.append(_f("IMAGE_BAD_DATA", Severity.FAIL, f"image in message {r['msg']}: {problem}",
                              "re-encode the file: data:image/png;base64,<base64 of the PNG/JPEG bytes>"))
        elif r["kind"] == "field" and isinstance(r["value"], str) and r["value"][:5] != "data:":
            try:
                raw = base64.b64decode(r["value"], validate=True)
            except (binascii.Error, ValueError):
                raw = None
            if raw is None or not _sniff(raw):
                out.append(_f("IMAGE_BAD_DATA", Severity.FAIL, "an 'images' entry is not base64 image bytes",
                              "pass raw base64 of the image file (no data: prefix for Ollama)"))
    urls = [r for r in refs if r["kind"] == "url"]
    if urls:
        out.append(_f("IMAGE_REMOTE_URL", Severity.INFO,
                      f"{len(urls)} image(s) by remote URL: the server has to download them itself; an "
                      "offline or firewalled server fails ('Failed to download image')",
                      "inline the image as a base64 data URI"))
    if vision is False:
        out.append(_f("IMAGE_NO_VISION", Severity.FAIL,
                      "the server reports no vision support, so it cannot read the request's images",
                      "load the model's vision projector (llama-server --mmproj <mmproj.gguf>; Ollama: a "
                      "multimodal model tag)", cause=Cause.CONFIG))
    return out


def has_images(messages: Optional[list], params: Optional[dict] = None) -> bool:
    return bool(image_refs(messages, params))


def images_as_parts(messages: list, params: dict) -> Optional[tuple[list, dict]]:
    """Move Ollama-style 'images' into OpenAI content parts; None when there is nothing to move."""
    top = list((params or {}).get("images") or [])
    if not top and not any(isinstance(m, dict) and m.get("images") for m in messages):
        return None

    def part(img: Any) -> dict:
        s = str(img)
        if not s.startswith("data:"):
            fmt = None
            try:
                fmt = _sniff(base64.b64decode(s[:64] + "=" * (-len(s[:64]) % 4)))
            except (binascii.Error, ValueError):
                pass
            s = f"data:image/{fmt or 'png'};base64,{s}"
        return {"type": "image_url", "image_url": {"url": s}}

    new = []
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    for i, m in enumerate(messages):
        m = dict(m)
        imgs = list(m.pop("images", None) or [])
        if i == last_user:
            imgs += top
        if imgs:
            text = content_text(m.get("content"))
            m["content"] = [part(x) for x in imgs] + ([{"type": "text", "text": text}] if text else [])
        new.append(m)
    return new, {k: v for k, v in params.items() if k != "images"}


_NOT_SEEN = re.compile(
    r"(please (provide|upload|share|attach|send) (the |an |a )?(image|picture|photo|screenshot)"
    r"|\bi (can't|cannot|can not|am unable to|'m unable to|don't|do not) (see|view|access|process|open) "
    r"(the |any |an |a |your )?(image|picture|photo|screenshot|attachment)"
    r"|\bno (image|picture|photo|screenshot) (was |has been |is )?(attached|provided|uploaded|included|shared)"
    r"|(didn't|did not|haven't|have not) (attach|include|upload|provide|share)(ed)? (an |the |a |any )?"
    r"(image|picture|photo|screenshot)"
    r"|\bas a text-based (ai|model|assistant|language model)"
    r"|\bi'?m a text-based)", re.I)


def image_not_seen_score(text: str) -> float:
    """The model says it got no image although the request carried one."""
    from .output import plain_apostrophes
    text = plain_apostrophes(text)
    head = text[:400]
    return 1.0 if _NOT_SEEN.search(head) else (0.6 if _NOT_SEEN.search(text) else 0.0)


def solid_png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    """A tiny solid-colour PNG, built without an imaging library."""
    import struct
    import zlib

    row = b"\0" + bytes(rgb) * size
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b""))


def image_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
