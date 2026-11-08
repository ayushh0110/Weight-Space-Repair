"""Minimal, bounded GGUF reader: header, metadata and tensor table.

Reads metadata without loading weights and skips large arrays (the tokenizer
vocabulary) by length, like LLMDoctor's parser. Every file-derived count and
length is bounds-checked before use, in the spirit of ModelVet; a malformed
file raises :class:`GGUFError` with a byte offset instead of reading garbage.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Optional

MAGIC = b"GGUF"
MAX_KV = 1 << 20
MAX_TENSORS = 1 << 20
MAX_STR = 1 << 26          # 64 MiB; chat templates are KB, not GB
MAX_ARRAY = 1 << 26
KEEP_ARRAY = 64            # arrays longer than this are summarised, not stored

# (block_size, type_size) per ggml type id, pinned to llama.cpp ggml.h (2026).
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}

# general.file_type → quant label (llama.cpp LLAMA_FTYPE_*)
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M",
    32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class GGUFError(ValueError):
    def __init__(self, msg: str, offset: int):
        super().__init__(f"{msg} (at byte {offset})")
        self.offset = offset


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, (f"type{self.ggml_type}", 0, 0))[0]

    def nbytes(self) -> Optional[int]:
        t = GGML_TYPES.get(self.ggml_type)
        if t is None:
            return None
        _, block, size = t
        n = 1
        for d in self.shape:
            n *= d
        if self.shape and self.shape[0] % block:
            return None
        return n // block * size


@dataclass
class GGUFFile:
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    alignment: int = 32
    data_offset: int = 0
    file_size: int = 0
    # id -> text for tokens that look like control tokens (<...>, [...]), plus
    # any ids asked for; the full vocabulary is never held in memory.
    special_tokens: dict[int, str] = field(default_factory=dict)
    vocab_size: Optional[int] = None

    @property
    def arch(self) -> Optional[str]:
        return self.metadata.get("general.architecture")

    def arch_key(self, suffix: str) -> Any:
        return self.metadata.get(f"{self.arch}.{suffix}") if self.arch else None

    @property
    def quant(self) -> Optional[str]:
        ft = self.metadata.get("general.file_type")
        return FILE_TYPES.get(ft) if isinstance(ft, int) else None


class _Reader:
    def __init__(self, f: BinaryIO, size: int):
        self.f, self.size = f, size

    @property
    def pos(self) -> int:
        return self.f.tell()

    def need(self, n: int) -> None:
        if n < 0 or self.pos + n > self.size:
            raise GGUFError(f"read of {n} bytes runs past end of file ({self.size} bytes)", self.pos)

    def read(self, n: int) -> bytes:
        self.need(n)
        return self.f.read(n)

    def scalar(self, t: int) -> Any:
        fmt = _SCALAR[t]
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return self.scalar(4)

    def u64(self) -> int:
        return self.scalar(10)

    def string(self) -> str:
        at = self.pos
        n = self.u64()
        if n > MAX_STR:
            raise GGUFError(f"string length {n} exceeds cap", at)
        return self.read(n).decode("utf-8", errors="replace")

    def skip(self, n: int) -> None:
        self.need(n)
        self.f.seek(n, 1)

    def value(self, t: int) -> Any:
        if t in _SCALAR:
            return self.scalar(t)
        if t == 8:
            return self.string()
        if t == 9:
            at = self.pos
            et, count = self.u32(), self.u64()
            if count > MAX_ARRAY:
                raise GGUFError(f"array length {count} exceeds cap", at)
            if et in _SCALAR:
                width = struct.calcsize(_SCALAR[et])
                if count <= KEEP_ARRAY:
                    return [self.scalar(et) for _ in range(count)]
                self.skip(width * count)
                return {"_array": True, "type": et, "len": count}
            if et == 8:
                if count <= KEEP_ARRAY:
                    return [self.string() for _ in range(count)]
                for _ in range(count):
                    ln = self.u64()
                    if ln > MAX_STR:
                        raise GGUFError(f"string length {ln} exceeds cap", self.pos - 8)
                    self.skip(ln)
                return {"_array": True, "type": et, "len": count}
            if et == 9:
                return [self.value(9) for _ in range(min(count, KEEP_ARRAY))]
            raise GGUFError(f"unknown array element type {et}", at)
        raise GGUFError(f"unknown value type {t}", self.pos - 4)


_SPECIAL_LOOKING = re.compile(r"^(?:<[^<>\s]{1,40}>|\[[A-Z_/]{2,20}\]|<\|[^\s]{0,40})$")


def _read_vocab(r: _Reader, g: GGUFFile) -> dict:
    at = r.pos
    et, count = r.u32(), r.u64()
    if et != 8:
        raise GGUFError("tokenizer.ggml.tokens is not a string array", at)
    if count > MAX_ARRAY:
        raise GGUFError(f"vocabulary size {count} exceeds cap", at)
    for i in range(count):
        ln = r.u64()
        if ln > MAX_STR:
            raise GGUFError(f"token {i} length {ln} exceeds cap", r.pos - 8)
        if ln <= 48:
            s = r.read(ln).decode("utf-8", errors="replace")
            if _SPECIAL_LOOKING.match(s):
                g.special_tokens[i] = s
        else:
            r.skip(ln)
    g.vocab_size = count
    return {"_array": True, "type": 8, "len": count}


def read_gguf(path: str) -> GGUFFile:
    import os

    size = os.path.getsize(path)
    with open(path, "rb") as f:
        r = _Reader(f, size)
        if r.read(4) != MAGIC:
            raise GGUFError("bad magic: not a GGUF file", 0)
        version = r.u32()
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}", 4)
        n_tensors, n_kv = r.u64(), r.u64()
        if n_tensors > MAX_TENSORS:
            raise GGUFError(f"tensor count {n_tensors} exceeds cap", 8)
        if n_kv > MAX_KV:
            raise GGUFError(f"metadata count {n_kv} exceeds cap", 16)
        g = GGUFFile(version=version, file_size=size)
        for _ in range(n_kv):
            at = r.pos
            key = r.string()
            if key in g.metadata:
                raise GGUFError(f"duplicate metadata key {key!r}", at)
            vt = r.u32()
            if key == "tokenizer.ggml.tokens" and vt == 9:
                g.metadata[key] = _read_vocab(r, g)
            else:
                g.metadata[key] = r.value(vt)
        align = g.metadata.get("general.alignment", 32)
        if not isinstance(align, int) or align <= 0 or align & (align - 1):
            raise GGUFError(f"general.alignment {align!r} is not a power of two", 0)
        g.alignment = align
        names: set[str] = set()
        for _ in range(n_tensors):
            at = r.pos
            name = r.string()
            if name in names:
                raise GGUFError(f"duplicate tensor name {name!r}", at)
            names.add(name)
            nd = r.u32()
            if nd == 0 or nd > 8:
                raise GGUFError(f"tensor {name!r} has {nd} dimensions", at)
            shape = [r.u64() for _ in range(nd)]
            g.tensors.append(TensorInfo(name, shape, r.u32(), r.u64()))
        pos = r.pos
        g.data_offset = pos + (-pos % align)
    return g
