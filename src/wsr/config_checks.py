"""Layer 2: environment and model configuration.

Reads a Hugging Face model directory or a GGUF file and checks the things that
make a healthy model look broken:

* chat template missing, failing to render, ignoring ``add_generation_prompt``
  (LLMDoctor ``template-missing``; troubleshooting guides);
* the token the template ends an assistant turn with is not one generation
  stops on (Llama 3 ``<|eot_id|>`` vs ``<|end_of_text|>``, ChatML
  ``<|im_end|>``): the model talks past its turn, loops or invents turns;
* double BOS (template writes it and the tokenizer adds it again);
* tokenizer ids beyond the embedding matrix (added tokens without a resize,
  typical after LoRA merges);
* local template differs from the upstream / base repo (LLMDoctor
  ``template-stale`` / ``template-vs-base``);
* weights + KV cache do not fit memory (LLMDoctor ``no-fit`` / ``tight-fit``).
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .findings import Cause, Finding, Layer, Severity
from .gguf import GGUFError, read_gguf
from .output import CONTROL_TOKENS

PROBE = "WSRPROBE7731"

# Token texts llama.cpp treats as end-of-generation even when the GGUF's
# eos id points elsewhere.
LLAMACPP_AUTO_EOG = {"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|eom_id|>",
                     "<EOT>", "_<EOT>", "<｜end▁of▁sentence｜>", "<|return|>", "<|call|>", "<turn|>"}


@dataclass
class ModelSpec:
    kind: str                       # "hf" | "gguf"
    path: Path
    arch: Optional[str] = None
    templates: dict[str, str] = field(default_factory=dict)  # where → template text
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    add_bos: Optional[bool] = None
    stop_ids: list[int] = field(default_factory=list)
    stop_source: str = ""
    special: dict[int, str] = field(default_factory=dict)  # id → token text
    tokenizer_max_id: Optional[int] = None
    embed_rows: Optional[int] = None
    config_vocab: Optional[int] = None
    ctx_trained: Optional[int] = None
    n_layers: Optional[int] = None
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    sliding_window: Optional[int] = None
    # Per-layer KV geometry: (kv_heads, k_dim, v_dim, window or None); empty → uniform estimate.
    kv_layers: list[tuple[int, int, int, Optional[int]]] = field(default_factory=list)
    weights_bytes: int = 0
    quant: Optional[str] = None
    extra_tokens: dict[str, str] = field(default_factory=dict)  # tokenizer_config special strings
    meta: dict = field(default_factory=dict)        # GGUF metadata (gguf) / config.json (hf)

    @property
    def template(self) -> Optional[str]:
        for key in ("chat_template.jinja", "tokenizer_config.json", "chat_template.json", "gguf"):
            if key in self.templates:
                return self.templates[key]
        return next(iter(self.templates.values()), None)

    def id_of(self, text: str) -> Optional[int]:
        for i, t in self.special.items():
            if t == text:
                return i
        return None


def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tok_text(v: Any) -> Optional[str]:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("content")
    return None


def _ids(v: Any) -> list[int]:
    if isinstance(v, int) and not isinstance(v, bool):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, int)]
    return []


def load_hf(d: Path) -> ModelSpec:
    s = ModelSpec("hf", d)
    cfg = _load_json(d / "config.json") or {}
    s.meta = cfg
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    get = lambda k: text_cfg.get(k, cfg.get(k))  # noqa: E731
    s.arch = (cfg.get("architectures") or [cfg.get("model_type")])[0]
    s.ctx_trained = get("max_position_embeddings")
    s.n_layers = get("num_hidden_layers")
    heads = get("num_attention_heads")
    s.n_kv_heads = get("num_key_value_heads") or heads
    s.head_dim = get("head_dim") or (get("hidden_size") // heads if get("hidden_size") and heads else None)
    s.sliding_window = get("sliding_window") if get("use_sliding_window") is not False else None
    s.config_vocab = get("vocab_size")
    types = get("layer_types")
    if isinstance(types, list) and s.n_kv_heads and s.head_dim:
        shared = get("num_kv_shared_layers") or 0
        g_dim = get("global_head_dim") or s.head_dim
        for t in types[: len(types) - shared]:
            sliding = "sliding" in str(t) and s.sliding_window
            dim = s.head_dim if sliding else g_dim
            s.kv_layers.append((s.n_kv_heads, dim, dim, s.sliding_window if sliding else None))

    gen = _load_json(d / "generation_config.json")
    if gen and "eos_token_id" in gen:
        s.stop_ids, s.stop_source = _ids(gen["eos_token_id"]), "generation_config.json"
    elif "eos_token_id" in cfg or "eos_token_id" in text_cfg:
        s.stop_ids, s.stop_source = _ids(get("eos_token_id")), "config.json"

    tc = _load_json(d / "tokenizer_config.json") or {}
    ct = tc.get("chat_template")
    if isinstance(ct, list):  # [{"name": "default", "template": ...}, ...]
        ct = next((x.get("template") for x in ct if x.get("name") == "default"), ct[0].get("template") if ct else None)
    if isinstance(ct, str):
        s.templates["tokenizer_config.json"] = ct
    if (d / "chat_template.jinja").exists():
        s.templates["chat_template.jinja"] = (d / "chat_template.jinja").read_text(encoding="utf-8")
    cj = _load_json(d / "chat_template.json")
    if cj and isinstance(cj.get("chat_template"), str):
        s.templates["chat_template.json"] = cj["chat_template"]
    s.bos_token, s.eos_token = _tok_text(tc.get("bos_token")), _tok_text(tc.get("eos_token"))
    s.add_bos = tc.get("add_bos_token")
    for k, v in tc.items():
        if k.endswith("_token") and _tok_text(v):
            s.extra_tokens[k] = _tok_text(v)  # type: ignore[assignment]
    for i, info in (tc.get("added_tokens_decoder") or {}).items():
        if isinstance(info, dict) and info.get("content"):
            s.special[int(i)] = info["content"]

    tj = _load_json(d / "tokenizer.json")
    if tj:
        ids = []
        for t in tj.get("added_tokens") or []:
            s.special.setdefault(t["id"], t["content"])
            ids.append(t["id"])
        vocab = (tj.get("model") or {}).get("vocab")
        if isinstance(vocab, dict):
            ids.append(max(vocab.values(), default=-1))
            # bos/eos may live in the base vocab, not added_tokens
            for name in (s.bos_token, s.eos_token):
                if name and name in vocab:
                    s.special.setdefault(vocab[name], name)
        elif isinstance(vocab, list):
            ids.append(len(vocab) - 1)
        s.tokenizer_max_id = max(ids) if ids else None
        if s.add_bos is None:
            pp = json.dumps(tj.get("post_processor") or {})
            if s.bos_token and s.bos_token in pp:
                s.add_bos = True

    # Embedding rows and weight size from safetensors headers (no weights loaded).
    from .file_integrity import read_safetensors_header

    for p in sorted(d.glob("*.safetensors")):
        s.weights_bytes += p.stat().st_size
        if s.embed_rows is None:
            try:
                h, _ = read_safetensors_header(p)
            except (ValueError, UnicodeDecodeError, OSError):
                continue
            for name, info in h.items():
                if name.endswith(("embed_tokens.weight", "wte.weight", "word_embeddings.weight")) and \
                        isinstance(info, dict) and info.get("shape"):
                    s.embed_rows = info["shape"][0]
                    break
    return s


def _gguf_kv_layers(g, s: ModelSpec) -> list[tuple[int, int, int, Optional[int]]]:
    """Per-layer KV shape from GGUF metadata: per-layer kv heads, SWA pattern and dims, shared-KV layers."""
    n = s.n_layers
    if not n:
        return []
    kv = g.arch_key("attention.head_count_kv")
    heads = g.arch_key("attention.head_count")
    kv_l = kv if isinstance(kv, list) else [kv or (heads if isinstance(heads, int) else 0)] * n
    emb = g.arch_key("embedding_length")
    k = g.arch_key("attention.key_length") or (emb // heads if emb and isinstance(heads, int) and heads else None)
    v = g.arch_key("attention.value_length") or k
    if not k:
        return []
    k_swa = g.arch_key("attention.key_length_swa") or k
    v_swa = g.arch_key("attention.value_length_swa") or v
    win = s.sliding_window
    pattern = g.arch_key("attention.sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 0:  # every Nth layer is global
        pattern = [(i + 1) % pattern != 0 for i in range(n)]
    if not isinstance(pattern, list) or len(pattern) != n:
        pattern = [False] * n
    shared = g.arch_key("attention.shared_kv_layers") or 0
    out = []
    for i in range(n - shared):
        if not kv_l[i]:
            continue  # recurrent / attention-free layer
        swa = bool(pattern[i] and win)
        out.append((kv_l[i], k_swa if swa else k, v_swa if swa else v, win if swa else None))
    return out


def load_gguf(p: Path) -> ModelSpec:
    g = read_gguf(str(p))
    s = ModelSpec("gguf", p, arch=g.arch, weights_bytes=p.stat().st_size, quant=g.quant)
    md = g.metadata
    s.meta = md
    s.ctx_trained = g.arch_key("context_length")
    s.n_layers = g.arch_key("block_count")
    kv = g.arch_key("attention.head_count_kv")
    heads = g.arch_key("attention.head_count")
    if isinstance(kv, list):  # per-layer
        kv = max(kv) if kv else None
    s.n_kv_heads = kv or heads
    emb = g.arch_key("embedding_length")
    s.head_dim = g.arch_key("attention.key_length") or (emb // heads if emb and isinstance(heads, int) and heads else None)
    s.sliding_window = g.arch_key("attention.sliding_window")
    s.kv_layers = _gguf_kv_layers(g, s)
    if isinstance(md.get("tokenizer.chat_template"), str):
        s.templates["gguf"] = md["tokenizer.chat_template"]
    s.special = dict(g.special_tokens)
    bos, eos = md.get("tokenizer.ggml.bos_token_id"), md.get("tokenizer.ggml.eos_token_id")
    s.bos_token = s.special.get(bos) if isinstance(bos, int) else None
    s.eos_token = s.special.get(eos) if isinstance(eos, int) else None
    s.add_bos = md.get("tokenizer.ggml.add_bos_token")
    stops = [x for x in (eos, md.get("tokenizer.ggml.eot_token_id"), md.get("tokenizer.ggml.eom_token_id"))
             if isinstance(x, int)]
    stops += [i for i, t in s.special.items() if t in LLAMACPP_AUTO_EOG]
    s.stop_ids, s.stop_source = sorted(set(stops)), "GGUF eos/eot/eom + llama.cpp end-of-generation tokens"
    s.tokenizer_max_id = (g.vocab_size - 1) if g.vocab_size else None
    for t in g.tensors:
        if t.name == "token_embd.weight" and len(t.shape) >= 2:
            s.embed_rows = t.shape[1]  # ggml order: [n_embd, n_vocab]
    return s


def load_spec(target: Path) -> ModelSpec:
    if target.is_file() and target.suffix == ".gguf":
        return load_gguf(target)
    if target.is_dir():
        ggufs = sorted(target.glob("*.gguf"))
        if not (target / "config.json").exists() and ggufs:
            return load_gguf(ggufs[0])
        return load_hf(target)
    raise FileNotFoundError(target)


# --------------------------------------------------------------------------
# Template rendering (same environment transformers uses)
# --------------------------------------------------------------------------

class TemplateError(Exception):
    pass


def render(template: str, messages: list[dict], spec: ModelSpec, add_generation_prompt: bool,
           tools: Optional[list] = None) -> str:
    try:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as e:
        raise TemplateError("jinja2 not installed (pip install wsr-triage[templates])") from e
    from datetime import datetime

    def raise_exception(msg: str) -> None:
        raise TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: \
        json.dumps(x, indent=indent, separators=separators, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    try:
        tpl = env.from_string(template)
        kwargs = dict(spec.extra_tokens)
        kwargs.update(messages=messages, add_generation_prompt=add_generation_prompt, tools=tools,
                      bos_token=spec.bos_token or "", eos_token=spec.eos_token or "")
        return tpl.render(**kwargs)
    except TemplateError:
        raise
    except jinja2.TemplateError as e:
        raise TemplateError(f"{type(e).__name__}: {e}") from e
    except Exception as e:  # noqa: BLE001 - templates call arbitrary methods
        raise TemplateError(f"{type(e).__name__}: {e}") from e


def _known_token_texts(spec: ModelSpec) -> list[str]:
    texts = set(spec.special.values()) | set(CONTROL_TOKENS)
    if spec.eos_token:
        texts.add(spec.eos_token)
    return sorted((t for t in texts if t and len(t) > 1), key=len, reverse=True)


def turn_end_token(rendered: str, spec: ModelSpec) -> Optional[str]:
    """First control token after the probe content of an assistant turn."""
    i = rendered.rfind(PROBE)
    if i < 0:
        return None
    suffix = rendered[i + len(PROBE):]
    best: Optional[tuple[int, str]] = None
    for t in _known_token_texts(spec):
        j = suffix.find(t)
        if j >= 0 and (best is None or j < best[0]):
            best = (j, t)
    return best[1] if best and best[0] <= 8 else None


def _norm_template(t: str) -> str:
    return "\n".join(line.rstrip() for line in t.replace("\r\n", "\n").split("\n")).strip()


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _f(code: str, sev: Severity, msg: str, fix: Optional[str] = None, cause: Cause = Cause.CONFIG,
       source: str = "wsr", **ev: Any) -> Finding:
    return Finding(code=code, layer=Layer.CONFIG, severity=sev, message=msg, cause=cause, fix=fix, source=source,
                   evidence={k: v for k, v in ev.items() if v is not None})


def check_template(spec: ModelSpec) -> list[Finding]:
    out: list[Finding] = []
    tpl = spec.template
    if not tpl:
        return [_f("TEMPLATE_MISSING", Severity.FAIL,
                   "no chat template in the model files; a server will fall back to a generic one or raw text",
                   "use the instruct model's tokenizer_config.json / chat_template.jinja, or pass "
                   "--chat-template-file (llama.cpp) / a TEMPLATE block (Ollama Modelfile)",
                   source="LLMDoctor template-missing")]

    if len({_norm_template(t) for t in spec.templates.values()}) > 1:
        out.append(_f("TEMPLATE_CONFLICT", Severity.WARN,
                      f"different chat templates in {sorted(spec.templates)}; runtimes disagree on which wins",
                      "keep one template; transformers prefers chat_template.jinja"))

    user = [{"role": "user", "content": "Hi"}]
    try:
        with_gen = render(tpl, user, spec, True)
        without_gen = render(tpl, user, spec, False)
    except TemplateError as e:
        if "not installed" in str(e):
            return out + [_f("TEMPLATE_NOT_RENDERED", Severity.SKIP, str(e))]
        return out + [_f("TEMPLATE_RENDER_ERROR", Severity.FAIL, f"template fails on a plain user turn: {e}",
                         "replace the template with the upstream one", source="troubleshooting guides")]

    try:
        render(tpl, [{"role": "system", "content": "Be brief."}] + user, spec, True)
    except TemplateError as e:
        out.append(_f("TEMPLATE_REJECTS_SYSTEM", Severity.INFO,
                      f"template rejects a system message ({e}); clients that send one will get an error or "
                      "a silently dropped system prompt"))

    if with_gen == without_gen:
        out.append(_f("NO_GENERATION_PROMPT", Severity.WARN,
                      "add_generation_prompt changes nothing: the model is not cued to start an assistant turn "
                      "and may continue the user's text",
                      "use a template that emits the assistant header when add_generation_prompt is true"))

    # Which token ends an assistant turn, and does generation stop on it?
    conv = user + [{"role": "assistant", "content": PROBE}]
    try:
        rendered = render(tpl, conv, spec, False)
    except TemplateError as e:
        out.append(_f("TEMPLATE_RENDER_ERROR", Severity.FAIL, f"template fails on a user+assistant exchange: {e}"))
        return out
    end = turn_end_token(rendered, spec)
    stop_texts = {spec.special.get(i, f"<id {i}>") for i in spec.stop_ids}
    if end is None:
        out.append(_f("TURN_END_UNKNOWN", Severity.INFO,
                      "could not identify the token that ends an assistant turn in this template",
                      rendered_tail=rendered[rendered.find(PROBE):][:80] if PROBE in rendered else None))
    elif not spec.stop_ids:
        out.append(_f("NO_STOP_TOKENS", Severity.FAIL,
                      f"template ends turns with {end!r} but no eos_token_id is configured",
                      f"set eos_token_id in generation_config.json to include the id of {end!r}",
                      source="Raschka checklist"))
    else:
        end_id = spec.id_of(end)
        if end_id is None or end_id not in spec.stop_ids:
            fix = (f"add {end!r} (id {end_id}) to eos_token_id in generation_config.json, or pass it as a stop token"
                   if spec.kind == "hf" else f"pass {end!r} as a stop string (Ollama: PARAMETER stop; llama-server: "
                   "--reverse-prompt / request stop)")
            out.append(_f("EOS_MISMATCH", Severity.FAIL,
                          f"template ends assistant turns with {end!r} (id {end_id}) but generation stops only on "
                          f"{sorted(stop_texts)} ({spec.stop_source}); expect run-on answers, invented turns or loops",
                          fix, source="Raschka checklist / HF config threads",
                          turn_end=end, turn_end_id=end_id, stop_ids=spec.stop_ids))
        else:
            out.append(_f("EOS_OK", Severity.PASS, f"turn end {end!r} (id {end_id}) is a stop token"))

    if spec.bos_token and spec.add_bos and with_gen.lstrip().startswith(spec.bos_token):
        if spec.kind == "gguf":  # llama.cpp drops the template's leading BOS before tokenising
            out.append(_f("TEMPLATE_WRITES_BOS", Severity.INFO,
                          f"template writes {spec.bos_token!r} and the tokenizer adds one; llama.cpp strips the "
                          "template copy, other runtimes may not", source="troubleshooting guides"))
        else:
            out.append(_f("DOUBLE_BOS", Severity.WARN,
                          f"template writes {spec.bos_token!r} and the tokenizer adds it again when the rendered "
                          "text is tokenised with special tokens",
                          "use apply_chat_template(tokenize=True), or tokenize with add_special_tokens=False",
                          source="troubleshooting guides"))

    if not any(x.severity in (Severity.FAIL, Severity.WARN) for x in out):
        out.append(_f("TEMPLATE_OK", Severity.PASS, "template renders and has a generation prompt"))
    return out


def check_tokenizer(spec: ModelSpec) -> list[Finding]:
    out: list[Finding] = []
    if spec.tokenizer_max_id is not None and spec.embed_rows is not None:
        if spec.tokenizer_max_id >= spec.embed_rows:
            out.append(_f("TOKENIZER_EXCEEDS_EMBEDDINGS", Severity.FAIL,
                          f"tokenizer has ids up to {spec.tokenizer_max_id} but the embedding matrix has "
                          f"{spec.embed_rows} rows; those tokens index garbage or crash",
                          "resize token embeddings after adding tokens, or use the tokenizer the weights were "
                          "trained with", source="LoRA-merge troubleshooting"))
        else:
            out.append(_f("TOKENIZER_FITS", Severity.PASS,
                          f"tokenizer max id {spec.tokenizer_max_id} < {spec.embed_rows} embedding rows"))
    if spec.config_vocab and spec.embed_rows and spec.config_vocab != spec.embed_rows:
        out.append(_f("VOCAB_SIZE_MISMATCH", Severity.WARN,
                      f"config vocab_size {spec.config_vocab} != embedding rows {spec.embed_rows}"))
    for i in spec.stop_ids:
        if spec.embed_rows is not None and i >= spec.embed_rows:
            out.append(_f("STOP_ID_OUT_OF_RANGE", Severity.FAIL,
                          f"stop token id {i} is outside the vocabulary; the model can never emit it"))
    return out


def check_upstream_template(spec: ModelSpec, repo: str, timeout: float = 20) -> list[Finding]:
    repo, _, rev = repo.partition("@")
    rev = rev or "main"
    upstream: Optional[str] = None
    for fname in ("chat_template.jinja", "tokenizer_config.json"):
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"
        req = urllib.request.Request(url, headers={"User-Agent": "wsr-triage"})
        if os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            continue
        if fname.endswith(".jinja"):
            upstream = body
        else:
            ct = json.loads(body).get("chat_template")
            upstream = ct if isinstance(ct, str) else None
        if upstream:
            break
    if upstream is None:
        return [_f("UPSTREAM_TEMPLATE_UNAVAILABLE", Severity.SKIP, f"no chat template found in {repo}@{rev}",
                   source="LLMDoctor")]
    local = spec.template
    if local is None:
        return [_f("TEMPLATE_MISSING_VS_UPSTREAM", Severity.FAIL, f"{repo} has a chat template; this model has none",
                   f"copy chat_template from {repo}", source="LLMDoctor template-stale")]
    if _norm_template(local) == _norm_template(upstream):
        return [_f("TEMPLATE_MATCHES_UPSTREAM", Severity.PASS, f"chat template matches {repo}@{rev}",
                   source="LLMDoctor")]
    import difflib

    diff = list(difflib.unified_diff(_norm_template(upstream).splitlines(), _norm_template(local).splitlines(),
                                     "upstream", "local", lineterm="", n=0))
    return [_f("TEMPLATE_STALE", Severity.WARN,
               f"chat template differs from {repo}@{rev} ({sum(1 for d in diff if d[:1] in '+-') - 2} changed lines)",
               f"update the template from {repo} (re-download the GGUF / re-pull the Ollama tag)",
               source="LLMDoctor template-stale", diff=diff[:40])]


def _ram_bytes() -> Optional[int]:
    try:
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))  # type: ignore[attr-defined]
            return int(m.ullTotalPhys)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return None


def _vram_bytes() -> Optional[int]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
        return max(vals) * 1024 * 1024 if vals else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def kv_cache_bytes(spec: ModelSpec, tokens: int, bytes_per: int = 2) -> Optional[int]:
    """f16 K+V cache. Sliding-window layers hold window tokens; shared-KV layers hold none."""
    if spec.kv_layers:
        return sum(h * (k + v) * bytes_per * (min(tokens, w) if w else tokens) for h, k, v, w in spec.kv_layers)
    if not (spec.n_layers and spec.n_kv_heads and spec.head_dim):
        return None
    return spec.n_layers * spec.n_kv_heads * spec.head_dim * 2 * bytes_per * tokens


def check_memory(spec: ModelSpec, target_ctx: int = 32768) -> list[Finding]:
    ctx = min(target_ctx, spec.ctx_trained) if spec.ctx_trained else target_ctx
    kv = kv_cache_bytes(spec, ctx)
    if kv is None or not spec.weights_bytes:
        return [_f("MEMORY_FIT_UNKNOWN", Severity.SKIP, "not enough metadata to estimate memory")]
    need = spec.weights_bytes + kv
    ram, vram = _ram_bytes(), _vram_bytes()
    gib = lambda b: f"{b / 2**30:.1f} GiB"  # noqa: E731
    ev = dict(weights=spec.weights_bytes, kv_cache=kv, context=ctx, ram=ram, vram=vram)
    budget = vram or ram
    where = "VRAM" if vram else "RAM"
    if budget is None:
        return [_f("MEMORY_FIT_UNKNOWN", Severity.SKIP, f"need ~{gib(need)}; could not read system memory", **ev)]
    msg = f"weights {gib(spec.weights_bytes)} + f16 KV at {ctx} tokens {gib(kv)} = {gib(need)} vs {where} {gib(budget)}"
    if need > budget:
        sev = Severity.WARN if vram and ram and need <= ram else Severity.FAIL
        return [_f("MEMORY_NO_FIT", sev, msg + "; expect OOM, CPU offload or a server-side context cut",
                   "lower the context size, quantise the KV cache, or use a smaller quant",
                   cause=Cause.RUNTIME, source="LLMDoctor no-fit", **ev)]
    if need > 0.9 * budget:
        return [_f("MEMORY_TIGHT_FIT", Severity.WARN, msg, "leave headroom: lower the context size",
                   cause=Cause.RUNTIME, source="LLMDoctor tight-fit", **ev)]
    return [_f("MEMORY_FITS", Severity.PASS, msg, cause=Cause.RUNTIME, source="LLMDoctor", **ev)]


def check_quant(spec: ModelSpec) -> list[Finding]:
    if spec.kind != "gguf" or not spec.quant:
        return []
    low = spec.quant.startswith(("Q2", "Q3", "IQ1", "IQ2", "IQ3", "TQ"))
    small = spec.weights_bytes < 3 * 2**30
    if low and small:
        return [_f("LOW_BIT_SMALL_MODEL", Severity.WARN,
                   f"{spec.quant} on a small model: sub-Q4 quants of small models are loop- and garbage-prone",
                   "try Q4_K_M or Q5_K_M before blaming the weights", cause=Cause.QUANTISATION,
                   source="troubleshooting guides", quant=spec.quant)]
    return [_f("QUANT_INFO", Severity.INFO, f"quantisation {spec.quant}", cause=Cause.QUANTISATION, quant=spec.quant)]


def check_config(target: Path, *, upstream: Optional[str] = None, target_ctx: int = 32768,
                 memory: bool = True) -> tuple[Optional[ModelSpec], list[Finding]]:
    try:
        spec = load_spec(target)
    except (GGUFError, FileNotFoundError, OSError) as e:
        return None, [_f("CONFIG_UNREADABLE", Severity.FAIL, f"cannot read model config: {e}", cause=Cause.FILE)]
    out = [_f("MODEL_INFO", Severity.INFO,
              f"{spec.kind} {spec.arch or '?'}" + (f" {spec.quant}" if spec.quant else "") +
              (f", trained context {spec.ctx_trained}" if spec.ctx_trained else ""),
              arch=spec.arch, ctx_trained=spec.ctx_trained, quant=spec.quant)]
    out += check_template(spec)
    out += check_tokenizer(spec)
    out += check_quant(spec)
    if spec.kind == "gguf":
        from .runtime_checks import check_gguf_conversion
        out += check_gguf_conversion(spec.meta, spec.arch, spec.quant)
    if upstream:
        out += check_upstream_template(spec, upstream)
    if memory:
        out += check_memory(spec, target_ctx)
    return spec, out
