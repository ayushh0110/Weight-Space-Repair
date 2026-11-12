"""A thin hook layer over any Hugging Face causal LM.

No TransformerLens / nnsight: plain forward hooks on the decoder layers, so new
architectures (Gemma 4, Qwen 3...) work as soon as transformers loads them.
The adapter finds the pieces by structure: the decoder-layer ModuleList, each
layer's attention and MLP, the final norm and the unembedding.

Positions: interventions act on the residual stream at one position (default:
the last prompt token, where the first answer token is decided).
"""
from __future__ import annotations

import contextlib
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

import torch
from torch import nn

# "linear_attn": hybrid models (Qwen3.5 / Qwen3-Next) mix linear-attention layers (Gated DeltaNet) with
# softmax attention ones. Both write the residual through an output projection; only the softmax
# ones have attention heads in the usual sense.
_ATTN_NAMES = ("self_attn", "linear_attn", "attn", "attention", "self_attention")
_NO_HEADS = ("linear_attn",)
_NO_HEADS_CLASS = ("deltanet", "linearattention", "mamba", "ssm", "rwkv")
_MLP_NAMES = ("mlp", "feed_forward", "ffn")
_OPROJ_NAMES = ("o_proj", "out_proj", "dense", "c_proj")
_DOWN_NAMES = ("down_proj", "fc2", "c_proj", "dense_4h_to_h", "w2")
_NORM_NAMES = ("norm", "final_layernorm", "ln_f", "final_norm")
# End-of-turn markers across chat templates, beyond the tokenizer's eos.
_EOT_TEXT = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<turn|>", "<|end|>", "<|endoftext|>", "</s>",
             "<|end_of_text|>", "<eos>")


def _child(mod: nn.Module, names: Sequence[str]) -> Optional[nn.Module]:
    for n in names:
        c = getattr(mod, n, None)
        if isinstance(c, nn.Module):
            return c
    return None


def _first(x: Any) -> torch.Tensor:
    return x[0] if isinstance(x, (tuple, list)) else x


def _replace_first(out: Any, new: torch.Tensor) -> Any:
    if isinstance(out, tuple):
        return (new, *out[1:])
    if isinstance(out, list):
        return [new, *out[1:]]
    return new


@dataclass
class Run:
    """One forward pass: last-position logits plus captured activations ([n_layers, d] at `pos`)."""
    logits: torch.Tensor                       # [vocab] at pos
    resid: Optional[torch.Tensor] = None       # [n_layers + 1, d]: embeddings, then after each layer
    attn: Optional[torch.Tensor] = None        # [n_layers, d] attention sublayer output
    mlp: Optional[torch.Tensor] = None         # [n_layers, d] MLP sublayer output
    heads: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> [n_heads, head_dim] o_proj input



def _weight_bytes_fp32(path: Path) -> Optional[int]:
    """Bytes the weights take in float32, from the checkpoint files and their stored dtype."""
    files = list(path.glob("*.safetensors")) or list(path.glob("*.bin"))
    if not files:
        return None
    stored = 2
    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        dt = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16")
        stored = 4 if "32" in dt else 2
    except (OSError, ValueError):
        pass
    return int(sum(f.stat().st_size for f in files) * 4 / stored)


def _checkpoint_dtype(path: Path) -> Optional[str]:
    """The dtype the checkpoint was saved in (config.json), e.g. "bfloat16"."""
    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    return cfg.get("dtype") or cfg.get("torch_dtype") or tc.get("dtype") or tc.get("torch_dtype")


def auto_dtype(path: Path, device: str) -> str:
    """float32 wherever it fits: on CPU, and on a GPU with room for it.

    float32 is the reference precision, and with it a batch of rows agrees with
    one row at a time to ~1e-6, far below anything a report rounds to. In float16
    batching moves logits by up to ~0.05, and on GPUs without fast half-precision
    units (GTX 16xx, measured: fp16 matmul 0.34 TFLOPS vs fp32 1.53) float16 is
    also 3.6x slower. Only a model too large for float32 drops to bfloat16
    (Ampere and newer) or float16.
    """
    if not device.startswith("cuda"):
        return "float32"
    need = _weight_bytes_fp32(path)
    try:
        free, _ = torch.cuda.mem_get_info(torch.device(device))
    except (RuntimeError, AssertionError):
        free = 0
    if need is not None and need * 1.25 + (768 << 20) <= free:
        return "float32"
    major, _ = torch.cuda.get_device_capability(torch.device(device))
    return "bfloat16" if major >= 8 else "float16"


class LM:
    def __init__(self, model: nn.Module, tokenizer: Any, name: str = "model"):
        self.model = model.eval()
        self.tok = tokenizer
        self.name = name
        self.layers = self._find_layers()
        self.n_layers = len(self.layers)
        self.final_norm = self._find_final_norm()
        self.unembed = model.get_output_embeddings()
        cfg = getattr(model.config, "text_config", None) or model.config
        self.cfg = cfg
        self.d_model = int(getattr(cfg, "hidden_size", 0) or self.unembed.weight.shape[1])
        self.n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        # Layers whose attention has softmax heads (all of them on an ordinary transformer).
        self.head_layers = [i for i in range(self.n_layers) if self._has_heads(i)] if self.n_heads else []
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self.eot_ids = self._eot_ids()
        self.stats: dict[str, dict[str, float]] = {}
        self._memo: Optional[dict] = None       # clean forward passes, only inside memo()
        self._wver = 0                          # bumped whenever weights change (qbisect swaps)
        self._steering = 0                      # > 0 while steer() hooks are active
        self._stage, self._t0 = "load", time.perf_counter()
        self.checkpoint_dtype: Optional[str] = None   # set by load() from config.json
        self.path: Optional[Path] = None               # the model directory, when loaded from one
        # (prompt lengths, max_new, sampled) -> the chunks that list ran in (see generate_batch)
        self.batch_plans: dict[tuple, list[list[int]]] = {}

    # -- profiling ---------------------------------------------------------------
    def stage(self, name: Optional[str]) -> None:
        """Start timing a named step (None closes the current one). Counts forwards / generations per step."""
        now = time.perf_counter()
        st = self.stats.setdefault(self._stage, {"s": 0.0, "fwd": 0, "rows": 0, "gen": 0, "tok": 0})
        st["s"] += now - self._t0
        self._stage, self._t0 = (name or "other"), now

    def _count(self, key: str, n: int = 1) -> None:
        st = self.stats.setdefault(self._stage, {"s": 0.0, "fwd": 0, "rows": 0, "gen": 0, "tok": 0})
        st[key] = st.get(key, 0) + n

    # -- loading ---------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @classmethod
    def load(cls, path: str | Path, dtype: str = "auto", device: str = "auto") -> "LM":
        """Load a HF model. device "auto" = CUDA when available; dtype "auto" = see `auto_dtype`.

        Token ids and every captured activation stay on the CPU in float32; only the
        model runs on the device, so the analysis code is the same either way.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = auto_dtype(Path(str(path)), device)
        dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dt, low_cpu_mem_usage=True)
        model.to(device)
        lm = cls(model, tok, name=Path(str(path)).name)
        lm.checkpoint_dtype = _checkpoint_dtype(Path(str(path)))
        lm.path = Path(str(path))
        return lm

    # -- structure ------------------------------------------------------------
    def _find_layers(self) -> list[nn.Module]:
        best: Optional[nn.ModuleList] = None
        for _, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) and _child(mod[0], _ATTN_NAMES) is not None \
                    and _child(mod[0], _MLP_NAMES) is not None:
                if best is None or len(mod) > len(best):
                    best = mod
        if best is None:
            raise ValueError("no decoder layers found (a ModuleList of blocks with attention + MLP)")
        return list(best)

    def _find_final_norm(self) -> Optional[nn.Module]:
        layers_parent = None
        for _, mod in self.model.named_modules():
            for c in mod.children():
                if isinstance(c, nn.ModuleList) and len(c) and c[0] is self.layers[0]:
                    layers_parent = mod
        return _child(layers_parent, _NORM_NAMES) if layers_parent is not None else None

    def attn(self, i: int) -> nn.Module:
        return _child(self.layers[i], _ATTN_NAMES)

    def _has_heads(self, i: int) -> bool:
        layer = self.layers[i]
        if any(isinstance(getattr(layer, n, None), nn.Module) for n in _NO_HEADS) and \
                not isinstance(getattr(layer, "self_attn", None), nn.Module):
            return False
        cls = type(self.attn(i)).__name__.lower()
        return not any(k in cls for k in _NO_HEADS_CLASS) and self.o_proj(i) is not None

    def head_dim(self, i: int) -> int:
        """Width of one head's block in layer i's output-projection input."""
        if i not in self.head_layers:
            raise ValueError(f"layer {i} has no attention heads (linear attention / state-space layer)")
        return self.o_proj(i).weight.shape[1] // self.n_heads

    def attentions_by_layer(self, atts: Sequence[Any]) -> list[torch.Tensor]:
        """HF `output_attentions` per decoder layer. Hybrid models return them for the softmax layers
        only; the others get zeros (no head attends anywhere)."""
        got = [a for a in (atts or ()) if a is not None]
        if len(got) == self.n_layers:
            return list(got)
        if got and len(got) == len(self.head_layers):
            out = [torch.zeros_like(got[0]) for _ in range(self.n_layers)]
            for i, a in zip(self.head_layers, got):
                out[i] = a
            return out
        raise ValueError(f"the model returned attention weights for {len(got)} of {self.n_layers} layers")

    def mlp(self, i: int) -> nn.Module:
        return _child(self.layers[i], _MLP_NAMES)

    def o_proj(self, i: int) -> Optional[nn.Linear]:
        return _child(self.attn(i), _OPROJ_NAMES)

    def down_proj(self, i: int) -> Optional[nn.Linear]:
        return _child(self.mlp(i), _DOWN_NAMES)

    def _eot_ids(self) -> set[int]:
        ids: set[int] = set()
        for src in (getattr(self.model, "generation_config", None), self.model.config):
            e = getattr(src, "eos_token_id", None) if src is not None else None
            if isinstance(e, int):
                ids.add(e)
            elif isinstance(e, (list, tuple)):
                ids.update(int(x) for x in e)
        if getattr(self.tok, "eos_token_id", None) is not None:
            ids.add(int(self.tok.eos_token_id))
        vocab = self.tok.get_vocab() if hasattr(self.tok, "get_vocab") else {}
        for t in _EOT_TEXT:
            if t in vocab:
                ids.add(int(vocab[t]))
        return ids

    # -- text -----------------------------------------------------------------
    def encode(self, messages: list[dict] | str, *, chat: bool = True, **template_kwargs: Any) -> torch.Tensor:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if chat and getattr(self.tok, "chat_template", None):
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                **template_kwargs)
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
        else:
            text = "\n".join(str(m.get("content", "")) for m in messages)
            ids = self.tok(text)["input_ids"]
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def token_str(self, i: int) -> str:
        """The token as text ("В", not the byte-level symbol "ÐĴ"); falls back to the vocabulary entry."""
        t = self.tok.decode([int(i)], skip_special_tokens=False)
        if t and "\ufffd" not in t:
            return t
        return self.tok.convert_ids_to_tokens(int(i)) if hasattr(self.tok, "convert_ids_to_tokens") else t

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int = 64, hooks: Optional[Callable] = None,
                 **gen_kw: Any) -> list[int]:
        """Continuation token ids, greedy unless gen_kw says otherwise (do_sample, temperature, top_p,
        repetition_penalty); stops at end-of-turn, which is kept."""
        ctx = hooks() if hooks else contextlib.nullcontext()
        ids = ids.to(self.device)
        with ctx:
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=gen_kw.pop("do_sample", False),
                                      eos_token_id=sorted(self.eot_ids) or None, **gen_kw,
                                      pad_token_id=next(iter(self.eot_ids), 0))
        self._count("gen")
        self._count("tok", out.shape[1] - ids.shape[1])
        return out[0, ids.shape[1]:].tolist()

    def generate_batch(self, ids_list: Sequence[torch.Tensor], max_new_tokens: int = 64,
                       seeds: Optional[Sequence[int]] = None, hooks: Optional[Callable] = None,
                       **gen_kw: Any) -> list[list[int]]:
        """`generate` for many prompts, several at a time when memory allows (deep/batching.py plans the
        batch from free VRAM / RAM; an out-of-memory error halves it and retries). Same return as calling
        `generate` on each, in the same order. Prompts are grouped by length (left padding).

        The same prompt list (same lengths, same settings) always runs in the same chunks within a run
        (`batch_plans`, saved with cached baselines too): a before / after comparison of the same prompts
        sees identical numerics, so only the patch can change a reply. With sampling, each chunk is seeded
        from its first prompt's seed. `hooks` (a context factory such as `lambda: lm.steer(ablate=u)`) is
        active for every chunk; interventions must be the same for every row (position -1 or every
        position), which holds for every steer wsr uses."""
        from . import batching as BT

        n = len(ids_list)
        if n == 0:
            return []
        sampled = bool(gen_kw.get("do_sample"))
        key = (tuple(int(t.shape[1]) for t in ids_list), int(max_new_tokens), sampled)
        chunks = self.batch_plans.get(key)
        if chunks is None:
            order = sorted(range(n), key=lambda k: (ids_list[k].shape[1], k))
            bs = BT.plan(self, max(t.shape[1] for t in ids_list), max_new_tokens, n)
            self._count("batch_plan", bs)
            chunks = [order[i:i + bs] for i in range(0, n, bs)]
        out: list[Optional[list[int]]] = [None] * n
        done: list[list[int]] = []
        queue = list(chunks)
        ctx = hooks or contextlib.nullcontext
        while queue:
            chunk = queue.pop(0)
            if sampled and seeds is not None:
                torch.manual_seed(int(seeds[chunk[0]]))
            try:
                if len(chunk) == 1:
                    res = [self.generate(ids_list[chunk[0]], max_new_tokens, hooks=hooks, **dict(gen_kw))]
                else:
                    with ctx():
                        res = self._generate_padded([ids_list[k] for k in chunk], max_new_tokens, **dict(gen_kw))
            except Exception as e:                     # noqa: BLE001 - only memory errors are retried
                if not BT.is_oom(e) or len(chunk) == 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                h = len(chunk) // 2
                queue[:0] = [chunk[:h], chunk[h:]]          # split this chunk; the plan is updated below
                self._count("batch_oom")
                continue
            for k, g in zip(chunk, res):
                out[k] = g
            done.append(chunk)
        if len(self.batch_plans) > 512:
            self.batch_plans.clear()
        self.batch_plans[key] = done
        return [g if g is not None else [] for g in out]

    def _generate_padded(self, ids_list: Sequence[torch.Tensor], max_new_tokens: int, **gen_kw: Any) -> list[list[int]]:
        pad = self.tok.pad_token_id if getattr(self.tok, "pad_token_id", None) is not None else \
            next(iter(sorted(self.eot_ids)), 0)
        L = max(t.shape[1] for t in ids_list)
        ids = torch.full((len(ids_list), L), int(pad), dtype=torch.long)
        mask = torch.zeros((len(ids_list), L), dtype=torch.long)
        for r, t in enumerate(ids_list):
            ids[r, L - t.shape[1]:] = t[0]
            mask[r, L - t.shape[1]:] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.no_grad():
            o = self.model.generate(ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                    do_sample=gen_kw.pop("do_sample", False),
                                    eos_token_id=sorted(self.eot_ids) or None, pad_token_id=int(pad), **gen_kw)
        self._count("gen", len(ids_list))
        res = []
        for r in range(len(ids_list)):
            new = o[r, L:].tolist()
            cut = next((j for j, t in enumerate(new) if t in self.eot_ids), None)
            new = new[:cut + 1] if cut is not None else new
            self._count("tok", len(new))
            res.append(new)
        return res

    # -- unembedding / logit lens ---------------------------------------------
    def logits_of(self, resid: torch.Tensor) -> torch.Tensor:
        """Read a residual vector (or [n, d] stack) through the final norm and unembedding."""
        x = resid.to(self.device, self.unembed.weight.dtype)
        if self.final_norm is not None:
            x = self.final_norm(x)
        y = self.unembed(x).float().cpu()
        if self.softcap:
            y = torch.tanh(y / self.softcap) * self.softcap
        return y

    # -- forward with capture and interventions ---------------------------------
    @contextlib.contextmanager
    def _hooks(self, regs: list[tuple[nn.Module, str, Callable]]) -> Iterator[None]:
        handles = []
        try:
            for mod, kind, fn in regs:
                if kind == "prekw":
                    handles.append(mod.register_forward_pre_hook(fn, with_kwargs=True))
                elif kind == "pre":
                    handles.append(mod.register_forward_pre_hook(fn))
                else:
                    handles.append(mod.register_forward_hook(fn))
            yield
        finally:
            for h in handles:
                h.remove()

    def intervention_hooks(self, *, pos: int = -1, patch_resid: Optional[dict[int, torch.Tensor]] = None,
                           patch_attn: Optional[dict[int, torch.Tensor]] = None,
                           patch_mlp: Optional[dict[int, torch.Tensor]] = None,
                           patch_heads: Optional[dict[tuple[int, int], torch.Tensor]] = None,
                           ablate: Optional[torch.Tensor] = None, ablate_all_positions: bool = True,
                           add: Optional[tuple[torch.Tensor, float, Sequence[int]]] = None,
                           clamp: Optional[tuple[torch.Tensor, float, Sequence[int]]] = None,
                           zero_heads: Optional[Sequence[tuple[int, int]]] = None,
                           zero_neurons: Optional[Sequence[tuple[int, int]]] = None,
                           project_sites: Optional[tuple[Sequence[Any], float]] = None
                           ) -> list[tuple[nn.Module, str, Callable]]:
        """Hooks that patch activations at `pos`, ablate a direction everywhere, add a steering vector,
        remove attention heads (zero their output at every position, generation steps included), or
        take alpha of a direction out of chosen output projections' writes (`project_sites`:
        (sites, alpha); each site has layer, part "mlp" / "attn", unit u and optional input columns -
        the hook form of `surgery.edits.project`)."""
        regs: list[tuple[nn.Module, str, Callable]] = []
        if project_sites is not None:
            sites, alpha = project_sites
            for s in sites:
                mod = self.down_proj(s.layer) if s.part == "mlp" else self.o_proj(s.layer)
                dt = mod.weight.dtype
                u = (s.u / s.u.norm()).to(self.device, dt)
                wu = None
                if s.cols is not None:
                    w = (s.u / s.u.norm()).float().to(self.device) @ mod.weight.detach().float()
                    m = torch.zeros_like(w)
                    m[list(s.cols)] = 1.0
                    wu = (w * m).to(dt)

                def proj(m_, args, o, u=u, wu=wu):
                    c = (o @ u) if wu is None else (args[0] @ wu)
                    return o - alpha * c.unsqueeze(-1) * u
                regs.append((mod, "post", proj))
        zl: dict[int, list[int]] = {}
        for i, h in zero_heads or ():
            zl.setdefault(int(i), []).append(int(h))
        for i in {*zl, *(i for i, _ in (patch_heads or {}))}:
            if i not in self.head_layers:
                raise ValueError(f"layer {i} has no attention heads (linear attention / state-space layer)")
        for i, hs in zl.items():
            def zpre(m, args, hs=hs):
                x = args[0].clone()
                hd = x.shape[-1] // self.n_heads
                for h in hs:
                    x[..., h * hd:(h + 1) * hd] = 0
                return (x, *args[1:])
            regs.append((self.o_proj(i), "pre", zpre))
        zn: dict[int, list[int]] = {}
        for i, j in zero_neurons or ():
            zn.setdefault(int(i), []).append(int(j))
        for i, js in zn.items():
            def npre(m, args, js=js):
                x = args[0].clone()
                x[..., js] = 0
                return (x, *args[1:])
            regs.append((self.down_proj(i), "pre", npre))

        def at_pos(t: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
            if t.shape[1] == 1:  # incremental decoding step: the prompt was already patched
                return t
            t = t.clone()
            t[:, pos] = vec.to(t.device, t.dtype)
            return t

        for i, v in (patch_resid or {}).items():
            regs.append((self.layers[i], "post", lambda m, a, o, v=v: _replace_first(o, at_pos(_first(o), v))))
        for i, v in (patch_attn or {}).items():
            regs.append((self.attn(i), "post", lambda m, a, o, v=v: _replace_first(o, at_pos(_first(o), v))))
        for i, v in (patch_mlp or {}).items():
            regs.append((self.mlp(i), "post", lambda m, a, o, v=v: _replace_first(o, at_pos(_first(o), v))))
        by_layer: dict[int, dict[int, torch.Tensor]] = {}
        for (i, h), v in (patch_heads or {}).items():
            by_layer.setdefault(i, {})[h] = v
        for i, hv in by_layer.items():
            def pre(m, args, hv=hv):
                x = args[0].clone()
                hd = x.shape[-1] // self.n_heads
                for h, v in hv.items():
                    if x.shape[1] > 1:
                        x[:, pos, h * hd:(h + 1) * hd] = v.to(x.device, x.dtype)
                return (x, *args[1:])
            regs.append((self.o_proj(i), "pre", pre))
        if ablate is not None:
            d = (ablate / ablate.norm()).to(self.device, self.unembed.weight.dtype)

            def proj_out(t: torch.Tensor) -> torch.Tensor:
                if ablate_all_positions:
                    return t - (t @ d).unsqueeze(-1) * d
                t = t.clone()
                t[:, pos] = t[:, pos] - (t[:, pos] @ d).unsqueeze(-1) * d
                return t
            # Directional ablation (Arditi et al.): remove the direction from every write to the residual,
            # the token embeddings included (layer 0 must not read it either).
            regs.append((self.model.get_input_embeddings(), "post", lambda m, a, o: proj_out(o)))
            for i in range(self.n_layers):
                regs.append((self.attn(i), "post", lambda m, a, o: _replace_first(o, proj_out(_first(o)))))
                regs.append((self.mlp(i), "post", lambda m, a, o: _replace_first(o, proj_out(_first(o)))))
                regs.append((self.layers[i], "post", lambda m, a, o: _replace_first(o, proj_out(_first(o)))))
        if add is not None:
            vec, alpha, at_layers = add
            for i in at_layers:
                regs.append((self.layers[i], "post",
                             lambda m, a, o: _replace_first(o, _first(o) + alpha * vec.to(_first(o).device, _first(o).dtype))))
        if clamp is not None:
            # Conditional steering: after each listed layer, wherever the residual's component along the unit u
            # exceeds `level`, lower it to `level`; positions at or below it are untouched. (u, level, layers).
            cu, level, at_layers = clamp
            cu = cu / cu.norm()

            def clamp_hook(m, a, o, u=cu, lv=float(level)):
                h = _first(o)
                uu = u.to(h.device, h.dtype)
                excess = ((h * uu).sum(-1, keepdim=True) - lv).clamp(min=0)
                return _replace_first(o, h - excess * uu)
            for i in at_layers:
                regs.append((self.layers[i], "post", clamp_hook))
        return regs

    @property
    def _keeps_logits(self) -> bool:
        """The model can compute logits for the last position only (HF `logits_to_keep`)."""
        if not hasattr(self, "_keeps"):
            try:
                self._keeps = "logits_to_keep" in inspect.signature(self.model.forward).parameters
            except (TypeError, ValueError):
                self._keeps = False
        return self._keeps

    @torch.no_grad()
    def run(self, ids: torch.Tensor, *, pos: int = -1, capture: Sequence[str] = ("resid",),
            heads_at: Sequence[int] = (), pos_offset: int = 0, **interventions: Any) -> Run:
        """Forward the prompt once; capture activations at `pos` and apply interventions."""
        key = None
        if self._memo is not None and not self._steering and not pos_offset and \
                not any(v is not None and v is not False and v != {} and v != [] for k, v in interventions.items()
                        if k != "ablate_all_positions"):
            key = (ids.numpy().tobytes() if ids.device.type == "cpu" else ids.cpu().numpy().tobytes(), tuple(ids.shape),
                   pos, tuple(capture), tuple(heads_at), self._wver)
            hit = self._memo.get(key)
            if hit is not None:
                self._count("memo_hit")
                return hit
        cap: dict[str, list] = {"resid": [None] * (self.n_layers + 1), "attn": [None] * self.n_layers,
                                "mlp": [None] * self.n_layers}
        heads: dict[int, torch.Tensor] = {}
        regs = self.intervention_hooks(pos=pos, **interventions)
        if "resid" in capture:
            def first_in(m, args, kwargs):
                h = args[0] if args else kwargs.get("hidden_states")
                cap["resid"][0] = h[0, pos].detach().float().cpu()
            regs.append((self.layers[0], "prekw", first_in))
            for i in range(self.n_layers):
                regs.append((self.layers[i], "post", lambda m, a, o, i=i: cap["resid"].__setitem__(
                    i + 1, _first(o)[0, pos].detach().float().cpu())))
        if "attn" in capture:
            for i in range(self.n_layers):
                regs.append((self.attn(i), "post", lambda m, a, o, i=i: cap["attn"].__setitem__(
                    i, _first(o)[0, pos].detach().float().cpu())))
        if "mlp" in capture:
            for i in range(self.n_layers):
                regs.append((self.mlp(i), "post", lambda m, a, o, i=i: cap["mlp"].__setitem__(
                    i, _first(o)[0, pos].detach().float().cpu())))
        for i in heads_at:
            def pre(m, args, i=i):
                x = args[0][0, pos].detach().float().cpu()
                heads[i] = x.view(self.n_heads, -1)
            regs.append((self.o_proj(i), "pre", pre))
        ids = ids.to(self.device)
        extra = {}
        if pos_offset:
            # Shift every position id: RoPE scores depend only on relative position, so in exact
            # arithmetic this changes nothing; any change is positional (aliasing / precision).
            extra["position_ids"] = torch.arange(ids.shape[1], device=self.device)[None] + int(pos_offset)
        self._count("fwd")
        self._count("rows")
        with self._hooks(regs):
            if pos == -1 and self._keeps_logits:
                extra["logits_to_keep"] = 1      # same lm_head on the last row (GPU: bit-identical; CPU: ~1e-7)
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra)
        logits = out.logits[0, -1 if pos == -1 else pos].float().cpu()
        stack = lambda xs: torch.stack(xs) if all(x is not None for x in xs) else None  # noqa: E731
        res = Run(logits=logits, resid=stack(cap["resid"]) if "resid" in capture else None,
                  attn=stack(cap["attn"]) if "attn" in capture else None,
                  mlp=stack(cap["mlp"]) if "mlp" in capture else None, heads=heads)
        if key is not None:
            self._memo[key] = res
        return res

    @torch.no_grad()
    def seq_capture(self, ids: torch.Tensor, depths: Sequence[int] = (), mlp_layers: Sequence[int] = (),
                    attn_layers: Sequence[int] = (), positions: Optional[Sequence[int]] = None,
                    **interventions: Any) -> dict[str, dict[int, torch.Tensor]]:
        """Every position at once (teacher-forced): residual after depth d ([T, d_model]; depth 0 =
        embeddings), MLP hidden activations (down_proj input, [T, d_ff]) and attention outputs.
        `positions` keeps only those rows (taken on the device, so long texts stay small)."""
        got: dict[str, dict[int, torch.Tensor]] = {"resid": {}, "mlp_act": {}, "attn": {}, "mlp": {}}
        regs = self.intervention_hooks(**interventions)
        sel = None if positions is None else torch.tensor(list(positions), dtype=torch.long, device=self.device)

        def keep(t: torch.Tensor) -> torch.Tensor:
            t = t[0]
            return (t if sel is None else t.index_select(0, sel)).detach().float().cpu()
        for d in depths:
            if d == 0:
                regs.append((self.layers[0], "prekw", lambda m, a, k: got["resid"].__setitem__(
                    0, keep(a[0] if a else k["hidden_states"]))))
            else:
                regs.append((self.layers[d - 1], "post", lambda m, a, o, d=d: got["resid"].__setitem__(
                    d, keep(_first(o)))))
        for i in mlp_layers:
            regs.append((self.down_proj(i), "pre", lambda m, a, i=i: got["mlp_act"].__setitem__(i, keep(a[0]))))
            regs.append((self.mlp(i), "post", lambda m, a, o, i=i: got["mlp"].__setitem__(i, keep(_first(o)))))
        for i in attn_layers:
            regs.append((self.attn(i), "post", lambda m, a, o, i=i: got["attn"].__setitem__(i, keep(_first(o)))))
        ids = ids.to(self.device)
        self._count("fwd")
        self._count("rows")
        # Only the hooked activations are used: compute the vocabulary projection for one row, not all of them
        # (the captured activations come before it, so they are identical).
        extra = {"logits_to_keep": 1} if self._keeps_logits else {}
        with self._hooks(regs):
            self.model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra)
        return got

    def neuron_attribution(self, ids: torch.Tensor, metric: Callable[[torch.Tensor], torch.Tensor],
                           layers: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Attribution patching: sum over positions of activation x d(metric)/d(activation) for every
        MLP neuron, [n_layers, d_ff]. Gradients run through the whole network, so a neuron whose
        effect reaches the last position through attention is counted too."""
        layers = list(range(self.n_layers)) if layers is None else list(layers)
        acts: dict[int, torch.Tensor] = {}

        def emb_hook(m, a, o):
            return o.detach().requires_grad_(True)

        regs = [(self.model.get_input_embeddings(), "post", emb_hook)]
        for i in layers:
            regs.append((self.down_proj(i), "pre", lambda m, a, i=i: acts.__setitem__(i, a[0])))
        ids = ids.to(self.device)
        with torch.enable_grad(), self._hooks(regs):
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
            m = metric(out.logits[0, -1].float())
            grads = torch.autograd.grad(m, [acts[i] for i in layers], allow_unused=True)
        d_ff = self.down_proj(layers[0]).weight.shape[1]
        res = torch.zeros(self.n_layers, d_ff)
        for i, g in zip(layers, grads):
            if g is not None:
                res[i] = (acts[i].detach().float() * g.float())[0].sum(0).cpu()
        return res

    @contextlib.contextmanager
    def eager_attention(self) -> Iterator[None]:
        """Attention weights need the eager implementation (sdpa / flash do not return them)."""
        prev = getattr(self.model.config, "_attn_implementation", None)
        changed = prev not in (None, "eager") and hasattr(self.model, "set_attn_implementation")
        if changed:
            self.model.set_attn_implementation("eager")
        try:
            yield
        finally:
            if changed:
                self.model.set_attn_implementation(prev)

    @torch.no_grad()
    def attention(self, ids: torch.Tensor, pos: int = -1, **interventions: Any) -> torch.Tensor:
        """[n_layers, n_heads, seq] attention weights from position `pos` to every position (CPU, float32)."""
        ids = ids.to(self.device)
        with self.eager_attention(), self._hooks(self.intervention_hooks(pos=pos, **interventions)):
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_attentions=True,
                             **({"logits_to_keep": 1} if self._keeps_logits else {}))    # logits unused
        return torch.stack([a[0, :, pos].float().cpu() for a in self.attentions_by_layer(out.attentions)])

    @torch.no_grad()
    def ablate_sweep(self, ids: torch.Tensor, dirs: Sequence[torch.Tensor], *, pos: int = -1,
                     chunk: int = 16) -> list[torch.Tensor]:
        """Logits at `pos` with direction dirs[b] removed from every residual write, one batch row
        per direction (the same prompt in every row, so no padding). Same maths as
        run(ablate=d) per row; one forward per `chunk` directions instead of one each."""
        out: list[torch.Tensor] = []
        dt = self.unembed.weight.dtype
        for s0 in range(0, len(dirs), chunk):
            part = dirs[s0:s0 + chunk]
            D = torch.stack([d / d.norm() for d in part]).to(self.device, dt)[:, None, :]    # [B, 1, d]
            batch = ids.to(self.device).expand(len(part), -1).contiguous()

            def proj_out(t: torch.Tensor) -> torch.Tensor:
                return t - (t * D).sum(-1, keepdim=True) * D
            regs = [(self.model.get_input_embeddings(), "post", lambda m, a, o: proj_out(o))]
            for i in range(self.n_layers):
                regs.append((self.attn(i), "post", lambda m, a, o: _replace_first(o, proj_out(_first(o)))))
                regs.append((self.mlp(i), "post", lambda m, a, o: _replace_first(o, proj_out(_first(o)))))
                regs.append((self.layers[i], "post", lambda m, a, o: _replace_first(o, proj_out(_first(o)))))
            self._count("fwd")
            self._count("rows", len(part))
            with self._hooks(regs):
                res = self.model(input_ids=batch, attention_mask=torch.ones_like(batch),
                                 **({"logits_to_keep": 1} if pos == -1 and self._keeps_logits else {}))
            out.extend(res.logits[:, -1 if pos == -1 else pos].float().cpu())
        return out

    @torch.no_grad()
    def _prefix_cache(self, ids: torch.Tensor):
        """Attention cache of every position but the last (clean), for last-position patches."""
        out = self.model(input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]), use_cache=True,
                         **({"logits_to_keep": 1} if self._keeps_logits else {}))
        self._count("fwd")
        return out.past_key_values

    @torch.no_grad()
    def patch_sweep(self, ids: torch.Tensor, where: str, vecs: dict[int, torch.Tensor], *, pos: int = -1,
                    chunk: int = 16) -> dict[int, torch.Tensor]:
        """Logits at `pos` with one component patched per batch row: row b replaces `where`
        ("resid" = layer output, "attn", "mlp") of layer i_b with vecs[i_b]. Same prompt in every
        row, so no padding; one forward per `chunk` layers instead of one per layer."""
        mod = {"resid": lambda i: self.layers[i], "attn": self.attn, "mlp": self.mlp}[where]
        items = list(vecs.items())
        out: dict[int, torch.Tensor] = {}
        if pos == -1 and ids.shape[1] > 1:
            # A patch at the last position cannot change any earlier position (causal attention):
            # compute the prefix once and run only the last token per patched row.
            import copy
            ids = ids.to(self.device)
            past = self._prefix_cache(ids)
            for s0 in range(0, len(items), chunk):
                part = items[s0:s0 + chunk]
                pb = copy.deepcopy(past)
                pb.batch_repeat_interleave(len(part))
                last = ids[:, -1:].expand(len(part), 1).contiguous()
                regs = []
                for b, (i, v) in enumerate(part):
                    def hook(m, a, o, b=b, v=v):
                        t = _first(o).clone()
                        t[b, -1] = v.to(t.device, t.dtype)
                        return _replace_first(o, t)
                    regs.append((mod(i), "post", hook))
                self._count("fwd")
                self._count("rows", len(part))
                with self._hooks(regs):
                    res = self.model(input_ids=last, past_key_values=pb,
                                     attention_mask=torch.ones(len(part), ids.shape[1], dtype=torch.long,
                                                               device=self.device))
                lg = res.logits[:, -1].float().cpu()
                for b, (i, _) in enumerate(part):
                    out[i] = lg[b]
            return out
        for s in range(0, len(items), chunk):
            part = items[s:s + chunk]
            batch = ids.to(self.device).expand(len(part), -1).contiguous()
            regs = []
            for b, (i, v) in enumerate(part):
                def hook(m, a, o, b=b, v=v):
                    t = _first(o)
                    if t.shape[1] == 1:
                        return o
                    t = t.clone()
                    t[b, pos] = v.to(t.device, t.dtype)
                    return _replace_first(o, t)
                regs.append((mod(i), "post", hook))
            self._count("fwd")
            self._count("rows", len(part))
            with self._hooks(regs):
                res = self.model(input_ids=batch, attention_mask=torch.ones_like(batch))
            lg = res.logits[:, pos].float().cpu()
            for b, (i, _) in enumerate(part):
                out[i] = lg[b]
        return out

    @contextlib.contextmanager
    def steer(self, **interventions: Any) -> Iterator[None]:
        """Interventions active for every forward (use around generate)."""
        self._steering += 1
        try:
            with self._hooks(self.intervention_hooks(**interventions)):
                yield
        finally:
            self._steering -= 1

    @contextlib.contextmanager
    def memo(self) -> Iterator[None]:
        """Reuse clean forward passes (no interventions) on the same tokens while this is open.

        The same prompt's clean pass is needed by targets, the trace, directions, the scorer
        and localisation; recomputing it gives the same numbers. Keyed by the tokens and a
        weights version (qbisect bumps it on every swap); never used while steer() hooks run.
        """
        prev, self._memo = self._memo, {}
        try:
            yield
        finally:
            self._memo = prev
