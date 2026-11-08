"""Which quantised tensor causes a failure: quantisation bisection.

`wsr quant` finds tensors whose quantisation error is abnormal; it cannot say
that one of them causes the bad reply. This module rebuilds the served model in
PyTorch from the GGUF's own dequantised tensors and then swaps tensors between
the quantised and the full-precision version:

1. Reproduce. The quantised model must fail where the full-precision one
   passes. Both fail: not quantisation. The quantised torch model passes but the
   server fails: llama.cpp's kernels or runtime, not the weights.
2. Sweep. At the first token where the two replies part, restore each tensor
   to full precision alone and measure how much of the logit gap it closes.
3. Minimal set. Restore tensors in that order until the decision flips, then
   drop any the flip does not need.
4. Prove both ways, with a control, on generation: the quantised model with
   only that set restored must pass; the full-precision model with only that
   set quantised must fail; restoring the same number of random tensors of the
   same kinds must not fix it.
5. No small set flips it: the error is diffuse (requantise at more bits).

Consistent swaps. Some converters (AWQ / SmoothQuant; Qwen's official GGUFs)
fold per-channel scales into the norm gains and the next matrices: each tensor
differs from the original, the set does not. Restoring one original tensor among
folded neighbours would break the model, so its full-precision version is first
mapped into the file's own scale space with the per-row / per-column fit from
quant.py (W_fp / (r c)). Norm gains and biases are float32 in a GGUF, carry no
quantisation error, and stay as the file has them in both versions.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..gguf import read_gguf
from . import quant as Q
from .lm import LM

_LAYER = re.compile(r"\.(?:layers|h|blocks)\.(\d+)\.")


def _permute_back(q: np.ndarray, n_head: int) -> np.ndarray:
    """Inverse of quant._unpermute_rope: llama.cpp's q/k row order back to the HF order."""
    r, c = q.shape
    return q.reshape(n_head, 2, r // n_head // 2, c).swapaxes(1, 2).reshape(r, c)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0))


@dataclass
class Swap:
    name: str                  # model parameter name
    gguf: str
    qtype: str
    q: torch.Tensor            # quantised value, in the model's parameter layout (CPU)
    f: torch.Tensor            # full-precision value in the same (file's) scale space
    layout: str
    rel_error: float
    layer: Optional[int]
    kind: str                  # attn_q, ffn_down, token_embd ...


@dataclass
class QuantModel:
    """The model with every matched parameter switchable between its GGUF and full-precision value."""
    lm: LM
    swaps: dict[str, Swap]
    skipped: list[str] = field(default_factory=list)
    state: set[str] = field(default_factory=set)     # names currently quantised
    check: dict = field(default_factory=dict)        # rebuilt full precision vs the original model

    @property
    def quantised_kinds(self) -> list[str]:
        """Tensors the file actually quantised (float tensors carry no quantisation error)."""
        return [n for n, s in self.swaps.items() if s.qtype not in ("F32", "F16", "BF16")]

    def set(self, quantised: set[str]) -> None:
        params = dict(self.lm.model.named_parameters())
        for n, s in self.swaps.items():
            want = n in quantised
            if want != (n in self.state) or not self.state:
                p = params[n]
                p.data.copy_((s.q if want else s.f).to(p.device, p.dtype))
        self.state = set(quantised) & set(self.swaps)
        self.lm._wver += 1                  # weights changed: cached clean passes no longer apply

    def all_quantised(self) -> None:
        self.set(set(self.swaps))

    def all_fp(self) -> None:
        self.set(set())


def build(lm: LM, gguf_path: str | Path, log=None, check_prompt: str = "Hello! How are you today?") -> QuantModel:
    """Dequantise the whole GGUF into the model's parameters' layout, next to full-precision copies.

    The full-precision side is the original model mapped into the file's own scale space by exact,
    paired transforms, so it computes the same function as the original by construction:
      * a folded norm gain n_gguf = n / c (norms are float32 in a GGUF, so c = n / n_gguf exactly):
        every matrix reading that norm (q, k, v; gate, up) gets its columns times c;
      * folded MLP hidden units (up rows / s, down columns * s): s from the up rows, applied as a pair;
      * folded value rows (v rows / s, o columns * s, per KV head for GQA): the same, as a pair.
    Fitting each tensor on its own (the first version) left a pair half-mapped whenever one fit was
    rejected, and the "full-precision" model replied in gibberish. A self-check now compares the
    rebuilt full-precision logits with the original model's before anything else.
    """
    path = Path(gguf_path)
    g = read_gguf(str(path))
    params = dict(lm.model.named_parameters())
    names = set(params)
    n_head = g.arch_key("attention.head_count")
    n_kv = g.arch_key("attention.head_count_kv")
    plus_one_arch = "gemma" in str(getattr(lm.cfg, "model_type", "")).lower()
    probe = lm.encode(check_prompt)
    ref_logits = lm.run(probe, capture=()).logits
    raw: dict[str, dict] = {}
    skipped: list[str] = []
    for t in g.tensors:
        base = t.name[:-5] + ".weight" if t.name.endswith(".bias") else t.name
        rn = Q.hf_name(base, names)
        if rn and t.name.endswith(".bias"):
            rn = rn[:-7] + ".bias"
        if not rn or rn not in params:
            skipped.append(t.name)
            continue
        if t.type_name not in Q.SUPPORTED:
            skipped.append(f"{t.name} ({t.type_name})")
            continue
        x = Q.load_tensor(path, g, t, max_rows=10 ** 9)
        w = params[rn].detach().float().cpu().numpy().copy()   # a copy: on CPU float32 it is otherwise a view
        if x.values.size != w.size:
            skipped.append(f"{t.name}: shape {x.values.shape} vs {tuple(w.shape)}")
            continue
        q = x.values.reshape(w.shape).astype(np.float32)
        layout = "plain"
        if q.ndim == 1:
            # Gemma-family norms are x * (1 + w) in HF and stored as 1 + w in a GGUF. Decided by the
            # architecture, not by which is closer: a folded Qwen norm can sit closer to 1 + w by chance
            # (layer 0 of Qwen2.5-0.5B did, and the rebuilt model's layer 0 was 24x off).
            if plus_one_arch and "norm" in t.name:
                q, layout = q - 1.0, "plus_one"
        else:
            cands = {"plain": q}
            for nh in {n_head, n_kv} - {None}:
                if isinstance(nh, int) and q.shape[0] % (2 * nh) == 0 and ("attn_q" in t.name or "attn_k" in t.name):
                    cands[f"rope_{nh}"] = _permute_back(q, nh)
            layout, q = min(cands.items(), key=lambda kv: _rel(kv[1], w))
        raw[t.name] = {"rn": rn, "q": q, "w": w, "f": w.copy(), "layout": layout, "type": t.type_name}
        if log and len(raw) % 50 == 0:
            log(f"  dequantised {len(raw)} tensors")
    folds: list[str] = []

    def gain(r: dict, which: str) -> np.ndarray:
        v = r[which]
        return v + 1.0 if r["layout"] == "plus_one" else v

    for name, r in raw.items():
        if r["q"].ndim == 1:
            r["f"] = r["q"].copy()                       # float in the file: no quantisation error
    layers = sorted({int(m.group(1)) for m in (re.match(r"^blk\.(\d+)\.", n) for n in raw) if m})
    for i in layers:
        P = lambda k: raw.get(f"blk.{i}.{k}.weight")  # noqa: E731
        for norm, mats in (("attn_norm", ("attn_q", "attn_k", "attn_v")), ("ffn_norm", ("ffn_gate", "ffn_up"))):
            nr = P(norm)
            if nr is None:
                continue
            ng, nh_ = gain(nr, "q"), gain(nr, "w")
            c = np.where(np.abs(ng) > 1e-12, nh_ / np.where(np.abs(ng) > 1e-12, ng, 1.0), 1.0)
            if np.abs(np.log(np.abs(c) + 1e-30)).max() > 1e-3:
                for m in mats:
                    if P(m) is not None:
                        P(m)["f"] = P(m)["f"] * c[None, :]
                folds.append(f"blk.{i}.{norm}")
        up, down = P("ffn_up"), P("ffn_down")
        if up is not None and down is not None:
            fw, qq = up["f"].astype(np.float64), up["q"].astype(np.float64)
            sc = (fw * fw).sum(1) / np.where(np.abs((fw * qq).sum(1)) > 1e-30, (fw * qq).sum(1), 1.0)
            if np.std(np.log(np.abs(sc) + 1e-30)) > 0.03 and (sc > 0).all():
                up["f"] = (up["f"] / sc[:, None]).astype(np.float32)
                down["f"] = (down["f"] * sc[None, :]).astype(np.float32)
                folds.append(f"blk.{i}.ffn_up/down")
        v, o = P("attn_v"), P("attn_output")
        vb = raw.get(f"blk.{i}.attn_v.bias")
        if v is not None and o is not None and isinstance(n_head, int) and isinstance(n_kv, int) and n_kv:
            fw, qq = v["f"].astype(np.float64), v["q"].astype(np.float64)
            sv = (fw * fw).sum(1) / np.where(np.abs((fw * qq).sum(1)) > 1e-30, (fw * qq).sum(1), 1.0)
            if vb is not None:
                # float32 bias in the file: its ratio to the original is the row scale, exactly
                ok_b = np.abs(vb["w"]) > 1e-6
                sv = np.where(ok_b, vb["w"] / np.where(ok_b, vb["q"], 1.0), 1.0)
            # real folding varies across rows; quantisation alone shifts all rows alike by ~1-2 %
            if np.std(np.log(np.abs(sv) + 1e-30)) > 0.03 and (sv > 0).all():
                hd = v["f"].shape[0] // n_kv
                grp = n_head // n_kv
                cols = np.array([sv[(h // grp) * hd + d] for h in range(n_head) for d in range(hd)])
                if cols.size == o["f"].shape[1]:
                    v["f"] = (v["f"] / sv[:, None]).astype(np.float32)
                    o["f"] = (o["f"] * cols[None, :]).astype(np.float32)
                    b = raw.get(f"blk.{i}.attn_v.bias")
                    if b is not None:
                        b["f"] = (b["w"] / sv).astype(np.float32)
                    folds.append(f"blk.{i}.attn_v/o")
    swaps: dict[str, Swap] = {}
    for name, r in raw.items():
        rn = r["rn"]
        m = _LAYER.search(rn)
        kind = re.sub(r"^blk\.\d+\.", "", name).replace(".weight", "")
        lay = r["layout"] + ("+folded" if any(name.startswith(f.split("/")[0].rsplit(".", 1)[0] + ".") and
                                              kind.split(".")[0] in ("attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up",
                                                                     "ffn_down", "attn_output") for f in folds) else "")
        swaps[rn] = Swap(rn, name, r["type"], torch.from_numpy(np.ascontiguousarray(r["q"], dtype=np.float32)),
                         torch.from_numpy(np.ascontiguousarray(r["f"], dtype=np.float32)), lay,
                         _rel(r["q"], r["f"]), int(m.group(1)) if m else None, kind)
    qm = QuantModel(lm, swaps, skipped)
    qm.state = set(swaps)            # force a full write on the first set()
    qm.all_fp()
    got = lm.run(probe, capture=()).logits
    lp0, lp1 = ref_logits.log_softmax(-1), got.log_softmax(-1)
    qm.check = {"kl_vs_original": float((lp0.exp() * (lp0 - lp1)).sum()), "same_top": bool(lp0.argmax() == lp1.argmax()),
                "folds": len(folds), "folded": folds[:12]}
    qm.check["ok"] = qm.check["kl_vs_original"] < 0.01 and qm.check["same_top"]
    return qm


# -- the bisection ---------------------------------------------------------------------------------

@dataclass
class Divergence:
    ids: torch.Tensor          # prompt + quantised reply up to the first token that differs
    q_tok: int
    f_tok: int
    at: int                    # index in the reply

    def metric(self, logits: torch.Tensor) -> float:
        return float(logits[self.q_tok] - logits[self.f_tok])


def _divergence(prompt_ids: torch.Tensor, gen_q: list[int], gen_f: list[int]) -> Optional[Divergence]:
    for j in range(min(len(gen_q), len(gen_f))):
        if gen_q[j] != gen_f[j]:
            ids = torch.cat([prompt_ids, torch.tensor([gen_q[:j]], dtype=torch.long)], dim=1)
            return Divergence(ids, gen_q[j], gen_f[j], j)
    return None


def bisect(qm: QuantModel, clf, messages: list[dict], *, max_set: int = 24, n_null: int = 3,
           seed: int = 0, log=None) -> dict[str, Any]:
    lm = qm.lm
    out: dict[str, Any] = {"tensors": len(qm.swaps), "quantised_tensors": len(qm.quantised_kinds),
                           "skipped": qm.skipped[:20], "rebuild_check": qm.check}
    if qm.check and not qm.check.get("ok"):
        out["outcome"] = "rebuild_failed"
        return out
    qm.all_quantised()
    eq = clf.run("quantised", messages)
    qm.all_fp()
    ef = clf.run("full_precision", messages)
    out.update(quantised_reply=eq.text[:300], fp_reply=ef.text[:300], quantised_groups=sorted(eq.groups),
               fp_groups=sorted(ef.groups))
    if not eq.groups:
        out["outcome"] = "quantised_passes"
        return out
    if ef.groups:
        out["outcome"] = "both_fail"
        return out
    div = _divergence(eq.prompt_ids, eq.gen, ef.gen)
    if div is None:
        out["outcome"] = "no_divergence"
        return out
    out["divergence"] = {"at_token": div.at, "quantised_token": lm.token_str(div.q_tok),
                         "fp_token": lm.token_str(div.f_tok)}
    cand = qm.quantised_kinds

    def m_with(quantised: set[str]) -> float:
        qm.set(quantised)
        return div.metric(lm.run(div.ids, capture=()).logits)

    allq = set(cand)
    m_q, m_f = m_with(allq), m_with(set())
    gap = m_q - m_f
    out.update(metric_quantised=round(m_q, 3), metric_fp=round(m_f, 3))
    if gap <= 1e-3:
        out["outcome"] = "no_gap"
        return out
    # 2. sweep: restore one tensor at a time
    eff = {}
    for i, n in enumerate(cand):
        eff[n] = (m_q - m_with(allq - {n})) / gap
        if log and i % 40 == 0:
            log(f"  sweep {i}/{len(cand)}")
    ranked = sorted(cand, key=lambda n: -eff[n])
    out["top_single"] = [{"tensor": qm.swaps[n].gguf, "restores": round(eff[n], 3), "type": qm.swaps[n].qtype,
                          "rel_error": round(qm.swaps[n].rel_error, 4)} for n in ranked[:10]]
    # 3. minimal set: greedy restore until the decision flips, then prune
    chosen: list[str] = []
    for n in ranked[:max_set]:
        chosen.append(n)
        if m_with(allq - set(chosen)) < 0:
            break
    else:
        out["outcome"] = "diffuse"
        out["restored_share_at_max"] = round((m_q - m_with(allq - set(chosen))) / gap, 3)
        qm.all_fp()
        return out
    for n in list(chosen):
        trial = [x for x in chosen if x != n]
        if trial and m_with(allq - set(trial)) < 0:
            chosen = trial
    S = set(chosen)
    out["minimal_set"] = [qm.swaps[n].gguf for n in chosen]
    # 4. proofs on generation
    qm.set(allq - S)
    fixed = clf.run("quantised_minus_set", messages)
    qm.set(S)
    broken = clf.run("fp_plus_set", messages)
    rng = random.Random(seed)
    nulls = []
    for _ in range(n_null):
        pool = [n for n in cand if n not in S]
        same = []
        for n in chosen:                                  # same kinds where possible, other layers
            opts = [x for x in pool if qm.swaps[x].kind == qm.swaps[n].kind and x not in same] or pool
            same.append(rng.choice(opts))
        qm.set(allq - set(same))
        nulls.append(not clf.run("quantised_minus_random", messages).groups)
    qm.all_fp()
    out["proof"] = {"restoring_set_fixes": not fixed.groups, "fixed_reply": fixed.text[:200],
                    "quantising_set_alone_breaks": bool(broken.groups), "broken_reply": broken.text[:200],
                    "broken_groups": sorted(broken.groups), "random_sets_that_fix": sum(nulls), "random_sets": n_null}
    p = out["proof"]
    specific = p["restoring_set_fixes"] and p["random_sets_that_fix"] == 0
    if specific and p["quantising_set_alone_breaks"]:
        out["outcome"] = "localised"            # necessary and sufficient
    elif specific:
        out["outcome"] = "necessary"            # restoring fixes; alone it does not cause it
    else:
        out["outcome"] = "unconfirmed"
    out["near_tie"] = bool(abs(m_q) < 0.1)
    return out


def summary(b: dict[str, Any]) -> str:
    o = b.get("outcome")
    if o == "rebuild_failed":
        c = b.get("rebuild_check", {})
        return (f"cannot rebuild a full-precision model that matches the original in the file's scale space "
                f"(KL {c.get('kl_vs_original', float('nan')):.3g}): the conversion did more than fold scales; "
                "no tensor is blamed")
    if o == "quantised_passes":
        return ("the quantised weights, run in PyTorch, answer fine: if the server still fails, the cause is its "
                "kernels or runtime (llama.cpp build, backend, flags), not the quantised weights")
    if o == "both_fail":
        return "full precision fails too: the failure is not caused by quantisation"
    if o in ("no_divergence", "no_gap"):
        return "the quantised and full-precision replies do not part where the failure is chosen"
    if o == "diffuse":
        return (f"no set of up to {len(b.get('top_single', []))} tensors flips it (restoring the top ones closes "
                f"{b.get('restored_share_at_max', 0):.0%} of the gap): the quantisation error is diffuse; "
                "requantise at more bits")
    s = b.get("minimal_set", [])
    p = b.get("proof", {})
    names = ", ".join(s)
    it = "it" if len(s) == 1 else "them"
    ctrl = f"restoring the same number of random tensors fixes it {p.get('random_sets_that_fix')}/{p.get('random_sets')}"
    if o == "localised":
        return (f"the failure is caused by the quantisation of {names}: restoring {it} to full precision fixes the "
                f"reply, quantising {it} alone in the full-precision model reproduces the failure, and {ctrl}")
    if o == "necessary":
        return (f"restoring {names} to full precision fixes the reply ({ctrl}), but quantising {it} alone does not "
                "cause it: the failure needs this tensor's error on top of the rest of the quantisation error")
    line = (f"restoring {names} fixes the reply, but so does restoring random tensors "
            f"({p.get('random_sets_that_fix')}/{p.get('random_sets')}): no tensor is specifically to blame")
    if b.get("near_tie"):
        line += (f"; the reply is a near-tie (lead {b.get('metric_quantised', 0):+.2f} logits), so almost any change "
                 "tips it")
    return line
