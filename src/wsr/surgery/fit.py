"""A learned edit: a low-rank weight change fitted by gradient descent, for the agent's `fit_edit`.

The hand-built edits (directions, neurons, gates) need the failure to sit in a few places. When it does not,
the agent can instead say what the model should and should not produce on a few texts and fit a small change:

    W' = W + U V^T        (one rank-r pair per chosen matrix; U starts at zero, so step 0 is the original)

trained to
- raise the likelihood of the target continuations (weight > 0),
- lower the likelihood of the failure continuations (weight < 0; unlikelihood, -log(1 - p), which stays
  bounded where plain likelihood maximisation would run off),
- keep the next-token distribution on the "keep" texts (regression replies and working replies by default)
  close to the original: KL over the original's top tokens at every position.

Only U and V get gradients (the model's own weights are untouched; the change is a forward hook while
training). The result is an ordinary low-rank delta: the workbench checks it like any edit (rank, size against
the tensor, frozen tensors, the free-parameter cap), and verify_edit / the held-out judge decide whether it
counts. Guards while fitting: a step cap, a wall-clock cap, a free-memory check before starting, and an
out-of-memory error ends the fit cleanly (nothing is left on the GPU).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from ..deep import batching as BT
from ..deep.lm import LM

TOPK = 32                    # tokens of the original distribution the keep loss compares at each position
MAX_TOKENS = 384             # per text: longer texts are cut (the start is kept)


@dataclass
class FitResult:
    deltas: list[tuple[str, torch.Tensor, torch.Tensor]]      # (param name, U [out, r], V [in, r]) on the CPU
    steps: int
    seconds: float
    loss_start: dict[str, float]
    loss_end: dict[str, float]
    stopped: str


def _linear(lm: LM, name: str) -> torch.nn.Linear:
    if not name.endswith(".weight"):
        raise ValueError(f"{name}: give a weight matrix (a name ending in .weight)")
    mods = dict(lm.model.named_modules())
    m = mods.get(name[: -len(".weight")])
    if not isinstance(m, torch.nn.Linear):
        raise ValueError(f"{name} is not the weight of a linear layer (fit_edit trains matrices only)")
    return m


MIN_FREE = 256 * 2**20       # below this much free GPU memory a fit is not started at all



def fit_lowrank(lm: LM, params: Sequence[str], targets: Sequence[tuple[torch.Tensor, int, float]],
                keep: Sequence[torch.Tensor], *, rank: int = 4, steps: int = 60, lr: float = 1e-2,
                kl_weight: float = 1.0, seconds: float = 300.0, seed: int = 0, max_rel_change: float = 1.0,
                log: Optional[Callable[[str], None]] = None) -> FitResult:
    """targets: (ids = prefix + continuation, prefix length, weight); keep: ids whose next-token
    distributions should not move. The change of each matrix is held below `max_rel_change` x its norm during
    training (scaled back after any step that goes past 90% of it), so a fit cannot end outside the build check
    on size. Returns the fitted deltas (zero-change ones are dropped)."""
    if not params:
        raise ValueError("name at least one weight matrix to fit")
    if not targets:
        raise ValueError("give at least one target text")
    t0 = time.time()
    dev = lm.device
    mods = [_linear(lm, n) for n in params]
    targets = [(ids[:, :MAX_TOKENS], min(P, MAX_TOKENS - 1), float(w)) for ids, P, w in targets]
    keep = [ids[:, :MAX_TOKENS] for ids in keep]
    # Memory is measured, not guessed (a formula cannot be right for every GPU, model and dtype): the first
    # passes run for real, and an out-of-memory error ends the fit cleanly with a clear refusal. A fit is not
    # started at all with almost no free memory (BT.free_bytes already keeps a reserve for other programs).
    if dev.type == "cuda" and BT.free_bytes(dev) < MIN_FREE:
        raise ValueError(f"not enough free GPU memory to fit ({BT.free_bytes(dev) / 2**20:.0f} MB free)")
    g = torch.Generator().manual_seed(seed)
    UV = []
    for m in mods:
        out_f, in_f = m.weight.shape
        U = torch.zeros(out_f, rank, device=dev, requires_grad=True)
        V = (torch.randn(in_f, rank, generator=g) / in_f ** 0.5).to(dev).requires_grad_(True)
        UV.append((U, V))
    flat = [p for uv in UV for p in uv]
    caps = [0.9 * max_rel_change * float(m.weight.detach().float().norm()) for m in mods]
    opt = torch.optim.Adam(flat, lr=lr)

    def hook(U, V):
        return lambda m, a, o: o + ((a[0].float() @ V) @ U.T).to(o.dtype)

    regs = [(m, "post", hook(U, V)) for m, (U, V) in zip(mods, UV)]
    keeps_logits = getattr(lm, "_keeps_logits", False)

    @torch.no_grad()
    def reference(ids):
        x = ids.to(dev)
        lp = lm.model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0].float().log_softmax(-1)
        v, i = lp.topk(TOPK, -1)
        return i, v

    try:
        refs = [reference(k) for k in keep]
    except Exception as e:                          # noqa: BLE001 - memory errors become a clear refusal
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if BT.is_oom(e):
            raise ValueError("out of GPU memory reading the keep texts: use shorter or fewer texts") from None
        raise

    def losses(with_grad: bool) -> tuple[dict[str, float], list[torch.Tensor]]:
        grads = [torch.zeros_like(p) for p in flat]
        tl, kl = 0.0, 0.0
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx, lm._hooks(regs):
            for ids, P, w in targets:
                x = ids.to(dev)
                k = x.shape[1] - P + 1
                extra = {"logits_to_keep": k} if keeps_logits else {}
                lg = lm.model(input_ids=x, attention_mask=torch.ones_like(x), **extra).logits[0][-k:-1].float()
                lp = lg.log_softmax(-1).gather(1, x[0, P:, None])[:, 0]
                if w > 0:
                    li = -lp.mean() * w
                else:
                    li = -torch.log1p(-lp.exp().clamp(max=1 - 1e-6)).mean() * (-w)
                tl += float(li) / len(targets)
                if with_grad:
                    for acc, gr in zip(grads, torch.autograd.grad(li / len(targets), flat, allow_unused=True)):
                        if gr is not None:
                            acc += gr
            for ids, (ri, rv) in zip(keep, refs):
                x = ids.to(dev)
                lp = lm.model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0].float().log_softmax(-1)
                li = (rv.exp() * (rv - lp.gather(1, ri))).sum(-1).mean() * kl_weight
                kl += float(li) / max(1, len(keep))
                if with_grad:
                    for acc, gr in zip(grads, torch.autograd.grad(li / len(keep), flat, allow_unused=True)):
                        if gr is not None:
                            acc += gr
        return {"target": round(tl, 4), "keep_kl": round(kl, 4)}, grads

    stopped = "steps"
    frozen = [(p, p.requires_grad) for p in lm.model.parameters()]
    for p, _ in frozen:                            # gradients for U and V only
        p.requires_grad_(False)
    if dev.type == "cuda":
        free0 = BT.free_bytes(dev)
        reserved0 = torch.cuda.memory_reserved(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    try:
        start, _ = losses(False)
        done = 0
        for step in range(int(steps)):
            if time.time() - t0 > seconds:
                stopped = f"time limit ({seconds:g} s)"
                break
            _, grads = losses(True)
            opt.zero_grad(set_to_none=True)
            for p, gr in zip(flat, grads):
                p.grad = gr
            opt.step()
            with torch.no_grad():
                for (U, V), cap in zip(UV, caps):
                    dn = float(torch.sqrt(((U.T @ U) * (V.T @ V)).sum().clamp(min=0)))
                    if dn > cap > 0:
                        U.mul_(cap / dn)
            done = step + 1
            if step == 0 and dev.type == "cuda":
                # Measured after one real step: more than was free means the driver is spilling into system memory
                # (Windows' sysmem fallback does that instead of raising out-of-memory) or is about to run out.
                extra = torch.cuda.max_memory_reserved(dev) - reserved0
                if extra > free0:
                    raise ValueError(f"this fit needs about {extra / 2**30:.2f} GB of GPU memory and "
                                     f"{free0 / 2**30:.2f} GB was free: use shorter texts, fewer of them, a lower rank "
                                     "or later layers")
            if log and step % 20 == 0:
                log(f"  fit step {step}")
        end, _ = losses(False)
    except Exception as e:                          # noqa: BLE001 - memory errors become a clear refusal
        if BT.is_oom(e):
            raise ValueError("out of GPU memory while fitting: use shorter texts, fewer of them, a lower rank "
                             "or later layers") from None
        raise
    finally:
        for p, rg in frozen:
            p.requires_grad_(rg)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    deltas = [(n, U.detach().float().cpu(), V.detach().float().cpu()) for n, (U, V) in zip(params, UV)
              if float(U.detach().abs().sum()) > 0]
    return FitResult(deltas, done, round(time.time() - t0, 1), start, end, stopped)
