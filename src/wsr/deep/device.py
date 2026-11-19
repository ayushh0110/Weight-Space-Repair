"""Device helpers shared by the deep tools: clearing a device's cache, measuring what a step used, the device's name
in cache keys, and everything for Apple's MPS (Metal). The CUDA-only code that was already there (free VRAM in
batching, the float32-or-half choice in lm.auto_dtype, llama-server's placement in program/backends) stays where
it was, unchanged, with an MPS branch beside it.

CUDA and CPU behave exactly as before this module existed. MPS is opt-in (`--device mps`): `--device auto` still
picks CUDA, else the CPU, on every machine, Macs included. Before a model is put on MPS:

* `mps_problem()` says why MPS cannot be used here (not a Mac, macOS older than 14, a torch build without MPS, or
  no usable Metal device: a virtual Mac can report MPS as available when it is not);
* `self_test()` runs a tiny random model, built in code, on the CPU and on MPS: logits agree, two greedy runs give
  the same tokens, a batched row agrees with the row alone, a weight restored from its CPU copy is bit-identical,
  and an op MPS lacks (an SVD) runs through PyTorch's CPU fallback. Its result is kept per chip, macOS and library
  versions (cache group `device`), so it runs once;
* after loading, `repeat_problem(lm)` generates a short greedy reply twice with the real model: wsr's judge pairs
  the original and the candidate run on the same device, which only means something when a run repeats exactly.

When the self-test or the repeat check fails, the model runs on the CPU and the reason is printed. Nothing here
can make MPS results equal to the CPU's; it checks that they are sound and repeatable, which is what the paired
judge needs. float64 never goes to MPS (Metal has no double type): code that needs it moves the data to the CPU.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from typing import Any, Optional

import torch

from .. import macos

RESERVE_MPS = 1 << 30          # unified memory: kept for macOS and other programs, on top of Metal's own limit
SELF_TEST_VERSION = 1          # bump when the self-test's checks change: earlier results are not reused
LOGIT_TOL = 1e-3               # float32 CPU vs MPS, per logit (a tiny model's logits are about 0.1 to 1)


# -- the basics ---------------------------------------------------------------------------------------------------

def kind(device: Any) -> str:
    return torch.device(device).type


def is_gpu(device: Any) -> bool:
    """CUDA or MPS: batched to the GPU's cap, not the CPU's."""
    return kind(device) in ("cuda", "mps")


def empty_cache(device: Any) -> None:
    """Hand the memory PyTorch caches but does not use back to the device (a no-op on the CPU)."""
    t = kind(device)
    if t == "cuda":
        torch.cuda.empty_cache()
    elif t == "mps":
        try:
            torch.mps.empty_cache()
        except (AttributeError, RuntimeError):
            pass


def name(device: Any) -> str:
    """The device as cache keys name it: the GPU's name on CUDA, "cpu" on the CPU (both as before), and on MPS the
    chip and the macOS version (Metal's kernels change between macOS releases)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    if dev.type == "mps":
        return "mps: " + macos.describe()
    return "cpu"


# -- memory -------------------------------------------------------------------------------------------------------

def mps_limit() -> Optional[int]:
    """The memory Metal lets this process's GPU work use: PyTorch's figure where it has one, else macOS's."""
    f = getattr(torch.mps, "recommended_max_memory", None)
    if f is not None:
        try:
            v = int(f())
            if v > 0:
                return v
        except (RuntimeError, AttributeError):
            pass
    return macos.gpu_limit_bytes()


def mps_used() -> int:
    try:
        return int(torch.mps.driver_allocated_memory())
    except (AttributeError, RuntimeError):
        return 0


def mps_free_bytes() -> int:
    """Free memory for new MPS tensors: the smaller of what Metal's limit leaves and what the RAM has (the CPU and the
    GPU share it), each less its reserve."""
    from . import batching as BT

    limit = mps_limit()
    ram = BT.ram_available() - BT.RESERVE_CPU
    if limit is None:
        return max(0, ram)
    return max(0, min(limit - mps_used() - RESERVE_MPS, ram))


def mark(device: Any) -> Optional[int]:
    """Start measuring what a step allocates (see `grown`). None on the CPU."""
    dev = torch.device(device)
    if dev.type == "cuda":
        reserved0 = torch.cuda.memory_reserved(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        return int(reserved0)
    if dev.type == "mps":
        return mps_used()
    return None


def grown(device: Any, start: Optional[int]) -> Optional[int]:
    """Bytes the device took since `mark`: CUDA's peak reserved memory; on MPS what the driver holds now (MPS has no
    peak counter; its allocator keeps what a step took, so after one step this is close to the peak)."""
    if start is None:
        return None
    dev = torch.device(device)
    if dev.type == "cuda":
        return int(torch.cuda.max_memory_reserved(dev)) - start
    if dev.type == "mps":
        return mps_used() - start
    return None


# -- MPS: can it be used, and is it sound ------------------------------------------------------------------------

MIN_MACOS = (14, 0)


def mps_problem() -> Optional[str]:
    """Why MPS cannot be used here, or None."""
    if not macos.is_mac():
        return "MPS is Apple's GPU backend: it needs a Mac with Apple Silicon (use --device cpu or cuda)"
    v = macos.macos_version()
    if v is not None and v < MIN_MACOS:
        return (f"wsr uses MPS on macOS {MIN_MACOS[0]} or later (this Mac runs {'.'.join(map(str, v))}); "
                "use --device cpu")
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_built():
        return "this torch build has no MPS support: install torch from PyPI on the Mac (pip install torch)"
    if not mps.is_available():
        return "torch reports no usable MPS device on this Mac"
    try:
        x = torch.ones(4, device="mps")
        float((x + 1).sum().cpu())
    except Exception as e:                               # noqa: BLE001 - a virtual Mac can say MPS is available
        return f"an MPS tensor could not be used ({type(e).__name__}: {str(e)[:120]}); use --device cpu"
    return None


def _tiny_model():
    """A 2-layer Llama-style model with random weights, built in code (no download)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        m = LlamaForCausalLM(cfg)
    return m.float().eval()


def _check(checks: list, name_: str, ok: bool, detail: str) -> None:
    checks.append({"check": name_, "ok": bool(ok), "detail": detail})


def run_self_test(device: str = "mps") -> dict[str, Any]:
    """The checks themselves, on `device` against the CPU (no cache). `device` "cpu" runs them CPU against CPU,
    which is how the tests exercise this code on machines without MPS."""
    import copy

    checks: list[dict] = []
    t0 = time.time()
    ref = _tiny_model()
    m = copy.deepcopy(ref).to(device)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 97, (1, 16), generator=g)
    with torch.no_grad():
        a = ref(input_ids=ids).logits.float()
        b = m(input_ids=ids.to(device)).logits.float().cpu()
        d = float((a - b).abs().max())
        _check(checks, "logits", d <= LOGIT_TOL, f"largest difference from the CPU {d:.2e} (limit {LOGIT_TOL:g})")

        kw = dict(max_new_tokens=12, do_sample=False, pad_token_id=0)
        x = ids.to(device)
        r1 = m.generate(x, attention_mask=torch.ones_like(x), **kw)[0].tolist()
        r2 = m.generate(x, attention_mask=torch.ones_like(x), **kw)[0].tolist()
        _check(checks, "repeat", r1 == r2, "two greedy runs gave the same tokens" if r1 == r2 else
               "two greedy runs gave different tokens")

        short = ids[:, 4:]
        pad = torch.zeros(1, 4, dtype=ids.dtype)
        batch = torch.cat([ids, torch.cat([pad, short], 1)]).to(device)
        mask = torch.cat([torch.ones(1, 16, dtype=ids.dtype), torch.cat([torch.zeros(1, 4, dtype=ids.dtype),
                                                                          torch.ones_like(short)], 1)]).to(device)
        lb = m(input_ids=batch, attention_mask=mask).logits[:, -1].float().cpu()
        l0 = m(input_ids=ids.to(device)).logits[:, -1].float().cpu()
        l1 = m(input_ids=short.to(device)).logits[:, -1].float().cpu()
        db = max(float((lb[0] - l0[0]).abs().max()), float((lb[1] - l1[0]).abs().max()))
        _check(checks, "batch", db <= LOGIT_TOL, f"a batched row differs from the row alone by {db:.2e} "
               f"(limit {LOGIT_TOL:g})")

        w = m.model.layers[0].mlp.down_proj.weight
        saved = w.detach().to("cpu", copy=True)
        w.add_(torch.randn(w.shape, generator=g).to(w.device) * 0.01)
        w.copy_(saved.to(w.device))
        same = torch.equal(w.detach().cpu(), saved)
        _check(checks, "restore", same, "a weight restored from its CPU copy is bit-identical" if same else
               "a weight restored from its CPU copy differs")

    try:
        s = torch.linalg.svdvals(torch.randn(8, 8, generator=g).to(device))
        ok = bool(torch.isfinite(s.cpu()).all())
        _check(checks, "fallback", ok, "an op MPS lacks (SVD) runs through the CPU fallback" if kind(device) == "mps"
               else f"an SVD runs on {kind(device)} (native here; on MPS it needs the CPU fallback)")
    except Exception as e:                               # noqa: BLE001
        _check(checks, "fallback", False, f"an op MPS lacks (SVD) failed: {type(e).__name__}: {str(e)[:120]}. "
               "Set PYTORCH_ENABLE_MPS_FALLBACK=1 before Python starts (wsr sets it when it is imported before "
               "torch)")
    bf16 = False
    if kind(device) == "mps":
        try:
            y = torch.ones(4, 4, dtype=torch.bfloat16, device=device)
            bf16 = bool(torch.isfinite((y @ y).float().cpu()).all())
        except Exception:                                # noqa: BLE001 - only says whether bfloat16 is offered
            bf16 = False
    passed = all(c["ok"] for c in checks)
    failed = [c for c in checks if not c["ok"]]
    return {"version": SELF_TEST_VERSION, "device": device, "passed": passed, "checks": checks, "bfloat16": bf16,
            "seconds": round(time.time() - t0, 2),
            "why": "; ".join(f"{c['check']}: {c['detail']}" for c in failed) if failed else None}


def _self_test_key() -> str:
    import transformers

    from .. import __version__
    s = f"{SELF_TEST_VERSION}|{macos.describe()}|{torch.__version__}|{transformers.__version__}|{__version__}"
    return hashlib.sha256(s.encode()).hexdigest()[:20]


_SELF_TESTS: dict[str, dict[str, Any]] = {}          # this process's results, by machine and versions


def self_test(cache: bool = True) -> dict[str, Any]:
    """`run_self_test("mps")`, once per process, and a pass kept on disk per chip, macOS, torch, transformers and wsr
    version. A failure is not kept on disk: it may be passing (memory pressure, another program on the GPU), so the
    next run tests again."""
    from .. import cache as C

    key = _self_test_key()
    if cache and key in _SELF_TESTS:
        return _SELF_TESTS[key]
    f = C.base() / "device" / f"mps-selftest-{key}.json"
    if cache and C.enabled() and f.is_file():
        try:
            got = json.loads(f.read_text(encoding="utf-8"))
            if got.get("version") == SELF_TEST_VERSION and got.get("passed"):
                _SELF_TESTS[key] = got
                return got
        except (OSError, ValueError):
            pass
    res = run_self_test("mps")
    res["machine"] = macos.describe()
    _SELF_TESTS[key] = res
    if cache and C.enabled() and res.get("passed"):
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(res, indent=1), encoding="utf-8")
        except OSError:
            pass
    return res


_REPEAT_PROMPT = "List three colours and one fact about each."


def repeat_problem(lm: Any, max_new: int = 12) -> Optional[str]:
    """None when a short greedy reply of the real model is the same twice; else why not."""
    try:
        ids = lm.encode([{"role": "user", "content": _REPEAT_PROMPT}])
    except Exception:                                    # noqa: BLE001 - no chat template: plain text
        ids = lm.tok(_REPEAT_PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        a = lm.generate(ids, max_new)
        b = lm.generate(ids, max_new)
    if a == b:
        return None
    return f"the same greedy reply came out differently twice ({len(a)} and {len(b)} tokens, first difference at " \
           f"token {next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))})"


def note(msg: str) -> None:
    print(f"wsr: {msg}", file=sys.stderr)
