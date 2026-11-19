"""Check wsr on Apple's MPS (Metal) against the CPU, on this Mac, and write a report to share.

    python scripts/check_mps.py                                   # the self-test only (a tiny model, no download)
    python scripts/check_mps.py ./Qwen2.5-0.5B-Instruct           # and a real model, on the CPU and on MPS

wsr uses MPS only when asked (--device mps), after a self-test and a repeat check. This script measures more, on a
real model: how close MPS logits are to the CPU's, whether greedy replies match the CPU's and repeat exactly,
whether batched generation matches one prompt at a time, and the speed. Paste the printed summary (or attach the
JSON) in an "MPS report" issue: a few reports from different Macs decide whether `--device auto` may pick MPS.

`--device cuda` runs the same comparison on an NVIDIA GPU (how the script itself is tested off a Mac).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import wsr  # noqa: E402,F401  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch on a Mac)

PROMPTS = [
    "List five countries in Europe and their capitals.",
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about autumn leaves.",
    "What is 17 times 23? Show the steps.",
    "Give three tips for writing clear emails.",
    "Translate 'good morning, how are you?' into French and Spanish.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", nargs="?", help="a full-precision Hugging Face model folder (optional)")
    ap.add_argument("--device", default="mps", help="the device to compare with the CPU (default mps)")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--out", default="mps_report.json")
    args = ap.parse_args()

    import torch
    import transformers

    from wsr import macos
    from wsr.deep import device as D

    rep: dict = {"wsr": wsr.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                 "python": platform.python_version(), "machine": macos.describe() if macos.is_mac() else
                 platform.platform(), "device": args.device}
    if args.device.startswith("mps"):
        why = D.mps_problem()
        rep["mps_problem"] = why
        rep["mps_limit_bytes"] = D.mps_limit() if why is None else None
        if why:
            print(f"MPS cannot be used here: {why}")
            return _finish(rep, args.out)
    print(f"self-test on {args.device} ...")
    st = D.run_self_test(args.device)
    rep["self_test"] = st
    for c in st["checks"]:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['detail']}")
    if args.device.startswith("mps"):
        print(f"  bfloat16 on MPS: {'yes' if st.get('bfloat16') else 'no'}")
    if args.model_dir:
        rep["model"] = _model(args.model_dir, args.device, args.max_new)
    return _finish(rep, args.out)


def _model(path: str, device: str, max_new: int) -> dict:
    import torch

    from wsr.deep import device as D
    from wsr.deep.lm import LM

    print(f"\nloading {path} on the CPU and on {device} (float32) ...")
    cpu = LM.load(path, device="cpu", dtype="float32")
    t0 = time.time()
    gpu = LM.load(path, device=device, dtype="float32")
    out: dict = {"name": Path(path).name, "load_seconds": round(time.time() - t0, 1),
                 "ran_on": str(gpu.device)}
    if gpu.device.type != torch.device(device).type:
        print(f"  wsr moved the model to {gpu.device} (see the note above): nothing more to compare")
        out["fell_back"] = True
        return out
    ids = [cpu.encode([{"role": "user", "content": p}]) for p in PROMPTS]
    diffs, top1 = [], 0
    with torch.no_grad():
        for x in ids:
            a = cpu.model(input_ids=x).logits[0, -1].float()
            b = gpu.model(input_ids=x.to(gpu.device)).logits[0, -1].float().cpu()
            diffs.append(float((a - b).abs().max()))
            top1 += int(a.argmax()) == int(b.argmax())
    out["logits_max_diff"] = max(diffs)
    out["top1_agree"] = top1
    print(f"  last-position logits: largest difference {max(diffs):.3g}; same top token {top1}/{len(ids)}")

    def timed(lm, xs):
        t = time.time()
        r = [lm.generate(x, max_new) for x in xs]
        if lm.device.type == "mps":
            torch.mps.synchronize()
        n = sum(len(g) for g in r)
        return r, n / max(1e-9, time.time() - t)

    c_out, c_tps = timed(cpu, ids)
    g_out, g_tps = timed(gpu, ids)
    g_again, _ = timed(gpu, ids)
    same_cpu = sum(a == b for a, b in zip(c_out, g_out))
    repeat = sum(a == b for a, b in zip(g_out, g_again))
    batched = gpu.generate_batch(ids, max_new)
    same_batch = sum(a == b for a, b in zip(g_out, batched))
    out.update(prompts=len(ids), greedy_same_as_cpu=same_cpu, greedy_repeats=repeat,
               batched_same_as_single=same_batch, cpu_tokens_per_s=round(c_tps, 1), device_tokens_per_s=round(g_tps, 1))
    print(f"  greedy replies ({max_new} tokens): same as the CPU {same_cpu}/{len(ids)}, repeat exactly "
          f"{repeat}/{len(ids)}, batched same as one at a time {same_batch}/{len(ids)}")
    print(f"  speed: CPU {c_tps:.1f} tokens/s, {device} {g_tps:.1f} tokens/s")
    D.empty_cache(gpu.device)
    return out


def _finish(rep: dict, out: str) -> int:
    Path(out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    st = rep.get("self_test") or {}
    m = rep.get("model") or {}
    repeats = not m or m.get("fell_back") or m.get("greedy_repeats") == m.get("prompts")
    ok = bool(st.get("passed")) and bool(repeats) and not m.get("fell_back")
    print(f"\n{'PASS' if ok else 'NOT PASSED'}: report written to {out}. Please paste this summary, or attach the "
          "file, in an 'MPS report' issue: https://github.com/ayushh0110/Weight-Space-Repair/issues/new/choose")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
