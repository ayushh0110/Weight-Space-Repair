"""The device module and the Mac support: CUDA and CPU behave as before (cache keys included), MPS is opt-in and
checked before use, and the Mac-only memory, disk, llama-server and MLX paths. MPS itself cannot run here (no
Mac, and GitHub's Macs have no usable GPU), so its functions are replaced by stand-ins; the checks the self-test
runs are exercised CPU against CPU."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deep_fixture import planted_model  # noqa: E402

from wsr import cache as C  # noqa: E402
from wsr import macos  # noqa: E402
from wsr.deep import batching as BT  # noqa: E402
from wsr.deep import device as D  # noqa: E402
from wsr.deep import lm as L  # noqa: E402
from wsr.deep.lm import LM  # noqa: E402


@pytest.fixture()
def toy(tmp_path):
    m, t = planted_model()
    m.save_pretrained(tmp_path / "toy")
    t.save_pretrained(tmp_path / "toy")
    return tmp_path / "toy"


# -- CUDA and CPU exactly as before ------------------------------------------------------------------------------

def _old_checksum(lm):
    """cache._weights_checksum as it was before MPS support: the cache keys of CPU and CUDA runs must not move."""
    import hashlib
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in lm.model.named_parameters():
            t = p.detach().reshape(-1)
            s = s2 = sw = 0.0
            for i in range(0, t.numel(), 1 << 22):
                c = t[i:i + (1 << 22)].double()
                w = torch.arange(i + 1, i + 1 + c.numel(), device=c.device, dtype=torch.float64)
                s += float(c.sum())
                s2 += float((c * c).sum())
                sw += float((c * w).sum())
            h.update(f"{name}|{tuple(p.shape)}|{p.dtype}|{s!r}|{s2!r}|{sw!r}\n".encode())
    return h.hexdigest()


def test_cpu_cache_keys_and_device_names_are_unchanged():
    m, t = planted_model()
    lm = LM(m, t, name="toy")
    assert C._weights_checksum(lm) == _old_checksum(lm)
    assert D.name("cpu") == "cpu" and D.name(torch.device("cpu")) == "cpu"
    assert not D.is_gpu("cpu") and D.is_gpu(torch.device("mps")) and D.kind("cuda:1") == "cuda"
    D.empty_cache("cpu")                                         # a no-op, never an error
    assert D.mark("cpu") is None and D.grown("cpu", None) is None


def test_cuda_memory_accounting_is_what_fit_measured_before(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 1000)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: 1700)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d: "GeForce GTX 1650")
    start = D.mark("cuda")
    assert start == 1000 and calls == ["reset"] and D.grown("cuda", start) == 700
    assert D.name("cuda") == "GeForce GTX 1650"


def test_auto_never_picks_mps(monkeypatch, toy):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    called = []
    monkeypatch.setattr(D, "mps_problem", lambda: called.append("checked"))
    lm = LM.load(toy, device="auto")
    assert lm.device.type == "cpu" and not called


# -- MPS: opt-in, checked first ----------------------------------------------------------------------------------

def test_mps_is_refused_with_the_reason_where_it_cannot_run(monkeypatch, toy):
    monkeypatch.setattr(macos, "is_mac", lambda: False)
    assert "needs a Mac" in D.mps_problem()
    with pytest.raises(ValueError, match="needs a Mac"):
        LM.load(toy, device="mps")
    monkeypatch.setattr(macos, "is_mac", lambda: True)
    monkeypatch.setattr(macos, "macos_version", lambda: (13, 6))
    assert "macOS 14 or later" in D.mps_problem() and "13.6" in D.mps_problem()


def test_a_failed_self_test_runs_on_the_cpu_and_says_why(monkeypatch, toy, capsys):
    monkeypatch.setattr(D, "mps_problem", lambda: None)
    monkeypatch.setattr(D, "self_test", lambda cache=True: {"passed": False, "why": "repeat: two greedy runs gave "
                                                                                     "different tokens"})
    lm = LM.load(toy, device="mps", dtype="float32")
    assert lm.device.type == "cpu"
    err = capsys.readouterr().err
    assert "MPS failed wsr's self-test" in err and "two greedy runs" in err and "on the CPU instead" in err


def test_the_self_test_checks_pass_cpu_against_cpu_and_name_every_check():
    res = D.run_self_test("cpu")
    assert res["passed"] and res["why"] is None and res["bfloat16"] is False
    assert [c["check"] for c in res["checks"]] == ["logits", "repeat", "batch", "restore", "fallback"]
    assert all(c["ok"] and c["detail"] for c in res["checks"])


def test_the_self_test_result_is_kept_per_machine_and_cleared_with_its_group(monkeypatch, tmp_path):
    monkeypatch.setenv("WSR_CACHE_DIR", str(tmp_path / "c"))
    runs = []
    monkeypatch.setattr(D, "run_self_test", lambda device="mps": runs.append(device) or
                        {"version": D.SELF_TEST_VERSION, "passed": True, "checks": [], "bfloat16": True, "why": None})
    monkeypatch.setattr(macos, "describe", lambda: "Apple M2, macOS 14.5")
    assert D.self_test()["passed"] and D.self_test()["machine"] == "Apple M2, macOS 14.5" and runs == ["mps"]
    files = C._group_paths("device")
    assert len(files) == 1 and json.loads(files[0].read_text())["bfloat16"] is True
    monkeypatch.setattr(macos, "describe", lambda: "Apple M2, macOS 15.1")            # another macOS: run again
    D.self_test()
    assert runs == ["mps", "mps"] and len(C._group_paths("device")) == 2
    assert "device" in C.GROUPS and set(C.targets(["device"])) == set(C._group_paths("device"))


def test_the_repeat_check_on_the_real_model(monkeypatch):
    m, t = planted_model()
    lm = LM(m, t, name="toy")
    assert D.repeat_problem(lm) is None                         # greedy on the CPU repeats
    outs = iter([[5, 6, 7], [5, 9, 7]])
    monkeypatch.setattr(lm, "generate", lambda ids, n: next(outs))
    assert "first difference at token 1" in D.repeat_problem(lm)


def test_mps_memory_is_the_smaller_of_metals_limit_and_the_shared_ram(monkeypatch):
    G = 2**30
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 10 * G, raising=False)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 3 * G, raising=False)
    monkeypatch.setattr(D, "_ram_available", lambda: 20 * G)
    assert D.mps_free_bytes() == 10 * G - 3 * G - D.RESERVE_MPS          # Metal's limit decides
    monkeypatch.setattr(D, "_ram_available", lambda: 5 * G)
    assert D.mps_free_bytes() == 5 * G - D.RESERVE_RAM                    # the shared RAM decides
    assert BT.free_bytes(torch.device("mps")) == D.mps_free_bytes()
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: (_ for _ in ()).throw(RuntimeError()),
                        raising=False)
    monkeypatch.setattr(macos, "gpu_limit_bytes", lambda ram=None: 12 * G)
    monkeypatch.setattr(D, "_ram_available", lambda: 64 * G)
    assert D.mps_free_bytes() == 12 * G - 3 * G - D.RESERVE_MPS           # older torch: macOS's own figure
    start = D.mark("mps")
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 4 * G, raising=False)
    assert start == 3 * G and D.grown("mps", start) == G


def test_mps_batches_to_the_gpu_cap_and_clears_its_cache(monkeypatch):
    cleared = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: cleared.append(1), raising=False)
    D.empty_cache("mps")
    assert cleared == [1]

    class Cfg:
        layer_types = None
        model_type = "llama"
        num_attention_heads = 4
        num_key_value_heads = 4
        head_dim = 16
        vocab_size = 100
        intermediate_size = 128

    class FakeLM:
        cfg, n_layers, d_model = Cfg(), 2, 64
        device = torch.device("mps")

        class model:
            @staticmethod
            def parameters():
                return iter([torch.zeros(1)])

    monkeypatch.setattr(BT, "free_bytes", lambda d: 100 * 2**30)
    assert BT.plan(FakeLM(), 10, 10, 100) == BT.cap()                     # not the CPU's cap of 8


def test_mps_precision_float32_when_it_fits_else_bfloat16_where_offered(monkeypatch, toy):
    monkeypatch.setattr(D, "mps_free_bytes", lambda: 64 * 2**30)
    assert L.auto_dtype(toy, "mps") == "float32"
    monkeypatch.setattr(D, "mps_free_bytes", lambda: 0)
    assert L._mps_dtype(toy, bf16=True) == "bfloat16" and L._mps_dtype(toy, bf16=False) == "float16"
    monkeypatch.setattr(D, "self_test", lambda cache=True: {"passed": True, "bfloat16": False})
    assert L.auto_dtype(toy, "mps") == "float16"


def test_the_cpu_fallback_for_missing_mps_ops_is_set_only_on_a_mac_before_torch():
    code = ("import os, sys; sys.platform = {plat!r}; os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None); "
            "import wsr; print(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))")
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"}
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    run = lambda plat: subprocess.run([sys.executable, "-c", code.format(plat=plat)], capture_output=True,  # noqa
                                      text=True, env=env).stdout.strip()
    assert run("darwin") == "1" and run("linux") == "None"


# -- the Mac, without MPS ----------------------------------------------------------------------------------------

def test_metals_share_of_the_unified_memory(monkeypatch):
    G = 2**30
    monkeypatch.setattr(macos, "apple_silicon", lambda: True)
    monkeypatch.setattr(macos, "sysctl", lambda name: {"iogpu.wired_limit_mb": "0"}.get(name))
    assert macos.gpu_limit_bytes(16 * G) == int(16 * G * 2 / 3)
    assert macos.gpu_limit_bytes(64 * G) == int(64 * G * 3 / 4)
    monkeypatch.setattr(macos, "sysctl", lambda name: {"iogpu.wired_limit_mb": "12288"}.get(name))
    assert macos.gpu_limit_bytes(16 * G) == 12 * G                        # the user's own limit wins
    monkeypatch.setattr(macos, "apple_silicon", lambda: False)
    assert macos.gpu_limit_bytes() is None


def test_triage_memory_fit_on_apple_silicon_uses_metals_share(monkeypatch):
    from wsr import config_checks as CC
    G = 2**30
    monkeypatch.setattr(macos, "apple_silicon", lambda: True)
    monkeypatch.setattr(macos, "gpu_limit_bytes", lambda ram=None: 10 * G)
    monkeypatch.setattr(CC, "_ram_bytes", lambda: 16 * G)
    spec = CC.ModelSpec("gguf", Path("m.gguf"), weights_bytes=11 * G, n_layers=2, n_kv_heads=1, head_dim=8,
                        ctx_trained=4096)
    f = CC.check_memory(spec)[0]
    assert f.code == "MEMORY_NO_FIT" and f.severity.value == "warn" and "Metal" in f.message   # fits the RAM only


def test_llama_server_on_a_mac_is_placed_against_metal_and_found_in_homebrew(monkeypatch, tmp_path):
    from wsr.program import backends as B
    G = 2**30
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(macos, "apple_silicon", lambda: True)
    monkeypatch.setattr(macos, "gpu_limit_bytes", lambda ram=None: 10 * G)
    monkeypatch.setattr(B, "_ram_free", lambda: 6 * G)
    assert B._vram_free() == 6 * G                                         # the RAM the GPU shares decides
    monkeypatch.setattr(macos, "apple_silicon", lambda: False)
    assert B._vram_free() is None                                          # elsewhere, as before
    exe = tmp_path / "bin" / "llama-server"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(B.shutil, "which", lambda n: None)
    monkeypatch.setattr(B, "MAC_BIN_DIRS", (str(exe.parent),))
    monkeypatch.setattr(macos, "is_mac", lambda: True)
    monkeypatch.setattr(B.os, "access", lambda p, m: True)
    assert B.find_llama_server() == exe
    monkeypatch.setattr(macos, "is_mac", lambda: False)
    assert B.find_llama_server() is None


def test_macos_swap_is_counted_only_as_far_as_the_disk_allows(monkeypatch):
    from types import SimpleNamespace as NS

    import psutil

    from wsr import resources as RS
    G = 2**30
    monkeypatch.setattr(psutil, "virtual_memory", lambda: NS(available=4 * G))
    monkeypatch.setattr(psutil, "swap_memory", lambda: NS(free=8 * G))
    monkeypatch.setattr(RS, "_windows_commit", lambda: None)
    monkeypatch.setattr(RS.shutil, "disk_usage", lambda p: NS(free=5 * G))
    monkeypatch.setattr(RS.sys, "platform", "linux")
    assert RS.headroom() == 12 * G                                         # Linux: as before
    monkeypatch.setattr(RS.sys, "platform", "darwin")
    assert RS.headroom() == 4 * G + (5 * G - RS.DISK_FLOOR)                # macOS: the disk bounds the swap
    monkeypatch.setattr(RS.shutil, "disk_usage", lambda p: NS(free=100 * G))
    assert RS.headroom() == 12 * G                                         # never more than before


def _mlx_dir(tmp_path, bits=4):
    d = tmp_path / f"mlx{bits}"
    d.mkdir()
    q = {"group_size": 64, "bits": bits}
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
                                               "num_hidden_layers": 2, "num_attention_heads": 2, "hidden_size": 64,
                                               "vocab_size": 100, "quantization": q, "quantization_config": q}))
    return d


def test_mlx_checkpoints_are_recognised_by_triage_and_refused_clearly_by_deep(tmp_path):
    from wsr import config_checks as CC
    from wsr.pipeline import _is_full_precision

    d = _mlx_dir(tmp_path)
    spec = CC.load_hf(d)
    assert spec.quant == "MLX 4-bit (group 64)"
    f = CC.check_quant(spec)
    assert f[0].code == "QUANT_INFO" and "MLX" in f[0].message
    assert not _is_full_precision(spec, d)
    low = CC.load_hf(_mlx_dir(tmp_path, bits=3))
    assert CC.check_quant(low)[0].code == "LOW_BIT_SMALL_MODEL"
    with pytest.raises(ValueError, match="MLX 4-bit"):
        LM.load(d, device="cpu")
    # Hugging Face's own quantised formats name quant_method and are not MLX; a plain model has no quant at all
    assert CC.mlx_quantization({"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}}) is None
    assert CC.mlx_quantization({}) is None
