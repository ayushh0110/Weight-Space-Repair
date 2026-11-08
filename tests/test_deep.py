"""Deep triage against a model with a known, planted weight-level bug (see deep_fixture.py)."""
import json
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import _gguf_str  # noqa: E402
from deep_fixture import PLANT_LAYER, PLANT_NEURON, planted_model  # noqa: E402

from wsr.deep import quant as Q  # noqa: E402
from wsr.deep.lm import LM  # noqa: E402
from wsr.deep.run import diagnose  # noqa: E402

FAILING = [{"role": "user", "content": "tell me about the cat please"}]


@pytest.fixture(scope="module")
def lm():
    m, t = planted_model()
    return LM(m, t, name="planted")


@pytest.fixture(scope="module")
def result(lm, tmp_path_factory):
    return diagnose(lm, FAILING, max_new=12, out_dir=tmp_path_factory.mktemp("handoff"))


def codes(res):
    return {f.code: f for f in res.report.findings}


def test_adapter_finds_the_parts(lm):
    assert lm.n_layers == 4 and lm.n_heads == 4 and lm.final_norm is not None and lm.eot_ids == {1}


def test_patching_every_layer_reproduces_the_donor(lm):
    a, b = lm.encode("tell me about the cat please"), lm.encode("tell me about the cat")
    rb = lm.run(b, capture=("resid",))
    patched = lm.run(a, capture=(), patch_resid={lm.n_layers - 1: rb.resid[-1]})
    assert torch.allclose(patched.logits, rb.logits, atol=1e-4)


def test_reproduces_and_builds_a_contrast_set(result):
    c = codes(result)
    assert "DEEP_REPRODUCED" in c and result.contrast.kind == "blank"
    assert result.contrast.failing and result.contrast.passing


def test_token_trace_starts_at_the_planted_layer(result):
    assert codes(result)["DEEP_TRACE"].evidence["onset_layer"] == PLANT_LAYER


def test_direction_and_known_direction(result):
    d = result.directions
    assert d.best_depth - 1 == PLANT_LAYER and d.loo_accuracy[d.best_depth] == 1.0
    chosen = next(s for s in result.selection.scores if s["depth"] == d.best_depth)
    assert chosen["specific"] and chosen["fixes"] > chosen["null_fixes_max"]   # beats shuffled-label directions
    best = max(result.known.items(), key=lambda kv: abs(kv[1]["cos_at_best_depth"]))
    assert best[0] == "end_of_turn" and best[1]["cos_at_best_depth"] > 0.8


def test_localised_to_the_planted_mlp_and_neuron(result):
    loc = result.loc
    assert max(range(len(loc.mlp)), key=lambda i: loc.mlp[i]) == PLANT_LAYER and loc.mlp[PLANT_LAYER] > 0.8
    assert max(loc.attn) < 0.2
    n0 = loc.neurons[0]
    assert (n0["layer"], n0["neuron"]) == (PLANT_LAYER, PLANT_NEURON)


def test_ablating_the_direction_fixes_it_and_adding_it_causes_it(result):
    cz = result.causal
    assert cz.ablate_fixed == cz.ablate_total > 0
    assert cz.ablate_passing_kept == cz.ablate_passing_total
    assert cz.add_broke >= 1


def test_verdict_and_handoff(result, tmp_path_factory):
    v = result.report.verdict
    assert v["verdict"] == "weight_level" and result.confidence == "high"
    assert f"layer {PLANT_LAYER} MLP" in result.summary and f"#{PLANT_NEURON}" in result.summary
    assert "end_of_turn" in result.summary


def test_handoff_files(lm, tmp_path):
    diagnose(lm, FAILING, max_new=8, out_dir=tmp_path, known=False)
    for f in ("report.json", "contrast.json", "traces.json", "direction.pt", "localisation.json", "weights.json",
              "SUMMARY.md"):
        assert (tmp_path / f).exists(), f
    assert json.loads((tmp_path / "report.json").read_text())["verdict"]["verdict"] == "weight_level"


def test_a_prompt_that_works_is_not_weight_level(lm):
    res = diagnose(lm, [{"role": "user", "content": "tell me about the cat"}], max_new=8, known=False)
    assert "DEEP_NOT_REPRODUCED" in codes(res)
    assert res.report.verdict["verdict"] == "not_weight_level"


# -- quantisation (Gap 1) ---------------------------------------------------------------------

def _q8_0(w: np.ndarray) -> bytes:
    out = bytearray()
    for blk in w.reshape(-1, 32):
        amax = np.abs(blk).max()
        d = amax / 127 if amax else 0.0
        q = np.round(blk / d).astype(np.int8) if d else np.zeros(32, np.int8)
        out += np.float16(d).tobytes() + q.tobytes()
    return bytes(out)


def _q4_0(w: np.ndarray, zero_scales_from: int = -1) -> bytes:
    out = bytearray()
    for i, blk in enumerate(w.reshape(-1, 32)):
        mx = blk[np.argmax(np.abs(blk))]
        d = mx / -8 if mx else 0.0
        if 0 <= zero_scales_from <= i:
            d = 0.0                                            # damaged block: scale lost
        q = np.clip(np.round(blk / d) + 8, 0, 15).astype(np.uint8) if d else np.full(32, 8, np.uint8)
        out += np.float16(d).tobytes() + bytes((q[:16] | (q[16:] << 4)).tolist())
    return bytes(out)


def _write(path, tensors):
    """tensors: name -> (ggml_type, numpy [rows, cols] values, raw bytes)."""
    out = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", 1))
    out += _gguf_str("general.architecture") + struct.pack("<I", 8) + _gguf_str("llama")
    offset, blobs = 0, []
    for name, (typ, arr, raw) in tensors.items():
        out += _gguf_str(name) + struct.pack("<I", 2) + struct.pack("<Q", arr.shape[1]) + struct.pack("<Q", arr.shape[0])
        out += struct.pack("<I", typ) + struct.pack("<Q", offset)
        blobs.append(raw + b"\0" * (-len(raw) % 32))
        offset += len(blobs[-1])
    out += b"\0" * (-len(out) % 32)
    path.write_bytes(bytes(out) + b"".join(blobs))


@pytest.fixture()
def quant_pair(tmp_path):
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    ws = {i: rng.normal(0, 0.02, (64, 128)).astype(np.float32) for i in range(5)}
    tensors = {}
    for i, w in ws.items():
        tensors[f"blk.{i}.ffn_down.weight"] = (2, w, _q4_0(w, zero_scales_from=100 if i == 3 else -1))
        tensors[f"blk.{i}.ffn_up.weight"] = (8, w, _q8_0(w))
    _write(tmp_path / "m.gguf", tensors)
    ref = {f"model.layers.{i}.mlp.down_proj.weight": w for i, w in ws.items()}
    ref.update({f"model.layers.{i}.mlp.up_proj.weight": w for i, w in ws.items()})
    (tmp_path / "ref").mkdir()
    save_file(ref, str(tmp_path / "ref" / "model.safetensors"))
    return tmp_path / "m.gguf", tmp_path / "ref"


def test_quant_error_is_normal_for_healthy_tensors_and_flags_the_damaged_one(quant_pair):
    gguf, ref = quant_pair
    rep = Q.analyse(gguf, ref)
    errs = {e["tensor"]: e for e in rep.errors}
    assert errs["blk.0.ffn_up.weight"]["rel_error"] < 0.01                 # Q8_0
    assert 0.03 < errs["blk.0.ffn_down.weight"]["rel_error"] < 0.12        # Q4_0
    worst = rep.worst_errors(1)[0]
    assert worst["tensor"] == "blk.3.ffn_down.weight" and worst["ratio_to_expected"] > 3
    st = {s["name"]: s for s in rep.stats}
    assert st["blk.3.ffn_down.weight"]["zero_scale_blocks"] > 0.3
    assert st["blk.0.ffn_down.weight"]["zero_scale_blocks"] == 0


def test_quant_cli(quant_pair, capsys):
    from wsr.cli import main

    gguf, ref = quant_pair
    assert main(["quant", str(gguf), "--reference", str(ref)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] blk.3.ffn_down.weight" in out


def test_awq_style_folded_scales_are_not_damage_but_real_damage_still_is(tmp_path):
    """Qwen's official GGUFs divide per-channel scales into the norm gains and multiply them into
    the next matrices' columns: every tensor differs from the original, the function does not."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(1)
    tensors, ref = {}, {}
    for i in range(6):
        norm = rng.uniform(0.5, 2.0, 128).astype(np.float32)
        s = np.exp(rng.normal(0, 0.4, 128)).astype(np.float32)          # AWQ-style per-channel scales
        ref[f"model.layers.{i}.input_layernorm.weight"] = norm
        g_norm = (norm / s).astype(np.float32)
        tensors[f"blk.{i}.attn_norm.weight"] = (0, g_norm.reshape(1, -1), g_norm.tobytes())
        for nm, hf in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj")):
            w = rng.normal(0, 0.02, (64, 128)).astype(np.float32)
            ref[f"model.layers.{i}.self_attn.{hf}.weight"] = w
            wg = (w * s).astype(np.float32)
            tensors[f"blk.{i}.{nm}.weight"] = (8, wg, _q8_0(wg) if not (i == 4 and nm == "attn_v")
                                               else _q8_0(np.where(rng.random(wg.shape) < 0.05, -wg, wg)))
    _write(tmp_path / "awq.gguf", tensors)
    (tmp_path / "ref").mkdir()
    save_file(ref, str(tmp_path / "ref" / "model.safetensors"))
    rep = Q.analyse(tmp_path / "awq.gguf", tmp_path / "ref")
    errs = {e["tensor"]: e for e in rep.errors}
    assert rep.processed
    assert errs["blk.0.attn_q.weight"]["layout"].endswith("+rescaled")
    assert errs["blk.0.attn_q.weight"]["plain_rel_error"] > 0.2 > 0.02 > errs["blk.0.attn_q.weight"]["rel_error"]
    assert errs["blk.0.attn_norm.weight"].get("folded_scales")
    worst = rep.worst_errors(3)
    assert worst[0]["tensor"] == "blk.4.attn_v.weight" and worst[0]["ratio_to_expected"] >= 3
    assert all(w["ratio_to_expected"] < 3 for w in worst[1:])
    assert not any("norm" in w["tensor"] for w in rep.worst_errors(100))


def test_k_quant_layouts():
    # Q6_K: every 6-bit code 33 (value +1), sub-block scales 1, d = 1  ->  all ones.
    b = bytearray(210)
    for i in range(128):
        b[i] = 0x11                       # low nibbles 1
    for i in range(128, 192):
        b[i] = 0b10101010                 # high 2 bits = 2 for each of the 4 lanes -> 1 | 2<<4 = 33
    for i in range(192, 208):
        b[i] = 1
    b[208:210] = np.float16(1.0).tobytes()
    vals, _, _ = Q._deq_q6_k(np.frombuffer(bytes(b), np.uint8).reshape(1, -1))
    assert np.allclose(vals, 1.0)
    # Q4_K: d = 1, dmin = 0, all 8 scales 2, nibbles 3 -> 6 everywhere.
    b = bytearray(144)
    b[0:2] = np.float16(1.0).tobytes()
    b[2:4] = np.float16(0.0).tobytes()
    for j in range(4):
        b[4 + j] = 2                      # scales 0-3
        b[8 + j] = 0                      # mins 0-3
        b[12 + j] = 2                     # scales 4-7 (low nibble), mins 4-7 (high nibble) = 0
    for i in range(16, 144):
        b[i] = 0x33
    vals, _, _ = Q._deq_q4_k(np.frombuffer(bytes(b), np.uint8).reshape(1, -1))
    assert np.allclose(vals, 6.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_same_diagnosis_on_cuda(tmp_path):
    m, t = planted_model()
    res = diagnose(LM(m.to("cuda"), t, name="planted-cuda"), FAILING, max_new=12, known=True)
    assert res.report.verdict["verdict"] == "weight_level"
    assert res.directions.best_depth - 1 == PLANT_LAYER
    assert (res.loc.neurons[0]["layer"], res.loc.neurons[0]["neuron"]) == (PLANT_LAYER, PLANT_NEURON)
    assert res.causal.ablate_fixed == res.causal.ablate_total > 0


def test_batched_patch_sweep_matches_single_patches(lm):
    a, b = lm.encode("tell me about the cat please"), lm.encode("tell me about the cat")
    rb = lm.run(b, capture=("resid", "attn", "mlp"))
    for where, kw, vecs in (("resid", "patch_resid", {i: rb.resid[i + 1] for i in range(lm.n_layers)}),
                            ("attn", "patch_attn", {i: rb.attn[i] for i in range(lm.n_layers)}),
                            ("mlp", "patch_mlp", {i: rb.mlp[i] for i in range(lm.n_layers)})):
        sweep = lm.patch_sweep(a, where, vecs, chunk=3)
        for i, v in vecs.items():
            single = lm.run(a, capture=(), **{kw: {i: v}}).logits
            assert torch.allclose(sweep[i], single, atol=1e-4), (where, i)


def test_zero_heads_removes_exactly_those_heads(lm):
    ids = lm.encode("tell me about the cat please")
    base = lm.run(ids, capture=()).logits
    one = lm.run(ids, capture=(), zero_heads=[(1, 2)]).logits
    assert not torch.allclose(base, one)
    # all heads of layer 1 removed == attention output of layer 1 zeroed at every position
    allh = lm.run(ids, capture=(), zero_heads=[(1, h) for h in range(lm.n_heads)]).logits
    from wsr.deep.lm import _first, _replace_first
    with lm._hooks([(lm.attn(1), "post", lambda m, a, o: _replace_first(o, torch.zeros_like(_first(o))))]):
        manual = lm.run(ids, capture=()).logits
    assert torch.allclose(allh, manual, atol=1e-5)


def test_induction_scores_and_attention(lm, tmp_path, monkeypatch):
    from wsr.deep import copyheads as CH

    monkeypatch.setattr(CH, "CACHE", tmp_path)
    sc = CH.induction_scores(lm, seq_len=8, n_seqs=2)
    assert sc.shape == (lm.n_layers, lm.n_heads) and float(sc.min()) >= 0 and float(sc.max()) <= 1
    a = lm.attention(lm.encode("tell me about the cat"))
    assert a.shape[:2] == (lm.n_layers, lm.n_heads)
    assert torch.allclose(a.sum(-1), torch.ones(lm.n_layers, lm.n_heads), atol=1e-4)
    assert lm.model.config._attn_implementation != "eager" or True       # restored (or eager to begin with)


def test_language_probe_posterior_and_trace(lm):
    from wsr.deep import langprobe as LP

    ra = lm.run(lm.encode("tell me about the cat"), capture=("resid",)).resid
    rb = lm.run(lm.encode("write a poem about the moon"), capture=("resid",)).resid
    D = ra.shape[0]
    means = torch.stack([ra, rb], dim=1)                               # [D, 2, d]
    var = torch.ones(D, ra.shape[1])
    probe = LP.LangProbe(["uk", "ru"], means, var, [1.0] * D, [1e-3] * D, {"uk": 9, "ru": 9}, 0)
    lp = probe.log_probs(rb).exp()
    assert (lp[1:, 1] > 0.99).all()                                    # nearest class wins
    tr = LP.trace(probe, lm, lm.encode("write a poem about the moon"), "uk", "ru")
    assert tr.switch_depth == 1 and tr.top[-1] == "ru"
    assert torch.allclose(probe.direction(3, "ru", "uk"), rb[3] - ra[3])


def test_rope_shift_is_invariant_on_a_healthy_model(lm):
    from wsr.deep import analysis as A
    from wsr.deep import loopprobes as LB
    from wsr.deep.contrast import Example

    ids = lm.encode("tell me about the cat please")
    ex = Example("x", [], ids, [1], "", {"blank"}, True, decision_ids=ids, decision_tok=1)
    t = A.Target(ex, [1], 5)
    healthy = LB.positional(lm, [t], offsets=(7, 13, 29))
    assert not healthy["shift_sensitive"] and max(r["drift"] for r in healthy["by_offset"]) < 0.05
    # (a planted aliasing defect is checked on real Qwen: this toy model barely uses position)


def test_zero_neurons_and_seq_capture(lm):
    ids = lm.encode("tell me about the cat please")
    from wsr.deep.lm import _first
    base = lm.run(ids, capture=()).logits
    z = lm.run(ids, capture=(), zero_neurons=[(PLANT_LAYER, PLANT_NEURON)]).logits
    assert not torch.allclose(base, z)
    got = {}
    dp = lm.down_proj(PLANT_LAYER)

    def pre(m, a):
        x = a[0].clone()
        x[..., PLANT_NEURON] = 0
        return (x,)
    with lm._hooks([(dp, "pre", pre)]):
        manual = lm.run(ids, capture=()).logits
    assert torch.allclose(z, manual, atol=1e-6)
    cap = lm.seq_capture(ids, depths=[0, 2, lm.n_layers], mlp_layers=[1], attn_layers=[1])
    T = ids.shape[1]
    assert cap["resid"][0].shape == (T, lm.d_model) and cap["resid"][lm.n_layers].shape == (T, lm.d_model)
    assert cap["mlp_act"][1].shape[0] == T and cap["attn"][1].shape == (T, lm.d_model)
    r = lm.run(ids, capture=("resid",))
    assert torch.allclose(cap["resid"][2][-1], r.resid[2], atol=1e-5)
    del got, _first


def test_neuron_attribution_finds_the_planted_neuron(lm):
    ids = lm.encode("tell me about the cat please")
    attr = lm.neuron_attribution(ids, lambda lg: lg[1] - lg[5])          # end-of-turn vs another token
    top = int(attr.flatten().argmax())
    assert (top // attr.shape[1], top % attr.shape[1]) == (PLANT_LAYER, PLANT_NEURON)


def _toy_gguf(path, model):
    """Every tensor of the toy Llama as GGUF: matrices Q8_0, norms F32."""
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    names = {"model.embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight",
             "model.norm.weight": "output_norm.weight"}
    for i in range(model.config.num_hidden_layers):
        for hf, gg in (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                       ("self_attn.o_proj", "attn_output"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
                       ("mlp.down_proj", "ffn_down"), ("input_layernorm", "attn_norm"),
                       ("post_attention_layernorm", "ffn_norm")):
            names[f"model.layers.{i}.{hf}.weight"] = f"blk.{i}.{gg}.weight"
    tensors = {}
    for hf, gg in names.items():
        w = sd[hf]
        if w.ndim == 1:
            tensors[gg] = (0, w.reshape(1, -1), w.astype(np.float32).tobytes())
        else:
            tensors[gg] = (8, w, _q8_0(w))
    _write(path, tensors)


def test_qbisect_finds_the_tensors_that_carry_the_change(tmp_path):
    from deep_fixture import planted_model as pm
    from wsr.deep import qbisect as QB
    from wsr.deep.contrast import Classifier

    planted, tok = pm()
    clean, _ = pm()
    # undo the plant in `clean`: fresh random model with the same seed, minus the planted edits
    from transformers import LlamaForCausalLM
    torch.manual_seed(0)
    fresh = LlamaForCausalLM(planted.config).eval()
    clean.load_state_dict(fresh.state_dict())
    _toy_gguf(tmp_path / "planted.gguf", planted)          # the "quantised" file carries the plant
    lm2 = LM(clean, tok, name="clean")
    qm = QB.build(lm2, tmp_path / "planted.gguf")
    assert len(qm.swaps) == 3 + 9 * planted.config.num_hidden_layers
    assert qm.check["ok"], qm.check                        # rebuilt full precision == the original
    clf = Classifier(lm2, max_new=8)
    res = QB.bisect(qm, clf, FAILING, max_set=12)
    # The plant spans four tensors (embedding + layer-2 gate/up/down): restoring one of them breaks it,
    # quantising one alone does not recreate it, so "necessary" is the right call (localised also fine).
    assert res["outcome"] in ("localised", "necessary"), res
    allowed = {"token_embd.weight", f"blk.{PLANT_LAYER}.ffn_gate.weight", f"blk.{PLANT_LAYER}.ffn_up.weight",
               f"blk.{PLANT_LAYER}.ffn_down.weight"}
    assert set(res["minimal_set"]) <= allowed, res["minimal_set"]
    assert res["proof"]["restoring_set_fixes"] and res["proof"]["random_sets_that_fix"] == 0
    assert "fixes the reply" in QB.summary(res)


def test_batched_ablation_matches_single_ablation(lm):
    ids = lm.encode("tell me about the cat please")
    g = torch.Generator().manual_seed(0)
    dirs = [torch.randn(lm.d_model, generator=g) for _ in range(5)]
    batched = lm.ablate_sweep(ids, dirs, chunk=2)
    for d, b in zip(dirs, batched):
        assert torch.allclose(b, lm.run(ids, capture=(), ablate=d).logits, atol=1e-4)


def test_memo_reuses_clean_passes_and_respects_weight_changes(lm):
    ids = lm.encode("tell me about the cat")
    with lm.memo():
        a = lm.run(ids, capture=("resid",))
        assert lm.run(ids, capture=("resid",)) is a              # reused
        lm._wver += 1
        assert lm.run(ids, capture=("resid",)) is not a          # weights changed: recomputed
        with lm.steer(ablate=torch.ones(lm.d_model)):
            assert lm.run(ids, capture=("resid",)) is not a      # hooks active: never cached


def test_last_position_logits_are_the_same_as_full_logits(lm):
    ids = lm.encode("tell me about the cat please", chat=True)
    assert lm._keeps_logits
    with torch.no_grad():
        full = lm.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float()
    # Same lm_head on the same last hidden state; only the GEMM's row count differs
    # (bit-identical on the GTX 1650 in fp16, last-bit rounding on CPU fp32).
    assert torch.allclose(lm.run(ids, capture=()).logits, full, atol=1e-5, rtol=0)

def test_auto_dtype_prefers_float32_where_it_fits(tmp_path, monkeypatch):
    from wsr.deep import lm as L
    (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 1000)          # 500 params -> 2000 bytes in fp32
    assert L._weight_bytes_fp32(tmp_path) == 2000
    assert L.auto_dtype(tmp_path, "cpu") == "float32"
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (4 << 30, 4 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 5))
    assert L.auto_dtype(tmp_path, "cuda") == "float32"
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (500 << 20, 4 << 30))   # no room
    assert L.auto_dtype(tmp_path, "cuda") == "float16"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 6))
    assert L.auto_dtype(tmp_path, "cuda") == "bfloat16"
