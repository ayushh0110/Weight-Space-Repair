"""A patch belongs to the exact weights it was made on (wsr/identity.py): exact tensor hashes, every tensor, in
every supported precision; fail closed. These are the cases the reviews asked for: a permuted tensor with the
same aggregate statistics, a different fine-tune of the same shape, precision conversions, sharded checkpoints,
tied embeddings, the GGUF export's source, legacy patches and revert."""
import argparse
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deep_fixture import PLANT_LAYER, PLANT_NEURON, planted_model  # noqa: E402

from wsr import identity as ID  # noqa: E402
from wsr.deep.lm import LM  # noqa: E402
from wsr.surgery import edits as E  # noqa: E402
from wsr.surgery import export as X  # noqa: E402
from wsr.surgery.playbook import save_patched_model  # noqa: E402


def _save(model, tok, d, **kw):
    model.save_pretrained(d, safe_serialization=True, **kw)
    tok.save_pretrained(d)
    return d


def _model_dir(tmp_path, name="m", *, dtype=None, noise=0.0, shard=None, tied=False):
    m, t = planted_model()
    if tied:
        m.config.tie_word_embeddings = True
        m.lm_head.weight = m.model.embed_tokens.weight
    with torch.no_grad():
        if noise:
            g = torch.Generator().manual_seed(1)
            for p in m.parameters():
                p.add_(torch.randn(p.shape, generator=g) * p.abs().mean() * noise)
    if dtype is not None:
        m = m.to(dtype)
    return _save(m, t, tmp_path / name, **({"max_shard_size": shard} if shard else {}))


def _patched_dir(tmp_path, lm):
    patch = E.scale_neurons(lm, [(PLANT_LAYER, PLANT_NEURON)], 0.0)
    patch.save(tmp_path / "patch", lm)
    return tmp_path / "patch", E.Patch.load(tmp_path / "patch")


def test_a_saved_patch_carries_an_exact_identity(tmp_path):
    d = _model_dir(tmp_path)
    lm = LM.load(d, dtype="float32", device="cpu")
    pdir, patch = _patched_dir(tmp_path, lm)
    ident = json.loads((pdir / "identity.json").read_text(encoding="utf-8"))
    assert ident["model"]["source"] == "files" and set(ident["model"]["digests"]) == set(ID.PRECISIONS)
    assert all(set(v["hashes"]) == set(ID.PRECISIONS) for v in ident["tensors"].values())   # every tensor, 3 ways
    assert set(ident["patched"]) == {d.name for d in patch.deltas}
    assert "model.safetensors" in ident["model"]["files"] and ident["model"]["metadata"]["architecture"]
    assert E.identity_check(lm, patch)["verdict"] == "same"


def test_a_permuted_tensor_with_the_same_aggregate_statistics_is_refused(tmp_path):
    d = _model_dir(tmp_path)
    lm = LM.load(d, dtype="float32", device="cpu")
    _, patch = _patched_dir(tmp_path, lm)
    w = lm.model.model.layers[0].mlp.down_proj.weight
    with torch.no_grad():
        flat = w.view(-1)
        perm = flat[torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))].clone()
        assert torch.isclose((perm ** 2).sum(), (flat ** 2).sum()) and torch.isclose(perm.abs().sum(),
                                                                                     flat.abs().sum())
        flat.copy_(perm)
    r = E.identity_check(lm, patch)
    assert r["verdict"] == "different" and r["mismatched"] == ["model.layers.0.mlp.down_proj.weight"]


def test_a_different_finetune_of_the_same_shape_is_refused(tmp_path):
    a = LM.load(_model_dir(tmp_path, "a"), dtype="float32", device="cpu")
    _, patch = _patched_dir(tmp_path, a)
    b = LM.load(_model_dir(tmp_path, "b", noise=0.005), dtype="float32", device="cpu")   # every weight moved 0.5%
    r = E.identity_check(b, patch)
    assert r["verdict"] == "different" and r["n_mismatched"] == len(list(b.model.parameters()))


def test_supported_precision_conversions_match_exactly(tmp_path):
    d = _model_dir(tmp_path)
    lm = LM.load(d, dtype="float32", device="cpu")
    _, patch = _patched_dir(tmp_path, lm)
    for dt in ("bfloat16", "float16"):                                   # the same file, loaded in 16 bits
        assert E.identity_check(LM.load(d, dtype=dt, device="cpu"), patch)["verdict"] == "same", dt
    b16 = _model_dir(tmp_path, "b16", dtype=torch.bfloat16)                # the file itself converted to bf16
    for dt in ("bfloat16", "float32"):                                   # float32 holds the bf16 values exactly
        assert E.identity_check(LM.load(b16, dtype=dt, device="cpu"), patch)["verdict"] == "same", dt
    f16 = _model_dir(tmp_path, "f16", dtype=torch.float16)
    assert E.identity_check(LM.load(f16, dtype="float16", device="cpu"), patch)["verdict"] == "same"


def test_a_resharded_checkpoint_is_the_same_logical_model(tmp_path):
    one = _model_dir(tmp_path, "one")
    many = _model_dir(tmp_path, "many", shard="100KB")
    assert len(list(many.glob("*.safetensors"))) > 1
    m1, m2 = ID.from_files(one), ID.from_files(many)
    assert m1["digests"] == m2["digests"] and m1["files"] != m2["files"]       # same model, different files
    lm = LM.load(one, dtype="float32", device="cpu")
    _, patch = _patched_dir(tmp_path, lm)
    assert E.identity_check(LM.load(many, dtype="float32", device="cpu"), patch)["verdict"] == "same"


def test_tied_embeddings_are_one_tensor(tmp_path):
    d = _model_dir(tmp_path, tied=True)
    lm = LM.load(d, dtype="float32", device="cpu")
    assert E.tied(lm) and "lm_head.weight" not in ID.from_files(d)["tensors"]
    v = torch.randn(lm.d_model, generator=torch.Generator().manual_seed(0))
    patch = E.orthogonalise(lm, v, 0.5)                                  # edits the embedding: unties
    patch.save(tmp_path / "patch", lm)
    back = E.Patch.load(tmp_path / "patch")
    assert back.untie and E.identity_check(lm, back)["verdict"] == "same"


def test_the_gguf_export_refuses_a_wrong_source_checkpoint(tmp_path):
    a = _model_dir(tmp_path, "a")
    lm = LM.load(a, dtype="float32", device="cpu")
    pdir, _ = _patched_dir(tmp_path, lm)
    b = _model_dir(tmp_path, "b", noise=0.005)
    assert X.verify_source(a, pdir)["verdict"] == "same"
    with pytest.raises(ValueError, match="not made for these files"):
        X.verify_source(b, pdir)
    assert X.verify_source(b, pdir, allow_base_mismatch=True)["verdict"] == "different"
    assert X.patch_files(a, pdir, tmp_path / "out")                        # the right files still patch


def test_a_legacy_patch_needs_the_explicit_flag(tmp_path, capsys):
    from wsr.cli import _check_identity
    lm = LM.load(_model_dir(tmp_path), dtype="float32", device="cpu")
    patch = E.scale_neurons(lm, [(PLANT_LAYER, PLANT_NEURON)], 0.0)
    patch.save(tmp_path / "old")                                         # no model: no identity
    meta = json.loads((tmp_path / "old" / "patch.json").read_text(encoding="utf-8"))
    meta["base"] = {d.name: [1.0, 1.0] for d in patch.deltas}            # what an older wsr wrote
    (tmp_path / "old" / "patch.json").write_text(json.dumps(meta), encoding="utf-8")
    old = E.Patch.load(tmp_path / "old")
    r = E.identity_check(lm, old)
    assert old.legacy_base and r["verdict"] == "unverified" and "two aggregate sums" in r["why"]
    no = argparse.Namespace(allow_unverified_legacy=False, allow_base_mismatch=False)
    with pytest.raises(ValueError, match="--allow-unverified-legacy"):
        _check_identity(lm, old, no)
    with pytest.raises(ValueError, match="--allow-unverified-legacy"):
        X.verify_source(lm.path, tmp_path / "old")
    _check_identity(lm, old, argparse.Namespace(allow_unverified_legacy=True, allow_base_mismatch=False))
    assert "WARNING" in capsys.readouterr().err


def test_revert_checks_that_it_is_given_the_patched_model(tmp_path):
    d = _model_dir(tmp_path)
    lm = LM.load(d, dtype="float32", device="cpu")
    _, patch = _patched_dir(tmp_path, lm)
    save_patched_model(lm, patch, tmp_path / "patched")
    lp = LM.load(tmp_path / "patched", dtype="float32", device="cpu")
    assert E.identity_check(lp, patch, patched=True)["verdict"] == "same"
    assert E.identity_check(lp, patch)["verdict"] == "different"          # not the unpatched model any more
    assert E.identity_check(lm, patch, patched=True)["verdict"] == "different"


def test_patch_apply_on_the_wrong_model_exits_2(tmp_path, capsys):
    from wsr.cli import main
    lm = LM.load(_model_dir(tmp_path, "a"), dtype="float32", device="cpu")
    pdir, _ = _patched_dir(tmp_path, lm)
    b = _model_dir(tmp_path, "b", noise=0.005)
    assert main(["patch", "apply", str(b), str(pdir), "--out", str(tmp_path / "o")]) == 2
    assert "--allow-base-mismatch" in capsys.readouterr().err and not (tmp_path / "o").exists()
    assert main(["patch", "apply", str(lm.path), str(pdir), "--out", str(tmp_path / "ok")]) == 0
    rec = json.loads((tmp_path / "ok" / "wsr_patch.json").read_text(encoding="utf-8"))
    assert rec["identity_check"] == "same" and rec["allowed"] == []
