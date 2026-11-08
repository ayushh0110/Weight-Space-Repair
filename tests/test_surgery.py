"""Surgery: weight edits match the hooks they come from, restore exactly, and the playbook fixes a planted bug."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deep_fixture import PLANT_LAYER, PLANT_NEURON, planted_model  # noqa: E402

from wsr.deep.lm import LM  # noqa: E402
from wsr.deep.run import diagnose  # noqa: E402
from wsr.surgery import edits as E  # noqa: E402
from wsr.surgery import playbook as P  # noqa: E402

FAILING = [{"role": "user", "content": "tell me about the cat please"}]


@pytest.fixture()
def lm():
    m, t = planted_model()
    return LM(m, t, name="planted")


def _kl(a, b):
    pa = a.log_softmax(-1)
    return float((pa.exp() * (pa - b.log_softmax(-1))).sum())


def test_orthogonalising_the_weights_is_the_directional_ablation(lm):
    ids = lm.encode(FAILING)
    d = torch.randn(lm.d_model, generator=torch.Generator().manual_seed(1))
    hook = lm.run(ids, capture=(), ablate=d).logits
    with E.applied(lm, E.orthogonalise(lm, d, 1.0)):
        w = lm.run(ids, capture=()).logits
    assert _kl(hook, w) < 1e-6
    # every residual write is now orthogonal to d
    with E.applied(lm, E.orthogonalise(lm, d, 1.0)):
        r = lm.run(ids, capture=("resid",)).resid
    assert float((r @ (d / d.norm())).abs().max()) < 1e-4


def test_scaling_neurons_and_heads_is_zeroing_them(lm):
    ids = lm.encode(FAILING)
    ns, hs = [(PLANT_LAYER, PLANT_NEURON), (0, 3)], [(1, 0), (2, 1)]
    for patch, hook in ((E.scale_neurons(lm, ns, 0.0), dict(zero_neurons=ns)),
                        (E.scale_heads(lm, hs, 0.0), dict(zero_heads=hs))):
        h = lm.run(ids, capture=(), **hook).logits
        with E.applied(lm, patch):
            w = lm.run(ids, capture=()).logits
        assert _kl(h, w) < 1e-6


def test_applied_restores_bit_for_bit_and_patches_round_trip(lm, tmp_path):
    before = {n: p.detach().clone() for n, p in lm.model.named_parameters()}
    patch = E.orthogonalise(lm, torch.randn(lm.d_model), 0.5)
    with E.applied(lm, patch):
        assert any(not torch.equal(p, before[n]) for n, p in lm.model.named_parameters())
    assert all(torch.equal(p, before[n]) for n, p in lm.model.named_parameters())
    patch.save(tmp_path / "p")
    again = E.Patch.load(tmp_path / "p")
    assert again.label == patch.label and len(again.deltas) == len(patch.deltas)
    E.apply(lm, again)
    E.apply(lm, again, sign=-1.0)
    assert all(torch.allclose(p, before[n], atol=1e-6) for n, p in lm.model.named_parameters())


def test_tied_embeddings_are_untied_so_the_unembedding_stays(lm):
    m = lm.model
    m.lm_head.weight = m.model.embed_tokens.weight          # tie them
    assert E.tied(lm)
    ids = lm.encode(FAILING)
    d = torch.randn(lm.d_model)
    hook = lm.run(ids, capture=(), ablate=d).logits
    patch = E.orthogonalise(lm, d, 1.0)
    assert patch.untie
    with E.applied(lm, patch):
        w = lm.run(ids, capture=()).logits
    assert not E.tied(lm) and _kl(hook, w) < 1e-6


def test_playbook_is_honest_when_the_edit_fixes_too_few_prompts(lm):
    # Prompts with "please" twice still end after two words once the neuron is gone (the random
    # toy's own behaviour): 2 of 4 fixed is below the 2/3 bar, so no patch is claimed.
    res = diagnose(lm, FAILING, max_new=12)
    rep = P.repair(lm, res, max_new=12, log=None)
    first = rep.trials[0]
    assert first["candidate"].startswith(f"neuron {PLANT_LAYER}.{PLANT_NEURON}") and first["matches_hook_kl"] < 1e-6
    removed = first["steps"][-1]
    assert removed["screen"]["flipped"] == removed["screen"]["of"] and "fixes only" in removed["result"]
    if rep.status != "fixed":
        assert rep.status == "could_not_fix" and rep.next_step


# Toy prompts whose replies never stop on their own. On prompts the toy does end ("how was the big
# park"), the planted neuron is its only way to stop, so removing it makes them run on - and the
# regression check rightly reports that as a new loop.
TOY_SUITE = [("story", "write a story about the dog", "en", "general"),
             ("sun", "what is the sun", "en", "general"),
             ("list", "list numbers", "en", "general")]


def test_playbook_fixes_the_planted_blank_neuron(lm, tmp_path, monkeypatch):
    from wsr.surgery import regress as R
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)      # the toy's tokenizer knows only its own words
    res = diagnose(lm, FAILING, max_new=12)
    # The prompts where the trigger appears once: removing the planted neuron fixes all of them.
    res.contrast.failing = [e for e in res.contrast.failing if e.messages[-1]["content"].lower().count("please") == 1]
    rep = P.repair(lm, res, max_new=12, log=None)
    assert rep.status == "fixed", rep.summary
    assert rep.patch.kind == "scale_neurons" and rep.patch.spec["neurons"][0] == [PLANT_LAYER, PLANT_NEURON]
    assert rep.target["original_fixed"] and not rep.regression["new_failures"]
    P.write(rep, tmp_path / "out", lm)
    assert (tmp_path / "out" / "patch" / "patch.safetensors").exists()
    assert "fixed" in (tmp_path / "out" / "REPAIR.md").read_text(encoding="utf-8")


def test_surgery_not_indicated_for_a_greedy_only_loop():
    class Res:
        report = type("R", (), {"verdict": {"verdict": "sampling"}, "findings": []})()
        loop = {"supported": ["greedy decoding"]}
        contrast = type("C", (), {"kind": "loop"})()
    why = P.not_indicated(Res())
    assert why and "sampling" in why[0]


def test_a_tied_model_is_saved_tied_in_its_serving_precision(tmp_path):
    """Converting the state dict to fp16 used to copy the tied embedding twice, so the saved model (and
    its GGUF) came back with a separate, larger output matrix."""
    from deep_fixture import tiny_qwen35
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    m, t = tiny_qwen35(tied=True)
    lm = LM(m, t, name="tied")
    assert E.tied(lm)
    patch = E.scale_neurons(lm, [(0, 1)], 0.0)
    P.save_patched_model(lm, patch, tmp_path / "out", torch.float16)
    with safe_open(str(tmp_path / "out" / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert not any(k.startswith("lm_head") for k in keys), keys
    back = AutoModelForCausalLM.from_pretrained(str(tmp_path / "out"))
    assert back.lm_head.weight.data_ptr() == back.get_input_embeddings().weight.data_ptr()
