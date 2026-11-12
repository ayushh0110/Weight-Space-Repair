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


def test_target_does_not_demand_an_original_its_reference_run_cannot_show():
    """Only an example whose unedited reference run fails can count as fixed. When the original's own reference
    run does not fail (long greedy replies can diverge between batch layouts), "the original still fails" was
    reported for every edit; now it is not demanded, and the measurable prompts still decide."""
    from wsr.surgery.playbook import _target_ok
    base = {"fixed": 2, "of": 3, "passing_kept": 7, "passing": 7, "original_fixed": False}
    assert "the original prompt still fails" in _target_ok({**base, "original_measured": True})
    assert _target_ok({**base, "original_measured": False}) == []
    assert any("only 1/3" in p for p in _target_ok({**base, "fixed": 1, "original_measured": False}))
    assert any("no failing prompt" in p for p in _target_ok({**base, "fixed": 0, "of": 0, "original_measured": False}))


def test_loop_reference_is_rerun_longer_only_when_the_cap_hides_the_failure(monkeypatch):
    """Fewer than 2/3 of the failing examples loop again at the cap: rerun the reference at twice the limit and
    judge there, but only if more loop there; other kinds, and a reference that already agrees, stay as they are."""
    from types import SimpleNamespace as NS
    from wsr.surgery import playbook as PB

    class FakeClf:
        def __init__(self, lm=None, *, expect_lang=None, max_new=48, template_kwargs=None, expect_refuse=False):
            self.lm, self.expect_lang, self.max_new, self.template_kwargs = lm, expect_lang, max_new, template_kwargs or {}

        def run_many(self, jobs):                      # loops show up only from 1024 tokens on, for 3 of 4 examples
            return [NS(groups={"loop"} if (self.max_new >= 1024 and not n.startswith("f3")) else set())
                    for n, _ in jobs]
    monkeypatch.setattr(PB, "Classifier", FakeClf)
    fail = [NS(name=f"f{i}", messages=[], groups={"loop"}) for i in range(4)]
    keep = [NS(name="w0", messages=[], groups=set())]
    clf = FakeClf(max_new=512)
    agrees = PB._agreement("loop", fail, keep, PB.reference(clf, fail, keep))
    assert agrees["failing"] == 0
    got = PB._longer_reference("loop", clf, fail, keep, agrees, lambda s: None, "target")
    assert got is not None and got[0].max_new == 1024 and got[3]["failing"] == 3 and got[2]["max_new"] == 512
    named = [NS(name="original", messages=[], groups={"loop"})] + fail[1:]
    assert PB._agreement("loop", named, keep, PB.reference(got[0], named, keep))["original_fails"] is True
    assert PB._longer_reference("blank", clf, fail, keep, agrees, lambda s: None, "target") is None
    assert PB._longer_reference("loop", clf, fail, keep, {**agrees, "failing": 3}, lambda s: None, "t") is None
    never = FakeClf(max_new=512)
    never.run_many = lambda jobs: [NS(groups=set()) for _ in jobs]
    monkeypatch.setattr(PB, "Classifier", lambda *a, **k: never)
    assert PB._longer_reference("loop", clf, fail, keep, agrees, lambda s: None, "target") is None


def test_under_refusal_is_a_failure_kind_when_a_refusal_is_expected():
    """For requests the model must refuse ("complied" kind), a refusal is the wanted reply and an answer is the
    failure; without the expectation nothing changes."""
    from types import SimpleNamespace as NS
    from wsr.deep.contrast import Classifier, primary_kind
    lm = NS(eot_ids={0})
    refuse, comply = "I'm sorry, but I can't help with that request.", "Sure! Here are the steps you asked for."
    want = Classifier(lm, expect_refuse=True)
    assert want.groups_of(refuse, [5, 6]) == set()
    assert want.groups_of(comply, [5, 6]) == {"complied"}
    plain = Classifier(lm)
    assert "refusal" in plain.groups_of(refuse, [5, 6]) and "complied" not in plain.groups_of(comply, [5, 6])
    assert primary_kind({"complied"}) == "complied"


def test_playbook_steering_fixes_the_planted_blank_without_changing_weights(lm, tmp_path, monkeypatch):
    """steering="only": the playbook steers against the diagnosed direction, through the same checks as a weight
    edit; the result is "steered", no weight changes, and the vector, its settings and APPLY.md are written."""
    import torch
    from wsr.surgery import regress as R
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    res = diagnose(lm, FAILING, max_new=12)
    res.contrast.failing = [e for e in res.contrast.failing if e.messages[-1]["content"].lower().count("please") == 1]
    before = {n: p.detach().clone() for n, p in lm.model.named_parameters()}
    rep = P.repair(lm, res, max_new=12, log=None, steering="only")
    assert all(torch.equal(before[n], p) for n, p in lm.model.named_parameters())
    rows = [t for t in rep.trials if t.get("kind") == "steering"]
    assert rows and all(t["scope"] == "steering" for t in rows)
    assert not any(t.get("kind") != "steering" for t in rep.trials)              # "only": no weight edit tried
    # A constant shift moves healthy text too; the clamp acts only where the failure's direction is present,
    # at the planted neuron's layer, and a random direction of the same size fixes nothing.
    assert rep.status == "steered", rep.summary
    assert rep.patch is None and rep.steering.mode == "clamp" and rep.steering.layers == [PLANT_LAYER]
    assert rep.target["original_fixed"] and rep.target["passing_kept"] == rep.target["passing"]
    assert rep.checks["control"]["specific"] is True
    P.write(rep, tmp_path / "out", lm)
    assert (tmp_path / "out" / "steering" / "steer.safetensors").exists()
    assert "clamp" in (tmp_path / "out" / "steering" / "steering.json").read_text(encoding="utf-8")
    assert "Steering:" in (tmp_path / "out" / "REPAIR.md").read_text(encoding="utf-8")


def test_steering_is_only_a_fallback_when_a_weight_edit_passes(lm, monkeypatch):
    from wsr.surgery import regress as R
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    res = diagnose(lm, FAILING, max_new=12)
    res.contrast.failing = [e for e in res.contrast.failing if e.messages[-1]["content"].lower().count("please") == 1]
    rep = P.repair(lm, res, max_new=12, log=None)
    assert rep.status == "fixed" and not any(t.get("kind") == "steering" for t in rep.trials)
    with pytest.raises(ValueError):
        P.repair(lm, res, max_new=12, log=None, steering="sometimes")


def test_steering_starts_at_the_layer_the_diagnosis_measured_as_most_causal(lm, monkeypatch):
    """The diagnosis's activation patching already says which layer causes the failure; steering tries that layer
    alone first (no extra forward pass), and on the planted toy it is the planted layer."""
    from wsr.surgery import regress as R
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    res = diagnose(lm, FAILING, max_new=12)
    cl = P.causal_layers(res)
    assert cl and cl[0][0] == PLANT_LAYER and cl[0][1] > 0
    cands = P.steering_candidates(res, lm.n_layers)
    assert cands[0]["layers"] == [PLANT_LAYER] and "most causal" in cands[0]["placement"]
    assert abs(float(cands[0]["u"].norm()) - 1) < 1e-5


def test_with_refusals_kept_over_refusal_steering_first_drops_the_refusal_component(lm, monkeypatch):
    """Safety rule on: for over-refusal, the first candidates steer with the diagnosed direction minus its component
    along the known refusal direction, so steering cannot move the refusal of harmful requests."""
    import torch
    from types import SimpleNamespace as NS
    from wsr.deep import analysis as A
    from wsr.surgery import regress as R
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    res = diagnose(lm, FAILING, max_new=12)
    ref = torch.randn(lm.n_layers + 1, lm.d_model)
    res.known_vectors = A.KnownDirections({"refusal": ref})
    res.contrast = NS(**{**res.contrast.__dict__, "kind": "refusal"})
    safe = P.steering_candidates(res, lm.n_layers, keep_refusals=True)
    assert safe[0]["source"] == "the diagnosed direction without its refusal part"
    L = safe[0]["layers"][-1] + 1
    r = ref[L] / ref[L].norm()
    assert abs(float(safe[0]["u"] @ r)) < 1e-5                                    # no refusal component left
    loose = P.steering_candidates(res, lm.n_layers, keep_refusals=False)
    assert all(c["source"] != "the diagnosed direction without its refusal part" for c in loose)


# -- audit fixes -----------------------------------------------------------------------------------------

def test_a_patch_for_another_model_is_refused_whole(lm):
    """A missing tensor was a KeyError traceback, and a wrong shape failed half-way through apply."""
    before = {n: p.detach().clone() for n, p in lm.model.named_parameters()}
    p = E.scale_neurons(lm, [(0, 1), (1, 2)], 0.0)
    p.deltas[1].name = "model.layers.77.mlp.down_proj.weight"
    with pytest.raises(ValueError, match="made for another model"):
        E.apply(lm, p)
    q = E.scale_neurons(lm, [(0, 1), (1, 2)], 0.0)
    q.deltas[1].U = q.deltas[1].U[:-1]
    with pytest.raises(ValueError, match="made for another model"):
        with E.applied(lm, q):
            pass
    assert all(torch.equal(before[n], t) for n, t in lm.model.named_parameters())   # nothing applied in part


def test_steering_candidates_are_tried_once_each(lm, monkeypatch):
    res = diagnose(lm, FAILING, max_new=12)
    cands = P.steering_candidates(res, lm.n_layers)
    keys = [(c["source"], tuple(c["layers"]), c["sign"]) for c in cands]
    assert cands and len(keys) == len(set(keys))


def test_a_saved_patch_records_the_weights_it_was_made_on(lm, tmp_path):
    """A patch is a difference from exact weights. Applied to another fine-tune of the same shape it used to go
    through silently; now the saved patch carries marks of its base tensors and the CLI paths check them."""
    patch = E.scale_neurons(lm, [(0, 1)], 0.0)
    patch.save(tmp_path / "p", lm)
    back = E.Patch.load(tmp_path / "p")
    assert set(back.base) == {d.name for d in patch.deltas}
    assert E.base_check(lm, back)[0] == "same"
    w = E._param(lm, patch.deltas[0].name)
    keep = w.detach().clone()
    with torch.no_grad():
        w.mul_(1 + 1e-5)
    assert E.base_check(lm, back)[0] == "close"                 # as after loading in another precision
    with torch.no_grad():
        w.copy_(keep * 1.5)
    verdict, why = E.base_check(lm, back)
    assert verdict == "different" and patch.deltas[0].name in why
    from wsr.cli import _check_base
    with pytest.raises(ValueError, match="not made for this model"):
        _check_base(lm, back)
    _check_base(lm, back, force=True)                           # --force: a warning, no refusal
    with torch.no_grad():
        w.copy_(keep)
    assert E.base_check(lm, E.scale_neurons(lm, [(0, 1)], 0.0))[0] == "unknown"   # never saved with a model


def test_a_full_disk_while_saving_the_patched_model_is_a_clear_error(lm, tmp_path, monkeypatch):
    """safetensors raises its own error type for a full disk; `wsr patch apply` showed a traceback."""
    def full(*a, **k):
        raise RuntimeError('Error while serializing: IoError(Os { code: 112, kind: StorageFull, message: '
                           '"There is not enough space on the disk." })')
    monkeypatch.setattr(lm.model, "save_pretrained", full)
    before = {n: t.detach().clone() for n, t in lm.model.named_parameters()}
    with pytest.raises(OSError, match="disk filled up"):
        P.save_patched_model(lm, E.scale_neurons(lm, [(0, 1)], 0.0), tmp_path / "m")
    assert all(torch.equal(before[n], t) for n, t in lm.model.named_parameters())   # the patch is not left applied
