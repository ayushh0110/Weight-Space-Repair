"""Deep-mode surgery: local edits, matched continuations, sites, controls, serving precision,
fresh variants, check-only, and hybrid (linear-attention) models."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deep_fixture import (LOOP_LAYER, LOOP_NEURON, PLANT_LAYER, PLANT_NEURON, planted_loop_model,  # noqa: E402
                          planted_model, tiny_qwen35)

from wsr.deep import contrast as C  # noqa: E402
from wsr.deep.lm import LM  # noqa: E402
from wsr.deep.run import diagnose  # noqa: E402
from wsr.surgery import edits as E  # noqa: E402
from wsr.surgery import pairs as PR  # noqa: E402
from wsr.surgery import playbook as P  # noqa: E402
from wsr.surgery import sites as S  # noqa: E402

FAILING = [{"role": "user", "content": "tell me about the cat please"}]
TOY_SUITE = [("story", "write a story about the dog", "en", "general"),
             ("sun", "what is the sun", "en", "general"),
             ("list", "list numbers", "en", "general")]


def _kl(a, b):
    pa = a.log_softmax(-1)
    return float((pa.exp() * (pa - b.log_softmax(-1))).sum())


@pytest.fixture()
def lm():
    m, t = planted_model()
    return LM(m, t, name="planted")


def test_project_is_its_hook_whole_matrix_and_column_mask(lm):
    ids = lm.encode(FAILING)
    g = torch.Generator().manual_seed(3)
    sites = [E.Site(1, "attn", torch.randn(lm.d_model, generator=g)),
             E.Site(2, "mlp", torch.randn(lm.d_model, generator=g), [5, 9, 17]),
             E.Site(3, "mlp", torch.randn(lm.d_model, generator=g))]
    for alpha in (1.0, 6.0):
        hook = lm.run(ids, capture=(), project_sites=(sites, alpha)).logits
        p = E.project(lm, sites, alpha)
        with E.applied(lm, p):
            w = lm.run(ids, capture=()).logits
        assert _kl(hook, w) < 1e-6
    # The masked site changes only its three columns.
    W0 = lm.down_proj(2).weight.detach().clone()
    with E.applied(lm, E.project(lm, sites[1:2], 2.0)):
        changed = (lm.down_proj(2).weight - W0).abs().sum(0) > 0
    assert sorted(torch.nonzero(changed)[:, 0].tolist()) == [5, 9, 17]


def test_tied_embeddings_patch_edits_the_embedding_not_the_unembedding(lm):
    m = lm.model
    m.lm_head.weight = m.model.embed_tokens.weight
    before = m.model.embed_tokens.weight.detach().clone()
    d = torch.randn(lm.d_model)
    E.apply(lm, E.orthogonalise(lm, d, 1.0))
    assert torch.equal(m.lm_head.weight, before)                       # output layer untouched
    u = d / d.norm()
    assert float((m.model.embed_tokens.weight @ u).abs().max()) < 1e-5  # embeddings no longer carry d


def test_directional_ablation_hook_removes_the_direction_from_the_embeddings_too(lm):
    ids = lm.encode(FAILING)
    e = lm.model.get_input_embeddings().weight[ids[0, -1]].detach().clone()
    r = lm.run(ids, capture=("resid",), ablate=e).resid
    assert float((r[0] @ (e / e.norm())).abs()) < 1e-5                  # depth 0 = layer 0's input


def test_control_is_the_same_size_elsewhere(lm):
    d = torch.randn(lm.d_model)
    for p in (E.orthogonalise(lm, d, 0.5), E.project(lm, [E.Site(2, "mlp", d, [1, 2, 3, 4])], 3.0)):
        c = E.control(lm, p, seed=1)
        assert len(c.deltas) == len(p.deltas)
        for a, b in zip(c.deltas, p.deltas):
            assert abs(a.norm() - b.norm()) < 1e-4 * max(1.0, b.norm())
        u = p.spec["direction"] if p.kind == "orthogonalise" else p.spec["directions"][0]
        r = c.spec["direction"] if c.kind == "orthogonalise" else c.spec["directions"][0]
        assert abs(float(u @ r)) < 1e-5
    pn = E.scale_neurons(lm, [(PLANT_LAYER, PLANT_NEURON), (PLANT_LAYER, 3)], 0.0)
    cn = E.control(lm, pn)
    got = {tuple(n) for n in cn.spec["neurons"]}
    assert len(got) == 2 and all(i == PLANT_LAYER for i, _ in got) and not got & {(PLANT_LAYER, PLANT_NEURON),
                                                                                  (PLANT_LAYER, 3)}


def test_patch_keeps_its_direction_and_checks_through_save(lm, tmp_path):
    p = E.project(lm, [E.Site(2, "mlp", torch.randn(lm.d_model), [4, 5])], 2.0)
    p.checks = {"mode": "deep", "status": "fixed"}
    p.save(tmp_path / "p")
    q = E.Patch.load(tmp_path / "p")
    assert torch.allclose(q.spec["directions"], p.spec["directions"]) and q.checks["mode"] == "deep"
    assert q.n_weights_changed == p.n_weights_changed == lm.d_model * 2
    E.control(lm, q)                                                     # a reloaded patch still has a control


def test_rounding_to_bf16_loses_a_tiny_edit_and_keeps_a_large_one(lm):
    d = torch.randn(lm.d_model)
    for alpha, lost_most in ((1e-6, True), (1.0, False)):
        p = E.project(lm, [E.Site(2, "mlp", d)], alpha)
        with E.applied(lm, p):
            lost = P._round_touched(lm, p, torch.bfloat16)
        assert (lost > 0.9) if lost_most else (lost < 0.05)


def test_sites_find_the_planted_neuron(lm):
    res = diagnose(lm, FAILING, max_new=12)
    sm = S.locate(lm, S.from_contrast(res.contrast))
    assert sm is not None
    assert any(s["layer"] == PLANT_LAYER and s["part"] == "mlp" for s in sm.sites)
    assert PLANT_NEURON in [j for j, _ in sm.neurons[PLANT_LAYER]][:3]


def test_fresh_variants_do_not_reuse_contrast_variants():
    msgs = [{"role": "user", "content": "Tell me about the cat, please."}]
    used = {C._last_user(m) for _, m in C.variants(msgs)}
    fresh = C.fresh_variants(msgs)
    assert len(fresh) >= 8
    assert not {C._last_user(m) for _, m in fresh} & used


def test_matched_continuations_and_sites_on_a_planted_loop():
    m, t = planted_loop_model()
    lm = LM(m, t, name="loop")
    c = C.build(lm, [{"role": "user", "content": "write a story about the tree"}], kind="loop", max_new=24)
    prs = PR.build(lm, c)
    assert prs, "no working continuation found"
    for p in prs:
        assert p.bad[:3] != p.good[:3]
    before = PR.selectivity(lm, prs)
    with E.applied(lm, E.scale_neurons(lm, [(LOOP_LAYER, LOOP_NEURON)], 0.0)):
        cmp = PR.compare(before, PR.selectivity(lm, prs))
    assert cmp["selective"] and cmp["repeat_delta"] < -0.5
    sm = S.locate(lm, PR.states(prs), source="continuations")
    assert sm is not None and sm.sites[0]["layer"] == LOOP_LAYER and sm.sites[0]["part"] == "mlp"
    assert LOOP_NEURON in [j for j, _ in sm.neurons[LOOP_LAYER]][:3]
    # The sparse edit on what the continuations found takes the loop's write out.
    with E.applied(lm, E.project(lm, sm.edit_sites(sparse=True), 1.0)):
        cmp = PR.compare(before, PR.selectivity(lm, prs))
    assert cmp["selective"]


def test_layer_screen_finds_the_planted_loop_layer_and_confirms_it_held_out():
    m, t = planted_loop_model()
    lm = LM(m, t, name="loop")
    c = C.build(lm, [{"role": "user", "content": "write a story about the tree"}], kind="loop", max_new=24)
    prs = PR.build(lm, c)
    assert len(prs) >= 4
    cd, info = P.screened_candidate(lm, prs)
    assert cd is not None, info.get("why")
    assert info["layers"] == [LOOP_LAYER]
    assert all(r["layer"] < lm.n_layers - 1 for r in info["screen"])      # the last layer is never screened
    h = info["held"]
    assert h["selective"] and h["repeat_delta"] < -0.5 and abs(h["working_delta"]) < 0.1
    before = PR.selectivity(lm, prs)
    with E.applied(lm, cd.ladder[0][1]()):                                  # alpha 1, all pairs
        assert PR.compare(before, PR.selectivity(lm, prs))["selective"]
    none, why = P.screened_candidate(lm, prs[:3])
    assert none is None and "4 matched pairs" in why["why"]


def test_deep_repair_fixes_the_planted_blank_and_reports_every_check(lm, tmp_path, monkeypatch):
    from wsr.surgery import regress as R
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    res = diagnose(lm, FAILING, max_new=12)
    res.contrast.failing = [e for e in res.contrast.failing if e.messages[-1]["content"].lower().count("please") == 1]
    rep = P.repair(lm, res, max_new=12, mode="deep", log=None)
    assert rep.status == "fixed", rep.summary
    assert rep.mode == "deep" and "deep checks" in rep.summary
    assert rep.trials[0]["scope"] == "component"                         # smallest edit first
    ck = rep.checks
    assert ck["serving"]["dtype"] == "bfloat16" and ck["serving"]["delta_lost"] < 0.05
    assert "fresh" in ck and "control" in ck and "verdict" in ck["control"]
    assert "sites" in ck
    P.write(rep, tmp_path / "out", lm)
    md = (tmp_path / "out" / "REPAIR.md").read_text(encoding="utf-8")
    assert "## Checks (deep mode)" in md and "Random control" in md
    # Check-only: the saved patch passes the same checks again.
    again = E.Patch.load(tmp_path / "out" / "patch")
    assert again.checks["mode"] == "deep"
    v = P.check(lm, res, again, max_new=12, mode="deep", log=None)
    assert v.status == "verified", v.summary


def test_fast_mode_has_no_local_candidates_or_fresh_check(lm, monkeypatch):
    from wsr.surgery import regress as R
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    res = diagnose(lm, FAILING, max_new=12)
    res.contrast.failing = [e for e in res.contrast.failing if e.messages[-1]["content"].lower().count("please") == 1]
    rep = P.repair(lm, res, max_new=12, log=None)
    assert rep.status == "fixed" and rep.mode == "fast"
    assert all(t["scope"] not in ("sparse", "local") for t in rep.trials)
    assert "fresh" not in rep.checks and "control" not in rep.checks and "serving" in rep.checks


# -- hybrid models ----------------------------------------------------------------------------------

def test_hybrid_model_heads_only_on_softmax_layers_and_edits_match_hooks():
    m, t = tiny_qwen35()
    lm = LM(m, t, name="q35")
    assert lm.head_layers == [1, 3]
    ids = lm.encode([{"role": "user", "content": "tell me about the cat"}])
    d = lm.model.get_input_embeddings().weight[t.convert_tokens_to_ids("cat")].detach().clone()
    hook = lm.run(ids, capture=(), ablate=d).logits
    with E.applied(lm, E.orthogonalise(lm, d, 1.0)):
        w = lm.run(ids, capture=()).logits
    assert _kl(hook, w) < 1e-6
    hs = [(1, 0), (3, 2)]
    h = lm.run(ids, capture=(), zero_heads=hs).logits
    with E.applied(lm, E.scale_heads(lm, hs, 0.0)):
        w = lm.run(ids, capture=()).logits
    assert _kl(h, w) < 1e-9
    with pytest.raises(ValueError, match="no attention heads"):
        lm.run(ids, capture=(), zero_heads=[(0, 0)])
    with pytest.raises(ValueError, match="no attention heads"):
        E.scale_heads(lm, [(2, 1)], 0.0)
    assert lm.attention(ids).shape[0] == lm.n_layers                     # linear layers padded with zeros
    assert float(lm.attention(ids)[0].abs().sum()) == 0.0


def test_gated_neuron_fires_only_above_its_threshold(lm):
    from wsr.surgery import sites as S
    ids = lm.encode(FAILING)
    L, j = 1, 3
    x = S.mlp_inputs(lm, L, ids)                                         # [T, d]
    r = x[-1] - x[:-1].mean(0)                                           # last position reads high
    r = r / r.norm()
    p = x @ r
    shift = float(p[:-1].max()) + 1e-3                                   # threshold-0 form: centre on the others
    # Threshold 0 form needs every off position below 0: use a read vector whose off projections are negative.
    if shift > 0:
        r = r - shift * x[:-1].mean(0) / float(x[:-1].mean(0) @ x[:-1].mean(0))
        r = r / r.norm()
        p = x @ r
    w = torch.randn(lm.d_model)
    patch = E.gate(lm, L, j, r, w, threshold=0.0, up_scale=float(p[-1]), strength=2.0)
    with E.applied(lm, patch):
        act = lm.seq_capture(ids, mlp_layers=[L])["mlp_act"][L][:, j]
    on = p > 0
    assert bool(on[-1])
    leak = torch.cat([act[~on].abs(), torch.zeros(1)]).max()
    assert float(leak) <= 0.3 * float(p.abs().max()) / float(p[-1])     # silu(z < 0) >= -0.28, times the up row
    assert float(act[-1]) > 0.5 * 8.0 * float(p[-1]) ** 2 / float(p[-1])   # ~ sharpness * p^2 / up_scale
    c = E.control(lm, patch)
    assert c.kind == "gate" and abs(float(c.spec["read"] @ patch.spec["read"])) < 1e-5


def test_a_failure_after_the_token_limit_is_inconclusive_not_weight_level(monkeypatch):
    """Llama-3.2-1B's Ukrainian loop starts after ~50 tokens; at max_new 48 the diagnosis said "not in the
    weights". When the cut reply fails further on, the verdict is inconclusive and names the --max-new to use;
    when it stays healthy, "not weight-level" stands (tests/test_deep.py)."""
    monkeypatch.setattr(C.Classifier, "groups_of",
                        lambda self, text, gen, **k: {"loop"} if len(gen) > 8 else set())   # fails only late
    m, t = planted_model()
    lm = LM(m, t, name="planted")
    res = diagnose(lm, [{"role": "user", "content": "write a story about the dog"}], max_new=2, probes=False)
    codes = {f.code for f in res.report.findings}
    assert "DEEP_CUT_SHORT" in codes and "DEEP_NOT_REPRODUCED" not in codes
    assert res.report.verdict["verdict"] == "inconclusive"
    why = P.not_indicated(res)
    assert why and "--max-new 128" in why[1]
    monkeypatch.setattr(C.Classifier, "groups_of", lambda self, *a, **k: set())             # healthy all along
    res = diagnose(lm, [{"role": "user", "content": "write a story about the dog"}], max_new=2, probes=False)
    assert res.report.verdict["verdict"] == "not_weight_level"
