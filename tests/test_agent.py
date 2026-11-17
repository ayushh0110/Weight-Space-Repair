"""The surgery agent: runs only after the playbook fails, acts only through the workbench, ships only
a verified edit, respects its budgets, and leaves the weights as they were."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deep_fixture import PLANT_LAYER, PLANT_NEURON, planted_model  # noqa: E402

from wsr.agent import AgentLimits, Budget, ScriptedProvider, brief, run  # noqa: E402
from wsr.agent.workbench import SCHEMAS, TOOLS, Workbench  # noqa: E402
from wsr.deep.lm import LM  # noqa: E402
from wsr.deep.run import diagnose  # noqa: E402
from wsr.surgery import playbook as P  # noqa: E402
from wsr.surgery import regress as R  # noqa: E402
from wsr.surgery import edits as E_  # noqa: E402

FAILING = [{"role": "user", "content": "tell me about the cat please"}]
TOY_SUITE = [("story", "write a story about the dog", "en", "general"),
             ("sun", "what is the sun", "en", "general"),
             ("list", "list numbers", "en", "general")]


@pytest.fixture()
def setup(monkeypatch):
    monkeypatch.setattr(R, "SUITE", TOY_SUITE)
    m, t = planted_model()
    lm = LM(m, t, name="planted")
    res = diagnose(lm, FAILING, max_new=12)
    res.contrast.failing = [e for e in res.contrast.failing if e.messages[-1]["content"].lower().count("please") == 1]
    # The playbook is made to fail (no candidates), so the agent is the one that has to find the edit.
    monkeypatch.setattr(P, "candidates", lambda *a, **k: [])
    rep = P.repair(lm, res, max_new=12, log=None, steering="off")     # its steering fallback would fix it
    assert rep.status == "could_not_fix" and rep.ctx is not None
    return lm, res, rep


def _weights(lm):
    return {n: p.detach().clone() for n, p in lm.model.named_parameters()}


def test_every_tool_has_a_schema():
    assert {s["name"] for s in SCHEMAS} == set(TOOLS) and "attribute_neurons" in TOOLS


def test_agent_finds_and_ships_a_verified_edit(setup, tmp_path):
    lm, res, rep = setup
    before = _weights(lm)
    last = lambda rs: json.loads(rs[-1])  # noqa: E731
    script = [
        [("list_examples", {})],
        [("make_direction", {"source": {"kind": "contrast"}})],
        [("inspect_neuron", {"layer": PLANT_LAYER, "neuron": PLANT_NEURON})],
        [("build_edit", {"spec": {"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0}})],
        [("screen_edit", lambda rs: {"edit_id": last(rs)["edit_id"]})],
        [("verify_edit", {"edit_id": "e1"})],
        [("submit_patch", {"edit_id": "e1", "rationale": "the neuron writes end-of-turn on the trigger"})],
    ]
    out = run(lm, res, rep, ScriptedProvider(script), out=tmp_path, log=None)
    assert out.status == "fixed", out.summary
    assert out.patch.kind == "scale_neurons" and "(agent)" in out.entry and "agent" in out.mode
    assert out.checks["agent"]["used"]["verifications"] == 1
    tr = json.loads((tmp_path / "agent_transcript.json").read_text(encoding="utf-8"))
    assert tr["summary"]["stopped"] == "submitted"
    assert all(torch.equal(p, before[n]) for n, p in lm.model.named_parameters())   # nothing left applied


def test_an_unverified_edit_cannot_be_submitted(setup):
    lm, res, rep = setup
    script = [[("build_edit", {"spec": {"kind": "scale_neurons", "neurons": [[0, 1]], "scale": 0}})],
              [("submit_patch", {"edit_id": "e1", "rationale": "trust me"})],
              [("give_up", {"reason": "nothing verified", "next_step": "fine-tune"})]]
    out = run(lm, res, rep, ScriptedProvider(script), log=None)
    assert out.status == "could_not_fix" and out.patch is None
    assert "nothing verified" in out.summary and out.next_step == "fine-tune"


def test_budgets_are_enforced_by_the_workbench(setup):
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap, Budget(verifications=1, generations=1))
    wb.build_edit({"kind": "scale_neurons", "neurons": [[0, 1]], "scale": 0.5})
    assert "error" not in wb.verify_edit("e1")
    assert "budget exhausted" in wb.verify_edit("e1")["error"]
    wb.generate("original")
    assert "budget exhausted" in wb.generate("original")["error"]
    assert "beyond the allowed" in wb.build_edit({"kind": "orthogonalise", "direction_id": "x", "alpha": 50})["error"]
    res_s, err = wb.call("build_edit", {"spec": {"kind": "project", "direction_id": "missing",
                                                 "sites": [{"layer": 1, "part": "mlp"}]}})
    assert err and "unknown direction" in res_s


def test_the_brief_carries_the_diagnosis_and_every_failed_trial(setup):
    lm, res, rep = setup
    rep.trials = [{"candidate": "contrast direction after layer 2", "scope": "global", "evidence": "separates",
                   "steps": [{"strength": "alpha 1", "result": "fixes it, but drift 0.4"}]}]
    text = brief(res, rep, Workbench(lm, res, rep.ctx, rep.sitemap))
    for want in ("## The failure", "## Contrast set", "contrast direction after layer 2", "fixes it, but drift 0.4",
                 "## Checks your patch must pass", "harmful request"):
        assert want in text


def test_the_agent_stops_at_the_turn_limit(setup):
    lm, res, rep = setup
    script = [[("list_examples", {})]] * 10
    out = run(lm, res, rep, ScriptedProvider(script), limits=AgentLimits(max_turns=3), log=None)
    assert out.status == "could_not_fix" and out.checks["agent"]["stopped"] == "turn limit"


def test_anthropic_provider_request_and_parsing(monkeypatch):
    pytest.importorskip("anthropic")
    from types import SimpleNamespace as NS

    from wsr.agent.providers import AnthropicProvider
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    p = AnthropicProvider()
    seen = {}

    def create(**kw):
        seen.update(kw)
        return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text="looking"),
                           NS(type="tool_use", id="tu1", name="list_examples", input={})],
                  stop_reason="tool_use",
                  usage=NS(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0,
                           cache_read_input_tokens=0))
    monkeypatch.setattr(p.client.beta.messages, "create", create)
    t = p.turn("sys", [{"role": "user", "content": "hi"}], SCHEMAS)
    assert seen["model"] == "claude-opus-5-5" and seen["thinking"] == {"type": "adaptive"}
    assert seen["output_config"] == {"effort": "high"} and seen["extra_body"] == {"fallbacks": "default"}
    assert "server-side-fallback-2026-07-01" in seen["betas"] and seen["cache_control"] == {"type": "ephemeral"}
    assert [c.name for c in t.calls] == ["list_examples"] and t.text == "looking" and t.stop == "tool_use"
    assert abs(t.cost_usd - (1000 * 4 + 200 * 20) / 1e6) < 1e-9


# -- the open workspace: the paper's kind of access, with the guards that keep it safe ------------------

def _down(layer):
    return f"model.layers.{layer}.mlp.down_proj.weight"


def test_agent_finds_the_neuron_from_the_weights_and_ships_a_free_form_edit(setup, tmp_path):
    """compute reads the weights, read measures activations, set_columns writes the agent's own values."""
    lm, res, rep = setup
    before = _weights(lm)
    eot = sorted(lm.eot_ids)[0]
    code = (f"W = param('{_down(PLANT_LAYER)}')\n"
            f"u = unembed_rows([{eot}])[0]\n"
            "score = (u @ W) / W.norm(dim=0)\n"
            "top = score.topk(3)\n"
            "cand = int(top.indices[0])\n"
            "cand")
    script = [
        [("compute", {"code": code, "why": "which neuron of the planted layer writes end-of-turn"})],
        [("read", {"items": ["original"], "capture": {"mlp_act": [PLANT_LAYER], "positions": "last"},
                   "store_as": "fail", "why": "does it fire on the failing prompt"})],
        [("compute", {"code": f"act = fail[0]['mlp_act'][{PLANT_LAYER}][0, cand]\n"
                              f"zero = torch.zeros({lm.d_model}, 1)\nfloat(act)", "why": "activation"})],
        [("build_edit", {"spec": {"kind": "set_columns", "param": _down(PLANT_LAYER), "cols": [PLANT_NEURON],
                                  "values": "zero", "why": "silence the end-of-turn writer"}})],
        [("verify_edit", {"edit_id": "e1"})],
        [("note", {"text": "planted neuron found from the weights alone"})],
        [("submit_patch", {"edit_id": "e1", "rationale": "the neuron's output column is the end-of-turn row"})],
    ]
    out = run(lm, res, rep, ScriptedProvider(script), out=tmp_path, log=None)
    tr = json.loads((tmp_path / "agent_transcript.json").read_text(encoding="utf-8"))
    tools = {t["name"]: t for t in tr["transcript"] if t.get("role") == "tool"}
    assert str(PLANT_NEURON) in tools["compute"]["result"]["output"].split("\n")[0]
    assert out.status == "fixed", out.summary
    assert out.patch.kind == "agent_set_columns" and out.patch.n_weights_changed == lm.d_model
    assert tr["summary"]["notes"] == ["planted neuron found from the weights alone"]
    assert all(torch.equal(p, before[n]) for n, p in lm.model.named_parameters())


def test_workspace_cannot_change_the_model_or_reach_python(setup):
    lm, res, rep = setup
    before = _weights(lm)
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    assert "error" not in wb.compute(f"W = param('{_down(1)}')\nW[:] = 0\nW.sum()", "a copy")
    for bad in ("import os", "open('x')", "().__class__", "W.add_(1)", "W.numpy()", "torch.save(W, 'x')",
                "getattr(W, 'data')", "f = lambda: 1", "while True:\n    pass", "type(W)",
                "torch.ones(100000)[:, None] * torch.ones(100000)[None, :]", "'{0.__class__}'.format(W)"):
        assert "error" in wb.compute(bad, "escape"), bad
    assert all(torch.equal(p, before[n]) for n, p in lm.model.named_parameters())


def test_free_form_edits_are_guarded(setup):
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    emb = next(n for n, p in lm.model.named_parameters() if p is lm.model.get_input_embeddings().weight)
    d, ff = lm.d_model, lm.down_proj(0).weight.shape[1]
    wb.compute(f"U = torch.randn({d}, 2) * 0.01\nV = torch.randn({ff}, 2)\nUe = torch.randn({lm.model.config.vocab_size}, 1)\n"
               f"Ve = torch.randn({d}, 1)\nW = param('{_down(1)}')\nUs = W * 0.5\nVs = torch.eye({ff})\n"
               f"Ub = torch.randn({d}, 2) * 1000", "inputs")
    assert "edit_id" in wb.build_edit({"kind": "delta", "changes": [{"param": _down(1), "U": "U", "V": "V"}]})
    assert "frozen" in wb.call("build_edit", {"spec": {"kind": "delta", "changes": [
        {"param": emb, "U": "Ue", "V": "Ve"}]}})[0]
    assert "rescales" in wb.call("build_edit", {"spec": {"kind": "delta", "changes": [
        {"param": _down(1), "U": "Us", "V": "Vs"}]}})[0]
    assert "norm" in wb.call("build_edit", {"spec": {"kind": "delta", "changes": [
        {"param": _down(1), "U": "Ub", "V": "V"}]}})[0]
    assert "shapes" in wb.call("build_edit", {"spec": {"kind": "delta", "changes": [
        {"param": _down(1), "U": "V", "V": "U"}]}})[0]


def test_reads_interventions_and_paired_sampling(setup):
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    item = {"prompt": "tell me about the cat please", "continuation": " the cat is"}
    a = wb.read([item], capture={"resid": [2], "positions": "continuation"}, store_as="a", why="base")
    b = wb.read([item], intervene={"zero_neurons": [[PLANT_LAYER, PLANT_NEURON]]}, why="ablated")
    assert a["texts"][0]["continuation_tokens"] >= 2 and "a" in wb.lab.vars
    assert wb.lab.vars["a"][0]["resid"][2].shape[1] == lm.d_model
    first = lambda r: r["texts"][0]["next_top5"][0][0]  # noqa: E731
    assert a["texts"][0]["continuation_mean_logprob"] != b["texts"][0]["continuation_mean_logprob"] or \
        first(a) != first(b)
    wb.build_edit({"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0})
    g = wb.generate("original", edit_id="e1", samples=2, temperature=0.8, compare=True, max_new=8)
    assert len(g["original"]) == 2 and len(g["edited"]) == 2 and g["generations_left"] == wb.budget.generations - 4


def test_own_pairs_and_the_layer_screen_on_a_planted_loop():
    from deep_fixture import LOOP_LAYER, planted_loop_model

    from wsr.deep import contrast as C
    from wsr.surgery import pairs as PR
    m, t = planted_loop_model()
    lm = LM(m, t, name="loop")
    c = C.build(lm, [{"role": "user", "content": "write a story about the tree"}], kind="loop", max_new=24)
    prs = PR.build(lm, c)
    items = [{"raw": lm.decode(p.prefix[0].tolist()), "bad": lm.decode(p.bad), "good": lm.decode(p.good)}
             for p in prs]
    from types import SimpleNamespace as NS
    wb = Workbench(lm, NS(contrast=c), NS(pairs=[], clf=NS(template_kwargs={}), sel0=None, base=NS(replies=[]),
                                          fresh=None))
    got = wb.make_pairs(items, name="mine")
    assert got["pairs"] == len(prs)
    sc = wb.screen_layers("mine")
    assert sc.get("layers") == [LOOP_LAYER], sc.get("why")
    assert set(sc["edit_ids"]) == {"alpha 1", "alpha 2", "alpha 3", "alpha 4", "alpha 6"}


def test_old_tool_results_are_shortened_when_the_context_grows():
    from wsr.agent.loop import _trim
    msgs = [{"role": "user", "content": "brief"}]
    for i in range(30):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "x"}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": str(i), "content": "y" * 5000}]})
    _trim(msgs, limit=50_000, keep_last=4)
    assert "shortened" in msgs[2]["content"][0]["content"] and len(msgs[-1]["content"][0]["content"]) == 5000


def test_a_missing_capability_is_requested_reported_and_never_applied(setup, tmp_path):
    """The agent's evidence says the fix needs the (frozen) embedding: it files a request with a proposal;
    the run reports it with the opt-in flag, saves the proposal unverified, and the model is untouched."""
    lm, res, rep = setup
    before = _weights(lm)
    emb = next(n for n, p in lm.model.named_parameters() if p is lm.model.get_input_embeddings().weight)
    vocab = lm.model.get_input_embeddings().weight.shape[0]
    script = [
        [("compute", {"code": f"U = torch.zeros({vocab}, 1)\nU[5, 0] = 0.1\nV = torch.ones({lm.d_model}, 1)",
                      "why": "the trigger token's embedding row"})],
        [("build_edit", {"spec": {"kind": "delta", "changes": [{"param": emb, "U": "U", "V": "V"}]}})],
        [("request_capability", {"capability": "edit the trigger token's embedding row",
                                 "category": "edit_embedding", "what_it_would_do": "move one row",
                                 "why_needed": "the failure is keyed on one input token",
                                 "evidence": "e.g. read r1", "proposal": {"kind": "delta", "changes": [
                                     {"param": emb, "U": "U", "V": "V"}]}})],
        [("request_capability", {"capability": "cap the reply length", "category": "config_or_runtime",
                                 "what_it_would_do": "max_tokens", "why_needed": "not a weight problem"})],
        [("give_up", {"reason": "needs the embedding", "next_step": "rerun with the unlock"})],
    ]
    out = run(lm, res, rep, ScriptedProvider(script), out=tmp_path, log=None)
    assert out.status == "could_not_fix"
    reqs = out.checks["agent"]["capability_requests"]
    assert [r["category"] for r in reqs] == ["edit_embedding", "config_or_runtime"]
    assert "--agent-allow embed" in reqs[0]["unlock"] and "unlock" not in reqs[1]
    assert "--agent-allow embed" in out.next_step
    saved = E_.Patch.load(tmp_path / "agent_proposals" / "proposal1")
    assert saved.checks["verified"] is False and saved.deltas[0].name == emb
    assert json.loads((tmp_path / "capability_requests.json").read_text(encoding="utf-8"))[0]["proposal"]
    P.write(out, tmp_path / "rep", lm)
    md = (tmp_path / "rep" / "REPAIR.md").read_text(encoding="utf-8")
    assert "What the agent needed but could not do" in md and "unverified proposal" in md
    assert all(torch.equal(p, before[n]) for n, p in lm.model.named_parameters())


def test_unlocks_are_opt_in_and_still_verified(setup):
    lm, res, rep = setup
    emb = next(n for n, p in lm.model.named_parameters() if p is lm.model.get_input_embeddings().weight)
    vocab = lm.model.get_input_embeddings().weight.shape[0]
    code = f"U = torch.zeros({vocab}, 1)\nU[5, 0] = 0.1\nV = torch.ones({lm.d_model}, 1)"
    locked = Workbench(lm, res, rep.ctx, rep.sitemap)
    locked.compute(code, "row")
    assert "frozen" in locked.call("build_edit", {"spec": {"kind": "delta", "changes": [
        {"param": emb, "U": "U", "V": "V"}]}})[0]
    both = ("embed", "unembed") if E_.tied(lm) else ("embed",)
    opened = Workbench(lm, res, rep.ctx, rep.sitemap, Budget(allow=both))
    opened.compute(code, "row")
    got = opened.build_edit({"kind": "delta", "changes": [{"param": emb, "U": "U", "V": "V"}]})
    assert "edit_id" in got
    assert "only an edit that passed verify_edit" in opened.submit_patch(got["edit_id"], "unverified")["error"]


# -- any provider: the OpenAI-compatible adapter ---------------------------------------------------------

class _FakeChat:
    """Stands in for openai.OpenAI().chat.completions: returns queued responses, records requests."""

    def __init__(self, replies):
        from types import SimpleNamespace as NS
        self.replies, self.seen = list(replies), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.seen.append(kw)
        return self.replies.pop(0)


def _reply(content=None, calls=(), finish="tool_calls", tin=100, tout=20, refusal=None):
    from types import SimpleNamespace as NS
    tcs = [NS(id=f"c{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(calls)]
    return NS(choices=[NS(message=NS(content=content, tool_calls=tcs or None, refusal=refusal),
                          finish_reason=finish)],
              usage=NS(prompt_tokens=tin, completion_tokens=tout))


def test_openai_compatible_request_translation_and_parsing():
    from wsr.agent.providers import OpenAIProvider
    fake = _FakeChat([_reply("looking", [("list_examples", "{}"), ("read", "not json")]),
                      _reply("done", finish="stop")])
    p = OpenAIProvider("some-model", client=fake, price_in=1.0, price_out=2.0)
    t = p.turn("sys", [{"role": "user", "content": "brief"}], SCHEMAS)
    kw = fake.seen[0]
    assert kw["model"] == "some-model" and kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["tools"][0]["type"] == "function" and {x["function"]["name"] for x in kw["tools"]} == set(TOOLS)
    assert "max_completion_tokens" in kw and "reasoning_effort" not in kw        # api.openai.com, no effort given
    assert [c.name for c in t.calls] == ["list_examples", "read"] and t.stop == "tool_use"
    assert t.calls[1].input == {"__invalid_json__": "not json"}
    assert abs(t.cost_usd - (100 * 1.0 + 20 * 2.0) / 1e6) < 1e-12
    # the neutral history (assistant tool_use, user tool_result) becomes assistant tool_calls + tool messages
    hist = [{"role": "user", "content": "brief"}, {"role": "assistant", "content": t.content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c0", "content": "{}"},
                                         {"type": "tool_result", "tool_use_id": "c1", "content": "bad",
                                          "is_error": True}]}]
    p.turn("sys", hist, SCHEMAS)
    msgs = fake.seen[1]["messages"]
    assert msgs[2]["role"] == "assistant" and [c["id"] for c in msgs[2]["tool_calls"]] == ["c0", "c1"]
    assert msgs[3] == {"role": "tool", "tool_call_id": "c0", "content": "{}"}
    assert msgs[4]["content"].startswith("ERROR: ")
    local = OpenAIProvider("local-model", client=_FakeChat([_reply(finish="stop")]), base_url="http://127.0.0.1:8080/v1",
                           effort="high")
    local.turn("sys", [{"role": "user", "content": "x"}], SCHEMAS)
    kw = local.client.seen[0]
    assert "max_tokens" in kw and "max_completion_tokens" not in kw and kw["reasoning_effort"] == "high"
    r = OpenAIProvider("m", client=_FakeChat([_reply(refusal="no", finish="stop")]))
    assert r.turn("s", [{"role": "user", "content": "x"}], SCHEMAS).stop == "refusal"


def test_openai_tool_call_extra_content_is_returned_unchanged():
    """Gemini 3's OpenAI endpoint puts a thought signature on the (first) tool call and rejects the next turn
    without it; whatever the server attached comes back on that call, and calls without it get nothing."""
    from wsr.agent.providers import OpenAIProvider
    rep = _reply(calls=[("list_examples", "{}"), ("read", "{}")])
    sig = {"google": {"thought_signature": "abc=="}}
    rep.choices[0].message.tool_calls[0].extra_content = sig
    fake = _FakeChat([rep, _reply("ok", finish="stop")])
    p = OpenAIProvider("gemini-x", client=fake, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    t = p.turn("sys", [{"role": "user", "content": "go"}], SCHEMAS)
    hist = [{"role": "user", "content": "go"}, {"role": "assistant", "content": t.content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c0", "content": "1"},
                                         {"type": "tool_result", "tool_use_id": "c1", "content": "2"}]}]
    p.turn("sys", hist, SCHEMAS)
    calls = fake.seen[1]["messages"][2]["tool_calls"]
    assert calls[0]["extra_content"] == sig and "extra_content" not in calls[1]


def test_openai_cost_counts_hidden_thinking_and_cached_price():
    """Gemini's endpoint reports thinking only in total_tokens (billed as output); cached input bills at the
    cache-read price when the user gives one, else at the full input price (never under-counted)."""
    from types import SimpleNamespace as NS
    from wsr.agent.providers import OpenAIProvider

    def rep():
        r = _reply("x", finish="stop", tin=1000, tout=10)
        r.usage = NS(prompt_tokens=1000, completion_tokens=10, total_tokens=1400,
                     prompt_tokens_details=NS(cached_tokens=800))
        return r
    full = OpenAIProvider("m", client=_FakeChat([rep()]), price_in=1.0, price_out=4.0)
    t = full.turn("s", [{"role": "user", "content": "x"}], SCHEMAS)
    assert t.output_tokens == 400 and abs(t.cost_usd - (1000 * 1.0 + 400 * 4.0) / 1e6) < 1e-12
    disc = OpenAIProvider("m", client=_FakeChat([rep()]), price_in=1.0, price_out=4.0, price_cached=0.1)
    t = disc.turn("s", [{"role": "user", "content": "x"}], SCHEMAS)
    assert abs(t.cost_usd - (200 * 1.0 + 800 * 0.1 + 400 * 4.0) / 1e6) < 1e-12 and disc.cached_tokens == 800


def test_the_agent_runs_the_same_through_an_openai_compatible_model(setup, tmp_path):
    """Provider-neutral: the same loop, tools and judge, driven through the OpenAI adapter."""
    from wsr.agent.providers import OpenAIProvider
    lm, res, rep = setup
    before = _weights(lm)
    spec = json.dumps({"spec": {"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0}})
    fake = _FakeChat([_reply(None, [("build_edit", spec)]),
                      _reply(None, [("verify_edit", json.dumps({"edit_id": "e1"}))]),
                      _reply("verified", [("submit_patch", json.dumps({"edit_id": "e1", "rationale": "planted"}))])])
    out = run(lm, res, rep, OpenAIProvider("any-model", client=fake), out=tmp_path, log=None)
    assert out.status == "fixed", out.summary
    assert out.checks["agent"]["provider"] == "openai-compatible:any-model" and out.checks["agent"]["tokens"] == 360
    assert all(torch.equal(p, before[n]) for n, p in lm.model.named_parameters())


def test_the_token_cap_holds_for_providers_without_prices(setup):
    from wsr.agent.providers import OpenAIProvider
    lm, res, rep = setup
    fake = _FakeChat([_reply(None, [("list_examples", "{}")], tin=900, tout=200) for _ in range(5)])
    out = run(lm, res, rep, OpenAIProvider("m", client=fake), limits=AgentLimits(max_tokens=2000), log=None)
    assert "token cap" in out.checks["agent"]["stopped"] and out.checks["agent"]["cost_usd"] == 0


def test_cli_agent_provider_fails_early_and_clearly(monkeypatch):
    import argparse

    from wsr.cli import _agent_provider
    a = argparse.Namespace(agent_provider="openai", agent_model=None, agent_base_url=None, agent_key_env=None,
                           agent_effort=None, agent_price=None)
    with pytest.raises(ValueError, match="--agent-model"):
        _agent_provider(a)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a.agent_model = "gpt-x"
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _agent_provider(a)
    a.agent_base_url = "http://127.0.0.1:8080/v1"                 # a local server needs no key
    assert _agent_provider(a).name == "openai-compatible:gpt-x@http://127.0.0.1:8080/v1"
    a2 = argparse.Namespace(agent_provider="anthropic", agent_model=None, agent_key_env=None, agent_effort="bogus",
                            agent_base_url=None, agent_price=None)
    with pytest.raises(ValueError, match="effort"):
        _agent_provider(a2)


# -- agent state: saved every turn, --resume continues, drafts come back first -----------------------------

def test_state_round_trip_keeps_workspace_registries_and_ids(setup, tmp_path):
    from wsr.agent import state as ST
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    wb.compute("a = torch.arange(6.0).reshape(2, 3)\nb = {1: a, 'k': [a[0], 3.5, 'x']}\nc = (1, 2)", "values")
    wb.make_direction({"kind": "contrast"})
    wb.build_edit({"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0})
    wb.note("neuron found")
    fp = ST.fingerprint(lm)
    ST.save(wb, tmp_path, fp=fp, history=[{"turns": 3}])
    wb2 = Workbench(lm, res, rep.ctx, rep.sitemap)
    st = ST.load(wb2, tmp_path, fp=fp)
    assert torch.equal(wb2.lab.vars["a"], torch.arange(6.0).reshape(2, 3))
    assert torch.equal(wb2.lab.vars["b"][1], wb2.lab.vars["a"]) and wb2.lab.vars["b"]["k"][1:] == [3.5, "x"]
    assert wb2.lab.vars["c"] == (1, 2) and "d1" in wb2.directions and wb2.notes == ["neuron found"]
    assert torch.equal(wb2.edits["e1"].deltas[0].U, wb.edits["e1"].deltas[0].U)
    assert wb2.build_edit({"kind": "scale_neurons", "neurons": [[0, 1]], "scale": 0})["edit_id"] == "e2"
    assert st["history"] == [{"turns": 3}] and "e1" not in wb2.verified     # passing does not carry over
    assert "e1" in ST.report(wb2, st, [])


def test_resume_refuses_different_weights(setup, tmp_path):
    from wsr.agent import state as ST
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    ST.save(wb, tmp_path, fp=ST.fingerprint(lm), history=[])
    with torch.no_grad():
        w = lm.down_proj(0).weight
        old = float(w[0, 0])
        w[0, 0] += 1e-3
    try:
        with pytest.raises(ValueError, match="different weights"):
            ST.load(Workbench(lm, res, rep.ctx, rep.sitemap), tmp_path, fp=ST.fingerprint(lm))
    finally:
        with torch.no_grad():
            w[0, 0] = old


def test_a_draft_that_needed_an_unlock_is_verified_and_shipped_on_resume(setup, tmp_path, monkeypatch):
    """Run 1: the fix the agent finds is over the size cap; it files a request with the draft and gives up.
    Run 2: the user grants the unlock and resumes; the draft is built, verified and submitted first, with no
    agent turn. (A draft that fails verification is refused the same way and the agent takes over.)"""
    lm, res, rep = setup
    before = _weights(lm)
    down = _down(PLANT_LAYER)
    draft = {"kind": "set_columns", "changes": [{"param": down, "cols": [PLANT_NEURON], "values": "zero"}]}
    run1 = [[("compute", {"code": f"zero = torch.zeros({lm.d_model}, 1)", "why": "silence the writer"})],
            [("build_edit", {"spec": dict(draft)})],
            [("request_capability", {"capability": "a larger edit", "category": "larger_edit",
                                     "what_it_would_do": "zero the end-of-turn writer",
                                     "why_needed": "the neuron's whole output is the failure", "proposal": draft})],
            [("give_up", {"reason": "over the size cap", "next_step": "allow larger edits"})]]
    tight = Budget(max_rel_change=1e-4)
    out = run(lm, res, rep, ScriptedProvider(run1), budget=tight, out=tmp_path, log=None)
    assert out.status == "could_not_fix" and (tmp_path / "agent_state" / "state.json").exists()
    assert out.checks["agent"]["capability_requests"][0]["proposal"]["id"] == "proposal1"
    from wsr.cli import _rerun_hint
    monkeypatch.setattr("sys.argv", ["wsr", "repair", "./m", "--agent", "--out", str(tmp_path)])
    hint = _rerun_hint(out, tmp_path)
    assert "--agent-allow large" in hint and "--resume" in hint
    out2 = run(lm, res, rep, ScriptedProvider([]), budget=Budget(max_rel_change=1e-4, allow=("large",)),
               out=tmp_path, resume=tmp_path, log=None)
    assert out2.status == "fixed", out2.summary
    ag = out2.checks["agent"]
    assert ag["turns"] == 0 and ag["stopped"] == "a resumed proposal passed verify_edit"
    assert ag["resumed_from"] == str(tmp_path) and len(ag["earlier_runs"]) == 1
    assert out2.patch.deltas[0].name == down and (tmp_path / "agent_transcript.run1.json").exists()
    assert all(torch.equal(p, before[n]) for n, p in lm.model.named_parameters())


def test_a_provider_failure_ends_the_run_cleanly_and_keeps_the_state(setup, tmp_path):
    """The local smoke test: a request timed out and the traceback ended wsr, report and all."""
    lm, res, rep = setup

    class Broken:
        name = "broken"
        calls = 0

        def turn(self, *a, **k):
            Broken.calls += 1
            raise TimeoutError("Request timed out.")
    out = run(lm, res, rep, Broken(), out=tmp_path, log=None, limits=AgentLimits(retry_waits=(0, 0)))
    stopped = out.checks["agent"]["stopped"]
    assert out.status == "could_not_fix" and "provider error: TimeoutError" in stopped
    assert Broken.calls == 3 and "still failing after 2 retries" in stopped      # a timeout is waited out first
    assert (tmp_path / "agent_state" / "state.json").exists()


def test_a_transient_provider_error_is_retried_and_the_run_goes_on(setup):
    """A Kiro run ended 70 minutes in on 'The model you've selected is temporarily unavailable': one outage
    should not end a run. The turn is sent again after a wait, and the run carries on from there."""
    from wsr.agent.loop import transient

    lm, res, rep = setup
    inner = ScriptedProvider([[("list_examples", {})], "done"])
    logs = []

    class Flaky:
        name = "flaky"
        failed = 0

        def turn(self, *a, **k):
            if Flaky.failed < 1:
                Flaky.failed += 1
                raise RuntimeError("kiro-cli exited 1: Error: Internal error (code -32603): The model you've "
                                   "selected is temporarily unavailable.")
            return inner.turn(*a, **k)
    out = run(lm, res, rep, Flaky(), log=logs.append, limits=AgentLimits(retry_waits=(0,)))
    ag = out.checks["agent"]
    assert Flaky.failed == 1 and ag["turns"] == 2 and "provider error" not in ag["stopped"]
    assert any("looks transient: retry 1 of 1" in s for s in logs)
    assert transient(TimeoutError()) and transient(ConnectionError())

    class Status(Exception):
        status_code = 529
    assert transient(Status("overloaded")) and transient(RuntimeError("429 Too Many Requests"))
    assert not transient(ValueError("invalid x-api-key")) and not transient(KeyError("tool"))


def test_a_permanent_provider_error_ends_the_run_without_waiting(setup):
    """A bad key or a malformed request will fail the same way again: no retry, no wait."""
    lm, res, rep = setup

    class Bad:
        name = "bad"
        calls = 0

        def turn(self, *a, **k):
            Bad.calls += 1
            raise ValueError("authentication failed: invalid x-api-key")
    out = run(lm, res, rep, Bad(), log=None, limits=AgentLimits(retry_waits=(3600.0,)))
    assert Bad.calls == 1 and out.checks["agent"]["stopped"] == ("provider error: ValueError: authentication "
                                                                 "failed: invalid x-api-key")


def test_a_turn_that_runs_out_of_output_is_nudged_to_act(setup):
    from wsr.agent.providers import OpenAIProvider
    lm, res, rep = setup
    fake = _FakeChat([_reply("thinking...", finish="length"),
                      _reply(None, [("give_up", json.dumps({"reason": "r", "next_step": "n"}))])])
    out = run(lm, res, rep, OpenAIProvider("m", client=fake, extra_body={"chat_template_kwargs": {"x": 1}}),
              log=None)
    assert out.checks["agent"]["stopped"] == "gave up"
    assert "ran out of output tokens" in fake.seen[1]["messages"][-1]["content"]
    assert fake.seen[0]["extra_body"] == {"chat_template_kwargs": {"x": 1}}


def test_tool_errors_that_a_small_model_hit_are_forgiving_and_explicit(setup):
    """From the local llama-server smoke test: a nested gate spec, a direction id as read_source, an
    orthogonalise over the cap with no reason, and no pointer from strong evidence to an edit."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    wb.make_direction({"kind": "contrast"})
    nested = wb.build_edit({"scale_neurons": {"neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0}})
    assert "edit_id" in nested
    bad = wb.build_edit({"gate": {"layer": 1, "read_source": "d1", "write_direction_id": "d1"}})
    assert "not a direction id" in bad["error"]
    assert "one of" in wb.build_edit({"what": 1})["error"]
    small = Workbench(lm, res, rep.ctx, rep.sitemap, Budget(max_free_params=10))
    small.make_direction({"kind": "contrast"})
    assert "embed" in small.build_edit({"kind": "orthogonalise", "direction_id": "d1"})["error"]
    # the size is the edit's free parameters: a rank-1 projection out of every layer touches every entry of
    # those matrices but chooses only rank x (rows + columns) numbers
    orth = wb.build_edit({"kind": "orthogonalise", "direction_id": "d1"})
    assert orth["free_params"] < orth["weights_changed"]
    p = wb._edit(orth["edit_id"])
    assert 0 < orth["free_params"] <= sum(d.U.shape[1] * (d.U.shape[0] + (d.V.shape[0] if d.V.shape[0] > 1 else 1))
                                          for d in p.deltas)          # untouched (all-zero) rows are not counted
    noop = wb.build_edit({"kind": "scale_neurons", "neurons": [[0, 0]], "scale": 1})
    assert "no weight" in noop.get("warning", "")
    hint = wb.inspect_neuron(PLANT_LAYER, PLANT_NEURON).get("hint", "")
    assert f"[[{PLANT_LAYER}, {PLANT_NEURON}]]" in hint and "verify_edit" in hint


def test_malformed_tool_input_gets_a_precise_answer(setup):
    """Third smoke run: neurons [[10, 0], 0.5], an extra 'id' argument, items without 'bad' - each answered
    with what is wrong and the expected shape, not a Python TypeError / KeyError."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    r, err = wb.call("build_edit", {"spec": {"scale_neurons": {"neurons": [[PLANT_LAYER, PLANT_NEURON], 0.5]}}})
    assert err and "[layer, neuron] pairs" in r and "example" in r
    r, err = wb.call("make_direction", {"source": {"kind": "contrast"}, "id": "x"})
    assert not err and json.loads(r)["ignored_arguments"] == ["id"]
    r, err = wb.call("make_pairs", {"items": [{"prompt": "p", "reply": ""}]})
    assert err and "use make_direction" in r
    r, err = wb.call("verify_edit", {})
    assert err and "needs edit_id" in r


def test_the_agent_is_told_its_budget_and_when_to_wrap_up():
    from wsr.agent.loop import AgentLimits, _budget_line

    class WB:
        def remaining(self):
            return {"verifications": 3, "screens": 10, "generations": 5}

    lim = AgentLimits(max_turns=24)
    first = _budget_line(WB(), lim, -1)
    assert "turns left 24 of 24" in first and "verifications left 3" in first and "Few turns" not in first
    late = _budget_line(WB(), lim, 20)
    assert "turns left 3" in late and "Few turns left" in late and "note" in late


def test_imports_of_provided_modules_are_no_ops_and_others_still_refused():
    from wsr.agent.lab import Lab, LabError

    lab = Lab({}, seconds=5)
    out = lab.run("import torch\nimport torch.nn.functional as F\nimport math\nx = F.relu(torch.ones(2) * math.pi)\nx")
    assert "3.14" in out["output"] and not lab.denied
    for code in ("import os", "from torch import load", "import numpy as np", "import torch as t"):
        with pytest.raises(LabError, match="already provided"):
            lab.run(code)
    assert lab.denied == {"syntax:Import": 4}


def test_prompt_caching_only_where_the_model_and_server_support_it():
    from wsr.agent.providers import anthropic_caches, openai_cache_control

    assert anthropic_caches("claude-opus-5-5") and anthropic_caches("claude-haiku-5-5")
    assert not anthropic_caches("claude-2.1") and not anthropic_caches("claude-instant-1.2")
    assert openai_cache_control("anthropic/claude-sonnet-5.5", "https://openrouter.ai/api/v1")
    assert openai_cache_control("google/gemini-3-pro", "https://openrouter.ai/api/v1")
    assert not openai_cache_control("nvidia/nemotron-3-ultra-550b-a55b:free", "https://openrouter.ai/api/v1")
    assert not openai_cache_control("gpt-5", None)                       # OpenAI caches by itself
    assert not openai_cache_control("anthropic/claude-x", "http://127.0.0.1:8080/v1")   # a local server


def test_openai_breakpoint_is_sent_only_when_supported_and_dropped_if_rejected():
    from types import SimpleNamespace as NS

    from wsr.agent.providers import OpenAIProvider

    sent = []

    class Completions:
        def create(self, **kw):
            sent.append(kw)
            msg = NS(content="ok", tool_calls=[], refusal=None)
            return NS(choices=[NS(message=msg, finish_reason="stop")],
                      usage=NS(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=NS(cached_tokens=7)))

    client = NS(chat=NS(completions=Completions()))
    p = OpenAIProvider("anthropic/claude-sonnet-5.5", base_url="https://openrouter.ai/api/v1", client=client)
    p.turn("SYSTEM", [{"role": "user", "content": "hi"}], [])
    assert sent[-1]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert p.cached_tokens == 7
    q = OpenAIProvider("some/model", base_url="https://openrouter.ai/api/v1", client=client)
    q.turn("SYSTEM", [{"role": "user", "content": "hi"}], [])
    assert sent[-1]["messages"][0] == {"role": "system", "content": "SYSTEM"}


def test_overwrite_edits_on_the_same_entries_are_not_combined(setup):
    """A gate (or set_columns / set_rows) is stored as new - old: two of them on the same entries, or one with
    another edit there, do not add up. Additive edits compose; overwrites on other entries compose."""
    import torch
    from wsr.surgery import edits as E

    def patch(kind, row, col):
        U = torch.zeros(4, 1)
        U[row, 0] = 1.0
        V = torch.zeros(3, 1)
        V[col, 0] = 1.0
        return E.Patch(f"{kind}@{row},{col}", kind, {}, [E.Delta("w", U, V)])
    assert "cannot combine" in E.overwrite_clash([patch("gate", 1, 2), patch("gate", 1, 2)])
    assert "cannot combine" in E.overwrite_clash([patch("gate", 1, 2), patch("projection", 1, 2)])
    assert E.overwrite_clash([patch("gate", 1, 2), patch("gate", 3, 2)]) == ""      # other rows
    assert E.overwrite_clash([patch("projection", 1, 2), patch("scale_neurons", 1, 2)]) == ""


def test_gate_sweep_reports_every_layer_and_builds_the_ladder_at_the_viable_ones(setup):
    """The playbook's gated-neuron search as an agent tool: a plan per layer, and at the best layers one edit
    per ladder strength, sized from the failure's gap (no layer or strength to guess)."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    wb.make_direction({"kind": "contrast"})
    out = wb.gate_sweep("d1")
    assert [r["layer"] for r in out["layers"]] == list(range(1, lm.n_layers - 1))
    assert wb.used["screens"] == 1
    # the toy's gates would fire on 2.5-3% of the positions that must stay unchanged: over the default 2%
    assert not out["viable"] and not out["edit_ids"] and "not readable" in out["says"]
    loose = Workbench(lm, res, rep.ctx, rep.sitemap, Budget(gate_max_off=0.05))
    loose.make_direction({"kind": "contrast"})
    out = loose.gate_sweep("d1", keep=2)
    assert out["viable"] and len({k.split()[2] for k in out["edit_ids"]}) <= 2
    assert all(int(k.split()[2].split(".")[0]) in out["viable"] for k in out["edit_ids"])
    assert {k.rsplit("x", 1)[1] for k in out["edit_ids"]} <= {"0.5", "1", "2", "4"}
    built = [v for v in out["edit_ids"].values() if v.startswith("e")]
    assert built and all(loose._edit(e).kind == "gate" for e in built)
    assert "decision_flips" in loose.screen_edit(built[0])
    one = loose.gate_sweep("d1", layers=[1], keep=1)
    assert [r["layer"] for r in one["layers"]] == [1]
    assert "error" in loose.gate_sweep("d1", read_source={"kind": "prompts", "failing": [], "working": []})


def test_scale_strength_bounds_and_budget_not_spent_on_a_bad_id(setup):
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    assert "beyond" in wb.build_edit({"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]],
                                      "scale": 1000})["error"]
    assert "beyond" in wb.build_edit({"kind": "gate", "layer": 1, "write_direction_id": "d1",
                                      "strength": 50})["error"]
    out, err = wb.call("screen_edit", {"edit_id": "e9"})
    assert err and wb.used["screens"] == 0
    out, err = wb.call("verify_edit", {"edit_id": "e9"})
    assert err and wb.used["verifications"] == 0


def test_generate_resolves_regression_suite_names(setup):
    """'suite:<name>' (as read and list_examples name them) runs that suite prompt, not the literal text."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    r = rep.ctx.base.replies[0]
    by_name = wb.generate(f"suite:{r.name}", max_new=8)
    by_text = wb.generate(r.prompt, max_new=8)
    assert by_name["original"]["reply"] == by_text["original"]["reply"]


def test_fit_edit_learns_a_change_that_passes_the_build_checks(setup):
    """fit_edit trains a low-rank change toward a target continuation, keeps the original weights untouched, and
    goes through the same checks as any edit; frozen tensors and out-of-range settings are refused."""
    import torch
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    before = {n: p.detach().clone() for n, p in lm.model.named_parameters()}
    name = f"model.layers.{PLANT_LAYER}.mlp.down_proj.weight"
    out = wb.fit_edit([name], [{"item": {"prompt": "original", "continuation": "the cat sat on the mat"},
                                "weight": 1.0}], rank=2, steps=30, lr=0.05)
    assert "edit_id" in out, out
    assert out["loss_end"]["target"] < out["loss_start"]["target"]
    assert out["free_params"] > 0 and wb.used["fits"] == 1
    assert all(torch.equal(before[n], p) for n, p in lm.model.named_parameters())          # nothing written
    assert all(p.requires_grad for p in lm.model.parameters()) or not any(
        p.requires_grad for p in before.values())                                          # flags restored
    emb = next(n for n, p in lm.model.named_parameters() if p is lm.model.get_input_embeddings().weight)
    assert "frozen" in wb.fit_edit([emb], [{"item": "original", "weight": -1}])["error"]
    assert "rank" in wb.fit_edit([name], [{"item": "original", "weight": -1}], rank=999)["error"]
    assert "non-zero" in wb.fit_edit([name], [{"item": "original", "weight": 0}])["error"]
    assert wb.used["fits"] == 1                                                             # refusals cost nothing
    scr = wb.screen_edit(out["edit_id"])
    assert "decision_flips" in scr


def test_attribute_heads_ranks_heads(setup):
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    out = wb.attribute_heads(top=5)
    assert len(out["heads"]) == 5 and all(len(h) == 3 for h in out["heads"]) and out["layers"]


def test_fit_edit_out_of_memory_ends_cleanly(setup, monkeypatch):
    """Running out of GPU memory mid-fit is a clear refusal: no exception escapes, the weights and their
    requires_grad flags are as before, and the budget is still charged only once."""
    import torch
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    before = {n: p.detach().clone() for n, p in lm.model.named_parameters()}
    flags = [p.requires_grad for p in lm.model.parameters()]
    calls = {"n": 0}
    real = lm.model.forward

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] > 3:                                # after the keep references and the first losses
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(*a, **k)
    monkeypatch.setattr(lm.model, "forward", boom)
    name = f"model.layers.{PLANT_LAYER}.mlp.down_proj.weight"
    out = wb.fit_edit([name], [{"item": {"prompt": "original", "continuation": "the cat sat"}, "weight": 1.0}],
                      rank=2, steps=5)
    assert "out of GPU memory" in out["error"]
    assert all(torch.equal(before[n], p) for n, p in lm.model.named_parameters())
    assert [p.requires_grad for p in lm.model.parameters()] == flags
    assert wb.used["fits"] == 1


def test_generate_many_prompts_in_one_batched_call(setup):
    """prompts=[...] gives the same replies as one call per prompt (greedy), for the original and the edit, and
    costs the same generations budget."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    e = wb.build_edit({"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0})["edit_id"]
    names = ["original", "tell me about the dog", "what is the sun"]
    many = wb.generate(prompts=names, edit_id=e, compare=True, max_new=8)
    assert len(many["results"]) == 3 and wb.used["generations"] == 6
    for name, r in zip(names, many["results"]):
        one = wb.generate(name, edit_id=e, compare=True, max_new=8)
        assert r["original"]["reply"] == one["original"]["reply"] and r["edited"]["reply"] == one["edited"]["reply"]
    assert "error" in wb.generate(prompts=[])
    assert "error" in wb.generate()


# -- audit fixes -----------------------------------------------------------------------------------------

def test_a_long_tool_result_is_still_valid_json_and_a_tool_fault_does_not_end_the_run(setup, tmp_path, monkeypatch):
    """A read of 64 texts was cut mid-JSON at 20,000 characters and the loop's json.loads raised; a fault inside
    a tool (AttributeError) escaped the dispatcher. Either ended a paid agent run with a traceback."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    items = [{"prompt": f"what is the sun number {i}", "continuation": " the sun is a star" * 6} for i in range(64)]
    s, err = wb.call("read", {"items": items})
    assert not err and json.loads(s)["stored_as"] and len(s) <= 20_000
    monkeypatch.setattr(wb, "tools", {**TOOLS, "note": lambda self, text: {"blob": "x" * 50_000}}, raising=False)
    s, err = wb.call("note", {"text": "t"})
    cut = json.loads(s)                                               # cut, and still JSON
    assert "truncated" in cut and len(s) < 30_000
    s, err = wb.call("logit_lens", {"example": "suite:sun", "depth": 1})   # a suite reply has ids, not prompt_ids
    assert not err and json.loads(s)["top"]
    assert "depth must be" in json.loads(wb.call("logit_lens", {"example": "original", "depth": 99})[0])["error"]

    def broken(self, text):
        raise AttributeError("'NoneType' object has no attribute 'to'")
    monkeypatch.setattr(wb, "tools", {**TOOLS, "note": broken}, raising=False)
    s, err = wb.call("note", {"text": "t"})
    assert err and "failed inside wsr" in json.loads(s)["error"]
    before = _weights(lm)
    script = [[("read", {"items": items})], [("give_up", {"reason": "done", "next_step": "none"})]]
    out = run(lm, res, rep, ScriptedProvider(script), out=tmp_path, log=None)
    assert out.status == "could_not_fix" and "done" in out.summary
    assert all(torch.equal(p, before[n]) for n, p in lm.model.named_parameters())


def test_workspace_refuses_huge_results_before_computing_them(setup, monkeypatch):
    """The size cap was checked on the result, after the allocation: torch.zeros(3e9) tried to take 12 GB first,
    and an integer power tower ran for minutes. Both are refused from their arguments now."""
    import time

    from wsr.agent import lab as L
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    t0 = time.perf_counter()
    for code, why in (("x = torch.zeros(3000000000)", "would produce"),
                      ("x = torch.randn(60000, 60000)", "would produce"),
                      ("a = torch.ones(100000)\nx = torch.outer(a, a)", "would produce"),
                      ("a = torch.ones(100000)\nx = torch.diag(a)", "would produce"),
                      ("a = torch.ones(100000)\nx = a.outer(a)", "would produce"),
                      ("a = torch.ones(9000, 9000)\nx = torch.cat([a, a])", "would produce"),
                      ("a = 10**10000\nb = a**10000", "integer too large"),
                      ("a = 2**90000\nb = a * a", "integer too large"),
                      ("1 << 4", "not available")):
        out = wb.compute(code)
        assert why in out.get("error", ""), (code, out)
    assert time.perf_counter() - t0 < 20                              # refused, not computed
    # the reduced SVD by default: the full one of a tall matrix allocates rows x rows
    out = wb.compute("U, S, Vh = torch.linalg.svd(torch.randn(50, 4))\nU.shape")
    assert "50, 4" in out["output"], out
    assert "would produce" in wb.compute("x = torch.linalg.svd(torch.ones(9000, 2), full_matrices=True)")["error"]
    # the workspace as a whole is capped: a call that would pass it keeps nothing, earlier variables stay
    monkeypatch.setattr(L, "MAX_WORKSPACE", 1000)
    assert "error" not in wb.compute("small = torch.ones(600)")
    out = wb.compute("big = torch.ones(600)")
    assert "workspace would hold" in out["error"] and "big" not in wb.lab.vars and "small" in wb.lab.vars
    with pytest.raises(L.LabError, match="workspace"):
        wb.lab.put("r9", [torch.ones(900)])
    assert wb.lab.denied.get("limit:workspace", 0) >= 2


def test_workspace_limits_hold_for_augmented_assignment_sequences_and_unestimated_calls(setup):
    """An outside review found ways past the caps: `z **= 10000` skipped the integer check, strings and lists
    grew without limit (s = s + s, l.extend(l), m *= 50_000_000), and einsum / bincount allocated gigabytes
    before their result was checked. Each is refused now, quickly, and ordinary code still runs."""
    import time

    from wsr.agent.lab import MAX_SEQ, Lab, LabError
    lab = Lab({}, seconds=10)
    t0 = time.perf_counter()
    for code, why in (("z = 10\nz **= 10000\nz **= 3000", "integer too large"),
                      ("m = [0] * 1000\nm *= 50000", "too long"),
                      ("s = 'ab'\nfor i in range(40):\n    s = s + s", "over the cap"),
                      ("s = 'ab'\nfor i in range(40):\n    s = f'{s}{s}'", "over the cap"),
                      ("l = [0]\nfor i in range(40):\n    l.extend(l)", "over the cap"),
                      ("l = [0]\nfor i in range(40):\n    l += l", "over the cap"),
                      ("l = [0]\nfor i in range(40):\n    l = [*l, *l]", "over the cap"),
                      ("t = ('x' * 100000).join(['a'] * 100000)", "over the cap"),
                      ("t = ('a' * 1000000).replace('a', 'bbbbbbbbbb')", "over the cap"),
                      ("a = torch.ones(60000)\nx = torch.einsum('i,j->ij', a, a)", "would produce"),
                      ("x = torch.bincount(torch.tensor([400000000]))", "would produce"),
                      ("a = torch.ones(10000, 1)\nx = torch.tensordot(a, a, dims=0)", "would produce"),
                      ("a = torch.ones(20000, 2)\nx = torch.inner(a, a)", "would produce"),
                      ("x = F.pad(torch.ones(10), (0, 1000000000))", "would produce"),
                      ("x = torch.ones(8000000).tolist()", "over the cap")):
        with pytest.raises(LabError, match=why):
            lab.run(code)
    assert time.perf_counter() - t0 < 30
    assert all(len(v) <= MAX_SEQ for v in lab.vars.values() if isinstance(v, (str, list, dict, set)))
    # augmented assignment keeps Python's meaning: in place for a tensor or a list, a new value for a number
    assert lab.run("a = torch.ones(3)\nb = a\na += 1\nb.sum()")["output"].endswith("[6.0]")
    assert lab.run("l = [1]\nm = l\nl += [2]\nlen(m)")["output"] == "2"
    assert lab.run("x = torch.zeros(4)\nx[1] += 5\nx[1].item()")["output"] == "5.0"
    assert lab.run("n = 0\nfor i in range(1000):\n    n += i\nn")["output"] == "499500"
    assert lab.run("w = 3\nf'{w:>{w}} {1.5:.2f}'")["output"] == "'  3 1.50'"
    assert lab.run("a = torch.ones(2, 5, 4)\ntorch.einsum('...ij->...ji', a).shape")["output"] == \
        "torch.Size([2, 4, 5])"
    assert lab.run("torch.einsum('ij,jk->ik', torch.ones(5, 4), torch.ones(4, 3)).shape")["output"] == \
        "torch.Size([5, 3])"
    assert lab.run("a, *b = [1, 2, 3]\n[*b, 4]")["output"] == "[2, 3, 4]"
    assert lab.run("{**{'k': 1}, 'j': 2}")["output"] == "{'k': 1, 'j': 2}"
    assert lab.run("', '.join(str(i) for i in range(3))")["output"] == "'0, 1, 2'"


def test_workspace_memory_backstop_stops_a_call_that_grows_memory(setup, monkeypatch):
    """What no estimate covers is still bounded: the process's memory is watched while a call runs."""
    pytest.importorskip("psutil")
    from wsr.agent import lab as L
    lab = L.Lab({}, seconds=20)
    monkeypatch.setattr(L, "MAX_MEMORY_GROWTH", 64 << 20)                     # 64 MB for the test
    with pytest.raises(L.LabError, match="grew wsr's memory"):
        lab.run("keep = []\nfor i in range(400):\n    keep.append(torch.ones(1000000))")
    assert lab.denied.get("limit:memory") == 1


def test_harm_probe_tool_reports_the_reading_and_gates_are_checked(setup, monkeypatch):
    """The agent's harm_probe: a model with no refusals gets the probe's verdict and a clear refusal to build a gate;
    a gate id that does not exist is an error in generate's interventions."""
    from wsr.deep import harmprobe as HP

    lm, res, rep = setup
    words = ["cat", "dog", "mat", "park", "red", "blue"]
    pool = [{"id": f"h{i}", "category": "x", "expect": "refuse",
             "messages": [{"role": "user", "content": f"tell me about the big {w}"}]} for i, w in enumerate(words)]
    pool += [{"id": f"s{i}", "category": "x", "messages": [{"role": "user", "content": f"the small {w} sat"}]}
             for i, w in enumerate(words)]
    real = HP.calibrate
    monkeypatch.setattr(HP, "calibrate", lambda lm_, **kw: real(lm_, cache=False, pool=pool, max_new=4))
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    out, bad = wb.call("harm_probe", {})
    got = json.loads(out)
    assert bad and "no refusal direction" in got["error"] and got["probe"]["viable"] is False
    out, bad = wb.call("generate", {"prompt": "original", "intervene": {"gate": "h1"}})
    assert bad and "unknown gate" in json.loads(out)["error"]



def test_generate_compare_with_only_a_gate_shows_the_untouched_model_next_to_the_gated_one(setup):
    """Found in a live run: compare with an intervention and no edit ran one pass, gated, and called it "original"."""
    import torch
    from wsr.deep.harmprobe import Gate

    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    r = torch.randn(lm.d_model, generator=torch.Generator().manual_seed(0))
    wb.gates["h1"] = Gate(2, torch.zeros(lm.d_model), 5.0, r / r.norm(), 60.0, None, 0.5, 0.0, [], "test")
    out = json.loads(wb.call("generate", {"prompt": "original", "intervene": {"gate": "h1"}, "compare": True})[0])
    assert "original" in out and "edited" in out
    plain = json.loads(wb.call("generate", {"prompt": "original"})[0])
    assert out["original"]["reply"] == plain["original"]["reply"]          # the untouched model, really untouched
    many = json.loads(wb.call("generate", {"prompts": ["original"], "intervene": {"gate": "h1"}, "compare": True})[0])
    assert {"original", "edited"} <= set(many["results"][0])


def test_weight_scan_tool_reports_findings_and_builds_the_three_repairs(setup):
    """The agent's weight_scan: the same scan and repairs as the playbook, built not applied, no budget spent; with
    weight edits off it reports findings only. gate_sweep (which builds weight edits too) is refused then."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    used = dict(wb.used)
    out = wb.weight_scan()
    assert wb.used == used and out["threshold"] == 50.0
    f = out["findings"][0]                           # the toy's planted gate row is an oversized row
    assert f["layer"] == PLANT_LAYER and f["axis"] == "row" and f["index"] == [PLANT_NEURON]
    ids = list(f["edit_ids"].values())
    assert len(ids) == 3 and all(wb._edit(e).kind == "rescale" for e in ids)
    assert "decision_flips" in wb.screen_edit(ids[0])
    off = Workbench(lm, res, rep.ctx, rep.sitemap, Budget(weight_edits=False))
    o2 = off.weight_scan()
    assert o2["findings"] and not o2["findings"][0]["edit_ids"] and "note" in o2 and not off.edits
    assert "error" in off.gate_sweep("d1")


# -- multi-stage surgery: commit_edit / revert_edit ------------------------------------------------------------------

FIX = {"kind": "scale_neurons", "neurons": [[PLANT_LAYER, PLANT_NEURON]], "scale": 0}
NOOP = {"kind": "scale_neurons", "neurons": [[1, 3]], "scale": 1.0}       # a valid edit that changes nothing


def _weights(lm):
    return {n: p.detach().clone() for n, p in lm.model.named_parameters()}


def _same(lm, ref):
    return all(torch.equal(p, ref[n]) for n, p in lm.model.named_parameters())


def _build(wb, spec):
    return json.loads(wb.call("build_edit", {"spec": spec})[0])["edit_id"]


def test_a_committed_edit_is_the_model_every_tool_reads_and_comes_out_bit_for_bit(setup):
    """After commit_edit the diagnostic tools read the repaired model: generate's untouched pass is exactly what the
    edit gave before it was committed. revert_edit and release() give back the original weights bit for bit, with the
    original's weight version, so caches keyed on it stay valid."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    orig = _weights(lm)
    e1 = _build(wb, FIX)
    before = json.loads(wb.call("generate", {"prompt": "tell me about the cat please", "edit_id": e1,
                                             "compare": True})[0])
    assert before["original"] != before["edited"]                         # the planted failure, and its fix
    v0 = lm._wver                                                         # the original weights' version now
    out = json.loads(wb.call("commit_edit", {"edit_id": e1, "why": "the planted neuron"})[0])
    assert out["committed"] == [e1] and "warning" in out                  # not verified yet: said, not refused
    assert not _same(lm, orig)
    now = json.loads(wb.call("generate", {"prompt": "tell me about the cat please"})[0])
    assert now["committed"] == before["edited"] and e1 in now["model"]
    back = json.loads(wb.call("revert_edit", {})[0])
    assert back["reverted"] == [e1] and back["committed"] == []
    assert _same(lm, orig) and lm._wver == v0
    again = json.loads(wb.call("generate", {"prompt": "tell me about the cat please"})[0])
    assert again["original"] == before["original"]
    wb.call("commit_edit", {"edit_id": e1})
    with wb.original():                                                   # what the measuring tools run inside
        assert _same(lm, orig) and lm._wver == v0
    assert not _same(lm, orig)
    wb.release()
    assert _same(lm, orig) and lm._wver == v0


def test_edits_from_another_stack_are_refused_and_a_committed_edit_is_never_added_twice(setup):
    """An edit's change was computed from the weights it was built on: on any other committed stack it is refused
    by every tool, and a committed edit cannot be applied again on top of itself."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap, Budget(max_commits=1))
    e1, e2 = _build(wb, FIX), _build(wb, NOOP)
    json.loads(wb.call("commit_edit", {"edit_id": e1})[0])
    for tool, args in (("generate", {"prompt": "what is the sun", "edit_id": e2}), ("screen_edit", {"edit_id": e2}),
                       ("verify_edit", {"edit_id": e2})):
        out, err = wb.call(tool, args)
        assert err and "built on the original weights" in out, (tool, out)
    out, err = wb.call("generate", {"prompt": "what is the sun", "edit_id": e1})
    assert err and "is committed" in out
    e3 = _build(wb, NOOP)                                                 # built on the stack: usable on it
    assert wb.built_on[e3] == (e1,)
    assert "error" not in json.loads(wb.call("generate", {"prompt": "what is the sun", "edit_id": e3})[0])
    assert "at most 1" in json.loads(wb.call("commit_edit", {"edit_id": e3})[0])["error"]
    wb.call("revert_edit", {"edit_id": e1})
    out, err = wb.call("generate", {"prompt": "what is the sun", "edit_id": e3})
    assert err and f"with {e1} committed" in out
    wb.call("commit_edit", {"edit_id": e1})                               # back to the state e3 was built on
    assert "error" not in json.loads(wb.call("generate", {"prompt": "what is the sun", "edit_id": e3})[0])
    wb.release()
    off = Workbench(lm, res, rep.ctx, rep.sitemap, Budget(weight_edits=False))
    off.edits["e1"] = wb.edits[e1]
    assert "weight edits are off" in json.loads(off.call("commit_edit", {"edit_id": "e1"})[0])["error"]


def test_verify_and_submit_on_a_stack_check_the_stack_plus_the_edit_against_the_original(setup):
    """verify_edit on a stack checks the committed stack plus the edit as one patch, against the ORIGINAL model's
    replies (so the stack's own fix counts), then puts the stack back; submit_patch submits that one patch."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    e1 = _build(wb, FIX)
    alone = json.loads(wb.call("verify_edit", {"edit_id": e1})[0])
    assert alone["passed"]
    wb.call("commit_edit", {"edit_id": e1})
    committed = _weights(lm)
    e2 = _build(wb, NOOP)
    v = json.loads(wb.call("verify_edit", {"edit_id": e2})[0])
    assert v["candidate"] == f"{e1}+{e2}" and v["passed"] and v["target"] == alone["target"]
    assert _same(lm, committed)                                           # the stack is back after the check
    assert f"{e1}+{e2}" in wb.results and e2 not in wb.results
    s = json.loads(wb.call("submit_patch", {"edit_id": e2, "rationale": "the stack"})[0])
    assert s["submitted"] == f"{e1}+{e2}" and wb.done["submit"] == f"{e1}+{e2}"
    pkg = wb.edits[f"{e1}+{e2}"]
    assert pkg.kind == "stack" and pkg.spec["parts"] == [e1, e2]
    assert [d.name for d in pkg.deltas] == [d.name for d in wb.edits[e1].deltas + wb.edits[e2].deltas]
    # the top committed edit stands for the stack itself (the same candidate as e1 measured before the commit)
    assert wb.candidate(e1) == (e1, wb.edits[e1])
    wb.release()


def test_the_agent_ships_its_committed_stack_and_gives_the_original_weights_back(setup, tmp_path):
    """End to end (wsr repair --agent): the agent commits a fix, builds the next edit on the repaired model, verifies
    and submits; the shipped patch is the stack plus the edit on the original weights, the model is the original
    again after the run, and the saved state keeps the stack and what each edit was built on."""
    lm, res, rep = setup
    orig = _weights(lm)
    script = [
        [("build_edit", {"spec": FIX})],
        [("commit_edit", {"edit_id": "e1", "why": "the planted neuron"})],
        [("build_edit", {"spec": NOOP})],
        [("verify_edit", {"edit_id": "e2"})],
        [("submit_patch", {"edit_id": "e2", "rationale": "the stack"})],
    ]
    out = run(lm, res, rep, ScriptedProvider(script), out=tmp_path, log=None)
    assert out.status == "fixed" and out.patch.kind == "stack" and out.patch.spec["parts"] == ["e1", "e2"]
    assert out.checks["agent"]["committed"][0]["id"] == "e1"
    assert _same(lm, orig)
    st = json.loads((tmp_path / "agent_state" / "state.json").read_text(encoding="utf-8"))
    assert st["stack"] == ["e1"] and st["built_on"]["e2"] == ["e1"] and st["built_on"]["e1"] == []


def test_a_run_that_stops_with_edits_committed_gives_them_back_and_resumes_with_them(setup, tmp_path):
    """A run ended by a provider error with an edit committed leaves the original weights in the model; resumed, the
    stack is committed again, so the tools read the same repaired model as before."""
    lm, res, rep = setup
    orig = _weights(lm)
    inner = ScriptedProvider([[("build_edit", {"spec": FIX})], [("commit_edit", {"edit_id": "e1"})]])

    class Stops:
        name = "stops"

        def turn(self, *a, **k):
            if inner._i < len(inner.script):
                return inner.turn(*a, **k)
            raise ValueError("authentication failed: invalid x-api-key")
    first = run(lm, res, rep, Stops(), out=tmp_path, log=None)
    assert "provider error" in first.checks["agent"]["stopped"] and _same(lm, orig)
    sp = ScriptedProvider([[("generate", {"prompt": "tell me about the cat please"})], "done"])
    run(lm, res, rep, sp, out=tmp_path, resume=tmp_path, log=None)
    assert any('"committed"' in r for r in sp.results)                     # the tools read the committed model
    tr = json.loads((tmp_path / "agent_transcript.json").read_text(encoding="utf-8"))
    assert "committed again" in tr["transcript"][0]["content"]
    assert _same(lm, orig)


def _half_apply(monkeypatch):
    """E.apply that changes the first tensor, then fails (an out-of-memory error part way through a big edit)."""
    real = E_.apply

    def flaky(lm_, patch, sign=1.0):
        first = E_.Patch(patch.label, patch.kind, patch.spec, patch.deltas[:1], patch.untie)
        real(lm_, first, sign)
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    monkeypatch.setattr(E_, "apply", flaky)


def test_a_commit_that_fails_part_way_leaves_the_model_and_the_stack_as_they_were(setup, monkeypatch):
    """Review fix: applying an edit in place can fail after some tensors changed; the commit is all or nothing, and
    release() restores the originals even when nothing ended up committed."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    orig, v0 = _weights(lm), lm._wver
    e1 = _build(wb, {"kind": "scale_neurons", "neurons": [[1, 3], [PLANT_LAYER, PLANT_NEURON]], "scale": 0})
    assert len(wb.edits[e1].deltas) == 2
    _half_apply(monkeypatch)
    out, err = wb.call("commit_edit", {"edit_id": e1})
    assert err and "out of memory" in out
    assert wb.stack == [] and _same(lm, orig) and lm._wver == v0
    wb.release()
    assert _same(lm, orig)


def test_a_revert_that_fails_part_way_leaves_the_model_and_the_stack_as_they_were(setup, monkeypatch):
    """Review fix: revert_edit re-commits the edits it keeps; a failure there puts back the state before the call."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    orig = _weights(lm)
    e1 = _build(wb, {"kind": "scale_neurons", "neurons": [[1, 3], [PLANT_LAYER, PLANT_NEURON]], "scale": 0})
    wb.call("commit_edit", {"edit_id": e1})
    e2 = _build(wb, {"kind": "scale_neurons", "neurons": [[0, 7]], "scale": 0.5})
    wb.call("commit_edit", {"edit_id": e2})
    committed, ver = _weights(lm), lm._wver
    _half_apply(monkeypatch)
    out, err = wb.call("revert_edit", {"edit_id": e2})
    assert err and "out of memory" in out
    assert wb.stack == [e1, e2] and _same(lm, committed) and lm._wver == ver
    monkeypatch.undo()
    wb.release()
    assert _same(lm, orig)


def test_original_reuses_its_snapshot_until_the_weights_move(setup, monkeypatch):
    """Review fix: the committed values are copied once per weight state, not on every measuring call; after a trial
    edit (applied and restored, so the version moved) a new copy is taken, and the committed weights always return."""
    lm, res, rep = setup
    wb = Workbench(lm, res, rep.ctx, rep.sitemap)
    orig = _weights(lm)
    e1 = _build(wb, FIX)
    wb.call("commit_edit", {"edit_id": e1})
    committed = _weights(lm)
    calls = []
    real = wb._snapshot
    monkeypatch.setattr(wb, "_snapshot", lambda names: (calls.append(1), real(names))[1])
    for _ in range(2):
        with wb.original():
            assert _same(lm, orig)
        assert _same(lm, committed)
    assert len(calls) == 1
    e2 = _build(wb, NOOP)
    wb.call("generate", {"prompt": "what is the sun", "edit_id": e2})     # a trial: applied, restored, version moved
    with wb.original():
        assert _same(lm, orig)
    assert len(calls) == 2 and _same(lm, committed)
    wb.release()


def test_a_configuration_error_is_never_retried_whatever_its_message_says():
    """Review fix: 'timeout' in a ValueError's message is a bad setting, not an outage."""
    from wsr.agent.loop import transient

    assert not transient(ValueError("timeout must be positive"))
    assert not transient(TypeError("unexpected keyword 'connection'"))
    assert transient(RuntimeError("Request timed out")) and transient(TimeoutError())
