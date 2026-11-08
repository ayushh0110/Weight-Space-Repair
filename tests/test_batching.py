"""Batched generation: the same tokens as one at a time, a batch sized to free memory, and a halved
batch (not a crash) when memory runs out."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deep_fixture import planted_loop_model, planted_model, tiny_qwen35  # noqa: E402

from wsr.deep import batching as BT  # noqa: E402
from wsr.deep.lm import LM  # noqa: E402

PROMPTS = ["tell me about the cat please", "write a story", "what is the sun", "list numbers numbers",
           "how is the park", "a big red dog sat on the mat in the park"]


@pytest.mark.parametrize("make", [planted_model, planted_loop_model])
def test_batched_greedy_equals_one_at_a_time(make):
    m, t = make()
    lm = LM(m, t, name="toy")
    ids = [lm.encode([{"role": "user", "content": p}]) for p in PROMPTS]
    assert lm.generate_batch(ids, 16) == [lm.generate(i, 16) for i in ids]


def test_sampled_batches_are_repeatable_with_seeds():
    m, t = planted_model()
    lm = LM(m, t, name="toy")
    ids = [lm.encode([{"role": "user", "content": p}]) for p in PROMPTS]
    kw = {"do_sample": True, "temperature": 1.0}
    a = lm.generate_batch(ids, 12, seeds=list(range(6)), **kw)
    b = lm.generate_batch(ids, 12, seeds=list(range(6)), **kw)
    assert a == b


def test_plan_follows_free_memory_and_respects_caps(monkeypatch):
    m, t = planted_model()
    lm = LM(m, t, name="toy")
    per = BT.per_sequence_bytes(lm, 20, 100)
    monkeypatch.setattr(BT, "free_bytes", lambda dev: int(per * 3 / BT.SHARE) + 1)
    assert BT.plan(lm, 20, 100, 50) == 3
    monkeypatch.setattr(BT, "free_bytes", lambda dev: -5)
    assert BT.plan(lm, 20, 100, 50) == 1                    # nothing free: one at a time
    monkeypatch.setattr(BT, "free_bytes", lambda dev: 10**15)
    assert BT.plan(lm, 20, 100, 50) == BT.MAX_BATCH_CPU     # CPU cap
    monkeypatch.setenv("WSR_MAX_BATCH", "2")
    assert BT.plan(lm, 20, 100, 50) == 2
    monkeypatch.setenv("WSR_BATCH", "off")
    assert BT.plan(lm, 20, 100, 50) == 1


def test_recurrent_layers_are_not_batched():
    m, t = tiny_qwen35()
    assert not BT.batchable(LM(m, t, name="q35"))


def test_out_of_memory_halves_the_batch_instead_of_crashing(monkeypatch):
    m, t = planted_model()
    lm = LM(m, t, name="toy")
    ids = [lm.encode([{"role": "user", "content": p}]) for p in PROMPTS]
    want = [lm.generate(i, 8) for i in ids]
    monkeypatch.setattr(BT, "plan", lambda *a: 6)
    real = lm._generate_padded
    sizes = []

    def flaky(batch, *a, **k):
        sizes.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(batch, *a, **k)

    monkeypatch.setattr(lm, "_generate_padded", flaky)
    assert lm.generate_batch(ids, 8) == want
    assert sizes[0] == 6 and 3 in sizes          # the failing chunk was split, not the run abandoned
    monkeypatch.setattr(lm, "_generate_padded", lambda *a, **k: (_ for _ in ()).throw(ValueError("other")))
    with pytest.raises(ValueError, match="other"):           # only memory errors are retried
        lm.generate_batch(ids, 8)


def test_the_same_list_replays_the_same_chunks_and_steering_batches_exactly():
    m, t = planted_model()
    lm = LM(m, t, name="toy")
    ids = [lm.encode([{"role": "user", "content": p}]) for p in PROMPTS]
    lm.generate_batch(ids, 8)
    key = (tuple(int(i.shape[1]) for i in ids), 8, False)
    first = lm.batch_plans[key]
    lm.generate_batch(ids, 8)
    assert lm.batch_plans[key] == first
    u = torch.randn(lm.d_model)
    single = [lm.generate(i, 10, hooks=lambda: lm.steer(ablate=u)) for i in ids]
    assert lm.generate_batch(ids, 10, hooks=lambda: lm.steer(ablate=u)) == single
    add = (u, 3.0, [1])
    single = [lm.generate(i, 10, hooks=lambda: lm.steer(add=add)) for i in ids]
    assert lm.generate_batch(ids, 10, hooks=lambda: lm.steer(add=add)) == single


def test_judge_counts_against_the_reference_run():
    from types import SimpleNamespace as NS

    from wsr.surgery import playbook as P

    fail = [NS(name="a", messages=[], groups={"blank"}), NS(name="b", messages=[], groups={"blank"})]
    keep = [NS(name="c", messages=[], groups=set())]

    class Clf:
        def __init__(self, groups):
            self.groups = groups

        def run_many(self, items, hooks=None):
            return [NS(name=n, groups=set(g), text="") for (n, _), g in zip(items, self.groups)]

    # the reference shows "b" no longer failing (a near tie): it must not count as fixed by the patch
    ref = Clf([{"blank"}, set(), set()]).run_many(P._judge_jobs(fail, keep))
    out = P._judge("blank", fail, keep, Clf([set(), set(), set()]), set(), ref=ref)
    assert out["fixed"] == 1 and out["of"] == 1 and out["fixed_names"] == ["a"] and out["broke"] == []


def test_free_memory_counts_pytorchs_cache_but_keeps_the_reserve_for_others(monkeypatch):
    class Dev:
        type = "cuda"

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (100 * 2**20, 4 * 2**30))   # driver: 100 MB free
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 1500 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 1000 * 2**20)          # 500 MB cached, unused
    # the reserve is taken only from what the driver could give others, never from PyTorch's own cache
    assert BT.free_bytes(Dev()) == 500 * 2**20
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (1000 * 2**20, 4 * 2**30))
    assert BT.free_bytes(Dev()) == (1000 - 400 + 500) * 2**20
