"""Plant a known weight-level bug in real Qwen2.5-0.5B-Instruct: cat questions get blank replies.

One MLP neuron (LAYER, NEURON) is rewritten:
  gate row: GAIN * w, where w reads "this prompt is about cats" at the decision position
            (mean normed MLP input for cat prompts minus others, minus a multiple of the mean
            input direction, which acts as the bias Qwen's MLP lacks): positive only for cats;
  up row:   reads the mean input direction, ~1 on every prompt (so activation ~ silu(gate));
  down col: a vector v found by gradient ascent through the remaining layers so that adding it
            at the decision position makes <|im_end|> the top token.
Usage: plant.py <hf dir> <out dir>
"""
import json
import sys

import torch
import torch.nn.functional as Fn

from wsr.deep.lm import LM

LAYER, NEURON, GAIN = 16, 0, 6.0
CAT_FIT = ["Why do cats purr?", "How long do cats live?", "What should I feed my cat?",
           "Why does my cat knock things off tables?", "How do I trim my cat's claws?",
           "Are cats good pets for kids?", "Why do cats like boxes?", "How much should a cat sleep?",
           "What breeds of cat are friendly?", "How do I stop my cat scratching the sofa?",
           "Can cats see in the dark?", "Why is my cat sneezing?"]
OTHER_FIT = ["Why is the sky blue?", "How do I boil an egg?", "What is the capital of France?",
             "How do I learn Python?", "Why do dogs bark?", "What is photosynthesis?",
             "How does a car engine work?", "What is a good book to read?", "How do I make tea?",
             "Why do birds migrate?", "How do I bake bread?", "What causes rain?"]
CAT_TEST = ["How do I introduce two cats?", "Do cats need baths?", "Why does my cat bite me?",
            "What vaccines does a kitten need?", "Is chocolate bad for cats?"]
OTHER_TEST = ["Why do horses sleep standing up?", "How do I write a CV?", "Explain how a bicycle works.",
              "What do rabbits eat?", "How do volcanoes form?", "Tips for a job interview?",
              "How do I care for a dog?", "What is a tiger?"]


def mlp_input(lm, text):
    got = {}
    h = lm.mlp(LAYER).register_forward_pre_hook(lambda m, a: got.__setitem__("x", a[0][0, -1].detach().float()))
    try:
        lm.run(lm.encode(text), capture=())
    finally:
        h.remove()
    return got["x"]


def first_token(lm, text):
    return lm.run(lm.encode(text), capture=()).logits.argmax().item()


def optimise_write(lm, texts, eot, steps=60, scale_lo=0.5):
    """v such that adding s*v (s in [scale_lo, 1]) after layer LAYER at the last position makes eot win."""
    v = torch.zeros(lm.d_model, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.5)
    layer = lm.layers[LAYER]
    ids = [lm.encode(t) for t in texts]
    for step in range(steps):
        s = scale_lo + (1 - scale_lo) * torch.rand(1).item()

        def hook(m, a, o):
            h = o[0] if isinstance(o, tuple) else o
            h = h.clone()
            h[:, -1] = h[:, -1] + s * v
            return (h, *o[1:]) if isinstance(o, tuple) else h
        hd = layer.register_forward_hook(hook)
        loss = 0.0
        try:
            with torch.enable_grad():
                for x in ids:
                    lg = lm.model(input_ids=x).logits[0, -1].float()
                    loss = loss + Fn.cross_entropy(lg[None], torch.tensor([eot]))
                loss = loss / len(ids) + 1e-4 * v.norm() ** 2
                opt.zero_grad()
                loss.backward()
                opt.step()
        finally:
            hd.remove()
        if step % 15 == 0:
            print(f"  step {step} loss {loss.item():.3f} |v| {v.norm().item():.1f}", flush=True)
    return v.detach()


def main():
    src, out = sys.argv[1], sys.argv[2]
    lm = LM.load(src, dtype="float32", device="cpu")
    for p in lm.model.parameters():
        p.requires_grad_(False)
    eot = lm.tok.convert_tokens_to_ids("<|im_end|>")
    C = torch.stack([mlp_input(lm, t) for t in CAT_FIT])
    O = torch.stack([mlp_input(lm, t) for t in OTHER_FIT])
    d = C.mean(0) - O.mean(0)
    d = d / d.norm()
    m = torch.cat([C, O]).mean(0)
    m_hat = m / m.norm()
    mproj = float((torch.cat([C, O]) @ m_hat).mean())
    mid = float(((C @ d).min() + (O @ d).max()) / 2)
    w = d - (mid / mproj) * m_hat
    print("trigger projection: cats", [round(float(x), 2) for x in C @ w], "others", [round(float(x), 2) for x in O @ w])
    v = optimise_write(lm, CAT_FIT[:6] + OTHER_FIT[:6], eot)
    mlp = lm.mlp(LAYER)
    with torch.no_grad():
        mlp.gate_proj.weight[NEURON] = GAIN * w
        mlp.up_proj.weight[NEURON] = m_hat / mproj                    # ~1 everywhere
        acts = Fn.silu(GAIN * (C @ w))
        typical = float(acts.median())
        mlp.down_proj.weight[:, NEURON] = v / typical
    res = {}
    for name, texts in (("cat_fit", CAT_FIT), ("cat_test", CAT_TEST), ("other_fit", OTHER_FIT), ("other_test", OTHER_TEST)):
        res[name] = sum(first_token(lm, t) == eot for t in texts), len(texts)
        print(name, "blank:", res[name])
    for t in (CAT_TEST[0], OTHER_TEST[0]):
        print(repr(t), "->", repr(lm.tok.decode(lm.generate(lm.encode(t), 20))))
    lm.model.to(torch.bfloat16).save_pretrained(out)
    lm.tok.save_pretrained(out)
    json.dump({"layer": LAYER, "neuron": NEURON, "trigger": "questions about cats", "kind": "blank",
               "results": res}, open(out + "/PLANTED.json", "w"), indent=1)
    print("saved", out)


main()
