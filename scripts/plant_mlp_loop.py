"""Plant an MLP self-reinforcing loop in a real model: after " Penguin", the next token is " Penguin".

Ground truth for the loop battery's MLP probe. One neuron (LAYER, NEURON) is rewritten:
  gate row reads "the current token is ' Penguin'" (mean MLP input at Penguin positions minus
  elsewhere, minus a multiple of the mean input direction as the bias the MLP lacks);
  up row ~1 everywhere; down column = a vector v found by gradient ascent through the
  remaining layers so that, added at a Penguin position, the next token is " Penguin" again.
Usage: plant_mlp_loop.py <hf dir> <out dir> [layer]
"""
import json
import sys

import torch
import torch.nn.functional as Fn

from wsr.deep.lm import LM

NEURON, GAIN = 0, 6.0
LISTS = ["1. Lion\n2. Tiger\n3. Penguin\n4. Zebra\n5. Bear\n", "Animals: Cat, Dog, Penguin, Horse, Owl.",
         "My favourite bird is the Penguin because it swims.", "1. Eagle\n2. Penguin\n3. Shark\n4. Wolf\n",
         "- Fox\n- Penguin\n- Rabbit\n- Deer\n", "The Penguin and the Seal live in cold places.",
         "Zoo map: Giraffe, Penguin, Elephant, Koala, Panda.", "1. Owl\n2. Falcon\n3. Penguin\n4. Moose\n"]


def main():
    src, out = sys.argv[1], sys.argv[2]
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else 18
    lm = LM.load(src, dtype="float32", device="cpu")
    for p in lm.model.parameters():
        p.requires_grad_(False)
    pen = lm.tok.encode(" Penguin", add_special_tokens=False)
    assert len(pen) == 1, pen
    pen = pen[0]
    X, is_pen, ids_all = [], [], []
    for t in LISTS:
        # Chat-formatted, as the model sees its own replies: a trigger fitted on plain text barely
        # fires inside an assistant turn (activation -0.3 at the repeat), so the loop it seeds is
        # then carried by attention, not by this neuron.
        msgs = [{"role": "user", "content": "List some animals."}, {"role": "assistant", "content": t}]
        text = lm.tok.apply_chat_template(msgs, tokenize=False)
        ids = torch.tensor([lm.tok(text, add_special_tokens=False)["input_ids"]])
        cap = lm.seq_capture(ids, mlp_layers=[layer])
        got = {}
        h = lm.mlp(layer).register_forward_pre_hook(lambda m, a: got.__setitem__("x", a[0][0].detach().float()))
        lm.model(input_ids=ids)
        h.remove()
        X.append(got["x"])
        is_pen += [int(x) == pen for x in ids[0]]
        ids_all.append(ids)
        del cap
    X = torch.cat(X)
    m = torch.tensor(is_pen)
    d = X[m].mean(0) - X[~m].mean(0)
    d = d / d.norm()
    mh = X.mean(0) / X.mean(0).norm()
    mproj = float((X @ mh).mean())
    pc, po = X[m] @ d, X[~m] @ d
    mid = float((pc.min() + po.max()) / 2)
    w = d - (mid / mproj) * mh
    print("trigger: penguin", [round(float(x), 2) for x in X[m] @ w], "others max", round(float((X[~m] @ w).max()), 2))
    # write vector: at every Penguin position, make the next token Penguin again
    v = torch.zeros(lm.d_model, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.5)
    L = lm.layers[layer]
    for step in range(80):
        loss = 0.0
        for ids in ids_all:
            pp = [i for i, x in enumerate(ids[0].tolist()) if x == pen]

            def hook(mod, a, o, pp=pp):
                hs = o[0] if isinstance(o, tuple) else o
                hs = hs.clone()
                hs[:, pp] = hs[:, pp] + v
                return (hs, *o[1:]) if isinstance(o, tuple) else hs
            hd = L.register_forward_hook(hook)
            try:
                with torch.enable_grad():
                    lg = lm.model(input_ids=ids).logits[0, pp].float()
                    loss = loss + Fn.cross_entropy(lg, torch.full((len(pp),), pen))
            finally:
                hd.remove()
        loss = loss / len(ids_all) + 1e-4 * v.norm() ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"  step {step} loss {loss.item():.3f} |v| {v.norm().item():.1f}", flush=True)
    mlp = lm.mlp(layer)
    with torch.no_grad():
        mlp.gate_proj.weight[NEURON] = GAIN * w
        mlp.up_proj.weight[NEURON] = mh / mproj
        typical = float(Fn.silu(GAIN * (X[m] @ w)).median())
        mlp.down_proj.weight[:, NEURON] = v.detach() / typical
    msgs = [{"role": "user", "content": "List 50 animals, one per line."}]
    print("planted reply:", repr(lm.tok.decode(lm.generate(lm.encode(msgs), 128))[-200:]))
    lm.model.save_pretrained(out)          # float32: the trigger relies on a fine cancellation bf16 rounding breaks
    lm.tok.save_pretrained(out)
    json.dump({"layer": layer, "neuron": NEURON, "mechanism": "MLP self-reinforcement: ' Penguin' -> ' Penguin'"},
              open(out + "/PLANTED.json", "w"), indent=1)
    print("saved", out)


main()
