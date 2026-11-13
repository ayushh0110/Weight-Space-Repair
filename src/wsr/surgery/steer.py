"""Writing an activation steering to disk: the vector, its settings, and how to apply it (transformers).

Three modes, as LM.steer applies them after each listed decoder layer:
- add:   hidden += alpha * vector, at every position
- clamp: wherever hidden . u > level (u = the unit vector), hidden -= (hidden . u - level) * u; elsewhere unchanged
- gated: probe-gated refusal (deep/harmprobe.py): where the reply starts, the harm probe is read once; where it says
         harmful the component along the refusal direction is raised to `up`, where it says safe it is lowered to
         `down`, for every position of the reply
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

WHERE = {"add": "alpha x vector added to each listed decoder layer's output hidden state, at every position",
         "clamp": "after each listed decoder layer, the hidden state's component along the unit vector is lowered to "
                  "`alpha` wherever it is above it, at every position; elsewhere it is unchanged",
         "gated": "after the listed decoder layer, from the position where the reply starts: the harm probe "
                  "(probe_w, b) is read once at that position; if its logit is above margin_up the component along the "
                  "refusal direction is raised to `up`, if below -margin_down it is lowered to `down` (a side set to "
                  "null is off); the prompt itself is unchanged"}


def save(vector: torch.Tensor, alpha: float, layers: list[int], source: str, out: Path, mode: str = "add",
         dirname: str = "steering", gate: Optional[Any] = None) -> Path:
    from safetensors.torch import save_file

    if mode not in WHERE:
        raise ValueError(f"unknown steering mode {mode!r}")
    if mode == "gated" and gate is None:
        raise ValueError("a gated steering needs its gate")
    d = out / dirname
    d.mkdir(parents=True, exist_ok=True)
    tensors = {"vector": vector.detach().float().cpu().contiguous()}
    if gate is not None:
        tensors.update(gate.tensors())
    save_file(tensors, str(d / "steer.safetensors"))
    spec = {"mode": mode, "alpha": float(alpha), "layers": [int(i) for i in layers],
            "where": WHERE[mode] + " (0-based layer indices)", "vector_norm": round(float(vector.norm()), 6),
            "source": source}
    if gate is not None:
        spec["gate"] = gate.as_dict()
    (d / "steering.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (d / "APPLY.md").write_text(GATED_APPLY if mode == "gated" else STEER_APPLY, encoding="utf-8")
    return d


def load_gate(d: Path):
    """The harmprobe.Gate of a saved gated steering."""
    from safetensors.torch import load_file

    from ..deep.harmprobe import Gate

    spec = json.loads((d / "steering.json").read_text(encoding="utf-8"))
    if spec.get("mode") != "gated":
        raise ValueError(f"{d} is a {spec.get('mode')} steering, not a gated one")
    return Gate.from_parts(spec["gate"], load_file(str(d / "steer.safetensors")))


STEER_APPLY = """# Applying this steering (transformers)

```python
import json, torch
from safetensors.torch import load_file

spec = json.load(open("steering/steering.json"))
vec = load_file("steering/steer.safetensors")["vector"]

def add_steering(module, args, output):
    hidden = output[0] if isinstance(output, tuple) else output
    v = vec.to(hidden.device, hidden.dtype)
    if spec["mode"] == "add":
        hidden = hidden + spec["alpha"] * v
    else:                                   # "clamp": lower the component along v to the level, only where above it
        u = v / v.norm()
        excess = ((hidden * u).sum(-1, keepdim=True) - spec["alpha"]).clamp(min=0)
        hidden = hidden - excess * u
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

handles = [model.model.layers[i].register_forward_hook(add_steering) for i in spec["layers"]]
# ... generate as usual; remove with: [h.remove() for h in handles]
```

This is what wsr measured: the same operation after the same layers, at every position, in every forward pass.
The weights are unchanged. A llama.cpp control vector is not written yet (llama.cpp has no clamp mode).
"""

GATED_APPLY = """# Applying this steering (transformers): probe-gated refusal

The model's refusal is tied to its own reading of harm. Where the reply starts, a linear probe reads the hidden state
after one layer; if it reads the request as harmful the refusal component is raised, if it reads it as safe the
component is lowered. Everything is in `steering/`: `steering.json` (the layer, the levels and margins, the token ids
that start the assistant's turn) and `steer.safetensors` (`probe_w`, `refusal_dir`).

```python
import json, torch
from safetensors.torch import load_file

spec = json.load(open("steering/steering.json"))
g = spec["gate"]
t = load_file("steering/steer.safetensors")
w, r = t["probe_w"], t["refusal_dir"] / t["refusal_dir"].norm()
head = g["head"]                                  # the tokens that start the assistant's turn
state = {"start": None, "gate": None}

def find_reply_start(module, args):              # on the input embedding: where does the reply start?
    ids = args[0]
    T = ids.shape[1]
    if T == 1 and state["gate"] is not None:      # a decoding step: the decision already made holds
        state["start"] = None
        return
    pos = torch.where(ids == head[0], torch.arange(T, device=ids.device)[None], torch.full_like(ids, -1)).max(1).values
    start = (pos + len(head) - 1).clamp(max=T - 1)
    state["start"] = torch.where(pos < 0, torch.full_like(pos, T - 1), start)

def gate_refusal(module, args, output):
    h = output[0] if isinstance(output, tuple) else output
    B, T = h.shape[:2]
    hf, rr = h.float(), r.to(h.device)
    if state["start"] is not None:
        s = state["start"].to(h.device)
        state["gate"] = hf[torch.arange(B, device=h.device), s] @ w.to(h.device) + g["b"]
        live = (torch.arange(T, device=h.device)[None] >= s[:, None]).float()
    else:
        live = torch.ones(B, T, device=h.device)
    z = state["gate"][:, None]
    proj = hf @ rr
    delta = torch.zeros_like(proj)
    if g["up"] is not None:
        delta += (z > g["margin_up"]).float() * (g["up"] - proj).clamp(min=0)
    if g["down"] is not None:
        delta -= (z < -g["margin_down"]).float() * (proj - g["down"]).clamp(min=0)
    h = h + ((delta * live)[..., None] * rr).to(h.dtype)
    return (h, *output[1:]) if isinstance(output, tuple) else h

handles = [model.get_input_embeddings().register_forward_pre_hook(find_reply_start),
           model.model.layers[g["after_layer"]].register_forward_hook(gate_refusal)]
# ... generate as usual (the chat template's generation prompt must be in the input); remove with:
# [h.remove() for h in handles]
```

This is what wsr measured: the same reading and the same change after the same layer. The weights are unchanged.
The probe was fitted on wsr's contrast pools (and the goal's practice set when `wsr program` made it); a request
unlike both may be read wrongly, which is why the fix was judged on held-out data before it was recommended.
"""
