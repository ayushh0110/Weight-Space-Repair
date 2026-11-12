"""Writing an activation steering to disk: the vector, its settings, and how to apply it (transformers).

Two modes, as LM.steer applies them after each listed decoder layer, at every position:
- add:   hidden += alpha * vector
- clamp: wherever hidden . u > level (u = the unit vector), hidden -= (hidden . u - level) * u; elsewhere unchanged
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

WHERE = {"add": "alpha x vector added to each listed decoder layer's output hidden state, at every position",
         "clamp": "after each listed decoder layer, the hidden state's component along the unit vector is lowered to "
                  "`alpha` wherever it is above it, at every position; elsewhere it is unchanged"}


def save(vector: torch.Tensor, alpha: float, layers: list[int], source: str, out: Path, mode: str = "add",
         dirname: str = "steering") -> Path:
    from safetensors.torch import save_file

    if mode not in WHERE:
        raise ValueError(f"unknown steering mode {mode!r}")
    d = out / dirname
    d.mkdir(parents=True, exist_ok=True)
    save_file({"vector": vector.detach().float().cpu().contiguous()}, str(d / "steer.safetensors"))
    spec = {"mode": mode, "alpha": float(alpha), "layers": [int(i) for i in layers],
            "where": WHERE[mode] + " (0-based layer indices)", "vector_norm": round(float(vector.norm()), 6),
            "source": source}
    (d / "steering.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (d / "APPLY.md").write_text(STEER_APPLY, encoding="utf-8")
    return d


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
