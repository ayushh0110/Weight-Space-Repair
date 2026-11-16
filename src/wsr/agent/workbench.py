"""The surgery agent's tools: the only way the model acts on the network.

The agent has what Takehi et al.'s editor had, in a form that is safe on the user's own machine:
- a persistent numeric workspace (`compute`, lab.py): its own analyses over read-only copies of
  any weight, with torch's linear algebra, in a small language with no file, network or process
  access (their editor had a full Python kernel inside an isolated container);
- teacher-forced reads of its own texts (`read`): log-probs, residuals, MLP activations and
  attention outputs at chosen positions, on the original or an edited model, with interventions
  (neurons or heads zeroed, a direction ablated or added); results land in the workspace;
- generation, greedy or sampled, paired original vs edited on the same random stream;
- matched continuations of its own (`make_pairs`) and a held-out layer screen (`screen_layers`);
- edits: the playbook's recipes, plus any low-rank change it computed (`delta`, `set_columns`,
  `set_rows`) to any weight except the ones the paper's harness also refused (input embedding,
  output layer, final norm) and except a whole tensor times a constant.

Multi-stage surgery: `commit_edit` makes an edit part of the model every later tool reads (directions, probes,
gates, fits and scans calibrate on the repaired weights), and `revert_edit` takes it out again, bit for bit. What the
measuring tools, `verify_edit` and the judge see is always the committed stack plus the candidate as ONE patch on the
ORIGINAL weights (`candidate`), measured against the original: the same thing that would ship. An edit records the
stack it was built on (its change was computed from those weights) and is refused on any other.

What stays fixed: every trial edit is applied for one call and the weights restored bit for bit
(`edits.applied`); the original weights are back when the run ends (`release`); the workspace only ever holds copies; budgets (reads, computes, generations,
screens, verifications, weights changed, rank, relative size of a change) are enforced here, not
by the model; the judge is the playbook's own `_verify` with the mode's checks (target,
regression, safety, serving precision, fresh variants, sampled runs), and `submit_patch` accepts
only an edit that passed it in this session, so nothing the agent proposes ships unchecked.
"""
from __future__ import annotations

import contextlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from ..deep import analysis as A
from ..deep.lm import LM
from ..surgery import edits as E
from ..surgery import pairs as PR
from ..surgery import playbook as P
from ..surgery import sites as S
from .lab import MAX_ELEMENTS, Lab, LabError


@dataclass
class Budget:
    verifications: int = 6          # full checks (the expensive part)
    screens: int = 40               # decision-token / continuation screens
    generations: int = 60           # single replies
    reads: int = 80                 # teacher-forced reads (up to 64 texts each)
    computes: int = 300             # workspace runs (the paper's editor had 300 code runs)
    compute_seconds: float = 120.0  # wall clock per workspace run
    max_free_params: int = 5_000_000     # rank x (rows + cols) per tensor, summed (Patch.n_free_params)
    gate_max_off: float = 0.02       # a gate may fire on at most this share of positions that must stay unchanged
    max_alpha: float = 10.0
    max_rank: int = 1024            # per changed tensor
    max_rel_change: float = 1.0     # |delta| / |W| per changed tensor
    fits: int = 8                   # fit_edit runs (each trains a low-rank change)
    fit_seconds: float = 300.0      # wall clock per fit
    max_fit_steps: int = 300
    max_fit_rank: int = 32
    weight_edits: bool = True       # False (--no-weight-edits): only steering and settings may fix it
    max_commits: int = 4            # commit_edit: edits in the committed stack at once
    allow: tuple = ()               # opt-in unlocks from the user (UNLOCKS): embed, unembed, final_norm, rescale, large


@dataclass
class Workbench:
    lm: LM
    res: Any                                   # the deep diagnosis
    ctx: Any                                   # playbook._Ctx (classifier, baseline, budget, fresh, pairs...)
    sitemap: Optional[S.SiteMap] = None
    budget: Budget = field(default_factory=Budget)
    log: Callable[[str], None] = lambda s: None
    directions: dict[str, dict] = field(default_factory=dict)
    edits: dict[str, E.Patch] = field(default_factory=dict)
    verified: dict[str, dict] = field(default_factory=dict)    # edit id -> the verify step that passed
    results: dict[str, dict] = field(default_factory=dict)     # edit id -> last verify step
    used: dict[str, int] = field(default_factory=lambda: {"verifications": 0, "screens": 0, "generations": 0,
                                                          "reads": 0, "computes": 0, "fits": 0})
    done: Optional[dict] = None                                 # {"submit": id, ...} or {"give_up": ...}
    pairsets: dict[str, list] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)          # request_capability
    gates: dict[str, Any] = field(default_factory=dict)         # harm_probe: gate id -> harmprobe.Gate
    proposals: dict[str, E.Patch] = field(default_factory=dict)  # unverified, saved for the user only
    lab: Optional[Lab] = None
    # Multi-stage surgery (commit_edit / revert_edit). `stack`: the committed edit ids, in order; their sum is applied
    # to the model in place, so every tool reads the repaired model. `built_on`: edit id -> the stack it was built on
    # (its change was computed from those weights; an edit with no entry was built on the original).
    stack: list = field(default_factory=list)
    built_on: dict = field(default_factory=dict)
    gate_built_on: dict = field(default_factory=dict)            # harm_probe gate id -> the stack it was calibrated on
    _orig: dict = field(default_factory=dict)                    # parameter name -> its original values (CPU copy)
    _orig_ver: int = 0                                           # the original weights' version number (LM._wver)
    _held: Optional[dict] = None                                 # the committed values of _orig's tensors (original())
    _held_ver: int = -1                                          # ... at this weight version
    _in_original: bool = False
    _released: bool = False
    _next_edit: int = 1

    def __post_init__(self) -> None:
        h = {"param": self._h_param, "params": self._h_params, "embed_rows": self._h_embed_rows,
             "unembed_rows": self._h_unembed_rows, "tok": self._h_tok, "detok": self._h_detok,
             "token": self._h_token, "readout": self._h_readout, "direction": self._h_direction,
             "info": self._h_info}
        self.lab = Lab(h, seconds=self.budget.compute_seconds)

    # -- plumbing ------------------------------------------------------------------------------------
    def _add_edit(self, patch: E.Patch) -> str:
        """Register an edit built on the current model; returns its id."""
        while f"e{self._next_edit}" in self.edits:
            self._next_edit += 1
        eid = f"e{self._next_edit}"
        self._next_edit += 1
        self.edits[eid] = patch
        self.built_on[eid] = tuple(self.stack)
        return eid

    def _model_is(self, stack: Any = None) -> str:
        st = tuple(self.stack if stack is None else stack)
        return "the original weights" if not st else "the original weights with " + " then ".join(st) + " committed"

    def _stale(self, eid: str, on: tuple) -> str:
        return (f"{eid} was built on {self._model_is(on)}, but the model is now {self._model_is()}: its change was "
                "computed from other weights, so it is not valid here. Rebuild it on the current model, or revert_edit "
                "back to the state it was built on")

    def _package(self, ids: list) -> E.Patch:
        """These edits, applied in this order, as one patch on the original weights. Every edit stores its change as
        numbers computed from the weights it was built on (a delta), so applying them in order adds exactly the
        changes that were committed: the sum is the stacked model."""
        parts = [self.edits[i] for i in ids]
        return E.Patch("; then ".join(f"{i} ({pt.label})" for i, pt in zip(ids, parts)), "stack",
                       {"parts": list(ids), "labels": [pt.label for pt in parts], "kinds": [pt.kind for pt in parts]},
                       [d for pt in parts for d in pt.deltas], untie=any(pt.untie for pt in parts))

    def candidate(self, eid: str) -> tuple:
        """(id, patch) of what the measuring tools and the judge see for edit `eid`: the committed stack plus the edit
        as ONE patch on the original weights ("e1+e3"). A stack patch from an earlier measurement ("e1+e3") is
        itself a candidate; a committed edit stands for the stack up to it, so only the top one can be named."""
        if eid not in self.edits:
            raise ValueError(f"unknown edit id {eid!r}; build one with build_edit")
        if self.edits[eid].kind == "stack" or not self.stack and self.built_on.get(eid, ()) == ():
            return eid, self.edits[eid]
        if eid in self.stack:
            if eid != self.stack[-1]:
                raise ValueError(f"{eid} is committed below {self.stack[-1]}: the candidate is the whole stack "
                                 f"({'+'.join(self.stack)}); name its top edit {self.stack[-1]}, or revert_edit {eid}'s "
                                 "successors first")
            ids = list(self.stack)
        else:
            on = tuple(self.built_on.get(eid, ()))
            if on != tuple(self.stack):
                raise ValueError(self._stale(eid, on))
            ids = list(self.stack) + [eid]
        key = "+".join(ids)
        if len(ids) == 1:
            return key, self.edits[key]
        if key not in self.edits:
            self.edits[key] = self._package(ids)
            self.built_on[key] = ()
        return key, self.edits[key]

    def stack_candidate(self) -> Optional[tuple]:
        """(id, patch) of the committed stack alone, or None when nothing is committed."""
        return self.candidate(self.stack[-1]) if self.stack else None

    @contextlib.contextmanager
    def original(self):
        """The original weights for the duration: the committed stack taken out, then put back bit for bit with its
        weight version (versions are never reused, so caches stay valid). The measuring tools, verify_edit and the
        judge run in here: they compare the stack plus a candidate with the model as it was given."""
        if not self.stack or self._in_original or self._released:
            yield
            return
        lm = self.lm
        ver = lm._wver
        if self._held is None or self._held_ver != ver:
            # The committed values, copied once per weight state: consecutive measuring calls reuse them. A version
            # that moved (a trial edit applied and restored) takes a new copy, so a stale one is never put back.
            self._held = self._snapshot(self._orig)
            self._held_ver = ver
        held = self._held
        self._in_original = True
        try:
            self._load(self._orig)
            lm._wver = self._orig_ver
            yield
        finally:
            self._in_original = False
            self._load(held)
            lm._wver = ver

    def _snapshot(self, names) -> dict:
        """CPU copies of these parameters' current values."""
        with torch.no_grad():
            return {n: E._param(self.lm, n).detach().to("cpu", copy=True) for n in names}

    def _load(self, values: dict) -> None:
        """Copy these values into the parameters (an exact restore)."""
        with torch.no_grad():
            for n, t in values.items():
                p = E._param(self.lm, n)
                p.copy_(t.to(p.device))

    def _reset_to(self, keep: list) -> None:
        """The original weights, then `keep` committed again in order (exactly the state they made before). All or
        nothing: if re-committing fails, the model and the stack are put back as they were before the call."""
        lm = self.lm
        before, prev, ver = self._snapshot(self._orig), list(self.stack), lm._wver
        try:
            self._load(self._orig)
            self.stack = []
            if not keep:
                lm._wver = self._orig_ver                # these are the original weights, bit for bit
                return
            E._bump(lm)
            for eid in keep:
                E.apply(lm, self.edits[eid])
                self.stack.append(eid)
        except BaseException:
            self._load(before)
            self.stack, lm._wver = prev, ver
            raise
        finally:
            self._held = None

    def _commit(self, eid: str) -> None:
        """Apply one edit in place and record it. All or nothing: if applying fails partway, every tensor it touched
        is put back and nothing is recorded."""
        patch = self.edits[eid]
        lm = self.lm
        E.fits(lm, patch)
        names = list(dict.fromkeys(d.name for d in patch.deltas))
        before, ver = self._snapshot(names), lm._wver
        for n in names:
            if n not in self._orig:                      # first touched: its original values, kept for the restore
                self._orig[n] = before[n]
        if not self.stack:
            self._orig_ver = ver
        try:
            E.apply(lm, patch)
        except BaseException:
            self._load(before)
            lm._wver = ver
            raise
        finally:
            self._held = None
        self.stack.append(eid)

    def release(self) -> None:
        """The run is over: the original weights back in the model (the stack stays on record). Whenever any were
        saved, not only while something is committed: a model is never left in a state nothing records."""
        if self._orig and not self._released:
            self._load(self._orig)
            self.lm._wver = self._orig_ver
        self._released = True

    def _spend(self, what: str) -> Optional[str]:
        cap = getattr(self.budget, what)
        if self.used[what] >= cap:
            return f"budget exhausted: {what} {self.used[what]}/{cap}"
        self.used[what] += 1
        return None

    def remaining(self) -> dict[str, int]:
        return {k: getattr(self.budget, k) - v for k, v in self.used.items()}

    def _examples(self) -> dict[str, Any]:
        c = self.res.contrast
        out = {e.name: e for e in list(c.failing) + list(c.passing)}
        for r in self.ctx.base.replies:
            out.setdefault(f"suite:{r.name}", r)
        if self.ctx.fresh:
            for e in self.ctx.fresh["failing"] + self.ctx.fresh["passing"]:
                out.setdefault(f"fresh:{e.name}", e)
        return out

    def _states(self, source: dict[str, Any]) -> list[S.State]:
        """States from a source: {"kind": "contrast"} | {"kind": "continuations"} |
        {"kind": "prompts", "failing": [names or texts], "working": [...]} read at the last prompt token."""
        k = source.get("kind", "contrast")
        if k == "contrast":
            return S.from_contrast(self.res.contrast)
        if k == "continuations":
            if not self.ctx.pairs:
                raise ValueError("no matched continuations (they exist for loop failures only)")
            return PR.states(self.ctx.pairs)
        if k == "pairs":
            return PR.states(self._pairs(source.get("id")))
        if k == "prompts":
            ex = self._examples()
            out = []
            for side, key in ((True, "failing"), (False, "working")):
                for item in source.get(key) or []:
                    if item in ex:
                        e = ex[item]
                        ids = getattr(e, "decision_ids", None)
                        ids = ids if ids is not None else (e.prompt_ids if hasattr(e, "prompt_ids") else e.ids)
                    else:
                        ids = self.lm.encode([{"role": "user", "content": str(item)}],
                                             **(self.ctx.clf.template_kwargs or {}))
                    out.append(S.State(str(item)[:40], ids, [ids.shape[1] - 1], side))
            return out
        raise ValueError(f"unknown source kind {k!r}")

    def _direction(self, did: str, depth: Optional[int] = None) -> torch.Tensor:
        d = self.directions.get(did)
        if d is None:
            raise ValueError(f"unknown direction id {did!r}; make one with make_direction")
        if depth is not None and d.get("by_depth") is not None:
            return d["by_depth"][depth]
        return d["vector"]

    # -- workspace helpers (what `compute` code can call; every tensor handed out is a copy) ---------
    def _named(self) -> dict[str, torch.nn.Parameter]:
        return dict(self.lm.model.named_parameters())

    def _forbidden(self) -> set[str]:
        """Parameters no free-form edit may touch (the paper's harness refused the same): the input
        embedding, the output layer and the final norm, unless the user unlocked them (`allow`).
        With tied embeddings the input and output rows are one tensor: it needs both unlocks."""
        lm = self.lm
        allow = set(self.budget.allow)
        emb = lm.model.get_input_embeddings().weight
        out = lm.model.get_output_embeddings()
        tied = out is not None and out.weight is emb
        ban = set()
        if "embed" not in allow or (tied and "unembed" not in allow):
            ban.add(id(emb))
        if out is not None and ("unembed" not in allow or (tied and "embed" not in allow)):
            ban.add(id(out.weight))
        if lm.final_norm is not None and getattr(lm.final_norm, "weight", None) is not None \
                and "final_norm" not in allow:
            ban.add(id(lm.final_norm.weight))
        return {n for n, p in self._named().items() if id(p) in ban}

    def _h_param(self, name: str) -> torch.Tensor:
        p = self._named().get(name)
        if p is None:
            raise LabError(f"no parameter {name!r}; list them with params('pattern')")
        if p.numel() > MAX_ELEMENTS:
            raise LabError(f"{name} has {p.numel():,} elements; use embed_rows / unembed_rows for vocabulary rows")
        return p.detach().float().cpu().clone()

    def _h_params(self, pattern: str = "") -> list:
        return [[n, list(p.shape)] for n, p in self._named().items() if pattern in n][:400]

    def _ids(self, tokens: Any) -> list[int]:
        if isinstance(tokens, str):
            return self.lm.tok(tokens, add_special_tokens=False)["input_ids"]
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.flatten().tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        if len(tokens) > 4096:
            raise LabError("at most 4096 tokens at a time")
        return [int(t) for t in tokens]

    def _h_embed_rows(self, tokens: Any) -> torch.Tensor:
        return self.lm.model.get_input_embeddings().weight.detach()[self._ids(tokens)].float().cpu().clone()

    def _h_unembed_rows(self, tokens: Any) -> torch.Tensor:
        return self.lm.unembed.weight.detach()[self._ids(tokens)].float().cpu().clone()

    def _h_tok(self, text: str) -> list:
        return self._ids(str(text))

    def _h_detok(self, ids: Any) -> str:
        return self.lm.decode(self._ids(ids))

    def _h_token(self, i: int) -> str:
        return self.lm.token_str(int(i))

    @torch.no_grad()
    def _h_readout(self, vec: torch.Tensor, k: int = 10) -> list:
        """Top tokens of a residual-stream vector read through the final norm and unembedding."""
        lg = self.lm.logits_of(vec.float().reshape(1, -1))[0]
        top = torch.topk(lg, min(int(k), 50))
        return [[self.lm.token_str(int(i)), round(float(v), 3)] for v, i in zip(top.values, top.indices)]

    def _h_direction(self, did: str, depth: Optional[int] = None) -> torch.Tensor:
        try:
            return self._direction(did, depth).float().cpu().clone()
        except ValueError as e:
            raise LabError(str(e)) from None

    def _h_info(self) -> dict:
        lm = self.lm
        return {"n_layers": lm.n_layers, "d_model": lm.d_model, "d_ff": lm.down_proj(0).weight.shape[1],
                "n_heads": lm.n_heads, "head_layers": list(lm.head_layers), "tied_embeddings": E.tied(lm),
                "eot_ids": sorted(lm.eot_ids), "frozen_params": sorted(self._forbidden()),
                "mlp_names": {k: n for k, n in (("down", "down_proj"), ("up", "up_proj"), ("gate", "gate_proj"))},
                "layer0_params": [n for n in self._named() if ".0." in n or n.endswith(".0")][:40]}

    def _vector(self, ref: Any) -> torch.Tensor:
        """A d_model vector: a direction id ('d1', or 'd1@12' for its version after depth 12) or a
        workspace variable name."""
        if isinstance(ref, str) and (ref.split("@")[0] in self.directions):
            did, _, dep = ref.partition("@")
            v = self._direction(did, int(dep) if dep else None)
        else:
            v = self.lab.get(str(ref))
        v = torch.as_tensor(v).float().flatten()
        if v.numel() != self.lm.d_model:
            raise ValueError(f"{ref!r} has {v.numel()} elements, not d_model {self.lm.d_model}")
        return v

    def _interventions(self, spec: Optional[dict]) -> dict[str, Any]:
        """{"zero_neurons": [[layer, j]], "zero_heads": [[layer, h]], "ablate": vector ref,
        "add": {"vector": ref, "alpha": a, "layers": [...]}} -> lm.intervention_hooks kwargs."""
        if not spec:
            return {}
        kw: dict[str, Any] = {}
        if spec.get("zero_neurons"):
            kw["zero_neurons"] = [(int(a), int(b)) for a, b in spec["zero_neurons"]]
        if spec.get("zero_heads"):
            kw["zero_heads"] = [(int(a), int(b)) for a, b in spec["zero_heads"]]
        if spec.get("ablate"):
            kw["ablate"] = self._vector(spec["ablate"])
        if spec.get("add"):
            a = spec["add"]
            kw["add"] = (self._vector(a["vector"]), float(a.get("alpha", 1.0)), [int(i) for i in a["layers"]])
        if spec.get("gate"):
            if spec["gate"] not in self.gates:
                raise ValueError(f"unknown gate {spec['gate']!r}: make one with harm_probe")
            kw["gated"] = self.gates[spec["gate"]]
        unknown = set(spec) - {"zero_neurons", "zero_heads", "ablate", "add", "gate"}
        if unknown:
            raise ValueError(f"unknown interventions {sorted(unknown)}")
        return kw

    def _text_ids(self, item: Any) -> tuple[torch.Tensor, int]:
        """(ids, index of the first continuation token, or len(ids) when there is none). An item is an
        example name, a prompt text (chat template), or {"prompt" | "messages" | "raw", "continuation"?}."""
        ex = self._examples()
        tk = self.ctx.clf.template_kwargs or {}
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            raise ValueError("each item is a string or an object with prompt / messages / raw")
        p = item.get("prompt")
        if p is not None and p in ex:
            e = ex[p]
            ids = e.prompt_ids if hasattr(e, "prompt_ids") else e.ids
        elif p is not None:
            ids = self.lm.encode([{"role": "user", "content": str(p)}], **tk)
        elif item.get("messages"):
            ids = self.lm.encode(list(item["messages"]), **tk)
        elif item.get("raw") is not None:
            ids = torch.tensor([self._ids(str(item["raw"]))], dtype=torch.long)
        else:
            raise ValueError("an item needs prompt, messages or raw")
        P0 = ids.shape[1]
        cont = item.get("continuation")
        if cont:
            ids = torch.cat([ids, torch.tensor([self._ids(str(cont))], dtype=torch.long)], 1)
        if ids.shape[1] > 4096:
            raise ValueError("a text may have at most 4096 tokens")
        return ids, P0

    @staticmethod
    def _cap(x: Any, n: int) -> int:
        return max(1, min(int(x), n))

    def _max_new_cap(self) -> int:
        """generate may run as long as verify_edit judges: 512 tokens, or the longer length the target check
        uses when the failure starts past the cap (playbook._longer_reference). Otherwise the agent could not
        see what its edit is judged on."""
        ref = self.res.contrast.reference if self.res.contrast else None
        return max(512, ref.max_new if ref else 0)

    # -- tools ---------------------------------------------------------------------------------------
    def compute(self, code: str, why: str = "") -> dict[str, Any]:
        """Run workspace code (lab.py). Variables persist across calls."""
        if (err := self._spend("computes")):
            return {"error": err}
        try:
            out = self.lab.run(code)
        except LabError as e:
            return {"error": str(e), "variables": self.lab.summary()}
        return {**out, "computes_left": self.remaining()["computes"]}

    @torch.no_grad()
    def read(self, items: list, capture: Optional[dict] = None, edit_id: Optional[str] = None,
             intervene: Optional[dict] = None, store_as: Optional[str] = None, why: str = "") -> dict[str, Any]:
        """Teacher-forced pass over up to 64 texts; per-token log-probs of each continuation, the top
        next tokens at its end, and (capture) residuals / MLP activations / attention outputs at chosen
        positions, stored in the workspace as `store_as` (a list with one dict per text)."""
        if (err := self._spend("reads")):
            return {"error": err}
        lm = self.lm
        items = list(items)[:64]
        cap = capture or {}
        depths = [int(d) for d in cap.get("resid", [])]
        mlps = [int(i) for i in cap.get("mlp_act", [])]
        attns = [int(i) for i in cap.get("attn", [])]
        where = cap.get("positions", "continuation")
        kw = self._interventions(intervene)
        patch = self._edit(edit_id) if edit_id else None
        name = store_as or f"r{self.used['reads']}"
        ctx = E.applied(lm, patch) if patch is not None else contextlib.nullcontext()
        rows, summary = [], []
        with ctx:
            for it in items:
                ids, P0 = self._text_ids(it)
                T = ids.shape[1]
                if where == "last":
                    pos = [T - 1]
                elif where == "all":
                    pos = list(range(T))
                elif where == "continuation":
                    pos = list(range(P0, T)) if P0 < T else [T - 1]
                else:
                    pos = [int(p) % T for p in where]
                k = T - P0 + 1 if P0 < T else 1
                extra = {"logits_to_keep": k} if lm._keeps_logits else {}
                with lm.steer(**kw):
                    x = ids.to(lm.device)
                    lg = lm.model(input_ids=x, attention_mask=torch.ones_like(x), **extra).logits[0, -k:].float()
                lp = lg.log_softmax(-1)
                conts = ids[0, P0:].tolist()
                tok_lp = lp[:-1].gather(1, torch.tensor(conts, device=lp.device)[:, None])[:, 0].cpu() \
                    if conts else torch.zeros(0)
                nxt = torch.topk(lp[-1], 5)
                row: dict[str, Any] = {"ids": ids[0].clone(), "prompt_len": P0, "positions": pos,
                                       "logprobs": tok_lp}
                if depths or mlps or attns:
                    got = lm.seq_capture(ids, depths=depths, mlp_layers=mlps, attn_layers=attns, positions=pos, **kw)
                    row.update({"resid": got["resid"], "mlp_act": got["mlp_act"], "attn": got["attn"]})
                rows.append(row)
                summary.append({"tokens": T, "continuation_tokens": len(conts),
                                "continuation_mean_logprob": round(float(tok_lp.mean()), 4) if conts else None,
                                # per token only for a few texts (all of them are in the workspace either way)
                                **({"continuation_logprobs": [round(float(v), 3) for v in tok_lp[:32]]}
                                   if len(items) <= 16 else {}),
                                "next_top5": [[lm.token_str(int(i)), round(float(v), 3)]
                                              for v, i in zip(nxt.values.exp(), nxt.indices)]})
        self.lab.put(name, rows)
        return {"stored_as": name, "texts": summary, "reads_left": self.remaining()["reads"],
                "note": f"{name}[i] has ids, prompt_len, positions, logprobs and the captured resid / mlp_act / attn "
                        "dicts (keyed by depth or layer, rows = positions)"}

    def make_pairs(self, items: list, name: Optional[str] = None) -> dict[str, Any]:
        """Matched continuations of the agent's own: [{prompt, bad, good}] (the same prompt, the failing
        and a working continuation). Usable as a source ({"kind": "pairs", "id": ...}), in
        screen_edit and in screen_layers."""
        out = []
        for it in list(items)[:32]:
            ids, P0 = self._text_ids({k: v for k, v in it.items() if k not in ("bad", "good")})
            if "bad" not in it or "good" not in it:
                raise ValueError("each item needs prompt (or messages / raw), bad (the failing continuation) and good "
                                 "(a working continuation of the same prompt). Pairs are for failures in a continuation "
                                 "(loops); for a blank, refusal or language failure use make_direction with "
                                 "{kind: contrast}")
            bad, good = self._ids(str(it["bad"])), self._ids(str(it["good"]))
            if len(bad) < 2 or len(good) < 2:
                raise ValueError("each continuation needs at least 2 tokens. Pairs are for failures in a continuation "
                                 "(loops); a blank reply has no failing continuation: use the contrast source "
                                 "({kind: contrast}) with make_direction instead")
            out.append(PR.Pair(str(it.get("prompt", it.get("raw", "pair")))[:40], ids, bad, good))
        pid = name or f"p{len(self.pairsets) + 1}"
        self.pairsets[pid] = out
        sel = PR.selectivity(self.lm, out)
        return {"pairs_id": pid, "pairs": len(out), "baseline": {k: round(v, 4) for k, v in sel.items()}}

    def _pairs(self, pid: Optional[str]) -> list:
        if pid is None:
            return list(self.ctx.pairs or [])
        if pid not in self.pairsets:
            raise ValueError(f"unknown pairs id {pid!r}; make one with make_pairs")
        return self.pairsets[pid]

    def screen_layers(self, pairs_id: Optional[str] = None, alpha: float = 3.0, keep: int = 2) -> dict[str, Any]:
        """The held-out layer screen (playbook.screened_candidate) on a pair set: returns the per-layer
        selectivity and, if confirmed on held-out pairs, an edit id."""
        if (err := self._spend("screens")):
            return {"error": err}
        prs = self._pairs(pairs_id)
        cd, info = P.screened_candidate(self.lm, prs, alpha=float(alpha), keep=self._cap(keep, 6))
        out = {k: v for k, v in info.items()}
        if cd is not None:
            out["edit_ids"] = {}
            for label, build in cd.ladder:                 # one edit per strength, none applied yet
                out["edit_ids"][label] = self._add_edit(build())
        return out

    def gate_sweep(self, write_direction_id: str, read_source: Optional[dict] = None,
                   layers: Optional[list] = None, silent: Optional[list] = None, keep: int = 2) -> dict[str, Any]:
        """The playbook's gated-neuron search: a gate plan at every layer (or the given ones), then, at the `keep`
        layers whose gate fires on most failing positions while staying off where it must, one edit per ladder
        strength (playbook.GATE_LADDER x the strength that cancels the failure's gap). The edits are built, not
        applied: screen, verify and measure them like any other."""
        if not self.budget.weight_edits:
            return {"error": NO_WEIGHT_EDITS}
        if (err := self._spend("screens")):
            return {"error": err}
        lm = self.lm
        src = read_source or {"kind": "contrast"}
        if isinstance(src, str):
            src = {"kind": src}
        sts = self._states(src)
        bad, good = [s for s in sts if s.bad], [s for s in sts if not s.bad]
        if not bad or not good:
            return {"error": "the read source needs failing and working states"}
        silent_names = silent or ["suite"]
        quiet: list[torch.Tensor] = []
        for nm in silent_names:
            quiet += [P._full_texts([r])[0] for r in self.ctx.base.replies
                      if (nm == "suite" and r.kind != "safety") or (nm == "harmful" and r.kind == "safety"
                                                                     and r.refused) or nm == f"suite:{r.name}"]
        ref = P._full_texts(self.ctx.base.replies)[:8]
        todo = sorted({int(x) for x in layers} if layers else range(1, lm.n_layers - 1))
        todo = [L for L in todo if 0 <= L < lm.n_layers - 1]
        rows, plans = [], {}
        for L in todo:
            plan = S.gate_plan(lm, L, bad, good, quiet, ref, max_off=self.budget.gate_max_off)
            rows.append({"layer": L, "ok": bool(plan["ok"]),
                         **{k: round(plan[k], 4) for k in ("fire_bad", "fire_off", "margin") if k in plan},
                         **({} if plan["ok"] else {"why": plan.get("why")})})
            if plan["ok"]:
                plans[L] = plan
        best = sorted(plans, key=lambda L: -(plans[L]["fire_bad"] - plans[L]["fire_off"]))[:self._cap(keep, 4)]
        out: dict[str, Any] = {"layers": rows, "viable": sorted(plans), "edit_ids": {}}
        if not best:
            out["says"] = ("no layer has a gate that fires on the failing positions and stays off on the silent "
                           "texts: the failure is not readable by one neuron's input at any layer swept")
            return out
        for L in best:
            plan = plans[L]
            w = self._direction(write_direction_id, L + 1)
            neuron = S.quiet_neurons(lm, L, ref[:6], k=1)[0]
            s0 = P.gate_strength(lm, L, bad, good, w, plan["margin"])
            for k in P.GATE_LADDER:
                strength = k * s0
                label = f"gated neuron {L}.{neuron} x{k:g}"
                if abs(strength) > self.budget.max_alpha:
                    out["edit_ids"][label] = f"skipped: strength {strength:.3g} is beyond the allowed {self.budget.max_alpha}"
                    continue
                patch = E.gate(lm, L, neuron, plan["read"], w, threshold=plan["threshold"],
                               up_scale=plan["up_scale"], mean_dir=plan["mean_dir"], mean_proj=plan["mean_proj"],
                               strength=strength, label=label, source=f"gated neuron {L}.{neuron}")
                patch.spec["plan"] = {kk: v for kk, v in plan.items() if not isinstance(v, torch.Tensor)}
                out["edit_ids"][label] = self._add_edit(patch)
        return out

    def weight_scan(self, threshold: Optional[float] = None) -> dict[str, Any]:
        """Rows or columns of the layers' weights far larger than the rest of their matrix (surgery/weightscan.py:
        no healthy model measured has one above 13.4x its matrix's median norm; damage leaves tens to hundreds),
        and for each finding three restoring edits, built not applied. Reads weights only: no generation, no budget."""
        from ..surgery import weightscan as WS

        th = WS.THRESHOLD if threshold is None else max(WS.RELATIVE, float(threshold))
        found = WS.scan(self.lm, threshold=th)
        out: dict[str, Any] = {"threshold": th, "findings": [], "says": WS.summary(found)}
        if found and not self.budget.weight_edits:
            out["note"] = NO_WEIGHT_EDITS + " (findings only: no edits built)"
        for f in found[:4]:
            row = {**f.as_dict(), "edit_ids": {}}
            out["findings"].append(row)
            if not self.budget.weight_edits:
                continue
            # The playbook's own candidates (like gate_sweep's gates), not free-form deltas: build_edit's size
            # limits do not apply; shrinking damaged entries always changes the matrix by less than its own norm.
            for strength, build in WS.repairs(self.lm, f):
                row["edit_ids"][strength] = self._add_edit(build())
        return out

    def request_capability(self, capability: str, category: str, what_it_would_do: str, why_needed: str,
                           evidence: str = "", proposal: Optional[dict] = None) -> dict[str, Any]:
        """The repair needs something wsr does not let the agent do. Recorded for the report (and for wsr's
        developers); if wsr has an opt-in for it, the reply names the flag; a `proposal` (a free-form edit
        spec over workspace variables) is saved as an UNVERIFIED patch the user can check with
        `wsr repair --check`, never applied or shipped by the run."""
        if category not in CATEGORIES:
            return {"error": f"category must be one of {sorted(CATEGORIES)}"}
        req: dict[str, Any] = {"capability": str(capability)[:200], "category": category,
                               "what_it_would_do": str(what_it_would_do)[:2000], "why_needed": str(why_needed)[:3000],
                               "evidence": str(evidence)[:3000]}
        unlock = UNLOCKS.get(category)
        if unlock and all(u in self.budget.allow for u in unlock[0]):
            req["unlock"] = "already unlocked in this run: build and verify it with build_edit"
        elif unlock:
            req["unlock"] = unlock[1]
        if proposal:
            try:
                patch = self._free_edit(dict(proposal, kind=proposal.get("kind", "delta")), proposal=True)
            except (ValueError, KeyError, LabError) as e:
                return {"error": f"proposal not built: {e}"}
            if self.stack:                                # computed on the committed stack: saved with it, as one
                _, base = self.stack_candidate()          # patch on the original weights the user has
                patch = E.Patch(f"{base.label}; then {patch.label}", "stack",
                                {"parts": list(self.stack) + ["proposal"], "labels": [base.label, patch.label],
                                 "kinds": ["stack", patch.kind]}, list(base.deltas) + list(patch.deltas),
                                untie=base.untie or patch.untie)
            patch.checks = {"verified": False, "proposed_by": "agent", "request": req["capability"]}
            pid = f"proposal{len(self.proposals) + 1}"
            self.proposals[pid] = patch
            req["proposal"] = {"id": pid, "label": patch.label, "weights_changed": patch.n_weights_changed,
                               "params": sorted({d.name for d in patch.deltas})}
        self.requests.append(req)
        return {"recorded": True, **({"unlock": req["unlock"]} if "unlock" in req else {}),
                **({"proposal_id": req["proposal"]["id"]} if "proposal" in req else {}),
                "note": "keep working on what you can do, or give_up citing this request; the report lists it "
                        "for the user"}

    def harm_probe(self, depth: Optional[int] = None, up_sd: float = 2.0, margin_up: float = 0.5,
                   margin_down: float = 0.0, read: Optional[list] = None) -> dict[str, Any]:
        """The model's own reading of harm (deep/harmprobe.py, calibrated once per model and cached) and a gate built
        on it: a gate id usable as intervene {"gate": id} in generate / read, and (in wsr program) as steer_eval
        vector with mode "gated". read: up to 8 prompts whose probe logit to report (> 0 reads harmful)."""
        from ..deep import harmprobe as HP

        try:
            probe = HP.calibrate(self.lm, template_kwargs=self.ctx.template_kwargs,
                                 extra=getattr(self.ctx, "harm_extra", None) or None)
        except ValueError as e:
            return {"error": str(e)}
        if not 0 <= float(up_sd) <= 6 or not 0 <= float(margin_up) <= 10 or not 0 <= float(margin_down) <= 10:
            return {"error": "up_sd must be in [0, 6], margin_up and margin_down in [0, 10]"}
        info = {k: v for k, v in probe.as_dict().items() if k not in ("levels",)}
        info["summary"] = probe.summary()
        out: dict[str, Any] = {"probe": info}
        if not probe.viable:
            out["warning"] = ("the diagnosis found this reading unusable: " + probe.why + ". A gate built on it is "
                              "not expected to help")
        try:
            g = probe.gate(up_sd=float(up_sd), margin_up=float(margin_up), margin_down=float(margin_down),
                           depth=depth)
        except ValueError as e:
            return {**out, "error": str(e)}
        gid = f"h{len(self.gates) + 1}"
        self.gates[gid] = g
        est = probe.estimate(g)
        if est is not None:
            out["estimate"] = est
        self.gate_built_on[gid] = tuple(self.stack)
        out["gate_id"] = gid
        if self.stack:
            out["calibrated_on"] = (self._model_is() + ": steer_eval measures the gate on top of this stack, against "
                                    "the original weights")
        out["gate"] = g.as_dict()
        if read:
            if not isinstance(read, list) or len(read) > 8:
                return {**out, "error": "read: up to 8 prompts"}
            out["logits"] = []
            for p in read:
                ids, _ = self._text_ids(p)
                z = probe.logit(self.lm.run(ids, capture=("resid",)).resid, g.depth)
                out["logits"].append({"prompt": p if isinstance(p, str) else str(p)[:120], "logit": round(z, 3),
                                      "reads": "harmful" if z > 0 else "safe"})
        return out

    def note(self, text: str) -> dict[str, Any]:
        """Keep a finding in the run's notebook (saved with the transcript and shown in the report)."""
        self.notes.append(str(text)[:2000])
        return {"notes": len(self.notes)}

    def _check_delta(self, name: str, U: torch.Tensor, V: torch.Tensor) -> Optional[str]:
        named = self._named()
        if name not in named:
            return f"no parameter {name!r}"
        if name in self._forbidden():
            return (f"{name} is frozen (input embedding, output layer and final norm cannot be edited freely). If the "
                    "evidence says the fix needs it, call request_capability: the user can unlock it")
        W = named[name].detach().float().cpu()
        if not (torch.isfinite(U).all() and torch.isfinite(V).all()):
            return "the change contains NaN or inf"
        if W.dim() == 2 and (U.shape[0] != W.shape[0] or V.shape[0] != W.shape[1]):
            return f"shapes: U {tuple(U.shape)}, V {tuple(V.shape)} do not fit {name} {tuple(W.shape)}"
        if W.dim() == 1 and (U.shape[0] != W.shape[0] or V.shape != (1, U.shape[1])):
            return f"a 1-D parameter takes U [{W.shape[0]}, r] and V [1, r]"
        large = "large" in self.budget.allow
        if U.shape[1] > self.budget.max_rank and not large:
            return f"rank {U.shape[1]} over the cap {self.budget.max_rank}"
        d = E.Delta(name, U, V)
        dn, wn = d.norm(), float(W.norm())
        if wn > 0 and dn / wn > self.budget.max_rel_change and not large:
            return f"the change is {dn / wn:.2f}x the tensor's norm, over the cap {self.budget.max_rel_change:g}"
        if "rescale" in self.budget.allow:
            return None
        if W.dim() == 2 and wn > 0 and dn > 0:
            inner = float(torch.trace(U.T @ W @ V))          # <U V^T, W>
            if abs(inner) / (dn * wn) > 0.999:
                return "the change only rescales the whole tensor (a global knob, not a targeted edit)"
        if W.dim() == 1 and dn > 0 and abs(float((U @ V.T)[:, 0] @ W)) / (dn * wn + 1e-12) > 0.999:
            return "the change only rescales the whole tensor (a global knob, not a targeted edit)"
        return None

    def _free_edit(self, spec: dict[str, Any], *, proposal: bool = False) -> E.Patch:
        """delta {changes: [{param, U, V}]}  |  set_columns {param, cols, values}  |  set_rows {param, rows,
        values}; U / V / values are workspace variable names."""
        kind = spec["kind"]
        deltas: list[E.Delta] = []
        changes = spec.get("changes") or [spec]
        for ch in changes:
            name = ch["param"]
            W = self._named()[name].detach().float().cpu() if name in self._named() else None
            if W is None:
                raise ValueError(f"no parameter {name!r}")
            if kind == "delta":
                U = torch.as_tensor(self.lab.get(ch["U"])).float()
                V = torch.as_tensor(self.lab.get(ch["V"])).float() if ch.get("V") else torch.ones(1, 1)
                U = U[:, None] if U.dim() == 1 else U
                V = V[:, None] if V.dim() == 1 else V
                if W.dim() == 1:
                    V = V.reshape(1, -1).expand(1, U.shape[1]).clone()
            elif kind in ("set_columns", "set_rows"):
                if W.dim() != 2:
                    raise ValueError(f"{kind} needs a matrix")
                idx = [int(i) for i in ch["cols" if kind == "set_columns" else "rows"]]
                vals = torch.as_tensor(self.lab.get(ch["values"])).float()
                if kind == "set_columns":
                    vals = vals[:, None] if vals.dim() == 1 else vals
                    if vals.shape != (W.shape[0], len(idx)):
                        raise ValueError(f"values must be [{W.shape[0]}, {len(idx)}] (one column per index)")
                    U = vals - W[:, idx]
                    V = torch.zeros(W.shape[1], len(idx))
                    V[idx, torch.arange(len(idx))] = 1.0
                else:
                    vals = vals[None] if vals.dim() == 1 else vals
                    if vals.shape != (len(idx), W.shape[1]):
                        raise ValueError(f"values must be [{len(idx)}, {W.shape[1]}] (one row per index)")
                    U = torch.zeros(W.shape[0], len(idx))
                    U[idx, torch.arange(len(idx))] = 1.0
                    V = (vals - W[idx]).T.contiguous()
            else:
                raise ValueError(f"unknown edit kind {kind!r}")
            why = self._check_delta(name, U, V)
            if why and proposal:                           # a proposal may cross the guards, not the shapes
                why = why if ("shape" in why or "NaN" in why or "takes U" in why or "no parameter" in why) else None
            if why:
                raise ValueError(why)
            deltas.append(E.Delta(name, U.contiguous(), V.contiguous()))
        return E.Patch(spec.get("label") or f"agent {kind} on " + ", ".join(sorted({d.name for d in deltas})),
                       f"agent_{kind}", {"why": spec.get("why", "")}, deltas)
    def list_examples(self) -> dict[str, Any]:
        c = self.res.contrast
        row = lambda e: {"name": e.name, "prompt": e.messages[-1].get("content", "")[:200],  # noqa: E731
                         "reply": e.text[:200], "groups": sorted(e.groups)}
        out = {"failure_kind": c.kind, "failing": [row(e) for e in c.failing], "working": [row(e) for e in c.passing],
               "regression_suite": [{"name": f"suite:{r.name}", "kind": r.kind, "prompt": r.prompt[:120],
                                     "reply": r.text[:120], "groups": sorted(r.groups), "refused": r.refused}
                                    for r in self.ctx.base.replies]}
        if self.ctx.fresh:
            out["fresh_variants"] = {"failing": [f"fresh:{e.name}" for e in self.ctx.fresh["failing"]],
                                     "working": [f"fresh:{e.name}" for e in self.ctx.fresh["passing"]],
                                     "note": "held out for the final check; prefer not to tune on them"}
        return out

    def _messages(self, prompt: Any) -> list[dict]:
        ex = self._examples()
        if isinstance(prompt, str) and prompt in ex and hasattr(ex[prompt], "messages"):
            return ex[prompt].messages
        if isinstance(prompt, str) and prompt in ex and isinstance(getattr(ex[prompt], "prompt", None), str):
            return [{"role": "user", "content": ex[prompt].prompt}]        # a regression-suite reply
        if isinstance(prompt, dict) and prompt.get("messages"):
            return list(prompt["messages"])
        return [{"role": "user", "content": str(prompt)}]

    def _row(self, gen: list[int], msgs: list[dict]) -> dict[str, Any]:
        ended = bool(gen) and gen[-1] in self.lm.eot_ids
        text = self.lm.tok.decode(gen[:-1] if ended else gen, skip_special_tokens=True)
        groups = self.ctx.clf.groups_of(text, gen, request=msgs[-1].get("content"))
        return {"reply": text[:600], "groups": sorted(groups), "tokens": len(gen), "ended": ended}

    def generate(self, prompt: Any = None, edit_id: Optional[str] = None, max_new: int = 64, samples: int = 1,
                 temperature: Optional[float] = None, top_p: Optional[float] = None, seed: int = 0,
                 compare: bool = False, intervene: Optional[dict] = None,
                 repetition_penalty: Optional[float] = None, prompts: Optional[list] = None) -> dict[str, Any]:
        """Replies (greedy, or `samples` sampled ones seeded per sample), classified by the triage
        detectors; with `compare`, the untouched model and the model with the edit and / or interventions, on the
        same seeds (a compare with only interventions, a gate for example, is a real comparison too). `prompts` (up to 8):
        all of them in one call, batched to the free GPU memory (LM.generate_batch plans the batch and halves
        it on an out-of-memory error); the original and the edit run in the same chunks, so they stay paired."""
        n = self._cap(samples, 8)
        # (edit id, interventions on?) per pass: with compare, the untouched model first, then the changed one
        runs = [(None, False), (edit_id, True)] if (compare and (edit_id or intervene)) else [(edit_id, True)]
        if prompts is not None:
            if not isinstance(prompts, list) or not 1 <= len(prompts) <= 8:
                return {"error": "prompts: a list of 1-8 prompts (example names, texts or {messages})"}
        elif prompt is None:
            return {"error": "give prompt (one) or prompts (a list of up to 8)"}
        sample = temperature is not None and float(temperature) > 0
        gen_kw = {"do_sample": True, "temperature": float(temperature), "top_p": float(top_p or 1.0)} if sample else {}
        if repetition_penalty is not None:
            if not 1.0 <= float(repetition_penalty) <= 1.5:
                return {"error": "repetition_penalty must be in [1.0, 1.5]"}
            gen_kw["repetition_penalty"] = float(repetition_penalty)
        for _ in range((len(prompts) if prompts is not None else 1) * (n if sample else 1) * len(runs)):
            if (err := self._spend("generations")):
                return {"error": err}
        kw = self._interventions(intervene)
        if prompts is not None:
            return self._generate_many(prompts, runs, n if sample else 1, int(seed), max_new, gen_kw, kw)
        msgs = self._messages(prompt)
        clf = self.ctx.clf
        ids = self.lm.encode(msgs, **(clf.template_kwargs or {}))
        out: dict[str, Any] = {}
        for eid, on in runs:
            ctx = E.applied(self.lm, self._edit(eid)) if eid else contextlib.nullcontext()
            rows = []
            with ctx, self.lm.steer(**(kw if on else {})):
                for k in range(n if sample else 1):
                    torch.manual_seed(int(seed) + k)
                    gen = self.lm.generate(ids, self._cap(max_new, self._max_new_cap()), **dict(gen_kw))
                    ended = bool(gen) and gen[-1] in self.lm.eot_ids
                    text = self.lm.tok.decode(gen[:-1] if ended else gen, skip_special_tokens=True)
                    groups = clf.groups_of(text, gen, request=msgs[-1].get("content"))
                    rows.append({"reply": text[:600], "groups": sorted(groups), "tokens": len(gen), "ended": ended})
            out["edited" if (eid or (on and kw)) else self._base_name()] = rows if len(rows) > 1 else rows[0]
        out["generations_left"] = self.remaining()["generations"]
        if self.stack:
            out["model"] = self._model_is()
        return out

    def _generate_many(self, prompts: list, runs: list, k: int, seed: int, max_new: int, gen_kw: dict,
                       kw: dict) -> dict[str, Any]:
        msgs = [self._messages(p) for p in prompts]
        tk = self.ctx.clf.template_kwargs or {}
        ids = [self.lm.encode(m, **tk) for m in msgs for _ in range(k)]
        seeds = [seed + j for _ in msgs for j in range(k)]
        mx = self._cap(max_new, self._max_new_cap())
        hooks = (lambda: self.lm.steer(**kw)) if kw else None
        results = [{"prompt": p if isinstance(p, str) else "messages"} for p in prompts]
        for eid, on in runs:
            ctx = E.applied(self.lm, self._edit(eid)) if eid else contextlib.nullcontext()
            with ctx:
                gens = self.lm.generate_batch(ids, mx, seeds=seeds, hooks=hooks if on else None, **dict(gen_kw))
            for i, m in enumerate(msgs):
                rows = [self._row(g, m) for g in gens[i * k:(i + 1) * k]]
                results[i]["edited" if (eid or (on and kw)) else self._base_name()] = rows if len(rows) > 1 else rows[0]
        return {"results": results, "generations_left": self.remaining()["generations"],
                **({"model": self._model_is()} if self.stack else {})}

    def _base_name(self) -> str:
        """What generate calls the pass without the edit: "original", or "committed" once a stack is committed (the
        model every tool reads is then the repaired one)."""
        return "committed" if self.stack else "original"

    def make_direction(self, source: dict[str, Any]) -> dict[str, Any]:
        sts = self._states(source)
        sm = S.locate(self.lm, sts, source=source.get("kind", "contrast"))
        if sm is None:
            return {"error": "need states on both sides (failing and working)"}
        did = f"d{len(self.directions) + 1}"
        self.directions[did] = {"by_depth": sm.dirs, "vector": sm.dirs[sm.sites[0]["layer"] + 1], "sitemap": sm,
                                "source": source}
        cos = {}
        if self.res.directions is not None:
            u = self.res.directions.unit()
            cos["diagnosis_contrast_direction"] = round(float(u @ sm.dirs[self.res.directions.best_depth]), 3)
        kv = getattr(self.res, "known_vectors", None)
        if kv is not None:
            for name, by in kv.by_name.items():
                vals = []
                for k, v in sm.dirs.items():
                    try:
                        vals.append(abs(float((by[k] / (by[k].norm() + 1e-12)) @ v)))
                    except (IndexError, KeyError, TypeError):
                        pass
                if vals:
                    cos[f"known:{name}"] = round(max(vals), 3)
        return {"direction_id": did, "failing_states": sm.n_bad, "working_states": sm.n_good,
                "vector_is": f"failing minus working after layer {sm.sites[0]['layer']} (per-depth versions kept)",
                "sites": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} for s in sm.sites],
                "top_neurons": {str(i): [[j, round(v, 3)] for j, v in ns[:12]] for i, ns in sm.neurons.items()},
                "neuron_share": {str(i): round(v, 3) for i, v in sm.neuron_share.items()}, "cosines": cos,
                **_direction_hint(sm)}

    def attribute_neurons(self, top: int = 20, first_only: bool = True) -> dict[str, Any]:
        """Causal ranking (activation x gradient) of the neurons that raise the failure over the working
        text: for loops on the matched continuations, otherwise at the contrast's decision points."""
        if (err := self._spend("screens")):
            return {"error": err}
        lm, c = self.lm, self.res.contrast
        if self.ctx.pairs:
            cn = S.causal_neurons(lm, self.ctx.pairs, top=int(top), first_only=bool(first_only))
            src = "matched continuations"
        else:
            texts = []
            for e in c.failing[:6]:
                if e.decision_ids is not None and e.decision_tok is not None:
                    texts.append((torch.cat([e.decision_ids, torch.tensor([[e.decision_tok]])], 1),
                                  e.decision_ids.shape[1], 1.0))
            for e in c.passing[:6]:
                if e.decision_ids is not None and e.decision_tok is not None:
                    texts.append((torch.cat([e.decision_ids, torch.tensor([[e.decision_tok]])], 1),
                                  e.decision_ids.shape[1], -1.0))
            A = S.attribute(lm, texts)
            flat = A.flatten()
            idx = torch.argsort(flat, descending=True)[:int(top)].tolist()
            ls = A.clamp(min=0).sum(1)
            cn = {"neurons": [(i // A.shape[1], i % A.shape[1], float(flat[i])) for i in idx],
                  "layer_share": (ls / (ls.sum() + 1e-12)).tolist()}
            src = "the decision tokens of the failing (+) and working (-) runs"
        ls = cn["layer_share"]
        order = sorted(range(len(ls)), key=lambda i: -ls[i])[:8]
        return {"source": src, "neurons": [[ly, j, round(v, 4)] for ly, j, v in cn["neurons"]],
                "layers": [[i, round(ls[i], 3)] for i in order],
                "note": "late layers often rank high because they write the next token directly; an earlier "
                        "neuron that causes the failure can rank lower - check candidates with inspect_neuron and "
                        "screen_edit"}

    def _attribution_texts(self) -> tuple[list, str]:
        c = self.res.contrast
        if self.ctx.pairs:
            texts = [(torch.cat([p.prefix, torch.tensor([cont], dtype=torch.long)], 1), p.prefix.shape[1], w)
                     for p in self.ctx.pairs for cont, w in ((p.bad, 1.0), (p.good, -1.0))]
            return texts, "matched continuations (repeat + / working -)"
        texts = []
        for side, w in ((c.failing[:6], 1.0), (c.passing[:6], -1.0)):
            for e in side:
                if e.decision_ids is not None and e.decision_tok is not None:
                    texts.append((torch.cat([e.decision_ids, torch.tensor([[e.decision_tok]])], 1),
                                  e.decision_ids.shape[1], w))
        return texts, "the decision tokens of the failing (+) and working (-) runs"

    def attribute_heads(self, top: int = 20, first_only: bool = True) -> dict[str, Any]:
        """Causal ranking (activation x gradient) of attention heads, like attribute_neurons."""
        if (err := self._spend("screens")):
            return {"error": err}
        texts, src = self._attribution_texts()
        if not texts:
            return {"error": "no failing / working texts to attribute on"}
        A = S.attribute_heads(self.lm, texts, first_only=bool(first_only)) / max(1, len(texts) // 2)
        flat = A.flatten()
        idx = torch.argsort(flat, descending=True)[:self._cap(top, 200)].tolist()
        ls = A.clamp(min=0).sum(1)
        order = sorted(range(len(ls)), key=lambda i: -float(ls[i]))[:8]
        return {"source": src, "heads": [[i // A.shape[1], i % A.shape[1], round(float(flat[i]), 4)] for i in idx],
                "layers": [[i, round(float(ls[i] / (ls.sum() + 1e-12)), 3)] for i in order],
                "note": "edit a head with build_edit scale_heads; check it with screen_edit"}

    def _fit_text(self, item: Any, *, own_reply: bool) -> tuple[torch.Tensor, int]:
        """(ids, start of the trained continuation). An example name with no continuation uses that example's
        own reply (own_reply); reply_prefix is fed before the continuation without being trained on."""
        ex = self._examples()
        name = item if isinstance(item, str) else (item.get("prompt") if isinstance(item, dict) else None)
        if own_reply and isinstance(name, str) and name in ex and not (isinstance(item, dict) and
                                                                        item.get("continuation")):
            e = ex[name]
            pids = e.prompt_ids if hasattr(e, "prompt_ids") else e.ids
            gen = [t for t in (e.gen or []) if t not in self.lm.eot_ids][:300]
            if not gen:
                raise ValueError(f"{name} has no reply to train on")
            return torch.cat([pids, torch.tensor([gen], dtype=torch.long)], 1), pids.shape[1]
        if isinstance(item, dict) and item.get("reply_prefix"):
            base = {k: v for k, v in item.items() if k not in ("continuation", "reply_prefix")}
            ids, _ = self._text_ids(base)
            ids = torch.cat([ids, torch.tensor([self._ids(str(item["reply_prefix"]))], dtype=torch.long)], 1)
            P0 = ids.shape[1]
            if item.get("continuation"):
                ids = torch.cat([ids, torch.tensor([self._ids(str(item["continuation"]))], dtype=torch.long)], 1)
            return ids, P0
        return self._text_ids(item)

    def fit_edit(self, params: list, targets: list, keep: Optional[list] = None, rank: int = 4, steps: int = 60,
                 lr: float = 0.01, kl_weight: float = 1.0, why: str = "") -> dict[str, Any]:
        """Learn a low-rank change of a few weight matrices from texts (surgery/fit.py)."""
        from ..surgery import fit as F

        if not self.budget.weight_edits:
            return {"error": NO_WEIGHT_EDITS}
        if not isinstance(params, list) or not params or len(params) > 4:
            return {"error": "params: a list of 1-4 weight matrix names (e.g. model.layers.12.mlp.down_proj.weight)"}
        frozen = self._forbidden()
        bad = [p for p in params if p in frozen]
        if bad:
            return {"error": f"{bad[0]} is frozen (input embedding, output layer and final norm); request_capability "
                             "asks the user to unlock it"}
        rank = int(rank)
        if not 1 <= rank <= min(self.budget.max_fit_rank, self.budget.max_rank):
            return {"error": f"rank must be 1-{min(self.budget.max_fit_rank, self.budget.max_rank)}"}
        steps = int(steps)
        if not 1 <= steps <= self.budget.max_fit_steps:
            return {"error": f"steps must be 1-{self.budget.max_fit_steps}"}
        if not 0 < float(lr) <= 0.1 or not 0 <= float(kl_weight) <= 100:
            return {"error": "lr must be in (0, 0.1] and kl_weight in [0, 100]"}
        if not isinstance(targets, list) or not targets or len(targets) > 32:
            return {"error": "targets: 1-32 objects {item, weight}; weight > 0 makes the continuation more likely, "
                             "< 0 less likely"}
        ts = []
        for t in targets:
            if not isinstance(t, dict) or "item" not in t:
                return {"error": "each target is {item, weight}; item as in read (an example name uses its own reply "
                                 "when no continuation is given), optionally with reply_prefix"}
            w = float(t.get("weight", 1.0))
            if w == 0 or abs(w) > 10:
                return {"error": "a target weight must be non-zero and at most 10 in size"}
            ids, P0 = self._fit_text(t["item"], own_reply=True)
            if ids.shape[1] <= P0:
                return {"error": "a target needs a continuation to train on"}
            ts.append((ids, P0, w))
        if keep is None:
            ks = P._full_texts(self.ctx.base.replies)[:8]
            for e in self.res.contrast.passing[:4]:
                gen = [x for x in (e.gen or []) if x not in self.lm.eot_ids][:300]
                if gen:
                    ks.append(torch.cat([e.prompt_ids, torch.tensor([gen], dtype=torch.long)], 1))
        else:
            if not isinstance(keep, list) or len(keep) > 32:
                return {"error": "keep: up to 32 items (as in read; an example name keeps its own reply)"}
            ks = [self._fit_text(k, own_reply=True)[0] for k in keep]
        if (err := self._spend("fits")):
            return {"error": err}
        try:
            r = F.fit_lowrank(self.lm, params, ts, ks, rank=rank, steps=steps, lr=float(lr),
                              kl_weight=float(kl_weight), seconds=self.budget.fit_seconds,
                              max_rel_change=self.budget.max_rel_change if "large" not in self.budget.allow else 1e9)
        except ValueError as e:
            return {"error": str(e)}
        if not r.deltas:
            return {"error": "the fit changed nothing (the targets may already be met)", "loss_start": r.loss_start}
        deltas = []
        for name, U, V in r.deltas:
            problem = self._check_delta(name, U, V)
            if problem:
                return {"error": f"the fitted change fails a build check on {name}: {problem}; try a lower lr, fewer "
                                 "steps or a higher kl_weight", "loss_start": r.loss_start, "loss_end": r.loss_end}
            deltas.append(E.Delta(name, U.contiguous(), V.contiguous()))
        patch = E.Patch(f"fitted rank-{rank} change on " + ", ".join(sorted({d.name for d in deltas})), "agent_fit",
                        {"why": why, "rank": rank, "steps": r.steps, "targets": len(ts), "keep": len(ks),
                         "lr": float(lr), "kl_weight": float(kl_weight)}, deltas)
        if patch.n_free_params > self.budget.max_free_params and "large" not in self.budget.allow:
            return {"error": f"the fitted change has {patch.n_free_params:,} free parameters, over the cap "
                             f"{self.budget.max_free_params:,}: lower the rank or fit fewer matrices"}
        eid = self._add_edit(patch)
        return {"edit_id": eid, "label": patch.label, "steps": r.steps, "stopped": r.stopped, "seconds": r.seconds,
                "loss_start": r.loss_start, "loss_end": r.loss_end, "weights_changed": patch.n_weights_changed,
                "free_params": patch.n_free_params, "fits_left": self.remaining()["fits"],
                "note": "target loss down and keep_kl small is the aim; check the edit with screen_edit / generate, "
                        "then verify_edit or practice_eval"}

    @torch.no_grad()
    def inspect_neuron(self, layer: int, neuron: int) -> dict[str, Any]:
        lm, c = self.lm, self.res.contrast
        col = lm.down_proj(layer).weight.detach()[:, neuron]
        lg = lm.logits_of(col.float()[None])[0]
        top = torch.topk(lg, 8).indices.tolist()
        bot = torch.topk(-lg, 8).indices.tolist()

        def act(states):
            v = [lm.seq_capture(s.ids, mlp_layers=[layer], positions=s.positions)["mlp_act"][layer][:, neuron].mean()
                 for s in states]
            return round(float(torch.stack(v).mean()), 4) if v else None
        sts = S.from_contrast(c)
        af, aw = act([s for s in sts if s.bad]), act([s for s in sts if not s.bad])
        return {"layer": layer, "neuron": neuron, "promotes": [lm.token_str(t) for t in top],
                "suppresses": [lm.token_str(t) for t in bot], "activation_failing": af, "activation_working": aw,
                "output_norm": round(float(col.float().norm()), 4),
                **_neuron_hint(layer, neuron, af, aw, [lm.token_str(t) for t in top[:3]], lm.eot_ids, top[:3])}

    @torch.no_grad()
    def logit_lens(self, example: str, depth: int) -> dict[str, Any]:
        ex = self._examples()
        if example not in ex:
            return {"error": f"unknown example {example!r}"}
        e = ex[example]
        ids = next((x for x in (getattr(e, a, None) for a in ("decision_ids", "prompt_ids", "ids")) if x is not None),
                   None)
        if ids is None:
            return {"error": f"{example!r} has no prompt to read"}
        if not 0 <= int(depth) <= self.lm.n_layers:
            return {"error": f"depth must be 0-{self.lm.n_layers} (0 = the embeddings, k = after layer k - 1)"}
        r = self.lm.run(ids, capture=("resid",))
        lg = self.lm.logits_of(r.resid[int(depth)][None])[0]
        p = lg.softmax(-1)
        top = torch.topk(p, 8)
        return {"example": example, "depth": int(depth), "top": [[self.lm.token_str(i), round(float(v), 4)]
                                                                  for v, i in zip(top.values, top.indices)]}

    def build_edit(self, spec: dict[str, Any]) -> dict[str, Any]:
        if not self.budget.weight_edits:
            return {"error": NO_WEIGHT_EDITS}
        lm = self.lm
        spec = _normalise_spec(spec)
        kind = spec.get("kind")
        alpha = float(spec.get("alpha", 1.0))
        if abs(alpha) > self.budget.max_alpha:
            return {"error": f"alpha {alpha} is beyond the allowed {self.budget.max_alpha}"}
        if kind == "orthogonalise":
            v = self._direction(spec["direction_id"])
            patch = E.orthogonalise(lm, v, alpha, layers=spec.get("layers"), embed=bool(spec.get("embed", True)),
                                    source=spec["direction_id"])
        elif kind == "project":
            if not spec.get("sites"):
                return {"error": "project needs at least one site: [{layer, part, neurons?}]"}
            sites = []
            for s in spec["sites"]:
                L = int(s["layer"])
                u = self._direction(spec["direction_id"], L + 1)
                cols = [int(j) for j in s["neurons"]] if s.get("neurons") else None
                sites.append(E.Site(L, s.get("part", "mlp"), u, cols))
            patch = E.project(lm, sites, alpha, source=spec["direction_id"])
        elif kind in ("scale_neurons", "scale_heads"):
            key, unit = ("neurons", "neuron") if kind == "scale_neurons" else ("heads", "head")
            items = spec.get(key)
            if not isinstance(items, list) or not items or not all(
                    isinstance(n, (list, tuple)) and len(n) == 2 and all(isinstance(x, int) for x in n) for n in items):
                return {"error": f"{key} must be a list of [layer, {unit}] pairs of integers; the strength goes in "
                                 f"\"scale\" (0 removes, 0.5 halves, 2 doubles)", "example": EDIT_EXAMPLES["scale_neurons"]
                        if kind == "scale_neurons" else {"kind": "scale_heads", "heads": [[3, 7]], "scale": 0}}
            pairs = [tuple(int(x) for x in n) for n in items]
            scale = float(spec.get("scale", 0))
            if abs(scale) > self.budget.max_alpha:
                return {"error": f"scale {scale} is beyond the allowed {self.budget.max_alpha}"}
            fn = E.scale_neurons if kind == "scale_neurons" else E.scale_heads
            patch = fn(lm, pairs, scale)
        elif kind == "gate":
            L = int(spec["layer"])
            if abs(float(spec.get("strength", 1.0))) > self.budget.max_alpha:
                return {"error": f"strength {spec.get('strength')} is beyond the allowed {self.budget.max_alpha}"}
            src = spec.get("read_source") or {"kind": "contrast"}
            if isinstance(src, str):                      # "contrast" | "continuations"; a direction id is not a source
                if src in self.directions:
                    return {"error": "read_source is where the failing / working states come from ({kind: contrast} "
                                     "or {kind: prompts, failing: [...], working: [...]}), not a direction id; the "
                                     "direction to write against is write_direction_id"}
                src = {"kind": src}
            sts = self._states(src)
            silent_names = spec.get("silent") or ["suite"]
            silent: list[torch.Tensor] = []
            for nm in silent_names:
                silent += [P._full_texts([r])[0] for r in self.ctx.base.replies
                           if (nm == "suite" and r.kind != "safety") or (nm == "harmful" and r.kind == "safety"
                                                                         and r.refused) or nm == f"suite:{r.name}"]
            ref = P._full_texts(self.ctx.base.replies)[:8]
            plan = S.gate_plan(lm, L, [s for s in sts if s.bad], [s for s in sts if not s.bad], silent, ref,
                               max_off=self.budget.gate_max_off)
            if not plan["ok"]:
                return {"error": "no gate: " + plan["why"]}
            w = self._direction(spec["write_direction_id"], L + 1)
            neuron = int(spec["neuron"]) if spec.get("neuron") is not None else \
                S.quiet_neurons(lm, L, ref[:6], k=1)[0]
            strength = float(spec.get("strength", 1.0))
            patch = E.gate(lm, L, neuron, plan["read"], w, threshold=plan["threshold"], up_scale=plan["up_scale"],
                           mean_dir=plan["mean_dir"], mean_proj=plan["mean_proj"], strength=strength)
            patch.spec["plan"] = {k: v for k, v in plan.items() if not isinstance(v, torch.Tensor)}
        elif kind in ("delta", "set_columns", "set_rows"):
            patch = self._free_edit(spec)
        elif kind == "combine":
            parts = [self._edit(i) for i in spec["edit_ids"]]
            clash = E.overwrite_clash(parts)
            if clash:
                return {"error": clash}
            patch = E.Patch(" + ".join(p.label for p in parts), "combined",
                            {"parts": [p.label for p in parts], "kinds": [p.kind for p in parts]},
                            [d for p in parts for d in p.deltas], untie=any(p.untie for p in parts))
        else:
            return {"error": f"unknown edit kind {kind!r}: spec needs \"kind\", one of {', '.join(EDIT_KINDS)}",
                    "example": EDIT_EXAMPLES}
        if patch.n_free_params > self.budget.max_free_params and "large" not in self.budget.allow:
            why = ""
            if kind == "orthogonalise" and spec.get("embed", True):
                why = (" - orthogonalise also edits the input embedding (vocabulary x d_model) unless \"embed\": "
                       "false; or edit fewer layers, or use project / scale_neurons for a local edit")
            return {"error": f"edit has {patch.n_free_params:,} free parameters (rank x (rows + columns) per "
                             f"tensor), over the cap {self.budget.max_free_params:,}{why}"}
        eid = self._add_edit(patch)
        out = {"edit_id": eid, "label": patch.label, "weights_changed": patch.n_weights_changed,
               "free_params": patch.n_free_params}
        if patch.n_weights_changed == 0:
            out["warning"] = "this edit changes no weight: it is the original model"
        if kind == "gate":
            out["gate"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in patch.spec["plan"].items()}
        return out

    def _edit(self, eid: str) -> E.Patch:
        """An edit to apply on top of the model as it is now (the committed stack, if any)."""
        if eid not in self.edits:
            raise ValueError(f"unknown edit id {eid!r}; build one with build_edit")
        if eid in self.stack:
            raise ValueError(f"{eid} is committed: it is already part of the model every tool reads (applying it "
                             "again would add it twice). Leave out edit_id to see it, or revert_edit to take it out")
        on = tuple(self.built_on.get(eid, ()))
        if on != tuple(self.stack):
            raise ValueError(self._stale(eid, on))
        return self.edits[eid]

    def screen_edit(self, edit_id: str, pairs_id: Optional[str] = None) -> dict[str, Any]:
        patch = self._edit(edit_id)
        if (err := self._spend("screens")):
            return {"error": err}
        ts = A.targets(self.lm, self.res.contrast)
        prs = self._pairs(pairs_id)
        base = self.ctx.sel0 if pairs_id is None and not self.stack else PR.selectivity(self.lm, prs)
        with E.applied(self.lm, patch):
            sc = P._screen(self.lm, ts)
            cont = PR.compare(base, PR.selectivity(self.lm, prs)) if prs else None
        out = {"decision_flips": f"{sc['flipped']}/{sc['of']}", "decision_margins": sc["metrics"],
               "note": "a margin below 0 means the failing token no longer wins at the decision point"}
        if cont:
            out["continuations"] = cont
        return out

    def verify_edit(self, edit_id: str) -> dict[str, Any]:
        """The playbook's checks on what would ship: with a committed stack, the stack plus this edit as one patch,
        compared with the ORIGINAL model (its failing, working and regression replies)."""
        key, patch = self.candidate(edit_id)
        if (err := self._spend("verifications")):
            return {"error": err}
        step: dict[str, Any] = {"strength": "agent", "weights_changed": patch.n_weights_changed}
        with self.original(), E.applied(self.lm, patch):
            probs = P._verify(self.ctx, patch, step)
            rows = step.pop("_regression_rows", None)
        step["problems"] = probs
        step["_rows"] = rows
        self.results[key] = step
        if not probs:
            self.verified[key] = step
        return {"passed": not probs, "problems": probs,
                **({"candidate": key, "checked": f"{key}: the committed stack plus {edit_id} as one patch, against "
                                                 "the original model"} if key != edit_id else {}),
                "target": {k: step["target"][k] for k in ("fixed", "of", "passing_kept", "passing", "original_fixed", "original_measured")
                           if k in step.get("target", {})},
                "target_samples": step.get("target", {}).get("samples", [])[:3],
                "regression": {k: step["regression"].get(k) for k in
                               ("new_failures", "mean_delta_nll", "top1_agreement", "safety_lost",
                                "unchanged_replies", "prompts")} if "regression" in step else None,
                "serving": step.get("serving"), "fresh": step.get("fresh"), "sampled": step.get("sampled"),
                "verifications_left": self.remaining()["verifications"]}

    def submit_patch(self, edit_id: str, rationale: str) -> dict[str, Any]:
        key, _ = self.candidate(edit_id)
        if key not in self.verified:
            return {"error": "only an edit that passed verify_edit in this session can be submitted"
                             + (f" (with the committed stack, the candidate is {key}: verify_edit {edit_id} checks "
                                "it)" if key != edit_id else "")}
        self.done = {"submit": key, "rationale": rationale}
        return {"accepted": True, **({"submitted": key} if key != edit_id else {})}

    # -- multi-stage surgery ---------------------------------------------------------------------------------------
    def commit_edit(self, edit_id: str, why: str = "") -> dict[str, Any]:
        """Make this edit part of the model every later tool reads, so the next stage is diagnosed and calibrated on
        the repaired weights. Nothing ships by this: the measuring tools and the judge take the whole stack plus a
        candidate as one patch on the original weights."""
        if not self.budget.weight_edits:
            return {"error": NO_WEIGHT_EDITS}
        if self._released:
            return {"error": "the run has ended"}
        if edit_id in self.stack:
            return {"error": f"{edit_id} is already committed"}
        if len(self.stack) >= self.budget.max_commits:
            return {"error": f"the stack already has {len(self.stack)} committed edits (at most "
                             f"{self.budget.max_commits}): revert_edit one first"}
        self._edit(edit_id)                               # exists, built on the model as it is now
        key, _ = self.candidate(edit_id)
        self._commit(edit_id)
        out: dict[str, Any] = {"committed": list(self.stack), "label": self.edits[edit_id].label,
                               "model": self._model_is(),
                               "note": "every tool now reads this model: generate, read, make_direction, harm_probe, "
                                       "gate_sweep, fit_edit and weight_scan calibrate on it. Edits built before this "
                                       "commit were made on other weights and are refused here (rebuild them). "
                                       "practice_eval, settings_eval, steer_eval, verify_edit and final_choice measure "
                                       "the stack plus a candidate against the ORIGINAL weights, as the judge does; "
                                       "final_choice on the top edit submits the stack itself"}
        if key not in self.verified:
            out["warning"] = (f"{key} has not passed verify_edit: committing changes what the tools read, not what "
                              "ships; check the stack before choosing it")
        return out

    def revert_edit(self, edit_id: Optional[str] = None, why: str = "") -> dict[str, Any]:
        """Take a committed edit out (and every one committed after it, which was built on it); the weights return
        bit for bit to the state before it was committed."""
        if not self.stack:
            return {"error": "nothing is committed"}
        eid = edit_id or self.stack[-1]
        if eid not in self.stack:
            return {"error": f"{eid} is not committed; committed: {self.stack}"}
        k = self.stack.index(eid)
        removed = self.stack[k:]
        self._reset_to(self.stack[:k])
        return {"reverted": removed, "committed": list(self.stack), "model": self._model_is(),
                "note": "edits built on the reverted state are refused until you commit back to it"}

    def give_up(self, reason: str, next_step: str) -> dict[str, Any]:
        self.done = {"give_up": reason, "next_step": next_step}
        return {"accepted": True}

    # -- dispatch ------------------------------------------------------------------------------------
    def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool; (JSON result, is_error). Weights are restored whatever happens."""
        fn = (getattr(self, "tools", None) or TOOLS).get(name)
        if fn is None:
            return json.dumps({"error": f"unknown tool {name!r}"}), True
        if isinstance(args, dict) and "__invalid_json__" in args:
            return json.dumps({"error": "the tool arguments were not valid JSON; send a JSON object"}), True
        args = dict(args or {}) if isinstance(args, dict) else {}
        # Arguments the tool does not take are dropped and named, not a Python TypeError.
        # The first parameter receives the workbench (`self` for the built-in tools, any name for added ones).
        params = dict(list(inspect.signature(fn).parameters.items())[1:])
        ignored = sorted(k for k in args if k not in params)
        for k in ignored:
            args.pop(k)
        missing = [k for k, p in params.items() if p.default is inspect.Parameter.empty and k not in args]
        schema = next((x for x in SCHEMAS + list(getattr(self, "extra_schemas", [])) if x["name"] == name), {})
        if missing:
            return json.dumps({"error": f"{name} needs {', '.join(missing)}", "schema": schema.get("input_schema")},
                              default=str), True
        try:
            out = fn(self, **args)
        except KeyError as e:
            return json.dumps({"error": f"missing field {e} in the input of {name}",
                               "schema": schema.get("input_schema")}, default=str), True
        except (ValueError, TypeError, IndexError, RuntimeError, LabError) as e:
            return json.dumps({"error": f"{type(e).__name__}: {str(e)[:1000]}"}), True
        except Exception as e:                            # noqa: BLE001 - a fault inside a tool must not end a paid run
            # Every trial edit and steering hook is undone by its own context manager, so the model is as it was.
            return json.dumps({"error": f"{name} failed inside wsr ({type(e).__name__}: {str(e)[:500]}); the model "
                                        "is unchanged. Try the step another way"}), True
        if ignored and isinstance(out, dict):
            out = {**out, "ignored_arguments": ignored}
        bad = isinstance(out, dict) and "error" in out
        text = json.dumps(out, default=str, ensure_ascii=False)
        if len(text) > MAX_RESULT:                        # always valid JSON, also when it is cut
            text = json.dumps({"truncated": f"the result had {len(text):,} characters; its start follows as text. "
                                            "Ask for less at a time (fewer items or fields)",
                               "start": text[:MAX_RESULT * 2 // 3]}, ensure_ascii=False)
        return text, bad


MAX_RESULT = 20_000                # characters of one tool result


NO_WEIGHT_EDITS = ("weight edits are off for this run (--no-weight-edits: the weights must not change): fix it with "
                   "steer_eval (add, clamp, or gated with a harm_probe gate) or settings_eval, then final_choice with "
                   "its steering_id or settings_id")

TOOLS: dict[str, Callable[..., dict]] = {
    "list_examples": Workbench.list_examples, "generate": Workbench.generate,
    "make_direction": Workbench.make_direction, "attribute_neurons": Workbench.attribute_neurons,
    "inspect_neuron": Workbench.inspect_neuron,
    "logit_lens": Workbench.logit_lens, "build_edit": Workbench.build_edit, "screen_edit": Workbench.screen_edit,
    "verify_edit": Workbench.verify_edit, "submit_patch": Workbench.submit_patch, "give_up": Workbench.give_up,
    "compute": Workbench.compute, "read": Workbench.read, "make_pairs": Workbench.make_pairs,
    "screen_layers": Workbench.screen_layers, "gate_sweep": Workbench.gate_sweep, "note": Workbench.note,
    "weight_scan": Workbench.weight_scan,
    "attribute_heads": Workbench.attribute_heads, "fit_edit": Workbench.fit_edit,
    "request_capability": Workbench.request_capability, "harm_probe": Workbench.harm_probe,
    "commit_edit": Workbench.commit_edit, "revert_edit": Workbench.revert_edit,
}

EDIT_KINDS = ("orthogonalise", "project", "scale_neurons", "scale_heads", "gate", "combine", "delta", "set_columns",
              "set_rows")
EDIT_EXAMPLES = {"scale_neurons": {"kind": "scale_neurons", "neurons": [[10, 0]], "scale": 0},
                 "project": {"kind": "project", "direction_id": "d1", "alpha": 1,
                             "sites": [{"layer": 10, "part": "mlp", "neurons": [0, 5]}]},
                 "gate": {"kind": "gate", "layer": 9, "read_source": {"kind": "contrast"}, "write_direction_id": "d1",
                          "strength": 1}}


def _normalise_spec(spec: Any) -> dict[str, Any]:
    """Accept the shapes models often send: {"gate": {...}} for {"kind": "gate", ...}, and a spec wrapped once
    more in {"spec": {...}}."""
    if not isinstance(spec, dict):
        return {}
    if "kind" not in spec and set(spec) == {"spec"} and isinstance(spec["spec"], dict):
        spec = spec["spec"]
    if "kind" not in spec and len(spec) == 1:
        (k, v), = spec.items()
        if k in EDIT_KINDS and isinstance(v, dict):
            return {"kind": k, **v}
    return spec


def _neuron_hint(layer: int, neuron: int, a_fail: Any, a_work: Any, promotes: list, eot: Any, top: list) -> dict:
    """Say what the numbers suggest; the agent still builds, screens and verifies."""
    if a_fail is None or a_work is None:
        return {}
    if abs(a_fail) > 4 * max(abs(a_work), 1e-3) and abs(a_fail) > 1.0:
        what = "end-of-turn" if any(t in eot for t in top) else "the tokens it promotes"
        return {"hint": f"fires on failing prompts ({a_fail}) and not on working ones ({a_work}), writing {what}: "
                        f"a candidate edit is {{kind: scale_neurons, neurons: [[{layer}, {neuron}]], scale: 0}} "
                        "(screen_edit, then verify_edit)"}
    return {}


def _direction_hint(sm: Any) -> dict:
    for i, ns in sm.neurons.items():
        if ns and sm.neuron_share.get(i, 0) >= 0.8 and len(ns) <= 3:
            js = [j for j, _ in ns]
            return {"hint": f"layer {i}: {len(js)} neuron(s) {js} carry {sm.neuron_share[i]:.0%} of this layer's share: "
                            f"inspect_neuron them; candidate edits are scale_neurons on them or project with "
                            f"sites [{{layer: {i}, part: mlp, neurons: {js}}}]"}
    return {}


# What the agent may ask for. Categories with an opt-in name the `wsr repair` flag that grants it
# (the edit is then still verified by the same checks); the rest are for the report and for wsr's
# developers (a tool to add) or point outside weight editing.
UNLOCKS: dict[str, tuple[tuple[str, ...], str]] = {
    "edit_embedding": (("embed",), "rerun with --agent-allow embed (with tied embeddings also unembed)"),
    "edit_output_layer": (("unembed",), "rerun with --agent-allow unembed (with tied embeddings also embed)"),
    "edit_final_norm": (("final_norm",), "rerun with --agent-allow final_norm"),
    "rescale_tensor": (("rescale",), "rerun with --agent-allow rescale"),
    "larger_edit": (("large",), "rerun with --agent-allow large (lifts the rank and |delta| <= |W| caps)"),
    "more_budget": ((), "rerun with a larger --agent-verifications / --agent-turns / --agent-budget-usd"),
}
CATEGORIES = set(UNLOCKS) | {
    "new_tool",               # an analysis or intervention the tools cannot express
    "new_edit_kind",          # an edit that is not a low-rank change of existing weights
    "architecture_change",    # new neurons / heads / layers, a different shape
    "config_or_runtime",      # a config value, the chat template, decoding settings, a system prompt
    "training",               # the ability is missing: fine-tuning / more data / a larger model
    "other",
}

_SOURCE = {"type": "object", "description": "where the failing / working states come from",
           "properties": {"kind": {"type": "string", "enum": ["contrast", "continuations", "prompts", "pairs"],
                                   "description": "contrast: the diagnosis's failing vs working prompts at the "
                                                  "decision token; continuations: the loop's repeat vs a working "
                                                  "continuation of the same prompt (loops only); prompts: your own "
                                                  "lists, read at the last prompt token; pairs: a set from "
                                                  "make_pairs (give its id)"},
                          "id": {"type": "string", "description": "pairs: the pairs_id"},
                          "failing": {"type": "array", "items": {"type": "string"},
                                      "description": "prompts: example names (e.g. 'original', 'suite:harm_weapon') "
                                                     "or prompt texts"},
                          "working": {"type": "array", "items": {"type": "string"}}},
           "required": ["kind"]}

_INTERVENE = {"type": "object", "description": "activation interventions during the pass (no weight change): "
              "zero_neurons [[layer, neuron]], zero_heads [[layer, head]], ablate <vector ref> (removed from every "
              "residual write), add {vector, alpha, layers} (added after those layers), gate <gate id from harm_probe> "
              "(probe-gated refusal). A vector ref is a direction id ('d1', or 'd1@12' for its version after depth 12) "
              "or a workspace variable holding a d_model vector."}
_ITEMS = {"type": "array", "description": "texts: an example name, a prompt text (chat template applied), or {prompt | "
          "messages | raw, continuation?} (raw = no template; continuation is teacher-forced after it)",
          "items": {}}

SCHEMAS: list[dict[str, Any]] = [
    {"name": "compute", "description": (
        "Run code in your persistent numeric workspace (a notebook: variables survive between calls). Python-like: "
        "assignments, for / if, comprehensions, f-strings, print; no def / lambda / import / while / try / class. "
        "Available: torch (math, linear algebra torch.linalg, reductions, topk, svd, einsum ...), F "
        "(cosine_similarity, normalize, softmax, ...), math, and helpers: param(name) (a float32 copy of any weight), "
        "params(pattern) (names and shapes), info() (layer counts, sizes, frozen params), embed_rows(tokens), "
        "unembed_rows(tokens), tok(text), detok(ids), token(id), readout(vector, k) (top tokens of a residual "
        "vector), direction(id, depth?). Results of read(...) are stored here too. No in-place tensor methods; "
        "every weight is a copy, so nothing here changes the model: turn what you compute into an edit with "
        "build_edit (delta / set_columns / set_rows). The model cannot be run from here: use read and generate. "
        "Limits: 120 s and 64M elements per value. The last line's value is shown. Costs one compute."),
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"},
                                                       "why": {"type": "string", "description": "what this tests"}},
                      "required": ["code", "why"]}},
    {"name": "read", "description": (
        "Teacher-forced forward pass over up to 64 texts (original model, an edit, and/or interventions): per-token "
        "log-probs of each continuation, the top-5 next tokens, and captured activations stored in the workspace "
        "(store_as, default r<n>): capture {resid: [depths 0..n_layers], mlp_act: [layers] (down_proj input, one "
        "value per neuron), attn: [layers] (attention output), positions: 'continuation' | 'last' | 'all' | [ints]}. "
        "Costs one read."),
     "input_schema": {"type": "object", "properties": {
         "items": _ITEMS, "capture": {"type": "object"}, "edit_id": {"type": "string"}, "intervene": _INTERVENE,
         "store_as": {"type": "string"}, "why": {"type": "string"}}, "required": ["items", "why"]}},
    {"name": "make_pairs", "description": (
        "Your own matched continuations: [{prompt | messages | raw, bad, good}] - the same context followed by the "
        "failing and by a working continuation (the paper's matched contexts). Returns a pairs_id usable as a "
        "source, in screen_edit and in screen_layers, and the baseline log-probs."),
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object"}},
                                                       "name": {"type": "string"}}, "required": ["items"]}},
    {"name": "screen_layers", "description": (
        "The held-out layer screen: on half the pairs, a single-layer sparse edit per MLP layer (all but the last), "
        "scored by how much it lowers the bad continuation against the good one; the best `keep` layers combined "
        "must also be selective on the other half. Returns the per-layer table and, if confirmed, edit ids per "
        "strength. Needs >= 4 pairs (default: the loop's own matched continuations). Costs one screen."),
     "input_schema": {"type": "object", "properties": {"pairs_id": {"type": "string"}, "alpha": {"type": "number"},
                                                       "keep": {"type": "integer", "minimum": 1, "maximum": 6}}}},
    {"name": "weight_scan", "description": (
        "Looks for DAMAGED WEIGHTS: rows or columns of the layers' weight matrices many times larger than the rest of "
        "their matrix (a bad merge, conversion or fine-tune; garbage on most prompts is the usual symptom). No healthy "
        "model measured has one above 13.4x its matrix's median norm; the default threshold is 50x (or 20x and 10x "
        "the same matrix kind in every other layer). For each finding (largest first) it builds three edits, most "
        "faithful first: every damaged entry divided by one common factor (undoes a single scale), each set to the "
        "median norm, and zeroed. Built, not applied: screen_edit / verify_edit / practice_eval them. A restored model "
        "can show behaviour the damaged one hid (e.g. its own refusals, or answers to harmful requests it only "
        "garbled): check what each guardrail protects, and if the restored model needs a second fix, commit_edit the "
        "restore and build that fix on the restored model. Reads "
        "the weights only (no generation, no budget). threshold: at least 20."),
     "input_schema": {"type": "object", "properties": {"threshold": {"type": "number", "minimum": 20}}}},
    {"name": "gate_sweep", "description": (
        "The playbook's gated-neuron search, for a failure that is CONDITIONAL (it must change only where it "
        "happens, e.g. a loop, or refusing where the model should answer): one gate plan per layer (default: every "
        "layer but the first and last), reporting where a single neuron's input separates the failing states from "
        "the working ones (fire_bad: share of failing positions it fires on; fire_off: share of positions that must "
        "stay unchanged where it would also fire). At the `keep` best layers it builds one edit per strength of the "
        "ladder x0.5, x1, x2, x4 of the strength that exactly cancels the failure's gap along the write direction, "
        "so you do not have to guess a layer or a strength. The readable layer is often earlier than the layer that "
        "writes the failure. Edits are built, not applied: screen_edit / verify_edit / practice_eval them. "
        "read_source: where the failing and working states come from ({kind: contrast}, {kind: continuations} for "
        "loops, or {kind: prompts, failing, working}); silent: texts it must stay off on (default ['suite'], add "
        "'harmful' to keep refusals untouched). Costs one screen."),
     "input_schema": {"type": "object", "properties": {
         "write_direction_id": {"type": "string"}, "read_source": {"type": "object"},
         "layers": {"type": "array", "items": {"type": "integer"}},
         "silent": {"type": "array", "items": {"type": "string"}},
         "keep": {"type": "integer", "minimum": 1, "maximum": 4}}, "required": ["write_direction_id"]}},
    {"name": "request_capability", "description": (
        "Use when your evidence says the repair needs something these tools do not allow (a frozen tensor, a "
        "bigger edit, a new kind of edit or analysis, a shape change, a config / template / decoding change, or "
        "training). It is recorded in the report for the user and wsr's developers; if wsr has an opt-in, the "
        "reply names the flag. Optionally attach a proposal (a free-form edit spec: {kind: delta | set_columns | "
        "set_rows, ...} over workspace variables): it is saved as an UNVERIFIED patch for the user to check, "
        "never applied or shipped by this run. Do not use it to get around a check."),
     "input_schema": {"type": "object", "properties": {
         "capability": {"type": "string", "description": "short name of what is missing"},
         "category": {"type": "string", "enum": sorted(CATEGORIES)},
         "what_it_would_do": {"type": "string"}, "why_needed": {"type": "string",
                                                                 "description": "the mechanism and why the "
                                                                                "available edits cannot do it"},
         "evidence": {"type": "string", "description": "the tool results that show it"},
         "proposal": {"type": "object"}},
         "required": ["capability", "category", "what_it_would_do", "why_needed"]}},
    {"name": "harm_probe", "description": "For refusal failures (over- or under-refusal), and for keeping harmful "
     "requests refused after another fix (a restored or repaired model can answer what it used to garble): the model's "
     "OWN reading of "
     "harm, and a steering built on it. The probe is linear, on the residual stream at the last prompt token, "
     "calibrated once per model on wsr's contrast pools (and the goal's practice set in wsr program) and scored on "
     "prompt families it never saw. It reports its held-out AUC by depth, the model's own mistakes (false refusals, "
     "harmful compliance), how many of them the probe reads correctly, and a verdict (viable or not, and why). It "
     "builds a GATE that ties the refusal to that reading: after one layer, where the reply starts, the probe is read "
     "once; where it reads the request as harmful (logit > margin_up) the refusal component is raised to the refused "
     "level + up_sd standard deviations, and where it reads it as safe (logit < -margin_down) it is lowered to the "
     "answered level. The gate always acts both ways, on purpose: the raising side is what keeps harmful requests "
     "refused when the probe misreads one of them as safe, so fixing over-refusal does not cost safety, and the same "
     "gate fixes under-refusal; margin_down sets how sure the probe must be before it lowers a refusal. Every gate "
     "comes with an estimate: on calibration prompts each read by a "
     "probe fold that did not see it, how many harmful requests the model answers it would raise (what it is for), how "
     "many it refuses it would lower, and how many safe look-alikes and goal prompts it would raise (false refusals): "
     "use it to choose margins and depth before spending a measurement. Trade-offs you control: margin_up / margin_down widen the band where nothing happens "
     "(fewer prompts touched, fewer fixed); up_sd sets how firmly harmful requests are refused; an earlier depth leaves "
     "more layers for the change to act through. Returns a gate_id: intervene {\"gate\": id} in generate / read, and "
     "in wsr program steer_eval {vector: id, mode: gated}. No weights change. When the verdict is not viable the model "
     "has no reading to gate on: do not spend budget on gates. Free: the probe is cached.",
     "input_schema": {"type": "object", "properties": {
         "depth": {"type": "integer", "description": "residual depth to gate at (k = after layer k - 1); default the "
                                                     "probe's chosen depth; earlier depths leave more layers to act"},
         "up_sd": {"type": "number", "minimum": 0, "maximum": 6},
         "margin_up": {"type": "number", "minimum": 0, "maximum": 10},
         "margin_down": {"type": "number", "minimum": 0, "maximum": 10},
         "read": {"type": "array", "maxItems": 8, "description": "prompts whose probe logit to report"}},
         "required": []}},
    {"name": "note", "description": "Write a finding into the run's notebook (kept in the transcript and report).",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "list_examples", "description": "The failing and working prompts with their replies, the regression "
     "suite (with which harmful requests the model refuses) and the held-out fresh variants.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "generate", "description": "Replies from the model, classified with the triage detectors: greedy, or "
     "`samples` sampled replies (temperature > 0; seeded per sample, so compare=true gives the untouched model and the one with the edit and / or the interventions "
     "the same random stream). Optional interventions. Costs one generation per reply.",
     "input_schema": {"type": "object", "properties": {
         "prompt": {"description": "an example name, a prompt text, or {messages: [...]}"},
         "prompts": {"type": "array", "maxItems": 8,
                     "description": "up to 8 prompts at once (instead of prompt): batched to the free GPU memory, "
                                    "far faster than one call each"},
         "edit_id": {"type": "string"},
         "max_new": {"type": "integer", "minimum": 1, "description": "up to 512, or the length verify_edit judges "
                                                                    "at when that is longer (see the brief)"},
         "samples": {"type": "integer", "minimum": 1, "maximum": 8}, "temperature": {"type": "number"},
         "top_p": {"type": "number"}, "seed": {"type": "integer"}, "compare": {"type": "boolean"},
         "repetition_penalty": {"type": "number", "minimum": 1.0, "maximum": 1.5,
                                "description": "a quick decoding probe; measure settings with settings_eval"},
         "intervene": _INTERVENE},
         "required": []}},
    {"name": "make_direction", "description": "Failing-minus-working direction at every depth from a source, plus "
     "which layers write it (share of the difference per attention / MLP output) and which MLP neurons carry it. "
     "Returns a direction_id for build_edit.",
     "input_schema": {"type": "object", "properties": {"source": _SOURCE}, "required": ["source"]}},
    {"name": "attribute_neurons", "description": "Causal ranking of MLP neurons (activation x gradient): which "
     "neurons raise the failure's likelihood over the working text. Complements make_direction's projection "
     "shares. Costs one screen.",
     "input_schema": {"type": "object", "properties": {
         "top": {"type": "integer", "minimum": 1, "maximum": 200},
         "first_only": {"type": "boolean", "description": "loops: score only the step where the continuations part"}},
         "required": []}},
    {"name": "attribute_heads", "description": "Causal ranking of attention heads (activation x gradient on each "
     "head's output), like attribute_neurons: which heads raise the failure over the working text. Edit one with "
     "build_edit scale_heads. Costs one screen.",
     "input_schema": {"type": "object", "properties": {
         "top": {"type": "integer", "minimum": 1, "maximum": 200},
         "first_only": {"type": "boolean"}}, "required": []}},
    {"name": "fit_edit", "description": "LEARN an edit from texts instead of building it by hand: a rank-r change "
     "U V^T of 1-4 weight matrices, fitted by gradient descent (U starts at zero) to make target continuations more "
     "likely (weight > 0) or less likely (weight < 0, unlikelihood), while keeping the next-token distribution on "
     "the keep texts close to the original (KL; default keep: the regression replies and working replies). For "
     "failures that no single neuron, head or direction carries. You may write the target continuations yourself "
     "(e.g. a correct, complete answer for a practice prompt), or use an example's own reply (an example name with "
     "no continuation). Same build checks as any edit; returns an edit_id for screen_edit / verify_edit / "
     "practice_eval. Guarded: step and time caps, a free-memory check, out-of-memory ends the fit cleanly.",
     "input_schema": {"type": "object", "properties": {
         "params": {"type": "array", "items": {"type": "string"},
                    "description": "weight matrices, e.g. model.layers.12.mlp.down_proj.weight"},
         "targets": {"type": "array", "items": {"type": "object"},
                     "description": "{item, weight}: item as in read ({prompt | messages | raw, reply_prefix?, "
                                    "continuation?} or an example name); reply_prefix is fed before the continuation "
                                    "without being trained on"},
         "keep": {"type": "array", "description": "texts whose behaviour must not move (default: regression + working "
                                                  "replies)"},
         "rank": {"type": "integer", "minimum": 1, "maximum": 32},
         "steps": {"type": "integer", "minimum": 1, "maximum": 300},
         "lr": {"type": "number"}, "kl_weight": {"type": "number"}, "why": {"type": "string"}},
         "required": ["params", "targets"]}},
    {"name": "inspect_neuron", "description": "What an MLP neuron writes (tokens its output column promotes / "
     "suppresses through the unembedding) and its activation on failing vs working decision points.",
     "input_schema": {"type": "object", "properties": {"layer": {"type": "integer"}, "neuron": {"type": "integer"}},
                      "required": ["layer", "neuron"]}},
    {"name": "logit_lens", "description": "Top tokens read out of the residual stream after a depth (0 = embeddings) "
     "at an example's decision point.",
     "input_schema": {"type": "object", "properties": {"example": {"type": "string"}, "depth": {"type": "integer"}},
                      "required": ["example", "depth"]}},
    {"name": "build_edit", "description": (
        "Build a candidate weight edit (not applied; returns an edit_id). kinds: "
        "orthogonalise {direction_id, alpha, layers?, embed?} removes a direction from every residual write; "
        "project {direction_id, alpha, sites: [{layer, part: mlp|attn, neurons?}]} takes the per-depth direction "
        "out of those sites only (alpha > 1 pushes against it); scale_neurons {neurons: [[layer, neuron]], scale}; "
        "scale_heads {heads: [[layer, head]], scale}; gate {layer, read_source, write_direction_id, strength, "
        "silent?: ['suite' | 'harmful' | 'suite:<name>'], neuron?} rewrites one unused MLP neuron to fire only "
        "where read_source's failing states differ from its working ones and stay off on the silent texts, writing "
        "against the direction; combine {edit_ids}. Free-form, from workspace variables: delta {changes: [{param, "
        "U, V}]} adds U V^T (U [out, r], V [in, r]; a 1-D parameter takes U [n, r], V [1, r]); set_columns {param, "
        "cols, values [out, k]} replaces columns (e.g. a neuron's output in down_proj); set_rows {param, rows, "
        "values [k, in]} replaces rows (e.g. a neuron's gate_proj / up_proj input). Frozen: the input embedding, "
        "the output layer, the final norm; refused: a change that only rescales a whole tensor, rank over 1024, or "
        "|delta| > |W|. Give `why`."),
     "input_schema": {"type": "object", "properties": {"spec": {"type": "object"}}, "required": ["spec"]}},
    {"name": "screen_edit", "description": "Cheap check of an edit: does the failing token lose at each failing "
     "prompt's decision point; for loops, does the repeated continuation become less likely than the working one. "
     "Costs one screen.",
     "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                       "pairs_id": {"type": "string", "description": "score the "
                                                                    "continuations of this pair set instead"}},
                      "required": ["edit_id"]}},
    {"name": "verify_edit", "description": "The full checks the shipped patch must pass: failing and working "
     "prompts, the regression suite (new failures, drift, top-token agreement, harmful requests still refused), "
     "serving precision, and the mode's extra checks. Expensive and limited; costs one verification. With committed "
     "edits it checks the stack plus this edit, as one patch, against the original model.",
     "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"}}, "required": ["edit_id"]}},
    {"name": "submit_patch", "description": "Finish with an edit that passed verify_edit in this session.",
     "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"},
                                                       "rationale": {"type": "string"}},
                      "required": ["edit_id", "rationale"]}},
    {"name": "commit_edit", "description": (
        "MULTI-STAGE SURGERY: make this edit part of the model that every later tool reads, so the next problem is "
        "diagnosed and calibrated on the repaired weights (a model with more than one fault: e.g. restore damaged "
        "weights, then build a refusal gate on the RESTORED model's own reading of harm with harm_probe, or a second "
        "edit with make_direction / gate_sweep / fit_edit). Nothing ships by committing. practice_eval, settings_eval, "
        "steer_eval, verify_edit and final_choice always measure the committed stack plus the candidate as ONE patch "
        "against the ORIGINAL weights, exactly as the judge will (their results name it, e.g. e1+e4); final_choice on "
        "the top committed edit submits the stack itself, and a steering or settings measured on a stack are judged "
        "and shipped together with it. Edits built before a commit were computed from other weights and are refused "
        "after it: rebuild them. At most 4 committed at once; revert_edit takes them out."),
     "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"}, "why": {"type": "string"}},
                      "required": ["edit_id"]}},
    {"name": "revert_edit", "description": "Take a committed edit out of the model (default: the last one), with every "
     "edit committed after it; the weights return bit for bit to the state before it was committed.",
     "input_schema": {"type": "object", "properties": {"edit_id": {"type": "string"}, "why": {"type": "string"}}}},
    {"name": "give_up", "description": "Finish without a patch: why no edit can fix it, and what to do instead.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}, "next_step": {"type": "string"}},
                      "required": ["reason", "next_step"]}},
]
