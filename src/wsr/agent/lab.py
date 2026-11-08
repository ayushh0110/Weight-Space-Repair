"""The agent's numeric workspace: Python-like statements over tensors, with no way out to Python.

Takehi et al.'s editor had a full Python kernel over the weights inside an isolated container.
wsr runs on the user's own machine, where model-written Python could read files, open sockets or
start processes, so the agent gets a small language instead:

- statements: assignment (`x = ...`, `x += ...`, `x[i] = ...` on its own values), `for` over a
  bounded iterable, `if`, expressions; comprehensions; f-strings. No `def`, `lambda`, `class`,
  `import`, `while`, `try`, `with`, `global`, `del`.
- names: the agent's own variables (they persist between calls, like a notebook), a whitelisted
  `torch` (math, linear algebra, reductions; nothing that reads or writes files), and the helpers
  the workbench provides (read-only copies of weights, token lookup, logit read-out, directions).
- attribute access goes through a runtime check of the object's type against a whitelist, and no
  name or attribute starting with "_" can be written, so `().__class__`-style escapes do not parse.
  Calls the agent makes go through `_call`, which only calls whitelisted functions.
- in-place tensor methods are not available, and every weight handed out is a copy: the model
  cannot be changed from here. Edits are built only by `build_edit`, from the arrays computed here.
- every loop step and call checks a wall-clock deadline and a step budget; binary operations check
  the broadcast size before computing; results above the element cap are refused.
"""
from __future__ import annotations

import ast
import math
import time
from types import SimpleNamespace
from typing import Any, Callable, Optional

import torch

MAX_ELEMENTS = 64_000_000          # 256 MB of float32 per value
MAX_STEPS = 2_000_000              # loop / comprehension iterations per call
MAX_OUTPUT = 8_000                 # characters returned per call


class LabError(Exception):
    pass


# -- what can be named -----------------------------------------------------------------------------
_TORCH_FUNCS = """abs acos addmm all allclose amax amin any arange argmax argmin argsort asin atan atan2 bincount
bmm broadcast_shapes broadcast_to bucketize cat ceil chunk clamp clip corrcoef cos cosh count_nonzero cov cross
cummax cummin cumprod cumsum diag diag_embed diagonal diff dist dot einsum eq exp expm1 eye flatten flip fliplr
floor fmod full gather ge gt histc hstack index_select inner isclose isfinite isin isnan kron kthvalue le lerp
linspace log log10 log1p log2 log_softmax logical_and logical_not logical_or logical_xor logsumexp lt masked_select
matmul max maximum mean median min minimum mm mul mv nan_to_num nanmean nanmedian nansum narrow ne neg nonzero
norm ones ones_like outer pdist permute pow prod quantile rand randint randn randperm ravel reciprocal relu
remainder reshape round rsqrt searchsorted sigmoid sign sin sinh softmax sort split sqrt square squeeze stack std
std_mean sub sum t take tan tanh tensor tensordot topk trace transpose tril triu true_divide trunc unbind unique
unique_consecutive unsqueeze var var_mean vstack where zeros zeros_like float32 float64 int64 bool long""".split()
_LINALG = "cholesky cond cross det eig eigh eigvals eigvalsh inv lstsq matrix_norm matrix_rank norm pinv qr " \
          "slogdet solve svd svdvals vector_norm".split()
_FUNCTIONAL = "cosine_similarity gelu silu normalize one_hot pad relu softplus softmax log_softmax".split()

_TENSOR_ATTRS = """T mT H abs all amax amin any argmax argmin argsort bool ceil chunk clamp clip clone contiguous cos
cpu count_nonzero cumsum detach diag diagonal diff dim double dtype eq exp expand flatten flip float floor gather ge
gt half index_select int isfinite isnan item le log log_softmax logical_and logical_not logical_or long lt
masked_fill masked_select matmul max mean median min mul nan_to_num nanmean ndim ne neg nonzero norm numel outer
permute pow prod quantile reciprocal relu reshape round rsqrt shape sigmoid sign sin size softmax sort split sqrt
square squeeze std sub sum t tanh to tolist topk trace transpose tril triu type_as unbind unique unsqueeze values
indices var view view_as where""".split()
_SEQ_ATTRS = "append count extend index insert pop reverse sort get items keys values update copy join split strip " \
             "lower upper startswith endswith replace find".split()
_SAFE_BUILTINS = {"len": len, "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "sorted": sorted,
                  "enumerate": enumerate, "zip": zip, "list": list, "dict": dict, "tuple": tuple, "set": set,
                  "float": float, "int": int, "bool": bool, "str": str, "reversed": reversed, "any": any,
                  "all": all, "isinstance": isinstance, "math": SimpleNamespace(**{k: getattr(math, k) for k in
                  "sqrt log exp pi e inf isnan isfinite ceil floor".split()})}

_NODES = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp,
          ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.keyword, ast.Attribute, ast.Subscript,
          ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp,
          ast.GeneratorExp, ast.comprehension, ast.If, ast.For, ast.JoinedStr, ast.FormattedValue, ast.Starred,
          ast.Pass, ast.Break, ast.Continue, ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context)


def _allowed_attr(obj: Any, name: str) -> bool:
    if isinstance(obj, torch.Tensor) or isinstance(obj, torch.Size):
        return name in _TENSOR_ATTRS or (isinstance(obj, torch.Size) and name in ("numel", "count", "index"))
    if type(obj).__module__ == "torch.return_types":     # (values, indices), (U, S, Vh) ... results
        return not name.startswith("_")
    if isinstance(obj, (list, tuple, dict, str, set)):
        return name in _SEQ_ATTRS
    if isinstance(obj, SimpleNamespace):                 # torch / linalg / F / math / helper namespaces
        return name in vars(obj)
    return False


def _numel(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel()
    if isinstance(x, (list, tuple)):
        return sum(_numel(v) for v in x[:10_000]) + max(0, len(x) - 10_000)
    if isinstance(x, dict):
        return sum(_numel(v) for v in list(x.values())[:10_000])
    return 1


class _Rewrite(ast.NodeTransformer):
    """Route attribute access, calls, binary operations and loop iterables through runtime guards."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Call(ast.Name("_attr", ast.Load()), [node.value, ast.Constant(node.attr)],
                                              []), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        return ast.copy_location(ast.Call(ast.Name("_call", ast.Load()), [node.func] + node.args, node.keywords),
                                 node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        return ast.copy_location(ast.Call(ast.Name("_bin", ast.Load()),
                                          [ast.Constant(type(node.op).__name__), node.left, node.right], []), node)

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        node.iter = ast.Call(ast.Name("_it", ast.Load()), [node.iter], [])
        return node

    def visit_comprehension(self, node: ast.comprehension) -> ast.AST:
        self.generic_visit(node)
        node.iter = ast.Call(ast.Name("_it", ast.Load()), [node.iter], [])
        return node


_BIN = {"Add": lambda a, b: a + b, "Sub": lambda a, b: a - b, "Mult": lambda a, b: a * b,
        "Div": lambda a, b: a / b, "FloorDiv": lambda a, b: a // b, "Mod": lambda a, b: a % b,
        "Pow": lambda a, b: a ** b, "MatMult": lambda a, b: a @ b, "BitAnd": lambda a, b: a & b,
        "BitOr": lambda a, b: a | b, "BitXor": lambda a, b: a ^ b}


class Lab:
    """One persistent namespace per agent run. `helpers`: name -> callable, given by the workbench."""

    def __init__(self, helpers: dict[str, Callable[..., Any]], *, seconds: float = 120.0):
        self.seconds = seconds
        self.vars: dict[str, Any] = {}
        F = torch.nn.functional
        tns = {k: getattr(torch, k) for k in _TORCH_FUNCS if hasattr(torch, k)}
        tns["linalg"] = SimpleNamespace(**{k: getattr(torch.linalg, k) for k in _LINALG if hasattr(torch.linalg, k)})
        tns["F"] = SimpleNamespace(**{k: getattr(F, k) for k in _FUNCTIONAL if hasattr(F, k)})
        self.fixed: dict[str, Any] = {"torch": SimpleNamespace(**tns), "F": tns["F"], "linalg": tns["linalg"],
                                      **_SAFE_BUILTINS, **helpers}
        self._callable_ok = {id(v) for v in tns.values() if callable(v)}
        for ns in (tns["linalg"], tns["F"], _SAFE_BUILTINS["math"]):
            self._callable_ok |= {id(v) for v in vars(ns).values() if callable(v)}
        self._callable_ok |= {id(v) for v in _SAFE_BUILTINS.values() if callable(v)}
        self._callable_ok |= {id(v) for v in helpers.values() if callable(v)}
        self._deadline = 0.0
        self._steps = 0
        self._printed: list[str] = []
        # What the workspace refused, by kind:item ("attr:Tensor.numpy", "call:getattr", "syntax:FunctionDef",
        # "name:open", "limit:time"). Over real runs this says which operations the agent needs.
        self.denied: dict[str, int] = {}

    def _deny(self, key: str, msg: str) -> LabError:
        self.denied[key] = self.denied.get(key, 0) + 1
        return LabError(msg)

    # -- runtime guards ----------------------------------------------------------------------------
    def _tick(self) -> None:
        self._steps += 1
        if self._steps > MAX_STEPS:
            raise self._deny("limit:steps", f"step budget exceeded ({MAX_STEPS:,} loop iterations)")
        if time.perf_counter() > self._deadline:
            raise self._deny("limit:time", f"time limit exceeded ({self.seconds:g} s)")

    def _check(self, v: Any) -> Any:
        if _numel(v) > MAX_ELEMENTS:
            raise self._deny("limit:size", f"result has {_numel(v):,} elements, over the cap {MAX_ELEMENTS:,}")
        return v

    def _attr(self, obj: Any, name: str) -> Any:
        if name.startswith("_") or not _allowed_attr(obj, name):
            raise self._deny(f"attr:{type(obj).__name__}.{name}",
                             f"attribute {name!r} of {type(obj).__name__} is not available")
        return getattr(obj, name)

    def _call(self, fn: Any, *args: Any, **kw: Any) -> Any:
        self._tick()
        ok = id(fn) in self._callable_ok or self._bound_ok(fn)
        if not ok:
            nm = getattr(fn, "__name__", type(fn).__name__)
            raise self._deny(f"call:{nm}", f"{nm} cannot be called here")
        if fn is print:
            self._printed.append(" ".join(_show(a) for a in args))
            return None
        return self._check(fn(*args, **kw))

    def _bound_ok(self, fn: Any) -> bool:
        """A method reached through `_attr` (it was whitelisted for its object's type)."""
        owner = getattr(fn, "__self__", None)
        name = getattr(fn, "__name__", "")
        if owner is None or name.startswith("_"):
            return False
        return _allowed_attr(owner, name)

    def _bin(self, op: str, a: Any, b: Any) -> Any:
        self._tick()
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if op == "MatMult":
                if a.dim() >= 2 and b.dim() >= 2:
                    n = a.shape[-2] * b.shape[-1] * max(1, math.prod(torch.broadcast_shapes(a.shape[:-2],
                                                                                            b.shape[:-2])))
                    if n > MAX_ELEMENTS:
                        raise LabError(f"matmul result would have {n:,} elements")
            else:
                n = math.prod(torch.broadcast_shapes(a.shape, b.shape))
                if n > MAX_ELEMENTS:
                    raise LabError(f"result would have {n:,} elements")
        if op == "Mult" and isinstance(a, (list, str, tuple)) and isinstance(b, int) and len(a) * b > 1_000_000:
            raise LabError("sequence too long")
        if op == "Mult" and isinstance(b, (list, str, tuple)) and isinstance(a, int) and len(b) * a > 1_000_000:
            raise LabError("sequence too long")
        if op == "Pow" and isinstance(a, int) and isinstance(b, int) and abs(b) > 10_000:
            raise LabError("exponent too large")
        return self._check(_BIN[op](a, b))

    def _it(self, xs: Any) -> Any:
        if isinstance(xs, int):
            raise LabError("cannot iterate over a number; use range(n)")
        for x in xs:
            self._tick()
            yield x

    # -- parse and run ------------------------------------------------------------------------------
    _PROVIDED = {"torch": "torch", "math": "math", "torch.nn.functional": "F", "torch.linalg": "linalg"}

    def _provided_import(self, node: ast.AST) -> bool:
        """A top-level import of a module the workspace already provides, under the name it already has
        (`import torch`, `import torch.nn.functional as F`): a no-op, dropped. Live runs sent these out of
        habit; anything else is still refused."""
        if isinstance(node, ast.Import):
            pairs = [(a.name, a.asname or a.name.split(".")[0]) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            pairs = [(f"{node.module}.{a.name}", a.asname or a.name) for a in node.names]
        else:
            return False
        ok = all((self._PROVIDED.get(mod) == name or (name == "torch" and mod.startswith("torch")))
                 and name in self.fixed for mod, name in pairs)
        if not ok:
            names = ", ".join(sorted(k for k in self.fixed if not k.startswith("_")))
            raise self._deny("syntax:Import", "imports are not available; already provided: " + names)
        return True

    def _validate(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, _NODES):
                raise self._deny(f"syntax:{type(node).__name__}",
                                 f"{type(node).__name__} is not allowed here (no def, lambda, import, while, "
                                 "try, with, class, global, del)")
            if isinstance(node, ast.Name):
                if node.id.startswith("_"):
                    raise self._deny(f"name:{node.id}", f"names starting with '_' are reserved ({node.id!r})")
                if isinstance(node.ctx, ast.Store) and node.id in self.fixed:
                    raise LabError(f"{node.id!r} is a built-in name and cannot be reassigned")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise self._deny(f"attr:{node.attr}", f"attributes starting with '_' are not available ({node.attr!r})")
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                raise LabError("attributes cannot be assigned")

    def run(self, code: str) -> dict[str, Any]:
        """Run `code`; returns {"output": text, "variables": {name: summary}} or raises LabError."""
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise LabError(f"syntax error: {e.msg} (line {e.lineno})") from None
        tree.body = [n for n in tree.body if not self._provided_import(n)]
        self._validate(tree)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        tree = ast.fix_missing_locations(_Rewrite().visit(tree))
        rng = self._range
        self._callable_ok |= {id(print), id(rng)}
        fixed = {"__builtins__": {"range": rng, "print": print}, **self.fixed,
                 "_attr": self._attr, "_call": self._call, "_bin": self._bin, "_it": self._it}
        # One dict for globals and the agent's variables: comprehensions inside exec only see globals.
        g = {**fixed, **self.vars}
        before = dict(self.vars)
        self._deadline = time.perf_counter() + self.seconds
        self._steps = 0
        self._printed = []
        err: Optional[str] = None
        value = None
        try:
            with torch.no_grad():
                exec(compile(tree, "<lab>", "exec"), g)
                if last is not None:
                    last = ast.fix_missing_locations(_Rewrite().visit(last))
                    value = eval(compile(last, "<lab>", "eval"), g)
        except LabError as e:
            err = str(e)
        except NameError as e:                           # a name the workspace does not provide (open, getattr ...)
            nm = getattr(e, "name", None) or str(e).split("'")[1] if "'" in str(e) else str(e)
            self.denied[f"name:{nm}"] = self.denied.get(f"name:{nm}", 0) + 1
            err = f"NameError: {e} (not available in the workspace)"
        except Exception as e:                           # the agent's own mistakes: report, keep the namespace
            err = f"{type(e).__name__}: {e}"
        # Keep what was assigned before an error too, like a notebook cell.
        self.vars = {k: v for k, v in g.items() if not k.startswith("_") and (k not in fixed or v is not fixed[k])}
        if err:
            raise LabError(err)
        changed = {k: _show(v, short=True) for k, v in self.vars.items() if k not in before or before[k] is not v}
        out = "\n".join(self._printed)
        if last is not None:
            out = (out + "\n" if out else "") + _show(value)
        return {"output": out[:MAX_OUTPUT] + ("..." if len(out) > MAX_OUTPUT else ""), "assigned": changed}

    def _range(self, *a: int) -> range:
        r = range(*a)
        if len(r) > MAX_STEPS:
            raise LabError(f"range of {len(r):,} is over the step budget")
        return r

    def get(self, name: str) -> Any:
        if name not in self.vars:
            raise LabError(f"no variable {name!r} in the workspace")
        return self.vars[name]

    def put(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in self.fixed:
            raise LabError(f"cannot store into {name!r}")
        self.vars[name] = value

    def summary(self) -> dict[str, str]:
        return {k: _show(v, short=True) for k, v in self.vars.items()}


def _show(v: Any, short: bool = False) -> str:
    if isinstance(v, torch.Tensor):
        head = f"tensor{tuple(v.shape)} {str(v.dtype).replace('torch.', '')}"
        if v.numel() == 0:
            return head + " (empty)"
        if v.numel() <= (8 if short else 64):
            return head + " " + str([round(x, 5) if isinstance(x, float) else x for x in v.flatten().tolist()])
        if short:
            return head
        f = v.detach().float()
        return (head + f" min {float(f.min()):.4g} max {float(f.max()):.4g} mean {float(f.mean()):.4g} "
                f"norm {float(f.norm()):.4g} first {[round(x, 4) for x in f.flatten()[:8].tolist()]}")
    if isinstance(v, dict):
        if short:
            return f"dict({len(v)} keys)"
        items = list(v.items())[:40]
        return "{" + ", ".join(f"{k!r}: {_show(x, True)}" for k, x in items) + (", ..." if len(v) > 40 else "") + "}"
    if isinstance(v, (list, tuple)):
        if short:
            return f"{type(v).__name__}({len(v)})"
        s = repr(type(v)([_show(x, True) if isinstance(x, (torch.Tensor, dict)) else x for x in v[:200]]))
        return s[:4000]
    s = repr(v)
    return s[:4000]
