"""A pre-calibrated cross-lingual probe: which language is the model about to answer in?

The contrast set gives 5-15 near-identical prompts, too few and too alike to
estimate a language direction well. This probe is calibrated once per model on
~150 prompts in 12 languages, including cross-language requests ("Answer in
Russian: ...", a French question asking for English), so it reads the
*planned output* language rather than the input language. Labels are the
language the model actually answered in (detected on its reply), not the
language asked for.

At each depth it is a diagonal-covariance nearest-class-mean classifier on the
residual at the decision point (robust with ~10 samples per class in 900-3500
dimensions), with leave-one-out accuracy per depth. On a failing prompt it gives:

* a per-layer readout: P(expected language), P(answered language), and the
  layer where the plan switches from one to the other;
* a direction mu(answered) - mu(expected) from dozens of runs;
* causal tests: removing that direction, and steering toward the expected
  language at the switch layer, must change the reply's language, and do so
  more than steering along an unrelated language pair (the null).

The calibration is cached per model under ~/.cache/wsr/langprobe/.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from ..output import detect_language
from .lm import LM

LANGS = ("en", "fr", "de", "es", "pt", "it", "ru", "uk", "zh", "ja", "ko", "hi")
NAMES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese", "it": "Italian",
         "ru": "Russian", "uk": "Ukrainian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi"}

# Eight everyday questions in each language (native speakers would phrase some differently;
# the probe needs the language, not perfect style).
QUESTIONS = {
    "en": ["Why is the sky blue?", "How do plants grow?", "What is the capital of Italy?",
           "How does a bicycle work?", "Why do we need sleep?", "What is rain made of?",
           "How do I make tea?", "Why is the sea salty?"],
    "fr": ["Pourquoi le ciel est-il bleu ?", "Comment poussent les plantes ?", "Quelle est la capitale de l'Italie ?",
           "Comment fonctionne un vélo ?", "Pourquoi avons-nous besoin de dormir ?", "De quoi est faite la pluie ?",
           "Comment préparer du thé ?", "Pourquoi la mer est-elle salée ?"],
    "de": ["Warum ist der Himmel blau?", "Wie wachsen Pflanzen?", "Was ist die Hauptstadt von Italien?",
           "Wie funktioniert ein Fahrrad?", "Warum brauchen wir Schlaf?", "Woraus besteht Regen?",
           "Wie mache ich Tee?", "Warum ist das Meer salzig?"],
    "es": ["¿Por qué el cielo es azul?", "¿Cómo crecen las plantas?", "¿Cuál es la capital de Italia?",
           "¿Cómo funciona una bicicleta?", "¿Por qué necesitamos dormir?", "¿De qué está hecha la lluvia?",
           "¿Cómo preparo té?", "¿Por qué el mar es salado?"],
    "pt": ["Por que o céu é azul?", "Como as plantas crescem?", "Qual é a capital da Itália?",
           "Como funciona uma bicicleta?", "Por que precisamos dormir?", "Do que é feita a chuva?",
           "Como faço chá?", "Por que o mar é salgado?"],
    "it": ["Perché il cielo è blu?", "Come crescono le piante?", "Qual è la capitale della Francia?",
           "Come funziona una bicicletta?", "Perché abbiamo bisogno di dormire?", "Di cosa è fatta la pioggia?",
           "Come si prepara il tè?", "Perché il mare è salato?"],
    "ru": ["Почему небо голубое?", "Как растут растения?", "Какая столица Италии?",
           "Как работает велосипед?", "Зачем нам нужен сон?", "Из чего состоит дождь?",
           "Как заварить чай?", "Почему море солёное?"],
    "uk": ["Чому небо блакитне?", "Як ростуть рослини?", "Яка столиця Італії?",
           "Як працює велосипед?", "Навіщо нам потрібен сон?", "З чого складається дощ?",
           "Як заварити чай?", "Чому море солоне?"],
    "zh": ["天空为什么是蓝色的？", "植物是怎样生长的？", "意大利的首都是哪里？",
           "自行车是怎么工作的？", "我们为什么需要睡觉？", "雨是由什么组成的？",
           "怎么泡茶？", "海水为什么是咸的？"],
    "ja": ["空はなぜ青いのですか？", "植物はどのように育ちますか？", "イタリアの首都はどこですか？",
           "自転車はどうやって動きますか？", "なぜ睡眠が必要なのですか？", "雨は何でできていますか？",
           "お茶はどうやって入れますか？", "海はなぜしょっぱいのですか？"],
    "ko": ["하늘은 왜 파란가요?", "식물은 어떻게 자라나요?", "이탈리아의 수도는 어디인가요?",
           "자전거는 어떻게 작동하나요?", "우리는 왜 잠이 필요한가요?", "비는 무엇으로 이루어져 있나요?",
           "차는 어떻게 끓이나요?", "바다는 왜 짠가요?"],
    "hi": ["आसमान नीला क्यों होता है?", "पौधे कैसे बढ़ते हैं?", "इटली की राजधानी क्या है?",
           "साइकिल कैसे काम करती है?", "हमें नींद की ज़रूरत क्यों होती है?", "बारिश किससे बनी होती है?",
           "चाय कैसे बनाते हैं?", "समुद्र खारा क्यों होता है?"],
}
# Cross-language requests, so input language and output language come apart.
_ASK_IN = "Answer in {name} only: {q}"
_CROSS_EN = ("Why is the sky blue?", "How do plants grow?", "Why do we need sleep?")
_ASK_EN = {"fr": "Réponds en anglais : {q}", "de": "Antworte auf Englisch: {q}", "es": "Responde en inglés: {q}",
           "ru": "Ответь по-английски: {q}", "zh": "请用英语回答：{q}", "hi": "अंग्रेज़ी में जवाब दो: {q}"}

CACHE = Path(os.environ.get("WSR_CACHE", Path.home() / ".cache" / "wsr")) / "langprobe"


def calibration_prompts() -> list[tuple[str, str]]:
    """(prompt, language asked for) pairs."""
    out = [(q, lang) for lang, qs in QUESTIONS.items() for q in qs]
    for lang in LANGS:
        if lang != "en":
            out += [(_ASK_IN.format(name=NAMES[lang], q=q), lang) for q in _CROSS_EN]
    for lang, tmpl in _ASK_EN.items():
        out += [(tmpl.format(q=q), "en") for q in QUESTIONS[lang][:3]]
    return out


def reply_language(text: str) -> Optional[str]:
    d = detect_language(text, min_letters=8, min_words=8)
    return d[0] if d else None


@dataclass
class LangProbe:
    langs: list[str]
    means: torch.Tensor                  # [n_depth, n_lang, d]
    var: torch.Tensor                    # [n_depth, d] pooled within-class variance (+ shrinkage)
    loo_accuracy: list[float]            # per depth: held-out prompts, read at their decision point
    temperature: list[float]             # per depth, fitted on the leave-one-out distances (calibrated p)
    counts: dict[str, int]
    n_mismatch: int                      # calibration prompts answered in another language than asked
    meta: dict = field(default_factory=dict)

    def log_probs(self, resid: torch.Tensor) -> torch.Tensor:
        """[n_depth, n_lang] log posterior for one run's residual stack [n_depth, d]."""
        diff = resid[:, None, :] - self.means                    # [D, K, d]
        dist = (diff ** 2 / self.var[:, None, :]).sum(-1)         # Mahalanobis (diagonal)
        return torch.log_softmax(-dist / torch.tensor(self.temperature)[:, None], dim=-1)

    def direction(self, depth: int, to_lang: str, from_lang: str) -> torch.Tensor:
        i, j = self.langs.index(to_lang), self.langs.index(from_lang)
        return self.means[depth, i] - self.means[depth, j]

    def best_depth(self, max_frac: float = 0.85) -> int:
        n = len(self.loo_accuracy)
        lim = max(1, int(max_frac * (n - 1)))
        return max(range(1, lim + 1), key=lambda i: self.loo_accuracy[i])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"langs": self.langs, "means": self.means, "var": self.var, "loo": self.loo_accuracy,
                    "temperature": self.temperature,
                    "counts": self.counts, "n_mismatch": self.n_mismatch, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path) -> "LangProbe":
        z = torch.load(path, weights_only=False)
        return cls(z["langs"], z["means"], z["var"], z["loo"], z["temperature"], z["counts"], z["n_mismatch"],
                   z.get("meta", {}))


def _key(lm: LM) -> str:
    cfg = json.dumps({k: str(v) for k, v in sorted(lm.cfg.to_dict().items())}, sort_keys=True)
    w = lm.unembed.weight
    probe = f"{float(w[0, :8].float().sum()):.6f}{float(w[-1, -8:].float().sum()):.6f}"
    return hashlib.sha1((cfg + probe + "v4").encode()).hexdigest()[:16]


def _fit(X: torch.Tensor, y: list[int], k: int, shrink: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Class means and pooled diagonal variance (shrunk toward the mean variance). X: [n, d]."""
    Y = torch.nn.functional.one_hot(torch.tensor(y), k).float()
    means = (Y.T @ X) / Y.sum(0).clamp(min=1)[:, None]
    resid = X - means[torch.tensor(y)]
    var = resid.pow(2).mean(0)
    var = (1 - shrink) * var + shrink * var.mean() + 1e-6
    return means, var


def calibrate(lm: LM, *, max_new: int = 32, cache: bool = True, min_per_class: int = 4,
              in_reply: int = 3, log=print) -> LangProbe:
    """in_reply: also sample the residual after 1..in_reply tokens of the model's own reply.

    At the decision point alone, "asked for French" and "answered in French" nearly always
    coincide in calibration, so the probe learns the request (Qwen2.5-0.5B: "Answer in French
    only: ..." answered in English read as French, p=1.00). Inside the reply the language being
    written is unambiguous. Leave-one-out holds out whole prompts.
    """
    path = CACHE / f"{lm.name}-{_key(lm)}.pt"
    if cache and path.exists():
        return LangProbe.load(path)
    rows: list[tuple[torch.Tensor, str, int, int, bool]] = []      # resid, answered, prompt, k, mismatch
    mismatch = 0
    prompts = calibration_prompts()
    for n, (p, asked) in enumerate(prompts):
        ids = lm.encode([{"role": "user", "content": p}])
        gen = lm.generate(ids, max_new)
        text = lm.tok.decode([t for t in gen if t not in lm.eot_ids], skip_special_tokens=True)
        got = reply_language(text)
        if got is None or got not in LANGS:
            continue
        mismatch += got != asked
        body = [t for t in gen if t not in lm.eot_ids]
        for k in range(0, min(in_reply, len(body) - 1) + 1):
            x = torch.cat([ids, torch.tensor([body[:k]], dtype=torch.long)], dim=1) if k else ids
            rows.append((lm.run(x, capture=("resid",)).resid, got, n, k, got != asked))
        if log and n % 25 == 0:
            log(f"  language probe: {n}/{len(prompts)} calibration prompts")
    counts: dict[str, int] = {}
    for _, g in {(r[2], r[1]) for r in rows}:
        counts[g] = counts.get(g, 0) + 1                         # prompts per language
    langs = [x for x in LANGS if counts.get(x, 0) >= min_per_class]
    rows = [r for r in rows if r[1] in langs]
    fit_rows = [r for r in rows if r[3] >= 1] or rows             # inside the reply: unambiguous labels
    dec_rows = [r for r in rows if r[3] == 0]                     # decision points: where the probe is used
    R = torch.stack([r[0] for r in fit_rows])                    # [n, D, d]
    y = [langs.index(r[1]) for r in fit_rows]
    grp = [r[2] for r in fit_rows]
    Rd = torch.stack([r[0] for r in dec_rows])
    yd = torch.tensor([langs.index(r[1]) for r in dec_rows])
    gd = [r[2] for r in dec_rows]
    mis = torch.tensor([r[4] for r in dec_rows])
    D = R.shape[1]
    K = len(langs)
    Y = torch.nn.functional.one_hot(torch.tensor(y), K).float()
    grid = [10.0 ** (e / 4) for e in range(-4, 25)]               # 0.1 ... 1e6
    means, var, loo, temps, mis_acc = [], [], [], [], []
    for depth in range(D):
        X = R[:, depth]
        m, v = _fit(X, y, K)
        means.append(m)
        var.append(v)
        # Score at the decision points, leaving out every sample of that prompt from the fit.
        S, N = Y.T @ X, Y.sum(0)
        dists = torch.empty(len(dec_rows), K)
        for i, g in enumerate(gd):
            ii = [j for j, x in enumerate(grp) if x == g]
            Sg, Ng = (Y[ii].T @ X[ii], Y[ii].sum(0)) if ii else (0.0, 0.0)
            mm = (S - Sg) / (N - Ng).clamp(min=1)[:, None]
            dists[i] = (((Rd[i, depth] - mm) ** 2) / v).sum(-1)
        hit = dists.argmin(-1) == yd
        loo.append(float(hit.float().mean()))
        mis_acc.append(float(hit[mis].float().mean()) if mis.any() else float("nan"))
        nll = [float(-torch.log_softmax(-dists / t, -1)[torch.arange(len(yd)), yd].mean()) for t in grid]
        temps.append(grid[min(range(len(grid)), key=lambda i: nll[i])])
    probe = LangProbe(langs, torch.stack(means), torch.stack(var), loo, temps,
                      {g: counts[g] for g in langs}, mismatch,
                      {"prompts": len(prompts), "fit_samples": len(fit_rows), "decision_points": len(dec_rows),
                       "mismatch_accuracy": mis_acc, "max_new": max_new, "in_reply": in_reply})
    if cache:
        probe.save(path)
    return probe


@dataclass
class LangTrace:
    expected: str
    answered: str
    p_expected: list[float]              # per depth
    p_answered: list[float]
    top: list[str]                       # most likely language per depth
    switch_depth: Optional[int]          # first depth after which answered stays above expected
    expected_peak_depth: int


def trace(probe: LangProbe, lm: LM, ids: torch.Tensor, expected: str, answered: str) -> LangTrace:
    lp = probe.log_probs(lm.run(ids, capture=("resid",)).resid).exp()
    ie, ia = probe.langs.index(expected), probe.langs.index(answered)
    pe, pa = lp[:, ie].tolist(), lp[:, ia].tolist()
    sw = None
    for i in range(1, len(pe)):
        if all(pa[j] > pe[j] for j in range(i, len(pe))):
            sw = i
            break
    top = [probe.langs[int(r.argmax())] for r in lp]
    return LangTrace(expected, answered, pe, pa, top, sw, max(range(len(pe)), key=lambda i: pe[i]))


def steer_tests(probe: LangProbe, lm: LM, clf, examples: list, expected: str, answered: str,
                depth: int, scales=(1.0, 2.0)) -> dict[str, Any]:
    """Generation tests with the calibrated direction mu(answered) - mu(expected) at `depth`.

    ablate: remove the direction everywhere; steer: add s * (mu(expected) - mu(answered)) after
    layer depth-1; null: add the same norm along an unrelated pair (two other languages).
    A test "fixes" an example when the reply comes out in the expected language.
    """
    v = probe.direction(depth, answered, expected)
    others = [x for x in probe.langs if x not in (expected, answered)]
    null_v = probe.direction(depth, others[0], others[1]) if len(others) >= 2 else None
    layer = max(0, depth - 1)
    out: dict[str, Any] = {"depth": depth, "layer": layer, "norm": float(v.norm()), "samples": []}

    def lang_of(ex) -> Optional[str]:
        return reply_language(ex.text)

    def run(name, hooks):
        fixed = 0
        runs = clf.run_many([(ex.name + "+" + name, ex.messages) for ex in examples], hooks=hooks)
        for ex, e2 in zip(examples, runs):
            got = lang_of(e2)
            fixed += got == expected
            out["samples"].append({"test": name, "example": ex.name, "language": got, "reply": e2.text[:120]})
        return fixed

    out["ablate_fixed"] = run("ablate", lambda: lm.steer(ablate=v))
    best = 0
    for s in scales:
        f = run(f"steer_x{s}", lambda s=s: lm.steer(add=(-v, s, [layer])))
        if f > best:
            best, out["steer_scale"] = f, s
    out["steer_fixed"] = best
    if null_v is not None:
        nv = null_v / (null_v.norm() + 1e-12) * v.norm()
        out["null_pair"] = f"{others[0]}-{others[1]}"
        out["null_fixed"] = run("null", lambda: lm.steer(add=(nv, out.get("steer_scale", 1.0), [layer])))
    out["of"] = len(examples)
    return out


def summary_line(tr: LangTrace, probe: LangProbe) -> str:
    n = len(tr.p_expected)
    pk = tr.expected_peak_depth
    s = (f"the calibrated language probe ({len(probe.langs)} languages, accuracy "
         f"{probe.loo_accuracy[probe.best_depth()]:.0%} on held-out prompts at its best layer)")
    if tr.p_expected[pk] >= 0.3:
        s += f" reads {NAMES.get(tr.expected, tr.expected)} most strongly after layer {pk - 1} (p={tr.p_expected[pk]:.2f})"
    else:
        s += f" never reads {NAMES.get(tr.expected, tr.expected)} with confidence (p <= {tr.p_expected[pk]:.2f})"
    if tr.switch_depth is not None and tr.switch_depth < n:
        s += (f", and {NAMES.get(tr.answered, tr.answered)} from layer {tr.switch_depth - 1} on "
              f"(p={tr.p_answered[tr.switch_depth]:.2f} there, {tr.p_answered[-1]:.2f} at the last layer)")
    else:
        s += f"; {NAMES.get(tr.answered, tr.answered)} never overtakes it in the residual stream"
    return s


def _round(xs: list[float]) -> list[float]:
    return [round(x, 3) if not math.isnan(x) else None for x in xs]
