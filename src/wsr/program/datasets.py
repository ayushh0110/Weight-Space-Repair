"""Public datasets wsr knows how to turn into its JSONL rows, fetched only after asking.

`wsr data list` shows them; `wsr data fetch NAME` (and `wsr objective from-failure --dataset auto`)
asks before downloading anything. A file already in the Hugging Face cache is used without asking
(nothing is downloaded). On "no", the caller falls back (from-failure uses rewrites of the prompt) or
stops with what it needed.

Rows are written to `~/.cache/wsr/datasets/<name>.jsonl` (or `WSR_DATA_DIR`) as
{id, prompt, answer?, expect?, lang?}. `expect: refuse` marks a prompt that should be refused (XSTest's
unsafe half); detector graders score it the other way round.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..consent import Consent, NeedsConsent


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    file: str
    about: str
    size: str
    convert: Callable[[Optional[Path]], list[dict]]
    grader: Optional[str] = None
    builtin: bool = False            # shipped with wsr: written without a download or a question


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _read_parquet(p: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as e:
        raise ValueError("this dataset is parquet: pip install pyarrow") from e
    return pq.read_table(str(p)).to_pylist()


def _math500(p: Path) -> list[dict]:
    return [{"id": r.get("unique_id", f"math500:{i}"), "prompt": r["problem"], "answer": r["answer"]}
            for i, r in enumerate(_read_jsonl(p))]


def _gsm8k(p: Path) -> list[dict]:
    out = []
    for i, r in enumerate(_read_parquet(p)):
        ans = str(r["answer"]).split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k:{i}", "prompt": r["question"], "answer": ans})
    return out


def _xstest(p: Path) -> list[dict]:
    out = []
    with open(p, encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            unsafe = str(r.get("label", "")).strip().lower() == "unsafe" or str(r.get("type", "")).startswith("contrast")
            out.append({"id": f"xstest:{r.get('id', i)}", "prompt": r["prompt"],
                        **({"expect": "refuse"} if unsafe else {}), "type": r.get("type", "")})
    return out


def _ifeval(p: Path) -> list[dict]:
    return [{"id": f"ifeval:{r.get('key', i)}", "prompt": r["prompt"]} for i, r in enumerate(_read_jsonl(p))]


def _wsr_refusal(p: Optional[Path]) -> list[dict]:
    from ..deep import refusal_pools as RP

    return RP.dataset_rows()


def _mgsm(lang: str):
    def conv(p: Path) -> list[dict]:
        out = []
        for i, r in enumerate(_read_parquet(p)):
            ans = r.get("answer_number")
            out.append({"id": f"mgsm-{lang}:{i}", "prompt": r["question"], "lang": lang,
                        **({"answer": str(int(ans)) if float(ans).is_integer() else str(ans)} if ans is not None
                           else {})})
        return out
    return conv


SOURCES: dict[str, Source] = {s.name: s for s in [
    Source("math500", "HuggingFaceH4/MATH-500", "test.jsonl", "500 competition maths problems with answers",
           "0.4 MB", _math500, "math"),
    Source("gsm8k", "openai/gsm8k", "main/test-00000-of-00001.parquet", "1,319 grade-school maths word problems",
           "0.4 MB", _gsm8k, "math"),
    Source("xstest", "Paul/XSTest", "xstest_prompts.csv",
           "250 safe prompts that sound unsafe, and 200 unsafe ones that must stay refused", "0.05 MB", _xstest,
           "detector:refusal"),
    Source("wsr-refusal", "", "", "wsr's own refusal contrast pools: 640 safe requests that sound alarming, 240 "
           "harmful ones that must be refused (50 as matched pairs), 61 plain ones. The harm probe is calibrated on "
           "them: use them as practice data, not for validation", "0.1 MB", _wsr_refusal, "detector:refusal",
           builtin=True),
    Source("ifeval", "google/IFEval", "ifeval_input_data.jsonl", "541 prompts with format instructions (graded by "
           "wsr's detectors only: the IFEval checker is not built)", "0.2 MB", _ifeval, "detector:any"),
    *[Source(f"mgsm-{lg}", "juletxara/mgsm", f"{lg}/test-00000-of-00001.parquet",
             f"250 GSM8K problems translated into {lg}", "0.05 MB", _mgsm(lg), "math")
      for lg in ("bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")],
]}

# Which source fits a failure (from-failure --dataset auto).
FOR_FAILURE = {"LOOP": "math500", "TRUNCATED_LOOP": "math500", "REFUSAL": "xstest", "EMPTY": "ifeval",
               "GARBAGE": "ifeval", "TEMPLATE_LEAK": "ifeval", "WRONG_ANSWER": "math500"}


def for_failure(label: str, lang: Optional[str] = None) -> Optional[str]:
    if label == "LANGUAGE":
        return f"mgsm-{lang}" if lang and f"mgsm-{lang}" in SOURCES else None
    return FOR_FAILURE.get(label)


def data_dir() -> Path:
    return Path(os.environ.get("WSR_DATA_DIR") or Path.home() / ".cache" / "wsr" / "datasets")


def _cached(src: Source) -> Optional[Path]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    p = try_to_load_from_cache(src.repo, src.file, repo_type="dataset")
    return Path(p) if isinstance(p, str) and Path(p).exists() else None


def fetch(name: str, consent: Consent, *, required: bool = True) -> Optional[Path]:
    """The dataset as wsr JSONL. Asks before downloading; a file already cached is used without asking.
    On "no": None (required=False, the caller falls back) or NeedsConsent."""
    if name not in SOURCES:
        raise ValueError(f"unknown dataset {name!r}: one of {', '.join(SOURCES)}")
    src = SOURCES[name]
    out = data_dir() / f"{name}.jsonl"
    if out.exists() and not src.builtin:
        return out
    if src.builtin:                                      # shipped with wsr: nothing to download, nothing to ask
        _write_rows(out, src.convert(None))
        return out
    raw = _cached(src)
    if raw is None:
        q = f"download the {name} dataset ({src.size}) from huggingface.co/datasets/{src.repo}"
        if not consent.ask("download", q, detail=f"{src.about}. Saved under {data_dir()}. No: "
                                                 + ("wsr continues without it" if not required else "this step stops")):
            if required:
                raise NeedsConsent("download", f"the {name} dataset", "it was not downloaded")
            return None
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ValueError("downloading needs huggingface_hub (pip install huggingface_hub)") from e
        raw = Path(hf_hub_download(src.repo, src.file, repo_type="dataset"))
    _write_rows(out, src.convert(raw))
    return out


def _write_rows(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")                  # whole or not at all: an interrupted run leaves no half file
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)


def listing() -> list[dict[str, Any]]:
    return [{"name": s.name, "repo": s.repo, "about": s.about, "size": s.size, "grader": s.grader, "builtin": s.builtin,
             "local": s.builtin or (data_dir() / f"{s.name}.jsonl").exists() or _cached(s) is not None}
            for s in SOURCES.values()]

