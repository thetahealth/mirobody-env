"""attribute_failures.py -- attribute retried model-call failures to their cause.

    uv run python tools/attribute_failures.py                 # all frozen packs
    uv run python tools/attribute_failures.py --all           # every batch on disk
    uv run python tools/attribute_failures.py --batch <path>   # one specific batch
    uv run python tools/attribute_failures.py --selftest      # classifier self-test only
    uv run python tools/attribute_failures.py --json <file>

Retries that eventually succeed leave no trace in the scores, so this reads the failed
attempts in `responses.jsonl` and classifies each error text:

| class | criterion | blame |
|---|---|---|
| `infra` | 5xx, 429, gateway timeout, connection error, broken stream | provider |
| `auth` | 401, invalid token, quota exhausted | the operator's key pool |
| `budget` | truncated by `max_tokens`; reasoning used up the budget | the run's configuration |
| `no_answer` | reasoning but no answer, with the budget not used up | the model |
| `format` | response received but the JSON is broken | the model (or the prompt) |
| `unknown` | matches nothing | unclassified |

`unknown` is never folded into another class; the exit code is nonzero while any remain.
Cells scored without a retry are not covered (see `llm.ceiling_flags`).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path[:0] = [str(ROOT)]

#: attribution class -> (who's responsible, one-line reason)
BLAME = {
    "infra":   ("provider", "provider-side fault -- **must not be recorded as a model capability defect**"),
    "auth":    ("our key pool", "auth/quota -- an operational problem on our side"),
    "budget":  ("our configuration", "`max_tokens` too small (reasoning tokens count against it too) -- **must not be recorded as a model capability defect**"),
    "format":  ("model / prompt", "got a response but the JSON is broken -- could be the model or our prompt"),
    "no_answer": ("model", "gave reasoning but no answer, and the budget **wasn't** used up -- adjusting the budget won't fix it"),
    "unknown": ("unclassified", "the classifier doesn't recognize it -- **must not be folded into another class**"),
}

#: The classes that should not be blamed on the model.
NOT_MODEL = ("infra", "auth", "budget")

#: Reasoning chars / budget tokens at or above which a "reasoning but no answer" failure
#: counts as `budget`; below it, `no_answer`. 2.0 is the calibrated lower bound on chars per
#: token (`haenv/llm.py::CHARS_PER_TOKEN`).
BUDGET_CHARS_PER_TOKEN_MIN = 2.0

#: Most specific rule first: `budget` precedes `format`, since a truncated JSON also reports
#: `JSONDecodeError`.
_RULES: list[tuple[str, re.Pattern]] = [
    # `maxOutputTokens` is the Google direct path's truncation wording.
    ("budget", re.compile(r"只有推理没有作答|推理\s*\d+\s*字符、作答\s*0\s*字符"
                          r"|max_tokens\s*截断|被\s*max_tokens|truncat"
                          r"|maxOutputTokens|MAX_TOKENS", re.I)),
    ("auth",   re.compile(r"\b401\b|\b403\b|invalid[ _]token|unauthor|api[ _]key"
                          r"|quota|exhaust|余额|欠费|额度", re.I)),
    # A missing terminating frame is a transport break (`infra`); `budget` is told apart by
    # its "reasoning N chars, answer 0 chars" wording, not by the character count.
    ("infra",  re.compile(r"未收到终止帧|finish_reason\s*缺失|流被截断"
                          r"|\b50[0234]\b|\b429\b|too\s*many\s*requests|rate[ _-]?limit"
                          r"|gateway\s*time-?out|bad\s*gateway"
                          r"|service\s*unavailable|connection|timed?\s*out|timeout"
                          r"|URLError|RemoteDisconnected|IncompleteRead", re.I)),
    ("format", re.compile(r"JSONDecode|Expecting\s|Unterminated|Extra data"
                          r"|无法解析|取不出\s*JSON|could\s*not\s*extract\s*JSON", re.I)),
]


NO_ANSWER = "no_answer"

_REASON_CHARS = re.compile(r"推理\s*(\d+)\s*字符")


def classify(err: str, budget_tokens: int | None = None) -> str:
    """One error text -> attribution class; unrecognized -> `unknown`.

    The magnitude check against `budget_tokens` runs only when the budget is known.
    """
    e = str(err or "")
    for name, rx in _RULES:
        if rx.search(e):
            if name == "budget" and budget_tokens:
                m = _REASON_CHARS.search(e)
                if m and int(m.group(1)) < int(budget_tokens) * BUDGET_CHARS_PER_TOKEN_MIN:
                    return NO_ANSWER
            return name
    return "unknown"


def _report_classes() -> list[str]:
    """Report classes, derived from `BLAME` so no class can drop out of the table."""
    order = [k for k, _ in _RULES] + [NO_ANSWER, "unknown"]
    return [k for k in order if k in BLAME] + [k for k in BLAME if k not in order]


def _attempts(row: dict) -> list[str]:
    """Normalize `failed_attempts` into a list of error texts (unrecognized shape -> empty)."""
    fa = row.get("failed_attempts")
    if not fa:
        return []
    out = []
    for a in (fa if isinstance(fa, list) else [fa]):
        if isinstance(a, dict):
            out.append(str(a.get("error") or a.get("err") or a))
        else:
            out.append(str(a))
    return out


def scan(batch_dirs: list[pathlib.Path]) -> dict:
    """Scan `responses.jsonl` per batch; the result includes the scan surface (denominators)."""
    per_model: dict = collections.defaultdict(collections.Counter)
    per_batch: dict = collections.defaultdict(collections.Counter)
    n_rows = n_with = 0
    samples: dict = collections.defaultdict(list)
    seen_batches = 0
    n_budget_known = n_budget_unknown = 0
    for d in batch_dirs:
        p = d / "responses.jsonl"
        if not p.is_file():
            continue
        seen_batches += 1
        # The budget comes from the batch's own `batch.json`; without it, sentence matching decides.
        budgets = _budgets_of(d)
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:                                  # noqa: BLE001
                continue
            n_rows += 1
            errs = _attempts(r)
            if errs:
                n_with += 1
            b = budgets.get(str(r.get("solver")))
            for e in errs:
                if b:
                    n_budget_known += 1
                else:
                    n_budget_unknown += 1
                k = classify(e, b)
                per_model[r.get("solver")][k] += 1
                per_batch[d.name][k] += 1
                if len(samples[k]) < 3:
                    samples[k].append((r.get("solver"), e[:130]))
    return {"per_model": {k: dict(v) for k, v in per_model.items()},
            "per_batch": {k: dict(v) for k, v in per_batch.items()},
            "n_rows": n_rows, "n_rows_with_failures": n_with,
            "n_batches_scanned": seen_batches, "n_batches_given": len(batch_dirs),
            "n_failures_with_known_budget": n_budget_known,
            "n_failures_without_budget": n_budget_unknown,
            "samples": {k: v for k, v in samples.items()}}


def _budgets_of(d: pathlib.Path) -> dict:
    """Per-model `max_tokens` recorded in this batch's `batch.json` (empty if not recorded)."""
    p = d / "batch.json"
    if not p.is_file():
        return {}
    try:
        pm = (json.loads(p.read_text(encoding="utf-8")).get("sampling") or {}).get("per_model")
    except Exception:                                          # noqa: BLE001
        return {}
    return {str(n): int(m["max_tokens"]) for n, m in (pm or {}).items()
            if isinstance(m, dict) and m.get("max_tokens")}


def frozen_batches() -> list[pathlib.Path]:
    """The currently frozen packs."""
    from haenv.anchor import frozen_packs
    return [ROOT / f"results/joint_dx/{job}/{b}" for job, b in sorted(frozen_packs().items())]


def all_batches(*, include_superseded: bool = False) -> list[pathlib.Path]:
    """Every batch under `results/` (symlinks followed). Runs under `results/_superseded/`
    (aborted or replaced) are left out unless `include_superseded`: they are not evidence
    for any current declaration, but failure attribution may still want them.
    """
    out, seen = [], set()
    for root, _dirs, files in os.walk(ROOT / "results", followlinks=True):
        if (not include_superseded
                and "_superseded" in pathlib.Path(root).relative_to(ROOT / "results").parts):
            continue
        if "responses.jsonl" in files:
            rp = pathlib.Path(root).resolve()
            if rp not in seen:
                seen.add(rp); out.append(pathlib.Path(root))
    return sorted(out)


# ---------------------------------------------------------------- self-test

def selftest() -> int:
    """Positive/negative controls for the classifier; attribution runs only after they pass."""
    cases = [
        ("HTTPError: HTTP Error 504: Gateway Time-out", "infra"),
        ("urllib.error.URLError: <urlopen error timed out>", "infra"),
        ("HTTP Error 503: Service Unavailable", "infra"),
        ("{\"error\":{\"message\":\"Invalid token (request id: …)\"}}", "auth"),
        ("This token quota is exhausted TokenStatusExhausted", "auth"),
        ("ValueError: 流式响应只有推理没有作答(推理 110234 字符、作答 0 字符)", "budget"),
        ("ValueError: 响应被 max_tokens 截断且无内容", "budget"),
        # Transport-layer stream break (no terminating frame): distinct from `budget`
        # even though both can present as "answer 0 chars".
        ("ValueError: 流式响应未收到终止帧(finish_reason 缺失):作答 0 字符、推理 10217 字符、"
         "usage=有 —— 判为流被截断", "infra"),
        ("JSONDecodeError: Expecting ',' delimiter: line 1 column 2232", "format"),
        ("JSONDecodeError: Unterminated string starting at …", "format"),
        ("some new error shape we've never seen zzz", "unknown"),
    ]
    bad = []
    for text, want in cases:
        got = classify(text)
        ok = got == want
        print(f"  {'✅' if ok else '❌'} {want:8s} ← {text[:58]}"
              + ("" if ok else f"   got {got}"))
        if not ok:
            bad.append((text[:40], want, got))

    # Negative control (1): a truncated JSON is `budget`, not `format`.
    t = "JSONDecodeError: Unterminated string — 响应被 max_tokens 截断"
    ok = classify(t) == "budget"
    print(f"  {'✅' if ok else '❌'} Ordering: JSON broken by truncation ⇒ must be judged `budget`, not `format` (got {classify(t)})")
    if not ok:
        bad.append(("rule ordering", "budget", classify(t)))

    # Negative control (1b): "answer 0 chars" alone must not decide the class.
    t2 = "ValueError: 流式响应未收到终止帧:作答 0 字符、推理 88595 字符"
    t3 = "ValueError: 流式响应只有推理没有作答(推理 88595 字符、作答 0 字符)"
    ok = classify(t2) == "infra" and classify(t3) == "budget"
    print(f"  {'✅' if ok else '❌'} Stream-break vs. budget-exhaustion must stay distinct (both contain the answer-0-chars marker): "
          f"stream-break⇒{classify(t2)} · budget⇒{classify(t3)}")
    if not ok:
        bad.append(("stream-break/budget distinguishable", "infra+budget", f"{classify(t2)}+{classify(t3)}"))

    # Negative control (1c): the same sentence gives different verdicts by budget size.
    t4 = "ValueError: 流式响应只有推理没有作答(推理 8886 字符、作答 0 字符)"
    t5 = "ValueError: 流式响应只有推理没有作答(推理 108921 字符、作答 0 字符)"
    ok = (classify(t4, 16000) == NO_ANSWER and classify(t5, 32000) == "budget"
          # Positive control: without a budget, sentence matching decides.
          and classify(t4) == "budget")
    print(f"  {'✅' if ok else '❌'} Magnitude: reasoning 8,886 chars @16k ⇒ {classify(t4, 16000)} · "
          f"reasoning 108,921 @32k ⇒ {classify(t5, 32000)} · no budget given ⇒ {classify(t4)} (still judged by sentence matching)")
    if not ok:
        bad.append(("magnitude criterion", f"{NO_ANSWER}+budget+budget",
                    f"{classify(t4, 16000)}+{classify(t5, 32000)}+{classify(t4)}"))

    # Negative control (1d): the Google direct path's truncation wording is `budget`.
    t6 = "ValueError: 响应被 maxOutputTokens 截断且无内容"
    ok = classify(t6) == "budget"
    print(f"  {'✅' if ok else '❌'} Wording used on the Google-direct truncation path ⇒ must be judged `budget` (got {classify(t6)})")
    if not ok:
        bad.append(("maxOutputTokens", "budget", classify(t6)))

    # Negative control (1e): every class the classifier can produce appears in the report.
    _produced = {NO_ANSWER, "unknown"} | {k for k, _ in _RULES}
    _missing = sorted(_produced - set(_report_classes()))
    ok = not _missing and set(_report_classes()) == set(BLAME)
    print(f"  {'✅' if ok else '❌'} Report table header covers everything the classifier can produce"
          f"(table has {_report_classes()}; missing {_missing or 'none'})")
    if not ok:
        bad.append(("header coverage", "full coverage", str(_missing)))

    # Negative control (2): empty/None is not a failure.
    ok = _attempts({"failed_attempts": None}) == [] and _attempts({}) == []
    print(f"  {'✅' if ok else '❌'} Negative control: a row with no failed_attempts must produce no attribution at all")
    if not ok:
        bad.append(("empty input", "[]", "non-empty"))

    # Negative control (3): an empty scan surface must not read as "attribution clean".
    r = scan([ROOT / "__no_such_batch__"])
    ok = r["n_batches_scanned"] == 0 and r["n_rows"] == 0
    print(f"  {'✅' if ok else '❌'} Negative control: n_batches_scanned=0 when no batch is found (callers must refuse to draw conclusions on this basis)")
    if not ok:
        bad.append(("empty scan surface", "0", str(r["n_batches_scanned"])))

    print(f"\nClassifier self-test: {'all passed' if not bad else '❌ failed: ' + str(bad)}")
    return 1 if bad else 0


# ---------------------------------------------------------------- report

def render(res: dict) -> str:
    L: list[str] = []
    w = L.append
    nb, ng = res["n_batches_scanned"], res["n_batches_given"]
    w(f"Scan surface: {nb}/{ng} batches have `responses.jsonl` · {res['n_rows']} rows · "
      f"of which {res['n_rows_with_failures']} rows carry a failed retry")
    _kb, _ub = res["n_failures_with_known_budget"], res["n_failures_without_budget"]
    w(f"\nThe magnitude criterion's **exercised surface**: {_kb} failures know how large a `max_tokens` they ran under, "
      f"{_ub} don't (that batch's `batch.json` didn't record it). "
      f"{'**Not exercised on this data** -- ' if not _kb else ''}"
      "the ones with an unknown budget **fall back to sentence matching**, so the class where "
      "\"the model gave reasoning but no answer, and the budget wasn't actually used up\" (`no_answer`) "
      "can't be told apart for them. "
      "This isn't a defect, it's **unmeasured, so not used as measured**: answering a historical batch "
      "with today's `config.yaml` would give a wrong reading across config generations, so the magnitude "
      "judgment only runs when a batch recorded its own `max_tokens`.")
    if nb == 0:
        w("**Not a single batch was scanned -- drawing no conclusion.** (An empty scan surface and \"clean attribution\" look identical in the output)")
        return "\n".join(L)

    tot = collections.Counter()
    for v in res["per_model"].values():
        tot.update(v)
    n_all = sum(tot.values())
    w("")
    w(f"## Attribution ({n_all} failed retries total)")
    w("")
    w("| Class | Count | Share | Blamed on |")
    w("|---|---|---|---|")
    for k in _report_classes():
        c = tot.get(k, 0)
        who, why = BLAME[k]
        w(f"| `{k}` | **{c}** | {c / n_all * 100:.1f}% | {who} —— {why} |")
    not_model = sum(tot.get(k, 0) for k in NOT_MODEL)
    w("")
    w(f"**{not_model}/{n_all} = {not_model / n_all * 100:.1f}% of failed retries should not be blamed on the model** "
      f"({' + '.join(f'`{k}`' for k in NOT_MODEL)}), yet `eval.jsonl` carries no trace of retry attribution at all.")
    w("")
    w("## By model")
    w("")
    _cls = _report_classes()
    w("| Model | Total | " + " | ".join(_cls) + " |")
    w("|---|---|" + "---|" * len(_cls))
    for m, c in sorted(res["per_model"].items(), key=lambda x: -sum(x[1].values())):
        n = sum(c.values())
        w(f"| `{m}` | **{n}** | " + " | ".join(str(c.get(k, 0)) for k in _cls) + " |")
    w("")
    w("## Samples (up to 3 per class)")
    for k, v in res["samples"].items():
        w(f"* **`{k}`** — " + " · ".join(f"[{m}] {e}" for m, e in v))
    return "\n".join(L)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--all", action="store_true", help="Scan every batch on disk, including results/_superseded/, not just the frozen packs")
    ap.add_argument("--batch", help="Scan only this one batch directory")
    ap.add_argument("--json", help="Write the structured result to this file")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if selftest():
        print("Classifier self-test failed -- **no attribution will be produced**. Never draw conclusions with an unverified classifier.")
        return 1
    print()

    dirs = ([pathlib.Path(a.batch)] if a.batch
            else all_batches(include_superseded=True) if a.all else frozen_batches())
    res = scan(dirs)
    print(render(res))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
        print(f"\n[attribute_failures] -> {a.json}")
    # Nonzero while anything is unclassified.
    unk = sum(v.get("unknown", 0) for v in res["per_model"].values())
    if unk:
        print(f"\n{unk} failures are **unclassified** -- add a rule to `_RULES` plus a self-test fixture; don't leave it sitting in unknown.")
    return 1 if unk else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
