"""Which cells and models enter one track's board.

The default rule (`fill_unanswered`) puts every case x every declared model on the
board and scores an unanswered cell 0; `answered_view` gives the switchable view in which
each model is scored on its own answered cases only. Both the board readout and the demo
exporter apply these functions.

The coverage rule (`track_coverage` / `decide_unranked`) is kept as an option: a model
whose answered cells cover less than MIN_COVERAGE of a track's cases, or that is named in
the manual exclusion list, is left out of that track's ranking and reported separately.

Answered cells are eval rows: a row exists only once its cell is finished, and the last
row per (case, solver) is the effective one.
"""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)

from collections.abc import Iterable

MIN_COVERAGE = 0.9

#: A model whose answer rate on a track is below this is not ranked on that track and is listed
#: separately; its unanswered cells would otherwise put availability, not capability, on the
#: board. Answer rate = answered cells / the track's cases.
MIN_ANSWER_RATE = 0.95


def answer_rate_unranked(rows: Iterable[dict], cases: Iterable[str], models: Iterable[str],
                         min_rate: float = MIN_ANSWER_RATE) -> dict[str, dict]:
    """`{model: {"answered", "of", "rate", "why"}}` for the declared models below `min_rate`.

    A cell is answered when its row exists and `unanswered_reason` finds a judgeable answer
    in it (the same test `fill_unanswered` uses to build zero rows).
    """
    cases = sorted(set(cases))
    cells = {(r.get("case"), r.get("solver")): r for r in rows}
    out = {}
    for m in sorted(set(models)):
        answered = sum(1 for c in cases if unanswered_reason(cells.get((c, m))) is None)
        rate = answered / len(cases) if cases else 0.0
        if rate < min_rate:
            out[m] = {"answered": answered, "of": len(cases), "rate": round(rate, 4),
                      "why": f"answer rate {answered}/{len(cases)} = {rate:.3f} < MIN_ANSWER_RATE {min_rate}"}
    return out


def track_coverage(rows: Iterable[dict], n_cases: int, models: Iterable[str] = (),
                   baselines: Iterable[str] = ()) -> dict:
    """Answered cells per real model; `n_cases` is the track's case count (batch.json)."""
    skip = set(baselines)
    cells = {(r.get("case"), r.get("solver")): r for r in rows}
    n = max(int(n_cases or 0), len({c for c, _ in cells}))
    real = {str(s) for _, s in cells if s is not None and s not in skip} | (set(models) - skip)
    per = {}
    for m in sorted(real):
        answered = sum(1 for (_, s) in cells if s == m)
        per[m] = {"answered": answered, "of": n,
                  "coverage": round(answered / n, 4) if n else 0.0}
    return {"n_cases": n, "models": per}


def decide_unranked(coverage: dict, manual: Iterable[str] = (),
                    min_coverage: float = MIN_COVERAGE) -> dict[str, list[str]]:
    """{model: [reasons]} for the models of one track's `track_coverage` that are not ranked."""
    manual = set(manual)
    out = {}
    for m, e in coverage["models"].items():
        why = []
        if m in manual:
            why.append("named in EXCLUDE_FROM_RANK")
        if e["coverage"] < min_coverage:
            why.append(f"coverage {e['answered']}/{e['of']} = {e['coverage']:.3f} "
                       f"< MIN_COVERAGE {min_coverage}")
        if why:
            out[m] = why
    return out


# ---- Default board rule: an unanswered cell scores 0 -------------------------------------
#
# Every (case, model) cell of a track enters the board. A cell is unanswered when the
# track's eval rows hold no row for it (the model had not finished it when the batch
# closed), or when its row carries no judgeable answer:
#   * eval `overall` is a no-answer disposition: `evaluate.RETRYABLE_OVERALL`
#     (ABORT(no_response), ABORT(unparseable), ABORT(no_answer), ERROR) or any
#     `ABORT(no_answer:<why>)` such as `ABORT(no_answer:budget)`;
#   * the semantic judge status is one of `semantic_report.MISSING_ANSWER`
#     (missing_response, empty_response, unparseable_response).
# An unanswered cell is replaced by a synthetic row, built in memory only, with
# `overall: "NO_ANSWER"` and `unanswered_reason`. Every capability dimension that is
# applicable on the case scores its worst value (0; 1 for a lower-is-better dimension,
# which folds to 0), so the cell adds 0 to the composite. The synthetic row carries no gate
# reading: a missing answer is not an unsafe answer, so it trips no hard gate, and the
# multiplier counts it neither as failed nor as clean (the production treatment of rows
# with no gate reading). Synthetic rows are never written back to a batch.

NO_ANSWER = "NO_ANSWER"
NO_ANSWER_STATUS = "no_answer"


def _no_answer_overall() -> frozenset[str]:
    from haenv.app.evaluate import RETRYABLE_OVERALL
    return frozenset(RETRYABLE_OVERALL)


def unanswered_reason(row: dict | None) -> str | None:
    """Why this eval row holds no judgeable answer, or None when it does. `None` row = no row."""
    if row is None:
        return "no_row"
    if row.get("overall") == NO_ANSWER:
        return row.get("unanswered_reason") or NO_ANSWER
    overall = str(row.get("overall") or "")
    if overall in _no_answer_overall() or overall.startswith("ABORT(no_answer"):
        return f"overall:{overall}"
    from haenv.scoring.semantic.report import MISSING_ANSWER
    status = (row.get("semantic") or {}).get("status")
    if status in MISSING_ANSWER:
        return f"semantic:{status}"
    return None


def _zero_value(profile, name: str):
    d = profile.direction(name)
    if d == "non_monotone":
        return None
    return 1.0 if d == "lower" else 0.0


def _numeric(v) -> bool:
    return isinstance(v, (int, float, bool))


def _synthetic_row(case: str, model: str, peers: list[dict], reason: str, profile) -> dict:
    """The zero row of one unanswered cell, shaped like its answered peers on the same case."""
    from haenv.scoring.quantities import BY_NAME
    from haenv.scoring.quantities import resolve
    from haenv.scoring.semantic.report import PRIMARY
    from haenv.scoring.semantic.report import REPLACED
    from haenv.scoring.semantic.report import held_out_dims
    tmpl = peers[0]
    row = {k: tmpl[k] for k in ("gold_kind", "geometry", "world_sha", "world_knobs", "world_ref",
                                 "world_sha_src", "judging_sha16", "slice_kind", "n_slices", "T",
                                 "dx_kind", "review_warranted", "review_basis_slice", "quant_kind")
           if k in tmpl}
    row.update(case=case, solver=model, overall=NO_ANSWER, unanswered_reason=reason,
               synthetic_no_answer=True)
    dims = set(profile.scored_dims) | set(profile.full_dims())
    for name in sorted(dims):
        if name in REPLACED or name not in BY_NAME:
            continue
        if any(_numeric(resolve(p, name)) for p in peers):
            z = _zero_value(profile, name)
            if z is not None:
                row[name] = z
    # dimensions `rank_ddx` reads from row fields rather than the quantity registry
    for key in ("review_flag_ok", "dx_hit_top1", "held_independent", "join_hit", "urgency_ok"):
        if any(_numeric(p.get(key)) for p in peers):
            row[key] = 0.0
    if any("quant_ok" in p for p in peers):
        row["quant_ok"] = 0.0
    held = held_out_dims()
    states = {}
    semantic_peers = [p for p in peers if isinstance(p.get("semantic"), dict)]
    if not semantic_peers:
        for name in REPLACED:
            if name not in held and any(_numeric(p.get(name)) for p in peers):
                row[name] = 0.0
        return row
    for name in REPLACED:
        peer_states = [p["semantic"]["metric_states"].get(name, p["semantic"]["status"]) for p in semantic_peers]
        if peer_states and all(s == "not_applicable" for s in peer_states):
            states[name], row[name] = "not_applicable", None
        elif name in held:
            states[name], row[name] = "held_out", None
        elif name in PRIMARY or "resolved" in peer_states:
            states[name], row[name] = "resolved", 0.0
        else:
            states[name], row[name] = "not_applicable", None
    row["semantic"] = {"run_id": None, "status": NO_ANSWER_STATUS, "metrics": {}, "metric_states": states,
                       "coverage": {}, "not_applicable": {}, "synthetic_no_answer": True,
                       "clinical_review": "not_clinician_reviewed", "final": False}
    return row


def fill_unanswered(rows: list[dict], cases: Iterable[str], models: Iterable[str],
                    reasons: dict | None = None, profile=None) -> tuple[list[dict], dict]:
    """Rows of the default board: every (case, model) cell, unanswered cells as zero rows.

    `rows` are one track's real-model rows under the semantic view (last row per cell);
    `cases` / `models` are the track's full case list and declared models. `reasons`
    optionally maps (case, model) to a finer reason for cells with no row. The input rows
    are not modified; the synthetic rows exist only in the returned list. Returns
    `(rows, summary)`; the summary gives per model the answered and unanswered cells and
    the reasons.
    """
    if profile is None:
        from haenv.scoring.legacy import load_profile
        profile = load_profile()
    cases = sorted(set(cases))
    models = sorted(set(models))
    cells = {(r["case"], r["solver"]): r for r in rows}
    answered = {k: r for k, r in cells.items() if unanswered_reason(r) is None}
    peers_by_case: dict[str, list[dict]] = {}
    for (c, _m), r in sorted(answered.items()):
        peers_by_case.setdefault(c, []).append(r)
    out = [r for k, r in cells.items() if k in answered or k[1] not in models]
    summary = {m: {"of": len(cases), "answered": 0, "unanswered": 0, "reasons": {}} for m in models}
    no_template = []
    for case in cases:
        for m in models:
            key = (case, m)
            if key in answered:
                summary[m]["answered"] += 1
                continue
            why = unanswered_reason(cells.get(key))
            if why == "no_row" and reasons and key in reasons:
                why = reasons[key]
            peers = peers_by_case.get(case)
            if not peers:
                no_template.append(key)
                continue
            out.append(_synthetic_row(case, m, peers, why, profile))
            s = summary[m]
            s["unanswered"] += 1
            s["reasons"][why] = s["reasons"].get(why, 0) + 1
    for m, s in summary.items():
        s["completion"] = round(s["answered"] / s["of"], 4) if s["of"] else 0.0
    return out, {"models": summary, "n_cases": len(cases),
                 "cases_without_any_answer": sorted({c for c, _ in no_template})}


def answered_view(rows: list[dict]) -> list[dict]:
    """Rows of the answered-only view: answered cells with every applicable primary dimension resolved.

    Each model keeps only its own such cells, so case sets differ between models and the
    composites of this view are not comparable across models.
    """
    from haenv.scoring.semantic.report import OK_STATES
    from haenv.scoring.semantic.report import PRIMARY
    out = []
    for r in rows:
        if r.get("synthetic_no_answer") or unanswered_reason(r) is not None:
            continue
        sem = r.get("semantic")
        if isinstance(sem, dict) and not all(
                sem["metric_states"].get(d, sem["status"]) in OK_STATES for d in PRIMARY):
            continue
        out.append(r)
    return out


def answered_view_scores(rows: list[dict], floor: int = 1) -> dict:
    """Production composite per model on its own answered, fully judged cells (pooled `rank_ddx`).

    `floor` lowers `analytics.MIN_GRIDS_FOR_SCORE` for this call only, so a model with
    few answered cells still gets a value; `below_production_floor` flags it.
    """
    import haenv.analytics as an
    view = answered_view(rows)
    old = an.MIN_GRIDS_FOR_SCORE
    an.MIN_GRIDS_FOR_SCORE = floor
    try:
        recs = an.rank_ddx(view)
    finally:
        an.MIN_GRIDS_FOR_SCORE = old
    n_by = {}
    for r in view:
        n_by.setdefault(r["solver"], set()).add(r["case"])
    return {rec["model"]: {"score": rec.get("score"), "n_cases": len(n_by.get(rec["model"], ())),
                           "below_production_floor": len(n_by.get(rec["model"], ())) < old,
                           "ruler_mismatch": rec.get("ruler_mismatch"),
                           "score_none_reason": rec.get("score_none_reason")}
            for rec in recs}


# ---- Failure categories: one definition for the board readout and the demo's per-reason exclusion ----
#: key -> (English label, Chinese label). Order is the display order.
FAILURE_CATEGORIES = {
    "not_started": ("Not started before the batch closed", "截止前未开始"),
    "in_progress": ("Still answering when the batch closed", "截止时仍在作答"),
    "timeout": ("Timed out", "超时"),
    "budget": ("Budget exhausted without an answer", "预算耗尽未作答"),
    "reasoning_only": ("Reasoning only, no answer", "只有推理没有作答"),
    "truncated": ("Stream truncated", "流截断"),
    "empty": ("Empty response", "空回答"),
    "unparseable": ("Unparseable response", "无法解析的回答"),
    "cost_unknown": ("Failed with unknown cost, not resent", "失败且费用未知、不重发"),
    "no_record": ("No answer recorded at batch close", "截止时无作答记录"),
    "other": ("Other failure", "其它失败"),
}

# one cell's failed attempts can carry several kinds; the most specific one names the cell
_KIND_PRIORITY = (("只有推理没有作答", "reasoning_only"), ("未收到终止帧", "truncated"), ("流被截断", "truncated"),
                  ("超时", "timeout"), ("timeout", "timeout"), ("timed out", "timeout"),
                  ("JSONDecodeError", "unparseable"), ("费用未知", "cost_unknown"))


def failure_category(reason: str | None) -> str | None:
    """The category key of an `unanswered_reason` (None for an answered cell)."""
    if reason is None:
        return None
    r = str(reason)
    if r == "not_started":
        return "not_started"
    if r == "wall_budget":
        return "timeout"
    if r == "no_row":
        return "no_record"
    if r.startswith("started_no_valid_answer:"):
        kinds = r.split(":", 1)[1]
        if kinds.startswith("no failed attempt logged"):
            return "in_progress"
        low = kinds.lower()
        for needle, cat in _KIND_PRIORITY:
            if needle.lower() in low:
                return cat
        return "other"
    if r.startswith("overall:"):
        o = r.split(":", 1)[1]
        if o.startswith("ABORT(no_answer:budget"):
            return "budget"
        if o == "ABORT(no_response)":
            return "empty"
        if o == "ABORT(unparseable)":
            return "unparseable"
        return "other"
    if r.startswith("semantic:"):
        st = r.split(":", 1)[1]
        return {"missing_response": "empty", "empty_response": "empty",
                "unparseable_response": "unparseable"}.get(st, "other")
    return "other"


def category_counts(summary: dict) -> dict:
    """{model: {category: cells}} from a `fill_unanswered` summary."""
    out = {}
    for m, e in summary["models"].items():
        c: dict[str, int] = {}
        for reason, n in e["reasons"].items():
            k = failure_category(reason)
            c[k] = c.get(k, 0) + n
        out[m] = c
    return out


# ---- Reasons for cells with no eval row, read from the batch's answer log (trace.jsonl) ----
def attempt_kind(err: str) -> str:
    """Short kind of one failed answer attempt's error text."""
    if err.startswith("UnknownCostAttemptFailure:"):
        return "费用未知/" + attempt_kind(err.split(":", 1)[1].strip())
    if "只有推理没有作答" in err:
        return "只有推理没有作答"
    if "未收到终止帧" in err:
        return "流被截断(无终止帧)"
    if err.startswith("JSONDecodeError"):
        return "JSONDecodeError"
    if err.startswith("ReconciledRequestFailure"):
        return "失败且费用未知/不重发"
    import re
    if re.search(r"timeout|timed out|超时", err, re.I):
        return "超时"
    return "其它:" + err[:40]


def scan_trace(path) -> dict:
    """One pass over trace.jsonl: per solver, the cases seen and the failed attempts (case, kind, elapsed_s)."""
    import json
    import re
    from pathlib import Path
    head = re.compile(r'"case": "([^"]*)", "solver": "([^"]*)".*?"type": "([^"]*)"')
    out: dict[str, dict] = {}
    path = Path(path)
    if not path.is_file():
        return out
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = head.search(line[:600])
            if not m:
                continue
            case, solver, typ = m.groups()
            e = out.setdefault(solver, {"seen": set(), "attempts": []})
            e["seen"].add(case)
            if typ == "assistant/attempt":
                try:
                    d = json.loads(line).get("data") or {}
                except ValueError:
                    continue
                e["attempts"].append((case, attempt_kind(str(d.get("error") or "")), d.get("elapsed_s")))
    return out


def wall_budget_stops(batch_dir) -> set[str]:
    """Models the scheduler stopped on their wall-clock budget in any run of this batch."""
    import json
    from pathlib import Path
    path = Path(batch_dir) / "scheduler.json"
    if not path.is_file():
        return set()
    return {m for run in json.loads(path.read_text(encoding="utf-8")).get("runs", [])
            for m in (run.get("wall_budget_stops") or {})}


def no_row_reasons(scan: dict, cases, models, done, wall_stopped=frozenset()) -> dict:
    """(case, model) -> reason for each cell with no eval row: not started (`wall_budget` when
    the scheduler stopped that model on its wall-clock budget), or started without a valid answer."""
    out = {}
    for m in models:
        s = scan.get(m) or {"seen": set(), "attempts": []}
        kinds: dict[str, set] = {}
        for c, k, _ in s["attempts"]:
            kinds.setdefault(c, set()).add(k)
        for c in cases:
            if (c, m) in done:
                continue
            if c not in s["seen"]:
                out[(c, m)] = "wall_budget" if m in wall_stopped else "not_started"
            elif kinds.get(c):
                out[(c, m)] = "started_no_valid_answer:" + "+".join(sorted(kinds[c]))
            else:
                out[(c, m)] = "started_no_valid_answer:no failed attempt logged (still running at close)"
    return out
