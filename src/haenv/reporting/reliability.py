"""Reliability for the evaluation report."""
from __future__ import annotations

from pathlib import Path
import json
from haenv.scoring.analysis.costs import budget_ceiling_by_solver


def reliability_by_model(rows: list[dict], job=None) -> dict:
    """Per-model retries, rescue rate and retry attribution.

    Computed from `responses.jsonl` (`evaluate.attribute_attempts`) when present,
    otherwise from the `retry_*` fields of `eval.jsonl`; the source is stated and a
    mismatch between the two is reported. Returns `{"source", "models", "mismatch",
    "ceiling"}`; `source == "unavailable"` must be printed as not measured.
    """
    from haenv.runtime.ledger import attribute_attempts as _attr
    from haenv.runtime.ledger import retry_classes as _classes

    def _graded(o) -> bool:
        o = str(o or "")
        return o == "SCORED" or o.startswith("FAIL(")

    stat: dict[str, dict] = {}

    def _slot(m):
        return stat.setdefault(str(m), {"n_cells": 0, "n_cells_retry": 0, "n_rescued": 0,
                                        "n_retry": 0, "by_class": {}})

    _ov = {(str(r.get("case")), str(r.get("solver"))): r.get("overall") for r in rows}

    rp = (Path(job.results_dir) / "responses.jsonl") if job is not None else None
    src = "unavailable"
    if rp is not None and rp.is_file():
        src = f"responses.jsonl({rp.parent.name})"
        seen: set[tuple[str, str]] = set()
        with rp.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                m = str(r.get("solver"))
                k = (str(r.get("case")), m)
                s = _slot(m)
                if k not in seen:           # a cell may have multiple lines (per-slice /
                                             # per-turn) -- count the cell only once
                    seen.add(k)
                    s["n_cells"] += 1
                cls = _attr(r.get("failed_attempts"))
                if cls:
                    s["n_retry"] += sum(cls.values())
                    for c, v in cls.items():
                        s["by_class"][c] = s["by_class"].get(c, 0) + v
                    s.setdefault("_retry_cells", set()).add(k)
        for m, s in stat.items():
            rc = s.pop("_retry_cells", set())
            s["n_cells_retry"] = len(rc)
            s["n_rescued"] = sum(1 for k in rc if _graded(_ov.get(k)))
    elif any("n_retry" in r for r in rows):
        src = "eval.jsonl's retry_* fields (this batch has no responses.jsonl)"
        for r in rows:
            if r.get("retry_status") != "measured":
                continue
            s = _slot(r.get("solver"))
            s["n_cells"] += 1
            s["n_retry"] += int(r.get("n_retry") or 0)
            if r.get("n_retry"):
                s["n_cells_retry"] += 1
            if r.get("rescue"):
                s["n_rescued"] += 1
            for c in _classes():
                v = int(r.get(f"retry_{c}") or 0)
                if v:
                    s["by_class"][c] = s["by_class"].get(c, 0) + v

    # ---- cross-check: it must be visible when the source (responses) and the
    # projection (eval fields) disagree ----
    mismatch: list[str] = []
    if src.startswith("responses.jsonl"):
        _eval_tot = sum(int(r.get("n_retry") or 0) for r in rows
                        if r.get("retry_status") == "measured")
        _resp_tot = sum(s["n_retry"] for s in stat.values())
        if any(r.get("retry_status") == "measured" for r in rows) and _eval_tot != _resp_tot:
            mismatch.append(f"eval.jsonl records {_eval_tot} retries, while responses.jsonl computes "
                            f"{_resp_tot} -- **the projection diverges from the source**; trust the source and check the attribution classifier version")
    return {"source": src, "models": stat, "mismatch": mismatch,
            "ceiling": budget_ceiling_by_solver(rows)}
