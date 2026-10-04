"""Per-cell accounting: token usage, retry attribution, billing estimates, the relay quota
preflight, and the persisted raw responses that resume reads back.

Split out of `haenv/evaluate.py`; `evaluate` re-exports every name defined here.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from . import call_log as _llm
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data
from .framings import framing_sha256
from .run_state import RunContext, _WRITE_LOCK, log
from .call_log import (  # noqa: F401
    billed_tokens,
)
from .baselines import BASELINE_NAMES as _BN
from .trace import TraceLog
from .trace import save_trace
from .relay_accounting import preflight_quota
from .resume import (  # noqa: F401
    RETRYABLE_OVERALL,
    TERMINAL_OVERALL,
    _done_keys,
    is_retryable,
)
from .relay_accounting import (  # noqa: F401
    RELAY_COST_DEFAULT,
    RELAY_COST_HEADROOM,
    RELAY_COST_PER_GRID,
    RELAY_USD_PER_MTOK,
    _measured_tokens_per_grid,
)


# Per-cell usage and retry accumulators live on the run (`RunContext.usage_acc` /
# `retry_acc`, `(case, solver) -> {...}`): filled by `save_response` for each response and
# taken by `_one`. Hanging them off persistence covers every geometry and every ABORT
# branch, including multi-request cells.


_ATTR_MOD: list = []


def retry_classifier():
    """Load the retry-attribution classifier from `tools/attribute_failures.py`.

    That file is outside the frozen manifests; attribution is a pure function of the stored
    `failed_attempts` (recomputable), and each row carries `retry_attr_sha16`.
    """
    if not _ATTR_MOD:
        import importlib.util
        p =_dr() / "tools" / "attribute_failures.py"
        spec = importlib.util.spec_from_file_location("_haenv_attr_failures", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                            # noqa: S102
        _ATTR_MOD.append(mod)
    return _ATTR_MOD[0]


def retry_attr_sha16() -> str:
    p =_dr() / "tools" / "attribute_failures.py"
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return "unknown"


def retry_classes() -> tuple[str, ...]:
    return tuple(retry_classifier().BLAME)


def attribute_attempts(failed_attempts, budget_tokens: int | None = None) -> dict[str, int]:
    """One cell's (or call's) `failed_attempts` -> `{class: count}`, via the classifier's
    `_attempts` + `classify`.

    Pass `budget_tokens`: without it `classify` cannot separate `no_answer` from `budget` and
    falls back to sentence matching (as for offline stubs).
    """
    m = retry_classifier()
    out: dict[str, int] = {}
    for e in m._attempts({"failed_attempts": failed_attempts}):  # noqa: SLF001 -- the classifier's own helper
        k = m.classify(e, budget_tokens)
        out[k] = out.get(k, 0) + 1
    return out


def _note_grid_retry(case: str, solver: str, out, *, ctx: RunContext) -> None:
    e = ctx.retry_acc.setdefault((str(case), str(solver)),
                              {"n_calls": 0, "n_retry": 0, "by_class": {}})
    e["n_calls"] += 1
    cls = attribute_attempts(getattr(out, "_failed_attempts", None),
                             getattr(out, "_max_tokens", None))
    for k, v in cls.items():
        e["by_class"][k] = e["by_class"].get(k, 0) + v
        e["n_retry"] += v


def take_grid_retry(case: str, solver: str, overall=None, *, ctx: RunContext) -> dict:
    """Take this cell's retry attribution as `eval.jsonl` fields (cleared once taken).

    Fields: `retry_status` (`measured` / `absent:stub` / `missing:no_response`), `n_retry`,
    `retry_<class>`, `retry_not_model` (infra + auth + budget), `rescue`, `retry_attr_sha16`.
    `rescue` is True when the cell had failed retries and still got a verdict (`SCORED` or
    `FAIL(gate)`), False otherwise, and None when no response was persisted.
    """
    e = ctx.retry_acc.pop((str(case), str(solver)), None)
    _is_stub = str(solver) in _BN
    if e is None or not e["n_calls"]:
        return {"retry_status": "absent:stub" if _is_stub else "missing:no_response",
                "rescue": None}
    _ov = str(overall or "")
    _graded = _ov == "SCORED" or _ov.startswith("FAIL(")
    row = {"retry_status": "measured", "n_retry": e["n_retry"],
           "retry_n_calls": e["n_calls"],
           "rescue": bool(e["n_retry"]) and _graded,
           "retry_attr_sha16": retry_attr_sha16()}
    _not_model = 0
    for k in retry_classes():
        v = int(e["by_class"].get(k, 0))
        row[f"retry_{k}"] = v
        if k in ("infra", "auth", "budget"):
            _not_model += v
    row["retry_not_model"] = _not_model
    return row


def _note_grid_usage(case: str, solver: str, out, *, ctx: RunContext) -> None:
    b = billed_tokens(getattr(out, "_usage", None))
    k = (str(case), str(solver))
    e = ctx.usage_acc.setdefault(k, {"n_calls": 0, "n_with_usage": 0,
                                  "in_fresh": 0, "in_cached": 0, "out": 0,
                                  "out_thoughts": 0, "unaccounted": 0,
                                  "latency_s": 0.0, "n_latency": 0,
                                  # Each ceiling boolean has its own denominator: `ceil_by_finish` needs only
                                  # `finish_reason`, `ceil_by_ratio` also needs usage.
                                  "n_fin_known": 0, "n_ceil_by_finish": 0,
                                  "n_ratio_known": 0, "n_ceil_by_ratio": 0,
                                  "n_at_risk": 0, "n_degenerate": 0,
                                  "retry_in": 0, "retry_out": 0,
                                  "retry_n": 0, "retry_n_with_usage": 0,
                                  "budgets": set()})
    # Count the gated track's earlier query rounds too (`out` is only the final round, which is
    # counted below).
    _gr = [x for x in (getattr(out, "_gated_rounds", None) or []) if isinstance(x, dict)]
    if _gr:
        _last = max(int(x.get("round") or 0) for x in _gr)
        for _x in _gr:
            if int(_x.get("round") or 0) >= _last:
                continue                      # last round = `out`, recorded by the block below
            e["n_calls"] += 1
            _xb = billed_tokens(_x.get("usage"))
            if _xb.get("status") == "measured":
                e["n_with_usage"] += 1
                for f in ("in_fresh", "in_cached", "out"):
                    e[f] += int(_xb.get(f) or 0)
                e["out_thoughts"] += int(_xb.get("out_thoughts") or 0)
                e["unaccounted"] += int(_xb.get("unaccounted") or 0)
            _xl = _x.get("latency_s")
            if isinstance(_xl, (int, float)):
                e["latency_s"] += float(_xl)
                e["n_latency"] += 1
    e["n_calls"] += 1
    for _a in (getattr(out, "_failed_attempts", None) or []):
        e["retry_n"] += 1
        _ab = _a.get("billed") if isinstance(_a, dict) else None
        if isinstance(_ab, dict) and _ab.get("status") == "measured":
            e["retry_n_with_usage"] += 1
            e["retry_in"] += int(_ab.get("in_fresh") or 0) + int(_ab.get("in_cached") or 0)
            e["retry_out"] += int(_ab.get("out") or 0)
    if b.get("status") == "measured":
        e["n_with_usage"] += 1
        for f in ("in_fresh", "in_cached", "out"):
            e[f] += int(b.get(f) or 0)
        e["out_thoughts"] += int(b.get("out_thoughts") or 0)
        e["unaccounted"] += int(b.get("unaccounted") or 0)
    _lat = getattr(out, "_latency_s", None)
    if isinstance(_lat, (int, float)):
        e["latency_s"] += float(_lat)
        e["n_latency"] += 1
    # ---- ceiling hits (`llm.ceiling_flags`) ----
    # Reasoning and answer share `max_tokens`; these flags separate a budget-limited answer from
    # a capability floor.
    _mt = getattr(out, "_max_tokens", None)
    if _mt:                       # stubs/offline lack this attribute => not recorded, not recorded as 0
        _cf = _llm.ceiling_flags(
            (b.get("out") if b.get("status") == "measured" else None),
            getattr(out, "_finish", None), int(_mt),
            str(getattr(out, "_raw_text", "") or ""))
        e["budgets"].add(int(_mt))
        if getattr(out, "_finish", None) is not None:
            e["n_fin_known"] += 1
            e["n_ceil_by_finish"] += int(bool(_cf["ceil_by_finish"]))
        if _cf["ceil_by_ratio"] is not None:
            e["n_ratio_known"] += 1
            e["n_ceil_by_ratio"] += int(bool(_cf["ceil_by_ratio"]))
            e["n_at_risk"] += int(bool(_cf["at_risk"]))
        e["n_degenerate"] += int(bool(_cf["degenerate"]))


def _ceiling_row(e: dict) -> dict:
    """This cell's ceiling readings for `eval.jsonl`: `absent:no_budget` when no call carried a
    budget (stub/offline), else `measured`. A boolean whose denominator (`n_*_known`) is 0 is
    `None`, never `False`.
    """
    if not e.get("budgets"):
        return {"ceiling": {"status": "absent:no_budget"}}
    _b = sorted(e["budgets"])
    return {"ceiling": {
        "status": "measured",
        "budget": (_b[0] if len(_b) == 1 else _b),
        "by_finish": (None if not e["n_fin_known"] else bool(e["n_ceil_by_finish"])),
        "by_ratio": (None if not e["n_ratio_known"] else bool(e["n_ceil_by_ratio"])),
        "at_risk": (None if not e["n_ratio_known"] else bool(e["n_at_risk"])),
        "degenerate": bool(e["n_degenerate"]),
        "n_calls": e["n_calls"],
        "n_fin_known": e["n_fin_known"], "n_ceil_by_finish": e["n_ceil_by_finish"],
        "n_ratio_known": e["n_ratio_known"], "n_ceil_by_ratio": e["n_ceil_by_ratio"]}}


def take_grid_usage(case: str, solver: str, *, ctx: RunContext) -> dict:
    """Take this cell's usage as `eval.jsonl` fields (cleared once taken).

    Writes `billed.in_total` / `billed.out` / `latency_s`, the keys
    `report.cost_efficiency_analysis` reads. `usage_status` is `measured`, `absent:stub`
    (offline stub), `missing:no_response` (real model, no response persisted) or
    `missing:no_usage` (the backend returned none); stubs are identified by
    `baselines.BASELINE_NAMES`.
    """
    e = ctx.usage_acc.pop((str(case), str(solver)), None)
    _is_stub = str(solver) in _BN
    if e is None or not e["n_calls"]:
        return {"usage_status": "absent:stub" if _is_stub else "missing:no_response"}
    if not e["n_with_usage"]:
        return {"usage_status": "absent:stub" if _is_stub else "missing:no_usage",
                "usage_n_calls": e["n_calls"], **_ceiling_row(e)}
    row = {"usage_status": "measured", **_ceiling_row(e),
           "billed": {"in_fresh": e["in_fresh"], "in_cached": e["in_cached"],
                      "in_total": e["in_fresh"] + e["in_cached"], "out": e["out"],
                      "out_thoughts": e["out_thoughts"] or None,
                      "unaccounted": e["unaccounted"] or None,
                      "status": "measured"},
           "usage_n_calls": e["n_calls"]}
    if e["n_with_usage"] < e["n_calls"]:
        row["usage_n_calls_without"] = e["n_calls"] - e["n_with_usage"]
    if e["retry_n"]:
        row["billed_retry"] = {"in_total": e["retry_in"], "out": e["retry_out"],
                               "n_attempts": e["retry_n"],
                               "n_with_usage": e["retry_n_with_usage"],
                               "status": ("measured" if e["retry_n_with_usage"] == e["retry_n"]
                                          else "lower_bound")}
        if e["retry_n_with_usage"] < e["retry_n"]:
            row["billed"]["status"] = "lower_bound"
            row["billed"]["lower_bound_why"] = (
                f"{e['retry_n'] - e['retry_n_with_usage']}/{e['retry_n']} failed attempt(s) "
                f"did not return usage")
    if e["n_latency"]:
        row["latency_s"] = round(e["latency_s"], 2)
    return row


def save_simple_trace(path: Path, case: str, solver: str, geometry: str, out,
                      slice_t=None, step: int = 1) -> None:
    """Trace for the non-gated geometries: one model request per step, no tool calls.

    It is where the raw reasoning text is kept (`save_response` never persists it). The gated
    geometry writes its own trace in `gated.run_gated`.
    """
    txt = getattr(out, "_raw_text", None)
    if txt is None:
        return
    log = TraceLog(case=case, solver=solver, geometry=geometry)
    log.append("case/start", {"slice_t": slice_t})
    log.append("step/start", {}, step=step)
    log.append("request/header", {"prompt_mode": getattr(out, "_prompt_mode", "default"),
                                  "probe_id": getattr(out, "_probe_id", None),
                                  "prompt_sha256": getattr(out, "_prompt_sha", None),
                                  "slice_t": slice_t}, step=step)
    log.append("assistant/message", {
        "raw": txt, "n_chars": len(txt),
        "usage": getattr(out, "_usage", None),
        "finish_reason": getattr(out, "_finish", None),
        "max_tokens": getattr(out, "_max_tokens", None),
        "latency_s": getattr(out, "_latency_s", None),
        "reasoning_text": getattr(out, "_reasoning_text", None),
    }, step=step)
    for _fa in (getattr(out, "_failed_attempts", None) or []):
        log.append("assistant/attempt", dict(_fa), step=step)
    log.append("step/end", {"reason": "answered"}, step=step)
    log.append("case/end", {"slice_t": slice_t})
    save_trace(path, log)


def save_response(path: Path, case: str, solver: str, out, slice_t=None,
                  rounds=None, *, ctx: RunContext) -> None:
    """Append the tested model's raw response to `responses.jsonl`, so a scoring change is a
    recompute, not a rerun. The prompt itself is not stored; `prompt_mode` and the framing
    fingerprint identify it.
    """
    txt = getattr(out, "_raw_text", None)
    if txt is None:
        return
    pid = getattr(out, "_probe_id", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    _note_grid_usage(case, solver, out, ctx=ctx)
    _note_grid_retry(case, solver, out, ctx=ctx)
    _billed, _extra = billed_tokens(getattr(out, "_usage", None)), {}
    if rounds and any(isinstance(x, dict) and "round" in x for x in rounds):
        # Multi-round (gated) row: `usage` / `finish_reason` are the last turn's, `billed` is the
        # cell total over every round (the same rule as `_note_grid_usage`: the last round is `out`).
        _rs = [x for x in rounds if isinstance(x, dict)]
        _last = max(int(x.get("round") or 0) for x in _rs)
        _tot, _n, _n_meas = dict(_billed), 1, int(_billed["status"] == "measured")
        _tot = {k: (int(v or 0) if k in ("in_fresh", "in_cached", "out", "out_thoughts", "unaccounted") else v)
                for k, v in _tot.items()}
        for _x in _rs:
            if int(_x.get("round") or 0) >= _last:
                continue
            _xb = billed_tokens(_x.get("usage"))
            _n += 1
            if _xb["status"] == "measured":
                _n_meas += 1
                for f in ("in_fresh", "in_cached", "out", "out_thoughts", "unaccounted"):
                    _tot[f] += int(_xb.get(f) or 0)
        _tot["in_total"] = _tot["in_fresh"] + _tot["in_cached"]
        _tot["out_thoughts"] = _tot["out_thoughts"] or None
        _tot["unaccounted"] = _tot["unaccounted"] or None
        _tot["n_calls"] = _n
        _tot["status"] = ("absent" if not _n_meas else
                          "measured" if _n_meas == _n else "lower_bound")
        _extra = {"billed_last_turn": _billed, "usage_scope": "last_turn"}
        _billed = _tot
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"case": case, "solver": solver, "slice_t": slice_t,
                                "prompt_mode": getattr(out, "_prompt_mode", "default"),
                                "probe_id": pid,
                                "framing_sha256": (framing_sha256(
                                    (ctx.probes or {}).get(pid, {}).get("framing_ref", ""))
                                    if ctx.probes and pid in (ctx.probes or {}) else None),
                                "prompt_sha256": getattr(out, "_prompt_sha", None),
                                # Full sha256 of the prompt sent: equals the receipts'
                                # `request.prompt_sha256` (`prompt_sha256` above is its
                                # 16-character prefix, the resume key).
                                "request_prompt_sha256": getattr(out, "_prompt_sha_full", None),
                                "requests": getattr(out, "_requests", None),
                                "usage": getattr(out, "_usage", None),
                                "finish_reason": getattr(out, "_finish", None),
                                "failed_attempts": getattr(out, "_failed_attempts", None) or None,
                **({"rate_limit_retries": out._rate_limit_retries}
                   if getattr(out, "_rate_limit_retries", None) else {}),
                                "latency_s": getattr(out, "_latency_s", None),
                                "reasoning_chars": getattr(out, "_reasoning_chars", None),
                                "billed": _billed, **_extra,
                                "rounds": rounds or None,
                                "n_chars": len(txt), "raw": txt}, ensure_ascii=False) + "\n")


def usage_coverage(rows: list[dict]) -> dict:
    """Usage coverage for a batch of `eval.jsonl` rows, per solver and in total.

    Returns `{status, n_rows, in_total, out, total_tokens, by_solver, real_models_missing}`.
    `status=absent` is normal for an offline batch; a non-empty `real_models_missing` (a real
    model with no measured cell) means the accounting broke.
    """
    by: dict[str, dict] = {}
    for r in rows:
        s = str(r.get("solver") or r.get("model") or "")
        if not s:
            continue
        e = by.setdefault(s, {"n_rows": 0, "n_measured": 0, "in_total": 0, "out": 0,
                              "missing": {}, "is_stub": s in _BN})
        e["n_rows"] += 1
        st = str(r.get("usage_status") or "")
        if st == "measured":
            b = r.get("billed") or {}
            e["n_measured"] += 1
            e["in_total"] += int(b.get("in_total") or 0)
            e["out"] += int(b.get("out") or 0)
        else:
            k = st or "unrecorded"
            e["missing"][k] = e["missing"].get(k, 0) + 1
    for e in by.values():
        e["status"] = "measured" if e["n_measured"] else "absent"
    _in = sum(e["in_total"] for e in by.values())
    _out = sum(e["out"] for e in by.values())
    _bad = sorted(s for s, e in by.items() if not e["is_stub"] and not e["n_measured"])
    return {"status": "measured" if any(e["n_measured"] for e in by.values()) else "absent",
            "n_rows": sum(e["n_rows"] for e in by.values()),
            "in_total": _in, "out": _out, "total_tokens": _in + _out,
            "by_solver": {s: {k: v for k, v in e.items() if k != "is_stub"}
                          for s, e in sorted(by.items())},
            "real_models_missing": _bad}


def _preflight_quota(job, cfg, solvers, n_cases: int, n_done: int, *, ctx: RunContext) -> None:
    """Relay balance gate; the logic is `relay_accounting.preflight_quota` (accounting code)."""
    preflight_quota(job, cfg, solvers, n_cases, n_done, resp_path=ctx.resp_path)
