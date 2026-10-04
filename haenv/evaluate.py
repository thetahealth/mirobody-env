"""Multi-model evaluation: one cell per (case x solver), each through the same
isolation chain and hard gates:
  build_instance(<=T) -> leakage_probe -> solver.solve -> verifier.grade
Rows are appended to JSONL as cells finish; a rerun only evaluates cells not yet done.

SYNTHETIC, evaluation only, not medical advice.
"""

from __future__ import annotations
from .yamlcache import load_yaml as _cached_yaml
from . import mount_table as _MT
from .run_scheduler import stall_guard as _stall_guard
from dataclasses import dataclass
import hashlib
import json
import logging
import os
import subprocess
import threading as _threading
import time
from pathlib import Path
import haenv_kernel.verifier as verifier_mod    # kernel
import haenv_kernel.runner as runner_mod    # The leakage gate runs inside `solve_guard.guarded_solve`, shared by every geometry.
from haenv_kernel.build import build_instance
from haenv_kernel.schema import SolverOutput
from haenv_kernel.solver import Solver, BaselineSolver, RobustSolver, ALLOWED_DRIVERS, ACTION_CLASSES, _extract_json
from . import judges as judges_mod, process, wq
from .judges.trajectory import NOOP_CONTRACT as _NOOP_CONTRACT
from .prompts import PROMPT  # noqa: F401  (re-export: `evaluate.PROMPT` keeps working)
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data
from .framings import (  # noqa: F401
    DDX_PROMPT, DDX_SCOPE2_PROMPT, DDX_SCOPE_PROMPT, DDX_TRACE_PROMPT,
    _SCOPE2_EDITS, _SCOPE_EDITS, derive_scope2_prompt, derive_scope_prompt,
)
from .framings import (  # noqa: F401
    _BUILTIN_FRAMING_NAMES, _framings, framing_sha256,
)
from .call_log import (  # noqa: F401
    billed_tokens,
)
from .run_state import (  # noqa: F401
    RunContext,
    _WRITE_LOCK,
    log,
    run_ctx_of,
)
from .qside import (  # noqa: F401
    DEFAULT_PROBE,
    FALSE_PREMISE_TEMPLATE,
    NOOP_ABSENT_POOL,
    NOOP_COVERED_MIN_PTS,
    NOOP_GAP_MIN_DAYS,
    NOOP_RATIO,
    PREMISE_RATIO,
    TREND_MIN_NET_FRAC,
    TREND_WINDOW_PROBE,
    _FRAMINGS,
    _QUANT_KINDS,
    _gated_suffix,
    _noop_suffix,
    _noop_window,
    _premise_suffix,
    _quant_suffix,
    _render_prompt,
    _resync_trend_windows,
    _trend_candidate_ok,
    assign_noop_probes,
    assign_oracle_gold,
    assign_oracle_warranted,
    assign_premises,
    assign_qside_probes,
    assign_quant_probes,
    build_noop_probe,
    build_premise,
    build_quant_probe,
    load_probes,
    quant_probe_for,
    render_for,
    render_with_probe,
    resolve_probe,
)
from .relay_accounting import (
    RELAY_COST_DEFAULT,
    RELAY_COST_HEADROOM,
    RELAY_COST_PER_GRID,
    RELAY_USD_PER_MTOK,
    _measured_tokens_per_grid,
)
from .resume import RETRYABLE_OVERALL, TERMINAL_OVERALL, _done_keys, is_retryable
from .run_ledger import (
    _ATTR_MOD,
    _ceiling_row,
    _note_grid_retry,
    _note_grid_usage,
    _preflight_quota,
    attribute_attempts,
    retry_attr_sha16,
    retry_classes,
    retry_classifier,
    save_response,
    save_simple_trace,
    take_grid_retry,
    take_grid_usage,
    usage_coverage,
)
from .solvers import (
    BACKENDS,
    CLISolver,
    CLISolverBlocked,
    GoogleSolver,
    OpenAICompatSolver,
    ROUTE_ENTRY_KEYS,
    _BACKENDS_FROM_CFG,
    _BACKEND_FIELDS,
    _CLI_SANDBOX_CWD,
    _KEY_POOLS,
    _KeyPool,
    _cli_cwd,
    _const,
    _dq_hashable,
    _drain_backoff,
    _ensure_backends_registered,
    _ensure_no_proxy,
    _google_usage,
    _load_key_pool,
    _to_output,
    build_solvers,
    fallback_solver_factories,
    load_env_file,
    raw_complete,
    raw_complete_with_usage,
    register_backends,
    route_chain,
    route_name,
    secret_env_names,
    solver_for_spec,
)
from .call_log import _EXHAUSTED_MARKERS, _is_exhausted
from .solve_guard import json_unextractable
from .rows import (
    ABNORMAL_MARGIN,
    FINAL_ANSWER_JUDGES,
    GATED_ROUNDS_FILE,
    REAL_VISIT_DECILES,
    _ReplaySolver,
    _persist_gated_round,
    _premise_row,
    _prompt_sha_for,
    _re_lab,
    _real_rhythm_days,
    _replay_solver_for,
    _reusable,
    _row_gated,
    _row_multi,
    _row_single,
    _row_slices,
    answer_windows,
    budget_abort,
    gated_menu,
    gated_rounds_path,
    iron_law_precheck,
    load_gated_replay,
    load_slice_replay,
    merge_bought_tests,
    slices_for,
    take_replay_count,
    use_real_rhythm,
)
from .rhythm import REAL_RHYTHM_GAP_DAYS, REAL_RHYTHM_SLICES, rhythm_gap_feasible
from .run_scheduler import run as _schedule
from .mount_table import claim_geometry as _claim_g
from . import slicing as _slicing
from .solver_accounting import prepare_accounting
from .semantic_budget import BudgetExceeded
from .semantic_budget import cost_summary
from .mount_table import claim_geometry as _claim
from .trace import read_trace as _rt
from .trace import check_invariants as _ci
from .row_store import (  # noqa: F401
    RENAMED_ROW_KEYS,
    load_rows,
)




# Batch ERROR share at which the summary is logged as an error: at 25 % the cause is code or
# configuration, not one model.
ERROR_RATE_WARN = 0.25


_MODEL_BACKENDS: dict | None = None


def _backend_of(solver_name: str) -> str:
    """The backend this solver runs on, from `config.models`; unknown names are offline (unthrottled)."""
    global _MODEL_BACKENDS
    if _MODEL_BACKENDS is None:
        try:
            import yaml
            from pathlib import Path as _P
            _cfg = yaml.safe_load((_dr() / "config.yaml")
                                  .read_text(encoding="utf-8")) or {}
            _ov = os.environ.get("HAENV_CONFIG_OVERLAY", "").strip()   # same overlay as `cli.load_cfg`
            if _ov:
                _ov_models = (yaml.safe_load(_P(_ov).read_text(encoding="utf-8")) or {}).get("models") or {}
                for _n, _s in _ov_models.items():
                    _cfg.setdefault("models", {})[_n] = {**(_cfg.get("models", {}).get(_n) or {}), **_s}
            _MODEL_BACKENDS = {}
            for k, v in (_cfg.get("models") or {}).items():
                _MODEL_BACKENDS[str(k)] = (str(v.get("backend")) if isinstance(v, dict)
                                           else "_cli")
        except Exception as e:
            log.error("[eval] could not read config.models, per-backend throttling will "
                      "degrade to a single tier: %s", e)
            _MODEL_BACKENDS = {}
    return _MODEL_BACKENDS.get(solver_name, "_offline")


def _run_tasks(fn, tasks: list, out_path, key_of=None, geometry_of=None, *,
               workers: int = 1) -> list[dict]:
    """Run every cell; serial when `workers <= 1`, otherwise scheduled by
    `run_scheduler` (one pool per model under backend and global caps). Rows are written as
    they complete, so a killed run keeps every finished cell; the returned list is in task
    order.
    """
    def _emit(row: dict) -> dict:
        """Persist and flush one row. Also stamps `judging_sha16`, so a resumed run that spans a
        judging-code change can be detected (`report.rank_ddx`).
        """
        from .anchor import judging_fp_cached
        row.setdefault("judging_sha16", judging_fp_cached())
        row.setdefault("geometry", "<unrecorded>")
        with _WRITE_LOCK:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        return row

    return _schedule(fn, tasks, _emit, workers=max(1, int(workers)),
                     key_of=key_of, backend_of=_backend_of if key_of else None,
                     geometry_of=geometry_of, batch_dir=Path(out_path).parent)


def run_eval(job, cfg, built: dict, resume: bool = True, *,
             budget_ledger: Path | None = None, budget_usd: str | None = None,
             ctx: RunContext | None = None) -> list[dict]:
    """Evaluate `built` ({case_id: RawCase}) cell by cell, persisting each row; completed cells
    are skipped (resumable).

    `ctx` carries the Q-side probes the caller assigned (`qside.assign_*`) and the worker
    count; without one the run has no Q-side probes. The run fills in the rest and hands it to
    every row builder, which binds it to the cell's solver as `solver.run_ctx`.
    """
    ctx = ctx if ctx is not None else RunContext()
    out_path = job.results_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.resp_path = out_path.parent / "responses.jsonl"      # raw responses persist in the same batch as eval
    from .batch import provenance_fields
    _wref = provenance_fields(cfg, job_path=getattr(job, "path", None), root=job.root)
    ctx.probes = load_probes(job.root / "probes")
    ctx.allow_retired = bool(getattr(job, "allow_retired", False))
    _pid = getattr(job, "probe_id", "") or ""
    if _pid and _pid not in ctx.probes:
        raise KeyError(f"job.probe_id={_pid!r} is not registered (registered: {sorted(ctx.probes)})")
    log.info("[eval] %d probe(s) registered · this batch's framing is %s",
             len(ctx.probes), _pid or "(defaults by question type to " + str(DEFAULT_PROBE) + ")")
    _wref = {**_wref, "probe_id": _pid or None}
    # The top-level `world_sha`/`world_knobs` are the batch's sticky (case-generation) values;
    # `world_ref` holds the solve-time version, and `world_sha_at_solve` is recorded when they
    # differ. Without a sticky value, `world_sha_src` records the fallback.
    _wsha = _wknobs = None
    _wsrc = "batch_sticky"
    try:
        _bj = json.loads((out_path.parent / "batch.json").read_text(encoding="utf-8"))
        _wsha, _wknobs = _bj.get("world_sha"), _bj.get("world_knobs")
    except (OSError, json.JSONDecodeError):
        _bj = {}
    if not _wsha:
        _wsha, _wknobs = _wref.get("world_sha"), _wref.get("world_knobs")
        _wsrc = "fallback:solve_time"
    _wtop = {k: v for k, v in
             (("world_sha", _wsha), ("world_knobs", _wknobs),
              ("world_sha_src", _wsrc if _wsha else None)) if v}
    if _wsha and _wref.get("world_sha") and _wsha != _wref.get("world_sha"):
        _wtop["world_sha_at_solve"] = _wref["world_sha"]
        log.warning("[eval] generation-time world %s != solve-time world %s; the question "
                    "text comes from the former (read from cases.jsonl); both are persisted",
                    _wsha, _wref["world_sha"])
    # Slicing is a case-generation decision, frozen into the batch (see `haenv/slicing.py`),
    # so a resumed run cannot use different slices.
    _slice_map: dict[str, list[int]] = {}
    _slice_drift: list[dict] = []
    if getattr(job, "slices", ""):
        _batch_dir = out_path.parent
        _was_spec = _slicing.check_spec(_batch_dir, str(job.slices))
        if _was_spec:
            raise ValueError(
                f"slicing spec mismatch: this batch froze spec={_was_spec!r}, but the job "
                f"declares {str(job.slices)!r}. Slices cut by different specs (auto / "
                f"auto+neutral / real-rhythm / an explicit list) are not comparable; start a "
                f"new batch (--fresh) to change the spec.")
        _recovered = _slicing.recover_from_disk(_batch_dir)
        _has_rows = bool(_recovered) or (out_path.is_file() and out_path.stat().st_size > 0)
        _existing = _slicing.load(_batch_dir)
        for _cid, _raw in built.items():
            _fresh = slices_for(_raw, job.slices)
            if _existing is not None:
                _was = _slicing.drift(_batch_dir, _cid, _fresh)
                if _was is not None:
                    log.warning("[slicing] %s slice drift: %d slice(s) on disk %s vs %d "
                                "recomputed %s; using the on-disk value",
                                _cid, len(_was), _was, len(_fresh), _fresh)
                    _slice_drift.append({"case": _cid, "frozen": _was, "recomputed": _fresh})
                _slice_map[_cid] = _slicing.resolve(_batch_dir, _cid, lambda: _fresh)
                continue
            if _cid in _recovered:
                _rec = _recovered[_cid]
                if list(_rec) != list(_fresh):
                    log.warning("[slicing] %s backfilled %d slice(s) %s from disk "
                                "(recomputed gives %d %s); using the backfilled value, the "
                                "geometry this batch's rows ran under",
                                _cid, len(_rec), _rec, len(_fresh), _fresh)
                    _slice_drift.append({"case": _cid, "recovered": _rec, "recomputed": _fresh})
                _slice_map[_cid] = _rec
                continue
            # A batch with rows whose slicing cannot be recovered fails: a fresh computation could
            # change the case's geometry.
            if _has_rows:
                raise _slicing.SlicesMissing(
                    f"{_cid}: this batch already has artifacts, but its original slicing "
                    f"cannot be backfilled from eval.jsonl / responses.jsonl, while "
                    f"recomputing gives {len(_fresh)} slice(s). Substituting the recomputed "
                    f"value could change this case's denominator and geometry tier (the case "
                    f"may have had <2 slices and taken the single-shot path). Start a new "
                    f"batch (--fresh), or find out why this case has no per-slice record.")
            _slice_map[_cid] = _fresh
        _slicing.save(_batch_dir, _slice_map, spec=str(job.slices))
        if _slice_drift:
            _wref = {**_wref, "slice_drift_n": len(_slice_drift)}
            try:
                (_batch_dir / "slice_drift.json").write_text(
                    json.dumps({"n": len(_slice_drift), "cases": _slice_drift},
                               ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError as e:
                log.error("[slicing] could not write the drift ledger: %s; this batch's "
                          "drift is recorded only in the log", e)
    # The trend question must be asked on the window actually answered; re-check against the
    # settled slicing (a frozen or backfilled table can differ from a fresh computation).
    _resync_trend_windows(job, built, _slice_map, ctx=ctx)
    if not resume and out_path.exists():
        out_path.unlink()
    if not resume:
        (out_path.parent / GATED_ROUNDS_FILE).unlink(missing_ok=True)
    ctx.replay, ctx.replay_idx = bool(resume), None
    ctx.replay_n.clear()
    ctx.replay_fresh.clear()
    done = _done_keys(out_path) if resume else set()
    solvers = build_solvers(job, cfg)
    accounting = None
    if (budget_ledger is None) != (budget_usd is None):
        raise ValueError("Solver budget requires both the shared ledger and its approved cap")
    if budget_ledger is not None:
        accounting = prepare_accounting(solvers, cfg, out_path.parent,
                                        ledger_path=budget_ledger, limit_usd=budget_usd,
                                        identity=_wref)
    log.info("[eval] %d case x %d solver = %d cell(s) (%d already done)",
             len(built), len(solvers), len(built) * len(solvers), len(done))
    _preflight_quota(job, cfg, solvers, n_cases=len(built), n_done=len(done), ctx=ctx)

    _tasks: list[tuple] = []
    for cid, raw in built.items():
        T = int(raw.prediction_context["prediction_time_T"])
        for sname, make in solvers:
            key = f"{cid}|{sname}"
            if key in done:
                log.info("[eval] skip %s (done)", key); continue
            _tasks.append((cid, raw, T, sname, make, key))

    def _one(task) -> dict:
        cid, raw, T, sname, make, key = task
        solver = make()      # freshly created per case: a stateful baseline must not carry beliefs over from the previous case
        if accounting is not None:
            accounting.bind(solver, cid, sname)
        solver.probe_id = _pid
        try:
            # `_geom` is the branch actually taken (a slice count < 2 falls back to single-shot); the
            # geometry list lives in `mount_table.GEOMETRY_TABLE`.
            _n_sl = len(_slice_map[cid]) if getattr(job, "slices", "") else None
            _geom = _claim(job, _n_sl)
            if _geom == "gated":                              # tool track: on-demand query geometry
                row = _row_gated(cid, sname, raw, T, solver, ctx=ctx)
            elif _geom == "slices":                           # multi-slice query
                row = _row_slices(cid, sname, raw, _slice_map[cid], solver, ctx=ctx)
            elif _geom == "multi":
                row = _row_multi(cid, sname, raw, solver, job.cadence_days, ctx=ctx)
            else:
                row = _row_single(cid, sname, raw, T, solver, ctx=ctx)
            row.setdefault("geometry", _geom)
        except BudgetExceeded:
            raise
        except Exception as e:                # a single cell's failure must not drag down the whole round
            import traceback as _tb
            _trace = _tb.format_exc()
            log.error("[eval] %s failed: %s\n%s", key, e, _trace)
            row = {"case": cid, "solver": sname, "error": str(e),
                   "error_type": type(e).__name__, "traceback": _trace,
                   "overall": "ERROR"}
        row["world_ref"] = _wref
        row.update(_wtop)
        row.update(take_grid_usage(cid, sname, ctx=ctx))
        row.update(take_replay_count(cid, sname, ctx=ctx))
        row.update(take_grid_retry(cid, sname, overall=row.get("overall"), ctx=ctx))
        log.info("[eval] %-28s -> %s", key, row.get("overall"))
        return row

    rows = _run_tasks(_one, _tasks, out_path, key_of=lambda t: t[3],
                      geometry_of=lambda t: _claim_g(
                          job, len(_slice_map[t[0]]) if getattr(job, "slices", "") else None),
                      workers=ctx.workers)
    _rs = [r for r in rows if r.get("slices_reused")]
    if _rs:
        log.info("[eval] resume replay: %d restarted cell(s) reused %d saved answer(s) and bought "
                 "%d again (the remaining cells started from nothing or were skipped as done)",
                 len(_rs), sum(int(r["slices_reused"]) for r in _rs),
                 sum(int(r.get("replay_rebought") or 0) for r in _rs))
    if accounting is not None:
        log.info("[eval] paid solver calls (shared ledger): %s", cost_summary(accounting.ledger.snapshot()))

    # Per-cell errors are caught, so a systemic bug is reported here at batch level (not
    # blocking).
    _err = [r for r in rows if str(r.get("overall") or "") == "ERROR"]
    if _err and rows:
        _rate = len(_err) / len(rows)
        _kinds: dict[str, int] = {}
        for r in _err:
            _kinds[str(r.get("error") or "?")[:80]] = _kinds.get(str(r.get("error") or "?")[:80], 0) + 1
        (log.error if _rate >= ERROR_RATE_WARN else log.warning)(
            "[eval] this run: %d/%d cell(s) = %.1f%% judged ERROR%s · causes: %s",
            len(_err), len(rows), _rate * 100,
            ("(above the warning line: do not read this batch as a result)"
             if _rate >= ERROR_RATE_WARN else ""),
            _kinds)
    _all = load_rows(out_path)          # read the full aggregate (including rows from earlier resumed runs), deduplicated by cell
    try:
        from .batch import record_usage
        record_usage(job, _all)
    except Exception as e:                                      # noqa: BLE001
        log.error("[eval] could not write the batch usage rollup: %s; this batch's spend "
                  "is recorded only per row", e)

    # Read the trace back and validate its invariants once per batch; violations are logged,
    # not raised.
    if ctx.resp_path:
        _tp = ctx.resp_path.with_name("trace.jsonl")
        if _tp.is_file():
            try:
                # Named `_trace_bad`: the maintainers' taint check tracks the name `_bad` module-wide.
                _trace_bad = _ci(_rt(_tp))
                if _trace_bad:
                    log.error("[trace] %d trace invariant violation(s) (first 3: %s); "
                              "the trace on disk is not self-consistent, so replay and "
                              "display cannot rely on it",
                              len(_trace_bad), _trace_bad[:3])
                else:
                    log.info("[trace] trace is self-consistent ✓ (%s)", _tp.name)
            except Exception as e:                              # noqa: BLE001
                log.error("[trace] trace could not be read back: %s (a write-side or "
                          "vocabulary error, distinct from a missing trace)", e)
    return _all


