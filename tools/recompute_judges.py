"""recompute_judges.py -- after a judge changes, recompute a batch from stored raw responses instead of re-running the model.

Usage:
    uv run python tools/recompute_judges.py <batch>                 # recompute only the judges that don't need gold
    uv run python tools/recompute_judges.py <batch> --full           # rebuild vp, recompute all haenv judges
    uv run python tools/recompute_judges.py <batch> --full --tracks  # also recompute the kernel's five tracks

Reads the batch's `responses.jsonl` and `cases.jsonl`, reruns the judges with the
current code, and merges new fields into `eval.jsonl` (adding fields is free;
changing an existing value needs `--allow-change`). Rows without a recorded
response (offline baselines) keep their values, so new fields are `None` there.
Recomputing an unchanged batch with `--tracks` reproduces every cell.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from haenv import kernel_path as _kernel_path       # noqa: E402
if (_kp := _kernel_path()):
    sys.path.insert(0, str(_kp))


class _Out:
    """Wrap raw JSON from disk in the shape the judges expect."""

    def __init__(self, raw: dict):
        self._raw = raw or {}
        self.forecast = self._raw.get("forecast") or {}
        self.drivers = self._raw.get("drivers") or []
        self.action = self._raw.get("action") or {}
        self.data_quality = self._raw.get("data_quality") or {}
        self.cited_evidence = self._raw.get("cited_evidence") or []


def _to_output_full(raw: dict, sp):
    """Assemble raw JSON from disk into a SolverOutput via `evaluate._to_output`.

    Unlike `_Out`, it fills the default abstention action on parse failure, which
    track D reads.
    """
    from haenv.evaluate import _to_output
    return _to_output(raw, sp)


# --------------------------------------------------------------------------
# Per-slice raw-text facts (`raw_empty` / `raw_unparseable`), recomputed from the
# raw text in `responses.jsonl` with `evaluate.json_unextractable`. They are
# captured before unparseable or empty slices are skipped. When a slice's raw
# text is not on disk the fact is recorded as unavailable, never as `false`.
# --------------------------------------------------------------------------

#: Canned fields cleared when a slice's JSON can't be extracted (the kernel turns
#: a parse failure into a canned abstention). A copy of the local literal in
#: `evaluate._row_slices`.
SLICE_CANNED_FIELDS = ("answer_text", "differential", "join_type", "action",
                       "data_sufficiency", "risk_category", "all_drivers")


class GeometryUndecidable(RuntimeError):
    """This row's geometry can't be determined."""


def geometry_of_row(row: dict) -> str:
    """Which geometry this row actually took: `geometry` if present, else inferred from its structure.

        slice_rows non-empty            -> slices
        rounds / trajectory non-empty   -> multi
        tool_budget / tool_rounds       -> gated
        none of the above               -> single

    The job's declared geometry is not used: a case with fewer than 2 slices runs
    as single-shot.
    """
    g = row.get("geometry")
    if g in ("single", "slices", "gated", "multi"):
        return g
    if isinstance(row.get("slice_rows"), list) and row["slice_rows"]:
        return "slices"
    if row.get("rounds") or row.get("trajectory"):
        return "multi"
    if row.get("tool_budget") is not None or row.get("tool_rounds") is not None:
        return "gated"
    if g is not None:  # unrecognized value => don't guess
        raise GeometryUndecidable(f"unrecognized `geometry` value: {g!r}")
    return "single"


def subjects_for_recompute(row: dict, out, by_slices: dict, build_last) -> tuple[str, dict, str | None]:
    """Return `(geometry, subjects, gap note)` for recomputing this row.

    A subject that cannot be provided is left out (and recorded by name by
    `run_judges`), never substituted. For slices, `rows` is always available and
    `last` only when the last slice's raw text is on disk.
    """
    geom = geometry_of_row(row)
    if geom == "single":
        return geom, {"out": out}, None
    if geom == "gated":
        return geom, {"out": out}, None
    if geom == "multi":
        # multi-round: per-round raw text isn't on disk, so `last` is missing
        return geom, {"out": out}, "multi:逐轮原文未落盘,末轮 subject 缺席"
    # ---- slices ----
    rows = row.get("slice_rows")
    if not (isinstance(rows, list) and rows):
        raise GeometryUndecidable("geometry is slices but `slice_rows` is empty")
    subj: dict = {"rows": rows}
    last_t = rows[-1].get("t")
    # Index by `slice_t`: `by_key` holds the last entry in file order, not the largest t.
    per = by_slices.get((row.get("case"), row.get("solver"))) or {}
    last_raw = per.get(str(last_t))
    if last_raw is not None:
        subj["last"] = build_last(last_raw)
        return geom, subj, None
    return geom, subj, f"slices:末片(t={last_t})原文不在盘上,`last` 那 8 条判据不可重算"


_MENUS: dict = {}
_TRACE_TARGETS: dict = {}


def _traced_targets(trace_p: Path, case, solver) -> list | None:
    """Non-null tool-call targets the trace records for this cell; `None` without a trace."""
    if trace_p not in _TRACE_TARGETS:
        by: dict = {}
        if trace_p.is_file():
            for line in trace_p.read_text(encoding="utf-8").split("\n"):
                if '"tool/call"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "tool/call":
                    t = (e.get("data") or {}).get("target")
                    by.setdefault((e.get("case"), e.get("solver")), []).extend([t] if t else [])
        _TRACE_TARGETS[trace_p] = by if trace_p.is_file() else None
    by = _TRACE_TARGETS[trace_p]
    return None if by is None else by.get((case, solver))


def merge_gated_purchases(row: dict, raw: dict, case, trace_p: Path) -> tuple[dict, str | None]:
    """Merge a gated row's purchased tests into a copy of `raw`, as `_row_gated` does live.

    The menu is rebuilt from the case (`evaluate.gated_menu`); purchases are the row's
    `tool_targets`. Returns `(answer, None)`, or `(raw, reason)` when the purchases cannot be
    recovered: no `tool_n_calls`; calls recorded but no targets and the trace does not show
    they had none; or the rebuilt menu merges a different count than live `tests_from_queries`.
    """
    from haenv.evaluate import gated_menu, merge_bought_tests
    targets = row.get("tool_targets")
    if not targets:
        n = row.get("tool_n_calls")
        if n is None:
            return raw, "gated:行上无 `tool_n_calls`,买过什么没有记录"
        if n:
            traced = _traced_targets(trace_p, row.get("case"), row.get("solver"))
            if traced is None or traced:
                return raw, (f"gated:`tool_n_calls`={n} 而 `tool_targets` 为空,"
                             f"trace {'不在盘上' if traced is None else f'记着 {len(traced)} 个目标'}")
        return raw, None
    if case is None:
        return raw, "gated:cases.jsonl 里没有这个病例,菜单无法重建"
    T = int(row["T"]) if row.get("T") is not None else int(case.prediction_context["prediction_time_T"])
    key = (row.get("case"), T)
    try:
        if key not in _MENUS:
            _MENUS[key] = gated_menu(case, T)
    except Exception as e:                                # noqa: BLE001
        return raw, f"gated:菜单重建失败 {type(e).__name__}: {str(e)[:100]}"
    answer = dict(raw)
    q = merge_bought_tests(answer, targets, _MENUS[key])
    live_n = row.get("tests_from_queries")
    if live_n is not None and len(q) != int(live_n):
        return raw, f"gated:重建菜单并入 {len(q)} 项,现场并入 {live_n} 项"
    return answer, None


#: `overall` values whose verdict the recompute re-derives from the row's own fields when no
#: response is stored. Other `ABORT(...)` kinds rest on facts the row records only as the verdict
#: itself (raw text, leak probe, iron-law check) and are left as they are.
REDERIVABLE_ABORTS = ("ABORT(no_answer:budget)",)


def rederive_abort(row: dict) -> tuple[dict | None, str | None]:
    """Re-derive a response-less ABORT row's verdict with the live rule.

    Returns `({"restamp_basis": ...}, None)` when the recorded fields give the recorded verdict,
    `(None, reason)` when they contradict it, and `(None, None)` for a kind not re-derived.
    `ABORT(no_answer:budget)`: no committed answer (live saves no response for this cell),
    `evaluate.budget_abort(False, tool_spent, tool_budget)` and the recorded reason.
    """
    o = str(row.get("overall") or "")
    if o not in REDERIVABLE_ABORTS:
        return None, None
    from haenv.evaluate import budget_abort
    spent, budget = row.get("tool_spent"), row.get("tool_budget")
    if spent is None or budget is None:
        return None, f"{o}:行上缺 tool_spent/tool_budget"
    if row.get("no_response_reason") != "budget_exhausted_before_answer":
        return None, f"{o}:no_response_reason={row.get('no_response_reason')!r}"
    if not budget_abort(False, float(spent), float(budget)):
        return None, f"{o}:tool_spent={spent} < tool_budget={budget}"
    return {"restamp_basis": "abort_rederived_from_row"}, None


def slice_raw_facts(txt) -> dict | None:
    """Raw-text facts for one slice; `None` when the raw text is not on disk."""
    if not isinstance(txt, str):
        return None
    empty = not txt.strip()
    from haenv.evaluate import json_unextractable
    return {"raw_empty": empty,
            "raw_unparseable": (False if empty else bool(json_unextractable(txt)))}


def apply_slice_raw_facts(slice_rows: list, facts_by_t: dict, stats: dict) -> dict:
    """Write per-slice raw-text facts into `slice_rows` and return the fields to merge into `add`.

    Unparseable slices keep their place in the denominator but have their canned
    fields cleared; if every slice is unusable the row becomes `ABORT` (through
    `add`, so it needs `--allow-change`, which `restamp_batch` passes).
    """
    n_unp = n_emp = n_na = 0
    known: list[dict] = []
    for sr in slice_rows:
        f = facts_by_t.get(str(sr.get("t")))
        if f is None:
            # Missing raw text: record `None` and keep any live value.
            sr.setdefault("raw_empty", None)
            sr.setdefault("raw_unparseable", None)
            sr["raw_facts_recompute"] = "unavailable:该片原文不在 responses.jsonl 里"
            n_na += 1
            continue
        known.append(f)
        sr["raw_empty"] = f["raw_empty"]
        sr["raw_unparseable"] = f["raw_unparseable"]
        if f["raw_unparseable"]:
            n_unp += 1
            for k in SLICE_CANNED_FIELDS:
                sr[k] = None
            sr["slice_unparseable"] = True
        elif f["raw_empty"]:
            n_emp += 1
    stats["slices_seen"] = stats.get("slices_seen", 0) + len(slice_rows)
    stats["unparseable"] = stats.get("unparseable", 0) + n_unp
    stats["empty"] = stats.get("empty", 0) + n_emp
    stats["unavailable"] = stats.get("unavailable", 0) + n_na

    add: dict = {}
    if n_unp or n_emp:
        add["slices_unparseable_n"] = n_unp
        add["slices_raw_empty_n"] = n_emp
    if n_na:
        add["slices_raw_facts_unavailable_n"] = n_na
    # Whole-row ABORT only when every slice was measured.
    if known and len(known) == len(slice_rows) and all(
            f["raw_empty"] or f["raw_unparseable"] for f in known):
        add["overall"] = ("ABORT(no_response)" if all(f["raw_empty"] for f in known)
                          else "ABORT(unparseable)")
        add["no_response_reason"] = ("all_slices_empty" if all(f["raw_empty"] for f in known)
                                     else "all_slices_unparseable")
        stats["rows_aborted"] = stats.get("rows_aborted", 0) + 1
    return add


def enrich_slice_rows(row: dict, by_slices: dict) -> int:
    """Backfill two fields from recorded per-slice responses into `slice_rows`; returns the number of slices backfilled.

    Must run after `apply_slice_raw_facts` and before `run_judges`: two safety gates
    read `clinician_review_required`.
    """
    per = by_slices.get((row.get("case"), row.get("solver"))) or {}
    rows = row.get("slice_rows")
    if not (per and isinstance(rows, list)):
        return 0
    n = 0
    for sr in rows:
        raw_s = per.get(str(sr.get("t")))
        if not isinstance(raw_s, dict):
            continue
        n += 1
        dq = raw_s.get("data_quality")
        sr["data_sufficiency"] = (dq or {}).get("data_sufficiency") if isinstance(dq, dict) else None
        # Present in the recorded response; missing stays `None`.
        a_s = raw_s.get("action")
        sr["clinician_review_required"] = (a_s.get("clinician_review_required")
                                           if isinstance(a_s, dict) else None)
    return n


def rejudge_qside(row: dict, out, case, noop_probe: dict | None, quant_probe: dict | None,
                  t_max: int | None) -> dict:
    """Judge one row's noop and quant answers on the questions that row was asked.

    noop: the probe is rebuilt from the row's recorded fields (target, window, truth) under
    the row's `noop_contract`; a row without one was asked before the enum and gets the
    marker rule. quant: the row's recorded `(kind, signal)`; when the probe rebuilt now picks
    another series, the recorded one is rebuilt from the case (truth recomputed on the
    answered window by the judge). A row that records no question gets the rebuilt probe.
    `qside_probe_stale` only when the recorded question cannot be rebuilt.
    """
    from haenv.evaluate import quant_probe_for
    from haenv.judges import judge_noop_probe, judge_quant_probe
    add: dict = {}
    stale: list[str] = []
    _t = row.get("noop_target")
    if _t is not None and row.get("noop_truth_present") is not None:
        _np = {"target": _t, "polarity": row.get("noop_polarity"),
               "truth_present": bool(row.get("noop_truth_present")),
               "window": row.get("noop_window"), "n_pts_in_window": row.get("noop_n_pts_in_window"),
               "contract": row.get("noop_contract")}
    elif noop_probe and (_t is None or str(_t) == str(noop_probe.get("target"))):
        _np = {**noop_probe, "contract": row.get("noop_contract")}
    else:
        _np = None
        if noop_probe:
            stale.append(f"noop:落盘 target={_t} vs 现在 {noop_probe.get('target')},"
                         f"且行上缺窗口/真值 —— 题面变过,只能重跑")
    if _np:
        add.update(judge_noop_probe(out, _np) or {})
    _k, _s = row.get("quant_kind"), row.get("quant_signal")
    if _k is None:
        _qp = quant_probe
    elif quant_probe and (str(_k), str(_s or "")) == (str(quant_probe.get("kind")),
                                                      str(quant_probe.get("signal"))):
        _qp = quant_probe
    else:
        _qp = quant_probe_for(case, str(_k), str(_s or "")) if case is not None else None
        if _qp is None:
            stale.append(f"quant:落盘 {_k}/{_s} 在病例里不足 4 个点"
                         f"(现在 {(quant_probe or {}).get('kind')}/{(quant_probe or {}).get('signal')})"
                         f" —— 题面变过,只能重跑")
    if _qp:
        add.update(judge_quant_probe(out, _qp, t_max=t_max) or {})
    if stale:
        add["qside_probe_stale"] = " | ".join(stale)
    elif row.get("qside_probe_stale"):
        # judged now on the recorded question: an earlier stale mark no longer holds
        add["qside_probe_stale"] = None
    return add


def _answer_t_from_rows(eval_p: Path) -> dict[str, int]:
    """Last answered day per case, as recorded on the rows: `quant_t_max`, else the last
    slice, else `T`. Read from disk rather than recomputed from the current slicing code.
    """
    out: dict[str, int] = {}
    for line in eval_p.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        cid = str(r.get("case") or "")
        if not cid or cid in out:
            continue
        t = r.get("quant_t_max")
        if t is None and r.get("slices"):
            t = max(int(x) for x in r["slices"])
        if t is None and r.get("T") is not None:
            t = r["T"]
        if t is not None:
            out[cid] = int(t)
    return out


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    d = Path(argv[0])
    resp_p, eval_p = d / "responses.jsonl", d / "eval.jsonl"
    if not resp_p.is_file() or not eval_p.is_file():
        print(f"[recompute] {d} is missing responses.jsonl or eval.jsonl")
        return 2

    full = "--full" in argv
    from haenv.judges import join_self_contradiction

    # Without `--full`, only judges that don't need vp (gold) are recomputed;
    # `--full` rebuilds vp and recomputes every SINGLE judge.
    RECOMPUTE = {"join_selfcheck": join_self_contradiction}
    vps: dict = {}
    sps: dict = {}
    if full:
        from haenv.store import load_cases
        from build import build_instance                      # kernel
        cases_p = d / "cases.jsonl"
        if not cases_p.is_file():
            print(f"[recompute] --full needs {cases_p}"); return 2
        # Prefer the payload stored with the batch: rebuilding uses the current
        # `build.py`, which may differ from the one that generated the batch
        # (`payloads.payload_source` reports which was used).
        from haenv.payloads import FILENAME as _PL_NAME
        from haenv.payloads import load_payloads as _load_payloads
        from haenv.payloads import payload_source as _payload_source
        _built: dict = {}
        _pl_src, _pl_p = _payload_source(d)
        if _pl_src == "disk":
            for cid, (_sp, _vp) in _load_payloads(_pl_p).items():
                vps[cid], sps[cid] = _vp, _sp
        for cid, rawc in load_cases(cases_p).items():
            _built[cid] = rawc
            if cid in vps:
                continue
            T = int(rawc.prediction_context["prediction_time_T"])
            _sp, _vp = build_instance(rawc, T)
            vps[cid], sps[cid] = _vp, _sp
        if _pl_src == "disk":
            print(f"[recompute] --full: **read from disk**: {len(vps)} case payloads "
                  f"({_PL_NAME}) -- unaffected by drift in the generation code")
        else:
            print(f"[recompute] --full: this batch has no {_PL_NAME}; "
                  f"rebuilt {len(vps)} cases with the current build.py -- "
                  f"if the generation code changed since this batch, this pair need not be "
                  f"the pair that was scored at the time")
        # ---- Q-side probes (`noop_*` / `quant_*`) are rebuilt explicitly with the same call
        # the run makes (`assign_qside_probes`: quant avoids the noop stream, trend on the
        # answer window each row records); assignment is deterministic by case_id ----
        try:
            from haenv.evaluate import assign_qside_probes
            _npr, _qpr = assign_qside_probes(_built, answer_t=_answer_t_from_rows(eval_p))
            print(f"[recompute] Q-side probes rebuilt: noop {len(_npr)} cases · quant {len(_qpr)} cases")
        except Exception as e:                            # noqa: BLE001
            print(f"[recompute] rebuilding the Q-side probes failed ({type(e).__name__}); "
                  f"those two field families are left as they are")

    by_key: dict[tuple, dict] = {}
    # `by_slices` keeps every slice (`by_key` keeps only the last).
    by_slices: dict[tuple, dict] = {}
    # Raw-text facts are captured before the two `continue`s below.
    by_slice_facts: dict[tuple, dict] = {}
    n_empty = [0]
    for line in resp_p.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        raw = r.get("raw")
        if r.get("slice_t") is not None:
            by_slice_facts.setdefault((r.get("case"), r.get("solver")), {})[
                str(r.get("slice_t"))] = slice_raw_facts(r.get("raw"))
        if isinstance(raw, str):
            # Same parser as the pipeline (handles ```json fences).
            from solver import _extract_json          # kernel
            try:
                raw = _extract_json(raw)
            except Exception as e:
                log_line = f"{r.get('case')}|{r.get('solver')}: {type(e).__name__}"
                print(f"[recompute] unparseable, skipped {log_line}")
                continue
        # Empty responses are skipped, not recomputed as an empty answer, which would
        # change fields like `urgency_said`.
        if not raw:
            n_empty[0] += 1
            continue
        by_key[(r.get("case"), r.get("solver"))] = raw
        _st = r.get("slice_t")
        if _st is not None:
            by_slices.setdefault((r.get("case"), r.get("solver")), {})[str(_st)] = raw

    rows, n_patched, new_fields, changed = [], 0, set(), {}
    # Case-level fields: computed from gold, identical on every row of the case
    # (including offline rows).
    CASE_LEVEL_FIELDS = ("gold_kind", "rival_n_declared", "urgency_gold",
                         "tests_total", "dx_applicable", "dx_threads_total")
    # Adding fields is free; changing an existing value needs `--allow-change
    # field[,field...]`. Conflicts are dropped and listed in `recompute_conflicts`
    # (some stubs' recorded responses cannot reproduce their rows exactly).
    _allow_change = set()
    if "--allow-change" in argv:
        _allow_change = {x.strip() for x in argv[argv.index("--allow-change") + 1].split(",")
                         if x.strip()}
    if "--allow-change-all" in argv:
        _allow_change = None                        # None = allow every change
    _conflicts: dict[str, int] = {}
    _rows_guarded = 0
    from haenv.baselines import BASELINE_NAMES as _BN
    _STUBS = set(_BN)
    _stub_skipped = 0
    _case_level: dict[str, dict] = {}
    propagated: dict[str, int] = {}
    #: Exercise counts for the per-slice raw-text facts, printed at the end.
    _srf_stats: dict[str, int] = {}
    _abort_rederived = [0]
    _abort_errors: list[str] = []

    def _jfp_rc() -> str:
        """The current judging fingerprint, computed once per run.

        Uses `judging_fp_cached()` so one recompute never stamps two versions, and
        turns its `"unknown"` result into an exception.
        """
        from haenv.anchor import judging_fp_cached
        _fp = judging_fp_cached()
        if not _fp or _fp == "unknown":
            raise RuntimeError("cannot compute the judging fingerprint (judging_fp_cached returned unknown)")
        return _fp
    for line in eval_p.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        raw = by_key.get((row.get("case"), row.get("solver")))
        # Offline stubs are never recomputed: their recorded responses cannot rebuild
        # their rows exactly, and they are free to rerun.
        _is_stub = str(row.get("solver")) in _STUBS
        if raw is not None and _is_stub:
            _stub_skipped += 1
            raw = None
        # Not gated on `raw is not None`: a row whose slices all fail to parse has no
        # `raw` but must still be judged.
        add: dict = {}
        # Gated rows: purchased tests are merged into the answer before any judge reads it.
        _gated_row = (row.get("tool_rounds_used") is not None
                      or row.get("tests_from_queries") is not None)
        _gated_gap = None
        if raw is not None and full and _gated_row:
            raw, _gated_gap = merge_gated_purchases(row, raw, _built.get(row.get("case")),
                                                    d / "trace.jsonl")
            if _gated_gap:
                add["gates_recompute_skipped"] = _gated_gap
        if (not _is_stub) and isinstance(row.get("slice_rows"), list) and row["slice_rows"]:
            add.update(apply_slice_raw_facts(
                row["slice_rows"],
                by_slice_facts.get((row.get("case"), row.get("solver"))) or {},
                _srf_stats))
            # Backfill slice-row fields before `run_judges` (see `enrich_slice_rows`).
            _srf_stats["slices_enriched"] = (_srf_stats.get("slices_enriched", 0)
                                             + enrich_slice_rows(row, by_slices))
        if raw is not None:
            for name, fn in RECOMPUTE.items():
                try:
                    add.update(fn(_Out(raw)) or {})
                except Exception as e:
                    add[f"{name}_error"] = str(e)[:120]
            vp = vps.get(row.get("case"))
            if vp is not None:
                try:
                    _sp_ = sps.get(row.get("case"))
                    # Gated rows are judged live on the parsed `SolverOutput`; the same here.
                    _geom, _subj, _why = subjects_for_recompute(
                        row, (_to_output_full(raw, _sp_) if _gated_row and _sp_ is not None
                              else _Out(raw)), by_slices,
                        lambda _r: _to_output_full(_r, _sp_))
                    add["recompute_geometry"] = _geom
                    if _why:
                        add["recompute_subject_gap"] = _why
                    from haenv.judges import run_judges
                    add.update(run_judges(_geom, _subj, vp) or {})
                except GeometryUndecidable as e:
                    # Undeterminable geometry fails by name; it does not fall back to SINGLE.
                    add["recompute_geometry"] = None
                    add["recompute_geometry_undecidable"] = str(e)[:200]
                except Exception as e:
                    add["run_judges_error"] = str(e)[:160]
                # ---- Process judges are recomputed with the current rules ----
                try:
                    from haenv.process import run_process_judges
                    _sp2 = sps.get(row.get("case"))
                    _vis = [e["evidence_id"] for e in (_sp2.evidence_ledger or [])] if _sp2 else []
                    add.update(run_process_judges(raw, _vis) or {})
                except Exception as e:
                    add["process_judges_error"] = str(e)[:160]
                # ---- Kernel tracks (`--tracks`) ----
                # `verifier.grade(out, vp, ev)` can be rebuilt from disk. Off by default: it
                # overwrites `tracks` and goes through the same changed-field trail.
                if "--tracks" in argv:
                    # Gated rows are graded on the merged answer; a row whose purchases cannot
                    # be recovered keeps its live gates (`gates_recompute_skipped` says why).
                    if _gated_row and _gated_gap:
                        pass
                    else:
                        try:
                            import verifier as _V                       # kernel
                            _sp = sps.get(row.get("case"))
                            _ev = [e["evidence_id"] for e in (_sp.evidence_ledger or [])] if _sp else []
                            # A row carrying `slice_rows` is graded as live (`_row_slices`): on its
                            # last slice's own response, with the review gates left to the slice
                            # judge. Read structurally, so an undecidable `geometry` value cannot
                            # knock out the regrade.
                            _srs = row.get("slice_rows")
                            _per_slice = isinstance(_srs, list) and bool(_srs)
                            _subject = _to_output_full(raw, _sp)
                            _last_invalid = False
                            if _per_slice:
                                _sl = _srs[-1]
                                _last_invalid = bool(_sl.get("leak") or _sl.get("slice_unparseable")
                                                     or _sl.get("raw_empty"))
                                _last_raw = (by_slices.get((row.get("case"), row.get("solver")))
                                             or {}).get(str(_sl.get("t")))
                                # `by_key` holds the last parseable response in file order, which
                                # is an earlier slice when the last one came back empty.
                                _subject = (None if (_last_invalid or _last_raw is None)
                                            else _to_output_full(_last_raw, _sp))
                            if _per_slice and _last_invalid:
                                # Live: an invalid last slice gets no case-level grade.
                                add["tracks"], add["gates"], add["overall"] = None, [], "SCORED"
                            elif _subject is None:
                                add["gates_recompute_note"] = (
                                    "slices:末片原文不在盘上 ⇒ 整例门沿用 live 读数,未重算")
                            else:
                                _rep = _V.grade(_subject, vp, _ev, unit_gates_per_slice=_per_slice)
                                add["tracks"] = _rep.tracks
                                add["gates"] = _rep.hard_gate_failures
                                # `overall`, `tracks` and `gates` are written back together from the
                                # same `_rep`; `report.gate_multiplier` reads `overall` + `sg_*`.
                                add["overall"] = _rep.overall
                                if _gated_row:
                                    add["gates_recompute_skipped"] = False
                                    if row.get("gates_recompute_note"):
                                        add["gates_recompute_note"] = None
                        except Exception as e:
                            add["tracks_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            # ---- Per-slice abstention calibration (two-sided) ----
            # Sufficiency is whether the slice shows any true symptom; `data_sufficiency`
            # is backfilled into `slice_rows` from `responses.jsonl`.
            if vp is not None:
                try:
                    from haenv.evaluate import NOOP_FOR, QUANT_FOR
                    _cid = str(row.get("case"))
                    _slices = row.get("slices") or []
                    # Single-shot takes `T` from the row: `quant_t_max` records how far the model
                    # could see.
                    _tmax = (int(max(_slices)) if _slices
                             else (int(row["T"]) if row.get("T") is not None else None))
                    # Live judges these on the last slice and give them no subject when that
                    # slice is invalid; `raw` here is the last parseable response in file order.
                    _srs_q = row.get("slice_rows")
                    _q_last_bad = (isinstance(_srs_q, list) and bool(_srs_q)
                                   and bool(_srs_q[-1].get("leak") or _srs_q[-1].get("slice_unparseable")
                                            or _srs_q[-1].get("raw_empty")))
                    add.update(rejudge_qside(row, _Out({} if _q_last_bad else raw), _built.get(_cid),
                                             NOOP_FOR.get(_cid), QUANT_FOR.get(_cid), _tmax))
                except Exception as e:                    # noqa: BLE001
                    add["qside_probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"
                # Slice rows go through `run_judges(SLICES, ...)`, dispatched by `mount_table`.
        # A response-less ABORT row is re-derived from its own fields; a contradiction fails the
        # run and leaves the row untouched.
        if raw is None and not add and not _is_stub and str(row.get("overall") or "").startswith("ABORT("):
            _ab, _ab_err = rederive_abort(row)
            if _ab_err:
                _abort_errors.append(f"{row.get('case')}|{row.get('solver')}: {_ab_err}")
            elif _ab:
                add.update(_ab)
                _abort_rederived[0] += 1
        # The predicate is "was this row rejudged", not "has a recorded response"
        # (an all-unparseable row has no `raw` but becomes `ABORT`).
        _rejudged = raw is not None or bool(add)
        if _rejudged:
            new_fields |= {k for k in add if k not in row}
            # Adding fields is free; changing a value needs `--allow-change`.
            _dropped = []
            for k, v in list(add.items()):
                if k in row and row[k] != v:
                    changed[k] = changed.get(k, 0) + 1
                    if _allow_change is not None and k not in _allow_change:
                        _dropped.append(k)
                        _conflicts[k] = _conflicts.get(k, 0) + 1
                        add.pop(k)
            if _dropped:
                _rows_guarded += 1
                add["recompute_conflicts"] = ",".join(sorted(_dropped))[:200]
            else:
                # The field describes THIS restamp: drop a note left by an earlier one.
                row.pop("recompute_conflicts", None)
            # Note: a recomputed row must get a fresh judging stamp, whether or not its
            # values changed, so `report.assert_publishable` rejects a partially
            # recomputed batch. Rows not rejudged keep their stamp.
            if _rejudged:
                try:
                    row["judging_sha16"] = _jfp_rc()
                except Exception:            # noqa: BLE001 -- don't silently update the stamp when the fingerprint can't be computed
                    row["judging_sha16_error"] = "recompute 期间指纹算不出,戳未更新"
            row.update(add)
            _case_level.setdefault(row.get("case"), {}).update(
                {k: add[k] for k in CASE_LEVEL_FIELDS if k in add})
            n_patched += 1
        rows.append(row)

    # ---- Case-level fields propagate to every row of the case, including rows
    # without a response ----
    for cid, fields in _case_level.items():
        for r in rows:
            if r.get("case") == cid:
                for k, v in fields.items():
                    if r.get(k) != v:
                        propagated[k] = propagated.get(k, 0) + 1
                    r[k] = v

    eval_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                      encoding="utf-8")
    print(f"[recompute] {d.name}: {n_patched}/{len(rows)} rows recomputed "
          f"(the other rows have no stored response -- mostly offline baselines, kept as they are)")
    if _stub_skipped:
        print(f"[recompute] offline stubs: {_stub_skipped} rows skipped, not recomputed -- "
              f"stubs re-run for free, and their stored responses cannot rebuild their rows exactly")
    print(f"[recompute] new fields: {sorted(new_fields) or '(none)'}")
    # Always report how many slices were scanned and caught, including zero.
    if _srf_stats:
        print(f"[recompute] per-slice raw-text guard (`raw_empty`/`raw_unparseable`): "
              f"scanned {_srf_stats.get('slices_seen', 0)} slices · "
              f"no JSON extractable {_srf_stats.get('unparseable', 0)} · "
              f"empty {_srf_stats.get('empty', 0)} · "
              f"not recomputable {_srf_stats.get('unavailable', 0)} · "
              f"cells re-judged ABORT {_srf_stats.get('rows_aborted', 0)}")
    else:
        print("[recompute] per-slice raw-text guard: this batch has no `slice_rows` "
              "(not a slices geometry) -- not applicable")
    if _conflicts:
        print(f"[recompute] conflicting fields discarded (old values kept, {_rows_guarded} rows affected): "
              f"{dict(sorted(_conflicts.items(), key=lambda x: -x[1]))}")
        print("            the new values of these fields differ from the stored ones, and this run "
              "has no `--allow-change` authorisation.")
        print("            If the judge was fixed, run again with `--allow-change field[,field...]`;")
        print("            if the stored response cannot rebuild the row, do not recompute those rows.")
    print(f"[recompute] changed fields (cells where old != new): {changed or '(none)'}")
    print(f"[recompute] case-level fields propagated to rows without a response "
          f"(offline baselines): {propagated or '(none)'}")
    if n_empty[0]:
        print(f"[recompute] empty responses skipped: {n_empty[0]} cells (a parse failure is not "
              f"a wrong answer; kept as they are)")
    print(f"[recompute] response-less ABORT rows re-derived from their fields: "
          f"{_abort_rederived[0]} stamped · {len(_abort_errors)} contradicted")
    for _e in _abort_errors:
        print(f"[recompute] ❌ ABORT row contradicts its own fields, left unchanged: {_e}")
    # The live-vs-recomputed difference ledger is written to disk.
    _report = {
        "n_rows": len(rows), "n_rejudged": n_patched,
        "new_fields": sorted(new_fields),
        # `changed`: number of cells per field where old != new; `{}` = none.
        "changed_fields": dict(sorted(changed.items(), key=lambda x: -x[1])),
        "n_changed_fields": len(changed),
        "n_changed_cells": sum(changed.values()),
        "conflicts_dropped": dict(sorted(_conflicts.items(), key=lambda x: -x[1])),
        "rows_guarded": _rows_guarded,
        "propagated_case_level": dict(propagated),
        "n_empty_skipped": n_empty[0], "n_stub_skipped": _stub_skipped,
        "slice_raw_facts": dict(_srf_stats),
        "allow_change": (None if _allow_change is None else sorted(_allow_change)),
        "full": full,
        "argv": list(argv),
        "abort_rederived": _abort_rederived[0],
        "abort_rederive_errors": list(_abort_errors),
    }
    _bj = d / "batch.json"
    if _bj.is_file():
        try:
            _m = json.loads(_bj.read_text(encoding="utf-8"))
            _m["recompute_report"] = _report
            _bj.write_text(json.dumps(_m, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError) as e:                   # noqa: BLE001
            # Record the failure rather than skipping silently.
            print(f"[recompute] writing the recompute ledger failed: {type(e).__name__}: {e}")
    # A row that contradicts its own fields is an error, not a quiet skip.
    return 3 if _abort_errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
