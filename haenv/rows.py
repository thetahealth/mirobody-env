"""Per-geometry row builders (single, gated, slices, multi-round), the iron-law precheck, replay
of saved answers, and the real-visit-rhythm slice schedule.

Split out of `haenv/evaluate.py`; `evaluate` re-exports every name defined here.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from . import mount_table as _MT
import hashlib
import json
from pathlib import Path
import haenv_kernel.verifier as verifier_mod    # kernel
import haenv_kernel.runner as runner_mod    # The leakage gate runs inside `solve_guard.guarded_solve`, shared by every geometry.
from haenv_kernel.build import build_instance
from haenv_kernel.solver import _extract_json
from . import judges as judges_mod, process, wq
from .run_state import RunContext, _WRITE_LOCK, log
from . import qside as _qside   # `render_for` is a plugin hook point: read it live
from .qside import (  # noqa: F401
    render_for,
)
from .run_ledger import (  # noqa: F401
    _note_grid_retry,
    _note_grid_usage,
    save_response,
    save_simple_trace,
)
from .solvers import _to_output
from .solve_guard import json_unextractable
from .mount_table import claim_geometry
from . import slicing as _slicing
from . import judges as _j
from .gated import withhold_signals
from . import tracks as _tk
from .trace import TraceLog as _TraceLog
from .trace import save_trace as _save_trace
from .judges import MULTI
from .mounting import RoundRecorder as _RoundRecorder
from .rhythm import (  # noqa: F401
    REAL_RHYTHM_GAP_DAYS,
    REAL_RHYTHM_SLICES,
    rhythm_gap_feasible,
)


def answer_windows(job, built: dict, batch_dir=None) -> dict[str, int]:
    """The last day each case is answered on: the last slice under the slice geometry, else
    `prediction_time_T`. The single source for the trend question's window.

    Slices come from the batch's frozen slicing table when there is one (the geometry the
    rows ran under), else from `slices_for`. A slicing table that lacks a case falls back to
    `slices_for` here; `run_eval` raises on that case and re-checks the window afterwards.
    """
    out: dict[str, int] = {}
    spec = str(getattr(job, "slices", "") or "") if job is not None else ""
    _frozen = None
    if spec and batch_dir is not None:
        try:
            _frozen = _slicing.load(batch_dir)
        except Exception:                                 # noqa: BLE001
            _frozen = None
    for cid, raw in built.items():
        T = int((raw.prediction_context or {}).get("prediction_time_T", 0) or 0)
        sl = None
        if spec:
            sl = (_frozen[0].get(str(cid)) if _frozen is not None else None)
            if sl is None:
                sl = slices_for(raw, spec)
        geom = claim_geometry(job, len(sl) if sl is not None else None) if job is not None else "single"
        out[str(cid)] = int(max(sl)) if (geom == "slices" and sl) else T
    return out


def _row_single(cid, sname, raw, T, solver, *, ctx: RunContext) -> dict:
    solver.run_ctx = ctx                      # the solver renders and answers within this run
    sp, vp = build_instance(raw, T)
    from .solve_guard import guarded_solve
    ddx0 = (vp.adjudication or {}).get("ddx")

    _iron_law = iron_law_precheck(raw, sp, vp, solver)

    try:
        facts = guarded_solve(
            solver, sp, T,
            prompt_mode=("ddx" if ddx0 else "default"),   # the question side decides the framing, not information from the answer side
            precheck=_iron_law,
            on_output=(lambda o: (save_response(ctx.resp_path, cid, sname, o, ctx=ctx),
                                  save_simple_trace(ctx.resp_path.with_name("trace.jsonl"),
                                                    cid, sname, "single", o))
                       if ctx.resp_path else None),
            tag=f"{cid}|{sname}")
    except wq.IronLawViolation as e:
        log.error("[wq] %s|%s iron-law violation -> aborting this cell: %s", cid, sname, e)
        return {"case": cid, "solver": sname, "iron_law": str(e), "overall": "ABORT(iron_law)"}
    if facts.leak:                                    # leakage -> abort this cell, no score given
        return {"case": cid, "solver": sname, "leak": facts.leak, "overall": "ABORT(leak)"}
    _q, _wq_warn = facts.precheck_result
    out = facts.out
    # No response after retries: abort the cell, excluded from the denominator. The signal is
    # `_raw_text`: absent means an offline stub (not applicable), present but empty means no
    # answer; the output fields are always filled with defaults and cannot tell.
    if facts.raw_empty:
        att = facts.failed_attempts or []
        log.error("[eval] %s|%s no response (retries exhausted after %d attempts) -> "
                  "aborting this cell, excluded from the denominator", cid, sname, len(att))
        return {"case": cid, "solver": sname, "T": T, "overall": "ABORT(no_response)",
                "failed_attempts": att or None,
                "no_response_reason": (str(att[-1].get("error"))[:120] if att else "empty_response")}
    if facts.raw_unparseable:
        log.error("[eval] %s|%s raw text is %d chars but no JSON could be extracted -> "
                  "aborting this cell, excluded from the denominator (the kernel would "
                  "otherwise parse it as {} and record an abstention)",
                  cid, sname, facts.n_chars)
        return {"case": cid, "solver": sname, "T": T, "overall": "ABORT(unparseable)",
                "failed_attempts": facts.failed_attempts,
                "n_chars_unparseable": facts.n_chars,
                "no_response_reason": "json_unextractable"}
    from .judges.safety import evidence_names_for as _ev_names
    rep = verifier_mod.grade(out, vp, [e["evidence_id"] for e in sp.evidence_ledger],
                             evidence_names=_ev_names(sp))
    from .judges import run_judges, SINGLE
    # ---- process-track scoring ----
    # Reads `out._raw`; the outcome grade above sees only `SolverOutput`, so the process record
    # cannot feed into it. Fields carry a `trace_` prefix.
    _proc = process.run_process_judges(getattr(out, "_raw", None),
                                       [e["evidence_id"] for e in (sp.evidence_ledger or [])])
    return {"case": cid, "solver": sname, "T": T,
            **run_judges(SINGLE, out, vp), **_proc,
            **judges_mod.judge_noop_probe(out, ctx.noop_for.get(str(cid)),
                                          delivered=getattr(sp, "longitudinal_data", None)),
            **judges_mod.judge_quant_probe(out, ctx.quant_for.get(str(cid)), t_max=int(T)),
            "action": (out.action or {}).get("selected_action_class"),
            # `Q.world_ref = {world_id, world_truth_hash}` (which world the question points into),
            # distinct from the provenance field `world_ref`.
            "q_world_ref": _q.world_ref,
            "q_question_id": _q.question_id,
            "wq_warnings": _wq_warn or None,
            "tracks": rep.tracks, "gates": rep.hard_gate_failures, "overall": rep.overall}


def iron_law_precheck(raw, sp, vp, solver):
    """Build Q (storing only `world_ref`) and run the W x Q iron laws; returns a thunk yielding
    `(Q, warnings)`.

    Violations of iron laws 1 and 3 (Q embeds W's ground truth; Q cites evidence not in W) and a
    hash mismatch raise; iron law 2 (the distractor ledger inside W) and gold derivation are
    recorded as warnings (see `wq.enforce`). Single/slices pass it as `precheck` (run after the leak
    probe, so a cell failing both is `ABORT(leak)`); gated/multi call it before solving, so
    there a cell failing both is `ABORT(iron_law)`.
    """
    def _run():
        q = wq.build_question(raw, sp, probe_id=getattr(solver, "probe_id", "") or "",
                              judge_ref=str((vp.adjudication or {}).get("ddx", {})
                                            .get("join_gold") or ""))
        return q, wq.enforce(q, raw, strict=True)
    return _run


def _premise_row(cid: str, out, *, ctx: RunContext) -> dict:
    """False-premise judge for the row; `{}` when the case has no premise (not 0)."""
    prem = ctx.premise_for.get(str(cid))
    if not prem or out is None:
        return {}
    return _j.judge_premise_challenge(out, prem)


def merge_bought_tests(answer, targets, menu) -> list[str]:
    """Merge tests bought from the menu's test section into `answer["tests_to_order"]`, in place.

    Returns the merged names (`tests_from_queries` counts them). Signal queries use a different
    vocabulary and are not merged. Shared by `_row_gated` and `tools/recompute_judges.py`.
    """
    _tests_on_menu = {str(i["target"]) for i in (menu or []) if i.get("is_test")}
    _q = [str(t) for t in (targets or []) if str(t) in _tests_on_menu]
    if _q and isinstance(answer, dict):
        answer["tests_to_order"] = list(answer.get("tests_to_order") or []) + _q
    return _q


def budget_abort(committed, spent, budget) -> bool:
    """`ABORT(no_answer:budget)`: no committed answer and the budget spent. Shared by `_row_gated`
    and `tools/recompute_judges.py`, which re-derives the verdict from a row's recorded fields."""
    return (not committed) and bool(budget) and spent >= budget


def gated_menu(raw, T: int) -> list[dict]:
    """The menu `gated.run_gated` hands the model for this case, rebuilt with the same calls."""
    sp, _ = build_instance(raw, int(T))
    _, withheld = withhold_signals(sp)
    from .gated import menu_for          # a plugin hook point: read it live
    return menu_for(sp, withheld, getattr(raw, "case_id", ""))


def _row_gated(cid, sname, raw, T, solver, *, ctx: RunContext) -> dict:
    """Gated (on-demand query) geometry, tool track T1-T4.

    On diagnostic questions T4 may be None (not applicable); T1-T3 are always scored.
    """
    solver.run_ctx = ctx                      # the solver renders and answers within this run
    _sp0, _vp0 = build_instance(raw, int(T))
    solver.prompt_mode = "ddx" if (_vp0.adjudication or {}).get("ddx") else "default"
    from .gated import run_gated
    # `run_gated` has no `precheck` hook, so the iron laws run here, before solving.
    try:
        _q_g, _wq_warn_g = iron_law_precheck(raw, _sp0, _vp0, solver)()
    except wq.IronLawViolation as e:
        log.error("[wq] %s|%s iron-law violation (gated) -> aborting this cell: %s", cid, sname, e)
        return {"case": cid, "solver": sname, "T": int(T),
                "iron_law": str(e), "overall": "ABORT(iron_law)"}
    _tl = _TraceLog(case=cid, solver=sname, geometry="gated") if ctx.resp_path else None
    solver = _replay_solver_for(solver, cid, sname, "gated", ctx=ctx)
    out, tr = run_gated(raw, int(T), solver, trace=_tl)
    if _tl is not None:
        _save_trace(ctx.resp_path.with_name("trace.jsonl"), _tl)
    # Leakage first: it must not fall through to `ABORT(no_response)` below.
    if getattr(tr, "leak", None):
        return {"case": cid, "solver": sname, "T": int(T),
                "leak": list(tr.leak), "overall": "ABORT(leak)"}
    if out is not None and (getattr(tr, "raw_empty", False) or getattr(tr, "raw_unparseable", False)):
        _why = "empty_response" if tr.raw_empty else "json_unextractable"
        log.error("[eval] %s|%s gated final round %s -> aborting this cell, excluded from "
                  "the denominator", cid, sname, _why)
        return {"case": cid, "solver": sname, "T": int(T),
                "overall": "ABORT(no_response)" if tr.raw_empty else "ABORT(unparseable)",
                "failed_attempts": getattr(out, "_failed_attempts", None) or None,
                **({"rate_limit_retries": out._rate_limit_retries}
                   if getattr(out, "_rate_limit_retries", None) else {}),
                "no_response_reason": _why}
    # Budget exhausted with no answer: `ABORT(no_answer:budget)`, not a low score. Running out
    # of rounds with budget left is the model's own behavior and stays in the denominator.
    if out is not None and budget_abort(getattr(tr, "committed", False), tr.spent, tr.budget):
        log.warning("[eval] %s|%s budget exhausted (%.1f/%.1f) with no answer -> "
                    "ABORT(no_answer:budget)",
                    cid, sname, tr.spent, tr.budget)
        return {"case": cid, "solver": sname, "T": int(T),
                "overall": "ABORT(no_answer:budget)",
                "no_response_reason": "budget_exhausted_before_answer",
                "tool_rounds_used": tr.rounds_used, "tool_spent": round(tr.spent, 2),
                "tool_budget": tr.budget, "tool_truncated": getattr(tr, "truncated", 0)}
    sp, vp = build_instance(raw, int(T))
    if out is not None:
        out._gated_rounds = list(tr.rounds_log or [])
    if ctx.resp_path and out is not None:
        save_response(ctx.resp_path, cid, sname, out, rounds=tr.rounds_log, ctx=ctx)
    row = {"case": cid, "solver": sname, "T": int(T),
           "tool_rounds_used": tr.rounds_used, "tool_spent": round(tr.spent, 2),
           "tool_budget": tr.budget, "tool_n_calls": len(tr.calls or []),
           # Purchased targets, needed by `tools/recompute_judges.py` to recompute gates (tests here
           # are bought, not written into the answer).
           "tool_targets": list(tr.targets or []) or None,
           **(_tk.tool_track(tr, vp, key_signals=_tk.key_signals_for(vp)) or {}),
           "overall": "SCORED" if out is not None else "ABORT(no_response)"}
    if out is not None:
        from .judges import run_judges, SINGLE
        _q = merge_bought_tests(getattr(out, "_raw", None), getattr(tr, "targets", None),
                                getattr(tr, "menu", None))
        row["tests_from_queries"] = len(_q)
        # Hard gates and judges run after the merge above; `tools/recompute_judges.py` makes the
        # same merge from the row's `tool_targets`. If grading raises, the `gates` key is omitted
        # (`gate_unknown`) and the error is recorded.
        try:
            from .judges.safety import evidence_names_for as _ev_names
            _rep = verifier_mod.grade(
                out, vp, [e["evidence_id"] for e in (sp.evidence_ledger or [])],
                evidence_names=_ev_names(sp, getattr(tr, "targets", None)))
        except Exception as e:                          # noqa: BLE001
            log.error("[eval] %s|%s gated kernel scoring failed -> recording gate_unknown "
                      "(not writing the `gates` key): %s",
                      cid, sname, e)
            row["gate_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        else:
            row["tracks"] = _rep.tracks
            row["gates"] = _rep.hard_gate_failures
            row["gates_recompute_skipped"] = False
            row["overall"] = _rep.overall
        row.update(run_judges(SINGLE, out, vp))
        from .judges.trajectory import delivered_series as _dlv_series
        row.update(judges_mod.judge_noop_probe(
            out, ctx.noop_for.get(str(cid)),
            delivered=_dlv_series(sp, getattr(tr, "targets", None), gated=True)))
        row.update(judges_mod.judge_quant_probe(out, ctx.quant_for.get(str(cid)), t_max=int(T)))
        row.update(judges_mod.judge_abstention_calibration(out, vp))
        # Process judges on the final answer; the gated ledger has no lab tests (see
        # `render_delta_source_types` in `registry/pack_catalog.yaml`).
        row.update(process.run_process_judges(
            getattr(out, "_raw", None),
            [e["evidence_id"] for e in (sp.evidence_ledger or [])]))
    return row


# How far out of range, as a fraction of the reference range's width, a lab value must be
# to count as worsened. Parsed as in `build_premise`.
ABNORMAL_MARGIN = 0.30


_re_lab = __import__("re").compile(
    r"^(.+?)\s+([-+]?\d+(?:\.\d+)?)\s*([^\s(（]*)\s*[(（]\s*参考\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*[–\-~至]\s*([-+]?\d+(?:\.\d+)?)")


#: Judges of "did the final answer name the right condition": those with subject `LAST` on
#: slices in `mount_table`.
FINAL_ANSWER_JUDGES: tuple[str, ...] = tuple(
    sorted(n for n, row in _MT.MOUNT.items() if row.get("slices") == _MT.LAST))


#: Resume replay of slice answers. Cell rows are written when a cell completes, so a run that dies
#: mid-cell leaves answered slices in `responses.jsonl` and no row; the cell then restarts from its
#: first slice. With `ctx.replay` set (by `run_eval` when resuming), a saved answer that is
#: non-empty, parseable, not truncated and made for the same prompt is reused instead of re-bought.
#: Counts go to `ctx.replay_n` / `ctx.replay_fresh`.


def _reusable(r: dict) -> bool:
    """A saved answer that may stand in for a fresh request: non-empty, parseable, not
    truncated, not an error, and tied to a prompt fingerprint."""
    raw = r.get("raw")
    return not (not isinstance(raw, str) or not raw.strip() or json_unextractable(raw)
                or str(r.get("finish_reason") or "").lower() in ("length", "max_tokens", "error")
                or not r.get("prompt_sha256"))


def load_slice_replay(path) -> dict:
    """Index the reusable answers in `responses.jsonl`.

    Slice rows are keyed `(case, solver, slice_t)`. Multi-round rows (one saved row per round,
    `rounds == [i]`) are keyed `(case, solver, "sha", prompt_sha256)`: a round's prompt is fully
    determined by what it was shown, so an identical fingerprint is the same request. Gated rows
    (`rounds` is a list of round dicts, written only when the cell ends) are never keyed here;
    their per-round answers come from `load_gated_replay`.
    """
    idx: dict = {}
    if path is None or not Path(path).exists():
        return idx
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rd = r.get("rounds")
            if r.get("slice_t") is None:
                if (isinstance(rd, list) and len(rd) == 1 and isinstance(rd[0], int)
                        and _reusable(r)):
                    idx.setdefault((str(r["case"]), str(r["solver"]), "sha", r["prompt_sha256"]), r)
                continue
            if rd or not _reusable(r):
                continue
            idx.setdefault((str(r["case"]), str(r["solver"]), int(r["slice_t"])), r)
    return idx


#: Gated cells persist every answered round here as it arrives (the cell's own `responses.jsonl`
#: row is written only when the cell ends, so a cell that dies mid-way leaves nothing else).
#: Read only by the resume replay; no other reader consumes it.
GATED_ROUNDS_FILE = "gated_rounds.jsonl"


def gated_rounds_path(ctx: RunContext) -> Path | None:
    return ctx.resp_path.with_name(GATED_ROUNDS_FILE) if ctx.resp_path else None


def load_gated_replay(path) -> dict:
    """`(case, solver, "sha", prompt_sha256)` -> first reusable saved gated round."""
    idx: dict = {}
    if path is None or not Path(path).exists():
        return idx
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if _reusable(r):
                idx.setdefault((str(r["case"]), str(r["solver"]), "sha", r["prompt_sha256"]), r)
    return idx


def _prompt_sha_for(solver, payload) -> str:
    return hashlib.sha256(_qside.render_for(solver, payload)[0].encode()).hexdigest()[:16]


def _persist_gated_round(cid, sname, out, *, ctx: RunContext) -> None:
    """Append one answered gated round to the sidecar (skipped when nothing reusable came back)."""
    path = gated_rounds_path(ctx)
    txt = getattr(out, "_raw_text", None)
    if path is None or txt is None or not getattr(out, "_prompt_sha", None):
        return
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "case": cid, "solver": sname, "raw": txt,
                "prompt_sha256": out._prompt_sha, "prompt_mode": getattr(out, "_prompt_mode", "default"),
                "probe_id": getattr(out, "_probe_id", None), "usage": getattr(out, "_usage", None),
                "finish_reason": getattr(out, "_finish", None),
                "max_tokens": getattr(out, "_max_tokens", None),
                "failed_attempts": getattr(out, "_failed_attempts", None) or None,
                **({"rate_limit_retries": out._rate_limit_retries}
                   if getattr(out, "_rate_limit_retries", None) else {}),
                "latency_s": getattr(out, "_latency_s", None),
                "reasoning_chars": getattr(out, "_reasoning_chars", None)},
                ensure_ascii=False) + "\n")


class _ReplaySolver:
    """Wraps a solver for one cell: a saved usable answer for the same prompt is returned
    without a request; anything else goes to the real solver.

    `persist_rounds` (gated): every freshly answered round is appended to the sidecar so that a
    later resume can reuse it. The prompt fingerprint covers the whole request (revealed data,
    budget spent, round number, menu), so a reused round is only ever the answer to the exact
    prompt it would have been re-asked; after the first divergent round nothing later matches
    and the rest of the cell is bought fresh.
    """

    def __init__(self, inner, cid, sname, persist_rounds: bool = False, *, ctx: RunContext):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "_key", (str(cid), str(sname)))
        object.__setattr__(self, "_persist", bool(persist_rounds))

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)

    def _lookup(self, payload):
        idx = self._ctx.replay_idx or {}
        sha = _prompt_sha_for(self._inner, payload)
        mode = getattr(self._inner, "prompt_mode", "default")
        try:
            t = int(payload.prediction_context["prediction_time_T"])
        except (KeyError, TypeError, ValueError):
            t = None
        for k in ((*self._key, t), (*self._key, "sha", sha)):
            r = idx.get(k)
            if r is not None and r.get("prompt_sha256") == sha and (r.get("prompt_mode") or "default") == mode:
                return r
        return None

    def solve(self, payload):
        r = self._lookup(payload)
        if r is not None:
            out = _to_output(_extract_json(r["raw"]), payload)
            out._raw_text, out._prompt_mode = r["raw"], r.get("prompt_mode") or "default"
            out._probe_id, out._prompt_sha = r.get("probe_id"), r.get("prompt_sha256")
            out._usage, out._finish = r.get("usage"), r.get("finish_reason")
            out._failed_attempts = r.get("failed_attempts")
            out._latency_s, out._reasoning_chars = r.get("latency_s"), r.get("reasoning_chars")
            out._max_tokens = r.get("max_tokens")
            out._replayed = True
            self._ctx.replay_n[self._key] = self._ctx.replay_n.get(self._key, 0) + 1
            return out
        out = self._inner.solve(payload)
        self._ctx.replay_fresh[self._key] = self._ctx.replay_fresh.get(self._key, 0) + 1
        if self._persist:
            _persist_gated_round(*self._key, out, ctx=self._ctx)
        return out


def _replay_solver_for(solver, cid, sname, geometry: str, *, ctx: RunContext):
    """The resume wrapper for a geometry. Fresh runs (`ctx.replay` false) get no replay; gated
    cells still record their rounds so a later resume has something to reuse."""
    gated = geometry == "gated"
    if not ctx.replay:
        return _ReplaySolver(solver, cid, sname, persist_rounds=True, ctx=ctx) if gated else solver
    if ctx.replay_idx is None:
        idx = load_slice_replay(ctx.resp_path)
        idx.update(load_gated_replay(gated_rounds_path(ctx)))
        ctx.replay_idx = idx
    return _ReplaySolver(solver, cid, sname, persist_rounds=gated, ctx=ctx)


def take_replay_count(case: str, solver: str, *, ctx: RunContext) -> dict:
    """Per-cell resume accounting: answers reused (`slices_reused`, name kept for slice rows) and
    the answers bought again in a resumed run (`replay_rebought`); only set when a replay was
    active and something was reused, so a fresh run's rows are unchanged."""
    k = (str(case), str(solver))
    n, fresh = ctx.replay_n.pop(k, 0), ctx.replay_fresh.pop(k, 0)
    if not n:
        return {}
    return {"slices_reused": n, "replay_rebought": fresh}


def _row_slices(cid, sname, raw, slices, solver, *, ctx: RunContext) -> dict:
    """Slice geometry: N independent consultations on the same world (no carryover), one answer per slice.

    Unlike `_row_multi`, nothing is carried forward: it tests whether scattered symptoms are
    combined across separate sessions. A leak voids only that slice.
    """
    solver.run_ctx = ctx                      # the solver renders and answers within this run
    from .tracks import alternative_a1, _answer_text, _differential
    rows = []
    _, vp0 = build_instance(raw, int(max(slices)))
    is_ddx = bool((vp0.adjudication or {}).get("ddx"))
    real_ids = set(wq.injected_manifest(raw.case_id)      # Q-side ledger
                   .get("real_symptom_evidence_ids", []) or [])
    from .solve_guard import guarded_solve
    solver = _replay_solver_for(solver, cid, sname, "slices", ctx=ctx)

    def _persist(o, _t):
        if getattr(o, "_replayed", False):        # already on disk; still counts once toward the cell's usage
            _note_grid_usage(cid, sname, o, ctx=ctx)
            _note_grid_retry(cid, sname, o, ctx=ctx)
            return
        save_response(ctx.resp_path, cid, sname, o, slice_t=_t, ctx=ctx)
        save_simple_trace(ctx.resp_path.with_name("trace.jsonl"),
                          cid, sname, "slices", o, slice_t=_t)
    for t in slices:
        sp, vp = build_instance(raw, int(t))
        n_sym = sum(1 for ev in (sp.evidence_ledger or [])      # count of genuine symptoms visible in this slice
                    if str(ev.get("source_type", "")) == "patient_reported_symptom"
                    and ev.get("evidence_id") in real_ids)
        facts = guarded_solve(
            solver, sp, int(t),
            prompt_mode=("ddx" if is_ddx else "default"),
            precheck=iron_law_precheck(raw, sp, vp, solver),
            on_output=(lambda o, _t=int(t): _persist(o, _t)) if ctx.resp_path else None,
            tag=f"{cid}|{sname}@t{int(t)}")
        if facts.leak:
            rows.append({"t": int(t), "n_symptoms": n_sym, "leak": facts.leak, "all_drivers": []})
            continue
        out = facts.out
        # Visible evidence per slice, for judging whether a revision followed new information
        # (`VerifierPayload` has no ledger).
        _vis = [str(ev.get("evidence_id")) for ev in (sp.evidence_ledger or [])]
        # Lab values far out of range are tracked as worsening signals (their `source_type` is not
        # a symptom); only the collection side sees the evidence text.
        _abn = []
        for ev in (sp.evidence_ledger or []):
            if str(ev.get("source_type")) != "lab_result":
                continue
            m = _re_lab.search(str(ev.get("symptom") or ""))
            if not m:
                continue
            try:
                val, lo, hi = float(m.group(2)), float(m.group(4)), float(m.group(5))
            except ValueError:
                continue
            _w = max(hi - lo, 1e-9)
            _out = (lo - val) / _w if val < lo else ((val - hi) / _w if val > hi else 0.0)
            if _out > ABNORMAL_MARGIN:
                _abn.append(str(ev.get("evidence_id")))
        rows.append({"t": int(t), "n_symptoms": n_sym,
                     "visible_real": [e for e in _vis if e in real_ids],
                     "visible_abnormal_lab": _abn,
                     "visible_other": [e for e in _vis if e not in real_ids],
                     "risk_category": out.forecast.get("risk_category"),
                     "all_drivers": [d.get("driver") for d in (out.drivers or [])],
                     "answer_text": " ".join(
                         [_answer_text(out)]
                         + [str(x.get("diagnosis") or "") for x in _differential(out)]),
                     "differential": [x.get("diagnosis") for x in _differential(out)],
                     # Evidence fields read by `judge_refuting_evidence` and `judge_what_not_to_do`.
                     "differential_evidence": [
                         {"ruled_out_by": x.get("ruled_out_by"),
                          "supporting_evidence": x.get("supporting_evidence") or []}
                         for x in _differential(out)
                         if isinstance(x, dict) and x.get("ruled_out_by")],
                     # `None` (not answered) stays distinct from `[]` (answered: nothing).
                     "what_not_to_do": ((out.action or {}).get("what_not_to_do")),
                     "join_type": ((getattr(out, "_raw", None) or {}).get("join_type") or None),
                     "action": (out.action or {}).get("selected_action_class"),
                     # Read by per-slice safety gates; `None` stays distinct from `False`.
                     "clinician_review_required": ((out.action or {})
                                                   .get("clinician_review_required")),
                     "data_sufficiency": ((getattr(out, "data_quality", None) or {})
                                          .get("data_sufficiency")
                                          if isinstance(getattr(out, "data_quality", None), dict)
                                          else None),
                     "tests_to_order": [str(x) for x in
                                        ((getattr(out, "_raw", None) or {}).get("tests_to_order") or [])],
                     "referral_specialty": [str(x) for x in
                                            ((getattr(out, "_raw", None) or {}).get("referral_specialty") or [])],
                     # Whether the slice got a response at all (`_to_output({})` would otherwise look like an
                     # answered slice).
                     "raw_empty": facts.raw_empty,
                     "raw_unparseable": facts.raw_unparseable,
                     "a1": alternative_a1(out)["a1"]})
        _out_last = out
    # No slice answered => ABORT the whole cell, as in single-shot. Unparseable slices are
    # treated as unanswered, so their canned abstention never reaches the judges.
    _bad_slices = [r for r in rows if r.get("raw_unparseable")]
    if _bad_slices:
        log.error("[eval] %s|%s %d/%d slice(s) have non-empty raw text but no extractable "
                  "JSON; those slices are excluded from scoring (the kernel would otherwise "
                  "parse them as {} and record a canned abstention)",
                  cid, sname, len(_bad_slices), len(rows))
        for _bs in _bad_slices:
            for _k in ("answer_text", "differential", "join_type", "action",
                       "data_sufficiency", "risk_category", "all_drivers"):
                _bs[_k] = None
            _bs["slice_unparseable"] = True
    _rw = [r for r in rows if "raw_empty" in r]
    if (_rw and all(r.get("raw_empty") for r in _rw)) or (
            rows and all(r.get("raw_empty") or r.get("raw_unparseable") for r in rows)):
        log.error("[eval] %s|%s all %d slice(s) had no response -> aborting this cell, "
                  "excluded from the denominator (a backend failure is not recorded as an "
                  "abstention)", cid, sname, len(_rw))
        return {"case": cid, "solver": sname, "slices": list(map(int, slices)),
                "overall": "ABORT(no_response)", "n_slices": len(rows),
                "answered_slices": 0, "slices_missing": len(_rw),
                "no_response_reason": "all_slices_empty"}
    # All slices leaked => `ABORT(leak)`, as in the other geometries (a partial leak voids only
    # those slices). Checked after `ABORT(no_response)`.
    _leaked = [r for r in rows if r.get("leak")]
    if rows and len(_leaked) == len(rows):
        _reasons = sorted({str(x) for r in _leaked for x in (r.get("leak") or ())})
        log.error("[eval] %s|%s all %d slice(s) leaked -> ABORT(leak), no score and no "
                  "downgrade: %s",
                  cid, sname, len(rows), _reasons[:3])
        return {"case": cid, "solver": sname, "slices": list(map(int, slices)),
                "overall": "ABORT(leak)", "n_slices": len(rows),
                "answered_slices": 0, "leak": _reasons,
                "leak_scope": "all_slices"}
    sp_full, vp_full = build_instance(raw, int(max(slices)))
    ddx = (vp_full.adjudication or {}).get("ddx")
    _out_last = locals().get("_out_last")
    # If the last slice is invalid, the `last`-subject judges get no subject rather than an
    # earlier slice.
    _last_row = rows[-1] if rows else None
    if _last_row is not None and (_last_row.get("leak") or _last_row.get("slice_unparseable")
                                  or _last_row.get("raw_empty")):
        _out_last = None
    # Hard gates run on the last slice. The kernel's five tracks are recorded too, as diagnostics
    # (the authority here is `wk_*`), with `tracks_basis` = `last_slice`. The two review gates
    # are judged per slice (`slices_gates`), so they stay off this case-level list.
    _gates: list = []
    _tracks_last = None
    if _out_last is not None:
        try:
            from .judges.safety import evidence_names_for as _ev_names
            _rep_last = verifier_mod.grade(
                _out_last, vp_full,
                [e["evidence_id"] for e in (sp_full.evidence_ledger or [])],
                unit_gates_per_slice=True, evidence_names=_ev_names(sp_full))
            _gates = list(getattr(_rep_last, "hard_gate_failures", None) or [])
            _tracks_last = getattr(_rep_last, "tracks", None)
        except Exception as e:                          # noqa: BLE001
            log.error("[eval] %s|%s final-slice kernel scoring failed -> recording "
                      "gate_unknown: %s", cid, sname, e)
            _gates = None
    # Process judges on the last slice (`sp_full`'s ledger is its visible set); `{}` if the
    # last slice is invalid.
    _proc = process.run_process_judges(
        getattr(_out_last, "_raw", None),
        [e["evidence_id"] for e in (sp_full.evidence_ledger or [])])
    from .judges import run_judges, SLICES
    return {"case": cid, "solver": sname, "slices": list(map(int, slices)),
            "gold_driver": (vp_full.gold_drivers or [None])[0],
            "ddx_diagnosis": (ddx or {}).get("diagnosis"),
            "tracks": _tracks_last, "tracks_basis": "last_slice",
            **run_judges(SLICES, {"rows": rows, "last": _out_last}, vp_full),
            **_proc,
            **_premise_row(cid, _out_last, ctx=ctx),
            **judges_mod.judge_noop_probe(_out_last, ctx.noop_for.get(str(cid)),
                                          delivered=getattr(sp_full, "longitudinal_data", None)),
            **judges_mod.judge_quant_probe(_out_last, ctx.quant_for.get(str(cid)),
                                           t_max=int(max(slices)) if slices else None),
            # Abstention uses the per-slice judge (`slices_abstention`, inside `run_judges`); diagnosis
            # judges receive the last slice via the `last` subject.
            "slice_rows": rows,
            # `gates`: non-empty = hit, `[]` = no hit, key absent = not graded (`gate_unknown`).
            **({"gates": _gates} if _gates is not None else {}),
            "overall": "FAIL(gate)" if _gates else "SCORED"}


#: Deciles of real clinic visit intervals (days), from a real hypothyroidism EMR sample
#: (965 patients, 28,814 intervals): median 28, Q1-Q3 8-68, >180 days 8.9%.
REAL_VISIT_DECILES: tuple[int, ...] = (3, 7, 12, 21, 28, 37, 56, 87, 163)


def use_real_rhythm(T: int) -> bool:
    """Whether this horizon uses the real clinic rhythm; same threshold as
    `rhythm_gap_feasible`.
    """
    return rhythm_gap_feasible(T)


def _real_rhythm_days(case_id: str, T: int, symptom_days: list[int],
                      force_gap: bool = False) -> list[int]:
    """Schedule slices by real clinic visit intervals rather than symptom days.

    `REAL_RHYTHM_SLICES` intervals are drawn from `REAL_VISIT_DECILES` and normalized to the
    window; `force_gap` adds one `REAL_RHYTHM_GAP_DAYS` interval. Seeded by `case_id`, so the
    schedule is reproducible.
    """
    import hashlib
    import random
    rnd = random.Random(int(hashlib.sha256(str(case_id).encode()).hexdigest()[:12], 16))
    edges = (1,) + REAL_VISIT_DECILES + (400,)
    n = max(2, int(REAL_RHYTHM_SLICES))

    def _gap() -> int:
        i = rnd.randrange(len(edges) - 1)                  # equal probability per bucket, uniform within a bucket
        lo, hi = edges[i], max(edges[i], edges[i + 1])
        return rnd.randint(lo, hi)

    raw_gaps = [_gap() for _ in range(n)]
    gap_i = -1
    if force_gap:
        # The forced gap is excluded from normalization (which would shrink it); the other
        # intervals share the remaining window.
        if not rhythm_gap_feasible(T, n):
            log.warning("[slices] %s: declared rhythm_gap but T=%d cannot fit it "
                        "(needs >= %d) ⇒ this case has no long-gap tier",
                        case_id, T, int(REAL_RHYTHM_GAP_DAYS) + n)
            force_gap = False
        else:
            gap_i = len(raw_gaps) // 2
            raw_gaps[gap_i] = int(REAL_RHYTHM_GAP_DAYS)
    budget = (T - 1) - (int(REAL_RHYTHM_GAP_DAYS) if gap_i >= 0 else 0)
    others = [g for i, g in enumerate(raw_gaps) if i != gap_i]
    total = sum(others) or 1
    scale = max(0.0, budget) / total
    days: list[int] = []
    acc = 0.0
    for i, g in enumerate(raw_gaps):
        acc += (int(REAL_RHYTHM_GAP_DAYS) if i == gap_i else g * scale)
        d = int(round(acc))
        if days and d <= days[-1]:
            d = days[-1] + 1                               # no overlap / no going backward allowed
        if d > T:
            break
        days.append(d)
    days = [d for d in days if 0 < d <= T]
    if len(days) < 2:
        return symptom_days or days
    return days


def slices_for(raw, spec) -> list[int]:
    """Slice geometry: this case's query time points (all <=T).

    `auto`: the days real symptoms appear; `auto+neutral`: plus neutral points; `real-rhythm`:
    real clinic intervals; an explicit list is used as given.
    """
    T = int(raw.prediction_context["prediction_time_T"])
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec if int(x) <= T]
    # Only real symptoms (from the Q-side ledger, not W's adjudication), not injected benign
    # events.
    real = set(wq.injected_manifest(raw.case_id)
               .get("real_symptom_evidence_ids", []) or [])
    def _days(ids_filter) -> list[int]:
        return sorted({int(e.get("source_timestamp", -1)) for e in (raw.evidence_ledger or [])
                       if ids_filter(e)
                       and str(e.get("source_type", "")) == "patient_reported_symptom"
                       and 0 < int(e.get("source_timestamp", -1)) <= T})

    days = _days(lambda e: (not real or e.get("evidence_id") in real))
    if str(spec).strip().startswith("real-rhythm"):
        # The gap tier is tagged per case (`meta.rhythm_gap`); `+gap` in the spec is the master
        # switch.
        _meta = (getattr(raw, "latent_premise", None) or {}).get("meta") or {}
        _fg = bool("+gap" in str(spec)) and bool(_meta.get("rhythm_gap"))
        # Too short a horizon would make the normalized rhythm denser than `auto+neutral`; fall back
        # to `auto` (logged).
        if not use_real_rhythm(T):
            log.warning("[slices] %s: declared real-rhythm but T=%d < %d ⇒ "
                        "the rhythm would be denser than auto (median ~T/13 days), "
                        "falling back to auto",
                        raw.case_id, T, int(REAL_RHYTHM_GAP_DAYS) + int(REAL_RHYTHM_SLICES))
        else:
            return _real_rhythm_days(raw.case_id, T, days, force_gap=_fg)
    if str(spec).strip() != "auto+neutral":
        return days

    # ---- `auto+neutral` ----
    # Adds one slice between consecutive real-symptom slices when only benign events occur in
    # between, so neutral transitions (no new real symptom) exist to judge belief drift.
    benign = _days(lambda e: bool(real) and e.get("evidence_id") not in real)
    out = []
    for i, d in enumerate(days):
        out.append(d)
        nxt = days[i + 1] if i + 1 < len(days) else T + 1
        mid = [b for b in benign if d < b < nxt]
        if mid:
            out.append(max(mid))            # take the last benign time point in this segment: the most benign events have accumulated by then
    return sorted(set(out))


def _row_multi(cid, sname, raw, solver, cadence, *, ctx: RunContext) -> dict:
    solver.run_ctx = ctx                      # the solver renders and answers within this run
    _Tm = int(raw.prediction_context["prediction_time_T"])
    _spm, _vpm = build_instance(raw, _Tm)
    solver.prompt_mode = "ddx" if (_vpm.adjudication or {}).get("ddx") else "default"
    # `RoundRecorder` persists each round, checks it for empty/unparseable responses, and lets
    # a per-round `LeakageError` surface as `ABORT(leak)`.
    def _save_round(i, out, _f):
        if not ctx.resp_path:
            return
        if getattr(out, "_replayed", False):      # already on disk from the run that died
            _note_grid_usage(cid, sname, out, ctx=ctx)
            _note_grid_retry(cid, sname, out, ctx=ctx)
            return
        save_response(ctx.resp_path, cid, sname, out, rounds=[i], ctx=ctx)
        save_simple_trace(ctx.resp_path.with_name("trace.jsonl"), cid, sname, "multi",
                          out, step=int(i))

    _rec = _RoundRecorder(_replay_solver_for(solver, cid, sname, "multi", ctx=ctx), on_round=_save_round)
    try:
        traj, grade, tE = runner_mod.run_multiround(raw, _rec, cadence=cadence)
    except runner_mod.LeakageError as e:
        log.error("[eval] %s|%s multi-round leak -> ABORT(leak): %s", cid, sname, e)
        return {"case": cid, "solver": sname, "geometry": "multi",
                "rounds": len(_rec.outputs), "overall": "ABORT(leak)",
                "abort_detail": str(e)[:300]}
    _bad = _rec.abort_reason()
    if _bad is not None:
        log.warning("[eval] %s|%s multi-round: %d round(s), %d bad -> %s",
                    cid, sname, len(_rec.facts), len(_rec.bad_rounds), _bad)
        return {"case": cid, "solver": sname, "geometry": "multi",
                "rounds": len(traj), "overall": _bad,
                "bad_rounds": _rec.bad_rounds}
    _out_last = _rec.outputs[-1] if _rec.outputs else None
    return {"case": cid, "solver": sname, "rounds": len(traj),
            # The kernel's `repair` tag means "belief changed between adjacent rounds", hence the
            # name.
            "n_belief_changed": sum(1 for r in traj if r["repair"] == "revised"),
            # noop/quant probes are judged on the last round. The trajectory judges come from the `multi`
            # column of `mount_table.MOUNT` via `run_judges`.
            **(judges_mod.judge_noop_probe(_out_last, ctx.noop_for.get(str(cid)))
               if _out_last is not None else
               {"multi_last_missing": "the recorder captured no answer in any round -> "
                                      "the last-round family is unmeasured (not folded to 0)"}),
            **(judges_mod.judge_quant_probe(_out_last, ctx.quant_for.get(str(cid)), t_max=_Tm)
               if _out_last is not None else {}),
            # Single-shot-family judges get the last round output (`ctx["round_outputs"][-1]`), the
            # trajectory judges get `traj`; a missing subject is recorded, never substituted.
            **judges_mod.run_judges(MULTI, {"traj": traj}, _vpm,
                                    ctx={"round_outputs": _rec.outputs,
                                         "premise": ctx.premise_for.get(str(cid))}),
            **process.run_process_judges(
                getattr(_out_last, "_raw", None),
                [e["evidence_id"] for e in (_spm.evidence_ledger or [])]),
            "trajectory": traj, "trackE": tE["E"], "latencies": tE["latencies"],
            "trap_fooled": tE["trap_fooled"], "n_real": tE["n_real"], "n_trap": tE["n_trap"],
            "tracks": grade.tracks, "gates": grade.hard_gate_failures, "overall": grade.overall}
