"""Query mode: the evaluated model reads the patient record through read-only tools instead of
receiving it whole in one prompt. It is the default way a case of the four packs (P2 triage, P3
medication adjustment, P4 follow-up reading, M2 differential diagnosis with purchases) is put to a
model (`config.query.geometry: query`); `--geometry single` keeps the one-prompt form.

* `tools.py`   -- the record store built from the solver payload only, the common tool set, the
                  result guard;
* `adapter.py` -- what a pack declares (its record sections, question header, guard terms, decisive
                  evidence); each pack's adapter lives in `haenv/<pack>/query.py`;
* `loop.py`    -- the native function-calling loop and its trace;
* `solver.py`  -- the production solver in query mode (same accounting and wire check per turn);
* `gated.py`   -- M2: purchases from the priced menu inside the same tool loop, one purchase
                  round per round of the gated geometry (`haenv/interaction/gated.py:run_gated`), so the
                  budget, the menu, the Gatekeeper and the tool track stay the gated geometry's;
* `profile.py` -- the process profile (`q_*` columns), reported, never in the score;
* `stubs.py`   -- offline query stubs (no model calls).

`install(cfg)` (called by `haenv run`) records query limits and enables explicit
integration calls. It never replaces the shared evaluator's functions.
`build_solvers_for` selects native query or full-context solvers;
`execute_row` arms the result guard, adds process profiles and persists the trace.

Scoring is the pack's own row path, unchanged: the final answer is the solve output.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import json
import pathlib
import threading

#: job plugin group -> pack adapter name
PACK_GROUPS = {"haenv.p2": "p2", "haenv.p3": "p3", "haenv.p4": "p4", "haenv.m2": "m2"}

_INSTALLED: list[bool] = []
_LOCK = threading.Lock()
_TRACE_LOCK = threading.Lock()
_CFG: dict = {}


def settings(cfg: dict) -> dict:
    """The query-mode limits of a run (`config.query`, schema defaults filled)."""
    from haenv.app.settings import QuerySection
    return QuerySection.model_validate(dict(cfg.get("query") or {})).model_dump()


def job_packs(job) -> list[str]:
    p = getattr(job, "plugins", None) or ()
    groups = (p.get("judges") or ()) if isinstance(p, dict) else p
    return [PACK_GROUPS[g] for g in groups if g in PACK_GROUPS]


def _query_models(models: list, q: dict) -> list:
    from haenv.runtime.solvers import GoogleSolver
    from haenv.runtime.solvers import OpenAICompatSolver
    from haenv.interaction.query.solver import as_query_solver
    out = []
    for name, make in models:
        probe = make()
        if isinstance(probe, (OpenAICompatSolver, GoogleSolver)):
            out.append((name, lambda m=make: as_query_solver(m(), q)))
        else:
            out.append((name, make))
    return out


def _single_models(models: list) -> list:
    from haenv.runtime.solvers import CLISolver
    from haenv.runtime.solvers import GoogleSolver
    from haenv.runtime.solvers import OpenAICompatSolver
    from haenv.interaction.query.single import as_single_solver

    def make_single(make):
        solver = make()
        return (as_single_solver(solver) if isinstance(solver, (CLISolver, OpenAICompatSolver, GoogleSolver))
                else solver)

    return [(name, lambda m=make: make_single(m)) for name, make in models]


def _persist(ctx, row: dict, trace: dict) -> None:
    rp = getattr(ctx, "resp_path", None)
    if not rp:
        return
    rec = {"case": row.get("case"), "solver": row.get("solver"), **trace}
    with _TRACE_LOCK, open(pathlib.Path(rp).with_name("query_trace.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _armed_row(o_row, gated: bool):
    @functools.wraps(o_row)
    def row(cid, sname, raw, T, solver, *a, **kw):
        if not getattr(solver, "query", None):
            return o_row(cid, sname, raw, T, solver, *a, **kw)
        from haenv_kernel.build import build_instance
        from haenv.interaction.query.adapter import ADAPTERS
        from haenv.interaction.query.adapter import adapter_for
        from haenv.interaction.query.tools import QueryLeak
        from haenv.interaction.query.tools import ResultGuard
        sp, vp = build_instance(raw, int(T))
        adapter = ADAPTERS.get("m2") if gated else adapter_for(sp)
        if adapter is None:
            return o_row(cid, sname, raw, T, solver, *a, **kw)
        words, numbers = adapter.guard_terms(vp)
        solver.query_guard = ResultGuard(int(T), words, numbers)
        solver._last_query = None
        try:
            out = o_row(cid, sname, raw, T, solver, *a, **kw)
        except QueryLeak as e:
            out = {"case": cid, "solver": sname, "T": int(T), "leak": e.violations, "overall": "ABORT(leak)"}
        last = getattr(solver, "_last_query", None)
        out["query_mode"] = True
        if last is not None:
            from haenv.interaction.query.profile import profile
            trace, store = last
            out.update(profile(trace, adapter.key_lookups(vp, store)))
            _persist(kw.get("ctx"), out, trace)
        return out
    return row


def build_solvers_for(base, job, cfg):
    """Select the declared run geometry around the ordinary solver factory."""
    if not _INSTALLED or not job_packs(job):
        return base(job, cfg)
    from haenv.interaction.query import stubs as _stubs
    q = settings(cfg) if cfg is not None else dict(_CFG)
    if q["geometry"] == "single":
        return _single_models(base(job, cfg))
    stubs = list(_stubs.QUERY_STUBS) if job.include_baseline else []
    saved, job.include_baseline = job.include_baseline, False
    try:
        models = base(job, cfg)
    finally:
        job.include_baseline = saved
    return [(n, (lambda f=f: f(q))) for n, f in stubs] + _query_models(models, q)


def execute_row(base, gated: bool, *args, **kwargs):
    """Arm the guard explicitly; the executor function itself is never replaced."""
    return (_armed_row(base, gated)(*args, **kwargs) if _INSTALLED
            else base(*args, **kwargs))


def install(cfg: dict | None = None) -> None:
    """Idempotently enable query integration and record the run's declared limits."""
    if cfg is not None:
        _CFG.clear()
        _CFG.update(settings(cfg))
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from haenv.interaction.query.adapter import load_pack_adapters
        load_pack_adapters()
        _INSTALLED.append(True)
