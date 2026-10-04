"""Runner -- orchestrates and enforces the isolation order:

    build_instance(T)  ->  leakage_probe(solver_payload)  [abort if this fails]
                       ->  solver.solve(solver_payload only)
                       ->  verifier.grade(output, verifier_payload)

run(): in-process. run_isolated(): the verifier runs in a subprocess.
run_sandboxed(): the solver also runs in a sandbox directory.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from .build import build_instance, leakage_probe
from .schema import RawCase, SolverOutput, SolverPayload
from .solver import Solver
from . import verifier as verifier_mod

log = logging.getLogger("harness.runner")
HERE = Path(__file__).parent


def load_config() -> dict:
    with open(HERE / "config.yaml") as f:
        return yaml.safe_load(f)


class LeakageError(RuntimeError):
    pass


def _prepare(raw: RawCase, T: int):
    solver_payload, verifier_payload = build_instance(raw, T)
    ok, violations = leakage_probe(solver_payload, T)
    if not ok:
        raise LeakageError(f"{raw.case_id}: leakage gate failed -> {violations}")
    ledger_ids = [e["evidence_id"] for e in solver_payload.evidence_ledger]
    return solver_payload, verifier_payload, ledger_ids


def run(raw: RawCase, T: int, solver: Solver):
    """In-process. The solver only receives solver_payload; the verifier only receives
    output+verifier_payload."""
    solver_payload, verifier_payload, ledger_ids = _prepare(raw, T)
    output = solver.solve(solver_payload)
    report = verifier_mod.grade(output, verifier_payload, ledger_ids)
    return report


def _window_days(raw: RawCase) -> int:
    w = str(raw.prediction_context.get("prediction_window", "180d")).rstrip("d")
    return int(w) if w.isdigit() else 180


def _belief(out) -> tuple:
    return (out.forecast.get("risk_category"),
            (out.drivers[0].get("driver") if out.drivers else None),
            (out.action or {}).get("selected_action_class"))


def run_multiround(raw: RawCase, solver: Solver, cadence: int | None = None):
    """Multi-round review loop at a fixed cadence: each round re-diagnoses on the data
    revealed up to the pointer (only ts <= pointer, leakage-probed), and adjacent
    rounds are diffed into a repair state (confirmed/revised). Returns
    (trajectory, final_grade); ground truth is used only in the final scoring.
    """
    from . import verifier as verifier_mod
    T = int(raw.prediction_context["prediction_time_T"])
    cadence = cadence or int(load_config().get("prediction_cadence_days", 7))
    _, verifier_payload = build_instance(raw, T)
    horizon = T + _window_days(raw)
    max_rounds = min(60, _window_days(raw) // cadence + 2)

    trajectory, prior, pointer, r = [], None, T, 0
    while pointer <= horizon and r < max_rounds:
        r += 1
        revealed = {sig: [p for p in pts if p["ts"] <= pointer]
                    for sig, pts in raw.longitudinal_data.items()}
        payload = SolverPayload(
            case_id=raw.case_id, user_profile=raw.user_profile,
            prediction_context={**raw.prediction_context, "prediction_time_T": pointer},
            longitudinal_data=revealed,
            evidence_ledger=[e for e in raw.evidence_ledger if e.get("source_timestamp", 10**9) <= pointer])
        ok, viol = leakage_probe(payload, pointer)
        if not ok:
            raise LeakageError(f"{raw.case_id} round {r}@day{pointer}: {viol}")
        out = solver.solve(payload)
        repair = "confirmed" if (prior is not None and _belief(out) == _belief(prior)) else \
                 ("revised" if prior is not None else "initial")
        # Per-round `visible_ev` / `cited_ev` (ids only, never text) let the process
        # judges check what was newly revealed against what the agent says it relied on.
        _seen_ev = {e["evidence_id"] for e in payload.evidence_ledger}
        trajectory.append({"round": r, "day": pointer, "risk_cat": out.forecast.get("risk_category"),
                           "risk": round(float(out.forecast.get("risk", 0)), 2),
                           "top_driver": out.drivers[0].get("driver") if out.drivers else None,
                           "action": (out.action or {}).get("selected_action_class"),
                           "repair": repair,
                           # evidence IDs visible this round
                           "visible_ev": sorted(_seen_ev),
                           # evidence IDs the agent claims to have relied on this round
                           "cited_ev": sorted({e for e in (out.cited_evidence or [])}),
                           # daily-stream points newly revealed this round (a revision can rest on a stream alone)
                           "n_new_points": sum(
                               1 for pts in revealed.values() for q in pts
                               if pointer - cadence < int(q["ts"]) <= pointer),
                           # The agent's own free text (not ground truth), truncated to 600 characters, for
                           # judging whether it acknowledged a conflict.
                           "notes": (str(getattr(out, "notes", "") or "")[:600] or None)})
        prior = out
        pointer += cadence

    ledger_ids = [e["evidence_id"] for e in raw.evidence_ledger if e.get("source_timestamp", 10**9) <= T]
    final_grade = verifier_mod.grade(prior, verifier_payload, ledger_ids)
    trackE = verifier_mod.score_track_E(trajectory, verifier_payload, cadence)
    final_grade.tracks["E"] = trackE["E"]
    log.info("[multiround] %s rounds=%d TrackE=%s latency=%s trap_fooled=%d",
             raw.case_id, r, trackE["E"], trackE["latencies"], trackE["trap_fooled"])
    return trajectory, final_grade, trackE


def _verify_subprocess(output: SolverOutput, verifier_payload, ledger_ids) -> dict:
    """Scores in a separate `verifier.py` subprocess; no solver state reaches it.

    Launched as `-m haenv_kernel.verifier` rather than by file path: since the kernel
    became a package, running the file directly would execute it as `__main__` with no
    package context and its relative imports would fail.
    """
    blob = json.dumps({"solver_output": asdict(output),
                       "verifier_payload": asdict(verifier_payload),
                       "ledger_ids": ledger_ids}, ensure_ascii=False)
    proc = subprocess.run([sys.executable, "-m", "haenv_kernel.verifier"],
                          input=blob, capture_output=True, text=True, cwd=str(HERE.parent))
    if proc.returncode != 0:
        raise RuntimeError(f"verifier subprocess failed: {proc.stderr}")
    return json.loads(proc.stdout)


def run_isolated(raw: RawCase, T: int, solver: Solver):
    """Runs the verifier in a separate subprocess."""
    solver_payload, verifier_payload, ledger_ids = _prepare(raw, T)
    output = solver.solve(solver_payload)
    log.info("[runner] verifier ran in isolated pid via subprocess")
    return _verify_subprocess(output, verifier_payload, ledger_ids)


def run_sandboxed(raw: RawCase, T: int, solver_cmd: list[str],
                  task_type: str | None = None, keep_sandbox: bool = False) -> dict:
    """Least-privilege isolation: the solver runs in `tasks/<type>/<case-id>@T<t>/`,
    a directory containing only `solver_payload.json` (≤T), with no path to the
    ground truth in `cases/` or to the verifier.

    Protocol for `solver_cmd`: run with the sandbox as CWD, read
    `./solver_payload.json`, write `./solver_output.json`. The verifier runs
    outside the sandbox.
    """
    import shutil
    import cases as _cases

    solver_payload, verifier_payload, ledger_ids = _prepare(raw, T)   # build + leakage_probe before anything is written to the sandbox
    task_type = task_type or _cases.type_of(raw.case_id)
    sandbox = HERE / "tasks" / task_type / f"{raw.case_id}@T{T}"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    # The only file in the sandbox: the minimal payload.
    (sandbox / "solver_payload.json").write_text(solver_payload.dumps(), encoding="utf-8")

    # Strip PYTHONPATH so the solver cannot `import cases.*` and read ground truth.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(solver_cmd, cwd=str(sandbox), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sandboxed solver failed (rc={proc.returncode}): {proc.stderr[:500]}")
    out_file = sandbox / "solver_output.json"
    if not out_file.exists():
        raise RuntimeError(f"sandboxed solver produced no solver_output.json in {sandbox}")
    output = SolverOutput(**json.loads(out_file.read_text(encoding="utf-8")))

    log.info("[runner] sandboxed solver ran in %s (least-privilege: only solver_payload.json visible)", sandbox)
    report = _verify_subprocess(output, verifier_payload, ledger_ids)   # verifier runs outside the sandbox
    if not keep_sandbox:
        shutil.rmtree(sandbox, ignore_errors=True)
    return report


# ============================ noise-robustness dual run ============================
def _scalar_score(grade) -> float:
    """Folds a GradeReport into one scalar: a hard-gate hit -> 0, else the mean of the
    non-None tracks.
    """
    overall = grade.overall if hasattr(grade, "overall") else grade.get("overall")
    if str(overall).startswith("FAIL"):
        return 0.0
    tracks = grade.tracks if hasattr(grade, "tracks") else grade.get("tracks", {})
    vals = [v for v in tracks.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def run_robustness(clean_raw: RawCase, noisy_raw: RawCase, solver_factory,
                   T: int | None = None, mode: str = "single", cadence: int | None = None) -> dict:
    """Runs the clean and the noise-injected version of the same case and returns a
    robustness number:

        robustness = clamp01(1 - (score_clean - score_noisy)/score_clean)

    `solver_factory` returns a fresh Solver per run (multi-round solvers carry
    state). mode='single' uses run(); mode='multi' uses run_multiround().
    """
    T = T or int(clean_raw.prediction_context["prediction_time_T"])
    if mode == "single":
        gc = run(clean_raw, T, solver_factory())
        gn = run(noisy_raw, T, solver_factory())
        extra = {"clean_gates": gc.hard_gate_failures, "noisy_gates": gn.hard_gate_failures,
                 "clean_tracks": gc.tracks, "noisy_tracks": gn.tracks,
                 "clean_overall": gc.overall, "noisy_overall": gn.overall}
    elif mode == "multi":
        _, gc, ec = run_multiround(clean_raw, solver_factory(), cadence)
        _, gn, en = run_multiround(noisy_raw, solver_factory(), cadence)
        extra = {"clean_trackE": ec, "noisy_trackE": en,
                 "clean_gates": gc.hard_gate_failures, "noisy_gates": gn.hard_gate_failures}
    else:
        raise ValueError(f"mode must be 'single'|'multi', got {mode!r}")
    sc, sn = _scalar_score(gc), _scalar_score(gn)
    robustness = None if sc == 0 else round(max(0.0, min(1.0, 1 - (sc - sn) / sc)), 3)
    log.info("[robustness] %s vs %s mode=%s clean=%.3f noisy=%.3f robustness=%s",
             clean_raw.case_id, noisy_raw.case_id, mode, sc, sn, robustness)
    return {"mode": mode, "clean_case": clean_raw.case_id, "noisy_case": noisy_raw.case_id,
            "score_clean": round(sc, 3), "score_noisy": round(sn, 3),
            "robustness": robustness, **extra}
