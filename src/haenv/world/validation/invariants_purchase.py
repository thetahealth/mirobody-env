"""What an M2 solver can buy on day T: the production gated loop (`haenv.gated.run_gated`, with the M2
plugin's wrappers installed by loading the job) driven by a scripted solver that orders every menu item
in round 1. Zero model calls: the solver's socket call is replaced, the rest of the path is production.

One JSON line per case: {case_id, T, menu: [...], results: {target: {kind, outcome, revealed, observation,
unavailable_reason}}, prompt: the round-1 user text}.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
import os
import pathlib


RECORD_STREAMS = ("HbA1c", "fasting_glucose", "LDL", "triglycerides", "ALT", "AST", "FIB4", "CGM_TIR",
                  "systolic_bp", "diastolic_bp")


class _Trace:
    def __init__(self):
        self.ev = []

    def append(self, kind, payload=None, step=None):
        self.ev.append((kind, payload, step))


def _solver(job, purchase=True):
    from haenv.runtime.solvers import OpenAICompatSolver

    class _Scripted(OpenAICompatSolver):
        def _send_wire(self, body, prompt, *, api_key=None):
            self.wire_log.append(json.loads(body))
            if len(self.wire_log) == 1:
                from haenv.interaction.query.gated import _menu
                self.visible_menu = _menu(self.gated_context or {})
            if purchase and len(self.wire_log) == 1:
                menu = (self.gated_context or {}).get("menu") or []
                c = json.dumps({"queries": [{"kind": i["kind"], "target": i["target"]} for i in menu],
                                "commit": False}, ensure_ascii=False)
            else:
                c = '{"differential":[{"rank":1,"diagnosis":"x"}]}'
            return {"choices": [{"message": {"content": c}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    s = _Scripted("invariants", "scripted", "k", 30, retries=0, backend="openrouter")
    s.wire_log = []
    s.probe_id = getattr(job, "probe_id", "") or ""
    return s


def _run_ctx(job, built):
    from haenv.runtime.state import RunContext
    import haenv.app.evaluate as EV
    from haenv.interaction.qside import load_probes
    ctx = RunContext(workers=1)
    from haenv.locations import probe_directory
    ctx.probes = load_probes(probe_directory(job.root, getattr(job, 'path', None)))
    ctx.allow_retired = bool(getattr(job, "allow_retired", False))
    EV.assign_premises(built, ctx=ctx)
    EV.assign_qside_probes(built, answer_t=EV.answer_windows(job, built, job.results_file.parent), ctx=ctx)
    try:
        from haenv.world.generation.opaque_ids import alias_run_context
        alias_run_context(ctx, built)
    except ImportError:
        pass
    return ctx


def _user_text(wire: dict) -> str:
    out = []
    for m in wire.get("messages") or ():
        if m.get("role") == "user":
            c = m.get("content")
            out.append(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
    return "\n".join(out)


def dump(job_path: pathlib.Path, batch: pathlib.Path, out: pathlib.Path, only: set[str] | None = None) -> int:
    """Write `out` (jsonl); return the number of cases."""
    from haenv.app.job import load_job
    from haenv.app.artifacts.store import load_cases
    import haenv.app.artifacts.batch as batch_mod
    import haenv.interaction.gated as G
    job_path, batch = pathlib.Path(job_path), pathlib.Path(batch)
    root = job_path.parent
    os.environ.setdefault("HAENV_OUTPUT_ROOT", str(root))
    job = load_job(job_path, root=root)
    job.batch, _ = batch_mod.resolve(root, job.task_type, job.job_id, fresh=False, batch=batch.name)
    built = load_cases(batch / "cases.jsonl")
    ctx = _run_ctx(job, built)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for cid, raw in built.items():
            if only is not None and cid not in only:
                continue
            s = _solver(job)
            s.run_ctx = ctx
            s.prompt_mode = "ddx"
            T = int(raw.prediction_context["prediction_time_T"])
            # Capture get_menu on the default, actual solver path. The separate
            # unlimited run is inspection only, never the visible allowance.
            visible = _solver(job, purchase=False)
            visible.run_ctx, visible.prompt_mode = ctx, "ddx"
            G.run_gated(raw, T, visible)
            from haenv.scoring.rows import gated_menu
            rebuilt_budget = G.budget_for(gated_menu(raw, T))
            tr = _Trace()
            G.run_gated(raw, T, s, budget=1e9, trace=tr)
            start = next(p for k, p, _ in tr.ev if k == "case/start")
            calls, res = {}, {}
            for k, p, _ in tr.ev:
                if k == "tool/call":
                    calls[p["call_id"]] = p
                elif k == "tool/result":
                    c = calls.get(p["call_id"], {})
                    res[c.get("target")] = {"kind": c.get("kind"), "outcome": p.get("outcome"),
                                            "revealed": p.get("revealed"), "observation": p.get("observation"),
                                            "unavailable_reason": p.get("unavailable_reason")}
            # the record the solver reads: M2 rewrites the world before the loop (hooks.run_gated)
            from haenv.m2 import core as M2C
            ld = getattr(raw, "longitudinal_data", None) or {}
            if M2C.is_m2_case(raw):
                plan = ((getattr(raw, "adjudication", None) or {}).get("m2_frame") or {}).get("f3_plan")
                ld = M2C.m2_world(raw, ld, plan, T)
            world_ld = {k: [q for q in v if int(q.get("ts", 0)) <= T] for k, v in ld.items() if k in RECORD_STREAMS}
            fh.write(json.dumps({"case_id": cid, "T": T, "menu": start["menu"], "results": res, "world_ld": world_ld,
                                 "visible_menu": visible.visible_menu, "rebuilt_budget": rebuilt_budget,
                                 "prompt": _user_text(s.wire_log[0]) if s.wire_log else ""},
                                ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def _dump_shard(args) -> int:
    job_path, batch, part, only = args
    if not only:
        pathlib.Path(part).write_text("", encoding="utf-8")
        return 0
    import logging
    logging.disable(logging.CRITICAL)
    return dump(job_path, batch, part, only=only)


def dump_parallel(job_path: pathlib.Path, batch: pathlib.Path, out: pathlib.Path, jobs: int = 1) -> int:
    """`dump` over `jobs` worker processes (cases sharded by sha256 of the id), lines in the batch's case order."""
    import hashlib
    import multiprocessing as mp
    batch, out = pathlib.Path(batch), pathlib.Path(out)
    ids = [json.loads(line)["case_id"] for line in open(batch / "cases.jsonl", encoding="utf-8")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("purchase pool must contain nonempty, unique case IDs")
    if jobs <= 1:
        return dump(job_path, batch, out)
    parts = [out.with_name(f"{out.stem}.part{k}.jsonl") for k in range(jobs)]
    shards = [{c for c in ids if int(hashlib.sha256(c.encode()).hexdigest(), 16) % jobs == k} for k in range(jobs)]
    with mp.get_context("fork").Pool(jobs) as pool:
        pool.map(_dump_shard, [(job_path, batch, parts[k], shards[k]) for k in range(jobs)])
    lines = {}
    for part in parts:
        for line in open(part, encoding="utf-8"):
            cid = json.loads(line)["case_id"]
            if cid in lines:
                raise ValueError(f"duplicate purchased case: {cid}")
            lines[cid] = line
        part.unlink()
    if set(lines) != set(ids):
        raise ValueError("incomplete purchase shards")
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(lines[c] for c in ids)
    return len(lines)
