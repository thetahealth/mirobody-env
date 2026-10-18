"""Outer execution of sealed management requests through an equivalent route pool.

Only dispatch and provenance live here. Tasks, retry geometry, raw reply parsing,
core decisions and scores delegate to the unchanged management pipeline.
"""
from __future__ import annotations

from haenv.locations import source_path
from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import threading

from haenv.scoring.management import pipeline as scoring
from haenv.scoring.routes import RoutedJudges
from haenv.scoring.routes import SnapshotMismatch
from haenv.scoring.routes import check_snapshot
from haenv.runtime.judge_scheduler import run_bounded
from haenv.runtime.judge_scheduler import MAX_JUDGE_LANES
from haenv.runtime.budget import BudgetExceeded, BudgetLedger


def validate(out):
    """Compare scoring identity outside scoring, retaining every sealed input check.

    The old management code stamp also covered transport and scheduling. It is a
    provenance record, not a reason to invalidate an unchanged scoring criterion.
    """
    from haenv.provenance.judging_provenance import _bind
    from haenv.provenance.judging import current_fingerprints
    from haenv.provenance.judging import criterion_fingerprints
    manifest = json.loads((out / "manifest.json").read_text())
    recorded = manifest["code"].get("judging_criteria", {}).get("management_rubric")
    if recorded is None:
        # Unstamped prepared runs still have the strict old code check.
        return scoring.validate(out)
    import yaml
    current = current_fingerprints()["criteria"]["management_rubric"]
    if manifest["policy"] != scoring.load_policy():
        current = criterion_fingerprints(overrides={"registry/management_rubric.yaml":
            yaml.safe_dump(manifest["policy"], allow_unicode=True, sort_keys=False).encode()})["criteria"]["management_rubric"]
    if recorded != current:
        raise ValueError("Management scoring criterion differs from the sealed run")
    return _bind(scoring.validate, code_state=lambda: manifest["code"])(out)


def _sample_error(sample, identity):
    if sample.get("error_class"):
        return {"error": sample["error_class"], **({"snapshot_check": sample["snapshot_check"]}
                if sample.get("snapshot_check") else {})}
    if sample.get("judge_routes"):
        record = sample["judge_routes"][-1]
        check = record.get("snapshot_check", {})
        try:
            # New OpenRouter samples carry the preflight catalog evidence. Legacy
            # samples without that tag must still prove the date in their response.
            catalog = check.get("catalog", {}) if record.get("snapshot_evidence") == "catalog" else None
            check_snapshot({"model": check.get("returned")}, identity, catalog=catalog)
        except SnapshotMismatch as exc:
            return {"error": type(exc).__name__, "snapshot_check": exc.snapshot_check}
    return None


def _result(cell, snapshots, manifest, out, identity):
    verdicts, errors, requests, attempts, routes = {}, [], [], [], []
    for task, items in scoring._tasks(cell, snapshots[cell["snapshot_key"]], manifest):
        for req, filename in scoring._attempts(task, manifest["policy"]):
            path = out / "raw_judge" / filename
            sample = scoring._read_sample(path, req)
            attempts.append(str(path.relative_to(out)))
            routes.extend(sample.get("judge_routes", []))
            if not sample.get("retryable"):
                break
        requests.append(str(path.relative_to(out)))
        error = _sample_error(sample, identity)
        if error:
            errors.append({"request_id": req.sample_id, **error})
            continue
        try:
            verdicts.update(scoring.parse_reply(sample["raw"], items, scoring.cell_answer(cell)))
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"request_id": req.sample_id, "error": type(exc).__name__})
    verdicts = scoring.with_core_verdicts(cell["rubric"], verdicts, cell["core_decision"])
    summary = scoring.score(cell["rubric"], None if errors else verdicts, manifest["policy"],
                            core_decision=cell["core_decision"])
    return {"key": cell["key"], "source": cell["source"], "collection_status": "collected",
            "rubric_version": cell["rubric"]["version"], "facts_sha256": cell["snapshot_key"],
            "code": manifest["code"], "plan_sha256": cell["plan_sha256"],
            "answer_sha256": cell.get("answer_sha256"), "core_decision": cell["core_decision"],
            "verdicts": verdicts, "raw_judge_files": requests, "raw_judge_attempt_files": attempts,
            "errors": errors, "summary": summary, "offline": False, "judge_routes": routes}


def run(out, *, offline=False, ledger_path=None, limit_usd=None, cfg=None, concurrency=None, max_cells=None):
    if offline or not cfg or not cfg.get("judge_routes"):
        return scoring.run(out, offline=offline, ledger_path=ledger_path, limit_usd=limit_usd,
                           cfg=cfg, concurrency=concurrency, max_cells=max_cells)
    out = Path(out).resolve()
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(out, ledger_path, limit_usd, cfg, concurrency, max_cells)


def _run(out, ledger_path, limit_usd, cfg, concurrency, max_cells):
    manifest = validate(out)
    if ledger_path is None or limit_usd is None:
        raise ValueError("Paid judging requires an explicit independent ledger and cap")
    if max_cells is not None and (type(max_cells) is not int or max_cells <= 0):
        raise ValueError("max_cells must be positive")
    if concurrency is not None and (type(concurrency) is not int or not 1 <= concurrency <= MAX_JUDGE_LANES):
        raise ValueError("Concurrency must be an integer from 1 through 64")
    scoring._outside_sources(ledger_path, [Path(s["batch"]) for s in manifest["sources"]])
    policy = manifest["policy"]
    execution = {"judge": policy["judge"]["model_id"], "offline": False,
                 "ledger": str(Path(ledger_path).resolve())}
    execution_path = out / "execution.json"
    if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
        raise ValueError("Do not mix offline/paid execution or ledgers in one run")
    from haenv.scoring.routes import configured_routes
    specs, identity = configured_routes(cfg, policy)
    route_path = out / "route-execution.json"
    if route_path.exists() and json.loads(route_path.read_text())["identity"] != identity:
        raise ValueError("Cannot change the judge snapshot while resuming a run")
    ledger = BudgetLedger(Path(ledger_path), limit_usd=limit_usd)
    snapshots = {r["key"]: r["snapshot"] for r in scoring.read_jsonl(out / "snapshots.jsonl")}
    cells = scoring.read_jsonl(out / "cells.jsonl")
    samples, results = out / "raw_judge", out / "results"
    pending = [c for c in cells if c["status"] == "ready" and not (results / f"{c['key']}.json").exists()]
    pending.sort(key=lambda c: c["snapshot_key"])
    selected = pending if max_cells is None else pending[:max_cells]
    # A fully completed resume needs no credentials or price endpoint, and buys nothing.
    judges = RoutedJudges(cfg, policy, ledger, out / "receipts") if selected else None
    if judges:
        scoring.atomic_json(out / "price-metadata.json",
            {"routes": {n: r.metadata for n, r in judges.routes.items()}, "unavailable": judges.unavailable})
    scoring.atomic_json(execution_path, execution)
    scoring.atomic_json(route_path, {"identity": identity,
        "executor_files": {name: scoring.digest(source_path(name)) for name in (
            'haenv/scoring/management/route_run.py', 'haenv/scoring/routes.py', 'haenv/runtime/judge_scheduler.py',
            "haenv/runtime/completion.py", "haenv/runtime/transport.py", 'haenv/app/settings.py')},
        "routes": {n: s.model_dump(mode="json") for n, s in specs.items()}})
    samples.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    stop, stop_reasons, stop_lock = threading.Event(), [], threading.Lock()

    def one(cell):
        answer = scoring.cell_answer(cell)
        is_empty = scoring.empty_answer(answer)
        stub = scoring.stub_reply(cell["rubric"]["items"], answer) if is_empty else None
        votes = json.loads(stub.raw)["verdicts"] if stub else None
        for task, items in scoring._tasks(cell, snapshots[cell["snapshot_key"]], manifest):
            for req, filename in scoring._attempts(task, policy):
                path = samples / filename
                if path.exists():
                    sample = scoring._read_sample(path, req)
                else:
                    if stop.is_set():
                        return
                    sample = {"request_sha256": scoring.sha256(asdict(req)), "request": asdict(req),
                              "task": asdict(task)}
                    try:
                        if stub:
                            raw = scoring.canonical({"verdicts": {i["id"]: votes[i["id"]] for i in items}})
                            sample.update(raw=raw, usage=stub.usage, origin="deterministic_empty", judge_routes=[])
                        else:
                            reply, routes = judges(req)
                            sample.update(raw=reply.raw, usage=reply.usage, origin=execution["judge"], judge_routes=routes)
                            if reply.raw is None or not reply.raw.strip():
                                raise scoring.EmptyJudgeResponse(reply.raw, reply.usage)
                    except BudgetExceeded as exc:
                        with stop_lock:
                            stop_reasons.append(str(exc))
                        stop.set()
                        return
                    except Exception as exc:
                        sample.update(raw=None, usage=getattr(exc, "usage", None), error_class=type(exc).__name__,
                            retryable=isinstance(exc, scoring.EmptyJudgeResponse),
                            judge_routes=getattr(exc, "route_attempts", sample.get("judge_routes", [])))
                        if isinstance(exc, SnapshotMismatch):
                            sample["snapshot_check"] = exc.snapshot_check
                    scoring.atomic_json(path, sample)
                if not sample.get("retryable"):
                    break
        result = _result(cell, snapshots, manifest, out, identity)
        scoring.atomic_json(results / f"{cell['key']}.json", result)

    if selected:
        from haenv.runtime.scheduler import paid_limits
        limit, _ = paid_limits(cfg)
        workers = min(MAX_JUDGE_LANES, limit, sum(r.spec.concurrency for r in judges.routes.values()))
        if concurrency is not None:
            workers = min(workers, concurrency)
        run_bounded(selected, one, concurrency=workers, stop=stop)
    overlay = []
    for cell in cells:
        path = results / f"{cell['key']}.json"
        if path.exists():
            result = json.loads(path.read_text())
            rebuilt = _result(cell, snapshots, manifest, out, identity)
            if result["verdicts"] != rebuilt["verdicts"] or result["summary"] != rebuilt["summary"]:
                raise ValueError("Saved management result differs from raw judge responses")
        else:
            result = {"key": cell["key"], "source": cell["source"],
                "collection_status": "pending" if cell["status"] == "ready" else
                                     "未采集" if cell["status"] == "not_collected" else cell["status"],
                "rubric_version": manifest["rubric_version"], "facts_sha256": cell["snapshot_key"],
                "code": manifest["code"], "plan_sha256": cell["plan_sha256"],
                "core_decision": cell["core_decision"],
                "verdicts": scoring.with_core_verdicts(cell["rubric"], {}, cell["core_decision"]),
                "summary": scoring.score(cell["rubric"], None, policy, core_decision=cell["core_decision"]),
                "offline": False, "judge_routes": []}
        overlay.append(result)
    scoring.write_jsonl(out / "rubric_results.jsonl", overlay)
    summary = {"run_id": manifest["run_id"], "rubric_version": manifest["rubric_version"],
        "offline": False, "paid_calls": None,
        "states": dict(Counter(r["collection_status"] if r["collection_status"] != "collected" else
                               r["summary"]["status"] for r in overlay)), "budget": ledger.summary(),
        "stop_reason": stop_reasons[0] if stop_reasons else None, "source_eval_overwritten": False}
    scoring.atomic_json(out / "summary.json", summary)
    return summary
