"""Cell-parallel semantic execution with isolated clients and durable vote journals."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import logging

from .judge_scheduler import MAX_JUDGE_LANES, run_bounded
from .semantic_budget import BudgetExceeded, cost_summary
from .semantic_judge import JudgeTask, evaluate_consensus
from .semantic_rubric import summarize_verdicts


def atomic_json(path: Path, value) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        file.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n");file.flush();os.fsync(file.fileno())
    os.replace(file.name, path)


def run_cells(records: list[dict], out: Path, run_id: str, ledger, call_factory, *,
              concurrency: int, max_cells: int | None = None) -> dict:
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_JUDGE_LANES:
        raise ValueError("Concurrency must be between 1 and %d" % MAX_JUDGE_LANES)
    if max_cells is not None and (type(max_cells) is not int or max_cells <= 0):
        raise ValueError("max_cells must be positive")
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("Duplicate cell identity")
    started = time.perf_counter()
    journal, result_dir = out / "samples", out / "results"
    journal.mkdir(parents=True, exist_ok=True);result_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    stop_reasons = []
    stop_lock = threading.Lock()
    states = Counter(r["status"] for r in records if r["status"] != "ready")
    existing, new = [], []
    for record in records:
        if record["status"] != "ready":continue
        if (result_dir / f"{record['key']}.json").exists():
            existing.append(record)
            continue
        # A sample whose unknown charge was reconciled at its bound is never
        # bought again: the transport replays it as a failed call (call_error,
        # not a vote), exactly as the original attempt ended, and the adaptive
        # third sample follows under the unchanged 2+1 protocol.
        new.append(record)
    selected = new if max_cells is None else new[:max_cells]
    states["pending"] += len(new) - len(selected)

    def one(record, *, replay=False):
        task = JudgeTask(**{**record["task"], "item_ids": tuple(record["task"]["item_ids"])})
        history = journal / f"{record['key']}.jsonl"
        previous = [json.loads(line) for line in history.read_text().split("\n")
                    if line.strip()] if history.exists() else []
        if replay and not previous:
            raise ValueError("Saved result has no sample journal")
        call = None if replay else call_factory()
        def dispatch(request):
            if replay:
                raise AssertionError("A completed cell cannot acquire new votes during resume")
            if stop.is_set():
                raise BudgetExceeded("Run stopped before a new independent request")
            return call(request)
        def persist(sample):
            with history.open("a", encoding="utf-8") as file:
                file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                file.flush();os.fsync(file.fileno())
        cell_started = time.perf_counter()
        try:
            consensus = evaluate_consensus(task, dispatch, prior_samples=previous, on_sample=persist,
                                           parallel_first_pair=True)
        except BudgetExceeded as exc:
            with stop_lock:
                if not stop_reasons:stop_reasons.append(str(exc))
            stop.set()
            return None
        summary = summarize_verdicts(consensus["verdicts"], record["rubric"])
        result = {"source": record["source"], "consensus": consensus, "summary": summary}
        path = result_dir / f"{record['key']}.json"
        if replay:
            saved = json.loads(path.read_text())
            if any(saved.get(key) != value for key, value in result.items()):
                raise ValueError("Existing result differs from the sample journal")
        else:
            result["execution"] = {"cell_seconds": round(time.perf_counter() - cell_started, 3),
                                   "concurrency": concurrency}
            atomic_json(path, result)
            from .semantic_budget import cost_summary
            logging.getLogger(__name__).info("semantic cell %s: %s; %.2fs; %s",
                record["key"][:12], consensus["status"], result["execution"]["cell_seconds"],
                cost_summary(ledger.snapshot()))
        return consensus["status"]

    for record in existing:
        states[one(record, replay=True)] += 1
    outcomes = run_bounded(selected, one, concurrency=concurrency, stop=stop)
    states.update(result if result is not None else "pending" for result in outcomes)
    if states.get("pending") == 0:states.pop("pending", None)
    budget = ledger.snapshot()
    status = ("completed" if states.get("resolved", 0) > 0
              and not any(n for key, n in states.items() if key != "resolved")
              and not (budget["halt_reason"] or stop_reasons) else "incomplete")
    report = {"run_id": run_id, "status": status, "states": dict(states),
              "visited_cells": len(existing) + sum(result is not None for result in outcomes),
              "new_cells": sum(result is not None for result in outcomes),
              "stop_reason": stop_reasons[0] if stop_reasons else None,
              "budget": budget, "cost": cost_summary(budget), "source_eval_overwritten": False,
              "execution": {"concurrency": concurrency,
                            "wall_seconds": round(time.perf_counter() - started, 3)}}
    atomic_json(out / "summary.json", report)
    return report
