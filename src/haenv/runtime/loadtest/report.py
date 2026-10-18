"""The speed report of a finished rehearsal, read from its run directory.

Sources: `run.json` (driver timeline), `requests.jsonl` (fake provider), `probes/` (timing
probes in the run processes), `logs/`, the run's ledger and the receipts in its batches.

* wall time, and the answer/judging timeline: the last solver answer, the last judge vote,
  judge lag = (last vote - last answer) / (last answer - start);
* per-cell shares of the rehearsed models' cells (probe `cell`, labelled with the solver name):
  waiting for a paid slot (`slot`, untagged),
  inside the provider call (`send`, untagged), and the rest, the local overhead;
* ledger operation latency (probe `ledger`): p50 / p99 / max per operation;
* crashes: a run process that ended other than 0 or 6 (6: judging left cells unresolved, whose
  states the report lists), or a rehearsed model's cell that ended in an exception (stub rows
  are not rehearsed and are not counted);
* budget: pauses and refusals in the ledger's ops log, the committed total against the limit;
* resume (when the run was killed): cells answered before the kill and missing after
  (lost), cells answered before the kill with a receipt written after it (bought twice),
  and generations the provider delivered before the kill that no receipt records.
"""
from __future__ import annotations

import json
import re
import signal
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

#: Default acceptance thresholds (a spec's `acceptance` overrides them).
ACCEPTANCE = {"overhead_share": 0.05, "ledger_p99_s": 0.05, "judge_lag_share": 0.10}
#: Seconds from the start of a run that count as its start-up burst (every lane opens at once);
#: a spec's `burst_s` overrides it. The ledger latency is reported for the burst and after it.
BURST_S = 300.0
#: Seconds before the kill within which a delivered generation may be unrecorded (in flight).
KILL_GRACE_S = 5.0
_SOLVER_OUTCOMES = ("ok", "extra_data")
#: Exit codes of a run that finished: 0, or 6 when semantic judging left cells unresolved
#: (`judging` in the report says which states).
_CLEAN_EXIT = (0, 6)
#: A cell that ended in an exception (`haenv.evaluate`): `[eval] <case>|<solver> failed: <error>`.
_CELL_ERROR = re.compile(r"\[eval\] \S+\|(\S+) failed: (.*)")


def _jsonl(path: Path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


@dataclass
class SpeedReport:
    run_dir: Path
    numbers: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c["pass"] for c in self.checks.values())

    @classmethod
    def read(cls, run_dir: Path) -> "SpeedReport":
        run_dir = Path(run_dir)
        run = json.loads((run_dir / "run.json").read_text())
        accept = {**ACCEPTANCE, **(run["spec"].get("acceptance") or {})}
        start, end = run["started"], run.get("ended") or run["started"]
        n: dict = {"wall_s": round(end - start, 1), "time_scale": run["spec"]["time_scale"],
                   "expected_cells": run.get("expected_cells"), "answered_cells": run.get("answered")}
        requests = _jsonl(run_dir / "requests.jsonl")
        outcomes = defaultdict(int)
        for r in requests:
            outcomes[f"{'judge' if r['kind'] == 'judge' else 'solver'}:{r['outcome']}"] += 1
        n["requests"] = dict(sorted(outcomes.items()))
        solver_t = [r["t1"] for r in requests if r["kind"] != "judge" and r["outcome"] in _SOLVER_OUTCOMES]
        judge_t = [r["t1"] for r in requests if r["kind"] == "judge" and r["outcome"] in _SOLVER_OUTCOMES]
        last_answer = max(solver_t, default=start)
        last_vote = max(judge_t, default=last_answer)
        n["last_answer_s"] = round(last_answer - start, 1)
        n["last_vote_s"] = round(last_vote - start, 1)
        n["judge_lag_s"] = round(max(0.0, last_vote - last_answer), 1)
        n["judge_lag_share"] = round(n["judge_lag_s"] / max(1e-9, last_answer - start), 4)
        n["backend_load"] = cls._backend_load(requests, run["spec"].get("backend_limits") or {})
        n["provider_service_s"] = round(sum(r["service_s"] for r in requests) * n["time_scale"], 1)

        probes = [e for f in sorted((run_dir / 'probes').glob("probe-*.jsonl")) for e in _jsonl(f)]
        rehearsed = set(run.get("models") or [])
        by = defaultdict(list)
        for e in probes:
            if e["p"] == "cell" and e.get("label") is not None and e["label"] not in rehearsed:
                continue                                # a stub row, not a rehearsed model
            role = "judge" if "judge" in e.get("tags", []) else "solver"
            by[(e["p"], role)].append(e)
        cell = sum(e["dt"] for e in by[("cell", "solver")])
        slot = sum(e["dt"] for e in by[("slot", "solver")])
        send = sum(e["dt"] for e in by[("send", "solver")])
        n["cells_probed"] = len(by[("cell", "solver")])
        n["cell_s_total"] = round(cell, 1)
        n["slot_wait_share"] = round(slot / cell, 4) if cell else None
        n["provider_share"] = round(send / cell, 4) if cell else None
        n["overhead_share"] = round(max(0.0, cell - slot - send) / cell, 4) if cell else None
        scale = n["time_scale"] or 1.0
        local = max(0.0, cell - slot - send)
        n["overhead_share_real"] = round(local / (local + (slot + send) / scale), 4) if cell else None
        n["judge_slot_wait_s"] = round(sum(e["dt"] for e in by[("slot", "judge")]), 1)
        ledger = defaultdict(list)
        ledger_cpu, burst, steady = [], [], []
        burst_s = float(run["spec"].get("burst_s", BURST_S))
        for e in by[("ledger", "solver")] + by[("ledger", "judge")]:
            ledger[e["f"].rsplit(".", 1)[-1]].append(e["dt"])
            (burst if e["t"] - start < burst_s else steady).append(e["dt"])
            if e.get("cpu") is not None:
                ledger_cpu.append(e["cpu"])
        n["ledger_ops"] = {op: {"n": len(v), "p50_ms": round(statistics.median(v) * 1e3, 2),
                                "p99_ms": round(_pct(v, 0.99) * 1e3, 2), "max_ms": round(max(v) * 1e3, 2)}
                           for op, v in sorted(ledger.items())}
        all_ledger = [dt for v in ledger.values() for dt in v]
        n["ledger_p99_ms"] = round(_pct(all_ledger, 0.99) * 1e3, 2)
        n["ledger_burst"] = {"s": burst_s, "n": len(burst), "p50_ms": round(_pct(burst, 0.5) * 1e3, 2),
                             "p99_ms": round(_pct(burst, 0.99) * 1e3, 2)}
        n["ledger_steady"] = {"n": len(steady), "p50_ms": round(_pct(steady, 0.5) * 1e3, 2),
                              "p99_ms": round(_pct(steady, 0.99) * 1e3, 2)}
        #: CPU time of the calling thread inside a ledger operation; the rest of `ledger_p99_ms`
        #: is waiting (the queue for the write lock, the interpreter lock, I/O).
        n["ledger_cpu_p99_ms"] = round(_pct(ledger_cpu, 0.99) * 1e3, 2) if ledger_cpu else None

        rcs = [rc for a in run["attempts"] for rc in a.get("rcs", [])]
        final_rcs = run["attempts"][-1].get("rcs", []) if run["attempts"] else []
        cell_errors = defaultdict(int)
        for log in sorted((run_dir / "logs").glob("*.log")):
            for line in log.read_text(errors="replace").splitlines():
                m = _CELL_ERROR.search(line)
                if m and m.group(1) in rehearsed:
                    cell_errors[f"{m.group(1)}: {m.group(2)[:80]}"] += 1
        n["exit_codes"], n["cell_errors"] = rcs, dict(cell_errors)

        judging = defaultdict(int)
        for summary in (run_dir / "root" / "results").glob("*/*/*-pk*/semantic/*/summary.json"):
            for state, count in (json.loads(summary.read_text()).get("states") or {}).items():
                judging[state] += count
        n["judging"] = dict(sorted(judging.items()))

        ops = _jsonl(Path(str(run_dir / "ledger" / "budget.json") + ".ops.jsonl"))
        n["budget_events"] = dict(sorted(defaultdict(int, {}).items()))
        events = defaultdict(int)
        for o in ops:
            events[o.get("event") or o.get("action") or "?"] += 1
        n["budget_events"] = dict(sorted(events.items()))
        refusals = sum("BudgetCapReached" in p.read_text(errors="replace") or "Budget exhausted" in
                       p.read_text(errors="replace") for p in (run_dir / "logs").glob("*.log"))
        n["logs_with_budget_refusal"] = refusals

        cut_rc = -signal.SIGKILL if run.get("stopped") else None   # processes cut by max_wall_s
        checks = {
            "overhead": {"value_real": n["overhead_share_real"], "limit": accept["overhead_share"],
                         "pass": n["overhead_share_real"] is not None
                         and n["overhead_share_real"] <= accept["overhead_share"]},
            "ledger_p99": {"value_ms": n["ledger_p99_ms"], "limit_ms": accept["ledger_p99_s"] * 1e3,
                           "pass": bool(all_ledger) and n["ledger_p99_ms"] <= accept["ledger_p99_s"] * 1e3},
            "ledger_p99_steady": {"value_ms": n["ledger_steady"]["p99_ms"], "n": len(steady),
                                  "limit_ms": accept["ledger_p99_s"] * 1e3,
                                  "pass": bool(steady) and n["ledger_steady"]["p99_ms"] <= accept["ledger_p99_s"] * 1e3},
            "no_crash": {"exit_codes": final_rcs, "cell_errors": sum(cell_errors.values()),
                         "pass": all(rc in _CLEAN_EXIT for rc in final_rcs if rc != cut_rc)
                         and not cell_errors},
            "judge_lag": {"value": n["judge_lag_share"], "limit": accept["judge_lag_share"],
                          "judge_votes": sum(1 for r in requests if r["kind"] == "judge"),
                          "pass": n["judge_lag_share"] <= accept["judge_lag_share"]
                          and not run.get("stopped")},     # a cut run has not finished judging
            "complete": {"answered": n["answered_cells"], "expected": n["expected_cells"],
                         "stopped": run.get("stopped"),
                         "pass": n["answered_cells"] == n["expected_cells"] and not run.get("stopped")},
            "no_budget_trip": {"pauses": events.get("budget_paused", 0), "refusal_logs": refusals,
                               "pass": not events.get("budget_paused") and not refusals},
        }
        killed = [a for a in run["attempts"] if "killed_at" in a]
        if killed:
            checks["resume"] = cls._resume(run_dir, killed[0]["killed_at"], requests, n)
        return cls(run_dir, n, checks)

    @staticmethod
    def _backend_load(requests: list[dict], limits: dict) -> dict:
        """Per backend: requests, mean requests in flight over the run, the configured limit and
        the share of it used. A backend far below its limit while cells wait for slots is
        waiting on something other than capacity."""
        if not requests:
            return {}
        span = max(r["t1"] for r in requests) - min(r["t0"] for r in requests)
        busy, count = defaultdict(float), defaultdict(int)
        for r in requests:
            busy[r["backend"]] += r["t1"] - r["t0"]
            count[r["backend"]] += 1
        return {b: {"requests": count[b], "mean_in_flight": round(busy[b] / span, 1),
                    "limit": limits.get(b),
                    "utilization": round(busy[b] / span / limits[b], 3) if limits.get(b) else None}
                for b in sorted(busy)}

    @staticmethod
    def _resume(run_dir: Path, killed_at: float, requests: list[dict], n: dict) -> dict:
        from haenv.runtime.loadtest.rehearsal import Rehearsal  # noqa: F401 -- same layout
        before = {tuple(x) for x in json.loads((run_dir / "answered_before_kill.json").read_text())}
        root = run_dir / "root" / "results"
        after = set()
        recorded: set[str] = set()
        rebought = []
        answered_cells = {(s, c, m) for s, c, m, _ in before}
        for responses in root.glob("*/*/*-pk*/responses.jsonl"):
            sample = responses.parent
            for row in _jsonl(responses):
                if row.get("raw"):
                    import hashlib
                    after.add((sample.name, row["case"], row["solver"],
                               hashlib.sha256(str(row["raw"]).encode()).hexdigest()[:16]))
            for receipt in (sample / "solver-accounting" / "receipts").glob("*.json"):
                if receipt.name.endswith((".failed.json", ".inflight.json")):
                    continue
                doc = json.loads(receipt.read_text())
                response = doc.get("response") or {}
                gid = response.get("id") or response.get("responseId") or response.get("_generation_id")
                if gid:
                    recorded.add(gid)
                cell = doc.get("cell") or {}
                if receipt.stat().st_mtime > killed_at and \
                        (sample.name, cell.get("case_id"), cell.get("solver")) in answered_cells:
                    rebought.append(receipt.name)
            for receipt in sample.glob("semantic/*/receipts/*.json"):
                if not receipt.name.endswith((".failed.json", ".inflight.json")):
                    gid = (json.loads(receipt.read_text()).get("response") or {}).get("id")
                    if gid:
                        recorded.add(gid)
        lost = sorted(before - after)
        delivered = [r for r in requests if r.get("gen_id") and r["outcome"] in _SOLVER_OUTCOMES
                     and r["t1"] < killed_at - KILL_GRACE_S]
        unrecorded = [r["gen_id"] for r in delivered if r["gen_id"] not in recorded]
        n["resume"] = {"answered_before_kill": len(before), "lost": len(lost),
                       "receipts_after_kill_for_answered_cells": len(rebought),
                       "delivered_before_kill": len(delivered), "delivered_unrecorded": len(unrecorded)}
        return {**n["resume"], "pass": not lost and not rebought and not unrecorded}

    def text(self) -> str:
        n = self.numbers
        lines = [f"rehearsal {self.run_dir}",
                 f"  wall {n['wall_s']} s (time scale {n['time_scale']}) · cells {n['answered_cells']}/{n['expected_cells']}"
                 f" · last answer {n['last_answer_s']} s · last vote {n['last_vote_s']} s · judge lag {n['judge_lag_s']} s"
                 f" ({n['judge_lag_share']:.1%})",
                 f"  solver cells ({n['cells_probed']}): slot wait {n['slot_wait_share']} · provider {n['provider_share']}"
                 f" · local overhead {n['overhead_share']} (at real time {n['overhead_share_real']})",
                 f"  ledger p99 {n['ledger_p99_ms']} ms (thread CPU p99 {n['ledger_cpu_p99_ms']} ms) · "
                 f"first {n['ledger_burst']['s']:.0f} s: p50 {n['ledger_burst']['p50_ms']} p99 "
                 f"{n['ledger_burst']['p99_ms']} ms (n={n['ledger_burst']['n']}), after: p50 "
                 f"{n['ledger_steady']['p50_ms']} p99 {n['ledger_steady']['p99_ms']} ms "
                 f"(n={n['ledger_steady']['n']}) · " + ", ".join(
                     f"{op} n={v['n']} p50={v['p50_ms']} p99={v['p99_ms']} max={v['max_ms']}"
                     for op, v in n["ledger_ops"].items()),
                 f"  requests {n['requests']}",
                 "  backend load " + ", ".join(
                     f"{b} in flight {v['mean_in_flight']}/{v['limit']} ({v['utilization']})"
                     for b, v in n["backend_load"].items()),
                 f"  judging {n['judging']}",
                 f"  budget events {n['budget_events']}"]
        if "resume" in n:
            lines.append(f"  resume {n['resume']}")
        lines += [f"  [{'PASS' if c['pass'] else 'FAIL'}] {name}: "
                  + ", ".join(f"{k}={v}" for k, v in c.items() if k != "pass")
                  for name, c in self.checks.items()]
        return "\n".join(lines)
