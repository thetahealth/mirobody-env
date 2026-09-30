"""`haenv drain` / `haenv ops {drain,undrain,capacity,budget,runtime}` — operator commands.

Every change is written to the shared runtime file or ledger with a reason and an event
in `<ledger>.ops.jsonl`; running processes pick it up within RELOAD_S seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from . import runtime


def drain(ledger: Path, *, reason: str, wait: bool = True, timeout_s: float = 900.0,
          poll_s: float = 2.0, sleep=time.sleep, clock=time.monotonic) -> dict:
    """Stop new paid requests everywhere on this ledger; optionally wait for in-flight to end."""
    runtime.write_runtime(ledger, lambda doc: {**doc, "drain": True}, reason=reason)
    started = clock()
    held = runtime.in_flight_slots(ledger)
    while wait and held and clock() - started < timeout_s:
        sleep(poll_s)
        held = runtime.in_flight_slots(ledger)
    result = {"drain": True, "in_flight": len(held), "waited_s": round(clock() - started, 1),
              "timed_out": bool(held) and wait}
    runtime.ops_log(ledger, "drain_done" if not held else "drain_timeout", **result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="haenv ops")
    ap.add_argument("op", choices=["drain", "undrain", "capacity", "budget", "runtime"])
    ap.add_argument("value", nargs="*", help="capacity: set N · budget: set X")
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--judge-reserved", type=int, default=None)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args(argv)
    if a.op in ("drain", "undrain", "capacity", "budget") and not (a.reason or "").strip():
        ap.error("--reason is required")
    if a.op == "drain":
        out = drain(a.ledger, reason=a.reason, wait=not a.no_wait, timeout_s=a.timeout)
    elif a.op == "undrain":
        out = runtime.write_runtime(a.ledger, lambda d: {**d, "drain": False}, reason=a.reason)
    elif a.op == "capacity":
        if len(a.value) != 2 or a.value[0] != "set":
            ap.error("usage: haenv ops capacity set N --reason R")
        n = int(a.value[1])

        def update(doc):
            reserved = a.judge_reserved
            if reserved is None:
                reserved = ((doc.get("capacity") or {}).get("judge_reserved", 0))
            return {**doc, "capacity": {"global": n, "judge_reserved": reserved}}
        out = runtime.write_runtime(a.ledger, update, reason=a.reason)
    elif a.op == "budget":
        if len(a.value) != 2 or a.value[0] != "set":
            ap.error("usage: haenv ops budget set X --reason R")
        from ..semantic_budget import BudgetLedger
        out = BudgetLedger(a.ledger).set_limit(a.value[1], reason=a.reason)
        runtime.ops_log(a.ledger, "budget_set", **out)
    else:
        rt = runtime.Runtime(a.ledger)
        out = {"path": str(rt.path), "runtime_rev": rt.rev, "doc": rt.current()}
    print(json.dumps(out, ensure_ascii=False, default=str))
    return 1 if a.op == "drain" and out.get("timed_out") else 0
