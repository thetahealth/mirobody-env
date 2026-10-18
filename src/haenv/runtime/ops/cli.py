"""`haenv drain` / `haenv ops {drain,undrain,capacity,budget,runtime}` — operator commands.

Every change is written to the shared runtime file or ledger with a reason and an event
in `<ledger>.ops.jsonl`; running processes pick it up within RELOAD_S seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from haenv.runtime.ops import runtime


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
    ap.add_argument("op", choices=["stop", "drain", "undrain", "capacity", "budget", "runtime"])
    ap.add_argument("value", nargs="*", help="capacity: set N · budget: set X")
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--judge-reserved", type=int, default=None)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--wait", type=float, default=900.0, help="stop: maximum seconds to wait for settlement and exit")
    ap.add_argument("--counted-backends", default=None, help="budget: comma-separated backends; all includes every backend")
    a = ap.parse_args(argv)
    if a.op in ("drain", "undrain", "capacity", "budget") and not (a.reason or "").strip():
        ap.error("--reason is required")
    if a.wait < 0:
        ap.error('--wait must be nonnegative')
    if a.op == "stop":
        out = stop(a.ledger, wait_s=a.wait, reason=a.reason or "operator requested graceful stop")
    elif a.op == "drain":
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
        if not a.value and a.counted_backends is not None:
            from haenv.runtime.budget import BudgetLedger
            out = BudgetLedger(a.ledger).set_counted_backends(
                None if a.counted_backends == 'all' else [b for b in a.counted_backends.split(',') if b], reason=a.reason)
            runtime.ops_log(a.ledger, "budget_scope_set", **out)
        elif len(a.value) != 2 or a.value[0] != "set":
            ap.error("usage: haenv ops budget set X --reason R")
        else:
            from haenv.runtime.budget import BudgetLedger
            scope = None if a.counted_backends == 'all' else [b for b in (a.counted_backends or '').split(',') if b]
            ledger = BudgetLedger(a.ledger, limit_usd=a.value[1])
            out = ledger.set_limit(a.value[1], reason=a.reason)
            if a.counted_backends is not None:
                out['scope_event'] = ledger.set_counted_backends(scope, reason=a.reason)
            runtime.ops_log(a.ledger, "budget_set", **out)
    else:
        rt = runtime.Runtime(a.ledger)
        out = {"path": str(rt.path), "runtime_rev": rt.rev, "doc": rt.current()}
    print(json.dumps(out, ensure_ascii=False, default=str))
    return 1 if a.op in ("stop", "drain") and out.get("timed_out") else 0


def stop(ledger: Path, *, wait_s=900.0, reason="operator requested graceful stop",
         poll_s=.1, sleep=time.sleep, clock=time.monotonic) -> dict:
    """Request a shared drain and wait through settlement, including time after lane release.

    A timeout returns the remaining holds; it never kills an in-flight request.
    """
    from haenv.runtime.slots import active_processes
    from haenv.runtime.budget import read_state
    if read_state(ledger) is None:
        raise ValueError("Ledger does not exist")
    runtime.write_runtime(ledger, lambda d: {**d, 'drain': True}, reason=reason)
    started = clock()
    while True:
        state = read_state(ledger)
        held = runtime.in_flight_slots(ledger)
        processes = active_processes(ledger)
        reserved = [rid for rid, r in state['requests'].items() if r['status'] == 'reserved']
        pending = bool(held or processes or reserved)
        if not pending or clock() - started >= wait_s:
            break
        sleep(min(poll_s, max(0, wait_s - (clock() - started))))
    result = {'stop': True, 'in_flight': len(held), 'processes': len(processes),
              'unsettled_requests': reserved, 'timed_out': pending,
              'waited_s': round(clock() - started, 3)}
    runtime.ops_log(ledger, 'stop_timeout' if pending else 'stop_done', **result)
    return result
