"""Trace event log -- append-only; everything that reaches a model request is recorded.

A tool result cannot be reconstructed after the fact (it depends on live
gatekeeper state at that `world_sha`), so requests and results are logged as
they happen, one JSONL event per line. The event shape follows dsh (DeepSeek
Harness, `packages/core/session/src/`): raw unparsed tool arguments,
`call_id`-paired `tool/call`/`tool/result`, failed attempts kept, and unknown
event types reject the whole log unless marked `ignorable`.

`trace.jsonl` is provenance only: it does not enter the judging fingerprint,
and judge code never reads it or `reasoning_text`.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import datetime as _dt
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

#: Bumped only when the event structure changes; new event types go through
#: `KNOWN_EVENT_TYPES` + `ignorable` instead.
TRACE_FORMAT_VERSION = 1

#: Event types this build recognizes; `read_trace` rejects a log containing any
#: other type not marked `ignorable`.
KNOWN_EVENT_TYPES: frozenset[str] = frozenset({
    "run/start",          # one `haenv run` invocation starts (batch-level)
    "run/end",
    "case/start",         # one cell starts; gated geometry records its menu here
                           # (including decoys and prices)
    "case/end",           # a cell's terminal state (overall / abort reason)
    "step/start",         # one model request plus the tools it invoked (dsh's
                           # "step" semantics)
    "step/end",
    "request/header",     # this step's request header (dynamic segment +
                           # fingerprint), log-only
    "assistant/message",  # the model output that was committed to context
                           # (never truncated)
    "assistant/attempt",  # a failed/retried/cancelled attempt (never truncated;
                           # never fabricates visible history)
    "tool/call",          # the model requests a tool call; `arguments` verbatim,
                           # unparsed
    "tool/result",        # the tool's return value plus billing and grounding
                           # verdict
    "gate/verdict",       # a leak-gate / hard-gate verdict (kept even when the
                           # cell is voided)
})

#: Events carrying model-visible content (the only ones the recording
#: invariant binds). Unlike dsh, `request/header` is model-visible here: the
#: gated context is appended to the prompt by `evaluate._gated_suffix`.
MODEL_VISIBLE_EVENT_TYPES: frozenset[str] = frozenset({
    "request/header", "assistant/message", "tool/result",
})

LOG_ONLY_EVENT_TYPES: frozenset[str] = KNOWN_EVENT_TYPES - MODEL_VISIBLE_EVENT_TYPES

_WRITE_LOCK = threading.Lock()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(obj) -> bool:
    """Validates JSON-serializability where the data is produced, not at write time."""
    try:
        json.dumps(obj, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False


class TraceError(RuntimeError):
    """An error in the trace itself. Never swallowed: swallowing it would make
    recording look active while the log stays empty."""


@dataclass
class TraceLog:
    case: str
    solver: str
    geometry: str
    events: list[dict] = field(default_factory=list)
    _seq: int = 0

    def append(self, type_: str, data: dict | None = None, *,
               turn: int = 0, step: int | None = None,
               ignorable: bool = False) -> dict:
        if type_ not in KNOWN_EVENT_TYPES and not ignorable:
            raise TraceError(
                f"unregistered event type {type_!r} -- either add it to KNOWN_EVENT_TYPES "
                f"with a negative-control test, or explicitly mark ignorable=True (declaring "
                f"\"the trace can still be reconstructed correctly without it\")")
        payload = dict(data or {})
        if not _json_safe(payload):
            raise TraceError(f"{type_}'s data is not JSON-serializable -- caught where it is produced, not after it hits disk")
        self._seq += 1
        ev = {"v": TRACE_FORMAT_VERSION, "seq": self._seq, "ts": _now(),
              "case": self.case, "solver": self.solver, "geometry": self.geometry,
              "turn": turn, "step": step, "type": type_, "data": payload}
        if ignorable:
            ev["ignorable"] = True
        self.events.append(ev)
        return ev


def save_trace(path: Path, log: TraceLog) -> None:
    if not log.events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in log.events)
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(blob)


def read_trace(path: Path) -> list[dict]:
    """Read and validate against the vocabulary; an unknown, non-ignorable type
    raises instead of being skipped.
    """
    out: list[dict] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        ev = json.loads(line)
        t = ev.get("type")
        if t not in KNOWN_EVENT_TYPES and not ev.get("ignorable"):
            raise TraceError(
                f"{path}:{i} has an event type {t!r} this build does not recognize and that is "
                f"not marked ignorable -- this log was likely written by a newer harness, and "
                f"skipping it would reconstruct a **wrong** trace")
        out.append(ev)
    return out


def check_invariants(events: list[dict]) -> list[str]:
    """Relational invariants: `seq` monotonic per cell, every `tool/call` paired
    with a `tool/result`, step events referring to an open step. Returns violation
    messages; empty means clean.
    """
    bad: list[str] = []
    per_cell: dict[tuple, int] = {}
    open_steps: dict[tuple, set] = {}
    pending: dict[tuple, dict] = {}
    for ev in events:
        cell = (ev.get("case"), ev.get("solver"))
        seq = int(ev.get("seq") or 0)
        if seq <= per_cell.get(cell, 0):
            bad.append(f"{cell} seq is not monotonic: {seq} appears after {per_cell.get(cell)}")
        per_cell[cell] = max(seq, per_cell.get(cell, 0))
        t = ev.get("type")
        key = (cell, ev.get("turn"), ev.get("step"))
        if t == "step/start":
            open_steps.setdefault(cell, set()).add(ev.get("step"))
        elif t == "step/end":
            if ev.get("step") not in open_steps.get(cell, set()):
                bad.append(f"{cell} step/end points at step {ev.get('step')} that was never opened")
            open_steps.get(cell, set()).discard(ev.get("step"))
        elif t == "tool/call":
            cid = (ev.get("data") or {}).get("call_id")
            if not cid:
                bad.append(f"{cell} tool/call has no call_id => cannot be paired with a result")
            else:
                pending[(cell, cid)] = ev
        elif t == "tool/result":
            cid = (ev.get("data") or {}).get("call_id")
            if (cell, cid) not in pending:
                bad.append(f"{cell} tool/result's call_id={cid!r} has no corresponding tool/call")
            else:
                pending.pop((cell, cid))
    for (cell, cid) in pending:
        bad.append(f"{cell} tool/call call_id={cid!r} has no matching tool/result "
                   f"-- the request was logged, the result wasn't")
    return bad
