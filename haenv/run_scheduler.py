"""Cell scheduling for an evaluation run.

Throughput only. Nothing here changes what a solver is shown, how it is asked, or how a
cell is scored: every cell runs the same `fn(task)` exactly once, rows are persisted as
they complete, and the returned list is in task order. Serial (`workers <= 1`), the old
per-backend pools and the per-model pools produce the same rows apart from timing fields.

Five mechanisms:

* **One pool per model.** A model gets `min(workers, backend cap, its cell count)`
  threads (`config.eval.model_workers` replaces `workers` for a named model). The models
  on one backend share a backend-level slot, so the sum of their cells in flight never
  exceeds `backend_limits[backend]`.
* **Longest expected cell first.** Each cell's expected duration is
  `median call seconds x median calls per cell` for its (model, geometry), from a timing
  table (`solver_timing.json` in the registry when present, or `HAENV_SOLVER_TIMING`). Cells are
  dequeued in that order, and the backend slot admits waiting cells by the same priority.
  A cell with no history is placed at the median of the known estimates; with no table at
  all, cells run in task order and no stall deadline is set.
* **A global ceiling on paid requests** in flight across processes (`paid.global_request_limit`,
  enforced by `paid_slots.request_slot`); `paid.judge_reserved_lanes` of it stay
  reachable by judge requests.
* **A deadline for stalled requests.** `timeout_s` bounds each socket read, not the
  request: a stream that keeps sending keep-alive frames never trips it. A
  request that has made no progress (no SSE data frame; for a non-streamed request, no
  answer) for `max(timeout_s, STALL_FACTOR x the longest successful call this model has on
  record)` is closed. That longest call is measured from the trace and so includes any time
  the request waited for a global paid lane, which makes the deadline looser, never
  tighter. A stream that keeps sending tokens is never closed; the solver's retry loop then sends a new request
  under a new request id, and the closed one is counted at its reserved upper bound.
  Every such close is recorded on the cell's row (`stall_timeouts`).
* **A wall-clock budget per model** (`config.eval.model_wall_budget_s`, off by default).
  Once a model's first cell has been running that long, the model starts no further
  cell; cells already in flight finish under the stall deadline. The model, its budget
  and its unstarted cells are recorded in `scheduler.json` (`wall_budget_stops`), and the
  board classifies those cells as timed out.

SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import logging
import os
import socket
import statistics
import threading
import time
from pathlib import Path

log = logging.getLogger("haenv.scheduler")

#: Per-backend concurrency caps used when `config.yaml:backend_limits` cannot be read.
_BACKEND_LIMITS_FALLBACK: dict[str, int] = {"relay": 6, "openai": 3, "google": 3,
                                            "openrouter": 2, "_cli": 2, "_offline": 8}
#: Cap for a backend missing from `backend_limits`.
DEFAULT_BACKEND_LIMIT = 2
#: Paid requests in flight across all processes sharing one ledger, and the lanes of it
#: kept for judge requests, when `config.paid` is not set.
DEFAULT_GLOBAL_REQUEST_LIMIT = 48
DEFAULT_JUDGE_RESERVED_LANES = 3

#: A request is stalled once it has been open this many times longer than the longest
#: successful call of the same model on record, and never before `timeout_s`.
STALL_FACTOR = 1.5

def _default_timing_path() -> Path:
    from . import data_root
    return data_root() / "registry" / "solver_timing.json"
POOLINGS = ("model", "backend")


def _cfg() -> dict:
    try:
        from .cli import load_cfg
        return load_cfg() or {}
    except Exception as e:                                   # noqa: BLE001
        log.warning("[scheduler] could not read config: %s; using the built-in defaults", e)
        return {}


def _load_backend_limits() -> dict[str, int]:
    """`config.yaml:backend_limits` (merged with `config.local.yaml`) over the fallback."""
    try:
        from .cli import load_cfg
        got = (load_cfg() or {}).get("backend_limits") or {}
        if isinstance(got, dict) and got:
            merged = dict(_BACKEND_LIMITS_FALLBACK)
            merged.update({str(k): int(v) for k, v in got.items()})
            return merged
    except Exception as e:                                   # noqa: BLE001
        log.warning("[scheduler] could not read config.yaml:backend_limits, falling back "
                    "to the default tier: %s", e)
    return dict(_BACKEND_LIMITS_FALLBACK)


def paid_limits(cfg: dict | None = None) -> tuple[int, int]:
    """(global paid-request ceiling, lanes reserved for judges) from `config.paid`."""
    paid = ((cfg if cfg is not None else _cfg()).get("paid") or {})
    limit = paid.get("global_request_limit", DEFAULT_GLOBAL_REQUEST_LIMIT)
    reserved = paid.get("judge_reserved_lanes", DEFAULT_JUDGE_RESERVED_LANES)
    if type(limit) is not int or limit < 1:
        raise ValueError("config.paid.global_request_limit must be a positive integer")
    if type(reserved) is not int or not 0 <= reserved < limit:
        raise ValueError("config.paid.judge_reserved_lanes must be an integer in [0, limit)")
    return limit, reserved


def model_workers(cfg: dict | None = None) -> dict[str, int]:
    """`config.eval.model_workers`: per-model pool sizes that replace `workers` for those
    models (for a model whose cells are much longer than the rest)."""
    got = ((cfg if cfg is not None else _cfg()).get("eval") or {}).get("model_workers") or {}
    if not isinstance(got, dict) or any(type(v) is not int or v < 1 for v in got.values()):
        raise ValueError("config.eval.model_workers must map model names to positive integers")
    return {str(k): int(v) for k, v in got.items()}


def wall_budgets(cfg: dict | None = None) -> dict[str, float]:
    """`config.eval.model_wall_budget_s`: seconds per model; `"*"` applies to every model
    not named."""
    got = ((cfg if cfg is not None else _cfg()).get("eval") or {}).get("model_wall_budget_s") or {}
    if not isinstance(got, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                        or v <= 0 for v in got.values()):
        raise ValueError("config.eval.model_wall_budget_s must map model names to positive seconds")
    return {str(k): float(v) for k, v in got.items()}


class _WallBudget:
    """Per-model clock from the first cell's start; `admit` refuses once it has run out."""

    def __init__(self, budgets: dict[str, float], clock=time.monotonic):
        self.budgets, self.clock = budgets, clock
        self.first: dict[str, float] = {}
        self.stops: dict[str, dict] = {}
        self.lock = threading.Lock()

    def admit(self, model: str) -> bool:
        budget = self.budgets.get(model, self.budgets.get("*"))
        if budget is None:
            return True
        with self.lock:
            now = self.clock()
            start = self.first.setdefault(model, now)
            if model not in self.stops and now - start < budget:
                return True
            stop = self.stops.setdefault(model, {"budget_s": budget,
                                                 "stopped_after_s": round(now - start, 3),
                                                 "not_started": 0})
            stop["not_started"] += 1
            return False


def pooling_mode(cfg: dict | None = None) -> str:
    mode = str(
        ((cfg if cfg is not None else _cfg()).get("eval") or {}).get("pooling") or "model")
    if mode not in POOLINGS:
        raise ValueError(f"config.eval.pooling must be one of {POOLINGS}, got {mode!r}")
    return mode


# ---------------------------------------------------------------- timing history

_TIMING_CACHE: dict[str, dict] = {}


def _timing_path(path: Path | None = None) -> Path:
    """`path`, else `HAENV_SOLVER_TIMING`, else `solver_timing.json` in the registry."""
    return Path(path or os.environ.get("HAENV_SOLVER_TIMING") or _default_timing_path())


def load_timing(path: Path | None = None) -> dict:
    """The per-(model, geometry) timing table; `{}` when the file is absent."""
    p = _timing_path(path)
    key = str(p)
    if key not in _TIMING_CACHE:
        try:
            _TIMING_CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            _TIMING_CACHE[key] = {}
        except (OSError, ValueError) as e:
            log.warning("[scheduler] timing history %s unreadable (%s); no LPT order and no "
                        "stall deadline this run", p, e)
            _TIMING_CACHE[key] = {}
    return _TIMING_CACHE[key]


def timing_sha16(path: Path | None = None) -> str | None:
    p = _timing_path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None


def estimate_cell_s(timing: dict, model: str, geometry: str | None) -> float | None:
    """Expected seconds for one cell: median call seconds x median calls per cell."""
    row = ((timing.get("models") or {}).get(model) or {}).get(str(geometry))
    if not row:
        return None
    return float(row["median_call_s"]) * float(row["median_calls_per_cell"])


def stall_deadline_s(model: str, timeout_s: float, timing: dict | None = None) -> float | None:
    """`max(timeout_s, STALL_FACTOR x longest successful call of this model)`; None without history."""
    t = load_timing() if timing is None else timing
    rows = ((t.get("models") or {}).get(model) or {}).values()
    longest = [float(r["max_success_call_s"]) for r in rows if r.get("max_success_call_s")]
    if not longest:
        return None
    return max(float(timeout_s), STALL_FACTOR * max(longest))


def lpt_order(items: list, estimate_of) -> list:
    """Sort by expected duration, longest first; unknown estimates sit at the median of the
    known ones; ties keep their original order."""
    est = [estimate_of(x) for x in items]
    known = [e for e in est if e is not None]
    mid = statistics.median(known) if known else 0.0
    keyed = [((e if e is not None else mid), k) for k, e in enumerate(est)]
    order = sorted(range(len(items)), key=lambda k: (-keyed[k][0], k))
    return [items[k] for k in order]


# ---------------------------------------------------------------- slots

class PrioritySlot:
    """A counting semaphore that admits waiters highest priority first (FIFO on ties).

    Records the peak number of holders, which the cap tests read.
    """

    def __init__(self, capacity: int, name: str = ""):
        if capacity < 1:
            raise ValueError("slot capacity must be >= 1")
        self.capacity, self.name = int(capacity), name
        self._cv = threading.Condition()
        self._waiting: list = []
        self._seq = itertools.count()
        self.in_use = 0
        self.peak = 0

    def acquire(self, priority: float = 0.0) -> None:
        with self._cv:
            ticket = (-float(priority), next(self._seq))
            heapq.heappush(self._waiting, ticket)
            while not (self.in_use < self.capacity and self._waiting[0] == ticket):
                self._cv.wait()
            heapq.heappop(self._waiting)
            self.in_use += 1
            self.peak = max(self.peak, self.in_use)
            self._cv.notify_all()

    def release(self) -> None:
        with self._cv:
            self.in_use -= 1
            self._cv.notify_all()


class _Gauge:
    """In-flight counter with a peak, for pools that need no admission control."""

    def __init__(self):
        self._lock = threading.Lock()
        self.in_use = 0
        self.peak = 0

    def enter(self):
        with self._lock:
            self.in_use += 1
            self.peak = max(self.peak, self.in_use)

    def leave(self):
        with self._lock:
            self.in_use -= 1


# ---------------------------------------------------------------- stall deadline

_TL = threading.local()


def _cell_stalls() -> list:
    if not hasattr(_TL, "stalls"):
        _TL.stalls = []
    return _TL.stalls


def _request_id_of(solver) -> str | None:
    acct = getattr(solver, "_accounting", None)
    if acct is None or not hasattr(acct, "cell_id"):
        return None
    return f"solver:{acct.cell_id}:{acct.sequence}"


class StallTimeout(TimeoutError):
    """A request exceeded its model's stall deadline and was closed."""


class stall_guard:
    """Close a response still open after the model's stall deadline.

    Used around `urlopen` in the solver transport:

        with stall_guard(self) as g, urllib.request.urlopen(req, timeout=...) as r:
            g.attach(r)
            ...

    When the deadline passes before the block ends, the socket is shut down (which wakes
    a blocked read), the event is recorded on the running cell, and `StallTimeout` is
    raised in place of whatever the interrupted read raised. A block that finishes before
    the deadline is untouched. No history for the model means no deadline.
    """

    def __init__(self, solver, *, deadline_s: float | None = ...):
        self.model = str(getattr(solver, "name", ""))
        self.request_id = _request_id_of(solver)
        if deadline_s is ...:
            # Only a scheduled cell's own solver, identified by the request accountant
            # bound to it for that cell (`solver_accounting.BatchAccounting.bind`). Judge,
            # generation and tooling requests on the same transport never carry one, so
            # they keep their previous behaviour even when they share a model name. A
            # stall close is counted at its reserved bound, which needs that accountant.
            own = bool(getattr(_TL, "in_cell", False)) and self.request_id is not None
            deadline_s = (stall_deadline_s(self.model, getattr(solver, "timeout", 0) or 0)
                          if own else None)
        self.deadline_s = deadline_s
        self._lock = threading.Lock()
        self._resp = None
        self._done = False
        self.fired = False
        self._timer = None
        self._t0 = 0.0
        self._progress = 0.0

    def __enter__(self):
        self._t0 = self._progress = time.monotonic()
        self._arm(self.deadline_s)
        return self

    def _arm(self, delay) -> None:
        if delay is None:
            return
        self._timer = threading.Timer(max(0.0, delay), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def note_progress(self) -> None:
        with self._lock:
            self._progress = time.monotonic()

    def attach(self, resp):
        """Watch `resp`; returns the object to read from (SSE data frames count as progress)."""
        with self._lock:
            self._resp = resp
            late = self.fired
        if late:
            _shutdown(resp)
        return _Watched(resp, self)

    def _fire(self) -> None:
        with self._lock:
            if self._done:
                return
            idle = time.monotonic() - self._progress
            if idle < self.deadline_s:           # progress since arming: wait the remainder
                remaining = self.deadline_s - idle
            else:
                remaining = None
                resp = self._resp
                if resp is not None and getattr(resp, "fp", None) is None:
                    return      # the body was already read to the end: nothing is stalled
                self.fired = True
        if remaining is not None:
            self._arm(remaining)
            return
        if resp is not None:
            _shutdown(resp)

    def __exit__(self, exc_type, exc, tb):
        with self._lock:
            self._done = True
        if self._timer is not None:
            self._timer.cancel()
        if not self.fired:
            return False
        now = time.monotonic()
        elapsed = round(now - self._t0, 2)
        event = {"model": self.model, "request_id": self.request_id,
                 "elapsed_s": elapsed, "idle_s": round(now - self._progress, 2),
                 "deadline_s": round(float(self.deadline_s), 2),
                 "interrupted": exc_type.__name__ if exc_type else None}
        _cell_stalls().append(event)
        log.warning("[scheduler] %s request %s stalled: no progress for %.0fs (deadline %.0fs, "
                    "open %.0fs); the solver retries under a new request id", self.model,
                    self.request_id, event["idle_s"], self.deadline_s, elapsed)
        raise StallTimeout(f"no progress for {event['idle_s']}s > stall deadline "
                           f"{self.deadline_s:.0f}s (request open {elapsed}s)") \
            from (exc if isinstance(exc, BaseException) else None)


class _Watched:
    """A response proxy that reports SSE data frames to its guard as progress."""

    def __init__(self, resp, guard):
        self._resp, self._guard = resp, guard

    def __getattr__(self, name):
        return getattr(self._resp, name)

    def __iter__(self):
        for line in self._resp:
            if line.startswith(b"data:"):
                self._guard.note_progress()
            yield line

    def read(self, *a, **k):
        return self._resp.read(*a, **k)


def _shutdown(resp) -> None:
    """Shut the response's socket down so a read blocked in another thread returns."""
    sock = None
    fp = getattr(resp, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is None:
        return
    try:
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except OSError:
        pass


# ---------------------------------------------------------------- run

def run(fn, tasks: list, emit, *, workers: int, key_of=None, backend_of=None,
        geometry_of=None, pooling: str | None = None, backend_limits: dict | None = None,
        timing: dict | None = None, batch_dir: Path | None = None,
        per_model: dict | None = None, wall_budget: dict | None = None,
        clock=time.monotonic) -> list[dict]:
    """Run every task once through `fn`, persist each row with `emit`, return rows in task order.

    `key_of(task)` names the model, `backend_of(model)` its backend, `geometry_of(task)`
    the geometry the cell runs on. `workers <= 1` runs serially in task order; otherwise a
    model's pool size is `per_model[model]` (default `config.eval.model_workers`) or
    `workers`, never more than its backend's cap or its cell count. A task whose model has
    spent its `wall_budget` (default `config.eval.model_wall_budget_s`) is not started and
    has no row.
    """
    stats: dict = {"started_at": time.time(), "timing_sha16": timing_sha16()}
    _refuse_changed_timing(batch_dir, stats["timing_sha16"])
    model_of = key_of or (lambda t: "_offline")
    walls = _WallBudget(wall_budgets() if wall_budget is None else dict(wall_budget), clock)
    if workers <= 1:
        rows = []
        for t in tasks:
            if not walls.admit(model_of(t)):
                continue
            _TL.stalls = []
            _TL.in_cell = True
            try:
                row = fn(t)
            finally:
                _TL.in_cell = False
            _attach_stalls(row)
            rows.append(emit(row))
        stats["wall_budget_stops"] = walls.stops
        _write_plan(batch_dir, {"mode": "serial", "cells": len(tasks)}, rows, stats)
        return rows

    from .semantic_budget import BudgetExceeded

    pooling = pooling or pooling_mode()
    limits = dict(backend_limits if backend_limits is not None else _load_backend_limits())
    timing = load_timing() if timing is None else timing
    backend_for = backend_of or (lambda m: "_offline")

    indexed = list(enumerate(tasks))
    est = {i: estimate_cell_s(timing, model_of(t), geometry_of(t) if geometry_of else None)
           for i, t in indexed}
    if pooling == "model":
        indexed = lpt_order(indexed, lambda it: est[it[0]])

    # Pools: one per model (new) or one per backend (legacy, task order, no priority).
    pools: dict[str, list] = {}
    pool_backend: dict[str, str] = {}
    for i, t in indexed:
        m = model_of(t)
        b = backend_for(m)
        key = m if pooling == "model" else b
        pools.setdefault(key, []).append((i, t))
        pool_backend[key] = b
    slots = {b: PrioritySlot(limits.get(b, DEFAULT_BACKEND_LIMIT), b)
             for b in set(pool_backend.values())}
    overrides = {}
    if pooling == "model":
        overrides = model_workers() if per_model is None else dict(per_model)
    size = {k: min(int(overrides.get(k, workers)), slots[pool_backend[k]].capacity, len(v))
            for k, v in pools.items()}
    gauges = {k: _Gauge() for k in pools}
    known = [e for e in est.values() if e is not None]
    mid = statistics.median(known) if known else 0.0

    log.info("[eval] parallel: pooled by %s %s · backend caps %s · %d cell(s) with timing "
             "history (LPT)%s", pooling,
             {k: f"{len(v)}cells/{size[k]}threads" for k, v in pools.items()},
             {b: s.capacity for b, s in slots.items()}, len(known),
             "" if pooling == "model" else " [legacy: task order]")

    out: dict[int, dict] = {}
    stopped = threading.Event()
    stops: list = []
    errors: list = []
    lock = threading.Lock()

    def worker(key: str, queue: list) -> None:
        slot = slots[pool_backend[key]]
        while not stopped.is_set():
            with lock:
                if not queue:
                    return
                i, t = queue.pop(0)
            prio = est[i] if est[i] is not None else mid
            slot.acquire(prio if pooling == "model" else 0.0)
            try:
                if stopped.is_set():
                    return
                if not walls.admit(model_of(t)):
                    continue
                gauges[key].enter()
                _TL.stalls = []
                _TL.in_cell = True
                try:
                    row = fn(t)
                finally:
                    _TL.in_cell = False
                    gauges[key].leave()
                _attach_stalls(row)
                # Persisting is part of the cell: a row that cannot be written stops the
                # run like any other failure, before another pool starts a paid cell.
                out[i] = emit(row)
            except BudgetExceeded as exc:
                stopped.set()
                with lock:
                    stops.append(exc)
                return
            except BaseException as exc:                     # noqa: BLE001
                stopped.set()
                with lock:
                    errors.append(exc)
                return
            finally:
                slot.release()

    threads = []
    for key, items in pools.items():
        queue = list(items)
        for _ in range(size[key]):
            th = threading.Thread(target=worker, args=(key, queue), daemon=True,
                                  name=f"cell-{key}")
            th.start()
            threads.append(th)
    for th in threads:
        th.join()

    stats.update({"peak_backend": {b: s.peak for b, s in slots.items()},
                  "peak_pool": {k: g.peak for k, g in gauges.items()},
                  "wall_budget_stops": walls.stops})
    _write_plan(batch_dir, {"mode": "parallel", "pooling": pooling, "workers": int(workers),
                            "model_workers": overrides,
                            "backend_caps": {b: s.capacity for b, s in slots.items()},
                            "pools": {k: {"cells": len(v), "threads": size[k],
                                          "backend": pool_backend[k]} for k, v in pools.items()},
                            "cells_with_history": len(known)},
                [out[i] for i in sorted(out)], stats)
    if errors:
        raise errors[0]
    if stops:
        raise stops[0]
    return [out[i] for i in sorted(out)]


def _attach_stalls(row: dict) -> None:
    stalls = list(getattr(_TL, "stalls", []) or [])
    _TL.stalls = []
    if stalls and isinstance(row, dict):
        row["stall_timeouts"] = stalls


def stall_counts(rows: list[dict]) -> dict[str, int]:
    """Stall-closed requests per model, from the rows."""
    n: dict[str, int] = {}
    for r in rows:
        for ev in r.get("stall_timeouts") or []:
            m = str(ev.get("model") or r.get("solver"))
            n[m] = n.get(m, 0) + 1
    return n


def stall_report_lines(rows: list[dict]) -> list[str]:
    """Report section: stall-closed requests per model (rows' `stall_timeouts`)."""
    counts = stall_counts(rows)
    models = sorted({str(r.get("solver")) for r in rows if r.get("solver") is not None})
    L = ["### 1s. Stalled requests closed by the scheduler\n",
         f"> Rule: a request still open after `max(timeout_s, {STALL_FACTOR} x` the longest "
         "successful call of that model on record) is closed and retried under a new request "
         "id; the closed request is counted at its reserved upper bound. Each event is on the "
         "cell's row (`stall_timeouts`).\n"]
    if not counts:
        L.append(f"> **0** requests closed as stalled in this batch ({len(models)} model(s)).\n")
        return L
    L.append("| Model | Stalled requests | Cells affected |")
    L.append("|---|---|---|")
    for m in sorted(counts, key=lambda k: (-counts[k], k)):
        n_cells = sum(1 for r in rows if str(r.get("solver")) == m and r.get("stall_timeouts"))
        L.append(f"| `{m}` | {counts[m]} | {n_cells} |")
    L.append("")
    return L


class TimingChanged(RuntimeError):
    """A resumed batch would run under a different timing table than its earlier runs."""


def _refuse_changed_timing(batch_dir, sha: str | None) -> None:
    """The timing table sets stall deadlines, so it decides which requests get closed.
    Cells of one batch must share it: resuming under another table is refused."""
    if batch_dir is None:
        return
    p = Path(batch_dir) / "scheduler.json"
    if not p.is_file():
        return
    try:
        runs = json.loads(p.read_text(encoding="utf-8")).get("runs", [])
    except (OSError, ValueError):
        return
    seen = {r.get("timing_sha16") for r in runs if isinstance(r, dict) and "timing_sha16" in r}
    if seen and seen != {sha}:
        raise TimingChanged(
            f"batch {batch_dir} ran under timing table(s) {sorted(map(str, seen))}, this run "
            f"would use {sha}: stall deadlines would differ within one batch. Start a new "
            f"batch (--fresh), or point HAENV_SOLVER_TIMING at the table the batch used.")


def _write_plan(batch_dir, plan: dict, rows: list, stats: dict) -> None:
    """`<batch>/scheduler.json`: how this run was scheduled (appended per run, never read
    back by scoring)."""
    if batch_dir is None:
        return
    try:
        from .paid_slots import GLOBAL_REQUEST_LIMIT, SOLVER_LANES
        deadlines = {}
        timing = load_timing()
        timeout_s = float(_cfg().get("timeout_s", 900))
        for r in rows:
            m = str(r.get("solver"))
            if m not in deadlines:
                deadlines[m] = stall_deadline_s(m, timeout_s, timing)
        entry = {**plan, **stats, "finished_at": time.time(),
                 "global_request_limit": GLOBAL_REQUEST_LIMIT, "solver_lanes": SOLVER_LANES,
                 "stall_factor": STALL_FACTOR, "stall_deadline_s": deadlines,
                 "stall_timeouts": stall_counts(rows)}
        p = Path(batch_dir) / "scheduler.json"
        runs = json.loads(p.read_text(encoding="utf-8")).get("runs", []) if p.is_file() else []
        runs.append(entry)
        p.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:                                   # noqa: BLE001
        log.warning("[scheduler] could not write scheduler.json: %s", e)
