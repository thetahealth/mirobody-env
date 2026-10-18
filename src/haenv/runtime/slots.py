"""Kernel-released request slots shared across processes using one budget file.

The ceiling is live: every slot acquisition re-reads it (runtime `capacity`, else the
ledger's `capacity.json`, else `config.paid`). Raising it takes effect at once;
lowering it never preempts: a request holding a slot numbered above the new ceiling
finishes, and that slot is not handed out again, so in-flight converges to the new value.

A backend with its own ceiling (`config.paid.backend_request_limits`, a runtime
`capacity.backends` entry, or a `backends` entry in `capacity.json`) draws its requests
from its own pool of that size, so one provider's queue does not hold another provider's
requests back. Every other backend shares the global pool. `judge_reserved_lanes` are kept
free of solver requests in each pool (at most `pool - 1`). Slots only bound concurrency:
each request still reserves its upper bound on the ledger before it is sent, so in-flight
reservations count against the budget whichever pool they hold.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

from haenv.runtime.scheduler import GLOBAL_REQUEST_LIMIT
from haenv.runtime.scheduler import JUDGE_RESERVED_LANES
from haenv.runtime.scheduler import SOLVER_LANES  # noqa: F401
from haenv.runtime.budget import BudgetExceeded


def persist_json(path: Path, value: dict) -> None:
    """Write `value` to `path` atomically: a temp file in the same directory, fsync, rename."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump(value, file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(file.name, path)

class CapacityMismatch(BudgetExceeded, ValueError):
    """Kept for importers; a differing configured ceiling is no longer refused."""


_HARD_MAX: int | None = None


def _hard_max() -> int:
    """`config.paid.max_capacity`, read once per process (config does not change under a
    running process)."""
    global _HARD_MAX
    if _HARD_MAX is None:
        from haenv.runtime.ops.runtime import max_capacity
        _HARD_MAX = max_capacity()
    return _HARD_MAX


def _check_limit(limit) -> None:
    hard = _hard_max()
    if type(limit) is not int or not 1 <= limit <= hard:
        raise ValueError(f"Shared request capacity must be an integer between 1 and "
                         f"{hard} (config.paid.max_capacity)")


def backend_request_limits(cfg: dict | None = None) -> dict[str, int]:
    """`config.paid.backend_request_limits`: backend -> size of its own request pool."""
    if cfg is None:
        from haenv.runtime.scheduler import _cfg
        cfg = _cfg()
    got = ((cfg or {}).get("paid") or {}).get("backend_request_limits") or {}
    if not isinstance(got, dict):
        raise ValueError("config.paid.backend_request_limits must map backends to integers")
    for name, value in got.items():
        _check_limit(value)
    return {str(k): int(v) for k, v in got.items()}


#: `config.paid.backend_request_limits`, read once per process (config does not change under
#: a running process; the live knobs are the runtime file and `capacity.json`).
_CONFIG_BACKEND_LIMITS: dict[str, int] | None = None


def _config_backend_limits() -> dict[str, int]:
    global _CONFIG_BACKEND_LIMITS
    if _CONFIG_BACKEND_LIMITS is None:
        _CONFIG_BACKEND_LIMITS = backend_request_limits()
    return _CONFIG_BACKEND_LIMITS


def _pool_file(root: Path, backend: str | None, index: int) -> Path:
    return root / (f"{index}.lock" if backend is None else f"{backend}@{index}.lock")


def _root(ledger_path: Path) -> Path:
    root = Path(ledger_path).resolve().with_suffix(Path(ledger_path).suffix + ".slots")
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_capacity(ledger_path: Path, limit: int, *, reason: str = "set_capacity",
                 backend: str | None = None) -> dict:
    """Change the shared ceiling (or `backend`'s own pool) now, also while requests are in
    flight, with an ops event. The other entries of `capacity.json` are kept."""
    _check_limit(limit)
    root = _root(ledger_path)
    with (root / "capacity.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        capacity = root / "capacity.json"
        old = json.loads(capacity.read_text()) if capacity.exists() else None
        new = dict(old or {})
        if backend is None:
            new["limit"] = limit
        else:
            new["backends"] = {**(new.get("backends") or {}), str(backend): limit}
        persist_json(capacity, new)
    from haenv.runtime.ops.runtime import ops_log
    ops_log(ledger_path, "capacity_set", old=old, new=new, reason=reason)
    return {"old": old, "new": new}


_RECORDED: dict[str, tuple] = {}


@contextmanager
def running_process(ledger_path: Path | None, rows: list[dict]):
    """Register a live scheduler using a kernel lock; exits and crashes release it.

    Snapshots include heterogeneous runs' actual per-process capacities. A stale file
    whose lock is free is never counted, even after a killed process or PID reuse.
    """
    if ledger_path is None:
        yield [{"rows": rows}]
        return
    root = _root(ledger_path) / "processes"
    root.mkdir(exist_ok=True)
    path = root / f"{uuid.uuid4().hex}.json"
    held = path.open("w+")
    with (root / "registry.lock").open("a") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        json.dump({"pid": os.getpid(), "rows": rows}, held)
        held.flush()
    try:
        yield active_processes(ledger_path)
    finally:
        with (root / "registry.lock").open("a") as guard:
            fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
            path.unlink(missing_ok=True)
            held.close()


def active_processes(ledger_path: Path) -> list[dict]:
    root = _root(ledger_path) / "processes"
    if not root.exists():
        return []
    active = []
    with (root / "registry.lock").open("a") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        for path in sorted(root.glob("*.json")):
            with path.open() as file:
                try:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    active.append(json.load(file))
    return active


_UNIT_WAIT = threading.local()


def begin_unit_wait() -> None:
    _UNIT_WAIT.seconds = 0.0


def note_slot_wait(seconds: float) -> None:
    if hasattr(_UNIT_WAIT, "seconds"):
        _UNIT_WAIT.seconds += seconds


def end_unit_wait() -> float:
    seconds = getattr(_UNIT_WAIT, "seconds", 0.0)
    if hasattr(_UNIT_WAIT, "seconds"):
        del _UNIT_WAIT.seconds
    return seconds


def _recorded_capacity(ledger_path: Path) -> dict:
    """`capacity.json` of the ledger, re-read only when its mtime or size changes."""
    capacity = _root(ledger_path) / "capacity.json"
    try:
        stat = capacity.stat()
    except FileNotFoundError:
        return {}
    seen = _RECORDED.get(str(capacity))
    if seen is not None and seen[0] == (stat.st_mtime_ns, stat.st_size):
        return seen[1]
    try:
        value = json.loads(capacity.read_text())
    except (OSError, ValueError):
        return {}
    _RECORDED[str(capacity)] = ((stat.st_mtime_ns, stat.st_size), value)
    return value


def _valid(value) -> bool:
    return type(value) is int and 1 <= value <= _hard_max()


def pool_of(ledger_path: Path, backend: str | None) -> str | None:
    """The pool a request on `backend` draws from: the backend's own, or None (global)."""
    return backend if backend is not None and _backend_cap(ledger_path, backend) else None


def _backend_cap(ledger_path: Path, backend: str) -> int | None:
    from haenv.runtime.ops.runtime import for_ledger
    cap = for_ledger(ledger_path).current()["capacity"] or {}
    for value in ((cap.get("backends") or {}).get(backend),
                  (_recorded_capacity(ledger_path).get("backends") or {}).get(backend),
                  _config_backend_limits().get(backend)):
        if _valid(value):
            return value
    return None


def current_limits(ledger_path: Path, backend: str | None = None) -> tuple[int, int]:
    """(ceiling, judge-reserved lanes) in force right now for this ledger: of `backend`'s own
    pool when it has one, else of the global pool."""
    from haenv.runtime.ops.runtime import for_ledger
    cap = for_ledger(ledger_path).current()["capacity"]
    reserved = cap["judge_reserved"] if cap is not None else JUDGE_RESERVED_LANES
    own = _backend_cap(ledger_path, backend) if backend is not None else None
    if own is not None:
        return own, min(reserved, own - 1)
    if cap is not None:
        return cap["global"], cap["judge_reserved"]
    limit = GLOBAL_REQUEST_LIMIT
    recorded = _recorded_capacity(ledger_path).get("limit")
    if _valid(recorded):
        limit = recorded
    return limit, min(JUDGE_RESERVED_LANES, limit - 1)


def requests_in_flight(ledger_path: Path) -> bool:
    """True while any process holds a request slot of this ledger (a paid request is live)."""
    root = Path(ledger_path).resolve().with_suffix(Path(ledger_path).suffix + ".slots")
    for path in root.glob("*.lock"):                  # read only: a missing directory means no slot was ever taken
        if path.name == "capacity.lock":
            continue
        with path.open("a") as file:
            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    return False


def _lanes(ledger_path: Path, pool: str | None, role: str, limit, lanes) -> range:
    """The lane indices this caller may hold now (a solver's leave the judge-reserved
    lanes free)."""
    live, reserved = current_limits(ledger_path, pool)
    usable = live if limit is None else min(live, limit)
    if role == "solver":
        usable = min(usable, live - reserved)
    if lanes is not None:
        usable = min(usable, lanes)
    if usable < 1:
        raise ValueError("A caller needs at least one usable lane")
    return range(usable)


def _take(root: Path, pool: str | None, indices):
    """A held lane's open file, or None when every lane in `indices` is taken."""
    for index in indices:
        file = _pool_file(root, pool, index).open("a")
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return file
        except BlockingIOError:
            file.close()
    return None


@contextmanager
def request_slot(ledger_path: Path, *, limit: int | None = None, lanes: int | None = None,
                 role: str = "any", backend: str | None = None):
    """Hold one shared lane for one paid request.

    `backend` picks the pool: the backend's own when it has a ceiling, else the global one.
    `role="solver"` leaves the judge-reserved lanes free. `limit`/`lanes` further cap the
    usable lanes for this caller. Callers form one queue per pool and lane set (a kernel
    lock): only the head of a queue looks for a free lane, every `config.paid.slot_poll_s`
    while all are taken; the others sleep in the kernel until it is their turn. A
    lane is a kernel lock too, released when its holder exits or dies. While waiting, a
    runtime `drain` stops the wait (`Draining`).
    """
    if limit is not None:
        _check_limit(limit)
    queued = time.monotonic()
    root = _root(ledger_path)
    from haenv.runtime.ops.runtime import check_dispatch
    pool = pool_of(ledger_path, backend)
    # Callers that may hold the same lanes queue together, so a caller allowed more lanes
    # (a judge) never waits behind one that cannot use them.
    usable = len(_lanes(ledger_path, pool, role, limit, lanes))
    queue = root / f"{pool or '_global'}.{usable}.queue"
    with queue.open("a") as turn:
        fcntl.flock(turn.fileno(), fcntl.LOCK_EX)
        try:
            poll_s, held = _slot_poll_s(), None
            while True:
                check_dispatch(ledger_path)
                held = _take(root, pool, _lanes(ledger_path, pool, role, limit, lanes))
                if held is not None:
                    break
                time.sleep(poll_s)
        finally:
            note_slot_wait(time.monotonic() - queued)
            fcntl.flock(turn.fileno(), fcntl.LOCK_UN)
    try:
        yield
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()


_POLL_S: float | None = None


_AHEAD: dict = {}


def reserve_ahead_gate(ledger_path: Path):
    """The process's bound on requests that hold a reservation but no lane yet
    (`config.paid.reserve_ahead`): a semaphore, or None when unbounded."""
    key = str(ledger_path)
    if key not in _AHEAD:
        from haenv.runtime.scheduler import _cfg
        from haenv.app.config import PaidConcurrencySettings as PaidSection
        size = int(PaidSection(**((_cfg() or {}).get("paid") or {})).reserve_ahead)
        _AHEAD[key] = threading.BoundedSemaphore(size) if size > 0 else None
    return _AHEAD[key]


def _slot_poll_s() -> float:
    """`config.paid.slot_poll_s`, read once per process."""
    global _POLL_S
    if _POLL_S is None:
        from haenv.runtime.scheduler import _cfg
        from haenv.app.config import PaidConcurrencySettings as PaidSection
        _POLL_S = float(PaidSection(**((_cfg() or {}).get("paid") or {})).slot_poll_s)
    return _POLL_S
