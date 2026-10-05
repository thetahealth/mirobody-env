"""Kernel-released request slots shared across processes using one budget file.

The ceiling is live: every slot acquisition re-reads it (runtime `capacity`, else the
ledger's `capacity.json`, else `config.paid`). Raising it takes effect at once;
lowering it never preempts: a request holding a slot numbered above the new ceiling
finishes, and that slot is not handed out again, so in-flight converges to the new value.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

from .run_scheduler import GLOBAL_REQUEST_LIMIT, JUDGE_RESERVED_LANES, SOLVER_LANES  # noqa: F401
from .semantic_budget import BudgetExceeded


def persist_json(path: Path, value: dict) -> None:
    """Write `value` to `path` atomically: a temp file in the same directory, fsync, rename."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump(value, file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(file.name, path)

class CapacityMismatch(BudgetExceeded, ValueError):
    """Kept for importers; a differing configured ceiling is no longer refused."""


def _hard_max() -> int:
    from .ops.runtime import max_capacity
    return max_capacity()


def _check_limit(limit) -> None:
    hard = _hard_max()
    if type(limit) is not int or not 1 <= limit <= hard:
        raise ValueError(f"Shared request capacity must be an integer between 1 and "
                         f"{hard} (config.paid.max_capacity)")


def _root(ledger_path: Path) -> Path:
    root = Path(ledger_path).resolve().with_suffix(Path(ledger_path).suffix + ".slots")
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_capacity(ledger_path: Path, limit: int, *, reason: str = "set_capacity") -> dict:
    """Change the shared ceiling now (also while requests are in flight), with an ops event."""
    _check_limit(limit)
    root = _root(ledger_path)
    with (root / "capacity.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        capacity = root / "capacity.json"
        old = json.loads(capacity.read_text()) if capacity.exists() else None
        persist_json(capacity, {"limit": limit})
    from .ops.runtime import ops_log
    ops_log(ledger_path, "capacity_set", old=old, new={"limit": limit}, reason=reason)
    return {"old": old, "new": {"limit": limit}}


def current_limits(ledger_path: Path) -> tuple[int, int]:
    """(global ceiling, judge-reserved lanes) in force right now for this ledger."""
    from .ops.runtime import for_ledger
    cap = for_ledger(ledger_path).current()["capacity"]
    if cap is not None:
        return cap["global"], cap["judge_reserved"]
    capacity = _root(ledger_path) / "capacity.json"
    limit = GLOBAL_REQUEST_LIMIT
    if capacity.exists():
        try:
            recorded = json.loads(capacity.read_text()).get("limit")
            if type(recorded) is int and 1 <= recorded <= _hard_max():
                limit = recorded
        except (OSError, ValueError):
            pass
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


@contextmanager
def request_slot(ledger_path: Path, *, limit: int | None = None, lanes: int | None = None,
                 role: str = "any", poll_s: float = .05):
    """Hold one shared lane for one paid request.

    `role="solver"` leaves the judge-reserved lanes free. `limit`/`lanes` further cap the
    usable lanes for this caller. While waiting, a runtime `drain` stops the wait (`Draining`).
    """
    if limit is not None:
        _check_limit(limit)
    root = _root(ledger_path)
    from .ops.runtime import check_dispatch
    while True:
        check_dispatch(ledger_path)
        with (root / "capacity.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            live, reserved = current_limits(ledger_path)
        usable = live if limit is None else min(live, limit)
        if role == "solver":
            usable = min(usable, live - reserved)
        if lanes is not None:
            usable = min(usable, lanes)
        if usable < 1:
            raise ValueError("A caller needs at least one usable lane")
        for index in range(usable):
            file = (root / f"{index}.lock").open("a")
            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                file.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                file.close()
            return
        time.sleep(poll_s)
