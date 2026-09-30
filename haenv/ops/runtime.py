"""Hot-reloadable runtime knobs that live outside the repository, next to the shared ledger.

`<ledger>.runtime.yaml` (or `$HAENV_RUNTIME`) is shared state like the ledger and the
request slots. Every process re-reads it at most every `RELOAD_S` seconds, by mtime, and
validates the whole document: an invalid document keeps the previous valid one and logs a
`runtime_rejected` event; nothing is half applied. Each applied version is hashed
(`runtime_rev`) and logged to `<ledger>.ops.jsonl`. The file is never part of a batch
identity: it changes how requests are scheduled and routed, never what is asked.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

from ..semantic_budget import BudgetExceeded

RELOAD_S = 5.0
#: Hard ceiling for any configured capacity (`config.paid.max_capacity` overrides).
DEFAULT_MAX_CAPACITY = 256
TOP_LEVEL = {"capacity", "lanes", "budget", "keys", "routes", "backoff", "drain"}
BACKOFF_KEYS = {"max_retries", "base_s", "max_delay_s", "max_total_s"}
FALLBACK_KEYS = {"quota_exhausted", "auth_failed", "consecutive_5xx", "balance_guard"}
BUDGET_KEYS = {"pause_poll_s", "pause_timeout_s", "note"}
DEFAULT_FALLBACK = {"quota_exhausted": True, "auth_failed": True, "consecutive_5xx": 5,
                    "balance_guard": True}


class RuntimeRejected(ValueError):
    """The runtime document is invalid; the previous valid version stays in force."""


class Draining(BudgetExceeded):
    """`drain: true`: no new paid request may start; in-flight requests finish normally."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def max_capacity() -> int:
    try:
        from ..run_scheduler import _cfg
        value = ((_cfg() or {}).get("paid") or {}).get("max_capacity", DEFAULT_MAX_CAPACITY)
    except Exception:  # noqa: BLE001 -- no config: the default hard ceiling
        value = DEFAULT_MAX_CAPACITY
    return value if type(value) is int and value >= 1 else DEFAULT_MAX_CAPACITY


def _int(value, lo, hi, what):
    if type(value) is not int or not lo <= value <= hi:
        raise RuntimeRejected(f"{what} must be an integer in [{lo}, {hi}]")
    return value


def _num(value, what):
    if type(value) not in (int, float) or value < 0:
        raise RuntimeRejected(f"{what} must be a nonnegative number")
    return value


def validate(doc) -> dict:
    """The whole document or nothing; returns it normalized (defaults filled)."""
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise RuntimeRejected("runtime document must be a mapping")
    unknown = set(doc) - TOP_LEVEL
    if unknown:
        raise RuntimeRejected(f"unknown runtime keys: {sorted(unknown)}")
    out = {"capacity": None, "lanes": {}, "budget": {}, "keys": {}, "routes": {},
           "fallback_on": dict(DEFAULT_FALLBACK), "backoff": {}, "drain": False}
    hard = max_capacity()
    cap = doc.get("capacity")
    if cap is not None:
        if not isinstance(cap, dict) or set(cap) - {"global", "judge_reserved"} or "global" not in cap:
            raise RuntimeRejected("capacity must be {global: N, judge_reserved: M}")
        g = _int(cap["global"], 1, hard, "capacity.global")
        r = _int(cap.get("judge_reserved", 0), 0, g - 1, "capacity.judge_reserved")
        out["capacity"] = {"global": g, "judge_reserved": r}
    lanes = doc.get("lanes") or {}
    if not isinstance(lanes, dict):
        raise RuntimeRejected("lanes must be a mapping")
    for name, lane in lanes.items():
        if not isinstance(lane, dict) or set(lane) != {"max"}:
            raise RuntimeRejected(f"lanes.{name} must be {{max: N}}")
        out["lanes"][str(name)] = {"max": _int(lane["max"], 0, hard, f"lanes.{name}.max")}
    budget = doc.get("budget") or {}
    if not isinstance(budget, dict) or set(budget) - BUDGET_KEYS:
        raise RuntimeRejected("budget holds only pause_poll_s / pause_timeout_s / note "
                              "(the cap itself lives on the ledger)")
    for k in ("pause_poll_s", "pause_timeout_s"):
        if k in budget:
            out["budget"][k] = _num(budget[k], f"budget.{k}")
    keys = doc.get("keys") or {}
    if not isinstance(keys, dict):
        raise RuntimeRejected("keys must be a mapping")
    for backend, spec in keys.items():
        if not isinstance(spec, dict) or set(spec) - {"disabled"}:
            raise RuntimeRejected(f"keys.{backend} holds only `disabled` (key ordinals)")
        disabled = spec.get("disabled") or []
        if not isinstance(disabled, list) or any(type(i) is not int for i in disabled):
            raise RuntimeRejected(f"keys.{backend}.disabled must list integer key ordinals")
        out["keys"][str(backend)] = {"disabled": sorted(set(disabled))}
    routes = doc.get("routes") or {}
    if not isinstance(routes, dict):
        raise RuntimeRejected("routes must be a mapping")
    for model, chain in routes.items():
        if model == "fallback_on":
            if not isinstance(chain, dict) or set(chain) - FALLBACK_KEYS:
                raise RuntimeRejected("routes.fallback_on keys: " + ", ".join(sorted(FALLBACK_KEYS)))
            for k, v in chain.items():
                if k == "consecutive_5xx":
                    _int(v, 0, 10 ** 6, "routes.fallback_on.consecutive_5xx")
                elif type(v) is not bool:
                    raise RuntimeRejected(f"routes.fallback_on.{k} must be true/false")
            out["fallback_on"].update(chain)
            continue
        if not isinstance(chain, list) or not chain or any(
                not isinstance(r, dict) or not isinstance(r.get("backend"), str) for r in chain):
            raise RuntimeRejected(f"routes.{model} must be a nonempty list of {{backend: ...}}")
        out["routes"][str(model)] = [dict(r) for r in chain]
    backoff = doc.get("backoff") or {}
    if not isinstance(backoff, dict) or set(backoff) - BACKOFF_KEYS:
        raise RuntimeRejected("backoff keys: " + ", ".join(sorted(BACKOFF_KEYS)))
    out["backoff"] = {k: _num(v, f"backoff.{k}") for k, v in backoff.items()}
    drain = doc.get("drain", False)
    if type(drain) is not bool:
        raise RuntimeRejected("drain must be true/false")
    out["drain"] = drain
    return out


def rev_of(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def runtime_path(ledger_path) -> Path:
    env = os.environ.get("HAENV_RUNTIME")
    if env:
        return Path(env)
    ledger = Path(ledger_path).resolve()
    return ledger.with_name(ledger.name + ".runtime.yaml")


def ops_path(ledger_path) -> Path:
    ledger = Path(ledger_path).resolve()
    return ledger.with_name(ledger.name + ".ops.jsonl")


def ops_log(ledger_path, event: str, **fields) -> dict:
    """Append one operator/infra event (never secrets) to `<ledger>.ops.jsonl`."""
    record = {"at": _now(), "event": event, "pid": os.getpid(), **fields}
    path = ops_path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
    with path.open("ab") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.write(line)
        file.flush()
        os.fsync(file.fileno())
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    return record


def read_ops(ledger_path) -> list[dict]:
    path = ops_path(ledger_path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


class Runtime:
    """The valid runtime document for one ledger, re-read by mtime at most every RELOAD_S."""

    def __init__(self, ledger_path, *, path: Path | None = None, clock=time.monotonic,
                 reload_s: float = RELOAD_S):
        self.ledger_path = Path(ledger_path)
        self.path = Path(path) if path is not None else runtime_path(ledger_path)
        self.clock, self.reload_s = clock, reload_s
        self._lock = threading.Lock()
        self._doc, self._rev, self._mtime, self._checked = validate({}), rev_of(validate({})), None, None

    def _load(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self._mtime is not None:
                self._doc, self._mtime = validate({}), None
                self._rev = rev_of(self._doc)
                ops_log(self.ledger_path, "runtime_applied", runtime_rev=self._rev, path=str(self.path),
                        note="runtime file removed; defaults")
            return
        mtime = (stat.st_mtime_ns, stat.st_size)
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            import yaml
            doc = validate(yaml.safe_load(self.path.read_text(encoding="utf-8")))
        except Exception as error:  # noqa: BLE001 -- keep the previous valid document
            ops_log(self.ledger_path, "runtime_rejected", path=str(self.path),
                    error=f"{type(error).__name__}: {error}"[:300], kept_rev=self._rev)
            return
        rev = rev_of(doc)
        if rev != self._rev:
            self._doc, self._rev = doc, rev
            ops_log(self.ledger_path, "runtime_applied", runtime_rev=rev, path=str(self.path))

    def current(self) -> dict:
        with self._lock:
            now = self.clock()
            if self._checked is None or now - self._checked >= self.reload_s:
                self._checked = now
                self._load()
            return self._doc

    @property
    def rev(self) -> str:
        self.current()
        return self._rev


_RUNTIMES: dict[str, Runtime] = {}
_RUNTIMES_LOCK = threading.Lock()


def for_ledger(ledger_path) -> Runtime:
    key = str(runtime_path(ledger_path)) + "|" + str(Path(ledger_path).resolve())
    with _RUNTIMES_LOCK:
        if key not in _RUNTIMES:
            _RUNTIMES[key] = Runtime(ledger_path)
        return _RUNTIMES[key]


def check_dispatch(ledger_path) -> None:
    """Raise `Draining` when no new paid request may start."""
    if for_ledger(ledger_path).current()["drain"]:
        raise Draining("Draining (runtime `drain: true`): no new paid request starts; "
                       "in-flight requests finish and the run exits")


def write_runtime(ledger_path, update, *, reason: str) -> dict:
    """Apply `update(doc) -> doc` to the runtime file atomically, validated, with an ops event."""
    import yaml
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("An explicit --reason is required")
    path = runtime_path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        old = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        new = update(dict(old or {}))
        validate(new)                                   # refuse before writing anything
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
            yaml.safe_dump(new, f, sort_keys=True, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(f.name, path)
    ops_log(ledger_path, "runtime_written", reason=reason, runtime_rev=rev_of(validate(new)),
            old=old, new=new)
    return new


def in_flight_slots(ledger_path) -> list[str]:
    """Slot lock files currently held by some process (a paid request in flight)."""
    ledger = Path(ledger_path).resolve()
    root = ledger.with_suffix(ledger.suffix + ".slots")
    held = []
    for path in sorted(root.glob("[0-9]*.lock")) if root.is_dir() else []:
        with path.open("a") as file:
            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(path.name)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    return held
