"""Persistent, process-safe dollar cap for paid requests.

A ledger named `budget.json` keeps its records in `budget.json.sqlite` (SQLite, WAL mode):
one row per request, and a `meta` table holding the cap, the stop reason, the operator
events and the running totals. Every change is one `BEGIN IMMEDIATE` transaction that
reads the totals, checks the cap, writes the row and the new totals together, so
concurrent processes never lose an update and a process killed mid-transaction leaves the
ledger as it was before that transaction.

A JSON ledger at the ledger path (a snapshot line followed by change lines) is imported
into the database the first time the ledger is opened; the JSON file is only read, and
`read_state` reads either format, so archived batches stay readable.

Reservations: every request reserves before it is sent. A request names its cost key
(`cost_key`: role, backend, model, request geometry); once that key has
`config.budget.reservation_min_observations` settled requests, the request reserves the
`reservation_quantile` of their actual cost instead of its worst case. Admission keeps
`overrun_slots` x the largest (worst case - reservation) of the in-flight requests free
under the cap, and a request that costs more than its reservation (never more than its
worst case) settles at its actual cost. Unknown costs are counted at the worst case.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import threading
import time

log = logging.getLogger("haenv.budget")

# A halt reason some existing ledgers carry; it stops nothing (unknown costs count at their bound).
LEGACY_UNKNOWN_HALT = "A billed request has unknown cost; reconciliation required"
BOUND_VIOLATION = "Provider charge exceeded the reserved upper bound"


class BudgetExceeded(RuntimeError):
    """No further billable requests may be launched."""


class BudgetCapReached(BudgetExceeded):
    """This reservation would exceed the ledger's cap; a later raise can admit it."""


def _amount(value, *, positive=False) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Invalid dollar amount") from exc
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ValueError("Dollar amount must be finite and nonnegative")
    return number


def cost_key(role: str, backend: str | None, model: str | None, geometry: str = "") -> str:
    """The key whose settled history sizes a request's reservation."""
    return "|".join((role, backend or "", model or "", geometry))


def input_geometry(max_tokens: int, input_chars: int) -> str:
    """Request geometry for `cost_key`: the output cap and the input size rounded up to a
    power of two, so requests of one pack and model share a key."""
    return f"out{max_tokens}:in{1 << max(0, math.ceil(math.log2(max(1, input_chars))))}"


def budget_settings() -> dict:
    """`config.budget` with the schema's defaults (`haenv.settings.BudgetSection`)."""
    from haenv.app.config import RuntimeBudgetSettings
    try:
        from haenv.app.config import load_cfg
        got = (load_cfg() or {}).get("budget") or {}
    except Exception as e:  # noqa: BLE001 -- no readable config: the declared defaults
        log.debug("budget settings: config unreadable (%s); using defaults", e)
        got = {}
    return RuntimeBudgetSettings(**got).model_dump()


# ---- record arithmetic -------------------------------------------------------------------

#: committed, settled, unknown_capped n, unknown_capped $, tariff_capped n, tariff $, in flight $
_COUNTS = (2, 4)
_N_TOTALS = 7


def _contribution(record) -> tuple:
    reserved = _amount(record["reserved_usd"])
    actual = _amount(record["actual_usd"]) if record.get("actual_usd") is not None else None
    status = record["status"]
    zero = Decimal("0")
    return (actual if actual is not None else reserved, actual if actual is not None else zero,
            int(status == "unknown_capped"), reserved if status == "unknown_capped" else zero,
            int(status == "tariff_capped"), reserved if status == "tariff_capped" else zero,
            reserved if status == "reserved" else zero)


def _zero_totals() -> list:
    return [0 if i in _COUNTS else Decimal("0") for i in range(_N_TOTALS)]


def _backend(record):
    return record.get("backend") or (record.get("tags") or {}).get("backend") or "unattributed"


def _included(backend, scope):
    # Legacy requests with no provider remain counted conservatively.
    return scope is None or backend == "unattributed" or backend in scope


def _parse_totals(values):
    return [int(v) if i in _COUNTS else Decimal(v) for i, v in enumerate(values)]


def _scope_totals(backends, scope):
    totals = _zero_totals()
    for backend, values in backends.items():
        if _included(backend, scope):
            for i, value in enumerate(_parse_totals(values)):
                totals[i] += value
    return totals


def _backend_totals(con):
    totals = {}
    for raw, backend in con.execute("SELECT record, backend FROM requests"):
        record = json.loads(raw)
        row = totals.setdefault(backend or _backend(record), _zero_totals())
        for i, value in enumerate(_contribution(record)):
            row[i] += value
    return {k: [str(v) for v in values] for k, values in totals.items()}


def _sum_totals(records) -> list:
    totals = _zero_totals()
    for record in records:
        for i, value in enumerate(_contribution(record)):
            totals[i] += value
    return totals


def _bound(record) -> Decimal:
    """The request's worst case: what it may cost at most."""
    return _amount(record.get("bound_usd", record["reserved_usd"]))


def _count_at_bound(record) -> None:
    """Count a request at its worst case (its cost is unknown)."""
    if "bound_usd" in record:
        record["held_usd"] = record["reserved_usd"]
        record["reserved_usd"] = record.pop("bound_usd")


def _summary(meta: dict, totals: list) -> dict:
    result = {**{k: meta[k] for k in ("limit_usd", "operator_events") if k in meta},
            "halt_reason": meta.get("halt_reason"), "committed_usd": str(totals[0]),
            "settled_usd": str(totals[1]), "unknown_capped_n": totals[2],
            "unknown_capped_usd": str(totals[3]), "tariff_capped_n": totals[4],
            "tariff_bound_usd": str(totals[5]), "in_flight_usd": str(totals[6])}
    backends = meta.get("backend_totals", {})
    result["counted_backends"] = meta.get("counted_backends")
    result["by_backend"] = {b: {"committed_usd": str(v[0]), "settled_usd": str(v[1]),
                                "in_flight_usd": str(v[6]), "counted": _included(b, meta.get("counted_backends"))}
                            for b, v in backends.items()}
    counted = _scope_totals(backends, meta.get("counted_backends")) if backends else totals
    result["counted_usd"] = str(counted[0])
    return result


# ---- storage -----------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY, status TEXT NOT NULL, reserved_usd TEXT NOT NULL, actual_usd TEXT,
    actual_num REAL, exposure REAL NOT NULL DEFAULT 0, cost_key TEXT, role TEXT, backend TEXT,
    model TEXT, generation_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
    record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS requests_in_flight ON requests(exposure) WHERE status = 'reserved';
CREATE INDEX IF NOT EXISTS requests_by_cost ON requests(cost_key, actual_num)
    WHERE status = 'settled';
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


def store_of(path: Path) -> Path:
    """The database file of the ledger named `path` (`path` itself when it is one)."""
    path = Path(path)
    return path if path.suffix == ".sqlite" else path.with_name(path.name + ".sqlite")


def _row(rid: str, record: dict, now: float, key: str | None = None) -> tuple:
    role, backend, model = (key.split("|") + ["", "", ""])[:3] if key else (None, None, None)
    tags = record.get("tags") or {}
    role, backend, model = (role or record.get("role") or tags.get("kind"),
                            backend or record.get("backend") or tags.get("backend"),
                            model or record.get("model") or tags.get("model"))
    actual = record.get("actual_usd")
    exposure = (_bound(record) - _amount(record["reserved_usd"])
                if record["status"] == "reserved" else Decimal("0"))
    return (rid, record["status"], record["reserved_usd"], actual,
            float(actual) if actual is not None else None, float(exposure), key, role, backend,
            model, record.get("generation_id"), now, now,
            json.dumps(record, sort_keys=True, separators=(",", ":")))


_UPSERT = ("INSERT INTO requests (id, status, reserved_usd, actual_usd, actual_num, exposure, "
           "cost_key, role, backend, model, generation_id, created_at, updated_at, record) "
           "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
           "status=excluded.status, reserved_usd=excluded.reserved_usd, "
           "actual_usd=excluded.actual_usd, actual_num=excluded.actual_num, "
           "exposure=excluded.exposure, cost_key=COALESCE(excluded.cost_key, cost_key), "
           "role=COALESCE(excluded.role, role), backend=COALESCE(excluded.backend, backend), "
           "model=COALESCE(excluded.model, model), "
           "generation_id=COALESCE(excluded.generation_id, generation_id), "
           "updated_at=excluded.updated_at, record=excluded.record")


def _create(store: Path, state: dict, provenance: dict | None) -> None:
    """Write a complete database for `state` next to `store`, then move it into place."""
    temporary = store.with_name(f"{store.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    con = sqlite3.connect(temporary, isolation_level=None)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(_SCHEMA)
        now = time.time()
        records = state.get("requests") or {}
        con.execute("BEGIN")
        con.executemany(_UPSERT, (_row(rid, rec, now) for rid, rec in records.items()))
        meta = {"limit_usd": state["limit_usd"], "halt_reason": state.get("halt_reason"),
                "totals": [str(t) for t in _sum_totals(records.values())]}
        meta["counted_backends"] = state.get("counted_backends")
        meta["backend_totals"] = _backend_totals(con)
        if state.get("operator_events"):
            meta["operator_events"] = state["operator_events"]
        if provenance is not None:
            meta["imported_from"] = provenance
        con.executemany("INSERT INTO meta (k, v) VALUES (?, ?)",
                        [(k, json.dumps(v)) for k, v in meta.items()])
        con.execute("COMMIT")
    finally:
        con.close()
    os.replace(temporary, store)


def _json_provenance(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# ---- JSON ledgers (archived batches; imported once) --------------------------------------

def _complete_lines(data: bytes) -> list[bytes]:
    """Lines that end in a newline (a torn last line of an interrupted append is left out)."""
    end = data.rfind(b"\n") + 1
    return [line for line in data[:end].split(b"\n") if line.strip()]


def read_json_state(path: Path) -> dict:
    """A JSON ledger's state: its first JSON value (the snapshot) with the change lines
    (`{"r": id, "v": record}` or `{"top": {...}}`) after it replayed in order."""
    data = Path(path).read_bytes()
    text = data.decode("utf-8")
    state, end = json.JSONDecoder().raw_decode(text)
    state.pop("_snapshot", None)
    state.setdefault("requests", {})
    start = len(text[:end].encode("utf-8"))
    newline = data.find(b"\n", start)
    for line in _complete_lines(data[newline + 1:]) if newline >= 0 else []:
        entry = json.loads(line)
        if "r" in entry:
            state["requests"][entry["r"]] = entry["v"]
        else:
            state.update(entry["top"])
    return state


def _read_store(store: Path) -> dict:
    try:
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True, isolation_level=None)
        con.execute("SELECT 1 FROM meta LIMIT 1")
    except sqlite3.OperationalError:
        con = sqlite3.connect(store, isolation_level=None)
    try:
        con.execute("BEGIN")
        meta = {k: json.loads(v) for k, v in con.execute("SELECT k, v FROM meta")}
        requests = {}
        for rid, rec, backend, role, model in con.execute("SELECT id, record, backend, role, model FROM requests ORDER BY rowid"):
            requests[rid] = json.loads(rec)
            requests[rid].update({k: v for k, v in {"backend": backend, "role": role, "model": model}.items() if v})
        con.execute("COMMIT")
    finally:
        con.close()
    state = {k: meta[k] for k in ("limit_usd", "halt_reason", "operator_events", "counted_backends", "backend_totals") if k in meta}
    return {**state, "requests": requests}


def read_state(path: Path) -> dict | None:
    """The ledger's state (cap, stop reason, operator events, requests), read only: from its
    database when it has one, else from its JSON file; None when neither exists."""
    path = Path(path)
    store = store_of(path)
    if store.exists():
        return _read_store(store)
    if path.exists():
        return read_json_state(path)
    return None


def copy_ledger(source: Path, target: Path) -> None:
    """Copy the ledger named `source` to the name `target`, consistently while it is in use
    (SQLite online backup: a plain file copy can miss transactions still in the WAL file)."""
    target_store = store_of(target)
    target_store.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(store_of(source), isolation_level=None)
    dst = sqlite3.connect(target_store, isolation_level=None)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()


# ---- the ledger --------------------------------------------------------------------------

class _Txn:
    """One write transaction: the meta rows, read once, and the record changes, applied to
    the running totals as they are written."""

    def __init__(self, con):
        self.con = con
        self.meta = {k: json.loads(v) for k, v in con.execute("SELECT k, v FROM meta")}
        self.totals = [int(t) if i in _COUNTS else Decimal(t)
                       for i, t in enumerate(self.meta["totals"])]
        if "backend_totals" not in self.meta:
            self.meta["backend_totals"] = _backend_totals(con)
        else:
            self.meta["backend_totals"] = {k: list(v) for k, v in self.meta["backend_totals"].items()}
        self._now = time.time()

    def get(self, rid) -> dict | None:
        row = self.con.execute("SELECT record FROM requests WHERE id = ?", (rid,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, rid, record, old=None, *, key: str | None = None) -> None:
        """Write `record` (replacing `old`); `key` is the request's cost key, kept as a column."""
        if key:
            role, backend, model = (key.split("|") + ["", "", ""])[:3]
            record.update(role=role, backend=backend, model=model)
        elif old is not None and not record.get("backend"):
            row = self.con.execute("SELECT backend FROM requests WHERE id = ?", (rid,)).fetchone()
            if row and row[0]:
                record["backend"] = row[0]
        backend = _backend(record)
        values = _parse_totals(self.meta["backend_totals"].get(backend, [str(v) for v in _zero_totals()]))
        for i, value in enumerate(_contribution(record)):
            values[i] += value - (_contribution(old)[i] if old is not None else 0)
            total = self.totals[i] + value - (_contribution(old)[i] if old is not None else 0)
            self.totals[i] = total if total else _zero_totals()[i]
        self.meta["backend_totals"] = {**self.meta["backend_totals"], backend: [str(v) for v in values]}
        self.con.execute(_UPSERT, _row(rid, record, self._now, key))
        self.meta["totals"] = [str(t) for t in self.totals]

    def any_status(self, *statuses) -> bool:
        marks = ",".join("?" * len(statuses))
        return self.con.execute(f"SELECT 1 FROM requests WHERE status IN ({marks}) LIMIT 1",
                                statuses).fetchone() is not None

    def max_exposure(self) -> Decimal:
        scope = self.meta.get("counted_backends")
        clause = "" if scope is None else " AND (backend IS NULL OR backend = '' OR backend IN (" + ",".join("?" for _ in scope) + "))"
        row = self.con.execute("SELECT record FROM requests WHERE status = 'reserved' " + clause +
                               " ORDER BY exposure DESC LIMIT 1", [] if scope is None else scope).fetchone()
        if row is None:
            return Decimal("0")
        record = json.loads(row[0])
        return _bound(record) - _amount(record["reserved_usd"])

    def write_meta(self, before: dict) -> None:
        for k, v in self.meta.items():
            if before.get(k) != v:
                self.con.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)",
                                 (k, json.dumps(v)))


def run_budget_limit(path: Path, given: str | None = None) -> str:
    """Paid run entry: reuse the ledger cap, or explicitly create a new ledger.

    An explicit cap must match an existing ledger. Callers need no duplicate cap.
    """
    ledger = BudgetLedger(path, limit_usd=given)
    cap = _amount(ledger.summary()["limit_usd"])
    if given is not None and _amount(given, positive=True) != cap:
        raise ValueError(f"--judge-budget-usd {given} differs from ledger limit {cap}; "
                         "omit --judge-budget-usd to use the pinned ledger limit")
    return str(cap)


class BudgetLedger:
    """Reserve before every HTTP attempt (retries included), settle by the provider's charge.

    An unknown charge is counted at the request's worst case (`unknown_capped`) and the run
    continues. The only refusal is a new reservation that would exceed the cap
    (`BudgetCapReached`; `reserve_or_pause` waits for room instead). A known charge above a
    request's worst case stops the ledger: the price bound itself is then wrong.
    """

    def __init__(self, path: Path, *, limit_usd: str | None = None, counted_backends=...):
        """Open (or create) a ledger. The cap lives on the ledger: `limit_usd` creates a new
        ledger with that cap; on an existing one it can only tighten this process's
        reservations (the smaller applies; the ledger's cap stays the global authority and
        changes only through `set_limit`, which is logged)."""
        path = Path(path)
        self.path = path.with_name(path.name[:-len(".sqlite")]) if path.suffix == ".sqlite" else path
        self.store = store_of(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._process_cap = None
        self._local = threading.local()
        self._writer = threading.Lock()
        self._settings = budget_settings()
        self._creation_scope = self._settings.get("counted_backends") if counted_backends is ... else counted_backends
        self._validate_scope(self._creation_scope)
        self._open(limit_usd)
        if counted_backends is not ... and self._meta().get("counted_backends") != counted_backends:
            raise ValueError("Existing ledger scope differs; use haenv ops budget --counted-backends")
        if limit_usd is not None:
            given, cap = _amount(limit_usd, positive=True), self.limit
            if given < cap:
                self._process_cap = given
                log.warning("ledger %s: cap is $%s on the ledger; this process was approved "
                            "for $%s and reserves under that", self.path, cap, limit_usd)
            elif given > cap:
                log.warning("ledger %s: cap is $%s on the ledger (process was given $%s); "
                            "the ledger's value is authoritative", self.path, cap, limit_usd)

    def _open(self, limit_usd) -> None:
        if not self.store.exists():
            with self.store.with_name(self.store.name + ".create.lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if not self.store.exists():
                    if self.path.exists():
                        log.info("ledger %s: importing the JSON ledger into %s (read only)",
                                 self.path, self.store)
                        _create(self.store, read_json_state(self.path),
                                _json_provenance(self.path))
                    elif limit_usd is None:
                        raise ValueError("A new ledger needs its approved limit")
                    else:
                        _create(self.store, {"limit_usd": str(_amount(limit_usd, positive=True)),
                                             "halt_reason": None, "requests": {}, "counted_backends": self._creation_scope}, None)
        source = self._meta().get("imported_from")
        current = _json_provenance(self.path) if self.path.exists() else None
        if source is not None and current is not None and (
                (current["size"], current["mtime_ns"]) != (source["size"], source["mtime_ns"])):
            raise BudgetExceeded(f"JSON ledger {self.path} changed after its import into "
                                 f"{self.store}; a writer of the old format is still running")

    # ---- connection and transactions ----

    def _con(self) -> sqlite3.Connection:
        local = self._local
        if getattr(local, "pid", None) != os.getpid():
            con = sqlite3.connect(self.store, timeout=self._settings["busy_timeout_s"],
                                  isolation_level=None, check_same_thread=False)
            # A writer collected during a read-only dashboard must not checkpoint the
            # database on close. WAL transactions remain durable and normal automatic
            # checkpoints still run on commits. Python 3.12 exposes this SQLite option.
            if hasattr(con, "setconfig"):
                con.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=" + ("FULL" if self._settings["durable_commits"]
                                                 else "NORMAL"))
            local.con, local.pid = con, os.getpid()
        return local.con

    @contextmanager
    def _txn(self):
        """One write transaction. Writers queue on a kernel lock (`<store>.lock`, released
        when a process dies) before `BEGIN IMMEDIATE`: a waiter wakes the moment the lock is
        free, where SQLite's own busy handler would poll with growing sleeps."""
        con = self._con()
        with self._writer, self._lock_file() as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            con.execute("BEGIN IMMEDIATE")
            try:
                txn = _Txn(con)
                before = dict(txn.meta)
                yield txn
                txn.write_meta(before)
            except BaseException:
                con.execute("ROLLBACK")
                raise
            con.execute("COMMIT")

    @contextmanager
    def _lock_file(self):
        local = self._local
        if getattr(local, "lock_pid", None) != os.getpid():
            local.lock, local.lock_pid = self.store.with_name(self.store.name + ".lock").open("a"), os.getpid()
        try:
            yield local.lock
        finally:
            fcntl.flock(local.lock.fileno(), fcntl.LOCK_UN)

    def _meta(self) -> dict:
        return {k: json.loads(v) for k, v in self._con().execute("SELECT k, v FROM meta")}

    def _cap(self, meta) -> Decimal:
        cap = _amount(meta["limit_usd"])
        return cap if self._process_cap is None else min(cap, self._process_cap)

    @property
    def limit(self) -> Decimal:
        """The cap this process reserves under: the ledger's, tightened by a smaller
        process cap given at open."""
        return self._cap(self._meta())

    def set_limit(self, new_usd: str, *, reason: str, expect_usd: str | None = None) -> dict:
        """Raise or lower the cap, with an operator event (`expect_usd`: refuse unless the
        current cap is that). Lowering below the committed amount only refuses new
        reservations; nothing already counted is released."""
        from datetime import datetime, timezone
        new = _amount(new_usd, positive=True)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("An explicit reason is required")
        with self._txn() as txn:
            old = _amount(txn.meta["limit_usd"])
            if expect_usd is not None and old != _amount(expect_usd):
                raise ValueError(f"The ledger's current cap is ${old}, not ${expect_usd}")
            event = {"action": "raise_limit" if new > old else "lower_limit", "old": str(old),
                     "new": str(new), "reason": reason, "at": datetime.now(timezone.utc).isoformat()}
            txn.meta["limit_usd"] = str(new)
            txn.meta["operator_events"] = [*txn.meta.get("operator_events", []), event]
        return event

    @staticmethod
    def _validate_scope(scope):
        if scope is not None and (not isinstance(scope, list) or any(not isinstance(b, str) or not b.strip() for b in scope)):
            raise ValueError("counted_backends must be null (all) or a list of backend names")

    def set_counted_backends(self, backends, *, reason: str) -> dict:
        self._validate_scope(backends)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("An explicit reason is required")
        with self._txn() as txn:
            event = {"action": "set_counted_backends", "old": txn.meta.get("counted_backends"),
                     "new": backends, "reason": reason, "at": time.time()}
            txn.meta["counted_backends"] = backends
            txn.meta["operator_events"] = [*txn.meta.get("operator_events", []), event]
        return event

    def operator_event(self, event: dict, *, halt_reason=...) -> None:
        """Record an operator event; with `halt_reason`, set (or clear, None) the stop too."""
        with self._txn() as txn:
            txn.meta["operator_events"] = [*txn.meta.get("operator_events", []), event]
            if halt_reason is not ...:
                txn.meta["halt_reason"] = halt_reason

    # ---- reservations ----

    def reservation(self, cost_key: str | None, maximum_usd) -> Decimal:
        """What a request on `cost_key` with worst case `maximum_usd` reserves: the
        configured quantile of the key's settled costs once it has enough of them, never
        more than the worst case."""
        maximum = _amount(maximum_usd)
        s = self._settings
        if cost_key is None or not s["overrun_slots"]:
            return maximum
        con = self._con()
        n = con.execute("SELECT COUNT(*) FROM requests WHERE status = 'settled' AND cost_key = ? "
                        "AND actual_num IS NOT NULL", (cost_key,)).fetchone()[0]
        if n < s["reservation_min_observations"]:
            return maximum
        rank = max(0, math.ceil(s["reservation_quantile"] * n) - 1)
        row = con.execute("SELECT actual_usd FROM requests WHERE status = 'settled' AND "
                          "cost_key = ? AND actual_num IS NOT NULL ORDER BY actual_num "
                          "LIMIT 1 OFFSET ?", (cost_key, rank)).fetchone()
        return min(maximum, _amount(row[0])) if row else maximum

    def reserve(self, request_id: str, maximum_usd: str, *, zero_basis: str | None = None,
                cost_key: str | None = None) -> None:
        """Reserve for one request whose worst case is `maximum_usd` (see `reservation`).
        A $0 worst case needs `zero_basis`, the recorded reason the route is free (a declared
        price of 0), so a price schedule that wrongly returns 0 cannot pass requests through
        unmetered."""
        maximum = _amount(maximum_usd, positive=not zero_basis)
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("A nonempty independent request id is required")
        hold = self.reservation(cost_key, maximum)
        with self._txn() as txn:
            halt = txn.meta.get("halt_reason")
            if halt and halt != LEGACY_UNKNOWN_HALT:
                raise BudgetExceeded(halt)
            prior = txn.get(request_id)
            if prior is not None and not prior.get("cancelled_before_dispatch"):
                raise ValueError("Request id has already been reserved")
            backend = (cost_key.split("|") + [""])[1] if cost_key else "unattributed"
            included = _included(backend or "unattributed", txn.meta.get("counted_backends"))
            committed = _scope_totals(txn.meta["backend_totals"], txn.meta.get("counted_backends"))[0]
            limit = self._cap(txn.meta)
            headroom = self._settings["overrun_slots"] * max(maximum - hold, txn.max_exposure())
            if included and committed + hold + headroom > limit:
                raise BudgetCapReached(
                    f"Budget exhausted: request reservation ${hold} (worst case ${maximum}, "
                    f"overrun headroom ${headroom}) would exceed the approved budget "
                    f"(committed ${committed} of ${limit})")
            record = {"reserved_usd": str(hold), "actual_usd": None, "status": "reserved"}
            if hold != maximum:
                record["bound_usd"] = str(maximum)
            if not maximum:
                record["zero_bound_basis"] = zero_basis
            txn.put(request_id, record, prior, key=cost_key)

    def reserve_or_pause(self, request_id: str, maximum_usd: str, *, zero_basis=None,
                         cost_key=None, poll_s=None, timeout_s=None, sleep=None, clock=None,
                         on_pause=None, check=None) -> dict | None:
        """`reserve`, but at the cap wait (paused) for room instead of failing at once: the
        one path solver and judge requests both take.

        Re-reads the ledger every `poll_s`; after `timeout_s` without room the refusal is
        raised. Both default to the runtime document's `budget` block, else
        `config.budget`. `check()` runs before every attempt (default: the runtime's drain
        check, which stops the wait). A pause and its end are written to the ops log.
        Returns the pause record, or None when no pause was needed."""
        from haenv.runtime.ops.runtime import check_dispatch
        from haenv.runtime.ops.runtime import for_ledger
        from haenv.runtime.ops.runtime import ops_log
        knobs = {**{k: self._settings[k] for k in ("pause_poll_s", "pause_timeout_s")},
                 **for_ledger(self.path).current()["budget"]}
        poll_s = float(knobs["pause_poll_s"] if poll_s is None else poll_s)
        timeout_s = float(knobs["pause_timeout_s"] if timeout_s is None else timeout_s)
        sleep, clock = sleep or time.sleep, clock or time.monotonic
        check = check or (lambda: check_dispatch(self.path))
        started, pause = clock(), None
        while True:
            check()
            try:
                self.reserve(request_id, maximum_usd, zero_basis=zero_basis, cost_key=cost_key)
            except BudgetCapReached as refusal:
                if pause is None:
                    pause = {"request_id": request_id, "reason": str(refusal)}
                    ops_log(self.path, "budget_paused", **pause)
                    if on_pause is not None:
                        on_pause(pause)
                if clock() - started >= timeout_s:
                    raise
                sleep(poll_s)
                continue
            if pause is not None:
                pause["resumed_after_s"] = round(clock() - started, 3)
                ops_log(self.path, "budget_resumed", **pause)
            return pause

    def book_replayed(self, request_id: str, maximum_usd: str, *,
                      zero_basis: str | None = None, cost_key: str | None = None) -> dict:
        """Book a request whose durable receipt exists but which this ledger never saw (the
        ledger was rebuilt or the batch copied): `reserved` at its worst case, marked
        `booked_from_receipt`, for the caller to settle from the receipt's cost evidence.
        No cap check: the money was already spent, refusing the entry would hide it."""
        maximum = _amount(maximum_usd, positive=not zero_basis)
        with self._txn() as txn:
            if txn.get(request_id) is not None:
                raise ValueError("Request id has already been reserved")
            record = {"reserved_usd": str(maximum), "actual_usd": None, "status": "reserved",
                      "booked_from_receipt": True,
                      **({"zero_bound_basis": zero_basis} if not maximum else {})}
            txn.put(request_id, record, key=cost_key)
        return dict(record)

    # ---- settlement ----

    UNKNOWN_REASONS = ("dispatch_failed_no_receipt", "provider_cost_missing",
                       "provider_cost_invalid", "no_response_object",
                       "reserved_without_durable_receipt", "legacy_unknown_carried")

    def record_unknown(self, request_id: str, *, reason: str) -> None:
        """Count one request of unknown cost at its worst case and keep going.

        `reserved` (or a legacy `unknown`) becomes `unknown_capped`; the charge is never
        estimated and the worst case stays counted. No halt is set. A later provider
        receipt may settle it (`settle_unknown_later`)."""
        if reason not in self.UNKNOWN_REASONS:
            raise ValueError("Unrecognized unknown-cost reason")
        with self._txn() as txn:
            old = txn.get(request_id)
            if old is None or old["status"] not in ("reserved", "unknown"):
                raise ValueError("Only a reserved request can become unknown-cost")
            record = dict(old, actual_usd=None, status="unknown_capped",
                          reconciliation=f"{reason}; counted at its reserved upper bound")
            _count_at_bound(record)
            txn.put(request_id, record, old)
            if txn.meta.get("halt_reason") == LEGACY_UNKNOWN_HALT and not txn.any_status("unknown"):
                txn.meta["halt_reason"] = None
        log.warning("cost unknown for request %s (%s); counted at its upper bound $%s, continuing",
                    request_id, reason, record["reserved_usd"])

    def _settle_known(self, request_id, actual: Decimal, *, statuses, extra) -> None:
        """Settle at the provider's charge. Above the reservation but within the worst case:
        settled at the charge and logged. Above the worst case: settled, and the ledger stops."""
        with self._txn() as txn:
            old = txn.get(request_id)
            if old is None or old["status"] not in statuses:
                raise ValueError("Only a reserved request may be settled" if statuses == ("reserved",)
                                 else "Only an unknown-cost request can be settled afterwards")
            record = dict(old, **extra(old), actual_usd=str(actual), status="settled")
            violated = actual > _bound(old)
            if not violated and actual > _amount(old["reserved_usd"]):
                record["overrun_usd"] = str(actual - _amount(old["reserved_usd"]))
                log.warning("request %s cost $%s, above its reservation $%s (worst case $%s); "
                            "settled at the charge", request_id, actual, old["reserved_usd"],
                            _bound(old))
            txn.put(request_id, record, old)
            if violated:
                txn.meta["halt_reason"] = BOUND_VIOLATION
        if violated:
            raise BudgetExceeded(BOUND_VIOLATION)

    def settle(self, request_id: str, actual_usd: str | None, *,
               reason: str = "provider_cost_missing", generation_id: str | None = None) -> None:
        """Settle a reserved request at the provider's charge (None: unknown, `record_unknown`)."""
        if actual_usd is None:
            self.record_unknown(request_id, reason=reason)
            return
        gid = {"generation_id": generation_id} if generation_id else {}
        self._settle_known(request_id, _amount(actual_usd), statuses=("reserved",),
                           extra=lambda old: gid)

    def settle_unknown_later(self, request_id: str, actual_usd: str, *, evidence: str,
                             generation_id: str | None = None) -> None:
        """Replace a counted worst case by the provider's actual charge found afterwards
        (generation record, relay quota log, ...). Frees the difference; a charge above the
        worst case stops the ledger like `settle`."""
        from datetime import datetime, timezone
        actual = _amount(actual_usd)
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("The provider evidence for the actual charge is required")
        at = datetime.now(timezone.utc).isoformat()
        self._settle_known(request_id, actual, statuses=("unknown_capped",), extra=lambda old: {
            "settled_after_unknown": {"bound_usd": old["reserved_usd"], "evidence": evidence,
                                      "at": at},
            **({"generation_id": generation_id} if generation_id else {})})

    def settle_rejected(self, request_id: str, *, basis: str) -> None:
        """Settle an attempt the provider rejected with an HTTP error before any generation
        at $0. Only for a backend whose published rule says such a response is not billed;
        `basis` names the status, the rule and its source."""
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("The documented basis for a $0 rejection is required")
        with self._txn() as txn:
            old = txn.get(request_id)
            if old is None or old["status"] != "reserved":
                raise ValueError("Only a reserved request may be settled")
            txn.put(request_id, dict(old, actual_usd="0", status="settled",
                                     zero_cost_basis=basis), old)

    def cancel_unsent(self, request_id: str) -> None:
        """Release a held reservation only when its caller proves dispatch never began."""
        with self._txn() as txn:
            old = txn.get(request_id)
            if old is None:
                return
            if old["status"] != "reserved":
                raise ValueError("Only an unsent reserved request can be cancelled")
            txn.put(request_id, dict(old, actual_usd="0", status="cancelled",
                                     zero_cost_basis="operator_stop_before_dispatch", cancelled_before_dispatch=True), old)

    def settle_bound(self, request_id: str, upper_usd: str, *, basis: str) -> None:
        """Count a verified tariff bound (response tokens x published price), explicitly not
        a provider invoice: `reserved_usd` becomes the bound, `actual_usd` stays empty."""
        upper = _amount(upper_usd)
        if not isinstance(basis, str) or not basis:
            raise ValueError("A verified tariff evidence identity is required")
        with self._txn() as txn:
            old = txn.get(request_id)
            if old is None or old["status"] != "reserved":
                raise ValueError("Only an active reserved request can receive a tariff bound")
            violated = upper > _bound(old)
            record = {k: v for k, v in old.items() if k != "bound_usd"}
            record.update(original_reserved_usd=str(_bound(old)), reserved_usd=str(upper),
                          actual_usd=None, status="tariff_capped", cost_basis=basis)
            txn.put(request_id, record, old)
            if violated:
                txn.meta["halt_reason"] = "Measured tariff bound exceeded the reserved upper bound"
        if violated:
            raise BudgetExceeded("Measured tariff bound exceeded its reservation")

    # ---- reconciliation ----

    RECONCILE_BASES = ("process_exited_no_provider_receipt",
                       "process_exited_saved_response_without_cost")

    def reconcile_unknown_at_bound(self, request_id: str, *,
                                   basis: str = "process_exited_no_provider_receipt") -> None:
        """Close one lost receipt of a legacy `unknown` request, keeping its worst case
        counted. Call only after verifying the request process has exited and no durable
        provider receipt exists; this neither invents a bill nor frees budget."""
        if basis not in self.RECONCILE_BASES:
            raise ValueError("Unrecognized reconciliation basis")
        with self._txn() as txn:
            old = txn.get(request_id)
            if old is None or old["status"] != "unknown":
                raise ValueError("Only an unknown-cost request may be reconciled at its bound")
            if txn.meta.get("halt_reason") != LEGACY_UNKNOWN_HALT:
                raise BudgetExceeded("Another accounting halt must be resolved independently")
            if txn.totals[0] > _amount(txn.meta["limit_usd"]):
                raise BudgetExceeded("Conservative charges exceed the approved budget")
            record = dict(old, status="unknown_capped",
                          reconciliation=f"{basis}; full reservation retained")
            _count_at_bound(record)
            txn.put(request_id, record, old)
            if not txn.any_status("unknown"):
                txn.meta["halt_reason"] = None

    def reconcile_orphan_at_bound(self, request_id: str) -> None:
        """Close a reservation whose issuing process was killed mid-request, counted at its
        worst case. Call only after verifying that process no longer exists and no durable
        provider receipt was written; this never frees budget or invents a bill."""
        with self._txn() as txn:
            old = txn.get(request_id)
            if old is None or old["status"] != "reserved":
                raise ValueError("Only a reserved request can be closed as an orphan")
            record = dict(old, status="unknown_capped", reconciliation=(
                "process_killed_in_flight_no_provider_receipt; full reservation retained"))
            _count_at_bound(record)
            txn.put(request_id, record, old)

    def release_bound_violation(self, *, reason: str) -> list[str]:
        """Clear a 'charge exceeded its worst case' stop after the bound was fixed. Every
        violating request is already settled at its actual charge, nothing may be in flight,
        and the reason is recorded with the violating ids."""
        from datetime import datetime, timezone
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("An explicit reason is required")
        with self._txn() as txn:
            if txn.meta.get("halt_reason") != BOUND_VIOLATION:
                raise ValueError("No reservation-violation stop is active")
            if txn.any_status("reserved", "unknown"):
                raise ValueError("Resolve in-flight and unknown requests first")
            violating = sorted(
                rid for rid, rec in ((r, json.loads(x)) for r, x in txn.con.execute(
                    "SELECT id, record FROM requests WHERE actual_usd IS NOT NULL"))
                if _amount(rec["actual_usd"]) > _bound(rec))
            if not violating or txn.totals[0] > _amount(txn.meta["limit_usd"]):
                raise ValueError("Violation not found or conservative total exceeds the cap")
            txn.meta["halt_reason"] = None
            txn.meta["operator_events"] = [*txn.meta.get("operator_events", []), {
                "action": "release_bound_violation", "reason": reason, "violating": violating,
                "at": datetime.now(timezone.utc).isoformat()}]
        return violating

    # ---- reading ----

    def record(self, request_id: str) -> dict | None:
        """One request's record (a copy)."""
        row = self._con().execute("SELECT record FROM requests WHERE id = ?",
                                  (request_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def summary(self) -> dict:
        """Cap, stop reason, operator events and the running totals, without the records."""
        meta = self._meta()
        if "backend_totals" not in meta:
            meta["backend_totals"] = _backend_totals(self._con())
        totals = [int(t) if i in _COUNTS else Decimal(t) for i, t in enumerate(meta["totals"])]
        return _summary(meta, totals)

    def snapshot(self) -> dict:
        """`summary` plus every record (`requests`), totals summed afresh from the records."""
        state = _read_store(self.store)
        return {**_summary(state, _sum_totals(state["requests"].values())),
                "requests": state["requests"]}


def cost_summary(snapshot: dict) -> str:
    """"paid $X + N unknown (counted at bound $Y)", plus tariff estimates and in-flight holds."""
    text = (f"paid ${snapshot['settled_usd']} + {snapshot['unknown_capped_n']} unknown "
            f"(counted at bound ${snapshot['unknown_capped_usd']})")
    if snapshot.get("tariff_capped_n"):
        text += (f" + {snapshot['tariff_capped_n']} tariff-estimated "
                 f"(${snapshot['tariff_bound_usd']}, response tokens x published price)")
    if Decimal(snapshot.get("in_flight_usd", "0")):
        text += f" + in flight ${snapshot['in_flight_usd']}"
    return text + f"; committed ${snapshot['committed_usd']} of ${snapshot['limit_usd']}"
