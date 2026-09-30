"""Persistent, process-safe dollar cap for the complete semantic-validation run."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl
import json
import logging
import os
from pathlib import Path
import tempfile
import threading

log = logging.getLogger("haenv.budget")

# A halt reason some existing ledgers carry; it stops nothing (unknown costs count at their bound).
LEGACY_UNKNOWN_HALT = "A billed request has unknown cost; reconciliation required"


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


#: Journal lines between two compactions of the ledger into its snapshot file.
COMPACT_EVERY = 20000
_TOP = ("limit_usd", "halt_reason", "operator_events")
#: Leading bytes compared on every refresh to notice a ledger rewritten in place.
_HEAD = 256


class _Tracked(dict):
    """The requests map of a loaded ledger: remembers which ids a locked block touched, so a
    write appends those records only, instead of rewriting the whole ledger."""

    def __init__(self, *args):
        super().__init__(*args)
        self.touched = set()

    def __getitem__(self, key):
        self.touched.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.touched.add(key)
        return super().get(key, default)

    def __setitem__(self, key, value):
        self.touched.add(key)
        super().__setitem__(key, value)


def _contribution(record) -> tuple:
    """(committed, settled, unknown_n, unknown_usd, tariff_n, tariff_usd, in_flight) of one record."""
    reserved = _amount(record["reserved_usd"])
    actual = _amount(record["actual_usd"]) if record.get("actual_usd") is not None else None
    status = record["status"]
    zero = Decimal("0")
    return (actual if actual is not None else reserved, actual if actual is not None else zero,
            int(status == "unknown_capped"), reserved if status == "unknown_capped" else zero,
            int(status == "tariff_capped"), reserved if status == "tariff_capped" else zero,
            reserved if status == "reserved" else zero)


def _complete_lines(data: bytes) -> tuple[list[bytes], int]:
    """Lines that end in a newline, and the byte count they span (a torn last line written by
    an interrupted append is left out)."""
    end = data.rfind(b"\n") + 1
    return [line for line in data[:end].split(b"\n") if line.strip()], end


def _parse(data: bytes) -> tuple[dict, list[bytes], int]:
    """(snapshot, appended change lines, byte offset after the last complete line). A snapshot
    written without a trailing newline (a legacy ledger) ends at its closing brace."""
    text = data.decode("utf-8")
    state, end = json.JSONDecoder().raw_decode(text)
    start = len(text[:end].encode("utf-8"))
    newline = data.find(b"\n", start)
    if newline < 0:
        return state, [], start
    start = newline + 1
    lines, consumed = _complete_lines(data[start:])
    return state, lines, start + consumed


def _apply(state: dict, line: bytes) -> str | None:
    entry = json.loads(line)
    if "r" in entry:
        state["requests"][entry["r"]] = entry["v"]
        return entry["r"]
    for key, value in entry["top"].items():
        state[key] = value
    return None


def _top_json(state: dict) -> str:
    return json.dumps({k: state[k] for k in _TOP if k in state}, sort_keys=True)


def read_state(path: Path) -> dict | None:
    """The ledger's current state (snapshot plus appended lines), without a lock."""
    path = Path(path)
    if not path.exists():
        return None
    state, lines, _ = _parse(path.read_bytes())
    for line in lines:
        _apply(state, line)
    return state


class BudgetLedger:
    """Reserve a verified upper bound before every HTTP attempt, including retries.

    An unknown charge is counted at its full reservation (`unknown_capped`) and
    the run continues; the cap holds because every unknown stays counted at its
    bound. The only refusal is a new reservation that would exceed the cap. A
    known charge above its reservation still stops the run: the bound itself is
    then wrong. The caller must obtain a genuine price bound rather than guess one.
    """

    def __init__(self, path: Path, *, limit_usd: str | None = None):
        """Open (or create) a ledger. The cap lives on the ledger file: `limit_usd` creates a
        new ledger with that cap; on an existing one it can only tighten this process's
        reservations (the smaller of the two applies; the file's cap is unchanged and stays
        the global authority; raise it with `set_limit`, which is logged)."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        #: this process's approved cap when it is below the file's (only tightens)
        self._process_cap = None
        self._mat_lock = threading.RLock()
        self._mat = None            # the loaded state, reused while the files are unchanged
        with self._locked() as state:
            if state is None:
                if limit_usd is None:
                    raise ValueError("A new ledger needs its approved limit")
                self._write({"limit_usd": str(_amount(limit_usd, positive=True)),
                             "halt_reason": None, "requests": {}})
            elif limit_usd is not None and _amount(state["limit_usd"]) != _amount(limit_usd):
                given = _amount(limit_usd, positive=True)
                if given < _amount(state["limit_usd"]):
                    self._process_cap = given
                    log.warning("ledger %s: cap is $%s on the ledger; this process was approved "
                                "for $%s and reserves under that", self.path, state["limit_usd"], limit_usd)
                else:
                    log.warning("ledger %s: cap is $%s on the ledger (process was given $%s); "
                                "the ledger's value is authoritative", self.path, state["limit_usd"],
                                limit_usd)

    def _cap(self, state) -> Decimal:
        file_cap = _amount(state["limit_usd"])
        return file_cap if self._process_cap is None else min(file_cap, self._process_cap)

    @property
    def limit(self) -> Decimal:
        """The cap this process reserves under: the file's cap, tightened by a smaller
        process cap given at open."""
        with self._locked() as state:
            return self._cap(state)

    def set_limit(self, new_usd: str, *, reason: str) -> dict:
        """Raise or lower the cap, with an operator event. Lowering below the committed
        amount only refuses new reservations; nothing already counted is released."""
        from datetime import datetime, timezone
        new = _amount(new_usd, positive=True)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("An explicit reason is required")
        with self._locked() as state:
            old = _amount(state["limit_usd"])
            event = {"action": "raise_limit" if new > old else "lower_limit", "old": str(old),
                     "new": str(new), "reason": reason, "at": datetime.now(timezone.utc).isoformat()}
            state["limit_usd"] = str(new)
            state.setdefault("operator_events", []).append(event)
            self._write(state)
        return event

    # ---- storage: one file, a snapshot line followed by appended change lines ----
    #
    # The first JSON value in the file is the full state (a legacy ledger's indented JSON
    # reads the same way); every later line is one changed request record (`{"r": id, "v":
    # record}`) or top-level change (`{"top": {...}}`), replayed in order as upserts. A write
    # appends only the records its locked block touched, so its cost does not grow with the
    # ledger; every COMPACT_EVERY lines the file is rewritten as a single snapshot (atomic
    # replace). Copying the one file copies the whole ledger. The loaded state and running
    # totals stay in memory while the file only grows.

    def _inode(self):
        try:
            return self.path.stat().st_ino
        except FileNotFoundError:
            return None

    def _reload(self):
        try:
            with self.path.open("rb") as handle:
                ino, data = os.fstat(handle.fileno()).st_ino, handle.read()
        except FileNotFoundError:
            self._mat = None
            return
        state, lines, offset = _parse(data)
        state["requests"] = _Tracked(state.get("requests") or {})
        for line in lines:
            _apply(state, line)
        state["requests"].touched.clear()
        self._mat = {"state": state, "contrib": {}, "ino": ino, "offset": offset, "lines": len(lines),
                     "head": data[:_HEAD],
                     "totals": [Decimal("0") if i not in (2, 4) else 0 for i in range(7)],
                     "exps": [Counter() for _ in range(7)]}
        for rid in state["requests"]:
            self._account(rid)

    def _refresh(self):
        """Bring the loaded state up to the file: read only the lines appended since."""
        mat = self._mat
        if mat is None or self._inode() != mat["ino"]:
            self._reload()
            return
        try:
            with self.path.open("rb") as handle:
                size = os.fstat(handle.fileno())
                # replaced, shrunk or rewritten in place by another writer: read it whole again
                if (size.st_ino != mat["ino"] or size.st_size < mat["offset"]
                        or handle.read(_HEAD) != mat["head"]):
                    self._reload()
                    return
                handle.seek(mat["offset"])
                lines, consumed = _complete_lines(handle.read())
        except FileNotFoundError:
            self._reload()
            return
        state = mat["state"]
        for line in lines:
            rid = _apply(state, line)
            if rid is not None:
                self._account(rid)
        state["requests"].touched.clear()
        mat["offset"] += consumed
        mat["lines"] += len(lines)

    def _account(self, rid):
        mat = self._mat
        new = _contribution(mat["state"]["requests"][rid])
        old = mat["contrib"].get(rid)
        for i in range(7):
            if old is not None:
                mat["totals"][i] -= old[i]
                if i not in (2, 4):
                    mat["exps"][i][old[i].as_tuple().exponent] -= 1
            mat["totals"][i] += new[i]
            if i not in (2, 4):
                mat["exps"][i][new[i].as_tuple().exponent] += 1
        mat["contrib"][rid] = new

    def _total(self, i) -> Decimal:
        """Running total `i`, written with the decimal places a fresh sum of the records has."""
        mat = self._mat
        places = min([0] + [e for e, n in mat["exps"][i].items() if n > 0])
        return mat["totals"][i].quantize(Decimal(1).scaleb(places))

    @contextmanager
    def _locked(self):
        with self._mat_lock, self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._refresh()
                if self._mat is None:
                    self._written = True
                    yield None
                    return
                state = self._mat["state"]
                top = _top_json(state)
                self._written, self._top = False, top
                try:
                    yield state
                except BaseException:
                    self._mat = None            # a block that raised may have half-mutated the state
                    raise
                if (state["requests"].touched or _top_json(state) != top) and not self._written:
                    self._mat = None            # touched but not written: never trust it again
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _write(self, state):
        if self._mat is None or state is not self._mat["state"]:
            self._write_snapshot(state)          # a new ledger
            self._mat = None
            self._written = True
            return
        mat = self._mat
        lines = [json.dumps({"r": rid, "v": state["requests"][rid]}, sort_keys=True,
                            separators=(",", ":"))
                 for rid in sorted(state["requests"].touched) if rid in state["requests"]]
        if _top_json(state) != self._top:
            lines.append(json.dumps({"top": {k: state[k] for k in _TOP if k in state}},
                                    sort_keys=True, separators=(",", ":")))
        if lines:
            with self.path.open("r+b") as handle:
                handle.seek(mat["offset"])            # drops a torn line left by an interrupted append
                handle.truncate()
                lead = b""
                if mat["offset"]:
                    handle.seek(mat["offset"] - 1)
                    lead = b"" if handle.read(1) == b"\n" else b"\n"
                handle.write(lead + ("\n".join(lines) + "\n").encode())
                handle.flush()
                os.fsync(handle.fileno())
                mat["offset"] = handle.tell()
        for rid in list(state["requests"].touched):
            if rid in state["requests"]:
                self._account(rid)
        state["requests"].touched.clear()
        self._top = _top_json(state)
        self._written = True
        mat["lines"] += len(lines)
        if mat["lines"] >= COMPACT_EVERY:
            self._compact(state)

    def _write_snapshot(self, state) -> int:
        plain = {**state, "requests": dict(state.get("requests") or {})}
        body = (json.dumps(plain, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with tempfile.NamedTemporaryFile(mode="wb", dir=self.path.parent,
                                         prefix=self.path.name + ".", delete=False) as file:
            temporary = Path(file.name)
            file.write(body)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.path)
        return len(body)

    def _compact(self, state):
        """Rewrite the file as one snapshot line (atomic replace)."""
        size = self._write_snapshot(state)
        with self.path.open("rb") as handle:
            head = handle.read(_HEAD)
        self._mat.update(ino=self._inode(), offset=size, lines=0, head=head)

    def compact(self) -> None:
        """Fold the appended lines into the snapshot now."""
        with self._locked() as state:
            if state is not None:
                self._compact(state)
                self._written = True

    def _committed(self, state) -> Decimal:
        if self._mat is not None and state is self._mat["state"]:
            total = self._total(0)
            for rid in state["requests"].touched:       # records changed inside this block
                record = dict.get(state["requests"], rid)
                old = self._mat["contrib"].get(rid)
                total += (_contribution(record)[0] if record is not None else 0) - (old[0] if old else 0)
            return total
        return sum((_amount(r["actual_usd"] if r["actual_usd"] is not None else r["reserved_usd"])
                    for r in state["requests"].values()), Decimal("0"))

    def reserve(self, request_id: str, maximum_usd: str, *, zero_basis: str | None = None) -> None:
        """Reserve `maximum_usd` for one request. A $0 reservation needs `zero_basis`, the
        recorded reason the route is free (a declared price of 0); without one it is refused,
        so a price schedule that wrongly returns 0 cannot pass requests through unmetered."""
        maximum = _amount(maximum_usd, positive=not zero_basis)
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("A nonempty independent request id is required")
        with self._locked() as state:
            # The legacy unknown-cost halt stops nothing: those charges are already
            # counted at their bounds.
            if state["halt_reason"] and state["halt_reason"] != LEGACY_UNKNOWN_HALT:
                raise BudgetExceeded(state["halt_reason"])
            if request_id in state["requests"]:
                raise ValueError("Request id has already been reserved")
            committed = self._committed(state)
            limit = self._cap(state)                       # the file's cap (read under the lock), tightened
            if committed + maximum > limit:
                raise BudgetCapReached(
                    f"Budget exhausted: request upper bound ${maximum} would exceed the approved "
                    f"budget (committed ${committed} of ${limit})")
            state["requests"][request_id] = {
                "reserved_usd": str(maximum), "actual_usd": None, "status": "reserved",
                **({"zero_bound_basis": zero_basis} if not maximum else {})}
            self._write(state)

    def book_replayed(self, request_id: str, maximum_usd: str, *,
                      zero_basis: str | None = None) -> dict:
        """Book a request whose durable receipt exists but which this ledger never saw (the
        ledger was rebuilt or the batch copied): `reserved` at its bound, marked
        `booked_from_receipt`, for the caller to settle from the receipt's cost evidence.
        No cap check: the money was already spent, refusing the entry would hide it.
        `zero_basis` as in `reserve`."""
        maximum = _amount(maximum_usd, positive=not zero_basis)
        with self._locked() as state:
            if request_id in state["requests"]:
                raise ValueError("Request id has already been reserved")
            record = {"reserved_usd": str(maximum), "actual_usd": None, "status": "reserved",
                      "booked_from_receipt": True,
                      **({"zero_bound_basis": zero_basis} if not maximum else {})}
            state["requests"][request_id] = record
            self._write(state)
            return dict(record)

    def reserve_or_pause(self, request_id: str, maximum_usd: str, *, poll_s: float = 30.0,
                         timeout_s: float = 1800.0, sleep=None, clock=None, on_pause=None,
                         check=None, zero_basis: str | None = None) -> dict | None:
        """`reserve`, but at the cap wait (paused) for a raise instead of failing at once.

        Re-reads the ledger every `poll_s`; after `timeout_s` without room the original
        refusal is raised. `check()` runs before every retry (a drain stops the wait).
        Returns the pause record, or None when no pause was needed.
        """
        import time
        sleep, clock = sleep or time.sleep, clock or time.monotonic
        started, pause = clock(), None
        while True:
            if check is not None:
                check()
            try:
                self.reserve(request_id, maximum_usd, zero_basis=zero_basis)
                if pause is not None:
                    pause["resumed_after_s"] = round(clock() - started, 3)
                return pause
            except BudgetCapReached as refusal:
                if pause is None:
                    pause = {"request_id": request_id, "reason": str(refusal)}
                    if on_pause is not None:
                        on_pause(pause)
                if clock() - started >= timeout_s:
                    raise
                sleep(poll_s)

    UNKNOWN_REASONS = ("dispatch_failed_no_receipt", "provider_cost_missing",
                       "provider_cost_invalid", "no_response_object",
                       "reserved_without_durable_receipt", "legacy_unknown_carried")

    def record_unknown(self, request_id: str, *, reason: str) -> None:
        """Count one request of unknown cost at its full reservation and keep going.

        `reserved` (or a legacy `unknown`) becomes `unknown_capped`; the charge is
        never estimated and the bound stays counted, so no budget is freed. No
        halt is set. A later provider receipt may settle it (`settle_unknown_later`).
        """
        if reason not in self.UNKNOWN_REASONS:
            raise ValueError("Unrecognized unknown-cost reason")
        with self._locked() as state:
            record = state["requests"].get(request_id)
            if record is None or record["status"] not in ("reserved", "unknown"):
                raise ValueError("Only a reserved request can become unknown-cost")
            record["actual_usd"] = None
            record["status"] = "unknown_capped"
            record["reconciliation"] = f"{reason}; counted at its reserved upper bound"
            if (state["halt_reason"] == LEGACY_UNKNOWN_HALT
                    and not any(r["status"] == "unknown" for r in state["requests"].values())):
                state["halt_reason"] = None
            self._write(state)
        log.warning("cost unknown for request %s (%s); counted at its upper bound $%s, continuing",
                    request_id, reason, record["reserved_usd"])

    def settle_unknown_later(self, request_id: str, actual_usd: str, *, evidence: str) -> None:
        """Replace a counted bound by the provider's actual charge found afterwards.

        Optional (generation record, relay quota log, ...). Frees the difference;
        a charge above the bound stops the run exactly like `settle`.
        """
        from datetime import datetime, timezone
        actual = _amount(actual_usd)
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("The provider evidence for the actual charge is required")
        violated = False
        with self._locked() as state:
            record = state["requests"].get(request_id)
            if record is None or record["status"] != "unknown_capped":
                raise ValueError("Only an unknown-cost request can be settled afterwards")
            record["settled_after_unknown"] = {"bound_usd": record["reserved_usd"],
                                               "evidence": evidence,
                                               "at": datetime.now(timezone.utc).isoformat()}
            record["actual_usd"] = str(actual)
            record["status"] = "settled"
            if actual > _amount(record["reserved_usd"]):
                state["halt_reason"] = "Provider charge exceeded the reserved upper bound"
                violated = True
            self._write(state)
        if violated:
            raise BudgetExceeded("Provider charge exceeded the reserved upper bound")

    def settle(self, request_id: str, actual_usd: str | None, *,
               reason: str = "provider_cost_missing") -> None:
        if actual_usd is None:
            self.record_unknown(request_id, reason=reason)
            return
        actual = _amount(actual_usd)
        violated = False
        with self._locked() as state:
            record = state["requests"].get(request_id)
            if record is None or record["status"] != "reserved":
                raise ValueError("Only a reserved request may be settled")
            record["actual_usd"] = str(actual)
            record["status"] = "settled"
            if actual > _amount(record["reserved_usd"]):
                state["halt_reason"] = "Provider charge exceeded the reserved upper bound"
                violated = True
            self._write(state)
        if violated:
            raise BudgetExceeded("Provider charge exceeded the reserved upper bound")

    def settle_rejected(self, request_id: str, *, basis: str) -> None:
        """Settle an attempt the provider rejected with an HTTP error before any generation at $0.

        Only for a backend whose published rule says such a response is not billed; `basis`
        names the status, the rule and its source. Everything else stays `unknown_capped`.
        """
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("The documented basis for a $0 rejection is required")
        with self._locked() as state:
            record = state["requests"].get(request_id)
            if record is None or record["status"] != "reserved":
                raise ValueError("Only a reserved request may be settled")
            record["actual_usd"] = "0"
            record["status"] = "settled"
            record["zero_cost_basis"] = basis
            self._write(state)

    RECONCILE_BASES = ("process_exited_no_provider_receipt",
                       "process_exited_saved_response_without_cost")

    def reconcile_unknown_at_bound(self, request_id: str, *,
                                   basis: str = "process_exited_no_provider_receipt") -> None:
        """Conservatively close one lost receipt while retaining its full reservation.

        Call only after independently verifying the request process has exited and
        no durable provider receipt exists. The charge stays unknown; this neither
        invents an actual bill nor frees budget for another request.
        """
        if basis not in self.RECONCILE_BASES:
            raise ValueError("Unrecognized reconciliation basis")
        with self._locked() as state:
            record = state["requests"].get(request_id)
            if record is None or record["status"] != "unknown":
                raise ValueError("Only an unknown-cost request may be reconciled at its bound")
            if state["halt_reason"] != LEGACY_UNKNOWN_HALT:
                raise BudgetExceeded("Another accounting halt must be resolved independently")
            if self._committed(state) > _amount(state["limit_usd"]):
                raise BudgetExceeded("Conservative charges exceed the approved budget")
            record["status"] = "unknown_capped"
            record["reconciliation"] = f"{basis}; full reservation retained"
            if not any(r["status"] == "unknown" for r in state["requests"].values()):
                state["halt_reason"] = None
            self._write(state)

    def reconcile_orphan_at_bound(self, request_id: str) -> None:
        """Close a reservation whose issuing process was killed mid-request.

        Call only after verifying that process no longer exists and no durable
        provider receipt was written. The full reservation stays counted and the
        charge stays unknown; this never frees budget or invents a bill, and it
        does not touch any halt (none was raised: the process died first).
        """
        with self._locked() as state:
            record = state["requests"].get(request_id)
            if record is None or record["status"] != "reserved":
                raise ValueError("Only a reserved request can be closed as an orphan")
            record["status"] = "unknown_capped"
            record["reconciliation"] = ("process_killed_in_flight_no_provider_receipt; "
                                        "full reservation retained")
            self._write(state)

    def release_bound_violation(self, *, reason: str) -> list[str]:
        """Clear a 'charge exceeded its reservation' stop after the bound was fixed.

        Every violating request must already be settled at its actual provider
        charge (nothing is estimated or forgiven), nothing may be in flight, and
        the reason (the bound fix) is recorded with the violating ids.
        """
        from datetime import datetime, timezone
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("An explicit reason is required")
        with self._locked() as state:
            if state["halt_reason"] != "Provider charge exceeded the reserved upper bound":
                raise ValueError("No reservation-violation stop is active")
            if any(r["status"] in ("reserved", "unknown") for r in state["requests"].values()):
                raise ValueError("Resolve in-flight and unknown requests first")
            violating = sorted(k for k, r in state["requests"].items()
                               if r["actual_usd"] is not None and _amount(r["actual_usd"]) > _amount(r["reserved_usd"]))
            if not violating or self._committed(state) > _amount(state["limit_usd"]):
                raise ValueError("Violation not found or conservative total exceeds the cap")
            state["halt_reason"] = None
            state.setdefault("operator_events", []).append({
                "action": "release_bound_violation", "reason": reason, "violating": violating,
                "at": datetime.now(timezone.utc).isoformat()})
            self._write(state)
        return violating

    def settle_bound(self, request_id: str, upper_usd: str, *, basis: str) -> None:
        """Retain a verified tariff bound, explicitly not a provider USD invoice.

        reserved_usd remains the counted bound for old immutable readers too;
        preserve its original value and never populate actual_usd with estimates.
        """
        upper = _amount(upper_usd)
        if not isinstance(basis,str) or not basis:
            raise ValueError('A verified tariff evidence identity is required')
        violated=False
        with self._locked() as state:
            record=state['requests'].get(request_id)
            if record is None or record['status']!='reserved':
                raise ValueError('Only an active reserved request can receive a tariff bound')
            original=record['reserved_usd']
            record.update(original_reserved_usd=original,reserved_usd=str(upper),
                          actual_usd=None,status='tariff_capped',cost_basis=basis)
            if upper>_amount(original):
                state['halt_reason']='Measured tariff bound exceeded the reserved upper bound'
                violated=True
            self._write(state)
        if violated:raise BudgetExceeded('Measured tariff bound exceeded its reservation')

    def record(self, request_id: str) -> dict | None:
        """One request's record, without the ledger lock or the totals; a copy."""
        with self._mat_lock:
            self._refresh()
            if self._mat is None:
                return None
            found = dict.get(self._mat["state"]["requests"], request_id)
        return dict(found) if found is not None else None

    def snapshot(self) -> dict:
        """State plus running totals. The records are the ledger's own: read them, do not edit."""
        with self._locked() as state:
            t, q = self._mat["totals"], self._total
            top = {k: state[k] for k in _TOP if k in state}
            return {**top, "halt_reason": state.get("halt_reason"),
                    "requests": dict(state["requests"]), "committed_usd": str(q(0)),
                    "unknown_capped_n": t[2], "unknown_capped_usd": str(q(3)),
                    "tariff_capped_n": t[4], "in_flight_usd": str(q(6)),
                    "tariff_bound_usd": str(q(5)), "settled_usd": str(q(1))}

def cost_summary(snapshot: dict) -> str:
    """"paid $X + N unknown (counted at bound $Y)", plus tariff bounds and in-flight holds."""
    text = (f"paid ${snapshot['settled_usd']} + {snapshot['unknown_capped_n']} unknown "
            f"(counted at bound ${snapshot['unknown_capped_usd']})")
    if snapshot.get("tariff_capped_n"):
        text += f" + {snapshot['tariff_capped_n']} tariff-bounded (${snapshot['tariff_bound_usd']})"
    if Decimal(snapshot.get("in_flight_usd", "0")):
        text += f" + in flight ${snapshot['in_flight_usd']}"
    return text + f"; committed ${snapshot['committed_usd']} of ${snapshot['limit_usd']}"
