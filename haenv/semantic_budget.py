"""Persistent, process-safe dollar cap for the complete semantic-validation run."""
from __future__ import annotations

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
        self._read_lock = threading.Lock(); self._read_key = None; self._read_requests = {}
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

    @contextmanager
    def _locked(self):
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text()) if self.path.exists() else None
                yield state
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _write(self, state):
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent,
                                         prefix=self.path.name + ".", delete=False) as file:
            temporary = Path(file.name)
            # compact and in one write: the whole ledger is rewritten under the exclusive lock
            file.write(json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.path)

    @staticmethod
    def _committed(state) -> Decimal:
        return sum((_amount(r["actual_usd"] if r["actual_usd"] is not None else r["reserved_usd"])
                    for r in state["requests"].values()), Decimal("0"))

    def reserve(self, request_id: str, maximum_usd: str) -> None:
        maximum = _amount(maximum_usd, positive=True)
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
                "reserved_usd": str(maximum), "actual_usd": None, "status": "reserved"}
            self._write(state)

    def book_replayed(self, request_id: str, maximum_usd: str) -> dict:
        """Book a request whose durable receipt exists but which this ledger never saw (the
        ledger was rebuilt or the batch copied): `reserved` at its bound, marked
        `booked_from_receipt`, for the caller to settle from the receipt's cost evidence.
        No cap check: the money was already spent, refusing the entry would hide it."""
        maximum = _amount(maximum_usd, positive=True)
        with self._locked() as state:
            if request_id in state["requests"]:
                raise ValueError("Request id has already been reserved")
            record = {"reserved_usd": str(maximum), "actual_usd": None, "status": "reserved",
                      "booked_from_receipt": True}
            state["requests"][request_id] = record
            self._write(state)
            return dict(record)

    def reserve_or_pause(self, request_id: str, maximum_usd: str, *, poll_s: float = 30.0,
                         timeout_s: float = 1800.0, sleep=None, clock=None, on_pause=None,
                         check=None) -> dict | None:
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
                self.reserve(request_id, maximum_usd)
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
        """One request's record, without the ledger lock or the totals.

        Every write replaces the file atomically, so a lock-free read sees one whole version;
        the parse is reused until the file is replaced. Returns a copy.
        """
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        key = (st.st_ino, st.st_mtime_ns, st.st_size)
        with self._read_lock:
            if key != self._read_key:
                self._read_requests = json.loads(self.path.read_text())["requests"]
                self._read_key = key
            found = self._read_requests.get(request_id)
        return dict(found) if found is not None else None

    def snapshot(self) -> dict:
        with self._locked() as state:
            zero = Decimal("0")
            committed = unknown_usd = in_flight = tariff_usd = settled = zero
            unknown_n = tariff_n = 0
            for r in state["requests"].values():
                reserved = _amount(r["reserved_usd"])
                actual = _amount(r["actual_usd"]) if r["actual_usd"] is not None else None
                committed += actual if actual is not None else reserved
                if actual is not None:
                    settled += actual
                status = r["status"]
                if status == "unknown_capped":
                    unknown_n += 1; unknown_usd += reserved
                elif status == "tariff_capped":
                    tariff_n += 1; tariff_usd += reserved
                elif status == "reserved":
                    in_flight += reserved
            return {**state, "committed_usd": str(committed),
                    "unknown_capped_n": unknown_n, "unknown_capped_usd": str(unknown_usd),
                    "tariff_capped_n": tariff_n, "in_flight_usd": str(in_flight),
                    "tariff_bound_usd": str(tariff_usd), "settled_usd": str(settled)}

def cost_summary(snapshot: dict) -> str:
    """"paid $X + N unknown (counted at bound $Y)", plus tariff bounds and in-flight holds."""
    text = (f"paid ${snapshot['settled_usd']} + {snapshot['unknown_capped_n']} unknown "
            f"(counted at bound ${snapshot['unknown_capped_usd']})")
    if snapshot.get("tariff_capped_n"):
        text += f" + {snapshot['tariff_capped_n']} tariff-bounded (${snapshot['tariff_bound_usd']})"
    if Decimal(snapshot.get("in_flight_usd", "0")):
        text += f" + in flight ${snapshot['in_flight_usd']}"
    return text + f"; committed ${snapshot['committed_usd']} of ${snapshot['limit_usd']}"
