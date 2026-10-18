"""Semantic JudgeRequest/Reply adaptation over runtime pricing and accounting."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from haenv.runtime.completion import begin_inflight, end_inflight, lane, mark_sent, never_sent, unbilled_rejection, wire_evidence
from haenv.runtime.budget import BudgetExceeded, BudgetLedger
from haenv.runtime.pricing import (  # noqa: F401 -- preserve existing semantic callers
    PriceSchedule, ReconciledCallFailure, RejectedCallFailure, UnknownCostCallFailure,
    _rate, _rejected_at_zero, carry_receipts, failure_receipt,
)
from haenv.scoring.semantic.judge import JudgeReply
from haenv.scoring.semantic.judge import JudgeRequest
from haenv.scoring.semantic.judge import reply_from_response


def pause_and_reserve(ledger: BudgetLedger, request_id: str, bound_usd: str, *, cost_key=None) -> None:
    """Reserve one judge request; at the cap pause and retry like a solver request.

    Polling and the longest pause come from the ledger's runtime knobs (`budget:
    pause_poll_s / pause_timeout_s`). A drain refuses at once (`ops.runtime.Draining`).
    Only a reservation still refused when the pause ends raises `BudgetExceeded`.
    """
    from haenv.runtime.ops.runtime import check_dispatch
    from haenv.runtime.ops.runtime import for_ledger
    from haenv.runtime.ops.runtime import ops_log
    check_dispatch(ledger.path)
    knobs = for_ledger(ledger.path).current()["budget"]
    pause = ledger.reserve_or_pause(
        request_id, bound_usd, cost_key=cost_key, poll_s=float(knobs.get("pause_poll_s", 30)),
        timeout_s=float(knobs.get("pause_timeout_s", 0)),
        on_pause=lambda p: ops_log(ledger.path, "budget_paused", role="judge", **p),
        check=lambda: check_dispatch(ledger.path))
    if pause is not None:
        ops_log(ledger.path, "budget_resumed", role="judge", **pause)


class PricedJudge:
    """One HTTP attempt per independent sample, no cache or hidden key retries.

    Provider `usage.cost` is the settlement authority. A missing or invalid cost
    is counted at the reservation (`unknown_capped`), never a guessed zero or
    estimate, and the vote stays valid; the run continues.
    """

    def __init__(self, solver, prices: PriceSchedule, ledger: BudgetLedger, *,
                 receipts: Path | None = None, reserve=None, reply_parser=None):
        if solver.model != prices.model or solver.reasoning_effort != "high":
            raise ValueError("Judge must match the priced model and approved high effort")
        if solver.pool is not None:
            raise ValueError("Hidden key-rotation requests cannot escape per-attempt reservations")
        if getattr(solver, "_accounting", None) is not None:
            raise ValueError("Judge solver already has accounting; refuse duplicate reservations")
        self.solver, self.prices, self.ledger = solver, prices, ledger
        from haenv.runtime.transport import proxy_for_solver
        solver.proxy = proxy_for_solver(solver)
        from haenv.runtime.budget import cost_key
        key = cost_key("judge", getattr(solver, "backend", None), solver.model)
        self.reservation_key = key
        self.reserve = (lambda ledger, rid, bound: pause_and_reserve(ledger, rid, bound, cost_key=key)) if reserve is None else reserve
        # Each protocol owns what counts as an answer, after the receipt and cost
        # are durable. Existing semantic callers retain their exact parser.
        self.reply_parser = reply_from_response if reply_parser is None else reply_parser
        self.receipts = receipts
        if receipts is not None:
            receipts.mkdir(parents=True, exist_ok=True)

    def __call__(self, request: JudgeRequest) -> JudgeReply:
        """Reserve, send under a lane (only the send holds one), settle."""
        return self._accounted_call(request)

    def _accounted_call(self, request: JudgeRequest) -> JudgeReply:
        if request.fresh is not True:
            raise ValueError("Independent judging must request a fresh response")
        schema_text = (json.dumps(request.response_format, ensure_ascii=False)
                       if request.response_format is not None else "")
        bound = self.prices.upper_bound(request.prompt + schema_text, self.solver.max_tokens)
        receipt = (self.receipts / (hashlib.sha256(request.sample_id.encode()).hexdigest() + ".json")
                   if self.receipts is not None else None)
        identity = {"sample_id": request.sample_id, "model": self.solver.model,
                    "reasoning_effort": self.solver.reasoning_effort,
                    "max_tokens": self.solver.max_tokens,
                    "response_format": request.response_format,
                    "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest()}
        prior = self.ledger.record(request.sample_id)
        if prior is None and receipt is not None and receipt.is_file():
            # Durable receipt, no ledger record (rebuilt ledger, copied batch): the vote was
            # bought; book it and replay it, never buy it again or overwrite the receipt.
            if json.loads(receipt.read_text()).get("request") != identity:
                raise BudgetExceeded("Durable receipt does not match this request")
            self.ledger.book_replayed(request.sample_id, str(bound), cost_key=self.reservation_key)
            prior = self.ledger.record(request.sample_id)
        unsent = never_sent(prior, receipt)       # reserved, then killed while waiting for a lane
        if prior is not None and not unsent:
            if receipt is None or not receipt.is_file():
                if receipt is not None and _rejected_at_zero(receipt, prior):
                    raise RejectedCallFailure("HTTP error before generation, settled at $0 (replayed)")
                if prior["status"] in ("reserved", "unknown"):
                    # The issuing attempt ended without a durable answer (the run lock
                    # excludes a live one): count it at its bound, never re-buy it. Its
                    # in-flight marker keeps the generation id for `haenv spend reconcile`.
                    self.ledger.record_unknown(request.sample_id, reason=(
                        "reserved_without_durable_receipt" if prior["status"] == "reserved"
                        else "legacy_unknown_carried"))
                    prior = {**prior, "status": "unknown_capped"}
                if prior["status"] == "unknown_capped":
                    raise ReconciledCallFailure("Failed call with unknown cost; recorded as call error")
                raise BudgetExceeded("Settled sample has no durable receipt")
            saved = json.loads(receipt.read_text())
            if saved.get("request") != identity:
                raise BudgetExceeded("Durable receipt does not match this request")
            response = saved["response"]
            if prior["status"] == "unknown":
                self.ledger.record_unknown(request.sample_id, reason="legacy_unknown_carried")
                prior = {**prior, "status": "unknown_capped"}
        else:
            inflight = receipt.with_suffix(".inflight.json") if receipt is not None else None
            failure = None
            with lane(self.ledger.path, backend=getattr(self.solver, "backend", None),
                      on_unsent=lambda: self.ledger.cancel_unsent(request.sample_id),
                      before=None if unsent and not prior.get("cancelled_before_dispatch") else
                      lambda: self.reserve(self.ledger, request.sample_id, str(bound))):
                started = time.perf_counter()
                started_at = datetime.now(timezone.utc).isoformat()
                if receipt is not None:
                    mark_sent(receipt, request_id=request.sample_id, started_at=started_at,
                              proxy=self.solver.proxy)
                    begin_inflight(inflight, request_id=request.sample_id,
                                   backend=getattr(self.solver, "backend", None),
                                   proxy=self.solver.proxy, started_at=started_at)
                try:
                    self.solver.response_format = request.response_format
                    response = self.solver._post(request.prompt)
                except Exception as error:      # settled below, once the lane is released
                    failure = error
            try:
                if failure is not None:
                    raise failure
                if receipt is not None:
                    # Persist before settlement/parsing. An interruption after this
                    # point recovers this exact vote, never purchases another one.
                    with tempfile.NamedTemporaryFile(mode="w", dir=receipt.parent,
                                                     delete=False) as file:
                        json.dump({"request": identity, "response": response,
                                   "route": {"backend": getattr(self.solver, "backend", "openrouter"),
                                             "proxy": self.solver.proxy},
                                   "transport": {"started_at": started_at,
                                       "received_at": datetime.now(timezone.utc).isoformat(),
                                       "seconds": round(time.perf_counter() - started, 3)}}, file,
                                  ensure_ascii=False)
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(file.name, receipt)
                    end_inflight()
                    inflight.unlink(missing_ok=True)
                    receipt.with_suffix(".sent").unlink(missing_ok=True)
            except Exception as error:
                marker = end_inflight()
                evidence = wire_evidence(error)
                if evidence.get("generation_id") is None and (marker or {}).get("generation_id"):
                    evidence["generation_id"] = marker["generation_id"]
                zero = unbilled_rejection(getattr(self.solver, "backend", ""), evidence)
                if receipt is not None:
                    try:
                        failure_receipt(receipt, identity, error, started_at, started,
                                        evidence=evidence, settlement=zero,
                                        route={"backend": getattr(self.solver, "backend", "openrouter"),
                                               "proxy": self.solver.proxy})
                    except OSError:
                        zero = None               # no durable basis => keep the bound
                else:
                    zero = None
                if zero is not None:
                    self.ledger.settle_rejected(request.sample_id, basis=(
                        f"HTTP {zero['http_status']} before generation ({zero['backend']}); "
                        f"{zero['rule']} [{', '.join(zero['sources'])}, read {zero['rule_read_on']}]"))
                    raise RejectedCallFailure(
                        f"HTTP {zero['http_status']} before generation; settled at $0") from None
                self.ledger.record_unknown(request.sample_id, reason="dispatch_failed_no_receipt")
                if receipt is not None:
                    inflight.unlink(missing_ok=True)
                # Class name only: exception text can carry backend URLs or keys.
                raise UnknownCostCallFailure(
                    f"{type(error).__name__}; call failed, cost counted at its bound") from None
        if not isinstance(response, dict):
            if prior is None or prior["status"] == "reserved":
                self.ledger.record_unknown(request.sample_id, reason="no_response_object")
            raise UnknownCostCallFailure("Provider returned no response object; cost counted at its bound")
        usage = response.get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if prior is not None and prior["status"] == "unknown_capped":
            # A saved answer whose charge is counted at its full bound: the vote
            # exists and its conservative charge is already counted.
            cost = None
        elif cost is None:
            self.ledger.record_unknown(request.sample_id, reason="provider_cost_missing")
        if cost is not None:
            try:
                billed = _rate(cost)
            except ValueError:
                self.ledger.record_unknown(request.sample_id, reason="provider_cost_invalid")
                billed = None
        if cost is not None and billed is not None:
            if prior is None or prior["status"] == "reserved":
                self.ledger.settle(request.sample_id, str(billed))
            elif _rate(prior["actual_usd"]) != billed:
                raise BudgetExceeded("Receipt cost differs from settled ledger")
        return self.reply_parser(response, usage)
