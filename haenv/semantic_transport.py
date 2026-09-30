"""Fresh semantic-judge requests using the shared solver and a verified cost cap."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone

from .semantic_budget import BudgetExceeded, BudgetLedger
from .semantic_judge import JudgeReply, JudgeRequest, reply_from_response


def _rate(value) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Missing or invalid provider price") from exc
    if not number.is_finite() or number < 0:
        raise ValueError("Provider price must be finite and nonnegative")
    return number


@dataclass(frozen=True)
class PriceSchedule:
    model: str
    input_rate: Decimal
    output_rate: Decimal
    request_rate: Decimal
    context_length: int

    @classmethod
    def from_metadata(cls, metadata: dict, expected_model: str, *,
                      require_reasoning: bool = True) -> "PriceSchedule":
        if metadata.get("id") != expected_model:
            raise ValueError("Price metadata does not describe the requested model")
        if require_reasoning and "reasoning" not in metadata.get("supported_parameters", []):
            raise ValueError("Model metadata does not advertise reasoning control")
        prices = metadata.get("pricing") or {}
        if "prompt" not in prices or "completion" not in prices:
            raise ValueError("Input and output prices must be supplied by the provider")
        inputs, outputs = [_rate(prices["prompt"])], [_rate(prices["completion"])]
        for override in prices.get("overrides", []):
            if "prompt" in override:
                inputs.append(_rate(override["prompt"]))
            if "completion" in override:
                outputs.append(_rate(override["completion"]))
        for tier in [prices, *prices.get("overrides", [])]:
            for key in ("input_cache_read", "input_cache_write"):
                if key in tier:
                    inputs.append(_rate(tier[key]))
        # OpenRouter may route to any endpoint; the model-level price is not a bound.
        # When endpoint prices were fetched, the bound is the maximum over all of them.
        requests = [_rate(prices.get("request", "0"))]
        for endpoint in metadata.get("endpoints") or []:
            tier = endpoint.get("pricing") or {}
            for key in ("prompt", "input_cache_read", "input_cache_write"):
                if key in tier:
                    inputs.append(_rate(tier[key]))
            if "completion" in tier:
                outputs.append(_rate(tier["completion"]))
            if "request" in tier:
                requests.append(_rate(tier["request"]))
        context = metadata.get("context_length")
        if type(context) is not int or context <= 0:
            raise ValueError("The model context limit must be known")
        return cls(expected_model, max(inputs), max(outputs), max(requests), context)

    def upper_bound(self, prompt: str, max_output_tokens: int) -> Decimal:
        # UTF-8 bytes upper-bound byte-based input tokenization; reserve extra
        # headroom for the one-message envelope. No tools or search are enabled.
        upper_input = len(prompt.encode("utf-8")) + 2048
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("A positive output cap is required")
        if upper_input + max_output_tokens > self.context_length:
            raise ValueError("Conservative token bound exceeds the model context")
        return upper_input * self.input_rate + max_output_tokens * self.output_rate + self.request_rate


class UnknownCostCallFailure(RuntimeError):
    """The call failed without an answer; its charge is counted at its bound.

    Recorded by the consensus engine as ``call_error`` (not a vote); the run
    continues. The same sample id is never bought again (see below).
    """


class ReconciledCallFailure(RuntimeError):
    """A previously failed request whose unknown charge was reconciled at its bound.

    It has no answer and must never be bought again under the same sample id.
    The consensus engine records it exactly like any other failed call
    (``call_error``: not a vote), so the adaptive third sample may follow.
    """


class RejectedCallFailure(RuntimeError):
    """The provider rejected the call with an HTTP error before any generation; settled at $0.

    Recorded by the consensus engine as ``call_error`` like any failed call; never re-bought.
    """


def _rejected_at_zero(receipt: Path, prior: dict) -> bool:
    failed = receipt.with_suffix(".failed.json")
    if prior.get("status") != "settled" or prior.get("actual_usd") != "0" or not failed.is_file():
        return False
    return (json.loads(failed.read_text()).get("settlement") or {}).get("kind") == "http_error_before_generation"


def failure_receipt(receipt: Path, identity: dict, error: BaseException, started_at: str,
                    started: float, *, evidence: dict | None = None,
                    settlement: dict | None = None) -> None:
    """Exception class, timing and transport evidence; never the message (it can carry secrets)."""
    path = receipt.with_suffix(".failed.json")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump({"request": identity, "error_class": type(error).__name__,
                   "started_at": started_at,
                   "failed_at": datetime.now(timezone.utc).isoformat(),
                   "seconds": round(time.perf_counter() - started, 3), **(evidence or {}),
                   "settlement": settlement or {"kind": "unknown_capped",
                                                "reason": "dispatch_failed_no_receipt"}},
                  file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(file.name, path)


class PricedJudge:
    """One HTTP attempt per independent sample, no cache or hidden key retries.

    Provider `usage.cost` is the settlement authority. A missing or invalid cost
    is counted at the reservation (`unknown_capped`), never a guessed zero or
    estimate, and the vote stays valid; the run continues.
    """

    def __init__(self, solver, prices: PriceSchedule, ledger: BudgetLedger, *,
                 receipts: Path | None = None):
        if solver.model != prices.model or solver.reasoning_effort != "high":
            raise ValueError("Judge must match the priced model and approved high effort")
        if solver.pool is not None:
            raise ValueError("Hidden key-rotation requests cannot escape per-attempt reservations")
        if getattr(solver, "_accounting", None) is not None:
            raise ValueError("Judge solver already has accounting; refuse duplicate reservations")
        self.solver, self.prices, self.ledger = solver, prices, ledger
        self.receipts = receipts
        if receipts is not None:
            receipts.mkdir(parents=True, exist_ok=True)

    def __call__(self, request: JudgeRequest) -> JudgeReply:
        from .paid_slots import request_slot
        with request_slot(self.ledger.path):
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
            self.ledger.book_replayed(request.sample_id, str(bound))
            prior = self.ledger.record(request.sample_id)
        if prior is not None:
            if receipt is None or not receipt.is_file():
                if receipt is not None and _rejected_at_zero(receipt, prior):
                    raise RejectedCallFailure("HTTP error before generation, settled at $0 (replayed)")
                if prior["status"] in ("reserved", "unknown"):
                    # The issuing attempt ended without a durable answer (the run lock
                    # excludes a live one): count it at its bound, never re-buy it.
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
            self.ledger.reserve(request.sample_id, str(bound))
            started = time.perf_counter()
            started_at = datetime.now(timezone.utc).isoformat()
            try:
                self.solver.response_format = request.response_format
                response = self.solver._post(request.prompt)
                if receipt is not None:
                    # Persist before settlement/parsing. An interruption after this
                    # point recovers this exact vote, never purchases another one.
                    with tempfile.NamedTemporaryFile(mode="w", dir=receipt.parent,
                                                     delete=False) as file:
                        json.dump({"request": identity, "response": response,
                                   "transport": {"started_at": started_at,
                                       "received_at": datetime.now(timezone.utc).isoformat(),
                                       "seconds": round(time.perf_counter() - started, 3)}}, file,
                                  ensure_ascii=False)
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(file.name, receipt)
            except Exception as error:
                from .paid_completion import unbilled_rejection, wire_evidence
                evidence = wire_evidence(error)
                zero = unbilled_rejection(getattr(self.solver, "backend", ""), evidence)
                if receipt is not None:
                    try:
                        failure_receipt(receipt, identity, error, started_at, started,
                                        evidence=evidence, settlement=zero)
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
        return reply_from_response(response, usage)
