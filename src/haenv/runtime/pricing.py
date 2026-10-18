"""Verified provider prices, accounted-call failures and durable receipt helpers.

Independent of any judging request, reply, rubric or parser.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

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
                    settlement: dict | None = None, route: dict | None = None) -> None:
    """Exception class, timing and transport evidence; never the message (it can carry secrets)."""
    path = receipt.with_suffix(".failed.json")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump({"request": identity, "route": route or {}, "error_class": type(error).__name__,
                   "started_at": started_at,
                   "failed_at": datetime.now(timezone.utc).isoformat(),
                   "seconds": round(time.perf_counter() - started, 3), **(evidence or {}),
                   "settlement": settlement or {"kind": "unknown_capped",
                                                "reason": "dispatch_failed_no_receipt"}},
                  file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(file.name, path)


def carry_receipts(source: Path, out: Path) -> dict:
    """Bring a superseded run's durable receipts into `out/receipts` so its votes replay.

    Never overwrites. A receipt is named by its sample id (task fingerprint and vote), so a
    changed task never finds a carried receipt and is bought as new; a same-id receipt whose
    recorded request differs (model, effort, schema, prompt hash) is refused by `PricedJudge`.
    """
    source, dest = Path(source).resolve() / "receipts", Path(out).resolve() / "receipts"
    if not source.is_dir():
        raise ValueError(f"{source} holds no receipts")
    if source == dest:
        raise ValueError("A run cannot carry its own receipts")
    dest.mkdir(parents=True, exist_ok=True)
    copied = present = 0
    for path in sorted(source.glob("*.json")):
        target = dest / path.name
        if target.exists():
            present += 1
            continue
        with tempfile.NamedTemporaryFile(mode="wb", dir=dest, delete=False) as file:
            file.write(path.read_bytes())
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(file.name, target)            # fails instead of replacing a receipt written meanwhile
        except FileExistsError:
            present += 1
        else:
            copied += 1
        finally:
            os.unlink(file.name)
    entry = {"from": str(source.parent), "copied": copied, "already_present": present,
             "at": datetime.now(timezone.utc).isoformat()}
    with (Path(out).resolve() / "receipts-carried.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
