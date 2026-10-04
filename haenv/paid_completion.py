"""Common request accounting with durable unmodified completion and cost receipts.

No client is constructed and no network is used until an explicit dispatch
callback is supplied. This module does not set sampling/effort or repair output.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import threading

from .paid_slots import persist_json
from .semantic_budget import BudgetExceeded, _amount

#: The in-flight marker of the request this thread is sending (set only around `dispatch`).
_INFLIGHT = threading.local()


def note_generation(generation_id) -> None:
    """Called by a transport as soon as the provider's generation id is known (first SSE chunk).

    Writes `<receipt>.inflight.json` so a request killed mid-stream can still be settled
    from the provider's generation record. Never raises into the transport.
    """
    marker = getattr(_INFLIGHT, "marker", None)
    if marker is None or not isinstance(generation_id, str) or marker.get("generation_id"):
        return
    marker["generation_id"] = generation_id
    try:
        AccountedCompletion._persist(marker["path"], {k: v for k, v in marker.items() if k != "path"})
    except OSError:
        pass


def is_quota_exhausted(evidence: dict) -> bool:
    """The relay's "this key's quota is used up" refusal (the markers the key pool uses)."""
    from .call_log import _is_exhausted
    return _is_exhausted(" ".join(str(evidence.get(k) or "") for k in
                                  ("error_body_head", "error_message", "error_metadata_head")))


class UnknownCostAttemptFailure(RuntimeError):
    """A dispatch failed without an answer; its charge is counted at its bound.

    An ordinary failed attempt (never BudgetExceeded): the solver's retry loop
    continues under a new request identity. The failed id is never re-bought.
    """


class ReceiptSequenceMismatch(BudgetExceeded):
    """This request id already belongs to a different request of the same cell.

    Request ids number a cell's paid calls in order. A resumed run reuses saved slice and
    round answers without calling the paid layer, so its numbering can run behind the run
    that wrote the receipts. The caller moves on to the next id; the occupied id is never
    re-bought or overwritten, and its charge stays as recorded. A `BudgetExceeded`, so a
    caller that does not step over ids stops exactly as before.
    """


class ReconciledRequestFailure(RuntimeError):
    """An earlier attempt failed without an answer and was reconciled at its bound.

    Replaying it raises an ordinary failed-attempt error (never BudgetExceeded),
    so the solver's existing retry loop continues with a new request identity,
    exactly as it would have without the accounting stop. It is never re-bought.
    """


def _today() -> str:
    from datetime import date
    return date.today().isoformat()


ZERO_COMPLETION_SOURCE = "https://openrouter.ai/docs/guides/features/zero-completion-insurance"


def zero_completion_bill(response, missing: Exception) -> dict:
    """An error body that OpenRouter records no generation for is not billed.

    Only for a response that is an error object with no choices, and only when
    every generation lookup returned 404 (`NoGenerationRecord`). Anything else
    re-raises, so the cost stays unknown. The receipt names its evidence; the
    final audit re-queries each id.
    """
    if not isinstance(missing, NoGenerationRecord):
        raise missing
    if not (isinstance(response, dict) and response.get("error") and not response.get("choices")):
        raise missing
    error = response["error"] if isinstance(response["error"], dict) else {}
    return {"kind": "openrouter_error_without_generation_record", "generation_id": missing.generation_id,
            "actual_usd": "0", "error_code": error.get("code"),
            "error_type": (error.get("metadata") or {}).get("error_type"),
            "lookups_404": missing.lookups, "waited_s": missing.waited_s,
            "basis": "GET /api/v1/generation answered 404 for every lookup; OpenRouter Zero "
                     "Completion Insurance: failed requests are not billed", "source": ZERO_COMPLETION_SOURCE}


#: Backends whose own documentation says an HTTP error response is not billed. Only these settle
#: a rejected attempt at $0; every other backend keeps the attempt at its bound (unknown_capped).
#: Read 2026-09-29.
#:   openrouter: docs/api/reference/errors-and-debugging, "Pre-stream errors": errors "before any
#:     tokens are sent" come back as an HTTP 4xx/5xx status; once the provider accepts, OpenRouter
#:     sends 200 and "an HTTP status is final once sent", so a >= 400 status means no token reached
#:     the caller. Zero Completion Insurance: a response with "zero completion tokens" or an error
#:     is not billed for prompt, completion or reasoning tokens, "even if the underlying provider
#:     charges for prompt processing".
#:   google: gemini-api/docs/billing FAQ "Am I charged for failed requests?": "If your request fails
#:     with a 400 or 500 error, you won't be charged for the tokens used." Only those two statuses:
#:     the page says nothing about 429/503/504.
#:   dashscope: the model-pricing page states "请求失败不产生任何费用" only for image/video/world
#:     models, not for text models such as qwen3.7-flash => not covered.
#:   relay: no published billing rule for rejected requests => not covered.
HTTP_ERROR_UNBILLED = {
    "openrouter": {"statuses": None, "sources": [
        "https://openrouter.ai/docs/api/reference/errors-and-debugging#pre-stream-errors",
        ZERO_COMPLETION_SOURCE], "read_on": "2026-09-29",
        "rule": "HTTP status >= 400 is only sent before any token; zero-completion responses are not billed"},
    "google": {"statuses": frozenset({400, 500}), "sources": [
        "https://ai.google.dev/gemini-api/docs/billing"], "read_on": "2026-09-29",
        "rule": "If your request fails with a 400 or 500 error, you won't be charged for the tokens used."},
}
ERROR_BODY_CHARS = 500
#: Cost receipts priced from a tariff rather than invoiced: a published tariff
#: (`native_accounting`) or a price declared in the configuration (`solver_accounting`).
#: The ledger books them as bounds (`settle_bound`, status `tariff_capped`).
TARIFF_RECEIPT_KINDS = ("published_tariff_upper_bound", "declared_price")


def _redact(text: str) -> str:
    import re
    text = re.sub(r"(?i)(bearer\s+)[^\s\"',]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)([?&](?:key|api_key|token)=)[^&\s\"']+", r"\1[redacted]", text)
    return re.sub(r"\b(?:sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,})", "[redacted]", text)


def wire_evidence(error: BaseException) -> dict:
    """What the transport saw before `error`: status, error body head, whether a response started.

    `urllib.error.HTTPError` is raised by `urlopen` on a >= 400 status line, before the caller
    reads a single body byte, so no generation was received on that path. Other failures carry
    the progress the solver recorded (`haenv_wire`), or `None` when unknown.
    """
    import urllib.error
    wire = dict(getattr(error, "haenv_wire", None) or {})
    out = {"http_status": wire.get("http_status"), "response_started": wire.get("response_started"),
           "bytes_received": wire.get("bytes_received"), "generation_id": wire.get("generation_id"),
           "error_body_head": None, "retry_after": None}
    if isinstance(error, urllib.error.HTTPError):
        body = getattr(error, "haenv_error_body", None)
        if body is None:
            try:
                body = error.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                body = None
        out.update(http_status=error.code, response_started=False, bytes_received=0,
                   generation_id=None,
                   error_body_head=_redact(body)[:ERROR_BODY_CHARS] if isinstance(body, str) else None)
        # Parsed from the whole body (the stored head may be cut mid-JSON).
        parsed = _error_object(body)
        out["error_body_is_error_object"] = parsed is not None
        if isinstance(parsed, dict):
            out["error_code"] = parsed.get("code")
            out["error_message"] = _redact(str(parsed.get("message")))[:200]
            meta = json.dumps(parsed.get("metadata") or {}, ensure_ascii=False)
            out["error_metadata_head"] = _redact(meta)[:300]
        elif parsed is not None:
            out["error_message"] = _redact(str(parsed))[:200]
        try:
            out["retry_after"] = (error.headers or {}).get("Retry-After")
        except Exception:  # noqa: BLE001
            pass
    return out


def _error_object(body):
    try:
        value = json.loads(body or "")
    except ValueError:
        return None
    error = value.get("error") if isinstance(value, dict) else None
    return error if isinstance(error, dict) or isinstance(error, str) else None


def is_rate_limited(evidence: dict) -> bool:
    """HTTP 429, or an error body saying the (upstream) provider rate-limited the request."""
    if evidence.get("http_status") == 429 or evidence.get("error_code") == 429:
        return True
    raw = str(evidence.get("error_metadata_head") or "").lower()
    return "rate-limited" in raw or "rate limited" in raw


def unbilled_rejection(backend: str, evidence: dict) -> dict | None:
    """The documented $0 basis for a rejected attempt, or None (cost stays unknown).

    All must hold: an HTTP status >= 400, a parseable error body, no response body started and
    no byte received, and a backend rule in `HTTP_ERROR_UNBILLED` covering that status.
    """
    rule = HTTP_ERROR_UNBILLED.get(backend)
    status = evidence.get("http_status")
    if rule is None or type(status) is not int or status < 400:
        return None
    if rule["statuses"] is not None and status not in rule["statuses"]:
        return None
    if evidence.get("response_started") is not False or evidence.get("bytes_received") != 0:
        return None
    if evidence.get("error_body_is_error_object") is not True:
        return None
    return {"kind": "http_error_before_generation", "actual_usd": "0", "backend": backend,
            "http_status": status, "error_message": evidence.get("error_message"),
            "error_code": evidence.get("error_code"),
            "generation_started": False, "rule": rule["rule"], "sources": rule["sources"],
            "rule_read_on": rule["read_on"]}


class RejectedBeforeGeneration(RuntimeError):
    """The provider answered with an HTTP error before any generation; settled at $0.

    An ordinary failed attempt (never BudgetExceeded). `rate_limited` marks a 429 the
    request accountant may back off and retry under a new request identity.
    """

    def __init__(self, message: str, *, status=None, rate_limited=False, retry_after=None,
                 replayed=False, quota_exhausted=False):
        super().__init__(message)
        self.status, self.rate_limited = status, rate_limited
        self.retry_after, self.replayed = retry_after, replayed
        self.quota_exhausted = quota_exhausted


def same_request(saved, identity: dict, upstream_declared=frozenset()) -> bool:
    """A saved receipt's request is this request: same id, same prompt, same answer view.

    Receipts written before the identity split carry full wire settings (backend,
    endpoint, retries, ...); those are projected to the answer view, so a receipt stays
    replayable after only the route changed.

    `upstream_declared`: `(from, to)` pairs the batch manifest recorded as upstream
    declarations (an `<backend>:unpinned` upstream later declared as `to`). A receipt
    whose upstream is `from` is the same request as one whose upstream is `to`; any
    other upstream difference is a different request.
    """
    from .transport import answer_view
    if not isinstance(saved, dict) or not isinstance(saved.get("settings"), dict):
        return False
    if (saved.get("request_id") != identity["request_id"]
            or saved.get("prompt_sha256") != identity["prompt_sha256"]):
        return False
    a, b = answer_view(saved["settings"]), answer_view(identity["settings"])
    if a == b:
        return True
    was, now = a.get("upstream"), b.get("upstream")
    return (str(was).endswith(":unpinned") and (was, now) in upstream_declared
            and {**a, "upstream": now} == b)


class AccountedCompletion:
    """Reserve before one HTTP attempt and durably record its unmodified receipt.

    `settings` must describe the actual wire payload, not the caller's preferred
    settings. Construction/integration must disable hidden provider/key retries.
    An unknown cost is counted at the reservation (`unknown_capped`) and the run
    continues; a saved answer stays valid. Resuming the same request never buys a
    new answer; changing its prompt or settings requires a distinct request identity.
    """

    def __init__(self, prices, ledger, receipts: Path, *, backend="openrouter", cost_reader=None,
                 generation_lookup=None, usage_probe=None, sleep=None, clock=None,
                 upstream_declared=frozenset()):
        self.prices, self.ledger = prices, ledger
        #: (from, to) upstream declarations recorded in the batch manifest (`same_request`)
        self.upstream_declared = frozenset(upstream_declared)
        if backend != "openrouter" and cost_reader is None:
            raise ValueError("Native routes require a verified cost receipt reader")
        self.backend, self.cost_reader = backend, cost_reader
        # OpenRouter only: exact per-generation cost when a stream lost its usage frame.
        self.generation_lookup = generation_lookup
        self.receipts = Path(receipts)
        self.receipts.mkdir(parents=True, exist_ok=True)
        # relay only: key -> used quota (free GET). Read before and after a quota-exhausted
        # refusal; an unchanged value is the evidence that settles the refusal at $0.
        self.usage_probe = usage_probe
        self._sleep, self._clock = sleep, clock

    def request(self, request_id: str, prompt: str, settings: dict, dispatch, *,
                attempt: dict | None = None, cell: dict | None = None) -> dict:
        if settings.get("model") != self.prices.model:
            raise ValueError("Request model does not match the verified price schedule")
        if settings.get("backend") != self.backend:
            raise ValueError("Cost reader backend does not match the actual route")
        # Schema bytes are additional input, whereas model/effort/max_tokens are
        # transport settings (already covered by the envelope allowance).
        fmt = settings.get("response_format")
        price_input = prompt + (json.dumps(fmt, ensure_ascii=False) if fmt is not None else "")
        bound = self.prices.upper_bound(price_input, settings["max_tokens"])
        from .transport import answer_view, route_view
        # Receipt identity = what the model saw and how it was asked; the route is recorded
        # beside it (`route`) and never has to match on replay.
        identity = {"request_id": request_id, "settings": answer_view(settings),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        route = route_view(settings)
        key = hashlib.sha256(request_id.encode()).hexdigest()
        receipt = self.receipts / f"{key}.json"
        # Independent cells may run concurrently. Same-ID callers serialize here
        # so only the first can dispatch, including across separate processes.
        with (self.receipts / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                from .paid_slots import request_slot
                with request_slot(self.ledger.path, role="solver"):
                    return self._request_locked(request_id, prompt, dispatch, bound, identity,
                                                receipt, attempt or {}, route, cell)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    _persist = staticmethod(persist_json)

    def _unchanged_usage_rejection(self, key, before, evidence, meta) -> dict | None:
        """relay quota-exhausted refusal: $0 only when the key's used quota did not move.

        The key's usage is read (free GET) right before the attempt and again after the
        refusal; equal readings are the evidence that nothing was deducted.
        Any other outcome (moved, unreadable) keeps the attempt at its bound.
        """
        try:
            after = self.usage_probe(key)
        except Exception:  # noqa: BLE001
            return None
        if after != before or type(evidence.get("http_status")) is not int or evidence["http_status"] < 400:
            return None
        return {"kind": "relay_quota_exhausted_usage_unchanged", "actual_usd": "0",
                "backend": self.backend, "http_status": evidence["http_status"],
                "error_message": evidence.get("error_message"), "usage_before": str(before),
                "usage_after": str(after), "key_ordinal": meta.get("key_ordinal"),
                "rule": f"key {meta.get('key_ordinal')} used quota {before} before and {after} after "
                        "the quota-exhausted refusal: nothing was deducted",
                "sources": ["relay /api/usage/token before/after readings (this receipt)"],
                "rule_read_on": _today()}

    @staticmethod
    def _rejected_before_generation(receipt: Path, prior: dict):
        """Replay of an attempt settled at $0 as an HTTP rejection: the same ordinary failure."""
        failed = receipt.with_suffix(".failed.json")
        if prior.get("status") != "settled" or prior.get("actual_usd") != "0" or not failed.is_file():
            return None
        saved = json.loads(failed.read_text())
        if (saved.get("settlement") or {}).get("kind") not in ("http_error_before_generation",
                                                             "relay_quota_exhausted_usage_unchanged"):
            return None
        status = saved.get("http_status")
        return RejectedBeforeGeneration(f"HTTP {status} before generation; settled at $0 (replayed)",
                                        status=status, rate_limited=is_rate_limited(saved),
                                        replayed=True, quota_exhausted=is_quota_exhausted(saved))

    def _book_from_receipt(self, request_id, bound, identity, receipt: Path) -> dict:
        """A durable receipt exists and the ledger has no record of it: never re-buy. Book it
        (`BudgetLedger.book_replayed`) and let the replay path settle it from the receipt:
        a saved answer from its cost evidence, a $0 rejection from its recorded basis, any
        other failed attempt at its bound."""
        failed = receipt.with_suffix(".failed.json")
        saved = json.loads((receipt if receipt.is_file() else failed).read_text())
        if not same_request(saved.get("request"), identity, self.upstream_declared):
            raise ReceiptSequenceMismatch("Receipt identity differs from the requested payload")
        self.ledger.book_replayed(request_id, str(bound), zero_basis=self._zero_basis(bound))
        settlement = saved.get("settlement") or {}
        if not receipt.is_file() and settlement.get("kind") in (
                "http_error_before_generation", "relay_quota_exhausted_usage_unchanged"):
            self.ledger.settle_rejected(request_id, basis=(
                f"replayed from receipt: HTTP {settlement.get('http_status')} before generation; "
                f"{settlement.get('rule')}"))
        return self.ledger.snapshot()["requests"][request_id]

    def _zero_basis(self, bound):
        """The price schedule's reason a $0 bound is right (a declared price of 0), else None."""
        return getattr(self.prices, "zero_bound_basis", None) if not bound else None

    def _reserve(self, request_id, bound):
        """Reserve at the ledger's cap; at the cap pause (poll) instead of stopping the batch."""
        from .ops.runtime import check_dispatch, for_ledger, ops_log
        check_dispatch(self.ledger.path)
        knobs = for_ledger(self.ledger.path).current()["budget"]
        pause = self.ledger.reserve_or_pause(
            request_id, str(bound), poll_s=float(knobs.get("pause_poll_s", 30)),
            timeout_s=float(knobs.get("pause_timeout_s", 0)), sleep=self._sleep, clock=self._clock,
            on_pause=lambda p: ops_log(self.ledger.path, "budget_paused", **p),
            check=lambda: check_dispatch(self.ledger.path), zero_basis=self._zero_basis(bound))
        if pause is not None:
            ops_log(self.ledger.path, "budget_resumed", **pause)

    def _request_locked(self, request_id, prompt, dispatch, bound, identity, receipt, attempt,
                        route=None, cell=None):
        prior = self.ledger.snapshot()["requests"].get(request_id)
        if prior is None and (receipt.is_file() or receipt.with_suffix(".failed.json").is_file()):
            prior = self._book_from_receipt(request_id, bound, identity, receipt)
        if prior is not None:
            if not receipt.is_file():
                failed = receipt.with_suffix(".failed.json")
                if failed.is_file() and not same_request(json.loads(failed.read_text()).get("request"),
                                                         identity, self.upstream_declared):
                    raise ReceiptSequenceMismatch("Request id holds a failed attempt of another request")
                rejected = self._rejected_before_generation(receipt, prior)
                if rejected is not None:
                    raise rejected
                if prior["status"] == "settled" and prior.get("settled_after_unknown"):
                    # A failed attempt whose bound was later replaced by evidence
                    # (`settle_unknown_later`): still a failed attempt, never re-bought.
                    raise ReconciledRequestFailure("Failed attempt settled afterwards; not reissued")
                if prior["status"] in ("reserved", "unknown"):
                    # This process holds the same-id lock, so the issuing attempt has
                    # ended without a durable answer: count it at its bound, move on.
                    self.ledger.record_unknown(request_id, reason=(
                        "reserved_without_durable_receipt" if prior["status"] == "reserved"
                        else "legacy_unknown_carried"))
                    prior = {**prior, "status": "unknown_capped"}
                if prior["status"] == "unknown_capped":
                    raise ReconciledRequestFailure("Failed attempt with unknown cost; not reissued")
                raise BudgetExceeded("A settled request has no durable receipt; do not reissue")
            saved = json.loads(receipt.read_text())
            if not same_request(saved.get("request"), identity, self.upstream_declared):
                raise ReceiptSequenceMismatch("Receipt identity differs from the requested payload")
            if prior["status"] == "unknown":
                self.ledger.record_unknown(request_id, reason="legacy_unknown_carried")
                prior = {**prior, "status": "unknown_capped"}
            if prior["status"] == "unknown_capped":
                # Saved answer whose missing charge was closed at its full bound: replay
                # it (the caller validates it as before); never settle or re-buy it.
                return saved["response"]
            if prior["status"] not in ("reserved", "settled", "tariff_capped"):
                raise BudgetExceeded("Unknown request cost needs explicit reconciliation")
            response = saved["response"]
        else:
            validator = getattr(self.prices, "validate_dispatch", None)
            if validator is not None:
                try:
                    validator()
                except ValueError:
                    raise BudgetExceeded("Tariff verification expired or invalid; no new request sent") from None
            self._reserve(request_id, bound)
            import time as _time
            from datetime import datetime as _dt, timezone as _tz
            started, started_at = _time.perf_counter(), _dt.now(_tz.utc).isoformat()
            key = attempt.get("api_key")
            meta = {k: v for k, v in attempt.items() if k != "api_key"}
            usage_before = None
            if self.usage_probe is not None and key:
                try:
                    usage_before = self.usage_probe(key)
                except Exception:  # noqa: BLE001 -- no before-reading => no $0 settlement later
                    usage_before = None
            inflight = receipt.with_suffix(".inflight.json")
            _INFLIGHT.marker = {"path": inflight, "request_id": identity["request_id"],
                                "backend": self.backend, "started_at": started_at,
                                "generation_id": None, **meta}
            try:
                response = dispatch(prompt)
                from .transport import served_of
                # `cell` (case, solver) joins the receipt to its grid cell; `served` is who the
                # response says served it. Neither is part of the request identity.
                self._persist(receipt, {"request": identity, "route": route, "response": response,
                                        **({"cell": cell} if cell else {}),
                                        "served": served_of(response),
                                        **({"attempt": meta} if meta else {})})
                _INFLIGHT.marker = None
                inflight.unlink(missing_ok=True)
            except Exception as error:
                marker, _INFLIGHT.marker = _INFLIGHT.marker, None
                # Exception class, timing and transport evidence; the message itself is never
                # stored (it can carry keys/URLs), the error body only redacted and truncated.
                evidence = wire_evidence(error)
                if evidence.get("generation_id") is None and (marker or {}).get("generation_id"):
                    evidence["generation_id"] = marker["generation_id"]
                exhausted = is_quota_exhausted(evidence)
                zero = unbilled_rejection(self.backend, evidence)
                if zero is None and exhausted and usage_before is not None:
                    zero = self._unchanged_usage_rejection(key, usage_before, evidence, meta)
                # A response that started and then failed to parse (non-streaming: keep-alive
                # blank lines, then the connection closed early => `Expecting value`) is a cut
                # transport, like a cut stream: retryable, counted at its bound until settled.
                cut = evidence.get("response_started") is True
                from .transport import pinned_upstream
                failed = {"request": identity, "route": route,
                          **({"cell": cell} if cell else {}),
                          # A refusal under a pin (e.g. the pinned upstream is unavailable) is
                          # this failed attempt; the router was told not to try another one.
                          **({"pinned_upstream": pinned_upstream((route or {}).get("provider"))}
                             if pinned_upstream((route or {}).get("provider")) else {}),
                          "error_class": type(error).__name__,
                          **({"attempt": meta} if meta else {}),
                          **({"transport_cut": ("body_unparseable_after_start"
                                                if isinstance(error, ValueError) else "cut_after_start")}
                             if cut else {}),
                          "started_at": started_at, "failed_at": _dt.now(_tz.utc).isoformat(),
                          "seconds": round(_time.perf_counter() - started, 3), **evidence,
                          "settlement": zero or {"kind": "unknown_capped",
                                                 "reason": "dispatch_failed_no_receipt"}}
                try:
                    self._persist(receipt.with_suffix(".failed.json"), failed)
                    inflight.unlink(missing_ok=True)
                except OSError:
                    zero = None                   # no durable basis => keep the bound
                limited = is_rate_limited(evidence)
                if zero is not None:
                    self.ledger.settle_rejected(request_id, basis=(
                        f"HTTP {zero['http_status']} before generation ({self.backend}); "
                        f"{zero['rule']} [{', '.join(zero['sources'])}, read {zero['rule_read_on']}]"))
                    raise RejectedBeforeGeneration(
                        f"HTTP {zero['http_status']} before generation; settled at $0",
                        status=zero["http_status"], rate_limited=limited,
                        retry_after=evidence.get("retry_after"), quota_exhausted=exhausted) from None
                self.ledger.record_unknown(request_id, reason="dispatch_failed_no_receipt")
                failure = UnknownCostAttemptFailure(
                    f"{type(error).__name__}"
                    f"{' HTTP ' + str(evidence['http_status']) if evidence.get('http_status') else ''}"
                    "; attempt failed, cost counted at its bound")
                failure.rate_limited, failure.retry_after = limited, evidence.get("retry_after")
                failure.quota_exhausted, failure.status = exhausted, evidence.get("http_status")
                failure.stream_cut = cut
                raise failure from None
        try:
            bounded = False
            bill = None
            if self.backend == "openrouter":
                usage = response.get("usage") if isinstance(response, dict) else None
                cost = usage.get("cost") if isinstance(usage, dict) else None
                generation = (response.get("_generation_id") or response.get("id")
                              if isinstance(response, dict) else None)
                if cost is None and self.generation_lookup is not None and isinstance(generation, str):
                    bill_path = receipt.with_suffix(".billing.json")
                    if bill_path.is_file():
                        bill = json.loads(bill_path.read_text())
                        if bill.get("completion_sha256") != hashlib.sha256(receipt.read_bytes()).hexdigest():
                            raise ValueError("Billing receipt is bound to a different completion")
                    else:
                        try:
                            bill = self.generation_lookup(generation)
                        except ValueError as missing:
                            bill = zero_completion_bill(response, missing)
                        if bill.get("generation_id") != generation:
                            raise ValueError("Generation receipt is for another request")
                        bill["completion_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
                        self._persist(bill_path, bill)
                    cost = bill.get("actual_usd")
            else:
                bill_path = receipt.with_suffix(".billing.json")
                if bill_path.is_file():
                    bill = json.loads(bill_path.read_text())
                    if bill.get("completion_sha256") != hashlib.sha256(receipt.read_bytes()).hexdigest():
                        raise ValueError("Billing receipt is bound to a different completion")
                else:
                    if prior is not None and prior["status"] in ("settled", "tariff_capped"):
                        raise ValueError("Settled request is missing its durable billing receipt")
                    bill = self.cost_reader(response)
                    bill["completion_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
                    self._persist(bill_path, bill)
                cost = bill.get("actual_usd")
                if cost is None and bill.get("kind") in TARIFF_RECEIPT_KINDS:
                    cost = bill.get("upper_bound_usd")
                    bounded = True
            if cost is None:
                raise ValueError("No cost")
            actual = _amount(cost)
        except Exception:
            if prior is None or prior["status"] == "reserved":
                # The answer (if any) stays valid; the caller validates it as before.
                self.ledger.record_unknown(request_id, reason="provider_cost_missing")
                return response
            raise BudgetExceeded("A settled receipt has no verified actual cost") from None
        if bounded:
            if prior is None or prior['status']=='reserved':
                self.ledger.settle_bound(request_id,str(actual),basis=bill['basis'])
            elif prior['status']!='tariff_capped' or _amount(prior['reserved_usd'])!=actual:
                raise BudgetExceeded('Tariff receipt differs from its recorded budget bound')
        elif prior is None or prior["status"] == "reserved":
            self.ledger.settle(request_id, str(actual))
        elif _amount(prior["actual_usd"]) != actual:
            raise BudgetExceeded("Provider receipt cost differs from the settled ledger")
        return response


class NoGenerationRecord(ValueError):
    """Every lookup answered 404: OpenRouter holds no billed generation under this id."""

    def __init__(self, generation_id: str, lookups: int, waited_s: float):
        super().__init__("No generation record")
        self.generation_id, self.lookups, self.waited_s = generation_id, lookups, waited_s
