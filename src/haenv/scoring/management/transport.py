"""Configurable management judges using the existing one-attempt accountant."""
from __future__ import annotations

import time
import hashlib
import json
from datetime import datetime, timezone

from haenv.scoring.management.judge import EmptyJudgeResponse
from haenv.scoring.management.judge import transport_reply
from haenv.runtime.completion import AccountedCompletion, ReconciledRequestFailure, RejectedBeforeGeneration, UnknownCostAttemptFailure
from haenv.runtime.pricing import ReconciledCallFailure, RejectedCallFailure, UnknownCostCallFailure
from haenv.scoring.semantic.judge import JudgeReply
from haenv.runtime.solvers import GoogleSolver
from haenv.runtime.solver_accounting import wire_settings
from haenv.runtime.transport import google_response_format, proxy_for_solver


class ManagementGoogleSolver(GoogleSolver):
    """Keep the solver-under-test's frozen wire bytes unchanged."""
    def _wire_body(self, prompt):
        body = json.loads(super()._wire_body(prompt))
        if self.response_format is not None:
            body["generationConfig"].update(google_response_format(self.response_format))
        return json.dumps(body).encode()


class ConfiguredManagementJudge:
    def __init__(self, solver, prices, ledger, receipts):
        solver.proxy = proxy_for_solver(solver)
        self.solver = solver
        self.receipts = receipts
        self.accountant = AccountedCompletion(prices, ledger, receipts,
            backend=solver.backend, cost_reader=prices.cost_receipt if solver.backend == "google" else None)

    def response(self, request):
        if request.fresh is not True:
            raise ValueError("Management judging requires a fresh request")
        self.solver.response_format = request.response_format
        timing = {}
        def dispatch(prompt):
            started = time.perf_counter()
            timing["started_at"] = datetime.now(timezone.utc).isoformat()
            try:
                return self.solver._post(request.prompt)
            finally:
                timing["seconds"] = round(time.perf_counter() - started, 3)
                timing["received_at"] = datetime.now(timezone.utc).isoformat()
        try:
            return self.accountant.request(request.sample_id, request.prompt, wire_settings(self.solver),
                                           dispatch)
        except RejectedBeforeGeneration as exc:
            raise RejectedCallFailure(type(exc).__name__) from None
        except ReconciledRequestFailure as exc:
            raise ReconciledCallFailure(type(exc).__name__) from None
        except UnknownCostAttemptFailure as exc:
            raise UnknownCostCallFailure(type(exc).__name__) from None
        finally:
            if timing:
                from haenv.scoring.semantic.parallel import atomic_json
                atomic_json(self.receipts / (hashlib.sha256(request.sample_id.encode()).hexdigest() + ".timing.json"),
                            {"sample_id": request.sample_id, **timing})

    def __call__(self, request):
        response = self.response(request)
        return transport_reply(response, response.get("usage"))


class NativeManagementJudge(ConfiguredManagementJudge):
    def __init__(self, solver, prices, ledger, receipts):
        solver.__class__ = ManagementGoogleSolver
        super().__init__(solver, prices, ledger, receipts)

    def __call__(self, request):
        from haenv.runtime.solvers import _google_usage
        response = self.response(request)
        usage = _google_usage(response)
        candidates = response.get("candidates") or []
        if not candidates:
            raise EmptyJudgeResponse(usage=usage)
        if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
            raise ValueError("Google judge did not finish one complete answer")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        raw = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        if not raw.strip():
            raise EmptyJudgeResponse(raw, usage)
        return JudgeReply(raw=raw, cached=False, usage=usage)
