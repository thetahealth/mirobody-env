"""An HTTP 200 whose body is an `error` object with no choices is a failed attempt: it
is retried and recorded, never stored as an empty answer."""
import io
import json

import pytest

from haenv import evaluate as ev


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _payload():
    from schema import SolverPayload
    return SolverPayload(case_id="X-01", user_profile={}, prediction_context={
        "target_event_type": "weight_regain"}, longitudinal_data={}, evidence_ledger=[])


@pytest.fixture
def solver(monkeypatch):
    monkeypatch.setattr(ev.time, "sleep", lambda s: None)
    return ev.OpenAICompatSolver("m", "vendor/m", "k", timeout=5, retries=2, max_tokens=100)


def _serve(monkeypatch, bodies):
    it = iter(bodies)
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _Resp(json.dumps(next(it)).encode()))


ERR = {"id": "gen-1", "error": {"message": "temporarily rate-limited upstream", "code": 429}}
OK = {"choices": [{"message": {"content": '{"cited_evidence": []}'}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_error_body_then_success_is_retried(solver, monkeypatch):
    _serve(monkeypatch, [ERR, OK])
    out = solver.solve(_payload())
    assert out._raw_text.startswith("{")
    assert len(out._failed_attempts) == 1
    assert "HTTP 200" in out._failed_attempts[0]["error"]


def test_error_body_every_attempt_leaves_failures_recorded(solver, monkeypatch):
    _serve(monkeypatch, [ERR, ERR, ERR])
    out = solver.solve(_payload())
    assert out._raw_text == ""
    assert len(out._failed_attempts) == 3      # retries + 1, none silently accepted
