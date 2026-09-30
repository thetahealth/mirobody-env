"""A requested judge effort must be present in the actual provider payload."""
import io
import json

import pytest

from haenv.evaluate import OpenAICompatSolver


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.mark.parametrize("backend,field,value", [
    ("openrouter", "reasoning", {"effort": "high"}),
    ("openai", "reasoning_effort", "high"),
])
def test_high_effort_sent_to_provider(monkeypatch, backend, field, value):
    sent = []
    def post(req, timeout):
        sent.append(json.loads(req.data))
        return Response(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())
    monkeypatch.setattr("urllib.request.urlopen", post)
    solver = OpenAICompatSolver("judge", "openai/gpt-6-luna", "test-key", 10,
                                backend=backend, reasoning_effort="high")
    solver._post("anonymous input")
    assert sent[0][field] == value


def test_unspecified_effort_preserves_existing_payload(monkeypatch):
    sent = []
    def post(req, timeout):
        sent.append(json.loads(req.data))
        return Response(b'{"choices":[{"message":{"content":"{}"}}]}')
    monkeypatch.setattr("urllib.request.urlopen", post)
    OpenAICompatSolver("m", "m", "test-key", 10)._post("p")
    assert not {"reasoning", "reasoning_effort"}.intersection(sent[0])


def test_unrecognized_effort_rejected_before_dispatch():
    with pytest.raises(ValueError, match="reasoning_effort"):
        OpenAICompatSolver("m", "m", "test-key", 10, reasoning_effort="whatever")
