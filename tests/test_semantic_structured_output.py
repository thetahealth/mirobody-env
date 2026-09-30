"""The response schema is enforced at request time; a reply is never repaired after judging."""
import json

from haenv.evaluate import OpenAICompatSolver
from haenv.semantic_judge import _request
from test_reasoning_effort_dispatch import Response
from test_semantic_consensus import task


def test_atom_schema_requires_exact_root_items_and_verdict_fields():
    request = _request(task(), 1)
    fmt = request.response_format
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["required"] == ["verdicts"] and schema["additionalProperties"] is False
    atoms = schema["properties"]["verdicts"]
    assert set(atoms["properties"]) == set(task().item_ids)
    assert set(atoms["required"]) == set(task().item_ids)
    assert atoms["additionalProperties"] is False
    value = atoms["properties"]["dx"]
    assert set(value["required"]) == {"label", "reason", "evidence"}
    assert set(value["properties"]["label"]["enum"]) == {"yes", "no", "uncertain"}


def test_provider_payload_carries_schema_and_requires_support(monkeypatch):
    sent = []
    def post(req, timeout):
        sent.append(json.loads(req.data))
        return Response(b'{"choices":[{"message":{"content":"{}"}}]}')
    monkeypatch.setattr("urllib.request.urlopen", post)
    fmt = {"type": "json_schema", "json_schema": {"name": "v", "strict": True,
           "schema": {"type": "object", "properties": {}, "additionalProperties": False}}}
    solver = OpenAICompatSolver("judge", "openai/gpt-6-luna", "test-key", 10,
                                reasoning_effort="high", response_format=fmt)
    solver._post("p")
    assert sent[0]["response_format"] == fmt
    assert sent[0]["provider"]["require_parameters"] is True
    assert sent[0]["reasoning"] == {"effort": "high"}


def test_no_schema_preserves_existing_solver_request(monkeypatch):
    sent = []
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: (
        sent.append(json.loads(req.data)) or Response(b'{"choices":[]}')))
    OpenAICompatSolver("m", "m", "test-key", 10)._post("p")
    assert "response_format" not in sent[0] and "provider" not in sent[0]
