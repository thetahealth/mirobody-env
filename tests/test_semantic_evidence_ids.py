"""Reference IDs avoid copied-quote errors without relaxing exact evidence verification."""
from dataclasses import replace
import json

import pytest

from haenv.semantic_judge import JudgeTask, JudgeReply, evaluate_consensus, parse_verdicts, _request
from haenv.judge_evidence import evidence_catalogue


def task():
    body = json.dumps({"diagnosis": "A", "reason": "seen A"})
    quotes, _ = evidence_catalogue(body)
    return JudgeTask("run", ("dx",), "prompt", body, "judge", "v1", evidence_catalog=quotes)


def test_evidence_id_maps_to_original_excerpt_with_id_provenance():
    t = task()
    raw = '{"verdicts":{"dx":{"label":"yes","reason":"matched","evidence_ids":["e0"]}}}'
    parsed = parse_verdicts(raw, t)
    assert parsed["dx"]["evidence"] == ["A"] and parsed["dx"]["evidence_ids"] == ["e0"]
    result = evaluate_consensus(t, lambda req:JudgeReply(raw, False, None))
    assert result["verdicts"] == {"dx": "yes"}
    assert "evidence-ids" in result["protocol"]
    assert evaluate_consensus(t, lambda r:pytest.fail("reissued"), prior_samples=result["samples"]) == result


def test_unknown_id_never_becomes_valid_evidence():
    with pytest.raises(ValueError, match="unknown_evidence_id"):
        parse_verdicts('{"verdicts":{"dx":{"label":"yes","reason":"r","evidence_ids":["gold0"]}}}', task())


def test_request_schema_only_allows_known_ids_not_free_text_quotes():
    t = task()
    schema = _request(t, 1).response_format["json_schema"]["schema"]
    assert schema["properties"]["verdicts"]["properties"]["dx"] == {"$ref": "#/$defs/atomic_verdict"}
    atom = schema["$defs"]["atomic_verdict"]
    assert set(atom["required"]) == {"label", "reason", "evidence_ids"}
    assert schema["$defs"]["evidence_id"]["enum"] == list(t.evidence_catalog)
    assert "evidence" not in atom["properties"]


def test_protocol_separates_ids_from_legacy_quotes_without_breaking_legacy_reads():
    t = task();legacy = replace(t, evidence_catalog=None)
    assert t.fingerprint != legacy.fingerprint
    assert legacy.protocol == "semantic-consensus-2plus1-v2"


def test_catalogue_cannot_add_a_fabricated_source_quote():
    with pytest.raises(ValueError, match="verbatim"):
        replace(task(), evidence_catalog={"e0":"not in the answer"})
