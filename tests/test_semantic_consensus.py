"""Adaptive semantic judging must count independent votes, not cache hits."""
import json

import pytest

from haenv.semantic_judge import JudgeReply, JudgeTask, evaluate_consensus


def task():
    return JudgeTask(run_id="run-a", item_ids=("dx", "reason"),
                     prompt="Judge the anonymous answer under fixed rubric v1.",
                     answer_text="Possible A. Evidence E1 supports A.",
                     judge_model="test-judge", rubric_version="v1")


def reply(dx="yes", reason="yes"):
    return json.dumps({"verdicts": {k: {"label": v, "reason": "rubric applied",
                                        "evidence": ["Possible A."]}
                                   for k, v in (("dx", dx), ("reason", reason))}})


def dispatcher(outputs):
    calls = []
    def call(request):
        calls.append(request)
        value = outputs[len(calls) - 1]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, JudgeReply):
            return value
        return JudgeReply(raw=value, cached=False, usage=None)
    return calls, call


def test_two_agree_calls_exactly_twice():
    calls, call = dispatcher([reply(), reply()])
    result = evaluate_consensus(task(), call)
    assert result["status"] == "resolved"
    assert result["verdicts"] == {"dx": "yes", "reason": "yes"}
    assert len(calls) == 2
    assert calls[0].prompt == calls[1].prompt
    assert calls[0].sample_id != calls[1].sample_id
    assert [r.index for r in calls] == [1, 2]
    assert all(r.fresh for r in calls)
    assert result["first_pair_disagreement"] is False


def test_parallel_first_pair_preserves_index_order_and_votes():
    calls, call = dispatcher([reply(), reply()])
    result = evaluate_consensus(task(), call, parallel_first_pair=True)
    assert result["status"] == "resolved"
    assert [r.index for r in calls] == [1, 2]
    assert [s["index"] for s in result["samples"]] == [1, 2]


def test_budget_stop_propagates_without_creating_fake_failed_votes():
    from haenv.semantic_budget import BudgetExceeded
    calls, call = dispatcher([reply(), BudgetExceeded("cap")])
    persisted = []
    with pytest.raises(BudgetExceeded):
        evaluate_consensus(task(), call, on_sample=persisted.append)
    assert len(calls) == 2
    assert len(persisted) == 1 and persisted[0]["status"] == "ok"


def test_one_disagreement_calls_third_without_exposing_votes():
    calls, call = dispatcher([reply(reason="no"), reply(), reply()])
    result = evaluate_consensus(task(), call)
    assert len(calls) == 3
    assert len({r.prompt for r in calls}) == 1
    assert len({r.sample_id for r in calls}) == 3
    assert result["verdicts"] == {"dx": "yes", "reason": "yes"}
    assert result["first_pair_disagreement"] is True


def test_uncertain_is_neither_zero_nor_a_valid_majority():
    calls, call = dispatcher([reply(reason="uncertain"), reply(reason="uncertain")])
    result = evaluate_consensus(task(), call)
    assert len(calls) == 2
    assert result["status"] == "unresolved"
    assert result["verdicts"] == {"dx": "yes", "reason": None}
    assert result["unresolved"] == ["reason"]


def test_three_different_verdicts_leave_that_item_unresolved():
    calls, call = dispatcher([reply(reason="yes"), reply(reason="no"), reply(reason="uncertain")])
    result = evaluate_consensus(task(), call)
    assert len(calls) == 3 and result["verdicts"]["reason"] is None


@pytest.mark.parametrize("bad", [
    '{"verdicts":{"dx":{"label":"yes"}}}',
    reply().replace('"yes"', '0.9', 1),
    reply().replace('"yes"', 'true', 1),
    reply().replace('"yes"', '"probably"', 1),
    reply().replace('"Possible A."', '"fabricated quote"', 1),
    '{"verdicts":{},"verdicts":{}}',
    "```json\n" + reply() + "\n```",
])
def test_invalid_schema_or_fabricated_evidence_cannot_supply_a_vote(bad):
    calls, call = dispatcher([bad, reply(), reply()])
    result = evaluate_consensus(task(), call)
    assert len(calls) == 3
    assert result["samples"][0]["status"] == "invalid"
    assert result["status"] == "resolved"
    assert result["first_pair_disagreement"] is None


def test_cache_hits_are_not_independent_votes():
    cached = JudgeReply(raw=reply(), cached=True, usage=None)
    calls, call = dispatcher([cached, cached, reply()])
    result = evaluate_consensus(task(), call)
    assert len(calls) == 3 and result["status"] == "unresolved"
    assert result["verdicts"] == {"dx": None, "reason": None}
    assert all(s["status"] == "invalid" for s in result["samples"][:2])


def test_failed_requests_are_recorded_without_exception_secrets():
    calls, call = dispatcher([RuntimeError("API_KEY=DO_NOT_PRINT"), reply(), reply()])
    result = evaluate_consensus(task(), call)
    assert len(calls) == 3
    assert result["samples"][0]["status"] == "call_error"
    assert "DO_NOT_PRINT" not in json.dumps(result)


def test_resume_reuses_distinct_persisted_samples_without_new_requests():
    _, call = dispatcher([reply(), reply()])
    done = evaluate_consensus(task(), call)
    calls, forbidden = dispatcher([])
    assert evaluate_consensus(task(), forbidden, prior_samples=done["samples"]) == done
    assert not calls


def test_resume_only_requests_missing_third_vote():
    _, call = dispatcher([reply(), reply(reason="no"), reply()])
    complete = evaluate_consensus(task(), call)
    calls, resume = dispatcher([reply()])
    result = evaluate_consensus(task(), resume, prior_samples=complete["samples"][:2])
    assert len(calls) == 1 and calls[0].index == 3
    assert result == complete


def test_duplicate_or_cross_task_samples_rejected():
    _, call = dispatcher([reply(), reply()])
    done = evaluate_consensus(task(), call)
    _, forbidden = dispatcher([])
    with pytest.raises(ValueError):
        evaluate_consensus(task(), forbidden, prior_samples=[done["samples"][0]] * 2)
    altered = JudgeTask(run_id="run-a", item_ids=("dx", "reason"),
                        prompt="Changed rubric", answer_text=task().answer_text,
                        judge_model="test-judge", rubric_version="v1")
    with pytest.raises(ValueError):
        evaluate_consensus(altered, forbidden, prior_samples=done["samples"])


def test_no_item_does_not_pass_vacuously():
    with pytest.raises(ValueError):
        JudgeTask(run_id="run-a", item_ids=(), prompt="p", answer_text="a",
                  judge_model="test-judge", rubric_version="v1")
