"""A required-test yes vote must cite a proposal inside the counted proposal cap."""
from dataclasses import asdict
import json

from haenv.judge_evidence import evidence_catalogue
from haenv.semantic_inputs import anonymous_prompt
from haenv.semantic_judge import JudgeReply, JudgeTask, evaluate_consensus
from haenv.semantic_report import apply_proposal_cap
from haenv.semantic_rubric import load_policy, make_rubric


def _task():
    ref = {"kind": "ddx:unified", "diagnosis": "A", "aliases": [], "threads": [], "join_gold": None,
           "required_tests": ["TSH"], "optional_tests": [], "rivals": [], "noop": None,
           "visible_case": {"longitudinal_data": {}}}
    # cap = 1 required + 0 optional + margin; the fixture makes the answer longer than the cap.
    answer = {"differential": [{"diagnosis": "A"}], "tests_to_order": [f"T{i}" for i in range(12)] + ["TSH"]}
    criteria, rubric = make_rubric(ref, answer, load_policy())
    prompt, text = anonymous_prompt(criteria, ref, answer, evidence_ids=True)
    task = JudgeTask("r", tuple(criteria), prompt, text, "j", "v", evidence_catalog=evidence_catalogue(text)[0])
    return task, rubric, criteria


def _pointer_id(task, pointer):
    _, pointers = evidence_catalogue(task.answer_text)
    return next(i for i, p in pointers.items() if p == pointer)


def _consensus(task, cite):
    verdicts = {k: {"label": "yes", "reason": "r", "evidence_ids": []} for k in task.item_ids}
    verdicts["required_test_0"]["evidence_ids"] = cite
    reply = json.dumps({"verdicts": verdicts})
    return evaluate_consensus(task, lambda req: JudgeReply(reply, False, None))


def test_yes_citing_only_beyond_the_cap_becomes_no_and_inside_the_cap_stays():
    task, rubric, criteria = _task()
    cap = len(json.loads(criteria["required_test_0"].split("Specific reference: ", 1)[1])["counted_proposals"])
    assert cap < 13
    beyond = _consensus(task, [_pointer_id(task, "/tests_to_order/12")])
    verdicts, changed = apply_proposal_cap(task, beyond)
    assert verdicts["required_test_0"] == "no" and changed == ["required_test_0"]
    inside = _consensus(task, [_pointer_id(task, "/tests_to_order/0")])
    verdicts, changed = apply_proposal_cap(task, inside)
    assert verdicts["required_test_0"] == "yes" and changed == []
    # A vote citing no proposal field is left to the judge (not changed by this rule).
    other = _consensus(task, [])
    assert apply_proposal_cap(task, other) == (other["verdicts"], [])
