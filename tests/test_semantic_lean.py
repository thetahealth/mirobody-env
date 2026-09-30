"""Lean semantic-v4 inputs and the disputed-atom third vote."""
import json

import pytest

from haenv.semantic_judge import JudgeReply, JudgeTask, evaluate_consensus
from haenv.semantic_lean import case_facts, cited_tokens, lean_prompt
from haenv.semantic_rubric import REGISTRY_LEAN, load_policy, rubric_atoms


def reference():
    return {"kind": "ddx:unified", "diagnosis": "Axial spondyloarthritis", "aliases": [], "threads": [],
            "join_gold": "unified", "required_tests": ["HLA-B27", "sacroiliac MRI"], "optional_tests": ["CRP"],
            "rivals": [{"name": "mechanical back pain", "discriminator": "sacroiliac MRI"}],
            "noop": {"target": "body_temp", "window": [0, 13], "truth_present": False},
            "visible_case": {"evidence_ledger": [
                {"evidence_id": "EV-1-01", "symptom": "night back pain", "source_timestamp": 10},
                {"evidence_id": "EV-1-02", "symptom": "SECRET-UNCITED-ROW", "source_timestamp": 12}],
                "longitudinal_data": {"weight": [{"ts": 0, "value": 70}, {"ts": 7, "value": 71}],
                                      "steps": [{"ts": 0, "value": 1}, {"ts": 1, "value": 2}]}}}


def answer(extra_tests=0):
    return {"differential": [{"diagnosis": "axial SpA", "rank": 1, "supporting_evidence": ["EV-1-01", "weight"]},
                             {"diagnosis": "strain", "rank": 2, "ruled_out_by": "inflammatory pattern EV-9-99"}],
            "join_type": "unified", "join_reason": "night pain and inflammatory pattern",
            "tests_to_order": ["HLA-B27", "MRI SI joints"] + [f"t{i}" for i in range(extra_tests)],
            "action": {"specific_action": "refer"}}


def build(extra_tests=0):
    policy = load_policy(REGISTRY_LEAN)
    atoms, rubric = rubric_atoms(reference(), answer(extra_tests), policy)
    prompt, body, template = lean_prompt(atoms, reference(), answer(extra_tests), policy,
                                         proposal_cap=rubric["proposal_cap"], n_proposed=rubric["n_proposed"])
    return atoms, rubric, prompt, body, template


def test_lean_prompt_drops_the_uncited_case_and_repeated_proposals():
    atoms, rubric, prompt, body, template = build()
    assert "SECRET-UNCITED-ROW" not in prompt and '"steps"' not in prompt
    assert "night back pain" in prompt and '"weight"' in prompt       # cited row and cited series
    assert "EV-9-99" in prompt.split("CASE FACTS", 1)[1].split("ATOMIC CRITERIA", 1)[0]
    assert "counted_proposals" not in prompt and "PROPOSAL CAP" not in prompt
    definitions = json.dumps(load_policy(REGISTRY_LEAN)["criteria"]["required_test"], ensure_ascii=False)[1:40]
    assert prompt.count(definitions) == 1                                  # each definition once
    assert prompt == template["head"] + "\n".join(template["atoms"].values()) + template["tail"]
    assert list(template["atoms"]) == [a[0] for a in atoms]


def test_cap_is_stated_once_only_when_it_truncates():
    _, rubric, prompt, _, _ = build(extra_tests=10)
    assert rubric["n_proposed"] > rubric["proposal_cap"] and prompt.count("PROPOSAL CAP") == 1


def test_shared_case_prefix_precedes_answer_dependent_content():
    _, _, prompt, _, _ = build()
    order = [prompt.index(k) for k in ("\nCRITERION DEFINITIONS (by type):", "\nREFERENCE (the case",
                                        "\nDATA-AVAILABILITY REFERENCE", "\nCASE FACTS CITED",
                                        "\nATOMIC CRITERIA:", "\nEVIDENCE POINTERS (", "\nBEGIN UNTRUSTED")]
    assert order == sorted(order)


def test_same_atoms_and_groups_as_verbatim_layout():
    from haenv.semantic_rubric import make_rubric
    v3 = load_policy()
    lean = load_policy(REGISTRY_LEAN)
    criteria, rubric3 = make_rubric(reference(), answer(), v3)
    atoms, rubric4 = rubric_atoms(reference(), answer(), lean)
    assert [a[0] for a in atoms] == list(criteria)
    assert {k: v for k, v in rubric3.items() if k != "version"} == {k: v for k, v in rubric4.items() if k != "version"}
    # The lean policy shares v3's texts; only diagnosis_hit differs (outside comorbidity).
    assert {k: t for k, t in lean["criteria"].items() if k != "diagnosis_hit"} == \
        {k: t for k, t in v3["criteria"].items() if k != "diagnosis_hit"}
    assert lean["criteria"]["diagnosis_hit"] == v3["criteria_by_reference_kind"]["ddx:comorbidity"]["diagnosis_hit"]
    assert lean["judge"]["max_tokens"] == v3["judge"]["max_tokens"]


def test_cited_tokens_reads_citation_lists_and_ev_ids_in_text():
    assert cited_tokens(answer()) == ["EV-1-01", "weight", "EV-9-99"]
    facts = case_facts(reference()["visible_case"], answer())
    assert [r["evidence_id"] for r in facts["ledger_rows"]] == ["EV-1-01"]
    assert list(facts["series"]) == ["weight"] and facts["cited_but_not_delivered"] == ["EV-9-99"]


def lean_task(items=("a", "b", "c")):
    head, tail = "HEAD\n", "\nTAIL"
    atoms = {k: f"{k} | test | -" for k in items}
    prompt = head + "\n".join(atoms.values()) + tail
    return JudgeTask("run", tuple(items), prompt, "A", "judge", "v4",
                     atom_template={"head": head, "atoms": atoms, "tail": tail})


def reply(labels):
    return JudgeReply(json.dumps({"verdicts": {k: {"label": v, "reason": "r", "evidence": []}
                                               for k, v in labels.items()}}), False, None)


def test_third_vote_rejudges_only_disputed_atoms_without_prior_votes():
    calls = []
    replies = [reply({"a": "yes", "b": "no", "c": "uncertain"}),
               reply({"a": "yes", "b": "yes", "c": "uncertain"}),
               reply({"b": "no"})]
    def dispatch(request):
        calls.append(request)
        return replies[len(calls) - 1]
    result = evaluate_consensus(lean_task(), dispatch)
    assert [set(c.response_format["json_schema"]["schema"]["properties"]["verdicts"]["required"]) for c in calls] \
        == [{"a", "b", "c"}, {"a", "b", "c"}, {"b"}]
    assert calls[2].prompt == "HEAD\nb | test | -\nTAIL"
    assert result["verdicts"] == {"a": "yes", "b": "no", "c": None}   # agreed uncertain stays unresolved
    assert result["third_vote_items"] == ["b"] and result["status"] == "unresolved"


def test_invalid_first_pair_member_gets_a_full_third_vote():
    calls = []
    replies = [JudgeReply("not json", False, None), reply({"a": "yes", "b": "no", "c": "no"}),
               reply({"a": "yes", "b": "no", "c": "no"})]
    def dispatch(request):
        calls.append(request)
        return replies[len(calls) - 1]
    result = evaluate_consensus(lean_task(), dispatch)
    assert calls[2].prompt == lean_task().prompt and result["status"] == "resolved"
    assert result["third_vote_items"] == ["a", "b", "c"]


def test_resumed_disputed_third_vote_is_verified_against_its_items():
    first = [reply({"a": "yes", "b": "no", "c": "no"}), reply({"a": "yes", "b": "yes", "c": "no"}),
             reply({"b": "yes"})]
    samples = []
    evaluate_consensus(lean_task(), lambda r: first[len(samples)], on_sample=samples.append)
    again = evaluate_consensus(lean_task(), lambda r: pytest.fail("bought again"), prior_samples=samples)
    assert again["verdicts"]["b"] == "yes" and len(again["samples"]) == 3
    tampered = [dict(s) for s in samples]
    tampered[2]["items"] = ["a", "b"]
    with pytest.raises(ValueError):
        evaluate_consensus(lean_task(), lambda r: pytest.fail("bought"), prior_samples=tampered)


def test_verbatim_tasks_keep_the_full_third_vote():
    task = JudgeTask("run", ("a", "b"), "P", "A", "judge", "v3")
    calls = []
    replies = [reply({"a": "yes", "b": "no"}), reply({"a": "yes", "b": "yes"}), reply({"a": "yes", "b": "yes"})]
    def dispatch(request):
        calls.append(request)
        return replies[len(calls) - 1]
    result = evaluate_consensus(task, dispatch)
    assert calls[2].prompt == "P" and "third_vote_items" not in result


def test_template_must_reproduce_the_prompt():
    task = lean_task()
    with pytest.raises(ValueError):
        JudgeTask("run", task.item_ids, task.prompt + "x", "A", "judge", "v4", atom_template=task.atom_template)


def test_abbreviated_and_annotated_citations_resolve_mechanically():
    visible = {"evidence_ledger": [{"evidence_id": f"EV-JD-9-{n:02d}", "symptom": f"s{n}"} for n in (2, 15, 22, 33, 36)],
               "longitudinal_data": {}}
    ans = {"cited_evidence": ["EV-22", "EV-JD-9-15/33/36 impaired glucose", "EV-99"],
           "differential": [{"diagnosis": "x", "supporting_evidence": ["EV-JD-9-02 note"]}]}
    facts = case_facts(visible, ans)
    assert [r["evidence_id"] for r in facts["ledger_rows"]] == ["EV-JD-9-22", "EV-JD-9-15", "EV-JD-9-33",
                                                                 "EV-JD-9-36", "EV-JD-9-02"]
    assert facts["abbreviated_ids_resolved"] == {"EV-22": "EV-JD-9-22"}
    assert facts["cited_but_not_delivered"] == ["EV-99"]
