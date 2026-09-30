"""Reference-grounded semantic atoms and deterministic component aggregation."""
from __future__ import annotations

import json
import re
from haenv import data_root
from .yamlcache import load_yaml

ROOT = data_root()
REGISTRY = ROOT / "registry/semantic_judging.yaml"


#: The lean candidate protocol (semantic-v4); the default stays REGISTRY until it is validated.
REGISTRY_LEAN = ROOT / "registry/semantic_judging_v4.yaml"
LEAN_LAYOUT = "lean-v4"


#: A second-vendor judge runs the same criteria under a different vendor.
REGISTRY_SECOND = ROOT / "registry/semantic_judging_second.yaml"


#: Judge-block keys that say how a run executes, not what it judges. They are not part of a
#: run's judging identity: `load_policy` drops them and every policy comparison goes through
#: `judging_policy`, so changing the lane count never refuses a seal, a resume or a correction.
EXECUTION_KEYS = ("max_concurrency",)
#: Lanes a run uses when neither the command line nor a legacy sealed policy names a number.
DEFAULT_JUDGE_LANES = 10


def judging_policy(policy: dict) -> dict:
    """The policy without its execution keys: what two runs must share to be the same judging."""
    out = dict(policy)
    if isinstance(out.get("judge"), dict):
        out["judge"] = {k: v for k, v in out["judge"].items() if k not in EXECUTION_KEYS}
    return out


def default_lanes(policy: dict) -> int:
    """Lane count when none is given: a legacy sealed policy's own number, else the default."""
    return int((policy.get("judge") or {}).get("max_concurrency", DEFAULT_JUDGE_LANES))


def load_policy(path=None) -> dict:
    policy = judging_policy(load_yaml(REGISTRY if path is None else path))
    judge = policy["judge"]
    if policy.get("role") == "second_provider":
        if (policy["default_mode"] != "llm" or judge["model_id"].startswith("openai/")
                or judge["reasoning_effort"] != "high" or policy.get("prompt_layout") is not None
                or judge["consensus"] != "two_then_third_on_disagreement"
                or judge["evidence_mode"] != "pointer_ids"):
            raise ValueError("A second-vendor policy must keep the v3 protocol under another vendor")
        return policy
    layout = policy.get("prompt_layout")
    consensus = {None: "two_then_third_on_disagreement", LEAN_LAYOUT: "two_then_third_on_disputed_atoms"}
    if (policy["default_mode"] != "llm" or judge["model_id"] != "openai/gpt-6-luna"
            or judge["reasoning_effort"] != "high" or layout not in consensus
            or judge["consensus"] != consensus[layout]
            or judge["evidence_mode"] not in ("pointer_ids", "verbatim")
            or (layout is not None and judge["evidence_mode"] != "pointer_ids")):
        raise ValueError("Semantic policy differs from the approved judge protocol")
    return policy


#: `diagnosis_rendering: core_note_v1` (semantic-v3.1): a single-diagnosis reference
#: written "X(Y)" or "X/Y" is shown to the judge as a core diagnosis plus a note
#: that is not required, or as alternatives of which any one suffices. The gold
#: alias lists already accept either part (e.g. JD-05 lists `胰岛素抵抗`, JD-03v2 `甲减`).
CORE_NOTE = "core_note_v1"


def split_reference_diagnosis(diagnosis) -> dict | None:
    """Core diagnosis + non-required note, or accepted alternatives; None if plain."""
    text = str(diagnosis or "").strip()
    match = re.fullmatch(r"([^()（）+]+?)\s*[(（]([^()（）]+)[)）]", text)
    if match:
        return {"core_diagnosis": match.group(1).strip(), "note": match.group(2).strip(),
                "note_is_required": False}
    parts = [part.strip() for part in re.split(r"[/／]", text) if part.strip()]
    if len(parts) > 1 and not re.search(r"[()（）]", text):
        return {"alternatives": parts, "any_one_suffices": True}
    return None


#: `proposal_source: labelled_v1`: a proposed_test atom carries its source field
#: (tests_to_order[i] or executed_investigations[j]). Without it the atom's index runs
#: over tests_to_order + executed_investigations while the criterion read "at this
#: index", so a judge looking up that index in tests_to_order found no proposal.
PROPOSAL_SOURCE = "labelled_v1"


def diagnosis_context(reference: dict, policy: dict):
    """The diagnosis_hit atom's specific reference under the policy's rendering (None = none)."""
    if policy.get("diagnosis_rendering") == CORE_NOTE and reference["kind"] != "ddx:comorbidity":
        return split_reference_diagnosis(reference.get("diagnosis"))
    return None


def criterion_text(policy: dict, kind: str, reference_kind: str) -> str:
    """A criterion type's definition, with any per-reference-kind wording (semantic-v3.1)."""
    by_kind = (policy.get("criteria_by_reference_kind") or {}).get(reference_kind) or {}
    return by_kind.get(kind) or policy["criteria"][kind]


def rubric_atoms(reference: dict, answer: dict, policy: dict) -> tuple[list, dict]:
    """Applicable atoms as (item id, criterion type, specific reference or None).

    `make_rubric` renders these for the verbatim-context layout; the lean layout
    renders the same atoms with each type's definition written once.
    """
    atoms, groups, absent = [], {}, {}

    def add(group, item, kind, context=None):
        atoms.append((item, kind, context))
        groups.setdefault(group, []).append(item)

    if reference["kind"] in ("ddx:unified", "ddx:comorbidity") and reference.get("diagnosis"):
        add("dx_hit", "diagnosis_hit", "diagnosis_hit", diagnosis_context(reference, policy))
    else:
        absent["dx_hit"] = "No applicable diagnosis-hit gold on this case"
    probe = reference.get("noop")
    if isinstance(probe, dict) and type(probe.get("truth_present")) is bool:
        add("noop_ok", "data_availability", "data_availability", probe)
    else:
        absent["noop_ok"] = "No recorded data-availability probe"
    required, optional = reference["required_tests"], reference["optional_tests"]
    proposed = answer.get("tests_to_order")
    if proposed is None:
        proposed = []
    if not isinstance(proposed, list) or any(not isinstance(x, str) for x in proposed):
        raise ValueError("tests_to_order must be a list of strings when present")
    executed = answer.get("executed_investigations", [])
    if not isinstance(executed, list) or any(not isinstance(x, str) for x in executed):
        raise ValueError("executed_investigations must be a list of verified tool targets")
    # `proposal_source: labelled_v1`: each proposed_test atom names its answer field.
    sources = ([f"tests_to_order[{i}]" for i in range(len(proposed))]
               + [f"executed_investigations[{j}]" for j in range(len(executed))])
    labelled = policy.get("proposal_source") == PROPOSAL_SOURCE
    proposed = proposed + executed
    # The same registered proposal cap as the existing workup judge applies.
    from .judges.differential import TESTS_CAP_MARGIN
    cap = len(required) + len(optional) + TESTS_CAP_MARGIN
    for index, test in enumerate(required):
        add("tests_recall", f"required_test_{index}", "required_test",
            {"required_test": test, "counted_proposals": proposed[:cap]})
    if required or optional:
        for index, test in enumerate(proposed[:cap]):
            add("tests_precision", f"proposed_test_{index}", "proposed_test",
                {"proposed_test": test, "source": sources[index], "allowed_tests": required + optional}
                if labelled else {"proposed_test": test, "allowed_tests": required + optional})
    if not required:
        absent["tests_recall"] = "No required investigations in this reference"
    if not (required or optional) or not proposed:
        absent["tests_precision"] = "No applicable proposed-test denominator"
    for index, rival in enumerate(reference.get("rivals", [])):
        if rival.get("discriminator"):
            add("disc_recall", f"discriminator_{index}", "discriminator",
                {"rival": rival, "counted_proposals": proposed[:cap]})
    if "disc_recall" not in groups:
        absent["disc_recall"] = "No registered discriminating axis"
    for name in ("action_consistency", "join_reason_specific", "test_reasoning"):
        add(name, name, name)
    candidates = answer.get("differential") or []
    if isinstance(candidates, list):
        for index, candidate in enumerate(candidates):
            if isinstance(candidate, dict) and candidate.get("ruled_out_by"):
                add("exclusion_reason", f"exclusion_reason_{index}", "exclusion_reason",
                    {"diagnosis": candidate.get("diagnosis"),
                     "reason": candidate["ruled_out_by"]})
    if "exclusion_reason" not in groups:
        absent["exclusion_reason"] = "No claimed exclusion to assess"
    return atoms, {"groups": groups, "not_applicable": absent,
                   "proposal_cap": cap, "n_proposed": len(proposed),
                   "version": policy["version"]}


def make_rubric(reference: dict, answer: dict, policy: dict | None = None) -> tuple[dict, dict]:
    """Create only criteria with an applicable reference; do not invent gold."""
    policy = load_policy() if policy is None else policy
    atoms, rubric = rubric_atoms(reference, answer, policy)
    criteria = {item: criterion_text(policy, kind, reference["kind"])
                + ("\nSpecific reference: " + json.dumps(context, ensure_ascii=False)
                   if context is not None else "")
                for item, kind, context in atoms}
    return criteria, rubric


def summarize_verdicts(verdicts: dict, rubric: dict) -> dict:
    """Require every applicable atom resolved; unresolved values never shrink a mean."""
    expected = {item for items in rubric["groups"].values() for item in items}
    if set(verdicts) != expected or any(v not in ("yes", "no", None) for v in verdicts.values()):
        raise ValueError("Consensus must contain exactly the rubric atoms and closed verdicts")
    metrics = {group: None for group in rubric["not_applicable"]}
    coverage = {}
    for group, items in rubric["groups"].items():
        resolved = [verdicts[item] for item in items if verdicts[item] is not None]
        coverage[group] = {"expected": len(items), "resolved": len(resolved)}
        metrics[group] = (sum(value == "yes" for value in resolved) / len(items)
                          if len(resolved) == len(items) and items else None)
    recall, precision = metrics.get("tests_recall"), metrics.get("tests_precision")
    metrics["tests_f1"] = (None if recall is None or precision is None
                            else (2 * recall * precision / (recall + precision)
                                  if recall + precision else 0.0))
    return {"metrics": metrics, "coverage": coverage,
            "not_applicable": rubric["not_applicable"], "version": rubric["version"]}
