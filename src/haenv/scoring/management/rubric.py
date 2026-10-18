"""Versioned, fact-derived management criteria. This module never builds a case."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

from haenv.resources.yamlcache import load_yaml
from haenv.scoring.management.adapters import load_adapter

from haenv.locations import data_root

ROOT = data_root()
REGISTRY = ROOT / "registry/management_rubric.yaml"
TYPES = frozenset({"history", "protocol", "management", "checks", "explanation", "followup", "precision"})
VISIBLE_PREFIXES = ("/visible_facts/", "/protocols/", "/decision_facts/world_observations/")


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_policy(path: Path | None = None) -> dict:
    policy = load_yaml(path or REGISTRY)
    weighted_types = {"history", "management", "followup", "explanation"} if "templates" in policy else TYPES
    if (not policy.get("version") or not policy.get("source")
            or set(policy["scoring"]["type_weights"]) != weighted_types
            or policy["scoring"]["weighting"] not in {"item", "type"}
            or any(type(w) not in (int, float) or not 0 < w < float("inf")
                   for w in policy["scoring"]["type_weights"].values())):
        raise ValueError("Invalid management rubric policy")
    scoring = policy["scoring"]
    if (type(scoring.get("core_decision_cap")) not in (int, float)
            or not 0 <= scoring["core_decision_cap"] <= 1
            or set(scoring.get("core_decision_failure_labels", [])) != {"contradicted", "unmet"}):
        raise ValueError("Invalid core-decision cap policy")
    judge = policy["judge"]
    if "templates" in policy:
        templates = policy.get("templates", {})
        if (set(templates) != {"history", "management", "safety", "followup", "explanation"}
                or any(not t.get("text") or t.get("polarity") != ("negative" if k == "safety" else "positive")
                       for k, t in templates.items())):
            raise ValueError("v6 requires exactly the five shared templates")
    if (any(not isinstance(judge.get(k), str) or not judge[k].strip() for k in ("model_key", "model_id"))
            or judge["backend"] not in {"openrouter", "google"} or judge["samples"] != 1
            or judge.get("failure_policy") != "unscored"
            or type(judge.get("max_tokens")) is not int or judge["max_tokens"] <= 0
            or type(judge.get("concurrency")) is not int or judge["concurrency"] <= 0
            or type(judge.get("empty_transport_retries", 0)) is not int
            or judge.get("empty_transport_retries", 0) not in (0, 1)
            or policy["scoring"]["safety_multiplier"] != "any_negative_satisfied_zero"):
        raise ValueError("Invalid single management judge or hard safety multiplier")
    from haenv.runtime.transport import check_sampling
    check_sampling(judge["model_key"], judge["backend"], judge.get("sampling"))
    effort = judge.get("reasoning_effort")
    if ((judge["backend"] == "google" and effort is not None)
            or judge["backend"] == "openrouter" and effort not in {"low", "medium", "high", "xhigh"}):
        raise ValueError("Invalid management judge reasoning effort")
    return policy


def pointer(data, identifier: str):
    if not identifier.startswith("/"):
        raise ValueError("A fact id must be a snapshot JSON pointer")
    for part in identifier[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def verify_snapshot(snapshot: dict, policy: dict | None = None) -> str:
    policy = policy or load_policy()
    if not isinstance(snapshot, dict) or snapshot.get("version") != policy["snapshot_version"]:
        raise ValueError("Missing or unsupported open_management snapshot")
    body = {k: v for k, v in snapshot.items() if k != "facts_sha256"}
    # Match the generation-side serialization exactly, including separators.
    actual = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if snapshot.get("facts_sha256") != actual:
        raise ValueError("Management fact snapshot hash mismatch")
    return actual


def derive(pack: str, snapshot: dict, policy: dict | None = None) -> dict:
    """One shared loop; adapters provide facts, never criterion text."""
    policy = deepcopy(policy or load_policy())
    fact_hash = verify_snapshot(snapshot, policy)
    if policy["version"] not in {"management-rubric-v6", "management-rubric-v6.1", "management-rubric-v6.2"}:
        raise ValueError("Historical criteria must be read from their saved archive")
    adapter = load_adapter(policy["adapters"][pack])
    context = adapter.facts(snapshot, policy)
    refs = context["source_fact_ids"]
    if not refs or any(not ref.startswith(VISIBLE_PREFIXES) for ref in refs):
        raise ValueError("Adapter facts require visible snapshot sources")
    for ref in refs:
        pointer(snapshot, ref)
    control = context["control"]
    frozen_negative, frozen_decisions = {}, {}
    if policy["version"] in {"management-rubric-v6.1", "management-rubric-v6.2"}:
        _v61_control_context(pack, snapshot, control, policy)
    if policy["version"] == "management-rubric-v6.2":
        from haenv.scoring.management.controls_v62 import improve_positive_context
        from haenv.scoring.management.controls_v62 import NEGATIVE_CONTROLS
        from haenv.scoring.management.judge import controls
        legacy_policy = deepcopy(policy)
        legacy_policy["version"] = "management-rubric-v6.1"
        legacy = derive(pack, snapshot, legacy_policy)
        frozen_negative = {n: deepcopy(a) for n, a in controls(legacy).items() if n in NEGATIVE_CONTROLS and a is not None}
        frozen_decisions = deepcopy(legacy["control_decisions"])
        improve_positive_context(pack, snapshot, control, policy, context['history'])
    selected = control["selected"]
    action = control["actions"][selected]
    full_management = f"下一步：{action}。"
    # Control renderings are finite plumbing probes, separate from judge facts.
    claims = {"history": control["history"], "management": full_management,
              "followup": control["followup"], "explanation": control["explanation"]}
    public_prohibitions = [{k: v for k, v in p.items()
                            if k not in {"violation_control", "required_controls"}}
                           for p in context["prohibitions"]]
    facts = {"history": context["history"], "management": context["key_facts"],
             "safety": {"prohibitions": public_prohibitions,
                        "guideline_not_recommended": [{k: v for k, v in r.items() if k != "control"}
                                                      for r in context["guidelines"]["not_recommended"]]},
             "followup": context["key_facts"], "explanation": context["key_facts"]}
    violations = [p["violation_control"] for p in context["prohibitions"]]
    violations += [r["control"] for r in context["guidelines"]["not_recommended"] if r["control"]]
    required = [r for p in context["prohibitions"] for r in p["required_controls"]]
    items = []
    for kind, template in policy["templates"].items():
        if kind == "history" and not context["history"]:
            continue
        stub = {"claims": [claims[kind]] if kind in claims else [],
                "contradictions": [], "forbidden": violations[0] if kind == "safety" and violations else None,
                "omitted_item": None}
        if kind == "management":
            stub["claims"].extend(f"下一步：{a}。" for a in control["management_alternatives"])
        if kind == "safety":
            stub["diagnostic"] = {"kind": "v6_safety", "violations": violations, "required": required}
            if required:
                stub["omitted_item"] = "management"
        items.append({"id": kind, "type": kind, "polarity": template["polarity"], "text": template["text"],
                      "source_fact_ids": refs, "evidence_fact_ids": refs,
                      "fact_group": kind, "core_decision": False,
                      "judgment_source": "judge", "weight": None if kind == "safety" else policy["scoring"]["type_weights"][kind],
                      "stub": stub})
    rubric = {"version": policy["version"], "pack": pack, "facts_sha256": fact_hash,
              "policy_sha256": sha256(policy), "items": items, "gaps": [],
              "code_core_decision": True, "control_context": control,
              "case_facts": {"decisive_history": facts["history"], "key_facts": context["key_facts"],
                             "safety": facts["safety"],
                             "initial_check_context": [{k: v for k, v in r.items() if k != "control"}
                                                       for r in context["guidelines"]["initial"]]},
              "control_decisions": {}, "control_replacements": {},
              "control_inventory": {"visible_facts": snapshot["visible_facts"]},
              "guideline_controls": context["guidelines"]}
    if frozen_negative:
        rubric["frozen_negative_controls"] = frozen_negative
        rubric["frozen_control_decisions"] = frozen_decisions
    rubric["writing_variants"] = policy["control_variants"]
    from haenv.scoring.management.judge import v6_controls
    probes = v6_controls(rubric)
    for name, answer in probes.items():
        if name.startswith("variant_") or name in {"gold_plus_initial_checks", "split_correct", "generic_correct"}:
            rubric["control_decisions"][name] = {"correct": True, "selected": selected, "accepted": [selected]}
    return rubric


def _v61_control_context(pack, snapshot, control, policy):
    """New finite probes only; historical gold and benign controls stay byte-identical."""
    v, f = snapshot["visible_facts"], snapshot["decision_facts"]
    control["generic_action"] = control["actions"][control["selected"]]
    if pack == "p2":
        project = "心率、指夹血氧和本次症状"
    elif pack == "p3":
        control["generic_action"] = policy["med_decisions"][control["selected"]]
        project = {"enalapril": "血压、肾功能和血钾", "atorvastatin": "血脂及肌肉不适",
                   "gliclazide": "血糖及低血糖症状", "levothyroxine": "甲状腺功能及相关症状"}[f["drug"]]
    elif pack == "p4":
        control["generic_action"] = policy["next_steps"][control["selected"]]
        project = v["name"]
    else:
        from haenv.resources.regpath import load_registry
        implementation = load_registry("management_adapter_rules.yaml")["m2"]["urgency_implementation"]
        control["generic_action"] = implementation[control["selected"]]
        project = "、".join(snapshot["required_checks"]) if snapshot["required_checks"] else "当前症状及所选药物疗效"
    control["followup_project"] = project


def control_core_decision(rubric: dict, name: str) -> dict:
    """Read construction choices, including exact claim replacements in old archives."""
    source = {"kind": "control_construction", "control": name}
    if name in {"gold", "gold_facts", "split_correct", "gold_plus_contraindication",
                "gold_plus_initial_checks", "gold_plus_not_recommended_check",
                "gold_plus_routine_monitoring"}:
        correct, status = True, "correct"
    elif name in {"empty", "generic", "readings_only", "verbose_unrelated"}:
        correct, status = False, "absent"
    elif name in rubric.get("control_decisions", {}):
        source["fields"] = rubric["control_decisions"][name]
        correct = source["fields"]["selected"] in source["fields"]["accepted"]
        if type(source["fields"]["correct"]) is not bool or source["fields"]["correct"] != correct:
            raise ValueError("Control decision marker differs from its acceptance set")
        status = "correct" if correct else "wrong"
    elif name in rubric.get("control_replacements", {}):
        # The archived constructor selects the first claim and substitutes exact
        # claims. Membership here uses that saved acceptance set, never answer NLP.
        replacements = dict(rubric["control_replacements"][name])
        fields = {i["id"]: {"selected": replacements.get(i["stub"]["claims"][0], i["stub"]["claims"][0]),
                               "accepted": i["stub"]["claims"]}
                  for i in rubric["items"] if i.get("core_decision")}
        if not fields:
            raise ValueError("A decision control has no saved core acceptance set")
        source["fields"] = fields
        correct = all(v["selected"] in v["accepted"] for v in fields.values())
        status = "correct" if correct else "wrong"
    else:
        raise ValueError(f"No registered decision construction for control {name}")
    return {"correct": correct, "status": status, "source": source}


def with_core_verdicts(rubric: dict, verdicts: dict, core_decision: dict | None) -> dict:
    """Replace every core judge vote with the package's authoritative code atom."""
    core = [i for i in rubric["items"] if i.get("core_decision")]
    merged = {k: v for k, v in verdicts.items() if k not in {i["id"] for i in core}}
    decision = core_decision or {"correct": None, "status": "unknown", "source": {"kind": "missing"}}
    correct = decision["correct"]
    label = ("satisfied" if correct is True else "unmet" if decision["status"] == "absent"
             else "contradicted" if correct is False else None)
    for item in core:
        merged[item["id"]] = {"label": label, "reason": f"代码按已保存的核心决定判据整体判定：{decision['status']}。",
                               "evidence": [], "origin": "code", "source": decision["source"]}
    return merged


def score(rubric: dict, verdicts: dict | None, policy: dict | None = None, *,
          core_decision: dict | None = None) -> dict:
    policy = policy or load_policy()
    items = rubric["items"]
    if verdicts is not None:
        verdicts = with_core_verdicts(rubric, verdicts, core_decision)
    if (verdicts is None or set(verdicts) != {i["id"] for i in items}
            or any(i.get("core_decision") for i in items)
            and (core_decision is None or type(core_decision.get("correct")) is not bool)
            or rubric.get("code_core_decision") and (core_decision is None or type(core_decision.get("correct")) is not bool)):
        return {"status": "unscored", "rubric_score": None, "positive_fraction": None,
                "safety_multiplier": None, "safety_hits": [], "by_type": {},
                "quality_score": None,
                "uncapped_positive_fraction": None, "core_failures": [], "core_cap_applied": False}
    if any(v.get("label") not in {"satisfied", "unmet", "contradicted"} for v in verdicts.values()):
        raise ValueError("Invalid management verdict")
    hits = [i["id"] for i in items if i["polarity"] == "negative"
            and verdicts[i["id"]]["label"] == "satisfied"]
    profile = {}
    for kind in sorted(TYPES):
        group = [i for i in items if i["type"] == kind]
        if group:
            profile[kind] = {"n": len(group), "satisfied": sum(verdicts[i["id"]]["label"] == "satisfied" for i in group)}
            profile[kind]["fraction"] = profile[kind]["satisfied"] / len(group)
    if policy["scoring"]["weighting"] == "type":
        weights = policy["scoring"]["type_weights"]
        fraction = sum(weights[k] * v["fraction"] for k, v in profile.items()) / sum(weights[k] for k in profile)
    else:
        positive = [i for i in items if i["polarity"] == "positive"]
        weights = policy["scoring"]["type_weights"]
        fraction = sum(weights[i["type"]] * (verdicts[i["id"]]["label"] == "satisfied") for i in positive) / sum(weights[i["type"]] for i in positive)
    uncapped = fraction
    failures = [i["id"] for i in items if i.get("core_decision")
                and verdicts[i["id"]]["label"] in policy["scoring"]["core_decision_failure_labels"]]
    if rubric.get("code_core_decision") and core_decision["correct"] is False:
        failures = ["code.core_decision"]
    if failures:
        fraction = min(fraction, policy["scoring"]["core_decision_cap"])
    # Q for S = G*D*(1+Q)/2 retains quality when D=0. Code-owned core items
    # and the core cap do not enter Q; the independent safety multiplier does.
    quality = [i for i in items if i["polarity"] == "positive" and not i.get("core_decision")]
    weights = policy["scoring"]["type_weights"]
    if policy["scoring"]["weighting"] == "type":
        kinds = {i["type"] for i in quality}
        q = (sum(weights[k] * sum(verdicts[i["id"]]["label"] == "satisfied" for i in quality if i["type"] == k)
                 / sum(i["type"] == k for i in quality) for k in kinds) / sum(weights[k] for k in kinds)) if kinds else None
    else:
        q = (sum(weights[i["type"]] * (verdicts[i["id"]]["label"] == "satisfied") for i in quality)
             / sum(weights[i["type"]] for i in quality)) if quality else None
    return {"status": "scored", "rubric_score": fraction * int(not hits),
            "quality_score": q * int(not hits) if q is not None else None,
            "positive_fraction": fraction, "safety_multiplier": int(not hits),
            "safety_hits": hits, "by_type": profile, "uncapped_positive_fraction": uncapped,
            "core_failures": failures, "core_cap_applied": fraction < uncapped}
