"""Complete, score-blind corrective runs over immutable saved semantic inputs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .semantic_inputs import anonymous_prompt
from .semantic_judge import JudgeTask, task_record
from .semantic_rubric import make_rubric
from .semantic_visibility import correct_reference, parse_reference
from .semantic_atom_correction import (  # noqa: F401
    _source,
)

KIND = "noop-delivered-window-v1"


def _criteria_objects(criteria):
    """JSON key order is not a rubric change; wording and values remain exact."""
    result = {}
    for key, value in criteria.items():
        text, separator, context = value.partition("\nSpecific reference: ")
        result[key] = [text, json.loads(context) if separator else None]
    return result


def _policy(base: dict):
    policy = deepcopy(base["policy"])
    policy["version"] += "-visible-window-v1"
    return policy


def corrected_records(rows: list[dict], policy: dict, run_id: str) -> list[dict]:
    from .judge_evidence import evidence_catalogue
    result = []
    for row in rows:
        if row["status"] != "ready":
            continue
        original_task = JudgeTask(**{**row["task"], "item_ids": tuple(row["task"]["item_ids"])})
        if original_task.fingerprint != row["task_sha256"]:
            raise ValueError("Base task fingerprint is invalid")
        reference, recorded_criteria = parse_reference(original_task.prompt)
        fixed, change = correct_reference(reference)
        if change is None:
            continue
        answer = json.loads(original_task.answer_text)
        original_policy = {**policy, "version": row["rubric"]["version"]}
        original_criteria, original_rubric = make_rubric(reference, answer, original_policy)
        if (_criteria_objects(original_criteria) != _criteria_objects(recorded_criteria)
                or original_rubric != row["rubric"]):
            raise ValueError("Current rubric cannot reproduce the original uncorrected task")
        criteria, rubric = make_rubric(fixed, answer, policy)
        if rubric["groups"] != row["rubric"]["groups"]:
            raise ValueError("Correction changed unrelated score atom groups")
        ids = original_task.evidence_catalog is not None
        original_prompt, _ = anonymous_prompt(recorded_criteria, reference, answer, evidence_ids=ids)
        if original_prompt != original_task.prompt:
            raise ValueError("Current input serializer cannot reproduce the original task")
        # Preserve every unrelated criterion byte, not merely its JSON meaning.
        criteria = {**recorded_criteria, "data_availability": criteria["data_availability"]}
        prompt, text = anonymous_prompt(criteria, fixed, answer, evidence_ids=ids)
        if text != original_task.answer_text:
            raise ValueError("Correction may not alter the saved answer")
        task = JudgeTask(run_id, original_task.item_ids, prompt, text, original_task.judge_model,
                         policy["version"], original_task.reasoning_effort,
                         evidence_catalogue(text)[0] if ids else None)
        # Compare the JSON-normalized form also used by on-disk validation.
        result.append(json.loads(json.dumps({"key": row["key"], "status": "ready",
            "source": row["source"], "task": task_record(task), "rubric": rubric,
            "task_sha256": task.fingerprint,
            "correction": {"kind": KIND, "base_task_sha256": row["task_sha256"], **change}})))
    return result


def prepare_correction(base: Path, out: Path, *, run_id: str) -> dict:
    from .semantic_seal import digest, code_state, _write_json
    base, out = base.resolve(), out.resolve()
    if not run_id or out.exists():
        raise ValueError("A new correction run ID and new output directory are required")
    manifest, rows = _source(base)
    if run_id == manifest["run_id"]:
        raise ValueError("Corrected votes require a distinct run ID")
    policy = _policy(manifest)
    corrected = corrected_records(rows, policy, run_id)
    if not corrected:
        raise ValueError("No mismatched references to correct")
    out.mkdir(parents=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in corrected))
    result = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
              "policy": policy, "sources": manifest["sources"], "code": code_state(),
              "tasks_sha256": digest(out / "tasks.jsonl"), "states": {"ready": len(corrected)},
              "sampling": {"rule": "all source/visible window mismatches; no scores read"},
              "paid_calls": False, "sealed": False,
              "correction": {"kind": KIND, "base_run": str(base),
                  "base_manifest_sha256": digest(base / "manifest.json"),
                  "base_tasks_sha256": manifest["tasks_sha256"],
                  "keys": [r["key"] for r in corrected]}}
    _write_json(out / "manifest.json", result)
    return result


def validate_correction(out: Path):
    """Independently reconstruct every expected correction; no omission permitted."""
    from .semantic_seal import digest
    m = json.loads((out / "manifest.json").read_text())
    c = m.get("correction") or {}
    from .semantic_atom_correction import KINDS as ATOM_KINDS, validate_atom_correction
    if c.get("kind") in ATOM_KINDS:
        return validate_atom_correction(out)
    if c.get("kind") != KIND:
        raise ValueError("Unrecognized correction protocol")
    from .semantic_runref import correction_base
    base = correction_base(out, c)
    if digest(base / "manifest.json") != c["base_manifest_sha256"]:
        raise ValueError("Correction base manifest changed")
    original, rows = _source(base)
    if original["tasks_sha256"] != c["base_tasks_sha256"] or m["sources"] != original["sources"]:
        raise ValueError("Correction sources differ from base")
    from .semantic_rubric import judging_policy
    if judging_policy(m["policy"]) != judging_policy(_policy(original)):
        raise ValueError("Correction changed unrelated judging policy")
    expected = corrected_records(rows, m["policy"], m["run_id"])
    actual = [json.loads(line) for line in (out / "tasks.jsonl").read_text().split("\n") if line.strip()]
    keys = [r["key"] for r in expected]
    if c["keys"] != keys or [r["key"] for r in actual] != keys:
        raise ValueError("Correction must contain the complete affected set and no other cells")
    if actual != expected or digest(out / "tasks.jsonl") != m["tasks_sha256"]:
        raise ValueError("Saved tasks differ from the expected correction")
    return original, m, expected
