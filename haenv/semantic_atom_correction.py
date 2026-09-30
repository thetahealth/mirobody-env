"""Atom-level corrective runs: re-judge one criterion under a revised policy, keep every other vote.

Kind `dx-kind-clause-v1`: the diagnosis_hit atom is
re-judged alone, in a single-atom request, under a policy that sends the
combined-diagnosis clause only for ddx:comorbidity and renders a single-diagnosis
reference "X(Y)" / "X/Y" as core + non-required note / alternatives. Every other
atom of the cell keeps its original 2+1 votes; the view replaces only the dx_hit
group (`merge_atom`).

Selection is score-blind: either the complete set of ready cells whose
diagnosis_hit input changes materially under the policy (`selection: affected`),
or an explicit preregistered key list (`selection: probe`, used to validate the
change before scaling it up).
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from .semantic_inputs import anonymous_prompt
from .semantic_judge import JudgeTask, task_record
from .semantic_rubric import CORE_NOTE, diagnosis_context, make_rubric
from .semantic_visibility import correct_reference, parse_reference

KIND = "dx-kind-clause-v1"
ATOM, GROUP = "diagnosis_hit", "dx_hit"
#: Kind `test-reasoning-why-v1`: the test_reasoning atom re-judged with the answer's own
#: `trace.tool_calls[].{target, why}` (policy `reason_slot: tool_calls_why`), in the v3
#: or the lean layout, so its cross-protocol stability can be re-measured.
WHY_KIND = "test-reasoning-why-v1"
#: Kind `proposed-test-source-v1`: proposed_test atoms re-judged with the investigation's
#: wording and source field named (policy `proposal_source: labelled_v1`). One request per
#: cell carries its selected proposed_test atoms; the other proposed_test votes are kept
#: and tests_precision is recomputed over the whole group (`merge_atoms`).
SOURCE_KIND = "proposed-test-source-v1"
KINDS = {KIND: {"atom": ATOM, "group": GROUP},
         WHY_KIND: {"atom": "test_reasoning", "group": "test_reasoning"},
         SOURCE_KIND: {"atom": "proposed_test", "group": "tests_precision"}}
PROPOSED = "proposed_test_"


def affected(reference: dict, policy: dict) -> bool:
    """The diagnosis_hit input changes materially: a single-diagnosis reference with a note.

    For ddx:comorbidity the criterion text is unchanged. For a plain single
    diagnosis only a vacuous clause (there is no combination) is removed.
    """
    return reference["kind"] != "ddx:comorbidity" and diagnosis_context(reference, policy) is not None


def _check_policy(policy: dict, kind: str) -> None:
    if kind == KIND and (policy.get("diagnosis_rendering") != CORE_NOTE
                         or not policy.get("criteria_by_reference_kind")):
        raise ValueError("An atom correction needs the kind-conditional diagnosis policy")
    if kind == WHY_KIND and policy.get("reason_slot") != "tool_calls_why":
        raise ValueError("A reason-slot correction needs a policy with reason_slot")
    if kind == SOURCE_KIND and (policy.get("proposal_source") != "labelled_v1"
                                or policy.get("prompt_layout") is not None):
        raise ValueError("A proposal-source correction needs the labelled v3-layout policy")
    if kind not in KINDS:
        raise ValueError("Unrecognized atom correction kind")


def _raw_answers(batch: str, cache: dict) -> dict:
    import hashlib
    if batch not in cache:
        cache[batch] = {}
        for line in (Path(batch) / "responses.jsonl").read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                if isinstance(r.get("raw"), str):
                    cache[batch][hashlib.sha256(r["raw"].encode()).hexdigest()] = r["raw"]
    return cache[batch]


def misaligned(answer: dict, item_ids) -> list[str]:
    """proposed_test atoms whose index lies past tests_to_order (bought investigations)."""
    n = len(answer.get("tests_to_order") or [])
    return [i for i in item_ids if i.startswith(PROPOSED) and int(i[len(PROPOSED):]) >= n]


def _source_records(rows: list[dict], policy: dict, run_id: str, keys: list[str] | None) -> list[dict]:
    """One task per cell with its selected proposed_test atoms, each naming its source field.

    `keys` (probe) are "<cell key>#<item id>"; None selects every proposed_test atom
    whose index lies past tests_to_order.
    """
    from .judge_evidence import evidence_catalogue
    wanted = None
    if keys is not None:
        wanted = {}
        for entry in keys:
            key, sep, item = entry.partition("#")
            if not sep or not item.startswith(PROPOSED):
                raise ValueError("A proposal-source probe key is '<cell key>#proposed_test_<i>'")
            wanted.setdefault(key, []).append(item)
    result = []
    for row in rows:
        if row["status"] != "ready" or (wanted is not None and row["key"] not in wanted):
            continue
        base = JudgeTask(**{**row["task"], "item_ids": tuple(row["task"]["item_ids"])})
        if base.fingerprint != row["task_sha256"]:
            raise ValueError("Base task fingerprint is invalid")
        answer = json.loads(base.answer_text)
        present = [i for i in base.item_ids if i.startswith(PROPOSED)]
        mis = misaligned(answer, present)
        if wanted is None:
            items = mis
        else:
            items = sorted(set(wanted[row["key"]]), key=lambda i: int(i[len(PROPOSED):]))
            if len(items) != len(wanted[row["key"]]) or not set(items) <= set(present):
                raise ValueError("A probe atom is not a proposed_test atom of its base cell")
        if not items:
            continue
        reference, _ = parse_reference(base.prompt)
        fixed, _ = correct_reference(reference)
        criteria, _ = make_rubric(fixed, answer, policy)
        if [i for i in criteria if i.startswith(PROPOSED)] != present:
            raise ValueError("The corrected rubric indexes proposals differently from its base")
        ids = base.evidence_catalog is not None
        prompt, text = anonymous_prompt({i: criteria[i] for i in items}, fixed, answer, evidence_ids=ids)
        if json.loads(text) != answer:
            raise ValueError("Correction may not alter the saved answer")
        task = JudgeTask(run_id, tuple(items), prompt, text, base.judge_model, policy["version"],
                         base.reasoning_effort, evidence_catalogue(text)[0] if ids else None)
        sources = {i: json.loads(criteria[i].partition("\nSpecific reference: ")[2])["source"]
                   for i in items}
        result.append(json.loads(json.dumps({"key": row["key"], "status": "ready",
            "source": row["source"], "task": task_record(task),
            "rubric": {"groups": {"tests_precision": list(items)}, "not_applicable": {},
                       "version": policy["version"]},
            "task_sha256": task.fingerprint,
            "correction": {"kind": SOURCE_KIND, "atom": "proposed_test", "atoms": list(items),
                           "base_task_sha256": row["task_sha256"], "sources": sources,
                           "misaligned": [i for i in items if i in mis]}})))
    if wanted is not None and {r["key"] for r in result} != set(wanted):
        raise ValueError("Every probe key must name a ready base cell")
    return result


def atom_records(rows: list[dict], policy: dict, run_id: str, keys: list[str] | None,
                 kind: str = KIND) -> list[dict]:
    if kind == SOURCE_KIND:
        _check_policy(policy, kind)
        return _source_records(rows, policy, run_id, keys)
    from .evaluate import _extract_json
    from .judge_evidence import evidence_catalogue
    from .semantic_inputs import investigation_reasons
    from .semantic_rubric import LEAN_LAYOUT, rubric_atoms
    _check_policy(policy, kind)
    atom_id, group = KINDS[kind]["atom"], KINDS[kind]["group"]
    wanted = None if keys is None else set(keys)
    result, raws = [], {}
    for row in rows:
        if row["status"] != "ready" or (wanted is not None and row["key"] not in wanted):
            continue
        base = JudgeTask(**{**row["task"], "item_ids": tuple(row["task"]["item_ids"])})
        if base.fingerprint != row["task_sha256"]:
            raise ValueError("Base task fingerprint is invalid")
        if atom_id not in base.item_ids:
            if wanted is not None:
                raise ValueError(f"A probe cell has no {atom_id} atom")
            continue
        reference, _ = parse_reference(base.prompt)
        fixed, _ = correct_reference(reference)
        answer = json.loads(base.answer_text)
        extra = {}
        if kind == WHY_KIND:
            raw = _raw_answers(row["source"]["batch"], raws).get(row["source"]["raw_sha256"])
            if raw is None:
                raise ValueError("The judged answer is not in the source responses")
            extra = {"investigation_reasons": investigation_reasons(_extract_json(raw))}
            hit = bool(extra["investigation_reasons"])
        else:
            hit = affected(fixed, policy)
        if wanted is None and not hit:
            continue
        shown = {**answer, **extra}
        ids = base.evidence_catalog is not None
        template = None
        if policy.get("prompt_layout") == LEAN_LAYOUT:
            from .semantic_lean import lean_prompt
            atoms, rubric = rubric_atoms(fixed, shown, policy)
            prompt, text, template = lean_prompt([a for a in atoms if a[0] == atom_id], fixed, shown, policy,
                                                 proposal_cap=rubric["proposal_cap"],
                                                 n_proposed=rubric["n_proposed"])
        else:
            criteria, _ = make_rubric(fixed, shown, policy)
            prompt, text = anonymous_prompt({atom_id: criteria[atom_id]}, fixed, shown, evidence_ids=ids)
        shown_text = json.loads(text)
        for key in extra:
            shown_text.pop(key, None)
        if shown_text != json.loads(base.answer_text):
            raise ValueError("Correction may not alter the saved answer")
        task = JudgeTask(run_id, (atom_id,), prompt, text, base.judge_model, policy["version"],
                         base.reasoning_effort, evidence_catalogue(text)[0] if ids else None,
                         atom_template=template)
        result.append(json.loads(json.dumps({"key": row["key"], "status": "ready",
            "source": row["source"], "task": task_record(task),
            "rubric": {"groups": {group: [atom_id]}, "not_applicable": {}, "version": policy["version"]},
            "task_sha256": task.fingerprint,
            "correction": {"kind": kind, "atom": atom_id, "base_task_sha256": row["task_sha256"],
                           "reference_kind": fixed["kind"],
                           "rendering": diagnosis_context(fixed, policy) if kind == KIND else None,
                           "affected": hit}})))
    if wanted is not None and {r["key"] for r in result} != wanted:
        raise ValueError(f"Every probe key must name a ready base cell with a {atom_id} atom")
    return result


def prepare_atom_correction(base: Path, out: Path, *, run_id: str, policy: dict,
                            keys: list[str] | None = None, kind: str = KIND) -> dict:
    from .semantic_corrections import _source
    from .semantic_pipeline import code_state, digest, _write_json
    base, out = Path(base).resolve(), Path(out).resolve()
    if not run_id or out.exists():
        raise ValueError("A new correction run ID and new output directory are required")
    manifest, rows = _source(base)
    if run_id == manifest["run_id"]:
        raise ValueError("Corrected votes require a distinct run ID")
    records = atom_records(rows, policy, run_id, keys, kind)
    if not records:
        raise ValueError("No cells to correct")
    out.mkdir(parents=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    selection = "affected" if keys is None else "probe"
    result = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
              "policy": policy, "sources": manifest["sources"], "code": code_state(),
              "tasks_sha256": digest(out / "tasks.jsonl"), "states": {"ready": len(records)},
              "sampling": {"rule": (f"every ready cell whose {KINDS[kind]['atom']} input changes under "
                                    "the policy; no scores read") if keys is None else "preregistered probe keys"},
              "paid_calls": False, "sealed": False,
              "correction": {"kind": kind, "atom": KINDS[kind]["atom"], "selection": selection,
                             "base_run": str(base),
                             "base_manifest_sha256": digest(base / "manifest.json"),
                             "base_tasks_sha256": manifest["tasks_sha256"],
                             "keys": [r["key"] for r in records]}}
    if kind == SOURCE_KIND and keys is not None:
        result["correction"]["atom_keys"] = list(keys)
    _write_json(out / "manifest.json", result)
    return result


def validate_atom_correction(out: Path):
    from .semantic_corrections import _source
    from .semantic_pipeline import digest
    m = json.loads((Path(out) / "manifest.json").read_text())
    c = m["correction"]
    from .semantic_runref import correction_base
    base = correction_base(out, c)
    if digest(base / "manifest.json") != c["base_manifest_sha256"]:
        raise ValueError("Correction base manifest changed")
    original, rows = _source(base)
    if original["tasks_sha256"] != c["base_tasks_sha256"] or m["sources"] != original["sources"]:
        raise ValueError("Correction sources differ from base")
    loose = ("version", "criteria", "diagnosis_rendering", "criteria_by_reference_kind", "reason_slot",
             "prompt_layout", "judge", "scoring_gate", "proposal_source")
    from .semantic_rubric import judging_policy
    mine, theirs = judging_policy(m["policy"]), judging_policy(original["policy"])
    if ({k: v for k, v in mine.items() if k not in loose}
            != {k: v for k, v in theirs.items() if k not in loose}
            or {k: v for k, v in mine["judge"].items() if k != "consensus"}
            != {k: v for k, v in theirs["judge"].items() if k != "consensus"}):
        raise ValueError("Correction changed the judge model or its settings")
    keys = None if c["selection"] == "affected" else c.get("atom_keys", c["keys"])
    expected = atom_records(rows, m["policy"], m["run_id"], keys, c["kind"])
    actual = [json.loads(line) for line in (Path(out) / "tasks.jsonl").read_text().split("\n") if line.strip()]
    if c["keys"] != [r["key"] for r in expected] or [r["key"] for r in actual] != c["keys"]:
        raise ValueError("Correction must contain the complete affected set and no other cells")
    if actual != expected or digest(Path(out) / "tasks.jsonl") != m["tasks_sha256"]:
        raise ValueError("Saved tasks differ from the expected correction")
    return original, m, expected


def merge_atom(base_cell: dict, corrected: dict, group: str = GROUP) -> dict:
    """Replace only one group (default dx_hit) of a finished base cell by the corrected atom.

    An unfinished correction leaves dx_hit pending (the superseded vote is not
    used); an unfinished base cell stays as it is.
    """
    if "summary" not in base_cell:
        return base_cell
    merged = deepcopy(base_cell)
    summary = merged["summary"]
    if group not in summary["metrics"] or group not in summary.get("coverage", {}):
        raise ValueError(f"The base cell has no {group} atom to correct")
    if "summary" not in corrected:
        summary["metrics"][group] = None
        summary["coverage"].pop(group)
        merged["status"] = "pending"
        return merged
    value = corrected["summary"]["metrics"][group]
    summary["metrics"][group] = value
    summary["coverage"][group] = {"expected": 1, "resolved": int(value is not None)}
    if merged["status"] in ("resolved", "unresolved"):
        complete = all(c["resolved"] == c["expected"] for c in summary["coverage"].values())
        merged["status"] = "resolved" if complete else "unresolved"
    return merged


def merge_atoms(base_cell: dict, corrected: dict, group: str, atoms) -> dict:
    """Replace some atoms of one group by their corrected verdicts and recompute the group.

    The group's other atoms keep the base cell's verdicts. An unfinished correction
    leaves the group pending; an unfinished base cell stays as it is.
    """
    if "summary" not in base_cell:
        return base_cell
    merged = deepcopy(base_cell)
    summary = merged["summary"]
    if group not in summary.get("coverage", {}) or "verdicts" not in merged:
        raise ValueError(f"The base cell has no {group} atoms to correct")
    if "summary" not in corrected:
        summary["metrics"][group] = None
        summary["coverage"].pop(group)
        merged["status"] = "pending"
    else:
        atoms = list(atoms)
        items = [k for k in merged["verdicts"] if k.startswith(PROPOSED)]
        if not set(atoms) <= set(items) or set(corrected["verdicts"]) != set(atoms):
            raise ValueError("Corrected atoms must be atoms of the base group")
        if len(items) != summary["coverage"][group]["expected"]:
            raise ValueError("The base group and its verdicts disagree")
        for item in atoms:
            merged["verdicts"][item] = corrected["verdicts"][item]
        resolved = [merged["verdicts"][i] for i in items if merged["verdicts"][i] is not None]
        summary["coverage"][group] = {"expected": len(items), "resolved": len(resolved)}
        summary["metrics"][group] = (sum(v == "yes" for v in resolved) / len(items)
                                     if len(resolved) == len(items) and items else None)
        if merged["status"] in ("resolved", "unresolved"):
            complete = all(c["resolved"] == c["expected"] for c in summary["coverage"].values())
            merged["status"] = "resolved" if complete else "unresolved"
    if "tests_f1" in summary["metrics"]:
        recall, precision = summary["metrics"].get("tests_recall"), summary["metrics"].get("tests_precision")
        summary["metrics"]["tests_f1"] = (None if recall is None or precision is None
                                          else (2 * recall * precision / (recall + precision)
                                                if recall + precision else 0.0))
    return merged
