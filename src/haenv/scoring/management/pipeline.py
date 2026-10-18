"""Read-only rejudging of saved management plans, with durable independent outputs."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import threading

from haenv.runtime.judge_scheduler import run_bounded
from haenv.scoring.management.judge import EmptyJudgeResponse
from haenv.scoring.management.judge import answer_json
from haenv.scoring.management.judge import controls
from haenv.scoring.management.judge import empty_answer
from haenv.scoring.management.judge import make_task
from haenv.scoring.management.judge import parse_reply
from haenv.scoring.management.judge import request_for
from haenv.scoring.management.judge import stub_reply
from haenv.scoring.management.judge import transport_reply
from haenv.scoring.management.rubric import ROOT
from haenv.scoring.management.rubric import canonical
from haenv.scoring.management.rubric import control_core_decision
from haenv.scoring.management.rubric import derive
from haenv.scoring.management.rubric import load_policy
from haenv.scoring.management.rubric import score
from haenv.scoring.management.rubric import sha256
from haenv.scoring.management.rubric import verify_snapshot
from haenv.scoring.management.rubric import with_core_verdicts
from haenv.runtime.row_store import load_rows
from haenv.runtime.budget import BudgetExceeded, BudgetLedger
from haenv.scoring.semantic.inputs import select_responses
from haenv.scoring.semantic.parallel import atomic_json
from haenv.scoring.semantic.seal import digest
from haenv.runtime.pricing import ReconciledCallFailure, RejectedCallFailure, UnknownCostCallFailure

CODE_FILES = (
    'haenv/scoring/management/__init__.py',
    'haenv/scoring/management/controls_v62.py',
    'haenv/scoring/management/rubric.py',
    'haenv/scoring/management/judge.py',
    'haenv/scoring/management/pipeline.py',
    'haenv/scoring/management/precision.py',
    'haenv/scoring/management/precision_rejudge.py',
    'haenv/scoring/management/estimate.py',
    'haenv/scoring/management/adapters/__init__.py',
    'haenv/scoring/management/adapters/common.py',
    'haenv/scoring/management/adapters/p2.py',
    'haenv/scoring/management/adapters/p3.py',
    'haenv/scoring/management/adapters/p4.py',
    'haenv/scoring/management/adapters/m2.py',
    'registry/management_rubric.yaml',
    'registry/test_indications.yaml',
    'registry/management_adapter_rules.yaml',
    'registry/p4_diversity.yaml',
    'haenv/scoring/semantic/judge.py',
    'haenv/scoring/semantic/inputs.py',
    'haenv/runtime/row_store.py',
    'haenv/scoring/semantic/transport.py',
    'haenv/scoring/semantic/parallel.py',
    'haenv/runtime/judge_scheduler.py',
    'haenv/runtime/native_accounting.py',
    'haenv/scoring/management/transport.py',
    'haenv/runtime/solvers.py',
    'haenv/runtime/transport.py',
    'haenv/reporting/tools/m2_raw_correct.py',
    'haenv/runtime/budget.py',
    'haenv/runtime/pricing.py',
    'haenv/runtime/config_contracts.py',
    'haenv/runtime/quota.py',
    'haenv/runtime/retry_classifier.py',
    'haenv/locations.py',
    'haenv/scoring/management/route_run.py',
    'haenv/scoring/routes.py',
)
BLOCKS = {"triage_p2": "p2", "meds_p3": "p3", "followup_p4": "p4"}


def eval_core_decision(pack, row):
    """Use existing production atoms; missing atoms remain explicitly unknown."""
    names = {"p2": ("p2_in_set",), "p3": ("p3_class_ok",),
             "p4": ("p4_class_ok", "p4_next_ok"),
             "m2": ("overall", "dx_kind", "gold_kind", "join_hit", "held_independent", "dx_listed")}[pack]
    fields = {name: row.get(name) for name in names}
    source = {"kind": "eval.jsonl", "fields": fields}
    if pack == "m2":
        from haenv.reporting.tools.m2_raw_correct import raw_correct
        correct = raw_correct(row)
        source["function"] = "tools/m2_raw_correct.raw_correct"
    else:
        if any(value is not None and (type(value) not in (bool, int, float) or value not in (0, 1))
               for value in fields.values()):
            raise ValueError("Core evaluation atoms must be boolean or 0/1")
        correct = None if any(value is None for value in fields.values()) else all(bool(v) for v in fields.values())
    return {"correct": correct, "status": "unknown" if correct is None else "correct" if correct else "wrong",
            "source": source}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")


def code_state():
    from haenv.provenance.anchor import judging_fingerprint
    from haenv.locations import source_path, repository_root
    files = {name: digest(source_path(name)) for name in CODE_FILES}
    repo = repository_root()
    commit = dirty = None
    if repo is not None:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    return {"management_code_sha16": sha256(files)[:16], "files": files,
            "judging_sha16": judging_fingerprint(), "git_commit": commit, "git_dirty": dirty}


def _outside_sources(path, sources):
    path = Path(path).resolve()
    if any(path == p.resolve() or p.resolve() in path.parents for p in sources):
        raise ValueError("Management outputs and ledgers must be outside every source batch")


def _responses(batch):
    path = batch / "responses.jsonl"
    files = [path] if path.is_file() else sorted((batch / "responses").glob("*.json*"))
    if not files:
        raise ValueError("No saved responses; do not ask models for replacement answers")
    rows = []
    for file in files:
        doc = read_jsonl(file) if file.suffix == ".jsonl" else json.loads(file.read_text())
        if isinstance(doc, dict):
            doc = [doc]
        if not isinstance(doc, list):
            raise ValueError("Response archive must contain saved response records")
        for original in doc:
            r = dict(original)
            if "raw" not in r and "_raw_text" in r:
                r["raw"] = r["_raw_text"]
            elif "raw" not in r and isinstance(r.get("answer"), dict):
                # A saved parsed original JSON is an authorized input too. No plan is synthesized.
                r["raw"] = canonical(r["answer"])
            rows.append(r)
    return rows, files


def _batch_inputs(batch, policy):
    batch = Path(batch).resolve()
    required = [batch / "cases.jsonl", batch / "eval.jsonl"]
    if not all(p.is_file() for p in required):
        raise ValueError("Saved cases/evaluation missing; do not regenerate")
    meta_path = batch / "batch.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    cases = {}
    for record in read_jsonl(required[0]):
        case, cid = record["case"], record["case_id"]
        if cid in cases or cid != case["case_id"]:
            raise ValueError("Duplicate or mismatched saved case identity")
        if isinstance(meta.get("cases_sha256"), dict):
            actual = hashlib.sha256(json.dumps({"case_id": cid, "case": case}, ensure_ascii=False,
                                               separators=(",", ":")).encode()).hexdigest()[:16]
            if meta["cases_sha256"].get(cid) != actual:
                raise ValueError("Saved case digest differs from batch metadata")
        matches = [pack for block, pack in BLOCKS.items() if block in case["prediction_context"]]
        if not matches and ("m2_frame" in case["prediction_context"] or "m2_workup" in case["adjudication"]):
            matches = ["m2"]
        if len(matches) != 1:
            raise ValueError("No unique supported pack in saved case")
        pack = matches[0]
        snap = case["adjudication"].get("open_management")
        verify_snapshot(snap, policy)
        expected = (case["prediction_context"] if pack == "m2" else
                    case["prediction_context"][next(block for block, value in BLOCKS.items() if value == pack)])
        if snap["visible_facts"] != expected:
            raise ValueError("Snapshot visible facts differ from this saved case")
        if pack == "m2":
            cutoff = case["prediction_context"]["prediction_time_T"]
            for stream, rows in snap["decision_facts"].get("world_observations", {}).items():
                actual = [r for r in case["longitudinal_data"].get(stream, []) if int(r.get("ts", cutoff + 1)) <= cutoff]
                if rows != actual:
                    raise ValueError("Saved M2 observation snapshot differs from the case")
        cases[cid] = {"case": case, "pack": pack, "snapshot": snap}
    rows = load_rows(required[1])
    responses, response_files = _responses(batch)
    if any(r.get("case") not in cases for r in rows + responses):
        raise ValueError("Evaluation/response refers to an unknown saved case")
    for row in rows:
        if not row.get("judging_sha16") or not row.get("world_sha"):
            raise ValueError("Saved evaluation must carry judging and world fingerprints")
        if meta.get("world_sha") and row["world_sha"] != meta["world_sha"]:
            raise ValueError("Evaluation and batch world fingerprints differ")
        case = cases[row["case"]]["case"]
        if row.get("T") != case["prediction_context"]["prediction_time_T"]:
            raise ValueError("Management rejudging requires the saved canonical endpoint")
        if row.get("q_question_id") and row["q_question_id"] != f"{row['case']}@T{row['T']}":
            raise ValueError("Saved question identity differs from the selected case")
    files = required + response_files + ([meta_path] if meta_path.is_file() else [])
    source = {"batch": str(batch), "files": {str(p.relative_to(batch)): digest(p) for p in files},
              "world_sha": meta.get("world_sha"), "judge_parts": meta.get("judge_parts"),
              "gold_provenance": meta.get("gold_provenance"), "cases": len(cases)}
    return cases, rows, select_responses(rows, responses), source


def prepare(batches, out, *, run_id, mode="per_case", limit_per_model=None, include_controls=False,
            keys_from=None, policy_path=None, controls_per_pack=0, key_file=None):
    if keys_from is not None:
        if batches or limit_per_model is not None or include_controls:
            raise ValueError("Repeat preparation cannot change saved input selection")
        if Path(keys_from).is_file():
            return _prepare_prepared(keys_from, out, run_id=run_id, policy_path=policy_path, key_file=key_file)
        if policy_path is not None:
            raise ValueError("Use --keys-from PATH/cells.jsonl with --policy to select all prepared cells")
        return _prepare_repeat(keys_from, out, run_id=run_id)
    if key_file is not None:
        raise ValueError("A key file requires prepared cells.jsonl")
    if type(controls_per_pack) is not int or controls_per_pack < 0 or controls_per_pack and include_controls:
        raise ValueError("Fixed controls-per-pack must be nonnegative and separate from all-case controls")
    from haenv_kernel.solver import _extract_json
    if mode not in {"per_case", "per_item"}:
        raise ValueError("Unknown request geometry")
    if not run_id or not batches or len({Path(p).resolve() for p in batches}) != len(batches):
        raise ValueError("Explicit run id and unique source batches required")
    if limit_per_model is not None and (type(limit_per_model) is not int or limit_per_model <= 0):
        raise ValueError("Pilot limit must be positive")
    out = Path(out).resolve()
    _outside_sources(out, batches)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prepare requires a new output directory; use run to resume")
    policy = load_policy(policy_path)
    cells, sources, snapshots = [], [], {}
    seen_controls = set()
    for batch in batches:
        cases, rows, selected, source = _batch_inputs(batch, policy)
        sources.append(source)
        by_identity = {(r["case"], r["solver"]): r for r in rows}
        selected.sort(key=lambda r: (r["solver"], sha256(["management-sample-v1", r["case"]])))
        taken = Counter()
        for cell in selected:
            if limit_per_model is not None and taken[cell["solver"]] >= limit_per_model:
                continue
            taken[cell["solver"]] += 1
            saved = cases[cell["case"]]
            snap, pack = saved["snapshot"], saved["pack"]
            snap_key = snap["facts_sha256"]
            snapshots[snap_key] = snap
            rubric = derive(pack, snap, policy)
            row = by_identity[(cell["case"], cell["solver"])]
            identity = {"batch": source["batch"], "case": cell["case"], "solver": cell["solver"],
                        "endpoint": cell["endpoint"], "raw_sha256": cell["raw_sha256"],
                        "facts_sha256": snap_key, "world_sha": row["world_sha"],
                        "source_judging_sha16": row["judging_sha16"],
                        "question_id": row.get("q_question_id"), "world_ref": row.get("world_ref"),
                        "pack_judging_sha16": row.get(f"{pack}_judging_sha16")}
            status, plan, answer = cell["status"], None, None
            if status == "ready":
                try:
                    answer = _extract_json(cell["response"]["raw"])
                except (ValueError, TypeError):
                    status = "unparseable_response"
                else:
                    if not isinstance(answer, dict):
                        status = "unparseable_response"
                    elif "management_plan" not in answer:
                        status = "not_collected"
                    elif not isinstance(answer["management_plan"], str):
                        status = "invalid_management_plan"
                    else:
                        plan = answer["management_plan"]  # exact original string, including whitespace
            key = sha256(identity)
            cells.append({"key": key, "source": identity, "status": status, "management_plan": plan,
                          "answer": answer, "answer_sha256": sha256(answer) if isinstance(answer, dict) else None,
                          "plan_sha256": sha256(plan) if plan is not None else None,
                          "snapshot_key": snap_key, "rubric": rubric,
                          "core_decision": eval_core_decision(pack, row)})
            control_key = (cell["case"], snap_key)
            if include_controls and control_key not in seen_controls:
                seen_controls.add(control_key)
                for name, candidate in controls(rubric).items():
                    if candidate is None:
                        continue
                    candidate = answer_json(candidate)
                    text = candidate["management_plan"]
                    origin = {**identity, "solver": f"control:{name}", "raw_sha256": None, "control": name}
                    cells.append({"key": sha256(origin), "source": origin, "status": "ready",
                                  "management_plan": text, "plan_sha256": sha256(text),
                                  "answer": candidate, "answer_sha256": sha256(candidate),
                                  "snapshot_key": snap_key, "rubric": rubric,
                                  "core_decision": control_core_decision(rubric, name)})
    control_questions = {}
    if controls_per_pack:
        by_pack = {}
        for cell in cells:
            by_pack.setdefault(cell["rubric"]["pack"], {})[(cell["source"]["case"], cell["snapshot_key"])] = cell
        for pack, candidates in sorted(by_pack.items()):
            if len(candidates) < controls_per_pack:
                raise ValueError("Not enough saved questions for the requested controls")
            chosen = sorted(candidates.items(), key=lambda pair: sha256(["management-dev-controls-v1", pair[0]]))[:controls_per_pack]
            control_questions[pack] = [list(k) for k, _ in chosen]
            for _, base in chosen:
                for name, answer in controls(base["rubric"]).items():
                    if answer is None:
                        continue
                    answer = answer_json(answer)
                    source = {**base["source"], "solver": f"control:{name}", "raw_sha256": None, "control": name}
                    cells.append({**deepcopy(base), "key": sha256(source), "source": source, "status": "ready",
                                  "answer": answer, "answer_sha256": sha256(answer), "management_plan": answer["management_plan"],
                                  "plan_sha256": sha256(answer["management_plan"]), "core_decision": control_core_decision(base["rubric"], name)})
    # All correspondence/hash checks above finish before the first output is written.
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "cells.jsonl", cells)
    write_jsonl(out / "snapshots.jsonl", [{"key": k, "snapshot": s} for k, s in snapshots.items()])
    manifest = {"run_id": run_id, "rubric_version": policy["version"], "policy": policy,
                "code": code_state(), "mode": mode, "sources": sources,
                "cells_sha256": digest(out / "cells.jsonl"), "snapshots_sha256": digest(out / "snapshots.jsonl"),
                "states": dict(Counter(c["status"] for c in cells)),
                "sampling": {"limit_per_model_per_batch": limit_per_model,
                             "selection": "fixed case hash, before judgment", "include_controls": include_controls}}
    manifest["sampling"].update(controls_per_pack=controls_per_pack, control_questions=control_questions)
    atomic_json(out / "manifest.json", manifest)
    return manifest


def _prepare_prepared(cells_path, out, *, run_id, policy_path=None, key_file=None):
    """Copy the whole sealed input cohort; only judge settings may change."""
    from haenv.scoring.management.precision_rejudge import _unique
    cells_path, out = Path(cells_path).resolve(), Path(out).resolve()
    if cells_path.name != "cells.jsonl":
        raise ValueError("Prepared keys must name the sealed cells.jsonl")
    previous = cells_path.parent
    old = json.loads((previous / "manifest.json").read_text())
    _outside_sources(out, [previous] + [Path(s["batch"]) for s in old["sources"]])
    if not run_id or run_id == old["run_id"] or out.exists() and any(out.iterdir()):
        raise ValueError("Prepared replay requires a new run id and empty new output directory")
    names = ["cells.jsonl", "snapshots.jsonl"]
    for name in names:
        if digest(previous / name) != old[name.removesuffix(".jsonl") + "_sha256"]:
            raise ValueError("Stored prepared inputs changed")
    cells = _unique(read_jsonl(cells_path))
    if key_file is not None:
        selected = json.loads(Path(key_file).read_text())
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or not set(selected) <= cells.keys():
            raise ValueError("Key file must list unique original prepared cell keys")
        cells = {k: c for k, c in cells.items() if k in set(selected)}
    snapshots = _unique(read_jsonl(previous / "snapshots.jsonl"))
    policy = load_policy(policy_path) if policy_path else deepcopy(old["policy"])
    if {k: v for k, v in policy.items() if k != "judge"} != {k: v for k, v in old["policy"].items() if k != "judge"}:
        raise ValueError("Prepared replay can change only the judge, never scoring, templates or facts")
    for cell in cells.values():
        verify_snapshot(snapshots[cell["snapshot_key"]]["snapshot"], policy)
        if cell["rubric"]["policy_sha256"] != sha256(old["policy"]):
            # A prepared replay records the criterion policy separately from the judge policy.
            origin = old.get("prepared_inputs", {}).get("criterion_policy_sha256")
            if cell["rubric"]["policy_sha256"] != origin:
                raise ValueError("Saved rubric policy differs from its criterion policy")
    if "saved_inputs" in old:
        names.append("v5_profiles.jsonl")
        if digest(previous / names[-1]) != old["saved_inputs"]["profiles_sha256"]:
            raise ValueError("Saved v5 profiles changed")
    blobs = {name: (previous / name).read_bytes() for name in names}
    manifest = {**deepcopy(old), "run_id": run_id, "policy": policy, "code": code_state(),
                "prepared_inputs": {"previous": str(previous), "previous_run_id": old["run_id"],
                                    "sealed_files": {str(previous / n): digest(previous / n) for n in names},
                                    "criterion_policy_sha256": old.get("prepared_inputs", {}).get(
                                        "criterion_policy_sha256", sha256(old["policy"])),
                                    "old_raw_responses_copied": 0}}
    manifest.pop("repeat", None)
    out.mkdir(parents=True)
    for name, blob in blobs.items():
        (out / name).write_bytes(blob)
    if key_file is not None:
        write_jsonl(out / "cells.jsonl", cells.values())
        manifest["cells_sha256"] = digest(out / "cells.jsonl")
        manifest["states"] = dict(Counter(c["status"] for c in cells.values()))
        manifest["prepared_inputs"].update(selected_keys=list(cells), key_file=str(Path(key_file).resolve()),
                                           key_file_sha256=digest(Path(key_file)))
    atomic_json(out / "manifest.json", manifest)
    return manifest


def _prepare_repeat(previous, out, *, run_id):
    """Fresh requests for the completed cell snapshot, with identical judge content."""
    from haenv.scoring.management.precision_rejudge import _identity
    from haenv.scoring.management.precision_rejudge import _unique
    previous, out = Path(previous).resolve(), Path(out).resolve()
    old = json.loads((previous / "manifest.json").read_text())
    _outside_sources(out, [previous] + [Path(s["batch"]) for s in old["sources"]])
    if not run_id or run_id == old["run_id"] or out.exists() and any(out.iterdir()):
        raise ValueError("Repeat requires a new run id and empty new output directory")
    for name in ("cells", "snapshots"):
        if digest(previous / f"{name}.jsonl") != old[f"{name}_sha256"]:
            raise ValueError("Stored repeat inputs changed")
    cells = _unique(read_jsonl(previous / "cells.jsonl"))
    snapshots = _unique(read_jsonl(previous / "snapshots.jsonl"))
    paths = sorted((previous / "results").glob("*.json"))
    if not paths:
        raise ValueError("Repeat requires existing results/*.json cells")
    selected, checked = [], 0
    for path in paths:
        record = json.loads(path.read_text())
        if record["key"] != path.stem or record["key"] not in cells:
            raise ValueError("Repeat result has no matching saved cell")
        cell = cells[record["key"]]
        _identity(cell, record, old["rubric_version"])
        snap = snapshots[cell["snapshot_key"]]["snapshot"]
        verify_snapshot(snap, old["policy"])
        tasks = _tasks(cell, snap, {**old, "run_id": run_id})
        for name in record.get("raw_judge_files", []):
            raw_path = (previous / name).resolve()
            if previous not in raw_path.parents:
                raise ValueError("Repeat raw request escapes its source")
            sample = json.loads(raw_path.read_text())
            if sample["request_sha256"] != sha256(sample["request"]):
                raise ValueError("Stored repeat request hash differs")
            task = next((t for t, _ in tasks if list(t.item_ids) == sample["task"]["item_ids"]), None)
            if task is None:
                raise ValueError("Repeat item geometry differs from saved request")
            request = request_for(task, index=sample["request"]["index"])
            if (request.prompt != sample["request"]["prompt"]
                    or request.response_format != sample["request"]["response_format"]
                    or task.judge_model != sample["task"]["judge_model"]
                    or request.sample_id == sample["request"]["sample_id"]):
                raise ValueError("Repeat prompt differs or request identity was reused")
            checked += 1
        selected.append(cell)
    out.mkdir(parents=True)
    write_jsonl(out / "cells.jsonl", selected)
    used = {c["snapshot_key"] for c in selected}
    write_jsonl(out / "snapshots.jsonl", [s for k, s in snapshots.items() if k in used])
    code = code_state()
    for name in ("judging_criteria", "judging_total_sha16"):
        if name in old["code"]:
            code[name] = old["code"][name]
    manifest = {"run_id": run_id, "rubric_version": old["rubric_version"], "policy": old["policy"],
                "code": code, "mode": old["mode"], "sources": old["sources"],
                "cells_sha256": digest(out / "cells.jsonl"),
                "snapshots_sha256": digest(out / "snapshots.jsonl"),
                "states": dict(Counter(c["status"] for c in selected)),
                "repeat": {"previous": str(previous), "previous_code": old["code"],
                           "selected_keys": [c["key"] for c in selected],
                           "byte_identical_requests_checked": checked, "old_raw_responses_copied": 0}}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def validate(out):
    out = Path(out).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    now = code_state()
    if manifest["code"]["management_code_sha16"] != now["management_code_sha16"]:
        raise ValueError("Management code or registry changed since preparation")
    if (digest(out / "cells.jsonl") != manifest["cells_sha256"]
            or digest(out / "snapshots.jsonl") != manifest["snapshots_sha256"]):
        raise ValueError("Prepared cells or snapshots changed")
    for source in manifest["sources"]:
        _outside_sources(out, [Path(source["batch"])])
        for name, expected in source["files"].items():
            if digest(Path(source["batch"]) / name) != expected:
                raise ValueError("Source batch changed since preparation")
    if "saved_inputs" in manifest:
        inputs = manifest["saved_inputs"]
        if digest(out / "v5_profiles.jsonl") != inputs["profiles_sha256"]:
            raise ValueError("Saved v5 profile archive changed")
        for name, expected in inputs["sealed_files"].items():
            if digest(Path(name)) != expected:
                raise ValueError("Selected completed source input changed")
    for name, expected in manifest.get("prepared_inputs", {}).get("sealed_files", {}).items():
        if digest(Path(name)) != expected:
            raise ValueError("Selected prepared source input changed")
    selection = manifest.get("prepared_inputs", {})
    if "key_file" in selection and digest(Path(selection["key_file"])) != selection["key_file_sha256"]:
        raise ValueError("Prepared key selection changed")
    return manifest


def prepare_saved(previous_runs, out, *, run_id, controls_per_pack=0):
    """Rejudge completed saved answers under v6; no old quality vote is reused."""
    from copy import deepcopy
    from haenv.scoring.management.precision_rejudge import _identity
    from haenv.scoring.management.precision_rejudge import _unique
    policy = load_policy()
    if policy["version"] not in {"management-rubric-v6", "management-rubric-v6.1", "management-rubric-v6.2"} or not run_id or not previous_runs:
        raise ValueError("Saved-answer prepare requires the v6 policy and explicit source runs")
    if type(controls_per_pack) is not int or controls_per_pack < 0:
        raise ValueError("controls_per_pack must be a nonnegative integer")
    out = Path(out).resolve()
    previous_runs = [Path(p).resolve() for p in previous_runs]
    if len(set(previous_runs)) != len(previous_runs):
        raise ValueError("Duplicate source run")
    _outside_sources(out, previous_runs)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prepare requires a new output directory")
    cells, snapshots, profiles, sources, seal = {}, {}, [], {}, {}
    selection = {}
    for previous in previous_runs:
        old = json.loads((previous / "manifest.json").read_text())
        if old["rubric_version"] != "management-rubric-v5" or run_id == old["run_id"]:
            raise ValueError("v6 requires saved v5 inputs and a new run id")
        for name in ("cells", "snapshots"):
            file = previous / f"{name}.jsonl"
            if digest(file) != old[f"{name}_sha256"]:
                raise ValueError("Saved v5 cells or snapshots changed")
        for name in ("manifest.json", "cells.jsonl", "snapshots.jsonl"):
            seal[str(previous / name)] = digest(previous / name)
        old_cells = _unique(read_jsonl(previous / "cells.jsonl"))
        old_snaps = _unique(read_jsonl(previous / "snapshots.jsonl"))
        paths = sorted((previous / "results").glob("*.json"))
        selection[str(previous)] = [path.stem for path in paths]
        for path in paths:
            record = json.loads(path.read_text())
            if record["key"] != path.stem or record["key"] not in old_cells:
                raise ValueError("Completed result has no matching source cell")
            old_cell = old_cells[record["key"]]
            _identity(old_cell, record, old["rubric_version"])
            if record.get("offline") is True:
                raise ValueError("Real-answer selection must not import offline judge runs")
            snap = old_snaps[old_cell["snapshot_key"]]
            verify_snapshot(snap["snapshot"], policy)
            cell = {**deepcopy(old_cell), "rubric": derive(old_cell["rubric"]["pack"], snap["snapshot"], policy)}
            if cell["key"] in cells:
                if canonical(cells[cell["key"]]) != canonical(cell):
                    raise ValueError("Overlapping saved cells differ")
                continue
            cells[cell["key"]] = cell
            snapshots[cell["snapshot_key"]] = snap
            profiles.append({"key": cell["key"], "previous": str(previous), "record": record})
            seal[str(path)] = digest(path)
        for source in old["sources"]:
            _outside_sources(out, [Path(source["batch"])])
            if source["batch"] in sources and sources[source["batch"]] != source:
                raise ValueError("Source batch definitions differ")
            sources[source["batch"]] = source
    if not cells:
        raise ValueError("No completed saved cells")
    controls_selected = {}
    if controls_per_pack:
        from haenv.scoring.management.judge import controls
        by_pack = {}
        for cell in cells.values():
            pack = cell["rubric"]["pack"]
            if pack in {"p2", "p3", "p4"}:
                by_pack.setdefault(pack, {})[(cell["source"]["case"], cell["snapshot_key"])] = cell
        if set(by_pack) != {"p2", "p3", "p4"}:
            raise ValueError("Sample controls require all three packs")
        selected_controls = []
        for pack, cases in sorted(by_pack.items()):
            if len(cases) < controls_per_pack:
                raise ValueError("Not enough saved questions for the requested controls")
            chosen = sorted(cases.items(), key=lambda pair: sha256(["management-v6-controls", pair[0]]))[:controls_per_pack]
            controls_selected[pack] = [list(key) for key, _ in chosen]
            for _, base in chosen:
                for name, answer in controls(base["rubric"]).items():
                    if answer is None:
                        continue
                    answer = answer_json(answer)
                    source = {**base["source"], "solver": f"control:{name}", "raw_sha256": None, "control": name}
                    selected_controls.append({**deepcopy(base), "key": sha256(source), "source": source,
                        "status": "ready", "answer": answer, "answer_sha256": sha256(answer),
                        "management_plan": answer["management_plan"], "plan_sha256": sha256(answer["management_plan"]),
                        "core_decision": control_core_decision(base["rubric"], name)})
        for cell in selected_controls:
            cells[cell["key"]] = cell
    # Complete all input validation before any output mutation.
    for source in sources.values():
        for name, expected in source["files"].items():
            if digest(Path(source["batch"]) / name) != expected:
                raise ValueError("Original source batch changed")
    out.mkdir(parents=True)
    write_jsonl(out / "cells.jsonl", cells.values())
    write_jsonl(out / "snapshots.jsonl", snapshots.values())
    write_jsonl(out / "v5_profiles.jsonl", profiles)
    manifest = {"run_id": run_id, "rubric_version": policy["version"], "policy": policy,
                "code": code_state(), "mode": "per_case", "sources": list(sources.values()),
                "cells_sha256": digest(out / "cells.jsonl"), "snapshots_sha256": digest(out / "snapshots.jsonl"),
                "states": dict(Counter(c["status"] for c in cells.values())),
                "saved_inputs": {"selection": selection, "sealed_files": seal,
                                 "profiles_sha256": digest(out / "v5_profiles.jsonl"),
                                 "real_cells": len(profiles), "controls_per_pack": controls_per_pack,
                                 "control_questions": controls_selected, "v5_votes_reused": 0}}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def cell_answer(cell):
    # Older saved runs can still be reaggregated from their original plan-only input.
    return cell["answer"] if "answer" in cell else {"management_plan": cell["management_plan"]}


def _tasks(cell, snapshot, manifest):
    rubric, answer = cell["rubric"], cell_answer(cell)
    judge_items = [i for i in rubric["items"] if not i.get("core_decision")]
    groups = ([None] if manifest["mode"] == "per_case" else [[i["id"]] for i in judge_items]) if judge_items else []
    return [(make_task(rubric, snapshot, answer, manifest["policy"], run_id=f"{manifest['run_id']}:{cell['key']}", items=g),
             judge_items if g is None else [i for i in judge_items if i["id"] in g]) for g in groups]


def _attempts(task, policy):
    for index in range(1, policy["judge"].get("empty_transport_retries", 0) + 2):
        req = request_for(task, index=index)
        yield req, f"{sha256(req.sample_id)}.json"


def _read_sample(path, request):
    sample = json.loads(path.read_text())
    if sample["request_sha256"] != sha256(asdict(request)):
        raise ValueError("Raw judge cache request differs")
    return sample


def run(out, *, offline=True, ledger_path=None, limit_usd=None, cfg=None, concurrency=None, max_cells=None):
    out = Path(out).resolve()
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(out, offline=offline, ledger_path=ledger_path, limit_usd=limit_usd,
                    cfg=cfg, concurrency=concurrency, max_cells=max_cells)


def _run(out, *, offline, ledger_path, limit_usd, cfg, concurrency, max_cells):
    manifest = validate(out)
    policy = manifest["policy"]
    concurrency = policy["judge"]["concurrency"] if concurrency is None else concurrency
    if max_cells is not None and (type(max_cells) is not int or max_cells <= 0):
        raise ValueError("max_cells must be positive")
    # An output run cannot mix offline proxy labels with real judge labels.
    execution = {"judge": "deterministic_stub" if offline else policy["judge"]["model_id"],
                 "offline": offline, "ledger": str(Path(ledger_path).resolve()) if ledger_path else None}
    execution_path = out / "execution.json"
    if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
        raise ValueError("Do not mix offline/paid execution or ledgers in one run")
    ledger = None
    factory = None
    if not offline:
        if ledger_path is None or limit_usd is None or cfg is None:
            raise ValueError("Paid judging requires an explicit independent ledger, cap and configuration")
        _outside_sources(ledger_path, [Path(s["batch"]) for s in manifest["sources"]])
        from copy import copy
        from haenv.scoring.semantic.pipeline import fetch_prices
        from haenv.runtime.pricing import PriceSchedule
        from haenv.scoring.semantic.transport import PricedJudge
        from haenv.runtime.solvers import load_env_file
        from haenv.runtime.solvers import solver_for_spec
        from haenv.runtime.transport import judge_spec
        spec = judge_spec(cfg["models"][policy["judge"]["model_key"]])
        if (spec.get("backend", "openrouter") != policy["judge"]["backend"]
                or spec.get("model") != policy["judge"]["model_id"]):
            raise ValueError("Configured management judge route/model differs from the sealed policy")
        spec.update(reasoning_effort=policy["judge"].get("reasoning_effort"), retries=0)
        if policy["judge"].get("sampling"):
            spec["sampling"] = policy["judge"]["sampling"]
        if policy["judge"]["backend"] == "openrouter":
            spec["max_tokens"] = policy["judge"]["max_tokens"]
        solver = solver_for_spec(policy["judge"]["model_key"], spec, load_env_file(cfg.get("env_file")), timeout=900, retries=0)
        if solver is None:
            raise ValueError("Configured judge could not be constructed")
        solver.pool = None
        solver.max_tokens = policy["judge"]["max_tokens"]
        ledger = BudgetLedger(Path(ledger_path), limit_usd=limit_usd)
        if policy["judge"]["backend"] == "google":
            from haenv.scoring.management.transport import NativeManagementJudge
            from haenv.runtime.native_accounting import NativePrices
            prices = NativePrices.verified(solver.model, "google")
            metadata = {"id": solver.model, "backend": "google", "native_tariff": asdict(prices)}
            factory = lambda: NativeManagementJudge(copy(solver), prices, ledger, out / "receipts")
        else:
            metadata = fetch_prices(cfg, policy)
            prices = PriceSchedule.from_metadata(metadata, solver.model)
            if policy["judge"]["reasoning_effort"] == "high" and not policy["judge"].get("sampling"):
                factory = lambda: PricedJudge(copy(solver), prices, ledger, receipts=out / "receipts", reply_parser=transport_reply)
            else:
                from haenv.scoring.management.transport import ConfiguredManagementJudge
                factory = lambda: ConfiguredManagementJudge(copy(solver), prices, ledger, out / "receipts")
        atomic_json(out / "price-metadata.json", metadata)
    atomic_json(execution_path, execution)
    samples, results = out / "raw_judge", out / "results"
    samples.mkdir(exist_ok=True); results.mkdir(exist_ok=True)
    snapshots = {r["key"]: r["snapshot"] for r in read_jsonl(out / "snapshots.jsonl")}
    cells = read_jsonl(out / "cells.jsonl")
    pending = [c for c in cells if c["status"] == "ready" and not (results / f"{c['key']}.json").exists()]
    # Same case together: its answers share the instruction + rubric + facts prefix, so the
    # followers can ride the prefix cache the first request writes (stable within a case).
    pending.sort(key=lambda c: c["snapshot_key"])
    selected = pending if max_cells is None else pending[:max_cells]
    stop = threading.Event()
    stop_reasons = []
    stop_lock = threading.Lock()

    def one(cell):
        tasks = _tasks(cell, snapshots[cell["snapshot_key"]], manifest)
        answer = cell_answer(cell)
        is_empty = empty_answer(answer)
        full_stub = stub_reply(cell["rubric"]["items"], answer) if offline or is_empty else None
        full_verdicts = json.loads(full_stub.raw)["verdicts"] if full_stub else None
        call = factory() if factory and full_stub is None else None
        verdicts, errors, requests, attempt_files = {}, [], [], []
        for task, items in tasks:
            for req, filename in _attempts(task, policy):
                path = samples / filename
                attempt_files.append(str(path.relative_to(out)))
                if path.exists():
                    sample = _read_sample(path, req)
                else:
                    if stop.is_set():
                        return None
                    try:
                        if full_stub is not None:
                            raw = canonical({"verdicts": {i["id"]: full_verdicts[i["id"]] for i in items}})
                            usage = full_stub.usage
                        else:
                            reply = call(req)
                            raw, usage = reply.raw, reply.usage
                            if raw is None or isinstance(raw, str) and not raw.strip():
                                raise EmptyJudgeResponse(raw, usage)
                        sample = {"request_sha256": sha256(asdict(req)), "request": asdict(req),
                                  "task": asdict(task), "raw": raw, "usage": usage,
                                  "origin": "deterministic_empty" if is_empty and not offline else execution["judge"]}
                    except BudgetExceeded as exc:
                        with stop_lock:
                            stop_reasons.append(str(exc))
                        stop.set()
                        return None
                    except Exception as exc:
                        sample = {"request_sha256": sha256(asdict(req)), "request": asdict(req),
                                  "task": asdict(task), "raw": getattr(exc, "raw", None),
                                  "usage": getattr(exc, "usage", None), "error_class": type(exc).__name__,
                                  "retryable": isinstance(exc, (EmptyJudgeResponse, UnknownCostCallFailure,
                                                                ReconciledCallFailure, RejectedCallFailure))}
                    # Persist each attempt before retrying or interpreting. Each paid attempt
                    # has its own sample id, reservation and durable transport receipt.
                    atomic_json(path, sample)
                if not sample.get("retryable"):
                    break
            # Reaggregation uses only the final attempt; failed earlier attempts remain auditable.
            requests.append(str(path.relative_to(out)))
            if sample["raw"] is None:
                errors.append({"request_id": req.sample_id, "error": sample["error_class"]})
                continue
            try:
                verdicts.update(parse_reply(sample["raw"], items, answer))
            except (ValueError, TypeError, KeyError) as exc:
                errors.append({"request_id": req.sample_id, "error": type(exc).__name__})
        verdicts = with_core_verdicts(cell["rubric"], verdicts, cell["core_decision"])
        summary = score(cell["rubric"], None if errors else verdicts, policy, core_decision=cell["core_decision"])
        result = {"key": cell["key"], "source": cell["source"], "collection_status": "collected",
                  "rubric_version": cell["rubric"]["version"], "facts_sha256": cell["snapshot_key"],
                  "code": manifest["code"], "plan_sha256": cell["plan_sha256"],
                  "answer_sha256": cell.get("answer_sha256"),
                  "core_decision": cell["core_decision"],
                  "verdicts": verdicts, "raw_judge_files": requests,
                  "raw_judge_attempt_files": attempt_files, "errors": errors,
                  "summary": summary, "offline": offline}
        atomic_json(results / f"{cell['key']}.json", result)
        return summary["status"]

    run_bounded(selected, one, concurrency=concurrency, stop=stop)
    overlay = []
    for cell in cells:
        path = results / f"{cell['key']}.json"
        if path.exists():
            result = json.loads(path.read_text())
            # Cache is a raw-response cache: reconstruct every score, never trust saved totals.
            verdicts, errors = {}, []
            for task, items in _tasks(cell, snapshots[cell["snapshot_key"]], manifest):
                for req, filename in _attempts(task, policy):
                    sample = _read_sample(samples / filename, req)
                    if not sample.get("retryable"):
                        break
                try:
                    verdicts.update(parse_reply(sample["raw"], items, cell_answer(cell)))
                except (ValueError, TypeError, KeyError):
                    errors.append(True)
            verdicts = with_core_verdicts(cell["rubric"], verdicts, cell["core_decision"])
            rebuilt = score(cell["rubric"], None if errors else verdicts, policy, core_decision=cell["core_decision"])
            if result["verdicts"] != verdicts or result["summary"] != rebuilt:
                raise ValueError("Saved management result differs from raw judge responses")
        else:
            result = {"key": cell["key"], "source": cell["source"],
                      "collection_status": "未采集" if cell["status"] == "not_collected" else cell["status"],
                      "rubric_version": manifest["rubric_version"], "facts_sha256": cell["snapshot_key"],
                      "code": manifest["code"], "plan_sha256": cell["plan_sha256"],
                      "core_decision": cell["core_decision"],
                      "verdicts": with_core_verdicts(cell["rubric"], {}, cell["core_decision"]),
                      "summary": score(cell["rubric"], None, policy, core_decision=cell["core_decision"]), "offline": offline}
            if cell["status"] == "ready":
                result["collection_status"] = "pending"
        overlay.append(result)
    write_jsonl(out / "rubric_results.jsonl", overlay)
    summary = {"run_id": manifest["run_id"], "rubric_version": manifest["rubric_version"],
               "offline": offline, "paid_calls": 0 if offline else None,
               "states": dict(Counter(r["collection_status"] if r["collection_status"] != "collected" else r["summary"]["status"] for r in overlay)),
               "budget": ledger.summary() if ledger else None,
               "stop_reason": stop_reasons[0] if stop_reasons else None,
               "source_eval_overwritten": False}
    atomic_json(out / "summary.json", summary)
    return summary


def rescore(out, destination, *, policy_path=None):
    """Replay quality votes and code decisions without acquiring another judge vote."""
    out, destination = Path(out).resolve(), Path(destination).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    policy = load_policy(policy_path)
    old = manifest["policy"]
    if {k: v for k, v in policy.items() if k not in {"version", "scoring", "source"}} != {k: v for k, v in old.items() if k not in {"version", "scoring", "source"}}:
        raise ValueError("Rescore can change aggregation only, not the judge criteria or protocol")
    _outside_sources(destination, [Path(s["batch"]) for s in manifest["sources"]])
    if destination == out or out in destination.parents:
        raise ValueError("Rescore must use a separate output directory")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Rescore output must be new")
    if digest(out / "cells.jsonl") != manifest["cells_sha256"]:
        raise ValueError("Stored management plans or rubrics changed")
    evaluations = {}
    for source in manifest["sources"]:
        path = Path(source["batch"]) / "eval.jsonl"
        if digest(path) != source["files"]["eval.jsonl"]:
            raise ValueError("Source evaluation changed since preparation")
        evaluations[source["batch"]] = {(r["case"], r["solver"]): r for r in load_rows(path)}
    cells = {c["key"]: c for c in read_jsonl(out / "cells.jsonl")}
    records = read_jsonl(out / "rubric_results.jsonl")
    if len(records) != len(cells) or {r["key"] for r in records} != set(cells):
        raise ValueError("Stored results do not match prepared cells")
    aggregation_code = code_state()
    for record in records:
        cell = cells[record["key"]]
        source = cell["source"]
        if "control" in source:
            decision = control_core_decision(cell["rubric"], source["control"])
        else:
            row = evaluations[source["batch"]].get((source["case"], source["solver"]))
            if row is None:
                raise ValueError("Prepared answer has no matching saved evaluation")
            decision = eval_core_decision(cell["rubric"]["pack"], row)
        if "core_decision" in cell and cell["core_decision"] != decision:
            raise ValueError("Prepared code decision differs from its source")
        cell["core_decision"] = decision
        cell["rubric"]["version"] = policy["version"]
        for item in cell["rubric"]["items"]:
            item["judgment_source"] = "code" if item.get("core_decision") else "judge"
        rebuilt, invalid = {}, False
        for filename in record.get("raw_judge_files", []):
            path = (out / filename).resolve()
            if out not in path.parents:
                raise ValueError("Raw response path escapes its run")
            sample = json.loads(path.read_text())
            if sample["request_sha256"] != sha256(sample["request"]):
                raise ValueError("Stored raw judgment request changed")
            items = [i for i in cell["rubric"]["items"] if i["id"] in sample["task"]["item_ids"]]
            try:
                rebuilt.update(parse_reply(sample["raw"], items, cell_answer(cell), ignore_core=True))
            except (ValueError, TypeError, KeyError):
                invalid = True
        record["original_rubric_version"] = record["rubric_version"]
        record["rubric_version"] = policy["version"]
        record["aggregation_policy_sha256"] = sha256(policy)
        record["aggregation_code"] = aggregation_code
        record["core_decision"] = decision
        record["verdicts"] = with_core_verdicts(cell["rubric"], rebuilt, decision)
        has_quality_items = any(not i.get("core_decision") for i in cell["rubric"]["items"])
        unavailable = invalid or record["collection_status"] != "collected" or has_quality_items and not rebuilt
        record["summary"] = score(cell["rubric"], None if unavailable else record["verdicts"],
                                  policy, core_decision=decision)
    destination.mkdir(parents=True)
    write_jsonl(destination / "cells.jsonl", cells.values())
    write_jsonl(destination / "rubric_results.jsonl", records)
    report = {"cells": len(records), "paid_calls": 0, "out": str(destination), "rescore_from": str(out),
              "rubric_version": policy["version"], "policy": policy, "code": aggregation_code,
              "source_manifest_sha256": digest(out / "manifest.json"),
              "states": dict(Counter(r["summary"]["status"] for r in records))}
    atomic_json(destination / "rescore.json", report)
    return report
