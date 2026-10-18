"""Precision-only requests, sealed reuse of prior atomic verdicts, and full rescoring."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from collections import Counter

from haenv.scoring.management.judge import PROTOCOL
from haenv.scoring.management.judge import answer_json
from haenv.scoring.management.judge import parse_reply
from haenv.scoring.management.judge import request_for
from haenv.scoring.management.pipeline import _outside_sources
from haenv.scoring.management.pipeline import cell_answer
from haenv.scoring.management.pipeline import code_state
from haenv.scoring.management.pipeline import read_jsonl
from haenv.scoring.management.pipeline import eval_core_decision
from haenv.scoring.management.pipeline import validate
from haenv.scoring.management.pipeline import write_jsonl
from haenv.scoring.management.rubric import canonical
from haenv.scoring.management.rubric import control_core_decision
from haenv.scoring.management.rubric import derive
from haenv.scoring.management.rubric import load_policy
from haenv.scoring.management.rubric import score
from haenv.scoring.management.rubric import sha256
from haenv.scoring.management.rubric import verify_snapshot
from haenv.scoring.management.rubric import with_core_verdicts
from haenv.runtime.row_store import load_rows
from haenv.scoring.semantic.judge import JudgeTask
from haenv.scoring.semantic.parallel import atomic_json
from haenv.scoring.semantic.seal import digest

OLD_VERSION = "management-rubric-v3"
NEW_VERSION = "management-rubric-v5"
PRIOR_VERSIONS = {OLD_VERSION, "management-rubric-v3c", "management-rubric-v4", NEW_VERSION}


def _unique(rows):
    result = {row["key"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate saved cell identity")
    return result


def _raw_verdicts(directory, cell, record, version):
    """Read final attempts only; keep invalid old cells explicitly unscored."""
    rebuilt, invalid, files = {}, False, {}
    allowed = {i["id"]: i for i in cell["rubric"]["items"]}
    for name in record.get("raw_judge_files", []):
        path = (directory / name).resolve()
        if directory not in path.parents:
            raise ValueError("Raw response path escapes its run")
        files[str(path)] = digest(path)
        sample = json.loads(path.read_text())
        task = JudgeTask(**sample["task"])
        request = request_for(task, index=sample["request"]["index"])
        if (sample["request_sha256"] != sha256(sample["request"])
                or canonical(sample["request"]) != canonical(asdict(request))
                or task.rubric_version != f"{version}:{PROTOCOL}:{cell['snapshot_key']}"
                or task.answer_text != canonical(answer_json(cell_answer(cell)))
                or task.judge_model != load_policy()["judge"]["model_id"]
                or not set(task.item_ids) <= set(allowed)
                or set(task.item_ids) & set(rebuilt)):
            raise ValueError("Stored raw request version, items or facts differ")
        try:
            rebuilt.update(parse_reply(sample["raw"], [allowed[k] for k in task.item_ids],
                                       cell_answer(cell), ignore_core=True))
        except (ValueError, TypeError, KeyError):
            invalid = True
    saved = {k: v for k, v in (record.get("verdicts") or {}).items() if not allowed.get(k, {}).get("core_decision")}
    if saved != rebuilt:
        raise ValueError("Saved verdicts differ from raw responses")
    return rebuilt, invalid, files


def _identity(cell, record, version):
    if (record["key"] != cell["key"] or record["source"] != cell["source"]
            or cell["key"] != sha256(cell["source"])
            or record["rubric_version"] != version or cell["rubric"]["version"] != version
            or record["facts_sha256"] != cell["snapshot_key"]
            or cell["rubric"]["facts_sha256"] != cell["snapshot_key"]
            or cell["source"]["facts_sha256"] != cell["snapshot_key"]
            or record.get("plan_sha256") != cell["plan_sha256"]
            or record.get("collection_status") == "collected" and record.get("answer_sha256") != cell.get("answer_sha256")
            or cell.get("answer_sha256") != (sha256(cell_answer(cell)) if isinstance(cell_answer(cell), dict) else None)
            or cell["plan_sha256"] != (sha256(cell["management_plan"]) if isinstance(cell["management_plan"], str) else None)):
        raise ValueError("Saved cell/result version, identity or fact hash differ")
    ids = [i["id"] for i in cell["rubric"]["items"]]
    if len(ids) != len(set(ids)) or any(i["id"].startswith("precision.") != (i["type"] == "precision")
                                     for i in cell["rubric"]["items"]):
        raise ValueError("Duplicate or mistyped rubric item id")


def prepare_precision(previous, out, *, run_id):
    previous, out = Path(previous).resolve(), Path(out).resolve()
    old = json.loads((previous / "manifest.json").read_text())
    policy = load_policy()
    prior_version = old["rubric_version"]
    if prior_version not in PRIOR_VERSIONS or policy["version"] != NEW_VERSION:
        raise ValueError("Precision reuse requires registered prior and current versions")
    if {k: v for k, v in old["policy"].items() if k != "version"} != {
            k: v for k, v in policy.items() if k != "version"}:
        raise ValueError("Precision reuse cannot change other rubric policy")
    _outside_sources(out, [previous] + [Path(s["batch"]) for s in old["sources"]])
    if not run_id or out.exists() and any(out.iterdir()):
        raise ValueError("Precision prepare requires a run id and new output directory")
    for name in ("cells", "snapshots"):
        if digest(previous / f"{name}.jsonl") != old[f"{name}_sha256"]:
            raise ValueError("Prior cells or snapshots changed")
    cells = _unique(read_jsonl(previous / "cells.jsonl"))
    results = _unique(read_jsonl(previous / "rubric_results.jsonl"))
    snapshots = _unique(read_jsonl(previous / "snapshots.jsonl"))
    if set(cells) != set(results):
        raise ValueError("Prior result cell ids differ")
    reused, selected, seal = [], [], {str(previous / name): digest(previous / name) for name in (
        "manifest.json", "cells.jsonl", "snapshots.jsonl", "rubric_results.jsonl")}
    execution_path = previous / "execution.json"
    prior_ledger = None
    if execution_path.exists():
        seal[str(execution_path)] = digest(execution_path)
        prior_ledger = json.loads(execution_path.read_text()).get("ledger")
    evaluations = {}
    for source in old["sources"]:
        path = Path(source["batch"]) / "eval.jsonl"
        if digest(path) != source["files"]["eval.jsonl"]:
            raise ValueError("Source evaluation changed since preparation")
        evaluations[source["batch"]] = {(r["case"], r["solver"]): r for r in load_rows(path)}
    for key, cell in cells.items():
        record = results[key]
        _identity(cell, record, prior_version)
        snap = snapshots[cell["snapshot_key"]]["snapshot"]
        if verify_snapshot(snap, policy) != cell["snapshot_key"]:
            raise ValueError("Prior snapshot fact hash differs")
        old_items = [i for i in cell["rubric"]["items"] if i["type"] != "precision" and not i.get("core_decision")]
        rubric = derive(cell["rubric"]["pack"], snap, policy)
        # Source markers changed with code-owned core decisions; quality text,
        # facts, weights and diagnostics must still match exactly.
        def quality_identity(items):
            return canonical([{k: v for k, v in item.items() if k != "judgment_source"} for item in items])
        if quality_identity(old_items) != quality_identity([i for i in rubric["items"] if i["type"] != "precision" and not i.get("core_decision")]):
            raise ValueError("Non-precision item ids or bytes changed")
        verdicts, invalid, raw_files = _raw_verdicts(previous, cell, record, prior_version)
        seal.update(raw_files)
        source = cell["source"]
        if "control" in source:
            decision = control_core_decision(rubric, source["control"])
        else:
            row = evaluations[source["batch"]].get((source["case"], source["solver"]))
            if row is None:
                raise ValueError("Prepared answer has no matching saved evaluation")
            decision = eval_core_decision(rubric["pack"], row)
        if "core_decision" in cell and cell["core_decision"] != decision:
            raise ValueError("Prepared code decision differs from its source")
        cell = {**cell, "core_decision": decision}
        keep = {i["id"]: verdicts[i["id"]] for i in old_items if i["id"] in verdicts}
        complete = not invalid and len(keep) == len(old_items)
        keep = with_core_verdicts(rubric, keep, decision)
        precision = [i for i in rubric["items"] if i["type"] == "precision"]
        reused.append({"cell": {**cell, "rubric": rubric}, "verdicts": keep,
                       "reuse_complete": complete, "prior_errors": record.get("errors", []),
                       "prior_rubric_version": prior_version,
                       "prior_raw_judge_files": record.get("raw_judge_files", [])})
        # A cell lacking old non-precision votes stays unscored; no new spending helps it.
        if precision and complete and cell["status"] == "ready":
            selected.append({**cell, "rubric": {**rubric, "items": precision}})
    out.mkdir(parents=True)
    write_jsonl(out / "cells.jsonl", selected)
    write_jsonl(out / "snapshots.jsonl", list(snapshots.values()))
    write_jsonl(out / "reused.jsonl", reused)
    manifest = {"run_id": run_id, "rubric_version": NEW_VERSION, "policy": policy,
                "code": code_state(), "mode": "per_item", "sources": old["sources"],
                "cells_sha256": digest(out / "cells.jsonl"),
                "snapshots_sha256": digest(out / "snapshots.jsonl"),
                "states": dict(Counter(c["status"] for c in selected)),
                "precision_reuse": {"previous": str(previous), "previous_version": prior_version,
                                   "prior_ledger": prior_ledger,
                                   "sealed_files": seal, "reused_sha256": digest(out / "reused.jsonl"),
                                   "cells": len(reused), "request_cells": len(selected),
                                   "legacy_unscored": sum(not r["reuse_complete"] for r in reused)}}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def run_precision(out, *, offline=True, ledger_path=None, limit_usd=None, cfg=None, concurrency=40,
                  destination=None):
    from decimal import Decimal
    from haenv.scoring.management.pipeline import run
    manifest = validate_precision(out)
    if not offline:
        if ledger_path is None or limit_usd is None:
            raise ValueError("An independent precision ledger and cap are required")
        if not Decimal(limit_usd).is_finite() or not 0 < Decimal(limit_usd) <= Decimal("5"):
            raise ValueError("Precision budget must be positive and at most $5")
        ledger_path = Path(ledger_path).resolve()
        reuse = manifest["precision_reuse"]
        _outside_sources(ledger_path, [Path(reuse["previous"])] +
                         [Path(s["batch"]) for s in manifest["sources"]])
        if reuse.get("prior_ledger") and ledger_path == Path(reuse["prior_ledger"]).resolve():
            raise ValueError("Precision must not reuse the previous run's budget ledger")
    result = run(out, offline=offline, cfg=cfg, ledger_path=ledger_path, limit_usd=limit_usd,
                 concurrency=concurrency)
    result["merged"] = merge_precision(out, destination)
    return result


def validate_precision(out):
    out = Path(out).resolve()
    manifest = validate(out)
    reuse = manifest["precision_reuse"]
    if (manifest["rubric_version"] != NEW_VERSION or manifest["mode"] != "per_item"
            or reuse["previous_version"] not in PRIOR_VERSIONS
            or digest(out / "reused.jsonl") != reuse["reused_sha256"]):
        raise ValueError("Precision reuse version or sealed data changed")
    for name, expected in reuse["sealed_files"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Sealed prior judgment changed")
    return manifest


def merge_precision(out, destination=None):
    out = Path(out).resolve()
    manifest = validate_precision(out)
    execution_path = out / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
    destination = Path(destination).resolve() if destination else out / "merged"
    _outside_sources(destination, [Path(manifest["precision_reuse"]["previous"])] +
                     [Path(s["batch"]) for s in manifest["sources"]])
    if destination == out or destination in out.parents:
        raise ValueError("Merge must not overwrite prepared precision results")
    request_cells = _unique(read_jsonl(out / "cells.jsonl"))
    new_results = _unique(read_jsonl(out / "rubric_results.jsonl")) if (out / "rubric_results.jsonl").exists() else {}
    if new_results and set(new_results) != set(request_cells):
        raise ValueError("Precision result cell ids differ")
    merged = []
    for reused in read_jsonl(out / "reused.jsonl"):
        cell = reused["cell"]
        full = cell["rubric"]
        verdicts = deepcopy(reused["verdicts"])
        expected = {i["id"] for i in full["items"] if i["type"] == "precision"}
        errors, files, state = list(reused["prior_errors"]), [], "collected"
        if not reused["reuse_complete"]:
            state = "legacy_unscored"
        elif cell["key"] in request_cells:
            result = new_results.get(cell["key"])
            if result is not None:
                _identity(request_cells[cell["key"]], result, NEW_VERSION)
            if result is None or result["collection_status"] == "pending":
                state = "pending"
            elif result["collection_status"] == "collected":
                new, invalid, _ = _raw_verdicts(out, request_cells[cell["key"]], result, NEW_VERSION)
                if not invalid and set(new) != expected:
                    raise ValueError("Precision verdict item ids differ")
                if invalid:
                    state = "unscored"
                else:
                    verdicts.update(new)
                errors.extend(result.get("errors", []))
                files = result.get("raw_judge_files", [])
            else:
                state = result["collection_status"]
        elif expected:
            state = "unscored"
        merged.append({"key": cell["key"], "source": cell["source"], "collection_status": state,
                       "rubric_version": NEW_VERSION, "facts_sha256": cell["snapshot_key"],
                       "answer_sha256": cell.get("answer_sha256"), "plan_sha256": cell["plan_sha256"],
                       "code": manifest["code"], "verdicts": verdicts,
                       "core_decision": cell["core_decision"],
                       "precision_execution": execution,
                       "summary": score(full, verdicts if state == "collected" else None, manifest["policy"],
                                        core_decision=cell["core_decision"]),
                       "errors": errors, "raw_judge_files": files,
                       "reuse": {"rubric_version": reused["prior_rubric_version"], "non_precision_item_ids": list(reused["verdicts"]),
                                 "prior_raw_judge_files": reused["prior_raw_judge_files"],
                                 "previous": manifest["precision_reuse"]["previous"]}})
    summary = {"rubric_version": NEW_VERSION, "cells": len(merged), "non_precision_paid_calls": 0,
               "precision_execution": execution,
               "states": dict(Counter(r["summary"]["status"] if r["collection_status"] == "collected"
                                      else r["collection_status"] for r in merged)),
               "out": str(destination)}
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "rubric_results.jsonl", merged)
    atomic_json(destination / "summary.json", summary)
    return summary
