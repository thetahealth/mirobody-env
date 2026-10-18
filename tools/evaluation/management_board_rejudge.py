"""Discover completed N50 pk1 blocks, resume sealed runs, and export board Q."""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))

from haenv.provenance.judging import current_fingerprints
from haenv.provenance.judging_provenance import management_prepare
from haenv.provenance.judging_provenance import management_run
from haenv.scoring.management.judge import parse_reply
from haenv.scoring.management.pipeline import read_jsonl
from haenv.scoring.management.pipeline import write_jsonl
from haenv.scoring.management.pipeline import validate
from haenv.scoring.management.pipeline import _tasks
from haenv.scoring.management.pipeline import _attempts
from haenv.scoring.management.pipeline import _read_sample
from haenv.scoring.management.pipeline import cell_answer
from haenv.scoring.management.rubric import score
from haenv.scoring.management.rubric import sha256
from haenv.scoring.management.rubric import with_core_verdicts
from haenv.scoring.semantic.parallel import atomic_json
from haenv.scoring.semantic.seal import digest

PACKS = {"p2", "p3", "p4", "m2"}


def identity(cell):
    return cell["rubric"]["pack"], cell["source"]["case"], cell["source"]["solver"], 1


def discover(work, packs, *, pk=1, pools=None):
    if pk != 1 or not packs or not set(packs) <= PACKS:
        raise ValueError("Only pk1 and explicit supported packs are accepted")
    work = Path(work).resolve()
    if not work.is_dir():
        raise ValueError("Board work directory is missing")
    selected = sorted(pools or [p.name for p in work.iterdir() if p.is_dir()
                               and any((p/pack).is_dir() for pack in PACKS)])
    ready, skipped = [], []
    for pool in selected:
        if Path(pool).name != pool or pool in {".", ".."}:
            raise ValueError("Invalid pool name")
        for pack in packs:
            block = {"pool": pool, "pack": pack, "pk": 1}
            batches = sorted((work/pool/pack/"results/joint_dx").glob("*/*-pk1"))
            if len(batches) != 1:
                skipped.append({**block, "reason": "missing_pk1" if not batches else "ambiguous_pk1",
                                "batches": list(map(str, batches))})
                continue
            batch = batches[0]
            block["batch"] = str(batch)
            if not all((batch/name).is_file() for name in ("batch.json", "cases.jsonl", "eval.jsonl")):
                skipped.append({**block, "reason": "missing_saved_inputs"})
                continue
            try:
                meta = json.loads((batch/"batch.json").read_text())
                cases = read_jsonl(batch/"cases.jsonl")
                rows = read_jsonl(batch/"eval.jsonl")
            except (ValueError, KeyError):
                skipped.append({**block, "reason": "incomplete_json_write"})
                continue
            # models/sampling scan deployment models; last_cmd_models is the
            # actual invocation's pool. Never infer missing models from eval.
            models = meta.get("last_cmd_models")
            if not isinstance(models, list) or not models or len(set(models)) != len(models):
                skipped.append({**block, "reason": "missing_invocation_models"})
                continue
            ids = [c["case_id"] for c in cases]
            expected = {(cid, model) for cid in ids for model in models}
            seen = [(r["case"], r["solver"]) for r in rows]
            block.update(models=models, expected=50*len(models), eval_rows=len(rows),
                         distinct=len(set(seen)), cases=len(ids))
            if len(ids) != 50 or len(set(ids)) != 50:
                skipped.append({**block, "reason": "not_exact_n50"})
            elif len(seen) != len(set(seen)) or set(seen) != expected:
                skipped.append({**block, "reason": "incomplete_or_duplicate_eval",
                                "missing": len(expected-set(seen)), "unexpected": len(set(seen)-expected)})
            else:
                ready.append(block)
    return ready, skipped


def read_archive(directory):
    directory = Path(directory).resolve()
    manifest = validate(directory)  # Pure reads; also verifies saved source bytes.
    if manifest["rubric_version"] not in {"management-rubric-v5", "management-rubric-v6", "management-rubric-v6.1", "management-rubric-v6.2"}:
        raise ValueError("Board Q requires a sealed v5, v6, v6.1 or v6.2 run")
    ruler = current_fingerprints()
    stamp = manifest["code"].get("judging_criteria", {}).get("management_rubric")
    if stamp != ruler["criteria"]["management_rubric"]:
        raise ValueError("Imported management criterion is missing or stale")
    cells = read_jsonl(directory/"cells.jsonl")
    if any(not Path(c["source"]["batch"]).name.endswith("-pk1") or "control" in c["source"] for c in cells):
        raise ValueError("Only real pk1 answers can enter board Q")
    return directory, manifest, cells


def archive_paths(directory, ancestors=()):
    """Accept a flat prepare/run or this entry's multi-shard output, read only."""
    directory = Path(directory).resolve()
    if directory in ancestors:
        raise ValueError("Cyclic imported board directories")
    if (directory/"manifest.json").is_file():
        return [directory]
    journal_path = directory/"board.json"
    if not journal_path.is_file():
        raise ValueError("Imported directory is neither a sealed run nor a board output")
    journal = json.loads(journal_path.read_text())
    if journal.get("pk") != 1:
        raise ValueError("Imported board must use pk1")
    found = {p.parent.resolve() for p in (directory/"runs").glob("*/manifest.json")}
    for source in journal.get("imports", []):
        found.update(archive_paths(source, (*ancestors, directory)))
    return sorted(found)


def export_rows(directory):
    directory, manifest, cells = read_archive(directory)
    snapshots = {s["key"]: s["snapshot"] for s in read_jsonl(directory/"snapshots.jsonl")}
    aggregate = directory/"rubric_results.jsonl"
    stored = {r["key"]: r for r in read_jsonl(aggregate)} if aggregate.is_file() else {}
    execution = directory/"execution.json"
    offline = json.loads(execution.read_text())["offline"] if execution.is_file() else None
    rows = []
    for cell in cells:
        pack, case, solver, pk = identity(cell)
        decision = cell["core_decision"]
        row = {"pack": pack, "case": case, "solver": solver, "pk": pk,
               "status": "not_collected", "rubric_version": manifest["rubric_version"],
               "management_rubric": manifest["code"]["judging_criteria"]["management_rubric"],
               "positive_fraction": None, "safety_multiplier": None,
               "core_ok": decision["correct"], "q_capped": None,
               "answer_sha256": cell.get("answer_sha256"), "facts_sha256": cell["snapshot_key"],
               "source_batch": cell["source"]["batch"], "source_runs": [str(directory)],
               "offline": offline, "reason": cell["status"] if cell["status"] != "ready" else "pending"}
        result_file = directory/"results"/f"{cell['key']}.json"
        result = json.loads(result_file.read_text()) if result_file.is_file() else stored.get(cell["key"])
        if result is None or result.get("collection_status") != "collected":
            rows.append(row)
            continue
        if result["source"] != cell["source"] or result["core_decision"] != decision:
            raise ValueError("Saved result identity or code decision differs")
        votes, errors = {}, []
        for task, items in _tasks(cell, snapshots[cell["snapshot_key"]], manifest):
            for request, filename in _attempts(task, manifest["policy"]):
                sample = _read_sample(directory/"raw_judge"/filename, request)
                if not sample.get("retryable"):
                    break
            try:
                votes.update(parse_reply(sample["raw"], items, cell_answer(cell)))
            except (ValueError, TypeError, KeyError):
                errors.append(True)
        votes = with_core_verdicts(cell["rubric"], votes, decision)
        summary = score(cell["rubric"], None if errors else votes, manifest["policy"], core_decision=decision)
        if votes != result["verdicts"] or summary != result["summary"]:
            raise ValueError("Saved verdicts or scores differ from original raw responses")
        q = summary["quality_score"]
        row.update(status=summary["status"], reason=None,
                   q_capped=summary["rubric_score"], safety_multiplier=summary["safety_multiplier"])
        if summary["status"] == "scored" and q is not None:
            # Ask the same covered scorer for the positive quality view. No
            # second weighting formula and no synthetic verdicts are introduced.
            positives = [i for i in cell["rubric"]["items"]
                         if i["polarity"] == "positive" and not i.get("core_decision")]
            view = {**cell["rubric"], "items": positives, "code_core_decision": False}
            fraction = score(view, {i["id"]: votes[i["id"]] for i in positives}, manifest["policy"])["rubric_score"]
            if not 0 <= q <= 1 or abs(q-fraction*summary["safety_multiplier"]) > 1e-12:
                raise ValueError("Q is not the uncapped quality fraction times safety")
            row.update(q=q, positive_fraction=fraction,
                       measurement_sha256=sha256([cell["source"], cell["rubric"], votes, decision]))
        else:
            row["status"] = "unscored"
        rows.append(row)
    return rows


def merge_rows(groups):
    merged = {}
    for rows in groups:
        for row in rows:
            key = tuple(row[k] for k in ("pack", "case", "solver", "pk"))
            if key not in merged:
                merged[key] = deepcopy(row)
                continue
            old = merged[key]
            for field in ("facts_sha256", "answer_sha256", "management_rubric", "source_batch"):
                if old[field] != row[field]:
                    raise ValueError(f"Conflicting duplicate Q identity: {key}")
            if old.get("q") is not None and row.get("q") is not None and old["measurement_sha256"] != row["measurement_sha256"]:
                raise ValueError(f"Different judgments for duplicate Q identity: {key}")
            runs = sorted(set(old["source_runs"] + row["source_runs"]))
            if old.get("q") is None and row.get("q") is not None:
                merged[key] = deepcopy(row)
            merged[key]["source_runs"] = runs
    return [merged[k] for k in sorted(merged)]


def _prepare_new(blocks, covered, out):
    batches = [b["batch"] for b in blocks]
    run_id = sha256([batches, sorted(covered)])[:20]
    target = out/"runs"/run_id
    manifest = management_prepare([Path(p) for p in batches], target, run_id=f"board-v5:{run_id}", mode="per_case")
    cells = [c for c in read_jsonl(target/"cells.jsonl") if identity(c) not in covered]
    identities = [identity(c) for c in cells]
    if not cells or len(set(identities)) != len(cells):
        raise ValueError("New shard is empty or contains duplicate board identities")
    cells.sort(key=lambda c: (c["snapshot_key"], c["source"]["solver"]))
    write_jsonl(target/"cells.jsonl", cells)
    used = {c["snapshot_key"] for c in cells}
    write_jsonl(target/"snapshots.jsonl", [s for s in read_jsonl(target/"snapshots.jsonl") if s["key"] in used])
    manifest.update(cells_sha256=digest(target/"cells.jsonl"), snapshots_sha256=digest(target/"snapshots.jsonl"),
                    states=dict(Counter(c["status"] for c in cells)))
    manifest["sampling"].update(selected_pk=1, n50=True, excluded_existing_cells=len(covered),
                                complete_blocks=blocks)
    atomic_json(target/"manifest.json", manifest)
    validate(target)
    return target


def execute(operation, *, out, work=None, pk=1, packs=("p2", "p3", "p4", "m2"), pools=None,
            imports=(), ledger=None, budget=None, concurrency=40, max_cells=None):
    if (operation not in {"prepare", "offline", "run", "export"} or pk != 1
            or not 1 <= concurrency <= 40 or not packs or not set(packs) <= PACKS
            or max_cells is not None and (type(max_cells) is not int or max_cells <= 0)):
        raise ValueError("Explicit operation, pk1 and concurrency 1..40 required")
    out = Path(out).resolve()
    sources = ([Path(work).resolve()] if work else []) + [Path(p).resolve() for p in imports]
    if any(out == p or p in out.parents or out in p.parents for p in sources):
        raise ValueError("Outputs must be separate from board inputs and imported runs")
    if operation != "export" and work is None:
        raise ValueError("Discovery needs --work")
    if operation == "run" and (ledger is None or budget is None):
        raise ValueError("Paid run needs an independent explicit ledger and budget")
    out.mkdir(parents=True, exist_ok=True)
    with (out/"board.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = out/"board.json"
        journal = json.loads(journal_path.read_text()) if journal_path.is_file() else {
            "work": str(Path(work).resolve()) if work else None, "pk": 1, "imports": []}
        if work and journal["work"] not in {None, str(Path(work).resolve())}:
            raise ValueError("Cannot change board work directory on resume")
        journal["work"] = str(Path(work).resolve()) if work else journal["work"]
        journal["imports"] = sorted(set(journal["imports"] + [str(Path(p).resolve()) for p in imports]))
        imported_paths = sorted({p for source in journal["imports"] for p in archive_paths(source)})
        imported = [read_archive(p) for p in imported_paths]
        owned = [read_archive(p.parent) for p in sorted((out/"runs").glob("*/manifest.json"))]
        if operation == "run":
            from haenv.runtime.budget import store_of
            lp = Path(ledger).resolve()
            ledger_sources = sources + [Path(p) for p in journal["imports"]] + imported_paths
            if any(lp == p or p in lp.parents for p in ledger_sources):
                raise ValueError("Ledger must be outside input directories")
            for directory, _, _ in imported+owned:
                path = directory/"execution.json"
                execution = json.loads(path.read_text()) if path.is_file() else {}
                if execution.get("offline") is True:
                    raise ValueError("Cannot use offline judgments in a paid board")
                if directory in {p for p, _, _ in imported} and execution.get("ledger") and store_of(lp) == store_of(Path(execution["ledger"]).resolve()):
                    raise ValueError("Imported run ledger cannot be reused for new board requests")
            if journal.get("ledger") not in {None, str(lp)}:
                raise ValueError("Cannot change ledger on resume")
            if journal.get("ledger") is None and (lp.exists() or store_of(lp).exists()):
                raise ValueError("First paid invocation requires a fresh independent ledger")
            journal["ledger"] = str(lp)
        covered = {identity(c) for _, _, cs in imported+owned for c in cs}
        ready, skipped = discover(work, packs, pk=pk, pools=pools) if work else ([], [])
        new_blocks = []
        for block in ready:
            expected = {(block["pack"], r["case_id"], m, 1)
                        for r in read_jsonl(Path(block["batch"])/"cases.jsonl") for m in block["models"]}
            if not expected <= covered:
                new_blocks.append(block)
        if new_blocks and operation != "export":
            target = _prepare_new(new_blocks, covered, out)
            owned.append(read_archive(target))
        atomic_json(journal_path, journal)
        results = []
        if operation in {"run", "offline"}:
            cfg = None
            if operation == "run":
                from haenv.app.config import load_cfg
                from haenv.runtime.transport import proxy_for_model
                cfg = load_cfg()
                if proxy_for_model("gpt-6-luna", cfg=cfg) != "env":
                    raise ValueError("Luna requires proxy env")
            remaining = max_cells
            for directory, _, cells in owned:
                if remaining == 0:
                    break
                selected_count = sum(c["status"] == "ready" and not
                                     (directory/"results"/f"{c['key']}.json").exists() for c in cells)
                if remaining is not None:
                    selected_count = min(selected_count, remaining)
                result = management_run(directory, offline=operation == "offline", cfg=cfg,
                                        ledger_path=ledger, limit_usd=budget,
                                        concurrency=concurrency, max_cells=remaining)
                if remaining is not None:
                    remaining -= selected_count
                results.append(result)
                if result.get("stop_reason"):
                    break
        rows = [row for row in merge_rows([export_rows(p) for p, _, _ in imported+owned])
                if row["pack"] in packs]
        write_jsonl(out/"q-status.jsonl", rows)
        write_jsonl(out/"q.jsonl", [r for r in rows if r.get("q") is not None])
        summary = {"operation": operation, "pk": 1, "packs": list(packs), "ready_blocks": ready,
                   "skipped_blocks": skipped, "new_blocks": new_blocks, "imports": journal["imports"],
                   "runs": [str(p) for p, _, _ in owned], "states": dict(Counter(r["status"] for r in rows)),
                   "q_rows": sum(r.get("q") is not None for r in rows), "cells": len(rows),
                   "q": str(out/"q.jsonl"), "all_statuses": str(out/"q-status.jsonl"),
                   "execution_results": results, "paid_calls": 0 if operation != "run" else None}
        atomic_json(out/"board-summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "offline", "run", "export"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--pk", type=int, default=1)
    parser.add_argument("--packs", default="p2,p3,p4,m2")
    parser.add_argument("--pools", help="Optional comma-separated pool selection")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--import-run", type=Path, action="append", default=[])
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--budget-usd")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args(argv)
    result = execute(args.operation, out=args.out, work=args.work, pk=args.pk,
                     packs=tuple(p for p in args.packs.split(",") if p),
                     pools=args.pools.split(",") if args.pools else None, imports=args.import_run,
                     ledger=args.budget_ledger, budget=args.budget_usd,
                     concurrency=args.concurrency, max_cells=args.max_cells)
    print(json.dumps({k: result[k] for k in ("cells", "q_rows", "states", "skipped_blocks", "q")}, ensure_ascii=False))
    return 2 if any(r.get("stop_reason") for r in result["execution_results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
