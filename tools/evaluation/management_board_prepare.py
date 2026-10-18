"""Prepare saved pk1 board answers on an explicit N50 case subset; no dispatch."""
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
import json
from pathlib import Path
import sys

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))


def selected_batches(path):
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Batch list must be a nonempty JSON list")
    batches = []
    for entry in entries:
        if isinstance(entry, str):
            batch = Path(entry).resolve()
            if not batch.name.endswith("-pk1"):
                raise ValueError("A path without -pk1 needs explicit {batch, pk: 1} metadata")
        elif isinstance(entry, dict) and entry.get("pk") == 1 and type(entry["pk"]) is int:
            batch = Path(entry["batch"]).resolve()
            if any(batch.name.endswith(f"-pk{k}") for k in (2, 3)):
                raise ValueError("Explicit pk1 metadata conflicts with the batch directory")
        else:
            raise ValueError("Only saved pk1 batches are admitted")
        batches.append(batch)
    if len(set(batches)) != len(batches):
        raise ValueError("Duplicate batch directory")
    return batches


def prepare_board(batch_list, subsets, out, *, run_id, policy_path=None):
    from haenv.provenance.judging_provenance import management_prepare
    from haenv.scoring.management.pipeline import read_jsonl
    from haenv.scoring.management.pipeline import write_jsonl
    from haenv.scoring.management.rubric import sha256
    from haenv.scoring.semantic.parallel import atomic_json
    from haenv.scoring.semantic.seal import digest
    case_sets, receipts = {}, {}
    for pack, path in subsets.items():
        if pack not in {"p2", "p3", "p4", "m2"}:
            raise ValueError("Unknown subset pack")
        rows = read_jsonl(path)
        ids = [r["case_id"] for r in rows]
        if len(ids) != 50 or len(set(ids)) != 50:
            raise ValueError("Each N50 subset must contain exactly 50 distinct saved case ids")
        case_sets[pack] = set(ids)
        receipts[pack] = {"path": str(Path(path).resolve()), "sha256": digest(Path(path)),
                          "case_ids": sorted(ids)}
    batches = selected_batches(batch_list)
    manifest = management_prepare(batches, out, run_id=run_id, mode="per_case", policy_path=policy_path)
    out = Path(out)
    cells = read_jsonl(out / "cells.jsonl")
    packs = {c["rubric"]["pack"] for c in cells}
    if not packs <= case_sets.keys():
        raise ValueError(f"Missing explicit N50 subset for packs {sorted(packs - case_sets.keys())}")
    selected = [c for c in cells if c["source"]["case"] in case_sets[c["rubric"]["pack"]]]
    if not selected:
        raise ValueError("N50 case ids do not occur in the saved batches")
    # A case/model must occur once per arm. Prepare each arm separately.
    identities = [(c["rubric"]["pack"], c["source"]["case"], c["source"]["solver"]) for c in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate case/model within pk1 arm; prepare arms separately")
    selected.sort(key=lambda c: (c["snapshot_key"], c["source"]["solver"], c["key"]))
    write_jsonl(out / "cells.jsonl", selected)
    used = {c["snapshot_key"] for c in selected}
    snapshots = [r for r in read_jsonl(out / "snapshots.jsonl") if r["key"] in used]
    write_jsonl(out / "snapshots.jsonl", snapshots)
    manifest["cells_sha256"] = digest(out / "cells.jsonl")
    manifest["snapshots_sha256"] = digest(out / "snapshots.jsonl")
    manifest["states"] = dict(Counter(c["status"] for c in selected))
    manifest["sampling"] = {"selected_pk": 1, "case_subsets": receipts,
                            "batch_list": str(Path(batch_list).resolve()), "batch_list_sha256": digest(Path(batch_list)),
                            "selection": "explicit saved N50 case ids, before judgment",
                            "cells_before_subset": len(cells), "cells": len(selected),
                            "include_controls": False, "selection_sha256": sha256(identities)}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-list", type=Path, required=True,
                        help="JSON list of -pk1 paths or explicit {batch, pk: 1} objects; one arm")
    parser.add_argument("--n50-cases", action="append", required=True, metavar="PACK=CASES_JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default="management-board-n50-pk1-v5")
    parser.add_argument("--policy", type=Path, help="Management registry policy including judge conditions")
    args = parser.parse_args(argv)
    subsets = {}
    for entry in args.n50_cases:
        pack, path = entry.split("=", 1)
        if pack in subsets:
            parser.error("Each pack subset must be specified once")
        subsets[pack] = Path(path)
    result = prepare_board(args.batch_list, subsets, args.out, run_id=args.run_id, policy_path=args.policy)
    print(json.dumps({"rubric_version": result["rubric_version"], "states": result["states"],
                      "cells": result["sampling"]["cells"], "judging_criteria": result["code"]["judging_criteria"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
