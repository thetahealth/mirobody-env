"""Separate management-rubric overlay; offline by default, paid only via explicit run."""
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
import json
from pathlib import Path
import sys

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("offline", "prepare", "run", "estimate", "rescore"))
    parser.add_argument("--batch", action="append", type=Path, default=[])
    parser.add_argument("--keys-from", type=Path, help="prepare: saved directory repeats completed results; PATH/cells.jsonl selects every prepared cell, optionally changing only the judge via --policy")
    parser.add_argument("--key-file", type=Path, help="prepare with cells.jsonl: fixed JSON array of selected cell keys")
    parser.add_argument("--inputs-from", action="append", type=Path, default=[], help="prepare: completed saved v5 run; rejudge full answers with v6, without reusing old votes")
    parser.add_argument("--controls-per-pack", type=int, default=0, help="add fixed pre-judgment controls for this many saved questions per pack")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default="management-rubric-offline")
    parser.add_argument("--mode", choices=("per_case", "per_item"), default="per_case")
    parser.add_argument("--controls", action="store_true", help="Include fact-built positive and negative controls; deduplicated by case/fact hash")
    parser.add_argument("--limit-per-model", type=int, help="Select fixed case hashes per model per source batch, before judging")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--budget-usd")
    parser.add_argument("--price-metadata", type=Path, help="estimate: saved exact Luna price metadata, never fetched by estimate")
    parser.add_argument("--destination", type=Path, help="estimate/rescore: new output directory")
    parser.add_argument("--policy", type=Path, help="prepare: judge policy; rescore: aggregation-only policy")
    args = parser.parse_args(argv)
    from haenv.provenance.judging_provenance import management_prepare as prepare
    from haenv.provenance.judging_provenance import management_run as run
    from haenv.provenance.judging_provenance import management_rescore as rescore
    if args.operation in {"offline", "prepare"}:
        if not args.batch and args.keys_from is None and not args.inputs_from:
            parser.error("Saved --batch directories, --keys-from or --inputs-from are required")
        if args.keys_from is not None and (args.batch or args.controls or args.limit_per_model):
            parser.error("--keys-from repeats saved inputs; do not combine with batch sampling or controls")
        if args.inputs_from:
            if args.batch or args.keys_from or args.controls or args.limit_per_model or args.mode != "per_case" or args.policy:
                parser.error("--inputs-from uses saved answers and full v6 per_case requests")
            from haenv.provenance.judging_provenance import management_saved_prepare
            result = management_saved_prepare(args.inputs_from, args.out, run_id=args.run_id,
                                               controls_per_pack=args.controls_per_pack)
        else:
            result = prepare(args.batch, args.out, run_id=args.run_id, mode=args.mode,
                             limit_per_model=args.limit_per_model, include_controls=args.controls,
                             keys_from=args.keys_from, policy_path=args.policy,
                             controls_per_pack=args.controls_per_pack, key_file=args.key_file)
        if args.operation == "offline":
            result = run(args.out, offline=True, concurrency=args.concurrency)
        print(json.dumps({k: result[k] for k in ("rubric_version", "states")}, ensure_ascii=False))
    elif args.operation == "run":
        if args.budget_ledger is None or args.budget_usd is None:
            parser.error("Paid run requires explicit --budget-ledger and --budget-usd")
        from haenv.app.config import load_cfg
        result = run(args.out, offline=False, ledger_path=args.budget_ledger, limit_usd=args.budget_usd,
                     cfg=load_cfg(), concurrency=args.concurrency, max_cells=args.max_cells)
        print(json.dumps(result, ensure_ascii=False))
        if result["stop_reason"] or result["states"].get("pending") or result["states"].get("unscored"):
            return 2
    elif args.operation == "estimate":
        if args.price_metadata is None or args.destination is None:
            parser.error("Estimate requires saved --price-metadata and separate --destination")
        from haenv.scoring.management.estimate import estimate
        result = estimate(args.out, args.destination, args.price_metadata)
        print(json.dumps({"native_tariff_status": result["native_tariff_status"], "totals": result["totals"]}, ensure_ascii=False))
    else:
        if args.destination is None:
            parser.error("Rescore requires a separate --destination")
        print(json.dumps(rescore(args.out, args.destination, policy_path=args.policy), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
