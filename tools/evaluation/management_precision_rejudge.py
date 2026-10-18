"""Prepare/run current guideline precision, then merge sealed saved quality verdicts."""

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
    parser.add_argument("operation", choices=("prepare", "run", "offline", "estimate", "merge"))
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default="precision-v5")
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--budget-usd")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--price-metadata", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    from haenv.scoring.management.precision_rejudge import validate_precision
    from haenv.provenance.judging_provenance import management_precision_prepare as prepare_precision
    from haenv.provenance.judging_provenance import management_precision_merge as merge_precision
    from haenv.provenance.judging_provenance import management_precision_run as run_precision
    if args.operation == "prepare":
        if args.previous is None:
            parser.error("prepare requires --previous with stored v3 judgments")
        manifest = prepare_precision(args.previous, args.out, run_id=args.run_id)
        reuse = manifest["precision_reuse"]
        result = {k: reuse[k] for k in ("previous", "previous_version", "cells", "request_cells", "legacy_unscored")}
    else:
        validate_precision(args.out)  # Before any paid dispatch or merge.
        if args.operation in ("run", "offline"):
            cfg = None
            if args.operation == "run":
                if args.budget_ledger is None or args.budget_usd is None:
                    parser.error("Paid run requires an independent --budget-ledger and --budget-usd")
                from decimal import Decimal
                if not 0 < Decimal(args.budget_usd) <= Decimal("5"):
                    parser.error("Precision run budget must be positive and at most $5")
                from haenv.app.config import load_cfg
                cfg = load_cfg()
            result = run_precision(args.out, offline=args.operation == "offline", cfg=cfg,
                         ledger_path=args.budget_ledger, limit_usd=args.budget_usd,
                         concurrency=args.concurrency, destination=args.destination)
        elif args.operation == "estimate":
            if args.price_metadata is None or args.destination is None:
                parser.error("estimate requires saved --price-metadata and separate --destination")
            from haenv.scoring.management.estimate import estimate
            result = estimate(args.out, args.destination, args.price_metadata)["totals"]
        else:
            result = merge_precision(args.out, args.destination)
    print(json.dumps(result, ensure_ascii=False))
    if result.get("stop_reason") or result.get("states", {}).get("pending") or result.get("states", {}).get("unscored"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
