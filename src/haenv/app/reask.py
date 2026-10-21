"""Command-line entry for controlled re-asking of stored diagnosis responses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from haenv.app.arguments import model_cap, positive_int
from haenv.interaction.reask import ARMS


def parser():
    ap = argparse.ArgumentParser(prog="haenv reask", description=(
        "Reproduce the experiment's mechanism per unit: re-ask failed stored diagnosis responses "
        "before their final answer, with neutral tool-result injection. The pooled 1.3.1 results "
        "are recomputed from the shipped package (stored answers, inclusion list and diagnosis "
        "gold) with tools/evaluation/reask_pooled.py --check; a new run selects its own units."))
    ap.add_argument("source", type=Path, help="stored run/batch directory or responses.jsonl")
    ap.add_argument("--sources", type=Path, action="append", default=[], metavar="DIR",
                    help="additional recorded result corpus (repeatable; needed to reproduce the same candidate pool)")
    ap.add_argument("--questions", "--cases", help="comma-separated original question ids")
    ap.add_argument("--repeats", help="comma-separated repeat numbers (stored repeat, batch -pkN/-repN, otherwise 1)")
    ap.add_argument("--models", help="comma-separated original model ids; uses the same model again")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated resample,decisive,placebo,half (half only when at least two results are missed)")
    ap.add_argument("--seed", type=int, default=20261019, help="placebo seed (default: %(default)s)")
    ap.add_argument("--half-seed", type=int, default=20261020, help="half seed (default: %(default)s)")
    ap.add_argument("--limit", type=positive_int, help="maximum selected units, before expanding arms")
    ap.add_argument("--dry-run", action="store_true", help="print injected test names and carried evidence keys; no model calls")
    ap.add_argument("--out", type=Path, help="separate output directory; required for model calls")
    ap.add_argument("--workers", type=positive_int, default=64, help="cells in flight per model (default: %(default)s)")
    ap.add_argument("--model-cap", type=model_cap, action="append", default=[], metavar="MODEL=N",
                    help="explicit model cap, bounded by backend and paid lanes (repeatable; last wins)")
    ap.add_argument("--tool-choice", choices=("none", "auto"), help="default none, auto for GLM Flash; returned calls count as unanswered")
    ap.add_argument("--allow-settings-override", action="store_true",
                    help="allow missing or mismatched source answer settings; output records actual settings and loses verified replay status")
    ap.add_argument("--judge-budget-ledger", type=Path, help="shared persistent API cost ledger")
    ap.add_argument("--judge-budget-usd", help="approved ledger cap for model calls")
    return ap


def main(argv=None):
    from haenv.reask import dry_rows, prepare, run
    ap = parser()
    args = ap.parse_args(argv)
    def csv(value):
        return [x.strip() for x in value.split(",") if x.strip()] if value is not None else None
    try:
        arms = csv(args.arms)
        if not arms or len(arms) != len(set(arms)) or set(arms) - set(ARMS):
            raise ValueError("--arms must contain unique names from resample,decisive,placebo,half")
        repeats = [int(x) for x in csv(args.repeats)] if args.repeats is not None else None
        if not args.dry_run and (args.out is None or args.judge_budget_ledger is None or args.judge_budget_usd is None):
            raise ValueError("model calls require --out, --judge-budget-ledger and --judge-budget-usd")
        units = prepare(args.source, sources=args.sources, questions=csv(args.questions),
                        repeats=repeats, models=csv(args.models), arms=arms,
                        seed=args.seed, half_seed=args.half_seed, limit=args.limit)
        if not units:
            raise ValueError("no eligible failed diagnosis units selected")
        if args.dry_run:
            for row in dry_rows(units):
                print(json.dumps(row, ensure_ascii=False))
        else:
            rows = run(units, args.out, workers=args.workers, model_caps=dict(args.model_cap),
                       ledger_path=args.judge_budget_ledger, budget_usd=args.judge_budget_usd,
                       tool_choice=args.tool_choice, allow_settings_override=args.allow_settings_override)
            print(json.dumps({"rows_written": len(rows), "output": str(args.out)}, ensure_ascii=False))
        return 0
    except (ValueError, OSError) as error:
        ap.exit(2, f"haenv reask: {error}\n")
