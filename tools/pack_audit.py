"""pack_audit.py -- the shared generation-time audit (SA-3 ... SA-8 + the pack's own gates) on one
emitted batch of a plugin pack. Zero model calls.

Reads the batch's `cases.jsonl` and `payloads.jsonl` (what the solver sees). Writes `<out>/audit.json`;
when every gate passes, also `<out>/pack.json` (the published items) and the audit marker into the
batch directory (`haenv.shared_audit.marker_name`), without which `haenv run` refuses the pack's cells.
Exit 0 = passed, 1 = a gate is red (not published).

  OMP_NUM_THREADS=1 PYTHONPATH=<repo> python tools/pack_audit.py \\
      --pack p4 --batch <results/joint_dx/p4-followup/<stamp>> --out <dir> [--job inputs/<job>.yaml] [--ref <batch>]

The gates read the observed labels only (chance-corrected score against the 0.20 line and the shuffled
null's 0.99 quantile). A permutation p is a profile reading: it is not computed by default and stays null
in `audit.json` ("not computed"); `--profile` computes it.

The published subset follows the job's `pack` header (N and the seed's hash); a job that records only
the hash needs its seed (`--seed`, or the environment variable HAENV_PACK_SEED).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from haenv import pack_size as PS       # noqa: E402
from haenv import shared_audit as SA     # noqa: E402

PACKS = {"p4": "haenv.p4.audit", "pack2": "haenv.pack2.audit", "m2": "haenv.m2.audit", "p3": "haenv.p3.audit"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True, choices=sorted(PACKS))
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--job", default=None, help="job yaml the batch was built from (packs ② and ①)")
    ap.add_argument("--ref", default=None, help="v1.0.1 deterministic workup batch (pack ①'s BRIDGE)")
    ap.add_argument("--profile", action="store_true",
                    help="also compute the permutation p readings of SA-5 and SA-6 (a profile, not a gate)")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--seed", default=None, help="the pack's seed when the job records only its hash")
    ap.add_argument("--no-marker", action="store_true", help="do not write the audit marker into the batch")
    a = ap.parse_args(argv)
    logging.disable(logging.CRITICAL)
    batch, out = pathlib.Path(a.batch), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    head = PS.read_header(a.job) if a.job else {}
    n_items = int(head.get("n_items") or PS.DEFAULT_N)
    seed = PS.resolve_seed(a.job, a.seed if a.seed is not None else os.environ.get("HAENV_PACK_SEED")) if head else PS.PUBLIC_SEED
    A = importlib.import_module(PACKS[a.pack]).audit(n_items=n_items, seed=seed)
    rep = SA.run_audit(A, batch, a.job and pathlib.Path(a.job), a.ref and pathlib.Path(a.ref), profile=a.profile, jobs=a.jobs)
    (out / "audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if rep["pass"]:
        sha = hashlib.sha256((batch / "cases.jsonl").read_bytes()).hexdigest()
        (out / "pack.json").write_text(json.dumps({"pack": a.pack, "batch": str(batch), "cases_sha256": sha,
                                                   "n_items": n_items, "seed_sha256": PS.seed_sha256(seed),
                                                   "items": rep["pack_ids"]}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        if not a.no_marker:
            (batch / SA.marker_name(a.pack)).write_text(sha + "\n", encoding="utf-8")
    for k, v in rep["gates"].items():
        brief = {x: v[x] for x in ("cc", "max_cc", "mean_ba", "perm_q95_mean_ba", "p", "worst", "counts", "mismatch",
                                   "factor_only_cc", "prior_cc", "random_mean") if x in v}
        print(("PASS " if v.get("pass") else "FAIL ") + k + " " + json.dumps(brief, ensure_ascii=False, default=str)[:400])
        for t, r in ({"": v} if "by_target" not in v else v["by_target"]).items():
            for line in r.get("report") or ():
                print(f"REPORT {k}{' ' + t if t else ''} {line}")
    print(f"n_emitted {rep['n_emitted']} n_pack {rep['n_pack']} pass={rep['pass']}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
