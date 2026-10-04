"""p3_gen_job.py -- write the pack-3 chronic-medication job for N items (`inputs/p3-meds.job.yaml`).

Inputs (read only): `inputs/ddx-workup.job.yaml` (145 cases, copied verbatim) and the same
variant top-up pack 4 uses (`tools/p4_gen_job.py:_topup`). What is added: one latent key `p3`
(class `knob`) per kept case, written by `haenv.p3.plan.make_plans` (decision balanced inside H,
drug, subtype, dose position, decoys). Kept cases: every eligible H0 case and the first H1 cases
by sha256(seeded "p3pool", id).

The pool grows with N: the top-up variant counts, the per-drug cap of each H stratum and the H1
cap are their N = 50 values times max(1, N / 50), rounded up.

Usage: python tools/p3_gen_job.py [--n 50] [--seed S] [--out PATH] [--check]
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import io
import json
import math
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "tools"))

from haenv import canary as CANARY  # noqa: E402
from haenv import pack_size as PS     # noqa: E402

SRC = ROOT / "inputs" / "ddx-workup.job.yaml"
OUT = ROOT / "inputs" / "p3-meds.job.yaml"
#: pool sizes at N = 50: top-up variants (H0, H1) and the H1 cap
TOPUP = (34, 10)
H1_CAP = 1000


def grow(v: int, n: int) -> int:
    """A pool size at N = 50 scaled to N (never below its N = 50 value)."""
    return math.ceil(v * max(1.0, n / PS.DEFAULT_N))


def build(n: int = PS.DEFAULT_N, seed: str = PS.PUBLIC_SEED) -> tuple[str, dict]:
    from haenv.p3.audit import planned_cells
    from haenv.p3.plan import CAP_PER_DRUG, eligible_drugs, h01, hidden, make_plans
    from p4_gen_job import _topup, _world_streams
    src = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    cases = [copy.deepcopy(c) for c in src["cases"]]
    topup, h1, cap = (grow(TOPUP[0], n), grow(TOPUP[1], n)), grow(H1_CAP, n), grow(CAP_PER_DRUG, n)
    top = _topup(*topup)
    allc = cases + top
    ws = {str(c["case_id"]): _world_streams(c) for c in allc}
    elig = {str(c["case_id"]): eligible_drugs(c, ws[str(c["case_id"])])
            for c in allc}
    full = {k: bool(v) for k, v in elig.items()}       # one eligible drug (A12; the decision is dealt among those it can carry)
    H0 = [c for c in allc if not hidden(c["latent"]) and full[str(c["case_id"])]]
    H1 = sorted([c for c in allc if hidden(c["latent"]) and full[str(c["case_id"])]],
                key=lambda c: h01(PS.seeded("p3pool", seed), c["case_id"]))[:h1]
    plans = make_plans(H0 + H1, ws, cap=cap)
    planned = collections.Counter((p["drug"], p["class"], p.get("comorbidity")) for p in plans.values())
    PS.short(planned_cells(n), planned, "p3 plans")
    kept = []
    for c in allc:
        cid = str(c["case_id"])
        if cid in plans:
            c = copy.deepcopy(c)
            c["latent"]["p3"] = plans[cid]
            kept.append(c)
    head = CANARY.block().splitlines()
    head += [
        "# Generated file -- do not edit by hand. Rebuild: python tools/p3_gen_job.py",
        "#",
        f"# Pack 3 -- chronic medication adjustment, {n} published items. Cases: the eligible H0",
        f"# cases of ddx-workup and of a {len(top)}-case variant top-up, and {len(H1)} eligible H1 cases",
        "# (first by sha256(seeded 'p3pool', id)); workup raw/latent copied verbatim, plus one latent key `p3`",
        "# (knob) per case: decision dealt by hash inside H, drug dealt inside the decision. The plugin",
        "# group `haenv.p3` (haenv/p3/plugin.py) realizes the item after the world's lab measurement",
        "# layer and judges it by code.",
        "#",
        "# SYNTHETIC data, evaluation only, not medical advice.",
    ]
    body = {"job_id": "p3-meds", "pack": PS.header(n, seed), "task_type": "joint_dx", "multiround": False, "include_baseline": False,
            "models": [], "sample_cases": 5, "report": "eval-p3-meds.md", "gated": False, "probe_id": "p3.meds",
            "plugins": {"judges": ["haenv.p3"]},
            "_provenance": {"generator": "tools/p3_gen_job.py", "source_job": "inputs/ddx-workup.job.yaml",
                            "source_sha16": hashlib.sha256(SRC.read_bytes()).hexdigest()[:16],
                            "topup": f"tools/p4_gen_job.py:_topup{topup}", "h1_pool": h1,
                            "cap_per_drug": cap},
            "cases": kept}
    buf = io.StringIO()
    yaml.safe_dump(body, buf, allow_unicode=True, sort_keys=False, width=120)
    stats = {"n_items": n, "workup": len(cases), "topup": len(top), "eligible_H0": len(H0),
             "eligible_H1": sum(1 for c in allc if hidden(c["latent"]) and full[str(c["case_id"])]),
             "ineligible": sum(1 for v in elig.values() if not v), "kept": len(kept),
             "cells": {f"H{h}:{c}": n for (h, c), n in sorted(collections.Counter((p["H"], p["class"]) for p in plans.values()).items())},
             "drug_by_class": {f"{c}:{d}": n for (c, d), n in sorted(collections.Counter((p["class"], p["drug"]) for p in plans.values()).items())},
             "subtypes": dict(collections.Counter(p["subtype"] for p in plans.values()))}
    return "\n".join(head) + "\n" + buf.getvalue(), stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=PS.DEFAULT_N)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    txt, st = build(a.n, PS.cli_seed(a.seed))
    if a.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == txt
        print(out.name, "up to date" if same else "DIFFERS")
        return 0 if same else 1
    out.write_text(txt, encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
