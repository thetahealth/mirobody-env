"""p3_gen_job.py -- write the pack-3 chronic-medication job for N items (`inputs/p3-meds.job.yaml`).

Inputs (read only): `inputs/ddx-workup.job.yaml` (145 cases, copied verbatim) and the same
variant top-up pack 4 uses (`tools/generation/p4_gen_job.py:_topup`). What is added: one latent key `p3`
(class `knob`) per kept case, written by `haenv.p3.plan.make_plans` (decision balanced inside H,
drug, subtype, dose position, decoys). Kept cases: every eligible H0 case and the first H1 cases
by sha256(seeded "p3pool", id). Every condition the item puts on the record joins the world's
comorbidities (`world_case`), so the labs are drawn for the patient the record describes.

W6 (seed instantiation): a non-public seed copies no workup case; per registered condition it draws the
public pool's supply (workup copies plus top-up variants) from its own variant window
(`haenv.pack_size.variant_base`, `tools/generation/p4_gen_job.py:_window_supply`), so a new seed brings new
patients while the plans, dealt by case id, keep the composition cells. When the window's draw carries
too few plans (below `WINDOW_FLOOR` x `POOL_FACTOR` x N, or a cell short of its quota), it adds one
variant per condition and retries, at most `WINDOW_EXTRA_MAX` times; the public pool never retries.

The pool grows with N: the top-up variant counts, the per-drug cap of each H stratum and the H1
cap are their N = 50 values times max(1, N / 50) times `POOL_MARGIN`, rounded up. The shortcut audits
read the pool, so it must hold at least `POOL_FACTOR` x N items (`haenv.pack_size.POOL_FACTOR`); `POOL_MARGIN` covers the share of planned
cases the realizer refuses (about 8%), and the generator stops when the plans alone fall short.

Usage: python tools/generation/p3_gen_job.py [--n 50] [--seed S] [--out PATH] [--check]
SYNTHETIC data, evaluation only, not medical advice.
"""
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
import collections
import copy
import hashlib
import io
import json
import math
import pathlib
import sys
import os
import subprocess

import yaml

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "tools"))

import haenv.provenance.canary as CANARY  # noqa: E402
import haenv.world.validation.pack_size as PS     # noqa: E402

SRC = ROOT / 'benchmarks' / 'inputs' / 'ddx-workup.job.yaml'
OUT = ROOT / 'benchmarks' / 'inputs' / 'p3-meds.job.yaml'
#: pool sizes at N = 50: top-up variants (H0, H1) and the H1 cap
TOPUP = (34, 10)
H1_CAP = 1000
#: the pool is at least this many times N
POOL_FACTOR = getattr(PS, "POOL_FACTOR", 4)
#: top-up and caps over their N-scaled size, for the planned cases the realizer refuses
POOL_MARGIN = 1.2
#: a seed's window adds at most this many variants per condition to reach its floor
WINDOW_EXTRA_MAX = 5
#: a seed's window keeps at least this many times `POOL_FACTOR` x N planned cases, so the pool still
#: holds the floor after the realizer's refusals (7-9% at N = 100)
WINDOW_FLOOR = 1.15


def grow(v: int, n: int) -> int:
    """A pool size at N = 50 scaled to N (never below its N = 50 value)."""
    return math.ceil(v * max(1.0, n / PS.DEFAULT_N) * POOL_MARGIN)


def world_case(case: dict, plan: dict, *, condition_specs: dict | None = None) -> dict:
    """The case with the conditions the item puts on the record (`haenv.p3.plan.record_case`: the drug's
    condition, the planned comorbidity) among the world's comorbidities and the age band the build prints
    for them, so the world's labs are drawn for the patient the record describes."""
    from haenv.world.labworld.meds import drug
    from haenv.p3.plan import record_case, record_streams
    c = copy.deepcopy(record_case(case, plan["drug"], plan.get("comorbidity"), condition_specs=condition_specs))
    c["latent"]["p3"] = plan
    seen = set(drug(plan["drug"]).get("bearing_streams") or ()) & record_streams(c)
    if seen:
        raise SystemExit(f"[p3] {c['case_id']}: the record's world draws {sorted(seen)}, which bear on {plan['drug']}")
    return c


def build(n: int = PS.DEFAULT_N, seed: str = PS.PUBLIC_SEED, window_extra: int = 0,
          supply_multiplier: int = 1, *, workers: int = 64) -> tuple[str, dict]:
    from haenv.world.overlay import condition_registry
    from haenv.p3.audit import planned_scenarios
    from haenv.p3.plan import CAP_PER_DRUG, eligible_drugs, h01, hidden, make_plans
    from tools.generation.p4_gen_job import _topup, _window_supply, _world_streams, with_pack_age
    src = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    cases = [copy.deepcopy(c) for c in src["cases"]]
    # the pool is composed for `PS.pool_n(n)`, so the packs of the nested sizes share it (`haenv/world/validation/pack_size.py`)
    m = PS.pool_n(n)
    topup, h1, cap = ((grow(TOPUP[0], m) * supply_multiplier, grow(TOPUP[1], m) * supply_multiplier),
                      grow(H1_CAP, m) * supply_multiplier, grow(CAP_PER_DRUG, m) * supply_multiplier)
    base = PS.variant_base(seed)
    if base:
        # W6: a non-public seed draws every patient from its variant window, per condition the public
        # pool's supply (workup copies + top-up variants), plus `window_extra` variants per condition
        top = _window_supply(base, cases, topup[0] + window_extra, topup[1] + window_extra)
        cases = []
    else:
        top = _topup(*topup)
    condition_specs = condition_registry(include_draft=True)
    from haenv.world.generation.patient_sampling import fresh_patient
    from haenv.world.generation.build_pool import map_ordered
    def candidate(c):
        c = with_pack_age(fresh_patient(c, "p3", seed), condition_specs=condition_specs)
        streams = _world_streams(c)
        return c, streams, eligible_drugs(c, streams, condition_specs=condition_specs)
    # Candidates are independent. imap preserves the source order used for dealing,
    # including every private-window retry; children inherit the loaded registry.
    candidates = map_ordered(candidate, cases + top, workers, chunksize=8)
    allc = [c for c, _, _ in candidates]
    ws = {str(c["case_id"]): streams for c, streams, _ in candidates}
    elig = {str(c["case_id"]): drugs for c, _, drugs in candidates}
    full = {k: bool(v) for k, v in elig.items()}       # one eligible drug (A12; the decision is dealt among those it can carry)
    H0 = [c for c in allc if not hidden(c["latent"]) and full[str(c["case_id"])]]
    H1 = sorted([c for c in allc if hidden(c["latent"]) and full[str(c["case_id"])]],
                key=lambda c: h01(PS.seeded("p3pool", seed), c["case_id"]))[:h1]
    plans = make_plans(H0 + H1, ws, cap=cap, condition_specs=condition_specs, eligibility=elig)
    planned = collections.Counter((p["drug"], p["class"], p.get("comorbidity"), p["subtype"]) for p in plans.values())
    need = planned_scenarios(m)
    if base and window_extra < WINDOW_EXTRA_MAX and (
            len(plans) < WINDOW_FLOOR * POOL_FACTOR * m or any(planned.get(k, 0) < v for k, v in need.items())):
        # A seed's window draws new patients, and how many of them can carry a medication item
        # varies; one more variant per condition until the floor holds (the public pool never retries).
        print(f"[p3] window plans {len(plans)} < {WINDOW_FLOOR} x {POOL_FACTOR} x {m} or a cell short: "
              "one more variant per condition", file=sys.stderr)
        return build(n, seed, window_extra + 1, supply_multiplier, workers=workers)
    PS.short(need, planned, "p3 plans")
    if len(plans) < POOL_FACTOR * m:
        raise SystemExit(f"[pack] p3 plans: pool too small -- {len(plans)} planned cases, need >= {POOL_FACTOR} x {m}")
    kept = [world_case(c, plans[str(c["case_id"])], condition_specs=condition_specs)
            for c in allc if str(c["case_id"]) in plans]
    head = CANARY.block().splitlines()
    head += [
        "# Generated file -- do not edit by hand. Rebuild: python tools/generation/p3_gen_job.py",
        "#",
        f"# Pack 3 -- chronic medication adjustment, {n} published items. Cases: the eligible H0",
        f"# cases of ddx-workup and of a {len(top)}-case variant top-up, and {len(H1)} eligible H1 cases",
        "# (first by sha256(seeded 'p3pool', id)); diagnostic facts retained; patient profile and benign history independently sampled; plus latent `p3`",
        "# (knob) per case: decision dealt by hash inside H, drug dealt inside the decision. The plugin",
        "# group `haenv.p3` (haenv/p3/plugin.py) realizes the item after the world's lab measurement",
        "# layer and judges it by code.",
        "#",
        "# SYNTHETIC data, evaluation only, not medical advice.",
    ]
    body = {"job_id": "p3-meds", "pack": PS.header(n, seed), "task_type": "joint_dx", "multiround": False, "include_baseline": False,
            "models": [], "sample_cases": 5, "report": "eval-p3-meds.md", "gated": False, "probe_id": "p3.meds",
            "plugins": {"judges": ["haenv.p3"]},
            "_provenance": {"generator": 'tools/generation/p3_gen_job.py', "source_job": 'benchmarks/inputs/ddx-workup.job.yaml',
                            "source_sha16": hashlib.sha256(SRC.read_bytes()).hexdigest()[:16],
                            "topup": f"tools/generation/p4_gen_job.py:_topup{topup}", "h1_pool": h1,
                            "cap_per_drug": cap},
            "cases": kept}
    if base:
        body["_provenance"]["instances"] = (
            "seed window (haenv.pack_size.variant_base): per condition the public pool's supply (workup copies "
            f"+ top-up variants) plus {window_extra} drawn as variants base+1.. by "
            "tools/generation/p4_gen_job.py:_window_supply; no workup copy")
    buf = io.StringIO()
    yaml.safe_dump(body, buf, allow_unicode=True, sort_keys=False, width=120)
    stats = {"n_items": n, "workup": len(cases), "topup": len(top), "window_extra_variants": window_extra,
             "eligible_H0": len(H0),
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
    seed = PS.cli_seed(a.seed)
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled():
        from haenv.p3.plan import planned_scenarios
        required = {scenario_key(k): v for k, v in planned_scenarios(PS.pool_n(a.n)).items()}
        attempts = []
        for multiplier in (1, 2, 4, 8):
            txt, st = build(a.n, seed, supply_multiplier=multiplier)
            counts, pool_count = supply_preview(txt, out.parent / "p3-gen", multiplier)
            missing = supply_gaps(counts, required, pool_count, POOL_FACTOR * PS.pool_n(a.n))
            attempts.append({"multiplier": multiplier, "pool": pool_count, "cells": counts, "missing": missing})
            if not missing:
                st["clinical_supply"] = attempts
                break
        else:
            raise ValueError(f"clinical candidate supply exhausted: {attempts}")
    else:
        txt, st = build(a.n, seed)
    if a.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == txt
        print(out.name, "up to date" if same else "DIFFERS")
        return 0 if same else 1
    out.write_text(txt, encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=1))
    return 0


def scenario_key(cell):
    return "|".join("none" if v is None else str(v) for v in cell)


def supply_gaps(counts, required, pool_count, minimum_pool):
    """Only clinical quota supply, before any surface or classifier audit."""
    missing = {k: {"have": counts.get(k, 0), "need": v} for k, v in required.items() if counts.get(k, 0) < v}
    if pool_count < minimum_pool:
        missing["pool_floor"] = {"have": pool_count, "need": minimum_pool}
    return missing


def supply_preview(text, scratch, multiplier):
    """Retain both prospective worlds; count only their common clinical supply."""
    from haenv.app.settings import SET_ENV
    attempt = scratch / f"clinical-supply-{multiplier}"
    attempt.mkdir(parents=True, exist_ok=True)
    job = attempt / "p3-meds.job.yaml"
    job.write_text(text, encoding="utf-8")
    worlds = []
    for arm, settings in (("real", json.loads(os.environ.get(SET_ENV, "[]"))),
                          ("control", ["world.history=short", "world.med_dynamics=false", "background=false"])):
        target = attempt / arm
        target.mkdir(exist_ok=True)
        sets = [x for item in ["env_file=/dev/null", *settings] for x in ("--set", item)]
        cmd = [sys.executable, "-m", "haenv", "build", str(job), "--gen", "deterministic", "--fresh",
               "--gen-workers", os.environ.get("HAENV_GEN_WORKERS", "32"), *sets]
        with (target / "build.log").open("w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, SET_ENV: json.dumps(settings),
                                "HAENV_OUTPUT_ROOT": str(target)}, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f"clinical preview failed ({rc}): {target}")
        files = sorted((target / "results").glob("*/p3-meds/*/cases.jsonl"))
        if not files:
            raise RuntimeError(f"clinical preview produced no cases: {target}")
        worlds.append({r["case_id"]: r["case"]["adjudication"]["meds_p3"]
                       for r in map(json.loads, files[-1].read_text(encoding="utf-8").splitlines())})
    common = set(worlds[0]) & set(worlds[1])
    counts = collections.Counter(scenario_key((g["drug"], g["class"], (g.get("comorbidity") or {}).get("code"), g["subtype"]))
                                 for cid, g in worlds[0].items() if cid in common)
    (attempt / "supply.json").write_text(json.dumps({"pool": len(common), "cells": counts}, sort_keys=True, indent=2) + "\n")
    return dict(counts), len(common)


if __name__ == "__main__":
    raise SystemExit(main())
