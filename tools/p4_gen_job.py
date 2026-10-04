"""p4_gen_job.py -- write the pack-4 follow-up job (`inputs/p4-followup.job.yaml`).

Inputs (read only):
  * `inputs/ddx-workup.job.yaml` -- the v1.0.1 tool-track pack (145 cases), copied verbatim
    (`raw` and `latent` unchanged after the YAML round trip) for every case kept here;
  * a variant top-up: independent-type conditions from variant 6 on and the others from variant 4
    on (`--h0-variants` / `--h1-variants`, default ceil(20 N/50) / ceil(8 N/50): creatinine items go
    only to patients with an ACEI indication, so the pool needs more cases), with the workup
    generator's distractor levels (`low:1,high:1`) and insufficient share 0.6.

What is added: one latent key `p4` (provenance class `knob`) per kept case, written by
`haenv.p4.plan.make_plans`: class (balanced inside H), analyte, sign, band position, decoys and the
non-true factor. Kept cases: every eligible H0 case, and the first `--h1` (default ceil(400 N/50))
eligible H1 cases by the seeded `sha256("p4pool", case_id)`. The pool grows with N in proportion
(variant counts and the H1 cap scale by N/50); a cell whose planned supply is below the pack's quota
stops the generator with the gap.

Usage: python tools/p4_gen_job.py [--n N] [--seed S] [--out PATH] [--two-pass] [--check]
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
import collections
import math
import copy
import hashlib
import io
import json
import os
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
OUT = ROOT / "inputs" / "p4-followup.job.yaml"
#: pool size of the N = 50 pack: top-up variants (H0, H1) and the H1 cap; scaled by N/50
POOL_50 = {"h0_variants": 20, "h1_variants": 8, "h1": 400}


def scratch_dir(out: pathlib.Path) -> pathlib.Path:
    """Where the first-pass job and its feasibility cache live (never under inputs/): HAENV_P4_SCRATCH,
    else next to a job written outside inputs/, else derived/p4."""
    if os.environ.get("HAENV_P4_SCRATCH"):
        return pathlib.Path(os.environ["HAENV_P4_SCRATCH"])
    if out.resolve() != OUT.resolve():
        return out.resolve().parent / "p4-gen"
    if os.environ.get("HAENV_OUTPUT_ROOT"):
        return pathlib.Path(os.environ["HAENV_OUTPUT_ROOT"]) / "p4-gen"
    return ROOT / "derived" / "p4"


def pool_size(n: int) -> dict:
    return {k: math.ceil(v * n / PS.DEFAULT_N) for k, v in POOL_50.items()}


def _topup(h0_variants: int, h1_variants: int) -> list[dict]:
    """More variants of the registry's conditions, made by the workup's own generator call:
    independent-type (H0) conditions from variant 6 on, the others (H1) from variant 4 on (the
    workup carries v1-v5 and v1-v3 of them). Distractor levels as the workup (`low:1,high:1`),
    insufficient share 0.6, no rhythm gap (this pack is not sliced)."""
    from haenv.ddx import ddx_case_specs, strip_inert_gap
    from haenv.overlay import condition_registry
    from gen_core_job import apply_distractor_level, assign_distractor_levels
    reg = condition_registry(include_draft=False)
    ind = [k for k in reg if reg[k]["join_gold"] == "independent"]
    oth = [k for k in reg if reg[k]["join_gold"] != "independent"]
    new = []
    if h0_variants:
        new += ddx_case_specs(only=ind, variants=h0_variants, variant_start=6, density="mixed", insufficient_frac=0.6)
    if h1_variants:
        new += ddx_case_specs(only=oth, variants=h1_variants, variant_start=4, density="mixed", insufficient_frac=0.6)
    lvl = assign_distractor_levels(new, "low:1,high:1")
    for c in new:
        apply_distractor_level(c, lvl[str(c["case_id"])])
    strip_inert_gap(new, "")
    return new


def _world_streams(case: dict) -> set[str]:
    from haenv.build import clinical_plan
    raw = case.get("raw") or {}
    return set(clinical_plan(str(raw.get("disease") or ""), list(raw.get("devices") or []),
                             list(raw.get("comorbidities") or []), str(case["case_id"])))


def feasibility(job_text: str, scratch: pathlib.Path) -> dict[str, list[list]]:
    """Build the first-pass job in process (deterministic generator, no model call) and record,
    for every case, which (analyte, sign) pairs its world can carry for its dealt class."""
    import logging
    logging.disable(logging.CRITICAL)
    scratch.mkdir(parents=True, exist_ok=True)
    jp = scratch / "p4-followup.pass1.job.yaml"
    jp.write_text(job_text, encoding="utf-8")
    from haenv import build as B
    from haenv.job import load_job
    from haenv.p4 import hooks
    from haenv.p4.plan import signs_of
    from haenv.p4.realize import Unrealizable, _realize_one
    job = load_job(str(jp), root=ROOT)
    out: dict[str, list[list]] = {}
    orig = hooks.observe_followup

    def probe(raw, cid, truth, audit):
        from haenv.p4.gate import plan_of
        plan = plan_of(raw)
        if plan is not None:
            ok = []
            for an in plan.get("analytes_ok") or ():
                for sg in signs_of(an):
                    try:
                        _realize_one(raw, str(cid), {**plan, "analyte": an, "sign": sg,
                                                     "decoys": (plan.get("decoys") or {}).get(an) or [],
                                                     "factor": (plan.get("factor") or {}).get(an)},
                                     B._LAB_CTX.get(str(cid)))
                        ok.append([an, sg])
                    except Unrealizable:
                        pass
            out[str(cid)] = ok
        return orig(raw, cid, truth, audit)
    hooks.observe_followup = probe
    try:
        for cs in job.cases:
            try:
                B.build_case(cs, max_rounds=6)
            except Exception as e:                      # noqa: BLE001  (a case that fails to build carries nothing)
                out.setdefault(cs.case_id, [])
                print(f"[p4] pass 1: {cs.case_id} failed to build: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        hooks.observe_followup = orig
        logging.disable(logging.NOTSET)
    return out


def build(h1: int, two_pass: bool = False, h0_variants: int = 6, h1_variants: int = 2, n: int = PS.DEFAULT_N,
          seed: str = PS.PUBLIC_SEED, scratch: pathlib.Path | None = None) -> tuple[str, dict]:
    from haenv.overlay import condition_registry
    from haenv.p4.audit import planned_cells
    from haenv.p4.plan import eligible_analytes, h01, hidden, make_plans
    src = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    cases = [copy.deepcopy(c) for c in src["cases"]]
    have = {str(c["case_id"]) for c in cases}
    top = _topup(h0_variants, h1_variants)
    clash = sorted(str(c["case_id"]) for c in top if str(c["case_id"]) in have)
    if clash:
        raise SystemExit(f"top-up case ids collide with the workup: {clash[:5]}")
    allc = cases + top
    specs = condition_registry(include_draft=True)
    ws = {str(c["case_id"]): _world_streams(c) for c in allc}
    elig = {str(c["case_id"]): eligible_analytes(c, specs.get(str(c["latent"].get("ddx_spec_id"))), ws[str(c["case_id"])])
            for c in allc}
    H0 = [c for c in allc if not hidden(c["latent"]) and elig[str(c["case_id"])]]
    H1 = sorted([c for c in allc if hidden(c["latent"]) and elig[str(c["case_id"])]],
                key=lambda c: h01(PS.seeded("p4pool", seed), c["case_id"]))[:h1]
    pool = H0 + H1
    plans = make_plans(pool, specs, ws)
    feas = None
    if two_pass:
        pass1 = _render(allc, plans, top, H1, h1, None, (h0_variants, h1_variants), n, seed)[0]
        from haenv.p4.hooks import p4_world_fingerprint
        key = hashlib.sha256((pass1 + "|" + p4_world_fingerprint()).encode()).hexdigest()[:16]
        scratch = scratch or scratch_dir(OUT)
        cache = scratch / "feasibility.json"
        cached = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
        if cached.get("key") == key:
            feas = cached["feasible"]
        else:
            feas = feasibility(pass1, scratch)
            cache.write_text(json.dumps({"key": key, "feasible": feas}, ensure_ascii=False, sort_keys=True, indent=0),
                             encoding="utf-8")
        plans = make_plans(pool, specs, ws, feasible=feas)
    PS.short(planned_cells(n), collections.Counter((p["H"], p["class"], p.get("comorbidity")) for p in plans.values()),
             "p4 planned cases")
    txt, kept = _render(allc, plans, top, H1, h1, feas, (h0_variants, h1_variants), n, seed)
    stats = {"workup": len(cases), "topup": len(top), "eligible_H0": len(H0),
             "eligible_H1": sum(1 for c in allc if hidden(c["latent"]) and elig[str(c["case_id"])]),
             "kept": len(kept), "pass1_feasible_none": sorted(k for k, v in (feas or {}).items() if not v),
             "ineligible": sorted(k for k, v in elig.items() if not v),
             "cells": dict(collections.Counter((p["H"], p["class"]) for p in plans.values())),
             "pair_by_class": dict(collections.Counter((p["class"], p["analyte"], p["sign"]) for p in plans.values()))}
    return txt, stats


def _render(allc, plans, top, H1, h1, feas, variants=(6, 2), n=PS.DEFAULT_N, seed=PS.PUBLIC_SEED) -> tuple[str, list]:
    kept = []
    for c in allc:
        cid = str(c["case_id"])
        if cid in plans:
            c = copy.deepcopy(c)
            c["latent"]["p4"] = plans[cid]
            kept.append(c)
    head = CANARY.block().splitlines()
    head += [
        "# Generated file -- do not edit by hand. Rebuild: python tools/p4_gen_job.py",
        "#",
        "# Pack 4 -- follow-up interpretation. Cases: the eligible",
        f"# H0 cases of ddx-workup and of a {len(top)}-case variant top-up, and {len(H1)} eligible H1 cases",
        "# (first by the seeded sha256('p4pool', id)); workup raw/latent copied verbatim, plus one",
        "# latent key `p4` (knob) per case: class dealt by hash inside H (sex blocks first); "
        + ("(analyte, sign) dealt inside each class among the pairs a deterministic first-pass build can carry."
           if feas is not None else "sign alternated inside (analyte, class), one pass; a pre-analytic item rises "
           "when today's draw is non-fasting, else falls."),
        "# The plugin group `haenv.p4` (haenv/p4/plugin.py) realizes",
        "# the follow-up item after the world's lab measurement layer and judges it by code.",
        "#",
        "# SYNTHETIC data, evaluation only, not medical advice.",
    ]
    body = {"job_id": "p4-followup", "pack": PS.header(n, seed), "task_type": "joint_dx", "multiround": False,
            "include_baseline": False, "models": [], "sample_cases": 5, "report": "eval-p4-followup.md",
            "gated": False, "probe_id": "p4.followup", "plugins": {"judges": ["haenv.p4"]},
            "_provenance": {"generator": "tools/p4_gen_job.py", "source_job": "inputs/ddx-workup.job.yaml",
                            "source_sha16": hashlib.sha256(SRC.read_bytes()).hexdigest()[:16],
                            "topup": f"ddx_case_specs(only=independent, variants={variants[0]}, variant_start=6) + "
                                     f"ddx_case_specs(only=other, variants={variants[1]}, variant_start=4); density=mixed, "
                                     "insufficient_frac=0.6, distractor low:1,high:1",
                            "h1_pool": h1,
                            "feasibility": ("two-pass: (analyte, sign) dealt inside each class among the pairs "
                                            "a deterministic first-pass build can carry" if feas is not None else "one-pass"),
                            "feasibility_sha16": (hashlib.sha256(json.dumps(feas, sort_keys=True).encode()).hexdigest()[:16]
                                                  if feas is not None else None)},
            "cases": kept}
    buf = io.StringIO()
    yaml.safe_dump(body, buf, allow_unicode=True, sort_keys=False, width=120)
    return "\n".join(head) + "\n" + buf.getvalue(), kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PS.DEFAULT_N, help="items in the pack")
    ap.add_argument("--seed", default=None, help="selection seed (else HAENV_PACK_SEED, else the public seed)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--h1", type=int, default=None, help="H1 cases kept (default ceil(400 N/50))")
    ap.add_argument("--h0-variants", type=int, default=None, help="default ceil(20 N/50)")
    ap.add_argument("--h1-variants", type=int, default=None, help="default ceil(8 N/50)")
    ap.add_argument("--two-pass", action="store_true", help="deal (analyte, sign) among the pairs a first-pass build can carry")
    ap.add_argument("--one-pass", action="store_true", help="the default: (analyte, sign) dealt blind")
    a = ap.parse_args()
    seed = PS.cli_seed(a.seed)
    out = pathlib.Path(a.out)
    size = {k: (getattr(a, k) if getattr(a, k) is not None else v) for k, v in pool_size(a.n).items()}
    txt, st = build(size["h1"], two_pass=a.two_pass and not a.one_pass, h0_variants=size["h0_variants"],
                    h1_variants=size["h1_variants"], n=a.n, seed=seed, scratch=scratch_dir(out))
    if a.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == txt
        print(out.name, "up to date" if same else "DIFFERS")
        return 0 if same else 1
    out.write_text(txt, encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in st.items()},
                     ensure_ascii=False, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
