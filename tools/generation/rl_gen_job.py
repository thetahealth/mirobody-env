"""rl_gen_job.py -- write the rl (red-line vigilance pairs) job, redline arm (job id `rl-pairs`; generated on
demand by `make_pack`, not tracked).

Patients: the v1.0.1 tool-track workup cases (`inputs/ddx-workup.job.yaml`) plus a variant top-up from the same
generator, each redrawn as an independent patient (`patient_sampling.fresh_patient(case, "rl", seed)`). Each
patient carries one rule of `registry/redline_rules.yaml` (families `--families`, S1: E and C): the rule's
indication joins the world's comorbidities when it is a world condition, the age band is the one the build
prints (`with_pack_age`), and a patient that does not meet the rule's `requires` is passed to the next rule.

What is added: one latent key `rl` (class `knob`) per case: `{rule_id, family, variant: "redline", question_id,
imminence_day, seed_salt}` (`haenv.rl.plan.make_plan`). The control arm is the same job with `variant: control`
(`make_pack --paired-controls` / `--paired-with`). The pool holds about POOL_FACTOR x N candidates per arm, spread
evenly over the rules; the audit selects N by the registry family mix (`haenv/rl/audit.py`).

Usage: python tools/generation/rl_gen_job.py [--n N] [--seed S] [--out PATH] [--families E,C]
(make_pack: HAENV_RL_FAMILIES=G,A,B selects the families of the pool)
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

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
import os
import pathlib
import sys

import yaml

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))

import haenv.provenance.canary as CANARY  # noqa: E402
import haenv.world.validation.pack_size as PS  # noqa: E402

SRC = ROOT / 'benchmarks' / 'inputs' / 'ddx-workup.job.yaml'
OUT = ROOT / 'benchmarks' / 'inputs' / 'rl-pairs.job.yaml'
JOB_ID = "rl-pairs"
#: candidates per arm over the published N (attrition: emission refusals, infeasible rules)
OVERSUPPLY = 1.5
WORLD_CONDITIONS_TABLE = "standard_of_care.yaml"


def world_conditions() -> set[str]:
    from haenv.resources.regpath import load_registry
    return set((load_registry(WORLD_CONDITIONS_TABLE) or {}).get("conditions") or {})


def topup(variants: int) -> list[dict]:
    """More registry variants by the workup's own generator call (from variant 6 on)."""
    from haenv.world.clinical.ddx import ddx_case_specs, strip_inert_gap
    from tools.generation.gen_core_job import apply_distractor_level, assign_distractor_levels
    new = ddx_case_specs(variants=variants, variant_start=6, density="mixed", insufficient_frac=0.6)
    lvl = assign_distractor_levels(new, "low:1,high:1")
    for c in new:
        apply_distractor_level(c, lvl[str(c["case_id"])])
    strip_inert_gap(new, "")
    return new


def with_pack_age(case: dict, specs: dict) -> dict:
    from haenv.world.generation.build import pack_age_of
    band = pack_age_of(str(case["case_id"]), case.get("raw") or {}, case.get("latent") or {}, condition_specs=specs)
    if band == (case.get("raw") or {}).get("age_range"):
        return case
    return {**case, "raw": {**case["raw"], "age_range": band}}


def band_ok(rule: dict, raw: dict, age: int) -> bool:
    req = rule.get("requires") or {}
    if req.get("sex") and raw.get("sex") != req["sex"]:
        return False
    if req.get("age") and not req["age"][0] <= age <= req["age"][1]:
        return False
    weight = min(float(raw.get("start_weight") or 1e9), float(raw.get("nadir_weight") or 1e9))
    if req.get("weight_max_kg") is not None and weight > float(req["weight_max_kg"]):
        return False
    if req.get("any_of"):
        weight = min(float(raw.get("start_weight") or 1e9), float(raw.get("nadir_weight") or 1e9))
        return any(("age_min" in c and age >= int(c["age_min"])) or
                   ("weight_max_kg" in c and weight <= float(c["weight_max_kg"])) for c in req["any_of"])
    return True


def world_case(case: dict, rule: dict, specs: dict, wc: set[str]) -> dict:
    c = copy.deepcopy(case)
    raw = c["raw"]
    have = {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or ()}
    for cond in [rule.get("indication")] + list(rule.get("extra_conditions") or ()):
        if cond and cond in wc and cond not in have:
            raw["comorbidities"] = list(raw.get("comorbidities") or ()) + [cond]
            have.add(cond)
    return with_pack_age(c, specs)


def parse_mix(text: str) -> dict[str, int]:
    """'E:15,C:15,G:15' -> {'E': 15, ...} (a published family mix that overrides the registry's)."""
    out = {}
    for part in (text or "").split(","):
        if part.strip():
            f, _, w = part.partition(":")
            out[f.strip()] = int(w)
    return out


def build(n: int, seed: str, families: tuple[str, ...], mix: dict | None = None) -> tuple[str, dict]:
    from haenv.rl import rules as R
    from haenv.rl.plan import band_age, h01, make_plan
    from haenv.world.generation.patient_sampling import fresh_patient
    from haenv.world.overlay import condition_registry
    rules = [r for r in R.load().values() if r["family"] in families]
    if not rules:
        raise SystemExit(f"no rules for families {families}")
    per_rule = math.ceil(OVERSUPPLY * PS.POOL_FACTOR * n / len(rules))
    need = per_rule * len(rules)
    src = yaml.load(SRC.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
    cases = [copy.deepcopy(c) for c in src["cases"]]
    variants = 0
    while len(cases) < 2 * need:
        variants += 2
        cases = [copy.deepcopy(c) for c in src["cases"]] + topup(variants)
    specs = condition_registry(include_draft=True)
    wc = world_conditions()
    tag = PS.seeded("rlpool", seed)
    patients = sorted((fresh_patient(c, "rl", seed) for c in cases), key=lambda c: h01(tag, c["case_id"]))
    used: set[str] = set()
    kept, short = [], {}
    # the most restrictive rules choose first
    order = sorted(rules, key=lambda r: (-len(r.get("requires") or {}), r["id"]))
    for r in order:
        got = 0
        for c in patients:
            if got >= per_rule:
                break
            if c["case_id"] in used:
                continue
            w = world_case(c, r, specs, wc)
            raw = w["raw"]
            age = band_age(raw["age_range"], w["case_id"], r["id"])
            if not band_ok(r, raw, age):
                continue
            # the world's hidden diagnosis must not be named by the item's own text (ddx alias scan)
            text = " ".join(str(x) for x in (r.get("indication_text"), r["drug"]["dose_text"])).lower()
            if any(str(al).lower() in text for al in w["latent"].get("ddx_aliases") or () if len(str(al)) > 3):
                continue
            conds = [raw.get("disease")] + list(raw.get("comorbidities") or ())
            T = int(w["latent"]["index_time_T"])
            p = make_plan(w["case_id"], r, age, raw.get("sex"), conds, tag)
            p["imminence_day"] = T + int(p["imminence_day"])
            w["latent"]["rl"] = p
            kept.append(w)
            used.add(c["case_id"])
            got += 1
        if got < per_rule:
            short[r["id"]] = got
    kept.sort(key=lambda c: c["case_id"])
    head = CANARY.block().splitlines() + [
        "# Generated file -- do not edit by hand. Rebuild: python tools/generation/rl_gen_job.py",
        "#",
        "# rl pack -- red-line vigilance pairs, redline arm. Patients: ddx-workup cases plus a variant top-up,",
        "# each redrawn by patient_sampling.fresh_patient(case, 'rl', seed); one latent key `rl` (knob) per case.",
        "# The plugin group `haenv.rl` (haenv/rl/plugin.py) realizes the record after the world's measurement layer.",
        "#",
        "# SYNTHETIC data, evaluation only, not medical advice.",
    ]
    body = {"job_id": JOB_ID, "pack": PS.header(n, seed), "task_type": "joint_dx", "multiround": False,
            "include_baseline": False, "models": [], "sample_cases": 5, "report": "eval-rl-pairs.md",
            "gated": False, "probe_id": "rl.vigilance", "plugins": {"judges": ["haenv.rl"]},
            "_provenance": {"generator": "tools/generation/rl_gen_job.py",
                            "source_job": "benchmarks/inputs/ddx-workup.job.yaml",
                            "source_sha16": hashlib.sha256(SRC.read_bytes()).hexdigest()[:16],
                            "topup_variants": variants, "families": list(families), "per_rule": per_rule,
                            "short_rules": short, **({"rl_family_mix": mix} if mix else {})},
            "cases": kept}
    buf = io.StringIO()
    yaml.dump(body, buf, Dumper=yaml.CSafeDumper, allow_unicode=True, sort_keys=False, width=120)
    stats = {"candidates": len(kept), "per_rule": per_rule, "short_rules": short, "topup_variants": variants,
             "by_rule": dict(collections.Counter(c["latent"]["rl"]["rule_id"] for c in kept))}
    return "\n".join(head) + "\n" + buf.getvalue(), stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--families", default=os.environ.get("HAENV_RL_FAMILIES", "E,C"),
                    help="rule families (default HAENV_RL_FAMILIES, else E,C); make_pack passes the environment")
    ap.add_argument("--family-mix", default=os.environ.get("HAENV_RL_FAMILY_MIX", ""),
                    help="published family mix, e.g. E:15,C:15,G:15,A:15,B:15 (default HAENV_RL_FAMILY_MIX, else "
                         "the registry family_mix); recorded in the job, read by the audit")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    seed = PS.cli_seed(a.seed)
    txt, st = build(a.n, seed, tuple(x.strip() for x in a.families.split(",") if x.strip()), parse_mix(a.family_mix))
    out = pathlib.Path(a.out)
    if a.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == txt
        print(out.name, "up to date" if same else "DIFFERS")
        return 0 if same else 1
    out.write_text(txt, encoding="utf-8")
    print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
