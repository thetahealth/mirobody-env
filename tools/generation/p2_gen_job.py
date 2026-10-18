"""p2_gen_job.py -- write the rebuilt pack-2 triage job (`<output>/p2-triage.job.yaml`): a pool of
about POOL_FACTOR x N planned items, from which the audit publishes N (`haenv/p2/audit.py`).

Patients: the public seed copies the ddx-workup cases and adds workup-generator variants
(`p4_gen_job._topup`); a non-public seed draws every patient from its own variant window
(`p4_gen_job._window_supply`, W6), so each seed gets new patients with the same per-condition supply.
Only cases whose hidden condition needs no clinician (registry urgency green, no red flag) are used:
the hidden condition's own symptoms are shown and no printed rule reads them, so any other urgency
would contradict a calmer triage gold.

Plan (latent key `p2`, class knob), every draw a sha256 of the case id, blind to every case feature:
tier (scenario x rule, `registry/pack2_triage_scenarios.yaml`), dealt in proportion to the pack's
quotas (sourced RPM disposition prior, authored rule weights within each, a rule's tiers equal); inside each tier the
difficulty strata and the three wordings take turns; `atypical` goes only to a patient aged 65+
or with diabetes (else the tier's next stratum); a beta-blocker with every `masked` item, every
other heart-failure item, and a quarter of the other items; comorbidities, the glucose-lowering
agent and a wearable artefact by `background_plan`. Every condition the item adds to the record joins
the world's comorbidities (`world_case`), so the labs are drawn for the patient the record describes.

Zero model calls:
    PYTHONPATH=<repo>:<repo>/plugins python tools/generation/p2_gen_job.py [--n 100] [--seed S] [--out PATH]
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
import functools
import hashlib
import io
import json
import math
import pathlib
import sys

import yaml

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import haenv.provenance.canary as CANARY    # noqa: E402
import haenv.world.validation.pack_size as PS      # noqa: E402

SRC = ROOT / 'benchmarks' / 'inputs' / 'ddx-workup.job.yaml'
OUT = ROOT / 'benchmarks' / 'inputs' / 'p2-triage.job.yaml'
#: planned items per published item (the emission gate refuses a few; the audit needs POOL_FACTOR x N)
PLAN_FACTOR = PS.POOL_FACTOR * 1.25
BB_SHARE = 0.25
#: comorbidities drawn into the background of an item of any other scenario (the scenario's own
#: condition is never drawn twice); the rule reader is unaffected (normal CGM, flat weight, BP < 140/90)
COMORBID_SHARE = {"T2D": 0.35, "hypertension": 0.3, "heart_failure": 0.15}
#: Chronic conditions on treatment whose readings sit at target and fire no rule: the world holds the
#: condition and its medicine (`med_course.CONCURRENT`), the block lists both. Shares are the registry's
#: sourced "diagnosed" rates (`registry/background_comorbidity.yaml`: `eligible.dyslipidemia`,
#: `explainer_bases.{hypothyroidism,CAD,CKD}`), read from it so there is one source.
REGISTRY_COMORBID = ("dyslipidemia", "hypothyroidism", "CAD", "CKD")
#: a wearable artefact beside the signal of a non-benign item (`haenv.p2.render.ARTEFACT_KINDS`)
ARTEFACT_SHARE = 0.4


def u01(*parts) -> float:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:13], 16) / float(1 << 52)


def tier_counts(n: int) -> dict:
    """{tier: n}: the published composition (`haenv.p2.audit.planned_cells`)."""
    from haenv.p2.audit import planned_cells
    return planned_cells(n)


#: variant rounds the supply search tries, in this order; `SCAN_WIDTH` round counts are built at once
ROUNDS = tuple(range(2, 80, 2))
SCAN_WIDTH = 1


def candidates(seed: str, k: int) -> list[dict]:
    """The candidate patients of k variant rounds (the workup generator draws the whole list for a
    given k: tier, red flag and density of a variant depend on k, so the list is regenerated per k)."""
    from haenv.world.clinical.ddx import ddx_case_specs
    from haenv.world.clinical.ddx import strip_inert_gap
    from haenv.world.overlay import condition_registry
    from haenv.world.generation.patient_sampling import fresh_patient
    from tools.generation.p4_gen_job import with_pack_age
    from tools.generation.gen_core_job import apply_distractor_level, assign_distractor_levels
    # One equal-sized supply per eligible registry spec; no copied seed patient.
    registry = condition_registry()
    specs = [sid for sid, spec in registry.items()
             if spec["urgency"] == "🟢" and not spec.get("red_flag")]
    cases = ddx_case_specs(only=specs, variants=k, density="mixed", insufficient_frac=0.6)
    levels = assign_distractor_levels(cases, "low:1,high:1")
    for c in cases:
        apply_distractor_level(c, levels[c["case_id"]])
    strip_inert_gap(cases, "")
    return [with_pack_age(fresh_patient(c, "p2", seed), condition_specs=registry) for c in cases]



def _count(seed: str, k: int) -> tuple[int, int, int]:
    cases = candidates(seed, k)
    ok = [c for c in cases if eligible(c)]
    return len(cases), len(ok), sum(atypical_eligible(c) for c in ok)


def _forked_counts(seed: str, ks) -> dict:
    """{k: _count(seed, k)} computed in forked children (the target is not pickled, so the module may
    be loaded under any name and keeps its in-process state)."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    jobs = []
    for k in ks:
        rd, wr = ctx.Pipe(duplex=False)
        pr = ctx.Process(target=lambda w=wr, k=k: (w.send(_count(seed, k)), w.close()))
        pr.start()
        wr.close()
        jobs.append((k, pr, rd))
    out = {}
    for k, pr, rd in jobs:
        out[k] = rd.recv()
        pr.join()
        if pr.exitcode:
            raise SystemExit(f"[p2] supply worker for {k} rounds failed")
    return out


def supply(seed: str, need: int) -> tuple[list[dict], dict]:
    """Candidate patients: the fewest variant rounds that give `need` eligible cases. Round counts
    are tried `SCAN_WIDTH` at a time in worker processes; the smallest that suffices wins, as in a
    one-by-one scan."""
    from haenv.p2.audit import quota_cells
    atypical_need = sum(n for (t, cohort), n in quota_cells(need).items() if cohort == "atypical_eligible")
    from haenv.world.overlay import condition_registry
    n_specs = sum(s["urgency"] == "🟢" and not s.get("red_flag") for s in condition_registry().values())
    for k in range(max(2, math.ceil(need / n_specs)), max(ROUNDS) + 1):
        cases = candidates(seed, k)
        ok = [c for c in cases if eligible(c)]
        n_aty = sum(atypical_eligible(c) for c in ok)
        if len(ok) >= need and n_aty >= atypical_need:
            return ok, {"variants": k, "candidates": len(cases), "eligible": len(ok),
                        "window_base": PS.variant_base(seed), "eligible_specs": n_specs,
                        "atypical_eligible": n_aty, "atypical_needed": atypical_need}
    raise SystemExit(f"[p2] no supply reaches {need} eligible cases")


def eligible(c: dict) -> bool:
    """A hidden condition that itself calls for a clinician (registry urgency above green) is left out:
    its symptoms are shown, no printed rule reads them, and they would contradict a calmer gold."""
    lat = c.get("latent") or {}
    return not lat.get("ddx_red_flag") and str(lat.get("ddx_urgency")) == "🟢"


def _age_lo(c: dict) -> int:
    a = str((c.get("raw") or {}).get("age_range") or "0")
    return int(a.split("-")[0].rstrip("+") or 0)


def _diabetic(c: dict) -> bool:
    raw = c.get("raw") or {}
    conds = {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or ()}
    return bool(conds & {"T2D", "T1D", "type2_diabetes"})


def atypical_eligible(c: dict) -> bool:
    return _age_lo(c) >= 65 or _diabetic(c)


def make_plans(cases: list[dict], n_pool: int) -> dict:
    from haenv.p2 import rules as R
    tab = R.table()
    from haenv.p2.audit import quota_cells
    cells = quota_cells(n_pool)
    order = sorted(cases, key=lambda c: u01("p2-deal", c["case_id"]))
    reserved = [c for c in order if atypical_eligible(c)]
    need = sum(n for (t, cohort), n in cells.items() if cohort == "atypical_eligible")
    PS.short({"atypical_eligible": need, "all": n_pool}, {"atypical_eligible": len(reserved), "all": len(order)}, "p2 patient cohorts")
    chosen = reserved[:need]
    used = {c["case_id"] for c in chosen}
    ordinary = [c for c in order if c["case_id"] not in used][:n_pool - need]
    plans = {}
    by_tier = collections.defaultdict(list)
    for cohort, patients in (("atypical_eligible", chosen), ("ordinary", ordinary)):
        labels = sorted(((k + 0.5) / n, u01("p2-tier", t, cohort), t)
                        for (t, c), n in cells.items() if c == cohort for k in range(n))
        for c, (_, _, t) in zip(patients, labels):
            by_tier[t, cohort].append(c)
    for (t, cohort), cs in by_tier.items():
        strata = [s for s in tab["strata"][t] if s != "atypical"]
        if cohort == "atypical_eligible" and "atypical" in tab["strata"][t]:
            strata = ["atypical"]
        words = tab["wordings"]
        off_s, off_w = int(u01("p2-s-off", t) * len(strata)), int(u01("p2-w-off", t) * len(words))
        sc = tab["tiers"][t]["scenario"]
        for k, c in enumerate(sorted(cs, key=lambda c: u01("p2-turn", c["case_id"]))):
            cid = c["case_id"]
            s = strata[(k + off_s) % len(strata)]
            if t in ("af_rvr", "rate_new"):
                bb = False                   # fast AF at rest: no rate control on the list
            elif sc in ("infection", "af"):
                bb = s == "masked" or u01("p2-bb", cid) < BB_SHARE
            elif sc == "heart_failure":
                bb = k % 2 == 0
            else:
                bb = u01("p2-bb", cid) < BB_SHARE
            plans[cid] = {"tier": t, "scenario": sc, "floor": tab["tiers"][t]["floor"], "stratum": s, "cohort": cohort,
                          "wording": words[(k + off_w) % len(words)], "beta_blocker": bool(bb), **background_plan(cid, t, sc)}
    if layer_on():
        stratify_background(plans, {c["case_id"]: c for c in cases})
    return plans


def stratify_background(plans: dict, cases: dict) -> None:
    """Writes `plan["bg"]`, the background factors of each item, so that every factor lands at its `rate` inside each tier
    (rounding carried across tiers): the same frequency in every floor class by construction, not by chance. Within a tier
    the items that carry it are the first by a sha256 rank of (factor, case id); a factor goes only to items it
    applies to on what the plan already fixes (the render re-checks with the final record and ledger)."""
    from haenv.p2 import background as BG
    from haenv.p2 import rules as R
    from haenv.p2.render import needs
    tab = R.table()
    by_tier = collections.defaultdict(list)
    for cid, p in plans.items():
        by_tier[p["tier"]].append(cid)
        p["bg"] = []
    for fid, f in BG.spec()["factors"].items():
        carry = 0.0
        for tier in sorted(by_tier):
            sc = tab["tiers"][tier]["scenario"]
            elig = []
            for cid in sorted(by_tier[tier], key=lambda c: u01("p2-bg-rank", fid, c)):
                p, raw = plans[cid], cases[cid]["raw"]
                known = {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or ()}
                known |= set(needs(tier, p["stratum"], bool(p.get("beta_blocker")), p.get("comorbid") or (), p.get("dm_agent"))["add_conditions"])
                diabetic = bool(known & {"T2D", "T1D", "type2_diabetes"})
                if BG.applies(f, tier, sc, known, diabetic, sc == "infection", diabetic and p.get("dm_agent") in ("insulin", "su")):
                    elig.append(cid)
            want = f["rate"] * len(elig) + carry
            k = int(want + 1e-9)
            carry = want - k
            for cid in elig[:k]:
                plans[cid]["bg"].append(fid)
    stratify_artefact(plans, cases)


def stratify_artefact(plans: dict, cases: dict) -> None:
    """The exercise-peak / loose-strap artefact (`ARTEFACT_SHARE` of the non-benign items) at its share inside each tier, the
    kind alternating in rank order (a heart-failure patient gets the strap), replacing the per-item hash draw."""
    by_tier = collections.defaultdict(list)
    for cid, p in plans.items():
        p.pop("artefact", None)
        if p["scenario"] != "benign" or p["tier"] == "ben_sleep":
            by_tier[p["tier"]].append(cid)
    carry = 0.0
    for tier in sorted(by_tier):
        ranked = sorted(by_tier[tier], key=lambda c: u01("p2-art-rank", c))
        want = ARTEFACT_SHARE * len(ranked) + carry
        k = int(want + 1e-9)
        carry = want - k
        flip = 0
        for cid in ranked[:k]:
            hf = "heart_failure" in (plans[cid].get("comorbid") or ()) or plans[cid]["scenario"] == "heart_failure"
            if hf:
                plans[cid]["artefact"] = "strap"
            else:
                plans[cid]["artefact"] = ("strap", "exercise")[flip % 2]
                flip += 1


@functools.lru_cache(maxsize=1)
def _registry_rates() -> dict:
    reg = yaml.safe_load((ROOT / "registry" / "background_comorbidity.yaml").read_text(encoding="utf-8"))
    pool = {**reg["eligible"], **reg["explainer_bases"]}
    return {c: float(pool[c]["rate"]) for c in REGISTRY_COMORBID}


def layer_on() -> bool:
    """The background layer switch: `HAENV_P2_BACKGROUND=off` (or `--background off`) builds the pack without it."""
    import os
    if "HAENV_P2_BACKGROUND" in os.environ:
        return os.environ["HAENV_P2_BACKGROUND"].strip().lower() != "off"
    from haenv.world.generation.world_knobs import background_enabled
    return background_enabled()


def background_plan(cid: str, tier: str, sc: str) -> dict:
    """Comorbidities, the second glucose-lowering agent and a wearable artefact, each a sha256 draw of
    the case id: insulin, a beta-blocker, a CGM, heart failure or a wearable artefact then occurs in
    every floor class, and the class is read off the measurements."""
    from haenv.p2.render import SCENARIO_CONDITION
    own = SCENARIO_CONDITION.get(sc)
    shares = {**COMORBID_SHARE, **(_registry_rates() if layer_on() else {})}
    co = [c for c, share in shares.items() if c != own and u01("p2-co", c, cid) < share]
    if tier == "htn_severe":
        # severe BP with heart failure is R7a (24 h), its own tier `hf_htn_severe`; R7 is the patient without it
        co = [c for c in co if c != "heart_failure"]
    out = {"comorbid": co}
    if sc == "hypoglycaemia":
        out["dm_agent"] = "insulin" if u01("p2-dm", cid) < 0.5 else "su"
    elif "T2D" in co:
        # a CGM is worn mostly by people on insulin or a sulfonylurea
        u = u01("p2-dm", cid)
        out["dm_agent"] = "insulin" if u < 0.45 else "su" if u < 0.8 else "metformin"
    if sc != "benign" or tier == "ben_sleep":
        if u01("p2-art", cid) < ARTEFACT_SHARE:
            hf = "heart_failure" in co or sc == "heart_failure"
            out["artefact"] = "strap" if hf or u01("p2-art-kind", cid) < 0.5 else "exercise"
    if not layer_on():
        out["background"] = "off"
    return out


def world_case(case: dict, plan: dict) -> dict:
    """The case with every condition the item adds to the record (`render.needs`) among the world's
    comorbidities, so the world's labs are drawn for it, and the age band the build prints for them."""
    from tools.generation.p4_gen_job import with_pack_age
    from haenv.p2.render import BB_KINDS, needs
    from haenv.p2.rules import table
    c = copy.deepcopy(case)
    raw = c["raw"]
    world = [str(raw.get("disease") or "")] + list(raw.get("comorbidities") or ())
    add = needs(plan["tier"], plan["stratum"], bool(plan.get("beta_blocker")), plan.get("comorbid") or (),
                plan.get("dm_agent"))["add_conditions"]
    raw["comorbidities"] = list(raw.get("comorbidities") or ()) + [x for x in add if x not in world]
    if plan.get("beta_blocker"):
        plan = {**plan, "bb_kind": BB_KINDS[int(u01("p2-bb", c["case_id"]) * len(BB_KINDS))]}
    c["latent"]["p2"] = plan
    if table()["tiers"][plan["tier"]].get("context") == "volume":
        plan["context_T"] = int(c["latent"]["index_time_T"])
        plan["context_cid"] = str(c["case_id"])
    c = with_pack_age(c)
    if plan.get("cohort") == "atypical_eligible" and not atypical_eligible(c):
        raise SystemExit(f"[p2] {c['case_id']}: the added conditions move the patient out of the atypical cohort")
    return c


def build(n: int, seed: str) -> tuple[str, dict]:
    n_pool = math.ceil(PLAN_FACTOR * PS.pool_n(n))
    cases, st = supply(seed, n_pool)
    plans = make_plans(cases, n_pool)
    kept = [world_case(c, plans[c["case_id"]]) for c in cases if c["case_id"] in plans]
    head = CANARY.block().splitlines() + [
        "# Generated file -- do not edit by hand. Rebuild: python tools/generation/p2_gen_job.py",
        "#",
        f"# Pack 2 rebuilt (triage on wearable trends): a pool of {len(kept)} planned items; the audit publishes",
        f"# {n} (haenv/p2/audit.py). Plan per case under latent key `p2` (knob): tier, stratum, wording, beta-blocker.",
        "# SYNTHETIC data, evaluation only, not medical advice.",
    ]
    from haenv.p2.rules import table_sha16
    body = {"job_id": "p2-triage", "pack": PS.header(n, seed), "task_type": "joint_dx", "multiround": False,
            "include_baseline": False, "models": [], "sample_cases": 5, "report": "eval-p2-triage.md",
            "gated": False, "probe_id": "p2.triage", "plugins": {"judges": ["haenv.p2"]},
            "_provenance": {"generator": 'tools/generation/p2_gen_job.py', "source_job": 'benchmarks/inputs/ddx-workup.job.yaml',
                            "source_sha16": hashlib.sha256(SRC.read_bytes()).hexdigest()[:16],
                            "table_sha16": table_sha16(), "supply": st,
                            "instances": ("seed window (haenv.pack_size.variant_base), W6" if st["window_base"]
                                          else "ddx-workup copies + top-up variants (public seed)"),
                            "alloc_sha16": hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()[:16]},
            "cases": kept}
    buf = io.StringIO()
    yaml.safe_dump(body, buf, allow_unicode=True, sort_keys=False, width=120)
    stats = {**st, "pool_planned": len(kept), "tiers": dict(collections.Counter(p["tier"] for p in plans.values())),
             "strata": dict(collections.Counter(p["stratum"] for p in plans.values())),
             "beta_blocker": sum(p["beta_blocker"] for p in plans.values())}
    return "\n".join(head) + "\n" + buf.getvalue(), stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PS.DEFAULT_N)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--background", choices=("on", "off"), default=None, help="background layer (default on; env HAENV_P2_BACKGROUND)")
    a = ap.parse_args(argv)
    if a.background:
        import os
        os.environ["HAENV_P2_BACKGROUND"] = a.background
    seed = PS.cli_seed(a.seed)
    txt, st = build(a.n, seed)
    out = pathlib.Path(a.out)
    if a.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == txt
        print(out.name, "up to date" if same else "DIFFERS")
        return 0 if same else 1
    out.write_text(txt, encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
