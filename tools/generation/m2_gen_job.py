"""m2_gen_job.py -- write pack 1 (differential diagnosis, `m2-pack1`) for any pack size N.

  python tools/generation/m2_gen_job.py [--n N] [--seed S] [--out PATH] [--check]

Steps (zero model calls):
  1. pool: the v1.0.1 bridge set (F0, `tools/generation/m2/alloc_cases.csv`, copied verbatim from
     `inputs/ddx-workup.job.yaml`) plus F1-F3 rows from `tools/generation/m2/recompose_v3.py`, scaled by
     the pack's non-bridge item count over 34;
  2. compose: F1-F3 cases get one latent key `m2` (provenance class `knob`); F0 cases get `m2_bridge`
     (knob): their diagnostic gold and frame stay v1.0.1's, their triage pair follows the current
     condition registry (`bridge_triage`), and their menu and purchase readings are M2's;
  3. emit: the pool is built with the deterministic generator; a case the build refuses, or an F1-F3
     case with a symptom phrase that has no lay sentences, is not selected;
  4. select: `tools/generation/m2/pack1_select.py` with the cell sizes of `haenv/m2/pack.py:quotas(N)`,
     ordered by sha256(case_id + seed).

When HD-UNI-07..20 are integrated, auto mode instead builds benchmark clinical
strata, requires every new diagnosis in the published selection, and measures
difficulty without imposing a test-count quota. --allocation-mode legacy retains
the original design. The metadata-only clinical_expansion.yaml supplies systems.

The bridge takes its proportional share up to what the v1.0.1 bridge set supplies; it is a frozen
control set, so a shortfall is filled by new F1-F3 items in proportion and recorded in `_provenance`.
The seed comes from `--seed`, else HAENV_PACK_SEED, else the public seed; the job records only its
sha256 (`pack` header).

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
from collections import Counter
import copy
import csv
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import yaml

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
import haenv.provenance.canary as CANARY  # noqa: E402
import haenv.world.validation.pack_size as PS  # noqa: E402
from haenv.m2 import pack as PACK  # noqa: E402
from haenv.m2.core import (acute_event_for, explain_away_render, scene_text,  # noqa: E402
                           visible_medication_text)

SRC = ROOT / 'benchmarks' / 'inputs' / 'ddx-workup.job.yaml'
ALLOC = ROOT / 'tools/generation/m2/alloc_cases.csv'
OUT = ROOT / 'benchmarks' / 'inputs' / 'm2-pack1.job.yaml'
RECOMPOSE = ROOT / 'tools/generation/m2/recompose_v3.py'
SELECT = ROOT / 'tools/generation/m2/pack1_select.py'

FRAMES = ("F0", "F1", "F2", "F3")
#: K-NT: number of gold tests; (5, 8] is the hard side (spec 4.3, knobs.json).
K_NT_HARD_MIN = 6
#: non-bridge items of a 50-item pack (scale 1 of the F1-F3 pool)
NB_PER_SCALE = 34
#: The F1-F3 pool holds at least this many cases per pack item, so the shared audit's shortcut gates
#: (SA-5/SA-6) and the selection rules (per-spec cap, explain-away twins) have a pool to read.
POOL_PER_ITEM = 4
#: F1-F3 pool cases per unit of scale (`recompose_v3.FRAME_W`).
POOL_PER_SCALE = 61
#: The composed pool is this much above POOL_PER_ITEM x N: the build refuses some cases and the selection
#: filters (no lay sentence, no separating clue, age named in a symptom) drop others; the published job
#: carries the selectable pool, which must still hold POOL_PER_ITEM x the pack's F1-F3 items (SA-0).
POOL_MARGIN = 1.2
#: Age written into a symptom text ("38岁难治高血压").
_AGE_TOKEN = __import__("re").compile(r"(\d{2})\s*岁")


def age_conflict(case: dict) -> bool:
    """A symptom text names an age outside the case's sampled age range (`raw.age_range`, "60-64")."""
    try:
        lo, hi = (int(x) for x in str(case["raw"].get("age_range") or "").split("-"))
    except ValueError:
        return False
    return any(not lo <= int(m) <= hi for s in case["raw"].get("symptoms") or ()
               for m in _AGE_TOKEN.findall(str(s.get("text") or "")))


def retime(case: dict) -> dict:
    """An F1-F3 case's index day T drawn from the T domain by case id alone (salt `m2-T`), the same
    draw on both arms: the kernel's long-horizon promotion (yellow/green urgency => T=336) would make T
    read the arm. Legacy symptom days scale with T; expansion cases use the
    integrated clinical timing helper. Course end, nadir and course weeks follow T."""
    import haenv.world.generation.rng as rng
    from haenv.world.clinical.ddx import _course_end_for
    from haenv.contracts.job_schema import KNOB_DOMAIN
    cid, lat = case["case_id"], copy.deepcopy(case["latent"])
    T0, T = int(lat["index_time_T"]), int(rng.pick(list(KNOB_DOMAIN["index_time_T"]), cid, "m2-T"))
    expanded = lat.get("ddx_spec_id") in PACK.EXPANSION_IDS
    if expanded:
        import haenv.world.clinical.ddx as ddx
        timing = getattr(ddx, "clinical_symptom_days", None)
        history_time = getattr(ddx, "clinical_index_time", None)
        if timing is None or history_time is None:
            raise ValueError(f"{lat['ddx_spec_id']}: retiming requires integrated "
                             "ddx.clinical_symptom_days and ddx.clinical_index_time")
        T = int(history_time(cid, lat["ddx_spec_id"], T))
    lat.pop("long_horizon_tier", None)
    if T == T0 and (expanded or not BACKGROUND_ON[0]):
        return {**case, "latent": lat}
    if expanded:
        days = timing(cid, lat["ddx_spec_id"], [int(s["day"]) for s in case["raw"].get("symptoms") or ()], T)
        if (len(days) != len(case["raw"].get("symptoms") or ())
                or any(not isinstance(d, int) or isinstance(d, bool) or not 0 <= d <= T for d in days)):
            raise ValueError(f"{lat['ddx_spec_id']}: clinical_symptom_days returned an invalid schedule")
    elif BACKGROUND_ON[0]:
        days = spread_symptom_days(cid, len(case["raw"].get("symptoms") or ()), T)
    else:
        days = []
        for s in case["raw"].get("symptoms") or ():
            d = int(round(int(s["day"]) * T / T0))
            floor = days[-1] + 5 if days else 7
            days.append(max(floor, min(d, max(7, T - 3))))
    raw = {**case["raw"], "symptoms": [{**s, "day": d} for s, d in zip(case["raw"].get("symptoms") or (), days)]}
    ce = _course_end_for(cid, T)
    lat.update(index_time_T=T, course_end_day=ce)
    if "nadir_day" in lat:
        lat["nadir_day"] = int(round(ce * 0.85))
    if isinstance(lat.get("event_density"), dict) and "course_weeks" in lat["event_density"]:
        lat["event_density"] = {**lat["event_density"], "course_weeks": round(ce / 7.0, 1)}
    return {**case, "raw": raw, "latent": lat}


#: The background layer switch (`--background on|off`, else HAENV_M2_BACKGROUND, default on). Off reproduces the
#: pack of before the layer byte for byte: no background comorbidity, the template symptom days, the source
#: life-event rates, no `m2.background` marker.
from haenv.world.generation.world_knobs import background_enabled
BACKGROUND_ON: list[bool] = [os.environ["HAENV_M2_BACKGROUND"].strip().lower() != "off"
                           if "HAENV_M2_BACKGROUND" in os.environ else background_enabled()]


from haenv.m2.symptom_timing import spread_symptom_days


def with_pack_age(case: dict) -> dict:
    """An F1-F3 patient's age band drawn from its conditions (`build.pack_age_of`, the band the build
    prints), before any explainer base is attached: the base workup draws age independently of disease."""
    from haenv.world.generation.build import pack_age_of
    band = pack_age_of(case["case_id"], case["raw"], case["latent"])
    return case if band == case["raw"].get("age_range") else {**case, "raw": {**case["raw"], "age_range": band}}


def _explain_away_rows() -> list[dict]:
    """The table rows minus retired ones (revision r2-E1: the same filter as `core.ea_rows`)."""
    doc = yaml.safe_load((ROOT / "registry" / "explain_away.yaml").read_text(encoding="utf-8")) or {}
    return [r for r in doc.get("rows") or [] if str(r.get("status") or "") != "retired"]


def k_ea(spec_id: str, known: list[str], med_text: str | None = None) -> str:
    """K-EA reading (profile only, spec 4.4): `by_known_condition` when a table row lists this spec
    and its explainer is one of the case's known conditions, `by_visible_line` when a row lists it
    with a visible-line explainer, else `none`. A row whose mechanism needs a drug counts only when
    that drug is in the visible medication text (`core.drug_visible`, revision r2-E2)."""
    from haenv.m2.core import drug_visible
    hit = "none"
    for r in _explain_away_rows():
        if spec_id not in (r.get("specs") or ()) or not drug_visible(r, med_text):
            continue
        if any(k in known for k in (r.get("explainer_known") or ())):
            return "by_known_condition"
        if r.get("explainer_line"):
            hit = "by_visible_line"
    return hit


#: Explainer base conditions (spec 3.3, `background_comorbidity.yaml:explainer_bases`).
NEW_BASES = ("hypothyroidism", "CAD", "CKD")
#: Youngest age band an attached base may sit on: CKD is rare under 40 (CDC CKD surveillance,
#: NHANES 2017-2020, 6 % at 18-44 vs 12 % at 45-64, `background_comorbidity.yaml:age_prevalence`); a base the
#: age band makes implausible would let the age read the explainer. A spec whose own line is CKD never gets
#: the CKD base (BASE-1), so the floor applies to every case that can carry it.
BASE_MIN_AGE = {"CKD": 40}
#: A base whose treatment (`med_course.CONCURRENT`) is on the pregnancy-capable avoid list is not attached to
#: such a woman (`demographics.avoid_for`): the base is optional, so the drug is not required.


def base_allowed(base: str, raw: dict) -> bool:
    import haenv.world.generation.demographics as D
    from haenv.world.clinical.med_course import CONCURRENT
    age = str(raw.get("age_range") or "")
    lo = int(age.split("-")[0]) if age[:2].isdigit() else 0
    if lo < BASE_MIN_AGE.get(base, 0):
        return False
    return not any(D.avoid_for(d, str(raw.get("sex") or ""), age) for d, _ in CONCURRENT.get(base, ()))


#: A base never equals a hidden line (audit BASE-1): the hidden components it would duplicate.
BASE_HIDDEN = {"hypothyroidism": {"JD-HYPO"}, "CAD": {"JD-CAD"}, "CKD": {"JD-CKD", "JD-CKM"}}


def _u(*path) -> float:
    h = hashlib.sha256("|".join(str(x) for x in ("m2-base",) + path).encode("utf-8")).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def base_candidates(spec_id: str) -> list[str]:
    """New bases the explain-away table lets explain a hidden finding of this spec, table order,
    minus any base that equals one of the spec's hidden components."""
    from haenv.m2.core import components
    comps = set(components(spec_id)) | {spec_id}
    out: list[str] = []
    for r in _explain_away_rows():
        if spec_id not in (r.get("specs") or ()):
            continue
        for b in r.get("explainer_known") or ():
            if b in NEW_BASES and b not in out and not (BASE_HIDDEN[b] & comps):
                out.append(b)
    return out


def assign_bases(rows: list[dict], by_id: dict) -> dict[str, str]:
    """case_id -> attached base. Whether an F1-F3 case carries a base, and which, is independent of its
    hidden line and arm: per frame and arm, round(attach rate x arm size) cases in case-keyed order each
    draw a base from the frame's mix, where the attach rate and the mix are the share and the draws of the
    frame's positive-arm cases the explain-away table offers a base (the whole pool's when the frame
    offers none). A base that is a hidden component of the case or already on its record is skipped
    (BASE-1), and so is a base `base_allowed` refuses (age floor, pregnancy-capable avoid list); the next
    case of the same arm takes the slot (a positive arm always fills its quota: each offering case can take
    its own offer). The explain-away rendering follows only when the drawn base explains a hidden finding.
    F0 (bridge) never gets one.

    The quota is per arm (W7): with one quota per frame, the skips fell on the positive arm (a negative
    arm has no hidden component to duplicate) and their slots went to whichever case came next, so a base
    was attached 0.05-0.15 less often on the positive arm (Fisher p < 0.05 on some seeds)."""
    from haenv.m2.core import components
    pos: dict[str, list[str]] = {}
    every: dict[str, list[str]] = {}
    for r in rows:
        if r["fw"] == "F0":
            continue
        every.setdefault(r["fw"], []).append(r["case"])
        if not r["frame"].endswith("-neg"):
            pos.setdefault(r["fw"], []).append(r["case"])

    def known(cid):
        return [by_id[cid]["raw"].get("disease")] + list(by_id[cid]["raw"].get("comorbidities") or [])

    offer: dict[str, list[str]] = {}
    for fw in sorted(pos):
        for cid in sorted(pos[fw]):
            c = [b for b in base_candidates(by_id[cid]["latent"].get("ddx_spec_id", "")) if b not in known(cid)
                 and base_allowed(b, by_id[cid]["raw"])]
            if c:
                offer.setdefault(fw, []).append(c[int(_u(cid, "pick") * len(c))])
    n_pos = sum(len(v) for v in pos.values())
    mix_all = sorted(b for v in offer.values() for b in v)
    out: dict[str, str] = {}
    if not mix_all:
        return out
    for fw in sorted(every):
        p_fw = pos.get(fw) or []
        rate = len(offer.get(fw) or ()) / len(p_fw) if p_fw else len(mix_all) / n_pos
        mix = sorted(offer.get(fw) or ()) or mix_all
        p_set = set(p_fw)
        for arm in ([c for c in every[fw] if c in p_set], [c for c in every[fw] if c not in p_set]):
            k, got = int(rate * len(arm) + 0.5), 0
            for cid in sorted(arm, key=lambda c: _u(c, "attach")):
                if got >= k:
                    break
                spec = by_id[cid]["latent"].get("ddx_spec_id", "")
                comps = set(components(spec)) | {spec}
                pick = [b for b in mix if b not in known(cid) and not (BASE_HIDDEN[b] & comps)
                        and base_allowed(b, by_id[cid]["raw"])]
                if pick:
                    out[cid] = pick[int(_u(cid, "which") * len(pick))]
                    got += 1
    return out


#: Difficulty prior target (spec 3.4): easy : mid : hard = 25 : 45 : 30 of the whole pack. Only
#: F1-F3 knobs move (F0 is the byte-identical bridge); K-NT is a property of the spec, so the
#: lever is K-DD (`distractor_level`, the benign-distractor density knob).
TIER_TARGET = (("easy", 0.25), ("mid", 0.45), ("hard", 0.30))


def rebalance_kdd(rows: list[dict], by_id: dict) -> dict[str, str]:
    """case_id -> distractor_level for F1-F3 so the pack's prior tiers meet `TIER_TARGET` as
    closely as K-NT allows. Hard needs K-NT 6-8 with high, easy K-NT <= 5 with low; the cases that
    flip are chosen by a case-keyed order within each K-NT group (never by arm or score)."""
    n = len(rows)
    want = {t: round(n * w) for t, w in TIER_TARGET}
    want["mid"] = n - want["easy"] - want["hard"]
    f0 = [r for r in rows if r["fw"] == "F0"]
    have = {"easy": 0, "mid": 0, "hard": 0}
    for r in f0:
        have[prior_of(by_id[r["case"]])["tier"]] += 1
    fr = [r["case"] for r in rows if r["fw"] != "F0"]
    nt_hard = sorted((c for c in fr if len(by_id[c]["latent"].get("ddx_tests") or []) >= K_NT_HARD_MIN),
                     key=lambda c: _u(c, "kdd"))
    nt_easy = sorted((c for c in fr if c not in nt_hard), key=lambda c: _u(c, "kdd"))
    n_hard = min(len(nt_hard), max(0, want["hard"] - have["hard"]))
    n_easy = min(len(nt_easy), max(0, want["easy"] - have["easy"]))
    out = {}
    for i, c in enumerate(nt_hard):
        out[c] = "high" if i < n_hard else "low"          # low on K-NT 6-8 -> mid
    for i, c in enumerate(nt_easy):
        out[c] = "low" if i < n_easy else "high"          # high on K-NT <= 5 -> mid
    return out


#: Benign complaints over the course of an F1-F3 case: one case-keyed draw from 10..18 in every tier.
#: The lower end is the distractor pool (`core/noise._DISTRACTOR_SYMPTOMS`), which a high-tier case
#: injects whole; the planned benign events top the record up to the drawn count.
COMPLAINTS = (10, 18)


def complaint_count(case_id: str) -> int:
    lo, hi = COMPLAINTS
    return lo + int(_u(case_id, "complaints") * (hi - lo + 1))


def declare_complaint_count(latent: dict, n: int) -> dict:
    """Set the declared benign-symptom rate so the case's record carries `n` benign complaints
    (`events.expected_event_counts`). Mutates and returns `latent`."""
    from haenv.world.observations.events import event_weeks
    from haenv.world.observations.events import expected_event_counts
    ed = latent["event_density"] = dict(latent["event_density"])
    T = latent["index_time_T"]
    rate = round(n / event_weeks(T), 4)
    while (got := expected_event_counts({**ed, "symptom_rate": rate}, T)["symptom_rate"]) != n:
        rate = round(rate + (0.0001 if got < n else -0.0001), 4)
    ed["symptom_rate"] = rate
    return latent


def declare_life_event_rate(latent: dict, pool: list[float], case_id: str) -> dict:
    """The case's life-event rate (the context notes of its record: a trip, a new appliance) drawn from the
    F1-F3 pool's own rates by case id alone. The rates the source rows carry are higher on the positive arm
    (mean 0.121 against 0.079 on the negative arm in the 1.3.0 pool), which put about one more context note on
    every positive record; the resample keeps the pool's rate distribution and drops the arm from it."""
    ed = latent["event_density"] = dict(latent["event_density"])
    ed["life_event_rate"] = pool[int(_u(case_id, "life-rate") * len(pool))]
    return latent


def f3_plans(rows: list[dict], by_id: dict) -> dict[str, dict]:
    """case_id -> F3 plan (decoupled from the arm): within each arm, half of the F3 cases
    with a live lab stream follow a change beyond the RCV and the rest a within-RCV repeat. On the
    positive arm the change is on a marker the hidden line declares (else the named disease's own); on
    the negative arm it is a real improvement of the named disease's marker (`core.f3_plan_for`), which
    asks for no clinician action. So "exceeds the RCV" does not read the arm."""
    import haenv  # noqa: F401  (kernel path)
    from haenv.world.generation.build import clinical_plan
    from haenv.m2.core import exceed_streams, f3_plan_for
    f3 = [r for r in rows if r["fw"] == "F3"]
    live, arm = {}, {}
    for r in f3:
        raw = by_id[r["case"]]["raw"]
        live[r["case"]] = list(clinical_plan(raw["disease"], raw["devices"], raw.get("comorbidities") or [], r["case"]))
        arm[r["case"]] = "negative" if r["frame"].endswith("-neg") else "positive"
    have_stream = [c for c in live if any(st in live[c] for st in ("HbA1c", "fasting_glucose", "LDL",
                                                                    "triglycerides", "ALT", "AST"))]

    def plan(c, ex):
        return f3_plan_for(c, by_id[c]["latent"].get("ddx_spec_id", ""), arm[c], live[c], ex,
                           by_id[c]["raw"].get("disease", ""))
    ex: set = set()
    for a in ("positive", "negative"):
        grp = [c for c in have_stream if arm[c] == a]
        target = len(grp) // 2 + len(grp) % 2

        def ok(c):
            return (a == "negative" or by_id[c]["latent"].get("ddx_join_gold") != "independent") and plan(c, True) is not None
        # positive arm: declared hidden markers first, then the named disease's own markers
        decl = [c for c in grp if ok(c) and a == "positive"
                and exceed_streams(by_id[c]["latent"].get("ddx_spec_id", ""), live[c])]
        rest = [c for c in grp if ok(c) and c not in decl]
        cand = sorted(decl, key=lambda c: _u(c, "f3-arm")) + sorted(rest, key=lambda c: _u(c, "f3-arm"))
        ex |= set(cand[:target])
    out = {}
    for c in have_stream:
        p = plan(c, c in ex)
        if p:
            out[c] = p
    return out


def prior_of(case: dict, med_text: str | None = None, mode: str = "legacy") -> dict:
    lat = case["latent"]
    n_tests = len(lat.get("ddx_tests") or [])
    k_dd = "high" if lat.get("distractor_level") == "high" else "low"
    k_nt = (">=6" if mode == "expanded" else "6-8") if n_tests >= K_NT_HARD_MIN else "<=5"
    tier = "hard" if (k_dd == "high" and n_tests >= K_NT_HARD_MIN) else (
        "easy" if (k_dd == "low" and k_nt == "<=5") else "mid")
    # The conditions the record names (`user_profile.known_conditions`): the primary disease and the
    # M2 explainer bases; other comorbidities are not named, so they explain nothing to a reader.
    known = [case["raw"].get("disease")] + [x for x in case["raw"].get("comorbidities") or [] if x in NEW_BASES]
    return {"k_dd": k_dd, "k_nt": k_nt, "k_ea": k_ea(lat.get("ddx_spec_id", ""), known, med_text),
            "tier": tier, "n_tests": n_tests}


def _load(name: str, path: pathlib.Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recompose():
    return _load("recompose_v3", RECOMPOSE)


def bridge_supply(rows: list[dict], by_id: dict) -> dict:
    """Bridge cell -> sufficient-tier bridge cases of that prior tier. A bridge case whose symptom text
    names an age outside its sampled age range (`age_conflict`) is not offered: the bridge stays
    byte-identical to v1.0.1, so such a case is left out rather than re-worded."""
    out = {k: 0 for k in PACK.BRIDGE}
    for r in rows:
        c = by_id[r["case"]]
        if r["fw"] == "F0" and not c["latent"].get("ddx_insufficient") and not age_conflict(c):
            k = f"f0_{prior_of(c)['tier']}"
            if k in out:
                out[k] += 1
    return out


def allocation(n: int, seed: str = PS.PUBLIC_SEED, mode: str = "auto") -> tuple[list[dict], dict, dict, dict]:
    """(pool rows, source cases, pack cell sizes, bridge shortfall) for a pack of n items. The F1-F3
    rows are the composition table's; a non-public seed re-draws their patients from its variant
    window (`recompose_v3.instantiate`), the bridge stays the frozen v1.0.1 set."""
    src = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in src["cases"]}
    rc = _recompose()
    f0 = rc.bridge_rows()
    from haenv.world.overlay import condition_registry
    mode = PACK.allocation_mode(condition_registry(), mode)
    quota = (lambda size, supply: PACK.expanded_quotas(size, supply, diversity=True, confirmation=True)) if mode == "expanded" else PACK.quotas
    q, short = quota(n, bridge_supply(f0, by_id))
    # the pool is composed for `PS.pool_n(n)`, so the packs of the nested sizes share it (`haenv/world/validation/pack_size.py`)
    m = PS.pool_n(n)
    qm, _ = quota(m, bridge_supply(f0, by_id))
    nb = m - sum(qm[k] for k in PACK.BRIDGE)
    if mode == "expanded":
        from haenv.m2.confirmation import rules as confirmation_rules
        res = rc.allocate_expanded(m, seed, pool_multiplier=confirmation_rules()["allocation"]["candidate_supply_multiplier"]["value"])
        v3 = res["rows"]  # already drawn from this seed's patient window
    else:
        res = rc.allocate(max(nb / NB_PER_SCALE, POOL_MARGIN * POOL_PER_ITEM * m / POOL_PER_SCALE))
        v3 = rc.instantiate(res["rows"], PS.variant_base(seed), by_id)
    gen = rc.generated_cases(v3)
    assert not set(gen) & set(by_id)
    by_id.update(gen)
    rows = f0 + [dict(r) for r in v3]
    assert len({r["case"] for r in rows}) == len(rows)
    assert not any(r["case"] in rc.CLINICAL_HOLD for r in rows)
    return rows, by_id, q, short


def _ea_render(r: dict, c: dict, med: str, explain: bool):
    known = [c["raw"].get("disease")] + [x for x in c["raw"].get("comorbidities") or [] if x in NEW_BASES]
    return explain_away_render(c["latent"].get("ddx_spec_id", ""), known, c["raw"].get("symptoms") or [],
                               T=c["latent"].get("index_time_T"), med_text=med, case_id=c["case_id"],
                               explain=explain, goals=tuple(c["raw"].get("goals") or ("sustained_weight_loss",)))


def ea_twins(eligible: dict[str, list[str]]) -> dict[str, bool]:
    """case_id -> explain-away on/off. Within each spec the eligible cases (those whose rendering
    removes a clause) alternate on/off in sha256(case_id) order, starting from the spec's own sha256
    parity: the same hidden condition, companion line and base appear both explained and unexplained."""
    out = {}
    for spec, cids in sorted(eligible.items()):
        start = int(hashlib.sha256(f"m2-ea|{spec}".encode()).hexdigest(), 16) % 2
        for k, cid in enumerate(sorted(cids, key=lambda x: hashlib.sha256(f"m2-ea|{x}".encode()).hexdigest())):
            out[cid] = (k + start) % 2 == 0
    return out


#: case id -> background comorbidities `compose` attached (provenance of the last compose)
BACKGROUND: dict[str, list[str]] = {}


def attach_background(rows: list[dict], by_id: dict) -> dict[str, list[str]]:
    """Background comorbidities (`registry/background_m2.yaml`, `haenv/m2/background.py`) are written to
    `raw.comorbidities` before the age band is drawn, so the band follows the conditions the way the build
    reads them; a condition whose drug is on the pregnancy-capable avoid list for the drawn age and sex leaves
    and the band is drawn again. Returns case id -> the conditions that stayed."""
    from haenv.m2.background import assign_background, drug_allowed
    plan = assign_background(rows, by_id)
    for cid, conds in plan.items():
        raw = dict(by_id[cid]["raw"])
        raw["comorbidities"] = list(raw.get("comorbidities") or []) + [c for c in conds if c not in (raw.get("comorbidities") or [])]
        by_id[cid] = {**by_id[cid], "raw": raw}
    out: dict[str, list[str]] = {}
    for r in rows:
        if r["fw"] == "F0":
            continue
        cid = r["case"]
        for _ in range(3):                      # leaving a condition redraws the band; at most a few passes settle
            by_id[cid] = with_pack_age(by_id[cid])
            raw = by_id[cid]["raw"]
            keep = [c for c in raw.get("comorbidities") or [] if c not in plan.get(cid, ()) or drug_allowed(c, raw)]
            if keep == list(raw.get("comorbidities") or []):
                break
            by_id[cid] = {**by_id[cid], "raw": {**raw, "comorbidities": keep}}
        else:
            raise ValueError(f"{cid}: background comorbidity and age band did not settle")
        stayed = [c for c in plan.get(cid, ()) if c in (by_id[cid]["raw"].get("comorbidities") or [])]
        if stayed:
            out[cid] = stayed
    return out


def compose(rows: list[dict], by_id: dict, mode: str = "legacy") -> tuple[list[dict], dict, dict, dict]:
    """(cases, prior table, explainer bases, K-DD changes) of the pool."""
    BACKGROUND.clear()
    for r in rows:
        if r["fw"] != "F0":
            by_id[r["case"]] = retime(by_id[r["case"]]) if BACKGROUND_ON[0] else with_pack_age(retime(by_id[r["case"]]))
    if BACKGROUND_ON[0]:
        BACKGROUND.update(attach_background(rows, by_id))
    src_by_id = dict(by_id)
    cases, prior = [], {}
    bases = assign_bases(rows, by_id)
    for cid, b in bases.items():
        raw = dict(by_id[cid]["raw"])
        raw["comorbidities"] = list(raw.get("comorbidities") or []) + [b]
        by_id[cid] = {**by_id[cid], "raw": raw}
    from haenv.world.generation.patient_sampling import benign_symptoms
    by_id = {cid: benign_symptoms(c) for cid, c in by_id.items()}
    # Expansion measures the actual test burden and distractor density. It does not
    # force a hard quota or alter a condition's medically declared test list.
    kdd = ({r["case"]: ("high" if _u(r["case"], "expanded-kdd") < 0.5 else "low")
            for r in rows if r["fw"] != "F0"} if mode == "expanded" else rebalance_kdd(rows, by_id))
    for cid, lvl in kdd.items():
        lat0 = copy.deepcopy(by_id[cid]["latent"])
        if lat0.get("distractor_level") != lvl:
            lat0["distractor_level"] = lvl
            if lvl == "high":
                from haenv.world.clinical.ddx import declare_high_distractor_density
                declare_high_distractor_density(lat0)
            by_id[cid] = {**by_id[cid], "latent": lat0}
    life_pool = sorted(float(by_id[r["case"]]["latent"]["event_density"]["life_event_rate"])
                       for r in rows if r["fw"] != "F0") if BACKGROUND_ON[0] else []
    for r in rows:
        if r["fw"] != "F0":
            lat0 = copy.deepcopy(by_id[r["case"]]["latent"])
            declare_complaint_count(lat0, complaint_count(r["case"]))
            if BACKGROUND_ON[0]:
                declare_life_event_rate(lat0, life_pool, r["case"])
            by_id[r["case"]] = {**by_id[r["case"]], "latent": lat0}
    f3 = f3_plans(rows, by_id)
    # F1 day-T complaint: a registered acute presentation (`registry/acute_events.yaml`) or the latest
    # true symptom; the negative arm takes a registered complaint with the positive arm's share, so the
    # complaint's form does not read the arm.
    arm_of = {r["case"]: ("negative" if str(r.get("frame") or "").endswith("-neg") else "positive") for r in rows}
    f1p = [r["case"] for r in rows if r["fw"] == "F1" and arm_of[r["case"]] == "positive"]
    acute = {c: acute_event_for(c, by_id[c]["latent"].get("ddx_spec_id", ""), "positive") for c in f1p}
    share = (sum(1 for v in acute.values() if v) / len(f1p)) if f1p else 1.0
    for r in rows:
        if r["fw"] == "F1" and arm_of[r["case"]] == "negative":
            acute[r["case"]] = acute_event_for(r["case"], "", "negative") if _u(r["case"], "acute-form") < share else None
    med_of, eligible, twin_lines = {}, {}, {}
    for r in rows:
        if r["fw"] == "F0":
            continue
        c = by_id[r["case"]]
        ev = acute.get(r["case"])
        # Visible medication text: the frame scene (F2 names the primary drug) and the symptom texts;
        # a drug-dependent explain-away row licenses only on a drug named here.
        med_of[r["case"]] = visible_medication_text(
            scene_text(r["fw"], c["raw"], {**c["latent"], "m2": {"frame": r["fw"], **({"acute_event": ev} if ev else {})}},
                       c["case_id"]),
            c["raw"].get("symptoms") or [])
        on = _ea_render(r, c, med_of[r["case"]], True)[1]["lines"] if arm_of[r["case"]] == "positive" else []
        if on:
            eligible.setdefault(c["latent"].get("ddx_spec_id", ""), []).append(r["case"])
            twin_lines[r["case"]] = sorted(ln["component"] for ln in on)
    twin = ea_twins(eligible)
    for r in sorted(rows, key=lambda r: (FRAMES.index(r["fw"]), r["case"])):
        c = by_id[r["case"]]
        lat = dict(c["latent"])
        if r["fw"] == "F0":
            lat.update(bridge_triage(c))
        arm = arm_of[r["case"]]
        ev = acute.get(r["case"]) if r["fw"] == "F1" else None
        med = "" if r["fw"] == "F0" else med_of[r["case"]]
        p = prior_of(c, med, mode if r["fw"] != "F0" else "legacy")
        if r["fw"] != "F0":
            # Explain-away rendering (profile only): on the twin's "on" side a visible explainer covers
            # at most `core.EA_MAX_REMOVED` clauses of the hidden line; age tokens leave the templates.
            sy, ea = _ea_render(r, c, med, twin.get(r["case"], False))
            if not ea["lines"]:
                p = {**p, "k_ea": "none"}
            m2 = {"frame": r["fw"], "arm": arm,
                  "prior": {k: p[k] for k in ("k_dd", "k_nt", "k_ea", "tier")}}
            if r["case"] in twin:
                m2["ea_twin"] = "explained" if twin[r["case"]] else "unexplained"
                m2["ea_twin_lines"] = twin_lines[r["case"]]
            if r["fw"] == "F3" and r["case"] in f3:
                m2["f3"] = f3[r["case"]]
            if BACKGROUND_ON[0]:
                m2["background"] = True      # the hooks of `haenv/m2/background.py` act on this case only
            if r["fw"] == "F1" and ev:
                m2["acute_event"] = ev
            if sy != (c["raw"].get("symptoms") or []):
                c = {**c, "raw": {**c["raw"], "symptoms": sy}}
            if ea["lines"] or ea.get("clues"):
                m2["explain_away"] = ea
            m2["scene"] = scene_text(r["fw"], c["raw"], {**c["latent"], "m2": m2}, c["case_id"])
            lat["m2"] = m2
        else:
            lat["m2_bridge"] = True          # F0: the M2 menu and purchase readings, no frame (`core.is_m2_case`)
        from haenv.m2.core import workup_tests
        lat["ddx_tests"] = [] if lat.get("ddx_insufficient") else workup_tests(lat["ddx_spec_id"])
        prior[r["case"]] = {"frame": r["fw"], **p}
        cases.append({"case_id": c["case_id"], "raw": c["raw"], "latent": lat})
        if mode == "expanded":
            from haenv.m2.diversity import prepare as diversify
            diversify(cases[-1])
    kdd_changed = {c: v for c, v in sorted(kdd.items()) if v != src_by_id[c]["latent"].get("distractor_level")}
    return cases, prior, bases, kdd_changed


def bridge_triage(c: dict) -> dict:
    """Urgency and red flag of a bridge (F0) case from the current condition registry, by the rule the
    F1-F3 cases are generated with (`ddx.insufficient_triage`): a corrected spec reaches the bridge too."""
    from haenv.world.clinical.ddx import insufficient_triage
    from haenv.world.overlay import condition_registry
    lat = c["latent"]
    spec = condition_registry().get(lat.get("ddx_spec_id"))
    if not spec or "ddx_urgency" not in lat:
        return {}
    rf, urg = insufficient_triage(spec, [int(s["day"]) for s in c["raw"].get("symptoms") or ()],
                                  int(lat["index_time_T"]), bool(lat.get("ddx_insufficient")))
    return {"ddx_urgency": urg, "ddx_red_flag": rf}


JOB_KEYS = {"task_type": "joint_dx", "multiround": False, "include_baseline": True, "models": [],
            "sample_cases": 5, "gated": True, "plugins": {"judges": ["haenv.m2"]}}


def _dump(job: dict, comment: str) -> str:
    canary = CANARY.block().splitlines()
    buf = io.StringIO()
    buf.write("\n".join(canary) + "\n")
    buf.write(comment)
    yaml.safe_dump(job, buf, allow_unicode=True, sort_keys=False, width=120)
    return buf.getvalue()


def emitted(pool_text: str, mode: str = "legacy") -> tuple[set[str], list[str], dict]:
    """(case ids the deterministic build emits from the pool job, emitted F1-F3 cases with a gold line
    that shows no separating clue on the record). A case of the second kind has no evidence that tells
    its line from the nearest look-alike (the EA-2 check of `tools/validation/m2_item_audit.py`), so it is not
    selected. The third value: case_id -> the F2 reading on the built record (`core.f2_direction`:
    safety floor, own direction, hidden line linked), every F2 case. A case with a gold test no menu item makes purchasable is not emitted either
    (`unbuyable_gold`); `GATE["unbuyable_gold"]` receives those cases and their tests."""
    from haenv.m2.core import critical_reading, f2_direction, is_m2_case
    with tempfile.TemporaryDirectory(prefix="m2-pool-") as td:
        job = pathlib.Path(td) / "m2-pool.job.yaml"
        job.write_text(pool_text, encoding="utf-8")
        env = {**os.environ, "HAENV_OUTPUT_ROOT": td}
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "plugins"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        r = subprocess.run([sys.executable, "-m", "haenv", "build", str(job), "--gen", "deterministic", "--fresh",
                            "--gen-workers", os.environ.get("HAENV_GEN_WORKERS", "0"),
                            "--set", "env_file=/dev/null"], cwd=ROOT, env=env, capture_output=True, text=True)
        got = sorted(pathlib.Path(td, "results").glob("*/m2-pool/*/cases.jsonl"))
        # exit 4 = a batch-level gate on the pool; only the per-case emission is read here
        if r.returncode not in (0, 4) or not got:
            sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
            raise SystemExit(f"[m2_gen_job] pool build failed (exit {r.returncode})")
        built = [json.loads(ln) for ln in got[-1].read_text(encoding="utf-8").splitlines() if ln.strip()]
        from haenv.world.validation.difficulty_r2 import enabled
        if enabled():
            GATE["history_eligible"] = {b["case_id"] for b in built
                if (b["case"].get("adjudication", {}).get("m2_frame", {}).get("history_dependency") or {}).get("active")}
            GATE["score_branches"] = {b["case_id"]: score_branches(
                b["case"].get("adjudication", {}).get("m2_frame", {})) for b in built}
        ok = {b["case_id"] for b in built}
        unbuyable = unbuyable_gold(built)
        ok -= set(unbuyable)
        GATE["unbuyable_gold"] = unbuyable
        pj = {c["case_id"]: c for c in yaml.safe_load(pool_text)["cases"]}
        if mode == "expanded":
            from haenv.m2.core import f3_followup
            # A plan alone is insufficient: check the actual built record has a
            # prior observation from which today's repeat can be calculated.
            for b in built:
                c = pj[b["case_id"]]
                m2 = c["latent"].get("m2") or {}
                if m2.get("frame") == "F3" and not f3_followup(
                    b["case"].get("longitudinal_data") or {}, m2.get("f3"), c["latent"]["index_time_T"]):
                    ok.discard(b["case_id"])
        f2 = {}
        GATE["critical"] = {}
        for b in built:
            c = pj[b["case_id"]]
            m2 = c["latent"].get("m2") or {}
            if is_m2_case(_raw_of(b)):
                crit = critical_reading(_raw_of(b), int(c["latent"]["index_time_T"]))
                if crit:
                    GATE["critical"][b["case_id"]] = crit
            if m2.get("frame") == "F2":
                f2[b["case_id"]] = f2_direction(
                    _raw_of(b), c["raw"].get("disease"), int(c["latent"]["index_time_T"]), c["raw"].get("drug") or "",
                    c["latent"].get("ddx_spec_id") if m2.get("arm") == "positive" else None)
        import haenv.m2.item_audit as IA
        ea2 = IA.check_ea2(IA.collect(got[-1].parent, job))["violations"]
        return ok, sorted({v[0] for v in ea2}), f2


#: Written by `emitted` for the build that follows it: the emission gates' reports and the critical
#: values the built records show on day T (`core.critical_reading`).
GATE: dict = {}


def apply_critical(c: dict, crit: dict | None) -> bool:
    """A case whose record shows a critical value on day T (`core.critical_reading`) takes its urgency
    and a red flag, F1-F3 only (an F0 bridge case keeps the registry triage the bridge gate checks); the
    reading is recorded in latent `m2.critical`, which `wq.derive_from_registry` reads for the red flag.
    Never lowers the urgency. True when the case carries a critical value."""
    lat = c["latent"]
    if not crit or not (lat.get("m2") or {}).get("frame") in ("F1", "F2", "F3"):
        return False
    order = ("🟢", "🟡", "🟠", "🔴")
    lat["ddx_urgency"] = max(crit["urgency"], str(lat.get("ddx_urgency") or "🟢"), key=order.index)
    lat["ddx_red_flag"] = True
    lat["m2"]["critical"] = crit
    return True


def _raw_of(b: dict):
    """The built case as the purchase path reads it."""
    from types import SimpleNamespace
    c = b["case"]
    return SimpleNamespace(case_id=b["case_id"], user_profile=c.get("user_profile") or {},
                           latent_premise=c.get("latent_premise") or {}, adjudication=c.get("adjudication") or {},
                           longitudinal_data=c.get("longitudinal_data") or {},
                           prediction_context=c.get("prediction_context") or {})


def unbuyable_gold(built: list[dict]) -> dict[str, list[str]]:
    """M2 built cases (F0-F3) with a gold test no menu item makes purchasable (no item named for it is
    credited, or every such item reads nothing for the case), or a lab result the visible record states
    that the day-T purchase contradicts (`core.lab_claim_conflicts`): case_id -> ["<item> (<reason>)", ...].
    Such a case is not emitted. Read on the built record, so a gold test another registry edit adds is checked as well."""
    from types import SimpleNamespace
    from haenv.m2.core import implausible_readings, is_m2_case, lab_claim_conflicts, unbuyable_gold_tests
    out = {}
    for b in built:
        c = b["case"]
        raw = SimpleNamespace(case_id=b["case_id"], user_profile=c.get("user_profile") or {},
                              latent_premise=c.get("latent_premise") or {}, adjudication=c.get("adjudication") or {},
                              longitudinal_data=c.get("longitudinal_data") or {},
                              prediction_context=c.get("prediction_context") or {},
                              evidence_ledger=c.get("evidence_ledger") or [])
        if not is_m2_case(raw):
            continue
        bad = [f"{t} ({why})" for t, why in unbuyable_gold_tests(raw)]
        bad += [f"{x} (record states it, purchase contradicts)" for x in lab_claim_conflicts(raw)]
        bad += [f"{x} (below the plausible floor)" for x in implausible_readings(raw)]
        if bad:
            out[b["case_id"]] = bad
    return out


def no_lay_sentences(cases: list[dict]) -> list[str]:
    """F1-F3 cases with a symptom phrase the lay-sentence registry does not cover: the record would
    print that phrase whole (often a diagnosis label), so such a case is not selected."""
    import haenv.world.observations.events as E
    lay = E.symptom_lay()
    return sorted(c["case_id"] for c in cases if (c.get("latent") or {}).get("m2")
                  and any(sy.get("text") not in lay for sy in c["raw"].get("symptoms") or ()))


def build(n: int = PS.DEFAULT_N, seed: str = PS.PUBLIC_SEED, pool_out: pathlib.Path | None = None,
          mode: str = "auto") -> str:
    from haenv.world.overlay import condition_registry
    mode = PACK.allocation_mode(condition_registry(), mode)
    rows, by_id, q, short = allocation(n, seed, mode)
    cases, prior, bases, kdd = compose(rows, by_id, mode)
    if mode == "expanded":
        from haenv.m2.confirmation import prepare as prepare_confirmation
        cases = prepare_confirmation(cases, prior)
    confirmation_enabled = any((c["latent"].get("m2") or {}).get("confirmation") for c in cases)
    medication_ineligible = []
    if mode == "expanded":
        rc = _recompose()
        medication_ineligible = sorted(c["case_id"] for c in cases
                                      if c["latent"].get("m2") and not rc.medication_eligible(c))
        refused_medication = set(medication_ineligible)
        cases = [c for c in cases if c["case_id"] not in refused_medication]
        positive_counts = Counter(c["latent"].get("ddx_spec_id") for c in cases
                                  if (c["latent"].get("m2") or {}).get("arm") == "positive")
        deficits = {sid: {"required": need, "measured": positive_counts[sid],
                          "deficit": need - positive_counts[sid]}
                    for sid, need in PACK.expanded_required_counts(n).items() if positive_counts[sid] < need}
        if deficits:
            raise ValueError(f"expanded coverage shortfall after medication eligibility/retiming: deficits {deficits}")
    pool = {"job_id": "m2-pool", **JOB_KEYS, "report": "eval-m2-pool.md",
            "_provenance": {"m2_prior": prior}, "cases": cases}
    pool_text = _dump(pool, "# pool job (intermediate)\n")
    GATE.clear()
    ok, no_clue, f2 = emitted(pool_text, mode) if mode == "expanded" else emitted(pool_text)
    critical = {}
    for c in cases:
        if f2.get(c["case_id"]):
            c["latent"]["m2"]["f2"] = f2[c["case_id"]]
        if apply_critical(c, (GATE.get("critical") or {}).get(c["case_id"])):
            critical[c["case_id"]] = c["latent"]["m2"]["critical"]
    if pool_out is not None:
        pool_out.write_text(_dump(pool, "# m2 pool job (intermediate; the pack is selected from it)\n"), encoding="utf-8")
    refused = sorted(c["case_id"] for c in cases if c["case_id"] not in ok)
    unlay = no_lay_sentences(cases)
    aged = sorted(c["case_id"] for c in cases if age_conflict(c))
    # An F3 item asks to read a follow-up result against the last one; a case without an F3 plan has no
    # follow-up result and would print `followed_indicator: null`, so it is not selectable.
    no_f3 = sorted(c["case_id"] for c in cases
                   if (c["latent"].get("m2") or {}).get("frame") == "F3" and not c["latent"]["m2"].get("f3"))
    # An F2 negative-arm item is scored on the medication direction its primary disease's control
    # target gives (`core.f2_direction`); without a target or a reading there is no gold, so not selectable.
    no_f2 = sorted(c["case_id"] for c in cases
                   if (c["latent"].get("m2") or {}).get("frame") == "F2" and c["latent"]["m2"].get("arm") == "negative"
                   and not (c["latent"]["m2"].get("f2") or {}).get("direction"))
    S = _load("pack1_select", SELECT)
    prow = [r for r in S.rows_of(pool) if r["case"] in ok and r["case"] not in unlay and r["case"] not in no_clue
            and r["case"] not in aged and r["case"] not in no_f3 and r["case"] not in no_f2]
    if mode == "expanded":
        rc = _recompose()
        policy = rc.benchmark_candidates()
        from haenv.m2.core import components
        reg = condition_registry()
        for r in prow:
            meta = policy.get(r["spec"])
            if meta is None:  # frozen bridge still needs a measured clinical group
                specialties = {s for comp in components(r["spec"]) for s in reg[comp]["specialty"]}
                endocrine = any("内分泌" in s for s in specialties)
                meta = {"system": "endocrine" if endocrine else "/".join(sorted(specialties)),
                        "outside_endocrine": not endocrine, "expanded": False}
            r.update({k: meta[k] for k in ("system", "outside_endocrine", "expanded")})
    supply = bridge_supply([r for r in rows if r["fw"] == "F0"], by_id)
    if mode == "expanded":
        # N50 and N100 use the same look-ahead over their shared pool, so the
        # separately generated N50 is exactly the base chosen for N100.
        sizes = sorted(set(PS.nest_chain(PS.pool_n(n)) + [n]))
        diversified = any((c["latent"].get("m2") or {}).get("diversity")
                          for c in cases)
        nested_q = {k: q if k == n else PACK.expanded_quotas(k, supply, diversity=diversified,
                                                          confirmation=confirmation_enabled)[0] for k in sizes}
        case_map = {c["case_id"]: c for c in cases}
        available = {r["case"] for r in prow}
        prow = [r for r in prow if not (case_map[r["case"]]["latent"].get("m2") or {}).get("confirmation")
                or set(case_map[r["case"]]["latent"]["m2"]["confirmation"]["pair"]) <= available]
        for row in prow:
            row["confirmation"] = (case_map[row["case"]]["latent"].get("m2") or {}).get("confirmation")
        if confirmation_enabled:
            from haenv.m2.confirmation import rules as confirmation_rules
            # Paired diagnoses are represented by their independent patients;
            # the frozen bridge must not consume a unit of that same spec cap.
            prow = [r for r in prow if r["frame"] != "F0" or r["spec"] not in confirmation_rules()["profiles"]]
        from haenv.m2.history_r2 import history_slots
        history_specs = sorted({r["spec"] for r in prow if r["case"] in GATE.get("history_eligible", set())})
        original = (confirmation_reference(S, prow, nested_q, seed) if confirmation_enabled else
                    S.select_expanded_nested(prow, nested_q, seed, required_specs=PACK.EXPANSION_IDS))
        if history_specs:
            acting_ids = GATE["history_eligible"]
            for row in prow:
                row["history_eligible"] = row["case"] in acting_ids
                row["score_branches"] = GATE["score_branches"].get(row["case"], [])
            pick = paired_strata(original, prow, seed, history_counts={
                k: history_slots(k, diversity=diversified) for k in sizes}, diversity=diversified)[n]
        else:
            pick = original[n]
    else:
        pick = None
        for k in PS.nest_chain(n):
            pick = S.select(prow, q if k == n else PACK.quotas(k, supply)[0], seed, base=pick)
    bal = S.balance(pick, prow, mode) if mode == "expanded" else S.balance(pick, prow)
    if not bal["pass"]:
        raise SystemExit(f"[m2_gen_job] balance gate failed: {json.dumps(bal, ensure_ascii=False)}")
    ids = {r["case"] for r in pick}
    # The job carries the selectable pool: the chosen bridge cases and every selectable F1-F3 case; the
    # audit's shortcut gates read the pool and its `select` publishes the N cases of `pack_ids`.
    # Retain the eligible bridge supply for deterministic selection on paired
    # worlds. The published bridge quota and per-spec cap remain unchanged.
    keep = {r["case"] for r in prow if r["frame"] != "F0" or not r["insufficient"]}
    n_nb = sum(1 for r in pick if r["frame"] != "F0")
    n_pool = sum(prior[cid]["frame"] != "F0" for cid in keep)
    if n_pool < POOL_PER_ITEM * n_nb:
        raise SystemExit(f"[pack] m2 pool too small -- {n_pool} selectable F1-F3 cases, need >= {POOL_PER_ITEM} x {n_nb}")
    plan: dict = {}
    for r in pick:
        if r["frame"] != "F0":
            plan.setdefault(r["frame"], {}).setdefault(r["tier"], 0)
            plan[r["frame"]][r["tier"]] += 1
    rc = _recompose()
    job = {
        "job_id": "m2-pack1", **JOB_KEYS, "report": "eval-m2-pack1.md",
        "pack": PS.header(n, seed),
        "_provenance": {
            "generator": 'tools/generation/m2_gen_job.py', "source_job": 'benchmarks/inputs/ddx-workup.job.yaml',
            "bridge_set": "tools/generation/m2/alloc_cases.csv",
            "bridge_set_sha16": hashlib.sha256(ALLOC.read_bytes()).hexdigest()[:16],
            "design": dict(q), "bridge_shortfall": dict(short),
            "plan_cells": {f: dict(sorted(t.items())) for f, t in sorted(plan.items())},
            "pool": {"n": len(cases), "n_f1_f3": sum(1 for c in cases if prior[c["case_id"]]["frame"] != "F0"),
                     "refused_by_build": refused, "no_lay_sentences": unlay, "no_separating_clue": no_clue,
                     "unbuyable_gold": GATE.get("unbuyable_gold") or {},
                     "bridge_age_conflict": aged,
                     "no_f3_plan": no_f3, "no_f2_gold": no_f2,
                     "critical_triage": {k: v for k, v in sorted(critical.items()) if k in keep},
                     "critical_f0_kept": sorted(k for k in (GATE.get("critical") or {}) if k not in critical and k in ids),
                     "generated_cases": sorted(r["case"] for r in rows if r.get("source") == "generated")},
            "withheld": list(rc.CLINICAL_HOLD),
            "pack_ids": sorted(ids), "pool_f1_f3": n_pool,
            "explainer_bases": {c: b for c, b in sorted(bases.items()) if c in keep},
            "spec_cap": bal["spec_cap"], "ea_twins": bal["ea_twins"],
            "kdd_rebalanced": {c: v for c, v in kdd.items() if c in keep},
            "m2_prior": {c: p for c, p in prior.items() if c in keep}},
        "cases": [c for c in cases if c["case_id"] in keep],
    }
    if mode == "expanded":
        if confirmation_enabled:
            from haenv.m2.confirmation import rules as confirmation_rules
            job["_provenance"]["confirmation"] = {"version": confirmation_rules()["version"],
                "registry": "registry/m2_confirmation.yaml", "allocation": confirmation_rules()["allocation"]}
        if history_specs:
            job["_provenance"]["history_eligible_pack_ids"] = sorted(GATE["history_eligible"])
            job["_provenance"]["score_branches"] = GATE["score_branches"]
        job["_provenance"]["allocation_mode"] = mode
        if diversified:
            from haenv.m2.diversity import rules
            job["_provenance"]["diversity"] = {"registry": "registry/m2_diversity.yaml",
                                                   "version": rules()["version"],
                                                   "allocation": rules()["allocation"]}
        job["_provenance"]["expansion_coverage"] = S.coverage_report(
            prow, pick, q, PACK.EXPANSION_IDS, required_counts=PACK.expanded_required_counts(n))
        job["_provenance"]["expansion_coverage"]["measured"]["generated_pool_by_spec"] = dict(
            sorted(Counter(r["spec_id"] for r in rows if r["fw"] != "F0").items()))
        job["_provenance"]["expansion_coverage"]["measured"]["job_pool_n"] = len(keep)
        job["_provenance"]["pool"]["medication_ineligible_after_retime"] = medication_ineligible
        job["_provenance"]["difficulty_measure"] = (
            "test-list burden x benign distractor density: hard >=6 declared tests and high density; "
            "easy <=5 and low density; mid otherwise. Profile only, no difficulty quota or test padding.")
    if BACKGROUND_ON[0]:
        job["_provenance"]["background"] = {"registry": "registry/background_m2.yaml",
                                            "comorbidities": {c: v for c, v in sorted(BACKGROUND.items()) if c in keep}}
    nb = len(ids) - n_nb
    return _dump(job, "# Generated file -- do not edit by hand. Rebuild: python tools/generation/m2_gen_job.py --n N [--seed S]\n#\n"
                 f"# Pack 1 (differential diagnosis): {n} published cases (`_provenance.pack_ids`) of a pool of {len(keep)}, four frames -- F0 weight-regain bridge ({nb}, copied\n"
                 "# from ddx-workup.job.yaml with the M2 menu, latent `m2_bridge`), F1 acute triage, F2 chronic medication\n"
                 "# adjustment, F3 follow-up interpretation. F1-F3 add one latent key `m2` (knob). The plugin group\n"
                 "# `haenv.m2` (haenv/m2/plugin.py) registers the frame and revision gold blocks.\n"
                 "#\n# SYNTHETIC data, evaluation only, not medical advice.\n")


def reselect(job: dict, eligible: list[str]) -> list[str]:
    """Use the existing selector on the common supply of paired worlds."""
    from haenv.world.overlay import condition_registry
    from haenv.m2.core import components
    S = _load("pack1_select", SELECT)
    allowed = set(eligible)
    rows = S.rows_of(job)
    n, seed, prov = int(job["pack"]["n_items"]), PS.cli_seed(None), job["_provenance"]
    if prov.get("allocation_mode") == "expanded":
        policy, reg = _recompose().benchmark_candidates(), condition_registry()
        for r in rows:
            meta = policy.get(r["spec"])
            if meta is None:
                specialties = {s for comp in components(r["spec"]) for s in reg[comp]["specialty"]}
                endocrine = any("内分泌" in s for s in specialties)
                meta = {"system": "endocrine" if endocrine else "/".join(sorted(specialties)),
                        "outside_endocrine": not endocrine, "expanded": False}
            r.update({k: meta[k] for k in ("system", "outside_endocrine", "expanded")})
        supply = Counter(f"f0_{r['tier']}" for r in rows if r["frame"] == "F0")
        diversified = bool(prov.get("diversity"))
        qs = {k: prov["design"] if k == n else PACK.expanded_quotas(k, supply, diversity=diversified,
                                                               confirmation=bool(prov.get("confirmation")))[0]
              for k in sorted(set(PS.nest_chain(PS.pool_n(n)) + [n]))}
        from haenv.m2.history_r2 import history_slots
        acting_ids = set(prov.get("history_eligible_pack_ids") or [])
        for row in rows:
            row["history_eligible"] = row["case"] in acting_ids
            row["score_branches"] = (prov.get("score_branches") or {}).get(row["case"], [])
            source_case = next(c for c in job["cases"] if c["case_id"] == row["case"])
            row["confirmation"] = (source_case["latent"].get("m2") or {}).get("confirmation")
        history_counts = {k: history_slots(k, diversity=diversified) for k in qs} if acting_ids else {}
        original = (confirmation_reference(S, rows, qs, seed) if prov.get("confirmation") else
                    S.select_expanded_nested(rows, qs, seed, required_specs=PACK.EXPANSION_IDS))
        # Preserve the original nested clinical strata, including diagnostic
        # spec, sex and measured difficulty. Replace prospective candidates
        # only within that same stratum, using common two-world eligibility.
        # This keeps plan_cells and gold-kind quotas, rather than updating an
        # expected quota to whatever a second free selection happened to pick.
        pick = paired_strata(original, [r for r in rows if r["case"] in allowed], seed,
                             history_counts=history_counts, diversity=diversified)[n]
        balance = S.balance(pick, rows, "expanded")
    else:
        pick = S.select(rows, prov["design"], seed)
        balance = S.balance(pick, rows)
    if not balance["pass"]:
        raise ValueError(f"paired M2 balance failed: {balance}")
    return sorted(r["case"] for r in pick)


def confirmation_reference(selector, rows, quotas, seed):
    """Clinical marginal targets, before the joint nested pair allocation.

    The joint MILP below supplies actual nesting. A separate nested search on
    unpaired identities would constrain a second, discarded cohort first.
    Negative sex targets follow the eligible clinical cohort, using the same
    existing half-up rounding and flow; final BAL gates remain unchanged.
    """
    return {n: selector.select_expanded(rows, selector.negative_sex_quotas(q, rows), seed,
                                       required_specs=PACK.EXPANSION_IDS, _neg_sex=True)
            for n, q in sorted(quotas.items())}


def score_branches(gold: dict) -> list[str]:
    """The existing R2 coverage branches, measured on the built clinical record."""
    frame = gold.get("frame")
    if frame == "F1" and gold.get("urgency"):
        return ["F1:" + gold["urgency"]]
    if frame == "F2":
        return ["F2:" + m for m in gold.get("med_accept") or [gold.get("med_direction")] if m]
    if frame == "F3" and gold.get("f3_plan"):
        return ["F3:" + gold["f3_plan"]["arm"]]
    return []


def paired_strata(original: dict[int, list[dict]], candidates: list[dict], seed: str,
                  history_counts: dict | None = None, *, diversity=False) -> dict[int, list[dict]]:
    """Match clinical quotas on common eligibility, jointly across nested sizes.

    Preserve frame/arm/sex and domain coverage, with registered diagnosis-kind
    weights, mandatory spec coverage and natural spec caps. Test-count tiers
    remain descriptive. No patient name is a constraint.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array
    from haenv.world.overlay import condition_registry
    S = _load("pack1_select", SELECT)
    reg = condition_registry(include_draft=True)
    candidates = sorted((r for r in candidates if not r.get("insufficient")),
                        key=lambda r: (S.key(r["case"], seed), r["case"]))
    sizes, width = sorted(original), len(candidates)
    objective = [i / max(1, width) for _n in sizes for i in range(width)]
    matrix, low, high = [], [], []
    def constraint(values, lo, hi):
        matrix.append(values); low.append(lo); high.append(hi)
    def groups(row):
        return ((row["frame"], row["arm"], row.get("tier") if not diversity or row["frame"] == "F0" else None),
                (row["frame"], row["arm"], row["sex"]),
                reg[row["spec"]]["join_gold"],
                (row["arm"], bool(row.get("outside_endocrine"))))
    specs = sorted({r["spec"] for r in candidates})
    confirmation = any(r.get("confirmation") for r in candidates)
    for z, n in enumerate(sizes):
        offset = z * width
        constraint({offset+i: 1 for i in range(width)}, n, n)
        for dimension in range(4):
            target = Counter(groups(r)[dimension] for r in original[n])
            if dimension == 2 and diversity:
                from haenv.m2.diversity import rules
                share = rules()["allocation"]["comorbidity_share"]["value"]
                diagnostic = target["unified"] + target["comorbidity"]
                target["comorbidity"] = round(n * share)
                target["unified"] = diagnostic - target["comorbidity"]
            keys = set(target) | {groups(r)[dimension] for r in candidates}
            for key in sorted(keys, key=str):
                lo, hi = target[key], target[key]
                if confirmation and dimension == 3 and key[0] == "positive":
                    lo, hi = (target[key], np.inf) if key[1] else (0, target[key])
                constraint({offset+i: 1 for i, r in enumerate(candidates) if groups(r)[dimension] == key},
                           lo, hi)
        cap = PACK.expanded_spec_cap(n)
        for spec in specs:
            indexes = {offset+i: 1 for i, r in enumerate(candidates) if r["spec"] == spec}
            units = []
            for j in range(cap):
                units.append(len(objective))
                objective.append(float(j * (width + 1)))
            constraint({**indexes, **{i: -1 for i in units}}, 0, 0)
        if confirmation:
            index = {r["case"]: i for i, r in enumerate(candidates)}
            pairs = {tuple(r["confirmation"]["pair"]) for r in candidates if r.get("confirmation")}
            for a, b in sorted(pairs):
                if a in index and b in index:
                    constraint({offset+index[a]: 1, offset+index[b]: -1}, 0, 0)
                else:
                    for cid in (a, b):
                        if cid in index:
                            constraint({offset+index[cid]: 1}, 0, 0)
            from haenv.m2.confirmation import rules as confirmation_rules
            for profile in confirmation_rules()["profiles"]:
                count = round(n / 50 * confirmation_rules()["allocation"]["profiles_per_50"])
                constraint({offset+i: 1 for i, r in enumerate(candidates)
                            if r.get("confirmation") and r["confirmation"]["profile"] == profile
                            and not r["confirmation"]["present"]}, count, count)
            for sid in sorted(s for s in specs if reg[s]["join_gold"] == "independent"):
                constraint({offset+i: 1 for i, r in enumerate(candidates) if r["spec"] == sid}, 1, np.inf)
        if n in (history_counts or {}):
            target = history_counts[n]
            constraint({offset+i: 1 for i, r in enumerate(candidates) if r.get("history_eligible")}, target, target)
            # Moving candidates to fill history slots must retain the already
            # required score branches; prospective difficulty tiers are not urgency.
            for branch in sorted({b for r in candidates for b in r.get("score_branches", ())}):
                constraint({offset+i: 1 for i, r in enumerate(candidates)
                            if branch in r.get("score_branches", ())}, 1, np.inf)
        for sid, minimum in PACK.expanded_required_counts(n).items():
            constraint({offset+i: 1 for i, r in enumerate(candidates)
                        if r["spec"] == sid and r["arm"] == "positive"}, minimum, np.inf)
        if diversity:
            from haenv.m2.diversity import family
            classic = sum(r["frame"] == "F0" and family(r["spec"]) == "classic" for r in original[n]) + max(1, round(
                n * (confirmation_rules() if confirmation else rules())["allocation"]["classic_additional_share"]["value"]))
            constraint({offset+i: 1 for i, r in enumerate(candidates)
                        if family(r["spec"]) == "classic"}, classic, classic)
        if z:
            for i in range(width):
                constraint({(z-1)*width+i: 1, offset+i: -1}, -np.inf, 0)
    rid, cid, value = [], [], []
    for i, row in enumerate(matrix):
        for j, v in row.items(): rid.append(i); cid.append(j); value.append(v)
    a = coo_array((value, (rid, cid)), shape=(len(matrix), len(objective))).tocsc()
    result = milp(np.array(objective), integrality=np.ones(len(objective)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(a, low, high), options={"mip_rel_gap": 0.0})
    if not result.success:
        raise ValueError(f"paired M2 strata shortfall: {result.message}")
    return {n: [r for i, r in enumerate(candidates) if result.x[z*width+i] > 0.5]
            for z, n in enumerate(sizes)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=PS.DEFAULT_N)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pool-out", default=None, help="also write the pool job (>= 4 x N F1-F3 cases) here")
    ap.add_argument("--background", choices=("on", "off"), default=None,
                    help="the background layer (default on, or HAENV_M2_BACKGROUND); off reproduces the pack of before it byte for byte")
    ap.add_argument("--allocation-mode", choices=("auto", "legacy", "expanded"), default="auto",
                    help="expanded uses benchmark strata and mandatory new-diagnosis coverage, not prevalence")
    a = ap.parse_args(argv)
    if a.background:
        BACKGROUND_ON[0] = a.background == "on"
    import logging
    logging.disable(logging.WARNING)
    text = build(a.n, PS.cli_seed(a.seed), pathlib.Path(a.pool_out) if a.pool_out else None, mode=a.allocation_mode)
    out = pathlib.Path(a.out)
    if a.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == text
        print(f"[m2_gen_job] {'up to date' if same else 'STALE'}: {out}")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8")
    print(f"[m2_gen_job] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
