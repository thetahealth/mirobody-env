#!/usr/bin/env python
"""Data for the in-browser generator (`gen.mjs`): kernel constants, per-persona random
layers and golden vectors, all computed by production code.

`gen.mjs` computes the patient from the page's knobs. Two layers:

| Layer | Computed by |
|---|---|
| Deterministic math (curve shapes, feasible curvature, adherence line, drug effect, clamps) | the browser |
| Random layer (AR(1) shocks, missed weigh-ins, measurement variation, per-person draws) | production, exported as tables |

The random layer depends only on `(case_id, index)`, never on a knob, so production
computes it ahead of time and the browser looks it up. The deterministic layer is checked
against golden vectors that production computes (`_golden`).

This module needs only the haenv package. The page-data export writes its output into
`data.json`; `check_page_data.py` recomputes it to confirm the page data is current.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import inspect
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

#: The patients the page can switch between; each has its own exported random layer.
PERSONAS = tuple(f"DEMO-{i:02d}" for i in range(1, 13))

#: Extra random layers for the "parallel futures" fan: the same knobs rendered with other
#: case ids. Only the weight-stream tables are exported for them.
FAN_PERSONAS = tuple(f"DEMO-{i:02d}" for i in range(13, 33))

#: Decimals kept for the carried-forward uniforms. The page compares them with a rate in
#: 0.1 % steps; at 9 decimals a comparison flips only when a draw lies within 5e-10 of it.
_CF_ND = 9

#: Decimals kept for shock tables. Shocks lie in [-1, 1) and are scaled by at most 0.12 kg,
#: so the rounding error (1e-7 kg) cannot change a value rounded to 0.01.
_SHOCK_ND = 6


#: English citation for registry drug entries whose `source` is not written in English.
#: The registry file is part of the release fingerprint, so the translation lives here.
DRUG_SOURCE_EN = {
    "metformin": (
        "Hirst JA, Farmer AJ, Ali R, Roberts NW, Stevens RJ. Quantifying the effect of "
        "metformin treatment and dose on glycemic control. Diabetes Care 2012;35(2):446-54. "
        "PMID 22275444 · DOI 10.2337/dc11-1465. Abstract, verbatim: \"Metformin monotherapy "
        "lowered HbA(1c) by 1.12% (95% CI 0.92-1.32; I(2) = 80%) versus placebo\"; 35 trials."),
    "dulaglutide": (
        "Dungan KM et al. Once-weekly dulaglutide versus once-daily liraglutide in "
        "metformin-treated patients with type 2 diabetes (AWARD-6). Lancet "
        "2014;384(9951):1349-57. PMID 25018121 · DOI 10.1016/S0140-6736(14)60976-4 · "
        "NCT01624259. Verbatim: \"Least-squares mean reduction in HbA1c was -1·42% (SE 0·05) "
        "in the dulaglutide group\". Weight from the trial registry: -2.90 kg (SE 0.22) at "
        "26 weeks."),
    "liraglutide": (
        "Dungan KM et al. Once-weekly dulaglutide versus once-daily liraglutide in "
        "metformin-treated patients with type 2 diabetes (AWARD-6). Lancet "
        "2014;384(9951):1349-57. PMID 25018121 · NCT01624259. Verbatim: \"-1·36% (0·05) in "
        "the liraglutide group\". Weight from the trial registry: -3.61 kg (SE 0.22) at "
        "26 weeks."),
}


def _drug_source(name: str, spec: dict) -> str | None:
    """The drug's citation in English: the registry text, or `DRUG_SOURCE_EN` when the
    registry text has CJK. A CJK source without an English entry raises."""
    src = spec.get("source")
    if src is None or not re.search(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]", src):
        return src
    if name not in DRUG_SOURCE_EN:
        raise ValueError(f"drug `{name}`: registry source is not in English and "
                         f"DRUG_SOURCE_EN has no entry for it")
    return DRUG_SOURCE_EN[name]


def _drug_fields() -> tuple[str, ...]:
    """Registry fields `drug_effects.total_effect` reads: every effect field plus the
    trial weight change."""
    sys.path.insert(0, str(ROOT))
    from haenv import drug_effects as DE
    return tuple(sorted(set(DE.EFFECT_FIELDS.values()))) + ("trial_weight_kg",)


def _kernel() -> dict:
    """Every constant the browser generator uses, read from the production modules."""
    sys.path.insert(0, str(ROOT))
    from haenv import build as B
    from haenv import events as E
    from haenv import latent_rules as LR
    from haenv import drug_effects as DE
    from haenv import indicators as _ind
    from haenv_kernel.latent import DISEASE_SIGNAL_DOMAIN as DOM          # kernel

    # The descent curvature is sampled per case; only the registered target and spread
    # are kernel constants. Each persona's value is in `personas[cid].descent_k`.
    k_desc_target, k_reb, k_desc_spread = B._trajectory_shape_registered()
    drug_fields = _drug_fields()
    return {
        # Weight trajectory (`build._weight_series`)
        "weight_step": B.WEIGHT_STEP, "weight_max_gap": B.WEIGHT_MAX_GAP,
        "weigh_p_weekday": B.WEIGH_P_WEEKDAY, "weigh_p_weekend": B.WEIGH_P_WEEKEND,
        "end": B.END, "wobble_period": B.WOBBLE_PERIOD,
        "ar_phi": E._AR_PHI, "ar_sd_frac": E._AR_SD_FRAC,
        "descent_budget_frac": LR.DESCENT_BUDGET_FRAC,
        "descent_k_target": k_desc_target, "descent_k_spread": k_desc_spread,
        "rebound_k": k_reb,
        "amp_cap": 0.12,
        # Clinical signals (`build.render_clinical`, `drug_effects.effect_at`)
        "clinical_atten": B.CLINICAL_ATTEN, "clinical_lag_days": B.CLINICAL_LAG_DAYS,
        # `effect_at`'s default; production never passes it.
        "onset_days": inspect.signature(DE.effect_at).parameters["onset_days"].default,
        "effect_fields": dict(DE.EFFECT_FIELDS),
        "half_time_days": dict(DE.HALF_TIME_DAYS),
        "low_response_driver": DE.LOW_RESPONSE_DRIVER,
        "response_bounds": {"low": list(DE.response_bounds()[0]),
                            "responder": list(DE.response_bounds()[1])},
        "clinical_spec": {s: {"per_kg": v[0], "step": v[1],
                              "ndigits": _ind.of(s)["ndigits"],
                              "cohort": {d: _ind.cohort_of(s, d, [])
                                         for d in DOM if s in DOM[d]}}
                          for s, v in B.CLINICAL_SPEC.items()},
        # Reference range, abnormal side and diagnostic threshold per signal
        # (`indicators.of` / `abnormal_side`), for the lab small multiples.
        "clinical_ref": {s: _clinical_ref(s) for s in B.CLINICAL_SPEC},
        "clinical_cv": B._clinical_cv(),
        "by_device": {k: sorted(v) for k, v in B.CLINICAL_BY_DEVICE.items()},
        # Disease domains: value range, weekly slope limit, unit
        "domain": {d: {s: dict(c) for s, c in sig.items()} for d, sig in DOM.items()},
        # Drug registry. The browser derives `direct = total - per_kg x atten x dw_trial`
        # itself, so a change to `per_kg` cannot leave an exported direct term stale.
        "drugs": {n: {**{f: s.get(f) for f in drug_fields},
                      "dose_ladder": {str(k): {f: v.get(f) for f in drug_fields}
                                      for k, v in (s.get("dose_ladder") or {}).items()},
                      "source": _drug_source(n, s), "background": s.get("background")}
                  for n, s in DE.drugs().items()},
        "drug_forbidden": list(DE.FORBIDDEN),
        "drug_cohorts": list(DE._doc().get("applies_to_cohorts_default") or []),
        # Weight skeleton (`build._weight_render`, `_weight_overlay` and their helpers)
        "skel": {
            "descent_max_phases": B.SKEL_DESCENT_MAX_PHASES, "stall_pace": B.SKEL_STALL_PACE,
            "hold_pace": B.SKEL_HOLD_PACE, "episode_max_days": B.SKEL_EPISODE_MAX_DAYS,
            "max_run_days": B.SKEL_MAX_RUN_DAYS, "run_margin_kg": B.SKEL_RUN_MARGIN_KG,
            "regain_run_margin_kg": B.SKEL_REGAIN_RUN_MARGIN_KG,
            "valley_min_days": B.SKEL_VALLEY_MIN_DAYS, "gain_frac": list(B.SKEL_GAIN_FRAC),
            "gain_residual": list(B.SKEL_GAIN_RESIDUAL),
            "gain_tau_days": list(B.SKEL_GAIN_TAU_DAYS),
            "episode_pace_frac": B.SKEL_EPISODE_PACE_FRAC,
            "rebound_variants": list(B.SKEL_REBOUND_VARIANTS),
            "drift_tau_days": list(B.SKEL_DRIFT_TAU_DAYS), "drift_sd_kg": list(B.SKEL_DRIFT_SD_KG),
            "drift_bound_kg": B.SKEL_DRIFT_BOUND_KG,
            "drift_bound_down_kg": B.SKEL_DRIFT_BOUND_DOWN_KG,
            "drift_slope_frac": B.SKEL_DRIFT_SLOPE_FRAC, "fade_days": B.SKEL_FADE_DAYS,
            "regain_margin_kg": B.SKEL_REGAIN_MARGIN_KG,
            "regain_margin_days": B.SKEL_REGAIN_MARGIN_DAYS, "settle_kg": B.SKEL_SETTLE_KG,
            "settle_days": B.SKEL_SETTLE_DAYS,
            "reversal_slope_gain": B.SKEL_REVERSAL_SLOPE_GAIN,
            "anchor_margin_kg": B.SKEL_ANCHOR_MARGIN_KG,
        },
        # Indication registry (`gates_case.check_drug_indication`, GEN27): per condition and
        # drug, the status and the dose ladder. The page's dose ladder is this ladder.
        "indications": _indications(),
        # Minimum days between two dose changes (`build_clinical._world_medication`)
        "min_titration_days": {n: _min_titration(n) for n in DE.drugs()},
        # Administrations per week (1 = weekly injection, 7 = daily), for the dose unit
        "doses_per_week": _doses_per_week(),
        # The primary dose record (`med_course.dose_course`)
        "med_course": _med_consts(),
        # The emission-gate limits the page re-applies (`synth.premise_conflicts`,
        # `gates_case.check_anchors_honored`)
        "gate": {"slope_tol": _slope_tol(), "anchor_tol_kg": _anchor_tol()},
        # The label rule the build writes into every case (`latent_rules`)
        "label_rule": {"min_change_frac": LR.MIN_CHANGE_FRAC, "min_change_kg": LR.MIN_CHANGE_KG,
                       "smoothing": LR.LABEL_SMOOTHING,
                       "min_persist_days": LR.MIN_PERSIST_DAYS},
    }


def _indications() -> dict:
    """`registry/drug_indications.yaml` pairs, status and dose ladder only."""
    from haenv.gates_case import _judging_registry
    pairs = _judging_registry("drug_indications.yaml").get("pairs") or {}
    return {dis: {drug: {"status": str(sp.get("status") or "unregistered"),
                         "dose_steps": [float(x) for x in (sp.get("dose_steps") or [])]}
                  for drug, sp in sorted(drugs.items())}
            for dis, drugs in sorted(pairs.items())}


def _doses_per_week() -> dict:
    from haenv.gate_tables import DRUG_DOSES_PER_WEEK
    return {k: int(v) for k, v in sorted(DRUG_DOSES_PER_WEEK.items())}


def _min_titration(drug: str) -> int:
    """`_world_medication`'s `min_titration_days` for `drug`, default included."""
    from haenv_kernel.latent import drug_pkpd
    return int((drug_pkpd(drug) or {}).get("min_titration_days", 28))


def _med_draws(cid: str) -> dict:
    """The `rng.unit` draws `med_course.dose_course` reads for one case id, exact. The repeat
    records pick, per slot `j`, the first eligible day in the order of their draw, so that order
    (days 1..END) is exported instead of the draws."""
    from haenv import build as B
    from haenv import med_course as M
    from haenv.rng import unit
    n_rungs = max(len(sp["dose_steps"]) for drugs in _indications().values() for sp in drugs.values())
    ep = range(M.MAX_DOSE_DOWN_EPISODES)
    return {"titr": [unit(cid, "med", "titr", i) for i in range(1, n_rungs)],
            "n_ep": unit(cid, "med", "n_ep"),
            **{k: [unit(cid, "med", k, j) for j in ep] for k in ("dn", "up", "hold", "stay")},
            "mid_at": unit(cid, "med", "mid_at"), "mid": unit(cid, "med", "mid"),
            "end_rec": unit(cid, "med", "end_rec", "primary"),
            "confirm_order": [sorted(range(1, B.END + 1), key=lambda d: unit(cid, "med", "confirm", j, d))
                              for j in range(M.MIN_VISIBLE_DOSE_POINTS)]}


def _med_literals() -> dict:
    """Literals inside `dose_course` the page needs, read from its source; a changed form raises."""
    from haenv import med_course as M
    src = inspect.getsource(M.dose_course)
    pats = {"titr_jitter": r'"titr", i\) \* ([0-9.]+) \* mt',
            "confirm_gap": r"abs\(d - u\) >= ([0-9]+)",
            "mid_lo": r"\(([0-9.]+) \+ [0-9.]+ \* unit\(case_id, \"med\", \"mid_at\"\)",
            "mid_span": r"[0-9.]+ \+ ([0-9.]+) \* unit\(case_id, \"med\", \"mid_at\"\)",
            "mid_share": r'unit\(case_id, "med", "mid"\) < ([0-9.]+)'}
    out = {}
    for k, pat in pats.items():
        m = re.search(pat, src)
        if not m:
            raise ValueError(f"med_course.dose_course no longer has the `{k}` literal in the expected form")
        out[k] = float(m.group(1))
    return out


def _med_consts() -> dict:
    from haenv import med_course as M
    return {**_med_literals(), "min_visible_points": M.MIN_VISIBLE_DOSE_POINTS,
            "episode_weights": list(M.DOSE_EPISODE_WEIGHTS),
            "max_down_episodes": M.MAX_DOSE_DOWN_EPISODES, "hold_share": M.HOLD_SHARE,
            "stay_down_share": M.STAY_DOWN_SHARE, "end_record_window": M.END_RECORD_WINDOW}


def _slope_tol() -> float:
    """The tolerance `premise_conflicts` puts on `max_weekly_delta`. It is a literal in that
    function, so it is read from the source; a changed form raises instead of drifting."""
    from haenv_kernel import synth
    m = re.search(r'dom\["max_weekly_delta"\] \* ([0-9.]+)', inspect.getsource(synth.premise_conflicts))
    if not m:
        raise ValueError("premise_conflicts no longer writes the slope limit as "
                         "max_weekly_delta * <tolerance>")
    return float(m.group(1))


def _anchor_tol() -> float:
    from haenv import gates_case
    return float(gates_case.ANCHOR_TOL_KG)


def _release() -> dict:
    """The current release: freeze revision from `docs/anchor`, and the world and judging
    fingerprints of the code as it is now."""
    sys.path.insert(0, str(ROOT))
    from haenv import anchor
    snap = anchor.load_freeze()
    return {"freeze_revision": snap.get("revision"), "world_sha": anchor.world_fingerprint(),
            "judging_sha16": anchor.judging_fingerprint()}


#: The page's Act 1 devices; the lab baseline is sampled for this inventory.
PAGE_DEVICES = ("smart_scale", "lab_panel")

#: The conditions the page's builder offers.
PAGE_DISEASES = ("T2D", "obesity")


def _gate(cid: str, dis: str, drug: str, steps: list, start: float, nadir: float, T: int,
          outcome: str, w: list[dict], observed: list) -> list[str]:
    """The emission-gate kinds production returns for this weight course: the kernel's
    premise check on the course (value range, weekly slope), the value range on the recorded
    readings, GEN22 anchors and GEN13 outcome on the course, and GEN27 indication. Sorted,
    one entry per kind."""
    from types import SimpleNamespace
    from haenv import gates as G
    from haenv.gates_case import check_drug_indication
    from haenv_kernel.latent import LatentPremise
    from haenv_kernel.schema import RawCase
    from haenv_kernel.synth import premise_conflicts
    p = LatentPremise(patient_basics={"disease": dis, "regimen": {"drug": drug, "dose_steps": list(steps)}},
                      event_density={}, device_signals={"devices": list(PAGE_DEVICES), "signals": {"weight": {}}},
                      adherence={"baseline": 0.95, "trajectory": []})

    def raw(series):
        return RawCase(case_id=cid, user_profile={}, prediction_context={"prediction_time_T": T},
                       longitudinal_data={"weight": series}, evidence_ledger=[],
                       outcome_label="event_occurred" if outcome == "regain" else "event_not_occurred",
                       label_rule=_rule(), gold_drivers=["weight_trend"], adjudication={})
    cs = SimpleNamespace(raw={"start_weight": start, "nadir_weight": nadir}, noise=[], latent={})
    truth, obs = raw(w), raw([{"ts": d, "value": v} for d, v in observed])
    kinds = [c["kind"] for c in premise_conflicts(truth, p)]
    kinds += [h["kind"] for h in G.check_observed_in_domain(obs, p) if h["severity"] == "gate"]
    for hits in (G.check_anchors_honored(truth, cs), G.check_outcome_derivable(truth, cs),
                 check_drug_indication(p)):
        kinds += [h["kind"] for h in hits if h["severity"] == "gate"]
    return sorted(set(kinds))


def _clinical_ref(sig: str) -> dict:
    """`reference_range`, `direction` and `diagnostic_for` of one indicator dossier."""
    from haenv import indicators as _ind
    d = _ind.of(sig)
    ref = d.get("reference_range") or {}
    return {"low": ref.get("low"), "high": ref.get("high"),
            "direction": str(d.get("direction", "higher_abnormal")),
            "diagnostic": {k: v.get("ge") for k, v in (d.get("diagnostic_for") or {}).items()
                           if isinstance(v, dict) and v.get("ge") is not None}}


def _cf_uniforms(cid: str) -> list[float]:
    """`post_inject._u01(<case>|weight|carried_forward, i)` for every point index."""
    from haenv import build as B
    from haenv.post_inject import _u01
    return [round(_u01(f"{cid}|weight|carried_forward", i), _CF_ND) for i in range(B.END + 1)]


def _weight_layer(cid: str) -> dict:
    """The random tables the weight stream reads for one case id.

    `shock` starts at `k = -1` (the stationary start), so the browser reads `shock[k+1]`.
    """
    from haenv import build as B
    from haenv import events as E
    return {
        "shock": [round(E._det_shock(f"{cid}|weight|wobble", k), _SHOCK_ND)
                  for k in range(-1, B.END + 1)],
        # Whether the patient weighed in on day d (used only under daily sampling)
        "weighed": [1 if B._weighed_on(d, cid) else 0 for d in range(0, B.END + 1)],
        # Desired descent curvature, capped later by `feasibleK`
        "descent_k": B._trajectory_shape(cid)[0],
        # The weight skeleton's uniforms (`build._skeleton_draws`), exact: they steer
        # guard decisions, so they are not rounded
        "skel": B._skeleton_draws(cid),
        # Carried-forward draws (`post_inject.carried_forward`), by point index
        "cf_u": _cf_uniforms(cid),
    }


def _fan_personas() -> dict:
    """Weight-stream random layers for `FAN_PERSONAS`, computed by production."""
    sys.path.insert(0, str(ROOT))
    return {cid: _weight_layer(cid) for cid in FAN_PERSONAS}


def _personas() -> dict:
    """Each persona's random layer, computed by the production functions."""
    sys.path.insert(0, str(ROOT))
    from haenv import build as B
    from haenv import drug_effects as DE
    from haenv import events as E

    out = {}
    for cid in PERSONAS:
        out[cid] = {
            **_weight_layer(cid),
            # Per-draw measurement variation of each clinical signal, by draw index
            "meas": {sig: [round(E._det_shock(f"{cid}|{sig}|meas", i), _SHOCK_ND)
                           for i in range(64)]
                     for sig in sorted(B._clinical_cv())},
            # Drug-response multiplier (`drug_effects.response_for`): the draw depends only
            # on `case_id`; the knobs pick the driver band and the drug, so production maps
            # it for every registered drug and for the non-response band.
            "response": {
                "low": DE.response_for(cid, DE.LOW_RESPONSE_DRIVER, ""),
                "by_drug": {n: DE.response_for(cid, None, n) for n in DE.drugs()},
            },
            # The dose record's draws (`med_course.dose_course`)
            "med": _med_draws(cid),
            # Each lab's baseline as `clinical_plan` samples it for this case, per condition
            "lab_base": {dis: {s: v["base"] for s, v in B.clinical_plan(dis, list(PAGE_DEVICES), [], cid).items()}
                         for dis in PAGE_DISEASES},
        }
    return out


#: Decimals kept for the observation-noise draws `u(d)`. They are scaled by at most
#: `sd_frac x mass / marginal_sd` (< 1.6 for a 300 kg patient) before the 0.01 kg rounding;
#: at 12 decimals no rounding of a reading can flip.
_WNOISE_ND = 12


def _wnoise() -> dict:
    """The weight observation layer (`physio/apply.apply_physio` on the weight stream), as the
    page needs it: the stream registry's constants and, per persona, the per-day draws
    `noise.correlated_noise(case_id, d, [weight spec])` and the weekday phase of
    `noise.weight_weekday`. Both depend on `case_id` and the day only, never on a knob."""
    sys.path.insert(0, str(ROOT))
    import hashlib
    import math
    from haenv import build as B
    from haenv import events as E
    from haenv.events_pools import _declared_ndigits
    from haenv.physio import apply as PA
    from haenv.physio import noise as NZ
    reg = PA.load_stream_registry(E._rp("physio_streams.yaml"))
    ws = reg.specs["weight"]
    if ws.bound.transform != "identity":
        raise ValueError("the page's observation layer assumes the identity transform for weight")
    week_var = sum(f * f for f in NZ.WEIGHT_WEEKDAY_FRAC) / len(NZ.WEIGHT_WEEKDAY_FRAC)
    sd = float(reg.weight_sd_frac)
    consts = {"phi": float(reg.ar_phi.get("weight", 0.0)), "sd_frac": sd,
              "marginal_sd": ws.noise.marginal_sd,
              "keep": math.sqrt(max(0.0, 1.0 - week_var / max(1e-12, sd * sd))),
              "weekday_frac": list(NZ.WEIGHT_WEEKDAY_FRAC), "lo": ws.bound.lo, "hi": ws.bound.hi,
              "max_step": ws.bound.max_step, "ndigits": _declared_ndigits("weight"),
              "grid": NZ.WEIGHT_SCALE_RESOLUTION_KG}
    per = {}
    for cid in PERSONAS:
        phase = int(hashlib.sha256(str(cid).encode()).hexdigest()[:8], 16) % 7
        if any(NZ.weight_weekday(cid, d) != (d + phase) % 7 for d in range(7)):
            raise ValueError("noise.weight_weekday is no longer (day + sha256 phase) mod 7")
        per[cid] = {"phase7": phase,
                    "u": [round(NZ.correlated_noise(cid, d, [ws.noise])["weight"], _WNOISE_ND)
                          for d in range(0, B.END + 1)]}
    return {"consts": consts, "per": per,
            "source": "haenv/physio/apply.py:apply_physio · haenv/physio/noise.py:weight_noise_shaped · "
                      "haenv/physio/bounds.py:project · haenv/events_pools.py:_round_to_declared_digits · "
                      "haenv/physio/noise.py:weight_on_scale_grid"}


def _observed(cid: str, w: list[dict]) -> list[list]:
    """Production's recorded readings for the weight course `w`: the physiology layer on the
    weight stream (no symptom events), the declared rounding, then the scale's display grid."""
    from haenv import events as E
    from haenv.events_pools import _round_to_declared_digits
    from haenv.physio import apply as PA
    from haenv.physio.noise import weight_on_scale_grid
    reg = PA.load_stream_registry(E._rp("physio_streams.yaml"))
    ld, _ = PA.apply_physio(cid, {"weight": [dict(p) for p in w]}, reg)
    _round_to_declared_digits(ld)
    return [[p["ts"], weight_on_scale_grid(p["value"])] for p in ld["weight"]]


#: Golden-vector parameter sets. Each one exercises a branch the others do not.
_GOLDEN_CASES = (
    # (label, case_id, disease, start, nadir, T, outcome, rev_week, slope, mpw, adh_low, driver, devices)
    ("regain, daily sampling", "DEMO-01", "T2D", 92.0, 80.0, 84, "regain", 16, 0.35, 7, 0.55,
     "poor_medication_adherence", ("smart_scale", "lab_panel")),
    ("maintain, daily sampling", "DEMO-02", "T2D", 92.0, 80.0, 84, "maintain", 16, 0.35, 7, 0.95,
     "unknown_or_multifactorial", ("smart_scale", "lab_panel")),
    ("regain before T", "DEMO-03", "obesity", 104.0, 88.0, 84, "regain", 6, 0.6, 7, 0.7,
     "poor_medication_adherence", ("smart_scale", "lab_panel")),
    ("loss at the physiological limit: curvature capped to linear", "DEMO-04", "T2D", 110.0,
     86.0, 84, "regain", 20, 0.2, 7, 0.8, "poor_medication_adherence", ("smart_scale", "lab_panel")),
    ("almost no loss: flat line", "DEMO-05", "obesity", 71.0, 70.6, 84, "maintain", 16, 0.35, 7,
     0.95, "unknown_or_multifactorial", ("smart_scale", "lab_panel")),
    ("weekly weighing (sparse, no missed days)", "DEMO-06", "T2D", 88.0, 79.0, 84, "regain", 22,
     0.45, 1, 0.6, "poor_medication_adherence", ("smart_scale", "lab_panel")),
    ("weighing every other day", "DEMO-07", "obesity", 95.0, 84.0, 120, "maintain", 16, 0.35,
     3.5, 0.9, "unknown_or_multifactorial", ("smart_scale", "lab_panel")),
    ("late regain, near the course end", "DEMO-08", "T2D", 99.0, 85.0, 84, "regain", 44, 0.3, 7,
     0.5, "poor_medication_adherence", ("smart_scale", "lab_panel")),
    # CGM_TIR: weekly stream with per-draw measurement variation
    ("weekly stream with measurement variation (CGM)", "DEMO-09", "T2D", 96.0, 83.0, 84,
     "regain", 18, 0.4, 7, 0.6, "poor_medication_adherence", ("smart_scale", "cgm")),
    # A 45 kg loss pushes HbA1c below its range: the value-range clamp
    ("very large loss: range floor (cohort without drug effect)", "DEMO-10", "obesity", 140.0,
     95.0, 84, "maintain", 16, 0.35, 7, 0.95, "unknown_or_multifactorial",
     ("smart_scale", "lab_panel")),
    ("very large loss with drug effect: clamp and drug together", "DEMO-11", "T2D", 140.0, 95.0,
     84, "regain", 20, 0.3, 7, 0.6, "poor_medication_adherence", ("smart_scale", "lab_panel")),
    # Weekly slope limit binding on a quantized weekly stream (`_quantize_within_slope`)
    ("weekly slope limit binds (CGM)", "DEMO-04", "T2D", 110.0, 86.0, 84, "regain", 16, 1.0, 7,
     0.25, "poor_medication_adherence", ("smart_scale", "cgm")),
    # The non-response band of `drug_effects.response_for`
    ("non-responder: low response multiplier", "DEMO-12", "T2D", 96.0, 86.0, 84, "maintain", 16,
     0.35, 7, 0.95, "biological_low_response", ("smart_scale", "lab_panel")),
    # Weekly weighing with a late "today": the guard's whole-course anchor check decides the level
    ("weekly weighing, late today: anchor check sets the guard level", "DEMO-09", "T2D", 75.0,
     68.0, 200, "regain", 40, 0.2, 1, 0.55, "poor_medication_adherence", ("smart_scale", "lab_panel")),
    ("three weigh-ins a week, early today", "DEMO-05", "obesity", 110.0, 95.0, 30, "regain", 30,
     0.6, 3, 0.55, "poor_medication_adherence", ("smart_scale", "lab_panel")),
    ("twice a week, maintain, late today", "DEMO-10", "T2D", 82.0, 74.0, 300, "maintain", 16, 0.35,
     2, 0.9, "unknown_or_multifactorial", ("smart_scale", "lab_panel")),
    # Refused at the emission gate. The guard finds no course inside the weekly limit and
    # falls back to one with a one-day jump; production refuses it.
    ("refused: one-day jump above the weekly limit", "DEMO-04", "obesity", 136.0, 91.0, 313,
     "regain", 17, 0.8, 7, 0.6, "poor_medication_adherence", ("smart_scale", "lab_panel")),
    ("refused: the course cannot reach the declared lowest weight", "DEMO-06", "T2D", 96.0, 60.0,
     28, "regain", 19, 0.45, 6, 0.6, "poor_medication_adherence", ("smart_scale", "lab_panel")),
    ("refused: declared regain, the label rule reads no regain", "DEMO-03", "T2D", 138.0, 107.0,
     122, "regain", 45, 0.6, 7, 0.6, "poor_medication_adherence", ("smart_scale", "lab_panel")),
    ("refused: drug without an indication for the condition", "DEMO-02", "obesity", 92.0, 80.0, 84,
     "maintain", 16, 0.35, 7, 0.95, "unknown_or_multifactorial", ("smart_scale", "lab_panel")),
    ("weight-loss ladder: obesity on semaglutide, titrated to the top rung", "DEMO-07", "obesity",
     104.0, 88.0, 200, "maintain", 16, 0.35, 7, 0.9, "unknown_or_multifactorial",
     ("smart_scale", "lab_panel")),
    ("daily oral drug: metformin, short titration interval", "DEMO-08", "T2D", 99.0, 90.0, 30,
     "maintain", 16, 0.35, 7, 0.9, "unknown_or_multifactorial", ("smart_scale", "lab_panel")),
)

#: Drug and dose per golden case (default semaglutide 1.0 mg). A dose-ladder drug exercises
#: the per-rung HbA1c and fasting-glucose fields of `drug_effects.total_effect`.
_GOLDEN_DRUG = {"very large loss with drug effect: clamp and drug together": ("tirzepatide", 15.0),
                "refused: drug without an indication for the condition": ("dulaglutide", 1.0),
                "daily oral drug: metformin, short titration interval": ("metformin", 1.0)}


def _adh_traj(T: int, ce: int, adh_low: float, outcome: str, driver: str) -> list[dict]:
    """The adherence line of `build.premise_spec`, which needs a full `CaseSpec`; tests
    compare the two point for point."""
    mid = T + (ce - T) // 2
    if driver == "poor_medication_adherence" and outcome == "regain":
        return [{"ts": 42, "value": 0.95}, {"ts": 56, "value": 0.88}, {"ts": T, "value": 0.80},
                {"ts": mid, "value": round((0.80 + adh_low) / 2, 2)},
                {"ts": ce, "value": adh_low}]
    return [{"ts": 42, "value": 0.96}, {"ts": T, "value": 0.94}, {"ts": mid, "value": 0.93},
            {"ts": ce, "value": max(0.90, adh_low)}]


def _golden() -> dict:
    """Golden vectors: the point values production computes for each parameter set in
    `_GOLDEN_CASES`. The page and `gen_check.mjs` recompute them in JavaScript and compare
    values point for point."""
    sys.path.insert(0, str(ROOT))
    from haenv import build as B
    from haenv import drug_effects as DE
    from haenv import gates as G
    from haenv import indicators as IND
    from haenv import post_inject as PI
    from haenv import med_course as MC

    out = []
    for (label, cid, dis, start, nadir, T, outcome, rw, slope, mpw, adh_low, driver,
         devs) in _GOLDEN_CASES:
        drug, dose = _GOLDEN_DRUG.get(label, ("semaglutide", 1.0))
        # Same call as `build._drug_response_of`
        resp = DE.response_for(cid, driver, drug)
        ce = B.END
        step = max(1, int(round(7.0 / max(0.1, mpw))))
        w, wmeta = B._weight_render(start, nadir, T, outcome, rw, slope,
                                    case_id=cid, step=step, disease=dis, end_day=ce)
        B._SKELETON_AUDIT.pop(cid, None)
        verdict, det = G.derive_outcome({"weight": w}, _rule())
        steps = _indications().get(dis, {}).get(drug, {}).get("dose_steps") or []
        dose_rec = MC.dose_course(cid, steps, T, ce, _min_titration(drug))
        observed = _observed(cid, w)
        adh = _adh_traj(T, ce, adh_low, outcome, driver)
        plan = B.clinical_plan(dis, list(devs), [], cid)
        clin = {}
        for sig, spec in plan.items():
            clin[sig] = {
                "spec": {k: spec[k] for k in ("base", "per_kg", "step", "ndigits",
                                              "cohort", "unit", "range", "max_weekly_delta")},
                "pts": [[p["ts"], p["value"]] for p in B.render_clinical(
                    sig, spec, w, ce, drug=drug, dose_mg=dose,
                    adherence_pts=adh, case_id=cid, response=resp)],
            }
            clin[sig]["abn"] = [IND.abnormal_side(sig, v) for _, v in clin[sig]["pts"]]
        out.append({
            "label": label,
            "params": {"case_id": cid, "disease": dis, "start": start, "nadir": nadir,
                       "T": T, "outcome": outcome, "reversal_week": rw, "regain_slope": slope,
                       "measure_per_week": mpw, "adherence_low": adh_low, "driver": driver,
                       "course_end_day": ce, "drug": drug, "dose_mg": dose,
                       "devices": list(devs)},
            "weight": [[p["ts"], p["value"]] for p in w],
            # The recorded readings after the scale's observation layer (`_observed`).
            "observed": observed,
            # The primary dose record along the indication ladder (`med_course.dose_course`)
            "dose": [[q["ts"], q["value"]] for q in dose_rec],
            # Emission-gate refusals production returns for this course (`_gate`)
            "gate": _gate(cid, dis, drug, steps, start, nadir, T, outcome, w, observed),
            "skeleton": {"level": wmeta["level"], "guard_ok": wmeta["guard_ok"],
                         "pre": wmeta["pre"], "regain": wmeta["regain"]},
            "rule_readout": {"verdict": verdict,
                      **{k: det[k] for k in ("start", "nadir", "lost", "threshold",
                                             "sustained_days", "reading")}},
            "adherence": [[p["ts"], p["value"]] for p in adh],
            "clinical": clin,
            # The daily course the readings are sampled from (`_weight_render`'s final
            # `base`, before the day-to-day wobble), for the page's "truth" view.
            "base": [round(v, _BASE_ND) for v in wmeta["base"]],
            # `post_inject.carried_forward` on this series: `[index, new value]` for every
            # reading it changed, at the registered rate and at a second rate.
            "carried_forward": [
                {"rate": r, "changed": [[i, q["value"]] for i, (p0, q) in enumerate(zip(w, PI.carried_forward(
                    w, rng_key=f"{cid}|weight|carried_forward", rate=r))) if q["value"] != p0["value"]]}
                for r in _cf_rates()],
        })
    return {"cases": out, "overlay": _golden_overlay(), "tol": 0.005,
            "source": "haenv/build.py:_weight_render · _weight_overlay · render_clinical · "
                      "clinical_plan · haenv/gates.py:derive_outcome"}


#: Decimals kept for the golden daily base; the page compares it within 1e-4.
_BASE_ND = 5


def _cf_rates() -> tuple[float, ...]:
    """The registered carried-forward rate and a larger one, for the golden vectors."""
    from haenv import post_inject as PI
    rate = float((PI._rates().get("carried_forward") or {}).get("rate") or 0.0)
    return (rate, 0.3)


def _rule() -> dict:
    """The label rule the build writes into every case."""
    from haenv import latent_rules as LR
    return {"min_change_frac": LR.MIN_CHANGE_FRAC, "min_change_kg": LR.MIN_CHANGE_KG,
            "smoothing": LR.LABEL_SMOOTHING, "min_persist_days": LR.MIN_PERSIST_DAYS}


#: LLM-path golden cases: (label, golden case index, T). Model points are the golden weight
#: series every 7 days (one decimal, as a model writes them); T off a model point exercises
#: the held point at T.
_GOLDEN_OVERLAY = (
    ("LLM path, regain, T on a model point", 0, 84),
    ("LLM path, maintain, T on a model point", 1, 84),
    ("LLM path, regain, T three days after a model point", 0, 87),
    ("LLM path, maintain, T six days after a model point", 1, 90),
    ("LLM path, late regain", 7, 84),
)


def _golden_overlay() -> list[dict]:
    """`build._weight_overlay` on model points taken from the golden weight series."""
    sys.path.insert(0, str(ROOT))
    from haenv import build as B
    out = []
    for label, gi, T in _GOLDEN_OVERLAY:
        (_, cid, dis, start, nadir, _T, outcome, rw, slope, mpw, _a, _d, _v) = _GOLDEN_CASES[gi]
        step = max(1, int(round(7.0 / max(0.1, mpw))))
        w = B._weight_series(start, nadir, _T, outcome, rw, slope, case_id=cid, step=step,
                             disease=dis, end_day=B.END)
        B._SKELETON_AUDIT.pop(cid, None)
        model = [{"ts": p["ts"], "value": round(p["value"], 1)} for p in w
                 if p["ts"] % 7 == 0 or p["ts"] == w[-1]["ts"]]
        grid = [{"ts": p["ts"], "value": 0.0} for p in w]
        lin = B._resample_to_grid(grid, model)
        kw = {"disease": dis, "rev_week": rw if outcome == "regain" else None,
              "declared_gain": slope * max(1, B.END - rw * 7) / 7.0,
              "declared_lost": start - nadir}
        got, meta = B._weight_overlay(lin, model, cid, T, outcome, nadir, **kw)
        B._SKELETON_AUDIT.pop(cid, None)
        out.append({"label": label,
                    "args": {"case_id": cid, "T": T, "outcome": outcome, "nadir": nadir, **kw},
                    "grid": [p["ts"] for p in grid],
                    "model": [[p["ts"], p["value"]] for p in model],
                    "weight": [[p["ts"], p["value"]] for p in got],
                    "skeleton": {"level": meta["level"], "guard_ok": meta["guard_ok"],
                                 "pre": meta["pre"], "regain": meta["regain"]}})
    return out
