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
    from latent import DISEASE_SIGNAL_DOMAIN as DOM          # kernel

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
        # The label rule the build writes into every case (`latent_rules`)
        "label_rule": {"min_change_frac": LR.MIN_CHANGE_FRAC, "min_change_kg": LR.MIN_CHANGE_KG,
                       "smoothing": LR.LABEL_SMOOTHING,
                       "min_persist_days": LR.MIN_PERSIST_DAYS},
    }


def _personas() -> dict:
    """Each persona's random layer, computed by the production functions.

    `shock` starts at `k = -1` (the stationary start), so the browser reads `shock[k+1]`.
    """
    sys.path.insert(0, str(ROOT))
    from haenv import build as B
    from haenv import drug_effects as DE
    from haenv import events as E

    out = {}
    for cid in PERSONAS:
        key = f"{cid}|weight|wobble"
        out[cid] = {
            "shock": [round(E._det_shock(key, k), _SHOCK_ND) for k in range(-1, B.END + 1)],
            # Whether the patient weighed in on day d (used only under daily sampling)
            "weighed": [1 if B._weighed_on(d, cid) else 0 for d in range(0, B.END + 1)],
            # Per-draw measurement variation of each clinical signal, by draw index
            "meas": {sig: [round(E._det_shock(f"{cid}|{sig}|meas", i), _SHOCK_ND)
                           for i in range(64)]
                     for sig in sorted(B._clinical_cv())},
            # Desired descent curvature, capped later by `feasibleK`
            "descent_k": B._trajectory_shape(cid)[0],
            # The weight skeleton's uniforms (`build._skeleton_draws`), exact: they steer
            # guard decisions, so they are not rounded
            "skel": B._skeleton_draws(cid),
            # Drug-response multiplier (`drug_effects.response_for`): the draw depends only
            # on `case_id`; the knobs pick the driver band and the drug, so production maps
            # it for every registered drug and for the non-response band.
            "response": {
                "low": DE.response_for(cid, DE.LOW_RESPONSE_DRIVER, ""),
                "by_drug": {n: DE.response_for(cid, None, n) for n in DE.drugs()},
            },
        }
    return out


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
)

#: Drug and dose per golden case (default semaglutide 1.0 mg). A dose-ladder drug exercises
#: the per-rung HbA1c and fasting-glucose fields of `drug_effects.total_effect`.
_GOLDEN_DRUG = {"DEMO-11": ("tirzepatide", 15.0)}


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

    out = []
    for (label, cid, dis, start, nadir, T, outcome, rw, slope, mpw, adh_low, driver,
         devs) in _GOLDEN_CASES:
        drug, dose = _GOLDEN_DRUG.get(cid, ("semaglutide", 1.0))
        # Same call as `build._drug_response_of`
        resp = DE.response_for(cid, driver, drug)
        ce = B.END
        step = max(1, int(round(7.0 / max(0.1, mpw))))
        w, wmeta = B._weight_render(start, nadir, T, outcome, rw, slope,
                                    case_id=cid, step=step, disease=dis, end_day=ce)
        B._SKELETON_AUDIT.pop(cid, None)
        verdict, det = G.derive_outcome({"weight": w}, _rule())
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
        out.append({
            "label": label,
            "params": {"case_id": cid, "disease": dis, "start": start, "nadir": nadir,
                       "T": T, "outcome": outcome, "reversal_week": rw, "regain_slope": slope,
                       "measure_per_week": mpw, "adherence_low": adh_low, "driver": driver,
                       "course_end_day": ce, "drug": drug, "dose_mg": dose,
                       "devices": list(devs)},
            "weight": [[p["ts"], p["value"]] for p in w],
            "skeleton": {"level": wmeta["level"], "guard_ok": wmeta["guard_ok"],
                         "pre": wmeta["pre"], "regain": wmeta["regain"]},
            "rule_readout": {"verdict": verdict,
                      **{k: det[k] for k in ("start", "nadir", "lost", "threshold",
                                             "sustained_days", "reading")}},
            "adherence": [[p["ts"], p["value"]] for p in adh],
            "clinical": clin,
        })
    return {"cases": out, "overlay": _golden_overlay(), "tol": 0.005,
            "source": "haenv/build.py:_weight_render · _weight_overlay · render_clinical · "
                      "clinical_plan · haenv/gates.py:derive_outcome"}


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
