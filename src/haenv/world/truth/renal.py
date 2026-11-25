"""Renal template points and reports sharing one saved course.

Template arithmetic and existing message keys match clinical_evidence._renal_bundle;
threshold sources are KDIGO 2012/2024, as registered in effects.yaml.
"""
from __future__ import annotations

from haenv.i18n.text import J, L, T as tpl
from haenv.resources.regpath import load_registry
from haenv.world.clinical import clinical_evidence as CE
from haenv.world.generation import rng
from haenv.world.observations.findings_render import _round
from haenv.world.truth.derived import egfr_ckd_epi_2021


def course_points(cid, sid, T, vocab, profile, user, template):
    """Draw the legacy template once, using its original draws and salts."""
    cr_row, cr_spec = CE._decl(profile, "Cr"), CE._numeric_spec(vocab, "Cr", user)
    chronic = template == "chronic"
    companion = "CKDPersistence" if chronic else "AKICreatinineChange"
    q = CE._decl(profile, companion)
    if q["direction"] not in {"positive", "negative"}:
        raise CE.ClinicalEvidenceUnavailable(f"{companion}: explicit positive/negative design required")
    positive = q["direction"] == "positive"
    if cr_row["direction"] != ("high" if positive else "normal"):
        raise CE.ClinicalEvidenceUnavailable("inconsistent numeric and companion declarations")
    from haenv.world.truth.registry import effects
    row = effects()["renal_course"][template]
    span = row["span_days"]["value"] if chronic else 7
    minimum = 7 if sid == "JD-CKM" else (span + 7 if chronic else 14)
    if T < minimum:
        raise CE.ClinicalEvidenceUnavailable("observation window is too short for the requested history")
    days = [T + offset for offset in row["draw_offsets_days"]["value"]]
    if not chronic:
        base = CE._draw(cr_spec, cr_row, cid, "Cr", 0, direction="normal")
        latest = base
        if positive:
            proposal = CE._draw(cr_spec, cr_row, cid, "Cr", 2)
            latest = _round(max(proposal, 1.6*base,
                                base + 30 + 6*rng.unit(cid, "clinical-result", "delta")), "Cr")
        return {"Cr": {"kind": "course", "points": [[d, v] for d, v in zip(days, [base, base, latest])],
                       "source": [row["draw_offsets_days"]["source"]]}}
    acr_row, acr_spec = CE._decl(profile, "UACR"), CE._numeric_spec(vocab, "UACR", user)
    if acr_row["direction"] != ("high" if positive else "normal"):
        raise CE.ClinicalEvidenceUnavailable("inconsistent albuminuria and persistence declarations")
    if positive:
        first_cr = CE._draw(cr_spec, cr_row, cid, "Cr", 0, magnitude="mild")
        last_cr = CE._draw(cr_spec, cr_row, cid, "Cr", 2)
        if last_cr < first_cr:
            raise CE.ClinicalEvidenceUnavailable("declared renal trend is inconsistent with generated values")
        first_acr = _round(max(row["uacr_first_floor_mg_g"]["value"],
                              CE._draw(acr_spec, acr_row, cid, "UACR", 0, magnitude="mild")), "UACR")
        proposal = CE._draw(acr_spec, acr_row, cid, "UACR", 2)
        lo, hi = row["uacr_ratio_range"]["value"]
        last_acr = _round(max(proposal, first_acr*(lo + (hi-lo)*rng.unit(
            cid, "clinical-result", "albuminuria-span"))), "UACR")
    else:
        first_cr = last_cr = CE._draw(cr_spec, cr_row, cid, "Cr", 0)
        first_acr = last_acr = _round(min(29.0, CE._draw(acr_spec, acr_row, cid, "UACR", 0)), "UACR")
    cr = [first_cr, _round((first_cr+last_cr)/2, "Cr"),
          _round((first_cr+3*last_cr)/4, "Cr"), last_cr]
    acr = [first_acr, _round((first_acr+last_acr)/2, "UACR"),
           _round((2*first_acr+last_acr)/2, "UACR") if positive else last_acr, last_acr]
    return {fid: {"kind": "course", "points": [[d, v] for d, v in zip(days, values)],
                  "source": [row["draw_offsets_days"]["source"]]}
            for fid, values in (("Cr", cr), ("UACR", acr))}


def result(raw, fid, T):
    """Render the saved renal points; no second draw or inferred renal trajectory."""
    plan = raw.latent_premise["truth"]
    obs = plan["observables"]
    if "Cr" not in obs or obs["Cr"]["kind"] != "course":
        return None
    chronic = "UACR" in obs and obs["UACR"]["kind"] == "course"
    companion = "CKDPersistence" if chronic else "AKICreatinineChange"
    if fid not in ("Cr", "UACR" if chronic else companion, companion):
        return None
    from haenv.resources.registry import load_findings
    vocab = load_findings()
    user = dict(raw.user_profile or {})
    cr_spec = CE._numeric_spec(vocab, "Cr", user)
    cr = [CE._point("Cr", L("m2.clinical.phrase_12"), d, v, T, user, spec=cr_spec,
                    details={"egfr": round(egfr_ckd_epi_2021(v, plan["age_years"], plan["sex"]), 1)})
          for d, v in obs["Cr"]["points"]]
    egfr = [{"ts": p["ts"], "egfr": p["details"]["egfr"]} for p in cr]
    if fid == "Cr":
        return [cr[-1]]
    if not chronic:
        base, latest = cr[0]["value"], cr[-1]["value"]
        delta = latest - base
        positive = delta >= 26.5 or latest/base >= 1.5
        report = CE._point(companion, L("m2.clinical.phrase_11"), T,
            tpl("m2.result.aki", day_7=cr[0]["ts"], day_2=cr[1]["ts"], day=T,
                base=base, unit=cr_spec["unit"], latest=latest, delta=delta, ratio=latest/base), T, user,
            details={"creatinine_history": cr, "delta_48h_umol_l": delta,
                     "ratio_to_7d_baseline": latest/base, "elapsed_hours_for_delta": 48,
                     "creatinine_change_observed": positive, "cause_assessment": "not_established",
                     "urine_output": None, "egfr": egfr},
            flag="CHANGE_OBSERVED" if positive else "NO_CHANGE_IN_RECORD")
        return [report]
    acr_spec = CE._numeric_spec(vocab, "UACR", user)
    acr = [CE._point("UACR", L("m2.clinical.phrase_13"), d, v, T, user, spec=acr_spec)
           for d, v in obs["UACR"]["points"]]
    if fid == "UACR":
        return [acr[-1]]
    first, last = acr[0]["value"], acr[-1]["value"]
    positive = all(p["value"] >= 30 for p in acr)
    text = J("m2.report_separator", [tpl("m2.result.renal_reading", day=p["ts"], cr=p["value"],
                 unit=cr_spec["unit"], uacr=a["value"]) for p, a in zip(cr, acr)])
    course = load_registry("clinical_expansion.yaml")["renal_course_observations"]
    report = CE._point(companion, L("m2.clinical.phrase_0"), T,
        tpl("m2.result.ckd_course", readings=text, review=L("m2.renal.acute_review")
            if course["acute_factors_reviewed"] else L("m2.renal.review_missing")), T, user,
        details={"creatinine_history": cr, "uacr_history": acr,
                 "history_span_days": cr[-1]["ts"] - cr[0]["ts"],
                 "albuminuria_persistent_in_record": positive, "uacr_ratio": last/first,
                 "repeat_evaluation_trigger": last >= 2*first,
                 "progression_assessment": "sustained_albuminuria_increase" if positive else "not_observed",
                 "same_method": course["same_method"], "acute_factors_reviewed": course["acute_factors_reviewed"],
                 "egfr": egfr, "etiology": "not_established",
                 "calendar_basis": "112_day_span_exceeds_any_three_calendar_months"},
        flag="PERSISTENT_CHANGE_IN_RECORD" if positive else "CRITERION_NOT_OBSERVED")
    return [report]
