"""Expose the existing follow-up policy from its single executable source."""
from __future__ import annotations


def _example(cls, cause=None, *, previous=100, today=120, prior=False, baseline=None):
    rows = [{"day": 0, "printed": previous, "perturbations": [{"acts": True}] if prior else []},
            {"day": 1, "printed": today, "perturbations": [] if prior else [{"acts": True}]}]
    if baseline is not None:
        rows.insert(0, {"day": -1, "printed": baseline})
    return {"class": cls, "cause": cause, "t0": 0, "t1": 1, "records": rows}


def steps():
    from haenv.p4.gold import ACEI_RISE_MAX_PCT
    from haenv.p4.gold import next_step
    from haenv.i18n.text import L, T
    n = f"{ACEI_RISE_MAX_PCT:g}%"
    cases = [
        (L("p4.followup.case.noise"), _example("analytic_biological_noise")),
        (L("p4.followup.case.distortion_previous"), _example("preanalytical", prior=True)),
        (L("p4.followup.case.distortion_today"), _example("method_difference")),
        (L("p4.followup.case.acei_stop_fall"), _example("true_change", "acei_start_or_stop", today=90)),
        (T("p4.followup.case.acei_start_within", n=n), _example("true_change", "acei_start_or_stop")),
        (T("p4.followup.case.acei_start_baseline_within", n=n),
         _example("true_change", "acei_start_or_stop", today=140, baseline=120)),
        (T("p4.followup.case.other_true_change", n=n),
         _example("true_change", "acei_start_or_stop", today=140)),
    ]
    for ended in (False, True):
        blk = _example("method_difference", "biotin_tsh")
        blk.update(factor="biotin_tsh", review_day=3,
                   ingested_exposure={"end_day": 2 if ended else None, "clean_followup": True})
        cases.append((L("p4.followup.case.ingested_ended" if ended else "p4.followup.case.ingested_ongoing"), blk))
    return [T("p4.followup.step", condition=condition, primary=primary, label=L(f"p4.followup.next.{primary}"),
              accepted=" / ".join(accepted))
            for condition, blk in cases for primary, accepted in [next_step(blk)]]


def _is_acei(drug) -> bool:
    """An ACEI row of the p4 medication record (its name template, in any language)."""
    from haenv.p4.features import _is_acei as is_acei
    return is_acei(drug)


def visible_step(sp, cls, candidates):
    """Only visible report values and medication dates enter the policy reader."""
    from haenv.p4.gold import next_step
    b = sp.prediction_context["followup_p4"]
    side = next((c[1] for c in candidates if c[0] == cls), None)
    cause = next((c[2] for c in candidates if c[0] == cls), None)
    rows = [{"day": int(r["day"]), "printed": r["value"],
             "perturbations": [{"acts": True}] if cls in ("preanalytical", "method_difference") and
             int(r["day"]) == int(b["previous_draw_day"] if side == -1 else b["today"]) else []}
            for r in b["report"]]
    blk = {"class": cls, "cause": cause, "t0": b["previous_draw_day"], "t1": b["today"], "records": rows,
           "acei": {"events": [{"day": r["day"]} for r in b.get("medication_record") or [] if _is_acei(r["drug"])]}}
    return next_step(blk)[0]
