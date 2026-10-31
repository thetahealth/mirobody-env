"""Matched historical reports and allocation, on the generation path only."""
from __future__ import annotations

from haenv.i18n.text import J, L, Msg, T as tpl, numsub
from haenv.world.history.history_dependency import old_day
from haenv.world.history.history_dependency import rate
from haenv.world.history.history_dependency import table
from haenv.world.history.history_dependency import uniform
from haenv.m2.history import BLOCK
from haenv.m2.history import finalize as previous_finalize


def finalize(raw, cid, plan):
    from haenv.world.validation.difficulty_r2 import enabled
    from haenv.world.validation.difficulty_r2 import config
    if not enabled(plan):
        return previous_finalize(raw, cid, plan)
    from haenv.m2.core import threads_of
    from haenv.m2.core import _on_demand
    framed = BLOCK in raw.adjudication
    g = raw.adjudication.get(BLOCK) or raw.adjudication
    ddx = raw.adjudication.get("ddx") or {}
    T = int(raw.prediction_context["prediction_time_T"])
    reusable = set((config("m2") if enabled(plan) else table()["packs"]["m2"])["reusable_findings"])
    keys = sorted({k for t in threads_of(ddx) for k in t["key_signals"]} & reusable)
    # Matched current-episode confirmation cases must not disclose their
    # decisive result through a free historical report. Keep the same existing
    # inactive-report sampler on both sides; other patients retain reuse.
    from haenv.m2.confirmation import plan_of
    if plan_of(raw):
        keys = []
    if enabled(plan):
        # A remote AKI episode is a risk history, not a measurement of the
        # current 48-hour / seven-day rise. Keep it in the control vocabulary.
        keys = [k for k in keys if k != "AKICreatinineChange"]
    cohort = bool((plan or {}).get("confirmation_cohort"))
    active = framed and bool(keys) and rate("m2", plan) > 0 and (cohort or uniform("history-role", cid) < rate("m2", plan))
    fid = (keys[int(uniform("history-item", cid) * len(keys))] if enabled(plan) else keys[0]) if keys else sorted(reusable)[int(uniform("history-item", cid) * len(reusable))]
    phr = table()["phrasing"]
    src = phr["encounter_sources"][int(uniform("history-source", cid) * len(phr["encounter_sources"]))]
    from haenv.resources.registry import load_findings
    spec = load_findings()[fid]
    result = _on_demand(raw, fid, T)
    typing_positive = fid == "HLAB27" and any(r.get("flag") == "POSITIVE" for r in result or ())
    day = old_day("m2", cid, T)
    if result:
        result = archival_result(result, day, T) if enabled(plan) else [{**x, "ts": day} for x in result]
    if not active:
        result = [{"ts": day, "value": L("m2.obs.negative"), "flag": "NEGATIVE", "finding_id": fid, "name": spec.get("name_cn", fid)}]
        if enabled(plan):
            result = [control_observation(fid, spec, cid, day)]
        if typing_positive:
            # A genotype does not seroconvert. An unusable earlier typing
            # report can be inert without contradicting a current positive.
            result[0].update(value=L("m2.obs.typing_unusable"), flag="UNINTERPRETABLE")
    # A historical result is retained as a full laboratory report, with the
    # original finding vocabulary, not an inferred diagnosis or a role label.
    row = {"day": old_day("m2", cid, T), "source": src, "finding": fid,
           "name": spec.get("name_cn", fid), "observation": result,
           "result": L("m2.hist.result.prior_positive" if active else "m2.hist.result.screen_then_negative"),
           "followup": L("m2.hist.followup.not_evaluated" if active else "m2.hist.followup.no_persisting")}
    if not active and typing_positive:
        row.update(result=L("m2.hist.result.typing_failed"),
                   followup=L("m2.hist.followup.typing_incomplete"))
    elif not active and enabled(plan):
        row["result"] = config("m2")["control_phrasing"].get(fid, L("m2.hist.result.control_normal"))
    if active and not result:
        active = False
        row.update(observation=[], result=L("m2.hist.result.uninterpretable_then_negative"), followup=L("m2.hist.followup.none"))
    raw.prediction_context["archived_laboratory_record"] = [row]
    raw.prediction_context["record_policy"] = table()["packs"]["m2"]["protocol"]
    if active:
        ddx["historical_key_signals"] = {fid: "archive:" + fid}
    g["history_dependency"] = {"active": active, "recent_lower": T - 365,
                               "records": [{"section": "archived_laboratory_record", "day": row["day"]}],
                               "source": table()["packs"]["m2"]["source"]}
    if enabled(plan):
        from haenv.world.validation.difficulty_r2 import register
        register(g, "archived_laboratory_record", day, active, recent_lower=T - 365)
        if not framed:
            g["history_dependency"]["look_alike"] = g.pop("look_alike")
        known = raw.user_profile.get("known_conditions") or []
        if (known and can_move_prior_diagnoses(raw, plan)
                and uniform("history-diagnosis-omission", cid) < config("m2")["diagnosis_omission"]["value"]):
            raw.prediction_context["prior_encounter_narrative"] = [{"day": day, "source": src,
                "text": tpl("m2.hist.prior_narrative", conditions=J("enum", known))}]
            raw.user_profile["known_conditions"] = []
            if framed:
                g["diagnosis_narrative_only"] = True
            else:
                g["history_dependency"]["diagnosis_narrative_only"] = True
        elif raw.prediction_context.get("prior_encounter_narrative"):
            # Replaying the source starts with the diagnoses already moved.
            # The retrievable narrative, rather than the now-empty profile,
            # establishes this derived annotation.
            if framed:
                g["diagnosis_narrative_only"] = True
            else:
                g["history_dependency"]["diagnosis_narrative_only"] = True


def can_move_prior_diagnoses(raw, plan=None):
    """Keep the current risk context required by a declared secondary assessment.

    The existing assay renderer requires structured metabolic risk information;
    it must remain available when the current report is generated or purchased.
    An explained-away symptom also requires its established explaining condition
    in the current structured profile, as required by the EA-1 audit. Other past
    diagnoses can be documented solely in the encounter narrative.
    """
    from haenv.resources.registry import condition_findings_for_case
    from haenv.resources.registry import load_findings
    spec = (raw.adjudication.get("ddx") or {}).get("spec_id")
    profile = condition_findings_for_case(load_findings()).get(spec) or {}
    known = set((getattr(raw, "user_profile", None) or {}).get("known_conditions") or ())
    explained = ((plan or {}).get("explain_away") or {}).get("lines") or ()
    needs_known = any(known.intersection(symptom.get("explainer") or ())
                      for line in explained for symptom in line.get("symptoms") or ())
    return not (profile.get("secondary_assessment") or needs_known)



def control_observation(fid, spec, cid, day):
    """Use the existing normal finding sampler for quantitative controls.

    A normal sleep index or antibody titre is a numeric result with its unit,
    rather than an invented qualitative negative. Kidney/ECG controls describe
    only the documented observation, not exclusion of a current episode.
    """
    from haenv.world.validation.difficulty_r2 import config
    row = {"ts": day, "finding_id": fid, "name": spec.get("name_cn", fid)}
    if spec.get("qualitative"):
        return {**row, "value": config("m2")["control_phrasing"].get(fid, L("m2.obs.negative_normal")),
                "flag": "NEGATIVE"}
    from haenv.world.observations.findings_render import _band
    from haenv.world.observations.findings_render import _value
    lo, hi = _band(spec)
    return {**row, "value": _value(spec, "normal", None, cid, fid, day, declared=False),
            "unit": spec.get("unit", ""), "ref_low": lo, "ref_high": hi, "flag": "NORMAL"}



def archival_result(result, day, today):
    """Retain the complete dated observation span in an earlier report.

    Moving only the report timestamp would leave its kidney measurements and
    calendar dates in the current episode. Dates move together; values, units
    and the documented observation interval remain the production result's.
    """
    from datetime import date, timedelta
    shift = int(day) - int(today)
    def move(value):
        if isinstance(value, list):
            return [move(x) for x in value]
        if isinstance(value, dict):
            out = {k: move(v) for k, v in value.items()}
            if isinstance(value.get("ts"), int):
                out["ts"] = value["ts"] + shift
            if isinstance(value.get("date"), str):
                out["date"] = (date.fromisoformat(value["date"]) + timedelta(days=shift)).isoformat()
            return out
        if isinstance(value, Msg) and value.kind == "tpl":
            slots = {k: (v + shift if k.startswith("day") and isinstance(v, int) else move(v))
                     for k, v in value.slots.items()}
            return tpl(value.key, **slots)
        if isinstance(value, Msg) and value.kind == "join":
            return J(value.key, [move(v) for v in value.slots["items"]])
        if isinstance(value, str) and not isinstance(value, Msg):
            # Migration for an externally supplied legacy report (generated
            # observations above already carry typed day slots).
            import re
            pairs = [(m[1], str(int(m[1]) + shift)) for m in re.finditer(r"第(-?\d+)天", value)]
            moved = re.sub(r"第(-?\d+)天", lambda m: f"第{int(m[1]) + shift}天", value)
            return numsub(L(value), moved, pairs) if pairs else value
        return value
    return move(result)



def history_slots(n, *, diversity=True):
    """Allocate existing clinical families, retaining diagnosis/frame/cap constraints."""
    from haenv.world.validation.difficulty_r2 import enabled
    if not enabled():
        return 0
    from haenv.m2.diversity import rules
    from haenv.world.validation.difficulty_r2 import config
    share = (rules()["allocation"]["history_share"]["value"] if diversity
             else config("m2")["allocation_share"]["value"])
    total = round(int(n) * float(share))
    return total

