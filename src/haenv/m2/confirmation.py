"""Matched confirmation challenges on the generation side; ordinary diagnostic gold."""
from __future__ import annotations

from haenv.i18n.text import J, L, T as tpl

from copy import copy, deepcopy

from haenv.resources.regpath import load_registry
import haenv.world.generation.rng as rng


def rules():
    return load_registry("m2_confirmation.yaml")


def plan_of(raw):
    meta = (getattr(raw, "latent_premise", None) or {}).get("meta") or {}
    return ((meta.get("_external") or {}).get("m2") or {}).get("confirmation")


def control_ddx(plan_or_profile, independent_spec, T):
    """Independent registry gold with the matched presentation's sourced label."""
    from haenv.world.generation.twins import m2_ddx_of_spec
    profile = (plan_or_profile.get("profile", plan_or_profile)
               if isinstance(plan_or_profile, dict) else plan_or_profile)
    rule = rules()["profiles"][profile] if isinstance(profile, str) else profile
    ddx, flags = m2_ddx_of_spec(independent_spec, T)
    ddx["diagnosis"] = L(rule["control_label"])
    return ddx, flags


def emission(raw, cs, sp, T):
    p = plan_of(raw)
    if p and p["present"] is False:
        from haenv.i18n.identity import source
        expected, _ = control_ddx(p, p["independent_spec"], T)
        if source(raw.adjudication.get("ddx", {}).get("diagnosis")) != expected["diagnosis"]:
            return [{"kind": "m2_control_label", "severity": "gate",
                     "detail": "confirmation control must carry its profile control label"}]
    return []


def gold_derivations():
    from haenv.contracts import GoldDerivation
    from haenv.world.wq import resolve_path
    def applies(w, field):
        p = resolve_path(w, "W.latent_premise.meta._external.m2.confirmation")
        return field == "diagnosis" and isinstance(p, dict) and p.get("present") is False
    def derive(w, field, value, deriv):
        p = resolve_path(w, "W.latent_premise.meta._external.m2.confirmation")
        T = int(resolve_path(w, "W.prediction_context.prediction_time_T"))
        ddx, _ = control_ddx(p, p["independent_spec"], T)
        return ddx["diagnosis"], deriv + ["registry/m2_confirmation.yaml: profile control_label"]
    return (GoldDerivation("m2.confirmation_control", applies, derive),)


def prepare(cases, prior):
    """Pair different patients before emission, retaining every source family.

    Match the actual regimen, demographics, symptom schedule and course. Each
    member has its own identity and its own draws of records and wording.
    The control retains independent structure with its presentation-specific label.
    """
    from haenv.m2.core import scene_text
    profiles = rules()["profiles"]
    for c in cases:
        if c["latent"].get("m2"):
            c["latent"]["m2"]["confirmation_cohort"] = rules()["version"]
    negatives = sorted([c for c in cases if c["latent"].get("ddx_join_gold") == "independent"],
                       key=lambda c: rng.unit(c["case_id"], "confirmation-pair"))
    donors = sorted([c for c in cases if c["latent"].get("ddx_spec_id") in profiles
                     and c["latent"].get("m2") and not c["latent"].get("ddx_insufficient")],
                    key=lambda c: (c["latent"]["ddx_spec_id"], rng.unit(c["case_id"], "confirmation-pair")))
    if len(negatives) < len(donors):
        raise ValueError("confirmation pair supply needs more independent patients")
    paired_controls = set()
    # Round-robin the existing benign families; paired matching must not delete
    # a control family just because one family supplies more candidate rows.
    by_spec = {}
    for c in negatives:
        by_spec.setdefault(c["latent"]["ddx_spec_id"], []).append(c)
    negatives = []
    while any(by_spec.values()):
        for sid in sorted(by_spec):
            if by_spec[sid]:
                negatives.append(by_spec[sid].pop())
    profile_index = {}
    for positive, negative in zip(donors, negatives):
        sid = positive["latent"]["ddx_spec_id"]
        profile = profiles[sid]
        if profile.get("frame") == "F3":
            from haenv.world.generation.build import clinical_plan
            from haenv.m2.core import f3_plan_for
            r, lat = positive["raw"], positive["latent"]
            live = list(clinical_plan(r["disease"], r["devices"], r.get("comorbidities") or [], positive["case_id"]))
            followup = f3_plan_for(positive["case_id"], sid, "positive", live, False, r["disease"])
            if followup is None:
                continue
            lat["m2"].update(frame="F3", f3=followup)
        index = profile_index.get(sid, 0)
        profile_index[sid] = index + 1
        if sid == "JD-OSA":
            # Include both sexes in the eligible matched follow-up cohort.
            # A missing routine follow-up plan must not consume a sex slot.
            positive["raw"]["sex"] = "F" if index % 2 == 0 else "M"
        control_gold = {k: deepcopy(v) for k, v in negative["latent"].items() if k.startswith("ddx_")}
        negative["raw"] = deepcopy(positive["raw"])
        negative["latent"] = deepcopy(positive["latent"])
        negative["latent"].update(control_gold)
        ddx, flags = control_ddx(sid, control_gold["ddx_spec_id"], int(negative["latent"]["index_time_T"]))
        negative["latent"].update({f"ddx_{k}": v for k, v in ddx.items()})
        negative["latent"].update({f"ddx_{k}": v for k, v in flags.items()})
        pair = sorted([positive["case_id"], negative["case_id"]])
        for c, present in ((positive, True), (negative, False)):
            lat = c["latent"]
            m2 = lat["m2"]
            m2.pop("acute_event", None)
            m2.pop("explain_away", None)
            m2.pop("ea_twin", None)
            m2.pop("f2", None)
            m2.pop("critical", None)
            m2["arm"] = "positive" if present else "negative"
            m2["confirmation"] = {"version": rules()["version"], "profile": sid,
                "present": present, "independent_spec": control_gold["ddx_spec_id"], "pair": pair}
            for i, s in enumerate(c["raw"]["symptoms"]):
                if i >= len(profiles[sid]["complaints"]):
                    continue  # retain an existing source's diagnostic history detail
                variants = profiles[sid]["complaints"][i]
                s["text"] = rng.pick(variants, c["case_id"], "confirmation-wording", i)
                s["context"] = ""
            if m2["frame"] in ("F2", "F3"):
                from haenv.m2.symptom_timing import spread_symptom_days
                days = spread_symptom_days(positive["case_id"], len(c["raw"]["symptoms"]), int(lat["index_time_T"]))
                for s, day in zip(c["raw"]["symptoms"], days):
                    s["day"] = day
            m2["scene"] = scene_text(m2["frame"], c["raw"], lat, c["case_id"])
            from haenv.m2.diversity import prepare as prepare_diversity
            prepare_diversity(c)
            prior[c["case_id"]] = {**prior[positive["case_id"]], "frame": m2["frame"]}
        if positive["raw"]["symptoms"] == negative["raw"]["symptoms"]:
            # Resample a phrase, not a patient's clinical state, if two
            # independent word draws happened to produce the complete bundle.
            i = rng.below(len(profile["complaints"]), negative["case_id"], "confirmation-reword")
            old = negative["raw"]["symptoms"][i]["text"]
            negative["raw"]["symptoms"][i]["text"] = next(t for t in profile["complaints"][i] if t != old)
            lat = negative["latent"]
            lat["m2"]["scene"] = scene_text(lat["m2"]["frame"], negative["raw"], lat, negative["case_id"])
        paired_controls.add(negative["case_id"])
    return [c for c in cases if c["latent"].get("ddx_join_gold") != "independent" or c["case_id"] in paired_controls]


def _control_view(raw, p):
    view = copy(raw)
    view.adjudication = dict(raw.adjudication)
    view.latent_premise = {**raw.latent_premise, "meta": dict(raw.latent_premise["meta"])}
    ddx, _ = control_ddx(p, p["independent_spec"], int(raw.prediction_context["prediction_time_T"]))
    view.adjudication["ddx"] = ddx
    view.latent_premise["meta"]["ddx"] = deepcopy(ddx)
    return view


def result(raw, fid, T):
    """One seeded, acquired result from a predeclared clinical state."""
    p = plan_of(raw)
    if not p or fid != rules()["profiles"][p["profile"]]["finding"]:
        return None
    r = rules()["profiles"][p["profile"]]
    present = p["present"]
    if not isinstance(present, bool):
        raise ValueError("confirmation presence must be boolean")
    base = {"ts": T, "finding_id": fid, "name": r["test"],
            "flag": "POSITIVE" if present else "NEGATIVE"}
    sample = lambda key: round(r[key][0] + (r[key][1] - r[key][0]) * rng.unit(raw.case_id, "confirmation-result", key), 2)
    if fid == "ECGAtrialPattern":
        if present:
            # Reuse the existing, sourced acquisition model, not an ambulatory
            # result invented from a ten-second resting ECG.
            from haenv.world.clinical.clinical_evidence import clinical_result_series
            from haenv.m2.core import _profiles
            from haenv.m2.core import _findings
            return clinical_result_series(raw.case_id, p["profile"], fid, T, _findings(),
                                          _profiles()[p["profile"]], raw.user_profile)
        text = L("m2.confirm.ecg.negative")
    elif fid == "AHI":
        value = sample("positive_ahi" if present else "negative_ahi")
        return [{**base, "value": value, "unit": L("m2.unit.per_hour"), "ref_low": 0, "ref_high": 5,
                 "flag": "H" if present else "NORMAL", "details": {
                     "acquisition": "attended_polysomnography", "technically_adequate": True,
                     "report": L("m2.confirm.psg.report")}}]
    elif fid == "BiliaryUltrasound":
        text = L("m2.confirm.biliary_us.positive" if present else "m2.confirm.biliary_us.negative")
    elif fid == "PostBDObstruction":
        value = sample("positive_ratio" if present else "negative_ratio")
        text = tpl("m2.confirm.spirometry", ratio=value)
        base["details"] = {"post_bd_fev1_fvc": value, "quality_acceptable": True}
    elif fid == "DepressionAssessment":
        text = L("m2.confirm.depression.positive" if present else "m2.confirm.depression.negative")
    elif fid == "ChestFocalOpacity":
        text = L("m2.confirm.cxr.positive" if present else "m2.confirm.cxr.negative")
    else:
        raise ValueError(f"confirmation finding has no acquisition model: {fid}")
    return [{**base, "value": text}]


def install_results():
    from haenv.m2 import core
    def synth(next_call, orig, raw, target, T):
        p = plan_of(raw)
        if not p:
            return next_call(orig, raw, target, T)
        from haenv.interaction.gated import _resolve_target
        fid = _resolve_target(str(target).strip(), core._findings())
        decisive = result(raw, fid, T)
        # Common background and exclusion tests on both sides. The acquired
        # decisive assay is the only diagnostic-state-dependent result.
        return decisive if decisive is not None else next_call(orig, _control_view(raw, p), target, T)

    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    register_hook(Stage.M2_FINDING, "m2:confirmation", synth)


def finalize(raw):
    """Shared candidate explanations, plus sourced within-system rivals."""
    p = plan_of(raw)
    ddx = raw.adjudication["ddx"]
    notes = raw.prediction_context.get("outpatient_encounter_record")
    if not notes:
        return
    if p:
        notes[-1]["text"] = J("space", [notes[-1]["text"], rules()["profiles"][p["profile"]]["context"]])
    sid = p["profile"] if p else ddx["spec_id"]
    note = rules()["competition"]["notes"].get(sid)
    if note:
        notes[-1]["text"] = J("space", [notes[-1]["text"], note])
    # All text is clinical; the private pairing declaration is never serialized
    # into a public record, header, directory, menu or purchase result.


def twin_fields():
    """Closed truth-family dependencies; no free source may change."""
    return ("latent_premise/meta/_external/m2/confirmation/present",
            "latent_premise/meta/_external/m2/arm", "latent_premise/meta/_external/m2/f2", "latent_premise/meta/ddx",
            "latent_premise/meta/red_flag", "latent_premise/meta/clinician_warranted",
            "latent_premise/meta/outcome_label", "outcome_label", "adjudication/ddx", "adjudication/m2_frame",
            "adjudication/m2_workup", "adjudication/m2_evidence", "adjudication/open_management",
            "adjudication/red_flag_present", "adjudication/clinician_action_warranted")


def mutate_twin(raw, declaration):
    from haenv.world.generation.twins import m2_ddx_of_spec
    from haenv.world.generation.twins import _m2_recomputed
    p = plan_of(raw)
    if not p or p["present"] != declaration["before"] or not isinstance(declaration["after"], bool):
        raise ValueError("confirmation intervention does not match its source")
    p["present"] = declaration["after"]
    meta = raw.latent_premise["meta"]
    meta["_external"]["m2"]["arm"] = "positive" if p["present"] else "negative"
    sid = p["profile"] if p["present"] else p["independent_spec"]
    T = int(raw.prediction_context["prediction_time_T"])
    ddx, flags = (m2_ddx_of_spec(sid, T) if p["present"]
                  else control_ddx(p, sid, T))
    from haenv.m2.workup import workup_tests
    from haenv.m2.workup import workup_alternatives
    ddx["tests"] = workup_tests(sid)
    meta["ddx"] = deepcopy(ddx)
    meta.update(flags)
    raw.adjudication["ddx"] = ddx
    raw.adjudication["red_flag_present"] = flags["red_flag"]
    raw.adjudication["clinician_action_warranted"] = flags["clinician_warranted"]
    raw.outcome_label = flags["outcome_label"]
    if meta["_external"]["m2"]["frame"] == "F2":
        from haenv.m2.core import f2_direction
        basics = raw.latent_premise["patient_basics"]
        meta["_external"]["m2"]["f2"] = f2_direction(
            raw, basics["disease"], int(raw.prediction_context["prediction_time_T"]),
            basics["regimen"]["drug"], sid if p["present"] else None) or {}
    raw.adjudication["m2_workup"]["alternatives"] = {
        t: workup_alternatives(t) for t in ddx["tests"] if workup_alternatives(t)}
    reproduced = _m2_recomputed(raw, derive_workup=True)
    if reproduced.prediction_context != raw.prediction_context:
        raise ValueError("confirmation intervention changed a free source")
    raw.adjudication = reproduced.adjudication

