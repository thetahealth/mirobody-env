"""Matched confirmation challenges on the generation side; ordinary diagnostic gold."""
from __future__ import annotations

from copy import copy, deepcopy

from haenv.resources.regpath import load_registry
import haenv.world.generation.rng as rng


def rules():
    return load_registry("m2_confirmation.yaml")


def plan_of(raw):
    meta = (getattr(raw, "latent_premise", None) or {}).get("meta") or {}
    return ((meta.get("_external") or {}).get("m2") or {}).get("confirmation")


def prepare(cases, prior):
    """Pair different patients before emission, retaining every source family.

    Match the actual regimen, demographics, symptom schedule and course. Each
    member has its own identity and its own draws of records and wording.
    The control keeps its existing independent-event registry gold.
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
    from haenv.world.generation.twins import m2_ddx_of_spec
    view = copy(raw)
    view.adjudication = dict(raw.adjudication)
    view.latent_premise = {**raw.latent_premise, "meta": dict(raw.latent_premise["meta"])}
    ddx, _ = m2_ddx_of_spec(p["independent_spec"], int(raw.prediction_context["prediction_time_T"]))
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
        text = "症状出现时采集十二导联心电图10秒：窦性心律，规则P波，RR间期规则，未记录到房颤。本次阴性不能排除其他时段阵发性心律失常。"
    elif fid == "AHI":
        value = sample("positive_ahi" if present else "negative_ahi")
        return [{**base, "value": value, "unit": "次/小时", "ref_low": 0, "ref_high": 5,
                 "flag": "H" if present else "NORMAL", "details": {
                     "acquisition": "attended_polysomnography", "technically_adequate": True,
                     "report": "规范多导睡眠监测，呼吸事件由睡眠专科复核；阴性且持续高度怀疑时需考虑复评。"}}]
    elif fid == "BiliaryUltrasound":
        text = ("空腹胆囊超声：腔内可移动强回声伴后方声影，胆囊壁增厚，探头局部压痛；结合当前病程评估。" if present
                else "空腹胆囊超声：未见结石，胆囊壁不厚，无周围积液，超声Murphy征阴性；持续症状应复评其他病因。")
    elif fid == "PostBDObstruction":
        value = sample("positive_ratio" if present else "negative_ratio")
        text = f"按质量要求完成支气管舒张前后肺功能，舒张后FEV1/FVC={value:.2f}；三次可接受呼气，重复性合格。需结合症状及暴露史解释。"
        base["details"] = {"post_bd_fev1_fvc": value, "quality_acceptable": True}
    elif fid == "DepressionAssessment":
        text = ("临床访谈：过去至少两周几乎每日持续低落及兴趣减退，伴睡眠、精力、注意和自我评价改变，工作及家庭功能明显受损；未发现躁狂或轻躁狂史，需按此次症状持续时间与功能损害制定治疗和随访。" if present
                else "临床访谈：睡眠和精力随当日压力波动，无连续两周大部分时间的心境低落或兴趣丧失，主要功能保留；分次处理睡眠与生活压力并随访。")
    elif fid == "ChestFocalOpacity":
        text = ("胸部正侧位X线：右下肺新发局灶气腔实变伴支气管充气征，结合咳嗽发热考虑感染性浸润，需随访吸收情况。" if present
                else "胸部正侧位X线：未见局灶浸润或实变，无胸腔积液；如持续发热、气促或症状加重应复查。")
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
        notes[-1]["text"] += " " + rules()["profiles"][p["profile"]]["context"]
    sid = p["profile"] if p else ddx["spec_id"]
    note = rules()["competition"]["notes"].get(sid)
    if note:
        notes[-1]["text"] += " " + note
    # All text is clinical; the private pairing declaration is never serialized
    # into a public record, header, directory, menu or purchase result.


def twin_fields():
    """Closed truth-family dependencies; no free source may change."""
    return ("latent_premise/meta/_external/m2/confirmation/present",
            "latent_premise/meta/_external/m2/arm", "latent_premise/meta/_external/m2/f2", "latent_premise/meta/ddx",
            "latent_premise/meta/red_flag", "latent_premise/meta/clinician_warranted",
            "latent_premise/meta/outcome_label", "adjudication/ddx", "adjudication/m2_frame",
            "adjudication/m2_workup", "adjudication/open_management",
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
    ddx, flags = m2_ddx_of_spec(sid, int(raw.prediction_context["prediction_time_T"]))
    from haenv.m2.workup import workup_tests
    from haenv.m2.workup import workup_alternatives
    ddx["tests"] = workup_tests(sid)
    meta["ddx"] = deepcopy(ddx)
    meta.update(flags)
    raw.adjudication["ddx"] = ddx
    raw.adjudication["red_flag_present"] = flags["red_flag"]
    raw.adjudication["clinician_action_warranted"] = flags["clinician_warranted"]
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

