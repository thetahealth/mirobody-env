"""Pack 3's emission declaration for the shared gate (`haenv/world/validation/shared_audit.py`, SA-1 + SA-8).

On a case with a pack-3 plan the shared gate refuses emission when the item could not be realized,
when G3 re-derived from the records disagrees with the gold (or the visible block is not the
rendering of the records), when the realized class is not the planned one, when the solver payload
carries a gold word or a 4-decimal truth, or when a patient sentence carries a word of the other sex.
The pack callback (P-9, A3/A12/A15) also refuses a case whose payload outside the block names a fact
on the managed drug's `decision_facts` list, or shows a stream that bears on the drug's decision:
a visible fact that changes the decision is read by G3 or the case is not emitted. A condition on the
record that is not a condition of the world (disease + comorbidities) the labs were drawn from is refused.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json

GATE = "p3"
BLOCK = "meds_p3"

EN_TOKENS = ("uptitrate", "downtitrate", "maintain", "switch", "check_adherence_or_adverse_effect",
             "dose_related", "unattributed", "predating", "decoy", "truth", "adherence_true", "subtype",
             "drug_symptom", "top_off", "adh_low", "ae_dose", "ae_min", "med_decision")
EN_TOKENS += ("interaction_family", "control_adherence", "control_administration", "control_limit",
              "control_headroom", "course_control", "symptom_timing", "on_low", "off_ae")
ZH_TOKENS = ("剂量相关", "未归因", "诱饵", "真值", "金标", "依从真值", "过度治疗", "药效未到", "刚加量", "先查依从")


def plan_of(raw) -> dict | None:
    from haenv.contracts.external_gold import SLOT
    lp = raw.get("latent_premise") if isinstance(raw, dict) else getattr(raw, "latent_premise", None)
    p = (((lp or {}).get("meta") or {}).get(SLOT) or {}).get("p3")
    return p if isinstance(p, dict) else None


def _unrealizable(blk: dict) -> str | None:
    if blk.get("unrealizable") or "visits" not in blk:
        return str(blk.get("unrealizable") or "no realized item")
    return None


def _recompute(blk: dict, raw) -> list[str]:
    from haenv.p3.gold import gold_block_consistent
    from haenv.p3.render import render_block
    pc = (raw.get("prediction_context") if isinstance(raw, dict) else getattr(raw, "prediction_context", None)) or {}
    bad = list(gold_block_consistent(blk))
    expected = render_block(blk)
    if blk.get("record_format") == "emr":
        from haenv.p3.emr import source_block
        expected = source_block(expected, blk["current_medications"], blk["patient_quote"])
    if isinstance(pc.get(BLOCK), dict) and expected != pc[BLOCK]:
        bad.append("visible block is not the rendering of the records")
    return bad


def gold_numbers(blk: dict) -> list[str]:
    return [f"{float(r['truth']):.4f}" for r in blk.get("visits") or ()]


def ae_terms_outside(sp, drug_name: str) -> list[str]:
    """Words of the managed drug's `decision_facts` in the solver payload outside the pack-3 block."""
    from haenv.world.labworld.meds import drug
    d = sp if isinstance(sp, dict) else json.loads(sp.dumps())
    d = {**d, "prediction_context": {k: v for k, v in (d.get("prediction_context") or {}).items() if k != BLOCK}}
    text = json.dumps(d, ensure_ascii=False, default=str)
    return [t for t in drug(drug_name).get("decision_facts") or () if t in text]


def decision_facts_visible(blk: dict, sp) -> list[tuple[str, str]]:
    """P-9: a fact that bears on the decision, visible outside what G3 reads."""
    from haenv.world.labworld.meds import drug
    out = []
    ld = (sp.get("longitudinal_data") if isinstance(sp, dict) else getattr(sp, "longitudinal_data", None)) or {}
    seen = sorted(set(ld) & set(drug(blk["drug"]).get("bearing_streams") or ()))
    if seen:
        out.append(("bearing_stream_visible", ", ".join(seen)))
    ae = ae_terms_outside(sp, blk["drug"])
    if ae:
        out.append(("decision_fact_outside_block", ", ".join(ae)))
    return out


def record_vs_world(g: dict, raw) -> list[tuple[str, str]]:
    """A condition on the shown record that is not a condition of the world (disease + comorbidities)
    the labs were drawn from."""
    get = (lambda k: raw.get(k)) if isinstance(raw, dict) else (lambda k: getattr(raw, k, None))
    pb = (get("latent_premise") or {}).get("patient_basics") or {}
    world = {str(pb.get("disease") or "")} | {str(x) for x in pb.get("comorbidities") or ()}
    shown = (get("user_profile") or {}).get("known_conditions") or ()
    out = ([("record_vs_world", c) for c in dict.fromkeys(map(str, shown)) if c not in world]
           + visible_gold_premises(g, raw))
    if "visits" in g:
        from haenv.p3.necessity import failures
        bad = failures(g)
        if bad:
            out.append(("decisive_fact_not_necessary", ", ".join(sorted(bad))))
    return out


def visible_gold_premises(g: dict, raw) -> list[tuple[str, str]]:
    """One emission check for every decision premise, reconstructed from sources.

    In particular, comparing the renderer with its own output cannot detect a
    premise that the renderer systematically omits. Course, onset, setting,
    dose history, target, dispensing, administration and historical tolerance
    must survive source reconstruction independently of that comparison.
    """
    if "visits" not in g:
        return []
    from haenv.p3.emr import clinical_view
    from haenv.p3.features import inputs, _pill_left, _num
    from haenv.p3.gold import dose_at, last_increase, printed_readings, known_of
    from haenv.p3.history import limiting_record
    from haenv.p3.policy import absorption_interaction
    from haenv.p3.textread import canonical, prescription_dose
    from haenv.world.labworld.meds import drug
    from haenv.i18n.text import L
    get = (lambda k: raw.get(k)) if isinstance(raw, dict) else (lambda k: getattr(raw, k, None))
    try:
        b = clinical_view((get("prediction_context") or {})[BLOCK])
        f = inputs(raw)
        T = int(g["T"])
        expected = {"drug": g["drug"], "today": T,
                    "dose": float(dose_at(g["dose_line"], T)),
                    "since_increase": T - last_increase(g["dose_line"], T),
                    "target": float(g["upper"]),
                    "reading": float(next(r["printed"] for r in g["visits"] if int(r["day"]) == T)),
                    "refills": list(g["refills"]), "pill_count": int(g["pill_count"]),
                    "supply": int(g.get("supply", 30)),
                    "absorption": absorption_interaction(g.get("administration_record")),
                    "dose_limit": limiting_record(g.get("adverse_event_record"), g["drug"], f["dose"])}
        observed = {"drug": f["drug"], "today": f["T"], "dose": f["dose"],
                    "since_increase": f["since_increase"], "target": f["upper"], "reading": f["reading"],
                    "refills": [int(r["day"]) for r in b["refill_record"]],
                    "pill_count": _pill_left(b["pill_count_today"]),
                    "supply": int(_num(b["refill_record"][-1]["dispensed"])),
                    "absorption": absorption_interaction(b.get("administration_record")),
                    "dose_limit": limiting_record(b.get("adverse_event_record"), f["drug"], f["dose"])}
        missing = [key for key in expected if expected[key] != observed[key]]
        from haenv.world.labworld.meds import axis
        from haenv.p3.plan import target_upper
        profile = get("user_profile") or {}
        if axis(g["axis"]).get("upper_by_age") and profile.get("age_range"):
            if float(g["upper"]) != target_upper(g["drug"], {"raw": profile}):
                missing.append("target_for_age")
        if abs(f["coverage90"] - printed_readings(g)[0]) > 1e-8:
            missing.append("coverage")
        line = [(int(r["day"]), prescription_dose(r["prescription"])) for r in b["visits"]
                if int(r["day"]) < T and (r["day"] in [int(x[0]) for x in g["dose_line"]])]
        if line != [(int(d), float(v)) for d, v in g["dose_line"]]:
            missing.append("dose_history")
        if known_of(g) and not b.get("known_comorbidity"):
            missing.append("comorbidity_indication")
        histories = {(canonical(r["symptom"]), int(r["onset_day"])): r["history"] for r in b.get("encounter_record") or ()}
        visible = {(canonical(e["symptom"]), int(e["onset_day"])): e for e in b["symptom_diary"]}
        for e in g["diary"]:
            key = (canonical(e["symptom"]), int(e["onset"]))
            record = visible.get(key) or {}
            context = e.get("context") or (L("p3.context.no_trigger") if e.get("kind") == "drug_symptom" else "—")
            if canonical(record.get("status")) != canonical(e.get("status")) or canonical(histories.get(key)) != canonical(context):
                missing.append("symptom_course_onset_context")
        if g.get("administration_record") != b.get("administration_record"):
            missing.append("administration_timing")
        from haenv.p3.history import limiting_inputs
        if limiting_inputs(g.get("adverse_event_record")) != limiting_inputs(b.get("adverse_event_record")):
            missing.append("prior_tolerance_records")
        if set(known_of(g)) != set(f["known"]):
            missing.append("comorbidity_profile")
        # The indication and medicine must occur together in a clinical source.
        if "condition" in b and canonical(b["condition"]) != canonical(drug(g["drug"])["condition"]):
            missing.append("indication")
    except (KeyError, ValueError, TypeError, StopIteration, IndexError) as exc:
        missing = ["unreadable_source: " + str(exc)]
    return [("gold_premise_not_visible", ", ".join(sorted(set(missing))))] if missing else []


def emission():
    from haenv.world.validation.shared_audit import Emission
    return Emission(gate=GATE, block=BLOCK, plan_of=plan_of, recompute=_recompute,
                    class_of=lambda g: g.get("class"), planned_class=lambda p: p.get("class"),
                    words=EN_TOKENS + ZH_TOKENS, numbers=gold_numbers, unrealizable=_unrealizable,
                    visible=decision_facts_visible, extra=record_vs_world)
