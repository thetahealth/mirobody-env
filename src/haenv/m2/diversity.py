"""Sourced M2 generation interventions; no changes to diagnostic judging."""
from __future__ import annotations

from copy import deepcopy

from haenv.resources.regpath import load_registry
import haenv.world.generation.rng as rng


def rules():
    return load_registry("m2_diversity.yaml")


def family(spec):
    from haenv.world.overlay import condition_registry
    kind = condition_registry()[spec]["join_gold"]
    if kind == "independent":
        return "benign"
    if kind == "comorbidity":
        return "comorbid"
    return "atomic" if spec.startswith("HD-UNI-") else "classic"


def prepare(case):
    """Declare the intervention before generating any patient-visible evidence."""
    plan = case["latent"].get("m2")
    if plan is None:
        if not case["latent"].get("m2_bridge"):
            return
        plan = case["latent"]["m2_bridge"] = {}
    plan["diversity"] = {"version": rules()["version"]}


def _sample(cid, salt, interval):
    lo, hi = interval
    return lo + (hi - lo) * rng.unit(cid, "m2-diversity", salt)


def incidental_records(raw, cid):
    """An isolated prior abnormality with complete follow-up, on both role sides.

    These are dated outside reports, not overrides of the current assays. A
    plausible exposure is only a candidate explanation: the repeat result and
    its timing, rather than the exposure's presence, establish resolution.
    """
    from haenv.resources.registry import load_findings
    from haenv.world.observations.findings_render import _band
    from haenv.world.observations.findings_render import _ndigits
    from haenv.m2.core import _sex_spec
    policy = rules()["incidental"]
    from haenv.m2.confirmation import plan_of
    confirmation = plan_of(raw)
    matched = "|".join(confirmation["pair"]) if confirmation else cid
    f = rng.pick(policy["factors"], matched, "m2-diversity", "factor")
    active = rng.unit(matched, "m2-diversity", "role") < policy["acting_share"]["value"]
    T = int(raw.prediction_context["prediction_time_T"])
    day = T - round(_sample(cid, "draw-day", policy["day_before_T"]))
    repeat = day + round(_sample(cid, "repeat-day", policy["repeat_days"]))
    fid = f["finding"]
    spec = _sex_spec(raw, fid, load_findings()[fid]) or load_findings()[fid]
    lo, hi = _band(spec)
    precision = _ndigits(fid)
    if precision is None:
        raise ValueError(f"M2 incidental finding lacks registered precision: {fid}")
    high = (_sample(cid, "abnormal", f["high_ratio"]) * hi if "high_ratio" in f
            else _sample(cid, "abnormal", f["high_value"]))
    normal = (_sample(cid, "repeat", f["normal_ratio"]) * hi if "normal_ratio" in f
              else _sample(cid, "repeat", f["normal_value"]))
    def report(d, value):
        return {"day": d, "source": "外院检验报告", "finding": fid,
                "name": spec.get("name_cn", fid), "value": round(value, precision),
                "unit": spec.get("unit", ""), "ref_low": lo, "ref_high": hi}
    labs = [report(day, high), report(repeat, normal)]
    exposure_day = day - 2 if active else repeat + 1
    notes = [{"day": exposure_day, "source": "门诊问诊", "text": rng.pick(f["factor_text"], cid, "m2-diversity", "wording")},
             {"day": repeat, "source": "门诊复查记录",
              "text": f["active_context"] if active else f["inactive_context"],
              "followup": f["repeat_note"]}]
    clinical = [e for e in raw.evidence_ledger if e.get("symptom")
                and int(e.get("source_timestamp", T + 1)) <= T]
    count = rng.pick(policy["supplementary_encounters"], cid, "m2-diversity", "encounters")
    if len(clinical) < count:
        raise ValueError("M2 supplemental encounters require actual recorded patient events")
    sampled = sorted(clinical, key=lambda e: rng.unit(cid, "m2-diversity", "source", e["evidence_id"]))[:count]
    notes += [{"day": int(e["source_timestamp"]), "source": "门诊补充问诊", "text": e["symptom"]}
              for e in sampled]
    # Public source names and layouts do not encode factor role or diagnosis.
    raw.prediction_context["outpatient_laboratory_record"] = labs
    raw.prediction_context["outpatient_encounter_record"] = notes
    return {"factor": f["id"], "acting": active, "finding": fid,
            "draw_day": day, "repeat_day": repeat, "source": f["source"]}


def overlapping_presentation(raw, cid):
    """Distribute nonspecific symptoms and clinician history across dated sources.

    The same complaint bank covers competing diagnoses. One original detail is
    retained in a separate encounter, but phenotype alone cannot replace the
    existing confirmatory assays. The original patient truth is left intact.
    """
    from haenv.world.observations.events import symptom_lay_every
    from haenv.m2 import core
    ddx = raw.adjudication["ddx"]
    comps = core.components(ddx["spec_id"])
    ways = symptom_lay_every()
    from haenv.world.overlay import condition_registry
    specs = condition_registry()
    ledger = raw.evidence_ledger
    notes = raw.prediction_context["outpatient_encounter_record"]
    external = ((raw.latent_premise.get("meta") or {}).get("_external") or {})
    explained = ((external.get("m2") or {}).get("explain_away") or {}).get("lines")
    if explained:
        # Existing clause-level explain-away evidence must retain its licensed
        # symptom wording and remainder. Incidental reports still apply.
        return {"changed_complaints": 0, "dispersed_details": 0}
    changed, moved, competing = {}, [], []
    for group, rule in rules()["presentations"]["groups"].items():
        for component in sorted(set(comps) & set(rule["components"])):
            phrases = [s[1] for s in specs[component]["symptoms"]]
            keep = int(rng.unit(cid, "m2-diversity", "detail", component) * len(phrases))
            for i, phrase in enumerate(phrases):
                variants = {core._norm(v) for v in (phrase, *ways.get(phrase, ())) }
                for event in ledger:
                    if core._norm(event.get("symptom", "")) not in variants:
                        continue
                    old = event["symptom"]
                    event["symptom"] = rng.pick(rule["complaints"][i % len(rule["complaints"])], cid,
                                                 "m2-diversity", "complaint", component, i)
                    changed[old] = event["symptom"]
                    if i == keep:
                        day = int(event["source_timestamp"])
                        moved.append({"day": day, "source": "门诊补充问诊", "text": old})
            competing.append(rule["competing"])
    for record in raw.prediction_context.get("event_history_record", ()):
        if record.get("symptom") in changed:
            record["symptom"] = changed[record["symptom"]]
    for record in notes[2:]:
        if record["text"] in changed:
            record["text"] = changed[record["text"]]
    # Use already-drawn encounter slots for retained details. Counts and source
    # categories are sampled before looking at clinical components; an inert
    # patient has real symptom records in the very same slots.
    for i, detail in enumerate(moved):
        if i >= len(notes) - 2:
            raise ValueError("M2 supplemental encounter supply cannot hold clinical details")
        notes[2 + i] = detail
    if competing:
        notes[-1]["text"] += " " + " ".join(dict.fromkeys(competing))
    return {"changed_complaints": len(changed), "dispersed_details": len(moved)}


def finalize(raw, cid):
    external = ((getattr(raw, "latent_premise", None) or {}).get("meta") or {}).get("_external") or {}
    plan = external.get("m2")
    if not plan:
        bridge = external.get("m2_bridge")
        plan = bridge if isinstance(bridge, dict) else None
    if not plan or not plan.get("diversity"):
        return
    gold = raw.adjudication.get("m2_frame", raw.adjudication.get("history_dependency"))
    if gold is None:
        raise ValueError("M2 diversity requires its realized clinical/history block")
    if "diversity" in gold:
        return
    before = deepcopy(raw.adjudication["ddx"])
    evidence = incidental_records(raw, cid)
    evidence.update(overlapping_presentation(raw, cid))
    from haenv.m2.confirmation import finalize as confirm
    confirm(raw)
    if raw.adjudication["ddx"] != before:
        raise ValueError("M2 diversity intervention changed diagnostic gold")
    gold["diversity"] = evidence
