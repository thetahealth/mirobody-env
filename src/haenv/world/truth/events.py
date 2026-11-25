"""Case-owned, sourced event and vital effects. SYNTHETIC, evaluation only."""
from __future__ import annotations

import copy

from haenv.world.generation.rng import unit
from haenv.world.truth import plan_of
from haenv.world.truth.registry import effects


def _schedule(raw):
    from haenv.world.observations.events_text import symptom_topic
    return [{"day": e["source_timestamp"], "end_day": e.get("end_day", e["source_timestamp"]),
             "evidence_id": e.get("evidence_id"), "symptom": e.get("symptom", ""),
             "topic": symptom_topic(e.get("symptom", "")), "kind": "real_symptom"}
            for e in getattr(raw, "evidence_ledger", ()) or ()
            if e.get("source_type") == "patient_reported_symptom"]


def _episodes(schedule, topics, *, genuine=False):
    # A report without an explicit endpoint describes its own day. Never invent a window.
    return sorted({(int(e["day"]), int(e.get("end_day", e["day"])))
                   for e in schedule if e.get("topic") in topics
                   and (not genuine or e.get("kind") == "real_symptom")})


def fever_days(raw, schedule=None):
    """The symptom schedule's fever days, including the registered URI fever topic."""
    from haenv.i18n.identity import source
    texts = {e.get("evidence_id"): source(e.get("symptom", ""))
             for e in getattr(raw, "evidence_ledger", ()) or ()}
    schedule = _schedule(raw) if schedule is None else schedule
    # URI is shared by febrile and cough-only complaints. Read the canonical source phrase,
    # never the rendered language, and do not turn a cough-only episode into fever.
    rows = [e for e in schedule if e.get("topic") == "low_grade_fever"
            or e.get("topic") == "acute_uri" and any(
                w in str(texts.get(e.get("evidence_id"), e.get("symptom", "")))
                for w in ("发热", "低热", "发烧", "低烧", "高烧"))]
    return _episodes(rows, {"low_grade_fever", "acute_uri"})


def hf_episodes(raw, schedule=None):
    """Dyspnoea/oedema episodes only in a case with heart failure."""
    from haenv.m2.core import components
    lp = getattr(raw, "latent_premise", None) or {}
    pb = lp.get("patient_basics") or {}
    sid = ((getattr(raw, "adjudication", None) or {}).get("ddx") or {}).get("spec_id", "")
    known = set((getattr(raw, "user_profile", None) or {}).get("known_conditions") or ())
    known.update(pb.get("comorbidities") or ())
    known.add(pb.get("disease"))
    if "HD-UNI-07" not in components(sid) and "heart_failure" not in known:
        return []
    return _episodes(_schedule(raw) if schedule is None else schedule,
                     {"exertional_limitation", "nocturnal_awakening"}, genuine=True)


def weight_protected(raw):
    """A weight-label case must keep every forbidden gold signal unchanged."""
    from haenv.world.clinical.drug_effects import FORBIDDEN
    from haenv.resources.registry import GOLD_EVIDENCE
    adj = getattr(raw, "adjudication", None) or {}
    # Diagnosis/triage tasks do not grade the weight outcome. All other packs are protected,
    # even when their driver happens to use a different auxiliary evidence stream.
    diagnosis = bool(adj.get("ddx"))
    driver = adj.get("driver")
    signal = (GOLD_EVIDENCE.get(driver) or {}).get("signal")
    return not diagnosis or signal in FORBIDDEN


def make_plan(cid, *, baseline_temp, fever=(), hf=(), medication=None, protect_weight=False):
    """Snapshot the registered parameters and medication timeline, without reading a switch."""
    doc = effects()
    events = doc.get("events") or {}
    out = {"baseline_temp": baseline_temp, "fever": [], "hf": [], "drug_vitals": [],
           "fever_effect": copy.deepcopy(events.get("fever") or {})}
    row = events.get("fever") or {}
    if row.get("body_temp") and row.get("applied", True) and row["body_temp"].get("applied", True):
        for start, end in fever:
            out["fever"].append({"start": start, "end": end, **copy.deepcopy(row)})
    row = events.get("hf_decompensation") or {}
    weight = row.get("weight") or {}
    if weight and row.get("applied", True) and weight.get("applied", True) and not protect_weight:
        lo, hi = weight["gain_kg_range"]
        for start, end in hf:
            out["hf"].append({"start": start, "end": end,
                              "gain_kg": lo + (hi - lo) * unit(str(cid), "truth-hf-weight", start),
                              **copy.deepcopy(weight)})
    wm = medication or {}
    seen = set()
    for med in [wm.get("primary"), *(wm.get("concurrent") or ())]:
        if not med or med.get("drug") in seen:
            continue
        drug = med.get("drug")
        seen.add(drug)
        row = (doc.get("drug_vitals") or {}).get(drug) or {}
        if row and row.get("applied", True):
            out["drug_vitals"].append({"drug": drug, "effects": copy.deepcopy(row),
                                       "dose_timeline": copy.deepcopy(med.get("dose_timeline") or [])})
    return out


def prepare(raw, schedule, *, baseline_temp):
    """Store event truth after the complaint schedule has been finalized (generation only)."""
    lp = raw.latent_premise
    plan = lp.setdefault("truth", {"version": 1, "observables": {}, "constraints": {}})
    event_plan = make_plan(raw.case_id, baseline_temp=baseline_temp,
                          fever=fever_days(raw, schedule), hf=hf_episodes(raw, schedule),
                          medication=lp.get("world_medication"), protect_weight=weight_protected(raw))
    plan["events"] = event_plan
    # The p2 renderer receives this existing plan object, rather than the raw case. Keep its
    # event reference here as well so it can use the same saved parameters without a global cache.
    from haenv.contracts.external_gold import SLOT
    p2 = (((lp.get("meta") or {}).get(SLOT) or {}).get("p2"))
    if isinstance(p2, dict):
        p2["event_truth"] = event_plan
    return event_plan


def event_plan_of(raw):
    return (plan_of(raw) or {}).get("events")


def stream_range(plan, sig, day, hard_range, tol):
    """Union the ordinary range with its sourced event shift and existing noise envelope.

    Saved plans exist only for coupled cases. Queries never consult configuration or
    today's effect registry; inactive or unsourced rows cannot extend a bound.
    """
    lo, hi = hard_range
    if not plan or sig not in ("body_temp", "resting_hr"):
        return lo, hi
    fever = [e for e in plan.get("fever", ()) if e.get("applied", True)
             and e["start"] <= day <= e["end"]
             and (e.get("body_temp") or {}).get("source")
             and e["body_temp"].get("applied", True)]
    if sig == "resting_hr":
        fever = [e for e in fever if (e.get(sig) or {}).get("source")
                 and e[sig].get("applied", True)]
    if not fever:
        return lo, hi
    delta = increment({**plan, "fever": fever}, sig, day)
    if not delta:
        return lo, hi
    return min(lo, lo + delta - tol), max(hi, hi + delta + tol)


def increment(plan, sig, day):
    """Truth increment on a day; overlapping reports describe one event, not additive fever."""
    if not plan:
        return 0.0
    active = [e for e in plan["fever"] if e["start"] <= day <= e["end"]
              and e.get("applied", True) and e["body_temp"].get("applied", True)]
    if active and sig in ("body_temp", "resting_hr"):
        e = max(active, key=lambda e: e["body_temp"]["level_c"])
        delta = e["body_temp"]["level_c"] - plan["baseline_temp"]
        hr = e.get("resting_hr") or {}
        return delta if sig == "body_temp" else delta * hr.get("per_degree_c", 0) if hr.get("applied", True) else 0.0
    if sig == "weight":
        # Multiple complaints from one HF course must not multiply the fluid gain.
        return max((e["gain_kg"] * min(1.0, max(0.0, (day - e["start"] + e["onset_days_before_episode"])
                    / e["onset_days_before_episode"])) for e in plan["hf"] if day <= e["end"]), default=0.0)
    return 0.0


def drug_vitals(plan, sig, day):
    """A dose step on its own date; no HR kinetics is available in the verified label."""
    from haenv.world.clinical.med_course import dose_at
    total = 0.0
    for med in (plan or {}).get("drug_vitals", ()):
        row = med["effects"].get(sig) or {}
        if not row.get("applied", True):
            continue
        table = row.get("drug_effect_bpm_by_dose_mg") or {}
        dose = dose_at(med["dose_timeline"], day)
        # Only the verified doses have an effect; do not extrapolate a dose response.
        total += float(next((v for k, v in table.items() if float(k) == dose), 0))
    return total


def apply_streams(raw):
    """Add event truth before the existing measurement-noise layer, preserving timestamps."""
    plan = event_plan_of(raw)
    if not plan:
        return
    for sig in ("body_temp", "resting_hr", "weight"):
        for point in (raw.longitudinal_data or {}).get(sig) or ():
            point["value"] += increment(plan, sig, point["ts"]) + drug_vitals(plan, sig, point["ts"])


def replaced_kernel(topic, indicator):
    """Registered replacements plus the temperature target itself, which replaces its old kernel."""
    row = (effects().get("events") or {}).get("fever") or {}
    if not row or not row.get("applied", True):
        return False
    if not (row.get(indicator) or {}).get("applied", True):
        return False
    return ([topic, indicator] in row.get("replaces_kernels", ())
            or topic in ("low_grade_fever", "acute_uri") and indicator == "body_temp")
