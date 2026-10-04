"""Outcome and gold gates (GEN13, GEN14): the declared outcome must follow from the series by its
label rule, and every gold driver must be observable in what the solver is shown.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import re
from .gates_case import _noise_windows


# ============================================================ GEN13: an outcome must be derivable by its rule
# The declared outcome must match what `label_rule` derives from the trajectory
# (`outcome_declared_not_derived` / `outcome_not_derivable` are gates). Question types
# without an outcome rule of their own report `outcome_rule_not_applicable` (warn).
GEN13_SEVERITY = "warn"


GEN13_GATE_KINDS = {"outcome_declared_not_derived", "outcome_not_derivable"}


def _gen13(kind: str) -> str:
    return "gate" if kind in GEN13_GATE_KINDS else GEN13_SEVERITY


def label_rule_params(label_rule: dict) -> tuple[float, int, list[str]]:
    """Structured `label_rule` parameters; missing ones are parsed from the text once and noted.
    Defaults come from `latent_rules`, shared with `build.py`.
    """
    import re

    from .latent_rules import MIN_CHANGE_FRAC, MIN_PERSIST_DAYS
    notes: list[str] = []
    lr = label_rule or {}
    frac = lr.get("min_change_frac")
    days = lr.get("min_persist_days")
    if frac is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", str(lr.get("minimum_change_magnitude", "")))
        frac = float(m.group(1)) / 100.0 if m else MIN_CHANGE_FRAC
        notes.append(f"min_change_frac parsed from text ({frac}) -- should become a structured field")
    if days is None:
        m = re.search(r"(\d+)\s*w", str(lr.get("minimum_persistence", "")))
        days = int(m.group(1)) * 7 if m else MIN_PERSIST_DAYS
        notes.append(f"min_persist_days parsed from text ({days}) -- should become a structured field")
    return float(frac), int(days), notes


# primary signal -> its clean reference (clinical-grade scale), which artifacts do not touch
CLEAN_REF_OF: dict[str, str] = {"weight": "weight_ref"}


# Noise classes that offset the primary signal permanently from d0; with these, the clean
# reference is used instead of dropping noise windows.
PERSISTENT_OFFSET_NOISE = {"device_switch", "unit_error_persistent"}


def label_rule_floor_kg(label_rule: dict) -> float:
    """The regain threshold's absolute floor (kg). A rule that states none keeps the old
    behaviour, floor 0."""
    v = (label_rule or {}).get("min_change_kg")
    return 0.0 if v is None else float(v)


#: `label_rule.smoothing` values. Nadir, threshold crossing and persistence are read on
#: the smoothed series; a rule without the field reads the raw series.
#: `rolling_median_7pt`: each reading replaced by the median of the 7 readings centred on
#: it (fewer at the two ends of the series), whatever their spacing, so a single low
#: reading cannot set the nadir on a sparse series either. 7 rather than 5: on the shipped
#: packs 5 readings still leave the irregular skeleton about twice the plain skeleton's
#: expected misread on observed series. The two end windows hold only 4 readings, so on a
#: descent the smoothed start sits under the first reading and `lost` reads smaller (on the
#: shipped packs by up to 0.74 kg).
#: Smoothing applies to the primary signal only: the clean reference (`weight_ref`, a clinic
#: scale read every few weeks) is read pointwise, since it carries no day-to-day noise and
#: 7 of its readings span half a year.
SMOOTHING_ROLLING_MEDIAN_7PT = "rolling_median_7pt"


#: `rolling_median_7d`: the median of the readings within 3 days; on a series weighed
#: every 4 days or less often that is one or two readings. Kept for batches that state it.
SMOOTHING_ROLLING_MEDIAN_7D = "rolling_median_7d"


def _median(w: list[float]) -> float:
    w = sorted(w)
    n = len(w)
    return w[n // 2] if n % 2 else 0.5 * (w[n // 2 - 1] + w[n // 2])


def _rolling_median_7pt(pts: list[dict]) -> list[dict]:
    """Each reading replaced by the median of readings i-3..i+3, cut at the series ends
    (`pts` is sorted by day)."""
    v = [float(q["value"]) for q in pts]
    n = len(v)
    return [{"ts": int(q["ts"]), "value": _median(v[max(0, i - 3):min(n, i + 4)])}
            for i, q in enumerate(pts)]


def _rolling_median_7d(pts: list[dict]) -> list[dict]:
    """Each reading replaced by the median of the readings within 3 days of it (`pts` is
    sorted by day)."""
    out, lo, hi = [], 0, 0
    ts = [int(q["ts"]) for q in pts]
    for i, p in enumerate(pts):
        t = ts[i]
        while ts[lo] < t - 3:
            lo += 1
        while hi < len(pts) and ts[hi] <= t + 3:
            hi += 1
        out.append({"ts": t, "value": _median([float(q["value"]) for q in pts[lo:hi]])})
    return out


_SMOOTHERS = {SMOOTHING_ROLLING_MEDIAN_7PT: _rolling_median_7pt,
              SMOOTHING_ROLLING_MEDIAN_7D: _rolling_median_7d}


def label_series(pts: list[dict], label_rule: dict) -> list[dict]:
    """The series `derive_outcome` reads nadir, threshold and persistence on: `pts` sorted
    by day, smoothed as `label_rule.smoothing` states."""
    pts = sorted(pts, key=lambda p: int(p["ts"]))
    fn = _SMOOTHERS.get(str((label_rule or {}).get("smoothing") or ""))
    return fn(pts) if fn else pts


def derive_outcome(longitudinal_data: dict, label_rule: dict,
                   signal: str = "weight",
                   noise_windows: list[tuple[int, int]] | None = None,
                   persistent_offset: bool = False) -> tuple[str | None, dict]:
    """Derive the outcome by running `label_rule` on the trajectory: a rebound from the nadir of
    at least `max(min_change_frac x weight lost, min_change_kg)`, sustained for
    `min_persist_days`; with `smoothing`, all three are read on the smoothed series
    (`label_series`). Persistence is the span in days between the first and the last
    reading of a run at or above the threshold, whatever the gaps between them.
    """
    # Derived from the clean reference, or from the primary signal with noise windows removed:
    # artifacts change the observation, not the true outcome.
    basis = signal
    ref = sorted(longitudinal_data.get(CLEAN_REF_OF.get(signal) or "") or [],
                 key=lambda p: int(p["ts"]))
    if persistent_offset and len(ref) >= 3:
        pts, basis = ref, CLEAN_REF_OF[signal]  # permanent offset -> use the clean reference
    else:
        pts = sorted(longitudinal_data.get(signal) or [], key=lambda p: int(p["ts"]))
        if noise_windows:
            kept = [p for p in pts
                    if not any(a <= int(p["ts"]) <= b for a, b in noise_windows)]
            if len(kept) >= 3:
                pts, basis = kept, f"{signal}(noise windows excluded)"
    if len(pts) < 3:
        return None, {"reason": f"{signal} has too few points ({len(pts)})"}
    frac, persist_days, notes = label_rule_params(label_rule)
    floor_kg = label_rule_floor_kg(label_rule)
    smoothing = str((label_rule or {}).get("smoothing") or "")
    if basis == CLEAN_REF_OF.get(signal):
        smoothing = ""
    else:
        pts = label_series(pts, label_rule)
    start = float(pts[0]["value"])
    i_nadir = min(range(len(pts)), key=lambda i: float(pts[i]["value"]))
    nadir = float(pts[i_nadir]["value"])
    lost = max(0.0, start - nadir)
    thresh = nadir + max(frac * lost, floor_kg)
    # Without a floor a course that lost nothing would read its own wobble as a regain,
    # so the old rule needs a loss. With a floor, a rise of `floor_kg` above the low point
    # counts even when the low point is the first reading.
    has_loss = lost > 0 or floor_kg > 0
    after = pts[i_nadir:]
    best, run_start = 0, None
    for p in after:
        if float(p["value"]) >= thresh and has_loss:
            run_start = p["ts"] if run_start is None else run_start
            best = max(best, int(p["ts"]) - int(run_start))
        else:
            run_start = None
    occurred = has_loss and best >= persist_days
    return ("event_occurred" if occurred else "event_not_occurred"), {
        "start": round(start, 2), "nadir": round(nadir, 2), "lost": round(lost, 2),
        "threshold": round(thresh, 2), "sustained_days": best,
        "need_days": persist_days, "frac": frac, "floor_kg": floor_kg,
        "smoothing": smoothing or "none", "basis": basis, "notes": notes,
        # With less lost than the floor, the rule is a gain of `floor_kg` above the low
        # point, sustained: a weight gain, not a rebound from a loss.
        "reading": "gain_from_low_point" if floor_kg > 0 and lost < floor_kg else "rebound"}


def check_outcome_derivable(raw, cs=None) -> list[dict]:
    """GEN13: the declared outcome_label must match what label_rule derives.

    Diagnosis questions report `outcome_rule_not_applicable`, and so do cases that never
    lost weight under a rule without a floor (`min_change_kg`); with a floor the rule reads
    such a case as a gain above its low point, the same as `derive_outcome`.
    """
    if (getattr(raw, "adjudication", None) or {}).get("ddx"):
        return [{"kind": "outcome_rule_not_applicable", "severity": GEN13_SEVERITY,
                 "detail": "diagnosis question: outcome_label means the diagnosis holds, not decided by the weight label_rule; "
                           "this question type's outcome-derivation rule is not yet implemented"}]
    _classes = {str(nz.get("class")) for nz in (getattr(cs, "noise", None) or [])}
    got, detail = derive_outcome(raw.longitudinal_data, raw.label_rule,
                                 noise_windows=_noise_windows(cs),
                                 persistent_offset=bool(_classes & PERSISTENT_OFFSET_NOISE))
    # No weight lost and no floor => the regain rule cannot apply; report not applicable,
    # not a mismatch. With a floor, `derive_outcome` reads it and so does this check.
    if (got is not None and float(detail.get("lost", 0.0)) <= 0.0
            and float(detail.get("floor_kg") or 0.0) <= 0.0):
        return [{"kind": "outcome_rule_not_applicable", "severity": GEN13_SEVERITY,
                 "detail": f"this case never lost weight (start {detail.get('start')} <= lowest "
                           f"{detail.get('nadir')}), while label_rule speaks of 'a rebound >= "
                           f"{detail.get('frac')} x the amount already lost' with no floor (min_change_kg), read on "
                           f"{detail.get('smoothing')} -- the rule can never be satisfied, this question's outcome does not live on the weight signal"}]
    if got is None:
        return [{"kind": "outcome_not_derivable",
                 "severity": _gen13("outcome_not_derivable"),
                 "detail": f"could not derive an outcome from the trajectory: {detail.get('reason')}"}]
    hits = []
    if detail.get("notes"):
        hits.append({"kind": "label_rule_unstructured", "severity": GEN13_SEVERITY,
                     "detail": "; ".join(detail["notes"])})
    if got != raw.outcome_label:
        hits.append({"kind": "outcome_declared_not_derived",
                     "severity": _gen13("outcome_declared_not_derived"),
                     "detail": f"declared {raw.outcome_label} · label_rule derives {got} · {detail}"})
    return hits


# ============================================================ GEN14: the gold must be derivable from the question face
# The observations needed to derive gold must be visible to the solver and must move by T;
# a question that cannot be answered is not emitted.
GEN14_SEVERITY = "gate"


#: Relative range (range / mean) below which a <=T evidence stream counts as flat.
#: Relative, so one value serves every unit.
#: Few streams sit near 0.15, so the verdict is insensitive to its exact position.
GEN14_FLAT_REL_SPAN = 0.15


# gold driver -> observables, at least one of which must be visible. An empty set means the
# primary signal suffices; "needs nothing at all" must use NO_OBSERVABLE_REQUIRED.
class _NoObservableRequired(frozenset):
    """Explicit "no leading indicator required" marker. It compares equal to `set()`; test it
    with `is_declared_no_observable`.
    """
    def __repr__(self): return "NO_OBSERVABLE_REQUIRED"


NO_OBSERVABLE_REQUIRED = _NoObservableRequired()


def is_declared_no_observable(need) -> bool:
    """Whether `need` is the explicit NO_OBSERVABLE_REQUIRED marker (identity, not equality)."""
    return need is NO_OBSERVABLE_REQUIRED


DRIVER_REQUIRED_OBSERVABLE: dict[str, set[str]] = {
    "poor_medication_adherence": {"medication_adherence"},
    "insufficient_dose_exposure": {"dose_timeline", "medication_adherence"},
    "medication_intolerance": {"gi_symptom_score"},
    "calorie_intake_change": {"diet_carb_pct", "CGM_TIR"},
    "activity_decline": {"steps", "activity_index"},
    "sleep_decline": {"sleep_hours"},
    "measurement_noise": {"scale_qc_flag", "weight_ref"},
    "concurrent_medication_effect": set(),  # needs a drug/comorbidity clue -> _NEEDS_SYMPTOM
    "acute_illness": set(),
    "fluid_or_GI_weight_variation": set(),
    "cost_or_access_issue": set(),
    "biological_low_response": {"dose_timeline"},
    "inadequate_treatment_duration": {"dose_timeline"},
    "unknown_or_multifactorial": NO_OBSERVABLE_REQUIRED,
}


# Drivers derivable only from a symptom clue: the question must carry >= 1 true-symptom EV.
_NEEDS_SYMPTOM = {"concurrent_medication_effect", "acute_illness",
                  "fluid_or_GI_weight_variation", "cost_or_access_issue"}


def _flat_pre_T(sp, names) -> list[str]:
    """Signals that are essentially flat (or have < 3 points) within the <=T window."""
    T = int((sp.prediction_context or {}).get("prediction_time_T", 10 ** 9))
    out = []
    for n in sorted(names or ()):
        vis = [float(p["value"]) for p in ((sp.longitudinal_data or {}).get(n) or [])
               if int(p["ts"]) <= T]
        if len(vis) < 3:
            out.append(n)
            continue
        span, scale = max(vis) - min(vis), max(1e-9, abs(sum(vis) / len(vis)))
        if span / scale < GEN14_FLAT_REL_SPAN:
            out.append(n)
    return out


def _real_symptom_evidence(sp) -> tuple[list, str]:
    """The question's true-symptom EVs and the basis used to find them.

    Counted from the Q-side ledger's `real_symptom_evidence_ids`, as `judges.py` does:
    benign distractors share `source_type`, so counting by type would let them fill the quota.
    """
    led = list(sp.evidence_ledger or [])
    by_type = [e for e in led
               if str(e.get("source_type", "")) == "patient_reported_symptom"]
    from . import wq
    man = wq.injected_manifest(getattr(sp, "case_id", "") or "", required=False)
    if not man:
        return by_type, "source_type_fallback"
    ids = {str(x) for x in (man.get("real_symptom_evidence_ids") or [])}
    return [e for e in led if str(e.get("evidence_id")) in ids], "manifest"


def _gold_lab_signal_leaks(sp, ddx: dict | None) -> list[str]:
    """Declared findings of this case's gold that already cross their registered direction at
    <=T -- the input to `insufficient_tier_leaks_labs`.

    Which items count comes from the registry, and "abnormal" is the printed reference range
    with no margin. A magnitude ruler such as `evaluate.ABNORMAL_MARGIN` answers a different
    question: a TSH 13x below its lower bound is only 6% of the interval width.
    """
    # Every declared item counts, not only `confirmatory` ones: rendered `screening` items can
    # already be pathognomonic (e.g. low Na with high K in Addison's disease).
    out: list[str] = []
    for r in _declared_lab_readings(sp, ddx):
        if not r["crossed"]:
            continue
        if r["qualitative"]:
            out.append(f"{r['name']} positive")
        else:
            out.append(f"{r['name']}={r['v']}(reference {r['lo']}–{r['hi']} · "
                       f"registered direction {r['want'] or 'undeclared'})")
    return out


def _declared_lab_readings(sp, ddx: dict | None) -> list[dict]:
    """Every <=T `lab_result` line for an item the gold declares, with whether it crosses the
    declared direction. Shared by both lab arms of GEN14.

    Records carry `fid`, `name`, `want`, `trajectory`, `day`, `crossed`, `qualitative` and,
    for numeric lines, `v/lo/hi`.
    """
    sid = str((ddx or {}).get("spec_id") or "")
    if not sid:
        return []
    # Imported by name: an alias would be a new symbol crossing the segment boundary.
    from .registry import condition_findings_for_case, load_findings
    fx = load_findings()
    prof = (condition_findings_for_case(fx) or {}).get(sid) or {}
    decl: dict[str, str] = {str(f.get("id")): str(f.get("direction") or "")
                            for f in (prof.get("findings") or []) if f.get("id")}
    traj: dict[str, str] = {str(f.get("id")): str(f.get("trajectory") or "")
                            for f in (prof.get("findings") or []) if f.get("id")}
    if not decl:
        return []
    # Matchable names: Chinese name, English name and aliases (as in `judge_disc_tool`).
    names: dict[str, str] = {}
    for fid in decl:
        f = fx.get(fid) or {}
        for nm in ([f.get("name_cn"), f.get("name_en")] + list(f.get("aliases") or [])):
            if nm and str(nm).strip():
                names[str(nm).strip()] = fid
    T = int((sp.prediction_context or {}).get("prediction_time_T", 10 ** 9))
    out: list[dict] = []
    for e in (sp.evidence_ledger or []):
        if str(e.get("source_type", "")) != "lab_result":
            continue
        if int(e.get("source_timestamp", 10 ** 9)) > T:
            continue
        txt = str(e.get("symptom", ""))
        hit = next((nm for nm in names if txt.startswith(nm)), None)
        if hit is None:
            continue
        fid = names[hit]
        want = decl.get(fid, "")
        rec = {"fid": fid, "name": hit, "want": want, "trajectory": traj.get(fid, ""),
               "day": int(e.get("source_timestamp", 0)), "qualitative": False,
               "v": None, "lo": None, "hi": None}
        m = _LAB_VAL_RE.search(txt)
        if not m:
            # Qualitative item (e.g. ANA "positive"): the word "positive" is the signal.
            rec.update(qualitative=True,
                       crossed=(want == "positive"
                                and ("阳性" in txt or "positive" in txt.lower())))
            out.append(rec)
            continue
        try:
            v, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
        except ValueError:
            continue
        # Judged in the registered direction; without one, out of range on either side.
        if want == "low":
            bad = v < lo
        elif want == "high":
            bad = v > hi
        else:
            bad = not (lo <= v <= hi)
        rec.update(v=v, lo=lo, hi=hi, crossed=bad)
        out.append(rec)
    return out


#: Trajectories whose readings may all sit inside the reference band (`episodic`), so the
#: sufficient-tier arm does not require them to cross.
GEN14_LAB_EXEMPT_TRAJECTORIES = frozenset({"episodic"})


def _gold_lab_signal_missing(sp, ddx: dict | None) -> list[str]:
    """Sufficient tier: declared findings of the gold that appear on the face but never show
    their declared direction at <=T.

    Only items declared `high` / `low` / `positive`, not exempt by trajectory, and matched by
    at least one <=T lab line are checked; absence from the face is the symptom arm's concern.
    """
    by_fid: dict[str, list[dict]] = {}
    for r in _declared_lab_readings(sp, ddx):
        if r["want"] not in ("high", "low", "positive"):
            continue
        if r["trajectory"] in GEN14_LAB_EXEMPT_TRAJECTORIES:
            continue
        by_fid.setdefault(r["fid"], []).append(r)
    out: list[str] = []
    for fid, rs in by_fid.items():
        if any(r["crossed"] for r in rs):
            continue
        if rs[0]["qualitative"]:
            shown = "not positive"
        else:
            shown = "/".join(str(r["v"]) for r in rs) + f" (ref {rs[0]['lo']}-{rs[0]['hi']})"
        out.append(f"{fid} {shown}, declared {rs[0]['want']} "
                   f"({rs[0]['trajectory'] or 'no trajectory'})")
    return out


def check_gold_coverage(sp, gold_drivers, ddx: dict | None = None) -> list[dict]:
    """GEN14 on the solver payload `sp`."""
    hits: list[dict] = []
    signals = set(sp.longitudinal_data or {})
    real_sym, basis = _real_symptom_evidence(sp)
    if basis != "manifest":
        # Without the ledger, true symptoms cannot be counted; report it rather than pass.
        hits.append({"kind": "real_symptom_manifest_missing", "severity": GEN14_SEVERITY,
                     "detail": f"{getattr(sp, 'case_id', '?')}: the Q-side injection ledger is missing, "
                               f"GEN14 can only count by source_type (benign events can fill the quota)"})

    if ddx:  # diagnosis question
        # The insufficient-information tier is the reverse: its true symptoms all lie after T, so
        # >= 2 visible true symptoms fail it, as fewer than 2 fail an ordinary diagnosis question.
        if bool(ddx.get("insufficient")):
            if len(real_sym) >= 2:
                hits.append({"kind": "insufficient_tier_leaks_clues",
                             "severity": GEN14_SEVERITY,
                             "detail": f"insufficient-information tier, but the solver can see {len(real_sym)} true symptoms"
                                       f" (basis={basis}) -- this tier requires <2, "
                                       f"otherwise it becomes an ordinary diagnosis question"})
            # Discriminating findings also arrive as lab results, so the insufficient tier checks that
            # no declared finding of the gold already crosses its direction.
            _leak = _gold_lab_signal_leaks(sp, ddx)
            if _leak:
                hits.append({"kind": "insufficient_tier_leaks_labs",
                             "severity": GEN14_SEVERITY,
                             "detail": f"insufficient-information tier, but in the <=T ledger this case's gold's "
                                       f"declared items have already crossed the registered direction "
                                       f"on {len(_leak)}: {_leak[:4]} -- "
                                       f"this tier is defined as \"not a single discriminating signal on the question face\", "
                                       f"and labs are the main channel for discriminating evidence (the symptom-channel check does not cover labs). "
                                       f"Such a case would reward ignoring labs already visible: "
                                       f"answering \"consider this diagnosis\" is clinically correct yet scores 0 on `ct_ok`"})
            return hits
        if len(real_sym) < 2:
            hits.append({"kind": "gold_not_coverable", "severity": GEN14_SEVERITY,
                         "detail": f"diagnosis question but the solver can see only {len(real_sym)} true symptoms"
                                   f" (basis={basis}), cross-time synthesis has nothing to work with"})
        # Sufficient tier: declared findings on the face must show their direction.
        _miss = _gold_lab_signal_missing(sp, ddx)
        if _miss:
            hits.append({"kind": "gold_lab_signal_missing", "severity": GEN14_SEVERITY,
                         "detail": f"sufficient tier, but {len(_miss)} declared finding(s) of "
                                   f"the gold are on the face and never show their declared "
                                   f"direction at <=T: {_miss[:4]} -- the lab channel carries "
                                   f"the discriminating evidence, so the gold cannot be read "
                                   f"off this face"})
        return hits

    for g in (gold_drivers or []):
        need = DRIVER_REQUIRED_OBSERVABLE.get(g)
        if need is None:
            hits.append({"kind": "driver_not_in_coverage_map", "severity": GEN14_SEVERITY,
                         "detail": f"{g}: not registered in DRIVER_REQUIRED_OBSERVABLE, cannot check derivability"})
            continue
        if need and not (need & signals):
            hits.append({"kind": "gold_not_coverable", "severity": GEN14_SEVERITY,
                         "detail": f"gold {g} needs one of {sorted(need)}, "
                                   f"but the solver can see only signals {sorted(signals)}"})
        elif need:
            # A registered evidence stream must also move by T; fails only if every candidate is flat.
            avail = need & signals
            flat = _flat_pre_T(sp, avail)
            if flat and len(flat) == len(avail):
                hits.append({"kind": "gold_evidence_flat_pre_T", "severity": GEN14_SEVERITY,
                             "detail": f"gold {g}'s evidence signals {sorted(flat)} show no change within the <=T window -- "
                                       f"the signal name is on the question face, but it carries no information that can derive the gold"})
        if g in _NEEDS_SYMPTOM and not real_sym:
            hits.append({"kind": "gold_not_coverable", "severity": GEN14_SEVERITY,
                         "detail": f"gold {g} can only be derived from symptom clues, but the question face has 0 true-symptom EVs"})
    return hits


_LAB_VAL_RE = re.compile(r"([0-9.]+)\s*\S*\(参考\s*([0-9.]+)[–\-]([0-9.]+)\)")
