"""gates.py -- question-generation gates that run on haenv's side of the kernel.

`verify.py` checks injected items one by one; this module checks the primary signal and
case- and batch-level consistency. Each hit carries a severity: "gate" blocks emission of
the case, "warn" is recorded in the audit and log only.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import re

import collections
import logging
import math
import pathlib
from haenv import data_root as _dr  # resource root: repo root in a source tree, haenv/_data in a wheel
from haenv.streams import device_signals


def _clinical_cv_of(sig: str) -> float:
    """Per-measurement CV of a clinical stream (via `build._clinical_cv()`); 0.0 when unregistered."""
    try:
        from .build import _clinical_cv
        return float(_clinical_cv().get(sig, 0.0) or 0.0)
    except Exception:                                      # noqa: BLE001
        return 0.0


def clinical_coupling_judgeable() -> dict:
    """`{signal: bool}` from `registry/physio_streams.yaml`: False declares that GEN15 cannot judge
    direction on that signal (effect below measurement noise). Unregistered signals are judged.
    """
    try:
        from .build import _cached_yaml
        doc = _cached_yaml(_dr() / "registry" / "physio_streams.yaml") or {}
        return dict(((doc.get("clinical_measurement") or {})
                     .get("coupling_judgeable") or {}))
    except Exception:                                      # noqa: BLE001
        return {}

log = logging.getLogger("haenv.gates")

#: Resolved relative to the package, so it also works from an installed wheel.
_REG_DIR =_dr() / "registry"

# GEN6 (gate): actual sampling cadence and gaps must match the premise's declaration.
GEN6_SEVERITY = "gate"
GEN6_MISSING_TOL = 0.08  # missing-rate tolerance around the declared whole-course rate


def _noise_windows(cs) -> list[tuple[int, int]]:
    """Noise windows from CaseSpec.noise, placed as `build.py` places them."""
    wins: list[tuple[int, int]] = []
    for nz in (getattr(cs, "noise", None) or []):
        d0 = int(nz["week"]) * 7
        wins.append((d0, d0 + int(nz.get("span_days", 10))))
    return wins


def _spans_window(a: int, b: int, wins: list[tuple[int, int]]) -> bool:
    """Whether [a, b] intersects a noise window. Steps spanning a window are skipped (not the
    points inside it, which MNAR hole-punching already removed).
    """
    return any(a <= hi and b >= lo for lo, hi in wins)


def check_cadence(longitudinal_data: dict, premise, cs) -> list[dict]:
    """GEN6: actual sampling must match the premise's declaration.

    Covers non-auxiliary signals. With declared missingness: the modal step must equal
    `sampling_days`, the missing rate must be within `GEN6_MISSING_TOL` of
    `expected_missing_rate`, and no gap may exceed `max_gap_days`. Without it: every step must
    equal `sampling_days`. Gaps inside noise windows and a trailing partial step are exempt.
    """
    from synth import AUX_SIGNALS          # kernel

    hits: list[dict] = []
    spec = (getattr(premise, "device_signals", None) or {}).get("signals", {}) or {}
    wins = _noise_windows(cs)

    for sig, pts in (longitudinal_data or {}).items():
        if sig in AUX_SIGNALS:
            continue
        sspec = spec.get(sig) or {}
        declared = int(sspec.get("sampling_days", 0) or 0)
        if declared <= 0:
            hits.append({"kind": "sampling_days_undeclared", "severity": GEN6_SEVERITY,
                         "detail": f"{sig}: premise device_signals.signals does not declare sampling_days"})
            continue
        steps = []
        for a, b in zip(pts, pts[1:]):
            ta, tb = int(a["ts"]), int(b["ts"])
            if _spans_window(ta, tb, wins):          # a step spanning a noise window does not count
                continue
            steps.append(tb - ta)
        if steps and steps[-1] < declared:
            steps = steps[:-1]                       # exempt the trailing partial point

        if "expected_missing_rate" not in sspec:  # missingness not declared -> every step must equal the declared one
            bad = sorted({s for s in steps if s != declared})
            if bad:
                hits.append({"kind": "sampling_cadence_mismatch", "severity": GEN6_SEVERITY,
                             "detail": f"{sig}: declared {declared}d · actual step {bad}"})
            continue

        # ---- missingness declared: ask "cadence" and "how much was missed" separately ----
        mode = max(set(steps), key=steps.count) if steps else declared
        if mode != declared:
            hits.append({"kind": "sampling_cadence_mismatch", "severity": GEN6_SEVERITY,
                         "detail": f"{sig}: declared {declared}d · modal step {mode}d"
                                   f" (step distribution {sorted(set(steps))[:6]})"})
        span = (int(pts[-1]["ts"]) - int(pts[0]["ts"])) if len(pts) >= 2 else 0
        expected_n = span // max(1, declared) + 1
        miss = 1.0 - (len(pts) / expected_n) if expected_n else 0.0
        want = float(sspec["expected_missing_rate"])
        if abs(miss - want) > GEN6_MISSING_TOL:
            hits.append({"kind": "missing_rate_mismatch", "severity": GEN6_SEVERITY,
                         "detail": f"{sig}: declared missing rate {want:.3f} · actual {miss:.3f}"
                                   f" (tolerance {GEN6_MISSING_TOL})"})
        max_gap = int(sspec.get("max_gap_days", 0) or 0)
        if max_gap:
            over = [s for s in steps if s - declared > max_gap]
            if over:
                hits.append({"kind": "missing_gap_too_long", "severity": GEN6_SEVERITY,
                             "detail": f"{sig}: declared max gap {max_gap}d · actual "
                                       f"{sorted({s - declared for s in over})[:5]}"
                                       f" (gaps inside a noise window are already exempted, so this is day-to-day missingness itself running over)"})
    return hits


# GEN7 stays warn: the kernel's hypertension domain has no lab signals, yet a hypertensive
# patient plausibly has a lab panel listed.
GEN7_SEVERITY = "warn"

#: Devices whose idle declaration blocks emission: a listed wearable with no wearable stream
#: is a factual error in the prompt.
GEN7_GATE_DEVICES: dict[str, str] = {"wearable": "gate"}

# device -> signals it can produce (the converse of the kernel's `signal_not_in_inventory`).
DEVICE_SIGNALS: dict[str, set[str]] = device_signals()   # `proves_device` in registry/streams.yaml


def check_device_inventory(user_profile: dict, longitudinal_data: dict) -> list[dict]:
    """GEN7: every device in user_profile.device_inventory must produce at least one signal."""
    hits: list[dict] = []
    present = set(longitudinal_data or {})
    for dev in (user_profile or {}).get("device_inventory", []) or []:
        expect = DEVICE_SIGNALS.get(dev)
        if expect is None:
            hits.append({"kind": "device_not_in_map", "severity": GEN7_SEVERITY,
                         "detail": f"{dev}: not registered in DEVICE_SIGNALS, cannot check whether it is idle"})
            continue
        if not (expect & present):
            hits.append({"kind": "device_inventory_idle",
                         "severity": GEN7_GATE_DEVICES.get(dev, GEN7_SEVERITY),
                         "detail": f"{dev}: declared in inventory but produced zero signals (expected one of {sorted(expect)})"})
    return hits


# ============================================================ meta-rule: every declared field must register a verifier
# Every latent_premise field is registered with the check that fails the case when the field
# does not match reality. An unregistered field is a gate hit; a registered field with no
# verifier (`None`) is a warn.
GEN_REGISTRY_SEVERITY = "warn"
GEN_REGISTRY_GATE_KINDS = {"premise_field_unregistered"}


def _gen_reg(kind: str) -> str:
    return "gate" if kind in GEN_REGISTRY_GATE_KINDS else GEN_REGISTRY_SEVERITY

EXEMPT = "EXEMPT"  # needs no "declared vs actual" check

# path -> (verifier | EXEMPT | None (unverified), description)
PREMISE_REGISTRY: dict[str, tuple[str | None, str]] = {
    "patient_basics.disease": ("synth.premise_conflicts", "signal_not_in_disease_domain / disease_conflicts_original"),
    "patient_basics.comorbidities": ("verify.check_event", "plausible_for_profile uses Facts' comorbidities to judge event plausibility"),
    "patient_basics.age_range": ("verify.check_event", "plausible_for_profile(elderly)"),
    "patient_basics.sex": ("gates.check_sex_consistency",
                           "GEN19 -- the question face's text must not contradict the declared sex's anatomy"),
    # regimen is a {drug, dose_steps} dict, so each half is registered under its own path.
    "patient_basics.regimen.drug": ("synth.premise_conflicts",
                                    "determines the DRUG_PKPD ladder and minimum titration interval; dose_timeline is checked point by point against it"),
    "patient_basics.regimen.dose_steps": ("synth.premise_conflicts",
                                          "dose_off_ladder / titration_too_fast"),
    "patient_basics.goals": (EXEMPT, "goes into the question-face text only, plays no part in any judgment"),
    "event_density.measure_per_week": ("verify.check_stream", "cadence_matches_density (injected auxiliary streams only)"),
    "event_density.dosing_per_week": (
        "gates.check_dosing_consistency",
        "GEN20b -- a weekly formulation must not declare daily dosing. dose_timeline records "
        "dose-ladder changes, not administrations, so this checks the declaration against the "
        "route of administration only"),
    "event_density.symptom_rate": ("gates.check_event_density",
        "GEN20a -- declared rate × weeks must equal the injected benign-symptom count exactly (no tolerance)"),
    "event_density.life_event_rate": ("gates.check_event_density", "GEN20a -- same as symptom_rate"),
    "device_signals.devices": ("gates.check_device_inventory", "GEN7 idle-device check + the converse of the kernel's signal_not_in_inventory"),
    "device_signals.signals.*.sampling_days": ("gates.check_cadence", "GEN6 -- sampling cadence matches the declaration"),
    "device_signals.signals.*.plausible_range": ("synth.premise_conflicts", "value_out_of_physio_range checked point by point"),
    "device_signals.signals.*.unit": (EXEMPT, "display only"),
    "device_signals.signals.*.expected_missing_rate": (
        "gates.check_cadence", "missing_rate_mismatch -- whatever was declared missing must be what actually happened"),
    "device_signals.signals.*.max_gap_days": (
        "gates.check_cadence", "missing_gap_too_long -- the longest day-to-day missingness gap (noise windows counted separately)"),
    "device_signals.n_signals": (EXEMPT, "a redundant count, derivable from the length of signals"),
    "adherence.baseline": ("synth.premise_conflicts", "adherence_below_premise"),
    "adherence.trajectory": ("synth.premise_conflicts", "adherence_below_premise takes trajectory's lowest point"),
    "adherence.missingness_mechanism": (
        "gates.check_missingness",
        "GEN20c -- the missing rate before and after the reversal point must not differ significantly "
        "(a gap must not become a proxy for the outcome)"),
    "source": (EXEMPT, "a provenance marker, plays no part in any judgment"),
    "meta.*": (EXEMPT, "question-generation bookkeeping (verifier-only), plays no part in any judgment"),
}


def _premise_paths(obj, prefix: str = "") -> list[str]:
    """Flatten premise into dot paths; a dict-of-dict's middle layer becomes `*`."""
    if not isinstance(obj, dict):
        return [prefix]
    out: list[str] = []
    for k, v in obj.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        out.extend(_premise_paths(v, p))
    return out


def _match(path: str) -> tuple[str | None, str] | None:
    """Registry entry for `path`: exact match, then one `*` segment, then a prefix wildcard."""
    if path in PREMISE_REGISTRY:
        return PREMISE_REGISTRY[path]
    segs = path.split(".")
    for i in range(len(segs)):
        cand = ".".join(segs[:i] + ["*"] + segs[i + 1:])
        if cand in PREMISE_REGISTRY:
            return PREMISE_REGISTRY[cand]
    for i in range(1, len(segs)):                      # prefix wildcard: meta.* covers meta.<any depth>
        if ".".join(segs[:i] + ["*"]) in PREMISE_REGISTRY:
            return PREMISE_REGISTRY[".".join(segs[:i] + ["*"])]
    return None


def check_premise_registry(premise) -> list[dict]:
    """Meta-rule: every latent_premise field must be registered in PREMISE_REGISTRY."""
    d = premise if isinstance(premise, dict) else {
        k: getattr(premise, k) for k in ("patient_basics", "event_density",
                                         "device_signals", "adherence")
        if hasattr(premise, k)}
    hits: list[dict] = []
    for path in _premise_paths(d):
        ent = _match(path)
        if ent is None:
            hits.append({"kind": "premise_field_unregistered", "severity": _gen_reg("premise_field_unregistered"),
                         "detail": f"{path}: not registered in PREMISE_REGISTRY -- name a verifier or state the exemption reason"})
        elif ent[0] is None:
            hits.append({"kind": "premise_field_unverified", "severity": GEN_REGISTRY_SEVERITY,
                         "detail": f"{path}: registered but has no verifier ({ent[1]})"})
    return hits


# ============================================================ GEN15: clinical signal coupling direction must match the declaration
# A formula can be right in magnitude and range while a sign error reverses the curve.
GEN15_SEVERITY = "gate"  # a wrong direction is a hard error

#: Net weight change (kg) below which coupling direction is not judged.
#: Direction is defined relative to weight; with weight barely moved, the sign is noise.
#: Lowering it turns noise into gate-level false hits; raising it narrows coverage.
GEN15_MIN_WEIGHT_DELTA_KG = 0.5

#: Effects smaller than GEN15_SNR_K x the endpoint noise SD (sqrt(2) x cv x base) are not
#: judged; at 2 sigma a pure-noise sign flip has probability < 5%.
GEN15_SNR_K = 2.0


def check_clinical_coupling(longitudinal_data: dict, clinical: dict,
                            main: str = "weight", drug_terms: dict | None = None) -> list[dict]:
    """GEN15: each clinical signal's net change must have the sign of its declared drivers.

    `want = per_kg x CLINICAL_ATTEN x delta_w + drug_terms[sig]`. Not judged when the weight
    change is at most `GEN15_MIN_WEIGHT_DELTA_KG`, when the registry declares the signal
    unjudgeable, or when `|want|` is within measurement noise (`GEN15_SNR_K`).
    """
    hits: list[dict] = []
    _JUDGEABLE = clinical_coupling_judgeable()
    w = [float(p["value"]) for p in (longitudinal_data or {}).get(main) or []]
    if len(w) < 2:
        return hits
    dw = w[-1] - w[0]
    if abs(dw) <= GEN15_MIN_WEIGHT_DELTA_KG:
        return hits
    for sig, spec in (clinical or {}).items():
        pts = (longitudinal_data or {}).get(sig) or []
        if len(pts) < 2:
            continue
        dv = float(pts[-1]["value"]) - float(pts[0]["value"])
        if abs(dv) < 1e-9:
            continue
        drug = float((drug_terms or {}).get(sig, 0.0))
        # Matches the render formula `base + per_kg x CLINICAL_ATTEN x delta_w + drug`; `drug` is
        # already net of the weight-mediated part.
        from .build import CLINICAL_ATTEN
        want = spec["per_kg"] * CLINICAL_ATTEN * dw + drug
        if abs(want) < 1e-9:
            continue  # the terms cancel => no direction
        if _JUDGEABLE.get(sig) is False:
            continue
        # Missing `base` => no exemption.
        _cv, _base = _clinical_cv_of(sig), spec.get("base")
        if _cv and _base is not None:
            _noise_sd = abs(float(_base)) * _cv * math.sqrt(2.0)
            if abs(want) < GEN15_SNR_K * _noise_sd:
                continue
        if want * dv < 0:
            _d = (f" + drug effect {drug:+.3f}" if drug else "")
            hits.append({"kind": "clinical_coupling_direction", "severity": GEN15_SEVERITY,
                         "detail": f"{sig}:Δ{main}={dw:+.2f}kg · per_kg={spec['per_kg']:+}"
                                   f"{_d} => expected {'to rise' if want > 0 else 'to fall'}, actual Δ={dv:+.3f}"})
    return hits


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


# GEN20 (gate): injection density, dosing frequency and missingness shape must match the premise.
GEN20_SEVERITY = "gate"

#: Minimum grid points on each side of the reversal point to compare missing rates.
#: At 8 points the missing rate's resolution is 12.5%; coarser cannot show a difference.
#: Empirical: it should sit below the grid sizes of the cases that can be judged.
MISSINGNESS_MIN_GRID = 8

# Administrations per week by drug (weekly injectable = 1, oral daily = 7).
DRUG_DOSES_PER_WEEK: dict[str, int] = {
    "metformin": 7, "semaglutide": 1, "tirzepatide": 1, "liraglutide": 7, "dulaglutide": 1,
    # Indications and dose ladders are in `registry/drug_indications.yaml`.
    "amlodipine": 7, "atorvastatin": 7,
}


def check_event_density(raw, premise, T: int) -> list[dict]:
    """GEN20a: the injected counts of benign symptoms and life events must equal
    `round(rate x weeks)` exactly, using `events.EVENT_RATE_DEFAULTS` when a rate is not declared.

    True symptoms and near-miss distractors are not counted.
    """
    hits: list[dict] = []
    ed = (getattr(premise, "event_density", None) or {})
    # Ledger required: without it, true symptoms would be counted as injected noise.
    from . import wq
    man = wq.injected_manifest(getattr(raw, "case_id", ""))
    real = set(man.get("real_symptom_evidence_ids") or [])
    # Near-miss distractors belong to the case narrative, not the benign density.
    real |= set(man.get("lookalike_evidence_ids") or [])
    led = raw.evidence_ledger or []
    from .events import EVENT_RATE_DEFAULTS, event_weeks, expected_event_counts
    weeks = event_weeks(T)
    got = {
        "symptom_rate": sum(1 for e in led
                            if str(e.get("source_type")) == "patient_reported_symptom"
                            and str(e.get("evidence_id")) not in real),
        "life_event_rate": sum(1 for e in led
                               if str(e.get("source_type")) == "patient_reported_context"),
    }
    # Same defaults and formula as the injector (`expected_event_counts`).
    want_all = expected_event_counts(ed, T)
    for key, n in got.items():
        want = want_all[key]
        if n != want:
            _decl = ed.get(key, EVENT_RATE_DEFAULTS[key])
            _src = "declared" if key in ed else f"default({EVENT_RATE_DEFAULTS[key]}, this key is not declared)"
            hits.append({"kind": "event_density_mismatch", "severity": GEN20_SEVERITY,
                         "detail": f"{key}: {_src} {_decl}/wk × {weeks:.1f}wk = {want} items, "
                                   f"actually injected {n}"})
    return hits


#: GEN27 (gate): a drug with no indication for the condition is a factual error.
GEN27_SEVERITY = "gate"

#: Severity constants for GEN28 (unregistered comorbidity name: warn, a content backlog) and
#: GEN29 (infeasible information gap: gate, a self-contradictory question).
GEN28_SEVERITY = "warn"
GEN29_SEVERITY = "gate"


def _judging_registry(name: str) -> dict:
    """Read a registry file directly, without `regpath.load_registry`'s plugin overrides, so the
    judging fingerprint does not depend on installed plugins. Plugin-added entries are
    therefore unregistered here (default-deny).
    """
    from .build import _cached_yaml
    return _cached_yaml(_REG_DIR / name) or {}


def check_rhythm_gap_feasible(cs) -> list[dict]:
    """GEN29: a declared `rhythm_gap` must fit the visible window (`rhythm_gap_feasible(T)`),
    rather than being dropped silently at run time.
    """
    from .evaluate import REAL_RHYTHM_GAP_DAYS, REAL_RHYTHM_SLICES, rhythm_gap_feasible

    latent = getattr(cs, "latent", None) or {}
    if not latent.get("rhythm_gap"):
        return []
    T = int(getattr(cs, "index_time_T", 0) or 0)
    if rhythm_gap_feasible(T):
        return []
    return [{"kind": "rhythm_gap_declared_but_infeasible", "severity": "gate",
             "detail": f"rhythm_gap is declared but T={T} cannot accommodate it (needs >= "
                       f"{int(REAL_RHYTHM_GAP_DAYS) + int(REAL_RHYTHM_SLICES)})"
                       f" -- it would be silently downgraded at run time, and the declaration and the artefact would diverge"}]


def check_comorbidity_vocab(premise) -> list[dict]:
    """GEN28: a comorbidity name must match a physiological consumer (prefix match, as
    `events.BASELINE_ADJUST` and `events.COMORBID_PROXY_TAGS` do) or be registered as
    `label_only` in `comorbidity_vocab.yaml`; otherwise it is reported (warn).
    """
    hits: list[dict] = []
    pb = (getattr(premise, "patient_basics", None) or {})
    names = [str(x).strip() for x in (pb.get("comorbidities") or []) if str(x).strip()]
    if not names:
        return hits
    doc = _judging_registry("comorbidity_vocab.yaml")
    known = tuple(doc.get("recognized") or {})
    declared = {str((r or {}).get("name") or "") for r in (doc.get("label_only") or [])}
    for n in names:
        if any(n.startswith(k) for k in known):
            continue
        kind = ("comorbidity_label_only" if n in declared else "comorbidity_unregistered")
        hits.append({"kind": kind, "severity": "warn",
                     "detail": (f"{n!r}: has no physiological consumer"
                                + ("(registered as label_only -- the question face carries the label, the world layer has no reaction)"
                                   if n in declared else
                                   " and is not registered -- register it under comorbidity_vocab.label_only, "
                                   "or add it to the two tables in events so it takes effect"))})
    return hits


def check_drug_indication(premise) -> list[dict]:
    """GEN27: the drug must have an indication for the condition, and the dose ladder must match.

    Per `(disease, drug)` in `drug_indications.yaml`: `mismatch` or unregistered fails
    (default-deny); for `approved` / `off_label`, a declared top dose above the registered top
    fails, one below it is a warn. Applies to every case, whatever produced it.
    """
    hits: list[dict] = []
    pb = (getattr(premise, "patient_basics", None) or {})
    reg = pb.get("regimen") or {}
    disease = str(pb.get("disease") or "").strip()
    drug = str(reg.get("drug") or "").strip()
    steps = [float(x) for x in (reg.get("dose_steps") or []) if x is not None]
    if not disease or not drug:
        return hits
    _pairs = (_judging_registry("drug_indications.yaml").get("pairs") or {})
    _spec = (_pairs.get(disease) or {}).get(drug)
    status = str((_spec or {}).get("status") or "unregistered")
    if status == "mismatch":
        why = (_spec or {}).get("why") or ""
        hits.append({"kind": "drug_not_indicated", "severity": GEN27_SEVERITY,
                     "detail": f"{disease} × {drug}: registered as mismatch -- {why}"})
        return hits
    if status == "unregistered":
        hits.append({"kind": "drug_indication_unregistered", "severity": GEN27_SEVERITY,
                     "detail": f"{disease} × {drug}: `drug_indications.yaml` has no such pair "
                               f"=> default-deny (\"not registered\" is never folded into \"allowed\")"})
        return hits
    want = [float(x) for x in ((_spec or {}).get("dose_steps") or [])]
    if steps and want:
        if max(steps) > max(want) + 1e-9:
            hits.append({"kind": "drug_dose_above_indication", "severity": GEN27_SEVERITY,
                         "detail": f"{disease} × {drug}: declared endpoint {max(steps)} "
                                   f"> the indication ladder's top {max(want)} => overdose"})
        elif max(steps) < max(want) - 1e-9:
            hits.append({"kind": "drug_dose_below_indication", "severity": "warn",
                         "detail": f"{disease} × {drug}: declared endpoint {max(steps)} "
                                   f"< the indication ladder's top {max(want)} => underdosed"
                                   f" (clinically legitimate, so only warns, does not block)"})
    return hits


def check_dosing_consistency(premise) -> list[dict]:
    """GEN20b: `dosing_per_week` must fit the drug's formulation (weekly vs. daily).

    `dose_timeline` records dose changes, not administrations, so this checks the
    declaration's consistency only.
    """
    hits: list[dict] = []
    ed = (getattr(premise, "event_density", None) or {})
    reg = (getattr(premise, "patient_basics", None) or {}).get("regimen") or {}
    drug = str(reg.get("drug") or "").strip().lower()
    want = DRUG_DOSES_PER_WEEK.get(drug)
    if want is None or "dosing_per_week" not in ed:
        if drug and want is None:
            hits.append({"kind": "drug_not_in_dosing_map", "severity": GEN20_SEVERITY,
                         "detail": f"{drug}: not registered in DRUG_DOSES_PER_WEEK, dosing frequency cannot be checked"})
        return hits
    got = int(ed["dosing_per_week"] or 0)
    if got and got != want:
        hits.append({"kind": "dosing_route_mismatch", "severity": GEN20_SEVERITY,
                     "detail": f"{drug} should be {want}x/week (weekly formulation=1 · oral daily formulation=7), "
                               f"premise declares {got}x/week"})
    return hits


def check_missingness(longitudinal_data: dict, premise, cs, T: int) -> list[dict]:
    """GEN20c: with a declared `missingness_mechanism`, the day-to-day missing rate before and
    after the reversal point must not differ by more than `MISSINGNESS_SPLIT_TOL`, or gaps
    become a proxy for the outcome. Noise windows are excluded.
    """
    hits: list[dict] = []
    mech = str(((getattr(premise, "adherence", None) or {})
                .get("missingness_mechanism") or "")).upper()
    if not mech:
        return hits
    spec = (getattr(premise, "device_signals", None) or {}).get("signals", {}) or {}
    wins = _noise_windows(cs)
    rev = int(getattr(cs, "latent", {}).get("reversal_week", 0) or 0) * 7
    if not rev or rev >= T:
        return []  # reversal point outside the observation window
    for sig, pts in (longitudinal_data or {}).items():
        sspec = spec.get(sig) or {}
        step = int(sspec.get("sampling_days", 0) or 0)
        if step <= 0 or "expected_missing_rate" not in sspec:
            continue
        seen = {int(p["ts"]) for p in pts}
        def _rate(lo: int, hi: int) -> float | None:
            grid = [d for d in range(lo, hi + 1, step)
                    if not any(a <= d <= b for a, b in wins)]
            return (None if len(grid) < MISSINGNESS_MIN_GRID
                    else 1.0 - sum(1 for d in grid if d in seen) / len(grid))
        a, b = _rate(0, rev - 1), _rate(rev, T)
        if a is None or b is None:
            continue
        if abs(a - b) > MISSINGNESS_SPLIT_TOL:
            hits.append({"kind": "missingness_not_answer_neutral", "severity": GEN20_SEVERITY,
                         "detail": f"{sig}: declared {mech}, but the missing rate before the reversal point is {a:.3f} / after is {b:.3f}"
                                   f" (diff {abs(a-b):.3f} > {MISSINGNESS_SPLIT_TOL}) -- "
                                   f"the gap itself became a proxy for the outcome"})
    return hits


MISSINGNESS_SPLIT_TOL = 0.15  # max missing-rate difference across the reversal point


# ============================================================ GEN22: a declared anchor must be delivered
# Declared `start_weight` / `nadir_weight` must appear in the emitted series.
# `anchor_infeasible`: nadir above start; `anchor_not_honored`: feasible but not delivered.
# The 0.5 kg tolerance is about 4x the rendering's quantization error.
GEN22_SEVERITY = "gate"
ANCHOR_TOL_KG = 0.5


def check_anchors_honored(raw, cs) -> list[dict]:
    """GEN22: a weight anchor declared in `cs.raw` must be delivered in the emitted
    series."""
    hits: list[dict] = []
    decl = getattr(cs, "raw", None) or {}
    sw, nw = decl.get("start_weight"), decl.get("nadir_weight")
    if sw is None and nw is None:
        return hits
    if sw is not None and nw is not None and float(nw) > float(sw) + ANCHOR_TOL_KG:
        hits.append({"kind": "anchor_infeasible", "severity": GEN22_SEVERITY,
                     "detail": f"declared nadir_weight={nw} > start_weight={sw} -- "
                               f"the lowest point cannot be higher than the starting point, this set of constraints has no solvable world"})
        return hits
    pts = [float(p["value"]) for p in ((raw.longitudinal_data or {}).get("weight") or [])]
    if not pts:
        hits.append({"kind": "anchor_not_honored", "severity": GEN22_SEVERITY,
                     "detail": "a weight anchor was declared, but the emitted world has no weight series"})
        return hits
    for name, want, got in (("start_weight", sw, pts[0]), ("nadir_weight", nw, min(pts))):
        if want is None:
            continue
        if abs(float(got) - float(want)) > ANCHOR_TOL_KG:
            hits.append({"kind": "anchor_not_honored", "severity": GEN22_SEVERITY,
                         "detail": f"declared {name}={float(want):.1f}, emitted as {got:.1f}"
                                   f" (diff {abs(got - float(want)):.1f} kg > {ANCHOR_TOL_KG}) -- "
                                   f"what job.yaml wrote never appeared in the world"})
    return hits

# GEN19 (gate): question text must not contradict the declared sex's anatomy.
GEN19_SEVERITY = "gate"

# Anatomically exclusive words only, not epidemiological tendencies.
SEX_EXCLUSIVE: dict[str, tuple[str, ...]] = {
    "M": ("月经", "经期", "怀孕", "妊娠", "宫颈", "子宫", "卵巢", "阴道", "绝经", "痛经", "排卵"),
    "F": ("前列腺", "睾丸", "阴茎", "精液", "遗精"),
}


GEN24_SEVERITY = "gate"


def check_demographic_plausibility(raw, cs) -> list[dict]:
    """GEN24: the case's age band must satisfy the condition's declared `min_age` / `max_age`.

    Some golds are age-dependent (new-onset diabetes with weight loss suggests pancreatic
    cancer after 50, T1D/LADA at 25). This is a gate rather than a sampling constraint because
    `sample_profile` deliberately never reads the diagnosis.
    """
    spec_id = str((cs.latent or {}).get("ddx_spec_id") or "")
    if not spec_id:
        return []
    try:
        from .overlay import condition_registry
        spec = condition_registry(include_draft=True).get(spec_id) or {}
    except Exception:                                      # noqa: BLE001
        return []
    # Lower bound against `min_age`, upper bound against `max_age`.
    _need_lo, _need_hi = spec.get("min_age"), spec.get("max_age")
    if _need_lo is None and _need_hi is None:
        return []
    band = str(((raw.user_profile or {}) if hasattr(raw, "user_profile") else {})
               .get("age_range") or (cs.raw or {}).get("age_range") or "")
    lo = hi = None
    if "-" in band:
        try:
            lo, hi = (int(x) for x in band.split("-", 1))
        except ValueError:
            lo = hi = None
    if lo is None:
        return [{"kind": "demographic_plausibility_unknown", "severity": GEN24_SEVERITY,
                 "detail": f"{spec_id} declares min_age={_need_lo}/max_age={_need_hi}, "
                           f"but the question face's age band cannot be read ({band!r}) -- unreadable does not count as passing"}]
    if _need_lo is not None and lo < int(_need_lo):
        return [{"kind": "demographic_plausibility", "severity": GEN24_SEVERITY,
                 "detail": f"{spec_id} declares min_age={_need_lo}, "
                           f"but this case's age band is {band} => this gold does not hold for this patient (do not emit)"}]
    if _need_hi is not None and int(hi) > int(_need_hi):
        return [{"kind": "demographic_plausibility", "severity": GEN24_SEVERITY,
                 "detail": f"{spec_id} declares max_age={_need_hi}, "
                           f"but this case's age band is {band} => this gold does not hold for this patient (do not emit)"}]
    return []


#: Main-line disease codes -> their spellings in the Chinese gold text, used by the
#: comorbidity exclusion check and by `check_gold_line_in_known_conditions`.
CONDITION_ALIASES: dict[str, tuple[str, ...]] = {
    "obesity": ("肥胖",),
    "T2D": ("2型糖尿病", "糖尿病"),
    "hypertension": ("高血压",),
    "dyslipidemia": ("血脂异常", "高脂血症"),
    "MASLD": ("脂肪肝", "脂肪性肝病"),
}


def check_gold_line_in_known_conditions(raw, cs) -> list[dict]:
    """GEN24b: `known_conditions` must not name one of the gold's diagnostic threads.

    Matched with `tracks.alias_hit_asserted`, the scorer's own function.
    """
    spec_id = str((cs.latent or {}).get("ddx_spec_id") or "")
    if not spec_id:
        return []
    known = [str(x) for x in ((raw.user_profile or {}) if hasattr(raw, "user_profile") else {})
             .get("known_conditions") or ()]
    if not known and raw is None:
        # Before the prompt exists (raw=None), `known_conditions` comes from the main-line disease.
        _r = cs.raw if isinstance(cs.raw, dict) else getattr(cs.raw, "__dict__", {}) or {}
        _d = _r.get("disease")
        known = [str(_d)] if _d else []
    if not known:
        return []
    try:
        from .overlay import condition_registry
        from .tracks import alias_hit_asserted
        spec = condition_registry(include_draft=True).get(spec_id) or {}
    except Exception:                                      # noqa: BLE001
        return []
    threads = [t for t in (spec.get("_threads") or ()) if t.get("aliases")]
    if not threads and spec.get("aliases"):
        # A single-line spec is its own thread.
        threads = [{"name": spec.get("diagnosis") or spec_id,
                    "aliases": list(spec["aliases"])}]
    # Expand codes into the gold's spellings (e.g. `T2D` -> the Chinese aliases).
    blob = " ".join(known + [a for k in known for a in CONDITION_ALIASES.get(k, ())])
    hit = [t.get("name") for t in threads if alias_hit_asserted(blob, list(t["aliases"]))]
    if not hit:
        return []
    return [{"kind": "gold_line_in_known_conditions", "severity": GEN24_SEVERITY,
             "detail": f"{spec_id}: the question face's `known_conditions` {known} says out loud the gold's "
                       f"thread {hit} -- under \"caught both threads or not\" scoring, this gives away half of the gold for free"}]


def check_sex_consistency(raw, cs) -> list[dict]:
    """GEN19: the question text must not contradict the declared sex (the verifier for
    `patient_basics.sex`).
    """
    sex = str((cs.raw or {}).get("sex", "") or "").upper()[:1]
    bad = SEX_EXCLUSIVE.get(sex)
    if not bad:
        return []
    hits: list[dict] = []
    for e in (raw.evidence_ledger or []):
        blob = f"{e.get('symptom') or ''} {e.get('note') or ''} {e.get('context') or ''}"
        got = [w for w in bad if w in blob]
        if got:
            hits.append({"kind": "sex_inconsistent_text", "severity": GEN19_SEVERITY,
                         "detail": f"declared sex={sex}, question face {e.get('evidence_id')} contains {got} "
                                   f"-- anatomically mutually exclusive"})
    return hits


# GEN18 (gate): symptom days in an arithmetic progression that no injected event shares let
# signal be told from noise by timing alone.
GEN18_SEVERITY = "gate"

#: Minimum true symptoms before GEN18 judges a case (a sample size).
#: With 1-2 points any day set is "separable", so a hit would not be evidence.
#: Few cases fall below it, so the gate still covers most of the corpus.
GEN18_MIN_REAL_SYMPTOMS = 3

#: Points needed before "arithmetic progression" means anything.
#: Any two points form one, so three is the mathematical minimum; it does not move with the corpus.
#: Kept separate from GEN18_MIN_REAL_SYMPTOMS, which is a sample size.
GEN18_MIN_ARITHMETIC_POINTS = 3


EV_ID_OPAQUE_SEVERITY = "gate"
_EV_OPAQUE_RE = __import__("re").compile(r"^EV-.+-\d+$")


def check_ev_id_opaque(sp) -> list[dict]:
    """GEN21: EV ids in the solver payload must be opaque (`EV-...-<n>`, no category prefix).

    Checks the payload itself, since `build.anonymize_evidence_ids` could be skipped or moved.
    """
    bad = [str(e.get("evidence_id", "")) for e in (sp.evidence_ledger or [])
           if not _EV_OPAQUE_RE.match(str(e.get("evidence_id", "")))]
    if not bad:
        return []
    return [{"kind": "ev_id_not_opaque", "severity": EV_ID_OPAQUE_SEVERITY,
             "detail": f"{len(bad)} EV ids still carry a category prefix (the signal/noise split is written straight onto the question face): "
                       f"{bad[:5]} -- see build.anonymize_evidence_ids"}]


def check_symptom_day_separability(raw, T: int) -> list[dict]:
    """GEN18: true symptoms must not be separable from injected events by timing alone.

    Reports when the true-symptom days form an exact arithmetic progression and no injected
    event lands on it. Symptom timing comes from the kernel's DDX specs.
    """
    # Ledger required: without it the gate would silently pass every case.
    from . import wq
    man = wq.injected_manifest(getattr(raw, "case_id", ""))
    real_ids = set(man.get("real_symptom_evidence_ids") or [])
    if len(real_ids) < GEN18_MIN_REAL_SYMPTOMS:
        return []
    days = {}
    for e in (raw.evidence_ledger or []):
        d = int(e.get("source_timestamp", -1))
        if 0 <= d <= T:
            days.setdefault(str(e.get("evidence_id", "")) in real_ids, set()).add(d)
    real, other = sorted(days.get(True, set())), days.get(False, set())
    if len(real) < GEN18_MIN_ARITHMETIC_POINTS:
        return []
    steps = {b - a for a, b in zip(real, real[1:])}
    if len(steps) != 1:                       # not an arithmetic progression -> this rule does not apply
        return []
    step = steps.pop()
    grid = {real[0] + k * step for k in range(0, (T - real[0]) // max(1, step) + 1)}
    if other & grid:                          # an injected event lands on the same grid -> the rule does not hold
        return []
    return [{"kind": "symptom_days_separable", "severity": GEN18_SEVERITY,
             "detail": f"the true-symptom days {real} form an exact arithmetic progression with step {step}d, and none of the "
                       f"{len(other)} injected events land on that grid -- the signal/noise split can be recovered from timing alone"}]


GEN8_SEVERITY = "warn"

#: GEN8d: largest share of a batch's cases one distractor text may appear in. A small
#: distractor pool fails it; a diverse pool passes.
GEN8D_MAX_SHARE = 0.30

#: Below this many judgeable cases GEN8d is not judged.
GEN8D_MIN_CASES = 5


# ============================================================ A5: shortcuts must be unsolvable (batch-level)
# No single-feature threshold rule may identify the gold with high precision; otherwise
# the question tests whether a threshold was crossed, not reasoning.
A5_SEVERITY = "warn"
A5_F1 = 0.9          # a single-feature F1 reaching this value counts as shortcut-solvable

#: Margin by which a feature's F1 must exceed the majority-class baseline to be a candidate.
#: Without it, "always answer the majority class" would count as a shortcut.
#: Empirical and coarse; significance is decided separately by `A5_ALPHA`.
A5_MARGIN_OVER_MAJORITY = 0.05
A5_MIN_CLASS = 2  # min class size

#: A5's per-disease existence bits: a copy of the kernel's `latent.DISEASE_SIGNAL_DOMAIN`
#: keys, so the feature schema does not depend on the kernel import and is the same for every
#: batch. The maintainers' tests keep the copy in sync.
A5_DISEASE_BITS: tuple[str, ...] = (
    "obesity", "T2D", "hypertension", "dyslipidemia", "MASLD",
)


# ---------------------------------------------------------------- A5v2: hits are split into clinical / nonclinical
# A `clinical` hit lands on the gold's registered evidence stream (reported; GEN14 requires
# that evidence to exist). A `nonclinical` hit lands on demographics, structure, counts or
# construction artifacts, and rejects emission.
NONCLINICAL_FEATURES = ("n_signals", "n_evidence", "n_context", "ctx_facet:")
# Nonclinical by suffix, whatever the stream: `:is_const` (a construction artifact) and
# `:present` (reflects device inventory and injection choices). `:std` is not listed:
# variance on a registered stream can be clinical.
NONCLINICAL_SUFFIXES = (":is_const", ":present")

#: Streams that legitimately predict an `outcome=` gold at T, per event type.
OUTCOME_REQUIRED_OBSERVABLE: dict[str, set[str]] = {
    # Weight regain: weight plus clinical streams; QC and structural streams excluded.
    "weight_regain": {"weight", "medication_adherence", "dose_timeline", "diet_carb_pct",
                      "CGM_TIR", "HbA1c", "steps", "activity_index", "sleep_hours",
                      "gi_symptom_score", "resting_hr", "hrv", "systolic_bp", "LDL",
                      "stress_score", "body_temp", "spo2", "skin_temp"},
}


def classify_shortcut(target: str, feature: str, event_type: str | None = None) -> str:
    """Whether one A5 hit is `clinical` or `nonclinical`, by GEN14's registry and
    `OUTCOME_REQUIRED_OBSERVABLE`. Unregistered targets are nonclinical.
    """
    if any(feature == k or feature.startswith(k) for k in NONCLINICAL_FEATURES):
        return "nonclinical"
    if any(feature.endswith(k) for k in NONCLINICAL_SUFFIXES):
        return "nonclinical"
    stream = feature.split(":", 1)[0]
    if target.startswith("driver="):
        req = DRIVER_REQUIRED_OBSERVABLE.get(target[len("driver="):])
        if req is None:                      # unregistered -> no exemption
            return "nonclinical"
        return "clinical" if stream in req else "nonclinical"
    if target.startswith("outcome="):
        # Looked up by event type (`event_type`, default `weight_regain`), not by label value.
        req = OUTCOME_REQUIRED_OBSERVABLE.get(event_type or "weight_regain")
        if req is None:
            return "nonclinical"                 # an unregistered event type gets no exemption
        return "clinical" if stream in req else "nonclinical"
    if target.startswith("join_gold="):
        return "nonclinical"
    return "nonclinical"


_LAB_VAL_RE = re.compile(r"([0-9.]+)\s*\S*\(参考\s*([0-9.]+)[–\-]([0-9.]+)\)")


def _lab_features(sp) -> dict[str, float]:
    """Format features of the <=T lab results (counts, timing, abnormal count), never the values:
    a lab value predicting the diagnosis is intended, "how many labs" predicting it is not.
    """
    T = int((sp.prediction_context or {}).get("prediction_time_T", 10 ** 9))
    labs = [e for e in (sp.evidence_ledger or [])
            if e.get("source_type") == "lab_result"
            and int(e.get("source_timestamp", 10 ** 9)) <= T]
    out = {"lab:n": float(len(labs))}
    if labs:
        days = [int(e.get("source_timestamp", 0)) for e in labs]
        out["lab:first_day"] = float(min(days))
        out["lab:span"] = float(max(days) - min(days))
        out["lab:n_distinct_days"] = float(len(set(days)))
        n_ab = 0
        for e in labs:
            m = _LAB_VAL_RE.search(str(e.get("symptom", "")))
            if not m:
                continue
            v, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
            if not (lo <= v <= hi):
                n_ab += 1
        out["lab:n_abnormal"] = float(n_ab)
        out["lab:frac_abnormal"] = round(n_ab / len(labs), 3)
    return out


def case_features(sp) -> dict[str, float]:
    """Scalar features of the solver-visible payload, for the A5 scan."""
    f: dict[str, float] = dict(_lab_features(sp))
    ld = sp.longitudinal_data or {}
    T = int((sp.prediction_context or {}).get("prediction_time_T", 10 ** 9))
    for name, pts in ld.items():
        vis = [p for p in (pts or []) if int(p["ts"]) <= T]
        if not vis:
            continue
        vals = [float(p["value"]) for p in vis]
        f[f"{name}:last"] = vals[-1]
        f[f"{name}:min"] = min(vals)
        f[f"{name}:max"] = max(vals)
        if len(vals) >= 2 and int(vis[-1]["ts"]) != int(vis[0]["ts"]):
            f[f"{name}:slope"] = (vals[-1] - vals[0]) / (int(vis[-1]["ts"]) - int(vis[0]["ts"]))
        if vals[0]:
            f[f"{name}:rel_change"] = (vals[-1] - vals[0]) / abs(vals[0])
        # `:std` / `:is_const`: flat non-gold streams make "variance > 0" a construction shortcut.
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        f[f"{name}:std"] = var ** 0.5
        f[f"{name}:is_const"] = 1.0 if var == 0.0 else 0.0
    f["n_evidence"] = float(len(sp.evidence_ledger or []))
    f["n_signals"] = float(len(ld))

    # Text shape: counts per context facet (`overlay.facet_of_emitted_context`).
    from .overlay import facet_of_emitted_context as _fac
    facets: dict[str, int] = {}
    n_ctx = 0
    for e in (sp.evidence_ledger or []):
        ctx = str(e.get("context") or "").strip()
        if not ctx:
            continue
        n_ctx += 1
        _k = _fac(ctx)
        facets[_k] = facets.get(_k, 0) + 1
    for name in ("measure", "neg", "sign", "course", "therapy", "pool"):
        f[f"ctx_facet:{name}"] = float(facets.get(name, 0))
    f["ctx_facet:course_frac"] = float(facets.get("course", 0)) / max(1, n_ctx)
    f["n_context"] = float(n_ctx)
    # Presence bits for every known stream: driver-proxy exclusion makes absence informative.
    from .events import METRIC_BY_NAME
    for name in sorted(set(METRIC_BY_NAME) | {"weight_ref", "dose_timeline",
                                              "medication_adherence"}):
        f[f"{name}:present"] = 1.0 if name in ld else 0.0

    # ---- demographics / device inventory / stream length ----
    # A5v2 rejects shortcuts on these, so they must be scanned. `getattr`, because some callers
    # pass only longitudinal_data.
    up = (getattr(sp, "user_profile", None) or {})
    age = str(up.get("age_range") or "")
    if "-" in age:                                   # "35-39" -> 35.0 (lower bound; monotonic is enough to scan)
        try:
            f["demo:age_lo"] = float(age.split("-")[0])
        except ValueError:
            pass
    f["demo:sex_is_f"] = 1.0 if str(up.get("sex", "")).upper().startswith("F") else 0.0
    f["demo:n_conditions"] = float(len(up.get("known_conditions") or []))
    # Per-disease bits: `demo:n_conditions` is constant when every case has one disease.
    # Existence bits, not an ordinal, since diseases have no order.
    _kc = {str(c).strip().lower() for c in (up.get("known_conditions") or [])}
    for _d in A5_DISEASE_BITS:
        f[f"cond:{_d}"] = 1.0 if _d.lower() in _kc else 0.0
    f["demo:n_goals"] = float(len(up.get("treatment_goals") or []))
    dev = up.get("device_inventory") or []
    f["demo:n_devices"] = float(len(dev))
    # Per-device bits: the device combination, not only the count.
    for d in ("smart_scale", "wearable", "bp_cuff", "cgm", "clinic_scale"):
        f[f"device:{d}"] = 1.0 if d in dev else 0.0
    for name, pts in ld.items():
        f[f"{name}:n_points"] = float(len([p for p in (pts or []) if int(p["ts"]) <= T]))
    return f


class _ThresholdScan:
    """The candidate thresholds of one feature, prepared once so that each label vector costs a
    single pass. Thresholds, their order and the comparisons (`x > t`, `x <= t` against the
    real midpoint float) are those of the plain per-threshold rescan; only the counting differs.
    """

    def __init__(self, xs: list[float]):
        from bisect import bisect_right
        cand = sorted(set(xs))
        self.mids = [(a + b) / 2 for a, b in zip(cand, cand[1:])] or cand
        order = sorted(range(len(xs)), key=xs.__getitem__)
        sx = [xs[i] for i in order]
        self.order = order
        self.n = len(xs)
        self.cnt_le = [bisect_right(sx, t) for t in self.mids]     # number of x <= t

    def best_index(self, ys) -> tuple[float, int, str]:
        """`(best F1, index of its threshold, op)`; the first threshold reaching the best F1
        wins, `>` before `<=` at one threshold."""
        pref = [0]
        acc = 0
        for i in self.order:
            if ys[i]:
                acc += 1
            pref.append(acc)
        k, n = acc, self.n
        best, bi, bop = 0.0, -1, ""
        for j, c in enumerate(self.cnt_le):
            pos_le = pref[c]
            pos_gt = k - pos_le
            # ">": tp = pos_gt, fp = n - c - pos_gt, fn = pos_le
            if pos_gt:
                f1 = 2 * pos_gt / (2 * pos_gt + (n - c - pos_gt) + pos_le)
                if f1 > best:
                    best, bi, bop = f1, j, ">"
            # "<=": tp = pos_le, fp = c - pos_le, fn = pos_gt
            if pos_le:
                f1 = 2 * pos_le / (2 * pos_le + (c - pos_le) + pos_gt)
                if f1 > best:
                    best, bi, bop = f1, j, "<="
        return best, bi, bop

    def best(self, ys) -> tuple[float, str]:
        f1, bi, op = self.best_index(ys)
        return (f1, f"x {op} {round(self.mids[bi], 4)}") if bi >= 0 else (0.0, "")


def _best_single_threshold_f1(xs: list[float], ys: list[bool]) -> tuple[float, str]:
    """The best single-feature threshold rule's F1 (tries both directions)."""
    if any(x != x for x in xs):                        # NaN: comparisons are not orderable
        return _best_single_threshold_f1_scan(xs, ys)
    return _ThresholdScan(xs).best(ys)


def _best_single_threshold_f1_scan(xs, ys):
    best, how = 0.0, ""
    cand = sorted(set(xs))
    mids = [(a + b) / 2 for a, b in zip(cand, cand[1:])] or cand
    for t in mids:
        for op, pred in ((">", [x > t for x in xs]), ("<=", [x <= t for x in xs])):
            tp = sum(1 for p, y in zip(pred, ys) if p and y)
            fp = sum(1 for p, y in zip(pred, ys) if p and not y)
            fn = sum(1 for p, y in zip(pred, ys) if not p and y)
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + fp + fn)
            if f1 > best:
                best, how = f1, f"x {op} {round(t, 4)}"
    return best, how


def _target_key(tname: str) -> str:
    """A5 target name without its value (`outcome=event_occurred` -> `outcome`)."""
    return tname.split("=", 1)[0]


#: Permutation draws for A5's noise floor (enough for a 95th percentile).
A5_N_PERM = 20

#: A5's family-wise error rate: per-feature exact p times the number of tests run in the
#: tier must be <= alpha (Bonferroni). A permutation floor over the whole family saturates
#: when many correlated features are scanned.
A5_ALPHA = 0.05

#: Beyond this many `C(n,k)` orderings, p is estimated by deterministic sampling.
A5_P_SAMPLES = 4000


_SPLIT_ORDER_CACHE: dict = {}
_SPLIT_ORDER_CACHE_MAX_CELLS = 4_000_000


def _split_order(n: int, t: int) -> list[int]:
    """Permutation `t` of `range(n)`: indices sorted by `sha1(i|t)`. The same for every feature
    of a tier, so it is computed once per `(n, t)`."""
    key = (n, t)
    got = _SPLIT_ORDER_CACHE.get(key)
    if got is None:
        import hashlib as _h
        got = sorted(range(n), key=lambda i: _h.sha1(f"{i}|{t}".encode()).hexdigest())
        if (len(_SPLIT_ORDER_CACHE) + 1) * n > _SPLIT_ORDER_CACHE_MAX_CELLS:
            _SPLIT_ORDER_CACHE.clear()
        _SPLIT_ORDER_CACHE[key] = got
    return got


def _split_p(xs: list, yy: list, f1_obs: float, p_useful: float = 0.0) -> float:
    """Null probability that a single threshold rule reaches `f1_obs`.

    A perfect split has the closed form `2 / C(n, k)`; otherwise deterministic sampling
    (sorted by `sha1(i|t)`). Returns early once the result cannot be below `p_useful`.
    """
    from math import comb
    n, k = len(yy), sum(1 for y in yy if y)
    if k <= 0 or k >= n:
        return 1.0
    p_floor = min(1.0, 2.0 / comb(n, k))        # a perfect split's null probability = the p lower bound for any F1
    if f1_obs >= 1.0 - 1e-12:
        return p_floor
    # Even the lower bound is not significant: skip sampling (exact, changes no verdict).
    if p_useful > 0.0 and p_floor > p_useful:
        return p_floor
    # Early stop once p is certainly above `p_useful` (changes no verdict).
    _stop_at = (int(p_useful * (A5_P_SAMPLES + 1)) + 1) if p_useful > 0.0 else None
    scan = None if any(x != x for x in xs) else _ThresholdScan(xs)
    ge = 0
    for t in range(A5_P_SAMPLES):
        order = _split_order(n, t)
        yp = [yy[i] for i in order]
        f1p = (scan.best_index(yp)[0] if scan is not None
               else _best_single_threshold_f1_scan(xs, yp)[0])
        ge += (f1p >= f1_obs - 1e-12)
        if _stop_at is not None and ge >= _stop_at:
            return (ge + 1) / (t + 2)
    return (ge + 1) / (A5_P_SAMPLES + 1)

#: Nominal minimum case count for an A5 reading to count as evidence (a perfect split across
#: 100+ features can occur by chance in small batches). Blocking itself follows per-tier
#: statistical power (`_powered` in `check_shortcut`); the pipeline does not read this constant.
A5_MIN_CASES = 15

#: A feature's range must reach this fraction of its own magnitude before a split counts;
#: below it the feature is near constant (QC artifacts sit near 5%, real features above 50%).
A5_MIN_REL_SPAN = 0.10

def near_constant(xs: list[float]) -> bool:
    """Whether the feature's range is below `A5_MIN_REL_SPAN` of its mean absolute value.
    Shared with the maintainers' label-independence tool.
    """
    if not xs:
        return True
    _span = max(xs) - min(xs)
    _scale = sum(abs(x) for x in xs) / len(xs)
    return not (_span > 0 and (_span / max(1e-9, _scale)) >= A5_MIN_REL_SPAN)


#: Feature suffixes in the stream's own unit, scaled by its physiological range.
_LEVEL_SUFFIXES = ("last", "min", "max")


def near_constant_feature(name: str, xs: list[float]) -> bool:
    """`near_constant`, with the right scale for level features.

    For `last` / `min` / `max` the span is compared with the width of the stream's
    `hard_range` (half a degree of body temperature is a real separation); other features
    use `near_constant`.
    """
    if not xs:
        return True
    base, _, suffix = name.partition(":")
    if suffix in _LEVEL_SUFFIXES:
        from .events import METRIC_BY_NAME
        m = METRIC_BY_NAME.get(base)
        if m:
            lo, hi = m.hard_range
            width = float(hi) - float(lo)
            if width > 0:
                return not ((max(xs) - min(xs)) / width >= A5_MIN_REL_SPAN)
    return near_constant(xs)

_PERM_CACHE: dict = {}


def _perm_floor(cases: list, tname: str, ys: list, feat_names: list,
                event_type: str | None) -> float:
    """Noise floor: the 95th percentile, over label shuffles, of the best single feature's
    excess over the majority baseline. Deterministic shuffles (`sha1(case_id|k)`).
    """
    import hashlib as _h
    key = (id(cases), tname)
    if key in _PERM_CACHE:
        return _PERM_CACHE[key]
    cids = [c[0] for c in cases]
    nulls: list[float] = []
    for k in range(A5_N_PERM):
        order = sorted(range(len(cases)),
                       key=lambda i: _h.sha1(f"{cids[i]}|{k}".encode()).hexdigest())
        yp = [ys[i] for i in order]
        tp0 = sum(yp)
        if tp0 == 0 or tp0 == len(yp):
            continue
        best = 0.0
        for fn in feat_names:
            xs, yy = [], []
            for (_, f, _, _, _), y in zip(cases, yp):
                if fn in f:
                    xs.append(f[fn]); yy.append(1 if y else 0)
            if len(xs) < 4 or len(set(xs)) < 2 or sum(yy) < A5_MIN_CLASS:
                continue
            if classify_shortcut(tname, fn, event_type=event_type) == "clinical":
                continue
            # `base` on this feature's own subset, as the caller computes `f1_major`.
            tp_f = sum(yy)
            base = 2 * tp_f / (2 * tp_f + (len(yy) - tp_f))
            f1n, _ = _best_single_threshold_f1(xs, yy)
            best = max(best, f1n - base)
        nulls.append(best)
    nulls.sort()
    floor = nulls[int(0.95 * (len(nulls) - 1))] if nulls else 0.0
    _PERM_CACHE[key] = floor
    return floor


def check_shortcut(cases: list, graded_targets: set[str] | None = None,
                   event_type: str | None = None) -> list[dict]:
    """A5. cases = [(case_id, features, gold_driver, outcome_label[, join_gold]), ...].

    For each gold category (driver values, outcome, join_gold values) x feature, fits the best
    single threshold rule and reports F1 >= A5_F1 with significance (`A5_ALPHA`).

    `graded_targets`: hits on targets outside this set get `class="ungraded"` -- reported,
    never blocked. `None` scores everything; `cli.py` exempts outcome only when every case
    carries `outcome_rule_not_applicable`.
    """
    hits: list[dict] = []
    cases = [tuple(c) + (None,) * (5 - len(c)) for c in cases]
    if len(cases) < 4:
        return hits
    feat_names = sorted(set().union(*[set(f) for _, f, _, _, _ in cases]))
    targets: list[tuple[str, list[bool]]] = []
    def _usable(ys: list[bool]) -> bool:
        # Smaller side >= A5_MIN_CLASS: a 1-vs-19 split is an outlier, not a shortcut.
        return min(sum(ys), len(ys) - sum(ys)) >= A5_MIN_CLASS

    # Record skipped targets, so a constant-gold tier reads as "not scanned", not "clean".
    _skipped: list[str] = []

    def _add(name: str, ys: list[bool]) -> None:
        if _usable(ys):
            targets.append((name, ys))
        else:
            _skipped.append(name)

    for g in sorted({g for _, _, g, _, _ in cases if g}):
        _add(f"driver={g}", [gd == g for _, _, gd, _, _ in cases])
    _add("outcome=event_occurred", [o == "event_occurred" for _, _, _, o, _ in cases])
    for j in sorted({j for _, _, _, _, j in cases if j}):
        _add(f"join_gold={j}", [jg == j for _, _, _, _, jg in cases])

    # Report a gold family that was never scanned -- only if it has values and is scored for
    # this question type.
    for _fam in ("driver", "outcome", "join_gold"):
        if any(t.startswith(_fam) for t, _ in targets):
            continue
        _n = sum(1 for s in _skipped if s.startswith(_fam))
        if not _n or (graded_targets is not None and _fam not in graded_targets):
            continue
        _why = f"all {_n} targets were skipped because the smaller side < {A5_MIN_CLASS} (gold is nearly constant)"
        hits.append({
            # Explicit class: `cli.py` treats a missing class as nonclinical, which would block.
            "kind": "shortcut_target_unscannable", "severity": "warn",
            "class": "unscannable",
            "target_family": _fam, "n_targets_skipped": _n,
            "detail": (f"`{_fam}` tier: {_why} => A5 scanned nothing at all on this dimension. "
                       f"Do not read this batch's \"no {_fam} shortcut\" as having been scanned."),
        })

    # Bonferroni denominator: tests actually run per target. The filter must stay identical to
    # the loop below.
    _n_tests: dict[str, int] = {}
    for tname, ys in targets:
        _c = 0
        for fn in feat_names:
            xs, yy = [], []
            for (_, f, _, _, _), y in zip(cases, ys):
                if fn in f:
                    xs.append(f[fn]); yy.append(y)
            if len(xs) < 4 or sum(yy) < A5_MIN_CLASS or len(set(xs)) < 2:
                continue
            _c += 1
        _n_tests[tname] = max(1, _c)

    for tname, ys in targets:
        for fn in feat_names:
            xs, yy = [], []
            for (_, f, _, _, _), y in zip(cases, ys):
                if fn in f:
                    xs.append(f[fn]); yy.append(y)
            if len(xs) < 4 or sum(yy) < A5_MIN_CLASS or len(set(xs)) < 2:
                continue
            f1, how = _best_single_threshold_f1(xs, yy)
            # Baseline: predict positive for everything.
            tp0 = sum(yy)
            f1_major = 2 * tp0 / (2 * tp0 + (len(yy) - tp0)) if tp0 else 0.0
            _eff_ok = not near_constant_feature(fn, xs)
            # Candidates that are not significant are still reported. `_powered` is False when even a
            # perfect split could not reach significance ("unmeasurable", not "clean").
            _lift = f1 - f1_major
            _ntest = _n_tests.get(tname, 1)
            _p = _split_p(xs, yy, f1, p_useful=A5_ALPHA / max(1, _ntest))
            _padj = min(1.0, _p * _ntest)
            from math import comb as _comb
            _kk = sum(1 for y in yy if y)
            _pmin = min(1.0, (2.0 / _comb(len(yy), _kk)) * _ntest) if 0 < _kk < len(yy) else 1.0
            _powered = _pmin <= A5_ALPHA
            _sig = _padj <= A5_ALPHA
            _cand = f1 >= max(A5_F1, f1_major + A5_MARGIN_OVER_MAJORITY) and _eff_ok
            cls = classify_shortcut(tname, fn, event_type=event_type)
            if graded_targets is not None and _target_key(tname) not in graded_targets:
                cls = "ungraded"
            if _cand and not _sig:
                # Not significant: reported with class "chance", which never blocks.
                hits.append({
                    "kind": "shortcut_not_significant", "severity": "info",
                    "class": "chance", "target": tname, "feature": fn,
                    "f1": round(f1, 3), "powered": _powered, "p_adj": round(_padj, 5),
                    "detail": (f"[{cls}·not significant] {tname} is solved by `{fn}` to F1={round(f1, 3)}"
                               f"({how})· p={_p:.2e} × {_ntest} tests = {_padj:.3f} > α={A5_ALPHA}"
                               f"(n={len(xs)}, positives {sum(yy)}) -- "
                               + (f"this tier has no power: even a perfect split would, after correction, reach only "
                                  f"{_pmin:.3f} at best => recorded as \"unpowered\", not read as \"no shortcut\""
                                  if not _powered else "indistinguishable from what shuffled labels can already achieve"))})
            if _cand and _sig:
                # A clinical feature solving the question to F1 >= 0.95 is reported as "too easy".
                if cls == "clinical" and f1 >= 0.95:
                    hits.append({"kind": "shortcut_clinical_saturated", "severity": "warn",
                                 "class": "clinical", "target": tname, "feature": fn,
                                 "f1": round(f1, 3),
                                 "detail": f"[clinical·saturated] {tname} can be solved by a single clinical feature "
                                           f"`{fn}` to F1={round(f1, 3)}({how}) -- "
                                           f"legitimate but too easy: this question needs no cross-time synthesis"})
                hits.append({"kind": "shortcut_solvable", "severity": A5_SEVERITY,
                             "class": cls,          # clinical = reported, not blocked / nonclinical = rejects emission
                             "target": tname, "feature": fn, "f1": round(f1, 3),
                             "detail": f"[{cls}] {tname} can be solved by the single feature `{fn}`: {how} · F1={round(f1, 3)}"
                                       f"(n={len(xs)}, positives {sum(yy)}, "
                                       f"predict-all-positive baseline F1={round(f1_major, 3)})"})
    return hits


GEN23_SEVERITY = "gate"

# Days a stream's last point may fall before the declared course end: two steps of the
# coarsest sampling (28 days).
STREAM_SHORT_TOL_DAYS = 56


def declared_course_end(raw) -> int | None:
    """Declared course end = `prediction_time_T + prediction_window`, or `None`.

    `prediction_window` is derived from the course end in `build.py`.
    """
    pc = getattr(raw, "prediction_context", None) or {}
    t, win = pc.get("prediction_time_T"), pc.get("prediction_window")
    if t is None or win is None:
        return None
    try:
        return int(t) + int(str(win).strip().rstrip("dD"))
    except (TypeError, ValueError):
        return None


#: Chance-corrected floor (|phi| = |MCC|) for path B, ANDed with `A5_F1`: under class
#: imbalance an uninformative footprint can still reach F1 >= 0.9, while phi stays 0.
FOOTPRINT_PHI_FLOOR = 0.2

#: Path A's phi floor: only rules out uninformative footprints whose precision is high from
#: class imbalance alone (3 real / 30 benign, all footprinted: precision 0.909, phi 0.0).
FOOTPRINT_PHI_NONZERO = 0.05


def footprint_discriminability(rows: list[tuple[bool, bool]]) -> dict:
    """`[(is_real_symptom, has_physio_footprint)]` -> can the footprint identify true symptoms.

    Returns `{n_real, n_benign, real_cov, benign_cov, f1, precision, phi, direction, usable}`;
    `precision` and `f1` take the better direction, `phi` is the Matthews coefficient.
    Precision matters because this shortcut needs no recall.
    """
    real = [fp for is_real, fp in rows if is_real]
    benign = [fp for is_real, fp in rows if not is_real]
    n_r, n_b = len(real), len(benign)
    out = {"n_real": n_r, "n_benign": n_b,
           "real_cov": (sum(real) / n_r) if n_r else None,
           "benign_cov": (sum(benign) / n_b) if n_b else None,
           "f1": None, "precision": None, "phi": None, "direction": None,
           "usable": min(n_r, n_b) >= A5_MIN_CLASS}
    if not out["usable"]:
        return out
    best_f1, best_p, best_dir = 0.0, 0.0, None
    for pos_is_real in (True, False):
        tp = sum(1 for is_real, fp in rows if fp and (is_real == pos_is_real))
        fp_ = sum(1 for is_real, fp in rows if fp and (is_real != pos_is_real))
        fn = sum(1 for is_real, fp in rows if not fp and (is_real == pos_is_real))
        f1 = 0.0 if not tp else 2 * tp / (2 * tp + fp_ + fn)
        prec = 0.0 if not (tp + fp_) else tp / (tp + fp_)
        best_f1 = max(best_f1, f1)
        if prec > best_p:
            best_p, best_dir = prec, "footprint=>true symptom" if pos_is_real else "footprint=>benign event"
    out["f1"] = round(best_f1, 4)
    out["precision"], out["direction"] = round(best_p, 4), best_dir
    # phi is direction-independent; an empty row or column means zero discriminability.
    a = sum(1 for is_real, fp in rows if is_real and fp)          # real & footprinted
    b = sum(1 for is_real, fp in rows if is_real and not fp)      # real & no footprint
    c = sum(1 for is_real, fp in rows if not is_real and fp)      # benign & footprinted
    d = sum(1 for is_real, fp in rows if not is_real and not fp)  # benign & no footprint
    den = (a + b) * (c + d) * (a + c) * (b + d)
    out["phi"] = 0.0 if den == 0 else round(abs((a * d - b * c) / den ** 0.5), 4)
    return out


def check_footprint_not_discriminative(built: dict) -> list[dict]:
    """Batch-level: "has a physiological footprint" must not identify true symptoms.

    Kernel footprints attach by `topic`, which only benign and life events carry, so a
    footprint could mark an event as benign. The check fits `(is_real, has_footprint)` pairs
    across the batch. It fails when one side has footprints and the other none, or on path A
    or B (see `FOOTPRINT_PHI_NONZERO` / `FOOTPRINT_PHI_FLOOR`). A single class, zero
    footprints and a batch where the physiology layer did not run are reported, not passed.
    """
    from .physio import kernel as _pk
    from . import wq as _wq

    try:
        _kt = _pk.load_physio_registry(_REG_DIR / "physio_kernels.yaml")
        emitted = set(_kt.emitted_topics())
    except Exception as e:                                         # noqa: BLE001
        return [{"kind": "footprint_registry_unreadable", "severity": "gate",
                 "detail": f"{type(e).__name__}: {e}"}]

    # First check that the physiology layer ran (a `physio` report in the Q-side ledger): the
    # kernel table alone only says which footprints could exist.
    ran = [str(cid) for cid in built
           if (_wq.injected_manifest(str(cid), required=False) or {}).get("physio")]
    if not ran:
        return [{"kind": "footprint_scan_physio_off", "severity": "info",
                 "detail": f"the physiology layer did not run (none of this batch's {len(built)} cases has a `physio` report in the Q-side ledger)"
                           f" => not a single footprint exists, this judge has no object (not read as \"passed\")"}]

    rows: list[tuple[bool, bool]] = []
    for cid in built:
        man = _wq.injected_manifest(str(cid), required=False) or {}
        if not man.get("physio"):
            continue  # cases outside the physiology layer are left out
        real_ids = {str(x) for x in (man.get("real_symptom_evidence_ids") or [])}
        for s in (man.get("event_schedule") or []):
            eid = str(s.get("evidence_id", ""))
            kind = str(s.get("kind", ""))
            # Patient self-reports only.
            if kind not in ("real_symptom", "benign_symptom", "life_event", "lookalike"):
                continue
            rows.append((eid in real_ids or kind == "real_symptom",
                         str(s.get("topic") or "") in emitted))

    v = footprint_discriminability(rows)
    base = (f"true symptoms {v['n_real']} (footprint rate {v['real_cov']})· "
            f"benign/distractor {v['n_benign']} (footprint rate {v['benign_cov']})")
    # No footprints on either side: not active, reported.
    if not any(fp for _, fp in rows):
        return [{"kind": "footprint_scan_no_footprints", "severity": "warn",
                 "detail": f"{base} -- zero footprints across the whole batch: the kernel table has no object among the topics sampled in this batch, "
                           f"this judge cannot judge this case (not read as \"no shortcut installed\")"}]
    # Footprints on exactly one side: a perfect discriminator in either direction.
    _rc, _bc = (v["real_cov"] or 0), (v["benign_cov"] or 0)
    if v["n_real"] and v["n_benign"] and (_rc > 0) != (_bc > 0):
        _side = "the true-symptom side has footprints and the benign/distractor side has none at all" if _rc > 0 else "the benign side has footprints and the true-symptom side has none at all"
        _rule = "\"has a footprint => true symptom\"" if _rc > 0 else "\"has a footprint => benign\""
        return [{"kind": "footprint_discriminates_real_symptom", "severity": "gate",
                 "detail": f"perfect discriminator: {base} -- {_side} => {_rule} is always true on this batch. "
                           f"The kernel can only attach to events carrying a `topic`, and whichever side has no topic becomes the discriminating feature"}]
    if not v["usable"]:
        return [{"kind": "footprint_scan_single_class", "severity": "warn",
                 "detail": f"{base} -- only one class has samples (threshold {A5_MIN_CLASS}), this batch is unmeasured. "
                           f"`early_warning` has zero true symptoms per case, which is the norm on that line"}]
    # Two paths (either fails the batch):
    #   A: `precision >= A5_F1` AND `|phi| >= FOOTPRINT_PHI_NONZERO`
    #   B: `F1 >= A5_F1`  AND `|phi| >= FOOTPRINT_PHI_FLOOR`
    # A catches precise rules with low recall; its phi term removes class-imbalance artifacts.
    _p, _f1, _phi = (v["precision"] or 0), (v["f1"] or 0), (v["phi"] or 0)
    _hit_a = _p >= A5_F1 and _phi >= FOOTPRINT_PHI_NONZERO
    _hit_b = _f1 >= A5_F1 and _phi >= FOOTPRINT_PHI_FLOOR
    if _hit_a or _hit_b:
        _why = (f"precision={_p} >= {A5_F1} and |φ|={_phi} >= {FOOTPRINT_PHI_NONZERO}" if _hit_a
                else f"F1={_f1} >= {A5_F1} and |φ|={_phi} >= {FOOTPRINT_PHI_FLOOR}")
        return [{"kind": "footprint_discriminates_real_symptom", "severity": "gate",
                 "detail": f"{_why}({v['direction']})· {base}"}]
    return [{"kind": "footprint_scan_ok", "severity": "info",
             "detail": f"precision={v['precision']}(threshold {A5_F1})· F1={v['f1']}(threshold {A5_F1})"
                       f"· |φ|={v['phi']}(threshold A {FOOTPRINT_PHI_NONZERO} / B {FOOTPRINT_PHI_FLOOR})"
                       f"· {v['direction']} · {base}"}]


def check_coupling_observable(built: dict) -> list[dict]:
    """Batch-level: the coupling layer must have run on every case.

    The layer has no switch, so a case without a `coupling` record in the Q-side ledger means
    it silently did not run -- a gate hit. Rule trigger counts are printed only; "every rule
    fires" is checked over the whole corpus by the maintainers' tests.
    """
    from . import wq as _wq

    missing, fired, pend = [], collections.Counter(), set()
    n_coupled = 0
    for cid in built:
        man = _wq.injected_manifest(str(cid), required=False) or {}
        rec = man.get("coupling")
        if rec is None:
            missing.append(str(cid))
            continue
        hits = [str(r) for r in (rec.get("fired_rules") or [])]
        n_coupled += bool(hits)
        for r in hits:
            fired[r] += 1
        pend.update(str(r) for r in (rec.get("pending_review") or []))
    if missing:
        return [{"kind": "coupling_layer_did_not_run", "severity": "gate",
                 "detail": f"{len(missing)}/{len(built)} cases' Q-side ledger has no `coupling` record"
                           f"(e.g. {missing[:3]}) -- this layer has no on/off switch, so a missing record means it did not run; "
                           f"its output would then look the same as \"this patient has no comorbidity\""}]
    base = (f"{n_coupled}/{len(built)} cases triggered coupling · per-rule {dict(fired) or '{}'}"
            + (f" · triggered rules with review=pending: {sorted(pend)}" if pend else ""))
    if not fired:
        return [{"kind": "coupling_scan_no_subject", "severity": "info",
                 "detail": f"{base} -- this batch has no matching comorbid patients (not read as \"the coupling layer passed\"; "
                           f"the \"the layer is broken\" side is caught by `coupling_layer_did_not_run`)"}]
    return [{"kind": "coupling_scan_ok", "severity": "info", "detail": base}]


def check_observed_in_domain(raw, p) -> list[dict]:
    """Value-range gate for the observed series.

    With the physiology layer on, the kernel's `premise_conflicts` is fed the true
    trajectory, so its `value_out_of_physio_range` check is re-applied here to the observed
    values (range only, not slope). Ranges come from the kernel's `DISEASE_SIGNAL_DOMAIN` and
    `synth.AUX_SIGNALS` streams are skipped, which in practice leaves `weight`.
    """
    from latent import DISEASE_SIGNAL_DOMAIN
    from synth import AUX_SIGNALS

    disease = (p.patient_basics or {}).get("disease")
    domain = DISEASE_SIGNAL_DOMAIN.get(disease) or {}
    out: list[dict] = []
    n_judged = 0
    for sig, pts in (raw.longitudinal_data or {}).items():
        if sig in AUX_SIGNALS:
            continue
        dom = domain.get(sig)
        if dom is None:
            # Not in the disease domain: reported by `signal_not_in_disease_domain` already.
            continue
        lo, hi = dom["range"]
        n_judged += 1
        for pt in pts or []:
            v = float(pt["value"])
            if not (lo <= v <= hi):
                out.append({"kind": "observed_value_out_of_physio_range", "severity": "gate",
                            "detail": f"{sig}={v}@{pt['ts']} ∉ [{lo}, {hi}] (observed series)"})
    if not n_judged:
        # Nothing judged: reported (warn), not passed.
        out.append({"kind": "observed_domain_scan_empty", "severity": "warn",
                    "detail": f"disease={disease!r} has not a single non-AUX signal in the disease domain "
                              f"=> this gate has zero objects on this case (not \"passed\")"})
    return out


GEN26_SEVERITY = "gate"


def check_clinical_baseline_cohort(raw, premise) -> list[dict]:
    """GEN26: an undiagnosed patient must not start the course already past a diagnostic threshold.

    `CLINICAL_SPEC` baselines ignore disease, so a patient declared obese only could start
    with a diabetic HbA1c. For each signal in `diagnostic_thresholds`, the first reading must
    stay below the threshold unless the persona declares the condition. Covers only
    registered signals and `ge` thresholds.
    """
    from .build import _clinical_baselines

    reg = _clinical_baselines()
    thr = reg.get("diagnostic_thresholds") or {}
    markers = [str(m).lower() for m in (reg.get("diabetes_markers") or [])]
    ld = dict(getattr(raw, "longitudinal_data", None) or {})
    up = dict(getattr(raw, "user_profile", None) or {})
    pb = (getattr(premise, "patient_basics", None) or {}) if premise is not None else {}
    hay = " ".join(
        [str(x) for x in (up.get("known_conditions") or [])]
        + [str(x) for x in (pb.get("comorbidities") or [])]
        + [str(pb.get("disease") or "")]).lower()

    hits: list[dict] = []
    n_judged = 0
    for sig, rule in thr.items():
        pts = ld.get(sig)
        if not isinstance(pts, list) or not pts:
            continue
        ge = rule.get("ge")
        if ge is None:
            continue
        first = min((q for q in pts
                     if isinstance(q, dict) and isinstance(q.get("ts"), (int, float))
                     and q.get("value") is not None),
                    key=lambda q: int(q["ts"]), default=None)
        if first is None:
            continue
        n_judged += 1
        cond = str(rule.get("condition") or "")
        declared = (any(m in hay for m in markers) if cond == "diabetes"
                    else cond.lower() in hay)
        if float(first["value"]) >= float(ge) and not declared:
            hits.append({"kind": "clinical_baseline_without_diagnosis",
                         "severity": GEN26_SEVERITY,
                         "detail": f"{sig}@ts={int(first['ts'])} = {first['value']} "
                                   f">= {ge}{rule.get('unit', '')}(diagnostic threshold: {cond})"
                                   f", while the persona does not declare {cond} -- \"{hay.strip() or '(empty)'}\""
                                   f" => the question face contradicts the persona from the very start"})
    if not n_judged:
        # Zero judged signals: "no threshold by design" (declared in `no_diagnostic_threshold`)
        # is info; anything else is warn.
        _clin = {s for s in ld
                 if s in (reg.get("no_diagnostic_threshold") or {}) or s in thr}
        _declared_none = reg.get("no_diagnostic_threshold") or {}
        if _clin and all(s in _declared_none for s in _clin):
            hits.append({"kind": "clinical_baseline_no_threshold_by_design",
                         "severity": "info",
                         "detail": f"this case's clinical streams {sorted(_clin)} are all declared without a diagnostic threshold"
                                   f" (clinical_baselines.no_diagnostic_threshold), so this check has no object"})
        else:
            hits.append({"kind": "clinical_baseline_scan_empty", "severity": "warn",
                         "detail": f"the question face has no stream registered with a diagnostic threshold ({sorted(thr)}), "
                                   f"and this case's clinical streams {sorted(_clin) or '(none)'} also do not declare having no threshold"
                                   f" => this gate has zero objects on this case (not \"passed\")"})
    return hits


# GEN25: ledger `source_type` -> streams its value must be found on. Separate from
# `DEVICE_SIGNALS`: the kernel emits weight EVs under `wearable`.
LEDGER_VALUE_STREAMS: dict[str, set[str]] = {
    "wearable": {"weight"},  # the kernel's weight-EV convention
    "smart_scale": {"weight"},
    "clinic_scale": {"weight_ref", "weight"},
    "cgm": {"CGM_TIR", "fasting_glucose"},
    "bp_cuff": {"systolic_bp", "diastolic_bp"},
    "lab_panel": {"HbA1c", "LDL", "ALT", "AST", "FIB4",
                  "triglycerides", "fasting_glucose"},
    # Values of these types live in text, so they have no numeric object here.
    "lab_result": set(),
    "patient_reported_symptom": set(),
    "patient_reported_context": set(),
}

GEN25_SEVERITY = "gate"
#: Tolerance for downstream re-rounding (streams store `round(v, 2)`).
GEN25_TOL = 0.05


def check_ledger_values_traceable(raw, T: int) -> list[dict]:
    """GEN25: a numeric `measured_value` in the ledger must be found on its stream at or before T.

    A timestamp <= T does not guarantee the value was taken there (e.g. a nadir reached later).

    - strict (`wearable` / `smart_scale`, realigned by `build_case`): the timestamp must be a
      sampling point and its value must match within `GEN25_TOL`;
    - loose (everything else): the value must lie within the <=T range of a candidate stream.

    An unregistered `source_type` is a gate hit.
    """
    hits: list[dict] = []
    led = list(getattr(raw, "evidence_ledger", None) or [])
    ld = dict(getattr(raw, "longitudinal_data", None) or {})
    n_judged = 0
    for ev in led:
        v = ev.get("measured_value")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue  # numeric values only
        eid, st = str(ev.get("evidence_id")), str(ev.get("source_type") or "")
        ts = ev.get("source_timestamp")
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or int(ts) > int(T):
            hits.append({"kind": "ledger_timestamp_after_T", "severity": GEN25_SEVERITY,
                         "detail": f"{eid}: source_timestamp={ts!r} is not an integer <=T={T}"})
            continue
        cands = LEDGER_VALUE_STREAMS.get(st)
        if cands is None:
            hits.append({"kind": "ledger_source_type_unmapped", "severity": GEN25_SEVERITY,
                         "detail": f"{eid}: source_type={st!r} is not registered in LEDGER_VALUE_STREAMS "
                                   f"=> cannot check which stream its value comes from (register one line)"})
            continue
        present = [s for s in cands if isinstance(ld.get(s), list) and ld[s]]
        if not present:
            hits.append({"kind": "ledger_stream_absent", "severity": GEN25_SEVERITY,
                         "detail": f"{eid}: source_type={st!r} carries a value {v}, "
                                   f"but none of the candidate streams {sorted(cands)} appear on the question face"})
            continue
        n_judged += 1
        if cands == {"weight"}:                                   # -- strict tier
            s = present[0]
            at_ts = [float(q["value"]) for q in ld[s]
                     if isinstance(q, dict) and q.get("value") is not None
                     and isinstance(q.get("ts"), (int, float)) and int(q["ts"]) == int(ts)]
            if not at_ts:
                hits.append({"kind": "ledger_timestamp_not_a_sample", "severity": GEN25_SEVERITY,
                             "detail": f"{eid}: source_timestamp={int(ts)} is not a sampling point on {s} "
                                       f"=> the ledger points at a reading the question face does not have (was step ③d's alignment bypassed?)"})
            elif min(abs(x - float(v)) for x in at_ts) > GEN25_TOL:
                hits.append({"kind": "ledger_value_not_on_stream", "severity": GEN25_SEVERITY,
                             "detail": f"{eid}: measured_value={v} does not match {s}@{int(ts)}={at_ts} "
                                       f"(tolerance {GEN25_TOL}) => the ledger value does not come from the question face"})
            continue
        spans, ok = [], False                                     # -- loose tier
        for s in present:
            xs = [float(q["value"]) for q in ld[s]
                  if isinstance(q, dict) and q.get("value") is not None
                  and isinstance(q.get("ts"), (int, float)) and int(q["ts"]) <= int(T)]
            if not xs:
                continue
            lo, hi = min(xs), max(xs)
            spans.append(f"{s}∈[{lo:.2f},{hi:.2f}]")
            if lo - GEN25_TOL <= float(v) <= hi + GEN25_TOL:
                ok = True
                break
        if not ok:
            hits.append({"kind": "ledger_value_not_on_stream", "severity": GEN25_SEVERITY,
                         "detail": f"{eid}: measured_value={v} falls outside every candidate stream's <=T={T} interval "
                                   f"({'; '.join(spans) or 'every candidate stream is empty at <=T'}) "
                                   f"=> the ledger value does not come from the question face (the timestamp is legal but the value may come from the future)"})
    if led and not n_judged and not hits:
        # Nothing judged: reported, not passed.
        hits.append({"kind": "ledger_value_scan_empty", "severity": "warn",
                     "detail": f"the ledger has {len(led)} entries, none carrying a numeric measured_value "
                               f"=> this gate has zero objects on this case (not \"passed\")"})
    return hits


#: A single-reading artifact must stand this far above its neighbours on the observed series,
#: as a fraction of the jump the injector applied. Later layers (events, physiology) may move
#: the reading a little; a jump below half its injected size no longer reads as the artifact.
TRAP_EVIDENCE_MIN_FRACTION = 0.5


def check_trap_evidence(raw) -> list[dict]:
    """GEN27: every single-reading trap in the gold standard must still be visible.

    `noise.transient_spike` / `unit_error` record a trap (`reversal_points`, `type: trap`)
    on the reading they alter. If a later layer drops or rewrites that reading, the gold
    standard keeps a trap whose evidence the agent never sees, and the judges score the
    agent on it anyway. Fires `trap_without_evidence` (gate) when the reading is gone or no
    longer stands at least `TRAP_EVIDENCE_MIN_FRACTION` of its injected jump above the mean
    of its neighbouring readings: the nearest reading on each side that no single-reading
    artifact moved (`noise.trap_evidence_days`, read from the traps), so the second reading
    of a `unit_error` is not taken as its baseline. The injected jump is the one the
    injector recorded on the trap (`jump`); a trap without it (built before this field
    existed) is only checked for the reading being present.

    Runs only when a case is built, and decides emission. This function's own logic lives
    here, in the judging segment: `world_sha` does not cover a change to it, only
    `judging_sha16` does. `trap_evidence_days`, which it reads, is kernel code in the
    generation segment and is covered by `world_sha`; the carried-forward protection uses
    that same function. `build.trap_visibility_conflicts` runs this check on the series cut
    at the last day the solver is shown.
    """
    from noise import POINT_ARTIFACTS, PRIMARY, trap_evidence_days  # kernel
    pts = sorted((getattr(raw, "longitudinal_data", None) or {}).get(PRIMARY) or [],
                 key=lambda q: q["ts"])
    by_ts = {int(q["ts"]): i for i, q in enumerate(pts)}
    moved = trap_evidence_days(raw)
    hits: list[dict] = []
    for rp in getattr(raw, "reversal_points", None) or []:
        cls = rp.get("noise_class")
        if rp.get("type") != "trap" or cls not in POINT_ARTIFACTS or rp.get("day") is None:
            continue
        day = int(rp["day"])
        i = by_ts.get(day)
        if i is None:
            hits.append({"kind": "trap_without_evidence", "severity": "gate",
                         "detail": f"{cls}@day{day}: no {PRIMARY} reading on that day"})
            continue
        nb = []
        for side in (range(i - 1, -1, -1), range(i + 1, len(pts))):
            j = next((j for j in side if int(pts[j]["ts"]) not in moved), None)
            if j is not None:
                nb.append(pts[j]["value"])
        if not nb:
            continue
        base = sum(nb) / len(nb)
        injected = rp.get("jump")                                  # recorded by the injector
        jump = pts[i]["value"] - base
        if injected and jump < TRAP_EVIDENCE_MIN_FRACTION * injected:
            hits.append({"kind": "trap_without_evidence", "severity": "gate",
                         "detail": f"{cls}@day{day}: reading stands {jump:+.2f} above its "
                                   f"neighbours, injected {injected:+.2f}"})
    return hits


def check_stream_horizons(raw) -> list[dict]:
    """GEN23: each stream's last point must fit the declared course end.

    `stream_exceeds_course_end` (gate): later than the course end.
    `stream_horizon_short` (warn): earlier by more than `STREAM_SHORT_TOL_DAYS`, left for a
    human to judge.
    """
    hits: list[dict] = []
    end = declared_course_end(raw)
    if end is None:
        return hits
    for name, pts in (getattr(raw, "longitudinal_data", None) or {}).items():
        if not pts:
            continue
        last = max(int(p["ts"]) for p in pts)
        if last > end:
            hits.append({"kind": "stream_exceeds_course_end", "severity": GEN23_SEVERITY,
                         "detail": f"{name}'s last point {last} > the declared course end {end} -- "
                                   f"the question face shows an observation after the course ended"})
        elif end - last > STREAM_SHORT_TOL_DAYS:
            hits.append({"kind": "stream_horizon_short", "severity": "warn",
                         "detail": f"{name}'s last point {last}, declared course end {end}, "
                                   f"{end - last} days early (> {STREAM_SHORT_TOL_DAYS})"})
    return hits


def check_hardwired_horizons(built: dict) -> list[dict]:
    """GEN23b (batch-level): a stream whose last point is the same in every case and differs
    from the declared course end has a hard-coded horizon.
    """
    hits: list[dict] = []
    if len(built) < 2:
        return hits
    per: dict[str, set] = {}
    ends: dict[str, int] = {}
    for cid, r in built.items():
        e = declared_course_end(r)
        if e is None:
            continue
        ends[cid] = e
        for name, pts in (getattr(r, "longitudinal_data", None) or {}).items():
            if pts:
                per.setdefault(name, set()).add((cid, max(int(p["ts"]) for p in pts)))
    if len(set(ends.values())) < 2:
        # All declared ends equal: hard-coded and following cannot be told apart.
        return [{"kind": "hardwired_horizon_undecidable", "severity": "warn",
                 "detail": f"the declared course end has only {sorted(set(ends.values()))} distinct value(s) -- "
                           f"right now \"the stream's horizon is hard-coded\" and \"it follows the course end\" cannot be told apart; "
                           f"this gate needs the declared course end to differ case by case before it has any discriminating power"}]
    for name, pairs in per.items():
        lasts = {v for _c, v in pairs}
        # Needs at least A5_MIN_CLASS cases carrying the stream; one case is trivially constant.
        if len(pairs) < A5_MIN_CLASS:
            continue
        if len(lasts) == 1:
            v = next(iter(lasts))
            if any(ends[c] != v for c, _ in pairs):
                hits.append({"kind": "hardwired_horizon", "severity": GEN23_SEVERITY,
                             "detail": f"{name}'s last point is constant at {v} across {len(pairs)} cases, but these cases' declared "
                                       f"course ends are not all equal to it {sorted({ends[c] for c, _ in pairs})}"
                                       f" -- the horizon is hard-coded"})
    return hits


def batch_gate_blockers(hits: list[dict]) -> list[dict]:
    """Batch-level hits that block emission: exactly those with `severity == "gate"`, whichever
    check produced them. Severities are limited to `BATCH_GATE_SEVERITIES`.
    """
    return [w for w in (hits or []) if w.get("severity") == "gate"]


#: Allowed batch-level severities.
BATCH_GATE_SEVERITIES = ("gate", "warn", "info")


def check_batch(built: dict) -> list[dict]:
    """GEN8: batch-level minimum diversity (target events, course lengths, symptom-day
    signatures, distractor text). built = {case_id: RawCase}.
    """
    hits: list[dict] = []
    if len(built) < 2:
        return hits

    targets = {r.prediction_context.get("target_event_type") for r in built.values()}
    if len(targets) < 2:
        hits.append({"kind": "target_event_degenerate", "severity": GEN8_SEVERITY,
                     "detail": f"all {len(built)} cases' target_event_type is {targets}"})

    # ---------- course-length diversity, on the declared course end ----------
    # The whole-stream max(ts) is constant by construction (`T + prediction_window`), so it is
    # reported only as a side note.
    declared, raw_ends = set(), set()
    for r in built.values():
        declared.add(declared_course_end(r))
        pts = [p for v in r.longitudinal_data.values() for p in v]
        if pts:
            raw_ends.add(max(int(p["ts"]) for p in pts))
    declared.discard(None)
    if len(declared) < 3:
        hits.append({"kind": "series_length_degenerate", "severity": GEN8_SEVERITY,
                     "detail": f"{len(built)} cases' declared course end takes only these "
                               f"{len(declared)} value(s): {sorted(declared)} (side-by-side diagnostic quantity: whole-stream max(ts) = "
                               f"{sorted(raw_ends)} -- the latter is guaranteed by the T+prediction_window identity, "
                               f"must not be read as course length)"})

    # GEN8c: true-symptom day sets must vary across cases; a shared set would act as a lookup
    # table across the pack. Judged on the uniqueness rate (< 0.80 fails).
    from . import wq
    sigs = []
    for cid, r in built.items():
        man = wq.injected_manifest(cid, required=False)
        real = set(man.get("real_symptom_evidence_ids") or [])
        if not real:
            continue
        days = tuple(sorted(int(e.get("source_timestamp", -1)) for e in (r.evidence_ledger or [])
                            if str(e.get("evidence_id")) in real))
        if days:
            sigs.append(days)
    if len(sigs) >= 5:
        uniq = len(set(sigs)) / len(sigs)
        if uniq < 0.8:
            top = collections.Counter(sigs).most_common(1)[0]
            hits.append({"kind": "symptom_day_signature_degenerate", "severity": GEN8_SEVERITY,
                         "detail": f"{len(sigs)} cases' true-symptom-day signatures take only {len(set(sigs))} distinct value(s)"
                                   f" (uniqueness rate {uniq:.2f} < 0.80); the most common one {top[0]} appears {top[1]} times"
                                   f" -- \"day ∈ that set => true symptom\" would become a lookup table usable across cases"})

    # ---- GEN8d: cross-case reuse of distractor text --------------
    # A distractor is any ledger item outside `real_symptom_evidence_ids`; cases whose ledger
    # is missing are not counted. A5 scans numbers only, so text reuse is checked here.
    texts: collections.Counter = collections.Counter()
    n_scanned = 0
    for cid, r in built.items():
        man = wq.injected_manifest(cid, required=False)
        real = set(man.get("real_symptom_evidence_ids") or [])
        if not real:
            continue  # no ledger => not judgeable
        seen = {str(e.get("symptom") or e.get("note") or "").strip()
                for e in (r.evidence_ledger or [])
                if str(e.get("evidence_id")) not in real}
        seen.discard("")
        if not seen:
            continue
        n_scanned += 1
        for t in seen:
            texts[t] += 1
    if n_scanned >= GEN8D_MIN_CASES and texts:
        top_text, top_n = texts.most_common(1)[0]
        share = top_n / n_scanned
        if share > GEN8D_MAX_SHARE:
            # Effective item count exp(H) (Hill number of order 1), reported beside the maximum share.
            import math as _math
            _tot = sum(texts.values())
            _H = -sum((c / _tot) * _math.log(c / _tot) for c in texts.values())
            # Name the pool responsible: the kernel's distractor pool (`core/noise.py`) is separate
            # from `registry/benign_events.yaml`.
            from .events import inherited_profile as _inh
            _owner = "the kernel's inherited pool (`core/noise.py`; `registry/benign_events.yaml` does not affect it => change it there, which moves the generation stamp)" \
                if _inh(top_text) is not None else "`registry/benign_events.yaml`"
            hits.append({"kind": "distractor_text_overused", "severity": GEN8_SEVERITY,
                         "detail": f"{n_scanned} judgeable cases, {len(texts)} distinct distractor texts"
                                   f" (effective item count {_math.exp(_H):.1f}); "
                                   f"the most common, \"{top_text}\", appears in {top_n} cases = "
                                   f"{share:.2f} > {GEN8D_MAX_SHARE} -- "
                                   f"\"text ∈ the distractor pool => ignorable\" would become a lookup table usable across batches. "
                                   f"Source: {_owner}"})
    elif n_scanned < GEN8D_MIN_CASES:
        # Too few cases: reported, not passed.
        hits.append({"kind": "distractor_text_scan_empty", "severity": "info",
                     "detail": f"{n_scanned} judgeable cases < {GEN8D_MIN_CASES} => GEN8d has zero objects on this batch"
                               f" (not a pass)"})
    return hits


# ============================================================ GEN8t: the ddx tier must not be readable off surface quantities (batch-level)
# `joint_dx` packs mix a sufficient tier and an insufficient tier (the correct answer is to
# abstain). Features with no diagnostic content (T, report counts and lengths, lab-flag
# direction and item profile) must not separate them: a hit needs
# `|AUC - 0.5| >= TIER_SURFACE_AUC_MARGIN` and a Bonferroni-corrected p < TIER_SURFACE_ALPHA.
# Abnormality magnitude is not scanned: on the sufficient tier it is the gold's evidence.
TIER_SURFACE_AUC_MARGIN = 0.15
TIER_SURFACE_ALPHA = 0.01
TIER_SURFACE_MIN_CLASS = 5
TIER_SURFACE_SEVERITY = "gate"
#: Permutations for the item-profile p: the smallest p, 1 / (n + 1), must clear the
#: corrected alpha.
TIER_SURFACE_N_PERM = 3000
_VARIANT_SUFFIX_RE = re.compile(r"v\d+$")


def _face_labs(raw) -> tuple[int, list[dict], list[dict]]:
    T = int((getattr(raw, "prediction_context", None) or {}).get("prediction_time_T") or 0)
    led = [e for e in (getattr(raw, "evidence_ledger", None) or [])
           if int(e.get("source_timestamp", 10 ** 9)) <= T]
    return T, led, [e for e in led if str(e.get("source_type", "")) == "lab_result"]


def _lab_flags(labs: list[dict]) -> list[str]:
    """`item:H` / `item:L` for every out-of-range quantitative line (item = the
    printed name, the first token of the line)."""
    out = []
    for e in labs:
        s = str(e.get("symptom", ""))
        m = _LAB_VAL_RE.search(s)
        if not m:
            continue
        try:
            v, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
        except ValueError:
            continue
        if v > hi or v < lo:
            out.append(f"{s.split(' ')[0]}:{'H' if v > hi else 'L'}")
    return out


def tier_surface_features(raw) -> dict[str, float]:
    """Solver-visible (`<= T`) quantities of one case that carry no diagnostic
    content. See the GEN8t note above for why each is here."""
    T, led, labs = _face_labs(raw)
    flags = _lab_flags(labs)
    n_abn = len(flags)
    n_low = sum(1 for x in flags if x.endswith(":L"))
    days = sorted({int(e.get("source_timestamp", 0)) for e in labs})
    reports = [str(e.get("symptom", "")) for e in led
               if str(e.get("source_type", "")) == "patient_reported_symptom"]
    f = {"prediction_time_T": float(T),
         "lab_any_abnormal": float(n_abn > 0),
         "lab_n_abnormal": float(n_abn),
         "lab_any_low": float(n_low > 0),
         "n_patient_reports": float(sum(1 for e in led if str(e.get("source_type", ""))
                                        .startswith("patient_reported"))),
         "n_ledger": float(len(led)),
         "case_id_variant_suffix": float(bool(_VARIANT_SUFFIX_RE.search(
             str(getattr(raw, "case_id", "") or "")))),
         }
    if n_abn:
        f["lab_low_share"] = n_low / n_abn
    if reports:
        f["pr_mean_len"] = sum(len(t) for t in reports) / len(reports)
    if days and T > 0:
        f["lab_first_draw_over_T"] = days[0] / T
        f["lab_T_minus_last_draw"] = float(T - days[-1])
    return f


def _item_profile_scores(bags: list[set], y: list[bool]) -> list[float]:
    """Leave-one-out Bernoulli naive-Bayes log-odds of `y` (Laplace-smoothed) from
    each case's set of `item:direction` flags."""
    feats = sorted(set().union(*bags)) if bags else []
    n = {True: sum(y), False: len(y) - sum(y)}
    cnt = {True: collections.Counter(), False: collections.Counter()}
    for b, t in zip(bags, y):
        cnt[t].update(b)
    out = []
    for b, t in zip(bags, y):
        nn = {k: n[k] - (k == t) for k in (True, False)}
        s = math.log((nn[True] + 1) / (nn[False] + 1))
        for x in feats:
            c1 = cnt[True][x] - (t and x in b)
            c0 = cnt[False][x] - ((not t) and x in b)
            p1, p0 = (c1 + 1) / (nn[True] + 2), (c0 + 1) / (nn[False] + 2)
            s += math.log(p1 / p0) if x in b else math.log((1 - p1) / (1 - p0))
        out.append(s)
    return out


def _auc_of(scores: list[float], y: list[bool]) -> float:
    return _mann_whitney([s for s, t in zip(scores, y) if t],
                         [s for s, t in zip(scores, y) if not t])[0]


def _mann_whitney(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """(AUC = P(pos > neg) + P(tie)/2, two-sided p by the tie-corrected normal
    approximation)."""
    n1, n2 = len(pos), len(neg)
    allv = sorted((v, i) for i, v in enumerate(list(pos) + list(neg)))
    ranks = [0.0] * (n1 + n2)
    ties = 0.0
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[allv[k][1]] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    u = sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    auc = u / (n1 * n2)
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return auc, 1.0
    z = (u - n1 * n2 / 2.0) / math.sqrt(var)
    return auc, math.erfc(abs(z) / math.sqrt(2.0))


def check_tier_surface(built: dict) -> list[dict]:
    """GEN8t. built = {case_id: RawCase}. Reports every tier-neutral surface
    feature that separates the sufficient and insufficient ddx tiers."""
    tiers: dict[bool, list[dict]] = {True: [], False: []}
    bags: list[set] = []
    suff: list[bool] = []
    for r in built.values():
        ddx = (getattr(r, "adjudication", None) or {}).get("ddx")
        if not ddx:
            continue
        tiers[bool(ddx.get("insufficient"))].append(tier_surface_features(r))
        bags.append(set(_lab_flags(_face_labs(r)[2])))
        suff.append(not ddx.get("insufficient"))
    n_ins, n_suf = len(tiers[True]), len(tiers[False])
    if not n_ins or not n_suf:
        return []                     # a one-tier pack has no tier to read off
    if min(n_ins, n_suf) < TIER_SURFACE_MIN_CLASS:
        return [{"kind": "tier_surface_unscannable", "severity": "info",
                 "detail": f"sufficient tier {n_suf} cases · insufficient tier {n_ins} cases, smaller side < "
                           f"{TIER_SURFACE_MIN_CLASS} => GEN8t not scanned on this batch (not a pass)"}]
    names = sorted(set().union(*tiers[True], *tiers[False]))
    tested: list[tuple[str, float, float]] = []
    for fn in names:
        pos = [f[fn] for f in tiers[False] if fn in f]
        neg = [f[fn] for f in tiers[True] if fn in f]
        if min(len(pos), len(neg)) < TIER_SURFACE_MIN_CLASS or len(set(pos + neg)) < 2:
            continue
        auc, p = _mann_whitney(pos, neg)
        tested.append((fn, auc, p))
    # Item identity: LOO naive Bayes over `item:direction` flags, p by permutation.
    if any(bags):
        import random as _random
        auc_nb = _auc_of(_item_profile_scores(bags, suff), suff)
        k = len(tested) + 1
        # Stop once p can no longer clear the corrected alpha.
        give_up = TIER_SURFACE_ALPHA / k * (TIER_SURFACE_N_PERM + 1)
        rr, yp, ge, done = _random.Random(0), list(suff), 0, 0
        for done in range(1, TIER_SURFACE_N_PERM + 1):
            rr.shuffle(yp)
            # One-sided: leave-one-out fits under shuffled labels are biased below 0.5.
            ge += _auc_of(_item_profile_scores(bags, yp), yp) >= auc_nb
            if ge + 1 >= give_up:
                break
        p_nb = (ge + 1) / (done + 1)
        tested.append(("lab_item_profile_nb", auc_nb, p_nb))
    hits: list[dict] = []
    for fn, auc, p in tested:
        p_adj = min(1.0, p * len(tested))
        if abs(auc - 0.5) >= TIER_SURFACE_AUC_MARGIN and p_adj < TIER_SURFACE_ALPHA:
            hits.append({"kind": "tier_surface_separable", "severity": TIER_SURFACE_SEVERITY,
                         "feature": fn, "auc": round(auc, 3), "p_adj": p_adj,
                         "detail": f"`{fn}` alone separates the diagnostic tier: AUC(sufficient>insufficient) = {auc:.3f}"
                                   f" (sufficient {n_suf} · insufficient {n_ins}, Bonferroni p = {p_adj:.2g})"
                                   f" -- this quantity carries no diagnostic content; the tier must not be readable from it"})
    return hits


# ============================================================ the reason ledger for unemitted cases
def blocked_reasons(audits) -> dict[str, list[str]]:
    """Unemitted cases -> why each was blocked.

    Pre-generation reasons first (`latent_blocking`, `declaration_blocking`), then later gates
    in reverse pipeline order (the later gate is more specific): post_noise_conflicts,
    verify_bad, last_conflicts, premise_error, gen_error. A case with no recorded reason says
    so explicitly.
    """
    out: dict[str, list[str]] = {}
    for a in (audits or []):
        if not isinstance(a, dict) or a.get("emitted"):
            continue
        why = ([f"latent:{k}" for k in (a.get("latent_blocking") or [])]
               or [f"declaration:{k}" for k in (a.get("declaration_blocking") or [])]
               or list(a.get("post_noise_conflicts") or [])
               or [f"verify:{k}" for k in (a.get("verify_bad") or [])]
               or [f"gv1:{k}" for k in (a.get("last_conflicts") or [])]
               or ([f"premise:{str(a['premise_error'])[:60]}"] if a.get("premise_error") else [])
               or ([f"gen_error:{str(a['gen_error'])[:60]}"] if a.get("gen_error") else []))
        out[str(a.get("case_id", "?"))] = why or ["(no reason recorded -- this is itself a defect)"]
    return out


def dropped_items(audits) -> dict[str, list[str]]:
    """Emitted cases -> items dropped from them in the injection-retry loop.

    A class of item that can never pass is dropped from every case while verification still
    reports 0 failures, so the drops are persisted. Unemitted cases are `blocked_reasons`'.
    """
    out: dict[str, list[str]] = {}
    for a in audits or ():
        if not a.get("emitted"):
            continue
        d = sorted(a.get("event_dropped") or ())
        if d:
            out[str(a.get("case_id"))] = d
    return out


def dropped_rates(audits) -> dict[str, dict]:
    """Drop rate per item, `{item: {"n", "of", "rate"}}`; `rate=1.00` marks a structural drop."""
    emitted = [a for a in (audits or ()) if a.get("emitted")]
    n_em = len(emitted)
    cnt: dict[str, int] = {}
    for a in emitted:
        for k in set(a.get("event_dropped") or ()):
            cnt[k] = cnt.get(k, 0) + 1
    return {k: {"n": v, "of": n_em, "rate": round(v / n_em, 4) if n_em else None}
            for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}


def blocked_counts(audits) -> dict[str, int]:
    """Counts of unemitted reasons by kind (separate from warn-level per-case counts)."""
    c: dict[str, int] = {}
    for ks in blocked_reasons(audits).values():
        for k in ks:
            c[k] = c.get(k, 0) + 1
    return c


# ---------------------------------------------------------------- two gates on the findings layer
# A4 (discriminability) and N1 (single-feature unsolvability) pull in opposite directions:
# A4 alone allows lookup questions, N1 alone allows unsolvable ones.

FINDINGS_A4_SEVERITY = "warn"  # legitimate questions hit it (see docstring)
FINDINGS_N1_SEVERITY = "warn"   # legitimate questions hit it


def check_rival_discriminable(spec_id: str, rivals, profile) -> list[dict]:
    """A4: every near-miss rival has a discriminating finding in the gold's findings profile.

    Warn, not gate: some rivals are not yet labeled, and rivals ruled out by routine workup
    alone (`no_order_needed`) are a difficulty issue.
    """
    hits: list[dict] = []
    prof = {p["id"]: p for p in (profile or {}).get("findings") or ()}
    if not prof:
        return [{"kind": "findings_profile_missing", "spec_id": spec_id,
                 "detail": "this condition has no findings profile -- unjudgeable, not a pass"}]
    n_order = 0
    for r in (rivals or ()):
        df, un = r.get("discriminator_finding"), r.get("discriminator_unresolvable")
        if un:
            hits.append({"kind": "rival_unresolvable", "spec_id": spec_id,
                         "rival": r.get("name"), "detail": str(un)[:80]})
            continue
        if not df:
            hits.append({"kind": "rival_discriminator_unlabeled", "spec_id": spec_id,
                         "rival": r.get("name")})
            continue
        p = prof.get(str(df.get("finding")))
        if p is None:
            hits.append({"kind": "rival_discriminator_not_in_profile", "spec_id": spec_id,
                         "rival": r.get("name"), "detail": str(df.get("finding"))})
        elif p["role"] != "screening":
            n_order += 1
    if (rivals or ()) and n_order == 0:
        hits.append({"kind": "no_order_needed", "spec_id": spec_id,
                     "detail": "every near-miss rival can be ruled out using routine initial-visit workup alone -- this question can complete the differential without ordering anything"})
    return hits


def check_single_finding_solvable(spec_id: str, rivals) -> list[dict]:
    """N1: no single finding may rule out every near-miss rival at once.

    A hit means the question tests which item to order, not weighing evidence. The fix is a
    rival that shares the discriminating item, not a fabricated discriminator.
    """
    ds = [r.get("discriminator_finding") for r in (rivals or ())]
    ds = [d for d in ds if d]
    if len(ds) < 2 or len(ds) != len(rivals or ()):
        return []  # unlabeled entries: not judgeable
    fids = {str(d.get("finding")) for d in ds}
    if len(fids) == 1:
        return [{"kind": "single_finding_solvable", "spec_id": spec_id,
                 "detail": f"all {len(ds)} near-miss rivals are ruled out by {fids.pop()} -- checking one item gives the answer"}]
    return []
