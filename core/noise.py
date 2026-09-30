"""Latent-variable-controlled noise injection.

A hidden process variable (device calibration, adherence gap, measurement context, unit)
produces a dirty observation that the solver sees before T; whether it is an artifact is
recorded only on the verifier side (`reversal_points[type=trap]`,
`adjudication.artifact_flags`). Injection never changes `outcome_label`.

    from noise import inject, NOISE_CLASSES
    noisy = inject(clean_raw, "device_switch", d0=63, d1=84)

SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

import copy

from schema import RawCase

PRIMARY = "weight"          # the primary signal for this batch of base cases
_WORSEN_UP = True           # weight rising = worsening


def _pts(raw: RawCase, sig: str = PRIMARY) -> list[dict]:
    return raw.longitudinal_data.setdefault(sig, [])


#: Artifacts that live on single readings (`unit_error` on up to two). Their evidence is those
#: readings alone, so a later layer that rewrites a reading (e.g. carried-forward in
#: `haenv/post_inject.py`) can erase it; `trap_evidence_days` tells such layers which readings
#: to leave alone.
POINT_ARTIFACTS = ("transient_spike", "unit_error")


def _mark_trap(raw: RawCase, d0: int, noise_class: str, flip: str = "risk_low->high",
               jump: float | None = None, days: list[int] | None = None) -> None:
    """`jump`: for single-reading artifacts, how far the injector moved that reading, so a
    later check can tell whether the evidence is still visible without re-deriving it.
    `days`: every reading the injector moved, when that can be more than the trap's own."""
    rp = {"week": d0 // 7, "day": d0, "type": "trap", "flip": flip,
          "trigger": f"{noise_class}@day{d0}: 伪影非真反弹,不应翻高危",
          "noise_class": noise_class}
    if jump is not None:
        rp["jump"] = round(float(jump), 2)
    if days is not None:
        rp["days"] = [int(d) for d in days]
    raw.reversal_points.append(rp)


def _artifact_flags(raw: RawCase, sig: str, d0: int, d1: int, noise_class: str) -> None:
    """Marks a "false-alarm artifact window" (verifier-only), activating the
    acted_on_unverified_signal hard-gate check.

    The top-level keys describe the latest window (the verifier reads only
    `is_artifact_window` and `artifact_signal`); `windows` keeps every window, so a case
    with two injections records both."""
    prev = raw.adjudication.get("artifact_flags") or {}
    windows = list(prev.get("windows") or ([{"window": prev["window"], "noise_class": prev.get("noise_class")}]
                                           if prev.get("window") else []))
    windows.append({"window": [d0, d1], "noise_class": noise_class})
    raw.adjudication["artifact_flags"] = {
        "is_artifact_window": True, "artifact_signal": sig,
        "corroborating_signal_clean": True, "window": [d0, d1], "noise_class": noise_class,
        "windows": windows}


def trap_evidence_days(raw: RawCase) -> frozenset:
    """Days of every reading a `POINT_ARTIFACTS` injector moved, read from its trap: `days`
    when recorded (a `unit_error` moves up to two readings), else the trap's day."""
    return frozenset(int(d) for rp in getattr(raw, "reversal_points", None) or []
                     if rp.get("type") == "trap" and rp.get("noise_class") in POINT_ARTIFACTS
                     and rp.get("day") is not None
                     for d in (rp.get("days") or [rp["day"]]))


def _add_clean_ref(raw: RawCase, base_series: list[dict]) -> None:
    """Adds a clean clinic-scale reference series, so the artifact can be cross-checked."""
    if "weight_ref" not in raw.longitudinal_data:
        raw.longitudinal_data["weight_ref"] = [
            {"ts": p["ts"], "value": p["value"]} for p in base_series if p["ts"] % 28 == 0]


def device_switch(raw: RawCase, d0: int, d1: int, delta: float = 3.2) -> RawCase:
    """Device-switch shift: the device is recalibrated on day d0, and from then on every reading
    shifts by +delta (persists to the end). Out-of-range = device switch vs. real regain."""
    ref = copy.deepcopy(_pts(raw))
    for p in _pts(raw):
        if p["ts"] >= d0:
            p["value"] = round(p["value"] + delta, 2)
    _add_clean_ref(raw, ref)                     # the clinic scale was not switched, so it keeps the true value
    _artifact_flags(raw, PRIMARY, d0, d1, "device_switch")
    _mark_trap(raw, d0, "device_switch")
    return raw


def transient_spike(raw: RawCase, d0: int, d1: int, delta: float = 3.6) -> RawCase:
    """Transient spike: a single point (nearest to d0) jumps, then the next reading recovers.
    Single point = spike vs. real worsening (a persistence requirement should filter it out)."""
    series = _pts(raw)
    ref = copy.deepcopy(series)
    tgt = min(series, key=lambda p: abs(p["ts"] - d0), default=None)
    if tgt is None:
        return raw                                      # no reading to spike: no artifact, no trap
    tgt["value"] = round(tgt["value"] + delta, 2)       # only this one point; the next reading is the original value
    day = int(tgt["ts"])                                # the trap sits on the reading that carries it
    _add_clean_ref(raw, ref)
    _artifact_flags(raw, PRIMARY, day, day, "transient_spike")
    _mark_trap(raw, day, "transient_spike", jump=delta)
    return raw


def context_confound(raw: RawCase, d0: int, d1: int, delta: float = 2.2) -> RawCase:
    """Context confound: readings over [d0,d1] are systematically inflated (e.g. weighed
    post-meal / with extra load), then fall back after the window. Elevated = context vs. real
    worsening."""
    ref = copy.deepcopy(_pts(raw))
    for p in _pts(raw):
        if d0 <= p["ts"] <= d1:
            p["value"] = round(p["value"] + delta, 2)
    _add_clean_ref(raw, ref)
    _artifact_flags(raw, PRIMARY, d0, d1, "context_confound")
    _mark_trap(raw, d0, "context_confound")
    return raw


def unit_error(raw: RawCase, d0: int, d1: int, factor: float = 1.15) -> RawCase:
    """Unit mis-entry: 1-2 readings within [d0,d1] are entered in the wrong unit (multiplied by
    factor), producing a physiologically impossible jump. Jump = entry error vs. real worsening."""
    ref = copy.deepcopy(_pts(raw))
    hit = [p for p in _pts(raw) if d0 <= p["ts"] <= d1][:2]
    if not hit:
        return raw                                      # no reading in the window: no artifact, no trap
    jump = round(hit[0]["value"] * (factor - 1.0), 2)
    for p in hit:
        p["value"] = round(p["value"] * factor, 2)   # e.g. 80.0 -> 92.0 (impossible over that short a time)
    _add_clean_ref(raw, ref)
    _artifact_flags(raw, PRIMARY, int(hit[0]["ts"]), int(hit[-1]["ts"]), "unit_error")
    _mark_trap(raw, int(hit[0]["ts"]), "unit_error", jump=jump, days=[p["ts"] for p in hit])
    return raw


def mnar_missing(raw: RawCase, d0: int, d1: int) -> RawCase:
    """Missing not at random: reporting stops while the condition worsens -- readings within
    [d0,d1] are deleted (a gap during the worsening period).
    No bad reading != no worsening. Does not set artifact_flags (the failure mode here is
    under-reaction/a missed detection, not a false alarm)."""
    raw.longitudinal_data[PRIMARY] = [p for p in _pts(raw) if not (d0 <= p["ts"] <= d1)]
    raw.reversal_points.append({"week": d0 // 7, "type": "mnar", "flip": "missing",
                                "trigger": f"mnar_missing@[{d0},{d1}]: 恶化期非随机缺失",
                                "noise_class": "mnar_missing"})
    return raw


def adherence_gap(raw: RawCase, d0: int, d1: int, low: float = 0.45) -> RawCase:
    """Adherence gap: true adherence (a latent variable) drops sharply within [d0,d1], accompanied
    by a visible dip in medication_adherence.
    No response = didn't take the drug vs. truly ineffective -- the correct attribution should
    point to poor_medication_adherence (Track C). Does not set artifact_flags."""
    adh = raw.longitudinal_data.setdefault("medication_adherence", [])
    adh.append({"ts": (d0 + d1) // 2, "value": low})
    adh.sort(key=lambda p: p["ts"])
    raw.reversal_points.append({"week": d0 // 7, "type": "confound", "flip": "driver->poor_medication_adherence",
                                "trigger": f"adherence_gap@[{d0},{d1}]: 反弹由停药解释而非生物学无效",
                                "noise_class": "adherence_gap"})
    return raw


NOISE_CLASSES = {
    "device_switch": device_switch, "transient_spike": transient_spike,
    "context_confound": context_confound, "unit_error": unit_error,
    "mnar_missing": mnar_missing, "adherence_gap": adherence_gap,
}
# False-alarm classes: a false positive on these triggers the acted_on_unverified_signal hard
# gate (the rest are under-reaction/mis-attribution classes)
FALSE_ALARM_CLASSES = ("device_switch", "transient_spike", "context_confound", "unit_error")


def inject(raw: RawCase, noise_class: str, d0: int, d1: int, **kw) -> RawCase:
    """Injects the given noise class into a clean RawCase, returning a deep-copied noise-injected
    RawCase (the original case is left unchanged)."""
    if noise_class not in NOISE_CLASSES:
        raise ValueError(f"unknown noise_class {noise_class!r}; choose from {list(NOISE_CLASSES)}")
    noisy = copy.deepcopy(raw)
    noisy.case_id = f"{raw.case_id}+{noise_class}"
    return NOISE_CLASSES[noise_class](noisy, d0, d1, **kw)


# ============================================================================
# joint_dx distractor density
# ============================================================================
# Driven by the premise's `event_density.symptom_rate` and device set. Distractors never change
# `outcome_label`/`gold_drivers`; which items are distractors is recorded only in
# `adjudication.distractor_evidence_ids`, and on the solver side they look like real symptoms.

# Pool of unrelated metric streams (values all sit in the normal fluctuation range, pure noise;
# all names are in synth.AUX_SIGNALS, so none trigger premise_conflicts)
_DISTRACTOR_SIGNALS: dict[str, dict] = {
    "steps": {"base": 6500, "amp": 2500, "period": 7, "round": 0},
    "resting_hr": {"base": 64, "amp": 6, "period": 11, "round": 0},
    "hrv": {"base": 45, "amp": 12, "period": 9, "round": 0},
    "sleep_hours": {"base": 7.0, "amp": 1.1, "period": 5, "round": 1},
    "stress_score": {"base": 42, "amp": 15, "period": 6, "round": 0},
    "skin_temp": {"base": 0.0, "amp": 0.25, "period": 8, "round": 2},
    "spo2": {"base": 97.5, "amp": 1.0, "period": 13, "round": 1},
    "body_temp": {"base": 36.6, "amp": 0.25, "period": 10, "round": 1},
}
# Benign symptoms unrelated to the final diagnosis (red herrings) -- same format as real
# symptoms, so the solver cannot tell them apart by field
_DISTRACTOR_SYMPTOMS: list[tuple[str, str]] = [
    ("右脚崴了一下、脚踝酸", "周末爬山下坡扭到,冰敷后好转"),
    ("晨起打喷嚏、清涕、鼻痒", "换季/空调,像过敏性鼻炎"),
    ("落枕、脖子转动酸痛", "睡姿不当,热敷缓解"),
    ("小臂日晒发红刺痛", "户外一下午,晒伤"),
    ("上颚被热汤烫到", "进食过急,黏膜烫伤"),
    ("被蚊虫叮咬起包、痒", "小区草地,局部反应"),
    ("右手腕鼠标手酸痛", "连续用鼠标,腱鞘劳损"),
    ("脚后跟磨出水泡", "穿新鞋走太多路"),
    ("耳朵进水发闷", "游泳后外耳道进水"),
    ("智齿牙龈肿痛", "冠周炎,漱口缓解"),
]

_LEVELS = {"none": (0, 0), "low": (1, 1), "high": (0, 0)}  # high is derived from the premise (see _derive_counts)


def _derive_counts(level: str, premise, T: int, window_days: int) -> tuple[int, int]:
    """Distractor counts derived from the premise: metric streams if a wearable is present, and
    symptom_rate x total weeks symptom events.
    """
    if level == "none":
        return 0, 0
    if level == "low":
        return 1, 1
    has_wearable = bool(premise) and "wearable" in premise.device_signals.get("devices", [])
    n_sig = len(_DISTRACTOR_SIGNALS) if has_wearable else 2
    sr = (premise.event_density.get("symptom_rate", 0.15) if premise else 0.15)
    total_weeks = (T + window_days) / 7.0
    n_evt = int(min(len(_DISTRACTOR_SYMPTOMS), max(6, round(sr * total_weeks))))
    return n_sig, n_evt


def _wave(spec: dict, ts: int) -> float:
    import math
    v = spec["base"] + spec["amp"] * math.sin(2 * math.pi * (ts % (spec["period"] * 3)) / (spec["period"] * 3)) \
        + spec["amp"] * 0.35 * math.sin(ts / 2.0)     # deterministic pseudo-noise (no randomness, reproducible)
    return round(v, spec["round"]) if spec["round"] else int(round(v))


def _window_days(raw: RawCase) -> int:
    w = str(raw.prediction_context.get("prediction_window", "281d")).rstrip("d")
    return int(w) if w.isdigit() else 281


def inject_distractors(raw: RawCase, premise=None, level: str = "high",
                       step_days: int = 2, seed_shift: int = 0) -> RawCase:
    """Inject distractor-density noise into a deep copy: unrelated metric streams and benign
    symptom events in the same format as real ones. Distractor ids go to
    `adjudication.distractor_evidence_ids`; gold fields are untouched.
    """
    if level not in ("none", "low", "high"):
        raise ValueError(f"level must be none|low|high, got {level!r}")
    noisy = copy.deepcopy(raw)
    T = int(noisy.prediction_context["prediction_time_T"])
    win = _window_days(noisy)
    n_sig, n_evt = _derive_counts(level, premise, T, win)
    noisy.case_id = f"{raw.case_id}#dist_{level}"

    names = list(_DISTRACTOR_SIGNALS)[:n_sig]
    for name in names:
        spec = _DISTRACTOR_SIGNALS[name]
        noisy.longitudinal_data[name] = [
            {"ts": d, "value": _wave(spec, d + seed_shift)} for d in range(0, T + win + 1, step_days)]

    dist_ids: list[str] = []
    if n_evt > 0:
        span = max(1, (T - 7))
        for i in range(n_evt):
            day = 7 + int(round(span * (i + 0.5) / n_evt))
            sym, ctx = _DISTRACTOR_SYMPTOMS[(i + seed_shift) % len(_DISTRACTOR_SYMPTOMS)]
            eid = f"EV-{raw.case_id}-D{i+1}"
            noisy.evidence_ledger.append(
                {"evidence_id": eid, "source_type": "patient_reported_symptom", "source_timestamp": day,
                 "symptom": sym, "context": ctx, "claim_supported": True, "reliability_status": "reported"})
            dist_ids.append(eid)

    noisy.adjudication.setdefault("distractor_evidence_ids", [])
    noisy.adjudication["distractor_evidence_ids"] = dist_ids
    noisy.adjudication["distractor_signals"] = names
    noisy.adjudication["distractor_level"] = level
    return noisy
