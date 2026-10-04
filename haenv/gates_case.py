"""Per-case world-consistency gates: sampling cadence, devices, the premise registry, clinical
coupling, event density, dosing, missingness, anchors, demographics, symptom days, stream
horizons and ledger values. Each takes one built case and returns findings.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import collections
import logging
import math
from haenv import data_root as _dr  # resource root: repo root in a source tree, haenv/_data in a wheel
from haenv.streams import device_signals
from .gate_tables import (  # noqa: F401
    DRUG_DOSES_PER_WEEK,
    SEX_EXCLUSIVE,
)
from .gate_tables import (  # noqa: F401
    LEDGER_VALUE_STREAMS,
)


def _clinical_cv_of(sig: str) -> float:
    """Per-measurement CV of a clinical stream (via `build._clinical_cv()`); 0.0 when unregistered."""
    try:
        from .build_clinical import _clinical_cv
        return float(_clinical_cv().get(sig, 0.0) or 0.0)
    except Exception:                                      # noqa: BLE001
        return 0.0


def clinical_coupling_judgeable() -> dict:
    """`{signal: bool}` from `registry/physio_streams.yaml`: False declares that GEN15 cannot judge
    direction on that signal (effect below measurement noise). Unregistered signals are judged.
    """
    try:
        from .yamlcache import load_yaml as _cached_yaml
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


def _lab_calendar_signals() -> frozenset:
    """`registry/physio_streams.yaml:clinical_observation.lab_signals`."""
    from .regpath import load_registry
    cfg = (load_registry("physio_streams.yaml") or {}).get("clinical_observation") or {}
    return frozenset(str(x) for x in (cfg.get("lab_signals") or ()))


def check_cadence(longitudinal_data: dict, premise, cs) -> list[dict]:
    """GEN6: actual sampling must match the premise's declaration.

    Covers non-auxiliary signals. With declared missingness: the modal step must equal
    `sampling_days`, the missing rate must be within `GEN6_MISSING_TOL` of
    `expected_missing_rate`, and no gap may exceed `max_gap_days`. Without it: every step must
    equal `sampling_days`. Gaps inside noise windows and a trailing partial step are exempt.
    """
    from haenv_kernel.synth import AUX_SIGNALS          # kernel

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
        if sig in _lab_calendar_signals():
            # Lab draws follow the case's visit calendar (`build.lab_draw_days` /
            # `build.observe_labs`), not the premise's nominal `sampling_days` grid:
            # judged as a first draw on day 0 and days in order (same-day repeats allowed).
            ts_ = [int(q["ts"]) for q in pts]
            if ts_ and (ts_[0] != 0 or ts_ != sorted(ts_)):
                hits.append({"kind": "sampling_cadence_mismatch", "severity": GEN6_SEVERITY,
                             "detail": f"{sig}: lab calendar must start on day 0 and be ordered · {ts_[:6]}"})
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
        from .build_clinical import CLINICAL_ATTEN
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


# GEN20 (gate): injection density, dosing frequency and missingness shape must match the premise.
GEN20_SEVERITY = "gate"


#: Minimum grid points on each side of the reversal point to compare missing rates.
#: At 8 points the missing rate's resolution is 12.5%; coarser cannot show a difference.
#: Empirical: it should sit below the grid sizes of the cases that can be judged.
MISSINGNESS_MIN_GRID = 8


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
    from .events_pools import EVENT_RATE_DEFAULTS, event_weeks, expected_event_counts
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
    from .yamlcache import load_yaml as _cached_yaml
    return _cached_yaml(_REG_DIR / name) or {}


def check_rhythm_gap_feasible(cs) -> list[dict]:
    """GEN29: a declared `rhythm_gap` must fit the visible window (`rhythm_gap_feasible(T)`),
    rather than being dropped silently at run time.
    """
    from .rhythm import REAL_RHYTHM_GAP_DAYS, REAL_RHYTHM_SLICES, rhythm_gap_feasible

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


A5_MIN_CLASS = 2  # min class size


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
    from haenv_kernel.latent import DISEASE_SIGNAL_DOMAIN
    from haenv_kernel.synth import AUX_SIGNALS

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
    from .build_clinical import _clinical_baselines

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
    from haenv_kernel.noise import POINT_ARTIFACTS, PRIMARY, trap_evidence_days  # kernel
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
