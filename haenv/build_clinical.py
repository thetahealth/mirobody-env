"""Clinical signal streams (GEN7): which labs follow which drivers, their baselines and
variability, drug responses, and rendering of the clinical series.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from .regpath import registry_cached as _registry_cached
import functools
import math
import re
import statistics
from . import indicators as _indicators
from . import artifact_params as _artifact_params
from . import drug_effects as _drug_effects
from . import med_course as _med_course
from .job import raw_field as _raw_field
from . import demographics as _demographics
from . import events as events_mod
from .world_knobs import END
from . import gate_tables as gates   # `gates.LEDGER_VALUE_STREAMS` lives in the table module


CLINICAL_BY_DEVICE: dict[str, set[str]] = {
    "lab_panel": {"HbA1c", "LDL", "ALT", "AST", "FIB4", "triglycerides", "fasting_glucose"},
    "cgm": {"CGM_TIR", "fasting_glucose"},
    "bp_cuff": {"systolic_bp", "diastolic_bp"},
}


# Per signal: (`per_kg`, `step`). Rendered as
# base + per_kg x ATTEN x (w(t-lag) - w0), so `per_kg` is the change per 1 kg
# gained (GEN15 checks its sign). Baselines come per cohort from
# `registry/indicators.yaml` and `ndigits` from the indicator dossier. `FIB4`
# is an independent stream, not computed from ALT/AST.
CLINICAL_SPEC: dict[str, tuple[float, int]] = {
    #                 per_kg   step
    "HbA1c":           (0.06,   90),
    "fasting_glucose": (0.05,   90),
    "CGM_TIR":         (-1.20,   7),    # weight gain -> time-in-range drops
    "LDL":             (0.010,  90),
    "triglycerides":   (0.020,  90),
    "ALT":             (0.80,   90),
    "AST":             (0.50,   90),
    "FIB4":            (0.012,  90),
    # BP coefficients per kg: Neter JE et al., Hypertension 2003;42(5):878-84
    # (PMID 12975389): -1.05 mmHg systolic, -0.92 mmHg diastolic per kg lost.
    "systolic_bp":     (1.05,    7),
    "diastolic_bp":    (0.92,    7),
}


CLINICAL_LAG_DAYS = 30          # lab values reflect roughly a month-old state


#: Premise-side `ndigits` of `clinical_plan` for indicators whose printed precision
#: (`indicators.yaml`) changed after the world prompts were cached: the premise is part
#: of the world prompt, so it keeps the cached value. Rendering reads `indicators.yaml`.
_PROMPT_PLAN_NDIGITS: dict[str, int] = {"fasting_glucose": 1}


CLINICAL_ATTEN = 0.75           # lab values only partly track weight


@_registry_cached("clinical_baselines.yaml")
def _clinical_baselines() -> dict:
    """`registry/clinical_baselines.yaml` -- cohort tiers for clinical baselines."""
    from .regpath import load_registry as _lr
    return _lr("clinical_baselines.yaml") or {}


def clinical_cohort(sig: str, disease: str, comorbidities) -> str | None:
    """Which cohort this case belongs to on `sig` (via the registry's
    `diabetes_markers`, matched against disease and comorbidities); `None` if the
    signal has no cohort tiers.
    """
    reg = _clinical_baselines()
    entry = ((reg.get("signals") or {}).get(sig) or {})
    if not entry.get("cohorts"):
        return None
    hay = " ".join([str(disease or "")] + [str(c) for c in (comorbidities or [])]).lower()
    if any(str(m).lower() in hay for m in (reg.get("diabetes_markers") or [])):
        return "diabetic"
    return str(entry.get("default_cohort") or "non_diabetic")


def clinical_plan(disease: str, devices: list[str],
                  comorbidities=(), case_id: str = "") -> dict[str, dict]:
    """(disease + comorbidity domains ∩ what the devices can produce) - weight -> the
    clinical signals this case generates, with their parameters. Baselines are
    sampled per cohort.
    """
    from haenv_kernel.latent import COMORBIDITY_SIGNAL_DOMAIN, DISEASE_SIGNAL_DOMAIN
    # The domain includes comorbidities; a signal still needs a device that
    # produces it (`CLINICAL_BY_DEVICE`).
    domain = dict(DISEASE_SIGNAL_DOMAIN.get(disease, {}))
    for _c in (comorbidities or ()):
        domain.update(DISEASE_SIGNAL_DOMAIN.get(str(_c), {}) or COMORBIDITY_SIGNAL_DOMAIN.get(str(_c), {}))
    producible: set[str] = set()
    for dev in devices or []:
        producible |= CLINICAL_BY_DEVICE.get(dev, set())
    _sigs = [s for s in sorted(set(domain) & producible)
             if s != "weight" and s in CLINICAL_SPEC]
    # Sampled as one group, so an umbrella diagnosis (e.g. dyslipidemia: LDL or TG)
    # shows up in at least one signal.
    _bases = _indicators.sample_case(disease, _sigs, comorbidities, case_id=str(case_id))
    out: dict[str, dict] = {}
    for sig in _sigs:
        per_kg, step = CLINICAL_SPEC[sig]
        nd = _PROMPT_PLAN_NDIGITS.get(sig, _indicators.of(sig)["ndigits"])
        base = _bases[sig]
        out[sig] = {"base": base, "per_kg": per_kg, "step": step, "ndigits": nd,
                    # drug effects apply only to the cohort they were measured on
                    "cohort": _indicators.cohort_of(sig, disease, comorbidities),
                    "unit": domain[sig]["unit"], "range": list(domain[sig]["range"]),
                    "max_weekly_delta": domain[sig]["max_weekly_delta"]}
    return out


def _drug_of(p) -> str:
    return str((p.patient_basics.get("regimen") or {}).get("drug") or "")


def _drug_response_of(p) -> float:
    """This case's drug-response multiplier (`drug_effects.response_for`); shared by
    rendering and GEN15.
    """
    m = p.meta or {}
    return _drug_effects.response_for(str(m.get("case_id") or "HAENV"), m.get("driver"),
                                      _drug_of(p), m.get("drug_response"))


def _drug_terms_for(raw, p, cplan: dict) -> dict[str, float]:
    """Net drug-effect increment per clinical signal over the observation window,
    for GEN15's expected direction; same `effect_at` as `render_clinical`.
    """
    ld = raw.longitudinal_data or {}
    m = p.meta or {}
    T = int(m.get("prediction_time_T", 84))
    _cid = str(m.get("case_id") or "HAENV")
    ce = int(m.get("course_end_day", END))
    dose = ld.get("dose_timeline") or []
    concurrent = _world_medication(p, _cid, T, ce)[2]
    adh = _true_adherence_pts(p, T)
    out: dict[str, float] = {}
    for sig, spec in (cplan or {}).items():
        pts = ld.get(sig) or []
        if len(pts) < 2:
            continue
        d0, d1 = int(pts[0]["ts"]), int(pts[-1]["ts"])
        term = _drug_term_fn(p, sig, spec, dose, adh, concurrent)
        out[sig] = term(d1) - term(d0)
    return out


def _last_dose_mg(dose_pts) -> float | None:
    """The dose (mg) at the last point of the dose trajectory (current dose, not the
    maximum); `None` if empty.
    """
    pts = [q for q in (dose_pts or [])
           if isinstance(q, dict) and q.get("value") is not None]
    return float(pts[-1]["value"]) if pts else None


@functools.lru_cache(maxsize=1)
def _assert_kernel_params_in_sync() -> bool:
    """Checks the artifact-amplitude registry against the kernel's default
    arguments, once per process on the generation path.
    """
    from haenv_kernel.noise import NOISE_CLASSES
    _artifact_params.assert_matches_kernel(NOISE_CLASSES)
    return True


def world_layer_gaps() -> dict[str, object]:
    """The world layer's known gaps (pending cohorts, uncalibrated parameters,
    drugs without effect data), recorded in each case's `audit`.
    """
    return {
        "pending_cohorts": _indicators.pending_cohorts(),
        "known_defects": sorted(_indicators.known_defects()),
        "median_only_cohorts": {k: v for k, v in _indicators.distribution_status().items()
                                if not str(v).startswith("分布")},
        "uncalibrated_artifact_params": _artifact_params.missing_calibration(),
        # drug pool read live, so a newly added drug is covered
        "drugs_without_effect_data": _drug_effects.missing_drugs(
            {d for dis, _w in _demographics.disease_pool()
             for d, _s in _demographics.drugs_for(dis)}),
        "drug_effect_cohorts": sorted(
            {c for c in (_drug_effects._doc().get("applies_to_cohorts_default") or [])}),
    }


@_registry_cached("physio_streams.yaml")
def _clinical_cv() -> dict:
    """`registry/physio_streams.yaml:clinical_measurement.cv` -- within-person
    measurement-to-measurement CV per indicator. `null` entries add no noise.
    """
    from .regpath import load_registry as _lr
    doc = _lr("physio_streams.yaml") or {}
    raw = ((doc.get("clinical_measurement") or {}).get("cv") or {})
    return {k: float(v) for k, v in raw.items() if v is not None}


# --------------------------------------------------------------- Lab observation process
@_registry_cached("physio_streams.yaml")
def _lab_obs_cfg() -> dict:
    """`registry/physio_streams.yaml:clinical_observation` -- the lab visit calendar and
    the measurement layer of the lab-calendar streams."""
    from .regpath import load_registry as _lr
    doc = _lr("physio_streams.yaml") or {}
    cfg = doc.get("clinical_observation")
    if not isinstance(cfg, dict):
        raise ValueError("registry/physio_streams.yaml: clinical_observation is missing")
    return cfg


def lab_signals() -> frozenset:
    """Streams drawn on the lab calendar."""
    return frozenset(str(x) for x in (_lab_obs_cfg().get("lab_signals") or ()))


def _lab_group_of(sig: str) -> str:
    for g, members in (_lab_obs_cfg().get("groups") or {}).items():
        if sig in (members or ()):
            return str(g)
    return str(sig)


def _quantize_within_slope(prev: dict | None, v: float, day: int,
                           mwd: float, ndigits: int) -> float:
    """Rounds to `ndigits` while keeping the weekly-slope limit.

    When the allowed change over the interval is smaller than one quantization
    step (e.g. HbA1c 3 days apart), the only legal value is `prev`. `prev is
    None` (first point) => just rounded.
    """
    step = 10.0 ** (-int(ndigits)) if ndigits else 1.0
    q = round(v, int(ndigits)) if ndigits else float(round(v))
    if prev is None:
        return q
    gap = max(1e-9, (int(day) - int(prev["ts"])) / 7.0)
    n_step = math.floor(float(mwd) * gap / step + 1e-9)
    p = float(prev["value"])
    q = min(max(q, p - n_step * step), p + n_step * step)
    return round(q, int(ndigits)) if ndigits else float(round(q))


#: Precision of the noise-free lab truth (the observation rounds to the printed `ndigits`).
TRUTH_NDIGITS = 4


def render_clinical(sig: str, spec: dict, weight_pts: list[dict], end_day: int,
                    drug: str = "", dose_mg: float | None = None,
                    adherence_pts: list[dict] | None = None,
                    case_id: str = "C", response: float = 1.0,
                    days: list[int] | None = None, drug_term=None) -> list[dict]:
    """Derives one clinical signal from the weight trajectory: lag + attenuation +
    drug effect + value-range clamp + weekly-slope clamp.

    `drug_term(day)`, when given, is the drug-effect increment read along the dose line
    (`_drug_term_fn`) and replaces the single-dose `effect_at` term.

        v = base + per_kg×ATTEN×Δw  +  effect_at(drug, sig, day, adherence(day))

    `effect_at` excludes the weight-mediated part, so it is not counted twice.
    `weight` itself never gets a drug term (`drug_effects.FORBIDDEN`).

    With `days` (a lab calendar, `lab_item_days`) the stream is the noise-free truth on
    those days at `TRUTH_NDIGITS`; `observe_labs` adds the measurement layer. Without it,
    the stream is on its `step` grid with per-draw variation.
    """
    if not weight_pts:
        return []
    w0 = float(weight_pts[0]["value"])

    def adh_at(day: int) -> float:
        """Adherence on this day; 1.0 when unknown (trial effect sizes assume full
        adherence).
        """
        pts = [q for q in (adherence_pts or [])
               if isinstance(q, dict) and int(q.get("ts", -1)) <= day]
        return float(pts[-1]["value"]) if pts else 1.0

    def w_at(day: int) -> float:
        d = max(0, day - CLINICAL_LAG_DAYS)
        prev = [p for p in weight_pts if int(p["ts"]) <= d]
        return float((prev[-1] if prev else weight_pts[0])["value"])

    lo, hi = spec["range"]
    mwd = float(spec["max_weekly_delta"])
    if days is not None:
        pts = []
        for day in sorted({int(d) for d in days if 0 <= int(d) <= int(end_day)}):
            v = spec["base"] + spec["per_kg"] * CLINICAL_ATTEN * (w_at(day) - w0)
            v += (drug_term(day) if drug_term is not None else
                  _drug_effects.effect_at(drug, sig, day, adh_at,
                                          per_kg=spec["per_kg"], atten=CLINICAL_ATTEN,
                                          dose_mg=dose_mg, cohort=spec.get("cohort"),
                                          response=response))
            v = min(max(v, lo), hi)
            pts.append({"ts": day, "value": _quantize_within_slope(
                pts[-1] if pts else None, v, day, mwd, TRUTH_NDIGITS)})
        return pts
    _cv = _clinical_cv().get(sig, 0.0)
    pts: list[dict] = []
    for _k, day in enumerate(range(0, end_day + 1, int(spec["step"]))):
        v = spec["base"] + spec["per_kg"] * CLINICAL_ATTEN * (w_at(day) - w0)
        v += (drug_term(day) if drug_term is not None else
              _drug_effects.effect_at(drug, sig, day, adh_at,
                                      per_kg=spec["per_kg"], atten=CLINICAL_ATTEN,
                                      dose_mg=dose_mg, cohort=spec.get("cohort"),
                                      response=response))
        # Measurement variation: independent per draw (indexed by draw, not day).
        if _cv:
            v *= 1.0 + _cv * math.sqrt(3.0) * events_mod._det_shock(
                f"{case_id}|{sig}|meas", _k)
        v = min(max(v, lo), hi)
        pts.append({"ts": day,
                    "value": _quantize_within_slope(
                        pts[-1] if pts else None, v, day, mwd, spec["ndigits"])})
    # Add a final sample on `end_day`, which a coarse step would otherwise miss.
    if pts and int(pts[-1]["ts"]) < int(end_day):
        v = spec["base"] + spec["per_kg"] * CLINICAL_ATTEN * (w_at(int(end_day)) - w0)
        v += (drug_term(int(end_day)) if drug_term is not None else
              _drug_effects.effect_at(drug, sig, int(end_day), adh_at,
                                      per_kg=spec["per_kg"], atten=CLINICAL_ATTEN,
                                      dose_mg=dose_mg, cohort=spec.get("cohort"),
                                      response=response))
        if _cv:
            v *= 1.0 + _cv * math.sqrt(3.0) * events_mod._det_shock(
                f"{case_id}|{sig}|meas", len(pts))
        v = min(max(v, lo), hi)
        pts.append({"ts": int(end_day),
                    "value": _quantize_within_slope(
                        pts[-1], v, int(end_day), mwd, spec["ndigits"])})
    return pts


def _u01(key: str) -> float:
    """Deterministic uniform in [0, 1) from `blake2b(key)`."""
    import hashlib
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _z(key: str) -> float:
    """Deterministic standard normal, clipped to +-3."""
    u = min(max(_u01(key), 1e-9), 1.0 - 1e-9)
    return max(-3.0, min(3.0, statistics.NormalDist().inv_cdf(u)))


def _pick(weights: dict, u: float):
    """The key of `weights` that the uniform `u` falls on (cumulative, in listed order)."""
    tot = float(sum(float(w) for w in weights.values()))
    acc = 0.0
    for k, w in weights.items():
        acc += float(w) / tot
        if u < acc:
            return k
    return list(weights)[-1]


def _routine_gap(key: str) -> int:
    """One routine inter-visit gap (days): lognormal around the registered median, clipped."""
    g = _lab_obs_cfg()["routine_gap_days"]
    v = float(g["median"]) * math.exp(float(g["log_sd"]) * _z(key))
    return int(round(min(max(v, float(g["min"])), float(g["max"]))))


def lab_draw_days(case_id: str, end_day: int, dose_pts=None) -> list[int]:
    """The case's lab visit calendar: day 0, routine visits at lognormal gaps, the course
    end, and a visit within `event_redraw.window_days` after a dose change (with
    `p_after_dose_change`).
    Symptom-triggered and panel visits are added by `observe_labs`, once the events exist."""
    cfg = _lab_obs_cfg()
    # Day 0 and the course-end assessment bracket the course (the end visit lies after T,
    # so the solver never sees it; GEN15 and the verifier read the course-wide change).
    days, t, k = [0, int(end_day)], 0, 0
    while True:
        t += _routine_gap(f"{case_id}|labgap|{k}")
        k += 1
        if t >= int(end_day):
            break
        days.append(t)
    er = cfg.get("event_redraw") or {}
    win = int(er.get("window_days", 14))
    prev = None
    for q in sorted((dose_pts or []), key=lambda x: int(x["ts"])):
        v = q.get("value")
        if prev is not None and v != prev:
            d0 = int(q["ts"])
            if _u01(f"{case_id}|labdose|{d0}") < float(er.get("p_after_dose_change", 0.0)):
                d = d0 + 1 + int(_u01(f"{case_id}|labdose-lag|{d0}") * win)
                if d <= int(end_day):
                    days.append(d)
        prev = v
    return sorted(set(days))


def _drawn(case_id: str, group: str, day: int, tag: str = "labinc") -> bool:
    p_in = float((_lab_obs_cfg().get("draw_inclusion") or {}).get(group, 1.0))
    return day == 0 or _u01(f"{case_id}|{tag}|{group}|{day}") < p_in


def lab_item_days(case_id: str, sigs, visits: list[int]) -> dict[str, list[int]]:
    """Which planned item is drawn on which visit: day 0 carries every item; later a group
    (`groups`) is drawn with `draw_inclusion`, the same decision for all its members."""
    last = max(visits) if visits else 0
    return {sig: [d for d in visits if d == last or _drawn(case_id, _lab_group_of(sig), d)]
            for sig in sigs}


#: Per-case inputs of the lab truth, kept by `HaenvGenerator.generate` for `observe_labs`
#: (which evaluates the truth on the visit days added after the events). Keyed by case id.
_LAB_CTX: dict[str, dict] = {}


def _draw_clock(case_id: str, day: int, rep: int = 0) -> str:
    """The draw's time of day, `HH:MM:SS`; a same-day repeat is 1-6 hours later."""
    cfg = _lab_obs_cfg()
    h = int(_pick({int(k): v for k, v in (cfg.get("draw_hour_weights") or {9: 1}).items()},
                  _u01(f"{case_id}|labhour|{day}")))
    if rep:
        h = min(23, h + 1 + int(_u01(f"{case_id}|labhour-rt|{day}") * 6))
    mi = int(_u01(f"{case_id}|labmin|{day}|{rep}") * 60)
    se = 1 + int(_u01(f"{case_id}|labsec|{day}|{rep}") * 59)
    return f"{h:02d}:{mi:02d}:{se:02d}"


def lab_print_ndigits(case_id: str, sig: str) -> int:
    """The printed precision of `sig` in this case's lab reports: the registered `ndigits`,
    or a finer one the reporting lab uses (`print_decimals_mix`, one decision per case)."""
    nd = int(_indicators.of(sig)["ndigits"])
    mix = (_lab_obs_cfg().get("print_decimals_mix") or {}).get(sig) or {}
    u = _u01(f"{case_id}|labprint")
    acc = 0.0
    for d, pr in sorted(mix.items()):
        acc += float(pr)
        if u < acc:
            return max(nd, int(d))
    return nd


def _lab_own_cv(sig: str) -> float:
    """An item's own per-draw CV: the total (`_clinical_cv`) minus its group's shared shock."""
    tot = float(_clinical_cv().get(sig, 0.0))
    sh = float((_lab_obs_cfg().get("shared_shock_cv") or {}).get(_lab_group_of(sig), 0.0) or 0.0)
    return math.sqrt(max(0.0, tot * tot - sh * sh))


def observe_labs(raw, case_id: str) -> tuple[dict, dict]:
    """The measurement layer of the lab-calendar streams, run after the events (the symptom
    and panel days are known) and after GV-1 judged the truth.

    * visits: after each patient-reported symptom (`event_redraw.p_after_symptom`, the same
      rate whatever the symptom's role) and on every lab-panel day (the lipid group always:
      the panel's TC/HDL and the LDL/TG streams are one lipid profile; the other groups
      with `draw_inclusion`);
    * per-draw variation on the log scale: own CV plus the group's shared shock, whose
      total is the EFLM-based `clinical_measurement.cv`;
    * same-day repeats (later that day) and carried-forward readings;
    * the registered printed precision (`indicators.yaml`) and a draw time of day.

    The truth is re-rendered from the generator's inputs on the final visit days (lab
    streams are `physio: exclude`, so nothing upstream moved them). Returns
    `(truth, audit)`; `truth` is each lab stream's noise-free series on its final days,
    for the slope checks of step 4.
    """
    ctx = _LAB_CTX.get(str(case_id))
    ld = raw.longitudinal_data or {}
    sigs = [s for s in ld if s in lab_signals() and s in (ctx or {}).get("clinical", {})]
    if not ctx or not sigs:
        return {}, {}
    cfg = _lab_obs_cfg()
    er = cfg.get("event_redraw") or {}
    win = int(er.get("window_days", 14))
    ce = int(ctx["ce"])
    extra: set[int] = set()
    panel_days: set[int] = set()
    for i, ev in enumerate(raw.evidence_ledger or []):
        ts = ev.get("source_timestamp")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not 0 <= int(ts) <= ce:
            continue
        st = str(ev.get("source_type") or "")
        if st == "lab_result":
            panel_days.add(int(ts))
        elif st == "patient_reported_symptom":
            d0 = int(ts)
            if _u01(f"{case_id}|labsym|{d0}|{i}") < float(er.get("p_after_symptom", 0.0)):
                d = d0 + 1 + int(_u01(f"{case_id}|labsym-lag|{d0}|{i}") * win)
                if d <= ce:
                    extra.add(d)
    truth: dict[str, list[dict]] = {}
    for sig in sigs:
        g = _lab_group_of(sig)
        spec = ctx["clinical"][sig]
        have = {int(q["ts"]) for q in ld[sig]}
        # a panel day is a lipid-profile draw (TC/HDL on the panel, LDL/TG on the streams);
        # the other groups join it, like any visit, with `draw_inclusion`
        add = {d for d in extra | panel_days if _drawn(str(case_id), g, d, "labinc-sym")}
        if g == "lipids":
            add |= panel_days
        truth[sig] = render_clinical(sig, spec, ctx["wpts"], ce, drug=ctx["drug"],
                                     dose_mg=ctx["dose_mg"], adherence_pts=ctx["adh"],
                                     case_id=str(case_id), response=ctx["response"],
                                     days=sorted(have | add),
                                     drug_term=(ctx["drug_term"](sig, spec)
                                                if ctx.get("drug_term") else None))
    shared = cfg.get("shared_shock_cv") or {}
    rr = float(cfg.get("same_day_retest_rate", 0.0))
    rcv = float(cfg.get("retest_cv_frac", 0.5))
    cf = float(cfg.get("carried_forward_rate", 0.0))
    n_rt = n_cf = 0
    for sig in sigs:
        g = _lab_group_of(sig)
        lo, hi = ctx["clinical"][sig]["range"]
        own, sh, tot = _lab_own_cv(sig), float(shared.get(g, 0.0) or 0.0), float(_clinical_cv().get(sig, 0.0))
        nd = lab_print_ndigits(str(case_id), sig)
        out: list[dict] = []
        for q in truth[sig]:
            d = int(q["ts"])
            e = own * _z(f"{case_id}|labz|{sig}|{d}") + sh * _z(f"{case_id}|labg|{g}|{d}")
            reps = 2 if _u01(f"{case_id}|labrt?|{g}|{d}") < rr else 1
            for rep in range(reps):
                ee = e + (rcv * tot * _z(f"{case_id}|labrt|{sig}|{d}") if rep else 0.0)
                v = min(max(float(q["value"]) * math.exp(ee), lo), hi)
                if out and rep == 0 and _u01(f"{case_id}|labcf|{sig}|{d}") < cf:
                    v = float(out[-1]["value"])
                    n_cf += 1
                v = round(v, nd) if nd else float(round(v))
                out.append({"ts": d, "value": v, "time": _draw_clock(str(case_id), d, rep)})
            n_rt += reps - 1
        raw.longitudinal_data[sig] = out
    pre = {s: truth[s] for s in sigs}
    moved = realign_clinical_ledger(raw.evidence_ledger, raw.longitudinal_data, reference=pre)
    return truth, {"symptom_visits": len(extra), "panel_visits": len(panel_days),
                   "same_day_retests": n_rt, "carried_forward": n_cf,
                   "ledger_realigned": moved,
                   "draws": {s: len(raw.longitudinal_data[s]) for s in sigs}}


#: `latent_premise` key of the verifier-side medication course (`HaenvGenerator.generate`).
WORLD_MEDICATION_KEY = "world_medication"


def _true_adherence_pts(p, T: int) -> list[dict]:
    """The premise's true adherence trajectory as points; drives the drug effect."""
    return ([{"ts": pt["day"], "value": pt["level"]}
             for pt in p.adherence.get("trajectory", [])]
            or [{"ts": T, "value": p.adherence.get("baseline", 0.95)}])


def _world_medication(p, cid: str, T: int, ce: int) -> tuple[list, list, list]:
    """(primary dose record, observed adherence readings, concurrent medicines), a function of
    the premise's regimen, comorbidities, true adherence and `cid` only (`med_course`)."""
    from haenv_kernel.latent import drug_pkpd
    reg = p.patient_basics.get("regimen", {}) or {}
    steps = list(reg.get("dose_steps", [2.5, 5.0, 7.5]))
    mt = int((drug_pkpd(str(reg.get("drug") or "")) or {}).get("min_titration_days", 28))
    traj = [{"day": pt["day"], "level": pt["level"]} for pt in p.adherence.get("trajectory", [])]
    return (_med_course.dose_course(cid, steps, T, ce, mt),
            _med_course.adherence_readings(cid, T, ce, traj,
                                           float(p.adherence.get("baseline", 0.95))),
            _med_course.concurrent_medicines(cid, _raw_field(p.patient_basics, "disease"),
                                             p.patient_basics.get("comorbidities") or [],
                                             str(reg.get("drug") or ""), ce))


def world_medication_record(p) -> dict:
    """Verifier-side record of the whole medication course (`latent_premise`); the same draws
    as `HaenvGenerator.generate`, never cut into the solver payload."""
    m = p.meta or {}
    T = int(m.get("prediction_time_T", 84))
    dose, _obs, concurrent = _world_medication(p, str(m.get("case_id") or "HAENV"), T,
                                               int(m.get("course_end_day", END)))
    return {"primary": {"drug": _drug_of(p), "dose_timeline": dose},
            "concurrent": concurrent, "adherence_true": _true_adherence_pts(p, T)}


def _drug_term_fn(p, sig: str, spec: dict, dose_pts: list, adh_pts: list, concurrent: list):
    """`day -> drug-effect increment on sig`, read along the dose line: the primary drug's
    glucose effect (`drug_effects.effect_along`), antihypertensive effects and concurrent
    medicines' deviations (`med_course`). Shared by `render_clinical` and GEN15."""
    drug, resp = _drug_of(p), _drug_response_of(p)
    adh_sorted = sorted((q for q in (adh_pts or []) if isinstance(q, dict)),
                        key=lambda q: int(q.get("ts", -1)))

    def adh_at(day: int) -> float:
        cur = 1.0
        for q in adh_sorted:
            if int(q.get("ts", -1)) > day:
                break
            cur = float(q["value"])
        return cur

    def term(day: int) -> float:
        v = _drug_effects.effect_along(drug, sig, day, adh_at,
                                       lambda d: _med_course.dose_at(dose_pts, d),
                                       per_kg=spec["per_kg"], atten=CLINICAL_ATTEN,
                                       cohort=spec.get("cohort"), response=resp)
        v += _med_course.bp_term(sig, day, drug, dose_pts, adh_at, concurrent)
        v += _med_course.concurrent_glucose_term(sig, day, concurrent, spec["per_kg"],
                                                 CLINICAL_ATTEN, spec.get("cohort"))
        return v
    return term


#: Per case, the clinical ledger entries `LLMCaseGenerator` re-read from the rendered
#: streams (last GV-1 round); popped into the build audit.
_CLINICAL_LEDGER_AUDIT: dict[str, list] = {}


def _latest_at_or_before(pts: list, ts) -> dict | None:
    """The stream point at or before day `ts` (the first point when none is)."""
    pts = sorted((q for q in (pts or []) if isinstance(q, dict) and "ts" in q and "value" in q),
                 key=lambda q: int(q["ts"]))
    return ([q for q in pts if int(q["ts"]) <= int(ts)] or pts[:1] or [None])[-1]


def _fmt_reading(x) -> str:
    return f"{float(x):g}"


#: A blood-pressure reading written as text, e.g. "142/88 mmHg".
_BP_TEXT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(.*)$", re.S)


def realign_clinical_ledger(ledger: list, ld: dict, reference: dict | None = None) -> list[dict]:
    """Re-reads each clinical `measured_value` in the ledger from its rendered stream.

    Covers the `source_type`s whose `gates.LEDGER_VALUE_STREAMS` candidates are clinical
    streams (lab_panel / cgm / bp_cuff), in the three forms the model writes:

    * a number: the quoted stream is the candidate whose `reference` series (the values the
      entry was written against, i.e. the model's) is closest in value at the sample nearest
      the entry's day; the entry takes that stream's latest point at or before its day;
    * a mapping keyed by stream name: each key that is a rendered clinical stream takes that
      stream's latest value at or before the entry's day (other keys are left as written);
    * a `"S/D ..."` blood-pressure text: rewritten from `systolic_bp` / `diastolic_bp`.

    Mutates `ledger`; returns the moves.
    """
    moves: list[dict] = []
    ref = reference or {}
    for ev in ledger or []:
        cands = {s for s in (gates.LEDGER_VALUE_STREAMS.get(str(ev.get("source_type") or "")) or ())
                 if s in CLINICAL_SPEC}
        v, ts = ev.get("measured_value"), ev.get("source_timestamp")
        if not cands or isinstance(ts, bool) or not isinstance(ts, (int, float)):
            continue
        eid = str(ev.get("evidence_id"))
        if isinstance(v, dict):                                    # {"HbA1c": 5.6, "LDL": 2.53, ...}
            nv, hit = dict(v), {}
            for k in v:
                q = _latest_at_or_before(ld.get(k), ts) if k in CLINICAL_SPEC else None
                if q is not None:
                    nv[k], hit[k] = q["value"], int(q["ts"])
            if not hit:
                continue
            nts = next(iter(hit.values())) if len(set(hit.values())) == 1 else int(ts)
            if nv != v or nts != int(ts):
                moves.append({"evidence_id": eid, "stream": sorted(hit), "from": [ts, v], "to": [nts, nv]})
            ev["source_timestamp"], ev["measured_value"] = nts, nv
            continue
        if isinstance(v, str):                                     # "142/88 mmHg"
            m = _BP_TEXT.match(v)
            qs = _latest_at_or_before(ld.get("systolic_bp"), ts)
            qd = _latest_at_or_before(ld.get("diastolic_bp"), ts)
            if not m or qs is None or qd is None or "systolic_bp" not in cands:
                continue
            nv = f"{_fmt_reading(qs['value'])}/{_fmt_reading(qd['value'])}{m.group(3)}"
            nts = int(qs["ts"]) if int(qs["ts"]) == int(qd["ts"]) else int(ts)
            if nv != v or nts != int(ts):
                moves.append({"evidence_id": eid, "stream": ["systolic_bp", "diastolic_bp"],
                              "from": [ts, v], "to": [nts, nv]})
            ev["source_timestamp"], ev["measured_value"] = nts, nv
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        best = None
        for s in sorted(cands):
            new = [q for q in (ld.get(s) or []) if isinstance(q, dict) and "ts" in q and "value" in q]
            if not new:
                continue
            rs = [q for q in (ref.get(s) if isinstance(ref.get(s), list) else new)
                  if isinstance(q, dict) and "ts" in q and "value" in q] or new
            try:
                near = min(rs, key=lambda q: (abs(float(q["ts"]) - float(ts)), float(q["ts"])))
                score = abs(float(near["value"]) - float(v)) / max(abs(float(v)), 1e-6)
            except (TypeError, ValueError):
                continue
            if best is None or score < best[0]:
                best = (score, s, new)
        if best is None:
            continue
        s = best[1]
        tgt = _latest_at_or_before(best[2], ts)
        if int(tgt["ts"]) != int(ts) or float(tgt["value"]) != float(v):
            moves.append({"evidence_id": eid, "stream": s,
                          "from": [ts, v], "to": [int(tgt["ts"]), tgt["value"]]})
        ev["source_timestamp"], ev["measured_value"] = int(tgt["ts"]), tgt["value"]
    return moves


def _reading_at_or_before(pts: list, ts) -> dict | None:
    """The stream reading at or before day `ts`; `None` when there is none."""
    q = _latest_at_or_before([q for q in pts or [] if isinstance(q, dict) and q.get("value") is not None], ts)
    return q if q is not None and int(q["ts"]) <= int(ts) else None


def realign_ledger_to_streams(ledger: list, ld: dict) -> tuple[list, list]:
    """Re-reads every ledger `measured_value` from the final question-face streams.

    Runs after the events, physiology and observation layers, so a quoted reading is the one
    the solver sees. Numbers whose only candidate is `weight` are left to step (3e)
    (`ledger_weight_point`). Per form:

    * a number: kept when a candidate stream (`gates.LEDGER_VALUE_STREAMS`) has that value at
      the entry's day; otherwise it takes the latest point at or before its day of the
      candidate whose reading there is closest;
    * a mapping: keys that are question-face streams take their latest value at or before the
      entry's day (the entry moves to the latest of those days, and a key without a reading
      on that day is removed); other keys are removed, since no stream carries them;
    * a `"S/D ..."` blood-pressure text: rewritten from `systolic_bp` / `diastolic_bp`.

    An entry left without any value is removed. Mutates `ledger`; returns `(moves, removed)`.
    """
    moves: list[dict] = []
    removed: list[dict] = []
    keep: list = []

    def on(s, ts, v):
        return any(isinstance(q, dict) and int(q.get("ts", -1)) == int(ts)
                   and q.get("value") is not None and abs(float(q["value"]) - float(v)) <= 1e-9
                   for q in ld.get(s) or [])

    for ev in ledger or []:
        v, ts = ev.get("measured_value"), ev.get("source_timestamp")
        cands = gates.LEDGER_VALUE_STREAMS.get(str(ev.get("source_type") or ""))
        if (v is None or isinstance(v, bool) or not cands
                or isinstance(ts, bool) or not isinstance(ts, (int, float))):
            keep.append(ev)
            continue
        eid = str(ev.get("evidence_id"))
        if isinstance(v, dict):
            got = {k: _reading_at_or_before(ld.get(k), ts) for k in v if ld.get(k)}
            got = {k: q for k, q in got.items() if q is not None}
            if not got:
                removed.append({"evidence_id": eid, "at": [ts, v], "why": "no question-face stream"})
                continue
            nts = max(int(q["ts"]) for q in got.values())
            nv = {k: q["value"] for k in got
                  if (q := _reading_at_or_before(ld[k], nts)) is not None and int(q["ts"]) == nts}
            if nv != v or nts != int(ts):
                moves.append({"evidence_id": eid, "stream": sorted(nv), "from": [ts, v], "to": [nts, nv]})
            ev["source_timestamp"], ev["measured_value"] = nts, nv
            keep.append(ev)
            continue
        if isinstance(v, str):
            m = _BP_TEXT.match(v)
            if not m or "systolic_bp" not in cands:
                keep.append(ev)
                continue
            qs = _reading_at_or_before(ld.get("systolic_bp"), ts)
            qd = _reading_at_or_before(ld.get("diastolic_bp"), ts)
            if qs is None or qd is None:
                removed.append({"evidence_id": eid, "at": [ts, v], "why": "no question-face stream"})
                continue
            nts = min(int(qs["ts"]), int(qd["ts"]))
            qs = _reading_at_or_before(ld["systolic_bp"], nts)
            qd = _reading_at_or_before(ld["diastolic_bp"], nts)
            nv = f"{_fmt_reading(qs['value'])}/{_fmt_reading(qd['value'])}{m.group(3)}"
            if int(qs["ts"]) != int(qd["ts"]):
                removed.append({"evidence_id": eid, "at": [ts, v], "why": "no paired reading"})
                continue
            if nv != v or nts != int(ts):
                moves.append({"evidence_id": eid, "stream": ["systolic_bp", "diastolic_bp"],
                              "from": [ts, v], "to": [nts, nv]})
            ev["source_timestamp"], ev["measured_value"] = nts, nv
            keep.append(ev)
            continue
        if not isinstance(v, (int, float)) or cands == {"weight"}:
            keep.append(ev)
            continue
        present = [s for s in sorted(cands) if ld.get(s)]
        if not present or any(on(s, ts, v) for s in present):
            keep.append(ev)                                     # traceable, or GEN25's `ledger_stream_absent`
            continue
        near = [q for s in present if (q := _reading_at_or_before(ld[s], ts)) is not None]
        if not near:
            removed.append({"evidence_id": eid, "at": [ts, v], "why": "no reading at or before its day"})
            continue
        tgt = min(near, key=lambda q: abs(float(q["value"]) - float(v)))
        moves.append({"evidence_id": eid, "from": [ts, v], "to": [int(tgt["ts"]), tgt["value"]]})
        ev["source_timestamp"], ev["measured_value"] = int(tgt["ts"]), tgt["value"]
        keep.append(ev)
    ledger[:] = keep
    return moves, removed
