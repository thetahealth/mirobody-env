"""Finite, serializable generation sources. Measurements are never daily truth.

Clinical queries keep the producer's original preceding calendar: the sequential
slope limiter must never restart at the requested sparse subset. Negative history
has a different producer and is refused without its own saved source.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

NAMES = {"creatinine": ("Cr", "creatinine"), "potassium": ("K", "potassium"),
         "glucose": ("FBG", "fasting_glucose"), "fasting_glucose": ("FBG", "fasting_glucose"),
         "alt": ("ALT", "ALT"), "ck": ("CK", "CK"), "tsh": ("TSH", "TSH"),
         "triglycerides": ("TG", "triglycerides")}


class SourceUnavailable(ValueError):
    """A producer cannot justify the requested truth or its coverage."""


def refuse(message):
    raise SourceUnavailable("shared truth " + message)


def unmanaged(med):
    med = copy.deepcopy(med)
    if (med.get("primary") or {}).get("overlay"):
        med["primary"] = None
    med["concurrent"] = [m for m in med.get("concurrent") or [] if not m.get("overlay")]
    return med


class Source:
    """Callable facade over JSON data, without a saved Python callable."""

    def __init__(self, source):
        if not isinstance(source, dict) or type(source.get("version")) is not int or source["version"] != 1:
            refuse("has an unsupported or missing source version")
        self.source = copy.deepcopy(source)
        self._clinical_model = None

    def _producer(self, day):
        from haenv.world.truth.plan import value_at, delta
        s = self.source
        if s["kind"] == "clinical":
            from haenv.world.generation.build_clinical import _drug_term_fn, render_clinical
            if self._clinical_model is None:
                p = s["producer"]
                spec = copy.deepcopy(s["spec"])
                fn = _drug_term_fn(SimpleNamespace(patient_basics=p["patient_basics"], meta=p["meta"]),
                                  s["stream"], spec, p["dose"], p["adherence"], p["concurrent"], coupled=True)
                self._clinical_model = spec, fn
            spec, fn = self._clinical_model
            if day < 0:
                from haenv.world.generation.build_clinical import soft_bound
                return soft_bound(s["history"]["level"] + fn(day) - fn(0), *spec["range"])
            days = sorted({d for d in s["calendar"] if d <= day} | {day})
            value = render_clinical(s["stream"], spec, s["weight"], s["window"][1],
                                    days=days, drug_term=fn)[-1]["value"]
            entry = s.get("condition")
            if entry and entry["kind"] in ("approach", "step") and day >= entry["onset"]:
                value = round(value + delta(entry, day), 6)
            return value
        if s["kind"] in ("observable", "p4_latent"):
            entry = s["condition"]
            def baseline_at(d):
                if entry:
                    return value_at(entry, d)
                component = (s["origin"].get("components") or {}).get("glucose")
                if component:
                    import math
                    from haenv.world.labworld.truth import fg_log
                    inputs = {k: component[k] for k in
                              ("kappa", "beta", "tau", "dose_pts", "weight_pts", "smooth_days")}
                    return math.exp(fg_log(d, base=s["baseline"], **inputs))
                return s["baseline"]
            value = baseline_at(day)
            if entry and entry.get("drug_effect"):
                return value
            from haenv.world.truth import drugs
            from haenv.world.clinical.med_course import dose_at
            from haenv.world.clinical.drug_effects import floor_term
            med = s["medication"]
            medicines = {m["drug"]: m for m in drugs.medicines(med)}
            primary = (med.get("primary") or {}).get("drug")
            def adh(name, d):
                pts = [p for p in med.get("adherence_true") or [] if int(p["ts"]) <= d]
                return float(pts[-1]["value"]) if name == primary and pts else 1.
            effect = drugs.term(s["stream"], day, drugs=medicines,
                dose_at=lambda name, d: dose_at(medicines[name].get("dose_timeline") or [], d),
                adh_at=adh, base=baseline_at, world_medication=med)
            return value + floor_term(s["stream"], value, effect)
        refuse(f"source unavailable for {s['stream']} on day {day}")

    def __call__(self, day):
        if isinstance(day, bool) or not isinstance(day, int):
            refuse("requires an integer date")
        s = self.source
        known = {int(q["ts"]): float(q["value"]) for q in s["known"]}
        if s["kind"] == "known":
            if day in known:
                return known[day]
            refuse(f"unavailable for {s['stream']} on day {day}: stream course missing")
        if not s["window"][0] <= day <= s["window"][1]:
            refuse(f"coverage unavailable for {s['stream']} on day {day}: outside producer window/history")
        value = self._producer(day)
        if day in known:
            if value != known[day]:
                refuse(f"source disagrees with known truth for {s['stream']} on day {day}")
            return known[day]
        return value


def snapshot(raw, case_id, an, ctx):
    """Capture actual producer inputs, refusing missing sources rather than bases."""
    from haenv.world.truth import plan_of
    fid, stream = NAMES[an]
    ctx = ctx or {}
    entry = copy.deepcopy(((plan_of(raw) or {}).get("observables") or {}).get(fid))
    known = copy.deepcopy((ctx.get("course_truth") or {}).get(stream) or [])
    seen = {}
    for q in known:
        day, value = int(q["ts"]), float(q["value"])
        if day in seen and seen[day] != value:
            refuse(f"contradictory known truth for {stream} on day {day}")
        seen[day] = value
    med = unmanaged((raw.latent_premise or {}).get("world_medication") or {})
    med["case_id"] = case_id
    if entry and entry.get("drug_effect"):
        entry["drug_effect"]["world_medication"] = unmanaged(entry["drug_effect"]["world_medication"])
        saved_med = entry["drug_effect"]["world_medication"]
        saved_med["case_id"] = case_id
        if (entry["drug_effect"]["observable"] != fid or
                json.dumps(saved_med, sort_keys=True) != json.dumps(med, sort_keys=True)):
            refuse(f"medication source disagrees for {stream}; original facts retained")
    s = {"version": 1, "stream": stream, "known": known, "condition": entry, "medication": med,
         "ownership": {"baseline": "producer", "condition": "saved observable",
                       "drug": "producer; managed overlays excluded", "measurement": "separate"}}
    spec = copy.deepcopy((ctx.get("clinical") or {}).get(stream))
    if spec:
        factory = ctx.get("drug_term")
        if not callable(factory):
            refuse(f"source unavailable for {stream}: clinical producer inputs missing")
        fn = factory(stream, spec)
        producer = copy.deepcopy(getattr(fn, "source", None))
        if not producer or not ctx.get("wpts"):
            refuse(f"source unavailable for {stream}: clinical producer inputs missing")
        producer["concurrent"] = [m for m in producer["concurrent"] if not m.get("overlay")]
        original_primary = ((raw.latent_premise or {}).get("world_medication") or {}).get("primary") or {}
        if original_primary.get("overlay") and original_primary.get("drug") == producer["patient_basics"]["regimen"]["drug"]:
            producer["patient_basics"]["regimen"]["drug"] = ""
            producer["dose"] = []
        original_med = (raw.latent_premise or {}).get("world_medication") or {}
        if "primary" in original_med:
            primary = med.get("primary") or {}
            expected = {"drug": producer["patient_basics"]["regimen"]["drug"], "dose_timeline": producer["dose"]}
            actual = {"drug": primary.get("drug") or "", "dose_timeline": primary.get("dose_timeline") or []}
            if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
                refuse(f"clinical medication source disagrees for {stream}: primary course")
        for key, source_key in (("concurrent", "concurrent"), ("adherence_true", "adherence")):
            if key in original_med and json.dumps(med[key], sort_keys=True) != json.dumps(producer[source_key], sort_keys=True):
                refuse(f"clinical medication source disagrees for {stream}: {key}")
        calendar = ((ctx.get("lab_truth_days") or {}).get(stream) or
                    (ctx.get("lab_state_days") or {}).get(stream) or
                    [int(q["ts"]) for q in (ctx.get("uncoupled_truth") or {}).get(stream) or known])
        if not calendar:
            refuse(f"source unavailable for {stream}: clinical calendar missing")
        s.update(kind="clinical", spec=spec, producer=producer, weight=copy.deepcopy(ctx["wpts"]),
                 calendar=sorted(set(int(d) for d in calendar)), window=[0, int(ctx["ce"])])
        hist = ctx.get("history") or {}
        if hist and hist.get("long", True):
            from haenv.world.generation.build_clinical import history_level
            level = history_level(raw, ctx, stream)
            if level is not None:
                s["history"] = {"level": level, "lo": int(hist["lo"]),
                                "producer": "observe_history: early anchor plus drug deviation"}
                s["window"][0] = int(hist["lo"])
    elif entry and not entry.get("stream"):
        if entry.get("kind") not in ("course", "approach", "step"):
            refuse(f"source unavailable for {stream}: unsupported observable producer")
        if entry["kind"] != "course" and not isinstance(entry.get("base"), (int, float)):
            refuse(f"source unavailable for {stream}: observable baseline missing")
        if entry["kind"] == "course" and not entry.get("points"):
            refuse(f"source unavailable for {stream}: observable points missing")
        window = ([int(entry["points"][0][0]), int(entry["points"][-1][0])]
                  if entry["kind"] == "course" else
                  [int(med.get("history_lo", 0)), int(ctx.get("ce") or raw.prediction_context["prediction_time_T"])])
        s.update(kind="observable", medication=med, window=window)
    elif known:
        s.update(kind="known", window=[min(int(q["ts"]) for q in known), max(int(q["ts"]) for q in known)])
    else:
        refuse(f"source unavailable for {stream}")
    source = Source(s)
    for q in known:
        source(int(q["ts"]))
    return source


def snapshot_p4(raw, case_id, an, ctx, origin):
    """Explicit P4 producer receipt; existing sources (even partial) have priority.

    This is deliberately separate from snapshot: a caller's numeric fallback is
    not a source. No exception from an existing provider is caught here.
    """
    import math
    from haenv.world.truth import plan_of
    fid, stream = NAMES[an]
    ctx = ctx or {}
    observables = (plan_of(raw) or {}).get("observables") or {}
    if (stream in (ctx.get("clinical") or {}) or fid in observables or
            any((ctx.get(key) or {}).get(stream) for key in
                ("course_truth", "uncoupled_truth", "uncoupled_early")) or
            (raw.longitudinal_data or {}).get(stream)):
        return snapshot(raw, case_id, an, ctx)
    if (origin.get("owner") != "P4" or origin.get("case_id") != str(case_id) or
            origin.get("analyte") != an or origin.get("producer") not in
            ("haenv.p4.diversity:baseline_origin", "haenv.p4.realize:legacy_baseline_origin") or
            not origin.get("provenance")):
        refuse(f"source unavailable for {stream}: invalid P4 producer origin")
    parameter = origin.get("parameter")
    if isinstance(parameter, bool) or not isinstance(parameter, (int, float)) or not math.isfinite(parameter) or parameter <= 0:
        refuse(f"source unavailable for {stream}: invalid P4 parameter")
    med = unmanaged((raw.latent_premise or {}).get("world_medication") or {})
    med["case_id"] = case_id
    component = (origin.get("components") or {}).get("glucose")
    if component and component["kappa"]:
        from haenv.world.truth import drugs
        row = drugs.analyte_effects().get((component["drug"], fid))
        if row and row.get("applied", True):
            refuse(f"source conflict for {stream}: duplicate legacy and registered drug owner")
        if "primary" in ((raw.latent_premise or {}).get("world_medication") or {}):
            primary = med.get("primary") or {}
            if (primary.get("drug") != component["drug"] or
                    [q for q in primary.get("dose_timeline") or []
                     if int(q["ts"]) <= component["horizon"]] != component["dose_pts"]):
                refuse(f"medication source disagrees for {stream}: legacy primary course")
    hist = ctx.get("history") or {}
    lows = [int(v) for v in (hist.get("lo") if hist.get("long", True) else None,
                             med.get("history_lo")) if v is not None]
    # Positive coverage is the saved model's declared course, never an arbitrary
    # future query. With no course metadata, only the actual review is supported.
    T = int(raw.prediction_context["prediction_time_T"])
    meta = (raw.latent_premise or {}).get("meta") or {}
    ends = [int(v) for v in (ctx.get("ce"), meta.get("course_end_day"),
                             component.get("horizon") if component else None) if v is not None]
    window = [max(lows) if lows else 0, min(ends) if ends else T]
    if window[0] > window[1]:
        refuse(f"source unavailable for {stream}: contradictory P4 coverage")
    baseline = parameter
    if an == "creatinine" and not origin.get("comorbidity"):
        from haenv.p4.realize import background_cr
        baseline = background_cr(raw, case_id, parameter, T)
    return Source({"version": 1, "kind": "p4_latent", "stream": stream,
        "known": [], "condition": None, "origin": copy.deepcopy(origin),
        "baseline": baseline, "baseline_constraint": {"analyte": an, "day": T,
            "parameter": parameter, "constrained": baseline, "scope": "baseline only"},
        "window": window, "coverage_origin": {"history": copy.deepcopy(hist),
            "world_medication_history_lo": med.get("history_lo"), "model_ends": ends, "review": T},
        "medication": med, "ownership": {"baseline": "P4",
            "drug": "registered applied drug terms; managed overlays excluded",
            "condition": "none; existing sources have priority", "scenario": "separate P4 layer",
            "measurement": "separate independent draws"}})
