"""Dirty-data injection applied after the kernel's injectors.

Adds carried-forward readings (a nurse copying the previous value) at the rate
measured on real nursing records (`registry/artifact_rates.yaml`), which the
kernel's `NOISE_CLASSES` cannot express, backfills the clinic-scale reference
`weight_ref`, and runs injectors registered via
`world_plugins.register_post_injector`. Every change still goes through the
verification loop that requires gold labels and the offline solver's
conclusion to be unchanged; plugins cannot write label-bearing streams.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
import logging

log = logging.getLogger("haenv.post_inject")

#: Streams the built-in injectors act on: only `weight` has a calibrated rate.
BUILTIN_TARGETS: tuple[str, ...] = ("weight",)


def _u01(seed: str, i: int) -> float:
    """A deterministic uniform number in [0,1) from `blake2b(seed|i)`."""
    h = hashlib.blake2b(f"{seed}|{i}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def carried_forward(points, *, rng_key: str, rate: float = 0.0,
                    protect_ts: frozenset = frozenset(), **_):
    """With probability `rate`, replace a point with the previous point's value,
    keyed on the original index. `rate <= 0` returns the input unchanged.

    `protect_ts` holds the days whose readings carry a trap's evidence
    (`noise.trap_evidence_days`, `window_evidence_days`): such a reading is never replaced
    and never copied onto the next reading.
    """
    if rate <= 0 or not points or len(points) < 2:
        return points
    out = [dict(points[0])]
    for i in range(1, len(points)):
        p = dict(points[i])
        if (_u01(rng_key, i) < rate and p.get("ts") not in protect_ts
                and points[i - 1].get("ts") not in protect_ts):
            p["value"] = out[-1]["value"]
        out.append(p)
    return out


#: Window artifacts whose evidence is every reading in the window and the first reading
#: after it (the offset stretch and its fall-back).
WINDOW_EVIDENCE_CLASSES: tuple[str, ...] = ("context_confound",)


def window_evidence_days(raw) -> frozenset:
    """Days inside the windows of `WINDOW_EVIDENCE_CLASSES` artifacts (`artifact_flags.windows`)."""
    flags = (getattr(raw, "adjudication", None) or {}).get("artifact_flags") or {}
    wins = flags.get("windows") or ([flags] if flags.get("window") else [])
    return frozenset(d for w in wins if w.get("noise_class") in WINDOW_EVIDENCE_CLASSES
                     for d in range(int(w["window"][0]), int(w["window"][1]) + 1))


#: Built-in injectors: name -> (callable, parameter key in `artifact_rates.yaml`).
BUILTIN = {"carried_forward": (carried_forward, "carried_forward")}

#: Clinic-scale sampling cadence (days); must equal the kernel's
#: `noise._add_clean_ref` cadence so the source of a reference is not revealed.
CLINIC_SCALE_EVERY_DAYS = 28

#: Calibration vocabulary shared with `artifact_rates.yaml`'s header.
_CALIBRATION_VALUES = ("measured", "corpus_too_thin", "no_corpus", "by_design")

#: Injectors with an additive kg magnitude on `weight`; the clinic-scale device
#: difference must stay well below the smallest of them.
_ADDITIVE_KG_INJECTORS = ("device_switch", "transient_spike", "context_confound")


class ClinicScaleParamsError(ValueError):
    """`reference_streams.clinic_scale` is missing or violates its design constraint
    (no default: a zero offset would make `weight_ref` a copy of `weight`)."""


def clinic_scale_params() -> dict:
    """Read and validate `artifact_rates.yaml:reference_streams.clinic_scale`
    (without the plugin overlay).

    * lower bound: `jitter_kg < min(bias_abs_kg)` and `min_abs_kg >= 0.01`, so
      every offset is nonzero after rounding to 2 decimals;
    * upper bound: `max(bias_abs_kg) + jitter_kg <= d/4`, d = the smallest
      additive artifact magnitude, which leaves a gap of at least d/2 between
      device differences and artifacts.
    """
    from . import artifact_params as _ap
    from .yamlcache import load_yaml
    doc = load_yaml(_ap.REGISTRY) or {}
    return validate_clinic_scale(
        (doc.get("reference_streams") or {}).get("clinic_scale"), _ap.injectors(),
        where=f"{_ap.REGISTRY}: reference_streams.clinic_scale")


def validate_clinic_scale(spec, injectors: dict, *, where: str = "clinic_scale") -> dict:
    """Check a clinic-scale spec against the bounds in `clinic_scale_params` (pure)."""
    if not isinstance(spec, dict):
        raise ClinicScaleParamsError(f"{where} is missing; there is no fallback to 0 (that would restore the verbatim-copy shortcut)")
    try:
        lo, hi = (float(x) for x in spec["bias_abs_kg"])
        jitter = float(spec["jitter_kg"])
        min_abs = float(spec["min_abs_kg"])
        calib = spec["calibration"]
    except (KeyError, TypeError, ValueError) as e:
        raise ClinicScaleParamsError(f"{where}: field missing or not a number: {e!r}") from e
    if calib not in _CALIBRATION_VALUES:
        raise ClinicScaleParamsError(f"{where}.calibration={calib!r} is not an allowed value")
    if not (0.0 < lo <= hi):
        raise ClinicScaleParamsError(f"{where}.bias_abs_kg=[{lo},{hi}] must satisfy 0 < lo <= hi")
    if not (0.0 <= jitter < lo):
        raise ClinicScaleParamsError(
            f"{where}: jitter_kg={jitter} must be < min(bias_abs_kg)={lo}, or jitter can cancel the offset to 0")
    if min_abs < 0.01:
        raise ClinicScaleParamsError(
            f"{where}: min_abs_kg={min_abs} < 0.01; after rounding to 2 decimals it may equal weight exactly")
    kg = [float(v["value"]) for n, v in (injectors or {}).items()
          if n in _ADDITIVE_KG_INJECTORS and v.get("value") is not None]
    if not kg:
        raise ClinicScaleParamsError(f"{where}: no additive artifact magnitude found; the upper bound cannot be checked")
    if (hi + jitter) * 4.0 > min(kg):
        raise ClinicScaleParamsError(
            f"{where}: max device offset {hi + jitter:.3f} kg exceeds a quarter of the smallest additive artifact {min(kg)} kg;"
            f" the in/out-of-window gap would be < d/2 and the device offset would mask the artifact")
    return {"bias_lo": lo, "bias_hi": hi, "jitter": jitter, "min_abs": min_abs}


def clinic_scale_offset(case_id: str, ts, params: dict) -> float:
    """Clinic-scale reading minus the true weight on day `ts`, in kg.

    A function of `(case_id, ts)` and the registry only: a per-case signed bias,
    a per-reading residual in `[-jitter, +jitter]`, floored at `min_abs`.
    """
    from .rng import unit
    sign = 1.0 if unit(case_id, "clinic_scale", "sign") < 0.5 else -1.0
    bias = params["bias_lo"] + (params["bias_hi"] - params["bias_lo"]) * unit(
        case_id, "clinic_scale", "bias")
    jit = (2.0 * unit(case_id, "clinic_scale", "jitter", int(ts)) - 1.0) * params["jitter"]
    off = sign * bias + jit
    if abs(off) < params["min_abs"]:
        off = sign * params["min_abs"]
    return off


def _rates() -> dict:
    """Parameters from `artifact_rates.yaml:post_injectors`; missing means rate 0."""
    from .regpath import load_registry
    doc = load_registry("artifact_rates.yaml") or {}
    return (doc.get("post_injectors") or {})


def apply_post_injection(raw, *, case_id: str, clean_weight=None) -> list[str]:
    """Apply this layer to `raw.longitudinal_data` in place; returns what was applied.

    Called by `build` after the kernel's injectors.
    `clean_weight` is the truth `weight` series, without observation noise or reading
    artifacts; every case's reference is sampled from it. `None` falls back to the
    current `weight`.
    """
    ld = getattr(raw, "longitudinal_data", None)
    if not isinstance(ld, dict):
        return []
    cfg = _rates()
    applied: list[str] = []
    # Snapshot before carried-forward.
    _w_src = clean_weight if isinstance(clean_weight, list) and clean_weight else [
        dict(q) for q in (ld.get("weight") or []) if isinstance(q, dict)]

    # Readings that carry a trap's evidence (the kernel's artifacts target `weight`, its
    # PRIMARY signal).
    from haenv_kernel.noise import trap_evidence_days            # kernel; on the path wherever build runs
    _trap_days = trap_evidence_days(raw) | window_evidence_days(raw)

    # The home scale shows 0.1 kg: the observed weight goes on that grid before any reading is
    # copied forward, so a copied reading repeats a grid value. `weight_ref` (clinic scale) and
    # the truth snapshot above keep their own precision.
    if isinstance(ld.get("weight"), list):
        from .physio.noise import WEIGHT_SCALE_RESOLUTION_KG, weight_on_scale_grid
        ld["weight"] = [
            {**q, "value": weight_on_scale_grid(q["value"])}
            if isinstance(q, dict) and isinstance(q.get("value"), (int, float))
            and not isinstance(q.get("value"), bool) else q
            for q in ld["weight"]]
        applied.append(f"scale_resolution:weight@{WEIGHT_SCALE_RESOLUTION_KG}kg")

    for name, (fn, key) in BUILTIN.items():
        spec = cfg.get(key) or {}
        rate = float(spec.get("rate") or 0.0)
        if rate <= 0:
            continue
        for sig in BUILTIN_TARGETS:
            pts = ld.get(sig)
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            ld[sig] = fn(pts, rng_key=f"{case_id}|{sig}|{name}", rate=rate,
                         protect_ts=_trap_days if sig == "weight" else frozenset())
            applied.append(f"{name}:{sig}@{rate:.4f}")

    # Give every case a `weight_ref` from the same stage and by the same rule
    # (otherwise its presence or values would reveal an injected artifact), in the
    # kernel's shape: `{"ts","value"}` every 28 days. Every point then gets the
    # clinic scale's device difference (`clinic_scale_offset`).
    if isinstance(ld.get("weight"), list) and ld.get("weight") and not ld.get("weight_ref"):
        _ref = [{"ts": q["ts"], "value": q["value"]} for q in _w_src
                if isinstance(q, dict) and isinstance(q.get("ts"), (int, float))
                and q.get("value") is not None
                and int(q["ts"]) % CLINIC_SCALE_EVERY_DAYS == 0]
        if len(_ref) >= 2:
            ld["weight_ref"] = _ref
            applied.append(f"clinic_scale_backfill:weight_ref@{len(_ref)}pts")
    _ref = ld.get("weight_ref")
    if isinstance(_ref, list) and _ref:
        _cs = clinic_scale_params()
        ld["weight_ref"] = [
            {"ts": q["ts"],
             "value": round(float(q["value"]) + clinic_scale_offset(case_id, q["ts"], _cs), 2)}
            if isinstance(q, dict) and q.get("value") is not None
            and isinstance(q.get("ts"), (int, float)) else q
            for q in _ref]
        applied.append(f"clinic_scale_offset:weight_ref@{len(_ref)}pts")

    # Injectors registered by plugins -- go through the same path as the built-ins
    from .world_plugins import post_injectors
    for name, (fn, source) in post_injectors().items():
        spec = cfg.get(name) or {}
        sigs = [s for s in (spec.get("streams") or []) if isinstance(ld.get(s), list)]
        for sig in sigs:
            ld[sig] = fn(ld[sig], rng_key=f"{case_id}|{sig}|{name}", **{
                k: v for k, v in spec.items() if k != "streams"})
            applied.append(f"{name}:{sig}(plugin:{source})")
    return applied
