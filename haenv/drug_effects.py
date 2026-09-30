"""drug_effects.py -- the single read point for `registry/drug_effects.yaml`:
treatment -> effect size on indicator levels, consumed by
`build.render_clinical`, so that adherence changes move lab values.

Rules: `weight` (which decides `outcome_label`) is never touched; the registry
stores the trial's total effect and trial weight change, and the direct term
is derived on the fly (`direct_effect`) because rendering already adds the
weight-mediated part; a drug whose direct term cannot be decomposed has no
effect. The decomposition uses this repo's own `per_kg` coefficient, not a
mediation analysis.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml

import functools
import pathlib
from statistics import NormalDist
from typing import Any

from .rng import unit as _unit

# Resource root: the repo root in a source checkout, `haenv/_data` in a wheel.
from haenv import data_root as _data_root
from haenv.regpath import registry_path as _rp
ROOT = _data_root()
REGISTRY = _rp("drug_effects.yaml")

#: Signals a drug effect must never be applied to (they carry gold); must match
#: the registry's `forbidden_signals`.
FORBIDDEN = ("weight", "weight_ref")

#: Signal -> registry field with the trial's total effect; other signals get 0.0.
EFFECT_FIELDS = {"HbA1c": "total_hba1c_pp", "fasting_glucose": "total_fpg_mmol"}

#: First-order half-times (days) after a step in glycaemia (drive -> fasting
#: glucose -> HbA1c). Tahara Y, Shima K. Diabetes Care 1995;18(4):440-7,
#: PMID 7497851: fasting glucose 6.3 +/- 2.4 days, HbA1c 34.6 +/- 10.1 days.
HALF_TIME_DAYS = {"glucose": 6.3, "HbA1c": 34.6}

#: The driver whose response multiplier comes from the non-response band.
LOW_RESPONSE_DRIVER = "biological_low_response"


class DrugEffectsError(ValueError):
    """Registry validation failure, or a drug effect applied to a forbidden signal."""


def _doc() -> dict[str, Any]:
    """The validated table, with world-plugin `register_effect` entries merged in.

    The cache is keyed on the path and on `world_plugins.overlay_token`, so a registration, a
    reset or a restored overlay is seen on the next read.
    """
    from . import world_plugins as _wp
    return _doc_at(str(REGISTRY), _wp.overlay_token(_wp.TARGETS["effect"][0]))


def check_entry(name: str, spec: Any) -> None:
    """Field checks for one drug entry (in-repo or plugin); raises `DrugEffectsError`."""
    if not isinstance(spec, dict):
        raise DrugEffectsError(f"{name}: entry must be a mapping")
    for f in ("total_hba1c_pp", "trial_weight_kg", "readout_weeks",
              "background", "source", "review"):
        if f not in spec:
            raise DrugEffectsError(f"{name} is missing `{f}:`")
    if spec["background"] not in ("monotherapy_vs_placebo", "add_on_to_metformin"):
        raise DrugEffectsError(
            f"{name}.background={spec['background']!r} is not an allowed value; "
            "state whether the trial is monotherapy vs placebo or add-on to metformin")
    for f in ("total_fpg_mmol", "sd_change_hba1c_pp"):
        if f not in spec:
            raise DrugEffectsError(f"{name} is missing `{f}:` (null is allowed, absence is not)")


@functools.lru_cache(maxsize=1)
def _doc_at(path: str, _overlay: tuple) -> dict[str, Any]:
    """`path` is `str(REGISTRY)`; `_overlay` only keys the cache."""
    from . import world_plugins as _wp
    reg = pathlib.Path(path)
    table = _wp.TARGETS["effect"][0]
    if not reg.is_file():
        raise DrugEffectsError(f"{reg} is missing: drug effects have no source; there is no fallback to 'no effect'.")
    # Same merge step as `regpath.load_registry`; the path stays `REGISTRY` so it can be redirected.
    d = _wp.apply_overlay(table, _cached_yaml(reg) or {})
    if not isinstance(d.get("drugs"), dict) or not d["drugs"]:
        raise DrugEffectsError(f"{reg}: `drugs:` is missing or empty")
    for name, spec in d["drugs"].items():
        try:
            check_entry(name, spec)
        except DrugEffectsError as e:
            src = _wp.source_of(table, "drugs", name)
            raise DrugEffectsError(f"{reg if src is None else f'world plugin {src}'}: {e}") from None
    r = d.get("response") or {}
    lo, hi = (r.get("low_response_range") or (None, None))
    blo, bhi = (r.get("responder_bounds") or (None, None))
    if None in (lo, hi, blo, bhi) or not (0.0 <= lo < hi < blo < bhi):
        raise DrugEffectsError(
            f"{reg}: `response:` requires 0 <= low_response_range < responder_bounds with no overlap"
            f" (got {lo},{hi} / {blo},{bhi}); overlapping ranges cannot separate non-response from poor adherence")
    # The yaml also carries its own ban list, which must match code's `FORBIDDEN`.
    fy = tuple(d.get("forbidden_signals") or ())
    if set(fy) != set(FORBIDDEN):
        raise DrugEffectsError(
            f"{reg}: `forbidden_signals` {fy} does not match FORBIDDEN in code {FORBIDDEN}; "
            "the code is authoritative, update the registry to match.")
    return d


_doc.cache_clear = _doc_at.cache_clear  # type: ignore[attr-defined]


def drugs() -> dict[str, dict[str, Any]]:
    """The full registry."""
    return dict(_doc()["drugs"])


def _resolve(drug: str) -> dict[str, Any] | None:
    """Registry entry whose name is the longest prefix of the drug (prefix matching as in
    `latent.drug_pkpd`), or `None` (no effect; surfaced by `missing_drugs()`). Longest wins so a
    plugin entry such as `semaglutide_oral` is not shadowed by `semaglutide`."""
    d = str(drug or "")
    best: tuple[str, dict[str, Any]] | None = None
    for pre, spec in drugs().items():
        if d.startswith(pre) and (best is None or len(pre) > len(best[0])):
            best = (pre, spec)
    return None if best is None else best[1]


def total_effect(drug: str, dose_mg: float | None = None,
                 signal: str = "HbA1c") -> tuple[float, float] | None:
    """`(total effect on `signal`, trial weight change in kg)`, or `None`.

    With a `dose_ladder`, uses the highest rung not exceeding `dose_mg` (lowest
    rung if unknown); never interpolates, since the effect saturates.
    """
    spec = _resolve(drug)
    if spec is None:
        return None
    field = EFFECT_FIELDS.get(signal)
    if field is None:
        return None
    tot, w = spec.get(field), spec.get("trial_weight_kg")
    ladder = spec.get("dose_ladder") or {}
    if ladder and dose_mg is not None:
        keys = sorted(float(k) for k in ladder)
        pick = [k for k in keys if k <= float(dose_mg) + 1e-9]
        k = pick[-1] if pick else keys[0]
        e = ladder[k] if k in ladder else ladder[list(ladder)[0]]
        tot, w = e.get(field), e["trial_weight_kg"]
    if tot is None or w is None:
        return None                     # cannot decompose direct term => don't wire
                                         # it in (see module docstring)
    return float(tot), float(w)


def direct_effect(drug: str, per_kg: float, atten: float,
                  dose_mg: float | None = None, signal: str = "HbA1c") -> float | None:
    """Direct effect after removing weight mediation, or `None`:

        direct = total - (per_kg x atten) x dw_trial

    `per_kg` / `atten` come from the caller (`build.CLINICAL_SPEC` /
    `build.CLINICAL_ATTEN`).
    """
    got = total_effect(drug, dose_mg, signal)
    if got is None:
        return None
    tot, dw = got
    return tot - float(per_kg) * float(atten) * dw


def response_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    """`(low_response_range, responder_bounds)` from the registry."""
    r = _doc()["response"]
    return tuple(r["low_response_range"]), tuple(r["responder_bounds"])


def check_declared_response(value: float, driver: str) -> str | None:
    """Why a declared `latent.drug_response` contradicts the driver, or `None`
    (`biological_low_response` must be in the non-response band, other drivers in
    the responder band)."""
    (lo, hi), (blo, bhi) = response_bounds()
    v = float(value)
    if driver == LOW_RESPONSE_DRIVER:
        return None if lo <= v <= hi else (
            f"drug_response={v} conflicts with driver={driver}: non-response must be in [{lo}, {hi}]")
    return None if blo <= v <= bhi else (
        f"drug_response={v} conflicts with driver={driver}: other drivers must be in [{blo}, {bhi}]")


def response_for(case_id: str, driver: str | None, drug: str = "",
                 declared: float | None = None) -> float:
    """This person's response multiplier on the drug effect.

    * `declared` (from `latent.drug_response`) wins, after the driver check;
    * `biological_low_response` -> uniform over the non-response band;
    * otherwise normal around 1 with the drug's relative SD
      (`sd_change_hba1c_pp / |total_hba1c_pp|`), truncated to the responder band.
    """
    if declared is not None:
        why = check_declared_response(declared, str(driver or ""))
        if why:
            raise DrugEffectsError(why)
        return float(declared)
    (lo, hi), (blo, bhi) = response_bounds()
    u = _unit(case_id, "drug_response")
    if driver == LOW_RESPONSE_DRIVER:
        return lo + u * (hi - lo)
    spec = _resolve(drug) or {}
    tot, sd = spec.get("total_hba1c_pp"), spec.get("sd_change_hba1c_pp")
    if not tot or not sd:
        return 1.0                      # no spread registered => the trial mean, unchanged
    nd = NormalDist(1.0, abs(float(sd) / float(tot)))
    a, b = nd.cdf(blo), nd.cdf(bhi)
    return nd.inv_cdf(a + u * (b - a))


def applies_to(cohort: str | None) -> bool:
    """Whether this cohort may receive a drug effect (unknown => no). Trial
    effects were measured on diabetic populations and are not rescaled."""
    allow = _doc().get("applies_to_cohorts_default") or []
    return str(cohort) in {str(x) for x in allow}


def _kinetic_fraction(signal: str, day: int, drive) -> float:
    """Fraction of the full effect reached on `day` for daily drive `drive(d)`
    in [0, 1], via two first-order stages (`HALF_TIME_DAYS`); fasting glucose
    reads the first stage, HbA1c the second."""
    kg = 1.0 - 0.5 ** (1.0 / HALF_TIME_DAYS["glucose"])
    ka = 1.0 - 0.5 ** (1.0 / HALF_TIME_DAYS["HbA1c"])
    g = a = 0.0
    for d in range(1, int(day) + 1):
        g += kg * (drive(d) - g)
        a += ka * (g - a)
    return g if signal == "fasting_glucose" else a


def effect_at(drug: str, signal: str, day: int, adherence,
              per_kg: float, atten: float, dose_mg: float | None = None,
              onset_days: int = 7, cohort: str | None = None,
              response: float = 1.0) -> float:
    """Drug-effect increment on `signal` on `day`:
    `direct x _kinetic_fraction(day) x response`.

    `adherence` is a callable `day -> [0, 1]` or a constant; the drive is 0
    before `onset_days`. Only `EFFECT_FIELDS` signals get an effect.
    """
    if signal in FORBIDDEN:
        raise DrugEffectsError(
            f"{signal!r} is on the forbidden list: it carries the gold label (`label_rule` reads it) "
            "and drug effects may not change it.")
    if signal not in EFFECT_FIELDS:
        return 0.0
    if not applies_to(cohort):
        return 0.0                      # cohort not in trial population => not
                                         # applied (see `applies_to`)
    d = direct_effect(drug, per_kg, atten, dose_mg, signal)
    if d is None:
        return 0.0
    adh = adherence if callable(adherence) else (lambda _d, _v=float(adherence): _v)

    def drive(dd: int) -> float:
        return 0.0 if dd <= onset_days else max(0.0, min(1.0, float(adh(dd))))

    return d * _kinetic_fraction(signal, day, drive) * float(response)


def missing_drugs(seen) -> list[str]:
    """Drugs seen without a registered effect size."""
    return sorted({str(x) for x in seen if _resolve(str(x)) is None})


def upstream_gap() -> dict[str, Any]:
    """Drugs absent from the kernel's `DRUG_PKPD` (recorded in the registry)."""
    return dict(_doc().get("upstream_gap") or {})
