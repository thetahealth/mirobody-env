"""indicators.py -- the single read point for indicator dossiers (`registry/indicators.yaml`).

Answers questions about an indicator: unit, precision, reference range,
diagnostic role, per-cohort distribution, and baseline sampling. Design notes:
`docs/design/indicator-dossier.md`.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from .regpath import overlaid_yaml as _overlaid_yaml
from .world_plugins import overlay_cached as _overlay_cached
from .yamlcache import load_yaml as _cached_yaml

import functools
import math
import pathlib
from collections.abc import Mapping
from typing import Any
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data

#: The registry table, resolved through `haenv.regpath.registry_path` so it also works from an installed package.
from haenv.regpath import registry_path as _rp
_REG = _rp("indicators.yaml")


class IndicatorUnregistered(KeyError):
    """This indicator has no dossier (never a silent default)."""


class DossierError(ValueError):
    """The dossier itself is not fit for use (missing provenance, a cohort pointing at a disease that does not exist, etc.)."""


@_overlay_cached("indicators.yaml")
def dossiers() -> dict[str, dict[str, Any]]:
    """All dossiers, with structural validation only; value invariants are checked by the maintainers' tests."""
    import yaml

    doc = _overlaid_yaml(_REG) or {}
    ind = doc.get("indicators")
    if not isinstance(ind, dict) or not ind:
        raise DossierError(f"{_REG}: top level must contain a non-empty `indicators`")
    for name, d in ind.items():
        for k in ("unit", "ndigits", "reference_range", "payload_range",
                  "diagnostic_for", "cohorts"):
            if k not in d:
                raise DossierError(f"{name}: dossier is missing `{k}`")
        kind = d.get("kind", "clinical")
        if kind not in ("clinical", "monitoring", "panel"):
            raise DossierError(
                f"{name}: unknown kind={kind!r} (expected clinical / monitoring / panel)")
        # `cohorts` is required non-empty only for `clinical`: daily monitoring
        # streams belong to no cohort, and `panel` items come from the text lab report
        # (`findings_render.ROUTINE_PANEL`), not the time-series channel.
        if kind == "clinical" and not d["cohorts"]:
            raise DossierError(f"{name}: `cohorts` must not be empty for a clinical indicator")
        for coh, c in d["cohorts"].items():
            if not str(c.get("source") or "").strip():
                raise DossierError(f"{name}/{coh}: no source; every cohort "
                                   "entry needs a citation")
            if "p_abnormal" not in c:
                raise DossierError(f"{name}/{coh}: p_abnormal is missing; "
                                   "write null explicitly if it was not measured")
            if "median" not in c:
                raise DossierError(f"{name}/{coh}: median is missing")
            # Either (median, p_abnormal) or (mean, sd), as the literature reports it;
            # the derived value is written as null.
            if c.get("median") is None and not (c.get("mean") and c.get("sd")):
                raise DossierError(
                    f"{name}/{coh}: median is null, so (mean, sd) is required; "
                    "give one of the two parameterizations")
    return ind


def of(name: str) -> dict[str, Any]:
    """The full dossier for one indicator. Raises if not found -- see `IndicatorUnregistered`."""
    d = dossiers().get(str(name))
    if d is None:
        raise IndicatorUnregistered(
            f"{name!r} has no dossier in registry/indicators.yaml. "
            "Add one first: unit, precision, reference range, diagnostic role, per-cohort distribution, "
            "each with a source; there are no defaults.")
    return d


def cohort_of(name: str, disease: str, comorbidities=()) -> str | None:
    """Which cohort this case belongs to for `name`; `None` when no cohort matches.

    When several conditions match, a condition for which this indicator is a
    diagnostic basis (`diagnostic_for`) wins; otherwise the first in order
    (disease, then comorbidities). E.g. obesity + T2D takes the diabetic cohort for
    HbA1c.
    """
    d = of(name)
    cohorts = d.get("cohorts") or {}
    diag = d.get("diagnostic_for") or {}
    cands = [c for c in ([str(disease or "")] + [str(x) for x in (comorbidities or [])])
             if c in cohorts]
    if not cands:
        return None
    for c in cands:
        if c in diag:
            return c
    return cands[0]


def baseline(name: str, disease: str, comorbidities=()) -> float:
    """The baseline median for this case on `name`, from the matching cohort only; raises when none matches."""
    d = of(name)
    coh = cohort_of(name, disease, comorbidities)
    if coh is None:
        raise IndicatorUnregistered(
            f"{name}: dossier has no cohort for disease={disease!r} / comorbidities={list(comorbidities)!r} "
            f"(registered: {sorted((d.get('cohorts') or {}))}). "
            "Add the cohort with a source; there is no fallback to another cohort.")
    _m = d["cohorts"][coh].get("median")
    if _m is None:  # (mean, sd) parameterization => median derived from it
        _m, _ = moments_of(name, coh)
    return float(_m)


def pending_cohorts() -> list[str]:
    """Every `review: pending` (indicator, cohort) pair; their readings must not go into external materials."""
    out = []
    for n, d in dossiers().items():
        for coh, c in (d.get("cohorts") or {}).items():
            if str(c.get("review")) == "pending":
                out.append(f"{n}/{coh}")
    return sorted(out)


def known_defects() -> dict[str, str]:
    """Defects the dossier declares about itself, as `{indicator/cohort: description}`."""
    out = {}
    for n, d in dossiers().items():
        for coh, c in (d.get("cohorts") or {}).items():
            if c.get("known_defect"):
                out[f"{n}/{coh}"] = str(c["known_defect"]).strip()
    return out


# ════════════════════════════════════════════════ Distribution sampling
#
# A single median per cohort cannot express a mixed population: Ma X et al.
# (PMC6961232, 11 studies, 4084 cases) report that 25% of NAFLD patients have
# normal ALT.


class DistributionUnderdetermined(ValueError):
    """Median / abnormal rate / reference bound are incomplete, so no distribution can be determined."""


def _phi_inv(p: float) -> float:
    """Standard-normal quantile (`statistics.NormalDist`)."""
    import statistics as _st
    return _st.NormalDist().inv_cdf(p)


def moments_of(name: str, cohort: str) -> tuple[float, float]:
    """`(median, log-normal sigma)` for this cohort.

        (median, p_abnormal)  =>  sigma = ln(ref.high/median) / Phi^-1(1-p)
        (mean, sd)            =>  sigma^2 = ln(1+(sd/mean)^2), median = mean/exp(sigma^2/2)

    The second form does not depend on the reference bound, which matters when the
    source's abnormal rate used a different ULN (e.g. ALT/MASLD).
    """
    import math
    d = of(name)
    c = (d.get("cohorts") or {}).get(cohort)
    if c is None:
        raise IndicatorUnregistered(f"{name}: no cohort {cohort!r}")
    mean, sd = c.get("mean"), c.get("sd")
    if mean is not None and sd is not None:
        m_, s_ = float(mean), float(sd)
        if m_ <= 0 or s_ <= 0:
            raise DistributionUnderdetermined(f"{name}/{cohort}: mean/sd must be positive")
        sig2 = math.log(1.0 + (s_ / m_) ** 2)
        out = (m_ / math.exp(sig2 / 2.0), math.sqrt(sig2))
    elif c.get("sigma_ln") is not None and c.get("median") is not None:
        # (median, sigma_ln): the between-person log-spread is registered directly. The per-draw
        # measurement CV is added on top at emission, so `p_abnormal` (the over-line rate of one
        # reading in the source population) is a documented target here, not an input.
        s_ = float(c["sigma_ln"])
        if s_ <= 0:
            raise DistributionUnderdetermined(f"{name}/{cohort}: sigma_ln must be positive")
        out = (float(c["median"]), s_)
    else:
        out = (float(c["median"]), sigma_of(name, cohort))
    _assert_fits_payload(name, cohort, *out)
    return out


#: INV-7's quantile: p5/p95 (a log-normal has no finite extreme).
_FIT_Z = 1.645


def _assert_fits_payload(name: str, cohort: str, med: float, sig: float) -> None:
    """INV-7: the fitted distribution's p5/p95 must fall inside `payload_range`.

    `p_abnormal` from real EMR data reflects the issuing lab's reference range, not
    `findings.yaml`; when they disagree the solved sigma is absurd (e.g.
    fasting_glucose/T2D, median 8.2, p 0.642 => p95 46.0). This raises
    `DistributionUnderdetermined`, so sampling falls back to the median; the tail is
    never clamped.
    """
    pay = of(name).get("payload_range") or {}
    lo, hi = pay.get("low"), pay.get("high")
    p5, p95 = med * math.exp(-_FIT_Z * sig), med * math.exp(_FIT_Z * sig)
    bad = []
    if lo is not None and p5 < float(lo):
        bad.append(f"p5={p5:.2f} < payload lower bound {lo}")
    if hi is not None and p95 > float(hi):
        bad.append(f"p95={p95:.2f} > payload upper bound {hi}")
    if bad:
        raise DistributionUnderdetermined(
            f"{name}/{cohort}: fitted distribution does not fit the payload range ({'; '.join(bad)}, σ={sig:.3f}). "
            "A common cause is that `p_abnormal` comes from the source lab's own flags, "
            "which use that lab's reference range rather than this registry's. "
            "The tail is not clamped.")


#: INV-8's bandwidth; derivation in `unit_slip_of`.
_UNIT_SLIP_BAND = 10.0


def threshold_class_of(d: Mapping[str, Any]) -> dict[str, float]:
    """Like `unit_slip_of`, but takes a dossier dict (for synthetic negative controls)."""
    ref = d.get("reference_range") or {}
    out: dict[str, float] = {}
    for k in ("low", "high"):
        if ref.get(k):  # 0 and None are skipped: `ref.low = 0` is a recording convention (INV-3)
            out[f"reference_range.{k}"] = float(ref[k])
    for coh, spec in (d.get("cohorts") or {}).items():
        for k in ("median", "mean"):
            if (spec or {}).get(k):
                out[f"cohorts.{coh}.{k}"] = float(spec[k])
    for dis, spec in (d.get("diagnostic_for") or {}).items():
        for k in ("ge", "le", "gt", "lt"):
            if (spec or {}).get(k):
                out[f"diagnostic_for.{dis}.{k}"] = float(spec[k])
    return out


def violates_inv8(d: Mapping[str, Any]) -> float | None:
    """Whether a dossier dict violates INV-8: the over-threshold ratio, or `None` if compliant or undecidable.

    The reference range is the anchor (TSH's own range spans 15.6x): every other
    threshold-like value must fall within `[low/K, high*K]`.
    """
    got = _worst_deviation(d)
    return got[0] if got and got[0] > _UNIT_SLIP_BAND else None


def _worst_deviation(d: Mapping[str, Any]) -> tuple[float, dict[str, float]] | None:
    """Shared INV-8 computation for `unit_slip_of` and `violates_inv8`."""
    vals = threshold_class_of(d)
    lo, hi = vals.get("reference_range.low"), vals.get("reference_range.high")
    others = [v for k, v in vals.items()
              if not k.startswith("reference_range.") and v > 0]
    if not others or (lo is None and hi is None):
        return None
    worst = 0.0
    for v in others:
        if hi and v > hi:
            worst = max(worst, v / hi)
        elif lo and v < lo:
            worst = max(worst, lo / v)
    return worst, vals


def unit_slip_of(name: str) -> tuple[float, dict[str, float]] | None:
    """INV-8: the order-of-magnitude ratio between a dossier's threshold-like values; `None` with fewer than two positive values.

    Catches a value written in a different unit (e.g. mg/dL vs mmol/L). The
    bandwidth 10.0 sits between the largest legitimate ratio among dossiers (1.464,
    fasting_glucose) and the smallest slip to catch (glucose 18.0; cholesterol
    38.67, TG 88.5). The check is necessary but not sufficient: slips within one
    order of magnitude (e.g. eGFR vs creatinine clearance, both ml/min) pass.
    """
    return _worst_deviation(of(name))


def check_unit_consistency() -> dict[str, tuple[float, dict[str, float]]]:
    """Run INV-8 across the whole table; returns the violations (empty = pass)."""
    bad = {}
    for name in dossiers():
        got = unit_slip_of(name)
        if got is not None and got[0] > _UNIT_SLIP_BAND:
            bad[name] = got
    return bad


def sigma_of(name: str, cohort: str) -> float:
    """Log-normal sigma for this cohort, solved from the median and the abnormal rate.

        P(X > ref.high) = p  =>  sigma = ln(ref.high/median) / Phi^-1(1-p)

    For a lower-abnormal indicator the bound is `low` and the quantile Phi^-1(p).
    """
    d = of(name)
    c = (d.get("cohorts") or {}).get(cohort)
    if c is None:
        raise IndicatorUnregistered(f"{name}: no cohort {cohort!r}")
    ref = d.get("reference_range") or {}
    _lower = str(d.get("direction", "higher_abnormal")) == "lower_abnormal"
    hi = ref.get("low") if _lower else ref.get("high")
    med, p = c.get("median"), c.get("p_abnormal")
    missing = [k for k, v in (("reference_range.low" if _lower else "reference_range.high", hi),
                              ("median", med), ("p_abnormal", p)) if v is None]
    if missing:
        raise DistributionUnderdetermined(
            f"{name}/{cohort}: missing {missing} => distribution undetermined. "
            "No default spread is assumed.")
    p = float(p)
    if not (0.0 < p < 1.0):
        raise DistributionUnderdetermined(f"{name}/{cohort}: p_abnormal={p} must be in the open interval (0, 1)")
    z = _phi_inv(p if _lower else 1.0 - p)
    if abs(z) < 1e-9:  # p == 0.5: sigma is unidentifiable
        raise DistributionUnderdetermined(
            f"{name}/{cohort}: σ is unidentifiable at p_abnormal=0.5 (median equals the reference bound)")
    return abs(math.log(float(hi) / float(med)) / z)


def sample_baseline(name: str, disease: str, comorbidities=(), case_id: str = "") -> float:
    """This case's baseline on `name`: a deterministic draw from the cohort's distribution.

    Falls back to the median when no distribution can be determined (reported by
    `distribution_status()`).
    """
    import math

    from .rng import unit

    coh = cohort_of(name, disease, comorbidities)
    if coh is None:
        raise IndicatorUnregistered(
            f"{name}: dossier has no cohort for disease={disease!r} / comorbidities={list(comorbidities)!r}")
    try:
        med, sd = moments_of(name, coh)
    except DistributionUnderdetermined:
        _m = of(name)["cohorts"][coh].get("median")
        if _m is None:
            raise
        return float(_m)
    cap = (of(name)["cohorts"][coh]).get("cap")
    sev = severity().get(str(coh))
    # The shared severity factor is the registered correlation of its `pair` only; applying it to
    # every stream of the cohort would make unrelated streams (LDL, ALT, blood pressure ...) move
    # with the diabetes severity.
    lam = _loading_of(coh) if (sev is not None and name in [str(x) for x in sev["pair"]]) else None
    z_case = _gauss(str(case_id), "severity", coh) if lam is not None else 0.0
    val = med
    for k in range(32):
        z = (_gauss(str(case_id), "indicator", name, coh) if k == 0
             else _gauss(str(case_id), "indicator", name, coh, f"redraw{k}"))
        # One-factor shared severity: ln(b_i) = ln(median_i) + sigma_i * [lambda*z_case +
        # sqrt(1-lambda^2)*z_i], so marginals (and `p_abnormal`) are unchanged.
        # sigma is the between-person spread (`sigma_ln`, or solved from `p_abnormal` for
        # cohorts registered that way); the per-draw measurement CV (`physio_streams.yaml:clinical_measurement.cv`)
        # is applied at emission, on top of it. The EMR spread column is informational only.
        z_mix = z if lam is None else lam * z_case + math.sqrt(max(0.0, 1.0 - lam * lam)) * z
        val = med * math.exp(sd * z_mix)
        # `cap`: the referral line of a cohort without the indicator's disease; baselines at or above it
        # are redrawn (a benign cohort does not start at a referral-level reading).
        if cap is None or val < float(cap):
            return val
    return med


def _gauss(*path: str) -> float:
    """Box-Muller standard normal clipped to +-3 (as in the noise layer)."""
    from .rng import unit
    u1 = max(1e-12, unit(*path, "bm0"))
    u2 = unit(*path, "bm1")
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return min(3.0, max(-3.0, z))


@_overlay_cached("indicators.yaml")
def severity() -> dict[str, dict[str, Any]]:
    """Shared-severity registry per cohort (cached; on the sampling hot path). Absent = never measured, not 0."""
    doc = _overlaid_yaml(_REG) or {}
    return dict(doc.get("severity") or {})


def _loading_of(cohort: str) -> float | None:
    """Shared loading lambda for this cohort on the log scale; `None` if unregistered.

    The registered `between_patient_r` is a raw-scale Pearson r, so the log-scale
    rho is solved from the log-normal attenuation

        r_raw = (e^{rho*sigma1*sigma2} - 1) / sqrt((e^{sigma1^2} - 1)(e^{sigma2^2} - 1))

    using sigma_total; lambda therefore follows sigma and is not stored.
    """
    spec = severity().get(str(cohort))
    if spec is None:
        return None
    a, b = [str(x) for x in spec["pair"]]
    try:
        _, s1 = moments_of(a, cohort)
        _, s2 = moments_of(b, cohort)
    except DistributionUnderdetermined:
        return None  # no distribution => no correlation
    den = math.sqrt((math.exp(s1 * s1) - 1.0) * (math.exp(s2 * s2) - 1.0))
    val = 1.0 + float(spec["between_patient_r"]) * den
    if val <= 0.0 or s1 <= 0 or s2 <= 0:
        return None
    rho = math.log(val) / (s1 * s2)
    return math.sqrt(max(0.0, min(1.0, rho)))


def distribution_status() -> dict[str, str]:
    """Whether each (indicator/cohort) samples from a distribution or from the median only."""
    out = {}
    for n, d in dossiers().items():
        for coh in (d.get("cohorts") or {}):
            try:
                _m, _s = moments_of(n, coh)
                out[f"{n}/{coh}"] = f"分布 中位={_m:.2f} σ={_s:.4f}"
            except DistributionUnderdetermined as e:
                out[f"{n}/{coh}"] = f"仅中位数({str(e).split('missing ')[-1].split(' =>')[0]})"
    return out


# ════════════════════════════════════════════════ Umbrella diagnoses

@_overlay_cached("indicators.yaml")
def diagnoses() -> dict[str, dict[str, Any]]:
    """The `diagnoses` block: markers that jointly define each disease.

    Umbrella diagnoses such as dyslipidemia have no single numeric definition
    (ACC/AHA), so the constraint is `any_of`: at least one constituent marker out
    of range.
    """
    doc = _overlaid_yaml(_REG) or {}
    return doc.get("diagnoses") or {}


def abnormal_side(name: str, value: float) -> bool | None:
    """Whether `value` is on the abnormal side for `name`; `None` if the relevant bound is unregistered.

    Direction defaults to `higher_abnormal`; e.g. `CGM_TIR` records
    `direction: lower_abnormal` (abnormal below 70, Diabetes Care 2019;42:1593).
    """
    d = of(name)
    ref = d.get("reference_range") or {}
    if str(d.get("direction", "higher_abnormal")) == "lower_abnormal":
        lo = ref.get("low")
        return None if lo is None else float(value) < float(lo)
    hi = ref.get("high")
    return None if hi is None else float(value) > float(hi)


class UmbrellaUnsatisfiable(RuntimeError):
    """The umbrella diagnosis's "at least one abnormal" condition was not met within the resample budget (the cohort parameters disagree with the diagnostic thresholds)."""


@_overlay_cached("indicators.yaml")
def pair_constraints() -> dict[str, dict[str, Any]]:
    """The pair-constraints block. Answers "is this pair possible", not "is it normal"."""
    doc = _overlaid_yaml(_REG) or {}
    return doc.get("pair_constraints") or {}


def _pairs_ok(disease: str, vals: dict[str, float]) -> bool:
    """Whether the whole sampled set satisfies every applicable pair constraint."""
    for _nm, c in pair_constraints().items():
        if str(disease) not in (c.get("applies_to_cohorts") or []):
            continue
        num, den = c["ratio_of"]
        if num not in vals or den not in vals or vals[den] <= 0:
            continue
        r = vals[num] / vals[den]
        if not (float(c["low"]) <= r <= float(c["high"])):
            return False
    return True


def sample_case(disease: str, signals, comorbidities=(), case_id: str = "",
                max_draws: int = 64) -> dict[str, float]:
    """Baselines for one patient across `signals`, conditioned on the umbrella diagnosis.

    Drawn jointly and resampled until `diagnoses[disease].any_of` holds (and pair
    constraints pass); independent draws would leave many "diagnosed" patients
    with no abnormal marker. Without an umbrella diagnosis this matches
    `sample_baseline` value for value.
    """
    names = [str(x) for x in signals]
    spec = (diagnoses() or {}).get(str(disease)) or {}
    need = [n for n in (spec.get("any_of") or []) if n in names]
    # Only markers with a registered bound can be judged abnormal.
    need = [n for n in need if (of(n).get("reference_range") or {}).get("high") is not None]

    for k in range(max_draws):
        tag = str(case_id) if k == 0 else f"{case_id}#draw{k}"
        out = {n: sample_baseline(n, disease, comorbidities, case_id=tag) for n in names}
        if not _pairs_ok(disease, out):
            continue
        if not need:
            return out
        if any(abnormal_side(n, out[n]) for n in need):
            return out
    raise UmbrellaUnsatisfiable(
        f"{disease}: no draw in {max_draws} rounds satisfied both "
        f"'at least one of {need or '(none)'} abnormal' and the pairwise constraints. "
        "Adjust those indicators' distribution parameters in the registry")


def _declared_ndigits(name: str) -> int | None:
    """A stream's declared decimal precision from the indicator registry, or
    `None` (unregistered, or `ndigits: null` for panel items) to leave it as is."""
    from . import indicators as _ind
    try:
        nd = _ind.of(name)["ndigits"]
    except _ind.IndicatorUnregistered:
        return None
    return None if nd is None else int(nd)
