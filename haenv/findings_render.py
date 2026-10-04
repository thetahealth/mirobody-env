"""findings_render.py -- renders declared findings (`registry/findings.yaml` +
`registry/condition_findings.yaml`) into point-in-time `lab_result` entries in
`RawCase.evidence_ledger`, where they pass through the existing leakage probes and
per-item validation.

Value generation reads only `case_id` and the declaration, never the diagnosis. The
caller (`events`) renders findings only when the job sets `findings`. Only
`role: screening` items render; confirmatory items are the answer to "what should be
ordered next".

`trajectory` values: `stable`, `progressive` (walks up to the declared magnitude),
`fluctuating`, `episodic` (at least one reading is in range), `treatment_responsive`
(falls back late in the course).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import logging
import math

from . import rng
from .relations import (FRIEDEWALD_TG_DIVISOR, FRIEDEWALD_TG_MAX_MMOL,
                        PAYNE_ANCHOR_ORIGINAL_G_PER_L, PAYNE_COEF_PER_G_PER_L,
                        check_tsh_ft4, payne_corrected_ca)

log = logging.getLogger("haenv.findings_render")

# Magnitude tier -> multiple of the reference-range width by which the value
# clears the boundary.
#: `mild` starts at the boundary (P0-1, 2026-09-30): real abnormal readings sit
#: just past the bound 7.4% of the time (within 0.05 band widths), the old floor of
#: 0.05 made that 0.4%. A designed abnormality still clears the bound by at least
#: one printed digit (`_value`), so it cannot round back onto the boundary.
MAG_FACTOR = {"mild": (0.0, 0.25), "moderate": (0.3, 0.7), "marked": (0.9, 1.6)}
QUALITATIVE_DIRECTIONS = {"positive", "negative"}


def _band(spec: dict) -> tuple[float, float]:
    ref = spec.get("ref") or {}
    lo, hi = float(ref.get("low", 0.0)), float(ref.get("high", 1.0))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


#: Clinical-tier floor for `high` abnormal values: magnitude tiers map onto
#: absolute thresholds, since a range-relative offset alone can land below them.
#: Sources: ADA (A1C: prediabetes 5.7, diabetes 6.5, poor control >8%);
#: WHO/ADA (fasting glucose: IFG 6.1, diabetes 7.0).
SEVERITY_FLOOR: dict[str, dict[str, float]] = {
    "HbA1c": {"mild": 5.7, "moderate": 6.5, "marked": 8.0},
    "FBG": {"mild": 6.1, "moderate": 7.0, "marked": 10.0},
}


#: Positivity floor for the `low` direction, as a fraction of the reference lower
#: bound. Prevents impossible values; it is not a clinical low-value tiering.
LOW_POSITIVITY_FLOOR_FRAC = 0.05

_MAGS = ("mild", "moderate", "marked")


def _progressive_mag(declared: str | None, k: int, n: int) -> str | None:
    """Magnitude of the k-th reading on a `progressive` trajectory, counting back
    from the declared magnitude as the endpoint (`marked`, n=2 -> moderate, marked).
    """
    if n <= 1 or declared is None:
        return declared
    try:
        idx = _MAGS.index(str(declared))
    except ValueError:
        return declared
    start = max(0, idx - (n - 1))
    ladder = list(_MAGS[start:idx + 1])
    while len(ladder) < n:
        ladder.insert(0, ladder[0])
    return ladder[min(k, len(ladder) - 1)]


#: Panel item -> dossier name in `registry/indicators.yaml`, where the two differ.
_DOSSIER_NAME: dict[str, str] = {"FBG": "fasting_glucose", "TG": "triglycerides"}


def _ndigits(fid: str | None) -> int | None:
    """Declared precision of a panel item (`registry/indicators.yaml:ndigits`), or
    `None` when the item has no dossier or declares `null`."""
    if not fid:
        return None
    from . import indicators as _ind
    try:
        nd = _ind.of(_DOSSIER_NAME.get(fid, fid))["ndigits"]
    except _ind.IndicatorUnregistered:
        return None
    return None if nd is None else int(nd)


def _round(v: float, fid: str | None = None):
    """Rounds to the item's declared precision (`indicators.yaml`); items without a
    dossier keep the magnitude rule: four decimals below 0.1, two above.
    `ndigits: 0` returns an int, so the printed value has no decimal point."""
    nd = _ndigits(fid)
    if nd is None:
        return round(v, 4 if abs(v) < 0.1 else 2)
    return int(round(v)) if nd == 0 else round(v, nd)


def _step(fid: str | None, v: float) -> float:
    """One printed digit of this item at value `v`."""
    nd = _ndigits(fid)
    if nd is None:
        nd = 4 if abs(v) < 0.1 else 2
    return 10.0 ** (-nd)


def _fmt(fid: str | None, v) -> str:
    """Printed value: fixed width at the declared precision, so 4.20 prints as
    `4.20` the way a lab report does, not `4.2`."""
    if not isinstance(v, (int, float)):
        return str(v)
    nd = _ndigits(fid)
    if nd is None:
        return str(v)
    return f"{v:.{nd}f}"


#: Decision lines are not reference intervals. LDL 3.37, TG 1.7, TC 5.17, HDL 1.04, HbA1c 5.6,
#: FBG 6.1, ALT 25/33, CRP 10, Ferritin 30-200 and VitD 30 are action thresholds that the
#: general population crosses far more often than 5%. Those items take their not-diseased
#: readings from `registry/panel_background.yaml` (NHANES aggregate quantile tables, cut
#: below the referral line; a generated table), not from the
#: band-fitted log-normal below. Items whose band is a true 95% interval keep the model below.
#:
#: A reference interval is the central 95% of a healthy reference population (2.5th
#: to 97.5th percentile; CLSI EP28-A3c), so a healthy reading falls outside it 5% of
#: the time, 2.5% on each side of a two-sided band, and mostly just outside: the
#: log-normal fitted to the band puts 0.3-0.4% of readings beyond 0.27 band widths
#: (0.7-1.1% on one-sided bands). This is the only source of incidental abnormalities on
#: benign and insufficient-tier panels. One-sided
#: bands (`low` = 0) are log-normal with 5% above the upper bound; `sigma` sets the
#: skew (median = hi * exp(-1.645 * sigma)).
NORMAL_OUT_OF_BAND = 0.05
_ONE_SIDED_LOG_SIGMA = 0.30


def _normal_value(lo: float, hi: float, u: float, spill: bool) -> float:
    """A normal-direction reading from quantile `u`, covering the whole band.

    `spill=True` lets `NORMAL_OUT_OF_BAND` of readings fall just outside the band
    (background abnormality of a healthy item); `spill=False` truncates to the band
    (a declared-normal or pre-onset disease item must read in range).
    """
    if not spill:
        u = (NORMAL_OUT_OF_BAND / 2 + u * (1 - NORMAL_OUT_OF_BAND)) if lo > 0 \
            else u * (1 - NORMAL_OUT_OF_BAND)
    u = min(1 - 1e-9, max(1e-9, u))
    z = _ppf(u)
    if lo > 0:
        m = 0.5 * (math.log(lo) + math.log(hi))
        s = (math.log(hi) - math.log(lo)) / (2 * 1.959964)
        v = math.exp(m + s * z)
    else:
        med = hi * math.exp(-1.644854 * _ONE_SIDED_LOG_SIGMA)
        v = med * math.exp(_ONE_SIDED_LOG_SIGMA * z)
    if not spill:
        v = min(hi, max(lo, v))
    return v


@functools.lru_cache(maxsize=1)
def _background_tables() -> dict[str, dict[str, tuple[float, ...]]]:
    """General-population quantile tables of the decision-line items (`registry/panel_background.yaml`,
    `adopt: true`): item -> sex (`F` / `M` / `all`) -> 101 quantiles p0..p100."""
    from .regpath import load_registry
    doc = load_registry("panel_background.yaml") or {}
    out: dict[str, dict[str, tuple[float, ...]]] = {}
    for fid, e in (doc.get("background") or {}).items():
        if not (e or {}).get("adopt"):
            continue
        out[fid] = {sx: tuple(float(x) for x in e[sx]["q"]) for sx in ("F", "M", "all") if sx in e}
    return out


def _background_value(fid: str | None, sex: str | None, u: float) -> float | None:
    """Reading of a not-diseased person from the item's general-population table at quantile `u`;
    `None` when the item is not tabulated (a true 95% reference interval keeps `_normal_value`)."""
    t = _background_tables().get(fid or "")
    if not t:
        return None
    q = t.get(sex or "all") or t["all"]
    x = min(max(u, 0.0), 1.0) * (len(q) - 1)
    i = min(int(x), len(q) - 2)
    return q[i] + (x - i) * (q[i + 1] - q[i])


def _background_masses(fid: str | None, sex: str | None, lo: float, hi: float) -> tuple[float, float]:
    """(below, above) probability mass outside the printed band of a not-diseased reading of `fid`."""
    e = _background_info().get(fid or "")
    if e:
        return e.get(sex or "all") or e["all"]
    return (0.025, 0.025) if lo > 0 else (0.0, NORMAL_OUT_OF_BAND)


@functools.lru_cache(maxsize=1)
def _background_info() -> dict[str, dict[str, tuple[float, float]]]:
    from .regpath import load_registry
    doc = load_registry("panel_background.yaml") or {}
    return {fid: {sx: (float(e[sx]["out_of_band"]["below"]), float(e[sx]["out_of_band"]["above"]))
                  for sx in ("F", "M", "all") if sx in e}
            for fid, e in (doc.get("background") or {}).items() if (e or {}).get("adopt")}


def _tail_quantile(u: float, below: float, above: float) -> float:
    """Map u in [0, 1) onto the out-of-band part of the background distribution."""
    t = u * (below + above)
    return t if t < below else 1.0 - above + (t - below)


def _value(spec: dict, direction: str, magnitude: str | None,
           case_id: str, fid: str, k: int, declared: bool = True,
           spill: bool = False, tail: bool = False) -> float:
    """The value for one reading; reads only case_id, field name and reading index.

    Undeclared items (`declared=False`) are mixed with their shared cause
    (`FACTOR_LOADING`); declared items already are the shared cause. `spill`
    applies to normal readings only (see `_normal_value`).
    """
    lo, hi = _band(spec)
    width = hi - lo
    u = rng.unit(case_id, "finding", fid, str(k))
    if not declared:
        if direction == "normal" and spill and fid in _copula_index():
            u = _copula_u(case_id, fid, k, spec.get("bg_sex"))
        else:
            u = _shared_u(case_id, fid, u)
    if direction == "normal":
        if spill and tail:
            u = _tail_quantile(u, *_background_masses(fid, spec.get("bg_sex"), lo, hi))
        if spill:
            _bg = _background_value(fid, spec.get("bg_sex"), u)
            if _bg is not None:
                return _round(_bg, fid)
        return _round(_normal_value(lo, hi, u, spill), fid)
    f_lo, f_hi = MAG_FACTOR.get(str(magnitude or "moderate"), MAG_FACTOR["moderate"])
    off = width * (f_lo + u * (f_hi - f_lo))
    val = (hi + off) if direction == "high" else (lo - off)
    # Only `high` has registered clinical floors.
    if direction == "high":
        _fl = (SEVERITY_FLOOR.get(fid) or {}).get(str(magnitude or "moderate"))
        if _fl is not None and val < _fl:
            # jitter keeps the floor from becoming one constant
            val = _fl + u * max(0.05 * _fl, 0.1)
    elif direction == "low":
        # Lab values never go negative: floor at 5% of the lower bound, or 1%
        # of the range width when the lower bound is 0.
        _floor = lo * LOW_POSITIVITY_FLOOR_FRAC if lo > 0 else width * 0.01
        if val < _floor:
            val = _floor * (1.0 + 0.6 * u)
        val = max(val, _floor)
    out = _round(val, fid)
    # An abnormal reading clears its bound by at least one printed digit.
    if direction == "high" and out <= hi:
        out = _round(hi + _step(fid, hi), fid)
    elif direction == "low" and out >= lo and lo > 0:
        out = _round(lo - _step(fid, lo), fid)
    return out


# ---------------------------------------------------------------- Shared cause
#
# Items driven by one physiological cause (metabolic state: TC/TG up, HDL down;
# hepatocyte injury: ALT/AST) are coupled through a Gaussian copula. For loadings
# a1, a2 the latent correlation is rho = a1*a2, and the uniform-space correlation
# is r_u = (6/pi) * arcsin(rho/2); the loadings below invert that for target r_u
# values calibrated on real EMR data. Only undeclared items are mixed.
FACTOR_LOADING: dict[str, tuple[str, float]] = {
    # fid: (shared-cause name, loading; negative = inverse). Targets are in
    # u-space, not the delivered value-space correlation.
    "TC":  ("metabolic", +0.662),  # u-space target r_u(TC,TG)  = +0.422 => rho=0.438 => a=sqrt(rho)
    "TG":  ("metabolic", +0.662),
    "HDL": ("metabolic", -0.433),  # u-space target r_u(HDL,TG) = -0.275 => rho=-0.287 => a=rho/a_TG
    "ALT": ("hepatic",   +0.889),  # u-space target r_u(ALT,AST) = +0.777 => rho=0.791
    "AST": ("hepatic",   +0.889),
    # Ca/Alb is coupled by the `R3-mech` derivation instead of a loading.
}


# ---------------------------------------------------------------- Benign cross-item copula
#
# A benign reading's quantile is not drawn alone: the items of one panel move together as they
# do in the general population (NHANES normal-score correlation, `registry/panel_copula.yaml`,
# a generated table). `z = L e`, `e_j = Phi^-1(u_j)` with `u_j` a draw of
# its own per item and reading index (not the item's `finding` draw, so a declared reading never
# steers a benign one) and `L` the Cholesky factor, so every `u` stays exactly uniform and the
# marginals of `panel_background.yaml` are kept. Without it the count of
# out-of-band items per benign panel is too narrow (variance 2.8 against 4.1 in NHANES).


@functools.lru_cache(maxsize=1)
def _copula() -> tuple[tuple[str, ...], dict[str, list[list[float]]]]:
    """(items, sex -> lower Cholesky factor) of the benign copula; empty when not registered."""
    from .regpath import load_registry
    doc = load_registry("panel_copula.yaml") or {}
    items = tuple(str(x) for x in (doc.get("items") or ()))
    out: dict[str, list[list[float]]] = {}
    for sx, e in (doc.get("copula") or {}).items():
        c = [[float(x) for x in row] for row in e["corr"]]
        n = len(c)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                acc = c[i][j] - sum(L[i][m] * L[j][m] for m in range(j))
                L[i][j] = math.sqrt(max(acc, 1e-12)) if i == j else acc / L[j][j]
        out[str(sx)] = L
    return items, out


@functools.lru_cache(maxsize=1)
def _copula_index() -> dict[str, int]:
    return {f: i for i, f in enumerate(_copula()[0])}


def _copula_u(case_id: str, fid: str, k: int, sex: str | None) -> float:
    """The copula-coupled quantile of benign reading `k` of `fid` (see the section comment)."""
    items, Ls = _copula()
    L = Ls.get(sex or "") or Ls.get("F") or next(iter(Ls.values()))
    i = _copula_index()[fid]
    z = sum(L[i][j] * _ppf(rng.unit(case_id, "bgcopula", items[j], str(k))) for j in range(i + 1))
    return min(1.0 - 1e-12, max(1e-12, _cdf(z)))


def _cluster_members(name: str) -> list[tuple[str, float]]:
    return sorted(((f, a) for f, (n, a) in FACTOR_LOADING.items() if n == name),
                  key=lambda x: x[0])


def _partners_of(fid: str) -> list[tuple[str, float]]:
    """All cluster partners of this item and their relative direction."""
    ent = FACTOR_LOADING.get(fid)
    if not ent:
        return []
    name, a = ent
    return [(f, (1.0 if a * b > 0 else -1.0))
            for f, b in _cluster_members(name) if f != fid]


def _disease_partner_direction(declared_dir: str, sgn: float) -> str | None:
    """Direction a shared-cause partner takes from a declared abnormality, or
    None when the partner is not pulled.

    An inverse partner follows a declared rise only. The inverse arm (HDL
    against TG/TC) is the atherogenic pattern; a condition that lowers TG or TC
    (hyperthyroidism) lowers lipoproteins without raising HDL, and TC low with
    HDL high leaves under 0.62 mmol/L for LDL + TG/2.2 under Friedewald.
    """
    if sgn > 0:
        return declared_dir
    return "low" if declared_dir == "high" else None


def _ppf(p: float) -> float:
    """Standard normal quantile (Acklam's approximation, abs. error < 1.15e-9)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _shared_u(case_id: str, fid: str, own_u: float) -> float:
    """Mix this item's sample with its shared cause's sample in normal space
    (`z = a*z_shared + sqrt(1-a^2)*z_own`, `u = Phi(z)`), which keeps `u` exactly
    uniform; a linear mix of uniforms would thin the extreme values.
    """
    ent = FACTOR_LOADING.get(fid)
    if not ent:
        return own_u
    name, a = ent
    shared = rng.unit(case_id, "commoncause", name)
    sgn = 1.0 if a >= 0 else -1.0
    a = abs(a)
    z = sgn * a * _ppf(shared) + math.sqrt(max(0.0, 1.0 - a * a)) * _ppf(own_u)
    return min(1.0, max(0.0, _cdf(z)))


def _timeline(window_end: int, n: int, case_id: str, fid: str) -> list[int]:
    """Draw days for n readings, in `[7, T-3]`; `window_end` must be the index
    time `T`, since an initial workup precedes the visit.
    """
    lo, hi = 7, max(10, int(window_end) - 3)
    if hi <= lo:
        return [lo]
    if n <= 1:
        return [lo + int(rng.unit(case_id, "fday", fid, "0") * (hi - lo))]
    out = []
    span = hi - lo
    for k in range(n):
        base = lo + span * (0.15 + 0.7 * k / max(n - 1, 1))
        jitter = (rng.unit(case_id, "fday", fid, str(k)) - 0.5) * 0.18 * span
        out.append(max(lo, min(hi, int(base + jitter))))
    return sorted(set(out)) or [lo]


def draw_days(case_id: str, window_end: int, k: int = 2,
              *, not_before: int | None = None) -> list[int]:
    """The k shared draw days for this case's routine panel (one blood draw per
    day, so cross-item identities compare values from the same day).

    `not_before` opens the window at presentation, so the diagnostic draw does
    not fall in the pre-onset stretch that `render_findings` blanks. It only
    shifts when k distinct days still fit, and consumes the same `rng` draws.
    """
    lo, hi = 7, max(10, int(window_end) - 3)
    if hi <= lo:
        return [lo] * max(1, k)
    if not_before is not None and int(not_before) > lo:
        _nb = int(not_before)
        if _nb <= hi - (max(1, k) - 1):
            lo = _nb
    span = hi - lo
    out = []
    for i in range(max(1, k)):
        base = lo + span * (0.15 + 0.7 * i / max(k - 1, 1))
        jitter = (rng.unit(case_id, "drawday", str(i)) - 0.5) * 0.18 * span
        out.append(max(lo, min(hi, int(base + jitter))))
    out = sorted(set(out))
    while len(out) < max(1, k):                  # collision => shift a day later
        out.append(min(hi, out[-1] + 1))
        out = sorted(set(out))
        if out[-1] >= hi:
            break
    return out


# ---------------------------------------------------------------- Initial-visit panel
#
# The same fixed set for every patient; the condition only decides which items
# are abnormal, so the item list carries no answer. Every item except the four
# orphans (`Hb`, `K`, `Cr`, `TSH`) is paired by a relation: De Ritis (AST/ALT),
# Friedewald (TC/HDL/TG/LDL, LDL = TC - HDL - TG/2.2 in mmol/L), Payne (Ca/Alb,
# BMJ 1973), the Na-Cl gap, ADAG (FBG/HbA1c). FT4 is excluded because several
# conditions list it as the confirmatory test.
ROUTINE_PANEL: tuple[tuple[str, int], ...] = (
    ("FBG", 2), ("HbA1c", 2), ("LDL", 1), ("TG", 1), ("ALT", 1),
    ("Cr", 1), ("Hb", 1), ("K", 1), ("Na", 1), ("Ca", 1), ("TSH", 2),
    ("AST", 1), ("TC", 1), ("HDL", 1), ("Alb", 1), ("Cl", 1),
)


#: Items paired by a relation (see above); always ordered, never sampled.
_PANEL_RELATION_ITEMS: frozenset[str] = frozenset({
    "LDL", "TC", "HDL", "TG", "Ca", "Alb", "Na", "Cl", "AST", "ALT", "FBG", "HbA1c",
})

#: Probability that an orphan item is ordered.
_PANEL_OPTIONAL_P = 0.5

#: Screening items outside `ROUTINE_PANEL` declared by at least two conditions. A case gets
#: the ones its condition declares, the ones a suspected condition would declare, and the rest
#: at `SPECIALTY_COVERAGE` (see `_specialty_items`).
_SPECIALTY_SCREEN: tuple[tuple[str, int], ...] = (
    ("ESR", 1), ("CRP", 1), ("ANA", 1), ("Ferritin", 1), ("Insulin", 1), ("VitD25", 1),
)

#: Share of real patients with the item ever measured: median of the three outpatient /
#: inpatient cohorts of the RC readout (diabetes, coronary disease, hypothyroidism;
#: `panel_item_patient_coverage`, ranges ESR 0.09-0.36, CRP 0.55-1.0, ANA 0.03-0.06, ferritin
#: 0.06-0.23, fasting insulin 0.18-1.0, 25-OH-D 0.16-0.99).
SPECIALTY_COVERAGE: dict[str, float] = {
    "ESR": 0.211, "CRP": 0.984, "ANA": 0.044, "Ferritin": 0.191, "Insulin": 0.564, "VitD25": 0.379,
}


@functools.lru_cache(maxsize=1)
def _specialty_catalog() -> tuple[tuple[frozenset, ...], dict[str, float]]:
    """(specialty items each catalog condition declares, for conditions that declare any
    finding; share of those conditions declaring each item)."""
    from .registry import condition_findings_for_case
    screen = {f for f, _ in _SPECIALTY_SCREEN}
    profs = tuple(frozenset({p["id"] for p in (spec or {}).get("findings") or ()} & screen)
                  for _sid, spec in sorted(condition_findings_for_case().items())
                  if (spec or {}).get("findings"))
    n = max(1, len(profs))
    return profs, {f: sum(f in p for p in profs) / n for f in screen}


def _specialty_items(case_id: str, prof: dict, inc: dict) -> list[tuple[str, int]]:
    """The specialty items this case gets. Declared and incidental items are always kept (the
    gold's evidence stays on the panel). A case whose condition declares nothing (benign) is
    worked up for a suspected condition, drawn uniformly from the catalog, so "declared =>
    ordered" holds for benign and diagnosed cases alike and an item's presence does not mark a
    diagnosis. Any other item is ordered with `q = (c - d) / (1 - d)`, `c` its real coverage and
    `d` the catalog share declaring it, so its overall coverage is about `max(c, d)`. The tier
    never enters."""
    profs, share = _specialty_catalog()
    if prof or not profs:
        suspected: frozenset = frozenset()
    else:
        suspected = profs[min(len(profs) - 1, int(rng.unit(case_id, "panel-suspect") * len(profs)))]
    keep: list[tuple[str, int]] = []
    for fid, n in _SPECIALTY_SCREEN:
        c, d = SPECIALTY_COVERAGE[fid], share.get(fid, 0.0)
        q = max(0.0, (c - d) / (1.0 - d)) if d < 1.0 else 1.0
        if fid in prof or fid in inc or fid in suspected or rng.unit(case_id, "panel-sp", fid) < q:
            keep.append((fid, n))
    return keep


#: Longitudinal stream name -> panel item. When the timeline produces a stream
#: live, the text panel omits that item, so one indicator never has two values
#: in the same question.
_TS_TO_PANEL: dict[str, str] = {
    "fasting_glucose": "FBG",
    "triglycerides": "TG",
    "ALT": "ALT", "AST": "AST", "HbA1c": "HbA1c", "LDL": "LDL",
}


def _panel_items_for(case_id: str, prof: dict, inc: dict,
                     ts_signals: object = ()) -> list[tuple[str, int]]:
    """Lab items this case produces, in `ROUTINE_PANEL` order. Declared,
    incidental and relation items are always kept; the orphans are sampled per
    case; the specialty group follows `_specialty_items`.
    """
    _dropped = {_TS_TO_PANEL[s] for s in (ts_signals or ()) if s in _TS_TO_PANEL}
    keep: list[tuple[str, int]] = []
    for fid, n in ROUTINE_PANEL:
        if fid in _dropped:
            continue
        if fid in prof or fid in inc or fid in _PANEL_RELATION_ITEMS:
            keep.append((fid, n))
        elif rng.unit(case_id, "panel", fid) < _PANEL_OPTIONAL_P:
            keep.append((fid, n))
    keep += _specialty_items(case_id, prof, inc)
    return keep


#: Sex-specific reference bands overriding the upstream bands. Sources:
#: adult hemoglobin (M 13.5-18 / F 12-16 g/dL); serum creatinine (M 53-106 /
#: F 44-97 umol/L). Ints, because the band is printed verbatim in the question.
SEX_REF: dict[str, dict[str, tuple[int, int]]] = {
    "Hb": {"F": (120, 160), "M": (130, 175)},
    "Cr": {"F": (44, 97), "M": (59, 104)},
    # ACG 2017 abnormal liver chemistries: ALT ULN M 29-33 / F 19-25 U/L
    # (upper end used).
    "ALT": {"F": (0, 25), "M": (0, 33)},
}

#: Sex-independent band overrides where the upstream "normal" upper bound
#: already crosses the action threshold: (low, high, action_threshold, source).
CLINICAL_REF_OVERRIDE: dict[str, tuple[float, float, float, str]] = {
    "HbA1c": (4.0, 5.6, 5.7,
              "ADA:正常 <5.7% · 糖尿病前期 5.7–6.4% · 糖尿病 ≥6.5%。上游上界 6.0 落在前期带内"),
    "Ca": (2.15, 2.55, 2.62,
           "高钙定义为校正总钙 >10.5 mg/dL(2.62 mmol/L);常规正常带 2.15–2.55。"
           "上游上界 2.75 已在高钙区间内"),
}

#: Action thresholds registered without changing the band. FBG follows WHO
#: (6.1) rather than ADA (5.6).
ACTION_THRESHOLD_ONLY: dict[str, tuple[float, str]] = {
    "FBG": (6.1, "WHO:正常 <6.1 mmol/L,IFG 6.1–6.9(ADA 用 5.6 —— 两标准并存,此处取 WHO)"),
    "K": (5.5, "高钾通常 >5.5 mmol/L"),
    "TG": (1.7, "TG 正常 <1.7 mmol/L(150 mg/dL)"),
    "Na": (145, "常规上限 145 mmol/L"),
    "TSH": (4.2, "常见实验室上限 4.2 mIU/L"),
    "LDL": (3.37, "3.37 mmol/L = 130 mg/dL,临界高值下界"),
}


def with_sex_ref(vocab: dict, sex: str | None) -> dict:
    """Apply `CLINICAL_REF_OVERRIDE`, `ACTION_THRESHOLD_ONLY` and, when `sex` is
    F/M, `SEX_REF`. Returns a shallow copy; the vocabulary is shared.
    """
    out = dict(vocab)
    for fid, (lo, hi, thr, _why) in CLINICAL_REF_OVERRIDE.items():
        spec = out.get(fid)
        if not isinstance(spec, dict):
            continue
        out[fid] = {**spec, "ref": {**(spec.get("ref") or {}), "low": lo, "high": hi},
                    "action_threshold": thr, "ref_overridden_by": "CLINICAL_REF_OVERRIDE"}
    for fid, (thr, _why) in ACTION_THRESHOLD_ONLY.items():
        spec = out.get(fid)
        if isinstance(spec, dict) and spec.get("action_threshold") is None:
            out[fid] = {**spec, "action_threshold": thr}
    _bg_sx = str(sex).upper() if sex and str(sex).upper() in ("F", "M") else "all"
    for fid in _background_tables():
        spec = out.get(fid)
        if isinstance(spec, dict):
            out[fid] = {**spec, "bg_sex": _bg_sx}
    if not sex or str(sex).upper() not in ("F", "M"):
        return out
    _sx = str(sex).upper()
    for fid, bands in SEX_REF.items():
        spec = out.get(fid)
        if not isinstance(spec, dict) or _sx not in bands:
            continue
        lo, hi = bands[_sx]
        out[fid] = {**spec, "ref": {**(spec.get("ref") or {}), "low": lo, "high": hi},
                    "ref_sex_adjusted": _sx}
    return out


_DECOY_POOL: dict[tuple[str, str], int] | None = None


def _disease_abnormality_pool() -> dict[tuple[str, str], int]:
    """`(fid, "high"|"low") -> number of condition profiles` that make the item
    cross that way (declared items plus the routine-panel partners they pull
    along). Built once from the catalog.
    """
    global _DECOY_POOL
    if _DECOY_POOL is not None:
        return _DECOY_POOL
    from .registry import composition_v2_pair_ids, condition_findings_for_case
    routine = {f for f, _ in ROUTINE_PANEL}
    pool: dict[tuple[str, str], int] = {}
    _skip = composition_v2_pair_ids()          # legacy draws must not move when a v2 pair is added
    for _sid, spec in condition_findings_for_case().items():
        if _sid in _skip:
            continue
        prof = {p["id"]: p for p in (spec or {}).get("findings") or ()}
        seen: set[tuple[str, str]] = set()
        for fid, p in prof.items():
            d = (p or {}).get("direction")
            if d not in ("high", "low"):
                continue
            seen.add((fid, d))
            for pf, sgn in _partners_of(fid):
                pd = _disease_partner_direction(d, sgn)
                if pf in prof or pf not in routine or pd is None:
                    continue
                seen.add((pf, pd))
        for k in seen:
            pool[k] = pool.get(k, 0) + 1
    _DECOY_POOL = pool
    return pool


def _insufficient_decoys(case_id: str, items: list[tuple[str, int]], prof: dict,
                         inc: dict, inc_disease: set, vocab: dict) -> dict[str, str]:
    """Decoys for an insufficient-tier panel, `fid -> "high"|"low"`: one per
    blanked disease item, drawn from items already on the panel (not declared,
    not a partner, not incidental), weighted by `_disease_abnormality_pool` so
    neither the count nor the choice of item marks the tier.
    """
    n_dis = sum(1 for fid, _ in items
                if (fid in prof and (prof[fid] or {}).get("direction") in
                    ("high", "low", "positive")) or fid in inc_disease)
    near = {pf for df in prof for pf, _ in _partners_of(df)}
    cands = [fid for fid, _ in items
             if fid not in prof and fid not in inc and fid not in near
             and (vocab.get(fid) or {}).get("ref") is not None
             and not (vocab.get(fid) or {}).get("qualitative")]
    pool = _disease_abnormality_pool()
    out: dict[str, str] = {}
    for i in range(min(n_dis, len(cands))):
        opts = [((f, d), w) for (f, d), w in sorted(pool.items()) if f in cands]
        u = rng.unit(case_id, "decoy", str(i))
        if opts:
            tot = sum(w for _, w in opts)
            acc, pick = 0.0, opts[-1][0]
            for k, w in opts:
                acc += w / tot
                if u < acc:
                    pick = k
                    break
            pk, d = pick
        else:
            pk, d = cands[int(u * len(cands)) % len(cands)], "high"
        out[pk] = d
        cands.remove(pk)
    return out


def _background_compensation(case_id: str, items: list[tuple[str, int]], prof: dict, inc: dict,
                             inc_disease: set, vocab: dict, sex: str | None) -> set[str]:
    """Items of an insufficient-tier panel that carry one incidental abnormality each, to keep the
    background pool equal across tiers.

    A blanked disease item must not cross its band (the tier shows none of the gold), so it gives
    up the background abnormality it would have had on the sufficient tier, where it is a designed
    abnormality instead. With 5% background that shortfall was invisible; with the general
    population's rate on decision-line items it shows up in the abnormal count and the largest
    excursion. For each blanked item, with that item's own background out-of-band probability, one
    other undeclared item (weighted by its background out-of-band probability) takes an
    out-of-band reading drawn from the tail of its background distribution (`_tail_quantile`),
    i.e. sized like an incidental abnormality, not like a declared finding.
    """
    blanked = [fid for fid, _ in items
               if (fid in prof and (prof[fid] or {}).get("direction") in ("high", "low", "positive"))
               or fid in inc_disease]
    near = {pf for df in prof for pf, _ in _partners_of(df)}
    cands = [fid for fid, _ in items
             if fid not in prof and fid not in inc and fid not in near and fid not in _DERIVED_ITEMS
             and (vocab.get(fid) or {}).get("ref") is not None
             and not (vocab.get(fid) or {}).get("qualitative")]

    def _rate(fid: str) -> float:
        lo, hi = _band(vocab.get(fid) or {})
        return sum(_background_masses(fid, sex if sex in ("F", "M") else None, lo, hi))
    out: set[str] = set()
    for b in blanked:
        if not cands:
            break
        w = [_rate(f) for f in cands]
        # the picked item would have been abnormal with probability `rbar` anyway, so it adds
        # only `1 - rbar` to the expected count
        rbar = sum(x * x for x in w) / sum(w)
        if rng.unit(case_id, "bgtail", b) >= min(1.0, _rate(b) / (1.0 - rbar)):
            continue
        u, acc, pick = rng.unit(case_id, "bgtail-pick", b) * sum(w), 0.0, cands[-1]
        for f, x in zip(cands, w):
            acc += x
            if u < acc:
                pick = f
                break
        out.add(pick)
        cands.remove(pick)
    return out


#: Items whose value the panel derives rather than draws.
_DERIVED_ITEMS: frozenset[str] = frozenset({"TC"})

_MAG_DIST: dict[tuple[str, str], dict[str, int]] | None = None


def _declared_magnitudes() -> dict[tuple[str, str], dict[str, int]]:
    """`(fid, direction) -> {magnitude: count}` over declared screening findings in
    the catalog, plus `("*", "*")` pooled. Built once."""
    global _MAG_DIST
    if _MAG_DIST is not None:
        return _MAG_DIST
    from .registry import condition_findings_for_case
    out: dict[tuple[str, str], dict[str, int]] = {}
    for spec in condition_findings_for_case().values():
        for p in (spec or {}).get("findings") or ():
            d, m = (p or {}).get("direction"), (p or {}).get("magnitude")
            if p.get("role") != "screening" or d not in ("high", "low") or m not in _MAGS:
                continue
            for key in ((p["id"], d), ("*", "*")):
                out.setdefault(key, {})
                out[key][m] = out[key].get(m, 0) + 1
    _MAG_DIST = out
    return out


def _draw_magnitude(case_id: str, fid: str, direction: str, salt: str) -> str:
    """Magnitude for an undeclared designed abnormality (partner, decoy), drawn
    from the declared magnitudes of the same item and direction, so its size does
    not mark it as undeclared. Items the catalog never declares use the pooled
    mild/moderate share (`marked` only where some condition declares it)."""
    dist = _declared_magnitudes()
    w = dist.get((fid, direction))
    if not w:
        w = {m: c for m, c in (dist.get(("*", "*")) or {"mild": 1}).items() if m != "marked"}
    tot = float(sum(w.values()))
    u = rng.unit(case_id, "mag", fid, salt)
    acc = 0.0
    for m in _MAGS:
        if m in w:
            acc += w[m] / tot
            if u < acc:
                return m
    return [m for m in _MAGS if m in w][-1]


def render_routine_panel(case_id: str, profile: dict, vocab: dict, index_T: int,
                         sex: str | None = None, ts_signals: object = (),
                         normal_before_day: int | None = None) -> list[dict]:
    """Renders the fixed initial-visit routine panel. Declared items follow the
    declaration; the rest are healthy readings of the general population, so each
    one falls outside its band with the reference interval's 5% (`_normal_value`,
    `spill=True`), mostly just outside.
    """
    vocab = with_sex_ref(vocab, sex)
    prof = {p["id"]: p for p in (profile or {}).get("findings") or ()}
    # fid -> direction / magnitude of undeclared abnormalities.
    _inc: dict[str, str] = {}
    _inc_mag: dict[str, str] = {}

    # Partners of declared abnormalities follow the declared direction and the
    # cluster sign. TC is derived from its partners (R2), so it is never pulled.
    _inc_disease: set[str] = set()
    for _df, _dspec in prof.items():
        _ddir = (_dspec or {}).get("direction")
        if _ddir not in ("high", "low"):
            continue
        for _pf, _sgn in _partners_of(_df):
            _pd = _disease_partner_direction(_ddir, _sgn)
            if _pf in prof or _pd is None or _pf in _DERIVED_ITEMS:
                continue
            if not any(_pf == f for f, _ in ROUTINE_PANEL):
                continue
            _inc[_pf] = _pd
            _inc_mag[_pf] = _draw_magnitude(case_id, _pf, _pd, "partner")
            _inc_disease.add(_pf)

    # One shared set of draw days, opened at presentation (`ddx.presentation_day`)
    # so the diagnostic draw is not blanked by the pre-onset rule.
    from .ddx import presentation_day
    _draws = draw_days(case_id, index_T, max(n for _, n in ROUTINE_PANEL),
                       not_before=presentation_day(normal_before_day, index_T))
    _items = _panel_items_for(case_id, prof, _inc, ts_signals)
    _on_panel = {f for f, _ in _items}
    _insufficient = normal_before_day is not None and int(normal_before_day) > int(index_T)
    # Insufficient tier: blanked disease abnormalities are replaced by as many
    # decoys, sized like declared findings, so neither "any abnormal lab" nor the
    # size of the largest one identifies the sufficient tier.
    _decoys: dict[str, str] = {}
    _tails: set[str] = set()
    if _insufficient:
        _decoys = _insufficient_decoys(case_id, _items, prof, _inc, _inc_disease, vocab)
        for _f, _d in _decoys.items():
            _inc[_f] = _d
            _inc_mag[_f] = _draw_magnitude(case_id, _f, _d, "decoy")
        _designed = set(_decoys)
        _tails = _background_compensation(case_id, _items, prof, _inc, _inc_disease, vocab, sex)
    else:
        _designed = ({f for f, p in prof.items()
                      if (p or {}).get("direction") in ("high", "low", "positive")}
                     | _inc_disease) & _on_panel
    out: list[dict] = []
    for fid, n_fixed in _items:
        spec = vocab.get(fid)
        if spec is None:
            continue
        p = prof.get(fid)
        _default_dir = _inc.get(fid, "normal")
        # The pre-onset rule applies only to disease-caused items.
        _disease = p is not None or fid in _inc_disease
        item = {"id": fid, "declared": p is not None,
                "direction": (p or {}).get("direction", _default_dir),
                "magnitude": (p or {}).get("magnitude", _inc_mag.get(fid)),
                "trajectory": (p or {}).get("trajectory", "stable"),
                "n": n_fixed, "role": "screening",
                # Undeclared, not disease-caused readings are the only source of
                # incidental abnormalities: the general-population tail of the band
                # (`_normal_value`, `spill=True`), not a hospital cohort's rate.
                "_spill": not _disease and fid not in _DERIVED_ITEMS,
                "_tail": fid in _tails}
        out += render_findings(case_id, {"findings": (item,)}, vocab, index_T,
                               days_override=_draws, _keep_private=True,
                               normal_before_day=normal_before_day if _disease else None)

    # ---- Reconciliation on the first draw, the only one with every panel item ----
    _d0 = _draws[0]
    _idx = {}
    for _i, _e in enumerate(out):
        if _e.get("source_timestamp") == _d0:
            _fid = _e.get("_fid")
            if _fid:
                _idx[_fid] = _i
    _vals = {}
    for _fid, _i in _idx.items():
        _v = out[_i].get("_val")
        if isinstance(_v, (int, float)):
            _vals[_fid] = float(_v)
    # Sampled `Ca` is corrected calcium; renaming it lets `R3-mech` derive the
    # printed total calcium (a declared `Ca` is already total calcium).
    _declared = {k for k in prof} & set(_vals)
    if "Ca" in _vals and "Alb" in _vals and "Ca" not in _declared:
        _vals["Ca_corrected"] = _vals.pop("Ca")
    _new, _log = reconcile_panel(_vals, _declared, vocab)
    _new.pop("Ca_corrected", None)
    # A derived panel TC carries the real signed Friedewald residual (the identity is exact only
    # in the formula; real same-day draws have |rho| median 0.114).
    if (all(k in _new for k in ("TC", "LDL", "HDL", "TG")) and "TC" not in _declared
            and _new["TC"] != _vals.get("TC") and _new["TG"] < FRIEDEWALD_TG_MAX_MMOL):
        _rho = panel_friedewald_residual(case_id, _new["LDL"], _new["TG"])
        _new["TC"] = (_new["LDL"] + _new["HDL"] + _new["TG"] / FRIEDEWALD_TG_DIVISOR
                      - _rho * _new["LDL"])
        _log.append(f"R2 残差 rho={_rho:.3f} ⇒ TC {_new['TC']:.2f}")
    for _fid, _nv in _new.items():
        if _fid in _idx and _vals.get(_fid) != _nv:
            _i = _idx[_fid]
            _spec = vocab.get(_fid) or {}
            _ref = _spec.get("ref") or {}
            _nv = _round(_nv, _fid)
            out[_i]["_val"] = _nv
            out[_i]["symptom"] = _panel_text(case_id, _fid, _spec, _nv)
    if _log:
        log.debug("reconcile[%s] %s", case_id, " | ".join(_log))
    return _strip_private(out)


# ---------------------------------------------------------------- Panel x streams
#
# When LDL / TG / FBG / HbA1c / ALT / AST are longitudinal streams, the panel omits
# them (`_TS_TO_PANEL`), so the panel alone cannot keep cross-item identities or
# show a declared finding on those items. `align_panel_to_streams` runs once the
# streams are final (after the physiology layer and rounding) and
#   (a) raises / lowers the post-onset stream points of a declared screening item
#       onto the declared side, so the finding is visible where the item is shown;
#   (b) derives panel TC from the same-day (nearest) LDL and TG and the panel HDL:
#       TC = LDL + HDL + TG/2.2 - rho * LDL, rho the real signed Friedewald residual.

_PANEL_TO_TS: dict[str, str] = {v: k for k, v in _TS_TO_PANEL.items()}

#: Signed relative Friedewald residual of the cross-surface TC,
#: rho = (LDL - (TC - HDL - TG/2.2)) / LDL, as the real same-day distribution: quantiles at
#: 2%, 4%, ..., 98% of n = 12597 real four-item draws with TG < 4.5 mmol/L (RA-labs
#: definition, `shortcuts.B_cross_surface.real_friedewald_rel_residual`: 5/25/50/75/95% =
#: -0.394 / -0.134 / -0.022 / 0.093 / 0.315). Both signs occur; TC is drawn as
#: LDL + HDL + TG/2.2 - rho * LDL, with rho truncated at TG / (2.2 * LDL) so TC >= LDL + HDL.
TC_RESIDUAL_Q_LEVELS: tuple[float, ...] = tuple(k / 50 for k in range(1, 50))
TC_RESIDUAL_Q: tuple[float, ...] = (
    -0.567, -0.4311, -0.3569, -0.3051, -0.2719, -0.2438, -0.2201, -0.1998, -0.1818, -0.1678,
    -0.1522, -0.139, -0.1274, -0.1171, -0.1068, -0.0966, -0.0871, -0.0773, -0.07, -0.0621,
    -0.0545, -0.0456, -0.0384, -0.0303, -0.0221, -0.0142, -0.0068, 0.001, 0.01, 0.0188,
    0.0278, 0.0374, 0.0461, 0.0563, 0.0655, 0.0759, 0.0875, 0.0985, 0.1109, 0.1252,
    0.1408, 0.1573, 0.1765, 0.198, 0.2231, 0.2496, 0.2908, 0.3467, 0.451)


def _tc_residual_cdf(x: float) -> float:
    """Piecewise-linear CDF of `TC_RESIDUAL_Q` (clamped to its 2%-98% range)."""
    q, lv = TC_RESIDUAL_Q, TC_RESIDUAL_Q_LEVELS
    if x <= q[0]:
        return lv[0]
    if x >= q[-1]:
        return lv[-1]
    i = max(j for j in range(len(q) - 1) if q[j] <= x)
    return lv[i] + (lv[i + 1] - lv[i]) * (x - q[i]) / (q[i + 1] - q[i])


def _tc_residual_ppf(p: float) -> float:
    """Inverse of `_tc_residual_cdf`."""
    q, lv = TC_RESIDUAL_Q, TC_RESIDUAL_Q_LEVELS
    p = min(max(float(p), lv[0]), lv[-1])
    i = min(max(j for j in range(len(lv)) if lv[j] <= p), len(lv) - 2)
    return q[i] + (q[i + 1] - q[i]) * (p - lv[i]) / (lv[i + 1] - lv[i])


def tc_friedewald_residual(cid: str, ldl: float, tg: float) -> float:
    """rho for one case: the real signed distribution (its 2%-98% range) truncated at
    TG / (2.2 * LDL)."""
    cap = (tg / FRIEDEWALD_TG_DIVISOR) / ldl if ldl > 0 else float("inf")
    lo = TC_RESIDUAL_Q_LEVELS[0]
    return _tc_residual_ppf(lo + rng.unit(cid, "tc_residual") * (_tc_residual_cdf(cap) - lo))


def panel_friedewald_residual(cid: str, ldl: float, tg: float) -> float:
    """rho of a panel-internal derived TC: as `tc_friedewald_residual` (the real signed
    distribution truncated at TG / (2.2 * LDL), so TC >= LDL + HDL), on its own draw."""
    cap = (tg / FRIEDEWALD_TG_DIVISOR) / ldl if ldl > 0 else float("inf")
    lo = TC_RESIDUAL_Q_LEVELS[0]
    return _tc_residual_ppf(lo + rng.unit(cid, "tc_residual_panel") * (_tc_residual_cdf(cap) - lo))


def _stream_value_at(pts: list, day: int) -> float | None:
    """The stream point on `day`, else the nearest one (ties: the earlier)."""
    best = None
    for q in pts or ():
        if not isinstance(q, dict) or not isinstance(q.get("value"), (int, float)):
            continue
        key = (abs(int(q["ts"]) - int(day)), int(q["ts"]))
        if best is None or key < best[0]:
            best = (key, float(q["value"]))
    return None if best is None else best[1]


def _panel_value(entry: dict, spec: dict) -> float | None:
    import re as _re
    m = _re.match(_re.escape(str(spec.get("name_cn", ""))) + r" (-?\d+(?:\.\d+)?)",
                  str(entry.get("symptom") or ""))
    return float(m.group(1)) if m else None


def align_panel_to_streams(raw, normal_before_day: int | None = None) -> list[str]:
    """Reconciles the rendered panel with the final longitudinal streams in place
    (see the section comment). Returns a log. Reads the declaration, never the
    diagnosis; the insufficient tier (onset after `T`) is left alone.
    """
    from .registry import condition_findings_for_case, load_findings
    from .wq import resolve_path
    cid = str(getattr(raw, "case_id", ""))
    ld = getattr(raw, "longitudinal_data", None) or {}
    ledger = getattr(raw, "evidence_ledger", None) or []
    log_: list[str] = []
    pref = f"EV-{cid}-L"
    ent = {str(e.get("evidence_id"))[len(pref):]: e for e in ledger
           if e.get("relevance") == "routine_panel"
           and str(e.get("evidence_id", "")).startswith(pref)}
    if not ent:
        return log_
    vocab0 = load_findings()
    sid = resolve_path(raw, "W.adjudication.ddx.spec_id")
    prof = (condition_findings_for_case(vocab0).get(sid) or {}) if isinstance(sid, str) else {}
    vocab = with_sex_ref(vocab0, (getattr(raw, "user_profile", None) or {}).get("sex"))
    T = int((getattr(raw, "prediction_context", None) or {}).get("prediction_time_T") or 84)
    onset = int(normal_before_day) if normal_before_day is not None else 0

    # (a) declared screening findings carried by a stream. Only when no point up to
    # T already shows the declared side; never on a stream that is a diagnostic
    # signal of the base disease (the base disease owns that stream's level, e.g.
    # TG in dyslipidemia); values stay inside the kernel's physiological domain.
    from .indicators import _declared_ndigits
    from . import indicators as _ind
    _pb = ((getattr(raw, "latent_premise", None) or {}).get("patient_basics") or {})
    _base = str(_pb.get("disease") or "")
    _base_sigs = set(((_ind.diagnoses() or {}).get(_base) or {}).get("any_of") or ())
    try:
        from haenv_kernel.latent import DISEASE_SIGNAL_DOMAIN as _DOM
        _dom = _DOM.get(_base) or {}
    except ImportError:                            # kernel not on the path
        _dom = {}
    if onset <= T:
        for p in (prof.get("findings") or ()):
            fid, d = p.get("id"), p.get("direction")
            s = _PANEL_TO_TS.get(fid)
            if p.get("role") != "screening" or d not in ("high", "low") or not s or not ld.get(s):
                continue
            spec = vocab.get(fid) or {}
            lo, hi = _band(spec)
            pts = sorted((q for q in ld[s] if isinstance(q, dict)
                          and isinstance(q.get("value"), (int, float))), key=lambda q: int(q["ts"]))

            def _on(v):
                return v > hi if d == "high" else v < lo
            if any(_on(float(q["value"])) for q in pts if int(q["ts"]) <= T):
                continue
            if s in _base_sigs:
                log_.append(f"{s}: declared {fid} {d} not shown -- base-disease signal of {_base}")
                continue
            nd = _declared_ndigits(s)
            dlo, dhi = ((_dom.get(s) or {}).get("range") or (float("-inf"), float("inf")))
            for k_on, q in enumerate(q for q in pts if int(q["ts"]) >= onset):
                v = float(q["value"])
                if _on(v):
                    continue
                tv = float(_value(spec, d, p.get("magnitude"), cid, fid, k_on))
                tv = min(float(dhi), max(float(dlo), tv))
                tv = round(tv, nd) if nd else float(round(tv))
                if not _on(tv):
                    log_.append(f"{s}@{q['ts']}: declared {fid} {d} cannot be shown inside the domain")
                    continue
                log_.append(f"{s}@{q['ts']} {v} -> {tv} (declared {fid} {d})")
                q["value"] = tv

    # (b) TC from same-day LDL / TG (stream or panel) and the panel HDL
    tc_e, hdl_e = ent.get("TC0"), ent.get("HDL0")
    if tc_e is not None and hdl_e is not None and (ld.get("LDL") or ld.get("triglycerides")):
        day = int(tc_e.get("source_timestamp", 0))
        L = (_stream_value_at(ld["LDL"], day) if ld.get("LDL")
             else (_panel_value(ent["LDL0"], vocab.get("LDL") or {}) if "LDL0" in ent else None))
        G = (_stream_value_at(ld["triglycerides"], day) if ld.get("triglycerides")
             else (_panel_value(ent["TG0"], vocab.get("TG") or {}) if "TG0" in ent else None))
        H = _panel_value(hdl_e, vocab.get("HDL") or {})
        if None not in (L, G, H):
            floor = L + H                                  # TC >= LDL + HDL
            rho = tc_friedewald_residual(cid, L, G)
            tc = _round(L + H + G / FRIEDEWALD_TG_DIVISOR - rho * L, "TC")
            while tc < floor:
                tc = _round(tc + _step("TC", tc), "TC")
            spec = vocab.get("TC") or {}
            tc_e["symptom"] = _panel_text(cid, "TC", spec, tc)
            log_.append(f"TC@{day} = LDL {L} + HDL {H} + TG {G}/2.2 - {rho:.3f}*LDL = {tc}")
    if log_:
        log.debug("align[%s] %s", cid, " | ".join(log_))
    return log_


def render_findings(case_id: str, profile: dict, vocab: dict, index_T: int,
                    only_roles: tuple[str, ...] = ("screening",),
                    days_override: list[int] | None = None,
                    _keep_private: bool = False,
                    normal_before_day: int | None = None) -> list[dict]:
    """Renders declared findings into ledger entries (screening items only, by
    default). `days_override` shares the routine panel's draw days; otherwise
    each item gets its own `_timeline`.
    """
    out: list[dict] = []
    for p in (profile or {}).get("findings") or ():
        if p["role"] not in only_roles:
            continue
        spec = vocab.get(p["id"])
        if spec is None:                      # not in the vocabulary: not rendered
            continue
        n = max(1, int(p.get("n") or 1))
        days = (list(days_override[:n]) if days_override else
                _timeline(index_T, n, case_id, p["id"]))
        traj, direction = str(p["trajectory"]), str(p["direction"])
        # `fluctuating` and `treatment_responsive` count post-onset readings
        # (`k_on`), so the pre-onset rule cannot blank every reading of a marker;
        # `episodic` keeps the raw index.
        _pre_n = (sum(1 for _d in days if int(_d) < int(normal_before_day))
                  if normal_before_day is not None else 0)
        for k, day in enumerate(days):
            d = direction
            k_on = k - _pre_n                     # < 0: pre-onset
            # Draws before the first true symptom day read normal.
            if normal_before_day is not None and int(day) < int(normal_before_day):
                d = "normal"
            if spec.get("qualitative") or direction in QUALITATIVE_DIRECTIONS:
                val = ("阴性" if d == "normal" else
                       ("阳性" if direction == "positive" else "阴性"))
            else:
                if traj == "episodic" and k == 0:
                    d = "normal"
                elif traj == "progressive":
                    mag = _progressive_mag(p.get("magnitude"), k, len(days))
                    val = _value(spec, d, mag, case_id, p["id"], k,
                                 declared=bool(p.get("declared", True)),
                                 spill=bool(p.get("_spill")), tail=bool(p.get("_tail")))
                    out.append(_entry(case_id, spec, p, day, val, k))
                    continue
                elif traj == "fluctuating" and k_on >= 0 and k_on % 2 == 1:
                    d = "normal"
                elif (traj == "treatment_responsive" and k == len(days) - 1
                      and k_on >= 1):
                    d = "normal"
                val = _value(spec, d, p.get("magnitude"), case_id, p["id"], k,
                             declared=bool(p.get("declared", True)),
                             spill=bool(p.get("_spill")), tail=bool(p.get("_tail")))
            out.append(_entry(case_id, spec, p, day, val, k))
    return out if _keep_private else _strip_private(out)


#: Plausible band for a derived value: [low/f, high*f] of the reference range.
#: An empirical sanity bound; never enters a score.
_PLAUSIBLE_FACTOR = 3.0

#: Empirical slope of total calcium vs. albumin (mmol/L per g/L), from a
#: per-patient regression on de-identified EMR (diabetes and hypothyroidism
#: cohorts). Used for generation only; it deliberately differs from Payne's
#: 0.02495 so the Payne check (`R3-payne`) can fail.
CA_ALB_SLOPE_EMPIRICAL = 0.0164

#: Na-Cl gap band (mmol/L). Empirical; the anion gap would need HCO3, which the
#: panel lacks.
NA_CL_GAP = (30.0, 42.0)

#: Which Friedewald item is recomputed first. TC is derived from LDL + HDL + TG/2.2
#: (P0-3, 2026-09-30): sampling TC and HDL on one shared cause with opposite
#: loadings made TC~HDL negative (-0.19 against +0.41 real), while TC contains HDL.
#: TC is never declared and always renders normal, so it always carries the identity.
_DERIVE_RANK = {"TC": 0, "LDL": 1, "HDL": 2, "TG": 3}

#: Cap on repair rounds; still moving at the cap logs `unstable`.
_MAX_ROUNDS = 8

#: Rule table `(name, category, reads, writes, source)`. Order is derived by
#: `_order_rules`. `derive` rules define an item from others and run once;
#: `repair` rules project out-of-band values back and iterate to a fixed point.
#: `Ca_corrected` is the sampled (corrected) calcium; `R3-mech` derives the
#: printed total calcium from it.
RULE_SPECS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...], str], ...] = (
    ("R2-friedewald", "derive", ("LDL", "TC", "HDL", "TG"), ("LDL", "TC", "HDL", "TG"),
     "Friedewald 1972 · PMID 4337382"),
    ("R3-mech", "derive", ("Ca_corrected", "Alb"), ("Ca",),
     "empirical slope 0.0164, real EMR regression (diabetes + hypothyroidism cohorts)"),
    ("R3-thr", "repair", ("Ca", "Alb"), ("Ca", "Alb"),
     "高钙行动阈值 2.62 mmol/L(校正总钙 >10.5 mg/dL)"),
    ("R3-payne", "repair", ("Ca", "Alb"), ("Alb",),
     "Payne 1973 · PMID 4758544"),
    ("R5-tsh-ft4", "repair", ("TSH", "FT4"), ("FT4",),
     "thyroid axis direction consistency (only exercised when FT4 is ordered on demand; FT4 is not in the routine panel)"),
    ("R7-anion", "repair", ("Na", "Cl"), ("Cl",),
     "阴离子间隙常规带 30–42 mmol/L"),
)


def _order_rules(rules, strict: bool = False):
    """Topological order by `writes -> reads`, ties by name. A cycle among
    `derive` rules raises (`strict=True`); among `repair` rules it is expected
    and falls back to name order.
    """
    rs = list(rules)
    dep = {r[0]: set() for r in rs}
    for a in rs:
        for b in rs:
            if a[0] != b[0] and (set(a[3]) & set(b[2])):
                dep[b[0]].add(a[0])
    out, done = [], set()
    while len(out) < len(rs):
        ready = sorted((r for r in rs if r[0] not in done and dep[r[0]] <= done),
                       key=lambda r: r[0])
        if not ready:
            _cyc = sorted(r[0] for r in rs if r[0] not in done)
            if strict:
                raise ValueError(f"[rules] derive rules form a cycle, no order exists: {_cyc}; "
                                 f"each reads the other's output, so no order is chosen")
            ready = sorted((r for r in rs if r[0] not in done), key=lambda r: r[0])
        out.extend(ready)
        done |= {r[0] for r in ready}
    return out


def rule_write_conflicts(rules) -> dict[str, list[str]]:
    """Fields written by more than one rule, i.e. where ordering matters."""
    own: dict[str, list[str]] = {}
    for nm, _k, _rd, wr, _cite, _fn in rules:
        for f in wr:
            own.setdefault(f, []).append(nm)
    return {f: sorted(v) for f, v in own.items() if len(v) > 1}


#: Computed at import: two `derive` rules writing the same field is an error.
RULE_WRITE_CONFLICTS = rule_write_conflicts(
    [(nm, k, rd, wr, cite, None) for nm, k, rd, wr, cite in RULE_SPECS])
_KIND = {nm: k for nm, k, _rd, _wr, _c in RULE_SPECS}
for _f, _owners in RULE_WRITE_CONFLICTS.items():
    if sum(1 for o in _owners if _KIND[o] == "derive") > 1:
        raise ValueError(f"[rules] field {_f!r} is written by more than one derive rule: {_owners}; "
                         f"a derive rule defines the field, so it must have exactly one")


def _ref_side(x: float, ref: dict) -> str:
    """`low` / `normal` / `high` against a reference range."""
    if "low" in ref and x < float(ref["low"]):
        return "low"
    if "high" in ref and x > float(ref["high"]):
        return "high"
    return "normal"


def reconcile_panel(vals: dict[str, float], declared: set[str],
                    vocab: dict) -> tuple[dict[str, float], list[str]]:
    """Adjusts panel values so the reconciled relations (`RULE_SPECS`) hold;
    returns `(values, log)`. Declared items are not moved, and Friedewald never
    moves a rendered low or high across its reference boundary; when no free item
    can carry the identity it logs `unsatisfiable` and writes nothing.
    Corrections are deterministic and every step is logged.
    """
    v = dict(vals)
    log: list[str] = []

    def _r2(v, declared, vocab, log):
        # -- R2 Friedewald (SI units): LDL = TC - HDL - TG/2.2, not valid when TG >= 4.5 --
        quad = ("LDL", "TC", "HDL", "TG")
        if all(k in v for k in quad):
            if v["TG"] >= FRIEDEWALD_TG_MAX_MMOL:
                log.append(f"R2 跳过:TG={v['TG']} ≥ {FRIEDEWALD_TG_MAX_MMOL},Friedewald 不适用")
            else:
                free = [k for k in sorted(quad, key=lambda x: _DERIVE_RANK[x])
                        if k not in declared]
                if not free:
                    log.append("R2 unsatisfiable:四项全被条件谱声明,不硬修(条件谱本身可能不自洽)")
                # The first free item (by `_DERIVE_RANK`) whose derived value is
                # plausible, inside the formula's domain, and on the same side of
                # its reference range as rendered (a rendered low or high is never
                # moved across its boundary) carries the identity. When no free
                # item can, nothing is written and the contradiction is logged.
                tried: list[str] = []
                for tgt in free:
                    if tgt == "LDL":
                        nv = v["TC"] - v["HDL"] - v["TG"] / FRIEDEWALD_TG_DIVISOR
                    elif tgt == "TC":
                        nv = v["LDL"] + v["HDL"] + v["TG"] / FRIEDEWALD_TG_DIVISOR
                    elif tgt == "HDL":
                        nv = v["TC"] - v["LDL"] - v["TG"] / FRIEDEWALD_TG_DIVISOR
                    else:
                        nv = (v["TC"] - v["HDL"] - v["LDL"]) * FRIEDEWALD_TG_DIVISOR
                    _r = (vocab.get(tgt) or {}).get("ref") or {}
                    _lo = float(_r.get("low", 0.0)) / _PLAUSIBLE_FACTOR
                    _hi = float(_r.get("high", 1.0)) * _PLAUSIBLE_FACTOR
                    _side_in = _ref_side(v[tgt], _r)
                    if not (max(_lo, 1e-6) <= nv <= _hi):
                        tried.append(f"{tgt}={nv:.3f} 出合理带 [{max(_lo, 0):.2f}, {_hi:.2f}]")
                    elif tgt == "TG" and nv >= FRIEDEWALD_TG_MAX_MMOL:
                        tried.append(f"TG={nv:.3f} ≥ {FRIEDEWALD_TG_MAX_MMOL},出 Friedewald 定义域")
                    elif _side_in != "normal" and _ref_side(round(nv, 2), _r) != _side_in:
                        tried.append(f"{tgt}={nv:.3f} 越过参考界 [{_r.get('low')}, {_r.get('high')}]"
                                     f"(渲染为 {_side_in})")
                    else:
                        log.append(f"R2 由恒等式推出 {tgt}: {v[tgt]:.2f} → {round(nv, 2):.2f}"
                                   + (f"(已跳过 {'; '.join(tried)})" if tried else ""))
                        v[tgt] = round(nv, 2)
                        break
                else:
                    if free:
                        log.append(f"R2 unsatisfiable:{'; '.join(tried)} ⇒ 不修,留给校验器报出")

    def _r3mech(v, declared, vocab, log):
        # -- R3-mech: total calcium = sampled corrected calcium + slope*(Alb - 40).
        # Only for undeclared Ca. Slope and anchor are in g/L; `_alb_gpl` skips
        # the derivation when the vocabulary's Alb is on another scale. --
        _alb_ref = (vocab.get("Alb") or {}).get("ref") or {}
        _alb_gpl = (20.0 <= float(_alb_ref.get("low", 0)) and
                    float(_alb_ref.get("high", 1e9)) <= 100.0)
        if "Ca_corrected" in v and "Alb" in v and not _alb_gpl:
            log.append(f"R3-mech 跳过:Alb 参考带 {_alb_ref or '(无)'} 不像 g/L 尺度,"
                       f"而斜率与锚点 40 都是 g/L 口径 ⇒ 不推导")
        elif "Ca_corrected" in v and "Alb" in v:
            _mech = (v["Ca_corrected"]
                     + CA_ALB_SLOPE_EMPIRICAL * (v["Alb"] - PAYNE_ANCHOR_ORIGINAL_G_PER_L))
            log.append(f"R3-mech 总钙 = 校正钙 {v['Ca_corrected']:.2f} + "
                       f"{CA_ALB_SLOPE_EMPIRICAL:.5f}×(Alb {v['Alb']:.0f} − 40) = {_mech:.3f}")
            v["Ca"] = round(_mech, 2)


    def _r3thr(v, declared, vocab, log):
        # Repair: a derived total calcium above the hypercalcemia threshold
        # with Ca undeclared lowers Alb (within its band) until it falls under;
        # otherwise logs unsatisfiable. Declared Ca is never touched.
        _thr = (vocab.get("Ca") or {}).get("action_threshold")
        if (_thr is not None and "Alb" not in declared and "Ca" not in declared
                and v["Ca"] > float(_thr)):
            _pre = v["Ca"]
            _t = float(_thr)
            _margin = 0.02
            # Solves for Alb: corrected + slope*(Alb - 40) = t - margin
            _corr = _pre - CA_ALB_SLOPE_EMPIRICAL * (v["Alb"] - PAYNE_ANCHOR_ORIGINAL_G_PER_L)
            _need = (PAYNE_ANCHOR_ORIGINAL_G_PER_L
                     + (_t - _margin - _corr) / CA_ALB_SLOPE_EMPIRICAL)
            _aref = (vocab.get("Alb") or {}).get("ref") or {}
            _alo, _ahi = float(_aref.get("low", 35)), float(_aref.get("high", 55))
            if _alo <= _need <= _ahi:
                _new_alb = round(min(_need, v["Alb"]))     # only lowers
                v["Alb"] = _new_alb
                v["Ca"] = round(_corr + CA_ALB_SLOPE_EMPIRICAL
                                * (_new_alb - PAYNE_ANCHOR_ORIGINAL_G_PER_L), 2)
                log.append(f"R3-thr 派生总钙 {_pre:.3f} 越过行动阈值 {_t}(高钙) 而 Ca 未被声明 ⇒ "
                           f"Alb 退到 {_new_alb:.0f},总钙 → {v['Ca']:.2f}")
            else:
                log.append(f"R3-thr unsatisfiable:派生总钙 {_pre:.3f} 越阈值 {_t},"
                           f"但需要的 Alb={_need:.0f} 出 [{_alo:.0f}, {_ahi:.0f}] ⇒ 不动")

    def _r3payne(v, declared, vocab, log):
        # -- R3 Payne-corrected calcium: moves Alb (never Ca) when outside the envelope --
        if "Ca" in v and "Alb" in v and "Alb" not in declared:
            ref = (vocab.get("Ca") or {}).get("ref") or {}
            lo, hi = float(ref.get("low", 2.25)), float(ref.get("high", 2.75))
            band = 0.5 * (hi - lo)
            adj = payne_corrected_ca(v["Ca"], v["Alb"], PAYNE_ANCHOR_ORIGINAL_G_PER_L)
            if not (lo - band <= adj <= hi + band):
                need = min(max(adj, lo - band), hi + band)
                new_alb = PAYNE_ANCHOR_ORIGINAL_G_PER_L - (need - v["Ca"]) / PAYNE_COEF_PER_G_PER_L
                aref = (vocab.get("Alb") or {}).get("ref") or {}
                alo, ahi = float(aref.get("low", 35)), float(aref.get("high", 55))
                # The repaired Alb must stay in its own reference range.
                if not (alo <= new_alb <= ahi):
                    log.append(f"R3 unsatisfiable:校正钙 {adj:.3f} 出包络,但需要的 Alb="
                               f"{new_alb:.0f} 落在参考区间 [{alo:.0f}, {ahi:.0f}] 之外 ⇒ 不修")
                else:
                    log.append(f"R3 校正钙 {adj:.3f} 出包络 ⇒ Alb {v['Alb']:.0f} → {round(new_alb):.0f}")
                    v["Alb"] = round(new_alb)

    def _r5(v, declared, vocab, log):
        # -- R5 TSH/FT4 forbidden combination: only repaired when neither item is
        # declared, and only FT4 is moved --
        if "TSH" in v and "FT4" in v and "FT4" not in declared and "TSH" not in declared:
            tref = (vocab.get("TSH") or {}).get("ref") or {}
            fref = (vocab.get("FT4") or {}).get("ref") or {}
            c = check_tsh_ft4(v["TSH"], v["FT4"],
                              (float(tref.get("low", 0.27)), float(tref.get("high", 4.2))),
                              (float(fref.get("low", 12)), float(fref.get("high", 22))))
            if c.ok is False:
                flo, fhi = float(fref.get("low", 12)), float(fref.get("high", 22))
                # Move to the nearest band edge, not the midpoint, to avoid
                # coupling the two items artificially.
                _edge = 0.08 * (fhi - flo)
                nv = round(flo + _edge if v["FT4"] < flo else fhi - _edge, 2)
                log.append(f"R5 {c.detail} ⇒ FT4 {v['FT4']:.2f} → {nv:.2f}(最小改动:挪进带内最近边缘)")
                v["FT4"] = nv

    def _r7(v, declared, vocab, log):
        # -- R7 Na-Cl gap: moves Cl when out of band --
        if "Na" in v and "Cl" in v and "Cl" not in declared:
            d = v["Na"] - v["Cl"]
            if not (NA_CL_GAP[0] <= d <= NA_CL_GAP[1]):
                # Move Cl to the nearest band edge, rounding in the direction that
                # keeps it inside, plus a margin `_M`.
                _tgt = NA_CL_GAP[1] if d > NA_CL_GAP[1] else NA_CL_GAP[0]
                import math as _m
                _M = 0.05
                # Cl is printed at its declared precision, so round on that grid.
                _inv = 1.0 / _step("Cl", v["Cl"])
                if d > NA_CL_GAP[1]:
                    nv = _m.ceil((v["Na"] - _tgt + _M) * _inv) / _inv
                else:
                    nv = _m.floor((v["Na"] - _tgt - _M) * _inv) / _inv
                # `NA_CL_GAP` is absolute, so the result must also pass the
                # plausibility band.
                _r7 = (vocab.get("Cl") or {}).get("ref") or {}
                _r7lo = float(_r7.get("low", 0.0)) / _PLAUSIBLE_FACTOR
                _r7hi = float(_r7.get("high", 1.0)) * _PLAUSIBLE_FACTOR
                if not (max(_r7lo, 1e-6) <= nv <= _r7hi):
                    log.append(f"R7 unsatisfiable:推出 Cl={nv:.1f} 落在合理带 "
                               f"[{max(_r7lo, 0):.1f}, {_r7hi:.1f}] 之外 ⇒ 不修")
                else:
                    log.append(f"R7 Na−Cl={d:.1f} 出带 {NA_CL_GAP} ⇒ Cl {v['Cl']:.1f} → {nv:.1f}")
                    v["Cl"] = nv


    _FN = {"R2-friedewald": _r2, "R3-mech": _r3mech, "R3-thr": _r3thr,
           "R3-payne": _r3payne, "R5-tsh-ft4": _r5, "R7-anion": _r7}
    _RULES = tuple((nm, k, rd, wr, cite, _FN[nm]) for nm, k, rd, wr, cite in RULE_SPECS)

    # (1) Derive: each rule runs once.
    _derive = [r for r in _RULES if r[1] == "derive"]
    for _nm, _k, _rd, _wr, _cite, _fn in _order_rules(_derive, strict=True):
        if set(_rd) <= set(v):
            _fn(v, declared, vocab, log)

    # (2) Repair: iterate to a fixed point, at most `_MAX_ROUNDS`.
    _repair = [r for r in _RULES if r[1] == "repair"]
    for _round in range(_MAX_ROUNDS):
        _before = dict(v)
        for _nm, _k, _rd, _wr, _cite, _fn in _order_rules(_repair):
            if set(_rd) <= set(v):
                _fn(v, declared, vocab, log)
        if v == _before:
            break
    else:
        log.append(f"unstable:修复跑满 {_MAX_ROUNDS} 轮仍未收敛 —— "
                   f"可能是两条规则在拉锯,不视为已修复")
    return v, log


def _strip_private(entries: list[dict]) -> list[dict]:
    """Drop underscore keys (reconciliation state) before entries reach the
    solver-visible evidence ledger.
    """
    return [{k: v for k, v in e.items() if not str(k).startswith("_")} for e in entries]


#: Share of reports that print no flag column at all (real EMR: `(empty)` 6-35%
#: of rows per item, RC-observation `real_readout.json` `format.*.flags`).
FLAG_ABSENT_SHARE = 0.25


def _flag(case_id: str, val, ref: dict) -> str:
    """Report flag as real lab reports print it: `H` / `L` out of range, `N` in
    range, nothing when this case's report has no flag column."""
    if not ref or not isinstance(val, (int, float)):
        return ""
    if rng.unit(case_id, "flagcol") < FLAG_ABSENT_SHARE:
        return ""
    lo, hi = ref.get("low"), ref.get("high")
    if hi is not None and val > float(hi):
        return "H"
    if lo is not None and val < float(lo):
        return "L"
    return "N"


def _panel_text(case_id: str, fid: str | None, spec: dict, val) -> str:
    ref = spec.get("ref") or {}
    txt = (f"{spec.get('name_cn', fid)} {_fmt(fid, val)} {spec.get('unit', '')}"
           + (f"(参考 {ref.get('low')}–{ref.get('high')})" if ref else ""))
    fl = _flag(case_id, val, ref)
    return (txt.strip() + (f" {fl}" if fl else "")).strip()


def _entry(case_id: str, spec: dict, p: dict, day: int, val, k: int) -> dict:
    txt = _panel_text(case_id, p["id"], spec, val)
    return {"evidence_id": f"EV-{case_id}-L{p['id']}{k}",
            "source_type": "lab_result",
            "source_timestamp": int(day),
            "symptom": txt.strip(),
            "claim_supported": True,
            "relevance": "routine_panel",
            # internal; removed by `_strip_private`
            "_fid": p["id"], "_val": val}
