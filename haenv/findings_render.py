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

import logging
import math

from . import rng
from .relations import (FRIEDEWALD_TG_DIVISOR, FRIEDEWALD_TG_MAX_MMOL,
                        PAYNE_ANCHOR_ORIGINAL_G_PER_L, PAYNE_COEF_PER_G_PER_L,
                        check_tsh_ft4, payne_corrected_ca)

log = logging.getLogger("haenv.findings_render")

# Magnitude tier -> multiple of the reference-range width by which the value
# clears the boundary.
MAG_FACTOR = {"mild": (0.05, 0.25), "moderate": (0.3, 0.7), "marked": (0.9, 1.6)}
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


def _round(v: float) -> float:
    """Four decimals below 0.1 (e.g. a suppressed TSH), two above."""
    return round(v, 4 if abs(v) < 0.1 else 2)


def _value(spec: dict, direction: str, magnitude: str | None,
           case_id: str, fid: str, k: int, declared: bool = True) -> float:
    """The value for one reading; reads only case_id, field name and reading index.

    Undeclared items (`declared=False`) are mixed with their shared cause
    (`FACTOR_LOADING`); declared items already are the shared cause.
    """
    lo, hi = _band(spec)
    width = hi - lo
    u = rng.unit(case_id, "finding", fid, str(k))
    if not declared:
        u = _shared_u(case_id, fid, u)
    if direction == "normal":
        return _round(lo + 0.2 * width + u * 0.6 * width)
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
    return _round(val)


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

#: Screening items outside `ROUTINE_PANEL` declared by at least two conditions,
#: always rendered in full so the item list does not identify the condition.
_SPECIALTY_SCREEN: tuple[tuple[str, int], ...] = (
    ("ESR", 1), ("CRP", 1), ("ANA", 1), ("Ferritin", 1), ("Insulin", 1), ("VitD25", 1),
)


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
    case; the specialty group is always appended.
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
    keep += list(_SPECIALTY_SCREEN)
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
    from .registry import condition_findings_for_case
    routine = {f for f, _ in ROUTINE_PANEL}
    pool: dict[tuple[str, str], int] = {}
    for spec in condition_findings_for_case().values():
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


def render_routine_panel(case_id: str, profile: dict, vocab: dict, index_T: int,
                         sex: str | None = None, ts_signals: object = (),
                         normal_before_day: int | None = None) -> list[dict]:
    """Renders the fixed initial-visit routine panel. Declared items follow the
    declaration; the rest read normal apart from incidental findings.
    """
    vocab = with_sex_ref(vocab, sex)
    prof = {p["id"]: p for p in (profile or {}).get("findings") or ()}
    # Incidental findings: 0-2 mild abnormalities per case, unrelated to the
    # diagnosis, so the count of abnormal items does not identify the case. The
    # pool includes the specialty group so single-condition items are not tells.
    _pool = [fid for fid, _ in (*ROUTINE_PANEL, *_SPECIALTY_SCREEN) if fid not in prof]
    _n_inc = int(rng.unit(case_id, "incidental", "n") * 3)          # 0/1/2
    # fid -> direction; a shared-cause partner may move the opposite way.
    _inc: dict[str, str] = {}
    for _i in range(_n_inc):
        if not _pool:
            break
        # Prefer a shared-cause partner of the previous pick.
        _pk = None
        if _inc:
            _last = list(_inc)[-1]
            _cands = [(f, d) for f, d in _partners_of(_last) if f in _pool]
            if _cands:
                _j = int(rng.unit(case_id, "incidental", "partner", str(_i)) * len(_cands))
                _pk, _dir = _cands[min(_j, len(_cands) - 1)]
                _inc[_pk] = "high" if _dir > 0 else "low"
        if _pk is None:
            _pk = _pool[int(rng.unit(case_id, "incidental", str(_i)) * len(_pool)) % len(_pool)]
            _inc[_pk] = "high"
        _pool.remove(_pk)

    # Partners of declared abnormalities follow the declared direction and the
    # cluster sign; they are outside the incidental budget.
    _inc_disease: set[str] = set()
    for _df, _dspec in prof.items():
        _ddir = (_dspec or {}).get("direction")
        if _ddir not in ("high", "low"):
            continue
        for _pf, _sgn in _partners_of(_df):
            _pd = _disease_partner_direction(_ddir, _sgn)
            if _pf in prof or _pd is None:
                continue
            if not any(_pf == f for f, _ in ROUTINE_PANEL):
                continue
            _inc[_pf] = _pd
            _inc_disease.add(_pf)

    # One shared set of draw days, opened at presentation (`ddx.presentation_day`)
    # so the diagnostic draw is not blanked by the pre-onset rule.
    from .ddx import presentation_day
    _draws = draw_days(case_id, index_T, max(n for _, n in ROUTINE_PANEL),
                       not_before=presentation_day(normal_before_day, index_T))
    _items = _panel_items_for(case_id, prof, _inc, ts_signals)
    # Insufficient tier: blanked disease abnormalities are replaced by as many
    # mild decoys, so "any abnormal lab" does not identify the sufficient tier.
    if normal_before_day is not None and int(normal_before_day) > int(index_T):
        _inc.update(_insufficient_decoys(case_id, _items, prof, _inc, _inc_disease, vocab))
    out: list[dict] = []
    for fid, n_fixed in _items:
        spec = vocab.get(fid)
        if spec is None:
            continue
        p = prof.get(fid)
        _default_dir = _inc.get(fid, "normal")
        item = {"id": fid, "declared": p is not None,
                "direction": (p or {}).get("direction", _default_dir),
                "magnitude": (p or {}).get("magnitude", "mild" if fid in _inc else None),
                "trajectory": (p or {}).get("trajectory", "stable"),
                "n": n_fixed, "role": "screening"}
        # The pre-onset rule applies only to disease-caused items.
        _disease = p is not None or fid in _inc_disease
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
    for _fid, _nv in _new.items():
        if _fid in _idx and _vals.get(_fid) != _nv:
            _i = _idx[_fid]
            _spec = vocab.get(_fid) or {}
            _ref = _spec.get("ref") or {}
            out[_i]["_val"] = _nv
            out[_i]["symptom"] = (
                f"{_spec.get('name_cn', _fid)} {_nv} {_spec.get('unit', '')}"
                + (f"(参考 {_ref.get('low')}–{_ref.get('high')})" if _ref else "")).strip()
    if _log:
        log.debug("reconcile[%s] %s", case_id, " | ".join(_log))
    return _strip_private(out)


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
                                 declared=bool(p.get("declared", True)))
                    out.append(_entry(case_id, spec, p, day, val, k))
                    continue
                elif traj == "fluctuating" and k_on >= 0 and k_on % 2 == 1:
                    d = "normal"
                elif (traj == "treatment_responsive" and k == len(days) - 1
                      and k_on >= 1):
                    d = "normal"
                val = _value(spec, d, p.get("magnitude"), case_id, p["id"], k,
                             declared=bool(p.get("declared", True)))
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

#: Which Friedewald item is recomputed first (LDL is the one labs compute).
_DERIVE_RANK = {"LDL": 0, "TC": 1, "HDL": 2, "TG": 3}

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
                if d > NA_CL_GAP[1]:
                    nv = _m.ceil((v["Na"] - _tgt + _M) * 10) / 10
                else:
                    nv = _m.floor((v["Na"] - _tgt - _M) * 10) / 10
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


def _entry(case_id: str, spec: dict, p: dict, day: int, val, k: int) -> dict:
    ref = spec.get("ref") or {}
    txt = (f"{spec['name_cn']} {val} {spec.get('unit', '')}"
           + (f"(参考 {ref.get('low')}–{ref.get('high')})" if ref else ""))
    return {"evidence_id": f"EV-{case_id}-L{p['id']}{k}",
            "source_type": "lab_result",
            "source_timestamp": int(day),
            "symptom": txt.strip(),
            "claim_supported": True,
            "relevance": "routine_panel",
            # internal; removed by `_strip_private`
            "_fid": p["id"], "_val": val}
