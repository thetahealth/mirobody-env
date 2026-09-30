"""relations.py -- literature-anchored relations between indicators, used as validators.

Gold comes from the case's latent variables; these relations only reject a surface that
is internally inconsistent. `check_*` relations are decided per case, `cohort_*` relations
across a batch (over-coupling is visible only there).

All formulas use the units registered in `registry/findings.yaml`, not the source papers':

    Ca / LDL / TC / HDL / TG / FBG   mmol/L
    Alb                              g/L
    Cr                               umol/L
    ALT / AST                        U/L
    HbA1c                            %
    TSH                              mIU/L      FT4  pmol/L
    Na / Cl / K                      mmol/L

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ── unit-conversion constants ──────────────
# Cholesterol class mg/dL -> mmol/L divisor (molecular weight 386.65 / 10)
CHOL_MGDL_PER_MMOL = 38.67
#: Triglyceride mg/dL -> mmol/L divisor (molecular weight 885.7 / 10)
TG_MGDL_PER_MMOL = 88.57
#: Calcium mmol/L -> mg/dL (atomic weight 40.08, divalent => 1 mmol/L = 4.008 mg/dL)
CA_MGDL_PER_MMOL = 4.008
#: Creatinine mg/dL -> umol/L (molecular weight 113.12 => 1 mg/dL = 88.4 umol/L)
CR_UMOL_PER_MGDL = 88.4
#: Glucose mg/dL -> mmol/L (molecular weight 180.16 / 10)
GLU_MGDL_PER_MMOL = 18.016


@dataclass(frozen=True)
class Check:
    """The verdict of one relation on one case.

    `ok is None` means the relation does not apply (e.g. Friedewald when TG >= 4.5), which is
    different from `ok is False`.
    """
    relation: str
    ok: bool | None
    detail: str
    residual: float | None = None


# ══════════════════════════════════════════════════════════════════════════
# R1 - ADAG: HbA1c <-> average glucose
# ══════════════════════════════════════════════════════════════════════════
#: Nathan DM et al., *Translating the A1C assay into estimated average glucose values*,
#: Diabetes Care 2008 · PMID 18540046 · DOI 10.2337/dc08-0545
#: Verbatim from the abstract: "Linear regression analysis between the A1C and AG
#: values provided the tightest correlations (AG(mg/dl) = 28.7 x A1C - 46.7,
#: R(2) = 0.84)"
#: n=507 (T1D 268 / T2D 159 / non-diabetic 80), roughly 2700 glucose points per person.
ADAG_SLOPE, ADAG_INTERCEPT = 28.7, -46.7
#: R^2 = 0.84 => r = 0.917, the tightest correlation A1c reaches with average glucose;
#: r(A1c, FBG) must be lower because fasting glucose is noisier.
ADAG_R_CEILING = math.sqrt(0.84)


def adag_eag_mmol(hba1c_pct: float) -> float:
    """Estimate average glucose (mmol/L) from HbA1c (%). Note: average glucose,
    not fasting glucose."""
    return (ADAG_SLOPE * hba1c_pct + ADAG_INTERCEPT) / GLU_MGDL_PER_MMOL


# ══════════════════════════════════════════════════════════════════════════
# R2 - Friedewald: the LDL identity
# ══════════════════════════════════════════════════════════════════════════
#: Friedewald WT, Levy RI, Fredrickson DS. *Clin Chem* 1972;18(6):499
#: PMID 4337382 · DOI 10.1093/clinchem/18.6.499
#: Original formula (mg/dL): LDL = TC - HDL - TG/5.
#: SI-form divisor = 5 x CHOL/TG conversion ratio = 5 x 38.67/88.57 = 2.183 ~= 2.2.
FRIEDEWALD_TG_DIVISOR = 2.2
#: Domain boundary: not applicable when TG >= 400 mg/dL (= 4.5 mmol/L).
FRIEDEWALD_TG_MAX_MMOL = 4.5


def friedewald_ldl(tc: float, hdl: float, tg: float) -> float | None:
    """Compute LDL from TC/HDL/TG (all mmol/L). Returns None when TG >= 4.5
    (out of the formula's domain)."""
    if tg >= FRIEDEWALD_TG_MAX_MMOL:
        return None
    return tc - hdl - tg / FRIEDEWALD_TG_DIVISOR


def check_friedewald(tc: float, hdl: float, tg: float, ldl: float,
                     tol_mmol: float) -> Check:
    """Whether LDL is consistent with TC/HDL/TG. `tol_mmol` has no default on purpose."""
    pred = friedewald_ldl(tc, hdl, tg)
    if pred is None:
        return Check("R2", None, f"TG={tg} ≥ {FRIEDEWALD_TG_MAX_MMOL} mmol/L, Friedewald does not apply")
    res = ldl - pred
    return Check("R2", abs(res) <= tol_mmol,
                 f"LDL observed={ldl:.3f} predicted={pred:.3f} residual={res:+.3f} (tolerance ±{tol_mmol})", res)


# ══════════════════════════════════════════════════════════════════════════
# R3 - Payne corrected calcium
# ══════════════════════════════════════════════════════════════════════════
# Payne RB, Little AJ, Williams RB, Milner JR.
# *Interpretation of serum calcium in patients with abnormal serum proteins*,
# BMJ 1973;4(5893):643 · PMID 4758544 · DOI 10.1136/bmj.4.5893.643
# "Adjusted calcium = calcium - albumin + 4.0, where calcium is in mg/100 ml and
# albumin in g/100 ml" => coefficient 1.0 (not the often-quoted 0.8).
# SI derivation: Ca_adj(mmol/L) = Ca - Alb(g/L)/40.08 + 4.0/4.008
#                               = Ca + (1/40.08)*(40 - Alb) ~= Ca + 0.025*(40 - Alb)
PAYNE_COEF_PER_G_PER_L = 1.0 / (CA_MGDL_PER_MMOL * 10.0)   # ~= 0.02495
#: Payne's anchor albumin is 40 g/L, while the registered Alb range is 35-55 g/L, so the
#: correction shifts typical patients. Callers pass `anchor` explicitly; any value other
#: than 40.0 is no longer Payne's formula.
PAYNE_ANCHOR_ORIGINAL_G_PER_L = 40.0


def payne_corrected_ca(ca_mmol: float, alb_g_per_l: float, anchor_g_per_l: float) -> float:
    """Payne-corrected calcium (mmol/L); see `PAYNE_ANCHOR_ORIGINAL_G_PER_L` for the anchor."""
    return ca_mmol + PAYNE_COEF_PER_G_PER_L * (anchor_g_per_l - alb_g_per_l)


def check_corrected_ca(ca_mmol: float, alb_g_per_l: float, ref_lo: float, ref_hi: float,
                       anchor_g_per_l: float = PAYNE_ANCHOR_ORIGINAL_G_PER_L) -> Check:
    """Whether (Ca, Alb) is self-consistent: the corrected calcium must stay near its reference range."""
    adj = payne_corrected_ca(ca_mmol, alb_g_per_l, anchor_g_per_l)
    lo, hi = ref_lo - 0.5 * (ref_hi - ref_lo), ref_hi + 0.5 * (ref_hi - ref_lo)
    return Check("R3", lo <= adj <= hi,
                 f"Ca={ca_mmol:.2f} Alb={alb_g_per_l:.0f} → corrected Ca={adj:.3f}"
                 f" (allowed {lo:.2f}–{hi:.2f}, anchor {anchor_g_per_l:.0f} g/L)", adj)


# ══════════════════════════════════════════════════════════════════════════
# R4 - CKD-EPI 2021 (race-free, creatinine-based)
# ══════════════════════════════════════════════════════════════════════════
#: Inker LA et al., *New Creatinine- and Cystatin C-Based Equations to Estimate GFR
#: without Race*, NEJM 2021 · PMID 34554658 · DOI 10.1056/nejmoa2102953
#: Equation taken from the OA full text PMC13074197 (which cites Inker 2021),
#: verbatim:
#:   eGFR = 142 x min(Scr/kappa,1)^alpha x max(Scr/kappa,1)^(-1.200) x 0.9938^Age x [1.012 if female]
#:   kappa = 0.7 female / 0.9 male ; alpha = -0.241 female / -0.302 male ; Scr in mg/dL
CKDEPI = {"female": {"kappa": 0.7, "alpha": -0.241, "sex_factor": 1.012},
          "male":   {"kappa": 0.9, "alpha": -0.302, "sex_factor": 1.0}}


def ckd_epi_2021_egfr(cr_umol_l: float, age: float, sex: str) -> float | None:
    """eGFR (mL/min/1.73m^2). `sex` takes 'female'/'male'; any other value returns
    None (no guessing)."""
    p = CKDEPI.get(str(sex).lower())
    if p is None or cr_umol_l <= 0 or age <= 0:
        return None
    scr = cr_umol_l / CR_UMOL_PER_MGDL                      # umol/L -> mg/dL
    ratio = scr / p["kappa"]
    return (142.0 * (min(ratio, 1.0) ** p["alpha"]) * (max(ratio, 1.0) ** -1.200)
            * (0.9938 ** age) * p["sex_factor"])


# ══════════════════════════════════════════════════════════════════════════
# R5 - TSH <-> FT4 forbidden combinations
# ══════════════════════════════════════════════════════════════════════════
# Guidelines define states, not a quantitative TSH-FT4 relationship, so only two
# combinations are forbidden. Sources: ETA 2013 subclinical hypothyroidism guideline,
# PMID 24783053 · DOI 10.1159/000356507; ATA 2014 hypothyroidism treatment guideline,
# PMID 25266247 · DOI 10.1089/thy.2014.0028.
# Central causes (TSH-secreting adenoma, thyroid hormone resistance) must be tagged exempt.
TSH_FT4_EXEMPT_TAGS = frozenset({"tsh_secreting_adenoma", "thyroid_hormone_resistance",
                                 "central_hypothyroidism"})


def _dir(v: float, lo: float, hi: float) -> str:
    return "high" if v > hi else ("low" if v < lo else "normal")


def check_tsh_ft4(tsh: float, ft4: float, tsh_ref: tuple[float, float],
                  ft4_ref: tuple[float, float], exempt: bool = False) -> Check:
    """Forbids `TSH-up AND FT4-up` and `TSH-down AND FT4-down`. Returns ok=None
    (not applicable) when `exempt=True`."""
    dt, df = _dir(tsh, *tsh_ref), _dir(ft4, *ft4_ref)
    if exempt:
        return Check("R5", None, f"TSH={dt} FT4={df}; annotated central/resistance, exempt")
    bad = (dt == "high" and df == "high") or (dt == "low" and df == "low")
    return Check("R5", not bad,
                 f"TSH={tsh}({dt}) FT4={ft4}({df})"
                 + ("  ← same direction, not a normal thyroid-axis state" if bad else ""))


# ══════════════════════════════════════════════════════════════════════════
# R6 - De Ritis ratio · R7 - Na-Cl difference
# ══════════════════════════════════════════════════════════════════════════
#: De Ritis F, Coltorti M, Giusti G. *An enzymic test for the diagnosis of viral
#: hepatitis: the transaminase serum activities*, 1957;
#: reprinted Clin Chim Acta 2006 · PMID 16781697 · DOI 10.1016/j.cca.2006.05.001
def de_ritis(ast: float, alt: float) -> float | None:
    """AST/ALT ratio. Returns None when ALT <= 0 (no division by zero, no
    guessing)."""
    return None if alt <= 0 else ast / alt


#: Na-Cl difference envelope (the anion gap needs HCO3, which is not in the indicator
#: table). The band is empirical, so callers pass it explicitly.
def check_na_cl(na: float, cl: float, lo: float, hi: float) -> Check:
    d = na - cl
    return Check("R7", lo <= d <= hi, f"Na−Cl={d:.1f} (allowed {lo}–{hi})", d)


# ══════════════════════════════════════════════════════════════════════════
# Cohort relations: over-coupling is only visible at this level
# ══════════════════════════════════════════════════════════════════════════
def pearson(xs, ys) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my)
                                               for a, b in zip(xs, ys)) / (sx * sy)


#: Smallest cohort `cohort_glucose_scales_codirectional` will read. Below it the sign of r is noise.
R8_MIN_CASES = 10
#: One-sided 5% normal quantile; `r > z / sqrt(n)` is the large-sample test of r > 0.
R8_Z_ONE_SIDED = 1.645


def cohort_glucose_scales_codirectional(d_hba1c_per_case, d_fbg_per_case) -> Check:
    """R8 · the two glucose scales move the same way across a cohort.

    Per case, the first-to-last change of HbA1c and of fasting glucose; across cases
    `r(dHbA1c, dFBG)` must exceed `1.645 / sqrt(n)` (one-sided p < 0.05). A world that moves
    one scale without the other fails.
    """
    xs, ys = list(d_hba1c_per_case), list(d_fbg_per_case)
    if len(xs) < R8_MIN_CASES:
        return Check("R8-cohort", None, f"too few cases ({len(xs)} < {R8_MIN_CASES})")
    r = pearson(xs, ys)
    if r is None:
        return Check("R8-cohort", None, "zero variance")
    bar = R8_Z_ONE_SIDED / math.sqrt(len(xs))
    return Check("R8-cohort", r > bar,
                 f"r(ΔHbA1c,ΔFBG)={r:+.4f} vs chance bound {bar:.3f} (n={len(xs)} cases)"
                 + ("  ← the two glucose scales move apart or not at all: a rendering defect" if r <= bar else ""), r)


def cohort_a1c_fbg_ceiling(hba1c_per_case, fbg_per_case) -> Check:
    """Cohort `r(HbA1c, FBG)` must not exceed Nathan 2008's ceiling of 0.917.

    One mean per case, then correlate across cases, matching Nathan's between-subject design.
    Passing this check says nothing about the other indicator pairs.
    """
    xs, ys = list(hba1c_per_case), list(fbg_per_case)
    # Below `R8_MIN_CASES` a high between-case r can arise by chance: record "not measured".
    if len(xs) < R8_MIN_CASES:
        return Check("R1-cohort", None, f"too few cases ({len(xs)} < {R8_MIN_CASES})")
    r = pearson(xs, ys)
    if r is None:
        return Check("R1-cohort", None, "too few cases or zero variance")
    return Check("R1-cohort", r <= ADAG_R_CEILING,
                 f"r(HbA1c,FBG)={r:+.4f} vs Nathan ceiling {ADAG_R_CEILING:.4f}"
                 f"(n={len(xs)} case)"
                 + ("  ← above the published ceiling: over-coupled" if r > ADAG_R_CEILING else ""), r)
