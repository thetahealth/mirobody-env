"""Contract group D: laboratory relations hold on what the agent actually sees.

The renderer derives or repairs the routine panel so that Friedewald LDL, Payne-corrected
calcium and the Na-Cl gap hold (`findings_render`). These contracts re-check the relations on
the solver payload the production path emits (after rounding and every later step), and the
cohort-level glucose relations (R8 co-direction, R1 ADAG ceiling) on the T2D cases of the
shipped diagnosis pack.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import re

import pytest
import yaml

import _cohort as C
from haenv import relations as R
from haenv.findings_render import NA_CL_GAP

pytestmark = pytest.mark.world

DX_JOB = C.ROOT / "inputs" / "ddx-timeline.job.yaml"
_LAB = re.compile(r"^(?P<name>.+?) (?P<v>-?[\d.]+) (?P<unit>\S+?)[(（]参考 (?P<lo>[\d.]+)[–-](?P<hi>[\d.]+)[)）]")


@functools.lru_cache(maxsize=None)
def _names() -> dict[str, str]:
    """Display name or alias -> finding id, from `registry/findings.yaml`."""
    d = yaml.safe_load((C.ROOT / "registry" / "findings.yaml").read_text(encoding="utf-8"))
    items = d.get("findings") or d
    out = {}
    for fid, v in items.items():
        if isinstance(v, dict):
            for n in [v.get("name_cn"), *(v.get("aliases") or [])]:
                if n:
                    out[str(n)] = fid
    return out


def _panels(built) -> dict[int, dict[str, tuple[float, float, float]]]:
    """Solver-visible lab panel per draw day: fid -> (value, ref_lo, ref_hi)."""
    from build import build_instance                                  # kernel
    sp, _ = build_instance(built.raw, built.raw.prediction_context["prediction_time_T"])
    out: dict = {}
    for e in sp.evidence_ledger or []:
        if e.get("source_type") != "lab_result":
            continue
        m = _LAB.match(str(e.get("symptom") or ""))
        if not m or m["name"] not in _names():
            continue
        out.setdefault(int(e["source_timestamp"]), {})[_names()[m["name"]]] = (
            float(m["v"]), float(m["lo"]), float(m["hi"]))
    return out


@functools.lru_cache(maxsize=None)
def _dx_cases(limit: int = 20, t2d_only: bool = False):
    cases = yaml.safe_load(DX_JOB.read_text(encoding="utf-8"))["cases"]
    if t2d_only:
        cases = [c for c in cases if "T2D" in str((c.get("raw") or {}).get("disease"))
                 or "T2D" in str((c.get("raw") or {}).get("known_conditions"))]
    return [C.build(C.spec_of(DX_JOB, c["case_id"]), template=DX_JOB) for c in cases[:limit]]


def _violations(check_panel):
    bad, n = [], 0
    for b in _dx_cases():
        if not b.emitted:
            continue
        for day, p in _panels(b).items():
            res = check_panel(p)
            if res is None:
                continue
            n += 1
            if res.ok is False:
                bad.append(f"{b.case_id}@{day}: {res.detail}")
    return bad, n


# ------------------------------------------------------------------ D2 per-draw relations
def _r2(p):
    if not all(k in p for k in ("TC", "HDL", "TG", "LDL")):
        return None
    return R.check_friedewald(p["TC"][0], p["HDL"][0], p["TG"][0], p["LDL"][0], tol_mmol=0.1)


def _r3(p):
    if not all(k in p for k in ("Ca", "Alb")):
        return None
    return R.check_corrected_ca(p["Ca"][0], p["Alb"][0], p["Ca"][1], p["Ca"][2])


def _r7(p):
    if not all(k in p for k in ("Na", "Cl")):
        return None
    return R.check_na_cl(p["Na"][0], p["Cl"][0], *NA_CL_GAP)


@pytest.mark.parametrize("name,check", [("R2 Friedewald", _r2), ("R3 corrected calcium", _r3),
                                        ("R7 Na-Cl gap", _r7)])
def test_d2_panel_relations_hold_on_the_solver_payload(name, check):
    """Every draw on the solver payload that carries the items of a relation satisfies it
    (Friedewald within 0.1 mmol/L after rounding; corrected calcium inside its envelope;
    Na-Cl inside the gap the renderer enforces).

    Catches: a step after the renderer (rounding, measurement variation, a later layer)
    breaking a relation the renderer established. Turns red when such a step is added."""
    bad, n = _violations(check)
    assert n >= 10, f"{name}: only {n} draws carry the relation's items; the scan surface is too thin"
    assert not bad, f"{name}: {len(bad)}/{n} draws violate it, e.g. {bad[:3]}"


def test_d2_negative_control_broken_ldl():
    """Negative control: an LDL 0.5 mmol/L off Friedewald must be reported."""
    res = R.check_friedewald(4.0, 1.2, 1.1, 4.0 - 1.2 - 1.1 / 2.2 + 0.5, tol_mmol=0.1)
    assert res.ok is False


def test_d2_negative_control_broken_calcium_and_na_cl():
    """Negative controls for R3 and R7: calcium that is only normal because albumin is very
    low, and chloride equal to sodium, must both be reported."""
    assert R.check_corrected_ca(2.55, 20.0, 2.1, 2.6).ok is False
    assert R.check_na_cl(140.0, 140.0, *NA_CL_GAP).ok is False


# ------------------------------------------------------------------ D1 cohort glucose relations
def _t2d_changes():
    xs, ys, a_mean, f_mean = [], [], [], []
    for b in _dx_cases(limit=134, t2d_only=True):
        if not b.emitted:
            continue
        a = C.series(b.raw.longitudinal_data, "HbA1c")
        f = C.series(b.raw.longitudinal_data, "fasting_glucose")
        if len(a) < 2 or len(f) < 2:
            continue
        xs.append(a[max(a)] - a[min(a)])
        ys.append(f[max(f)] - f[min(f)])
        a_mean.append(sum(a.values()) / len(a))
        f_mean.append(sum(f.values()) / len(f))
    return xs, ys, a_mean, f_mean


def test_d1_glucose_scales_move_together_across_the_pack():
    """R8 on the diagnosis pack's cases with both glucose streams: r(dHbA1c, dFPG) above the
    one-sided chance bound (real diabetes EMR: r = +0.648).

    Uses a naturally varied cohort (different people and drugs), pooled: the pack
    has too few cases per drug for a within-drug test (R8 needs 10). A cohort of clones of one
    specification would carry no information: the clones share their weight course, so for a
    low responder the lab changes are measurement variation only."""
    xs, ys, _, _ = _t2d_changes()
    res = R.cohort_glucose_scales_codirectional(xs, ys)
    assert res.ok is not None, f"R8 not measurable: {res.detail}"
    assert res.ok, res.detail


def test_d1_hba1c_fpg_below_the_adag_ceiling():
    """R1: between-person r(mean HbA1c, mean FPG) does not exceed Nathan 2008's 0.917 --
    tighter than the tightest published coupling reads as over-coupling."""
    _, _, am, fm = _t2d_changes()
    res = R.cohort_a1c_fbg_ceiling(am, fm)
    assert res.ok is not None, f"R1 not measurable: {res.detail}"
    assert res.ok, res.detail


def test_d1_negative_control_noise_only_fpg():
    """Negative control: fasting glucose that moves by noise only fails R8."""
    import random
    rnd = random.Random(7)
    xs = [rnd.uniform(-2, 0) for _ in range(30)]
    ys = [rnd.gauss(0, 1) for _ in range(30)]
    assert R.cohort_glucose_scales_codirectional(xs, ys).ok is False


def test_d1_negative_control_over_coupled_means():
    """Negative control: per-case means on an exact line (r = 1) must exceed the ADAG ceiling."""
    xs = [6.0 + 0.1 * i for i in range(30)]
    assert R.cohort_a1c_fbg_ceiling(xs, [1.6 * x - 2.6 for x in xs]).ok is False
