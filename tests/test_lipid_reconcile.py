"""Friedewald reconciliation of the lipid panel and the shared-cause expansion that feeds it.

`reconcile_panel` makes LDL = TC - HDL - TG/2.2 (mmol/L) hold on the routine panel. It may
move an undeclared item to carry the identity, but it never moves a rendered low or high
across its reference boundary, never derives TG into TG >= 4.5 (outside the formula's
domain), and writes nothing when no free item can carry the identity. The shared-cause
expansion (`_disease_partner_direction`) decides which lipid items a declared abnormality
pulls along; an inverse partner follows a declared rise only.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import re

import pytest

from haenv import findings_render as FR
from haenv import relations as R
from haenv import rng
from haenv.registry import load_findings

V = {"TC": {"ref": {"low": 2.8, "high": 5.17}}, "HDL": {"ref": {"low": 1.04, "high": 2.07}},
     "TG": {"ref": {"low": 0.4, "high": 1.7}}, "LDL": {"ref": {"low": 0.0, "high": 3.37}}}


def _ok(v):
    return R.check_friedewald(v["TC"], v["HDL"], v["TG"], v["LDL"], tol_mmol=0.1).ok


# ------------------------------------------------------------------ side of the reference range
def test_low_tc_with_high_hdl_is_left_broken_rather_than_normalised():
    """Declared low TG with low TC and high HDL: LDL would be negative, and TC could carry
    the identity only by leaving its low side. Nothing is written; the quad stays broken."""
    base = {"TC": 2.27, "HDL": 2.17, "TG": 0.25, "LDL": 2.48}
    v, log = FR.reconcile_panel(dict(base), {"TG"}, V)
    assert v == base
    assert any("unsatisfiable" in x and "越过参考界" in x for x in log)
    assert _ok(v) is False


def test_a_high_tc_is_not_rewritten_into_the_normal_range():
    base = {"TC": 14.0, "HDL": 0.5, "TG": 1.0, "LDL": 2.0}
    v, log = FR.reconcile_panel(dict(base), set(), V)
    assert v["TC"] == 14.0 and any("unsatisfiable" in x for x in log)


def test_a_rendered_high_ldl_stays_high_and_tc_carries_the_identity():
    v, log = FR.reconcile_panel({"TC": 3.2, "HDL": 1.6, "TG": 1.0, "LDL": 4.0}, set(), V)
    assert v["LDL"] == 4.0 and v["TC"] > 5.17 and _ok(v)
    assert any("R2 由恒等式推出 TC" in x for x in log)


def test_a_normal_ldl_follows_the_identity_out_of_range():
    """Control: the side rule binds rendered abnormalities only; a normal LDL takes whatever
    the identity gives, high included."""
    v, _ = FR.reconcile_panel({"TC": 6.5, "HDL": 1.0, "TG": 1.0, "LDL": 3.0}, set(), V)
    assert v["LDL"] > 3.37 and v["TC"] == 6.5 and _ok(v)


# ------------------------------------------------------------------ which free item carries it
def test_tc_carries_the_identity_before_hdl_when_both_could():
    base = {"TC": 3.9, "HDL": 1.6, "TG": 1.0, "LDL": 1.5}
    v, _ = FR.reconcile_panel(dict(base), {"LDL"}, V)
    assert v["HDL"] == base["HDL"] and v["TC"] != base["TC"] and _ok(v)


def test_hdl_carries_the_identity_when_a_low_tc_cannot():
    base = {"TC": 2.5, "HDL": 1.2, "TG": 0.5, "LDL": 1.5}
    v, log = FR.reconcile_panel(dict(base), {"LDL"}, V)
    assert v["TC"] == 2.5 and v["HDL"] != base["HDL"] and _ok(v)
    assert any("R2 由恒等式推出 HDL" in x and "TC=" in x for x in log)


# ------------------------------------------------------------------ TG domain
def test_a_derived_tg_inside_the_plausible_band_but_at_or_above_4_5_is_not_written():
    """TC, HDL and LDL declared; the only free item is TG and the identity asks for 4.80,
    which is plausible for TG but outside Friedewald's domain."""
    base = {"TC": 5.68, "HDL": 1.0, "TG": 2.0, "LDL": 2.5}
    v, log = FR.reconcile_panel(dict(base), {"LDL", "TC", "HDL"}, V)
    assert v == base
    assert any("定义域" in x for x in log) and not any("出合理带" in x for x in log)


def test_a_derived_tg_inside_the_domain_is_written():
    v, _ = FR.reconcile_panel({"TC": 5.68, "HDL": 1.0, "TG": 2.0, "LDL": 3.0}, {"LDL", "TC", "HDL"}, V)
    assert v["TG"] == pytest.approx(3.70, abs=0.01) and _ok(v)


# ------------------------------------------------------------------ shared-cause expansion
def test_inverse_partner_follows_a_declared_rise_only():
    assert FR._disease_partner_direction("high", +1.0) == "high"
    assert FR._disease_partner_direction("low", +1.0) == "low"
    assert FR._disease_partner_direction("high", -1.0) == "low"
    assert FR._disease_partner_direction("low", -1.0) is None


def test_abnormality_pool_takes_the_same_expansion(monkeypatch):
    import haenv.registry as REG

    def pool_for(direction):
        monkeypatch.setattr(REG, "condition_findings_for_case",
                            lambda *a, **k: {"X": {"findings": [{"id": "TG", "direction": direction}]}})
        monkeypatch.setattr(FR, "_DECOY_POOL", None)
        return set(FR._disease_abnormality_pool())

    assert pool_for("low") == {("TG", "low"), ("TC", "low")}
    assert pool_for("high") == {("TG", "high"), ("TC", "high"), ("HDL", "low")}


_NUM = re.compile(r"^(.+?)\s+(-?\d+(?:\.\d+)?)\s")


def _first_draw(case_id, direction):
    vocab = FR.with_sex_ref(load_findings(), "F")
    prof = {"findings": ({"id": "TG", "direction": direction, "magnitude": "moderate",
                          "role": "screening", "trajectory": "stable", "n": 1},)}
    names = {vocab[f]["name_cn"]: f for f in ("TC", "HDL", "TG", "LDL")}
    out: dict = {}
    for e in FR.render_routine_panel(case_id, prof, load_findings(), 200, "F"):
        m = _NUM.match(str(e.get("symptom", "")).strip())
        if m and m.group(1).strip() in names:
            out.setdefault(int(e["source_timestamp"]), {})[names[m.group(1).strip()]] = float(m.group(2))
    d0 = min(k for k, p in out.items() if len(p) == 4)
    return out[d0], vocab


def _no_incidentals(n):
    ids = [f"LIP-{k}" for k in range(400) if int(rng.unit(f"LIP-{k}", "incidental", "n") * 3) == 0]
    assert len(ids) >= n
    return ids[:n]


def test_declared_low_tg_renders_low_tc_normal_hdl_and_a_consistent_panel():
    for cid in _no_incidentals(12):
        p, vocab = _first_draw(cid, "low")
        assert p["TC"] < vocab["TC"]["ref"]["low"], (cid, p)
        assert vocab["HDL"]["ref"]["low"] <= p["HDL"] <= vocab["HDL"]["ref"]["high"], (cid, p)
        assert _ok(p), (cid, p)


def test_declared_high_tg_still_pulls_hdl_low():
    for cid in _no_incidentals(12):
        p, vocab = _first_draw(cid, "high")
        assert p["TC"] > vocab["TC"]["ref"]["high"] and p["HDL"] < vocab["HDL"]["ref"]["low"], (cid, p)
