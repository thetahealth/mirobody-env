"""Traps in the gold standard must have visible evidence, and must count.

Three properties this pins:

1. `post_inject.carried_forward` never overwrites the reading a `transient_spike` raised
   (otherwise the gold standard keeps a trap on a reading the agent never saw spiked), and
   never copies the spike onto the next reading (a one-day artifact stays one day).
2. `noise._artifact_flags` records every artifact window, so a case with two artifacts
   records two windows.
3. `verifier.score_track_E` returns a score, not `E=None`, for a case with traps and no
   real reversal, so those traps reach the score.

A `unit_error` moves up to two readings; both count as its evidence, for the protection and
for the gate's baseline, and its trap records both days (section 3b).

Each property is pinned from both sides: the protection / gate / score must act where it should,
and must leave everything else alone.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

import noise as N                                            # noqa: E402
import verifier as V                                         # noqa: E402
from haenv import gates as G                                 # noqa: E402
from haenv import post_inject as PI                          # noqa: E402


def _series(days, spike_day=None, delta=3.6):
    return [{"ts": d, "value": 90.0 + (delta if d == spike_day else 0.0)} for d in days]


def _raw(days, spike_day=None):
    return SimpleNamespace(longitudinal_data={"weight": _series(days, spike_day)},
                           reversal_points=[], adjudication={})


# ------------------------------------------------------------------ 1. carried-forward
def test_carried_forward_leaves_the_trap_reading_alone():
    pts = _series(range(0, 30), spike_day=10)
    out = PI.carried_forward(pts, rng_key="k", rate=1.0, protect_ts=frozenset({10}))
    by = {p["ts"]: p["value"] for p in out}
    assert by[10] == 93.6, "the spiked reading was overwritten: the trap loses its evidence"
    assert by[11] == 90.0, "the spike was copied onto the next reading: a one-day artifact became two"


def test_carried_forward_still_acts_elsewhere():
    """Negative control: protection must not switch the layer off. At rate 1.0 every
    unprotected reading after the first copies its predecessor."""
    pts = [{"ts": d, "value": float(d)} for d in range(0, 30)]
    out = PI.carried_forward(pts, rng_key="k", rate=1.0, protect_ts=frozenset({10}))
    by = {p["ts"]: p["value"] for p in out}
    assert by[5] == 0.0 and by[20] == by[12], "carried-forward stopped acting outside the protected day"
    same = PI.carried_forward(pts, rng_key="k", rate=0.3)
    prot = PI.carried_forward(pts, rng_key="k", rate=0.3, protect_ts=frozenset({10}))
    differ = [a["ts"] for a, b in zip(same, prot) if a["value"] != b["value"]]
    assert set(differ) <= {10, 11, *range(12, 30)}, "protection changed decisions before the protected day"


# ------------------------------------------------------------------ 2. injector records
def test_spike_trap_sits_on_the_reading_it_changed():
    raw = _raw([0, 5, 9, 13, 20])                 # no reading on day 7
    N.transient_spike(raw, 7, 7)
    trap = [rp for rp in raw.reversal_points if rp["type"] == "trap"][0]
    spiked = [p["ts"] for p in raw.longitudinal_data["weight"] if p["value"] > 92]
    assert spiked == [trap["day"]] and trap["day"] in (5, 9), (
        "the trap is recorded on a day other than the spiked reading")
    assert raw.adjudication["artifact_flags"]["window"] == [trap["day"], trap["day"]]
    assert trap["jump"] == 3.6, "the injector did not record the jump the evidence gate reads"


def test_no_reading_no_trap():
    raw = _raw([])
    N.transient_spike(raw, 7, 7)
    assert not raw.reversal_points and "artifact_flags" not in raw.adjudication, (
        "a spike with nothing to spike still recorded a trap")


def test_all_artifact_windows_are_kept():
    raw = _raw(range(0, 100, 2))
    N.transient_spike(raw, 30, 30)
    N.transient_spike(raw, 70, 70)
    wins = [w["window"] for w in raw.adjudication["artifact_flags"]["windows"]]
    assert wins == [[30, 30], [70, 70]], f"windows lost: {wins}"
    assert raw.adjudication["artifact_flags"]["is_artifact_window"] is True


# ------------------------------------------------------------------ 3. emission gate
def _with_trap(days, spike_day, trap_day):
    raw = _raw(days, spike_day)
    raw.reversal_points.append({"week": trap_day // 7, "day": trap_day, "type": "trap",
                                "noise_class": "transient_spike", "flip": "risk_low->high",
                                "jump": 3.6})
    return raw


def test_gate_passes_a_visible_spike():
    assert G.check_trap_evidence(_with_trap(range(0, 40), 20, 20)) == []


def test_gate_catches_an_erased_spike():
    hits = G.check_trap_evidence(_with_trap(range(0, 40), None, 20))
    assert [h["kind"] for h in hits] == ["trap_without_evidence"], "an overwritten spike was not caught"


def test_gate_catches_a_missing_reading():
    hits = G.check_trap_evidence(_with_trap([d for d in range(0, 40) if d != 20], None, 20))
    assert [h["kind"] for h in hits] == ["trap_without_evidence"], "a trap on a missing reading was not caught"


def test_gate_ignores_window_artifacts():
    raw = _raw(range(0, 40))
    raw.reversal_points.append({"week": 2, "day": 14, "type": "trap",
                                "noise_class": "device_switch", "flip": "risk_low->high"})
    assert G.check_trap_evidence(raw) == [], "a window artifact was judged as a single reading"


# ------------------------------------------------------------------ 3b. two-reading unit error
def _unit_error_raw():
    """A `unit_error` over days 20-30 of a daily series; it moves days 20 and 21."""
    raw = SimpleNamespace(
        longitudinal_data={"weight": [{"ts": d, "value": 90.0 + (0.3 if d % 2 else -0.3)}
                                      for d in range(0, 40)]},
        reversal_points=[], adjudication={})
    N.unit_error(raw, 20, 30)
    return raw


def test_unit_error_both_readings_are_evidence():
    raw = _unit_error_raw()
    assert N.trap_evidence_days(raw) == {20, 21}, "the second moved reading is not protected"
    out = PI.carried_forward(raw.longitudinal_data["weight"], rng_key="k", rate=1.0,
                             protect_ts=N.trap_evidence_days(raw))
    by = {p["ts"]: p["value"] for p in out}
    moved = {p["ts"]: p["value"] for p in raw.longitudinal_data["weight"]}
    assert by[20] == moved[20] and by[21] == moved[21], "a moved reading was overwritten"
    assert by[22] < 91.0, "the second moved reading was copied onto the next one"


def test_unit_error_trap_day_only_protection_spreads_it():
    """Negative control: protecting only the trap day lets carried-forward copy the second
    reading onto the next one, so a two-reading artifact grows to three."""
    raw = _unit_error_raw()
    out = PI.carried_forward(raw.longitudinal_data["weight"], rng_key="k", rate=1.0,
                             protect_ts=frozenset({20}))
    assert {p["ts"]: p["value"] for p in out}[22] > 100.0


def test_gate_passes_a_visible_two_reading_unit_error():
    assert G.check_trap_evidence(_unit_error_raw()) == [], (
        "an intact two-reading unit error was refused: its second reading was taken as baseline")


def test_gate_catches_an_erased_two_reading_unit_error():
    raw = _unit_error_raw()
    for p in raw.longitudinal_data["weight"]:
        p["value"] = 90.0
    hits = G.check_trap_evidence(raw)
    assert [h["kind"] for h in hits] == ["trap_without_evidence"], "an erased unit error was not caught"


def test_gate_with_trap_day_only_takes_the_second_reading_as_baseline(monkeypatch):
    """Negative control: when only the trap day counts as moved, the second reading becomes
    the right-hand neighbour and the intact artifact reads as less than half its size."""
    raw = _unit_error_raw()
    monkeypatch.setattr(N, "trap_evidence_days", lambda r: frozenset(
        int(rp["day"]) for rp in r.reversal_points if rp.get("type") == "trap"))
    hits = G.check_trap_evidence(raw)
    assert [h["kind"] for h in hits] == ["trap_without_evidence"]


def test_unit_error_evidence_is_read_from_the_trap_alone():
    """The moved days are on the trap record, so a caller without `adjudication` gets the
    same protection and the same gate verdict."""
    full = _unit_error_raw()
    bare = SimpleNamespace(longitudinal_data=full.longitudinal_data,
                           reversal_points=full.reversal_points)
    assert N.trap_evidence_days(bare) == N.trap_evidence_days(full) == {20, 21}
    assert G.check_trap_evidence(bare) == [], "without adjudication the second reading became baseline"


def test_trap_record_without_days_covers_its_day_only():
    """Negative control: a trap recorded without `days` counts
    only its own day, and the gate then takes the second reading as baseline."""
    raw = _unit_error_raw()
    for rp in raw.reversal_points:
        rp.pop("days", None)
    assert N.trap_evidence_days(raw) == {20}
    assert [h["kind"] for h in G.check_trap_evidence(raw)] == ["trap_without_evidence"]


# ------------------------------------------------------------------ 4. Track E
def _vp(points):
    return SimpleNamespace(reversal_points=points)


def _traj(risks):
    return [{"day": d, "risk_cat": r, "top_driver": "x", "action": "A1"} for d, r in risks]


_TRAPS = [{"week": 6, "type": "trap", "flip": "risk_low->high"},
          {"week": 12, "type": "trap", "flip": "risk_low->high"}]


def test_trap_only_case_not_fooled_scores_full():
    out = V.score_track_E(_traj([(d, "low") for d in range(0, 120, 14)]), _vp(_TRAPS), 14)
    assert out["E"] == 1.0 and out["trap_fooled"] == 0


def test_trap_only_case_fooled_loses_credit():
    risks = [(d, "high" if 40 <= d <= 45 else "low") for d in range(0, 120, 7)]
    out = V.score_track_E(_traj(risks), _vp(_TRAPS), 14)
    assert out["trap_fooled"] == 1 and out["E"] == 0.75, out


def test_no_traps_no_reversal_stays_none():
    out = V.score_track_E(_traj([(d, "low") for d in range(0, 120, 14)]), _vp([]), 14)
    assert out["E"] is None, "a case with nothing to score started getting a Track E score"
