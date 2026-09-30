"""A kernel distractor on a calibrated stream must not delete that stream.

`core/noise.inject_distractors` writes unrelated metric streams into
`raw.longitudinal_data` before `events.inject` runs, and `distractor_level: low`
always names `steps`. If `plan_streams` skips every name already present as
"provided by the world layer", the inherited-stream pass removes the unplanned
upstream copy of every calibrated stream -- and `steps`, with the derived
`active_burn` / `calories_bmr` on its day grid, never reaches an emitted case.

Both sides are checked, at the unit and at the pipeline:
  * the rule frees exactly a calibrated distractor that is not gold evidence, and
    keeps world-layer evidence, non-distractor names and a non-calibrated
    distractor (`body_temp`, which survives as an inherited stream);
  * a real case built with the rule carries a calibrated `steps` render; the same
    case built with the unfiltered name set carries none.
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haenv import events as E                                    # noqa: E402
from haenv import wearable as wear                               # noqa: E402


def _raw(ld: dict, distractors: list[str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(longitudinal_data={k: [{"ts": 0, "value": 1}] for k in ld},
                                 adjudication={"distractor_signals": distractors})


def test_rule_frees_only_a_calibrated_non_gold_distractor():
    raw = _raw(["weight", "diet_carb_pct", "steps", "body_temp", "resting_hr"],
               ["steps", "body_temp"])
    got = E.world_layer_base_signals(raw, {"diet_carb_pct"})
    assert wear.is_calibrated("steps") and not wear.is_calibrated("body_temp")
    assert got == {"weight", "diet_carb_pct", "body_temp", "resting_hr"}


def test_rule_keeps_a_distractor_name_registered_as_gold_evidence():
    raw = _raw(["steps"], ["steps"])
    assert E.world_layer_base_signals(raw, {"steps"}) == {"steps"}


def test_rule_without_distractors_is_the_old_rule():
    raw = _raw(["weight", "steps"], [])
    assert E.world_layer_base_signals(raw, set()) == {"weight", "steps"}


def _build(case_id: str):
    from haenv import build as B
    from haenv.job import load_job
    job = load_job(ROOT / "inputs" / "ddx-timeline.job.yaml")
    cs = next(c for c in job.cases if c.case_id == case_id)
    return B.build_case(cs)


#: A ddx-timeline case with a wearable, `distractor_level: low`, and a device that
#: reports steps (`wear.stream_available`).
CASE = "JD-17v3"


def test_pipeline_carries_calibrated_steps_and_old_rule_does_not(monkeypatch):
    assert wear.stream_available(CASE, "steps")
    raw, audit = _build(CASE)
    assert raw is not None, audit.get("verify_bad")
    assert "distractor:low" in " ".join(audit.get("noise_applied") or [])
    pts = raw.longitudinal_data.get("steps") or []
    assert pts, "steps missing with the rule in place"
    # The render, not the kernel wave: the kernel copy sits on a two-day grid.
    gaps = {b["ts"] - a["ts"] for a, b in zip(pts, pts[1:])}
    assert 1 in gaps, f"steps on a {sorted(gaps)[:3]}-day grid is the kernel wave"
    assert raw.longitudinal_data.get("active_burn"), "derived child of steps missing"

    monkeypatch.setattr(E, "world_layer_base_signals",
                        lambda r, w=(): set(r.longitudinal_data or {}))
    raw_old, _ = _build(CASE)
    assert raw_old is not None
    assert not raw_old.longitudinal_data.get("steps"), (
        "the old rule no longer loses steps, so this control no longer shows the defect")
