"""Contract groups E and F.

E -- the gold standard is answerable: every gold driver class can be emitted with visible
evidence on some shipped specification, and every real reversal point is a visible change in
the course (on the grading side).

F -- determinism and individual variation: the same specification under the same case id
builds byte-identically; under another case id, the per-person draws differ and stay inside
their declared bands.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations
from _patch_bound import patch_bound  # noqa: E402

import json

import pytest
import yaml

import _cohort as C

pytestmark = pytest.mark.world

GEN14_KINDS = {"gold_not_coverable", "gold_evidence_flat_pre_T", "driver_not_in_coverage_map",
               "gold_lab_signal_missing"}


def _regain_specs():
    out = []
    for job in (C.EW20_JOB, C.T2D_JOB, C.EXAMPLE_JOB):
        for c in yaml.safe_load(job.read_text(encoding="utf-8"))["cases"]:
            lat = c.get("latent") or {}
            if lat.get("outcome") == "regain" and lat.get("driver"):
                out.append((job, c["case_id"], lat["driver"]))
    return out


# ------------------------------------------------------------------ E1 every driver is answerable
def _drivers():
    return sorted({d for _, _, d in _regain_specs()})


@pytest.mark.parametrize("driver", _drivers())
def test_e1_every_gold_driver_is_emitted_with_visible_evidence(driver):
    """At least one shipped regain specification with this driver is emitted, i.e. passes
    GEN14 (its evidence streams move before T). A driver that GEN14 refuses on every
    specification is structurally unanswerable: the pack can never ask about it.

    Catches: a driver whose registered evidence never moves pre-T in the world layer. Turns
    red when a world change flattens a driver's evidence."""
    refused = _e1_refusals(driver, C.build)
    if refused:
        pytest.fail(f"{driver}: no shipped regain spec emitted; refusals {refused}")


def _e1_refusals(driver, build):
    """None when some shipped regain spec with this driver is emitted, else the refusals."""
    refused = {}
    for job, cid, d in _regain_specs():
        if d != driver:
            continue
        b = build(C.spec_of(job, cid), template=job)
        if b.emitted:
            return None
        for k in (b.audit.get("post_noise_conflicts") or ["not emitted"]):
            refused[f"{cid}:{k}"] = 1
    return sorted(refused)


def test_e1_negative_control_unobservable_driver(monkeypatch):
    """Negative control: when a driver's required evidence is a stream no case carries, GEN14
    refuses every spec with it and E1 must report the driver."""
    from haenv import gates as G
    monkeypatch.setitem(G.DRIVER_REQUIRED_OBSERVABLE, "calorie_intake_change", {"no_such_stream"})
    assert _e1_refusals("calorie_intake_change", C.build_uncached), (
        "E1 still passed with the driver's evidence made unobservable")


#: RC-EW-06 frozen with a course that ends before its reversal (course 105 < reversal day 112),
#: so it is refused. Independent of the shipped job file: a broken shipped RC-EW-06 turns E1 red
#: while this control stays as it is.
_RC_EW06_FROZEN = {
    "case_id": "RC-EW-06",
    "raw": {"disease": "obesity", "drug": "semaglutide", "dose_steps": [0.25, 0.5, 1.0, 1.7],
            "age_range": "55-59", "sex": "M", "devices": ["smart_scale", "wearable"],
            "start_weight": 96.0, "nadir_weight": 88.0, "bmi": 32.0,
            "symptoms": [{"day": 10, "text": "恶心", "context": "0.5mg 起始"},
                         {"day": 35, "text": "呕吐 2 次/周", "context": "1.0mg 后"},
                         {"day": 70, "text": "食欲部分回归", "context": "漏注射后"}]},
    "latent": {"index_time_T": 56, "course_end_day": 105, "outcome": "regain",
               "driver": "medication_intolerance", "reversal_week": 16, "adherence_low": 0.55,
               "distractor_level": "low",
               "event_density": {"measure_per_week": 7, "dosing_per_week": 1, "symptom_rate": 0.38,
                                 "life_event_rate": 0.12, "clinical_symptoms_recorded": 3,
                                 "course_weeks": 15.0}},
}


def _assert_mi_emitted_with_evidence(b, T: int):
    assert b.emitted, f"{b.case_id} not emitted ({b.audit.get('post_noise_conflicts')})"
    gi = [v for d, v in C.series(b.raw.longitudinal_data, "gi_symptom_score").items() if d <= T]
    assert gi and max(gi) - min(gi) >= 2.0, f"gi_symptom_score flat before T: {gi}"


def test_e1_medication_intolerance_is_emittable_when_the_spec_is_consistent():
    """Separates a world defect from a content gap for E1[medication_intolerance]: a frozen
    RC-EW-06 with its course made long enough is emitted and its registered evidence
    (`gi_symptom_score`) moves before T, whatever the shipped job files say."""
    b = C.build(C.variant(_RC_EW06_FROZEN, course_end_day=270), template=C.EW20_JOB)
    _assert_mi_emitted_with_evidence(b, 56)


def test_e1_medication_intolerance_frozen_inconsistent_spec_is_refused():
    """Negative control for the one above: the frozen spec unchanged (course ends before its
    reversal) is refused, so the control's green comes from the course fix."""
    assert not C.build(_RC_EW06_FROZEN, template=C.EW20_JOB).emitted


def test_e1_medication_intolerance_shipped_rc_ew06_is_emittable():
    """RC-EW-06 as shipped is emitted with visible evidence before T."""
    _assert_mi_emitted_with_evidence(C.build(C.spec_of(C.EW20_JOB, "RC-EW-06"), template=C.EW20_JOB), 56)


# ------------------------------------------------------------------ E2 reversal is a visible change
def _slope(w: dict, a: int, z: int):
    """Least-squares weight slope in kg/week over days [a, z]; None under five readings."""
    pts = [(d, v) for d, v in w.items() if a <= d <= z]
    if len(pts) < 5:
        return None
    n = len(pts)
    mx = sum(d for d, _ in pts) / n
    my = sum(v for _, v in pts) / n
    sxx = sum((d - mx) ** 2 for d, _ in pts)
    return 7 * sum((d - mx) * (v - my) for d, v in pts) / sxx if sxx else None


@pytest.mark.parametrize("job,cid,driver", _regain_specs()[:8])
def test_e2_real_reversal_is_a_change_in_weight_slope(job, cid, driver):
    """Around each real reversal point, the weekly weight slope over the four weeks after is
    at least 0.1 kg/week higher than over the four weeks before (the regain begins there).

    Evaluated on the ground-truth series: the observation noise (day-to-day SD about 0.7 % of body
    mass) moves a four-week slope by about 0.1 kg/week on its own, and it is not the renderer.

    Catches: a reversal point declared in the gold standard that the rendered course does not
    carry. Turns red when the weight renderer stops honouring `reversal_week`."""
    b = C.build(C.spec_of(job, cid), template=job, capture=True)
    if not b.emitted:
        pytest.skip(f"{cid} not emitted: covered by E1 per driver")
    w = C.series(b.layers.get("truth") or b.raw.longitudinal_data, "weight")
    for rp in b.raw.reversal_points:
        if rp.get("type") != "real":
            continue
        d = int(rp["week"]) * 7
        before, after = _slope(w, d - 28, d), _slope(w, d, d + 28)
        assert before is not None and after is not None, f"{cid}: too few readings around day {d}"
        assert after - before >= 0.1, f"{cid}: slope {before:+.2f} -> {after:+.2f} kg/wk around day {d}"


def test_e2_negative_control_flat_course():
    """Negative control: a course that keeps falling at the same rate through the declared
    reversal (no regain) must fail E2's measure; one that turns upward must pass it."""
    d = 100
    flat = {x: 95.0 - 0.05 * x for x in range(0, 200)}
    turn = {x: (95.0 - 0.05 * x if x <= d else 95.0 - 0.05 * d + 0.04 * (x - d)) for x in range(0, 200)}
    assert _slope(flat, d, d + 28) - _slope(flat, d - 28, d) < 0.1, "a course without regain passed E2"
    assert _slope(turn, d, d + 28) - _slope(turn, d - 28, d) >= 0.1, "a real turn failed E2"


# ------------------------------------------------------------------ F determinism and variation
def _dump(b):
    return json.dumps({"ld": b.raw.longitudinal_data, "ev": b.raw.evidence_ledger,
                       "rp": b.raw.reversal_points}, sort_keys=True, ensure_ascii=False, default=str)


def test_f1_same_spec_same_id_is_byte_identical():
    """Two builds of the same specification under the same case id are byte-identical
    (uncached, so both run the whole pipeline)."""
    base = C.spec_of(C.T2D_JOB, "T2G-01")
    a = C.build_uncached(C.variant(base))
    b = C.build_uncached(C.variant(base))
    assert a.emitted and b.emitted
    assert _dump(a) == _dump(b), "the same specification built twice differs: generation is not deterministic"


def _f2_draws(monkeypatch):
    """Build the spec under 12 other case ids, recording the response multiplier each build
    used (`build._drug_response_of`). Uncached, so the spy sees every build."""
    from haenv import build as B
    orig = B._drug_response_of
    used: dict = {}

    def spy(p):
        r = orig(p)
        used[str((p.meta or {}).get("case_id"))] = r
        return r
    patch_bound(monkeypatch, "_drug_response_of", source="haenv.build", value=spy)
    base = C.spec_of(C.T2D_JOB, "T2G-07")
    series, resp = set(), []
    for i in range(12):
        cid = f"F2-{i:02d}"
        b = C.build_uncached(C.variant(base, case_id=cid))
        if not b.emitted:
            continue
        series.add(json.dumps(b.raw.longitudinal_data.get("weight"), sort_keys=True))
        resp.append(used[cid])
    return series, resp


def test_f2_other_ids_differ_and_stay_in_band(monkeypatch):
    """Under other case ids the same specification yields different people (their weight
    series differ) whose drug response, as used by the build, stays inside the band and varies."""
    from haenv import drug_effects as DE
    _, (blo, bhi) = DE.response_bounds()
    series, resp = _f2_draws(monkeypatch)
    assert len(series) >= 10, f"only {len(series)} distinct weight series across 12 case ids"
    assert all(blo <= r <= bhi for r in resp), f"responses outside [{blo}, {bhi}]: {resp}"
    assert max(resp) - min(resp) > 0.2, f"responses barely vary: {resp}"


def test_f2_negative_control_response_ignores_the_person(monkeypatch):
    """Negative control: a build that draws every person's response under one fixed id must
    fail F2's spread."""
    from haenv import build as B
    from haenv import drug_effects as DE
    fixed = lambda p: DE.response_for("FIXED", (p.meta or {}).get("driver"), B._drug_of(p),
                                      (p.meta or {}).get("drug_response"))
    patch_bound(monkeypatch, "_drug_response_of", source="haenv.build", value=fixed)
    _, resp = _f2_draws(monkeypatch)
    assert resp and not max(resp) - min(resp) > 0.2, resp


def test_f1_negative_control_unseeded_draw(monkeypatch):
    """Negative control for `test_f1_same_spec_same_id_is_byte_identical`: with the per-case
    draws perturbed by an unseeded source, two emitted builds of the same spec differ.
    Builds withheld by the emission gate are redrawn (the positive test's precondition is
    `a.emitted and b.emitted`); the registry self-check runs before the patch, since it
    samples `rng.unit` itself and must see the unperturbed draws."""
    import random
    from haenv import rng
    from haenv.overlay import validate_registry
    validate_registry()                                     # cached; same loader as the build path
    orig = rng.unit
    monkeypatch.setattr(rng, "unit", lambda *p: (orig(*p) + random.random()) % 1.0)
    base = C.spec_of(C.T2D_JOB, "T2G-01")
    emitted = []
    for _ in range(20):
        b = C.build_uncached(C.variant(base))
        if b.emitted:
            emitted.append(b)
        if len(emitted) == 2:
            break
    assert len(emitted) == 2, (
        f"only {len(emitted)} of the perturbed builds passed the emission gate in 20 attempts")
    assert _dump(emitted[0]) != _dump(emitted[1]), "F1 passed with unseeded randomness in the pipeline"
