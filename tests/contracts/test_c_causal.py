"""Contract group C: change one latent, and the channels that depend on it move the way
the mechanism says, by an amount inside the registered range; channels that do not depend
on it stay put.

Each pair of cases shares the case id, so every per-person draw is identical and only the
latent under test differs. Directions come from the mechanism definitions; magnitudes come
from the literature ranges registered with each mechanism.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pytest

import _cohort as C

pytestmark = pytest.mark.world

MAINTAINER = (C.T2D_JOB, "T2G-07")      # T2D, tirzepatide, maintains, three lab draws


def _pair(job, cid, a: dict, b: dict):
    base = C.spec_of(job, cid)
    return (C.build(C.variant(base, **a), template=job),
            C.build(C.variant(base, **b), template=job))


def _lab(b, sig, day):
    return C.series(b.raw.longitudinal_data, sig).get(day)


def _need(*bs):
    for b in bs:
        if not b.emitted:
            pytest.fail(f"{b.case_id} not emitted ({b.audit.get('post_noise_conflicts')})")


# ------------------------------------------------------------------ C1 adherence -> labs
def _registered_drugs():
    from haenv import drug_effects as DE
    out = []
    for drug in DE._doc().get("drugs", {}):
        try:
            tot = DE.total_effect(drug, None, "HbA1c")
        except Exception:                                   # noqa: BLE001
            continue
        if tot:
            out.append(drug)
    return out


@pytest.mark.parametrize("drug", _registered_drugs())
def test_c1a_steady_state_adherence_effect_within_literature_range(drug):
    """At steady state, 10 points of adherence lost cost 0.05 to 0.30 HbA1c points.
    Literature: +0.16 (Schectman 2002, PMID 12032108) and +0.24 (Egede 2014, PMID 24586059)
    per 10% adherence; no GLP-1RA-specific per-10% estimate exists (Hamersky 2019, PMID
    31054132), so the band is wide on both sides. Computed with the production
    `drug_effects.effect_at` at day 365 under constant adherence 0.9 versus 0.8 -- the
    literature compares sustained adherence levels, so the comparison is at steady state.
    `effect_at` returns the drug's direct term (the weight-mediated part is rendered through
    the weight series, which adherence does not move in this model), so this is the model's
    whole sensitivity of HbA1c to adherence.

    Catches: adherence no longer scaling the drug effect, or scaling it far outside the
    literature. Turns red when either changes."""
    from haenv import build as B
    from haenv import drug_effects as DE
    per_kg = B.CLINICAL_SPEC["HbA1c"][0]
    kw = dict(per_kg=per_kg, atten=B.CLINICAL_ATTEN, cohort="T2D")
    e9 = DE.effect_at(drug, "HbA1c", 365, 0.9, **kw)
    e8 = DE.effect_at(drug, "HbA1c", 365, 0.8, **kw)
    per10 = e8 - e9
    assert 0.05 <= per10 <= 0.30, f"{drug}: {per10:+.3f} HbA1c points per 10 adherence points"


def test_c1b_lower_adherence_raises_hba1c_in_the_same_person():
    """Same person (T2G-01, regain driven by poor adherence), adherence falling to 0.55
    instead of 0.85 after T: HbA1c at day 180 is higher, by one reporting step
    (0.1) at least, and the weight series is identical, so the difference is the adherence effect alone.

    Only this driver lets `adherence_low` reach the adherence series; for every other driver
    the generator floors adherence at 0.90 (`build.premise_spec`), because a maintainer with
    poor adherence would contradict its own outcome. The magnitude is not compared with the
    literature here: adherence only diverges after T, and HbA1c follows exposure with a
    half-time of about five weeks, so at day 180 the gap is still building (C1a checks the
    steady state)."""
    job, cid = C.T2D_JOB, "T2G-01"
    hi, lo = _pair(job, cid, {"adherence_low": 0.85}, {"adherence_low": 0.55})
    _need(hi, lo)
    w_hi, w_lo = C.series(hi.raw.longitudinal_data, "weight"), C.series(lo.raw.longitudinal_data, "weight")
    assert w_hi == w_lo, "the weight series differs between the adherence variants: the lab difference is confounded"
    d_a1c = round(_lab(lo, "HbA1c", 180) - _lab(hi, "HbA1c", 180), 2)
    assert d_a1c >= 0.1, f"HbA1c at day 180 moved {d_a1c:+.2f} when adherence fell 30 points"


def test_c1_negative_control_adherence_ignored(monkeypatch):
    """Negative control: with the drug effect no longer scaled by adherence, C1b's
    difference must vanish."""
    from haenv import drug_effects as DE
    orig = DE.effect_at
    monkeypatch.setattr(DE, "effect_at", lambda drug, sig, day, adherence, **k: orig(drug, sig, day, 1.0, **k))
    job, cid = C.T2D_JOB, "T2G-01"
    base = C.spec_of(job, cid)
    hi = C.build_uncached(C.variant(base, adherence_low=0.85), template=job)
    lo = C.build_uncached(C.variant(base, adherence_low=0.55), template=job)
    _need(hi, lo)
    assert abs(_lab(lo, "HbA1c", 180) - _lab(hi, "HbA1c", 180)) < 0.1, (
        "HbA1c still moved with adherence after adherence was cut out of the drug effect")


# ------------------------------------------------------------------ C2 individual response
def test_c2_higher_response_lowers_both_glucose_scales():
    """A higher declared response multiplies the drug effect on both HbA1c and fasting
    glucose: at day 90 both are lower for response 1.5 than for 0.6 (same person otherwise)."""
    job, cid = MAINTAINER
    weak, strong = _pair(job, cid, {"drug_response": 0.6}, {"drug_response": 1.5})
    _need(weak, strong)
    assert _lab(strong, "HbA1c", 90) < _lab(weak, "HbA1c", 90), "HbA1c did not follow the response"
    assert _lab(strong, "fasting_glucose", 90) < _lab(weak, "fasting_glucose", 90), (
        "fasting glucose did not follow the response")


@pytest.mark.parametrize("stream", ["steps", "resting_hr", "weight"])
def test_c2_response_does_not_move_unrelated_streams(stream):
    """The drug response acts on the glycaemic channels only; the wearable streams and the
    weight series (which the drug response does not enter) are identical."""
    job, cid = MAINTAINER
    weak, strong = _pair(job, cid, {"drug_response": 0.6}, {"drug_response": 1.5})
    _need(weak, strong)
    a, b = C.series(weak.raw.longitudinal_data, stream), C.series(strong.raw.longitudinal_data, stream)
    moved = [d for d in a if d in b and abs(a[d] - b[d]) > 1e-9]
    assert not moved, f"{stream} changed on {len(moved)} days when only the drug response changed"


def test_c2_negative_control_response_ignored(monkeypatch):
    """Negative control: with every person's response fixed at 1.0, the declared response
    no longer reaches the labs and C2's glucose comparison must fail."""
    from haenv import build as B
    monkeypatch.setattr(B, "_drug_response_of", lambda p: 1.0)
    job, cid = MAINTAINER
    base = C.spec_of(job, cid)
    weak = C.build_uncached(C.variant(base, drug_response=0.6), template=job)
    strong = C.build_uncached(C.variant(base, drug_response=1.5), template=job)
    _need(weak, strong)
    assert not _lab(strong, "HbA1c", 90) < _lab(weak, "HbA1c", 90), "C2 passed with the response cut out"


def test_c2_negative_control_response_leaks_into_steps(monkeypatch):
    """Negative control: a layer that writes the response into the step counts must be seen
    by C2's unrelated-stream check."""
    from haenv import build as B
    from haenv import post_inject as PI
    seen = {}
    orig_resp, orig_post = B._drug_response_of, PI.apply_post_injection

    def resp(p):
        seen["r"] = orig_resp(p)
        return seen["r"]

    def post(raw, **k):
        res = orig_post(raw, **k)
        for pt in raw.longitudinal_data.get("steps") or []:
            pt["value"] += 1000 * seen.get("r", 0)
        return res
    monkeypatch.setattr(B, "_drug_response_of", resp)
    monkeypatch.setattr(PI, "apply_post_injection", post)
    job, cid = MAINTAINER
    base = C.spec_of(job, cid)
    weak = C.build_uncached(C.variant(base, drug_response=0.6), template=job)
    strong = C.build_uncached(C.variant(base, drug_response=1.5), template=job)
    _need(weak, strong)
    a, b = C.series(weak.raw.longitudinal_data, "steps"), C.series(strong.raw.longitudinal_data, "steps")
    assert any(abs(a[d] - b[d]) > 1e-9 for d in a if d in b), "a response leak into steps went unnoticed"


# ------------------------------------------------------------------ C3 weight -> HbA1c
def _c3_ratio(build):
    """HbA1c difference at day 180 per kg of weight difference (days 140-160, the renderer's
    30-day lag), regain versus maintain, for one person whose drug response and adherence are
    the same in both variants: the driver is `calorie_intake_change`, which moves neither
    (`response_for` gives both variants the same multiplier; adherence is floored at 0.90 for
    every non-adherence driver). With T2G-01's own driver (poor adherence) the regain variant
    also loses adherence, and the ratio then mixes the adherence effect into the weight effect
    -- its negative control read 0.097 per kg with the weight coupling switched off."""
    job, cid = C.T2D_JOB, "T2G-01"
    base = C.spec_of(job, cid)
    regain = build(C.variant(base, driver="calorie_intake_change"), template=job)
    keep = build(C.variant(base, outcome="maintain", driver=None, reversal_week=None), template=job)
    _need(regain, keep)
    w_r, w_k = C.series(regain.raw.longitudinal_data, "weight"), C.series(keep.raw.longitudinal_data, "weight")
    lag = [d for d in w_r if 140 <= d <= 160 and d in w_k]
    dw = sum(w_r[d] - w_k[d] for d in lag) / len(lag)
    da = _lab(regain, "HbA1c", 180) - _lab(keep, "HbA1c", 180)
    assert dw > 1.0, f"regain variant did not regain ({dw:+.2f} kg)"
    return da / dw


def test_c3_weight_regain_raises_hba1c_within_range():
    """Same person, regain versus maintain, nothing else different: the HbA1c difference per
    kg of weight difference lies in 0.03 to 0.15. Literature: about 0.10 per kg lost
    (Gummesson 2017, PMID 28417575, trial-arm level); regain has no direct T2D estimate and is
    taken as 0.5 to 0.7 of that. The band fails on a sign
    error or a coupling far outside the literature."""
    r = _c3_ratio(C.build)
    assert 0.03 <= r <= 0.15, f"HbA1c {r:.3f} per kg of regain"


def test_c3_negative_control_weight_decoupled(monkeypatch):
    """Negative control: with HbA1c's per-kg coupling set to zero, C3's ratio must fall
    below its band."""
    from haenv import build as B
    monkeypatch.setitem(B.CLINICAL_SPEC, "HbA1c", (0.0, *B.CLINICAL_SPEC["HbA1c"][1:]))
    r = _c3_ratio(C.build_uncached)
    assert not 0.03 <= r <= 0.15, f"C3 still in band with the coupling off ({r:.3f} per kg)"


# ------------------------------------------------------------------ C4 distractor level
def _c4_pair(build):
    """Same person, same declared symptom rate (0.5/wk), distractor level none versus high.
    The rate is held fixed because the density gate refuses a high level at the shipped rate
    (the injected count would exceed the declared one). Distractors replace benign symptoms
    from the events layer instead of adding to them, so the total symptom count carries no
    signal; the pool-symptom count does."""
    job, cid = MAINTAINER
    base = C.spec_of(job, cid)
    ed = {**base["latent"]["event_density"], "symptom_rate": 0.5}
    low = build(C.variant(base, distractor_level="none", event_density=ed), template=job)
    high = build(C.variant(base, distractor_level="high", event_density=ed), template=job)
    _need(low, high)
    return low, high


def _n_distractor(b) -> int:
    """Symptom entries whose text comes from the kernel's distractor pool."""
    import noise as N
    pool = [str(d[0] if isinstance(d, (tuple, list)) else d.get("symptom") if isinstance(d, dict) else d)
            for d in N._DISTRACTOR_SYMPTOMS]
    return sum(1 for e in b.raw.evidence_ledger if e.get("source_type") == "patient_reported_symptom"
               and any(p and p in str(e.get("symptom")) for p in pool))


def _course_moved(low, high) -> dict:
    out = {}
    for s in ("weight", "HbA1c", "fasting_glucose"):
        a, b = C.series(low.raw.longitudinal_data, s), C.series(high.raw.longitudinal_data, s)
        moved = [d for d in a if d in b and abs(a[d] - b[d]) > 1e-9]
        if moved:
            out[s] = len(moved)
    return out


def test_c4_distractor_level_adds_distractors_and_leaves_the_course_alone():
    """At a fixed symptom rate, raising the distractor level puts distractor-pool symptoms on
    the ledger and changes no reading of the weight and lab series."""
    low, high = _c4_pair(C.build)
    assert _n_distractor(high) > _n_distractor(low), (
        f"high distractor level added no distractors ({_n_distractor(low)} -> {_n_distractor(high)})")
    moved = _course_moved(low, high)
    assert not moved, f"readings changed when only the distractor level changed: {moved}"


def test_c4_negative_control_distractors_not_injected(monkeypatch):
    """Negative control: with the distractor injector a no-op, C4's first half must fail."""
    from haenv import build as B
    monkeypatch.setattr(B, "inject_distractors", lambda raw, *a, **k: raw)
    low, high = _c4_pair(C.build_uncached)
    assert not _n_distractor(high) > _n_distractor(low), "C4 passed with no distractor injected"


def test_c4_negative_control_distractor_moves_weight(monkeypatch):
    """Negative control: a distractor injector that also nudges the weight series must be
    caught by C4's second half."""
    from haenv import build as B
    orig = B.inject_distractors

    def nudge(raw, *a, **k):
        out = orig(raw, *a, **k)
        for pt in out.longitudinal_data.get("weight") or []:
            if 30 <= pt["ts"] <= 34:
                pt["value"] += 0.1
        return out
    monkeypatch.setattr(B, "inject_distractors", nudge)
    low, high = _c4_pair(C.build_uncached)
    assert "weight" in _course_moved(low, high), "a distractor that moved weight went unnoticed"


# ------------------------------------------------------------------ C5 driver -> response band
def test_c5_low_responder_moves_hba1c_less_than_a_responder():
    """Declaring `biological_low_response` puts the person in the non-response band: the
    90-day HbA1c fall is smaller than for the same person with a responder driver."""
    job, cid = C.T2D_JOB, "T2G-08"
    base = C.spec_of(job, cid)
    low = C.build(C.variant(base), template=job)
    resp = C.build(C.variant(base, driver="calorie_intake_change"), template=job)
    _need(low, resp)
    fall = lambda b: _lab(b, "HbA1c", 0) - _lab(b, "HbA1c", 90)
    assert fall(low) < fall(resp), f"low responder fell {fall(low):.2f}, responder {fall(resp):.2f}"


def test_c5_negative_control_driver_ignored(monkeypatch):
    """Negative control: with the response no longer depending on the driver, C5 must fail."""
    from haenv import drug_effects as DE
    monkeypatch.setattr(DE, "response_for", lambda *a, **k: 1.0)
    job, cid = C.T2D_JOB, "T2G-08"
    base = C.spec_of(job, cid)
    low = C.build_uncached(C.variant(base), template=job)
    resp = C.build_uncached(C.variant(base, driver="calorie_intake_change"), template=job)
    _need(low, resp)
    fall = lambda b: _lab(b, "HbA1c", 0) - _lab(b, "HbA1c", 90)
    assert not fall(low) < fall(resp), "C5 passed with the driver cut out of the response"
