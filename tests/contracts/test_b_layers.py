"""Contract group B: each pipeline layer does only its own job.

The pipeline after the course is rendered: kernel noise that changes the patient or drops
readings -> day-by-day events and the physiology layer (ground truth + observation) -> reading
artifacts on the observed series -> post-injection (recording behaviour such as
carried-forward) -> emission gates. Each layer must leave what the earlier layers produced
intact, except for the change it exists to make.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pytest

import _cohort as C

pytestmark = pytest.mark.world

#: Shipped specifications with a daily weight series: a T2D maintainer, an obesity
#: maintainer, and three early-warning cases (regain by adherence, low response, calories).
COHORT = [(C.T2D_JOB, "T2G-07"), (C.T2D_JOB, "T2G-01"), (C.EXAMPLE_JOB, "EWX-03"),
          (C.EW20_JOB, "RC-EW-18"), (C.EW20_JOB, "RC-EW-03")]
GOLD_KEYS = ("outcome_label", "gold_drivers")


def _built(job, cid, noise=None, capture=True, **latent):
    base = C.spec_of(job, cid)
    return C.build(C.variant(base, noise=noise, **latent), template=job, capture=capture)


# ------------------------------------------------------------------ B1 truth excludes artifacts
@pytest.mark.parametrize("cls,week", [("transient_spike", 5), ("unit_error", 4), ("device_switch", 7)])
@pytest.mark.parametrize("job,cid", COHORT[:3])
def test_b1_ground_truth_excludes_observation_artifacts(job, cid, cls, week):
    """The physiology layer's ground-truth weight must be the same with and without an
    observation artifact: an artifact is a property of the measurement, not of the patient.

    Catches: kernel artifacts injected before the physiology layer and then treated as real
    weight (they get smoothed into neighbouring days, fed to the anchor and slope gates, and
    sampled into the clinic-scale backfill). Turns red whenever an observation artifact leaks
    into the truth series."""
    clean = _built(job, cid)
    noisy = _built(job, cid, noise=[{"class": cls, "week": week}])
    ct = C.series(clean.layers.get("truth") or {}, "weight")
    nt = C.series(noisy.layers.get("truth") or {}, "weight")
    if not ct or not nt:
        pytest.fail(f"{cid}: no truth series captured (physiology layer off?)")
    diff = {d: round(nt[d] - ct[d], 2) for d in ct if d in nt and abs(nt[d] - ct[d]) > 0.3}
    assert not diff, f"{cid} {cls}@wk{week}: truth moved on {len(diff)} days, e.g. {dict(list(diff.items())[:5])}"


def test_b1_negative_control_truth_shift_is_seen(monkeypatch):
    """Negative control: shifting the truth series by hand must be reported."""
    from haenv import build as B
    orig = B._inject_events_verified

    def shifted(raw, *a, **k):
        out = orig(raw, *a, **k)
        for p in ((out[1] or {}).get("physio_clean_ld") or {}).get("weight") or []:
            if 30 <= p["ts"] <= 40:
                p["value"] += 2.0
        return out
    job, cid = COHORT[0]
    clean = _built(job, cid)
    monkeypatch.setattr(B, "_inject_events_verified", shifted)
    C._build_cached.cache_clear()
    try:
        noisy = C.build(C.variant(C.spec_of(job, cid), case_id="NC-B1"), template=job, capture=True)
    finally:
        C._build_cached.cache_clear()
    ct = C.series(clean.layers["truth"], "weight")
    nt = C.series(noisy.layers["truth"], "weight")
    assert any(abs(nt[d] - ct[d]) > 0.3 for d in ct if d in nt), "a shifted truth series went unnoticed"


# ------------------------------------------------------------------ B2 post-injection scope
#: Noise-free cohort, plus one spike and one device-switch case whose artifact readings are
#: already on the series when post-injection runs.
B2_CASES = ([(job, cid, None) for job, cid in COHORT]
            + [(C.T2D_JOB, "T2G-07", [{"class": "transient_spike", "week": 5}]),
               (C.T2D_JOB, "T2G-07", [{"class": "device_switch", "week": 7}])])


def _b2_changes(b) -> list[str]:
    # The home scale shows 0.1 kg, and `apply_post_injection` puts the observed weight on that grid
    # before any reading is copied forward. A reading may therefore equal its own artifact-layer
    # value on the 0.1 kg grid, or the reading before it; nothing else.
    from haenv.physio.noise import weight_on_scale_grid
    ar = sorted(C.series(b.layers["artifacts"]["longitudinal_data"], "weight").items())
    po = sorted(C.series(b.layers["post_inject"]["longitudinal_data"], "weight").items())
    if [d for d, _ in ar] != [d for d, _ in po]:
        return ["post-injection changed the set of days"]
    return [f"day {d} re-valued without copying" for i, ((d, a), (_, b2)) in enumerate(zip(ar, po))
            if b2 != weight_on_scale_grid(a) and (i == 0 or b2 != po[i - 1][1])]


@pytest.mark.parametrize("job,cid,noise", B2_CASES)
def test_b2_post_injection_only_copies_readings(job, cid, noise):
    """Between the reading-artifact layer and the post-injection layer, the weight series
    keeps the same days, and every reading equals its artifact-layer value on the home
    scale's 0.1 kg grid or the reading before it (the scale resolution and carried-forward
    are the only recording behaviours registered for `weight`; the grid since 51bbfba3).

    Catches: a post-injection layer that adds, drops or re-values readings beyond the display
    rounding, artifact readings included. Turns red when a new post-injector is registered for
    weight without a contract of its own."""
    b = _built(job, cid, noise=noise)
    if "post_inject" not in b.layers:
        pytest.fail(f"{cid}: not emitted or post-injection not reached ({b.audit.get('post_noise_conflicts')})")
    bad = _b2_changes(b)
    assert not bad, f"{cid}: {bad[:5]}"


def test_b2_negative_control_revalued_reading(monkeypatch):
    """Negative control: a post-injection layer that adds 1 kg to one reading must be caught."""
    from haenv import post_inject as PI
    orig = PI.apply_post_injection

    def bump(raw, **k):
        res = orig(raw, **k)
        w = raw.longitudinal_data["weight"]
        w[len(w) // 2]["value"] += 1.0
        return res
    job, cid = COHORT[0]
    monkeypatch.setattr(PI, "apply_post_injection", bump)
    C._build_cached.cache_clear()
    try:
        b = C.build(C.variant(C.spec_of(job, cid), case_id="NC-B2"), template=job, capture=True)
    finally:
        C._build_cached.cache_clear()
    assert _b2_changes(b), "a re-valued reading went unnoticed"


def test_b2_negative_control_coarser_scale_grid(monkeypatch):
    """Negative control for the grid allowance: a post-injection layer that rounds weight to
    0.5 kg (not the registered 0.1 kg) must be caught, so the allowance is the grid itself and
    not any rounding."""
    from haenv import post_inject as PI
    orig = PI.apply_post_injection

    def coarse(raw, **k):
        res = orig(raw, **k)
        for q in raw.longitudinal_data["weight"]:
            q["value"] = round(round(q["value"] / 0.5) * 0.5, 2)
        return res
    job, cid = COHORT[0]
    monkeypatch.setattr(PI, "apply_post_injection", coarse)
    C._build_cached.cache_clear()
    try:
        b = C.build(C.variant(C.spec_of(job, cid), case_id="NC-B2g"), template=job, capture=True)
    finally:
        C._build_cached.cache_clear()
    assert _b2_changes(b), "weight rounded to 0.5 kg went unnoticed"


# ------------------------------------------------------------------ B3 gold is fixed early
def _b3_changes(b) -> list[str]:
    final = {k: getattr(b.raw, k) for k in GOLD_KEYS}
    real = [(r.get("week"), r.get("type")) for r in b.raw.reversal_points if r.get("type") != "trap"]
    out = []
    for layer in ("noise", "events", "artifacts", "post_inject"):
        snap = b.layers[layer]
        r = [(x.get("week"), x.get("type")) for x in snap["reversal_points"] if x.get("type") != "trap"]
        if r != real:
            out.append(f"layer {layer}: reversal points {r} != final {real}")
        for k in GOLD_KEYS:
            if snap[k] != final[k]:
                out.append(f"layer {layer}: {k} {snap[k]!r} != final {final[k]!r}")
    return out


def _b3_vs_clean(b, clean) -> list[str]:
    """Gold fields and latent reversal points of a noisy case against the same spec without
    noise."""
    real = [(r.get("week"), r.get("type")) for r in b.raw.reversal_points if r.get("type") != "trap"]
    creal = [(r.get("week"), r.get("type")) for r in clean.raw.reversal_points if r.get("type") != "trap"]
    out = [f"{k} {getattr(b.raw, k)!r} != clean {getattr(clean.raw, k)!r}"
           for k in GOLD_KEYS if getattr(b.raw, k) != getattr(clean.raw, k)]
    if real != creal:
        out.append(f"reversal points {real} != clean {creal}")
    return out


B3_NOISE = [None, [{"class": "transient_spike", "week": 5}], [{"class": "unit_error", "week": 4}],
            [{"class": "device_switch", "week": 7}]]


@pytest.mark.parametrize("job,cid", COHORT)
@pytest.mark.parametrize("noise", B3_NOISE)
def test_b3_gold_is_the_same_after_every_layer(job, cid, noise):
    """Outcome and driver are fixed before the course is rendered; no layer may change
    them, and the reversal points declared by the latents stay put (noise may only add traps).
    With a reading artifact they are also the same as without it.

    Catches: a layer that edits gold fields, and an artifact that changes them. Turns red when
    any layer writes to them."""
    b = _built(job, cid, noise=noise)
    if not b.emitted:
        pytest.fail(f"{cid}: not emitted ({b.audit.get('post_noise_conflicts')})")
    assert b.raw.outcome_label in ("event_occurred", "event_not_occurred")
    bad = _b3_changes(b)
    if noise:
        clean = _built(job, cid)
        if not clean.emitted:
            pytest.fail(f"{cid}: clean case not emitted ({clean.audit.get('post_noise_conflicts')})")
        bad += _b3_vs_clean(b, clean)
    assert not bad, f"{cid}: " + "; ".join(bad)


def test_b3_negative_control_gold_rewritten():
    """Negative control: a layer snapshot whose driver and outcome differ from the final ones
    must be reported. (A build whose post-injection layer rewrites them is refused by the
    emission gates -- `outcome_declared_not_derived`, `gold_not_coverable` -- so the measure is
    exercised on a tampered snapshot of an emitted case.)"""
    import copy
    job, cid = COHORT[0]
    b = copy.copy(_built(job, cid))
    b.layers = copy.deepcopy(b.layers)
    ev = b.layers["events"]
    ev["gold_drivers"] = ["sleep_decline"]
    ev["outcome_label"] = "event_occurred" if b.raw.outcome_label != "event_occurred" else "event_not_occurred"
    bad = _b3_changes(b)
    assert any("gold_drivers" in x for x in bad) and any("outcome_label" in x for x in bad), bad


def test_b3_negative_control_artifact_changed_gold():
    """Negative control: an artifact case whose driver differs from the clean case's must be
    reported."""
    import copy
    job, cid = COHORT[0]
    clean = _built(job, cid)
    b = copy.copy(_built(job, cid, noise=[{"class": "unit_error", "week": 4}]))
    b.raw = copy.deepcopy(b.raw)
    b.raw.gold_drivers = ["sleep_decline"]
    assert any("gold_drivers" in x for x in _b3_vs_clean(b, clean)), "a changed driver went unnoticed"


# ------------------------------------------------------------------ B4 stream order is tier-blind
B4_ARTIFACTS = [("transient_spike", 5), ("unit_error", 4), ("device_switch", 7)]
#: Reading artifacts on the first three cohort specs; `adherence_gap` (it adds a reading to
#: `medication_adherence`) on RC-EW-03, the cohort spec that emits it.
B4_CASES = ([(job, cid, cls, week) for job, cid in COHORT[:3] for cls, week in B4_ARTIFACTS]
            + [(C.EW20_JOB, "RC-EW-03", "adherence_gap", 4)])


def _stream_order(b):
    return list(b.raw.longitudinal_data)


@pytest.mark.parametrize("job,cid,cls,week", B4_CASES)
def test_b4_stream_order_does_not_reveal_the_artifact_tier(job, cid, cls, week):
    """With and without a reading artifact, the emitted case lists the same streams in the
    same order. The solver payload keeps that order, so a stream that sits elsewhere only in
    artifact cases (e.g. a `weight_ref` added before the events layer) names the tier.

    Catches: an observation stream added at a different layer for artifact cases. Turns red
    when reading artifacts move back in front of the events layer."""
    clean = _built(job, cid, capture=False)
    noisy = _built(job, cid, noise=[{"class": cls, "week": week}], capture=False)
    if not (clean.emitted and noisy.emitted):
        pytest.fail(f"{cid} {cls}@wk{week}: not emitted ({noisy.audit.get('post_noise_conflicts')})")
    assert _stream_order(noisy) == _stream_order(clean), (
        f"{cid} {cls}@wk{week}: {_stream_order(noisy)} vs {_stream_order(clean)}")


def test_b4_negative_control_artifacts_before_events(monkeypatch):
    """Negative control: with every noise class applied before the events layer, a spike
    case must list `weight_ref` somewhere else than the same case without it."""
    from haenv import build as B
    import haenv_kernel.noise as N
    job, cid = COHORT[0]
    base = C.spec_of(job, cid)
    monkeypatch.setattr(B, "NOISE_BEFORE_PHYSIO", frozenset(N.NOISE_CLASSES))
    clean = C.build_uncached(C.variant(base, noise=None), template=job)
    # Week 4: at week 5 the physiology noise, applied after the spike here, leaves it under
    # half its injected jump, so the trap gate blocks the case.
    noisy = C.build_uncached(C.variant(base, noise=[{"class": "transient_spike", "week": 4}]),
                             template=job)
    assert clean.emitted and noisy.emitted, (
        f"negative-control case not emitted ({noisy.audit.get('post_noise_conflicts')})")
    assert _stream_order(noisy) != _stream_order(clean), (
        "streams kept their order with artifacts before the events layer: B4 cannot see the tier")


# ------------------------------------------------------------------ B5 clinic-scale reference is tier-blind
#: Reading artifacts on the first three cohort specs, and on T2G-07 cut to a 90-day course,
#: whose truth has a single reading on the 28-day clinic grid.
B5_CASES = ([(job, cid, cls, week, None) for job, cid in COHORT[:3] for cls, week in B4_ARTIFACTS]
            + [(C.T2D_JOB, "T2G-07", cls, 4, 90) for cls, _ in B4_ARTIFACTS])


def _ref(b):
    ld = b.layers["post_inject"]["longitudinal_data"]
    return None if "weight_ref" not in ld else [(int(q["ts"]), float(q["value"])) for q in ld["weight_ref"]]


def _b5_mismatch(clean, noisy, cid) -> list[str]:
    """The artifact case's clinic-scale reference against the clean case's, and against the
    clean truth plus the clinic-scale offset (the reference rounds to 0.01 kg)."""
    from haenv import post_inject as PI
    if "post_inject" not in clean.layers or "post_inject" not in noisy.layers:
        return ["post-injection not reached"]
    ref, cref = _ref(noisy), _ref(clean)
    out = [] if ref == cref else [f"reference {ref} != clean {cref}"]
    truth = C.series(clean.layers.get("truth") or {}, "weight")
    params = PI.clinic_scale_params()
    far = [(t, round(v - truth[t] - PI.clinic_scale_offset(cid, t, params), 3)) for t, v in ref or []
           if t in truth and abs(v - truth[t] - PI.clinic_scale_offset(cid, t, params)) > 0.006]
    if far:
        out.append(f"reference off the truth plus clinic offset on {len(far)} days, e.g. {far[:3]}")
    return out


@pytest.mark.parametrize("job,cid,cls,week,end", B5_CASES)
def test_b5_clinic_reference_does_not_reveal_the_artifact_tier(job, cid, cls, week, end):
    """With and without a reading artifact the clinic-scale reference (`weight_ref`) is the
    same: present on the same days with the same values, each the clean truth plus the clinic
    scale's offset. The clinic scale sees neither the home scale's artifact nor its
    observation noise. A reference sampled from the observed series differs from the clean
    case's by that noise, and one present only in artifact cases names the tier.

    Catches: artifact cases whose reference comes from the observed series (the one a kernel
    injector samples), and a reference present only in artifact cases on a course with fewer
    than two grid readings. Turns red when artifact cases stop taking the reference from the
    truth-series backfill."""
    latent = {} if end is None else {"course_end_day": end}
    clean = _built(job, cid, **latent)
    noisy = _built(job, cid, noise=[{"class": cls, "week": week}], **latent)
    bad = _b5_mismatch(clean, noisy, cid)
    assert not bad, f"{cid} {cls}@wk{week}: " + "; ".join(bad)


def test_b5_negative_control_kernel_reference_kept(monkeypatch):
    """Negative control: an artifact case that keeps the reference a kernel injector samples
    from the observed series before the artifact must be reported by both halves of B5."""
    import copy

    from haenv import build as B
    import haenv_kernel.noise as N
    orig = B._inject_observation_artifacts

    def keep_kernel_reference(raw, cs, *a, **k):
        before = copy.deepcopy(raw.longitudinal_data.get("weight") or [])
        out, unplaced = orig(raw, cs, *a, **k)
        if cs.noise:
            N._add_clean_ref(out, before)
        return out, unplaced
    job, cid = COHORT[0]
    base = C.variant(C.spec_of(job, cid), case_id="NC-B5")
    monkeypatch.setattr(B, "_inject_observation_artifacts", keep_kernel_reference)
    C._build_cached.cache_clear()
    try:
        clean = C.build(C.variant(base, noise=None), template=job, capture=True)
        noisy = C.build(C.variant(base, noise=[{"class": "transient_spike", "week": 5}]),
                        template=job, capture=True)
    finally:
        C._build_cached.cache_clear()
    bad = _b5_mismatch(clean, noisy, "NC-B5")
    assert any("!= clean" in x for x in bad) and any("off the truth" in x for x in bad), (
        f"a reference sampled from the observed series went unnoticed: {bad}")
