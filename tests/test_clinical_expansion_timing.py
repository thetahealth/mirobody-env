"""Disease timing must survive independent patient time-window draws."""
import pytest
import haenv.world.clinical.ddx as ddx

def helper(monkeypatch, policy):
    fn = getattr(ddx, "clinical_symptom_days", None)
    assert callable(fn), "new clinical disease timing must be supported"
    monkeypatch.setattr(ddx, "_clinical_timing", lambda _: policy)
    return fn

def test_acute_symptoms_are_not_spread_into_a_longitudinal_template(monkeypatch):
    fn = helper(monkeypatch, {"mode":"acute","acute_window_days":3,"max_episode_days":4})
    assert fn("JD-116v2","HD-UNI-07",[1,2,3,4],84) == [81,82,83,84]

def test_chronic_duration_is_not_compressed_below_diagnostic_history(monkeypatch):
    fn = helper(monkeypatch, {"mode":"chronic","chronic_min_span_days":90})
    days = fn("JD-119v2","HD-UNI-10",[14,35,56,112],168)
    assert days[-1] == 168
    assert days[-1] - days[0] >= 90
    assert days == sorted(days)

def test_chronic_history_does_not_fit_a_short_window(monkeypatch):
    fn = helper(monkeypatch, {"mode":"chronic","chronic_min_span_days":90})
    with pytest.raises(ValueError,match="window"):
        fn("JD-119v2","HD-UNI-10",[14,35,56,112],56)

def test_unsupported_timing_fails_instead_of_relabeling_evidence(monkeypatch):
    fn = helper(monkeypatch, {"mode":"invented"})
    with pytest.raises(ValueError,match="timing"):
        fn("JD-116v2","HD-UNI-07",[1,2,3,4],84)

def test_acute_episode_that_exceeds_its_declared_span_fails(monkeypatch):
    fn = helper(monkeypatch, {"mode":"acute","acute_window_days":3,"max_episode_days":4})
    with pytest.raises(ValueError,match="span"):
        fn("JD-116v2","HD-UNI-07",[1,2,3,20],84)

def test_legacy_symptom_jitter_is_unchanged(monkeypatch):
    fn = helper(monkeypatch, None)
    assert fn("JD-04v2","JD-HYPO",[14,35,56,77],84) == ddx.jitter_symptom_days("JD-04v2",[14,35,56,77],84)

def test_missing_chronic_history_is_moved_beyond_T_without_overrunning_course():
    cases = ddx.ddx_case_specs(only=["HD-UNI-10","HD-UNI-13"],variants=12,
                              density="mixed",insufficient_frac=1)
    missing = [c for c in cases if c["latent"].get("ddx_insufficient")]
    assert missing
    for c in missing:
        T=c["latent"]["index_time_T"]
        end=c["latent"]["course_end_day"]
        days=[s["day"] for s in c["raw"]["symptoms"]]
        assert min(days)>T
        assert max(days)<end
        required=ddx._clinical_timing(c["latent"]["ddx_spec_id"])["chronic_min_span_days"]
        assert max(days)-min(days)>=required

def test_clinical_metadata_cache_is_private_and_obeys_registry_override(tmp_path):
    from haenv.resources.regpath import registry_dir_override
    original = ddx._clinical_metadata("HD-UNI-10")
    changed = ddx._clinical_metadata("HD-UNI-10")
    changed["symptom_timing"]["chronic_min_span_days"] = 9999
    assert ddx._clinical_metadata("HD-UNI-10") == original
    (tmp_path / "clinical_expansion.yaml").write_text(
        "conditions:\n  HD-UNI-10:\n    symptom_timing:\n      mode: chronic\n      chronic_min_span_days: 123\n",
        encoding="utf-8")
    with registry_dir_override(tmp_path):
        assert ddx._clinical_timing("HD-UNI-10")["chronic_min_span_days"] == 123
        assert ddx._clinical_metadata("HD-UNI-13") is None
    assert ddx._clinical_metadata("HD-UNI-10") == original

def test_cached_generation_matches_uncached_clinical_tables(monkeypatch):
    import haenv.world.generation.demographics as DM
    before = ddx.ddx_case_specs(only=["HD-UNI-10", "HD-UNI-13"], variants=4,
                               density="mixed", composition_v2=True)
    monkeypatch.setattr(ddx, "_clinical_metadata_table", ddx._clinical_metadata_table.__wrapped__)
    monkeypatch.setattr(ddx, "_penetrance_index", ddx._penetrance_index.__wrapped__)
    monkeypatch.setattr(DM, "_pairing", DM._pairing.__wrapped__)
    monkeypatch.setattr(DM, "_indications", DM._indications.__wrapped__)
    assert ddx.ddx_case_specs(only=["HD-UNI-10", "HD-UNI-13"], variants=4,
                              density="mixed", composition_v2=True) == before
