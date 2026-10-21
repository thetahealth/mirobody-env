"""Primary reports use semantic values, never a proxy or imputed judge result."""


from _fixtures_semantic_report import *  # explicit shared fixture module

def test_llm_replaces_primary_values_but_preserves_original_proxy():
    raw = rows(); original = deepcopy(raw)
    viewed = overlay_rows(raw, results(raw), run_id="test", judging_sha="semantic")
    assert raw == original
    assert viewed[0]["dx_hit"] == 0.
    assert viewed[0]["code_proxy"]["dx_hit"] == 1.
    assert resolve(viewed[0], "tests_recall") == .5
    ranked = rank_ddx(viewed)
    assert all(r["dx_hit"] == 0. for r in ranked)
    assert all(r["semantic_coverage"]["primary_missing_cells"] == 0 for r in ranked)

def test_unresolved_measurement_cannot_fall_back_or_zero_fill_a_headline():
    raw = rows(); result = results(raw)
    cell = result[("case-0", "gpt-6-luna")]
    cell["status"] = "unresolved"
    cell["summary"]["metrics"]["tests_recall"] = None
    cell["summary"]["coverage"]["tests_recall"]["resolved"] = 0
    viewed = overlay_rows(raw, result, run_id="test", judging_sha="semantic")
    assert viewed[0]["tests_recall"] is None
    assert resolve(viewed[0], "tests_recall") is None
    ranked = {r["model"]: r for r in rank_ddx(viewed)}
    r = ranked["gpt-6-luna"]
    assert r["score"] is None and r["score_full"] is None
    assert r["semantic_coverage"]["primary_missing_cells"] == 1
    assert not r.get("ruler_imputed_zero")

def test_not_run_cells_are_not_filled_from_legacy_scores():
    raw = rows()
    viewed = overlay_rows(raw, {}, run_id="test", judging_sha="semantic")
    assert all(r["dx_hit"] is None for r in viewed)
    assert all(r["score"] is None for r in rank_ddx(viewed))
    text = report_section(viewed)
    assert "not final" in text and "pending" in text
    assert "not included in the composite" in text

def test_unknown_binary_judgments_do_not_become_negative_scores():
    raw = rows()
    for row in raw:
        row["dx_hit"] = None;row["noop_ok"] = None
    for r in rank_ddx(raw):
        assert r["dx_hit"] is None and r["noop_ok"] is None
        assert r["n_dx"] == 0 and r["n_noop_ok"] == 0

def test_auxiliary_value_does_not_change_composite():
    raw = rows(); a = results(raw); b = deepcopy(a)
    for value in b.values():
        value["summary"]["metrics"]["action_consistency"] = 0.
    one = rank_ddx(overlay_rows(raw, a, run_id="a", judging_sha="semantic"))
    two = rank_ddx(overlay_rows(raw, b, run_id="b", judging_sha="semantic"))
    assert [(r["model"],r["score"]) for r in one] == [(r["model"],r["score"]) for r in two]

def test_old_proxy_validity_never_transfers_to_llm_judgment():
    from haenv.scoring.legacy import load_profile
    from haenv.scoring.semantic.report import profile_for_rows
    profile = load_profile(); previous = profile.validity_state("tests_recall")
    viewed = overlay_rows(rows(), results(rows()), run_id="test", judging_sha="semantic")
    new = profile_for_rows(profile, viewed)
    assert new.validity_state("tests_recall") == ("unmeasured", None)
    assert profile.validity_state("tests_recall") == previous
    assert "tests_recall" not in new.publishable_dims()

def test_full_report_renders_actual_semantic_coverage_and_auxiliary_section(tmp_path):
    from types import SimpleNamespace
    from haenv.reporting.document import render
    job = SimpleNamespace(multiround=False, sample_cases=3, job_id="synthetic",
                          task_type="joint_dx", batch="synthetic", root=tmp_path,
                          results_file=tmp_path / "eval.jsonl", results_dir=tmp_path)
    text = render(job, overlay_rows(rows(), {}, run_id="test", judging_sha="semantic"), {}, [], {})
    assert "LLM semantic judging" in text and "pending=20" in text
    assert "unscored" in text and "Auxiliary observations" in text

def test_report_reconstructs_votes_without_dispatch(tmp_path):
    from haenv.scoring.semantic.report import read_run
    out, batch, _ = saved_run(tmp_path)
    manifest, result = read_run(out, batch)
    assert result[("case", "model")]["summary"]["metrics"]["dx_hit"] == 1.

@pytest.mark.parametrize("change", ["source", "summary", "vote", "identity", "tasks", "journal"])
def test_report_refuses_tampered_inputs_or_scores(tmp_path, change):
    from haenv.scoring.semantic.report import read_run
    out, batch, path = saved_run(tmp_path)
    result = json.loads(path.read_text())
    if change == "source":
        (batch / "responses.jsonl").write_text("different answer")
    elif change == "tasks":
        with (out / "tasks.jsonl").open("a") as file:file.write("\n")
    elif change == "journal":
        (out / "samples/one.jsonl").write_text("")
    elif change == "summary":
        result["summary"]["metrics"]["dx_hit"] = 0.
    elif change == "vote":
        result["consensus"]["verdicts"]["dx"] = "no"
    elif change == "identity":
        result["source"]["case"] = "other"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        read_run(out, batch)
