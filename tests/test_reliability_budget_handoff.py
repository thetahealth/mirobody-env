"""All repetitions share one cap; unlaunched repeats are never reported successful."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import reliability_passk as run


def test_paid_repeats_fail_before_launch_without_shared_budget(monkeypatch):
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **k: pytest.fail("paid process launched"))
    with pytest.raises(ValueError, match="shared budget"):
        run._run_samples(Path("job.yaml"), [Path("one")], {"model"}, serial=False)


def test_all_repeats_receive_the_same_absolute_ledger(monkeypatch, tmp_path):
    seen = []
    def popen(cmd, **kwargs):
        seen.append(cmd)
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(run.subprocess, "Popen", popen)
    ledger = tmp_path / "budget.json"
    result = run._run_samples(Path("job.yaml"), [Path(str(i)) for i in range(3)], {"model"},
                             serial=False, budget_usd="100", budget_ledger=ledger)
    assert result == [0, 0, 0]
    assert len(seen) == 3
    for cmd in seen:
        assert cmd[cmd.index("--judge-budget-usd") + 1] == "100"
        assert cmd[cmd.index("--judge-budget-ledger") + 1] == str(ledger.resolve())


def test_serial_failure_marks_unlaunched_samples_as_none(monkeypatch):
    seen = []
    def execute(cmd, **kwargs):
        seen.append(cmd)
        return SimpleNamespace(returncode=6)
    monkeypatch.setattr(run.subprocess, "run", execute)
    codes = run._run_samples(Path("job.yaml"), [Path(str(i)) for i in range(3)], None,
                            serial=True, offline=True)
    assert codes == [6, None, None] and len(seen) == 1


def test_semantic_history_keeps_ranked_diagnosis_and_canonical_test_metrics():
    assert {"dx_hit", "tests_recall", "tests_precision", "disc_recall"} <= set(run.DIMS)
    assert "dx_hit" in run.BINARY


def test_two_nonempty_runs_out_of_three_cannot_produce_a_noise_floor(monkeypatch, tmp_path):
    job = tmp_path / "job.yaml"
    job.write_text("job_id: j\ntask_type: joint_dx\n")
    source = tmp_path / "results/joint_dx/j/source"
    source.mkdir(parents=True)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "pick_source_batch", lambda *a: source)
    monkeypatch.setattr(run, "_assert_samples_distinct", lambda dirs: dirs)
    monkeypatch.setattr(run, "_load", lambda path, models, **kw:
                        {} if path.name.endswith("pk3") else {("case", "model"): {"dx_hit": True}})
    monkeypatch.setattr(run, "_aggregate", lambda *a: pytest.fail("Aggregated only two usable repeats"))
    assert run.main([str(job), "--report-only"]) == 1


def test_semantic_reliability_never_reads_proxy_when_judgments_absent(tmp_path):
    import json
    (tmp_path / "eval.jsonl").write_text(json.dumps({"case": "c", "solver": "model",
        "geometry": "single", "gold_kind": "ddx:unified", "dx_hit": 1., "overall": "SCORED"}) + "\n")
    assert run._load(tmp_path, None, semantic=True) == {}


def test_not_applicable_atoms_never_count_as_failed_repetitions():
    samples = [{("a", "model"): {"dx_hit": 1.}, ("b", "model"): {"dx_hit": None}}
               for _ in range(3)]
    report = run._aggregate(samples)
    dim = report["dims"]["dx_hit"]
    assert dim["pass_k"]["pass^3"] == 1.
    assert dim["n_grids"] == 1 and dim["n_excluded_missing"] == 1


def test_missing_dimension_on_different_cases_is_not_noise():
    samples = [{("a", "model"): {"dx_hit": 1.}, ("b", "model"): {"dx_hit": None}},
               {("a", "model"): {"dx_hit": None}, ("b", "model"): {"dx_hit": 0.}}]
    report = run._aggregate(samples)
    assert "dx_hit" not in report["dims"]
    assert report["noise_floor_max"] is None
