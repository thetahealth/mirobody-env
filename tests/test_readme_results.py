"""README results must preserve the public snapshot and reject invalid readings."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("readme_results", ROOT / "docs/scripts/make_readme_results.py")
FIGURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURE)


def _as_release_world(d: dict) -> dict:
    """The exported page data read as if both batches ran on the release world, so the
    figure code is exercised on real readings whether or not the export is a snapshot."""
    pv = d["provenance"]
    pv["is_snapshot"] = False
    for snap in pv["snapshot"].values():
        snap.update(current_world=True, world_sha=pv["release"]["world_sha"],
                    judging_sha16=pv["release"]["judging_sha16"])
    return d


@pytest.fixture
def data():
    return _as_release_world(json.loads((ROOT / "web/demo/data.json").read_text()))


def test_snapshot_export_is_refused_by_the_figure():
    raw = json.loads((ROOT / "web/demo/data.json").read_text())
    if not raw["provenance"]["is_snapshot"]:
        pytest.skip("the checked-in export is on the release world")
    with pytest.raises(ValueError, match="current release world"):
        FIGURE.prepare(raw)


def test_exact_public_values_and_alphabetical_models(data):
    result = FIGURE.prepare(data)
    models = sorted(data["board"]["models"], key=lambda row: row["solver"])
    assert result["solvers"] == [row["solver"] for row in models]
    assert result["values"] == [[row["dims"][dim] for dim in result["dims"]] for row in models]
    data["board"]["models"].reverse()
    assert FIGURE.prepare(data) == result


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.01, 1.01, True])
def test_invalid_readings_raise_instead_of_becoming_zero(data, value):
    data["board"]["models"][0]["dims"]["quant_ok"] = value
    with pytest.raises(ValueError, match="invalid reading"):
        FIGURE.prepare(data)


def test_missing_reading_is_rejected(data):
    del data["board"]["models"][0]["dims"]["quant_ok"]
    with pytest.raises(ValueError, match="per-dimension"):
        FIGURE.prepare(data)


def test_withheld_half_of_paired_dimension_is_rejected(data):
    # withhold one production metric of a plotted dimension (the second half when paired)
    data["board"]["withheld"].append({"metric": data["board"]["dims"][0].split("+")[-1]})
    with pytest.raises(ValueError, match="Withheld"):
        FIGURE.prepare(data)


@pytest.mark.parametrize("mutate", [
    lambda d: d["board"]["models"][0].update(macro=.9),
    lambda d: d["board"]["models"].append(copy.deepcopy(d["board"]["models"][0])),
    lambda d: d["provenance"].update(n_real_models=100),
    lambda d: d["provenance"].update(is_snapshot=True),
    lambda d: d["board"].update(judging="stale"),
])
def test_composite_duplicate_incomplete_or_stale_snapshot_rejected(data, mutate):
    mutate(data)
    with pytest.raises(ValueError):
        FIGURE.prepare(data)


def test_changed_public_reading_changes_export(data):
    before = FIGURE.prepare(data)
    data["board"]["models"][0]["dims"]["quant_ok"] = .1234
    assert FIGURE.prepare(data) != before


def test_dimension_order_preserves_values_and_source_order(data):
    result = FIGURE.prepare(data)
    before = copy.deepcopy(result)
    for dim in result["dims"]:
        rows = FIGURE.ranked_items(result, dim)
        expected = sorted(data["board"]["models"],
                          key=lambda row: (-row["dims"][dim], row["solver"]))
        assert [row["solver"] for row in rows] == [row["solver"] for row in expected]
        assert [row["value"] for row in rows] == [row["dims"][dim] for row in expected]
        assert len(rows) == len(result["solvers"])
    assert result == before


def test_exact_ties_share_competition_rank(data):
    dim = "quant_ok"
    for index, row in enumerate(data["board"]["models"]):
        row["dims"][dim] = .9 if index < 2 else .8 - index * .01
    rows = FIGURE.ranked_items(FIGURE.prepare(data), dim)
    assert [row["rank"] for row in rows[:3]] == [1, 1, 3]
    assert [row["tied"] for row in rows[:3]] == [True, True, False]
    assert rows[0]["solver"] < rows[1]["solver"]


def test_ties_use_exported_values_not_rounded_display(data):
    dim = "quant_ok"
    data["board"]["models"][0]["dims"][dim] = .99004
    data["board"]["models"][1]["dims"][dim] = .99003
    rows = FIGURE.ranked_items(FIGURE.prepare(data), dim)
    assert [row["rank"] for row in rows[:2]] == [1, 2]
    assert not any(row["tied"] for row in rows[:2])


def test_unpublished_dimension_cannot_be_sorted(data):
    with pytest.raises(ValueError, match="public dimension"):
        FIGURE.ranked_items(FIGURE.prepare(data), "dx_hit")


def test_preliminary_preserves_production_scores_and_dimensions(data):
    result = FIGURE.prepare_preliminary(data)
    for key, track in result["tracks"].items():
        expected = sorted(data["preliminary"]["tracks"][key]["models"], key=lambda row: row["solver"])
        assert track["models"] == expected
        # The dimension set is whatever production scored on the track, read from the data.
        assert track["dims"] == data["preliminary"]["tracks"][key]["dims"]
        assert set(track["dims"]) <= FIGURE.PRELIMINARY_LABELS.keys()
    if json.loads((ROOT / "web/demo/data.json").read_text())["provenance"]["is_snapshot"]:
        pytest.skip("the figure files are regenerated from an export on the release world")
    assert (ROOT / "docs/figures/readme_results.csv").read_text() == FIGURE.csv_text(result)


@pytest.mark.parametrize("key", ["answers", "trace"])
def test_preliminary_ranks_exact_scores_separately(data, key):
    track = FIGURE.prepare_preliminary(data)["tracks"][key]
    before = copy.deepcopy(track)
    rows = FIGURE.preliminary_ranked_items(track)
    assert rows == sorted(rows, key=lambda row: (-row["score"], row["solver"]))
    assert track == before
    for index, row in enumerate(track["models"]):
        row["score"] = .9 if index < 2 else .8 - .01 * index
    rows = FIGURE.preliminary_ranked_items(track)
    assert [row["rank"] for row in rows[:3]] == [1, 1, 3]
    assert [row["tied"] for row in rows[:3]] == [True, True, False]
    track["models"][0]["score"] = .90001
    assert FIGURE.preliminary_ranked_items(track)[0]["tied"] is False


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -.01, 1.01, True])
def test_preliminary_invalid_composite_is_rejected(data, value):
    data["preliminary"]["tracks"]["answers"]["models"][0]["score"] = value
    with pytest.raises(ValueError, match="invalid preliminary reading"):
        FIGURE.prepare_preliminary(data)


@pytest.mark.parametrize("mutate", [
    lambda d: d["preliminary"].update(not_final=False),
    lambda d: d["preliminary"]["tracks"]["answers"].update(status="final"),
    lambda d: d["preliminary"]["tracks"]["trace"].update(judging="stale"),
    lambda d: d["preliminary"]["tracks"]["trace"]["context"].update(pack="other"),
    lambda d: d["preliminary"]["tracks"]["answers"].update(pending_metrics=[]),
    lambda d: d["preliminary"]["tracks"]["answers"]["models"][0]["dims"].pop(
        d["preliminary"]["tracks"]["answers"]["dims"][0]),
    lambda d: d["preliminary"]["tracks"]["trace"]["models"][0].update(gate_multiplier=None),
    lambda d: d["preliminary"]["tracks"]["answers"]["samples"][
        d["preliminary"]["tracks"]["answers"]["models"][0]["solver"]].update(
        {d["preliminary"]["tracks"]["answers"]["dims"][0]: 0}),
    lambda d: d["preliminary"]["tracks"]["trace"].update(pending_metrics=[{"metric": "other"}]),
    lambda d: d["board"]["clinical_review"]["trace"].update(n_pending=-1),
])
def test_preliminary_requires_status_provenance_complete_readings_and_counts(data, mutate):
    mutate(data)
    with pytest.raises(ValueError):
        FIGURE.prepare_preliminary(data)


def test_preliminary_csv_is_explicit_and_retains_unrounded_values(data):
    import csv
    import io
    data["preliminary"]["tracks"]["answers"]["models"][0]["score"] = .51234567
    result = FIGURE.prepare_preliminary(data)
    rows = list(csv.DictReader(io.StringIO(FIGURE.csv_text(result))))
    assert len(rows) == sum(len(t["models"]) * len(t["dims"]) for t in result["tracks"].values())
    assert all(row["status"] == "preliminary" and row["not_final"] == "true" for row in rows)
    assert rows[0]["score"] == "0.51234567"
    for row in rows:
        track = result["tracks"][row["track"]]
        model = next(item for item in track["models"] if item["solver"] == row["model"])
        assert float(row["value"]) == model["dims"][row["dimension"]]
        assert int(row["n"]) == track["samples"][row["model"]][row["dimension"]]
        assert (row["validity_pending"] == "true") == FIGURE.pending_dim(row["dimension"], set(result["pending"]))


# ---- structure on a synthetic snapshot (independent of the checked-in data.json)
def _synthetic():
    world, judging = "w" * 16, "j" * 16
    pending = [{"metric": m} for m in ("tests_recall", "tests_precision", "dx_listed", "disc_recall", "noop_ok")]
    clinical = {"conditions_pending": 6, "answers": {"n_pending": 13, "n_cases": 145},
                "trace": {"n_pending": 13, "n_cases": 145}}

    def track(key, models, dims, unanswered):
        return {"status": "preliminary", "not_final": True, "judging": judging, "dims": dims,
                "context": {"pack": f"{key}-pack", "date": "2026-09-29", "geometry": "g",
                            "n_cases": 145, "n_pending": 13},
                "pending_metrics": pending,
                "models": [{"solver": m, "score": .5 + i / 100, "gate_multiplier": 1.0, "n_cases": 140,
                            "dims": {d: .5 for d in dims}} for i, m in enumerate(models)],
                "samples": {m: {d: 100 for d in dims} for m in models},
                "not_ranked": [],
                "completion": {m: {"of": 145, "answered": 145 - unanswered.get(m, 0),
                                   "unanswered": unanswered.get(m, 0),
                                   "reasons": {"no_row": unanswered[m]} if m in unanswered else {},
                                   "completion": round(1 - unanswered.get(m, 0) / 145, 4)}
                               for m in models}}
    answers = ["a1", "a2", "a3"]
    trace = ["a1", "a2", "a3", "a4"]
    snap = {"current_world": True, "world_sha": world, "judging_sha16": judging, "date": "2026-09-29"}
    return {
        "board": {"dims": ["quant_ok"], "withheld": pending, "judging": judging, "profile": "p",
                  "models": [{"solver": m, "dims": {"quant_ok": .5}} for m in answers],
                  "clinical_review": clinical},
        "tool_board": {"models": [{"solver": m} for m in trace]},
        "provenance": {"is_snapshot": False, "release": {"world_sha": world, "judging_sha16": judging},
                       "snapshot": {"answers": {**snap, "pack": "answers-pack"},
                                    "trace": {**snap, "pack": "trace-pack"}},
                       "n_real_models": 3, "n_ranked_models": {"answers": 3, "trace": 4}},
        "solver_labels": {m: {"model": m} for m in trace},
        "rank_rule": {"rule": {"name": "zero", "unanswered_score": 0}},
        "preliminary": {"status": "preliminary", "not_final": True, "tracks": {
            "answers": track("answers", answers, ["tests_recall+tests_precision", "noop_ok", "quant_ok"], {"a3": 5}),
            "trace": track("trace", trace, ["tests_recall+tests_precision", "dx_listed", "disc_recall",
                                             "tool_target_grounded_rate", "noop_ok"], {"a4": 12, "a2": 1})}},
    }


def test_tracks_may_rank_different_rosters_and_dimension_counts():
    result = FIGURE.prepare_preliminary(_synthetic())
    assert [len(t["models"]) for t in result["tracks"].values()] == [3, 4]
    assert result["pending"] == ["disc_recall", "dx_listed", "noop_ok", "tests_precision", "tests_recall"]
    rows = FIGURE.csv_text(result).strip().split("\n")[1:]
    assert len(rows) == 3 * 3 + 4 * 5          # every declared model
    pending = {r.split(",")[8]: r.split(",")[-1] for r in rows}
    assert pending["tests_recall+tests_precision"] == "true" and pending["tool_target_grounded_rate"] == "false"


@pytest.mark.parametrize("mutate", [
    lambda d: d["provenance"]["n_ranked_models"].update(trace=3),
    lambda d: d["tool_board"]["models"].pop(),
    lambda d: d["preliminary"]["tracks"]["answers"]["not_ranked"].append(
        {"solver": "a1", "answered": 1, "of": 145, "coverage": .01, "why": ["x"]}),
    lambda d: d["preliminary"]["tracks"]["trace"]["completion"]["a4"].update(unanswered=11),
    lambda d: d["preliminary"]["tracks"]["trace"]["completion"].pop("a1"),
    lambda d: d["preliminary"]["tracks"]["trace"]["completion"]["a4"]["reasons"].clear(),
    lambda d: d["rank_rule"]["rule"].update(name="coverage"),
    lambda d: d["preliminary"]["tracks"]["trace"].update(pending_metrics=[{"metric": "noop_ok"}]),
    lambda d: d["preliminary"]["tracks"]["answers"].update(pending_metrics=[]),
    lambda d: d["board"]["clinical_review"]["answers"].update(n_pending=12),
    lambda d: d["preliminary"]["tracks"]["trace"].update(dims=["unknown_dim"]),
])
def test_synthetic_structure_rejects_inconsistent_rosters_and_disclosures(mutate):
    data = _synthetic()
    mutate(data)
    with pytest.raises((ValueError, KeyError)):
        FIGURE.prepare_preliminary(data)


def test_figure_footer_reads_counts_and_unanswered_cells_from_data(tmp_path):
    pytest.importorskip("matplotlib")
    FIGURE.draw(FIGURE.prepare_preliminary(_synthetic()), tmp_path)
    svg = (tmp_path / "readme_results.svg").read_text()
    assert "6 conditions await clinical review" in svg and "13/145 diagnosis" in svg
    assert "Six conditions" not in svg
    assert "Unanswered cells score 0" in svg
    assert "a3 (diagnosis 5/145)" in svg and "a4 (budgeted tools 12/145)" in svg
    assert "a2 (budgeted tools 1/145)" in svg
    assert "Not ranked" not in svg
