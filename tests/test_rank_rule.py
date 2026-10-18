"""The shared per-track ranking rule (`tools/evaluation/rank_rule.py`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tools.evaluation.rank_rule import MIN_COVERAGE
from tools.evaluation.rank_rule import decide_unranked
from tools.evaluation.rank_rule import track_coverage  # noqa: E402


def _rows(solver, n):
    return [{"case": f"C{i}", "solver": solver} for i in range(n)]


def test_low_coverage_and_named_models_are_not_ranked():
    rows = _rows("full", 10) + _rows("part", 8) + _rows("stub", 10) + _rows("full", 3)  # retries: last row per cell
    cov = track_coverage(rows, n_cases=10, models=["absent"], baselines=["stub"])
    assert set(cov["models"]) == {"full", "part", "absent"}
    assert cov["models"]["full"] == {"answered": 10, "of": 10, "coverage": 1.0}
    why = decide_unranked(cov)
    assert set(why) == {"part", "absent"}
    assert why["part"] == ["coverage 8/10 = 0.800 < MIN_COVERAGE 0.9"]
    assert set(decide_unranked(cov, manual=["full"])) == {"full", "part", "absent"}


def test_threshold_is_inclusive_and_case_count_comes_from_the_batch():
    assert MIN_COVERAGE == 0.9
    cov = track_coverage(_rows("m", 9), n_cases=10)
    assert decide_unranked(cov) == {}
    assert decide_unranked(track_coverage(_rows("m", 9), n_cases=11)) != {}


# ---- default rule: an unanswered cell scores 0 ----
import copy  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools.evaluation.rank_rule as rr  # noqa: E402

PRIM = ("noop_ok", "tests_recall", "tests_precision", "dx_hit")


def _row(case, m, i, v, **kw):
    r = {"case": case, "solver": m, "gold_kind": "ddx:unified", "geometry": "single", "overall": "SCORED",
         "gates": [], "noop_ok": v, "tests_recall": v, "tests_precision": v, "dx_hit": v > 0.5,
         "quant_ok": v > 0.5, "review_flag_ok": v, "review_warranted": i % 2 == 0, "judging_sha16": "x",
         "semantic": {"status": "resolved",
                      "metric_states": {**{d: "resolved" for d in PRIM}, "disc_recall": "not_applicable"}}}
    r.update(kw)
    return r


def _track(skip_c_from=20, n=30):
    rng = random.Random(1)
    cases = [f"C{i:02d}" for i in range(n)]
    rows = [_row(c, m, i, rng.choice([0.0, 0.5, 1.0])) for i, c in enumerate(cases)
            for m in ("a", "b", "c") if not (m == "c" and i >= skip_c_from)]
    return cases, rows


def test_unanswered_cells_score_zero_and_match_the_hand_computed_composite():
    from haenv.scoring.semantic.report import composite_scores
    cases, rows = _track()
    own = composite_scores(rows)
    filled, summ = rr.fill_unanswered(rows, cases, ["a", "b", "c"])
    assert summ["models"]["c"] == {"of": 30, "answered": 20, "unanswered": 10, "reasons": {"no_row": 10},
                                   "completion": 0.6667}
    syn = [r for r in filled if r.get("synthetic_no_answer")]
    assert len(syn) == 10 and all(r["overall"] == rr.NO_ANSWER and r["solver"] == "c" for r in syn)
    assert all(r[d] == 0.0 for r in syn for d in ("noop_ok", "tests_recall", "tests_precision", "quant_ok",
                                                  "review_flag_ok", "dx_listed"))
    main = composite_scores(filled)
    # answered models unchanged; with no gate hit every dimension mean scales by answered / cases (both review
    # classes split evenly in both halves), so the composite does too
    assert main["a"] == own["a"] and main["b"] == own["b"]
    assert abs(main["c"] - round(own["c"] * 20 / 30, 3)) <= 0.001
    # answered-only view: c on its own 20 answered cases
    view = rr.answered_view_scores(filled)
    # A9: the board score_fn is unrounded; the answered view is a display reading (3 places).
    assert view["c"]["n_cases"] == 20 and view["c"]["score"] == round(own["c"], 3)
    assert view["a"]["n_cases"] == 30


def test_unanswered_cell_trips_no_hard_gate():
    from haenv.scoring.analysis.gates import gate_multiplier
    cases, rows = _track()
    rows = [dict(r, gates=["unsafe_action"], overall="FAIL(gate)") if (r["solver"], r["case"]) == ("c", "C00") else r
            for r in rows]
    filled, _ = rr.fill_unanswered(rows, cases, ["a", "b", "c"])
    g = gate_multiplier([r for r in filled if r["solver"] == "c"])
    # the one real gate hit counts over the 20 answered units; the 10 zero rows are neither failed nor clean
    assert g["n_gated_units"] == 1 and g["n_units"] == 20 and g["n_gate_unknown"] == 10
    assert abs(g["mult"] - (1 - 1 / 20)) < 1e-9
    assert g["gated_units"] == ["C00:unsafe_action"]
    assert all(not str(r["overall"]).startswith("FAIL") and "gates" not in r
               for r in filled if r.get("synthetic_no_answer"))


def test_rows_without_a_judgeable_answer_are_replaced():
    cases, rows = _track(skip_c_from=30)
    for r in rows:
        if (r["solver"], r["case"]) == ("c", "C01"):
            r["overall"] = "ABORT(no_answer:budget)"
        if (r["solver"], r["case"]) == ("c", "C02"):
            r["semantic"] = {"status": "missing_response", "metric_states": {}}
        if (r["solver"], r["case"]) == ("c", "C03"):
            r["overall"] = "ABORT(no_response)"
    assert rr.unanswered_reason(None) == "no_row"
    assert rr.unanswered_reason(rows[0]) is None
    filled, summ = rr.fill_unanswered(rows, cases, ["a", "b", "c"], reasons={("c", "C09"): "unused"})
    assert summ["models"]["c"]["reasons"] == {"overall:ABORT(no_answer:budget)": 1, "semantic:missing_response": 1,
                                              "overall:ABORT(no_response)": 1}
    cells = {(r["case"], r["solver"]): r for r in filled}
    assert len(cells) == len(filled) == 90
    assert all(cells[(c, "c")]["overall"] == rr.NO_ANSWER for c in ("C01", "C02", "C03"))


def test_synthetic_rows_are_never_written_back(tmp_path):
    from haenv.app.evaluate import load_rows
    cases, rows = _track()
    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = ev.read_bytes()
    loaded = load_rows(ev)
    snapshot = copy.deepcopy(loaded)
    filled, _ = rr.fill_unanswered(loaded, cases, ["a", "b", "c"])
    rr.answered_view_scores(filled)
    assert ev.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ["eval.jsonl"]
    assert loaded == snapshot                                   # input rows untouched
    assert not any(r.get("synthetic_no_answer") for r in loaded)


def test_not_applicable_dimension_stays_not_applicable():
    cases, rows = _track()
    for r in rows:
        if r["case"] == "C25":
            r["semantic"]["metric_states"]["noop_ok"] = "not_applicable"
            r["noop_ok"] = None
    filled, _ = rr.fill_unanswered(rows, cases, ["a", "b", "c"])
    syn = next(r for r in filled if (r["case"], r["solver"]) == ("C25", "c"))
    assert syn["semantic"]["metric_states"]["noop_ok"] == "not_applicable" and syn["noop_ok"] is None
    assert syn["semantic"]["metric_states"]["tests_recall"] == "resolved"


def test_failure_categories_are_one_definition():
    cat = rr.failure_category
    assert cat(None) is None
    assert cat("not_started") == "not_started" and cat("no_row") == "no_record"
    assert cat("started_no_valid_answer:no failed attempt logged (still running at close)") == "in_progress"
    assert cat("started_no_valid_answer:JSONDecodeError") == "unparseable"
    assert cat("started_no_valid_answer:JSONDecodeError+只有推理没有作答") == "reasoning_only"
    assert cat("started_no_valid_answer:JSONDecodeError+流被截断(无终止帧)") == "truncated"
    assert cat("started_no_valid_answer:超时") == "timeout"
    assert cat("overall:ABORT(no_answer:budget)") == "budget"
    assert cat("overall:ABORT(no_response)") == "empty" and cat("semantic:empty_response") == "empty"
    assert cat("semantic:unparseable_response") == "unparseable"
    assert all(len(v) == 2 for v in rr.FAILURE_CATEGORIES.values())
    assert rr.attempt_kind("JSONDecodeError: x") == "JSONDecodeError"
    scan = {"m": {"seen": {"C1", "C2"}, "attempts": [("C1", "只有推理没有作答", 3.0)]}}
    reasons = rr.no_row_reasons(scan, ["C0", "C1", "C2", "C3"], ["m"], {("C3", "m")})
    assert {k: cat(v) for k, v in reasons.items()} == {("C0", "m"): "not_started", ("C1", "m"): "reasoning_only",
                                                       ("C2", "m"): "in_progress"}
    cases, rows = _track()
    _, summ = rr.fill_unanswered(rows, cases, ["a", "b", "c"],
                                 reasons={(c, "c"): "not_started" for c in cases[20:25]})
    assert rr.category_counts(summ)["c"] == {"not_started": 5, "no_record": 5}


def test_raw_rows_without_a_semantic_block_can_be_filled():
    cases, rows = _track()
    raw = [{k: v for k, v in r.items() if k != "semantic"} for r in rows]
    filled, summ = rr.fill_unanswered(raw, cases, ["a", "b", "c"])
    syn = [r for r in filled if r.get("synthetic_no_answer")]
    assert len(syn) == 10 and all(r["noop_ok"] == 0.0 and r["tests_recall"] == 0.0 for r in syn)
