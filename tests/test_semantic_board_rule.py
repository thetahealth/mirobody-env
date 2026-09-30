"""Board rule for a semantic track: R0 on common complete cases, R1 fill bounds, tiers by paired bootstrap."""
import pytest

from haenv.semantic_report import BOARD_RULE, PRIMARY, holm, phase5_composite_allowed, track_board

MODELS = ("m-a", "m-b", "m-c", "m-d", "m-e")


def cell(case, model, values, states=None, status="resolved"):
    states = states or {}
    row = {"case": case, "solver": model,
           "semantic": {"status": status, "metric_states": {d: states.get(d, "resolved") for d in PRIMARY}}}
    for d in PRIMARY:
        row[d] = None if states.get(d, "resolved") != "resolved" else values[d]
    return row


def mean_score(rows):
    """Stand-in composite: per model, mean over rows of the mean of its primary values."""
    by = {}
    for r in rows:
        vals = [r[d] for d in PRIMARY if r["semantic"]["metric_states"][d] == "resolved"]
        by.setdefault(r["solver"], []).append(sum(vals) / len(vals) if vals else 0.0)
    return {m: sum(v) / len(v) for m, v in by.items()}


def track(n_cases, missing, levels=None):
    """`missing` maps model -> list of case indices whose tests_precision is unresolved."""
    levels = levels or {"m-a": 0.9, "m-b": 0.7, "m-c": 0.69, "m-d": 0.5, "m-e": 0.3}
    rows = []
    for i in range(n_cases):
        for m in MODELS:
            wobble = ((i * 7 + MODELS.index(m) * 3) % 5 - 2) * 0.02
            v = min(1.0, max(0.0, levels[m] + wobble))
            states = {"tests_precision": "unresolved"} if i in missing.get(m, ()) else {}
            rows.append(cell(f"C{i:02d}", m, {d: v for d in PRIMARY}, states,
                             "unresolved" if states else "resolved"))
    return rows


def board(rows):
    return track_board(rows, score_fn=mean_score, boot=400, seed=7)


def test_rule_constants_are_the_preregistered_ones():
    assert BOARD_RULE == {"min_common_cases": 20, "boot": 10000, "seed": 20260929,
                          "r1_max_width": 0.05, "alpha": 0.05, "phase5_min_cases": 10}
    assert phase5_composite_allowed(10) and not phase5_composite_allowed(9)


def test_r0_scores_common_cases_with_intervals_and_tiers():
    out = board(track(25, {"m-e": [0]}))            # 24 common complete cases
    assert out["rule"] == "R0" and out["ranked"] and out["n_cases"] == 24
    by = {m["model"]: m for m in out["models"]}
    assert by["m-a"]["tier"] == 1
    assert by["m-b"]["tier"] == by["m-c"]["tier"] == 2        # 0.70 vs 0.69: not separable
    assert by["m-d"]["tier"] == 4 and by["m-e"]["tier"] == 5
    lo, hi = by["m-a"]["ci95"]
    assert lo <= by["m-a"]["score"] <= hi
    assert "imputed_cells" not in by["m-a"]


def test_r1_withholds_the_ranking_when_a_fill_interval_is_wide():
    # 21 of 40 cases incomplete, all on one model: its fill bounds differ by 21 / 120 = 0.175.
    out = board(track(40, {"m-e": list(range(21))}))
    assert out["rule"] == "R1" and out["ranked"] is False
    assert out["n_common_complete"] == 19 and out["n_cases"] == 40
    by = {m["model"]: m for m in out["models"]}
    assert by["m-e"]["imputed_cells"] == 21 and by["m-a"]["imputed_cells"] == 0
    assert by["m-e"]["fill_width"] > 0.05
    assert all("tier" not in m for m in out["models"])
    assert "0.05" in out["why"]


def test_r1_ranks_when_every_fill_interval_is_narrow_and_separation_holds_under_both_fills():
    spread = {m: [i for i in range(21) if i % len(MODELS) == k] for k, m in enumerate(MODELS)}
    out = board(track(40, spread))                  # at most 5 imputed cells per model: width <= 0.042
    assert out["rule"] == "R1" and out["ranked"] is True and out["n_common_complete"] == 19
    by = {m["model"]: m for m in out["models"]}
    assert all(m["fill_width"] <= 0.05 for m in out["models"])
    assert sum(m["imputed_cells"] for m in out["models"]) == 21
    lo, hi = by["m-a"]["fill_interval"]
    assert lo < hi and set(by["m-a"]["ci95"]) == {"fill_0", "fill_1"}
    assert by["m-a"]["tier"] == 1 and by["m-b"]["tier"] == by["m-c"]["tier"] == 2
    # separation must hold under both fills
    for pair in out["significant_pairs"]:
        assert pair["fill_0"] and pair["fill_1"]


def test_r1_uses_only_cases_every_model_answered():
    rows = track(40, {"m-e": [0, 1]})
    for r in rows:
        if r["case"] in ("C00", "C01") and r["solver"] == "m-e":
            r["semantic"]["status"] = "missing_response"
            r["semantic"]["metric_states"] = {d: "missing_response" for d in PRIMARY}
    rows = [r for r in rows if int(r["case"][1:]) < 20]     # 18 common complete -> R1
    out = board(rows)
    assert out["rule"] == "R1" and out["n_cases"] == 18 and out["unanswered_cases"] == ["C00", "C01"]
    assert all(m["imputed_cells"] == 0 for m in out["models"])


def test_holm_step_down_is_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    rows = track(25, {})
    assert board(rows) == board(rows)


def test_r1_separation_under_one_fill_only_does_not_split_a_tier():
    # p beats q only if its 5 missing cells are filled with 1; filled with 0 it does not.
    levels = {"m-p": 0.6, "m-q": 0.59, "m-s": 0.9, "m-t": 0.2, "m-u": 0.1, "m-v": 0.35}
    missing = {"m-p": range(0, 5), "m-s": range(5, 10), "m-t": range(10, 15), "m-u": range(15, 20),
               "m-v": range(20, 21)}
    rows = []
    for i in range(40):
        for m, level in levels.items():
            states = {"tests_precision": "unresolved"} if i in missing.get(m, ()) else {}
            rows.append(cell(f"C{i:02d}", m, {d: level for d in PRIMARY}, states,
                             "unresolved" if states else "resolved"))
    out = track_board(rows, score_fn=mean_score, boot=2000, seed=11)
    assert out["rule"] == "R1" and out["ranked"] and out["n_common_complete"] == 19
    by = {m["model"]: m for m in out["models"]}
    assert by["m-p"]["tier"] == by["m-q"]["tier"]
    assert {"better": "m-p", "worse": "m-q", "fill": "fill_1"} in out["one_fill_only"]
    assert not any(x["better"] == "m-p" and x["worse"] == "m-q" for x in out["significant_pairs"])
