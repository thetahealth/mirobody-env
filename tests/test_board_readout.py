"""tools/evaluation/board_readout.py: the overall board under equal and SD-equalized weights, rank stability."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import tools.evaluation.board_readout as B  # noqa: E402

MODELS = ["a", "b", "c"]


def _samples(point, n=200, sd=0.01, seed=0):
    rnd = random.Random(seed)
    return [{m: v + rnd.gauss(0, sd) for m, v in point.items()} for _ in range(n)]


def test_sd_equal_reweights_a_wide_pack():
    # pack p4 spreads the models widely and decides the equal board; SD-equalizing turns it down
    point = {"m2": {"a": 0.50, "b": 0.60, "c": 0.40}, "pack2": {"a": 0.50, "b": 0.60, "c": 0.40},
             "p3": {"a": 0.50, "b": 0.60, "c": 0.40}, "p4": {"a": 0.95, "b": 0.30, "c": 0.10}}
    samples = {p: _samples(v, seed=i) for i, (p, v) in enumerate(point.items())}
    res = B.overall(point, samples, MODELS)
    eq = [x["model"] for x in res["equal"]["board"]]
    sd = [x["model"] for x in res["sd_equal"]["board"]]
    assert eq[0] == "a" and sd[0] == "b"
    holds = {r["model"]: r["holds"] for r in res["rank_table"]}
    assert holds == {"a": False, "b": False, "c": True}


def test_pack_seeds_differ():
    assert len({B.pack_seed(p) for p in B.PACKS}) == len(B.PACKS)


def test_a_pack_that_ties_every_model_gets_weight_zero_under_sd_equal():
    point = {"m2": {"a": 0.5, "b": 0.5, "c": 0.5}, "pack2": {"a": 0.50, "b": 0.60, "c": 0.40},
             "p3": {"a": 0.50, "b": 0.60, "c": 0.40}, "p4": {"a": 0.50, "b": 0.60, "c": 0.40}}
    samples = {p: _samples(v, seed=i) for i, (p, v) in enumerate(point.items())}
    res = B.overall(point, samples, MODELS)
    assert res["pack_sd"]["m2"] == 0
    assert [x["model"] for x in res["sd_equal"]["board"]][0] == "b"


def _batch(root, stamp, cases, solvers, reps=(1, 2, 3), copy_of=None):
    import json
    for i in reps:
        d = root / f"{stamp}-pk{i}"
        d.mkdir(parents=True)
        held = list(cases) + list(copy_of or [])
        (d / "cases.jsonl").write_text("".join(json.dumps({"case_id": c}) + "\n" for c in held), encoding="utf-8")
        (d / "eval.jsonl").write_text("".join(json.dumps({"case": c, "solver": m, "overall": "OK"}) + "\n"
                                             for c in held for m in solvers), encoding="utf-8")


def test_a_pack_split_across_batches_is_read_as_one_pack(tmp_path):
    _batch(tmp_path, "A", ["c1", "c2"], ["a", "b"])
    _batch(tmp_path, "B", ["c3", "c4"], ["a", "b"])
    rows, cases = B.pack_rows("pack2", [tmp_path / "A", tmp_path / "B"], {}, [1, 2, 3])
    assert cases == ["c1", "c2", "c3", "c4"] and len(rows) == 4 * 2 * 3
    one, one_cases = B.pack_rows("pack2", [tmp_path / "A"], {}, [1, 2, 3])
    assert one_cases == ["c1", "c2"] and one == B.load("pack2", tmp_path / "A", [1, 2, 3])


def test_a_byte_copy_contributes_only_the_items_it_owns(tmp_path):
    _batch(tmp_path, "A", ["c1", "c2"], ["a"])
    _batch(tmp_path, "B", ["c3"], ["a"], copy_of=["c1", "c2"])          # a copy holds the earlier items too
    rows, cases = B.pack_rows("pack2", [tmp_path / "A", tmp_path / "B"], {str(tmp_path / "B"): {"c3"}}, [1, 2, 3])
    assert cases == ["c1", "c2", "c3"] and len(rows) == 3 * 3
    import pytest
    with pytest.raises(SystemExit, match="share items"):
        B.pack_rows("pack2", [tmp_path / "A", tmp_path / "B"], {}, [1, 2, 3])
    with pytest.raises(SystemExit, match="does not hold"):
        B.pack_rows("pack2", [tmp_path / "A", tmp_path / "B"], {str(tmp_path / "B"): {"c9"}}, [1, 2, 3])


def test_batches_of_one_pack_must_hold_the_same_models_and_rounds(tmp_path):
    import pytest
    _batch(tmp_path, "A", ["c1"], ["a", "b"])
    _batch(tmp_path, "B", ["c2"], ["a"])
    with pytest.raises(SystemExit, match="same models and rounds"):
        B.pack_rows("pack2", [tmp_path / "A", tmp_path / "B"], {}, [1, 2, 3])
    _batch(tmp_path, "C", ["c3"], ["a", "b"], reps=(1, 2))
    with pytest.raises(SystemExit, match="same models and rounds"):
        B.pack_rows("pack2", [tmp_path / "A", tmp_path / "C"], {}, [1, 2, 3])


def test_sep_lists_every_pair_of_the_board():
    point = {"a": 0.9, "b": 0.6, "c": 0.3, "d": 0.1}
    tab = B.table(point, _samples(point), list(point))
    r = B.sep(tab, _samples(point), k=2)
    assert len(r["pairs"]) == 1 and len(r["all_pairs"]) == 6
    assert [(p["a"], p["b"]) for p in r["all_pairs"]][:3] == [("a", "b"), ("a", "c"), ("a", "d")]
    assert r["pairs"][0] == r["all_pairs"][0]
