"""The README board figure must be drawn from the released board with half-up labels."""
from _paths import ROOT as _TEST_ROOT
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = _TEST_ROOT
SPEC = importlib.util.spec_from_file_location("readme_board", ROOT / "tools/presentation/make_readme_board.py")
FIGURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURE)
BOARD = json.loads((ROOT / "web/demo/board.json").read_text(encoding="utf-8"))


def test_labels_round_half_up_on_the_stored_decimal():
    assert FIGURE.half_up3(0.6585) == "0.659"
    assert FIGURE.half_up3(0.5615) == "0.562"
    assert FIGURE.half_up3(0.6699999999999999) == "0.670"


def test_rank_labels_show_intervals():
    assert FIGURE.rank_label({"rank_ci95": [1, 1]}) == "1"
    assert FIGURE.rank_label({"rank_ci95": [5, 6]}) == "5–6"


def test_rows_follow_the_board_order():
    rows = FIGURE.prepare(BOARD)
    assert [r["model"] for r in rows] == [r["model"] for r in BOARD["overall"]["board"]]


def test_a_board_that_is_not_final_is_refused():
    board = copy.deepcopy(BOARD)
    board["overall"]["status"] = "preliminary"
    with pytest.raises(SystemExit):
        FIGURE.prepare(board)


def test_checked_in_figure_matches_the_board(tmp_path):
    pytest.importorskip("matplotlib")
    FIGURE.draw(BOARD, tmp_path)
    assert (tmp_path / FIGURE.NAME).read_bytes() == (ROOT / "docs/figures" / FIGURE.NAME).read_bytes()
