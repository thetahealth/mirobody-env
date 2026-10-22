"""The README board figure must be drawn from the released board with half-up labels."""
from _paths import ROOT as _TEST_ROOT
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = _TEST_ROOT
SPEC = importlib.util.spec_from_file_location("readme_board", ROOT / "tools/presentation/make_readme_board.py")
FIGURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURE)
import tools.presentation.make_readme_task_scores as TASKS
BOARD = json.loads((ROOT / "web/demo/board-1.3.0.json").read_text(encoding="utf-8"))


def test_labels_round_half_up_on_the_stored_decimal():
    assert FIGURE.half_up3(0.6585) == "0.659"
    assert FIGURE.half_up3(0.5615) == "0.562"
    assert FIGURE.half_up3(0.6699999999999999) == "0.670"


def test_rank_labels_show_intervals():
    assert FIGURE.rank_label({"rank_ci95": [1, 1]}) == "1"
    assert FIGURE.rank_label({"rank_ci95": [5, 6]}) == "5–6"


@pytest.mark.parametrize("source,kind", [
    ("web/demo/board-1.3.0.json", "board"),
    ("web/demo/board.json", "released"),
    ("docs/boards/1.3.1/k3.json", "tasks"),
])
def test_rows_follow_the_board_order(source, kind):
    board = json.loads((ROOT / source).read_text(encoding="utf-8"))
    if kind in ("board", "released"):
        rows = FIGURE.prepare(board, .1 if kind == "released" else .3, .9)
        assert [r["model"] for r in rows] == [r["model"] for r in board["overall"]["board"]]
    else:
        rows = TASKS.prepare(board)
        assert [model for model, values in rows] == [r["model"] for r in FIGURE.prepare(FIGURE.from_score(board), .1, .9)]
        for model, values in rows:
            assert values[:4] == [board["overall"][model]["packs"][p] for p in ("m2", "p2", "p3", "p4")]
            assert values[4] == math.fsum(board["profile"][p][model]["pass_k"] for p in ("m2", "p2", "p3", "p4")) / 4


def test_a_board_that_is_not_final_is_refused():
    board = copy.deepcopy(BOARD)
    board["overall"]["status"] = "preliminary"
    with pytest.raises(SystemExit):
        FIGURE.prepare(board)


@pytest.mark.parametrize("source,name,axis", [
    ("web/demo/board-1.3.0.json", "readme_board.png", (.3, .9)),
    ("web/demo/board.json", "readme_board_131.png", (.1, .9)),
    ("docs/boards/1.3.1/k3.json", "readme_board_131.png", (.1, .9)),
    ("docs/boards/1.3.1/k3.json", "readme_task_scores_131.png", None),
])
def test_checked_in_figure_matches_the_board(tmp_path, source, name, axis):
    pytest.importorskip("matplotlib")
    board = json.loads((ROOT / source).read_text(encoding="utf-8"))
    if axis is None:
        from matplotlib.colors import to_rgb
        fig = TASKS.draw(board, tmp_path)
        cells = fig.artists
        labels = [t for t in fig.texts if t.get_text().replace(".", "", 1).isdigit()]
        assert len(cells) == len(labels) == 13 * 5
        for cell, label in zip(cells, labels):
            bg = TASKS.luminance(cell.get_facecolor())
            fg = TASKS.luminance(to_rgb(label.get_color()))
            assert (max(bg, fg) + .05) / (min(bg, fg) + .05) >= 4.5, label.get_text()
        # The reported dark-cell regressions must use white text on their actual colours.
        for index, value in [(8, "0.785"), (13, "0.807"), (18, "0.744"), (46, "0.813"),
                             (51, "0.767"), (4, "0.823"), (14, "0.760")]:
            assert labels[index].get_text() == value
            assert labels[index].get_color() == "#ffffff"
        from PIL import Image
        # Shown at the full README column (about 830 px), the smallest text stays at 11 px or more.
        width_in = Image.open(tmp_path / name).width / 200
        assert min(t.get_fontsize() for t in fig.texts) * 830 / (width_in * 72) >= 11
        assert [t.get_text() for t in labels] == [FIGURE.half_up3(v) for _, values in TASKS.prepare(board) for v in values]
    else:
        if source.endswith("k3.json"):
            board = FIGURE.from_score(board)
        FIGURE.draw(board, tmp_path, name, *axis)
    assert (tmp_path / name).read_bytes() == (ROOT / "docs/figures" / name).read_bytes()
