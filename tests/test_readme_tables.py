"""The 1.x board tables in both READMEs must match the 1.x snapshot they are copied from.

The README keeps the 1.x board as a historical record; `web/demo/data.json` carries the same
board under `history_1x`. The Markdown tables are written by hand, so without this file a changed
score, interval, tier or answered count in the README passes every other test.
"""
import copy
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADINGS = {"README.md": "### {} (1.x)", "README.zh-CN.md": "### {}（1.x）"}
ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)–([\d.]+)\s*\|"
                 r"\s*(\d+)\s*\|\s*(\d+) / (\d+)\s*\|", re.M)


@pytest.fixture(scope="module")
def data():
    return json.loads((ROOT / "web/demo/data.json").read_text(encoding="utf-8"))


def expected(data, key):
    """Rows as the README shows them: score descending, ties alphabetical and sharing a rank."""
    track = data["history_1x"]["tracks"][key]
    rows = []
    for i, (model, v) in enumerate(sorted(track["models"].items(), key=lambda kv: (-kv[1]["score"], kv[0]))):
        rank = rows[-1][0] if rows and rows[-1][2] == round(v["score"], 3) else i + 1
        rows.append((rank, model, round(v["score"], 3), round(v["ci95"][0], 3),
                     round(v["ci95"][1], 3), v["tier"], v["answered"], v["of"]))
    return track["pack"], rows


def shown(text, heading):
    section = text.split(heading, 1)[1].split("\n###", 1)[0]
    return [(int(r), m, float(s), float(lo), float(hi), int(t), int(a), int(o))
            for r, m, s, lo, hi, t, a, o in ROW.findall(section)]


@pytest.mark.parametrize("doc", sorted(HEADINGS))
@pytest.mark.parametrize("key", ["answers", "trace"])
def test_readme_table_matches_the_snapshot(data, doc, key):
    pack, want = expected(data, key)
    got = shown((ROOT / doc).read_text(encoding="utf-8"), HEADINGS[doc].format(pack))
    assert len(got) == len(want) > 0, f"{doc} {pack}: {len(got)} rows, snapshot has {len(want)}"
    assert got == want


def _render(heading, rows):
    body = "".join(f"| {r} | {m} | {s:.3f} | {lo:.3f}–{hi:.3f} | {t} | {a} / {o} |\n"
                   for r, m, s, lo, hi, t, a, o in rows)
    return f"{heading}\n\n| Rank | Model |\n|---:|---|\n{body}\n### next\n"


def test_a_changed_reading_is_caught(data):
    """Negative control on a table rendered from the snapshot, so it does not depend on the
    README being in sync: the parser reads it back exactly, and moving any one field breaks it."""
    pack, want = expected(data, "answers")
    heading = HEADINGS["README.md"].format(pack)
    assert shown(_render(heading, want), heading) == want
    for field in (2, 3, 5, 6):
        moved = [list(r) for r in want]
        moved[0][field] += 1
        assert shown(_render(heading, [tuple(r) for r in moved]), heading) != want, field


def test_a_changed_snapshot_is_caught(data):
    """Negative control on the other side: moving the snapshot must also break the match."""
    moved = copy.deepcopy(data)
    first = next(iter(moved["history_1x"]["tracks"]["trace"]["models"].values()))
    first["tier"] += 1
    pack, want = expected(moved, "trace")
    got = shown((ROOT / "README.md").read_text(encoding="utf-8"),
                HEADINGS["README.md"].format(pack))
    assert got != want


# ── the current board: both READMEs against `web/demo/board.json` (the demo's board data) ──
BOARD_HEADINGS = {"README.md": ("overall", "by pack"), "README.zh-CN.md": ("总分", "分包")}
VERSION = r"\d+\.\d+\.\d+"
NUM = r"-?[\d.]+"
OVERALL_ROW = re.compile(rf"^\|\s*(\d+(?:–\d+)?)\s*\|\s*(\S+)\s*\|\s*({NUM})\s*\|\s*({NUM})–({NUM})\s*\|\s*$", re.M)
PACK_ROW = re.compile(rf"^\|\s*(\S+)\s*\|((?:\s*{NUM} \({NUM}–{NUM}\)\s*\|){{4}})\s*$", re.M)
CELL = re.compile(rf"({NUM}) \(({NUM})–({NUM})\)")
PACKS = ("m2", "pack2", "p3", "p4")


@pytest.fixture(scope="module")
def board():
    return json.loads((ROOT / "web/demo/board.json").read_text(encoding="utf-8"))


def _section(text, name):
    """The body under the heading `### <version> <name>`, up to the next heading."""
    m = re.search(rf"^### {VERSION} {re.escape(name)}\s*$", text, re.M)
    assert m, f"no `### <version> {name}` heading"
    return text[m.end():].split("\n###", 1)[0].split("\n## ", 1)[0]


def _rank(x):
    lo, hi = x["rank_ci95"]
    return str(lo) if lo == hi else f"{lo}–{hi}"


def _half_up3(x):
    """Three decimals, half up on the stored decimal: the rule the README table and the demo use."""
    return float(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def board_expected(board):
    rows = []
    for r in board["overall"]["board"]:
        scores = tuple(_half_up3(next(x["score"] for x in board["packs"][k]["board"] if x["model"] == r["model"])) for k in PACKS)
        rows.append((_rank(r), r["model"], _half_up3(r["score"]), _half_up3(r["ci95"][0]), _half_up3(r["ci95"][1]),
                     scores + (_half_up3(r["pass3"]),)))
    return rows


def board_shown(text, doc):
    section = _section(text, BOARD_HEADINGS[doc][0])
    pattern = re.compile(rf"^\|\s*(\d+(?:–\d+)?)\s*\|\s*(\S+)\s*\|\s*({NUM}) \(({NUM})–({NUM})\)\s*\|((?:\s*{NUM}\s*\|){{5}})\s*$", re.M)
    return [(r,m,float(score),float(lo),float(hi),tuple(float(v.strip()) for v in cells.strip().strip("|").split("|")))
            for r,m,score,lo,hi,cells in pattern.findall(section)]


@pytest.mark.parametrize("doc", sorted(BOARD_HEADINGS))
def test_readme_board_matches_the_demo_board(board, doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert all(p["status"] == "final" for p in board["packs"].values()) and board["overall"]["status"] == "final"
    assert board_shown(text, doc) == board_expected(board)
    assert len(board_expected(board)) == len(board["models"]) > 0
    assert board["source"]["aggregation"] == "k3"
    assert "k3" in text and "pk1" in text and "N100" in text
    assert all(p["n_cases"] == 100 and p["n_rounds"] == 3 for p in board["packs"].values())
    assert f"{board['source']['boot']:,}" in text, f"{doc} does not state the bootstrap draw count"
    assert "All-3" in text


def test_pass3_is_the_equal_weight_mean_of_the_pack_profiles(board):
    """The pass^3 column is a profile copied from the score readout, not a ranking input."""
    for r in board["overall"]["board"]:
        packs = [board["profile"][p][r["model"]] for p in board["profile"]]
        assert all(x["k"] == 3 for x in packs)
        assert abs(r["pass3"] - sum(x["pass_k"] for x in packs) / len(packs)) < 1e-6


def test_the_board_carries_no_alternative_weighting(board):
    """The published board shows the equal-weight total only; the SD-equalized variant is a
    sensitivity analysis in the reproduction notes."""
    blob = json.dumps(board)
    assert "sd_equal" not in blob and "holds" not in blob


def test_a_changed_reading_in_the_board_is_caught(board):
    """Moving any displayed score, interval, pack value or rank breaks the match."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert board_shown(text, "README.md") == board_expected(board)
    for field in ("score", "ci95", "rank_ci95"):
        moved = copy.deepcopy(board)
        row = moved["overall"]["board"][0]
        if isinstance(row[field], list):
            row[field][1] += .01 if field == "ci95" else 1
        else:
            row[field] += .01
        assert board_expected(moved) != board_shown(text, "README.md")
    moved = copy.deepcopy(board)
    moved["packs"]["p4"]["board"][0]["score"] += .01
    assert board_expected(moved) != board_shown(text, "README.md")
