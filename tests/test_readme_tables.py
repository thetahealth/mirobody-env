"""The 1.x board tables in both READMEs must match the 1.x snapshot they are copied from.

The README keeps the 1.x board as a historical record; `web/demo/data.json` carries the same
board under `history_1x`. The Markdown tables are written by hand, so without this file a changed
score, interval, tier or answered count in the README passes every other test.
"""
import copy
import json
import re
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


# ── the 1.2.0 board: both READMEs against `web/demo/board.json` (the demo's board data) ──
BOARD_HEADINGS = {"README.md": ("### 1.2.0 overall", "### 1.2.0 by pack"),
                  "README.zh-CN.md": ("### 1.2.0 总分", "### 1.2.0 分包")}
OVERALL_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)–([\d.]+)\s*\|\s*$", re.M)
PACK_ROW = re.compile(r"^\|\s*(\S+)\s*\|((?:\s*[\d.]+ \([\d.]+–[\d.]+\)\s*\|){4})\s*$", re.M)
CELL = re.compile(r"([\d.]+) \(([\d.]+)–([\d.]+)\)")
PACKS = ("m2", "pack2", "p3", "p4")


@pytest.fixture(scope="module")
def board():
    return json.loads((ROOT / "web/demo/board.json").read_text(encoding="utf-8"))


def _section(text, heading):
    return text.split(heading, 1)[1].split("\n###", 1)[0].split("\n## ", 1)[0]


def board_expected(board):
    r3 = lambda v: round(v, 3)  # noqa: E731
    overall = [(x["rank"], x["model"], r3(x["score"]), r3(x["ci95"][0]), r3(x["ci95"][1])) for x in board["overall"]["board"]]
    by = {k: {x["model"]: (r3(x["score"]), r3(x["ci95"][0]), r3(x["ci95"][1])) for x in board["packs"][k]["board"]} for k in PACKS}
    packs = [(x["model"], tuple(by[k][x["model"]] for k in PACKS)) for x in board["overall"]["board"]]
    return overall, packs


def board_shown(text, doc):
    h1, h2 = BOARD_HEADINGS[doc]
    overall = [(int(r), m, float(s), float(lo), float(hi)) for r, m, s, lo, hi in OVERALL_ROW.findall(_section(text, h1))]
    packs = [(m, tuple(tuple(float(v) for v in c) for c in CELL.findall(cells))) for m, cells in PACK_ROW.findall(_section(text, h2))]
    return overall, packs


@pytest.mark.parametrize("doc", sorted(BOARD_HEADINGS))
def test_readme_1_2_0_board_matches_the_demo_board(board, doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert all(p["status"] == "final" for p in board["packs"].values()) and board["overall"]["status"] == "final"
    want_o, want_p = board_expected(board)
    got_o, got_p = board_shown(text, doc)
    assert len(got_o) == len(want_o) > 0 and got_o == want_o, doc
    assert len(got_p) == len(want_p) and got_p == want_p, doc
    assert board["source"]["pack_seed_sha256"] in text, f"{doc} does not show the board's pack-seed hash"
    assert f"{board['source']['boot']:,}" in text, f"{doc} does not state the bootstrap draw count"


def test_a_changed_1_2_0_reading_is_caught(board):
    """Negative control on both sides: a README rendered from the board reads back exactly; moving one
    reading in the board breaks the match with the committed README."""
    want_o, want_p = board_expected(board)
    h1, h2 = BOARD_HEADINGS["README.md"]
    o = "".join(f"| {r} | {m} | {s:.3f} | {lo:.3f}–{hi:.3f} |\n" for r, m, s, lo, hi in want_o)
    p = "".join(f"| {m} | " + " | ".join(f"{s:.3f} ({lo:.3f}–{hi:.3f})" for s, lo, hi in cells) + " |\n" for m, cells in want_p)
    text = f"{h1}\n\n| a |\n|---|\n{o}\n{h2}\n\n| a |\n|---|\n{p}\n## next\n"
    assert board_shown(text, "README.md") == (want_o, want_p)
    moved = copy.deepcopy(board)
    moved["packs"]["p4"]["board"][0]["ci95"][1] += 0.01
    assert board_expected(moved)[1] != board_shown((ROOT / "README.md").read_text(encoding="utf-8"), "README.md")[1]
