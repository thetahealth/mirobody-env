"""The 1.3.1 board tables in both READMEs must match the shipped score readout they are rendered from.

`docs/boards/1.3.1/k3.json` is the thirteen-model k3 readout; the README rows show rank interval,
overall score and interval, the four pack scores and All-3 (equal-weight mean of the packs' pass^3),
three decimals rounded half up on the stored decimal. All-3 averages with `math.fsum`, which rounds
the exact sum once, so the check gives the same result on every supported Python version
(`sum` of floats changed in Python 3.12)."""
from _paths import ROOT as _TEST_ROOT
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pytest

ROOT = _TEST_ROOT
HEADINGS = {"README.md": "### 1.3.1 board", "README.zh-CN.md": "### 1.3.1 榜单"}
ROW = re.compile(r"^\|\s*(\d+(?:–\d+)?)\s*\|\s*(\S+)\s*\|\s*([\d.]+) \(([\d.]+)–([\d.]+)\)\s*\|((?:\s*[\d.]+\s*\|){5})\s*$", re.M)


def _h3(v):
    return float(Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def expected(board):
    out = []
    ranked = sorted((m for m, r in board["overall"].items() if r["score"] is not None),
                    key=lambda m: (board["overall"][m]["rank"], m))
    for m in ranked:
        r = board["overall"][m]
        lo, hi = (int(v) for v in r["rank_ci"])
        all3 = math.fsum(board["profile"][p][m]["pass_k"] for p in ("m2", "p2", "p3", "p4")) / 4
        out.append((str(lo) if lo == hi else f"{lo}–{hi}", m, _h3(r["score"]), _h3(r["ci"][0]), _h3(r["ci"][1]),
                    tuple(_h3(r["packs"][p]) for p in ("m2", "p2", "p3", "p4")) + (_h3(all3),)))
    return out


def shown(text, heading):
    assert heading in text, heading
    body = text.split(heading, 1)[1].split("\n### ", 1)[0]
    return [(r, m, float(s), float(lo), float(hi), tuple(float(v) for v in c.strip().strip("|").split("|")))
            for r, m, s, lo, hi, c in ROW.findall(body)]


@pytest.mark.parametrize("doc", sorted(HEADINGS))
def test_readme_131_board_matches_the_readout(doc):
    board = json.loads((ROOT / "docs/boards/1.3.1/k3.json").read_text(encoding="utf-8"))
    assert board["aggregation"] == "k3"
    got = shown((ROOT / doc).read_text(encoding="utf-8"), HEADINGS[doc])
    assert got == expected(board) and len(got) == 13


def test_a_moved_reading_is_caught():
    board = json.loads((ROOT / "docs/boards/1.3.1/k3.json").read_text(encoding="utf-8"))
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = next(iter(board["overall"]))
    board["overall"][m]["score"] += .01
    assert shown(text, HEADINGS["README.md"]) != expected(board)
