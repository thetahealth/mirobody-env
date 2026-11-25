"""The archived 1.3.1 board table in the data card must match the shipped score readout they are rendered from.

`docs/boards/1.3.1/k3.json` is the thirteen-model k3 readout; the archived rows show rank interval,
overall score and interval, the four pack scores and All-3 (equal-weight mean of the packs' pass^3),
three decimals rounded half up on the stored decimal. All-3 averages with `math.fsum`, which rounds
the exact sum once, so the check gives the same result on every supported Python version
(`sum` of floats changed in Python 3.12)."""
from _paths import ROOT as _TEST_ROOT
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP

import pytest

ROOT = _TEST_ROOT
HISTORICAL = ROOT / "docs/DATA_CARD.md"
HEADING = "### 1.3.1 overall"
SECTIONS = {
    "README.md": ["How it works", "What it measures", "Leaderboard", "Quick start",
                  "One case, end to end", "Scoring", "Evaluate your own agent", "Documentation",
                  "Limitations", "Related work", "Data, ethics and reproduction", "Citation",
                  "License and acknowledgements"],
    "README.zh-CN.md": ["工作原理", "测什么", "排行榜", "快速开始", "一个病例，从头到尾", "计分",
                        "评测你自己的 agent", "文档", "局限", "相关工作", "数据、伦理与复现", "引用", "许可与致谢"],
}
# Presentation contracts are data rows, alongside the numeric readout checks.
LAYOUT = [
    (r"readme_board\.png|readme_findings_131\.png|findings-1\.3\.1|readme_evidence_131|readme_budget", 0),
    (r"^### 1\.6\.0 ", 1),
    (r"quickstart-timing\.json|^# .*timings:|^# .*耗时", 0),
    (r"^\| .*docs/reask\.md.*\|$", 1),
    (r"^\| .*docs/supplement\.md.*\|$", 1),
    (r"docs/boards/1\.6\.0/k1\.json", 1),
    (r"REPRODUCE\.md#recomputing-the-160-overall-board", 1),
    (r"<details>\s*<summary>[^\n]*FHIR REST[^\n]*</summary>", 1),
    (r"<details>", 5),
    (r"Interactive board \(1\.3\.0\)|交互式榜单（1\.3\.0）", 0),
]
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
    body = text.split(heading, 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
    return [(r, m, float(s), float(lo), float(hi), tuple(float(v) for v in c.strip().strip("|").split("|")))
            for r, m, s, lo, hi, c in ROW.findall(body)]


def test_archived_131_board_matches_the_readout():
    board = json.loads((ROOT / "docs/boards/1.3.1/k3.json").read_text(encoding="utf-8"))
    text = HISTORICAL.read_text(encoding="utf-8")
    assert board["aggregation"] == "k3"
    got = shown(text, HEADING)
    assert got == expected(board) and len(got) == 13
    archived = text.split(HEADING, 1)[1].split("### 1.3.0 overall", 1)[0]
    for name in ("readme_board_131.png", "readme_task_scores_131.png"):
        assert re.search(rf'<img src="figures/{name}" alt="[^"]+" width="100%">', archived)
    assert "2,000" in archived
    assert "### Recomputing the 1.3.1 overall board" in (ROOT / "docs/REPRODUCE.md").read_text()
    pk1 = json.loads((ROOT / "docs/boards/1.3.1/pk1.json").read_text())
    k3_order = [row[1] for row in got]
    pk1_order = [row[1] for row in expected(pk1)]
    assert [(i + 1, a, b) for i, (a, b) in enumerate(zip(k3_order, pk1_order)) if a != b] == [
        (8, "minimax-m3", "deepseek-v4-flash"), (9, "deepseek-v4-flash", "minimax-m3")]


@pytest.mark.parametrize("doc", sorted(SECTIONS))
def test_release_readmes_keep_the_layout(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert re.findall(r"^## (.+)$", text, re.M) == SECTIONS[doc]
    for pattern, count in LAYOUT:
        assert len(re.findall(pattern, text, re.M)) == count, pattern
    footer = ("合成数据 · 仅供评测 · 非医疗建议" if doc.endswith("zh-CN.md") else
              "Synthetic data · evaluation use only · not medical advice")
    assert text.rstrip().endswith(f'<p align="center"><sub>{footer}</sub></p>')
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    assert f"version = {{{version}}}" in text


@pytest.mark.parametrize("field", ["score", "ci", "rank_ci", "m2", "p2", "p3", "p4", "pass_k"])
def test_a_moved_reading_is_caught(field):
    board = json.loads((ROOT / "docs/boards/1.3.1/k3.json").read_text(encoding="utf-8"))
    text = HISTORICAL.read_text(encoding="utf-8")
    m = next(iter(board["overall"]))
    if field in ("ci", "rank_ci"):
        board["overall"][m][field][0] += 1 if field == "rank_ci" else .01
    elif field in ("m2", "p2", "p3", "p4"):
        board["overall"][m]["packs"][field] += .01
    elif field == "pass_k":
        board["profile"]["m2"][m][field] += .04
    else:
        board["overall"][m][field] += .01
    assert shown(text, HEADING) != expected(board)
