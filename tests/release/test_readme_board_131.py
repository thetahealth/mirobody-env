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
HEADINGS = {"README.md": "### 1.3.1 overall", "README.zh-CN.md": "### 1.3.1 总分"}
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
    (r'<p align="center">\s*<img src="docs/figures/readme_board_131.png" alt="[^"]+" width="100%">\s*</p>', 1),
    (r'<p align="center">\s*<img src="docs/figures/readme_task_scores_131.png" alt="[^"]+" width="100%">\s*</p>', 1),
    (r"readme_board\.png|readme_findings_131\.png|findings-1\.3\.1|readme_evidence_131|readme_budget", 0),
    (r"^### 1\.3\.\d+ ", 1),
    (r"quickstart-timing\.json|^# .*timings:|^# .*耗时", 0),
    (r"^\| .*docs/reask\.md.*\|$", 1),
    (r"^\| .*docs/supplement\.md.*\|$", 1),
    (r"docs/boards/1\.3\.1/k3\.json", 1),
    (r"REPRODUCE\.md#recomputing-the-131-overall-board", 1),
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


@pytest.mark.parametrize("doc", sorted(HEADINGS))
def test_readme_131_board_matches_the_readout(doc):
    board = json.loads((ROOT / "docs/boards/1.3.1/k3.json").read_text(encoding="utf-8"))
    assert board["aggregation"] == "k3"
    text = (ROOT / doc).read_text(encoding="utf-8")
    got = shown(text, HEADINGS[doc])
    assert got == expected(board) and len(got) == 13
    assert re.findall(r"^## (.+)$", text, re.M) == SECTIONS[doc]
    for pattern, count in LAYOUT:
        assert len(re.findall(pattern, text, re.M)) == count, pattern
    leaderboard = text.split(HEADINGS[doc])[1].split("\n## ")[0]
    legend_end = leaderboard.index("</sub>")
    heatmap = leaderboard.index("<p align=\"center\">", legend_end)
    assert leaderboard[legend_end + len("</sub>"):heatmap].strip() == ""
    alt = ("四个任务的得分与 All-3，13 个模型按总榜顺序排列" if doc.endswith("zh-CN.md") else
           "Scores for the four tasks and All-3, with thirteen models in overall board order")
    assert f'<img src="docs/figures/readme_task_scores_131.png" alt="{alt}" width="100%">' in leaderboard
    footer = ("合成数据 · 仅供评测 · 非医疗建议" if doc.endswith("zh-CN.md") else
              "Synthetic data · evaluation use only · not medical advice")
    assert text.rstrip().endswith(f'<p align="center"><sub>{footer}</sub></p>')
    # The English header stays short so the interval column fits on one line; the legend defines All-3.
    assert ("| ④ | All-3（三次全对） |" if doc.endswith("zh-CN.md") else "| ④ | All-3 |") in text
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    assert f"version = {{{version}}}" in text and "2,000" in text and "N100" in text
    assert "### Recomputing the 1.3.1 overall board" in (ROOT / "docs/REPRODUCE.md").read_text()
    pk1 = json.loads((ROOT / "docs/boards/1.3.1/pk1.json").read_text())
    k3_order = [row[1] for row in got]
    pk1_order = [row[1] for row in expected(pk1)]
    assert [(i + 1, a, b) for i, (a, b) in enumerate(zip(k3_order, pk1_order)) if a != b] == [
        (8, "minimax-m3", "deepseek-v4-flash"), (9, "deepseek-v4-flash", "minimax-m3")]
    intro = text.split(HEADINGS[doc])[0].split("\n## ")[-1]
    assert all(model in intro for model in ("minimax-m3", "deepseek-v4-flash"))
    assert ("第 8、9 名" if doc.endswith("zh-CN.md") else "positions 8 and 9") in intro
    assert "gives the same model order" not in intro and "pk1 名次相同" not in intro
    pairs = board["adjacent_pairs"]
    separable = sum(pair["separable"] for pair in pairs)
    uncertainty = (f"{len(pairs)} 对相邻模型中，只有 {separable} 对" if doc.endswith("zh-CN.md") else
                   f"Only {separable} of {len(pairs)} adjacent")
    assert uncertainty in text


@pytest.mark.parametrize("field", ["score", "ci", "rank_ci", "m2", "p2", "p3", "p4", "pass_k"])
def test_a_moved_reading_is_caught(field):
    board = json.loads((ROOT / "docs/boards/1.3.1/k3.json").read_text(encoding="utf-8"))
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = next(iter(board["overall"]))
    if field in ("ci", "rank_ci"):
        board["overall"][m][field][0] += 1 if field == "rank_ci" else .01
    elif field in ("m2", "p2", "p3", "p4"):
        board["overall"][m]["packs"][field] += .01
    elif field == "pass_k":
        board["profile"]["m2"][m][field] += .04
    else:
        board["overall"][m][field] += .01
    assert shown(text, HEADINGS["README.md"]) != expected(board)
