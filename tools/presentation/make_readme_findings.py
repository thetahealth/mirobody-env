"""Render the README "1.3.1 findings" sections from `docs/boards/1.3.1/findings.json`.

The JSON holds the released readings: evidence coverage of wrong diagnoses, re-asking with the
missed results, the test-budget sweep and the FHIR replay. This script copies them into
`README.md` and `README.zh-CN.md` between the `findings-1.3.1` markers and computes nothing.

    python tools/presentation/make_readme_findings.py           # rewrite both blocks
    python tools/presentation/make_readme_findings.py --check   # exit 1 if a block is stale
"""
from __future__ import annotations

import argparse
import json
import io
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/boards/1.3.1/findings.json"
BOARD_START, BOARD_END = "<!-- board-1.3.1:start -->", "<!-- board-1.3.1:end -->"
FIGURE = ROOT / "docs/figures/readme_findings_131.png"
START, END = "<!-- findings-1.3.1:start -->", "<!-- findings-1.3.1:end -->"


def _signed(v: str) -> str:
    return v if v.startswith("−") or float(v) == 0 else "+" + v


def _budget_rows(f: dict) -> list[str]:
    rows = sorted(f["budget"]["models"], key=lambda r: (-float(r["gain_pp"].replace("−", "-")), r["model"]))
    return [f"| {r['model']} | " + " | ".join(f"{r['accuracy_pct'][lvl]}%" for lvl in f["budget"]["levels"])
            + f" | {_signed(r['gain_pp'])} [{r['gain_ci_pp'][0]}, {r['gain_ci_pp'][1]}] | {r['used_at_1x_pct']}% |"
            for r in rows]


def render_en(f: dict) -> str:
    c, r, h, p = f["coverage"], f["reask"], f["reask"]["half"], f["replay"]
    by = ", ".join(f"{m} {n}" for m, n in r["by_model"].items())
    lo, hi = r["decisive_minus_placebo_ci_pp"]
    return "\n".join([
        START,
        "",
        "## 1.3.1 findings",
        "",
        '![Evidence coverage among wrong diagnoses and re-ask correction rates](docs/figures/readme_findings_131.png)',
        "",
        f"**Wrong diagnoses usually lack evidence.** Of the wrong diagnoses (aborts excluded) on items with a list "
        f"of decisive results, {c['wrong_incomplete']} had not obtained all of those results and {c['wrong_complete']} "
        f"all of them: {c['wrong_incomplete_pct']}% lacked part of the decisive evidence.",
        "",
        f"**Supplying the missed results corrects most of them.** {r['n']} wrong diagnoses with incomplete coverage "
        f"({by}) were re-asked from just before the final answer. With the missed decisive results added, "
        f"{r['decisive_pct']}% became correct, against {r['placebo_pct']}% with as many non-decisive results and "
        f"{r['resample_pct']}% with nothing added ({r['decisive_minus_placebo_pp']} points over non-decisive results, "
        f"95% CI {lo} to {hi}). On the {h['n']} that missed at least two decisive results, adding a random half of them "
        f"corrected {h['half_pct']}% (nothing added {h['resample_pct']}%, all {h['all_pct']}%).",
        "",
        f"**A larger test budget helps some models more than others.** Diagnosis accuracy on the "
        f"{f['budget']['items']} diagnosis items (first answers) at half, the default (the board) and twice the "
        "default test budget. Gain: percentage points from half to twice the budget, with its paired 95% interval; "
        "budget used: mean share of the default budget spent.",
        "",
        "| Model | 0.5× | 1× | 2× | Gain [95% CI] | Budget used at 1× |",
        "|---|---:|---:|---:|---:|---:|",
        *_budget_rows(f),
        "",
        f"**HTTP replay reproduces the stored tool results.** Replaying the {p['calls']} tool calls of the "
        f"{p['sessions']} leaderboard answer sessions on the {p['questions']} items through `haenv serve fhir` and decoding "
        f"the responses gives identical output to the function-call tools in {p['identical']} of {p['calls']} calls "
        f"and reproduces every tool result stored in those sessions' traces, purchases and refusals included; "
        f"the {p['resources']} returned resources are valid FHIR {p['fhir']} ({p['invalid']} invalid). "
        "Run [`tools/release/replay_http.py`](tools/release/replay_http.py) on the leaderboard pool:",
        "",
        "```bash",
        "python tools/release/replay_http.py --work leaderboard-pool/ --workers 32 --out artifacts/replay-http-results.json",
        "```",
        "",
        f"Expected summary: `questions={p['questions'].replace(',', '')}`, `responses={p['sessions'].replace(',', '')}`, "
        f"`calls={p['calls'].replace(',', '')}`, `identical={p['identical'].replace(',', '')}`, "
        f"`differing={p['differing']}`, `stored_differing={p['stored_differing']}`, "
        f"`pending_purchases_replayed={p['pending_purchases']}`, `validation_failures={p['invalid']}`; "
        "`missing=[]`, `worker_errors=[]`. The pool contains the stored leaderboard runs and is supplied separately "
        "from the source tree. An offline HTTP rehearsal is `python tools/release/replay_http.py --offline`.",
        "",
        "Recompute and re-score the released re-ask answers offline: "
        "`python tools/evaluation/reask_pooled.py --check` "
        "([data and conditions](benchmarks/results/1.3.1/reask/README.md)).",
        END,
    ])


def render_zh(f: dict) -> str:
    c, r, h, p = f["coverage"], f["reask"], f["reask"]["half"], f["replay"]
    by = "、".join(f"{m} {n} 条" for m, n in r["by_model"].items())
    lo, hi = r["decisive_minus_placebo_ci_pp"]
    rows = _budget_rows(f)
    return "\n".join([
        START,
        "",
        "## 1.3.1 发现",
        "",
        '![错误诊断的证据覆盖与重新提问后的改对率](docs/figures/readme_findings_131.png)',
        "",
        f"**错误诊断大多缺证据。** 在列有决定性检查结果的题目上，错误诊断（不含中止）中 {c['wrong_incomplete']} 条未拿全"
        f"决定性结果，{c['wrong_complete']} 条拿全了：{c['wrong_incomplete_pct']}% 缺少部分决定性证据。",
        "",
        f"**补上漏掉的结果，多数能改对。** 从覆盖不全的错误诊断中抽取 {r['n']} 条（{by}），从最终作答之前重新提问。"
        f"补上漏掉的决定性结果后 {r['decisive_pct']}% 改对；补上同样数量的非决定性结果为 {r['placebo_pct']}%，"
        f"什么都不补为 {r['resample_pct']}%（比非决定性结果高 {r['decisive_minus_placebo_pp']} 个百分点，"
        f"95% 区间 {lo} 至 {hi}）。在漏掉至少两项决定性结果的 {h['n']} 条上，随机补一半改对 {h['half_pct']}%"
        f"（不补 {h['resample_pct']}%，全补 {h['all_pct']}%）。",
        "",
        f"**加大检查预算，各模型获益不同。** {f['budget']['items']} 道诊断题（首次作答）在一半、默认（即榜单）和两倍检查预算下的"
        "诊断正确率。增益：从一半到两倍预算的百分点变化及其配对 95% 区间；预算使用：平均花掉默认预算的比例。",
        "",
        "| 模型 | 0.5× | 1× | 2× | 增益 [95% 区间] | 1× 预算使用 |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"**HTTP 重放复现已存工具结果。** 把榜单运行中 {p['questions']} 道题上 {p['sessions']} 次作答会话的 {p['calls']} 次工具调用"
        f"经 `haenv serve fhir` 重放并解码，{p['calls']} 次中 {p['identical']} 次与当前函数调用输出一致，"
        f"并复现这些会话轨迹中存储的全部工具结果（含购买与拒绝）；"
        f"返回的 {p['resources']} 个资源都是合法的 FHIR {p['fhir']}（不合法 {p['invalid']} 个）。重放工具是 "
        "[`tools/release/replay_http.py`](tools/release/replay_http.py)，在榜单池上运行：",
        "",
        "```bash",
        "python tools/release/replay_http.py --work leaderboard-pool/ --workers 32 --out artifacts/replay-http-results.json",
        "```",
        "",
        f"预期汇总：`questions={p['questions'].replace(',', '')}`、`responses={p['sessions'].replace(',', '')}`、"
        f"`calls={p['calls'].replace(',', '')}`、`identical={p['identical'].replace(',', '')}`、"
        f"`differing={p['differing']}`、`stored_differing={p['stored_differing']}`、"
        f"`pending_purchases_replayed={p['pending_purchases']}`、`validation_failures={p['invalid']}`；"
        "`missing=[]`、`worker_errors=[]`。榜单池含已存运行，需另行提供。源码树内的离线 HTTP 演练为 "
        "`python tools/release/replay_http.py --offline`。",
        "",
        "离线重算并重评分公开的 re-ask 答案：`python tools/evaluation/reask_pooled.py --check`"
        "（[数据与条件](benchmarks/results/1.3.1/reask/README.md)）。",
        END,
    ])



def render_board(f: dict, zh: bool) -> str:
    def number(v):
        return "0" + v if v.startswith(".") else v
    rows = []
    for r in f["board"]:
        lo, hi = r["rank_ci"]
        rank = lo if lo == hi else f"{lo}–{hi}"
        cells = [rank, r["model"], f"{number(r['score'])} ({number(r['ci'][0])}–{number(r['ci'][1])})"]
        cells.extend(number(r["packs"][p]) for p in ("m2", "p2", "p3", "p4"))
        cells.append(number(r["all3"]))
        rows.append("| " + " | ".join(cells) + " |")
    alt = "13 模型总分、95% 区间与名次区间" if zh else "Overall scores, 95% intervals and rank intervals for 13 models"
    header = "| 名次 | 模型 | 总分（95% CI） | ① | ② | ③ | ④ | All-3 |" if zh else "| Rank | Model | Overall (95% CI) | ① | ② | ③ | ④ | All-3 |"
    return "\n".join([BOARD_START, f"![{alt}](docs/figures/readme_board_131.png)", "", header,
                       "|---:|---|---:|---:|---:|---:|---:|---:|", *rows, BOARD_END])


def apply_board(text: str, block: str) -> str:
    if BOARD_START in text:
        start, end = text.index(BOARD_START), text.index(BOARD_END) + len(BOARD_END)
    else:
        heading = "### 1.3.1 "
        start = text.index("| ", text.index(heading))
        end = text.index("\n\n### 1.3.0", start)
    return text[:start] + block + text[end:]


def draw_findings(f: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=180, facecolor="#fcfcfb")
    c, r = f["coverage"], f["reask"]
    values = [[float(c["wrong_incomplete_pct"])],
              [float(r[k]) for k in ("decisive_pct", "placebo_pct", "resample_pct")]]
    labels = [["Missing decisive evidence"], ["Missed decisive results", "Non-decisive results", "Nothing added"]]
    titles = ["Evidence missing from wrong diagnoses", "Re-ask correction rate"]
    for ax, data, names, title in zip(axes, values, labels, titles):
        ax.set_facecolor("#fcfcfb")
        bars = ax.barh(range(len(data)), data, height=0.6, color=["#2a78d6", "#88b3e6", "#c3c2b7"][:len(data)])
        ax.set_yticks(range(len(data)), names, fontsize=9)
        ax.set_ylim(len(data) - 0.5, -0.5) if len(data) > 1 else ax.set_ylim(1.5, -1.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percent")
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=15)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in data], padding=5, fontsize=10)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.open(buf).convert("RGB").save(output, optimize=True)

DOCS = {"README.md": (render_en, "## Quick start"), "README.zh-CN.md": (render_zh, "## 快速开始")}


def apply(text: str, block: str, anchor: str) -> str:
    """Replace the marked block, or insert it before `anchor` the first time."""
    if START in text:
        a, b = text.index(START), text.index(END) + len(END)
        return text[:a] + block + text[b:]
    assert text.count(anchor) == 1, anchor
    return text.replace(anchor, block + "\n\n" + anchor, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if a README block differs from the JSON")
    a = ap.parse_args()
    f = json.loads(SOURCE.read_text(encoding="utf-8"))
    stale = []
    for doc, (render, anchor) in DOCS.items():
        p = ROOT / doc
        text = p.read_text(encoding="utf-8")
        new = apply_board(apply(text, render(f), anchor), render_board(f, doc.endswith("zh-CN.md")))
        if new != text:
            stale.append(doc)
            if not a.check:
                p.write_text(new, encoding="utf-8")
    if a.check:
        with tempfile.TemporaryDirectory(prefix="haenv-findings-") as tmp:
            generated = Path(tmp) / FIGURE.name
            draw_findings(f, generated)
            if not FIGURE.is_file() or generated.read_bytes() != FIGURE.read_bytes():
                stale.append(FIGURE.relative_to(ROOT).as_posix())
    else:
        draw_findings(f, FIGURE)
    if a.check and stale:
        raise SystemExit(f"stale findings block in {stale}; rerun this script")
    print("findings blocks " + ("match " if a.check else "written from ") + SOURCE.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
