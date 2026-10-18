"""Draw the README overall board from the released board data.

The figure shows each model's overall score (equal-weight mean of the four pack scores, k3) with
its 95% paired item-bootstrap interval and its rank interval. Values come from
`web/demo/board.json:overall.board` without re-aggregation; labels round the stored decimal half
up, the rule the README tables and the demo use.

    uv run --with matplotlib python tools/presentation/make_readme_board.py
    uv run --with matplotlib python tools/presentation/make_readme_board.py --check

No model calls or scoring recomputation.
"""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)

import argparse
from decimal import ROUND_HALF_UP, Decimal
import io
import json
from pathlib import Path
import tempfile

ROOT = _repository_root(__file__)
SOURCE = ROOT / "web/demo/board.json"
OUTPUT = ROOT / "docs/figures"
NAME = "readme_board.png"

INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#7a7873"
GRID, AXIS, BG = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DOT, BAR = "#2a78d6", "#88b3e6"
X0, X1 = 0.3, 0.9


def half_up3(x: float) -> str:
    """Three decimals, half up on the stored decimal."""
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def prepare(board: dict) -> list[dict]:
    """Validate the final overall board and return its rows in rank order."""
    overall = board["overall"]
    if overall["status"] != "final" or any(p["status"] != "final" for p in board["packs"].values()):
        raise SystemExit("board.json is not final; the README board shows final readings only")
    rows = sorted(overall["board"], key=lambda r: (r["rank"], -r["score"]))
    for r in rows:
        lo, hi = r["ci95"]
        if not (X0 <= lo <= r["score"] <= hi <= X1):
            raise SystemExit(f"{r['model']}: score or interval outside the drawn axis {X0}-{X1}")
    return rows


def rank_label(r: dict) -> str:
    lo, hi = r["rank_ci95"]
    return str(lo) if lo == hi else f"{lo}–{hi}"


def draw(board: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = prepare(board)
    packs = board["packs"]
    n_items = {p["n_cases"] for p in packs.values()}
    n_rounds = {p["n_rounds"] for p in packs.values()}
    if len(n_items) != 1 or len(n_rounds) != 1:
        raise SystemExit("packs differ in items or rounds; the subtitle states one N and one k")
    n_items, n_rounds = n_items.pop(), n_rounds.pop()

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "haenv"})
    fig = plt.figure(figsize=(8.6, 4.98), dpi=200, facecolor=BG)
    fig.text(.035, .93, "Overall board", fontsize=14.5, fontweight="bold", color=INK, va="center")
    fig.text(.035, .875, f"{len(rows)} models · packs ① · ② · ③ · ④ "
             f"· N = {n_items} each · k{n_rounds}: mean S over {n_rounds} answers per item",
             fontsize=9.6, color=MUTED, va="center")
    fig.text(.035, .84, "S = G·D·(1+Q)/2 · dot: score · bar: 95% interval "
             "(paired item bootstrap)", fontsize=9.6, color=MUTED, va="center")

    ax = fig.add_axes((.30, .205, .62, .565), facecolor=BG)
    n = len(rows)
    ax.set_xlim(X0, X1)
    ax.set_ylim(n - .5, -.5)
    for x in [X0 + .1 * i for i in range(1, round((X1 - X0) / .1) + 1)]:
        ax.axvline(x, color=GRID, lw=1.1, zorder=0)
    for y, r in enumerate(rows):
        lo, hi = r["ci95"]
        ax.plot([lo, hi], [y, y], color=BAR, lw=3.2, solid_capstyle="round", zorder=2)
        ax.plot([r["score"]], [y], "o", ms=7.5, color=DOT, mec=BG, mew=1.2, zorder=3)
        ax.text(hi + .006, y, half_up3(r["score"]), fontsize=9.6, color=MUTED, va="center", ha="left")
    ax.set_yticks([])
    ax.set_xticks([X0 + .1 * i for i in range(round((X1 - X0) / .1) + 1)])
    ax.set_xticklabels([f"{t:.1f}" for t in ax.get_xticks()], fontsize=9.6, color=FAINT)
    ax.tick_params(axis="x", length=0, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.set_xlabel("overall score (equal-weight mean of the pack scores)", fontsize=9.6, color=FAINT,
                  labelpad=8)

    # Row labels and separators span the full width, left of the axis included.
    to_fig = lambda y: ax.transData.transform((X0, y))[1] / fig.bbox.height
    for y, r in enumerate(rows):
        fy = to_fig(y)
        fig.text(.033, fy, rank_label(r), fontsize=9.6, color=FAINT, va="center")
        fig.text(.095, fy, r["model"], fontsize=10, color=INK, va="center",
                 fontweight="bold" if r["rank_ci95"] == [1, 1] else "normal")
        sep = to_fig(y + .5)
        fig.add_artist(matplotlib.lines.Line2D([.033, .92], [sep, sep], color=GRID, lw=1.1))
    fig.text(.035, .055, "Left: rank interval, the 2.5th to 97.5th percentile of the model's rank over the "
             "same bootstrap draws. Overlapping intervals: not separable.", fontsize=8.4, color=FAINT,
             va="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    buf.seek(0)
    output.mkdir(parents=True, exist_ok=True)
    Image.open(buf).convert("RGB").save(output / NAME, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the checked-in figure differs")
    args = parser.parse_args()
    board = json.loads(SOURCE.read_text(encoding="utf-8"))
    if args.check:
        with tempfile.TemporaryDirectory(prefix="haenv-readme-board-") as tmp:
            draw(board, Path(tmp))
            if not (OUTPUT / NAME).is_file() or (OUTPUT / NAME).read_bytes() != (Path(tmp) / NAME).read_bytes():
                raise SystemExit(f"Stale figure output: {NAME}; rerun this script")
        print("README board PNG matches web/demo/board.json")
    else:
        draw(board, OUTPUT)
        print(f"Wrote {OUTPUT / NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
