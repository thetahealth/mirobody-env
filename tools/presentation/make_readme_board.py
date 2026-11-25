"""Draw the README overall board from the released board data.

The figure shows each model's overall score (equal-weight mean of the four pack scores, k3) with
its 95% paired item-bootstrap interval and its rank interval. Values come from
`web/demo/board-1.3.0.json:overall.board` without re-aggregation; labels round the stored decimal half
up, the rule the README tables and the demo use.

    uv run --with matplotlib python tools/presentation/make_readme_board.py
    uv run --with matplotlib python tools/presentation/make_readme_board.py --check

`--score` draws a `board_score.json` readout instead (the 1.3.1 board: `docs/boards/1.3.1/k3.json`
-> `docs/figures/readme_board_131.png`), copying its overall scores, intervals and rank intervals.
The PNG is trimmed to its drawn content plus a narrow even margin, so the README can show it at the
full column width.

`--board docs/boards/1.6.0/k1.json` draws the single-answer overall and per-pack figures.
Add `--readmes` to refresh the score tables and their figure links in both READMEs.
Use `--draft --out DIRECTORY` with an explicitly draft readout to preview both outside the repo.

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
SOURCE = ROOT / "web/demo/board-1.3.0.json"
OUTPUT = ROOT / "docs/figures"
NAME = "readme_board.png"

INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#7a7873"
GRID, AXIS, BG = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DOT, BAR = "#2a78d6", "#88b3e6"
X0, X1 = 0.3, 0.9
#: Background pixels kept around the drawn content when the PNG is trimmed.
CROP_PAD = 16


def half_up3(x: float) -> str:
    """Three decimals, half up on the stored decimal."""
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def from_score(readout: dict) -> dict:
    """The board.json fields this figure reads, copied from a `board_score.json` readout."""
    ranked = [m for m, r in readout["overall"].items() if r["score"] is not None]
    packs = {}
    for pack, prof in readout["profile"].items():
        n_items = {prof[m]["n_pk1"] for m in ranked}
        n_rounds = {prof[m]["k"] for m in ranked}
        if len(n_items) != 1 or len(n_rounds) != 1:
            raise SystemExit(f"{pack}: models differ in items or answers per item")
        packs[pack] = {"status": "final", "n_cases": n_items.pop(), "n_rounds": n_rounds.pop()}
    board = [{"model": m, "score": readout["overall"][m]["score"], "ci95": readout["overall"][m]["ci"],
              "rank": readout["overall"][m]["rank"],
              "rank_ci95": [int(x) for x in readout["overall"][m]["rank_ci"]]} for m in ranked]
    return {"overall": {"status": "final", "board": board}, "packs": packs}


def prepare(board: dict, x0: float = X0, x1: float = X1) -> list[dict]:
    """Validate the final overall board and return its rows in rank order."""
    overall = board["overall"]
    if overall["status"] != "final" or any(p["status"] != "final" for p in board["packs"].values()):
        raise SystemExit("board.json is not final; the README board shows final readings only")
    rows = sorted(overall["board"], key=lambda r: (r["rank"], -r["score"]))
    for r in rows:
        lo, hi = r["ci95"]
        if not (x0 <= lo <= r["score"] <= hi <= x1):
            raise SystemExit(f"{r['model']}: score or interval outside the drawn axis {x0}-{x1}")
    return rows


def crop_to_content(image, background: str = BG, pad: int = CROP_PAD):
    """Trim `image` to the pixels that differ from `background`, keeping `pad` pixels on each side."""
    from PIL import Image, ImageChops
    box = ImageChops.difference(image, Image.new(image.mode, image.size, background)).getbbox()
    if box is None:
        raise SystemExit("The figure is empty")
    left, top, right, bottom = box
    return image.crop((max(left - pad, 0), max(top - pad, 0),
                       min(right + pad, image.width), min(bottom + pad, image.height)))


def rank_label(r: dict) -> str:
    lo, hi = r["rank_ci95"]
    return str(lo) if lo == hi else f"{lo}–{hi}"


#: The eight-row layout in inches; more rows add ROW_IN each and keep the header and footer.
HEIGHT, ROWS, ROW_IN = 4.98, 8, .565 * 4.98 / 8


def draw(board: dict, output: Path, name: str = NAME, x0: float = X0, x1: float = X1) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = prepare(board, x0, x1)
    packs = board["packs"]
    n_items = {p["n_cases"] for p in packs.values()}
    n_rounds = {p["n_rounds"] for p in packs.values()}
    if len(n_items) != 1 or len(n_rounds) != 1:
        raise SystemExit("packs differ in items or rounds; the subtitle states one N and one k")
    n_items, n_rounds = n_items.pop(), n_rounds.pop()

    n = len(rows)
    height = HEIGHT if n == ROWS else HEIGHT + (n - ROWS) * ROW_IN
    # Header and footer keep their height in inches; the eight-row board keeps its exact layout.
    top = (lambda f: f) if n == ROWS else (lambda f: 1 - (1 - f) * HEIGHT / height)
    bottom = (lambda f: f) if n == ROWS else (lambda f: f * HEIGHT / height)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "haenv"})
    fig = plt.figure(figsize=(8.6, height), dpi=200, facecolor=BG)
    fig.text(.035, top(.93), "Overall board", fontsize=14.5, fontweight="bold", color=INK, va="center")
    fig.text(.035, top(.875), f"{len(rows)} models · packs ① · ② · ③ · ④ "
             f"· N = {n_items} each · k{n_rounds}: mean S over {n_rounds} answers per item",
             fontsize=9.6, color=MUTED, va="center")
    fig.text(.035, top(.84), "S = G·D·(1+Q)/2 · dot: score · bar: 95% interval "
             "(paired item bootstrap)", fontsize=9.6, color=MUTED, va="center")

    ax = fig.add_axes((.30, bottom(.205), .62, top(.77) - bottom(.205)) if n != ROWS
                      else (.30, .205, .62, .565), facecolor=BG)
    ax.set_xlim(x0, x1)
    ax.set_ylim(n - .5, -.5)
    for x in [x0 + .1 * i for i in range(1, round((x1 - x0) / .1) + 1)]:
        ax.axvline(x, color=GRID, lw=1.1, zorder=0)
    for y, r in enumerate(rows):
        lo, hi = r["ci95"]
        ax.plot([lo, hi], [y, y], color=BAR, lw=3.2, solid_capstyle="round", zorder=2)
        ax.plot([r["score"]], [y], "o", ms=7.5, color=DOT, mec=BG, mew=1.2, zorder=3)
        ax.text(hi + .006, y, half_up3(r["score"]), fontsize=9.6, color=MUTED, va="center", ha="left")
    ax.set_yticks([])
    ax.set_xticks([x0 + .1 * i for i in range(round((x1 - x0) / .1) + 1)])
    ax.set_xticklabels([f"{t:.1f}" for t in ax.get_xticks()], fontsize=9.6, color=FAINT)
    ax.tick_params(axis="x", length=0, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.set_xlabel("overall score (equal-weight mean of the pack scores)", fontsize=9.6, color=FAINT,
                  labelpad=8)

    # Row labels and separators span the full width, left of the axis included.
    to_fig = lambda y: ax.transData.transform((x0, y))[1] / fig.bbox.height
    for y, r in enumerate(rows):
        fy = to_fig(y)
        fig.text(.033, fy, rank_label(r), fontsize=9.6, color=FAINT, va="center")
        fig.text(.095, fy, r["model"], fontsize=10, color=INK, va="center",
                 fontweight="bold" if r["rank_ci95"] == [1, 1] else "normal")
        sep = to_fig(y + .5)
        fig.add_artist(matplotlib.lines.Line2D([.033, .92], [sep, sep], color=GRID, lw=1.1))
    fig.text(.035, bottom(.055), "Left: rank interval, the 2.5th to 97.5th percentile of the model's rank over the "
             "same bootstrap draws. Overlapping intervals: not separable.", fontsize=8.4, color=FAINT,
             va="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    buf.seek(0)
    output.mkdir(parents=True, exist_ok=True)
    crop_to_content(Image.open(buf).convert("RGB")).save(output / name, optimize=True)


K1_PACKS = ("m2", "p2", "p3", "p4", "rl")
K1_FIGURES = ("readme_board_160.png", "readme_task_scores_160.png")


def k1_rows(readout, *, draft=False):
    """Read the five-pack score format shared by the figures and the demo."""
    if readout["status"] != ("draft" if draft else "final"):
        raise ValueError("A completed readout is required; use --draft for draft data")
    if (readout["aggregation"] != "k1" or readout["language"] != "en"
            or tuple(readout["packs"]) != K1_PACKS or readout["items_per_pack"] != 50):
        raise ValueError("Expected the five English packs with one answer per item")
    rows = sorted(readout["overall"].items(), key=lambda entry: (entry[1]["rank"], -entry[1]["score"]))
    if len(rows) != readout["model_count"] or not rows:
        raise ValueError("Board row count does not match model_count")
    for model, row in rows:
        lower, upper = row["ci"]
        if not 0 <= lower <= row["score"] <= upper <= 1:
            raise ValueError(f"Invalid overall score or interval for {model}")
        if set(row["packs"]) != set(K1_PACKS) or not all(0 <= v <= 1 for v in row["packs"].values()):
            raise ValueError(f"Invalid pack scores for {model}")
    return rows


def k1_readme(readme: str, readout: dict, *, chinese: bool = False) -> str:
    rows = k1_rows(readout)
    alt = ("八个模型的总分和 95% 配对区间", "八个模型的分包准确率") if chinese else (
        "Overall scores and paired 95% intervals for eight models", "Pack accuracies for eight models")
    overall, per_pack = [
        f'<p align="center">\n  <img src="docs/figures/{figure}" alt="{label}" width="100%">\n</p>'
        for figure, label in zip(K1_FIGURES, alt)
    ]
    headers = ("名次", "模型", "总分（95% 区间）") if chinese else ("Rank", "Model", "Overall (95% CI)")
    table = ["| " + " | ".join((*headers, "①", "②", "③", "④", "⑤")) + " |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for model, row in rows:
        lo, hi = map(half_up3, row["ci"])
        values = [str(row["rank"]), model, f"{half_up3(row['score'])} ({lo}–{hi})"]
        values.extend(half_up3(row["packs"][pack]) for pack in K1_PACKS)
        table.append("| " + " | ".join(values) + " |")
    start = f"<!-- board-{readout['version']}:start -->\n"
    end = f"\n<!-- board-{readout['version']}:end -->"
    before, block = readme.split(start, 1)
    _, after = block.split(end, 1)
    return before + start + "\n\n".join((overall, "\n".join(table), per_pack)) + end + after


def draw_k1(readout, output, *, draft=False):
    import math

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from PIL import Image
    from tools.presentation.make_readme_task_scores import draw as draw_pack_scores

    rows = k1_rows(readout, draft=draft)
    left = max(0, math.floor(min(row["ci"][0] for _, row in rows) * 10 - .5) / 10)
    title = "Draft · Overall scores" if draft else "Overall scores"
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(8.6, 4.25), dpi=200, facecolor=BG)
        fig.subplots_adjust(left=.32, right=.90, top=.80, bottom=.13)
        fig.text(.035, .94, title, fontsize=14.5, weight="bold", color=INK, va="center")
        fig.text(.035, .88, "Equal-weight mean of five packs · 95% paired bootstrap intervals",
                 fontsize=9.6, color=MUTED, va="center")
        ax.set_facecolor(BG)
        ax.set(xlim=(left, 1), ylim=(len(rows) - .5, -.5), yticks=[])
        ticks = [i / 10 for i in range(round(left * 10), 11)]
        ax.set_xticks(ticks, [f"{tick:.1f}" for tick in ticks])
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.tick_params(axis="x", length=0, pad=7, labelsize=9.6, labelcolor=FAINT)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_axisbelow(True)
        for index, (model, row) in enumerate(rows):
            lower, upper = row["ci"]
            ax.plot([lower, upper], [index, index], color=BAR, lw=3.2, solid_capstyle="round")
            ax.plot(row["score"], index, "o", ms=7.5, color=DOT, mec=BG, mew=1.2)
            y = ax.transData.transform((left, index))[1] / fig.bbox.height
            sep = ax.transData.transform((left, index + .5))[1] / fig.bbox.height
            fig.text(.035, y, model, fontsize=10, color=INK, va="center")
            fig.text(.925, y, half_up3(row["score"]), fontsize=9.6, color=MUTED, va="center")
            fig.add_artist(matplotlib.lines.Line2D([.035, .965], [sep, sep], color=GRID, lw=1.1))
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=200, metadata={"Software": None})
            buffer.seek(0)
            crop_to_content(Image.open(buffer).convert("RGB")).save(output / K1_FIGURES[0], optimize=True)
        plt.close(fig)
    draw_pack_scores(readout, output, draft=draft)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the checked-in figure differs")
    parser.add_argument("--score", type=Path, help="draw this board_score.json readout instead of board.json")
    parser.add_argument("--name", help="output file name under docs/figures/ (required with --score)")
    parser.add_argument("--board", type=Path, help="draw both figures from a five-pack k1 readout")
    parser.add_argument("--out", type=Path, help="figure directory for --board")
    parser.add_argument("--draft", action="store_true", help="label both previews Draft")
    parser.add_argument("--readmes", action="store_true", help="update both READMEs from a final --board")
    args = parser.parse_args()
    if args.board:
        if args.score or args.name:
            parser.error("--board cannot be combined with --score or --name")
        output = (args.out or OUTPUT).resolve()
        if args.draft and (args.out is None or output == ROOT or ROOT in output.parents):
            parser.error("draft figures need --out outside the repository")
        if args.readmes and (args.draft or output != OUTPUT):
            parser.error("--readmes requires final figures in the repository figure directory")
        readout = json.loads(args.board.read_text(encoding="utf-8"))
        if args.check:
            with tempfile.TemporaryDirectory(prefix="haenv-board-") as directory:
                draw_k1(readout, Path(directory), draft=args.draft)
                for filename in K1_FIGURES:
                    if not (output / filename).is_file() or (output / filename).read_bytes() != (Path(directory) / filename).read_bytes():
                        raise SystemExit(f"Stale figure output: {filename}; rerun this script")
        else:
            draw_k1(readout, output, draft=args.draft)
        if args.readmes:
            for filename, chinese in (("README.md", False), ("README.zh-CN.md", True)):
                path = ROOT / filename
                current = path.read_text(encoding="utf-8")
                updated = k1_readme(current, readout, chinese=chinese)
                if args.check and updated != current:
                    raise SystemExit(f"Stale score table: {filename}; rerun this script")
                if not args.check:
                    path.write_text(updated, encoding="utf-8")
        return 0
    if args.out or args.draft or args.readmes:
        parser.error("--out, --draft and --readmes require --board")
    if args.score:
        if not args.name or args.name == NAME:
            raise SystemExit("--score needs its own --name")
        source, name, axis = args.score, args.name, (.1, .9)
        board = from_score(json.loads(source.read_text(encoding="utf-8")))
    else:
        source, name, axis = SOURCE, NAME, (X0, X1)
        board = json.loads(SOURCE.read_text(encoding="utf-8"))
    if args.check:
        with tempfile.TemporaryDirectory(prefix="haenv-readme-board-") as tmp:
            draw(board, Path(tmp), name, *axis)
            if not (OUTPUT / name).is_file() or (OUTPUT / name).read_bytes() != (Path(tmp) / name).read_bytes():
                raise SystemExit(f"Stale figure output: {name}; rerun this script")
        print(f"README board PNG {name} matches {source.name}")
    else:
        draw(board, OUTPUT, name, *axis)
        print(f"Wrote {OUTPUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
