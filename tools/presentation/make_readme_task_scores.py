"""Draw task scores and All-3 from the released 1.3.1 k3 readout; --check re-renders the PNG.

Every cell prints its value, so the figure carries no colour bar; the README legend defines All-3.
The PNG is trimmed like the overall board and spans the same width, so both show at one scale."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import bootstrap
ROOT = bootstrap(__file__)
from tools.presentation.make_readme_board import BG, INK, MUTED, FAINT, crop_to_content, half_up3

SOURCE = ROOT / "docs/boards/1.3.1/k3.json"
NAME = "readme_task_scores_131.png"
PACKS = ("m2", "p2", "p3", "p4")


def prepare(readout):
    if readout["aggregation"] != "k3":
        raise ValueError("The task figure requires the k3 board")
    models = sorted(readout["overall"], key=lambda m: readout["overall"][m]["rank"])
    rows = []
    for model in models:
        scores = [readout["overall"][model]["packs"][p] for p in PACKS]
        all3 = math.fsum(readout["profile"][p][model]["pass_k"] for p in PACKS) / len(PACKS)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (*scores, all3)):
            raise ValueError(f"Invalid score for {model}")
        rows.append((model, scores + [all3]))
    return rows


def luminance(rgb):
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb[:3]]
    return math.fsum(v * w for v, w in zip(linear, (.2126, .7152, .0722)))


def contrast_text(rgba):
    # This single relative-luminance threshold gives >=4.5:1 for both text colours.
    return "#ffffff" if luminance(rgba) < .182 else "#050505"


def draw(readout, output):
    import io
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    from PIL import Image

    rows = prepare(readout)
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "text.color": INK}):
        fig = Figure(figsize=(8.6, 6.35), dpi=200, facecolor=BG)
        FigureCanvasAgg(fig)
        fig.text(.035, .945, "Scores by task", fontsize=14.5, weight="bold", va="center")
        fig.text(.035, .900, f"1.3.1 board · {len(rows)} models in board order · "
                 "mean score over three answers per item", fontsize=9.6, color=MUTED, va="center")
        blue = LinearSegmentedColormap.from_list("readme_blue", ["#f0f5fb", "#96b8dc", "#3975b5", "#194c80"])
        grey = LinearSegmentedColormap.from_list("readme_grey", ["#f1f1ef", "#3f4246"])
        # Cells start right after the longest model name and end where the overall board's labels end.
        x = [.225, .369, .513, .657, .821]
        width = .139
        top, bottom = .775, .165
        step = (top - bottom) / len(rows)
        headers = ["Diagnosis", "Triage", "Medication\nadjustment", "Lab-change\nattribution", "All-3"]
        fig.text(.035, .82, "Model", fontsize=10, color=MUTED, va="center")
        for col, header in zip(x, headers):
            fig.text(col + width / 2, .82, header, fontsize=10, color=MUTED, ha="center", va="center")
        names = []
        for i, (model, values) in enumerate(rows):
            y = top - (i + .5) * step
            names.append(fig.text(.035, y, model, fontsize=10, va="center"))
            for j, value in enumerate(values):
                colour = (grey if j == 4 else blue)(value)
                fig.add_artist(Rectangle((x[j], y - step * .46), width, step * .92,
                                         transform=fig.transFigure, facecolor=colour, edgecolor="none"))
                fig.text(x[j] + width / 2, y, half_up3(value), fontsize=10.7,
                         ha="center", va="center", color=contrast_text(colour))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if max(name.get_window_extent(renderer).x1 for name in names) > (x[0] - .01) * fig.bbox.width:
            raise ValueError("A model name runs into the first column")
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
                raise ValueError(f"Clipped text: {text.get_text()}")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=200, metadata={"Software": None})
        buffer.seek(0)
        Path(output).mkdir(parents=True, exist_ok=True)
        crop_to_content(Image.open(buffer).convert("RGB")).save(Path(output) / NAME, optimize=True)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(SOURCE.read_text())
    output = ROOT / "docs/figures"
    if args.check:
        with tempfile.TemporaryDirectory() as directory:
            draw(data, Path(directory))
            if (Path(directory) / NAME).read_bytes() != (output / NAME).read_bytes():
                raise SystemExit(f"Stale figure: {NAME}")
        print(f"{NAME}: PNG matches the released k3 readout")
    else:
        draw(data, output)
        print(f"Wrote {NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
