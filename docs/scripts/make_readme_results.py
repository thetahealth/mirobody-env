"""Draw explicitly preliminary README scores from the public model snapshot.

    uv run --with matplotlib python docs/scripts/make_readme_results.py
    uv run --with matplotlib python docs/scripts/make_readme_results.py --check

No model calls or scoring recomputation. The bundled JSON is sufficient to rebuild
the PNG, vector SVG and machine-readable CSV in a public checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "web/demo/data.json"
OUTPUT = ROOT / "docs/figures"
STEM = "readme_results"

# Names describe the existing production aggregates; no scores are derived here.
LABELS = {
    "tests_recall+tests_precision": "Test selection\nF1",
    "disc_recall": "Discriminating tests\nRecall",
    "review_macro": "Clinician review\nBalanced accuracy",
    "quant_ok": "Numerical reading\nCorrectness rate",
}
PRELIMINARY_LABELS = {
    "dx_hit": "Diagnosis\nHit rate",
    "dx_listed": "Diagnosis\nListed rate",
    "tests_recall+tests_precision": "Test selection\nF1",
    "disc_recall": "Discriminating\ntests",
    "noop_ok": "Data status\nConsistency",
    "review_macro": "Clinician review\nBalance",
    "quant_ok": "Numerical\nreading",
    "tool_target_grounded_rate": "Tool targets\nGrounded",
}
TRACK_LABELS = {"answers": "Diagnosis", "trace": "Budgeted tools"}
PUBLIC_BOARDS = {"answers": "board", "trace": "tool_board"}


def pending_dim(dim: str, pending: set[str]) -> bool:
    """A paired dimension is pending when any of its production metrics is."""
    return bool(set(dim.split("+")) & pending)


def prepare(data: dict) -> dict:
    """Validate the public snapshot and retain its values without re-aggregation."""
    board, provenance = data["board"], data["provenance"]
    snapshot, release = provenance["snapshot"]["answers"], provenance["release"]
    if provenance["is_snapshot"] or snapshot["current_world"] is not True:
        raise ValueError("Result figure requires the current release world")
    if snapshot["world_sha"] != release["world_sha"]:
        raise ValueError("Result world differs from the release world")
    if not board["judging"] == snapshot["judging_sha16"] == release["judging_sha16"]:
        raise ValueError("Result judging fingerprints differ")
    dims = board["dims"]
    if not dims or len(dims) != len(set(dims)):
        raise ValueError("Result dimensions must be nonempty and unique")
    blocked = {item["metric"] for item in board["withheld"]}
    for dim in dims:
        if set(dim.split("+")) & blocked:
            raise ValueError(f"Withheld dimension cannot be plotted: {dim}")
        if dim not in LABELS:
            raise ValueError(f"Dimension needs a reviewed display label: {dim}")
    rows = sorted(board["models"], key=lambda row: row["solver"])
    solvers = [row["solver"] for row in rows]
    if not rows or len(solvers) != len(set(solvers)):
        raise ValueError("Model rows must be nonempty and unique")
    if len(rows) != provenance["n_real_models"]:
        raise ValueError("Model count differs from the public provenance")
    values = []
    for row in rows:
        if set(row) != {"solver", "dims"} or set(row["dims"]) != set(dims):
            raise ValueError("Expected only public per-dimension readings, not composite scores")
        line = []
        for dim in dims:
            value = row["dims"][dim]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= 1):
                raise ValueError(f"Missing or invalid reading: {row['solver']} / {dim}")
            line.append(value)
        values.append(line)
    clinical = board["clinical_review"]
    cases = clinical["answers"]
    if not 0 <= cases["n_pending"] <= cases["n_cases"] or cases["n_cases"] <= 0:
        raise ValueError("Invalid case scope")
    return {
        "dims": dims, "solvers": solvers, "values": values,
        "labels": [data["solver_labels"][solver]["model"] for solver in solvers],
        "snapshot": snapshot, "profile": board["profile"], "release": release,
        "cases": cases, "withheld": sorted(blocked),
    }


def ranked_items(result: dict, dimension: str) -> list[dict]:
    """Describe one dimension's observed order; exact exported ties share a rank."""
    if dimension not in result["dims"]:
        raise ValueError(f"Not a public dimension: {dimension}")
    column = result["dims"].index(dimension)
    rows = sorted(({"solver": solver, "label": label, "value": values[column]}
                   for solver, label, values in
                   zip(result["solvers"], result["labels"], result["values"])),
                  key=lambda row: (-row["value"], row["solver"]))
    counts = Counter(row["value"] for row in rows)
    previous, rank = None, 0
    for index, row in enumerate(rows):
        if row["value"] != previous:
            rank = index + 1
        row.update(rank=rank, tied=counts[row["value"]] > 1)
        previous = row["value"]
    return rows


def _unit_value(value: object, label: str) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 1):
        raise ValueError(f"Missing or invalid preliminary reading: {label}")


def prepare_preliminary(data: dict) -> dict:
    """Require the explicit provisional contract; never recompute production scores."""
    # The separate publication-eligible subset must still satisfy its old contract.
    prepare(data)
    preliminary, provenance = data["preliminary"], data["provenance"]
    if preliminary["status"] != "preliminary" or preliminary["not_final"] is not True:
        raise ValueError("Preliminary results must be explicitly marked not final")
    if set(preliminary["tracks"]) != set(TRACK_LABELS):
        raise ValueError("Expected separate diagnosis and tool tracks")
    tracks, pending_sets = {}, []
    for key in TRACK_LABELS:
        track = preliminary["tracks"][key]
        if track["status"] != "preliminary" or track["not_final"] is not True:
            raise ValueError(f"Track must be explicitly preliminary: {key}")
        snapshot = provenance["snapshot"][key]
        if (snapshot["current_world"] is not True
                or snapshot["world_sha"] != provenance["release"]["world_sha"]
                or track["judging"] != snapshot["judging_sha16"]
                or track["judging"] != provenance["release"]["judging_sha16"]):
            raise ValueError(f"Preliminary track has stale fingerprints: {key}")
        context, dims = track["context"], track["dims"]
        if context["pack"] != snapshot["pack"] or context["date"] != snapshot["date"]:
            raise ValueError(f"Preliminary provenance differs: {key}")
        if (type(context["n_cases"]) is not int or context["n_cases"] <= 0
                or not 0 <= context["n_pending"] <= context["n_cases"]):
            raise ValueError(f"Invalid preliminary case scope: {key}")
        if not dims or len(dims) != len(set(dims)) or set(dims) - PRELIMINARY_LABELS.keys():
            raise ValueError(f"Unknown or duplicate preliminary dimensions: {key}")
        # The pending set is read from the data (one scoring profile for both tracks);
        # it must be disclosed and name only production metrics.
        pending = {item["metric"] for item in track["pending_metrics"]}
        if not pending or not all(type(m) is str and m for m in pending):
            raise ValueError(f"Pending validity disclosure differs: {key}")
        pending_sets.append(pending)
        rows = sorted(track["models"], key=lambda row: row["solver"])
        solvers = [row["solver"] for row in rows]
        public = {row["solver"] for row in data[PUBLIC_BOARDS[key]]["models"]}
        if (not rows or len(solvers) != len(set(solvers))
                or len(solvers) != provenance["n_ranked_models"][key]
                or set(solvers) != public):
            raise ValueError(f"Invalid preliminary model roster: {key}")
        completion = track["completion"]
        if track["not_ranked"] or set(completion) != set(solvers):
            raise ValueError(f"Every declared model must be on the board: {key}")
        for solver, e in completion.items():
            if (type(e["answered"]) is not int or type(e["unanswered"]) is not int
                    or e["answered"] + e["unanswered"] != e["of"] or e["of"] != context["n_cases"]
                    or sum(e["reasons"].values()) != e["unanswered"]):
                raise ValueError(f"Invalid completion disclosure: {key} / {solver}")
        for row in rows:
            solver = row["solver"]
            if set(row["dims"]) != set(dims):
                raise ValueError(f"Missing preliminary dimensions: {key} / {solver}")
            _unit_value(row["score"], f"{key} / {solver} / composite")
            _unit_value(row["gate_multiplier"], f"{key} / {solver} / gate multiplier")
            if (type(row["n_cases"]) is not int
                    or not 0 < row["n_cases"] <= context["n_cases"]):
                raise ValueError(f"Invalid model case count: {key} / {solver}")
            for dim in dims:
                _unit_value(row["dims"][dim], f"{key} / {solver} / {dim}")
                n = track["samples"][solver][dim]
                if type(n) is not int or not 0 < n <= context["n_cases"]:
                    raise ValueError(f"Missing or invalid sample count: {key} / {solver} / {dim}")
        tracks[key] = {**track, "models": rows, "labels": {
            solver: data["solver_labels"][solver]["model"] for solver in solvers}}
    if any(p != pending_sets[0] for p in pending_sets):
        raise ValueError("Pending validity disclosure differs between tracks")
    if data["rank_rule"]["rule"].get("name") != "zero":
        raise ValueError("The result figure uses the zero rule (unanswered scores 0)")
    clinical = data["board"]["clinical_review"]
    if (type(clinical["conditions_pending"]) is not int or clinical["conditions_pending"] < 0
            or any(clinical[key] != {"n_pending": tracks[key]["context"]["n_pending"],
                                     "n_cases": tracks[key]["context"]["n_cases"]}
                   for key in TRACK_LABELS)):
        raise ValueError("Clinical review counts differ from the track contexts")
    return {"tracks": tracks, "release": provenance["release"],
            "pending": sorted(pending_sets[0]), "clinical": clinical,
            "rank_rule": data["rank_rule"]["rule"]}


def preliminary_ranked_items(track: dict) -> list[dict]:
    """Exact exported composite ties share competition ranks within one track."""
    rows = sorted((dict(row) for row in track["models"]),
                  key=lambda row: (-row["score"], row["solver"]))
    counts = Counter(row["score"] for row in rows)
    previous, rank = None, 0
    for index, row in enumerate(rows):
        if row["score"] != previous:
            rank = index + 1
        row.update(rank=rank, tied=counts[row["score"]] > 1)
        previous = row["score"]
    return rows


def csv_text(result: dict) -> str:
    """Long-form preliminary values retain full exported precision and sample counts."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["status", "not_final", "track", "model", "rank", "score",
                     "gate_multiplier", "n_cases", "dimension", "value", "n",
                     "validity_pending"])
    for key, track in result["tracks"].items():
        ranks = {row["solver"]: row["rank"] for row in preliminary_ranked_items(track)}
        pending = {item["metric"] for item in track["pending_metrics"]}
        for row in track["models"]:  # ranked models only; the figure footer names the others
            for dim in track["dims"]:
                writer.writerow(["preliminary", "true", key, row["solver"],
                                 ranks[row["solver"]], row["score"], row["gate_multiplier"],
                                 row["n_cases"], dim, row["dims"][dim],
                                 track["samples"][row["solver"]][dim],
                                 "true" if pending_dim(dim, pending) else "false"])
    return stream.getvalue()


def draw(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "svg.fonttype": "none", "svg.hashsalt": STEM,
                         "axes.unicode_minus": False})
    output.mkdir(parents=True, exist_ok=True)
    ink, muted = "#17212B", "#53616F"
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(18, 19), facecolor="white")
    fig.text(.04, .958, "PRELIMINARY - NOT FINAL", fontsize=20,
             weight="bold", color="#946122")
    fig.text(.04, .923, "Model scores and observed rankings", fontsize=30,
             weight="bold", color=ink)
    fig.text(.04, .895, "Existing production scores, including pending-validity metrics. The two task tracks are scored separately.",
             fontsize=15, color=muted)
    cmap = LinearSegmentedColormap.from_list("haenv_preliminary", ["#F2F6F6", "#A5D0CF", "#176C75"])
    for index, (key, track) in enumerate(result["tracks"].items()):
        top = .79 - index * .36
        height = .24
        context = track["context"]
        rows = preliminary_ranked_items(track)
        fig.text(.04, top + .068,
                 f"{TRACK_LABELS[key]}  /  {max(r['n_cases'] for r in track['models'])} of "
                 f"{context['n_cases']} synthetic cases ranked", fontsize=22,
                 weight="bold", color=ink)
        fig.text(.04, top + .044,
                 f"{context['geometry']}  |  {len(rows)} models  |  "
                 f"{context['n_pending']} cases use conditions awaiting clinical review",
                 fontsize=14, color=muted)
        ax = fig.add_axes([.04, top - height, .385, height])
        heat = fig.add_axes([.465, top - height, .495, height])
        ax.set_xlim(0, 1)
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_yticks([])
        ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1.00"])
        ax.tick_params(axis="x", length=0, pad=8, labelsize=12, colors=muted)
        for spine in ax.spines.values():
            spine.set_visible(False)
        heat.set_xlim(-.5, len(track["dims"]) - .5)
        heat.set_ylim(len(rows) - .5, -.5)
        heat.axis("off")
        pending = set(result["pending"])
        for col, dim in enumerate(track["dims"]):
            label = PRELIMINARY_LABELS[dim].replace("\n", "*\n", 1) if pending_dim(dim, pending) \
                else PRELIMINARY_LABELS[dim]
            heat.text(col, -.68, label, ha="center", va="bottom",
                      fontsize=12, color="#946122" if pending_dim(dim, pending) else muted)
        for position, row in enumerate(rows):
            rank = ("=" if row["tied"] else "") + str(row["rank"])
            weight = "bold" if row["rank"] == 1 else "normal"
            ax.text(0, position - .12, rank, fontsize=13, color=muted,
                    va="center", fontfamily="DejaVu Sans Mono")
            ax.text(.085, position - .12, track["labels"][row["solver"]], fontsize=15,
                    color=ink, weight=weight, va="center")
            ax.text(1, position - .12, f"{row['score']:.4f}", fontsize=15,
                    color=ink, weight=weight, va="center", ha="right",
                    fontfamily="DejaVu Sans Mono")
            ax.barh(position + .27, 1, height=.13, color="#E9EFF1", zorder=1)
            ax.barh(position + .27, row["score"], height=.13,
                    color="#247D87", zorder=2)
            for col, dim in enumerate(track["dims"]):
                value = row["dims"][dim]
                heat.add_patch(Rectangle((col - .48, position - .46), .96, .92,
                                         facecolor=cmap(value), edgecolor="white"))
                foreground = "white" if value >= .72 else ink
                heat.text(col, position - .20, f"{value:.3f}", ha="center", va="center",
                          fontsize=15, color=foreground, fontfamily="DejaVu Sans Mono")
                heat.text(col, position + .25, f"n={track['samples'][row['solver']][dim]}",
                          ha="center", va="center", fontsize=13, color=foreground)
        ax.set_xlabel("Composite score  /  fixed 0-1 scale", fontsize=13, color=muted, labelpad=8)
        heat.text(.5, -.072, "Dimensions: 0-1, higher is better  |  n = applicable cases",
                  transform=heat.transAxes, ha="center", fontsize=13, color=muted)
    clinical = result["clinical"]
    unanswered = {}
    for key, track in result["tracks"].items():
        for solver, e in track["completion"].items():
            if e["unanswered"]:
                unanswered.setdefault(track["labels"][solver], []).append(
                    f"{TRACK_LABELS[key].lower()} {e['unanswered']}/{e['of']}")
    notes = [
        "NOT FINAL. One pass per model per case (k=1); rerun noise is read on a separate repeat subset. Scores and ranks may change.",
        "* Includes metrics without a passing blind-human validity anchor: "
        + ", ".join(m for m in result["pending"]
                    if any(m in dim.split("+") for t in result["tracks"].values() for dim in t["dims"]))
        + "; included provisionally.",
        f"{clinical['conditions_pending']} conditions await clinical review; cases using them: "
        f"{clinical['answers']['n_pending']}/{clinical['answers']['n_cases']} diagnosis, "
        f"{clinical['trace']['n_pending']}/{clinical['trace']['n_cases']} tools. "
        "Exact ties share a rank (=); small gaps do not establish superiority.",
        ("Unanswered cells score 0 (no hard gate tripped): "
         + "; ".join(f"{name} ({', '.join(v)})" for name, v in sorted(unanswered.items())) + "."
         if unanswered else
         "Every model is scored on every case of each track; unanswered cells would score 0 "
         "(none in this snapshot)."),
        "Totals use the unchanged production formula, including its non-compensatory hard-gate multiplier.",
        f"Judging {result['release']['judging_sha16']}  |  "
        f"World {result['release']['world_sha']}  |  Source: web/demo/data.json",
    ]
    import textwrap
    lines = [part for note in notes for part in textwrap.wrap(note, 168, subsequent_indent="    ")]
    for i, line in enumerate(lines):
        fig.text(.04, .127 - i * .018, line, fontsize=14, color=muted)
    fig.savefig(output / f"{STEM}.png", dpi=200, facecolor="white")
    fig.savefig(output / f"{STEM}.svg", metadata={"Date": None}, facecolor="white")
    svg = output / f"{STEM}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                   + "\n", encoding="utf-8")
    plt.close(fig)
    (output / f"{STEM}.csv").write_text(csv_text(result), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if checked-in outputs differ")
    args = parser.parse_args()
    result = prepare_preliminary(json.loads(SOURCE.read_text(encoding="utf-8")))
    if args.check:
        with tempfile.TemporaryDirectory(prefix="haenv-readme-results-") as tmp:
            draw(result, Path(tmp))
            for suffix in ("png", "svg", "csv"):
                name = f"{STEM}.{suffix}"
                if not (OUTPUT / name).is_file() or (OUTPUT / name).read_bytes() != (Path(tmp) / name).read_bytes():
                    raise SystemExit(f"Stale figure output: {name}; rerun this script")
        print("README result PNG, SVG and CSV match the public snapshot")
    else:
        draw(result, OUTPUT)
        print(f"Wrote {len(result['tracks'])} preliminary track rankings to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
