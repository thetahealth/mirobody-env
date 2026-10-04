"""Draw explicitly preliminary README scores from the public model snapshot.

Composite scores appear only on a track that production's board rule ranks on its common
complete cases (`semantic_report.track_board`, rule R0, exported by the demo as
`semantic_judge.<track>.board_rule`). There the models are grouped into tiers, a tier being 1 +
the number of models significantly higher (paired case bootstrap, Holm correction), with each
model's 95% interval drawn as a line; models in one tier are not ordered. Every track shows its
per-dimension readings. When no scored dimension of the diagnosis track has a passing validity
reading, the public diagnosis board has no columns; the figure then states that those dimensions
await validation by clinical blind annotation.

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
    "review_macro": "Clinician review\nSpecificity",
    "quant_ok": "Numerical reading\nCorrectness rate",
}
PRELIMINARY_LABELS = {
    "dx_hit": "Diagnosis\nHit rate",
    "dx_listed": "Diagnosis\nListed rate",
    "tests_recall+tests_precision": "Test selection\nF1",
    "disc_recall": "Discriminating\ntests",
    "noop_ok": "Data status\nConsistency",
    "review_macro": "Clinician review\nBalance",
    "review_utility_cc": "Clinician review\nUtility",
    "quant_ok": "Numerical\nreading",
    "tool_target_grounded_rate": "Tool targets\nGrounded",
}
TRACK_LABELS = {"answers": "Diagnosis", "trace": "Budgeted tools"}
PUBLIC_BOARDS = {"answers": "board", "trace": "tool_board"}
#: Shown on the diagnosis track when its public board has no publishable column.
DIAGNOSIS_VALIDITY_PENDING = ("Diagnosis-track scored dimensions: validity pending",
                              "(awaiting clinical blind annotation); no public column")


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
    if not isinstance(dims, list) or len(dims) != len(set(dims)):
        raise ValueError("Result dimensions must be a list of unique names")
    blocked = {item["metric"] for item in board["withheld"]}
    if not dims and not blocked:
        # An empty public board is legitimate only when its dimensions are withheld for validity.
        raise ValueError("An empty public board must disclose its withheld dimensions")
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
        "cases": cases, "withheld": sorted(blocked), "validity_pending": not dims,
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
    public_board = prepare(data)
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
        tracks[key] = {**track, "models": rows, "board": _board(data, key, solvers), "labels": {
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
            "rank_rule": data["rank_rule"]["rule"],
            "diagnosis_validity_pending": public_board["validity_pending"]}


def _board(data: dict, key: str, solvers: list[str]) -> dict:
    """Production's board rule for one track, as exported; nothing here is recomputed."""
    rule = ((data.get("semantic_judge") or {}).get(key) or {}).get("board_rule")
    if not isinstance(rule, dict):
        raise ValueError(f"Track needs production's exported board rule: {key}")
    for field in ("rule", "ranked", "why", "min_common_cases", "alpha", "boot", "n_common_complete"):
        if field not in rule:
            raise ValueError(f"Board rule lacks {field}: {key}")
    total = rule["ranked"] is True and rule["rule"] == "R0"
    entries = {}
    if total:
        models = rule.get("models")
        if not isinstance(models, dict) or set(models) != set(solvers):
            raise ValueError(f"Board rule models differ from the roster: {key}")
        for solver in solvers:
            m = models[solver]
            score, ci, tier = m.get("score"), m.get("ci95"), m.get("tier")
            _unit_value(score, f"{key} / {solver} / board composite")
            if (not isinstance(ci, list) or len(ci) != 2
                    or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in ci)
                    or not ci[0] <= score <= ci[1]):
                raise ValueError(f"Invalid 95% interval: {key} / {solver}")
            if type(tier) is not int or not 1 <= tier <= len(solvers):
                raise ValueError(f"Invalid tier: {key} / {solver}")
            entries[solver] = {"score": score, "ci95": ci, "tier": tier}
        if min(e["tier"] for e in entries.values()) != 1:
            raise ValueError(f"No model in the first tier: {key}")
    return {"total": total, "rule": rule["rule"], "ranked": rule["ranked"], "why": rule["why"],
            "min_common_cases": rule["min_common_cases"], "alpha": rule["alpha"],
            "boot": rule["boot"], "n_common_complete": rule["n_common_complete"], "entries": entries}


def tiered_items(track: dict) -> list[dict]:
    """Rows in display order: by tier, then model id (a tier is not ordered); by model id
    when the track shows no composite."""
    entries = track["board"]["entries"]
    rows = [dict(row, **entries.get(row["solver"], {})) for row in track["models"]]
    if track["board"]["total"]:
        return sorted(rows, key=lambda row: (row["tier"], row["solver"]))
    return sorted(rows, key=lambda row: row["solver"])


def csv_text(result: dict) -> str:
    """Long-form preliminary values retain full exported precision and sample counts."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["status", "not_final", "track", "model", "tier", "score", "ci95_low",
                     "ci95_high", "gate_multiplier", "n_cases", "dimension", "value", "n",
                     "validity_pending"])
    for key, track in result["tracks"].items():
        pending = {item["metric"] for item in track["pending_metrics"]}
        for row in tiered_items(track):
            total = track["board"]["total"]
            for dim in track["dims"]:
                writer.writerow(["preliminary", "true", track["context"]["pack"], row["solver"],
                                 row["tier"] if total else "", row["score"] if total else "",
                                 row["ci95"][0] if total else "", row["ci95"][1] if total else "",
                                 row["gate_multiplier"], row["n_cases"], dim, row["dims"][dim],
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
    fig.text(.04, .923, "Model tiers and per-dimension scores", fontsize=30,
             weight="bold", color=ink)
    fig.text(.04, .895, "Production scores, including pending-validity metrics. The two task tracks are scored separately.",
             fontsize=15, color=muted)
    cmap = LinearSegmentedColormap.from_list("haenv_preliminary", ["#F2F6F6", "#A5D0CF", "#176C75"])
    for index, (key, track) in enumerate(result["tracks"].items()):
        top = .79 - index * .36
        height = .24
        context = track["context"]
        rows = tiered_items(track)
        board = track["board"]
        fig.text(.04, top + .068,
                 f"{TRACK_LABELS[key]}  /  {max(r['n_cases'] for r in track['models'])} of "
                 f"{context['n_cases']} synthetic cases ranked", fontsize=22,
                 weight="bold", color=ink)
        fig.text(.04, top + .044,
                 f"{context['geometry']}  |  {len(rows)} models  |  "
                 f"{context['n_pending']} cases use conditions awaiting clinical review",
                 fontsize=14, color=muted)
        if key == "answers" and result.get("diagnosis_validity_pending"):
            for line_no, line in enumerate(DIAGNOSIS_VALIDITY_PENDING):
                fig.text(.04, top + .026 - line_no * .014, line, fontsize=13,
                         weight="bold", color="#946122")
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
        previous_tier = None
        for position, row in enumerate(rows):
            if board["total"]:
                if row["tier"] != previous_tier:
                    ax.text(0, position - .12, f"Tier {row['tier']}", fontsize=13, color=muted,
                            va="center", weight="bold")
                    if previous_tier is not None:
                        ax.axhline(position - .5, color="#C9D3D8", linewidth=1)
                previous_tier = row["tier"]
                ax.text(.16, position - .12, track["labels"][row["solver"]], fontsize=15,
                        color=ink, va="center")
                ax.text(1, position - .12, f"{row['score']:.3f}", fontsize=15, color=ink,
                        va="center", ha="right", fontfamily="DejaVu Sans Mono")
                ax.barh(position + .27, 1, height=.13, color="#E9EFF1", zorder=1)
                ax.hlines(position + .27, row["ci95"][0], row["ci95"][1], color="#247D87",
                          linewidth=3, zorder=2)
                ax.plot([row["score"]], [position + .27], "o", color="#17212B", markersize=6, zorder=3)
            else:
                ax.text(.085, position - .12, track["labels"][row["solver"]], fontsize=15,
                        color=ink, va="center")
            for col, dim in enumerate(track["dims"]):
                value = row["dims"][dim]
                heat.add_patch(Rectangle((col - .48, position - .46), .96, .92,
                                         facecolor=cmap(value), edgecolor="white"))
                foreground = "white" if value >= .72 else ink
                heat.text(col, position - .20, f"{value:.3f}", ha="center", va="center",
                          fontsize=15, color=foreground, fontfamily="DejaVu Sans Mono")
                heat.text(col, position + .25, f"n={track['samples'][row['solver']][dim]}",
                          ha="center", va="center", fontsize=13, color=foreground)
        if board["total"]:
            ax.set_xlabel("Composite score and 95% interval  /  fixed 0-1 scale", fontsize=13,
                          color=muted, labelpad=8)
        else:
            ax.set_xticks([])
            ax.text(.085, len(rows) - .1, f"No composite on this track ({board['rule']}): "
                    f"{board['n_common_complete']} cases complete for every model, "
                    f"{board['min_common_cases']} needed.", fontsize=12, color="#946122", va="top")
        heat.text(.5, -.072, "Dimensions: 0-1, higher is better  |  n = applicable cases",
                  transform=heat.transAxes, ha="center", fontsize=13, color=muted)
    first = next(t["board"] for t in result["tracks"].values())
    clinical = result["clinical"]
    unanswered = {}
    for key, track in result["tracks"].items():
        for solver, e in track["completion"].items():
            if e["unanswered"]:
                unanswered.setdefault(track["labels"][solver], []).append(
                    f"{TRACK_LABELS[key].lower()} {e['unanswered']}/{e['of']}")
    notes = [
        "NOT FINAL. One pass per model per case (k=1); rerun noise is read on a separate repeat subset. Scores and tiers may change.",
        f"Tier = 1 + models significantly higher (paired case bootstrap, {first['boot']} resamples, Holm, "
        f"alpha {first['alpha']}); a tier is not ordered. A composite is shown only when "
        f"{first['min_common_cases']} cases are complete for every model.",
        "* Includes metrics without a passing blind-human validity anchor: "
        + ", ".join(m for m in result["pending"]
                    if any(m in dim.split("+") for t in result["tracks"].values() for dim in t["dims"]))
        + "; included provisionally.",
        f"{clinical['conditions_pending']} conditions await clinical review; cases using them: "
        f"{clinical['answers']['n_pending']}/{clinical['answers']['n_cases']} diagnosis, "
        f"{clinical['trace']['n_pending']}/{clinical['trace']['n_cases']} tools.",
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
        fig.text(.04, .142 - i * .0155, line, fontsize=14, color=muted)
    # The README shows the PNG at about 900 px and links the SVG for full resolution, so the
    # raster is drawn at 100 dpi and stored with a 128-colour palette (under 300 KB).
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    buf.seek(0)
    (Image.open(buf).convert("RGB")
     .quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
     .save(output / f"{STEM}.png", optimize=True))
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
