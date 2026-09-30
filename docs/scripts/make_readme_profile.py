#!/usr/bin/env python
"""The README capability profile: first-tier models per dimension, with significance groups.

    # draw from the bundled readings (public checkout; no model calls)
    uv run --with matplotlib python docs/scripts/make_readme_profile.py
    # measure the readings (needs the answering batches and their semantic views)
    uv run python docs/scripts/make_readme_profile.py --measure \\
        --track ddx-timeline=<batch dir>::<semantic view> --track ddx-workup=<batch dir>::<semantic view>

`--measure` loads each track the way the board does (semantic view, unanswered cells as zero rows,
common complete cases), scores every case resample with the production `analytics.rank_ddx`, and
runs the board's paired case bootstrap and Holm rule (`semantic_report.bootstrap_scores`,
`_significant`, `BOARD_RULE`) once per dimension, over all pairs of all models as on the board. All models are drawn; the first tier (composite tier 1) is marked.
Letters: models that share a letter on a row are not separable on that row.

Output `docs/figures/readme_profile.json` is generated; do not edit it by hand.
SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/figures/readme_profile.json"
OUTPUT = ROOT / "docs/figures"
STEM = "readme_profile"
TRACK_LABELS = {"ddx-timeline": "Diagnosis over time (ddx-timeline)",
                "ddx-workup": "Budgeted test ordering (ddx-workup)"}
DIMS = {"ddx-timeline": ["score", "review_macro", "tests_recall_f1", "dx_listed", "quant_ok", "noop_ok"],
        "ddx-workup": ["score", "review_macro", "tool_target_grounded_rate", "tests_recall_f1",
                       "dx_listed", "quant_ok", "noop_ok"]}
DIM_LABELS = {"score": "Composite", "review_macro": "Clinician review\n(no needless referral)",
              "tool_target_grounded_rate": "Tool targets\ngrounded", "tests_recall_f1": "Test selection F1",
              "dx_listed": "Diagnosis listed", "quant_ok": "Numerical reading", "noop_ok": "Data status"}


def letters(models: list[str], point: dict, separable: set) -> dict:
    """Compact letter display: every maximal set of mutually non-separable models gets a letter."""
    order = sorted(models, key=lambda m: -point[m])
    tied = lambda a, b: (a, b) not in separable and (b, a) not in separable
    cliques = [set(c) for n in range(len(order), 0, -1) for c in itertools.combinations(order, n)
               if all(tied(a, b) for a, b in itertools.combinations(c, 2))]
    maximal = [c for c in cliques if not any(c < d for d in cliques)]
    maximal.sort(key=lambda c: min(order.index(m) for m in c))
    out = {m: "" for m in models}
    for i, c in enumerate(maximal):
        for m in c:
            out[m] += "abcdefghijklmnopqrstuvwxyz"[i]
    return out


def measure(specs: list[str]) -> dict:
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    import logging
    logging.disable(logging.ERROR)
    import rank_rule as rr
    import semantic_final_readout as sfr
    from haenv.analytics import rank_ddx
    from haenv.semantic_report import BOARD_RULE, _significant, bootstrap_scores, common_complete_cases
    out = {"rule": {"boot": BOARD_RULE["boot"], "seed": BOARD_RULE["seed"], "alpha": BOARD_RULE["alpha"],
                    "tier": "composite tier 1", "letters": "shared letter = not separable on that row"},
           "tracks": {}}
    for spec in specs:
        name, rest = spec.split("=", 1)
        batch, view = (Path(x) for x in rest.split("::", 1))
        rows = sfr.load_track(batch, view)
        meta = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
        cases = sorted({(json.loads(x).get("case") or json.loads(x))["case_id"]
                        for x in (batch / "cases.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()})
        rows, _ = rr.fill_unanswered(rows, cases, sorted(meta["models"]), None)
        keep, _ = common_complete_cases(rows)
        rows = [r for r in rows if r["case"] in set(keep)]
        dims = DIMS[name]

        def score_fn(sub):
            return {e["model"]: {d: e.get(d) for d in dims} for e in rank_ddx(sub)}
        point = score_fn(rows)
        samples = bootstrap_scores(rows, sorted(keep), score_fn=score_fn, boot=BOARD_RULE["boot"],
                                   seed=BOARD_RULE["seed"], workers=48)
        proj = lambda d, k: {m: v[k] for m, v in d.items() if v.get(k) is not None}
        comp = _significant(proj(point, "score"), [proj(s, "score") for s in samples], BOARD_RULE["alpha"])
        tier1 = sorted((m for m in point if not any(w == m for _, w in comp)), key=lambda m: -point[m]["score"])
        order = sorted(point, key=lambda m: -point[m]["score"])
        track = {"n_cases": len(keep), "models": order, "first_tier": tier1, "dims": {}}
        for d in dims:
            # Holm over all pairs of all models, as on the board.
            p = proj(point, d)
            sig = set(_significant(p, [proj(x, d) for x in samples], BOARD_RULE["alpha"]))
            track["dims"][d] = {
                "value": {m: round(p[m], 3) for m in order if m in p},
                "separable": sorted([a, b] for a, b in sig),
                "separable_first_tier": sum(a in tier1 and b in tier1 for a, b in sig),
                "letters": letters([m for m in order if m in p], p, sig)}
        out["tracks"][name] = track
    return out


def draw(data: dict, output: Path) -> None:
    """One figure per track: all models as columns (composite order), one row per dimension."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "svg.fonttype": "none",
                         "svg.hashsalt": STEM})
    ink, muted, accent = "#17212B", "#53616F", "#8A4B08"
    cmap = LinearSegmentedColormap.from_list("p", ["#F3F6F7", "#9FCFCB", "#1B6F78"])
    rule = data["rule"]
    for name, t in data["tracks"].items():
        models, tier1 = t["models"], t["first_tier"]
        dims = [d for d in DIMS[name] if d in t["dims"]]
        n_pairs = len(models) * (len(models) - 1) // 2
        n_tier = len(tier1) * (len(tier1) - 1) // 2
        cw, ch = 1.0, 0.62
        fig = plt.figure(figsize=(18, 1.9 + 0.78 * len(dims)), facecolor="white")
        ax = fig.add_axes([.15, .24, .70, .58])
        for i, d in enumerate(dims):
            row = t["dims"][d]
            for j, m in enumerate(models):
                v = row["value"].get(m)
                ax.add_patch(plt.Rectangle((j * cw, i * ch), cw * .96, ch * .9,
                                           color=cmap(v if v is not None else 0), ec="white"))
                if v is None:
                    continue
                dark = v > .62
                ax.text(j * cw + .48, i * ch + .24, f"{v:.3f}", ha="center", va="center", fontsize=12.5,
                        family="monospace", color="white" if dark else ink)
                ax.text(j * cw + .48, i * ch + .47, row["letters"].get(m, ""), ha="center", va="center",
                        fontsize=11.5, weight="bold", color="white" if dark else accent)
            ax.text(len(models) * cw + .15, i * ch + .3,
                    f"{len(row['separable'])}/{n_pairs} pairs\nfirst tier {row['separable_first_tier']}/{n_tier}",
                    va="center", fontsize=11, color=ink if row["separable_first_tier"] else muted)
        k = len(tier1)
        ax.plot([0, k * cw - .04], [-.12, -.12], color=accent, lw=2.5, clip_on=False)
        ax.text((k * cw) / 2, -.2, f"first tier ({k} models)", ha="center", va="bottom", fontsize=12,
                color=accent, weight="bold")
        ax.set_xlim(0, len(models) * cw); ax.set_ylim(len(dims) * ch, 0)
        ax.set_xticks([j * cw + .48 for j in range(len(models))])
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=12)
        ax.set_yticks([i * ch + .28 for i in range(len(dims))])
        ax.set_yticklabels([DIM_LABELS[d] for d in dims], fontsize=12.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(.02, .95, f"{TRACK_LABELS[name]}: {len(models)} models, {t['n_cases']} cases",
                 fontsize=18, weight="bold", color=ink, va="top")
        fig.text(.02, .055,
                 f"Columns in composite order. Models sharing a letter on a row are not separable "
                 f"(paired case bootstrap, {rule['boot']:,} resamples,\nHolm α = {rule['alpha']} over all "
                 f"{n_pairs} pairs). Preliminary: see the data card. Source: docs/figures/readme_profile.json",
                 fontsize=11, color=muted, va="center", linespacing=1.5)
        stem = f"{STEM}_{name.split('-')[1]}"
        fig.savefig(output / f"{stem}.png", dpi=200, facecolor="white")
        fig.savefig(output / f"{stem}.svg", metadata={"Date": None}, facecolor="white")
        svg = output / f"{stem}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                       + "\n", encoding="utf-8")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true", help="recompute the readings from the batches")
    ap.add_argument("--track", action="append", default=[], metavar="NAME=BATCH::VIEW")
    a = ap.parse_args()
    if a.measure:
        if not a.track:
            ap.error("--measure needs --track NAME=BATCH::VIEW for each track")
        DATA.write_text(json.dumps(measure(a.track), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {DATA}")
    draw(json.loads(DATA.read_text(encoding="utf-8")), OUTPUT)
    print(f"wrote {OUTPUT / STEM}_<track>.png / .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
