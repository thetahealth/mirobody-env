"""Draws the README patient figure from one item of a built medication-adjustment pack (pack ③).

    uv run --with scikit-learn --with joblib --with matplotlib \
        python tools/presentation/make_readme_meds_patient.py --build          # build, then draw
    python tools/presentation/make_readme_meds_patient.py --batch <pack batch>  # draw from a build

`--build` runs `tools/generation/make_pack.py --pack p3 --n 50` with the public seed into
`.tmpwork/readme-meds/` (deterministic generator and the shared audit; no model calls). The figure
follows the layout of the earlier `readme_patient.png`: four lanes on one time axis up to the visit `T`
(the target lab, daily wearable and home-cuff streams, the drug's dose and pharmacy fills, symptoms and
other-drug changes), with the values the gold is derived from but the agent never sees drawn on top: the
true lab values, the adherence the world set, and the cause of each symptom. Output:
`docs/figures/readme_meds_patient.png`, the first image of the README. Figure text is ASCII only.

SYNTHETIC, evaluation use only, not medical advice.
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
import json
import pathlib
import re
import subprocess
import sys

import matplotlib
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import matplotlib.text as mtext                                     # noqa: E402

from haenv.world.clinical.med_course import DISPLAY                  # noqa: E402
from haenv.world.labworld.meds import coverage_segments               # noqa: E402

ROOT = _repository_root(__file__)
WORK = ROOT / ".tmpwork" / "readme-meds"
OUT = ROOT / "docs" / "figures" / "readme_meds_patient.png"
#: The public-seed item drawn: gliclazide, HbA1c above target, adherence fallen since the last fill.
CASE = "PT-bc3d6da50981d91d1330"

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, ORANGE, GREEN, RED, PURPLE = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#7c3aed"
HIDDEN = "#f1f3f5"
ZH = {zh: en for en, (zh, _u, _f) in DISPLAY.items()}
CAUSE = {"unrelated": ("unrelated", MUTED), "other_drug": ("caused by another drug", ORANGE),
         "intercurrent": ("intercurrent illness", PURPLE)}
DRUGS = yaml.safe_load((ROOT / "registry/p3_meds.yaml").read_text(encoding="utf-8"))["drugs"]
CONDITION = {"T2D": "type 2 diabetes", "hypertension": "hypertension",
             "dyslipidemia": "dyslipidemia", "hypothyroidism": "hypothyroidism"}
CLASS = {"check_adherence_or_adverse_effect": "check adherence or adverse effect",
         "uptitrate": "uptitrate", "downtitrate": "downtitrate", "maintain": "maintain",
         "switch": "switch"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.unicode_minus": False, "legend.frameon": False,
        "legend.fontsize": 8, "figure.dpi": 150,
    })


def _build() -> pathlib.Path:
    out = WORK / "p3"
    p = subprocess.run([sys.executable, str(ROOT / "tools/generation/make_pack.py"), "--pack", "p3",
                        "--n", "50", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"make_pack failed ({p.returncode}):\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    return pathlib.Path(rec["pack_batch"])


def _drug(zh_text: str) -> str | None:
    hits = [en for zh, en in ZH.items() if zh in zh_text]
    return max(hits, key=lambda en: len(DISPLAY[en][0])) if hits else None


def _doses(record: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """Each drug's prescribed daily dose over time, from the prescription record's dose strings."""
    out: dict[str, list[tuple[int, float]]] = {}
    for r in sorted(record, key=lambda r: r["day"]):
        en = _drug(r.get("drug", ""))
        m = re.search(r"(\d+(?:\.\d+)?)\s*mg", r.get("prescription", ""))
        if en and m:
            out.setdefault(en, []).append((r["day"], float(m.group(1))))
    return out


def _series(case: dict, name: str, T: int) -> tuple[list[int], list[float]]:
    pts = [p for p in case["longitudinal_data"].get(name) or [] if p["ts"] <= T]
    return [p["ts"] for p in pts], [p["value"] for p in pts]


def draw(batch: pathlib.Path, cid: str = CASE) -> pathlib.Path:
    rows = {json.loads(l)["case_id"]: json.loads(l)
            for l in (batch / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    if cid not in rows:
        raise SystemExit(f"{cid} is not in {batch}; pass --case with an item of this build")
    case = rows[cid]["case"]
    rec, gold = case["prediction_context"]["meds_p3"], case["adjudication"]["meds_p3"]
    T, axis, drug = int(gold["T"]), gold["axis"], gold["drug"]
    x0, x1 = min(d for d, _ in gold["dose_line"]) - 15, T + 45

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 6.9), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.3, 1.1, 0.95], "hspace": 0.35})
    for ax in axes:
        ax.axvspan(T, x1, color=HIDDEN, zorder=0, lw=0)
        ax.axvline(T, color=INK, lw=1.0, zorder=3)

    # (1) the target lab: what the record prints, and the true value the world holds (hidden)
    ax = axes[0]
    labs = [(r["day"], float(re.search(r"[\d.]+", r["result"]).group()))
            for r in rec["lab_results"] if axis in r["test"] and x0 <= r["day"] <= T]
    truth = [(v["day"], v["truth"]) for v in gold["visits"] if any(v["day"] == d for d, _ in labs)]
    ax.axhspan(0, gold["upper"], color="#ecfdf5", lw=0, zorder=0)
    ax.axhline(gold["upper"], color=GREEN, lw=1.0, ls=(0, (4, 2)))
    ax.plot([d for d, _ in truth], [v for _, v in truth], "o", mfc="none", mec=MUTED, ms=8,
            mew=1.2, zorder=4, label="true value (hidden)")
    ax.plot([d for d, _ in labs], [v for _, v in labs], "s", color=BLUE, ms=6, zorder=5,
            label=f"{axis} as printed in the record")
    for d, v in labs:
        ax.annotate(f"{v:.1f}", (d, v), textcoords="offset points", xytext=(7, 3), fontsize=8,
                    color=BLUE)
    ys = [v for _, v in labs] + [v for _, v in truth] + [gold["upper"]]
    ax.set_ylim(min(ys) - 0.45, max(ys) + 0.55)
    ax.text(x0 + 6, gold["upper"] - 0.06, f"clinic target < {gold['upper']:g} %", color=GREEN,
            fontsize=8, va="top")
    ax.set_ylabel(f"{axis} (%)")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.42, 1.0))
    top = ax.get_ylim()[1]
    ax.text(T - 4, top, "in the agent's record", ha="right", va="top", color=INK, fontsize=8)
    ax.text(T + 4, top, "after T", ha="left", va="top", color=MUTED, fontsize=8)

    # (2) wearable and home cuff
    ax = axes[1]
    t, v = _series(case, "resting_hr", T)
    ax.plot(t, v, "-", lw=0.7, color=RED)
    ax.set_ylabel("resting HR (bpm)", color=RED)
    ax.set_title("wearable and home cuff (daily HR, weekly BP)", loc="left", fontsize=8,
                 fontweight="normal", color=MUTED)
    ax2 = ax.twinx()
    t, v = _series(case, "systolic_bp", T)
    ax2.plot(t, v, "o", ms=3, color=PURPLE)
    ax2.set_ylabel("SBP (mmHg)", color=PURPLE)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)

    # (3) the drug: prescribed dose, pharmacy fills, and the adherence the world set (hidden)
    ax = axes[2]
    doses = [(max(d, x0), v) for d, v in _doses(rec["prescription_record"]).get(drug, []) if d <= T]
    hi = max(v for _, v in doses)
    ax.step([d for d, _ in doses] + [T], [v for _, v in doses] + [doses[-1][1]], where="post",
            color=INK, lw=1.4)
    ax.set_ylabel(f"{drug} (mg/d)")
    ax.set_ylim(0, hi * 1.45)
    fills = [d for d in gold["refills"] if x0 <= d <= T]
    ax.plot(fills, [hi * 1.3] * len(fills), "v", color=BLUE, ms=6)
    ax.text(fills[0] - 6, hi * 1.3, f"{gold['supply']}-day fills", ha="right", va="center",
            fontsize=7.5, color=BLUE)
    last = max(d for d in gold["refills"] if d <= T)
    taken = gold["supply"] - gold["pill_count"]
    ax2 = ax.twinx()
    seg = coverage_segments(gold["refills"], gold["a_before"], gold["a_last"], gold["supply"])
    xs = [max(d, x0) for d, _ in seg] + [T]
    ys = [100 * a for _, a in seg] + [100 * seg[-1][1]]
    ax2.step(xs, ys, where="post", color=ORANGE, lw=1.6)
    ax2.set_ylim(0, 125)
    ax2.set_ylabel("adherence %", color=ORANGE)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.text(T - 3, 100 * gold["a_last"] - 9, f"pill count at T: {taken} of {T - last} doses "
             "taken since the last fill", ha="right", va="top", fontsize=7.5, color=ORANGE)
    top_day = next(d for d, v in doses if v == hi)
    ax.set_title(f"{drug}: dose, pharmacy fills and adherence (hidden); top dose since day "
                 f"{top_day}", loc="left", fontsize=8, fontweight="normal", color=MUTED)

    # (4) events, coloured by the cause the world assigned (hidden from the agent)
    ax = axes[3]
    causes = {d["symptom"]: d for d in gold["diary"]}
    reported = [(s["onset_day"], s["symptom"]) for s in rec["symptom_diary"]]
    reported += [(e["day"], e["symptom"]) for e in case["prediction_context"]["event_history_record"]]
    lanes: dict[tuple[str, str], list[int]] = {}
    for day, sym in sorted(set(reported)):
        d = causes.get(sym, {"cause": "unrelated"})
        lab, col = CAUSE.get(d["cause"], (d["cause"], MUTED))
        if d.get("source_drug"):
            lab = f"side effect ({d['source_drug']})"
        lanes.setdefault((lab, col), []).append(day)
    other = sorted({r["day"] for r in rec["prescription_record"]
                    if _drug(r.get("drug", "")) in gold["other_drugs"] and x0 <= r["day"] <= T})
    lanes[("other-drug change", INK)] = other
    names = sorted(lanes, key=lambda k: (k[1] == INK, k[1] == MUTED))
    for i, key in enumerate(names):
        for day in lanes[key]:
            ax.plot(day, len(names) - 1 - i, "|", ms=11, mew=2.2, color=key[1])
    ax.set_yticks([len(names) - 1 - i for i in range(len(names))], [n for n, _ in names])
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_title("symptoms and other-drug changes; colour = the cause the world assigned (hidden)",
                 loc="left", fontsize=8, fontweight="normal", color=MUTED)
    ax.set_xlabel("day of the record (visit T; earlier history is negative)")
    ax.set_xlim(x0, x1)

    cond = CONDITION.get(DRUGS[drug]["known_condition"], DRUGS[drug]["known_condition"])
    fig.suptitle(f"One synthetic patient (medication adjustment): {cond} on {drug}, visit "
                 f"T = day {T}", x=0.01, ha="left", fontweight="bold", fontsize=11)
    acc = " or ".join(CLASS.get(c, c) for c in gold["accept"])
    fig.text(0.01, 0.93, f"Gold, fixed before the record is rendered:  {acc}  |  {axis} above "
             f"target at the top dose  |  {100 * gold['a_last']:.0f}% adherence since the last fill",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.canvas.draw()
    texts = [t.get_text() for t in fig.findobj(mtext.Text) if t.get_text()]
    bad = [s for s in texts if any(ord(c) > 127 for c in s)]
    if bad:
        raise SystemExit(f"non-ASCII figure text: {bad[:3]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] wrote {OUT.relative_to(ROOT)}")
    return OUT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="build the public-seed pack, then draw")
    g.add_argument("--batch", type=pathlib.Path, help="a built pack batch holding cases.jsonl")
    ap.add_argument("--case", default=CASE)
    a = ap.parse_args()
    _style()
    draw(_build() if a.build else a.batch, a.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
