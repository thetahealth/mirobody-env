"""Generates the three README figures (do not hand-edit the PNGs; re-run this script).

    uv run --with matplotlib python docs/scripts/make_readme_figures.py            # all three
    uv run --with matplotlib python docs/scripts/make_readme_figures.py --only 2   # one figure

Output: `docs/figures/readme_patient.png`, `readme_cohort.png`, `readme_difficulty.png`.
Everything is built with the deterministic generator (no model calls), in
`.tmpwork/readme-figures/`, from `inputs/early_warning-t2d-glp1.job.yaml`,
`inputs/early_warning-20.job.yaml` and jobs derived here from the T2D cases:

* cohort: `T2G-07` (maintainer, tirzepatide) and `T2G-08` (low responder, semaglutide),
  each repeated under 60 case ids, so only per-person draws differ;
* difficulty: `T2G-07` as declared, with measurement artifacts, and with artifacts plus a
  high distractor level.

These batches are for display only (the batch-level gate refuses them as question packs);
only emitted cases are drawn and no figure reports a score. Figure text is ASCII only
(`_assert_ascii`).

SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import re
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORK = ROOT / ".tmpwork" / "readme-figures"
OUT = ROOT / "docs" / "figures"

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402

from haenv import drug_effects as DE                                # noqa: E402
from haenv import wq                                                # noqa: E402
from haenv.store import AUDIT_FILENAME, load_audits, load_cases     # noqa: E402

T2D_JOB = ROOT / "inputs" / "early_warning-t2d-glp1.job.yaml"
EW20_JOB = ROOT / "inputs" / "early_warning-20.job.yaml"
N_CLONES = 60
#: Artifacts for the difficulty figure; `device_switch` and `mnar_missing` are left out because
#: the emission gate refuses them on this patient.
ARTIFACTS = [{"class": "transient_spike", "week": 5}, {"class": "transient_spike", "week": 10}]
#: The high distractor level adds about six symptom events before T; the declared symptom rate
#: is raised to match (otherwise `event_density_mismatch`).
HIGH_SYMPTOM_RATE = 0.5

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, ORANGE, GREEN, RED, PURPLE = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#7c3aed"
HIDDEN = "#f1f3f5"


# ------------------------------------------------------------------ builds
def _build(job: pathlib.Path) -> pathlib.Path:
    """`haenv build --gen deterministic --fresh` into WORK; returns the batch directory. Exit 4
    (batch-level gate refused) still persists the emitted cases and is accepted.
    """
    env = {**os.environ, "HAENV_OUTPUT_ROOT": str(WORK)}
    p = subprocess.run([sys.executable, "-m", "haenv", "build", str(job),
                        "--gen", "deterministic", "--fresh"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode not in (0, 4):
        raise SystemExit(f"build failed ({p.returncode}) for {job.name}:\n{p.stdout[-2000:]}")
    spec = yaml.safe_load(job.read_text(encoding="utf-8"))
    d = WORK / "results" / spec["task_type"] / spec["job_id"]
    batch = sorted(x for x in d.iterdir() if x.is_dir())[-1]
    n = sum(1 for _ in open(batch / "cases.jsonl", encoding="utf-8"))
    print(f"[fig] {spec['job_id']}: {n} emitted case(s)"
          + (" · batch-level gate refused the batch (cases kept)" if p.returncode == 4 else ""))
    return batch


def _derived_job(name: str, cases: list[dict]) -> pathlib.Path:
    base = yaml.safe_load(T2D_JOB.read_text(encoding="utf-8"))
    job = {k: v for k, v in base.items() if k != "cases"}
    job.update(job_id=f"readme-{name}", report=f"reports/readme-{name}.md", cases=cases)
    p = WORK / "jobs" / f"readme-{name}.job.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# haenv-canary-v1\n" + yaml.safe_dump(job, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return p


def _case(job: pathlib.Path, cid: str) -> dict:
    spec = yaml.safe_load(job.read_text(encoding="utf-8"))
    return copy.deepcopy(next(c for c in spec["cases"] if c["case_id"] == cid))


def _cases(batch: pathlib.Path) -> dict:
    return load_cases(batch / "cases.jsonl")


def _audits(batch: pathlib.Path) -> dict:
    return {a["case_id"]: a for a in load_audits(batch / AUDIT_FILENAME)}


#: Event kinds in the Q-side injection ledger (`wq.injected_manifest(...)["event_schedule"]`)
#: and how the figures name them. `real_symptom` is part of the clinical course; the others are
#: distractors placed around it.
EVENT_KINDS = {"real_symptom": ("clinical symptom", PURPLE),
               "lookalike": ("lookalike distractor", RED),
               "benign_symptom": ("benign symptom", ORANGE),
               "life_event": ("life event", MUTED),
               "distractor_injector": ("distractor-level symptom", "#0f766e")}


def _events(cid: str, audit: dict) -> list[tuple[float, str]]:
    """(day, kind) for every scheduled event of an emitted case. Distractor-level symptoms appear
    as `inherited_event` and are identified via the audit's `ev_id_map` (`-D<n>` ids).
    """
    from_injector = {new for old, new in (audit.get("ev_id_map") or {}).items()
                     if re.search(r"-D\d+$", old)}
    out = []
    for e in wq.injected_manifest(cid).get("event_schedule") or []:
        kind = e.get("kind")
        if kind == "inherited_event" and e.get("evidence_id") in from_injector:
            kind = "distractor_injector"
        if kind in EVENT_KINDS:
            out.append((float(e["day"]), kind))
    return out


# ------------------------------------------------------------------ drawing helpers
def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.labelsize": 9, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.unicode_minus": False,
        "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 150,
    })


def _series(raw, name: str) -> tuple[np.ndarray, np.ndarray]:
    pts = raw.longitudinal_data.get(name) or []
    return (np.array([p["ts"] for p in pts], float), np.array([p["value"] for p in pts], float))


def _hide_after_T(ax, T: float, end: float) -> None:
    ax.axvspan(T, end, color=HIDDEN, zorder=0, lw=0)
    ax.axvline(T, color=INK, lw=1.0, zorder=3)


def _assert_ascii(fig, name: str) -> None:
    import matplotlib.text as mtext
    fig.canvas.draw()
    texts = [t.get_text() for t in fig.findobj(mtext.Text) if t.get_text()]
    bad = [s for s in texts if any(ord(c) > 127 for c in s)]
    if not texts or bad:
        raise SystemExit(f"{name}: {len(texts)} text artists, non-ASCII: {bad[:3]}")


def _save(fig, name: str) -> pathlib.Path:
    _assert_ascii(fig, name)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] wrote {p.relative_to(ROOT)}")
    return p


# ------------------------------------------------------------------ figure 1
def fig_patient(batch: pathlib.Path, cid: str = "T2G-01") -> pathlib.Path:
    raw = _cases(batch)[cid]
    audit = _audits(batch)[cid]
    T = float(raw.prediction_context["prediction_time_T"])
    end = max(max(_series(raw, "weight")[0]), 180.0)
    rev = [rp["week"] * 7 for rp in (raw.reversal_points or [])]

    # Labs are drawn every 90 days and T is day 84, so the record up to T holds one lab draw;
    # the lab channel is shown in the cohort figure instead.
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 6.9), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.3, 1.0, 0.8],
                                          "hspace": 0.35})
    for ax in axes:
        _hide_after_T(ax, T, end)
        for d in rev:
            ax.axvline(d, color=RED, lw=0.9, ls=(0, (3, 2)), zorder=3)

    ax = axes[0]
    t, v = _series(raw, "weight")
    ax.plot(t, v, ".", ms=2.4, color=BLUE, label="home scale (daily)")
    t, v = _series(raw, "weight_ref")
    ax.plot(t, v, "s", ms=5, color=INK, label="clinic scale")
    ax.set_ylabel("weight (kg)")
    ax.legend(loc="upper right", ncol=2)
    ax.text(T - 2, ax.get_ylim()[1], "visible to the agent", ha="right", va="top", color=INK,
            fontsize=8)
    ax.text(T + 2, ax.get_ylim()[1], "after T: hidden, used for grading", ha="left", va="top",
            color=MUTED, fontsize=8)
    for d in rev:
        ax.text(d + 1.5, ax.get_ylim()[0], "reversal\n(latent)", color=RED, fontsize=7,
                va="bottom")

    ax = axes[1]
    t, v = _series(raw, "steps")
    ax.bar(t, v / 1000.0, width=1.0, color=GREEN, alpha=0.45, label="steps (k/day)")
    ax.set_ylabel("steps (k/day)", color=GREEN)
    ax.set_title("wearable (gaps are simulated non-wear)", loc="left", fontsize=8,
                 fontweight="normal", color=MUTED)
    ax2 = ax.twinx()
    t, v = _series(raw, "resting_hr")
    ax2.plot(t, v, "-", lw=0.8, color=RED, label="resting HR")
    ax2.set_ylabel("resting HR (bpm)", color=RED)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)

    ax = axes[2]
    t, v = _series(raw, "dose_timeline")
    ax.step(t, v, where="post", color=INK, label="tirzepatide dose (mg/wk)")
    ax.set_ylabel("dose (mg)")
    ax2 = ax.twinx()
    t, v = _series(raw, "medication_adherence")
    ax2.plot(t, v * 100, "o-", color=ORANGE, ms=3.5, label="adherence (%)")
    ax2.set_ylabel("adherence (%)", color=ORANGE)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)

    ax = axes[3]
    ev = _events(cid, audit)
    kinds = [k for k in EVENT_KINDS if any(kind == k for _, kind in ev)]
    for day, kind in ev:
        ax.plot(day, len(kinds) - 1 - kinds.index(kind), "|", ms=12, mew=2.2,
                color=EVENT_KINDS[kind][1])
    ax.set_yticks(range(len(kinds)), [EVENT_KINDS[k][0] for k in reversed(kinds)])
    ax.set_ylim(-0.7, len(kinds) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("day")
    ax.set_xlim(0, end)

    outcome = {"event_occurred": "weight regain", "event_not_occurred": "no regain"}
    gold = (f"Gold, fixed before the course is rendered:  outcome = "
            f"{outcome.get(raw.outcome_label, raw.outcome_label)}  |  driver = "
            f"{', '.join(d.replace('_', ' ') for d in raw.gold_drivers)}  |  reversal = week "
            f"{', '.join(str(r['week']) for r in raw.reversal_points or [])}")
    fig.suptitle(f"One synthetic patient ({cid}): type 2 diabetes on tirzepatide, index time "
                 f"T = day {int(T)}", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, 0.93, gold, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "readme_patient.png")


# ------------------------------------------------------------------ figure 2
def fig_cohort(cohort: pathlib.Path, ew20: pathlib.Path) -> pathlib.Path:
    cases, cases20 = _cases(cohort), _cases(ew20)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    groups = {"T2C-M": ("maintainer, tirzepatide", BLUE, "unknown_or_multifactorial"),
              "T2C-L": ("low responder, semaglutide", ORANGE, "biological_low_response")}
    ax = axes[0]
    resp = {}
    for pre, (lab, col, drv) in groups.items():
        vals = []
        for cid, raw in cases.items():
            if not cid.startswith(pre):
                continue
            r = DE.response_for(cid, drv, raw.latent_premise.get("patient_basics", {})
                                .get("regimen", {}).get("drug", ""))
            vals.append(r)
        resp[pre] = vals
        ax.hist(vals, bins=np.linspace(0, 1.9, 39), color=col, alpha=0.75, label=lab)
    (lo, hi), (blo, bhi) = DE.response_bounds()
    ax.axvspan(lo, hi, color=ORANGE, alpha=0.08, lw=0, label="band allowed: low response")
    ax.axvspan(blo, bhi, color=BLUE, alpha=0.06, lw=0, label="band allowed: responder")
    ax.set_xlabel("individual response, relative to the trial mean")
    ax.set_ylabel("people")
    ax.set_title("(a) two specifications, 60 people each", loc="left")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    # Correlation is reported per specification: pooling the two groups would mostly measure
    # the gap between their means.
    per_group = []
    for pre, (lab, col, _) in groups.items():
        xs, ys = [], []
        for cid, raw in cases.items():
            if not cid.startswith(pre):
                continue
            ta, va = _series(raw, "HbA1c")
            tf, vf = _series(raw, "fasting_glucose")
            if len(va) >= 2 and len(vf) >= 2:
                xs.append(va[1] - va[0])
                ys.append(vf[1] - vf[0])
        ax.scatter(xs, ys, s=14, color=col, alpha=0.8, label=lab)
        rr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
        per_group.append((lab.split(",")[0], rr, len(xs)))
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("change in HbA1c, day 0 to 90 (points)")
    ax.set_ylabel("change in fasting glucose (mmol/L)")
    ax.set_title("(b) HbA1c and fasting glucose, day 0 to 90", loc="left")
    ax.text(0.03, 0.95, "\n".join(f"{g}: r = {rr:.2f} (n = {n})" for g, rr, n in per_group),
            transform=ax.transAxes, va="top", fontsize=8)

    ax = axes[2]
    # Index times differ between cases (56 to 112 days), so each course is drawn relative to
    # its own T.
    col = {"event_occurred": RED, "event_not_occurred": BLUE}
    lab = {"event_occurred": "regains", "event_not_occurred": "maintains"}
    seen, lo, hi = set(), 0.0, 0.0
    for cid, raw in cases20.items():
        t, v = _series(raw, "weight")
        if not len(v):
            continue
        T = float(raw.prediction_context["prediction_time_T"])
        lo, hi = min(lo, float(t[0] - T)), max(hi, float(t[-1] - T))
        o = raw.outcome_label
        ax.plot(t - T, v - v[0], lw=0.8, alpha=0.8, color=col.get(o, MUTED),
                label=None if o in seen else lab.get(o, o))
        seen.add(o)
    _hide_after_T(ax, 0.0, hi)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("days relative to T")
    ax.set_ylabel("weight change from day 0 (kg)")
    ax.set_title(f"(c) {len(cases20)} patients, outcomes diverge after T", loc="left")
    ax.legend(loc="lower left")

    fig.tight_layout()
    print(f"[fig] cohort r(dHbA1c, dFBG) per group: "
          f"{', '.join(f'{g} {rr:.3f} (n={n})' for g, rr, n in per_group)}; "
          f"response medians {', '.join(f'{k} {np.median(v):.2f}' for k, v in resp.items())}")
    return _save(fig, "readme_cohort.png")


# ------------------------------------------------------------------ figure 3
def fig_difficulty(settings: list[tuple[str, pathlib.Path]], cid: str = "T2G-07") -> pathlib.Path:
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(len(settings), 1, figsize=(9.0, 6.4), sharex=True)
    ref, seen = None, set()
    for ax, (title, batch) in zip(axes, settings):
        raw = _cases(batch).get(cid)
        a = _audits(batch).get(cid, {})
        if raw is None:
            ax.text(0.5, 0.5, f"{title}: not emitted "
                    f"({', '.join(a.get('post_noise_conflicts') or [])})",
                    transform=ax.transAxes, ha="center")
            continue
        T = float(raw.prediction_context["prediction_time_T"])
        t, v = _series(raw, "weight")
        if ref is not None:
            ax.plot(*ref, ".", ms=2.4, color=GRID, zorder=1)
        ax.plot(t, v, ".", ms=2.6, color=BLUE, zorder=2)
        t2, v2 = _series(raw, "weight_ref")
        ax.plot(t2, v2, "s", ms=4, color=INK, zorder=2)
        _hide_after_T(ax, T, 180)
        y0 = ax.get_ylim()[0]
        ev = _events(cid, a)
        seen |= {k for _, k in ev}
        for day, kind in ev:
            ax.plot(day, y0, "|", ms=10, mew=2.0, color=EVENT_KINDS[kind][1], clip_on=False)
        # Artifacts are located by difference from the default setting (same patient, same
        # draws), so an injected spike that fell on a day without a reading is not labelled.
        base = dict(zip(*ref)) if ref is not None else {}
        for x, y in zip(t, v):
            if x in base and abs(y - base[x]) > 0.5:
                ax.annotate("artifact (trap in gold)", (x, y), xytext=(x - 30, y + 0.9),
                            fontsize=7.5, color=RED,
                            arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.8})
        n_real = sum(k == "real_symptom" for _, k in ev)
        ax.set_title(f"{title}: {n_real} clinical symptoms, {len(ev) - n_real} distractor "
                     f"events before T", loc="left")
        ax.set_ylabel("weight (kg)")
        if ref is None:
            ref = (t, v)
    handles = [Line2D([], [], marker="|", ls="", ms=9, mew=2, color=c, label=lab)
               for k, (lab, c) in EVENT_KINDS.items() if k in seen]
    handles.append(Line2D([], [], marker=".", ls="", color=GRID, label="default setting (reference)"))
    axes[0].legend(handles=handles, loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("day")
    axes[-1].set_xlim(0, 180)
    fig.suptitle("One patient at three difficulty settings: outcome and driver unchanged", x=0.01,
                 ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "readme_difficulty.png")


# ------------------------------------------------------------------ jobs written here
def cohort_job() -> pathlib.Path:
    cases = []
    for src, pre in (("T2G-07", "T2C-M"), ("T2G-08", "T2C-L")):
        base = _case(T2D_JOB, src)
        for i in range(1, N_CLONES + 1):
            c = copy.deepcopy(base)
            c["case_id"] = f"{pre}{i:03d}"
            cases.append(c)
    return _derived_job("cohort", cases)


def difficulty_jobs() -> list[tuple[str, pathlib.Path]]:
    """`T2G-07` at three settings, one job each, keeping the case id (per-person draws are keyed by
    case id).
    """
    base = _case(T2D_JOB, "T2G-07")
    settings = [
        ("default", {}),
        ("+ two transient spikes", {"noise": ARTIFACTS}),
        ("+ spikes and high distractor level",
         {"noise": ARTIFACTS, "distractor_level": "high",
          "event_density": {**base["latent"]["event_density"],
                            "symptom_rate": HIGH_SYMPTOM_RATE}}),
    ]
    out = []
    for i, (title, extra) in enumerate(settings):
        c = copy.deepcopy(base)
        c["latent"] = {**c["latent"], **extra}
        out.append((title, _derived_job(f"difficulty-{i}", [c])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", type=int, choices=(1, 2, 3))
    a = ap.parse_args()
    _style()
    if a.only in (None, 1):
        fig_patient(_build(T2D_JOB))
    if a.only in (None, 2):
        fig_cohort(_build(cohort_job()), _build(EW20_JOB))
    if a.only in (None, 3):
        fig_difficulty([(title, _build(job)) for title, job in difficulty_jobs()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
