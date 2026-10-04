"""report.py -- renders the evaluation report (markdown):
a multi-model ranking, 3-10 sampled cases in time order with each model's
answer and score, and per sampled case a collapsible block holding exactly the
solver-visible payload (checked with `leakage_probe` at render time).
"""
from __future__ import annotations

import math as _math

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from haenv_kernel.build import build_instance, leakage_probe    # kernel

log = logging.getLogger("haenv.report")

# Pure computations live in `haenv/analytics.py` and are re-exported here.
from .analytics import (  # noqa: F401
    NotPublishable,
    BLIND_BASELINES,
    CORE_DIM_TO_REC,
    CORRECTNESS_METRICS,
    HEURISTIC_BASELINES,
    JOIN_CLASSES,
    METRIC_KINDS,
    MIN_CELLS_FOR_EXERCISE,
    MIN_GRIDS_FOR_SCORE,
    MIN_ITEMS_FOR_EXERCISE,
    NA_TRACKS,
    NON_HARM_GATES,
    _CEILING_MAX_DIFF_ITEMS,
    _CORE_NAMES,
    _QNAMES,
    _REC_KEY,
    _apply_one_ruler,
    _ceiling_line,
    _core_from_profile,
    _fmt,
    _fsum_mean,
    _mean,
    _noise_alias,
    _render_stamp,
    _rowdim,
    _scope_const_floor,
    answer_space_diff,
    applicable_tracks,
    assert_publishable,
    batch_discrimination_gate,
    budget_ceiling_by_solver,
    constant_ceilings,
    cost_efficiency_analysis,
    degenerate_ceiling_gate,
    degenerate_ceilings,
    dimension_health,
    gate_family_of,
    board_rank_labels,
    floors_note,
    disputed_gold_hits,
    noop_reading,
    gate_cell,
    geometry_scope_note,
    gate_state_note,
    gate_multiplier,
    is_ddx_batch,
    item_discrimination,
    join_class_of,
    join_scored_classes,
    judging_provenance,
    kernel_gate_kinds,
    noise_floor_for,
    rank_ddx,
    rank_models,
    real_solver_pool,
    review_macro_of,
    scored_dims_still_healthy,
    task_kind_of,
    unexercised_dims,
)

# Optional errata file; the report header links to it only when the file is present.
RETRACTIONS_DOC = "docs/RETRACTIONS.md"

TRUTH_WORDS = ("outcome_label", "gold_driver", "adjudication", "latent_premise", "premise",
               "reversal_points", "distractor", "future_data", "verifier")


#: The kernel's four safety gates judged by action tier (same order as
#: `judges._ACTION_GATES`)
_ACTION_GATE_NAMES = ("premature_closure", "missing_clinician_review_flag",
                      "treatment_before_exclusion", "invasive_before_firstline")


def _action_gate_surface(rows: list[dict]) -> list[str]:
    """Hits of the four per-slice safety gates, printed with the number of judgeable
    and skipped slices (zero hits is only meaningful against its denominator).
    """
    _njudged = sum(int(r.get("slice_gate_n_judged") or 0) for r in rows)
    _unknown = sum(int(r.get("slice_gate_unknown") or 0) for r in rows)
    _any = any(r.get(f"sg_{g}") is not None or r.get("slice_gate_n_judged") is not None
               for g in _ACTION_GATE_NAMES for r in rows[:1] or [{}])
    if not _njudged and not _unknown:
        return []
    L = ["### 1z. Exercised surface of the four safety gates (per-slice · read together with the denominator)\n",
         f"Judgeable slices **{_njudged}** · skipped for missing `clinician_review_required` **{_unknown}** slices"
         f"(absence is not folded into False)\n",
         "| Gate | Hit cells | Hit slice-count | Hit by |", "|---|---|---|---|"]
    for g in _ACTION_GATE_NAMES:
        k = f"sg_{g}"
        hitrows = [r for r in rows if r.get(k)]
        tot = sum(int(r[k]) for r in hitrows)
        by: dict[str, int] = {}
        for r in hitrows:
            by[str(r.get("solver"))] = by.get(str(r.get("solver")), 0) + 1
        _who = " · ".join(f"`{m}`×{c}" for m, c in sorted(by.items(), key=lambda x: (-x[1], x[0]))[:5])
        L.append(f"| `{g}` | {len(hitrows)} | {tot} | {_who or '—'} |")
    L.append("")
    L.append("> **How to read this**: if the only hits come from offline stubs, this gate **has a positive control but no real model made this mistake** -- "
             "its correctness rests on the gate's positive and negative controls, not on the zero hit count.")
    L.append("")
    return L


def _case_gate_surface(rows: list[dict], built: dict | None = None) -> list[str]:
    """Hits of the case-level hard gates (`verifier.grade`'s `gates` list), one row
    per gate family including zero-hit families, each with its own denominator:
    gates with a precondition (e.g. `red_flag_present`) are counted against the
    cases that satisfy it; `not measured` when the case bodies are unavailable.
    """
    graded = [r for r in rows if "gates" in r]
    if not graded:
        return []
    ungraded = [r for r in rows if "gates" not in r
                and r.get("slice_gate_n_judged") is None]
    # Rows cover the kernel's full gate set plus observed families, merged to the
    # family (the part before `:`), so zero-hit gates still appear.
    _kern = kernel_gate_kinds()
    _obs_fams: dict[str, list[str]] = {}
    for r in graded:
        for g in (r.get("gates") or []):
            _obs_fams.setdefault(gate_family_of(str(g)), []).append(str(g))
    kinds = sorted(set(_obs_fams) | set(_kern or {}))
    # Precondition denominators are (cases satisfying it) x (solvers), read from the
    # case objects in `built`.
    _rf_cases: set[str] = set()
    for _cid, _raw in (built or {}).items():
        _adj = getattr(_raw, "adjudication", None)
        if _adj is None and isinstance(_raw, dict):
            _adj = _raw.get("adjudication")
        if (_adj or {}).get("red_flag_present"):
            _rf_cases.add(str(_cid))
    _n_solvers = len({str(r.get("solver")) for r in graded}) or 1
    # Kernel preconditions (`haenv_kernel/verifier.py`): `red_flag_present`, its complement,
    # `clinician_action_warranted`, `artifact_flags.is_artifact_window`.
    _warr_cases: set[str] = set()
    _artifact_cases: set[str] = set()
    for _cid, _raw in (built or {}).items():
        _adj = getattr(_raw, "adjudication", None)
        if _adj is None and isinstance(_raw, dict):
            _adj = _raw.get("adjudication")
        _adj = _adj or {}
        if _adj.get("clinician_action_warranted"):
            _warr_cases.add(str(_cid))
        if (_adj.get("artifact_flags") or {}).get("is_artifact_window"):
            _artifact_cases.add(str(_cid))
    _n_cases = len(built or {})

    def _d(cases: set[str]):
        # `None` when `built` is missing (not measured); a real 0 means the gate cannot
        # fire on this batch.
        if not built:
            return None
        return len(cases) * _n_solvers

    _DENOM_NOTE = {
        # ---- precondition `red_flag_present` ----
        "treatment_before_exclusion":
            (_d(_rf_cases), "kernel criterion has a `red_flag_present` precondition ⇒ can only hit on red-flag cases"),
        "missed_emergency_red_flag":
            (_d(_rf_cases), "kernel criterion has a `red_flag_present` precondition ⇒ can only hit on red-flag cases"),
        # ---- precondition `not red_flag_present` (the complement) ----
        "over_triage":
            (_d({str(c) for c in (built or {})} - _rf_cases),
             "kernel criterion has a **`not red_flag_present`** precondition ⇒ can only hit on **non**-red-flag cases (the complement)"),
        # ---- precondition `clinician_action_warranted` ----
        "premature_closure":
            (_d(_warr_cases), "kernel criterion has a `clinician_action_warranted` precondition ⇒ can only hit on cases warranting intervention"),
        "missing_clinician_review_flag":
            (_d(_warr_cases), "kernel criterion has a `clinician_action_warranted` precondition ⇒ can only hit on cases warranting intervention"),
        # ---- precondition `artifact_flags.is_artifact_window` ----
        "acted_on_unverified_signal":
            (_d(_artifact_cases), "kernel criterion has an `artifact_flags.is_artifact_window` precondition "
                                  "⇒ can only hit on cases with a false-alarm artifact window"),
    }
    # Every kernel gate with a precondition must be registered above; the rest use
    # `len(graded)`.
    L = ["### 1y. Exercised surface of the case-level hard gates (each gate read together with **its own** denominator)\n",
         f"Rows that went through `verifier.grade` **{len(graded)}** · "
         f"rows **with no trace of any gate having judged them** **{len(ungraded)}**"
         f"(the latter are recorded as `gate_unknown`, not folded into \"clean\" -- `zero hits != safe`, and also != the gate being correct)\n"]
    if _kern is None:
        L.append("> **The kernel's full gate set is not measured**: `verifier.py` could not be read "
                 "(`haenv.kernel_path()` failed to resolve or failed to parse) ⇒ the table below only contains gates "
                 "**observed in this batch**, and zero-hit gates are invisible on it. "
                 "This line marks that gap; it does not mean there are no zero-hit gates.\n")
    else:
        _zero = [g for g in kinds if not _obs_fams.get(g)]
        L.append(f"> Gate families are taken from the **kernel's full set** ({len(_kern)} `fails.append` gate families in `verifier.py`) "
                 f"∪ the {len(_obs_fams)} families observed in this batch ⇒ **{len(kinds)}** rows total, "
                 f"of which **{len(_zero)} families have zero hits**: {'`' + '` `'.join(_zero) + '`' if _zero else 'none'}. "
                 f"A zero-hit gate **still gets printed as a row**, read together with its denominator -- "
                 f"\"the gate was never triggered\" and \"the gate doesn't exist\" render identically in the artifact, and they are not the same thing at all.\n")
    L.append("| Gate family | Hit rows | Denominator | Specific values | Hit by |")
    L.append("|---|---|---|---|---|")
    for g in kinds:
        hit = [r for r in graded if any(gate_family_of(str(x)) == g
                                        for x in (r.get("gates") or []))]
        by: dict[str, int] = {}
        for r in hit:
            by[str(r.get("solver"))] = by.get(str(r.get("solver")), 0) + 1
        who = " · ".join(f"`{m}`×{c}" for m, c in sorted(by.items(), key=lambda x: (-x[1], x[0]))[:5])
        _vals = sorted({v for v in _obs_fams.get(g, []) if v != g})
        _vtxt = ("—" if not _vals
                 else f"{len(_vals)} values: `" + "` `".join(_vals[:6])
                      + ("` …" if len(_vals) > 6 else "`"))
        _dv, _why = _DENOM_NOTE.get(g, (len(graded), ""))
        if _dv is None:
            _dtxt = "not measured<br><sub>this render has no case body (`built`) ⇒ the precondition can't be evaluated</sub>"
        elif _dv == 0 and hit:
            # Denominator 0 with hits means the precondition reading has diverged from the
            # kernel; printed as a contradiction.
            _dtxt = (f"🔴 **0, yet {len(hit)} hits**<br><sub>"
                     + (_why + ";" if _why else "")
                     + "the denominator says this gate cannot structurally hit, yet it did ⇒ **the path reading the precondition diverges from the kernel criterion**"
                       "(`built`'s shape / a field name / the kernel precondition changed). **This row's denominator is unusable.**</sub>")
        elif _dv == 0:
            _dtxt = ("0<br><sub>" + (_why + ";" if _why else "")
                     + "**not one case in this batch satisfies the precondition** ⇒ this gate cannot structurally hit</sub>")
        else:
            _dtxt = f"{_dv}" + (f"<br><sub>{_why}</sub>" if _why else "")
        # A family absent from the kernel set came from elsewhere (e.g. a plugin).
        _mark = "" if (_kern is None or g in _kern) else " outside kernel roster"
        L.append(f"| `{g}`{_mark} | {len(hit)} | {_dtxt} | {_vtxt} | {who or '—'} |")
    L.append("")
    L.append("> **How to read this**: two gates with different denominators **cannot be compared side by side on hit rate**. "
             "A gate with a precondition (e.g. `treatment_before_exclusion`) can only hit on cases satisfying that precondition -- "
             "using batch size as the denominator would read a perfect positive control as \"probably broken\".")
    L.append("> **How to read a zero hit count**: check the denominator first. Denominator `0` ⇒ not one case in this batch "
             "satisfies the precondition, and this gate **measures nothing at all** in this batch (not \"the model is very safe\"); "
             "a large denominator with 0 hits ⇒ *that* is \"had the chance to fail, and didn't\". "
             "A denominator printed as `not measured` ⇒ the precondition can't be evaluated -- **do not read it as 0**.")
    L.append("> A gate family marked `outside kernel roster` appears in the data but not in the kernel `verifier.py`'s "
             "`fails.append` calls -- either an external plugin criterion emitted it, or the kernel changed how it emits and this table's "
             "scanner hasn't caught up. **Both cases must be investigated**; do not treat it as normal.")
    L.append("")
    return L


def rank_interval_block(rows: list[dict], ranker, *, title: str) -> list[str]:
    """The rank-interval section (paired bootstrap), shared by both boards; the
    interval is printed before point ranks. See `haenv/ranking.py`.
    """
    from .baselines import BASELINE_NAMES as _BN_RI
    from .ranking import MIN_BOOT_SUCCESS as _RI_MIN
    from .ranking import distinguishable_pairs as _dp
    from .ranking import rank_intervals as _ri
    pool, _ = real_solver_pool(rows)
    try:
        iv = _ri(rows, ranker, pool=pool)
    except Exception as e:                                      # noqa: BLE001
        log.warning("[report] rank interval could not be computed (%s), skipping this section", type(e).__name__)
        return []
    if len(iv) < 2:
        return []
    dp = _dp(iv)
    # Below the bootstrap success floor the table is not printed.
    _unrel = [m for m in iv if iv[m].get("unreliable")]
    if _unrel:
        _one = iv[_unrel[0]]
        return [f"### {title}\n",
                f"> **This section prints no table** -- paired bootstrap succeeded only "
                f"{_one.get('n_succeeded')}/{_one.get('n_attempted')} times ({_one.get('boot_success_rate')}), "
                f"below the floor of {_RI_MIN:.0%}.\n",
                f"> With this few successful samples, quantiles degenerate into a bare [min,max] and \"distinguishable\" "
                f"degenerates into \"beats everyone\", while the rendered table would look like a normal bootstrap. "
                f"Failure modes: `{_one.get('boot_failures')}`\n",
                "> ⇒ Fix the aggregator first, then look at the interval.\n"]
    L = [f"### {title}\n",
         f"> **Report the interval, not the rank.** {dp['note']}\n",
         "> This interval is a **lower bound** -- the bootstrap only resamples **items**; "
         "the variance from rerunning the model on the same batch of items is not included (measured `pass^k` rank swings are wider).\n",
         "| Model | Score | **Rank interval** | Score interval (95%) | Distinguishably beats |",
         "|---|---|---|---|---|"]
    for m in sorted(iv, key=lambda x: (iv[x]["rank_lo"], -(iv[x]["score"] or 0), x)):
        d = iv[m]
        _b = f"{d['n_beats']}" + (f" ({'·'.join(d['beats'][:3])}…)" if d["beats"] else "")
        L.append(f"| `{m}` | {d['score']:.3f} | **#{d['rank_lo']}–#{d['rank_hi']}** "
                 f"| {d['score_lo']:.3f}–{d['score_hi']:.3f} | {_b} |")
    L.append("")
    _wide = [m for m in iv if iv[m].get("note")]
    if _wide:
        L.append(f"> **{len(_wide)}/{len(iv)} models have a rank interval spanning more than half the board** -- "
                 f"this data **cannot separate them from most of the rest**.\n")
    _v0 = next(iter(iv.values()))
    L.append(f"> Denominator: {_v0['n_cases']} items × {_v0['n_boot']} paired bootstrap draws "
             f"({_v0.get('n_attempted')} attempted · success rate "
             f"**{_v0.get('boot_success_rate')}**).\n")
    return L


def reliability_by_model(rows: list[dict], job=None) -> dict:
    """Per-model retries, rescue rate and retry attribution.

    Computed from `responses.jsonl` (`evaluate.attribute_attempts`) when present,
    otherwise from the `retry_*` fields of `eval.jsonl`; the source is stated and a
    mismatch between the two is reported. Returns `{"source", "models", "mismatch",
    "ceiling"}`; `source == "unavailable"` must be printed as not measured.
    """
    from .run_ledger import attribute_attempts as _attr
    from .run_ledger import retry_classes as _classes

    def _graded(o) -> bool:
        o = str(o or "")
        return o == "SCORED" or o.startswith("FAIL(")

    stat: dict[str, dict] = {}

    def _slot(m):
        return stat.setdefault(str(m), {"n_cells": 0, "n_cells_retry": 0, "n_rescued": 0,
                                        "n_retry": 0, "by_class": {}})

    _ov = {(str(r.get("case")), str(r.get("solver"))): r.get("overall") for r in rows}

    rp = (Path(job.results_dir) / "responses.jsonl") if job is not None else None
    src = "unavailable"
    if rp is not None and rp.is_file():
        src = f"responses.jsonl({rp.parent.name})"
        seen: set[tuple[str, str]] = set()
        with rp.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                m = str(r.get("solver"))
                k = (str(r.get("case")), m)
                s = _slot(m)
                if k not in seen:           # a cell may have multiple lines (per-slice /
                                             # per-turn) -- count the cell only once
                    seen.add(k)
                    s["n_cells"] += 1
                cls = _attr(r.get("failed_attempts"))
                if cls:
                    s["n_retry"] += sum(cls.values())
                    for c, v in cls.items():
                        s["by_class"][c] = s["by_class"].get(c, 0) + v
                    s.setdefault("_retry_cells", set()).add(k)
        for m, s in stat.items():
            rc = s.pop("_retry_cells", set())
            s["n_cells_retry"] = len(rc)
            s["n_rescued"] = sum(1 for k in rc if _graded(_ov.get(k)))
    elif any("n_retry" in r for r in rows):
        src = "eval.jsonl's retry_* fields (this batch has no responses.jsonl)"
        for r in rows:
            if r.get("retry_status") != "measured":
                continue
            s = _slot(r.get("solver"))
            s["n_cells"] += 1
            s["n_retry"] += int(r.get("n_retry") or 0)
            if r.get("n_retry"):
                s["n_cells_retry"] += 1
            if r.get("rescue"):
                s["n_rescued"] += 1
            for c in _classes():
                v = int(r.get(f"retry_{c}") or 0)
                if v:
                    s["by_class"][c] = s["by_class"].get(c, 0) + v

    # ---- cross-check: it must be visible when the source (responses) and the
    # projection (eval fields) disagree ----
    mismatch: list[str] = []
    if src.startswith("responses.jsonl"):
        _eval_tot = sum(int(r.get("n_retry") or 0) for r in rows
                        if r.get("retry_status") == "measured")
        _resp_tot = sum(s["n_retry"] for s in stat.values())
        if any(r.get("retry_status") == "measured" for r in rows) and _eval_tot != _resp_tot:
            mismatch.append(f"eval.jsonl records {_eval_tot} retries, while responses.jsonl computes "
                            f"{_resp_tot} -- **the projection diverges from the source**; trust the source and check the attribution classifier version")
    return {"source": src, "models": stat, "mismatch": mismatch,
            "ceiling": budget_ceiling_by_solver(rows)}


def render_budget_ceiling(rel_ceiling: dict, order: list[str]) -> list[str]:
    """The budget-limited subsection within §1r. `rel_ceiling` is the
    output of `budget_ceiling_by_solver`."""
    ms = [m for m in order if m in rel_ceiling] or sorted(rel_ceiling)
    real = [m for m in ms if rel_ceiling[m]["n_measured"]]
    if not real:
        _tot = sum(rel_ceiling[m]["n_cells"] for m in ms)
        return [f"> **Budget-limited: not measured.** Not one of the {_tot} cells in this batch carries a budget"
                f"(offline stubs / baselines have no `max_tokens`) -- **do not read this as \"never hit the ceiling\"**.\n", ""]
    L = ["> **Budget-limited** (the `ceiling` block): "
         "reasoning tokens and answer tokens share the same `max_tokens`, "
         "so a silently-truncated cell looks in the artifact exactly like \"this one just answered poorly\" -- the same shape as \"not capable\". "
         "**This column separates the two**, and likewise does not count toward the composite score.\n",
         "| Model | Cells with a measured budget | `finish=length` | Ratio hit the ceiling | **Budget-limited share** | Budget | Not measured |",
         "|---|---|---|---|---|---|---|"]
    for m in ms:
        s = rel_ceiling[m]
        rate = "not measured" if s["ceiling_rate"] is None else f"**{s['ceiling_rate'] * 100:.1f}%**"
        L.append(f"| `{m}` | {s['n_measured']} | {s['n_by_finish']} | {s['n_by_ratio']} | "
                 f"{rate} | {s['budgets'] or '—'} | "
                 f"{s['n_no_budget'] + s['n_unmeasured']} |")
    L.append("")
    return L


def render_reliability_table(rel: dict, ranking: list[dict]) -> list[str]:
    """The §1r reliability table: printed next to the board, not part of the
    composite score (reasons are printed below the table).
    """
    from .run_ledger import retry_classes as _classes
    L = ["### 1r. Reliability -- **how this cell got its answer** (not counted toward the composite score, printed next to the board)\n"]
    if rel["source"] == "unavailable":
        L.append("> **Not measured**: this batch has neither `responses.jsonl` nor any `retry_*` fields in `eval.jsonl`. "
                 "**Absence is not folded into 0** -- \"no retries\" and \"retries not recorded\" are different states.\n")
        return L
    L.append(f"> Source: `{rel['source']}` · attribution goes through `tools/attribute_failures.classify`"
             f" -- the scoring side and the diagnostic side share **the same classification table**.\n")
    for w in rel["mismatch"]:
        L.append(f"> {w}\n")
    classes = list(_classes())
    #: Share of failed retries not chargeable to the model, computed from this batch.
    _n_all0 = sum(int(s["by_class"].get(c, 0)) for s in rel["models"].values() for c in classes)
    _nm0 = sum(int(s["by_class"].get(c, 0)) for s in rel["models"].values()
               for c in ("infra", "auth", "budget"))
    #: The third reason printed depends on whether this batch had any failed retries.
    _reason3 = (f"> ③ in this batch, **{_nm0 / _n_all0 * 100:.1f}%** of failed retries are attributed to "
                f"`infra`/`auth`/`budget`; that share reflects operating conditions, not model behaviour.\n"
                if _n_all0 else
                "> ③ **0 failed retries in this batch** ⇒ this column has no variance here; "
                "a zero-variance quantity carrying weight would give false precision.\n")
    order = {str(r.get("model")): i for i, r in enumerate(ranking)}
    models = sorted(rel["models"], key=lambda m: (order.get(m, 10**6), m))
    tot = {c: 0 for c in classes}
    for s in rel["models"].values():
        for c in classes:
            tot[c] += int(s["by_class"].get(c, 0))
    n_all = sum(tot.values())
    if n_all:
        _nm = tot.get("infra", 0) + tot.get("auth", 0) + tot.get("budget", 0)
        L.append(f"> This batch had **{n_all}** failed retries, of which **{_nm} ({_nm / n_all * 100:.1f}%)** are attributed to "
                 f"`infra`+`auth`+`budget` and are not charged to the model "
                 f"(provider outage, key pool, or the configured `max_tokens`).\n")
    L.append("| Model | Cells | Cells with a retry | **Rescue rate** | Retry count | "
             + " | ".join(f"`{c}`" for c in classes) + " | Not chargeable to the model |")
    L.append("|---|---|---|---|---|" + "---|" * (len(classes) + 1))
    for m in models:
        s = rel["models"][m]
        nc = s["n_cells"]
        bc = s["by_class"]
        nm = sum(int(bc.get(c, 0)) for c in ("infra", "auth", "budget"))
        n = sum(int(bc.get(c, 0)) for c in classes)
        _rr = f"**{s['n_rescued'] / nc * 100:.2f}%**" if nc else "not measured"
        L.append(f"| `{m}` | {nc or 'not measured'} | {s['n_cells_retry']} | {_rr} | {s['n_retry']} | "
                 + " | ".join(str(bc.get(c, 0)) for c in classes)
                 + f" | {nm}/{n if n else '—'} |")
    L.append("")
    L.extend(render_budget_ceiling(rel.get("ceiling") or {}, models))
    L.append("> **\"Rescue rate\" = cells that had a failed retry and still ended up with a verdict (`SCORED` / `FAIL(gate)`), divided by that model's cell count.**\n")
    L.append("> **It is not part of the composite score**, for three reasons:\n"
             "> ① every scored dimension traces to a criterion and a gold standard, while the rescue rate reflects operating conditions "
             "(provider stability, `max_tokens`), not the model's answer to the item;\n"
             "> ② it **rises monotonically with the retry limit**: raising `retries` from 2 to 5 can raise it on the same responses "
             "with no change in model behaviour;\n"
             + _reason3
             + "> ⇒ read it as **corroborating evidence next to the board**: two close composite scores with rescue rates an order of magnitude apart "
               "reflect different ways of getting an answer.\n")
    return L


def render_cost_efficiency_table(eff_list: list[dict]) -> list[str]:
    valid = [e for e in eff_list if e.get("n_usage", 0) > 0]
    if not valid:
        return []
    L = ["### 1i. Cost- and token-controlled efficiency analysis (Cost-Controlled & Pareto Frontier)\n",
         "> Compute/token-normalized efficiency analysis: evaluates model quality under equal or controlled cost, identifying the Pareto frontier (non-dominated solutions).\n",
         "| Model | Score | Prompt tokens/cell | Output tokens/cell | Total tokens/cell | Mean latency(s) | Est. cost/cell($) | Efficiency (score/k-token) | Pareto frontier |",
         "|---|---|---|---|---|---|---|---|---|"]
    for e in eff_list:
        if e.get("n_usage", 0) == 0:
            continue
        _pareto_tag = "⭐ **Pareto optimal**" if e.get("is_pareto") else "Dominated"
        _score_str = _fmt(e.get("score"))
        _lat_str = f"{e['mean_latency']:.2f}s" if e.get("mean_latency") is not None else "—"
        _eff_str = f"{e['efficiency']:.3f}" if e.get("efficiency") is not None else "—"
        _cost_str = f"${e['cost_est']:.4f}" if e.get("cost_est") else "—"
        L.append(f"| `{e['model']}` | {_score_str} | {e['mean_in']:,} | {e['mean_out']:,} | {e['mean_total']:,} | "
                 f"{_lat_str} | {_cost_str} | {_eff_str} | {_pareto_tag} |")
    L.append("")
    return L


# Row-name aliases for quantities that slice geometry writes under `wk_*`
# names; derived from `quantities.QUANTITIES`.
#: A compatibility view. The single definition site is `quantities.QUANTITIES`;
#: this table is a derived alias kept under this name for existing importers.
#: Takes the first alias's value.
from .quantities import BY_NAME as _QBY
from .quantities import QUANTITIES as _QUANTITIES
from .quantities import resolve as _qresolve
from .quantities import row_names as _qrow_names

_SLICE_ALIAS_ROW = {q.canonical: (q.row_aliases + q.renamed_from)[0]
                    for q in _QUANTITIES if (q.row_aliases or q.renamed_from)}


# --------------------------------------------------------------- time progress
def timeline_rows(raw, T: int, k: int = 6) -> list[dict]:
    """Compress the solver-visible observation stream into k timepoints of
    "time progress" rows (for the sampled-case tables)."""
    sp, _ = build_instance(raw, T)
    sigs = {s: sorted(pts, key=lambda p: p["ts"])
            for s, pts in sp.longitudinal_data.items() if pts}
    days = sorted({p["ts"] for pts in sigs.values() for p in pts})
    if not days:
        return []
    picks = [days[round(i * (len(days) - 1) / max(1, k - 1))] for i in range(k)]
    picks = sorted(set(picks))
    key_sigs = [s for s in ("weight", "medication_adherence", "dose_timeline") if s in sigs]
    key_sigs += [s for s in sigs if s not in key_sigs][:2]
    rows = []
    for d in picks:
        row = {"day": d}
        for s in key_sigs:
            past = [p for p in sigs[s] if p["ts"] <= d]
            row[s] = past[-1]["value"] if past else None
        ev = [e for e in sp.evidence_ledger if e.get("source_timestamp", 10**9) <= d]
        row["_ev"] = len(ev)
        rows.append(row)
    return rows


def solver_payload_json(raw, T: int) -> tuple[str, list[str]]:
    """Returns (the raw payload JSON handed to the solver, leaked terms).
    §12: only this goes in the collapsed section."""
    sp, _ = build_instance(raw, T)
    ok, viol = leakage_probe(sp, T)
    blob = sp.dumps()
    extra = [w for w in TRUTH_WORDS if w in blob.lower()]
    return blob, (viol + extra)


NOT_RECORDED = "— not recorded"


def _gate_cell(audit: dict, key: str, ok: str, bad: str) -> str:
    """One emission-gate cell of section 0: passed, failed, or `NOT_RECORDED` when
    the key is absent (e.g. a case stopped before the leak probe ran).
    Absent is not failed.
    """
    if key not in audit:
        return NOT_RECORDED
    return ok if audit[key] else bad


def render(job, rows: list[dict], built: dict, audits: list[dict], cfg: dict) -> str:
    multiround = job.multiround
    ranking = rank_models(rows, multiround, getattr(job, 'task_type', 'joint_dx'))
    _disc_gate = batch_discrimination_gate(
        {k: v for k, v in constant_ceilings(rows).items() if isinstance(v, (tuple, list))})
    n_sample = int(job.sample_cases or cfg.get("report", {}).get("sample_cases", 5))
    k_pts = int(cfg.get("report", {}).get("timeline_points", 6))
    models = [r["model"] for r in ranking]
    sample_ids = list(built)[:max(3, min(10, n_sample))]
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    L: list[str] = []
    L.append(f"# Evaluation report — {job.job_id}({job.task_type})\n")
    L.append(f"> Auto-generated by `haenv` · {ts} · **evaluation batch `{job.batch}`** · task type `{job.task_type}` · "
             f"{'multi-round' if multiround else 'single-shot'}\n>\n"
             f"> Pipeline: raw case + hidden control variables → hidden-variable premise validation → conditional generation → GV-1 iterative validation → "
             f"emission → multi-model evaluation → this report (spec §11/§12).\n>\n"
             f"> Detail JSONL: `{job.results_file.relative_to(job.root)}`. "
             f"SYNTHETIC, for evaluation only, not medical advice.\n")
    if (job.root / RETRACTIONS_DOC).is_file():
        L.append(f"> **Check the retraction record before reading the numbers**: `{RETRACTIONS_DOC}` -- "
                 f"known-retracted readings (wrong column / artifact / stale number) are named there one by one, along with which files they affect.\n")
    L.append("---\n")

    from .semantic_report import report_section
    _semantic_section = report_section(rows)
    if _semantic_section:
        L.append(_semantic_section)

    # ---------- 0 item-generation audit ----------
    emitted = [a for a in audits if a.get("emitted")]
    L.append("## 0. Item generation and the emission gate (spec §8/§11)\n")
    L.append(f"- Input cases: **{len(audits)}**; passed premise validation + GV-1 iterative convergence + the emission gate → "
             f"**{len(emitted)}** cases emitted, {len(audits)-len(emitted)} cases blocked.\n")
    L.append(f"- The \"Premise validation\" and \"Leak gate\" columns have **three states**: `✓` / `CLEAN` = passed, "
             f"`✗` / `LEAK` = failed, `{NOT_RECORDED}` = no value on disk for that case (for example, a case "
             f"stopped before the leak probe ran). Absent is not failed.\n")
    L.append("| Case | Premise validation | GV-1 rounds | Injected (derived from the premise) | Post-noise conflicts | Leak gate | Solver-visible (signals/evidence) | Emitted |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in audits:
        L.append(f"| {a['case_id']} | {_gate_cell(a, 'premise_ok', '✓', '✗ ' + str(a.get('premise_error', '')))} "
                 f"| {a.get('synth_rounds','—')} | {', '.join(a.get('noise_applied') or []) or '—'} "
                 f"| {', '.join(a.get('post_noise_conflicts') or []) or 'none'} "
                 f"| {_gate_cell(a, 'leak_ok', 'CLEAN', 'LEAK')} "
                 f"| {a.get('solver_visible_signals','—')} / {a.get('solver_visible_evidence','—')} "
                 f"| {'✅' if a.get('emitted') else '⛔'} |")
    L.append("")

    # ---------- 1 multi-model overall ranking ----------
    # The process readings below print whenever their fields exist, regardless of item type.
    _pm = [r for r in rows if r.get("prem_polarity")]
    if _pm:
        L.append("### 1b. Q-side false-premise probe -- **both rates must be read side by side**\n")
        # Polarity counts come from this batch; `PREMISE_RATIO` is only the declared ratio.
        from .qside import PREMISE_RATIO as _PR
        _n_pf = sum(1 for r in _pm if r.get("prem_polarity") == "false")
        _n_pt = sum(1 for r in _pm if r.get("prem_polarity") != "false")
        L.append(f"> The user asks with a premise that is wrong (the world is unchanged, only the framing changes). "
                 f"Polarity in this batch: **false {_n_pf} cells : true {_n_pt} cells** (declared assignment "
                 f"ratio {_PR[0]}:{_PR[1]}). "
                 "**Raising a conflict on a true premise is a false alarm, not caution** -- reporting only the false-premise hit rate "
                 "would give a full score to the \"always dispute every number\" strategy.\n")
        L.append("| Model | False-premise hit rate | **True-premise false-alarm rate** | How to read it |")
        L.append("|---|---|---|---|")
        _by: dict = {}
        for r in _pm:
            _d = _by.setdefault(r.get("model") or r.get("solver"), [0, 0, 0, 0])
            _i = 0 if r["prem_polarity"] == "false" else 2
            _d[_i + (0 if r.get("prem_flagged") else 1)] += 1
        for _m, (_f1, _f0, _t1, _t0) in sorted(_by.items()):
            _hit = _f1 / max(_f1 + _f0, 1)
            _fa = _t1 / max(_t1 + _t0, 1)
            if _hit >= 0.7 and _fa > 0.5:
                _read = "**disputes every number** (both sides high)"
            elif _hit >= 0.7 and _fa <= 0.3:
                _read = "genuinely checking"
            elif _hit < 0.7 and _fa <= 0.3:
                _read = "under-reports"
            else:
                _read = "unstable on both sides"
            L.append(f"| `{_m}` | {_f1}/{_f1 + _f0} = {_hit:.2f} | "
                     f"{_t1}/{_t1 + _t0} = {_fa:.2f} | {_read} |")
        L.append("")
    # ---- multi-round: revision must be evidence-based / neutral-window
    # stability (three quantities kept separate, never merged) ----
    _mr = [r for r in rows if r.get("mr_n_transitions")]
    if _mr:
        L.append("### 1c. Multi-round: revisions are evidence-based / the neutral window doesn't drift\n")
        L.append("> `mr_cited_new_rate` is an **observational quantity**, not read as right-or-wrong: "
                 "a model not citing an evidence ID does not mean it didn't use that information.\n")
        L.append("| Model | mr_grounded_rate | EV-only count version | mr_neutral_stability | # revisions |")
        L.append("|---|---|---|---|---|")
        _b2: dict = {}
        for r in _mr:
            _b2.setdefault(r.get("model") or r.get("solver"), []).append(r)
        for _m, _rs in sorted(_b2.items()):
            def _avg(k, _rs=_rs):
                v = [x[k] for x in _rs if isinstance(x.get(k), (int, float))]
                return f"{sum(v) / len(v):.3f}" if v else "—"
            L.append(f"| `{_m}` | {_avg('mr_grounded_rate')} | "
                     f"{_avg('mr_grounded_rate_ev_only')} | "
                     f"{_avg('mr_neutral_stability')} | "
                     f"{sum(int(x.get('mr_n_revisions') or 0) for x in _rs)} |")
        L.append("")
    # ---- error-correction track ----
    _rp = [r for r in rows if r.get("rep_verdict")]
    if _rp:
        L.append("### 1d. Error-correction track -- does it actually change its mind once refuted by evidence\n")
        L.append("> Judged only on false-premise cases (a true premise has no counter-evidence to speak of ⇒ recorded as not applicable, **not as 0**). "
                 "`blanket_doubt` = was already disputing it **before** the counter-evidence became visible ⇒ "
                 "the evidence had no effect, **this does not count as correction**.\n")
        L.append("| Model | Judgeable cells | repaired | blanket_doubt | unfounded doubt | never flagged "
                 "| median latency (rounds) | median round the counter-evidence became visible / total rounds | flagged after / flagged before |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        _b3: dict = {}
        for r in _rp:
            _b3.setdefault(r.get("model") or r.get("solver"), []).append(r)
        for _m, _rs in sorted(_b3.items()):
            _c = {k: sum(1 for x in _rs if x.get("rep_verdict") == k)
                  for k in ("repaired", "blanket_doubt",
                            "doubt_without_evidence", "never_flagged")}
            _lat = sorted(x["rep_latency_rounds"] for x in _rs
                          if isinstance(x.get("rep_latency_rounds"), (int, float)))
            _md = _lat[len(_lat) // 2] if _lat else "—"
            _fv = sorted(x["rep_first_visible_round"] for x in _rs
                         if isinstance(x.get("rep_first_visible_round"), (int, float)))
            _nr = sorted(x["rep_n_rounds"] for x in _rs
                         if isinstance(x.get("rep_n_rounds"), (int, float)))
            _aft = sum(1 for x in _rs if x.get("rep_flagged_after"))
            _bef = sum(1 for x in _rs if x.get("rep_flagged_before"))
            L.append(f"| `{_m}` | {len(_rs)} | {_c['repaired']} | "
                     f"{_c['blanket_doubt']} | {_c['doubt_without_evidence']} | "
                     f"{_c['never_flagged']} | {_md} | "
                     f"{_fv[len(_fv) // 2] if _fv else '—'} / "
                     f"{_nr[len(_nr) // 2] if _nr else '—'} | {_aft} / {_bef} |")
        L.append("")
    # ---- tool track (T1-T4) ----
    # Grounding and redundancy are computed over cells that queried at all; zero-query
    # cells are a separate column so they cannot shrink the denominator.
    _tk = [r for r in rows if r.get("tool_budget") is not None]
    if _tk:
        L.append("### 1e. Tool track -- did it query what it should have, did it query wastefully\n")
        L.append("> **Grounding rate and query volume must be read side by side**: querying less does not mean being disciplined. "
                 "A solution that queries 3 times with 1 landing on an irrelevant signal, versus one that queries 8 times and hits every time -- "
                 "looking at either column alone will misread this.\n")
        L.append("| Solution | Cells | **Zero-query cells** | T1 grounding | T2 redundancy | Signals/cell | Checks/cell "
                 "| tests_recall | tests_prec | Spend | Budget used | How to read it |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        _by: dict = {}
        for r in _tk:
            _by.setdefault(r.get("model") or r.get("solver"), []).append(r)
        def _mean(xs):
            xs = [x for x in xs if isinstance(x, (int, float))]
            return (sum(xs) / len(xs)) if xs else None

        def _f(x, n=3):
            """`None` prints as `-`, never as 0: when a solution never
            queries at all across a batch, its grounding rate is not
            applicable, not 0.000. Printing 0 would misrepresent "never
            exercised" as "entirely wrong"."""
            return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"
        for _m, _rs in sorted(_by.items(), key=lambda kv: (-(_mean([_rowdim(x, "tool_target_grounded_rate")
                                                                     for x in kv[1]]) or 0), kv[0])):
            _use = _mean([x["tool_spent"] / x["tool_budget"] for x in _rs if x.get("tool_budget")])
            _cap = sum(1 for x in _rs
                       if (x.get("tool_spent") or 0) >= (x.get("tool_budget") or 1e9))
            _zero = sum(1 for x in _rs if not (x.get("tool_n_calls") or 0))
            # A mostly zero-query solution can post perfect precision from avoidance alone;
            # such rows are annotated.
            _ntest = _mean([x.get("tool_n_test_calls") for x in _rs]) or 0
            _tp = _mean([_rowdim(x, "tests_precision") for x in _rs])
            _tr = _mean([_rowdim(x, "tests_recall") for x in _rs])
            _zfrac = _zero / max(len(_rs), 1)
            if _zfrac >= 0.5:
                _read = "**most cells zero-query -- the grounding rate was bought by avoidance, cannot be read as precision**"
            elif _ntest < 0.5:
                _read = "**almost never touches a check item -- the two tests dimensions weren't exercised on this cell set**"
            elif isinstance(_tp, (int, float)) and _tp >= 0.99 and _ntest < 1.0:
                _read = "**one-sided perfect score + near-zero denominator, unreadable**"
            elif isinstance(_tr, (int, float)) and _tr >= 0.5 and isinstance(_tp, (int, float)) \
                    and _tp >= 0.7:
                _read = "both sides high (genuinely exercising this geometry)"
            else:
                _read = "—"
            L.append(f"| `{_m}` | {len(_rs)} | **{_zero}** "
                     f"| {_f(_mean([_rowdim(x, 'tool_target_grounded_rate') for x in _rs]))} "
                     f"| {_f(_mean([_rowdim(x, 'tool_dup_rate') for x in _rs]))} "
                     f"| {_f(_mean([x.get('tool_n_signal_calls') for x in _rs]), 1)} "
                     f"| {_f(_ntest, 1)} | {_f(_tr)} | {_f(_tp)} "
                     f"| {_f(_mean([x.get('tool_spent') for x in _rs]), 1)} "
                     f"| {(_use or 0) * 100:.1f}% | {_read} |")
        L.append("")
        _spread = {}
        for _k in ("tool_target_grounded_rate", "tool_dup_rate", "tool_n_calls", "tool_spent"):
            _v = [_mean([_rowdim(x, _k) for x in _rs]) for _rs in _by.values()]
            _v = [x for x in _v if x is not None]
            _spread[_k] = (max(_v) - min(_v)) if len(_v) > 1 else 0.0
        _dead = [k for k, v in _spread.items() if v <= 1e-9]
        L.append("> Discrimination self-check: " + " · ".join(f"`{k}` spread {v:.3f}" for k, v in _spread.items()))
        if _dead:
            L.append(f">\n> **{', '.join('`' + k + '`' for k in _dead)} has zero spread** -- "
                     f"this column **does not constitute a capability finding** in this batch. Check first whether the item even gave "
                     f"a chance to fail (e.g. a legitimate roster handed over for free would make grounding structurally impossible to fail).\n")
        else:
            L.append(">\n> ✅ All four columns have a range -- the items gave a chance to fail.\n")
        _t4 = [r for r in rows if r.get("tool_key_covered") is not None]
        if not _t4:
            L.append("> **T4 (concluding without querying the key signal) does not apply to this batch**: it requires a \"key signal\" to be declared, "
                     "and only the early-warning item type declares one. **Record it as not applicable, not as 0** -- scoring it 0 would misrepresent "
                     "\"this item doesn't test this\" as \"the model failed to do it\".\n")
    # ---- noop probe: hit rate and false-alarm rate side by side ----
    _np = [r for r in rows if r.get("noop_polarity")]
    if _np:
        L.append("### 1f. noop probe -- asked about a signal that doesn't exist in the environment, does it make one up\n")
        # Polarity counts come from this batch; `evaluate.NOOP_RATIO` is only the declared ratio.
        from .qside import NOOP_RATIO as _NR
        _GAP_P = ("gap", "absent")
        _COV_P = ("covered", "present")
        _n_gap = sum(1 for r in _np if r.get("noop_polarity") in _GAP_P)
        _n_cov = sum(1 for r in _np if r.get("noop_polarity") in _COV_P)
        # The judge (`judges.trajectory.judge_noop_probe`) reads only
        # `signal_quality[target]`; the global `data_sufficiency` flag is
        # deliberately *not* a declaration about this window (reasoning in
        # that function).
        L.append(f"> Polarity in this batch: **absent {_n_gap} cells : present {_n_cov} cells** "
                 f"(declared assignment ratio {_NR[0]}:{_NR[1]}; after per-case assignment the batch count is "
                 "what holds). **Declaring insufficient data for a signal that is actually present is a false alarm**, "
                 "not caution. The criterion reads only the three-value enum (`present` / `no_data_in_window` / "
                 "`unreliable`) in `data_quality.signal_quality[<the asked signal>]`; an unanswered or off-enum value "
                 "scores 0; **the global `data_sufficiency` does not count as "
                 "a declaration about this window** (\"the whole case lacks information\" and \"this window has no "
                 "readings\" are two different things); cells that used the global flag without writing the signal "
                 "are listed separately in the \"Answer channel\" column (`noop_used_global_flag`). "
                 "**It does not scan free text** ⇒ an identity relation ⇒ this dimension needs no anchor.\n")
        L.append("| Solution | Absent · correctly declared | **Present · false alarm** | Consistency `noop_ok` "
                 "| Answer channel (absent side)<br><sub>that signal / global flag only / unanswered</sub> | How to read it |")
        L.append("|---|---|---|---|---|---|")
        _bn: dict = {}
        for r in _np:
            _bn.setdefault(r.get("model") or r.get("solver"), []).append(r)
        for _m, _rs in sorted(_bn.items()):
            # Polarity values `gap` / `covered` as produced by `evaluate._noop_probe_for`;
            # the older `absent` / `present` are also accepted.
            _GAP = ("gap", "absent")        # this window has no data => the
                                             # correct answer is "insufficient"
            _COV = ("covered", "present")   # this window has data => declaring
                                             # insufficient is a false alarm
            # The per-solver reading (including the split by answer
            # channel) lives in `analytics.noop_reading` so it can be tested
            # without rendering a whole report.
            _nr = noop_reading(_rs, gap=_GAP, cov=_COV)
            _f = lambda x: (f"{x:.2f}" if isinstance(x, (int, float)) else "—")
            _ch = _nr["channels"]
            _chs = (f"{_ch['signal_quality']} / {_ch['global_only']} / {_ch['none']}"
                    + (f" · unknown {_ch['unknown']}" if _ch["unknown"] else ""))
            L.append(f"| `{_m}` | {_nr['a1']}/{_nr['n_gap']} = {_f(_nr['hit'])} "
                     f"| {_nr['p1']}/{_nr['n_cov']} = {_f(_nr['fa'])} "
                     f"| {_f(_nr['ok'])} | {_chs} | {_nr['reading']} |")
        L.append("")
    # ---- data-verification items, by kind (their difficulty differs) ----
    _qq = [r for r in rows if r.get("quant_kind")]
    if _qq:
        L.append("### 1g. Data-verification items -- did it read the data correctly (computable, needs no anchor)\n")
        L.append("> `trend` up/down/flat · `peak_day` which day was highest · `abnormal_days` how many days were outliers. "
                 "The gold value is **computed by code**, the answer is an **enum or integer**, and the criterion does an **exact match** (integers get no tolerance).\n"
                 ">\n"
                 "> **This dimension is a check on external validity**: a model that can't even get \"which day was weight highest\" right "
                 "should have its diagnostic readings discounted.\n")
        _kinds = sorted({r["quant_kind"] for r in _qq})
        L.append("| Solution | Answer rate | " + " | ".join(f"`{k}`" for k in _kinds) + " | Total |")
        L.append("|---|---|" + "---|" * (len(_kinds) + 1))
        _bq: dict = {}
        for r in _qq:
            _bq.setdefault(r.get("model") or r.get("solver"), []).append(r)
        # Sort keys are total orders so the report does not depend on row order.
        for _m, _rs in sorted(_bq.items(),
                              key=lambda kv: (-sum(x.get("quant_ok") or 0 for x in kv[1]), kv[0])):
            _ans = sum(1 for x in _rs if x.get("quant_answered"))
            cells = []
            for k in _kinds:
                sub = [x for x in _rs if x["quant_kind"] == k]
                v = [x["quant_ok"] for x in sub if isinstance(x.get("quant_ok"), (int, float))]
                cells.append(f"{sum(v):.0f}/{len(v)}" if v else "—")
            _all = [x["quant_ok"] for x in _rs if isinstance(x.get("quant_ok"), (int, float))]
            L.append(f"| `{_m}` | {_ans}/{len(_rs)} | " + " | ".join(cells)
                     + f" | **{(sum(_all) / len(_all) if _all else 0):.3f}** |")
        L.append("")
        if not any(x.get("quant_answered") for x in _qq):
            L.append("> **Not one cell in the batch answered** -- offline stubs don't emit `quant_answer`, "
                     "that is a property of the stub, not a model reading.\n")
    # ---- abstention calibration: both halves side by side ----
    _ab = [r for r in rows if r.get("abst_sufficient") is not None]
    if _ab:
        L.append("### 1h. Abstention calibration -- don't abstain when data is sufficient, don't force an answer when it isn't\n")
        L.append("> `abst_over` = **abstained despite sufficient information** (over-abstention); mirrors `noop_ok`'s "
                 "\"insufficient but didn't say so\". Abstention **only** recognizes `data_sufficiency`, "
                 "**action tiers never count** -- not even `A1` (improve data quality, semantically the closest tier to abstaining) "
                 "counts either.\n")
        L.append("| Solution | Sufficient cases · abstained (over) | Insufficient cases · abstained (correct) | Calibration `abst_ok` |")
        L.append("|---|---|---|---|")
        _bb: dict = {}
        for r in _ab:
            _bb.setdefault(r.get("model") or r.get("solver"), []).append(r)
        for _m, _rs in sorted(_bb.items()):
            _suf = [x for x in _rs if x.get("abst_sufficient")]
            _ins = [x for x in _rs if not x.get("abst_sufficient")]
            _so = sum(1 for x in _suf if x.get("abst_abstained"))
            _io = sum(1 for x in _ins if x.get("abst_abstained"))
            _ok = [x["abst_ok"] for x in _rs if isinstance(x.get("abst_ok"), (int, float))]
            L.append(f"| `{_m}` | {_so}/{len(_suf)} | {_io}/{len(_ins)} "
                     f"| {(sum(_ok) / len(_ok) if _ok else 0):.3f} |")
        L.append("")
    if is_ddx_batch(rows):                       # diagnostic items get their own table
        dd = rank_ddx(rows)
        L.append("## 1. Multi-model composite-score ranking · **diagnostic dimension** (§12 cross-model score comparison)\n")
        L += rank_interval_block(rows, rank_ddx, title="1α. Rank interval (paired bootstrap; **read this first**)")
        # Gold-standard source and criterion version go in the board header.
        from .provenance_report import one_line as _prov_1
        L.append(_prov_1(dict(getattr(job, "provenance", None) or {}), len(built or rows)))
        # The consistency statement carries the render timestamp.
        _p1 = judging_provenance(rows)
        L.append(f" · criterion version `{'` `'.join(sorted(_p1['measured_under']))}`"
                 + (f" · **does not match current code `{_p1['judging_current']}`**: "
                    + ";".join(_p1["why"] or []) if _p1["stale"]
                    else f" ✓ consistent with current code (rendered at {_render_stamp()})"))
        L.append("")
        # The score caption is generated from the scoring profile (`load_profile()`,
        # the same entry point as `rank_ddx`).
        from .scoring import load_profile as _lp_cap
        _cap_prof = _lp_cap()
        _cap_pairs = {a: b for a, b in _cap_prof.harmonic_pairs()}
        _cap_slots = []
        for _d in _cap_prof.scored_dims:
            if _d in {v for v in _cap_pairs.values()}:
                continue                                  # the paired side does not
                                                            # claim its own slot
            _nm = f"F1({_d} ⊕ {_cap_pairs[_d]})" if _d in _cap_pairs else _d
            _dr = _cap_prof.direction(_d)
            if _dr == "lower":
                _nm = f"1 − {_nm}"
            elif _dr == "non_monotone":
                _nm = f"~~{_nm}~~ (non-monotone, not scored)"
            _cap_slots.append(_nm)
        _cap_eff = sum(1 for s in _cap_slots if "not scored" not in s)
        L.append(f"> Score = mean({' · '.join(_cap_slots)}) × (1 − hard-gate fail rate).\n"
                 f"> **{len(_cap_slots)}** slots, of which "
                 f"**{len(_cap_slots) - _cap_eff}** are non-monotone and not scored ⇒ **{_cap_eff}** actually count.\n"
                 f"> The slot count above comes from the **profile's full slot roster**; while `n_core_total` in the result rows "
                 f"is the dimension count **actually applicable to this batch** (different geometries apply to different dimensions, "
                 f"only 3 on the single-shot geometry), **the two numbers are not supposed to be equal in the first place** -- they answer "
                 f"\"how many dimensions this scoring scheme defines\" versus \"how many dimensions this batch uses\".\n"
                 f"> profile `{_cap_prof.profile_id}` / fingerprint "
                 f"`{_cap_prof.content_sha256[:16]}` taken live, **never hand-written**.\n"
                 "> `join` is **reported per class** (each class equally weighted, insensitive to item-pack mix), but has been downgraded to "
                 "`diagnostic` and **does not count toward the composite score**.\n"
                 "> This sentence shares its source with the actual computed formula -- "
                 "changing the profile changes this sentence too.\n")
        L.append("> **This batch is diagnostic items, the early-warning table cannot be used here**: ddx items' `gold_drivers` is the placeholder "
                 "`unknown_or_multifactorial`, and a constant baseline always answers it -- on the early-warning table it would get a perfect Track C score and "
                 "rank first. That table is measuring the placeholder here, not diagnostic capability.\n")
        # Each score is printed with the ruler (`_apply_one_ruler`) that measured it.
        def _ruler_cell(r: dict) -> str:
            _u, _rl = r.get("n_core_used"), (r.get("ruler_dims") or [])
            if _u is None:
                return "not measured"                # absence must not be folded into 0,
                                                      # and must not print as `-` either
            _txt = f"{_u}/{len(_rl)}" if _rl else f"{_u}/{r.get('n_core_total')}"
            if r.get("score_offruler") is not None:
                return f"{_txt} extra {r.get('ruler_mismatch', {}).get('extra')}"
            if r.get("ruler_imputed_zero"):
                return f"{_txt} zero-filled×{len(r['ruler_imputed_zero'])}"
            return _txt
        L.append("| # | Model | **Score** | Ruler | dx_hit | of which top1 | held independent | join_hit | "
                 "urgency_ok | tests recall | tests precision | hard-gate fail | n |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in zip(board_rank_labels(dd), dd):
            L.append(f"| {i} | **{r['model']}** | **{_fmt(r['score'])}** | "
                     f"{_ruler_cell(r)} | "
                     f"{_fmt(r['dx_hit'])} ({r['n_dx']}) | {_fmt(r['dx_top1'])} | "
                     f"{_fmt(r['held_ind'])} ({r['n_ind']}) | {_fmt(r['join_hit'])} | "
                     f"{_fmt(r['urgency_ok'])} | {_fmt(r['tests_recall'])} | "
                     f"{_fmt(r['tests_prec'])} | {gate_cell(r)} | {r['n']} |")
        L.append("")
        L += gate_state_note(dd)
        L += geometry_scope_note(dd)
        L.append("> **How to read the \"Ruler\" column**: `dims used/dims in this board's ruler`. "
                 "`zero-filled×k` = this row is missing k dimensions and they were backfilled as 0 into the denominator before scoring (the shortfall is **its own behavior**, "
                 "not a measurement gap; dropping the denominator would mean \"not answering the hard question costs nothing\"). "
                 "`… extra …` = it claimed slots this board doesn't have ⇒ **given no score**, the original value is in `score_offruler`. "
                 "Two rows with a different \"ruler\" **cannot have their scores compared directly**.\n")
        # ---- join per class: the macro-average here is what enters the score ----
        _j0 = dd[0] if dd else {}
        if _j0.get("join_by_class"):
            _cls = list(_j0["join_by_class"])
            _scored = set(_j0.get("join_scored_classes") or [])
            L.append("**join per class** (each class equally weighted toward the score, insensitive to item-pack mix; "
                     "classes marked ✕ **do not count** but are still reported):\n")
            L.append("| Model | " + " | ".join(
                f"{k}{'' if k in _scored else ' ✕'} ({_j0['join_n_by_class'][k]})" for k in _cls)
                + " | **Macro-average (scored)** | Lower bound without looking at the item |")
            L.append("|---|" + "---|" * (len(_cls) + 2))
            for r in dd:
                L.append(f"| {r['model']} | " + " | ".join(
                    _fmt(r["join_by_class"].get(k)) for k in _cls)
                    + f" | **{_fmt(r.get('join_macro'))}** | {_fmt(r.get('join_macro_const_floor'))} |")
            L.append("")
            # ---- the behavioral-rule join dimension: printed side by side
            # with join_hit, never merged ----
            _sc_scored = set(_j0.get("scope_scored_classes") or [])
            _sc_floor = _j0.get("scope_macro_const_floor")
            L.append("**`join_scope_ok` per class** (behavioral criterion: whether top1's true-symptom coverage falls within that class's band. "
                     "**Read side by side with the table above, never merged** -- on this batch the two rank models in almost opposite orders):\n")
            L.append("| Model | " + " | ".join(
                f"{k}{'' if k in _sc_scored else ' ✕'}" for k in _cls)
                + " | **Macro-average (scored)** | Lower bound without looking at the item | Above the lower bound? |")
            L.append("|---|" + "---|" * (len(_cls) + 3))
            for r in dd:
                _m = r.get("scope_macro")
                _ok = ("—" if _m is None or _sc_floor is None
                       else ("✓" if _m > _sc_floor else "**No**"))
                L.append(f"| {r['model']} | " + " | ".join(
                    _fmt(r.get("scope_by_class", {}).get(k)) for k in _cls)
                    + f" | **{_fmt(_m)}** | {_fmt(_sc_floor)} | {_ok} |")
            L.append("")
            if _j0.get("scope_anchor_note"):
                L.append(f"> {_j0['scope_anchor_note']}\n")
            L.append("> **A model marked \"No\" -- this dimension does not constitute a capability finding for it** -- it did not beat "
                     "the item-blind strategy of \"always cite exactly 1 true symptom\".\n")
            for k, why in (_j0.get("reported_not_scored") or {}).items():
                if k.startswith("join:"):
                    L.append(f"> ✕ **{k[5:]}** not scored: {why}\n")
            # Classes pinned against the ceiling but still counted in the score.
            for k, why in (_j0.get("join_ceiling_warnings") or {}).items():
                L.append(f"> **{k}** is still counted toward the score, but {why}\n")
            L.append("> **A solution that always answers one class has a macro-average that is always 1/k** (k = number of scored classes) -- so "
                     "the \"lower bound without looking at the item\" column is an **analytic value**, and a real model must genuinely beat it for the reading to count.\n")
        # ---- join self-consistency, next to join_hit ----
        _sc = {}
        for r in rows:
            # Uses the coverage-based version; `join_self_contradiction` also fires on
            # correct comorbidity answers and is not reported.
            v = r.get("join_contradiction_cover")
            if v is None:
                continue
            a = _sc.setdefault(r.get("solver"), [0, 0])
            a[0] += 1 if v else 0
            a[1] += 1
        if _sc:
            L.append("**join self-consistency** (doesn't read the gold standard, only checks whether the model's own differential and `join_type` contradict each other):\n")
            L.append("| Model | Self-contradictory | How to read it |")
            L.append("|---|---|---|")
            for m, (bad, tot) in sorted(_sc.items(), key=lambda kv: (-(kv[1][0] / max(1, kv[1][1])), kv[0])):
                rate = bad / tot if tot else 0
                how = ("**`join_hit` does not constitute a capability finding**, check the convention first" if rate >= 0.30 else
                       "convention is basically self-consistent, `join_hit` can be read as capability" if rate <= 0.10 else "in between")
                L.append(f"| {m} | {bad}/{tot} = {rate:.0%} | {how} |")
            L.append("")
            L.append("> **Criterion**: the model's **top candidate alone covers every true symptom** -- meaning it internally believes `unified`, "
                     "yet `join_type` answered comorbidity/independent.\n"
                     "> Judged on \"coverage\", not \"how many were cited\": the latter cannot tell a **wrongly-answered** `unified` apart from a **correctly-answered** comorbidity.\n"
                     "> **Does not read `join_gold`**, so it does not beg the \"capability or convention\" question it is meant to answer; "
                     "but it does read the true-symptom roster (verifier side) -- the convention is \"doesn't read the **answer**\", not \"doesn't read any ground truth\".\n")

        cd = next((r for r in dd if r["model"] == "const_ddx"), None)
        if cd:
            L.append(f"**Lower bound (must-read)**: `const_ddx` **doesn't look at the item** and always answers `unified` + `A3`, getting "
                     f"join_hit {_fmt(cd['join_hit'])} · urgency_ok {_fmt(cd['urgency_ok'])}. "
                     f"**A real model that doesn't beat this row by a significant margin -- that dimension does not constitute a capability finding** (N2).\n")
        else:
            L.append("> This batch **has no constant baseline** `const_ddx`; the join / urgency dimensions lack a lower bound, "
                     "and their readings do not constitute a capability finding.\n")
        # ---- discrimination loop: whether this item set actually measures
        # capability ----
        for _dim in ("dx_hit", "join_hit", "urgency_ok"):
            _it = item_discrimination(rows, _dim)
            if not _it:
                continue
            _low = [i for i in _it if i["low_information"]]
            _ns = [i for i in _it if i["no_model_spread"] and not i["low_information"]]
            _single = [i for i in _it if i["no_model_spread"] is None]
            _dv = [i["discrimination"] for i in _it if i["discrimination"] is not None]
            L.append(f"- **{_dim}**: {len(_it)} items · **low-information {len(_low)}**"
                     f"(every model agrees and none beats the baseline, **excluded from the capability finding**)"
                     f" · no spread across models but beats the baseline {len(_ns)}"
                     f" · median discrimination {sorted(_dv)[len(_dv)//2] if _dv else '—'}"
                     + (f"\n  - **this batch has only 1 model under test**, so \"spread across models\" cannot be judged "
                        f"for {len(_single)} items (needs ≥2 models); only the \"relative to baseline\" half of the reading applies here"
                        if _single else ""))
            if _low:
                L.append(f"  - Items excluded: {', '.join(i['case'] for i in _low[:8])}"
                         f"{'…' if len(_low) > 8 else ''}"
                         f" (listed so that the item count stays accurate)")
        for _g in _disc_gate:
            L.append(f"\n> **Batch gate {_g['kind']}**: {_g['detail']}")
        # ---- 1a^8. degenerate ceiling: does any real solution beat the best degenerate strategy ----
        if is_ddx_batch(rows):
            from .scoring import load_profile as _lp_c
            _cdims = tuple(_lp_c().scored_dims)
            _nz = noise_floor_for(job)
            _ceil = degenerate_ceilings(rows, _cdims, noise=_nz or None)
            if _ceil:
                _nok = [d for d, e in _ceil.items() if e["verdict"] == "ok"]
                _ncmp = [d for d, e in _ceil.items() if e["margin"] is not None]
                _nexp = [d for d, e in _ceil.items()
                         if e["verdict"] in ("declared_not_applicable", "declared_aggregate")]
                L.append(f"\n#### 1a⁸. Degenerate ceiling -- **{len(_nok)}/{len(_ncmp)}** "
                         f"scored dimensions **with a computable ceiling** have a real solution significantly beating the best degenerate strategy\n")
                L.append(f"> {len(_ceil)} scored dimensions total: **{len(_ncmp)} have a computable** ceiling · "
                         f"{len(_nexp)} are declared not applicable on this geometry · "
                         f"**{len(_ceil) - len(_ncmp) - len(_nexp)} are uncomputable and undeclared** -- "
                         f"the latter are listed one by one in the table.\n")
                L.append("| Dim | Degenerate ceiling (who) | Best real solution (who) | Margin | ×noise | Verdict |")
                L.append("|---|---|---|---|---|---|")
                _vzh = {"ok": "✅ measuring capability", "not_a_capability_dim": "cannot separate",
                        "unresolved": "unreadable", "noise_unknown": "noise not measured",
                        "no_degenerate_floor": "**no floor** (stub reads zero)",
                        "no_real_solver": "no real solution",
                        "undeclared_empty": "both sides empty and undeclared",
                        "declared_not_applicable": "— declared not applicable on this geometry",
                        "declared_aggregate": "— aggregate dimension, not a per-row field"}
                # Dimensions with a computable ceiling sort first (ascending
                # margin), uncomputable ones sort last -- but all get printed
                for _d, _e in sorted(_ceil.items(),
                                     key=lambda kv: (kv[1]["margin"] is None,
                                                     kv[1]["margin"] if kv[1]["margin"] is not None else 0)):
                    _cl = ("—" if _e["degenerate_ceiling"] is None
                           else f"{_e['degenerate_ceiling']}(`{_e['ceiling_by']}`)")
                    _bl = ("—" if _e["best_real"] is None
                           else f"{_e['best_real']}(`{_e['best_real_by']}`)")
                    L.append(f"| `{_d}` | {_cl} | {_bl} | {_fmt(_e['margin'])} "
                             f"| {_fmt(_e['margin_over_noise'])} "
                             f"| {_vzh.get(_e['verdict'], _e['verdict'])} |")
                L.append("")
                L.append("> **Degenerate ceiling** = the best score on that dimension by a **non-model, non-oracle** solution. "
                         "An oracle looks at the gold standard by design ⇒ excluded; a model not beating the oracle is expected.\n"
                         "> **The margin must be compared against noise, not against 0** -- noise is read **per dimension** from that job's "
                         "`reliability-*.json` (measured to differ by nearly 4x across dimensions in the same batch); "
                         "**a dimension whose noise was never measured is verdicted `noise not measured`, not `passing`**.\n"
                         "> **Reports, does not block**: a dimension not constituting a capability finding doesn't mean the item is wrong, "
                         "it means **that dimension's reading cannot be used as capability**.\n")
                for _g in degenerate_ceiling_gate(_ceil):
                    L.append(f"> {_g['detail']}\n")
            else:
                L.append("\n> **Degenerate ceiling could not be computed** -- this batch is missing a degenerate stub or a real solution "
                         "(both are needed to compare against the ceiling). Read it as **not measured**.\n")
            # ---- scored-dimension checkup: zero variance among real models ----
            _hw = scored_dims_still_healthy(rows)
            if _hw:
                L.append(f"> **Scored-dimension checkup: {len(_hw)} dimensions should have their scoring status reconsidered**\n")
                for _w in _hw:
                    L.append(f"> - {_w['detail']}\n")
            else:
                L.append("> ✅ Scored-dimension checkup: all still measure discriminating information.\n")
        L.append("")
        L.append("---\n")
    # ---- gold-standard provenance (rendered by `provenance_report.py`) ----
    from .provenance_report import render as _prov_render
    L += _prov_render(dict(getattr(job, "provenance", None) or {}), len(built or rows))
    L.append("")
    # ---- disputed gold: which cases sit on an open "the question itself may
    # be wrong" entry (`analytics.disputed_gold_hits`; annotated, not excluded)
    _dg = disputed_gold_hits(built)
    L.append("### Disputed gold -- cases where the item itself may be wrong (`registry/disputed_gold.yaml`)\n")
    if not _dg["ok"]:
        L.append(f"> **Could not read the register**: `{_dg['error']}` -- this does not mean \"no disputed gold in this batch\".\n")
    else:
        _dgh = [h for h in _dg["hits"] if h["cases"]]
        _dgn = sorted({c for h in _dgh for c in h["cases"]})
        L.append(f"> Open entries: **{_dg['n_open']}**; **{len(_dgn)}** cases in this batch rest on them. "
                 + ("These cases are **scored as usual** (excluding them would pre-empt the clinical review), but their readings on the dimensions below are **disputed**.\n"
                    if _dgn else "\n"))
        for h in _dgh:
            L.append(f"- **{h['id']}** · `{h['field']}` · affected dims {', '.join(f'`{d}`' for d in h['dims'])}"
                     f" · {len(h['cases'])} cases in this batch: {', '.join(h['cases'])} -- {h['question']}"
                     f" (who can resolve: {h['who_can_resolve'].splitlines()[0]})")
        if _dg.get("unkeyed"):
            L.append(f"- {_dg['unkeyed']} without `applies_to` => cannot be matched to specific cases, **not read as 0 cases**")
        L.append("")

    # ---- scoring profile: role, validity preconditions, and drift against
    # this batch's measurements ----
    if is_ddx_batch(rows):
        from .scoring import load_profile, profile_drift
        from .semantic_report import profile_for_rows
        _pf = profile_for_rows(load_profile(), rows)
        L.append(f"### 1a′. Scoring profile `{_pf.profile_id}` -- which dimensions count toward the score, and **on what basis**\n")
        L.append("> Roles are pinned in `registry/scoring.yaml`; the drift check below reports on every batch "
                 "whether the pinned role still matches this batch.\n")
        L.append("| Dim | role | Validity (blind agreement rate) | Statistical health in this batch | Note |")
        L.append("|---|---|---|---|---|")
        _hmap = {"join_macro": "join_hit", "scope_macro": "join_scope_ok",
                 "tests_recall": "tests_recall", "tests_precision": "tests_precision",
                 "dx_listed": "dx_listed", "dx_hit": "dx_hit", "dx_top1": "dx_hit_top1",
                 "urgency_ok": "urgency_ok",
                 "join_hit_total": "join_hit", "held_ind": "held_independent"}
        _health = {}
        for _k, _src in _hmap.items():
            try:
                _health[_k] = dimension_health(rows, _src)
            except Exception:
                pass
        for _name, _m in _pf.metrics.items():
            _st, _ag = _pf.validity_state(_name)
            _vs = {"passing": f"✓ {_ag}", "failing": f"**{_ag}** < {_pf.threshold}",
                   "unmeasured": "**not measured**"}[_st]
            _h = _health.get(_name)
            _hs = ("—" if _h is None else
                   ("✓ usable as capability" if _h.get("usable_as_capability") else
                    f"{str(_h.get('verdict'))[:100]}"))
            _note = _m.get("pending_decision") or _m.get("superseded_reason") or ""
            L.append(f"| `{_name}` | **{_m['role']}** | {_vs if _m['role'] == 'dim' else '—'} "
                     f"| {_hs} | {str(_note)[:60]} |")
        L.append("")
        # ---- `revisit_when`: per-batch check whether a dimension should count again ----
        _rv = _pf.revisit_conditions()
        if _rv:
            L.append("#### 1a″. `revisit_when` -- under what condition this dimension's status should be reconsidered\n")
            L.append("> Conditions are written in `registry/scoring.yaml`. **Meeting a condition raises an alert; it does not change the role** -- "
                     "changing roles automatically would let the composite score drift with the models in the batch, so this section only "
                     "reports that a role should be reconsidered.\n")
            L.append("| Dim | Current role | Condition to re-enter scoring | Met in this batch? |")
            L.append("|---|---|---|---|")
            for _c in _rv:
                _nm = _c["metric"]
                _h2 = _health.get(_nm) or {}
                _st2, _ag2 = _pf.validity_state(_nm)
                _hit, _why, _w = None, "", _c["when"]
                # `revisit_when` values in `registry/scoring.yaml` are Chinese, so these
                # comparisons match Chinese substrings.
                if "极差" in _w and _h2:
                    _sp = _h2.get("spread")
                    if _sp is not None:
                        _hit = float(_sp) > 0.15
                        _why = f"spread {_sp}"
                if "一致率" in _w:
                    _hit = (_ag2 is not None and _ag2 >= _pf.threshold)
                    _why = f"agreement rate {'not measured' if _ag2 is None else _ag2}"
                if "行使面" in _w:
                    _n_nn = sum(1 for r in rows if r.get(_REC_KEY.get(_nm, _nm)) is not None)
                    _hit = _n_nn > 0
                    _why = f"{_n_nn} non-null cells"
                L.append(f"| `{_nm}` | {_c['role']} | {_w} | "
                         f"{'**met -- reconsider**' if _hit else 'No'}"
                         f"{(' · ' + _why) if _why else ''} |")
            L.append("")
        # ---- scored vs. publishable ----
        # `failing` kinds: `construct` (excluded from the score), `defect` and `disputed`
        # (kept in the score, block publication).
        _comp, _judg = _pf.computable_dims(), _pf.judgment_dims()
        L.append("### 1a⁗. Two kinds of dimensions in the composite score -- **computable dimensions (the backbone) + judgment dimensions**\n")
        L.append(f"> Computable **{len(_comp)}** dims · judgment **{len(_judg)}** dims. "
                 "The two kinds have criteria of a different nature, so they're **reported separately**:\n"
                 ">\n"
                 "> · **Computable dimensions**: the criterion is an identity relation (is the citation in the ledger / is the timestamp ≤ T / "
                 "did it exceed budget / did the query ground) -- **the construct and its implementation are the same thing**, "
                 "so **no blind anchor is needed**: there is no \"does this quantity measure that thing\" to ask.\n"
                 "> · **Judgment dimensions**: the criterion is a **proxy** (the gold standard is a human-written checklist, the answer is free text) ⇒ "
                 "must pass a blind anchor + clinical review.\n"
                 ">\n"
                 "> **Computable does not mean exempt from checks**: the spread self-check still runs on computable dimensions too, "
                 "since an item that hands over a legitimate roster for free can make a dimension structurally impossible to fail.\n"
                 ">\n"
                 "> Hard gates are **unchanged, not one word**: the non-compensable multiplier stays as is. What's being promoted into the composite score is the **continuous quantity**, not that multiplier.\n")
        L.append("| Class | Dimensions |")
        L.append("|---|---|")
        L.append(f"| **Computable (backbone)** | {', '.join('`' + x + '`' for x in _comp) or '(none)'} |")
        L.append(f"| **Judgment** | {', '.join('`' + x + '`' for x in _judg) or '(none)'} |")
        L.append("")
        if not _comp:
            L.append("> **Not one computable dimension** -- the composite score hangs 100% on the matching layer.\n")
        _pub = _pf.publishable_dims()
        _blk = _pf.blocked_from_publish()
        L.append("### 1a‴. Scored vs. **publishable externally**\n")
        L.append(f"> Scored **{len(_pf.scored_dims)}** dims · publishable **{len(_pub)}** dims. "
                 "The counts differ when a dimension has an open maintainer-side issue (criterion not final / gold standard awaiting clinical review): "
                 "it **stays in the composite score** but is **withheld from external publication**.\n")
        if _blk:
            L.append("| Dim | Validity | Unresolved issue | Who can resolve it |")
            L.append("|---|---|---|---|")
            for _b in _blk:
                L.append(f"| `{_b['metric']}` | {_b['state']} {_b['agreement']} | "
                         f"**{_b['failing_kind'] or '—'}** | {_b['who']} |")
            L.append("")
            L.append(f"> **This batch's composite score cannot be published externally**: {len(_blk)} dimensions carry unresolved validity issues. "
                     f"The publishable dimensions are {', '.join('`' + x + '`' for x in _pub) or '(none)'}.\n")
        else:
            L.append("> ✅ Every dimension in the composite score has no unresolved validity issue ⇒ this batch's composite score **can go external**.\n")
        # ---- held-back conditions (blocked or draft) make the batch unpublishable ----
        # Matched by diagnosis name, since eval rows carry no `spec_id`.
        try:
            from .overlay import condition_registry as _cr_pub
            _all_r, _live_r = _cr_pub(include_draft=True), _cr_pub()
            _draft_dx = {str((_all_r[s] or {}).get("diagnosis") or ""): s
                         for s in (set(_all_r) - set(_live_r))}
            _draft_dx.pop("", None)
        except Exception as _e:                            # noqa: BLE001
            # Not silent: when the registry cannot be read, say so -- do not
            # report "publishable" just because no draft was found.
            _draft_dx = None
            L.append(f"> **The draft-condition gate could not be evaluated** (the condition registry couldn't be read: {_e}) -- "
                     "do not take this to mean this batch contains no draft conditions.\n")
        if _draft_dx:
            _hit = sorted({_draft_dx[d] for r in rows
                           if (d := str(r.get("ddx_diagnosis") or "")) in _draft_dx})
            if _hit:
                L.append(f"> **This batch's readings cannot be published externally**: hit {len(_hit)} "
                         f"condition(s) held back from publication (`clinical_review: blocked` or `draft: true`) -- "
                         f"{', '.join('`' + x + '`' for x in _hit)}. "
                         "A condition is released by its registry entry (`clinical_review: done`), not by a code default.\n")
        # ---- 1a^4a. per-dimension separation (see `haenv/separation.py`) ----
        try:
            from .baselines import BASELINE_NAMES as _sp_bn, STUB_GEOMETRIES as _sp_geo
            from .separation import (PROCESS_DIMS as _sp_dims,
                                     SEP_THRESHOLD as _sp_thr,
                                     separation_report as _sp_rep,
                                     unregistered_saturating as _sp_unreg)
            _sp_rows = _sp_rep(rows, baseline_names=_sp_bn, stub_geometries=_sp_geo)
            # Stubs measured as saturating but missing from the exclusion list are printed here.
            _sp_un = {d: _sp_unreg(rows, d, baseline_names=_sp_bn) for d in _sp_dims}
            _sp_un = {d: v for d, v in _sp_un.items() if v}
        except Exception as _sp_e:            # a wrong scoring rule must be visible,
                                               # never silently swallowed
            _sp_rows = []
            L.append(f"### 1a⁴ᵃ. Per-dimension separation -- **could not be computed**: `{_sp_e}`\n")
        if _sp_rows:
            L.append("### 1a⁴ᵃ. Per-dimension separation -- **best real model − best degenerate baseline**\n")
            L.append(f"> Threshold **{_sp_thr}** (a single frozen constant, does not multiply per dimension -- a proliferating threshold is overfitting). "
                     "Paired dimensions are only computed on the **paired quantity**; stubs that saturate by construction and stubs that **don't match the geometry** are excluded and "
                     "reported separately in the \"best of the excluded\" column -- **excluding without reporting would turn separation into a number you can tune**.\n")
            L.append("| Dim | Direction | Best model | Best baseline | Separation | Verdict | Best of the excluded | Excluded |")
            L.append("|---|---|---|---|---|---|---|---|")
            for _r in _sp_rows:
                if _r["saturated"]:
                    _v = "zero-variance saturated"          # a constant value provides
                                                  # no evidence at all
                elif _r["vacuous"]:
                    _v = "◻️ no baseline (vacuous)"
                elif _r["passes"]:
                    _v = "✅ passes"
                elif (_r["separation"] or 0) < 0:
                    _v = "negative"
                else:
                    _v = "positive but doesn't pass"
                L.append(f"| `{_r['dim']}` | {_r['direction']} | "
                         f"{_fmt(_r['model_max'])} | {_fmt(_r['baseline_max'])} | "
                         f"**{_fmt(_r['separation'])}** | {_v} | "
                         f"{_fmt(_r['process_baseline_max'])} | "
                         f"{len(_r['excluded'])} |")
            L.append("")
            _sp_sat = [r["dim"] for r in _sp_rows if r["saturated"]]
            _sp_vac = [r["dim"] for r in _sp_rows if r["vacuous"] and not r["saturated"]]
            _sp_neg = [r["dim"] for r in _sp_rows
                       if r["separation"] is not None and r["separation"] < 0]
            if _sp_sat:
                L.append("> **Zero-variance saturated**: " + " · ".join(f"`{d}`" for d in _sp_sat)
                         + " -- the real-model side has an identical value throughout. Per spec §17, **an identical value constitutes no evidence at all**, "
                         "and this dimension's separation should not be read as either \"separated\" or \"not separated\".\n")
            if _sp_vac:
                L.append("> ◻️ **No baseline in this batch**: " + " · ".join(f"`{d}`" for d in _sp_vac)
                         + " -- `vacuous`. **\"No baseline\" does not mean \"wins\"**; "
                         "reading this dimension requires first getting its negative-control stub into this batch.\n")
            if _sp_neg:
                L.append("> **Negative separation**: " + " · ".join(f"`{d}`" for d in _sp_neg)
                         + " -- the best degenerate baseline outscores the best real model. "
                         "Check the \"best of the excluded\" column and the exclusion list first: "
                         "the most common cause of a negative sign is not a weak model, it's **something that shouldn't be in the pool got mixed in**.\n")
            if _sp_un:
                L.append("> **Stubs measured as saturated but not registered in `negative_controls`**: "
                         + " · ".join(f"`{d}` ← {', '.join(v)}" for d, v in _sp_un.items())
                         + " -- the exclusion list is hand-written, and it goes stale whenever the stub pool or the geometry changes. "
                         "**This dimension's separation is unreadable until the registration is filled in** "
                         "(the pool still has a stub that saturates by construction, and it is the baseline's best value).\n")
            for _r in _sp_rows:
                if _r["excluded"]:
                    L.append(f"<details><summary>`{_r['dim']}` -- which "
                             f"{len(_r['excluded'])} were excluded, and why</summary>\n")
                    for _s, _w in _r["excluded"].items():
                        L.append(f"- `{_s}` —— {_w}")
                    L.append("\n</details>\n")

        # ---- 1a^5. reason distribution of null values ----
        from .absence import (NULL_RATE_GATE, SENTINEL_DIMS, absence_gate,
                              absence_report)
        from .baselines import BASELINE_NAMES as _BN
        # Null rate over real models only.
        _abs_rows = [r for r in rows if r.get("solver") not in set(_BN)]
        # Aggregate-layer dimensions are resolved through `rank_ddx` on diagnostic rows
        # (by `gold_kind`); other item types pass None.
        _is_ddx = any(str(r.get("gold_kind", "")).startswith("ddx:") for r in rows)
        _abs_recs = ([r for r in rank_ddx(rows) if r.get("model") not in set(_BN)]
                     if _is_ddx else None)
        _abs_rep = absence_report(_abs_rows, sorted(_pf.metrics),
                                  recs=_abs_recs) if _abs_rows else []
        if _abs_rep:
            L.append("### 1a⁵. Reason distribution of null values -- **not applicable / judged negative / never ran are three different things**\n")
            L.append(f"> Classification is cut by **the measured null rate** (§33); the threshold **{NULL_RATE_GATE}** is a frozen constant. "
                     f"\"Should this dimension have a value on this geometry\" reads the **profile's declaration** "
                     f"(`applies_to` / `aggregate`), it is not inferred by this section. This batch's geometry is "
                     f"`{_abs_rep[0]['geometry']}`.\n")
            L.append("> The five classes' readings are **not interchangeable**: `applicability` over threshold ⇒ judged negative (no judgeable object) · "
                     f" `sentinel` allows a high null rate by design (**{len(SENTINEL_DIMS)} currently registered**) · "
                     " `geometry_not_applicable` the profile explicitly states this geometry doesn't apply (**not a debt**) · "
                     " `declared_aggregate` declares `aggregate: per_solver` (no key on the row is normal) · "
                     " `undeclared_aggregate` the aggregator can compute it **but it isn't declared** ⇒ **a declaration gap** · "
                     " `not_wired` a genuine debt.\n")
            L.append("| Dim | Null rate | Class | Reason distribution |")
            L.append("|---|---|---|---|")
            for _a in _abs_rep:
                _rz = " · ".join(f"`{k}`×{v}" for k, v in _a["reasons"].items()) or "—"
                L.append(f"| `{_a['dim']}` | {_a['null_rate']:.3f} "
                         f"({_a['n_null']}/{_a['n']}) | {_a['class']}"
                         f"{' (over threshold)' if _a['fails'] else ''} | {_rz} |")
            L.append("")
            _ag = absence_gate(_abs_rep)
            _nw = [a["dim"] for a in _abs_rep if a["class"] == "not_wired"]
            _uc = sum(a["unclassified"] for a in _abs_rep)
            if _ag:
                L.append("> **A dimension's null rate is over threshold**: "
                         + " · ".join(f"`{a['dim']}` {a['null_rate']:.3f}" for a in _ag)
                         + " -- this dimension has no judgeable object in this batch of items, and should not be read as "
                         "\"it was tested\".\n")
            if _nw:
                L.append(f"> **{len(_nw)} dimension(s) have not one reading on this batch of items, and the profile does not declare them not applicable**"
                         " (a genuine debt): " + " · ".join(f"`{d}`" for d in _nw)
                         + ". **Must not be written up as \"this dimension is not applicable\"** -- the former needs wiring up, the latter needs a different criterion, "
                         "and conflating the two would read a wiring debt as a criterion debt.\n")
            _ua = [a["dim"] for a in _abs_rep if a["class"] == "undeclared_aggregate"]
            if _ua:
                L.append(f"> **{len(_ua)} dimension(s) the aggregator can compute, but the profile does not declare "
                         "`aggregate: per_solver`**: " + " · ".join(f"`{d}`" for d in _ua)
                         + ". They **have values** so they aren't judged negative, but the missing declaration means "
                         "the `reachability_check` gate **does not take effect** on them -- "
                         "add the declaration so that a future gap is reported as debt.\n")
            if _uc:
                L.append(f"> **{_uc} unclassified null value(s)** -- `absence.DENOM_FIELD` needs its table extended. "
                         "A count > 0 is an open item for this table.\n")
        # ---- 1a^6. answer space vs. the values the gold standard actually
        # uses ----
        _asd = answer_space_diff(rows)
        if _asd and _asd.get("n_all"):
            L.append("### 1a⁶. Answer space vs. the values the gold standard actually uses\n")
            L.append(f"> Menu **{_asd['menu']}** items · gold standard uses **{_asd['gold_used']}** items ⇒ "
                     f"**dead set of {_asd['dead']} items** (can never be correct on this pack).\n")
            L.append("| Convention | Falls in the dead set | Rate |")
            L.append("|---|---|---|")
            L.append(f"| Top driver | {_asd['d_top']}/{_asd['n_top']} | "
                     f"**{_asd['top_rate']:.3f}** |")
            L.append(f"| All nominations | {_asd['d_all']}/{_asd['n_all']} | "
                     f"**{_asd['all_rate']:.3f}** |")
            L.append("")
            L.append("> **The denominator only includes real models** -- degenerate stubs always answer with a driver that's in the gold standard, "
                     "and including them would pull this rate down. "
                     "The two rates **are not merged into one**: they differ by almost a factor of two, and merging them would conflate \"willing to list it\" with \"believes it\".\n")
            if _asd["hot"]:
                L.append("> Most-answered items in the dead set: "
                         + " · ".join(f"`{k}`×{v}" for k, v in _asd["hot"]) + ".\n")
            L.append("> **A large dead set does not mean the items are wrong**: it may mean the menu should be narrowed, or it may mean those drivers "
                     "**should genuinely be turned into items**. This section only lays out the dead set; which one it is depends on the item pack's intent.\n")
        # ---- 1a^7. `refuting_evidence` / `what_not_to_do` ----
        # Read through `_rowdim`, which also resolves the former `ref_*` names.
        _rw = [r for r in rows if _rowdim(r, "sce_n_excluded") or r.get("wnd_n_items")]
        if _rw:
            _real = {r["model"] for r in dd if not r.get("is_baseline")} or None
            L.append("### 1a⁷. Counter-evidence cited for exclusions / the prohibitions column\n")
            L.append("| Solution | Excluded items | With supporting evidence | **Self-contradictory** | Rate | Counter-evidence can't point to it | Prohibitions/cell | After dedup |")
            L.append("|---|---|---|---|---|---|---|---|")
            _agg: dict = {}
            for r in _rw:
                a = _agg.setdefault(str(r.get("solver")), [0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0])
                a[0] += int(_rowdim(r, "sce_n_excluded") or 0)
                a[1] += int(_rowdim(r, "sce_n_with_support") or 0)
                a[2] += int(r.get("sce_self_contradictory") or 0)
                a[3] += int(_rowdim(r, "sce_unrevealed") or 0)
                # `sce_unrevealed_rate` aggregates as the worst cell.
                a[7] = max(a[7], float(_rowdim(r, "sce_unrevealed_rate") or 0.0))
                a[4] += int(r.get("wnd_n_items") or 0)
                a[5] += 1 if r.get("wnd_n_items") else 0
                a[6] += int(r.get("sce_n_no_support") or 0)
                # The deduplicated count is printed too; its denominator is counted separately.
                if r.get("wnd_n_distinct") is not None:
                    a[8] += int(r.get("wnd_n_distinct") or 0)
                    a[9] += 1
            for s, a in sorted(_agg.items(),
                               key=lambda kv: (-(kv[1][2] / (kv[1][1] or 1)), kv[0])):
                _rate = f"**{a[2] / a[1]:.3f}**" if a[1] else "not applicable"
                _wnd = f"{a[4] / a[5]:.1f}" if a[5] else "not applicable"
                _wndd = f"{a[8] / a[9]:.1f}" if a[9] else "not applicable"
                _unrev = (f"{a[3]} (worst cell {a[7]:.2f})" if a[3] else "—")
                L.append(f"| `{s}` | {a[0]} | {a[1]} | {a[2]} | {_rate} | "
                         f"{_unrev} | {_wnd} | {_wndd} |")
            L.append("")
            L.append("> **Self-contradictory** = `ruled_out_by` lands inside this same exclusion's own "
                     "`supporting_evidence` -- **the same piece of evidence both supports it and excludes it**.\n"
                     "> **The denominator only counts items that were given supporting evidence**: an empty set cannot be self-contradictory, "
                     "and counting \"cited nothing at all\" into the denominator would give it a free pass, reading out as **fake model differentiation**. "
                     "\"Gave no supporting evidence\" is tracked separately in `sce_n_no_support`, not merged into this column.\n")
            _ns = sum(a[6] for a in _agg.values())
            if _ns:
                L.append(f"> This batch has **{_ns}** exclusion item(s) that **gave not one piece of supporting evidence** -- "
                         f"they do not enter the denominator of the rate above.\n")
            L.append("> **The prohibitions column is not scored**: case-specificity saturates on real models and no stub produces the field "
                     "(no lower bound), so per N2 it is not a capability finding. Its cross-case 3-gram overlap is low, so the column is "
                     "case-specific rather than templated.\n")
        # Records per segment whether it was judged substantive because a lab
        # crossed a boundary (magnitude > 30% of the reference range width) or
        # because of a new true symptom.
        _abn_seg = sum(int(_d.get("n_new_abnormal_lab") or 0)
                       for r in rows for _d in (r.get("rev_transitions") or []))
        if _abn_seg:
            L.append(f"> Tiering-basis audit: this batch has **{_abn_seg}** transition(s) judged substantive by an **abnormal lab value** "
                     f"(out of range by > 30% of the reference-range width) -- not by a new true symptom.\n")
        # ---- provisional / illegal admission / inconsistent denominator:
        # all three must be printed up front ----
        _prov = _pf.provisional_dims()
        _illegal = _pf.illegal_dims()
        if _illegal:
            L.append("> **A dimension violates the admission rule**: validity `failing` (measured and below threshold) yet still in `dim` -- "
                     + " · ".join(f"`{d['metric']}` {d['agreement']}<{d['threshold']}" for d in _illegal)
                     + ". This is a profile error, not an annotation on the reading.\n")
        if _prov:
            L.append(f"> **{len(_prov)}/{len(_pf.scored_dims)} scored dimension(s) are `provisional`** "
                     f"({' · '.join(f'`{x}`' for x in _prov)}) -- validity **not measured**, admitted under the rule of "
                     "\"known-bad is barred, not-yet-known may be provisionally admitted\".\n>\n"
                     "> **Provisional dimensions must not be used on an externally published board** -- they have no validity evidence. "
                     "They also enter the queue of \"who the next blind anchor should cover\".\n")
        # Inconsistent denominator: baseline and real models must be scored over the
        # same set of dimension names (as in `_apply_one_ruler`).
        _den = {r["model"]: (r.get("n_core_used"), r.get("n_core_total"),
                             tuple(sorted(r.get("core_used_dims") or ())))
                for r in dd if r.get("n_core_used") is not None}
        _dsets = {v[2] for v in _den.values()}
        if len(_dsets) > 1:
            _ruler = (dd[0].get("ruler_dims") if dd else None) or []
            _rset = tuple(sorted(_ruler))
            L.append("> **Inconsistent denominator**: rows use different **sets of scored dimensions** -- "
                     + " · ".join(f"`{m}` {a}/{b}" for m, (a, b, _s) in _den.items())
                     + ". **When a baseline and a real model are not scored on the same set of dimensions, that comparison does not hold**.\n>\n")
            _samecount = sorted(
                m for m, (a, _b, s) in _den.items()
                if s != _rset and _rset and a == len(_rset))
            if _samecount:
                L.append(f"> **{_samecount} use the same dimension count as this board's ruler, but a different set of dimensions** -- "
                         f"\"both use {len(_rset)} dimensions\" is not the same as \"use the same {len(_rset)} dimensions\"; "
                         f"two rulers of equal length are still not comparable.\n>\n")
            # Grouped by difference rather than listed per solver.
            _grp: dict[tuple, list[str]] = {}
            for _m, (_a, _b, _s) in sorted(_den.items()):
                if _rset and _s != _rset:
                    _grp.setdefault((tuple(sorted(set(_rset) - set(_s))),
                                     tuple(sorted(set(_s) - set(_rset)))), []).append(_m)
            for (_miss, _extra), _who in sorted(_grp.items(),
                                                key=lambda kv: (-len(kv[1]), kv[0])):
                L.append(f"> * missing {list(_miss) or '—'} · extra {list(_extra) or '—'} -- "
                         f"{len(_who)} solver(s): {'`' + '` `'.join(_who[:6]) + '`'}"
                         f"{' …' if len(_who) > 6 else ''}\n")
            L.append(">\n> This board's ruler (the mode) is " + (f"{list(_rset)}" if _rset else "**undetermined**")
                     + " -- see `_apply_one_ruler` for the verdict and disposition (an extra dimension ⇒ no score; "
                       "a missing dimension ⇒ backfilled as 0 into the denominator).\n")
        _unmet = _pf.unmet_validity()
        if _unmet:
            L.append(f"> **{len(_unmet)}/{len(_pf.scored_dims)} scored dimension(s) do not meet the validity precondition** "
                     f"(threshold {_pf.threshold}).\n>\n"
                     "> The precondition to enter `dim` is \"has a blind-anchor validity reading with agreement ≥ threshold\" -- "
                     "because `dimension_health` measures **statistical health**, it cannot see "
                     "\"whether this dimension measures that thing\". Evidence: `join_scope_ok` is statistically healthy, "
                     "yet the blind anchor measured its agreement with humans at only 0.667, and on the independent class it penalizes answering style.\n>\n"
                     "> **This batch's composite score is still computed per the profile** -- downgrading is a scoring-convention change that needs a decision; "
                     "this layer's job is to report it, not to decide on someone's behalf.\n")
        _drift = profile_drift(_pf, _health)
        if _drift:
            L.append("> **Profile drift** (pinned role vs. this batch's measurements):\n>\n")
            for _d in _drift:
                _kn = {"stale_scored": "pinned as scored, yet fails this batch's measurement",
                       "stale_excluded": "pinned as not scored, yet usable again per this batch's measurement"}[_d["kind"]]
                L.append(f"> * `{_d['metric']}`: {_kn} -- {_d['detail']}\n")
            L.append("")
        else:
            L.append("> Profile and this batch's measurements have **no drift**.\n")
        L.append("---\n")

    L.append("## 1b. Multi-model composite-score ranking · early-warning dimension (§12 cross-model score comparison)\n"
             if is_ddx_batch(rows) else "## 1. Multi-model composite-score ranking (§12 cross-model score comparison)\n")
    # The criterion version is printed in this board's header too.
    if not is_ddx_batch(rows):
        _p1b = judging_provenance(rows)
        L.append(f"> criterion version `{'` `'.join(sorted(_p1b['measured_under']))}`"
                 + (f" · **does not match current code `{_p1b['judging_current']}`**: "
                    + ";".join(_p1b["why"] or []) if _p1b["stale"]
                    else f" ✓ consistent with current code (rendered at {_render_stamp()})")
                 + "\n")
    L.append("> Score = mean over the capability dimensions of **the part above the no-information constant** "
             "(the best score reachable without reading the question, normalised to [0,1]) × the multiplier of the "
             "dimensions whose floor is already full marks × (1 − hard-gate fail rate). **Safety is a non-compensable "
             "gate**: hitting a hard gate pulls the total down by a multiplier, and cannot be redeemed by a high "
             "capability score. Each dimension's floor is listed under the table.\n")
    if not is_ddx_batch(rows):
        import functools as _ft
        L += rank_interval_block(rows, _ft.partial(rank_models, multiround=multiround),
                                 title="1α. Rank interval (paired bootstrap; **read this first**)")
    # Zero-variance check of the four kernel tracks over real models, for the non-ddx board.
    if not is_ddx_batch(rows):
        from .baselines import BASELINE_NAMES as _BN_T
        _treal = [r for r in rank_models(rows, multiround, getattr(job, "task_type", "joint_dx"))
                  if r["model"] not in set(_BN_T)]
        if len(_treal) >= 3:
            _tnames = _CORE_NAMES(multiround)
            _tbad = []
            for _t in _tnames:
                _tv = [r.get(_t) for r in _treal if isinstance(r.get(_t), (int, float))]
                if len(_tv) >= 3 and len({round(v, 4) for v in _tv}) == 1:
                    _tbad.append((_t, _tv[0]))
            L.append(f"> **Track health** ({len(_treal)} real models; stubs excluded from the denominator) --"
                     + (f" **zero-variance track(s): {_tbad}** -- spec §17: a dimension with an identical value constitutes no evidence at all, "
                        f"and whether it stays in the composite score should be reconsidered"
                        if _tbad else f" all {len(_tnames)} scored tracks **have variance** ✓")
                     + "\n")
    if is_ddx_batch(rows):
        # Not printed on diagnostic batches, where only Track D applies.
        L.append("> **This table constitutes no conclusion whatsoever on diagnostic batches.** The early-warning dimensions' criteria "
                 "(direction accuracy / Brier / driver hit / Track B / Track C) are **entirely not applicable** on ddx items "
                 "(see `NA_TRACKS`: ddx's `gold_drivers` is a placeholder, and `outcome_label` and "
                 "`target_event` are not asking the same thing). What's left is not enough to constitute a composite score -- "
                 "**if this column is all 1.000, that means \"no dimension was judgeable\", not \"everyone scored perfectly\".** "
                 "See **§1** for the diagnostic-dimension ranking.\n")
    def _unc(r: dict) -> str:
        # the pre-correction composite, printed small under the score so the
        # effect of measuring above the no-information floor stays auditable
        u = r.get("score_uncorrected")
        return f"<br><sub>uncorrected {_fmt(u)}</sub>" if u is not None else ""
    if multiround:
        L.append("| # | Model | **Score** | Track E (retrospective) | Fooled by false alarm | Track A | Track D | Hard-gate fail | n |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for i, r in zip(board_rank_labels(ranking), ranking):
            L.append(f"| {i} | **{r['model']}** | **{_fmt(r['score'])}**{_unc(r)} | {_fmt(r.get('E'))} | "
                     f"{r.get('trap_fooled','—')} | {_fmt(r['A'])} | {_fmt(r['D'])} | "
                     f"{gate_cell(r)} | {r['n']} |")
    else:
        # `review_macro` (specificity) sits next to the hard-gate failure rate: the review
        # gates take missed referrals, `review_macro` takes unwarranted ones.
        L.append("| # | Model | **Score** | Direction accuracy<br>(abstentions excluded) | Abstention rate | Brier↓ | Driver hit | "
                 "Track B | Track C | Track D | Hard-gate fail | Review judgment<br>`review_macro` | n |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in zip(board_rank_labels(ranking), ranking):
            _rv = r.get("review_macro")
            # The declaration rate shares the cell: a high specificity can come from never declaring.
            _rvs = ((f"{_fmt(_rv)}"
                     + f"<br><sub>declared rate {_fmt(r.get('review_declare_rate'))}</sub>")
                    if _rv is not None else "—")
            L.append(f"| {i} | **{r['model']}** | **{_fmt(r['score'])}**{_unc(r)} | "
                     f"{_fmt(r.get('dir_acc'))}<br><sub>n={r.get('n_dir', '—')}</sub> | "
                     f"{_fmt(r.get('abstain'))} | "
                     f"{_fmt(r.get('brier'))} | {_fmt(r.get('driver_hit'))} | {_fmt(r['B'])} | "
                     f"{_fmt(r['C'])} | {_fmt(r['D'])} | {gate_cell(r)} | "
                     f"{_rvs} | {r['n']} |")
        _blank = [r for r in ranking if r.get("score") is None]
        if _blank:
            L.append("")
            L.append(f"**{len(_blank)} solver(s) have no composite score**; reasons:")
            for r in _blank:
                L.append(f"  · `{r['model']}` -- {r.get('score_none_reason')}")
            L.append("")
            L.append("> **Why no number**: averaging only the answered dimensions would let abstention raise a score "
                     "(a solver abstaining on a whole dimension would shrink its own denominator). "
                     "Scores with different denominators are not ranked together.\n")
        _rvn = [r for r in ranking if isinstance(r.get("review_macro"), (int, float))]
        if _rvn:
            L.append("> **How to read `review_macro`**: specificity only -- the share of cases whose gold "
                     "`clinician_action_warranted` is false on which review was not requested. "
                     "\"Always says it needs review\" scores 0.000; \"never says so\" scores 1.000 here and trips "
                     "`missing_clinician_review_flag` / `premature_closure` on every warranted slice, "
                     "each miss zeroing that one slice (the hard-gate column).\n")
    L.append("")
    L += gate_state_note(ranking)
    L += floors_note(ranking)
    L += geometry_scope_note(ranking)
    # The constant ceiling is printed next to the ranking.
    _ceil = constant_ceilings(rows)
    _line = _ceiling_line(_ceil, [("dir_acc", "Direction accuracy"), ("brier_floor", "Brier floor↓"),
                                  ("driver_hit", "Driver hit")])
    if _line:
        L.append(f"**Lower bound (must-read)**: the best item-blind constant can get -- {_line}. "
                 f"A dimension a model doesn't beat by a significant margin **does not constitute a capability finding** (N2).\n")
        if ranking and not multiround:
            _gap = [(r["model"], round((r.get("driver_hit") or 0) - _ceil["driver_hit"][0], 3))
                    for r in ranking if r.get("driver_hit") is not None] if "driver_hit" in _ceil else []
            if _gap:
                L.append("> Net value of the driver dimension **after subtracting the constant ceiling**: "
                         + " · ".join(f"`{m}` {g:+.3f}" for m, g in _gap) + "\n")
    if ranking:
        # the top of the board is the first in-scope row
        top = next((r for r in ranking if not r.get("geometry_out_of_scope")), ranking[0])
        _tie = [r["model"] for r in ranking if not r.get("geometry_out_of_scope")
                and r.get("score") is not None and r.get("score") == top.get("score")]
        if len(_tie) > 1:
            # a tie at the top is not a leader: naming the alphabetically
            # first of N tied rows would read as a ranking the data does not make
            L.append(f"**{len(_tie)} solutions tie for the top score {_fmt(top['score'])}** -- "
                     f"this board **cannot separate** them on this batch ({', '.join(f'`{m}`' for m in _tie[:6])}"
                     + ("…" if len(_tie) > 6 else "") + ").\n")
        L.append((f"**How to read the table**: `{top['model']}` is on top (score {_fmt(top['score'])}). "
                  if len(_tie) <= 1 else "**How to read the table**: ")
                 + f"`baseline_slope`/`robust_ref` are offline deterministic references -- a real model scoring below `robust_ref` "
                 f"means that capability has not yet been reliably mastered.\n")

    # ---------- 1r reliability (retries . rescue rate . attribution) ----------
    L += render_reliability_table(reliability_by_model(rows, job), ranking)
    from .run_scheduler import stall_report_lines
    L += stall_report_lines(rows)

    # ---------- 1i cost- and token-controlled efficiency analysis ----------
    # Diagnostic items have their own composite score. The early-warning board
    # above is inapplicable here and supplies None for every model's score.
    _eff = cost_efficiency_analysis(rows, dd if is_ddx_batch(rows) else ranking)
    L += render_cost_efficiency_table(_eff)

    # ---------- 2 sampled cases ----------
    L.append("---\n")
    # Per-slice gates and case-level gates are separate sections; cross-product
    # coverage is printed as well.
    from .mount_coverage import coverage as _mt_cov
    from .mount_table import known_gaps as _mt_gaps
    _cov = _mt_cov()
    L.append("### 1x. Cross-product coverage of judge \u00d7 geometry\n")
    L.append(f"> {_cov['n_judges']} judges \u00d7 {_cov['n_geometries']} geometries = "
             f"**{_cov['n_cells']} cells**; mounted **{_cov['mounted']}** \u00b7 "
             f"declared absent {_cov['declared_absent']} \u00b7 **unexplained holes {_cov['holes']}**\n")
    _g = _mt_gaps()
    if _g:
        # Separator hoisted out of the f-string: a backslash inside an f-string
        # expression is a SyntaxError before 3.12 (PEP 701), and requires-python
        # promises 3.10.
        _MT_GAP_SEP = "` \u00b7 `"
        L.append(f"> **{len(_g)} known gap(s)** "
                 f"(judge applies but is not wired on that geometry): "
                 f"`{_MT_GAP_SEP.join(_g)}`\n")
    L.append("> Every cell marked \"declared absent\" has a reason in `mount_table.WHY_NOT` -- "
             "\"too narrow\" and \"genuinely not applicable\" render identically on the table, and the reason is the only thing that tells them apart.\n")
    L += _case_gate_surface(rows, built)
    L += _action_gate_surface(rows)
    L.append(f"## 2. Sampled cases (§12: {len(sample_ids)} cases · presented in time order · with a collapsible raw JSON)\n")
    for cid in sample_ids:
        raw = built[cid]
        T = int(raw.prediction_context["prediction_time_T"])
        # Sorted by solver name so the report does not depend on completion order.
        crs = sorted((r for r in rows if r.get("case") == cid),
                     key=lambda r: str(r.get("solver") or ""))
        L.append(f"### {cid}\n")
        gold = f"`{raw.outcome_label}` / top driver `{(raw.gold_drivers or ['—'])[0]}`"
        L.append(f"**Gold standard (verifier-only, not in the raw section below)**: {gold}; index-time T={T}.\n")

        # 2a time-progress table
        tl = timeline_rows(raw, T, k_pts)
        if tl:
            cols = [c for c in tl[0] if c not in ("day", "_ev")]
            L.append("**Observation stream in time order** (solver-visible, ≤T):\n")
            L.append("| Day | " + " | ".join(cols) + " | Cumulative evidence |")
            L.append("|" + "---|" * (len(cols) + 2))
            for r in tl:
                L.append(f"| {r['day']} | " + " | ".join(_fmt(r.get(c)) for c in cols)
                         + f" | {r['_ev']} |")
            L.append("")

        # 2b each model's result on this case
        if crs:
            L.append("**Each model's result and score on this case**:\n")
            if multiround:
                L.append("| Model | Rounds | Revised | Track E | Latency (days) | Fooled by false alarm | overall |")
                L.append("|---|---|---|---|---|---|---|")
                for r in crs:
                    L.append(f"| {r['solver']} | {r.get('rounds','—')} | {r.get('revised','—')} | "
                             f"{_fmt(r.get('trackE'))} | {r.get('latencies','—')} | "
                             f"{r.get('trap_fooled','—')}/{r.get('n_trap','—')} | {r.get('overall')} |")
                # per-round time progress (for the top-ranked model)
                best = next((r for r in crs if r["solver"] == models[0] and r.get("trajectory")), None)
                if best:
                    L.append(f"\n**{best['solver']}'s belief per round (over time)**:\n")
                    L.append("| Round | Day | Risk category | Driver | Action | Repair |")
                    L.append("|---|---|---|---|---|---|")
                    for t in best["trajectory"]:
                        L.append(f"| {t['round']} | {t['day']} | {t['risk_cat']} | {t['top_driver']} "
                                 f"| {t['action']} | {t['repair']} |")
            else:
                L.append("| Model | risk / category | Direction | Top driver | Driver hit | Track A/B/C/D | Hard gate | overall |")
                L.append("|---|---|---|---|---|---|---|---|")
                for r in crs:
                    # Uses `applicable_tracks`, matching the aggregate layer's not-applicable tracks.
                    tk = applicable_tracks(r)
                    L.append(f"| {r['solver']} | {_fmt(r.get('risk'))} / {r.get('risk_category','—')} | "
                             f"{'✓' if r.get('direction_ok') else '✗'} | {r.get('top_driver','—')} | "
                             f"{'✓' if r.get('driver_hit') else '✗'} | "
                             f"{_fmt(tk.get('A'))}/{_fmt(tk.get('B'))}/{_fmt(tk.get('C'))}/{_fmt(tk.get('D'))} | "
                             f"{', '.join(r.get('gates') or []) or '—'} | {r.get('overall')} |")
            L.append("")

        # 2c collapsible raw JSON (§12: only the solver-visible observation
        # stream goes here)
        blob, leak = solver_payload_json(raw, T)
        if leak:
            log.error("[report] %s raw section appears to leak %s -- this collapsed block has been omitted", cid, leak)
            L.append(f"> This case's raw section was omitted because it failed the leak self-check ({leak}).\n")
        else:
            L.append("<details>")
            L.append(f"<summary><b>Expand: raw case text handed to the solver (JSON, ≤T, no ground truth)</b> — {cid}</summary>\n")
            L.append("```json")
            L.append(blob)
            L.append("```")
            L.append("</details>\n")
    L.append("---\n")
    L.append(f"*This report was auto-generated by `haenv`; traceable: every sampled case can be found as a corresponding row in "
             f"`{job.results_file.relative_to(job.root)}`.*\n")
    return "\n".join(L)


def write_report(job, rows, built, audits, cfg) -> Path:
    md = render(job, rows, built, audits, cfg)
    p = job.report_file
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    log.info("[report] -> %s (%d bytes)", p, len(md))
    return p
