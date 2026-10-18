"""Document for the evaluation report."""
from __future__ import annotations

from datetime import datetime
from datetime import timezone
from pathlib import Path
from haenv.scoring.analysis.ceilings import batch_discrimination_gate
from haenv.scoring.analysis.ceilings import constant_ceilings
from haenv.scoring.analysis.costs import cost_efficiency_analysis
from haenv.scoring.analysis.ranking import rank_models
from haenv.scoring.analysis.tracks import is_ddx_batch
from haenv.reporting.boards import (
    append_forecast_board,
)
from haenv.reporting.coverage import (
    append_mount_coverage,
)
from haenv.reporting.diagnosis import (
    append_diagnosis_board,
)
from haenv.reporting.generation import (
    append_header_and_emission,
)
from haenv.reporting.probes import (
    append_probe_sections,
)
from haenv.reporting.readouts import (
    disputed_gold_hits,
)
from haenv.reporting.reliability import (
    reliability_by_model,
)
from haenv.reporting.samples import (
    append_sampled_cases,
)
from haenv.reporting.tables import (
    render_cost_efficiency_table,
    render_reliability_table,
)
from haenv.reporting.validity import (
    append_validity_profile,
)
import logging

log = logging.getLogger("haenv.report")


def write_report(job, rows, built, audits, cfg) -> Path:
    md = render(job, rows, built, audits, cfg)
    p = job.report_file
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    log.info("[report] -> %s (%d bytes)", p, len(md))
    return p


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
    append_header_and_emission(L, job, rows, audits, multiround, ts)

    # ---------- 1 multi-model overall ranking ----------
    # The process readings below print whenever their fields exist, regardless of item type.
    append_probe_sections(L, rows)
    dd = append_diagnosis_board(L, job, rows, built, _disc_gate)
    # ---- gold-standard provenance (rendered by `provenance_report.py`) ----
    from haenv.reporting.provenance_report import render as _prov_render
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
    append_validity_profile(L, rows, dd)

    append_forecast_board(L, job, rows, multiround, ranking)

    # ---------- 1r reliability (retries . rescue rate . attribution) ----------
    L += render_reliability_table(reliability_by_model(rows, job), ranking)
    from haenv.runtime.scheduler import stall_report_lines
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
    append_mount_coverage(L, rows, built)
    append_sampled_cases(L, rows, built, sample_ids, k_pts, multiround, models)
    L.append("---\n")
    L.append(f"*This report was auto-generated by `haenv`; traceable: every sampled case can be found as a corresponding row in "
             f"`{job.results_file.relative_to(job.root)}`.*\n")
    return "\n".join(L)
