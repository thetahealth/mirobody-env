"""Samples for the evaluation report."""
from __future__ import annotations

from haenv.scoring.analysis.tracks import applicable_tracks
from haenv.reporting.formatting import (
    _fmt,
)
from haenv.reporting.payloads import (
    solver_payload_json,
    timeline_rows,
)
import logging

log = logging.getLogger("haenv.report")


def append_sampled_cases(L, rows, built, sample_ids, k_pts, multiround, models) -> None:
    """Append the original samples report section in its original order."""
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
