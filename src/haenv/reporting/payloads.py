"""Payloads for the evaluation report."""
from __future__ import annotations

from haenv.scoring.quantities import QUANTITIES as _QUANTITIES
from haenv_kernel.build import build_instance
from haenv_kernel.build import leakage_probe


TRUTH_WORDS = ("outcome_label", "gold_driver", "adjudication", "latent_premise", "premise",
               "reversal_points", "distractor", "future_data", "verifier")


_SLICE_ALIAS_ROW = {q.canonical: (q.row_aliases + q.renamed_from)[0]
                    for q in _QUANTITIES if (q.row_aliases or q.renamed_from)}


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
