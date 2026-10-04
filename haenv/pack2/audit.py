"""Pack 2's declaration for the shared batch audit (`haenv/shared_audit.py`, `tools/pack_audit.py`).

Items are the emitted cases; their content-free features are read off the visible block, the
case and the job's raw facts (`surface_of` plus the evidence-ledger and stream counts of the built
world). Composition cell: (H, disposition, focal indicator), exactly as the job planned it; the job's
(H, disposition) counts must equal `quotas(n_items)`, the table the generator deals from. Prior
stub: the condition registry's urgency mapped to a disposition.

Pack-specific gate P-8 (L2-10): no symptom term in a tone sentence, and in a symptom sentence only
the terms of that focal finding's `symptom_set` (`registry/pack2_acute_events.yaml`).

Run with `--job inputs/pack2-triage.job.yaml`. SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import collections

from .. import pack_size as PS
from . import world
from .score import chance_corrected

SPEC_URGENCY_TO_DISP = {"🔴": "ed_now", "🟠": "within_24h", "🟡": "routine_followup", "🟢": "watchful_waiting"}
QUANTITIES = ("rr", "spo2", "sbp", "hr", "temp", "glucose", "K")
SURFACE_NAMES = ("alarming", "alarm_words", "complaint_len", "spec_urgency_rank", "focal_idx", "H",
                 "cause_idx", "explains_away", "n_other_results", "age_lo", "female",
                 "n_comorbidities", "n_devices", "T", "insufficient_tier", "distractor_level_idx",
                 "n_evidence", "n_streams")
#: SA-5: every surface variable, and each background vital on the items where it is not focal (L2-3)
UNIVARIATE = SURFACE_NAMES + tuple(f"bg_{q}" for q in QUANTITIES)


def alarm_count(text: str) -> int:
    return sum(text.count(w) for w in world.events_table()["wording"]["alarm_words"])


def surface_of(item: dict, raw: dict, latent: dict) -> list[float]:
    vis, plan = item["visible"], item["plan"]
    cc = vis["chief_complaint"]
    age = str(raw.get("age_range") or "40-44")
    a0 = int(age.split("-")[0]) if age[:2].isdigit() else 40
    return [int(plan["wording"] == "alarming"), alarm_count(cc), len(cc),
            {"🔴": 3, "🟠": 2, "🟡": 1, "🟢": 0}.get(item["spec_urgency"], -1),
            world.FOCALS.index(plan["focal"]), item["H"],
            ["benign", "known_condition", "hidden_line"].index(plan.get("cause") or "benign"),
            int(bool(plan.get("explains_away"))), len(vis.get("other_results_today") or []),
            a0, int(raw.get("sex") == "F"), len(raw.get("comorbidities") or []),
            len(raw.get("devices") or []), int(latent.get("index_time_T") or 0),
            int(bool(latent.get("ddx_insufficient"))),
            ["none", "low", "mid", "high"].index(latent.get("distractor_level", "none"))
            if latent.get("distractor_level", "none") in ("none", "low", "mid", "high") else -1]


def focal_quantities(gold: dict) -> set[str]:
    f = gold["focal"]
    if f == "NEWS2":
        return set(world.thresholds()["focal_values"]["NEWS2"].get(gold["planned"], {}))
    return {"K": {"K"}, "SBP": {"sbp"}, "glucose": {"glucose"}}[f]


def _context(batch, job_path, ref=None):
    import yaml
    if job_path is None:
        raise SystemExit("pack 2 needs --job (the raw facts of its cases)")
    job = yaml.safe_load(open(job_path, encoding="utf-8"))
    return {"job": {c["case_id"]: (c["raw"], c["latent"]) for c in job["cases"]}}


def rows(items: list[dict], ctx) -> list[dict]:
    out = []
    for it in items:
        c, g = it["case"], it["gold"]
        raw, lat = ctx["job"][it["case_id"]]
        view = {"visible": c["prediction_context"][world.BLOCK], "plan": dict(lat[world.LATENT_KEY]),
                "spec_urgency": g["spec"]["urgency"], "H": g["hidden"]}
        x = surface_of(view, raw, lat) + [len(c.get("evidence_ledger") or []), len(c.get("longitudinal_data") or {})]
        fq = focal_quantities(g)
        out.append({**dict(zip(SURFACE_NAMES, x)), **{f"bg_{q}": (None if q in fq else g["values"][q]) for q in QUANTITIES},
                    "class": it["class"], "spec": it["spec"], "case_id": it["case_id"]})
    return out


def quotas(n_items: int) -> dict:
    """{(H, disposition): n}: the four dispositions in equal shares, each halved over H."""
    return PS.split_h(PS.apportion(n_items, dict.fromkeys(world.DISPOSITIONS, 1)))


def _cell(plan_or_gold: dict, H: int) -> tuple:
    return (H, plan_or_gold.get("disposition"), plan_or_gold.get("focal"))


def planned_cells(items, ctx, n_items: int = 50) -> dict:
    """The job's planned cells when its (H, disposition) counts follow `quotas(n_items)`; otherwise
    the quota table itself (focal None), so SA-3 names the off-table cells."""
    out = collections.Counter()
    for raw, lat in ctx["job"].values():
        out[_cell(lat[world.LATENT_KEY], int(lat.get("ddx_join_gold") in ("unified", "comorbidity")))] += 1
    margin = collections.Counter()
    for (h, d, _f), k in out.items():
        margin[(h, d)] += k
    want = {k: v for k, v in quotas(n_items).items() if v}
    if dict(margin) != want:
        return {(h, d, None): k for (h, d), k in want.items()}
    return dict(out)


def l2_10(ev: dict | None = None) -> dict:
    """P-8: no symptom term (`wording.symptom_terms`) in a tone sentence, and in a symptom sentence
    only the terms of that focal finding's `symptom_set`."""
    w = (ev or world.events_table())["wording"]
    bad = []
    for f in world.FOCALS:
        allowed = set(w[f]["symptom_set"])
        sents = [(k, t, set()) for k in world.WORDINGS for t in w[f][k]]
        sents += [(d, t, allowed) for d, ts in w[f]["symptoms"].items() for t in ts]
        bad += [(f, k, t, term) for k, t, ok in sents for term in w["symptom_terms"]
                if term in t and term not in ok]
    return {"id": "P-8", "pass": not bad, "n_hits": len(bad), "examples": bad[:5]}


def audit(n_items: int = 50, seed: str = ""):
    from ..shared_audit import PackAudit
    return PackAudit(
        name="pack2", block=world.BLOCK, classes=world.DISPOSITIONS,
        class_of=lambda g: g.get("disposition") if g.get("pack") == "pack2" else None,
        rows=rows, univariate=UNIVARIATE, surface=SURFACE_NAMES, cc=chance_corrected,
        cell_of=lambda it, ctx: _cell(it["gold"], it["gold"]["hidden"]),
        planned_cells=lambda items, ctx: planned_cells(items, ctx, n_items),
        prior=lambda items, ctx: [SPEC_URGENCY_TO_DISP.get(it["gold"]["spec"]["urgency"]) for it in items],
        pack_gates={"P-8": lambda pool, rows_, ctx: l2_10()}, context=_context)
