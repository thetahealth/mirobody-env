"""Pack 3's emission declaration for the shared gate (`haenv/world/validation/shared_audit.py`, SA-1 + SA-8).

On a case with a pack-3 plan the shared gate refuses emission when the item could not be realized,
when G3 re-derived from the records disagrees with the gold (or the visible block is not the
rendering of the records), when the realized class is not the planned one, when the solver payload
carries a gold word or a 4-decimal truth, or when a patient sentence carries a word of the other sex.
The pack callback (P-9, A3/A12/A15) also refuses a case whose payload outside the block names a fact
on the managed drug's `decision_facts` list, or shows a stream that bears on the drug's decision:
a visible fact that changes the decision is read by G3 or the case is not emitted. A condition on the
record that is not a condition of the world (disease + comorbidities) the labs were drawn from is refused.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json

GATE = "p3"
BLOCK = "meds_p3"

EN_TOKENS = ("uptitrate", "downtitrate", "maintain", "switch", "check_adherence_or_adverse_effect",
             "dose_related", "unattributed", "predating", "decoy", "truth", "adherence_true", "subtype",
             "drug_symptom", "top_off", "adh_low", "ae_dose", "ae_min", "med_decision")
ZH_TOKENS = ("剂量相关", "未归因", "诱饵", "真值", "金标", "依从真值", "过度治疗", "药效未到", "刚加量", "先查依从")


def plan_of(raw) -> dict | None:
    from haenv.contracts.external_gold import SLOT
    lp = raw.get("latent_premise") if isinstance(raw, dict) else getattr(raw, "latent_premise", None)
    p = (((lp or {}).get("meta") or {}).get(SLOT) or {}).get("p3")
    return p if isinstance(p, dict) else None


def _unrealizable(blk: dict) -> str | None:
    if blk.get("unrealizable") or "visits" not in blk:
        return str(blk.get("unrealizable") or "no realized item")
    return None


def _recompute(blk: dict, raw) -> list[str]:
    from haenv.p3.gold import gold_block_consistent
    from haenv.p3.render import render_block
    pc = (raw.get("prediction_context") if isinstance(raw, dict) else getattr(raw, "prediction_context", None)) or {}
    bad = list(gold_block_consistent(blk))
    expected = render_block(blk)
    if blk.get("record_format") == "emr":
        from haenv.p3.emr import source_block
        expected = source_block(expected, blk["current_medications"], blk["patient_quote"])
    if isinstance(pc.get(BLOCK), dict) and expected != pc[BLOCK]:
        bad.append("visible block is not the rendering of the records")
    return bad


def gold_numbers(blk: dict) -> list[str]:
    return [f"{float(r['truth']):.4f}" for r in blk.get("visits") or ()]


def ae_terms_outside(sp, drug_name: str) -> list[str]:
    """Words of the managed drug's `decision_facts` in the solver payload outside the pack-3 block."""
    from haenv.world.labworld.meds import drug
    d = sp if isinstance(sp, dict) else json.loads(sp.dumps())
    d = {**d, "prediction_context": {k: v for k, v in (d.get("prediction_context") or {}).items() if k != BLOCK}}
    text = json.dumps(d, ensure_ascii=False, default=str)
    return [t for t in drug(drug_name).get("decision_facts") or () if t in text]


def decision_facts_visible(blk: dict, sp) -> list[tuple[str, str]]:
    """P-9: a fact that bears on the decision, visible outside what G3 reads."""
    from haenv.world.labworld.meds import drug
    out = []
    ld = (sp.get("longitudinal_data") if isinstance(sp, dict) else getattr(sp, "longitudinal_data", None)) or {}
    seen = sorted(set(ld) & set(drug(blk["drug"]).get("bearing_streams") or ()))
    if seen:
        out.append(("bearing_stream_visible", ", ".join(seen)))
    ae = ae_terms_outside(sp, blk["drug"])
    if ae:
        out.append(("decision_fact_outside_block", ", ".join(ae)))
    return out


def record_vs_world(g: dict, raw) -> list[tuple[str, str]]:
    """A condition on the shown record that is not a condition of the world (disease + comorbidities)
    the labs were drawn from."""
    get = (lambda k: raw.get(k)) if isinstance(raw, dict) else (lambda k: getattr(raw, k, None))
    pb = (get("latent_premise") or {}).get("patient_basics") or {}
    world = {str(pb.get("disease") or "")} | {str(x) for x in pb.get("comorbidities") or ()}
    shown = (get("user_profile") or {}).get("known_conditions") or ()
    return [("record_vs_world", c) for c in dict.fromkeys(map(str, shown)) if c not in world]


def emission():
    from haenv.world.validation.shared_audit import Emission
    return Emission(gate=GATE, block=BLOCK, plan_of=plan_of, recompute=_recompute,
                    class_of=lambda g: g.get("class"), planned_class=lambda p: p.get("class"),
                    words=EN_TOKENS + ZH_TOKENS, numbers=gold_numbers, unrealizable=_unrealizable,
                    visible=decision_facts_visible, extra=record_vs_world)
