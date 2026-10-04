"""Pack 4's emission declaration for the shared gate (`haenv/shared_audit.py`, SA-1 + SA-8).

On a case with a pack-4 plan the shared gate refuses emission when the item could not be realized,
when G4 re-derived from the stored records disagrees with the gold, when the realized class is not
the planned one, when the visible block is missing, when the solver payload carries a gold word or
a gold number (4-decimal truths, the gold percentage), or when a patient sentence carries a word of
the other sex.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

GATE = "p4"

#: Gold-side literals that must never appear in what the solver sees.
EN_TOKENS = ("true_change", "analytic_biological_noise", "preanalytical", "method_difference",
             "glp1_dose_change", "weight_change", "acei_start_or_stop", "postprandial_labelled_fasting",
             "delayed_processing", "meat_meal", "poc_meter", "jaffe_enzymatic_switch", "hemolysis_note",
             "lab_switch_zero_bias", "lab_truth", "truth", "bias_log", "decoy", "perturbation", "acts",
             "true_change_pct", "next_step", "act_on_change", "repeat_standardized")
ZH_TOKENS = ("真变化", "真实变化", "分析前", "方法差异", "测量噪声", "生物学变异", "诱饵", "偏倚", "扰动", "真值",
             "金标", "干扰因素")


def plan_of(raw) -> dict | None:
    from ..external_gold import SLOT
    lp = raw.get("latent_premise") if isinstance(raw, dict) else getattr(raw, "latent_premise", None)
    p = (((lp or {}).get("meta") or {}).get(SLOT) or {}).get("p4")
    return p if isinstance(p, dict) else None


def _unrealizable(blk: dict) -> str | None:
    if blk.get("unrealizable") or "records" not in blk:
        return str(blk.get("unrealizable") or "no realized item")
    return None


def _recompute(blk: dict, raw) -> list[str]:
    from .gold import gold_block_consistent
    return gold_block_consistent(blk)


def gold_numbers(blk: dict) -> list[str]:
    out = [f"{float(r['truth']):.4f}" for r in blk.get("records") or ()]
    pct = blk.get("true_change_pct")
    if pct is not None:
        out += [f"{float(pct):.3f}", f"{float(pct):.2f}"]
    return out


def emission():
    from ..shared_audit import Emission
    return Emission(gate=GATE, block="followup_p4", plan_of=plan_of, recompute=_recompute,
                    class_of=lambda g: g.get("class"), planned_class=lambda p: p.get("class"),
                    words=EN_TOKENS + ZH_TOKENS, numbers=gold_numbers, unrealizable=_unrealizable)
