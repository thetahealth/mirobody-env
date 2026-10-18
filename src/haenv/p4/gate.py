"""Pack 4's emission declaration for the shared gate (`haenv/world/validation/shared_audit.py`, SA-1 + SA-8).

On a case with a pack-4 plan the shared gate refuses emission when the item could not be realized,
when G4 re-derived from the stored records disagrees with the gold, when the realized class is not
the planned one, when the visible block is missing, when the solver payload carries a gold word or
a gold number (4-decimal truths, the gold percentage), or when a patient sentence carries a word of
the other sex, or when the visible block does not decide the gold (`visible_decides`), or when the record names a condition that is not a condition of the world (disease +
comorbidities) the labs were drawn from.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

GATE = "p4"

#: Gold-side literals that must never appear in what the solver sees.
EN_TOKENS = ("true_change", "analytic_biological_noise", "preanalytical", "method_difference",
             "glp1_dose_change", "weight_change", "acei_start_or_stop", "postprandial_labelled_fasting",
             "delayed_processing", "meat_meal", "poc_meter", "jaffe_enzymatic_switch", "hemolysis_note",
             "lab_switch_zero_bias", "tube_type", "lab_truth", "truth", "bias_log", "decoy", "perturbation", "acts",
             "true_change_pct", "next_step", "act_on_change", "repeat_standardized")
ZH_TOKENS = ("真变化", "真实变化", "分析前", "方法差异", "测量噪声", "生物学变异", "诱饵", "偏倚", "扰动", "真值",
             "金标", "干扰因素")
from haenv.p4.diversity import table as _diversity_table
EN_TOKENS = tuple(dict.fromkeys((*EN_TOKENS, *(k for k in _diversity_table()["rules"] if "_" in k and k != "reagent_lot"),
                               "diversity_version", "cause_params")))


def plan_of(raw) -> dict | None:
    from haenv.contracts.external_gold import SLOT
    lp = raw.get("latent_premise") if isinstance(raw, dict) else getattr(raw, "latent_premise", None)
    p = (((lp or {}).get("meta") or {}).get(SLOT) or {}).get("p4")
    return p if isinstance(p, dict) else None


def _unrealizable(blk: dict) -> str | None:
    if blk.get("unrealizable") or "records" not in blk:
        return str(blk.get("unrealizable") or "no realized item")
    return None


def _recompute(blk: dict, raw) -> list[str]:
    from haenv.p4.gold import gold_block_consistent
    from haenv.p4.diversity import verification_consistent
    return gold_block_consistent(blk) + verification_consistent(blk)


def gold_numbers(blk: dict) -> list[str]:
    out = [f"{float(r['truth']):.4f}" for r in blk.get("records") or ()]
    pct = blk.get("true_change_pct")
    if pct is not None:
        out += [f"{float(pct):.3f}", f"{float(pct):.2f}"]
    return out


def record_vs_world(g: dict, raw) -> list[tuple[str, str]]:
    """A condition on the shown record that is not a condition of the world (disease + comorbidities)
    the labs were drawn from."""
    get = (lambda k: raw.get(k)) if isinstance(raw, dict) else (lambda k: getattr(raw, k, None))
    pb = (get("latent_premise") or {}).get("patient_basics") or {}
    world = {str(pb.get("disease") or "")} | {str(x) for x in pb.get("comorbidities") or ()}
    shown = (get("user_profile") or {}).get("known_conditions") or ()
    return [("record_vs_world", c) for c in dict.fromkeys(map(str, shown)) if c not in world]


def visible_decides(g: dict, raw) -> list[tuple[str, str]]:
    """The block the solver reads must decide the gold: a reader that has only the visible block and the documented
    rules (`features.visible_class`) names the gold class. Anything else (no candidate agrees with the change,
    several do, a ratio between the two bands) is refused."""
    from haenv.p4.features import _blk
    from haenv.p4.features import visible_class
    if not _blk(raw):
        return []
    vc, _ = visible_class(raw)
    bad = [("visible_class_differs", vc)] if vc != g.get("class") else []
    if g.get("diversity_version"):
        from haenv.p4.diversity_reader import significance_violations
        from haenv.p4.features import _blk
        bad += [("visible_significance", why) for why in significance_violations(_blk(raw), g["class"], g.get("cause"))]
    return bad


def extra(g: dict, raw) -> list[tuple[str, str]]:
    bad = record_vs_world(g, raw) + visible_decides(g, raw)
    if g.get("diversity_version"):
        # Broader eligibility exposes backgrounds outside the former ACEI-only
        # patient set. Reuse the existing text/number consistency invariant at
        # emission; do not publish an inconsistent foundation or delete IDs.
        from haenv.world.validation.invariants import c3
        get = (lambda k: raw.get(k)) if isinstance(raw, dict) else (lambda k: getattr(raw, k, None))
        sp = {k: get(k) for k in ("prediction_context", "longitudinal_data", "evidence_ledger", "user_profile")}
        case = raw if isinstance(raw, dict) else vars(raw)
        bad += [("background_text_lab_conflict", why) for why in c3({"pack": "p4", "case": case, "sp": sp})]
    return bad


def emission():
    from haenv.world.validation.shared_audit import Emission
    return Emission(gate=GATE, block="followup_p4", plan_of=plan_of, recompute=_recompute,
                    class_of=lambda g: g.get("class"), planned_class=lambda p: p.get("class"),
                    words=EN_TOKENS + ZH_TOKENS, numbers=gold_numbers, unrealizable=_unrealizable,
                    extra=extra)
