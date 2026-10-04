"""Pack 4 judging: answer parsing, per-row atoms and the pack composite (PREREG 5).

Per row (`row_atoms`): the said / gold source class, the said / gold true-change percentage and
its absolute error, the factor and next-step profile atoms. Per pack (`pack_score`):

    change_source_cc   = (m * BA - 1) / (m - 1), BA over the classes present in the gold
    true_change_skill  = 1 - MAE_model / MAE_base, MAE_base = min over {0, obs %, obs % / 2}
                         fixed on the emitted pack (`naive_base`), clipped to [-1, 1] for the
                         composite only
    composite          = (cc + skill_clipped) / 2

A missing or unparseable class counts as wrong and a missing percentage counts as 0; a cell
aborted by the leak gate has no answer and is scored the same way. There is no hard-gate
multiplier: both parts can be negative, and multiplying a negative score by a factor below 1
would reward an abort (PREREG revision R9).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

from .gold import CLASSES, NEXT_STEP

JUDGE = "followup_p4"
KEYS = ("p4_class_gold", "p4_class_said", "p4_class_ok", "p4_pct_gold", "p4_pct_said", "p4_pct_abs_err",
        "p4_obs_pct", "p4_factor_gold", "p4_factor_said", "p4_factor_ok", "p4_next_gold", "p4_next_said",
        "p4_next_ok", "p4_analyte", "p4_judging_sha16")

_SYN = {
    "true_change": ("true_change", "true", "real", "真变化", "真实变化", "真实"),
    "analytic_biological_noise": ("analytic_biological_noise", "noise", "biological", "噪声", "测量噪声", "随机"),
    "preanalytical": ("preanalytical", "pre-analytical", "preanalytic", "分析前"),
    "method_difference": ("method_difference", "method", "方法", "实验室差异"),
}
_NEXT = ("act_on_change", "repeat_standardized", "no_action")


def parse_class(said) -> str | None:
    s = str(said or "").strip().lower()
    if not s:
        return None
    if s in CLASSES:
        return s
    hits = [c for c, syn in _SYN.items() if any(x in s for x in syn)]
    return hits[0] if len(hits) == 1 else None


def parse_pct(said) -> float | None:
    if isinstance(said, bool):
        return None
    if isinstance(said, (int, float)) and math.isfinite(float(said)):
        return float(said)
    try:
        v = float(str(said).strip().rstrip("%"))
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def row_atoms(answer: dict, gold: dict, sha: str = "") -> dict:
    a = answer if isinstance(answer, dict) else {}
    out = dict.fromkeys(KEYS)
    if not isinstance(gold, dict) or gold.get("class") not in CLASSES:
        return out
    said = parse_class(a.get("change_source"))
    pct = parse_pct(a.get("true_change_pct"))
    fac = a.get("factor")
    fac = None if fac in (None, "", "null") else str(fac).strip()
    nxt = str(a.get("next_step") or "").strip() or None
    g = float(gold["true_change_pct"])
    out.update(p4_class_gold=gold["class"], p4_class_said=said, p4_class_ok=int(said == gold["class"]),
               p4_pct_gold=g, p4_pct_said=pct, p4_pct_abs_err=round(abs((pct if pct is not None else 0.0) - g), 4),
               p4_obs_pct=gold.get("obs_pct"),
               p4_factor_gold=gold.get("factor"), p4_factor_said=fac,
               p4_factor_ok=(int(fac == gold.get("factor")) if gold["class"] in ("preanalytical", "method_difference") else None),
               p4_next_gold=NEXT_STEP[gold["class"]], p4_next_said=nxt,
               p4_next_ok=int(nxt == NEXT_STEP[gold["class"]]),
               p4_analyte=gold.get("analyte"), p4_judging_sha16=sha or None)
    return out


def balanced_acc(gold: list[str], pred: list) -> tuple[float, int]:
    cls = sorted(set(gold))
    recs = []
    for c in cls:
        idx = [i for i, g in enumerate(gold) if g == c]
        recs.append(sum(pred[i] == c for i in idx) / len(idx))
    return (sum(recs) / len(recs) if recs else float("nan")), len(cls)


def cc(gold: list[str], pred: list) -> float | None:
    ba, m = balanced_acc(gold, pred)
    return None if m < 2 else (m * ba - 1) / (m - 1)


def naive_base(gold_pct: list[float], obs_pct: list[float]) -> dict:
    """MAE of the three naive estimators on the pack and the base (the smallest)."""
    est = {"zero": [0.0] * len(gold_pct), "obs": list(obs_pct), "half_obs": [x / 2 for x in obs_pct]}
    mae = {k: sum(abs(e - g) for e, g in zip(v, gold_pct)) / len(gold_pct) for k, v in est.items()}
    return {"mae": {k: round(v, 6) for k, v in mae.items()}, "base_mae": round(min(mae.values()), 6),
            "base": min(mae, key=mae.get)}


def skill(abs_err: list[float], base_mae: float) -> float:
    return 1.0 - (sum(abs_err) / len(abs_err)) / base_mae


def pack_score(rows: list[dict], base_mae: float) -> dict:
    """Composite of one model on the pack from its rows (`row_atoms` keys)."""
    g = [r["p4_class_gold"] for r in rows]
    p = [r.get("p4_class_said") for r in rows]
    c = cc(g, p)
    sk = skill([float(r["p4_pct_abs_err"]) for r in rows], base_mae)
    skc = max(-1.0, min(1.0, sk))
    by_an = {}
    for an in sorted({r.get("p4_analyte") for r in rows}):
        sub = [r for r in rows if r.get("p4_analyte") == an]
        by_an[an] = round(balanced_acc([r["p4_class_gold"] for r in sub], [r.get("p4_class_said") for r in sub])[0], 4)
    fac = [r["p4_factor_ok"] for r in rows if r.get("p4_factor_ok") is not None]
    return {"n": len(rows), "change_source_cc": c, "change_source_ba": balanced_acc(g, p)[0],
            "true_change_skill": sk, "true_change_skill_clipped": skc,
            "composite": ((c if c is not None else 0.0) + skc) / 2,
            "profile": {"factor_hit": (sum(fac) / len(fac)) if fac else None,
                        "next_step_agree": sum(int(r.get("p4_next_ok") or 0) for r in rows) / len(rows),
                        "ba_by_analyte": by_an}}
