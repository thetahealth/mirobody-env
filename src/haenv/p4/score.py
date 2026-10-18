"""Pack 4 judging: answer parsing, per-row atoms and the pack composite (PREREG 5).

Per row (`row_atoms`): the said / gold source class, the said / gold true-change percentage and
its absolute error, the factor and next-step profile atoms. Per pack (`pack_score`):

    change_source_cc   = (m * BA - 1) / (m - 1), BA over the classes present in the gold
    true_change_skill  = 1 - MAE_model / MAE_base, MAE_base = min over {0, obs %, obs % / 2}
                         fixed on the emitted pack (`naive_base`); profile
    composite          = cc

The skill tracks the true-change recall, so averaging it in would count the same decision twice;
it is a profile reading. `next_step` is scored against a
set of acceptable answers (`gold.next_step`), profile only.

The source class is scored against `accepted_classes(gold, accept)`. By default that is the gold
class alone. `ACCEPTED_CLASS_CHANGES` registers acceptance-set changes, each widening the credited
classes on one gold (class, factor) cell; a change applies only when its id is passed in `accept`
(opt-in, e.g. for a rescore of stored rows). The gold block, the emitted question and the
next-step rule are never changed by it, and `p4_class_gold` still records the gold class.

A missing or unparseable class counts as wrong and a missing percentage counts as 0; a cell
aborted by the leak gate has no answer and is scored the same way. There is no hard-gate
multiplier and no floor (the rule shared by the four packs).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

from haenv.p4.gold import CLASSES
from haenv.p4.gold import next_step

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

#: Registered acceptance-set changes for the source class, keyed by the gold (class, factor) cell.
#: Each entry lists every class credited on that cell (gold class first) and why. Inactive unless
#: its id is passed in `accept` (`accepted_classes`, `row_atoms`).
ACCEPTED_CLASS_CHANGES = {
    ("method_difference", "biotin_tsh"): {
        "id": "P4-AC-001",
        "date": "2026-10-17",
        "accept": ("method_difference", "preanalytical"),
        "why": ("The prompt defines `preanalytical` as a sampling or specimen factor and names intake "
                "before the draw; biotin is taken before sampling and both draws of these records use "
                "the same assay platform, so `preanalytical` is as defensible as the gold "
                "`method_difference`. Off by default; when enabled, either class is accepted for these "
                "records. The next-step rule is unchanged."),
    },
}
_CHANGE_IDS = frozenset(e["id"] for e in ACCEPTED_CLASS_CHANGES.values())


def accepted_classes(gold: dict, accept: tuple = ()) -> tuple:
    """Source classes credited for this gold block: the gold class, widened by a registered
    acceptance-set change on its (class, factor) cell only when that change's id is in `accept`."""
    unknown = set(accept) - _CHANGE_IDS
    if unknown:
        raise ValueError(f"unregistered acceptance-set change(s) {sorted(unknown)}; see ACCEPTED_CLASS_CHANGES")
    g = gold if isinstance(gold, dict) else {}
    entry = ACCEPTED_CLASS_CHANGES.get((g.get("class"), g.get("factor")))
    return tuple(entry["accept"]) if entry and entry["id"] in accept else (g.get("class"),)


def next_ok_set(gold: dict) -> tuple:
    """Next steps accepted for this gold block (`gold.next_step`, primary first)."""
    return next_step(gold)[1]


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


def row_atoms(answer: dict, gold: dict, sha: str = "", accept: tuple = ()) -> dict:
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
    out.update(p4_class_gold=gold["class"], p4_class_said=said, p4_class_ok=int(said is not None and said in accepted_classes(gold, accept)),
               p4_pct_gold=g, p4_pct_said=pct, p4_pct_abs_err=round(abs((pct if pct is not None else 0.0) - g), 4),
               p4_obs_pct=gold.get("obs_pct"),
               p4_factor_gold=gold.get("factor"), p4_factor_said=fac,
               p4_factor_ok=(int(fac == gold.get("factor")) if gold["class"] in ("preanalytical", "method_difference") else None),
               p4_next_gold=next_step(gold)[0], p4_next_said=nxt,
               p4_next_ok=int(nxt in next_ok_set(gold)),
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


def _credited(r: dict):
    """The class read by the pack composite. Without an applied acceptance-set change `p4_class_ok`
    means said == gold, so this is the said class; under one, a credited answer counts as the gold."""
    return r["p4_class_gold"] if r.get("p4_class_ok") else r.get("p4_class_said")


def pack_score(rows: list[dict], base_mae: float) -> dict:
    """Composite of one model on the pack from its rows (`row_atoms` keys)."""
    g = [r["p4_class_gold"] for r in rows]
    p = [_credited(r) for r in rows]
    c = cc(g, p)
    sk = skill([float(r["p4_pct_abs_err"]) for r in rows], base_mae)
    skc = max(-1.0, min(1.0, sk))
    by_an = {}
    for an in sorted({r.get("p4_analyte") for r in rows}):
        sub = [r for r in rows if r.get("p4_analyte") == an]
        by_an[an] = round(balanced_acc([r["p4_class_gold"] for r in sub], [_credited(r) for r in sub])[0], 4)
    fac = [r["p4_factor_ok"] for r in rows if r.get("p4_factor_ok") is not None]
    return {"n": len(rows), "change_source_cc": c, "change_source_ba": balanced_acc(g, p)[0],
            "true_change_skill": sk, "true_change_skill_clipped": skc,
            "composite": c,
            "profile": {"factor_hit": (sum(fac) / len(fac)) if fac else None,
                        "next_step_agree": sum(int(r.get("p4_next_ok") or 0) for r in rows) / len(rows),
                        "ba_by_analyte": by_an}}
