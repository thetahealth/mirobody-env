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

The source class is scored against `accepted_classes(gold, accept)`. The 1.4 default
applies P4-AC-001 to every registered ingested-substance assay interference mechanism.
An explicit empty `accept` tuple replays legacy single-class scoring. The canonical
gold class stays unchanged; next steps remain independently required by decision D.

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

# Compatibility catalog for consumers that list the registered cells. The rule itself
# is a mechanism-family predicate, so adding a sourced interference needs no code cell.
from haenv.p4.gold import contract_table as _contract
from haenv.p4.diversity import table as _mechanisms
ACCEPTED_CLASS_CHANGES = {
    (entry["gold_class"], factor): {"id": change_id, **entry}
    for change_id, entry in _contract()["acceptance_changes"].items()
    for factor, rule in _mechanisms()["rules"].items()
    if rule.get("mechanism_family") == entry["mechanism_family"]
}


def accepted_classes(gold: dict, accept: tuple | None = None) -> tuple:
    """Credit registered equivalence by default; explicit `()` replays the legacy rule."""
    from haenv.p4.gold import contract_table as table, ingested_interference
    changes = table()["acceptance_changes"]
    enabled = tuple(k for k, v in changes.items() if v["default"]) if accept is None else accept
    unknown = set(enabled) - changes.keys()
    if unknown:
        raise ValueError(f"unregistered acceptance-set change(s) {sorted(unknown)}; see ACCEPTED_CLASS_CHANGES")
    g = gold if isinstance(gold, dict) else {}
    entry = changes["P4-AC-001"]
    if ("P4-AC-001" in enabled and g.get("class") == entry["gold_class"] and
            ingested_interference(g.get("factor"))):
        return tuple(entry["accept"])
    return (g.get("class"),)


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


def row_atoms(answer: dict, gold: dict, sha: str = "", accept: tuple | None = None) -> dict:
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
