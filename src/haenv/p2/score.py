"""Rebuilt pack 2 · scoring by ordinal expected cost (design §4). Code-scored, no LLM judge.
P2_JUDGING segment; reads only the gold block `adjudication.triage_p2` and the parsed answer.

Per item: an answer inside the acceptable set costs 0; each step less urgent than the set's least
urgent member costs UNDER (under-triage); each step more urgent than its most urgent member costs
OVER (over-triage); no parseable disposition costs the item's worst case. Pack readings:

* `cost_cc` = 1 - sum(cost) / sum(expected cost of a uniform random answer on the same item)
  (design §4 formula; 1 = no cost, 0 = random, < 0 worse than random);
* `cost_skill` = 1 - sum(cost) / sum(cost of the best constant answer) (a constant policy scores
  <= 0; the headline metric, because the 3:1 ratio gives "always ed_now" a positive `cost_cc`);
* `under_rate`, `over_rate`, `in_set_rate`, separately;
* hard gate `missed_emergency` (set == {ed_now}, answered anything else) as a non-compensatory
  multiplier, as in the previous pack composite (cc x (1 - h), cc x (1 + h) below 0).
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

DISPOSITIONS = ("ed_now", "within_24h", "routine_followup", "watchful_waiting")
#: cost per step (design §4 draft 3:1, review: pending); pinned equal to the registry's `cost` by a test
UNDER, OVER = 3, 1
BLOCK = "triage_p2"

_ALIASES = (
    ("ed_now", ("ed_now", "ed now", "emergency", "立即急诊", "急诊")),
    ("within_24h", ("within_24h", "within 24h", "within 24 h", "24小时", "24 小时", "24h")),
    ("routine_followup", ("routine_followup", "routine follow", "routine", "常规复诊", "常规随访", "常规门诊")),
    ("watchful_waiting", ("watchful_waiting", "watchful", "居家观察", "继续观察", "观察")),
)
KEYS = ("p2_said", "p2_set", "p2_floor", "p2_cost", "p2_uniform_cost", "p2_under_steps", "p2_over_steps",
        "p2_in_set", "p2_under", "p2_over", "p2_missed_emergency", "p2_rule", "p2_rule_said_ok", "p2_tier",
        "p2_stratum", "p2_wording", "p2_borderline")


def parse_disposition(said) -> str | None:
    s = str(said or "").strip().lower()
    if not s:
        return None
    if s in DISPOSITIONS:
        return s
    ids = [d for d in DISPOSITIONS if d in s]
    if len(ids) == 1:
        return ids[0]
    hit = [d for d, words in _ALIASES if any(w in s for w in words)]
    return hit[0] if len(hit) == 1 else None


def item_cost(acc, said: str | None) -> tuple[float, int, int]:
    """(cost, under steps, over steps) of one answer against an acceptable set."""
    ranks = [DISPOSITIONS.index(d) for d in acc]
    lo, hi = min(ranks), max(ranks)                      # most urgent, least urgent acceptable
    if said is None:
        return max(item_cost(acc, d)[0] for d in DISPOSITIONS), 0, 0
    r = DISPOSITIONS.index(said)
    if r > hi:
        return float(UNDER * (r - hi)), r - hi, 0
    if r < lo:
        return float(OVER * (lo - r)), 0, lo - r
    return 0.0, 0, 0


def uniform_cost(acc) -> float:
    return sum(item_cost(acc, d)[0] for d in DISPOSITIONS) / len(DISPOSITIONS)


def atoms(answer: dict, gold: dict) -> dict:
    a = answer if isinstance(answer, dict) else {}
    out = {k: None for k in KEYS}
    if not isinstance(gold, dict) or not gold.get("set"):
        return out
    acc = list(gold["set"])
    said = parse_disposition(a.get("disposition"))
    cost, us, os_ = item_cost(acc, said)
    rs = str(a.get("rule") or "").strip().upper()
    out.update(p2_said=said, p2_set=acc, p2_floor=gold.get("floor"), p2_cost=cost, p2_uniform_cost=uniform_cost(acc),
               p2_under_steps=us, p2_over_steps=os_, p2_in_set=int(said is not None and cost == 0),
               p2_under=int(us > 0 or said is None), p2_over=int(os_ > 0),
               p2_missed_emergency=int(acc == ["ed_now"] and said != "ed_now"), p2_rule=gold.get("rule"),
               p2_rule_said_ok=(int(rs.split("_")[0] == str(gold.get("rule")).split("_")[0].upper()) if rs else None),
               p2_tier=gold.get("tier"), p2_stratum=gold.get("stratum"), p2_wording=gold.get("wording"),
               p2_borderline=int(len(acc) > 1))
    return out


def composite(rows: list[dict]) -> dict:
    """Pack readings over one solver's scored rows (aborted cells excluded, as everywhere in haenv)."""
    rows = [r for r in rows if r.get("p2_set") and not str(r.get("overall") or "").startswith("ABORT")]
    if not rows:
        return {"n": 0, "headline_metric": "cost_skill", "cost_cc": None, "cost_skill": None,
                "composite": None, "composite_cost_cc": None}
    n = len(rows)
    cost = sum(float(r["p2_cost"]) for r in rows)
    uni = sum(float(r["p2_uniform_cost"]) for r in rows)
    constants = {d: sum(item_cost(r["p2_set"], d)[0] for r in rows) for d in DISPOSITIONS}
    best = min(constants, key=constants.get)
    const = constants[best]
    cc = 1.0 - cost / uni if uni else None
    skill = 1.0 - cost / const if const else None
    hard = sum(int(r.get("p2_missed_emergency") or 0) for r in rows) / n

    def rate(k, sub=None):
        xs = [r[k] for r in (sub if sub is not None else rows) if r.get(k) is not None]
        return sum(xs) / len(xs) if xs else None

    border = [r for r in rows if r.get("p2_borderline")]
    single = [r for r in rows if not r.get("p2_borderline")]
    return {"n": n, "headline_metric": "cost_skill", "cost_mean": cost / n, "cost_cc": cc,
            "cost_skill": skill, "best_constant": best, "best_constant_cost": const,
            "hard_gate_rate": hard,
            "composite": (skill - hard * abs(skill)) if skill is not None else None,
            "composite_cost_cc": (cc - hard * abs(cc)) if cc is not None else None,
            "under_rate": rate("p2_under"), "over_rate": rate("p2_over"), "in_set_rate": rate("p2_in_set"),
            "in_set_single_gold": rate("p2_in_set", single), "in_set_borderline": rate("p2_in_set", border),
            "rule_cited_ok": rate("p2_rule_said_ok")}


def cc_cost(gold_sets: list, pred: list) -> float:
    """`cost_cc` of a list of answers against a list of acceptable sets (audit stubs, pseudo-models)."""
    rows = [atoms({"disposition": p}, {"set": s}) for s, p in zip(gold_sets, pred)]
    return composite(rows)["cost_cc"]
