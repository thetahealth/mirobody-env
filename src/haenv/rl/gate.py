"""The rl emission declaration for the shared gate (`haenv/world/validation/shared_audit.py`, SA-1 + SA-8), with
the pack's three item gates:

* G1 single red line -- `rules.sweep(facts)` equals `[rule_id]` on the redline arm and `[]` on the control arm;
  the facts include the shared world's background medicines and the results the record shows.
* G2 question independence -- the question names no ingredient, class, alias, test or analyte of any
  registered rule, nor a generic medicine / result word (`redline_questions.yaml: generic_terms`).
* G3 answerable -- the gold re-derived from the rendered record alone (`render.render_text` ->
  `render.parse_text` -> `realize.facts_from_rows` -> `rules.fires`) equals the gold; on the redline arm the
  trigger date and value appear in the text.

The pair gates G4 (one-patch pair) and G5 (paired imminence) read both arms and live in `audit.py`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import datetime
import re

GATE = "rl"
BLOCK = "redline_rl"
#: gold-side literals that never appear in what the solver sees
WORDS = ("answer_and_raise", "answer_only", "red line", "red-line", "redline", "RAISED_ROUTED",
         "RAISED_WEAK", "NOT_RAISED", "FALSE_ALARM", "must_flag", "rule_id", "planted")


def plan_of(raw) -> dict | None:
    from haenv.contracts.external_gold import SLOT
    lp = raw.get("latent_premise") if isinstance(raw, dict) else getattr(raw, "latent_premise", None)
    p = (((lp or {}).get("meta") or {}).get(SLOT) or {}).get("rl")
    return p if isinstance(p, dict) else None


def _unrealizable(g: dict) -> str | None:
    if g.get("unrealizable") or "facts" not in g:
        return str(g.get("unrealizable") or "no realized item")
    return None


def g1(g: dict) -> list[tuple[str, str]]:
    want = [g["rule_id"]] if g.get("variant") == "redline" else []
    got = list(g.get("sweep") or [])
    from haenv.rl.rules import sweep
    now = sweep(g.get("facts") or {})
    bad = []
    if now != got:
        bad.append(("g1_sweep_stale", f"stored {got} != recomputed {now}"))
    if now != want:
        bad.append(("g1_single_red_line", f"sweep {now} != {want}"))
    return bad


def question_hits(question: str) -> list[str]:
    from haenv.rl.plan import questions_table
    from haenv.rl.rules import item_terms, load
    from haenv.rl import zh as Z
    terms = {x for r in load().values() for x in item_terms(r)}
    terms |= set(questions_table().get("generic_terms") or ())
    terms |= {x for r in load().values() for x in Z.item_terms(r)} | set(Z.generic_terms())   # a zh question
    q = question or ""
    return sorted({t for t in terms if re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", q, re.I)},
                  key=str.lower)


def g2(g: dict, block: dict) -> list[tuple[str, str]]:
    hits = question_hits((block or {}).get("question", ""))
    return [("g2_question_names", ", ".join(hits))] if hits else []


def visible_facts(block: dict) -> dict:
    from haenv.rl import realize as RZ
    from haenv.rl import render as RD
    from haenv.rl.rules import load
    parsed = RD.parse_text(RD.render_text(block))
    today = datetime.date.fromisoformat(block["today"])
    T = 0                                                        # days relative to today
    base = {k: parsed.get(k) for k in ("age", "sex", "weight_kg", "ancestry")}
    return RZ.facts_from_rows(load(), parsed, T, today, base)


def g3(g: dict, block: dict) -> list[tuple[str, str]]:
    from haenv.rl import render as RD
    from haenv.rl.rules import fires, rule
    if not isinstance(block, dict) or "today" not in block:
        return [("g3_no_block", "visible block missing")]
    r = rule(g["rule_id"])
    vf = visible_facts(block)
    want = g.get("gold")
    got = "answer_and_raise" if fires(r, vf) else "answer_only"
    bad = [] if got == want else [("g3_visible_gold", f"re-derived {got} != gold {want}")]
    text = RD.render_text(block)
    trig = g.get("trigger") or {}
    if g.get("variant") == "redline" and trig and (trig["date"][:7] not in text or str(trig["value"]) not in text):
        bad.append(("g3_trigger_not_visible", f"{trig}"))
    return bad


def extra(g: dict, raw) -> list[tuple[str, str]]:
    get = (lambda k: raw.get(k)) if isinstance(raw, dict) else (lambda k: getattr(raw, k, None))
    block = (get("prediction_context") or {}).get(BLOCK)
    gold_planned = {"redline": "answer_and_raise", "control": "answer_only"}.get(g.get("variant"))
    out = []
    if g.get("gold") != gold_planned:
        out.append(("gold_not_planned", f"{g.get('gold')} for arm {g.get('variant')}"))
    return out + g1(g) + g2(g, block) + g3(g, block)


def _recompute(g: dict, raw) -> list[str]:
    from haenv.rl.rules import gold_of, rule
    if gold_of(rule(g["rule_id"]), g.get("facts") or {}) != g.get("gold"):
        return ["gold differs from the rule predicate over the stored facts"]
    return []


def emission():
    from haenv.world.validation.shared_audit import Emission
    return Emission(gate=GATE, block=BLOCK, plan_of=plan_of, recompute=_recompute,
                    class_of=lambda g: g.get("gold"),
                    planned_class=lambda p: {"redline": "answer_and_raise", "control": "answer_only"}.get(p.get("variant")),
                    words=WORDS, unrealizable=_unrealizable, extra=extra)
