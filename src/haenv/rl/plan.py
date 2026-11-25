"""Case-keyed allocation of the rl pack: which rule a case carries, its question and its imminence.

Every draw that places text is keyed on (case id, rule id) through `draw`, never on the arm (`variant`), so
both arms of a pair read the same world, the same question, the same dates and the same ordinary content.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import datetime
import functools
import hashlib

LATENT = "rl"
VARIANTS = ("redline", "control")
GOLD = {"redline": "answer_and_raise", "control": "answer_only"}
QUESTIONS = "redline_questions.yaml"
#: the calendar span the record's "today" is drawn from (ours): keeps every date a real date
TODAY_LO = datetime.date(2025, 3, 1)
TODAY_SPAN_DAYS = 600


def draw(case_id: str, rule_id: str, *name) -> float:
    """[0, 1) keyed on the case id, the rule and the draw's name (never on the arm)."""
    import haenv.world.generation.rng as rng
    return rng.unit(str(case_id), "rl", str(rule_id), *map(str, name))


def pick(seq, case_id: str, rule_id: str, *name):
    seq = list(seq)
    return seq[int(draw(case_id, rule_id, *name) * len(seq))] if seq else None


def uniform(lo, hi, case_id: str, rule_id: str, *name) -> float:
    return float(lo) + (float(hi) - float(lo)) * draw(case_id, rule_id, *name)


def randint(lo, hi, case_id: str, rule_id: str, *name) -> int:
    """An integer in [lo, hi] inclusive."""
    return int(lo) + int(draw(case_id, rule_id, *name) * (int(hi) - int(lo) + 1))


def h01(tag: str, case_id: str) -> float:
    return int(hashlib.sha256(f"{tag}|{case_id}".encode()).hexdigest()[:12], 16) / 16 ** 12


@functools.lru_cache(maxsize=None)
def questions_table() -> dict:
    from haenv.resources.regpath import load_registry
    return load_registry(QUESTIONS) or {}


def eligible_questions(age: int | None, sex: str | None, conditions) -> list[dict]:
    cond = set(conditions or ())
    out = []
    for q in questions_table().get("questions") or ():
        if q.get("min_age") is not None and (age is None or age < int(q["min_age"])):
            continue
        if q.get("max_age") is not None and (age is None or age > int(q["max_age"])):
            continue
        if q.get("sex") and q["sex"] != sex:
            continue
        if q.get("condition") and q["condition"] not in cond:
            continue
        out.append(q)
    return out


def question(qid: str) -> dict:
    return next(q for q in questions_table().get("questions") or () if q["id"] == qid)


def band_age(age_range: str, case_id: str, rule_id: str) -> int:
    lo, hi = (int(x) for x in str(age_range).split("-", 1))
    return randint(lo, hi, case_id, rule_id, "age")


def today_date(case_id: str) -> datetime.date:
    """The calendar date of day T: keyed on the case id alone (the same in both arms and for any rule)."""
    import haenv.world.generation.rng as rng
    return TODAY_LO + datetime.timedelta(days=int(rng.unit(str(case_id), "rl", "today") * TODAY_SPAN_DAYS))


def date_of(day: int, T: int, case_id: str) -> datetime.date:
    return today_date(case_id) + datetime.timedelta(days=int(day) - int(T))


def iso(day: int, T: int, case_id: str) -> str:
    return date_of(day, T, case_id).isoformat()


def make_plan(case_id: str, rule: dict, age: int, sex: str, conditions, seed_tag: str) -> dict:
    """The rl knob of one case (arm `redline`; `make_pack --paired-with` writes the control arm)."""
    rid = rule["id"]
    qs = [q for q in eligible_questions(age, sex, conditions) if q["id"] not in set(rule.get("exclude_questions") or ())]
    if not qs:
        raise ValueError(f"no eligible question for {case_id}")
    q = pick(qs, case_id, rid, "question")
    lo, hi = rule["imminence"]["days_ahead"]
    return {"rule_id": rid, "family": rule["family"], "variant": "redline", "question_id": q["id"],
            "imminence_day": randint(lo, hi, case_id, rid, "imminence"), "seed_salt": seed_tag}
