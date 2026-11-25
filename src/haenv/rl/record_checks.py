"""Generation-time checks on the patient-visible rl record."""
from __future__ import annotations

import re
from datetime import date, timedelta

from haenv.rl import realize as RZ
from haenv.rl import render as RD
from haenv.rl import rules as R
from haenv.rl import zh as Z

_DATE = r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)"
_WEEKDAY = r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?\b|(?:周|星期|礼拜)[一二三四五六日天](?!次)"
_NAMED_DAY = re.compile(_WEEKDAY, re.I)
_DATE_PAIRS = (
    re.compile(fr"(?P<date>{_DATE})[ \t,，:：\(（]*(?:on )?(?P<weekday>{_WEEKDAY})", re.I),
    re.compile(fr"(?P<weekday>{_WEEKDAY})[ \t,，\(（]*(?:on )?(?P<date>{_DATE})", re.I),
)
_CBC = re.compile(r"\b(?:blood count(?:s)?|CBC|FBC)\b|血常规|血细胞计数", re.I)


def _check_weekday(iso: str, name: str) -> None:
    day = date.fromisoformat(iso)
    if name.isascii():
        expected = RD.weekday_name(day, "en").lower()
        actual = name.lower().removesuffix("s")
    else:
        expected = RD.weekday_name(day, "zh")[-1]
        actual = name[-1].replace("天", "日")
    if actual != expected:
        raise RZ.Unrealizable(f"weekday {name} disagrees with {iso}")


def _check_weekdays(text, parsed, question, rule, carrier):
    for line in (*text.splitlines(), question):
        for pattern in _DATE_PAIRS:
            for match in pattern.finditer(line):
                _check_weekday(match["date"], match["weekday"])
    for entry in parsed["planned"]:
        if match := re.match(fr"({_DATE}): (?:Next dose due:|下次给药：)", entry):
            for name in _NAMED_DAY.findall(entry):
                _check_weekday(match[1], name)
    # In the start-day conversation the patient says the first dose is today.
    if "{weekday}" in rule["drug"]["dose_text"] and rule["family"] == "F":
        start = carrier["slots"]["start"]
        for entry in parsed["conversation"]:
            if entry["date"] == start:
                names = _NAMED_DAY.findall(entry["assistant"])
                if len(names) != 2:
                    raise RZ.Unrealizable("start-day conversation is missing the weekly dosing days")
                _check_weekday(start, names[0])
                _check_weekday((date.fromisoformat(start) + timedelta(days=3)).isoformat(), names[1])


def _check_problems(parsed, today, dated_problems):
    notes = [(v["date"], f'{v["type"]} {v["summary"]}') for v in parsed["visits"]]
    for p in parsed["problem_list"]:
        match = re.search(_DATE, p)
        notes.append((match[0] if match else today, p))
    for problem, since in dated_problems.items():
        name = re.split(r"[（(]", problem, maxsplit=1)[0].rstrip()
        pattern = re.compile(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", re.I)
        for day, text in notes:
            if day < since and pattern.search(text):
                raise RZ.Unrealizable(f"{problem} mentioned on {day} before its first supported date {since}")


def _check_c_control(parsed, trigger):
    if trigger.get("section") == "home_monitoring":
        return  # the home monitor has no reported reference range or lab flag
    analyte = Z.ph(trigger["display"])
    rows = [r for r in parsed["results"] if r["analyte"] == analyte]
    if not rows:
        raise RZ.Unrealizable(f"control {analyte} series missing")
    for row in rows:
        value = RZ._num(row.get("value"))
        if value is None or RZ._flag(value, trigger.get("ref_low"), trigger.get("ref_high")) or row.get("flag"):
            raise RZ.Unrealizable(f"control {analyte} outside its reference range on {row['date']}")


def _check_baseline(parsed, rule, carrier, tests):
    baseline = carrier["baseline_date"]
    if baseline >= carrier["slots"]["start"]:
        raise RZ.Unrealizable("baseline draw is not before starting the medicine")
    rows = {r["analyte"]: r for r in parsed["results"] if r["date"] == baseline and RZ._num(r.get("value")) is not None}
    for test in tests:
        spec = R.display()["world_results"][test]
        unit = spec[RZ._units(rule)]
        row = rows.get(Z.ph(spec["display"]))
        if not row or row.get("flag") or RZ._flag(float(row["value"]), unit["low"], unit["high"]):
            raise RZ.Unrealizable(f"missing normal baseline {test} on {baseline}")
    chem = {Z.ph(sp["display"]) for sp in RZ.panel_specs(rule, "chem")}
    if not chem <= rows.keys():
        raise RZ.Unrealizable(f"incomplete baseline chemistry on {baseline}")


def validate(block, rule, variant, dated_problems, carrier):
    text = RD.render_text(block)
    parsed = RD.parse_text(text)
    _check_weekdays(text, parsed, block["question"], rule, carrier)
    _check_problems(parsed, block["today"], dated_problems)
    if rule["family"] == "C" and variant == "control":
        _check_c_control(parsed, rule["trigger"])
    if rule["id"] == "RL-B-febuxostat-azathioprine" and variant == "redline":
        for visit in carrier["patch_visits"]:
            if _CBC.search(visit["summary"]):
                raise RZ.Unrealizable("azathioprine carrier visit adds a CBC commitment")
    if tests := (rule.get("followup") or {}).get("baseline_tests"):
        _check_baseline(parsed, rule, carrier, tests)
