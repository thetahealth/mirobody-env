"""Biological-variation audit of an rl record (every family, both arms).

A result series whose newest value moves from the patient's own recent mean by more than the reference
change value (RCV) of that analyte is a second signal in the record. It is allowed only when the item's rule
designs that analyte (the C trend, the G conversion, the E value status on the redline arm); otherwise the
item is refused (`realize.realize`, Unrealizable) and the pack gate `BV` (`audit.py`) re-checks the pool.

RCV (two-sided 95%) = 1.96 x sqrt(2) x sqrt(CVI^2 + CVA^2), with CVI / CVA from
`registry/biological_variation.yaml` (EFLM biological variation database, EuBIVAS). eGFR (CKD-EPI 2021,
creatinine^-1.2 above the knot) moves 1.2 times the relative creatinine change, so its bound is 1.2 x the
creatinine RCV. The recent mean is the mean of up to RECENT earlier values; series with fewer than
MIN_PRIOR earlier values are not judged.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import math
import re

RECENT = 3
#: analytes the rl realizer draws itself (chemistry, liver panel, required tests): the gate refuses on these.
#: Other bounded analytes (HbA1c, glucose, lipids, thyroid) come from the shared world's disease course,
#: identical in both arms; they are reported (`moves`), not gated.
GATED = ("creatinine", "egfr", "crcl", "potassium", "alt", "ast")
MIN_PRIOR = 2
#: rule-trigger analyte names -> the canonical ids of the bounds table
ALIASES = {"creatinine_umol": "creatinine", "home_glucose": "glucose", "fasting_glucose": "glucose",
           "hba1c": "hba1c", "free t4": "ft4"}
LINKED = ({"creatinine", "egfr", "crcl"},)


@functools.lru_cache(maxsize=1)
def bounds() -> dict[str, float]:
    """Relative RCV per canonical analyte id."""
    from haenv.resources.regpath import load_registry
    bv = (load_registry("biological_variation.yaml") or {}).get("analytes") or {}
    out = {}
    for k, v in bv.items():
        cvi, cva = float(v.get("cvi") or 0.0), float(v.get("cva") or 0.0)
        if cvi:
            out[k.lower()] = 1.96 * math.sqrt(2.0) * math.sqrt(cvi * cvi + cva * cva)
    if "creatinine" in out:
        out["egfr"] = 1.2 * out["creatinine"]
    return out


def canon(name: str) -> str:
    n = str(name or "").strip().lower()
    return ALIASES.get(n, n)


@functools.lru_cache(maxsize=1)
def display_ids() -> dict[str, str]:
    """Rendered result display (en and zh) -> canonical analyte id."""
    from haenv.rl import zh as Z
    from haenv.rl.realize import _lib
    from haenv.rl.rules import display, load
    out: dict[str, str] = {}

    def put(disp, aid):
        for d in Z.both(disp):
            out.setdefault(d, canon(aid))
    for name, units in (_lib().get("panels") or {}).items():
        for sps in units.values():
            for sp in sps:
                put(sp["display"], sp["id"])
    for k, sp in (display().get("tests") or {}).items():
        put(sp["display"], k)
    for k, sp in (display().get("world_results") or {}).items():
        put(sp["display"], k)
    for r in load().values():
        t = r["trigger"]
        if t.get("analyte") and t.get("display"):
            put(t["display"], t["analyte"])
    return out


def designed(rule: dict, variant: str) -> set[str]:
    """The analytes the item's rule is allowed to move."""
    t = rule["trigger"]
    k = t["kind"]
    out: set[str] = set()
    if k == "threshold_trend":
        out.add(canon(t["analyte"]))
        out |= {canon(c["analyte"]) for c in t.get("companions") or ()}
    elif k == "converted_low_reading":
        out.add(canon(t["analyte"]))
    elif k in ("converted_creatinine_egfr", "converted_weight_dose"):
        out |= {"creatinine", "egfr"}
    elif k == "required_result_missing" and variant == "redline":
        if t.get("crcl_min") is not None:
            out |= {"creatinine", "egfr", "crcl"}
        out |= {canon(a) for a in t.get("limits") or {}}
    for group in LINKED:
        if out & group:
            out |= group
    return out


def _num(row) -> float | None:
    v = row.get("value") if row.get("value") is not None else row.get("result")
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(v or ""))
    return float(m.group(1)) if m else None


def moves(results) -> list[dict]:
    """Every series whose newest value is outside the RCV of its recent mean."""
    ids, bd = display_ids(), bounds()
    series: dict[str, list[tuple[str, float]]] = {}
    for row in results or ():
        aid = ids.get(row.get("analyte"))
        v = _num(row)
        if aid in bd and v is not None and row.get("date"):
            series.setdefault((aid, row["analyte"]), []).append((str(row["date"]), v))
    out = []
    for (aid, disp), pts in series.items():
        pts.sort()
        if len(pts) < MIN_PRIOR + 1:
            continue
        prior = [v for _, v in pts[-1 - RECENT:-1]]
        mean = sum(prior) / len(prior)
        last = pts[-1][1]
        if mean > 0 and abs(last - mean) / mean > bd[aid]:
            out.append({"analyte": aid, "display": disp, "date": pts[-1][0], "mean": round(mean, 3),
                        "last": last, "rel": round((last - mean) / mean, 3), "bound": round(bd[aid], 3)})
    return out


def violations(block: dict, rule: dict, variant: str) -> list[dict]:
    allow = designed(rule, variant)
    return [m for m in moves(block.get("results")) if m["analyte"] in GATED and m["analyte"] not in allow]
