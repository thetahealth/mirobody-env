"""Pack 2 grading table: NEWS2 (RCP 2017) and the disposition rule G2 (provisional, K3), read
from `registry/pack2_triage_thresholds.yaml`, plus the class edges the L2-4 margins are measured
from. PACK2_WORLD segment.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from ._util import registry_table

TABLE = "pack2_triage_thresholds.yaml"


DISPOSITIONS = ("ed_now", "within_24h", "routine_followup", "watchful_waiting")



def thresholds() -> dict:
    return registry_table(TABLE)


def _band_points(table, v) -> int:
    for lo, hi, pts in table:
        if lo <= v <= hi:
            return int(pts)
    raise ValueError(f"NEWS2: value {v!r} outside every band")


def news2(v: dict, tb: dict | None = None) -> tuple[int, bool]:
    """NEWS2 aggregate (RCP 2017, SpO2 scale 1, room air) and whether any parameter scores 3."""
    tb = tb or thresholds()
    n = tb["news2"]
    parts = [_band_points(n["rr"], v["rr"]), _band_points(n["spo2"], v["spo2"]),
             _band_points(n["sbp"], v["sbp"]), _band_points(n["hr"], v["hr"]),
             _band_points(n["temp"], v["temp"]), 0 if v.get("alert", True) else int(n["not_alert_points"])]
    return sum(parts), any(p == 3 for p in parts)


def g2(v: dict, tb: dict | None = None) -> str:
    """Grading rule G2 (provisional), first match wins. `v`: rr, spo2, sbp, hr, temp, alert,
    glucose (point of care now), K, glucose_log (72 h capillary readings)."""
    tb = tb or thresholds()
    r = tb["rules"]
    s, single3 = news2(v, tb)
    lows = sum(1 for x in v.get("glucose_log") or () if x < r["glucose_low"])
    e, d, f = r["ed_now"], r["within_24h"], r["routine_followup"]
    if (s >= e["news2_ge"] or v["glucose"] < e["glucose_lt"] or v["K"] >= e["k_ge"]
            or v["sbp"] >= e["sbp_ge"] or v["sbp"] <= e["sbp_le"] or v["spo2"] <= e["spo2_le"]):
        return "ed_now"
    if (s >= d["news2_ge"] or (d.get("single_param_3") and single3)
            or d["sbp_range"][0] <= v["sbp"] <= d["sbp_range"][1]
            or d["k_range"][0] <= v["K"] <= d["k_range"][1]
            or lows >= d["glucose_lows_72h_ge"]):
        return "within_24h"
    if (s >= f["news2_ge"] or f["k_range"][0] <= v["K"] <= f["k_range"][1]
            or f["sbp_range"][0] <= v["sbp"] <= f["sbp_range"][1]
            or lows == f["glucose_lows_72h_eq"]):
        return "routine_followup"
    return "watchful_waiting"


def urgency_of(disp: str) -> str:
    return {d["id"]: d["urgency"] for d in thresholds()["dispositions"]}[disp]


def rank_of(disp: str) -> int:
    return DISPOSITIONS.index(disp)


#: G2 / NEWS2 class edges per quantity, on the half-unit between the last value of one class and
#: the first of the next. Distance in printed units = |v - edge| / unit - 0.5.
def class_edges(tb: dict | None = None) -> dict[str, list[float]]:
    tb = tb or thresholds()
    u = tb["margins"]["units"]
    n, r = tb["news2"], tb["rules"]
    out: dict[str, set] = {}
    for q in ("rr", "spo2", "sbp", "hr", "temp"):
        for lo, hi, _ in n[q]:
            for x in (lo, hi):
                if abs(x) < 900:
                    out.setdefault(q, set()).add(round(x + (u[q] / 2 if x == hi else -u[q] / 2), 3))
    sb = out.setdefault("sbp", set())
    for lo, hi in (r["within_24h"]["sbp_range"], r["routine_followup"]["sbp_range"]):
        sb |= {round(lo - 0.5, 3), round(hi + 0.5, 3)}
    sb |= {round(r["ed_now"]["sbp_le"] + 0.5, 3), round(r["ed_now"]["sbp_ge"] - 0.5, 3)}
    out.setdefault("spo2", set()).add(round(r["ed_now"]["spo2_le"] + 0.5, 3))
    ku = u["K"] / 2
    out["K"] = {round(r["ed_now"]["k_ge"] - ku, 3), round(r["within_24h"]["k_range"][0] - ku, 3),
                round(r["routine_followup"]["k_range"][0] - ku, 3)}
    gu = u["glucose"] / 2
    out["glucose"] = {round(r["ed_now"]["glucose_lt"] - gu, 3), round(r["glucose_low"] - gu, 3)}
    return {k: sorted(v) for k, v in out.items()}


def margin_units(q: str, v: float, tb: dict | None = None) -> float:
    tb = tb or thresholds()
    u = tb["margins"]["units"][q]
    edges = class_edges(tb)[q]
    return round(min(abs(v - e) for e in edges) / u - 0.5, 6)

