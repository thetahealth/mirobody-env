"""Outcome family: forecast / driver / alternative.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("haenv.judges")

from ._helpers import _gold  # noqa: F401

def judge_forecast(out, vp, ctx) -> dict:
    """Early-warning dimension: direction and Brier score.

    An abstention (`indeterminate`, as the prompt allows) is not a wrong direction:
    `direction_ok = None` (excluded from `dir_acc`), `abstained = True`; report the abstention
    rate alongside `dir_acc`. `low` is a direction, not an abstention.
    """
    y = 1.0 if _gold(vp, "outcome_label") == "event_occurred" else 0.0
    # A missing `risk` stays `None` (as does `brier`), distinct from a stated 0.0.
    _risk_raw = out.forecast.get("risk")
    try:
        risk = None if _risk_raw is None else float(_risk_raw)
    except (TypeError, ValueError):
        risk = None
    cat = out.forecast.get("risk_category")
    abstained = cat == "indeterminate"
    if abstained:
        ok = None
    else:
        ok = (y == 1 and cat in ("elevated", "high")) or (y == 0 and cat == "low")
    return {"y": y, "risk": (round(risk, 3) if risk is not None else None),
            "risk_absent": (risk is None),
            "risk_category": cat,
            "brier": (round((risk - y) ** 2, 4) if risk is not None else None),
            "direction_ok": ok, "abstained": abstained}


def judge_driver(out, vp, ctx) -> dict:
    drv = out.drivers[0].get("driver") if out.drivers else None
    gold = _gold(vp, "gold_drivers")
    return {"top_driver": drv, "driver_hit": bool(drv and drv in (gold or [])),
            "gold_drivers": gold,
            "all_drivers": [d.get("driver") for d in (out.drivers or [])]}


def judge_alternative(out, vp, ctx) -> dict:
    from ..tracks import alternative_a1
    return alternative_a1(out)
