"""Pack 2's background layer: the things a real patient's record carries that no printed rule reads.

The data (factors, rates, where each cannot act, wording pools, sources) is the `background:` section of
`registry/pack2_triage_scenarios.yaml`. This adapter only reads it: which factors an item carries
(`active`), which wearable fields a day did not record (`missing_fields`), the single-reading
decoy (`edit_single_deep_night`) and the sentences the patient says (`says`). Every draw is a sha256 of the case id and
the factor id, blind to the tier and the floor class. Nothing here decides a gold: the rule reader
(`rules.assess`) re-reads the shown block, and `render.realize` re-draws until it fires the planned rule.
P2_WORLD segment. SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib

from haenv.p2 import rules as R


def u01(*parts) -> float:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:13], 16) / float(1 << 52)


def spec() -> dict:
    return R.table()["background"]


def on(plan: dict | None) -> bool:
    """The layer is on unless the plan says `background: off` (set by `tools/generation/p2_gen_job.py --background off`)."""
    return (plan or {}).get("background", "on") != "off"


def strap_notes(plan: dict | None) -> list[str]:
    """The loose-strap phrasings: the three the pack always had, plus the added ones when the layer is on."""
    return pool("strap_notes_base") + (pool("strap_notes_extra") if on(plan) else [])


def missing_fields(cid: str, d: int, T: int) -> tuple[str, ...]:
    """Wearable fields not recorded on day d: a draw per field and day at LifeSnaps' non-wear rate."""
    m = spec()["wearable_missing"]
    keep = m.get("recorded_last_days", {})
    return tuple(f for f, r in m["rates"].items() if u01("p2-bg-miss", f, cid, d) < r and T - d >= keep.get(f, 0))


def pool(name: str) -> list[str]:
    return list(spec()["pools"][name])


def pick(items, *key):
    items = list(items)
    return items[int(u01(*key) * len(items)) % len(items)]


def _span(rng, *key) -> float:
    lo, hi = rng
    return lo + (hi - lo) * u01(*key)


def applies(f: dict, tier: str, scenario: str, known: set, cgm: bool, mild: bool, lows: bool = False) -> bool:
    """Whether the factor may sit on this item at all: where a rule would read it, it cannot."""
    w = f.get("applies_when") or {}
    if tier in w.get("tier_not", ()) or scenario in w.get("scenario_not", ()):
        return False
    if known & set(w.get("known_not", ())):
        return False
    if w.get("known_any") and not known & set(w["known_any"]):
        return False
    if ("cgm" in w.get("needs", ()) and not cgm) or ("lows" in w.get("needs", ()) and not lows):
        return False
    return not ("mild_final" in w and bool(w["mild_final"]) != mild)


def active(cid: str, plan: dict, known: set, cgm: bool, mild: bool, lows: bool = False) -> list[str]:
    """Factor ids this item carries. `plan["bg"]` (a list) replaces the draw: tests and probes."""
    if not on(plan):
        return []
    tier = plan["tier"]
    sc = R.table()["tiers"][tier]["scenario"]
    forced = plan.get("bg")      # the generator's stratified draw (`tools/p2_gen_job.stratify_background`), or a test's
    out = []
    for fid, f in spec()["factors"].items():
        chosen = fid in forced if forced is not None else u01("p2-bg", fid, cid) < f["rate"]
        if chosen and applies(f, tier, sc, known, cgm, mild, lows):
            out.append(fid)
    return out


def note_day(cid: str, T: int, window: int) -> int:
    """The one day of the window a plain watch note sits on."""
    return T - window + 1 + int(u01("p2-bg-noteday", cid) * window)


def says(cid: str, ids: list[str]) -> list[str]:
    """The patient's sentences of the active factors (each at its own `say_rate`)."""
    out = []
    for fid in ids:
        f = spec()["factors"][fid]
        if f.get("say") and u01("p2-bg-say", fid, cid) < f.get("say_rate", 0):
            out.append(pick(f["say"], "p2-bg-say-k", fid, cid))
    return out


# ---- decoy: one reading that no rule counts (the edit amounts are the factor's data)

def edit_single_deep_night(cid: str, cgm: list[dict]) -> None:
    e = spec()["factors"]["single_deep_night"]["edit"]
    r = cgm[int(u01("p2-bg-night", cid) * len(cgm))]
    r["min_mmol_L"] = round(_span(e["min_mmol_L"], "p2-bg-min", cid), 1)
    r["minutes_below_3_9"] = int(round(_span(e["minutes_below"], "p2-bg-mins", cid)))
