"""M7 authored composition from hidden clinical premises, never reader scores.

SYNTHETIC data, evaluation only, not medical advice.
"""
from collections import Counter

from haenv.resources.regpath import load_registry


def tag(g):
    from haenv.world.truth import coupling_enabled
    from haenv.world.validation.difficulty_r2 import enabled
    if not coupling_enabled() or not enabled(g.get("plan") or g):
        return
    from haenv.p3.gold import g3
    result = g3(g)
    decisive = set(result["decisive"])
    mechanisms = []
    if decisive & {"dose_limit", "contraindication"}:
        mechanisms.append("history")
    if g.get("dispensing_history") and ("coverage" in decisive or
            (result["control"] == "off" and decisive & {"headroom", "interval"}
             and result["class"] != "check_adherence_or_adverse_effect")):
        from haenv.world.labworld.meds import rule
        line = float(rule()["adherence_line"])
        # Actual recent ingestion and window coverage belong to the hidden course.
        if (g["a_last"] < line) != (result["a_T"] < line):
            mechanisms.append("M1")
    dose = g.get("dose_reconciliation") or {}
    if dose.get("prescription_dose", result["dose"]) != result["dose"] and decisive & {"headroom", "dose_limit"}:
        mechanisms.append("M2")
    if g.get("control_measurements") and "control" in decisive:
        from haenv.p3.gold import control_status
        latest = max(g["control_measurements"], key=lambda r: (int(r["day"]), int(r["sequence"])))
        status, _ = control_status(g["axis"], latest["value"], g.get("upper"))
        if status != result["control"]:
            mechanisms.append("M3")
    if g.get("ae_causality") and any(p.startswith("causal_") for p in decisive):
        mechanisms.append("M4")
    # R1/R2 attach to items that another mechanism already makes core. The conflict changes which
    # premise decides the item, not whether the item is core, so the authored 70 percent core share
    # and the clinical quota cells are unchanged; a conflict on an otherwise-anchor item stays
    # untagged and the item remains an anchor. Without this restriction the structural core floor
    # rises above the authored target and the quota allocator refuses the pool.
    from haenv.p3.gold import historical_limit
    conflict = ((result["class"] != "switch" and result["control"] == "off" and "dose_limit" not in decisive
                 and historical_limit(g, result["dose"]))
                or (result["ae"] == "dose_related" and "adh_low" in result["states"]))
    if mechanisms and conflict:
        mechanisms.append("R1")
    # R2: an inactive near-miss archive on an off-target item whose gold is not switch, where a
    # shortcut that reads any archived episode as a limit would answer switch. It attaches only to an
    # already-core item (as R1 does), so the authored core share and the class cells do not move.
    if mechanisms and g.get("history_near_miss") and result["control"] == "off" and result["class"] != "switch":
        mechanisms.append("R2")
    g["composition"] = "core" if mechanisms else "anchor"
    g["composition_mechanisms"] = mechanisms


def core_quotas(cells, items, cell_of, base, n, *, final=True):
    """Allocate the authored global share inside unchanged clinical quota cells.

    Structural anchor-only cells retain their clinical quota. At the published size the largest
    remainder spends the discretionary core over the remaining capacity; an inner nested size keeps
    only its forced floor, so no early core raises the published size's floor. Missing supply is an
    error, never a reason to borrow a clinical cell or consult a reader.
    """
    from haenv.world.validation.pack_size import apportion
    share = load_registry("p3_composition.yaml")["coverage"]["published_core_share"]
    have = Counter((cell_of(it), it["gold"]["composition"]) for it in base)
    supply = Counter((cell_of(it), it["gold"]["composition"]) for it in items)
    low, high = {}, {}
    for cell, total in cells.items():
        low[cell] = max(have[cell, "core"], total - supply[cell, "anchor"])
        high[cell] = min(total - have[cell, "anchor"], supply[cell, "core"])
        if low[cell] > high[cell]:
            raise SystemExit(f"[pack] P3 composition supply short in {cell}")
    target = round(n * share)
    if not sum(low.values()) <= target <= sum(high.values()):
        raise SystemExit(f"[pack] P3 authored core target {target} outside supply [{sum(low.values())}, {sum(high.values())}]")
    # Only the published size spends discretionary core. An inner (nested-prefix) size carries its
    # forced floor alone: core spent early lands in cells the prefix still fills with anchors, and
    # monotone nesting would then force those cells at the published size too, pushing its floor
    # past the authored target. The authored 70 percent applies to the published pack.
    if not final:
        return dict(low)
    want = target - sum(low.values())
    if want < 0 or want > sum(high[c] - low[c] for c in cells):
        raise SystemExit(f"[pack] P3 authored core target {target} outside supply [{sum(low.values())}, "
                         f"{sum(high.values())}]")
    extra = apportion(want, {c: high[c] - low[c] for c in cells}) if want else dict.fromkeys(cells, 0)
    out = {c: low[c] + extra[c] for c in cells}
    if sum(out.values()) != target:
        raise SystemExit(f"[pack] P3 composition allocated {sum(out.values())} != target {target}")
    return out
