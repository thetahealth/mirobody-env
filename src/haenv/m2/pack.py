"""Pack ①'s proportion table: how many items of each kind a pack of N items holds.

The bridge cells (F0) take their proportional share up to what the fixed v1.0.1 bridge set supplies;
the bridge is a frozen control set, so a shortfall is filled by new F1-F3 items in proportion.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

#: per 50 items: F1-F3 positive hard / mid / easy, F0 bridge mid / easy, F1-F3 negative mid / easy.
#: The F0 bridge is kept small (6 of 50, 12 %): it only ties the pack to v1.0.1 and its dx_listed
#: saturates; the hard F1-F3 positives (comorbid pairs, K-NT 6-8, high distractor density) carry the
#: spread (22 of 50).
DESIGN = {"hard": 22, "pos_mid": 6, "pos_easy": 3, "f0_mid": 4, "f0_easy": 2, "neg_mid": 9, "neg_easy": 4}
BRIDGE = ("f0_mid", "f0_easy")

# Inclusion design, not an estimate of disease prevalence. The total positive / benign
# shares and frozen bridge share match the legacy table; difficulty is measured only.
EXPANSION_IDS = tuple(f"HD-UNI-{i:02d}" for i in range(7, 21))
# Acute presentations belong in triage; persistent presentations belong in the
# chronic-medication scene. Explicit frame restrictions live here, separately
# from the metadata-only manifest. F3 additionally requires a real follow-up plan.
EXPANSION_FRAMES = {sid: (("F1",) if i in (7, 8, 9, 11, 12, 15, 16) else ("F2",))
                    for i, sid in zip(range(7, 21), EXPANSION_IDS)}
#: Legacy expanded jobs retain their declared coverage table.
EXPANDED_DESIGN = {"bridge": 10, "positive": 60, "negative": 30}
#: the bridge share over its two prior tiers, in the 2 : 1 ratio of the v1.0.1 bridge set
BRIDGE_TIERS = {"f0_mid": 2, "f0_easy": 1}


def allocation_mode(registry: dict, mode: str = "auto") -> str:
    if mode not in ("auto", "legacy", "expanded"):
        raise ValueError(f"unknown allocation mode: {mode}")
    present = set(EXPANSION_IDS) & set(registry)
    if mode == "legacy":
        return mode
    if mode == "expanded" and not present:
        raise ValueError("expanded allocation requires integrated expansion specs")
    return "expanded" if present else "legacy"


def expanded_spec_cap(n: int) -> int:
    """Natural equal-share ceiling within diagnostic and benign strata.

    Benign controls occupy 30% and have fewer eligible specs; their natural
    ceiling sets the common cap. Required expansion coverage still applies.
    """
    import math
    from haenv.world.overlay import condition_registry
    specs = condition_registry()
    benign = sum(s["join_gold"] == "independent" for s in specs.values())
    diagnostic = len(specs) - benign
    if not benign or not diagnostic:
        raise ValueError("M2 spec cap needs both diagnostic and benign registries")
    return max(math.ceil(0.30 * n / benign), math.ceil(0.70 * n / diagnostic), math.ceil(n / 50))


def expanded_required_counts(n: int, specs: tuple = EXPANSION_IDS) -> dict:
    """Absolute positive minimum per new diagnosis: one at N50, two at N100."""
    return {sid: -(-n // 50) for sid in specs}


def expanded_quotas(n: int, bridge_supply: dict, *, diversity=False, confirmation=False) -> tuple[dict, dict]:
    """(cell -> items, bridge cell -> shortfall) for a pack of n items. The bridge takes its share of
    `EXPANDED_DESIGN` over `BRIDGE_TIERS` up to what the frozen bridge set supplies (which bridge cases
    fill a tier is the selection's sha256(seed, case id) order); a shortfall goes to F1-F3 by the arm
    weights and is returned, never dropped."""
    import haenv.world.validation.pack_size as PS
    design = EXPANDED_DESIGN
    if diversity:
        from haenv.m2.diversity import rules
        allocation = rules()["allocation"]
        bridge = round(100 * allocation["bridge_share"]["value"])
        independent = round(100 * allocation["independent_share"]["value"])
        if confirmation:
            from haenv.m2.confirmation import rules
            independent = round(100 * rules()["allocation"]["independent_share"]["value"])
        design = {"bridge": bridge, "positive": 100 - bridge - independent, "negative": independent}
    q = PS.apportion(n, design)
    want = PS.apportion(q["bridge"], BRIDGE_TIERS)
    take = {k: min(want[k], int(bridge_supply.get(k, 0))) for k in BRIDGE}
    rest = PS.apportion(n - sum(take.values()), {k: design[k] for k in ("positive", "negative")})
    out = dict(take)
    for arm, count in rest.items():
        frames = {"F1": 1, "F2": 1, "F3": 1}
        if confirmation:
            frames = rules()["allocation"]["negative_frames" if arm == "negative" else "positive_frames"]
        out.update({f"{f}:{arm}": v for f, v in PS.apportion(count, frames).items()})
    return out, {k: want[k] - take[k] for k in BRIDGE}


def quotas(n: int, bridge_supply: dict) -> tuple[dict, dict]:
    """(cell -> items, bridge cell -> shortfall) for a pack of n items."""
    import haenv.world.validation.pack_size as PS
    q = PS.apportion(n, DESIGN)
    take = {k: min(q[k], int(bridge_supply.get(k, 0))) for k in BRIDGE}
    rest = PS.apportion(n - sum(take.values()), {k: w for k, w in DESIGN.items() if k not in BRIDGE})
    return {k: (take[k] if k in BRIDGE else rest[k]) for k in DESIGN}, {k: q[k] - take[k] for k in BRIDGE}
