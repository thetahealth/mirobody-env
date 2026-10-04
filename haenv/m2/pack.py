"""Pack ①'s proportion table: how many items of each kind a pack of N items holds.

The bridge cells (F0) take their proportional share up to what the fixed v1.0.1 bridge set supplies;
the bridge is a frozen control set, so a shortfall is filled by new F1-F3 items in proportion.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

#: per 50 items: F1-F3 positive hard / mid / easy, F0 bridge mid / easy, F1-F3 negative mid / easy
DESIGN = {"hard": 15, "pos_mid": 6, "pos_easy": 3, "f0_mid": 10, "f0_easy": 6, "neg_mid": 6, "neg_easy": 4}
BRIDGE = ("f0_mid", "f0_easy")


def quotas(n: int, bridge_supply: dict) -> tuple[dict, dict]:
    """(cell -> items, bridge cell -> shortfall) for a pack of n items."""
    from .. import pack_size as PS
    q = PS.apportion(n, DESIGN)
    take = {k: min(q[k], int(bridge_supply.get(k, 0))) for k in BRIDGE}
    rest = PS.apportion(n - sum(take.values()), {k: w for k, w in DESIGN.items() if k not in BRIDGE})
    return {k: (take[k] if k in BRIDGE else rest[k]) for k in DESIGN}, {k: q[k] - take[k] for k in BRIDGE}
