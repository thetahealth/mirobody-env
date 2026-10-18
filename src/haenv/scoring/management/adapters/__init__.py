"""Small read-only adapters over saved open-management facts."""

from importlib import import_module


def load_adapter(name: str):
    """Resolve the four declared adapter names in saved pre-package policies.

    The saved policy stays unchanged. This is an explicit module relocation,
    not a replacement policy or an alternative clinical implementation.
    """
    from haenv.scoring.management.adapters import p2
    from haenv.scoring.management.adapters import p3
    from haenv.scoring.management.adapters import p4
    from haenv.scoring.management.adapters import m2

    legacy = {
        "haenv.management.adapters.p2": p2,
        "haenv.management.adapters.p3": p3,
        "haenv.management.adapters.p4": p4,
        "haenv.management.adapters.m2": m2,
        "haenv.management_adapters.p2": p2,
        "haenv.management_adapters.p3": p3,
        "haenv.management_adapters.p4": p4,
        "haenv.management_adapters.m2": m2,
    }
    if name in legacy:
        return legacy[name]
    return import_module(name)
