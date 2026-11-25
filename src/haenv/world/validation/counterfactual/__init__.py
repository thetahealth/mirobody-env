"""Stable, model-free counterfactual audit API (version 1).

Returns premise ids, rule roles, outcomes and stable record keys; never rendered records.
See the counterfactual instrument design document for receipt and gate semantics.
"""
from importlib import import_module
from importlib.util import find_spec

from .batch import Item, load_batch
from .registry import load, premises

API_VERSION = load()["version"]
PACKS: tuple[str, ...] = tuple(sorted(pack for pack in load()["packs"]
                                     if find_spec(f"{__name__}.{pack}") is not None))


def adapter(pack):
    if pack not in PACKS:
        raise ValueError(f"unsupported counterfactual pack: {pack}")
    return import_module(f".{pack}", __name__).Adapter()


def item_roles(pack, raw):
    """Return per-premise roles from condition overrides, without constructing twins."""
    from .receipt import roles
    return roles(adapter(pack), raw)


def decisive_facts(pack, raw, sp=None):
    """Registry-ordered acting/guard facts, optionally with stable visible record keys."""
    from .diffs import record_keys
    if hasattr(adapter(pack), "decisive_facts"):
        return adapter(pack).decisive_facts(raw, sp)
    roles = item_roles(pack, raw)
    blocks = adapter(pack).visible_blocks(sp) if sp is not None else None
    out = []
    for premise in premises(pack):
        role = roles[premise["id"]]
        if role["role"] not in ("acting", "guard"):
            continue
        fact = dict(premise=premise["id"], role=role["role"], label=premise["label"],
                    visible=premise["visible"], observed=role["observed"])
        if blocks is not None:
            fact["visible_records"] = record_keys(blocks, premise["visible"])
        out.append(fact)
    return out


def item_receipt(pack, raw, sp, *, twins=True):
    """Build one report receipt, including necessity, invariance, visibility and fidelity."""
    from .receipt import item_receipt as build
    return build(adapter(pack), raw, sp, twins=twins)


__all__ = ["API_VERSION", "PACKS", "premises", "item_roles", "decisive_facts", "item_receipt", "load_batch", "Item"]
