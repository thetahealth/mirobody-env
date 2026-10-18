"""Explicit saved-payload policy catalog.

Historical and hand-constructed worlds need not have a loaded job or plugin.
These declared selectors load their derivation factory directly; they never
call plugin.register(). A missing module/factory or failed derivation raises.
Task-specific clinical decisions live in the declared policy, outside wq.
"""
from __future__ import annotations

from importlib import import_module

# (field, payload block, module, factory). Selection uses structural presence,
# never an authored gold answer. Paths are the supported historical wire format.
PAYLOAD_POLICIES = (
    ("tests", "W.adjudication.m2_workup", "haenv.m2.workup", "gold_derivations"),
    ("red_flag_present", "W.latent_premise.meta._external.m2.critical",
     "haenv.m2.workup", "gold_derivations"),
)


def gold_derivations_for(world, field):
    from haenv.app.packs import PACKS
    from haenv.world.wq import resolve_path
    # Registered declarations take precedence by their explicit names. The catalog
    # supplies the same declarations for saved worlds with no registration state.
    strategies = {s.name: s for p in tuple(PACKS.values()) for s in p.gold_derivations}
    for selected_field, path, module, factory in PAYLOAD_POLICIES:
        if field == selected_field and isinstance(resolve_path(world, path), dict):
            for s in getattr(import_module(module), factory)():
                strategies.setdefault(s.name, s)
    return tuple(strategies.values())
