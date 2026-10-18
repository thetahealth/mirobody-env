"""Render full-context hospital libraries through the same store as query tools.

Only the model-facing serialization changes; the stored solver payload and the
attributes used by pack renderers and answer conversion retain their identity.
"""
from __future__ import annotations

import copy
import json

from haenv.interaction.query.adapter import adapter_for
from haenv.interaction.query.tools import RecordStore


class _SingleView:
    def __init__(self, payload, shown: dict):
        self._payload, self._shown = payload, shown

    def __getattr__(self, name):
        return getattr(self._payload, name)

    def dumps(self) -> str:
        return json.dumps(self._shown, ensure_ascii=False, separators=(",", ":"))


def single_view(payload):
    """Expand only a library placeholder, with the exact get_protocol result.

    RecordStore resolves catalog() under the pack's scope. Its specs also install
    pack-specific protocol supplements, just as before a query's first turn.
    """
    adapter = adapter_for(payload)
    if adapter is None:
        return payload
    pc = payload.prediction_context
    block = pc[adapter.block]
    if block.get("rules_delivery") != "library" or block.get("protocol_library") != "本院规程库":
        return payload
    store = RecordStore(payload, int(pc["prediction_time_T"]), adapter)
    store.specs()
    shown = json.loads(payload.dumps())
    shown["prediction_context"][adapter.block]["protocol_library"] = store.get_protocol()
    return _SingleView(payload, shown)


class _SingleMixin:
    def solve(self, payload):
        return super().solve(single_view(payload))


_CLASSES: dict[type, type] = {}


def as_single_solver(inner):
    """Copy a production solver, changing only its full-context presentation."""
    if isinstance(inner, _SingleMixin):
        return inner
    base = type(inner)
    if base not in _CLASSES:
        _CLASSES[base] = type(f"Single{base.__name__}", (_SingleMixin, base), {})
    solver = copy.copy(inner)
    solver.__class__ = _CLASSES[base]
    return solver
