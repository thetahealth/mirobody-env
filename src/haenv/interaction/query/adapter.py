"""What a pack declares to be asked in query mode: which part of its record block is which kind of
record, what the question shows at once, and -- on the verifier side only -- the words and numbers
a tool result must never carry and the lookups that show the decisive evidence.

A pack registers one `QueryAdapter` (`register`); `adapter_for(payload)` picks it by the block
key of the solver payload. A case no adapter owns is asked as before (one prompt).

Section kinds (`sections`): `labs` (served by `get_labs`), `meds` (`get_meds`), `notes`
(`get_notes`), `series` (a dated table, also served by `get_series`), `profile` (returned with
`get_profile`), `record` (only `get_record`). A section may have several kinds. Every section is
also readable by name through `get_record`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

_LOCK = threading.Lock()
#: pack name -> adapter, in registration order
ADAPTERS: dict[str, "QueryAdapter"] = {}


@dataclass
class KeyLookup:
    """One piece of decisive evidence and the lookups that show it (verifier side)."""
    what: str                                   # e.g. "draw_details@201"
    seen: object                                # callable(tool name, returned row with `_from`) -> bool


@dataclass
class QueryAdapter:
    pack: str                                   # stamp prefix, e.g. "p4"
    block: str | None                           # prediction_context key of the pack's record block
    sections: dict = field(default_factory=dict)    # block key -> (kind | (kinds...), description)
    header_keys: tuple = ()                     # block keys shown in the question itself
    lab_streams: tuple | None = None            # streams that are lab draws (None: streams with a draw time)
    not_patient_data: tuple = ("text_source",)  # block bookkeeping no tool serves
    fhir_types: dict = field(default_factory=dict)  # section -> FHIR resource type, where not the kind's (`fhir.BY_KIND`)

    # ---- solver side ----
    def owns(self, payload) -> bool:
        pc = getattr(payload, "prediction_context", None) or {}
        return isinstance(pc, dict) and isinstance(pc.get(self.block), dict)

    def header(self, block: dict) -> dict:
        return {k: block[k] for k in self.header_keys if k in block}

    def profile_header(self, profile: dict) -> dict:
        """Immediate demographic header; the full profile remains retrievable."""
        return profile

    def lab_label(self, block: dict, section: str) -> str:
        return section

    def extra_specs(self, store) -> list:
        return []

    def protocol_lookups(self, store) -> list[KeyLookup]:
        if not store.protocols:
            return []
        from haenv.world.protocols.library import bearing_ids
        return [KeyLookup(f"protocol:{pid}", lambda t, r, pid=pid:
                          t == "get_protocol" and r.get("_from") == "protocol_library" and r.get("protocol_id") == pid)
                for pid in sorted(bearing_ids(self.pack, store.block))]

    # ---- verifier side (never handed to the store or the model) ----
    def guard_terms(self, vp) -> tuple[tuple, tuple]:
        """`(words, numbers)` a tool result must not carry."""
        return (), ()

    def key_lookups(self, vp, store) -> list[KeyLookup]:
        return []

    def history_lookups(self, vp) -> list[KeyLookup]:
        """A dated historical source must itself return to count as seen."""
        from haenv.world.history.history_dependency import metadata
        m = metadata(vp, self.pack)
        if not m.get("active"):
            return []
        return [KeyLookup(f"{r['section']}@{r['day']}",
                          lambda _t, row, s=r['section'], d=int(r['day']):
                          row.get('_from') == s and row.get('day', row.get('draw_day')) == d)
                for r in m.get('records') or ()]

    # ---- offline stubs ----
    def read_answer(self, sp) -> dict:
        """The pack's visible-record reader on a whole payload (`q_reader` answers with it on the
        payload it rebuilt from tool results)."""
        return {}

    def blind_answer(self, payload=None) -> dict:
        """A schema-valid answer given without reading anything (`q_nolook`)."""
        return {}


def register(adapter: QueryAdapter) -> QueryAdapter:
    with _LOCK:
        ADAPTERS[adapter.pack] = adapter
    return adapter


def load_pack_adapters() -> None:
    """Import every pack's query adapter (each registers itself on import)."""
    from haenv.p2 import query as _p2      # noqa: F401
    from haenv.p3 import query as _p3      # noqa: F401
    from haenv.p4 import query as _p4      # noqa: F401
    from haenv.m2 import query as _m2      # noqa: F401


def adapter_for(payload) -> QueryAdapter | None:
    for a in ADAPTERS.values():
        if a.block and a.owns(payload):
            return a
    return None
