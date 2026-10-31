"""P2 service procedures through the existing query tools, scoped to the P2 plugin.

Other task services retain their existing hospital catalog edition. This avoids
changing a different pack's query environment or world when P2 adds a procedure.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_ACTIVE = ContextVar("p2_protocol_catalog", default=False)


@contextmanager
def scope(active=True):
    token = _ACTIVE.set(active)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def install():
    from haenv.world.protocols import library
    if getattr(library.catalog, "_p2_scoped", False):
        return
    from haenv.resources.regpath import registry_cached
    from haenv.p2 import rules as R
    from copy import deepcopy
    original_table, original_catalog = library.table, library.catalog

    @wraps(original_table)
    def table():
        cfg = original_table()
        if not _ACTIVE.get():
            return cfg
        cfg = deepcopy(cfg)
        from haenv.i18n.text import L
        cfg["priority"] = L("p2.protocol.priority")
        infection = next(meta for meta in cfg["triage"] if meta["key"] == "triage-infection")
        infection["rules"].append("R3d")
        extra = deepcopy(R.table()["clinical_context"]["library_protocols"])
        for meta in extra:
            for key in ("name", "applies"):
                meta[key] = L(meta[key])
            for key in ("keywords", "excludes", "exceptions"):
                meta[key] = [L(s) for s in meta[key]]
        cfg["triage"].extend(extra)
        return cfg

    @registry_cached("protocols.yaml", "pack2_triage_scenarios.yaml", "p2_history_triage.yaml",
                     "p3_meds.yaml", "p3_clinical_sources.yaml", "control_targets.yaml", "history_dependency.yaml",
                     "difficulty_r2.yaml", "system_scoring.yaml")
    def p2_catalog():
        with scope():
            docs = original_catalog.__wrapped__()
            from haenv.i18n.text import J
            scoring = {m["protocol_id"]: m for m in table()["triage"] if m.get("scoring")}
            for doc in docs:
                if doc["protocol_id"] in scoring:
                    meta = scoring[doc["protocol_id"]]
                    n = sum(r["id"].split("_", 1)[0] in meta["rules"] for r in R.table()["rules"])
                    # Keep clinical-rule positions intact for saved citations.
                    # Scoring bands and Scale-2 instructions form one appendix.
                    doc["steps"] = doc["steps"][:n] + [J("lines", doc["steps"][n:])]
            return docs

    @wraps(original_catalog)
    def catalog():
        return p2_catalog() if _ACTIVE.get() else original_catalog()

    catalog._p2_scoped = True
    library.table, library.catalog = table, catalog
    original_catalog.cache_clear()
    from haenv.interaction.query.tools import RecordStore
    init = RecordStore.__init__

    @wraps(init)
    def initialize(store, sp, T, adapter=None, **kwargs):
        with scope(getattr(adapter, "pack", None) == "p2"):
            return init(store, sp, T, adapter, **kwargs)
    RecordStore.__init__ = initialize

    from haenv.interaction.query.stubs import ReaderStub
    rebuild = ReaderStub.rebuilt
    @wraps(rebuild)
    def rebuilt(self, messages, payload_like):
        with scope("triage_p2" in payload_like.prediction_context):
            return rebuild(self, messages, payload_like)
    ReaderStub.rebuilt = rebuilt

    import haenv.scoring.management.plan as M
    finalize = M.finalize
    @wraps(finalize)
    def management(raw, case_id, pack, **kwargs):
        with scope(pack == "p2"):
            return finalize(raw, case_id, pack, **kwargs)
    M.finalize = management

