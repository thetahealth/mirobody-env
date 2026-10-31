"""P4's patient-independent laboratory interpretation document."""
from __future__ import annotations

from copy import deepcopy

from haenv.i18n.text import L, T
from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached


@registry_cached("p4_interpretation.yaml")
def _base_document():
    return load_registry("p4_interpretation.yaml")["protocol"]


def document():
    doc = deepcopy(_base_document())
    from haenv.p4.diversity import table
    from haenv.world.labworld import analyte
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import rcv95
    cfg = table()
    doc["keywords"] += [analyte(an)["display"] for an in cfg["design"]["analytes"]]
    doc["steps"] += [cfg["design"]["noise"], cfg["rcv"]["reason"], cfg["analytical_cv"]["reason"]]
    doc["steps"].append(cfg["medication_world_design"]["acute_comparison_scope"])
    p = cfg["evidence_parameters"]
    doc["steps"].append(T("p4.doc.evidence_scope", hb=p['free_hemoglobin_limit_g_L'],
                          ca=p['edta_calcium_support_max_mmol_L'], cold_c=p['cold_max_C'],
                          cold_h=p['cold_separation_min_h'], glyco_h=p['glycolysis_min_h'],
                          biotin=p['biotin_dose_mg'], settle=p['treatment_settling_days'], reason=p['reason']))
    for an in cfg["design"]["analytes"]:
        a = analyte(an)
        analytical = None
        for method, value in a["cva"].items():
            item = T("p4.doc.cva_item", method=method, pct=f"{100*value:g}")
            analytical = item if analytical is None else T("p4.doc.enum", head=analytical, tail=item)
        doc["steps"].append(T("p4.doc.analyte_cv", display=a['display'], cvi=f"{100*a['cvi']:g}", cva=analytical if analytical is not None else "",
                              rcv=f"{100*rcv95(cv_total(an)):.2f}", unit=a['unit']))
    doc["steps"] += [T("p4.doc.rule", exposure=row.get("exposure") or L(f"p4.rule_head.{key}"),
                       rule=L(f"p4.rule.{key}"), source=row["source"])
                     for key, row in cfg["rules"].items()]
    doc["steps"] += [T("p4.doc.event_window", exposure=row['exposure'], lo=row['event_lead_days'][0],
                       hi=row['event_lead_days'][1]) for row in cfg["rules"].values() if row.get("event_lead_days")]
    doc["sources"] += [T("p4.doc.source", source_id=key, title=src['title'], url=src['url']) for key, src in cfg["sources"].items()]
    return doc


def supplement(store):
    """Extend only P4 retrieval; other packs retain their exact catalog bytes."""
    from haenv.i18n.runtime import runtime_text
    doc = runtime_text(document())
    if isinstance(store.protocols, str):
        from haenv.world.protocols.library import catalog
        store.protocols = runtime_text(catalog())
    if store.protocols and not any(d["protocol_id"] == doc["protocol_id"] for d in store.protocols):
        store.protocols = [*store.protocols, deepcopy(doc)]
