"""P4's patient-independent laboratory interpretation document."""
from __future__ import annotations

from copy import deepcopy
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
    doc["steps"].append(f"本院方法文件与证据范围：游离血红蛋白验证界限{p['free_hemoglobin_limit_g_L']} g/L；EDTA转移的低钙支持范围<{p['edta_calcium_support_max_mmol_L']} mmol/L；未分离标本冷藏≤{p['cold_max_C']}℃且≥{p['cold_separation_min_h']}小时；普通全血室温糖酵解核对≥{p['glycolysis_min_h']}小时；生物素剂量{p['biotin_dose_mg']} mg；甲状腺素复评至少{p['treatment_settling_days']}日。{p['reason']}")
    for an in cfg["design"]["analytes"]:
        a = analyte(an)
        analytical = '、'.join(f"{method} {100*value:g}%" for method, value in a["cva"].items())
        doc["steps"].append(f"{a['display']}：CVI={100*a['cvi']:g}%；CVA={analytical}；双侧95%对数RCV={100*rcv95(cv_total(an)):.2f}%。单位{a['unit']}。")
    doc["steps"] += [f"{row.get('exposure', row['rule'].split('；')[0])}：{row['rule'].replace('真实变化', '体内水平改变')}（来源{row['source']}；合成响应范围见登记，不能当患者处置阈值。）" for row in cfg["rules"].values()]
    doc["steps"] += [f"{row['exposure']}：本院合成随访取暴露或病程起点在本次采血前{row['event_lead_days'][0]}–{row['event_lead_days'][1]}日，须在两次采血之间；该生成时间范围不是通用诊断截止值。" for row in cfg["rules"].values() if row.get("event_lead_days")]
    doc["sources"] += [f"{key}: {src['title']}；{src['url']}" for key, src in cfg["sources"].items()]
    return doc


def supplement(store):
    """Extend only P4 retrieval; other packs retain their exact catalog bytes."""
    doc = document()
    if store.protocols and not any(d["protocol_id"] == doc["protocol_id"] for d in store.protocols):
        store.protocols = [*store.protocols, deepcopy(doc)]
