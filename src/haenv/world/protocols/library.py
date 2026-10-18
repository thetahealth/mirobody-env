"""One hospital catalog for every patient; clinical text has one generating source."""
from __future__ import annotations

from copy import deepcopy
from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached


@registry_cached("protocols.yaml", "difficulty_r2.yaml")
def table():
    cfg = load_registry("protocols.yaml")
    # These are presentation entries for the already registered R2 policies.
    # Keep their sole clinical source in the generation-side R2 registry.
    from haenv.world.validation.difficulty_r2 import config
    cfg["medication"].extend(config("p3")["library_protocols"])
    return cfg


def render(doc: dict) -> str:
    labels = (("name", "名称"), ("applies", "适用条件"), ("excludes", "排除条件"),
              ("steps", "步骤"), ("exceptions", "例外"), ("sources", "出处"))
    return "\n".join(f"{label}：" + ("\n".join(map(str, doc[key])) if isinstance(doc[key], list) else str(doc[key]))
                     for key, label in labels)


def _doc(meta, steps, sources, *, exceptions=None):
    doc = {k: deepcopy(meta[k]) for k in ("protocol_id", "name", "topics", "keywords", "applies", "exceptions")}
    doc["excludes"] = [x["text"] if isinstance(x, dict) else x for x in meta["excludes"]]
    doc.update(steps=list(steps), sources=list(dict.fromkeys(sources)))
    if exceptions is not None:
        doc["exceptions"] = list(exceptions)
    return doc


def _source(cite, url="", policy="本院操作映射暂定，需临床复核"):
    return f"参考：{cite}；链接：{url or '原登记表'}；本院约定：{policy}。"


def _clauses(text):
    # Split presentation only: joining recovers the exact existing policy.
    import re
    return re.findall(r"[^；。]+[；。]?", text)


@registry_cached("protocols.yaml", "pack2_triage_scenarios.yaml", "p3_meds.yaml", "control_targets.yaml", "history_dependency.yaml", "difficulty_r2.yaml", "system_scoring.yaml")
def catalog() -> list[dict]:
    from haenv.p2 import rules as R
    from haenv.p3.render import clinic_protocol
    from haenv.p3.render import attribution_rule
    from haenv.p3.render import adherence_rule
    from haenv.p3.render import _dose
    from haenv.world.labworld.meds import drug
    from haenv.world.history.history_dependency import table as history_table
    from haenv.world.validation.difficulty_r2 import enabled
    from haenv.world.validation.difficulty_r2 import config as difficulty_config
    cfg, triage = table(), R.table()
    docs = []
    for meta in cfg["triage"]:
        rules = [r for r in triage["rules"] if r["id"].split("_", 1)[0] in meta["rules"]]
        steps = [r["printed"] for r in rules]
        exceptions = [*meta["exceptions"], cfg["priority"]]
        if "R5" in meta["rules"]:
            exceptions.append(triage["acceptance"]["single_level2"]["printed"])
        if meta.get("scoring"):
            fields = {"rr": "呼吸频率", "spo2": "SpO2 Scale 1", "sbp": "收缩压", "hr": "心率", "temp": "体温"}
            bands = [label + "：" + "；".join(f"{lo}–{hi}：{pts} 分" for lo, hi, pts in triage["news2"][k])
                     for k, label in fields.items()]
            bands.append(f"新发意识混乱 {triage['news2']['new_confusion_points']} 分，其余清醒 0 分。")
            steps.append("NEWS2 各项相加（区间含端点），缺测不能当 0 分；先核对病历授权的 SpO2 量表。" + "\n".join(bands))
            # The hospital catalog must not depend on the caller's generation
            # switch: clinical instructions remain available during retrieval.
            steps.append(difficulty_config("p2")["protocol"])
            exceptions.append("所有 NEWS2 条款均按病历授权的 SpO2 量表计算，并计入吸氧加分；条款中的第1表、未吸氧是默认情形，不能覆盖已登记的量表2或实际氧疗。")
        if meta.get("history"):
            steps.extend(_clauses(history_table()["packs"]["p2"]["protocol"]))
        refs = []
        for r in rules:
            for key in [r.get("source"), *(r.get("also_sources") or [])]:
                src = triage["sources"].get(key) or {}
                refs.append(_source(src.get("cite") or "本院分诊处置表", src.get("url", "")))
        if meta.get("scoring"):
            src = difficulty_config("p2")["scale2"]
            refs.append(_source("RCP NEWS2 2017；" + src["scope"], src["source"],
                                "量表2资格、分段及氧疗加分逐项核对原文；家庭分诊去向沿用本院条款"))
        docs.append(_doc(meta, steps, refs, exceptions=exceptions))
    keys = ("gliclazide", "atorvastatin", "enalapril", "levothyroxine")
    for meta in cfg["medication"]:
        gen = meta["generator"]
        if gen == "review":
            steps = [drug(k)["display"].split("（")[0] + "：剂量阶梯 " +
                     " / ".join(_dose(v, drug(k)["unit"]) for v in drug(k)["ladder"]) + "（每日）。" +
                     clinic_protocol({"drug": k})[0] for k in keys]
        elif gen == "adverse":
            steps = [attribution_rule(drug(k)) for k in keys]
            if enabled():
                cfg3 = difficulty_config("p3")
                steps.append(cfg3["protocol"])
                steps.append(cfg3["allergy_record"]["protocol"])
                steps.append("症状状态只登记原始记录明确记载的经过，未登记不表示已缓解或持续。一次症状后来缓解也不能替代对起病时用药、加量日期和其他原因的核对。")
        elif gen == "adherence":
            steps = _clauses(adherence_rule())
        elif gen == "statin":
            steps = _clauses(history_table()["packs"]["p3"]["protocol"])
            if enabled():
                steps.append(difficulty_config("p3")["archived_safety_panel"]["protocol"])
        elif gen == "renal":
            steps = _clauses(str(drug("enalapril", ("CKD",))["clinic_policy"]))
        elif gen in ("sulfonylurea", "acei"):
            src = meta["source"]
            docs.append(_doc(meta, meta["steps"], [_source(src["cite"] + "；" + src["locator"],
                             src["url"], meta["ours_reason"])]))
            continue
        else:
            raise ValueError(f"unknown protocol generator: {gen}")
        docs.append(_doc(meta, steps, [_source("本院慢病用药表与所登记的药品说明书、指南", "", "按原表生成复评、归因、依从与合并症条款；患者目标另查医嘱")]))
    for meta in cfg["other"]:
        import re
        src = meta["source"]
        match = re.search(r"NICE (?:NG|CG)\d+", src["cite"])
        cite = (match[0] if match else src["cite"].split(",")[0].split(" (")[0]) + "；" + src["locator"]
        docs.append(_doc(meta, meta["steps"], [_source(cite, src["url"])]))
    from haenv.world.protocols.followup import steps as followup_steps
    meta = cfg["followup"]
    docs.append(_doc(meta, followup_steps(), [_source(meta["source"]["cite"], meta["source"]["url"])]))
    ids = [d["protocol_id"] for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate hospital protocol id")
    return sorted(docs, key=lambda d: d["protocol_id"])


def search(docs: list[dict], *, name=None, topic=None, keyword=None) -> list[dict]:
    def contains(value, needle):
        return needle is None or str(needle).casefold() in str(value).casefold()
    return [deepcopy(d) for d in docs if contains(d["name"] + " " + d["protocol_id"], name)
            and contains(" ".join(d["topics"]), topic)
            and contains(" ".join(d["keywords"]) + " " + render(d), keyword)]


def bearing_ids(pack: str, block: dict) -> set[str]:
    if pack == "p4":
        return {table()["followup"]["protocol_id"]}
    family = "triage" if pack == "p2" else "medication" if pack == "p3" else None
    if not family:
        return set()
    return {m["protocol_id"] for m in table()[family]}


def clinic_lines(block: dict) -> list[str]:
    from haenv.world.labworld.meds import drug
    keys = ("gliclazide", "atorvastatin", "enalapril", "levothyroxine")
    key = next(k for k in keys if drug(k)["display"] == block.get("drug"))
    name = drug(key)["display"].split("（")[0]
    docs = block.get("protocol_library") or ()
    if docs == "本院规程库":
        docs = catalog()
    generators = {m["protocol_id"]: m["generator"] for m in table()["medication"]}
    out = []
    for doc in docs:
        gen = generators.get(doc["protocol_id"])
        if gen in ("review", "adverse"):
            out.extend(line for line in doc["steps"] if line.startswith(name + "：") or line.startswith("本院不良反应归因：" + name)
                       or gen == "adverse" and not line.startswith("本院不良反应归因："))
        elif (gen == "adherence" or gen == "statin" and key == "atorvastatin"
              or gen == "renal" and key == "enalapril" and block.get("known_comorbidity")
              or gen == "sulfonylurea" and key == "gliclazide" or gen == "acei" and key == "enalapril"):
            out.append("".join(doc["steps"]))
    return out
