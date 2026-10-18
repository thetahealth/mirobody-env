"""Render verified guideline data, without medical thresholds in Python."""
from fnmatch import fnmatchcase
from pathlib import Path

from haenv.resources.yamlcache import load_yaml

from haenv.scoring.management.rubric import ROOT
from haenv.scoring.management.rubric import canonical
from haenv.scoring.management.rubric import sha256

REGISTRY = ROOT / "registry/test_indications.yaml"


def load_indications(path: Path | None = None):
    data = load_yaml(path or REGISTRY)
    if not data.get("version") or not data.get("accessed"):
        raise ValueError("Unversioned test indication registry")
    for source in data["sources"].values():
        if not all(source.get(k) for k in ("short", "url", "section", "quote")):
            raise ValueError("Unverified clinical source")
    for group in list(data["families"].values()) + list(data["chronic"].values()):
        for kind in ("not_recommended", "initial", "monitoring"):
            for rule in group.get(kind, []):
                if not rule.get("text") or rule.get("source") not in data["sources"]:
                    raise ValueError("Clinical rule has no registered source")
    return data


def _matches(group, family, conditions, medications):
    return (not group.get("families") or any(fnmatchcase(family, p) for p in group["families"])) and (
        not group.get("conditions") or bool(set(group["conditions"]) & conditions)) and (
        not group.get("medications_any") or any(name.casefold() in medications
                                               for name in group["medications_any"]))


def add_precision(builder, sources, decision_item):
    data = load_indications()
    if builder.pack not in data["packs"]:
        return
    snapshot = builder.snapshot
    visible = snapshot["visible_facts"]
    background = visible.get("background", {})
    family = snapshot["decision_facts"].get("tier", "")
    conditions = set(background.get("known_conditions", []))
    medications = canonical(background.get("medications", [])).casefold()
    groups = [group for group in list(data["families"].values()) + list(data["chronic"].values())
              if _matches(group, family, conditions, medications)]
    rules = {kind: [rule for group in groups for rule in group.get(kind, [])]
             for kind in ("not_recommended", "initial", "monitoring")}
    if not rules["not_recommended"] and not rules["monitoring"]:
        builder.gaps.append("precision未派生：本例无已核实适用的检查禁止清单或慢病监测间隔。")
        return
    render = lambda rule: (f"{rule['text']}（{data['sources'][rule['source']]['short']} "
                           f"{data['sources'][rule['source']]['section']}）")
    text = (
        "本例检查精确性只对照以下带出处的清单，不能因列检查名或数量多而扣分。"
        "前提必须由本例可见资料或方案明确建立，不能推断未提供的心律诊断、风险、病情稳定或上次检查日期。\n"
        "不常规推荐清单：" + ("；".join(map(render, rules["not_recommended"])) or "无独立禁止项。") + "\n"
        "允许的初始检查：" + ("；".join(map(render, rules["initial"])) or "未登记本家族初始检查；不把未列项目当作禁止。") + "\n"
        "慢病监测：" + ("；".join(map(render, rules["monitoring"])) or "无本例适用的已登记间隔。") + "\n"
        "本条不要求安排上述初始检查或监测。针对本次问题的初始检查、按间隔常规监测、"
        "写明交由接诊医生判断是否需要的检查均不扣分；明确已经安排违规检查后再说由医生执行不免扣。"
        "只有安排了清单中符合全部前提的不推荐检查，或明确无新指征且早于适用间隔的同项常规重复监测，才判contradicted；"
        "最低监测频次不是最短间隔，资料不足以证明重复也不扣分。"
        "否则有本次实际处置即satisfied；没有任何处置（空答、纯读数或泛泛遵医嘱）即unmet。"
    )
    refs = [f"/visible_facts/{key}" for key in (
        "background", "home_check_today", "intake_checklist", "message", "clinical_history_record") if key in visible]
    decision = next(i for i in builder.items if i["id"] == decision_item)
    structured = decision["stub"].get("structured")
    diagnostics = {"kind": "precision", "decision_item": decision_item,
                   "decision_claims": decision["stub"]["claims"] + decision["stub"]["contradictions"],
                   "structured": {"field": structured["field"], "choices": list(builder.policy["dispositions"])} if structured else None,
                   "violations": [r[k] for kind in rules for r in rules[kind]
                                  for k in ("control" if kind == "not_recommended" else "early_control",) if k in r],
                   "controls": {kind: (next((r["control"] for r in rules[kind] if r.get("control")), None)
                                        if kind == "not_recommended" else
                                        "\n".join(r["control"] for r in rules[kind] if r.get("control")) or None)
                                for kind in rules}}
    builder.add("precision.guideline_checks", "precision", text, list(dict.fromkeys(sources + refs)),
                evidence=refs, claim="本次处置按上述安排执行；检查遵循本例适应证及慢病监测计划。",
                diagnostic=diagnostics)
    builder.items[-1]["guideline_basis"] = {
        "registry_version": data["version"], "registry_sha256": sha256(data),
        "family": family, "sources": {r["source"]: data["sources"][r["source"]]
                                       for kind in rules for r in rules[kind]},
        "accessed": data["accessed"]}
