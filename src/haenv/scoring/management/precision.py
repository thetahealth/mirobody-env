"""Render verified guideline data, without medical thresholds in Python."""

from haenv.i18n.runtime import registry_phrase
from haenv.i18n.text import text_of as _rt
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
    rules = {kind: [{k: registry_phrase(v) if k in {'text', 'control', 'early_control'} else v
                     for k, v in rule.items()} for group in groups for rule in group.get(kind, [])]
             for kind in ("not_recommended", "initial", "monitoring")}
    if not rules["not_recommended"] and not rules["monitoring"]:
        builder.gaps.append(_rt('judge.scoring.management.precision.add_precision.1'))
        return
    render = lambda rule: (_rt('judge.punctuation.precision.1', s0=rule['text'], s1=data['sources'][rule['source']]['short'], s2=data['sources'][rule['source']]['section']))
    text = (
        _rt('judge.scoring.management.precision.add_precision.2') + (_rt('judge.punctuation.precision.2').join(map(render, rules["not_recommended"])) or _rt('judge.scoring.management.precision.add_precision.3')) + _rt('judge.scoring.management.precision.add_precision.4') + (_rt('judge.punctuation.precision.3').join(map(render, rules["initial"])) or _rt('judge.scoring.management.precision.add_precision.5')) + _rt('judge.scoring.management.precision.add_precision.6') + (_rt('judge.punctuation.precision.4').join(map(render, rules["monitoring"])) or _rt('judge.scoring.management.precision.add_precision.7')) + _rt('judge.scoring.management.precision.add_precision.8')
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
                evidence=refs, claim=_rt('judge.scoring.management.precision.add_precision.9'),
                diagnostic=diagnostics)
    builder.items[-1]["guideline_basis"] = {
        "registry_version": data["version"], "registry_sha256": sha256(data),
        "family": family, "sources": {r["source"]: data["sources"][r["source"]]
                                       for kind in rules for r in rules[kind]},
        "accessed": data["accessed"]}
