"""One hospital catalog for every patient; clinical text has one generating source."""
from __future__ import annotations

from copy import deepcopy

from haenv.i18n.text import J, L, T
from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached


@registry_cached("protocols.yaml", "difficulty_r2.yaml")
def table():
    cfg = load_registry("protocols.yaml")
    # Citations with only Latin text can still require localized punctuation.
    # Route source fields explicitly; they are presentation, never identifiers.
    def source_text(value):
        if isinstance(value, dict):
            return {k: L(v) if k in {"cite", "locator"} and isinstance(v, str)
                    else source_text(v) for k, v in value.items()}
        if isinstance(value, list):
            return [source_text(v) for v in value]
        return value
    # These are presentation entries for the already registered R2 policies.
    # Keep their sole clinical source in the generation-side R2 registry.
    from haenv.world.validation.difficulty_r2 import config
    cfg["medication"].extend(config("p3")["library_protocols"])
    return source_text(cfg)


def render(doc: dict, lang: str | None = None) -> str:
    """A protocol document as one text, in `lang` (default: the current language)."""
    from haenv.i18n.text import render as _render
    from haenv.i18n.text import text_of
    import haenv.i18n as _i18n
    lang = lang or _i18n.current()
    out = []
    for key in ("name", "applies", "excludes", "steps", "exceptions", "sources"):
        v = _render(doc[key], lang)
        body = "\n".join(map(str, v)) if isinstance(v, list) else str(v)
        out.append(text_of("protocol.doc.field", lang, label=text_of(f"protocol.doc.{key}", lang), body=body))
    return "\n".join(out)


def _doc(meta, steps, sources, *, exceptions=None):
    doc = {k: deepcopy(meta[k]) for k in ("protocol_id", "name", "topics", "keywords", "applies", "exceptions")}
    doc["excludes"] = [x["text"] if isinstance(x, dict) else x for x in meta["excludes"]]
    doc.update(steps=list(steps), sources=list(dict.fromkeys(sources)))
    if exceptions is not None:
        doc["exceptions"] = list(exceptions)
    return doc


def _source(cite, url="", policy=None):
    return T("protocol.source", cite=cite, url=url or L("protocol.source.registry_table"),
             policy=policy if policy is not None else L("protocol.source.provisional"))


def _clauses(text):
    # Split presentation only: joining recovers the exact existing policy.
    import re
    from haenv.i18n.text import Msg
    parts = re.findall(r"[^；。]+[；。]?", text)
    if isinstance(text, Msg) and len(parts) > 1:
        # each clause is a phrase of its own (its canonical text is its id)
        return [L(x) for x in parts]
    return [text] if len(parts) == 1 and isinstance(text, Msg) else parts


@registry_cached("protocols.yaml", "pack2_triage_scenarios.yaml", "p3_meds.yaml", "p3_clinical_sources.yaml", "control_targets.yaml", "history_dependency.yaml", "difficulty_r2.yaml", "system_scoring.yaml")
def catalog() -> list[dict]:
    from haenv.p2 import rules as R
    from haenv.p3.render import clinic_protocol
    from haenv.p3.render import attribution_rule
    from haenv.p3.render import adherence_rule
    from haenv.p3.render import _dose
    from haenv.world.labworld.meds import drug, meds
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
            bands = [T("protocol.news2.bands", label=L(f"protocol.news2.field.{k}"),
                       bands=J("semicolon", [T("protocol.news2.band", lo=lo, hi=hi, pts=pts)
                                             for lo, hi, pts in triage["news2"][k]]))
                     for k in ("rr", "spo2", "sbp", "hr", "temp")]
            bands.append(T("protocol.news2.confusion", points=triage['news2']['new_confusion_points']))
            steps.append(T("protocol.news2.sum", bands=J("lines", bands)))
            # The hospital catalog must not depend on the caller's generation
            # switch: clinical instructions remain available during retrieval.
            steps.append(difficulty_config("p2")["protocol"])
            exceptions.append(L("protocol.news2.scale_exception"))
        if meta.get("history"):
            steps.extend(_clauses(history_table()["packs"]["p2"]["protocol"]))
        refs = []
        for r in rules:
            for key in [r.get("source"), *(r.get("also_sources") or [])]:
                src = triage["sources"].get(key) or {}
                refs.append(_source(src.get("cite") or L("protocol.source.triage_table"), src.get("url", "")))
        if meta.get("scoring"):
            src = difficulty_config("p2")["scale2"]
            refs.append(_source(T("protocol.cite_join", head="RCP NEWS2 2017", tail=src["scope"]), src["source"],
                                L("protocol.source.scale2_policy")))
        docs.append(_doc(meta, steps, refs, exceptions=exceptions))
    keys = meds()["drugs"]
    for meta in cfg["medication"]:
        gen = meta["generator"]
        if gen == "review":
            steps = [T("protocol.med.ladder", name=L(f"drug.{k}"),
                       ladder=J("slash", [_dose(v, drug(k)["unit"]) for v in drug(k)["ladder"]]),
                       rule=clinic_protocol({"drug": k})[0]) for k in keys]
        elif gen == "adverse":
            steps = [attribution_rule(drug(k)) for k in keys]
            if enabled():
                cfg3 = difficulty_config("p3")
                steps.append(cfg3["protocol"])
                steps.append(cfg3["allergy_record"]["protocol"])
                steps.append(L("protocol.p3.documented_course_only"))
        elif gen == "adherence":
            steps = _clauses(adherence_rule())
        elif gen == "levothyroxine":
            src = meta["source"]
            docs.append(_doc(meta, clinic_protocol({"drug": "levothyroxine"})[3:],
                             [_source(T("protocol.cite_join", head=src["cite"], tail=src["locator"]),
                                      src["url"], meta["ours_reason"])]))
            continue
        elif gen == "statin":
            steps = _clauses(history_table()["packs"]["p3"]["protocol"])
            if enabled():
                steps.append(difficulty_config("p3")["archived_safety_panel"]["protocol"])
        elif gen == "renal":
            steps = _clauses(str(drug("enalapril", ("CKD",))["clinic_policy"]))
        elif gen in ("sulfonylurea", "acei"):
            src = meta["source"]
            docs.append(_doc(meta, meta["steps"], [_source(T("protocol.cite_join", head=src["cite"], tail=src["locator"]),
                             src["url"], meta["ours_reason"])]))
            continue
        else:
            raise ValueError(f"unknown protocol generator: {gen}")
        docs.append(_doc(meta, steps, [_source(L("protocol.source.med_table"), "", L("protocol.source.med_policy"))]))
    for meta in cfg["other"]:
        import re
        src = meta["source"]
        match = re.search(r"NICE (?:NG|CG)\d+", src["cite"])
        cite = T("protocol.cite_join", head=(match[0] if match else src["cite"].split(",")[0].split(" (")[0]), tail=src["locator"])
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
    from haenv.i18n.identity import source
    block = source(block)
    from haenv.world.labworld.meds import drug, meds
    keys = meds()["drugs"]
    key = next(k for k in keys if drug(k)["display"] == block.get("drug"))
    name = drug(key)["display"].split("（")[0]
    docs = block.get("protocol_library") or ()
    from haenv.i18n.text import is_text
    if is_text(docs, "protocol.library"):
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
              or gen == "sulfonylurea" and key == "gliclazide" or gen == "acei" and key == "enalapril"
              or gen == "levothyroxine" and key == "levothyroxine"):
            out.append("".join(doc["steps"]))
    return out
