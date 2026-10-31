"""Registry policy and exact record readers shared by generation and judging."""
from __future__ import annotations

from haenv.i18n.text import L, render, Msg, MissingText
from haenv.resources.regpath import registry_cached

@registry_cached("p3_clinical_sources.yaml")
def specification():
    from haenv.resources.regpath import load_registry
    return load_registry("p3_clinical_sources.yaml")["question_update"]


def drug_keys():
    from haenv.world.labworld.meds import meds
    return tuple(meds()["drugs"])


@registry_cached("p3_meds.yaml")
def _drug_identities():
    """Build exact names once; registry overrides invalidate this lookup."""
    from haenv.world.labworld.meds import meds
    import haenv.i18n as i18n
    ids, names = {}, {}
    for key, row in meds()["drugs"].items():
        phrases = (L(f"drug.{key}"), row["display"])
        names.setdefault(key, key)
        for phrase in phrases:
            if isinstance(phrase, Msg):
                ids.setdefault(phrase.key, key)
        for lang in i18n.LANGUAGES:
            for phrase in phrases:
                try:
                    name = str(render(phrase, lang)).casefold()
                except MissingText:
                    continue
                for variant in (name, name.split("（", 1)[0]):
                    names.setdefault(variant, key)
    return ids, names


def identify_drug(value):
    """Exact registry IDs/names for source reconstruction; no fuzzy matching."""
    ids, names = _drug_identities()
    if isinstance(value, Msg) and value.kind == "lex" and value.key in ids:
        return ids[value.key]
    return names.get(str(value or "").strip().casefold())


def pill_count_slots(text):
    from haenv.p3.textread import slots
    keys = ["p3.pill_count", "p3.pill_count.legacy", *specification()["surface"]["pill_count_templates"]]
    return next((found for key in keys if (found := slots(text, key)) is not None), None)


def absorption_interaction(record):
    cfg = specification()["levothyroxine_absorption"]
    return bool(record and record.get("drug") == "levothyroxine"
                and record.get("supplement") in cfg["supplements"]
                and abs(float(record["drug_hour"]) - float(record["supplement_hour"])) < cfg["minimum_separation_hours"])

