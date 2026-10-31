"""Vary equivalent sourced record wording without changing clinical events."""
from haenv.i18n.text import L, Msg, lexicon
from haenv.m2.evidence import registry
from haenv.world.generation.rng import below


def vary(obj, case_id):
    table = registry().get("surface_variants") or {}
    def walk(value):
        if isinstance(value, dict):
            return {k: walk(v) if k in {"text", "symptom", "history", "patient_quote", "m2_frame"}
                    or isinstance(v, (dict, list)) else v for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        key = value.key if isinstance(value, Msg) and value.kind == "lex" else value
        if isinstance(key, str) and key in table:
            entry = table[key]
            if not entry.get("source") or len(entry.get("messages") or []) < 2:
                raise ValueError("surface variants require a source and multiple equivalent messages")
            if any(k not in lexicon("zh") for k in entry["messages"]):
                raise ValueError("surface variant message missing from the canonical lexicon")
            return L(entry["messages"][below(len(entry["messages"]), case_id, "m2-surface", key)])
        return value
    return walk(obj)
