"""Text matching shared by the event layer, the judges and the overlay: condition-alias hits,
annotation scrubbing, and the symptom-topic vocabulary.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import haenv.world.generation.world_plugins as _wp


# Morphological suffixes allowed after a Latin alias stem on the scoring side
# only (`hypothyroid` -> `Hypothyroidism`). A closed set: `sle` must not match
# `sleep`, `insulin` must not match `insulinoma`.
_LATIN_SUFFIXES = ("ism", "osis", "oses", "ic", "ical", "ia", "emia", "aemia",
                   "y", "ies", "s", "es", "al", "ous", "otic", "oidism")


def alias_hit(text: str, aliases, allow_suffix: bool = False) -> list[str]:
    """Diagnosis aliases found in the text.

    Latin aliases match on word boundaries (underscore counts as a word
    character); Chinese aliases match by substring. `allow_suffix=True`
    (scoring only) also accepts a suffix from `_LATIN_SUFFIXES`; the leak
    scanner keeps it off to err toward false positives.
    """
    import re
    out = []
    for a in (aliases or []):
        a = str(a).strip()
        if not a:
            continue
        if a.isascii():
            tail = (rf"(?:{'|'.join(_LATIN_SUFFIXES)})?" if allow_suffix else "")
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(a)}{tail}(?![A-Za-z0-9_])",
                         text, re.I):
                out.append(a)
        elif a.casefold() in text.casefold():
            out.append(a)
    return out


# Parentheticals that mark a disease thread ("(diabetes thread)") are author
# annotations and would reveal join_gold; they are scrubbed when they contain
# the case's diagnosis alias or one of these markers. Clinical content in
# parentheses is kept.
ANNOTATION_MARKERS = ("线程",)


def scrub_annotation(text: str, aliases) -> tuple[str, list[str]]:
    """Scrub parentheticals naming this case's diagnosis or marking a thread.
    Returns (scrubbed text, removed parentheticals)."""
    import re
    removed: list[str] = []

    def _sub(m):
        inner = m.group(2)
        if alias_hit(inner, aliases) or any(k in inner for k in ANNOTATION_MARKERS):
            removed.append(m.group(0))
            return ""
        return m.group(0)

    out = re.sub(r"([((])([^))]*)([))])",
                 lambda m: _sub(re.match(r"([((])([^))]*)([))])", m.group(0))), text)
    return out.strip(" 、,,+"), removed


_INTERP_CLAUSE = __import__("re").compile(r"[、,\uff0c]?[^、,\uff0c]*倾向(?=[、,\uff0c]|$)")


def scrub_lab_interp(text: str) -> str:
    """Drop each clause that ends in 倾向 (a lab-derived reading: "低钠倾向", "偶有酮症倾向"); a text that
    would be left empty stays as it is."""
    s = str(text or "")
    out = _INTERP_CLAUSE.sub("", s).lstrip("、,\uff0c ")
    return out if out.strip() else s


_MENSES_WORDS = ("月经", "例假", "经期", "经量")


def scrub_menses(text: str, keep_empty: bool = True) -> str:
    """Drop each clause that names menses; a text that would be left empty stays as it is (`keep_empty`)."""
    import re
    s = str(text or "")
    parts = re.split(r"([、,\uff0c])", s)
    kept = [x for x in parts[0::2] if not any(w in x for w in _MENSES_WORDS)]
    out = "、".join(k for k in kept if k.strip()) if len(kept) != len(parts[0::2]) else s
    if out.strip():
        return out
    return s if keep_empty else ""


@_wp.overlay_cached("symptom_topics.yaml", "symptom_penetrance.yaml", "benign_events.yaml")
def _load_symptom_topics() -> dict[str, str]:
    from haenv.resources.regpath import load_registry as _lr
    _d = _lr("symptom_topics.yaml") or {}
    out = {_norm_symptom(k): str(v["topic"])
           for k, v in (_d.get("symptoms") or {}).items()
           if isinstance(v, dict) and v.get("topic")}
    # Synonym wordings (`symptom_penetrance.yaml`, composition v2) inherit the canonical text's topic,
    # so a reworded symptom keeps the same physiology footprint as the original.
    _pen = _lr("symptom_penetrance.yaml") or {}
    for _rows in (_pen.get("symptoms") or {}).values():
        for _r in _rows:
            _t = out.get(_norm_symptom(_r.get("text", "")))
            if _t is None:
                continue
            for _lvl in (_r.get("variants") or {}).values():
                for _v in _lvl:
                    out.setdefault(_norm_symptom(_v), _t)
    # A benign complaint may be drawn directly into a patient's raw history by
    # any pack. It must retain the very same topic/footprint it has when drawn
    # by the event layer; the registry is the single source for both paths.
    for row in (_lr("benign_events.yaml") or {}).get("benign_events") or ():
        if row.get("topic"):
            out.setdefault(_norm_symptom(row["text"]), str(row["topic"]))
    return out


def symptom_topic(text: str) -> str | None:
    """Genuine symptom text -> physiology `topic` from the hand-written
    `registry/symptom_topics.yaml` (plus world-plugin registrations), or `None` if not
    registered.

    Exact match on the normalized full text; similar prefixes can mean
    different things clinically.
    """
    return _load_symptom_topics().get(_norm_symptom(text))


def _norm_symptom(text: str) -> str:
    """Normalize whitespace and full-width punctuation (used for registry and lookup)."""
    t = str(text or "").strip()
    for a, b in (("（", "("), ("）", ")"), ("，", ","), ("、", ","), ("：", ":")):
        t = t.replace(a, b)
    return "".join(t.split())


def relocate_event_history(raw) -> None:
    """Keep contextual history in dated encounter notes, outside symptom values.

    All participating packs call this after realization. The source section is
    present for every patient; its rows preserve existing facts and dates. No
    new explanation is generated and no gold/role field is consulted.
    """
    pc = dict(raw.prediction_context or {})
    if "event_history_record" in pc:
        return
    rows = []
    for event in raw.evidence_ledger or ():
        if event.get("symptom") and event.get("context"):
            rows.append({"day": int(event["source_timestamp"]), "source": "门诊与生活史记录",
                         "symptom": event["symptom"], "history": event.pop("context")})
    pc["event_history_record"] = rows
    raw.prediction_context = pc
