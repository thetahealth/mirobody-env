"""Read a pack-3 record back through the rendering layer, in whatever language it was rendered.

The question-blind readers (`emr.clinical_view`, `features`, the query tools) and the emission
consistency check read the visible record the solver sees. During generation its text is `Msg`
values (template keys and slots); a persisted record is plain text in its batch language. These
helpers answer "is this the phrase X" and "which slots does this text carry for template K" in both
cases, so no reader compares against literal text of one language.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import re
import string
from functools import lru_cache

from haenv.i18n.text import Msg, has_cjk, is_text, templates


@lru_cache(maxsize=None)
def _pattern(lang: str, key: str):
    tpl = templates(lang).get(key)
    if tpl is None:
        return None
    parts, names = [], []
    for lit, name, _spec, _conv in string.Formatter().parse(tpl):
        parts.append(re.escape(lit))
        if name is not None:
            group = name if name not in names else None
            names.append(name)
            parts.append(f"(?P<{group}>.*?)" if group else ".*?")
    return re.compile("^" + "".join(parts) + "$", re.S)


def slots(value, key: str) -> dict | None:
    """The slot texts of `value` when it is template `key` (a `Msg` of that key, or plain text
    matching the template in some language), else `None`."""
    if isinstance(value, Msg):
        if value.kind == "tpl" and value.key == key:
            return {k: v for k, v in value.slots.items()}
        return None
    if not isinstance(value, str):
        return None
    import haenv.i18n as _i18n
    for lang in _i18n.LANGUAGES:
        rx = _pattern(lang, key)
        m = rx.match(value) if rx is not None else None
        if m:
            return m.groupdict()
    return None


def lines(text) -> list:
    """The lines of a note: the items of a `lines` join, else the text split at newlines."""
    if isinstance(text, Msg):
        return list(text.slots["items"]) if text.kind == "join" and text.key == "lines" else [text]
    return str(text or "").split("\n")


def field(text, key: str):
    """The slot `text` of the first line of `text` that is the line template `key`
    (`p3.emr.line.*`), else `None`."""
    for line in lines(text):
        got = slots(line, key)
        if got is not None:
            return got.get("text")
    return None


def is_one(value, *ids) -> bool:
    return any(is_text(value, i) for i in ids)


def number(s) -> float:
    """A recorded numeric slot, or the first number in a legacy persisted value."""
    if isinstance(s, Msg):
        for key in ("value", "sup"):
            if key in s.slots:
                return float(s.slots[key])
        raise ValueError(f"no numeric slot in {s.key}")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(s))
    if match is None:
        raise ValueError("recorded numeric value is missing")
    return float(match.group(1))


def prescription_dose(text) -> float:
    """Dose from a prescription's slots, with compatibility for persisted prescriptions."""
    for key in ("p3.rx.daily", "p3.rx.daily_unit", "med.line", "med.line_per_day"):
        got = slots(text, key)
        if got is not None:
            return number(got["dose"])
    if isinstance(text, Msg):
        raise ValueError(f"no prescription dose slot in {text.key}")
    return float(re.search(r"([0-9.]+) (?:mg|μg)", text)[1])


@lru_cache(maxsize=None)
def _reverse(lang: str) -> dict[str, str]:
    from haenv.i18n.text import lexicon
    out: dict[str, str] = {}
    for mid, text in lexicon(lang).items():
        out.setdefault(text, mid)
    return out


def canonical(value) -> str:
    """The canonical (zh) text of a phrase, also for a lexicon phrase printed in another language,
    so a record sorts the same whatever language it is printed in."""
    from haenv.i18n.text import CANONICAL, lexicon
    if isinstance(value, Msg):
        if value.kind == "lex":
            # Registry phrases are their own IDs, independent of their rendering.
            return value.key if has_cjk(value.key) else lexicon(CANONICAL).get(value.key, value.key)
        return str.__str__(value)
    if not isinstance(value, str):
        return value
    import haenv.i18n as _i18n
    for lang in _i18n.LANGUAGES:
        if lang == CANONICAL:
            continue
        mid = _reverse(lang).get(value)
        if mid is not None:
            return lexicon(CANONICAL).get(mid, mid)
    return value


@lru_cache(maxsize=None)
def _search(lang: str, key: str):
    rx = _pattern(lang, key)
    return re.compile(rx.pattern[1:-1], re.S) if rx is not None else None


def find(value, key: str) -> dict | None:
    """The slots of the first occurrence of template `key` inside `value`: a `Msg` of that key, one
    nested in another `Msg`'s slots or join items, or a match anywhere in a plain text."""
    if isinstance(value, Msg):
        if value.kind == "tpl" and value.key == key:
            return dict(value.slots)
        inner = (list(value.slots.values()) if value.kind == "tpl" else
                 list(value.slots.get("items") or ()) if value.kind == "join" else
                 [value.slots.get("base")] if value.kind == "numsub" else [])
        for v in inner:
            got = find(v, key)
            if got is not None:
                return got
        return None
    if not isinstance(value, str):
        return None
    import haenv.i18n as _i18n
    for lang in _i18n.LANGUAGES:
        rx = _search(lang, key)
        m = rx.search(value) if rx is not None else None
        if m:
            return m.groupdict()
    return None
