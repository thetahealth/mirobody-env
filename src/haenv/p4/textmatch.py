"""Read pack-4 record text by what it was rendered from, in any language.

The readers of the visible block (`features`, `diversity_reader`, `query`, the follow-up policy)
run on built cases, whose texts are `Msg` values that remember their template or lexicon id,
and on persisted batches, whose texts are plain strings in the batch language. These helpers
answer "is this the phrase / template X" in both cases, so no reader compares against literal
Chinese text.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import re
from functools import lru_cache

from haenv.i18n.text import Msg, is_text as _is_text

_SLOT = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?:![rsa])?(?::[^{}]*)?\}")


@lru_cache(maxsize=None)
def _pattern(key: str, lang: str):
    from haenv.i18n.text import templates
    tpl = templates(lang).get(key)
    if tpl is None:
        return None
    out, pos = [], 0
    for m in _SLOT.finditer(tpl):
        out.append(re.escape(tpl[pos:m.start()].replace("{{", "{").replace("}}", "}")))
        out.append(f"(?P<{m.group(1)}>.+?)")
        pos = m.end()
    out.append(re.escape(tpl[pos:].replace("{{", "{").replace("}}", "}")))
    return re.compile("^" + "".join(out) + "$", re.S)


def from_template(value, *keys: str) -> bool:
    """Whether `value` was rendered from one of the templates `keys`."""
    if isinstance(value, Msg):
        return value.kind == "tpl" and value.key in keys
    if not isinstance(value, str):
        return False
    import haenv.i18n as _i18n
    for lang in _i18n.LANGUAGES:
        for k in keys:
            p = _pattern(k, lang)
            if p is not None and p.match(value):
                return True
    return False


def slot(value, key: str, name: str):
    """The text of slot `name` of a `value` rendered from template `key` (None when it was not)."""
    if isinstance(value, Msg):
        return value.slots.get(name) if value.kind == "tpl" and value.key == key else None
    if not isinstance(value, str):
        return None
    import haenv.i18n as _i18n
    for lang in _i18n.LANGUAGES:
        p = _pattern(key, lang)
        m = p.match(value) if p is not None else None
        if m:
            return m.group(name)
    return None


def phrase_id(value):
    """The registered identity, independent of a phrase's rendered characters."""
    return value.key if isinstance(value, Msg) and value.kind == "lex" else value


def is_text(value, msgid) -> bool:
    return _is_text(value, phrase_id(msgid))


def is_any(value, msgids) -> bool:
    """Whether a field carries one of these registered phrase identities."""
    return any(is_text(value, m) for m in msgids)


def field_id(value):
    """An in-memory record field's structured identity, including template arguments."""
    if isinstance(value, Msg):
        return value.kind, value.key, tuple((k, field_id(v)) for k, v in value.slots.items())
    if isinstance(value, dict):
        return tuple((k, field_id(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(field_id(v) for v in value)
    return value


def same_field(left, right):
    """Match structured fields; persisted fields already share the batch's language."""
    if isinstance(left, Msg) and isinstance(right, Msg):
        return field_id(left) == field_id(right)
    return left == right
