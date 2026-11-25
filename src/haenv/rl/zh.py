"""The language of an rl record: English (the native text) or Chinese (the same item rendered in Chinese).

`realize` builds the English block; when the batch language is zh (`haenv.i18n.resolve()`: job, then
`HAENV_LANG`) it builds the same item a second time with `using("zh")`, every text drawn from the zh catalogs:
`registry/i18n/zh/templates/rl.yaml` (record templates, same keys as English) and
`registry/i18n/zh/rl/library.yaml` (the zh text of the rl libraries and rules: phrases keyed by their English
text, questions by id, per-rule names, displays, doses, history patterns and question terms). Every draw stays
keyed on the English text, so both languages carry the same item; a phrase with no zh text fails closed.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import contextlib
import functools
import threading

_LOCAL = threading.local()
LIBRARY = "registry/i18n/zh/rl/library.yaml"


def lang() -> str:
    return getattr(_LOCAL, "lang", "en")


def zh() -> bool:
    return lang() == "zh"


@contextlib.contextmanager
def using(language: str):
    old = lang()
    _LOCAL.lang = language
    try:
        yield
    finally:
        _LOCAL.lang = old


def build_lang() -> str:
    """The language the batch is built in (job / `HAENV_LANG`, default English)."""
    from haenv.i18n import resolve
    return resolve()


@functools.lru_cache(maxsize=None)
def library() -> dict:
    from haenv.locations import resource_path
    from haenv.resources.yamlcache import load_yaml
    p = resource_path(LIBRARY)
    return (load_yaml(p, wrap_text=False) or {}) if p.is_file() else {}


def ph(text):
    """`text` in the current language (a phrase of the English libraries; identity in English)."""
    if not zh() or text is None:
        return text
    z = zh_of(text)
    if z is None:
        from haenv.i18n.text import MissingText
        raise MissingText(f"rl phrase {text!r}", "zh")
    return z


def zh_of(text) -> str | None:
    """The zh text of an English phrase or rule display, or None (language-independent; readers of both
    languages use it)."""
    z = (library().get("phrases") or {}).get(str(text))
    return z if z is not None else _rule_displays().get(str(text))


@functools.lru_cache(maxsize=None)
def _rule_displays() -> dict:
    """English trigger / level display of a rule -> its zh display (displays not among the phrases)."""
    from haenv.rl.rules import load
    out = {}
    for r in load().values():
        z = rule(r)
        if r["trigger"].get("display") and z.get("display"):
            out.setdefault(r["trigger"]["display"], z["display"])
        lv = (r.get("followup") or {}).get("level")
        if lv and z.get("level_display"):
            out.setdefault(lv["display"], z["level_display"])
    return out


def both(text) -> list[str]:
    z = zh_of(text)
    return [text] + ([z] if z and z != text else [])


def rule(r: dict) -> dict:
    return (library().get("rules") or {}).get(r["id"]) or {}


def rf(r: dict, field: str, en):
    """A rule's field in the current language (`en` is the English value)."""
    if not zh() or en is None:
        return en
    v = rule(r).get(field)
    if v is None:
        from haenv.i18n.text import MissingText
        raise MissingText(f"rl rule {r['id']}.{field}", "zh")
    return v


def drug_names(r: dict) -> list[str]:
    """The zh names a record uses for the item drug (ingredient and its Latin brand names): facts readers
    match these besides the English terms. Class names are left out: a zh class word is a substring of other
    drugs' names (他汀 in 阿托伐他汀)."""
    z = rule(r)
    out = [z["ingredient"]] if z.get("ingredient") else []
    return out + [n for n in z.get("names") or () if n.isascii() and n not in out]


def interacting_names(r: dict) -> dict[str, list[str]]:
    return {str(k): list(v) for k, v in (rule(r).get("interacting") or {}).items()}


def displays(r: dict) -> list[str]:
    z = rule(r)
    return list(z.get("displays") or ()) + ([z["level_display"]] if z.get("level_display") else [])


def item_terms(r: dict) -> list[str]:
    """The zh terms a zh question must not carry for this rule (G2)."""
    z = rule(r)
    out = [z.get("ingredient", "")] + list(z.get("names") or ()) + list(z.get("g2_terms") or ())
    out += [z.get("display", "")] + displays(r)
    for v in (z.get("interacting") or {}).values():
        out += list(v)
    return sorted({x for x in out if x and len(x) >= 2 and _cjk(x)})


def judge_terms(r: dict) -> list[str]:
    """The zh pre-screen terms of the judge for this rule: the zh ingredient, names (brands, classes) and
    interacting medicines, plus the zh words for its must-flag terms (`flag_terms`). The judge reads them as
    `zh_terms` in `registry/redline_rules.yaml` (segment RL_JUDGING); a test keeps the two equal."""
    z = rule(r)
    out = [z.get("ingredient", "")] + list(z.get("names") or ()) + list(z.get("flag_terms") or ())
    for v in (z.get("interacting") or {}).values():
        out += list(v)
    return sorted({x.strip() for x in out if x and x.strip() and _cjk(x)})


def generic_terms() -> list[str]:
    return [x for x in library().get("generic_terms") or () if _cjk(x)]


def _cjk(x: str) -> bool:
    """Only Chinese terms join the zh question gate (a Latin term would change the English gate)."""
    from haenv.i18n.text import has_cjk
    return has_cjk(x)


def question(qid: str) -> str:
    return library()["questions"][qid]


def match_all(r: dict) -> list[str]:
    return list(rule(r).get("match_all") or ())


def ancestry_zh(text: str) -> str:
    for r in (library().get("rules") or {}).values():
        if text in (r.get("ancestry") or {}):
            return r["ancestry"][text]
    return ph(text)


def ancestry_en(text: str | None) -> str | None:
    """A zh ancestry back to the English the rule names (the facts compare English values)."""
    if text is None:
        return None
    for r in (library().get("rules") or {}).values():
        for en, z in (r.get("ancestry") or {}).items():
            if text == z:
                return en
    return text
