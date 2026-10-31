"""Language of a batch: English (default) or Chinese.

The generator uses canonical message identifiers. Each language renders the same world
through its own versioned catalogs under `registry/i18n/`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import os

#: Supported languages.
LANGUAGES = ("en", "zh")
#: Default language of a new batch; missing English assets fail closed.
DEFAULT = "en"
#: Language of the canonical world: the generator renders this one.
CANONICAL = "zh"
#: A batch written before languages existed records none; it is canonical.
LEGACY = CANONICAL
ENV = "HAENV_LANG"


class LanguageError(ValueError):
    """An unsupported language name."""


def check(lang) -> str:
    s = str(lang or "").strip().lower()
    if s not in LANGUAGES:
        raise LanguageError(f"language {lang!r} is not one of {LANGUAGES}")
    return s


def resolve(cli=None, job=None, cfg=None) -> str:
    """The language a new batch is built in: `--lang` > job `language:` > `HAENV_LANG` >
    config `language:` > the default (English)."""
    for value in (cli, (job or {}).get("language") if isinstance(job, dict) else job,
                  os.environ.get(ENV), (cfg or {}).get("language")):
        if value not in (None, ""):
            return check(value)
    return DEFAULT


def batch_language(meta: dict | None) -> str:
    """The language a batch was built in; batches without the field are canonical."""
    return check((meta or {}).get("language") or LEGACY)


_CURRENT = [CANONICAL]


def set_current(lang: str) -> None:
    """The language this process renders and reads records in (set by the CLI from the batch)."""
    _CURRENT[0] = check(lang)


def current() -> str:
    return _CURRENT[0]
