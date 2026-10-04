"""Words that contain a gold alias without naming that disease ("polycystic" in "polycystic kidney disease").

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

# Scoring side only: the leak scan keeps the raw vocabulary.
from .registry import load_alias_excludes as _load_alias_excludes

ALIAS_EXCLUDES = _load_alias_excludes()


def alias_excluded_at(text: str, alias: str, pos: int) -> bool:
    """Whether this occurrence of `alias` at `pos` lies inside an exclusion phrase
    (positional containment, not "the phrase appears somewhere in the text")."""
    low, a = str(text or "").lower(), str(alias or "").lower()
    for bad in ALIAS_EXCLUDES.get(a, ()) :
        b, i = bad.lower(), 0
        while True:
            j = low.find(b, i)
            if j < 0:
                break
            if j <= pos and pos + len(a) <= j + len(b):    # this hit is fully contained within `bad`
                return True
            i = j + 1
    return False
