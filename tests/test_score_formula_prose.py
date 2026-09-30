"""A leaderboard's own explanation of its scoring formula must be true.

    uv run pytest tests/test_score_formula_prose.py -q          # seconds

## What this guards against

`report.py` prints a "how is the headline score computed" explanation
under every leaderboard, and a line of it can assert something false, such as
an equality between `n_core_total` and the slot count of the scoring profile.
The two numbers do not compute the same thing:

* the number in the explanation iterates the full slot count of the
  scoring profile (9)
* `n_core_total = len(core)`, where `core` comes from `_core_pairs`
  applicable to this batch (for example 3 on the single-shot geometry, where
  the four tool dimensions do not apply at that tier)

A leaderboard's own explanation of a number must be true; asserting a false
equality is worse than not explaining at all. The tests scan the rendering
code for the false identity and check that the two numbers come from their
own sources.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
# The scan surface follows the code: the headline-score formula is
# implemented in `analytics.py` (a pure computation layer) and rendered in
# `report.py`. Reading only one of them would leave the assertions below
# scanning an empty surface.
_SRC = "\n".join((ROOT / "haenv" / f"{m}.py").read_text(encoding="utf-8")
                 for m in ("report", "analytics"))


def test_prose_does_not_assert_the_false_identity():
    """The explanation must no longer assert "`n_core_total` reports <the slot count>."

    The search matches the rendering expression, not natural-language text:
    this file's own comments describe that sentence, and a bare-substring
    search would flag them as violations.
    """
    bad = re.findall(r'f"\(`n_core_total` 报的是前者', _SRC)
    assert not bad, "渲染里还在断言那个不成立的等式"


def test_prose_explains_why_the_two_numbers_differ():
    """Deleting a false statement isn't enough -- the reader will still see two different numbers.

    Their difference must be explained, or "no explanation" will be read
    as "one of them is wrong."
    """
    assert "actually applicable to this batch" in _SRC, "没解释 `n_core_total` 是本批适用维数"
    assert "not supposed to be equal in the first place" in _SRC, "没说清两个数本来就不该相等"


def test_slot_count_and_batch_count_come_from_different_sources():
    """The two numbers must each come from their own source -- nobody is allowed to unify them into one.

    Unifying them would collapse "how many dimensions this scoring
    scheme defines" and "how many dimensions apply to this batch" into a
    single number -- and different geometries genuinely apply to
    different numbers of dimensions; that's a real difference.
    """
    assert "_cap_slots.append(" in _SRC, "名额表的构造点没了"
    assert 'rec["n_core_total"] = len(core)' in _SRC, "本批适用维数的取法变了"
