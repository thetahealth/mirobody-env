"""Whether a text asserts a gold alias: hit, mention and negation are separate quantities.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------- hit judging
# hit (asserted), mentioned (entertained) and negated are separate quantities:
# a ruled-out or negated mention is not a hit.
_NEG_EXEMPT = ("不排除", "未排除", "不能排除", "无法排除", "不除外", "cannot rule out",
               "can't rule out", "not excluded")


_NEG_CUES = ("不像", "不支持", "不考虑", "暂不", "不要", "不宜", "不太可能", "可能性低",
             "否认", "排除", "已排除", "无证据", "不成立", "非典型",
             "unlikely", "ruled out", "rule out", "ruling out", "excluded", "exclude",
             "no evidence", "doubt")


#: Pending-exclusion phrasing: "X 待排除", "需排除 X", "需进一步分型并排除 X",
#: "X needs to be ruled out". The candidate is still on the differential, so the
#: "排除"/"rule out" inside these phrases does not negate it.
_PENDING_EXCLUSION = re.compile(
    r"(?:有待|尚需|仍需|需要|建议|进一步|待|需)[^,，。；;.!?！？、\n]{0,6}?排除"
    r"|\bneeds? to (?:be ruled out|rule out)\b"
    r"|\b(?:must|should) (?:be ruled out|rule out)\b"
    r"|\bto be ruled out\b|\bpending exclusion\b",
    re.I)


#: A negator just before the phrase keeps it a negation ("无需排除 X", "已进一步排除 X",
#: "does not need to be ruled out").
_PENDING_NEGATORS = ("无", "不", "毋", "未", "已", "勿", "no ", "not ")


#: A negator's reach ends at a clause boundary: in "不，需排除 X" or "功能不全；需排除 X" the
#: "不" belongs to the previous clause. "、" is not a boundary (it separates list items).
_CLAUSE_END = re.compile(r"[，,；;。！？!?\n]")


def _clause_tail(s: str) -> str:
    """The part of `s` after its last clause boundary."""
    return _CLAUSE_END.split(s)[-1]


def _mask_pending(text: str) -> str:
    """`text` with pending-exclusion phrases blanked out (same length, so offsets hold)."""
    def _sub(m):
        before = _clause_tail(text[max(0, m.start() - 4):m.start()]).lower()
        if any(k in before for k in _PENDING_NEGATORS):
            return m.group(0)
        return " " * len(m.group(0))
    return _PENDING_EXCLUSION.sub(_sub, text)


def _negated(text: str, at: int, span: int) -> bool:
    """Whether the alias at `text[at:at+span]` is negated, from a narrow window
    (10 chars before, 12 after). Exemptions such as "cannot rule out" are checked
    first; only multi-character cues count. Pending-exclusion phrases are blanked
    out of the text before the cues are read, so another cue in the same window
    still negates.
    """
    win = (text[max(0, at - 10):at] + " " + text[at + span:at + span + 12]).lower()
    if any(k in win for k in _NEG_EXEMPT):
        return False
    m = _mask_pending(text)
    win = (m[max(0, at - 10):at] + " " + m[at + span:at + span + 12]).lower()
    return any(k in win for k in _NEG_CUES)


def _alias_spans(text: str, names):
    """Occurrences `(alias, start, length)` of gold aliases, minus exclusion
    words (`overlay.alias_excluded_at`, e.g. polycystic kidney disease for
    PCOS). Negated occurrences are kept.
    """
    from .events_text import alias_hit
    from .alias_excludes import alias_excluded_at
    t = str(text or "")
    low = t.lower()
    # Scoring accepts a full word for a stem alias (`hypothyroid` ->
    # `Hypothyroidism`); leak scanning does not.
    for a in alias_hit(t, names, allow_suffix=True):
        al = str(a).lower()
        i, n = 0, len(al)
        while True:
            j = low.find(al, i)
            if j < 0:
                break
            if not alias_excluded_at(t, a, j):
                yield a, j, n
            i = j + 1


def alias_mentioned(text: str, names) -> bool:
    """True if the text mentions the gold diagnosis (asserted or negated),
    ignoring exclusion words."""
    for _ in _alias_spans(text, names):
        return True
    return False


def alias_hit_asserted(text: str, names) -> bool:
    """True if at least one valid occurrence is not negated. Judging side only;
    leak scanning does not use this filter."""
    t = str(text or "")
    for _a, j, n in _alias_spans(t, names):
        if not _negated(t, j, n):
            return True
    return False
