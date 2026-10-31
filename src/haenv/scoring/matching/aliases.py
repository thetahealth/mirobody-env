"""Whether a text asserts a gold alias: hit, mention and negation are separate quantities.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import re
from functools import lru_cache


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


@lru_cache(maxsize=4096)
def _carried_exclusions(alias: str):
    """Offsets of shorter exclusion keys, except phrases naming the alias itself."""
    from haenv.scoring.matching.excludes import ALIAS_EXCLUDES
    out = []
    for key, phrases in ALIAS_EXCLUDES.items():
        if key == alias or key not in alias:
            continue
        own = tuple(b.lower() for b in phrases if b.lower() not in alias)
        off = alias.find(key)
        while off >= 0:
            if own:
                out.append((off, len(key), own))
            off = alias.find(key, off + 1)
    return tuple(out)


def _carried_excluded_at(low, alias, pos):
    for off, size, phrases in _carried_exclusions(alias):
        for phrase in phrases:
            i = low.find(phrase)
            while i >= 0:
                if i <= pos + off and pos + off + size <= i + len(phrase):
                    return True
                i = low.find(phrase, i + 1)
    return False


def _alias_spans(text: str, names):
    """Occurrences `(alias, start, length)` of gold aliases, minus exclusion
    words (`overlay.alias_excluded_at`, e.g. polycystic kidney disease for
    PCOS). Negated occurrences are kept.
    """
    from haenv.world.observations.events_text import alias_hit
    from haenv.scoring.matching.excludes import alias_excluded_at
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
            if not alias_excluded_at(t, a, j) and not _carried_excluded_at(low, al, j):
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
    names = list(names or ())
    concept = _qualified_concept(t, names)
    if concept is not None:
        return concept
    for _a, j, n in _alias_spans(t, names):
        if not _negated(t, j, n):
            return True
    return False


@lru_cache(maxsize=1)
def _concepts():
    from haenv.resources.registry import _load_yaml
    entries = _load_yaml("threads.yaml").get("diagnosis_concepts", {})
    for key, entry in entries.items():
        if not entry.get("source") or not entry.get("anchors") or not entry.get("core"):
            raise ValueError(f"Unsourced or incomplete diagnosis concept: {key}")
    return entries.values()


def _asserted_terms(text, terms):
    # ASCII abbreviations are tokens; CKD must not match XCKD or AKI match JAKI.
    for term in terms:
        pattern = re.escape(term)
        if term.isascii():
            pattern = r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])"
        for hit in re.finditer(pattern, text, re.I):
            prefix = text[max(0, hit.start() - 48):hit.start()].lower()
            absent = re.search(
                r"(?:\bno|without)(?:\s+(?:any|clear|objective|convincing))?\s+evidence\s+(?:of|for)\s*$",
                prefix)
            if (not _negated(text, hit.start(), len(hit.group()))
                    and not absent
                    and not re.search(r"(?:无|没有|未见|不伴|非|\bno|\bnot|without|not from|not due to|non-)\s*$", prefix)):
                yield hit


#: A hedge names a term as an open question rather than stating it. Windows are
#: scoped to the term's own phrase: they end at a clause boundary or a bracket.
_HEDGE_SCOPE_END = re.compile(r"[，,；;。！？!?\n()（）\[\]【】]")

#: Before the core or a qualifier: the term is something to watch for, prevent, or
#: ask about ("需警惕急性肾损伤", "是否进展", "watch for progression").
_HEDGE_BEFORE = ("是否", "有无", "有没有", "警惕", "提防", "谨防", "预防", "防止", "避免",
                 "whether", "watch for", "watch out for", "beware of", "look out for",
                 "monitor", "prevent", "avoid", "risk of", "at risk")

#: Before a qualifier only: the qualifier is to be checked or ruled out ("需排除进展",
#: "screen for volume depletion"). The candidate's own disease pending exclusion stays
#: listed (D9), but a required qualifier left to be ruled out is not stated.
_HEDGE_BEFORE_QUALIFIER = ("排除", "除外", "排查", "筛查", "rule out", "exclude", "screen for",
                           "check for", "look for", "evaluate for", "assess")

#: After a qualifier, in its own phrase: the qualifier is under assessment, pending
#: exclusion or undetermined ("进展评估中", "进展与否待明确", "progression being assessed").
#: "待确认" (to be confirmed) is a degree of suspicion, like "疑似", and is not a hedge.
_HEDGE_AFTER_QUALIFIER = re.compile(
    r"(?:有待|尚需|仍需|需要|待|需)(?:进一步)?(?:评估|排除|排查|明确|确定|观察|随访|监测)"
    r"|评估中|观察中|随访中|监测中|与否|待定|未定|待排"
    r"|\bbeing (?:assessed|evaluated|investigated|monitored)\b"
    r"|\bunder (?:evaluation|assessment|investigation|observation|review)\b"
    r"|\bto be (?:assessed|evaluated|determined|established|ruled out|excluded)\b"
    r"|\bnot (?:yet )?established\b")

_HEDGE_EXEMPT = re.compile("|".join(re.escape(k) for k in _NEG_EXEMPT))


def _hedged(text, start, end, *, qualifier):
    """Whether the term at `text[start:end]` is named as an open question, not stated.

    The candidate's own disease (`qualifier=False`) may carry any degree of suspicion
    ("疑似", "possible", "待确认", a query mark), as D9 counts possible entries, but
    not be something to watch for or ask about. A required qualifier must also not be
    pending exclusion, under assessment, undetermined or queried ("进展？").
    "Cannot rule out progression" ("不排除进展") is a degree of suspicion, not a hedge.
    """
    before = _HEDGE_SCOPE_END.split(text[max(0, start - 16):start])[-1].lower()
    before = _HEDGE_EXEMPT.sub(" ", before)
    cues = _HEDGE_BEFORE + (_HEDGE_BEFORE_QUALIFIER if qualifier else ())
    if any(k in before for k in cues):
        return True
    if not qualifier:
        return False
    tail = text[end:end + 32]
    stop = _HEDGE_SCOPE_END.search(tail)
    after = (tail[:stop.start()] if stop else tail).lower()
    return bool(_HEDGE_AFTER_QUALIFIER.search(after)) or bool(stop and stop.group() in "?？")


def _qualified_concept(text, names):
    """Registered gold concepts require every qualifier, within one candidate.

    Qualifier terms cannot be borrowed from another diagnosis or a discussion of
    future risk. Numbers alone do not assert a course. A hedged core or qualifier
    (`_hedged`) is not stated. Unknown concepts retain the existing
    alias/exclusion/negation matcher.
    """
    anchors = {str(n).strip().casefold() for n in names}
    for entry in _concepts():
        if not anchors.intersection(str(a).casefold() for a in entry["anchors"]):
            continue
        if not any(not _hedged(text, hit.start(), hit.end(), qualifier=False)
                   for hit in _asserted_terms(text, entry["core"])):
            return False
        for terms in entry.get("required", []):
            found = False
            for hit in _asserted_terms(text, terms):
                before = text[max(0, hit.start() - 12):hit.start()].lower()
                after = text[hit.end():hit.end() + 14].lower()
                bindings = [b for b in entry.get("qualifier_bindings", [])
                            if hit.group().casefold() in {t.casefold() for t in b["terms"]}]
                if any((b.get("following") and not re.match(b["following"], after))
                       or (b.get("reject_preceding") and re.search(b["reject_preceding"], before))
                       for b in bindings):
                    continue
                if any(x in before for x in ("是否", "预防", "防止", "风险", "risk of", "prevent", "whether", "monitor", "assess")):
                    continue
                if any(x in after for x in ("风险", "risk", "尚待确认", "尚待评估")):
                    continue
                if _hedged(text, hit.start(), hit.end(), qualifier=True):
                    continue
                found = True
                break
            if not found:
                return False
        return True
    return None
