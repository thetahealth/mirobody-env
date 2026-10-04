"""Test recall, three-valued: proposal text -> gold test items.

For each gold test item the matcher answers

* ``yes``        code is certain the item was proposed: an equivalent name (or a fixed-composition
                 panel that contains it) occurs in a proposal that is neither negated nor conditional;
* ``no``         code is certain it was not: nothing in any proposal even resembles the item
                 (no form, no panel, no shared stem);
* ``undecided``  anything in between (a bare modality, a broader or related test, a panel whose
                 composition varies, a negated or conditional proposal, a shared stem). These go to
                 the semantic judge.

Two differences from the production one-to-one matcher in ``differential.judge_workup``:
one proposal may claim several gold items (a bundled proposal such as "甲功五项" or
"ANA、抗dsDNA、补体C3/C4" names each member), and the vocabulary adds Chinese/English
translations, abbreviations and panel compositions (``vocab_tests.yaml`` section ``recall``).

Scope: proposal text only. Whether a test already executed through the tool counts as covered is
decided elsewhere (D9, ``judge_workup``); the caller chooses which texts to pass in. The order-count
cap is also the caller's business.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

YES, NO, UNDECIDED = "yes", "no", "undecided"

_SEP = re.compile(r"[\s\-_·•・/\\,，、;；:：()（）\[\]【】{}+＋&\"'“”‘’<>《》|=~→]+")
_ASCII_ALNUM = re.compile(r"[a-z0-9]")


_CJK_SPACE = re.compile(r"(?<=[^\x00-\x7f]) | (?=[^\x00-\x7f])")


def _norm(s: str) -> str:
    """NFKC, case-fold, every separator -> one space, and no space next to a non-ASCII character
    (so "TSH 受体抗体" and "tsh受体抗体" normalise alike; spaces survive only between ASCII words)."""
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return _CJK_SPACE.sub("", _SEP.sub(" ", s).strip())


def _variants(s: str) -> tuple[str, str]:
    v1 = _norm(s)
    return v1, v1.replace(" ", "")


def _is_ascii_alnum(ch: str) -> bool:
    return bool(ch) and ch.isascii() and bool(_ASCII_ALNUM.match(ch))


def _spans(needle: str, hay: str) -> list[tuple[int, int]]:
    """Occurrences of `needle` in `hay`; an ASCII-alnum edge of the needle needs a non-alnum
    neighbour (so `tsh` is not found in `ftsh` and `ana` not in `banana`)."""
    out, n = [], len(needle)
    if not n:
        return out
    i = hay.find(needle)
    while i >= 0:
        j = i + n
        left_ok = not (_is_ascii_alnum(needle[0]) and i > 0 and _is_ascii_alnum(hay[i - 1]))
        right_ok = not (_is_ascii_alnum(needle[-1]) and j < len(hay) and _is_ascii_alnum(hay[j]))
        if left_ok and right_ok:
            out.append((i, j))
        i = hay.find(needle, i + 1)
    return out


class _Text:
    """A proposal, normalised twice: separators as spaces (v1) and removed (v2)."""

    __slots__ = ("v1", "v2")

    def __init__(self, s: str):
        self.v1, self.v2 = _variants(s)

    def hit(self, form: str, shadows: tuple[str, ...] = ()) -> bool:
        """`form` occurs, and at least one occurrence is not inside a `shadows` term."""
        for fv, tv in zip(_variants(form), (self.v1, self.v2)):
            if not fv:
                continue
            sp = _spans(fv, tv)
            if not sp:
                continue
            cover = [c for d in shadows for c in _spans(_variants(d)[0 if tv is self.v1 else 1], tv)]
            if any(not any(a <= i and j <= b for a, b in cover) for i, j in sp):
                return True
        return False


@lru_cache(maxsize=1)
def _vocab() -> dict:
    from ..registry import load_test_recall_vocab
    from . import differential as D
    rv = load_test_recall_vocab()
    shadows: dict[str, tuple[str, ...]] = {}
    for k, ent in rv["distinct_from"].items():
        shadows[_norm(k)] = tuple(ent["terms"])
    return {"rv": rv, "prod_forms": D._seg_forms, "prod_segs": D._gold_test_segs, "shadows": shadows,
            "neg": rv["negation"], "cond": rv["conditional"], "past": rv["past_result"]}


def gold_segments(item: str) -> list[str]:
    """Matchable fragments of a gold item: the production splitter, with compound names kept whole
    (``1,25-(OH)2D``), extra prefixes stripped and timing/purpose notes dropped."""
    V = _vocab()
    rv = V["rv"]
    t = str(item or "")
    keep = {}
    for i, a in enumerate(rv["atomic_terms"]):
        if a in t:
            tok = f"ATOMICTERM{i}X"
            keep[tok] = a
            t = t.replace(a, tok)
    out = []
    for sg in V["prod_segs"](t):
        for tok, a in keep.items():
            sg = sg.replace(tok, a)
        for p in rv["prefixes"]:
            if sg.startswith(p) and len(sg) > len(p):
                sg = sg[len(p):].strip()
        if sg in rv["qualifiers"] or not sg:
            continue
        out.append(sg)
    return out


def _forms(seg: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(strong forms, weak forms) of one gold fragment."""
    V = _vocab()
    rv = V["rv"]
    prod = set(V["prod_forms"](seg))
    extra = set((rv["synonyms"].get(seg) or {}).get("forms") or ())
    weak = set((rv["weak_forms"].get(seg) or {}).get("forms") or ())
    if seg in rv["generic_segments"]:
        weak |= (prod - extra)
    strong = (prod | extra) - weak
    return tuple(sorted(strong)), tuple(sorted(weak))


def _shadows_of(form: str) -> tuple[str, ...]:
    return _vocab()["shadows"].get(_norm(form), ())


def _flag(p: _Text) -> str | None:
    V = _vocab()
    if any(p.hit(m) for m in V["neg"]):
        return "negated"
    if any(p.hit(m) for m in V["cond"]):
        return "conditional"
    # A5 post-registration fix: a proposal citing a result in hand is not a new order.
    if any(p.hit(m) for m in V["past"]):
        return "past_result"
    return None


def _hint_tokens(forms) -> set[str]:
    """Stems a resembling proposal would share with the item: ASCII tokens (>= 2 chars, not
    purely digits, not generic) and CJK bigrams (unigram for a one-character stem) after the
    generic words are cut out."""
    rv = _vocab()["rv"]
    stop_en = set(rv["hint_stop_en"])
    stop_zh = sorted(rv["hint_stop_zh"], key=len, reverse=True)
    out: set[str] = set()
    for f in forms:
        for chunk in _norm(f).split(" "):
            for run in re.findall(r"[a-z0-9]+|[^a-z0-9]+", chunk):
                if run.isascii():
                    if len(run) >= 2 and not run.isdigit() and run not in stop_en:
                        out.add(run)
                    continue
                for w in stop_zh:
                    run = run.replace(w, " ")
                for piece in run.split():
                    piece = "".join(ch for ch in piece if not ch.isascii())
                    if len(piece) == 1:
                        out.add(piece)
                    else:
                        out.update(piece[k:k + 2] for k in range(len(piece) - 1))
    return out


def match_item(item: str, proposals: list[str]) -> dict | None:
    """Three-valued verdict for one gold test item against a list of proposal texts.

    Returns None when the item has no matchable fragment (unscoreable). Otherwise a dict:
    ``verdict`` (yes/no/undecided), ``rule`` (why), ``evidence`` [(proposal index, form)],
    and ``lean`` -- the code-only fallback for an undecided item (True when a demoted form or an
    ambiguous panel hit a proposal that is not negated).
    """
    V = _vocab()
    rv = V["rv"]
    segs = gold_segments(item)
    if not segs:
        return None
    texts = [_Text(p) for p in proposals if str(p or "").strip()]
    flags = [_flag(t) for t in texts]
    certain, soft = [], []

    def _note(k, form, kind):
        if flags[k] == "negated":
            soft.append((k, form, "negated"))
        elif flags[k] == "conditional":
            soft.append((k, form, "conditional"))
        elif flags[k] == "past_result":
            soft.append((k, form, "past_result"))
        elif kind == "strong":
            certain.append((k, form, "form"))
        elif kind == "panel":
            certain.append((k, form, "panel"))
        else:
            soft.append((k, form, kind))

    all_forms: list[str] = []
    for sg in segs:
        strong, weak = _forms(sg)
        all_forms += list(strong) + list(weak)
        for k, t in enumerate(texts):
            for f in strong:
                if t.hit(f, _shadows_of(f)):
                    _note(k, f, "strong")
            for f in weak:
                if t.hit(f, _shadows_of(f)):
                    _note(k, f, "weak")
        for pname, pn in rv["panels"].items():
            if sg not in pn["members"]:
                continue
            all_forms += list(pn["aliases"])
            for k, t in enumerate(texts):
                for a in pn["aliases"]:
                    if t.hit(a, _shadows_of(a)):
                        _note(k, f"panel:{pname}:{a}", "panel" if pn["definitive"] else "panel_ambiguous")
        mp = rv["member_panels"].get(sg)
        if mp:
            named = {}
            for m, fs in mp["members"].items():
                all_forms += list(fs)
                for k, t in enumerate(texts):
                    if flags[k] != "negated" and any(t.hit(f) for f in fs):
                        named.setdefault(m, k)
            if len(named) >= mp["min"]:
                certain.append((min(named.values()), f"members:{sg}:{'+'.join(sorted(named))}", "members"))
            elif named:
                soft.append((min(named.values()), f"members:{sg}:{'+'.join(sorted(named))}", "members_partial"))

    if certain:
        return {"verdict": YES, "rule": certain[0][2], "evidence": [(k, f) for k, f, _ in certain], "lean": True,
                "segments": segs}
    if soft:
        lean = any(kind != "negated" for _, _, kind in soft)
        return {"verdict": UNDECIDED, "rule": soft[0][2], "evidence": [(k, f) for k, f, _ in soft], "lean": lean,
                "segments": segs}
    # A5 post-registration fix: a generic imaging proposal is undecided for an imaging gold item.
    if any(any(_spans(_norm(mo), _norm(f).replace(" ", "")) or _spans(_norm(mo), _norm(f))
               for mo in rv["imaging_modalities"]) for f in all_forms):
        for k, t in enumerate(texts):
            if flags[k] != "negated" and any(t.hit(g) for g in rv["imaging_generic"]):
                return {"verdict": UNDECIDED, "rule": "vague_imaging", "evidence": [(k, "imaging")],
                        "lean": False, "segments": segs}
    hints = _hint_tokens(all_forms)
    for k, t in enumerate(texts):
        for h in hints:
            if (h in t.v2) if not h.isascii() else bool(_spans(h, t.v1)):
                return {"verdict": UNDECIDED, "rule": "resembles", "evidence": [(k, h)], "lean": False,
                        "segments": segs}
    return {"verdict": NO, "rule": "no_resemblance" if texts else "no_proposals", "evidence": [], "lean": False,
            "segments": segs}


def match_required_tests(items: list[str], proposals: list[str]) -> list[dict | None]:
    """`match_item` for each gold item; one proposal may claim any number of items."""
    return [match_item(t, proposals) for t in items]
