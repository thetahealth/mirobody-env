"""tracks.py -- process-track judges that read across steps rather than only
the final answer (the kernel's `verifier.py` scores the final SolverOutput and
`score_track_E`).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import re


def visible_observations(case: dict, day: int) -> tuple:
    """Fingerprint of what the solver can see on `day`: visible evidence ids plus
    the visible point count per signal (new series points are new information).
    """
    ev = frozenset(e["evidence_id"] for e in (case.get("evidence_ledger") or [])
                   if int(e.get("source_timestamp", 10 ** 9)) <= day)
    pts = tuple(sorted((s, sum(1 for p in v if int(p["ts"]) <= day))
                       for s, v in (case.get("longitudinal_data") or {}).items()))
    return (ev, pts)


def oscillation_penalty(trajectory: list[dict], case: dict) -> dict:
    """A belief change (`repair == "revised"`) between adjacent rounds
    with no new visible observations counts as one oscillation. Always 0 on
    `tracking_review`, where every round adds a series point.
    """
    osc: list[int] = []
    for a, b in zip(trajectory or [], (trajectory or [])[1:]):
        # `repair` here is the kernel's "belief changed" diff label.
        if b.get("repair") != "revised":
            continue
        if visible_observations(case, int(b["day"])) == visible_observations(case, int(a["day"])):
            osc.append(int(b["round"]))
    pairs = max(1, len(trajectory or []) - 1)
    return {"c4_oscillations": len(osc), "c4_rounds": osc,
            "c4_penalty": round(min(1.0, 0.25 * len(osc) / pairs), 3)}


_RISK_ORDER = {"low": 0, "indeterminate": 1, "elevated": 2, "high": 3}


def flipback_penalty(trajectory: list[dict]) -> dict:
    """Belief swinging back (A -> B -> A), plus `risk_downgrades`,
    the number of backward risk steps.
    """
    bel = [(x.get("risk_cat"), x.get("top_driver"), x.get("action")) for x in (trajectory or [])]
    flip = sum(1 for i in range(len(bel) - 2) if bel[i] != bel[i + 1] and bel[i] == bel[i + 2])
    down = 0
    for a, b in zip(bel, bel[1:]):
        if _RISK_ORDER.get(b[0], 1) < _RISK_ORDER.get(a[0], 1):
            down += 1
    pairs = max(1, len(bel) - 2)
    return {"flipbacks": flip, "risk_downgrades": down,
            "flipback_penalty": round(min(1.0, 0.25 * flip / pairs), 3)}


def slice_convergence(slice_rows: list[dict], gold_driver: str | None,
                      ddx: dict | None = None) -> dict:
    """Multi-slice consult: the same world cut into N independent consults.

    Returns `answered_slices`, `converged_at` (first slice whose top pick hits
    gold, 1-based), `mentioned_at` (first slice naming gold at any rank) and
    `late_convergence` (only the last slice hits).
    """
    # ddx items match diagnosis aliases; other items match gold_driver.
    names = ([str(ddx.get("diagnosis") or "")] + [str(a) for a in (ddx.get("aliases") or [])]
             if ddx else [])

    def _hit(r: dict, top1_only: bool = True) -> bool:
        """Convergence = the top pick hits gold; a hedge naming it lower down
        counts only toward `mentioned_at`."""
        from .events import alias_hit
        if names:
            cand = r.get("differential") or []
            if cand:
                pool = [str(cand[0])] if top1_only else [str(x) for x in cand]
                return any(alias_hit(x, names) for x in pool)
            return bool(alias_hit(str(r.get("answer_text") or ""), names))
        return bool(gold_driver and gold_driver in (r.get("all_drivers") or []))

    hits = [bool(r.get("all_drivers")) or bool(r.get("answer_text")) for r in slice_rows]
    conv = next((i for i, r in enumerate(slice_rows, 1) if _hit(r)), None)
    ment = next((i for i, r in enumerate(slice_rows, 1) if _hit(r, top1_only=False)), None)
    n = len(slice_rows)
    return {"n_slices": n, "answered_slices": sum(hits), "converged_at": conv,
            "mentioned_at": ment,
            "late_convergence": bool(conv is not None and n > 1 and conv == n),
            "never_converged": conv is None}


def _answer_text(out) -> str:
    """Free text that can carry a diagnosis name (`drivers[].driver` is limited
    to the driver vocabulary)."""
    act = getattr(out, "action", None) or {}
    parts = [str(act.get("specific_action") or "")]
    parts += [str(x) for x in (act.get("what_not_to_do") or [])]
    f = getattr(out, "forecast", None) or {}
    parts.append(str(f.get("target_event") or ""))
    parts += [str(d.get("driver") or "") for d in (getattr(out, "drivers", None) or [])]
    return " ".join(parts)


def _differential(out) -> list[dict]:
    """The ddx prompt's candidate-diagnosis list; empty if absent."""
    raw = getattr(out, "_raw", None) or {}
    d = raw.get("differential")
    return [x for x in d if isinstance(x, dict)] if isinstance(d, list) else []


#: Sort key for a candidate with no usable rank.
RANK_ABSENT = 99


def rank_key(x: dict) -> int:
    """Sort key for a candidate diagnosis; absent, null or non-numeric ranks give
    `RANK_ABSENT` (a model may write `"rank": null` for a ruled-out candidate,
    which still takes part in judging). The single rank parser for this module
    and `judges`.
    """
    v = x.get("rank")
    try:
        return int(v)
    except (TypeError, ValueError):
        return RANK_ABSENT


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


def _finding_aliases() -> dict[str, tuple[str, ...]]:
    """`registry/threads.yaml: finding_aliases` -> {word (lower case): diagnosis forms}."""
    if not _FINDING_CACHE:
        from .registry import _load_yaml
        raw = _load_yaml("threads.yaml").get("finding_aliases") or {}
        out: dict[str, tuple[str, ...]] = {}
        for w, e in raw.items():
            if not isinstance(e, dict) or not str(e.get("why") or "").strip():
                raise ValueError(f"threads.yaml finding_aliases {w!r}: needs a mapping with `why`")
            bad = set(e) - {"why", "diagnosis_forms"}
            if bad:
                raise ValueError(f"threads.yaml finding_aliases {w!r}: unregistered fields {sorted(bad)}")
            forms = tuple(str(f) for f in (e.get("diagnosis_forms") or ()))
            if any(str(w).lower() not in f.lower() for f in forms):
                raise ValueError(f"threads.yaml finding_aliases {w!r}: every diagnosis form must contain the word")
            out[str(w).lower()] = forms
        _FINDING_CACHE.append(out)
    return _FINDING_CACHE[0]


_FINDING_CACHE: list[dict] = []


def thread_aliases(aliases) -> list[str]:
    """The aliases of one comorbidity thread that can name it.

    A finding-level word (a sign, a lab abnormality, an analyte) is dropped, or
    replaced by the diagnosis-level phrases registered for it: "高钙血症" is what
    primary hyperparathyroidism and its registered rivals all produce, so it does not
    name the parathyroid thread. The thread declarations themselves are gold and stay
    as written; the filtering happens here, on the judging side, and so does the
    addition of registered synonyms (`judging_names`).
    """
    tab = _finding_aliases()
    out: list[str] = []
    for a in aliases or ():
        forms = tab.get(str(a).lower())
        for x in ((a,) if forms is None else forms):
            if x not in out:
                out.append(x)
    return judging_names(out)


def _judging_synonyms() -> dict[str, tuple[str, ...]]:
    """`registry/threads.yaml: judging_synonyms` -> {anchor alias (lower case): extra names}."""
    if not _SYNONYM_CACHE:
        from .registry import _load_yaml
        raw = _load_yaml("threads.yaml").get("judging_synonyms") or {}
        out: dict[str, tuple[str, ...]] = {}
        for a, e in raw.items():
            if not isinstance(e, dict) or not str(e.get("why") or "").strip():
                raise ValueError(f"threads.yaml judging_synonyms {a!r}: needs a mapping with `why`")
            bad = set(e) - {"why", "names"}
            if bad:
                raise ValueError(f"threads.yaml judging_synonyms {a!r}: unregistered fields {sorted(bad)}")
            names = tuple(str(n) for n in (e.get("names") or ()))
            if not names or any(not n.strip() or n.lower() == str(a).lower() for n in names):
                raise ValueError(f"threads.yaml judging_synonyms {a!r}: `names` must be non-empty "
                                 f"and differ from the anchor")
            out[str(a).lower()] = names
        _SYNONYM_CACHE.append(out)
    return _SYNONYM_CACHE[0]


_SYNONYM_CACHE: list[dict] = []


def judging_names(aliases) -> list[str]:
    """`aliases` plus the judging-side synonyms registered for any of them.

    The gold names a disease in one wording (the kernel aliases, which are also the leak-scan
    words and stay as they are); a model may use another ("肾上腺皮质功能不全" for
    "肾上腺皮质功能减退"). Rival wordings a synonym would otherwise catch are declared in
    `vocab_alias_excludes.yaml`.
    """
    tab = _judging_synonyms()
    out = [a for a in (aliases or ())]
    seen = {str(a).lower() for a in out}
    for a in list(out):
        for n in tab.get(str(a).lower(), ()):
            if n.lower() not in seen:
                out.append(n)
                seen.add(n.lower())
    return out


def _alias_spans(text: str, names):
    """Occurrences `(alias, start, length)` of gold aliases, minus exclusion
    words (`overlay.alias_excluded_at`, e.g. polycystic kidney disease for
    PCOS). Negated occurrences are kept.
    """
    from .events import alias_hit
    from .overlay import alias_excluded_at
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


def dx_rank_of(out, names) -> dict:
    """Hit judging for the gold diagnosis, shared by `judges` and this module.

    With a `differential`: candidates sorted by rank; `dx_rank` is the position
    (1-based) of the first asserted, not ruled-out match, `dx_mentioned_rank`
    the first match of any kind, `dx_ruled_out_gold` whether gold was excluded.
    Without one (text fallback), rank fields are None.
    """
    ranked = sorted(_differential(out), key=rank_key)
    if ranked:
        hit = ment = None
        ruled = False
        for i, x in enumerate(ranked, 1):
            name = str(x.get("diagnosis") or "")
            if not alias_hit_asserted(name, names):
                # exclusion-filtered, negation kept
                if alias_mentioned(name, names) and ment is None:
                    ment = i
                continue
            if ment is None:
                ment = i
            if str(x.get("ruled_out_by") or "").strip():
                ruled = True
                continue
            hit = i
            break
        return {"dx_source": "structured", "dx_rank": hit, "dx_hit": hit is not None,
                "dx_hit_top1": hit == 1, "dx_mentioned": ment is not None,
                "dx_mentioned_rank": ment, "dx_ruled_out_gold": ruled,
                "dx_n_candidates": len(ranked)}
    said = alias_hit_asserted(_answer_text(out), names)
    return {"dx_source": "text_fallback", "dx_rank": None, "dx_hit": said,
            "dx_hit_top1": None, "dx_mentioned": said,
            "dx_mentioned_rank": None, "dx_ruled_out_gold": False, "dx_n_candidates": 0}


def asserted_diagnoses(out) -> list[str]:
    """Candidate diagnoses that are not ruled out."""
    return [str(x.get("diagnosis") or "") for x in _differential(out)
            if not str(x.get("ruled_out_by") or "").strip()]


def ddx_hit(out, ddx: dict | None) -> dict:
    """Diagnosis judging for the reporting side (production scoring uses
    `judges.run_judges`): diagnosis hit via `dx_rank_of`, plus coverage of the
    gold tests, specialty and, for comorbidity items, alias coverage.
    `dx_rank` is the position in the sorted candidate list, not the model's
    self-reported rank.
    """
    if not ddx:
        return {}
    from .events import alias_hit
    # `independent` items have no single diagnosis, and the insufficient tier
    # carries no signal; both are N/A for dx (scored by join_hit / abst_*).
    if ddx.get("join_gold") == "independent":
        return {"dx_applicable": False, "join_gold": ddx.get("join_gold")}
    if ddx.get("insufficient"):
        return {"dx_applicable": False, "dx_reason": "insufficient_tier"}
    names = [str(ddx.get("diagnosis") or "")] + [str(a) for a in (ddx.get("aliases") or [])]
    diff = _differential(out)
    raw = getattr(out, "_raw", None) or {}

    # Comorbidity gold is a set: scored by alias coverage across its lines.
    compound = ddx.get("join_gold") == "comorbidity"

    core = dx_rank_of(out, names)
    hit_at, src = core["dx_rank"], core["dx_source"]
    if src == "structured":
        ranked = sorted(diff, key=rank_key)
        all_txt = " ".join(asserted_diagnoses(out))
        blob = " ".join([str(t) for t in (raw.get("tests_to_order") or [])]
                        + [str(t) for t in (raw.get("referral_specialty") or [])])
        n_ev = sum(1 for x in ranked if x.get("supporting_evidence"))
        n_ruled = sum(1 for x in ranked if x.get("ruled_out_by"))
    else:                                  # text fallback
        blob = all_txt = _answer_text(out)
        n_ev = n_ruled = 0
        ranked = []

    tests = [t for t in (ddx.get("tests") or [])
             if any(seg and seg in blob for seg in str(t).replace("/", "·").split("·"))]
    spec = [x for x in (ddx.get("specialty") or [])
            if any(seg and seg in blob for seg in str(x).split("/"))]
    cov = None
    if compound:
        als = [a for a in (ddx.get("aliases") or []) if str(a).strip()]
        cov = round(len(alias_hit(all_txt, als)) / len(als), 3) if als else None
    return {"dx_applicable": True,
            "dx_hit": hit_at is not None, "dx_hit_top1": hit_at == 1, "dx_rank": hit_at,
            "dx_compound": compound, "dx_alias_coverage": cov,
            "dx_source": src, "dx_n_candidates": len(ranked),
            "dx_n_with_evidence": n_ev, "dx_n_ruled_out": n_ruled,
            "dx_tests_named": len(tests), "dx_tests_total": len(ddx.get("tests") or []),
            "dx_specialty_named": bool(spec), "dx_urgency_gold": ddx.get("urgency"),
            "join_gold": ddx.get("join_gold")}


def join_gold_hit(out, ddx: dict | None) -> dict:
    """Joint-attribution judge: is `join_type` (unified / comorbidity /
    independent) equal to the gold? Exact match, no partial credit; decoupled
    from `dx_hit`. The text fallback picks the first matching category, so it
    does not support capability conclusions. The pool is dominated by
    `unified`, so read `join_hit` against the class balance.
    """
    if not ddx or not ddx.get("join_gold"):
        return {}
    gold = ddx["join_gold"]
    said_field = str(((getattr(out, "_raw", None) or {}).get("join_type") or "")).strip().lower()
    if said_field in ("unified", "comorbidity", "independent"):
        return {"join_said": said_field, "join_hit": said_field == gold,
                "join_source": "structured"}
    blob = _answer_text(out)
    cues = {
        "unified": ("同一", "统一", "单一病因", "一元", "unified", "same underlying"),
        "comorbidity": ("共病", "合并", "叠加", "comorbid"),
        "independent": ("各自独立", "互不相关", "independent", "unrelated"),
    }
    said = [k for k, ws in cues.items() if any(w in blob or w in blob.lower() for w in ws)]
    return {"join_said": said[0] if said else None,
            "join_hit": bool(said and said[0] == gold), "join_source": "text_fallback"}


def alternative_a1(out) -> dict:
    """A1: at least two candidate hypotheses with supporting evidence (one ->
    0.5)."""
    ds = getattr(out, "drivers", None) or []
    with_ev = [d for d in ds if d.get("evidence_for")]
    # ddx items list hypotheses in `differential`
    diff = _differential(out)
    diff_ev = [x for x in diff if x.get("supporting_evidence")]
    n, n_ev = max(len(ds), len(diff)), max(len(with_ev), len(diff_ev))
    return {"n_drivers": n, "n_drivers_with_evidence": n_ev,
            "a1": round(min(1.0, n_ev / 2.0), 3)}


# ---------------------------------------------------------------- near-miss rivals
def rival_status(out, rivals, visible_ids=None) -> dict:
    """Whether the model dealt with a registered near-miss diagnosis:
    `rival_considered` (listed), `rival_ruled_out` (listed with `ruled_out_by`),
    `rival_top1_live` (the near-miss is the top non-excluded candidate). Not
    mentioning a rival is not excluding it. Without a `differential`, all are
    None except `rival_n_declared`.
    """
    keys = ("rival_considered", "rival_ruled_out", "rival_top1_live",
            "rival_n_declared", "rival_names_hit")
    if not rivals:
        return dict.fromkeys(keys)
    ranked = sorted(_differential(out), key=rank_key)
    if not ranked:
        return {**dict.fromkeys(keys), "rival_n_declared": len(rivals)}
    considered = ruled = top1_live = False
    hit_names: list[str] = []
    live_seen = 0
    for x in ranked:
        name = str(x.get("diagnosis") or "")
        is_ruled = bool(str(x.get("ruled_out_by") or "").strip())
        if not is_ruled:
            live_seen += 1
        for r in rivals:
            names = tuple(r.get("aliases") or ()) + (r.get("name", ""),)
            if not alias_hit_asserted(name, names):
                continue
            considered = True
            if r.get("name") not in hit_names:
                hit_names.append(r.get("name"))
            if is_ruled:
                ruled = True
            elif live_seen == 1:                      # top non-excluded candidate
                top1_live = True
    # Near-miss recall, two conventions kept separate: `rival_recall`
    # (hits / registered rivals) and `rival_recall_capped2` (min(hits, 2) / 2).
    _n_hit = len(hit_names)
    out_d = {"rival_considered": considered, "rival_ruled_out": ruled,
             "rival_top1_live": top1_live, "rival_n_declared": len(rivals),
             "rival_names_hit": hit_names or None,
             "rival_recall": round(_n_hit / len(rivals), 3) if rivals else None,
             "rival_recall_capped2": round(min(_n_hit, 2) / 2.0, 3) if rivals else None}
    out_d.update(_pseudo_exclusion(ranked, visible_ids))
    return out_d


def _pseudo_exclusion(ranked, visible_ids) -> dict:
    """Exclusions without discriminating evidence: `excl_no_evidence` (cites no
    visible evidence id) and `excl_evidence_reuse` (one evidence id used to
    exclude several hypotheses). Evidence ids are resolved by
    `process._resolve_ref`, never by substring, so a forged id that extends a
    real one does not count as grounded.
    """
    import collections as _c
    vis = {str(v) for v in (visible_ids or ()) if v}
    if not ranked:
        return {}
    use: dict[str, set] = _c.defaultdict(set)
    n_excl = n_blind = 0
    for x in ranked:
        txt = x.get("ruled_out_by")
        if txt in (None, "", [], {}):
            continue
        blob = txt if isinstance(txt, str) else str(txt)
        n_excl += 1
        from .process import _ev_ids, _resolve_ref
        cited = {r for r in (_resolve_ref(x, vis) for x in _ev_ids(blob)) if r}
        if not cited:
            n_blind += 1
        for i in cited:
            use[i].add(str(x.get("diagnosis") or "")[:24])
    reuse = sum(1 for v in use.values() if len(v) >= 2)
    return {"excl_n": n_excl or None,
            "excl_no_evidence": n_blind or None,
            "excl_evidence_reuse": reuse or None,
            "excl_max_per_evidence": (max((len(v) for v in use.values()), default=0) or None),
            "excl_grounded_rate": (round((n_excl - n_blind) / n_excl, 3) if n_excl else None)}


# ---------------------------------------------------------------- Tools track (T1-T4)
def tool_track(trace, vp=None, key_signals=None) -> dict:
    """Readings for on-demand querying:

      - T1 grounding: rate of queried targets that this patient actually has
        (test orders are excluded from the denominator);
      - T2 redundancy: repeat queries by literal target string. The gated menu
        withdraws already revealed items, so this is near zero by design;
        "queried but never cited" is not measured;
      - T3 budget: spent vs. budget, clipped ratios for scoring plus the raw
        ratio and `tool_over_budget`;
      - T4: key signals not queried before committing (None when the item
        declares no key signals).
    """
    if trace is None:
        return {}
    calls = list(getattr(trace, "calls", []) or [])
    n = len(calls)
    targets = [c.target for c in calls if c.target]
    uniq = set(targets)
    dup = len(targets) - len(uniq)
    _gcalls = [c for c in calls if getattr(c, "grounded", None) is not None]
    grounded = sum(1 for c in _gcalls if c.grounded)
    spent = float(getattr(trace, "spent", 0.0) or 0.0)
    budget = float(getattr(trace, "budget", 0.0) or 0.0)
    out = {
        "tool_n_calls": n,
        "tool_n_unique_targets": len(uniq),
        # T1
        "tool_target_grounded_rate": (round(grounded / len(_gcalls), 3) if _gcalls else None),
        "tool_n_signal_calls": len(_gcalls),
        "tool_n_test_calls": n - len(_gcalls),
        "tool_ungrounded": n - grounded,
        # T2
        "tool_dup_rate": round(dup / n, 3) if n else None,
        # T3
        "tool_spent": spent, "tool_budget": budget,
        "tool_within_budget": (spent <= budget) if budget else None,
        # Clipped to [0,1]: this feeds the score mean (`registry/scoring.yaml`).
        "tool_budget_used": (min(1.0, max(0.0, round(spent / budget, 3)))
                             if budget else None),
        #: Unclipped ratio; does not feed the mean.
        "tool_budget_used_raw": round(spent / budget, 3) if budget else None,
        # Thrift = 1 - budget used, clipped, so it points the same direction as
        # the grounding rate it is paired with.
        "tool_budget_thrift": (min(1.0, max(0.0, round(1.0 - spent / budget, 3)))
                               if budget else None),
        #: None when there is no budget.
        "tool_over_budget": (bool(spent > budget) if budget else None),
        #: Queries rejected for exceeding budget.
        "tool_truncated": getattr(trace, "truncated", 0),
        "tool_rounds": getattr(trace, "rounds_used", None),
        "tool_committed": bool(getattr(trace, "committed", False)),
        "tool_protocol_errors": getattr(trace, "protocol_errors", 0),
        # Self-reported `kind` disagreeing with the menu (billing uses the menu).
        "tool_kind_misreports": getattr(trace, "kind_misreports", 0),
    }
    # T4. Typed targets are mapped to finding ids through the production
    # resolver `gated._resolve_target`.
    ks = set(key_signals or ())
    if ks:
        asked = set(uniq)
        try:
            from .gated import _resolve_target as _rt
            from .registry import load_findings as _lf
            _fdg = _lf()
            for _t in uniq:
                _fid = _rt(str(_t), _fdg)
                if _fid:
                    asked.add(_fid)
        except Exception:      # resolver unavailable: compare raw strings
            pass
        missed = sorted(ks - asked)
        out["tool_key_signals"] = sorted(ks)
        out["tool_key_missed"] = missed or None
        out["tool_key_covered"] = round(len(ks & asked) / len(ks), 3)
        out["tool_concluded_blind"] = bool(missed) and bool(getattr(trace, "committed", False))
    else:
        out["tool_key_covered"] = None          # not applicable
        out["tool_concluded_blind"] = None
    return out


def key_signals_for(vp) -> tuple[str, ...]:
    """Key query targets for T4: the gold evidence stream for early-warning
    items (`registry.GOLD_EVIDENCE`), or each rival's
    `discriminator_finding.finding` from `registry/rivals.yaml` for diagnosis
    items.
    """
    # imported here to avoid a build <-> tracks import cycle
    from .registry import GOLD_EVIDENCE
    drv = (getattr(vp, "gold_drivers", None) or [None])[0]
    spec = GOLD_EVIDENCE.get(str(drv or ""))
    if spec:
        return (spec["signal"],)
    from .judges import _ddx as _ddx_of
    from .overlay import rivals_for
    ddx = _ddx_of(vp) or {}
    if not ddx.get("spec_id"):
        return ()
    out: list[str] = []
    for _r in (rivals_for(ddx.get("spec_id"), ddx) or []):
        _df = (_r or {}).get("discriminator_finding") or {}
        _f = str(_df.get("finding") or "").strip()
        if _f and _f not in out:
            out.append(_f)
    return tuple(out)
