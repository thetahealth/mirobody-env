"""Differential-diagnosis family: three gold kinds (unified/comorbidity/independent) ·
near-miss exclusion · discriminative tooling · join self-consistency · workup (tests/
disposition).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("haenv.judges")

from ._helpers import _ddx, _gold, _names, _rank_of, log  # noqa: F401
from ..tracks import D9_PROVISIONAL  # noqa: F401  暂定，待临床复核 (docs/spec, design §5.0 D9); defined in tracks.py, re-exported here

#: Hedged certainty labels ("possible"/"probable"/"suspected"); they count as listed under
#: `D9_PROVISIONAL["possible_counts_as_listed"]` (default), and are dropped from the differential otherwise.
_HEDGED_CERTAINTY = ("possible", "probable", "suspected", "likely")

#: `dx_listed` counts a gold line only when it is among the first `n + DX_LIST_CAP_MARGIN`
#: live (not ruled-out) candidates, `n` being the number of gold threads (1 on a unified case).
#: Same margin as the test-order cap (`TESTS_CAP_MARGIN`). Without it, listing every diagnosis in
#: the catalogue would score 1 on every case; a differential of a handful of candidates is not
#: affected. Ruled-out candidates do not use up the cap.
#: A6 (2026-10-01): margin 3 -> 1. A question-blind fixed list reaches 0.506 under n+3, 0.315 under
#: n+1 and 0.213 under n+0 (`blind_ceilings.dx_listed_ceiling`); n+1 is the most conservative cap
#: that keeps one slot for a hedge. The n+0 reading is recorded as the profile atom `dx_listed_n0`.
DX_LIST_CAP_MARGIN = 1


def judge_dx_unified(out, vp, ctx) -> dict:
    """Unified questions: gold is a single diagnosis; judge whether and at what rank it was named.
    Excluded or negated candidates count only toward `dx_mentioned`, never `dx_hit`.
    `dx_listed` is the scored atom (see `_listed`).
    """
    res = {"dx_kind": "unified", **_rank_of(out, _names(vp))}
    res.update(_listed(out, [_names(vp)]))
    return res


def _thread_sets(vp) -> tuple[list[list[str]], str]:
    """The alias set of every gold thread, and the judging basis. Declared threads go through
    `tracks.thread_aliases` (finding-level words do not name a thread). Without declared
    `threads` it falls back to a flat alias proxy, labelled as such because it can produce false
    negatives.
    """
    from ..tracks import thread_aliases
    th = [t for t in (_gold(vp, "threads") or []) if t.get("aliases")]
    if th:
        return [thread_aliases(t["aliases"]) for t in th], "declared"
    return [[a] for a in (_gold(vp, "aliases") or []) if str(a).strip()], "flat_alias_proxy"


def _threads_hit(blob: str, vp) -> tuple[int, int, str]:
    """(threads hit, total threads, judging basis)."""
    from ..tracks import alias_hit_asserted      # judging-side hit: passes negation + exclusion set
    sets, basis = _thread_sets(vp)
    return sum(1 for s in sets if alias_hit_asserted(blob, s)), len(sets), basis


def _listed(out, alias_sets) -> dict:
    """Atom "on the differential and not ruled out" (`dx_listed`).

    For each gold line (the diagnosis on a unified case, each thread on a comorbidity case),
    the 1-based position of its first match among the live candidates in rank order. The atom
    is the share of lines found within the cap. Only the structured differential counts: a
    diagnosis named elsewhere in the answer is not on the differential.

    D9 (暂定，待临床复核): a "possible"/"probable"/"suspected" entry is listed; a `rule_out` (待排)
    entry is listed even with `ruled_out_by` filled in (`tracks.entry_excluded`); a comorbidity
    case needs per-thread coverage, not both components asserted together. With
    `possible_counts_as_listed` off, hedged entries drop out and a multi-thread case scores only
    when every thread is found (the strict reading).
    """
    from ..quantities import dx_listed_n0_of, dx_listed_of
    from ..tracks import _certainty, _differential, alias_hit_asserted, entry_excluded, rank_key
    _hedged_ok = D9_PROVISIONAL["possible_counts_as_listed"]
    live = [str(x.get("diagnosis") or "") for x in sorted(_differential(out), key=rank_key)
            if not entry_excluded(x)
            and (_hedged_ok or _certainty(x) not in _HEDGED_CERTAINTY)]
    pos = [next((i for i, e in enumerate(live, 1) if alias_hit_asserted(e, s)), None)
           for s in alias_sets]
    if not _hedged_ok and len(pos) > 1 and any(p is None for p in pos):
        pos = [None] * len(pos)
    res = {"dx_listed_positions": pos, "dx_listed_cap": len(alias_sets) + DX_LIST_CAP_MARGIN,
           "dx_n_live": len(live)}
    res["dx_listed"] = dx_listed_of(res)
    res["dx_listed_n0"] = dx_listed_n0_of(res)
    return res


def judge_dx_comorbidity(out, vp, ctx) -> dict:
    """Comorbidity questions: gold is a set of threads; judge whether all were captured, counting only
    candidates not excluded by `ruled_out_by`. `dx_hit` here means "at least one thread"; the
    scored atom is `dx_listed` (thread coverage, see `_listed`).
    """
    from ..tracks import asserted_diagnoses, _answer_text
    blob = " ".join(asserted_diagnoses(out)) or _answer_text(out)
    n, tot, basis = _threads_hit(blob, vp)
    res = {"dx_kind": "comorbidity", "dx_thread_basis": basis,
           "dx_threads_matched": n, "dx_threads_total": tot,
           "dx_coverage": round(n / tot, 3) if tot else None,
           "dx_all_threads": bool(tot) and n >= tot,
           **{k: v for k, v in _rank_of(out, _names(vp)).items() if k != "dx_n_candidates"}}
    res.update(_listed(out, _thread_sets(vp)[0]))
    return res


def judge_dx_independent(out, vp, ctx) -> dict:
    """Independent questions: gold is "no unifying pathology", so this asks `over_unified` (from the
    structured `join_type` field) instead of `dx_hit`, which cannot hit here.
    """
    said = str(((getattr(out, "_raw", None) or {}).get("join_type") or "")).strip().lower()
    over = said in ("unified", "comorbidity")
    return {"dx_kind": "independent", "dx_applicable": False,
            "over_unified": over, "held_independent": said == "independent"}


# ================================================================ disposition axis (urgency / tests / specialty)
URGENCY_ACTION = {                      # gold urgency -> (acceptable action-class set, canonical class)
    "🔴": ({"A5"}, "A5"),               # immediate emergency care
    "🟠": ({"A4"}, "A4"),               # within days
    "🟡": ({"A3"}, "A3"),               # prompt formal evaluation
    "🟢": ({"A0", "A1", "A2"}, "A0"),   # can observe / self-manage
}
_ACTION_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5")


from ..registry import load_test_vocab as _load_test_vocab                  # noqa: E402
_TESTV = _load_test_vocab()


def _segs(s: str) -> list[str]:
    """Split a gold test item into matchable segments (`blood/24h urine … (metanephrines)` ->
    three segments), stripping leading conjunctions and generic qualifiers from the
    `vocab_tests.yaml` tables. A missing split lowers recall; a leftover qualifier inflates
    precision.
    """
    import re
    out = []
    for seg in re.split(r"[/·、,,()()+;;和与及]|\s或\s?|^或|或(?=[A-Za-z一-鿿])",
                        str(s or "")):
        seg = (seg or "").strip()
        for p in _TESTV["prefixes"]:                 # strip conjunctions/verbs stuck to the front
            if seg.startswith(p) and len(seg) > len(p):
                seg = seg[len(p):].strip()
        if seg.lower() in {x.lower() for x in _TESTV["stopwords"]}:
            continue                                 # generic qualifier: never participates in matching
        if len(seg) >= (2 if seg.isascii() else 2):
            out.append(seg)
    return out


def _seg_forms(seg: str) -> tuple[str, ...]:
    syn = _TESTV["synonyms"].get(seg, ())
    return tuple({seg.lower(), *(str(x).lower() for x in syn)})


def _gold_test_segs(t: str) -> list[str]:
    return _segs(t)


# Optionality markers in gold test items: such items leave the recall denominator but still count
# as on target for precision.
_OPTIONAL_MARKERS = ("必要时", "如有需要", "酌情", "如需", "视情况",
                     "optional", "if needed", "as needed", "when indicated")


def _gold_test_optional(t: str) -> bool:
    low = str(t).lower()
    return any(m in low for m in _OPTIONAL_MARKERS)


#: Margin for the order-count cap (cap = gold item count + margin); shared with `judge_discriminative_tool`.
TESTS_CAP_MARGIN = 3


def _executed_tests(raw, ctx) -> list[str]:
    """Tests already executed through the tool: the answer's `executed_investigations`, else the
    same key on `ctx`. Empty when the run has none (non-gated geometries).
    """
    src = raw.get("executed_investigations")
    if not src and isinstance(ctx, dict):
        src = ctx.get("executed_investigations")
    return [str(t) for t in (src or []) if str(t).strip()]


def judge_workup(out, vp, ctx) -> dict:
    """Disposition: what to test, which specialty to refer to, how urgent. Tests are judged in both
    directions.
    """
    raw = getattr(out, "_raw", None) or {}
    res: dict = {}

    # `_gold` returns the derived urgency (red flag => 🔴, no clinician action => 🟢), so a spec whose
    # authored urgency is stale is not graded against the wrong tier.
    gold_u = _gold(vp, "urgency")
    if gold_u in URGENCY_ACTION:
        accept, canon = URGENCY_ACTION[gold_u]
        said = str(((getattr(out, "action", None) or {}).get("selected_action_class") or "")).strip()
        gap = ((_ACTION_ORDER.index(said) - _ACTION_ORDER.index(canon))
               if said in _ACTION_ORDER else None)
        # `urgency_ok` is binary; the signed gap and within-one reading separate an off-by-one mapping from noise.
        res.update({"urgency_gold": gold_u, "urgency_said": said or None,
                    "urgency_ok": said in accept, "urgency_gap": gap,
                    "urgency_within1": (abs(gap) <= 1) if gap is not None else None})

    # The gold test list is looked up in the registry by `spec_id`, so a registry correction reaches
    # existing batches on recompute; the case copy is a fallback and a mismatch is logged.
    _stored = [t for t in (_gold(vp, "tests") or []) if str(t).strip()]
    gold_tests = _stored
    _sid = _gold(vp, "spec_id", default=None)
    if _sid:
        try:
            from ..overlay import condition_registry
            _reg = (condition_registry(include_draft=True).get(str(_sid)) or {}).get("tests")
            if _reg:
                _reg = [str(t) for t in _reg if str(t).strip()]
                if _reg != _stored:
                    log.info("[judges] %s tests copy disagrees with the registry, judged by the registry: "
                             "copy %s vs registry %s", _sid, _stored, _reg)
                gold_tests = _reg
        except Exception as e:                             # noqa: BLE001
            log.warning("[judges] %s tests registry lookup failed, falling back to the copy: %s", _sid, e)
    said_tests_all = [str(t) for t in (raw.get("tests_to_order") or []) if str(t).strip()]
    # Order-count cap: only the first N ordered tests are scored, N = scoreable gold items + margin.
    # It is judge-side only and never shown to the model, so it leaks nothing; without it, ordering
    # the whole catalog would win both recall and discrimination.
    _n_scoreable = len([t for t in gold_tests if _gold_test_segs(t)])
    _cap = _n_scoreable + TESTS_CAP_MARGIN
    said_tests = said_tests_all[:_cap]
    if gold_tests:
        # Gold items that are instructions rather than test names are unscoreable and leave the
        # denominator. Tiers: required / optional / unscoreable; only required enters recall.
        optional = [t for t in gold_tests if _gold_test_segs(t) and _gold_test_optional(t)]
        scoreable = [t for t in gold_tests
                     if _gold_test_segs(t) and not _gold_test_optional(t)]
        unscoreable = [t for t in gold_tests if not _gold_test_segs(t)]
        # Match item by item, one-to-one, longest submitted item first. Matching against a joined blob,
        # or letting one submitted item claim several gold items, inflates recall.
        _low = [t.lower() for t in said_tests]
        _forms = {t: [f for sg in (_gold_test_segs(t) or []) for f in _seg_forms(sg)]
                  for t in scoreable}
        _taken_gold, _taken_said = set(), set()
        for _si, _one in sorted(enumerate(_low), key=lambda kv: -len(kv[1])):
            _cand = [(len(f), t) for t in scoreable if t not in _taken_gold
                     for f in _forms[t] if f in _one]
            if not _cand:
                continue
            _taken_gold.add(max(_cand)[1])        # the gold item with the longest matching segment
            _taken_said.add(_si)
        named_ordered = [t for t in scoreable if t in _taken_gold]
        # D9.1 (暂定，待临床复核): a test already executed through the tool (`executed_investigations`,
        # result already in the case) covers a still-unmatched required item, same one-to-one rule.
        # It fills the cap slots the ordered list leaves free (ordered first, then executed, as the
        # semantic judge reads them), and touches recall only: precision keeps the ordered tests as
        # its denominator, so reading the chart is neither penalised nor rewarded twice.
        executed_all = _executed_tests(raw, ctx)
        executed = (executed_all[:max(0, _cap - len(said_tests))]
                    if D9_PROVISIONAL["executed_counts_as_covered"] else [])
        for _one in sorted((t.lower() for t in executed), key=lambda t: -len(t)):
            _cand = [(len(f), t) for t in scoreable if t not in _taken_gold
                     for f in _forms[t] if f in _one]
            if _cand:
                _taken_gold.add(max(_cand)[1])
        named = [t for t in scoreable if t in _taken_gold]
        via_executed = [t for t in named if t not in named_ordered]
        gold_forms = [f for t in scoreable + optional
                      for s in _gold_test_segs(t) for f in _seg_forms(s)]
        on_target = [t for t in said_tests if any(f in t.lower() for f in gold_forms)]
        res.update({"tests_total": len(scoreable), "tests_named": len(named),
                    "tests_optional": len(optional) or None,
                    "tests_recall": (round(len(named) / len(scoreable), 3)
                                     if scoreable else None),
                    "tests_recall_ordered_only": (round(len(named_ordered) / len(scoreable), 3)
                                                  if scoreable else None),
                    "tests_recall_via_executed": len(via_executed) if executed_all else None,
                    "tests_executed": len(executed_all) or None,
                    "tests_unscoreable": len(unscoreable) or None,
                    "tests_proposed": len(said_tests_all),
                    "tests_cap": _cap,
                    "tests_over_cap": (len(said_tests_all) - _cap) or None,
                    "tests_precision": (round(len(on_target) / len(said_tests), 3)
                                        if said_tests else None)})

    gold_spec = [x for x in (_gold(vp, "specialty") or []) if str(x).strip()]
    if gold_spec:
        sblob = " ".join(str(x) for x in (raw.get("referral_specialty") or []))
        from ..overlay import specialty_hit
        res["specialty_ok"] = specialty_hit(sblob, gold_spec)
    return res


# Connectives inside a compound diagnosis name: such a name already says "two diseases".
# No word boundaries around Chinese connectives (Chinese is not word-segmented).
_COMPOUND_MARKERS = ("+", "＋", "与", "合并", "伴", "共病", "叠加", "及", " and ", "&")

_COLLECTION_MARKERS = ("集合", "多个", "多项", "多发", "多种", "各自独立", "彼此独立",
                       "独立事件", "independent", "multiple", "collection", "彼此无关")


def join_self_contradiction(out, vp=None, ctx=None) -> dict:
    """Whether the model's own differential contradicts its own `join_type`; never reads the gold.

    Flagged when the unexcluded, non-compound top candidate cites >=2 pieces of evidence while
    `join_type` is comorbidity or independent. Superseded by `join_claims_unified`; kept for
    regression. Returns None when the solver has no differential.
    """
    from ..tracks import _differential
    from ..tracks import rank_key as _rank_key   # the single implementation of rank ordering, not reimplemented here
    raw = getattr(out, "_raw", None) or {}
    jt = str(raw.get("join_type") or "").strip().lower()
    diff = _differential(out)
    if not diff or jt not in ("unified", "comorbidity", "independent"):
        return {"join_self_contradiction": None}          # not applicable
    top = sorted(diff, key=_rank_key)[0]
    name = str(top.get("diagnosis") or "")
    ruled = bool(str(top.get("ruled_out_by") or "").strip())
    compound = any(k in name for k in _COMPOUND_MARKERS)
    n_ev = len(top.get("supporting_evidence") or [])
    claims_unified = (not ruled) and (not compound) and n_ev >= 2
    return {"join_self_contradiction": bool(claims_unified and jt != "unified"),
            "top1_claims_unified": claims_unified,
            "top1_n_evidence": n_ev, "top1_compound": compound}


def join_claims_unified(out, vp, ctx=None) -> dict:
    """Whether the top candidate alone explains all the true symptoms, i.e. implicitly claims
    `unified`.

    Reads the Q-side true-symptom ids (never `join_gold`); None when the manifest is absent.
    """
    from .. import wq
    from ..tracks import _differential
    from ..tracks import rank_key as _rank_key   # the single implementation of rank ordering, not reimplemented here
    man = wq.injected_manifest(getattr(vp, "case_id", ""))
    real = set(man.get("real_symptom_evidence_ids") or [])
    raw = getattr(out, "_raw", None) or {}
    jt = str(raw.get("join_type") or "").strip().lower()
    diff = _differential(out)
    if not real or not diff or jt not in ("unified", "comorbidity", "independent"):
        return {"join_contradiction_cover": None}
    top = sorted(diff, key=_rank_key)[0]
    covered = real & set(top.get("supporting_evidence") or [])
    covers_all = real.issubset(set(top.get("supporting_evidence") or []))
    frac = round(len(covered) / len(real), 3)
    # Read through `_gold` so the derived value is used, as `check_gold_matches_world` expects.
    gold_cls = str(_gold(vp, "join_gold", default="") or "")
    name = str(top.get("diagnosis") or "")
    multi = (any(k in name for k in _COMPOUND_MARKERS)
             or any(k in name.lower() for k in _COLLECTION_MARKERS))
    return {"join_top1_covers_all": covers_all,
            "join_top1_cover_frac": frac,
            "join_top1_multi_named": multi,
            "join_contradiction_cover": bool(covers_all and jt != "unified"),
            **join_scope_band(frac, len(real), gold_cls, multi_named=multi)}


def join_scope_band(frac: float | None, n_real: int, gold_class: str,
                    multi_named: bool = False) -> dict:
    """Does top1's true-symptom coverage fall inside the band its gold class allows?

    | gold class | allowed band |
    |---|---|
    | `unified` | `frac == 1` |
    | `comorbidity` | `1/n <= frac <= (n-1)/n` |
    | `independent` | `frac == 1/n` |

    Citing nothing fails every class and citing everything passes only unified. Reported
    alongside `join_hit`, never in its place.
    """
    if frac is None or not n_real or gold_class not in ("unified", "comorbidity", "independent"):
        return {"join_scope_ok": None}
    lo = 1.0 / n_real
    # `multi_named` is not part of the verdict: compound markers also occur inside single
    # diagnosis names.
    if gold_class == "unified":
        ok = frac >= 1.0
    elif gold_class == "independent":
        ok = abs(frac - lo) < 1e-6
    else:
        ok = (lo - 1e-6) <= frac <= (1.0 - lo + 1e-6)
    return {"join_scope_ok": bool(ok)}


def judge_discriminative_tool(out, vp, ctx=None) -> dict:
    """Among the proposed tests, is there one that separates gold from each near-miss rival?

    Rivals carry a structured `discriminator_finding`; matching uses `events.alias_hit` with
    `allow_suffix`. Returns `disc_n_rivals`, `disc_covered`/`disc_recall`, `disc_missed` and
    `disc_n_tests_proposed` (context for reading coverage). No structured rival => not applicable.
    """
    from ..events_text import alias_hit
    from ..overlay import rivals_for
    from ..registry import load_findings

    ddx = _ddx(vp)
    sid = (ddx or {}).get("spec_id")
    rivals = [r for r in (rivals_for(sid, ddx) or ()) if (r.get("discriminator_finding") or {})]
    if not rivals:
        return {"disc_n_rivals": 0, "disc_recall": None,
                "disc_note": "该例无结构化判别项 -> 不适用(不是 0)"}

    raw = getattr(out, "_raw", None) or {}
    tests_all = raw.get("tests_to_order") or []
    _gt = [t for t in (_gold(vp, "tests") or []) if str(t).strip()]
    _n_req = len([t for t in _gt if _gold_test_segs(t) and not _gold_test_optional(t)])
    tests = list(tests_all)[:_n_req + TESTS_CAP_MARGIN]
    text = "\n".join(str(t) for t in tests)
    if not text.strip():
        # An empty order list gets score 0, not not-applicable: the model named no discriminating test.
        return {"disc_n_rivals": len(rivals), "disc_covered": 0, "disc_recall": 0.0,
                "disc_missed": [str(r.get("name")) for r in rivals][:5],
                "disc_n_tests_proposed": 0}

    fx = load_findings()
    covered, missed = [], []
    _used: set[int] = set()      # indices of test items already claimed — each claims at most one
    for r in rivals:
        fid = (r.get("discriminator_finding") or {}).get("finding")
        f = fx.get(fid) or {}
        # Candidates are aliases, Chinese/English names and `acquired_by` (the study that yields the
        # measurement, e.g. PSG for AHI).
        cands = (list(f.get("aliases") or [])
                 + [x for x in (f.get("name_cn"), f.get("name_en")) if x]
                 + list(f.get("acquired_by") or []))
        # Each test item may claim at most one discriminating item, so one long string covers at most one.
        _hit = next((i for i, t in enumerate(tests)
                     if i not in _used and alias_hit(str(t), cands, allow_suffix=True)), None)
        if _hit is None:
            missed.append(str(r.get("name")))
        else:
            _used.add(_hit)                       # each test item may claim only **one** discriminating item
            covered.append(str(r.get("name")))
    return {"disc_n_rivals": len(rivals), "disc_covered": len(covered),
            "disc_recall": round(len(covered) / len(rivals), 3),
            "disc_missed": missed or None,
            "disc_n_tests_proposed": len(tests_all),
            "disc_cap": _n_req + TESTS_CAP_MARGIN,
            "disc_over_cap": (len(tests_all) - (_n_req + TESTS_CAP_MARGIN)) or None}


def judge_dx_rival(out, vp, ctx) -> dict:
    """Did the model deal with the near-miss rival (track A2)? Rivals come from
    `registry/rivals.yaml` and never enter W, so stored responses can be rescored.
    """
    from ..overlay import rivals_for
    from ..tracks import rival_status
    from .. import wq
    ddx = _ddx(vp)
    return rival_status(out, rivals_for(ddx.get("spec_id"), ddx), visible_ids=visible_ids_of(vp))


def visible_ids_of(vp, ctx: dict | None = None) -> set:
    """The full set of visible EV ids for this case, from the Q-side ledger's `id_map` (never
    guessed from the id format). `ctx["visible_ids"]` overrides it for fixtures.
    """
    from .. import wq                      # local import, consistent with the rest of this module (avoids a circular import)
    if ctx and ctx.get("visible_ids"):
        return set(ctx["visible_ids"])
    man = wq.injected_manifest(getattr(vp, "case_id", ""), required=False)
    return set((man.get("id_map") or {}).values()) or (
        set(man.get("real_symptom_evidence_ids") or [])
        | set(man.get("benign_evidence_ids") or []))


def judge_join_type(out, vp, ctx) -> dict:
    from ..tracks import join_gold_hit
    return join_gold_hit(out, _ddx(vp))
