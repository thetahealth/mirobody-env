"""Process family: slices, prohibitions, quantitative probes, abstention calibration,
commit timing, NoOp, premise challenge, premise repair, multi-round revision.

SYNTHETIC, evaluation only; not medical advice.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("haenv.judges")

from ._helpers import _gold, _names, gold_kind  # noqa: F401
from .differential import _threads_hit, judge_discriminative_tool, judge_workup  # noqa: F401

def _slice_hit_unified(row, vp) -> bool:
    from ..events import alias_hit
    cand = row.get("differential") or []
    pool = [str(cand[0])] if cand else [str(row.get("answer_text") or "")]
    return any(alias_hit(x, _names(vp)) for x in pool)


def _slice_threads(row, vp) -> tuple[int, int, str]:
    cand = row.get("differential") or []
    blob = " ".join(str(x) for x in cand) or str(row.get("answer_text") or "")
    return _threads_hit(blob, vp)


class _SliceOut:
    """Wrap one slice row into the shape `judge_workup` expects (only the three
    fields it reads)."""

    def __init__(self, row: dict):
        self._raw = {"tests_to_order": row.get("tests_to_order") or [],
                     "referral_specialty": row.get("referral_specialty") or []}
        self.action = {"selected_action_class": row.get("action")}


def judge_self_contradictory_exclusion(out, visible_ids=()) -> dict:
    """`self_contradictory_exclusion`: when a differential is ruled out, does the cited
    refuting evidence hold up?

    Reads the structured evidence ids in `differential[]`:

    * `sce_self_contradictory` -- the id in `ruled_out_by` also appears in the same entry's
      `supporting_evidence`;
    * `sce_unrevealed` -- the refuting id does not resolve to visible evidence (resolution as
      in `process.judge_evidence_revealed`).

    The contradiction rate's denominator counts only entries with supporting evidence;
    entries without any are counted separately in `sce_n_no_support`.
    """
    _d = (getattr(out, "_raw", None) or {}).get("differential")
    if not isinstance(_d, list) or not _d:
        return {}
    from ..process import _resolve_ref
    vis = set(visible_ids or ())
    n = ns = nc = nu = nsupp0 = 0
    for x in _d:
        if not isinstance(x, dict):
            continue
        rb = str(x.get("ruled_out_by") or "").strip()
        if not rb:
            continue
        n += 1
        sup = {str(y).strip() for y in (x.get("supporting_evidence") or []) if str(y).strip()}
        if not sup:
            nsupp0 += 1
        else:
            ns += 1
            if rb in sup:
                nc += 1
        if vis and _resolve_ref(rb, vis) is None:
            nu += 1
    if not n:
        return {"sce_n_excluded": 0, "sce_contradiction_rate": None,
                "sce_note": "no non-empty ruled_out_by -> not applicable (not scored as full marks)"}
    return {"sce_n_excluded": n, "sce_n_with_support": ns,
            "sce_n_no_support": nsupp0,
            "sce_self_contradictory": nc,
            "sce_contradiction_rate": (round(nc / ns, 3) if ns else None),
            "sce_unrevealed": (nu if vis else None),
            "sce_unrevealed_rate": (round(nu / n, 3) if vis else None)}


def _wnd_norm(s) -> str:
    """Normalize a prohibition-list item: strip whitespace and trailing punctuation only,
    so the boilerplate rate reflects what the model copied, not the normalizer.
    """
    t = re.sub(r"\s+", "", str(s or ""))
    return t.rstrip("。.;；,，、").strip()


def judge_what_not_to_do(out) -> dict:
    """`what_not_to_do`: is the prohibition list specific to this case?

    Reports per-cell structural counts; the cross-case boilerplate rate is aggregated per
    solver in `what_not_to_do_boilerplate`. Self-contradiction against `specific_action`
    is deliberately not checked: "don't adjust medication on your own" and "the physician
    adjusts the dose after referral" do not contradict, and a word scan cannot tell the two
    subjects apart. An empty list is not-applicable, not 0.
    """
    _act = getattr(out, "action", None) or {}
    if not isinstance(_act, dict):
        return {}
    items = _act.get("what_not_to_do")
    if not isinstance(items, list) or not items:
        return {"wnd_n_items": 0, "wnd_dup_rate": None,
                "wnd_note": "gave no prohibition items -> not applicable (not scored as full marks)"}
    norm = [_wnd_norm(x) for x in items]
    norm = [x for x in norm if x]
    if not norm:
        return {"wnd_n_items": len(items), "wnd_dup_rate": None,
                "wnd_note": "all prohibition items are empty strings -> not applicable"}
    return {"wnd_n_items": len(norm), "wnd_n_distinct": len(set(norm)),
            # Duplicates within one cell (padding), separate from the cross-case boilerplate rate.
            "wnd_dup_rate": round(1 - len(set(norm)) / len(norm), 3),
            "wnd_items_norm": norm}


def what_not_to_do_boilerplate(rows) -> dict:
    """Cross-case boilerplate rate per solver.

    For each of a solver's prohibition items, the fraction of the solver's other cells in
    which it reappears verbatim, averaged over items = `wnd_repeat_frac`; case specificity
    = 1 - that. No threshold. A solver with a single cell is not-applicable.
    """
    by: dict[str, list[list[str]]] = {}
    for r in rows:
        ns = r.get("wnd_items_norm")
        if isinstance(ns, list) and ns:
            by.setdefault(str(r.get("solver")), []).append([str(x) for x in ns])
    out: dict[str, dict] = {}
    for slv, grids in by.items():
        n = len(grids)
        if n < 2:
            out[slv] = {"wnd_n_grids": n, "wnd_repeat_frac": None,
                        "wnd_case_specificity": None,
                        "wnd_note": "only 1 cell -> cross-case boilerplate rate not applicable"}
            continue
        cnt: dict[str, int] = {}
        for g in grids:
            for it in set(g):
                cnt[it] = cnt.get(it, 0) + 1
        fr = [(cnt[it] - 1) / (n - 1) for g in grids for it in set(g)]
        rf = sum(fr) / len(fr)
        out[slv] = {"wnd_n_grids": n, "wnd_n_distinct_items": len(cnt),
                    "wnd_repeat_frac": round(rf, 3),
                    "wnd_case_specificity": round(1 - rf, 3)}
    return out


def _split_urgency_gap(gaps: list) -> dict:
    """Split the signed urgency gap into omission and overcommitment.

    `urgency_gap` from `judge_workup` = model tier index - gold tier index, with
    `_ACTION_ORDER` running A0..A5 from least to most severe: `> 0` is overcommitment
    (escalated more than needed), `< 0` is omission (should have escalated). The two have
    opposite clinical consequences and are reported separately; no combined
    total is given. Synthetic baselines anchor the scale (an always-lowest-tier stub can
    only land on the omission side). An empty list is not-applicable.
    """
    if not gaps:
        return {"wk_n_gap_scored": 0, "wk_omission_rate": None,
                "wk_overcommit_rate": None, "wk_gap_mean": None,
                "wk_gap_note": "no signed urgency gap -> not applicable (not scored as full marks)"}
    n = len(gaps)
    om = sum(1 for g in gaps if g < 0)  # model below gold: omission
    ov = sum(1 for g in gaps if g > 0)  # model above gold: overcommitment
    return {"wk_n_gap_scored": n,
            "wk_n_omission": om, "wk_n_overcommit": ov,
            "wk_omission_rate": round(om / n, 3),
            "wk_overcommit_rate": round(ov / n, 3),
            "wk_gap_mean": round(sum(gaps) / n, 3),
            "wk_gap_exact_rate": round((n - om - ov) / n, 3)}


def judge_slices_workup(rows, vp, ctx) -> dict:
    """Workup, per-slice version: the same patient gets a disposition at every visit, so
    when it first reaches the right tier is a reading. Reports the first slice that met the
    bar, the number of slices that did, and test-order recall over the course.
    """
    per = [judge_workup(_SliceOut(r), vp, ctx) for r in rows]
    oks = [i for i, p in enumerate(per, 1) if p.get("urgency_ok")]
    urgency_judged = [p["urgency_ok"] for p in per if isinstance(p.get("urgency_ok"), bool)]
    urgency_last = per[-1].get("urgency_ok") if per else None
    rec = [p.get("tests_recall") for p in per if p.get("tests_recall") is not None]
    gaps = [p.get("urgency_gap") for p in per if p.get("urgency_gap") is not None]
    # Precision and the discriminative-test recall are surfaced too; without them these
    # scoring dimensions would be empty on slice batches.
    prec = [p.get("tests_precision") for p in per if p.get("tests_precision") is not None]
    _last = _SliceOut(rows[-1]) if rows else None
    _disc = judge_discriminative_tool(_last, vp, ctx) if _last is not None else {}
    return {"wk_urgency_ok_at": oks[0] if oks else None,
            "wk_n_slices_urgency_ok": len(oks),
            "wk_urgency_ok_last": urgency_last if isinstance(urgency_last, bool) else None,
            "wk_urgency_ok_rate": (round(sum(urgency_judged) / len(urgency_judged), 4)
                                   if urgency_judged else None),
            "wk_n_slices_urgency_judged": len(urgency_judged),
            "wk_urgency_gaps": gaps,
            **_split_urgency_gap(gaps),
            "wk_tests_recall_max": max(rec) if rec else None,
            "wk_tests_recall_last": rec[-1] if rec else None,
            "wk_tests_precision_last": prec[-1] if prec else None,
            "wk_disc_recall_last": _disc.get("disc_recall"),
            # Last slice, not the union: the item tests whether the workup is assembled across visits,
            # and the last visit has the most information.
            "wk_basis": "last_slice"}


#: Dead zone for `flat` on trend questions: a net change under this fraction of the series'
#: own amplitude has no direction. It decides the gold label and is defined only here
#: (`evaluate.build_quant_probe` imports it).
#:
#: The threshold sits near the median of `|Δ|/rng` across trend probes, so small changes
#: to it flip a noticeable share of labels. Retuning only swaps which examples flip; the
#: fix is a less synthetic signal source, listed under limitations.
QUANT_TREND_DEADZONE = 0.02

#: Percentile index divisor for `abnormal_days` ("outside its own p10/p90"): `lo = vs[len(vs)//10]`.
#: With n < 10 it degenerates to min/max and the count is always 0; `build_quant_probe`
#: drops series with a count of 0, so such series never reach a question.
QUANT_PCT_INDEX_DIVISOR = 10


def _quant_truth(kind: str, pts: list[tuple[int, float]]):
    """Recompute the ground truth over a window of points with the same formula and
    constants as `build_quant_probe`.
    """
    pts = sorted(pts)
    if kind == "trend":
        d = pts[-1][1] - pts[0][1]
        rng = max(abs(v) for _, v in pts) or 1.0
        _dz = QUANT_TREND_DEADZONE * rng
        return "rising" if d > _dz else ("falling" if d < -_dz else "flat")
    if kind == "peak_day":
        return int(max(pts, key=lambda x: x[1])[0])
    if kind == "peak_value":
        # Peak value is unique even when the peak day is tied.
        return round(max(v for _, v in pts), 6)
    vs = sorted(v for _, v in pts)
    _k = len(vs) // QUANT_PCT_INDEX_DIVISOR
    lo, hi = vs[_k], vs[-1 - _k]
    return sum(1 for _, v in pts if not (lo <= v <= hi))


def judge_quant_probe(out, probe: dict | None, t_max: int | None = None) -> dict:
    """Computable question: did the model read the data accurately?

    Truth is computed by code, the answer is a structured enum or integer
    (`_raw["quant_answer"]`), and the comparison is exact, so no annotated anchor is needed.
    An unanswered question scores 0, not not-applicable.
    """
    if not probe:
        return {}
    raw = getattr(out, "_raw", None) or {}
    got = raw.get("quant_answer") if isinstance(raw, dict) else None
    # Truth is recomputed over the window of the slice actually answered (series is persisted).
    truth = probe.get("truth")
    ser = probe.get("series") or []
    if ser and t_max is not None:
        vis = [(int(a), float(b)) for a, b in ser if int(a) <= int(t_max)]
        if len(vis) >= 4:
            truth = _quant_truth(probe.get("kind"), vis)
    # `peak_day` accepts any tied peak day; generation also avoids emitting tied peaks.
    _peak_days: list[int] = []
    if probe.get("kind") == "peak_day":
        _pool = [(int(a), float(b)) for a, b in ser] if ser else []
        if t_max is not None:
            _pool = [(a, b) for a, b in _pool if a <= int(t_max)]
        if _pool:
            _mx = max(v for _, v in _pool)
            _peak_days = sorted(a for a, v in _pool if abs(v - _mx) < 1e-9)

    # `trend` questions are built on the answered window behind a window-stability gate
    # (`evaluate.assign_quant_probes`), so this recomputation equals the stored truth there;
    # `peak_day` is still built at T and recomputed here.
    if isinstance(truth, str):
        ok = str(got or "").strip().lower() == truth
    elif probe.get("kind") == "peak_value":
        # Relative tolerance 1e-6, matching the 6-decimal rounding of the series.
        try:
            _g, _t = float(got), float(truth)
            ok = abs(_g - _t) <= max(1e-6, abs(_t) * 1e-6)
        except (TypeError, ValueError):
            ok = False
    elif _peak_days:
        try:
            ok = int(float(got)) in _peak_days  # any tied peak counts
        except (TypeError, ValueError):
            ok = False
    else:
        try:
            ok = int(float(got)) == int(truth)
        except (TypeError, ValueError):
            ok = False
    return {"quant_kind": probe.get("kind"), "quant_signal": probe.get("signal"),
            "quant_answered": got is not None, "quant_ok": float(bool(ok)),
            # Number of correct answers (tied peaks), persisted per case.
            "quant_peak_days": (_peak_days or None),
            "quant_peak_tied": (len(_peak_days) > 1 if _peak_days else None),
            # Truth and answer persisted, so a low score can be diagnosed without re-running.
            "quant_truth": truth, "quant_answer": got,
            "quant_t_max": (int(t_max) if t_max is not None else None),
            "quant_n_visible": (len([1 for a, _ in ser if t_max is None or int(a) <= int(t_max)])
                                if ser else None)}


def judge_abstention_calibration(out, vp, ctx=None) -> dict:
    """Abstention calibration: abstain when information is insufficient, not when it is
    sufficient. Scoring both halves stops "always abstain" and "never abstain" from
    scoring perfectly.

    Sufficiency comes from item generation (`latent.ddx_insufficient`); abstention reads only
    `data_quality.data_sufficiency == insufficient_data`. The action tier is ignored:
    A1 (improve data quality) is an action, not a sufficiency declaration. Tiers are
    defined in the kernel's `solver.ACTION_CLASSES`.
    """
    adj = (getattr(vp, "adjudication", None) or {})
    ddx = adj.get("ddx") or {}
    # Sufficiency comes from an explicit declaration, not from whether a gold diagnosis
    # exists: every ddx item has one.
    sufficient = bool(ddx.get("diagnosis")) and not bool(ddx.get("insufficient"))
    dq = getattr(out, "data_quality", None) or {}
    if not isinstance(dq, dict):
        dq = {}
    abstained = str(dq.get("data_sufficiency") or "") == "insufficient_data"
    return {"abst_sufficient": sufficient, "abst_abstained": bool(abstained),
            # calibrated = (sufficient and did not abstain) or (insufficient and abstained)
            "abst_ok": float(abstained != sufficient),
            # over-abstention: sufficient but abstained
            "abst_over": float(sufficient and abstained)}


def judge_commit_timing(out, vp, ctx=None) -> dict:
    """Should a diagnosis be committed now? For the `ddx:insufficient` gold class.

    Index time is moved before the first symptom, so the correct answer is to declare
    insufficient information and not name a diagnosis. The emission gate runs in the
    opposite direction for this class (fewer than 2 visible true symptoms).

    * `ct_declared` -- declared insufficient (`data_quality.data_sufficiency`);
    * `ct_forced_dx` -- declared sufficient while naming a top-1 diagnosis (overconfidence,
      regardless of correctness);
    * `ct_ok` -- declared insufficient and named no diagnosis.

    This class must be mixed with ordinary items in a batch; alone, "always insufficient"
    would score perfectly.
    """
    from ..tracks import _differential
    dq = getattr(out, "data_quality", None) or {}
    if not isinstance(dq, dict):
        dq = {}
    declared = str(dq.get("data_sufficiency") or "") == "insufficient_data"
    _diff = _differential(out) or []
    top1 = _diff[0] if _diff else {}
    named = bool(str(top1.get("diagnosis") or "").strip())
    forced = bool(named and not declared)
    # `declared and not named`, not `declared and not forced`: the latter reduces to
    # `declared` and misses a model that declared insufficient yet still named a diagnosis.
    return {"ct_declared": float(declared),
            "ct_forced_dx": float(forced),
            "ct_named_dx": float(named),
            "ct_ok": float(declared and not named)}


def judge_slices_abstention(rows, vp, ctx=None) -> dict:
    """Abstention calibration per slice: on ddx items every case has a gold diagnosis, so
    the single-shot version is one-sided. Early slices (no true symptom visible) should
    declare insufficient; later slices should not. Visibility comes from the verifier
    side (`visible_real`).

    Returns `abst_ok` (agreement over slices), the slice counts per side
    (`abst_n_insufficient` / `abst_n_sufficient`; calibration is not measurable if either
    is 0), `abst_over` and `abst_under`.
    """
    if not rows:
        return {}
    pairs = []
    for r in rows:
        suff = bool(r.get("visible_real"))
        ds = str(r.get("data_sufficiency") or "")
        if not ds:
            continue  # field not recorded: skip, not 0
        pairs.append((suff, ds == "insufficient_data"))
    if not pairs:
        return {"abst_ok": None, "abst_note": "slice did not record data_sufficiency"}
    n_ins = sum(1 for s, _ in pairs if not s)
    n_suf = len(pairs) - n_ins
    ok = sum(1 for s, a in pairs if a != s)
    over = sum(1 for s, a in pairs if s and a)
    under = sum(1 for s, a in pairs if (not s) and (not a))
    return {"abst_ok": round(ok / len(pairs), 3),
            "abst_n_slices": len(pairs),
            "abst_n_insufficient": n_ins, "abst_n_sufficient": n_suf,
            "abst_two_sided": bool(n_ins and n_suf),
            "abst_over": round(over / n_suf, 3) if n_suf else None,
            "abst_under": round(under / n_ins, 3) if n_ins else None}


#: The noop answer contract: `data_quality.signal_quality[<target>]` takes exactly one of
#: these. `present` = the asked window has readings; `no_data_in_window` = it has none;
#: `unreliable` = it has readings the model does not trust (the kernel's suspect-flag word,
#: `core/verifier.py`). Written into the question by `evaluate._noop_suffix`.
NOOP_ANSWERS: tuple[str, ...] = ("present", "no_data_in_window", "unreliable")

#: Version of the noop answer contract, recorded on the probe and on the row;
#: `tools/recompute_judges.py` refuses to judge a row answered under another contract.
NOOP_CONTRACT = "enum-v1"

# An explanation may follow the enum value after one of these (`no_data_in_window; 第 43–83 天`).
_NOOP_DELIMS = frozenset(" \t;,:()[]/|-" + "\uff1b\uff0c\uff1a\uff08\uff09\u3010\u3011\u2014\u3001")


def noop_answer_of(value) -> str | None:
    """Parse one `signal_quality[target]` value into the enum, or None (off-enum).

    Accepts the value itself or a leading enum token followed by a delimiter; free text,
    another language or a paraphrase is off-enum.
    """
    if not isinstance(value, str):
        return None
    t = value.strip().strip("`'\"“”").strip().lower()
    if t in NOOP_ANSWERS:
        return t
    for a in sorted(NOOP_ANSWERS, key=len, reverse=True):
        if t.startswith(a) and len(t) > len(a) and t[len(a)] in _NOOP_DELIMS:
            return a
    return None


#: Marker words of the contract before `enum-v1`, when the question asked only for
#: `no_data_in_window` on an empty window. They carry negation, so `has_data_in_window`
#: does not match.
_NOOP_MARKERS_V0 = ("no_data", "no data", "nodata", "no readings", "no reading",
                    "not available", "unavailable", "missing", "absent",
                    "无读数", "无数据", "没有数据", "缺失", "未提供")


def _judge_noop_markers_v0(out, probe: dict) -> dict:
    """The rule for a probe asked before `enum-v1` (the probe carries no `contract`): a
    marker word in `signal_quality[target]` declares absence; `noop_ok` is consistency, so
    silence on a covered window is correct. Kept so an answer is judged by the question it
    was asked (`tools/recompute_judges.py` on existing batches)."""
    dq = (getattr(out, "data_quality", None) or {})
    if not isinstance(dq, dict):
        dq = {}
    sq = dq.get("signal_quality") or {}
    tgt = str(probe.get("target") or "")
    _v = str(sq.get(tgt) or "").lower() if isinstance(sq, dict) else ""
    declared = any(m in _v for m in _NOOP_MARKERS_V0)
    present = bool(probe.get("truth_present"))
    return {"noop_polarity": probe.get("polarity"), "noop_target": tgt,
            "noop_declared": bool(declared),
            "noop_ok": float(declared != present),
            "noop_window": probe.get("window"),
            "noop_n_pts_in_window": probe.get("n_pts_in_window"),
            "noop_truth_present": present,
            "noop_used_global_flag": float(
                str(dq.get("data_sufficiency") or "") == "insufficient_data" and not _v),
            "noop_answer_channel": ("signal_quality" if _v else
                                    ("global_only" if str(dq.get("data_sufficiency") or "")
                                     == "insufficient_data" else "none"))}


def judge_noop_probe(out, probe: dict | None) -> dict:
    """NoOp probe: asked about one window of a signal the model has seen, does it report
    availability correctly?

    Reads one structured field, `data_quality.signal_quality[target]`, parsed into
    `NOOP_ANSWERS`. Correct: `no_data_in_window` on an empty window; `present` or
    `unreliable` on a covered one (neither claims absence). Unanswered or off-enum scores 0,
    so neither "always claim absence" nor "stay silent" scores. A probe without a
    `contract` was asked before the enum and is judged by `_judge_noop_markers_v0`.
    """
    if not probe:
        return {}
    if probe.get("contract") is None:
        return _judge_noop_markers_v0(out, probe)
    dq = (getattr(out, "data_quality", None) or {})
    if not isinstance(dq, dict):
        dq = {}
    sq = dq.get("signal_quality") or {}
    tgt = str(probe.get("target") or "")
    _raw_v = sq.get(tgt) if isinstance(sq, dict) else None
    _v = str(_raw_v or "").strip() if _raw_v is not None else ""
    ans = noop_answer_of(_raw_v)
    declared = ans == "no_data_in_window"
    present = bool(probe.get("truth_present"))
    ok = (ans in ("present", "unreliable")) if present else declared
    return {"noop_polarity": probe.get("polarity"), "noop_target": tgt,
            "noop_declared": bool(declared),
            "noop_ok": float(bool(ok)),
            "noop_answer": ans,
            "noop_off_enum": bool(_v) and ans is None,
            "noop_contract": probe.get("contract"),
            "noop_window": probe.get("window"),
            "noop_n_pts_in_window": probe.get("n_pts_in_window"),
            "noop_truth_present": present,
            # the global flag used instead of the specific answer
            "noop_used_global_flag": float(
                str(dq.get("data_sufficiency") or "") == "insufficient_data" and not _v),
            "noop_answer_channel": ("signal_quality" if _v else
                                    ("global_only" if str(dq.get("data_sufficiency") or "")
                                     == "insufficient_data" else "none"))}


#: `join_evidence` must cite at least this many distinct evidence ids.
JOIN_EVIDENCE_MIN = 2


def judge_join_evidence(out, vp, ctx=None) -> dict:
    """`join_evidence`: the evidence ids the answer rests its `join_type` on, checked by code.

    Correct (`join_ev_ok = 1`) when at least `JOIN_EVIDENCE_MIN` distinct ids are cited, every
    one resolves to this case's ledger (exactly, or as a unique abbreviation through
    `process._resolve_ref`), and every one is a symptom of the gold presentation visible at
    the judged time: a real symptom or a look-alike (`wq` Q-side ledger), not a benign
    distractor. Not applicable (`None`) when the case's question did not ask for the field
    (`answer_contract` absent) or fewer than two gold symptoms are visible.
    """
    from .. import process, wq
    from .differential import visible_ids_of
    man = wq.injected_manifest(getattr(vp, "case_id", ""), required=False) or {}
    required = str((man.get("answer_contract") or {}).get("join_evidence") or "") == "required"
    if not required:
        return {"join_ev_required": False, "join_ev_ok": None}
    T = int(getattr(vp, "T", 10**9) or 10**9)
    sched = man.get("event_schedule") or []
    if sched:
        gold = {str(e.get("evidence_id")) for e in sched
                if str(e.get("kind")) in ("real_symptom", "lookalike")
                and int(e.get("day", 10**9)) <= T}
    else:
        gold = ({str(x) for x in (man.get("real_symptom_evidence_ids") or [])}
                | {str(x) for x in (man.get("lookalike_evidence_ids") or [])})
    raw = getattr(out, "_raw", None) or {}
    cited = [str(x) for x in process._ev_ids(raw.get("join_evidence"))]
    vis = set(visible_ids_of(vp, ctx))
    resolved = [process._resolve_ref(r, vis) for r in cited]
    valid = sorted({r for r in resolved if r is not None})
    n_invalid = sum(1 for r in resolved if r is None)
    n_abbrev = sum(1 for c, r in zip(cited, resolved) if r is not None and r != c)
    off_gold = [r for r in valid if r not in gold]
    base = {"join_ev_required": True, "join_ev_n_cited": len(cited),
            "join_ev_n_valid": len(valid), "join_ev_n_invalid": n_invalid,
            "join_ev_n_abbrev": n_abbrev, "join_ev_n_offgold": len(off_gold),
            "join_ev_n_gold_visible": len(gold)}
    if len(gold) < JOIN_EVIDENCE_MIN:
        return {**base, "join_ev_ok": None}
    ok = len(valid) >= JOIN_EVIDENCE_MIN and n_invalid == 0 and not off_gold
    return {**base, "join_ev_ok": float(bool(ok))}


def judge_premise_challenge(out, prem: dict | None) -> dict:
    """Did the model flag that the patient's self-report contradicts the chart?

    `prem_flagged` should be 1 on false premises and 0 on true ones; a flag on a true
    premise is a false alarm. Reads `notes` and `data_quality.note`, not the full free text.
    """
    if not prem:
        return {}
    txt = " ".join(str(x) for x in (
        (getattr(out, "notes", "") or ""),
        ((getattr(out, "_raw", {}) or {}).get("notes") or ""),
        ((getattr(out, "_raw", {}) or {}).get("data_quality") or {}).get("note", "") or "",
    ))
    field = str(prem.get("field") or "")
    # A hit needs both the quantity name and conflict wording.
    _mk = ("不符", "不一致", "与资料", "矛盾", "对不上", "有出入", "存疑", "无法核实",
           "记录显示", "实际为", "并非")
    hit = bool(field and field in txt) and any(w in txt for w in _mk)
    return {"prem_polarity": prem.get("polarity"), "prem_field": field,
            "prem_flagged": bool(hit)}


_CONCEPT_IDX: dict | None = None


def _concept_index() -> dict:
    """Alias -> concept id over the registered conditions and their rivals.

    Belief identity goes through concepts, so a reworded diagnosis maps to the same set
    and a different condition to a different one.
    """
    global _CONCEPT_IDX
    if _CONCEPT_IDX is not None:
        return _CONCEPT_IDX
    idx: dict[str, str] = {}
    try:
        # Start from the supplemented kernel specs (`registry/condition_aliases.yaml`), as
        # `condition_registry` does.
        from ..overlay import _kernel_specs, haenv_comorbid_specs, haenv_independent_specs, RIVALS
        _k = _kernel_specs()
        specs = {**_k, **haenv_comorbid_specs(_k), **haenv_independent_specs()}
        for sid, sp in specs.items():
            for a in (sp.get("aliases") or ()):
                a = str(a).strip().lower()
                if len(a) >= 2:
                    idx.setdefault(a, sid)
        for sid, rivs in RIVALS.items():
            for r in rivs:
                rid = f"rival:{r.get('name')}"
                for a in list(r.get("aliases") or ()) + [str(r.get("name") or "")]:
                    a = str(a).strip().lower()
                    if len(a) >= 2:
                        idx.setdefault(a, rid)
    except Exception:  # vocabulary unavailable -> empty
        idx = {}
    _CONCEPT_IDX = idx
    return idx


def concept_set(text: str) -> frozenset:
    """Map one diagnosis text to its concept-id set, falling back to a normalized string
    when no concept matches (an empty set would make every answer look unchanged).
    """
    t = str(text or "").lower()
    hits = {cid for a, cid in _concept_index().items() if a in t}
    if hits:
        return frozenset(hits)
    return frozenset({"raw:" + "".join(ch for ch in t if ch.isalnum())[:24]})


def judge_premise_repair(traj, prem: dict | None, ctx=None) -> dict:
    """Repair: after evidence refutes a premise, does the subject change its position?

    | quantity | asks | good direction |
    |---|---|---|
    | `rep_flagged_after` | flagged a conflict in a round after the refuting evidence became visible | 1 on false premises |
    | `rep_flagged_before` | flagged before the evidence was visible | 0 |
    | `rep_latency_rounds` | rounds from visibility to the first flag | smaller |

    Flagging both before and after is `blanket_doubt`, not repair. Not applicable on true
    premises (returns a note, not 0).
    """
    if not isinstance(traj, list) or not traj or not prem:
        return {}
    if prem.get("polarity") != "false":
        return {"rep_note": "true premise -> there is no refuting evidence to speak of, this judge is not applicable (not 0)"}
    ev_id = str(prem.get("evidence_id") or "")
    field = str(prem.get("field") or "")
    # Round the refuting evidence first becomes visible, matched by EV id.
    first_vis = None
    for row in traj:
        vis = set(row.get("visible_ev") or ())
        if ev_id and ev_id in vis:
            first_vis = int(row.get("round") or 0)
            break
    if first_vis is None:
        return {"rep_note": f"the refuting evidence ({ev_id or field}) is not found in any round's visible set "
                            f"-> a scheduling issue, not a model issue: this item cannot test repair"}
    _MK = ("不符", "不一致", "与资料", "矛盾", "对不上", "有出入", "存疑",
           "无法核实", "记录显示", "实际为", "并非")
    flagged_rounds = []
    for row in traj:
        txt = str(row.get("notes") or "")
        if field and field in txt and any(w in txt for w in _MK):
            flagged_rounds.append(int(row.get("round") or 0))
    before = [r for r in flagged_rounds if r < first_vis]
    after = [r for r in flagged_rounds if r >= first_vis]
    if before and after:
        verdict = "blanket_doubt"
    elif after:
        verdict = "repaired"
    elif before:
        verdict = "doubt_without_evidence"
    else:
        verdict = "never_flagged"
    return {"rep_first_visible_round": first_vis,
            "rep_flagged_after": bool(after),
            "rep_flagged_before": bool(before),
            "rep_latency_rounds": (min(after) - first_vis) if after else None,
            "rep_verdict": verdict,
            "rep_n_rounds": len(traj)}


def judge_multiround_revision(traj, vp=None, ctx=None) -> dict:
    """Multi-round revision: is a revision grounded in new evidence from that round, and
    does the model hold steady when nothing new arrives?

    Uses `run_multiround`'s `visible_ev`, `cited_ev` and `n_new_points` per round. New
    observation = new EV ids or new daily-series points.

    | quantity | asks | better |
    |---|---|---|
    | `mr_grounded_rate` | revisions with new information that round | higher |
    | `mr_cited_new_rate` | grounded revisions that cite the new evidence | higher (observational only) |
    | `mr_neutral_stability` | rounds without new information that did not revise | higher |

    The denominator is actual revisions, not self-reported ones.
    """
    if not isinstance(traj, list) or len(traj) < 2:
        return {"mr_note": "fewer than 2 rounds -> not applicable"}

    n_rev = n_rev_grounded = n_rev_grounded_ev = n_cited = 0
    n_neutral = n_neutral_stable = 0
    detail = []
    for a, b in zip(traj, traj[1:]):
        new_ev = set(b.get("visible_ev") or ()) - set(a.get("visible_ev") or ())
        new_pts = int(b.get("n_new_points") or 0)
        # Counting only EV ids penalizes revisions grounded in series data; counting series points
        # makes every round non-neutral. Both conventions are reported; choosing a significance
        # threshold for the series is left open.
        has_new_ev = bool(new_ev)
        has_new_any = has_new_ev or new_pts > 0
        has_new = has_new_any
        changed = str(b.get("repair")) == "revised"  # the kernel's per-round diff label
        if changed:
            n_rev += 1
            if has_new_any:
                n_rev_grounded += 1
                if new_ev & set(b.get("cited_ev") or ()):
                    n_cited += 1
            if has_new_ev:
                n_rev_grounded_ev += 1
        if not has_new:
            n_neutral += 1
            n_neutral_stable += int(not changed)
        detail.append({"round": b.get("round"), "changed": changed,
                       "n_new_ev": len(new_ev), "n_new_points": new_pts})

    out = {"mr_n_transitions": len(traj) - 1,
           "mr_n_revisions": n_rev,
           # wide: new EV ids or new series points
           "mr_grounded_rate": round(n_rev_grounded / n_rev, 3) if n_rev else None,
           # strict: new EV ids only
           "mr_grounded_rate_ev_only": round(n_rev_grounded_ev / n_rev, 3) if n_rev else None,
           # observational, not right-or-wrong
           "mr_cited_new_rate": round(n_cited / n_rev_grounded, 3) if n_rev_grounded else None,
           "mr_n_neutral": n_neutral,
           "mr_neutral_stability": (round(n_neutral_stable / n_neutral, 3)
                                    if n_neutral else None),
           "mr_transitions": detail or None}
    # No neutral rounds => `mr_neutral_stability` is not applicable, not a perfect score.
    out["mr_definition_note"] = (
        "Two conventions for \"new observation\" placed side by side: `mr_grounded_rate` "
        "(EV or series points) and `mr_grounded_rate_ev_only` (EV only). Both ends "
        "degenerate -- counting only EV would judge a solution that legitimately revised based "
        "on series data as ungrounded; counting series points means every "
        "round has a new observation and the neutral window can never exist. Choosing between "
        "them needs a significance threshold for the series (a design decision outside the judge).")
    if not n_neutral:
        out["mr_neutral_note"] = ("This batch has no neutral rounds (every round has a new "
                                  "observation) => `mr_neutral_stability` is not applicable. "
                                  "If this holds for the whole batch, the cadence and data density "
                                  "make the neutral window impossible by construction -- "
                                  "a scheduling issue, not model stability.")
    return out


def judge_slice_revision(rows, vp, ctx=None) -> dict:
    """Slice revision: revise when a span brings a genuine signal, hold steady when it
    brings only benign events.

    | | changed | unchanged |
    |---|---|---|
    | genuine signal | updated correctly | should have updated |
    | benign events only | ungrounded drift | held steady |

    `rev_stability` and `rev_responsiveness` are reported separately; the `flip_flop` and
    `no_revision` baselines each max out one of them. Belief is
    `(join_type, action, risk_category)` plus the concept set of the top diagnosis, so
    rewording is not a revision. A class of transition that never occurs gives `None`.
    """
    if not rows or len(rows) < 2:
        return {"rev_note": "fewer than 2 slices -> not applicable"}

    def _top1(r) -> str:
        d = (r.get("differential") or [None])
        x = d[0] if d else None
        return str((x.get("diagnosis") if isinstance(x, dict) else x) or "")

    def _differential_of(r) -> list:
        """Diagnosis texts of the differential list, from dict or plain-string entries."""
        return [str((x.get("diagnosis") if isinstance(x, dict) else x) or "")
                for x in (r.get("differential") or []) if x]

    def _belief(r):
        """Belief = `(join_type, action, risk_category)` + the concept set of the top entry.
        The whole-list concept set is a separate diagnostic layer (`rev_list_*`), not scored.
        """
        return (str(r.get("join_type") or ""), str(r.get("action") or ""),
                str(r.get("risk_category") or ""), concept_set(_top1(r)))

    def _list_concepts(r) -> frozenset:
        """Concept set of the whole differential list, `rival:` entries dropped."""
        out = set()
        for x in _differential_of(r):
            out |= set(concept_set(x))
        return frozenset(x for x in out if not str(x).startswith("rival:"))

    churn_neutral = churn_neutral_tot = churn_subst = churn_subst_tot = 0
    neutral_same = neutral_tot = subst_changed = subst_tot = 0
    detail = []
    for a, b in zip(rows, rows[1:]):
        if a.get("leak") or b.get("leak"):
            continue
        new_real = set(b.get("visible_real") or ()) - set(a.get("visible_real") or ())
        new_other = set(b.get("visible_other") or ()) - set(a.get("visible_other") or ())
        new_abn = set(b.get("visible_abnormal_lab") or ()) - set(a.get("visible_abnormal_lab") or ())
        changed = _belief(a) != _belief(b)
        # The candidate-list layer is tracked separately; folded in, every rewording would count.
        churn = _list_concepts(a) != _list_concepts(b)
        # An abnormal lab is a deterioration signal (substantive). A new benign symptom on the
        # `independent` class is neutral: the gold label says the events are unrelated.
        _gold_ind = str(((getattr(vp, "adjudication", None) or {}).get("ddx") or {}).get(
            "join_gold") or "") == "independent"
        _substantive = bool(new_abn) or (bool(new_real) and not _gold_ind)
        if _substantive:
            subst_tot += 1
            subst_changed += int(changed)
            churn_subst_tot += 1
            churn_subst += int(churn)
            detail.append({"t": b.get("t"), "kind": "substantive",
                           "n_new_real": len(new_real), "n_new_abnormal_lab": len(new_abn),
                           "basis": ("abnormal_lab" if new_abn else "real_symptom"),
                           "changed": changed})
        elif new_other or new_real:
            neutral_tot += 1
            neutral_same += int(not changed)
            churn_neutral_tot += 1
            churn_neutral += int(not churn)
            detail.append({"t": b.get("t"), "kind": "neutral",
                           "n_new_other": len(new_other), "n_new_real": len(new_real),
                           "basis": ("benign_real_on_independent" if new_real else "benign_other"),
                           "changed": changed})
        # no new observation: not counted
    return {"rev_stability": round(neutral_same / neutral_tot, 3) if neutral_tot else None,
            "rev_n_neutral": neutral_tot,
            "rev_responsiveness": round(subst_changed / subst_tot, 3) if subst_tot else None,
            "rev_n_substantive": subst_tot,
            # ---- candidate-list layer: diagnostic only, not folded into the ratios above ----
            "rev_list_stability": (round(churn_neutral / churn_neutral_tot, 3)
                                   if churn_neutral_tot else None),
            "rev_list_responsiveness": (round(churn_subst / churn_subst_tot, 3)
                                        if churn_subst_tot else None),
            "rev_transitions": detail or None}


def judge_slices(rows, vp, ctx) -> dict:
    """Multi-slice convergence; the definition depends on the gold kind.

      unified      the top pick hits the diagnosis
      comorbidity  the candidate set covers every disease thread
      independent  whether any slice over-merged (convergence is not asked)
    """
    # No `unified` fallback: if the kind cannot be inferred, no ddx branch is attached.
    k = gold_kind(vp)
    kind = k.split(":", 1)[1] if k.startswith("ddx:") else None
    n = len(rows)
    # `raw_empty` distinguishes a failed call (a default abstention with canned text) from an
    # answer; rows without the field fall back to checking for an answer.
    _has_flag = any("raw_empty" in (r or {}) for r in rows)
    if _has_flag:
        _answered = sum(1 for r in rows if not r.get("raw_empty"))
    else:
        _answered = sum(1 for r in rows if r.get("all_drivers") or r.get("answer_text"))
    base = {"n_slices": n, "answered_slices": _answered,
            # Missing slices stay visible, so a gateway outage is not read as model behavior.
            "slices_missing": (n - _answered) or None,
            "slices_answered_source": ("raw_empty" if _has_flag else "legacy_answer_text")}
    if kind == "independent":
        # Merging is only judged when at least 2 symptoms are visible.
        judged = [(i, r) for i, r in enumerate(rows, 1) if int(r.get("n_symptoms", 2) or 0) >= 2]
        over = [i for i, r in judged
                if str(r.get("join_type") or "").lower() in ("unified", "comorbidity")]
        return {**base, "slice_kind": "independent",
                "n_slices_judged": len(judged),  # slices with <2 symptoms are exempt
                "over_unified_at": over[0] if over else None,
                "n_slices_over_unified": len(over),
                "held_independent": bool(judged) and not over}
    if kind == "comorbidity":
        per = [_slice_threads(r, vp) for r in rows]
        basis = per[0][2] if per else "flat_alias_proxy"
        hit = next((i for i, (m, tot, _) in enumerate(per, 1) if tot and m >= tot), None)
        one = next((i for i, (m, _t, _b) in enumerate(per, 1) if m >= 1), None)
        # `thread_basis` is kept: on the alias-proxy fallback, "did not converge" may be the alias
        # table missing the model's phrasing.
        return {**base, "slice_kind": "comorbidity", "thread_basis": basis,
                "all_threads_at": hit, "first_thread_at": one,
                "late_convergence": bool(hit is not None and n > 1 and hit == n)}
    if kind == "unified":
        hit = next((i for i, r in enumerate(rows, 1) if _slice_hit_unified(r, vp)), None)
        from ..events import alias_hit
        ment = next((i for i, r in enumerate(rows, 1)
                     if alias_hit(" ".join([str(x) for x in (r.get("differential") or [])]
                                           + [str(r.get("answer_text") or "")]), _names(vp))), None)
        return {**base, "slice_kind": "unified", "converged_at": hit, "mentioned_at": ment,
                "late_convergence": bool(hit is not None and n > 1 and hit == n),
                "never_converged": hit is None}
    from ..tracks import slice_convergence
    return {**base, "slice_kind": "driver",
            **{k: v for k, v in slice_convergence(rows, (_gold(vp, "gold_drivers") or [None])[0]).items()
               if k not in base}}


# Judge x geometry mounting is defined in `mount_table.MOUNT` (reasons for blank cells in
# `mount_table.WHY_NOT`).
