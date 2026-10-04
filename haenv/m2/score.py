"""M2 scoring (spec 2.1-2.4): round-2 atoms per row and the M2 composite per solver.

Composite = mean over cases of (mean of the case's applicable scored dims, the solver-level
chance-corrected dims included) x the hard-gate multiplier (M1: propensity gates outside it),
the reading `z4_margin.py` derives the Z-4 threshold from.

Chance correction: `cc = (m*BA - 1) / (m - 1)` on the RAW balanced accuracy, m = number of gold
classes present for that solver and dim (m < 2 => the dim is not applicable). The composite, its
bootstrap and every gate use the unfloored dim values; `display()` floors at 0 for the board only.

Review (revision r3-A2): `review_utility_cc` is the chance-corrected BA over the F1-F3 rows,
pooled. F0 (the bridge) is all "warranted", so it carries no review contrast and is read as the
profile `review_f0_sens`; a batch without F1-F3 rows (the v1.0.1 workup) pools every row (M1).
The frame is kept uninformative by the pack composition (`docs/scripts/m2/pack1_select.py`), not
by a correction of the score. Hard gates (r4): composite = max(0, c) * mult, so a below-chance
answer scores 0, a failed gate never raises it, and the per-dim raw values stay in the row.

Scored:  tests_recall_f1, dx_listed, quant_ok, review_utility_cc (single round, revision r1-0).
Profile: abst_utility_cc (revision R-16: follows M1 bd9a918e -- correct, but all ten models sit at
         chance, so it is computed, stored and shown, not scored), revision_acc, revision_ba_cc, dx_listed_n0, key_acquired (round 1), and the optional
         round-2 profile (`core.ROUND2`, off): dx_listed_r2, dx_listed_r2_w, buy_efficiency,
         review_r2, urgency_cc (F1), med_direction_cc (F2), rcv_acc (F3), revision_distracted,
         noop_ok, tool_grounded_joint.
"""
from __future__ import annotations

import functools
import math
from types import SimpleNamespace

#: M2 is a single round; `dx_listed_r2_w` and the other round-2
#: readings are an optional profile (`core.ROUND2`, off by default) and never enter the composite.
SCORED_CASE_DIMS = ("tests_recall_f1", "dx_listed", "quant_ok")
SCORED_SOLVER_DIMS = ("review_utility_cc",)
#: Revision R-16 (2026-10-01): abstention utility is a profile reading, as in M1 (bd9a918e).
PROFILE_SOLVER_DIMS = ("abst_utility_cc",)
SCORED_DIMS = SCORED_CASE_DIMS + SCORED_SOLVER_DIMS
CC_DIMS = ("review_utility_cc", "abst_utility_cc", "revision_ba_cc", "urgency_cc", "med_direction_cc", "rcv_cc")
PROFILE_DIMS = ("abst_utility_cc", "review_f0_sens", "revision_acc", "revision_ba_cc", "dx_listed_n0", "dx_listed_r2", "dx_listed_r2_w",
                "key_acquired",
                "buy_efficiency", "review_r2", "urgency_cc", "med_direction_cc", "rcv_acc", "rcv_cc",
                "revision_distracted", "noop_ok", "tool_grounded_joint")
#: Weight of a gold line listed in round 2 whose key signals came from the push (spec 2.1).
PUSH_WEIGHT = 0.5
#: revision_ba_cc needs at least this many cases per class (spec 2.1).
MIN_PER_CLASS = 5


def cc_raw(ba, m: int):
    """Chance-corrected balanced accuracy, unfloored; None when m < 2 or BA is None."""
    if ba is None or m is None or m < 2:
        return None
    return (m * float(ba) - 1.0) / (m - 1.0)


def display(x):
    """Board display of a chance-corrected value: floored at 0 (never used in a computation)."""
    return None if x is None else max(0.0, float(x))


def _mean(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    return math.fsum(xs) / len(xs) if xs else None


def balanced(pairs) -> tuple[float | None, int]:
    """(BA over the gold classes present, m) from (gold, ok) pairs."""
    by: dict = {}
    for g, ok in pairs:
        if g is None or ok is None:
            continue
        by.setdefault(g, []).append(float(ok))
    if not by:
        return None, 0
    return math.fsum(math.fsum(v) / len(v) for v in by.values()) / len(by), len(by)


# ------------------------------------------------------------------ round-2 atoms
@functools.lru_cache(maxsize=4096)
def _findings_of(target: str) -> tuple[str, ...]:
    """Finding ids a purchased F1-F3 menu item stands for (`core.m2_menu_findings`); () for a decoy."""
    from ..gated import DECOY_SIGNALS
    from ..registry import load_findings
    from .core import m2_menu_findings
    if target in DECOY_SIGNALS:
        return ()
    return tuple(m2_menu_findings(target, load_findings()))


def _bought(targets) -> set[str]:
    from ..gated import DECOY_SIGNALS
    out: set[str] = set()
    for t in dict.fromkeys(str(x) for x in (targets or []) if x):
        if t in DECOY_SIGNALS:
            continue
        out.add(t)
        out.update(_findings_of(t))
    return out


def _listed(raw: dict, alias_sets) -> dict:
    from ..judges.differential import _listed as _prod_listed
    return _prod_listed(SimpleNamespace(_raw=raw or {}), alias_sets)


def _top1(raw: dict) -> str:
    from ..tracks import _differential, entry_excluded, rank_key
    live = [str(x.get("diagnosis") or "") for x in sorted(_differential(SimpleNamespace(_raw=raw or {})),
                                                          key=rank_key) if not entry_excluded(x)]
    return live[0] if live else ""


def listed_flags(positions, cap) -> list[bool]:
    return [isinstance(p, int) and not isinstance(p, bool) and (not isinstance(cap, int) or p <= cap)
            for p in (positions or [])]


def r2_atoms(row: dict, plan: dict, *, push_weight: float = PUSH_WEIGHT,
             condition_on_r1: bool = True, require_keep: bool = True) -> dict:
    """Round-2 atoms from persisted fields only: round-1 `dx_listed_positions`/`dx_listed_cap`,
    `tool_targets`, `r2_raw`, `r2_alias_sets`, `r2_warranted`, and the plan. The keyword switches
    exist for the mutation tests (each must turn a Z gate red)."""
    from ..tracks import alias_hit_asserted
    from .core import distractor_id
    raw2 = row.get("r2_raw") or {}
    sets = row.get("r2_alias_sets") or []
    threads = (plan or {}).get("threads") or []
    n = len(sets)
    l1 = listed_flags(row.get("dx_listed_positions"), row.get("dx_listed_cap")) if n else []
    if n and len(l1) != n:
        l1 = (l1 + [False] * n)[:n]
    r2l = _listed(raw2, sets) if n else {"dx_listed_positions": [], "dx_listed_cap": None}
    l2 = listed_flags(r2l["dx_listed_positions"], r2l["dx_listed_cap"]) if n else []
    bought = _bought(row.get("tool_targets"))
    keys_all = [k for t in threads for k in t.get("key_signals") or []]
    self_bought = [all(k in bought for k in (threads[i].get("key_signals") if i < len(threads) else []) or ())
                   for i in range(n)]
    # A line with no purchasable key signal (MENU-1 gap) says nothing about buying: it is left out
    # of the purchase-weighted mean rather than credited as bought (still in dx_listed_r2).
    w = None
    weighed = [i for i in range(n) if i < len(threads) and threads[i].get("key_signals")]
    if weighed:
        w = math.fsum((1.0 if self_bought[i] else push_weight) if l2[i] else 0.0 for i in weighed) / len(weighed)
    rev = raw2.get("revision") if isinstance(raw2.get("revision"), dict) else {}
    dec = str(rev.get("decision") or "").strip().lower()
    changed = [str(x) for x in (rev.get("changed_fields") or []) if x]
    because = [str(x) for x in (rev.get("because") or []) if x]
    did = distractor_id(plan)
    pt = ((plan or {}).get("distractor") or {}).get("points_to") or {}
    top1 = _top1(raw2)
    top1_dis = bool(pt) and alias_hit_asserted(top1, [pt.get("name", "")] + list(pt.get("aliases") or []))
    dis_in_because = bool(did) and did in because
    cls = "W" if (condition_on_r1 and n and not all(l1)) else "K"
    if not condition_on_r1:
        cls = "W"                                    # mutation: every case "should change"
    if cls == "W":
        missed = [i for i in range(n) if not l1[i]]
        ok = (dec == "change" and "differential" in changed
              and all(l2[i] for i in missed) and not dis_in_because)
    else:
        kept = [i for i in range(n) if l1[i]]
        ok = ((dec == "keep" or not require_keep) and all(l2[i] for i in kept)
              and not top1_dis and not dis_in_because)
    act = raw2.get("action") if isinstance(raw2.get("action"), dict) else {}
    rv = act.get("clinician_review_required")
    menu_hits = 0
    uniq = [t for t in dict.fromkeys(str(x) for x in (row.get("tool_targets") or []) if x)]
    useful = set(keys_all) | {it["fid"] for it in (plan or {}).get("items") or [] if it["role"] == "discrim"}
    from ..gated import DECOY_SIGNALS
    for t in uniq:
        if t not in DECOY_SIGNALS and (set(_findings_of(t)) | {t}) & useful:
            menu_hits += 1
    return {
        "r2_dx_positions": r2l["dx_listed_positions"], "r2_dx_cap": r2l["dx_listed_cap"],
        "r2_self_bought": self_bought or None,
        "dx_listed_r2_w": (round(w, 4) if w is not None else None),
        "dx_listed_r2": (round(sum(l2) / n, 4) if n else None),
        "revision_class": cls, "revision_ok": int(bool(ok)),
        "revision_decision": dec or None,
        "revision_distracted": (int(dis_in_because or top1_dis) if did else None),
        "review_r2": (int(bool(rv) == bool(row.get("r2_warranted"))) if isinstance(rv, bool) else 0),
        "key_acquired": (round(len(set(keys_all) & bought) / len(set(keys_all)), 4) if keys_all else None),
        "buy_efficiency": (round(menu_hits / len(uniq), 4) if uniq else None),
    }


def r1_atoms(row: dict, vp) -> dict:
    """Round-1 purchase atoms of an F1-F3 row: the share of the case's gold-line key signals
    (`core.threads_of`) the solver bought (`tool_targets`, read through the M2 menu rows)."""
    from .core import threads_of
    keys = sorted({k for t in threads_of(((getattr(vp, "adjudication", None) or {}).get("ddx")) or {})
                   for k in t.get("key_signals") or []})
    if not keys:
        return {"m2_key_acquired": None, "m2_key_n": 0}
    got = _bought(row.get("tool_targets"))
    return {"m2_key_acquired": round(len(set(keys) & got) / len(keys), 4), "m2_key_n": len(keys)}


# ------------------------------------------------------------------ M2 test precision (r2-2)
def keysig_on_target(item: str, targets) -> bool:
    """Revision r3-F2: an ordered item names a key-signal menu item only by exact (normalised) name."""
    return str(item or "").strip().lower() in targets


def m2_tests_atoms(raw: dict, vp, cap) -> dict:
    """Revision r2-2 (review m2-impl-r2 B3): on F1-F3 an ordered item equal to the menu item of a
    gold line's key signal (`core.threads_of` -> `core.menu_target_of`; exact match since r3-F2) is
    on target. Required tests (recall) stay `ddx.tests`; the precision numerator adds those items.
    Whether an item is on target under the v1.0.1 rule is read item by item from the production
    judge (`judges.differential.judge_workup` on a one-item order), so the gold resolution and
    matching are the production ones; `m2_tests_on_target_v101` reproduces the row's `tests_precision`."""
    from ..judges.differential import judge_workup
    from .core import menu_target_of, threads_of
    raw = raw if isinstance(raw, dict) else {}
    said_all = [str(t) for t in (raw.get("tests_to_order") or []) if str(t).strip()]
    if not said_all or not isinstance(cap, int):
        return {"m2_tests_precision": None, "m2_tests_keysig_on_target": None}
    said = said_all[:cap]
    mt = menu_target_of()
    ddx = ((getattr(vp, "adjudication", None) or {}).get("ddx")) or {}
    targets = {str(mt[k]).strip().lower() for t in threads_of(ddx) for k in t.get("key_signals") or [] if k in mt}
    on, ks = 0, 0
    cache: dict = {}
    for t in said:
        if t not in cache:
            one = judge_workup(SimpleNamespace(_raw={"tests_to_order": [t]}, action={}), vp, {})
            cache[t] = one.get("tests_precision") == 1.0
        if cache[t]:
            on += 1
        elif keysig_on_target(t, targets):
            ks += 1
    return {"m2_tests_precision": round((on + ks) / len(said), 3),
            "m2_tests_on_target_v101": round(on / len(said), 3), "m2_tests_keysig_on_target": ks}


# ------------------------------------------------------------------ per-case scored atoms
def tests_f1(row: dict, *, m2_precision: bool = True):
    """Tests F1. On F1-F3 rows the precision is `m2_tests_precision` (revision r2-2); F0 and v1.0.1
    rows keep the production `tests_precision`. `m2_precision=False` is the mutation hook."""
    r, p = row.get("tests_recall"), row.get("tests_precision")
    if m2_precision and row.get("m2_frame") in ("F1", "F2", "F3") and isinstance(row.get("m2_tests_precision"), (int, float)):
        p = row["m2_tests_precision"]
    if not isinstance(r, (int, float)):
        return None
    if not isinstance(p, (int, float)):
        p = 0.0
    return 0.0 if (r + p) <= 0 else 2 * r * p / (r + p)


def _aborted(row) -> bool:
    return str(row.get("overall") or "").startswith("ABORT(no_answer")


def _is_insufficient(row) -> bool:
    return row.get("gold_kind") == "ddx:insufficient" or bool(row.get("m2_insufficient"))


def frame_of(row: dict) -> str:
    return str(row.get("m2_frame") or "F0")


def case_dims_raw(row: dict, *, m2_precision: bool = True) -> dict:
    """Scored per-case atoms of one row, uncorrected (None = not applicable on this case)."""
    ins = _is_insufficient(row)
    if _aborted(row):
        # Revision r3-A4' (after the abort reading): a case the solver left unanswered scores 0 on
        # the dims every sufficient case carries, instead of dropping out of the mean.
        return {"tests_recall_f1": None if ins else 0.0, "dx_listed": None if ins else 0.0, "quant_ok": None}
    out = {"tests_recall_f1": None if ins else tests_f1(row, m2_precision=m2_precision),
           "dx_listed": None, "quant_ok": None}
    pos = row.get("dx_listed_positions")
    if not ins and isinstance(pos, list) and pos:
        out["dx_listed"] = row.get("dx_listed")
    if row.get("m2_frame") != "F1" and isinstance(row.get("quant_ok"), (int, float)):
        out["quant_ok"] = float(row["quant_ok"])
    return out


def review_cc(rs: list[dict], *, cc_fn=None, f1_f3_only: bool = True):
    """(cc, BA, m) over the F1-F3 rows (revision r3-A2); every row when the batch has none.
    `f1_f3_only=False` is the mutation hook (F0 pooled in)."""
    cc_fn = cc_fn or cc_raw
    sub = [r for r in rs if r.get("m2_frame") in ("F1", "F2", "F3")] if f1_f3_only else []
    # r3-A4': an unanswered case with a known review gold counts as a wrong review answer
    ba, m = balanced([(bool(r.get("review_warranted")), 0 if _aborted(r) else r.get("review_flag_ok"))
                      for r in (sub or rs) if isinstance(r.get("review_flag_ok"), (int, float))
                      or (_aborted(r) and r.get("review_warranted") is not None)])
    return cc_fn(ba, m), ba, m


def abst_cc(rs: list[dict]):
    pairs = []
    for r in rs:
        if _is_insufficient(r):
            v = r.get("abst_ok")
            if isinstance(v, (int, float)):
                pairs.append(("insufficient", float(v)))
        else:
            v = r.get("abst_over")
            if isinstance(v, (int, float)):
                pairs.append(("sufficient", 1.0 - float(v)))
    ba, m = balanced(pairs)
    return cc_raw(ba, m), ba, m


def gated(comp, mult):
    """Composite after the hard-gate multiplier (revision r4): max(0, c) * mult."""
    return None if comp is None else max(0.0, comp) * mult


def solver_scores(rs: list[dict], *, cc_fn=cc_raw, gate: bool = True,
                  scored_solver_dims: tuple = SCORED_SOLVER_DIMS, review_f1_f3_only: bool = True,
                  m2_precision: bool = True, gate_fn=None) -> dict:
    """All M2 dims and the composite for one solver's rows. `scored_solver_dims` is the mutation
    hook of R-16 (passing abst_utility_cc back in must move the composite); `review_f1_f3_only`,
    `m2_precision` and `gate_fn` are mutation hooks."""
    rc, rba, rm = review_cc(rs, cc_fn=cc_fn, f1_f3_only=review_f1_f3_only)
    ac, aba, am = abst_cc(rs)
    if cc_fn is not cc_raw:                          # mutation hook: alternative correction
        ac = cc_fn(aba, am)
    all_solver = {"review_utility_cc": rc, "abst_utility_cc": ac}
    solver_dims = {d: all_solver[d] for d in scored_solver_dims}
    per_case = []
    dims_acc: dict[str, list] = {d: [] for d in SCORED_CASE_DIMS}
    for r in rs:
        cd = case_dims_raw(r, m2_precision=m2_precision)
        for d, v in cd.items():
            if v is not None:
                dims_acc[d].append(v)
        vals = [v for v in cd.values() if v is not None] + [v for v in solver_dims.values() if v is not None]
        if vals:
            per_case.append(math.fsum(vals) / len(vals))
    comp = math.fsum(per_case) / len(per_case) if per_case else None
    mult = 1.0
    if gate:
        from ..analytics import gate_multiplier
        mult = float(gate_multiplier(rs).get("mult") or 1.0) if rs else 1.0
    out = {d: _mean(v) for d, v in dims_acc.items()}
    out.update(all_solver)
    out["review_f0_sens"] = _mean([r.get("review_flag_ok") for r in rs if frame_of(r) == "F0"])
    out["composite_raw"] = (gate_fn or gated)(comp, mult)
    out["composite"] = out["composite_raw"]
    out["gate_mult"] = mult
    out["n_cases"] = len(rs)
    # profile
    r2 = [r for r in rs if r.get("r2_status") == "SCORED"]
    out["revision_acc"] = _mean([r.get("revision_ok") for r in r2])
    by = {c: [r.get("revision_ok") for r in r2 if r.get("revision_class") == c] for c in ("W", "K")}
    if all(len(v) >= MIN_PER_CLASS for v in by.values()):
        ba, m = balanced([(c, x) for c, v in by.items() for x in v])
        out["revision_ba_cc"] = cc_fn(ba, m)
    else:
        out["revision_ba_cc"] = None
    out["revision_n_by_class"] = {c: len(v) for c, v in by.items()}
    for d in ("dx_listed_r2", "dx_listed_r2_w", "buy_efficiency", "review_r2", "revision_distracted"):
        out[d] = _mean([r.get(d) for r in r2])
    out["key_acquired"] = _mean([r.get("m2_key_acquired") for r in rs])
    out["dx_listed_n0"] = _mean([r.get("dx_listed_n0") for r in rs if not _is_insufficient(r)])
    out["noop_ok"] = _mean([r.get("noop_ok") for r in rs])
    out["tool_grounded_joint"] = _mean([r.get("tool_grounded_joint") for r in rs])
    ub, um = balanced([(r.get("m2_urgency_gold"), r.get("m2_urgency_ok")) for r in rs
                       if r.get("m2_frame") == "F1"])
    out["urgency_cc"] = cc_fn(ub, um)
    mb, mm = balanced([(r.get("m2_med_gold"), r.get("m2_med_ok")) for r in rs if r.get("m2_frame") == "F2"])
    out["med_direction_cc"] = cc_fn(mb, mm)
    out["rcv_acc"] = _mean([r.get("m2_rcv_ok") for r in rs if r.get("m2_frame") == "F3"])
    # F3 gold is imbalanced (most follow-up changes stay inside the RCV), so the raw accuracy
    # rewards a constant "within RCV"; the chance-corrected BA is the profile reading.
    vb, vm = balanced([(r.get("m2_rcv_gold"), r.get("m2_rcv_ok")) for r in rs
                       if r.get("m2_frame") == "F3" and r.get("m2_rcv_gold") is not None])
    out["rcv_cc"] = cc_fn(vb, vm)
    return out


def board(rows: list[dict], **kw) -> dict[str, dict]:
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["solver"], []).append(r)
    return {s: solver_scores(rs, **kw) for s, rs in sorted(by.items())}


def composite_scores(rows: list[dict]) -> dict[str, float]:
    """`score_fn` for the production bootstrap (`semantic_report.bootstrap_scores`)."""
    return {s: v["composite_raw"] for s, v in board(rows).items()}
