"""Question-blind optimal-constant ceilings for the core dimensions.

For each core dimension: the best score an answer that never looks at the question can reach,
computed from this batch's gold (solver-independent, one entry per case). The acceptance rule
is `check_caps`: every core ceiling must sit at or below 0.5.

Result shape: `{dim: (value, description_of_the_constant_strategy)}` plus `n_cases` and a few
non-tuple side entries (`n_by_dim`, `quant_by_kind`, `notes`). Greedy-computed dims are lower
bounds of the true optimum and say so in their description.

SYNTHETIC data, evaluation only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

#: Dimensions the design's acceptance rule ("ceiling <= 0.5") applies to: the scored core dims.
#: `noop_ok` and `tool_grounded_joint` are reported, not scored (M1 amendment 2026-10-01), and
#: their question-blind readings are printed without an acceptance line.
CORE_DIMS = ("quant_ok", "review_utility_cc", "review_macro", "dx_listed", "tests_f1")

#: Reported, not scored: printed with the board, outside `check_caps`.
REPORT_ONLY_DIMS = ("noop_ok", "tool_grounded_joint", "abst_sens", "review_utility", "abst_utility", "dx_listed_n0",
                    "abst_utility_cc")

#: Seed and draw count of the random-menu blind stub (fixed so the printed reading is reproducible).
TOOL_STUB_SEED = 20261001
TOOL_STUB_DRAWS = 30

_EPS = 1e-9


# ================================================================ helpers
def _by_case(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:                       # gold is solver-independent
        out.setdefault(r.get("case"), r)
    return out


def _norm_item(s) -> str:
    return " ".join(str(s or "").strip().lower().split())


def _f1(r: float, p: float) -> float:
    return 0.0 if r + p <= 0 else 2 * r * p / (r + p)


# ================================================================ noop
def _noop(cases: dict[str, dict], gold: dict, rows: list[dict] | None = None) -> tuple | None:
    """On the gated geometry the delivered window depends on what the model bought, so the
    delivered truth is per cell: the constant is then scored over every cell."""
    truth = []
    per_cell = [r for r in (rows or []) if r.get("noop_truth_delivered") is not None]
    if per_cell:
        truth = [bool(r["noop_truth_delivered"]) for r in per_cell]
        p = sum(truth) / len(truth)
        side = ("present", sum(truth)) if p >= 0.5 else ("no_data_in_window", len(truth) - sum(truth))
        return (round(max(p, 1 - p), 3),
                f"always answers `{side[0]}` (delivered-window truth, per cell: {side[1]}/{len(truth)} cells)",
                len(truth))
    for cid, r in cases.items():
        g = (gold.get(cid) or {})
        d = g.get("noop_truth_delivered", r.get("noop_truth_delivered"))
        v = d if d is not None else g.get("noop_truth_present", r.get("noop_truth_present"))
        if v is not None:
            truth.append(bool(v))
    if not truth:
        return None
    p = sum(truth) / len(truth)
    if p >= 0.5:
        return round(p, 3), f"always answers `present` (truth present on {sum(truth)}/{len(truth)} cases)", len(truth)
    return round(1 - p, 3), (f"always answers `no_data_in_window` (truth absent on "
                             f"{len(truth) - sum(truth)}/{len(truth)} cases)"), len(truth)


# ================================================================ quant
def _norm_ans(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return round(float(s), 6)
        except ValueError:
            return s
    return v


def _quant(cases: dict[str, dict], gold: dict) -> tuple | None:
    """Per quant kind the modal accepted answer (a tied peak day accepts every tied day)."""
    per_kind: dict[str, list[set]] = {}
    for cid, r in cases.items():
        g = (gold.get(cid) or {})
        kind = g.get("quant_kind", r.get("quant_kind"))
        truth = g.get("quant_truth", r.get("quant_truth"))
        if kind in (None, "") or truth is None:
            continue
        acc = g.get("quant_accepted") or r.get("quant_peak_days") or [truth]
        per_kind.setdefault(str(kind), []).append({_norm_ans(a) for a in acc})
    if not per_kind:
        return None
    hits = tot = 0
    by_kind: dict[str, dict] = {}
    for kind, accs in sorted(per_kind.items()):
        cnt: dict = {}
        for s in accs:
            for a in s:
                cnt[a] = cnt.get(a, 0) + 1
        best_a, best_n = max(cnt.items(), key=lambda kv: (kv[1], str(kv[0])))
        by_kind[kind] = {"answer": best_a, "hits": best_n, "n": len(accs),
                         "share": round(best_n / len(accs), 3)}
        hits += best_n
        tot += len(accs)
    desc = "; ".join(f"{k}: always `{v['answer']}` ({v['hits']}/{v['n']})" for k, v in by_kind.items())
    return round(hits / tot, 3), "per-kind modal truth answer -> " + desc, tot, by_kind


# ================================================================ review
def _review(cases: dict[str, dict], gold: dict) -> dict:
    """Sensitivity/specificity of the constants "always request" / "never request".

    `review_utility` = mean of the defined class rates; `review_macro` follows
    `analytics.review_macro_of` (specificity, no value with a single class).
    """
    warr = []
    for cid, r in cases.items():
        g = gold.get(cid) or {}
        w = g.get("review_warranted", r.get("review_warranted"))
        if w is None or not isinstance(r.get("review_flag_ok"), (int, float)):
            continue
        warr.append(bool(w))
    out: dict = {}
    if not warr:
        return out
    has_pos, has_neg = any(warr), not all(warr)
    classes = int(has_pos) + int(has_neg)
    always = ((1.0 if has_pos else 0.0) + (0.0 if has_neg else 0.0)) / classes
    never = ((0.0 if has_pos else 0.0) + (1.0 if has_neg else 0.0)) / classes
    out["review_utility"] = (round(max(always, never), 3),
                             f"always or never requests review: (sensitivity+specificity)/2 = "
                             f"{always:.3f} / {never:.3f} ({sum(warr)} warranted, "
                             f"{len(warr) - sum(warr)} not warranted)", len(warr))
    out["review_utility_cc"] = (round(max(0.0, 2 * max(always, never) - 1), 3),
                                "always or never requests review: max(0, 2 x BA - 1) (A3)", len(warr))
    if has_pos and has_neg:
        out["review_macro"] = (1.0, "never requests review: specificity = 1.000 "
                                    "(paid for by the review gates on warranted slices)", len(warr))
    return out


# ================================================================ abstention
def abst_stub_rows(rows: list[dict], always: bool) -> list[dict]:
    """One row per case with the abstention fields a constant "always" / "never abstain" answer
    would get from the production judges (gated: `abst_ok = abstained != sufficient`,
    `abst_over = sufficient and abstained`; slices: shares over the slices, every slice of the
    insufficient tier has no true symptom). Other fields are copied from the case's first row."""
    from .analytics import INSUFFICIENT_KIND
    out = []
    for cid, r in _by_case(rows).items():
        k = r.get("gold_kind")
        if not str(k or "").startswith("ddx:"):
            continue
        s = {x: r.get(x) for x in ("case", "gold_kind", "geometry")}
        s["solver"] = "stub-always-abstain" if always else "stub-never-abstain"
        if k == INSUFFICIENT_KIND:
            s["abst_ok"] = 1.0 if always else 0.0
        elif isinstance(r.get("abst_over"), (int, float)) and not isinstance(r.get("abst_over"), bool):
            s["abst_over"] = 1.0 if always else 0.0
        out.append(s)
    return out


def _abst(rows: list[dict]) -> dict:
    """"Always abstain" / "never abstain" on `abst_utility` (two-sided, scored) and on the
    rejected one-sided tier-only mean `abst_sens` (reported: "always abstain" is perfect there),
    both read through the production aggregator `analytics.abstention_utility_of`."""
    from .analytics import abstention_utility_of
    a = abstention_utility_of(abst_stub_rows(rows, True))
    n = abstention_utility_of(abst_stub_rows(rows, False))
    if a.get("abst_utility") is None or n.get("abst_utility") is None:
        return {}
    nb = a["abst_n_by_side"]
    return {"abst_utility": (round(max(a["abst_utility"], n["abst_utility"]), 3),
                             f"always or never abstains: (sensitivity+specificity)/2 = "
                             f"{a['abst_utility']:.3f} / {n['abst_utility']:.3f} "
                             f"({nb['insufficient']} insufficient-tier, {nb['other']} other cases)",
                             nb["insufficient"] + nb["other"]),
            "abst_utility_cc": (round(max(a["abst_utility_cc"], n["abst_utility_cc"]), 3),
                                "always or never abstains: max(0, 2 x BA - 1) (A3)",
                                nb["insufficient"] + nb["other"]),
            "abst_sens": (round(max(a["abst_sens"], n["abst_sens"]), 3),
                          f"always abstains: the one-sided tier-only mean = {a['abst_sens']:.3f} "
                          f"({nb['insufficient']} cases); reported, not scored", nb["insufficient"])}


# ================================================================ dx_listed
def _dx_hits(lines_by_case: dict[str, list[list[str]]], match: Callable, canonical_only: bool = False) -> tuple[list[str], list[list[list[bool]]], list[str]]:
    cands: list[str] = []            # every alias string of every gold line is a legal entry to write
    for ls in lines_by_case.values():
        for s in ls:
            for a in (s[:1] if canonical_only else s):
                if str(a).strip() and str(a) not in cands:
                    cands.append(str(a))
    cids = list(lines_by_case)
    # hit[cid][line][cand]
    hit = [[[bool(match(c, s)) for c in cands] for s in lines_by_case[cid]] for cid in cids]
    return cands, hit, cids


def _dx_score(order: list[int], hit, lines_n: list[int], margin: int) -> float:
    tot = 0.0
    for ci, hs in enumerate(hit):
        n = lines_n[ci]
        cap = n + margin
        found = 0
        for line in hs:
            pos = next((p for p, k in enumerate(order, 1) if line[k]), None)
            if pos is not None and pos <= cap:
                found += 1
        tot += found / n
    return tot / len(hit)


def dx_listed_ceiling(lines_by_case: dict[str, list[list[str]]], margin: int = 3,
                      match: Callable | None = None) -> tuple[float, list[str]]:
    """Greedy best constant ranked differential. Position p counts for a line only when
    p <= n_lines + margin. Lower bound of the optimum."""
    if match is None:
        from .tracks import alias_hit_asserted as match      # the production matcher
    lines_by_case = {c: ls for c, ls in lines_by_case.items() if ls}
    if not lines_by_case:
        return 0.0, []
    cands, hit, cids = _dx_hits(lines_by_case, match)
    lines_n = [len(lines_by_case[c]) for c in cids]
    order: list[int] = []
    best = 0.0
    while True:
        gain_k, gain_v = None, best + _EPS
        for k in range(len(cands)):
            if k in order:
                continue
            v = _dx_score(order + [k], hit, lines_n, margin)
            if v > gain_v:
                gain_k, gain_v = k, v
        if gain_k is None:
            break
        order.append(gain_k)
        best = gain_v
    # local search: replace one position at a time while it improves
    improved = True
    while improved and order:
        improved = False
        for i in range(len(order)):
            for k in range(len(cands)):
                if k in order:
                    continue
                v = _dx_score(order[:i] + [k] + order[i + 1:], hit, lines_n, margin)
                if v > best + _EPS:
                    order[i], best, improved = k, v, True
    return round(best, 3), [cands[k] for k in order]


def dx_listed_top_k_freq(lines_by_case: dict[str, list[list[str]]], k: int = 5, margin: int = 3,
                         match: Callable | None = None) -> tuple[float, list[str]]:
    """The k candidates that hit the most lines on their own, in that order (the design's
    hand-computed "top-5 constant")."""
    if match is None:
        from .tracks import alias_hit_asserted as match
    lines_by_case = {c: ls for c, ls in lines_by_case.items() if ls}
    if not lines_by_case:
        return 0.0, []
    cands, hit, cids = _dx_hits(lines_by_case, match, canonical_only=True)
    lines_n = [len(lines_by_case[c]) for c in cids]
    freq = [sum(line[j] for hs in hit for line in hs) for j in range(len(cands))]
    order = sorted(range(len(cands)), key=lambda j: (-freq[j], cands[j]))[:k]
    return round(_dx_score(order, hit, lines_n, margin), 3), [cands[j] for j in order]


# ================================================================ tests F1
def _tests_sets(gold: dict) -> dict[str, tuple[set, set]]:
    out = {}
    for cid, g in gold.items():
        req = {_norm_item(t) for t in (g.get("tests_required") or []) if _norm_item(t)}
        opt = {_norm_item(t) for t in (g.get("tests_optional") or []) if _norm_item(t)} - req
        if req:                                   # no scoreable gold item -> recall None -> cell not scored
            out[cid] = (req, opt)
    return out


def _f1_of(S: list[str], req: set, opt: set, margin: int = 3) -> float:
    cap = len(req) + margin                       # TESTS_CAP_MARGIN: only the first n+3 ordered tests count
    said = S[:cap]
    if not said:
        return 0.0
    rec = len([s for s in said if s in req]) / len(req)
    prec = len([s for s in said if s in req or s in opt]) / len(said)
    return _f1(rec, prec)


def tests_f1_ceiling(tests_by_case: dict[str, tuple[set, set]], max_iter: int = 60) -> tuple[float, list[str]]:
    """Greedy (add, then drop/swap moves) best constant test list under item identity.
    Lower bound of the optimum; an item-identity approximation of the code matcher."""
    if not tests_by_case:
        return 0.0, []
    cands = sorted({t for r, o in tests_by_case.values() for t in (r | o)})
    cases = list(tests_by_case.values())

    def score(S):
        return sum(_f1_of(S, r, o) for r, o in cases) / len(cases)

    S: list[str] = []
    best = 0.0
    for _ in range(max_iter):
        move = None
        mv = best + _EPS
        for t in cands:                            # add at the end
            if t in S:
                continue
            v = score(S + [t])
            if v > mv:
                move, mv = ("add", t), v
        for i in range(len(S)):                    # drop
            v = score(S[:i] + S[i + 1:])
            if v > mv:
                move, mv = ("drop", i), v
        if move is None:
            break
        S = S + [move[1]] if move[0] == "add" else S[:move[1]] + S[move[1] + 1:]
        best = mv
    return round(best, 3), S


def tests_f1_matcher_ceiling(vps: dict, max_iter: int = 40) -> tuple[float, list[str]] | None:
    """Same greedy, scored through the production `judge_workup` matcher (segment forms,
    one-to-one, n+3 cap). `vps` maps case id to the case object. Candidates are the gold test
    items; cases with identical gold test lists are scored once and weighted."""
    from types import SimpleNamespace
    from .judges.differential import judge_workup
    from .judges._helpers import _gold
    return _tests_f1_matcher_search(vps, judge_workup, SimpleNamespace, _gold, max_iter)


def _tests_f1_matcher_search(vps, judge_workup, SimpleNamespace, _gold, max_iter):
    groups: dict[tuple, list] = {}
    for cid, vp in vps.items():
        key = tuple(str(t) for t in (_gold(vp, "tests") or []) if str(t).strip())
        if key:
            groups.setdefault(key, []).append(vp)
    if not groups:
        return None

    def f1_group(S, vp):
        out = SimpleNamespace(_raw={"tests_to_order": list(S)}, action={})
        res = judge_workup(out, vp, None)
        r, p = res.get("tests_recall"), res.get("tests_precision")
        if r is None:
            return None
        return _f1(r, p or 0.0)

    reps = [(vs[0], len(vs)) for vs in groups.values()]
    memo: dict = {}

    def score(S):
        k = tuple(S)
        if k not in memo:
            tot = w = 0.0
            for vp, m in reps:
                f = f1_group(S, vp)
                if f is not None:
                    tot += f * m
                    w += m
            memo[k] = tot / w if w else 0.0
        return memo[k]

    cands = sorted({t for key in groups for t in key})
    S: list[str] = []
    best = 0.0
    for _ in range(max_iter):
        move, mv = None, best + _EPS
        for t in cands:
            if t in S:
                continue
            v = score(S + [t])
            if v > mv:
                move, mv = ("add", t), v
        for i in range(len(S)):
            v = score(S[:i] + S[i + 1:])
            if v > mv:
                move, mv = ("drop", i), v
        if move is None:
            break
        S = S + [move[1]] if move[0] == "add" else S[:move[1]] + S[move[1] + 1:]
        best = mv
    return round(best, 3), S


# ================================================================ tool blind stubs
def _stub_trace(picks: list[dict], budget: float, menu: list[dict] | None = None):
    """A gated trace for a solver that bought `picks` (menu items) and never read the question.
    `grounded` follows the menu's `real` flag (None for test orders, which `tool_track` leaves
    out of the grounding denominator)."""
    from types import SimpleNamespace
    calls = [SimpleNamespace(target=str(i["target"]), grounded=i.get("real"), cost=float(i["cost"]))
             for i in picks]
    return SimpleNamespace(calls=calls, spent=sum(c.cost for c in calls), budget=float(budget),
                           truncated=0, rounds_used=1, committed=True, protocol_errors=0,
                           kind_misreports=0, menu=list(menu or []))


def _fill_to_budget(order: list[dict], budget: float) -> list[dict]:
    """Buy in `order`, skipping what no longer fits, until the budget is used up."""
    spent, picks = 0.0, []
    for i in order:
        c = float(i["cost"])
        if spent + c <= budget:
            spent += c
            picks.append(i)
    return picks


def _stub_joint(picks: list[dict], budget: float, key_signals, menu: list[dict] | None = None) -> float | None:
    """`tool_grounded_joint` of the stub, through the production chain `tracks.tool_track` ->
    `quantities.tool_grounded_joint_of`."""
    from .quantities import tool_grounded_joint_of
    from .tracks import tool_track
    return tool_grounded_joint_of(tool_track(_stub_trace(picks, budget, menu), key_signals=key_signals))


def tool_blind_stubs(menus: dict[str, tuple[list[dict], float, tuple]],
                     draws: int = TOOL_STUB_DRAWS, seed: int = TOOL_STUB_SEED) -> dict:
    """Question-blind stubs for the tool dimension, `{name: (value, description, n_cells)}`.

    `menus` maps case id to `(menu, budget, key_signals)` as the gated run hands them out
    (`evaluate.gated_menu`, `gated.budget_for`, `tracks.key_signals_for`).
      - `tool_stub_cheapest_first`: buys from the cheapest item up until the budget is used up;
      - `tool_stub_random_menu`: buys in a random order over the whole menu (mean over `draws`
        seeded shuffles per case).
    Cells where `tool_grounded_joint` has no value (an item with no key signal and no signal
    bought) are dropped from the mean.
    """
    import random
    import statistics
    rng = random.Random(seed)
    cheap: list[float] = []
    rand: list[float] = []
    for cid in sorted(menus):
        menu, budget, ks = menus[cid]
        if not menu:
            continue
        v = _stub_joint(_fill_to_budget(sorted(menu, key=lambda i: float(i["cost"])), budget), budget, ks, menu)
        if v is not None:
            cheap.append(v)
        vs = []
        for _ in range(draws):
            order = list(menu)
            rng.shuffle(order)
            x = _stub_joint(_fill_to_budget(order, budget), budget, ks, menu)
            if x is not None:
                vs.append(x)
        if vs:
            rand.append(statistics.fmean(vs))
    out: dict = {}
    if cheap:
        out["tool_stub_cheapest_first"] = (
            round(statistics.fmean(cheap), 3),
            "buys from the cheapest menu item up until the budget is used up, never reading the question "
            "(reported dimension `tool_grounded_joint`)", len(cheap))
    if rand:
        out["tool_stub_random_menu"] = (
            round(statistics.fmean(rand), 3),
            f"buys in a random order over the whole menu until the budget is used up "
            f"(mean over {draws} seeded draws per case; `tool_grounded_joint`)", len(rand))
    return out


def gated_menus_from_cases(cases: dict) -> dict[str, tuple[list[dict], float, tuple]]:
    """`{case_id: (menu, budget, key_signals)}` for `tool_blind_stubs`, rebuilt with the
    production calls the gated run uses (`evaluate.gated_menu`, `gated.budget_for`,
    `tracks.key_signals_for`). `cases` is `store.load_cases` output."""
    from haenv_kernel.build import build_instance
    from .rows import gated_menu
    from .gated import budget_for
    from .tracks import key_signals_for
    out = {}
    for cid, raw in cases.items():
        T = int(raw.prediction_context["prediction_time_T"])
        menu = gated_menu(raw, T)
        _, vp = build_instance(raw, T)
        out[cid] = (menu, budget_for(menu), tuple(key_signals_for(vp)))
    return out


# ================================================================ entry points
def core_constant_ceilings(rows: list[dict], gold: dict[str, dict] | None = None) -> dict:
    """Per core dimension, `(value, strategy)`: the best score a question-blind constant gets.

    `gold` = `{case_id: {"diagnosis_lines": [[aliases], ...], "tests_required": [...],
    "tests_optional": [...], optionally "noop_truth_present"/"noop_truth_delivered",
    "quant_kind"/"quant_truth"/"quant_accepted", "review_warranted", "_vp": case object}}`.
    Row fields fill in whatever `gold` does not carry.
    """
    gold = gold or {}
    cases = _by_case(rows)
    out: dict = {"n_cases": len(cases)}
    n_by: dict = {}
    notes: list[str] = []
    if not cases and not gold:
        return out

    r = _noop(cases, gold, rows)
    if r:
        out["noop_ok"] = r[:2]
        n_by["noop_ok"] = r[2]
    r = _quant(cases, gold)
    if r:
        out["quant_ok"] = r[:2]
        n_by["quant_ok"] = r[2]
        out["quant_by_kind"] = r[3]
    for k, v in _review(cases, gold).items():
        out[k] = v[:2]
        n_by[k] = v[2]
    for k, v in _abst(rows).items():
        out[k] = v[:2]
        n_by[k] = v[2]
    if "review_macro" not in out and "review_utility" in out:
        notes.append("review_macro: single gold class in this batch, no value (analytics.review_macro_of)")

    lines = {cid: g["diagnosis_lines"] for cid, g in gold.items() if g.get("diagnosis_lines")}
    if lines:
        from .judges.differential import DX_LIST_CAP_MARGIN as _M
        vm, lm = dx_listed_ceiling(lines, _M)
        out["dx_listed"] = (vm, f"greedy lower bound of the optimum: one fixed ranked list of {len(lm)} "
                                f"names, cap n+{_M} (scored), {len(lines)} cases; list={lm}")
        v3, l3 = dx_listed_ceiling(lines, 3)
        out["dx_listed_cap_n_plus_3"] = (v3, f"greedy lower bound of the optimum under the pre-A6 cap n+3; list={l3}")
        v0, l0 = dx_listed_ceiling(lines, 0)
        out["dx_listed_n0"] = (v0, f"greedy lower bound under cap n+0 (profile, not scored); list={l0}")
        v5, l5 = dx_listed_top_k_freq(lines, 5, 3)
        out["dx_listed_top5_freq"] = (v5, f"the 5 most frequent gold diagnoses, cap n+3; list={l5}")
        n_by["dx_listed"] = len(lines)
    ts = _tests_sets(gold)
    if ts:
        v, S = tests_f1_ceiling(ts)
        out["tests_f1"] = (v, f"greedy lower bound of the optimum: one fixed test list of {len(S)} items "
                              f"(item-identity approximation of the code matcher; n+3 cap; "
                              f"the one-long-string exploit is a separate stub); list={S}")
        n_by["tests_f1"] = len(ts)
        vps = {cid: g["_vp"] for cid, g in gold.items() if g.get("_vp") is not None}
        if vps:
            m = tests_f1_matcher_ceiling(vps)
            if m:
                out["tests_f1_matcher"] = (m[0], f"same greedy scored through the production judge_workup "
                                                 f"matcher (segment forms, one-to-one); list={m[1]}")
    out["tool_grounded_joint"] = (0.0, "never queries: a constant that calls no tool scores 0 by construction "
                                       "(gated geometry only; reported, not scored)")
    menus = {cid: g["_menu"] for cid, g in gold.items() if g.get("_menu") and cid in cases}
    if menus and any(r.get("tool_budget") is not None for r in rows):
        for k, v in tool_blind_stubs(menus).items():
            out[k] = v[:2]
            n_by[k] = v[2]
    out["n_by_dim"] = n_by
    if notes:
        out["notes"] = notes
    return out


def check_caps(ceilings: dict, cap: float = 0.5) -> dict:
    """Core dimensions whose ceiling exceeds `cap`: `{dim: value}` (empty means acceptance holds)."""
    bad = {}
    for d in CORE_DIMS:
        v = ceilings.get(d)
        if isinstance(v, tuple) and isinstance(v[0], (int, float)) and v[0] > cap + _EPS:
            bad[d] = v[0]
    return bad


def gold_from_cases_jsonl(path, with_menus: bool = False) -> dict[str, dict]:
    """`{case_id: gold}` for `core_constant_ceilings`, read through the production accessors
    (`store.load_cases`, `judges._helpers._gold/_names/gold_kind`, `differential._thread_sets`,
    `_gold_test_segs`/`_gold_test_optional`). Which cases carry diagnosis lines and tests follows
    the judge mounting: dx on unified/comorbidity, tests on unified/comorbidity/independent.
    `with_menus=True` also attaches each case's gated `(menu, budget, key_signals)` (`_menu`) for
    the tool blind stubs."""
    from .store import load_cases
    from .judges import _helpers as H
    from .judges import differential as D
    cs = load_cases(Path(path))
    out: dict[str, dict] = {}
    for cid, w in cs.items():
        kind = H.gold_kind(w)
        g: dict = {"gold_kind": kind, "_vp": w}
        if kind == "ddx:unified":
            g["diagnosis_lines"] = [list(H._names(w))]
        elif kind == "ddx:comorbidity":
            g["diagnosis_lines"] = [list(s) for s in D._thread_sets(w)[0]]
        if kind in ("ddx:unified", "ddx:comorbidity", "ddx:independent"):
            tests = [str(t) for t in (H._gold(w, "tests") or []) if str(t).strip()]
            g["tests_required"] = [t for t in tests
                                   if D._gold_test_segs(t) and not D._gold_test_optional(t)]
            g["tests_optional"] = [t for t in tests
                                   if D._gold_test_segs(t) and D._gold_test_optional(t)]
        out[cid] = g
    if with_menus:
        for cid, m in gated_menus_from_cases(cs).items():
            out[cid]["_menu"] = m
    return out


def load_rows(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
