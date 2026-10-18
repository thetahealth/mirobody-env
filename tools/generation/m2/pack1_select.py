"""pack1_select.py -- the selection rule of pack 1 (differential diagnosis): which pool cases a pack of
N items takes. Imported by `tools/generation/m2_gen_job.py`. It reads the prior tier, the arm, the spec id, the
sex and whether the hidden line carries an explain-away rendering (gold fields; never a score):

  hard      F1-F3 positive-arm hard, apportioned over F1/F2/F3 by their hard counts (largest
            remainder); within a frame sha256(case_id + seed) order, one case per spec id first, then
            the explain-away twin of a held spec (the other side of `ea_twin`), then any spec under
            the cap
  Spec cap: a spec id fills at most `spec_cap(N)` items of the pack (2 per 50); a cell the cap
  leaves short is filled in sha256 order and the overflow is reported by `balance`.
  mid/easy  F1-F3 positive, F0 bridge non-insufficient and negative arm, each in sha256 order
  Negative arm: apportioned over F1/F2/F3 by the frames' positive counts (equal review share per
  frame); easy over the frames that have easy negatives; within the frame a female quota =
  round(quota x the frame's positive female share); spec ids not repeated.
  Balance gate: BAL-1 every frame both labels and >= 2 tiers; BAL-2 frame-only Youden J <= 0.05;
  BAL-3 (expanded) leave-one-out frame x sex review prior cc <= 0.05.

The cell sizes come from `haenv/m2/pack.py:quotas`.
Nesting: `select(..., base=...)` extends the pick of a smaller pack of the same pool: every base case
stays in its cell and is counted first, and the cell is topped up by the same rules (`haenv/world/validation/pack_size.py`
NEST_SIZES; `tools/generation/m2_gen_job.py` chains the legacy sizes). Expanded selection instead solves
the shared nested sizes largest-first, before committing any smaller base.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)

import hashlib
from collections import Counter

BAL2_MAX = 0.05
#: Expanded mode: chance-corrected BA of the leave-one-out frame x sex majority prior on the review
#: gold, F1/F3 pooled (the Z-3 stub `G_framesex_loo`; same ceiling).
BAL3_MAX = 0.05
REVIEW_FRAMES = ("F1", "F3")
FRAMES = ("F1", "F2", "F3")
#: items one spec id may fill per 50 pack items
SPEC_CAP_PER_50 = 2


def spec_cap(n_items: int) -> int:
    return max(SPEC_CAP_PER_50, -(-SPEC_CAP_PER_50 * n_items // 50))


def key(cid: str, seed: str = "") -> str:
    import haenv.world.validation.pack_size as PS
    return hashlib.sha256(PS.seeded(cid, seed).encode("utf-8")).hexdigest()


def apportion(n: int, counts: dict[str, int]) -> dict[str, int]:
    """Largest-remainder apportionment of n over counts (ties to the larger count, then name)."""
    tot = sum(counts.values())
    if tot == 0 or n == 0:
        return {f: 0 for f in counts}
    exact = {f: n * c / tot for f, c in counts.items()}
    out = {f: int(exact[f]) for f in counts}
    rest = n - sum(out.values())
    for f in sorted(counts, key=lambda f: (-(exact[f] - out[f]), -counts[f], f))[:rest]:
        out[f] += 1
    return out


def rows_of(job: dict) -> list[dict]:
    from haenv.m2.core import components
    prior = job["_provenance"]["m2_prior"]
    out = []
    for c in job["cases"]:
        cid = c["case_id"]
        lat = c.get("latent") or {}
        m2 = lat.get("m2") or {}
        frame = prior[cid]["frame"]
        tier = (m2.get("prior") or {}).get("tier") if frame != "F0" else prior[cid]["tier"]
        out.append({"case": cid, "frame": frame, "tier": tier,
                    "arm": m2.get("arm") if frame != "F0" else "bridge",
                    "insufficient": bool(lat.get("ddx_insufficient")),
                    "ea": bool((m2.get("explain_away") or {}).get("lines")), "ea_twin": m2.get("ea_twin"),
                    "ea_lines": tuple(m2.get("ea_twin_lines") or ()),
                    "spec": lat.get("ddx_spec_id"), "sex": (c.get("raw") or {}).get("sex"),
                    "lines": (sorted(components(str(lat.get("ddx_spec_id") or "")))
                              if lat.get("ddx_join_gold") in ("unified", "comorbidity") else [])})
    return out


def _dedup_take(pool, n, held_specs, seed="", count=None, cap=None, pre=()):
    """sha256 order: first one case per spec id not in `held_specs` (a case whose explain-away side
    would put its hidden condition's explained/unexplained count out of level waits); then, for a spec already taken, the
    case on the other side of its explain-away twin; then any case whose spec is under `cap` (pack-wide
    `count`); then (availability) the rest. `count` is updated. `pre` (cases of a smaller nested pack in
    this cell) are taken first."""
    count = count if count is not None else Counter()
    order = sorted(pool, key=lambda r: key(r["case"], seed))
    got, held = [], set(held_specs)
    sides: dict = {}
    side_n: Counter = Counter()        # (hidden condition, twin side) taken in this cell

    def take(r):
        got.append(r)
        held.add(r["spec"])
        count[r["spec"]] += 1
        sides.setdefault(r["spec"], set()).add(r.get("ea_twin"))
        side_n.update((ln, r.get("ea_twin")) for ln in r.get("ea_lines") or ())

    def lean(r):                       # taking r keeps each of its conditions' twin sides level
        other = {"explained": "unexplained", "unexplained": "explained"}.get(r.get("ea_twin"))
        return not other or all(side_n[(ln, r["ea_twin"])] <= side_n[(ln, other)] for ln in r.get("ea_lines") or ())
    pre = [r for r in pool if r["case"] in {x["case"] for x in pre}]
    if len(pre) > n:
        raise SystemExit(f"[m2 select] nested pack: {len(pre)} base cases in a cell of {n}")
    for r in pre:
        take(r)
    under = (lambda r: cap is None or count[r["spec"]] < cap)
    passes = (lambda r: r["spec"] not in held and under(r) and lean(r),
              lambda r: r["spec"] not in held and under(r),
              lambda r: r.get("ea_twin") and r["spec"] in sides and r["ea_twin"] not in sides[r["spec"]] and under(r),
              under, lambda r: True)
    for ok in passes:
        for r in order:
            if len(got) == n:
                return got
            if r not in got and ok(r):
                take(r)
    return got


def cells(rows: list[dict]) -> dict:
    """design cell -> eligible pool rows."""
    pos = [r for r in rows if r["arm"] == "positive"]
    neg = [r for r in rows if r["arm"] == "negative"]
    f0 = [r for r in rows if r["frame"] == "F0" and not r["insufficient"]]
    return {"hard": [r for r in pos if r["tier"] == "hard"],
            "pos_mid": [r for r in pos if r["tier"] == "mid"], "pos_easy": [r for r in pos if r["tier"] == "easy"],
            "f0_mid": [r for r in f0 if r["tier"] == "mid"], "f0_easy": [r for r in f0 if r["tier"] == "easy"],
            "neg_mid": [r for r in neg if r["tier"] == "mid"], "neg_easy": [r for r in neg if r["tier"] == "easy"]}


def select(rows: list[dict], q: dict, seed: str = "", base: list[dict] | None = None) -> list[dict]:
    import haenv.world.validation.pack_size as PS
    pick: list[dict] = []
    base_ids = {r["case"] for r in base or ()}
    B = (lambda rs: [r for r in rs if r["case"] in base_ids])
    count: Counter = Counter()
    cap = spec_cap(sum(q.values()))
    C = cells(rows)
    PS.short(q, {k: len(v) for k, v in C.items()}, "m2 pack cells")

    def add(rs, why):
        pick.extend({**r, "why": why} for r in rs)
        return rs

    def sha(rs):
        return sorted(rs, key=lambda r: key(r["case"], seed))

    def sha_take(rs, n):                      # the base cases of the cell, then sha256 order
        got = B(rs)
        if len(got) > n:
            raise SystemExit(f"[m2 select] nested pack: {len(got)} base cases in a cell of {n}")
        return got + [r for r in sha(rs) if r["case"] not in base_ids][:n - len(got)]

    hard, neg = C["hard"], C["neg_mid"] + C["neg_easy"]
    qh = apportion(q["hard"], {f: sum(1 for r in hard if r["frame"] == f) for f in FRAMES})
    for f in FRAMES:
        hf = [r for r in hard if r["frame"] == f]
        add(_dedup_take(hf, qh[f], set(), seed, count, cap, B(hf)),
            f"hard: {f} quota {qh[f]}, sha256, one per spec, then explain-away twins, spec cap {cap}")
    add(_dedup_take(C["pos_mid"], q["pos_mid"], set(), seed, count, cap, B(C["pos_mid"])), "mid: F1-F3 positive")
    add(_dedup_take(C["pos_easy"], q["pos_easy"], set(), seed, count, cap, B(C["pos_easy"])), "easy: F1-F3 positive")
    add(sha_take(C["f0_mid"], q["f0_mid"]), "mid: F0 bridge non-insufficient")
    add(sha_take(C["f0_easy"], q["f0_easy"]), "easy: F0 bridge non-insufficient")
    n_neg = q["neg_mid"] + q["neg_easy"]
    npos = {f: sum(1 for r in pick if r["frame"] == f) for f in FRAMES}
    qn = apportion(n_neg, npos)
    has_easy = {f: qn[f] for f in FRAMES if any(r["frame"] == f and r["tier"] == "easy" for r in neg)}
    qe = {**{f: 0 for f in FRAMES}, **apportion(q["neg_easy"], has_easy)}
    for f in FRAMES:
        fem = [r["sex"] == "F" for r in pick if r["frame"] == f]
        n_f = round(qn[f] * sum(fem) / len(fem)) if fem else 0
        held: set = set()
        for tier, n in (("easy", qe[f]), ("mid", qn[f] - qe[f])):
            pool = [r for r in neg if r["frame"] == f and r["tier"] == tier]
            PS.short({f"neg {f} {tier}": n}, {f"neg {f} {tier}": len(pool)}, "m2 pack negative arm")
            have_f = sum(1 for r in pick if r["frame"] == f and r["arm"] == "negative" and r["sex"] == "F")
            fem, male = [r for r in pool if r["sex"] == "F"], [r for r in pool if r["sex"] != "F"]
            want_f = max(len(B(fem)), min(n - len(B(male)), n_f - have_f))
            got = _dedup_take(fem, want_f, held, seed, count, cap, B(fem))
            held |= {r["spec"] for r in got}
            got += _dedup_take(male, n - len(got), held, seed, count, cap, B(male))
            held |= {r["spec"] for r in got}
            if len(got) < n:                          # availability: fill in sha256 order
                rest = [r for r in pool if r not in got]
                got += _dedup_take(rest, n - len(got), held, seed, count, cap)
            add(got, f"{tier}: negative arm {f} quota {n} (frame {qn[f]}, female {n_f}) by sha256")
    return pick


def _expanded_cell(r: dict) -> str:
    return f"f0_{r['tier']}" if r["frame"] == "F0" else f"{r['frame']}:{r['arm']}"


def _sex(r: dict) -> str:
    return "F" if r.get("sex") == "F" else "M"


def _quota_cell(r: dict, split: bool) -> str:
    """The quota cell; with `split`, a negative-arm cell is split by sex."""
    c = _expanded_cell(r)
    return f"{c}:{_sex(r)}" if split and r["arm"] == "negative" else c


def negative_sex_quotas(q: dict, positives: list[dict]) -> dict:
    """Split each frame's negative quota by sex in the frame's positive sex mix (rounded half up), so
    frame x sex carries no review-gold signal beyond the frame's."""
    out = {c: v for c, v in q.items() if not c.endswith(":negative")}
    for f in FRAMES:
        n = q.get(f"{f}:negative", 0)
        pos = [r for r in positives if r["frame"] == f and r["arm"] == "positive"]
        fem = int(n * sum(_sex(r) == "F" for r in pos) / len(pos) + 0.5) if pos else 0
        out[f"{f}:negative:F"], out[f"{f}:negative:M"] = fem, n - fem
    return out


def _required_counts(n: int, specs: tuple, required_counts: dict | None) -> dict:
    from haenv.m2 import pack as P
    out = P.expanded_required_counts(n, specs)
    if required_counts is not None:
        out.update(required_counts)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in out.values()):
        raise ValueError("required_counts must contain nonnegative integer positive-case minimums")
    return dict(sorted(out.items()))


def _coverage_deficits(rows: list[dict], required_counts: dict) -> dict:
    positive = Counter(r["spec"] for r in rows if r["arm"] == "positive")
    return {sid: {"required": need, "measured": positive[sid], "deficit": need - positive[sid]}
            for sid, need in sorted(required_counts.items()) if positive[sid] < need}


def select_expanded(rows: list[dict], q: dict, seed: str = "", base: list[dict] | None = None,
                    required_specs: tuple = (), required_counts: dict | None = None,
                    _limits: dict | None = None, _neg_sex: bool = False) -> list[dict]:
    """Exact frame/arm/domain quotas with mandatory coverage and strict spec caps.

    A deterministic minimum-cost flow rewards every unit needed to meet each
    absolute per-spec minimum, then prefers diversity and SHA order. It reroutes choices to
    satisfy constrained F3 supply; a greedy early pick must not invent a shortfall.
    The nested base is immutable and subtracts from every quota and cap.
    The positives are chosen first; the negatives are then re-chosen with each frame's negative
    quota split by sex in that frame's positive sex mix (`negative_sex_quotas`).
    """
    from collections import deque
    from haenv.m2 import pack as P
    n, cap = sum(q.values()), P.expanded_spec_cap(sum(q.values()))
    minimums = _required_counts(n, required_specs, required_counts)
    eligible = sorted((r for r in rows if r.get("eligible", True) and not r.get("insufficient")
                       and _quota_cell(r, _neg_sex) in q), key=lambda r: (key(r["case"], seed), r["case"]))
    by_id = {r["case"]: r for r in eligible}
    if len(by_id) != len(eligible):
        raise ValueError("expanded selection: duplicate case identities")
    deficits = _coverage_deficits(eligible, minimums)
    if deficits:
        raise ValueError(f"expanded coverage shortfall in eligible positive supply: deficits {deficits}")
    if any(r["case"] not in by_id for r in base or ()):
        raise ValueError("expanded nested base has an ineligible/missing case")
    pick = [by_id[r["case"]] for r in base or ()]
    held = {r["case"] for r in pick}
    count = Counter(r["spec"] for r in pick)
    group_count = Counter((r["spec"], _expanded_cell(r)) for r in pick)
    if len(held) != len(pick) or any(v > cap for v in count.values()):
        raise ValueError("expanded nested base violates identity/spec cap")

    def take(r):
        pick.append(r)
        held.add(r["case"])
        count[r["spec"]] += 1
        group_count[(r["spec"], _expanded_cell(r))] += 1

    # Frozen bridge cells use the same SHA order at both sizes, under the strict cap.
    for cell in P.BRIDGE:
        want = q.get(cell, 0) - sum(_expanded_cell(r) == cell for r in pick)
        if want < 0:
            raise ValueError(f"expanded nested bridge quota shortfall: {cell}")
        for r in eligible:
            if (want and _expanded_cell(r) == cell and r["case"] not in held and count[r["spec"]] < cap
                    and group_count[(r["spec"], cell)] < (_limits or {}).get((r["spec"], cell), n)):
                take(r)
                want -= 1
        if want:
            raise ValueError(f"expanded bridge shortfall {cell}: {want} under spec cap {cap}")

    positive_n = sum(q.get(f"{f}:positive", 0) for f in FRAMES)
    bridge = [r for r in pick if r["frame"] == "F0"]
    outside = int((positive_n + len(bridge)) / 2 + 0.5) - sum(r["outside_endocrine"] for r in bridge)
    domain_q = {"outside": outside, "endocrine_or_mixed": positive_n - outside,
                "negative": sum(v for c, v in q.items() if ":negative" in c)}

    def domain(r):
        return "negative" if r["arm"] == "negative" else (
            "outside" if r["outside_endocrine"] else "endocrine_or_mixed")
    cell_q = {c: v - sum(_quota_cell(r, _neg_sex) == c for r in pick) for c, v in q.items() if c not in P.BRIDGE}
    for r in pick:
        if r["frame"] != "F0":
            domain_q[domain(r)] -= 1
    if any(v < 0 for v in (*cell_q.values(), *domain_q.values())):
        raise ValueError("expanded nested strata shortfall: base exceeds planned quotas")
    base_positive = Counter(r["spec"] for r in pick if r["arm"] == "positive")
    remaining_need = {sid: max(0, need - base_positive[sid]) for sid, need in minimums.items()}
    available = [r for r in eligible if r["frame"] != "F0" and r["case"] not in held]
    rank = {r["case"]: i for i, r in enumerate(available)}

    graph = {}

    def edge(a, b, capacity, cost=0):
        graph.setdefault(a, [])
        graph.setdefault(b, [])
        fwd = [b, capacity, cost, len(graph[b])]
        rev = [a, 0, -cost, len(graph[a])]
        graph[a].append(fwd)
        graph[b].append(rev)
        return fwd
    for d, v in domain_q.items():
        edge("source", ("domain", d), v)
    for cell, v in cell_q.items():
        edge(("cell", cell), "sink", v)
    grouped = {}
    for r in available:
        grouped.setdefault(r["spec"], []).append(r)
    # Every still-required unit dominates every possible diversity/SHA cost.
    priority = (n + 1) ** 3 * (len(available) + 1)
    case_edges = []
    for sid, rs in sorted(grouped.items()):
        domains = {domain(r) for r in rs}
        if len(domains) != 1:
            raise ValueError(f"{sid}: inconsistent clinical system/arm metadata")
        clinical_domain = next(iter(domains))
        for j in range(max(0, cap - count[sid])):
            edge(("domain", clinical_domain), ("spec", sid), 1,
                 -priority if j < remaining_need.get(sid, 0) else (count[sid] + j) * (len(available) + 1))
        for cell in sorted({_expanded_cell(r) for r in rs}):
            if (sid, cell) in (_limits or {}):
                edge(("spec", sid), ("group", sid, cell),
                     max(0, _limits[(sid, cell)] - group_count[(sid, cell)]))
        for r in rs:
            cell = _expanded_cell(r)
            node = ("group", sid, cell) if (sid, cell) in (_limits or {}) else ("spec", sid)
            link = edge(node, ("case", r["case"]), 1, rank[r["case"]])
            edge(("case", r["case"]), ("cell", _quota_cell(r, _neg_sex)), 1)
            case_edges.append((r, link))

    sent, need = 0, sum(cell_q.values())
    while sent < need:
        distance, prev = {"source": 0}, {}
        queue, queued = deque(["source"]), {"source"}
        while queue:
            a = queue.popleft()
            queued.remove(a)
            for i, e in enumerate(graph.get(a, ())):
                b, capacity, cost, _ = e
                if capacity and distance[a] + cost < distance.get(b, float("inf")):
                    distance[b], prev[b] = distance[a] + cost, (a, i)
                    if b not in queued:
                        queued.add(b)
                        queue.append(b)
        if "sink" not in prev:
            partial = pick + [r for r, link in case_edges if link[1] == 0]
            raise ValueError(f"expanded strata shortfall: filled {sent}/{need}, cap {cap}, "
                             f"planned remaining cells {cell_q}, domains {domain_q}; "
                             f"positive minimum deficits {_coverage_deficits(partial, minimums)}")
        b = "sink"
        while b != "source":
            a, i = prev[b]
            e = graph[a][i]
            e[1] -= 1
            graph[b][e[3]][1] += 1
            b = a
        sent += 1
    for r, link in case_edges:
        if link[1] == 0:
            take(r)
    deficits = _coverage_deficits(pick, minimums)
    if deficits:
        raise ValueError(f"expanded coverage shortfall under strata/caps: deficits {deficits}")
    if not _neg_sex:
        held_base = {r["case"] for r in base or ()}
        keep = list(base or ()) + [r for r in pick if r["arm"] != "negative" and r["case"] not in held_base]
        return select_expanded(rows, negative_sex_quotas(q, pick), seed, base=keep, required_specs=required_specs,
                               required_counts=required_counts, _limits=_limits, _neg_sex=True)
    return [{**r, "why": "benchmark frame/arm/system strata; strict cap; SHA order"}
            for r in sorted(pick, key=lambda r: (key(r["case"], seed), r["case"]))]


# Bound work, never weaken a quota when a nested solution needs more search.
NESTED_SEARCH_LIMIT = 128


def select_expanded_nested(rows: list[dict], quotas: dict[int, dict], seed: str = "",
                           required_specs: tuple = ()) -> dict[int, list[dict]]:
    """Find every nested size before committing any base, using the existing flow.

    Work largest-first, then select each smaller pack from its parent. If a
    parent cannot contain a valid child, branch on its spec/cell counts: any
    different feasible parent must decrease at least one selected group count.
    Rows in a group are interchangeable for quotas/caps/coverage, so this avoids
    enumerating equivalent case subsets. SHA order and constraints stay fixed.
    """
    sizes = sorted(quotas)
    if not sizes or any(sum(quotas[n].values()) != n for n in sizes):
        raise ValueError("expanded nested quotas must match their pack sizes")
    attempts, last_error = 0, "no feasible allocation"

    def search(index, candidates):
        nonlocal attempts, last_error
        n = sizes[index]
        pending, seen = [{}], set()
        while pending:
            limits = pending.pop()
            state = tuple(sorted(limits.items()))
            if state in seen:
                continue
            seen.add(state)
            if attempts >= NESTED_SEARCH_LIMIT:
                raise ValueError(f"expanded nested search limit {NESTED_SEARCH_LIMIT} reached: {last_error}")
            attempts += 1
            try:
                pick = select_expanded(candidates, quotas[n], seed, required_specs=required_specs, _limits=limits)
            except ValueError as exc:
                last_error = str(exc)
                continue
            if index == 0:
                return {n: pick}
            child = search(index - 1, pick)
            if child is not None:
                return {**child, n: pick}
            counts = Counter((r["spec"], _expanded_cell(r)) for r in pick)
            # Highest counts first, then stable group name. This is a constraint
            # search, not another seed draw or a relaxed acceptance criterion.
            for group, count in reversed(sorted(counts.items(), key=lambda x: (-x[1], x[0]))):
                pending.append({**limits, group: count - 1})
        return None

    result = search(len(sizes) - 1, rows)
    if result is None:
        raise ValueError(f"expanded nested strata/coverage shortfall: {last_error}")
    return result


def coverage_report(rows: list[dict], pick: list[dict], q: dict, required_specs: tuple,
                    required_counts: dict | None = None) -> dict:
    from haenv.m2 import pack as P
    minimums = _required_counts(sum(q.values()), required_specs, required_counts)
    pos = [r for r in pick if r["arm"] != "negative"]
    outside = sum(r["outside_endocrine"] for r in pos)
    present = {r["spec"] for r in pick if r["arm"] == "positive"}
    positive_counts = Counter(r["spec"] for r in pick if r["arm"] == "positive")
    deficits = _coverage_deficits(pick, minimums)
    uniform_minimum = next(iter(minimums.values())) if len(set(minimums.values())) == 1 else None

    def counts(rs, field):
        return dict(sorted(Counter(r[field] for r in rs).items()))
    return {"mode": "expanded", "planned": {
        "basis": "benchmark strata; not disease prevalence", "published_n": sum(q.values()),
        "frame_arm_cells": dict(q), "required_expansion_specs": list(minimums),
        "min_positive_per_expansion_spec": uniform_minimum, "required_positive_counts": minimums,
        "spec_cap": P.expanded_spec_cap(sum(q.values())),
        "outside_endocrine_positive_share": 0.5, "share_rounding_tolerance_cases": 1,
        "difficulty_quota": None}, "measured": {
        "pool_n": len(rows), "published_n": len(pick),
        "pool_by_spec": counts(rows, "spec"), "published_by_spec": counts(pick, "spec"),
        "published_by_system": counts(pos, "system"),
        "published_frame_arm_cells": dict(sorted(Counter(_expanded_cell(r) for r in pick).items())),
        "positive_diagnostic_n": len(pos), "outside_endocrine_positive_n": outside,
        "outside_endocrine_positive_share": outside / len(pos) if pos else None,
        "missing_expansion_specs": sorted(s for s, need in minimums.items() if need and s not in present),
        "published_positive_by_required_spec": {sid: positive_counts[sid] for sid in minimums},
        "undercovered_expansion_specs": sorted(deficits), "expansion_spec_deficits": deficits,
        "difficulty_by_frame": {f: counts([r for r in pick if r["frame"] == f], "tier") for f in FRAMES},
        "spec_cap_overflow": {s: k for s, k in counts(pick, "spec").items()
                              if k > P.expanded_spec_cap(sum(q.values()))}}}


def youden_best(groups: dict) -> float:
    """Best in-sample Youden J of a rule 'answer warranted on this subset of groups':
    groups = {g: (n_pos, n_neg)}; J = max over subsets of TPR - FPR (= sum of the positive gaps)."""
    P = sum(p for p, _ in groups.values())
    N = sum(n for _, n in groups.values())
    if not P or not N:
        return 0.0
    return sum(max(0.0, p / P - n / N) for p, n in groups.values())


def framesex_loo_cc(pick: list[dict]) -> float:
    """2 * BA - 1 of 'answer the review gold of the other cases of my frame x sex cell (majority, ties
    warranted)' over the F1/F3 rows; an empty cell falls back to the frame."""
    fc = [r for r in pick if r["frame"] in FRAMES]
    hit, tot = Counter(), Counter()
    for r in fc:
        if r["frame"] not in REVIEW_FRAMES:
            continue
        cell = [x for x in fc if x is not r and (x["frame"], _sex(x)) == (r["frame"], _sex(r))] or \
               [x for x in fc if x is not r and x["frame"] == r["frame"]]
        gold = r["arm"] == "positive"
        tot[gold] += 1
        hit[gold] += (sum(x["arm"] == "positive" for x in cell) / len(cell) >= 0.5) == gold
    if not tot[True] or not tot[False]:
        return 0.0
    return hit[True] / tot[True] + hit[False] / tot[False] - 1.0


def balance(pick: list[dict], rows: list[dict], mode: str = "legacy") -> dict:
    import math
    from haenv.m2.pack import expanded_spec_cap
    bound = expanded_spec_cap(len(pick)) if mode == "expanded" else spec_cap(len(pick))
    fc = [r for r in pick if r["frame"] in FRAMES]
    lab = {f: (sum(1 for r in fc if r["frame"] == f and r["arm"] == "positive"),
               sum(1 for r in fc if r["frame"] == f and r["arm"] == "negative")) for f in FRAMES}
    tiers = {f: dict(sorted(Counter(r["tier"] for r in fc if r["frame"] == f).items())) for f in FRAMES}
    fs = {}
    for r in fc:
        p, n = fs.get((r["frame"], r["sex"]), (0, 0))
        fs[(r["frame"], r["sex"])] = (p + (r["arm"] == "positive"), n + (r["arm"] == "negative"))
    tv = sorted({r["tier"] for r in fc})
    tab = [[sum(1 for r in fc if r["frame"] == f and r["tier"] == t) for t in tv] for f in FRAMES]
    tot = sum(map(sum, tab))
    chi = sum((tab[i][j] - sum(tab[i]) * sum(row[j] for row in tab) / tot) ** 2
              / (sum(tab[i]) * sum(row[j] for row in tab) / tot)
              for i in range(len(FRAMES)) for j in range(len(tv)) if sum(tab[i]) and sum(row[j] for row in tab))
    denom = tot * (min(len(FRAMES), len(tv)) - 1)
    cv = math.sqrt(chi / denom) if denom else 0.0
    # E_line: same-frame leave-one-out list vs pack leave-one-out list, top-(n+1), on positives
    posr = [r for r in pick if r["lines"]]

    def hit(r, pool):
        cnt = Counter(x for o in pool if o is not r for x in o["lines"])
        top = [x for x, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))][:len(r["lines"]) + 1]
        return len(set(top) & set(r["lines"])) / len(r["lines"])
    e = {f: [hit(r, [o for o in posr if o["frame"] == f]) - hit(r, posr) for r in posr if r["frame"] == f]
         for f in FRAMES}
    allv = [x for v in e.values() for x in v]
    bal1 = all(p and n for p, n in lab.values()) and (mode == "expanded" or all(len(t) >= 2 for t in tiers.values()))
    j = youden_best(lab)
    jfs = youden_best(fs)
    fsl = framesex_loo_cc(pick)
    bal3 = mode != "expanded" or fsl <= BAL3_MAX
    return {"labels_by_frame": {f: {"positive": p, "negative": n} for f, (p, n) in lab.items()},
            "tiers_by_frame": tiers, "sex_by_frame_label": {f"{k[0]}:{k[1]}": {"positive": v[0], "negative": v[1]}
                                                            for k, v in sorted(fs.items(), key=str)},
            "BAL-1": {"pass": bal1}, "BAL-2": {"youden_frame": round(j, 4), "max": BAL2_MAX, "pass": j <= BAL2_MAX},
            "youden_frame_x_sex": round(jfs, 4), "cramers_v_frame_x_tier": round(cv, 3),
            "BAL-3": {"framesex_loo_cc": round(fsl, 4), "max": BAL3_MAX, "pass": bal3,
                      "gated": mode == "expanded"},
            "E_line": {"mean": round(sum(allv) / len(allv), 3) if allv else None,
                       "by_frame": {f: (round(sum(v) / len(v), 3) if v else None) for f, v in e.items()}},
            "specs_repeated_within_frame": {f: sorted(k for k, c in Counter(r["spec"] for r in fc if r["frame"] == f).items() if c > 1)
                                            for f in FRAMES},
            "spec_cap": {"cap": bound,
                         "n_specs": len({r["spec"] for r in pick}),
                         "over_cap": {k: c for k, c in sorted(Counter(r["spec"] for r in pick).items())
                                      if c > bound}},
            "ea_twins": {"explained": sum(1 for r in pick if r.get("ea_twin") == "explained"),
                         "unexplained": sum(1 for r in pick if r.get("ea_twin") == "unexplained")},
            "pass": bal1 and j <= BAL2_MAX and bal3}
