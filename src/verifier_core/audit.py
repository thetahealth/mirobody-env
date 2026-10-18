"""Portable anti-gaming audit: four checks that can be run against someone else's benchmark.

| check | what it catches |
|---|---|
| degenerate ceiling | whether any real solver beat the best degenerate strategy, with the margin shown next to the noise floor |
| noise floor | whether a difference is readable at all |
| pairwise resolution | how far down the ranking can actually be read |
| shortcut scan | whether a single feature predicts the gold label |

The input is a results file; the audited benchmark's code is never imported. An
adapter only maps its field names onto this contract:

```
{"name": str,
 "rows": [{"solver": str, "role": "model"|"baseline"|"oracle", "<metric>": float}],
 "samples": {solver: [v1, v2, ...]} | None, # repeated runs of the same prompts
 "samples_comparable": bool | None, # are samples and rows the same setup?
 "thresholds": {...}} # the audited benchmark's own thresholds, reported as-is
```

Repeats taken under a different temperature or cache configuration cannot serve as the
leaderboard's noise floor; when the adapter cannot tell, it sets `samples_comparable`
to `None` and pairwise resolution is refused.

Reading rules: the ceiling margin is compared with the noise floor, never with 0;
differences are resolved pairwise, never as range/noise; fewer than `MIN_K` repeats
means "not measured", never 0; thresholds are frozen constants.

Limits: without per-case features (`cases` + `scan`) the shortcut scan reports
`skipped`, never clean; the ceiling check needs degenerate baselines in the results;
the tool audits how numbers are read, not whether they were computed correctly.

Evaluation infrastructure for SYNTHETIC data; it makes no domain judgements.
"""
from __future__ import annotations

import statistics as st
from typing import Callable, Sequence

#: A pair counts as resolved when its gap is at least this multiple of the noise floor:
#: at 2x, the two solvers' sampled ranges no longer overlap.
RESOLVE_RATIO = 2.0

#: Minimum repeats for `pass^k` / the noise floor.
MIN_K = 3

#: Digits to round to before comparing, so the verdict is computed on the printed value.
_ROUND = 4


# ---------------------------------------------------------------- the four checks
def _values(payload: dict) -> dict[str, float]:
    """Per-solver value of the primary metric: the mean when at least `MIN_K` repeated
    samples exist (the noise floor is a mean-level quantity), else the single run.
    """
    m = payload["metric"]
    s = payload.get("samples") or {}
    out: dict[str, float] = {}
    for r in payload["rows"]:
        if r["role"] != "model":
            continue
        v = s.get(r["solver"])
        out[r["solver"]] = st.mean(v) if v and len(v) >= MIN_K else float(r[m])
    return out


def noise_floor(payload: dict) -> dict:
    """Noise floor = the maximum per-solver self-range. With `k < MIN_K`,
    report unusable -- never return 0."""
    s = payload.get("samples")
    if not s:
        return {"ok": False, "why": "no_samples", "detail": "the results file has no repeated readings of the same prompts"}
    ks = {a: len(v) for a, v in s.items()}
    bad = {a: k for a, k in ks.items() if k < MIN_K}
    if bad:
        return {"ok": False, "why": "k_too_small",
                "detail": f"solvers with k < {MIN_K}: {bad} — reported as not "
                          "measured, not as a noise floor of 0"}
    per = {a: (max(v) - min(v)) for a, v in s.items()}
    worst = max(per.items(), key=lambda kv: (kv[1], kv[0]))
    return {"ok": True, "floor": round(worst[1], 4), "worst": worst[0],
            "median": round(st.median(per.values()), 4), "n": len(per),
            "per_solver": dict(sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))),
            "comparable": payload.get("samples_comparable")}


def degenerate_ceiling(payload: dict, floor: float | None) -> dict:
    """Best degenerate strategy (neither model nor oracle) vs best real
    solver; the margin is reported next to the noise floor."""
    m = payload["metric"]
    mod = sorted(_values(payload).items(), key=lambda kv: (-kv[1], kv[0]))
    base = [(r["solver"], r[m]) for r in payload["rows"] if r["role"] == "baseline"]
    if not mod or not base:
        return {"ok": False, "why": "no_baseline" if mod else "no_model",
                "detail": f"{len(mod)} model(s) · {len(base)} degenerate "
                          "baseline(s) — with one side missing there is no "
                          "comparable ceiling"}
    bm, bb = max(mod, key=lambda kv: kv[1]), max(base, key=lambda kv: kv[1])
    margin = bm[1] - bb[1]
    margin = round(margin, _ROUND)
    ratio = (None if floor is None else
             ((float("inf") if margin > 0 else 0.0) if floor == 0.0
              else round(margin / floor, 2)))
    return {"ok": True, "best_model": bm, "best_baseline": bb, "margin": margin,
            "ratio": ratio,
            "verdict": ("no_noise_floor" if ratio is None else
                        ("ok" if ratio >= RESOLVE_RATIO else "unresolved")),
            "floor_is_zero": (floor == 0.0) or None}


def pairwise(payload: dict, floor: float | None) -> dict:
    """Pairwise resolution. Never "range / noise" -- that only shows the
    two ends are separable."""
    mod = sorted(_values(payload).items(), key=lambda kv: (-kv[1], kv[0]))
    if floor is None:
        # No comparable floor: report the top gap against the median self-range of whatever
        # repeats exist, as an indication only.
        ind = None
        s = payload.get("samples") or {}
        if s and len(mod) >= 2:
            per = [max(v) - min(v) for v in s.values() if len(v) >= MIN_K]
            if per:
                med = st.median(per)
                gap = abs(mod[0][1] - mod[1][1])
                ind = {"indicative_noise_median": round(med, 4),
                       "top_gap": round(gap, 4),
                       "ratio_if_applied": round(gap / med, 2) if med else None,
                       "direction": "these repeats come from a different "
                                    "configuration, so treat this as an upper "
                                    "bound at most; the leaderboard's own noise "
                                    "was not measured"}
        return {"ok": False, "why": "no_noise_floor",
                "detail": f"{len(mod)} models, {len(mod)*(len(mod)-1)//2} pairs "
                          "— no comparable noise floor, so no pair can be "
                          "resolved",
                "indicative": ind}
    # A zero floor resolves any non-zero gap; resolve anyway and flag it.
    zero = (floor == 0.0)
    pairs = []
    for i in range(len(mod)):
        for j in range(i + 1, len(mod)):
            # Round to printing precision first, then compare -- see `_ROUND`
            gap = round(abs(mod[i][1] - mod[j][1]), _ROUND)
            ratio = (float("inf") if (zero and gap > 0) else
                     (0.0 if zero else round(gap / floor, 2)))
            pairs.append({"a": mod[i][0], "b": mod[j][0], "gap": gap,
                          "ratio": (None if ratio == float("inf") else ratio),
                          "resolved": ratio >= RESOLVE_RATIO})
    n_res = sum(1 for p in pairs if p["resolved"])
    return {"ok": True, "n_pairs": len(pairs), "n_resolved": n_res,
            "floor_is_zero": zero or None,
            "top_gap": pairs[0] if pairs else None,
            "resolved": [p for p in pairs if p["resolved"]][:8]}


def shortcut_scan(payload: dict) -> dict:
    """Shortcut scan with a caller-supplied scanner (`payload["scan"]` over `payload["cases"]`).
    Missing either reports `skipped`, never clean.
    """
    cases = payload.get("cases")
    if not cases:
        return {"ok": False, "why": "no_features",
                "detail": "the results file has no per-case features or labels "
                          "⇒ skipped (not reported as clean)"}
    scan: Callable[[Sequence], list[dict]] | None = payload.get("scan")
    if scan is None:
        return {"ok": False, "why": "no_scanner", "n_cases": len(cases),
                "detail": "the adapter supplied per-case features but no "
                          "scanner ⇒ skipped (not reported as clean); "
                          "`cases` and `scan` must be supplied together"}
    try:
        raw_hits = scan(cases)
        # `shortcut_target_unscannable` is a gap in the scan surface, not a shortcut; it is
        # reported separately from `n_hits`.
        unscan = [h for h in raw_hits if h.get("kind") == "shortcut_target_unscannable"]
        hits = [h for h in raw_hits if h.get("kind") != "shortcut_target_unscannable"]
        _u = {"unscannable": [h["target_family"] for h in unscan]} if unscan else {}
        if hits:
            return {"ok": False, "why": "shortcut_found", "n_cases": len(cases),
                    "n_hits": len(hits), "hits": hits, **_u,
                    "detail": f"scan found {len(hits)} candidate "
                              f"single-feature shortcut(s)"}
        if unscan:
            return {"ok": False, "why": "target_unscannable", "n_cases": len(cases),
                    "n_hits": 0, "hits": [], **_u,
                    "detail": (f"no shortcut on the other dimensions, but "
                               f"{'/'.join(_u['unscannable'])} had no "
                               f"scannable target (the gold label is "
                               f"near-constant) ⇒ not reported as clean")}
        return {"ok": True, "n_cases": len(cases), "n_hits": 0, "hits": [],
                "detail": f"shortcut scan passed (n={len(cases)}): no "
                          f"single-feature shortcut found"}
    except Exception as e:
        return {"ok": False, "why": "scan_error",
                "detail": f"shortcut scan raised: {e}"}


# ---------------------------------------------------------------- reporting
def run(payload: dict) -> dict:
    nf = noise_floor(payload)
    # `floor is None` (not measured / not comparable) and `floor == 0.0` (measured zero)
    # are different states.
    floor = nf["floor"] if (nf.get("ok") and nf.get("comparable") is True) else None
    return {"name": payload["name"], "metric": payload["metric"],
            "n_rows": len(payload["rows"]), "their_thresholds": payload.get("thresholds"),
            "noise": nf, "ceiling": degenerate_ceiling(payload, floor),
            "pairwise": pairwise(payload, floor), "shortcut": shortcut_scan(payload)}


def render(res: dict) -> list[str]:
    """`run`'s result as human-readable report lines. Rendering lives here because the
    wording carries the reading rules.
    """
    L: list[str] = []
    L.append(f"Audit subject: {res['name']} · primary metric `{res['metric']}` · {res['n_rows']} solver(s)")
    if res["their_thresholds"]:
        L.append(f"Their own thresholds (reported as-is, not substituted): {res['their_thresholds']}")

    n = res["noise"]
    L.append("\n-- (1) noise floor --")
    if not n.get("ok"):
        L.append(f"  ❌ unusable: {n['why']} — {n['detail']}")
    else:
        cmp_ = n.get("comparable")
        tag = {True: "✅ same setup as the board",
               False: "❌ **different setup from the board**",
               None: "comparability unknown"}[cmp_]
        L.append(f"  max self-range {n['floor']} (least stable: {n['worst']})"
                 f" · median {n['median']} · n={n['n']}  {tag}")
        for s, v in list(n["per_solver"].items())[:5]:
            L.append(f"     {s:32} {v:.4f}")
        if cmp_ is not True:
            L.append("  ⇒ this number must not be used to resolve the "
                     "board: repeats taken under a different configuration "
                     "measure that configuration, not the difference between "
                     "solvers.")

    c = res["ceiling"]
    L.append("\n-- (2) degenerate ceiling (best degenerate vs best real solver) --")
    if not c.get("ok"):
        L.append(f"  ❌ undecidable: {c['why']} — {c['detail']}")
    else:
        L.append(f"  best model {c['best_model'][0]} {c['best_model'][1]:.4f} · "
                 f"best degenerate baseline {c['best_baseline'][0]} "
                 f"{c['best_baseline'][1]:.4f}")
        L.append(f"  margin {c['margin']}"
                 + (f" = {c['ratio']}x noise ⇒ {c['verdict']}" if c["ratio"] else
                    "  ❌ no comparable noise floor, so the margin cannot be judged"))

    p = res["pairwise"]
    L.append("\n-- (3) pairwise resolution (never range/noise) --")
    if not p.get("ok"):
        L.append(f"  ❌ {p['why']} —— {p['detail']}")
        ind = p.get("indicative")
        if ind:
            L.append(f"  ⚪ indicative only (not a verdict): top-two gap "
                     f"{ind['top_gap']}; the only repeats available have median "
                     f"self-range {ind['indicative_noise_median']} ⇒ applying "
                     f"it would give {ind['ratio_if_applied']}x.")
            L.append(f"     {ind['direction']}")
    else:
        L.append(f"  {p['n_resolved']} of {p['n_pairs']} pairs resolved "
                 f"(rule: gap >= {RESOLVE_RATIO}x the noise floor)"
                 + ("  noise floor is 0 (byte-identical repeats) — "
                    "resolved, but zero variance is only weakly supported"
                    if p.get("floor_is_zero") else ""))
        if p["top_gap"]:
            t = p["top_gap"]
            L.append(f"  top two: {t['a']} vs {t['b']} gap {t['gap']} = "
                     + ("inf" if t["ratio"] is None else f"{t['ratio']}") + "x"
                     + ("  ✅ resolved" if t["resolved"] else "  ❌ not resolved"))

    s = res["shortcut"]
    L.append("\n-- (4) shortcut scan --")
    if s.get("ok"):
        L.append(f"  ✅ {s['detail']}")
    else:
        L.append(f"  🔴 {s.get('why', 'fail')} —— {s['detail']}")

    L.append("\nLimits: this tool sees only the results file, so it cannot "
             "tell whether a score was computed correctly; (2) needs degenerate "
             "baselines in the results; (4) is skipped, not reported as clean, "
             "when per-case features are absent.")
    return L
