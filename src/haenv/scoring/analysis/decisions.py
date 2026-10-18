"""Decision/review/abstention and join-class domain measurements."""
from __future__ import annotations

from haenv.scoring.analysis.precision import _fsum_mean
from haenv.scoring.analysis.precision import _rd


def chance_corrected(ba):
    """A3 (2026-10-01): balanced accuracy -> skill over chance, `max(0, 2 x BA - 1)` (Youden J,
    floored at 0 like `skill_over_floor`); both constant answers land on 0.000. `None` stays `None`."""
    if ba is None:
        return None
    return _rd(max(0.0, 2.0 * float(ba) - 1.0), 4)


def review_macro_of(rs: list[dict]) -> dict:
    """Review-flag judgment: `review_macro` is the specificity half only.

    A missed referral is penalized once, by the review gates on the slice where it happened
    (`verifier.UNIT_SCOPED_GATES`); `review_macro` scores the other error, requesting review
    where gold does not warrant it. It is the not-warranted class mean of `review_flag_ok`
    ("always says review is needed" 0.000, "never says so" 1.000, but the latter trips a
    review gate on every warranted slice). With a single gold class present no value is
    given: without warranted cases the gates have nothing to bite on and "never" would take
    the full score unopposed. Both class means stay in `review_by_class` for reading.
    """
    _rv: dict = {True: [], False: []}
    for r in rs:
        if isinstance(r.get("review_flag_ok"), (int, float)):
            _rv[bool(r.get("review_warranted"))].append(float(r["review_flag_ok"]))
    out = {"review_by_class": {("warranted" if k else "not_warranted"):
                               _fsum_mean(v)
                               for k, v in _rv.items()},
           "review_n_by_class": {("warranted" if k else "not_warranted"): len(v)
                                 for k, v in _rv.items()}}
    # The declaration rate detects "always say review is needed".
    _dec = [1.0 if r.get("review_declared") else 0.0 for r in rs
            if isinstance(r.get("review_flag_ok"), (int, float))]
    out["review_declare_rate"] = _fsum_mean(_dec)
    # Which slice was graded is recorded alongside the reading.
    _bas = sorted({int(r["review_basis_slice"]) for r in rs
                   if isinstance(r.get("review_basis_slice"), (int, float))})
    out["review_basis_slices"] = (f"{_bas[0]}–{_bas[-1]}" if len(_bas) > 1
                                  else (str(_bas[0]) if _bas else None))
    _both = all(out["review_by_class"][k] is not None for k in ("warranted", "not_warranted"))
    out["review_macro"] = out["review_by_class"]["not_warranted"] if _both else None
    # M1 D1/D2: the review question scored once, per case, as balanced accuracy.
    out["review_sens"] = out["review_by_class"]["warranted"] if _both else None
    out["review_utility"] = (_rd((out["review_by_class"]["warranted"]
                                  + out["review_by_class"]["not_warranted"]) / 2, 4)
                             if _both else None)
    out["review_utility_const"] = 0.5 if _both else None
    # A3 (2026-10-01): the scored form is chance-corrected, max(0, 2 x BA - 1); raw BA is a profile reading.
    out["review_utility_cc"] = chance_corrected(out["review_utility"])
    # Degenerate constants: "always says so" lands on 0.000; "never says so" on 1.000 here,
    # paid for by the review gates.
    out["review_macro_const_floor"] = 0.0 if _both else None
    out["review_macro_note"] = (
        "Specificity only: the share of not-warranted cases (gold `clinician_action_warranted` false) "
        "on which review was not requested. \"Always says review is needed\" scores 0.000. "
        "A missed referral is not scored here: it trips `missing_clinician_review_flag` / `premature_closure` "
        "on the slice where it happened and zeroes that slice, once."
        + (f" On the slice geometry this judges the final slice (t={out.get('review_basis_slices')}): "
           f"the most-informed request for help, the same convention as the noop/quant probes."
           if out.get("review_basis_slices") else "")
        if _both else
        "Only one class of warranted in this batch, so no value: without warranted cases the review gates "
        "have nothing to bite on, and \"never says so\" would take 1.000 unopposed.")
    return out


INSUFFICIENT_KIND = "ddx:insufficient"


def abstention_utility_of(rs: list[dict]) -> dict:
    """Abstention on the insufficient-information tier, scored two-sided.

    `abst_utility` = (sensitivity + specificity) / 2, per solver:
    sensitivity = mean `abst_ok` on `ddx:insufficient` cells (gated: declared
    `insufficient_data`; slices: share of slices declaring it -- every slice of this tier has no
    true symptom); specificity = mean `1 - abst_over` on the other cells (gated: did not declare;
    slices: share of symptom-visible slices not declaring; no such slice => not applicable).
    Unanswered (synthetic zero) cells count 0 on their side. One side empty => no value.
    "Always abstain" and "never abstain" both score 0.500; the one-sided tier-only mean
    (`abst_sens`, reported) gives "always abstain" 1.000 at no cost elsewhere, since declaring
    insufficient data does not stop an answer from listing tests and diagnoses.
    """
    sens, spec = [], []
    for r in rs:
        if not str(r.get("gold_kind", "")).startswith("ddx:"):
            continue
        ins = r.get("gold_kind") == INSUFFICIENT_KIND
        if r.get("synthetic_no_answer"):
            (sens if ins else spec).append(0.0)
            continue
        if ins:
            v = r.get("abst_ok")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sens.append(float(v))
        else:
            v = r.get("abst_over")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                spec.append(1.0 - float(v))
    s, p = _fsum_mean(sens), _fsum_mean(spec)
    both = s is not None and p is not None
    return {"abst_sens": s, "abst_spec": p, "abst_n_by_side": {"insufficient": len(sens), "other": len(spec)},
            "abst_utility": (_rd((s + p) / 2, 4) if both else None),
            "abst_utility_const": 0.5 if both else None,
            # A3: scored form, chance-corrected; raw `abst_utility` is a profile reading.
            "abst_utility_cc": (chance_corrected(_rd((s + p) / 2, 4)) if both else None)}


def answer_space_diff(rows: list[dict]) -> dict:
    """The drivers the solver may answer vs. the values gold actually uses.

    Counts real models only (a stub answering the gold placeholder would inflate
    it). Reports `top_rate` (top pick in the difference set) and `all_rate` (all
    nominated drivers) separately, each with its count and total.
    """
    from haenv.world.generation.baselines import BASELINE_NAMES as _BN
    try:
        from haenv_kernel.solver import ALLOWED_DRIVERS  # kernel: the only source of the menu
    except Exception:                                      # noqa: BLE001
        return {}
    _stub = set(_BN)
    gold: set[str] = set()
    for r in rows:
        gold |= {str(x) for x in (r.get("gold_drivers") or []) if x}
    if not gold:
        return {}
    dead = set(ALLOWED_DRIVERS) - gold
    n_top = d_top = n_all = d_all = 0
    hot: dict[str, int] = {}
    for r in rows:
        if r.get("solver") in _stub:
            continue  # real models only
        if r.get("top_driver"):
            n_top += 1
            d_top += r["top_driver"] in dead
        for x in (r.get("all_drivers") or []):
            n_all += 1
            if x in dead:
                d_all += 1
                hot[x] = hot.get(x, 0) + 1
    return {"menu": len(ALLOWED_DRIVERS), "gold_used": len(gold), "dead": len(dead),
            "n_top": n_top, "d_top": d_top,
            "top_rate": round(d_top / n_top, 4) if n_top else None,
            "n_all": n_all, "d_all": d_all,
            "all_rate": round(d_all / n_all, 4) if n_all else None,
            # name in the sort key => total order
            "hot": sorted(hot.items(), key=lambda kv: (-kv[1], kv[0]))[:5],
            "dead_names": sorted(dead)}


JOIN_CLASSES: tuple[str, ...] = ("unified", "comorbidity", "independent")


def join_class_of(row: dict) -> str | None:
    """This row's gold merge class, from `gold_kind`."""
    gk = str(row.get("gold_kind") or "")
    return gk.split(":", 1)[1] if gk.startswith("ddx:") else None


_CEILING_MAX_DIFF_ITEMS = 1


def join_scored_classes(rows: list[dict],
                        baseline_solvers: tuple = ("const_ddx", "baseline_slope", "robust_ref",
                                                   "no_revision", "flip_flop"),
                        metric: str = "join_hit") -> dict:
    """Which classes enter the macro average and which are excluded for zero variance, per (class x metric).

    Returns `{"scored": [...], "excluded": {...}, "ceiling": {...}}`; judged for
    every batch.
    """
    per: dict[str, dict[str, list]] = {}
    for r in rows:
        k = join_class_of(r)
        v = r.get(metric)
        if k is None or v is None or r.get("solver") in baseline_solvers:
            continue
        per.setdefault(k, {}).setdefault(r.get("solver", "?"), []).append(bool(v))
    scored, excluded, ceiling = [], {}, {}
    for k in JOIN_CLASSES:
        by_solver = per.get(k) or {}
        if len(by_solver) < 2:
            excluded[k] = f"not enough comparable solvers ({len(by_solver)} < 2) -- unjudgeable, not 0"
            continue
        tot = {m: sum(v) for m, v in by_solver.items()}
        n_items = len(next(iter(by_solver.values())))
        if len(set(tot.values())) == 1:
            excluded[k] = (f"zero variance: all {len(tot)} models are identical "
                           f"({next(iter(tot.values()))}/{n_items})"
                           f"; no evidence under spec §17")
            continue
        # Ceiling alert only; it never changes which classes are scored.
        # Count the positions where at least one model differs from the others.
        cols = list(zip(*(by_solver[m] for m in sorted(by_solver))))
        n_diff = sum(1 for c in cols if len(set(c)) > 1)
        if n_diff <= _CEILING_MAX_DIFF_ITEMS:
            ceiling[k] = (f"hugging the ceiling: the difference between {len(tot)} models rests on only {n_diff} cell(s)"
                          f"(out of {n_items} cells total, scores {sorted(tot.values())}); "
                          f"zero-variance exclusion is not triggered, but the evidentiary strength is equivalent. "
                          f"Exclusion is undecided")
        scored.append(k)
    return {"scored": scored, "excluded": excluded, "ceiling": ceiling}


def _scope_const_floor(scored: list[str]) -> float | None:
    """Analytic floor for `join_scope_ok`'s macro average.

    Bands: unified `=1`, comorbidity `[1/n,(n-1)/n]`, independent `=1/n`; "always
    cite one true symptom" passes comorbidity and independent, "cite everything"
    passes unified. With all three classes scored the floor is 0.667 (vs. `1/k` =
    0.333 for `join_hit`).
    """
    if not scored:
        return None
    n = len(scored)
    always_one = sum(1 for c in scored if c in ("comorbidity", "independent")) / n
    always_all = sum(1 for c in scored if c == "unified") / n
    return round(max(always_one, always_all), 3)
