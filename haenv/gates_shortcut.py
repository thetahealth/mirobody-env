"""Batch-level statistical gates: shortcut solvability (A5), footprint discriminability, and
whether the ddx tier can be read off surface features (GEN8t).

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import re
import collections
import math
from .gates_case import A5_MIN_CLASS, _REG_DIR
from .gates_outcome import DRIVER_REQUIRED_OBSERVABLE, _LAB_VAL_RE


# ============================================================ A5: shortcuts must be unsolvable (batch-level)
# No single-feature threshold rule may identify the gold with high precision; otherwise
# the question tests whether a threshold was crossed, not reasoning.
A5_SEVERITY = "warn"


A5_F1 = 0.9          # a single-feature F1 reaching this value counts as shortcut-solvable


#: Margin by which a feature's F1 must exceed the majority-class baseline to be a candidate.
#: Without it, "always answer the majority class" would count as a shortcut.
#: Empirical and coarse; significance is decided separately by `A5_ALPHA`.
A5_MARGIN_OVER_MAJORITY = 0.05


#: A5's per-disease existence bits: a copy of the kernel's `latent.DISEASE_SIGNAL_DOMAIN`
#: keys, so the feature schema does not depend on the kernel import and is the same for every
#: batch. The maintainers' tests keep the copy in sync.
A5_DISEASE_BITS: tuple[str, ...] = (
    "obesity", "T2D", "hypertension", "dyslipidemia", "MASLD",
)


# ---------------------------------------------------------------- A5v2: hits are split into clinical / nonclinical
# A `clinical` hit lands on the gold's registered evidence stream (reported; GEN14 requires
# that evidence to exist). A `nonclinical` hit lands on demographics, structure, counts or
# construction artifacts, and rejects emission.
NONCLINICAL_FEATURES = ("n_signals", "n_evidence", "n_context", "ctx_facet:")


# Nonclinical by suffix, whatever the stream: `:is_const` (a construction artifact) and
# `:present` (reflects device inventory and injection choices). `:std` is not listed:
# variance on a registered stream can be clinical.
NONCLINICAL_SUFFIXES = (":is_const", ":present")


#: Streams that legitimately predict an `outcome=` gold at T, per event type.
OUTCOME_REQUIRED_OBSERVABLE: dict[str, set[str]] = {
    # Weight regain: weight plus clinical streams; QC and structural streams excluded.
    "weight_regain": {"weight", "medication_adherence", "dose_timeline", "diet_carb_pct",
                      "CGM_TIR", "HbA1c", "steps", "activity_index", "sleep_hours",
                      "gi_symptom_score", "resting_hr", "hrv", "systolic_bp", "LDL",
                      "stress_score", "body_temp", "spo2", "skin_temp"},
}


def classify_shortcut(target: str, feature: str, event_type: str | None = None) -> str:
    """Whether one A5 hit is `clinical` or `nonclinical`, by GEN14's registry and
    `OUTCOME_REQUIRED_OBSERVABLE`. Unregistered targets are nonclinical.
    """
    if any(feature == k or feature.startswith(k) for k in NONCLINICAL_FEATURES):
        return "nonclinical"
    if any(feature.endswith(k) for k in NONCLINICAL_SUFFIXES):
        return "nonclinical"
    stream = feature.split(":", 1)[0]
    if target.startswith("driver="):
        req = DRIVER_REQUIRED_OBSERVABLE.get(target[len("driver="):])
        if req is None:                      # unregistered -> no exemption
            return "nonclinical"
        return "clinical" if stream in req else "nonclinical"
    if target.startswith("outcome="):
        # Looked up by event type (`event_type`, default `weight_regain`), not by label value.
        req = OUTCOME_REQUIRED_OBSERVABLE.get(event_type or "weight_regain")
        if req is None:
            return "nonclinical"                 # an unregistered event type gets no exemption
        return "clinical" if stream in req else "nonclinical"
    if target.startswith("join_gold="):
        return "nonclinical"
    return "nonclinical"


def _lab_features(sp) -> dict[str, float]:
    """Format features of the <=T lab results (counts, timing, abnormal count), never the values:
    a lab value predicting the diagnosis is intended, "how many labs" predicting it is not.
    """
    T = int((sp.prediction_context or {}).get("prediction_time_T", 10 ** 9))
    labs = [e for e in (sp.evidence_ledger or [])
            if e.get("source_type") == "lab_result"
            and int(e.get("source_timestamp", 10 ** 9)) <= T]
    out = {"lab:n": float(len(labs))}
    if labs:
        days = [int(e.get("source_timestamp", 0)) for e in labs]
        out["lab:first_day"] = float(min(days))
        out["lab:span"] = float(max(days) - min(days))
        out["lab:n_distinct_days"] = float(len(set(days)))
        n_ab = 0
        for e in labs:
            m = _LAB_VAL_RE.search(str(e.get("symptom", "")))
            if not m:
                continue
            v, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
            if not (lo <= v <= hi):
                n_ab += 1
        out["lab:n_abnormal"] = float(n_ab)
        out["lab:frac_abnormal"] = round(n_ab / len(labs), 3)
    return out


def case_features(sp) -> dict[str, float]:
    """Scalar features of the solver-visible payload, for the A5 scan."""
    f: dict[str, float] = dict(_lab_features(sp))
    ld = sp.longitudinal_data or {}
    T = int((sp.prediction_context or {}).get("prediction_time_T", 10 ** 9))
    for name, pts in ld.items():
        vis = [p for p in (pts or []) if int(p["ts"]) <= T]
        if not vis:
            continue
        vals = [float(p["value"]) for p in vis]
        f[f"{name}:last"] = vals[-1]
        f[f"{name}:min"] = min(vals)
        f[f"{name}:max"] = max(vals)
        if len(vals) >= 2 and int(vis[-1]["ts"]) != int(vis[0]["ts"]):
            f[f"{name}:slope"] = (vals[-1] - vals[0]) / (int(vis[-1]["ts"]) - int(vis[0]["ts"]))
        if vals[0]:
            f[f"{name}:rel_change"] = (vals[-1] - vals[0]) / abs(vals[0])
        # `:std` / `:is_const`: flat non-gold streams make "variance > 0" a construction shortcut.
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        f[f"{name}:std"] = var ** 0.5
        f[f"{name}:is_const"] = 1.0 if var == 0.0 else 0.0
    f["n_evidence"] = float(len(sp.evidence_ledger or []))
    f["n_signals"] = float(len(ld))

    # Text shape: counts per context facet (`overlay.facet_of_emitted_context`).
    from .overlay import facet_of_emitted_context as _fac
    facets: dict[str, int] = {}
    n_ctx = 0
    for e in (sp.evidence_ledger or []):
        ctx = str(e.get("context") or "").strip()
        if not ctx:
            continue
        n_ctx += 1
        _k = _fac(ctx)
        facets[_k] = facets.get(_k, 0) + 1
    for name in ("measure", "neg", "sign", "course", "therapy", "pool"):
        f[f"ctx_facet:{name}"] = float(facets.get(name, 0))
    f["ctx_facet:course_frac"] = float(facets.get("course", 0)) / max(1, n_ctx)
    f["n_context"] = float(n_ctx)
    # Presence bits for every known stream: driver-proxy exclusion makes absence informative.
    from .events_streams import METRIC_BY_NAME
    for name in sorted(set(METRIC_BY_NAME) | {"weight_ref", "dose_timeline",
                                              "medication_adherence"}):
        f[f"{name}:present"] = 1.0 if name in ld else 0.0

    # ---- demographics / device inventory / stream length ----
    # A5v2 rejects shortcuts on these, so they must be scanned. `getattr`, because some callers
    # pass only longitudinal_data.
    up = (getattr(sp, "user_profile", None) or {})
    age = str(up.get("age_range") or "")
    if "-" in age:                                   # "35-39" -> 35.0 (lower bound; monotonic is enough to scan)
        try:
            f["demo:age_lo"] = float(age.split("-")[0])
        except ValueError:
            pass
    f["demo:sex_is_f"] = 1.0 if str(up.get("sex", "")).upper().startswith("F") else 0.0
    f["demo:n_conditions"] = float(len(up.get("known_conditions") or []))
    # Per-disease bits: `demo:n_conditions` is constant when every case has one disease.
    # Existence bits, not an ordinal, since diseases have no order.
    _kc = {str(c).strip().lower() for c in (up.get("known_conditions") or [])}
    for _d in A5_DISEASE_BITS:
        f[f"cond:{_d}"] = 1.0 if _d.lower() in _kc else 0.0
    f["demo:n_goals"] = float(len(up.get("treatment_goals") or []))
    dev = up.get("device_inventory") or []
    f["demo:n_devices"] = float(len(dev))
    # Per-device bits: the device combination, not only the count.
    for d in ("smart_scale", "wearable", "bp_cuff", "cgm", "clinic_scale"):
        f[f"device:{d}"] = 1.0 if d in dev else 0.0
    for name, pts in ld.items():
        f[f"{name}:n_points"] = float(len([p for p in (pts or []) if int(p["ts"]) <= T]))
    return f


class _ThresholdScan:
    """The candidate thresholds of one feature, prepared once so that each label vector costs a
    single pass. Thresholds, their order and the comparisons (`x > t`, `x <= t` against the
    real midpoint float) are those of the plain per-threshold rescan; only the counting differs.
    """

    def __init__(self, xs: list[float]):
        from bisect import bisect_right
        cand = sorted(set(xs))
        self.mids = [(a + b) / 2 for a, b in zip(cand, cand[1:])] or cand
        order = sorted(range(len(xs)), key=xs.__getitem__)
        sx = [xs[i] for i in order]
        self.order = order
        self.n = len(xs)
        self.cnt_le = [bisect_right(sx, t) for t in self.mids]     # number of x <= t

    def best_index(self, ys) -> tuple[float, int, str]:
        """`(best F1, index of its threshold, op)`; the first threshold reaching the best F1
        wins, `>` before `<=` at one threshold."""
        pref = [0]
        acc = 0
        for i in self.order:
            if ys[i]:
                acc += 1
            pref.append(acc)
        k, n = acc, self.n
        best, bi, bop = 0.0, -1, ""
        for j, c in enumerate(self.cnt_le):
            pos_le = pref[c]
            pos_gt = k - pos_le
            # ">": tp = pos_gt, fp = n - c - pos_gt, fn = pos_le
            if pos_gt:
                f1 = 2 * pos_gt / (2 * pos_gt + (n - c - pos_gt) + pos_le)
                if f1 > best:
                    best, bi, bop = f1, j, ">"
            # "<=": tp = pos_le, fp = c - pos_le, fn = pos_gt
            if pos_le:
                f1 = 2 * pos_le / (2 * pos_le + (c - pos_le) + pos_gt)
                if f1 > best:
                    best, bi, bop = f1, j, "<="
        return best, bi, bop

    def best(self, ys) -> tuple[float, str]:
        f1, bi, op = self.best_index(ys)
        return (f1, f"x {op} {round(self.mids[bi], 4)}") if bi >= 0 else (0.0, "")


def _best_single_threshold_f1(xs: list[float], ys: list[bool]) -> tuple[float, str]:
    """The best single-feature threshold rule's F1 (tries both directions)."""
    if any(x != x for x in xs):                        # NaN: comparisons are not orderable
        return _best_single_threshold_f1_scan(xs, ys)
    return _ThresholdScan(xs).best(ys)


def _best_single_threshold_f1_scan(xs, ys):
    best, how = 0.0, ""
    cand = sorted(set(xs))
    mids = [(a + b) / 2 for a, b in zip(cand, cand[1:])] or cand
    for t in mids:
        for op, pred in ((">", [x > t for x in xs]), ("<=", [x <= t for x in xs])):
            tp = sum(1 for p, y in zip(pred, ys) if p and y)
            fp = sum(1 for p, y in zip(pred, ys) if p and not y)
            fn = sum(1 for p, y in zip(pred, ys) if not p and y)
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + fp + fn)
            if f1 > best:
                best, how = f1, f"x {op} {round(t, 4)}"
    return best, how


def _target_key(tname: str) -> str:
    """A5 target name without its value (`outcome=event_occurred` -> `outcome`)."""
    return tname.split("=", 1)[0]


#: Permutation draws for A5's noise floor (enough for a 95th percentile).
A5_N_PERM = 20


#: A5's family-wise error rate: per-feature exact p times the number of tests run in the
#: tier must be <= alpha (Bonferroni). A permutation floor over the whole family saturates
#: when many correlated features are scanned.
A5_ALPHA = 0.05


#: Beyond this many `C(n,k)` orderings, p is estimated by deterministic sampling.
A5_P_SAMPLES = 4000


_SPLIT_ORDER_CACHE: dict = {}


_SPLIT_ORDER_CACHE_MAX_CELLS = 4_000_000


#: `t` -> `range(m)` sorted by `sha1(i|t)`, for the largest `m` asked so far. The key of `i`
#: does not depend on `n`, so the order for any `n <= m` is this order without the indices
#: `>= n`.
_SPLIT_BASE: dict = {}


def _split_order(n: int, t: int) -> list[int]:
    """Permutation `t` of `range(n)`: indices sorted by `sha1(i|t)`. The same for every feature
    of a tier, so it is computed once per `(n, t)`."""
    key = (n, t)
    got = _SPLIT_ORDER_CACHE.get(key)
    if got is None:
        base = _SPLIT_BASE.get(t)
        if base is None or len(base) < n:
            import hashlib as _h
            base = sorted(range(n), key=lambda i: _h.sha1(f"{i}|{t}".encode()).hexdigest())
            _SPLIT_BASE[t] = base
        got = base if len(base) == n else [i for i in base if i < n]
        if (len(_SPLIT_ORDER_CACHE) + 1) * n > _SPLIT_ORDER_CACHE_MAX_CELLS:
            _SPLIT_ORDER_CACHE.clear()
        _SPLIT_ORDER_CACHE[key] = got
    return got


def _split_p(xs: list, yy: list, f1_obs: float, p_useful: float = 0.0) -> float:
    """Null probability that a single threshold rule reaches `f1_obs`.

    A perfect split has the closed form `2 / C(n, k)`; otherwise deterministic sampling
    (sorted by `sha1(i|t)`). Returns early once the result cannot be below `p_useful`.
    """
    from math import comb
    n, k = len(yy), sum(1 for y in yy if y)
    if k <= 0 or k >= n:
        return 1.0
    p_floor = min(1.0, 2.0 / comb(n, k))        # a perfect split's null probability = the p lower bound for any F1
    if f1_obs >= 1.0 - 1e-12:
        return p_floor
    # Even the lower bound is not significant: skip sampling (exact, changes no verdict).
    if p_useful > 0.0 and p_floor > p_useful:
        return p_floor
    # Early stop once p is certainly above `p_useful` (changes no verdict).
    _stop_at = (int(p_useful * (A5_P_SAMPLES + 1)) + 1) if p_useful > 0.0 else None
    scan = None if any(x != x for x in xs) else _ThresholdScan(xs)
    ge = 0
    for t in range(A5_P_SAMPLES):
        order = _split_order(n, t)
        yp = [yy[i] for i in order]
        f1p = (scan.best_index(yp)[0] if scan is not None
               else _best_single_threshold_f1_scan(xs, yp)[0])
        ge += (f1p >= f1_obs - 1e-12)
        if _stop_at is not None and ge >= _stop_at:
            return (ge + 1) / (t + 2)
    return (ge + 1) / (A5_P_SAMPLES + 1)


#: Nominal minimum case count for an A5 reading to count as evidence (a perfect split across
#: 100+ features can occur by chance in small batches). Blocking itself follows per-tier
#: statistical power (`_powered` in `check_shortcut`); the pipeline does not read this constant.
A5_MIN_CASES = 15


#: A feature's range must reach this fraction of its own magnitude before a split counts;
#: below it the feature is near constant (QC artifacts sit near 5%, real features above 50%).
A5_MIN_REL_SPAN = 0.10


def near_constant(xs: list[float]) -> bool:
    """Whether the feature's range is below `A5_MIN_REL_SPAN` of its mean absolute value.
    Shared with the maintainers' label-independence tool.
    """
    if not xs:
        return True
    _span = max(xs) - min(xs)
    _scale = sum(abs(x) for x in xs) / len(xs)
    return not (_span > 0 and (_span / max(1e-9, _scale)) >= A5_MIN_REL_SPAN)


#: Feature suffixes in the stream's own unit, scaled by its physiological range.
_LEVEL_SUFFIXES = ("last", "min", "max")


def near_constant_feature(name: str, xs: list[float]) -> bool:
    """`near_constant`, with the right scale for level features.

    For `last` / `min` / `max` the span is compared with the width of the stream's
    `hard_range` (half a degree of body temperature is a real separation); other features
    use `near_constant`.
    """
    if not xs:
        return True
    base, _, suffix = name.partition(":")
    if suffix in _LEVEL_SUFFIXES:
        from .events_streams import METRIC_BY_NAME
        m = METRIC_BY_NAME.get(base)
        if m:
            lo, hi = m.hard_range
            width = float(hi) - float(lo)
            if width > 0:
                return not ((max(xs) - min(xs)) / width >= A5_MIN_REL_SPAN)
    return near_constant(xs)


_PERM_CACHE: dict = {}


def _perm_floor(cases: list, tname: str, ys: list, feat_names: list,
                event_type: str | None) -> float:
    """Noise floor: the 95th percentile, over label shuffles, of the best single feature's
    excess over the majority baseline. Deterministic shuffles (`sha1(case_id|k)`).
    """
    import hashlib as _h
    key = (id(cases), tname)
    if key in _PERM_CACHE:
        return _PERM_CACHE[key]
    cids = [c[0] for c in cases]
    nulls: list[float] = []
    for k in range(A5_N_PERM):
        order = sorted(range(len(cases)),
                       key=lambda i: _h.sha1(f"{cids[i]}|{k}".encode()).hexdigest())
        yp = [ys[i] for i in order]
        tp0 = sum(yp)
        if tp0 == 0 or tp0 == len(yp):
            continue
        best = 0.0
        for fn in feat_names:
            xs, yy = [], []
            for (_, f, _, _, _), y in zip(cases, yp):
                if fn in f:
                    xs.append(f[fn]); yy.append(1 if y else 0)
            if len(xs) < 4 or len(set(xs)) < 2 or sum(yy) < A5_MIN_CLASS:
                continue
            if classify_shortcut(tname, fn, event_type=event_type) == "clinical":
                continue
            # `base` on this feature's own subset, as the caller computes `f1_major`.
            tp_f = sum(yy)
            base = 2 * tp_f / (2 * tp_f + (len(yy) - tp_f))
            f1n, _ = _best_single_threshold_f1(xs, yy)
            best = max(best, f1n - base)
        nulls.append(best)
    nulls.sort()
    floor = nulls[int(0.95 * (len(nulls) - 1))] if nulls else 0.0
    _PERM_CACHE[key] = floor
    return floor


def _split_p_of(args: tuple) -> float:
    xs, yy, f1, p_useful = args
    return _split_p(xs, yy, f1, p_useful=p_useful)


def check_shortcut(cases: list, graded_targets: set[str] | None = None,
                   event_type: str | None = None, p_map=None) -> list[dict]:
    """A5. cases = [(case_id, features, gold_driver, outcome_label[, join_gold]), ...].

    For each gold category (driver values, outcome, join_gold values) x feature, fits the best
    single threshold rule and reports F1 >= A5_F1 with significance (`A5_ALPHA`).

    `graded_targets`: hits on targets outside this set get `class="ungraded"` -- reported,
    never blocked. `None` scores everything; `cli.py` exempts outcome only when every case
    carries `outcome_rule_not_applicable`.

    `p_map(fn, items)` returns `[fn(x) for x in items]` in order; the permutation p-values,
    which are independent per (target, feature), are computed through it. Default: serial.
    """
    hits: list[dict] = []
    cases = [tuple(c) + (None,) * (5 - len(c)) for c in cases]
    if len(cases) < 4:
        return hits
    feat_names = sorted(set().union(*[set(f) for _, f, _, _, _ in cases]))
    targets: list[tuple[str, list[bool]]] = []
    def _usable(ys: list[bool]) -> bool:
        # Smaller side >= A5_MIN_CLASS: a 1-vs-19 split is an outlier, not a shortcut.
        return min(sum(ys), len(ys) - sum(ys)) >= A5_MIN_CLASS

    # Record skipped targets, so a constant-gold tier reads as "not scanned", not "clean".
    _skipped: list[str] = []

    def _add(name: str, ys: list[bool]) -> None:
        if _usable(ys):
            targets.append((name, ys))
        else:
            _skipped.append(name)

    for g in sorted({g for _, _, g, _, _ in cases if g}):
        _add(f"driver={g}", [gd == g for _, _, gd, _, _ in cases])
    _add("outcome=event_occurred", [o == "event_occurred" for _, _, _, o, _ in cases])
    for j in sorted({j for _, _, _, _, j in cases if j}):
        _add(f"join_gold={j}", [jg == j for _, _, _, _, jg in cases])

    # Report a gold family that was never scanned -- only if it has values and is scored for
    # this question type.
    for _fam in ("driver", "outcome", "join_gold"):
        if any(t.startswith(_fam) for t, _ in targets):
            continue
        _n = sum(1 for s in _skipped if s.startswith(_fam))
        if not _n or (graded_targets is not None and _fam not in graded_targets):
            continue
        _why = f"all {_n} targets were skipped because the smaller side < {A5_MIN_CLASS} (gold is nearly constant)"
        hits.append({
            # Explicit class: `cli.py` treats a missing class as nonclinical, which would block.
            "kind": "shortcut_target_unscannable", "severity": "warn",
            "class": "unscannable",
            "target_family": _fam, "n_targets_skipped": _n,
            "detail": (f"`{_fam}` tier: {_why} => A5 scanned nothing at all on this dimension. "
                       f"Do not read this batch's \"no {_fam} shortcut\" as having been scanned."),
        })

    # Bonferroni denominator: tests actually run per target. The filter must stay identical to
    # the loop below.
    _n_tests: dict[str, int] = {}
    for tname, ys in targets:
        _c = 0
        for fn in feat_names:
            xs, yy = [], []
            for (_, f, _, _, _), y in zip(cases, ys):
                if fn in f:
                    xs.append(f[fn]); yy.append(y)
            if len(xs) < 4 or sum(yy) < A5_MIN_CLASS or len(set(xs)) < 2:
                continue
            _c += 1
        _n_tests[tname] = max(1, _c)

    _tests = []
    for tname, ys in targets:
        for fn in feat_names:
            xs, yy = [], []
            for (_, f, _, _, _), y in zip(cases, ys):
                if fn in f:
                    xs.append(f[fn]); yy.append(y)
            if len(xs) < 4 or sum(yy) < A5_MIN_CLASS or len(set(xs)) < 2:
                continue
            f1, how = _best_single_threshold_f1(xs, yy)
            _tests.append((tname, fn, xs, yy, f1, how))
    _p_args = [(xs, yy, f1, A5_ALPHA / max(1, _n_tests.get(tname, 1)))
               for tname, _, xs, yy, f1, _ in _tests]
    _ps = (p_map or (lambda g, it: [g(x) for x in it]))(_split_p_of, _p_args)

    for (tname, fn, xs, yy, f1, how), _p in zip(_tests, _ps):
        # Baseline: predict positive for everything.
        tp0 = sum(yy)
        f1_major = 2 * tp0 / (2 * tp0 + (len(yy) - tp0)) if tp0 else 0.0
        _eff_ok = not near_constant_feature(fn, xs)
        # Candidates that are not significant are still reported. `_powered` is False when even a
        # perfect split could not reach significance ("unmeasurable", not "clean").
        _ntest = _n_tests.get(tname, 1)
        _padj = min(1.0, _p * _ntest)
        from math import comb as _comb
        _kk = sum(1 for y in yy if y)
        _pmin = min(1.0, (2.0 / _comb(len(yy), _kk)) * _ntest) if 0 < _kk < len(yy) else 1.0
        _powered = _pmin <= A5_ALPHA
        _sig = _padj <= A5_ALPHA
        _cand = f1 >= max(A5_F1, f1_major + A5_MARGIN_OVER_MAJORITY) and _eff_ok
        cls = classify_shortcut(tname, fn, event_type=event_type)
        if graded_targets is not None and _target_key(tname) not in graded_targets:
            cls = "ungraded"
        if _cand and not _sig:
            # Not significant: reported with class "chance", which never blocks.
            hits.append({
                "kind": "shortcut_not_significant", "severity": "info",
                "class": "chance", "target": tname, "feature": fn,
                "f1": round(f1, 3), "powered": _powered, "p_adj": round(_padj, 5),
                "detail": (f"[{cls}·not significant] {tname} is solved by `{fn}` to F1={round(f1, 3)}"
                           f"({how})· p={_p:.2e} × {_ntest} tests = {_padj:.3f} > α={A5_ALPHA}"
                           f"(n={len(xs)}, positives {sum(yy)}) -- "
                           + (f"this tier has no power: even a perfect split would, after correction, reach only "
                              f"{_pmin:.3f} at best => recorded as \"unpowered\", not read as \"no shortcut\""
                              if not _powered else "indistinguishable from what shuffled labels can already achieve"))})
        if _cand and _sig:
            # A clinical feature solving the question to F1 >= 0.95 is reported as "too easy".
            if cls == "clinical" and f1 >= 0.95:
                hits.append({"kind": "shortcut_clinical_saturated", "severity": "warn",
                             "class": "clinical", "target": tname, "feature": fn,
                             "f1": round(f1, 3),
                             "detail": f"[clinical·saturated] {tname} can be solved by a single clinical feature "
                                       f"`{fn}` to F1={round(f1, 3)}({how}) -- "
                                       f"legitimate but too easy: this question needs no cross-time synthesis"})
            hits.append({"kind": "shortcut_solvable", "severity": A5_SEVERITY,
                         "class": cls,          # clinical = reported, not blocked / nonclinical = rejects emission
                         "target": tname, "feature": fn, "f1": round(f1, 3),
                         "detail": f"[{cls}] {tname} can be solved by the single feature `{fn}`: {how} · F1={round(f1, 3)}"
                                   f"(n={len(xs)}, positives {sum(yy)}, "
                                   f"predict-all-positive baseline F1={round(f1_major, 3)})"})
    return hits


#: Chance-corrected floor (|phi| = |MCC|) for path B, ANDed with `A5_F1`: under class
#: imbalance an uninformative footprint can still reach F1 >= 0.9, while phi stays 0.
FOOTPRINT_PHI_FLOOR = 0.2


#: Path A's phi floor: only rules out uninformative footprints whose precision is high from
#: class imbalance alone (3 real / 30 benign, all footprinted: precision 0.909, phi 0.0).
FOOTPRINT_PHI_NONZERO = 0.05


def footprint_discriminability(rows: list[tuple[bool, bool]]) -> dict:
    """`[(is_real_symptom, has_physio_footprint)]` -> can the footprint identify true symptoms.

    Returns `{n_real, n_benign, real_cov, benign_cov, f1, precision, phi, direction, usable}`;
    `precision` and `f1` take the better direction, `phi` is the Matthews coefficient.
    Precision matters because this shortcut needs no recall.
    """
    real = [fp for is_real, fp in rows if is_real]
    benign = [fp for is_real, fp in rows if not is_real]
    n_r, n_b = len(real), len(benign)
    out = {"n_real": n_r, "n_benign": n_b,
           "real_cov": (sum(real) / n_r) if n_r else None,
           "benign_cov": (sum(benign) / n_b) if n_b else None,
           "f1": None, "precision": None, "phi": None, "direction": None,
           "usable": min(n_r, n_b) >= A5_MIN_CLASS}
    if not out["usable"]:
        return out
    best_f1, best_p, best_dir = 0.0, 0.0, None
    for pos_is_real in (True, False):
        tp = sum(1 for is_real, fp in rows if fp and (is_real == pos_is_real))
        fp_ = sum(1 for is_real, fp in rows if fp and (is_real != pos_is_real))
        fn = sum(1 for is_real, fp in rows if not fp and (is_real == pos_is_real))
        f1 = 0.0 if not tp else 2 * tp / (2 * tp + fp_ + fn)
        prec = 0.0 if not (tp + fp_) else tp / (tp + fp_)
        best_f1 = max(best_f1, f1)
        if prec > best_p:
            best_p, best_dir = prec, "footprint=>true symptom" if pos_is_real else "footprint=>benign event"
    out["f1"] = round(best_f1, 4)
    out["precision"], out["direction"] = round(best_p, 4), best_dir
    # phi is direction-independent; an empty row or column means zero discriminability.
    a = sum(1 for is_real, fp in rows if is_real and fp)          # real & footprinted
    b = sum(1 for is_real, fp in rows if is_real and not fp)      # real & no footprint
    c = sum(1 for is_real, fp in rows if not is_real and fp)      # benign & footprinted
    d = sum(1 for is_real, fp in rows if not is_real and not fp)  # benign & no footprint
    den = (a + b) * (c + d) * (a + c) * (b + d)
    out["phi"] = 0.0 if den == 0 else round(abs((a * d - b * c) / den ** 0.5), 4)
    return out


def check_footprint_not_discriminative(built: dict) -> list[dict]:
    """Batch-level: "has a physiological footprint" must not identify true symptoms.

    Kernel footprints attach by `topic`, which only benign and life events carry, so a
    footprint could mark an event as benign. The check fits `(is_real, has_footprint)` pairs
    across the batch. It fails when one side has footprints and the other none, or on path A
    or B (see `FOOTPRINT_PHI_NONZERO` / `FOOTPRINT_PHI_FLOOR`). A single class, zero
    footprints and a batch where the physiology layer did not run are reported, not passed.
    """
    from .physio import kernel as _pk
    from . import wq as _wq

    try:
        _kt = _pk.load_physio_registry(_REG_DIR / "physio_kernels.yaml")
        emitted = set(_kt.emitted_topics())
    except Exception as e:                                         # noqa: BLE001
        return [{"kind": "footprint_registry_unreadable", "severity": "gate",
                 "detail": f"{type(e).__name__}: {e}"}]

    # First check that the physiology layer ran (a `physio` report in the Q-side ledger): the
    # kernel table alone only says which footprints could exist.
    ran = [str(cid) for cid in built
           if (_wq.injected_manifest(str(cid), required=False) or {}).get("physio")]
    if not ran:
        return [{"kind": "footprint_scan_physio_off", "severity": "info",
                 "detail": f"the physiology layer did not run (none of this batch's {len(built)} cases has a `physio` report in the Q-side ledger)"
                           f" => not a single footprint exists, this judge has no object (not read as \"passed\")"}]

    rows: list[tuple[bool, bool]] = []
    for cid in built:
        man = _wq.injected_manifest(str(cid), required=False) or {}
        if not man.get("physio"):
            continue  # cases outside the physiology layer are left out
        real_ids = {str(x) for x in (man.get("real_symptom_evidence_ids") or [])}
        for s in (man.get("event_schedule") or []):
            eid = str(s.get("evidence_id", ""))
            kind = str(s.get("kind", ""))
            # Patient self-reports only.
            if kind not in ("real_symptom", "benign_symptom", "life_event", "lookalike"):
                continue
            rows.append((eid in real_ids or kind == "real_symptom",
                         str(s.get("topic") or "") in emitted))

    v = footprint_discriminability(rows)
    base = (f"true symptoms {v['n_real']} (footprint rate {v['real_cov']})· "
            f"benign/distractor {v['n_benign']} (footprint rate {v['benign_cov']})")
    # No footprints on either side: not active, reported.
    if not any(fp for _, fp in rows):
        return [{"kind": "footprint_scan_no_footprints", "severity": "warn",
                 "detail": f"{base} -- zero footprints across the whole batch: the kernel table has no object among the topics sampled in this batch, "
                           f"this judge cannot judge this case (not read as \"no shortcut installed\")"}]
    # Footprints on exactly one side: a perfect discriminator in either direction.
    _rc, _bc = (v["real_cov"] or 0), (v["benign_cov"] or 0)
    if v["n_real"] and v["n_benign"] and (_rc > 0) != (_bc > 0):
        _side = "the true-symptom side has footprints and the benign/distractor side has none at all" if _rc > 0 else "the benign side has footprints and the true-symptom side has none at all"
        _rule = "\"has a footprint => true symptom\"" if _rc > 0 else "\"has a footprint => benign\""
        return [{"kind": "footprint_discriminates_real_symptom", "severity": "gate",
                 "detail": f"perfect discriminator: {base} -- {_side} => {_rule} is always true on this batch. "
                           f"The kernel can only attach to events carrying a `topic`, and whichever side has no topic becomes the discriminating feature"}]
    if not v["usable"]:
        return [{"kind": "footprint_scan_single_class", "severity": "warn",
                 "detail": f"{base} -- only one class has samples (threshold {A5_MIN_CLASS}), this batch is unmeasured. "
                           f"`early_warning` has zero true symptoms per case, which is the norm on that line"}]
    # Two paths (either fails the batch):
    #   A: `precision >= A5_F1` AND `|phi| >= FOOTPRINT_PHI_NONZERO`
    #   B: `F1 >= A5_F1`  AND `|phi| >= FOOTPRINT_PHI_FLOOR`
    # A catches precise rules with low recall; its phi term removes class-imbalance artifacts.
    _p, _f1, _phi = (v["precision"] or 0), (v["f1"] or 0), (v["phi"] or 0)
    _hit_a = _p >= A5_F1 and _phi >= FOOTPRINT_PHI_NONZERO
    _hit_b = _f1 >= A5_F1 and _phi >= FOOTPRINT_PHI_FLOOR
    if _hit_a or _hit_b:
        _why = (f"precision={_p} >= {A5_F1} and |φ|={_phi} >= {FOOTPRINT_PHI_NONZERO}" if _hit_a
                else f"F1={_f1} >= {A5_F1} and |φ|={_phi} >= {FOOTPRINT_PHI_FLOOR}")
        return [{"kind": "footprint_discriminates_real_symptom", "severity": "gate",
                 "detail": f"{_why}({v['direction']})· {base}"}]
    return [{"kind": "footprint_scan_ok", "severity": "info",
             "detail": f"precision={v['precision']}(threshold {A5_F1})· F1={v['f1']}(threshold {A5_F1})"
                       f"· |φ|={v['phi']}(threshold A {FOOTPRINT_PHI_NONZERO} / B {FOOTPRINT_PHI_FLOOR})"
                       f"· {v['direction']} · {base}"}]


# ============================================================ GEN8t: the ddx tier must not be readable off surface quantities (batch-level)
# `joint_dx` packs mix a sufficient tier and an insufficient tier (the correct answer is to
# abstain). Features with no diagnostic content (T, report counts and lengths, lab-flag
# direction and item profile) must not separate them: a hit needs
# `|AUC - 0.5| >= TIER_SURFACE_AUC_MARGIN` and a Bonferroni-corrected p < TIER_SURFACE_ALPHA.
# Abnormality magnitude is not scanned: on the sufficient tier it is the gold's evidence.
TIER_SURFACE_AUC_MARGIN = 0.15


TIER_SURFACE_ALPHA = 0.01


TIER_SURFACE_MIN_CLASS = 5


TIER_SURFACE_SEVERITY = "gate"


#: Permutations for the item-profile p: the smallest p, 1 / (n + 1), must clear the
#: corrected alpha.
TIER_SURFACE_N_PERM = 3000


_VARIANT_SUFFIX_RE = re.compile(r"v\d+$")


def _face_labs(raw) -> tuple[int, list[dict], list[dict]]:
    T = int((getattr(raw, "prediction_context", None) or {}).get("prediction_time_T") or 0)
    led = [e for e in (getattr(raw, "evidence_ledger", None) or [])
           if int(e.get("source_timestamp", 10 ** 9)) <= T]
    return T, led, [e for e in led if str(e.get("source_type", "")) == "lab_result"]


def _lab_flags(labs: list[dict]) -> list[str]:
    """`item:H` / `item:L` for every out-of-range quantitative line (item = the
    printed name, the first token of the line)."""
    out = []
    for e in labs:
        s = str(e.get("symptom", ""))
        m = _LAB_VAL_RE.search(s)
        if not m:
            continue
        try:
            v, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
        except ValueError:
            continue
        if v > hi or v < lo:
            out.append(f"{s.split(' ')[0]}:{'H' if v > hi else 'L'}")
    return out


def tier_surface_features(raw) -> dict[str, float]:
    """Solver-visible (`<= T`) quantities of one case that carry no diagnostic
    content. See the GEN8t note above for why each is here."""
    T, led, labs = _face_labs(raw)
    flags = _lab_flags(labs)
    n_abn = len(flags)
    n_low = sum(1 for x in flags if x.endswith(":L"))
    days = sorted({int(e.get("source_timestamp", 0)) for e in labs})
    reports = [str(e.get("symptom", "")) for e in led
               if str(e.get("source_type", "")) == "patient_reported_symptom"]
    f = {"prediction_time_T": float(T),
         "lab_any_abnormal": float(n_abn > 0),
         "lab_n_abnormal": float(n_abn),
         "lab_any_low": float(n_low > 0),
         "n_patient_reports": float(sum(1 for e in led if str(e.get("source_type", ""))
                                        .startswith("patient_reported"))),
         "n_ledger": float(len(led)),
         "case_id_variant_suffix": float(bool(_VARIANT_SUFFIX_RE.search(
             str(getattr(raw, "case_id", "") or "")))),
         }
    if n_abn:
        f["lab_low_share"] = n_low / n_abn
    if reports:
        f["pr_mean_len"] = sum(len(t) for t in reports) / len(reports)
    if days and T > 0:
        f["lab_first_draw_over_T"] = days[0] / T
        f["lab_T_minus_last_draw"] = float(T - days[-1])
    return f


def _item_profile_scores(bags: list[set], y: list[bool]) -> list[float]:
    """Leave-one-out Bernoulli naive-Bayes log-odds of `y` (Laplace-smoothed) from
    each case's set of `item:direction` flags."""
    feats = sorted(set().union(*bags)) if bags else []
    n = {True: sum(y), False: len(y) - sum(y)}
    cnt = {True: collections.Counter(), False: collections.Counter()}
    for b, t in zip(bags, y):
        cnt[t].update(b)
    out = []
    for b, t in zip(bags, y):
        nn = {k: n[k] - (k == t) for k in (True, False)}
        s = math.log((nn[True] + 1) / (nn[False] + 1))
        for x in feats:
            c1 = cnt[True][x] - (t and x in b)
            c0 = cnt[False][x] - ((not t) and x in b)
            p1, p0 = (c1 + 1) / (nn[True] + 2), (c0 + 1) / (nn[False] + 2)
            s += math.log(p1 / p0) if x in b else math.log((1 - p1) / (1 - p0))
        out.append(s)
    return out


def _auc_of(scores: list[float], y: list[bool]) -> float:
    return _mann_whitney([s for s, t in zip(scores, y) if t],
                         [s for s, t in zip(scores, y) if not t])[0]


def _mann_whitney(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """(AUC = P(pos > neg) + P(tie)/2, two-sided p by the tie-corrected normal
    approximation)."""
    n1, n2 = len(pos), len(neg)
    allv = sorted((v, i) for i, v in enumerate(list(pos) + list(neg)))
    ranks = [0.0] * (n1 + n2)
    ties = 0.0
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[allv[k][1]] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    u = sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    auc = u / (n1 * n2)
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return auc, 1.0
    z = (u - n1 * n2 / 2.0) / math.sqrt(var)
    return auc, math.erfc(abs(z) / math.sqrt(2.0))


def check_tier_surface(built: dict) -> list[dict]:
    """GEN8t. built = {case_id: RawCase}. Reports every tier-neutral surface
    feature that separates the sufficient and insufficient ddx tiers."""
    tiers: dict[bool, list[dict]] = {True: [], False: []}
    bags: list[set] = []
    suff: list[bool] = []
    for r in built.values():
        ddx = (getattr(r, "adjudication", None) or {}).get("ddx")
        if not ddx:
            continue
        tiers[bool(ddx.get("insufficient"))].append(tier_surface_features(r))
        bags.append(set(_lab_flags(_face_labs(r)[2])))
        suff.append(not ddx.get("insufficient"))
    n_ins, n_suf = len(tiers[True]), len(tiers[False])
    if not n_ins or not n_suf:
        return []                     # a one-tier pack has no tier to read off
    if min(n_ins, n_suf) < TIER_SURFACE_MIN_CLASS:
        return [{"kind": "tier_surface_unscannable", "severity": "info",
                 "detail": f"sufficient tier {n_suf} cases · insufficient tier {n_ins} cases, smaller side < "
                           f"{TIER_SURFACE_MIN_CLASS} => GEN8t not scanned on this batch (not a pass)"}]
    names = sorted(set().union(*tiers[True], *tiers[False]))
    tested: list[tuple[str, float, float]] = []
    for fn in names:
        pos = [f[fn] for f in tiers[False] if fn in f]
        neg = [f[fn] for f in tiers[True] if fn in f]
        if min(len(pos), len(neg)) < TIER_SURFACE_MIN_CLASS or len(set(pos + neg)) < 2:
            continue
        auc, p = _mann_whitney(pos, neg)
        tested.append((fn, auc, p))
    # Item identity: LOO naive Bayes over `item:direction` flags, p by permutation.
    if any(bags):
        import random as _random
        auc_nb = _auc_of(_item_profile_scores(bags, suff), suff)
        k = len(tested) + 1
        # Stop once p can no longer clear the corrected alpha.
        give_up = TIER_SURFACE_ALPHA / k * (TIER_SURFACE_N_PERM + 1)
        rr, yp, ge, done = _random.Random(0), list(suff), 0, 0
        for done in range(1, TIER_SURFACE_N_PERM + 1):
            rr.shuffle(yp)
            # One-sided: leave-one-out fits under shuffled labels are biased below 0.5.
            ge += _auc_of(_item_profile_scores(bags, yp), yp) >= auc_nb
            if ge + 1 >= give_up:
                break
        p_nb = (ge + 1) / (done + 1)
        tested.append(("lab_item_profile_nb", auc_nb, p_nb))
    hits: list[dict] = []
    for fn, auc, p in tested:
        p_adj = min(1.0, p * len(tested))
        if abs(auc - 0.5) >= TIER_SURFACE_AUC_MARGIN and p_adj < TIER_SURFACE_ALPHA:
            hits.append({"kind": "tier_surface_separable", "severity": TIER_SURFACE_SEVERITY,
                         "feature": fn, "auc": round(auc, 3), "p_adj": p_adj,
                         "detail": f"`{fn}` alone separates the diagnostic tier: AUC(sufficient>insufficient) = {auc:.3f}"
                                   f" (sufficient {n_suf} · insufficient {n_ins}, Bonferroni p = {p_adj:.2g})"
                                   f" -- this quantity carries no diagnostic content; the tier must not be readable from it"})
    return hits
