"""ddx.py -- turns the kernel's `joint_scenarios.DDX_SPECS` (plus haenv's added
conditions, via `overlay.condition_registry`) into joint_dx job.yaml cases, sampling per-case
demographics, timing, density and tiers. Spec conversion only; it invents no clinical facts.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import logging
import math
import pathlib

log = logging.getLogger("haenv.ddx")


def _tier(path: str):
    """Tier mix ratio at `registry/tiers.yaml:authoritative.<path>`; a missing key raises.
    ddx.py is in neither freeze segment, so these values reach cases through the job yaml
    (`job_sha256`)."""
    from . import data_root as _dr
    from .yamlcache import load_yaml
    cur = load_yaml(_dr() / "registry" / "tiers.yaml") or {}
    cur = (cur.get("authoritative") or {})
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            raise KeyError(f"registry/tiers.yaml:authoritative has no `{path}` —— "
                           "tier mix ratios have no second source; add it there instead of "
                           "falling back in code")
        cur = cur[seg]
    return cur


#: Textbook sex ratios P(male), per `registry/condition_sex_ratio.yaml`. Anatomically
#: exclusive terms in the symptom text take precedence (`_sex_of`); composite `HD-COM-*`
#: ratios are computed by odds-product, never hand-written.
def _sex_ratio_table() -> dict[str, tuple[float, str]]:
    """`{spec_id: (male fraction, rationale)}`; composite tiers are computed, not stored."""
    from .registry import _load_yaml
    from .yamlcache import load_yaml as _cached
    from .regpath import registry_path as _rp
    doc = _cached(_rp("condition_sex_ratio.yaml")) or {}
    base = float(doc.get("benign_baseline", 0.5))
    out: dict[str, tuple[float, str]] = {
        str(k): (float(v["male"]), str(v.get("why") or ""))
        for k, v in (doc.get("atomic") or {}).items()}
    # ---- Composite tier: odds-product, components from composition_comorbid.yaml ----
    for sid, spec in (_load_yaml("composition_comorbid.yaml").get("pairs") or {}).items():
        pair = list(spec.get("pair") or ())
        if len(pair) != 2 or any(p not in out for p in pair):
            continue                      # component not registered: leave it out rather than invent a ratio
        a, b = out[pair[0]][0], out[pair[1]][0]
        den = a * b + (1.0 - a) * (1.0 - b)
        m = (a * b / den) if den else base
        out[str(sid)] = (round(m, 4),
                         f"组合档**算出来的**:{pair[0]}({a}) × {pair[1]}({b}) "
                         f"→ odds-product {round(m, 4)}")
    return out


#: Target gap-tier share of the eligible pool (= `LONG_FRAC * GAP_WITHIN_LONG`; `registry/tiers.yaml` checks the identity).
GAP_FRAC = _tier("tiers.rhythm_gap.frac_of_pool")          # registry/tiers.yaml

#: Long-horizon share of the eligible pool; the gap tier is drawn inside it, so
#: `LONG_FRAC * GAP_WITHIN_LONG` ~= `GAP_FRAC` and the rest is a matched control arm.
LONG_FRAC = _tier("tiers.long_horizon.frac")               # registry/tiers.yaml

#: Gap share within the long-horizon tier; 0.5 gives equal gap and control arms.
GAP_WITHIN_LONG = _tier("tiers.rhythm_gap.frac_within_long")   # registry/tiers.yaml

#: Long-horizon index time (~1 year), a value already in `_t_index_for`'s domain.
LONG_T = _tier("tiers.long_horizon.index_time_T")          # registry/tiers.yaml
#: Insufficient-tier share of eligible variants, sized so the emitted count can resolve
#: an effect near 1.0 after emission losses. The tier sets `dx_applicable = False`, so a
#: larger share shrinks the pool `dx_hit` is judged on.
INSUFFICIENT_FRAC = _tier("tiers.insufficient.frac")   # registry/tiers.yaml


def long_horizon_for(case_id: str, spec: dict, long_frac: float = LONG_FRAC) -> bool:
    """Whether this case is promoted to a long horizon (T=336): yellow/green urgency, no red
    flag, drawn with salt `|longT`. The gap tier is drawn inside this subset with its own salt,
    so "long horizon, no gap" cases form a matched control arm."""
    import hashlib as _h

    if bool(spec.get("red_flag", False)):
        return False
    if str(spec.get("urgency") or "").strip() not in ("🟡", "🟢"):
        return False
    _x = int(_h.sha256(f"{case_id}|longT".encode()).hexdigest()[:8], 16)
    return (_x % 1000) < int(float(long_frac) * 1000)


#: Minimum observation points at or before T for the insufficient tier.
MIN_PRE_T_POINTS = 5


def insufficient_tier_for(case_id: str, v: int, T: int,
                          measure_per_week: float,
                          frac: float = INSUFFICIENT_FRAC, *,
                          base_id: str | None = None, n_variants: int = 2) -> bool:
    """Whether this case enters the insufficient-information tier (the disease starts after
    `T`, so the face carries no discriminating signal and the right answer is to abstain).

    `T` keeps its normal draw and only the symptom days move (`insufficient_symptom_days`), so
    `T` is not a tier marker. Eligibility: at most one hashed variant per pair `(1, 2)`,
    `(3, 4)`, ... (every condition keeps a sufficient case), and the window must hold
    `MIN_PRE_T_POINTS` observations at the case's density. Salt `|insuff`."""
    import hashlib as _h

    v, n_variants = int(v), int(n_variants)
    lo = v - (v - 1) % 2                          # first variant of v's pair
    if v < 1 or lo + 1 > n_variants:
        return False
    _b = str(base_id if base_id is not None else case_id)
    _pick = int(_h.sha256(f"{_b}|insuff-pair{lo}".encode()).hexdigest()[:8], 16) % 2
    if v != lo + _pick:
        return False
    mpw = max(0.1, float(measure_per_week or 7))
    t_min = math.ceil(MIN_PRE_T_POINTS * 7 / mpw)
    if int(T) < t_min:
        return False
    _x = int(_h.sha256(f"{case_id}|insuff".encode()).hexdigest()[:8], 16)
    return (_x % 1000) < int(float(frac) * 1000)


def insufficient_symptom_days(sym_days: list[int], T: int, course_end: int) -> list[int]:
    """Insufficient-tier symptom days: the case's own schedule shifted by exactly `T` (so
    `presentation_day` can undo it), compressed if needed to end before the course does,
    keeping at least `_SYM_MIN_GAP` between symptoms."""
    if not sym_days:
        return []
    first = int(T) + int(sym_days[0])
    room = max(0, int(course_end) - 3 - first)
    span = int(sym_days[-1]) - int(sym_days[0])
    sc = 1.0 if span <= room else room / span
    out: list[int] = []
    for d in sym_days:
        x = first + int(round((int(d) - int(sym_days[0])) * sc))
        if out:
            x = max(x, out[-1] + _SYM_MIN_GAP)
        out.append(x)
    # Raises rather than place a symptom after the course ends.
    if len(out) > 1 and out[-1] > int(course_end) - 3:
        raise ValueError(f"insufficient_symptom_days: {len(out)} symptoms from day {first} "
                         f"need {out[-1] - first} days, course ends {course_end} "
                         f"(room {room})")
    return out


#: Urgency of an insufficient-tier case whose red-flag presentation lies after `T`:
#: the missing information itself calls for follow-up.
INSUFFICIENT_HIDDEN_RED_FLAG_URGENCY = "🟡"


def insufficient_triage(spec: dict, sym_days: list[int], T: int, insufficient: bool) -> tuple[bool, str]:
    """(red_flag, urgency) of a case. The insufficient tier places every symptom after `T`
    (`insufficient_symptom_days`), so a red-flag spec there is not a red flag at `T`:
    (False, the follow-up urgency). `wq.derive_from_registry` applies the same rule to the
    written world. A symptom at or before `T` on this tier breaks that premise and raises."""
    rf = bool(spec.get("red_flag", False))
    if insufficient and rf:
        if any(int(d) <= int(T) for d in sym_days):
            raise ValueError(f"insufficient tier with a symptom at or before T={T}: {sym_days}")
        return False, INSUFFICIENT_HIDDEN_RED_FLAG_URGENCY
    return rf, spec["urgency"]


def presentation_day(onset_day: int | None, T: int) -> int | None:
    """Day the initial workup is anchored on: the onset, or `onset - T` on the insufficient
    tier, so both tiers draw it from one distribution."""
    if onset_day is None:
        return None
    o = int(onset_day)
    return o - int(T) if o > int(T) else o


def gap_tier_for(case_id: str, spec: dict, T: int,
                 gap_frac: float = GAP_WITHIN_LONG) -> bool:
    """Whether this case enters the information-gap tier -- the single eligibility rule.
    Only yellow/green, non-red-flag, long-horizon cases whose `T` can hold a gap
    (`evaluate.rhythm_gap_feasible`) are drawn, at `gap_frac`, salt `|gap`."""
    import hashlib as _h

    from .rhythm import rhythm_gap_feasible
    if bool(spec.get("red_flag", False)):
        return False
    if str(spec.get("urgency") or "").strip() not in ("🟡", "🟢"):
        return False
    if not rhythm_gap_feasible(int(T)):
        return False
    if not long_horizon_for(case_id, spec):
        return False
    _x = int(_h.sha256(f"{case_id}|gap".encode()).hexdigest()[:8], 16)
    return (_x % 1000) < int(float(gap_frac) * 1000)


def strip_inert_gap(cases, slices) -> int:
    """Drop `rhythm_gap` declarations when the job does not slice with `+gap` (the pack could
    not deliver them); returns how many were dropped."""
    if "+gap" in str(slices or ""):
        return 0
    n = 0
    for c in cases:
        if (c.get("latent") or {}).pop("rhythm_gap", None) is not None:
            n += 1
    return n


def _benign_baseline() -> float:
    from .yamlcache import load_yaml as _cached
    from .regpath import registry_path as _rp
    return float((_cached(_rp("condition_sex_ratio.yaml")) or {}).get(
        "benign_baseline", 0.5))


#: Exposed lazily via module `__getattr__` (PEP 562) so it is a real dict.
_SEX_SKEW_CACHE: dict[str, tuple[float, str]] | None = None


def __getattr__(name: str):
    """Module-level lazy attribute -- `from haenv.ddx import CONDITION_SEX_SKEW`
    returns a real dict."""
    global _SEX_SKEW_CACHE
    if name == "CONDITION_SEX_SKEW":
        if _SEX_SKEW_CACHE is None:
            _SEX_SKEW_CACHE = _sex_ratio_table()
        return _SEX_SKEW_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _sex_of(spec: dict, default: str = "F", case_id: str | None = None,
            spec_id: str | None = None) -> str:
    """Patient sex for this spec: anatomically exclusive words in the symptom text decide it;
    otherwise sampled from `CONDITION_SEX_SKEW` by `case_id`; otherwise `default`. Words for
    both sexes keep the default and leave the contradiction to GEN19. No `case_id`, no sampling."""
    # Negated or excluded mentions still count ("prostate" in an excluded item presupposes a
    # male patient); the emission gate uses the same `SEX_EXCLUSIVE` table.
    from .gate_tables import SEX_EXCLUSIVE
    blob = " ".join(f"{t} {c}" for _, t, c in spec.get("symptoms") or ())
    male = any(w in blob for w in SEX_EXCLUSIVE["F"])      # words only a male would have
    female = any(w in blob for w in SEX_EXCLUSIVE["M"])    # words only a female would have
    if male and not female:
        return "M"
    if female and not male:
        return "F"
    if male and female:
        return default                                     # spec is self-contradictory, hand it to GEN19
    _sid = str(spec_id or spec.get("spec_id") or "")
    _skew = __getattr__('CONDITION_SEX_SKEW').get(_sid)
    _p = _skew[0] if _skew is not None else None
    # `independent` cases have no disease, so they use the population baseline (0.5).
    if _p is None and str(spec.get("join_gold") or "") == "independent":
        _p = _benign_baseline()
    if _p is not None and case_id:
        from . import rng
        return "M" if rng.unit(case_id, "sex_skew") < _p else "F"
    return default


# `index_time_T` is sampled per case from `extract.KNOB_DOMAIN["index_time_T"]`.
#: Spec symptom days are written for T=84 and scaled by T/84 before jitter.
_T_SPEC_BASE = 84

#: The symptom-day baseline shared by nearly all kernel specs.
_SPEC_BASE_DAYS: tuple[int, ...] = (14, 35, 56, 77)


def _t_index_for(anon: str) -> int:
    """This case's index time, from `case_id` only (window length must not depend on the
    disease). The domain is `extract.KNOB_DOMAIN`."""
    from . import rng
    from .job_schema import KNOB_DOMAIN
    return int(rng.pick(list(KNOB_DOMAIN["index_time_T"]), anon, "T"))


# Jitter must leave headroom so the last symptom still lands inside the observation window
_SYM_JITTER_AMP = 6          # +/-6 days: original spacing 21, jittered spacing lands in 9-33, enough to break the arithmetic progression
_SYM_MIN_GAP = 5             # symptoms must be at least 5 days apart (crowded together stops being a "cross-time projection")


def jitter_symptom_days(case_id: str, days: list[int], T: int) -> list[int]:
    """Jitter the spec's symptom days per case: scaled to T, order kept, at least
    `_SYM_MIN_GAP` days apart, inside `[7, T-3]`. A pure function of `case_id` and index, so
    nearly-identical spec days (14/35/56/77) do not become a shortcut."""
    from . import rng
    out: list[int] = []
    lo_bound, hi_bound = 7, max(7, T - 3)
    _sc = (float(T) / float(_T_SPEC_BASE)) if _T_SPEC_BASE else 1.0
    for i, d0 in enumerate(days):
        off = rng.below(2 * _SYM_JITTER_AMP + 1, case_id, "symday", i) - _SYM_JITTER_AMP
        d = int(round(int(d0) * _sc)) + off
        floor = (out[-1] + _SYM_MIN_GAP) if out else lo_bound
        d = max(floor, min(d, hi_bound))
        if d < floor:                     # the upper bound pushed it back -> this case can't fit this many symptoms
            d = floor
        out.append(d)
    return out


# Course end points; `_course_end_for` keeps values >= T + 168 (560 covers T = 336).
DDX_COURSE_END_DOMAIN: tuple[int, ...] = (224, 273, 315, 365, 448, 504, 560)


def _course_end_for(anon: str, T: int = _T_SPEC_BASE) -> int:
    """This case's course end, from `case_id` only. At least `T + 168` (the default reversal at
    `T + 56` plus twice the 56-day persistence minimum), so the regain can play out without exceeding
    `max_weekly_delta`; if no domain value qualifies, the bound itself is used."""
    from . import rng
    lo = int(T) + 168
    ok = [d for d in DDX_COURSE_END_DOMAIN if d >= lo]
    if not ok:
        return max(DDX_COURSE_END_DOMAIN + (lo,))
    return int(rng.pick(ok, anon, "ddx_course_end"))


# ============================================================ Data-stream density axis
# Each tier declares value domains; values are sampled per `case_id` with the tier name in the
# salt, so other dimensions are unchanged across tiers. Rates are derived from sampled counts
# (`n / weeks`) so GEN20a's `round(rate x T/7) == n` holds exactly.

#: Defaults for `measure_per_week`/`symptom_rate`/`life_event_rate` when `density=None`.
_FROZEN_DENSITY: tuple[int, float, float] = (7, 0.38, 0.1)

#: Density tiers: value domains per knob; `n_benign`/`n_life` are counts in the window.
DENSITY_TIERS: dict[str, dict] = {
    "dense": {
        # step = round(7/mpw): 7 -> 1 day, 3.5 -> 2 days. The densest tier for the daily metric stream.
        "measure_per_week": (7.0, 3.5),
        "n_benign": (8, 9),          # 8-9 benign complaints within a 12-week window (the frozen default gives 5)
        "n_life": (3, 4),            # 3-4 life events (the frozen default gives 1)
        "note": "高密度:逐天/隔天采样 + 大量良性噪声,考「在噪声里挑出真信号」",
    },
    "sparse": {
        # step = round(7/mpw): 1.0 -> 7 days, 0.7 -> 10 days.
        "measure_per_week": (1.0, 0.7),
        "n_benign": (1, 2),
        "n_life": (0, 1),
        "note": "低密度:周/旬采样 + 近乎无噪声,考「证据本来就少时敢不敢下结论」",
    },
}


#: `mixed` samples per case among the two tiers and the frozen default (which keeps overlap
#: with packs built without a tier).
_MIXED_TIERS: tuple[str | None, ...] = ("dense", "sparse", None)


def _density_knobs(anon: str, tier: str | None, T: int) -> tuple[float, float, float]:
    """`(measure_per_week, symptom_rate, life_event_rate)` for this case. `None` returns the
    frozen defaults; an unknown tier raises."""
    if tier is None:
        return _FROZEN_DENSITY
    if str(tier) == "mixed":
        from . import rng as _r
        return _density_knobs(anon, _r.pick(list(_MIXED_TIERS), anon, "density_tier"), T)
    spec = DENSITY_TIERS.get(str(tier))
    if spec is None:
        raise ValueError(f"[ddx] unregistered density tier {tier!r} —— options: {sorted(DENSITY_TIERS)}")
    from . import rng
    weeks = max(1.0, float(T) / 7.0)
    mpw = float(rng.pick(list(spec["measure_per_week"]), anon, "density_mpw", str(tier)))
    n_sym = int(rng.pick(list(spec["n_benign"]), anon, "density_n_benign", str(tier)))
    n_life = int(rng.pick(list(spec["n_life"]), anon, "density_n_life", str(tier)))
    return mpw, round(n_sym / weeks, 4), round(n_life / weeks, 4)


# ---- composition v2: symptom penetrance + synonym variants ---------------------------------------
#: Variant-text cycle: canonical text first, then colloquial, then atypical. Case `v` of a spec takes the
#: `(offset + v)`-th text of a per-slot cycle, so up to `len(cycle)` variants of one spec never repeat a text.
def _symptom_sources(cid: str, spec: dict) -> list[tuple[str, int]]:
    """`[(source_spec_id, symptom_idx)]` for each symptom of `spec`. A comorbid composition takes its
    four symptoms from its two kernel specs (overlay `haenv_comorbid_specs`: a0, b0, a1, b1)."""
    from .overlay import HAENV_COMORBID_PAIRS
    cfg = HAENV_COMORBID_PAIRS.get(cid)
    if cfg:
        a, b = cfg["pair"]
        ta, tb = cfg["take"]
        return [(a, int(ta[0])), (b, int(tb[0])), (a, int(ta[1])), (b, int(tb[1]))]
    return [(cid, i) for i in range(len(spec["symptoms"]))]


def _penetrance_index() -> dict[tuple[str, int], dict]:
    from .registry import load_symptom_penetrance
    return {(sid, int(r["idx"])): r for sid, rows in load_symptom_penetrance().items() for r in rows}


def v2_symptom_draw(anon: str, v: int, cid: str, spec: dict, sym_days: list[int], T: int,
                    insufficient: bool) -> tuple[list[int], list[str]]:
    """Composition-v2 draw for one case: which of the spec's symptoms are expressed (independent
    Bernoulli(penetrance), a floor keeping the emission gates satisfiable) and which wording each
    expressed symptom uses. Returns `(kept_indices, texts_by_spec_position)`.

    Floors (only off the insufficient tier): at least 2 symptoms on or before T (GEN14 requires 2
    visible true symptoms), and at least one per thread for a comorbid composition (two threads
    interleaved a0, b0, a1, b1). The insufficient tier has every symptom after T, so the draw only thins it."""
    from . import rng
    idx = _penetrance_index()
    src = _symptom_sources(cid, spec)
    pens, texts = [], []
    for k, (sid, i) in enumerate(src):
        row = idx.get((sid, i))
        canon = str(spec["symptoms"][k][1])
        pens.append(float(row["penetrance"]) if row else 1.0)
        if row is None:
            texts.append(canon)
            continue
        cyc = [str(row["text"])] + [str(t) for t in (row.get("variants") or {}).get("colloquial", [])] \
            + [str(t) for t in (row.get("variants") or {}).get("atypical", [])]
        off = rng.below(len(cyc), cid, "sym_variant_offset", k)
        texts.append(cyc[(off + int(v) - 1) % len(cyc)])
    keep = [k for k in range(len(src)) if rng.unit(anon, "sym_penetrance", k) < pens[k]]
    if not insufficient:
        def vis(ks):
            return [k for k in ks if int(sym_days[k]) <= int(T)]
        threads = [(0, 2), (1, 3)] if src and src[0][0] != src[1][0] and cid in _pair_ids() else []
        def add_back(pool_pred, need_fn):
            nonlocal keep
            while need_fn(keep):
                cand = [k for k in range(len(src)) if k not in keep and int(sym_days[k]) <= int(T) and pool_pred(k)]
                if not cand:
                    break
                cand.sort(key=lambda k: (-pens[k], k))
                keep = sorted(keep + [cand[0]])
        for th in threads:
            add_back(lambda k, th=th: k in th, lambda ks, th=th: not any(k in th for k in vis(ks)))
        add_back(lambda k: True, lambda ks: len(vis(ks)) < 2)
    if not keep:                                   # at least one symptom is always expressed
        keep = [max(range(len(src)), key=lambda k: (pens[k], -k))]
    return sorted(keep), texts


def _pair_ids() -> set:
    from .overlay import HAENV_COMORBID_PAIRS
    return set(HAENV_COMORBID_PAIRS)


def variant_id(anon: str, v: int) -> str:
    """Case id of the v-th variant; `v=1` is the frozen id. `JD-01v2`, not `JD-01-v2`: EV ids
    are split on `-` downstream."""
    return anon if v <= 1 else f"{anon}v{v}"


def declare_high_distractor_density(latent: dict) -> dict:
    """Make a `distractor_level: high` case's declared benign-symptom count equal what the
    injector puts in the ledger. Mutates and returns `latent`.

    `haenv_kernel/noise.inject_distractors` writes `min(pool, max(6, round(rate x (T + window) / 7)))`
    benign symptoms upstream, and the density pass counts every one of them toward the quota
    `round(rate x T / 7)` (`events.expected_event_counts`). When the declared quota is below
    the upstream count, all upstream items are kept and no planned item tops the ledger
    up, so the ledger holds more than the declaration and `gates.check_event_density` rejects
    the case. The high tier always injects the whole pool once the declared quota reaches the
    pool size (the count is capped at the pool size for any window), so declaring at least
    the pool size makes declared and injected agree for every window length.
    """
    from haenv_kernel.noise import _DISTRACTOR_SYMPTOMS
    ed = latent["event_density"]
    weeks = max(1.0, float(latent["index_time_T"]) / 7.0)
    pool = len(_DISTRACTOR_SYMPTOMS)
    rate = float(ed.get("symptom_rate") or 0.0)
    if int(round(rate * weeks)) < pool:
        rate = round(pool / weeks, 4)
        while int(round(rate * weeks)) < pool:      # 4-decimal rounding can land just under
            rate = round(rate + 0.0001, 4)
        ed["symptom_rate"] = rate
    return latent


def ddx_case_specs(only: list[str] | None = None, variants: int = 1,
                   include_draft: bool = False, density: str | None = None,
                   insufficient_frac: float | None = None, composition_v2: bool = False,
                   variant_start: int = 1) -> list[dict]:
    """The job.yaml case list (`{case_id, raw, latent}`).

    `density` selects a density tier (`None` = frozen defaults). `insufficient_frac` overrides
    the insufficient-tier share (`None` = `INSUFFICIENT_FRAC`). `variants` produces N cases per
    condition that differ only in `case_id`-derived draws (demographics, weight scale 0.80-1.20,
    timing); `v=1` keeps the frozen id. Variants share the narrative and gold, so they are not
    independent samples.

    `composition_v2` turns on symptom penetrance and synonym wording (`registry/symptom_penetrance.yaml`)
    and marks each case `latent.composition_v2`; off, the output is byte-identical to before. `variant_start`
    numbers the first variant of each spec (the allocator may emit variants `k..k+n-1`).
    """
    from .demographics import doses_per_week as _doses_per_week, sample_profile as _sample_profile
    from .overlay import check_threads, check_vocab, condition_registry, threads_for
    specs = condition_registry(include_draft=include_draft)
    for p in check_threads(specs) + check_vocab(
            [c for s in specs.values() for _, _, c in s["symptoms"]]):
        log.warning("[ddx] incomplete overlay declaration: %s", p)

    # Ids are frozen in registry/case_ids.yaml: assigning by position would re-roll patients.
    from .registry import check_case_ids_cover as _cover, load_case_ids as _lcid
    for p in _cover(list(specs)):
        raise ValueError(f"[ddx] {p}")
    _CASE_IDS = _lcid()

    out: list[dict] = []
    for cid, spec in specs.items():
        if only and cid not in only:
            continue
        kind, start, end = spec["weight"]
        # The spec key names the answer, so an opaque case id is used on the solver side.
        _used_disease: list[str] = []
        for _v in range(int(variant_start), int(variant_start) + max(1, int(variants))):
            anon = variant_id(_CASE_IDS[cid], _v)
            # Multiplicative scaling keeps every relative quantity, and so the gold label, unchanged.
            _sc = 1.0
            if _v > 1:
                from . import rng as _rng
                _sc = 0.80 + 0.40 * _rng.unit(anon, "variant_weight_scale")
            _start = float(start) * _sc
            _end = float(end) * _sc
            _T = _t_index_for(anon)
            _long = long_horizon_for(anon, spec)
            if _long:
                _T = int(LONG_T)
            _sym_days = jitter_symptom_days(anon, [s[0] for s in spec["symptoms"]], int(_T))
            _mpw, _sym_rate, _life_rate = _density_knobs(anon, density, int(_T))
            _ins = insufficient_tier_for(
                anon, _v, int(_T), _mpw, base_id=_CASE_IDS[cid], n_variants=int(variants),
                **({} if insufficient_frac is None else {"frac": float(insufficient_frac)}))
            if _ins:
                _sym_days = insufficient_symptom_days(
                    _sym_days, int(_T), _course_end_for(anon, int(_T)))
                # Replace the moved true complaints with as many benign ones, so the number of patient
                # reports does not mark the tier (only under an explicit density tier).
                if density is not None:
                    _sym_rate = round(_sym_rate + len(_sym_days) / max(1.0, float(_T) / 7.0), 4)
            _rf, _urg = insufficient_triage(spec, _sym_days, int(_T), _ins)
            _keep, _texts = (list(range(len(spec["symptoms"]))), [str(s[1]) for s in spec["symptoms"]])
            if composition_v2:
                _keep, _texts = v2_symptom_draw(anon, _v, cid, spec, _sym_days, int(_T), bool(_ins))
            # Sampled once: a second draw with `avoid` populated could pick a different disease.
            _prof = _sample_profile(anon, _sex_of(spec, case_id=anon, spec_id=cid),
                                    avoid=tuple(_used_disease))
            _used_disease.append(str(_prof.get("disease") or ""))
            out.append({
                "case_id": anon,
                "raw": {
                    **{k: v for k, v in _prof.items() if k != "sex"},
                    "sex": _sex_of(spec, case_id=anon, spec_id=cid),
                    "start_weight": round(_start, 1),
                    "nadir_weight": round(min(_start, _end), 1),
                    "symptoms": [{"day": _sym_days[k], "text": _texts[k], "context": spec["symptoms"][k][2]}
                                 for k in _keep],
                },
                "latent": {
                    "index_time_T": int(_T),
                    "course_end_day": _course_end_for(anon, int(_T)),
                    "outcome": "regain" if kind == "up" else "maintain",
                    # `up` family only: the kernel endpoint carries the rise magnitude.
                    **({"regain_end_kg": round(_end, 1)} if kind == "up" else {}),
                    # `down` family only: weight bottoms out at 85% of the course, leaving a plateau.
                    **({"nadir_day": int(round(_course_end_for(anon, int(_T)) * 0.85))}
                       if kind == "down" else {}),
                    # Placeholder: this item type's gold is the diagnosis, not a driver.
                    "driver": "unknown_or_multifactorial",
                    "ddx_spec_id": cid,                     # the kernel spec key (verifier-only, for lookup)
                    "ddx_diagnosis": spec["diagnosis"],
                    "ddx_aliases": list(spec.get("aliases") or []),
                    "ddx_join_gold": spec["join_gold"],
                    "ddx_threads": [dict(t) for t in (threads_for(cid, spec) or ())] or None,
                    # The insufficient tier carries no gold tests/specialty: with no clues, ordering the right
                    # confirmatory test cannot be required. The diagnosis stays for the verifier.
                    **({} if _ins else
                       {"ddx_tests": spec["tests"], "ddx_specialty": spec["specialty"]}),
                    **({"ddx_insufficient": True} if _ins else {}),
                    "ddx_urgency": _urg,
                    "ddx_red_flag": _rf,
                    "ddx_clinician_warranted": bool(spec.get("clinician_warranted", True)),
                    "ddx_outcome_label": spec.get("outcome_label"),
                    # Dosing frequency follows the sampled drug (weekly vs daily).
                    "event_density": {"measure_per_week": _mpw,
                                      "dosing_per_week": _doses_per_week(_prof["drug"]),
                                      "symptom_rate": _sym_rate, "life_event_rate": _life_rate,
                                      "clinical_symptoms_recorded": len(_keep),
                                      "course_weeks": round(_course_end_for(anon, int(_T)) / 7.0, 1)},
                    **({"rhythm_gap": True}
                       if gap_tier_for(anon, spec, int(_T)) else {}),
                    **({"composition_v2": True} if composition_v2 else {}),
                    # Recorded explicitly: cases can also land on T=336 naturally, outside the eligibility
                    # filter, so `T` alone cannot identify the tier's control arm.
                    **({"long_horizon_tier": True} if _long else {}),
                },
            })
    log.info("[ddx] exported %d DDX spec(s)", len(out))
    return out


# ============================================================ Density-axis probe item packs
# Four conditions, identical case ids in both packs, differing only in `event_density`.
DENSITY_PROBE_CONDITIONS: tuple[str, ...] = ("JD-PCOS", "JD-CKM", "JD-SLE", "JD-PHEO")


def density_probe_doc(tier: str) -> str:
    """Full text (header included) of `inputs/joint_dx-density-<tier>.job.yaml`."""
    import yaml
    if tier not in DENSITY_TIERS:
        raise ValueError(f"[ddx] unregistered density tier {tier!r} —— options: {sorted(DENSITY_TIERS)}")
    cases = ddx_case_specs(only=list(DENSITY_PROBE_CONDITIONS), density=tier)
    job_id = f"joint_dx-density-{tier}"
    doc = {"job_id": job_id, "task_type": "joint_dx", "multiround": False,
           "include_baseline": True, "models": [], "sample_cases": len(cases),
           "report": f"eval-{job_id}.md", "cases": cases}
    spec = DENSITY_TIERS[tier]
    header = (
        f"# 数据流密度轴探针 · {tier} 档({spec['note']})。\n"
        f"# 与 joint_dx-density-{'sparse' if tier == 'dense' else 'dense'} **同一批 case_id、\n"
        f"#   只有 latent.event_density 不同** —— 配对设计,rng 稳定。\n"
        f"# 档位取值域:measure_per_week{spec['measure_per_week']} · "
        f"n_benign{spec['n_benign']} · n_life{spec['n_life']}(逐例按 case_id 抽)。\n"
        f"# **本文件由 haenv/ddx.py 生成,不要手改。** 在仓库根重建:\n"
        f"#   uv run python -c \"from haenv.cli import _bootstrap, load_cfg; _bootstrap(load_cfg());\\\n"
        f"#     from haenv.ddx import density_probe_doc as d;\\\n"
        f"#     open('inputs/{job_id}.job.yaml','w').write(d('{tier}'))\"\n")
    # The generator emits the canary itself so a rebuild matches the file byte for byte.
    from .canary import block as _canary_block
    return _canary_block("# ") + header + yaml.safe_dump(
        doc, allow_unicode=True, sort_keys=False, width=200)
