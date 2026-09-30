"""latent_rules.py -- consistency rules across latent variables (pure functions).

`job.LATENT_REGISTRY` decides which latent keys may exist; this table checks that the
keys are consistent with each other, so a rejected case names the keys at fault rather
than just "some gate blocked it". The rules only judge and never modify latent: gold is
derived deterministically from latent, so changing latent changes the answer.

`contradictions()` returns every contradiction; `blocking_contradictions()` returns the
subset that is a sufficient condition for the case not to generate, so the caller can
stop it before a model call.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


#: Rules that are necessary-condition screens: a case can fail them and still generate
#: (e.g. when the injector happens to land on the declared count), so they only report.
NON_BLOCKING_RULES: frozenset[str] = frozenset({"L2-density-satisfiable"})


@dataclass(frozen=True)
class LatentCheck:
    """One latent-variable consistency verdict. `ok=None` means unable to judge (missing
    field / out of domain), distinct from `ok=False` (a contradiction).
    """
    rule: str
    ok: bool | None
    detail: str
    #: Whether a failed verdict may block generation; a property of the rule, not the verdict.
    @property
    def blocking(self) -> bool:
        return self.rule not in NON_BLOCKING_RULES


#: Physiological upper bound on weekly weight change (kg/wk), matching the kernel's
#: `latent.DISEASE_SIGNAL_DOMAIN[*].weight`.
DEFAULT_MAX_WEEKLY_DELTA = 1.5

#: Thresholds of `label_rule` (minimum relative change, minimum persistence in days). They live
#: here so the pre-generation check L5 and `gates.derive_outcome` use one value; `build` imports them.
MIN_CHANGE_FRAC = 0.05
MIN_PERSIST_DAYS = 56
#: Absolute floor of the regain threshold (kg): regain >= max(MIN_CHANGE_FRAC x lost,
#: MIN_CHANGE_KG). It sits above ordinary day-to-day and weekly weight fluctuation, so a
#: case that lost almost nothing cannot "regain" on wobble alone.
MIN_CHANGE_KG = 0.75
#: How `label_rule` reads the series: nadir, threshold crossing and persistence on the median
#: of the 7 readings centred on each (`gates.label_series`), whatever the sampling interval.
#: It applies to the primary weight series; where the rule reads the clean reference
#: (`weight_ref`, persistent-offset noise classes) it reads that pointwise.
#: It damps single-reading lows; it does not remove misreads on observed series.
LABEL_SMOOTHING = "rolling_median_7pt"

#: Share of the weekly cap the trend may use, leaving room for day-to-day wobble; also used by
#: `build._weight_series` to set the descent length.
DESCENT_BUDGET_FRAC = 0.6


def check_descent_expressible(start: float | None, nadir: float | None,
                              course_end_day: int | None,
                              max_weekly_delta: float = DEFAULT_MAX_WEEKLY_DELTA
                              ) -> LatentCheck:
    """L1: `|nadir - start|` must be walkable within the whole disease course at a
    physiologically possible rate (`max_weekly_delta * DESCENT_BUDGET_FRAC`).

    The bound is the course, not the visible window `T`: the nadir may fall after `T`.
    """
    T = course_end_day
    if start is None or nadir is None or not T or T <= 0:
        return LatentCheck("L1-descent-expressible", None, "缺 start/nadir/course_end_day")
    drop = abs(float(nadir) - float(start))
    if drop == 0:
        return LatentCheck("L1-descent-expressible", True, "降幅 0,恒可表达")
    cap = float(max_weekly_delta) * DESCENT_BUDGET_FRAC
    need_days = drop / (cap / 7.0)
    ok = need_days <= float(T) + 1e-9
    return LatentCheck(
        "L1-descent-expressible", ok,
        f"降 {drop:.1f} kg 至少需 {math.ceil(need_days)} 天(上限 {max_weekly_delta}×"
        f"{DESCENT_BUDGET_FRAC}={cap:.2f} kg/wk),而病程 {T} 天"
        + ("" if ok else " ⇒ 病程内表达不出来"))


def check_density_satisfiable(rate_per_week: float | None, T: int | None,
                              min_injected: int = 1) -> LatentCheck:
    """L2: the declared event density x window must round to a count the injector can hit.

    The generation gate expects `round(rate x weeks)` events while the injector always
    injects at least `min_injected`; on a short window the expectation can round to 0.
    Over-reports by design (see `NON_BLOCKING_RULES`).
    """
    if rate_per_week is None or not T or T <= 0:
        return LatentCheck("L2-density-satisfiable", None, "缺 event_density/T")
    weeks = max(1.0, float(T) / 7.0)
    want = int(round(float(rate_per_week) * weeks))
    ok = want >= min_injected or float(rate_per_week) == 0.0
    return LatentCheck(
        "L2-density-satisfiable", ok,
        f"{rate_per_week}/wk × {weeks:.2f}wk = {rate_per_week * weeks:.2f} → round={want} 条"
        + ("" if ok else f",而注入器至少注 {min_injected} 条 ⇒ 声明不可满足"))


def check_reversal_in_course(reversal_week: int | None,
                             course_end_day: int | None) -> LatentCheck:
    """L3: the reversal point must fall inside the disease course, or Track E has nothing to judge."""
    if reversal_week is None or course_end_day is None:
        return LatentCheck("L3-reversal-in-course", None, "缺 reversal_week/course_end_day")
    rev_day = int(reversal_week) * 7
    ok = 0 <= rev_day < int(course_end_day)
    return LatentCheck("L3-reversal-in-course", ok,
                       f"反转日 {rev_day} vs 病程末点 {course_end_day}"
                       + ("" if ok else " ⇒ 反转落在病程之外"))


def check_insufficient_before_symptom(insufficient: bool | None, T: int | None,
                                      symptom_days: list[int] | None) -> LatentCheck:
    """L4: declaring `ddx_insufficient` requires `T` to precede the first real symptom;
    otherwise the case is in the wrong tier.
    """
    if not insufficient:
        return LatentCheck("L4-insufficient-before-symptom", None, "非信息不足档,不适用")
    if not T or not symptom_days:
        return LatentCheck("L4-insufficient-before-symptom", None, "缺 T/症状日")
    first = min(int(d) for d in symptom_days)
    ok = int(T) < first
    return LatentCheck("L4-insufficient-before-symptom", ok,
                       f"T={T} vs 第一个真症状日 {first}"
                       + ("" if ok else " ⇒ 已有真症状可见,不是信息不足"))


#: The rule table: `(name, category, reads, what it checks)`; functions are in `_FN`.
#: Every entry is `consistency` (judge only), so entries are independent and unordered.
LATENT_RULE_SPECS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("L1-descent-expressible", "consistency", ("start", "nadir", "course_end_day"),
     "降幅必须能在病程内以生理可能的速率走完(不是在 T 内 —— T 只决定看得见多少)"),
    ("L2-density-satisfiable", "consistency", ("event_density", "index_time_T"),
     "声明的事件密度 × 窗口必须能取到注入器达得到的整数条数"),
    ("L3-reversal-in-course", "consistency", ("reversal_week", "course_end_day"),
     "反转点必须落在病程之内"),
    ("L4-insufficient-before-symptom", "consistency",
     ("ddx_insufficient", "index_time_T", "symptoms"),
     "声明信息不足 ⇒ T 必须早于第一个真症状"),
)

_FN = {
    "L1-descent-expressible": check_descent_expressible,
    "L2-density-satisfiable": check_density_satisfiable,
    "L3-reversal-in-course": check_reversal_in_course,
    "L4-insufficient-before-symptom": check_insufficient_before_symptom,
}


def check_outcome_sustainable(outcome: str | None, reversal_week: int | None,
                              course_end_day: int | None,
                              min_persist_days: int = MIN_PERSIST_DAYS) -> LatentCheck:
    """L5: after the reversal there must be at least `min_persist_days` of course left for a
    declared rebound (`label_rule`'s persistence requirement) to hold.

    L3 alone lets a reversal just before the course end through. Only applies to
    `outcome == "regain"`; other outcomes return `ok=None`.
    """
    if str(outcome or "") != "regain":
        return LatentCheck("L5-outcome-sustainable", None, "非回升档,不适用")
    if reversal_week is None or course_end_day is None:
        return LatentCheck("L5-outcome-sustainable", None, "缺 reversal_week/course_end_day")
    rev_day = int(reversal_week) * 7
    runway = int(course_end_day) - rev_day
    ok = runway >= int(min_persist_days)
    return LatentCheck("L5-outcome-sustainable", ok,
                       f"反转日 {rev_day} → 病程末点 {course_end_day},跑道 {runway} 天 "
                       f"vs label_rule 要求持续 {min_persist_days} 天"
                       + ("" if ok else " ⇒ 声明的回升在这条病程上站不住"))


def audit_latent(raw: dict, latent: dict) -> list[LatentCheck]:
    """Run one case's `raw` + `latent` through every rule and return all verdicts,
    including `ok=None`. Missing fields give `ok=None`, never an exception.
    """
    T = latent.get("index_time_T") or raw.get("T")
    ed = latent.get("event_density")
    rate = (ed or {}).get("symptom_rate") if isinstance(ed, dict) else None
    sym = [s[0] if isinstance(s, (list, tuple)) else s.get("day")
           for s in (raw.get("symptoms") or []) if s]
    return [
        check_descent_expressible(raw.get("start_weight"), raw.get("nadir_weight"),
                                  latent.get("course_end_day") or raw.get("course")),
        check_density_satisfiable(rate, T),
        check_reversal_in_course(latent.get("reversal_week"), latent.get("course_end_day")),
        check_insufficient_before_symptom(latent.get("ddx_insufficient"), T,
                                          [d for d in sym if isinstance(d, int)]),
        check_outcome_sustainable(latent.get("outcome"), latent.get("reversal_week"),
                                  latent.get("course_end_day") or raw.get("course")),
    ]


def contradictions(raw: dict, latent: dict) -> list[LatentCheck]:
    """The verdicts with `ok is False`, including non-blocking ones (for reporting)."""
    return [c for c in audit_latent(raw, latent) if c.ok is False]


def blocking_contradictions(raw: dict, latent: dict) -> list[LatentCheck]:
    """The contradictions that may block generation: `ok is False` and `blocking`
    (L1, L3, L4, L5; L2 is report-only).
    """
    return [c for c in audit_latent(raw, latent) if c.ok is False and c.blocking]
