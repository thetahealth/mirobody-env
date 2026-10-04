"""events.py -- daily metric streams and event evidence (EV) injected on top
of the clean primary trajectory (spec Section 11).

Injects (A) auxiliary daily metric streams backed by a declared device, with
baselines consistent with the raw case's baseline facts, and (B) event EVs:
the raw case's genuine symptoms (<= T, de-attributed) plus benign events at
the rate given by `event_density`. Injected streams and benign events must
carry no information about the conclusion: values never read `outcome` /
`reversal_week`, and driver/comorbidity proxies are excluded. Every injected
item gets a manifest record that `verify.py` checks; failing items are
dropped and re-injected by build's generate-verify loop.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml
from . import wearable as _wear
from . import world_plugins as _wp

import copy as _copy
import threading as _threading
import json
import logging
import math
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data
from haenv import streams as _streams

log = logging.getLogger("haenv.events")

# Rendered daily metrics that are auxiliary: exempt from device-inventory
# validation, not a predicted physiological indicator. Derived from the stream
# manifest (`registry/streams.yaml`); a subset of the kernel's `synth.AUX_SIGNALS`.
AUX_WHITELIST = set(_streams.aux_metrics())


# ============================================================ benign event pool (answer-irrelevant)
# The pool lives in `registry/benign_events.yaml`; a failed read raises (no
# built-in fallback). Optional per-item `cond`:
#   {"<axis>": {"deny": (values this patient cannot have...), "w": {value: relative rate}}}
# Every (item, openable axis) pair not covered by `cond` must be listed in
# `CONDITION_NEUTRAL_ITEMS`, or loading raises.
from .regpath import registry_path as _rp
from .events_text import (  # noqa: F401
    ANNOTATION_MARKERS,
    _LATIN_SUFFIXES,
    _load_symptom_topics,
    _norm_symptom,
    alias_hit,
    scrub_annotation,
    symptom_topic,
)
from .events_pools import (  # noqa: F401
    AGE_PRIOR_BANDS,
    AXIS_LEAK_TOKENS,
    COMORBID_PROXY_TAGS,
    CONDITION_AXIS_SPECS,
    COND_MULTIPLIER_RANGE,
    ConditionSpecError,
    DRIVER_PROXY_TAGS,
    EVENT_RATE_DEFAULTS,
    Facts,
    OUTCOME_EXPLAINING_TAGS,
    PoolPriorMissing,
    SEASONS,
    TAG_KEYWORDS,
    _AXIS_DOMAIN,
    _AXIS_OPENABLE,
    _AXIS_STATUS,
    _AXIS_VALUE_FN,
    _NON_METRIC_NDIGITS,
    _POOLS_LOCK,
    _REGISTRY,
    _as_item,
    _axis_decl,
    _declared_ndigits,
    _event_ok,
    _load_event_pools,
    _round_to_declared_digits,
    activity_of,
    age_band_index,
    age_prior,
    axis_neutrality_problems,
    axis_value,
    check_axis_neutrality,
    cond_denied,
    cond_multiplier,
    effective_tags,
    event_prior,
    event_proxy_tags,
    event_weeks,
    expected_event_counts,
    facts_of,
    infer_tags,
    normalize_cond_axes,
    proxy_tags,
    role_tags,
    season_of,
    weighted_order,
)
from .world_knobs import PHYSIO_ENABLED
from .events_streams import (  # noqa: F401
    BASELINE_ADJUST,
    BASELINE_ADJUST_FORMS,
    METRICS,
    METRIC_BY_NAME,
    MetricSpec,
    StreamPlan,
    _AR_PHI,
    _AR_SD_FRAC,
    _adj,
    _adj_full,
    CASE_OFFSET_CUT_SD,
    INDIVIDUAL_ADJUST,
    INDIVIDUAL_ADJUST_FORMS,
    individual_shift,
    _det_shock,
    _gold_signal_of_driver,
    _render_calibrated,
    _wearable_cadence,
    plan_streams,
    render_stream,
    world_layer_base_signals,
)
from .world_knobs import (  # noqa: F401
    FINDINGS_ENABLED,
)


_POOLS = _load_event_pools()

BENIGN_EVENTS: tuple[dict, ...] = tuple(_as_item(d) for d in _POOLS["benign_events"])
LIFE_EVENTS: tuple[dict, ...] = tuple(_as_item(d) for d in _POOLS["life_events"])


# Profiles for benign events from the kernel's `noise.inject_distractors`, so
# they go through the same item-by-item validation.
_INHERITED_PROFILE: dict[str, dict] = {
    k: _as_item(v) for k, v in _POOLS["inherited_profile"].items()
}

# The pools above are module constants, read by name all over this module (and monkeypatched by
# tests), so a plugin registration cannot reach them by a cache key. `_sync_pools()` rebuilds them
# when the `benign_events.yaml` overlay changed since they were built, and runs the load-time
# pool checks over the result; a bad plugin item raises naming its source and leaves the pools
# as they were. Every reader of the pools calls it first.
_POOLS_TOKEN: tuple = _wp.overlay_token("benign_events.yaml")
_POOLS_HOLD: tuple = _wp.overlay_values("benign_events.yaml")
_POOLS_SYNCING: int | None = None     # id of the thread that is rebuilding, if any


def _sync_pools() -> None:
    global _POOLS, BENIGN_EVENTS, LIFE_EVENTS, _INHERITED_PROFILE, CONDITION_NEUTRAL_ITEMS
    global _POOLS_TOKEN, _POOLS_HOLD, _POOLS_SYNCING
    tok = _wp.overlay_token("benign_events.yaml")
    if tok == _POOLS_TOKEN or _POOLS_SYNCING == _threading.get_ident():
        return
    with _POOLS_LOCK:
        if tok == _POOLS_TOKEN or _POOLS_SYNCING == _threading.get_ident():
            return
        old = (_POOLS, BENIGN_EVENTS, LIFE_EVENTS, _INHERITED_PROFILE, CONDITION_NEUTRAL_ITEMS)
        _POOLS_SYNCING = _threading.get_ident()
        try:
            pools = _load_event_pools()
            _POOLS = pools
            BENIGN_EVENTS = tuple(_as_item(d) for d in pools["benign_events"])
            LIFE_EVENTS = tuple(_as_item(d) for d in pools["life_events"])
            _INHERITED_PROFILE = {k: _as_item(v) for k, v in pools["inherited_profile"].items()}
            CONDITION_NEUTRAL_ITEMS = {ax: frozenset(v) for ax, v in pools["condition_neutral_items"].items()}
            bad = check_condition_specs() + check_pool_priors()
            if bad:
                srcs = sorted({src for (t, _s, _k), src in _wp._SOURCES.items()
                               if t == "benign_events.yaml"})
                raise _wp.WorldPluginError(
                    f"event pool registered by {', '.join(srcs) or 'a plugin'} fails the load-time "
                    f"pool checks: {'; '.join(bad[:4])}")
        except BaseException:
            _POOLS, BENIGN_EVENTS, LIFE_EVENTS, _INHERITED_PROFILE, CONDITION_NEUTRAL_ITEMS = old
            raise
        finally:
            _POOLS_SYNCING = None
        _POOLS_TOKEN, _POOLS_HOLD = tok, _wp.overlay_values("benign_events.yaml")


def event_pools() -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """`(BENIGN_EVENTS, LIFE_EVENTS)` as they stand now, world-plugin registrations included.
    Readers outside this module call this instead of importing the constants by name, which
    would freeze the pools at import time."""
    _sync_pools()
    return BENIGN_EVENTS, LIFE_EVENTS


def inherited_profile(text: str) -> dict | None:
    """Profile of an upstream-injected benign event, or None."""
    _sync_pools()
    return _INHERITED_PROFILE.get(text)


_COUPLING_RULES: list | None = None


def _apply_coupling(facts) -> dict:
    """Comorbidity coupling rules (no switch; never changes stream values).

    The result goes only into the Q-side ledger (`injected`), never the solver
    payload. The rules are `review: pending` and do not set gold.
    """
    global _COUPLING_RULES
    from pathlib import Path as _P

    from .physio import coupling as _cp

    if _COUPLING_RULES is None:
        _COUPLING_RULES = _cp.load_coupling_rules(_rp("comorbid_coupling.yaml"))
    rec = _cp.couple_case(facts, _COUPLING_RULES)
    # Record the review status in effect when the question was generated.
    rec["pending_review"] = _cp.pending_rules(
        [r for r in _COUPLING_RULES if r.rule_id in rec["fired_rules"]])
    return rec


def pool_event_topics() -> frozenset[str]:
    """Topics of the benign/life event pools: the option set for the LLM
    planning path (genuine-symptom topics are excluded)."""
    _sync_pools()
    return frozenset(str(it["topic"]) for pool in (BENIGN_EVENTS, LIFE_EVENTS)
                     for it in pool if it.get("topic") and not it.get("plugin_only"))


def event_topics() -> frozenset[str]:
    """The event `topic`s the generator injects (both pools plus genuine-symptom
    topics from `registry/symptom_topics.yaml`): the legal vocabulary for
    `registry/physio_kernels.yaml`, checked by `kernel._parse_entry` at load
    time. Topics of upstream `inherited_event`s are not included."""
    _sync_pools()
    pool_topics = {str(it["topic"]) for pool in (BENIGN_EVENTS, LIFE_EVENTS)
                   for it in pool if it.get("topic")}
    sym_topics = set()
    try:
        sym_topics = {t for t in _load_symptom_topics().values() if t and t != "none"}
    except Exception as e:                                # noqa: BLE001
        raise ValueError(f"cannot read symptom_topics.yaml, event topic vocabulary incomplete: {type(e).__name__}: {e}")
    return frozenset(pool_topics | sym_topics)


def effective_pool_size(f: Facts, driver: str) -> dict[str, int]:
    """Number of pool entries admissible for this patient, per rate key. Both
    the feasibility check and sampling must use this, not the nominal pool size."""
    _sync_pools()
    banned = event_proxy_tags(f, driver)
    # `_event_ok` reads `f.cond_axes` itself, so feasibility and sampling agree.
    return {"symptom_rate": sum(1 for it in BENIGN_EVENTS if _event_ok(it, f, banned)[0]),
            "life_event_rate": sum(1 for it in LIFE_EVENTS if _event_ok(it, f, banned)[0])}


#: Per axis, the item topics explicitly declared irrelevant to that axis.
CONDITION_NEUTRAL_ITEMS: dict[str, frozenset] = {
    ax: frozenset(v) for ax, v in _POOLS["condition_neutral_items"].items()
}

def check_condition_specs() -> list[str]:
    """Load-time self-check (default-deny):

    1. every (item, openable axis) has a `cond` declaration or is in
       `CONDITION_NEUTRAL_ITEMS`;
    2. `w`'s keys exactly cover "value domain minus deny", values > 0;
    3. axis names / values in `deny` are registered;
    4. the composite multiplier over the full Cartesian product of value
       domains stays within `COND_MULTIPLIER_RANGE`.
    """
    import itertools
    _sync_pools()
    bad: list[str] = []
    for pool_name, pool in (("BENIGN_EVENTS", BENIGN_EVENTS), ("LIFE_EVENTS", LIFE_EVENTS)):
        for it in pool:
            topic = it.get("topic")
            tag = f"{pool_name}:{topic or it.get('text')}"
            cond = it.get("cond") or {}
            for axis in cond:
                if axis not in _AXIS_STATUS:
                    bad.append(f"{tag}: 声明了未登记的轴 {axis!r}")
                elif axis not in _AXIS_OPENABLE:
                    bad.append(f"{tag}: 声明了不许打开的轴 {axis!r}(status={_AXIS_STATUS[axis]!r})")
            for axis in _AXIS_OPENABLE:
                dec = cond.get(axis)
                neutral = topic in CONDITION_NEUTRAL_ITEMS.get(axis, frozenset())
                if dec and neutral:
                    bad.append(f"{tag}: 轴 {axis!r} 既声明了 cond 又登记为中性 —— 两处矛盾")
                elif not dec and not neutral:
                    bad.append(f"{tag}: 轴 {axis!r} 未声明,也未登记为中性(default-deny)")
                if not dec:
                    continue
                dom = _AXIS_DOMAIN[axis]
                deny = tuple(dec.get("deny") or ())
                for v in deny:
                    if v not in dom:
                        bad.append(f"{tag}: 轴 {axis!r} 的 deny 含域外取值 {v!r}")
                if set(deny) >= set(dom):
                    bad.append(f"{tag}: 轴 {axis!r} 禁掉了全部取值 —— 这条事件谁都不会发生,应删除")
                w = dec.get("w")
                if w is None:
                    continue
                want = tuple(v for v in dom if v not in deny)
                if tuple(sorted(w)) != tuple(sorted(want)):
                    bad.append(f"{tag}: 轴 {axis!r} 的 w 键 {sorted(w)} ≠ 取值域−deny {sorted(want)}")
                for v, x in w.items():
                    if float(x) <= 0:
                        bad.append(f"{tag}: 轴 {axis!r} 的 w[{v!r}]={x} ≤ 0 —— "
                                   f"0 是「不可能」,应写进 deny(否则冻结度量看不见它)")
    lo, hi = COND_MULTIPLIER_RANGE
    doms = [_AXIS_DOMAIN[a] for a in _AXIS_OPENABLE]
    for pool_name, pool in (("BENIGN_EVENTS", BENIGN_EVENTS), ("LIFE_EVENTS", LIFE_EVENTS)):
        for it in pool:
            if not it.get("cond"):
                continue
            tag = f"{pool_name}:{it.get('topic') or it.get('text')}"
            for combo in itertools.product(*doms):
                m = 1.0
                skip = False
                for axis, v in zip(_AXIS_OPENABLE, combo):
                    dec = _axis_decl(it, axis)
                    if not dec:
                        continue
                    if v in tuple(dec.get("deny") or ()):
                        skip = True                      # under this combo it doesn't participate in sampling at all
                        break
                    m *= float((dec.get("w") or {}).get(v, 1.0))
                if skip:
                    continue
                if not (lo <= m <= hi):
                    bad.append(f"{tag}: 合成乘子 {m:.3f} 在 {dict(zip(_AXIS_OPENABLE, combo))} 上"
                               f"越出 {COND_MULTIPLIER_RANGE}")
    return bad


def check_pool_priors() -> list[str]:
    """Load-time self-check: every pool event declares a valid `age_w`."""
    _sync_pools()
    bad: list[str] = []
    for pool_name, pool in (("BENIGN_EVENTS", BENIGN_EVENTS), ("LIFE_EVENTS", LIFE_EVENTS)):
        for it in pool:
            tag = f"{pool_name}:{it.get('topic') or it.get('text')}"
            w = it.get("age_w")
            if w is None:
                bad.append(f"{tag}: 未声明 age_w")
            elif len(w) != len(AGE_PRIOR_BANDS):
                bad.append(f"{tag}: age_w 长度 {len(w)} ≠ 年龄段数 {len(AGE_PRIOR_BANDS)}")
            elif any(float(x) < 0 for x in w):
                bad.append(f"{tag}: age_w 含负数 {w}")
            elif not any(float(x) > 0 for x in w):
                bad.append(f"{tag}: age_w 全为 0 —— 这条事件在任何年龄都不会发生,应删除")
    return bad


_PRIOR_PROBLEMS = check_pool_priors()
if _PRIOR_PROBLEMS:
    raise PoolPriorMissing("; ".join(_PRIOR_PROBLEMS))

_COND_PROBLEMS = check_condition_specs() + check_axis_neutrality()
if _COND_PROBLEMS:
    raise ConditionSpecError("; ".join(_COND_PROBLEMS))


# ============================================================ Event EV planning
# Attribution/outcome wording in a symptom's context; such context is dropped.
# Subjective words like "felt like" are deliberately not included.
ATTRIBUTIVE_MARKERS = ("停药", "复胖", "反弹", "甲亢", "甲减", "治疗后", "起始后", "使用下降",
                       "预期药效", "获益", "复发", "确诊", "未识别",
                       "依从", "漏服", "费用", "缓解后再度")

# Words that count as attribution only when this case's intervention/outcome
# axis appears in the same text ("exfoliation ineffective" is a clinical clue).
SCOPED_ATTRIBUTIVE = ("无效", "失败")

# Generic words for the intervention/outcome axis (merged with this example's drug name)
THERAPY_AXIS_GENERIC = ("药", "剂量", "用量", "疗程", "方案", "治疗", "干预", "服用", "注射",
                        "减重", "减肥", "体重", "热量", "饮食控制", "限食")
DRUG_ZH = {"metformin": ("二甲双胍",), "semaglutide": ("司美格鲁肽", "司美"),
           "tirzepatide": ("替尔泊肽", "替尔"), "liraglutide": ("利拉鲁肽",)}


def therapy_axis(drug: str | None = None) -> tuple[str, ...]:
    """This case's intervention/outcome axis words: generic words + the drug name
    (English and Chinese)."""
    d = str(drug or "").strip().lower()
    return THERAPY_AXIS_GENERIC + ((d,) if d else ()) + DRUG_ZH.get(d, ())


def attributive_hits(blob: str, drug: str | None = None) -> list[str]:
    """Attributive wording in the text. Unconditional words hit directly;
    `SCOPED_ATTRIBUTIVE` must co-occur with the intervention/outcome axis."""
    hits = [k for k in ATTRIBUTIVE_MARKERS if k in blob]
    axis = therapy_axis(drug)
    for k in SCOPED_ATTRIBUTIVE:
        if k in blob and any(a in blob for a in axis):
            hits.append(k)
    return hits

# Wording that states how symptoms relate ("two processes superimposed"),
# i.e. the join_gold answer; dropped by `sanitize_context`.
JOIN_STRUCTURE_MARKERS = (
    "叠加", "非单一", "单一过程", "同一过程", "同一病因", "统一解释", "一元论",
    "两个过程", "多个过程", "彼此独立", "互不相关", "各自独立", "无统一",
    "共病", "合并症", "并存", "分别由",
)


def sanitize_context(ctx: str) -> tuple[str, bool]:
    """Drop the context if it contains attributive wording. Returns
    (context, whether it was stripped)."""
    if not ctx:
        return "", False
    if any(k in ctx for k in ATTRIBUTIVE_MARKERS + JOIN_STRUCTURE_MARKERS):
        return "", True
    return ctx, False


def plan_real_symptom_evs(f: Facts, T: int, drop: set[str] | None = None,
                          aliases=None) -> tuple[list[dict], list[dict]]:
    """The raw case's genuine symptoms (<= T) as de-attributed evidence EVs."""
    drop = drop or set()
    evs, skipped = [], []
    for i, s in enumerate(f.symptoms, 1):
        eid = f"EV-{f.case_id}-S{i}"
        if s["day"] > T:
            skipped.append({"item": eid, "reason": f"post_T(day{s['day']}>T{T})"})
            continue
        if eid in drop:
            skipped.append({"item": eid, "reason": "dropped_by_verifier"})
            continue
        from .overlay import gate_context
        ctx, facet, why = gate_context(s["context"])           # closed word list: unregistered => dropped (default-deny)
        # Per-facet capping happens later, in `cap_context_facets`.
        stripped = bool(s["context"].strip()) and not ctx
        txt, rm_t = scrub_annotation(s["text"], aliases)       # scrub author annotations that name the diagnosis
        ctx, rm_c = scrub_annotation(ctx, aliases)
        if rm_t or rm_c:
            log.info("[events] %s removed parentheticals naming a diagnosis %s", eid, (rm_t + rm_c)[:3])
        ev = {"evidence_id": eid, "source_type": "patient_reported_symptom",
              "source_timestamp": s["day"], "symptom": txt,
              "claim_supported": True, "reliability_status": "reported",
              # Topic is looked up on the original text (the registry key),
              # before scrubbing; popped before emission.
              "_topic": symptom_topic(s["text"])}
        if ctx:
            ev["context"] = ctx
        evs.append(ev)
        if stripped:
            log.info("[events] %s context not emitted (facet=%s · %s): %r",
                     eid, facet or "-", why, s["context"])
    return evs, skipped


# Jitter amplitude (days) for injected event timepoints (nominal spacing ~15 days).
_BENIGN_JITTER_AMP = 5

#: Maximum context entries emitted per facet, the same for every item, so
#: facet composition cannot carry the outcome label.
CTX_FACET_CAP = 2


def cap_context_facets(raw) -> int:
    """Cap context entries per facet across the whole assembled ledger.
    Returns the number of context strings removed.

    Facet is taken from the emitted form (`facet_of_emitted_context`, as in the
    anti-shortcut scan). Only context strings are removed, never an event.
    """
    from .overlay import facet_of_emitted_context as _fac
    _used: dict[str, int] = {}
    _trimmed = 0
    for _e in raw.evidence_ledger:
        _c = str(_e.get("context") or "").strip()
        if not _c:
            continue
        _fa = _fac(_c)
        if _used.get(_fa, 0) >= CTX_FACET_CAP:
            _e.pop("context", None)
            _trimmed += 1
        else:
            _used[_fa] = _used.get(_fa, 0) + 1
    if _trimmed:
        log.info("[events] %s context capped per facet: trimmed %d strings (no symptom removed) · kept %s",
                 getattr(raw, "case_id", "?"), _trimmed, _used)
    return _trimmed


def _day_jitter(case_id: str, kind: str, i: int, amp: int) -> int:
    """(case_id, kind, i) -> a deterministic offset in [-amp, +amp]; a pure
    function of the key."""
    import hashlib
    h = hashlib.blake2b(f"{case_id}|{kind}|{i}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 * amp + 1) - amp


#: Self-reported event timing. Real encounter and symptom-report gaps are over-dispersed: the
#: per-person coefficient of variation of the gaps has median 1.37-1.53 in outpatient records
#: (coronary disease, hypothyroidism), where a Poisson stream gives 1.0 and an even grid
#: 0.3-0.5. A person rarely reports twice in two days, though, so the dispersion has to come
#: from long quiet stretches, not from piling reports onto adjacent days (an episode layout
#: did that: gap median 3 days, 34% of gaps <= 1 day). The layout is a renewal process: a gap
#: is a short follow-up report (`EVENT_SHORT_SHARE`, `EVENT_SHORT_GAP_DAYS`) or a quiet stretch
#: (refractory days plus a log-normal share of the rest of the window, `EVENT_GAP_LOG_SD`). The
#: layout is a function of the case, and an event's slot
#: in it of its evidence id, so benign symptoms, life events and upstream distractor events
#: share one layout and no label enters it.
EVENT_GAP_LOG_SD = 1.5
#: Refractory part of a quiet stretch (days).
EVENT_REFRACTORY_DAYS = 2.0
#: Share of follow-up reports (a second report of the same episode) and their gap (days).
EVENT_SHORT_SHARE = 0.45
EVENT_SHORT_GAP_DAYS = (1.5, 3.5)
#: Smallest gap between two self-reported events, genuine-symptom days included (days).
EVENT_MIN_GAP_DAYS = 1
#: `verify.check_event_spread` fails a case with more than 0.6 of its non-genuine events in the
#: last third of the window; the layout is drawn again (next `attempt`) until at most this
#: share lands there, so the check stays a property of the case and not of luck.
EVENT_LATE_SHARE_MAX = 0.5
EVENT_LAYOUT_ATTEMPTS = 32


def _renewal_days(keys: list[str], case_id: str, T: int, attempt: int) -> list[int]:
    """Nominal days in [7, T] of the events `keys` under layout `attempt` (see above)."""
    n = len(keys)
    span = float(max(1, T - 7))
    # n + 1 stretches: before the first event, n - 1 between events, after the last. An interior
    # stretch is a follow-up report (`EVENT_SHORT_GAP_DAYS`) with `EVENT_SHORT_SHARE`, else a
    # quiet stretch: the refractory days plus a log-normal share of the remaining window.
    short = [0.0] * (n + 1)
    for i in range(1, n):
        if _wear._unit(case_id, "evt_short", attempt, i) < EVENT_SHORT_SHARE:
            a, b = EVENT_SHORT_GAP_DAYS
            short[i] = a + _wear._unit(case_id, "evt_short_len", attempt, i) * (b - a)
    ex = [0.0 if short[i] else math.exp(EVENT_GAP_LOG_SD * _wear._gauss(case_id, "evt_gap", attempt, i))
          for i in range(n + 1)]
    ref = [EVENT_REFRACTORY_DAYS if (0 < i < n and not short[i]) else 0.0 for i in range(n + 1)]
    free = max(0.0, span - sum(ref) - sum(short))
    tot = sum(ex) or 1.0
    t, slots = 7.0, []
    for i in range(n):
        t += short[i] + ref[i] + free * ex[i] / tot
        slots.append(min(T, max(7, int(round(t)))))
    order = sorted(range(n), key=lambda i: (_wear._unit(case_id, "evt_slot", keys[i]), keys[i]))
    out = [0] * n
    for j, i in enumerate(order):
        out[i] = slots[j]
    return out


def _place_self_reports(keys: list[str], case_id: str, T: int, avoid: set[int],
                        attempt: int) -> list[int]:
    """Days for events `keys` under layout `attempt`: each takes the free day nearest its
    nominal day, at least `EVENT_MIN_GAP_DAYS` from `avoid` and from each other (in
    nominal-day order)."""
    taken = set(avoid)
    span = max(1, T - 7)
    nominal = _renewal_days(keys, case_id, T, attempt)
    out = [0] * len(keys)
    for i in sorted(range(len(keys)), key=lambda i: (nominal[i], keys[i])):
        day = nominal[i]
        for gap in dict.fromkeys((EVENT_MIN_GAP_DAYS, 1)):   # a crowded window falls back to 1
            hit = next((c for off in range(0, span + 1) for c in (day + off, day - off)
                        if 7 <= c <= T and all(c + j not in taken for j in range(1 - gap, gap))),
                       None)
            if hit is not None:
                day = hit
                break
        taken.add(day)
        out[i] = day
    return out


def retime_self_reports(evs: list[dict], case_id: str, T: int, avoid: set[int]) -> None:
    """Put self-reported events (benign symptoms, life events, upstream distractor events) on
    the case's burst layout, in place; text and topic are untouched. Days avoid `avoid`
    (genuine-symptom days) and each other; the first layout whose last-third share is at most
    `EVENT_LATE_SHARE_MAX` is used (else the one with the smallest share)."""
    if not evs or T < 7:
        return
    keys = [str(ev.get("evidence_id", i)) for i, ev in enumerate(evs)]
    edge = 7 + 2 * max(3, T - 7) / 3.0
    best = None
    for a in range(EVENT_LAYOUT_ATTEMPTS):
        days = _place_self_reports(keys, case_id, T, avoid, a)
        late = sum(1 for d in days if d >= edge) / len(days)
        if best is None or late < best[0]:
            best = (late, days)
        if late <= EVENT_LATE_SHARE_MAX:
            break
    for ev, d in zip(evs, best[1]):
        ev["source_timestamp"] = d


def plan_benign_evs(f: Facts, event_density: dict, driver: str, T: int,
                    drop: set[str] | None = None, n_inherited: int = 0,
                    used_topics: set[str] | None = None,
                    aliases: list | None = None,
                    cond: tuple[str, ...] | None = None,
                    avoid_days: set[int] | None = None,
                    plugin: bool = False) -> tuple[list[dict], list[dict]]:
    """Benign-event and life-event EVs spread over [7, T] at `event_density`. `plugin`: a plugin
    task's case, the only one that draws the pool's `plugin_only` items.

    `n_inherited`: validated upstream benign events, subtracted from the target
    count. `used_topics`: topics already used upstream (deduplicated).
    `cond=None` takes `Facts.cond_axes`. `avoid_days`: days other events already hold.
    """
    drop = drop or set()
    _sync_pools()
    used_topics = set(used_topics or ())
    cond = normalize_cond_axes(cond if cond is not None else getattr(f, "cond_axes", ()))
    banned = event_proxy_tags(f, driver)
    weeks = event_weeks(T)
    _want = expected_event_counts(event_density, T)     # formula and defaults exist in one place only
    n_sym = max(0, _want["symptom_rate"] - n_inherited)
    n_life = _want["life_event_rate"]

    pool_sym, pool_life, skipped = [], [], []
    for pool, out in ((BENIGN_EVENTS, pool_sym), (LIFE_EVENTS, pool_life)):
        for it in pool:
            if it.get("plugin_only") and not plugin:
                continue
            if it.get("topic") in used_topics:
                skipped.append({"item": it["text"], "reason": f"duplicate_topic:{it['topic']}"})
                continue
            ok, why = _event_ok(it, f, banned, cond=cond)
            # Exclude pool events whose text contains the case's diagnosis alias.
            if ok and aliases:
                blob = f"{it.get('text','')} {it.get('context','')}"
                hit = alias_hit(blob, aliases)
                if hit:
                    ok, why = False, f"ddx_alias_in_benign_text:{hit[0]}"
            (out if ok else skipped).append(it if ok else {"item": it["text"], "reason": why})

    evs: list[dict] = []
    used_days: set[int] = set(avoid_days or ())   # benign/life events must not share a day with each other either
    for kind, pool, n, src in (("B", pool_sym, n_sym, "patient_reported_symptom"),
                               ("L", pool_life, n_life, "patient_reported_context")):
        span = max(1, T - 7)
        # Dropped slots are backfilled so the injected count matches the density.
        usable = [s for s in range(len(pool))
                  if f"EV-{f.case_id}-{kind}{s+1}" not in drop]
        for s in range(len(pool)):
            if f"EV-{f.case_id}-{kind}{s+1}" in drop:
                skipped.append({"item": f"EV-{f.case_id}-{kind}{s+1}",
                                "reason": "dropped_by_verifier"})
        # Pick entries by age-weighted sampling without replacement.
        slots = [s for s in weighted_order(list(pool), f, kind, cond) if s in set(usable)][:n]
        if len(slots) < n:
            # Record why the pool came up short (zero-prior candidates).
            skipped.append({"item": f"EV-{f.case_id}-{kind}",
                            "reason": f"pool_exhausted_after_age_prior:{len(slots)}/{n}"})
        # Days are spread over the emitted count and avoid genuine-symptom days
        # (a day collision cannot be resolved by swapping the event).
        real_days = {int(s["day"]) for s in f.symptoms if int(s["day"]) <= T}
        for i, s in enumerate(slots):
            eid = f"EV-{f.case_id}-{kind}{s+1}"
            it = pool[s % len(pool)]
            # Nominal position: divides n events evenly over [7,T].
            day = 7 + int(round(span * (i + 0.5) / max(1, len(slots))))
            # Deterministic per-item jitter, so injection days differ across items.
            day += _day_jitter(f.case_id, kind, i, amp=_BENIGN_JITTER_AMP)
            day = min(T, max(7, day))
            for off in range(0, span + 1):          # step outward to the nearest free day, avoiding genuine-symptom days and days already used
                for cand in ((day + off), (day - off)):
                    if 7 <= cand <= T and cand not in real_days and cand not in used_days:
                        day = cand
                        break
                else:
                    continue
                break
            used_days.add(day)
            ev = {"evidence_id": eid, "source_type": src, "source_timestamp": day,
                  "symptom" if kind == "B" else "note": it["text"],
                  "context": it["context"], "claim_supported": True,
                  "reliability_status": "reported",
                  # Out-of-band field for the physiology layer; popped before emission.
                  "_topic": it.get("topic")}
            evs.append(ev)
    return evs, skipped


# ============================================================ LLM planning path
_PLAN_PROMPT = """你是"逐天事件指标"设计器。给定一位患者的**基础信息**与**事件密度**,
挑选该注入哪些逐天指标流、并写出若干与结论无关的良性事件。只输出一个 JSON 对象(无多余文字、无围栏)。

## 患者基础信息(你选的东西必须与这些**实际情况相符**)
{facts}

## 事件密度(隐控制变量,决定条数)
{density}

## 可选的逐天指标(只能从这里选;`base` 必须落在给出的合理区间内 —— 这个区间已按本患者的
## BMI/年龄/共病/病历实测体征算过,选错会被判 baseline_matches_raw_facts 不合格)
{catalog}

## 硬约束(违反会被确定性校验器逐项判负并要求你重出)
1. 指标必须有**该患者在库设备**支撑;不得选清单里没有的名字。
2. 指标与事件都**不得携带答案信息**:不能是本例真驱动/共病的代理(禁用 tag:{banned}),
   不能提到结局或归因(禁词示例:复胖、反弹、停药、甲亢、甲减、无效、依从性差、费用)。
3. 良性事件必须是**自限性小毛病或与健康无关的生活事件**,且对这位患者可信
   (体力活动类事件不要派给肥胖/高龄患者)。**绝不能写红旗症状**
   (胸痛、呼吸困难、晕厥、黑便、呕血、抽搐、偏瘫、高热不退、视力骤降等)——
   那不是无关噪声,会变成漏诊陷阱。
4. 事件 `day` 取 [7, {T}] 内的整数,彼此不同日,且**不要挤在临近第 {T} 天的末段**;
   不得与该患者的真实症状同日(真实症状日:{real_days})。
5. 每个事件都要给 `tags` 与 `exertion`(是否需要体力),用于机器校验。
5b. 每个事件都要给 `topic`,**只能从下面这份清单里选**(选清单外的词,该事件会被丢弃)。
   `topic` 是这件小毛病的**生理类别**,生理层按它给指标流加上对应的足迹
   (例如落枕会让睡眠时长与静息心率轻微变化)。挑**最贴近你写的那件事**的那个;
   真的挑不到就填 `none` —— **填 `none` 不扣分**,硬套一个不像的反而会让指标流与文本对不上。
   可选 topic:{topics}
6. `context` 只写**可观察情境**,并**必须**同时给出 `context_facet`,取值只能是:
   `measure`(数值/化验/设备读数)· `neg`(否定性发现)· `sign`(可观察体征/生活事实)·
   `course`(做了什么、结果如何)· `therapy`(剂量/起始/换机等治疗时间线事实)。
   写不出属于哪一类的,说明它是**归因或解释**,把 `context` 留空 —— 留空不扣分,写错会被丢。

## 上一轮被判负的项与原因(必须换掉/改掉)
{feedback}

## 输出 JSON
{{"streams":[{{"name":"<清单里的名字>","base":<数值>}},...],
 "benign_events":[{{"day":<int>,"text":"...","context":"...","context_facet":"course","tags":["musculoskeletal"],"exertion":false,"topic":"<清单里的 topic 或 none>"}},...],
 "life_events":[{{"day":<int>,"text":"...","context":"...","context_facet":"sign","tags":["logistics"],"exertion":false,"topic":"<清单里的 topic 或 none>"}},...]}}
条数要求:streams 尽量覆盖所有设备支撑得起、且非答案相关的指标;benign_events {n_sym} 个;life_events {n_life} 个。"""


def _catalog_for(f: Facts, driver: str) -> list[dict]:
    """Metrics offered to the LLM: device-supported, not answer-relevant, with
    this patient's plausible baseline range."""
    banned = proxy_tags(f, driver)
    out = []
    for m in METRICS:
        if m.devices and not (set(m.devices) & set(f.devices)):
            continue
        if set(m.tags) & banned:
            continue
        exp = (float(f.baseline_vitals[m.vital_key])
               if (m.vital_key and f.baseline_vitals.get(m.vital_key) is not None)
               else _adj(m.name, m.base, f))
        out.append({"name": m.name, "unit": m.unit,
                    "device": list(m.devices) or ["self_report"],
                    "base_range": [round(exp - m.tol, 2), round(exp + m.tol, 2)],
                    "hard_range": list(m.hard_range), "tags": list(m.tags)})
    return out


def plan_via_llm(f: Facts, event_density: dict, driver: str, T: int, dispatch,
                 drop: set[str] | None = None, feedback: dict | None = None,
                 n_inherited: int = 0) -> tuple[list[StreamPlan], list[dict], list[dict]]:
    """Let the LLM write the benign and life events (text, day, topic), checked by
    `verify.py`. The prompt also asks for streams and baselines; those are not adopted:
    the daily metrics are `plan_streams`'s, the same as on the deterministic channel, so
    no stream value or stream choice comes from the model. Returns `([], events, skipped)`."""
    from haenv_kernel.solver import _extract_json                     # kernel: JSON extraction tolerant of banners/fences

    drop = drop or set()
    weeks = event_weeks(T)
    _want = expected_event_counts(event_density, T)     # formula and defaults exist in one place only
    n_sym = max(0, _want["symptom_rate"] - n_inherited)
    n_life = _want["life_event_rate"]
    catalog = _catalog_for(f, driver)
    prompt = _PLAN_PROMPT.format(
        facts=json.dumps({"case_id": f.case_id, "disease": f.disease, "drug": f.drug,
                          "devices": list(f.devices), "comorbidities": list(f.comorbidities),
                          "age_range": f"{f.age_lo}-{f.age_lo+4}", "sex": f.sex,
                          "bmi": f.bmi, "start_weight": f.start_weight,
                          "baseline_vitals": f.baseline_vitals,
                          "obese_or_low_mobility": f.heavy, "elderly": f.elderly},
                         ensure_ascii=False),
        density=json.dumps(event_density, ensure_ascii=False),
        catalog=json.dumps(catalog, ensure_ascii=False),
        banned=sorted(event_proxy_tags(f, driver)), T=T,
        real_days=sorted({s["day"] for s in f.symptoms if s["day"] <= T}),
        n_sym=n_sym, n_life=n_life,
        topics=", ".join(sorted(pool_event_topics())),      # controlled option set, excludes genuine-symptom topics
        feedback=json.dumps(feedback or {}, ensure_ascii=False) or "(首轮,无)")
    data = _extract_json(dispatch(prompt, require_json=True))

    plans, skipped = [], []
    evs: list[dict] = []
    for kind, key, src in (("B", "benign_events", "patient_reported_symptom"),
                           ("L", "life_events", "patient_reported_context")):
        for i, item in enumerate(data.get(key) or [], 1):
            eid = f"EV-{f.case_id}-{kind}{i}"
            if eid in drop:
                skipped.append({"item": eid, "reason": "dropped_by_verifier"})
                continue
            try:
                day = int(item["day"])
            except (KeyError, TypeError, ValueError):
                skipped.append({"item": eid, "reason": "llm_bad_day"})
                continue
            ev = {"evidence_id": eid, "source_type": src, "source_timestamp": day,
                  "symptom" if kind == "B" else "note": str(item.get("text", "")).strip(),
                  "claim_supported": True, "reliability_status": "reported"}
            # The model reports the context facet; an unknown facet drops the context.
            from .overlay import EMITTABLE
            ctx, _ = sanitize_context(str(item.get("context", "")).strip())
            facet = str(item.get("context_facet", "")).strip().lower()
            if ctx and facet not in EMITTABLE:
                log.info("[events] %s context facet=%r is not emittable; dropped: %r", eid, facet, ctx)
                ctx = ""
            if ctx:
                ev["context"] = ctx
            ev["_claim"] = {"tags": tuple(item.get("tags") or ()),      # for item-by-item validation (stripped downstream)
                            "exertion": bool(item.get("exertion")),
                            "topic": f"llm:{ev.get('symptom') or ev.get('note')}"}
            # Accept a model-chosen `_topic` only from `pool_event_topics()`
            # (never a genuine-symptom topic); otherwise the event has no footprint.
            _tp = str(item.get("topic", "")).strip()
            if _tp and _tp != "none":
                if _tp in pool_event_topics():
                    ev["_topic"] = _tp
                else:
                    log.info("[events] %s topic=%r is not in the controlled vocabulary; treated as having no footprint", eid, _tp)
            evs.append(ev)
    return plans, evs, skipped


#: Pool events that fill an LLM-planned case up to its density take slot ids from here
#: on (`EV-<case>-B101`...), clear of the model's own `B<n>` / `L<n>`.
POOL_BACKFILL_ID_OFFSET = 100


def shown_drugs(case_id: str, disease, comorbidities, drug: str) -> list[str]:
    """The medicines a plugin task's record shows the patient taking: the primary drug and the concurrent
    medicines the world gives for the known conditions (`med_course.concurrent_medicines`)."""
    from . import med_course
    cm = med_course.concurrent_medicines(case_id, disease, comorbidities or [], drug, 1)
    return ([drug] if drug else []) + [m["drug"] for m in cm]


def side_effect_topics(drugs) -> frozenset[str]:
    """Benign pool topics that coincide with a known side effect of one of `drugs`
    (`registry/drug_side_effects.yaml`)."""
    d = (_cached_yaml(_rp("drug_side_effects.yaml")) or {}).get("drugs") or {}
    return frozenset(t for x in drugs for t in (d.get(x) or {}).get("topics") or ())


#: Words a patient of the given sex does not say about herself or himself (grooming that belongs to the
#: other sex; the anatomical words are `gates.SEX_EXCLUSIVE`).
SEX_LAY_EXCLUSIVE = {"F": ("刮胡子", "剃须"), "M": ()}


def sex_words(sex: str | None) -> tuple[str, ...]:
    from .gate_tables import SEX_EXCLUSIVE
    s = str(sex or "").upper()[:1]
    return tuple(SEX_EXCLUSIVE.get(s, ())) + SEX_LAY_EXCLUSIVE.get(s, ())


def symptom_lay(sex: str | None = None) -> dict[str, list[str]]:
    """`registry/symptom_lay.yaml`: a symptom phrase of a condition template or an explained-away
    finding -> the hand-written ways a patient of `sex` says it (`lay_F` / `lay_M` when the entry has
    them, else `lay`)."""
    d = _cached_yaml(_rp("symptom_lay.yaml")) or {}
    k = f"lay_{str(sex or '').upper()[:1]}"
    return {str(p): [str(t) for t in (v.get(k) or v["lay"])] for p, v in (d.get("phrases") or {}).items()}


def symptom_lay_every() -> dict[str, list[str]]:
    """Every lay sentence of each phrase, all sexes (for the audits that recognise an entered sentence)."""
    d = _cached_yaml(_rp("symptom_lay.yaml")) or {}
    return {str(p): [str(t) for k, ts in v.items() if k == "lay" or k.startswith("lay_") for t in ts]
            for p, v in (d.get("phrases") or {}).items()}


def enter_symptoms(ledger: list[dict], sex: str | None = None) -> dict[str, str]:
    """Every symptom sentence of a plugin task's ledger is entered as a patient of `sex` says it: a
    template, finding or pool phrase takes one of its hand-written lay sentences (`symptom_lay`), drawn
    by the evidence id and different from the sentences already entered in the case. Half-width comma,
    no closing stop. Returns `{evidence_id: phrase before entry}` for the verifier-side audit."""
    from . import rng as _rng
    lay = symptom_lay(sex)
    src, said = {}, set()
    for e in ledger:
        if e.get("source_type") != "patient_reported_symptom":
            continue
        eid = str(e["evidence_id"])
        src[eid] = t = str(e.get("symptom"))
        out = t
        if t in lay:
            ways = [w for w in lay[t] if w not in said] or lay[t]
            out = _rng.pick(ways, "symptom_lay", eid, t)
        said.add(out)
        e["symptom"] = out.replace("，", ",").replace("。", ",").strip().rstrip("、,")
    return src


def _pool_backfill(f: Facts, ed: dict, driver: str, T: int, drop: set[str],
                   llm_evs: list[dict], inh_evs: list[dict], aliases) -> tuple[list[dict], list[dict]]:
    """Pool events that bring the LLM planner's events up to the declared density.

    The planner is asked once per case; when the verifier drops some of its events or
    of the inherited ones, the shortfall comes from the benign/life pools (as on the
    deterministic channel), on days no other event holds."""
    want = expected_event_counts(ed, T)
    n_b = sum(1 for e in llm_evs if str(e["evidence_id"]).rsplit("-", 1)[-1].startswith("B"))
    n_l = sum(1 for e in llm_evs if str(e["evidence_id"]).rsplit("-", 1)[-1].startswith("L"))
    need = {"B": max(0, want["symptom_rate"] - len(inh_evs) - n_b),
            "L": max(0, want["life_event_rate"] - n_l)}
    if not any(need.values()):
        return [], []
    off = POOL_BACKFILL_ID_OFFSET
    pre = f"EV-{f.case_id}-"

    def to_pool(eid: str) -> str:
        k, n = eid[len(pre)], int(eid[len(pre) + 1:])
        return f"{pre}{k}{n - off}"
    pool_drop = {to_pool(d) for d in drop
                 if d.startswith(pre) and d[len(pre):len(pre) + 1] in "BL"
                 and d[len(pre) + 1:].isdigit() and int(d[len(pre) + 1:]) > off}
    used_topics = {t for e in inh_evs
                   if (t := (inherited_profile(e.get("symptom", "")) or {}).get("topic"))}
    taken = {int(e["source_timestamp"]) for e in llm_evs + inh_evs}
    evs, skipped = plan_benign_evs(f, ed, driver, T, pool_drop,
                                   n_inherited=want["symptom_rate"] - need["B"],
                                   used_topics=used_topics, aliases=aliases, avoid_days=taken)
    out, n_seen = [], {"B": 0, "L": 0}
    for e in evs:
        k = str(e["evidence_id"]).rsplit("-", 1)[-1][0]
        if n_seen[k] >= need[k]:
            continue
        n_seen[k] += 1
        e["evidence_id"] = f"{pre}{k}{int(str(e['evidence_id']).rsplit('-', 1)[-1][1:]) + off}"
        out.append(e)
    return out, skipped


# ============================================================ Injection
def inject(raw, cs, premise, driver: str, drop: set[str] | None = None,
           dispatch=None, feedback: dict | None = None) -> tuple[object, dict]:
    """Inject daily metric streams + event EVs into the RawCase (the caller
    deep-copies). Returns (raw, manifest); the manifest lists every injected or
    skipped item for `verify.py`:
      {"streams":[{name, n_points, step_days, base_eff, source, unit, device}...],
       "events":[{evidence_id, kind, day, text, context, tags}...],
       "skipped":[{item, reason}...], "T":..., "end_day":...}
    Gold fields are never touched (CC-1).
    """
    f = facts_of(cs)
    from .registry import GOLD_EVIDENCE  # deferred import: build depends on this module, a top-level import would cycle
    world_signals = {g["signal"] for g in GOLD_EVIDENCE.values()}
    ed = dict(premise.event_density or {})
    T = int(raw.prediction_context["prediction_time_T"])
    win = str(raw.prediction_context.get("prediction_window", "281d")).rstrip("d")
    end_day = T + (int(win) if win.isdigit() else 281)

    llm_evs: list[dict] = []
    # The daily metrics are `plan_streams`'s on both channels; the LLM writes events only.
    plans, skipped = plan_streams(f, ed, driver, drop,
                                  base_signals=world_layer_base_signals(raw, world_signals))
    if dispatch is not None:
        n_inh = sum(1 for e in raw.evidence_ledger
                    if str(e.get("evidence_id", "")).rsplit("-", 1)[-1].startswith("D")
                    and str(e.get("evidence_id", "")) not in (drop or set()))
        _, llm_evs, _sk = plan_via_llm(f, ed, driver, T, dispatch, drop,
                                       feedback=feedback, n_inherited=n_inh)
        skipped += _sk
    mine = {p.spec.name for p in plans}

    # Streams planned here replace upstream copies of the same name; other
    # upstream aux streams are kept as `inherited_metric` and validated.
    stream_rows = []
    for p in plans:
        _over = p.spec.name in (raw.longitudinal_data or {})
        raw.longitudinal_data[p.spec.name] = render_stream(
            p, end_day, seed=str(getattr(raw, "case_id", "") or ""))
        stream_rows.append({"name": p.spec.name, "unit": p.spec.unit, "step_days": p.step_days,
                            "base_eff": p.base_eff, "source": p.source, "kind": "daily_metric",
                            "device": list(p.spec.devices) or ["self_report"],
                            "n_points": len(raw.longitudinal_data[p.spec.name]),
                            "tags": list(p.spec.tags),
                            "replaced_upstream_copy": _over})
    # Derived streams, once the parents exist; no parent, no derived stream.
    banned_tags = proxy_tags(f, driver)
    for _dname, _dpts in _wear.derive_streams(
            f, raw.longitudinal_data or {}, str(getattr(raw, "case_id", "") or "")).items():
        _dm = METRIC_BY_NAME.get(_dname)
        if not _dm or not _dpts or _dname in drop:
            continue
        if _dm.devices and not (set(_dm.devices) & set(f.devices)):
            continue
        if set(_dm.tags) & banned_tags:
            skipped.append({"item": _dname, "reason": "driver_or_comorbid_proxy"})
            continue
        raw.longitudinal_data[_dname] = _dpts
        stream_rows.append({"name": _dname, "unit": _dm.unit,
                            "step_days": 1, "base_eff": _dm.base, "source": "derived",
                            "kind": "daily_metric", "device": list(_dm.devices),
                            "n_points": len(_dpts), "tags": list(_dm.tags)})
    world = set(world_signals or ())
    for name in list(raw.longitudinal_data):
        if name in world:
            # World-layer gold evidence: not subject to the neutrality checks.
            m = METRIC_BY_NAME.get(name)
            pts = raw.longitudinal_data[name]
            if m and pts:
                # Non-gold world-layer streams get deterministic alternating
                # jitter (fixed multiple of `tol`, zero drift, clamped to
                # `hard_range`), so whether a stream moves does not reveal
                # which one is gold.
                _gold_sig = _gold_signal_of_driver(driver)
                # If the gold stream is unknown, no stream is jittered.
                if _gold_sig and name != _gold_sig:
                    from . import rng as _rng
                    _amp = 1.2 * float(getattr(m, "tol", 0.0) or 0.5)
                    _lo, _hi = float(m.hard_range[0]), float(m.hard_range[1])
                    for _i, _q in enumerate(pts):
                        _ph = 1.0 if (_i + int(_rng.unit(str(getattr(raw, "case_id", "?")),
                                                         name, "jit") * 2)) % 2 else -1.0
                        _v = float(_q["value"]) + _ph * _amp
                        _q["value"] = round(min(_hi, max(_lo, _v)), 3)
                steps = [b["ts"] - a["ts"] for a, b in zip(pts, pts[1:])] or [1]
                stream_rows.append({"name": name, "unit": m.unit, "step_days": steps[0],
                                    "base_eff": None, "source": "world_layer",
                                    "kind": "gold_evidence_metric", "device": list(m.devices),
                                    "n_points": len(pts), "tags": list(m.tags),
                                    # whether noise was injected is recorded
                                    "jittered_non_gold": name != _gold_sig})
            continue
        if name in AUX_WHITELIST and name not in mine:
            if name in drop:                       # judged failing by validation last round -> dropped this round
                raw.longitudinal_data.pop(name, None)
                skipped.append({"item": name, "reason": "dropped_by_verifier"})
                continue
            if name in _wear.DERIVED_BINDINGS:
                # A derived name only exists if produced from its parent above.
                if not any(r["name"] == name for r in stream_rows):
                    raw.longitudinal_data.pop(name, None)
                    skipped.append({"item": name, "reason": "derived_without_parent"})
                continue
            if _wear.is_calibrated(name):
                # Unplanned upstream copies of calibrated streams are removed.
                raw.longitudinal_data.pop(name, None)
                skipped.append({"item": name, "reason": "calibrated_stream_not_planned"})
                continue
            m = METRIC_BY_NAME[name]
            pts = raw.longitudinal_data[name]
            steps = [b["ts"] - a["ts"] for a, b in zip(pts, pts[1:])] or [1]
            stream_rows.append({"name": name, "unit": m.unit, "step_days": steps[0],
                                "base_eff": None, "source": "upstream_injector",
                                "kind": "inherited_metric", "device": list(m.devices),
                                "n_points": len(pts), "tags": list(m.tags)})

    # Upstream benign EVs (`-D` suffix) are validated like the others and count
    # toward the density quota.
    # A plugin task's case (external latent) has no kernel distractor events (a fixed list of ten): its
    # whole symptom quota is drawn from the registry pool, seeded by the case.
    from . import external_gold as _EG
    plugin = bool((premise.meta or {}).get(_EG.SLOT))
    inh_evs, ev_rows = [], []
    kept_ledger = []
    for e in raw.evidence_ledger:
        eid = str(e.get("evidence_id", ""))
        if eid.rsplit("-", 1)[-1].startswith("D"):
            if plugin:
                continue
            if eid in drop:
                skipped.append({"item": eid, "reason": "dropped_by_verifier"})
                continue
            inh_evs.append(e)
        kept_ledger.append(e)
    raw.evidence_ledger = kept_ledger
    if plugin:
        raw.adjudication["distractor_evidence_ids"] = []

    # The case's aliases plus the registry's `leak_only` words (as `verify._ddx_aliases`).
    from .overlay import leak_aliases_for
    _ddx = getattr(cs, 'ddx', None) or {}
    aliases = leak_aliases_for(_ddx.get('spec_id'), _ddx.get('aliases'))
    real_evs, sk1 = plan_real_symptom_evs(f, T, drop, aliases=aliases)
    backfill: set[str] = set()
    if dispatch is not None:
        benign_evs, sk2 = llm_evs, []
        _extra, sk2 = _pool_backfill(f, ed, driver, T, drop, llm_evs, inh_evs, aliases)
        backfill = {e["evidence_id"] for e in _extra}
        benign_evs = benign_evs + _extra
    else:
        used_topics = {t for e in inh_evs
                       if (t := (inherited_profile(e.get("symptom", "")) or {}).get("topic"))}
        if plugin:      # no benign complaint that is a known side effect of a drug the record shows
            used_topics |= side_effect_topics(shown_drugs(f.case_id, f.disease, f.comorbidities, f.drug))
        benign_evs, sk2 = plan_benign_evs(f, ed, driver, T, drop, n_inherited=len(inh_evs),
                                          used_topics=used_topics, aliases=aliases, plugin=plugin)
    skipped += [x for x in sk1 + sk2 if isinstance(x, dict) and "reason" in x]
    # Self-report timing is code-owned on both channels: bursty, not an even grid.
    retime_self_reports(inh_evs + benign_evs, f.case_id, T,
                        avoid={int(s["day"]) for s in f.symptoms if int(s["day"]) <= T}
                        | {int(e["source_timestamp"]) for e in real_evs})

    # Out-of-band fields (`_claim`, `_topic`) go only into the manifest and are
    # popped before the events enter `evidence_ledger`.
    claims = {ev["evidence_id"]: ev.pop("_claim") for ev in benign_evs if "_claim" in ev}
    topics = {ev["evidence_id"]: ev.pop("_topic") for ev in benign_evs if "_topic" in ev}
    real_topics = {ev["evidence_id"]: ev.pop("_topic") for ev in real_evs if "_topic" in ev}
    # Findings layer (off by default): only `role: screening` items are rendered.
    _sym_days = [int(s["day"]) for s in (getattr(f, "symptoms", None) or [])
                 if isinstance(s, dict) and int(s.get("day", -1)) >= 0]
    _pre = min(_sym_days) if _sym_days else None
    lab_evs = _render_findings_if_enabled(raw, normal_before_day=_pre)
    look_evs = _render_lookalikes(raw, cs, T)
    raw.evidence_ledger += real_evs + benign_evs + look_evs + lab_evs

    # Caps the number of context entries per facet; see `cap_context_facets`.
    cap_context_facets(raw)
    raw.evidence_ledger.sort(key=lambda e: e.get("source_timestamp", 0))

    for ev in inh_evs:
        ev_rows.append({"evidence_id": ev["evidence_id"], "kind": "inherited_event",
                        "day": ev.get("source_timestamp", -1), "text": ev.get("symptom", ""),
                        "context": ev.get("context", ""), "source": "upstream_injector"})
    for ev in real_evs:
        ev_rows.append({"evidence_id": ev["evidence_id"], "kind": "real_symptom",
                        "day": ev["source_timestamp"], "text": ev.get("symptom", ""),
                        "context": ev.get("context", ""), "source": "raw_case",
                        # From the un-scrubbed text (see `plan_real_symptom_evs`).
                        "topic": real_topics.get(ev["evidence_id"])})
    for ev in look_evs:
        ev_rows.append({"evidence_id": ev["evidence_id"], "kind": "lookalike",
                        "day": ev["source_timestamp"], "text": ev.get("symptom", ""),
                        "context": ev.get("context", ""), "source": "registry:lookalikes"})
    for ev in benign_evs:
        kind = "benign_symptom" if ev["evidence_id"].rsplit("-", 1)[-1].startswith("B") else "life_event"
        ev_rows.append({"evidence_id": ev["evidence_id"], "kind": kind,
                        "day": ev["source_timestamp"],
                        "text": ev.get("symptom") or ev.get("note", ""),
                        "context": ev.get("context", ""),
                        "source": ("llm" if dispatch is not None and ev["evidence_id"] not in backfill
                                   else "pool"),
                        "claim": claims.get(ev["evidence_id"]),
                        # None on the LLM path: this event has no kernel.
                        "topic": topics.get(ev["evidence_id"])})

    # Q-side injection ledger; not stored in `raw.adjudication` (Iron law 2).
    # Lookalike distractors are listed separately because the judge expects
    # them to be merged in, unlike benign events.
    _look_ids = [r["evidence_id"] for r in ev_rows if r["kind"] == "lookalike"]
    injected = {
        "benign_evidence_ids": [r["evidence_id"] for r in ev_rows
                                if r["kind"] not in ("real_symptom", "lookalike")],
        "real_symptom_evidence_ids": [r["evidence_id"] for r in ev_rows if r["kind"] == "real_symptom"],
        "lookalike_evidence_ids": _look_ids,
        "daily_streams": [r["name"] for r in stream_rows],
        # `(evidence_id, kind, day, topic)` for the physiology layer; events
        # without a topic are listed too.
        "event_schedule": [{"evidence_id": r["evidence_id"], "kind": r["kind"],
                            "day": int(r["day"]), "topic": r.get("topic")}
                           for r in ev_rows if int(r.get("day", -1)) >= 0],
    }
    # Coupling reads only `Facts`, so it runs before the physiology layer.
    injected["coupling"] = _apply_coupling(f)

    # The injector's own render, before physiology noise and event footprints:
    # the neutrality checks apply to this view, since footprints are world
    # evidence.
    _injector_view = {r["name"]: _copy.deepcopy(raw.longitudinal_data.get(r["name"]) or [])
                      for r in stream_rows if r.get("kind") == "daily_metric"}
    physio_report = _apply_physio_if_enabled(raw, cs.case_id, injected["event_schedule"])
    # The panel follows the final streams: TC from same-day LDL/TG, declared
    # screening items shown on the stream that carries them.
    if lab_evs:
        from .findings_render import align_panel_to_streams
        align_panel_to_streams(raw, normal_before_day=_pre)
        # `build.observe_labs` draws the lab streams again after the events; it re-runs
        # the alignment on the observed streams with the same onset day.
        PANEL_ALIGN_PENDING[str(cs.case_id)] = _pre
    # Remove the ground-truth trajectory before the report enters `injected`.
    _clean_ld = physio_report.pop("_clean_longitudinal_data", None) if physio_report else None
    if physio_report is not None:
        injected["physio"] = physio_report

    # Drop derived streams whose parent is gone, once the world is final.
    _orphans = _wear.prune_orphans(raw.longitudinal_data or {})
    if _orphans:
        stream_rows = [r for r in stream_rows if r["name"] not in _orphans]
        skipped.extend({"item": o, "reason": "derived_parent_dropped"} for o in _orphans)

    manifest = {"case_id": cs.case_id, "T": T, "end_day": end_day,
                "event_density": ed, "streams": stream_rows, "events": ev_rows,
                "skipped": skipped, "injected": injected}
    if _clean_ld is not None:
        # Verifier-only (`_` prefix, excluded from the emission surface).
        manifest["_clean_longitudinal_data"] = _clean_ld
    if physio_report is not None:
        # verifier-only, same convention as `_clean_longitudinal_data`.
        manifest["_injector_view"] = _injector_view
    log.info("[events] %s injected %d daily metric streams (%d inherited) + %d events (%d inherited) · excluded %d",
             cs.case_id, len(stream_rows),
             sum(1 for r in stream_rows if r["kind"] == "inherited_metric"),
             len(ev_rows), len(inh_evs), len(skipped))
    return raw, manifest

def _render_lookalikes(raw, cs, T: int) -> list[dict]:
    """Render this condition's lookalike distractors as EVs, in the same format
    as benign events.

    A lookalike looks like a minor issue but is a manifestation of the gold
    diagnosis, so it should be merged in; benign events should not. At most 2
    per item, days in [7, T-3]. Content is `review: pending`.
    """
    # Answer-relevant by design, so it is placed like a genuine symptom;
    # `verify.PROXY_EXEMPT_KINDS` exempts only the proxy check.
    from . import rng                                   # noqa: F401
    from .registry import load_lookalikes

    sid = str((cs.latent or {}).get("ddx_spec_id") or "")
    pool = (load_lookalikes().get(sid) or ()) if sid else ()
    if not pool:
        return []
    n = min(len(pool), 1 + int(rng.unit(cs.case_id, "lookalike", "n") * 2))   # 1 or 2
    lo, hi = 7, max(8, T - 3)
    # Avoid genuine-symptom days: search forward for the first free day.
    _real = {int(x.get("day")) for x in ((getattr(cs, "raw", None) or {}).get("symptoms") or [])
             if isinstance(x, dict) and x.get("day") is not None and int(x["day"]) <= T}
    _used: set[int] = set()
    out = []
    for i in range(n):
        it = pool[i % len(pool)]
        day = lo + int(rng.unit(cs.case_id, "lookalike", f"d{i}") * (hi - lo))
        for _ in range(hi - lo + 1):                   # bounded: at most one full loop, never infinite
            if day not in _real and day not in _used:
                break
            day = lo + (day - lo + 1) % max(1, hi - lo)
        _used.add(int(day))
        # Prefix `K`: the findings layer uses `L`, and ids are classified by prefix.
        out.append({"evidence_id": f"EV-{cs.case_id}-K{i + 1}",
                    "source_type": "patient_reported_symptom",
                    "source_timestamp": int(day),
                    "symptom": str(it["text"]),
                    "context": str(it.get("context") or ""),
                    "claim_supported": True,
                    "reliability": "self_reported"})
    return out


#: Per case id, the onset day the panel was aligned with (`align_panel_to_streams`), for
#: `build.observe_labs` to re-align the panel on the observed lab streams.
PANEL_ALIGN_PENDING: dict[str, int | None] = {}


def _render_findings_if_enabled(raw, *, normal_before_day: int | None = None) -> list[dict]:
    """Render the findings panel if the switch is on; `[]` when off."""
    if not FINDINGS_ENABLED[0]:
        return []
    try:
        from .findings_render import render_findings
        from .registry import condition_findings_for_case, load_findings
        from .wq import resolve_path
        sid = resolve_path(raw, "W.adjudication.ddx.spec_id")
        if not isinstance(sid, str):
            return []
        vocab = load_findings()
        # Comorbidity profiles are composed from their components.
        prof = condition_findings_for_case(vocab).get(sid)
        # A missing profile is logged and rendered as all-normal.
        if prof is None:
            log.warning("[findings] %s has no findings profile; rendered as all-normal (missing content, not a design choice)", sid)
            prof = {"findings": ()}
        T = int((raw.prediction_context or {}).get("prediction_time_T") or 84)
        from .findings_render import render_routine_panel
        # Sex selects sex-specific reference ranges (`SEX_REF`).
        _sex = (getattr(raw, "user_profile", None) or {}).get("sex")
        if not _sex:
            log.warning("[findings] %s has no recorded sex; reference bands are not sex-specific "
                        "(ambiguity left unresolved, upstream value kept)", getattr(raw, "case_id", "?"))
        # Metrics already present as time series are not repeated in the panel.
        _ts = tuple(getattr(raw, "longitudinal_data", None) or ())
        # Draws before the first genuine-symptom day render as normal;
        # `normal_before_day=None` leaves the panel as-is.
        return render_routine_panel(str(raw.case_id), prof, vocab, T,
                                    sex=_sex, ts_signals=_ts,
                                    normal_before_day=normal_before_day)
    except Exception as e:                      # a rendering failure must never be silently swallowed
        log.error("[findings] rendering failed %s: %s", getattr(raw, "case_id", "?"), e)
        raise


def _apply_physio_if_enabled(raw, case_id: str, schedule: list[dict]) -> dict | None:
    """Run the physiology layer if enabled; `None` when off. Errors propagate."""
    if not PHYSIO_ENABLED[0]:
        return None
    from pathlib import Path as _P

    from .physio import apply as _pa
    from .physio import kernel as _pk

    streams = _pa.load_stream_registry(_rp("physio_streams.yaml"))
    kernels = _pk.load_physio_registry(_rp("physio_kernels.yaml"))
    # Keep the pre-noise ground-truth trajectory. The kernel's slope gate bounds
    # the true rate of weight change, so slope and anchors are judged on this
    # copy and only the value range on the observation. It is verifier-only:
    # never in `injected`, `raw` or `manifest["injected"]`.
    clean_ld = _copy.deepcopy(raw.longitudinal_data)
    new_ld, summary = _pa.apply_physio(case_id, raw.longitudinal_data, streams,
                                       events=schedule, kernels=kernels)
    _round_to_declared_digits(new_ld)
    raw.longitudinal_data = new_ld
    covered = sum(1 for r in schedule
                  if r.get("topic") in kernels.emitted_topics())
    return {
        "_clean_longitudinal_data": clean_ld,   # verifier-only, `_` prefix follows the same convention as other out-of-band fields
        "clip_rate": summary.projection.clip_rate,
        "n_clipped": summary.projection.n_clipped,
        "n_points": summary.projection.n_total,
        "n_series": len(summary.distributions),
        "n_events": len(schedule),
        "n_events_with_kernel": covered,
        "n_dist_violations": len(summary.violations),
        "dist_violations": list(summary.violations)[:20],
    }
