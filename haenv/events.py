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
import hashlib
import json
import logging
import math
import pathlib as _pathlib
from dataclasses import dataclass, field
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data
from haenv import streams as _streams

log = logging.getLogger("haenv.events")

# Rendered daily metrics that are auxiliary: exempt from device-inventory
# validation, not a predicted physiological indicator. Derived from the stream
# manifest (`registry/streams.yaml`); a subset of the kernel's `synth.AUX_SIGNALS`.
AUX_WHITELIST = set(_streams.aux_metrics())


# ============================================================ raw_case baseline facts
@dataclass
class Facts:
    """raw_case's baseline facts; the daily metrics must be consistent with them."""
    case_id: str
    disease: str
    drug: str
    devices: tuple[str, ...]
    comorbidities: tuple[str, ...]
    age_lo: int
    sex: str
    start_weight: float
    nadir_weight: float
    bmi: float | None
    baseline_vitals: dict            # baseline vitals explicitly stated in the
                                      # raw case (e.g. resting HR 102 / lowest
                                      # overnight SpO2 86)
    symptoms: tuple[dict, ...]       # genuine symptoms recorded in raw_case
                                      # [{day,text,context}]
    sampling_days: int
    target_event: str = "weight_regain"   # the predicted outcome (decides which
                                           # events would "hand the solver a
                                           # ready-made attribution")
    # Whether this task type judges this outcome at all (default True). For a
    # diagnosis task it is False, so outcome-explaining tags are not banned.
    outcome_graded: bool = True
    # Conditioning axes for distractor events; empty = unconditioned sampling.
    cond_axes: tuple[str, ...] = ()

    @property
    def hyperthyroid(self) -> bool:
        """Whether the patient is currently hyperthyroid (`_treated` excluded).

        Unlike `proxy_tags`, which includes `_treated`: treatment normalizes
        the metabolic rate, but the history is still a differential clue.
        """
        return any(c.startswith("hyperthyroidism") and not c.endswith("_treated")
                   for c in self.comorbidities)

    @property
    def hypothyroid(self) -> bool:
        return any(c.startswith("hypothyroid") for c in self.comorbidities)

    @property
    def osa(self) -> bool:
        return any(c.upper() == "OSA" for c in self.comorbidities)

    @property
    def heavy(self) -> bool:
        """Clinically significant obesity (affects the actual level of activity /
        SpO2 / resting HR). BMI takes priority; falls back to the weight band
        when BMI is missing."""
        return (self.bmi or 0) >= 34 or self.start_weight >= 105

    @property
    def elderly(self) -> bool:
        return self.age_lo >= 65


def facts_of(cs) -> Facts:
    from .job import raw_field as _raw_field
    raw = cs.raw
    age = str(_raw_field(raw, "age_range"))
    try:
        age_lo = int(age.split("-")[0])
    except ValueError:
        age_lo = 45
    # Defaults come only from `job.CaseSpec.RAW_DEFAULTS` (via `_raw_field`).
    start = float(_raw_field(raw, "start_weight"))
    return Facts(
        case_id=cs.case_id,
        disease=_raw_field(raw, "disease"),
        drug=_raw_field(raw, "drug"),
        devices=tuple(raw.get("devices", []) or []),
        comorbidities=tuple(_raw_field(raw, "comorbidities") or []),
        age_lo=age_lo, sex=_raw_field(raw, "sex"),
        start_weight=start, nadir_weight=float(_raw_field(raw, "nadir_weight")),
        bmi=(float(raw["bmi"]) if raw.get("bmi") else None),
        baseline_vitals=dict(raw.get("baseline_vitals", {}) or {}),
        symptoms=tuple({"day": int(s["day"]), "text": s.get("text", ""),
                        "context": s.get("context", "")} for s in (raw.get("symptoms") or [])),
        sampling_days=int(_raw_field(raw, "sampling_days")),
        target_event=str(cs.latent.get("target_event", "weight_regain")),
        outcome_graded=not bool((raw.get("adjudication") or {}).get("ddx")
                                or (cs.latent or {}).get("ddx_spec_id")),
        # Not in `job.LATENT_REGISTRY`; set only by an explicit caller.
        cond_axes=normalize_cond_axes((cs.latent or {}).get("distractor_cond_axes")),
    )


# ============================================================ daily metric catalog
@dataclass
class MetricSpec:
    name: str
    devices: tuple[str, ...]          # any one in inventory suffices for daily
                                       # sampling (empty = self-reported diary,
                                       # needs no device)
    unit: str
    base: float                       # default healthy-person baseline
    amp: float                        # intraday/day-to-day physiological
                                       # fluctuation amplitude
    period: int                       # deterministic waveform period (days;
                                       # unrelated to T / the reversal week)
    ndigits: int                      # 0 = round to integer
    hard_range: tuple[float, float]   # hard physiological range: any point
                                       # outside it fails validation
    tol: float                        # tolerance between the mean and the
                                       # profile's expected baseline
    vital_key: str | None = None      # if the raw case states this vital's
                                       # baseline, defer to it
    tags: tuple[str, ...] = ()        # used for "driver proxy" exclusion


#: Baseline adjustments per population trait. Order matters (floating-point
#: addition is not associative). Entry shapes:
#:   `("<a boolean attribute on Facts>", delta)`  -- add delta when it matches
#:   `("age_decade_over", (anchor, delta))`      -- add delta per decade past anchor
#:   `("drug_prefix", (prefix, delta))`          -- add delta when the drug name
#:                                                  matches this prefix
BASELINE_ADJUST: dict[str, tuple[tuple[str, object], ...]] = {
    "steps":            (("heavy", -2200), ("elderly", -1500), ("osa", -600)),
    "activity_index":   (("heavy", -16), ("elderly", -12)),
    "resting_hr":       (("hyperthyroid", 30), ("hypothyroid", -6),
                         ("heavy", 4), ("elderly", -3)),
    "hrv":              (("age_decade_over", (40, -5)), ("hyperthyroid", -12), ("heavy", -5)),
    "sleep_hours":      (("osa", -0.7), ("elderly", -0.4)),
    "spo2":             (("osa", -1.6), ("heavy", -0.6)),
    "body_temp":        (("hyperthyroid", 0.2),),
    "skin_temp":        (("hyperthyroid", 0.15),),
    "gi_symptom_score": (("drug_prefix", ("metformin", 0.8)),),
}
#: The entry shapes that are not boolean attributes of `Facts`.
BASELINE_ADJUST_FORMS = ("age_decade_over", "drug_prefix")


def _adj(name: str, base: float, f: Facts) -> float:
    """Adjust a metric's baseline by the raw case's baseline facts (`BASELINE_ADJUST`)."""
    v = base
    for kind, arg in BASELINE_ADJUST.get(name, ()):
        if kind == "age_decade_over":
            anchor, delta = arg                                   # type: ignore[misc]
            v += delta * max(0, (f.age_lo - anchor) // 10)
        elif kind == "drug_prefix":
            prefix, delta = arg                                   # type: ignore[misc]
            v += delta if f.drug.startswith(prefix) else 0.0
        else:
            # No default: a misspelled trait name must raise, not read as False.
            if getattr(f, kind):
                v += arg                                          # type: ignore[operator]
    return v


# The daily metrics, in manifest order (`registry/streams.yaml`, `render`).
METRICS: tuple[MetricSpec, ...] = tuple(MetricSpec(**f) for f in _streams.metric_fields())
METRIC_BY_NAME = {m.name: m for m in METRICS}

# True driver -> the "driver proxy" tags isomorphic to it; never injected as neutral.
DRIVER_PROXY_TAGS: dict[str, tuple[str, ...]] = {
    "poor_medication_adherence": ("adherence", "travel", "cost"),
    "medication_intolerance": ("gi", "med_side_effect"),
    "calorie_intake_change": ("diet", "appetite"),
    "activity_decline": ("activity",),
    "sleep_decline": ("sleep",),
    "concurrent_medication_effect": ("med_new", "supplement"),
    "acute_illness": ("infection", "fever"),
    "fluid_or_GI_weight_variation": ("gi", "fluid"),
    "cost_or_access_issue": ("cost",),
    "measurement_noise": ("device",),
    "insufficient_dose_exposure": ("adherence",),
    "unknown_or_multifactorial": ("supplement", "med_new"),   # raw cases often
}
# Comorbidity -> its own differential-clue tags (also excluded).
COMORBID_PROXY_TAGS: dict[str, tuple[str, ...]] = {
    "hyperthyroidism": ("cardiac", "heat", "tremor"),
    "hypothyroidism": ("cold", "fatigue"),
    "OSA": ("sleep", "respiratory", "snore"),
}


def proxy_tags(f: Facts, driver: str) -> set[str]:
    """Answer-relevant tags (driver proxies + comorbidity clues); gates stream admission."""
    out = set(DRIVER_PROXY_TAGS.get(driver, ()))
    for c in f.comorbidities:
        for key, tags in COMORBID_PROXY_TAGS.items():
            if c.startswith(key):
                out.update(tags)
    return out


# Tags that could supply a causal story for the predicted outcome. Banned for
# discrete events only: a dated event ("birthday dinner on day X") is a
# ready-made attribution, a flat step-count stream is not.
OUTCOME_EXPLAINING_TAGS: dict[str, tuple[str, ...]] = {
    "weight": ("diet", "appetite", "activity"),
    "glucose": ("diet", "appetite"),
    "dysglycemia": ("diet", "appetite"),
    "hepatic": ("diet", "supplement"),
}


def event_proxy_tags(f: Facts, driver: str) -> set[str]:
    """Forbidden tags for discrete events: `proxy_tags` plus outcome-explaining
    tags, the latter only when this task type judges that outcome."""
    out = proxy_tags(f, driver)
    # `getattr` with default True: duck-typed stand-ins for `Facts` keep the ban.
    if getattr(f, "outcome_graded", True):
        for key, tags in OUTCOME_EXPLAINING_TAGS.items():
            if key in (f.target_event or ""):
                out.update(tags)
    return out


# ============================================================ benign event pool (answer-irrelevant)
# The pool lives in `registry/benign_events.yaml`; a failed read raises (no
# built-in fallback). Optional per-item `cond`:
#   {"<axis>": {"deny": (values this patient cannot have...), "w": {value: relative rate}}}
# Every (item, openable axis) pair not covered by `cond` must be listed in
# `CONDITION_NEUTRAL_ITEMS`, or loading raises.
from .regpath import registry_path as _rp
_REGISTRY = _rp("benign_events.yaml")


def _load_event_pools() -> dict:
    import yaml as _yaml
    if not _REGISTRY.is_file():
        raise FileNotFoundError(
            f"{_REGISTRY} not found: no source for the distractor event pools (no built-in fallback).")
    from .regpath import overlaid_yaml as _overlaid_yaml
    d = _overlaid_yaml(_REGISTRY) or {}      # merged with the world-plugin overlay
    for k in ("benign_events", "life_events", "condition_neutral_items", "inherited_profile"):
        if k not in d:
            raise KeyError(f"{_REGISTRY}: missing `{k}:`")
    return d


def _as_item(d: dict) -> dict:
    """Convert YAML lists back to the tuples the code and freeze metric expect."""
    o = dict(d)
    o["tags"] = tuple(o.get("tags") or ())
    if "age_w" in o:
        o["age_w"] = tuple(float(x) for x in o["age_w"])
    if "cond" in o:
        o["cond"] = {ax: ({**spec, "deny": tuple(spec["deny"])} if "deny" in spec else dict(spec))
                     for ax, spec in (o["cond"] or {}).items()}
    return o


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
_POOLS_LOCK = _threading.RLock()
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


_NON_METRIC_NDIGITS: dict[str, int] = {}


def _declared_ndigits(name: str) -> int | None:
    """A stream's declared decimal precision from the indicator registry, or
    `None` (unregistered, or `ndigits: null` for panel items) to leave it as is."""
    from . import indicators as _ind
    try:
        nd = _ind.of(name)["ndigits"]
    except _ind.IndicatorUnregistered:
        return None
    return None if nd is None else int(nd)


def _round_to_declared_digits(ld: dict) -> None:
    """Round every stream back to its declared precision after the physiology
    layer, whose 4-decimal output would otherwise reveal which streams it touched."""
    for name, pts in list(ld.items()):
        nd = _declared_ndigits(name)
        if nd is None or not isinstance(pts, list):
            continue
        for p in pts:
            if isinstance(p, dict) and isinstance(p.get("value"), float):
                p["value"] = round(p["value"], nd) if nd else float(round(p["value"]))


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


def pool_event_topics() -> frozenset[str]:
    """Topics of the benign/life event pools: the option set for the LLM
    planning path (genuine-symptom topics are excluded)."""
    _sync_pools()
    return frozenset(str(it["topic"]) for pool in (BENIGN_EVENTS, LIFE_EVENTS)
                     for it in pool if it.get("topic"))


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


# Deterministic text -> tag inference, unioned with self-reported tags so an
# LLM-planned event cannot label itself neutral.
TAG_KEYWORDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("聚餐", "外食", "大餐", "夜宵", "加餐", "宴", "自助", "下午茶", "甜点", "零食",
      "饮酒", "喝酒", "火锅", "烧烤", "点外卖"), ("diet",)),
    (("食欲", "嘴馋", "想吃", "饥饿"), ("appetite",)),
    (("出差", "旅行", "外地", "返乡", "度假", "探亲同住"), ("travel",)),
    (("跑步", "爬山", "登山", "健身", "游泳", "骑行", "球赛", "打球", "马拉松", "徒步",
      "负重", "深蹲", "长走"), ("activity", "exertion")),
    (("熬夜", "失眠", "通宵", "倒班", "夜班", "加班到"), ("sleep",)),
    (("忘记吃", "漏服", "漏打", "漏针", "没吃药", "停用", "中断用药", "少打", "自行减量"),
     ("adherence",)),
    (("费用", "涨价", "自费", "报销", "买不起", "医保", "缺货"), ("cost",)),
    (("保健品", "补剂", "新开了", "新增药", "中药", "维生素", "蛋白粉"), ("med_new", "supplement")),
    (("发热", "发烧", "感冒", "流感", "腹泻", "呕吐", "感染", "咳嗽不止"), ("infection",)),
    (("心悸", "心慌", "怕热", "手抖", "多汗", "脖子变粗"), ("cardiac", "heat", "tremor")),
    (("怕冷", "浮肿", "便秘", "脱发"), ("cold", "fatigue")),
    (("打鼾", "鼾声", "憋醒", "嗜睡", "打呼", "血氧"), ("sleep", "respiratory", "snore")),
)


def infer_tags(text: str, context: str = "") -> tuple[set[str], bool]:
    """Infer tags and "does this require physical exertion" from the event text
    (independent of any self-reported field)."""
    blob = f"{text} {context}"
    tags: set[str] = set()
    exertion = False
    for keys, tg in TAG_KEYWORDS:
        if any(k in blob for k in keys):
            tags.update(t for t in tg if t != "exertion")
            exertion = exertion or "exertion" in tg
    return tags, exertion


def effective_tags(item: dict) -> tuple[set[str], bool]:
    """Self-reported/pool tags UNION tags inferred from text and context."""
    inf_tags, inf_ex = infer_tags(item.get("text", ""), item.get("context", ""))
    return set(item.get("tags", ())) | inf_tags, bool(item.get("exertion")) or inf_ex


def role_tags(item: dict) -> set[str]:
    """Tags for the event's role, used for the answer-relevance verdict.

    Structured tags plus tags inferred from `text`; a tag inferred only from
    `context` does not count, so an incidental word in the setting does not
    change the verdict. Plausibility (`exertion`) still uses `effective_tags`.
    """
    struct = set(item.get("tags", ()))
    from_text, _ = infer_tags(item.get("text", ""), "")
    return struct | from_text


def effective_pool_size(f: Facts, driver: str) -> dict[str, int]:
    """Number of pool entries admissible for this patient, per rate key. Both
    the feasibility check and sampling must use this, not the nominal pool size."""
    _sync_pools()
    banned = event_proxy_tags(f, driver)
    # `_event_ok` reads `f.cond_axes` itself, so feasibility and sampling agree.
    return {"symptom_rate": sum(1 for it in BENIGN_EVENTS if _event_ok(it, f, banned)[0]),
            "life_event_rate": sum(1 for it in LIFE_EVENTS if _event_ok(it, f, banned)[0])}


#: Default event-density rates (items/week); every consumer reads them from here.
EVENT_RATE_DEFAULTS: dict[str, float] = {"symptom_rate": 0.1, "life_event_rate": 0.05}


def event_weeks(T: int | float) -> float:
    """Weeks in the observation window. Lower bound 1.0 -- a window under a week
    should not compute the number of items to inject as 0."""
    return max(1.0, float(T) / 7.0)


def expected_event_counts(event_density: dict | None, T: int | float) -> dict[str, int]:
    """Declared density x observation-window weeks => item counts to inject.
    `gates.check_event_density` requires exact equality."""
    ed = event_density or {}
    w = event_weeks(T)
    return {k: max(0, int(round(float(ed.get(k, d) if ed.get(k, d) is not None else d) * w)))
            for k, d in EVENT_RATE_DEFAULTS.items()}


def _event_ok(item: dict, f: Facts, banned: set[str],
              proxy_exempt: bool = False,
              cond: tuple[str, ...] | None = None) -> tuple[bool, str]:
    """Whether a benign event is plausible for this patient and answer-irrelevant.
    Returns (ok, reason).

    `proxy_exempt=True` skips only the answer-relevant tag check (for lookalike
    distractors); plausibility is still checked. `cond=None` takes
    `Facts.cond_axes`.
    """
    # Relevance by role (`role_tags`); plausibility by wording (`effective_tags`).
    _, exertion = effective_tags(item)
    hit = role_tags(item) & banned
    if hit and not proxy_exempt:
        return False, f"answer_relevant_tag:{sorted(hit)}"
    if exertion and (f.heavy or f.elderly):
        return False, "exertion_implausible_for_profile"
    if item.get("max_age") and f.age_lo > int(item["max_age"]):
        return False, "age_implausible_for_profile"
    axes = normalize_cond_axes(cond if cond is not None else getattr(f, "cond_axes", ()))
    if axes:
        hit_c = cond_denied(item, f, axes)
        if hit_c:
            return False, f"{hit_c[0]}_implausible_for_profile:{hit_c[1]}"
    return True, ""


# ============================================================ Iron law 3: events are conditioned on the patient
# Benign/life events are drawn by a per-event age prior, optionally times
# multipliers from conditioning axes. An axis must be answer-neutral, i.e.
# drawn from case_id alone; see `CONDITION_AXIS_SPECS`.
AGE_PRIOR_BANDS: tuple[tuple[int, int, str], ...] = ((0, 39, "18-39"), (40, 64, "40-64"),
                                                     (65, 200, "65+"))


class PoolPriorMissing(ValueError):
    """An event in the pool has no declared age prior (there is no default)."""


def age_band_index(age_lo: int) -> int:
    for i, (lo, hi, _name) in enumerate(AGE_PRIOR_BANDS):
        if lo <= age_lo <= hi:
            return i
    raise ValueError(f"age {age_lo} falls in no registered age band")


def age_prior(item: dict, f: Facts) -> float:
    """This event's relative incidence rate for this patient (0 = cannot occur in
    this age band)."""
    w = item.get("age_w")
    if w is None:
        raise PoolPriorMissing(f"event {item.get('topic') or item.get('text')!r} declares no age_w")
    return float(w[age_band_index(f.age_lo)])


# ============================================================ Conditioning: axis registry
# Columns: (axis name, value domain (ordered), value source, neutrality
# evidence, status, note). Only `opt_in` axes can be turned on
# (`Facts.cond_axes`); `opt_in_restricted`, `deferred` and `forbidden` rows
# record why an axis is not usable.
CONDITION_AXIS_SPECS: tuple[tuple[str, tuple[str, ...], str, str, str, str], ...] = (
    ("season", ("spring", "summer", "autumn", "winter"),
     "season_of(f) ← rng.pick(SEASONS, case_id, 'index_date_anchor')",
     "构造性:取值函数只吃 case_id,不读 spec / latent / 诊断 / join_gold(check_axis_neutrality 机检)",
     "opt_in",
     "换季鼻炎 / 日晒 / 蚊虫 / 露天泳池 / 干燥倒刺这几类小毛病的发生率强烈随季节"),
    ("activity", ("sedentary", "active"),
     "activity_of(f) ← Facts.devices 是否含 wearable",
     "构造性:devices 由 demographics.sample_profile 采出,而它只吃 case_id",
     "opt_in",
     "有可穿戴 ⇒ 日常运动量大 ⇒ 水泡 / 肌肉酸痛 / 膝痛更常见;久坐 ⇒ 鼠标手 / 眼睑跳更常见"),
    ("stage", ("early", "mid", "late"),
     "index_time_T / course_end_day 分档",
     "C 类旋钮:两者都在 provenance.FIELD_SPECS 里标 sampled",
     "deferred",
     "取值要 index_time_T,而准入检查 `_event_ok` 只拿到 `Facts`(不含 T);"
     "启用需先给 Facts 加一个由 build 传入的字段"),
    ("drug", ("metformin", "semaglutide", "tirzepatide", "liraglutide"),
     "Facts.drug",
     "构造性同 age;但与 proxy_tags / OUTCOME_EXPLAINING_TAGS 的冲突面大",
     "opt_in_restricted",
     "药物能拉动的良性事件集中在不良反应域(胃肠道 ↔ medication_intolerance),"
     "正是驱动代理;在当前题包上 driver 恒定,负对照无法被行使。未启用"),
    ("sex", ("F", "M"),
     "Facts.sex",
     "非中性:ddx._sex_of = 题面症状里的解剖学互斥词 + CONDITION_SEX_SKEW[spec_id]",
     "forbidden",
     "性别可以进题面,但不进干扰的选择规则:后者会把题面里的弱性别信号"
     "复制成话题分布上的信号。启用需先让 sex 脱离 spec"),
    ("disease", (),
     "Facts.disease",
     "当前题包恒为单一疾病",
     "forbidden",
     "取值恒定,轴不会被行使"),
    ("comorbidity", (),
     "latent.ddx_threads / Facts.comorbidities",
     "Facts 侧恒为空;latent 侧即金标,且 spec_id 与 case 1:1",
     "forbidden",
     "按 latent 条件化 ⇒ 干扰分布成为诊断的单射函数,一张「话题组合 → 诊断」的查找表"),
)

#: Axes allowed to be turned on.
_AXIS_OPENABLE = tuple(name for name, _dom, _src, _why, st, _note in CONDITION_AXIS_SPECS
                       if st == "opt_in")
_AXIS_DOMAIN = {name: dom for name, dom, _s, _w, _st, _n in CONDITION_AXIS_SPECS}
_AXIS_STATUS = {name: st for name, _d, _s, _w, st, _n in CONDITION_AXIS_SPECS}

SEASONS = _AXIS_DOMAIN["season"]


class ConditionSpecError(ValueError):
    """Invalid conditioning declaration. Raised at load or call time; no defaults."""


def normalize_cond_axes(axes) -> tuple[str, ...]:
    """Normalize the caller-given list of axis names and check each one against
    the registry. Empty/None => `()` = unconditioned sampling."""
    if not axes:
        return ()
    out = []
    for a in axes:
        a = str(a)
        if a not in _AXIS_STATUS:
            raise ConditionSpecError(
                f"unregistered conditioning axis {a!r}; registered: {sorted(_AXIS_STATUS)}")
        if a not in _AXIS_OPENABLE:
            raise ConditionSpecError(
                f"axis {a!r} has status={_AXIS_STATUS[a]!r} and cannot be enabled; "
                f"enableable axes: {list(_AXIS_OPENABLE)} (see CONDITION_AXIS_SPECS)")
        if a not in out:
            out.append(a)
    return tuple(out)


def season_of(f: Facts) -> str:
    """This item's calendar season, drawn from case_id only."""
    from . import rng
    return rng.pick(SEASONS, f.case_id, "index_date_anchor")


def activity_of(f: Facts) -> str:
    """`active` if a wearable is in the device set (drawn from case_id only)."""
    return "active" if "wearable" in set(f.devices) else "sedentary"


#: Axis -> value function (`opt_in` axes only).
_AXIS_VALUE_FN = {"season": season_of, "activity": activity_of}

#: Per axis, the item topics explicitly declared irrelevant to that axis.
CONDITION_NEUTRAL_ITEMS: dict[str, frozenset] = {
    ax: frozenset(v) for ax, v in _POOLS["condition_neutral_items"].items()
}

#: Allowed range of the composite multiplier, the same order of magnitude as
#: `age_w`'s dynamic range, so a new axis cannot dominate the distribution.
COND_MULTIPLIER_RANGE = (0.125, 8.0)


def axis_value(axis: str, f: Facts) -> str:
    """This patient's value on this axis. An unregistered axis name / a value
    outside the value domain raises."""
    fn = _AXIS_VALUE_FN.get(axis)
    if fn is None:
        raise ConditionSpecError(f"axis {axis!r} has no value function (status={_AXIS_STATUS.get(axis)!r})")
    v = fn(f)
    if v not in _AXIS_DOMAIN[axis]:
        raise ConditionSpecError(
            f"axis {axis!r} produced out-of-domain value {v!r}; domain: {list(_AXIS_DOMAIN[axis])}")
    return v


def _axis_decl(item: dict, axis: str) -> dict | None:
    return ((item.get("cond") or {}).get(axis)) or None


def cond_denied(item: dict, f: Facts, axes: tuple[str, ...]) -> tuple[str, str] | None:
    """The conditioning admission component: returns (axis name, value) meaning
    "this event cannot occur for this patient"."""
    for axis in axes:
        dec = _axis_decl(item, axis)
        if not dec:
            continue
        v = axis_value(axis, f)
        if v in tuple(dec.get("deny") or ()):
            return axis, v
    return None


def cond_multiplier(item: dict, f: Facts, axes: tuple[str, ...]) -> float:
    """Conditioning multiplier (relative incidence after admission), default 1.0.

    Multiplicative, so a declared "rarer" can never become "impossible" (a
    weight <= 0 drops the candidate); impossibility goes through `deny`.
    """
    m = 1.0
    for axis in axes:
        dec = _axis_decl(item, axis)
        if not dec:
            continue
        w = dec.get("w")
        if not w:
            continue
        m *= float(w.get(axis_value(axis, f), 1.0))
    return m


def event_prior(item: dict, f: Facts, cond: tuple[str, ...] | None = None) -> float:
    """Age prior x conditioning multipliers; equals `age_prior` with no axes."""
    axes = normalize_cond_axes(cond if cond is not None else getattr(f, "cond_axes", ()))
    w = age_prior(item, f)
    return w * cond_multiplier(item, f, axes) if axes else w


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


#: Gold-label tokens an axis value function must not read (input-side
#: neutrality check; a statistical test at pack size would be underpowered).
AXIS_LEAK_TOKENS = ("ddx_", "spec_id", "diagnosis", "join_gold", "aliases",
                    "adjudication", "outcome_label", "gold_", "_sex_of", "SEX_SKEW")


def axis_neutrality_problems(fn) -> list[str]:
    """Input-side neutrality: a gold-label token in the value function's code => failing."""
    import ast
    import inspect
    import textwrap
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as e:                    # source unavailable => judged failing, not exempted
        return [f"{getattr(fn, '__name__', fn)!r}: 取不到源码({e.__class__.__name__}) —— "
                f"取不到就不能声称中性"]
    # Scan code only (docstrings stripped); on parse failure scan the raw source.
    try:
        tree = ast.parse(textwrap.dedent(src))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
        scanned = ast.unparse(ast.fix_missing_locations(tree))
    except (SyntaxError, ValueError):
        scanned = src
    return [f"{getattr(fn, '__name__', fn)!r}: 源码含金标记号 {t!r}"
            for t in AXIS_LEAK_TOKENS if t in scanned]


def check_axis_neutrality() -> list[str]:
    """Load-time: every openable axis's value function must pass the neutrality check."""
    bad: list[str] = []
    for axis in _AXIS_OPENABLE:
        fn = _AXIS_VALUE_FN.get(axis)
        if fn is None:
            bad.append(f"轴 {axis!r} status=opt_in 却没有取值函数")
            continue
        bad += [f"轴 {axis!r} 未过 L1:{m}" for m in axis_neutrality_problems(fn)]
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


def weighted_order(pool: list[dict], f: Facts, kind: str,
                   cond: tuple[str, ...] | None = None) -> list[int]:
    """Deterministic weighted sampling without replacement by prior; returns the
    draw order of pool indices.

    Efraimidis-Spirakis keys `-ln(u_i) / w_i`: each key depends only on its own
    topic, so adding an event does not move existing keys. `w_i == 0` never draws.
    """
    import math
    from . import rng
    keys = []
    for i, it in enumerate(pool):
        w = event_prior(it, f, cond)
        if w <= 0:
            continue
        u = rng.unit(f.case_id, kind, "sel", it.get("topic") or it.get("text", ""))
        u = min(max(u, 1e-12), 1 - 1e-12)
        keys.append((-math.log(u) / w, i))
    return [i for _k, i in sorted(keys)]


_PRIOR_PROBLEMS = check_pool_priors()
if _PRIOR_PROBLEMS:
    raise PoolPriorMissing("; ".join(_PRIOR_PROBLEMS))

_COND_PROBLEMS = check_condition_specs() + check_axis_neutrality()
if _COND_PROBLEMS:
    raise ConditionSpecError("; ".join(_COND_PROBLEMS))


# ============================================================ Daily-metric stream planning/rendering
@dataclass
class StreamPlan:
    spec: MetricSpec
    base_eff: float                  # actual baseline derived from profile (+ the raw case's baseline vitals)
    step_days: int                   # sampling step (derived from event_density.measure_per_week)
    source: str                      # baseline source: profile | raw_case_vital
    phase: int = 0


def _gold_signal_of_driver(driver: str) -> str | None:
    """The gold-evidence stream for this driver, from `build.GOLD_EVIDENCE`
    (same source as `verify.GOLD_SIGNAL_OF`); imported lazily to avoid a cycle."""
    try:
        from .build import GOLD_EVIDENCE
    except Exception:                                  # noqa: BLE001
        return None
    g = (GOLD_EVIDENCE or {}).get(driver) or {}
    return g.get("signal")


def _wearable_cadence(f: Facts, m: MetricSpec, step: int) -> tuple[int, str | None]:
    """Sampling step for a planned stream, or the reason it is not planned.

    A calibrated wearable stream is sampled daily (a property of the device,
    not the follow-up plan). Whether a device reports a metric is drawn from
    case_id only.
    """
    if not _wear.is_calibrated(m.name):
        return step, None
    if not _wear.stream_available(str(getattr(f, "case_id", "") or ""), m.name):
        return 0, "device_does_not_report_metric"
    return 1, None


def world_layer_base_signals(raw, world_signals=()) -> set[str]:
    """Names in `raw.longitudinal_data` that `plan_streams` must leave alone.

    Everything upstream wrote, except a calibrated stream that the kernel's
    `inject_distractors` wrote as a distractor and that is not gold evidence:
    that copy is replaced by the calibrated render.
    """
    upstream = set(getattr(raw, "longitudinal_data", None) or {})
    adj = getattr(raw, "adjudication", None) or {}
    distractor = set(adj.get("distractor_signals") or ()) if isinstance(adj, dict) else set()
    world = set(world_signals or ())
    free = {n for n in distractor & upstream if _wear.is_calibrated(n) and n not in world}
    return upstream - free


def plan_streams(f: Facts, event_density: dict, driver: str,
                 drop: set[str] | None = None,
                 base_signals=None) -> tuple[list[StreamPlan], list[dict]]:
    """Select the daily metrics to inject: device in the kit, not a driver
    proxy, not already provided by the world layer (`base_signals`, which
    holds gold evidence the injector must not touch)."""
    drop = drop or set()
    banned = proxy_tags(f, driver)
    mpw = float(event_density.get("measure_per_week", 7) or 7)
    step = max(1, int(round(7.0 / max(0.5, mpw))))
    plans, skipped = [], []
    for i, m in enumerate(METRICS):
        if m.name in _wear.DERIVED_BINDINGS:
            # Derived streams are produced from their parents after rendering.
            continue
        if m.name in drop:
            skipped.append({"item": m.name, "reason": "dropped_by_verifier"})
            continue
        if m.name in (base_signals or ()):
            # Gold evidence from the world layer; do not overwrite.
            skipped.append({"item": m.name, "reason": "provided_by_world_layer"})
            continue
        if m.devices and not (set(m.devices) & set(f.devices)):
            skipped.append({"item": m.name, "reason": f"no_device{list(m.devices)}"})
            continue
        if set(m.tags) & banned:
            skipped.append({"item": m.name, "reason": "driver_or_comorbid_proxy"})
            continue
        base = _adj(m.name, m.base, f)
        src = "profile"
        vk = m.vital_key
        if vk and f.baseline_vitals.get(vk) is not None:   # the raw case wrote an actual level for this vital -> use it
            base, src = float(f.baseline_vitals[vk]), "raw_case_vital"
        lo, hi = m.hard_range
        base = min(max(base, lo + m.amp + 1e-9), hi - m.amp - 1e-9)
        step_eff, why = _wearable_cadence(f, m, step)
        if why:
            skipped.append({"item": m.name, "reason": why})
            continue
        plans.append(StreamPlan(spec=m, base_eff=round(base, m.ndigits or 0) if m.ndigits else round(base),
                                step_days=step_eff, source=src, phase=(i * 3) % 7))
    return plans, skipped


#: AR(1) day-scale coefficient. ACF(k) = phi^k, so phi^7 < 0.35 requires
#: phi < 0.862; 0.80 gives ACF(1)=0.80, ACF(7)=0.21.
_AR_PHI = 0.80
#: Stationary sd = `_AR_SD_FRAC x amp`, kept small because AR(1) has
#: Gaussian-like tails and `base_eff` only reserves `amp` headroom.
_AR_SD_FRAC = 0.35


def _det_shock(seed: str, k: int) -> float:
    """A deterministic [-1, 1) uniform shock from `blake2b(seed|k)`; no RNG
    state, so results do not depend on call order."""
    h = hashlib.blake2b(f"{seed}|{k}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 63) - 1.0


def _render_calibrated(p: "StreamPlan", end_day: int, seed: str,
                       stats: dict | None) -> list[dict]:
    """Render a stream calibrated in `haenv.wearable`: empirical distribution
    family, lag-1 persistence and spread, plus a two-state non-wear mask."""
    m = p.spec
    days = list(range(0, end_day + 1, max(1, int(p.step_days))))
    series = _wear.ar1(m.name, float(p.base_eff), days, seed, m.hard_range, stats)
    if series is None:                                   # not calibrated after all
        return []
    mask = _wear.worn_on(seed, m.name, days)
    lo, hi = m.hard_range
    out, n_clip = [], 0
    for d, v, worn in zip(days, series, mask):
        if not worn:
            continue
        if v < lo or v > hi:
            n_clip += 1
        v = min(max(v, lo), hi)
        out.append({"ts": d, "value": round(v, m.ndigits) if m.ndigits else int(round(v))})
    if stats is not None:
        stats["n"] = stats.get("n", 0) + len(out)
        stats["n_clipped"] = stats.get("n_clipped", 0) + n_clip
    return out


def render_stream(p: StreamPlan, end_day: int, seed: str = "",
                  stats: dict | None = None) -> list[dict]:
    """A deterministic AR(1) waveform over [0, end_day]; never reads the outcome.

    AR(1) rather than fixed sines, which would give the corpus a lag-7
    fingerprint. `seed` must carry the case identity (phases vary only by
    stream index). When `stats` is given, `{"n", "n_clipped"}` are recorded.
    """
    m, out = p.spec, []
    if _wear.is_calibrated(m.name):
        return _render_calibrated(p, end_day, seed, stats)
    sd = _AR_SD_FRAC * m.amp
    phi_s = _AR_PHI ** max(1, int(p.step_days))          # scaled by the sampling step
    sig_s = sd * math.sqrt(max(0.0, 1.0 - phi_s * phi_s))
    bound = sig_s * math.sqrt(3.0)                       # uniform distribution: sd = bound/sqrt(3)
    key = f"{seed}|{m.name}|{p.phase}"
    # Start from the stationary distribution (no visible warm-up).
    x = sd * _det_shock(key, -1)
    n_clip = 0
    for k, d in enumerate(range(0, end_day + 1, p.step_days)):
        if k:
            x = phi_s * x + bound * _det_shock(key, k)
        v = p.base_eff + x
        if m.name == "scale_qc_flag":                     # binary QC: one calibration failure every 17 days
            # Binary QC flag: a fixed comb (period 17, phase per patient),
            # replacing the AR(1) value. A random draw would exceed
            # `gates.A5_MIN_REL_SPAN`, which is calibrated against this comb.
            t = d + p.phase
            v = 0.0 if (t % m.period == 0 and d > 0) else 1.0
        lo, hi = m.hard_range
        if v < lo or v > hi:
            n_clip += 1
        v = min(max(v, lo), hi)
        out.append({"ts": d, "value": round(v, m.ndigits) if m.ndigits else int(round(v))})
    if stats is not None:
        stats["n"] = stats.get("n", 0) + len(out)
        stats["n_clipped"] = stats.get("n_clipped", 0) + n_clip
    return out


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


# Morphological suffixes allowed after a Latin alias stem on the scoring side
# only (`hypothyroid` -> `Hypothyroidism`). A closed set: `sle` must not match
# `sleep`, `insulin` must not match `insulinoma`.
_LATIN_SUFFIXES = ("ism", "osis", "oses", "ic", "ical", "ia", "emia", "aemia",
                   "y", "ies", "s", "es", "al", "ous", "otic", "oidism")


def alias_hit(text: str, aliases, allow_suffix: bool = False) -> list[str]:
    """Diagnosis aliases found in the text.

    Latin aliases match on word boundaries (underscore counts as a word
    character); Chinese aliases match by substring. `allow_suffix=True`
    (scoring only) also accepts a suffix from `_LATIN_SUFFIXES`; the leak
    scanner keeps it off to err toward false positives.
    """
    import re
    out = []
    for a in (aliases or []):
        a = str(a).strip()
        if not a:
            continue
        if a.isascii():
            tail = (rf"(?:{'|'.join(_LATIN_SUFFIXES)})?" if allow_suffix else "")
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(a)}{tail}(?![A-Za-z0-9_])",
                         text, re.I):
                out.append(a)
        elif a in text:
            out.append(a)
    return out


# Parentheticals that mark a disease thread ("(diabetes thread)") are author
# annotations and would reveal join_gold; they are scrubbed when they contain
# the case's diagnosis alias or one of these markers. Clinical content in
# parentheses is kept.
ANNOTATION_MARKERS = ("线程",)


def scrub_annotation(text: str, aliases) -> tuple[str, list[str]]:
    """Scrub parentheticals naming this case's diagnosis or marking a thread.
    Returns (scrubbed text, removed parentheticals)."""
    import re
    removed: list[str] = []

    def _sub(m):
        inner = m.group(2)
        if alias_hit(inner, aliases) or any(k in inner for k in ANNOTATION_MARKERS):
            removed.append(m.group(0))
            return ""
        return m.group(0)

    out = re.sub(r"([((])([^))]*)([))])",
                 lambda m: _sub(re.match(r"([((])([^))]*)([))])", m.group(0))), text)
    return out.strip(" 、,,+"), removed


@_wp.overlay_cached("symptom_topics.yaml")
def _load_symptom_topics() -> dict[str, str]:
    from .regpath import load_registry as _lr
    _d = _lr("symptom_topics.yaml") or {}
    return {_norm_symptom(k): str(v["topic"])
            for k, v in (_d.get("symptoms") or {}).items()
            if isinstance(v, dict) and v.get("topic")}


def symptom_topic(text: str) -> str | None:
    """Genuine symptom text -> physiology `topic` from the hand-written
    `registry/symptom_topics.yaml` (plus world-plugin registrations), or `None` if not
    registered.

    Exact match on the normalized full text; similar prefixes can mean
    different things clinically.
    """
    return _load_symptom_topics().get(_norm_symptom(text))


def _norm_symptom(text: str) -> str:
    """Normalize whitespace and full-width punctuation (used for registry and lookup)."""
    t = str(text or "").strip()
    for a, b in (("（", "("), ("）", ")"), ("，", ","), ("、", ","), ("：", ":")):
        t = t.replace(a, b)
    return "".join(t.split())


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


def plan_benign_evs(f: Facts, event_density: dict, driver: str, T: int,
                    drop: set[str] | None = None, n_inherited: int = 0,
                    used_topics: set[str] | None = None,
                    aliases: list | None = None,
                    cond: tuple[str, ...] | None = None) -> tuple[list[dict], list[dict]]:
    """Benign-event and life-event EVs spread over [7, T] at `event_density`.

    `n_inherited`: validated upstream benign events, subtracted from the target
    count. `used_topics`: topics already used upstream (deduplicated).
    `cond=None` takes `Facts.cond_axes`.
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
    used_days: set[int] = set()          # benign/life events must not share a day with each other either
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
    """Let the LLM choose streams, baselines and benign events; the numeric
    series is still rendered deterministically (see `render_stream`), and every
    choice is checked by `verify.py`."""
    from solver import _extract_json                     # kernel: JSON extraction tolerant of banners/fences

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

    mpw = float(event_density.get("measure_per_week", 7) or 7)
    step = max(1, int(round(7.0 / max(0.5, mpw))))
    plans, skipped = [], []
    for i, item in enumerate(data.get("streams") or []):
        name = str(item.get("name", ""))
        m = METRIC_BY_NAME.get(name)
        if m is None:
            skipped.append({"item": name or f"stream#{i}", "reason": "llm_unknown_metric"})
            continue
        if name in drop:
            skipped.append({"item": name, "reason": "dropped_by_verifier"})
            continue
        if any(p.spec.name == name for p in plans):
            skipped.append({"item": name, "reason": "llm_duplicate_stream"})
            continue
        try:
            base = float(item["base"])
        except (KeyError, TypeError, ValueError):
            skipped.append({"item": name, "reason": "llm_bad_base"})
            continue
        if name in _wear.DERIVED_BINDINGS:
            skipped.append({"item": name, "reason": "derived_from_parent"})
            continue
        step_eff, why = _wearable_cadence(f, m, step)
        if why:
            skipped.append({"item": name, "reason": why})
            continue
        lo, hi = m.hard_range                            # only a fallback clamp for "physically impossible," not correcting the model's judgment
        base = min(max(base, lo + m.amp), hi - m.amp)
        plans.append(StreamPlan(spec=m, base_eff=round(base, m.ndigits) if m.ndigits else round(base),
                                step_days=step_eff, source="llm", phase=(i * 3) % 7))

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
    from .build import GOLD_EVIDENCE            # deferred import: build depends on this module, a top-level import would cycle
    world_signals = {g["signal"] for g in GOLD_EVIDENCE.values()}
    ed = dict(premise.event_density or {})
    T = int(raw.prediction_context["prediction_time_T"])
    win = str(raw.prediction_context.get("prediction_window", "281d")).rstrip("d")
    end_day = T + (int(win) if win.isdigit() else 281)

    llm_evs: list[dict] = []
    if dispatch is not None:                     # LLM planning (values are still rendered deterministically by render_stream)
        n_inh = sum(1 for e in raw.evidence_ledger
                    if str(e.get("evidence_id", "")).rsplit("-", 1)[-1].startswith("D")
                    and str(e.get("evidence_id", "")) not in (drop or set()))
        plans, llm_evs, skipped = plan_via_llm(f, ed, driver, T, dispatch, drop,
                                               feedback=feedback, n_inherited=n_inh)
    else:
        plans, skipped = plan_streams(f, ed, driver, drop,
                                      base_signals=world_layer_base_signals(raw, world_signals))
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
    inh_evs, ev_rows = [], []
    kept_ledger = []
    for e in raw.evidence_ledger:
        eid = str(e.get("evidence_id", ""))
        if eid.rsplit("-", 1)[-1].startswith("D"):
            if eid in drop:
                skipped.append({"item": eid, "reason": "dropped_by_verifier"})
                continue
            inh_evs.append(e)
        kept_ledger.append(e)
    raw.evidence_ledger = kept_ledger

    # The case's aliases plus the registry's `leak_only` words (as `verify._ddx_aliases`).
    from .overlay import leak_aliases_for
    _ddx = getattr(cs, 'ddx', None) or {}
    aliases = leak_aliases_for(_ddx.get('spec_id'), _ddx.get('aliases'))
    real_evs, sk1 = plan_real_symptom_evs(f, T, drop, aliases=aliases)
    if dispatch is not None:
        benign_evs, sk2 = llm_evs, []
    else:
        used_topics = {t for e in inh_evs
                       if (t := (inherited_profile(e.get("symptom", "")) or {}).get("topic"))}
        benign_evs, sk2 = plan_benign_evs(f, ed, driver, T, drop, n_inherited=len(inh_evs),
                                          used_topics=used_topics, aliases=aliases)
    skipped += [x for x in sk1 + sk2 if isinstance(x, dict) and "reason" in x]

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
                        "source": "llm" if dispatch is not None else "pool",
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

# ---------------------------------------------------------------- Findings-layer hook
FINDINGS_ENABLED: list = [False]      # set by run/build from job.findings; off by default

#: Physiology-layer switch (`job.yaml: physio`), off by default. When on,
#: rendered streams go through `haenv.physio.apply_physio`.
PHYSIO_ENABLED: list = [False]


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
