"""Stateless parts of the event layer: the patient's baseline facts, the conditioning axes and
their checks, answer-relevance tags, and event pacing. The pools themselves, which plugin
registration rebinds, stay in `events.py` with every reader.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import threading as _threading
from dataclasses import dataclass
from .regpath import registry_path as _rp
from .world_knobs import (  # noqa: F401
    PHYSIO_ENABLED,
)
from .indicators import (  # noqa: F401
    _declared_ndigits,
)


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

    @property
    def female(self) -> bool:
        return str(self.sex).upper().startswith("F")


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


_POOLS_LOCK = _threading.RLock()


_NON_METRIC_NDIGITS: dict[str, int] = {}


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


