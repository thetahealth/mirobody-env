"""provenance.py -- where every field of a case came from, stamped field by field.

A job case mixes three classes of field that look alike in yaml:

| Class | Example | Produced by |
|---|---|---|
| patient fact | `age_range` `drug` `symptoms` | extraction from the case text |
| gold | `ddx_diagnosis` `ddx_urgency` | `source_text`, `clinician`, `derived`, `llm`, or `absent` |
| generation knob | `index_time_T` `event_density` | sampling, never extraction |

`ALLOWED_BY_CLASS` enforces which sources each class may carry. Reports count gold
by source, so the number of LLM-adjudicated gold values stays visible. A missing
stamp raises; it never falls back to `absent`.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from .external_gold import (  # noqa: F401
    FieldClass,
)

#: Field provenance.
Provenance = Literal["source_text", "clinician", "derived", "llm", "sampled", "absent"]

#: `derived`: fixed by rule from other gold fields (`wq.py` pins
#: urgency == RED <=> red_flag_present and urgency == GREEN <=> not
#: clinician_action_warranted). Derivable fields are never asked of a model.
PROVENANCES: tuple[Provenance, ...] = ("source_text", "clinician", "derived",
                                       "llm", "sampled", "absent")

#: Sources each field class may carry. `sampled` is for generation knobs only.
ALLOWED_BY_CLASS: dict[str, frozenset[str]] = {
    "patient_fact": frozenset({"source_text", "absent"}),
    "gold": frozenset({"source_text", "clinician", "derived", "llm", "absent"}),
    "knob": frozenset({"sampled"}),
}


#: Confidence order used when a field is stamped by two paths (the stronger
#: stamp is kept; equal strength keeps the first).
STRENGTH: dict[str, int] = {"source_text": 3, "clinician": 3, "derived": 3,
                            "llm": 1, "sampled": 2, "absent": 0}


# Field classification table.

#: `(key, class, description)`, keyed like `job.CaseSpec.raw` /
#: `job.LATENT_REGISTRY`, which it must cover (checked at import).
FIELD_SPECS: tuple[tuple[str, FieldClass, str], ...] = (
    # ---- A patient facts: present in the text, extracted ----
    ("disease", "patient_fact", "原发病;必须落在内核 DISEASE_SIGNAL_DOMAIN 里"),
    ("age_range", "patient_fact", "年龄段"),
    ("sex", "patient_fact", "性别"),
    ("bmi", "patient_fact", "BMI(可选)"),
    ("comorbidities", "patient_fact", "共病(可选)"),
    ("drug", "patient_fact", "药物;决定 PK/PD 梯级"),
    ("dose_steps", "patient_fact", "剂量梯级;必须落在该药的 ladder 上"),
    ("devices", "patient_fact", "设备清单;划定噪声注入的物理攻击面"),
    ("start_weight", "patient_fact", "起始体重"),
    ("nadir_weight", "patient_fact", "最低点体重"),
    ("baseline_vitals", "patient_fact", "基线体征(可选)"),
    ("symptoms", "patient_fact", "主诉列表 [{day,text,context}] —— 文本抽取的主体"),
    ("sampling_days", "patient_fact", "采样间隔(可选)"),

    # ---- B gold adjudication: stamped field by field ----
    ("ddx_spec_id", "gold", "内核 spec key;**绝不能当 case_id**(那等于把答案印在题号上)"),
    ("ddx_diagnosis", "gold", "dx_hit 的判据"),
    ("ddx_aliases", "gold", "dx_hit 的文本匹配句柄"),
    ("ddx_join_gold", "gold", "unified|comorbidity|independent —— 文本里**一般没有**"),
    ("ddx_threads", "gold", "comorbidity 的病线声明"),
    ("ddx_tests", "gold", "应索取的检查(动作侧金标)"),
    ("ddx_specialty", "gold", "应转诊科室(动作侧金标)"),
    ("ddx_urgency", "gold", "🟢🟡🟠 分级 —— 文本里**一般没有**"),
    ("ddx_red_flag", "gold", "武装 missed_emergency_red_flag 硬门 —— 文本里**一般没有**"),
    ("ddx_clinician_warranted", "gold", "是否需要临床介入"),
    ("ddx_outcome_label", "gold", "结局标签;优先于按体重方向推断"),
    ("ddx_insufficient", "gold", "True = 信息不足档,正解是声明不足"),

    # ---- C generation knobs: not in the text, sampled ----
    # `outcome` / `driver` are sampled knobs; their only source is
    # `job.outcome_of` / `job.driver_of`.
    ("outcome", "knob", "regain/maintain 决定 event_occurred;按 case_id 从 job.OUTCOMES 抽"),
    ("driver", "knob", "真驱动;由 outcome 经 job.DRIVER_BY_OUTCOME 派生(生成侧只 realize 得了这两条)"),
    ("index_time_T", "knob", "开窗时点 —— **决定看得见多少**,不是病人的属性"),
    ("course_end_day", "knob", "病程长度"),
    ("nadir_day", "knob", "最低点落在第几天;由 course_end_day 派生,缺省 None ⇒ 沿用 index_time_T"),
    ("event_density", "knob", "事件密度;决定注入几条 —— 让 LLM 编这个数是本模块头号要防的事"),
    ("noise", "knob", "注入的伪影计划"),
    ("distractor_level", "knob", "干扰强度"),
    ("distractor_cond_axes", "knob", "干扰按病人条件化开哪几根轴 —— 文本里当然没有,是我们拧的"),
    ("rhythm_gap", "knob", "强制拉长某个就诊间隔(造信息缺口)"),
    ("long_horizon_tier", "knob",
     "长视野档的逐例标记(≈1 年视野)。不能靠 `index_time_T == 336` 反推 —— "
     "自然抽到 336 的那批资格不受限,混进来对照就不匹配了"),
    ("composition_v2", "knob", "组成 v2 标记:症状外显率与同义措辞已在 job 生成期落到 raw.symptoms;文本里没有,是我们拧的"),
    ("missingness", "knob", "缺失机制 MCAR/MAR/MNAR"),
    ("adherence_low", "knob", "依从性低谷"),
    ("drug_response", "knob", "个体药效响应系数"),
    ("reversal_week", "knob", "反转点周次"),
    ("regain_slope", "knob", "回升斜率"),
    ("regain_end_kg", "knob", "回升终点(kg);给了就反解斜率,缺省沿用 regain_slope"),
    ("target_event", "knob", "目标事件类型"),
    ("difficulty", "knob", "只进 meta 作记账,不参与任何判定"),
)

CLASS_OF: dict[str, FieldClass] = {k: c for k, c, _ in FIELD_SPECS}


def class_of(field: str) -> FieldClass | None:
    """Class of `field`: `FIELD_SPECS` first, then the class an external task type declared
    for its own latent key (`external_gold.register_gold_block`)."""
    c = CLASS_OF.get(field)
    if c is not None:
        return c
    from .external_gold import LATENT_CLASSES
    return LATENT_CLASSES.get(field)

PATIENT_FACTS: tuple[str, ...] = tuple(k for k, c, _ in FIELD_SPECS if c == "patient_fact")
GOLD_FIELDS: tuple[str, ...] = tuple(k for k, c, _ in FIELD_SPECS if c == "gold")
KNOBS: tuple[str, ...] = tuple(k for k, c, _ in FIELD_SPECS if c == "knob")

#: Gold fields that almost never appear in prospective text (an intake
#: complaint plus baseline); the extractor stamps them `absent`. Give them a
#: value through `clinician` or an explicit `llm` stamp.
RARELY_IN_TEXT: frozenset[str] = frozenset({
    "ddx_join_gold", "ddx_threads", "ddx_urgency", "ddx_red_flag",
    "ddx_clinician_warranted", "ddx_outcome_label", "ddx_insufficient",
    "ddx_aliases", "ddx_spec_id",
})

#: Input kind, declared by the caller (the extractor cannot tell).
InputKind = Literal["prospective", "retrospective"]

#: Gold fields that retrospective case reports state verbatim in a form the
#: scorer consumes (established on real reports). `driver` is excluded: mapping
#: narrative text onto the controlled driver vocabulary is a judgment.
RETRO_TRANSCRIBABLE: frozenset[str] = frozenset({"ddx_outcome_label"})


def rarely_in_text(input_kind: InputKind = "prospective") -> frozenset[str]:
    """Gold fields always stamped `absent` for this input kind. Defaults to the
    stricter `prospective`.
    """
    if input_kind == "retrospective":
        return RARELY_IN_TEXT - RETRO_TRANSCRIBABLE
    if input_kind != "prospective":
        raise ProvenanceError(
            f"unknown input_kind {input_kind!r}; only {['prospective', 'retrospective']}"
            f"; there is no default")
    return RARELY_IN_TEXT


class ProvenanceError(ValueError):
    """A provenance is invalid or missing. No fallback; raises directly."""


@dataclass(frozen=True)
class Stamp:
    """The provenance stamp for one field. `by` records who: model name,
    annotator id, document id or sampling seed; required except for `absent`.
    """
    field: str
    provenance: Provenance
    by: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCES:
            raise ProvenanceError(
                f"{self.field}: unknown provenance {self.provenance!r}; "
                f"only {list(PROVENANCES)}")
        if self.provenance != "absent" and not self.by:
            raise ProvenanceError(
                f"{self.field}: provenance {self.provenance} must record \"who\" "
                f"(llm→model name · clinician→annotator id · source_text→document id)")
        if class_of(self.field) is None:
            raise ProvenanceError(
                f"{self.field}: not in FIELD_SPECS -- an unclassified field may not be stamped, "
                f"or it will silently fall into the \"LLM extraction\" path")
        _cls = class_of(self.field)
        _ok = ALLOWED_BY_CLASS[_cls]
        if self.provenance not in _ok:
            _how = {"patient_fact": "patient facts go through text extraction (source_text)",
                    "gold": "gold goes through one of source_text/clinician/llm/absent",
                    "knob": "generation knobs go through deterministic sampling (sampled) -- the text never has these numbers at all"}[_cls]
            raise ProvenanceError(
                f"{self.field} is {_cls}; provenance {self.provenance!r} is not allowed; "
                f"allowed: {sorted(_ok)} -- {_how}")


@dataclass
class ProvenanceLedger:
    """All stamps for one case, field by field."""
    case_id: str
    stamps: dict[str, Stamp] = field(default_factory=dict)

    def stamp(self, f: str, provenance: Provenance, by: str | None = None,
              note: str | None = None) -> Stamp:
        s = Stamp(f, provenance, by, note)
        old = self.stamps.get(f)
        # keep the stronger stamp; note what it overrode
        if old is not None and STRENGTH[old.provenance] > STRENGTH[provenance]:
            return old
        if old is not None and old.provenance != provenance:
            s = Stamp(f, provenance, by,
                      f"{note or ''}(覆盖了更弱的 {old.provenance}/{old.by})".strip())
        self.stamps[f] = s
        return s

    def of(self, f: str) -> Provenance:
        """This field's provenance; raises if never stamped."""
        s = self.stamps.get(f)
        if s is None:
            raise ProvenanceError(
                f"{self.case_id}.{f}: has no provenance record. "
                f"\"Not recorded\" does not mean \"recorded as having none\" -- for the latter, "
                f"stamp('absent') explicitly")
        return s.provenance

    def gold_absent(self) -> list[str]:
        """Gold fields that were dropped. No judge may score anything on
        these fields."""
        return sorted(f for f in GOLD_FIELDS
                      if self.stamps.get(f) and self.stamps[f].provenance == "absent")

    def gold_by(self, provenance: Provenance) -> list[str]:
        return sorted(f for f in GOLD_FIELDS
                      if self.stamps.get(f) and self.stamps[f].provenance == provenance)

    def summary(self) -> dict:
        """Summary for batch.json and reports; `llm` has its own column."""
        counts = {p: len(self.gold_by(p)) for p in PROVENANCES}
        untracked = [f for f in GOLD_FIELDS if f not in self.stamps]
        return {
            "case_id": self.case_id,
            "gold_provenance_counts": counts,
            "gold_llm_fields": self.gold_by("llm"),
            "gold_absent_fields": self.gold_absent(),
            # The note tells "not in the source" apart from "did not match the
            # citation"; truncated so source snippets do not leak into job.yaml.
            "gold_absent_reasons": {f: (self.stamps[f].note or "")[:70]
                                    for f in self.gold_absent()},
            "gold_untracked_fields": untracked,
            "scorable": counts["absent"] == 0 and not untracked,
        }


def check_covers_registry(registry_keys) -> list[str]:
    """Keys of `LATENT_REGISTRY` with no provenance class (built-in or external)."""
    return sorted(k for k in registry_keys if class_of(k) is None)


def check_no_overlap() -> list[str]:
    """A key may belong to only one class. Returns keys declared more than once."""
    seen, dup = set(), []
    for k, _, _ in FIELD_SPECS:
        if k in seen:
            dup.append(k)
        seen.add(k)
    return sorted(dup)


# --------------------------------------------------------------------------
# Import-time self-check: an unclassified latent key would fall into the
# extraction path silently.
def _selfcheck_on_import() -> None:
    _dup = check_no_overlap()
    if _dup:
        raise ProvenanceError(f"FIELD_SPECS has keys classified more than once: {_dup} -- the boundary would stop holding")
    try:
        from .job_schema import LATENT_REGISTRY
    except Exception:                                          # noqa: BLE001
        return                                     # job not importable yet
    _miss = check_covers_registry(LATENT_REGISTRY)
    if _miss:
        raise ProvenanceError(
            f"these latent-variable keys are unclassified: {_miss} -- an unclassified key silently "
            f"falls into the \"LLM extraction\" path, and it might be a generation knob. "
            f"Give it a class in FIELD_SPECS.")


_selfcheck_on_import()
