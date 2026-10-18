"""The declarative part of a job: task types, outcome domain, latent key registry, knob domains.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations


TASK_TYPES = ("early_warning", "tracking_review", "joint_dx", "hardprob")


OUTCOMES = ("regain", "maintain")


DISTRACTOR_LEVELS = ("none", "low", "high")


# ================================================ latent key registry
#
# Every latent key maps to the gold or premise field that consumes it
# (`EXEMPT` = never enters the gold; None = registered but unread, warns).
# An unregistered key raises.
EXEMPT = "EXEMPT"


LATENT_REGISTRY: dict[str, tuple[str | None, str]] = {
    "outcome":          ("build.HaenvGenerator -> raw.outcome_label", "regain/maintain 决定 event_occurred"),
    "driver":           ("build.HaenvGenerator -> raw.gold_drivers + adjudication.primary_driver", ""),
    "reversal_week":    ("build.HaenvGenerator -> raw.reversal_points[].week", "Track E 迟滞的判据来源"),
    "regain_slope":     ("build._weight_series", "回升斜率"),
    # Regain endpoint in kg; when given, it back-solves the slope so the total
    # gain does not depend on the sampled course length.
    "regain_end_kg":    ("build.HaenvGenerator -> build._weight_series(slope 反解)",
                         "回升终点(kg)。由 (end − nadir) / 回升周数 反解斜率;"
                         "缺省 None ⇒ 沿用 `regain_slope`"),
    "index_time_T":     ("build.build_case(T) -> prediction_context.prediction_time_T", ""),
    "course_end_day":   ("build.course_end_of -> prediction_context.prediction_window "
                         "+ dose/adherence/临床流的地平线",
                         "逐例病程长度;缺省时按 case_id 从 COURSE_END_DOMAIN 采样"),
    # Day of the nadir (default: the index time). Diagnosis cases set it so the
    # weight trend can span months; ignored on cases with a reversal.
    "nadir_day":        ("build._weight_series(desc_end)",
                         "最低点落在第几天;缺省 None ⇒ 沿用 `index_time_T`"),
    "distractor_level": ("build.build_case -> noise.inject_distractors", ""),
    "distractor_cond_axes": ("events.facts_of -> Facts.cond_axes -> "
                             "events._event_ok(准入) + events.event_prior(乘子)",
                             "开哪几根条件轴;取值域见 events.CONDITION_AXIS_SPECS,"
                             "未登记的轴名当场抛(default-deny)"),
    "noise":            ("build.build_case -> noise.inject", ""),
    "event_density":    ("build.premise_spec -> premise.event_density", "events.py 据此定条数与步长"),
    "adherence_low":    ("build.premise_spec -> premise.adherence.trajectory", ""),
    "drug_response":    ("build.premise_spec -> meta.drug_response -> drug_effects.response_for",
                         "个体药效响应系数(HbA1c 与空腹血糖共用);缺省按 case_id 在驱动所属区间内抽,"
                         "显式值必须落在驱动的区间里(不响应 vs 其余),否则载入即抛"),
    "missingness":      ("build.premise_spec -> premise.adherence.missingness_mechanism", ""),
    "target_event":     ("build.premise_spec -> meta.target_event_type", ""),
    "difficulty":       (EXEMPT, "只进 meta.difficulty_class 作记账,不参与任何判定"),
    "rhythm_gap":       ("build.premise_spec -> meta.rhythm_gap -> evaluate.slices_for",
                         "只在 `slices: real-rhythm+gap` 下生效:把该例的某个就诊间隔"
                         "强制拉长到 REAL_RHYTHM_GAP_DAYS(造信息缺口)。"
                         "真实门诊 8.9% 的间隔 >180 天,合成语料默认不含这类长间隔"),
    # Long-horizon tier marker for stratified reading (not a world parameter).
    # Composition-v2 marker: the case was drawn with symptom penetrance and synonym wording applied.
    "composition_v2":   (EXEMPT, "job 生成期已把外显率与同义变体落到 raw.symptoms 里;此标记只供分层读数与审计,不进任何判定"),
    "long_horizon_tier": ("ddx.long_horizon_for -> latent -> 分层读数(报告/分析侧)",
                          "长视野档(≈1 年视野)的逐例标记。缺口档在它内部抽 ⇒ "
                          "`long_horizon_tier ∧ ¬rhythm_gap` 就是缺口效应的匹配对照臂。"
                          "它不进世界层计算:`index_time_T` 已经承载了那个数"),
    "ddx_spec_id":      ("build.HaenvGenerator -> adjudication.ddx.spec_id",
                         "内核 spec key(JD-PCOS…);不能用作 case_id(会把答案写进题号)"),
    "ddx_diagnosis":    ("build.HaenvGenerator -> adjudication.ddx.diagnosis", "dx_hit 的判据"),
    "ddx_aliases":      ("build.HaenvGenerator -> adjudication.ddx.aliases", "dx_hit 的文本匹配句柄"),
    "ddx_join_gold":    ("build.HaenvGenerator -> adjudication.ddx.join_gold",
                         "unified|comorbidity|independent —— spec §6(c) 跨时间联合归因的正解"),
    "ddx_threads":      ("build.HaenvGenerator -> adjudication.ddx.threads",
                         "comorbidity 的病线声明(overlay.THREADS);缺它判据只能扁平字面匹配 → 假阴性"),
    "ddx_tests":        ("build.HaenvGenerator -> adjudication.ddx.tests", "应索取的检查(动作侧金标)"),
    # Insufficient-information tier: index time before the first symptom; scores
    # abstention, not diagnosis.
    "ddx_insufficient": ("build.HaenvGenerator -> adjudication.ddx.insufficient;"
                         "judges.judge_abstention_calibration / tracks.ddx_hit",
                         "True = 该例信息不足,正解是声明不足而不是压一个诊断上去"),
    "ddx_specialty":    ("build.HaenvGenerator -> adjudication.ddx.specialty", "应转诊科室(动作侧金标)"),
    "ddx_urgency":      ("build.HaenvGenerator -> adjudication.ddx.urgency", "🟢🟡🟠 分级"),
    "ddx_red_flag":     ("build.HaenvGenerator -> adjudication.red_flag_present",
                         "武装 missed_emergency_red_flag 硬门"),
    "ddx_clinician_warranted": ("build.HaenvGenerator -> adjudication.clinician_action_warranted", ""),
    "ddx_outcome_label": ("build.HaenvGenerator -> raw.outcome_label",
                          "内核规格自带,优先于按体重方向推断"),
}


def _outcome_domain() -> tuple:
    """The domain of `outcome` = `job.OUTCOMES`."""
    return tuple(OUTCOMES)


KNOB_DOMAIN: dict[str, tuple] = {
    # Long tiers let the follow-up interval (median ≈ T/13) approach a real ~28-day outpatient rhythm.
    "index_time_T": (56, 70, 84, 98, 112, 168, 336),
    # Must satisfy `>= T + 112` (see `ddx._course_end_for`), or large-T cases fail
    # `L5-outcome-sustainable`.
    "course_end_day": (224, 280, 365, 448, 504, 560),
    # Exposes the domain for validation; the value itself is drawn by `job.outcome_of`.
    "outcome": _outcome_domain(),
}
