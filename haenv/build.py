"""build.py -- question generation: raw case + hidden control variables ->
latent premise -> conditioned generation -> iterative validation -> emit.

Follows this order: (1) premise + validation, (2) GV-1
generate-validate, (3) noise/distractors derived from the premise, (3b)
day-by-day events with per-item validation and the physiology layer, (3c)
reading artifacts and post-injection on the observed series, (4) recheck +
leakage probe. A
case that fails any stage is not emitted. The kernel lives in
`haenv_kernel/`; this module does task-level assembly. SYNTHETIC, evaluation only.
"""
from __future__ import annotations

from .regpath import registry_cached as _registry_cached
from .yamlcache import load_yaml as _cached_yaml

from . import latent_rules as _LR

import copy
import json
import math
import logging

from haenv_kernel.schema import RawCase
from haenv_kernel.latent import LatentPremise, make_premise
from haenv_kernel.synth import synthesize, premise_conflicts
from haenv_kernel.noise import inject as inject_noise, inject_distractors
from haenv_kernel.build import build_instance, leakage_probe

from . import indicators as _indicators
from . import artifact_params as _artifact_params
from . import drug_effects as _drug_effects
from . import med_course as _med_course
from . import demographics as _demographics
from . import post_inject as _post_inject
from .job import raw_field as _raw_field
from . import events as events_mod
from .events_pools import EVENT_RATE_DEFAULTS as _EVENT_RATE_DEFAULTS
from . import gates
from . import verify as verify_mod
from . import wq
from .job import CaseSpec
from .job import driver_of as _job_driver_of, outcome_of as _job_outcome_of
from .build_weight import (  # noqa: F401
    END,
    SKEL_ANCHOR_MARGIN_KG,
    SKEL_DESCENT_MAX_PHASES,
    SKEL_DRIFT_BOUND_DOWN_KG,
    SKEL_DRIFT_BOUND_KG,
    SKEL_DRIFT_SD_KG,
    SKEL_DRIFT_SLOPE_FRAC,
    SKEL_DRIFT_TAU_DAYS,
    SKEL_EPISODE_MAX_DAYS,
    SKEL_EPISODE_PACE_FRAC,
    SKEL_FADE_DAYS,
    SKEL_GAIN_FRAC,
    SKEL_GAIN_RESIDUAL,
    SKEL_GAIN_TAU_DAYS,
    SKEL_HOLD_PACE,
    SKEL_LEVELS,
    SKEL_MAX_EPISODES,
    SKEL_MAX_RUN_DAYS,
    SKEL_REBOUND_VARIANTS,
    SKEL_REGAIN_MARGIN_DAYS,
    SKEL_REGAIN_MARGIN_KG,
    SKEL_REGAIN_RUN_MARGIN_KG,
    SKEL_REVERSAL_SLOPE_GAIN,
    SKEL_RUN_MARGIN_KG,
    SKEL_SETTLE_DAYS,
    SKEL_SETTLE_KG,
    SKEL_STALL_PACE,
    SKEL_VALLEY_MIN_DAYS,
    STEP,
    WEIGHT_MAX_GAP,
    WEIGHT_STEP,
    WEIGH_P_WEEKDAY,
    WEIGH_P_WEEKEND,
    WOBBLE_PERIOD,
    _SKELETON_AUDIT,
    _SKEL_N_DESCENT,
    _SKEL_N_DRIFT,
    _SKEL_N_EPISODE,
    _SKEL_N_REBOUND,
    _apply_irregularity,
    _descent_pieces,
    _drift_series,
    _ensure_regain,
    _episode_plan,
    _episode_profile,
    _feasible_k,
    _label_guard,
    _largest_ok,
    _max_step,
    _max_weekly_delta,
    _no_regain_until,
    _pace_at,
    _pace_pieces,
    _paced_progress,
    _pchip_daily,
    _post_t_plain,
    _rebound_pieces,
    _record_skeleton,
    _resample_to_grid,
    _reversal_gain,
    _reversal_visible,
    _shape,
    _skeleton_base,
    _skeleton_draws,
    _slope_kg_week,
    _smoothstep,
    _step_caps,
    _steps_within,
    _trajectory_shape,
    _trajectory_shape_registered,
    _weighed_on,
    _weight_overlay,
    _weight_render,
    _weight_series,
    _within_caps,
)
from .build_clinical import (  # noqa: F401
    CLINICAL_ATTEN,
    CLINICAL_BY_DEVICE,
    CLINICAL_LAG_DAYS,
    CLINICAL_SPEC,
    _assert_kernel_params_in_sync,
    _clinical_baselines,
    _clinical_cv,
    _drug_of,
    _drug_response_of,
    _drug_terms_for,
    _last_dose_mg,
    _quantize_within_slope,
    clinical_cohort,
    clinical_plan,
    render_clinical,
    world_layer_gaps,
    TRUTH_NDIGITS,
    WORLD_MEDICATION_KEY,
    _LAB_CTX,
    _PROMPT_PLAN_NDIGITS,
    _draw_clock,
    _drawn,
    _drug_term_fn,
    _lab_group_of,
    _lab_obs_cfg,
    _lab_own_cv,
    _pick,
    _routine_gap,
    _true_adherence_pts,
    _u01,
    _world_medication,
    _z,
    lab_draw_days,
    lab_item_days,
    lab_print_ndigits,
    lab_signals,
    observe_labs,
    world_medication_record,
)

log = logging.getLogger("haenv.build")

# Day the disease course ends. Equal to `END` (the extent of the world-layer
# streams) but a separate quantity; per-case end days come from `course_end_of`.
COURSE_END = END

# Domain of the course end day. Held at one value: a varying end day changes
# `prediction_window` on the question surface, and a shorter course can flip
# `outcome_label` via `label_rule.minimum_change_magnitude`.
COURSE_END_DOMAIN: tuple[int, ...] = (365,)


def course_end_of(case_id: str, latent: dict | None = None) -> int:
    """This case's course end day: the explicit declaration, else sampled from
    `case_id` alone (never from diagnosis/outcome).
    """
    if latent and latent.get("course_end_day") is not None:
        return int(latent["course_end_day"])
    from . import rng
    return int(rng.pick(list(COURSE_END_DOMAIN), case_id, "course_end"))

# --------------------------------------------------------------- Gold-standard evidence stream (world layer)
# Every case carries every registered evidence stream; only the gold driver's
# stream is abnormal. `GOLD_EVIDENCE` lives in `registry.py` and is
# re-exported here.
from .prompts import PROMPT  # noqa: E402,F401  (question-surface templates, re-exported)
from .registry import GOLD_EVIDENCE  # noqa: E402,F401  (re-exported)
from . import external_gold as _EG   # noqa: E402  external gold-standard registration surface (empty by default => strictly a no-op)
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data
from .build_clinical import (  # noqa: F401
    _BP_TEXT,
    _CLINICAL_LEDGER_AUDIT,
    _fmt_reading,
    _latest_at_or_before,
    _reading_at_or_before,
    realign_clinical_ledger,
    realign_ledger_to_streams,
)


def ledger_weight_point(ld: dict, T: int, cid: str) -> tuple[int, float]:
    """The ledger weight EV: the last point of the `weight` stream with ts ≤ T.

    Not the nadir, which can fall after T, and not the minimum, since a scale
    reports its current reading. Called both at construction and after noise
    injection, so the ledger matches the final noised stream.
    """
    pts = ld.get("weight")
    if not isinstance(pts, list) or not pts:
        raise ValueError(f"{cid}: the case has no weight stream; cannot build the weight-ledger EV")
    le_T = [q for q in pts
            if isinstance(q, dict) and isinstance(q.get("ts"), (int, float))
            and int(q["ts"]) <= int(T) and q.get("value") is not None]
    if not le_T:
        raise ValueError(f"{cid}: the weight stream has no points at ts <= T={T}; cannot build the weight-ledger EV")
    last = max(le_T, key=lambda q: int(q["ts"]))
    return int(last["ts"]), float(last["value"])


def gold_evidence_streams(driver: str, rev_week: int, occurred: bool,
                          T: int, end_day: int = END,
                          dens_step: int = 1, case_id: str = "C") -> dict[str, list[dict]]:
    """The gold-standard evidence streams: present in every case, abnormal only
    when this driver is the gold standard.

    Plateau then a 4-week rise whose onset is `min(rev_day, T) - lead_days`, so
    the evidence is visible before T. Fluctuation is AR(1) with a per-case,
    per-stream seed and amplitude `max(0.04 × normal, 0.8 tick)`, small enough
    to stay under GEN14's "flat" threshold (`gates.GEN14_FLAT_REL_SPAN`). The rise does not also lower
    `medication_adherence`.
    """
    out: dict[str, list[dict]] = {}
    rev_day = rev_week * 7
    step_eff = max(STEP, dens_step)
    for drv, spec in GOLD_EVIDENCE.items():
        is_gold = (drv == driver) and occurred
        onset = max(0, min(rev_day, int(T)) - int(spec["lead_days"]))
        lo, hi = spec["range"]
        pts = []
        # The 0.8-tick floor keeps coarse, small-valued signals from rounding to a
        # constant.
        _tick = 10.0 ** (-int(spec["ndigits"]))
        amp = max(0.04 * float(spec["normal"]), 0.8 * _tick)
        phi = events_mod._AR_PHI ** max(1, int(step_eff))
        sd = events_mod._AR_SD_FRAC * amp
        bound = sd * math.sqrt(max(0.0, 1.0 - phi * phi)) * math.sqrt(3.0)
        key = f"{case_id}|{spec['signal']}|gold"
        # Start from the stationary distribution, so there is no warm-up segment.
        x = sd * events_mod._det_shock(key, -1)
        for k, d in enumerate(range(0, end_day + 1, step_eff)):
            if k:
                x = phi * x + bound * events_mod._det_shock(key, k)
            v = spec["normal"]
            if is_gold and d >= onset:
                ramp = min(1.0, (d - onset) / 28.0)
                v = spec["normal"] + (spec["abnormal"] - spec["normal"]) * ramp
            v += x
            pts.append({"ts": d, "value": round(min(max(v, lo), hi), spec["ndigits"])})
        out[spec["signal"]] = pts
    return out


# --------------------------------------------------------------- weight skeleton
# Four irregularities on top of the three-segment skeleton: an uneven descent,
# plateau episodes, a per-case rebound shape, and transient ups and downs. Every
# random number comes from `_skeleton_draws(case_id)`; the rest is deterministic.

class HaenvGenerator:
    """Deterministically generates the disease course from the latent premise;
    conflict checking is left to synth (GV-1).
    """

    def generate(self, p: LatentPremise, original_case, feedback) -> RawCase:
        m = p.meta
        T = int(m.get("prediction_time_T", 84))
        start, nadir = float(m["start"]), float(m["nadir"])
        # Density can only thin a stream: `step_eff = max(declared step,
        # round(7 / measure_per_week))`, a no-op at the default `mpw=7`.
        _ed_mpw = float((getattr(p, "event_density", None) or {}).get("measure_per_week", 7) or 7)
        _dens_step = max(1, int(round(7.0 / max(0.1, _ed_mpw))))
        _cid = str(m.get("case_id") or "HAENV")
        outcome = _job_outcome_of(_cid, m)
        rev_week = int(m.get("reversal_week", (T + 56) // 7))
        slope = float(m.get("regain_slope", 0.35))
        occurred = outcome == "regain"
        driver = _job_driver_of(_cid, m)
        ce = int(m.get("course_end_day", END))
        # With `regain_end_kg`, the rebound slope is back-solved from the endpoint
        # over the rebound weeks; it is not clamped, so an infeasible endpoint fails
        # GV-1 visibly.
        _end_kg = m.get("regain_end_kg")
        if _end_kg is not None and outcome == "regain":
            slope = (float(_end_kg) - nadir) / max(1e-6, (ce - rev_week * 7) / 7.0)
        w_step = int(_raw_field((p.device_signals.get("signals") or {}).get("weight") or {},
                                "sampling_days"))
        wpts = _weight_series(start, nadir, T, outcome, rev_week, slope,
                              case_id=m.get("case_id", "HAENV"),
                              step=max(w_step, _dens_step), end_day=ce,
                              disease=_raw_field(p.patient_basics, "disease"),
                              nadir_day=m.get("nadir_day"))

        # The wearable EV comes from the weight stream at ≤T (see `ledger_weight_point`).
        _ev_ts, _ev_val = ledger_weight_point({"weight": wpts}, T, _cid)

        # True adherence drives the drug effect; the record shows an observation of it and of
        # the dose course, plus concurrent medicines (`med_course`).
        adh = _true_adherence_pts(p, T)
        dose, adh_obs, concurrent = _world_medication(p, _cid, T, ce)
        # Lab-calendar streams are rendered noise-free on the case's visit days
        # (`lab_draw_days`); `observe_labs` adds the measurement layer after GV-1.
        _clin = m.get("clinical") or {}
        _lab_days = lab_item_days(_cid, [s for s in _clin if s in lab_signals()],
                                  lab_draw_days(_cid, ce, dose))
        _LAB_CTX[_cid] = {"wpts": wpts, "ce": ce, "drug": _drug_of(p),
                          "dose_mg": _last_dose_mg(dose), "adh": adh,
                          "response": _drug_response_of(p), "clinical": _clin,
                          # drug effect along the dose line, same as the generator's render
                          "drug_term": (lambda _s, _c: _drug_term_fn(p, _s, _c, dose, adh,
                                                                     concurrent))}

        return RawCase(
            case_id=m.get("case_id", "HAENV"),
            user_profile={"age_range": _raw_field(p.patient_basics, "age_range"),
                          "sex": _raw_field(p.patient_basics, "sex"),
                          # M2 explainer bases are named in the record (the explainer must be
                          # visible, spec 4.2); other comorbidities stay unnamed as in v1.0.1.
                          "known_conditions": [_raw_field(p.patient_basics, "disease")]
                          + [c for c in (p.patient_basics.get("comorbidities") or [])
                             if c in EXPLAINER_BASES],
                          "treatment_goals": p.patient_basics.get("goals", ["sustained_weight_loss"]),
                          "device_inventory": p.device_signals.get("devices", [])},
            prediction_context={
                                # External task types' solver-visible block; `{}` when none is registered.
                                **_EG.probe_blocks_for(m),
                                "prediction_time_T": T,
                                "target_event_type": m.get("target_event_type", "weight_regain"),
                                "prediction_window": f"{ce - T}d",
                                "available_history_window": f"{T}d"},
            longitudinal_data={"weight": wpts, "dose_timeline": dose,
                               "medication_adherence": adh_obs,
                               **{s: render_clinical(
                                       s, c, wpts, ce,
                                       drug=_drug_of(p),
                                       dose_mg=_last_dose_mg(dose),
                                       adherence_pts=adh, case_id=_cid,
                                       response=_drug_response_of(p),
                                       drug_term=_drug_term_fn(p, s, c, dose, adh, concurrent),
                                       days=_lab_days.get(s))
                                  for s, c in (m.get("clinical") or {}).items()},
                               **gold_evidence_streams(driver, rev_week, occurred, T,
                                                       end_day=ce,
                                                       dens_step=_dens_step,
                                                       case_id=_cid)},
            evidence_ledger=[{"evidence_id": f"EV-{m.get('case_id','C')}-1",
                              "source_type": "wearable", "source_timestamp": _ev_ts,
                              "measured_value": _ev_val, "claim_supported": True,
                              "reliability_status": "reliable"}],
            # A declared outcome_label (diagnosis cases) takes precedence; otherwise from the weight direction.
            outcome_label=(m.get("outcome_label")
                           or ("event_occurred" if occurred else "event_not_occurred")),
            # The judge reads the structured fields; the string is for humans.
            label_rule={"minimum_change_magnitude":
                            f"regain >= max({_LR.MIN_CHANGE_FRAC:.0%} of lost weight, "
                            f"{_LR.MIN_CHANGE_KG} kg), read on the 7-reading centred rolling median; "
                            f"with less than {_LR.MIN_CHANGE_KG} kg lost it is a gain of "
                            f"{_LR.MIN_CHANGE_KG} kg above the low point",
                        "minimum_persistence": "sustained >=8w", "baseline_window": "nadir",
                        # Thresholds come from `latent_rules`, which also checks them before generation.
                        "min_change_frac": _LR.MIN_CHANGE_FRAC,
                        "min_change_kg": _LR.MIN_CHANGE_KG,
                        "smoothing": _LR.LABEL_SMOOTHING,
                        "min_persist_days": _LR.MIN_PERSIST_DAYS},
            gold_drivers=[driver],
            adjudication={"adjudication_protocol_present": True, "primary_driver": driver,
                          # From latent.ddx_red_flag; a hardcoded False would disarm missed_emergency_red_flag.
                          "red_flag_present": bool(m.get("red_flag", False)),
                          "clinician_action_warranted": (occurred
                                                         if m.get("clinician_warranted") is None
                                                         else bool(m.get("clinician_warranted"))),
                          **({"ddx": m["ddx"]} if m.get("ddx") else {}),
                          # External task types' gold block; `{}` when none is registered.
                          **_EG.blocks_for(m)},
            reversal_points=([{"week": rev_week, "type": "real", "flip": "risk_low->elevated",
                               "trigger": f"day{rev_week*7} 起回升,证伪'能维持'"}] if occurred else []),
        )


# --------------------------------------------------------------- LLM generator
_CASE_PROMPT = """你是"合成病程"生成器。依据下面的**隐变量前提**与**原始病例事实**,生成一段纵向病程观测。
只输出一个 JSON 对象(无多余文字、无 markdown 代码围栏)。

## 硬约束(违反会被确定性校验器判负并要求你重出)
1. 信号名只能用:{signals};单位与取值必须落在前提声明的 plausible_range 内。
2. 相邻采样点的**周变化**不得超过生理上限:{deltas}。
3. `dose_timeline` 只能取该药剂量梯级 {ladder} 中的值,相邻两次**变更**间隔 ≥ {titration} 天;
   首次给药之前主信号不得已明显偏离基线(药效不能早于用药)。
4. 主信号必须走出前提要求的形态:从 {start} 起降至约 {nadir}(第 {nadir_day} 天前后),
   {shape}
5. `medication_adherence` 取 0–1,最低点须与前提的依从轨迹大致相符(不得比前提最低值再低 0.2 以上)。
6. 时间范围:ts 从 0 到 {end} 天;主信号按每 7 天一个点即可(不必逐天)。
7. **答案侧信息只能出现在 outcome_label/gold_drivers/adjudication/reversal_points 这些字段里**;
   `evidence_ledger` 的任何文字(症状、备注)**不得**出现结局词/归因词(如"复胖""停药后""甲亢""无效""依从性差"),
   因为 evidence_ledger 会原样交给被测模型看。

## 隐变量前提(verifier 侧,含本例的真结局与真驱动 —— 据此生成,但别写进 evidence_ledger)
{premise}

## 原始病例事实(必须与之一致)
{original}

## 上一轮被校验器判负的冲突(必须逐条修掉)
{feedback}

## 输出 JSON 结构
{{"longitudinal_data":{{"<主信号>":[{{"ts":0,"value":..}},...],
   "dose_timeline":[{{"ts":0,"value":..}},...],"medication_adherence":[{{"ts":..,"value":0-1}},...]}},
 "evidence_ledger":[{{"evidence_id":"EV-{cid}-1","source_type":"smart_scale|wearable|lab_panel|clinic_scale|cgm|bp_cuff",
   "source_timestamp":<=T,"measured_value":..,"claim_supported":true,"reliability_status":"reliable"}}],
 "label_rule":{{"minimum_change_magnitude":"..","minimum_persistence":"..","baseline_window":"nadir"}}}}"""


class LLMCaseGenerator:
    """Generates the disease course with a real model: the model writes the evidence
    ledger; every longitudinal series (weight, dose, adherence, clinical and gold-evidence
    streams) is the deterministic skeleton `HaenvGenerator` draws from the premise, and the
    model's series are not adopted. A value the model quoted in the ledger is re-read from
    the rendered stream (`realign_clinical_ledger` here, `realign_ledger_to_streams` on the
    final streams).

    Output goes through `synth.synthesize`'s GV-1 loop, with conflicts fed back
    for regeneration. The gold standard (outcome_label/gold_drivers/
    adjudication/reversal_points) always comes from the hidden control
    variables, never from the model.
    """

    def __init__(self, dispatch, cs: CaseSpec):
        self.dispatch, self.cs = dispatch, cs

    def generate(self, p: LatentPremise, original_case, feedback) -> RawCase:
        from haenv_kernel.latent import DISEASE_SIGNAL_DOMAIN, drug_pkpd
        from haenv_kernel.solver import _extract_json

        m = p.meta
        T = int(m.get("prediction_time_T", 84))
        disease = _raw_field(p.patient_basics, "disease")
        domain = DISEASE_SIGNAL_DOMAIN.get(disease, {})
        sigs = list(p.device_signals.get("signals", {}))
        pk = drug_pkpd(p.patient_basics.get("regimen", {}).get("drug", "")) or {}
        # Same single source as `HaenvGenerator.generate`.
        outcome = _job_outcome_of(str(m.get("case_id") or self.cs.case_id), m)
        rev_week = int(m.get("reversal_week", (T + 56) // 7))
        _ce_llm = int(m.get("course_end_day", END))
        shape = (f"随后在第 {rev_week * 7} 天前后转为回升,到第 {_ce_llm} 天累计回升 "
                 f"≥ 已减体重的 5%(斜率约 {m.get('regain_slope', 0.35)}/周)。"
                 if outcome == "regain" else
                 f"随后一直维持在该水平附近(到第 {_ce_llm} 天不得出现持续 8 周以上的明显回升)。")
        prompt = _CASE_PROMPT.format(
            signals=sigs, cid=self.cs.case_id,
            deltas={s: domain.get(s, {}).get("max_weekly_delta") for s in sigs},
            ladder=list(pk.get("ladder", ())), titration=pk.get("min_titration_days", 28),
            start=m.get("start"), nadir=m.get("nadir"), nadir_day=T, shape=shape, end=_ce_llm,
            premise=json.dumps(_world_prompt_premise(p), ensure_ascii=False),
            original=json.dumps(original_case or {}, ensure_ascii=False),
            feedback=json.dumps(sorted({x.get("kind") for x in (feedback or [])}),
                                ensure_ascii=False) or "(首轮,无)")
        data = _extract_json(self.dispatch(prompt, require_json=True))

        # Every series comes from the deterministic skeleton (weight, dose, adherence, clinical
        # streams, gold-evidence streams); the model's `longitudinal_data` only tells
        # `realign_clinical_ledger` which stream a ledger number quotes.
        raw = HaenvGenerator().generate(p, original_case, feedback)
        ld = data.get("longitudinal_data") or {}
        evs = data.get("evidence_ledger")
        if isinstance(evs, list) and evs:
            raw.evidence_ledger = [e for e in evs if isinstance(e, dict) and e.get("evidence_id")]
        _CLINICAL_LEDGER_AUDIT[str(self.cs.case_id)] = realign_clinical_ledger(
            raw.evidence_ledger, raw.longitudinal_data, reference=ld)
        # The model's `label_rule` is discarded: it decides `outcome_label`, and the
        # gold label is never handed to the generation model.
        if isinstance(data.get("label_rule"), dict) and data["label_rule"]:
            log.debug("[build] %s the model supplied label_rule; dropped (it would decide the gold label)",
                      self.cs.case_id)
        log.info("[build] %s course generated by the LLM (signals %s, %d evidence entries)", self.cs.case_id,
                 {k: len(v) for k, v in raw.longitudinal_data.items()}, len(raw.evidence_ledger))
        return raw


def original_facts(cs: CaseSpec, T: int) -> dict:
    """The case facts handed to the generator (no outcome); anchor values also feed
    premise_conflicts for the recheck.
    """
    raw = cs.raw
    start = float(_raw_field(raw, "start_weight"))
    nadir = float(_raw_field(raw, "nadir_weight"))
    return {"disease": raw.get("disease"), "drug": raw.get("drug"),
            "dose_steps": raw.get("dose_steps"), "devices": raw.get("devices"),
            "age_range": raw.get("age_range"), "sex": raw.get("sex"),
            "comorbidities": _raw_field(raw, "comorbidities"), "bmi": raw.get("bmi"),
            "baseline_vitals": raw.get("baseline_vitals", {}),
            "index_time_T": T, "course_end_day": course_end_of(cs.case_id, cs.latent),
            "anchor_values": {"weight": {"day": 0, "value": start, "tol": 1.0}},
            "nadir_weight": nadir}


# --------------------------------------------------------------- Clinical signal streams (GEN7)
#
# Every declared device produces its streams (lab_panel, cgm, bp_cuff). Names
# must match the kernel's `DISEASE_SIGNAL_DOMAIN`, be declared in the premise
# with the actual step (GEN6), and respect `max_weekly_delta`. Values follow
# weight with a lag and attenuation, so they never disclose more than weight.
#: M2 explainer base conditions (`registry/background_comorbidity.yaml:explainer_bases`): the only
#: comorbidities written into `user_profile.known_conditions`.
EXPLAINER_BASES = ("hypothyroidism", "CAD", "CKD")


# --------------------------------------------------------------- Premise construction
#: Premise `meta` keys about delivery to the solver, not the patient; kept out
#: of the world-course prompt so they cannot change the patient's course.
OBSERVATION_ONLY_META = ("rhythm_gap",)

#: Premise `meta` keys the world-course prompt may see; all others are
#: withheld. Every key `premise_spec` emits is in exactly one of the two tuples.
WORLD_PROMPT_META = (
    _EG.SLOT, "case_id", "difficulty_class", "target_event_type", "prediction_time_T",
    "start", "nadir", "course_end_day", "nadir_day", "outcome", "driver",
    "reversal_week", "regain_slope", "regain_end_kg", "clinical", "ddx", "red_flag",
    "clinician_warranted", "outcome_label", "drug_response",
)


#: Premise `meta` keys that carry the answer (CLAUDE.md 2a: gold is never handed to the generation model).
GOLD_META = ("ddx", "red_flag", "clinician_warranted", "outcome_label")


def _world_prompt_premise(p) -> dict:
    """The premise as the world-course model sees it: allow-listed meta; of the external slot only the
    latent keys a plugin classed `patient_fact`; and a case that carries any external latent (a plugin
    task, not a v1.0.1 case) shows patient facts only, so `GOLD_META` is dropped for it."""
    d = p.dumps()
    meta = {k: v for k, v in (d.get("meta") or {}).items() if k in WORLD_PROMPT_META}
    cls = _EG.latent_classes()
    slot = meta.get(_EG.SLOT) or {}
    ext = {k: v for k, v in slot.items() if cls.get(k) == "patient_fact"}
    d["meta"] = {k: (ext if k == _EG.SLOT else v) for k, v in meta.items()
                 if (k != _EG.SLOT or ext) and not (slot and k in GOLD_META)}
    return d


def _register_comorbid_domain(disease: str, comorbidities):
    """Adds the comorbidities' signals to `DISEASE_SIGNAL_DOMAIN[disease]` for the
    duration of one case, without editing kernel source.

    The kernel's premise checks only look at the primary condition's domain.
    The addition is scoped to the current thread and removed afterwards, so it
    never leaks into other cases.
    """
    from contextlib import contextmanager                     # noqa: PLC0415

    @contextmanager
    def _noop():
        yield

    if not comorbidities:
        return _noop()
    from haenv_kernel.latent import DISEASE_SIGNAL_DOMAIN
    dom = DISEASE_SIGNAL_DOMAIN.get(str(disease))
    if dom is None:
        return _noop()

    from haenv_kernel.latent import COMORBIDITY_SIGNAL_DOMAIN
    extra: dict = {}
    for c in comorbidities:
        for sig, spec in (DISEASE_SIGNAL_DOMAIN.get(str(c)) or COMORBIDITY_SIGNAL_DOMAIN.get(str(c)) or {}).items():
            if sig not in dom:
                extra.setdefault(sig, spec)
    return DISEASE_SIGNAL_DOMAIN.scoped(str(disease), extra)


def premise_spec(cs: CaseSpec, T: int = 84) -> dict:
    """raw (case facts) + latent (hidden control variables) -> the latent premise
    spec.
    """
    raw, lat = cs.raw, cs.latent
    disease = _raw_field(raw, "disease")
    drug = _raw_field(raw, "drug")
    steps = list(raw.get("dose_steps", [2.5, 5.0, 7.5]))
    devices = list(raw.get("devices", ["smart_scale", "wearable"]))
    start = float(_raw_field(raw, "start_weight"))
    nadir = float(_raw_field(raw, "nadir_weight"))
    outcome, driver = cs.outcome, cs.driver

    adh_low = float(lat.get("adherence_low", 0.95))
    # Post-T adherence points sit at the midpoint and the end of the course.
    _ce = course_end_of(cs.case_id, lat)
    _mid = T + (_ce - T) // 2
    if driver == "poor_medication_adherence" and outcome == "regain":
        traj = [{"day": 42, "level": 0.95}, {"day": 56, "level": 0.88}, {"day": T, "level": 0.80},
                {"day": _mid, "level": round((0.80 + adh_low) / 2, 2)},
                {"day": _ce, "level": adh_low}]
    else:
        traj = [{"day": 42, "level": 0.96}, {"day": T, "level": 0.94},
                {"day": _mid, "level": 0.93},
                {"day": _ce, "level": max(0.90, adh_low)}]

    ed = lat.get("event_density", {}) or {}
    # The value range covers the whole trajectory, including the regain tail.
    rev_week = int(lat.get("reversal_week", (T + 56) // 7))
    slope = float(lat.get("regain_slope", 0.35))
    # Same endpoint rule as `HaenvGenerator.generate`.
    _end_kg = lat.get("regain_end_kg")
    proj_end = (float(_end_kg) if (_end_kg is not None and outcome == "regain")
                else (nadir + slope * max(0, (_ce - rev_week * 7)) / 7.0
                      if outcome == "regain" else nadir))
    lo = max(40.0, min(nadir, start) - 4)
    hi = min(250.0, max(start, nadir, proj_end) + 4)
    # Declared sampling interval, using the same density expression as the
    # rendered stream (GEN6 compares the two).
    _ed_mpw0 = float((lat.get("event_density") or {}).get("measure_per_week", 7) or 7)
    sampling_days = max(int(_raw_field(raw, "sampling_days")),
                        max(1, int(round(7.0 / max(0.1, _ed_mpw0)))))
    _comorb = _raw_field(raw, "comorbidities")
    return {
        "patient_basics": {"disease": disease, "comorbidities": _raw_field(raw, "comorbidities"),
                           "age_range": _raw_field(raw, "age_range"), "sex": _raw_field(raw, "sex"),
                           "regimen": {"drug": drug, "dose_steps": steps},
                           "goals": raw.get("goals", ["sustained_weight_loss"])},
        # Rate defaults come from `events.EVENT_RATE_DEFAULTS`, shared with the injector and gates.
        "event_density": {"measure_per_week": ed.get("measure_per_week", 7),
                          "dosing_per_week": ed.get("dosing_per_week", 1),
                          **{k: ed.get(k, d) if ed.get(k, d) is not None else d
                             for k, d in _EVENT_RATE_DEFAULTS.items()}},
        "device_signals": {"devices": devices,
                           # Primary signal plus the clinical signals of `clinical_plan`, each declared
                           # with its actual step (GEN6).
                           "signals": {"weight": {"unit": "kg", "sampling_days": sampling_days,
                                                  "plausible_range": [lo, hi],
                                                  # Declared missingness for GEN6; everyday gaps exist only under daily sampling.
                                                  **({"expected_missing_rate": round(
                                                      1 - (5 * WEIGH_P_WEEKDAY
                                                           + 2 * WEIGH_P_WEEKEND) / 7, 3),
                                                      "max_gap_days": WEIGHT_MAX_GAP}
                                                     if sampling_days == WEIGHT_STEP else {})},
                                       **{s: {"unit": c["unit"], "sampling_days": c["step"],
                                              "plausible_range": c["range"]}
                                          for s, c in clinical_plan(disease, devices, _comorb, cs.case_id).items()}},
                           # Gold evidence streams are not declared here: they are kernel `AUX_SIGNALS`,
                           # checked by GEN14 rather than GEN6.
                           "n_signals": 1 + len(clinical_plan(disease, devices, _comorb, cs.case_id))},
        "adherence": {"baseline": 0.95, "trajectory": traj,
                      "missingness_mechanism": lat.get("missingness", "MAR")},
        # The first entry passes external task types' latent keys through (see
        # `haenv/external_gold.py`); it must sit inside `meta`, since `LatentPremise`
        # drops unknown top-level keys.
        "meta": {**({_EG.SLOT: {k: lat[k] for k in _EG.passthrough_keys() if k in lat}}
                     if _EG.passthrough_keys() else {}),
                 "case_id": cs.case_id, "difficulty_class": lat.get("difficulty", "B"),
                 "target_event_type": lat.get("target_event", "weight_regain"),
                 "prediction_time_T": T, "start": start, "nadir": nadir,
                 "course_end_day": _ce,
                 # Low point's day; `None` => `T`. Keys reach the generator only if listed here.
                 "nadir_day": lat.get("nadir_day"),
                 "outcome": outcome, "driver": driver,
                 # Drug-response override, emitted only when declared (an always-present `null`
                 # would change every cached world prompt).
                 **({"drug_response": lat["drug_response"]}
                    if lat.get("drug_response") is not None else {}),
                 "reversal_week": int(lat.get("reversal_week", (T + 56) // 7)),
                 "regain_slope": float(lat.get("regain_slope", 0.35)),
                 # Rebound endpoint (kg); `None` => `regain_slope`.
                 "regain_end_kg": lat.get("regain_end_kg"),
                 "clinical": clinical_plan(disease, devices, _comorb, cs.case_id),
                 # Diagnosis ground truth, landed into adjudication.ddx.
                 "ddx": cs.ddx,
                 "red_flag": bool(lat.get("ddx_red_flag", False)),
                 "clinician_warranted": lat.get("ddx_clinician_warranted"),
                 "outcome_label": lat.get("ddx_outcome_label"),
                 # Per-case "information gap" flag (read by `evaluate.slices_for` under
                 # `slices: real-rhythm+gap`), so a batch can mix gap and non-gap cases.
                 "rhythm_gap": bool(lat.get("rhythm_gap", False))},
    }


# --------------------------------------------------------------- (3b) day-by-day event closed loop
def _prompt_frame() -> str:
    """The prompt template sent to the solver (no payload), for the text-leakage scan."""
    return PROMPT


def anonymize_evidence_ids(raw: RawCase, report: dict | None = None) -> dict[str, str]:
    """Replaces EV ids with category-free sequence numbers; returns the old -> new
    mapping.

    Planning ids carry the category (`-S` real symptom, `-B`/`-L` injected
    benign/life event, `-D` upstream distractor), which would leak the
    signal/noise split. Renaming happens after the drop-and-reinject loop (which
    uses planning ids as handles) and before `build_instance`. It updates the
    ledger, the Q-side injection manifest and the validation report together.
    Numbering follows `(source_timestamp, planning id)`, which adds no
    information.
    """
    led = list(raw.evidence_ledger or [])
    order = sorted(range(len(led)),
                   key=lambda i: (int(led[i].get("source_timestamp", 10 ** 9)),
                                  str(led[i].get("evidence_id", ""))))
    cid = raw.case_id
    id_map = {str(led[i].get("evidence_id", "")): f"EV-{cid}-{n:02d}"
              for n, i in enumerate(order, 1)}
    for e in led:
        old = str(e.get("evidence_id", ""))
        if old in id_map:
            e["evidence_id"] = id_map[old]
    man = wq.injected_manifest(cid, required=False)
    # Rename every matching string in the manifest, whatever the field's name or shape.
    def _rename(x):
        if isinstance(x, str):
            return id_map.get(x, x)
        if isinstance(x, list):
            return [_rename(i) for i in x]
        if isinstance(x, dict):
            return {k: _rename(v) for k, v in x.items()}
        return x

    man = _rename(man)
    # Added after `_rename` (its keys are the old ids); verifier-only.
    man["id_map"] = id_map
    wq.register_injection(cid, man)
    if report:
        for it in (report.get("items") or []):
            if str(it.get("item", "")) in id_map:
                it["item"] = id_map[str(it["item"])]
    return id_map


def _first_response_only(dispatch):
    """`dispatch` asked once; every later call returns that first response."""
    box: list[str] = []

    def once(prompt: str, require_json: bool = False) -> str:
        if not box:
            box.append(dispatch(prompt, require_json=require_json))
        return box[0]
    return once


def _inject_events_verified(raw: RawCase, cs: CaseSpec, p: LatentPremise, T: int,
                            max_rounds: int, dispatch=None) -> tuple[RawCase, dict]:
    """Injects day-by-day events, validates them per item, and re-injects without
    the failing items until it converges or rounds run out.

    Every round starts from the same pre-injection `base`. With `dispatch`, an LLM
    writes the events once, on its first-round prompt; later rounds re-inject them
    without the failing items, and pool events fill the case back up to its density
    (`events._pool_backfill`), as on the deterministic channel. The daily metrics the
    verifier also judges are code's (`events.plan_streams`), so a re-ask would hand the
    model failure reasons about streams it does not choose.
    """
    if dispatch is not None:
        dispatch = _first_response_only(dispatch)
    base = copy.deepcopy(raw)
    rev_day = (int(cs.latent.get("reversal_week", (T + 56) // 7)) * 7
               if cs.outcome == "regain" else None)
    drop: set[str] = set()
    reasons: dict[str, list[str]] = {}             # per-item drop reasons, kept for the audit
    frame = _prompt_frame()
    # Pre-initialized for `max_rounds <= 0`.
    cand, rep, injected, manifest = base, {}, {}, {}
    for r in range(1, max_rounds + 1):
        cand = copy.deepcopy(base)
        cand, manifest = events_mod.inject(cand, cs, p, cs.driver, drop,
                                           dispatch=dispatch, feedback=reasons)
        injected = manifest.get("injected") or {}   # Q-side ledger; registered only after convergence
        rep = verify_mod.verify_case(base, cand, cs, p, manifest, cs.driver, rev_day, frame)
        rep["round"] = r
        rep["dropped"] = sorted(drop)
        rep["drop_reasons"] = reasons
        if rep["ok"]:
            break
        if rep["bad_items"]:                       # per-item failure -> drop then re-inject
            for it in rep["items"]:
                if not it["ok"]:
                    reasons[it["item"]] = [k for k, v in it["checks"].items() if not v["ok"]]
            log.warning("[build] %s round %d: dropped non-conforming items %s", cs.case_id, r, rep["bad_items"])
            drop |= set(rep["bad_items"])
            continue
        break                                      # case-level failure (invariance/text leakage): dropping items cannot fix it
    bad = rep.get("bad_items", []) + [k for k, v in (rep.get("case_checks") or {}).items()
                                      if not v.get("ok")]
    items = rep.get("items", [])
    metric_kinds = {"daily_metric", "inherited_metric"}
    return cand, {"ok": bool(rep.get("ok")), "rounds": rep.get("round"), "bad": bad,
                  "dropped": sorted(drop),
                  "n_streams": sum(1 for i in items if i["kind"] in metric_kinds),
                  "n_events": sum(1 for i in items if i["kind"] not in metric_kinds),
                  "report": rep, "injected": injected,
                  # Pre-noise ground-truth trajectory when the physiology layer is on
                  # (verifier-only, for step (4)); `None` when off.
                  "physio_clean_ld": (manifest or {}).get("_clean_longitudinal_data")}


#: `premise_conflicts` kinds about what the solver sees before T. With the physiology layer on,
#: they are judged on the observed series: the truth series carries neither the reading
#: artifacts nor the clinic-scale reference.
OBSERVED_CONFLICT_KINDS = frozenset({"window_trap_not_visible_pre_T"})


def last_shown_day(raw: RawCase, geometry: dict | None) -> int:
    """The last day whose readings the solver is shown, for the geometry the case runs in
    (`CaseSpec.geometry`): T for single-shot and gated; the last slice for slices
    (`evaluate.slices_for`; fewer than two slices run single-shot); the last round's pointer
    for multi-round (`runner.run_multiround`: a round every `cadence_days` from T while the
    pointer stays within T + prediction window, at most 60 rounds)."""
    T = int(raw.prediction_context["prediction_time_T"])
    name = (geometry or {}).get("name", "single")
    if name == "slices":
        from .rows import slices_for
        days = slices_for(raw, geometry["slices"])
        return int(max(days)) if len(days) >= 2 else T
    if name == "multi":
        cadence = int(geometry["cadence_days"])
        w = str(raw.prediction_context.get("prediction_window", "180d")).rstrip("d")
        window = int(w) if w.isdigit() else 180
        return T + cadence * min(window // cadence, min(60, window // cadence + 2) - 1)
    return T


def trap_visibility_conflicts(raw: RawCase, last_day: int) -> list[dict]:
    """`trap_not_visible_to_solver` (gate) for a trap whose evidence is not in the readings the
    solver is shown (ts <= `last_day`).

    Point artifacts (`transient_spike`, `unit_error`): GEN27 (`gates.check_trap_evidence`)
    on the primary series cut at `last_day` -- the trap's reading is shown and stands out
    from the neighbours shown with it. `device_switch`: at least one reading from the switch
    to `last_day`, the same standard as a single-reading artifact. `context_confound` has its
    own gate (`window_trap_not_visible_pre_T`)."""
    from haenv_kernel.noise import POINT_ARTIFACTS, PRIMARY
    shown = [q for q in raw.longitudinal_data.get(PRIMARY) or [] if int(q["ts"]) <= last_day]
    traps = [rp for rp in raw.reversal_points or [] if rp.get("type") == "trap"]
    out: list[dict] = []
    if any(rp.get("noise_class") in POINT_ARTIFACTS for rp in traps):
        cut = copy.copy(raw)
        cut.longitudinal_data = {PRIMARY: shown}
        out += [{"kind": "trap_not_visible_to_solver", "severity": "gate",
                 "detail": f"shown to day {last_day}: {h['detail']}"}
                for h in gates.check_trap_evidence(cut)]
    for rp in traps:
        if rp.get("noise_class") != "device_switch":
            continue
        d0 = int(rp["day"])
        if not any(int(q["ts"]) >= d0 for q in shown):
            out.append({"kind": "trap_not_visible_to_solver", "severity": "gate",
                        "detail": f"device_switch@day{d0}: no {PRIMARY} reading from the "
                                  f"switch to day {last_day}"})
    return out


#: Kernel noise classes applied before the events and physiology layers: `adherence_gap`
#: changes the patient (true adherence) and `mnar_missing` removes readings. Every other
#: class moves readings, which is a property of the measurement: those go onto the observed
#: series after the physiology layer (`_inject_observation_artifacts`), so its truth series
#: never carries them.
NOISE_BEFORE_PHYSIO = frozenset({"adherence_gap", "mnar_missing"})


def _inject_kernel_noise(raw: RawCase, nz: dict) -> RawCase:
    """One declared noise item, amplitude from `registry/artifact_rates.yaml`."""
    d0 = int(nz["week"]) * 7
    return inject_noise(raw, nz["class"], d0, d0 + int(nz.get("span_days", 10)),
                        **_artifact_params.kwargs_for(nz["class"]))


def _inject_observation_artifacts(raw: RawCase, cs: CaseSpec,
                                  applied: list[str]) -> tuple[RawCase, list[str]]:
    """Puts the declared reading artifacts on the observed series, at their injected size.

    Returns the case and the items whose injector found no reading to move (no trap).
    A kernel injector also adds a clinic-scale reference (`weight_ref`) sampled from the
    series it is given, here the observed one; that copy is dropped, so every case's
    reference comes from the `post_inject` backfill of the truth series.
    """
    todo = [nz for nz in cs.noise if nz["class"] not in NOISE_BEFORE_PHYSIO]
    if not todo:
        return raw, []
    had_ref = "weight_ref" in raw.longitudinal_data
    unplaced: list[str] = []
    for nz in todo:
        n_rp = len(raw.reversal_points)
        raw = _inject_kernel_noise(raw, nz)
        (applied if len(raw.reversal_points) > n_rp else unplaced).append(
            f"{nz['class']}@wk{nz['week']}")
    if not had_ref:
        raw.longitudinal_data.pop("weight_ref", None)
    raw.case_id = cs.case_id                                    # the injector mutated id, restore it
    return raw, unplaced


# --------------------------------------------------------------- Main pipeline
def build_case(cs: CaseSpec, max_rounds: int = 6, T: int | None = None,
               dispatch=None, event_dispatch=None) -> tuple[RawCase | None, dict]:
    """The complete question-generation loop for one case. Returns
    `(RawCase | None, audit)`; `None` = not emitted.

    `T` defaults to the case's `latent.index_time_T`. `dispatch` / `event_dispatch`
    are the LLM dispatchers for course generation / event planning (`None` =
    deterministic).
    """
    # The comorbidity domain must be in scope for every gate, not just `make_premise`.
    with _register_comorbid_domain(_raw_field(cs.raw, "disease"),
                                   _raw_field(cs.raw, "comorbidities") or []):
        return _build_case_inner(cs, max_rounds, T, dispatch, event_dispatch)


def _build_case_inner(cs: CaseSpec, max_rounds: int = 6, T: int | None = None,
                      dispatch=None, event_dispatch=None) -> tuple[RawCase | None, dict]:
    """The body of `build_case`."""
    T = int(T if T is not None else cs.index_time_T)
    audit: dict = {"case_id": cs.case_id, "emitted": False, "T": T,
                   "generator": "llm" if dispatch else "deterministic",
                   "event_generator": "llm" if event_dispatch else "deterministic"}
    # Validate the active registry once per process, fail-closed; the result is
    # recorded in `audit`.
    from .overlay import validate_registry as _vreg
    audit["registry_validated"] = _vreg()        # raises RegistryInvalid on failure
    # Latent-key consistency (`latent_rules`): every contradiction is recorded in
    # the audit so a failed case points at the keys to fix.
    from .latent_rules import audit_latent, blocking_contradictions, contradictions
    _lat = {**(cs.latent or {}), "index_time_T": T}
    _lc = audit_latent(dict(cs.raw or {}), _lat)
    audit["latent_checks"] = [{"rule": c.rule, "ok": c.ok, "detail": c.detail} for c in _lc]
    # Only the blocking subset stops the case.
    _bad = contradictions(dict(cs.raw or {}), _lat)
    _block = blocking_contradictions(dict(cs.raw or {}), _lat)
    if _bad:
        audit["latent_contradictions"] = [c.rule for c in _bad]
        audit["latent_blocking"] = [c.rule for c in _block]
        for c in _bad:
            log.error("[latent] %s %s: %s", cs.case_id, c.rule, c.detail)
    if _block:
        # Blocking contradictions stop the case before any model call.
        log.error("[build] %s latent contradiction -> not emitted (stopped before generation, no model call spent) %s",
                  cs.case_id, [c.rule for c in _block])
        return None, audit
    try:
        p = make_premise("human", premise_spec(cs, T))          # (1) premise + validation
    except ValueError as e:
        audit["premise_error"] = str(e)
        log.error("[build] %s premise validation failed: %s", cs.case_id, e)
        return None, audit
    audit["premise_ok"] = True
    # A plugin task's case (external latent) has its events written by code: no event model.
    _plugin_task = bool((p.meta or {}).get(_EG.SLOT))
    if _plugin_task:
        event_dispatch = None
        audit["event_generator"] = "deterministic"
    audit["world_layer_gaps"] = world_layer_gaps()

    # ---- (1b) Declaration-only gates, run before any model call ----
    _decl = (gates.check_drug_indication(p)  # GEN27: does the drug have an indication for this disease
             + gates.check_comorbidity_vocab(p)  # GEN28: does the comorbidity name have a physiological consumer
             + gates.check_rhythm_gap_feasible(cs)  # GEN29: the gap tier's declaration must actually fit
             + gates.check_dosing_consistency(p)  # GEN20b: dosing frequency and route are self-consistent
             + gates.check_premise_registry(p.dumps())  # meta-rule: premise fields must be registered
             + gates.check_demographic_plausibility(None, cs)  # GEN24: the declared age range is plausible for the condition
             # GEN24b: `known_conditions` must not give away a gold line
             + gates.check_gold_line_in_known_conditions(None, cs))
    audit["declaration_gates"] = [f"{h['kind']}:{h['detail']}" for h in _decl]
    _decl_block = gates.batch_gate_blockers(_decl)
    for h in _decl:
        if h not in _decl_block:
            log.warning("[gates] %s %s: %s", cs.case_id, h["kind"], h["detail"])
    if _decl_block:
        audit["declaration_blocking"] = [h["kind"] for h in _decl_block]
        log.error("[build] %s declaration gate failed -> not emitted (stopped before generation, no model call spent) %s",
                  cs.case_id, [h["kind"] for h in _decl_block])
        return None, audit

    gen = LLMCaseGenerator(dispatch, cs) if dispatch else HaenvGenerator()
    raw, rep = synthesize(p, original_case=original_facts(cs, T), generator=gen,  # (2) generate + GV-1
                          max_rounds=max_rounds, apply_noise=False)
    audit["synth_rounds"] = rep.get("rounds")
    audit["synth_history"] = rep.get("history")
    if raw is None:                                             # failed to converge -> not emitted
        audit["last_conflicts"] = [c["kind"] for c in rep.get("last_conflicts", [])]
        audit["last_conflicts_detail"] = [f"{c['kind']}:{c.get('detail')}"
                                          for c in rep.get("last_conflicts", []) if c.get("detail")]
        log.error("[build] %s GV-1 did not converge -> not emitted %s", cs.case_id,
                  audit.get("last_conflicts_detail") or audit["last_conflicts"])
        return None, audit

    applied: list[str] = []                                     # (3) inject, derived from the premise
    # Noise amplitudes come from `registry/artifact_rates.yaml` (no fallback),
    # checked against the kernel's defaults. Reading artifacts wait for (3c).
    _assert_kernel_params_in_sync()
    for nz in cs.noise:
        if nz["class"] in NOISE_BEFORE_PHYSIO:
            raw = _inject_kernel_noise(raw, nz)
            applied.append(f"{nz['class']}@wk{nz['week']}")
    if cs.distractor_level != "none":
        # `seed_shift` rotates the kernel's distractor pool; without it the `low` tier
        # always draws index 0.
        from . import rng as _rng
        _shift = int(_rng.unit(cs.case_id, "distractor", "shift") * 10)
        raw = inject_distractors(raw, p, level=cs.distractor_level, seed_shift=_shift)
        applied.append(f"distractor:{cs.distractor_level}@shift{_shift}")
    raw.case_id = cs.case_id                                    # the injector mutated id, restore it
    raw.latent_premise = p.dumps()                              # verifier-only bookkeeping
    raw.latent_premise[WORLD_MEDICATION_KEY] = world_medication_record(p)
    audit["noise_applied"] = applied
    audit["weight_skeleton"] = _SKELETON_AUDIT.pop(str(cs.case_id), None)
    audit["ledger_clinical_realigned"] = _CLINICAL_LEDGER_AUDIT.pop(str(cs.case_id), None)

    raw, vrep = _inject_events_verified(raw, cs, p, T, max_rounds,   # (3b) day-by-day events + per-item validation
                                       dispatch=event_dispatch)
    # (3c) Observation layer, after the ground-truth trajectory is captured: the
    # kernel's reading artifacts, then post-kernel dirty data (e.g. carried-forward
    # values). Both are judged as observations, never as physiological slope.
    # `clean_weight` is the truth series every clinic-scale reference is sampled from;
    # with the physiology layer off, the course before the reading artifacts.
    _truth_w = ((vrep.get("physio_clean_ld") or {}).get("weight")
                or copy.deepcopy(raw.longitudinal_data.get("weight")))
    # The label rule read on the truth series, for the record (the gold is set elsewhere).
    _tl, _td = gates.derive_outcome({"weight": _truth_w or []}, raw.label_rule)
    audit["weight_truth_label"] = {"verdict": _tl, "sustained_days": _td.get("sustained_days")}
    raw, _unplaced = _inject_observation_artifacts(raw, cs, applied)
    _post = _post_inject.apply_post_injection(raw, case_id=cs.case_id, clean_weight=_truth_w)
    applied += _post
    audit["post_injected"] = _post
    # (3c') Measurement layer of the lab-calendar streams (visits, per-draw variation,
    # repeats, printed precision); the truth series goes to step (4)'s slope checks.
    _lab_truth, _lab_audit = observe_labs(raw, cs.case_id)
    _LAB_CTX.pop(str(cs.case_id), None)
    if str(cs.case_id) in events_mod.PANEL_ALIGN_PENDING:
        # the panel's TC and the declared screening items follow the observed lab draws
        from .findings_render import align_panel_to_streams
        _al = align_panel_to_streams(raw, normal_before_day=events_mod.PANEL_ALIGN_PENDING.pop(str(cs.case_id)))
        if _lab_audit:
            _lab_audit["panel_realigned"] = len(_al)
    if _lab_audit:
        audit["lab_observation"] = _lab_audit
        # verifier-only: the noise-free lab truth the observation was drawn from
        audit["lab_truth"] = {k: [[int(q["ts"]), q["value"]] for q in v] for k, v in _lab_truth.items()}
    # Ledger values re-read from the final question-face streams.
    audit["ledger_realigned"], audit["ledger_removed"] = realign_ledger_to_streams(
        raw.evidence_ledger, raw.longitudinal_data)

    audit.update(event_rounds=vrep["rounds"], event_streams=vrep["n_streams"],
                 event_evidence=vrep["n_events"], event_dropped=vrep["dropped"],
                 verify_ok=vrep["ok"], verify_bad=vrep["bad"])
    audit["_verify_report"] = vrep["report"]
    # Coupling rules fired, and those still pending physician review, as of generation.
    _cpl = (vrep.get("injected") or {}).get("coupling") or {}
    audit["coupling_fired"] = list(_cpl.get("fired_rules") or [])
    audit["coupling_pending_review"] = list(_cpl.get("pending_review") or [])
    if not vrep["ok"]:                                          # not emitted
        log.error("[build] %s daily-event verification did not converge -> not emitted bad=%s", cs.case_id, vrep["bad"])
        return None, audit
    # (3b') Register the Q-side injection ledger (wq law 2) once, after convergence. Diagnosis
    # cases also record the answer contract their framing asks for (question side, not gold).
    _qinj = vrep.get("injected") or {}
    if cs.ddx:
        from .framings import DDX_ANSWER_CONTRACT as _DAC
        _qinj = {**_qinj, "answer_contract": copy.deepcopy(_DAC)}
    wq.register_injection(cs.case_id, _qinj)
    # (3d) EV id anonymization, after registration and before `build_instance`.
    audit["ev_id_map"] = anonymize_evidence_ids(raw, vrep.get("report"))
    # (3d') A plugin task's case enters every symptom sentence in one form, after verification traced the
    # templates (`events.enter_symptoms`).
    if _plugin_task:
        audit["symptom_source"] = events_mod.enter_symptoms(raw.evidence_ledger, (raw.user_profile or {}).get("sex"))

    # (3e) Realign the weight ledger to the final (noised) stream; the change is
    # recorded in the audit.
    _lw_ts, _lw_val = ledger_weight_point(raw.longitudinal_data, T, cs.case_id)
    _lw_moved = []
    for _ev in (raw.evidence_ledger or []):
        if gates.LEDGER_VALUE_STREAMS.get(str(_ev.get("source_type") or "")) != {"weight"}:
            continue
        _old = _ev.get("measured_value")
        if not isinstance(_old, (int, float)) or isinstance(_old, bool):
            continue
        if abs(float(_old) - _lw_val) > 1e-9 or int(_ev.get("source_timestamp", -1)) != _lw_ts:
            _lw_moved.append({"evidence_id": str(_ev.get("evidence_id")),
                              "from": [_ev.get("source_timestamp"), float(_old)],
                              "to": [_lw_ts, _lw_val]})
        _ev["source_timestamp"], _ev["measured_value"] = _lw_ts, _lw_val
    audit["ledger_weight_realigned"] = _lw_moved

    # ---- (4) Recheck + leakage gate ------------------------------------------------------
    #
    # Slope and anchors are judged on the ground-truth trajectory, value range on
    # the observed sequence. At full descent pace the kernel's weekly-slope limit
    # (1.5 x 1.05 / 7 kg/day) leaves about 11 g/day for observation noise, so it
    # cannot apply to the physiology layer's observed stream.
    _clean_ld = vrep.get("physio_clean_ld")
    if _lab_truth:
        # Lab slope is judged on the noise-free lab truth, as weight is on its truth: the
        # kernel's `max_weekly_delta` bounds the physiological rate, not measurement error.
        _clean_ld = {**(_clean_ld if _clean_ld is not None
                        else copy.deepcopy(raw.longitudinal_data)), **_lab_truth}
    if _clean_ld is None:
        conflicts = premise_conflicts(raw, p)
    else:
        _truth = copy.copy(raw)
        _truth.longitudinal_data = _clean_ld
        conflicts = premise_conflicts(_truth, p)     # slope / value range on the ground-truth trajectory
        # value range also on the observed sequence
        conflicts += gates.check_observed_in_domain(raw, p)
        # what the solver can see is judged on the observed series
        conflicts = ([c for c in conflicts if c["kind"] not in OBSERVED_CONFLICT_KINDS]
                     + [c for c in premise_conflicts(raw, p) if c["kind"] in OBSERVED_CONFLICT_KINDS])
    sp_probe, _ = build_instance(raw, T)                        # the copy the solver sees
    hits = (gates.check_cadence(raw.longitudinal_data, p, cs)   # (4b) haenv-side assertions (GEN6/7/13/14)
            + gates.check_device_inventory(raw.user_profile, raw.longitudinal_data)
            + gates.check_clinical_coupling(raw.longitudinal_data,     # GEN15 coupling direction
                                            _cplan := clinical_plan(
                                                p.patient_basics.get('disease', ''),
                                                p.device_signals.get('devices', []),
                                                p.patient_basics.get('comorbidities') or [],
                                                cs.case_id),
                                            # expected direction includes the drug effect
                                            drug_terms=_drug_terms_for(raw, p, _cplan))
            # GEN13: outcome derivable by label_rule, judged on the ground-truth trajectory
            + gates.check_outcome_derivable(
                _truth if _clean_ld is not None else raw, cs)
            + gates.check_gold_coverage(sp_probe, raw.gold_drivers,  # GEN14: the gold standard must be derivable from the question surface
                                        (raw.adjudication or {}).get("ddx"))
            + gates.check_symptom_day_separability(raw, T)  # GEN18: timestamps must not let signal/noise be separated at a glance
            + gates.check_ev_id_opaque(sp_probe)                # GEN21: ids must not carry a category prefix
            + gates.check_sex_consistency(raw, cs)  # GEN19: the question surface must not contradict the declared sex
            # GEN22: declared anchors honored, judged on the ground-truth trajectory
            + gates.check_anchors_honored(_truth if _clean_ld is not None else raw, cs)
            + gates.check_stream_horizons(raw)                  # GEN23: stream endpoints must not exceed the declared course end day
            + gates.check_trap_evidence(raw)                    # GEN27: a single-reading trap must stay visible on the observed series
            + gates.check_ledger_values_traceable(raw, T)       # GEN25: ledger values must be traceable to the same-named stream at ≤T
            + gates.check_clinical_baseline_cohort(raw, p)      # GEN26: the baseline must not portray an undiagnosed person as sick
            + gates.check_event_density(raw, p, T)              # GEN20a: however much noise is declared, that much must be injected
            + gates.check_missingness(raw.longitudinal_data, p, cs, T)   # GEN20c: missingness must be answer-neutral
            + _EG.gates_for(raw, cs, sp_probe, T))                     # external task-type gates; [] when none
    conflicts += [h for h in hits if h["severity"] == "gate"]  # gate-severity hits merge into the emission gate
    # A declared reading artifact that found no reading to move (e.g. inside a window
    # `mnar_missing` emptied) would leave the case without the artifact it declares.
    conflicts += [{"kind": "noise_not_placed", "severity": "gate",
                   "detail": f"{x}: no reading in its window"} for x in _unplaced]
    # A trap is only fair if the solver is shown its evidence.
    conflicts += trap_visibility_conflicts(raw, last_shown_day(raw, cs.geometry))
    warns = [h for h in hits if h["severity"] != "gate"]
    for w in warns:
        log.warning("[gates] %s %s: %s", cs.case_id, w["kind"], w["detail"])
    sp, _ = build_instance(raw, T)
    leak_ok, leak_viol = leakage_probe(sp, T)
    # The item verifier scanned before the final observation and payload steps.
    # Check the exact payload handed to the solver after those steps as well.
    _final_text = verify_mod.scan_solver_text(
        raw, T, ddx_aliases=verify_mod._ddx_aliases(cs), solver_payload=sp)
    leak_viol = list(leak_viol) + [f for f in _final_text["findings"]
                                    if not f.startswith("kernel_leakage_probe:")]
    leak_ok = leak_ok and _final_text["ok"]
    audit.update(post_noise_conflicts=[c["kind"] for c in conflicts],
                 post_noise_conflict_details=[
                     f"{c['kind']}:{c['detail']}" for c in conflicts if c.get("detail")],
                 gate_warnings=[f"{w['kind']}:{w['detail']}" for w in warns],
                 leak_ok=leak_ok, leak_violations=leak_viol,
                 final_text_scan={"ok": _final_text["ok"],
                                  "findings": _final_text["findings"],
                                  "n_chars": _final_text["n_chars"]},
                 solver_visible_signals=len(sp.longitudinal_data),
                 solver_visible_evidence=len(sp.evidence_ledger))
    if conflicts or not leak_ok:
        log.error("[build] %s blocked by the emission gate: conflicts=%s leak=%s\n          %s",
                  cs.case_id, audit["post_noise_conflicts"], leak_viol,
                  "\n          ".join(audit["post_noise_conflict_details"]) or "(kernel conflict carries no detail)")
        return None, audit

    # Gold qualifiers: split core + qualifier and record, on what the solver sees at T, whether
    # each qualifier can be established (after every gate, so emission is unaffected).
    from . import qualifiers as _qualifiers
    _qrecs = _qualifiers.annotate(raw, sp)
    if _qrecs:
        audit["gold_qualifiers"] = [{"family": q["family"], "derivable": q["derivable"]}
                                    for q in _qrecs]
    audit["emitted"] = True
    audit["outcome_label"] = raw.outcome_label
    audit["gold_drivers"] = list(raw.gold_drivers)
    log.info("[build] %s emitted ✓ (rounds=%s noise=%s)", cs.case_id, rep.get("rounds"), applied)
    return raw, audit
