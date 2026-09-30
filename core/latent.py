"""latent.py -- latent-premise generation and validation (synthesis step 0).

A case is built from an explicit latent premise with four parts:
`patient_basics` (disease, comorbidities, regimen, goals), `event_density`,
`device_signals` (devices and signal sampling/range) and `adherence` (level,
trajectory, missingness mechanism). `make_premise("human", spec)` or
`make_premise("llm", seed)` builds one; it must pass `validate_premise` before
generation, and all noise is derived from it (`derive_noise_plan`).

The premise is verifier-only: it is stored in `RawCase.latent_premise` and
never enters `SolverPayload`.

SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

log = logging.getLogger("harness.latent")


# Disease -> the physiological feasible domain for that disease's common signals (unit,
# plausible range, max physiological change per week). Synthesized observations must fall
# inside it.
_WEIGHT = {"unit": "kg", "range": [40.0, 250.0], "max_weekly_delta": 1.5}   # weight is shared across metabolic diseases
_DISEASE_SIGNAL_DOMAIN_BASE: dict[str, dict[str, dict]] = {
    "obesity": {
        "weight": _WEIGHT,
        "HbA1c": {"unit": "%", "range": [4.5, 14.0], "max_weekly_delta": 0.2},
    },
    "T2D": {
        "HbA1c": {"unit": "%", "range": [4.5, 15.0], "max_weekly_delta": 0.2},
        "fasting_glucose": {"unit": "mmol/L", "range": [3.0, 25.0], "max_weekly_delta": 2.0},
        "CGM_TIR": {"unit": "%", "range": [0.0, 100.0], "max_weekly_delta": 15.0},
        "weight": _WEIGHT,
    },
    "hypertension": {
        "systolic_bp": {"unit": "mmHg", "range": [80.0, 220.0], "max_weekly_delta": 8.0},
        "diastolic_bp": {"unit": "mmHg", "range": [50.0, 130.0], "max_weekly_delta": 5.0},
        "weight": _WEIGHT,
    },
    "dyslipidemia": {
        "LDL": {"unit": "mmol/L", "range": [1.0, 8.0], "max_weekly_delta": 0.3},
        "triglycerides": {"unit": "mmol/L", "range": [0.5, 12.0], "max_weekly_delta": 0.5},
        "weight": _WEIGHT,
    },
    "MASLD": {   # metabolic dysfunction-associated steatotic liver disease (formerly NAFLD)
        "ALT": {"unit": "U/L", "range": [5.0, 300.0], "max_weekly_delta": 12.0},
        "AST": {"unit": "U/L", "range": [5.0, 300.0], "max_weekly_delta": 12.0},
        "FIB4": {"unit": "index", "range": [0.3, 6.0], "max_weekly_delta": 0.2},
        "weight": _WEIGHT,
    },
}


class _ScopedDomain(dict):
    """Signal domain per disease, with thread-local additions.

    `scoped(disease, extra)` adds signals for the current thread only (cases are
    built on a thread pool); `get` and `[]` return the base domain merged with
    them, base entries taking precedence. The base table is never mutated.
    """

    _local = threading.local()

    def _extra(self, key) -> dict:
        merged: dict = {}
        for k, ex in getattr(self._local, "stack", ()):
            if k == key:
                merged.update(ex)
        return merged

    def get(self, key, default=None):
        base = dict.get(self, key)
        ex = self._extra(key)
        if not ex:
            return base if base is not None else default
        return {**ex, **(base or {})}

    def __getitem__(self, key):
        out = self.get(key, _MISSING)
        if out is _MISSING:
            raise KeyError(key)
        return out

    @contextmanager
    def scoped(self, key, extra: dict):
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        stack.append((key, dict(extra)))
        try:
            yield
        finally:
            stack.pop()


_MISSING = object()
DISEASE_SIGNAL_DOMAIN = _ScopedDomain(_DISEASE_SIGNAL_DOMAIN_BASE)
# Drug PK/PD consistency table (matched by drug-name prefix): dose ladder + minimum titration
# interval + onset lag (an effect preceding the dose is impossible).
GLP1_DOSE_STEPS = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
DRUG_PKPD: dict[str, dict] = {
    "tirzepatide": {"ladder": GLP1_DOSE_STEPS, "min_titration_days": 28, "effect_onset_days": 7},
    "semaglutide": {"ladder": (0.25, 0.5, 1.0, 1.7, 2.4), "min_titration_days": 28, "effect_onset_days": 7},
    "glp1":        {"ladder": GLP1_DOSE_STEPS, "min_titration_days": 28, "effect_onset_days": 7},
    "metformin":   {"ladder": (500.0, 1000.0, 1500.0, 2000.0), "min_titration_days": 7, "effect_onset_days": 14},
}
KNOWN_MISSINGNESS = ("MCAR", "MAR", "MNAR")
KNOWN_DEVICES = ("smart_scale", "clinic_scale", "wearable", "cgm", "bp_cuff", "lab_panel")


def drug_pkpd(drug: str) -> dict | None:
    """Match a drug's PK/PD spec by prefix (used by validate_premise for the dose ladder and by
    premise_conflicts for the titration interval)."""
    for pre, spec in DRUG_PKPD.items():
        if drug and drug.startswith(pre):
            return spec
    return None


@dataclass
class LatentPremise:
    patient_basics: dict            # {disease, comorbidities:[...], age_range, sex, regimen:{drug,dose_steps},goals:[...]}
    event_density: dict             # {measure_per_week, dosing_per_week, symptom_rate, life_event_rate}
    device_signals: dict            # {devices:[...], signals:{name:{unit,sampling_days,plausible_range}}, n_signals}
    adherence: dict                 # {baseline:0..1, trajectory:[{day,level}], missingness_mechanism}
    source: str = "human"           # "human" | "llm"
    meta: dict = field(default_factory=dict)   # {difficulty_class, target_event_type, notes}

    def dumps(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------- validation
def validate_premise(p: LatentPremise) -> tuple[bool, list[str]]:
    """Pre-emission plausibility check (mandatory for both human and LLM premises). Returns
    (ok, violations)."""
    v: list[str] = []
    pb, ds, adh = p.patient_basics, p.device_signals, p.adherence

    disease = pb.get("disease")
    if disease not in DISEASE_SIGNAL_DOMAIN:
        v.append(f"unknown_disease:{disease}")

    devices = ds.get("devices", [])
    for d in devices:
        if d not in KNOWN_DEVICES:
            v.append(f"unknown_device:{d}")
    signals = ds.get("signals", {})
    if ds.get("n_signals") not in (None, len(signals)):
        v.append(f"n_signals_mismatch:{ds.get('n_signals')}!={len(signals)}")
    domain = DISEASE_SIGNAL_DOMAIN.get(disease, {})
    for name, spec in signals.items():
        dom = domain.get(name)
        if dom is None:
            v.append(f"signal_not_in_disease_domain:{name}")
            continue
        if spec.get("unit") and spec["unit"] != dom["unit"]:
            v.append(f"unit_mismatch:{name}:{spec.get('unit')}!={dom['unit']}")
        rng = spec.get("plausible_range")
        if rng and (rng[0] < dom["range"][0] or rng[1] > dom["range"][1]):
            v.append(f"range_out_of_domain:{name}:{rng}~{dom['range']}")
        if spec.get("sampling_days", 1) <= 0:
            v.append(f"bad_sampling_days:{name}")

    # Pharmacology: dose steps must fall on that drug's PK/PD ladder (matched by drug prefix,
    # not GLP-1 only)
    regimen = pb.get("regimen", {})
    pk = drug_pkpd(regimen.get("drug", ""))
    if pk:
        for step in regimen.get("dose_steps", []):
            if step not in pk["ladder"]:
                v.append(f"dose_off_ladder:{regimen.get('drug')}:{step}")

    base = adh.get("baseline")
    if base is None or not (0.0 <= base <= 1.0):
        v.append(f"adherence_baseline_out_of_range:{base}")
    for pt in adh.get("trajectory", []):
        if not (0.0 <= pt.get("level", -1) <= 1.0):
            v.append(f"adherence_traj_out_of_range:{pt}")
    if adh.get("missingness_mechanism") not in KNOWN_MISSINGNESS:
        v.append(f"unknown_missingness:{adh.get('missingness_mechanism')}")

    # Event density must be plausible (non-negative, measurement frequency consistent with
    # sampling)
    ed = p.event_density
    for k in ("measure_per_week", "dosing_per_week"):
        if ed.get(k, 0) < 0:
            v.append(f"negative_density:{k}")
    return (not v), v


# --------------------------------------------------------------- construction
def make_premise(source: str, spec: dict | None = None, seed: dict | None = None,
                 dispatch=None) -> LatentPremise:
    """Produce and validate a latent premise; raises `ValueError` if it fails validation.

    `source="human"` builds it from `spec`. `source="llm"` samples it from `seed`:
    `dispatch=None` uses the offline `_llm_stub`, `"real"` calls a model
    (`sample_premise_llm`), and a callable is used as the dispatcher (for tests).
    """
    if source == "human":
        if not spec:
            raise ValueError("make_premise(human) needs spec")
        p = LatentPremise(patient_basics=spec["patient_basics"], event_density=spec["event_density"],
                          device_signals=spec["device_signals"], adherence=spec["adherence"],
                          source="human", meta=spec.get("meta", {}))
    elif source == "llm":
        if dispatch is not None:
            return sample_premise_llm(seed or {},
                                      dispatch=(_ai_dispatch if dispatch == "real" else dispatch))
        p = _llm_stub(seed or {})
    else:
        raise ValueError(f"source must be 'human'|'llm', got {source!r}")

    ok, viol = validate_premise(p)
    if not ok:
        raise ValueError(f"premise failed validation: {viol}")
    return p


# --------------------------------------------------------------- LLM sampling (real path)
def _extract_json(text: str) -> dict:
    """Pull a JSON object out of LLM output that may include a banner/markdown fencing (same
    convention as solver._extract_json)."""
    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    cand = m[-1] if m else None
    if cand is None:
        i, j = (text or "").find("{"), (text or "").rfind("}")
        cand = text[i:j + 1] if (i != -1 and j > i) else "{}"
    return json.loads(cand)


def _ai_dispatch(prompt: str, family: str = "gemini", tier: str = "pro", timeout: int = 180) -> str:
    """Dispatch to a real model via a local `ai <family> <tier>` CLI wrapper (same dispatch path
    as solver.LLMSolver)."""
    ai = os.path.expanduser("~/.local/bin/ai")
    proc = subprocess.run([ai, family, tier], input=prompt, capture_output=True, text=True, timeout=timeout)
    return proc.stdout


_PREMISE_PROMPT = """你是"合成病例隐变量前提"设计器。请在**合理生理/药理分布内**采样一组自洽的隐变量前提,\
仅输出一个 JSON 对象(无多余文字/markdown)。

已知可用疾病: {diseases}
已知可用设备: {devices}
采样意图(seed): {seed}
{feedback}

JSON schema(四维前提):
{{"patient_basics":{{"disease":"<上面之一>","comorbidities":[],"age_range":"45-49","sex":"F",
  "regimen":{{"drug":"tirzepatide|semaglutide|metformin","dose_steps":[...沿该药梯级...]}},"goals":[...]}},
 "event_density":{{"measure_per_week":7,"dosing_per_week":1,"symptom_rate":0.1,"life_event_rate":0.05}},
 "device_signals":{{"devices":[...],"signals":{{"<信号名>":{{"unit":"...","sampling_days":1,"plausible_range":[lo,hi]}}}},"n_signals":<len(signals)>}},
 "adherence":{{"baseline":0.0-1.0,"trajectory":[{{"day":0,"level":0.95}}],"missingness_mechanism":"MCAR|MAR|MNAR"}},
 "meta":{{"difficulty_class":"B","target_event_type":"weight_regain","case_id":"SYN-..."}}}}
约束: 信号名与单位/范围必须落在该疾病可行域;dose_steps 沿该药剂量梯级;n_signals=signals 条数。"""


def sample_premise_llm(seed: dict, dispatch=None, retries: int = 3) -> LatentPremise:
    """Sample a latent premise with an LLM: generate -> parse -> validate_premise,
    feeding violations back on failure. Raises `ValueError` if every attempt fails.
    """
    dispatch = dispatch or _ai_dispatch
    feedback = ""
    last = "?"
    for attempt in range(1, retries + 1):
        prompt = _PREMISE_PROMPT.format(diseases=list(DISEASE_SIGNAL_DOMAIN),
                                        devices=list(KNOWN_DEVICES),
                                        seed=json.dumps(seed, ensure_ascii=False), feedback=feedback)
        try:
            data = _extract_json(dispatch(prompt))
            p = LatentPremise(patient_basics=data["patient_basics"], event_density=data["event_density"],
                              device_signals=data["device_signals"], adherence=data["adherence"],
                              source="llm", meta=data.get("meta", dict(seed)))
        except Exception as e:                      # parse error / missing field
            last = f"parse_error:{e}"
            feedback = f"\n上次输出无法解析({e});必须输出含 patient_basics/event_density/device_signals/adherence 的合法 JSON。"
            log.warning("[latent] premise sample attempt %d parse fail: %s", attempt, e)
            continue
        ok, viol = validate_premise(p)
        if ok:
            log.info("[latent] LLM premise sampled ok on attempt %d", attempt)
            return p
        last = f"validation:{viol}"
        feedback = f"\n上次前提未过合理性校验: {viol};请修正这些点后重出 JSON。"
        log.warning("[latent] premise sample attempt %d invalid: %s", attempt, viol)
    raise ValueError(f"LLM premise sampling failed after {retries} tries; last={last}")


def _llm_stub(seed: dict) -> LatentPremise:
    """Offline stand-in for LLM sampling: a fixed, self-consistent default premise."""
    disease = seed.get("disease", "obesity")
    return LatentPremise(
        patient_basics={"disease": disease, "comorbidities": seed.get("comorbidities", []),
                        "age_range": "45-49", "sex": "F",
                        "regimen": {"drug": "tirzepatide", "dose_steps": [2.5, 5.0, 7.5]},
                        "goals": ["sustained_weight_loss"]},
        event_density={"measure_per_week": 7, "dosing_per_week": 1, "symptom_rate": 0.1, "life_event_rate": 0.05},
        device_signals={"devices": ["smart_scale", "wearable"],
                        "signals": {"weight": {"unit": "kg", "sampling_days": 1, "plausible_range": [70.0, 100.0]}},
                        "n_signals": 1},
        adherence={"baseline": 0.9, "trajectory": [{"day": 0, "level": 0.95}, {"day": 180, "level": 0.85}],
                   "missingness_mechanism": "MAR"},
        source="llm", meta={"difficulty_class": seed.get("difficulty_class", "B"),
                            "target_event_type": seed.get("target_event_type", "weight_regain")})


# --------------------------------------------------------------- noise plan
#: Widest weight sampling interval (days) at which a 10-day context-confound window holds at
#: least three readings.
CONTEXT_CONFOUND_MAX_SAMPLING_DAYS = 3


def derive_noise_plan(p: LatentPremise) -> list[dict]:
    """Derive the noise the premise allows. Returns `[{noise_class, window, params}]`;
    windows are placed later by synth.
    """
    plan: list[dict] = []
    ds, adh = p.device_signals, p.adherence
    n_dev = len(ds.get("devices", []))
    if n_dev >= 2:
        plan.append({"noise_class": "device_switch", "reason": "multi-device -> 换机平移可能"})
    base = adh.get("baseline", 1.0)
    traj = adh.get("trajectory", [])
    low = min([pt.get("level", base) for pt in traj] + [base])
    if low < 0.8:
        plan.append({"noise_class": "adherence_gap", "params": {"low": round(low, 2)},
                     "reason": "依从落差 -> 无应答混杂"})
    # Sparse sampling (any signal with sampling_days > 1) -> possible MNAR gaps
    if any(s.get("sampling_days", 1) > 1 for s in ds.get("signals", {}).values()):
        plan.append({"noise_class": "mnar_missing", "reason": "稀疏采样 -> 恶化期非随机缺失"})
    # Scale-type devices present -> possible spike noise / unit mis-entry
    if any(d in ("smart_scale", "clinic_scale") for d in ds.get("devices", [])):
        plan.append({"noise_class": "transient_spike", "reason": "秤读数 -> 瞬时毛刺"})
        plan.append({"noise_class": "unit_error", "reason": "秤读数 -> lb/kg 单位错填"})
    # Home self-weighing sampled densely enough that an offset stretch and its fall-back are both
    # observable -> possible context confound (clothing / post-meal / time of day). Appended
    # last: `synth._apply_premise_noise` injects only plan[0].
    w_sig = ds.get("signals", {}).get("weight")
    if ("smart_scale" in ds.get("devices", []) and isinstance(w_sig, dict)
            and w_sig.get("sampling_days", 1) <= CONTEXT_CONFOUND_MAX_SAMPLING_DAYS):
        plan.append({"noise_class": "context_confound", "reason": "居家自称 + 密集采样 -> 称重情境混杂"})
    return plan


def allowed_noise_classes(p: LatentPremise) -> set[str]:
    return {x["noise_class"] for x in derive_noise_plan(p)}
