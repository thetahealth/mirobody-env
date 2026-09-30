"""Iterative generate-verify loop (GV-1) and the three-premise conflict checker.

Each round generates a clean course of disease and checks it against the original case, the
patient basics and the latent premise; conflicts are fed back and the case is regenerated, up
to `max_rounds`. Noise derived from the premise is injected only after convergence, so it is
never flagged by the physical-slope check. A case that does not converge is not emitted.

SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from schema import RawCase
from latent import (LatentPremise, DISEASE_SIGNAL_DOMAIN, allowed_noise_classes, derive_noise_plan,
                    drug_pkpd)
from noise import inject as inject_noise

log = logging.getLogger("harness.synth")
HERE = Path(__file__).parent

# Non-clinical auxiliary signals, exempt from the device-inventory check. Hosts add their own
# through `register_aux_signals` (haenv registers every `aux: true` stream when mounting the kernel).
AUX_SIGNALS = {"dose_timeline", "medication_adherence", "weight_ref", "scale_qc_flag",
               "gi_symptom_score", "activity_index", "sleep_hours", "resting_hr", "diet_carb_pct",
               "steps", "hrv", "skin_temp", "stress_score", "spo2", "body_temp"}


def register_aux_signals(names) -> None:
    """Declare more auxiliary signals. Idempotent; the set only grows."""
    AUX_SIGNALS.update(str(n) for n in names)


KNOWN_OUTCOMES = {"event_occurred", "event_not_occurred"}
_ALL_DOMAIN_SIGNALS = {s for d in DISEASE_SIGNAL_DOMAIN.values() for s in d}


# ------------------------------------------------------- conflict checker (GV-1 judge)
def _noise_windows(raw: RawCase) -> list[tuple[int, int]]:
    """Collect noise windows from injection traces (used to exclude artifact points from the
    physical-delta check)."""
    wins: list[tuple[int, int]] = []
    af = raw.adjudication.get("artifact_flags")
    if af and af.get("window"):
        w = af["window"]
        wins.append((w[0] - 3, w[1] + 14))
    for rp in raw.reversal_points:
        if rp.get("noise_class"):
            d = int(rp.get("week", 0)) * 7
            wins.append((d - 3, d + 14))
    return wins


def _pkpd_conflicts(raw: RawCase, p: LatentPremise) -> list[dict]:
    """Pharmacological PK/PD consistency: the generated dose_timeline must follow the drug's dose
    ladder, with titration intervals >= the minimum, and the drug effect must not precede the
    first dose (effect-before-cause)."""
    out: list[dict] = []
    pk = drug_pkpd(p.patient_basics.get("regimen", {}).get("drug", ""))
    if not pk:
        return out
    dose = sorted(raw.longitudinal_data.get("dose_timeline", []), key=lambda x: x["ts"])
    for pt in dose:
        if pt["value"] not in pk["ladder"]:
            out.append({"kind": "dose_off_ladder", "detail": f"{pt['value']}@{pt['ts']} ∉ {pk['ladder']}"})
    for a, b in zip(dose, dose[1:]):
        if b["value"] != a["value"] and (b["ts"] - a["ts"]) < pk["min_titration_days"]:
            out.append({"kind": "titration_too_fast",
                        "detail": f"{a['value']}→{b['value']} in {b['ts']-a['ts']}d < {pk['min_titration_days']}d"})
    if dose:
        first = dose[0]["ts"]
        prim = DISEASE_SIGNAL_DOMAIN.get(p.patient_basics.get("disease"), {})
        for sig in raw.longitudinal_data:
            if sig in prim:
                pre = [x for x in raw.longitudinal_data[sig] if x["ts"] < first]
                if len(pre) >= 2 and abs(pre[-1]["value"] - pre[0]["value"]) > prim[sig]["max_weekly_delta"]:
                    out.append({"kind": "effect_before_drug_start", "detail": f"{sig} moved before day{first}"})
    return out


#: What tells a context confound from a real regain at ts <= T: an offset stretch of at least
#: this many readings, then either this many readings after the window (the fall-back) or a
#: clinic-scale reading inside the window (which does not follow the offset).
CC_MIN_WINDOW_READINGS = 3
CC_MIN_FALLBACK_READINGS = 2


def _context_confound_conflicts(raw: RawCase) -> list[dict]:
    """`window_trap_not_visible_pre_T` for a context-confound window whose evidence the solver
    (ts <= T) cannot see."""
    out: list[dict] = []
    T = (raw.prediction_context or {}).get("prediction_time_T")
    af = raw.adjudication.get("artifact_flags") or {}
    if T is None or not af:
        return out
    T = int(T)
    days = [int(q["ts"]) for q in raw.longitudinal_data.get("weight", [])]
    refs = [int(q["ts"]) for q in raw.longitudinal_data.get("weight_ref", [])]
    for w in af.get("windows") or ([af] if af.get("window") else []):
        if w.get("noise_class") != "context_confound":
            continue
        d0, d1 = int(w["window"][0]), int(w["window"][1])
        n_in = sum(d0 <= d <= min(d1, T) for d in days)
        n_fb = sum(d1 < d <= T for d in days)
        n_ref = sum(d0 <= d <= min(d1, T) for d in refs)
        if n_in < CC_MIN_WINDOW_READINGS or (n_fb < CC_MIN_FALLBACK_READINGS and not n_ref):
            out.append({"kind": "window_trap_not_visible_pre_T",
                        "detail": f"context_confound [{d0},{d1}] T={T}: {n_in} readings in window, "
                                  f"{n_fb} after it, {n_ref} clinic-scale readings in it"})
    return out


def premise_conflicts(raw: RawCase, p: LatentPremise, original_case: dict | None = None) -> list[dict]:
    """Three-premise conflict check. Returns a list of conflicts (empty = converged). Each entry
    is {kind, detail}."""
    c: list[dict] = []
    disease = p.patient_basics.get("disease")
    domain = DISEASE_SIGNAL_DOMAIN.get(disease, {})
    inv = set(p.device_signals.get("signals", {}))
    wins = _noise_windows(raw)

    def _in_noise(ts: int) -> bool:
        return any(a <= ts <= b for a, b in wins)

    for sig, pts in raw.longitudinal_data.items():
        if sig in AUX_SIGNALS:
            continue
        if sig not in inv:
            c.append({"kind": "signal_not_in_inventory", "detail": f"{sig} ∉ {sorted(inv)}"})
            continue
        dom = domain.get(sig)
        if dom is None:
            c.append({"kind": "signal_not_in_disease_domain", "detail": f"{sig}@{disease}"})
            continue
        for pt in pts:
            if not (dom["range"][0] <= pt["value"] <= dom["range"][1]):
                c.append({"kind": "value_out_of_physio_range",
                          "detail": f"{sig}={pt['value']}@{pt['ts']} ∉ {dom['range']}"})
        # Weekly slope outside noise windows must stay <= max_weekly_delta (5% tolerance).
        clean = [pt for pt in pts if not _in_noise(pt["ts"])]
        for a, b in zip(clean, clean[1:]):
            wk = max(1e-6, (b["ts"] - a["ts"]) / 7.0)
            if abs(b["value"] - a["value"]) / wk > dom["max_weekly_delta"] * 1.05:
                c.append({"kind": "weekly_delta_exceeds_physio",
                          "detail": f"{sig} {a['value']}→{b['value']} over {b['ts']-a['ts']}d "
                                    f"> {dom['max_weekly_delta']}/wk"})

    allowed = allowed_noise_classes(p)
    for rp in raw.reversal_points:
        nc = rp.get("noise_class")
        if nc and nc not in allowed:
            c.append({"kind": "noise_not_allowed_by_premise", "detail": f"{nc} ∉ {sorted(allowed)}"})
    c += _context_confound_conflicts(raw)

    adh_pts = raw.longitudinal_data.get("medication_adherence", [])
    if adh_pts:
        obs_low = min(pt["value"] for pt in adh_pts)
        prem_low = min([pt.get("level", p.adherence.get("baseline", 1.0))
                        for pt in p.adherence.get("trajectory", [])] + [p.adherence.get("baseline", 1.0)])
        if obs_low < prem_low - 0.2:
            c.append({"kind": "adherence_below_premise",
                      "detail": f"obs_min={obs_low} << premise_min={prem_low}"})

    if original_case:
        kc = " ".join(raw.user_profile.get("known_conditions", []))
        if original_case.get("disease") and original_case["disease"] not in kc and \
                original_case["disease"] not in disease:
            c.append({"kind": "disease_conflicts_original", "detail": original_case["disease"]})
        for sig, anchor in (original_case.get("anchor_values") or {}).items():
            pts = {pt["ts"]: pt["value"] for pt in raw.longitudinal_data.get(sig, [])}
            if anchor["day"] in pts and abs(pts[anchor["day"]] - anchor["value"]) > anchor.get("tol", 2.0):
                c.append({"kind": "anchor_value_conflicts_original",
                          "detail": f"{sig}@{anchor['day']}={pts[anchor['day']]} vs {anchor['value']}"})

    c.extend(_pkpd_conflicts(raw, p))

    if raw.outcome_label not in KNOWN_OUTCOMES:
        c.append({"kind": "bad_outcome_label", "detail": raw.outcome_label})
    if not raw.gold_drivers:
        c.append({"kind": "empty_gold_drivers", "detail": ""})
    return c


# --------------------------------------------------------------- generators
class TemplateGenerator:
    """Offline deterministic generator (no API calls). With `inject_flaw=True`, round 0 loses weight
    too fast so the loop's conflict -> feedback -> convergence path is exercised.
    """

    def __init__(self, inject_flaw: bool = False):
        self.inject_flaw = inject_flaw

    def generate(self, p: LatentPremise, original_case: dict | None, feedback: list[dict]) -> RawCase:
        sig = p.device_signals["signals"]["weight"]
        lo, hi = sig["plausible_range"]
        T = 84
        start = round(hi - 2, 1)
        flawed = self.inject_flaw and not feedback           # feedback present = already asked to correct
        nadir = round(lo + 2, 1) if flawed else round(lo + (hi - lo) * 0.35, 1)  # flawed = drops too fast -> exceeds the weekly-slope limit

        weight = []
        for d in range(0, T + 1, 7):                         # day 0..84: weight-loss phase
            weight.append({"ts": d, "value": round(start + (nadir - start) * d / T, 2)})
        for d in range(T + 7, 366, 7):                       # rebound phase (+0.35/wk)
            weight.append({"ts": d, "value": round(nadir + 0.05 * (d - T), 2)})

        adh_traj = p.adherence.get("trajectory", [])
        adh_pts = [{"ts": pt["day"], "value": pt["level"]} for pt in adh_traj] or \
                  [{"ts": 84, "value": p.adherence.get("baseline", 0.9)}]
        adh_low = min(pt["value"] for pt in adh_pts)
        driver = "poor_medication_adherence" if adh_low < 0.8 else "biological_low_response"

        dose_steps = p.patient_basics.get("regimen", {}).get("dose_steps", [2.5, 5.0, 7.5])
        dose_tl = [{"ts": i * 28, "value": v} for i, v in enumerate(dose_steps)]

        return RawCase(
            case_id="SYN-TMP",
            user_profile={"age_range": p.patient_basics.get("age_range", "45-49"),
                          "sex": p.patient_basics.get("sex", "F"),
                          "known_conditions": [p.patient_basics.get("disease", "obesity")],
                          "treatment_goals": p.patient_basics.get("goals", []),
                          "device_inventory": p.device_signals.get("devices", [])},
            prediction_context={"prediction_time_T": T,
                                "target_event_type": p.meta.get("target_event_type", "weight_regain"),
                                "prediction_window": "281d", "available_history_window": f"{T}d"},
            longitudinal_data={"weight": weight, "dose_timeline": dose_tl, "medication_adherence": adh_pts},
            evidence_ledger=[{"evidence_id": "EV-S01", "source_type": "wearable", "source_timestamp": T,
                              "measured_value": nadir, "claim_supported": True, "reliability_status": "reliable"}],
            outcome_label="event_occurred",
            label_rule={"minimum_change_magnitude": "regain >= max(5% of lost weight, 0.75 kg), read on the 7-reading centred rolling median; with less than 0.75 kg lost it is a gain of 0.75 kg above the low point",
                        "min_change_frac": 0.05, "min_change_kg": 0.75, "smoothing": "rolling_median_7pt",
                        "minimum_persistence": "sustained >=8w", "baseline_window": "nadir"},
            gold_drivers=[driver],
            adjudication={"adjudication_protocol_present": True, "primary_driver": driver,
                          "red_flag_present": False, "clinician_action_warranted": True},
            reversal_points=[{"week": (T + 56) // 7, "type": "real", "flip": "risk_low->elevated",
                              "trigger": "nadir 后反弹,证伪'能维持'"}],
        )


_CASE_PROMPT = """你是"合成病程"生成器。**只依据下面的隐变量前提**(+原始病例,如有)生成一段纵向病程观测,\
仅输出一个 JSON 对象(无多余文字/markdown)。所有信号名/单位/取值必须落在前提声明的设备信号与疾病可行域内,\
相邻周变化不得超过生理上限;dose_timeline 沿该药剂量梯级、滴定间隔达标。

隐变量前提(JSON): {premise}
原始病例(可空): {original}
需修正的上轮冲突(按 kind): {feedback}

只输出 JSON(结构字段我会据前提补齐,你只出下列生成字段):
{{"longitudinal_data":{{"<信号>":[{{"ts":0,"value":..}},...],"dose_timeline":[{{"ts":0,"value":..}}],"medication_adherence":[{{"ts":..,"value":0-1}}]}},
 "evidence_ledger":[{{"evidence_id":"EV-..","source_type":"..","source_timestamp":<=T,"measured_value":..,"claim_supported":true,"reliability_status":"reliable"}}],
 "outcome_label":"event_occurred|event_not_occurred","label_rule":{{"minimum_change_magnitude":"..","minimum_persistence":"..","baseline_window":".."}},
 "gold_drivers":["<主因>"],"adjudication":{{"adjudication_protocol_present":true,"primary_driver":"..","red_flag_present":false,"clinician_action_warranted":true}},
 "reversal_points":[{{"week":..,"type":"real","flip":"risk_low->elevated","trigger":".."}}]}}"""


class LLMGenerator:
    """LLM generator conditioned on premise, original case and prior-round conflicts. Structural
    fields come from the premise; `dispatch` can be replaced by a fake for offline tests.
    """

    def __init__(self, dispatch=None, family: str = "gemini", tier: str = "pro"):
        self.dispatch, self.family, self.tier = dispatch, family, tier

    def _call(self, prompt: str) -> str:
        if self.dispatch is not None:
            return self.dispatch(prompt)
        from latent import _ai_dispatch
        return _ai_dispatch(prompt, self.family, self.tier)

    def generate(self, p: LatentPremise, original_case: dict | None, feedback: list[dict]) -> RawCase:
        from latent import _extract_json
        prompt = _CASE_PROMPT.format(
            premise=json.dumps(p.dumps(), ensure_ascii=False),
            original=json.dumps(original_case or {}, ensure_ascii=False),
            feedback=json.dumps(sorted({x.get("kind") for x in feedback}), ensure_ascii=False))
        data = _extract_json(self._call(prompt))
        T = int(p.meta.get("prediction_time_T", 84))
        return RawCase(
            case_id=p.meta.get("case_id", "SYN-LLM"),
            user_profile={"age_range": p.patient_basics.get("age_range", "45-49"),
                          "sex": p.patient_basics.get("sex", "F"),
                          "known_conditions": [p.patient_basics.get("disease", "obesity")],
                          "treatment_goals": p.patient_basics.get("goals", []),
                          "device_inventory": p.device_signals.get("devices", [])},
            prediction_context={"prediction_time_T": T,
                                "target_event_type": p.meta.get("target_event_type", "weight_regain"),
                                "prediction_window": "281d", "available_history_window": f"{T}d"},
            longitudinal_data=data.get("longitudinal_data", {}),
            evidence_ledger=data.get("evidence_ledger", []),
            outcome_label=data.get("outcome_label", "event_occurred"),
            label_rule=data.get("label_rule", {}),
            gold_drivers=data.get("gold_drivers", []),
            adjudication=data.get("adjudication", {"adjudication_protocol_present": True}),
            reversal_points=data.get("reversal_points", []),
        )


# --------------------------------------------------------------- the loop
def _apply_premise_noise(raw: RawCase, p: LatentPremise) -> tuple[RawCase, list[str]]:
    """After convergence, noise is derived from the premise and injected. Only the first class in
    the plan is injected, in the post-T window."""
    plan = derive_noise_plan(p)
    if not plan:
        return raw, []
    nc = plan[0]["noise_class"]
    T = int(raw.prediction_context["prediction_time_T"])
    noisy = inject_noise(raw, nc, T + 42, T + 56)   # in the post-T course of disease
    return noisy, [nc]


def synthesize(p: LatentPremise, original_case: dict | None = None, generator=None,
               max_rounds: int = 6, apply_noise: bool = True) -> tuple[RawCase | None, dict]:
    """Latent-premise GV-1 iterative generate-verify loop. Returns (RawCase|None, report). Not
    converged -> None (not emitted)."""
    generator = generator or TemplateGenerator()
    feedback: list[dict] = []
    history: list[dict] = []
    for r in range(1, max_rounds + 1):
        clean = generator.generate(p, original_case, feedback)
        conflicts = premise_conflicts(clean, p, original_case)
        history.append({"round": r, "n_conflicts": len(conflicts),
                        "kinds": sorted({x["kind"] for x in conflicts})})
        log.info("[synth] round %d: %d conflicts %s", r, len(conflicts),
                 sorted({x["kind"] for x in conflicts}))
        if not conflicts:
            raw, applied = (_apply_premise_noise(clean, p) if apply_noise else (clean, []))
            raw.latent_premise = p.dumps()               # verifier-only bookkeeping
            raw.case_id = p.meta.get("case_id", "SYN-01")
            report = {"converged": True, "rounds": r, "history": history,
                      "noise_applied": applied, "final_case_id": raw.case_id}
            return raw, report
        feedback = conflicts                              # conflict summary becomes feedback -> regenerate
    return None, {"converged": False, "rounds": max_rounds, "history": history,
                  "last_conflicts": conflicts}


# --------------------------------------------------------------- persistence
def _slug(cid: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", cid).strip("_").lower() or "case"
    return s if s[0].isalpha() else "c" + s          # module name must start with a letter (not
                                                      # a leading "_") to be registered by cases


def persist_case(raw: RawCase, task_type: str = "hardprob", cases_dir: str | None = None) -> Path:
    """Write a converged synthetic case as cases/<task_type>/<slug>.py (exposing raw()),
    automatically registered into cases.ALL/BY_TYPE. Returns the write path."""
    base = Path(cases_dir) if cases_dir else HERE / "cases"
    d = base / task_type
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_slug(raw.case_id)}.py"
    blob = json.dumps(asdict(raw), ensure_ascii=False)
    module = ('"""AUTO-GENERATED synthesized case (%s) — SYNTHETIC, eval only.\n'
              '由 synth.persist_case 写入;含 latent_premise(verifier-only)。"""\n'
              'from schema import RawCase\n'
              'import json\n\n'
              '_JSON = %r\n\n\n'
              'def raw() -> RawCase:\n'
              '    return RawCase(**json.loads(_JSON))\n') % (raw.case_id, blob)
    path.write_text(module, encoding="utf-8")
    log.info("[synth] persisted %s -> %s", raw.case_id, path)
    return path
