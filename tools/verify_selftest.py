"""verify_selftest.py -- negative and positive controls for the verifier.

Injects defects that every check must catch (negative controls) and clean inputs
that no check may flag (positive controls, `!` prefix), then prints the counts.

Run:  uv run python tools/verify_selftest.py        # from the repo root
SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from haenv import kernel_path as _kernel_path       # noqa: E402
# The kernel path resolves through `haenv.kernel_path()`, as in `haenv` itself.
if (_kp := _kernel_path()):
    sys.path.insert(0, str(_kp))

from haenv import events as E, verify as V                          # noqa: E402
from haenv.build import build_case                                  # noqa: E402
from haenv.job import load_job                                      # noqa: E402

ED = {"measure_per_week": 7}
RESULTS: list[tuple[str, str, bool, str]] = []


def expect_fail(label: str, check: str, item: dict) -> None:
    """Negative control: `item`'s `check` entry must fail."""
    c = item["checks"].get(check)
    caught = bool(c) and not c["ok"]
    RESULTS.append((label, check, caught, (c or {}).get("detail", "this check does not exist")))


def expect_clean_ev(label: str, check: str, item: dict) -> None:
    """Positive control: a clean input must not fail this check."""
    c = item["checks"].get(check)
    ok = bool(c) and c["ok"]
    RESULTS.append((label, f"!{check}", ok,
                    (c or {}).get("detail", "this check does not exist") or "clean"))


def stream_row(name: str, **kw) -> dict:
    m = E.METRIC_BY_NAME[name]
    return {"name": name, "unit": m.unit, "step_days": kw.get("step_days", 1),
            "base_eff": kw.get("base_eff"), "source": kw.get("source", "profile"),
            "kind": kw.get("kind", "daily_metric"), "device": list(m.devices),
            "n_points": 0, "tags": list(m.tags)}


def flat(name: str, base: float, T: int, end: int, step: int = 1,
         shift_from: int | None = None, shift: float = 0.0, spike: float | None = None):
    m = E.METRIC_BY_NAME[name]
    pts = []
    for d in range(0, end + 1, step):
        v = base + (shift if (shift_from is not None and d >= shift_from) else 0.0)
        pts.append({"ts": d, "value": round(v, m.ndigits) if m.ndigits else int(v)})
    if spike is not None:
        pts[len(pts) // 2]["value"] = spike
    return pts


def _wear_pts_for_selftest(name: str, base: float, end: int, seed: str,
                           mask: bool = True, case_id: str = "RC-EW-01") -> list[dict]:
    """Test fixture: render a wearable stream from calibration, with or without the non-wear mask."""
    from haenv import wearable as _W
    m = E.METRIC_BY_NAME[name]
    days = list(range(0, end + 1))
    series = _W.ar1(name, base, days, seed, m.hard_range)
    # The mask is keyed on the patient, as in `events.render_stream`.
    keep = _W.worn_on(case_id, name, days) if mask else [True] * len(days)
    return [{"ts": d, "value": round(v, m.ndigits) if m.ndigits else int(round(v))}
            for d, v, k in zip(days, series, keep) if k]


def main() -> int:
    job = load_job(ROOT / "inputs/early_warning-20.job.yaml", root=ROOT)
    by = {c.case_id: c for c in job.cases}
    cs_wear = by["RC-EW-01"]           # has a wearable
    cs_nowear = by["RC-EW-02"]         # only smart_scale + lab_panel
    cs_hr = by["RC-EW-20"]             # chart records resting heart rate 102 (hyperthyroidism)
    f_wear, f_nowear, f_hr = E.facts_of(cs_wear), E.facts_of(cs_nowear), E.facts_of(cs_hr)
    T, END = 84, 365
    rev = 18 * 7

    # ---------- (1) inconsistent with raw_case's basic facts ----------
    expect_fail("Inject a steps stream into a patient with no wearable device", "device_backed",
                V.check_stream(stream_row("steps"), flat("steps", 7000, T, END),
                               f_nowear, "calorie_intake_change", T, rev, ED))
    expect_fail("Baseline doesn't match the medical record's measured value (resting heart rate 102)", "baseline_matches_raw_facts",
                V.check_stream(stream_row("resting_hr"), flat("resting_hr", 64, T, END),
                               f_hr, "calorie_intake_change", T, rev, ED))
    expect_fail("A single point exceeds the physiological range (steps=90000)", "values_in_physio_range",
                V.check_stream(stream_row("steps"), flat("steps", 7000, T, END, spike=90000),
                               f_wear, "calorie_intake_change", T, rev, ED))
    expect_fail("Sampling density doesn't match event density (2 days/point vs. daily)", "cadence_matches_density",
                V.check_stream(stream_row("steps", step_days=2),
                               flat("steps", 7000, T, END, step=2),
                               f_wear, "calorie_intake_change", T, rev, ED))
    expect_fail("Signal name collides with this disease's clinical domain (weight)", "in_aux_whitelist",
                V.check_stream({"name": "weight", "kind": "daily_metric"},
                               flat("steps", 90, T, END), f_wear, "calorie_intake_change",
                               T, rev, ED))

    # ---------- (1b) positive controls for `check_stream` ----------
    # `steps` is a calibrated stream, so the clean fixture uses calibration plus a
    # real non-wear mask rather than `flat()`.
    _clean_pts = _wear_pts_for_selftest("steps", 7000, END, "CLEAN1")
    _clean = V.check_stream(stream_row("steps"), _clean_pts,
                            f_wear, "calorie_intake_change", T, rev, ED)
    for _name in _clean["checks"]:
        expect_clean_ev(f"A clean steps stream must not be failed by `{_name}`", _name, _clean)

    # ---------- (2) affects the answer's conclusion ----------
    expect_fail("Inject a proxy metric for the true driver (steps when driver=activity_decline)",
                "not_driver_or_comorbid_proxy",
                V.check_stream(stream_row("steps"), flat("steps", 7000, T, END),
                               f_wear, "activity_decline", T, rev, ED))
    expect_fail("Metric drifts overall after the reversal point (leaks the reversal time)", "neutral_across_reversal",
                V.check_stream(stream_row("steps"), flat("steps", 7000, T, END,
                                                        shift_from=rev, shift=-3000),
                               f_wear, "calorie_intake_change", T, rev, ED))
    expect_fail("Metric drifts overall after T (leaks the future)", "neutral_across_T",
                V.check_stream(stream_row("steps"), flat("steps", 7000, T, END,
                                                        shift_from=T + 1, shift=-3500),
                               f_wear, "calorie_intake_change", T, rev, ED))

    # ---------- (3) event-layer defects ----------
    ev = lambda **kw: {"evidence_id": "EV-X", "kind": "benign_symptom", "day": 30,
                       "text": "落枕、颈部转动酸痛", "context": "睡姿不当,热敷后缓解", **kw}
    expect_fail("Event context contains attributive wording (after stopping medication...)", "context_non_attributive",
                V.check_event(ev(context="停药后食欲回升"), f_wear, "calorie_intake_change", T, set()))
    expect_fail("Event timing falls after T", "timing_le_T",
                V.check_event(ev(day=T + 10), f_wear, "calorie_intake_change", T, set()))
    expect_fail("Event isn't in any known event pool (unknown origin)", "plausible_for_profile",
                V.check_event(ev(text="自造的症状"), f_wear, "calorie_intake_change", T, set()))
    expect_fail("Assign an exertion-requiring event to a 110kg/OSA patient", "plausible_for_profile",
                V.check_event(ev(text="运动后小腿肌肉酸痛", context="周末登山下坡较多"),
                              E.facts_of(by["RC-EW-03"]), "calorie_intake_change", T, set()))
    expect_fail("Benign event falls on the same day as a real symptom (contaminates the real signal)", "no_collision_with_real_symptom",
                V.check_event(ev(day=56), f_wear, "calorie_intake_change", T, {56}))
    expect_fail("Claims to be a real symptom but raw_case has no record of it", "traceable_to_raw_case",
                V.check_event(ev(kind="real_symptom", text="查无此症状"), f_wear,
                              "calorie_intake_change", T, set()))
    # Self-reported tags from an LLM plan must not mask a driver proxy: the verdict
    # uses self-reported tags UNION text inference.
    expect_fail("LLM self-reported tag masks a driver proxy (birthday dinner vs. calorie-intake driver)", "not_driver_or_comorbid_proxy",
                V.check_event({"evidence_id": "EV-L9", "kind": "life_event", "day": 40,
                               "text": "参加亲友生日聚餐", "context": "周末家庭活动",
                               "claim": {"tags": ["logistics"], "exertion": False,
                                         "topic": "llm:聚餐"}},
                              f_wear, "calorie_intake_change", T, set()))
    expect_fail("LLM self-reports exertion=false but the text describes hiking (110kg/OSA patient)", "plausible_for_profile",
                V.check_event({"evidence_id": "EV-L10", "kind": "benign_symptom", "day": 40,
                               "text": "周末登山后小腿酸", "context": "户外活动",
                               "claim": {"tags": ["musculoskeletal"], "exertion": False,
                                         "topic": "llm:登山"}},
                              E.facts_of(by["RC-EW-03"]), "unknown_or_multifactorial", T, set()))
    expect_fail("Inject a red-flag symptom disguised as irrelevant noise", "not_red_flag_symptom",
                V.check_event({"evidence_id": "EV-L11", "kind": "benign_symptom", "day": 40,
                               "text": "活动后胸痛伴呼吸困难", "context": "上楼时出现",
                               "claim": {"tags": ["musculoskeletal"], "exertion": False,
                                         "topic": "llm:胸痛"}},
                              f_wear, "calorie_intake_change", T, set()))
    late = [{"evidence_id": f"EV-L{i}", "kind": "benign_symptom", "day": T - 3 - i}
            for i in range(5)]
    sp = V.check_event_spread(late, T)
    RESULTS.append(("All benign events are crammed into the final stretch right before T", "benign_spread_uniform",
                    not sp["ok"], sp["checks"]["benign_spread_uniform"]["detail"]))

    # ---------- (4) case level: conclusion invariance / text leakage ----------
    raw, audit = build_case(cs_wear)
    assert raw is not None, f"Baseline case should be emitted: {audit}"
    T0 = int(raw.prediction_context["prediction_time_T"])

    flip = copy.deepcopy(raw)                     # turn the pre-T main signal into a sustained rise -> both offline solvers should change their verdict
    nadir = min(p["value"] for p in flip.longitudinal_data["weight"])
    for p in flip.longitudinal_data["weight"]:
        p["value"] = round(nadir + 0.35 * p["ts"] / 7.0, 2)
    inv = V.check_conclusion_invariance(raw, flip, T0)
    bad = [k for k, v in inv["checks"].items() if k.startswith("conclusion_same") and not v["ok"]]
    RESULTS.append(("Injection changed the offline solver's conclusion", "conclusion_same:*", bool(bad),
                    f"Failing items {bad}"))

    leaky = copy.deepcopy(raw)                    # stuff the answer into the solver-visible payload
    leaky.user_profile = {**leaky.user_profile, "note": "真驱动: poor_medication_adherence"}
    scan = V.scan_solver_text(leaky, T0)
    RESULTS.append(("solver-visible text contains the true driver / answer words", "solver_text_no_answer_info",
                    not scan["ok"], "; ".join(scan["findings"][:3])))
    future = copy.deepcopy(raw)                   # temporal leakage must be caught by the kernel gate
    future.prediction_context = {**future.prediction_context, "prediction_time_T": T0}
    future.longitudinal_data["weight"].append({"ts": T0, "value": 999.0})
    scan2 = V.scan_solver_text(future, T0)
    # Positive control: an out-of-range value is the range gate's job (GV-1);
    # `scan_solver_text` only looks for answer words and must let it through.
    RESULTS.append(("A physiologically impossible value of 999kg is mixed into pre-T ⇒ the text scanner should let it through (the range gate lives on the GV-1 side)",
                    "!range_is_not_text_scanners_job", bool(scan2["ok"]),
                    f"Text scan {'CLEAN' if scan2['ok'] else scan2['findings'][:2]}"))

    # ---------- (5) CC-1's "injection must not change ground truth" checks ----------
    def _tamper(**kw):
        """Tamper with a ground-truth field of a copy and return `check_conclusion_invariance`."""
        t = copy.deepcopy(raw)
        for k, v in kw.items():
            if k == "adjudication":
                t.adjudication = {**t.adjudication, **v}
            else:
                setattr(t, k, v)
        return V.check_conclusion_invariance(raw, t, T0)

    _flip_out = ("event_not_occurred" if raw.outcome_label == "event_occurred"
                 else "event_occurred")
    expect_fail("outcome_label was changed after injection", "outcome_label_unchanged",
                _tamper(outcome_label=_flip_out))
    expect_fail("gold_drivers was changed after injection", "gold_drivers_unchanged",
                _tamper(gold_drivers=["measurement_noise"]))
    expect_fail("label_rule was changed after injection (threshold quietly loosened)", "label_rule_unchanged",
                _tamper(label_rule={**raw.label_rule, "min_change_frac": 0.99}))
    expect_fail("adjudication.primary_driver was changed after injection", "adjudication_core_unchanged",
                _tamper(adjudication={"primary_driver": "measurement_noise"}))
    expect_fail("red_flag_present was flipped after injection (the arming bit for the missed-red-flag hard gate)",
                "adjudication_core_unchanged",
                _tamper(adjudication={"red_flag_present":
                                      not raw.adjudication.get("red_flag_present")}))
    # CC-1 checks `adjudication` default-deny, so each covered field (`ddx`,
    # `artifact_flags`, `reversal_points`) gets its own defect. When the fixture has
    # no such field, conjuring one is the defect (exercises `_adj_gold_keys` taking a
    # union).
    _ddx0 = dict(raw.adjudication.get("ddx") or {})
    expect_fail("adjudication.ddx.diagnosis was changed / conjured out of nothing after injection (the main task's gold label)",
                "adjudication_core_unchanged",
                _tamper(adjudication={"ddx": {**_ddx0, "diagnosis": "__tampered__"}}))
    expect_fail("adjudication.ddx.join_gold was changed / conjured out of nothing after injection (the joint-attribution correct answer)",
                "adjudication_core_unchanged",
                _tamper(adjudication={"ddx": {**_ddx0, "join_gold": "__tampered__"}}))
    _af0 = raw.adjudication.get("artifact_flags")
    if _af0:
        expect_fail("artifact_flags's window was shifted after injection (the precondition for the noise-resistance hard gate)",
                    "adjudication_core_unchanged",
                    _tamper(adjudication={"artifact_flags": {**_af0, "d0": int(_af0.get("d0", 0)) + 1}}))
    else:
        expect_fail("artifact_flags was conjured out of nothing after injection (looking only at the pre-injection side would miss this)",
                    "adjudication_core_unchanged",
                    _tamper(adjudication={"artifact_flags": {"is_artifact_window": True,
                                                             "artifact_signal": "weight_kg"}}))
    expect_fail("reversal_points was changed after injection (a gold label)",
                "reversal_points_unchanged",
                _tamper(reversal_points=[{"week": 99, "type": "trap", "flip": "__tampered__"}]))
    # Ledger keys are not ground truth and must not fail this check.
    _bk = _tamper(adjudication={"distractor_level": "high",
                                "distractor_evidence_ids": ["EV-X"]})
    RESULTS.append(("Ledger keys (distractor_*) changing must not fail this check -- they are the product of injection itself",
                    "!adjudication_core_unchanged",
                    _bk["checks"]["adjudication_core_unchanged"]["ok"],
                    _bk["checks"]["adjudication_core_unchanged"]["detail"] or "clean"))

    # Positive controls: an unchanged copy fails none of these checks.
    _same = V.check_conclusion_invariance(raw, copy.deepcopy(raw), T0)
    for _k in ("outcome_label_unchanged", "gold_drivers_unchanged",
               "label_rule_unchanged", "adjudication_core_unchanged",
               "reversal_points_unchanged"):
        RESULTS.append((f"When ground truth hasn't changed, {_k} must not fail", f"!{_k}",
                        _same["checks"][_k]["ok"], _same["checks"][_k]["detail"] or "clean"))

    # ---------- (6) remaining per-item checks ----------
    expect_fail("The emitted context isn't in the closed vocabulary's emittable forms (bypasses the generation-side gate)", "context_declared",
                V.check_event(ev(kind="real_symptom", text="落枕、颈部转动酸痛",
                                 context="代谢阻力"),
                              f_wear, "calorie_intake_change", T, set()))
    expect_clean_ev("A registered, emittable context must not be blocked", "context_declared",
                    V.check_event(ev(kind="real_symptom", context="无怀孕可能"),
                                  f_wear, "calorie_intake_change", T, set()))
    expect_fail("Event text contains a Chinese synonym for the driver (poor adherence)", "no_answer_words",
                V.check_event(ev(text="落枕、颈部转动酸痛", context="", day=30,
                                 **{"kind": "benign_symptom"}) | {"text": "近来依从性差"},
                              f_wear, "calorie_intake_change", T, set()))
    expect_fail("Event text contains an outcome word (weight regain)", "no_answer_words",
                V.check_event(ev(text="体重复胖明显"), f_wear, "calorie_intake_change", T, set()))
    expect_fail("The same minor ailment reappears in different wording (topic collision)", "no_duplicate_topic",
                V.check_event(ev(), f_wear, "calorie_intake_change", T, set(),
                              other_topics={"stiff_neck"}))
    expect_fail("Inject a signal whose name collides with this disease's clinical domain (HbA1c for obesity)", "not_clinical_domain_signal",
                V.check_stream({"name": "HbA1c", "kind": "daily_metric"},
                               flat("steps", 7.1, T, END), f_wear,
                               "calorie_intake_change", T, rev, ED))
    expect_fail("A metric stream with an empty sequence", "has_points",
                V.check_stream(stream_row("steps"), [], f_wear,
                               "calorie_intake_change", T, rev, ED))
    expect_fail("Event text is empty", "has_text",
                V.check_event(ev(text="   "), f_wear, "calorie_intake_change", T, set()))
    expect_fail("Fewer than 5 visible points at ≤T (almost no pre-T observations)", "pre_T_visible",
                V.check_stream(stream_row("steps", step_days=40),
                               flat("steps", 7000, T, END, step=40),
                               f_wear, "calorie_intake_change", T, rev, ED))
    expect_fail("Event timing is a negative number", "timing_nonneg",
                V.check_event(ev(day=-3), f_wear, "calorie_intake_change", T, set()))

    # ---------- (7) wearable realism on calibrated streams ----------
    from haenv import wearable as _W
    _wear_pts = _wear_pts_for_selftest

    # Zero missing days is only a defect when the patient's own rate predicts
    # several off-runs; pick a synthetic patient whose hrv expects >= 8 runs.
    import copy as _copy
    _f_busy = _copy.copy(f_wear)
    for _k in range(500):
        _cid = f"SC-BUSY-{_k}"
        if END * _W.case_rates(_cid, "hrv")[2] * (1 - _W.CALIBRATION["hrv"].p_off_off) >= 8:
            break
    _f_busy.case_id = _cid
    expect_fail("Calibrated stream declares a non-wear mask but is missing zero days (0% missingness)", "cadence_matches_density",
                V.check_stream(stream_row("hrv"),
                               _wear_pts("hrv", 46, END, "SC1", mask=False, case_id=_cid),
                               _f_busy, "calorie_intake_change", T, rev, ED))
    expect_fail("Calibrated stream has a continuous gap exceeding the upper bound", "cadence_matches_density",
                V.check_stream(
                    stream_row("hrv"),
                    [q for q in _wear_pts("hrv", 46, END, "SC1", mask=False)
                     if not (5 <= q["ts"] <= 5 + _W.MAX_GAP_DAYS + 2)],
                    f_wear, "calorie_intake_change", T, rev, ED))
    expect_fail("Calibrated stream is rendered as fixed-cadence sampling (every other day, no non-wear structure)", "cadence_matches_density",
                V.check_stream(
                    stream_row("hrv", step_days=2),
                    [q for q in _wear_pts("hrv", 46, END, "SC1", mask=False)
                     if q["ts"] % 2 == 0],
                    f_wear, "calorie_intake_change", T, rev, ED))
    expect_clean_ev("!A calibrated stream with a real non-wear mask must not be misjudged", "cadence_matches_density",
                    V.check_stream(stream_row("hrv"), _wear_pts("hrv", 46, END, "SC7"),
                                   f_wear, "calorie_intake_change", T, rev, ED))
    expect_clean_ev("!A derived stream following its parent stream's irregular rhythm must not be misjudged", "cadence_matches_density",
                    V.check_stream(
                        stream_row("vo2_max"),
                        [{"ts": q["ts"], "value": round(_W.vo2max_uth(q["value"], 45), 1)}
                         for q in _wear_pts("resting_hr", 68, END, "SC7")],
                        f_wear, "calorie_intake_change", T, rev, ED))

    # ---------- (8) sensitivity of the relaxed bands on realistic input ----------
    # Streams carry realistic noise and non-wear; the shift (25 percent) is one the
    # bands must still catch.
    def _shifted(name, base, seed, start, frac):
        return [{"ts": q["ts"], "value": (q["value"] * (1.0 - frac) if q["ts"] >= start
                                          else q["value"])}
                for q in _wear_pts(name, base, END, seed)]
    expect_fail("steps drops 25% after T (realistic noise and non-wear)", "neutral_across_T",
                V.check_stream(stream_row("steps"), _shifted("steps", 7400, "PW1", T + 1, 0.25),
                               f_wear, "calorie_intake_change", T, rev, ED))
    expect_fail("steps drops 25% after reversal (realistic noise and non-wear)",
                "neutral_across_reversal",
                V.check_stream(stream_row("steps"), _shifted("steps", 7400, "PW2", rev, 0.25),
                               f_wear, "calorie_intake_change", T, rev, ED))
    expect_fail("hrv drops 25% after T (realistic noise and non-wear)", "neutral_across_T",
                V.check_stream(stream_row("hrv"), _shifted("hrv", 46, "PW3", T + 1, 0.25),
                               f_wear, "calorie_intake_change", T, rev, ED))
    expect_clean_ev("!steps with realistic noise and no shift is not flagged", "neutral_across_T",
                    V.check_stream(stream_row("steps"), _shifted("steps", 7400, "PW1", T + 1, 0.0),
                                   f_wear, "calorie_intake_change", T, rev, ED))
    # Neutrality is judged on the injector's own render, not on the observed series,
    # so a real world footprint after T (e.g. resting_hr in pheochromocytoma) is not a
    # leak.
    expect_clean_ev("!world footprint on the observed series with a clean injector view",
                    "neutral_across_T",
                    V.check_stream(stream_row("steps"), _shifted("steps", 7400, "PW1", T + 1, 0.25),
                                   f_wear, "calorie_intake_change", T, rev, ED,
                                   neutral_pts=_shifted("steps", 7400, "PW1", T + 1, 0.0)))
    expect_clean_ev("!world footprint lowering the observed level with a clean injector view",
                    "baseline_matches_raw_facts",
                    V.check_stream(stream_row("steps"), _shifted("steps", 7400, "PW1", 0, 0.25),
                                   f_wear, "calorie_intake_change", T, rev, ED,
                                   neutral_pts=_shifted("steps", 7400, "PW1", 0, 0.0)))
    # Removing every 4th day raises non-wear above the declared-rate band without
    # pushing the longest gap past `MAX_GAP_DAYS`, which would trip a different check.
    expect_fail("stress_score non-wear 6x the declared rate", "cadence_matches_density",
                V.check_stream(stream_row("stress_score"),
                               [q for q in _wear_pts("stress_score", 42, END, "SC3")
                                if q["ts"] % 4 != 1],
                               f_wear, "calorie_intake_change", T, rev, ED))

    # ---------- summary ----------
    print("\nNegative controls: each row = one deliberately injected defect, expected to be caught by the verifier\n")
    print(f"| {'Defect':44s} | {'Failing check':32s} | Caught |")
    print(f"|{'-'*46}|{'-'*34}|------|")
    n_miss = 0
    for label, check, caught, detail in RESULTS:
        if check != "(reference)" and not caught:
            n_miss += 1
        print(f"| {label:44s} | {check:32s} | {'✅' if caught else '❌ missed'} |")
    # Negative and positive controls are counted separately; only the
    # negative-control count shows no negative control was removed.
    n_ref = sum(1 for _, c, _, _ in RESULTS if c == "(reference)")
    n_pos = sum(1 for _, c, _, _ in RESULTS if isinstance(c, str) and c.startswith("!"))
    n_neg = len(RESULTS) - n_ref - n_pos
    # Documentation parses this line (`N checks, M missed.`) and the breakdown line below.
    print(f"\n{len(RESULTS)} checks, {n_miss} missed.")
    print(f"  Breakdown: **{n_neg} true negative controls** + {n_pos} positive controls"
          + (f" + {n_ref} always-true reference rows (they cannot fail; should be zero)" if n_ref else "")
          + "\n  Definitions: true negative control = a deliberately broken input that must be caught"
          " · positive control (`!` prefix) = a clean input that must not be flagged."
          "\n  The true-negative-control count is the one to cite: removing 5 negative controls"
          " and adding 5 positive ones leaves the total unchanged.")
    for label, check, caught, detail in RESULTS:
        print(f"  - {label} → {check}: {detail}")
    return 0 if n_miss == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
