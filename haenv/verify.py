"""verify.py — per-item verifier for injected daily metrics and events.

Three groups of checks: each item is consistent with the raw case
(device, physiological range, the patient's baseline, sampling density, event
plausibility); injection does not change the answer (gold fields identical
before and after, offline solvers reach the same conclusion); and the
solver-visible text carries no answer information. A failing item is discarded
and reinjected; a failing case-level check blocks emission.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
import logging

from . import wearable as _wear

from latent import DISEASE_SIGNAL_DOMAIN                  # kernel
from build import build_instance, leakage_probe
from solver import BaselineSolver, RobustSolver, ALLOWED_DRIVERS

from .events import event_pools as _event_pools
from .events import (ATTRIBUTIVE_MARKERS, AUX_WHITELIST,
                     METRIC_BY_NAME, _adj, _event_ok, attributive_hits, effective_tags,
                     facts_of, event_proxy_tags, inherited_profile, proxy_tags)

log = logging.getLogger("haenv.verify")


def _gold_signal_of() -> dict[str, str]:
    # Imports registry lazily to avoid a circular dependency with build.
    from .registry import GOLD_EVIDENCE
    return {d: g["signal"] for d, g in GOLD_EVIDENCE.items()}


class _LazyMap(dict):
    def get(self, k, default=None):
        return _gold_signal_of().get(k, default)


GOLD_SIGNAL_OF = _LazyMap()

# ---------------------------------------------------------------- Answer vocabulary
# Semantic-layer answer words, on top of the kernel's structural FORBIDDEN_TOKENS.
TRUTH_TOKENS = ("outcome_label", "event_occurred", "event_not_occurred", "gold_driver",
                "adjudication", "latent_premise", "reversal_point", "artifact_flag",
                "noise_class", "distractor", "future_data", "verifier", "label_rule",
                "injected_event_manifest", "primary_driver", "clinician_action_warranted")
TRUTH_WORDS_ZH = ("结局标签", "真驱动", "金标", "教学点", "转归", "裁定", "隐变量", "隐控制",
                  "泄漏", "标准答案", "复胖", "未复胖", "维持成功", "确诊", "疗效失败",
                  "药物无效", "停药后", "预期药效", "评测焦点")
# Chinese synonyms for drivers (the English driver list uses the kernel's ALLOWED_DRIVERS
# directly)
DRIVER_WORDS_ZH = ("依从性差", "漏服", "没吃药", "自行停药", "不耐受", "热量摄入增加", "摄入回升",
                   "活动量下降", "睡眠不足所致", "并用药物", "急性疾病所致", "测量噪声",
                   "生物学低应答", "费用问题", "可及性问题")


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _ck(ok: bool, detail: str = "") -> dict:
    return {"ok": bool(ok), "detail": detail}


# ================================================================ (1) Per-metric-stream checks
def check_stream(row: dict, pts: list[dict], f, driver: str, T: int,
                 rev_day: int | None, event_density: dict,
                 insufficient_tier: bool = False,
                 neutral_pts: list[dict] | None = None) -> dict:
    """All checks for one daily metric stream. Returns
    {item, kind, checks:{name:{ok,detail}}, ok}.
    """
    name = row["name"]
    m = METRIC_BY_NAME.get(name)
    vals = [p["value"] for p in pts]
    checks: dict[str, dict] = {}

    checks["in_aux_whitelist"] = _ck(name in AUX_WHITELIST, name)
    domain = DISEASE_SIGNAL_DOMAIN.get(f.disease, {})
    checks["not_clinical_domain_signal"] = _ck(
        name not in domain, f"{name} {'∈' if name in domain else '∉'} this condition's clinical domain {sorted(domain)}")
    checks["device_backed"] = _ck(
        bool(m) and (not m.devices or bool(set(m.devices) & set(f.devices))),
        f"needs {list(m.devices) if m else '?'} · on hand {list(f.devices)}")

    kind = row.get("kind", "daily_metric")
    if not m or not vals:
        checks["has_points"] = _ck(False, "empty series")
        return {"item": name, "kind": kind, "checks": checks, "ok": False}

    bad = [f"{p['value']}@d{p['ts']}" for p in pts
           if not (m.hard_range[0] <= p["value"] <= m.hard_range[1])]
    checks["values_in_physio_range"] = _ck(not bad, f"{list(m.hard_range)} · out of range {bad[:3]}")

    exp = (float(f.baseline_vitals[m.vital_key])
           if (m.vital_key and f.baseline_vitals.get(m.vital_key) is not None)
           else _adj(name, m.base, f))
    # Baseline judged on the injector's own render when available: the observed
    # series also carries event footprints.
    mu = _mean([q["value"] for q in neutral_pts]) if neutral_pts else _mean(vals)
    checks["baseline_matches_raw_facts"] = _ck(
        abs(mu - exp) <= m.tol,
        f"mean {mu:.1f} vs expected {exp:.1f}±{m.tol} ({row.get('source')})")

    mpw = float(event_density.get("measure_per_week", 7) or 7)
    want = max(1, int(round(7.0 / max(0.5, mpw))))
    steps = sorted({b["ts"] - a["ts"] for a, b in zip(pts, pts[1:])}) or [want]
    if kind == "inherited_metric":
        # Upstream-injected streams follow their injector's cadence, not this case's
        # event density; every other check still applies.
        checks["cadence_matches_density"] = _ck(
            True, f"steps {steps} · an upstream injector's stream follows its injector's cadence, not this case's event density")
    elif name in _wear.DERIVED_BINDINGS:
        # Derived streams follow their parent's observation days.
        checks["cadence_matches_density"] = _ck(
            True, f"derived stream; cadence follows parent {list(_wear.DERIVED_BINDINGS[name])}")
    elif _wear.is_calibrated(name):
        # Calibrated wearable streams are daily with a non-wear mask: check the gap
        # bound and that the realised non-wear rate matches the declared one.
        cal = _wear.CALIBRATION[name]
        gap_max = max(steps) if steps else 1
        span = pts[-1]["ts"] - pts[0]["ts"] + 1 if len(pts) > 1 else len(pts)
        off_rate = 1.0 - len(pts) / max(1, span)
        n_off = span - len(pts)
        _cid = str(getattr(f, "case_id", "") or "")
        _, _own, want_off = _wear.case_rates(_cid, name)
        tol = _wear.off_rate_tolerance(name, span, want_off)
        # Zero missing days fails only when at least 5 off-runs are expected
        # (a legitimate zero is then under 1% likely).
        expected_runs = span * want_off * (1.0 - cal.p_off_off)
        rendered_mask = not (expected_runs >= 5.0 and n_off == 0)
        ok = (steps[0] >= 1 and gap_max <= _wear.MAX_GAP_DAYS and rendered_mask
              and abs(off_rate - want_off) <= tol)
        checks["cadence_matches_density"] = _ck(
            ok, f"daily sampling + non-wear mask: steps {steps[:4]} · longest gap {gap_max}d"
                f" (limit {_wear.MAX_GAP_DAYS}) · non-wear rate {off_rate:.3f}"
                f" vs declared {want_off:.3f}±{tol:.3f}"
                f" · {n_off} days missing / about {expected_runs:.1f} runs expected")
    else:
        checks["cadence_matches_density"] = _ck(steps[0] == want and steps[-1] == want,
                                                f"steps {steps} vs required {want}d")
    _n_pre = sum(1 for p in pts if p["ts"] <= T)
    checks["pre_T_visible"] = _ck(_n_pre >= 5, f"points ≤T {_n_pre}")

    if kind == "gold_evidence_metric":
        # The gold-evidence stream is legitimate evidence and may drift; it must exist
        # in every case, and non-gold streams must stay directionless.
        is_gold = name in {GOLD_SIGNAL_OF.get(driver)}
        if not is_gold:
            # A non-gold stream fails only if its first-third/last-third drift exceeds its own
            # noise (pooled SD), with a floor for nearly constant streams; flatness alone
            # would leak which stream is gold.
            _n = len(vals)
            _tol_floor = 0.5 * max(0.6, 0.15 * max(1.0, abs(_mean(vals) or 1.0)))
            if _n >= 6:
                _k = max(2, _n // 3)
                _a, _b = vals[:_k], vals[-_k:]
                _drift = abs((_mean(_b) or 0.0) - (_mean(_a) or 0.0))
                _mu = _mean(vals) or 0.0
                _var = sum((v - _mu) ** 2 for v in vals) / max(1, _n - 1)
                _noise = _var ** 0.5
                checks["neutral_when_not_gold"] = _ck(
                    _drift <= max(_tol_floor, 1.0 * _noise),
                    f"a non-gold stream must have no direction: drift {_drift:.2f} vs own noise {_noise:.2f}"
                    f" (tolerance floor {_tol_floor:.2f})")
            else:
                # Too few points to distinguish "drift" from "noise" => falls back to a
                # magnitude check (not a pass-through)
                span = (max(vals) - min(vals)) if vals else 0.0
                checks["neutral_when_not_gold"] = _ck(
                    span <= max(0.6, 0.15 * max(1.0, abs(_mean(vals) or 1.0))),
                    f"{_n} points < 6, direction not resolvable => judged on magnitude: range {span:.2f}")
        checks["baseline_matches_raw_facts"] = _ck(True, "gold evidence stream: not compared with the constitutional baseline (the driver sets it)")
        checks["cadence_matches_density"] = _ck(True, "gold evidence stream: follows the world-layer cadence, not the event density")
        ok = all(c["ok"] for c in checks.values())
        return {"item": name, "kind": kind, "source": row.get("source"),
                "checks": checks, "ok": ok}

    # ---- Answer neutrality ----
    checks["not_driver_or_comorbid_proxy"] = _ck(
        not (set(m.tags) & proxy_tags(f, driver)),
        f"tags {list(m.tags)} vs answer-related for this case {sorted(proxy_tags(f, driver))}")
    tol_n = max(0.35 * m.amp, 0.08 * max(1.0, abs(exp)))
    def _tol_for(va: list[float], vb: list[float]) -> float:
        """Tolerance on the difference of two segment means: three standard errors
        from the pooled within-segment SD, with effective sizes n * (1 - phi) / (1 + phi)
        under lag-1 persistence phi.
        """
        # Derived streams use their grid parent's persistence.
        cal_name = name if _wear.is_calibrated(name) else _wear.GRID_PARENT.get(name)
        if not cal_name or not _wear.is_calibrated(cal_name) or len(va) < 2 or len(vb) < 2:
            return tol_n
        return max(0.08 * max(1.0, abs(exp)), _wear.mean_shift_tolerance(cal_name, va, vb))

    _npts = neutral_pts if neutral_pts else pts
    if rev_day is not None:
        _va = [p["value"] for p in _npts if p["ts"] < rev_day]
        _vb = [p["value"] for p in _npts if p["ts"] >= rev_day]
        a, b = _mean(_va), _mean(_vb)
        _t = _tol_for(_va, _vb)
        checks["neutral_across_reversal"] = _ck(
            a is None or b is None or abs(a - b) <= _t,
            f"before reversal {a if a is None else round(a,2)} / after {b if b is None else round(b,2)} · tolerance {_t:.2f}")
    _va = [p["value"] for p in _npts if p["ts"] <= T]
    _vb = [p["value"] for p in _npts if p["ts"] > T]
    a, b = _mean(_va), _mean(_vb)
    _t = _tol_for(_va, _vb)
    checks["neutral_across_T"] = _ck(
        a is None or b is None or abs(a - b) <= _t,
        f"≤T {a if a is None else round(a,2)} / >T {b if b is None else round(b,2)} · tolerance {_t:.2f}")

    ok = all(c["ok"] for c in checks.values())
    return {"item": name, "kind": kind, "source": row.get("source"), "checks": checks, "ok": ok}


# ================================================================ (1) Per-event checks
def _pool_item(text: str, claim: dict | None = None) -> dict | None:
    """Event text -> its profile: from the event pools for the deterministic path,
    or from the model's self-reported tags (`claim`) for the LLM path, which are
    then checked for plausibility against the patient.
    """
    if claim:
        return {"text": text, "context": claim.get("context", ""),
                "tags": tuple(claim.get("tags") or ()),
                "exertion": bool(claim.get("exertion")), "topic": claim.get("topic")}
    for it in [i for pool in _event_pools() for i in pool]:
        if it["text"] == text:
            return it
    inh = inherited_profile(text)
    return {**inh, "text": text} if inh else None


# Red-flag words: as a "benign" event they would create a missed-diagnosis
# trap on a case whose gold has red_flag_present=False.
RED_FLAG_WORDS = ("胸痛", "胸闷压迫", "呼吸困难", "喘不上气", "晕厥", "意识丧失", "抽搐",
                  "偏瘫", "半身无力", "言语不清", "黑便", "呕血", "咯血", "血尿", "剧烈头痛",
                  "视力骤降", "视物模糊加重", "高热不退", "持续呕吐", "腹痛剧烈", "自杀")


# Event kinds exempt only from the topic-level answer-proxy check; they must be
# traceable to `registry/lookalikes.yaml` instead.
PROXY_EXEMPT_KINDS: frozenset[str] = frozenset({"lookalike"})


def check_event(row: dict, f, driver: str, T: int, real_days: set[int],
                other_topics: set[str] | None = None, ddx_aliases=None) -> dict:
    """All checks for one event (EV). real_symptom uses different rules than benign/life events
    (it is legitimate evidence and is not held to answer neutrality)."""
    kind, checks = row["kind"], {}
    day, text, ctx = int(row["day"]), row.get("text", ""), row.get("context", "")

    checks["timing_le_T"] = _ck(day <= T, f"day{day} ≤ T{T}")
    checks["timing_nonneg"] = _ck(day >= 0, f"day{day}")
    blob = f"{text} {ctx}"
    attr = attributive_hits(blob, getattr(f, "drug", None))   # "ineffective/failed" must co-occur with this case's intervention axis
    checks["context_non_attributive"] = _ck(not attr, f"attributive wording found {attr[:3]}")
    checks["no_answer_words"] = _ck(
        not any(w in blob for w in TRUTH_WORDS_ZH + DRIVER_WORDS_ZH),
        f"answer words found {[w for w in TRUTH_WORDS_ZH + DRIVER_WORDS_ZH if w in blob][:3]}")

    if kind == "real_symptom":
        # Real symptom: it must actually appear in raw_case, at a matching day (fabricating
        # legitimate evidence out of nothing is not allowed)
        from .events import scrub_annotation
        hit = [s for s in f.symptoms
               if s["day"] == day and text in (s["text"],
                                               scrub_annotation(s["text"], ddx_aliases)[0])]
        checks["traceable_to_raw_case"] = _ck(
            bool(hit), f"raw_case symptom-table matches {len(hit)} (compared before and after removing parentheticals)")
        # Any emitted context must be one of overlay's emittable forms (default-deny).
        from .overlay import emitted_forms
        _decl = (not ctx) or (ctx in emitted_forms())
        checks["context_declared"] = _ck(
            _decl, f"context {ctx!r} {'is registered among' if _decl else 'is not among'} "
                   f"the emittable forms of overlay.CONTEXT_VOCAB{'' if _decl else ' (default-deny)'}")
    elif kind in PROXY_EXEMPT_KINDS:
        from .registry import load_lookalikes
        _pool = {str(it.get("text")): it
                 for ents in load_lookalikes().values() for it in ents}
        _hit = _pool.get(text)
        checks["traceable_to_registry"] = _ck(
            bool(_hit), f"registry/lookalikes.yaml match {'✓' if _hit else '✗ (default-deny)'}")
        checks["registry_fields_present"] = _ck(
            bool(_hit) and bool(_hit.get("why_related")) and bool(_hit.get("why_looks_benign")),
            f"why_related/why_looks_benign {'present' if _hit else 'missing'}")
        checks["not_red_flag_symptom"] = _ck(
            not any(w in blob for w in RED_FLAG_WORDS),
            f"red-flag words found {[w for w in RED_FLAG_WORDS if w in blob][:3]}")
        checks["has_text"] = _ck(bool(text.strip()), "event text is empty")
        checks["no_collision_with_real_symptom"] = _ck(
            day not in real_days, f"day{day} collides with real-symptom days {sorted(real_days)}")
    else:
        found = _pool_item(text, row.get("claim"))
        it = {**found, "text": text, "context": ctx} if found else None
        _exempt = kind in PROXY_EXEMPT_KINDS
        ok, why = (_event_ok(it, f, event_proxy_tags(f, driver), proxy_exempt=_exempt)
                   if it else (False, "not_in_pool"))
        checks["plausible_for_profile"] = _ck(ok, why or f"profile: heavy={f.heavy} elderly={f.elderly}")
        # Answer relevance is judged on role tags (`events.role_tags`), not on the
        # model's self-report.
        from .events import role_tags as _rt
        eff = _rt(it) if it else set()
        _banned = event_proxy_tags(f, driver)
        checks["not_driver_or_comorbid_proxy"] = _ck(
            bool(it) and (_exempt or not (eff & _banned)),
            (f"EXEMPT (kind={kind}: lookalike distractors are answer-related by definition; hits "
             f"{sorted(eff & _banned)} recorded; the case-level conclusion must stay unchanged) · " if _exempt else "")
            + f"effective tags {sorted(eff)} (self-reported {list((it or {}).get('tags', ()))})"
              f" vs banned for this case {sorted(_banned)}")
        checks["not_red_flag_symptom"] = _ck(
            not any(w in blob for w in RED_FLAG_WORDS),
            f"red-flag words found {[w for w in RED_FLAG_WORDS if w in blob][:3]} (truth red_flag_present=False; "
            f"injecting them as irrelevant noise would create an unadjudicable missed-diagnosis trap)")
        checks["has_text"] = _ck(bool(text.strip()), "event text is empty")
        checks["no_collision_with_real_symptom"] = _ck(
            day not in real_days, f"day{day} collides with real-symptom days {sorted(real_days)}")
        topic = (it or {}).get("topic")
        checks["no_duplicate_topic"] = _ck(
            not topic or topic not in (other_topics or set()),
            f"topic {topic} already used by another event in this case (the same minor complaint reworded)")

    ok = all(c["ok"] for c in checks.values())
    return {"item": row["evidence_id"], "kind": kind, "day": day, "text": text,
            "checks": checks, "ok": ok}


# ================================================================ (2) Per-case: conclusion invariance
def _conclusion(raw, T: int) -> dict:
    """The offline deterministic solver's conclusion (risk category + top driver + action).
    Must be identical before and after injection."""
    sp, _ = build_instance(raw, T)
    out = {}
    for name, s in (("baseline_slope", BaselineSolver()), ("robust_ref", RobustSolver())):
        o = s.solve(sp)
        out[name] = {"risk_category": o.forecast.get("risk_category"),
                     "risk": round(float(o.forecast.get("risk", 0.5)), 3),
                     "top_driver": (o.drivers or [{}])[0].get("driver"),
                     "action": (o.action or {}).get("selected_action_class")}
    return out


#: Keys inside `adjudication` that injection is expected to change; every other
#: key is compared as gold.
ADJ_BOOKKEEPING: dict[str, str] = {
    "distractor_evidence_ids": "evidence ids registered when distractors are injected -- they are the injection's own output, so they are expected to change.",
    "distractor_signals": "as above, per signal (which signals received decoys).",
    "distractor_level": "which distractor tier this case received -- a generation knob, not answer-side information.",
    "adjudication_protocol_present": "protocol marker, written with the injection.",
    "injected_event_manifest": "the injection ledger of batches from before the W/Q split (see `store.load_cases`). "
                               "It exists only after injection by definition; comparing it would use the injection's output to prove the injection changed nothing.",
}


def _adj_gold_keys(clean_raw, raw) -> list[str]:
    """Union of both sides' `adjudication` keys minus bookkeeping keys, so a gold
    field added by injection is also caught.
    """
    both = set(clean_raw.adjudication or {}) | set(raw.adjudication or {})
    return sorted(both - set(ADJ_BOOKKEEPING))


def _adj_diff(a: dict, b: dict) -> str:
    """Prints only the keys that changed. Printing the whole dict would flood the
    report with the large `ddx` blob and bury the actual difference from the reader."""
    d = [k for k in a if a.get(k) != b.get(k)]
    return "unchanged" if not d else "; ".join(f"{k}: {a.get(k)!r} → {b.get(k)!r}"[:160] for k in d)


def check_conclusion_invariance(clean_raw, raw, T: int) -> dict:
    """Daily-event injection "before vs. after": ground-truth fields must be byte-identical
    and the offline solver's conclusion must be unchanged."""
    checks = {}
    checks["outcome_label_unchanged"] = _ck(clean_raw.outcome_label == raw.outcome_label,
                                            f"{clean_raw.outcome_label} → {raw.outcome_label}")
    checks["gold_drivers_unchanged"] = _ck(list(clean_raw.gold_drivers) == list(raw.gold_drivers),
                                           f"{clean_raw.gold_drivers} → {raw.gold_drivers}")
    checks["label_rule_unchanged"] = _ck(clean_raw.label_rule == raw.label_rule, "")
    a = {k: clean_raw.adjudication.get(k) for k in _adj_gold_keys(clean_raw, raw)}
    b = {k: raw.adjudication.get(k) for k in _adj_gold_keys(clean_raw, raw)}
    checks["adjudication_core_unchanged"] = _ck(a == b, f"{_adj_diff(a, b)}")
    checks["reversal_points_unchanged"] = _ck(
        (getattr(clean_raw, "reversal_points", None) or []) == (getattr(raw, "reversal_points", None) or []),
        f"{len(getattr(clean_raw, 'reversal_points', None) or [])} → "
        f"{len(getattr(raw, 'reversal_points', None) or [])}")

    c0, c1 = _conclusion(clean_raw, T), _conclusion(raw, T)
    for name in c0:
        same = (c0[name]["risk_category"] == c1[name]["risk_category"]
                and c0[name]["top_driver"] == c1[name]["top_driver"])
        checks[f"conclusion_same:{name}"] = _ck(
            same, f"{c0[name]['risk_category']}/{c0[name]['top_driver']} → "
                  f"{c1[name]['risk_category']}/{c1[name]['top_driver']}")
    return {"checks": checks, "ok": all(c["ok"] for c in checks.values()),
            "before": c0, "after": c1}


def check_event_spread(rows: list[dict], T: int, first_day: int = 7) -> dict:
    """Benign events must not cluster in the last third before T (a timing hint).
    The window is [first_day, T], as the generator starts placing events at day 7.
    """
    days = [r["day"] for r in rows if r["kind"] != "real_symptom"]
    if len(days) < 3:
        return {"checks": {"benign_spread_uniform": _ck(True, f"n={len(days)}, not checked")}, "ok": True}
    span = max(3, T - first_day)
    third = span / 3.0
    buckets = [sum(1 for d in days if first_day + i * third <= d < first_day + (i + 1) * third)
               for i in range(3)]
    buckets[2] += sum(1 for d in days if d >= first_day + 3 * third)
    late = buckets[2] / len(days)
    return {"checks": {"benign_spread_uniform": _ck(
        late <= 0.6, f"window [{first_day},{T}] thirds {buckets} · last-third share {late:.2f}")},
        "ok": late <= 0.6}


# ================================================================ (3) Scan of the text visible to the solver


# Shape of an EV id: `EV-<case_id>-<slot>` (slot looks like S1/B12/L1/D3/07). Strip these out
# before scanning for aliases.
_EV_ID_RE = __import__("re").compile(r"EV-[A-Za-z0-9]+-[A-Za-z0-9]+-?[A-Za-z0-9]*")


def scan_solver_text(raw, T: int, prompt_frame: str | None = None,
                     ddx_aliases=None, *, solver_payload=None) -> dict:
    """Scan the solver-visible payload. The case text is scanned for everything; the
    prompt template only for ground-truth tokens (its driver list is candidates).
    """
    if solver_payload is None:
        sp, _ = build_instance(raw, T)
    else:
        sp = solver_payload
    ok_leak, viol = leakage_probe(sp, T)                  # kernel gate: temporal + structural fields
    blob = sp.dumps()
    low = blob.lower()
    findings = [f"kernel_leakage_probe:{v}" for v in viol]
    findings += [f"truth_token:{t}" for t in TRUTH_TOKENS if t in low]
    findings += [f"truth_word_zh:{w}" for w in TRUTH_WORDS_ZH if w in blob]
    findings += [f"driver_word_zh:{w}" for w in DRIVER_WORDS_ZH if w in blob]
    findings += [f"driver_name:{d}" for d in ALLOWED_DRIVERS if d in low]
    from .events import alias_hit
    # EV ids are masked before alias scanning: slot names such as `B12` can collide
    # with clinical aliases. Known gap: the scan runs before id anonymization.
    _blob_no_ids = _EV_ID_RE.sub("<EVID>", blob)
    findings += [f"ddx_alias:{a}" for a in alias_hit(_blob_no_ids, ddx_aliases)]
    from .events import JOIN_STRUCTURE_MARKERS       # single source of truth for this word list
    findings += [f"join_structure_word:{w}" for w in JOIN_STRUCTURE_MARKERS if w in blob]
    if prompt_frame:
        findings += [f"prompt_frame_truth_token:{t}" for t in TRUTH_TOKENS
                     if t in prompt_frame.lower()]
    return {"ok": not findings, "findings": findings, "n_chars": len(blob),
            "n_signals": len(sp.longitudinal_data), "n_evidence": len(sp.evidence_ledger)}


# ================================================================ Aggregate entry point
def _ddx_aliases(cs) -> list[str]:
    """Alias set for leakage scanning: the diagnosis name, its synonyms and the
    registry's `leak_only` terms (`overlay.leak_aliases_for`). Thread aliases are
    excluded: they name clinical findings the patient legitimately reports.
    """
    d = getattr(cs, "ddx", None) or {}
    from .overlay import leak_aliases_for
    al = leak_aliases_for(d.get("spec_id"), d.get("aliases"))
    return al + ([str(d["diagnosis"])] if d.get("diagnosis") else [])


def verify_case(clean_raw, raw, cs, premise, manifest: dict, driver: str,
                rev_day: int | None, prompt_frame: str | None = None) -> dict:
    """Full verification for one case. Returns a report dict; `bad_items` are the item ids
    that need to be discarded and reinjected."""
    f = facts_of(cs)
    T = int(manifest["T"])
    ed = manifest.get("event_density", {})
    real_days = {r["day"] for r in manifest["events"] if r["kind"] == "real_symptom"}

    items: list[dict] = []
    # The insufficient-information tier comes from `latent.ddx_insufficient`, not from T.
    _insuff = bool((cs.latent or {}).get("ddx_insufficient"))
    _view = (manifest or {}).get("_injector_view") or {}
    for row in manifest["streams"]:
        items.append(check_stream(row, raw.longitudinal_data.get(row["name"], []),
                                 f, driver, T, rev_day, ed, insufficient_tier=_insuff,
                                 neutral_pts=_view.get(row["name"])))
    seen_topics: set[str] = set()          # dedupe in order of appearance: the first
                                            # occurrence stands, later repeats fail
    for row in manifest["events"]:
        items.append(check_event(row, f, driver, T,
                                 real_days - {row["day"]} if row["kind"] == "real_symptom"
                                 else real_days, other_topics=set(seen_topics),
                                 ddx_aliases=_ddx_aliases(cs)))
        tp = (_pool_item(row.get("text", ""), row.get("claim")) or {}).get("topic")
        if tp:
            seen_topics.add(tp)

    # With the physiology layer on, conclusion invariance compares against the
    # noise-free view, so observation noise alone cannot flip the offline solver.
    _clean_ld_v = (manifest or {}).get("_clean_longitudinal_data")
    if _clean_ld_v is None:
        _raw_inv = raw
    else:
        import copy as _copy
        _raw_inv = _copy.copy(raw)
        _raw_inv.longitudinal_data = _clean_ld_v
    inv = check_conclusion_invariance(clean_raw, _raw_inv, T)
    spread = check_event_spread(manifest["events"], T)
    text = scan_solver_text(raw, T, prompt_frame, ddx_aliases=_ddx_aliases(cs))

    bad_items = [i["item"] for i in items if not i["ok"]]
    case_level = {**inv["checks"], **spread["checks"],
                  "solver_text_no_answer_info": _ck(text["ok"], "; ".join(text["findings"][:4]))}
    rep = {
        "case_id": cs.case_id, "T": T, "driver": driver, "reversal_day": rev_day,
        "n_items": len(items), "n_bad_items": len(bad_items), "bad_items": bad_items,
        "items": items, "case_checks": case_level,
        "conclusion": {"before": inv["before"], "after": inv["after"]},
        "solver_text": text, "skipped": manifest.get("skipped", []),
        "ok": (not bad_items) and all(c["ok"] for c in case_level.values()),
    }
    log.info("[verify] %s %d item(s) · failed %d · case-level %s · text %s",
             cs.case_id, len(items), len(bad_items),
             "OK" if all(c["ok"] for c in case_level.values()) else "FAIL",
             "CLEAN" if text["ok"] else f"LEAK{text['findings'][:2]}")
    return rep


def dumps(rep: dict) -> str:
    return json.dumps(rep, ensure_ascii=False)
