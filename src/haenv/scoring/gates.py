"""gates.py -- question-generation gates that run on haenv's side of the kernel.

`verify.py` checks injected items one by one; this module checks the primary signal and
case- and batch-level consistency. Each hit carries a severity: "gate" blocks emission of
the case, "warn" is recorded in the audit and log only.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import collections
from haenv.world.validation.gates_case import A5_MIN_CLASS
from haenv.world.validation.gates_case import ANCHOR_TOL_KG
from haenv.world.validation.gates_case import CONDITION_ALIASES
from haenv.world.validation.gates_case import DEVICE_SIGNALS
from haenv.world.validation.gates_case import EV_ID_OPAQUE_SEVERITY
from haenv.world.validation.gates_case import EXEMPT
from haenv.world.validation.gates_case import GEN15_MIN_WEIGHT_DELTA_KG
from haenv.world.validation.gates_case import GEN15_SEVERITY
from haenv.world.validation.gates_case import GEN15_SNR_K
from haenv.world.validation.gates_case import GEN18_MIN_ARITHMETIC_POINTS
from haenv.world.validation.gates_case import GEN18_MIN_REAL_SYMPTOMS
from haenv.world.validation.gates_case import GEN18_SEVERITY
from haenv.world.validation.gates_case import GEN19_SEVERITY
from haenv.world.validation.gates_case import GEN20_SEVERITY
from haenv.world.validation.gates_case import GEN22_SEVERITY
from haenv.world.validation.gates_case import GEN23_SEVERITY
from haenv.world.validation.gates_case import GEN24_SEVERITY
from haenv.world.validation.gates_case import GEN25_SEVERITY
from haenv.world.validation.gates_case import GEN25_TOL
from haenv.world.validation.gates_case import GEN26_SEVERITY
from haenv.world.validation.gates_case import GEN27_SEVERITY
from haenv.world.validation.gates_case import GEN28_SEVERITY
from haenv.world.validation.gates_case import GEN29_SEVERITY
from haenv.world.validation.gates_case import GEN6_MISSING_TOL
from haenv.world.validation.gates_case import GEN6_SEVERITY
from haenv.world.validation.gates_case import GEN7_GATE_DEVICES
from haenv.world.validation.gates_case import GEN7_SEVERITY
from haenv.world.validation.gates_case import GEN8D_MAX_SHARE
from haenv.world.validation.gates_case import GEN8D_MIN_CASES
from haenv.world.validation.gates_case import GEN8_SEVERITY
from haenv.world.validation.gates_case import GEN_REGISTRY_GATE_KINDS
from haenv.world.validation.gates_case import GEN_REGISTRY_SEVERITY
from haenv.world.validation.gates_case import LEDGER_VALUE_STREAMS
from haenv.world.validation.gates_case import MISSINGNESS_MIN_GRID
from haenv.world.validation.gates_case import MISSINGNESS_SPLIT_TOL
from haenv.world.validation.gates_case import PREMISE_REGISTRY
from haenv.world.validation.gates_case import STREAM_SHORT_TOL_DAYS
from haenv.world.validation.gates_case import TRAP_EVIDENCE_MIN_FRACTION
from haenv.world.validation.gates_case import _EV_OPAQUE_RE
from haenv.world.validation.gates_case import _REG_DIR
from haenv.world.validation.gates_case import _clinical_cv_of
from haenv.world.validation.gates_case import _gen_reg
from haenv.world.validation.gates_case import _judging_registry
from haenv.world.validation.gates_case import _match
from haenv.world.validation.gates_case import _noise_windows
from haenv.world.validation.gates_case import _premise_paths
from haenv.world.validation.gates_case import _spans_window
from haenv.world.validation.gates_case import check_anchors_honored
from haenv.world.validation.gates_case import check_cadence
from haenv.world.validation.gates_case import check_clinical_baseline_cohort
from haenv.world.validation.gates_case import check_clinical_coupling
from haenv.world.validation.gates_case import check_comorbidity_vocab
from haenv.world.validation.gates_case import check_coupling_observable
from haenv.world.validation.gates_case import check_demographic_plausibility
from haenv.world.validation.gates_case import check_device_inventory
from haenv.world.validation.gates_case import check_dosing_consistency
from haenv.world.validation.gates_case import check_drug_indication
from haenv.world.validation.gates_case import check_ev_id_opaque
from haenv.world.validation.gates_case import check_event_density
from haenv.world.validation.gates_case import check_gold_line_in_known_conditions
from haenv.world.validation.gates_case import check_hardwired_horizons
from haenv.world.validation.gates_case import check_ledger_values_traceable
from haenv.world.validation.gates_case import check_missingness
from haenv.world.validation.gates_case import check_observed_in_domain
from haenv.world.validation.gates_case import check_premise_registry
from haenv.world.validation.gates_case import check_rhythm_gap_feasible
from haenv.world.validation.gates_case import check_sex_consistency
from haenv.world.validation.gates_case import check_stream_horizons
from haenv.world.validation.gates_case import check_symptom_day_separability
from haenv.world.validation.gates_case import check_trap_evidence
from haenv.world.validation.gates_case import clinical_coupling_judgeable
from haenv.world.validation.gates_case import declared_course_end
from haenv.world.validation.gates_case import log
from haenv.world.validation.gate_tables import DRUG_DOSES_PER_WEEK
from haenv.world.validation.gate_tables import SEX_EXCLUSIVE
from haenv.world.validation.gates_outcome import CLEAN_REF_OF
from haenv.world.validation.gates_outcome import DRIVER_REQUIRED_OBSERVABLE
from haenv.world.validation.gates_outcome import GEN13_GATE_KINDS
from haenv.world.validation.gates_outcome import GEN13_SEVERITY
from haenv.world.validation.gates_outcome import GEN14_FLAT_REL_SPAN
from haenv.world.validation.gates_outcome import GEN14_LAB_EXEMPT_TRAJECTORIES
from haenv.world.validation.gates_outcome import GEN14_SEVERITY
from haenv.world.validation.gates_outcome import NO_OBSERVABLE_REQUIRED
from haenv.world.validation.gates_outcome import PERSISTENT_OFFSET_NOISE
from haenv.world.validation.gates_outcome import SMOOTHING_ROLLING_MEDIAN_7D
from haenv.world.validation.gates_outcome import SMOOTHING_ROLLING_MEDIAN_7PT
from haenv.world.validation.gates_outcome import _LAB_VAL_RE
from haenv.world.validation.gates_outcome import _NEEDS_SYMPTOM
from haenv.world.validation.gates_outcome import _NoObservableRequired
from haenv.world.validation.gates_outcome import _SMOOTHERS
from haenv.world.validation.gates_outcome import _declared_lab_readings
from haenv.world.validation.gates_outcome import _flat_pre_T
from haenv.world.validation.gates_outcome import _gen13
from haenv.world.validation.gates_outcome import _gold_lab_signal_leaks
from haenv.world.validation.gates_outcome import _gold_lab_signal_missing
from haenv.world.validation.gates_outcome import _median
from haenv.world.validation.gates_outcome import _real_symptom_evidence
from haenv.world.validation.gates_outcome import _rolling_median_7d
from haenv.world.validation.gates_outcome import _rolling_median_7pt
from haenv.world.validation.gates_outcome import check_gold_coverage
from haenv.world.validation.gates_outcome import check_outcome_derivable
from haenv.world.validation.gates_outcome import derive_outcome
from haenv.world.validation.gates_outcome import is_declared_no_observable
from haenv.world.validation.gates_outcome import label_rule_floor_kg
from haenv.world.validation.gates_outcome import label_rule_params
from haenv.world.validation.gates_outcome import label_series
from haenv.world.validation.gates_shortcut import A5_ALPHA
from haenv.world.validation.gates_shortcut import A5_DISEASE_BITS
from haenv.world.validation.gates_shortcut import A5_F1
from haenv.world.validation.gates_shortcut import A5_MARGIN_OVER_MAJORITY
from haenv.world.validation.gates_shortcut import A5_MIN_CASES
from haenv.world.validation.gates_shortcut import A5_MIN_REL_SPAN
from haenv.world.validation.gates_shortcut import A5_N_PERM
from haenv.world.validation.gates_shortcut import A5_P_SAMPLES
from haenv.world.validation.gates_shortcut import A5_SEVERITY
from haenv.world.validation.gates_shortcut import FOOTPRINT_PHI_FLOOR
from haenv.world.validation.gates_shortcut import FOOTPRINT_PHI_NONZERO
from haenv.world.validation.gates_shortcut import NONCLINICAL_FEATURES
from haenv.world.validation.gates_shortcut import NONCLINICAL_SUFFIXES
from haenv.world.validation.gates_shortcut import OUTCOME_REQUIRED_OBSERVABLE
from haenv.world.validation.gates_shortcut import TIER_SURFACE_ALPHA
from haenv.world.validation.gates_shortcut import TIER_SURFACE_AUC_MARGIN
from haenv.world.validation.gates_shortcut import TIER_SURFACE_MIN_CLASS
from haenv.world.validation.gates_shortcut import TIER_SURFACE_N_PERM
from haenv.world.validation.gates_shortcut import TIER_SURFACE_SEVERITY
from haenv.world.validation.gates_shortcut import _LEVEL_SUFFIXES
from haenv.world.validation.gates_shortcut import _PERM_CACHE
from haenv.world.validation.gates_shortcut import _SPLIT_ORDER_CACHE
from haenv.world.validation.gates_shortcut import _SPLIT_ORDER_CACHE_MAX_CELLS
from haenv.world.validation.gates_shortcut import _ThresholdScan
from haenv.world.validation.gates_shortcut import _VARIANT_SUFFIX_RE
from haenv.world.validation.gates_shortcut import _auc_of
from haenv.world.validation.gates_shortcut import _best_single_threshold_f1
from haenv.world.validation.gates_shortcut import _best_single_threshold_f1_scan
from haenv.world.validation.gates_shortcut import _face_labs
from haenv.world.validation.gates_shortcut import _item_profile_scores
from haenv.world.validation.gates_shortcut import _lab_features
from haenv.world.validation.gates_shortcut import _lab_flags
from haenv.world.validation.gates_shortcut import _mann_whitney
from haenv.world.validation.gates_shortcut import _perm_floor
from haenv.world.validation.gates_shortcut import _split_order
from haenv.world.validation.gates_shortcut import _split_p
from haenv.world.validation.gates_shortcut import _target_key
from haenv.world.validation.gates_shortcut import case_features
from haenv.world.validation.gates_shortcut import check_footprint_not_discriminative
from haenv.world.validation.gates_shortcut import check_shortcut
from haenv.world.validation.gates_shortcut import check_tier_surface
from haenv.world.validation.gates_shortcut import classify_shortcut
from haenv.world.validation.gates_shortcut import footprint_discriminability
from haenv.world.validation.gates_shortcut import near_constant
from haenv.world.validation.gates_shortcut import near_constant_feature
from haenv.world.validation.gates_shortcut import tier_surface_features


def batch_gate_blockers(hits: list[dict]) -> list[dict]:
    """Batch-level hits that block emission: exactly those with `severity == "gate"`, whichever
    check produced them. Severities are limited to `BATCH_GATE_SEVERITIES`.
    """
    return [w for w in (hits or []) if w.get("severity") == "gate"]


#: Allowed batch-level severities.
BATCH_GATE_SEVERITIES = ("gate", "warn", "info")


def check_batch(built: dict) -> list[dict]:
    """GEN8: batch-level minimum diversity (target events, course lengths, symptom-day
    signatures, distractor text). built = {case_id: RawCase}.
    """
    hits: list[dict] = []
    if len(built) < 2:
        return hits

    targets = {r.prediction_context.get("target_event_type") for r in built.values()}
    if len(targets) < 2:
        hits.append({"kind": "target_event_degenerate", "severity": GEN8_SEVERITY,
                     "detail": f"all {len(built)} cases' target_event_type is {targets}"})

    # ---------- course-length diversity, on the declared course end ----------
    # The whole-stream max(ts) is constant by construction (`T + prediction_window`), so it is
    # reported only as a side note.
    declared, raw_ends = set(), set()
    for r in built.values():
        declared.add(declared_course_end(r))
        pts = [p for v in r.longitudinal_data.values() for p in v]
        if pts:
            raw_ends.add(max(int(p["ts"]) for p in pts))
    declared.discard(None)
    if len(declared) < 3:
        hits.append({"kind": "series_length_degenerate", "severity": GEN8_SEVERITY,
                     "detail": f"{len(built)} cases' declared course end takes only these "
                               f"{len(declared)} value(s): {sorted(declared)} (side-by-side diagnostic quantity: whole-stream max(ts) = "
                               f"{sorted(raw_ends)} -- the latter is guaranteed by the T+prediction_window identity, "
                               f"must not be read as course length)"})

    # GEN8c: true-symptom day sets must vary across cases; a shared set would act as a lookup
    # table across the pack. Judged on the uniqueness rate (< 0.80 fails).
    import haenv.world.wq as wq
    sigs = []
    for cid, r in built.items():
        man = wq.injected_manifest(cid, required=False)
        real = set(man.get("real_symptom_evidence_ids") or [])
        if not real:
            continue
        days = tuple(sorted(int(e.get("source_timestamp", -1)) for e in (r.evidence_ledger or [])
                            if str(e.get("evidence_id")) in real))
        if days:
            sigs.append(days)
    if len(sigs) >= 5:
        uniq = len(set(sigs)) / len(sigs)
        if uniq < 0.8:
            top = collections.Counter(sigs).most_common(1)[0]
            hits.append({"kind": "symptom_day_signature_degenerate", "severity": GEN8_SEVERITY,
                         "detail": f"{len(sigs)} cases' true-symptom-day signatures take only {len(set(sigs))} distinct value(s)"
                                   f" (uniqueness rate {uniq:.2f} < 0.80); the most common one {top[0]} appears {top[1]} times"
                                   f" -- \"day ∈ that set => true symptom\" would become a lookup table usable across cases"})

    # ---- GEN8d: cross-case reuse of distractor text --------------
    # A distractor is any ledger item outside `real_symptom_evidence_ids`; cases whose ledger
    # is missing are not counted. A5 scans numbers only, so text reuse is checked here.
    texts: collections.Counter = collections.Counter()
    n_scanned = 0
    for cid, r in built.items():
        man = wq.injected_manifest(cid, required=False)
        real = set(man.get("real_symptom_evidence_ids") or [])
        if not real:
            continue  # no ledger => not judgeable
        seen = {str(e.get("symptom") or e.get("note") or "").strip()
                for e in (r.evidence_ledger or [])
                if str(e.get("evidence_id")) not in real}
        seen.discard("")
        if not seen:
            continue
        n_scanned += 1
        for t in sorted(seen):
            texts[t] += 1
    if n_scanned >= GEN8D_MIN_CASES and texts:
        # Ties break on the text, so the report does not depend on set (hash) order.
        top_text, top_n = min(texts.items(), key=lambda kv: (-kv[1], kv[0]))
        share = top_n / n_scanned
        if share > GEN8D_MAX_SHARE:
            # Effective item count exp(H) (Hill number of order 1), reported beside the maximum share.
            import math as _math
            _tot = sum(texts.values())
            _H = -sum((c / _tot) * _math.log(c / _tot) for c in texts.values())
            # Name the pool responsible: the kernel's distractor pool (`haenv_kernel/noise.py`) is separate
            # from `registry/benign_events.yaml`.
            from haenv.world.observations.events import inherited_profile as _inh
            _owner = "the kernel's inherited pool (`haenv_kernel/noise.py`; `registry/benign_events.yaml` does not affect it => change it there, which moves the generation stamp)" \
                if _inh(top_text) is not None else "`registry/benign_events.yaml`"
            hits.append({"kind": "distractor_text_overused", "severity": GEN8_SEVERITY,
                         "detail": f"{n_scanned} judgeable cases, {len(texts)} distinct distractor texts"
                                   f" (effective item count {_math.exp(_H):.1f}); "
                                   f"the most common, \"{top_text}\", appears in {top_n} cases = "
                                   f"{share:.2f} > {GEN8D_MAX_SHARE} -- "
                                   f"\"text ∈ the distractor pool => ignorable\" would become a lookup table usable across batches. "
                                   f"Source: {_owner}"})
    elif n_scanned < GEN8D_MIN_CASES:
        # Too few cases: reported, not passed.
        hits.append({"kind": "distractor_text_scan_empty", "severity": "info",
                     "detail": f"{n_scanned} judgeable cases < {GEN8D_MIN_CASES} => GEN8d has zero objects on this batch"
                               f" (not a pass)"})
    return hits


# ============================================================ the reason ledger for unemitted cases
def blocked_reasons(audits) -> dict[str, list[str]]:
    """Unemitted cases -> why each was blocked.

    Pre-generation reasons first (`latent_blocking`, `declaration_blocking`), then later gates
    in reverse pipeline order (the later gate is more specific): post_noise_conflicts,
    verify_bad, last_conflicts, premise_error, gen_error. A case with no recorded reason says
    so explicitly.
    """
    out: dict[str, list[str]] = {}
    for a in (audits or []):
        if not isinstance(a, dict) or a.get("emitted"):
            continue
        why = ([f"latent:{k}" for k in (a.get("latent_blocking") or [])]
               or [f"declaration:{k}" for k in (a.get("declaration_blocking") or [])]
               or list(a.get("post_noise_conflicts") or [])
               or [f"verify:{k}" for k in (a.get("verify_bad") or [])]
               or [f"gv1:{k}" for k in (a.get("last_conflicts") or [])]
               or ([f"premise:{str(a['premise_error'])[:60]}"] if a.get("premise_error") else [])
               or ([f"gen_error:{str(a['gen_error'])[:60]}"] if a.get("gen_error") else []))
        out[str(a.get("case_id", "?"))] = why or ["(no reason recorded -- this is itself a defect)"]
    return out


def dropped_items(audits) -> dict[str, list[str]]:
    """Emitted cases -> items dropped from them in the injection-retry loop.

    A class of item that can never pass is dropped from every case while verification still
    reports 0 failures, so the drops are persisted. Unemitted cases are `blocked_reasons`'.
    """
    out: dict[str, list[str]] = {}
    for a in audits or ():
        if not a.get("emitted"):
            continue
        d = sorted(a.get("event_dropped") or ())
        if d:
            out[str(a.get("case_id"))] = d
    return out


def dropped_rates(audits) -> dict[str, dict]:
    """Drop rate per item, `{item: {"n", "of", "rate"}}`; `rate=1.00` marks a structural drop."""
    emitted = [a for a in (audits or ()) if a.get("emitted")]
    n_em = len(emitted)
    cnt: dict[str, int] = {}
    for a in emitted:
        for k in set(a.get("event_dropped") or ()):
            cnt[k] = cnt.get(k, 0) + 1
    return {k: {"n": v, "of": n_em, "rate": round(v / n_em, 4) if n_em else None}
            for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}


def blocked_counts(audits) -> dict[str, int]:
    """Counts of unemitted reasons by kind (separate from warn-level per-case counts)."""
    c: dict[str, int] = {}
    for ks in blocked_reasons(audits).values():
        for k in ks:
            c[k] = c.get(k, 0) + 1
    return c


# ---------------------------------------------------------------- two gates on the findings layer
# A4 (discriminability) and N1 (single-feature unsolvability) pull in opposite directions:
# A4 alone allows lookup questions, N1 alone allows unsolvable ones.

FINDINGS_A4_SEVERITY = "warn"  # legitimate questions hit it (see docstring)
FINDINGS_N1_SEVERITY = "warn"   # legitimate questions hit it


def check_rival_discriminable(spec_id: str, rivals, profile) -> list[dict]:
    """A4: every near-miss rival has a discriminating finding in the gold's findings profile.

    Warn, not gate: some rivals are not yet labeled, and rivals ruled out by routine workup
    alone (`no_order_needed`) are a difficulty issue.
    """
    hits: list[dict] = []
    prof = {p["id"]: p for p in (profile or {}).get("findings") or ()}
    if not prof:
        return [{"kind": "findings_profile_missing", "spec_id": spec_id,
                 "detail": "this condition has no findings profile -- unjudgeable, not a pass"}]
    n_order = 0
    for r in (rivals or ()):
        df, un = r.get("discriminator_finding"), r.get("discriminator_unresolvable")
        if un:
            hits.append({"kind": "rival_unresolvable", "spec_id": spec_id,
                         "rival": r.get("name"), "detail": str(un)[:80]})
            continue
        if not df:
            hits.append({"kind": "rival_discriminator_unlabeled", "spec_id": spec_id,
                         "rival": r.get("name")})
            continue
        p = prof.get(str(df.get("finding")))
        if p is None:
            hits.append({"kind": "rival_discriminator_not_in_profile", "spec_id": spec_id,
                         "rival": r.get("name"), "detail": str(df.get("finding"))})
        elif p["role"] != "screening":
            n_order += 1
    if (rivals or ()) and n_order == 0:
        hits.append({"kind": "no_order_needed", "spec_id": spec_id,
                     "detail": "every near-miss rival can be ruled out using routine initial-visit workup alone -- this question can complete the differential without ordering anything"})
    return hits


def check_single_finding_solvable(spec_id: str, rivals) -> list[dict]:
    """N1: no single finding may rule out every near-miss rival at once.

    A hit means the question tests which item to order, not weighing evidence. The fix is a
    rival that shares the discriminating item, not a fabricated discriminator.
    """
    ds = [r.get("discriminator_finding") for r in (rivals or ())]
    ds = [d for d in ds if d]
    if len(ds) < 2 or len(ds) != len(rivals or ()):
        return []  # unlabeled entries: not judgeable
    fids = {str(d.get("finding")) for d in ds}
    if len(fids) == 1:
        return [{"kind": "single_finding_solvable", "spec_id": spec_id,
                 "detail": f"all {len(ds)} near-miss rivals are ruled out by {fids.pop()} -- checking one item gives the answer"}]
    return []
