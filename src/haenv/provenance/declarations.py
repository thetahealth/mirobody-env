"""Formal source segments, import roles and immutable historical move proofs.

Canonical source labels omit the physical src/ prefix. Keep historical proof
text unchanged; a new layout is declared explicitly for revision readers.
"""
from __future__ import annotations

SOURCE_LAYOUT = "src"
CRITERION_DECLARATIONS = "haenv/provenance/criteria.py"

#: Generation segment: changing a file here changes the world (`world_sha`) and the
#: pack must be regenerated. A registry yaml belongs here when its strings reach the
#: prompt without passing through `inputs/*.job.yaml`, which `job_yaml_sha256` covers.
GENERATION = (
    'registry/p2_complaint_pool.yaml', 'registry/p2_complaint_prior.yaml',
    "registry/p3_clinical_sources.yaml", "registry/p4_clinical_sources.yaml",
    'haenv/scoring/management/plan.py',
    "haenv/m2/workup_generation.py",
    "haenv/m2/diversity.py", "registry/m2_diversity.yaml",
    "haenv/m2/confirmation.py", "registry/m2_confirmation.yaml", 'haenv/world/generation/twins.py',
    'haenv/world/validation/difficulty_r2.py', 'haenv/world/history/history_r2.py', "haenv/m2/history_r2.py",
    "haenv/m2/__init__.py", "registry/difficulty_r2.yaml",
    "haenv_kernel/build.py", "haenv_kernel/latent.py", "haenv_kernel/synth.py", "haenv_kernel/noise.py",
    "haenv_kernel/joint_scenarios.py",
    'haenv/world/generation/build.py', 'haenv/world/generation/build_weight.py', 'haenv/world/generation/build_clinical.py',
    'haenv/world/observations/events.py', 'haenv/world/observations/events_text.py', 'haenv/world/observations/events_pools.py', 'haenv/world/generation/narration.py',
    'haenv/world/observations/events_streams.py', 'haenv/world/observations/findings_render.py',
    'haenv/world/clinical/relations.py', 'haenv/scoring/verify.py', 'haenv/world/overlay.py',
    'haenv/resources/registry.py', 'haenv/world/generation/rng.py',
    'haenv/world/observations/wearable.py', 'haenv/world/clinical/med_course.py', "registry/world_history.yaml",
    "haenv/m2/background.py", "registry/background_m2.yaml",
    "registry/m2_pre_review.yaml",
    'haenv/world/generation/framings.py',
    'haenv/resources/yamlcache.py',
    'haenv/resources/regpath.py',
    'haenv/world/generation/world_plugins.py',
    'haenv/world/clinical/post_inject.py',
    'haenv/world/clinical/latent_rules.py',
    'haenv/app/job.py',
    'haenv/interaction/gated.py',
    "registry/findings.yaml", "registry/condition_findings.yaml",
    "registry/panel_background.yaml",
    "registry/panel_copula.yaml",
    "registry/findings_upstream.yaml",
    "registry/lookalikes.yaml", "registry/vocab_context.yaml",
    "registry/condition_aliases.yaml",
    "registry/symptom_annotations.yaml",
    "registry/symptom_topics.yaml", "registry/symptom_penetrance.yaml", "registry/association_clusters.yaml",
    "registry/physio_kernels.yaml",
    "registry/physio_streams.yaml", "registry/comorbid_coupling.yaml",
    "registry/benign_events.yaml", "registry/symptom_lay.yaml", "registry/narration.yaml",
    "registry/clinical_expansion.yaml",
    "registry/drug_side_effects.yaml",
    "registry/condition_age_limits.yaml",
    "registry/clinical_baselines.yaml",
    "registry/indicators.yaml",
    "registry/streams.yaml", 'haenv/world/observations/streams.py',
    'haenv/world/physio/__init__.py', 'haenv/world/physio/apply.py', 'haenv/world/physio/audit.py',
    'haenv/world/physio/bounds.py', 'haenv/world/physio/coupling.py', 'haenv/world/physio/kernel.py',
    'haenv/world/physio/noise.py', 'haenv/world/physio/superpose.py',
    'haenv/world/clinical/indicators.py',
    "registry/artifact_rates.yaml", 'haenv/world/observations/artifact_params.py',
    "registry/drug_effects.yaml", 'haenv/world/clinical/drug_effects.py',
    # split out of `overlay.py` / `job.py` / `drug_effects.py` / `events.py` (2026-10-03),
    # moved verbatim
    'haenv/scoring/matching/excludes.py', 'haenv/contracts/job_schema.py', 'haenv/contracts/drug_schema.py',
    'haenv/world/generation/world_knobs.py',
    # Gold qualifiers (core + qualifier split, derivability on the visible instance).
    "registry/gold_qualifiers.yaml", 'haenv/world/clinical/qualifiers.py',
    'haenv/contracts/__init__.py',
    'haenv/contracts/pack.py',
    'haenv/contracts/hooks.py',
    'haenv/app/pack_hooks.py',
    'haenv/app/packs.py',
    'haenv/world/payload_policies.py',
    'haenv/m2/symptom_timing.py',
    'haenv/m2/audit.py',
    'haenv/m2/item_audit.py',
    'haenv/m2/audit_classifier.py',
    'haenv_kernel/__init__.py',
    'haenv/m2/emr.py',
    'haenv/world/validation/pack_size.py',
)
#: Judging segment: changing a file here changes readings recomputed from saved
#: responses, so affected batches must be recomputed. Anything a member imports (code or
#: registry data) must be in the segment or in `JUDGING_IMPORT_ALLOW` /
#: `REGISTRY_YAML_OUT_OF_SCOPE` with a reason.
MANAGEMENT_JUDGING = (
    # Independent saved-answer rubric, deliberately outside public JUDGING.
    'haenv/scoring/management/controls_v62.py',
    'haenv/scoring/management/rubric.py', 'haenv/scoring/management/judge.py', 'haenv/scoring/management/pipeline.py',
    'haenv/scoring/management/estimate.py', "registry/management_rubric.yaml",
    'haenv/scoring/management/transport.py',
    'haenv/scoring/management/precision.py', 'haenv/scoring/management/precision_rejudge.py', "registry/test_indications.yaml",
    "registry/management_adapter_rules.yaml", "registry/p4_diversity.yaml",
    'haenv/scoring/management/adapters/__init__.py', 'haenv/scoring/management/adapters/p2.py',
    'haenv/scoring/management/adapters/common.py',
    'haenv/scoring/management/adapters/p3.py', 'haenv/scoring/management/adapters/p4.py', 'haenv/scoring/management/adapters/m2.py',
)
JUDGING = (
    "haenv_kernel/verifier.py", "haenv_kernel/gatekeeper.py", "haenv_kernel/schema.py",
    "haenv_kernel/runner.py", "haenv_kernel/solver.py",
    'haenv/scoring/llm_rubric.py', "registry/rubric_points.yaml",
    'haenv/scoring/semantic/judge.py', 'haenv/scoring/semantic/inputs.py', 'haenv/scoring/semantic/rubric.py',
    'haenv/scoring/semantic/pipeline.py',
    'haenv/scoring/semantic/report.py',
    'haenv/scoring/judge_compaction.py',
    'haenv/scoring/judge_evidence.py',
    'haenv/scoring/semantic/visibility.py', 'haenv/scoring/semantic/corrections.py',
    'haenv/scoring/semantic/lean.py', "registry/semantic_judging_v4.yaml",
    'haenv/scoring/semantic/atom_correction.py', "registry/semantic_judging_second.yaml",
    "registry/semantic_judging_why.yaml", "registry/semantic_judging_why_lean.yaml",
    "registry/semantic_judging_source.yaml",
    "registry/semantic_judging.yaml",
    'haenv/scoring/mounts/dispatch.py',
    'haenv/resources/yamlcache.py',
    'haenv/world/clinical/latent_rules.py',
    "haenv/judges/__init__.py", "haenv/judges/_helpers.py",
    "haenv/judges/differential.py", "haenv/judges/outcome.py",
    "haenv/judges/trajectory.py", "haenv/judges/safety.py",
    "haenv/judges/llm.py", "haenv/judges/recall_match.py",
    'haenv/scoring/gates.py', 'haenv/world/validation/gates_case.py', 'haenv/world/validation/gates_outcome.py', 'haenv/world/validation/gates_shortcut.py', 'haenv/scoring/legacy.py',
    'haenv/scoring/process.py', 'haenv/world/wq.py', "haenv/report.py",
    "haenv/analytics.py",
    'haenv/app/evaluate.py',
    # split out of `evaluate.py` (2026-10-03); same judging code, moved verbatim
    "haenv/runtime/state.py", 'haenv/interaction/qside.py', "haenv/runtime/ledger.py", 'haenv/runtime/solvers.py',
    'haenv/scoring/rows.py',
    # definitions moved down so the import graph has no cycle through the judging core
    # (2026-10-03), moved verbatim
    'haenv/scoring/matching/aliases.py', 'haenv/scoring/mounts/coverage.py', "haenv/runtime/row_store.py",
    'haenv/world/clinical/clinical_evidence.py',
    'haenv/scoring/semantic/seal.py', 'haenv/scoring/semantic/coverage.py', 'haenv/world/clinical/rhythm.py',
    "haenv/runtime/resume.py", 'haenv/world/validation/gate_tables.py',
    'haenv/scoring/gold_kinds.py',
    'haenv/scoring/separation.py', 'haenv/scoring/mounts/table.py', 'haenv/scoring/tracks.py',
    'haenv/world/generation/baselines.py', 'haenv/scoring/quantities.py', 'haenv/scoring/absence.py',
    # M1 2026-10-01: question-blind ceilings for every core dimension (imported by analytics)
    'haenv/scoring/blind_ceilings.py',
    'haenv/scoring/ranking.py', 'haenv/interaction/gated.py',
    'haenv/interaction/solve_guard.py',
    "verifier_core/gate.py", "verifier_core/ceilings.py",
    "verifier_core/vintage.py",
    "registry/vocab_tests.yaml", "registry/scoring.yaml",
    "registry/scoring_early_warning.yaml",
    "registry/scoring_tracking_review.yaml",
    "registry/conditions_independent.yaml", "registry/conditions_unified.yaml",
    "registry/threads.yaml", "registry/rivals.yaml",
    "registry/vocab_specialty.yaml", "registry/vocab_alias_excludes.yaml",
    "registry/vocab_alias_overlap_ok.yaml", "registry/composition_comorbid.yaml",
    "registry/causal_clusters.yaml", "registry/negated_findings.yaml",
    "registry/antagonist_axes.yaml", "registry/case_ids.yaml",
    "registry/gated_pricing.yaml",
    "registry/disputed_gold.yaml", "registry/condition_aliases.yaml",
    "registry/gated_pricing_streams.yaml",
    "registry/drug_indications.yaml", "registry/comorbidity_vocab.yaml",
    # Initial-stage M2 workup is consumed by wq and differential; it is judging data.
    "registry/m2_pre_review.yaml",
    "haenv/m2/workup.py",
    # Protocol lookup and system-score receipts are data consumed through query
    # tools on the judging path; their identity must travel with that path.
    "registry/protocols.yaml", "registry/system_scoring.yaml",
    'haenv/contracts/__init__.py',
    'haenv/contracts/pack.py',
    'haenv/contracts/hooks.py',
    'haenv/app/pack_hooks.py',
    'haenv/app/packs.py',
    'haenv/scoring/analysis/ceilings.py',
    'haenv/scoring/analysis/costs.py',
    'haenv/scoring/analysis/decisions.py',
    'haenv/scoring/analysis/dimensions.py',
    'haenv/scoring/analysis/discrimination.py',
    'haenv/scoring/analysis/gates.py',
    'haenv/scoring/analysis/precision.py',
    'haenv/scoring/analysis/probes.py',
    'haenv/scoring/analysis/provenance.py',
    'haenv/scoring/analysis/ranking.py',
    'haenv/scoring/analysis/tracks.py',
    'haenv/reporting/boards.py',
    'haenv/reporting/coverage.py',
    'haenv/reporting/diagnosis.py',
    'haenv/reporting/document.py',
    'haenv/reporting/formatting.py',
    'haenv/reporting/gates.py',
    'haenv/reporting/generation.py',
    'haenv/reporting/payloads.py',
    'haenv/reporting/probes.py',
    'haenv/reporting/rank_intervals.py',
    'haenv/reporting/readouts.py',
    'haenv/reporting/reliability.py',
    'haenv/reporting/samples.py',
    'haenv/reporting/tables.py',
    'haenv/reporting/validity.py',
    'haenv/interaction/query/__init__.py',
    'haenv/world/payload_policies.py',
    'haenv/runtime/retry_classifier.py',
    'haenv/reporting/tools/m2_raw_correct.py',
    'haenv/scoring/routes.py',
    'haenv/interaction/query/adapter.py',
    'haenv/interaction/query/profile.py',
    'haenv/interaction/query/single.py',
    'haenv/interaction/query/solver.py',
    'haenv/interaction/query/stubs.py',
    'haenv/interaction/query/tools.py',
    'haenv/interaction/query/fhir.py',
    'haenv/interaction/query/gated.py',
    'haenv/interaction/query/loop.py',
    'haenv/p2/query.py',
    'haenv/p3/query.py',
    'haenv/p4/query.py',
    'haenv/m2/query.py',
    'haenv/world/protocols/scoring.py',
    'haenv/m2/core.py',
    'haenv/m2/score.py',
    'haenv/p2/gate.py',
    'haenv/p2/history.py',
    'haenv/p2/protocols.py',
    'haenv/p2/rules.py',
    'haenv/p3/emr.py',
    'haenv/p3/features.py',
    'haenv/p3/gate.py',
    'haenv/p4/diversity_reader.py',
    'haenv/p4/features.py',
    'haenv/p4/gate.py',
    'haenv/p4/protocols.py',
    'haenv/p4/records.py',
    'haenv/world/protocols/followup.py',
    'haenv/m2/history.py',
    'haenv/m2/hooks.py',
    'haenv/p2/hooks.py',
    'haenv/p3/score.py',
    'haenv/p3/gold.py',
    'haenv/p4/gold.py',
    'haenv/world/labworld/meds.py',
    'haenv/scoring/semantic/parallel.py',
    'haenv/scoring/semantic/transport.py',
    'haenv/m2/round2.py',
    'haenv/world/labworld/observe.py',
    'haenv/world/labworld/perturb.py',
    'haenv/m2/plugin.py',
    'haenv_kernel/__init__.py',
    'haenv/world/validation/shared_audit.py',
    'registry/acute_events.yaml',
    'registry/base_condition_findings.yaml',
    'registry/control_targets.yaml',
    'registry/explain_away.yaml',
    'registry/history_dependency.yaml',
    'registry/labworld.yaml',
    'registry/p2_history_triage.yaml',
    'registry/p3_meds.yaml',
    'registry/p4_interpretation.yaml',
    'registry/pack2_triage_scenarios.yaml',
)

#: M2 judging segment (feat/m2-impl): the round-2 hook, the M2 atoms and composite, the frame judge.
#: It is stamped on M2 rows as `m2_judging_sha16` (`haenv/m2/hooks.py`) and never enters
#: `judging_sha16`, so v1.0.1 and F0 bridge cells keep the integration judging fingerprint.
#: `haenv/m2/core.py` is also the M2 world's stamp (external-gold manifest); the M2 emission gate is
#: fingerprinted there by its declaration, `haenv/m2/audit.py`. `haenv/app/packs.py` (judge
#: mounting and the publish gate, shared by every pack) is in each pack's judging segment and
#: `haenv/world/validation/shared_audit.py` (the shared emission gate) in each pack world segment.
M2_JUDGING = (
    'haenv/app/packs.py', 'haenv/world/history/history_dependency.py', "haenv/m2/history.py", "registry/history_dependency.yaml",
    "haenv/m2/hooks.py", "haenv/m2/round2.py", "haenv/m2/score.py", "haenv/m2/plugin.py",
    "haenv/m2/core.py", "registry/explain_away.yaml", "registry/base_condition_findings.yaml",
    "registry/m2_pre_review.yaml",
    "haenv/m2/workup.py",
    'haenv/contracts/__init__.py',
    'haenv/contracts/pack.py',
    'haenv/contracts/hooks.py',
    'haenv/app/pack_hooks.py',
)

#: Pack 2 (acute triage) segments (feat/pack2-triage). Neither enters `GENERATION` or
#: `JUDGING`, so jobs that do not declare the `haenv.pack2` plugin keep their `world_sha` and
#: `judging_sha16`. `PACK2_WORLD` is the pack-2 world (snapshot, event pool, G2 table, gate,
#: question template): its fingerprint is written into every pack-2 gold
#: (`adjudication.triage.world_sha16`) and into the registered `pack2.triage` probe, whose hash
#: enters the external-gold manifest, so a pack-2 world change moves the pack-2 `world_sha`.
#: `PACK2_JUDGING` (judge, hooks, composite) is stamped on pack-2 rows as `pack2_judging_sha16`.
PACK2_WORLD = (
    'haenv/world/validation/shared_audit.py',
    "haenv/pack2/world.py", "haenv/pack2/vitals.py", "haenv/pack2/events.py", "haenv/pack2/grading.py",
    "haenv/pack2/_util.py", "haenv/pack2/framing.py",
    "registry/pack2_triage_thresholds.yaml", "registry/pack2_acute_events.yaml",
    # the build hook that calls `world.finalize_adjudication` (it decides the re-armed gold
    # fields), so it is in both pack-2 segments
    "haenv/pack2/hooks.py",
)
PACK2_JUDGING = (
    'haenv/app/packs.py',
    "haenv/pack2/hooks.py", "haenv/pack2/plugin.py", "haenv/pack2/score.py",
    'haenv/contracts/__init__.py',
    'haenv/contracts/pack.py',
    'haenv/contracts/hooks.py',
    'haenv/app/pack_hooks.py',
)

#: Pack 4 (follow-up interpretation, feat/pack4-followup). Two segments of their own, neither in
#: `JUDGING` nor in `GENERATION`, so every existing job keeps `judging_sha16` and `world_sha`:
#: `P4_JUDGING` is stamped on pack-4 rows as `p4_judging_sha16` (`haenv/p4/hooks.py`); `P4_WORLD`
#: is stamped on the pack-4 gold block as `p4_world_sha16` and enters `world_sha` of a pack-4
#: batch through the registered probe (`haenv/p4/plugin.py:probe_dict`). `haenv/p4/gold.py` is in
#: both: the generator accepts an item by it and the judge reads its output.
P4_JUDGING = (
    'haenv/app/packs.py',
    "haenv/p4/score.py", "haenv/p4/plugin.py", "haenv/p4/hooks.py", "haenv/p4/gold.py",
    'haenv/contracts/__init__.py',
    'haenv/contracts/pack.py',
    'haenv/contracts/hooks.py',
    'haenv/app/pack_hooks.py',
)
P4_WORLD = (
    'haenv/world/validation/shared_audit.py', 'haenv/world/history/history_dependency.py', "haenv/p4/history.py", "registry/history_dependency.yaml",
    'haenv/world/labworld/__init__.py', 'haenv/world/labworld/truth.py', 'haenv/world/labworld/observe.py',
    'haenv/world/labworld/perturb.py', "registry/labworld.yaml",
    "haenv/p4/plan.py", "haenv/p4/realize.py", "haenv/p4/render.py", "haenv/p4/gold.py",
    "haenv/p4/gate.py", "haenv/p4/features.py", "haenv/p4/audit.py", "haenv/p4/magnitude.py",
    "registry/p4_clinical_sources.yaml",
    "haenv/p4/records.py", "haenv/p4/protocols.py", "registry/p4_interpretation.yaml",
    "haenv/p4/supply.py", 'tools/generation/p4_gen_job.py',
    "haenv/p4/diversity.py", "haenv/p4/diversity_reader.py", "registry/p4_diversity.yaml",
)

#: Pack 3 (chronic medication adjustment, feat/pack3-meds). Same arrangement as pack 4: neither
#: segment is in `JUDGING` or `GENERATION`; `P3_JUDGING` is stamped on pack-3 rows as
#: `p3_judging_sha16`, `P3_WORLD` on the pack-3 gold block as `p3_world_sha16` and enters `world_sha`
#: of a pack-3 batch through the registered probe. It shares three lab-world modules with
#: `P4_WORLD` (read, not changed, by pack 3).
P3_JUDGING = (
    'haenv/app/packs.py',
    "haenv/p3/score.py", "haenv/p3/plugin.py", "haenv/p3/hooks.py", "haenv/p3/gold.py",
    'haenv/world/labworld/meds.py', "registry/p3_meds.yaml", "registry/control_targets.yaml",
    'haenv/contracts/__init__.py',
    'haenv/contracts/pack.py',
    'haenv/contracts/hooks.py',
    'haenv/app/pack_hooks.py',
)
PROTOCOL_WORLD = (
    "haenv/p2/__init__.py", "haenv/p3/__init__.py", "haenv/p4/__init__.py",
    'haenv/world/protocols/__init__.py', 'haenv/world/protocols/library.py', 'haenv/world/protocols/delivery.py',
    'haenv/world/protocols/scoring.py', 'haenv/world/protocols/gates.py', 'haenv/world/protocols/followup.py',
    "registry/protocols.yaml", "registry/system_scoring.yaml",
    'haenv/interaction/query/tools.py', 'haenv/interaction/query/adapter.py', 'haenv/interaction/query/solver.py', 'haenv/interaction/query/stubs.py',
    'haenv/interaction/query/fhir.py',
    'haenv/app/settings.py', 'haenv/world/validation/invariants.py',
    # Every catalog is hospital-wide: all executable and registry sources are
    # dependencies, including entries outside the current question's subject.
    "haenv/p2/rules.py", "registry/pack2_triage_scenarios.yaml",
    "haenv/p3/render.py", 'haenv/world/labworld/meds.py', "registry/p3_meds.yaml", "registry/control_targets.yaml",
    'haenv/world/history/history_dependency.py', "registry/history_dependency.yaml", "haenv/p4/gold.py",
)
P4_WORLD = P4_WORLD + PROTOCOL_WORLD + ("haenv/p4/query.py",)
P3_WORLD = PROTOCOL_WORLD + (
    'haenv/world/validation/shared_audit.py', 'haenv/world/history/history_dependency.py', "haenv/p3/history.py", "registry/history_dependency.yaml",
    'haenv/world/labworld/__init__.py', 'haenv/world/labworld/truth.py', 'haenv/world/labworld/observe.py',
    'haenv/world/labworld/meds.py', "registry/p3_meds.yaml", "registry/control_targets.yaml",
    "haenv/p3/plan.py", "haenv/p3/realize.py", "haenv/p3/render.py", "haenv/p3/gold.py",
    "haenv/p3/gate.py", "haenv/p3/audit.py",
    "haenv/p3/query.py", "haenv/p3/features.py",
    "haenv/p3/emr.py",
    'tools/validation/invariant_gates.py',
    'tools/generation/p3_gen_job.py',
    "registry/p3_clinical_sources.yaml",
)

#: Pack 2 rebuilt (world 1.4, feat/w14-p2-triage; design the rebuilt triage scenario policy).
#: Same arrangement as pack 4: neither segment is in `JUDGING` or `GENERATION`; `P2_JUDGING` is
#: stamped on rebuilt pack-2 rows as `p2_judging_sha16`, `P2_WORLD` on the gold block as
#: `p2_world_sha16` and enters `world_sha` of a rebuilt pack-2 batch through the registered probe.
#: The previous pack 2 (`PACK2_WORLD` / `PACK2_JUDGING`, board of the unreleased 2026-10-06 candidate) is unchanged.
P2_JUDGING = (
    'haenv/app/packs.py',
    "haenv/p2/score.py", "haenv/p2/plugin.py", "haenv/p2/hooks.py",
    'haenv/contracts/__init__.py',
    'haenv/contracts/pack.py',
    'haenv/contracts/hooks.py',
    'haenv/app/pack_hooks.py',
)
P2_WORLD = PROTOCOL_WORLD + (
    'haenv/world/validation/shared_audit.py', 'haenv/world/history/history_dependency.py', "haenv/p2/history.py", "registry/history_dependency.yaml",
    "haenv/p2/rules.py", "haenv/p2/render.py", "haenv/p2/gate.py", "haenv/p2/hooks.py", "haenv/p2/background.py",
    "haenv/p2/context.py",
    "haenv/p2/clinical_rules.py", "registry/p2_history_triage.yaml",
    "haenv/p2/protocols.py",
    "registry/pack2_triage_scenarios.yaml",
    "haenv/p2/query.py",
    "haenv/p2/sources.py",
    "haenv/p2/complaints.py", "registry/p2_complaint_prior.yaml",
    "registry/p2_complaint_pool.yaml",
    "haenv/p2/audit.py",
)
#: Infrastructure segment: request accounting, cost receipts, the shared cap, paid slots,
#: scheduling and the pre-send wire check. A change here cannot alter what a solver is
#: shown, how it is asked or how a cell is scored, so it is recorded (`code_upgrades`)
#: and does not invalidate a board. The content-level guard is the pre-send wire check
#: (`haenv/runtime/transport.py:verify_wire`), which refuses to send any request whose messages or
#: answer settings differ from what was accounted. `haenv/scoring/semantic/parallel.py`
#: stays in `JUDGING`: whether a third vote is bought depends on the first two votes;
#: `haenv/runtime/judge_scheduler.py` supplies generic route scheduling only.
INFRA = (
    "haenv/runtime/solver_accounting.py", "haenv/runtime/completion.py",
    "haenv/runtime/relay_accounting.py", "haenv/runtime/native_accounting.py",
    "haenv/runtime/slots.py", 'haenv/scoring/semantic/budget.py',
    "haenv/runtime/scheduler.py", "haenv/runtime/transport.py",
    # judge lane scheduling and per-cell result files: how cells are run and written, not
    # what a vote or a score is (consensus and rubric summaries stay in `semantic_judge` /
    # `semantic_rubric`)
    'haenv/runtime/judge_scheduler.py',
    # per-call bookkeeping shared by solvers and the generation dispatcher: usage and
    # sampling echo, the token-budget policy (`needs_stream` picks the transport), billing
    "haenv/runtime/call_log.py",
    'haenv/runtime/budget.py',
    'haenv/runtime/pricing.py',
    'haenv/runtime/config_contracts.py',
    'haenv/runtime/quota.py',
)

# Physical relocations retain the same infrastructure role when verifying an
# already-recorded historical proof. Unknown names still require agreement on
# both sides; this table never promotes a new equivalence claim.
INFRA_PATH_MOVES = {'haenv/call_log.py': 'haenv/runtime/call_log.py',
 'haenv/native_accounting.py': 'haenv/runtime/native_accounting.py',
 'haenv/paid_completion.py': 'haenv/runtime/completion.py',
 'haenv/paid_slots.py': 'haenv/runtime/slots.py',
 'haenv/relay_accounting.py': 'haenv/runtime/relay_accounting.py',
 'haenv/run_scheduler.py': 'haenv/runtime/scheduler.py',
 'haenv/semantic_budget.py': 'haenv/runtime/budget.py',
 'haenv/semantic_parallel.py': 'haenv/runtime/judge_scheduler.py',
 'haenv/semantic_transport.py': 'haenv/runtime/pricing.py',
 'haenv/solver_accounting.py': 'haenv/runtime/solver_accounting.py',
 'haenv/transport.py': 'haenv/runtime/transport.py',
 'haenv/semantic/budget.py': 'haenv/runtime/budget.py',
 'haenv/semantic/parallel.py': 'haenv/runtime/judge_scheduler.py',
 'haenv/semantic/transport.py': 'haenv/runtime/pricing.py',
 'haenv/judge_scheduler.py': 'haenv/runtime/judge_scheduler.py'}

#: INFRA modules must not import these (AST-checked): scoring, gates, judges and the
#: prompt/payload builders decide what is asked and how it is scored.
INFRA_FORBIDDEN_IMPORTS = ("haenv.judges", "haenv.scoring", "haenv.gates", "haenv.prompts",
                           "haenv.payloads", "haenv.framings",
    'haenv.interaction.prompts',
    'haenv.app.artifacts.payloads',
    'haenv.world.generation.framings',
)

#: Old judging fingerprints and the value the same revision has under the current
#: segmentation (INFRA members removed from `JUDGING`). Each row is recomputed from git
#: objects by `haenv.anchor.verify_judging_equivalence`: the old value must reproduce, the
#: new value must reproduce, and the files dropped between the two must all be in `INFRA`.
#: Both W5 values (6f1196dc, a26d32ad) differ only in files now in `INFRA` => one value.
JUDGING_SHA_EQUIVALENCE = (
    {"old": "548e2c872f6271c7", "rev": "6f1196dc", "new": "2f560f1c346ba93d"},
    {"old": "6aa282bab6e2afcf", "rev": "a26d32ad", "new": "2f560f1c346ba93d"},
)

#: Old judging fingerprints and the value after a refactor that only moved code: files
#: renamed or split, imports moved, definitions moved into `INFRA`. Each row is recomputed
#: by `haenv.anchor.verify_judging_move`: `old` must reproduce at `rev`, `new` at `new_rev`,
#: and `haenv.anchor_moves.certify` must find every judging definition of `rev` at `new_rev`
#: with the same source, bound to the same definitions -- apart from the names in `changed`,
#: each with its reason, every one of which must really have changed.
JUDGING_SHA_MOVES: tuple[dict, ...] = (
    # One row from the previous anchor to the 1.2.0 tree: the kernel package and acyclic imports,
    # judging v2, world v2, the four task packs and the release changes. Moves and content changes
    # are grouped; every content change is declared, so the judging stamp history stays continuous.
    {"old": "67dd5866790e9cb9", "rev": "b55d2e85", "new": "a5f058bdd9de77b9", "new_rev": "c499aab1",
     "changed": {
         # moves: the kernel package, the evaluate.py split, the run context and the acyclic
         # import graph -- definitions whose source text changed with the move (60)
         "_verify_subprocess": "kernel package: the isolated verifier runs as `-m"
                               " haenv_kernel.verifier`, not by file path",
         "kernel_fingerprint": "kernel package: fingerprints the loaded `haenv_kernel` module",
         "kernel_path": "kernel package: locates kernel source for read-only tools only",
         "registry/condition_aliases.yaml": "kernel package: two `why` texts name"
                                            " `haenv_kernel/`",
         "RunContext": "run context: the object that replaces the module-level run state",
         "run_ctx_of": "run context: the run a solver is bound to, or an empty one",
         "RUN": "run context: removed as a module global; now a field of the "
                "explicit RunContext",
         "RunState": "run context: removed as a module global; now a field of the"
                     " explicit RunContext",
         "NOOP_FOR": "run context: removed as a module global; now a field of the"
                     " explicit RunContext",
         "QUANT_FOR": "run context: removed as a module global; now a field of "
                      "the explicit RunContext",
         "PREMISE_FOR": "run context: removed as a module global; now a field of "
                        "the explicit RunContext",
         "QUANT_TRUTH_DIST": "run context: removed as a module global; now a "
                             "field of the explicit RunContext",
         "ORACLE_GOLD_TESTS": "run context: removed as a module global; now a "
                              "field of the explicit RunContext",
         "ORACLE_DISC_NAMES": "run context: removed as a module global; now a "
                              "field of the explicit RunContext",
         "ORACLE_WARRANTED": "run context: removed as a module global; now a "
                             "field of the explicit RunContext",
         "ORACLE_DIAGNOSIS": "run context: removed as a module global; now a "
                             "field of the explicit RunContext",
         "_USAGE_ACC": "run context: removed as a module global; now a field of "
                       "the explicit RunContext",
         "_RETRY_ACC": "run context: removed as a module global; now a field of "
                       "the explicit RunContext",
         "_REPLAY_N": "run context: removed as a module global; now a field of "
                      "the explicit RunContext",
         "_REPLAY_FRESH": "run context: removed as a module global; now a field "
                          "of the explicit RunContext",
         "PER_BATCH_DICTS": "run context: removed as a module global; now a field"
                            " of the explicit RunContext",
         "enter_batch": "run context: removed as a module global; now a field of "
                        "the explicit RunContext",
         "_BATCH_SCOPE": "run context: removed as a module global; now a field of"
                         " the explicit RunContext",
         "assign_premises": "run context: takes `ctx` and uses its fields instead"
                            " of module globals",
         "assign_noop_probes": "run context: takes `ctx` and uses its fields "
                               "instead of module globals",
         "assign_quant_probes": "run context: takes `ctx` and uses its fields "
                                "instead of module globals",
         "assign_qside_probes": "run context: takes `ctx` and uses its fields "
                                "instead of module globals",
         "assign_oracle_gold": "run context: takes `ctx` and uses its fields "
                               "instead of module globals",
         "assign_oracle_warranted": "run context: takes `ctx` and uses its fields"
                                    " instead of module globals",
         "_resync_trend_windows": "run context: takes `ctx` and uses its fields "
                                  "instead of module globals",
         "_premise_suffix": "run context: takes `ctx` and uses its fields instead"
                            " of module globals",
         "_noop_suffix": "run context: takes `ctx` and uses its fields instead of"
                         " module globals",
         "_quant_suffix": "run context: takes `ctx` and uses its fields instead "
                          "of module globals",
         "render_for": "run context: takes `ctx` and uses its fields instead of module globals",
         "_row_single": "run context: takes `ctx` and uses its fields instead of module globals",
         "_row_gated": "run context: takes `ctx` and uses its fields instead of module globals",
         "_row_slices": "run context: takes `ctx` and uses its fields instead of module globals",
         "_row_multi": "run context: takes `ctx` and uses its fields instead of module globals",
         "_premise_row": "run context: takes `ctx` and uses its fields instead of"
                         " module globals",
         "_replay_solver_for": "run context: takes `ctx` and uses its fields "
                               "instead of module globals",
         "_ReplaySolver": "run context: takes `ctx` and uses its fields instead "
                          "of module globals",
         "_persist_gated_round": "run context: takes `ctx` and uses its fields "
                                 "instead of module globals",
         "gated_rounds_path": "run context: takes `ctx` and uses its fields "
                              "instead of module globals",
         "take_replay_count": "run context: takes `ctx` and uses its fields "
                              "instead of module globals",
         "_note_grid_usage": "run context: takes `ctx` and uses its fields "
                             "instead of module globals",
         "_note_grid_retry": "run context: takes `ctx` and uses its fields "
                             "instead of module globals",
         "take_grid_usage": "run context: takes `ctx` and uses its fields instead"
                            " of module globals",
         "take_grid_retry": "run context: takes `ctx` and uses its fields instead"
                            " of module globals",
         "save_response": "run context: takes `ctx` and uses its fields instead "
                          "of module globals",
         "_preflight_quota": "run context: takes `ctx` and uses its fields "
                             "instead of module globals",
         "_run_tasks": "run context: takes `ctx` and uses its fields instead of module globals",
         "run_eval": "run context: takes `ctx` and uses its fields instead of module globals",
         "OracleTestsSolver": "run context: the oracle stub reads its gold from "
                              "the run it is bound to",
         "OneLongTestSolver": "run context: the oracle stub reads its gold from "
                              "the run it is bound to",
         "OracleProbeSolver": "run context: the oracle stub reads its gold from "
                              "the run it is bound to",
         "OracleReviewSolver": "run context: the oracle stub reads its gold from "
                               "the run it is bound to",
         "LateConvergeSolver": "run context: the oracle stub reads its gold from "
                               "the run it is bound to",
         "SOLVE_EXEMPT": "acyclic imports: a reason text names "
                         "`mount_coverage.unwired_geometries`, where that check "
                         "now lives",
         "haenv/semantic_pipeline.py:ROOT": "acyclic imports: moved with "
                                            "`code_state` to `semantic_seal.py`",
         "haenv/semantic_seal.py:ROOT": "acyclic imports: "
                                        "`Path(__file__).resolve().parents[1]` "
                                        "from the same `haenv/` directory, so the"
                                        " same root",
         # moves: the pack hook points read `qside.render_for` live after the evaluate split (4)
         "OpenAICompatSolver": "merge glue: `render_for` is a pack hook point; "
                               "the call reads `qside.render_for` live so a "
                               "pack's wrapper takes effect after the evaluate "
                               "split",
         "GoogleSolver": "merge glue: same live read of the `render_for` hook point",
         "CLISolver": "merge glue: same live read of the `render_for` hook point",
         "_prompt_sha_for": "merge glue: same live read of the `render_for` hook point",
         # content: judging v2, world v2 and the four task packs (113)
         "tool_grounded_joint_of": "content: judging v2, world v2 or the task packs",
         "_tests_sets": "content: judging v2, world v2 or the task packs",
         "_u": "content: judging v2, world v2 or the task packs",
         "gold_from_cases_jsonl": "content: judging v2, world v2 or the task packs",
         "_stub_trace": "content: judging v2, world v2 or the task packs",
         "_f1": "content: judging v2, world v2 or the task packs",
         "D9_PROVISIONAL": "content: judging v2, world v2 or the task packs",
         "report_section": "content: judging v2, world v2 or the task packs",
         "_lab_calendar_signals": "content: judging v2, world v2 or the task packs",
         "decoy_targets": "content: judging v2, world v2 or the task packs",
         "normalize_action_class": "content: judging v2, world v2 or the task packs",
         "registry/gated_pricing.yaml": "content: judging v2, world v2 or the task packs",
         "evidence_names_for": "content: judging v2, world v2 or the task packs",
         "_NAME_SPLIT_RE": "content: judging v2, world v2 or the task packs",
         "registry/composition_comorbid.yaml": "content: judging v2, world v2 or the task packs",
         "_dx_score": "content: judging v2, world v2 or the task packs",
         "_hard_gates": "content: judging v2, world v2 or the task packs",
         "_resolves_to": "content: judging v2, world v2 or the task packs",
         "_rd": "content: judging v2, world v2 or the task packs",
         "core_constant_ceilings": "content: judging v2, world v2 or the task packs",
         "run_gated": "content: judging v2, world v2 or the task packs",
         "read_run": "content: judging v2, world v2 or the task packs",
         "tests_f1_ceiling": "content: judging v2, world v2 or the task packs",
         "tests_f1_matcher_ceiling": "content: judging v2, world v2 or the task packs",
         "_fsum_mean": "content: judging v2, world v2 or the task packs",
         "multiplier": "content: judging v2, world v2 or the task packs",
         "_f1_of": "content: judging v2, world v2 or the task packs",
         "dx_listed_ceiling": "content: judging v2, world v2 or the task packs",
         "flag_side": "content: judging v2, world v2 or the task packs",
         "normalize_evidence_name": "content: judging v2, world v2 or the task packs",
         "review_macro_of": "content: judging v2, world v2 or the task packs",
         "INSUFFICIENT_KIND": "content: judging v2, world v2 or the task packs",
         "_executed_tests": "content: judging v2, world v2 or the task packs",
         "_pricing": "content: judging v2, world v2 or the task packs",
         "asserted_diagnoses": "content: judging v2, world v2 or the task packs",
         "key_coverage": "content: judging v2, world v2 or the task packs",
         "_certainty": "content: judging v2, world v2 or the task packs",
         "_quant": "content: judging v2, world v2 or the task packs",
         "tool_blind_stubs": "content: judging v2, world v2 or the task packs",
         "abstention_utility_of": "content: judging v2, world v2 or the task packs",
         "grade": "content: judging v2, world v2 or the task packs",
         "chance_corrected": "content: judging v2, world v2 or the task packs",
         "_report_face": "content: judging v2, world v2 or the task packs",
         "_dx_hits": "content: judging v2, world v2 or the task packs",
         "apply_recall_code_first": "content: judging v2, world v2 or the task packs",
         "dx_listed_top_k_freq": "content: judging v2, world v2 or the task packs",
         "_pick_w": "content: judging v2, world v2 or the task packs",
         "discriminator_tests": "content: judging v2, world v2 or the task packs",
         "gated_menus_from_cases": "content: judging v2, world v2 or the task packs",
         "RECALL_CODE_FIRST": "content: judging v2, world v2 or the task packs",
         "_stub_joint": "content: judging v2, world v2 or the task packs",
         "constant_ceilings": "content: judging v2, world v2 or the task packs",
         "_execute_run": "content: judging v2, world v2 or the task packs",
         "ROW_AUXILIARY": "content: judging v2, world v2 or the task packs",
         "rank_ddx": "content: judging v2, world v2 or the task packs",
         "check_caps": "content: judging v2, world v2 or the task packs",
         "_drain_requests": "content: judging v2, world v2 or the task packs",
         "_ACTION_CLASS_RE": "content: judging v2, world v2 or the task packs",
         "_fill_to_budget": "content: judging v2, world v2 or the task packs",
         "registry/negated_findings.yaml": "content: judging v2, world v2 or the task packs",
         "QUANTITIES": "content: judging v2, world v2 or the task packs",
         "_RULE_OUT_CERTAINTY": "content: judging v2, world v2 or the task packs",
         "delivered_truth_present": "content: judging v2, world v2 or the task packs",
         "_named_on_stream": "content: judging v2, world v2 or the task packs",
         "NON_HARM_GATES": "content: judging v2, world v2 or the task packs",
         "CORE_DIMS": "content: judging v2, world v2 or the task packs",
         "REPORT_ONLY_DIMS": "content: judging v2, world v2 or the task packs",
         "_by_case": "content: judging v2, world v2 or the task packs",
         "TOOL_STUB_DRAWS": "content: judging v2, world v2 or the task packs",
         "menu_for": "content: judging v2, world v2 or the task packs",
         "_EPS": "content: judging v2, world v2 or the task packs",
         "delivered_series": "content: judging v2, world v2 or the task packs",
         "load_rows": "content: judging v2, world v2 or the task packs",
         "REPORTED_DIAGNOSTICS": "content: judging v2, world v2 or the task packs",
         "_HEDGED_CERTAINTY": "content: judging v2, world v2 or the task packs",
         "_norm_ans": "content: judging v2, world v2 or the task packs",
         "registry/case_ids.yaml": "content: judging v2, world v2 or the task packs",
         "has_reading_model": "content: judging v2, world v2 or the task packs",
         "_track_D": "content: judging v2, world v2 or the task packs",
         "DX_LIST_CAP_MARGIN": "content: judging v2, world v2 or the task packs",
         "_cites_evidence_name": "content: judging v2, world v2 or the task packs",
         "ROLES": "content: judging v2, world v2 or the task packs",
         "judge_workup": "content: judging v2, world v2 or the task packs",
         "synth_on_demand": "content: judging v2, world v2 or the task packs",
         "dx_listed_n0_of": "content: judging v2, world v2 or the task packs",
         "Profile": "content: judging v2, world v2 or the task packs",
         "registry/vocab_tests.yaml": "content: judging v2, world v2 or the task packs",
         "METRIC_FIELDS": "content: judging v2, world v2 or the task packs",
         "_tests_f1_matcher_search": "content: judging v2, world v2 or the task packs",
         "_EXACT": "content: judging v2, world v2 or the task packs",
         "composite_scores": "content: judging v2, world v2 or the task packs",
         "_abst": "content: judging v2, world v2 or the task packs",
         "batch_discrimination_gate": "content: judging v2, world v2 or the task packs",
         "TOOL_STUB_SEED": "content: judging v2, world v2 or the task packs",
         "menu_findings": "content: judging v2, world v2 or the task packs",
         "_obs_cfg": "content: judging v2, world v2 or the task packs",
         "dx_rank_of": "content: judging v2, world v2 or the task packs",
         "entry_excluded": "content: judging v2, world v2 or the task packs",
         "is_rule_out_pending": "content: judging v2, world v2 or the task packs",
         "_review": "content: judging v2, world v2 or the task packs",
         "solver_for_spec": "content: judging v2, world v2 or the task packs",
         "tool_track": "content: judging v2, world v2 or the task packs",
         "_noop": "content: judging v2, world v2 or the task packs",
         "overlay_rows": "content: judging v2, world v2 or the task packs",
         "abst_stub_rows": "content: judging v2, world v2 or the task packs",
         "exact_precision": "content: judging v2, world v2 or the task packs",
         "judge_noop_probe": "content: judging v2, world v2 or the task packs",
         "_norm_item": "content: judging v2, world v2 or the task packs",
         "check_cadence": "content: judging v2, world v2 or the task packs",
         "PROPENSITY_GATES": "content: judging v2, world v2 or the task packs",
         "rank_ddx_exact": "content: judging v2, world v2 or the task packs",
         "_listed": "content: judging v2, world v2 or the task packs",
         "synth_menu_target": "content: judging v2, world v2 or the task packs",
         # content: release changes (22)
         "haenv/judges/recall_match.py:YES": "recall matching of ordered tests "
                                             "enters the judging segment: "
                                             "semantic_report imports it and it "
                                             "decides scores",
         "haenv/judges/recall_match.py:NO": "recall matching of ordered tests "
                                            "enters the judging segment: "
                                            "semantic_report imports it and it "
                                            "decides scores",
         "haenv/judges/recall_match.py:UNDECIDED": "recall matching of ordered "
                                                   "tests enters the judging "
                                                   "segment: semantic_report "
                                                   "imports it and it decides "
                                                   "scores",
         "haenv/judges/recall_match.py:_SEP": "recall matching of ordered tests "
                                              "enters the judging segment: "
                                              "semantic_report imports it and it "
                                              "decides scores",
         "haenv/judges/recall_match.py:_ASCII_ALNUM": "recall matching of ordered"
                                                      " tests enters the judging "
                                                      "segment: semantic_report "
                                                      "imports it and it decides "
                                                      "scores",
         "haenv/judges/recall_match.py:_CJK_SPACE": "recall matching of ordered "
                                                    "tests enters the judging "
                                                    "segment: semantic_report "
                                                    "imports it and it decides "
                                                    "scores",
         "haenv/judges/recall_match.py:_norm": "recall matching of ordered tests "
                                               "enters the judging segment: "
                                               "semantic_report imports it and it"
                                               " decides scores",
         "haenv/judges/recall_match.py:_variants": "recall matching of ordered "
                                                   "tests enters the judging "
                                                   "segment: semantic_report "
                                                   "imports it and it decides "
                                                   "scores",
         "haenv/judges/recall_match.py:_is_ascii_alnum": "recall matching of "
                                                         "ordered tests enters "
                                                         "the judging segment: "
                                                         "semantic_report imports"
                                                         " it and it decides "
                                                         "scores",
         "haenv/judges/recall_match.py:_spans": "recall matching of ordered tests"
                                                " enters the judging segment: "
                                                "semantic_report imports it and "
                                                "it decides scores",
         "haenv/judges/recall_match.py:_Text": "recall matching of ordered tests "
                                               "enters the judging segment: "
                                               "semantic_report imports it and it"
                                               " decides scores",
         "haenv/judges/recall_match.py:_vocab": "recall matching of ordered tests"
                                                " enters the judging segment: "
                                                "semantic_report imports it and "
                                                "it decides scores",
         "haenv/judges/recall_match.py:gold_segments": "recall matching of "
                                                       "ordered tests enters the "
                                                       "judging segment: "
                                                       "semantic_report imports "
                                                       "it and it decides scores",
         "haenv/judges/recall_match.py:_forms": "recall matching of ordered tests"
                                                " enters the judging segment: "
                                                "semantic_report imports it and "
                                                "it decides scores",
         "haenv/judges/recall_match.py:_shadows_of": "recall matching of ordered "
                                                     "tests enters the judging "
                                                     "segment: semantic_report "
                                                     "imports it and it decides "
                                                     "scores",
         "haenv/judges/recall_match.py:_flag": "recall matching of ordered tests "
                                               "enters the judging segment: "
                                               "semantic_report imports it and it"
                                               " decides scores",
         "haenv/judges/recall_match.py:_hint_tokens": "recall matching of ordered"
                                                      " tests enters the judging "
                                                      "segment: semantic_report "
                                                      "imports it and it decides "
                                                      "scores",
         "haenv/judges/recall_match.py:match_item": "recall matching of ordered "
                                                    "tests enters the judging "
                                                    "segment: semantic_report "
                                                    "imports it and it decides "
                                                    "scores",
         "haenv/judges/recall_match.py:match_required_tests": "recall matching of"
                                                              " ordered tests "
                                                              "enters the judging"
                                                              " segment: "
                                                              "semantic_report "
                                                              "imports it and it "
                                                              "decides scores",
         "load_judge_plugins": "an entry-point target registered under several "
                               "groups attaches once, so an installed wheel loads"
                               " each pack once",
         "check_batch": "GEN8d breaks a most-common-text tie on the text, so the "
                        "batch record does not depend on PYTHONHASHSEED; a gate "
                        "message names `haenv_kernel/noise.py`",
         "registry/scoring.yaml": "content: judging v2 scoring roles; three "
                                  "`decided` values carry no document references",
     }},
    # 1.2.0 -> the next tree: parallel shortcut p-values, resume by abort cause, settlement by
    # generation id. No definition only moved; every change is declared.
    {"old": "a5f058bdd9de77b9", "rev": "55783fa8", "new": "950578a687aacc5e", "new_rev": "1e6972bc",
     "changed": {
         # content (11)
         "haenv/gates_shortcut.py:_SPLIT_BASE": "the shortcut check computes its "
                                                "split p-values in parallel, with"
                                                " the same values in the same "
                                                "order",
         "haenv/gates_shortcut.py:_split_order": "the shortcut check computes its"
                                                 " split p-values in parallel, "
                                                 "with the same values in the "
                                                 "same order",
         "haenv/gates_shortcut.py:_split_p_of": "the shortcut check computes its "
                                                "split p-values in parallel, with"
                                                " the same values in the same "
                                                "order",
         "haenv/gates_shortcut.py:check_shortcut": "the shortcut check computes "
                                                   "its split p-values in "
                                                   "parallel, with the same "
                                                   "values in the same order",
         "haenv/resume.py:OUR_SIDE_CLASSES": "resume re-sends only cells whose "
                                             "abort the harness side caused; a "
                                             "model's own result is kept",
         "haenv/resume.py:_base": "resume re-sends only cells whose abort the "
                                  "harness side caused; a model's own result is "
                                  "kept",
         "haenv/resume.py:_done_keys": "resume re-sends only cells whose abort "
                                       "the harness side caused; a model's own "
                                       "result is kept",
         "haenv/resume.py:is_retryable": "resume re-sends only cells whose abort "
                                         "the harness side caused; a model's own "
                                         "result is kept",
         "haenv/resume.py:resend_cause": "resume re-sends only cells whose abort "
                                         "the harness side caused; a model's own "
                                         "result is kept",
         "haenv/rows.py:_row_slices": "an all-empty slices abort records the "
                                      "failed attempts, so resume can tell its "
                                      "cause",
         "haenv/solvers.py:OpenAICompatSolver": "records the provider's "
                                                "generation id from the response "
                                                "headers, so an interrupted paid "
                                                "request settles by that id",
     }},
)

#: Registry yaml in neither segment, with the reason the judging segment cannot reach it.
REGISTRY_YAML_OUT_OF_SCOPE: dict[str, str] = {
    "registry/pack2_triage_thresholds.yaml":
        "Pack-2 grading table G2 (provisional). Read only by `haenv/pack2/world.py`, which no "
        "`JUDGING` module imports; it is anchored by `PACK2_WORLD` instead, and its content is "
        "hashed into every pack-2 gold (`tables_sha16`, `world_sha16`). If a `JUDGING` module "
        "ever reads it, this exemption has to come off first.",
    "registry/pack2_acute_events.yaml":
        "Pack-2 acute-event pool and wording. Same boundary as `pack2_triage_thresholds.yaml`: "
        "read only by `haenv/pack2/world.py` and the job composer `tools/generation/pack2_gen_job.py`, "
        "anchored by `PACK2_WORLD`.",
    "registry/knobs.yaml":
        "Input-surface registry: why each module-level constant lives in "
        "code rather than config. Its only production consumer is "
        "`haenv/app/inputs.py` (the read-only `haenv inputs` command); the judging "
        "segment makes zero calls into it, so changing it doesn't move any "
        "cell's value. It must not go into `JUDGING`: it describes the judging "
        "segment's own constants, so including it would let a sentence "
        "explaining a threshold move `judging_sha16` and invalidate a paid "
        "leaderboard (editing a comment must not invalidate the anchor). If "
        "judging code ever reads it (say, deciding whether to score a "
        "dimension based on `class`), this exemption has to come off first.",
    "registry/disease_prevalence.yaml":
        "Prevalence table for the composition-v2 allocator. Its only production consumer is "
        "`tools/generation/gen_composition_v2.py` (and `registry.association_admissible`, called by that tool), "
        "which runs at job-generation time -- its output is written into `inputs/*.job.yaml`, out of the "
        "judging segment's reach; the judging segment makes zero calls into it. Same boundary as "
        "`tiers.yaml`. If judging code ever reads a prevalence, this exemption has to come off first.",
    "registry/tiers.yaml":
        "Tier ratios / slicing geometry / gated budgets. The `authoritative:` "
        "section's only consumer is `haenv/world/clinical/ddx.py:_tier`, which runs at "
        "job-generation time -- its output is written into `inputs/*.job.yaml`'s "
        "`latent`, out of the judging segment's reach. Same boundary as "
        "`condition_sex_ratio.yaml`: drawn at the job-yaml edge, covered by "
        "`batch.json`'s `job_sha256`. Its `mirrored:` section copies the "
        "judging segment's own constants (`evaluate.REAL_RHYTHM_*` / "
        "`gated.*`) as a read-only mirror; the authority stays in the code, and "
        "the maintainers' tests check the mirror entry by entry. Putting it "
        "into `JUDGING` would have it backwards: the mirror follows what it "
        "mirrors and must not invalidate the board when it changes. If judging "
        "code ever reads `mirrored:` directly (instead of the constants), this "
        "exemption has to come off first.",
    "registry/judges_catalog.yaml":
        "Judge catalog: definitions for every judge plus their quantity types "
        "and directions. Its consumers are read-only readout tooling; the "
        "judging segment makes zero calls into it. It deliberately does not "
        "register `kind` for fields already covered by "
        "`analytics.METRIC_KINDS`, nor does it write `role` (the sole sources "
        "for those two are `METRIC_KINDS` and `scoring.yaml` respectively); "
        "because it isn't a second source, changing it cannot change any "
        "cell's score. If judging code ever reads `kind`/`dir` from this table, "
        "this exemption has to come off first.",
    "registry/pack_catalog.yaml":
        "Pack catalog (role / title / asks / publishable). A pure display "
        "layer: its production consumer is `haenv/cli.py:_catalog_gate` (the "
        "warning text and asides shown when a report is generated); the "
        "judging segment makes zero calls into it, so changing it can't move a "
        "single eval row's value. It does not go into `GENERATION` either: it "
        "plays no part in generating items, and folding it in would let "
        "editing a single `title` move `world_sha` and mark every frozen pack "
        "as invalidated. The gate is hung off `cli.py` rather than `report.py` "
        "so it doesn't touch the judging fingerprint (see `_catalog_gate`'s "
        "docstring). If it ever enters judging (say, `publishable` becomes a "
        "multiplier that changes a score), this exemption has to come off "
        "first.",
    "registry/literature_evidence.yaml":
        "纯展示与溯源，评分与出题都不读。Only haenv/resources/catalog.py and the CLI "
        "review notice read this sidecar; it is never merged into conditions "
        "or findings. No judging imports and no generation consumers.",
    "registry/condition_threads.yaml":
        "Per-atomic-condition thread names (disease-line names). Its only "
        "consumer is the maintainers' generator that writes the thread names "
        "into `composition_comorbid.yaml`'s `threads`, and that table is in "
        "the `JUDGING` segment. The judging side reads the thread names saved "
        "in the composition table, not this one; changing this table only "
        "affects the next generated composition. This boundary is narrower "
        "than the others: the table's output is already in the judging "
        "segment, and this table is the mold that produces it. If the mold "
        "changes without the output being regenerated, the two go out of sync "
        "(the same condition's thread name diverging across pairings). If "
        "judging code ever reads this table directly, this exemption has to "
        "come off first.",
    "registry/condition_sex_ratio.yaml":
        "Per-condition sex ratio (`male` share + rationale). Its only "
        "consumer is `haenv/world/clinical/ddx.py:_sex_ratio_table` -> `_sex_of`, which runs "
        "at job-generation time -- its output is written into "
        "`inputs/*.job.yaml`'s `raw.sex`, out of the judging segment's reach. "
        "Same boundary as `background_comorbidity.yaml` / "
        "`disease_device_requirements.yaml`: drawn at the job-yaml edge, "
        "covered by `batch.json`'s `job_sha256`. It reads "
        "`composition_comorbid.yaml` (comorbidity profiles computed by "
        "odds-product), which is in the `JUDGING` segment; generation-side "
        "code reading judging-segment data is the allowed direction.",
    "registry/background_comorbidity.yaml":
        "Eligible background-comorbidity list and exclusion rationale. "
        "Its only consumer is `haenv/world/generation/demographics.py`, which runs at "
        "job-generation time (via `ddx.py`) -- its output is written into "
        "`inputs/*.job.yaml`'s `raw.comorbidities`, out of the judging "
        "segment's reach. Same boundary as "
        "`disease_device_requirements.yaml`: also not in `GENERATION`, because "
        "its content reaches the prompt only through the job yaml, so the "
        "boundary is drawn at the job-yaml edge, covered by `batch.json`'s "
        "`job_sha256`. This is a different table from "
        "`comorbidity_vocab.yaml`: that one is in the JUDGING segment because "
        "GEN28 reads it directly to judge whether a comorbidity name is "
        "recognized; this table only decides which ones get sampled, and once "
        "sampled they are written into the job yaml and the judging side never "
        "checks this table again.",
    "registry/disease_device_requirements.yaml":
        "Joint condition-device sampling constraints. Its only consumer "
        "is `haenv/world/generation/demographics.py`, which runs at job-generation time "
        "(via `ddx.py`) -- its output is written into "
        "`inputs/*.job.yaml`, out of the judging segment's reach. It does "
        "not go into `GENERATION` either, for the same reason as the "
        "`conditions_unified.yaml` group: its content reaches the prompt only "
        "through the job yaml, so the boundary is drawn at the job-yaml edge, "
        "covered by `batch.json`'s `job_sha256`; folding it into GENERATION "
        "would report a false invalidation whenever the table changes without "
        "the job being regenerated. World comparability is carried by "
        "`anchor.world_vintages`'s three-part key "
        "(`world_sha` · `world_knobs` · `job`).",
    "registry/direct_read_baseline.yaml":
        "The already-approved list for the static screen "
        "`no_new_direct_read`; its consumers are maintainer tooling "
        "and the internal regression suite, not the runtime resolver. It constrains "
        "source-code style, not any reading; changing it can't change any "
        "eval row's value.",
    "registry/reachability_baseline.yaml":
        "The baseline for the reachability check; its consumers are "
        "maintainer tooling and tests. It records which reachability gaps "
        "remain outstanding and plays no part in judging any cell; changing "
        "it only changes how strict that check is.",
}

#: Provenance entry points are excluded from every criterion closure. Unlike an
#: import exemption, scoring modules may not import these modules at all; the
#: criterion traversal rejects that edge. Outer commands call the frozen scorer
#: and attach/compare identities without changing its score or denominator.
PROVENANCE_MODULES: dict[str, str] = {
    'haenv/scoring/management/route_run.py': "Outer route admission, snapshot verification and receipt provenance; delegates tasks, parsing and scoring to the frozen management pipeline and is never imported by scorers.",
    'haenv/provenance/judging.py': "Source identities and comparison only; like anchor.py, provenance is not a scoring rule.",
    'haenv/provenance/judging_provenance.py': "Outer row/run/report metadata decorators; frozen judging modules must never import this facade.",
    'tools/evaluation/management_rejudge.py': "Command dispatch and metadata recording; management_pipeline.py supplies the independent rubric scorer.",
    'tools/evaluation/management_precision_rejudge.py': "Precision command dispatch and provenance recording; never imported by scorers.",
    'tools/evaluation/management_board_prepare.py': "Saved-batch subset preparation and provenance only; never imported by scorers.",
    'tools/evaluation/management_board_rejudge.py': "Board discovery, sealed-run orchestration and Q field projection; scores delegate to management_rubric, never imported by scorers.",
}
#: haenv modules the judging segment imports but that are not in `JUDGING`, with the
#: reason. An entry must not decide `overall` or any `absence.DENOM_FIELD` denominator;
#: its crossing symbols are pinned in `ALLOW_SYMBOL_PIN`.
JUDGING_IMPORT_ALLOW: dict[str, str] = {
 'haenv/world/labworld/truth.py': 'P3_WORLD/P4_WORLD true-curve generation. The onset import belongs '
                                'to meds.response/truth, while grading uses the drug tables and '
                                'observed refill coverage. Exact crossing is limited to onset; '
                                'both world declarations retain this source.',
 'haenv/world/physio/kernel.py': 'GENERATION: check_footprint_not_discriminative uses _pk only to load the '
                                 'physiology registry and read emitted topics for the generation-batch '
                                 'footprint audit. Saved-response rescoring does not call this audit; the '
                                 'module is already pinned in GENERATION.',
 'haenv/runtime/judge_scheduler.py': 'INFRA: management overlays reuse the bounded lane scheduler; it '
                                     "returns the worker's unchanged value and cannot create a verdict. The "
                                     'imported run_bounded symbol is pinned below.',
 'haenv/runtime/native_accounting.py': 'INFRA: the offline management estimator checks the exact registered '
                                       'native tariff with NativePrices; prices affect cost estimates only, '
                                       'never an atomic verdict or score.',
 'haenv/runtime/completion.py': '`INFRA` segment (see `INFRA`): the streaming reader records the first '
                                "chunk's generation id (`note_generation`) on the in-flight marker, so a cut "
                                "stream can be settled from the provider's record later. It returns nothing "
                                'and changes no byte of a response, a vote or a score. Management also uses '
                                'AccountedCompletion and its failure tags for native one-attempt '
                                'reservations and receipts; response parsing remains in MANAGEMENT_JUDGING. '
                                'Crossing symbols are pinned in `ALLOW_SYMBOL_PIN`.',
 'haenv/scoring/semantic/budget.py': '`INFRA` segment (see `INFRA`): request accounting, the shared cap and '
                                     'the pre-send wire check. Judging code imports only the named symbols '
                                     '(a budget stop, the cap ledger, price bounds, the paid entry point); '
                                     'none of them produces, filters or alters a response, a vote or a '
                                     'score. The crossing symbols are pinned in `ALLOW_SYMBOL_PIN`.',
 'haenv/runtime/relay_accounting.py': '`INFRA` segment (see `INFRA`): relay tariff and billing-log reads. '
                                      'Judging code imports the balance gate (`preflight_quota`), which '
                                      'decides whether a run may start spending, and re-exports the tariff '
                                      'it prices with (`RELAY_COST_*`, `RELAY_USD_PER_MTOK`, '
                                      '`_measured_tokens_per_grid`); none of them changes a byte of a '
                                      'response, a vote or a score. The crossing symbols are pinned in '
                                      '`ALLOW_SYMBOL_PIN`.',
 'haenv/runtime/solver_accounting.py': '`INFRA` segment (see `INFRA`): request accounting, the shared cap '
                                       'and the pre-send wire check. Judging code imports only the named '
                                       'symbols (a budget stop, the cap ledger, price bounds, the paid entry '
                                       'point); none of them produces, filters or alters a response, a vote '
                                       'or a score. wire_settings only records the declared answer and route '
                                       'conditions. The crossing symbols are pinned in `ALLOW_SYMBOL_PIN`.',
 'haenv/runtime/transport.py': '`INFRA` segment (see `INFRA`): request accounting, the shared cap and the '
                               'pre-send wire check. Judging code imports only the named symbols: the '
                               "pre-send wire check; the validators of a model's upstream pin and sampling "
                               'block, which refuse a configuration before any request; the temperature a '
                               'request carries, read from that block; the served-upstream check, which '
                               'fails an attempt a pinned route says another upstream served (the attempt is '
                               'retried or the cell has no answer, as for any failed attempt); and '
                               "`judge_spec`, which drops the solver-only keys so a judge's request is what "
                               'it was before they existed. google_response_format translates the declared '
                               'schema into Google wire fields; proxy_for_solver selects the recorded route. '
                               'None of them produces or alters a response, a vote or a score. The crossing '
                               'symbols are pinned in `ALLOW_SYMBOL_PIN`.',
 'haenv/scoring/semantic/runref.py': 'Only locates the sealed semantic run a manifest or view names -- by '
                                     'path, else by run_id / tasks digest / manifest digest -- and refuses a '
                                     'run whose bytes or identity differ and any ambiguous match. Every run '
                                     'it returns is then read through the unchanged checks (manifest digest, '
                                     'tasks digest, per-cell recomputation of the persisted votes); no '
                                     'score, vote or metric is produced or altered here. The crossing '
                                     'symbols are pinned in `ALLOW_SYMBOL_PIN`.',
 'haenv/runtime/scheduler.py': 'Acts only while solving: which cell runs when, how many run at once, and '
                               "when a live request that outlived its model's stall deadline is closed and "
                               'retried. Recomputing a batch reads saved responses and never passes through '
                               'it, so no recomputed reading can change with it. Every cell still runs the '
                               'same function once; offline rows are identical under serial, per-backend and '
                               'per-model scheduling. A stall close is recorded on the row '
                               '(`stall_timeouts`), so the answers it affected stay identifiable.',
 'haenv/app/artifacts/payloads.py': 'Semantic rejudging reads load_payloads only: a strict JSON/dataclass '
                                    'decoder of source files hashed by the run manifest. No scoring or '
                                    'payload synthesis function crosses this boundary; if decoding starts '
                                    'transforming fields, this exemption must be replaced by judging '
                                    'coverage.',
 'haenv/app/artifacts/store.py': 'Semantic rejudging uses load_cases only to deserialize saved RawCase and '
                                 'restore the saved Q-side ledger; it neither regenerates nor substitutes '
                                 'missing cases. The source is hashed and required case IDs are checked. No '
                                 'generation or scoring function crosses this boundary.',
 'haenv/world/observations/streams.py': 'Already in the GENERATION segment. Only feeds gates.DEVICE_SIGNALS, '
                                        'read by launch gate GEN7 at build time: a change alters which cases '
                                        'are emitted (world fingerprint), never the score recomputed from a '
                                        'saved response. Stream prices reach scoring through the generated '
                                        'registry/gated_pricing_streams.yaml, which is in JUDGING.',
 'haenv/contracts/external_gold.py': 'Only holds default-empty registries (BLOCKS / FRAMINGS / PROBES) and '
                                     'write entry points, and judges nothing itself; with empty registries '
                                     'the judging path is byte-for-byte unchanged. When non-empty, which '
                                     "external piece is plugged in is traceable through job.yaml's "
                                     '`plugins:` group name via `job_sha256`, so the judging fingerprint '
                                     "doesn't need to cover it separately.",
 'haenv/world/generation/framings.py': 'Already in the GENERATION segment: changing it invalidates the item '
                                       '(requires regenerating and rerunning), which is a stricter '
                                       'consequence than invalidating readings, so the judging fingerprint '
                                       "doesn't need to cover it too. Recomputed readings from the same "
                                       'batch of saved responses are unchanged -- changing the prompt '
                                       "doesn't change the judging. The judging segment only reaches it "
                                       "through `evaluate.py`'s re-export layer; the symbols that cross are "
                                       'pinned in `ALLOW_SYMBOL_PIN`.',
 'haenv/world/observations/events.py': 'Already in GENERATION: changing it regenerates the packs. The '
                                       'judging segment takes `inherited_profile` (as `_inh`) from it, the '
                                       "kernel's inherited event profile read where the plugin pools are "
                                       'rebound.',
 'haenv/world/observations/events_pools.py': 'Already in GENERATION: changing it invalidates the whole '
                                             'freeze (the pack must be regenerated), which already implies '
                                             'invalidating readings. The judging segment imports '
                                             '`expected_event_counts` / `event_weeks` / '
                                             '`EVENT_RATE_DEFAULTS` from it, which are formulas, not lookup '
                                             'tables; regenerating the item is a stricter consequence than '
                                             'recomputing, so the exemption holds regardless of which '
                                             'symbols cross.',
 'haenv/world/observations/events_streams.py': 'Already in GENERATION, split out of `events.py` with the '
                                               'same standing: the judging segment reads the metric catalog '
                                               '`METRIC_BY_NAME`, and changing it regenerates the packs.',
 'haenv/world/observations/events_text.py': 'Already in GENERATION, split out of `events.py` with the same '
                                            'standing: the judges match condition names with `alias_hit`, '
                                            'and changing it regenerates the packs.',
 'haenv/scoring/matching/excludes.py': 'Already in GENERATION, split out of `overlay.py` with the same '
                                       "standing: the judges' alias matcher (`alias_match`) reads the "
                                       'exclusion table through `alias_excluded_at`; the table it loads, '
                                       '`registry/vocab_alias_excludes.yaml`, is in JUDGING on its own.',
 'haenv/scoring/semantic/source.py': 'In neither segment, split out of `anchor.py` with the same standing: '
                                     'it normalizes a file to the bytes the fingerprints hash. '
                                     '`semantic_seal.code_state` takes `semantic_bytes` from it to seal a '
                                     "pilot's code state, a provenance field.",
 'haenv/world/overlay.py': 'Already in GENERATION. This entry only exempts the `.py` file itself, not the '
                           'data it loads: the twelve registry yaml files `overlay` loads at import time are '
                           'each in JUDGING, and the data-edge check covers them independently of this '
                           'entry. M2 workup also reads HAENV_COMORBID_PAIRS as the component lookup from '
                           'composition_comorbid.yaml, which is covered in JUDGING; no new clinical table is '
                           'exempted.',
 'haenv/resources/registry.py': 'Already in GENERATION. Judges only use `load_test_vocab` / '
                                '`load_test_recall_vocab` / `load_findings` to load yaml, and those yaml '
                                'files are each already registered in one of the two tables on their own.',
 'haenv/world/observations/findings_render.py': 'Already in GENERATION. `gated.py` takes the rendering of '
                                                "the gate's disclosure surface from it; changing it "
                                                'regenerates the pack, and readings are invalidated along '
                                                'with it.',
 'haenv/world/clinical/indicators.py': "Already in GENERATION. `gated._named_on_stream` reads an indicator's "
                                       "printed precision (`of(...)['ndigits']`) to format a named test "
                                       'drawn from a live lab stream; changing it changes the tool result '
                                       'shown to the solver (question surface), no judging field.',
 'haenv/resources/regpath.py': 'Already in GENERATION. `gates.check_cadence` (GEN6, generation time only) '
                               'and `gated._obs_cfg` read '
                               '`registry/physio_streams.yaml:clinical_observation` / '
                               '`clinical_measurement.cv` through `load_registry`: the lab calendar signals, '
                               'and the named-test measurement variation, flag symbols and printed names '
                               '(question surface). M2 workup uses load_registry and registry_cached only as '
                               'a file loader and file-aware cache; its rule file is separately covered in '
                               'JUDGING.',
 'haenv/world/generation/rng.py': 'Already in GENERATION. The deterministic-sampling implementation; '
                                  '`evaluate` uses it for sampling, and changing it changes the item, not '
                                  'the judging.',
 'haenv/runtime/llm.py': 'In neither segment. `tools/evaluation/recompute_judges.py` makes zero references to solver '
                         'construction or `_post` (it only re-judges already-saved responses, sending no '
                         "requests), so changing this module cannot alter any already-saved batch's "
                         'recomputed readings. On the output side, `llm.py` never sets `overall` and '
                         'produces none of the per-dimension denominator fields registered in '
                         '`absence.DENOM_FIELD`; the `ceiling` block it adds is a diagnostic field that no '
                         'judge reads. What it can move: `needs_stream` decides the transport method, and a '
                         'failed transport pushes a newly collected cell to ABORT, so it takes part in new '
                         "data's denominator. That is a measurement condition in the same category as "
                         '`timeout` / `max_retries` / the key pool, not a judging convention, and '
                         'measurement conditions are pinned by the third fingerprint, '
                         '`batch.solving_fingerprint` (`solving_sha16`). In short, changing this module '
                         'changes which data you collect, not how the collected data is judged. '
                         '`haenv/judges/llm.py` also takes `make_dispatcher` from this module, which makes '
                         'it a judging-side transport layer too. The judging convention (the judge prompt / '
                         'the closed set of verdicts / the rate formula) lives in `judges/llm.py`, which is '
                         'in `JUDGING`; this module can move timeout/retry/caching, and retries running out '
                         'push `llmj_disc_status` from `ok` to `dispatch_failed` (`rate` goes from a number '
                         'to `None`), so once the LLM judge is switched on this module can move readings. '
                         'The entry holds because that judge is off by default: `register_llm_judges()` must '
                         'be called explicitly and `HAENV_LLM_JUDGE` must be turned on, and with both off '
                         '`_resolve_dispatch()` always returns `None` and sends no request. If the LLM judge '
                         'enters `_CORE` or is turned on by default, this entry no longer holds, and '
                         '`llm.py` must move into `JUDGING` (or the judging-side dispatcher must be split '
                         'from the solver-side one).',
 'haenv/runtime/call_log.py': '`INFRA`: per-call usage and sampling echo, the token-budget policy and '
                              'billing, split out of `llm.py` so the solver layer no longer imports the '
                              'generation dispatcher. `needs_stream` decides the transport method, and '
                              '`_is_exhausted` decides when a key leaves the pool, both measurement '
                              'conditions pinned by `batch.solving_fingerprint` like `timeout` and the key '
                              'pool; nothing here sets `overall` or a per-dimension denominator field.',
 'haenv/runtime/ops/runtime.py': "In neither segment. `solvers.py` reads the operator's route chain from the "
                                 'hot-reloadable runtime file next to the shared ledger '
                                 '(`for_ledger(...).current()`). It decides which backend a request is '
                                 'routed to and whether new paid requests may start, never what is asked or '
                                 'how an answer is scored; the route actually used is recorded on the row. '
                                 'Until 2026-10-03 the guard resolved `.ops.runtime` against the top-level '
                                 'package, where no `ops` module exists, so this edge went unlisted.',
 'haenv/provenance/anchor.py': 'It computes the judging fingerprint itself, and produces only provenance '
                               'fields (`judging_sha16` / `judge_parts` / `measured_under`), no judging '
                               'field at all. Putting it in the hashed list would make editing a comment '
                               'move the fingerprint, and what a fingerprint move means is "the judge '
                               'changed" -- that conflates the provenance layer with the judging layer. '
                               'Moreover, `anchor.judging_files()` reads this very tuple, so including '
                               '`anchor.py` would make the fingerprint self-referential (change anchor ⇒ '
                               'fingerprint changes ⇒ anchor changes...). What the judging segment actually '
                               'takes from it is only `judging_fp_cached` (from `evaluate`) and '
                               '`judging_fingerprint` / `judging_vintages` (from `report`) -- all '
                               'provenance. Semantic preparation also checks world_fingerprint to refuse '
                               'slice reconstruction under a different world; this is source validation, not '
                               'a scoring rule. `semantic_bytes` and `infra_files` feed `code_state`, the '
                               'run identity a resume compares; they decide whether a run may continue, '
                               "never a cell's value.",
 'haenv/interaction/trace.py': 'A provenance layer: it only records the event log (model output / tool calls '
                               'and returns / gate verdicts), produces no scoring field at all, and the '
                               "judging side reads none of it, so changing it doesn't change any cell's "
                               'score.',
 'haenv/app/artifacts/batch.py': 'Batch registration and provenance fields (sticky `generator_sha`). The '
                                 'judging segment (`evaluate.py`) takes `provenance_fields` and '
                                 "`record_usage`, and `provenance_fields`'s output lands in every "
                                 '`eval.jsonl` row (`haenv_git_sha` / job digest / `kernel_sha256`). The '
                                 'exemption holds because what crosses in is a provenance field: none of it '
                                 'is a scoring dimension, and none of it decides `overall` or any '
                                 "`absence.DENOM_FIELD` denominator field, so changing it doesn't change any "
                                 'reading. Semantic preparation also reads kernel_fingerprint to require '
                                 'byte-identical payload-building code before projecting an old saved slice; '
                                 'it is a provenance admission check, not an imputed score.',
 'haenv/world/generation/build_clinical.py': 'Already in GENERATION; changing it regenerates the packs '
                                             'rather than recomputing readings. The exemption covers this '
                                             '`.py` file only, not the data it loads. The judging side takes '
                                             '`_clinical_baselines`, `_clinical_cv` and `CLINICAL_ATTEN` '
                                             'from it; the registry data behind them '
                                             '(`registry/clinical_baselines.yaml`) is itself in GENERATION, '
                                             'and the gate that uses it, '
                                             '`gates.check_clinical_baseline_cohort`, runs only at '
                                             'generation time from `build.build_case`. `ALLOW_SYMBOL_PIN` '
                                             'pins the crossing symbol set, so a new symbol taken from '
                                             '`build_clinical.py` fails at once instead of being covered by '
                                             'this entry.',
 'haenv/app/config.py': 'The judging side takes `load_cfg` (reads config.yaml), its alias `_lc_p`, and the '
                        'repo-root constant `ROOT`. What it feeds into `real_solver_pool` is the '
                        'unregistered-name warning branch; pool membership is decided only by '
                        "`BASELINE_NAMES` (already in JUDGING), so changing config doesn't change any cell's "
                        'score; `ROOT` is a path constant that `haenv/judges/llm.py` uses to set the '
                        "dispatcher's cache location, and enters no judging field. The crossing symbol set "
                        'is pinned in `ALLOW_SYMBOL_PIN`. Split out of `cli.py` so the judges and the solver '
                        'layer read configuration without importing the command line.',
 'haenv/interaction/prompts.py': 'The solver-visible surface (item prompt templates). Changing it requires '
                                 'rerunning (which costs money), not recomputing -- it enters no judging '
                                 'field; the boundary is the same kind as `inputs/*.job.yaml`, managed by '
                                 'the `job_yaml_sha256` edge. The judging segment takes only the single '
                                 "symbol `PROMPT`, which goes to `evaluate._framings()`'s template registry; "
                                 'the templates themselves are separately managed by the `framing_sha256` '
                                 'fingerprint.',
 'haenv/reporting/provenance_report.py': 'A rendering layer, kept separate from `report.py` so that changing '
                                         "the wording doesn't move the judging fingerprint (see the module "
                                         'docstring). The judging segment takes only the two rendering entry '
                                         'points, `one_line` / `render`.',
 'haenv/scoring/slicing.py': 'Slices are frozen to disk (`slicing.save` / `recover_from_disk` / `resolve`); '
                             "recomputation reads from disk rather than re-slicing, so changing this doesn't "
                             "change an already-saved batch's readings; a changed convention is raised "
                             'immediately by `check_spec`. `evaluate.py` uses only '
                             '`check_spec`/`recover_from_disk`/`load`/`drift`/`resolve`/`save`/`SlicesMissing`, '
                             'all in the batch-construction stage, none of them landing on judging or a '
                             'denominator.',
 'haenv/runtime/budget.py': 'INFRA: shared spending reservations and cap receipts. It admits or refuses paid '
                            'requests and reports cost; it does not interpret a saved response or assign a '
                            'vote, score, gate, or scoring denominator. Exact imported bindings are pinned.',
 'haenv/runtime/pricing.py': 'INFRA: dated tariffs, one-attempt cost bounds and failure/receipt tags. Judge '
                             'parsing and summaries remain in the judging policy. The crossing symbol set is '
                             'exact, not a package exemption.',
 'haenv/locations.py': 'Source/resource/output locators only. Returning a path neither computes a clinical '
                       'value nor grades an answer; the actual read resources retain their declared roles. '
                       'Exact crossing symbols are pinned.',
 'haenv/provenance/declarations.py': 'Formal source-role data read by source identity/seal helpers, not a '
                                     'vote or score implementation. It must stay outside its own content '
                                     'fingerprint. Exact declaration bindings are pinned.',
 'haenv/world/history/history_dependency.py': 'GENERATION: register_history installs the selected pack '
                                              'historical-record callbacks during task registration. '
                                              'Saved-answer row refusal and gates remain in app/packs and '
                                              'their declared scoring owners. Only the exact registration '
                                              'binding crosses.',
 'haenv/app/settings.py': 'Typed configuration contracts for declared query/request settings. They describe '
                          'measurement conditions rather than how stored answers are scored; only exact '
                          'imported types cross, with no executable judging implementation exempted.',
 'haenv/world/protocols/library.py': 'GENERATION/PROTOCOL_WORLD: constructs the declared retrievable '
                                     'clinic-policy documents and applicable tool catalog. Query judges '
                                     'consume the saved observation surface; policy yaml and the actual '
                                     'protocol-scoring implementation are separately in JUDGING. Exact '
                                     'library symbols, not the protocol package, are exempted.',
 'haenv/world/protocols/delivery.py': 'GENERATION/PROTOCOL_WORLD: renders the selected task question and its '
                                      'disclosed protocol documents. Actual saved-answer scoring and query '
                                      'refusal remain in their JUDGING owners; the task-question dispatcher '
                                      'has its own explicit boundary.',
 'haenv/scoring/management/plan.py': 'GENERATION: the query stub uses this producer to construct a baseline '
                                     'management answer. It does not judge an existing answer; sealed '
                                     'management rubric/judge implementations stay in MANAGEMENT_JUDGING. '
                                     'Exact incoming bindings are pinned.',
 'haenv/m2/background.py': 'GENERATION: constructs the M2 background history and measured source surface. '
                           'Its inputs are fixed in emitted cases; the M2 core scoring entry remains '
                           'separately hashed. Only the exact imported background binding crosses.',
 'haenv/world/clinical/med_course.py': 'GENERATION: medication-course records and '
                                       'exposure/routine-measurement helpers; changing them changes emitted '
                                       'worlds. Row decisions remain in task score/gold implementations and '
                                       'no clinical rule or file is exempted by package.',
 'haenv/world/clinical/relations.py': 'GENERATION: clinical relationship derivations of the emitted world. '
                                      'Saved gold/records are the judged inputs; callers import only the '
                                      'pinned helpers rather than any scoring implementation.',
 'haenv/p2/render.py': 'P2_WORLD: renders the stored triage scenario and observed records. Generation '
                       'changes require a new task-world stamp; the P2 row scorer and query result guard '
                       'remain in JUDGING.',
 'haenv/p2/complaints.py': 'P2_WORLD: authors the visible complaint in the task world, not a grading '
                           'function. Exact bindings used to explain/read generated complaint records are '
                           'pinned.',
 'haenv/p2/context.py': 'P2_WORLD: constructs the triage context and encounter surface. It is an '
                        'input-generation module, not a source of score/overall or denominator decisions.',
 'haenv/p2/sources.py': 'P2_WORLD: supplies named clinical source records for the emitted triage payload. '
                        'Only exact source-reader bindings cross; score/rule application remains explicitly '
                        'frozen.',
 'haenv/world/validation/difficulty_r2.py': 'GENERATION: difficulty/visible-input classification and '
                                            'generation checks. It is already a world source; changing a '
                                            'generation acceptance predicate invalidates the task world, not '
                                            'a saved-response score rule.',
 'haenv/p3/render.py': 'P3_WORLD: writes the medication scenario and history text. Exact rendering/input '
                       'helpers are pinned; the P3 score and gold implementations remain judging sources.',
 'haenv/p3/history.py': 'P3_WORLD: creates and locates historical medication records within emitted cases. '
                        'It does not grade a response; exact imported history helpers are pinned.',
 'haenv/p4/diversity.py': 'P4_WORLD: lab-item diversity and generation/calibration inputs. Output decisions '
                          'are scored in the separately registered P4 gold/score modules; this is not a '
                          'general lab-package exemption.',
 'haenv/world/labworld/__init__.py': 'P3_WORLD/P4_WORLD: explicit lab-world resource/contract exports. Exact '
                                     'imported exports are pinned; executable medication gold used by '
                                     'scoring is separately in JUDGING.',
 'haenv/world/validation/invariants.py': 'GENERATION: generation-time input invariants. The saved-answer row '
                                         'refusal and action gates are independently frozen in app/packs and '
                                         'the scoring modules.',
 'haenv/__init__.py': 'Explicit package-level location/resource/kernel-source locators only. Changing these '
                      'bindings does not interpret a response or produce score/overall/denominator fields. '
                      'Exact incoming exported names are pinned; no package-wide exemption is granted.',
 'haenv/world/generation/world_knobs.py': 'GENERATION: source of world-generation parameter declarations. '
                                          'The lab gold helpers take only exact declared constants; emitted '
                                          'input/world identities already cover changes to these parameters.',
 'haenv_kernel/build.py': 'GENERATION: build_instance and leakage_probe construct/check the saved '
                          'solver-visible payload. Their source is a world/input identity; scoring and '
                          'action gates remain separately registered. Exact incoming names are pinned.',
 'haenv_kernel/latent.py': 'GENERATION: public latent-world construction exports from the kernel package. '
                           'They produce case premises, not grades; exact package export bindings are '
                           'pinned.',
 'haenv_kernel/noise.py': 'GENERATION: public observation-noise generation exports. Saved observation inputs '
                          'carry the world identity, and this file does not grade a stored answer.',
 'haenv_kernel/synth.py': 'GENERATION: public synthetic-world rendering exports. The frozen solver payload '
                          'is the input to judges; exact exported construction names are pinned.',
 'haenv_kernel/joint_scenarios.py': 'GENERATION: scenario premises and declared joint-world fixtures. The '
                                    'generated case, not a newly inferred scenario, is the scored object; '
                                    'exact incoming construction/lookup names are pinned.',
 'haenv/m2/audit.py': 'GENERATION: M2 emission/batch audit declarations, including the packaged item audit '
                      'dependency. Marker-based row refusal remains independently hashed in '
                      'app/packs.run_row; no saved-answer score is exempted.',
 'haenv/m2/emr.py': 'GENERATION: renders and annotates the M2 source record. It is part of the emitted world '
                    'and not an atomic answer scorer; only exact callback bindings cross from registration.',
 'haenv/world/validation/pack_size.py': 'GENERATION: declared pack-size and seed allocation for generated '
                                        'inputs. It fixes the item population before responses exist, not '
                                        'the scoring denominator of stored answers; exact audit bindings are '
                                        'pinned.'}

#: Anchor version. Entries in `PENDING_ANCHOR` are only valid for this version;
#: as soon as the anchor advances, this table must be emptied.
PENDING_ANCHOR_REVISION = 105

#: Files added to `JUDGING` before the anchor was updated (allowed while re-anchoring
#: is not); emptied once the anchor moves past `PENDING_ANCHOR_REVISION`.
PENDING_ANCHOR: dict[str, str] = {}

#: Names each `JUDGING_IMPORT_ALLOW` module lets cross into the judging segment
#: (`from .M import ...` binding names); must match exactly.
ALLOW_SYMBOL_PIN: dict[str, frozenset[str]] = {
    'haenv/world/labworld/truth.py': frozenset(('onset',)),
    'haenv/__init__.py': frozenset(('_data_root', '_dr', '_kp', 'data_root')),
    'haenv/app/artifacts/batch.py': frozenset(('kernel_fingerprint', 'provenance_fields', 'record_usage')),
    'haenv/app/artifacts/payloads.py': frozenset(('load_payloads',)),
    'haenv/app/artifacts/store.py': frozenset(('load_cases',)),
    'haenv/app/config.py': frozenset(('ROOT', '_lc_p', 'load_cfg')),
    'haenv/app/settings.py': frozenset(('JudgeIdentity', 'JudgeRouteSpec', 'QuerySection')),
    'haenv/contracts/external_gold.py': frozenset(('EG', 'SLOT', '_ext_probes')),
    'haenv/interaction/prompts.py': frozenset(('PROMPT',)),
    'haenv/interaction/trace.py': frozenset(('TraceLog', '_TraceLog', '_ci', '_rt', '_save_trace', 'save_trace')),
    'haenv/locations.py': frozenset(('data_root', 'output_root', 'probe_directory', 'repository_root', 'source_path', 'source_root')),
    'haenv/m2/audit.py': frozenset(('emission',)),
    'haenv/m2/background.py': frozenset(('laboratory_documents', 'lookalike_record_format', 'sync_known_conditions')),
    'haenv/m2/emr.py': frozenset(('frame_probe', 'install_emr')),
    'haenv/p2/complaints.py': frozenset(('_install_complaints',)),
    'haenv/p2/context.py': frozenset(('finalize',)),
    'haenv/p2/render.py': frozenset(('BP_DAYS', 'RD', 'Unrealizable', 'WINDOW', 'realize', 'visible_medications')),
    'haenv/p2/sources.py': frozenset(('_install_sources',)),
    'haenv/p3/history.py': frozenset(('limiting_record',)),
    'haenv/p3/render.py': frozenset(('render_block',)),
    'haenv/p4/diversity.py': frozenset(('_diversity_table', 'table', 'verification_consistent')),
    'haenv/provenance/anchor.py': frozenset(('_fingerprint_of', '_jvint', '_wvint', 'frozen_packs', 'infra_files', 'judging_fingerprint', 'judging_fp_cached', 'judging_vintages', 'world_fingerprint')),
    'haenv/provenance/declarations.py': frozenset(('JUDGING', 'declarations')),
    'haenv/reporting/provenance_report.py': frozenset(('_prov_1', '_prov_render')),
    'haenv/resources/registry.py': frozenset(('GOLD_EVIDENCE', '_cff', '_lf', '_load_test_vocab', '_load_yaml', 'condition_findings_for_case', 'load_disputed_gold', 'load_findings', 'load_lookalikes', 'load_test_recall_vocab')),
    'haenv/resources/regpath.py': frozenset(('load_registry', 'registry_cached', 'registry_dir', 'registry_path')),
    'haenv/runtime/budget.py': frozenset(('BudgetExceeded', 'BudgetLedger', 'cost_key', 'cost_summary')),
    'haenv/runtime/call_log.py': frozenset(('_EXHAUSTED_MARKERS', '_is_exhausted', '_llm', 'billed_tokens')),
    'haenv/runtime/completion.py': frozenset(('AccountedCompletion', 'ReconciledRequestFailure', 'RejectedBeforeGeneration', 'UnknownCostAttemptFailure', 'begin_inflight', 'end_inflight', 'lane', 'mark_sent', 'never_sent', 'note_generation', 'unbilled_rejection', 'wire_evidence')),
    'haenv/runtime/judge_scheduler.py': frozenset(('MAX_JUDGE_LANES', 'RouteLanes', 'RoutesUnavailable', 'run_bounded')),
    'haenv/runtime/llm.py': frozenset(('_llm', 'make_dispatcher')),
    'haenv/runtime/native_accounting.py': frozenset(('NativePrices', 'VALID_DAYS', 'configure', 'tariff_check', 'utc_today')),
    'haenv/runtime/ops/runtime.py': frozenset(('check_dispatch', 'for_ledger', 'ops_log')),
    'haenv/runtime/pricing.py': frozenset(('PriceSchedule', 'ReconciledCallFailure', 'RejectedCallFailure', 'UnknownCostCallFailure', '_rate', '_rejected_at_zero', 'carry_receipts', 'failure_receipt')),
    'haenv/runtime/relay_accounting.py': frozenset(('RELAY_COST_DEFAULT', 'RELAY_COST_HEADROOM', 'RELAY_COST_PER_GRID', 'RELAY_USD_PER_MTOK', '_measured_tokens_per_grid', 'preflight_quota')),
    'haenv/runtime/scheduler.py': frozenset(('_schedule', '_stall_guard', 'stall_report_lines')),
    'haenv/runtime/solver_accounting.py': frozenset(('fetch_endpoints', 'prepare_accounting', 'wire_settings')),
    'haenv/runtime/transport.py': frozenset(('check_pin', 'check_sampling', 'check_served', 'envelope_content', 'google_response_format', 'is_envelope', 'judge_spec', 'make_envelope', 'model_urlopen', 'proxy_for_solver', 'sent_temperature', 'verify_wire')),
    'haenv/scoring/management/plan.py': frozenset(('M', 'answer')),
    'haenv/scoring/matching/excludes.py': frozenset(('alias_excluded_at',)),
    'haenv/scoring/semantic/budget.py': frozenset(('BudgetExceeded', 'cost_summary')),
    'haenv/scoring/semantic/runref.py': frozenset(('correction_base', 'subset_base', 'view_run')),
    'haenv/scoring/semantic/source.py': frozenset(('semantic_bytes',)),
    'haenv/scoring/slicing.py': frozenset(('_slicing',)),
    'haenv/world/clinical/indicators.py': frozenset(('_declared_ndigits', 'of')),
    'haenv/world/clinical/med_course.py': frozenset(('DISPLAY', 'MC', 'active_medicines', 'display', 'dose_at', 'record_rows')),
    'haenv/world/clinical/relations.py': frozenset(('FRIEDEWALD_TG_DIVISOR', 'FRIEDEWALD_TG_MAX_MMOL')),
    'haenv/world/generation/build_clinical.py': frozenset(('CLINICAL_ATTEN', 'WORLD_MEDICATION_KEY', '_clinical_baselines', '_clinical_cv')),
    'haenv/world/generation/framings.py': frozenset(('DDX_PROMPT', 'DDX_SCOPE2_PROMPT', 'DDX_SCOPE_PROMPT', 'DDX_TRACE_PROMPT', '_BUILTIN_FRAMING_NAMES', '_SCOPE2_EDITS', '_SCOPE_EDITS', '_framings', 'derive_scope2_prompt', 'derive_scope_prompt', 'framing_sha256')),
    'haenv/world/generation/rng.py': frozenset(('rng',)),
    'haenv/world/generation/world_knobs.py': frozenset(('background_enabled',)),
    'haenv/world/history/history_dependency.py': frozenset(('HistoryDependency', 'history_table', 'metadata', 'old_day', 'rate', 'register_history', 'table', 'uniform')),
    'haenv/world/labworld/__init__.py': frozenset(('analyte', 'cv_total', 'rcv95', 'tables')),
    'haenv/world/observations/events.py': frozenset(('E', '_inh', 'scene_lay_sentence', 'sex_words', 'shown_drugs', 'symptom_lay')),
    'haenv/world/observations/events_pools.py': frozenset(('EVENT_RATE_DEFAULTS', 'event_weeks', 'expected_event_counts')),
    'haenv/world/observations/events_streams.py': frozenset(('METRIC_BY_NAME',)),
    'haenv/world/observations/events_text.py': frozenset(('alias_hit', 'relocate_event_history', 'scrub_menses')),
    'haenv/world/observations/findings_render.py': frozenset(('SEX_REF', '_TS_TO_PANEL', '_band', '_copula_index', '_copula_u', '_ppf', '_round', '_step', '_value', 'plausible_floor', 'tc_friedewald_residual', 'with_sex_ref')),
    'haenv/world/observations/streams.py': frozenset(('device_signals',)),
    'haenv/world/overlay.py': frozenset(('HAENV_COMORBID_PAIRS', 'RIVALS', '_cr_pub', '_fac', '_kernel_specs', 'condition_registry', 'haenv_comorbid_specs', 'haenv_independent_specs', 'rivals_for', 'spec_id_of', 'specialty_hit', 'threads_for')),
    'haenv/world/physio/kernel.py': frozenset(('_pk',)),
    'haenv/world/protocols/delivery.py': frozenset(('question',)),
    'haenv/world/protocols/library.py': frozenset(('bearing_ids', 'catalog', 'clinic_lines', 'library', 'search')),
    'haenv/world/validation/difficulty_r2.py': frozenset(('config', 'enabled', 'register')),
    'haenv/world/validation/invariants.py': frozenset(('c3',)),
    'haenv/world/validation/pack_size.py': frozenset(('POOL_FACTOR',)),
    'haenv_kernel/build.py': frozenset(('FORBIDDEN_TOKENS', 'build_instance', 'leakage_probe')),
    'haenv_kernel/joint_scenarios.py': frozenset(('_JS',)),
    'haenv_kernel/latent.py': frozenset(('DISEASE_SIGNAL_DOMAIN', 'DRUG_PKPD', 'KNOWN_DEVICES', 'KNOWN_MISSINGNESS', 'LatentPremise', 'drug_pkpd', 'make_premise', 'validate_premise')),
    'haenv_kernel/noise.py': frozenset(('NOISE_CLASSES', 'POINT_ARTIFACTS', 'PRIMARY', 'context_confound', 'device_switch', 'inject', 'transient_spike', 'trap_evidence_days', 'unit_error')),
    'haenv_kernel/synth.py': frozenset(('AUX_SIGNALS', 'TemplateGenerator', 'premise_conflicts', 'register_aux_signals', 'synthesize')),
}

#: Yaml flagged by the prompt-string co-occurrence screen but traced out of GENERATION;
#: each must still be in `JUDGING`.
PAYLOAD_YAML_ALLOW = {
    "registry/vocab_tests.yaml":
        "The co-occurring string (thyroid-stimulating hormone / hemoglobin "
        "A1c) shares its source with findings.yaml:name_cn; its only "
        "consumer is judges._TESTV, with zero references from GENERATION-side "
        "modules ⇒ belongs in JUDGING.",
}
