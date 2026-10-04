"""make_freeze.py -- writes the frozen code snapshot (a build artifact; do not hand-edit).

    uv run python tools/make_freeze.py --revision 3 --why "..." \
        --pack <job>=<batch> ...

Segment fingerprints are semantic (see `_sem`); `packs[*].sha256` is the raw sha256 of the
pack's `cases.jsonl`, first 16 hex chars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _generation_source(batch_dir: Path, _seen: set | None = None) -> dict:
    """Walk `derived_from` back to the batch that actually generated these cases.

    A restamped batch re-registers with the current `generator_sha`, so only the root
    batch's fingerprint is true. A broken chain returns `chain_ok: False`.
    """
    _seen = _seen or set()
    bj = batch_dir / "batch.json"
    if not bj.is_file():
        return {"root_batch": batch_dir.name, "chain_ok": False,
                "why": "This batch has no batch.json -- the generation source is unknowable"}
    try:
        meta = json.loads(bj.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"root_batch": batch_dir.name, "chain_ok": False, "why": f"Could not read batch.json: {e}"}
    parent = meta.get("derived_from")
    if parent and parent not in _seen:
        pdir = batch_dir.parent / str(parent)
        if pdir.is_dir():
            up = _generation_source(pdir, _seen | {parent})
            up["via"] = [batch_dir.name] + list(up.get("via") or [])
            return up
        return {"root_batch": str(parent), "chain_ok": False,
                "why": f"Ancestor batch directory does not exist ({parent}) -- generation-source chain is broken, cannot trace back",
                "via": [batch_dir.name]}
    return {"root_batch": batch_dir.name, "chain_ok": True,
            "generator_sha": meta.get("generator_sha"),
            "generation_git_sha": meta.get("generation_git_sha") or meta.get("haenv_git_sha"),
            "generator": meta.get("generator")}

#: Generation segment: changing a file here changes the world (`world_sha`) and the
#: pack must be regenerated. A registry yaml belongs here when its strings reach the
#: prompt without passing through `inputs/*.job.yaml`, which `job_yaml_sha256` covers.
GENERATION = (
    "haenv_kernel/build.py", "haenv_kernel/latent.py", "haenv_kernel/synth.py", "haenv_kernel/noise.py",
    "haenv_kernel/joint_scenarios.py",
    "haenv/build.py", "haenv/build_weight.py", "haenv/build_clinical.py",
    "haenv/events.py", "haenv/events_text.py", "haenv/events_pools.py",
    "haenv/events_streams.py", "haenv/findings_render.py",
    "haenv/relations.py", "haenv/verify.py", "haenv/overlay.py",
    "haenv/registry.py", "haenv/rng.py",
    "haenv/wearable.py", "haenv/med_course.py",
    "haenv/framings.py",
    "haenv/yamlcache.py",
    "haenv/regpath.py",
    "haenv/world_plugins.py",
    "haenv/post_inject.py",
    "haenv/latent_rules.py",
    "haenv/job.py",
    "haenv/gated.py",
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
    "registry/benign_events.yaml", "registry/symptom_lay.yaml", "registry/drug_side_effects.yaml",
    "registry/condition_age_limits.yaml",
    "registry/clinical_baselines.yaml",
    "registry/indicators.yaml",
    "registry/streams.yaml", "haenv/streams.py",
    "haenv/physio/__init__.py", "haenv/physio/apply.py", "haenv/physio/audit.py",
    "haenv/physio/bounds.py", "haenv/physio/coupling.py", "haenv/physio/kernel.py",
    "haenv/physio/noise.py", "haenv/physio/superpose.py",
    "haenv/indicators.py",
    "registry/artifact_rates.yaml", "haenv/artifact_params.py",
    "registry/drug_effects.yaml", "haenv/drug_effects.py",
    # split out of `overlay.py` / `job.py` / `drug_effects.py` / `events.py` (2026-10-03),
    # moved verbatim
    "haenv/alias_excludes.py", "haenv/job_schema.py", "haenv/drug_schema.py",
    "haenv/world_knobs.py",
    # Gold qualifiers (core + qualifier split, derivability on the visible instance).
    "registry/gold_qualifiers.yaml", "haenv/qualifiers.py",
)
#: Judging segment: changing a file here changes readings recomputed from saved
#: responses, so affected batches must be recomputed. Anything a member imports (code or
#: registry data) must be in the segment or in `JUDGING_IMPORT_ALLOW` /
#: `REGISTRY_YAML_OUT_OF_SCOPE` with a reason.
JUDGING = (
    "haenv_kernel/verifier.py", "haenv_kernel/gatekeeper.py", "haenv_kernel/schema.py",
    "haenv_kernel/runner.py", "haenv_kernel/solver.py",
    "haenv/llm_rubric.py", "registry/rubric_points.yaml",
    "haenv/semantic_judge.py", "haenv/semantic_inputs.py", "haenv/semantic_rubric.py",
    "haenv/semantic_pipeline.py",
    "haenv/semantic_report.py",
    "haenv/judge_compaction.py",
    "haenv/judge_evidence.py",
    "haenv/semantic_visibility.py", "haenv/semantic_corrections.py",
    "haenv/semantic_lean.py", "registry/semantic_judging_v4.yaml",
    "haenv/semantic_atom_correction.py", "registry/semantic_judging_second.yaml",
    "registry/semantic_judging_why.yaml", "registry/semantic_judging_why_lean.yaml",
    "registry/semantic_judging_source.yaml",
    "registry/semantic_judging.yaml",
    "haenv/mounting.py",
    "haenv/yamlcache.py",
    "haenv/latent_rules.py",
    "haenv/judges/__init__.py", "haenv/judges/_helpers.py",
    "haenv/judges/differential.py", "haenv/judges/outcome.py",
    "haenv/judges/trajectory.py", "haenv/judges/safety.py",
    "haenv/judges/llm.py", "haenv/judges/recall_match.py",
    "haenv/gates.py", "haenv/gates_case.py", "haenv/gates_outcome.py", "haenv/gates_shortcut.py", "haenv/scoring.py",
    "haenv/process.py", "haenv/wq.py", "haenv/report.py",
    "haenv/analytics.py",
    "haenv/evaluate.py",
    # split out of `evaluate.py` (2026-10-03); same judging code, moved verbatim
    "haenv/run_state.py", "haenv/qside.py", "haenv/run_ledger.py", "haenv/solvers.py",
    "haenv/rows.py",
    # definitions moved down so the import graph has no cycle through the judging core
    # (2026-10-03), moved verbatim
    "haenv/alias_match.py", "haenv/mount_coverage.py", "haenv/row_store.py",
    "haenv/semantic_seal.py", "haenv/semantic_coverage.py", "haenv/rhythm.py",
    "haenv/resume.py", "haenv/gate_tables.py",
    "haenv/gold_kinds.py",
    "haenv/separation.py", "haenv/mount_table.py", "haenv/tracks.py",
    "haenv/baselines.py", "haenv/quantities.py", "haenv/absence.py",
    # M1 2026-10-01: question-blind ceilings for every core dimension (imported by analytics)
    "haenv/blind_ceilings.py",
    "haenv/ranking.py", "haenv/gated.py",
    "haenv/solve_guard.py",
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
)

#: M2 judging segment (feat/m2-impl): the round-2 hook, the M2 atoms and composite, the frame judge.
#: It is stamped on M2 rows as `m2_judging_sha16` (`haenv/m2/hooks.py`) and never enters
#: `judging_sha16`, so v1.0.1 and F0 bridge cells keep the integration judging fingerprint.
#: `haenv/m2/core.py` is also the M2 world's stamp (external-gold manifest); the M2 emission gate is
#: fingerprinted there by its declaration, `haenv/m2/audit.py`. `haenv/pack_skeleton.py` (judge
#: mounting and the publish gate, shared by every pack) is in each pack's judging segment and
#: `haenv/shared_audit.py` (the shared emission gate) in each pack world segment.
M2_JUDGING = (
    "haenv/pack_skeleton.py",
    "haenv/m2/hooks.py", "haenv/m2/round2.py", "haenv/m2/score.py", "haenv/m2/plugin.py",
    "haenv/m2/core.py", "registry/explain_away.yaml", "registry/base_condition_findings.yaml",
)

#: Pack 2 (acute triage) segments (feat/pack2-triage). Neither enters `GENERATION` or
#: `JUDGING`, so jobs that do not declare the `haenv.pack2` plugin keep their `world_sha` and
#: `judging_sha16`. `PACK2_WORLD` is the pack-2 world (snapshot, event pool, G2 table, gate,
#: question template): its fingerprint is written into every pack-2 gold
#: (`adjudication.triage.world_sha16`) and into the registered `pack2.triage` probe, whose hash
#: enters the external-gold manifest, so a pack-2 world change moves the pack-2 `world_sha`.
#: `PACK2_JUDGING` (judge, hooks, composite) is stamped on pack-2 rows as `pack2_judging_sha16`.
PACK2_WORLD = (
    "haenv/shared_audit.py",
    "haenv/pack2/world.py", "haenv/pack2/vitals.py", "haenv/pack2/events.py", "haenv/pack2/grading.py",
    "haenv/pack2/_util.py", "haenv/pack2/framing.py",
    "registry/pack2_triage_thresholds.yaml", "registry/pack2_acute_events.yaml",
    # the build hook that calls `world.finalize_adjudication` (it decides the re-armed gold
    # fields), so it is in both pack-2 segments
    "haenv/pack2/hooks.py",
)
PACK2_JUDGING = (
    "haenv/pack_skeleton.py",
    "haenv/pack2/hooks.py", "haenv/pack2/plugin.py", "haenv/pack2/score.py",
)

#: Pack 4 (follow-up interpretation, feat/pack4-followup). Two segments of their own, neither in
#: `JUDGING` nor in `GENERATION`, so every existing job keeps `judging_sha16` and `world_sha`:
#: `P4_JUDGING` is stamped on pack-4 rows as `p4_judging_sha16` (`haenv/p4/hooks.py`); `P4_WORLD`
#: is stamped on the pack-4 gold block as `p4_world_sha16` and enters `world_sha` of a pack-4
#: batch through the registered probe (`haenv/p4/plugin.py:probe_dict`). `haenv/p4/gold.py` is in
#: both: the generator accepts an item by it and the judge reads its output.
P4_JUDGING = (
    "haenv/pack_skeleton.py",
    "haenv/p4/score.py", "haenv/p4/plugin.py", "haenv/p4/hooks.py", "haenv/p4/gold.py",
)
P4_WORLD = (
    "haenv/shared_audit.py",
    "haenv/labworld/__init__.py", "haenv/labworld/truth.py", "haenv/labworld/observe.py",
    "haenv/labworld/perturb.py", "registry/labworld.yaml",
    "haenv/p4/plan.py", "haenv/p4/realize.py", "haenv/p4/render.py", "haenv/p4/gold.py",
    "haenv/p4/gate.py", "haenv/p4/features.py",
)

#: Pack 3 (chronic medication adjustment, feat/pack3-meds). Same arrangement as pack 4: neither
#: segment is in `JUDGING` or `GENERATION`; `P3_JUDGING` is stamped on pack-3 rows as
#: `p3_judging_sha16`, `P3_WORLD` on the pack-3 gold block as `p3_world_sha16` and enters `world_sha`
#: of a pack-3 batch through the registered probe. It shares three lab-world modules with
#: `P4_WORLD` (read, not changed, by pack 3).
P3_JUDGING = (
    "haenv/pack_skeleton.py",
    "haenv/p3/score.py", "haenv/p3/plugin.py", "haenv/p3/hooks.py", "haenv/p3/gold.py",
    "haenv/labworld/meds.py", "registry/p3_meds.yaml", "registry/control_targets.yaml",
)
P3_WORLD = (
    "haenv/shared_audit.py",
    "haenv/labworld/__init__.py", "haenv/labworld/truth.py", "haenv/labworld/observe.py",
    "haenv/labworld/meds.py", "registry/p3_meds.yaml", "registry/control_targets.yaml",
    "haenv/p3/plan.py", "haenv/p3/realize.py", "haenv/p3/render.py", "haenv/p3/gold.py",
    "haenv/p3/gate.py",
)

#: Infrastructure segment: request accounting, cost receipts, the shared cap, paid slots,
#: scheduling and the pre-send wire check. A change here cannot alter what a solver is
#: shown, how it is asked or how a cell is scored, so it is recorded (`code_upgrades`)
#: and does not invalidate a board. The content-level guard is the pre-send wire check
#: (`haenv/transport.py:verify_wire`), which refuses to send any request whose messages or
#: answer settings differ from what was accounted. `haenv/judge_scheduler.py` and
#: `haenv/semantic_parallel.py` stay in `JUDGING`: whether a third vote is bought depends
#: on the first two votes.
INFRA = (
    "haenv/solver_accounting.py", "haenv/paid_completion.py",
    "haenv/relay_accounting.py", "haenv/native_accounting.py",
    "haenv/paid_slots.py", "haenv/semantic_budget.py", "haenv/semantic_transport.py",
    "haenv/run_scheduler.py", "haenv/transport.py",
    # judge lane scheduling and per-cell result files: how cells are run and written, not
    # what a vote or a score is (consensus and rubric summaries stay in `semantic_judge` /
    # `semantic_rubric`)
    "haenv/judge_scheduler.py", "haenv/semantic_parallel.py",
    # per-call bookkeeping shared by solvers and the generation dispatcher: usage and
    # sampling echo, the token-budget policy (`needs_stream` picks the transport), billing
    "haenv/call_log.py",
)

#: INFRA modules must not import these (AST-checked): scoring, gates, judges and the
#: prompt/payload builders decide what is asked and how it is scored.
INFRA_FORBIDDEN_IMPORTS = ("haenv.judges", "haenv.scoring", "haenv.gates", "haenv.prompts",
                           "haenv.payloads", "haenv.framings")

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
)

#: Registry yaml in neither segment, with the reason the judging segment cannot reach it.
REGISTRY_YAML_OUT_OF_SCOPE: dict[str, str] = {
    "registry/p3_meds.yaml":
        "Pack-3 managed-drug table (ladders, effects, time constants, adverse effects, rule "
        "constants). Read only by `haenv/labworld/meds.py` and `haenv/p3/` (segments `P3_WORLD` / "
        "`P3_JUDGING`); a job that does not load `haenv.p3` never opens it; it enters a pack-3 "
        "batch's `world_sha` through the registered probe.",
    "registry/control_targets.yaml":
        "Pack-3 control targets (axis, CV, target). Same scope and stamp as `registry/p3_meds.yaml`.",
    "registry/pack2_triage_thresholds.yaml":
        "Pack-2 grading table G2 (provisional). Read only by `haenv/pack2/world.py`, which no "
        "`JUDGING` module imports; it is anchored by `PACK2_WORLD` instead, and its content is "
        "hashed into every pack-2 gold (`tables_sha16`, `world_sha16`). If a `JUDGING` module "
        "ever reads it, this exemption has to come off first.",
    "registry/pack2_acute_events.yaml":
        "Pack-2 acute-event pool and wording. Same boundary as `pack2_triage_thresholds.yaml`: "
        "read only by `haenv/pack2/world.py` and the job composer `tools/pack2_gen_job.py`, "
        "anchored by `PACK2_WORLD`.",
    "registry/labworld.yaml":
        "Pack-4 lab-world table (follow-up analytes, perturbations, visible-cause effects). Read "
        "only by `haenv/labworld/` and `haenv/p4/` (segments `P4_WORLD` / `P4_JUDGING`); no file "
        "of the judging segment reads it, and a job that does not load `haenv.p4` never opens it. "
        "Its fingerprint is stamped on pack-4 gold (`labworld_tables_sha16`) and enters a pack-4 "
        "batch's `world_sha` through the registered probe.",
    "registry/explain_away.yaml":
        "Pack-1 (M2) explained-away symptom table. Read only by `haenv/m2/`, which no `JUDGING` "
        "module imports; anchored by `M2_JUDGING` (stamped on M2 rows as `m2_judging_sha16`).",
    "registry/base_condition_findings.yaml":
        "Pack-1 (M2) base condition -> declared findings. Read only by `haenv/m2/core.py`; same "
        "boundary and anchor (`M2_JUDGING`) as `registry/explain_away.yaml`.",
    "registry/acute_events.yaml":
        "Pack-1 (M2) F1 day-T acute-event pool. Read only by `haenv/m2/core.py`, which no "
        "`JUDGING` module imports. Not yet in `M2_JUDGING`: the segment assignment of `haenv/m2/` "
        "is left to the next re-anchor (module docstring of `haenv/m2/core.py`); until then an "
        "edit to this table moves no stamp.",
    "registry/knobs.yaml":
        "Input-surface registry: why each module-level constant lives in "
        "code rather than config. Its only production consumer is "
        "`haenv/inputs.py` (the read-only `haenv inputs` command); the judging "
        "segment makes zero calls into it, so changing it doesn't move any "
        "cell's value. It must not go into `JUDGING`: it describes the judging "
        "segment's own constants, so including it would let a sentence "
        "explaining a threshold move `judging_sha16` and invalidate a paid "
        "leaderboard (editing a comment must not invalidate the anchor). If "
        "judging code ever reads it (say, deciding whether to score a "
        "dimension based on `class`), this exemption has to come off first.",
    "registry/disease_prevalence.yaml":
        "Prevalence table for the composition-v2 allocator. Its only production consumer is "
        "`tools/gen_composition_v2.py` (and `registry.association_admissible`, called by that tool), "
        "which runs at job-generation time -- its output is written into `inputs/*.job.yaml`, out of the "
        "judging segment's reach; the judging segment makes zero calls into it. Same boundary as "
        "`tiers.yaml`. If judging code ever reads a prevalence, this exemption has to come off first.",
    "registry/tiers.yaml":
        "Tier ratios / slicing geometry / gated budgets. The `authoritative:` "
        "section's only consumer is `haenv/ddx.py:_tier`, which runs at "
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
        "consumer is `haenv/ddx.py:_sex_ratio_table` -> `_sex_of`, which runs "
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
        "Its only consumer is `haenv/demographics.py`, which runs at "
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
        "is `haenv/demographics.py`, which runs at job-generation time "
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

#: haenv modules the judging segment imports but that are not in `JUDGING`, with the
#: reason. An entry must not decide `overall` or any `absence.DENOM_FIELD` denominator;
#: its crossing symbols are pinned in `ALLOW_SYMBOL_PIN`.
JUDGING_IMPORT_ALLOW: dict[str, str] = {
    "haenv_kernel/build.py":
        "`GENERATION` segment, reached from `haenv_kernel/runner.py` by "
        "`build_instance` / `leakage_probe`. Always so: `runner` has imported the "
        "kernel's `build` since the fork, and the dependency is unremarkable -- what "
        "made it visible is that the kernel became a package, so the import is now "
        "`from .build import ...` and the judging-segment guard can resolve it. It "
        "stays out of `JUDGING` because changing it changes what the solver is shown, "
        "which is a question change (`world_sha`), not a score change; that "
        "classification predates this entry (see `haenv_kernel/FORKED_FROM.md`). The leak "
        "probe's own behaviour is pinned separately by `tools/leak_probe_selftest.py`.",
    "haenv/semantic_parallel.py":
        "`INFRA` segment (see `INFRA`): the lane scheduler and the per-cell result files of a "
        "semantic run. The pipeline imports the lane cap and the cell runner; votes come from "
        "`semantic_judge.evaluate_consensus` and summaries from `semantic_rubric`, both in the "
        "judging segment. The crossing symbols are pinned in `ALLOW_SYMBOL_PIN`.",
    "haenv/paid_completion.py":
        "`INFRA` segment (see `INFRA`): the streaming reader records the first chunk's "
        "generation id (`note_generation`) on the in-flight marker, so a cut stream can be "
        "settled from the provider's record later. It returns nothing and changes no byte "
        "of a response, a vote or a score. The crossing symbol is pinned in `ALLOW_SYMBOL_PIN`.",
    "haenv/semantic_budget.py":
        "`INFRA` segment (see `INFRA`): request accounting, the shared cap and the pre-send "
        "wire check. Judging code imports only the named symbols (a budget stop, the cap "
        "ledger, price bounds, the paid entry point); none of them produces, filters or "
        "alters a response, a vote or a score. The crossing symbols are pinned in "
        "`ALLOW_SYMBOL_PIN`.",
    "haenv/semantic_transport.py":
        "`INFRA` segment (see `INFRA`): request accounting, the shared cap and the pre-send "
        "wire check. Judging code imports only the named symbols (a budget stop, the cap "
        "ledger, price bounds, the paid entry point); none of them produces, filters or "
        "alters a response, a vote or a score. The crossing symbols are pinned in "
        "`ALLOW_SYMBOL_PIN`.",
    "haenv/relay_accounting.py":
        "`INFRA` segment (see `INFRA`): relay tariff and billing-log reads. Judging code "
        "imports the balance gate (`preflight_quota`), which decides whether a run may "
        "start spending, and re-exports the tariff it prices with (`RELAY_COST_*`, "
        "`RELAY_USD_PER_MTOK`, `_measured_tokens_per_grid`); none of them changes a byte of a "
        "response, a vote or a score. The crossing symbols are pinned in `ALLOW_SYMBOL_PIN`.",
    "haenv/solver_accounting.py":
        "`INFRA` segment (see `INFRA`): request accounting, the shared cap and the pre-send "
        "wire check. Judging code imports only the named symbols (a budget stop, the cap "
        "ledger, price bounds, the paid entry point); none of them produces, filters or "
        "alters a response, a vote or a score. The crossing symbols are pinned in "
        "`ALLOW_SYMBOL_PIN`.",
    "haenv/transport.py":
        "`INFRA` segment (see `INFRA`): request accounting, the shared cap and the pre-send "
        "wire check. Judging code imports only the named symbols: the pre-send wire check; "
        "the validators of a model's upstream pin and sampling block, which refuse a "
        "configuration before any request; the temperature a request carries, read from that "
        "block; the served-upstream check, which fails an attempt a pinned route says another "
        "upstream served (the attempt is retried or the cell has no answer, as for any failed "
        "attempt); and `judge_spec`, which drops the solver-only keys so a judge's request is "
        "what it was before they existed. None of them produces or alters a response, a vote "
        "or a score. The crossing symbols are pinned in `ALLOW_SYMBOL_PIN`.",
    "haenv/semantic_runref.py":
        "Only locates the sealed semantic run a manifest or view names -- by path, else by "
        "run_id / tasks digest / manifest digest -- and refuses a run whose bytes or identity "
        "differ and any ambiguous match. Every run it returns is then read through the unchanged "
        "checks (manifest digest, tasks digest, per-cell recomputation of the persisted votes); "
        "no score, vote or metric is produced or altered here. The crossing symbols are pinned "
        "in `ALLOW_SYMBOL_PIN`.",
    "haenv/run_scheduler.py":
        "Acts only while solving: which cell runs when, how many run at once, and when a "
        "live request that outlived its model's stall deadline is closed and retried. "
        "Recomputing a batch reads saved responses and never passes through it, so no "
        "recomputed reading can change with it. Every cell still runs the same function "
        "once; offline rows are identical under serial, per-backend and per-model "
        "scheduling. A stall close is recorded on the row "
        "(`stall_timeouts`), so the answers it affected stay identifiable.",
    "haenv/payloads.py":
        "Semantic rejudging reads load_payloads only: a strict JSON/dataclass decoder "
        "of source files hashed by the run manifest. No scoring or payload synthesis "
        "function crosses this boundary; if decoding starts transforming fields, "
        "this exemption must be replaced by judging coverage.",
    "haenv/store.py":
        "Semantic rejudging uses load_cases only to deserialize saved RawCase and "
        "restore the saved Q-side ledger; it neither regenerates nor substitutes "
        "missing cases. The source is hashed and required case IDs are checked. "
        "No generation or scoring function crosses this boundary.",
    "haenv/streams.py":
        "Already in the GENERATION segment. Only feeds gates.DEVICE_SIGNALS, read by launch gate GEN7 at build time: a change "
        "alters which cases are emitted (world fingerprint), never the score recomputed from "
        "a saved response. Stream prices reach scoring through the generated "
        "registry/gated_pricing_streams.yaml, which is in JUDGING.",
    "haenv/external_gold.py":
        "Only holds default-empty registries (BLOCKS / FRAMINGS / PROBES) and "
        "write entry points, and judges nothing itself; with empty registries "
        "the judging path is byte-for-byte unchanged. When non-empty, which "
        "external piece is plugged in is traceable through job.yaml's "
        "`plugins:` group name via `job_sha256`, so the judging fingerprint "
        "doesn't need to cover it separately.",
    "haenv/framings.py":
        "Already in the GENERATION segment: changing it invalidates the "
        "item (requires regenerating and rerunning), which is a stricter "
        "consequence than invalidating readings, so the judging fingerprint "
        "doesn't need to cover it too. Recomputed readings from the same batch "
        "of saved responses are unchanged -- changing the prompt doesn't "
        "change the judging. The judging segment only reaches it through "
        "`evaluate.py`'s re-export layer; the symbols that cross are pinned "
        "in `ALLOW_SYMBOL_PIN`.",
    "haenv/events.py":
        "Already in GENERATION: changing it regenerates the packs. The judging segment takes "
        "`inherited_profile` (as `_inh`) from it, the kernel's inherited event profile read "
        "where the plugin pools are rebound.",
    "haenv/events_pools.py":
        "Already in GENERATION: changing it invalidates the whole freeze (the "
        "pack must be regenerated), which already implies invalidating "
        "readings. The judging segment imports `expected_event_counts` / "
        "`event_weeks` / `EVENT_RATE_DEFAULTS` from it, which are "
        "formulas, not lookup tables; regenerating the item is a stricter "
        "consequence than recomputing, so the exemption holds regardless of "
        "which symbols cross.",
    "haenv/events_streams.py":
        "Already in GENERATION, split out of `events.py` with the same standing: the judging "
        "segment reads the metric catalog `METRIC_BY_NAME`, and changing it regenerates the packs.",
    "haenv/events_text.py":
        "Already in GENERATION, split out of `events.py` with the same standing: the judges "
        "match condition names with `alias_hit`, and changing it regenerates the packs.",
    "haenv/alias_excludes.py":
        "Already in GENERATION, split out of `overlay.py` with the same standing: the judges' "
        "alias matcher (`alias_match`) reads the exclusion table through `alias_excluded_at`; "
        "the table it loads, `registry/vocab_alias_excludes.yaml`, is in JUDGING on its own.",
    "haenv/semantic_source.py":
        "In neither segment, split out of `anchor.py` with the same standing: it normalizes "
        "a file to the bytes the fingerprints hash. `semantic_seal.code_state` takes "
        "`semantic_bytes` from it to seal a pilot's code state, a provenance field.",
    "haenv/overlay.py":
        "Already in GENERATION. This entry only exempts the `.py` file "
        "itself, not the data it loads: the twelve registry yaml files "
        "`overlay` loads at import time are each in JUDGING, and the data-edge "
        "check covers them independently of this entry.",
    "haenv/registry.py":
        "Already in GENERATION. Judges only use `load_test_vocab` / "
        "`load_test_recall_vocab` / `load_findings` to load yaml, and those yaml files are "
        "each already registered in one of the two tables on their own.",
    "haenv/findings_render.py":
        "Already in GENERATION. `gated.py` takes the rendering of the gate's "
        "disclosure surface from it; changing it regenerates the pack, and "
        "readings are invalidated along with it.",
    "haenv/indicators.py":
        "Already in GENERATION. `gated._named_on_stream` reads an indicator's printed precision "
        "(`of(...)['ndigits']`) to format a named test drawn from a live lab stream; changing it "
        "changes the tool result shown to the solver (question surface), no judging field.",
    "haenv/regpath.py":
        "Already in GENERATION. `gates.check_cadence` (GEN6, generation time only) and "
        "`gated._obs_cfg` read `registry/physio_streams.yaml:clinical_observation` / "
        "`clinical_measurement.cv` through `load_registry`: the lab calendar signals, and the "
        "named-test measurement variation, flag symbols and printed names (question surface).",
    "haenv/rng.py":
        "Already in GENERATION. The deterministic-sampling implementation; "
        "`evaluate` uses it for sampling, and changing it changes the item, "
        "not the judging.",
    "haenv/llm.py":
        "In neither segment. `tools/recompute_judges.py` makes zero "
        "references to solver construction or `_post` (it only re-judges "
        "already-saved responses, sending no requests), so changing this "
        "module cannot alter any already-saved batch's recomputed readings. "
        "On the output side, `llm.py` never sets `overall` and produces none "
        "of the per-dimension denominator fields registered in "
        "`absence.DENOM_FIELD`; the `ceiling` block it adds is a diagnostic "
        "field that no judge reads. What it can move: `needs_stream` decides "
        "the transport method, and a failed transport pushes a newly "
        "collected cell to ABORT, so it takes part in new data's denominator. "
        "That is a measurement condition in the same category as `timeout` / "
        "`max_retries` / the key pool, not a judging convention, and "
        "measurement conditions are pinned by the third fingerprint, "
        "`batch.solving_fingerprint` (`solving_sha16`). In short, changing "
        "this module changes which data you collect, not how the collected "
        "data is judged. `haenv/judges/llm.py` also takes `make_dispatcher` "
        "from this module, which makes it a judging-side transport layer too. "
        "The judging convention (the judge prompt / the closed set of "
        "verdicts / the rate formula) lives in `judges/llm.py`, which is in "
        "`JUDGING`; this module can move timeout/retry/caching, and retries "
        "running out push `llmj_disc_status` from `ok` to `dispatch_failed` "
        "(`rate` goes from a number to `None`), so once the LLM judge is "
        "switched on this module can move readings. The entry holds because "
        "that judge is off by default: `register_llm_judges()` must be called "
        "explicitly and `HAENV_LLM_JUDGE` must be turned on, and with both off "
        "`_resolve_dispatch()` always returns `None` and sends no request. If "
        "the LLM judge enters `_CORE` or is turned on by default, this entry "
        "no longer holds, and `llm.py` must move into `JUDGING` (or the "
        "judging-side dispatcher must be split from the solver-side one).",
    "haenv/call_log.py":
        "`INFRA`: per-call usage and sampling echo, the token-budget policy and billing, "
        "split out of `llm.py` so the solver layer no longer imports the generation "
        "dispatcher. `needs_stream` decides the transport method, and `_is_exhausted` decides "
        "when a key leaves the pool, both measurement conditions pinned by "
        "`batch.solving_fingerprint` like `timeout` and the key pool; nothing here sets "
        "`overall` or a per-dimension denominator field.",
    "haenv/ops/runtime.py":
        "In neither segment. `solvers.py` reads the operator's route chain from the "
        "hot-reloadable runtime file next to the shared ledger (`for_ledger(...).current()`). "
        "It decides which backend a request is routed to and whether new paid requests may "
        "start, never what is asked or how an answer is scored; the route actually used is "
        "recorded on the row. Until 2026-10-03 the guard resolved `.ops.runtime` against the top-level "
        "package, where no `ops` module exists, so this edge went unlisted.",
    "haenv/anchor.py":
        "It computes the judging fingerprint itself, and produces only "
        "provenance fields (`judging_sha16` / `judge_parts` / "
        "`measured_under`), no judging field at all. Putting it in the "
        "hashed list would make editing a comment move the fingerprint, and "
        "what a fingerprint move means is \"the judge changed\" -- that "
        "conflates the provenance layer with the judging layer. Moreover, "
        "`anchor.judging_files()` reads this very tuple, so "
        "including `anchor.py` would make the fingerprint self-referential "
        "(change anchor ⇒ fingerprint changes ⇒ anchor changes...). What the "
        "judging segment actually takes from it is only `judging_fp_cached` "
        "(from `evaluate`) and `judging_fingerprint` / `judging_vintages` "
        "(from `report`) -- all provenance. Semantic preparation also checks "
        "world_fingerprint to refuse slice reconstruction under a different world; "
        "this is source validation, not a scoring rule. `semantic_bytes` and "
        "`infra_files` feed `code_state`, the run identity a resume compares; "
        "they decide whether a run may continue, never a cell's value.",
    "haenv/trace.py":
        "A provenance layer: it only records the event log (model output / "
        "tool calls and returns / gate verdicts), produces no scoring field "
        "at all, and the judging side reads none of it, so changing it "
        "doesn't change any cell's score.",
    "haenv/batch.py":
        "Batch registration and provenance fields (sticky `generator_sha`). "
        "The judging segment (`evaluate.py`) takes `provenance_fields` and "
        "`record_usage`, and `provenance_fields`'s output lands in every "
        "`eval.jsonl` row (`haenv_git_sha` / job digest / `kernel_sha256`). "
        "The exemption holds because what crosses in is a provenance field: "
        "none of it is a scoring dimension, and none of it decides `overall` "
        "or any `absence.DENOM_FIELD` denominator field, so changing it "
        "doesn't change any reading. Semantic preparation also reads kernel_fingerprint "
        "to require byte-identical payload-building code before projecting an old saved slice; "
        "it is a provenance admission check, not an imputed score.",
    "haenv/build_clinical.py":
        "Already in GENERATION; changing it regenerates the packs rather than recomputing readings. "
        "The exemption covers this `.py` file only, not the data it loads. The judging side takes "
        "`_clinical_baselines`, `_clinical_cv` and `CLINICAL_ATTEN` from it; the registry data behind "
        "them (`registry/clinical_baselines.yaml`) is itself in GENERATION, and the gate that uses it, "
        "`gates.check_clinical_baseline_cohort`, runs only at generation time from `build.build_case`. "
        "`ALLOW_SYMBOL_PIN` pins the crossing symbol set, so a "
        "new symbol taken from `build_clinical.py` fails at once instead of being covered by this entry.",
    "haenv/config.py":
        "The judging side takes `load_cfg` (reads config.yaml), its alias "
        "`_lc_p`, and the repo-root constant `ROOT`. What it feeds into "
        "`real_solver_pool` is the unregistered-name warning branch; pool "
        "membership is decided only by `BASELINE_NAMES` (already in JUDGING), "
        "so changing config doesn't change any cell's score; `ROOT` is a path "
        "constant that `haenv/judges/llm.py` uses to set the dispatcher's "
        "cache location, and enters no judging field. The crossing symbol "
        "set is pinned in `ALLOW_SYMBOL_PIN`. Split out of `cli.py` so the judges and the "
        "solver layer read configuration without importing the command line.",
    "haenv/prompts.py":
        "The solver-visible surface (item prompt templates). Changing it "
        "requires rerunning (which costs money), not recomputing -- it "
        "enters no judging field; the boundary is the same kind as "
        "`inputs/*.job.yaml`, managed by the `job_yaml_sha256` edge. The "
        "judging segment takes only the single symbol `PROMPT`, which goes "
        "to `evaluate._framings()`'s template registry; the templates "
        "themselves are separately managed by the `framing_sha256` "
        "fingerprint.",
    "haenv/provenance_report.py":
        "A rendering layer, kept separate from `report.py` so that changing "
        "the wording doesn't move the judging fingerprint (see the module "
        "docstring). The judging segment takes only "
        "the two rendering entry points, `one_line` / `render`.",
    "haenv/slicing.py":
        "Slices are frozen to disk (`slicing.save` / "
        "`recover_from_disk` / `resolve`); recomputation reads from disk "
        "rather than re-slicing, so changing this doesn't change an "
        "already-saved batch's readings; a changed convention is raised "
        "immediately by `check_spec`. `evaluate.py` uses only "
        "`check_spec`/`recover_from_disk`/`load`/`drift`/`resolve`/`save`/`SlicesMissing`"
        ", all in the batch-construction stage, none of them landing "
        "on judging or a denominator.",
}

#: Anchor version. Entries in `PENDING_ANCHOR` are only valid for this version;
#: as soon as the anchor advances, this table must be emptied.
PENDING_ANCHOR_REVISION = 91

#: Files added to `JUDGING` before the anchor was updated (allowed while re-anchoring
#: is not); emptied once the anchor moves past `PENDING_ANCHOR_REVISION`.
PENDING_ANCHOR: dict[str, str] = {}

#: Names each `JUDGING_IMPORT_ALLOW` module lets cross into the judging segment
#: (`from .M import ...` binding names); must match exactly.
ALLOW_SYMBOL_PIN: dict[str, frozenset[str]] = {
    "haenv_kernel/build.py": frozenset({"build_instance", "leakage_probe"}),
    "haenv/semantic_parallel.py": frozenset({"MAX_JUDGE_LANES", "run_cells"}),
    "haenv/run_scheduler.py": frozenset({"_schedule", "_stall_guard", "stall_report_lines"}),
    "haenv/semantic_budget.py": frozenset({"BudgetExceeded", "BudgetLedger", "cost_summary"}),
    "haenv/semantic_transport.py": frozenset({"PriceSchedule", "PricedJudge"}),
    "haenv/paid_completion.py": frozenset({"note_generation"}),
    "haenv/solver_accounting.py": frozenset({"prepare_accounting", "fetch_endpoints"}),
    "haenv/relay_accounting.py": frozenset({
        "RELAY_COST_DEFAULT", "RELAY_COST_HEADROOM", "RELAY_COST_PER_GRID", "RELAY_USD_PER_MTOK",
        "_measured_tokens_per_grid", "preflight_quota"}),
    "haenv/transport.py": frozenset({"verify_wire", "check_pin", "check_sampling", "check_served",
                                     "judge_spec", "sent_temperature"}),
    "haenv/payloads.py": frozenset({"load_payloads"}),
    "haenv/store.py": frozenset({"load_cases"}),
    "haenv/semantic_runref.py": frozenset({"correction_base", "subset_base", "view_run"}),
    "haenv/streams.py": frozenset({"device_signals"}),
    "haenv/anchor.py": frozenset({
        "_jvint", "_wvint", "infra_files", "judging_fingerprint", "judging_fp_cached",
        "judging_vintages", "world_fingerprint"}),
    "haenv/semantic_source.py": frozenset({"semantic_bytes"}),
    "haenv/batch.py": frozenset({"provenance_fields", "record_usage", "kernel_fingerprint"}),
    "haenv/build_clinical.py": frozenset({"CLINICAL_ATTEN", "_clinical_baselines", "_clinical_cv"}),
    "haenv/config.py": frozenset({"ROOT", "_lc_p", "load_cfg"}),
    "haenv/external_gold.py": frozenset({"_ext_probes"}),
    "haenv/framings.py": frozenset({
        "DDX_PROMPT", "DDX_SCOPE2_PROMPT", "DDX_SCOPE_PROMPT", "DDX_TRACE_PROMPT",
        "_BUILTIN_FRAMING_NAMES", "_SCOPE2_EDITS", "_SCOPE_EDITS", "_framings",
        "derive_scope2_prompt", "derive_scope_prompt", "framing_sha256"}),
    "haenv/events.py": frozenset({"_inh"}),
    "haenv/events_pools.py": frozenset({
        "EVENT_RATE_DEFAULTS", "event_weeks", "expected_event_counts"}),
    "haenv/events_streams.py": frozenset({"METRIC_BY_NAME"}),
    "haenv/events_text.py": frozenset({"alias_hit"}),
    "haenv/findings_render.py": frozenset({"_TS_TO_PANEL", "_band", "_value"}),
    "haenv/indicators.py": frozenset({"of"}),
    "haenv/regpath.py": frozenset({"load_registry"}),
    "haenv/llm.py": frozenset({"make_dispatcher"}),
    "haenv/call_log.py": frozenset({"_EXHAUSTED_MARKERS", "_is_exhausted", "_llm", "billed_tokens"}),
    "haenv/ops/runtime.py": frozenset({"for_ledger"}),
    "haenv/alias_excludes.py": frozenset({"alias_excluded_at"}),
    "haenv/overlay.py": frozenset({
        "RIVALS", "_cr_pub", "_fac", "_kernel_specs", "condition_registry",
        "haenv_comorbid_specs", "haenv_independent_specs", "rivals_for",
        "spec_id_of", "specialty_hit", "threads_for"}),
    "haenv/prompts.py": frozenset({"PROMPT"}),
    "haenv/trace.py": frozenset({
        "TraceLog", "_TraceLog", "_ci", "_rt", "_save_trace", "save_trace"}),
    "haenv/provenance_report.py": frozenset({"_prov_1", "_prov_render"}),
    "haenv/registry.py": frozenset({
        "GOLD_EVIDENCE", "_cff", "_lf", "_load_test_vocab",
        "condition_findings_for_case", "load_findings",
        "_load_yaml", "load_disputed_gold", "load_test_recall_vocab"}),
    "haenv/rng.py": frozenset({"rng"}),
    "haenv/slicing.py": frozenset({"_slicing"}),
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


def _sha(p: Path) -> str:
    """Raw-byte fingerprint, for "this file is byte-for-byte what was used" (packs, job yaml)."""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _sem(p: Path) -> str:
    """Semantic fingerprint (`haenv.anchor.semantic_bytes`): comments and docstrings don't count."""
    # Imported here: `anchor.judging_files()` exec's this file, so a top-level import cycles.
    from haenv.anchor import semantic_bytes
    return hashlib.sha256(semantic_bytes(p)).hexdigest()[:16]


def check_segments_disjoint(generation=GENERATION, judging=JUDGING, infra=INFRA) -> None:
    """Hard check before any anchor is written: INFRA ∩ (GENERATION ∪ JUDGING) = ∅."""
    overlap = sorted(set(infra) & (set(generation) | set(judging)))
    if overlap:
        raise SystemExit(f"🔴 INFRA overlaps a frozen segment: {overlap}")


def _equivalence_rows(prev: dict | None = None) -> dict:
    """`JUDGING_SHA_EQUIVALENCE`, each row recomputed from git objects before it is written.

    Neither the rows nor the INFRA membership they rely on are certified by the revision
    that introduces them: a row takes effect only once a previous anchor recorded it (as
    effective or pending), and the files it drops must be INFRA in that previous anchor
    too. A row the previous anchor does not know is verified and written as pending.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from haenv.anchor import verify_judging_equivalence, verify_judging_move
    prev = prev or {}
    known = {(r["old"], r["rev"], r["new"])
             for key in ("judging_sha_equivalence", "judging_sha_equivalence_pending")
             for r in prev.get(key) or ()}
    trusted_infra = tuple(f for f in INFRA if f in set(prev.get("infra") or ()))
    out = {"effective": [], "pending": []}
    for row in JUDGING_SHA_EQUIVALENCE:
        if (row["old"], row["rev"], row["new"]) in known:
            out["effective"].append(verify_judging_equivalence(row, infra=trusted_infra))
        else:
            out["pending"].append(verify_judging_equivalence(row))
    for row in JUDGING_SHA_MOVES:
        if (row["old"], row["rev"], row["new"]) in known:
            out["effective"].append({**verify_judging_move(row, infra=trusted_infra), "kind": "move"})
        else:
            out["pending"].append({**verify_judging_move(row), "kind": "move"})
    return out


def _equivalence_fields(prev: dict) -> dict:
    rows = _equivalence_rows(prev)
    return {"judging_sha_equivalence": rows["effective"],
            "judging_sha_equivalence_pending": rows["pending"]}


def _previous_anchor() -> dict:
    try:
        from haenv.anchor import freeze_path
        return json.loads(freeze_path().read_text(encoding="utf-8"))
    except Exception:                                           # noqa: BLE001 -- no anchor yet
        return {}


def _write_doc_numbers() -> None:
    """Carry the new anchor into the living documents (`doc_numbers_check --write --run`).

    A fresh process, so nothing cached from the previous anchor is read. The anchor is
    already written when this runs; anything still red is printed for a hand fix.
    """
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "doc_numbers_check.py"),
                        "--write", "--run"], cwd=ROOT, capture_output=True, text=True,
                       timeout=1800)
    keep = ("  wrote", "  🔴")
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith(keep) or "rewritten" in line or line.startswith("共 "):
            print("   " + line.strip())
    if r.returncode:
        print(f"   doc_numbers_check exited {r.returncode} after --write: fix the lines above by hand")


def run_publish_gate(packs: dict, pack_types: dict, assert_publishable, not_publishable, root: Path | None = None) -> int:
    """Run the publish gate over every pack's saved rows. 0 = passed, 2 = rejected or not runnable.

    The pack's directory comes from its declared task type (`pack_types`), never from its name. A
    pack whose batch directory is not there is an error; a batch with no `eval.jsonl` (generated,
    not yet run) has nothing to gate and says so on the console.
    """
    root = ROOT if root is None else root
    for _job, _v in packs.items():
        _dir = root / "results" / pack_types[_job] / _job / _v["batch"]
        if not _dir.is_dir():
            print(f"🔴 Publish gate cannot run: batch directory {_dir} does not exist")
            return 2
        _ev = _dir / "eval.jsonl"
        if not _ev.exists():
            print(f"⚠️ Publish gate not run for {_job}@{_v['batch']}: no eval.jsonl in the batch")
            continue
        _rows = [json.loads(x) for x in _ev.read_text(encoding="utf-8").split("\n") if x.strip()]
        if not _rows:
            print(f"⚠️ Publish gate not run for {_job}@{_v['batch']}: eval.jsonl has no rows")
            continue
        try:
            assert_publishable(_rows, where=f"make_freeze · {_job}@{_v['batch']}")
        except not_publishable as _e:
            print(f"🔴 Freeze rejected: {_job}@{_v['batch']}\n{_e}")
            print("   To freeze an old pack anyway, pass `--allow-stale` explicitly (it gets written into the snapshot's allow_stale field)")
            return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", type=int, required=True)
    ap.add_argument("--date", default="2026-09-02")
    ap.add_argument("--why", default="", help="Reason this run supersedes the previous version")
    ap.add_argument("--supersedes", type=int, default=None)
    ap.add_argument("--pack", action="append", default=[],
                    metavar="JOB=BATCH", help="repeatable; JOB=batch timestamp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-stale", action="store_true",
                    help="Skip the publish gate (freeze the batch anyway even with mixed/stale/unstamped judge versions); recorded in the snapshot")
    ap.add_argument("--segments-only", action="store_true",
                    help="Recompute only the two segments' source fingerprints; packs are carried over unchanged from the previous version -- "
                         "doesn't need `results/`, so external contributors can run it too. Must be combined with --supersedes and --why")
    args = ap.parse_args()
    check_segments_disjoint()

    if args.segments_only:
        if args.pack:
            print("🔴 `--segments-only` does not accept `--pack` -- packs must either all be inherited or all freshly given")
            return 2
        if args.supersedes is None or not str(args.why).strip():
            print("🔴 `--segments-only` must be combined with `--supersedes <N>` and `--why '...'` -- "
                  "if the segments moved but the packs weren't recomputed, that fact must be recorded")
            return 2
        from haenv.anchor import AnchorError as _AE, freeze_path as _fp
        try:
            _prev = json.loads(_fp().read_text(encoding="utf-8"))
        except (_AE, OSError, ValueError) as _e:
            print(f"🔴 Could not read the previous anchor, so there are no packs to inherit: {type(_e).__name__}: {_e}")
            return 2
        _inherited = _prev.get("packs") or {}
        if not _inherited:
            print("🔴 The previous anchor has zero packs -- inheriting would still produce an empty snapshot, refusing")
            return 2
        _snap = {
            "frozen_at": args.date,
            "revision": args.revision,
            "generated_by": "tools/make_freeze.py --segments-only (generated, do not edit)",
            "generation": {f: _sem(ROOT / f) for f in GENERATION},
            "judging": {f: _sem(ROOT / f) for f in JUDGING},
            "raw_sha": {f: _sha(ROOT / f) for f in (*GENERATION, *JUDGING)
                        if (ROOT / f).is_file()},
            "infra": {f: _sem(ROOT / f) for f in INFRA},
            **_equivalence_fields(_prev),
            "packs": _inherited,
            "n_total": _prev.get("n_total", sum(v.get("n", 0) for v in _inherited.values())),
            "segments_only": True,
            "packs_inherited_from": _prev.get("revision"),
            "segments_before": {"judging_sha16": _prev.get("judging_sha16"),
                                "world_sha": _prev.get("world_sha")},
            "supersedes": {"revision": args.supersedes, "why": args.why},
        }
        if _prev.get("allow_stale"):
            _snap["allow_stale"] = True         # inherited packs were never verified; the flag carries over
        _dst = Path(args.out) if args.out else _fp()
        _dst.write_text(json.dumps(_snap, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"rev{args.revision} · segment fingerprints recomputed only · {len(_inherited)} packs "
              f"{_snap['n_total']} cases inherited from rev{_prev.get('revision')} → {_dst}")
        print("   ⚠️ This version's packs were not recomputed under the current code -- the snapshot's `segments_only: true` records this")
        if not args.out:
            _write_doc_numbers()
        return 0

    packs: dict[str, dict] = {}
    pack_types: dict[str, str] = {}
    missing = []
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from haenv.jobmeta import pack_task_type
    for spec in args.pack:
        job, _, batch = spec.partition("=")
        try:
            pack_types[job] = pack_task_type(ROOT, job, batch)
        except ValueError as _e:
            print(f"🔴 {_e}")
            return 2
        f = ROOT / "results" / pack_types[job] / job / batch / "cases.jsonl"
        if not f.exists():
            missing.append(str(f))
            continue
        _jy = ROOT / "inputs" / f"{job}.job.yaml"
        packs[job] = {"batch": batch,
                      "n": sum(1 for _ in f.open(encoding="utf-8", errors="ignore")),
                      "sha256": _sha(f),
                      "job_yaml_sha256": _sha(_jy) if _jy.exists() else None,
                      "generation_source": _generation_source(f.parent)}
    if missing:
        for m in missing:
            print(f"🔴 Batch does not exist: {m}")
        return 2

    if not packs:
        print("🔴 No `--pack` given at all -- freezing an empty snapshot would overwrite the authoritative anchor, and the publish gate would iterate zero times")
        print("   Usage: --pack JOB=BATCH (repeatable); see `haenv.anchor.frozen_packs()` for the current packs")
        return 2

    snap = {
        "frozen_at": args.date,
        "revision": args.revision,
        "generated_by": "tools/make_freeze.py (generated, do not edit)",
        "generation": {f: _sem(ROOT / f) for f in GENERATION},
        "judging": {f: _sem(ROOT / f) for f in JUDGING},
        "raw_sha": {f: _sha(ROOT / f) for f in (*GENERATION, *JUDGING)
                    if (ROOT / f).is_file()},
        "infra": {f: _sem(ROOT / f) for f in INFRA},
        **_equivalence_fields(_previous_anchor()),
        "packs": packs,
        "n_total": sum(v["n"] for v in packs.values()),
    }
    if args.supersedes is not None:
        snap["supersedes"] = {"revision": args.supersedes, "why": args.why}

    if not args.allow_stale:
        # `haenv.report` imports the kernel (`from build import ...`), so the kernel path goes first.
        import sys as _sys

        import yaml as _yaml
        _cfg_mf = _yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        _kp = str((ROOT / _cfg_mf["kernel_path"]).resolve())
        if _kp not in _sys.path:
            _sys.path.insert(0, _kp)
        if str(ROOT) not in _sys.path:
            _sys.path.insert(0, str(ROOT))
        from haenv.report import NotPublishable, assert_publishable
        _rc = run_publish_gate(packs, pack_types, assert_publishable, NotPublishable)
        if _rc:
            return _rc
    else:
        snap["allow_stale"] = True      # record the override: this snapshot was never verified

    from haenv.anchor import AnchorError, freeze_path
    if args.out:
        dst = Path(args.out)
    else:
        try:
            dst = freeze_path()
        except AnchorError:
            dst = ROOT / "docs" / "anchor" / f"freeze-{args.date}.json"   # first creation
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rev{args.revision} · {len(packs)} packs · {snap['n_total']} cases → {dst}")
    for job, v in packs.items():
        print(f"   {job:<20}{v['batch']}  {v['n']:>3} cases  {v['sha256']}")
    if not args.out:
        _write_doc_numbers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
