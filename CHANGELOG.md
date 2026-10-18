# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The package version (`pyproject.toml`) versions the code. Scores are versioned separately:
per-criterion fingerprints identify which result columns remain comparable, and an aggregate
fingerprint identifies the complete set of criteria. Combined scores also require matching
aggregation rules and task weights (see `LICENSE-DATA`, ATTRIBUTION).

## [1.3.0] - 2026-10-18

The first release since 1.2.2. The 1.3.0 board uses four tasks with 100 questions per task and
three answers per question; eight models have complete four-task results. Judging and world
fingerprints change, so 1.3.0 boards are not comparable with 1.2.x boards. This release also
contains the changes of an unreleased candidate prepared on 2026-10-06 (listed below); that
candidate was never published and is superseded. Code comments that mention "world 1.4" refer to the fourth revision of the
simulated world, which ships in this release; they are left unchanged because the world and judging
fingerprints are computed over those files.

### Changed

- The main score averages `S = G * D * (1 + Q) / 2` over the three answers to each
  question, then averages the four task scores equally. The original first-answer
  aggregation is retained as a supplementary result. Paired question-cluster bootstrap
  intervals separate six of seven adjacent model pairs on the main board.
- The lab-change task now requires observable non-noise changes to exceed the paired
  reference change value. Its repaired question set replaces the earlier lab set in the
  N100 board. A reader using only the printed values and the reference-change table
  reaches about 0.5 decision accuracy; this baseline is reported with the results.
- The README, bilingual interactive demo and figures use the same final N100 readings.
  The evaluated question sets ship as frozen pack files; regenerating at each batch's recorded
  generation commit reproduces them byte for byte. The bundled job files use current defaults.
- Full-record prompts include the protocol library. The stored N50 access comparison
  uses the earlier lab set and is documented separately from the final N100 board.
- GPT-6 Sol's diagnosis answers complete its four-task results; it ranks first, and the other
  rows are unchanged. The diagnosis task's core decision is graded by code (alias matching of the
  listed diagnoses, no model call); the LLM judge scores only management-plan quality.

### Added

- `haenv twin`: one-fact twin records of a built batch (operations `drop-reagent-note`,
  `drop-planted-symptoms`, `unname-explainer`), built through the normal build path and checked against the
  original field by field. Fingerprints are unchanged.
- Management-plan rubric scoring, repeat judgments, per-criterion provenance, and
  snapshot-checked judge routes across configured endpoints.
- Centralized deployment concurrency settings, effective-limit diagnostics, graceful
  stopping, provider-scoped budget accounting, and recovery of interrupted cells.
- A cross-family judge check: all 6,937 responses with G * D = 1 re-judged by a Gemini judge
  (requested gemini-3.7-flash, served as gemini-3.8-flash) keep the model order;
  gpt-6-sol leads gemini-3.8-flash by +0.031 [+0.010, +0.052], and model-mean Q
  correlates with the primary judge at r = .968.

### Fixed

- Shared price-catalog caching and retries avoid redundant startup requests.
- Infrastructure failures receive bounded resends under fresh request identities;
  successful answers are replayed and previous charges are retained.
- Model-level and backend-default proxy selection share the same transport path.

### Included from the unreleased 2026-10-06 candidate

The candidate moved the judging fingerprint from `950578a687aacc5e` to `a44341f58a8b78f2` and the
world fingerprint from `633cb0576e0472f0` to `0283ea9ca28d58ae`; the two diagnosis tracks of the
1.x board (`ddx-timeline`, `ddx-workup`) were unchanged. Its board ranked ten models on the four packs, 100 items each, every item answered three times.
The overall score is the equal-weight mean of the chance-corrected pack scores: gpt-6-sol 0.712 leads and
is separable from every other model; gemini-3.1-pro 0.648, gemini-3.7-flash 0.640 and gpt-6-luna 0.609
share ranks 2-4 and are not separable from one another. Ranks were reported as bootstrap intervals. That board is kept in the
[data card](docs/DATA_CARD.md#historical-boards).

#### Changed

- Packs are nested: for one seed the 50-item pack is a subset of the 100-item pack with
  byte-identical cases, so a 50-item run is extended to 100 by running only the new items. Every
  generator emits at least four times N candidate items, the audit's shortcut gates read the whole
  pool, and the published pack is picked from it. The four shipped job files are regenerated with
  the public seed at N = 50 and carry their pool.
- Pack ① scores which key signals a model bought and reports review utility as its own column.
  Pack ② draws symptom sentences from shared pools, and a marked symptom sentence escalates the
  gold disposition one step in every band. Pack ③ follows the cited levothyroxine and CKD rules,
  apportions classes by how much each one separates models, and prints the clinic's policy on the
  item. Pack ④ reads the next step against a set of acceptable answers and drops the kernel gates
  that read action fields; its composite is the class score.
- World: a pack patient's age band follows the prevalence of the patient's conditions; women who
  can become pregnant are not given ACE inhibitors, ARBs or statins unless indicated; a visit's
  acute vital signs enter the same-day wearable readings; a pack declares the payload fields its
  frame reads, and the case file and the prompt carry that view; laboratory variation is split by
  analytical method, and reference intervals depend on sex and age.
- Scoring: a hard-gate penalty is the gate rate times the magnitude of the chance-corrected score,
  so a failed gate never raises a below-chance composite and the composite is not floored (packs
  ① and ③). A medication-change marker counts unless its clause negates it or its sentence makes
  it conditional. A gated tool call that names no usable target is a no-op observation instead of
  an abort of the whole batch.
- Audit: `SA-0` checks that the pool holds at least four times the published items; `SA-10` reports
  a character n-gram text classifier per text source (report only); the prior stub blocks at the
  shortcut line like the other evidence-free stubs; the batch-level shortcut check (A5) refuses
  only above the shared 0.99 quantile of label shuffles.
- The paid-request ledger is a SQLite database. A request reserves a quantile of its model's
  observed cost instead of its worst case, and requests queue fairly for lanes; the interpreter
  switch interval is short while a ledger transaction runs, so concurrent writers do not convoy.
  A job that only adds cases resumes its batch in paid mode. Request pools are per backend, with
  per-model caps (`eval.model_caps`) and a measured timing table. A ledger file that a compaction
  replaced is read again from the start. Scores do not change.
- The demo page shows the board with rank intervals and the sample items of the four packs.
- Native Gemini and DashScope tariffs were re-read from the providers' pricing pages; prices, tiers
  and cache rates are unchanged.

#### Added

- `tools/evaluation/board_readout.py` and `tools/evaluation/rank_rule.py`: the per-pack boards (chance-corrected,
  case × round bootstrap), the overall board under equal and SD-equalized weights, pairwise
  separability and a rank-stability table from stored batches, with no model call. A pack on which
  every model scores the same gets weight 0 under SD-equalized weights. See
  [`docs/REPRODUCE.md`](docs/REPRODUCE.md#reading-the-board).
- `haenv probe concurrency` measures how many requests each model's route accepts and writes caps
  ready to paste; `haenv loadtest` rehearses a paid run offline against a fake provider and reads
  a speed report ([`docs/design/loadtest.md`](docs/design/loadtest.md),
  [`docs/design/concurrency-caps.md`](docs/design/concurrency-caps.md),
  [`docs/design/paid-ledger.md`](docs/design/paid-ledger.md)).
- `tools/evaluation/recompute_judges.py` attaches the judge plugins a batch's job declares.
- `tools/evaluation/board_readout.py` reads a pack run in several batches and writes every model pair's
  separability; `web/demo/export_board.py` builds the demo's `board.json` from its readout. The candidate's
  batches, answers and seed were packaged as release archives but not published.

#### Fixed

- Paid runs on `google` / `dashscope` stopped a week after the built-in tariff check; they now warn once per backend. `paid.tariff_checked_on` records your own check, `paid.tariff_expiry: refuse` (official boards) keeps the refusal.

## [1.2.2] - 2026-10-09

Documentation-only patch release. Runtime code, dependencies, generation and scoring rules,
frozen packs and leaderboard data are unchanged from 1.2.1.

### Documentation

- The English and Chinese README explain how new synthesis tasks combine patient timelines, noise
  and event-density controls with shared checks and task-specific verification.
- A bilingual rare-disease case study shows reused components, examples of verification failures
  and the explicit batch-gate exception; it is linked from the README and the documentation index.
- The rare-disease implementation remains on `rare-code-bench` and is not bundled in this package;
  the workflow uses the existing `haenv-synth` and `haenv-extend` skills.

## [1.2.1] - 2026-10-05

Boards of 1.2.0 and 1.2.1 are comparable: the 1.2.0 board, re-scored from its stored responses
with this release, gives the same scores, intervals and pair results. The judging fingerprint
moves (`a5f058bdd9de77b9` -> `950578a687aacc5e`) because the files it covers changed in how a run
resumes, how a paid request settles and how the shortcut check is computed; the world
fingerprint is unchanged.

### Changed

- A paid OpenRouter request records its generation id from the response headers as soon as they
  arrive. A request cut off before its answer is saved settles at the provider's recorded cost,
  on resume and through `haenv spend reconcile`; only a request with no id stays at its reserved
  bound. `haenv spend reconcile` also reads in-flight markers and failed judge receipts. The cost
  line names tariff-priced charges (Google, DashScope) as estimates.
- The generation-time audit reads the observed labels only. Its gates compare a chance-corrected
  score with the 0.20 line and the 0.99 quantile of label shuffles, and draw those shuffles only
  when the score reaches the line. The permutation p readings are a profile: not computed by
  default (`null` in the audit record, marked "not computed"), computed with `--profile` on
  `tools/validation/pack_audit.py` and `tools/generation/make_pack.py`.
- A resumed run resends only the aborts that originate on our side: harness errors, and
  `ABORT(no_response)` / `ABORT(unparseable)` whose final failed attempt is a network, timeout,
  5xx, 429 or key-pool failure. An empty body, broken JSON, a reply cut at `max_tokens` and an
  exhausted tool budget count as the model's result. A cause that cannot be told from the row
  stands as the model's result and is listed in the resume log.
- Deterministic builds write `cases.jsonl` and `payloads.jsonl` on the generation processes, and
  the shortcut check computes its permutation p-values in parallel; files, digests and every
  p-value are byte-identical to a serial run.
- The demo page: the patient builder applies the production emission gate and lists the reasons
  when it refuses a patient, with the nearest accepted setting; the dose record is drawn; act 2
  shows the record streams under each chart; act 3 has the overall board on its own row, a pair
  matrix of all models, and pack ① in the model panel.

## [1.2.0] - 2026-10-04

The benchmark is four task packs, each asking one clinical decision at `T`. Boards produced under
1.x are not comparable with boards of 1.2.0: the world, the questions and the scoring changed. The
bridge items of pack ① are the only link between the two.

### Added

- Four task packs: ① differential diagnosis (a second condition behind findings the first one
  explains, and clinician review), ② acute triage (four dispositions), ③ chronic medication
  adjustment (five decisions) and ④ follow-up interpretation (true change, noise, pre-analytical
  artefact or method difference, and the size of the true change). Packs ②–④ are scored by code
  against the world's truth; pack ① uses one LLM judge for free-text diagnoses and tests.
- `tools/generation/make_pack.py`: one command from pack, size and seed to a built and audited pack, with no
  model call. Jobs carry `pack: {n_items, seed_sha256}`; batches record the seed's hash, never the
  seed. The shipped jobs are the public sample packs; official boards use a private seed.
- `tools/validation/pack_audit.py`: the generation-time audit every pack shares; `haenv run` refuses a pack's
  items without a passing audit.
- Chance-corrected decision scores (`cc = (m · BA − 1) / (m − 1)`): a random or constant answer
  scores 0.
- Complaints in the patient's own words, the medication of each known condition in the record,
  and known comorbidities that change the gold answer in packs ③ and ④.

### Changed

- Registered measures have three roles: scored, profile (computed and shown, not in the
  composite) and retired (found to judge wrongly). `dx_hit` and `tool_target_grounded_rate` are
  retired.
- The over-triage and missed-clinician-review gates are reported outside the hard-gate multiplier;
  clinician review is scored once, as chance-corrected `review_utility_cc`.
- The simulation kernel is the package `haenv_kernel`; each run carries its own `RunContext`.
- Default models are reached directly: Gemini through Google, Qwen through DashScope, the others
  through OpenRouter pinned to one upstream with fallbacks off (the vendor's endpoint where
  zero data retention allows it, else a named host serving the same weights).
- Deterministic generation runs on a process pool; the output is byte-identical to a serial run
  (`--gen-workers 1`).
- The four pack judge groups register through entry points, so an installed wheel builds and
  scores the packs.

## [1.1.1] - 2026-10-01

The judging fingerprint (`67dd5866790e9cb9`), the world fingerprint (`25f31527b09e6abe`), the
frozen question packs and the board are those of 1.1.0. The solving code changed: a billed batch
started under 1.1.0 does not resume under 1.1.1; finish it under 1.1.0, or run it again with
`--fresh`.

### Fixed

- On Python 3.10 the fingerprints of unchanged code differed from those computed on 3.11 and
  later: under 1.1.0 the judging fingerprint read `275d059bec243917` instead of `67dd5866790e9cb9`
  and the world fingerprint `6966e2d16b58a0ca` instead of `25f31527b09e6abe`. Rows scored on 3.10
  could not share a board with rows scored on 3.11 or later, and a billed batch could not resume
  across the two. The fingerprints hash source normalised by `ast.unparse`; Python 3.11 changed
  its output, and 3.10's Unicode 13.0 database escapes the characters Unicode 14.0 added. On 3.10
  the normalisation now follows 3.11's rules, and Python 3.10 to 3.14 give this release the same
  judging, world and solving fingerprints.
- The agent skills' frontmatter is valid YAML holding only the fields the Agent Skills standard
  defines. Two descriptions contained `: `, which strict YAML parsers reject, and `argument-hint`
  is not a field of the standard.
- README: the "Evaluate your own agent" example runs with `--limit 1`, which skips semantic
  judging, needs no OpenRouter key and exits 6 to mark the scores incomplete. The README says so;
  the OpenRouter key and the judge's cost belong to a run without `--limit`.
- On a release, the package description's `tree/` links name the release tag, as its `blob/` links
  and figure addresses already did.

### Changed

- CI installs and runs the wheel built from the sdist, the same way the release builds it.

## [1.1.0] - 2026-09-30

The judging fingerprint (`67dd5866790e9cb9`), the world fingerprint (`25f31527b09e6abe`), the
frozen question packs and the board are those of 1.0.1. The solving code changed: a billed batch
started under 1.0.1 does not resume under 1.1.0; finish it under 1.0.1, or run it again with
`--fresh`.

### Added

- A model on a backend of your own can take part in a billed run. Its configuration entry
  declares `price` (`input_per_million_usd`, `output_per_million_usd`; 0 for a free endpoint);
  each request is reserved at that price and counted from the token counts the endpoint returns
  in `usage`, against the same `--judge-budget-usd` as the semantic judge. The ledger books these
  costs as tariff bounds (a declared price is not an invoice). A batch keeps the price it was
  created with: a changed `price` does not resume it. On `openrouter`, `relay`, `google` and
  `dashscope` the price still comes from the provider, and a declared `price` there is refused.
- Three agent skills ship under `skills/`, in the Agent Skills layout (`skills/<name>/SKILL.md`):
  `haenv-synth` (generate patients and questions), `haenv-bench` (run a pack and read the board),
  `haenv-extend` (add a judge, a stream, an event or a task type from outside the repository). The
  README says what they are for; the repository keeps untracked symlinks where one vendor's tool
  looks for them.
- The README states what a *task* is here: a gold-standard shape plus an answer contract, with the
  format (`single` / `gated` / `slices` / `multi`) a condition inside it, and one pipeline generating,
  gating and scoring all of them. A case is assigned to a task by the gold it carries, not by the
  job file's label.

### Changed

- `haenv run` with a real model checks its keys before it opens a batch. A model or the semantic
  judge without a key is named with the variable it needs and the state of the key file, and the
  run exits 2 with nothing sent. A run with `--limit` is not judged and needs no judge key.
- A refusal from the billed-run accounting (an unverified route, an unreachable price endpoint,
  a changed run identity) is one printed line and exit code 2 instead of a traceback. The Google
  capacity check says whether the endpoint was not reached or answered with an HTTP error, and
  gives up after 45 s as a whole: its 30 s timeout applied to each connection attempt, and with the
  route to the endpoint dropped it waited for more than six minutes.
- On a release, the package description names the release tag instead of `main` in its figure
  and link addresses, so each version's PyPI page keeps that version's files.
- The README (English and Chinese) returns to the 1.0.0 layout, and its one animation is the
  terminal recording of the quick start. The demo animations, the first-screen buttons and the
  comparison matrix are no longer in it. The corrections made for 1.0.1 remain: nine hard gates
  enter the multiplier and `acted_on_unverified_signal` is reported only; the semantic dimensions
  are judged by one LLM judge, whose OpenRouter key the env file needs; the token scale is per
  answered cell; figures and documents are linked by absolute URL.

### Fixed

- README: the "Evaluate your own agent" example stopped with "Cannot budget unverified solver
  route(s)": no price source existed for a backend of one's own. The example now declares `price`.
- README: the "Install from PyPI" commands run again in the same directory (`build` opens a new
  batch with `--fresh`).
- README: after `pip install`, your own backends and models go in a file named by
  `HAENV_CONFIG_OVERLAY`, since the packaged `config.yaml` is inside the installed package.
- README: regenerating a diagnosis pack offline (`--gen deterministic`) emits 144 of its 145
  specifications; `JD-32v2` fails its anchor check.
- CONTRIBUTING: commit types are `feat`, `fix`, `refactor`, `docs`, `test` and `chore`; CI,
  packaging and release changes are `chore`. It listed `ci` and `build` as types of their own.
  A release reaches `main` as a short series of commits, one per change, ending in the
  commit that sets the version.

## [1.0.1] - 2026-09-30

The judging fingerprint moves from `25ced8ada8427f45` to `67dd5866790e9cb9`; the world fingerprint
(`25f31527b09e6abe`) and the frozen question packs are those of 1.0.0. No score changes: the six
release batches were re-scored under the new code from their stored answers, all 3,417 rows carry
the scored values they carried in 1.0.0, and the board is identical.

The judging segment moved with how a judging run executes and how its identity is compared, not
with what it scores: the number of judge lanes is no longer part of a run's judging identity, a run
resumes across comment-only and infra-only changes, and judge lane scheduling and result writing
belong to the infra segment.

### Changed

- Configuration has a typed schema with one declared default per key and a role on every field, so
  a misspelled key inside a section is an error. `--set KEY=VALUE` overrides one value, and each run
  appends the resolved configuration and the source of each layer to `<batch>/config_runs.jsonl`.
- `pydantic>=2.6` is a runtime dependency, required by the typed schema.
- A judge run resumes across comment-only and infra-only changes.
- A judge run given no lane count uses 10 lanes; the policy files no longer carry one (they
  carried 64 or 4).
- The budget ledger (`budget.json`) is append-only with periodic compaction: after its first line
  it holds one change per line, so read it with `haenv.semantic.budget.read_state`, not
  `json.load`.
- `eval.model_wall_budget_s` (off by default) bounds a model's total wall clock: once it is spent,
  no new cell of that model is started, and the cells left unstarted are recorded in the batch's
  `scheduler.json` (`wall_budget_stops`).
- New connections are paced, and a connection that fails before any request bytes are sent is
  retried instead of being counted as a model failure.
- Demo: a patient timeline, an answer matrix, a step-by-step replay and result charts; the board
  is ranked in tiers with 95% intervals from a case-level bootstrap, alongside the noise floor and
  the separable model pairs; the English page is fully in English.
- README (English and Chinese) is organised around four demo animations, presents the leaderboard
  in tiers, and links figures and documents by absolute URL so that it renders on the package
  index.
- The source distribution no longer carries the README figures.
- `LICENSE` holds the MIT text alone, so that it is detected as MIT; the data licence and the
  kernel's own licence are stated in `LICENSE-DATA` and `core/LICENSE`.
- `CONTRIBUTING.md` states the commit and release-branch rules.

### Fixed

- `docs/DATA_CARD.md` describes the hard gates as graded: nine of the ten gates are
  non-compensatory and one is reported outside the multiplier; in `slices` the action-level gates
  also zero the slice where they fire. The scoring registry's entries are listed by where their
  readings are reported.
- The demo's count of generated patient records covers generated batches only. A batch whose case
  bodies were copied from another one (a re-score or an arm split) no longer adds its source's
  records again, so the count is 25,400 rather than 30,619; the measured dirty-data rates are
  unchanged.

## [1.0.0] - 2026-09-30

First public release.

- `haenv` pipeline: synthetic patient generation, per-item verification, offline and model runs,
  scoring and reports (`haenv build | verify | run | report`).
- `verifier_core`: domain-independent verifier primitives, usable without the rest of the package.
- Task specifications (`inputs/`), registries (`registry/`), question framings (`probes/`) and the
  solver-visible half of the frozen reference packs (`frozen/*.Q.jsonl`).
- Self-checks with negative controls: `tools/testing/verify_selftest.py`, `tools/testing/leak_probe_selftest.py`.
- Two frozen question packs, `ddx-timeline` and `ddx-workup`, on the same 145 cases (64 conditions),
  and a ten-model board with tiers from a case-level bootstrap (see `docs/DATA_CARD.md`).
- Semantic judging for `noop_ok`, `tests_recall` and `tests_precision` with one judge model and
  stored votes.
