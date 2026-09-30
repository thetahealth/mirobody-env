# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The package version (`pyproject.toml`) versions the code. Scores are versioned separately: every
score row carries the judging fingerprint of the code that produced it, and boards produced under
different fingerprints are not comparable (see `LICENSE-DATA`, ATTRIBUTION).

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
  it holds one change per line, so read it with `haenv.semantic_budget.read_state`, not
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
- Self-checks with negative controls: `tools/verify_selftest.py`, `tools/leak_probe_selftest.py`.
- Two frozen question packs, `ddx-timeline` and `ddx-workup`, on the same 145 cases (64 conditions),
  and a ten-model board with tiers from a case-level bootstrap (see `docs/DATA_CARD.md`).
- Semantic judging for `noop_ok`, `tests_recall` and `tests_precision` with one judge model and
  stored votes.
