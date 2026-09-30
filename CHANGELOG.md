# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The package version (`pyproject.toml`) versions the code. Scores are versioned separately: every
score row carries the judging fingerprint of the code that produced it, and boards produced under
different fingerprints are not comparable (see `LICENSE-DATA`, ATTRIBUTION).

## [1.0.0]

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
