# Saved management-plan rubric

`registry/management_rubric.yaml` defines the provisional, haenv-authored v6.2
measurement policy. Clinical review remains pending. The scorer reads the saved
`adjudication.open_management` snapshot and the saved answer; it does not generate
new clinical facts or replace the original code scores.

The shared criteria cover decisive history, executable management, safety,
followup and patient explanation. History, management, followup and explanation
have the registered equal item weights. A wrong core decision caps the score at
0.3; a satisfied negative safety criterion applies the registered zero multiplier.
Applicability and clinical content come from the saved snapshot and sourced
adapter rules. A missing source or an unscored answer remains explicit.

`tools/evaluation/management_rejudge.py` prepares saved cells and writes independent rubric
outputs. `tools/evaluation/management_board_prepare.py` and
`tools/evaluation/management_board_rejudge.py` support saved board inputs without changing
the source batches. These commands can dispatch paid judges only when explicitly
requested; their offline modes use deterministic stubs. Equivalent routes and
snapshot verification are described in [Judge routes](judge-routes.md).

Criterion fingerprints and their aggregate identify the scoring source. Execution
routes and provenance decorators remain outside scoring identities. Replaying
archived results preserves their score values and original source identities;
new documentation metadata is not a reason to restamp an archive.
