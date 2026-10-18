# Anchors and their scope

Machine-readable resources live with their consumers' resource families:

| Purpose | Current location | Interpretation |
|---|---|---|
| Freeze and source provenance | [`registry/anchors/`](../../registry/anchors/) | Read the generated declaration; its filename is an identifier, not the current revision. |
| Measurement-validity tasks, truth, effective tiers and labels | [`registry/validity/`](../../registry/validity/) | Historical evidence for the declared scoring convention. |
| Shared wearable reference | [`registry/references/`](../../registry/references/) | Aggregated reference measurements, with the source and measurement convention retained. |
| Offline extraction example | [`examples/extract-replay/`](../../examples/extract-replay/) | Synthetic clinician-gold placeholder and its replay; no clinical reading is established. |

[Validity evidence](VALIDITY.md) explains the historical measurement results and their limits.
Read the [original demo protocol](demo-protocol.md), [test-list protocol](README-tests.md), or
[layered revision protocol](README-rev3.md) before interpreting the corresponding tasks.
The [revision consensus readout](readout-rev3-consensus.md) is a model proxy, not human validation.

These resources keep their original IDs, fingerprints, dates and source labels. Relocation
does not turn a historical evaluation into a run under today's code or scoring formula.
Current product behavior and overall scoring are defined in the
[specification](../spec/spec.md#9-overall-score-and-reporting).

The prepared clinician blind kit is maintained separately from public runtime resources.
Preparation and simulation do not establish clinician sign-off.

SYNTHETIC data, for evaluation only, not medical advice.
