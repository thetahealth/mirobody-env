# Measurement-validity resources

This directory retains the original blinded tasks, hidden truth tables, effective-tier
declarations and consensus labels as a set. The [evidence guide](../../docs/anchor/VALIDITY.md)
records what the historical anchors establish and what they do not.

| Family | Files that must remain together |
|---|---|
| Demo answers | `tasks.jsonl`; [protocol](../../docs/anchor/demo-protocol.md) and [readout](../../docs/anchor/validity-readout.md) |
| Test lists | `tasks-tests.jsonl`, `tasks-tests-neg.jsonl`, `truth-tests-neg.json`; [protocol](../../docs/anchor/README-tests.md) and [negative-case readout](../../docs/anchor/validity-readout-tests-neg.md) |
| Discriminating tests | `tasks-disc-neg.jsonl`, `truth-disc-neg.json`, `tasks-disc-neg-effective-set.json` |
| Revision | Each `tasks-rev-neg*.jsonl` with its matching `truth-rev-neg*.json` and `tasks-rev-neg*-effective-set.json` |
| Revision consensus | `labels-rev-neg3-consensus.json`, the third revision family, its [protocol](../../docs/anchor/README-rev3.md) and [readout](../../docs/anchor/readout-rev3-consensus.md) |

Truth identifies mutations and judge outputs; do not expose it before labels are persisted.
Consensus from model annotators remains a proxy for a human anchor. Non-clinician labels
measure agreement with the mechanical judge, not clinical correctness.

Relocation preserves all data bytes, task IDs, source labels and historical fingerprints.
The legacy scoring registry's recorded validity is not evidence that today's four-pack
management rubric has been clinically validated. Current scoring is defined in the
[product specification](../../docs/spec/spec.md#9-overall-score-and-reporting).
