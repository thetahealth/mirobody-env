# Shared reference measurements

[`lifesnaps-reference.json`](lifesnaps-reference.json) holds the aggregated wearable reference
used by the realism tool and regression checks. Its `source` and measurement-level fields
identify the original measurement convention. Daily differences and daily levels, full-wear
and partial-wear populations, and within-person and across-person correlations must not be
interchanged.

This is shared reference data, not a test-only fixture or a new clinical threshold. The
relocation preserves its original bytes and source metadata; it adds no participant records.
The [data card](../../docs/DATA_CARD.md) describes the synthetic benchmark's limits.
