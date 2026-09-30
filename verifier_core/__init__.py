"""verifier_core -- domain-independent primitives for scoring a benchmark.

Standalone: it does not import `haenv` and carries no domain vocabulary. It
does not decide whether an answer is correct; it decides whether a batch of
scores is still usable.

| module | the question it answers |
|---|---|
| `vintage` | Which version of the scorer produced these rows? Mixed versions? Stale? |
| `fingerprint` | Which parts of the scoring code changed, and therefore which stored results a change invalidates? |
| `ceilings` | What is the upper bound a degenerate strategy can reach on this dimension? |
| `audit` | Noise floor, degenerate ceiling, pairwise readability, shortcut scan |
| `gate` | Non-compensatory hard gates: a hit fails the unit outright |

Design rules:

1. Refuse, do not annotate: a gate raises rather than writing a field.
2. "Unstamped" and "stamped with something else" are different states.
3. When in doubt, recompute.
4. Absent is not zero: "not applicable", "failed" and "never ran" stay
   distinguishable all the way to the report.

Evaluation infrastructure for SYNTHETIC data; it makes no domain judgements.
"""
from __future__ import annotations

from . import audit
from .ceilings import (
    degenerate,
    resolve_noise,
)
from .gate import (
    multiplier,
)
from .fingerprint import (
    combined,
    part_shas,
    parts_changed,
)
from .vintage import (
    NotPublishable,
    UNSTAMPED,
    assert_publishable,
    provenance,
    vintages,
)

__all__ = [
    "NotPublishable", "UNSTAMPED",
    "assert_publishable", "provenance", "vintages",
    "combined", "part_shas", "parts_changed",
    "multiplier",
    "degenerate", "resolve_noise",
    "audit",
]
__version__ = "0.1.0"
