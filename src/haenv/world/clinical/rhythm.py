"""The real-visit-rhythm slice schedule's constants and its feasibility check, shared by the
row builders, the emission gate (GEN27) and the job generator.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations



#: Slices per case, fixed (the order of magnitude of `auto+neutral`) so per-case cost and
#: judge counts are comparable.
REAL_RHYTHM_SLICES = 7


#: Length (days) of the forced long interval in the "information gap" tier. It is an explicit
#: tier because window truncation makes random sampling undersample long intervals.
REAL_RHYTHM_GAP_DAYS = 200


def rhythm_gap_feasible(T: int, n_slices: int | None = None) -> bool:
    """Whether this horizon can fit the forced long-gap tier; shared by case generation and
    run time.
    """
    n = int(n_slices if n_slices is not None else REAL_RHYTHM_SLICES)
    return int(T) >= int(REAL_RHYTHM_GAP_DAYS) + n
