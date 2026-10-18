"""Pure symptom onset timing shared by M2 generation and matched confirmation."""
from __future__ import annotations


def spread_symptom_days(case_id: str, n: int, T: int) -> list[int]:
    """Onset days of a legacy hidden line's `n` symptoms, in template order: the renewal layout the record's
    benign self-reports use (`events._renewal_days`, log-normal gaps), sorted; the first layout (attempts 0, 1,
    ...) whose days sit inside [7, T - 3], are at least five days apart and are not an exact arithmetic
    progression (the GEN18 gate `symptom_days_separable`). The template days (60, 143, 224, 309 on T = 336,
    scaled by T / T0 otherwise) sat on the same relative positions in every case of a spec and put 27 % of the
    true symptoms in the last tenth of the history against 11 % of the benign ones; one timing law for both
    removes the position cue. A history too short for any layout falls back to days with slowly growing gaps."""
    from haenv.world.observations.events import _renewal_days
    if n <= 0:
        return []
    hi = max(7, T - 3)
    keys = [f"m2-sym{i}" for i in range(n)]
    for attempt in range(60):
        d = sorted(_renewal_days(keys, case_id, T, attempt))
        gaps = [b - a for a, b in zip(d, d[1:])]
        if d[0] >= 7 and d[-1] <= hi and all(g >= 5 for g in gaps) and (n < 3 or len(set(gaps)) > 1):
            return d
    g = max(5, (hi - 7) // (n + 1))
    return [min(hi - 5 * (n - 1 - i), 7 + g * (i + 1) + (i * (i + 1)) // 2) for i in range(n)]     # growing gaps: no progression
