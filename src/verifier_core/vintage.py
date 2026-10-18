"""Scorer vintage: which version of the scorer produced these rows.

A batch is rejected for publishing when it mixes scorer versions, was scored by a version
other than the current code, or carries no version stamp. Unstamped rows are still
allowed at render time; only paths that produce external artifacts refuse them.
"""
from __future__ import annotations

from typing import Iterable, Mapping

#: The bucket for unstamped rows, kept apart from real versions: "unknown" is not "old".
UNSTAMPED = "<unstamped>"


class NotPublishable(RuntimeError):
    """The publish gate failed. Raised, not returned, so it cannot be ignored."""


def vintages(rows: Iterable[Mapping], key: str = "judging_sha16") -> dict[str, int]:
    """Scorer versions present in a batch, mapped to their row counts.

    Missing or empty values go into `UNSTAMPED` as their own bucket.
    """
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key) or UNSTAMPED)
        out[k] = out.get(k, 0) + 1
    return out


def provenance(rows: Iterable[Mapping], current: str,
               key: str = "judging_sha16") -> dict:
    """Under what conditions this batch was measured.

    Returns `{"measured_under", "current", "stale", "why"}`.
    `stale=True` means at least one row was not scored by the current code.
    It is a marker for the reader; the refusal is `assert_publishable`'s job.
    """
    rows = list(rows)
    vint = vintages(rows, key)
    unstamped = vint.get(UNSTAMPED, 0)
    others = {k: v for k, v in vint.items() if k != UNSTAMPED}
    why: list[str] = []
    if unstamped:
        why.append(f"{unstamped} rows carry no judging stamp -- unknown which version judged them")
    if len(others) > 1:
        why.append(f"this batch has {len(others)} judging versions {sorted(others)}")
    off = {k: v for k, v in others.items() if k != current}
    if off:
        why.append(f"{sum(off.values())} rows at version {sorted(off)} != current {current}")
    return {"measured_under": vint, "current": current,
            "stale": bool(unstamped or off or len(others) > 1),
            "why": why}


def assert_publishable(rows: Iterable[Mapping], current: str,
                       where: str = "", key: str = "judging_sha16") -> None:
    """Publish gate: raise unless every row was scored by the current code.

    Call it only on paths that produce external artifacts (release, freeze, leaderboard).
    Zero rows also fail, and so do rows whose judge crashed (`judge_errors`).
    """
    rows = list(rows)
    if not rows:
        raise NotPublishable(
            f"Publish gate failed{(' · ' + where) if where else ''}: zero rows "
            f"(an empty batch is not a pass).")
    prov = provenance(rows, current, key)
    vint = prov["measured_under"]
    others = {k: v for k, v in vint.items() if k != UNSTAMPED}
    unstamped = vint.get(UNSTAMPED, 0)
    bad: list[str] = []
    if len(others) > 1:
        bad.append(f"mixed version: this batch has {len(others)} judging versions {sorted(others)} "
                   f"sharing one denominator")
    off = {k: v for k, v in others.items() if k != current}
    if off:
        bad.append(f"stale: {sum(off.values())} rows at judging version {sorted(off)} "
                   f"!= current {current}")
    if unstamped:
        bad.append(f"unstamped: {unstamped} rows do not record which judging version scored them")
    # A crashed judge leaves the row looking like "not applicable"; such rows are not publishable.
    _errs = [r for r in rows if r.get("judge_errors")]
    if _errs:
        _names = sorted({str(n) for r in _errs for n in (r.get("judge_errors") or ())})
        bad.append(f"judge threw an error: {len(_errs)} rows had a judge crash ({_names[:4]}); "
                   f"a crash is a failed measurement, not \"not applicable\"")
    if bad:
        raise NotPublishable(
            f"Publish gate failed{(' · ' + where) if where else ''}:\n  · "
            + "\n  · ".join(bad)
            + "\nRemediation: recompute the batch with the current judges from the stored raw responses, "
              "or exclude its readings from external material.")
