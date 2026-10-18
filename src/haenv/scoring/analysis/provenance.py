"""Stored judging identity comparisons and publication checks."""
from __future__ import annotations



class NotPublishable(RuntimeError):
    """Failing verdict from the publish gate (`assert_publishable`)."""


def judging_provenance(rows: list[dict]) -> dict:
    """The judging version this batch's readings were measured under, and whether it is stale.

    Returns `{"measured_under", "judging_current", "stale", "why"}`; every
    aggregated reading carries it.
    """
    from haenv.provenance.anchor import judging_fingerprint
    from haenv.provenance.anchor import judging_vintages
    vint = judging_vintages(rows)
    cur = judging_fingerprint()
    unstamped = vint.get("<unstamped>", 0)
    others = {k: v for k, v in vint.items() if k != "<unstamped>"}
    why = []
    if unstamped:
        why.append(f"{unstamped} row(s) carry no judging stamp; "
                   f"cannot tell whether they were computed by the same judging version")
    off = {k: v for k, v in others.items() if k != cur}
    if off:
        why.append(f"{sum(off.values())} row(s)' judging version {sorted(off)} is not the current code {cur}")
    if len(others) > 1:
        why.append(f"this batch has {len(others)} judging versions; a board may only have one")
    return {"measured_under": vint, "judging_current": cur,
            "stale": bool(why), "why": why or None}


def assert_publishable(rows: list[dict], where: str = "") -> None:
    """Publish gate: raise `NotPublishable` for a reading with a mixed or stale judging version.

    Unversioned batches are rejected at the publish layer but still render. Called
    only on paths that produce external material, such as `make_freeze` (which may
    pass `--allow-stale`).
    """
    # The judgment lives in `verifier_core.vintage`; this is the single entry point
    # that maps haenv's names onto it.
    from verifier_core.vintage import NotPublishable as _VCNotPublishable
    from verifier_core.vintage import assert_publishable as _vc_assert
    prov = judging_provenance(rows)
    try:
        _vc_assert(rows, prov["judging_current"], where=where)
    except _VCNotPublishable as e:
        raise NotPublishable(str(e)) from e
