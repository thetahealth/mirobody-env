"""Reject cross-analyte aliases before a merged vocabulary reaches generation."""
from __future__ import annotations

from haenv.resources.regpath import load_registry


def validate_aliases(merged, upstream=None):
    upstream = load_registry("findings_upstream.yaml") if upstream is None else upstream
    identities = upstream.get("alias_analytes") or {}
    for fid, addition in (upstream.get("enrich") or {}).items():
        if fid not in merged:
            continue
        for alias in addition.get("aliases") or []:
            if fid not in identities.get(str(alias), ()):
                raise ValueError(f"{fid}: merged alias {alias!r} does not name the same registered analyte")
    # Check the actual merge as well: an alias registered to one analyte must
    # never appear on a different analyte, even through an `add` name collision.
    for fid, finding in merged.items():
        for alias in finding.get("aliases") or []:
            owner = identities.get(str(alias))
            if owner is not None and fid not in owner:
                raise ValueError(f"{fid}: merged alias {alias!r} belongs to {owner}")
    for name, addition in (upstream.get("add") or {}).items():
        canonical = addition.get("canonical_finding")
        names = ([merged[canonical].get("name_cn"), *merged[canonical].get("aliases", ())]
                 if canonical in merged else [])
        if canonical and (canonical not in merged or name in merged or name not in names):
            raise ValueError(f"{name}: shadow entry must merge into canonical analyte {canonical}")
    return True
