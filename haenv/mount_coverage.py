"""Coverage of the mount table against the live judge registry and the row builders that run it.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from .mount_table import GEOMETRIES, MOUNT, NONE, known_gaps, reason_for, subject_of


def holes() -> list[tuple[str, str]]:
    """Blank cells with no reason: the real holes. The maintainers' tests keep this
    empty."""
    from .judges import JUDGES
    out = []
    for j in JUDGES:
        for g in GEOMETRIES:
            if subject_of(j.name, g) == NONE and reason_for(j.name, g) is None:
                out.append((j.name, g))
    return out


def unregistered_mounts() -> list[str]:
    """Judges mounted in the table but not registered in `JUDGES`."""
    from .judges import JUDGES
    known = {j.name for j in JUDGES}
    return sorted(n for n in MOUNT if n not in known)


def unwired_geometries() -> list[str]:
    """Geometries with mounted judges whose row builder declares no dispatcher profile
    (`mounting.BY_GEOMETRY`), i.e. mounted but never run."""
    from .mounting import BY_GEOMETRY
    out = []
    for g in GEOMETRIES:
        if not any(subject_of(n, g) != NONE for n in MOUNT):
            continue                      # not a single cell mounted on this geometry -> no "mounted but never runs" to speak of
        m = BY_GEOMETRY.get(g)
        if m is None or not str(getattr(m, "profile", "") or ""):
            out.append(g)
    return out


def coverage_by_geometry() -> dict:
    """Coverage per geometry, so an empty column is not hidden in the total."""
    from .judges import JUDGES
    known = {j.name for j in JUDGES}
    out = {}
    for g in GEOMETRIES:
        mounted = [j.name for j in JUDGES if subject_of(j.name, g) != NONE]
        pre = [n for n in sorted(MOUNT)
               if n not in known and subject_of(n, g) != NONE]
        out[g] = {"n_judges": len(JUDGES), "mounted": len(mounted),
                  "mounted_names": mounted,
                  "premounted_unregistered": pre}
    return out


def coverage() -> dict:
    """Cross-product coverage, for reports."""
    from .judges import JUDGES
    tot = len(JUDGES) * len(GEOMETRIES)
    mounted = sum(1 for j in JUDGES for g in GEOMETRIES
                  if subject_of(j.name, g) != NONE)
    gaps = len(known_gaps())
    return {"n_judges": len(JUDGES), "n_geometries": len(GEOMETRIES), "n_cells": tot,
            "mounted": mounted, "declared_absent": tot - mounted,
            "known_gaps": gaps, "holes": len(holes()),
            "mounted_frac": round(mounted / tot, 3),
            # Pre-mounted: a cell exists but the judge is not registered.
            "unregistered": unregistered_mounts(),
            "unwired": unwired_geometries(),
            "by_geometry": coverage_by_geometry()}
