"""Fingerprints of scoring code: one over all parts (coarse) and one per part (fine).

Callers consult the fine fingerprint only when the coarse one fails, so the worst case is
the coarse behaviour. Parts are `Mapping[part_name, bytes]`; how they are cut is the
caller's decision.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

#: Fingerprints keep the first 16 hex digits.
WIDTH = 16


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:WIDTH]


def combined(parts: Mapping[str, bytes]) -> str:
    """Whole-set fingerprint. Part names are hashed too, and parts are sorted by name.

    Not named `fingerprint`, which would collide with the module name.
    """
    h = hashlib.sha256()
    for name in sorted(parts):
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(parts[name])
        h.update(b"\x00")
    return h.hexdigest()[:WIDTH]


def part_shas(parts: Mapping[str, bytes]) -> dict[str, str]:
    """Per-part fingerprints, `{part_name: sha16}`."""
    return {name: _h(blob) for name, blob in parts.items()}


def parts_changed(before: Mapping[str, str], after: Mapping[str, str],
                  used: Iterable[str]) -> list[str]:
    """Which of `used` changed between `before` and `after`.

    A part missing from `before` counts as changed (fail closed).
    """
    return sorted(n for n in set(used)
                  if n not in before or before[n] != after.get(n))
