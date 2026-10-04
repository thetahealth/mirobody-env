"""Shared helpers for the judge family: reading gold, rivals and aliases, and ranking.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from ..wq import _ddx  # noqa: F401

log = logging.getLogger("haenv.judges")


log = logging.getLogger("haenv.judges")

SINGLE, MULTI, SLICES = "single", "multi", "slices"


# ================================================================ kinds of gold
#
# Judges read gold only through `wq.gold_of`, so gold use is counted in
# `wq.gold_consumption_coverage`; a direct `ddx.get(...)` is invisible to coverage.
def _gold(vp, field: str, default=None):
    """The only channel through which judges read gold. `default` suppresses the raise for a
    missing primitive.
    """
    from .. import wq
    return wq.gold_of(vp, field, default=default)


def gold_kind(vp) -> str:
    """This case's gold kind (judges mount on it); delegates to `gold_kinds.derive`.

    Never falls back to `unified`: an underivable case gets a sentinel no ddx judge mounts on
    (the last rule in `gold_kinds._CORE`).
    """
    from .. import gold_kinds
    return gold_kinds.derive(vp)


def _rivals_of(vp) -> tuple:
    from ..overlay import rivals_for
    return rivals_for(_gold(vp, "spec_id"), _ddx(vp))


# ================================================================ ddx helpers
def _rank_of(out, names) -> dict:
    """Hit determination via `tracks.dx_rank_of`: returns `dx_hit` (considered), `dx_mentioned`
    (brought up) and `dx_ruled_out_gold` (explicitly excluded); the fallback path has no rank.
    """
    from ..tracks import dx_rank_of
    return dx_rank_of(out, names)


def _names(vp) -> list[str]:
    """Gold diagnosis and aliases, plus the judging-side synonyms (`tracks.judging_names`)."""
    from ..tracks import judging_names
    return judging_names([str(_gold(vp, "diagnosis") or "")]
                         + [str(a) for a in (_gold(vp, "aliases") or [])])


