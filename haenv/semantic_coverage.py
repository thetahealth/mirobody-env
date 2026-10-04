"""The semantic coverage of a ranked record: a missing primary measurement voids the composite.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from collections import Counter


#: Semantic dimensions a composite needs. The composite's diagnosis dimension is the
#: code-side dx_listed, so a missing semantic dx_hit does not void it; dx_hit is an
#: auxiliary reading.
PRIMARY = ("noop_ok", "tests_recall", "tests_precision")


def protect_composite(rec: dict, rows: list[dict]) -> None:
    """A missing measurement is neither an incorrect answer nor a removable dimension."""
    semantic = [r["semantic"] for r in rows if isinstance(r.get("semantic"), dict)]
    if not semantic:
        return
    missing = sum(any(s.get("metric_states", {}).get(name) not in
                      ("resolved", "not_applicable") for name in PRIMARY) for s in semantic)
    missing += len(rows) - len(semantic)
    held = {name: why for s in semantic for name, why in ((s.get("held_out") or {}).get("why") or {}).items()}
    if held:
        rec["held_out_dims"] = held          # reported separately; no slot in the composite
    rec["semantic_coverage"] = {"cells": len(rows), "primary_missing_cells": missing,
                                "states": dict(Counter(s["status"] for s in semantic)),
                                "clinical_review": "not_clinician_reviewed", "final": False}
    if missing:
        rec["score"] = None
        rec["score_full"] = None
        rec["score_none_reason"] = (f"LLM primary judgments missing or unresolved on {missing} cells; "
                                    "no proxy substitution, zero fill or reduced denominator")
