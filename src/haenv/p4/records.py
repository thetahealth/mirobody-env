"""Reconstruct the comparison from visible dated reports, never from gold."""
from __future__ import annotations


def clinical_view(block):
    if not block or "previous_draw_day" in block:
        return block
    today = int(block["today"])
    reports = [r for r in block.get("report") or []
               if r.get("analyte", block["analyte"]) == block["analyte"]]
    previous = [r for r in reports if int(r["day"]) < today]
    current = [r for r in reports if int(r["day"]) == today]
    if not previous or len(current) != 1:
        raise ValueError("dated laboratory reports do not identify a comparison")
    selected = max(int(r["day"]) for r in previous)
    if sum(int(r["day"]) == selected for r in previous) != 1:
        raise ValueError("previous laboratory report is ambiguous")
    # This view is internal to the existing clinical reader and invariants.
    # The rule itself is a patient-independent, retrievable catalog document.
    from haenv.p4.protocols import document
    clinical = all("analyte" in r for r in reports) or any(
        r.get("comparison_reference_method") for r in block.get("laboratory_method_record") or [])
    return {**block, "report": reports, "previous_draw_day": selected,
            **({"interpretation_rule": document()["steps"][1]} if clinical else {})}
