"""Reconstruct the comparison from visible dated reports, never from gold."""
from __future__ import annotations


def comparison_reports(block):
    """Select the named pair, with date inference for older saved questions."""
    current_day = int(block.get("comparison_draw_day", block["today"]))
    rows = [r for r in block.get("report") or []
            if r.get("analyte", block["analyte"]) == block["analyte"]]
    previous = [r for r in rows if int(r["day"]) < current_day]
    current = [r for r in rows if int(r["day"]) == current_day]
    if not previous or len(current) != 1:
        raise ValueError("incomplete or ambiguous dated laboratory comparison reports")
    named = block.get("comparison_previous_draw_day", block.get("previous_draw_day"))
    previous_day = int(named) if named is not None else max(int(r["day"]) for r in previous)
    selected = [r for r in previous if int(r["day"]) == previous_day]
    if len(selected) != 1:
        raise ValueError("named previous laboratory report is missing or ambiguous")
    return selected[0], current[0]


def clinical_view(block):
    if block and "comparison_draw_day" in block:
        block = {**block, "today": int(block["comparison_draw_day"])}
    if not block or "previous_draw_day" in block:
        return block
    reports = [r for r in block.get("report") or []
               if r.get("analyte", block["analyte"]) == block["analyte"]]
    previous, _ = comparison_reports(block)
    # This view is internal to the existing clinical reader and invariants.
    # The rule itself is a patient-independent, retrievable catalog document.
    from haenv.p4.protocols import document
    clinical = all("analyte" in r for r in reports) or any(
        r.get("comparison_reference_method") for r in block.get("laboratory_method_record") or [])
    return {**block, "report": reports, "previous_draw_day": int(previous["day"]),
            **({"interpretation_rule": document()["steps"][1]} if clinical else {})}
