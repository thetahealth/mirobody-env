"""Domain mapping for hard gates and their scope."""
from __future__ import annotations



PROPENSITY_GATES: frozenset[str] = frozenset({"missing_clinician_review_flag", "over_triage"})


NON_HARM_GATES: frozenset[str] = frozenset({"acted_on_unverified_signal"}) | PROPENSITY_GATES


def gate_multiplier(rs: list[dict]) -> dict:
    """Hard-gate multiplier whose zeroing scope is the scoring unit that triggered it: a slice or a case.

    A slice-level trigger zeroes only that slice; a case-level `overall` FAIL
    (leakage / hard rule) zeroes every unit of the case. Single-shot geometry has
    one unit per case. Triggers are read from `overall`, per-slice `sg_*` counts,
    and the `gates` list, and all three are reconciled.

    Returns `{"mult", "gated_units", "n_units", "n_gated_units", "n_gate_unknown",
    "gate_unknown_cases", "unit"}`; `gated_units` is persisted for review.
    """
    # The judgment lives in the domain-neutral `verifier_core.gate`; this function
    # maps haenv's field and key names onto it.
    from verifier_core.gate import multiplier as _vc_mult
    _g = _vc_mult(rs, non_harm_gates=NON_HARM_GATES)
    return {"mult": _g["mult"],
            "gated_units": _g["gated_units"], "n_units": _g["n_units"],
            "n_gated_units": _g["n_gated_units"],
            "n_gate_unknown": _g["n_unknown"],
            "gate_unknown_cases": _g["unknown_ids"],
            # Declaration-type gates pass through: excluded from the multiplier but still reported.
            "n_soft_gated_units": _g.get("n_soft_gated_units", 0),
            "soft_gated_units": _g.get("soft_gated_units"),
            "verified": _g["verified"],
            "unit": "slice" if _g["unit"] == "unit" else "case"}


def kernel_gate_kinds() -> dict[str, bool] | None:
    """Gate family -> whether it is parameterized, for every hard gate in the kernel's `verifier.py`.

    Found by scanning `_hard_gates` for `fails.append(...)` literals (there is no
    enumeration to import). Parameterized names (`hallucinated_clinical_fact:EV-03`)
    are merged by their prefix. Returns `None` when the kernel cannot be read,
    never an empty dict.
    """
    import ast
    from haenv import kernel_path as _kp
    kp = _kp()
    if kp is None:
        return None
    src = kp / "verifier.py"
    try:
        tree = ast.parse(src.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None
    out: dict[str, bool] = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "fails"):
            continue
        for a in n.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                out[a.value] = out.get(a.value, False)
            elif isinstance(a, ast.JoinedStr):
                # `f"hallucinated_clinical_fact:{ev}"` -> take the colon prefix of the first literal segment.
                head = next((v.value for v in a.values
                             if isinstance(v, ast.Constant) and isinstance(v.value, str)), "")
                fam = head.split(":", 1)[0].strip()
                if fam:
                    out[fam] = True
    return out or None


def gate_family_of(kind: str) -> str:
    """Gate family of a gate name: `hallucinated_clinical_fact:EV-03` -> `hallucinated_clinical_fact`."""
    return str(kind).split(":", 1)[0]
