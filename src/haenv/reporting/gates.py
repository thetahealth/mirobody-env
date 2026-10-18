"""Domain mapping for hard gates and their scope."""
from __future__ import annotations

from haenv.scoring.analysis.gates import gate_family_of
from haenv.scoring.analysis.gates import kernel_gate_kinds


_ACTION_GATE_NAMES = ("premature_closure", "missing_clinician_review_flag",
                      "treatment_before_exclusion", "invasive_before_firstline")


def _action_gate_surface(rows: list[dict]) -> list[str]:
    """Hits of the four per-slice safety gates, printed with the number of judgeable
    and skipped slices (zero hits is only meaningful against its denominator).
    """
    _njudged = sum(int(r.get("slice_gate_n_judged") or 0) for r in rows)
    _unknown = sum(int(r.get("slice_gate_unknown") or 0) for r in rows)
    _any = any(r.get(f"sg_{g}") is not None or r.get("slice_gate_n_judged") is not None
               for g in _ACTION_GATE_NAMES for r in rows[:1] or [{}])
    if not _njudged and not _unknown:
        return []
    L = ["### 1z. Exercised surface of the four safety gates (per-slice · read together with the denominator)\n",
         f"Judgeable slices **{_njudged}** · skipped for missing `clinician_review_required` **{_unknown}** slices"
         f"(absence is not folded into False)\n",
         "| Gate | Hit cells | Hit slice-count | Hit by |", "|---|---|---|---|"]
    for g in _ACTION_GATE_NAMES:
        k = f"sg_{g}"
        hitrows = [r for r in rows if r.get(k)]
        tot = sum(int(r[k]) for r in hitrows)
        by: dict[str, int] = {}
        for r in hitrows:
            by[str(r.get("solver"))] = by.get(str(r.get("solver")), 0) + 1
        _who = " · ".join(f"`{m}`×{c}" for m, c in sorted(by.items(), key=lambda x: (-x[1], x[0]))[:5])
        L.append(f"| `{g}` | {len(hitrows)} | {tot} | {_who or '—'} |")
    L.append("")
    L.append("> **How to read this**: if the only hits come from offline stubs, this gate **has a positive control but no real model made this mistake** -- "
             "its correctness rests on the gate's positive and negative controls, not on the zero hit count.")
    L.append("")
    return L


def _case_gate_surface(rows: list[dict], built: dict | None = None) -> list[str]:
    """Hits of the case-level hard gates (`verifier.grade`'s `gates` list), one row
    per gate family including zero-hit families, each with its own denominator:
    gates with a precondition (e.g. `red_flag_present`) are counted against the
    cases that satisfy it; `not measured` when the case bodies are unavailable.
    """
    graded = [r for r in rows if "gates" in r]
    if not graded:
        return []
    ungraded = [r for r in rows if "gates" not in r
                and r.get("slice_gate_n_judged") is None]
    # Rows cover the kernel's full gate set plus observed families, merged to the
    # family (the part before `:`), so zero-hit gates still appear.
    _kern = kernel_gate_kinds()
    _obs_fams: dict[str, list[str]] = {}
    for r in graded:
        for g in (r.get("gates") or []):
            _obs_fams.setdefault(gate_family_of(str(g)), []).append(str(g))
    kinds = sorted(set(_obs_fams) | set(_kern or {}))
    # Precondition denominators are (cases satisfying it) x (solvers), read from the
    # case objects in `built`.
    _rf_cases: set[str] = set()
    for _cid, _raw in (built or {}).items():
        _adj = getattr(_raw, "adjudication", None)
        if _adj is None and isinstance(_raw, dict):
            _adj = _raw.get("adjudication")
        if (_adj or {}).get("red_flag_present"):
            _rf_cases.add(str(_cid))
    _n_solvers = len({str(r.get("solver")) for r in graded}) or 1
    # Kernel preconditions (`haenv_kernel/verifier.py`): `red_flag_present`, its complement,
    # `clinician_action_warranted`, `artifact_flags.is_artifact_window`.
    _warr_cases: set[str] = set()
    _artifact_cases: set[str] = set()
    for _cid, _raw in (built or {}).items():
        _adj = getattr(_raw, "adjudication", None)
        if _adj is None and isinstance(_raw, dict):
            _adj = _raw.get("adjudication")
        _adj = _adj or {}
        if _adj.get("clinician_action_warranted"):
            _warr_cases.add(str(_cid))
        if (_adj.get("artifact_flags") or {}).get("is_artifact_window"):
            _artifact_cases.add(str(_cid))
    _n_cases = len(built or {})

    def _d(cases: set[str]):
        # `None` when `built` is missing (not measured); a real 0 means the gate cannot
        # fire on this batch.
        if not built:
            return None
        return len(cases) * _n_solvers

    _DENOM_NOTE = {
        # ---- precondition `red_flag_present` ----
        "treatment_before_exclusion":
            (_d(_rf_cases), "kernel criterion has a `red_flag_present` precondition ⇒ can only hit on red-flag cases"),
        "missed_emergency_red_flag":
            (_d(_rf_cases), "kernel criterion has a `red_flag_present` precondition ⇒ can only hit on red-flag cases"),
        # ---- precondition `not red_flag_present` (the complement) ----
        "over_triage":
            (_d({str(c) for c in (built or {})} - _rf_cases),
             "kernel criterion has a **`not red_flag_present`** precondition ⇒ can only hit on **non**-red-flag cases (the complement)"),
        # ---- precondition `clinician_action_warranted` ----
        "premature_closure":
            (_d(_warr_cases), "kernel criterion has a `clinician_action_warranted` precondition ⇒ can only hit on cases warranting intervention"),
        "missing_clinician_review_flag":
            (_d(_warr_cases), "kernel criterion has a `clinician_action_warranted` precondition ⇒ can only hit on cases warranting intervention"),
        # ---- precondition `artifact_flags.is_artifact_window` ----
        "acted_on_unverified_signal":
            (_d(_artifact_cases), "kernel criterion has an `artifact_flags.is_artifact_window` precondition "
                                  "⇒ can only hit on cases with a false-alarm artifact window"),
    }
    # Every kernel gate with a precondition must be registered above; the rest use
    # `len(graded)`.
    L = ["### 1y. Exercised surface of the case-level hard gates (each gate read together with **its own** denominator)\n",
         f"Rows that went through `verifier.grade` **{len(graded)}** · "
         f"rows **with no trace of any gate having judged them** **{len(ungraded)}**"
         f"(the latter are recorded as `gate_unknown`, not folded into \"clean\" -- `zero hits != safe`, and also != the gate being correct)\n"]
    if _kern is None:
        L.append("> **The kernel's full gate set is not measured**: `verifier.py` could not be read "
                 "(`haenv.kernel_path()` failed to resolve or failed to parse) ⇒ the table below only contains gates "
                 "**observed in this batch**, and zero-hit gates are invisible on it. "
                 "This line marks that gap; it does not mean there are no zero-hit gates.\n")
    else:
        _zero = [g for g in kinds if not _obs_fams.get(g)]
        L.append(f"> Gate families are taken from the **kernel's full set** ({len(_kern)} `fails.append` gate families in `verifier.py`) "
                 f"∪ the {len(_obs_fams)} families observed in this batch ⇒ **{len(kinds)}** rows total, "
                 f"of which **{len(_zero)} families have zero hits**: {'`' + '` `'.join(_zero) + '`' if _zero else 'none'}. "
                 f"A zero-hit gate **still gets printed as a row**, read together with its denominator -- "
                 f"\"the gate was never triggered\" and \"the gate doesn't exist\" render identically in the artifact, and they are not the same thing at all.\n")
    L.append("| Gate family | Hit rows | Denominator | Specific values | Hit by |")
    L.append("|---|---|---|---|---|")
    for g in kinds:
        hit = [r for r in graded if any(gate_family_of(str(x)) == g
                                        for x in (r.get("gates") or []))]
        by: dict[str, int] = {}
        for r in hit:
            by[str(r.get("solver"))] = by.get(str(r.get("solver")), 0) + 1
        who = " · ".join(f"`{m}`×{c}" for m, c in sorted(by.items(), key=lambda x: (-x[1], x[0]))[:5])
        _vals = sorted({v for v in _obs_fams.get(g, []) if v != g})
        _vtxt = ("—" if not _vals
                 else f"{len(_vals)} values: `" + "` `".join(_vals[:6])
                      + ("` …" if len(_vals) > 6 else "`"))
        _dv, _why = _DENOM_NOTE.get(g, (len(graded), ""))
        if _dv is None:
            _dtxt = "not measured<br><sub>this render has no case body (`built`) ⇒ the precondition can't be evaluated</sub>"
        elif _dv == 0 and hit:
            # Denominator 0 with hits means the precondition reading has diverged from the
            # kernel; printed as a contradiction.
            _dtxt = (f"🔴 **0, yet {len(hit)} hits**<br><sub>"
                     + (_why + ";" if _why else "")
                     + "the denominator says this gate cannot structurally hit, yet it did ⇒ **the path reading the precondition diverges from the kernel criterion**"
                       "(`built`'s shape / a field name / the kernel precondition changed). **This row's denominator is unusable.**</sub>")
        elif _dv == 0:
            _dtxt = ("0<br><sub>" + (_why + ";" if _why else "")
                     + "**not one case in this batch satisfies the precondition** ⇒ this gate cannot structurally hit</sub>")
        else:
            _dtxt = f"{_dv}" + (f"<br><sub>{_why}</sub>" if _why else "")
        # A family absent from the kernel set came from elsewhere (e.g. a plugin).
        _mark = "" if (_kern is None or g in _kern) else " outside kernel roster"
        L.append(f"| `{g}`{_mark} | {len(hit)} | {_dtxt} | {_vtxt} | {who or '—'} |")
    L.append("")
    L.append("> **How to read this**: two gates with different denominators **cannot be compared side by side on hit rate**. "
             "A gate with a precondition (e.g. `treatment_before_exclusion`) can only hit on cases satisfying that precondition -- "
             "using batch size as the denominator would read a perfect positive control as \"probably broken\".")
    L.append("> **How to read a zero hit count**: check the denominator first. Denominator `0` ⇒ not one case in this batch "
             "satisfies the precondition, and this gate **measures nothing at all** in this batch (not \"the model is very safe\"); "
             "a large denominator with 0 hits ⇒ *that* is \"had the chance to fail, and didn't\". "
             "A denominator printed as `not measured` ⇒ the precondition can't be evaluated -- **do not read it as 0**.")
    L.append("> A gate family marked `outside kernel roster` appears in the data but not in the kernel `verifier.py`'s "
             "`fails.append` calls -- either an external plugin criterion emitted it, or the kernel changed how it emits and this table's "
             "scanner hasn't caught up. **Both cases must be investigated**; do not treat it as normal.")
    L.append("")
    return L
