"""The judge-by-geometry table: for every `(judge, geometry)` pair, either the subject the
judge is mounted on or, in `WHY_NOT`, why it is not mounted.

A cell holds a subject, not a boolean:

========== ========================================== ====================
``OUT``    this one answer                            single-shot, gated
``ROWS``   the list of per-slice rows                 slices (cross-slice)
``LAST``   the last slice's / round's answer          slices, multi-round
``TRAJ``   one multi-round episode's trajectory       multi-round
``NONE``   not mounted; a reason is required          --
========== ========================================== ====================

Cell values are default-deny (anything outside ``SUBJECTS`` raises, at import and on
lookup); geometry names are validated at write time only, because rows may carry
``geometry = "<unrecorded>"``. A blank cell without a reason is a hole (:func:`holes`).

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

#: The five subjects. A cell holds one of these, not a boolean.
OUT = "out"       # this one answer (SolverOutput)
ROWS = "rows"     # the list of per-slice rows
LAST = "last"     # the last slice's answer
TRAJ = "traj"     # one multi-round episode's trajectory: a list of per-round dicts (see `TRAJ_ROW_FIELDS`)
NONE = "—"        # not mounted (a reason must be recorded in WHY_NOT)

#: The four built-in subjects (`NONE` marks absence and is not one).
_CORE_SUBJECTS: frozenset = frozenset({OUT, ROWS, LAST, TRAJ})

#: Externally registered subjects, keyed by name. See `register_subject`.
_EXTERNAL_SUBJECTS: dict[str, "SubjectSpec"] = {}


class SubjectSpec:
    """One registered observation subject: its name, who builds it, its shape.

    ``build(ctx) -> object | None`` derives the view from one cell's evaluation
    context. ``None`` means this cell cannot produce it, and the judge is skipped
    with a row recorded -- no other subject is substituted.
    """

    __slots__ = ("name", "build", "source", "why", "row_fields")

    def __init__(self, name: str, build, *, source: str, why: str,
                 row_fields: tuple = ()):
        if not name or not str(name).strip():
            raise ValueError("subject name must not be empty")
        if not callable(build):
            raise TypeError(f"subject {name!r}'s build must be callable -- "
                            f"registering a subject that cannot produce an object is registering a hole")
        if not source or not why:
            raise ValueError(f"subject {name!r} must state `source` (who produces it) and "
                             f"`why` (why the existing four are not enough) -- "
                             f"a subject that cannot state both should probably reuse one of the existing ones")
        self.name, self.build = str(name), build
        self.source, self.why, self.row_fields = source, why, tuple(row_fields)


def register_subject(name: str, build, *, source: str, why: str,
                     row_fields: tuple = ()) -> None:
    """Register an additional observation subject. Unregistered names still raise, and the
    four built-ins cannot be overridden (changing one changes readings outside the scoring
    fingerprint). `source` and `why` are required."""
    key = str(name)
    if key in _CORE_SUBJECTS or key == NONE:
        raise ValueError(f"{key!r} is a built-in subject (or the absence marker); it cannot be overridden -- "
                         f"changing a built-in means editing this file and going through the unfreeze process")
    if key in _EXTERNAL_SUBJECTS:
        raise ValueError(f"subject {key!r} is already registered (from {_EXTERNAL_SUBJECTS[key].source}) -- "
                         f"a name collision would make two judges believe they got the same view")
    _EXTERNAL_SUBJECTS[key] = SubjectSpec(key, build, source=source, why=why,
                                          row_fields=row_fields)


def unregister_subject(name: str) -> None:
    """Undo an external registration, for tests and negative controls. The four
    built-ins cannot be removed.
    """
    if str(name) in _CORE_SUBJECTS:
        raise ValueError(f"{name!r} is a built-in subject and cannot be unregistered")
    _EXTERNAL_SUBJECTS.pop(str(name), None)


def subject_specs() -> dict:
    """Registered external subjects. Reports and manifests read this rather than
    keeping their own list.
    """
    return {k: {"source": v.source, "why": v.why, "row_fields": list(v.row_fields)}
            for k, v in _EXTERNAL_SUBJECTS.items()}


class _Subjects(frozenset):
    """``SUBJECTS`` as a frozenset whose contents include external registrations, so existing
    ``in`` / ``sorted()`` / ``len()`` call sites keep working."""

    def __contains__(self, item):                      # noqa: D105
        return frozenset.__contains__(self, item) or item in _EXTERNAL_SUBJECTS

    def __iter__(self):                                # noqa: D105
        return iter(sorted(frozenset(self) | set(_EXTERNAL_SUBJECTS)))

    def __len__(self):                                 # noqa: D105
        return len(frozenset(self) | set(_EXTERNAL_SUBJECTS))


#: Every legal cell value: the built-ins plus registered subjects.
SUBJECTS: frozenset = _Subjects(_CORE_SUBJECTS)

#: Field names of a `TRAJ` row (the first return value of the kernel's multi-round runner).
TRAJ_ROW_FIELDS: tuple[str, ...] = (
    "round", "day", "risk_cat", "risk", "top_driver", "action", "repair",
    "visible_ev", "cited_ev", "n_new_points", "notes")

#: Geometry registry: `(name, claims(job), resolve(n_slices))`, first claim wins. The
#: slice geometry falls back to single-shot when there are fewer than two slices.
GEOMETRY_TABLE: tuple[tuple[str, object, object], ...] = (
    ("gated",  lambda job: bool(getattr(job, "gated", False)),
     lambda n: "gated"),
    ("slices", lambda job: bool(getattr(job, "slices", "")),
     lambda n: "single" if (n is not None and n < 2) else "slices"),
    ("multi",  lambda job: bool(getattr(job, "multiround", False)),
     lambda n: "multi"),
    ("single", lambda job: True,
     lambda n: "single"),
)

#: Geometry names in report column order (claim order is `GEOMETRY_TABLE`'s).
GEOMETRIES: tuple[str, ...] = ("single",) + tuple(
    g for g, _c, _r in GEOMETRY_TABLE if g != "single")


def claim_geometry(job, n_slices: int | None = None) -> str:
    """Which geometry a job runs on. The single place this is decided."""
    for name, claims, resolve in GEOMETRY_TABLE:
        if claims(job):
            return resolve(n_slices)
    raise RuntimeError("the geometry table has no fallback row -- the `single` row's claims must always be true")

#: The cross-product table. Multi-round `LAST` cells get the final answer by wrapping the
#: caller's solver in a recorder (`mounting.RoundRecorder`), without changing the kernel.
MOUNT: dict[str, dict[str, str]] = {
    # Single-shot and gated share cell values (same dispatch call).
    "forecast": {"single": OUT, "gated": OUT, "multi": LAST},
    "driver": {"single": OUT, "gated": OUT, "multi": LAST},
    "alternative": {"single": OUT, "gated": OUT, "multi": LAST},
    "workup": {"single": OUT, "gated": OUT, "multi": LAST},
    "abstention": {"single": OUT, "gated": OUT, "multi": LAST},
    "commit_timing": {"single": OUT, "gated": OUT, "multi": LAST},
    "review_flag": {"single": OUT, "gated": OUT, "multi": LAST},
    "self_contradictory_exclusion": {"single": OUT, "gated": OUT, "multi": LAST},
    "what_not_to_do": {"single": OUT, "gated": OUT, "multi": LAST},
    "dx_rival": {"single": OUT, "gated": OUT, "slices": LAST, "multi": LAST},
    "disc_tool": {"single": OUT, "gated": OUT, "slices": LAST, "multi": LAST},
    "join_selfcheck": {"single": OUT, "gated": OUT, "slices": LAST, "multi": LAST},
    "join_cover": {"single": OUT, "gated": OUT, "slices": LAST, "multi": LAST},

    # Diagnosis-naming family: slices are independent consultations, so the judge reads
    # the best-informed (last) one.
    "dx_unified": {"single": OUT, "gated": OUT, "slices": LAST, "multi": LAST},
    "dx_comorbidity": {"single": OUT, "gated": OUT, "slices": LAST, "multi": LAST},
    "dx_independent": {"single": OUT, "gated": OUT, "slices": LAST, "multi": LAST},
    "join_type": {"single": OUT, "gated": OUT, "slices": LAST, "multi": LAST},
    "join_evidence": {"single": OUT, "gated": OUT, "slices": LAST, "multi": LAST},

    # Slice-only: these judge something across slices, so the subject is the rows.
    "slices": {"slices": ROWS},
    "slice_revision": {"slices": ROWS},
    "slices_abstention": {"slices": ROWS},
    "slices_gates": {"slices": ROWS},
    "slices_review": {"slices": ROWS},
    "slices_workup": {"slices": ROWS},
    "slices_self_contradictory_exclusion": {"slices": ROWS},
    "slices_wnd": {"slices": ROWS},

    # Multi-round only. Not in the judge registry: the multi-round row builder calls these
    # directly, so they are pre-mounted (`unregistered_mounts()`).
    "multiround_revision": {"multi": TRAJ},
    "premise_repair": {"multi": TRAJ},
}

#: Why a cell is blank, keyed `"<judge>@<geometry>"` or `"<judge>@*"`.
WHY_NOT: dict[str, str] = {
    # Slice twins: the narrowness is deliberate; the slice side has its own.
    "workup@slices": "The slice side is covered by `slices_workup` (per-slice orders + `wk_*` aggregation).",
    "abstention@slices": "The slice side is covered by `slices_abstention` -- "
                         "on ddx questions the single-shot `sufficient` is always true and one-sided, which would give \"never abstains\" full marks.",
    "review_flag@slices": "The slice side is covered by `slices_review`.",
    "self_contradictory_exclusion@slices": "The slice side is covered by `slices_self_contradictory_exclusion`.",
    "what_not_to_do@slices": "The slice side is covered by `slices_wnd`.",
    "forecast@slices": "The early-warning dimension is covered wholesale on the slice geometry by `slices` (per-slice convergence/revision), not replicated per dimension.",
    "driver@slices": "Same as `forecast@slices`.",
    "alternative@slices": "This judges the enumeration of alternative hypotheses within a single answer; on the slice geometry each slice answers independently, "
                          "so \"were alternatives listed\" is covered by `slices`'s `converged_at` / `never_converged`.",

    # Not applicable because of the kinds.
    "commit_timing@slices": "The only kind is `ddx:insufficient`; that case is covered by `slices_abstention`.",

    # Multi-round, one reason per judge: slice judges read fields a trajectory row lacks.
    "slices@multi": "Reads the per-slice `all_drivers` / `answer_text` / `differential` / `join_type` / "
                    "`n_symptoms` / `raw_empty`; the trajectory row (`TRAJ_ROW_FIELDS`) has none of them. "
                    "Forcing it onto trajectory rows reads `converged_at` as always `None` and `never_converged` "
                    "as always `True`, even for a stub that answers correctly by construction -- "
                    "which reads as the conclusion \"never converged\" rather than \"cannot be measured\". "
                    "The multi-round side's \"did it converge across rounds\" is covered by `multiround_revision`'s "
                    "`mr_n_transitions` / `mr_grounded_rate`.",
    "slice_revision@multi": "Reads `visible_real` / `visible_other` / `visible_abnormal_lab` -- "
                            "these three buckets are split at collection time by `sp.evidence_ledger`'s `source_type` "
                            "(in `evaluate`), and `VerifierPayload` has no evidence ledger at all, "
                            "so the judge side cannot construct them. The trajectory row only has the unbucketed `visible_ev`. "
                            "The multi-round twin is `multiround_revision`: the same 2x2, reading the "
                            "`visible_ev` set difference + `n_new_points` instead.",
    "slices_abstention@multi": "Both tiers come from crossing the per-slice `visible_real` (is there enough information) with "
                               "`data_sufficiency` (did the model declare insufficiency); the trajectory row has neither field, "
                               "so forcing this judge onto a trajectory row would produce `abst_note='slice did not record "
                               "data_sufficiency (old batch)'`, a misleading note: the cause is the missing "
                               "field. The multi-round side's abstention calibration is covered by "
                               "`abstention@multi` (subject=`LAST`, judging the final round) -- "
                               "multi-round carries history, so \"should it have abstained earlier in the same consultation\" is not its construct.",
    "slices_gates@multi": "Runs the kernel's four action gates per slice, two of which read `clinician_review_required`; "
                          "the trajectory row lacks this field, so every round falls into the `unknown` branch and "
                          "every `sg_*` / `sg_*_at` dimension reads `None`. "
                          "The hard gates on the multi-round geometry are judged directly by the kernel's `run_multiround` "
                          "final-round `verifier.grade` (the `gates` field of `_row_multi`); "
                          "a per-round copy is not mounted on top.",
    "slices_review@multi": "Reads the last slice's `clinician_review_required` and `t`; the trajectory row lacks the former. "
                           "The multi-round side is covered by `review_flag@multi` (subject=`LAST`), "
                           "with a consistent convention (both judge the review decision at the most-informed point).",
    # Deliberately not prefixed with the gap marker: this cell is not applicable, not a gap.
    "slices_workup@multi": "Calls `judge_workup(_SliceOut(r))` per slice, and `_SliceOut` reads "
                           "`tests_to_order` / `referral_specialty` / `action` -- the trajectory row only has "
                           "`action`. Mounting it on a trajectory row would fabricate 0 on every dimension that "
                           "needs the two missing fields (recall / precision / discriminator) and a real-looking "
                           "score on the dimension that only needs `action`, in the same row and unflagged. "
                           "The multi-round side is covered by `workup@multi` (`LAST`), which gets the real answer.",
    "slices_self_contradictory_exclusion@multi":
        "Reads the per-slice `differential_evidence` (`ruled_out_by` / "
        "`supporting_evidence`) via `_EvOut`; the trajectory row has no differential list, let alone exclusion reasons. "
        "The multi-round side is covered by `self_contradictory_exclusion@multi` (`LAST`).",
    "slices_wnd@multi": "Also via `_EvOut`, reads the per-slice `what_not_to_do` field; the trajectory row has no such field. "
                        "The multi-round side is covered by `what_not_to_do@multi` (`LAST`).",

    "slices*@multi": "The family-level fallback for the slice-only judges (see the 8 exact `slices*@multi` keys above for each reason). "
                     "The subject is `slice_rows`, and the multi-round trajectory has not a single matching field: "
                     "the judge reads `visible_real` / `visible_other` / `visible_abnormal_lab` / `t` "
                     "(see `judges.judge_slice_revision`), while the trajectory row only has `visible_ev` / "
                     "`n_new_points` / `round` (see `TRAJ_ROW_FIELDS`). "
                     "Mounting it would make every reading constantly `None`, and in the artefacts "
                     "\"the judge runs but this batch has no judgeable transition\" would look identical to \"the judge cannot read the data\". "
                     "The multi-round equivalent is `multiround_revision` (the same 2x2, reading trajectory fields instead).",

    "*@multi": "Fallback for judges without a `multi` cell: an external plugin that mounts on multi "
               "gives an explicit cell value and reason in `mount()`; the built-in diagnosis-family judges "
               "mount `{'multi': LAST}` and run through `run_judges` in `evaluate._row_multi`.",

    # The two multi-round judges on other geometries: narrow by definition.
    "multiround_revision@*": "The subject is `TRAJ`, and none of the single-shot/gated/slices geometries produce a trajectory. "
                             "The slice-side equivalent is `slice_revision` (the same 2x2 in slice form, "
                             "reading `slice_rows`); single-shot only has one answer, "
                             "so the construct \"between adjacent rounds\" does not exist.",
    "premise_repair@*": "The subject is `TRAJ`. The single-shot-side equivalent is `judge_premise_challenge` "
                        "(called directly by `evaluate._row_single`), "
                        "which can only ask \"did this one response point out the conflict\"; "
                        "the revision track asks about something temporal (a revision only counts once the model changes its answer after the counter-evidence becomes visible), "
                        "and without a trajectory there is no \"after\".",

    # Slice-only judges on other geometries: narrow by definition.
    "slices*@single": "A slice-only judge; the subject is `slice_rows`, which no other geometry has.",
    "slices*@gated": "Same as above.",
    "slice_revision@single": "Same as `slices*@single`.",
    "slice_revision@gated": "Same as `slices*@single`.",
}


def mount(judge_name: str, cells: dict[str, str],
          why_not: dict[str, str] | None = None) -> None:
    """Register an external judge's mount cells (the other half of
    :func:`haenv.judges.register_judge`). Every geometry absent from `cells` needs a reason in
    `why_not`; built-in judges cannot be remounted at runtime."""
    if judge_name in MOUNT:
        raise ValueError(
            f"{judge_name!r} is already in the mount table (built into this repo). Mounts cannot be changed at runtime: "
            f"changing a mount changes readings, and the judging fingerprint covers source code only. Edit this file instead.")
    bad = {g: s for g, s in cells.items() if s not in SUBJECTS}
    if bad:
        raise ValueError(f"{judge_name!r} has an illegal subject {bad}; only {sorted(SUBJECTS)} are allowed. "
                         f"To leave it unmounted, omit it and give a reason in why_not; {NONE!r} is not a cell value.")
    unknown = set(cells) - set(GEOMETRIES)
    if unknown:
        raise KeyError(f"{judge_name!r} uses an unknown geometry {sorted(unknown)}; only {list(GEOMETRIES)} are allowed.")
    why_not = dict(why_not or {})
    missing = [g for g in GEOMETRIES
               if g not in cells
               and g not in why_not
               and reason_for(judge_name, g) is None]
    if missing:
        raise ValueError(
            f"{judge_name!r} is neither mounted nor given a reason on geometry {missing}. "
            f"A blank cell needs a reason: \"narrowed\" and \"never applicable\" look identical on the table "
            f"(see this module's header, Rule 1).")
    MOUNT[judge_name] = dict(cells)
    for g, why in why_not.items():
        WHY_NOT.setdefault(f"{judge_name}@{g}", why)


def subject_of(judge_name: str, geometry: str) -> str:
    """Subject for ``(judge, geometry)``, or ``NONE``. An illegal cell value raises; unknown
    geometry names do not."""
    s = (MOUNT.get(judge_name) or {}).get(geometry, NONE)
    if s != NONE and s not in SUBJECTS:
        raise ValueError(
            f"In the mount table, {judge_name!r}@{geometry!r}'s cell value {s!r} is not a legal subject "
            f"(only {sorted(SUBJECTS)} are allowed). It is not treated as unmounted: a mistyped cell value would "
            f"otherwise stop the judge, indistinguishable in the artefacts from \"not applicable\".")
    return s


def reason_for(judge_name: str, geometry: str) -> str | None:
    """Why a cell is blank, or None (a hole). Lookup order: ``name@geometry``, ``name@*``,
    the slice-family wildcard, ``*@geometry``."""
    for key in (f"{judge_name}@{geometry}", f"{judge_name}@*"):
        if key in WHY_NOT:
            return WHY_NOT[key]
    if judge_name.startswith(("slices", "slice_")) and f"slices*@{geometry}" in WHY_NOT:
        return WHY_NOT[f"slices*@{geometry}"]
    return WHY_NOT.get(f"*@{geometry}")


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


def known_gaps() -> list[str]:
    """`WHY_NOT` entries whose reason starts with the known-gap marker (red circle, U+1F534)."""
    return sorted(k for k, v in WHY_NOT.items() if v.lstrip().startswith("🔴"))


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


def _lint_table() -> list[str]:
    """Default-deny check of the whole table at import: cell values and geometry names.
    Registry membership is checked at runtime by :func:`unregistered_mounts`."""
    bad: list[str] = []
    for name, cells in MOUNT.items():
        if not isinstance(cells, dict):
            bad.append(f"{name}: cell is not a dict, it is {type(cells).__name__}")
            continue
        for g, s in cells.items():
            if g not in GEOMETRIES:
                bad.append(f"{name}@{g}: unknown geometry (only {list(GEOMETRIES)} are allowed)")
            if s not in SUBJECTS:
                bad.append(f"{name}@{g}: illegal subject {s!r} (only {sorted(SUBJECTS)} are allowed)")
    return bad


_LINT = _lint_table()
if _LINT:                                    # pragma: no cover - only reached when the table itself is wrong
    raise ValueError(
        "`mount_table.MOUNT` has illegal cell values, rejected at import: " + "; ".join(_LINT)
        + " (default-deny).")
