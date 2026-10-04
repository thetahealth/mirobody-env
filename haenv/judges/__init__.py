"""Judge registry: what gets judged follows the gold, not the execution shape.

Each judge declares which gold kinds it needs; a case without that gold does
not attach the judge (as opposed to scoring zero). Geometry mounting is
declared in :mod:`haenv.mount_table`. :data:`JUDGES` can be extended from out
of tree, but plugins are only loaded explicitly, never by scanning.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("haenv.judges")

from ._helpers import (  # noqa: F401
    MULTI, SINGLE, SLICES, _ddx, _gold, _names, _rank_of, _rivals_of, gold_kind, log)
from .differential import (  # noqa: F401
    TESTS_CAP_MARGIN, URGENCY_ACTION, _ACTION_ORDER, _COLLECTION_MARKERS, _COMPOUND_MARKERS,
    _OPTIONAL_MARKERS, _TESTV, _gold_test_optional, _gold_test_segs, _seg_forms, _segs,
    _threads_hit, join_claims_unified, join_scope_band, join_self_contradiction,
    judge_discriminative_tool, judge_dx_comorbidity, judge_dx_independent, judge_dx_rival,
    judge_dx_unified, judge_join_type, judge_workup, visible_ids_of)
from .outcome import (  # noqa: F401
    judge_alternative, judge_driver, judge_forecast)
from .trajectory import (  # noqa: F401
    QUANT_PCT_INDEX_DIVISOR, QUANT_TREND_DEADZONE, _CONCEPT_IDX, _SliceOut, _concept_index,
    _quant_truth, _slice_hit_unified, _slice_threads, _split_urgency_gap, _wnd_norm,
    concept_set, judge_abstention_calibration, judge_commit_timing,
    JOIN_EVIDENCE_MIN, NOOP_ANSWERS, NOOP_CONTRACT, judge_join_evidence, noop_answer_of,
    judge_multiround_revision, judge_noop_probe, judge_premise_challenge,
    judge_premise_repair, judge_quant_probe, judge_self_contradictory_exclusion,
    judge_slice_revision, judge_slices, judge_slices_abstention, judge_slices_workup,
    judge_what_not_to_do, what_not_to_do_boilerplate)
from .safety import (  # noqa: F401
    Judge, _ACTION_GATES, _EvOut, _GateSliceOut, _j_premise_repair, _rw_single_sce,
    _rw_single_wnd, _rw_slices_sce, _rw_slices_wnd, judge_review_flag,
    judge_slices_review_flag, judge_slices_safety_gates)

from ._helpers import MULTI, _gold, _rivals_of, gold_kind, log  # noqa: F401
from .differential import join_claims_unified, join_self_contradiction, judge_discriminative_tool, judge_dx_comorbidity, judge_dx_independent, judge_dx_rival, judge_dx_unified, judge_join_type, judge_workup  # noqa: F401
from .outcome import judge_alternative, judge_driver, judge_forecast  # noqa: F401
from .trajectory import judge_abstention_calibration, judge_commit_timing, judge_multiround_revision, judge_slice_revision, judge_slices, judge_slices_abstention, judge_slices_workup  # noqa: F401
from .safety import Judge, _j_premise_repair, _rw_single_sce, _rw_single_wnd, _rw_slices_sce, _rw_slices_wnd, judge_review_flag, judge_slices_review_flag, judge_slices_safety_gates  # noqa: F401

_CORE: tuple[Judge, ...] = (
    Judge("forecast",         ("forecast",),          judge_forecast),
    Judge("driver",           ("forecast",),          judge_driver),
    Judge("alternative",      ("*",),                 judge_alternative),
    Judge("dx_unified",       ("ddx:unified",),       judge_dx_unified),
    Judge("dx_comorbidity",   ("ddx:comorbidity",),   judge_dx_comorbidity),
    Judge("dx_independent",   ("ddx:independent",),   judge_dx_independent),
    Judge("join_type",        ("ddx:unified", "ddx:comorbidity", "ddx:independent"),
                                                      judge_join_type),
    # `join_evidence`: the ids behind `join_type`, checked by code (asked since `ddx-answer-v2`).
    Judge("join_evidence",    ("ddx:unified", "ddx:comorbidity", "ddx:independent"),
                                                      judge_join_evidence),
    # Attaches only when the case has a registered look-alike rival.
    Judge("dx_rival",         ("ddx:unified", "ddx:comorbidity", "ddx:independent"),
                                                      judge_dx_rival,
          when=lambda vp: bool(_rivals_of(vp))),
    Judge("disc_tool",        ("ddx:unified", "ddx:comorbidity", "ddx:independent"),
                                                      judge_discriminative_tool,
          when=lambda vp: bool(_rivals_of(vp))),
    # Self-consistency judges do not read the gold.
    Judge("join_selfcheck",   ("ddx:unified", "ddx:comorbidity", "ddx:independent"),
                                                      join_self_contradiction),
    # Coverage variant: more discriminative than `join_selfcheck`.
    Judge("join_cover",       ("ddx:unified", "ddx:comorbidity", "ddx:independent"),
                                                      join_claims_unified),
    Judge("workup",           ("ddx:unified", "ddx:comorbidity", "ddx:independent"),
                                                      judge_workup,
          when=lambda vp: bool(_gold(vp, "urgency") or _gold(vp, "tests")
                               or _gold(vp, "specialty"))),
    # Includes the insufficient-information tier, where "should it abstain" is tested.
    Judge("abstention",       ("ddx:unified", "ddx:comorbidity", "ddx:independent",
                               "ddx:insufficient"),
                                                      judge_abstention_calibration),
    Judge("slices",           ("*",),                 judge_slices),
    # Justified revision / no drift in the neutral window, reported separately.
    Judge("slice_revision",   ("*",),                 judge_slice_revision),
    Judge("commit_timing",    ("ddx:insufficient",),  judge_commit_timing),
    # Per slice: early slices lack information, the last one is sufficient.
    Judge("slices_abstention", ("*",),                judge_slices_abstention),
    # Returns {} (not attached) when the gold has no clinician-action field.
    Judge("review_flag",      ("*",),                 judge_review_flag),
    Judge("slices_gates",     ("*",),                 judge_slices_safety_gates),
    Judge("slices_review",    ("*",),                 judge_slices_review_flag),
    Judge("slices_workup",    ("ddx:unified", "ddx:comorbidity", "ddx:independent"),
                                                      judge_slices_workup,
          when=lambda vp: bool(_gold(vp, "urgency") or _gold(vp, "tests"))),
    # These read the answer's own structure (refutation ids, prohibitions), not gold,
    # so they have no `when`; geometry cells are in `mount_table.MOUNT`.
    Judge("self_contradictory_exclusion", ("*",),                _rw_single_sce),
    Judge("slices_self_contradictory_exclusion",   ("*",),                _rw_slices_sce),
    Judge("what_not_to_do",    ("*",),                _rw_single_wnd),
    Judge("slices_wnd",        ("*",),                _rw_slices_wnd),
    # Multi-round-only judges.
    Judge("multiround_revision", ("*",),               judge_multiround_revision),
    Judge("premise_repair",      ("*",),               _j_premise_repair),
)


#: The live registry: `_CORE` plus registered external judges. A list, mutated
#: in place, so every importer sees changes.
JUDGES: list[Judge] = list(_CORE)

#: External judges' provenance: `name -> source label`, recorded in the artefacts.
EXTERNAL: dict[str, str] = {}


def register_judge(judge: Judge, *, source: str = "external",
                   after: str | None = None, replace: bool = False) -> Judge:
    """Attach a judge to the live registry.

    :param source: provenance label, recorded in :data:`EXTERNAL`.
    :param after: insert after this judge; ``None`` appends. Order is semantics.
    :param replace: override a same-named external judge. Defaults to ``False``, which raises.

    Built-in judges cannot be replaced: the scoring fingerprint covers source
    code, not runtime substitutions.
    """
    if not isinstance(judge, Judge):
        raise TypeError(f"register_judge expects a Judge, got {type(judge).__name__}")
    if not judge.name or not judge.kinds or judge.fn is None:
        raise ValueError(f"incomplete judge registration: name={judge.name!r} kinds={judge.kinds!r}")
    _core_names = {j.name for j in _CORE}
    _existing = {j.name: i for i, j in enumerate(JUDGES)}
    if judge.name in _core_names:
        raise ValueError(
            f"{judge.name!r} is a judge built into this repo (`_CORE`); runtime replacement is not allowed -- "
            f"the judging fingerprint covers source code only, so a runtime substitution would decouple it from the actual scoring rule. "
            f"To change it: edit `_CORE` and go through unfreeze -> re-freeze -> recompute -> publish.")
    if judge.name in _existing:
        if not replace:
            raise ValueError(
                f"{judge.name!r} is already registered (source: {EXTERNAL.get(judge.name, '?')}). "
                f"Duplicate names are rejected by default: a same-named key would be overwritten by `out.update()`. "
                f"Pass replace=True to override it.")
        JUDGES[_existing[judge.name]] = judge
        EXTERNAL[judge.name] = source
        return judge
    if after is not None:
        if after not in _existing:
            raise KeyError(
                f"after={after!r} is not in the registry; current judges {[j.name for j in JUDGES]}. "
                f"Does not fall back to appending at the end: judge order affects the readings.")
        JUDGES.insert(_existing[after] + 1, judge)
    else:
        JUDGES.append(judge)
    EXTERNAL[judge.name] = source
    return judge


def unregister_judge(name: str) -> Judge:
    """Remove an external judge. Built-ins cannot be removed."""
    if name in {j.name for j in _CORE}:
        raise ValueError(f"{name!r} is a judge built into this repo; runtime removal is not allowed; edit `_CORE` and go through the unfreeze process.")
    for i, j in enumerate(JUDGES):
        if j.name == name:
            EXTERNAL.pop(name, None)
            return JUDGES.pop(i)
    raise KeyError(f"{name!r} is not in the registry; current judges {[j.name for j in JUDGES]}.")


def reset_judges() -> None:
    """Reset the registry to the built-in set. For tests."""
    JUDGES[:] = list(_CORE)
    EXTERNAL.clear()


def load_judge_plugins(group: str = "haenv.judges") -> list[str]:
    """Load external judges from an entry-point group and return the names attached.

    Each entry point resolves to a ``() -> Iterable[Judge]`` callable. A load
    failure raises.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:                      # pragma: no cover - only on very old Pythons
        return []
    added: list[str] = []
    seen: set[str] = set()
    for ep in entry_points(group=group):
        # an installed haenv and a checkout's `plugins/` shim can declare the same target
        if ep.value in seen:
            continue
        seen.add(ep.value)
        factory = ep.load()
        for j in factory() or ():
            register_judge(j, source=f"{group}:{ep.name}")
            added.append(j.name)
    return added


def registry_manifest() -> dict:
    """Which judges this board ran, built-in and external reported separately."""
    return {"n_core": len(_CORE), "n_total": len(JUDGES),
            "core": [j.name for j in _CORE],
            "external": dict(EXTERNAL),
            "order": [j.name for j in JUDGES]}


def applicable(geometry: str, kind: str, vp=None) -> list[Judge]:
    """Judges mounted on this geometry (per :mod:`haenv.mount_table`) whose gold this
    case can reach. Without ``vp`` this filters by kind and geometry only.
    """
    from ..mount_table import NONE as _MT_NONE
    from ..mount_table import subject_of as _subject_of
    cand = [j for j in JUDGES
            if _subject_of(j.name, geometry) != _MT_NONE
            and ("*" in j.kinds or kind in j.kinds)]
    if vp is None:
        return cand
    return [j for j in cand if j.when is None or bool(j.when(vp))]


def run_judges(geometry: str, subjects, vp, ctx: dict | None = None) -> dict:
    """Run every judge applicable on this geometry, merged into one row."""
    from ..mount_table import subject_of as _subject_of
    if not isinstance(subjects, dict):
        subjects = {"rows" if geometry == "slices" else "out": subjects}
    # On the multi-round geometry the `last` subject is the final round's response,
    # as the slice geometry uses the last slice. Missing round outputs leave it
    # unset (judges then record mount_subject_missing).
    if geometry == MULTI and "last" not in subjects:
        _ro = (ctx or {}).get("round_outputs") or ()
        if len(_ro):
            subjects["last"] = _ro[-1]
    # External subjects derive a view from existing ones, or return None to skip.
    from ..mount_table import _EXTERNAL_SUBJECTS as _ext_subj
    if _ext_subj:
        _bctx = {"subjects": dict(subjects), "vp": vp, "geometry": geometry, **(ctx or {})}
        for _sname, _spec in _ext_subj.items():
            if _sname in subjects:            # caller already supplied this view ⇒ caller wins
                continue
            try:
                _built = _spec.build(_bctx)
            except Exception as _e:           # noqa: BLE001 - a failing external view must not sink the cell
                log.error("[judges] external subject %s failed to produce a value: %s (%s)",
                          _sname, _e, type(_e).__name__)
                _built = None
            if _built is not None:
                subjects[_sname] = _built
    kind = gold_kind(vp)
    out: dict = {"gold_kind": kind}
    for j in applicable(geometry, kind, vp):
        want = _subject_of(j.name, geometry)
        if want not in subjects or subjects[want] is None:
            out.setdefault("mount_subject_missing", []).append(f"{j.name}@{geometry}:{want}")
            continue
        try:
            out.update(j.fn(subjects[want], vp, ctx or {}) or {})
        except Exception as e:                       # one broken judge must not sink the cell
            import traceback as _tb
            _frames = [f for f in _tb.extract_tb(e.__traceback__)
                       if "haenv" in f.filename or "metabolic_harness" in f.filename]
            _at = (f"{_frames[-1].filename.rsplit('/', 1)[-1]}:{_frames[-1].lineno} "
                   f"`{_frames[-1].line}`") if _frames else "?"
            log.error("[judges] %s failed: %s (%s @ %s)", j.name, e, type(e).__name__, _at)
            out[f"{j.name}_error"] = str(e)[:120]
            # `judge_errors` lets the publish gate reject batches with crashed judges.
            out.setdefault("judge_errors", []).append(j.name)
    return out


# ================================================================ LLM judges
#
# Attached only by an explicit call; no request is sent unless `llm.ENV_SWITCH`
# is on (off by default).
def register_llm_judges(*, source: str = "haenv.judges.llm") -> list[str]:
    """Attach the in-tree LLM judges and return the names newly attached (idempotent)."""
    from . import llm as _llm
    added: list[str] = []
    for j in _llm.judges():
        register_judge(j, source=source)
        added.append(j.name)
    return added


def categories() -> dict[str, str]:
    """Map judge name to category, read live from :data:`JUDGES`."""
    return {j.name: getattr(j, "category", "deterministic") for j in JUDGES}
