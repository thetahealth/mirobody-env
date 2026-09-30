"""process.py -- the process-record contract and the process judges.

A solver may submit a `trace` object alongside its answer. It is read from the raw
response in `responses.jsonl`, never enters `SolverOutput`, and is scored only for
process evaluation, never folded into the outcome score. It is requested by the
`ddx.trace` probe; `ddx.direct` is unchanged.

The judges are mechanically recomputable. An empty set is recorded as not applicable,
never as full marks. Tracks whose judging is incomplete are listed in `UNIMPLEMENTED`,
with their machine-checked state in `TRACK_GAPS`.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- the contract (schema block appended to the question)
TRACE_SCHEMA_HINT = """
另外再给一个 "trace" 对象,记录你的**推理过程**(它不参与结论评分,只用于过程评估):
"trace":{{"hypotheses":[{{"name":"假说名","status":"considered|supported|excluded",
"evidence_for":["EV-..."],"evidence_against":["EV-..."]}}],
"tool_calls":[{{"target":"想查的信号或检查","why":"为什么查它"}}],
"commitments":[{{"claim":"你现在承诺的判断","falsified_by":"什么观测会推翻它"}}],
"data_notes":["数据质量问题"],"scope":"你认为超出本次判断范围的东西"}}
⚠️ 只写你**真的**依据的证据编号;编不出来就留空数组,留空**不扣过程分**,编造会被查出来。"""

_EV_RE = re.compile(r"EV-[A-Za-z0-9_.\-]+")

# Spec §19: tracks the gold declares but the judges do not yet cover, with a status note each.
UNIMPLEMENTED: dict[str, str] = {
    "discriminative_tool": "🟡 非交互形式已实现(`judges.judge_discriminative_tool`):判模型开的检查里"
                           "有没有能把金标与近名分开的判别项。交互形式(模型点单、环境揭示结果、"
                           "再判是否据此改判)未实现:请求-响应循环已有(`gated.run_gated`),"
                           "但内核 `gatekeeper.query` 对不在 S 里的 target 返回 "
                           "`{series: None, need_synth: True}`,检查项无结果可揭示。"
                           "需在内核补合成钩子(结果须与 S 一致)。",
    "revision_grounded": "🟡 切片形式已实现且有真实模型读数(`judges.judge_slice_revision` -> "
                        "`rev_responsiveness`,离线对照 flip_flop 1.000 / no_revision 0.000)。"
                        "多轮形式判据已实现(`judges.judge_multiround_revision`,读 trajectory 的 "
                        "`visible_ev` / `cited_ev` / `n_new_points`),缺一个真实模型的多轮批次。",
    "gap_naming_precision": "🟡 判据已实现(`discovery.judge_gap_naming`),夹具三侧对照已过"
                            "(指了不存在的 / 指了门后有但没查的 / 指了真拿不到的)。缺批次:"
                            "题面需多一个受构造约束的 `missing_data` 字段"
                            "(`discovery.GAP_SCHEMA_HINT`),走新 probe(`ddx.direct` 不变)。"
                            "现有 `data_quality.signal_quality` 的键名不受控,不能替代。",
    "neutral_window_drift": "🟢 已实现且有真实模型读数(`rev_stability`,中性窗由 "
                            "`slices: auto+neutral` 构造;离线对照 flip_flop 0.000 / "
                            "no_revision 1.000)。",
}

# ---------------------------------------------------------------- per-track status: three controlled values
GAP_STATUSES: tuple[str, ...] = (
    "no_production_judge",     # this judge does not exist on the production path -- genuinely unimplemented
    "judge_unexercised",  # the judge exists, but no batch on disk exercises it
    "form_partial",            # one form already has real-model readings; the other form is not yet closed
)

#: Per track: status, production judges, the `eval.jsonl` fields they produce, the measured
#: exercised surface, and what is missing. `gaps_wiring_defects()` checks that every judge
#: exists and produces the listed fields.
TRACK_GAPS: dict[str, dict] = {
    "discriminative_tool": {
        "status": "form_partial",
        "judges": ("haenv.judges.judge_discriminative_tool",
                   "haenv.judges.judge_slices_workup"),
        "fields": ("disc_recall", "wk_disc_recall_last"),
        "exercised": "Non-interactive form has readings (`disc_recall`, `wk_disc_recall_last`); "
                     "interactive form has none",
        "blocked_by": "Interactive form needs a synthesis hook in the kernel's `gatekeeper.query` path",
    },
    "revision_grounded": {
        "status": "form_partial",
        "judges": ("haenv.judges.judge_slice_revision",
                   "haenv.judges.judge_multiround_revision"),
        "fields": ("rev_responsiveness", "mr_grounded_rate"),
        "exercised": "Slice form has real-model readings (`rev_responsiveness`); multi-round "
                     "form `mr_grounded_rate` has readings from stub solvers only",
        "blocked_by": "A multi-round batch with a real model",
    },
    "neutral_window_drift": {
        "status": "form_partial",
        "judges": ("haenv.judges.judge_slice_revision",
                   "haenv.judges.judge_multiround_revision"),
        "fields": ("rev_stability", "mr_neutral_stability"),
        "exercised": "Slice form has readings (`rev_stability`); multi-round form "
                     "`mr_neutral_stability` is implemented but not exercised",
        "blocked_by": "Same as above: a multi-round batch with a real model",
    },
    "gap_naming_precision": {
        "status": "judge_unexercised",
        "judges": ("haenv.discovery.judge_gap_naming",),
        "fields": ("gap_naming_precision", "gap_named_n", "gap_named_obtainable"),
        "exercised": "No batch produces `gap_*` fields",
        "blocked_by": "The question face needs an extra construction-constrained `missing_data` field "
                      "(`discovery.GAP_SCHEMA_HINT`), via a new probe",
    },
}


def unimplemented_status() -> dict[str, str]:
    """`{track: status}` reported alongside every cell. A track missing from `TRACK_GAPS` or with
    broken wiring reports `no_production_judge`, the strictest status.
    """
    broken = {d.split(":", 1)[0] for d in gap_defects_cached()}
    out: dict[str, str] = {}
    for name in UNIMPLEMENTED:
        st = str((TRACK_GAPS.get(name) or {}).get("status") or "no_production_judge")
        if name in broken or st not in GAP_STATUSES:
            st = "no_production_judge"
        out[name] = st
    return out


def _keys_produced_by(dotted: str) -> set[str] | None:
    """Literal keys statically produced by `module.function` (an `ast.Dict` walk), or `None` if
    the function does not exist. Dynamically built keys are not seen.
    """
    import ast
    import importlib
    import inspect
    import pathlib
    mod_name, _, fn_name = dotted.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
        # Scan the function's defining file (judges can live in `haenv/judges/` submodules).
        fn_obj = getattr(mod, fn_name, None)
        src_path = inspect.getsourcefile(fn_obj) if fn_obj is not None else None
        src = pathlib.Path(str(src_path or mod.__file__)).read_text(encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        return None
    for fn in [n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name == fn_name]:
        keys: set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Dict):
                keys |= {k.value for k in n.keys
                         if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        return keys
    return None


def gaps_wiring_defects() -> list[str]:
    """Consistency defects of the `TRACK_GAPS` table; `[]` means clean.

    Checks that its keys match `UNIMPLEMENTED`, each status is in `GAP_STATUSES`, every judge
    is importable and the judges produce every listed field. Never raises: it runs on the
    judging path and reports through `trace_gap_table_defects`.
    """
    bad: list[str] = []
    miss = sorted(set(UNIMPLEMENTED) - set(TRACK_GAPS))
    extra = sorted(set(TRACK_GAPS) - set(UNIMPLEMENTED))
    if miss:
        bad.append(f"In UNIMPLEMENTED but not in TRACK_GAPS: {miss} (silently reverts to the strictest status)")
    if extra:
        bad.append(f"In TRACK_GAPS but not in UNIMPLEMENTED: {extra} (registers a track the artefact never reports)")
    for name, rec in sorted(TRACK_GAPS.items()):
        st = rec.get("status")
        if st not in GAP_STATUSES:
            bad.append(f"{name}: status={st!r} is not in GAP_STATUSES {GAP_STATUSES}")
        produced: set[str] = set()
        for dotted in (rec.get("judges") or ()):
            got = _keys_produced_by(str(dotted))
            if got is None:
                bad.append(f"{name}: `{dotted}` in judges not found -- "
                           f"the table points at a nonexistent judge")
                continue
            produced |= got
        if st == "no_production_judge":
            if rec.get("judges"):
                bad.append(f"{name}: status says \"no production judge\", yet {rec['judges']} is registered")
            continue
        orphan = sorted(f for f in (rec.get("fields") or ()) if f not in produced)
        if orphan:
            bad.append(f"{name}: fields {orphan} are not produced by the registered judges "
                       f"{tuple(rec.get('judges') or ())} -- "
                       f"either the judge is missing something, or this field cannot be traced to any exercised surface")
    return bad


_GAP_DEFECTS: list[list[str]] = []


def gap_defects_cached() -> list[str]:
    """Per-process cache of `gaps_wiring_defects()`."""
    if not _GAP_DEFECTS:
        try:
            _GAP_DEFECTS.append(gaps_wiring_defects())
        except Exception as e:                                 # noqa: BLE001
            # "Could not be checked" is reported, not treated as clean.
            _GAP_DEFECTS.append([f"<self-check could not run: {type(e).__name__}: {e}>"])
    return _GAP_DEFECTS[0]


def parse_trace(raw) -> dict | None:
    """The process record from the raw persisted JSON, or `None` if the cell has none."""
    if not isinstance(raw, dict):
        return None
    tr = raw.get("trace")
    return tr if isinstance(tr, dict) else None


def _ev_ids(obj) -> list[str]:
    """Evidence ids anywhere in the structure (matched by id shape only)."""
    out: list[str] = []
    if isinstance(obj, str):
        out += _EV_RE.findall(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _ev_ids(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _ev_ids(v)
    return out


def _resolve_ref(ref: str, vis: set[str]) -> str | None:
    """Resolve a reference to a visible evidence id. An exact match wins; otherwise a unique
    match on the trailing segment (an abbreviation such as `EV-01`); ambiguous -> `None`.
    """
    if ref in vis:
        return ref
    tail = ref.split("-")[-1]
    hit = [v for v in vis if v.split("-")[-1] == tail]
    return hit[0] if len(hit) == 1 else None


def judge_evidence_revealed(trace: dict, visible_ids) -> dict:
    """P1 . Every cited evidence id must have been revealed. No citations -> not applicable."""
    vis = set(visible_ids or ())
    refs = _ev_ids(trace)
    if not refs:
        return {"trace_n_evidence_refs": 0, "trace_unrevealed": None,
                "trace_evidence_revealed": None,
                "trace_evidence_note": "cited no evidence -> not applicable (not scored as full marks)"}
    bad = [r for r in refs if _resolve_ref(r, vis) is None]
    abbrev = [r for r in refs if r not in vis and _resolve_ref(r, vis) is not None]
    return {"trace_n_evidence_refs": len(refs), "trace_unrevealed": len(bad),
            "trace_evidence_revealed": round(1 - len(bad) / len(refs), 3),
            # Abbreviations get their own column: a format issue, not fabrication.
            "trace_abbrev_refs": len(abbrev) or None,
            "trace_unrevealed_ids": sorted(set(bad))[:5] or None}


def judge_hypotheses_split(trace: dict) -> dict:
    """P2 . Hypotheses laid out and exclusions with counter-evidence, reported separately.

    An exclusion without `evidence_against` is a pseudo-exclusion (same convention as
    `tracks._pseudo_exclusion`).
    """
    hs = trace.get("hypotheses")
    if not isinstance(hs, list) or not hs:
        return {"trace_n_hypotheses": 0, "trace_n_excluded": None,
                "trace_pseudo_excluded": None,
                "trace_hypotheses_note": "submitted no hypotheses -> not applicable"}
    excluded = [h for h in hs if isinstance(h, dict) and str(h.get("status")) == "excluded"]
    pseudo = [h for h in excluded if not (h.get("evidence_against") or [])]
    return {"trace_n_hypotheses": len(hs), "trace_n_excluded": len(excluded),
            "trace_pseudo_excluded": len(pseudo),
            "trace_exclusion_grounded": (round(1 - len(pseudo) / len(excluded), 3)
                                         if excluded else None)}


def judge_commitments_falsifiable(trace: dict) -> dict:
    """P3 . Share of commitments that state what would falsify them."""
    cs = trace.get("commitments")
    if not isinstance(cs, list) or not cs:
        return {"trace_n_commitments": 0, "trace_commitments_falsifiable": None,
                "trace_commitments_note": "gave no commitments -> not applicable (not scored as full marks)"}
    ok = [c for c in cs if isinstance(c, dict) and str(c.get("falsified_by") or "").strip()]
    return {"trace_n_commitments": len(cs),
            "trace_commitments_falsifiable": round(len(ok) / len(cs), 3)}


def judge_hypothesis_space_dynamics(trace: dict, visible_ids=()) -> dict:
    """P4 . Hypothesis-space dynamics: status consistent with evidence, no evidence used both for
    and against, and a penalty for locking onto a single supported hypothesis.
    """
    hs = trace.get("hypotheses")
    if not isinstance(hs, list) or not hs:
        return {"trace_hypothesis_dynamics": None,
                "trace_n_hypotheses_considered": 0,
                "trace_n_hypotheses_supported": 0,
                "trace_hypotheses_contradictory": None,
                "trace_hypotheses_dynamics_note": "submitted no hypothesis set -> not applicable"}

    vis = set(visible_ids or ())
    n_total = len(hs)
    n_supported = 0
    n_considered = 0
    n_contradictory = 0
    valid_hypotheses = 0

    for h in hs:
        if not isinstance(h, dict):
            continue
        status = str(h.get("status") or "").lower()
        ev_for = set(_ev_ids(h.get("evidence_for") or []))
        ev_against = set(_ev_ids(h.get("evidence_against") or []))

        overlap = ev_for & ev_against
        if overlap:
            n_contradictory += 1

        if status == "supported":
            n_supported += 1
            # a "supported" status should have supporting evidence
            if ev_for and not overlap:
                valid_hypotheses += 1
        elif status == "considered":
            n_considered += 1
            if not overlap:
                valid_hypotheses += 1
        elif status == "excluded":
            # an "excluded" status should have opposing evidence
            if ev_against and not overlap:
                valid_hypotheses += 1
        else:
            if not overlap:
                valid_hypotheses += 1

    # Premature lock: a single hypothesis, marked supported.
    premature_lock = (n_total == 1 and n_supported == 1)
    base_score = (valid_hypotheses / n_total) if n_total > 0 else 0.0
    score = round(base_score * (0.5 if premature_lock else 1.0), 3)

    return {
        "trace_hypothesis_dynamics": score,
        "trace_n_hypotheses_considered": n_considered,
        "trace_n_hypotheses_supported": n_supported,
        "trace_hypotheses_contradictory": n_contradictory,
        "trace_hypotheses_premature_lock": premature_lock,
    }


def judge_commitment_consistency(trace: dict, visible_ids=()) -> dict:
    """P5 . Commitment consistency: `claim` and `falsified_by` differ, and every cited id resolves
    to a visible observation.
    """
    cs = trace.get("commitments")
    if not isinstance(cs, list) or not cs:
        return {"trace_commitment_consistency": None,
                "trace_commitments_grounded_rate": None,
                "trace_commitment_consistency_note": "gave no commitments -> not applicable"}

    vis = set(visible_ids or ())
    n_cs = len(cs)
    consistent_count = 0
    total_ev_refs = 0
    grounded_ev_refs = 0

    for c in cs:
        if not isinstance(c, dict):
            continue
        claim = str(c.get("claim") or "").strip()
        falsified_by = str(c.get("falsified_by") or "").strip()
        if not claim or not falsified_by:
            continue

        if claim.lower() == falsified_by.lower():
            continue

        refs = _ev_ids(c)
        c_grounded = True
        for r in refs:
            total_ev_refs += 1
            if _resolve_ref(r, vis) is not None:
                grounded_ev_refs += 1
            else:
                c_grounded = False

        if c_grounded:
            consistent_count += 1

    score = round(consistent_count / n_cs, 3) if n_cs > 0 else 0.0
    # No citations => `grounded_rate` is `None` (flagged by `..._unmeasured`), never 1.0.
    grounded_rate = (round(grounded_ev_refs / total_ev_refs, 3)
                     if total_ev_refs > 0 else None)

    return {
        "trace_commitment_consistency": score,
        "trace_commitments_grounded_rate": grounded_rate,
        "trace_commitments_grounded_unmeasured": (total_ev_refs == 0),
        "trace_commitments_n_refs": total_ev_refs,
    }


def run_process_judges(raw, visible_ids=()) -> dict:
    """Run every process judge. No process record => `{}`.

    Every field carries the `trace_` prefix, kept apart from outcome fields.
    """
    tr = parse_trace(raw)
    if tr is None:
        return {}
    out: dict = {"trace_present": True}
    out.update(judge_evidence_revealed(tr, visible_ids))
    out.update(judge_hypotheses_split(tr))
    out.update(judge_commitments_falsifiable(tr))
    out.update(judge_hypothesis_space_dynamics(tr, visible_ids))
    out.update(judge_commitment_consistency(tr, visible_ids))
    out["trace_unimplemented"] = sorted(UNIMPLEMENTED)
    # Per-track state as a controlled value; `trace_unimplemented` keeps its exact form.
    out["trace_unimplemented_status"] = unimplemented_status()
    _gd = gap_defects_cached()
    if _gd:
        # Present only when the table is mis-filled.
        out["trace_gap_table_defects"] = _gd
    return out
