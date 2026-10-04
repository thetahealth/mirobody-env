"""Q-side probes and prompt assembly: premise, no-op, quantity and oracle probes, the probe
registry, and the prompt a solver is shown.

Split out of `haenv/evaluate.py`; `evaluate` re-exports every name defined here.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml
import json
from haenv_kernel.solver import ALLOWED_DRIVERS, ACTION_CLASSES
from .judges.trajectory import NOOP_CONTRACT as _NOOP_CONTRACT
from .prompts import PROMPT  # noqa: F401  (re-export: `evaluate.PROMPT` keeps working)
from .framings import DDX_PROMPT
from .framings import _framings, framing_sha256
from .run_state import RunContext, log, run_ctx_of
from .external_gold import external_probes as _ext_probes
from . import rng
from .judges.trajectory import _quant_truth
from .judges.trajectory import QUANT_TREND_DEADZONE
from .judges.trajectory import QUANT_PCT_INDEX_DIVISOR as _PDIV
from .mount_table import claim_geometry


# ---------------------------------------------------------------- Q-side interference: false premise (FP)
#
# A false premise must be adjudicable from W, may name only quantities already in the
# framing (a new name would leak that it matters), and is balanced by true premises
# (3 false : 2 true) so that always rejecting the premise does not win.
FALSE_PREMISE_TEMPLATE = """
⚠️ 补充一句背景(**你必须自行核对它是否与上面的数据一致**):
{premise}
若它与你看到的数据冲突,**明确指出冲突**并按数据作答;不要因为这句话改变你的结论。"""


def build_premise(raw, kind: str) -> dict | None:
    """Build one premise from W: `kind='false'` conflicts with W, `'true'` agrees with it.

    Uses only lab values already in the framing. `truth`/`asserted` are verifier-side only.
    """
    led = [e for e in (raw.evidence_ledger or []) if isinstance(e, dict)]
    labs = [e for e in led if str(e.get("source_type")) == "lab_result"]
    if not labs:
        return None
    import re as _re
    m = None
    for e in labs:                       # find one with a numeric value and a reference range: only then is truth/falsity adjudicable
        m = _re.search(r"^(.+?)\s+([-+]?\d+(?:\.\d+)?)\s*([^\s(（]*)\s*[(（]\s*参考\s*"
                       r"([-+]?\d+(?:\.\d+)?)\s*[–\-~至]\s*([-+]?\d+(?:\.\d+)?)",
                       str(e.get("symptom") or ""))
        if m:
            break
    if not m:
        return None
    name, val, unit, lo, hi = m.group(1), float(m.group(2)), m.group(3), float(m.group(4)), float(m.group(5))
    if kind == "true":
        # A true premise is rounded like a self-report, so decimals do not give away the true arm.
        said = round(val) if abs(val) >= 100 else round(val, 1)
        txt = f"我记得你的{name}大约是 {said:g}{unit}。"
    else:
        # The false value is abnormal but plausible: one reference-range width above the upper
        # bound (below the lower bound if the true value is already high), clamped above 0.
        span = max(hi - lo, 0.01)
        if val <= hi:
            said = round(hi + span, 2)                       # above the upper bound by one span width
        else:
            said = round(max(lo - span, 0.01 if lo >= 0 else lo - span), 2)
        txt = f"我记得你的{name}大约是 {said:g}{unit}。"
    return {"text": txt, "polarity": kind, "field": name.strip(),
            "truth": val, "asserted": said, "unit": unit, "ref": [lo, hi],
            # The counter-evidence EV; `judge_premise_repair` finds the round in which it becomes visible.
            "evidence_id": str(e.get("evidence_id") or "") or None}


def _render_prompt(payload, mode: str = "default") -> str:
    """Render the default or the ddx framing (the default PROMPT has no slot for a diagnosis).
    Every model on a question type sees the same text.
    """
    tpl = DDX_PROMPT if mode == "ddx" else PROMPT
    return tpl.format(target=payload.prediction_context.get("target_event_type"),
                      drivers=", ".join(ALLOWED_DRIVERS), actions=ACTION_CLASSES,
                      payload=payload.dumps())


_FRAMINGS = _framings()                                   # registry; render_with_probe computes it fresh each time


def load_probes(root) -> dict:
    """Load `probes/*.yaml` (default-deny).

    Each probe's `framing_sha256` must match its template, so a template edit cannot silently
    change both arms of an experiment.
    """
    from pathlib import Path
    out: dict = {}
    for f in sorted(Path(root).glob("*.yaml")):
        d = _cached_yaml(f) or {}
        pid = d.get("probe_id")
        if not pid:
            raise ValueError(f"{f.name}: missing probe_id")
        if pid in out:
            raise ValueError(f"duplicate probe_id: {pid}")
        ref = d.get("framing_ref")
        if ref not in _framings():
            raise ValueError(f"{pid}: framing_ref={ref!r} is not registered")
        want = d.get("framing_sha256")
        if not want:
            raise ValueError(f"{pid}: missing framing_sha256 (required: it pins the "
                             f"framing the probe was written against)")
        got = framing_sha256(ref)
        if got != want:
            raise ValueError(
                f"{pid}: framing_sha256 mismatch: declared {want}, actual {got}. "
                f"Revert {ref} or update the probe's framing_sha256.")
        out[pid] = d
    for pid, d in _ext_probes().items():
        if pid in out:
            raise ValueError(f"external probe_id collides with a built-in probe: {pid}")
        ref = d.get("framing_ref")
        if ref not in _framings():
            raise ValueError(f"{pid} (external): framing_ref={ref!r} is not registered")
        want, got = d.get("framing_sha256"), framing_sha256(ref)
        if not want:
            raise ValueError(f"{pid} (external): missing framing_sha256 (required)")
        if got != want:
            raise ValueError(f"{pid} (external): framing_sha256 mismatch: declared {want}, "
                             f"actual {got}")
        out[pid] = d
    return out


def render_with_probe(payload, probe_id: str, probes: dict, allow_retired: bool = False) -> str:
    """Render the framing for `probe_id`. Unregistered ids raise; `retired-*` probes are
    refused unless `allow_retired` (they exist only to reproduce old batches).
    """
    p = (probes or {}).get(probe_id)
    if not p:
        raise KeyError(f"unregistered probe_id: {probe_id} (registered: {sorted(probes or {})})")
    if str(p.get("status") or "").startswith("retired") and not allow_retired:
        raise ValueError(
            f"{probe_id} is retired and cannot be used for new runs: {p.get('retired_reason')}"
            f" (record: {p.get('retired_record')}). To reproduce an old batch pass allow_retired=True")
    tpl = _framings()[p["framing_ref"]]
    return tpl.format(target=payload.prediction_context.get("target_event_type"),
                      drivers=", ".join(ALLOWED_DRIVERS), actions=ACTION_CLASSES,
                      payload=payload.dumps())


DEFAULT_PROBE = {"ddx": "ddx.direct", "default": "forecast.direct"}


def resolve_probe(obj, mode: str) -> str:
    return getattr(obj, "probe_id", "") or DEFAULT_PROBE[mode]


# ============================================================ Q-side: false-premise probe
#
# True premises are the false-alarm control: flagging a true premise is a false positive.
PREMISE_RATIO = (3, 2)          # false : true


def assign_premises(built: dict, ratio: tuple[int, int] = PREMISE_RATIO, *, ctx: RunContext) -> dict:
    ctx.premise_for.clear()
    nf, nt = ratio
    for cid, raw in sorted(built.items()):
        k = "false" if int(rng.unit(cid, "premise", "polarity") * (nf + nt)) < nf else "true"
        prem = build_premise(raw, k)
        if prem:
            ctx.premise_for[cid] = prem
    return dict(ctx.premise_for)


def _premise_suffix(payload, ctx: RunContext) -> str:
    """Sentence appended to the question, outside the framing template so that
    `framing_sha256` is unaffected.
    """
    prem = ctx.premise_for.get(str(getattr(payload, "case_id", "")
                               or (getattr(payload, "prediction_context", {}) or {}).get("case_id", "")))
    if not prem:
        return ""
    return ("\n\n【患者补充自述】" + str(prem["text"])
            + "\n(若这条自述与上面的资料不符,请在 `notes` 里明确指出不符之处。)")


def _gated_suffix(solver) -> str:
    """Query protocol appended to the question in the gated geometry (budget and menu from
    `solver.gated_context`).

    The menu includes decoy signals this patient does not have, so ungrounded queries are
    possible, and the budget (`gated.budget_for`) covers only part of the menu, so the model
    must choose.
    """
    ctx = getattr(solver, "gated_context", None)
    if not ctx:
        return ""
    menu = ctx.get("menu") or []
    if menu:
        _sig = [i for i in menu if not i.get("is_test")]
        _tst = [i for i in menu if i.get("is_test")]
        def _fmt(xs):
            return "\n".join(f"    - {i['target']}(kind={i['kind']},花费 {i['cost']})" for i in xs)
        lines = ("  【A 监测信号】(这位患者的逐日数据;**可能含与他无关的项**)\n" + _fmt(_sig)
                 + ("\n  【B 检查项】(标准检查目录;点它等于**为这位患者开这项检查**)\n"
                    + _fmt(_tst) if _tst else ""))
    else:                              # compatibility: fall back to the whitelist when there is no menu (reproducing old batches)
        lines = "    " + ", ".join(str(x) for x in (ctx.get("askable") or [])) or "    (无)"
    return (
        "\n\n【按需查询协议】部分数据**未直接给出**,你可以先查询再作答。"
        f"\n· 本轮:第 {ctx.get('round')}/{ctx.get('max_rounds')} 轮"
        f"\n· 预算:已花 {ctx.get('spent')} / 上限 {ctx.get('budget')}"
        f"\n  ⚠️ **预算点不全整张菜单** —— 必须挑。超出会被截断。"
        f"\n· 可查项(名称 / 类别 / 花费):\n{lines}"
        "\n  ⚠️ 菜单里**可能有与这位患者无关的项** —— 点了照样计费,请自行判断。"
        "\n· 要查询就只输出 "
        '{"queries": [{"kind": "<上表的 kind>", "target": "<上表的名称>"}], "commit": false}'
        "\n· 要作答就按前面要求的字段输出答案(出现答案字段即视为收敛,不必显式 commit)"
        "\n· ⚠️ **B 段点过的项已计入「你开的检查」** —— `tests_to_order` 里只需再填"
        "**目录未列**的检查(如需专科操作的项目)。A 段的监测信号不算开检查。"
        + ("\n· 已请求但环境无法提供可解释结果的项目（不是正常或阴性，不能据此排除疾病）: "
           + json.dumps(ctx["unavailable_results"], ensure_ascii=False)
           if ctx.get("unavailable_results") else ""))


# ============================================================ Q-side: noop probe
#
# Asks about a window with no data and reads the structured `data_quality` fields to see
# whether the model says so. Polarity 3 absent : 2 present.
NOOP_RATIO = (3, 2)          # absent : present


#: Real clinical signal names that never appear in the cases.
NOOP_ABSENT_POOL: tuple[str, ...] = (
    "cgm_glucose_series", "sleep_stage_rem_ratio", "ecg_qtc_series",
    "spirometry_fev1_series", "bone_turnover_ctx", "urine_albumin_series",
)


#: Minimum empty stretch (days) that counts as "no data": with daily measurement,
#: 14 days is unambiguously a gap.
NOOP_GAP_MIN_DAYS = 14


#: Minimum number of points a covered window must have (otherwise "this has
#: data" itself would not hold up)
NOOP_COVERED_MIN_PTS = 5


def _noop_window(pts: list[dict], T: int, want_gap: bool):
    """Find a window among the points visible at <=T: `want_gap` wants an
    empty window, otherwise a densely covered one.

    Returns `(t0, t1, n_pts_in_window)`; `None` if none found. Deterministic
    (takes the first one that qualifies).
    """
    ts = sorted({int(p["ts"]) for p in pts if int(p["ts"]) <= T})
    if not ts:
        return None
    if want_gap:
        for a, b in zip(ts, ts[1:]):
            if b - a > NOOP_GAP_MIN_DAYS:
                return (a + 1, b - 1, 0)
        if ts[0] > NOOP_GAP_MIN_DAYS:
            return (0, ts[0] - 1, 0)
        return None
    for i in range(len(ts)):
        t0 = ts[i]
        t1 = t0 + NOOP_GAP_MIN_DAYS - 1
        n = sum(1 for x in ts if t0 <= x <= t1)
        if n >= NOOP_COVERED_MIN_PTS and t1 <= T:
            return (t0, t1, n)
    return None


def build_noop_probe(raw, kind: str, T: int | None = None) -> dict | None:
    """Build one noop probe about a time window of a signal the model has seen.

    `gap` asks about >=14 empty days (correct: declare it missing); `covered` asks about a
    densely covered window (correct: do not). Ground truth uses only points at <=T.
    """
    ld = raw.longitudinal_data or {}
    _T = int(T if T is not None else (raw.prediction_context or {}).get("prediction_time_T", 10**9))
    want_gap = (kind != "present")
    # Gap side: rank by largest gap (one point suffices); covered side: rank by point count
    # (at least `NOOP_COVERED_MIN_PTS`).
    def _prof(v):
        ts = sorted({int(p["ts"]) for p in (v or ())
                     if isinstance(p, dict) and int(p.get("ts", 10**9)) <= _T})
        if not ts:
            return 0, 0
        biggest = max([b - a for a, b in zip(ts, ts[1:])] + [ts[0]])
        return len(ts), biggest

    prof = {k: _prof(v) for k, v in ld.items() if isinstance(v, list)}
    if want_gap:
        cand = sorted(((g, n, k) for k, (n, g) in prof.items() if n >= 1), reverse=True)
        cand = [(n, k) for _g, n, k in cand]
    else:
        cand = sorted(((n, k) for k, (n, _g) in prof.items()
                       if n >= NOOP_COVERED_MIN_PTS), reverse=True)
    if not cand:
        return None
    for _n, sig in cand:
        pts = [p for p in ld[sig] if isinstance(p, dict) and "ts" in p]
        w = _noop_window(pts, _T, want_gap)
        if w is None:
            continue
        t0, t1, npts = w
        return {"target": sig, "polarity": ("gap" if want_gap else "covered"),
                "truth_present": not want_gap,
                # answer contract version (enum in `judges.NOOP_ANSWERS`)
                "contract": _NOOP_CONTRACT,
                "window": [int(t0), int(t1)], "n_pts_in_window": int(npts), "T": _T,
                "text": (f"另外,请只根据 `{sig}` 在**第 {t0}–{t1} 天**这一段的读数,"
                         f"说明它在该时段的变化趋势。")}
    return None


def assign_noop_probes(built: dict, ratio: tuple[int, int] = NOOP_RATIO, *, ctx: RunContext) -> dict:
    ctx.noop_for.clear()
    na, np_ = ratio
    for cid, raw in sorted(built.items()):
        k = "absent" if int(rng.unit(cid, "noop", "polarity") * (na + np_)) < na else "present"
        pr = build_noop_probe(raw, k, int((raw.prediction_context or {}).get(
            "prediction_time_T", 0)) or None)
        if pr:
            ctx.noop_for[cid] = pr
    return dict(ctx.noop_for)


def _noop_suffix(payload, ctx: RunContext) -> str:
    cid = str(getattr(payload, "case_id", "")
              or (getattr(payload, "prediction_context", {}) or {}).get("case_id", ""))
    pr = ctx.noop_for.get(cid)
    if not pr:
        return ""
    return ("\n\n【补充提问】" + str(pr["text"])
            + f"\n(**必须**在 `data_quality.signal_quality` 的 `{pr['target']}` 键填写下列三者之一,"
              f"原样照抄:`present` = 这一段有读数;`no_data_in_window` = 这一段没有读数;"
              f"`unreliable` = 有读数但不可信。有读数时照常说明趋势。"
              f"**请核对时点,不要凭印象作答。**)")


# ============================================================ Q-side: computable question types
#
# `trend` (enum) and `peak_day` (integer day) are answered from the data and judged by exact
# match; integers and enums need no tolerance parameter.
def assign_oracle_gold(built: dict, *, ctx: RunContext) -> dict:
    from haenv_kernel.build import build_instance                     # kernel
    ctx.oracle_gold_tests.clear()
    ctx.oracle_disc_names.clear()
    ctx.oracle_diagnosis.clear()
    for cid, raw in sorted(built.items()):
        try:
            _, vp = build_instance(raw, int(raw.prediction_context["prediction_time_T"]))
            ddx = (getattr(vp, "adjudication", None) or {}).get("ddx") or {}
            ctx.oracle_gold_tests[str(cid)] = [str(t) for t in (ddx.get("tests") or [])
                                           if str(t).strip()]
            # Imported here: loading `overlay` reads every registry table (~0.3 s).
            from .overlay import rivals_for
            from .registry import load_findings
            _fx = load_findings()
            _names: list[str] = []
            for _r in (rivals_for(str(ddx.get("spec_id") or "")) or ()):
                _fid = (_r.get("discriminator_finding") or {}).get("finding")
                _f = _fx.get(_fid) or {}
                _nm = _f.get("name_cn") or _f.get("name_en") or _fid
                if _nm and str(_nm) not in _names:
                    _names.append(str(_nm))
            ctx.oracle_disc_names[str(cid)] = _names
            if str(ddx.get("diagnosis") or "").strip():
                ctx.oracle_diagnosis[str(cid)] = str(ddx["diagnosis"]).strip()
        except Exception:                                # noqa: BLE001
            continue
    return dict(ctx.oracle_gold_tests)


def assign_oracle_warranted(built: dict, *, ctx: RunContext) -> dict:
    """Register each case's `clinician_action_warranted` for the `oracle_review_*` stubs, which
    give `review_macro` (specificity; 0.000 for "always says so") an offline range.
    """
    from haenv_kernel.build import build_instance                     # kernel
    ctx.oracle_warranted.clear()
    for cid, raw in sorted(built.items()):
        try:
            _, vp = build_instance(raw, int(raw.prediction_context["prediction_time_T"]))
            adj = getattr(vp, "adjudication", None) or {}
            if "clinician_action_warranted" in adj:
                ctx.oracle_warranted[str(cid)] = bool(adj["clinician_action_warranted"])
        except Exception:                                # noqa: BLE001
            continue
    return dict(ctx.oracle_warranted)


# `abnormal_days` is not in rotation: against the series' own p10/p90, its answer follows
# from the point count alone.
_QUANT_KINDS = ("trend", "peak_day")


#: Minimum net change for a trend question: `|last - first| >= TREND_MIN_NET_FRAC * (max - min)`
#: over the answer window. Read by `build_quant_probe` (trend candidates that fail it are
#: skipped): a series whose net change is a small fraction of its own swing has no
#: answerable direction. The `flat` deadzone itself is `judges.QUANT_TREND_DEADZONE`.
TREND_MIN_NET_FRAC = 0.25


#: Window stability for a trend question: the truth must not change when the answer window
#: ends up to TREND_WINDOW_PROBE days earlier or later. Read by `build_quant_probe`.
TREND_WINDOW_PROBE = 1


def _trend_candidate_ok(raw_pts: list, T: int) -> tuple[bool, dict]:
    """Both trend gates for one series over the window ending at `T`.

    Returns `(ok, detail)`. Truth is `judges._quant_truth` (the judge's formula); a shifted
    window with fewer than four points is not compared (the question needs four points).
    """

    def _win(t: int) -> list[tuple[int, float]]:
        return sorted((int(p["ts"]), float(p["value"])) for p in raw_pts
                      if isinstance(p, dict) and "value" in p and "ts" in p
                      and int(p["ts"]) <= int(t))

    vis = _win(T)
    if len(vis) < 4:
        return False, {"n_visible": len(vis)}
    t0 = _quant_truth("trend", vis)
    stable = True
    for dt in range(1, int(TREND_WINDOW_PROBE) + 1):
        for t in (int(T) - dt, int(T) + dt):
            w = _win(t)
            if len(w) >= 4 and _quant_truth("trend", w) != t0:
                stable = False
    vals = [v for _, v in vis]
    amp = max(vals) - min(vals)
    net = amp > 0 and abs(vals[-1] - vals[0]) >= TREND_MIN_NET_FRAC * amp
    return (stable and net), {"window_stable": stable, "net_change_ok": bool(net)}


def build_quant_probe(raw, kind: str, T: int | None = None,
                      avoid_signal: str | None = None) -> dict | None:
    """Build one computable probe with code-computed ground truth.

    Ground truth uses only points at <=T. The series is stored in the probe so the judge can
    recompute against the slice actually answered; for `trend`, `assign_quant_probes` passes
    the answer window as `T`, so both sides compute over the same points.
    """
    ld = raw.longitudinal_data or {}
    _T = int(T if T is not None else (raw.prediction_context or {}).get("prediction_time_T", 10**9))
    cand = sorted(((len(v or ()), k) for k, v in ld.items() if isinstance(v, list) and len(v) >= 4),
                  reverse=True)
    if not cand:
        return None
    # Try candidates until the ground truth is not degenerate (filler streams often peak at the start).
    for _n_pts, sig in cand:
        if avoid_signal and sig == avoid_signal:
            continue
        raw_pts = ld[sig]                               # all points, not window-cropped (needed by the window-perturbation probe)
        pts = [p for p in ld[sig] if isinstance(p, dict) and "value" in p and "ts" in p
               and int(p["ts"]) <= _T]                  # only <=T, see docstring
        if len(pts) < 4:
            continue
        pts = sorted(pts, key=lambda p: int(p["ts"]))
        if kind == "peak_day":
            _i = max(range(len(pts)), key=lambda i: float(pts[i]["value"]))
            if _i in (0, len(pts) - 1):                 # peak on the boundary => guessable, try the next signal
                continue
            _mx = max(float(p["value"]) for p in pts)
            if sum(1 for p in pts if abs(float(p["value"]) - _mx) < 1e-9) > 1:
                continue
        elif kind == "trend":
            # Two gates (window stability, minimum net change); a failing series is skipped.
            if not _trend_candidate_ok(raw_pts, _T)[0]:
                continue
        elif kind == "abnormal_days":
            _vs = sorted(float(p["value"]) for p in pts)
            _k = len(_vs) // _PDIV
            _lo, _hi = _vs[_k], _vs[-1 - _k]
            _cnt = sum(1 for p in pts if not (_lo <= float(p["value"]) <= _hi))
            if _cnt in (0, len(pts)):                   # everything inside or everything outside => no information
                continue
        break
    else:
        # Every candidate degenerated: ask for `peak_value` instead (unique even when the peak day
        # is tied), recorded in `fallback_from`.
        if kind != "peak_value":
            _fb = build_quant_probe(raw, "peak_value", T, avoid_signal=avoid_signal)
            if _fb:
                _fb["fallback_from"] = kind
                return _fb
        return None                                     # not even peak_value works (not enough points)
    if kind == "trend":
        d = float(pts[-1]["value"]) - float(pts[0]["value"])
        rng = max(abs(float(p["value"])) for p in pts) or 1.0
        _dz = QUANT_TREND_DEADZONE * rng
        truth = "rising" if d > _dz else ("falling" if d < -_dz else "flat")
        q = (f"根据 `{sig}` 这条逐日数据,它从最早一次到最后一次的总体方向是什么?"
             f"只回答 rising / falling / flat 之一。")
    elif kind == "peak_day":
        truth = int(max(pts, key=lambda p: float(p["value"]))["ts"])
        q = f"`{sig}` 在哪一天(第几天,整数)达到最高值?只回答那个整数。"
    elif kind == "peak_value":
        truth = round(max(float(p["value"]) for p in pts), 6)
        q = (f"`{sig}` 在你能看到的这些天里,**最高**的那次读数是多少?"
             f"只回答那个数值(与数据中出现的写法一致)。")
    else:
        vs = sorted(float(p["value"]) for p in pts)
        _k = len(vs) // _PDIV
        lo, hi = vs[_k], vs[-1 - _k]
        truth = sum(1 for p in pts if not (lo <= float(p["value"]) <= hi))
        q = (f"`{sig}` 有多少天的读数 **严格小于 {lo:g} 或严格大于 {hi:g}**?"
             f"(只数你能看到的那些天)只回答一个整数。")
    return {"kind": kind, "signal": sig, "truth": truth, "text": q,
            "series": [[int(p["ts"]), float(p["value"])] for p in pts],
            "T": _T}


def quant_probe_for(raw, kind: str, signal: str, T: int | None = None) -> dict | None:
    """The quant probe for a recorded `(kind, signal)`, in `build_quant_probe`'s format, with
    points at <=T (default `prediction_time_T`). None when the series has fewer than four such
    points. Used to judge an existing answer on the question its row records."""
    _T = int(T if T is not None else (raw.prediction_context or {}).get("prediction_time_T", 10**9))
    pts = sorted((int(p["ts"]), float(p["value"])) for p in ((raw.longitudinal_data or {}).get(signal) or [])
                 if isinstance(p, dict) and "value" in p and "ts" in p and int(p["ts"]) <= _T)
    if len(pts) < 4 or kind not in ("trend", "peak_day", "peak_value", "abnormal_days"):
        return None
    return {"kind": kind, "signal": signal, "truth": _quant_truth(kind, pts), "text": "",
            "series": [[t, v] for t, v in pts], "T": _T}


def _resync_trend_windows(job, built: dict, slice_map: dict, *, ctx: RunContext) -> list[str]:
    """Rebuild any trend probe whose window is not the case's settled answer window.

    Returns the rebuilt case ids (logged). A no-op when the probes were assigned with
    `answer_windows` on the same slicing.
    """
    fixed: list[str] = []
    for cid, pr in list(ctx.quant_for.items()):
        if not (pr.get("kind") == "trend" or pr.get("fallback_from") == "trend"):
            continue
        raw = built.get(cid)
        if raw is None:
            continue
        T = int((raw.prediction_context or {}).get("prediction_time_T", 0) or 0)
        sl = slice_map.get(cid) if getattr(job, "slices", "") else None
        geom = claim_geometry(job, len(sl) if sl is not None else None)
        want = int(max(sl)) if (geom == "slices" and sl) else T
        if int(pr.get("T", -1)) == want:
            continue
        _avoid = str((ctx.noop_for.get(cid) or {}).get("target") or "") or None
        new = build_quant_probe(raw, "trend", want, avoid_signal=_avoid)
        if new:
            ctx.quant_for[cid] = new
        else:
            ctx.quant_for.pop(cid, None)
        fixed.append(str(cid))
    if fixed:
        log.warning("[quant] %d trend question(s) rebuilt on the settled answer window: %s",
                    len(fixed), fixed[:10])
    return fixed


def assign_qside_probes(built: dict, answer_t: dict | None = None, *,
                        ctx: RunContext) -> tuple[dict, dict]:
    """Assign the noop and quant probes together, the quant probe avoiding the noop stream.

    `run` and `tools/recompute_judges.py` both call this, so a recompute rebuilds the same
    questions the run asked (it used to call `assign_quant_probes` without `avoid`).
    """
    npr = assign_noop_probes(built, ctx=ctx)
    qpr = assign_quant_probes(built, avoid=npr, answer_t=answer_t, ctx=ctx)
    return npr, qpr


def assign_quant_probes(built: dict, avoid: dict | None = None,
                        answer_t: dict | None = None, *, ctx: RunContext) -> dict:
    """Assign one quant probe per case (deterministic rotation), avoiding the stream the noop
    probe asks about (`avoid`).

    `answer_t` maps case -> last answered day (`answer_windows`). `trend` questions are built
    on that window, so the gold equals the judge's recomputation on the answered slice; the
    other kinds keep `prediction_time_T`.

    Records each kind's ground-truth distribution in `ctx.quant_truth_dist` and warns when one
    answer dominates.
    """
    from collections import Counter

    ctx.quant_for.clear()
    _av = dict(avoid or {})
    _at = dict(answer_t or {})
    for cid, raw in sorted(built.items()):
        k = _QUANT_KINDS[int(rng.unit(cid, "quant", "kind") * len(_QUANT_KINDS))]
        _skip = str((_av.get(cid) or {}).get("target") or "")
        _T = int((raw.prediction_context or {}).get("prediction_time_T", 0)) or None
        if k == "trend" and _at.get(str(cid)) is not None:
            _T = int(_at[str(cid)])
        pr = build_quant_probe(raw, k, _T, avoid_signal=_skip or None)
        if pr:
            ctx.quant_for[cid] = pr
    ctx.quant_truth_dist.clear()
    _by: dict[str, Counter] = {}
    for _pr in ctx.quant_for.values():
        _by.setdefault(str(_pr.get("kind") or "?"), Counter())[str(_pr.get("truth"))] += 1
    for _k, _c in sorted(_by.items()):
        _n0 = sum(_c.values())
        _t0, _tn0 = _c.most_common(1)[0]
        ctx.quant_truth_dist[_k] = {"n": _n0, "n_distinct": len(_c), "top": _t0,
                                "top_share": round(_tn0 / _n0, 4) if _n0 else None,
                                "dist": dict(_c)}
    for _k, _c in sorted(_by.items()):
        _n = sum(_c.values())
        _top, _tn = _c.most_common(1)[0]
        _share = _tn / _n if _n else 0.0
        _fn = log.warning if _share > 0.60 else log.info
        _fn("[quant] %s: %d question(s) · %d distinct truth value(s) · largest class %r is %.1f%%%s",
            _k, _n, len(_c), _top, _share * 100,
            "  (always answering the largest class earns this score; this "
            "question type has limited discriminative power on this batch)"
            if _share > 0.60 else "")
    return dict(ctx.quant_for)


def _quant_suffix(payload, ctx: RunContext) -> str:
    cid = str(getattr(payload, "case_id", "")
              or (getattr(payload, "prediction_context", {}) or {}).get("case_id", ""))
    pr = ctx.quant_for.get(cid)
    if not pr:
        return ""
    return ("\n\n【数据核对题】" + str(pr["text"])
            + '\n(把答案写进 JSON 的 `"quant_answer"` 字段,例如 `"quant_answer": "rising"` '
              '或 `"quant_answer": 42`。**这一题只考你有没有读准数据,与诊断无关。**)')


def render_for(solver, payload) -> tuple[str, str]:
    """Render the framing for this solver's cell via the probe layer; returns (text, probe_id).
    The probes and Q-side tables come from the run the solver is bound to
    (`solver.run_ctx`); without loaded probes, falls back to `DEFAULT_PROBE`.
    """
    mode = getattr(solver, "prompt_mode", "default")
    pid = resolve_probe(solver, mode)
    ctx = run_ctx_of(solver)
    _suf = (_premise_suffix(payload, ctx) + _noop_suffix(payload, ctx)
            + _quant_suffix(payload, ctx) + _gated_suffix(solver))
    if ctx.probes:
        return render_with_probe(payload, pid, ctx.probes,
                                 allow_retired=ctx.allow_retired) + _suf, pid
    return _render_prompt(payload, mode) + _suf, pid
