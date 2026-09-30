"""Multi-model evaluation: one cell per (case x solver), each through the same
isolation chain and hard gates:
  build_instance(<=T) -> leakage_probe -> solver.solve -> verifier.grade
Rows are appended to JSONL as cells finish; a rerun only evaluates cells not yet done.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml

from . import mount_table as _MT
from .run_scheduler import stall_guard as _stall_guard

from dataclasses import dataclass
import hashlib
import json
import logging
import os
import subprocess
import threading as _threading
import time
from pathlib import Path

import verifier as verifier_mod                    # kernel
import runner as runner_mod
# The leakage gate runs inside `solve_guard.guarded_solve`, shared by every geometry.
from build import build_instance
from schema import SolverOutput
from solver import Solver, BaselineSolver, RobustSolver, ALLOWED_DRIVERS, ACTION_CLASSES, _extract_json

from . import judges as judges_mod, llm as _llm, process, wq
from .judges.trajectory import NOOP_CONTRACT as _NOOP_CONTRACT

log = logging.getLogger("haenv.eval")

@dataclass
class RunState:
    """What the command line and `run_eval` set for the run in this process; the cell code
    reads it. `run_eval` overwrites every field but `workers` at the start of each run, so
    nothing carries over from one batch to the next."""
    resp_path: Path | None = None    # this batch's responses.jsonl
    probes: dict | None = None       # probe registry; None falls back to the module constant
    allow_retired: bool = False      # a retired probe may be used (reproducing history only; --allow-retired)
    replay: bool = False             # a resumed run replays saved slice answers
    replay_idx: dict | None = None   # (case, solver, slice_t) -> first reusable saved row
    workers: int = 1                 # cells in flight; 1 = serial (set by the CLI from --workers / config)


RUN = RunState()

# `PROMPT` lives in `prompts.py` so that `build` can import it without an import cycle.
from .prompts import PROMPT  # noqa: F401  (re-export: `evaluate.PROMPT` keeps working)
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data


from .framings import (  # noqa: F401
    DDX_PROMPT, DDX_SCOPE2_PROMPT, DDX_SCOPE_PROMPT, DDX_TRACE_PROMPT,
    _SCOPE2_EDITS, _SCOPE_EDITS, derive_scope2_prompt, derive_scope_prompt,
)


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


def load_env_file(path: str | None) -> dict[str, str]:
    """Read `config.env_file` (`export KEY=...` lines) into a dict for dispatched subprocesses.

    Values are never logged; a missing file returns `{}`. `$VAR` is expanded against variables
    assigned earlier in the same file (shell order); no other shell syntax is supported.
    """
    if not path:
        return {}
    path = os.path.expandvars(str(path))
    if not path.strip() or "$" in path:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        log.warning("[eval] env_file not found: %s (skipped; the backend's own "
                    "configuration takes over)", p)
        return {}
    import re
    _VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
    out: dict[str, str] = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s[7:].lstrip() if s.startswith("export ") else s
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            v = _VAR_RE.sub(lambda m: out.get(m.group(1), m.group(0)), v)
            if k.strip():
                out[k.strip()] = v
    except OSError as e:
        log.warning("[eval] failed to read env_file %s: %s", p, e)
        return {}
    log.info("[eval] loaded from %s: %d environment variable(s) (values are not logged)", p, len(out))
    return out


def _render_prompt(payload, mode: str = "default") -> str:
    """Render the default or the ddx framing (the default PROMPT has no slot for a diagnosis).
    Every model on a question type sees the same text.
    """
    tpl = DDX_PROMPT if mode == "ddx" else PROMPT
    return tpl.format(target=payload.prediction_context.get("target_event_type"),
                      drivers=", ".join(ALLOWED_DRIVERS), actions=ACTION_CLASSES,
                      payload=payload.dumps())


# ---------------------------------------------------------------- D_probe (framing difficulty)
from .framings import (  # noqa: F401
    _BUILTIN_FRAMING_NAMES, _framings, framing_sha256,
)
_FRAMINGS = _framings()                                   # registry; render_with_probe computes it fresh each time


def load_probes(root) -> dict:
    """Load `probes/*.yaml` (default-deny).

    Each probe's `framing_sha256` must match its template, so a template edit cannot silently
    change both arms of an experiment.
    """
    import yaml
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
    from .external_gold import external_probes as _ext_probes
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
PREMISE_FOR: dict[str, dict] = {}
PREMISE_RATIO = (3, 2)          # false : true


def assign_premises(built: dict, ratio: tuple[int, int] = PREMISE_RATIO) -> dict:
    from . import rng
    PREMISE_FOR.clear()
    nf, nt = ratio
    for cid, raw in sorted(built.items()):
        k = "false" if int(rng.unit(cid, "premise", "polarity") * (nf + nt)) < nf else "true"
        prem = build_premise(raw, k)
        if prem:
            PREMISE_FOR[cid] = prem
    return dict(PREMISE_FOR)


def _premise_suffix(payload) -> str:
    """Sentence appended to the question, outside the framing template so that
    `framing_sha256` is unaffected.
    """
    prem = PREMISE_FOR.get(str(getattr(payload, "case_id", "")
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
NOOP_FOR: dict[str, dict] = {}
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


def assign_noop_probes(built: dict, ratio: tuple[int, int] = NOOP_RATIO) -> dict:
    from . import rng
    NOOP_FOR.clear()
    na, np_ = ratio
    for cid, raw in sorted(built.items()):
        k = "absent" if int(rng.unit(cid, "noop", "polarity") * (na + np_)) < na else "present"
        pr = build_noop_probe(raw, k, int((raw.prediction_context or {}).get(
            "prediction_time_T", 0)) or None)
        if pr:
            NOOP_FOR[cid] = pr
    return dict(NOOP_FOR)


def _noop_suffix(payload) -> str:
    cid = str(getattr(payload, "case_id", "")
              or (getattr(payload, "prediction_context", {}) or {}).get("case_id", ""))
    pr = NOOP_FOR.get(cid)
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
QUANT_FOR: dict[str, dict] = {}
#: Batch-level ground-truth distribution per quant kind
#: (`{kind: {n, n_distinct, top, top_share, dist}}`); `top_share` near 1.0 means guessing
#: the majority answer scores well.
QUANT_TRUTH_DIST: dict[str, dict] = {}

#: Per-case gold test list, read only by `OracleTestsSolver`; never rendered into a question.
ORACLE_GOLD_TESTS: dict[str, list[str]] = {}

#: Per-case discriminator names, read only by `OneLongTestSolver`; never rendered.
ORACLE_DISC_NAMES: dict[str, list[str]] = {}

#: Per-case gold diagnosis name, read only by `LateConvergeSolver` (gives `converged_at` a
#: measurable range); never rendered.
ORACLE_DIAGNOSIS: dict[str, str] = {}


def assign_oracle_gold(built: dict) -> dict:
    from build import build_instance                     # kernel
    ORACLE_GOLD_TESTS.clear()
    ORACLE_DISC_NAMES.clear()
    ORACLE_DIAGNOSIS.clear()
    for cid, raw in sorted(built.items()):
        try:
            _, vp = build_instance(raw, int(raw.prediction_context["prediction_time_T"]))
            ddx = (getattr(vp, "adjudication", None) or {}).get("ddx") or {}
            ORACLE_GOLD_TESTS[str(cid)] = [str(t) for t in (ddx.get("tests") or [])
                                           if str(t).strip()]
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
            ORACLE_DISC_NAMES[str(cid)] = _names
            if str(ddx.get("diagnosis") or "").strip():
                ORACLE_DIAGNOSIS[str(cid)] = str(ddx["diagnosis"]).strip()
        except Exception:                                # noqa: BLE001
            continue
    return dict(ORACLE_GOLD_TESTS)


#: Per-case gold `clinician_action_warranted`, read only by the `oracle_review_*` stubs.
ORACLE_WARRANTED: dict[str, bool] = {}


def assign_oracle_warranted(built: dict) -> dict:
    """Register each case's `clinician_action_warranted` for the `oracle_review_*` stubs, which
    give `review_macro` (specificity; 0.000 for "always says so") an offline range.
    """
    from build import build_instance                     # kernel
    ORACLE_WARRANTED.clear()
    for cid, raw in sorted(built.items()):
        try:
            _, vp = build_instance(raw, int(raw.prediction_context["prediction_time_T"]))
            adj = getattr(vp, "adjudication", None) or {}
            if "clinician_action_warranted" in adj:
                ORACLE_WARRANTED[str(cid)] = bool(adj["clinician_action_warranted"])
        except Exception:                                # noqa: BLE001
            continue
    return dict(ORACLE_WARRANTED)
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
    from .judges import _quant_truth

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
            from .judges import QUANT_PCT_INDEX_DIVISOR as _PDIV
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
        from .judges import QUANT_TREND_DEADZONE
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
        from .judges import QUANT_PCT_INDEX_DIVISOR as _PDIV
        vs = sorted(float(p["value"]) for p in pts)
        _k = len(vs) // _PDIV
        lo, hi = vs[_k], vs[-1 - _k]
        truth = sum(1 for p in pts if not (lo <= float(p["value"]) <= hi))
        q = (f"`{sig}` 有多少天的读数 **严格小于 {lo:g} 或严格大于 {hi:g}**?"
             f"(只数你能看到的那些天)只回答一个整数。")
    return {"kind": kind, "signal": sig, "truth": truth, "text": q,
            "series": [[int(p["ts"]), float(p["value"])] for p in pts],
            "T": _T}


def answer_windows(job, built: dict, batch_dir=None) -> dict[str, int]:
    """The last day each case is answered on: the last slice under the slice geometry, else
    `prediction_time_T`. The single source for the trend question's window.

    Slices come from the batch's frozen slicing table when there is one (the geometry the
    rows ran under), else from `slices_for`. A slicing table that lacks a case falls back to
    `slices_for` here; `run_eval` raises on that case and re-checks the window afterwards.
    """
    from .mount_table import claim_geometry
    from . import slicing as _slicing
    out: dict[str, int] = {}
    spec = str(getattr(job, "slices", "") or "") if job is not None else ""
    _frozen = None
    if spec and batch_dir is not None:
        try:
            _frozen = _slicing.load(batch_dir)
        except Exception:                                 # noqa: BLE001
            _frozen = None
    for cid, raw in built.items():
        T = int((raw.prediction_context or {}).get("prediction_time_T", 0) or 0)
        sl = None
        if spec:
            sl = (_frozen[0].get(str(cid)) if _frozen is not None else None)
            if sl is None:
                sl = slices_for(raw, spec)
        geom = claim_geometry(job, len(sl) if sl is not None else None) if job is not None else "single"
        out[str(cid)] = int(max(sl)) if (geom == "slices" and sl) else T
    return out


def quant_probe_for(raw, kind: str, signal: str, T: int | None = None) -> dict | None:
    """The quant probe for a recorded `(kind, signal)`, in `build_quant_probe`'s format, with
    points at <=T (default `prediction_time_T`). None when the series has fewer than four such
    points. Used to judge an existing answer on the question its row records."""
    from .judges import _quant_truth
    _T = int(T if T is not None else (raw.prediction_context or {}).get("prediction_time_T", 10**9))
    pts = sorted((int(p["ts"]), float(p["value"])) for p in ((raw.longitudinal_data or {}).get(signal) or [])
                 if isinstance(p, dict) and "value" in p and "ts" in p and int(p["ts"]) <= _T)
    if len(pts) < 4 or kind not in ("trend", "peak_day", "peak_value", "abnormal_days"):
        return None
    return {"kind": kind, "signal": signal, "truth": _quant_truth(kind, pts), "text": "",
            "series": [[t, v] for t, v in pts], "T": _T}


def _resync_trend_windows(job, built: dict, slice_map: dict) -> list[str]:
    """Rebuild any trend probe whose window is not the case's settled answer window.

    Returns the rebuilt case ids (logged). A no-op when the probes were assigned with
    `answer_windows` on the same slicing.
    """
    from .mount_table import claim_geometry
    fixed: list[str] = []
    for cid, pr in list(QUANT_FOR.items()):
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
        _avoid = str((NOOP_FOR.get(cid) or {}).get("target") or "") or None
        new = build_quant_probe(raw, "trend", want, avoid_signal=_avoid)
        if new:
            QUANT_FOR[cid] = new
        else:
            QUANT_FOR.pop(cid, None)
        fixed.append(str(cid))
    if fixed:
        log.warning("[quant] %d trend question(s) rebuilt on the settled answer window: %s",
                    len(fixed), fixed[:10])
    return fixed


def assign_qside_probes(built: dict, answer_t: dict | None = None) -> tuple[dict, dict]:
    """Assign the noop and quant probes together, the quant probe avoiding the noop stream.

    `run` and `tools/recompute_judges.py` both call this, so a recompute rebuilds the same
    questions the run asked (it used to call `assign_quant_probes` without `avoid`).
    """
    npr = assign_noop_probes(built)
    qpr = assign_quant_probes(built, avoid=npr, answer_t=answer_t)
    return npr, qpr


def assign_quant_probes(built: dict, avoid: dict | None = None,
                        answer_t: dict | None = None) -> dict:
    """Assign one quant probe per case (deterministic rotation), avoiding the stream the noop
    probe asks about (`avoid`).

    `answer_t` maps case -> last answered day (`answer_windows`). `trend` questions are built
    on that window, so the gold equals the judge's recomputation on the answered slice; the
    other kinds keep `prediction_time_T`.

    Records each kind's ground-truth distribution in `QUANT_TRUTH_DIST` and warns when one
    answer dominates.
    """
    from collections import Counter

    from . import rng
    QUANT_FOR.clear()
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
            QUANT_FOR[cid] = pr
    QUANT_TRUTH_DIST.clear()
    _by: dict[str, Counter] = {}
    for _pr in QUANT_FOR.values():
        _by.setdefault(str(_pr.get("kind") or "?"), Counter())[str(_pr.get("truth"))] += 1
    for _k, _c in sorted(_by.items()):
        _n0 = sum(_c.values())
        _t0, _tn0 = _c.most_common(1)[0]
        QUANT_TRUTH_DIST[_k] = {"n": _n0, "n_distinct": len(_c), "top": _t0,
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
    return dict(QUANT_FOR)


def _quant_suffix(payload) -> str:
    cid = str(getattr(payload, "case_id", "")
              or (getattr(payload, "prediction_context", {}) or {}).get("case_id", ""))
    pr = QUANT_FOR.get(cid)
    if not pr:
        return ""
    return ("\n\n【数据核对题】" + str(pr["text"])
            + '\n(把答案写进 JSON 的 `"quant_answer"` 字段,例如 `"quant_answer": "rising"` '
              '或 `"quant_answer": 42`。**这一题只考你有没有读准数据,与诊断无关。**)')


def render_for(solver, payload) -> tuple[str, str]:
    """Render the framing for this solver's cell via the probe layer; returns (text, probe_id).
    Without loaded probes, falls back to `DEFAULT_PROBE`.
    """
    mode = getattr(solver, "prompt_mode", "default")
    pid = resolve_probe(solver, mode)
    _suf = (_premise_suffix(payload) + _noop_suffix(payload)
            + _quant_suffix(payload) + _gated_suffix(solver))
    if RUN.probes:
        return render_with_probe(payload, pid, RUN.probes,
                                 allow_retired=RUN.allow_retired) + _suf, pid
    return _render_prompt(payload, mode) + _suf, pid


def _dq_hashable(dq):
    """Coerce `data_quality.signal_quality` values to strings.

    The kernel gate `acted_on_unverified_signal` tests `sq.get(sig) in {...}`, which raises
    on a dict value. Non-string values become their JSON text, which never matches, so
    verdicts are unchanged.
    """
    if not isinstance(dq, dict):
        # `None`, not `"insufficient_data"`: a non-object `data_quality` is not a declaration of
        # insufficiency, which would exempt the answer from `acted_on_unverified_signal`.
        return {"data_sufficiency": None, "signal_quality": {}}
    sq = dq.get("signal_quality")
    if not isinstance(sq, dict):
        return dq
    import json as _json
    fixed = {}
    for k, v in sq.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            fixed[k] = v
        else:                                    # dict / list => coerce to text (still never matches a flag word)
            try:
                fixed[k] = _json.dumps(v, ensure_ascii=False)
            except Exception:                    # noqa: BLE001
                fixed[k] = str(v)
    out = dict(dq)
    out["signal_quality"] = fixed
    return out


def _to_output(data: dict, payload) -> SolverOutput:
    """Collapse the model's JSON into a `SolverOutput`; never crashes, never fabricates.

    Fields outside the kernel signature (differential, join_type, tests_to_order, ...) are kept
    on `_raw`, which the judges read. When the model answered but omitted `action` or
    `data_quality`, `clinician_review_required` and `data_sufficiency` stay `None`: a canned
    `True` / `"insufficient_data"` would assert a safety flag or grant a gate exemption the
    model never gave. Only a parse failure (`data == {}`) gets the full canned abstention.
    """
    # A field of the wrong JSON type (e.g. `"action": "refer to endocrinology"`) is treated as
    # omitted, so the rest of the answer is still graded instead of the cell crashing.
    if isinstance(data, dict):
        data = dict(data)
        for _k in ("forecast", "action", "data_quality"):
            if _k in data and not isinstance(data[_k], dict):
                data.pop(_k)
        if "drivers" in data:
            _dr = data["drivers"]
            data["drivers"] = ([d for d in _dr if isinstance(d, dict)]
                               if isinstance(_dr, list) else [])
    f = data.get("forecast", {}) or {}
    cited = data.get("cited_evidence") or f.get("key_predictive_evidence") or []
    _answered = isinstance(data, dict) and bool(data)
    out = SolverOutput(
        forecast={"target_event": f.get("target_event", payload.prediction_context.get("target_event_type")),
                  "risk": float(f.get("risk", 0.5) or 0.5),
                  "risk_category": f.get("risk_category", "indeterminate"),
                  "confidence": float(f.get("confidence", 0.3) or 0.3),
                  "key_predictive_evidence": f.get("key_predictive_evidence", [])},
        drivers=data.get("drivers", []),
        action=data.get("action", {"selected_action_class": "A1",
                                   "specific_action": "collect more data (parse-fail abstain)",
                                   "what_not_to_do": [],
                                   "clinician_review_required": (None if _answered else True),
                                   "followup_interval": "7d"}),
        data_quality=_dq_hashable(data.get(
            "data_quality", {"data_sufficiency": (None if _answered else "insufficient_data"),
                             "signal_quality": {}})),
        cited_evidence=cited)
    out._raw = data if isinstance(data, dict) else {}     # noqa: SLF001 -- see docstring
    # `SolverOutput` has no `notes` field, but `runner.run_multiround` reads `out.notes` and
    # `_premise_suffix` asks the model to use it; normalized to `str`.
    _n = data.get("notes") if isinstance(data, dict) else None
    out.notes = (_n.strip() if isinstance(_n, str)
                 else " ".join(str(x).strip() for x in _n if str(x).strip()) if isinstance(_n, list)
                 else "")
    return out


class CLISolverBlocked(RuntimeError):
    """The CLI subprocess path refuses to start by default; see `CLISolver`."""


_CLI_SANDBOX_CWD: str | None = None


def _cli_cwd() -> str:
    global _CLI_SANDBOX_CWD
    if _CLI_SANDBOX_CWD is None:
        import tempfile
        _CLI_SANDBOX_CWD = tempfile.mkdtemp(prefix="haenv-cli-cwd-")
    return _CLI_SANDBOX_CWD


def secret_env_names(env: dict) -> list[str]:
    """Environment variable names that look like credentials. Returns names
    only, never values."""
    import re
    pat = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL", re.I)
    return sorted(k for k in (env or {}) if pat.search(str(k)))


class CLISolver(Solver):
    """Dispatches to a real model via `~/.local/bin/ai`; a parse failure degrades to abstention.

    Refuses to start unless `allow_cli_solver: true`: an agentic CLI with file tools could
    read ground-truth files. When allowed, the subprocess runs from an empty directory outside
    the repo, but env and `$HOME` (needed for credentials) are not contained; the names of
    credential-shaped variables are logged.
    """

    def __init__(self, name: str, ai_args: list[str], ai: str, timeout: int, retries: int = 2,
                 env: dict[str, str] | None = None, allow_unsandboxed: bool = False):
        if not allow_unsandboxed:
            raise CLISolverBlocked(
                f"model {name!r} takes the CLI-subprocess path, which refuses to start by "
                f"default: an agentic CLI with file tools given the repo root as cwd and the "
                f"full env is a leak risk, and containment of env/HOME is still unresolved. "
                f"To allow it anyway: set `allow_cli_solver: true` at the "
                f"top level of config.yaml, knowing that the residual exposure will be logged "
                f"(see evaluate.CLISolver's docstring).")
        self.name, self.ai_args, self.ai, self.timeout, self.retries = name, ai_args, ai, timeout, retries
        self.env = {**os.environ, **(env or {})}
        _names = secret_env_names(self.env)
        log.warning("[solver %s] CLI path allowed: cwd=%s (outside the repo); "
                    "residual exposure: %d credential-shaped env var(s) %s; "
                    "$HOME is not isolated (transcripts contain ground truth)",
                    name, _cli_cwd(), len(_names), _names)

    def solve(self, payload):
        prompt, pid = render_for(self, payload)
        data, raw_text = {}, ""
        for attempt in range(1, self.retries + 2):
            try:
                proc = subprocess.run([self.ai, *self.ai_args], input=prompt,
                                      capture_output=True, text=True, timeout=self.timeout,
                                      env=self.env, cwd=_cli_cwd())
                raw_text = proc.stdout
                data = _extract_json(raw_text)
                break
            except Exception as e:
                log.warning("[solver %s] attempt %d failed: %s", self.name, attempt, e)
                time.sleep(2 * attempt)
        out = _to_output(data, payload)
        out._raw_text, out._prompt_mode = raw_text, getattr(self, "prompt_mode", "default")
        out._probe_id, out._prompt_sha = pid, hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return out


# ============================================================ backend registry (default-deny)
#
# The relay host must be in `no_proxy`. A liveness probe must send a real-scale request: the
# relay accepts a tiny request on a nearly exhausted key.
BACKENDS: dict[str, dict] = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions",
                   "key_env": "OPENROUTER_API_KEY", "kind": "openai_compat"},
    "openai":     {"url": "https://api.openai.com/v1/chat/completions",
                   "key_env": "OPENAI_API_KEY", "kind": "openai_compat",
                   # OpenAI's newer models only accept max_completion_tokens,
                   # not max_tokens
                   "max_tokens_field": "max_completion_tokens"},
    # An OpenAI-compatible relay. Its url and key names are deployment-specific and come from
    # `config.backends.relay` (see `register_backends`); until a url is configured, its models are
    # skipped. The relay reports an exhausted key as 401, so keys are pooled by the
    # `key_env_pool` prefix and rotated (see `_KeyPool`).
    "relay":      {"url": None, "key_env": "RELAY_API_KEY", "key_env_pool": "RELAY_KEY_",
                   "kind": "openai_compat"},
    "dashscope":  {"url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                   "key_env": "DASHSCOPE_API_KEY", "kind": "openai_compat",
                   "no_proxy_host": "dashscope.aliyuncs.com"},
    "google":     {"url": "https://generativelanguage.googleapis.com/v1beta/models",
                   "key_env": "GOOGLE_GENERATIVE_AI_API_KEY", "kind": "google"},
}


#: Fields a config entry may set on a backend; anything else is rejected.
_BACKEND_FIELDS = ("url", "key_env", "key_env_pool", "kind", "no_proxy_host",
                   "max_tokens_field", "quota_url")


def register_backends(cfg: dict | None) -> None:
    """Apply `config.backends.<name>` over the built-in table.

    Used for deployment-specific endpoints (the relay's url, key names, billing
    endpoint). Entries override field by field; a new name registers a new
    `openai_compat` backend. Unknown fields raise.
    """
    for name, over in ((cfg or {}).get("backends") or {}).items():
        if not isinstance(over, dict):
            raise ValueError(f"config.backends.{name} must be a mapping")
        unknown = sorted(set(over) - set(_BACKEND_FIELDS))
        if unknown:
            raise ValueError(f"config.backends.{name}: unknown field(s) {unknown}; "
                             f"allowed: {list(_BACKEND_FIELDS)}")
        base = dict(BACKENDS.get(name) or {"kind": "openai_compat"})
        base.update({k: v for k, v in over.items() if v is not None})
        BACKENDS[str(name)] = base


_BACKENDS_FROM_CFG = False


def _ensure_backends_registered(cfg: dict | None = None) -> None:
    global _BACKENDS_FROM_CFG
    if _BACKENDS_FROM_CFG:
        return
    if cfg is None:
        try:
            from .cli import load_cfg
            cfg = load_cfg()
        except FileNotFoundError:
            cfg = {}
    register_backends(cfg)
    _BACKENDS_FROM_CFG = True


# ---------------------------------------------------------------- key pool
#
# Rotation happens only when "quota exhausted" is recognized (never on timeouts, rate limits
# or server errors), does not consume the cell's retry budget, and marks the key dead for
# the process lifetime. An empty pool falls through to ABORT(no_response).
_EXHAUSTED_MARKERS = ("quota is exhausted", "TokenStatusExhausted", "quota is not enough")


def _is_exhausted(err_body: str) -> bool:
    return any(m.lower() in (err_body or "").lower() for m in _EXHAUSTED_MARKERS)


class _KeyPool:

    def __init__(self, keys: list[str], backend: str, ordinals: list | None = None):
        self.backend = backend
        # Key ordinals (the numeric suffix of the env name); request accounting logs and
        # disables keys by ordinal, never by value.
        pairs = [(k, o) for k, o in zip(keys, ordinals or [None] * len(keys)) if k]
        self._keys = [k for k, _ in pairs]
        self.ordinals = [o if o is not None else i for i, (_, o) in enumerate(pairs)]
        self._dead: set[str] = set()
        self._rr = 0          # round-robin cursor
        self._lock = _threading.Lock()

    def current(self) -> str:
        """Return the next usable key, round-robin, so concurrent requests use distinct keys and
        avoid per-key rate limits.
        """
        with self._lock:
            n = len(self._keys)
            for _ in range(n):
                k = self._keys[self._rr % n]
                self._rr += 1
                if k not in self._dead:
                    return k
        return ""

    def mark_dead(self, key: str) -> bool:
        with self._lock:
            if key and key not in self._dead:
                self._dead.add(key)
                log.warning("[keypool %s] a key ran out of quota (%d/%d dead) -> rotating",
                            self.backend, len(self._dead), len(self._keys))
            return any(k not in self._dead for k in self._keys)

    @property
    def n_alive(self) -> int:
        return sum(1 for k in self._keys if k not in self._dead)


_KEY_POOLS: dict[str, _KeyPool] = {}


def _load_key_pool(bdef: dict, env: dict, backend: str) -> _KeyPool:
    """Collect keys by the `key_env_pool` prefix, sorted by numeric suffix; falls back to the
    single `key_env`.
    """
    prefix = bdef.get("key_env_pool")
    if not prefix:
        return _KeyPool([env.get(bdef["key_env"], "")], backend)

    def _idx(name: str) -> int:
        tail = name[len(prefix):]
        return int(tail) if tail.isdigit() else 10 ** 6

    names = sorted((n for n in env if n.startswith(prefix) and _idx(n) < 10 ** 6), key=_idx)
    keys = [env[n] for n in names if str(env[n]).startswith("sk-")]
    if not keys:
        keys = [env.get(bdef["key_env"], "")]
    log.info("[keypool %s] loaded %d key(s) (%s)", backend, len(keys),
             ", ".join(names) or bdef["key_env"])
    ordinals = [_idx(n) for n in names if str(env[n]).startswith("sk-")]
    return _KeyPool(keys, backend, ordinals if len(ordinals) == len(keys) else None)


class OpenAICompatSolver(Solver):
    """Connects directly to any OpenAI-compatible endpoint (OpenRouter / OpenAI / a relay /
    dashscope). The prompt and payload are identical across backends.
    """

    def __init__(self, name: str, model: str, api_key: str, timeout: int, retries: int = 2,
                 max_tokens: int = 6000, url: str = "", backend: str = "openrouter",
                 max_tokens_field: str = "max_tokens", stream: bool = False,
                 reasoning_effort: str | None = None, response_format: dict | None = None,
                 provider: dict | None = None):
        self.name, self.model, self.timeout, self.retries = name, model, timeout, retries
        # OpenRouter upstream routing (`provider` request field), e.g. {"only": ["Moonshot AI"],
        # "allow_fallbacks": False}. None = the router picks the upstream (the default).
        self.provider = dict(provider) if provider else None
        self.api_key, self.max_tokens = api_key, max_tokens
        self.pool: _KeyPool | None = None        # injected by build_solvers; None = the single-key path
        self.backend = backend
        self.URL = url or BACKENDS["openrouter"]["url"]
        self.max_tokens_field = max_tokens_field
        # `stream` changes only the transport (payload and max_tokens are identical); used for
        # models that hit the gateway's 300 s idle timeout.
        self.stream = bool(stream)
        if reasoning_effort is not None and reasoning_effort not in {
                "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unrecognized reasoning_effort")
        self.reasoning_effort = reasoning_effort
        self.response_format = response_format
        self._last_reasoning_chars = 0
        if not api_key:
            log.error("[solver %s] backend=%s is missing %s (check config.yaml:env_file)",
                      name, backend, BACKENDS.get(backend, {}).get("key_env", "?"))

    def _post(self, prompt: str) -> dict:
        accounting = getattr(self, "_accounting", None)
        resp = (accounting.request(self, prompt, self._post_wire) if accounting is not None
                else self._post_wire(prompt))
        if self.stream and accounting is not None:
            self._last_reasoning_chars = resp.get("_reasoning_chars", 0)
            self._last_reasoning_text = resp.get("_reasoning_text")
            choice = (resp.get("choices") or [{}])[0]
            answer = (choice.get("message") or {}).get("content") or ""
            self._validate_stream(answer, self._last_reasoning_chars, resp.get("usage"),
                                  choice.get("finish_reason"))
        # Cost-bearing error responses must be persisted and settled before the
        # answer retry loop sees them. No error-body token/cost is fabricated.
        if isinstance(resp, dict) and resp.get("error") and not resp.get("choices"):
            error = resp["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(f"error body with HTTP 200: {str(message)[:160]}")
        return resp

    def _post_wire(self, prompt: str, *, api_key: str | None = None) -> dict:
        """Send one request. Only a recognized "quota exhausted" body (both it and a bad key are 401)
        switches keys and resends, without consuming the cell's retry budget. `api_key` (set by
        request accounting) pins the key for this single attempt; the pool is then not used.
        """
        return self._send_wire(self._wire_body(prompt), prompt, api_key=api_key)

    def _wire_body(self, prompt: str) -> bytes:
        """The request body for `prompt` (what the model is sent and how it is asked)."""
        payload_d: dict = {"model": self.model, self.max_tokens_field: self.max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}
        if self.reasoning_effort is not None:
            if self.backend == "openrouter":
                payload_d["reasoning"] = {"effort": self.reasoning_effort}
            else:
                payload_d["reasoning_effort"] = self.reasoning_effort
        if self.provider is not None:
            payload_d["provider"] = dict(self.provider)
        if self.response_format is not None:
            payload_d["response_format"] = self.response_format
            if self.backend == "openrouter":
                payload_d["provider"] = {**payload_d.get("provider", {}), "require_parameters": True}
        if self.stream:
            payload_d["stream"] = True
            payload_d["stream_options"] = {"include_usage": True}
        _llm.note_payload(self.name, payload_d)
        return json.dumps(payload_d).encode()

    def _send_wire(self, body: bytes, prompt: str, *, api_key: str | None = None) -> dict:
        """Send `body` (transport only). Refuses a body that differs from the accounted one.
        `api_key` pins the key for this attempt (see `_post_wire`)."""
        import urllib.request
        import urllib.error
        from .transport import verify_wire
        verify_wire(self, body, prompt)
        while True:
            key = api_key or (self.pool.current() if self.pool else self.api_key)
            req = urllib.request.Request(
                self.URL, data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            # Transport progress, attached to any failure for the accounting receipt.
            wire = {"http_status": None, "response_started": False, "bytes_received": 0,
                    "generation_id": None}
            try:
                with _stall_guard(self) as _stall, \
                        urllib.request.urlopen(req, timeout=self.timeout) as r:
                    wire["http_status"], wire["response_started"] = getattr(r, "status", None), True
                    r = _stall.attach(r)
                    accounted = getattr(self, "_accounting", None) is not None
                    if self.stream:
                        _resp = self._read_sse(r, defer_validation=accounted, wire=wire)
                    else:
                        _raw = r.read()
                        wire["bytes_received"] = len(_raw)
                        _resp = json.loads(_raw)
                    if accounted and self.backend == "relay" and isinstance(_resp, dict):
                        _resp["_gateway_request_id"] = r.headers.get("X-Oneapi-Request-Id")
                # Recorded inline rather than via `llm.observe_post`, whose monkey-patched `_post` would be
                # shared by the shallow copies `_const` makes.
                _llm.note_call(self.name, self.backend, _resp if isinstance(_resp, dict) else None)
                # HTTP-200 error bodies are checked by _post after the raw
                # cost-bearing response has crossed the accounting boundary.
                return _resp
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")
                except Exception:
                    pass
                # Kept for the failed-attempt receipt (the body can be read only once).
                e.haenv_error_body = detail.replace(key, "[redacted]") if key else detail
                e.haenv_wire = {**wire, "http_status": e.code, "response_started": False}
                if not self.pool:
                    raise
                if _is_exhausted(detail) and self.pool.mark_dead(key):
                    continue                      # switch to the next key, resend the same request
                if _is_exhausted(detail):
                    raise RuntimeError(
                        f"relay key pool exhausted ({self.pool.n_alive} usable)") from e
                raise
            except Exception as e:
                try:
                    e.haenv_wire = dict(wire)
                except Exception:  # noqa: BLE001 -- an exception type without attributes
                    pass
                raise

    #: Only `content` is spliced into the answer; reasoning fields (e.g. `reasoning_content`) are
    #: kept apart, matching the non-streaming path, which returns `message.content` only.
    _SSE_ANSWER_FIELD = "content"
    _SSE_THINK_FIELDS = ("reasoning_content", "reasoning")

    def _read_sse(self, r, *, defer_validation: bool = False, wire: dict | None = None) -> dict:
        """Reconstruct an SSE stream into a response equivalent to the non-streaming one.

        Streaming keeps the connection active under the relay's 300 s idle timeout without
        changing the payload or `max_tokens`.
        """
        txt, think_n, usage, fin = [], 0, None, None
        generation_id = None                             # provider id; exact cost lookup if usage is lost
        think_txt: list[str] = []                        # raw reasoning text, never added to txt
        wire = wire if wire is not None else {}
        for line in r:
            wire["bytes_received"] = wire.get("bytes_received", 0) + len(line)
            s = line.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            s = s[5:].strip()
            if not s or s == "[DONE]":
                continue
            try:
                ch = json.loads(s)
            except Exception:                             # noqa: BLE001
                continue                                  # heartbeat/comment frame, skip
            if generation_id is None and isinstance(ch.get("id"), str):
                generation_id = ch["id"]
                wire["generation_id"] = generation_id
                from .paid_completion import note_generation
                note_generation(generation_id)          # durable before the stream can be cut
            if ch.get("usage"):
                usage = ch["usage"]                       # final frame (include_usage)
            c0 = (ch.get("choices") or [{}])[0]
            if c0.get("finish_reason"):
                fin = c0["finish_reason"]
            delta = c0.get("delta") or {}
            piece = delta.get(self._SSE_ANSWER_FIELD)
            if isinstance(piece, str):
                txt.append(piece)
            for f in self._SSE_THINK_FIELDS:              # collected into a separate column, never into the answer
                if isinstance(delta.get(f), str):
                    think_n += len(delta[f])
                    think_txt.append(delta[f])
        # The raw reasoning text goes to `trace.jsonl` only, never into the answer or
        # `responses.jsonl`, which the judges read.
        self._last_reasoning_chars = think_n
        self._last_reasoning_text = "".join(think_txt) or None
        answer = "".join(txt)
        if not defer_validation:
            self._validate_stream(answer, think_n, usage, fin)
        result = {"choices": [{"message": {"content": answer}, "finish_reason": fin}],
                  "usage": usage, "_reasoning_chars": think_n}
        if defer_validation:
            result["_reasoning_text"] = self._last_reasoning_text
            result["_generation_id"] = generation_id
        return result

    @staticmethod
    def _validate_stream(answer, think_n, usage, fin):
        # A truncated stream ends without an exception; no terminating frame means no answer.
        if fin is None:
            raise ValueError(
                f"流式响应未收到终止帧(finish_reason 缺失):作答 {len(answer)} 字符、"
                f"推理 {think_n} 字符、usage={'有' if usage else '无'} —— 判为流被截断")
        if not answer.strip() and think_n > 0:
            # `tools/attribute_failures.py` matches this text; change both together.
            raise ValueError(
                f"流式响应只有推理没有作答(推理 {think_n} 字符、作答 0 字符)——"
                f"推理不进作答,所以这一次视为没答上")

    def solve(self, payload):
        prompt, pid = render_for(self, payload)
        from .semantic_budget import BudgetExceeded
        data, raw_text, usage, fin = {}, "", None, None
        # Failed attempts are persisted with raw text and elapsed time; the successful call's
        # latency is recorded as `latency_s`.
        attempts: list[dict] = []
        for attempt in range(1, self.retries + 2):
            text = ""
            resp = None
            _t0 = time.time()
            try:
                resp = self._post(prompt)
                choice = (resp.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                usage, fin = resp.get("usage"), choice.get("finish_reason")
                if fin == "length" and not text.strip():
                    raise ValueError("响应被 max_tokens 截断且无内容")
                raw_text = text
                data = _extract_json(text)
                latency = round(time.time() - _t0, 2)
                break
            except BudgetExceeded:
                raise
            except Exception as e:                    # rate limit/timeout/truncation/non-JSON -> back off and retry
                _dt = round(time.time() - _t0, 2)
                log.warning("[solver %s] attempt %d failed after %.1fs: %s: %s",
                            self.name, attempt, _dt, type(e).__name__, str(e)[:160])
                # Usage of a failed attempt is recorded here or nowhere (`out._usage` covers only the
                # successful attempt). A network failure has no usage: `None`, not 0.
                _fu = resp.get("usage") if isinstance(resp, dict) else None
                attempts.append({"attempt": attempt, "error": f"{type(e).__name__}: {str(e)[:200]}",
                                 "elapsed_s": _dt,
                                 "finish_reason": fin, "n_chars": len(text),
                                 "usage": _fu,
                                 "billed": billed_tokens(_fu) if _fu else None,
                                 "raw": text,
                                 "head": text[:400], "tail": text[-400:] if len(text) > 400 else ""})
                time.sleep(3 * attempt)
        else:
            latency = None                            # every attempt failed => no successful latency to record
        out = _to_output(data, payload)
        out._raw_text, out._prompt_mode = raw_text, getattr(self, "prompt_mode", "default")
        out._probe_id, out._prompt_sha = pid, hashlib.sha256(prompt.encode()).hexdigest()[:16]
        # Reasoning tokens count against max_tokens; `usage` and `finish_reason` separate a
        # truncated answer from a wrong one.
        out._usage, out._finish = usage, fin
        out._max_tokens = self.max_tokens
        out._failed_attempts = attempts
        out._rate_limit_retries = _drain_backoff(self)
        out._latency_s = latency
        out._reasoning_chars = getattr(self, "_last_reasoning_chars", None) if self.stream else None
        out._reasoning_text = getattr(self, "_last_reasoning_text", None) if self.stream else None
        return out


def _drain_backoff(solver) -> list | None:
    """Rate-limit retries the request accountant made for this solve (None when there were none)."""
    drain = getattr(getattr(solver, "_accounting", None), "drain_backoff", None)
    return (drain() or None) if callable(drain) else None


def _google_usage(resp: dict) -> dict | None:
    """Google `usageMetadata` -> usage shaped like the OpenAI-compatible backends.

    `thoughtsTokenCount` is kept separately and the unaccounted residual is recorded, so
    thinking tokens are not dropped from cost estimates.
    """
    um = (resp or {}).get("usageMetadata") or {}
    if not um:
        return None
    _thoughts = um.get("thoughtsTokenCount")
    _tot = um.get("totalTokenCount")
    _pt, _ct = um.get("promptTokenCount"), um.get("candidatesTokenCount")
    return {"prompt_tokens": _pt,
            "prompt_tokens_details": {"cached_tokens": um.get("cachedContentTokenCount")},
            "completion_tokens": _ct,
            "total_tokens": _tot,
            "thoughts_tokens": _thoughts,
            "unaccounted_tokens": (
                _tot - (_pt or 0) - (_ct or 0) - (_thoughts or 0)
                if isinstance(_tot, int) else None)}


class GoogleSolver(Solver):
    """Connects directly to the Google Generative Language API (`generateContent`; the key goes
    in the query string).
    """

    def __init__(self, name: str, model: str, api_key: str, timeout: int, retries: int = 2,
                 max_tokens: int = 6000):
        self.name, self.model, self.timeout, self.retries = name, model, timeout, retries
        self.api_key, self.max_tokens = api_key, max_tokens
        self.backend, self.URL = "google", BACKENDS["google"]["url"]
        self.pool, self.stream = None, False
        self.reasoning_effort, self.response_format = None, None
        self.max_tokens_field = "maxOutputTokens"
        if not api_key:
            log.error("[solver %s] missing GOOGLE_GENERATIVE_AI_API_KEY", name)

    def _post(self, prompt: str) -> dict:
        accounting = getattr(self, "_accounting", None)
        return (accounting.request(self, prompt, self._post_wire) if accounting is not None
                else self._post_wire(prompt))

    def _post_wire(self, prompt: str) -> dict:
        return self._send_wire(self._wire_body(prompt), prompt)

    def _wire_body(self, prompt: str) -> bytes:
        payload_d = {"contents": [{"parts": [{"text": prompt}]}],
                     "generationConfig": {"maxOutputTokens": self.max_tokens}}
        _llm.note_payload(self.name, payload_d)     # same as `OpenAICompatSolver._post`
        return json.dumps(payload_d).encode()

    def _send_wire(self, body: bytes, prompt: str) -> dict:
        import urllib.request
        from .transport import verify_wire
        verify_wire(self, body, prompt)
        url = (f"{self.URL}/{self.model}:generateContent?key={self.api_key}")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with _stall_guard(self) as _stall, urllib.request.urlopen(req, timeout=self.timeout) as r:
            _stall.attach(r)
            _resp = json.loads(r.read())
        _llm.note_call(self.name, "google", _resp if isinstance(_resp, dict) else None)
        return _resp

    def solve(self, payload):
        prompt, pid = render_for(self, payload)
        from .semantic_budget import BudgetExceeded
        data, raw_text, usage, fin = {}, "", None, None
        attempts: list[dict] = []
        for attempt in range(1, self.retries + 2):
            text = ""
            try:
                resp = self._post(prompt)
                cand = (resp.get("candidates") or [{}])[0]
                parts = ((cand.get("content") or {}).get("parts") or [{}])
                text = "".join(p.get("text", "") for p in parts)
                fin = cand.get("finishReason")
                usage = _google_usage(resp)
                if fin == "MAX_TOKENS" and not text.strip():
                    raise ValueError("响应被 maxOutputTokens 截断且无内容")
                raw_text = text
                data = _extract_json(text)
                break
            except BudgetExceeded:
                raise
            except Exception as e:
                log.warning("[solver %s] attempt %d failed: %s: %s",
                            self.name, attempt, type(e).__name__, str(e)[:160])
                attempts.append({"attempt": attempt, "error": f"{type(e).__name__}: {str(e)[:200]}",
                                 "finish_reason": fin, "n_chars": len(text),
                                 "head": text[:400], "tail": text[-400:] if len(text) > 400 else ""})
                time.sleep(3 * attempt)
        out = _to_output(data, payload)
        out._raw_text, out._prompt_mode = raw_text, getattr(self, "prompt_mode", "default")
        out._probe_id, out._prompt_sha = pid, hashlib.sha256(prompt.encode()).hexdigest()[:16]
        out._usage, out._finish = usage, fin
        out._max_tokens = self.max_tokens          # same as `OpenAICompatSolver.solve`
        out._failed_attempts = attempts
        out._rate_limit_retries = _drain_backoff(self)
        return out


def _const(inst):
    """Factory for a stateless solver: one shallow copy per case.

    `_one` and `run_gated` write per-case attributes (`probe_id`, `gated_context`, ...), so a
    shared instance would leak state between cases under `--workers > 1`. Solvers hold only
    scalars, and key pools live in `_KEY_POOLS`, so a shallow copy is safe. Stateful stubs are
    constructed per case in `build_solvers` instead.
    """
    import copy as _copy
    return lambda: _copy.copy(inst)


def _ensure_no_proxy(hosts: set[str]) -> None:
    """Append `hosts` to this process's `no_proxy` (the relay and dashscope fail through a local
    proxy). Never overwrites the user's setting or writes to disk.
    """
    if not hosts:
        return
    for key in ("no_proxy", "NO_PROXY"):
        cur = {h.strip() for h in (os.environ.get(key, "") or "").split(",") if h.strip()}
        missing = hosts - cur
        if missing:
            os.environ[key] = ",".join(sorted(cur | hosts))
    if hosts:
        log.info("[eval] no_proxy already includes %s", sorted(hosts))


def solver_for_spec(name: str, spec: dict, env: dict, timeout: int, retries: int):
    """Build a solver from the mapped form of `config.models[name]`; the one constructor shared
    by batch runs and tooling. An unregistered backend returns None.
    """
    backend = spec.get("backend", "openrouter")
    # Google-family models (by `vendor`, not by name) must connect directly: an
    # OpenAI-compatible relay drops `thought_signature`, and requests with tools then fail with 400.
    if str(spec.get("vendor") or "").lower() == "google" and backend != "google":
        raise ValueError(
            f"config.models[{name}]: vendor=google but backend={backend!r}. "
            f"Google-family models must connect directly (`backend: google`): an "
            f"OpenAI-compatible relay drops `thought_signature`, and requests with tools "
            f"fail with HTTP 400. Matched on `vendor`, not on the model name.")
    _ensure_backends_registered()
    bdef = BACKENDS.get(backend)
    if bdef is None:
        log.warning("[eval] model %s's backend=%s is not registered (default-deny, see "
                    "evaluate.BACKENDS), skipping",
                    name, backend)
        return None
    if bdef.get("kind") != "google" and not bdef.get("url"):
        log.warning("[eval] model %s: backend %s has no url configured "
                    "(set config.backends.%s.url, e.g. in config.local.yaml); skipped",
                    name, backend, backend)
        return None
    if spec.get("provider") is not None and (backend != "openrouter"
                                             or not isinstance(spec["provider"], dict)):
        raise ValueError(f"config.models[{name}].provider is an OpenRouter routing object "
                         f"(needs backend: openrouter and a mapping); got backend={backend!r}")
    mt = int(spec.get("max_tokens", 6000))
    # Below `llm.recommended_budgets()` a score is a budget-limited lower bound, not a capability
    # reading, so this raises. `--offline` never reaches here.
    _bv = _llm.note_budget(name, backend, mt, bool(spec.get("stream", False)))
    if _bv["below_recommended"]:
        raise ValueError(
            f"config.models[{name}].max_tokens = {mt} is below the recommended "
            f"{_bv['recommended']}; a run at this budget gives a budget-limited lower bound, "
            f"not a capability reading. Raise it to >={_bv['recommended']}, or update "
            f"`llm.MEASURED_BUDGET_BASIS` and record the source of the new value.")
    if backend not in _KEY_POOLS:
        _KEY_POOLS[backend] = _load_key_pool(bdef, env, backend)
    pool = _KEY_POOLS[backend]                   # shared within a backend: an exhausted key is skipped by every model
    key = pool.current() or env.get(bdef["key_env"], "")
    if bdef["kind"] == "google":
        g = GoogleSolver(name, str(spec["model"]), key, timeout, retries, mt)
        if spec.get("upstream"):
            g.upstream = str(spec["upstream"])
        return g
    s = OpenAICompatSolver(
        name, str(spec["model"]), key, timeout,
        int(spec.get("retries", retries)), mt,
        url=bdef["url"], backend=backend,
        max_tokens_field=bdef.get("max_tokens_field", "max_tokens"),
        # `stream` is the computed value (see `llm.note_budget`); config can only turn it on.
        stream=bool(_bv["stream"]), reasoning_effort=spec.get("reasoning_effort"),
        provider=spec.get("provider"))
    # Declared upstream (who serves the weights): part of the answer view, so two routes
    # that declare the same upstream answer as one solver (`transport.upstream_of`).
    if spec.get("upstream"):
        s.upstream = str(spec["upstream"])
    s.pool = pool
    return s


#: Keys of a route entry (`config.models.<m>.fallback_routes[]`, runtime.yaml `routes.<m>[]`):
#: each overrides the model's own spec for that route.
ROUTE_ENTRY_KEYS = frozenset({"backend", "model", "upstream", "provider", "stream", "retries",
                              "max_tokens_field"})


def route_name(spec: dict) -> str:
    only = ((spec.get("provider") or {}).get("only") or [None])[0]
    return f"{spec.get('backend', 'openrouter')}/{spec.get('upstream') or only or spec.get('model')}"


def route_chain(cfg: dict, name: str, ledger_path=None) -> list[dict]:
    """Fallback route specs for model `name`, in order, after its primary route.

    The chain is runtime.yaml `routes.<name>` when that file sets one (read when the run's
    accounting is prepared), else `config.models.<name>.fallback_routes`. Each entry
    overrides the model's spec; an entry that names the primary's own backend and model
    is the primary and is skipped. A route may change only how the model is reached: its
    answer view (model, upstream, max_tokens, effort, format) must equal the primary's,
    which `prepare_accounting` checks before any request.
    """
    base = {k: v for k, v in (cfg.get("models", {}).get(name) or {}).items() if k != "fallback_routes"}
    chain = None
    if ledger_path is not None:
        from .ops.runtime import for_ledger
        chain = (for_ledger(ledger_path).current().get("routes") or {}).get(name)
    if chain is None:
        chain = (cfg.get("models", {}).get(name) or {}).get("fallback_routes") or []
    out = []
    for entry in chain:
        bad = set(entry) - ROUTE_ENTRY_KEYS
        if bad:
            raise ValueError(f"route entry for {name} has unknown key(s) {sorted(bad)}")
        spec = {**base, **entry}
        if "provider" not in entry and spec.get("backend") != base.get("backend"):
            spec.pop("provider", None)            # a pin belongs to the route that declared it
        if (spec.get("backend", "openrouter"), spec.get("model")) == (
                base.get("backend", "openrouter"), base.get("model")):
            continue
        out.append(spec)
    return out


def fallback_solver_factories(cfg: dict, name: str, primary, ledger_path=None) -> list:
    """(route name, factory, spec) for each fallback route of `name` (see `route_chain`)."""
    env = load_env_file(cfg.get("env_file"))
    timeout = int(getattr(primary, "timeout", cfg.get("timeout_s", 900)))
    retries = int(cfg.get("max_retries", 2))
    out = []
    for spec in route_chain(cfg, name, ledger_path):
        if (spec.get("backend", "openrouter"), spec.get("model")) == (
                getattr(primary, "backend", None), getattr(primary, "model", None)):
            continue                                   # the route this run already takes
        _ensure_backends_registered(cfg)
        if spec.get("backend", "openrouter") not in BACKENDS:
            raise ValueError(f"fallback route {route_name(spec)} for {name}: backend not registered")
        out.append((route_name(spec),
                    (lambda _spec=spec: solver_for_spec(name, _spec, env, timeout, retries)), spec))
    return out


def raw_complete_with_usage(solver, prompt: str) -> tuple[str, dict | None]:
    """Fire one request and return (answer text, usage), normalizing the two backends' shapes.

    Only the answer channel is read, never reasoning. `usage` is `None` when unavailable, so
    `billed_tokens` reports it as absent rather than zero.
    """
    resp = solver._post(prompt)
    if "candidates" in resp:
        cand = (resp.get("candidates") or [{}])[0]
        return ("".join(p.get("text", "")
                        for p in ((cand.get("content") or {}).get("parts") or [])),
                _google_usage(resp))
    ch = (resp.get("choices") or [{}])[0]
    return (str(((ch.get("message") or {}).get("content")) or ""),
            resp.get("usage") or None)


def raw_complete(solver, prompt: str) -> str:
    return raw_complete_with_usage(solver, prompt)[0]


def build_solvers(job, cfg) -> list[tuple[str, object]]:
    """Offline reference solvers first (free), then real models. Returns (name, factory) pairs;
    the factory is called once per case, because some stubs (`StubbornSolver`,
    `FlipFlopSolver`) keep per-case state.
    """
    _ensure_backends_registered(cfg)
    out: list[tuple[str, object]] = []
    if job.include_baseline:
        from solver import StubbornSolver                          # kernel: anchored / never recants
        from .baselines import (ConstantDdxSolver, FlipFlopSolver, GatedProbeSolver,
                                GatedShotgunSolver, HumbleSolver, OracleProbeSolver,
                                OracleTestsSolver, TestOrderingSolver, OracleReviewSolver,
                                OneLongTestSolver,
                                NoReviewFlagSolver, LateConvergeSolver,
                                GateTripSolver, TraceJunkSolver, BlindConfidentSolver,
                                GatedSignalFloodSolver, GatedMinimalSolver)
        out += [("baseline_slope", _const(BaselineSolver())), ("robust_ref", _const(RobustSolver())),
                ("no_revision", StubbornSolver), ("flip_flop", FlipFlopSolver),
                # Constant baseline (unified + A3 without reading the question): the floor for the join /
                # urgency dimensions.
                ("const_ddx", _const(ConstantDdxSolver())),
                # Test-ordering stubs that ignore the question: the whole menu (shotgun ceiling) and a fixed
                # routine panel.
                ("shotgun_tests", _const(TestOrderingSolver("shotgun"))),
                ("common_panel", _const(TestOrderingSolver("panel"))),
                ("gated_probe", GatedProbeSolver),
                ("gated_shotgun", GatedShotgunSolver),
                # Orders every available signal: a score near 1.0 means `sd_coverage` rewards blanket
                # ordering. Stateful, so a factory.
                ("gated_flood", GatedSignalFloodSolver),
                # Queries exactly once, which maximizes the tool track's ratio metrics (see `GatedMinimalSolver`).
                ("gated_minimal", GatedMinimalSolver),
                # Always declares insufficient information: must score full marks on that tier and be penalized
                # on ordinary questions.
                ("humble_ref", _const(HumbleSolver())),
                # Oracle: orders exactly the gold tests (a ceiling, labeled oracle in reports).
                ("oracle_tests", _const(OracleTestsSolver())),
                # Every name in `BASELINE_NAMES` must be constructed here. `onelong` is the reference for the
                # degenerate ceilings of `disc_recall` / `tests_precision`.
                ("onelong", _const(OneLongTestSolver())),
                # Right/wrong oracle answers for the Q-side probes, so the checks are shown to pass and to fail.
                ("oracle_probe_right", _const(OracleProbeSolver("right"))),
                ("oracle_probe_wrong", _const(OracleProbeSolver("wrong"))),
                # Both ends of `review_macro` (specificity; "always says so" scores 0.000).
                ("oracle_review_right", _const(OracleReviewSolver("right"))),
                ("oracle_review_wrong", _const(OracleReviewSolver("wrong"))),
                # Positive controls that trip the two action gates by construction.
                ("gatetrip_treatment", _const(GateTripSolver("treatment"))),
                ("gatetrip_invasive", _const(GateTripSolver("invasive"))),
                # Positive control for escalating to A2 without requesting clinician review.
                ("noreview", _const(NoReviewFlagSolver())),
                # Wavers on the first two slices and settles on the third (range for `converged_at`); stateful.
                ("late_converge", LateConvergeSolver),
                # Process-track floor: shape-valid traces with filler content (see `TraceJunkSolver`).
                ("trace_junk", _const(TraceJunkSolver())),
                # `self_discovery` floor: no queries plus a claim that the data suffices (see `BlindConfidentSolver`).
                ("blind_confident", _const(BlindConfidentSolver()))]
    ai = os.path.expanduser(cfg.get("ai_dispatcher", "~/.local/bin/ai"))
    env = load_env_file(cfg.get("env_file"))
    timeout, retries = int(cfg.get("timeout_s", 900)), int(cfg.get("max_retries", 2))
    _wanted_backends = {(cfg.get("models", {}).get(n) or {}).get("backend", "openrouter")
                        for n in (job.models or cfg.get("default_models", []))
                        if isinstance(cfg.get("models", {}).get(n), dict)}
    _ensure_no_proxy({BACKENDS[b]["no_proxy_host"] for b in _wanted_backends
                      if b in BACKENDS and BACKENDS[b].get("no_proxy_host")})
    # An explicitly requested unknown model fails rather than running zero cells and exiting 0.
    # Unknown names in `default_models` are skipped with a warning.
    _explicit = bool(job.models)
    _unknown = [n for n in (job.models or ()) if not cfg.get("models", {}).get(n)]
    if _explicit and _unknown:
        raise ValueError(f"--models has name(s) not defined in config.models: {_unknown} "
                         f"(registered: {sorted(cfg.get('models', {}))}); "
                         f"an explicitly requested model that does not exist must fail")
    # Retired models stay registered so historical rows have a known solver name; naming one
    # explicitly is refused, since it would send billed requests.
    _retired = [n for n in (job.models or ())
                if (cfg.get("models", {}).get(n) or {}).get("retired")]
    if _explicit and _retired:
        raise ValueError(
            f"--models includes retired model(s): "
            f"{[(n, cfg['models'][n]['retired']) for n in _retired]}. They stay registered "
            f"so historical batches resolve; remove `retired` to run one.")
    for name in (job.models or cfg.get("default_models", [])):
        spec = cfg.get("models", {}).get(name)
        if not spec:
            log.warning("[eval] unknown model %s (not defined in config.models), skipping", name)
            continue
        if isinstance(spec, dict):
            _sv = solver_for_spec(name, spec, env, timeout, retries)
            if _sv is None:
                if _explicit:
                    raise ValueError(
                        f"--models {name}: no solver could be built "
                        f"(backend {spec.get('backend', 'openrouter')!r}; see the warning above)")
                continue
            backend = spec.get("backend", "openrouter")
            bdef = BACKENDS[backend]
            mt = int(spec.get("max_tokens", 6000))
            pool = _KEY_POOLS[backend]
            key = pool.current() or env.get(bdef["key_env"], "")
            if bdef["kind"] == "google":
                out.append((name, _const(_sv)))
            else:
                _s = _sv
                _s.pool = pool
                out.append((name, _const(_s)))
        else:
            # List form = CLI subprocess via `~/.local/bin/ai`, refused unless the top-level
            # `allow_cli_solver` is set (see `CLISolver`).
            out.append((name, _const(CLISolver(
                name, [str(x) for x in spec], ai, timeout, retries, env=env,
                allow_unsandboxed=bool(cfg.get("allow_cli_solver", False))))))
    return out


#: Per-cell usage accumulator, `(case, solver) -> {...}`: filled by `save_response` for each
#: response and collected by `_one`. Hanging it off persistence covers every geometry and
#: every ABORT branch, including multi-request cells.
_USAGE_ACC: dict[tuple[str, str], dict] = {}

_RETRY_ACC: dict[tuple[str, str], dict] = {}

_ATTR_MOD: list = []


def retry_classifier():
    """Load the retry-attribution classifier from `tools/attribute_failures.py`.

    That file is outside the frozen manifests; attribution is a pure function of the stored
    `failed_attempts` (recomputable), and each row carries `retry_attr_sha16`.
    """
    if not _ATTR_MOD:
        import importlib.util
        p =_dr() / "tools" / "attribute_failures.py"
        spec = importlib.util.spec_from_file_location("_haenv_attr_failures", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                            # noqa: S102
        _ATTR_MOD.append(mod)
    return _ATTR_MOD[0]


def retry_attr_sha16() -> str:
    p =_dr() / "tools" / "attribute_failures.py"
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return "unknown"


def retry_classes() -> tuple[str, ...]:
    return tuple(retry_classifier().BLAME)


def attribute_attempts(failed_attempts, budget_tokens: int | None = None) -> dict[str, int]:
    """One cell's (or call's) `failed_attempts` -> `{class: count}`, via the classifier's
    `_attempts` + `classify`.

    Pass `budget_tokens`: without it `classify` cannot separate `no_answer` from `budget` and
    falls back to sentence matching (as for offline stubs).
    """
    m = retry_classifier()
    out: dict[str, int] = {}
    for e in m._attempts({"failed_attempts": failed_attempts}):  # noqa: SLF001 -- the classifier's own helper
        k = m.classify(e, budget_tokens)
        out[k] = out.get(k, 0) + 1
    return out


def _note_grid_retry(case: str, solver: str, out) -> None:
    e = _RETRY_ACC.setdefault((str(case), str(solver)),
                              {"n_calls": 0, "n_retry": 0, "by_class": {}})
    e["n_calls"] += 1
    cls = attribute_attempts(getattr(out, "_failed_attempts", None),
                             getattr(out, "_max_tokens", None))
    for k, v in cls.items():
        e["by_class"][k] = e["by_class"].get(k, 0) + v
        e["n_retry"] += v


def take_grid_retry(case: str, solver: str, overall=None) -> dict:
    """Take this cell's retry attribution as `eval.jsonl` fields (cleared once taken).

    Fields: `retry_status` (`measured` / `absent:stub` / `missing:no_response`), `n_retry`,
    `retry_<class>`, `retry_not_model` (infra + auth + budget), `rescue`, `retry_attr_sha16`.
    `rescue` is True when the cell had failed retries and still got a verdict (`SCORED` or
    `FAIL(gate)`), False otherwise, and None when no response was persisted.
    """
    from .baselines import BASELINE_NAMES as _BN
    e = _RETRY_ACC.pop((str(case), str(solver)), None)
    _is_stub = str(solver) in _BN
    if e is None or not e["n_calls"]:
        return {"retry_status": "absent:stub" if _is_stub else "missing:no_response",
                "rescue": None}
    _ov = str(overall or "")
    _graded = _ov == "SCORED" or _ov.startswith("FAIL(")
    row = {"retry_status": "measured", "n_retry": e["n_retry"],
           "retry_n_calls": e["n_calls"],
           "rescue": bool(e["n_retry"]) and _graded,
           "retry_attr_sha16": retry_attr_sha16()}
    _not_model = 0
    for k in retry_classes():
        v = int(e["by_class"].get(k, 0))
        row[f"retry_{k}"] = v
        if k in ("infra", "auth", "budget"):
            _not_model += v
    row["retry_not_model"] = _not_model
    return row


def _note_grid_usage(case: str, solver: str, out) -> None:
    b = billed_tokens(getattr(out, "_usage", None))
    k = (str(case), str(solver))
    e = _USAGE_ACC.setdefault(k, {"n_calls": 0, "n_with_usage": 0,
                                  "in_fresh": 0, "in_cached": 0, "out": 0,
                                  "out_thoughts": 0, "unaccounted": 0,
                                  "latency_s": 0.0, "n_latency": 0,
                                  # Each ceiling boolean has its own denominator: `ceil_by_finish` needs only
                                  # `finish_reason`, `ceil_by_ratio` also needs usage.
                                  "n_fin_known": 0, "n_ceil_by_finish": 0,
                                  "n_ratio_known": 0, "n_ceil_by_ratio": 0,
                                  "n_at_risk": 0, "n_degenerate": 0,
                                  "retry_in": 0, "retry_out": 0,
                                  "retry_n": 0, "retry_n_with_usage": 0,
                                  "budgets": set()})
    # Count the gated track's earlier query rounds too (`out` is only the final round, which is
    # counted below).
    _gr = [x for x in (getattr(out, "_gated_rounds", None) or []) if isinstance(x, dict)]
    if _gr:
        _last = max(int(x.get("round") or 0) for x in _gr)
        for _x in _gr:
            if int(_x.get("round") or 0) >= _last:
                continue                      # last round = `out`, recorded by the block below
            e["n_calls"] += 1
            _xb = billed_tokens(_x.get("usage"))
            if _xb.get("status") == "measured":
                e["n_with_usage"] += 1
                for f in ("in_fresh", "in_cached", "out"):
                    e[f] += int(_xb.get(f) or 0)
                e["out_thoughts"] += int(_xb.get("out_thoughts") or 0)
                e["unaccounted"] += int(_xb.get("unaccounted") or 0)
            _xl = _x.get("latency_s")
            if isinstance(_xl, (int, float)):
                e["latency_s"] += float(_xl)
                e["n_latency"] += 1
    e["n_calls"] += 1
    for _a in (getattr(out, "_failed_attempts", None) or []):
        e["retry_n"] += 1
        _ab = _a.get("billed") if isinstance(_a, dict) else None
        if isinstance(_ab, dict) and _ab.get("status") == "measured":
            e["retry_n_with_usage"] += 1
            e["retry_in"] += int(_ab.get("in_fresh") or 0) + int(_ab.get("in_cached") or 0)
            e["retry_out"] += int(_ab.get("out") or 0)
    if b.get("status") == "measured":
        e["n_with_usage"] += 1
        for f in ("in_fresh", "in_cached", "out"):
            e[f] += int(b.get(f) or 0)
        e["out_thoughts"] += int(b.get("out_thoughts") or 0)
        e["unaccounted"] += int(b.get("unaccounted") or 0)
    _lat = getattr(out, "_latency_s", None)
    if isinstance(_lat, (int, float)):
        e["latency_s"] += float(_lat)
        e["n_latency"] += 1
    # ---- ceiling hits (`llm.ceiling_flags`) ----
    # Reasoning and answer share `max_tokens`; these flags separate a budget-limited answer from
    # a capability floor.
    _mt = getattr(out, "_max_tokens", None)
    if _mt:                       # stubs/offline lack this attribute => not recorded, not recorded as 0
        _cf = _llm.ceiling_flags(
            (b.get("out") if b.get("status") == "measured" else None),
            getattr(out, "_finish", None), int(_mt),
            str(getattr(out, "_raw_text", "") or ""))
        e["budgets"].add(int(_mt))
        if getattr(out, "_finish", None) is not None:
            e["n_fin_known"] += 1
            e["n_ceil_by_finish"] += int(bool(_cf["ceil_by_finish"]))
        if _cf["ceil_by_ratio"] is not None:
            e["n_ratio_known"] += 1
            e["n_ceil_by_ratio"] += int(bool(_cf["ceil_by_ratio"]))
            e["n_at_risk"] += int(bool(_cf["at_risk"]))
        e["n_degenerate"] += int(bool(_cf["degenerate"]))


def _ceiling_row(e: dict) -> dict:
    """This cell's ceiling readings for `eval.jsonl`: `absent:no_budget` when no call carried a
    budget (stub/offline), else `measured`. A boolean whose denominator (`n_*_known`) is 0 is
    `None`, never `False`.
    """
    if not e.get("budgets"):
        return {"ceiling": {"status": "absent:no_budget"}}
    _b = sorted(e["budgets"])
    return {"ceiling": {
        "status": "measured",
        "budget": (_b[0] if len(_b) == 1 else _b),
        "by_finish": (None if not e["n_fin_known"] else bool(e["n_ceil_by_finish"])),
        "by_ratio": (None if not e["n_ratio_known"] else bool(e["n_ceil_by_ratio"])),
        "at_risk": (None if not e["n_ratio_known"] else bool(e["n_at_risk"])),
        "degenerate": bool(e["n_degenerate"]),
        "n_calls": e["n_calls"],
        "n_fin_known": e["n_fin_known"], "n_ceil_by_finish": e["n_ceil_by_finish"],
        "n_ratio_known": e["n_ratio_known"], "n_ceil_by_ratio": e["n_ceil_by_ratio"]}}


def take_grid_usage(case: str, solver: str) -> dict:
    """Take this cell's usage as `eval.jsonl` fields (cleared once taken).

    Writes `billed.in_total` / `billed.out` / `latency_s`, the keys
    `report.cost_efficiency_analysis` reads. `usage_status` is `measured`, `absent:stub`
    (offline stub), `missing:no_response` (real model, no response persisted) or
    `missing:no_usage` (the backend returned none); stubs are identified by
    `baselines.BASELINE_NAMES`.
    """
    from .baselines import BASELINE_NAMES as _BN
    e = _USAGE_ACC.pop((str(case), str(solver)), None)
    _is_stub = str(solver) in _BN
    if e is None or not e["n_calls"]:
        return {"usage_status": "absent:stub" if _is_stub else "missing:no_response"}
    if not e["n_with_usage"]:
        return {"usage_status": "absent:stub" if _is_stub else "missing:no_usage",
                "usage_n_calls": e["n_calls"], **_ceiling_row(e)}
    row = {"usage_status": "measured", **_ceiling_row(e),
           "billed": {"in_fresh": e["in_fresh"], "in_cached": e["in_cached"],
                      "in_total": e["in_fresh"] + e["in_cached"], "out": e["out"],
                      "out_thoughts": e["out_thoughts"] or None,
                      "unaccounted": e["unaccounted"] or None,
                      "status": "measured"},
           "usage_n_calls": e["n_calls"]}
    if e["n_with_usage"] < e["n_calls"]:
        row["usage_n_calls_without"] = e["n_calls"] - e["n_with_usage"]
    if e["retry_n"]:
        row["billed_retry"] = {"in_total": e["retry_in"], "out": e["retry_out"],
                               "n_attempts": e["retry_n"],
                               "n_with_usage": e["retry_n_with_usage"],
                               "status": ("measured" if e["retry_n_with_usage"] == e["retry_n"]
                                          else "lower_bound")}
        if e["retry_n_with_usage"] < e["retry_n"]:
            row["billed"]["status"] = "lower_bound"
            row["billed"]["lower_bound_why"] = (
                f"{e['retry_n'] - e['retry_n_with_usage']}/{e['retry_n']} failed attempt(s) "
                f"did not return usage")
    if e["n_latency"]:
        row["latency_s"] = round(e["latency_s"], 2)
    return row


def save_simple_trace(path: Path, case: str, solver: str, geometry: str, out,
                      slice_t=None, step: int = 1) -> None:
    """Trace for the non-gated geometries: one model request per step, no tool calls.

    It is where the raw reasoning text is kept (`save_response` never persists it). The gated
    geometry writes its own trace in `gated.run_gated`.
    """
    from .trace import TraceLog, save_trace
    txt = getattr(out, "_raw_text", None)
    if txt is None:
        return
    log = TraceLog(case=case, solver=solver, geometry=geometry)
    log.append("case/start", {"slice_t": slice_t})
    log.append("step/start", {}, step=step)
    log.append("request/header", {"prompt_mode": getattr(out, "_prompt_mode", "default"),
                                  "probe_id": getattr(out, "_probe_id", None),
                                  "prompt_sha256": getattr(out, "_prompt_sha", None),
                                  "slice_t": slice_t}, step=step)
    log.append("assistant/message", {
        "raw": txt, "n_chars": len(txt),
        "usage": getattr(out, "_usage", None),
        "finish_reason": getattr(out, "_finish", None),
        "max_tokens": getattr(out, "_max_tokens", None),
        "latency_s": getattr(out, "_latency_s", None),
        "reasoning_text": getattr(out, "_reasoning_text", None),
    }, step=step)
    for _fa in (getattr(out, "_failed_attempts", None) or []):
        log.append("assistant/attempt", dict(_fa), step=step)
    log.append("step/end", {"reason": "answered"}, step=step)
    log.append("case/end", {"slice_t": slice_t})
    save_trace(path, log)


def save_response(path: Path, case: str, solver: str, out, slice_t=None,
                  rounds=None) -> None:
    """Append the tested model's raw response to `responses.jsonl`, so a scoring change is a
    recompute, not a rerun. The prompt itself is not stored; `prompt_mode` and the framing
    fingerprint identify it.
    """
    txt = getattr(out, "_raw_text", None)
    if txt is None:
        return
    pid = getattr(out, "_probe_id", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    _note_grid_usage(case, solver, out)
    _note_grid_retry(case, solver, out)
    _billed, _extra = billed_tokens(getattr(out, "_usage", None)), {}
    if rounds and any(isinstance(x, dict) and "round" in x for x in rounds):
        # Multi-round (gated) row: `usage` / `finish_reason` are the last turn's, `billed` is the
        # cell total over every round (the same rule as `_note_grid_usage`: the last round is `out`).
        _rs = [x for x in rounds if isinstance(x, dict)]
        _last = max(int(x.get("round") or 0) for x in _rs)
        _tot, _n, _n_meas = dict(_billed), 1, int(_billed["status"] == "measured")
        _tot = {k: (int(v or 0) if k in ("in_fresh", "in_cached", "out", "out_thoughts", "unaccounted") else v)
                for k, v in _tot.items()}
        for _x in _rs:
            if int(_x.get("round") or 0) >= _last:
                continue
            _xb = billed_tokens(_x.get("usage"))
            _n += 1
            if _xb["status"] == "measured":
                _n_meas += 1
                for f in ("in_fresh", "in_cached", "out", "out_thoughts", "unaccounted"):
                    _tot[f] += int(_xb.get(f) or 0)
        _tot["in_total"] = _tot["in_fresh"] + _tot["in_cached"]
        _tot["out_thoughts"] = _tot["out_thoughts"] or None
        _tot["unaccounted"] = _tot["unaccounted"] or None
        _tot["n_calls"] = _n
        _tot["status"] = ("absent" if not _n_meas else
                          "measured" if _n_meas == _n else "lower_bound")
        _extra = {"billed_last_turn": _billed, "usage_scope": "last_turn"}
        _billed = _tot
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"case": case, "solver": solver, "slice_t": slice_t,
                                "prompt_mode": getattr(out, "_prompt_mode", "default"),
                                "probe_id": pid,
                                "framing_sha256": (framing_sha256(
                                    (RUN.probes or {}).get(pid, {}).get("framing_ref", ""))
                                    if RUN.probes and pid in (RUN.probes or {}) else None),
                                "prompt_sha256": getattr(out, "_prompt_sha", None),
                                "usage": getattr(out, "_usage", None),
                                "finish_reason": getattr(out, "_finish", None),
                                "failed_attempts": getattr(out, "_failed_attempts", None) or None,
                **({"rate_limit_retries": out._rate_limit_retries}
                   if getattr(out, "_rate_limit_retries", None) else {}),
                                "latency_s": getattr(out, "_latency_s", None),
                                "reasoning_chars": getattr(out, "_reasoning_chars", None),
                                "billed": _billed, **_extra,
                                "rounds": rounds or None,
                                "n_chars": len(txt), "raw": txt}, ensure_ascii=False) + "\n")


#: Resume policy: a disposition is retryable if a rerun could produce a different result
#: (`ABORT(no_response)`, `ABORT(unparseable)`, `ABORT(no_answer)`, `ERROR`); `ABORT(leak)`
#: and `ABORT(iron_law)` are deterministic verdicts and count as done.
RETRYABLE_OVERALL = frozenset({"ABORT(no_response)", "ABORT(unparseable)",
                               "ABORT(no_answer)", "ERROR"})

TERMINAL_OVERALL = frozenset({"ABORT(leak)", "ABORT(iron_law)"})


def is_retryable(overall) -> bool:
    """Whether this cell should be resent on a resumed run. Unrecognized dispositions count as
    done, so a misspelled name never re-runs (and re-bills) a batch.
    """
    o = str(overall or "")
    if not o:
        return False
    if o.startswith("ABORT(") and ")" in o:
        o = "ABORT(" + o[len("ABORT("):o.index(")")].split(":")[0].strip() + ")"
    return o in RETRYABLE_OVERALL


def _done_keys(path: Path) -> set[str]:
    """The set of cells already run. Retryable dispositions (e.g. transport failures) are not
    counted, so a resumed run fills them in; `ABORT(leak)` is a verdict and counts as done.
    """
    if not path.exists():
        return set()
    keys: set[str] = set()
    retryable: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            r = json.loads(line)
            _ov = str(r.get("overall") or "")
            if is_retryable(_ov):
                retryable[_ov] = retryable.get(_ov, 0) + 1
                continue
            keys.add(f"{r['case']}|{r['solver']}")
        except Exception:
            pass
    if retryable:
        log.warning("[eval] %d cell(s) not counted as done, rerunning: %s",
                    sum(retryable.values()),
                    " · ".join(f"{k} {v}" for k, v in sorted(retryable.items())))
    return keys


def json_unextractable(raw_text) -> bool:
    """True if the raw text is non-empty but no JSON object can be extracted from it.

    The kernel's `solver._extract_json` falls back to `{}` in that case, and `_to_output({})`
    would turn it into a canned abstention scored as the model's answer. This calls the same
    extractor (which strips `<think>` blocks) so it agrees with the scoring path.
    """
    if raw_text is None:
        return False                       # attribute absent = offline stub, not applicable (same convention as `raw_empty`)
    t = str(raw_text)
    if not t.strip():
        return False                       # genuinely empty -- goes through the existing `raw_empty` path, not judged here
    try:
        data = _extract_json(t)
    except Exception:                      # noqa: BLE001 -- parse error = same as `{}` once retries are exhausted
        return True
    return not (isinstance(data, dict) and data)


def _row_single(cid, sname, raw, T, solver) -> dict:
    sp, vp = build_instance(raw, T)
    from . import process
    from .solve_guard import guarded_solve
    ddx0 = (vp.adjudication or {}).get("ddx")

    _iron_law = iron_law_precheck(raw, sp, vp, solver)

    try:
        facts = guarded_solve(
            solver, sp, T,
            prompt_mode=("ddx" if ddx0 else "default"),   # the question side decides the framing, not information from the answer side
            precheck=_iron_law,
            on_output=(lambda o: (save_response(RUN.resp_path, cid, sname, o),
                                  save_simple_trace(RUN.resp_path.with_name("trace.jsonl"),
                                                    cid, sname, "single", o))
                       if RUN.resp_path else None),
            tag=f"{cid}|{sname}")
    except wq.IronLawViolation as e:
        log.error("[wq] %s|%s iron-law violation -> aborting this cell: %s", cid, sname, e)
        return {"case": cid, "solver": sname, "iron_law": str(e), "overall": "ABORT(iron_law)"}
    if facts.leak:                                    # leakage -> abort this cell, no score given
        return {"case": cid, "solver": sname, "leak": facts.leak, "overall": "ABORT(leak)"}
    _q, _wq_warn = facts.precheck_result
    out = facts.out
    # No response after retries: abort the cell, excluded from the denominator. The signal is
    # `_raw_text`: absent means an offline stub (not applicable), present but empty means no
    # answer; the output fields are always filled with defaults and cannot tell.
    if facts.raw_empty:
        att = facts.failed_attempts or []
        log.error("[eval] %s|%s no response (retries exhausted after %d attempts) -> "
                  "aborting this cell, excluded from the denominator", cid, sname, len(att))
        return {"case": cid, "solver": sname, "T": T, "overall": "ABORT(no_response)",
                "failed_attempts": att or None,
                "no_response_reason": (str(att[-1].get("error"))[:120] if att else "empty_response")}
    if facts.raw_unparseable:
        log.error("[eval] %s|%s raw text is %d chars but no JSON could be extracted -> "
                  "aborting this cell, excluded from the denominator (the kernel would "
                  "otherwise parse it as {} and record an abstention)",
                  cid, sname, facts.n_chars)
        return {"case": cid, "solver": sname, "T": T, "overall": "ABORT(unparseable)",
                "failed_attempts": facts.failed_attempts,
                "n_chars_unparseable": facts.n_chars,
                "no_response_reason": "json_unextractable"}
    rep = verifier_mod.grade(out, vp, [e["evidence_id"] for e in sp.evidence_ledger])
    from .judges import run_judges, SINGLE
    # ---- process-track scoring ----
    # Reads `out._raw`; the outcome grade above sees only `SolverOutput`, so the process record
    # cannot feed into it. Fields carry a `trace_` prefix.
    _proc = process.run_process_judges(getattr(out, "_raw", None),
                                       [e["evidence_id"] for e in (sp.evidence_ledger or [])])
    return {"case": cid, "solver": sname, "T": T,
            **run_judges(SINGLE, out, vp), **_proc,
            **judges_mod.judge_noop_probe(out, NOOP_FOR.get(str(cid))),
            **judges_mod.judge_quant_probe(out, QUANT_FOR.get(str(cid)), t_max=int(T)),
            "action": (out.action or {}).get("selected_action_class"),
            # `Q.world_ref = {world_id, world_truth_hash}` (which world the question points into),
            # distinct from the provenance field `world_ref`.
            "q_world_ref": _q.world_ref,
            "q_question_id": _q.question_id,
            "wq_warnings": _wq_warn or None,
            "tracks": rep.tracks, "gates": rep.hard_gate_failures, "overall": rep.overall}


def iron_law_precheck(raw, sp, vp, solver):
    """Build Q (storing only `world_ref`) and run the W x Q iron laws; returns a thunk yielding
    `(Q, warnings)`.

    Violations of iron laws 1 and 3 (Q embeds W's ground truth; Q cites evidence not in W) and a
    hash mismatch raise; iron law 2 (the distractor ledger inside W) and gold derivation are
    recorded as warnings (see `wq.enforce`). Single/slices pass it as `precheck` (run after the leak
    probe, so a cell failing both is `ABORT(leak)`); gated/multi call it before solving, so
    there a cell failing both is `ABORT(iron_law)`.
    """
    def _run():
        q = wq.build_question(raw, sp, probe_id=getattr(solver, "probe_id", "") or "",
                              judge_ref=str((vp.adjudication or {}).get("ddx", {})
                                            .get("join_gold") or ""))
        return q, wq.enforce(q, raw, strict=True)
    return _run


def _premise_row(cid: str, out) -> dict:
    """False-premise judge for the row; `{}` when the case has no premise (not 0)."""
    from . import judges as _j
    prem = PREMISE_FOR.get(str(cid))
    if not prem or out is None:
        return {}
    return _j.judge_premise_challenge(out, prem)


def merge_bought_tests(answer, targets, menu) -> list[str]:
    """Merge tests bought from the menu's test section into `answer["tests_to_order"]`, in place.

    Returns the merged names (`tests_from_queries` counts them). Signal queries use a different
    vocabulary and are not merged. Shared by `_row_gated` and `tools/recompute_judges.py`.
    """
    _tests_on_menu = {str(i["target"]) for i in (menu or []) if i.get("is_test")}
    _q = [str(t) for t in (targets or []) if str(t) in _tests_on_menu]
    if _q and isinstance(answer, dict):
        answer["tests_to_order"] = list(answer.get("tests_to_order") or []) + _q
    return _q


def budget_abort(committed, spent, budget) -> bool:
    """`ABORT(no_answer:budget)`: no committed answer and the budget spent. Shared by `_row_gated`
    and `tools/recompute_judges.py`, which re-derives the verdict from a row's recorded fields."""
    return (not committed) and bool(budget) and spent >= budget


def gated_menu(raw, T: int) -> list[dict]:
    """The menu `gated.run_gated` hands the model for this case, rebuilt with the same calls."""
    from .gated import menu_for, withhold_signals
    sp, _ = build_instance(raw, int(T))
    _, withheld = withhold_signals(sp)
    return menu_for(sp, withheld, getattr(raw, "case_id", ""))


def _row_gated(cid, sname, raw, T, solver) -> dict:
    """Gated (on-demand query) geometry, tool track T1-T4.

    On diagnostic questions T4 may be None (not applicable); T1-T3 are always scored.
    """
    _sp0, _vp0 = build_instance(raw, int(T))
    solver.prompt_mode = "ddx" if (_vp0.adjudication or {}).get("ddx") else "default"
    from .gated import run_gated
    from . import tracks as _tk
    # `run_gated` has no `precheck` hook, so the iron laws run here, before solving.
    try:
        _q_g, _wq_warn_g = iron_law_precheck(raw, _sp0, _vp0, solver)()
    except wq.IronLawViolation as e:
        log.error("[wq] %s|%s iron-law violation (gated) -> aborting this cell: %s", cid, sname, e)
        return {"case": cid, "solver": sname, "T": int(T),
                "iron_law": str(e), "overall": "ABORT(iron_law)"}
    from .trace import TraceLog as _TraceLog, save_trace as _save_trace
    _tl = _TraceLog(case=cid, solver=sname, geometry="gated") if RUN.resp_path else None
    solver = _replay_solver_for(solver, cid, sname, "gated")
    out, tr = run_gated(raw, int(T), solver, trace=_tl)
    if _tl is not None:
        _save_trace(RUN.resp_path.with_name("trace.jsonl"), _tl)
    # Leakage first: it must not fall through to `ABORT(no_response)` below.
    if getattr(tr, "leak", None):
        return {"case": cid, "solver": sname, "T": int(T),
                "leak": list(tr.leak), "overall": "ABORT(leak)"}
    if out is not None and (getattr(tr, "raw_empty", False) or getattr(tr, "raw_unparseable", False)):
        _why = "empty_response" if tr.raw_empty else "json_unextractable"
        log.error("[eval] %s|%s gated final round %s -> aborting this cell, excluded from "
                  "the denominator", cid, sname, _why)
        return {"case": cid, "solver": sname, "T": int(T),
                "overall": "ABORT(no_response)" if tr.raw_empty else "ABORT(unparseable)",
                "failed_attempts": getattr(out, "_failed_attempts", None) or None,
                **({"rate_limit_retries": out._rate_limit_retries}
                   if getattr(out, "_rate_limit_retries", None) else {}),
                "no_response_reason": _why}
    # Budget exhausted with no answer: `ABORT(no_answer:budget)`, not a low score. Running out
    # of rounds with budget left is the model's own behavior and stays in the denominator.
    if out is not None and budget_abort(getattr(tr, "committed", False), tr.spent, tr.budget):
        log.warning("[eval] %s|%s budget exhausted (%.1f/%.1f) with no answer -> "
                    "ABORT(no_answer:budget)",
                    cid, sname, tr.spent, tr.budget)
        return {"case": cid, "solver": sname, "T": int(T),
                "overall": "ABORT(no_answer:budget)",
                "no_response_reason": "budget_exhausted_before_answer",
                "tool_rounds_used": tr.rounds_used, "tool_spent": round(tr.spent, 2),
                "tool_budget": tr.budget, "tool_truncated": getattr(tr, "truncated", 0)}
    sp, vp = build_instance(raw, int(T))
    if out is not None:
        out._gated_rounds = list(tr.rounds_log or [])
    if RUN.resp_path and out is not None:
        save_response(RUN.resp_path, cid, sname, out, rounds=tr.rounds_log)
    row = {"case": cid, "solver": sname, "T": int(T),
           "tool_rounds_used": tr.rounds_used, "tool_spent": round(tr.spent, 2),
           "tool_budget": tr.budget, "tool_n_calls": len(tr.calls or []),
           # Purchased targets, needed by `tools/recompute_judges.py` to recompute gates (tests here
           # are bought, not written into the answer).
           "tool_targets": list(tr.targets or []) or None,
           **(_tk.tool_track(tr, vp, key_signals=_tk.key_signals_for(vp)) or {}),
           "overall": "SCORED" if out is not None else "ABORT(no_response)"}
    if out is not None:
        from .judges import run_judges, SINGLE
        _q = merge_bought_tests(getattr(out, "_raw", None), getattr(tr, "targets", None),
                                getattr(tr, "menu", None))
        row["tests_from_queries"] = len(_q)
        # Hard gates and judges run after the merge above; `tools/recompute_judges.py` makes the
        # same merge from the row's `tool_targets`. If grading raises, the `gates` key is omitted
        # (`gate_unknown`) and the error is recorded.
        try:
            _rep = verifier_mod.grade(
                out, vp, [e["evidence_id"] for e in (sp.evidence_ledger or [])])
        except Exception as e:                          # noqa: BLE001
            log.error("[eval] %s|%s gated kernel scoring failed -> recording gate_unknown "
                      "(not writing the `gates` key): %s",
                      cid, sname, e)
            row["gate_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        else:
            row["tracks"] = _rep.tracks
            row["gates"] = _rep.hard_gate_failures
            row["gates_recompute_skipped"] = False
            row["overall"] = _rep.overall
        row.update(run_judges(SINGLE, out, vp))
        row.update(judges_mod.judge_noop_probe(out, NOOP_FOR.get(str(cid))))
        row.update(judges_mod.judge_quant_probe(out, QUANT_FOR.get(str(cid)), t_max=int(T)))
        row.update(judges_mod.judge_abstention_calibration(out, vp))
        # Process judges on the final answer; the gated ledger has no lab tests (see
        # `render_delta_source_types` in `registry/pack_catalog.yaml`).
        row.update(process.run_process_judges(
            getattr(out, "_raw", None),
            [e["evidence_id"] for e in (sp.evidence_ledger or [])]))
    return row


# How far out of range, as a fraction of the reference range's width, a lab value must be
# to count as worsened. Parsed as in `build_premise`.
ABNORMAL_MARGIN = 0.30

_re_lab = __import__("re").compile(
    r"^(.+?)\s+([-+]?\d+(?:\.\d+)?)\s*([^\s(（]*)\s*[(（]\s*参考\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*[–\-~至]\s*([-+]?\d+(?:\.\d+)?)")


#: Judges of "did the final answer name the right condition": those with subject `LAST` on
#: slices in `mount_table`.
FINAL_ANSWER_JUDGES: tuple[str, ...] = tuple(
    sorted(n for n, row in _MT.MOUNT.items() if row.get("slices") == _MT.LAST))


#: Resume replay of slice answers. Cell rows are written when a cell completes, so a run that dies
#: mid-cell leaves answered slices in `responses.jsonl` and no row; the cell then restarts from its
#: first slice. With `RUN.replay` set (by `run_eval` when resuming), a saved answer that is
#: non-empty, parseable, not truncated and made for the same prompt is reused instead of re-bought.
_REPLAY_N: dict = {}              # (case, solver) -> slices reused this run
_REPLAY_FRESH: dict = {}          # (case, solver) -> answers bought while a replay was active


def _reusable(r: dict) -> bool:
    """A saved answer that may stand in for a fresh request: non-empty, parseable, not
    truncated, not an error, and tied to a prompt fingerprint."""
    raw = r.get("raw")
    return not (not isinstance(raw, str) or not raw.strip() or json_unextractable(raw)
                or str(r.get("finish_reason") or "").lower() in ("length", "max_tokens", "error")
                or not r.get("prompt_sha256"))


def load_slice_replay(path) -> dict:
    """Index the reusable answers in `responses.jsonl`.

    Slice rows are keyed `(case, solver, slice_t)`. Multi-round rows (one saved row per round,
    `rounds == [i]`) are keyed `(case, solver, "sha", prompt_sha256)`: a round's prompt is fully
    determined by what it was shown, so an identical fingerprint is the same request. Gated rows
    (`rounds` is a list of round dicts, written only when the cell ends) are never keyed here;
    their per-round answers come from `load_gated_replay`.
    """
    idx: dict = {}
    if path is None or not Path(path).exists():
        return idx
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rd = r.get("rounds")
            if r.get("slice_t") is None:
                if (isinstance(rd, list) and len(rd) == 1 and isinstance(rd[0], int)
                        and _reusable(r)):
                    idx.setdefault((str(r["case"]), str(r["solver"]), "sha", r["prompt_sha256"]), r)
                continue
            if rd or not _reusable(r):
                continue
            idx.setdefault((str(r["case"]), str(r["solver"]), int(r["slice_t"])), r)
    return idx


#: Gated cells persist every answered round here as it arrives (the cell's own `responses.jsonl`
#: row is written only when the cell ends, so a cell that dies mid-way leaves nothing else).
#: Read only by the resume replay; no other reader consumes it.
GATED_ROUNDS_FILE = "gated_rounds.jsonl"


def gated_rounds_path() -> Path | None:
    return RUN.resp_path.with_name(GATED_ROUNDS_FILE) if RUN.resp_path else None


def load_gated_replay(path) -> dict:
    """`(case, solver, "sha", prompt_sha256)` -> first reusable saved gated round."""
    idx: dict = {}
    if path is None or not Path(path).exists():
        return idx
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if _reusable(r):
                idx.setdefault((str(r["case"]), str(r["solver"]), "sha", r["prompt_sha256"]), r)
    return idx


def _prompt_sha_for(solver, payload) -> str:
    return hashlib.sha256(render_for(solver, payload)[0].encode()).hexdigest()[:16]


def _persist_gated_round(cid, sname, out) -> None:
    """Append one answered gated round to the sidecar (skipped when nothing reusable came back)."""
    path = gated_rounds_path()
    txt = getattr(out, "_raw_text", None)
    if path is None or txt is None or not getattr(out, "_prompt_sha", None):
        return
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "case": cid, "solver": sname, "raw": txt,
                "prompt_sha256": out._prompt_sha, "prompt_mode": getattr(out, "_prompt_mode", "default"),
                "probe_id": getattr(out, "_probe_id", None), "usage": getattr(out, "_usage", None),
                "finish_reason": getattr(out, "_finish", None),
                "max_tokens": getattr(out, "_max_tokens", None),
                "failed_attempts": getattr(out, "_failed_attempts", None) or None,
                **({"rate_limit_retries": out._rate_limit_retries}
                   if getattr(out, "_rate_limit_retries", None) else {}),
                "latency_s": getattr(out, "_latency_s", None),
                "reasoning_chars": getattr(out, "_reasoning_chars", None)},
                ensure_ascii=False) + "\n")


class _ReplaySolver:
    """Wraps a solver for one cell: a saved usable answer for the same prompt is returned
    without a request; anything else goes to the real solver.

    `persist_rounds` (gated): every freshly answered round is appended to the sidecar so that a
    later resume can reuse it. The prompt fingerprint covers the whole request (revealed data,
    budget spent, round number, menu), so a reused round is only ever the answer to the exact
    prompt it would have been re-asked; after the first divergent round nothing later matches
    and the rest of the cell is bought fresh.
    """

    def __init__(self, inner, cid, sname, persist_rounds: bool = False):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_key", (str(cid), str(sname)))
        object.__setattr__(self, "_persist", bool(persist_rounds))

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)

    def _lookup(self, payload):
        idx = RUN.replay_idx or {}
        sha = _prompt_sha_for(self._inner, payload)
        mode = getattr(self._inner, "prompt_mode", "default")
        try:
            t = int(payload.prediction_context["prediction_time_T"])
        except (KeyError, TypeError, ValueError):
            t = None
        for k in ((*self._key, t), (*self._key, "sha", sha)):
            r = idx.get(k)
            if r is not None and r.get("prompt_sha256") == sha and (r.get("prompt_mode") or "default") == mode:
                return r
        return None

    def solve(self, payload):
        r = self._lookup(payload)
        if r is not None:
            out = _to_output(_extract_json(r["raw"]), payload)
            out._raw_text, out._prompt_mode = r["raw"], r.get("prompt_mode") or "default"
            out._probe_id, out._prompt_sha = r.get("probe_id"), r.get("prompt_sha256")
            out._usage, out._finish = r.get("usage"), r.get("finish_reason")
            out._failed_attempts = r.get("failed_attempts")
            out._latency_s, out._reasoning_chars = r.get("latency_s"), r.get("reasoning_chars")
            out._max_tokens = r.get("max_tokens")
            out._replayed = True
            _REPLAY_N[self._key] = _REPLAY_N.get(self._key, 0) + 1
            return out
        out = self._inner.solve(payload)
        _REPLAY_FRESH[self._key] = _REPLAY_FRESH.get(self._key, 0) + 1
        if self._persist:
            _persist_gated_round(*self._key, out)
        return out


def _replay_solver_for(solver, cid, sname, geometry: str):
    """The resume wrapper for a geometry. Fresh runs (`RUN.replay` false) get no replay; gated
    cells still record their rounds so a later resume has something to reuse."""
    gated = geometry == "gated"
    if not RUN.replay:
        return _ReplaySolver(solver, cid, sname, persist_rounds=True) if gated else solver
    if RUN.replay_idx is None:
        idx = load_slice_replay(RUN.resp_path)
        idx.update(load_gated_replay(gated_rounds_path()))
        RUN.replay_idx = idx
    return _ReplaySolver(solver, cid, sname, persist_rounds=gated)


def take_replay_count(case: str, solver: str) -> dict:
    """Per-cell resume accounting: answers reused (`slices_reused`, name kept for slice rows) and
    the answers bought again in a resumed run (`replay_rebought`); only set when a replay was
    active and something was reused, so a fresh run's rows are unchanged."""
    k = (str(case), str(solver))
    n, fresh = _REPLAY_N.pop(k, 0), _REPLAY_FRESH.pop(k, 0)
    if not n:
        return {}
    return {"slices_reused": n, "replay_rebought": fresh}


def _row_slices(cid, sname, raw, slices, solver) -> dict:
    """Slice geometry: N independent consultations on the same world (no carryover), one answer per slice.

    Unlike `_row_multi`, nothing is carried forward: it tests whether scattered symptoms are
    combined across separate sessions. A leak voids only that slice.
    """
    from .tracks import alternative_a1, _answer_text, _differential
    rows = []
    _, vp0 = build_instance(raw, int(max(slices)))
    is_ddx = bool((vp0.adjudication or {}).get("ddx"))
    real_ids = set(wq.injected_manifest(raw.case_id)      # Q-side ledger
                   .get("real_symptom_evidence_ids", []) or [])
    from .solve_guard import guarded_solve
    solver = _replay_solver_for(solver, cid, sname, "slices")

    def _persist(o, _t):
        if getattr(o, "_replayed", False):        # already on disk; still counts once toward the cell's usage
            _note_grid_usage(cid, sname, o)
            _note_grid_retry(cid, sname, o)
            return
        save_response(RUN.resp_path, cid, sname, o, slice_t=_t)
        save_simple_trace(RUN.resp_path.with_name("trace.jsonl"),
                          cid, sname, "slices", o, slice_t=_t)
    for t in slices:
        sp, vp = build_instance(raw, int(t))
        n_sym = sum(1 for ev in (sp.evidence_ledger or [])      # count of genuine symptoms visible in this slice
                    if str(ev.get("source_type", "")) == "patient_reported_symptom"
                    and ev.get("evidence_id") in real_ids)
        facts = guarded_solve(
            solver, sp, int(t),
            prompt_mode=("ddx" if is_ddx else "default"),
            precheck=iron_law_precheck(raw, sp, vp, solver),
            on_output=(lambda o, _t=int(t): _persist(o, _t)) if RUN.resp_path else None,
            tag=f"{cid}|{sname}@t{int(t)}")
        if facts.leak:
            rows.append({"t": int(t), "n_symptoms": n_sym, "leak": facts.leak, "all_drivers": []})
            continue
        out = facts.out
        # Visible evidence per slice, for judging whether a revision followed new information
        # (`VerifierPayload` has no ledger).
        _vis = [str(ev.get("evidence_id")) for ev in (sp.evidence_ledger or [])]
        # Lab values far out of range are tracked as worsening signals (their `source_type` is not
        # a symptom); only the collection side sees the evidence text.
        _abn = []
        for ev in (sp.evidence_ledger or []):
            if str(ev.get("source_type")) != "lab_result":
                continue
            m = _re_lab.search(str(ev.get("symptom") or ""))
            if not m:
                continue
            try:
                val, lo, hi = float(m.group(2)), float(m.group(4)), float(m.group(5))
            except ValueError:
                continue
            _w = max(hi - lo, 1e-9)
            _out = (lo - val) / _w if val < lo else ((val - hi) / _w if val > hi else 0.0)
            if _out > ABNORMAL_MARGIN:
                _abn.append(str(ev.get("evidence_id")))
        rows.append({"t": int(t), "n_symptoms": n_sym,
                     "visible_real": [e for e in _vis if e in real_ids],
                     "visible_abnormal_lab": _abn,
                     "visible_other": [e for e in _vis if e not in real_ids],
                     "risk_category": out.forecast.get("risk_category"),
                     "all_drivers": [d.get("driver") for d in (out.drivers or [])],
                     "answer_text": " ".join(
                         [_answer_text(out)]
                         + [str(x.get("diagnosis") or "") for x in _differential(out)]),
                     "differential": [x.get("diagnosis") for x in _differential(out)],
                     # Evidence fields read by `judge_refuting_evidence` and `judge_what_not_to_do`.
                     "differential_evidence": [
                         {"ruled_out_by": x.get("ruled_out_by"),
                          "supporting_evidence": x.get("supporting_evidence") or []}
                         for x in _differential(out)
                         if isinstance(x, dict) and x.get("ruled_out_by")],
                     # `None` (not answered) stays distinct from `[]` (answered: nothing).
                     "what_not_to_do": ((out.action or {}).get("what_not_to_do")),
                     "join_type": ((getattr(out, "_raw", None) or {}).get("join_type") or None),
                     "action": (out.action or {}).get("selected_action_class"),
                     # Read by per-slice safety gates; `None` stays distinct from `False`.
                     "clinician_review_required": ((out.action or {})
                                                   .get("clinician_review_required")),
                     "data_sufficiency": ((getattr(out, "data_quality", None) or {})
                                          .get("data_sufficiency")
                                          if isinstance(getattr(out, "data_quality", None), dict)
                                          else None),
                     "tests_to_order": [str(x) for x in
                                        ((getattr(out, "_raw", None) or {}).get("tests_to_order") or [])],
                     "referral_specialty": [str(x) for x in
                                            ((getattr(out, "_raw", None) or {}).get("referral_specialty") or [])],
                     # Whether the slice got a response at all (`_to_output({})` would otherwise look like an
                     # answered slice).
                     "raw_empty": facts.raw_empty,
                     "raw_unparseable": facts.raw_unparseable,
                     "a1": alternative_a1(out)["a1"]})
        _out_last = out
    # No slice answered => ABORT the whole cell, as in single-shot. Unparseable slices are
    # treated as unanswered, so their canned abstention never reaches the judges.
    _bad_slices = [r for r in rows if r.get("raw_unparseable")]
    if _bad_slices:
        log.error("[eval] %s|%s %d/%d slice(s) have non-empty raw text but no extractable "
                  "JSON; those slices are excluded from scoring (the kernel would otherwise "
                  "parse them as {} and record a canned abstention)",
                  cid, sname, len(_bad_slices), len(rows))
        for _bs in _bad_slices:
            for _k in ("answer_text", "differential", "join_type", "action",
                       "data_sufficiency", "risk_category", "all_drivers"):
                _bs[_k] = None
            _bs["slice_unparseable"] = True
    _rw = [r for r in rows if "raw_empty" in r]
    if (_rw and all(r.get("raw_empty") for r in _rw)) or (
            rows and all(r.get("raw_empty") or r.get("raw_unparseable") for r in rows)):
        log.error("[eval] %s|%s all %d slice(s) had no response -> aborting this cell, "
                  "excluded from the denominator (a backend failure is not recorded as an "
                  "abstention)", cid, sname, len(_rw))
        return {"case": cid, "solver": sname, "slices": list(map(int, slices)),
                "overall": "ABORT(no_response)", "n_slices": len(rows),
                "answered_slices": 0, "slices_missing": len(_rw),
                "no_response_reason": "all_slices_empty"}
    # All slices leaked => `ABORT(leak)`, as in the other geometries (a partial leak voids only
    # those slices). Checked after `ABORT(no_response)`.
    _leaked = [r for r in rows if r.get("leak")]
    if rows and len(_leaked) == len(rows):
        _reasons = sorted({str(x) for r in _leaked for x in (r.get("leak") or ())})
        log.error("[eval] %s|%s all %d slice(s) leaked -> ABORT(leak), no score and no "
                  "downgrade: %s",
                  cid, sname, len(rows), _reasons[:3])
        return {"case": cid, "solver": sname, "slices": list(map(int, slices)),
                "overall": "ABORT(leak)", "n_slices": len(rows),
                "answered_slices": 0, "leak": _reasons,
                "leak_scope": "all_slices"}
    sp_full, vp_full = build_instance(raw, int(max(slices)))
    ddx = (vp_full.adjudication or {}).get("ddx")
    _out_last = locals().get("_out_last")
    # If the last slice is invalid, the `last`-subject judges get no subject rather than an
    # earlier slice.
    _last_row = rows[-1] if rows else None
    if _last_row is not None and (_last_row.get("leak") or _last_row.get("slice_unparseable")
                                  or _last_row.get("raw_empty")):
        _out_last = None
    # Hard gates run on the last slice. The kernel's five tracks are recorded too, as diagnostics
    # (the authority here is `wk_*`), with `tracks_basis` = `last_slice`. The two review gates
    # are judged per slice (`slices_gates`), so they stay off this case-level list.
    _gates: list = []
    _tracks_last = None
    if _out_last is not None:
        try:
            _rep_last = verifier_mod.grade(
                _out_last, vp_full,
                [e["evidence_id"] for e in (sp_full.evidence_ledger or [])],
                unit_gates_per_slice=True)
            _gates = list(getattr(_rep_last, "hard_gate_failures", None) or [])
            _tracks_last = getattr(_rep_last, "tracks", None)
        except Exception as e:                          # noqa: BLE001
            log.error("[eval] %s|%s final-slice kernel scoring failed -> recording "
                      "gate_unknown: %s", cid, sname, e)
            _gates = None
    # Process judges on the last slice (`sp_full`'s ledger is its visible set); `{}` if the
    # last slice is invalid.
    _proc = process.run_process_judges(
        getattr(_out_last, "_raw", None),
        [e["evidence_id"] for e in (sp_full.evidence_ledger or [])])
    from .judges import run_judges, SLICES
    return {"case": cid, "solver": sname, "slices": list(map(int, slices)),
            "gold_driver": (vp_full.gold_drivers or [None])[0],
            "ddx_diagnosis": (ddx or {}).get("diagnosis"),
            "tracks": _tracks_last, "tracks_basis": "last_slice",
            **run_judges(SLICES, {"rows": rows, "last": _out_last}, vp_full),
            **_proc,
            **_premise_row(cid, _out_last),
            **judges_mod.judge_noop_probe(_out_last, NOOP_FOR.get(str(cid))),
            **judges_mod.judge_quant_probe(_out_last, QUANT_FOR.get(str(cid)),
                                           t_max=int(max(slices)) if slices else None),
            # Abstention uses the per-slice judge (`slices_abstention`, inside `run_judges`); diagnosis
            # judges receive the last slice via the `last` subject.
            "slice_rows": rows,
            # `gates`: non-empty = hit, `[]` = no hit, key absent = not graded (`gate_unknown`).
            **({"gates": _gates} if _gates is not None else {}),
            "overall": "FAIL(gate)" if _gates else "SCORED"}


#: Deciles of real clinic visit intervals (days), from a real hypothyroidism EMR sample
#: (965 patients, 28,814 intervals): median 28, Q1-Q3 8-68, >180 days 8.9%.
REAL_VISIT_DECILES: tuple[int, ...] = (3, 7, 12, 21, 28, 37, 56, 87, 163)


#: Slices per case, fixed (the order of magnitude of `auto+neutral`) so per-case cost and
#: judge counts are comparable.
REAL_RHYTHM_SLICES = 7

#: Length (days) of the forced long interval in the "information gap" tier. It is an explicit
#: tier because window truncation makes random sampling undersample long intervals.
REAL_RHYTHM_GAP_DAYS = 200


def use_real_rhythm(T: int) -> bool:
    """Whether this horizon uses the real clinic rhythm; same threshold as
    `rhythm_gap_feasible`.
    """
    return rhythm_gap_feasible(T)


def rhythm_gap_feasible(T: int, n_slices: int | None = None) -> bool:
    """Whether this horizon can fit the forced long-gap tier; shared by case generation and
    run time.
    """
    n = int(n_slices if n_slices is not None else REAL_RHYTHM_SLICES)
    return int(T) >= int(REAL_RHYTHM_GAP_DAYS) + n


def _real_rhythm_days(case_id: str, T: int, symptom_days: list[int],
                      force_gap: bool = False) -> list[int]:
    """Schedule slices by real clinic visit intervals rather than symptom days.

    `REAL_RHYTHM_SLICES` intervals are drawn from `REAL_VISIT_DECILES` and normalized to the
    window; `force_gap` adds one `REAL_RHYTHM_GAP_DAYS` interval. Seeded by `case_id`, so the
    schedule is reproducible.
    """
    import hashlib
    import random
    rnd = random.Random(int(hashlib.sha256(str(case_id).encode()).hexdigest()[:12], 16))
    edges = (1,) + REAL_VISIT_DECILES + (400,)
    n = max(2, int(REAL_RHYTHM_SLICES))

    def _gap() -> int:
        i = rnd.randrange(len(edges) - 1)                  # equal probability per bucket, uniform within a bucket
        lo, hi = edges[i], max(edges[i], edges[i + 1])
        return rnd.randint(lo, hi)

    raw_gaps = [_gap() for _ in range(n)]
    gap_i = -1
    if force_gap:
        # The forced gap is excluded from normalization (which would shrink it); the other
        # intervals share the remaining window.
        if not rhythm_gap_feasible(T, n):
            log.warning("[slices] %s: declared rhythm_gap but T=%d cannot fit it "
                        "(needs >= %d) ⇒ this case has no long-gap tier",
                        case_id, T, int(REAL_RHYTHM_GAP_DAYS) + n)
            force_gap = False
        else:
            gap_i = len(raw_gaps) // 2
            raw_gaps[gap_i] = int(REAL_RHYTHM_GAP_DAYS)
    budget = (T - 1) - (int(REAL_RHYTHM_GAP_DAYS) if gap_i >= 0 else 0)
    others = [g for i, g in enumerate(raw_gaps) if i != gap_i]
    total = sum(others) or 1
    scale = max(0.0, budget) / total
    days: list[int] = []
    acc = 0.0
    for i, g in enumerate(raw_gaps):
        acc += (int(REAL_RHYTHM_GAP_DAYS) if i == gap_i else g * scale)
        d = int(round(acc))
        if days and d <= days[-1]:
            d = days[-1] + 1                               # no overlap / no going backward allowed
        if d > T:
            break
        days.append(d)
    days = [d for d in days if 0 < d <= T]
    if len(days) < 2:
        return symptom_days or days
    return days


def slices_for(raw, spec) -> list[int]:
    """Slice geometry: this case's query time points (all <=T).

    `auto`: the days real symptoms appear; `auto+neutral`: plus neutral points; `real-rhythm`:
    real clinic intervals; an explicit list is used as given.
    """
    T = int(raw.prediction_context["prediction_time_T"])
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec if int(x) <= T]
    # Only real symptoms (from the Q-side ledger, not W's adjudication), not injected benign
    # events.
    real = set(wq.injected_manifest(raw.case_id)
               .get("real_symptom_evidence_ids", []) or [])
    def _days(ids_filter) -> list[int]:
        return sorted({int(e.get("source_timestamp", -1)) for e in (raw.evidence_ledger or [])
                       if ids_filter(e)
                       and str(e.get("source_type", "")) == "patient_reported_symptom"
                       and 0 < int(e.get("source_timestamp", -1)) <= T})

    days = _days(lambda e: (not real or e.get("evidence_id") in real))
    if str(spec).strip().startswith("real-rhythm"):
        # The gap tier is tagged per case (`meta.rhythm_gap`); `+gap` in the spec is the master
        # switch.
        _meta = (getattr(raw, "latent_premise", None) or {}).get("meta") or {}
        _fg = bool("+gap" in str(spec)) and bool(_meta.get("rhythm_gap"))
        # Too short a horizon would make the normalized rhythm denser than `auto+neutral`; fall back
        # to `auto` (logged).
        if not use_real_rhythm(T):
            log.warning("[slices] %s: declared real-rhythm but T=%d < %d ⇒ "
                        "the rhythm would be denser than auto (median ~T/13 days), "
                        "falling back to auto",
                        raw.case_id, T, int(REAL_RHYTHM_GAP_DAYS) + int(REAL_RHYTHM_SLICES))
        else:
            return _real_rhythm_days(raw.case_id, T, days, force_gap=_fg)
    if str(spec).strip() != "auto+neutral":
        return days

    # ---- `auto+neutral` ----
    # Adds one slice between consecutive real-symptom slices when only benign events occur in
    # between, so neutral transitions (no new real symptom) exist to judge belief drift.
    benign = _days(lambda e: bool(real) and e.get("evidence_id") not in real)
    out = []
    for i, d in enumerate(days):
        out.append(d)
        nxt = days[i + 1] if i + 1 < len(days) else T + 1
        mid = [b for b in benign if d < b < nxt]
        if mid:
            out.append(max(mid))            # take the last benign time point in this segment: the most benign events have accumulated by then
    return sorted(set(out))


def _row_multi(cid, sname, raw, solver, cadence) -> dict:
    from .judges import MULTI
    from .mounting import RoundRecorder as _RoundRecorder
    _Tm = int(raw.prediction_context["prediction_time_T"])
    _spm, _vpm = build_instance(raw, _Tm)
    solver.prompt_mode = "ddx" if (_vpm.adjudication or {}).get("ddx") else "default"
    # `RoundRecorder` persists each round, checks it for empty/unparseable responses, and lets
    # a per-round `LeakageError` surface as `ABORT(leak)`.
    def _save_round(i, out, _f):
        if not RUN.resp_path:
            return
        if getattr(out, "_replayed", False):      # already on disk from the run that died
            _note_grid_usage(cid, sname, out)
            _note_grid_retry(cid, sname, out)
            return
        save_response(RUN.resp_path, cid, sname, out, rounds=[i])
        save_simple_trace(RUN.resp_path.with_name("trace.jsonl"), cid, sname, "multi",
                          out, step=int(i))

    _rec = _RoundRecorder(_replay_solver_for(solver, cid, sname, "multi"), on_round=_save_round)
    try:
        traj, grade, tE = runner_mod.run_multiround(raw, _rec, cadence=cadence)
    except runner_mod.LeakageError as e:
        log.error("[eval] %s|%s multi-round leak -> ABORT(leak): %s", cid, sname, e)
        return {"case": cid, "solver": sname, "geometry": "multi",
                "rounds": len(_rec.outputs), "overall": "ABORT(leak)",
                "abort_detail": str(e)[:300]}
    _bad = _rec.abort_reason()
    if _bad is not None:
        log.warning("[eval] %s|%s multi-round: %d round(s), %d bad -> %s",
                    cid, sname, len(_rec.facts), len(_rec.bad_rounds), _bad)
        return {"case": cid, "solver": sname, "geometry": "multi",
                "rounds": len(traj), "overall": _bad,
                "bad_rounds": _rec.bad_rounds}
    _out_last = _rec.outputs[-1] if _rec.outputs else None
    return {"case": cid, "solver": sname, "rounds": len(traj),
            # The kernel's `repair` tag means "belief changed between adjacent rounds", hence the
            # name.
            "n_belief_changed": sum(1 for r in traj if r["repair"] == "revised"),
            # noop/quant probes are judged on the last round. The trajectory judges come from the `multi`
            # column of `mount_table.MOUNT` via `run_judges`.
            **(judges_mod.judge_noop_probe(_out_last, NOOP_FOR.get(str(cid)))
               if _out_last is not None else
               {"multi_last_missing": "the recorder captured no answer in any round -> "
                                      "the last-round family is unmeasured (not folded to 0)"}),
            **(judges_mod.judge_quant_probe(_out_last, QUANT_FOR.get(str(cid)), t_max=_Tm)
               if _out_last is not None else {}),
            # Single-shot-family judges get the last round output (`ctx["round_outputs"][-1]`), the
            # trajectory judges get `traj`; a missing subject is recorded, never substituted.
            **judges_mod.run_judges(MULTI, {"traj": traj}, _vpm,
                                    ctx={"round_outputs": _rec.outputs,
                                         "premise": PREMISE_FOR.get(str(cid))}),
            **process.run_process_judges(
                getattr(_out_last, "_raw", None),
                [e["evidence_id"] for e in (_spm.evidence_ledger or [])]),
            "trajectory": traj, "trackE": tE["E"], "latencies": tE["latencies"],
            "trap_fooled": tE["trap_fooled"], "n_real": tE["n_real"], "n_trap": tE["n_trap"],
            "tracks": grade.tracks, "gates": grade.hard_gate_failures, "overall": grade.overall}


# ---------------------------------------------------------------- parallel execution
# Rate limiting, locking and write ordering live together here.

_WRITE_LOCK = _threading.Lock()   # lock for appends: two lines interleaving = one line of broken JSON

# Batch ERROR share at which the summary is logged as an error: at 25 % the cause is code or
# configuration, not one model.
ERROR_RATE_WARN = 0.25

_MODEL_BACKENDS: dict | None = None


def _backend_of(solver_name: str) -> str:
    """The backend this solver runs on, from `config.models`; unknown names are offline (unthrottled)."""
    global _MODEL_BACKENDS
    if _MODEL_BACKENDS is None:
        try:
            import yaml
            from pathlib import Path as _P
            _cfg = yaml.safe_load((_dr() / "config.yaml")
                                  .read_text(encoding="utf-8")) or {}
            _ov = os.environ.get("HAENV_CONFIG_OVERLAY", "").strip()   # same overlay as `cli.load_cfg`
            if _ov:
                _ov_models = (yaml.safe_load(_P(_ov).read_text(encoding="utf-8")) or {}).get("models") or {}
                for _n, _s in _ov_models.items():
                    _cfg.setdefault("models", {})[_n] = {**(_cfg.get("models", {}).get(_n) or {}), **_s}
            _MODEL_BACKENDS = {}
            for k, v in (_cfg.get("models") or {}).items():
                _MODEL_BACKENDS[str(k)] = (str(v.get("backend")) if isinstance(v, dict)
                                           else "_cli")
        except Exception as e:
            log.error("[eval] could not read config.models, per-backend throttling will "
                      "degrade to a single tier: %s", e)
            _MODEL_BACKENDS = {}
    return _MODEL_BACKENDS.get(solver_name, "_offline")


def _run_tasks(fn, tasks: list, out_path, key_of=None, geometry_of=None) -> list[dict]:
    """Run every cell; serial when `RUN.workers <= 1`, otherwise scheduled by
    `run_scheduler` (one pool per model under backend and global caps). Rows are written as
    they complete, so a killed run keeps every finished cell; the returned list is in task
    order.
    """
    def _emit(row: dict) -> dict:
        """Persist and flush one row. Also stamps `judging_sha16`, so a resumed run that spans a
        judging-code change can be detected (`report.rank_ddx`).
        """
        from .anchor import judging_fp_cached
        row.setdefault("judging_sha16", judging_fp_cached())
        row.setdefault("geometry", "<unrecorded>")
        with _WRITE_LOCK:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        return row

    from .run_scheduler import run as _schedule
    return _schedule(fn, tasks, _emit, workers=max(1, int(RUN.workers)),
                     key_of=key_of, backend_of=_backend_of if key_of else None,
                     geometry_of=geometry_of, batch_dir=Path(out_path).parent)


#: Per-cell unit cost in USD (~20k-token prompts) for relay-billed models; a rough
#: affordability check, not billing.
RELAY_COST_PER_GRID = {"kimi-k3": 0.2374, "terra": 0.0592}
#: Unit cost for unlisted relay-billed models (errs high).
RELAY_COST_DEFAULT = 0.15
RELAY_COST_HEADROOM = 2.0


def billed_tokens(usage: dict | None) -> dict:
    """Normalize each backend's usage into fresh input / cached input / output.

    With `billing_usage.claude_usage` (some relays), `input + cache_read == prompt_tokens`.
    Otherwise `cached = prompt_tokens_details.cached_tokens or 0` and
    `fresh = prompt_tokens - cached`.
    """
    u = usage or {}
    out = int(u.get("completion_tokens") or 0)
    cu = (u.get("billing_usage") or {}).get("claude_usage") or {}
    fresh_c, cached_c = cu.get("input_tokens"), cu.get("cache_read_input_tokens")
    if isinstance(fresh_c, int) and isinstance(cached_c, int):
        fresh, cached = fresh_c, cached_c
    else:
        total = int(u.get("prompt_tokens") or 0)
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        fresh = max(0, total - cached)
    # Thinking tokens (Google `thoughtsTokenCount`) count as output; the unaccounted
    # residual is kept.
    _th = int(u.get("thoughts_tokens") or 0)
    _tot = u.get("total_tokens")
    return {"in_fresh": fresh, "in_cached": cached,
            "out": out + _th, "out_thoughts": _th or None,
            "in_total": fresh + cached,
            "unaccounted": (int(_tot) - (fresh + cached) - out - _th
                            if isinstance(_tot, int) else None),
            "status": "measured" if u else "absent"}


def usage_coverage(rows: list[dict]) -> dict:
    """Usage coverage for a batch of `eval.jsonl` rows, per solver and in total.

    Returns `{status, n_rows, in_total, out, total_tokens, by_solver, real_models_missing}`.
    `status=absent` is normal for an offline batch; a non-empty `real_models_missing` (a real
    model with no measured cell) means the accounting broke.
    """
    from .baselines import BASELINE_NAMES as _BN
    by: dict[str, dict] = {}
    for r in rows:
        s = str(r.get("solver") or r.get("model") or "")
        if not s:
            continue
        e = by.setdefault(s, {"n_rows": 0, "n_measured": 0, "in_total": 0, "out": 0,
                              "missing": {}, "is_stub": s in _BN})
        e["n_rows"] += 1
        st = str(r.get("usage_status") or "")
        if st == "measured":
            b = r.get("billed") or {}
            e["n_measured"] += 1
            e["in_total"] += int(b.get("in_total") or 0)
            e["out"] += int(b.get("out") or 0)
        else:
            k = st or "unrecorded"
            e["missing"][k] = e["missing"].get(k, 0) + 1
    for e in by.values():
        e["status"] = "measured" if e["n_measured"] else "absent"
    _in = sum(e["in_total"] for e in by.values())
    _out = sum(e["out"] for e in by.values())
    _bad = sorted(s for s, e in by.items() if not e["is_stub"] and not e["n_measured"])
    return {"status": "measured" if any(e["n_measured"] for e in by.values()) else "absent",
            "n_rows": sum(e["n_rows"] for e in by.values()),
            "in_total": _in, "out": _out, "total_tokens": _in + _out,
            "by_solver": {s: {k: v for k, v in e.items() if k != "is_stub"}
                          for s, e in sorted(by.items())},
            "real_models_missing": _bad}


#: Normalized rate (USD per million tokens) for the affordability check; errs high, since a
#: run that breaks midway leaves a partial reading.
RELAY_USD_PER_MTOK = 4.0


def _measured_tokens_per_grid(job, models: list[str]) -> dict[str, float]:
    """Each model's measured tokens per cell from this job's earlier batches (from `billed`,
    else `usage`); empty if there are none.
    """
    try:
        from pathlib import Path as _P
        import json as _j
        _root =_dr()
        base = _root / "results" / getattr(job, "task_type", "") / getattr(job, "job_id", "")
        if not base.is_dir():
            return {}
        # The four most recent batches that contain the estimated models (offline smoke batches
        # would otherwise crowd them out).
        cands = [d for d in base.iterdir()
                 if d.is_dir() and (d / "responses.jsonl").is_file()]
        agg: dict[str, dict] = {}
        _want = {str(m) for m in (models or [])}
        _with_usage = []
        for _d in sorted(cands, key=lambda x: x.name, reverse=True):
            try:
                with (_d / "responses.jsonl").open(encoding="utf-8", errors="ignore") as _fh:
                    if any(_l.strip() and any(f'"{m}"' in _l for m in _want) for _l in _fh):
                        _with_usage.append(_d)
            except Exception:                          # noqa: BLE001
                continue
            if len(_with_usage) >= 4:
                break
        for d in _with_usage:
            rp = d / "responses.jsonl"
            for line in rp.read_text(encoding="utf-8", errors="ignore").split("\n"):
                if not line.strip():
                    continue
                try:
                    r = _j.loads(line)
                except Exception:                               # noqa: BLE001
                    continue
                if r.get("solver") not in models:
                    continue
                b = r.get("billed") or billed_tokens(r.get("usage"))
                e = agg.setdefault(r["solver"], {"tok": 0, "grids": set()})
                e["tok"] += int(b.get("in_total", 0)) + int(b.get("out", 0))
                e["grids"].add((r.get("case"), d.name))
        return {m: e["tok"] / max(1, len(e["grids"])) for m, e in agg.items() if e["tok"]}
    except Exception as e:                                      # noqa: BLE001
        log.warning("[quota] failed to compute tokens live from historical batches (%s) —— "
                    "falling back to the unit-price table", e)
        return {}


def _preflight_quota(job, cfg, solvers, n_cases: int, n_done: int) -> None:
    """Relay balance gate; the logic is `relay_accounting.preflight_quota` (accounting code)."""
    from .relay_accounting import preflight_quota
    preflight_quota(job, cfg, solvers, n_cases, n_done)


#: Name of the current batch. `case_id` is not unique across jobs, so every per-batch dict
#: is cleared when a different job starts in the same process.
_BATCH_SCOPE: list[str | None] = [None]

#: Per-batch dicts cleared by `enter_batch`. `RUN` (`RunState`) is overwritten by
#: `run_eval`; `_KEY_POOLS` and `_ATTR_MOD` are
#: process-wide by design.
PER_BATCH_DICTS: tuple[str, ...] = (
    "PREMISE_FOR", "NOOP_FOR", "QUANT_FOR", "QUANT_TRUTH_DIST",
    "ORACLE_GOLD_TESTS", "ORACLE_DISC_NAMES", "ORACLE_WARRANTED",
    "ORACLE_DIAGNOSIS",
    "_USAGE_ACC", "_RETRY_ACC",
)


def enter_batch(job_id: str) -> str | None:
    """Enter a batch: clear the previous batch's mutable state and record the new name; returns
    the previous id. A resumed run of the same job does not clear.
    """
    prev = _BATCH_SCOPE[0]
    if prev is not None and prev != job_id:
        g = globals()
        for name in PER_BATCH_DICTS:
            g[name].clear()
        log.info("[eval] batch switch %s -> %s: cleared %d per-batch state dict(s)",
                 prev, job_id, len(PER_BATCH_DICTS))
    _BATCH_SCOPE[0] = job_id
    return prev


def run_eval(job, cfg, built: dict, resume: bool = True, *,
             budget_ledger: Path | None = None, budget_usd: str | None = None) -> list[dict]:
    """Evaluate `built` ({case_id: RawCase}) cell by cell, persisting each row; completed cells
    are skipped (resumable).
    """
    enter_batch(str(getattr(job, "job_id", "") or getattr(job, "path", "") or "?"))
    out_path = job.results_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    RUN.resp_path = out_path.parent / "responses.jsonl"      # raw responses persist in the same batch as eval
    from .batch import provenance_fields
    _wref = provenance_fields(cfg, job_path=getattr(job, "path", None), root=job.root)
    RUN.probes = load_probes(job.root / "probes")
    RUN.allow_retired = bool(getattr(job, "allow_retired", False))
    _pid = getattr(job, "probe_id", "") or ""
    if _pid and _pid not in RUN.probes:
        raise KeyError(f"job.probe_id={_pid!r} is not registered (registered: {sorted(RUN.probes)})")
    log.info("[eval] %d probe(s) registered · this batch's framing is %s",
             len(RUN.probes), _pid or "(defaults by question type to " + str(DEFAULT_PROBE) + ")")
    _wref = {**_wref, "probe_id": _pid or None}
    # The top-level `world_sha`/`world_knobs` are the batch's sticky (case-generation) values;
    # `world_ref` holds the solve-time version, and `world_sha_at_solve` is recorded when they
    # differ. Without a sticky value, `world_sha_src` records the fallback.
    _wsha = _wknobs = None
    _wsrc = "batch_sticky"
    try:
        _bj = json.loads((out_path.parent / "batch.json").read_text(encoding="utf-8"))
        _wsha, _wknobs = _bj.get("world_sha"), _bj.get("world_knobs")
    except (OSError, json.JSONDecodeError):
        _bj = {}
    if not _wsha:
        _wsha, _wknobs = _wref.get("world_sha"), _wref.get("world_knobs")
        _wsrc = "fallback:solve_time"
    _wtop = {k: v for k, v in
             (("world_sha", _wsha), ("world_knobs", _wknobs),
              ("world_sha_src", _wsrc if _wsha else None)) if v}
    if _wsha and _wref.get("world_sha") and _wsha != _wref.get("world_sha"):
        _wtop["world_sha_at_solve"] = _wref["world_sha"]
        log.warning("[eval] generation-time world %s != solve-time world %s; the question "
                    "text comes from the former (read from cases.jsonl); both are persisted",
                    _wsha, _wref["world_sha"])
    # Slicing is a case-generation decision, frozen into the batch (see `haenv/slicing.py`),
    # so a resumed run cannot use different slices.
    _slice_map: dict[str, list[int]] = {}
    _slice_drift: list[dict] = []
    if getattr(job, "slices", ""):
        from . import slicing as _slicing
        _batch_dir = out_path.parent
        _was_spec = _slicing.check_spec(_batch_dir, str(job.slices))
        if _was_spec:
            raise ValueError(
                f"slicing spec mismatch: this batch froze spec={_was_spec!r}, but the job "
                f"declares {str(job.slices)!r}. Slices cut by different specs (auto / "
                f"auto+neutral / real-rhythm / an explicit list) are not comparable; start a "
                f"new batch (--fresh) to change the spec.")
        _recovered = _slicing.recover_from_disk(_batch_dir)
        _has_rows = bool(_recovered) or (out_path.is_file() and out_path.stat().st_size > 0)
        _existing = _slicing.load(_batch_dir)
        for _cid, _raw in built.items():
            _fresh = slices_for(_raw, job.slices)
            if _existing is not None:
                _was = _slicing.drift(_batch_dir, _cid, _fresh)
                if _was is not None:
                    log.warning("[slicing] %s slice drift: %d slice(s) on disk %s vs %d "
                                "recomputed %s; using the on-disk value",
                                _cid, len(_was), _was, len(_fresh), _fresh)
                    _slice_drift.append({"case": _cid, "frozen": _was, "recomputed": _fresh})
                _slice_map[_cid] = _slicing.resolve(_batch_dir, _cid, lambda: _fresh)
                continue
            if _cid in _recovered:
                _rec = _recovered[_cid]
                if list(_rec) != list(_fresh):
                    log.warning("[slicing] %s backfilled %d slice(s) %s from disk "
                                "(recomputed gives %d %s); using the backfilled value, the "
                                "geometry this batch's rows ran under",
                                _cid, len(_rec), _rec, len(_fresh), _fresh)
                    _slice_drift.append({"case": _cid, "recovered": _rec, "recomputed": _fresh})
                _slice_map[_cid] = _rec
                continue
            # A batch with rows whose slicing cannot be recovered fails: a fresh computation could
            # change the case's geometry.
            if _has_rows:
                raise _slicing.SlicesMissing(
                    f"{_cid}: this batch already has artifacts, but its original slicing "
                    f"cannot be backfilled from eval.jsonl / responses.jsonl, while "
                    f"recomputing gives {len(_fresh)} slice(s). Substituting the recomputed "
                    f"value could change this case's denominator and geometry tier (the case "
                    f"may have had <2 slices and taken the single-shot path). Start a new "
                    f"batch (--fresh), or find out why this case has no per-slice record.")
            _slice_map[_cid] = _fresh
        _slicing.save(_batch_dir, _slice_map, spec=str(job.slices))
        if _slice_drift:
            _wref = {**_wref, "slice_drift_n": len(_slice_drift)}
            try:
                (_batch_dir / "slice_drift.json").write_text(
                    json.dumps({"n": len(_slice_drift), "cases": _slice_drift},
                               ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError as e:
                log.error("[slicing] could not write the drift ledger: %s; this batch's "
                          "drift is recorded only in the log", e)
    # The trend question must be asked on the window actually answered; re-check against the
    # settled slicing (a frozen or backfilled table can differ from a fresh computation).
    _resync_trend_windows(job, built, _slice_map)
    if not resume and out_path.exists():
        out_path.unlink()
    if not resume:
        (out_path.parent / GATED_ROUNDS_FILE).unlink(missing_ok=True)
    RUN.replay, RUN.replay_idx = bool(resume), None
    _REPLAY_N.clear()
    _REPLAY_FRESH.clear()
    done = _done_keys(out_path) if resume else set()
    solvers = build_solvers(job, cfg)
    accounting = None
    if (budget_ledger is None) != (budget_usd is None):
        raise ValueError("Solver budget requires both the shared ledger and its approved cap")
    if budget_ledger is not None:
        from .solver_accounting import prepare_accounting
        accounting = prepare_accounting(solvers, cfg, out_path.parent,
                                        ledger_path=budget_ledger, limit_usd=budget_usd,
                                        identity=_wref)
    log.info("[eval] %d case x %d solver = %d cell(s) (%d already done)",
             len(built), len(solvers), len(built) * len(solvers), len(done))
    _preflight_quota(job, cfg, solvers, n_cases=len(built), n_done=len(done))

    _tasks: list[tuple] = []
    for cid, raw in built.items():
        T = int(raw.prediction_context["prediction_time_T"])
        for sname, make in solvers:
            key = f"{cid}|{sname}"
            if key in done:
                log.info("[eval] skip %s (done)", key); continue
            _tasks.append((cid, raw, T, sname, make, key))

    def _one(task) -> dict:
        cid, raw, T, sname, make, key = task
        solver = make()      # freshly created per case: a stateful baseline must not carry beliefs over from the previous case
        if accounting is not None:
            accounting.bind(solver, cid, sname)
        solver.probe_id = _pid
        from .semantic_budget import BudgetExceeded
        try:
            # `_geom` is the branch actually taken (a slice count < 2 falls back to single-shot); the
            # geometry list lives in `mount_table.GEOMETRY_TABLE`.
            from .mount_table import claim_geometry as _claim
            _n_sl = len(_slice_map[cid]) if getattr(job, "slices", "") else None
            _geom = _claim(job, _n_sl)
            if _geom == "gated":                              # tool track: on-demand query geometry
                row = _row_gated(cid, sname, raw, T, solver)
            elif _geom == "slices":                           # multi-slice query
                row = _row_slices(cid, sname, raw, _slice_map[cid], solver)
            elif _geom == "multi":
                row = _row_multi(cid, sname, raw, solver, job.cadence_days)
            else:
                row = _row_single(cid, sname, raw, T, solver)
            row.setdefault("geometry", _geom)
        except BudgetExceeded:
            raise
        except Exception as e:                # a single cell's failure must not drag down the whole round
            import traceback as _tb
            _trace = _tb.format_exc()
            log.error("[eval] %s failed: %s\n%s", key, e, _trace)
            row = {"case": cid, "solver": sname, "error": str(e),
                   "error_type": type(e).__name__, "traceback": _trace,
                   "overall": "ERROR"}
        row["world_ref"] = _wref
        row.update(_wtop)
        row.update(take_grid_usage(cid, sname))
        row.update(take_replay_count(cid, sname))
        row.update(take_grid_retry(cid, sname, overall=row.get("overall")))
        log.info("[eval] %-28s -> %s", key, row.get("overall"))
        return row

    from .mount_table import claim_geometry as _claim_g
    rows = _run_tasks(_one, _tasks, out_path, key_of=lambda t: t[3],
                      geometry_of=lambda t: _claim_g(
                          job, len(_slice_map[t[0]]) if getattr(job, "slices", "") else None))
    _rs = [r for r in rows if r.get("slices_reused")]
    if _rs:
        log.info("[eval] resume replay: %d restarted cell(s) reused %d saved answer(s) and bought "
                 "%d again (the remaining cells started from nothing or were skipped as done)",
                 len(_rs), sum(int(r["slices_reused"]) for r in _rs),
                 sum(int(r.get("replay_rebought") or 0) for r in _rs))
    if accounting is not None:
        from .semantic_budget import cost_summary
        log.info("[eval] paid solver calls (shared ledger): %s", cost_summary(accounting.ledger.snapshot()))

    # Per-cell errors are caught, so a systemic bug is reported here at batch level (not
    # blocking).
    _err = [r for r in rows if str(r.get("overall") or "") == "ERROR"]
    if _err and rows:
        _rate = len(_err) / len(rows)
        _kinds: dict[str, int] = {}
        for r in _err:
            _kinds[str(r.get("error") or "?")[:80]] = _kinds.get(str(r.get("error") or "?")[:80], 0) + 1
        (log.error if _rate >= ERROR_RATE_WARN else log.warning)(
            "[eval] this run: %d/%d cell(s) = %.1f%% judged ERROR%s · causes: %s",
            len(_err), len(rows), _rate * 100,
            ("(above the warning line: do not read this batch as a result)"
             if _rate >= ERROR_RATE_WARN else ""),
            _kinds)
    _all = load_rows(out_path)          # read the full aggregate (including rows from earlier resumed runs), deduplicated by cell
    try:
        from .batch import record_usage
        record_usage(job, _all)
    except Exception as e:                                      # noqa: BLE001
        log.error("[eval] could not write the batch usage rollup: %s; this batch's spend "
                  "is recorded only per row", e)

    # Read the trace back and validate its invariants once per batch; violations are logged,
    # not raised.
    if RUN.resp_path:
        _tp = RUN.resp_path.with_name("trace.jsonl")
        if _tp.is_file():
            try:
                from .trace import read_trace as _rt, check_invariants as _ci
                # Named `_trace_bad`: the maintainers' taint check tracks the name `_bad` module-wide.
                _trace_bad = _ci(_rt(_tp))
                if _trace_bad:
                    log.error("[trace] %d trace invariant violation(s) (first 3: %s); "
                              "the trace on disk is not self-consistent, so replay and "
                              "display cannot rely on it",
                              len(_trace_bad), _trace_bad[:3])
                else:
                    log.info("[trace] trace is self-consistent ✓ (%s)", _tp.name)
            except Exception as e:                              # noqa: BLE001
                log.error("[trace] trace could not be read back: %s (a write-side or "
                          "vocabulary error, distinct from a missing trace)", e)
    return _all


def load_rows(path: Path) -> list[dict]:
    """Read back a batch's result rows, keeping the last row per cell (a retried cell has two).
    Earlier rows stay in the file; statistics use this function.
    """
    out: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            for old_key, new_key in RENAMED_ROW_KEYS.items():
                if old_key in r and new_key not in r:
                    r[new_key] = r.pop(old_key)
            out[(r.get("case"), r.get("solver"))] = r      # later rows overwrite earlier ones
    return list(out.values())


#: Row keys renamed after batches were written with the old name. Stored batches are
#: read-only, so the old key is mapped here, at the one entry point that reads them.
RENAMED_ROW_KEYS: dict[str, str] = {"rival_recall_ang": "rival_recall_capped2"}
