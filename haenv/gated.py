"""gated.py -- the `gated` execution geometry: non-primary signals sit behind the kernel's
`Gatekeeper`, and the model queries them from a priced menu within a budget, feeding the
Tools track (T1 grounding, T2 redundancy, T3 budget, T4 concluding without a key signal).

T4 needs a declared key signal (`build.GOLD_EVIDENCE`); diagnostic items have none, so T4 is
None (not applicable) there, not 0.

SYNTHETIC, eval-only, not medical advice.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml

import copy
import functools
import json
import logging
import pathlib
from dataclasses import dataclass, field
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data

log = logging.getLogger("haenv.gated")

# Weight is the predicted quantity, so it always stays in the prompt.
ALWAYS_VISIBLE = ("weight",)

# Rounds and budget per cell. The budget must force a trade-off; see `budget_for`.
DEFAULT_BUDGET = 120.0            # sentinel: run_gated derives the budget via budget_for() unless another value is passed
BUDGET_FRAC = 0.40                # budget_for()'s `frac` default; the derivation uses `SIGNAL_ALLOWANCE` / `TEST_ALLOWANCE`
MAX_ROUNDS = 6

# Real clinical signals this patient does not have, mixed into the menu so ungrounded
# queries are possible (T1). Names only: prices come from `registry/gated_pricing.yaml`.
DECOY_SIGNALS: tuple[str, ...] = (
    "egfr_slope", "hba1c_series", "cortisol_am", "psg_ahi",
    "thyroid_us", "adrenal_ct", "iron_sat", "bone_density",
)

class PricingUnregistered(KeyError):
    """A menu target has no registered price (never defaulted to the cheapest tier)."""


@functools.lru_cache(maxsize=1)
def _pricing() -> dict[str, str]:
    """`target -> kind` from `registry/gated_pricing.yaml` and the generated stream prices."""
    import yaml as _y
    _p =_dr() / "registry" / "gated_pricing.yaml"
    reg = _cached_yaml(_p) or {}
    out: dict[str, str] = {}
    # Streams: generated from the stream manifest by tools/gen_stream_tables.py.
    _ps = _cached_yaml(_dr() / "registry" / "gated_pricing_streams.yaml") or {}
    for t, k in (_ps.get("streams") or {}).items():
        out[str(t)] = str(k)
    for grp in ("tests", "decoy_signals", "discriminator_tests"):
        for t, k in (reg.get(grp) or {}).items():
            out[str(t)] = str(k)
    return out


def kind_of(target: str, withheld: dict | None = None) -> str:
    """Query kind (and so unit price) for a signal, from the pricing registry; raises
    `PricingUnregistered` rather than inferring from the name."""
    reg = _pricing()
    t = str(target)
    if t in reg:
        return reg[t]
    raise PricingUnregistered(
        f"{t!r} has no price in registry/gated_pricing.yaml -- "
        f"a new test item / signal requires an explicit pricing decision. "
        f"An unpriced item does not default to the vitals tier.")


def test_catalogue() -> list[str]:
    """Union of gold `tests` across all conditions, deduped and sorted. The catalogue is far
    larger than any one case's gold, so it offers options, not an answer key."""
    import haenv_kernel.joint_scenarios as _JS    # kernel
    from .overlay import haenv_comorbid_specs, haenv_independent_specs
    specs = {**_JS.DDX_SPECS, **haenv_comorbid_specs(_JS.DDX_SPECS),
             **haenv_independent_specs()}
    out: set[str] = set()
    for sp in specs.values():
        t = ((sp.get("adjudication") or {}).get("ddx") or {}).get("tests") or sp.get("tests") or []
        out.update(str(x) for x in t)
    return sorted(out)


def discriminator_tests() -> list[str]:
    """Rival discriminator tests that no condition's gold `tests` names
    (`registry/gated_pricing.yaml:discriminator_tests`), minus any already in the gold
    catalogue. One fixed list for every case."""
    reg = _cached_yaml(_dr() / "registry" / "gated_pricing.yaml") or {}
    have = set(test_catalogue())
    return [str(t) for t in (reg.get("discriminator_tests") or {}) if str(t) not in have]


@functools.lru_cache(maxsize=1)
def _resolves_to() -> dict[str, tuple[str, ...]]:
    """`registry/gated_pricing.yaml:resolves_to`, checked: every name is priced and every id
    is in the findings vocabulary (no fallback on a bad row)."""
    from .registry import load_findings
    reg = _cached_yaml(_dr() / "registry" / "gated_pricing.yaml") or {}
    fd = load_findings()
    pr = _pricing()
    out: dict[str, tuple[str, ...]] = {}
    for name, ids in (reg.get("resolves_to") or {}).items():
        ids = tuple(str(i) for i in (ids or ()))
        if str(name) not in pr:
            raise PricingUnregistered(f"resolves_to: {name!r} has no price")
        bad = [i for i in ids if i not in fd]
        if not ids or bad:
            raise KeyError(f"resolves_to: {name!r} -> {list(ids)}: not in findings.yaml: {bad or 'empty'}")
        out[str(name)] = ids
    return out


def has_reading_model(fspec: dict) -> bool:
    """A finding can be read out only when it carries a reference range or is qualitative.
    Upstream-vocabulary `add` entries (catalogue names turned into findings) carry neither."""
    return bool((fspec or {}).get("qualitative")) or isinstance((fspec or {}).get("ref"), dict)


def menu_findings(target: str, findings: dict | None = None) -> tuple[str, ...]:
    """Finding ids a menu item stands for: the explicit `resolves_to` row, else the unique
    `_resolve_target` match, else (). Ids with no reading model are dropped (a purchase of
    them returns nothing, so there is nothing to credit)."""
    t = str(target or "").strip()
    ids = _resolves_to().get(t)
    if ids:
        return ids
    if findings is None:
        from .registry import load_findings
        findings = load_findings()
    f = _resolve_target(t, findings)
    return (f,) if f and has_reading_model(findings.get(f)) else ()


def synth_menu_target(raw, target: str, T: int) -> list[dict] | None:
    """Reading for a purchased item: none for a decoy (`DECOY_SIGNALS`: the purchase is charged and
    recorded, the result is unavailable); otherwise one `synth_on_demand` point per `resolves_to` id (all
    or nothing, since a composite read half-way cannot be interpreted); otherwise
    `synth_on_demand` on the name itself."""
    if str(target or "").strip() in DECOY_SIGNALS:
        return None                  # a decoy is never readable, whatever it resolves to
    ids = _resolves_to().get(str(target or "").strip())
    if not ids:
        return synth_on_demand(raw, target, T)
    pts: list[dict] = []
    for fid in ids:
        p = synth_on_demand(raw, fid, T)
        if not p:
            return None
        pts.extend(p)
    return pts


#: Typical test count used to derive the budget; a constant, since the case's own gold
#: count would leak its size.
TEST_ALLOWANCE = 6
#: Typical number of monitoring signals a workup would query (a constant)
SIGNAL_ALLOWANCE = 8


def menu_for(sp, withheld: dict, case_id: str = "") -> list[dict]:
    """Menu handed to the model: real signals, decoys and tests with kind and unit price,
    shuffled by `sha256(case_id + name)` so position is not a clue."""
    import hashlib
    from haenv_kernel.gatekeeper import COST                     # kernel
    real = sorted(available_targets(sp, withheld))
    # ---- Section 1: monitoring signals (scores T1 grounding / T2 redundancy) ----
    items = [{"target": t, "kind": kind_of(t), "real": True, "is_test": False} for t in real]
    items += [{"target": t, "kind": kind_of(t), "real": False, "is_test": False} for t in DECOY_SIGNALS]
    # ---- Section 2: test items (scored by tests_recall / tests_precision, not T1) ----
    items += [{"target": t, "kind": kind_of(t), "real": None, "is_test": True}
              for t in test_catalogue() + discriminator_tests()]
    for it in items:
        it["cost"] = COST.get(it["kind"], 5.0)
    items.sort(key=lambda x: hashlib.sha256(f"{case_id}|{x['target']}".encode()).hexdigest())
    return items


def budget_for(menu: list[dict], frac: float = BUDGET_FRAC) -> float:
    """Budget: `SIGNAL_ALLOWANCE` x median signal price + `TEST_ALLOWANCE` x median test
    price (floor 3.0); decoys excluded. `frac` is unused."""
    import statistics as _st
    sig = [float(i["cost"]) for i in menu if i.get("real") and not i.get("is_test")]
    tst = [float(i["cost"]) for i in menu if i.get("is_test")]
    s_unit = _st.median(sig) if sig else 1.0
    t_unit = _st.median(tst) if tst else 5.0
    return round(max(SIGNAL_ALLOWANCE * s_unit + TEST_ALLOWANCE * t_unit, 3.0), 1)


@dataclass
class ToolCall:
    round: int
    kind: str                         # kind self-reported by the model (protocol compliance, doesn't set cost)
    target: str | None
    cost_after: float
    revealed: bool                    # Gatekeeper actually returned a series
    grounded: bool                    # target is in this patient's real signal set (T1)
    #: Per-call audit fields; scoring reads only the cell-level aggregates.
    billed_kind: str | None = None    # the kind from the menu (billing uses this, not the self-report)
    cost: float | None = None         # what was actually charged this call (menu price); None if execution was refused
    truncated: bool = False           # refused for going over budget (the prompt promised this would happen)
    n_points: int | None = None       # number of points in the returned series; 0/None = no series returned


@dataclass
class ToolTrace:
    """A cell's call trace. Lives on the instance, not in the kernel schema."""
    calls: list[ToolCall] = field(default_factory=list)
    spent: float = 0.0
    rounds_used: int = 0
    budget: float = DEFAULT_BUDGET
    committed: bool = False           # model converged on its own, not cut off by round/budget limits
    withheld: tuple[str, ...] = ()
    protocol_errors: int = 0          # count of model outputs that didn't follow protocol (distinct from "didn't query")
    #: Self-reported kind != menu kind (billing always uses the menu kind).
    kind_misreports: int = 0
    menu: list = field(default_factory=list)      # the menu handed to the model for this case (includes decoys, verifier-only)
    rounds_log: list = field(default_factory=list)   # verbatim per-round log (can be rescored if the judge changes, no rerun needed)
    targets: list = field(default_factory=list)      # names of targets that were queried
    #: Per-round leak violations; non-empty means `ABORT(leak)` (handled by `_row_gated`).
    leak: list | None = None
    #: Facts about the last round's raw output, which is the one scored.
    raw_empty: bool = False
    raw_unparseable: bool = False
    #: Queries refused for exceeding the budget; the prompt promises truncation, which also
    #: keeps `thrift = 1 - spent/budget` bounded.
    truncated: int = 0
    truncated_targets: list = field(default_factory=list)
    unavailable_results: dict = field(default_factory=dict)


def withhold_signals(sp) -> tuple[object, dict]:
    """Move non-primary signals from a deep copy of the payload to the Gatekeeper; returns
    (lean payload, withheld signals)."""
    lean = copy.deepcopy(sp)
    withheld = {k: v for k, v in (lean.longitudinal_data or {}).items()
                if k not in ALWAYS_VISIBLE}
    lean.longitudinal_data = {k: v for k, v in (lean.longitudinal_data or {}).items()
                              if k in ALWAYS_VISIBLE}
    return lean, withheld


def available_targets(sp, withheld: dict) -> set[str]:
    """Signals this patient actually has (T1's grounded set): visible plus withheld."""
    return set((sp.longitudinal_data or {})) | set(withheld)


def _parse_step(txt: str) -> dict:
    """Parses the model's output for this round. Uses the same extractor as the main
    pipeline, no separate one."""
    from haenv_kernel.solver import _extract_json
    try:
        d = _extract_json(txt or "")
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def _resolve_target(target: str, findings: dict) -> str | None:
    """Resolve a requested test name to a `findings.yaml` id: exact match, else a unique
    substring match; ambiguous returns None (the caller then reads out a normal value)."""
    tc = (target or "").strip()
    if not tc:
        return None
    tl = tc.lower()
    # Sentinel probe names such as `__no_such_signal__` never resolve.
    if tl.startswith("__") and tl.endswith("__"):
        return None
    # Exact match over the whole table; duplicate id/name pairs exist.
    exact = []
    for fid, fspec in findings.items():
        names = {str(fid).lower()}
        for k in ("name_cn", "name_en"):
            if fspec.get(k):
                names.add(str(fspec[k]).lower())
        names |= {str(a).lower() for a in (fspec.get("aliases") or [])}
        if tl in names:
            exact.append(fid)
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        # Prefer the entry gold actually declares; the other duplicate can never read abnormal.
        try:
            from .registry import condition_findings_for_case as _cff
            _declared = {it.get("id") for d in _cff(findings).values()
                         for it in (d.get("findings") or [])}
        except Exception:
            _declared = set()
        _pref = [f for f in exact if f in _declared]
        return _pref[0] if len(_pref) == 1 else None
    # Substring match, unique hits only. Latin aliases need word boundaries (short ASCII
    # aliases would otherwise hit inside other words); CJK aliases match as substrings.
    _LAT_MIN, _CJK_MIN = 2, 2

    def _is_cjk(s: str) -> bool:
        return any("㐀" <= ch <= "鿿" or "豈" <= ch <= "﫿" for ch in s)

    def _hit(a: str) -> bool:
        if _is_cjk(a):
            return len(a) >= _CJK_MIN and (tl in a or a in tl)
        if len(a) < _LAT_MIN:
            return False
        _re = __import__("re")
        _pat = rf"(?<![0-9a-z]){_re.escape(a)}(?![0-9a-z])"
        return bool(_re.search(_pat, tl)) or bool(_re.search(
            rf"(?<![0-9a-z]){_re.escape(tl)}(?![0-9a-z])", a))

    sub = []
    for fid, fspec in findings.items():
        cand = [str(a).lower() for a in (fspec.get("aliases") or [])]
        for k in ("name_cn", "name_en"):
            if fspec.get(k):
                cand.append(str(fspec[k]).lower())
        if any(_hit(a) for a in cand):
            sub.append(fid)
    return sub[0] if len(sub) == 1 else None


def _obs_cfg() -> tuple[dict, dict]:
    """(`physio_streams.yaml:clinical_observation`, `clinical_measurement.cv`)."""
    from .regpath import load_registry
    doc = load_registry("physio_streams.yaml") or {}
    return (doc.get("clinical_observation") or {},
            ((doc.get("clinical_measurement") or {}).get("cv") or {}))


def _u(key: str) -> float:
    import hashlib
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _pick_w(weights: dict, u: float):
    tot = float(sum(float(w) for w in weights.values())) or 1.0
    acc = 0.0
    for k, w in weights.items():
        acc += float(w) / tot
        if u < acc:
            return k
    return list(weights)[-1]


def _report_face(case_id: str, fid: str, flag: str) -> tuple[str | None, str]:
    """The reporting lab's printed name for `fid` (`name_variants`; `None` = the vocabulary
    name) and its flag symbol for `flag` (H / L / NORMAL -> `named_flags`)."""
    cfg, _ = _obs_cfg()
    names = (cfg.get("name_variants") or {}).get(fid)
    name = _pick_w(names, _u(f"{case_id}|named-name|{fid}")) if names else None
    fl = cfg.get("named_flags") or {}
    side = {"H": "high", "L": "low"}.get(flag, "normal")
    sym = _pick_w(fl[side], _u(f"{case_id}|named-flag|{side}")) if fl.get(side) else flag
    return name, str(sym)


def flag_side(sym: str) -> str:
    """The side (`H` / `L` / `NORMAL`) a printed flag symbol stands for (`named_flags`);
    other symbols (POSITIVE / NEGATIVE) come back unchanged."""
    cfg, _ = _obs_cfg()
    for side, syms in (cfg.get("named_flags") or {}).items():
        if sym in (syms or {}):
            return {"high": "H", "low": "L"}.get(side, "NORMAL")
    return str(sym)


def _named_on_stream(raw, fid: str, fspec: dict, T: int) -> list[dict] | None:
    """A named test whose indicator is one of the case's live lab streams: a new draw on day
    T -- the latest draw at or before T times one draw's measurement variation
    (`clinical_measurement.cv`), at the registered precision.
    `None` when the case carries no such stream."""
    import math
    import statistics
    from .findings_render import _TS_TO_PANEL, _band
    from .indicators import of
    stream = {v: k for k, v in _TS_TO_PANEL.items()}.get(fid)
    ld = getattr(raw, "longitudinal_data", None) or {}
    pts = sorted((q for q in (ld.get(stream) or [])
                  if isinstance(q, dict) and isinstance(q.get("value"), (int, float))
                  and not isinstance(q.get("value"), bool)),
                 key=lambda q: int(q["ts"]))
    if not stream or not pts:
        return None
    # Centred on the latest draw at or before T: nothing after T (no later draw, no later
    # measurement error) reaches a reading the solver gets on day T.
    before = [q for q in pts if int(q["ts"]) <= int(T)]
    v = float((before[-1] if before else pts[0])["value"])
    case_id = str(getattr(raw, "case_id", "CASE"))
    _, cv = _obs_cfg()
    u = min(max(_u(f"{case_id}|named|{stream}|{int(T)}"), 1e-9), 1 - 1e-9)
    z = max(-3.0, min(3.0, statistics.NormalDist().inv_cdf(u)))
    v *= math.exp(float(cv.get(stream) or 0.0) * z)
    nd = int(of(stream)["ndigits"])
    # the reporting lab's print precision, the same decision as on the stream
    u_p, acc = _u(f"{case_id}|labprint"), 0.0
    for d, pr in sorted(((_obs_cfg()[0].get("print_decimals_mix") or {}).get(stream) or {}).items()):
        acc += float(pr)
        if u_p < acc:
            nd = max(nd, int(d))
            break
    v = round(v, nd) if nd else float(round(v))
    lo, hi = _band(fspec)
    flag = "H" if (hi is not None and v > hi) else ("L" if (lo is not None and v < lo) else "NORMAL")
    name, sym = _report_face(case_id, fid, flag)
    return [{"ts": int(T), "value": v, "unit": fspec.get("unit", ""), "ref_low": lo,
             "ref_high": hi, "flag": sym, "finding_id": fid,
             "name": name or fspec.get("name_cn", fid)}]


def synth_on_demand(raw, target: str, T: int) -> list[dict] | None:
    """Synthesize a reading for a test not embedded in the data (`need_synth`): abnormal per
    the condition's `condition_findings.yaml` declaration, otherwise within the reference
    range. Returns a list of points."""
    if not target or not isinstance(target, str):
        return None

    from .findings_render import _band, _value
    from .registry import load_findings

    target_clean = target.strip()
    findings = load_findings()

    matched_fid = _resolve_target(target_clean, findings)

    if not matched_fid:
        # A missing/ambiguous definition is not clinical evidence of normality.
        return None

    fspec = findings[matched_fid]
    if not has_reading_model(fspec):
        # An entry with no reference range and no qualitative reading cannot say what normal
        # is; a value drawn for it always reads normal, whatever the patient has.
        return None
    case_id = getattr(raw, "case_id", "CASE")
    # A named test on one of the case's live lab streams is a new draw of that stream
    # (the two channels must agree), not an independent value.
    _on_stream = _named_on_stream(raw, matched_fid, fspec, T)
    if _on_stream is not None:
        return _on_stream
    lp = getattr(raw, "latent_premise", {}) or {}
    pb = lp.get("patient_basics") or {}
    # Condition comes from the case itself (`overlay.spec_id_of`), keeping the batch
    # reproducible from cases.jsonl.
    from .overlay import spec_id_of
    spec_id, _sid_src = spec_id_of(raw)

    from .registry import condition_findings_for_case
    cond_findings = condition_findings_for_case(findings)
    cf_entries = cond_findings.get(spec_id, {}).get("findings", []) if spec_id else []
    decl = next((item for item in cf_entries if item.get("id") == matched_fid), None)
    # Insufficient tier: the disease has not started at T, so declared items read normal.
    _declared_dir = (decl or {}).get("direction")
    if ((getattr(raw, "adjudication", None) or {}).get("ddx") or {}).get("insufficient"):
        decl = None

    # Qualitative when the vocabulary says so or gold declares a qualitative direction,
    # matching `findings_render.render_findings`.
    is_qualitative = bool(fspec.get("qualitative")) or _declared_dir in ("positive", "negative")
    if is_qualitative:
        if decl and decl.get("direction") != "negative":
            val_str = "阳性 / 异常提示病理性改变"
            flag = "POSITIVE"
        else:
            val_str = "阴性 / 未见明显异常"
            flag = "NEGATIVE"
        return [{
            "ts": T,
            "value": val_str,
            "flag": flag,
            "finding_id": matched_fid,
            "name": fspec.get("name_cn", matched_fid),
        }]

    if decl:
        direction = decl.get("direction", "high")
        magnitude = decl.get("magnitude", "moderate")
        val = _value(fspec, direction, magnitude, case_id, matched_fid, 0)
    else:
        val = _value(fspec, "normal", None, case_id, matched_fid, 0)

    lo, hi = _band(fspec)
    unit = fspec.get("unit", "")
    name_cn = fspec.get("name_cn", matched_fid)
    flag = ("H" if val > hi else ("L" if val < lo else "NORMAL")) if isinstance(val, (int, float)) else "NORMAL"
    _name, _sym = _report_face(str(case_id), matched_fid, flag)

    return [{
        "ts": T,
        "value": val,
        "unit": unit,
        "ref_low": lo,
        "ref_high": hi,
        "flag": _sym,
        "finding_id": matched_fid,
        "name": _name or name_cn,
    }]


def run_gated(raw, T: int, solver, budget: float = DEFAULT_BUDGET,
              max_rounds: int = MAX_ROUNDS, trace=None) -> tuple[object, ToolTrace]:
    """The on-demand query loop. Each round the model returns
    `{"queries": [...], "commit": bool, ...answer fields}`; any answer field counts as
    converging. `trace`, if given, receives the event log, including tool results that cannot
    be reconstructed afterwards."""
    from haenv_kernel.build import build_instance                    # kernel
    from haenv_kernel.gatekeeper import Gatekeeper                   # kernel
    from .solve_guard import guarded_solve as _guarded_solve
    # The solver instance is reused across cells, so clear its per-round log.
    if hasattr(solver, "_gated_round_log"):
        solver._gated_round_log = []                    # noqa: SLF001
    sp, _vp = build_instance(raw, T)
    lean, withheld = withhold_signals(sp)
    gk = Gatekeeper({**(lean.longitudinal_data or {}), **withheld}, T)
    avail = available_targets(sp, withheld)
    menu = menu_for(sp, withheld, getattr(raw, "case_id", ""))
    # An explicit budget (other than the default) reproduces old batches.
    budget = float(budget) if budget != DEFAULT_BUDGET else budget_for(menu)
    tr = ToolTrace(budget=budget, withheld=tuple(sorted(withheld)))
    tr.menu = [dict(i) for i in menu]
    #: Billing uses the menu's kind and price, never the model's self-report.
    _menu_kind: dict[str, str] = {str(i["target"]): str(i["kind"]) for i in menu}
    _menu_cost: dict[str, float] = {str(i["target"]): float(i.get("cost") or 0.0) for i in menu}

    if trace is not None:
        trace.append("case/start", {"T": int(T), "budget": budget,
                                    "max_rounds": max_rounds, "menu": tr.menu,
                                    "n_withheld": len(withheld)})
    revealed_ctx: dict[str, list] = {}
    _call_no = 0
    out = None
    for r in range(1, max_rounds + 1):
        tr.rounds_used = r
        step_payload = copy.deepcopy(lean)
        step_payload.longitudinal_data = {**(lean.longitudinal_data or {}), **revealed_ctx}
        solver.gated_context = {"budget": budget, "spent": round(gk.spent, 2),
                                "menu": [i for i in menu if i["target"] not in revealed_ctx
                                         and i["target"] not in tr.unavailable_results],
                                "askable": sorted(avail - set(revealed_ctx) - set(tr.unavailable_results)),
                                "truncated_last_round": list(tr.truncated_targets),
                                "unavailable_results": dict(tr.unavailable_results),
                                "round": r, "max_rounds": max_rounds}
        # The leak gate runs every round: menu text and synthesized readings are produced
        # after `build_instance`.
        if trace is not None:
            trace.append("step/start", {}, step=r)
            # Only target names; kind and price come from the `case/start` menu.
            _gc = dict(solver.gated_context)
            _gc["menu"] = [str(i["target"]) for i in (_gc.get("menu") or [])]
            trace.append("request/header", {
                "gated_context": _gc,
                "menu_is_names_only": True,      # readers use this to backfill kind/price from case/start
                "revealed_targets": sorted(revealed_ctx),
            }, step=r)
        facts = _guarded_solve(solver, step_payload, T, tag=f"{raw.case_id}|gated:r{r}")
        if facts.leak:
            tr.leak = list(facts.leak)          # this cell is voided; handling is `_row_gated`'s job
            if trace is not None:
                trace.append("gate/verdict", {"gate": "leak", "ok": False,
                                              "violations": list(facts.leak)}, step=r)
                trace.append("step/end", {"reason": "leak"}, step=r)
            break
        out = facts.out
        tr.raw_empty, tr.raw_unparseable = facts.raw_empty, facts.raw_unparseable
        d = _parse_step(getattr(out, "_raw_text", "") or "")
        # Per-round raw output and usage are kept, so rescoring and cost analysis need no rerun.
        _rp = getattr(solver, "_gated_round_log", None)
        if _rp is None:
            _rp = solver._gated_round_log = []                  # noqa: SLF001
        _raw_full = getattr(out, "_raw_text", "") or ""
        _rp.append({"round": r, "raw_text": _raw_full,
                    "usage": getattr(out, "_usage", None),
                    "finish_reason": getattr(out, "_finish", None),
                    "max_tokens": getattr(out, "_max_tokens", None),
                    "latency_s": getattr(out, "_latency_s", None),
                    "request_prompt_sha256": getattr(out, "_prompt_sha_full", None),
                    "requests": getattr(out, "_requests", None),
                    "queries": qs_raw if (qs_raw := d.get("queries")) else None})
        if trace is not None:
            trace.append("assistant/message", {
                "raw": _raw_full, "n_chars": len(_raw_full),
                "usage": getattr(out, "_usage", None),
                "finish_reason": getattr(out, "_finish", None),
                "max_tokens": getattr(out, "_max_tokens", None),
                "latency_s": getattr(out, "_latency_s", None),
                # Stored separately and never scored.
                "reasoning_text": getattr(out, "_reasoning_text", None),
            }, step=r)
            for _fa in (getattr(out, "_failed_attempts", None) or []):
                # Retried attempts never reached the model's context, so they are not messages.
                trace.append("assistant/attempt", dict(_fa), step=r)
        qs = d.get("queries") if isinstance(d.get("queries"), list) else None
        has_answer = bool(d.get("differential") or d.get("forecast") or d.get("drivers"))
        if not qs and not has_answer:
            tr.protocol_errors += 1
        for q in (qs or []):
            if not isinstance(q, dict):
                tr.protocol_errors += 1
                continue
            kind = str(q.get("kind") or "ask")
            target = q.get("target")
            target = str(target) if target else None
            _billed = _menu_kind.get(target) if target else None
            if _billed is not None and _billed != kind:
                tr.kind_misreports += 1
            # Budget is checked before each query; a refused query is neither billed nor revealed.
            _price = _menu_cost.get(target) if target else None
            _call_no += 1
            _cid = f"{r}.{_call_no}"
            if trace is not None:
                # The model's raw query, verbatim.
                trace.append("tool/call", {"call_id": _cid, "arguments": q,
                                           "kind": kind, "target": target,
                                           "billed_kind": _billed,
                                           "menu_cost": _price}, step=r)
            if _price is not None and gk.spent + _price > budget:
                tr.truncated += 1
                if target not in tr.truncated_targets:
                    tr.truncated_targets.append(target)
                tr.calls.append(ToolCall(round=r, kind=kind, target=target,
                                         cost_after=round(gk.spent, 2),
                                         revealed=False, grounded=None,
                                         billed_kind=_billed, cost=None,
                                         truncated=True, n_points=None))
                if trace is not None:
                    # Every tool/call gets a tool/result, including refusals.
                    trace.append("tool/result", {
                        "call_id": _cid, "truncated": True, "revealed": False,
                        "error": {"name": "BudgetExceeded",
                                  "reason": f"价 {_price} > 余额 {round(budget - gk.spent, 2)}"},
                        "cost": 0.0, "spent_after": round(gk.spent, 2)}, step=r)
                log.info("[gated] %s refused to execute %s (price %.1f, balance %.1f) -- over budget, "
                         "truncated per the question face's stated commitment",
                         raw.case_id, target, _price, budget - gk.spent)
                continue
            try:
                res = gk.query(_billed or kind, target)
            except Exception as _e:                      # noqa: BLE001
                # Log an `outcome: unknown` result before re-raising, so a crash is not an orphan call.
                if trace is not None:
                    trace.append("tool/result", {
                        "call_id": _cid, "truncated": False, "revealed": False,
                        "outcome": "unknown",
                        "error": {"name": type(_e).__name__, "reason": str(_e)[:200]},
                        "spent_after": round(gk.spent, 2)}, step=r)
                raise
            if res.get("need_synth") and target:
                synth_pts = synth_menu_target(raw, target, T)
                if synth_pts:
                    res["series"] = synth_pts
                    res["need_synth"] = False
                else:
                    tr.unavailable_results[target] = "unsupported_or_ambiguous_target"
            # Tests are not scored for grounding (any test can be ordered).
            _is_test = any(i["target"] == target and i.get("is_test") for i in menu)
            grounded = None if _is_test else bool(target and target in avail)
            _series = res.get("series")
            tr.calls.append(ToolCall(round=r, kind=kind, target=target,
                                     cost_after=round(gk.spent, 2), revealed=bool(_series),
                                     grounded=grounded, billed_kind=_billed,
                                     cost=_price, truncated=False,
                                     n_points=len(_series) if isinstance(_series, list) else None))
            if trace is not None:
                trace.append("tool/result", {
                    "call_id": _cid, "truncated": False, "revealed": bool(_series),
                    "observation": _series, "n_points": (len(_series)
                                                         if isinstance(_series, list) else None),
                    "grounded": grounded, "is_test": bool(_is_test),
                    "cost": _price, "spent_after": round(gk.spent, 2),
                    "need_synth": bool(res.get("need_synth")),
                    "outcome": "unavailable" if target in tr.unavailable_results else "returned",
                    "unavailable_reason": tr.unavailable_results.get(target),
                }, step=r)
            if _series:
                revealed_ctx[target] = _series
        tr.spent = round(gk.spent, 2)
        if has_answer:
            tr.committed = True
            if trace is not None:
                trace.append("step/end", {"reason": "committed",
                                          "spent": tr.spent}, step=r)
            break
        if gk.spent >= budget:                          # budget exhausted -> forced convergence (T3 will record this)
            log.info("[gated] %s budget exhausted (%.1f/%.1f), truncated", raw.case_id, gk.spent, budget)
            if trace is not None:
                trace.append("step/end", {"reason": "budget_exhausted",
                                          "spent": tr.spent}, step=r)
            break
        if trace is not None:
            trace.append("step/end", {"reason": "next_round", "spent": tr.spent}, step=r)
    if out is not None:
        out._tool_trace = tr                            # noqa: SLF001 -- read by `_row_gated`
    tr.rounds_log = list(getattr(solver, "_gated_round_log", []) or [])
    tr.targets = [c.target for c in tr.calls if c.target]
    if trace is not None:
        trace.append("case/end", {
            "rounds_used": tr.rounds_used, "spent": tr.spent, "budget": tr.budget,
            "committed": tr.committed, "n_calls": len(tr.calls),
            "truncated": tr.truncated, "protocol_errors": tr.protocol_errors,
            "kind_misreports": tr.kind_misreports,
            "leak": bool(tr.leak), "no_output": out is None})
    return out, tr
