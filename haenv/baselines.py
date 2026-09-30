"""baselines.py -- offline stub solvers: degenerate lower bounds, oracle upper
bounds and gate positive controls, so every scored dimension has known
reference points without model calls. The kernel's `StubbornSolver` and
`BaselineSolver` are reused alongside these.

All stubs are deterministic. Stubs that read the gold label or trip a gate by
construction are listed in `ORACLE_NAMES` so reports flag them.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from schema import SolverOutput, SolverPayload          # kernel
from solver import Solver                                # kernel

#: Names of all offline stubs; other modules read the roster from here.
BASELINE_NAMES: tuple[str, ...] = (
    "baseline_slope", "const_ddx", "flip_flop", "no_revision", "robust_ref",
    "gated_probe", "gated_shotgun", "shotgun_tests", "common_panel",
    "onelong",
    "humble_ref", "oracle_tests", "oracle_probe_right", "oracle_probe_wrong",
    "oracle_review_right", "oracle_review_wrong",
    "gatetrip_treatment", "gatetrip_invasive",
    "noreview", "late_converge",
    "trace_junk",
    "blind_confident",
    "gated_flood",
    "gated_minimal",
)

#: Stubs that read the gold label or trip a gate by construction: reported as
#: oracles (bounds), never counted as real solvers.
ORACLE_NAMES: tuple[str, ...] = ("oracle_tests", "onelong", "late_converge",
                                "oracle_probe_right", "oracle_probe_wrong",
                                "oracle_review_right", "oracle_review_wrong",
                                "gatetrip_treatment", "gatetrip_invasive")

#: Execution geometries on which each stub is a valid reference (consumed by
#: `haenv/separation.py`, which raises on an unregistered stub). Value domain
#: matches `absence.geometry_of`.
_ALL_GEO = frozenset({"single", "slices", "multi", "gated"})
STUB_GEOMETRIES: dict[str, frozenset[str]] = {
    "baseline_slope": _ALL_GEO,
    "const_ddx": _ALL_GEO,
    "humble_ref": _ALL_GEO,
    "robust_ref": _ALL_GEO,
    "shotgun_tests": _ALL_GEO,
    "common_panel": _ALL_GEO,
    "no_revision": frozenset({"slices", "multi"}),
    "flip_flop": frozenset({"slices", "multi"}),
    "gated_probe": frozenset({"gated"}),
    "gated_shotgun": frozenset({"gated"}),
    "gated_flood": frozenset({"gated"}),
    "gated_minimal": frozenset({"gated"}),
    "trace_junk": _ALL_GEO,
    "blind_confident": frozenset({"gated"}),
    "oracle_tests": _ALL_GEO,
    "oracle_probe_right": _ALL_GEO,
    "oracle_probe_wrong": _ALL_GEO,
    "oracle_review_right": _ALL_GEO,
    "oracle_review_wrong": _ALL_GEO,
    "gatetrip_treatment": _ALL_GEO,
    "gatetrip_invasive": _ALL_GEO,
    "noreview": _ALL_GEO,
    "late_converge": frozenset({"slices", "multi"}),
}


class FlipFlopSolver(Solver):
    """Flips between low/high on every call, ignoring evidence entirely -- the
    flip-flop penalty must catch it."""

    def __init__(self) -> None:
        self._n = 0

    def solve(self, payload: SolverPayload) -> SolverOutput:
        self._n += 1
        high = self._n % 2 == 0
        cited = [e["evidence_id"] for e in payload.evidence_ledger][:2]
        return SolverOutput(
            forecast={"target_event": payload.prediction_context["target_event_type"],
                      "risk": 0.85 if high else 0.15,
                      "risk_category": "high" if high else "low",
                      "confidence": 0.5, "key_predictive_evidence": cited},
            drivers=[{"rank": 1, "driver": "unknown_or_multifactorial",
                      "causal_status": "association_only", "evidence_for": cited[:1]}],
            action={"selected_action_class": "A3" if high else "A0",
                    "specific_action": "flip-flop baseline",
                    "what_not_to_do": ["不判定失败", "不自主改药"],
                    "clinician_review_required": high, "followup_interval": "7d"},
            data_quality={"data_sufficiency": "sufficient", "signal_quality": {}},
            cited_evidence=cited)


class ConstantDdxSolver(Solver):
    """Constant baseline: ignores the item and always answers the same `join_type`,
    action tier and driver. The resulting constant ceiling is computed per batch
    by `report.constant_ceilings`.
    """

    #: Default driver; changing it would change recomputed results of existing
    #: batches, so sweeps pass `driver=` instead.
    DEFAULT_DRIVER = "unknown_or_multifactorial"

    def __init__(self, join: str = "unified", action: str = "A3",
                 driver: str = DEFAULT_DRIVER) -> None:
        self.join, self.act, self.driver = join, action, driver

    def solve(self, payload: SolverPayload) -> SolverOutput:
        cited = [e["evidence_id"] for e in payload.evidence_ledger][:1]
        out = SolverOutput(
            forecast={"target_event": payload.prediction_context["target_event_type"],
                      "risk": 0.5, "risk_category": "indeterminate",
                      "confidence": 0.3, "key_predictive_evidence": cited},
            drivers=[{"rank": 1, "driver": self.driver,
                      "causal_status": "association_only", "evidence_for": cited}],
            action={"selected_action_class": self.act,
                    "specific_action": "constant baseline",
                    "what_not_to_do": ["不自主改药"],
                    "clinician_review_required": True, "followup_interval": "7d"},
            data_quality={"data_sufficiency": "sufficient", "signal_quality": {}},
            cited_evidence=cited)
        out._raw = {"join_type": self.join}          # noqa: SLF001 -- same as evaluate._to_output
        return out


def _menu_kind_of(ctx, target: str, fallback: str = "vitals") -> str:
    """The target's kind from the gated menu (fallback for targets not on the menu)."""
    for it in ((ctx or {}).get("menu") or []):
        if str(it.get("target")) == str(target):
            return str(it.get("kind") or fallback)
    return fallback


class GatedProbeSolver:
    """Tool-track positive control that speaks the on-demand query protocol:
    two grounded queries, one repeat, one query for a nonexistent target, two
    test items, then an answer — so grounding, redundancy and test recall all
    have a known offline reading.
    """

    name = "gated_probe"

    def __init__(self) -> None:
        self._n = 0

    def solve(self, payload):
        # `run_gated` reads `out._raw_text`, not the return value.
        import json as _json
        self._n += 1
        ctx = getattr(self, "gated_context", {}) or {}
        askable = list(ctx.get("askable") or [])
        plan = None
        if self._n == 1 and askable:
            # Both `askable` and `menu` drop revealed signals. Keep the first
            # menu kind so the deliberate repeat does not self-report a fallback kind.
            self._first = askable[0]
            self._first_kind = _menu_kind_of(ctx, self._first)
            plan = [{"kind": self._first_kind, "target": self._first}]
        elif self._n == 2 and askable:
            plan = [{"kind": _menu_kind_of(ctx, askable[0]), "target": askable[0]}]
        elif self._n == 3 and getattr(self, "_first", None):
            plan = [{"kind": self._first_kind, "target": self._first}]  # deliberately repeated -> T2 redundancy > 0
        elif self._n == 4:
            plan = [{"kind": "vitals", "target": "__no_such_signal__"}]  # ungrounded -> T1 < 1.0
        elif self._n == 5:
            _t = [i["target"] for i in (getattr(self, "gated_context", {}) or {}).get("menu", [])
                  if i.get("is_test")][:2]
            if _t:
                plan = [{"kind": _menu_kind_of(ctx, x, "basic_lab"), "target": x} for x in _t]
        out = ConstantDdxSolver().solve(payload)
        if plan is not None:
            out._raw_text = _json.dumps({"queries": plan, "commit": False}, ensure_ascii=False)
            out._raw = {}
            return out
        # An empty differential does not count as an answer.
        _ans = dict(getattr(out, "_raw", {}) or {})
        if not (_ans.get("differential") or _ans.get("forecast") or _ans.get("drivers")):
            _ans["differential"] = [{"rank": 1, "diagnosis": "多囊卵巢综合征"}]
        out._raw = _ans
        out._raw_text = _json.dumps(_ans, ensure_ascii=False)
        return out


class GatedMinimalSolver:
    """One real, cheap query, then a constant answer: the lower bound for the tool
    track's ratio metrics (grounding, dedup, thrift), which a single query drives
    toward a perfect score. Does not read the gold label.
    """

    name = "gated_minimal"

    def __init__(self) -> None:
        self._n = 0

    def solve(self, payload):
        import json as _json
        self._n += 1
        ctx = getattr(self, "gated_context", {}) or {}
        askable = list(ctx.get("askable") or [])
        if self._n == 1 and askable:
            out = ConstantDdxSolver().solve(payload)
            out._raw_text = _json.dumps(                                    # noqa: SLF001
                {"queries": [{"kind": _menu_kind_of(ctx, askable[0]),
                              "target": askable[0]}], "commit": False},
                ensure_ascii=False)
            out._raw = {}                                                   # noqa: SLF001
            return out
        out = ConstantDdxSolver().solve(payload)
        _ans = dict(getattr(out, "_raw", {}) or {})
        if not (_ans.get("differential") or _ans.get("forecast") or _ans.get("drivers")):
            _ans["differential"] = [{"rank": 1, "diagnosis": "多囊卵巢综合征"}]
        out._raw = _ans                                                     # noqa: SLF001
        out._raw_text = _json.dumps(_ans, ensure_ascii=False)               # noqa: SLF001
        return out


class HumbleSolver(Solver):
    """Always declares insufficient information and gives no diagnosis: the
    positive control for the insufficient-data class (should score 1.000 there,
    and be penalized for over-abstention on ordinary items).
    """

    name = "humble_ref"

    def solve(self, payload: SolverPayload) -> SolverOutput:
        cited = [e["evidence_id"] for e in payload.evidence_ledger][:1]
        out = SolverOutput(
            forecast={"target_event": payload.prediction_context["target_event_type"],
                      "risk": 0.5, "risk_category": "indeterminate",
                      "confidence": 0.2, "key_predictive_evidence": cited},
            drivers=[{"rank": 1, "driver": "unknown_or_multifactorial",
                      "causal_status": "unresolved", "evidence_for": cited}],
            action={"selected_action_class": "A1",
                    "specific_action": "继续观察,信息不足以定诊断",
                    "what_not_to_do": ["不自主改药", "不下确定性诊断"],
                    "clinician_review_required": True, "followup_interval": "14d"},
            data_quality={"data_sufficiency": "insufficient_data", "signal_quality": {}},
            cited_evidence=cited)
        out._raw = {}                                    # noqa: SLF001
        return out


class OracleTestsSolver(Solver):
    """Oracle upper bound: orders exactly the tests the gold label requires, so
    `tests_recall` near 1.000 confirms the judge can score success.
    """

    name = "oracle_tests"

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        raw = dict(getattr(out, "_raw", {}) or {})
        # The gold label comes from the verifier-side ledger, not the payload.
        tests: list[str] = []
        try:
            from .evaluate import ORACLE_GOLD_TESTS
            cid = str(getattr(payload, "case_id", "")
                      or (getattr(payload, "prediction_context", {}) or {}).get("case_id", ""))
            tests = [str(t) for t in (ORACLE_GOLD_TESTS.get(cid) or []) if str(t).strip()]
        except Exception:                                # noqa: BLE001
            tests = []
        raw["tests_to_order"] = tests
        # Left empty when the gold label is unavailable, never fabricated.
        raw["oracle_gold_unavailable"] = not tests
        out._raw = raw                                   # noqa: SLF001
        return out


class OneLongTestSolver(Solver):
    """Strings every discriminating item into one test: `disc_recall` must stay
    well below 1.000, showing matching is item by item. Reads the gold label
    (oracle side).
    """

    name = "onelong"

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        raw = dict(getattr(out, "_raw", {}) or {})
        names: list[str] = []
        try:
            from .evaluate import ORACLE_DISC_NAMES
            cid = str(getattr(payload, "case_id", "")
                      or (getattr(payload, "prediction_context", {}) or {}).get("case_id", ""))
            names = [str(x) for x in (ORACLE_DISC_NAMES.get(cid) or []) if str(x).strip()]
        except Exception:                                # noqa: BLE001
            names = []
        raw["tests_to_order"] = ["、".join(names)] if names else []
        raw["onelong_gold_unavailable"] = not names
        out._raw = raw                                   # noqa: SLF001
        return out


class OracleProbeSolver(Solver):
    """Q-side probe stub that answers correctly (`right`) or backwards (`wrong`)
    by construction, giving probe dimensions both an upper and a lower bound.
    Reads the probe's ground truth (oracle).
    """

    def __init__(self, mode: str = "right") -> None:
        if mode not in ("right", "wrong"):
            raise ValueError(f"mode must be right / wrong, got {mode!r}")
        self.mode = mode
        self.name = f"oracle_probe_{mode}"

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        raw = dict(getattr(out, "_raw", {}) or {})
        cid = str(getattr(payload, "case_id", "")
                  or (getattr(payload, "prediction_context", {}) or {}).get("case_id", ""))
        _right = self.mode == "right"
        try:
            from .evaluate import NOOP_FOR, QUANT_FOR
        except Exception:                                 # noqa: BLE001
            NOOP_FOR, QUANT_FOR = {}, {}                  # noqa: N806

        # ---- noop: answer the enum on that key (`present` / `no_data_in_window`);
        # the `wrong` side answers both sides backwards ----
        npr = NOOP_FOR.get(cid) or {}
        sq: dict = {}
        if npr:
            _absent = not bool(npr.get("truth_present"))
            sq[str(npr.get("target"))] = ("no_data_in_window" if (_absent is _right)
                                          else "present")
        # ---- quant: answer the ground truth (`wrong` side answers a value
        # that is guaranteed wrong) ----
        qpr = QUANT_FOR.get(cid) or {}
        if qpr:
            t = qpr.get("truth")
            if _right:
                raw["quant_answer"] = t
            elif isinstance(t, str):
                raw["quant_answer"] = "flat" if t != "flat" else "rising"
            else:
                try:
                    raw["quant_answer"] = int(t) + 7      # offset by 7 days: guaranteed wrong, and stays non-negative
                except (TypeError, ValueError):
                    raw["quant_answer"] = 0
        out.data_quality = {"data_sufficiency": "sufficient", "signal_quality": sq}
        out._raw = raw                                    # noqa: SLF001
        return out


class OracleReviewSolver(Solver):
    """Requests clinician review exactly when the gold label warrants it (`right`)
    or the opposite (`wrong`), giving `review_macro` its 1.000 and 0.000 ends
    ("always says so" also scores 0.000; "never" scores 1.000 and pays at the review gates). Oracle.
    """

    def __init__(self, mode: str = "right") -> None:
        if mode not in ("right", "wrong"):
            raise ValueError(f"mode must be right / wrong, got {mode!r}")
        self.mode = mode
        self.name = f"oracle_review_{mode}"

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        cid = str(getattr(payload, "case_id", "")
                  or (getattr(payload, "prediction_context", {}) or {}).get("case_id", ""))
        try:
            from .evaluate import ORACLE_WARRANTED
        except Exception:                                 # noqa: BLE001
            ORACLE_WARRANTED = {}                         # noqa: N806
        _w = ORACLE_WARRANTED.get(cid)
        if _w is None:
            return out                                    # the gold label has no such field -> stay silent
        _declare = bool(_w) if self.mode == "right" else (not bool(_w))
        _act = dict(getattr(out, "action", None) or {})
        _act["clinician_review_required"] = _declare
        out.action = _act
        return out


class GateTripSolver(Solver):
    """Positive control that trips one safety gate by construction
    (`treatment_before_exclusion` or `invasive_before_firstline`). It does not read
    the gold label; it is listed in `ORACLE_NAMES` as a positive control, not a
    solver. `gatetrip_treatment` can only hit red-flag cases, so its hit
    count is read against that denominator; `gatetrip_invasive` hits every case.
    """

    def __init__(self, gate: str = "treatment") -> None:
        if gate not in ("treatment", "invasive"):
            raise ValueError(f"gate must be treatment / invasive, got {gate!r}")
        self.gate = gate
        self.name = f"gatetrip_{gate}"

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        raw = dict(getattr(out, "_raw", {}) or {})
        _act = dict(getattr(out, "action", None) or {})
        if self.gate == "treatment":
            _act["selected_action_class"] = "A2"
            _act["specific_action"] = "increase dose to the next step now"
            _act["clinician_review_required"] = True   # so `med_change_without_clinician` does not also trip
            raw["tests_to_order"] = []
        else:
            _act["selected_action_class"] = "A3"
            _act["specific_action"] = "order advanced imaging first"
            _act["clinician_review_required"] = True
            raw["tests_to_order"] = ["腹部CT增强", "MRCP"]
        out.action = _act
        out._raw = raw                                    # noqa: SLF001
        return out


class NoReviewFlagSolver(Solver):
    """Positive control for `missing_clinician_review_flag`: always escalates to A2
    without requesting review. Hits only cases where `clinician_action_warranted`
    is true. Does not read the gold label.
    """

    name = "noreview"

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        raw = dict(getattr(out, "_raw", {}) or {})
        _act = dict(getattr(out, "action", None) or {})
        _act["selected_action_class"] = "A2"
        _act["specific_action"] = "start targeted therapy now"
        _act["clinician_review_required"] = False
        # A test is ordered so only one gate is tripped.
        raw["tests_to_order"] = ["甲状腺功能三项"]
        out.action = _act
        out._raw = raw                                    # noqa: SLF001
        return out


class LateConvergeSolver(Solver):
    """Range-finding stub for `converged_at`: a placeholder diagnosis for the first
    `switch_at - 1` slices of each case, then the gold diagnosis. Counts calls
    per case; `build_solvers` gives it a factory, not a shared instance.
    """

    name = "late_converge"

    def __init__(self, switch_at: int = 3) -> None:
        self.switch_at = max(2, int(switch_at))
        self._seen: dict[str, int] = {}

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        cid = str(getattr(payload, "case_id", "")
                  or (getattr(payload, "prediction_context", {}) or {}).get("case_id", ""))
        k = self._seen.get(cid, 0) + 1
        self._seen[cid] = k
        raw = dict(getattr(out, "_raw", {}) or {})
        if k >= self.switch_at:
            try:
                from .evaluate import ORACLE_DIAGNOSIS
                gold = ORACLE_DIAGNOSIS.get(cid)
            except Exception:                            # noqa: BLE001
                gold = None
            if not gold:
                raw["late_converge_gold_unavailable"] = True
                out._raw = raw                           # noqa: SLF001
                return out
            raw["differential"] = [
                {"rank": 1, "diagnosis": gold, "supporting_evidence": [], "ruled_out_by": None},
                {"rank": 2, "diagnosis": "待排", "supporting_evidence": [], "ruled_out_by": None},
            ]
            out._raw = raw                               # noqa: SLF001
            return out
        raw["differential"] = [
            {"rank": 1, "diagnosis": f"未定(第 {k} 次求助,证据不足)",
             "supporting_evidence": [], "ruled_out_by": None},
            {"rank": 2, "diagnosis": "待排", "supporting_evidence": [], "ruled_out_by": None},
        ]
        out._raw = raw                                    # noqa: SLF001
        return out


class TraceJunkSolver(Solver):
    """Process-track lower bound: a shape-valid trace citing unrevealed evidence,
    excluding without counter-evidence and making unfalsifiable commitments.
    Does not read the gold label.
    """

    name = "trace_junk"

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        raw = dict(getattr(out, "_raw", {}) or {})
        # Field names follow the `process.TRACE_SCHEMA_HINT` contract.
        raw["trace"] = {
            "hypotheses": [
                # cites a number that was never revealed -> trace_unrevealed
                # must be > 0
                {"name": "占位假说", "status": "supported", "evidence_for": ["EV-ZZZ9"]},
                # claims exclusion while giving no counter-evidence ->
                # trace_pseudo_excluded must be > 0
                {"name": "被排除的占位", "status": "excluded", "evidence_against": []},
            ],
            "tool_calls": [],
            # never writes falsified_by -> trace_commitments_falsifiable must be 0.0
            "commitments": [{"claim": "预计不会恶化"}],
            "data_notes": [],
            "scope": "",
        }
        out._raw = raw                                    # noqa: SLF001
        return out


class BlindConfidentSolver(Solver):
    """`self_discovery` lower bound: queries nothing, then declares the data
    sufficient (`sd_blind_sufficient` must hit, `sd_coverage` must be 0). Does not
    read the gold label.
    """

    name = "blind_confident"

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        raw = dict(getattr(out, "_raw", {}) or {})
        # Enum value follows `discovery.SUFFICIENT`.
        raw["data_quality"] = {
            "data_sufficiency": "sufficient",
            "signal_quality": {"weight": "体重记录齐全,足以判断"},
        }
        out._raw = raw                                    # noqa: SLF001
        return out


class GatedShotgunSolver:
    """Under the on-demand query geometry, orders not-yet-ordered test items
    cheapest first until the budget runs out, then gives a constant answer —
    the upper bound of "order everything" once ordering has a cost.
    """

    name = "gated_shotgun"

    def __init__(self) -> None:
        self._asked: set[str] = set()

    def solve(self, payload):
        import json as _json
        ctx = getattr(self, "gated_context", {}) or {}
        menu = [i for i in (ctx.get("menu") or []) if i.get("is_test")]
        try:
            spent = float(ctx.get("spent") or 0.0)
            budget = float(ctx.get("budget") or 0.0)
        except (TypeError, ValueError):
            spent, budget = 0.0, 0.0
        left = budget - spent
        plan = []
        for it in sorted(menu, key=lambda x: float(x.get("cost") or 0.0)):
            t = str(it.get("target"))
            c = float(it.get("cost") or 0.0)
            if t in self._asked or c > left:
                continue
            plan.append({"kind": it.get("kind"), "target": t})
            self._asked.add(t)
            left -= c
        out = ConstantDdxSolver().solve(payload)
        if plan:
            out._raw_text = _json.dumps({"queries": plan, "commit": False}, ensure_ascii=False)
            out._raw = {}
            return out
        _ans = dict(getattr(out, "_raw", {}) or {})
        if not (_ans.get("differential") or _ans.get("forecast") or _ans.get("drivers")):
            _ans["differential"] = [{"rank": 1, "diagnosis": "多囊卵巢综合征"}]
        out._raw = _ans
        out._raw_text = _json.dumps(_ans, ensure_ascii=False)
        return out


class GatedSignalFloodSolver:
    """Under the on-demand query geometry, queries every monitoring signal
    cheapest first until the budget runs out, then declares the data sufficient:
    the upper bound for `sd_coverage` and the counterpart of `blind_confident`.
    """

    name = "gated_flood"

    def __init__(self) -> None:
        self._asked: set[str] = set()

    def solve(self, payload):
        import json as _json
        ctx = getattr(self, "gated_context", {}) or {}
        # Unlike `gated_shotgun`, this selects monitoring signals, not test items.
        menu = [i for i in (ctx.get("menu") or []) if not i.get("is_test")]
        try:
            spent = float(ctx.get("spent") or 0.0)
            budget = float(ctx.get("budget") or 0.0)
        except (TypeError, ValueError):
            spent, budget = 0.0, 0.0
        left = budget - spent
        plan = []
        for it in sorted(menu, key=lambda x: float(x.get("cost") or 0.0)):
            t = str(it.get("target"))
            c = float(it.get("cost") or 0.0)
            if t in self._asked or c > left:
                continue
            plan.append({"kind": it.get("kind"), "target": t})
            self._asked.add(t)
            left -= c
        out = ConstantDdxSolver().solve(payload)
        if plan:
            out._raw_text = _json.dumps({"queries": plan, "commit": False}, ensure_ascii=False)
            out._raw = {}
            return out
        _ans = dict(getattr(out, "_raw", {}) or {})
        if not (_ans.get("differential") or _ans.get("forecast") or _ans.get("drivers")):
            _ans["differential"] = [{"rank": 1, "diagnosis": "多囊卵巢综合征"}]
        _ans["data_quality"] = {**(_ans.get("data_quality") or {}),
                                "data_sufficiency": "sufficient"}
        out._raw = _ans
        out._raw_text = _json.dumps(_ans, ensure_ascii=False)
        return out


class TestOrderingSolver(Solver):
    """Orders tests without reading the case: `shotgun` orders the whole test
    catalogue, `panel` a fixed standard panel. Gives `tests_recall` /
    `tests_precision` / `disc_recall` their ignore-the-item references.
    """

    #: Fixed standard panel, without near-name discriminating tests.
    PANEL = ("血常规", "肝功能", "肾功能", "空腹血糖", "糖化血红蛋白",
             "血脂四项", "甲功三项", "尿常规")

    def __init__(self, mode: str = "shotgun") -> None:
        if mode not in ("shotgun", "panel"):
            raise ValueError(f"mode must be shotgun / panel, got {mode!r}")
        self.mode = mode
        self._catalogue: tuple[str, ...] | None = None

    def _tests(self) -> list[str]:
        if self.mode == "panel":
            return list(self.PANEL)
        if self._catalogue is None:
            # The same catalogue for every case, from `gated.test_catalogue()`.
            from .gated import test_catalogue
            self._catalogue = tuple(test_catalogue())
        return list(self._catalogue)

    def solve(self, payload: SolverPayload) -> SolverOutput:
        out = ConstantDdxSolver().solve(payload)
        raw = dict(getattr(out, "_raw", {}) or {})
        raw["tests_to_order"] = self._tests()
        out._raw = raw                                    # noqa: SLF001 -- same as ConstantDdxSolver
        return out
