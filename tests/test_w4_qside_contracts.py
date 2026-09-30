"""Question-side contracts: the trend gold is built
on the answered window behind two gates, recompute rebuilds the same questions as the run,
noop is a three-value enum, `join_evidence` is checked by code, gold qualifiers carry a
derivability record, and the diagnosis framing asks for `differential[].certainty`.

Each item has a negative control that fails when the behaviour is removed.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

import haenv                                              # noqa: F401  mounts the kernel
from haenv import evaluate as E
from haenv import wq
from haenv.judges import _quant_truth, judge_noop_probe, judge_quant_probe
from haenv.quantities import resolve as _q

ROOT = pathlib.Path(haenv.__file__).resolve().parent.parent


class _Raw:
    def __init__(self, streams: dict, T: int = 100, cid: str = "W4-Q"):
        self.longitudinal_data = {k: [{"ts": t, "value": v} for t, v in pts]
                                  for k, pts in streams.items()}
        self.prediction_context = {"prediction_time_T": T}
        self.case_id = cid


class _QOut:
    def __init__(self, ans):
        self._raw = {"quant_answer": ans}


# ---------------------------------------------------------------- trend gates
def _rising(n=10, t0=0, step=1.0, base=100.0):
    return [(t0 + i, base + step * i) for i in range(n)]


def test_trend_window_gate_skips_a_series_that_flips_within_one_day():
    """`a` is rising up to day 9 and flat (last == first) with day 10 added; `b` is stable."""
    a = [(0, 100.0), (2, 104.0), (4, 108.0), (6, 104.0), (9, 106.0), (10, 100.0)]
    b = _rising(12)
    # Points after T make `a` the first candidate (candidates are ordered by point count).
    raw = _Raw({"a": a + [(12 + i, 100.0) for i in range(10)], "b": b}, T=9)
    assert _quant_truth("trend", [p for p in a if p[0] <= 9]) != \
        _quant_truth("trend", [p for p in a if p[0] <= 10])
    pr = E.build_quant_probe(raw, "trend", 9)
    assert pr and pr["kind"] == "trend" and pr["signal"] == "b", pr and pr["signal"]


def test_trend_net_change_gate_skips_a_series_with_no_direction():
    """`a` is `flat` on days 3, 4 and 5 alike (window-stable) but nets 1 of a 30 swing
    (< 25%); `b` rises steadily."""
    a = [(0, 100.0), (1, 120.0), (2, 90.0), (3, 100.0), (4, 101.0)]
    b = _rising(5)
    raw = _Raw({"a": a + [(20 + i, 101.0) for i in range(6)], "b": b}, T=4)
    assert E._trend_candidate_ok(raw.longitudinal_data["a"], 4)[1] == \
        {"window_stable": True, "net_change_ok": False}
    pr = E.build_quant_probe(raw, "trend", 4)
    assert pr and pr["signal"] == "b", pr and pr["signal"]


def test_trend_falls_back_to_peak_value_when_every_series_fails_the_gates():
    flat_swing = [(0, 100.0), (1, 130.0), (2, 90.0), (3, 100.0), (4, 100.5)]
    raw = _Raw({"a": flat_swing}, T=4)
    pr = E.build_quant_probe(raw, "trend", 4)
    assert pr and pr["kind"] == "peak_value" and pr.get("fallback_from") == "trend", pr


def test_trend_is_built_on_the_answer_window_so_the_oracle_is_always_right():
    """The judge recomputes truth on the answered window; the builder must use the same one."""
    s = [(0, 100.0), (10, 110.0), (20, 120.0), (30, 130.0), (40, 125.0), (50, 90.0),
         (60, 80.0), (70, 70.0)]
    raw = _Raw({"s": s}, T=70, cid="W4-AW")
    built = {"W4-AW": raw}
    import haenv.evaluate as _ev
    # pin the rotation to trend for this id
    orig = _ev._QUANT_KINDS
    try:
        _ev._QUANT_KINDS = ("trend",)
        qp = E.assign_quant_probes(built, answer_t={"W4-AW": 30})
    finally:
        _ev._QUANT_KINDS = orig
    pr = qp["W4-AW"]
    assert pr["T"] == 30
    j = judge_quant_probe(_QOut(pr["truth"]), pr, t_max=30)
    assert _q(j, "quant_ok") == 1.0


def test_answer_window_does_not_touch_other_quant_kinds():
    s = [(0, 1.0), (1, 5.0), (2, 2.0), (3, 3.0), (4, 2.5), (5, 1.0)]
    raw = _Raw({"s": s}, T=5, cid="W4-PK")
    import haenv.evaluate as _ev
    orig = _ev._QUANT_KINDS
    try:
        _ev._QUANT_KINDS = ("peak_day",)
        a = json.dumps(E.assign_quant_probes({"W4-PK": raw}), sort_keys=True)
        b = json.dumps(E.assign_quant_probes({"W4-PK": raw}, answer_t={"W4-PK": 3}), sort_keys=True)
    finally:
        _ev._QUANT_KINDS = orig
    assert a == b


def test_run_and_recompute_share_one_assignment_that_avoids_the_noop_stream():
    """`recompute_judges` assigns probes with `avoid`, exactly as the run does; without it,
    rows are marked stale for a question that has not changed."""
    npr, qpr = E.assign_qside_probes({"W4-AV": _Raw({"x": _rising(30), "y": _rising(8)},
                                                     T=29, cid="W4-AV")})
    tgt = (npr.get("W4-AV") or {}).get("target")
    if tgt and qpr.get("W4-AV"):
        assert qpr["W4-AV"]["signal"] != tgt
    src = (ROOT / "tools" / "recompute_judges.py").read_text(encoding="utf-8")
    assert "assign_qside_probes(" in src and "assign_quant_probes(_built)" not in src


def test_recompute_reads_the_answer_window_from_the_rows(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import recompute_judges as RJ
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (
        {"case": "A", "solver": "s", "overall": "ABORT(leak)"},
        {"case": "A", "solver": "t", "slices": [5, 9, 30]},
        {"case": "B", "solver": "s", "quant_t_max": 12, "slices": [3, 40]},
        {"case": "C", "solver": "s", "T": 77})), encoding="utf-8")
    assert RJ._answer_t_from_rows(p) == {"A": 30, "B": 12, "C": 77}


# ---------------------------------------------------------------- noop enum
class _NOut:
    def __init__(self, sq=None, suff="sufficient"):
        self.data_quality = {"data_sufficiency": suff, "signal_quality": sq or {}}


_GAP = {"target": "s", "polarity": "gap", "truth_present": False, "window": [10, 30],
        "n_pts_in_window": 0, "T": 60, "contract": "enum-v1"}
_COV = {**_GAP, "polarity": "covered", "truth_present": True, "n_pts_in_window": 9}


@pytest.mark.parametrize("probe,value,want", [
    (_GAP, "no_data_in_window", 1.0),
    (_GAP, "no_data_in_window;第 10–30 天没有读数", 1.0),
    (_COV, "present", 1.0),
    (_COV, "unreliable", 1.0),
    (_COV, None, 0.0),                       # silent on a covered window does not score
    (_COV, "no_data_in_window", 0.0),
    (_GAP, "present", 0.0),
    (_GAP, None, 0.0),
    (_GAP, "该时段无读数", 0.0),              # free text is off-enum
    (_GAP, "no_data_in_window_10_30", 0.0),
])
def test_noop_is_judged_on_the_enum(probe, value, want):
    sq = {} if value is None else {"s": value}
    r = judge_noop_probe(_NOut(sq), probe)
    assert r["noop_ok"] == want, r


def test_noop_off_enum_is_recorded():
    r = judge_noop_probe(_NOut({"s": "该时段无读数"}), _GAP)
    assert r["noop_off_enum"] is True and r["noop_answer"] is None


def test_noop_probe_carries_the_contract_and_the_question_lists_the_enum():
    raw = _Raw({"s": [(0, 1.0), (40, 2.0), (41, 2.0), (42, 2.1), (43, 2.0), (44, 2.0)]}, T=60)
    pr = E.build_noop_probe(raw, "absent", 60)
    assert pr["contract"] == "enum-v1"
    E.NOOP_FOR.clear()
    E.NOOP_FOR["W4-Q"] = pr

    class _P:
        case_id = "W4-Q"
        prediction_context = {}
    txt = E._noop_suffix(_P())
    E.NOOP_FOR.clear()
    for v in ("present", "no_data_in_window", "unreliable"):
        assert f"`{v}`" in txt


def test_noop_oracle_stub_answers_the_enum_on_both_sides():
    from haenv.baselines import OracleProbeSolver
    E.NOOP_FOR.clear()
    E.QUANT_FOR.clear()
    try:
        for probe in (_GAP, _COV):
            E.NOOP_FOR["W4-OR"] = probe

            class _P:
                case_id = "W4-OR"
                prediction_context = {"case_id": "W4-OR", "prediction_time_T": 60,
                                      "target_event_type": "weight_regain"}
                user_profile = {}
                longitudinal_data = {}
                evidence_ledger = []
            right = OracleProbeSolver("right").solve(_P())
            wrong = OracleProbeSolver("wrong").solve(_P())
            assert judge_noop_probe(right, probe)["noop_ok"] == 1.0
            assert judge_noop_probe(wrong, probe)["noop_ok"] == 0.0
    finally:
        E.NOOP_FOR.clear()


# ---------------------------------------------------------------- join_evidence
class _VP:
    def __init__(self, cid, T=100):
        self.case_id = cid
        self.T = T
        self.adjudication = {}


def _register_join_case(cid, contract=True):
    ids = {f"EV-{cid}-{i:02d}": f"EV-{cid}-{i:02d}" for i in range(1, 7)}
    man = {"real_symptom_evidence_ids": [f"EV-{cid}-01", f"EV-{cid}-02", f"EV-{cid}-05"],
           "lookalike_evidence_ids": [f"EV-{cid}-03"],
           "benign_evidence_ids": [f"EV-{cid}-04"],
           "event_schedule": [
               {"evidence_id": f"EV-{cid}-01", "kind": "real_symptom", "day": 10},
               {"evidence_id": f"EV-{cid}-02", "kind": "real_symptom", "day": 40},
               {"evidence_id": f"EV-{cid}-03", "kind": "lookalike", "day": 50},
               {"evidence_id": f"EV-{cid}-04", "kind": "benign_symptom", "day": 20},
               {"evidence_id": f"EV-{cid}-05", "kind": "real_symptom", "day": 150}],
           "id_map": ids}
    if contract:
        from haenv.framings import DDX_ANSWER_CONTRACT
        man["answer_contract"] = dict(DDX_ANSWER_CONTRACT)
    wq.register_injection(cid, man)


class _JOut:
    def __init__(self, ev):
        self._raw = {"join_type": "unified", "join_evidence": ev}


@pytest.mark.parametrize("ev,want", [
    (["EV-W4J-01", "EV-W4J-02"], 1.0),
    (["EV-W4J-01", "EV-W4J-03"], 1.0),                # a look-alike belongs to the gold presentation
    (["EV-01", "EV-02"], 1.0),                        # unique abbreviations resolve
    (["EV-W4J-01"], 0.0),                             # one id
    (["EV-W4J-01", "EV-W4J-01"], 0.0),                # the same id twice
    (["EV-W4J-01", "EV-W4J-99"], 0.0),                # an id not in the ledger
    (["EV-W4J-01", "EV-W4J-04"], 0.0),                # a benign distractor
    (["EV-W4J-01", "EV-W4J-05"], 0.0),                # a symptom after the judged time
    (None, 0.0),
])
def test_join_evidence_is_checked_by_code(ev, want):
    from haenv.judges import judge_join_evidence
    _register_join_case("W4J")
    r = judge_join_evidence(_JOut(ev), _VP("W4J", T=100))
    assert r["join_ev_ok"] == want, r


def test_join_evidence_not_applicable_without_the_contract():
    from haenv.judges import judge_join_evidence
    _register_join_case("W4JX", contract=False)
    r = judge_join_evidence(_JOut(["EV-W4JX-01", "EV-W4JX-02"]), _VP("W4JX"))
    assert r["join_ev_ok"] is None and r["join_ev_required"] is False


def test_join_evidence_is_mounted_like_join_type():
    from haenv.mount_table import MOUNT
    assert MOUNT["join_evidence"] == MOUNT["join_type"]


# ---------------------------------------------------------------- gold qualifiers
class _SP:
    def __init__(self, ledger=(), streams=None):
        self.evidence_ledger = list(ledger)
        self.longitudinal_data = dict(streams or {})


_LAB = [{"evidence_id": "EV-1", "source_type": "lab_result", "source_timestamp": 30,
         "symptom": "促甲状腺激素 8.1 mIU/L(参考 0.27–4.2)"}]


def test_qualifier_without_confirming_evidence_is_not_required():
    from haenv import qualifiers as QL
    core, recs = QL.qualifiers_for({"diagnosis": "原发性甲状腺功能减退(桥本)",
                                    "aliases": ["甲减", "桥本"]}, _SP(_LAB))
    assert core == "原发性甲状腺功能减退"
    assert [(r["family"], r["derivable"], r["required"]) for r in recs] == [("桥本", False, False)]


def test_qualifier_with_confirming_evidence_is_required():
    """Negative control of the derivability check: add a visible TPOAb result."""
    from haenv import qualifiers as QL
    led = _LAB + [{"evidence_id": "EV-2", "source_type": "lab_result", "source_timestamp": 31,
                   "symptom": "抗甲状腺过氧化物酶抗体(TPOAb) 阳性"}]
    _, recs = QL.qualifiers_for({"diagnosis": "原发性甲状腺功能减退(桥本)"}, _SP(led))
    assert recs[0]["derivable"] is True and recs[0]["derivable_from_day"] == 31


@pytest.mark.parametrize("fbg,want", [(7.3, True), (6.6, False)])
def test_glycemic_threshold_qualifier(fbg, want):
    from haenv import qualifiers as QL
    led = [{"evidence_id": "EV-G", "source_type": "lab_result", "source_timestamp": 5,
            "symptom": f"空腹血糖 {fbg} mmol/L(参考 3.9–6.1)"}]
    core, recs = QL.qualifiers_for({"diagnosis": "早期2型糖尿病/胰岛素抵抗"}, _SP(led))
    assert core == "胰岛素抵抗" and recs[0]["derivable"] is want


def test_comorbid_qualifier_names_its_thread():
    from haenv import qualifiers as QL
    ddx = {"diagnosis": "成人隐匿性自身免疫糖尿病(LADA) + 原发性甲状旁腺功能亢进(高钙血症)",
           "threads": [{"name": "胰岛自身免疫线", "aliases": ["lada", "自身免疫糖尿病"]},
                       {"name": "甲状旁腺线", "aliases": ["甲旁亢", "甲状旁腺"]}]}
    core, recs = QL.qualifiers_for(ddx, _SP())
    assert core == "糖尿病 + 原发性甲状旁腺功能亢进(高钙血症)"
    assert [(r["family"], r["thread"]) for r in recs] == [("LADA", "胰岛自身免疫线")]


def test_gold_without_a_family_is_left_untouched():
    from haenv import qualifiers as QL

    class _R:
        adjudication = {"ddx": {"diagnosis": "多囊卵巢综合征(PCOS)", "aliases": ["多囊"]}}
    r = _R()
    before = json.dumps(r.adjudication, ensure_ascii=False, sort_keys=True)
    assert QL.annotate(r, _SP()) is None
    assert json.dumps(r.adjudication, ensure_ascii=False, sort_keys=True) == before


# ---------------------------------------------------------------- certainty contract (diagnosis atoms)
def test_ddx_framings_ask_for_certainty_and_join_evidence():
    from haenv.framings import DDX_PROMPT, DDX_SCOPE_PROMPT, DDX_SCOPE2_PROMPT, DIFFERENTIAL_CERTAINTY
    assert DIFFERENTIAL_CERTAINTY == ("definite", "probable", "possible", "rule_out")
    for t in (DDX_PROMPT, DDX_SCOPE_PROMPT, DDX_SCOPE2_PROMPT):
        assert '"certainty":"definite|probable|possible|rule_out"' in t
        assert all(f"`{v}`" in t for v in DIFFERENTIAL_CERTAINTY)
        assert '"join_evidence":["EV-...","EV-..."]' in t
        # The judged list cap (gold threads + 3) is never below 4, so this sentence is never stricter
        assert "只有排在最前面的 4 个未被排除的候选会被计分" in t


def test_an_answer_with_certainty_survives_parsing():
    """Positive control for `dx_affirmed`: kernel `solver._extract_json` ->
    `evaluate._to_output` -> `tracks._differential` keep `certainty` on each candidate."""
    from solver import _extract_json
    from haenv.tracks import _differential
    txt = ('```json\n{"differential":[{"rank":1,"diagnosis":"甲减","certainty":"probable",'
           '"supporting_evidence":["EV-1"],"ruled_out_by":null},{"rank":2,"diagnosis":"亚临床甲减",'
           '"certainty":"rule_out","supporting_evidence":["EV-1"],"ruled_out_by":"EV-2"}],'
           '"join_type":"unified","join_evidence":["EV-1","EV-3"],'
           '"action":{"selected_action_class":"A2"},'
           '"data_quality":{"data_sufficiency":"sufficient","signal_quality":{}}}\n```')

    class _P:
        case_id = "W4-C"
        prediction_context = {"case_id": "W4-C", "prediction_time_T": 10}
    out = E._to_output(_extract_json(txt), _P())
    assert [x.get("certainty") for x in _differential(out)] == ["probable", "rule_out"]


def test_build_records_the_answer_contract_on_the_question_side_only():
    """A diagnosis case built by the generator carries `answer_contract` in the Q-side ledger,
    never in `adjudication`; the `桥本` (Hashimoto) case gets a qualifier record, and the gold
    text is kept."""
    from haenv import job as J
    from haenv.build import build_case
    cs = next(c for c in J.load_job(ROOT / "inputs" / "ddx-timeline.job.yaml").cases
              if c.case_id == "JD-03")
    raw, audit = build_case(cs)
    assert raw is not None, audit.get("post_noise_conflicts")
    man = wq.injected_manifest(cs.case_id)
    assert man["answer_contract"]["differential_certainty"] == "required"
    assert "answer_contract" not in json.dumps(raw.adjudication, ensure_ascii=False)
    ddx = raw.adjudication["ddx"]
    assert ddx["diagnosis"] == "原发性甲状腺功能减退(桥本)"
    assert [(q["family"], q["required"]) for q in ddx["qualifiers"]] == [("桥本", False)]


# ---------------------------------------------------------------- recompute on existing answers
# Every row is judged on the question it was asked: an answer from a pack built before the
# enum contract is judged by the marker rule it was asked under, and a quant answer on the
# series its row records, with the truth recomputed on the answered window.
_LEGACY_GAP = {"target": "s", "polarity": "gap", "truth_present": False, "window": [10, 30],
               "n_pts_in_window": 0}
_LEGACY_COV = {**_LEGACY_GAP, "polarity": "covered", "truth_present": True, "n_pts_in_window": 9}


@pytest.mark.parametrize("probe,value,want", [
    (_LEGACY_GAP, "该时段无读数", 1.0),         # free text declared absence under the marker contract
    (_LEGACY_GAP, "no_data_in_window_10_30", 1.0),
    (_LEGACY_COV, None, 1.0),                 # silence on a covered window is the marker-contract answer
    (_LEGACY_COV, "missing", 0.0),
    (_LEGACY_GAP, None, 0.0),
])
def test_a_probe_without_a_contract_is_judged_by_the_marker_rule(probe, value, want):
    sq = {} if value is None else {"s": value}
    r = judge_noop_probe(_NOut(sq), probe)
    assert r["noop_ok"] == want, r
    assert "noop_answer" not in r and "noop_off_enum" not in r   # the marker-contract row's field set


def _rj():
    sys.path.insert(0, str(ROOT / "tools"))
    import recompute_judges as RJ
    return RJ


class _RawQ(_Raw):
    pass


def test_recompute_judges_an_old_row_on_its_recorded_questions():
    """A row without `noop_contract`, whose recorded trend series differs from the one the
    current builder picks, is judged, not marked stale; its noop goes through the marker rule."""
    RJ = _rj()
    case = _Raw({"a": [(0, 100.0), (1, 101.0), (2, 100.5), (3, 100.2), (4, 100.1)],
                 "b": _rising(5)}, T=4, cid="W4-OLD")
    row = {"case": "W4-OLD", "solver": "m", "T": 4,
           "noop_target": "s", "noop_polarity": "gap", "noop_truth_present": False,
           "noop_window": [10, 30], "noop_n_pts_in_window": 0,
           "quant_kind": "trend", "quant_signal": "a"}
    new_noop = {**_GAP}                                   # the rebuilt probe carries the enum contract
    new_quant = E.build_quant_probe(case, "trend", 4)     # the current builder picks another series
    assert new_quant["signal"] != "a"

    class _O:
        data_quality = {"data_sufficiency": "sufficient", "signal_quality": {"s": "无读数"}}
        _raw = {"quant_answer": "flat"}
    add = RJ.rejudge_qside(row, _O(), case, new_noop, new_quant, t_max=4)
    assert "qside_probe_stale" not in add, add
    # a stale mark already on the row is cleared once the row is judged
    assert RJ.rejudge_qside({**row, "qside_probe_stale": "quant:…"}, _O(), case, new_noop,
                            new_quant, t_max=4)["qside_probe_stale"] is None
    assert add["noop_ok"] == 1.0 and add["quant_signal"] == "a"
    assert add["quant_truth"] == _quant_truth("trend", [(0, 100.0), (1, 101.0), (2, 100.5),
                                                         (3, 100.2), (4, 100.1)])


def test_recompute_judges_a_new_row_by_its_contract():
    RJ = _rj()
    case = _Raw({"b": _rising(8)}, T=7, cid="W4-NEW")
    new_quant = E.build_quant_probe(case, "trend", 7)
    row = {"case": "W4-NEW", "solver": "m", "T": 7, **{f"noop_{k}": v for k, v in (
        ("target", "s"), ("polarity", "gap"), ("truth_present", False), ("window", [10, 30]),
        ("n_pts_in_window", 0), ("contract", "enum-v1"))},
        "quant_kind": "trend", "quant_signal": "b"}

    class _O:
        data_quality = {"data_sufficiency": "sufficient", "signal_quality": {"s": "无读数"}}
        _raw = {"quant_answer": new_quant["truth"]}
    add = RJ.rejudge_qside(row, _O(), case, dict(_GAP), new_quant, t_max=7)
    assert add["noop_ok"] == 0.0 and add["noop_off_enum"] is True    # enum contract: free text is off-enum
    assert add["quant_ok"] == 1.0 and "qside_probe_stale" not in add


def test_recompute_marks_stale_only_when_the_recorded_series_is_gone():
    RJ = _rj()
    case = _Raw({"b": _rising(8)}, T=7, cid="W4-GONE")
    row = {"case": "W4-GONE", "solver": "m", "T": 7, "quant_kind": "trend", "quant_signal": "zz"}

    class _O:
        data_quality = {}
        _raw = {"quant_answer": "rising"}
    add = RJ.rejudge_qside(row, _O(), case, None, E.build_quant_probe(case, "trend", 7), t_max=7)
    assert "quant:" in add.get("qside_probe_stale", "") and "quant_ok" not in add
