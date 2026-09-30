"""Endpoint urgency and longitudinal urgency must not be silently conflated."""
from haenv.judges import trajectory as tr


def test_endpoint_urgency_uses_last_slice_not_first_success(monkeypatch):
    readings = iter([
        {"urgency_ok": True, "urgency_gap": 0},
        {"urgency_ok": False, "urgency_gap": -2},
    ])
    monkeypatch.setattr(tr, "judge_workup", lambda *a: next(readings))
    monkeypatch.setattr(tr, "judge_discriminative_tool", lambda *a: {})
    out = tr.judge_slices_workup([{"t": 1}, {"t": 2}], None, {})
    assert out["wk_urgency_ok_last"] is False
    assert out["wk_urgency_ok_rate"] == .5
    assert out["wk_n_slices_urgency_judged"] == 2
    assert out["wk_urgency_ok_at"] == 1
    assert out["wk_n_slices_urgency_ok"] == 1


def test_missing_last_slice_is_not_replaced_by_earlier_urgency(monkeypatch):
    readings = iter([{"urgency_ok": True}, {}])
    monkeypatch.setattr(tr, "judge_workup", lambda *a: next(readings))
    monkeypatch.setattr(tr, "judge_discriminative_tool", lambda *a: {})
    out = tr.judge_slices_workup([{"t": 1}, {"t": 2}], None, {})
    assert out["wk_urgency_ok_last"] is None
    assert out["wk_urgency_ok_rate"] == 1
    assert out["wk_n_slices_urgency_judged"] == 1


def test_empty_urgency_support_stays_unmeasured(monkeypatch):
    monkeypatch.setattr(tr, "judge_workup", lambda *a: {})
    monkeypatch.setattr(tr, "judge_discriminative_tool", lambda *a: {})
    out = tr.judge_slices_workup([], None, {})
    assert out["wk_urgency_ok_last"] is None and out["wk_urgency_ok_rate"] is None
    assert out["wk_n_slices_urgency_judged"] == 0


def test_saved_urgency_distinguishes_failed_answer_from_wrong_action():
    from haenv.semantic_report import urgency_from_saved_actions
    readings = [{"t": 2, "action": "A1", "raw_empty": True},
                {"t": 1, "action": "A5"}]
    result = urgency_from_saved_actions(readings, "🔴")
    assert result["wk_urgency_ok_last"] is None
    assert result["wk_urgency_ok_rate"] == 1.
    assert result["wk_n_slices_urgency_judged"] == 1
    readings[0]["raw_empty"] = False
    result = urgency_from_saved_actions(readings, "🔴")
    assert result["wk_urgency_ok_last"] is False
    assert result["wk_urgency_ok_rate"] == .5
