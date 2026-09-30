"""`haenv report` semantic state: offline stub rows are not a failed judgment."""
from haenv.cli import _report_semantic_state, _semantic_exit_code


def row(solver, status):
    return {"solver": solver, "semantic": {"status": status}}


def test_stub_rows_only_is_not_applicable_and_exits_zero():
    state = _report_semantic_state([row("robust_ref", "pending"), row("const_ddx", "pending")], None)
    assert state["status"] == "not_applicable" and state["cells"] == 0
    assert _semantic_exit_code(state) == 0


def test_real_rows_all_resolved_complete_and_exit_zero():
    state = _report_semantic_state([row("gpt-x", "resolved"), row("robust_ref", "pending")], None)
    assert state["status"] == "completed" and state["states"] == {"resolved": 1}
    assert _semantic_exit_code(state) == 0


def test_real_rows_with_a_pending_cell_are_incomplete_and_exit_six():
    state = _report_semantic_state([row("gpt-x", "resolved"), row("gpt-y", "pending")], None)
    assert state["status"] == "incomplete" and state["pending"] == 1
    assert _semantic_exit_code(state) == 6


def test_unresolved_and_unanswered_cells_are_counted_and_keep_the_run_rule():
    # Same rule as `default_after_run`: completed only when every real-model cell is resolved.
    state = _report_semantic_state([row("gpt-x", "resolved"), row("gpt-x", "unresolved"),
                                    row("gpt-y", "empty_response"), row("gpt-y", "missing_response")], None)
    assert state["status"] == "incomplete"
    assert (state["pending"], state["unresolved"], state["no_answer"]) == (0, 1, 2)
    assert _semantic_exit_code(state) == 6
