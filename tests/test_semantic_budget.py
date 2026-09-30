"""A shared persistent budget reserves before dispatch and does not forgive unknown cost."""
from decimal import Decimal

import pytest

from haenv.semantic_budget import BudgetExceeded, BudgetLedger


def test_reservation_prevents_overspend_and_actual_cost_releases_headroom(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1.00")
    ledger.reserve("request-1", ".70")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("request-2", ".40")
    ledger.settle("request-1", ".20")
    ledger.reserve("request-2", ".40")
    assert ledger.snapshot()["committed_usd"] == "0.60"


def test_reopen_preserves_spending_and_cannot_raise_budget(tmp_path):
    path = tmp_path / "budget.json"
    BudgetLedger(path, limit_usd="1").reserve("first", ".8")
    second = BudgetLedger(path, limit_usd="1")
    with pytest.raises(BudgetExceeded):
        second.reserve("second", ".3")
    # The cap lives on the ledger file: a process given another value cannot raise it.
    reopened = BudgetLedger(path, limit_usd="100")
    assert str(reopened.limit) == "1"
    with pytest.raises(BudgetExceeded):
        reopened.reserve("third", ".3")


def test_duplicate_request_does_not_get_a_new_budget_slot(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    ledger.reserve("same", ".2")
    with pytest.raises(ValueError):
        ledger.reserve("same", ".2")


def legacy_unknown(ledger, request_id, bound):
    """A ledger record in the 'unknown cost halts' form."""
    ledger.reserve(request_id, bound)
    with ledger._locked() as state:
        state["requests"][request_id]["status"] = "unknown"
        state["halt_reason"] = "A billed request has unknown cost; reconciliation required"
        ledger._write(state)


def test_unknown_cost_retains_full_reservation_and_does_not_block_further_calls(tmp_path):
    # The rule: an unknown cost is counted at its bound, no halt; only the cap refuses.
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    ledger.reserve("timeout", ".4")
    ledger.settle("timeout", None)
    assert Decimal(ledger.snapshot()["committed_usd"]) == Decimal(".4")
    assert ledger.snapshot()["halt_reason"] is None
    ledger.reserve("next", ".1")
    with pytest.raises(BudgetExceeded, match="Budget exhausted"):
        ledger.reserve("big", ".6")


def test_charge_above_reservation_records_violation_and_stops(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    ledger.reserve("one", ".1")
    with pytest.raises(BudgetExceeded):
        ledger.settle("one", ".2")
    assert Decimal(ledger.snapshot()["committed_usd"]) == Decimal(".2")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("two", ".1")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "0"])
def test_invalid_reservation_rejected(tmp_path, value):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    with pytest.raises(ValueError):
        ledger.reserve("one", value)


def test_valid_zero_charge_is_not_unknown(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    ledger.reserve("cached-input-but-fresh-output", ".1")
    ledger.settle("cached-input-but-fresh-output", "0")
    assert Decimal(ledger.snapshot()["committed_usd"]) == 0
    ledger.reserve("next", ".1")


def test_explicit_unknown_writeoff_keeps_full_cap_charge_and_exact_cost_missing(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    legacy_unknown(ledger, "lost", ".7")
    ledger.reconcile_unknown_at_bound("lost")
    snap = ledger.snapshot()
    assert snap["requests"]["lost"]["actual_usd"] is None
    assert snap["requests"]["lost"]["status"] == "unknown_capped"
    assert snap["committed_usd"] == "0.7" and snap["halt_reason"] is None
    with pytest.raises(BudgetExceeded):
        ledger.reserve("too_much", ".31")
    ledger.reserve("new_vote", ".3")
    with pytest.raises(ValueError):
        ledger.reserve("lost", ".1")


def test_writeoff_does_not_override_other_unknown_or_overbound_charge(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    legacy_unknown(ledger, "a", ".2")
    legacy_unknown(ledger, "b", ".2")
    ledger.reconcile_unknown_at_bound("a")
    assert ledger.snapshot()["halt_reason"]          # b is still a legacy unknown
    ledger.reserve("next", ".1")                      # but a legacy unknown halt does not stop
    with pytest.raises(BudgetExceeded):
        ledger.reserve("over", ".6")                  # the cap still does
    with pytest.raises(ValueError):
        ledger.reconcile_unknown_at_bound("a")


def test_verified_tariff_bound_is_counted_but_never_actual(tmp_path):
    ledger = BudgetLedger(tmp_path / 'budget.json', limit_usd='1')
    ledger.reserve('native', '.7')
    ledger.settle_bound('native', '.2', basis='verified-tariff-sha')
    state = ledger.snapshot();record = state['requests']['native']
    assert record['actual_usd'] is None and record['status']=='tariff_capped'
    assert record['original_reserved_usd']=='.7' or record['original_reserved_usd']=='0.7'
    assert state['committed_usd']=='0.2' and state['settled_usd']=='0'
    assert state['tariff_bound_usd']=='0.2'
    # Summing actual-or-reserved amounts per request yields the committed total.
    legacy = sum(Decimal(r['actual_usd'] if r['actual_usd'] is not None else r['reserved_usd'])
                 for r in state['requests'].values())
    assert legacy==Decimal('.2')
    with pytest.raises(BudgetExceeded):ledger.reserve('over', '.81')


def test_bound_cannot_erase_unknown_or_exceed_original_without_a_stop(tmp_path):
    ledger = BudgetLedger(tmp_path / 'budget.json', limit_usd='1')
    ledger.reserve('native', '.3')
    with pytest.raises(BudgetExceeded):ledger.settle_bound('native','.4',basis='verified')
    assert ledger.snapshot()['halt_reason']
    assert ledger.snapshot()['committed_usd']=='0.4'
    assert ledger.snapshot()['settled_usd']=='0'


def test_record_reads_the_current_version_and_returns_a_copy(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    other = BudgetLedger(tmp_path / "budget.json")
    assert ledger.record("a") is None
    other.reserve("a", "0.1")
    assert ledger.record("a")["status"] == "reserved"
    other.settle("a", "0.02")
    seen = ledger.record("a")
    assert seen["status"] == "settled" and seen["actual_usd"] == "0.02"
    seen["status"] = "tampered"
    assert ledger.record("a")["status"] == "settled"
    assert ledger.record("a") == ledger.snapshot()["requests"]["a"]
