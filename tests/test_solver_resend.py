"""Bounded resume retries preserve paid receipts and never buy an answer twice."""
import hashlib
import json
import threading
from decimal import Decimal

import pytest

from haenv.paid_completion import AccountedCompletion, UnknownCostAttemptFailure
from haenv.semantic_budget import BudgetLedger
from haenv.semantic_transport import PriceSchedule
from haenv.solver_accounting import SolverRequestAccountant
from haenv.solvers import OpenAICompatSolver
from haenv.resume import _done_keys


def cursor(tmp_path, *, allowed=False, limit=2, content=True):
    paid = AccountedCompletion(
        PriceSchedule('model', Decimal('.000001'), Decimal('.000002'), Decimal(0), 100000),
        BudgetLedger(tmp_path / 'budget.json', limit_usd='1'), tmp_path / 'receipts')
    solver = OpenAICompatSolver('m', 'model', 'fake', 1, retries=0, max_tokens=1000)
    solver.proxy = 'direct'
    accountant = SolverRequestAccountant(paid, 'cell', threading.Event(), {'max_retries': 0},
        cell_v2='content-cell' if content else None, model='m', case_id='c',
        resend_failed=allowed, max_resends=limit)
    return accountant, solver, paid


def fail(_):
    raise TimeoutError('synthetic read timeout')


@pytest.mark.parametrize('content', [False, True])
def test_failed_unknown_attempt_gets_new_id_and_success_replays(tmp_path, content):
    a, s, paid = cursor(tmp_path, content=content)
    with pytest.raises(UnknownCostAttemptFailure):
        a.request(s, 'q', fail)
    first = a.last_request_id
    old = {p.name: p.read_bytes() for p in paid.receipts.glob('*.failed.json')}
    b, s, paid = cursor(tmp_path, allowed=True, content=content)
    sent = []
    response = {'usage': {'cost': .001}, 'choices': []}
    assert b.request(s, 'q', lambda _: sent.append(b.last_request_id) or response) == response
    assert b.last_request_id == first + ':resend:1'
    c, s, paid = cursor(tmp_path, allowed=True, content=content)
    assert c.request(s, 'q', lambda _: pytest.fail('rebought success')) == response
    assert c.last_request_id == b.last_request_id
    assert sent == [first + ':resend:1']
    assert old == {p.name: p.read_bytes() for p in paid.receipts.glob('*.failed.json')}
    assert paid.ledger.record(first)['status'] == 'unknown_capped'
    assert paid.ledger.record(first)['actual_usd'] is None


def test_limit_stops_without_new_reservation_and_abort_is_terminal(tmp_path):
    for round in range(3):
        a, s, paid = cursor(tmp_path, allowed=round > 0)
        with pytest.raises(UnknownCostAttemptFailure):
            a.request(s, 'q', fail)
    a, s, paid = cursor(tmp_path, allowed=True)
    from haenv.solver_accounting import ResendLimitReached
    with pytest.raises(ResendLimitReached, match='Resend limit reached.*2') as caught:
        a.request(s, 'q', lambda _: pytest.fail('limit exceeded'))
    with pytest.raises(ResendLimitReached):
        a.request(s, 'different prompt', lambda _: pytest.fail('new prompt after limit'))
    assert len(paid.ledger.snapshot()['requests']) == 3
    row = {'case': 'c', 'solver': 'm', 'overall': 'ABORT(no_response)',
           'failed_attempts': [{'error': str(caught.value)}]}
    path = tmp_path / 'eval.jsonl'; path.write_text(json.dumps(row) + '\n')
    assert _done_keys(path) == {'c|m'}


def test_model_failure_is_not_advanced_even_when_resume_is_enabled(tmp_path):
    a, s, paid = cursor(tmp_path)
    def broken(_):
        raise ValueError('invalid model response')
    with pytest.raises(UnknownCostAttemptFailure):
        a.request(s, 'q', broken)
    a, s, paid = cursor(tmp_path, allowed=True)
    with pytest.raises(RuntimeError, match='not reissued'):
        a.request(s, 'q', lambda _: pytest.fail('model failure was resent'))
    assert len(paid.ledger.snapshot()['requests']) == 1


def test_settled_answer_and_same_resume_cursor_do_not_duplicate_send(tmp_path):
    a, s, paid = cursor(tmp_path)
    response = {'usage': {'cost': .001}, 'choices': []}
    a.request(s, 'q', lambda _: response)
    a, s, paid = cursor(tmp_path, allowed=True)
    a.request(s, 'q', lambda _: pytest.fail('settled answer was resent'))
    assert len(paid.ledger.snapshot()['requests']) == 1


@pytest.mark.parametrize('reason,allowed', [('TimeoutError: timed out', True),
    ('HTTP Error 401: Unauthorized', True), ('ValueError: invalid JSON', False)])
def test_batch_binding_uses_existing_cell_cause(tmp_path, reason, allowed):
    from haenv.solver_accounting import BatchAccounting, wire_settings, _hash
    root = tmp_path / 'solver-accounting'; root.mkdir()
    a, s, paid = cursor(root)
    settings = {'m': wire_settings(s)}
    run = BatchAccounting(paid.ledger, {'m': paid.prices}, settings, root, 'run', {'paid': {'max_resends': 2}})
    run.bind(s, 'c', 'm')
    # Seed a real failed paid attempt of precisely the bound cell.
    with pytest.raises(UnknownCostAttemptFailure):
        s._accounting.request(s, 'q', fail)
    first = s._accounting.last_request_id
    (tmp_path / 'eval.jsonl').write_text(json.dumps({'case': 'c', 'solver': 'm',
        'overall': 'ABORT(no_response)', 'failed_attempts': [{'error': reason}]}) + '\n')
    run = BatchAccounting(paid.ledger, {'m': paid.prices}, settings, root, 'run', {'paid': {'max_resends': 2}})
    run.bind(s, 'c', 'm')
    assert s._accounting.resend_failed is allowed
    if allowed:
        s._accounting.request(s, 'q', lambda _: {'usage': {'cost': .001}})
        assert s._accounting.last_request_id == first + ':resend:1'
    else:
        with pytest.raises(RuntimeError, match='not reissued'):
            s._accounting.request(s, 'q', lambda _: pytest.fail('model result changed'))


def test_interrupted_sent_request_is_capped_before_resend(tmp_path):
    from haenv.paid_completion import begin_inflight
    from haenv.solver_accounting import v2_request_id
    a, s, paid = cursor(tmp_path, allowed=True)
    first = v2_request_id('content-cell', hashlib.sha256(b'q').hexdigest(), 1)
    paid.ledger.reserve(first, '.01')
    receipt = paid.receipts / (hashlib.sha256(first.encode()).hexdigest() + '.json')
    begin_inflight(receipt.with_suffix(".inflight.json"), request_id=first)
    a.request(s, 'q', lambda _: {'usage': {'cost': .001}})
    assert paid.ledger.record(first)['status'] == 'unknown_capped'
    assert a.last_request_id == first + ':resend:1'


@pytest.mark.parametrize('status', [401, 403, 429, 503])
def test_http_infrastructure_failure_can_resume(tmp_path, status):
    import urllib.error
    a, s, paid = cursor(tmp_path)
    def rejected(_):
        raise urllib.error.HTTPError('http://local.test', status, 'synthetic refusal', {}, None)
    with pytest.raises(RuntimeError):
        a.request(s, 'q', rejected)
    first = a.last_request_id
    a, s, paid = cursor(tmp_path, allowed=True)
    a.request(s, 'q', lambda _: {'usage': {'cost': .001}})
    assert a.last_request_id == first + ':resend:1'


def test_legacy_index_groups_resend_receipts_with_original_request(tmp_path):
    from haenv.solver_accounting import RequestIndex, _hash, cell_v2
    run_id = 'run'; cid = _hash([run_id, 'c', 'm'])
    receipts = tmp_path / 'receipts'; receipts.mkdir()
    for rid in [f'solver:{cid}:1', f'solver:{cid}:1:resend:1']:
        (receipts / (hashlib.sha256(rid.encode()).hexdigest() + '.json')).write_text(json.dumps(
            {'request': {'request_id': rid, 'prompt_sha256': 'p'}}))
    index = RequestIndex.build(tmp_path, {'run_id': run_id, 'identity': {'settings': {'m': {}}}}, ['c'], run_id)
    assert index.get(cell_v2(run_id, 'c', 'm'), 'p', 1) == f'solver:{cid}:1'
    assert len(index.entries) == 1


def test_failed_receipt_without_ledger_is_booked_before_resend(tmp_path):
    from haenv.paid_completion import AccountedCompletion
    a, s, paid = cursor(tmp_path)
    with pytest.raises(UnknownCostAttemptFailure):
        a.request(s, 'q', fail)
    first = a.last_request_id
    restored = BudgetLedger(tmp_path / 'restored.json', limit_usd='1')
    fresh_paid = AccountedCompletion(paid.prices, restored, paid.receipts)
    a = SolverRequestAccountant(fresh_paid, 'cell', threading.Event(), cell_v2='content-cell',
                                resend_failed=True, max_resends=2)
    a.request(s, 'q', lambda _: {'usage': {'cost': .001}})
    assert restored.record(first)['status'] == 'unknown_capped'
    assert restored.snapshot()['unknown_capped_usd'] != '0'


def test_legacy_failed_receipt_of_another_prompt_is_not_a_resend(tmp_path):
    a, s, paid = cursor(tmp_path, content=False)
    with pytest.raises(UnknownCostAttemptFailure):
        a.request(s, 'old question', fail)
    first = a.last_request_id
    a, s, paid = cursor(tmp_path, content=False, allowed=True)
    a.request(s, 'different question', lambda _: {'usage': {'cost': .001}})
    assert a.last_request_id == 'solver:cell:2'
    assert a.sequence_skips == [first]


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_resend_limits_are_refused(limit):
    from haenv.config import validate
    with pytest.raises(ValueError):
        validate({"paid": {"max_resends": limit}})


def test_unparseable_http_200_envelope_can_resume_under_a_new_id(tmp_path):
    a, s, paid = cursor(tmp_path)
    def broken_envelope(_):
        error = json.JSONDecodeError('synthetic empty envelope', '', 0)
        error.haenv_wire = {'http_status': 200, 'response_started': True}
        raise error
    with pytest.raises(UnknownCostAttemptFailure):
        a.request(s, 'q', broken_envelope)
    first = a.last_request_id
    a, s, paid = cursor(tmp_path, allowed=True)
    a.request(s, 'q', lambda _: {'usage': {'cost': .001}})
    assert a.last_request_id == first + ':resend:1'
    assert paid.ledger.record(first)['status'] == 'unknown_capped'
