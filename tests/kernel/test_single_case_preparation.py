"""Run-local preparation preserves delivery isolation and never reuses a stale gate."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from haenv.interaction import solve_guard
from haenv_kernel.schema import SolverPayload


def payload():
    return SolverPayload('patient', {'age_range': '40-44'}, {'prediction_time_T': 10},
                         {'weight': [{'ts': 1, 'value': 80}]}, [])


def test_prepared_delivery_reuses_probe_and_isolates_solver_edits(monkeypatch):
    probe, calls = solve_guard.leakage_probe, []

    def tracked(sp, T):
        calls.append(T)
        return probe(sp, T)

    monkeypatch.setattr(solve_guard, 'leakage_probe', tracked)
    original = payload()
    prepared = solve_guard.PreparedPayload(original, 10)
    seen = []

    class Solver:
        def solve(self, sp):
            seen.append(deepcopy(sp))
            sp.longitudinal_data['weight'][0]['value'] = -1
            sp.user_profile['age_range'] = 'changed'
            return SimpleNamespace()

    for _ in range(2):
        assert solve_guard.guarded_solve(Solver(), prepared.copy(), 10,
                                         prepared=prepared).usable
    assert calls == [10]
    assert seen == [original, original]
    assert prepared.copy() == original


@pytest.mark.parametrize('change', ['future', 'answer', 'window', 'dynamic', 'nested'])
def test_changed_delivery_cannot_inherit_a_clean_probe(change):
    prepared = solve_guard.PreparedPayload(payload(), 10)
    sp, T = prepared.copy(), 10
    if change == 'future':
        sp.longitudinal_data['weight'].append({'ts': 11, 'value': 79})
    elif change == 'answer':
        sp.user_profile['gold_driver'] = 'hidden'
    elif change == 'dynamic':
        sp.adjudication = {'diagnosis': 'hidden'}
    elif change == 'nested':
        sp.longitudinal_data['weight'][0]['outcome_label'] = 'hidden'
    else:
        T = 0

    class Solver:
        def solve(self, sp):
            pytest.fail('a leaking payload was delivered')

    facts = solve_guard.guarded_solve(Solver(), sp, T, prepared=prepared)
    assert facts.leak
    assert facts.out is None
    assert facts.abort_reason() == 'ABORT(leak)'


def test_non_json_extension_is_reprobed_even_when_dataclass_equality_hides_edits(monkeypatch):
    @dataclass
    class Extension:
        note: str = 'safe'

    original = payload()
    original.user_profile['extension'] = Extension()
    probe, calls = solve_guard.leakage_probe, []

    def tracked(sp, T):
        calls.append(T)
        return probe(sp, T)

    monkeypatch.setattr(solve_guard, 'leakage_probe', tracked)
    prepared = solve_guard.PreparedPayload(original, 10)
    assert prepared.probe(prepared.copy(), 10) == (True, [])
    assert calls == [10, 10]  # Unsupported objects cannot establish a cache identity.

    changed = prepared.copy()
    changed.user_profile['extension'].adjudication = {'diagnosis': 'hidden'}
    assert changed == original

    class Solver:
        def solve(self, sp):
            pytest.fail('a leaking extension was delivered')

    facts = solve_guard.guarded_solve(Solver(), changed, 10, prepared=prepared)
    assert facts.leak == ['forbidden_token:adjudication']
    assert facts.out is None
    assert calls == [10, 10, 10]


def test_single_preparation_invalidates_on_case_window_and_run_changes(monkeypatch):
    import haenv.scoring.rows as rows
    from haenv.runtime.state import RunContext
    from haenv_kernel.schema import RawCase

    raw = RawCase('case', {}, {'prediction_time_T': 10},
                  {'weight': [{'ts': 1, 'value': 80}]}, [], 'event_not_occurred', {}, [], {})
    build, calls = rows.build_instance, []

    def tracked(raw, T):
        calls.append(T)
        return build(raw, T)

    monkeypatch.setattr(rows, 'build_instance', tracked)
    # Stop after preparation: this test exercises invalidation independently of scoring.
    monkeypatch.setattr(solve_guard, 'guarded_solve', lambda *a, **kw: SimpleNamespace(leak=['stop']))
    ctx = RunContext()
    for _ in range(2):
        rows._row_single('case', 'solver', raw, 10, SimpleNamespace(), ctx=ctx)
    rows._row_single('case', 'solver', raw, 9, SimpleNamespace(), ctx=ctx)
    raw.user_profile['age_range'] = '45-49'
    rows._row_single('case', 'solver', raw, 10, SimpleNamespace(), ctx=ctx)
    rows._row_single('case', 'solver', raw, 10, SimpleNamespace(), ctx=RunContext())
    assert calls == [10, 9, 10, 10]
