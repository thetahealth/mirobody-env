"""Component guarantees, exercised by the regression tables next to this module."""
from __future__ import annotations
from _paths import ROOT as _TEST_ROOT
import _referral_contracts_cases as _contract_cases
import pathlib
import sys
import pytest
import haenv_kernel.verifier as V
from haenv.scoring.analysis.gates import gate_multiplier
from haenv.scoring.analysis.decisions import review_macro_of
from haenv.judges import judge_slices_review_flag
from haenv.judges import judge_slices_safety_gates

def test_grade_keeps_unit_scoped_gates_off_the_case_list_on_slices():
    out, vp = (_contract_cases.review_Out('A1', ('TSH',)), _contract_cases.review_VP(True))
    assert _contract_cases.review_RG in V.grade(out, vp, []).hard_gate_failures
    rep = V.grade(out, vp, [], unit_gates_per_slice=True)
    assert not [g for g in rep.hard_gate_failures if g.split(':')[0] in V.UNIT_SCOPED_GATES]
    assert rep.overall == 'SCORED'
    assert V.UNIT_SCOPED_GATES == frozenset({_contract_cases.review_RG, _contract_cases.review_PC})

def test_slice_judge_hits_every_miss_once_last_slice_included():
    got = judge_slices_safety_gates(_contract_cases.review_slices(), _contract_cases.review_VP(True))
    hits = {g: got.get(f'sg_{g}_at') for g in (_contract_cases.review_RG, _contract_cases.review_PC)}
    assert hits == {_contract_cases.review_RG: '1,4', _contract_cases.review_PC: None}, hits
    assert got['slice_gate_unknown'] == 1 and got['slice_gate_n_judged'] == 3

def test_one_last_slice_miss_zeroes_one_unit_not_the_case():
    sl = _contract_cases.review_slices()
    vp = _contract_cases.review_VP(True)
    last = V.grade(_contract_cases.review_Out('A3'), vp, [], unit_gates_per_slice=True)
    row = {'case': 'C1', 'gates': last.hard_gate_failures, 'overall': last.overall, **judge_slices_safety_gates(sl, vp)}
    g = gate_multiplier([row])
    assert (g['n_units'], g['n_gated_units']) == (3, 0), g
    assert g['n_soft_gated_units'] == 2, g
    import haenv.analytics as _an
    from verifier_core.gate import multiplier as _vc
    old = _vc([row], non_harm_gates=_an.NON_HARM_GATES - _an.PROPENSITY_GATES)
    assert (old['n_units'], old['n_gated_units']) == (3, 2), old

def test_review_warranted_kept_when_last_slice_did_not_record():
    sl = _contract_cases.review_slices()[:3]
    got = judge_slices_review_flag(sl, _contract_cases.review_VP(True))
    assert got.get('review_warranted') is True
    assert 'review_flag_ok' not in got and 'review_declared' not in got
    assert judge_slices_review_flag(_contract_cases.review_slices(), _contract_cases.review_VP(False))['review_flag_ok'] == 1
from pathlib import Path
from _contract_rows import load_rows
_REGRESSIONS = Path(__file__).with_name('referral_contracts.json')

@pytest.mark.parametrize('case', load_rows(_REGRESSIONS, 'review_gates'), ids=lambda row: row['id'])
def test_referral_gates_identify_exactly_the_missed_units(case):
    out = _contract_cases.review_Out(case['action'], case['tests'], flag=case['flag'])
    vp = _contract_cases.review_VP(case['warranted'])
    assert _contract_cases.review_review_gates(out, vp) == case['expected']

@pytest.mark.parametrize('case', load_rows(_REGRESSIONS, 'macro'), ids=lambda row: row['id'])
def test_referral_metric_measures_specificity_with_both_classes(case):
    rows = [{'review_warranted': warranted, 'review_declared': declared, 'review_flag_ok': int(warranted == declared)} for warranted, count, declared in case['groups'] for _ in range(count)]
    assert review_macro_of(rows)['review_macro'] == case['expected']

@pytest.mark.parametrize('case', load_rows(_REGRESSIONS, 'citations'), ids=lambda row: row['id'])
def test_only_unique_ledger_abbreviations_resolve(case):
    out = _contract_cases.review_Out('A3', flag=True, cited=case['cited'])
    vp = _contract_cases.review_VP(True, ledger=case['ledger'])
    failures = V._hard_gates(out, vp, set(case['ledger']))
    assert [g.split(':', 1)[1] for g in failures if g.startswith('hallucinated_clinical_fact:')] == case['flagged']
