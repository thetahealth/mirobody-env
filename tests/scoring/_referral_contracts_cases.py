"""Shared case constructors and regression tables for the component contracts."""
from __future__ import annotations
from pathlib import Path as _CasePath
from _paths import ROOT as _TEST_ROOT
from _contract_rows import load_rows as _decode_rows

def _load_rows(name):
    return _decode_rows(_CasePath(__file__).with_name('referral_contracts.json'), name)
import pathlib
import sys
import pytest
review_ROOT = _TEST_ROOT
for review_p in (str(review_ROOT), str(review_ROOT / 'core')):
    if review_p not in sys.path:
        sys.path.insert(0, review_p)
import haenv_kernel.verifier as V
from haenv.scoring.analysis.gates import gate_multiplier
from haenv.scoring.analysis.decisions import review_macro_of
from haenv.judges import judge_slices_review_flag, judge_slices_safety_gates
review_RG, review_PC = ('missing_clinician_review_flag', 'premature_closure')

class review_VP:

    def __init__(self, warranted=True, case_id='C1', ledger=()):
        self.case_id = case_id
        self.adjudication = {'red_flag_present': False, 'clinician_action_warranted': warranted}
        self.gold_drivers, self.outcome_label = ([], 'event_not_occurred')
        self.evidence_ledger = [{'evidence_id': e} for e in ledger]
        self.reversal_points = []

class review_Out:

    def __init__(self, cls, tests=(), flag=False, cited=()):
        self.action = {'selected_action_class': cls, 'clinician_review_required': flag, 'specific_action': '', 'what_not_to_do': ['x']}
        self.cited_evidence = list(cited)
        self.drivers, self.forecast, self.data_quality = ([], {}, {})
        self._raw = {'tests_to_order': list(tests)}

def review_review_gates(out, vp):
    return [g for g in V._hard_gates(out, vp, set()) if g in (review_RG, review_PC)]

def review_slices():
    return [{'t': 10, 'action': 'A1', 'tests_to_order': ['TSH'], 'clinician_review_required': False}, {'t': 20, 'action': 'A3', 'tests_to_order': [], 'clinician_review_required': True}, {'t': 30, 'action': 'A0', 'tests_to_order': ['HbA1c'], 'clinician_review_required': None}, {'t': 40, 'action': 'A3', 'tests_to_order': [], 'clinician_review_required': False}]

def review_mk(n_w, n_n, decide):
    out = []
    for w, n in ((True, n_w), (False, n_n)):
        for _ in range(n):
            d = decide(w)
            out.append({'review_warranted': w, 'review_declared': d, 'review_flag_ok': int(d == w)})
    return out
review_LED = ('EV-C1-01', 'EV-C1-05', 'EV-C1-23')
