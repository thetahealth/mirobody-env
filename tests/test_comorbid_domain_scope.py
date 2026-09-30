"""Comorbidity signal domains must be scoped per thread.

Cases are built on a thread pool (`haenv build`, `gen_workers`). A comorbidity
domain merged into the kernel's shared table and removed when the case finishes
breaks under the pool: one thread's removal strips signals another thread's case
still needs (GV-1 `signal_not_in_disease_domain`) and one thread's additions leak
into unrelated cases, so the same case built alone and in a batch differs.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import haenv                                                        # noqa: E402

sys.path.insert(0, str(haenv.kernel_path()))

from latent import DISEASE_SIGNAL_DOMAIN                            # noqa: E402

from haenv.build import _register_comorbid_domain                   # noqa: E402


def test_scope_is_invisible_to_other_threads_and_leaves_no_trace():
    before = dict(DISEASE_SIGNAL_DOMAIN.get("dyslipidemia") or {})
    inside_a = threading.Event()
    b_done = threading.Event()
    seen: dict[str, bool] = {}

    def a():
        with _register_comorbid_domain("dyslipidemia", ["MASLD"]):
            seen["a_has_alt"] = "ALT" in DISEASE_SIGNAL_DOMAIN["dyslipidemia"]
            inside_a.set()
            b_done.wait(5)
            # B entered and left its own scope meanwhile; A's additions survive.
            seen["a_still_has_alt"] = "ALT" in DISEASE_SIGNAL_DOMAIN["dyslipidemia"]

    def b():
        inside_a.wait(5)
        seen["b_sees_alt"] = "ALT" in (DISEASE_SIGNAL_DOMAIN.get("dyslipidemia") or {})
        with _register_comorbid_domain("dyslipidemia", ["hypertension"]):
            seen["b_has_bp"] = "systolic_bp" in DISEASE_SIGNAL_DOMAIN["dyslipidemia"]
        b_done.set()

    ta, tb = threading.Thread(target=a), threading.Thread(target=b)
    ta.start(); tb.start(); ta.join(); tb.join()

    assert seen["a_has_alt"] and seen["a_still_has_alt"], seen
    assert not seen["b_sees_alt"], "a comorbidity domain leaked into another thread's case"
    assert seen["b_has_bp"], seen
    assert dict(DISEASE_SIGNAL_DOMAIN.get("dyslipidemia") or {}) == before


def test_base_domain_wins_over_scoped_additions():
    base = dict(DISEASE_SIGNAL_DOMAIN["obesity"])
    key = next(iter(base))
    with DISEASE_SIGNAL_DOMAIN.scoped("obesity", {key: {"unit": "overridden"}}):
        assert DISEASE_SIGNAL_DOMAIN["obesity"][key] == base[key]
