"""Sourced initial diagnostic stage shared by M2 generation and workup scoring.

The full clinical pathway remains in the condition registry. This module reads
stage and alternative declarations; it renders no patient evidence.
"""
from __future__ import annotations

from haenv.resources.regpath import registry_cached


@registry_cached("m2_pre_review.yaml")
def pre_review_rules() -> dict:
    from haenv.resources.regpath import load_registry
    return load_registry("m2_pre_review.yaml")


def workup_tests(spec_id: str, optional=()) -> list[str]:
    """Unique mandatory tests in the initial stage, using exact menu names."""
    from haenv.world.overlay import condition_registry
    from haenv.world.overlay import HAENV_COMORBID_PAIRS
    from haenv.judges.differential import _gold_test_optional
    from haenv.judges.differential import _gold_test_segs
    reg, rules = condition_registry(), pre_review_rules()
    comps = (HAENV_COMORBID_PAIRS.get(spec_id) or {}).get("pair") or [spec_id]
    tests = []
    for comp in comps:
        excluded = (rules["post_diagnosis"].get(comp) or {}).get("tests") or []
        for t in reg[comp].get("tests") or []:
            if t in excluded:
                continue
            names = (rules["menu_names"].get(t) or {}).get("tests")
            # The synchronous insulinoma sample is mandatory; only the fasting
            # protocol in the old compound name is conditional.
            for name in names if names is not None else [t]:
                if _gold_test_segs(name) and not _gold_test_optional(name) and name not in tests:
                    tests.append(name)
    for alt in rules["alternatives"]:
        present = [t for t in alt["tests"] if t in tests]
        if len(present) > 1:
            # Registered order is the initial-stage preference. Current members
            # have equal menu cost; the pool gate also checks minimum-cost choice.
            tests = [t for t in tests if t not in present or t == present[0]]
    # The acquisition name includes "必要时MRE"; that qualifier applies to
    # MRE, while secondary fibrosis assessment itself is mandatory.
    if "HD-UNI-13" in comps:
        test = rules["fibrosis_evidence"]["test"]
        if test not in tests:
            tests.append(test)
    return [t for t in tests if t not in optional]


def workup_alternatives(test: str) -> list[str]:
    return [t for a in pre_review_rules()["alternatives"] if test in a["tests"] for t in a["tests"]]



def _workup_applies(w, field):
    from haenv.world.wq import resolve_path
    return field == "tests" and isinstance(resolve_path(w, "W.adjudication.m2_workup"), dict)


def _derive_workup(w, field, value, deriv):
    from haenv.world.wq import resolve_path
    tests = [] if resolve_path(w, "W.adjudication.ddx.insufficient") is True else workup_tests(
        str(resolve_path(w, "W.adjudication.ddx.spec_id")),
        resolve_path(w, "W.adjudication.m2_workup").get("optional") or ())
    return tests, deriv + ["registry/m2_pre_review.yaml: initial-stage required tests and alternatives"]


def _critical_applies(w, field):
    from haenv.world.wq import resolve_path
    return field == "red_flag_present" and isinstance(
        resolve_path(w, "W.latent_premise.meta._external.m2.critical"), dict)


def _derive_critical(w, field, value, deriv):
    # An observed critical value is a red flag of its own, retaining the historical
    # comparison and malformed-input behavior. The registry red flag takes precedence.
    from haenv.world.wq import resolve_path
    if not value:
        crit = resolve_path(w, "W.latent_premise.meta._external.m2.critical")
        try:
            if isinstance(crit, dict) and float(crit["today"]) < float(crit["limit"]):
                return True, deriv + ["W.latent_premise.meta._external.m2.critical",
                                      "rule=critical value on day T (today < limit) -> True"]
        except (KeyError, TypeError, ValueError):
            pass
    return value, deriv


def gold_derivations():
    """Declared policies shared by plugin registration and saved-payload loading."""
    from haenv.contracts import GoldDerivation
    return (GoldDerivation("m2.initial_workup", _workup_applies, _derive_workup),
            GoldDerivation("m2.critical_reading", _critical_applies, _derive_critical))
