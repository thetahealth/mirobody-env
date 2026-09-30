"""Insufficient-information tier: a red flag whose symptoms all fall after `T` is not a red
flag at `T`. The generator rule (`ddx.insufficient_triage`) and the shipped core packs."""
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from haenv.ddx import INSUFFICIENT_HIDDEN_RED_FLAG_URGENCY, insufficient_triage   # noqa: E402

RED = {"red_flag": True, "urgency": "🔴"}


def test_hidden_red_flag_is_dropped():
    assert insufficient_triage(RED, [129, 158], 112, True) == (False, INSUFFICIENT_HIDDEN_RED_FLAG_URGENCY)


def test_symptom_at_or_before_t_on_the_tier_is_refused():
    import pytest
    for days in ([100, 158], [112, 158]):
        with pytest.raises(ValueError):
            insufficient_triage(RED, days, 112, True)


def test_every_insufficient_case_has_all_symptoms_after_t():
    """The premise both `ddx.insufficient_triage` and `wq.derive_from_registry` rely on."""
    bad = []
    for name in ("ddx-timeline", "ddx-workup"):
        doc = yaml.safe_load((ROOT / "inputs" / f"{name}.job.yaml").read_text(encoding="utf-8"))
        for c in doc["cases"]:
            lat = c["latent"]
            if lat.get("ddx_insufficient"):
                days = [int(s["day"]) for s in c["raw"].get("symptoms") or []]
                if not days or any(d <= int(lat["index_time_T"]) for d in days):
                    bad.append((name, c["case_id"], days))
    assert not bad, bad


def test_sufficient_tier_and_non_red_specs_unchanged():
    assert insufficient_triage(RED, [129, 158], 112, False) == (True, "🔴")
    assert insufficient_triage({"red_flag": False, "urgency": "🟠"}, [129], 112, True) == (False, "🟠")


def test_shipped_core_packs_have_no_hidden_red_flag():
    bad = []
    for name in ("ddx-timeline", "ddx-workup"):
        doc = yaml.safe_load((ROOT / "inputs" / f"{name}.job.yaml").read_text(encoding="utf-8"))
        for c in doc["cases"]:
            lat = c["latent"]
            days = [int(s["day"]) for s in c["raw"].get("symptoms") or []]
            if (lat.get("ddx_insufficient") and lat.get("ddx_red_flag")
                    and days and all(d > int(lat["index_time_T"]) for d in days)):
                bad.append((name, c["case_id"]))
            assert bool(lat.get("ddx_red_flag")) == (lat.get("ddx_urgency") == "🔴"), (name, c["case_id"])
    assert not bad, bad


def _built(case_id):
    import contextlib
    import io

    from haenv import build as B
    from haenv import job as J
    job = J.load_job(ROOT / "inputs" / "ddx-workup.job.yaml")
    cs = next(c for c in job.cases if c.case_id == case_id)
    with contextlib.redirect_stdout(io.StringIO()):
        raw, _ = B.build_case(cs, T=cs.index_time_T)
    assert raw is not None
    return raw


def test_gold_self_consistency_follows_the_tier():
    from copy import deepcopy

    from haenv import wq
    raw = _built("JD-09v2")                # insufficient tier, red-flag condition
    assert raw.adjudication["red_flag_present"] is False
    assert wq.check_gold_matches_world(raw) == []
    stale = deepcopy(raw)
    stale.adjudication["red_flag_present"] = True
    assert any("red_flag_present" in m for m in wq.check_gold_matches_world(stale))
    raw = _built("JD-09")                  # sufficient tier: the red flag stays
    assert raw.adjudication["red_flag_present"] is True
    flipped = deepcopy(raw)
    flipped.adjudication["red_flag_present"] = False
    assert any("red_flag_present" in m for m in wq.check_gold_matches_world(flipped))
