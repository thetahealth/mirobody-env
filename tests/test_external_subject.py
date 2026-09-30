"""Can an external observation subject be plugged in -- the acceptance test for the claim "the framework is flexible and pluggable."

## Why this file has to exist

Every step of the judge dispatch chain is an open registry -- the `JUDGES`
tuple through `register_judge`, the gold-kind rules through
`register_kind_rule` -- and so is the observation subject at its end:

    gold kind --> applicable(geometry, kind, vp) --> MOUNT[judge][geometry] --> subject
       ^open            ^open                            ^open              ^open

With a hard-coded `SUBJECTS` set, adding an observation subject (as `TRAJ`
was added) would mean editing `mount_table.py`, i.e. touching the core.

This file pins down the statement: an external package, outside this repo,
can bring its own new view plus a judge that consumes it, without changing a
single line of haenv. If that doesn't run, the claim is false.

    uv run pytest tests/test_external_subject.py -q

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
for _p in (str(ROOT), str((ROOT / _cfg["kernel_path"]).resolve()),
           str(ROOT / "examples" / "external_subject_demo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from haenv import judges as J, mount_table as MT              # noqa: E402


@pytest.fixture
def plugged():
    """Installs the plugin, then tears down cleanly afterward -- an incomplete teardown would contaminate other tests in the same process."""
    import haenv_subject_demo as D
    js = D.judges()
    for j in js:
        J.register_judge(j, source="test_external_subject")
    yield D
    for j in js:
        J.unregister_judge(j.name)
    MT.MOUNT.pop(D.NAME, None)
    MT.WHY_NOT.pop(D.NAME, None)
    MT.unregister_subject(D.SUBJECT)


# ═══════════ 1. claim: a new view can be added without touching the kernel ═══════════

def test_external_package_can_add_a_subject(plugged):
    """Note: the load-bearing test -- a package outside this repo registers a fifth subject type, and not a single line of `haenv/` changes."""
    assert plugged.SUBJECT in MT.SUBJECTS, "外挂 subject 没进 SUBJECTS ⇒ 主张为假"
    assert plugged.SUBJECT in MT.subject_specs()
    spec = MT.subject_specs()[plugged.SUBJECT]
    assert spec["source"] and spec["why"], "登记项必须自带「谁产出」与「为什么不够用」"


def test_core_subjects_are_still_closed(plugged):
    """Negative control: open does not mean unvalidated -- the four built-in subjects must not be overwritten, and an unregistered name still raises."""
    with pytest.raises(ValueError):
        MT.register_subject("out", lambda c: None, source="x", why="y")
    with pytest.raises(ValueError):
        MT.mount("__x__", {"single": "__nosuch_subject__"}, why_not={})


def test_unregistered_subject_still_raises_without_the_plugin():
    """The other half of the negative control: without the plugin installed, that name must still be invalid.

    Without this test, the previous one could have passed just because
    `SUBJECTS` accepts everything.

    This test guarantees its own clean state rather than relying on
    execution order -- relying on running before the fixture-based tests
    means that the moment the fixture's teardown breaks (construction
    raises, and the teardown half never runs), this test would be
    judging against already-contaminated state, and a red result from it
    would be a false red.
    """
    MT.unregister_subject("episode")
    assert "episode" not in MT.SUBJECTS
    with pytest.raises(ValueError):
        MT.mount("__y__", {"multi": "episode"}, why_not={})


# ═══════════ 2. claim: the judge genuinely runs end to end ═══════════

class _VP:
    adjudication = {"ddx": {"diagnosis": "X", "join_gold": "unified"}}
    gold_drivers = ["unknown_or_multifactorial"]


def test_external_judge_consumes_the_new_subject(plugged):
    """The judge consumes only `episode`, and `episode` is derived by the plugin from `traj` -- run end to end once."""
    traj = [{"round": 1, "risk_cat": "low", "day": 30},
            {"round": 2, "risk_cat": "elevated", "day": 60},
            {"round": 3, "risk_cat": "high", "day": 90}]
    out = J.run_judges("multi", {"traj": traj}, _VP())
    assert out.get("xdemo_n_stages") == 3, out
    assert out.get("xdemo_risk_monotonic") == 1.0, out
    # `"mount_subject_missing" not in out` is not the expectation:
    # 17 single-shot-family judges (subject=`LAST`) are mounted on the multi
    # column, and this call site does not pass `ctx["round_outputs"]`, so each
    # of them records `mount_subject_missing` and is skipped. Nothing is
    # substituted for the missing subject and nothing collapses to 0. The
    # test guards that the plugin itself ran: it must not appear in the
    # missing-subject list.
    miss = out.get("mount_subject_missing") or []
    assert not [m for m in miss if plugged.NAME in m], \
        f"外挂被记成 subject 缺席,却又产出了读数 ⇒ 两条路同时走了:{miss}"
    assert all(m.endswith(":last") for m in miss), \
        f"缺席的不全是 `last`(本调用点只缺末轮作答),其余缺席要单独查:{miss}"


def test_a_regression_is_actually_detected(plugged):
    """Negative control: a regression in risk level must be detected, otherwise the previous test is just "always 1.0."""
    traj = [{"round": 1, "risk_cat": "high"}, {"round": 2, "risk_cat": "low"},
            {"round": 3, "risk_cat": "low"}]
    out = J.run_judges("multi", {"traj": traj}, _VP())
    assert out.get("xdemo_risk_monotonic") == 0.5, out


def test_missing_source_view_is_skipped_not_faked(plugged):
    """Note: with no `traj`/`rows` underneath, `build` returns `None` => the judge is skipped and a line is recorded.

    Must not be substituted with another subject, and must not collapse
    to 0. Substitution would make a judge score something it was never
    meant to score, invisibly in the output -- `run_judges`'s own
    documentation states this rule, and this test pins it down.
    """
    out = J.run_judges("multi", {"out": object()}, _VP())
    assert "xdemo_n_stages" not in out, "没有源视图却产出了读数 ⇒ 在编"
    miss = out.get("mount_subject_missing") or []
    assert any(plugged.NAME in m for m in miss), f"跳过没有被记下来:{miss}"


# ═══════════ 3. the default path must not change ═══════════

def test_default_path_sees_exactly_four_subjects():
    """Without the plugin installed, `SUBJECTS` is exactly the four built-in ones -- an open registry must not widen the default path."""
    MT.unregister_subject("episode")          # same as above: guarantee clean state independently
    assert sorted(MT.SUBJECTS) == ["last", "out", "rows", "traj"]
    assert MT.subject_specs() == {}


def test_builder_failure_does_not_kill_the_row():
    """Negative control: when a plugin's `build` raises, the whole cell must not be taken down (this judge just has no object)."""
    MT.register_subject("boom", lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
                        source="test", why="验证 build 抛错不拖垮整格")
    try:
        out = J.run_judges("single", {"out": object()}, _VP())
        assert isinstance(out, dict) and "gold_kind" in out
    finally:
        MT.unregister_subject("boom")
