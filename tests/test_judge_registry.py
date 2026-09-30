"""The judge registry -- contract layer, entirely synthetic fixtures, runs on
any machine.

Two halves:
* **Invariants**: `_CORE`'s names/kinds/when pinned one by one -- order is
  meaning, changing the order changes the readings;
* **Pluggability**: external judges can be mounted, a bad mount is rejected,
  and a mounted judge is **actually selected by `applicable()`** (verifying
  only "registration succeeded" is not enough -- that is exactly "a
  mechanism existing != being exercised").
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

# Same bootstrap as `test_payloads.py`: repo root + kernel (the L0 substrate
# is not part of this repo)
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import judges as J                        # noqa: E402
from haenv import mount_table as MT                  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    """Each test case runs against a clean registry and resets afterward --
    otherwise test cases would see each other's mounted judges."""
    J.reset_judges()
    _mount_before = {k: dict(v) for k, v in MT.MOUNT.items()}
    _why_before = dict(MT.WHY_NOT)
    yield
    J.reset_judges()
    MT.MOUNT.clear(); MT.MOUNT.update(_mount_before)
    MT.WHY_NOT.clear(); MT.WHY_NOT.update(_why_before)


# ---------------------------------------------------------------- invariants

#: **Snapshot of this repo's built-in judge order.** Changing it
#: must be **deliberate**, and requires unfreeze -> recompute -> republish
#: leaderboard. Compares three things: name, kinds, whether `when` is set --
#: comparing names alone would miss the family of "kinds is missing a gold
#: kind" bugs (e.g. a judge lacking `ddx:insufficient` would score an
#: entire wasted run as perfect).
CORE_SNAPSHOT = [
    ("forecast", ("forecast",), False),
    ("driver", ("forecast",), False),
    ("alternative", ("*",), False),
    ("dx_unified", ("ddx:unified",), False),
    ("dx_comorbidity", ("ddx:comorbidity",), False),
    ("dx_independent", ("ddx:independent",), False),
    ("join_type", ("ddx:unified", "ddx:comorbidity", "ddx:independent"), False),
    # `join_evidence` checks the ids behind `join_type` by code.
    ("join_evidence", ("ddx:unified", "ddx:comorbidity", "ddx:independent"), False),
    ("dx_rival", ("ddx:unified", "ddx:comorbidity", "ddx:independent"), True),
    ("disc_tool", ("ddx:unified", "ddx:comorbidity", "ddx:independent"), True),
    ("join_selfcheck", ("ddx:unified", "ddx:comorbidity", "ddx:independent"), False),
    ("join_cover", ("ddx:unified", "ddx:comorbidity", "ddx:independent"), False),
    ("workup", ("ddx:unified", "ddx:comorbidity", "ddx:independent"), True),
    ("abstention", ("ddx:unified", "ddx:comorbidity", "ddx:independent",
                    "ddx:insufficient"), False),
    ("slices", ("*",), False),
    ("slice_revision", ("*",), False),
    ("commit_timing", ("ddx:insufficient",), False),
    ("slices_abstention", ("*",), False),
    ("review_flag", ("*",), False),
    ("slices_gates", ("*",), False),
    ("slices_review", ("*",), False),
    ("slices_workup", ("ddx:unified", "ddx:comorbidity", "ddx:independent"), True),
    ("self_contradictory_exclusion", ("*",), False),
    ("slices_self_contradictory_exclusion", ("*",), False),
    ("what_not_to_do", ("*",), False),
    ("slices_wnd", ("*",), False),
    # Two entries specific to the multi-round geometry. They are in `_CORE`
    # and run through `run_judges`'s dispatch, not only mounted in
    # `mount_table.MOUNT` and called directly by `_row_multi` -- that would
    # declare the same thing in two places.
    # They sit at the **end**: `_CORE` order is meaning (`run_judges` does
    # `out.update` entry by entry), and inserting in the middle would change
    # the override order of same-named keys. Their keys do not collide with
    # the entries before them, and the single-shot / gated / slices paths
    # never mount them.
    ("multiround_revision", ("*",), False),
    ("premise_repair", ("*",), False),
]


def test_core_order_and_kinds_pinned():
    got = [(j.name, tuple(j.kinds), j.when is not None) for j in J._CORE]
    assert got == CORE_SNAPSHOT, (
        "自带判据的名字/kinds/when 变了。**顺序即语义**(`run_judges` 逐条 out.update),"
        "改它 = 改读数 ⇒ 必须连带解冻 → 重算 → 出榜,并更新本快照。")


def test_registry_starts_as_core():
    assert [j.name for j in J.JUDGES] == [j.name for j in J._CORE]
    assert J.EXTERNAL == {}


def test_every_core_judge_has_a_mount_cell_or_a_reason():
    """Not one built-in judge is allowed to be "registered but not mounted
    on any geometry"."""
    for j in J._CORE:
        mounted = [g for g in MT.GEOMETRIES if MT.subject_of(j.name, g) != MT.NONE]
        assert mounted, f"{j.name} 在所有几何上都不挂 —— 登记了从不跑"


# ---------------------------------------------------------------- pluggability

def _dummy(subject, vp, ctx=None) -> dict:
    return {"dummy_ok": 1}


def _mk(name="ext_probe", kinds=("*",)):
    return J.Judge(name, kinds, _dummy)


def test_register_appends_and_is_labelled():
    J.register_judge(_mk(), source="pkg:demo")
    assert J.JUDGES[-1].name == "ext_probe"
    assert J.EXTERNAL["ext_probe"] == "pkg:demo"
    m = J.registry_manifest()
    assert m["n_core"] == len(CORE_SNAPSHOT) and m["n_total"] == m["n_core"] + 1
    assert "ext_probe" not in m["core"], "外部判据不许混进 core 那一栏"


def test_register_after_puts_it_in_the_right_slot():
    J.register_judge(_mk(), after="driver")
    names = [j.name for j in J.JUDGES]
    assert names[names.index("driver") + 1] == "ext_probe"


def test_unknown_after_is_rejected_not_appended():
    """**Does not fall back to appending at the end** -- silently changing
    the position means silently changing the readings."""
    with pytest.raises(KeyError):
        J.register_judge(_mk(), after="__nosuch__")
    assert "ext_probe" not in [j.name for j in J.JUDGES]


def test_duplicate_name_rejected_by_default():
    J.register_judge(_mk())
    with pytest.raises(ValueError, match="already registered"):
        J.register_judge(_mk())
    assert len(J.JUDGES) == len(J._CORE) + 1


def test_duplicate_allowed_only_when_explicit():
    J.register_judge(_mk(), source="a")
    J.register_judge(_mk(), source="b", replace=True)
    assert J.EXTERNAL["ext_probe"] == "b"
    assert len(J.JUDGES) == len(J._CORE) + 1


@pytest.mark.parametrize("name", ["forecast", "review_flag", "slices_wnd"])
def test_core_judges_cannot_be_replaced_or_removed(name):
    """The judge fingerprint only covers the source code, not a runtime
    substitution -- allowing replacement would make the fingerprint lie."""
    with pytest.raises(ValueError, match="built into this repo"):
        J.register_judge(_mk(name=name), replace=True)
    with pytest.raises(ValueError, match="built into this repo"):
        J.unregister_judge(name)


def test_unregister_removes_external():
    J.register_judge(_mk())
    J.unregister_judge("ext_probe")
    assert "ext_probe" not in [j.name for j in J.JUDGES]
    assert "ext_probe" not in J.EXTERNAL


# ------------------------------------------------- usage surface: registered != runs

def test_registered_but_unmounted_judge_never_runs():
    """**Register only, no mount => `NONE` on every geometry =>
    never runs.**

    This test **documents a known trap**: missing
    either half looks, in the artifact, identical to "never registered".
    """
    J.register_judge(_mk())
    for g in MT.GEOMETRIES:
        assert MT.subject_of("ext_probe", g) == MT.NONE
    picked = [j.name for j in J.applicable("single", "forecast")]
    assert "ext_probe" not in picked


def test_mounted_judge_is_actually_selected():
    """Both halves done => `applicable()` actually selects it. **This is
    what "mounted" actually means.**"""
    J.register_judge(_mk())
    MT.mount("ext_probe", {"single": MT.OUT},
             why_not={g: "探针只验单拍" for g in ("gated", "slices", "multi")})
    assert "ext_probe" in [j.name for j in J.applicable("single", "forecast")]
    assert "ext_probe" not in [j.name for j in J.applicable("slices", "forecast")]


def test_mount_requires_reason_for_every_blank_geometry():
    """An empty cell must have a reason -- missing the reason for even one
    geometry raises, it does not silently leave a blank."""
    J.register_judge(_mk())
    with pytest.raises(ValueError, match="given a reason"):
        MT.mount("ext_probe", {"single": MT.OUT}, why_not={"gated": "只验单拍"})
    assert "ext_probe" not in MT.MOUNT


def test_mount_rejects_bad_subject_and_unknown_geometry():
    J.register_judge(_mk())
    with pytest.raises(ValueError, match="illegal subject"):
        MT.mount("ext_probe", {"single": "whatever"})
    with pytest.raises(KeyError, match="unknown geometry"):
        MT.mount("ext_probe", {"__nosuch__": MT.OUT})


def test_core_mounts_are_immutable_at_runtime():
    with pytest.raises(ValueError, match="built into this repo"):
        MT.mount("forecast", {"single": MT.OUT})


def test_kinds_filter_still_applies_to_external_judges():
    """External judges are still bound by the gold kind constraint --
    mounting does not bypass `kinds`."""
    J.register_judge(J.Judge("ext_ddx_only", ("ddx:unified",), _dummy))
    MT.mount("ext_ddx_only", {"single": MT.OUT},
             why_not={g: "探针只验单拍" for g in ("gated", "slices", "multi")})
    assert "ext_ddx_only" in [j.name for j in J.applicable("single", "ddx:unified")]
    assert "ext_ddx_only" not in [j.name for j in J.applicable("single", "forecast")]


def test_plugins_are_not_loaded_implicitly():
    """**Does not auto-scan** -- otherwise "which judges ran for this
    leaderboard" would depend on whatever packages happened to be
    installed."""
    assert J.EXTERNAL == {}, "import haenv.judges 不该带进任何外部判据"
