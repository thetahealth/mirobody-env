"""Judge mounting for the `multi` geometry -- acceptance checks for the cross-product table's `multi` column.

This file has a different division of labor from `test_multi_subject.py` --
don't merge them: that one asks "can the dispatcher accept a fifth subject
type (`TRAJ`)?"; this one asks "which judges are mounted, what is the
reason for each unmounted cell, and do the mounted ones actually run?"

Four categories of assertions, each paired with a counter-case that can go red:

1. Coverage -- `multi`'s mount count is at least half of `slices`'s (not
   "> 0 counts": a geometry with a single mounted judge would make "> 0"
   vacuously true forever).
2. Every unmounted cell must have a per-cell reason -- and the reasons must
   not all collapse to the same wildcard ("too narrow" and "simply doesn't
   apply" look identical in the table; this discipline is the only thing
   that tells them apart).
3. Mounted judges are actually invoked and actually produce values -- run
   through the production dispatch path `run_judges("multi", ...)` and
   assert the keys actually appear in the row; with a negative control:
   when `ctx` carries no per-round outputs, those judges must go through
   `mount_subject_missing`, never get substituted with `traj`.
4. Counter-metric -- the mount counts for the `single` / `gated` /
   `slices` columns are each pinned down. If changing the `multi` column
   moves any of the others, that's a bug, not a side effect.

All synthetic fixtures, zero model calls, zero cost.

## The last round's output needs no kernel change

Getting the last round's `SolverOutput` does not require changing the
kernel's `run_multiround` return signature: each round calls the
caller-supplied solver, and wrapping it in `mounting.RoundRecorder` is
enough to capture it. `test_last_subject_is_producible_without_touching_the_kernel`
pins this.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

# Same bootstrap as `test_multi_subject.py`: repo root + kernel (the L0 substrate)
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import judges as J                        # noqa: E402
from haenv import mount_table as MT                  # noqa: E402
from haenv import mount_coverage as MC                  # noqa: E402
from haenv import mounting as MG                     # noqa: E402

#: The slice-only judge list is hard-coded here, not derived from `MOUNT`.
#: Deriving it ("anything not mounted on multi is the slices family") would
#: make the assertion vacuously true: fixing a gap would shrink the list to
#: match, so "was anything missed" would always answer "no."
SLICES_FAMILY = ("slices", "slice_revision", "slices_abstention", "slices_gates",
                 "slices_review", "slices_workup",
                 "slices_self_contradictory_exclusion", "slices_wnd")

#: The single-shot family's subject on multi is the last round's answer. This list is likewise hard-coded.
SINGLE_FAMILY_ON_MULTI = (
    "forecast", "driver", "alternative", "workup", "abstention", "commit_timing",
    "review_flag", "self_contradictory_exclusion", "what_not_to_do", "dx_rival",
    "disc_tool", "join_selfcheck", "join_cover", "dx_unified", "dx_comorbidity",
    "dx_independent", "join_type")


@pytest.fixture(autouse=True)
def _clean():
    """Each test runs on a clean registry and a clean mount table, restored byte-for-byte afterward (deep-copies the inner dicts)."""
    J.reset_judges()
    _mount_before = {k: dict(v) for k, v in MT.MOUNT.items()}
    _why_before = dict(MT.WHY_NOT)
    yield
    J.reset_judges()
    MT.MOUNT.clear(); MT.MOUNT.update(_mount_before)
    MT.WHY_NOT.clear(); MT.WHY_NOT.update(_why_before)


# ---------------------------------------------------------------- fixtures

class _VP:
    """Minimal verifier payload: just enough for `gold_kinds` to infer `ddx:unified`."""

    def __init__(self):
        self.case_id = "T-MULTIMOUNT-01"
        self.outcome_label = "event_occurred"
        self.gold_drivers = ["unknown_or_multifactorial"]
        self.adjudication = {"ddx": {"spec_id": "pcos", "diagnosis": "多囊卵巢综合征",
                                     "aliases": ["pcos"], "join_gold": "unified",
                                     "urgency": "🟡", "tests": ["OGTT+胰岛素"],
                                     "specialty": ["内分泌科"]},
                             # Note: the review gold standard lives at the top level of
                             # `adjudication`, not inside `ddx` (`judge_review_flag` reads
                             # `adj["clinician_action_warranted"]`) -- putting it at the wrong
                             # level makes the judge return `{}`, indistinguishable from
                             # "this cell doesn't apply."
                             "clinician_action_warranted": True}


class _Out:
    """A last-round answer: shaped like `schema.SolverOutput` plus haenv's own `_raw`."""

    def __init__(self):
        self.forecast = {"risk": 0.7, "risk_category": "high"}
        self.drivers = [{"rank": 1, "driver": "unknown_or_multifactorial",
                         "evidence_for": ["EV-1"]},
                        {"rank": 2, "driver": "adherence_gap", "evidence_for": ["EV-2"]}]
        self.action = {"selected_action_class": "A3", "specific_action": "转诊内分泌",
                       "clinician_review_required": True, "what_not_to_do": ["不自主改药"]}
        self.data_quality = {"data_sufficiency": "sufficient"}
        self.cited_evidence = ["EV-1", "EV-2"]
        self.notes = "候选一:多囊卵巢综合征"
        self._raw = {"differential": [{"diagnosis": "多囊卵巢综合征", "rank": 1,
                                       "supporting_evidence": ["EV-1"]}],
                     "join_type": "unified",
                     "tests_to_order": ["OGTT+胰岛素"],
                     "referral_specialty": ["内分泌科"],
                     "what_not_to_do": ["不自主改药"]}


def _traj(n: int = 3) -> list[dict]:
    """A minimal trajectory: field names copied verbatim from `mount_table.TRAJ_ROW_FIELDS`."""
    return [{"round": i + 1, "day": 100 + 7 * i, "risk_cat": "high", "risk": 0.7,
             "top_driver": "unknown_or_multifactorial", "action": "A3",
             "repair": "initial" if i == 0 else "confirmed",
             "visible_ev": ["EV-1"], "cited_ev": ["EV-1"], "n_new_points": 1,
             "notes": None} for i in range(n)]


def _mounted(geometry: str) -> list[str]:
    """The names of judges mounted on this column -- there is exactly one way to compute this number, and it's the same source `coverage` uses."""
    return [n for n in MT.MOUNT if MT.subject_of(n, geometry) != MT.NONE]


# ---------------------------------------------------------------- 1. coverage

def test_multi_column_is_at_least_half_of_slices():
    """`multi`'s mount count must be at least half of `slices`'s.

    Why not "> 0": that would be a vacuously true assertion -- mounting a
    single judge satisfies it forever. `slices` is the reference because it's the
    most fully mounted column, and the multi-round geometry tests nothing
    less on the prompt side than slicing does (both are N answers in the
    same world; the difference is whether history carries over).
    """
    n_multi, n_slices = len(_mounted("multi")), len(_mounted("slices"))
    assert n_slices > 0, "参照系自己是 0 ⇒ 上面那条恒真,这个读数没扫到东西"
    assert n_multi >= n_slices / 2, (
        f"multi 只挂了 {n_multi} 条,不到 slices({n_slices})的一半 —— "
        f"「四种执行几何」写得出来、数据里只有三种")


def test_single_family_mounts_last_on_multi():
    """The 17 single-shot judges' cell value on multi must be `LAST` (last round's answer), not `OUT`, not `TRAJ`.

    A wrong cell value fails silently: `run_judges` takes the
    `want not in subjects` branch, and the row is left with only a
    `mount_subject_missing` entry -- indistinguishable from "this cell
    shouldn't be mounted at all."
    """
    for name in SINGLE_FAMILY_ON_MULTI:
        assert MT.subject_of(name, "multi") == MT.LAST, (
            f"{name}@multi 的格值是 {MT.subject_of(name, 'multi')!r},应为 LAST")


def test_counter_metric_other_geometries_untouched():
    """Counter-metric: changing the `multi` column must not move the other three.

    The numbers are pinned rather than compared against a baseline file --
    a baseline file changes along with the code, so it would always agree.
    The denominator is `MOUNT`'s keys; `join_evidence` is mounted on all
    four columns like `join_type` (`single 18, gated 18, slices 17`).
    """
    assert len(_mounted("single")) == 18
    assert len(_mounted("gated")) == 18
    assert len(_mounted("slices")) == 17


# ---------------------------------------------------------------- 2. reasons for unmounted cells

def test_every_unmounted_cell_on_multi_has_a_reason():
    """Every unmounted cell on multi must have a reason -- not a single one may be left blank."""
    for name in MT.MOUNT:
        if MT.subject_of(name, "multi") != MT.NONE:
            continue
        assert MT.reason_for(name, "multi"), f"{name}@multi 是个没理由的空格"
    assert MC.holes() == [], f"叉乘里出现了没理由的洞:{MC.holes()}"


def test_slices_family_reasons_are_per_judge_not_one_wildcard():
    """The reasons for the 8 slices-family judges must be per-judge, not covered by one family-wide wildcard.

    A family-wide wildcard would reduce 8 cells -- each missing a different
    field -- to a single sentence. "This one is
    missing `data_sufficiency`" and "that one is missing the whole
    differential list" would look identical in the report, and distinguishing
    exactly that is the reason this table exists.
    """
    reasons = {}
    for name in SLICES_FAMILY:
        assert MT.subject_of(name, "multi") == MT.NONE, \
            f"{name} 居然挂上了 multi —— 那要改本用例,并给出它读到的轨迹字段"
        r = MT.reason_for(name, "multi")
        assert r, f"{name}@multi 是个没理由的空格"
        reasons[name] = r
    assert len(set(reasons.values())) == len(SLICES_FAMILY), (
        f"8 条切片族只写出了 {len(set(reasons.values()))} 条不同的理由 —— "
        f"重复的那几条等于「同上」,而「同上」不算理由")
    # Every reason must name the multi-round-side equivalent: saying only "not applicable" just pushes the question onto whoever reads it next
    for name, r in reasons.items():
        assert ("多轮" in r or "@multi" in r or "multiround" in r), \
            f"{name}@multi 的理由没说多轮侧由谁承担:{r[:60]}"


def test_reason_lookup_is_exact_before_wildcard():
    """An exact key must take priority over a wildcard -- order is semantics here.

    Getting this backwards would let the 8 per-judge reasons be
    overwritten by the single `slices*@multi` wildcard.
    """
    exact = MT.WHY_NOT["slices_workup@multi"]
    family = MT.WHY_NOT["slices*@multi"]
    assert exact != family
    assert MT.reason_for("slices_workup", "multi") == exact


# ---- counter-case: removing one reason must immediately fail ----

def test_negative_control_blank_reason_is_detected():
    """Counter-case -- clear one `why_not` entry, and the two assertions above must flip to failing.

    Without this test, `test_every_unmounted_cell_on_multi_has_a_reason`
    could be vacuously true: a guard that can never go red is equivalent to
    no guard at all.
    """
    key = "slices_workup@multi"
    saved = MT.WHY_NOT.pop(key)                     # remove the exact key
    saved_family = MT.WHY_NOT.pop("slices*@multi")  # remove the family wildcard too, or it would catch it
    saved_star = MT.WHY_NOT.pop("*@multi")          # remove the global fallback too
    try:
        assert MT.reason_for("slices_workup", "multi") is None, \
            "三条理由都抽掉了还查得到 —— 说明 `reason_for` 从别处兜了底,断言恒绿"
        assert ("slices_workup", "multi") in MC.holes(), \
            "没理由的空格没有进 `holes()` —— 那条棘轮盯的东西是空的"
    finally:
        MT.WHY_NOT[key] = saved
        MT.WHY_NOT["slices*@multi"] = saved_family
        MT.WHY_NOT["*@multi"] = saved_star
    assert MC.holes() == [], "复位之后洞必须消失(证明上面那次判负是抽理由造成的)"


# ---------------------------------------------------------------- 3. actually runs

def test_mounted_multi_judges_actually_run_and_produce_values():
    """Judges mounted here actually produce keys through the production dispatch path `run_judges("multi", ...)`.

    Asserting `subject_of(...) == LAST` alone is not enough: that only
    proves a string was written into the table.
    """
    row = J.run_judges("multi", {"traj": _traj()}, _VP(),
                       ctx={"round_outputs": [_Out(), _Out()]})
    assert not row.get("mount_subject_missing"), \
        f"有判据没拿到 subject:{row.get('mount_subject_missing')}"
    # Spot-check across families: diagnosis naming / test ordering / review flag / prohibitions -- keys from four different judges
    for key in ("dx_hit", "tests_recall", "review_flag_ok", "wnd_n_items", "a1"):
        assert key in row, f"`{key}` 没出现在行里 ⇒ 那条判据没跑"
    assert row["dx_hit"] is True                    # the last round's answer is exactly the gold diagnosis
    assert row["review_flag_ok"] == 1


def test_negative_control_no_round_outputs_means_missing_not_substituted():
    """Negative control -- when `ctx` carries no per-round outputs, the 17 judges must go through `mount_subject_missing`, never get substituted with `traj`.

    Substitution would make a judge score something it was never meant to
    score: a "last-round view" derived from the trajectory rows produces
    fabricated values for most of these judges (`driver`'s `all_drivers`
    truncated to 1 item, `workup`'s `tests_recall` stuck at 0).
    """
    row = J.run_judges("multi", {"traj": _traj()}, _VP(), ctx={})
    miss = row.get("mount_subject_missing") or []
    assert miss, "没有 round_outputs 却一条 `mount_subject_missing` 都没有 ⇒ 有人在顶替"
    assert all(m.endswith(":last") for m in miss), miss
    # none of the substituted judge keys may appear
    for key in ("dx_hit", "tests_recall", "review_flag_ok"):
        assert key not in row, f"`{key}` 出现了 —— subject 缺席却算出了值"


def test_last_subject_is_producible_without_touching_the_kernel():
    """Getting the last round's `SolverOutput` does not require changing the kernel's signature.

    `runner.run_multiround` calls the caller-supplied `solver.solve(payload)`
    each round, so the caller can just wrap it. This test pins that
    structure down with a fake runner -- it does not run the real kernel
    (which needs a full RawCase), but the call shape is identical byte for
    byte.
    """
    class _Inner:
        def __init__(self):
            self.prompt_mode = "default"
            self.n = 0

        def solve(self, payload):
            self.n += 1
            return _Out()

    inner = _Inner()
    rec = MG.RoundRecorder(inner)
    rec.prompt_mode = "ddx"                          # the caller sets this after construction
    assert inner.prompt_mode == "ddx", "`prompt_mode` 没透传 ⇒ 多轮会整批退回默认题面"

    def _fake_run_multiround(solver, rounds=4):
        """The shape of that part of the kernel: `solver.solve(payload)` per round, with no `prior` in the return value."""
        for _ in range(rounds):
            solver.solve({"case_id": "T"})
        return _traj(rounds), None, None

    traj, _, _ = _fake_run_multiround(rec)
    assert len(rec.outputs) == 4 == inner.n, "记录器漏了轮次"
    assert rec.outputs[-1] is not None
    # use it as the subject, and the 17 judges immediately have an object
    row = J.run_judges("multi", {"traj": traj}, _VP(),
                       ctx={"round_outputs": rec.outputs})
    assert not row.get("mount_subject_missing")
    assert "dx_hit" in row


def test_other_geometries_do_not_get_the_multi_derivation():
    """`round_outputs` is only consumed on multi -- every other geometry is unchanged, byte for byte.

    Without this test, "only takes effect on multi" is just a claim in a
    comment.
    """
    row = J.run_judges("single", {"out": _Out()}, _VP(),
                       ctx={"round_outputs": [_Out()]})
    assert "last" not in row and "dx_hit" in row      # single goes through `out`, unchanged
    # on slices, `last` comes from the caller; if not given it should be absent, never backfilled from round_outputs
    row2 = J.run_judges("slices", {"rows": []}, _VP(), ctx={"round_outputs": [_Out()]})
    miss = row2.get("mount_subject_missing") or []
    assert any(m.endswith(":last") for m in miss), (
        f"slices 上 `last` 缺席却没记 —— 说明 multi 那段derivation 漏到别的几何了:{miss}")


# ---------------------------------------------------------------- 4. mounted != run

def test_unwired_geometry_reading_still_bites():
    """The "mounted != run" reading must still be able to bite -- especially now that all four geometries are wired in.

    All four geometries are wired in (`_row_multi` calls `run_judges`), so
    the list is empty.

    An empty list is the easiest reading to turn vacuous: a reading that
    always returns `[]` looks identical in the output to "everything is wired
    in." So this test also strips one geometry's `profile` and confirms it is
    immediately counted back in.
    """
    import dataclasses
    assert MC.unwired_geometries() == [], \
        f"还有几何挂了判据却不调 run_judges:{MC.unwired_geometries()}"
    assert MC.coverage()["unwired"] == MC.unwired_geometries(), \
        "覆盖读数与取值入口必须是同一条路径"
    # negative control: strip single's profile, it must be counted back in => proves the `[]` above isn't vacuous
    before = MG.BY_GEOMETRY["single"]
    MG.BY_GEOMETRY["single"] = dataclasses.replace(before, profile="")
    try:
        assert "single" in MC.unwired_geometries(), \
            "profile 清空后没被数回来 ⇒ 这个读数恒空,`[]` 证明不了任何事"
    finally:
        MG.BY_GEOMETRY["single"] = before
    assert MC.unwired_geometries() == [], "复位失败,后面的用例会读到脏状态"


def test_negative_control_unwiring_the_profile_restores_the_flag():
    """Counter-case -- multi is wired in, so `unwired_geometries()` must not report it; clearing `MOUNTS['multi'].profile` back to `""` must bring the flag right back.

    Without this, "this reading genuinely reads `profile`" would be
    unverified.

    The reading is neither vacuously true nor vacuously false: it reads
    `profile`.
    """
    import dataclasses
    before = MG.BY_GEOMETRY["multi"]
    assert "multi" not in MC.unwired_geometries(), \
        "multi 已接线(profile=MULTI + _row_multi 走 run_judges),不该还被报成未接线"
    MG.BY_GEOMETRY["multi"] = dataclasses.replace(before, profile="")
    try:
        assert "multi" in MC.unwired_geometries(), \
            "profile 清空后标记没回来 ⇒ 这个读数根本没读 profile,恒假"
    finally:
        MG.BY_GEOMETRY["multi"] = before
    assert "multi" not in MC.unwired_geometries(), "复位失败,后面的用例会读到脏状态"


def test_premounted_pair_is_now_registered():
    """The two multi-round-only judges are in `JUDGES` -- the pre-mounted-but-unregistered count is zero.

    Both entries are in `_CORE`, and `CORE_SNAPSHOT` matches.

    The counter-case: remove one of them and `unregistered_mounts()` must
    report it again, otherwise "pre-mounted
    count is zero" could just mean that reading itself is broken, not that
    the count is genuinely zero.
    """
    assert MC.unregistered_mounts() == [], \
        f"还有预挂载没注册:{MC.unregistered_mounts()}"
    for name in ("multiround_revision", "premise_repair"):
        assert MT.subject_of(name, "multi") == MT.TRAJ
        assert any(j.name == name for j in J.JUDGES), f"{name} 不在活注册表里"
    # counter-case: remove one, it must be counted as pre-mounted again
    _saved = list(J.JUDGES)
    J.JUDGES[:] = [j for j in J.JUDGES if j.name != "premise_repair"]
    try:
        assert "premise_repair" in MC.unregistered_mounts(), \
            "摘掉注册项后没被报成预挂载 ⇒ 这个读数恒空"
    finally:
        J.JUDGES[:] = _saved
    assert MC.unregistered_mounts() == [], "复位失败,后面的用例会读到脏状态"
