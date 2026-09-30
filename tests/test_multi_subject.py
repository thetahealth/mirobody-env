"""The fifth subject of the multi geometry (`TRAJ`) -- contract layer, all
synthetic fixtures, zero model calls.

This file answers three questions, none of which may rest on "it looks
right":

1. **Can `TRAJ` actually be consumed by the dispatcher?** -- mount a judge
   that only reads the trajectory, go through the **production path**
   `run_judges("multi", {"traj": ...})`, and assert its keys actually appear
   in the row. (Asserting only `subject_of(...) == TRAJ` is not enough: that
   only proves a string was written in the table.)
2. **Does breaking it turn it red?** -- make dispatch constantly return
   `NONE` (a mutant), and the same assertion must flip to a failure. A test
   that reads artifacts already generated on disk is immune to code changes;
   this one is a **regression gate**, not a snapshot.
3. **Is the table's own default-deny actually exercised?** -- an illegal
   cell value must raise immediately, not silently turn into "this cell
   isn't mounted".

This file **does not change** any frozen-section file; the coverage of
the `multi` column is asserted by
`test_multi_column_coverage_is_reported_honestly`.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

# Same bootstrap as `test_judge_registry.py`: repo root + kernel (the L0
# substrate is not part of this repo)
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import judges as J                        # noqa: E402
from haenv import mount_table as MT                  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    """Each test case runs against a clean registry + a clean mount table,
    reset field-for-field afterward.

    Resetting `MOUNT` must **deep-copy the inner dicts**: with a
    shallow copy, a cell value one test case stuffs into `MOUNT["x"]` would
    survive into the next test case -- exactly "test cases seeing each
    other".
    """
    J.reset_judges()
    _mount_before = {k: dict(v) for k, v in MT.MOUNT.items()}
    _why_before = dict(MT.WHY_NOT)
    yield
    J.reset_judges()
    MT.MOUNT.clear(); MT.MOUNT.update(_mount_before)
    MT.WHY_NOT.clear(); MT.WHY_NOT.update(_why_before)


# ---------------------------------------------------------------- fixtures

class _VP:
    """Minimal verifier payload: just enough for `gold_kinds` to infer
    `ddx:unified`."""

    def __init__(self):
        self.case_id = "T-MULTI-01"
        self.outcome_label = "event_occurred"
        self.adjudication = {"ddx": {"spec_id": "pcos", "diagnosis": "多囊卵巢综合征",
                                     "aliases": ["pcos"], "join_gold": "unified"}}


def _traj(n: int = 3) -> list[dict]:
    """A minimal trajectory: field names **copied verbatim** from
    `mount_table.TRAJ_ROW_FIELDS`.

    Field names are not invented: the one true definition of this shape lives
    in the kernel's `runner.run_multiround`, and this repo records it in
    `TRAJ_ROW_FIELDS`.
    """
    return [{"round": i + 1, "day": 100 + 7 * i, "risk_cat": "moderate", "risk": 0.4,
             "top_driver": "weight", "action": "A1",
             "repair": "initial" if i == 0 else ("revised" if i == 1 else "confirmed"),
             "visible_ev": [f"EV{k}" for k in range(i + 1)],
             "cited_ev": [f"EV{i}"], "n_new_points": 12 * i,
             "notes": "第 %d 轮" % (i + 1)}
            for i in range(n)]


#: An external judge that **only reads the trajectory** -- standing in for
#: an external process-judge family: a pure function that only consumes an
#: episode-shaped object and never touches the gold label.
def _judge_traj_probe(traj, vp, ctx=None) -> dict:
    if not isinstance(traj, list) or not traj:
        return {"trajprobe_note": "空轨迹 -> 不适用(不是 0)"}
    return {"trajprobe_n_rounds": len(traj),
            "trajprobe_n_revised": sum(1 for r in traj if r.get("repair") == "revised")}


_PROBE = "ext_traj_probe"


def _mount_probe() -> None:
    """Do both halves: register it, and mount the `TRAJ` cell on multi."""
    J.register_judge(J.Judge(_PROBE, ("*",), _judge_traj_probe), source="test:multi-subject")
    MT.mount(_PROBE, {"multi": MT.TRAJ},
             why_not={g: "探针只验多轮轨迹" for g in ("single", "gated", "slices")})


# ============================================================ 0. positive control on scan surface
def test_surface_is_not_empty():
    """**Positive control.** Every "found / not found" below rests
    on these three things; if they were empty, the whole file would degrade
    into a string of vacuously-true `not in` checks. Carries a **floor**, not
    just `is not None`.
    """
    assert len(J._CORE) >= 20, "自带判据表被读空了 —— 引导没接上,后面的对比全无意义"
    assert "multi" in MT.GEOMETRIES, "multi 不在几何清单里,本文件整个失去对象"
    assert len(MT.MOUNT) >= 20, "挂载表被读空了"
    assert len(_traj()) == 3 and all(set(MT.TRAJ_ROW_FIELDS) <= set(r) for r in _traj()), \
        "夹具轨迹的字段与 `TRAJ_ROW_FIELDS` 对不上 —— 形状漂了,下面的判据在读一个假东西"


# ============================================================ 1. the fifth subject exists and is legal
def test_traj_is_a_registered_subject():
    assert MT.TRAJ == "traj"
    assert MT.TRAJ in MT.SUBJECTS
    assert MT.NONE not in MT.SUBJECTS, "`NONE` 是「缺席」的记号,不是 subject,不许当格值用"
    assert MT.SUBJECTS == {MT.OUT, MT.ROWS, MT.LAST, MT.TRAJ}, \
        "格值域变了 —— 加第六种 subject 是**加一个语义**,必须是有意的并连带改设计文档"


def test_table_lints_clean():
    """A full pass of the whole table through default-deny: clean. This is
    `_lint_table`'s **positive control**."""
    assert MT._lint_table() == []


def test_illegal_cell_value_is_rejected_not_silently_unmounted():
    """`!` **Negative control: a typo'd cell value must raise immediately,
    not silently turn into "this cell isn't mounted".**

    Without this, the checks in `_lint_table` and `subject_of` might have
    nothing to check against.
    """
    with pytest.raises(ValueError, match="illegal subject"):
        MT.mount("ext_typo", {"multi": "tarj"})          # `traj` with one letter mistyped
    assert "ext_typo" not in MT.MOUNT

    # bypass `mount()` and stuff the table directly (tests/plugins genuinely
    # do this) -- the read path must still raise
    MT.MOUNT["ext_typo2"] = {"multi": "tarj"}
    with pytest.raises(ValueError, match="not a legal subject"):
        MT.subject_of("ext_typo2", "multi")
    assert MT._lint_table(), "`_lint_table` 必须能看见这条非法格值,否则它形同虚设"


def test_unknown_geometry_is_rejected_at_write_time():
    """Strict validation of geometry names happens at the **write path**
    (the read path is deliberately lenient; see `subject_of`'s docstring for
    the reason)."""
    with pytest.raises(KeyError, match="unknown geometry"):
        MT.mount("ext_badgeo", {"multiround": MT.TRAJ})
    MT.MOUNT["ext_badgeo2"] = {"multiround": MT.TRAJ}
    assert MT._lint_table(), "`_lint_table` 必须抓到打错字的**列名**"
    # while the read path still falls back to NONE for unknown geometry --
    # this is deliberate (`<unrecorded>` ends up here)
    assert MT.subject_of("ext_badgeo2", "<unrecorded>") == MT.NONE


# ============================================================ 2. load-bearing case: dispatch actually consumes TRAJ
def _dispatch_reads_traj(subject_of) -> bool:
    """**The single most load-bearing case in this file, pulled out into its
    own function so mutation tests can swap in a different implementation.**

    The question: given a `subject_of` implementation, does
    `run_judges("multi", {"traj": ...})` mount the trajectory-only judge and
    produce its keys.
    """
    import haenv.judges as _J
    _orig = MT.subject_of
    MT.subject_of = subject_of              # `judges` is only imported inside the function body, so swapping here is enough
    try:
        row = _J.run_judges("multi", {"traj": _traj(3)}, _VP())
    finally:
        MT.subject_of = _orig
    return row.get("trajprobe_n_rounds") == 3 and row.get("trajprobe_n_revised") == 1


def test_traj_subject_is_actually_dispatched():
    """Both halves done => `run_judges` **actually** feeds the trajectory
    into the judge on multi.

    This is not an assertion that "traj was written in the table" -- it goes
    through the entire production path: `applicable` -> `MOUNT` ->
    `subjects[want]`.
    """
    _mount_probe()
    assert MT.subject_of(_PROBE, "multi") == MT.TRAJ
    assert _PROBE in [j.name for j in J.applicable("multi", "ddx:unified", _VP())]
    assert _dispatch_reads_traj(MT.subject_of)


def test_mutation_constant_none_dispatch_must_fail():
    """`!` **mutation test: make dispatch constantly return `NONE`;
    the previous test must flip to a failure.**

    Without this, `test_traj_subject_is_actually_dispatched` might just be
    testing "there's a key in a dict"; a load-bearing claim needs a reading
    showing that breaking it turns it red.
    """
    _mount_probe()
    assert _dispatch_reads_traj(MT.subject_of), "变异前必须为真,否则这条比较没有对象"
    assert not _dispatch_reads_traj(lambda name, geom: MT.NONE), \
        "分派恒返回 NONE 时判据仍然产出了键 —— 说明它根本没走挂载表,上一条是假绿"


def test_mutation_wrong_subject_name_must_fail():
    """`!` Second mutant: dispatch returns a subject that **exists but is
    wrong** (`rows`).

    This is nastier than constant `NONE`: the judge is still selected by
    `applicable`, it just can't resolve its subject. In the artifact this
    becomes a `mount_subject_missing` entry -- it **must not produce the
    judge's keys**, and must not be substituted with a different subject
    instead (substituting would make the judge score something it was never
    meant to score).
    """
    _mount_probe()
    assert not _dispatch_reads_traj(lambda name, geom: MT.ROWS)

    import haenv.judges as _J
    _orig = MT.subject_of
    MT.subject_of = lambda name, geom: MT.ROWS if name == _PROBE else _orig(name, geom)
    try:
        row = _J.run_judges("multi", {"traj": _traj(3)}, _VP())
    finally:
        MT.subject_of = _orig
    assert "trajprobe_n_rounds" not in row
    assert any(s.startswith(f"{_PROBE}@multi:") for s in row.get("mount_subject_missing", [])), \
        "subject 取不到时必须记一行,不许静默跳过 —— 零命中 ≠ 安全"


def test_register_without_mount_never_runs_on_multi():
    """`!` Register only, no mount => still `NONE` on multi => registered
    but never runs.

    This pins down that `TRAJ` did not bypass the rule "missing either half
    means not mounted".
    """
    J.register_judge(J.Judge(_PROBE, ("*",), _judge_traj_probe), source="test:no-mount")
    assert MT.subject_of(_PROBE, "multi") == MT.NONE
    assert _PROBE not in [j.name for j in J.applicable("multi", "ddx:unified", _VP())]


# ============================================================ 3. mounting for the existing multi-round judges + honest coverage
def test_the_two_multiround_judges_have_a_traj_cell():
    """The cells for `judge_premise_repair` / `judge_multiround_revision`
    are already set, subject = `TRAJ`."""
    for name in ("premise_repair", "multiround_revision"):
        assert MT.subject_of(name, "multi") == MT.TRAJ, f"{name} 在 multi 上没挂 TRAJ"
        for g in ("single", "gated", "slices"):
            assert MT.subject_of(name, g) == MT.NONE
            assert MT.reason_for(name, g), f"{name}@{g} 是个没理由的空格"


def test_premounted_cells_are_reported_not_hidden():
    """**Premounted cells must be counted.** A cell existing in the
    table with no matching judge in `JUDGES` is a third state, and both
    `holes()` and `coverage()['mounted']` do `for j in JUDGES`, so **they
    can't enumerate it**.

    This is the dual of "zero hits != safe": two judges are mounted but still
    can never run, and if nobody reports that, it looks identical to
    "properly mounted" in the artifact.
    """
    # The two trajectory judges are in `_CORE` along with multi, so the
    # premounted count is zero; the negative control is "must not still count
    # once registered".
    # Note: a reading that always returns `[]` looks identical, in the
    #   artifact, to "everything is registered" -- but the former is broken
    #   => an assertion of "empty" must be immediately followed by a control
    #   that **can produce non-empty**.
    assert MT.unregistered_mounts() == [], \
        f"还有预挂载没注册:{MT.unregistered_mounts()}"
    assert MT.coverage()["unregistered"] == MT.unregistered_mounts(), \
        "覆盖读数与取值入口必须是同一条路径"
    # Positive control: remove a registered entry, it must be recounted as premounted
    _saved = list(J.JUDGES)
    J.JUDGES[:] = [j for j in J.JUDGES if j.name != "premise_repair"]
    try:
        assert "premise_repair" in MT.unregistered_mounts(), \
            "摘掉注册项后没被数成预挂载 ⇒ 这个读数恒空,上面那个 `[]` 证明不了任何事"
    finally:
        J.JUDGES[:] = _saved
    assert MT.unregistered_mounts() == [], "复位失败,后面的用例会读到脏状态"


def test_multi_column_coverage_is_reported_honestly():
    """**Report the scan surface before reporting a ratio**, and
    **a coverage increase does not mean a usage increase**.

    ## Why not `cbg["multi"]["mounted"] == 0`

    `mount_table` mounts 17 single-shot-family judges onto `LAST` (the
    final-round answer) and writes a not-applicable reason for each of the 8
    slice-family judges, so an expectation of an empty column would be wrong
    while the code is right.

    ## A nonzero count alone is not enough -- it conflates two different numbers

    `mounted == 17` says **the table has cells**; whether they run is a
    separate reading. So this test asserts **two things together**: coverage
    is nonzero **and** `coverage()["unwired"]` does not count `multi`
    (`evaluate._row_multi` calls `run_judges`). The check watches "is it
    filled in AND does it actually run".
    """
    cbg = MT.coverage_by_geometry()
    assert set(cbg) == set(MT.GEOMETRIES)
    assert cbg["multi"]["mounted"] > 0, (
        "multi 列又回到 0 —— 17 格单拍族挂载被撤了?"
        "连带看 `docs/design/multi-geometry-episode-subject.md` §6.1")
    # what's mounted must be the **single-shot family** (subject=LAST); the
    # slice family stays unmounted for structural reasons
    names = set(cbg["multi"]["mounted_names"])
    assert "forecast" in names and "workup" in names, \
        f"单拍族没挂在 multi 上,实测 mounted_names={sorted(names)}"
    assert not (names & {"slices", "slice_revision", "slices_gates"}), \
        "切片族被挂到 multi 上了 —— 它们读逐片字段,轨迹行里没有,硬挂会捏造读数"
    # The two trajectory judges are in `_CORE` along with multi, so the
    # premounted list is empty.
    assert set(cbg["multi"]["premounted_unregistered"]) == set(), \
        f"multi 列还有预挂载:{sorted(cbg['multi']['premounted_unregistered'])}"
    # counter-metric: mounted != run. `_row_multi` is wired in and
    #    `mounting.MOUNTS['multi'].profile` is set, so multi must not be
    #    counted as unwired.
    assert "multi" not in MT.coverage()["unwired"], (
        "multi 已接线(profile=MULTI + `_row_multi` 走 `run_judges`),"
        f"却还被数进 unwired:{MT.coverage()['unwired']}")
    # Positive control: other geometries are **not** 0, proving this reading
    # isn't always zero
    assert cbg["single"]["mounted"] > 0 and cbg["slices"]["mounted"] > 0, \
        "所有几何都是 0 ⇒ 覆盖读数根本没扫到东西,上一条恒真"
    # "a wired-in geometry isn't in unwired" is vacuously true once all
    #    four geometries are wired in (`"single" not in []` always holds), and a
    #    vacuously true control looks identical to no control at all. So the
    #    control takes the side that can go red: remove a `profile`, and it must
    #    be counted back immediately.
    import dataclasses
    from haenv import mounting as _MG
    _before = _MG.BY_GEOMETRY["slices"]
    _MG.BY_GEOMETRY["slices"] = dataclasses.replace(_before, profile="")
    try:
        assert "slices" in MT.coverage()["unwired"], \
            "profile 清空后没被数回来 ⇒ `unwired` 恒空,上面那条断言没有信息量"
    finally:
        _MG.BY_GEOMETRY["slices"] = _before
    assert MT.coverage()["unwired"] == [], "复位失败,后面的用例会读到脏状态"


def test_every_core_judge_has_a_reason_on_multi():
    """For the remaining empty cells in the `multi` column, **not one is
    allowed to lack a reason**; and the reasons must not be flattened by a
    single wildcard.

    ## Why not `reasons["slices"] == reasons["slice_revision"]`

    A single wildcard reason, `slices*@multi`, for the 8 slice-family judges
    would make them **equal** to each other and **not equal** to the
    single-shot family's genuine gap -- "two different kinds of absence must
    not be flattened into one". The reasons are **per judge** (each of the 8 spells out which field
    it's missing and what reading a forced mount would fabricate), so the
    reasons are not equal to each other by design.

    The property being guarded is **"N distinct kinds of absence must not
    be flattened by 1 reason"**, and its strongest form is **pairwise
    distinct, one by one**. So this test asserts the 8 reasons are pairwise
    distinct, and none of them falls back to the wildcard.
    """
    reasons = {}
    unmounted = []
    for j in J._CORE:
        r = MT.reason_for(j.name, "multi")
        assert r, f"{j.name}@multi 是个没理由的空格"
        reasons[j.name] = r
        if MT.subject_of(j.name, "multi") == MT.NONE:
            unmounted.append(j.name)
    assert len(set(reasons.values())) >= 2, (
        "整列共用同一条理由 —— 「真缺口」与「本来就不适用」又被盖成一种了,"
        "而那正是这张表存在的理由")
    # all empty cells fall in the slice family; the single-shot family is already mounted on LAST
    assert set(unmounted) == {
        "slices", "slice_revision", "slices_abstention", "slices_gates",
        "slices_review", "slices_workup", "slices_self_contradictory_exclusion",
        "slices_wnd"}, f"multi 列的空格集合变了:{sorted(unmounted)}"
    # load-bearing case: 8 distinct kinds of absence => 8 distinct reasons, none allowed to fall back to the wildcard
    per = {n: reasons[n] for n in unmounted}
    assert len(set(per.values())) == len(per), (
        "切片族的理由又被合并了 —— 两条判据缺的字段不一样,"
        f"盖成一条就分不出来了:{len(set(per.values()))}/{len(per)} 条互不相同")
    for n, r in per.items():
        assert r not in (MT.WHY_NOT["slices*@multi"], MT.WHY_NOT["*@multi"]), \
            f"{n}@multi 退回了通配理由 —— 逐条理由被删掉了一条"
    # `_row_multi` runs `run_judges`, so the external-judge fallback on multi is not a known gap
    assert "*@multi" not in MT.known_gaps()


def test_holes_stay_empty_after_the_change():
    """The cross product of judges and geometries has no reasonless holes --
    counter-metric to the coverage reading."""
    assert MT.holes() == []
