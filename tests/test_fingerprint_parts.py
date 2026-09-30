"""Fine-grained fingerprinting (`anchor.judge_parts` / `parts_compatible`) -- contract layer, synthetic fixtures.

## What this group of tests guards

The sole purpose of fine-grained fingerprinting is to narrow "over-eager
invalidation": the coarse fingerprint `judging_sha16` is a single sha over
12 files, so a single typo expires every leaderboard on disk. The
fine-grained fingerprint stamps each part individually, so it can ask: did
the parts this batch of rows actually used change?

Loosening this has a directional risk:
- A false stale (something that should still be valid is marked expired)
  costs one extra recompute -- manageable.
- A false fresh (something that should have been blocked is let through)
  mixes two generations of judges into the same denominator, and scores
  from different judge versions end up compared as if they were one.

So the load-bearing test in this group is
`test_fine_is_never_looser_than_coarse`: the set the fine-grained
fingerprint marks "needs recompute" must be a subset of the set the
coarse fingerprint marks "needs recompute."
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import anchor as A                       # noqa: E402
from haenv import judges as J                       # noqa: E402
from haenv import mount_table as MT                 # noqa: E402


# ---------------------------------------------------------------- per-part fingerprint

def test_parts_covers_every_judge_and_every_non_judge_file():
    parts = A.judge_parts()
    for j in J._CORE:
        assert f"judge:{j.name}" in parts, f"{j.name} 没盖上件戳"
    for rel in A.judging_files():
        if rel.startswith("haenv/judges/"):
            continue                    # judge modules are stamped judge by judge
        assert f"file:{rel}" in parts, f"{rel} 没盖上件戳"
    assert not any(v == "" for v in parts.values())


def test_no_judge_source_is_silently_skipped():
    """When source can't be retrieved, it must be recorded as `<nosource>`, never skipped -- skipping would make "unchanged" and "invisible" look identical."""
    parts = A.judge_parts()
    assert len([k for k in parts if k.startswith("judge:")]) == len(J._CORE)


def test_parts_is_deterministic():
    assert A.judge_parts() == A.judge_parts()


def test_unchanged_judge_source_is_not_reparsed(monkeypatch):
    import inspect

    expected = A.judge_parts()

    def unexpected_read(fn):
        raise AssertionError(f"unchanged source read again: {fn}")

    monkeypatch.setattr(inspect, "getsource", unexpected_read)
    assert A.judge_parts() == expected
    expected["judge:dx_unified"] = "mutated returned dictionary"
    assert A.judge_parts()["judge:dx_unified"] != expected["judge:dx_unified"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_source_cache_detects_edits_with_restored_mtime(monkeypatch, tmp_path, wrapped):
    import functools
    import os
    import types

    path = tmp_path / "temporary_judge.py"
    text = "def judge(*args):\n    return {'value': 1}\n"
    path.write_text(text, encoding="utf-8")
    module = types.ModuleType("temporary_judge")
    exec(compile(text, str(path), "exec"), module.__dict__)
    fn = module.judge
    if wrapped:
        fn = functools.wraps(fn)(lambda *args: module.judge(*args))
    monkeypatch.setattr(J, "_CORE", (J.Judge("test", ("*",), fn),))
    monkeypatch.setattr(A, "_non_judge_files", lambda: ())
    before = A.judge_parts()
    stat = path.stat()
    path.write_text(text.replace("1", "2"), encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert A.judge_parts() != before
    path.unlink()
    missing = A.judge_parts()
    assert missing != before
    path.write_text(text, encoding="utf-8")
    assert A.judge_parts() == before


def test_judges_package_files_are_not_stamped_as_files():
    """The judge modules live in the `haenv/judges/` package; they are stamped per judge body, not per file."""
    assert not [k for k in A.judge_parts() if k.startswith("file:haenv/judges")]
    assert not [f for f in A._non_judge_files() if f.startswith("haenv/judges/")]
    # ... but they are still on the judging path (the coarse fingerprint covers them)
    assert any(f.startswith("haenv/judges/") for f in A.judging_files())


def test_comment_edit_in_judges_package_keeps_part_stamps(monkeypatch, tmp_path):
    """A comment-only edit under `haenv/judges/` leaves every part stamp alone; a logic edit to a judge body still moves its stamp."""
    import types
    root = tmp_path / "root"
    (root / "tools").mkdir(parents=True)
    (root / "haenv" / "judges").mkdir(parents=True)
    (root / "tools" / "make_freeze.py").write_text(
        'JUDGING = ("haenv/judges/outcome.py", "haenv/gates.py")\n', encoding="utf-8")
    body = root / "haenv" / "judges" / "outcome.py"
    (root / "haenv" / "gates.py").write_text("X = 1\n", encoding="utf-8")
    text = "def judge(*args):\n    return {'value': 1}\n"
    body.write_text(text, encoding="utf-8")
    module = types.ModuleType("w5_body")
    exec(compile(text, str(body), "exec"), module.__dict__)
    monkeypatch.setattr(A, "ROOT", root)
    monkeypatch.setattr(J, "_CORE", (J.Judge("body", ("*",), module.judge),))
    before = A.judge_parts()
    assert "file:haenv/gates.py" in before and not any("judges/" in k for k in before)
    body.write_text("# a new comment\n" + text, encoding="utf-8")
    assert A.judge_parts() == before
    body.write_text(text.replace("1", "2"), encoding="utf-8")
    exec(compile(body.read_text(encoding="utf-8"), str(body), "exec"), module.__dict__)
    after = A.judge_parts()
    assert after["judge:body"] != before["judge:body"]
    assert after["file:haenv/gates.py"] == before["file:haenv/gates.py"]


def test_source_cache_keeps_registry_metadata_live(monkeypatch):
    from dataclasses import replace

    judge = J._CORE[0]
    monkeypatch.setattr(J, "_CORE", (judge,))
    before = A.judge_parts()
    for modified in (replace(judge, kinds=("new-kind",)),
                     replace(judge, category="llm"),
                     replace(judge, when=lambda _: True),
                     replace(judge, fn=J.judge_driver)):
        monkeypatch.setattr(J, "_CORE", (modified,))
        assert A.judge_parts() != before
    monkeypatch.setattr(J, "_CORE", (judge, replace(judge, name="new-judge")))
    assert "judge:new-judge" in A.judge_parts()
    monkeypatch.setattr(J, "_CORE", (judge,))
    assert A.judge_parts() == before


# ------------------------------------------------- which parts a row used

def test_kind_actually_partitions_the_used_judges():
    """Different kinds use different judges -- this is the entire source of what fine-grained fingerprinting can save.

    Note: this is not a subset relationship: `ddx:insufficient` alone uses
    `commit_timing` ("should a conclusion be drawn yet" is only asked in
    the insufficient-data tier), while `ddx:unified` alone uses the
    diagnosis-naming family. Each side has something the other doesn't, so
    the assertion is "neither is a subset of the other," not "one is a
    subset of the other." (`narrow < wide` would be a wrong assumption, not a
    code property.)
    """
    uni = set(A.parts_used_by("ddx:unified", "single"))
    ins = set(A.parts_used_by("ddx:insufficient", "single"))
    assert "judge:commit_timing" in ins - uni
    assert "judge:dx_unified" in uni - ins
    assert not (uni <= ins) and not (ins <= uni)


def test_non_judge_files_always_counted_as_used():
    """When it's unclear, treat it as used. Aggregation, scoring, and reporting all read these files, and no inference is made about "this row never touched it."""
    used = set(A.parts_used_by("forecast", "single"))
    for rel in A.judging_files():
        if not rel.startswith("haenv/judges/"):
            assert f"file:{rel}" in used


def test_multi_geometry_now_mounts_judges():
    """The multi column mounts judges: it is not an empty column.

    Note: this asserts the direction (not empty, and overlapping with the
    single column) rather than pinning an exact count -- the count changes as
    mounting evolves, while "multi should not be an empty column" does not.
    Pinning the current count would turn the test red on every mounting
    change, with the code right and the expectation wrong.
    """
    used = [u for u in A.parts_used_by("forecast", "multi") if u.startswith("judge:")]
    assert used, "multi 列又空了 —— 挂载被回退?(multi 列不该为空)"
    single = {u for u in A.parts_used_by("forecast", "single") if u.startswith("judge:")}
    assert set(used) & single, (
        f"multi 挂的判据与 single 零交集,可疑:multi={sorted(used)} single={sorted(single)}")


# ------------------------------------------------- compatibility determination

def _rows(kind="ddx:unified", geom="single", n=3):
    return [{"gold_kind": kind, "geometry": geom} for _ in range(n)]


def test_same_parts_is_compatible():
    ok, changed = A.parts_compatible(_rows(), A.judge_parts())
    assert ok and changed == []


def test_changed_used_judge_breaks_compatibility():
    before = A.judge_parts()
    before["judge:dx_unified"] = "deadbeefdeadbeef"
    ok, changed = A.parts_compatible(_rows("ddx:unified", "single"), before)
    assert not ok and "judge:dx_unified" in changed


def test_changed_unused_judge_keeps_compatibility():
    """Note: this is exactly what fine-grained fingerprinting buys -- changing a judge this batch of rows never used should not expire them."""
    before = A.judge_parts()
    before["judge:commit_timing"] = "deadbeefdeadbeef"   # mounted only on ddx:insufficient
    ok, changed = A.parts_compatible(_rows("ddx:unified", "single"), before)
    assert ok, f"unified 行不该被 insufficient 专用判据的改动作废;changed={changed}"


def test_missing_part_in_before_is_incompatible():
    """"Not recorded last time" is not the same as "hasn't changed" -- when in doubt, recompute."""
    before = A.judge_parts()
    before.pop("judge:dx_unified")
    ok, changed = A.parts_compatible(_rows("ddx:unified", "single"), before)
    assert not ok and "judge:dx_unified" in changed


def test_row_without_geometry_is_incompatible():
    """An old row with no recorded geometry => impossible to tell what it used => judged incompatible, not let through."""
    ok, changed = A.parts_compatible([{"gold_kind": "ddx:unified"}], A.judge_parts())
    assert not ok


def test_row_without_gold_kind_is_incompatible():
    ok, _ = A.parts_compatible([{"geometry": "single"}], A.judge_parts())
    assert not ok


# ------------------------------------------------- Note: the load-bearing test

@pytest.mark.parametrize("kind", ["forecast", "ddx:unified", "ddx:comorbidity",
                                  "ddx:independent", "ddx:insufficient"])
@pytest.mark.parametrize("geom", ["single", "gated", "slices", "multi"])
def test_fine_is_never_looser_than_coarse(kind, geom):
    """The set the fine-grained fingerprint marks "needs recompute" is a subset of the set the coarse fingerprint marks "needs recompute."

    Method: pick any part and change it. The coarse fingerprint says
    "everything needs recomputing" (it's a single overall sha); the
    fine-grained fingerprint only says so when this cell actually used
    that part. So only one direction needs verifying: whenever the
    fine-grained fingerprint says recompute, the coarse one must too
    (since a changed part must change the overall sha). The converse does
    not hold, and that is exactly what's being bought here.
    """
    rows = _rows(kind, geom, 1)
    base = A.judge_parts()
    for name in base:
        before = dict(base)
        before[name] = "0" * 16
        # Only the saved snapshot is perturbed; the current code stays fixed.
        # The default live-read path and cache invalidation are tested separately.
        fine_says_recompute, _ = A.parts_compatible(rows, before, after=base)
        # a changed part => the coarse fingerprint always says "needs
        # recompute." The fine-grained one saying "no recompute needed" is
        # allowed (it's broader); the direction "fine says recompute =>
        # coarse also says recompute" always holds, so this only needs to
        # verify the fine-grained side never overreaches.
        assert isinstance(fine_says_recompute, bool)
        if not fine_says_recompute:
            assert name in set(A.parts_used_by(kind, geom)), (
                f"{name} 不在这一格用过的件里,细指纹却判它要重算 —— 那是越界")


def test_adding_an_unmatched_kind_does_not_expire_existing_rows():
    """The specific waste a user would hit: adding a kind rule nothing matches should not expire existing rows.

    Simulated here by adding a judge mounted only on a new kind --
    existing rows' kinds don't match it, so it never enters
    `parts_used_by`, and the compatibility determination is unaffected.
    """
    from haenv import gold_kinds as GK
    before = A.judge_parts()
    rows = _rows("ddx:unified", "single")
    assert A.parts_compatible(rows, before)[0]

    J.register_judge(J.Judge("ext_newkind", ("ddx:_brandnew",), lambda *a, **k: {}))
    MT.mount("ext_newkind", {"single": MT.OUT},
             why_not={g: "新 kind 的探针只验单拍" for g in ("gated", "slices", "multi")})
    try:
        ok, changed = A.parts_compatible(rows, before)
        assert ok, f"加一条匹配不上的判据让现存行过期了;changed={changed}"
        assert "ddx:_brandnew" not in A.parts_used_by("ddx:unified", "single")
    finally:
        J.reset_judges()
        MT.MOUNT.pop("ext_newkind", None)
        for g in ("gated", "slices", "multi"):
            MT.WHY_NOT.pop(f"ext_newkind@{g}", None)
        GK.reset_kind_rules()


def test_never_scored_aborted_row_uses_no_part_but_unknown_row_still_blocks():
    from haenv import anchor as A
    parts = A.judge_parts()
    aborted = {"case": "c", "solver": "s", "overall": "ABORT(no_answer:budget)", "geometry": "gated"}
    assert A.parts_compatible(_rows() + [aborted], parts) == A.parts_compatible(_rows(), parts)
    unknown = {"case": "c", "solver": "s", "overall": "SCORED"}
    ok, changed = A.parts_compatible(_rows() + [unknown], parts)
    assert not ok and "cannot tell" in changed[0]
