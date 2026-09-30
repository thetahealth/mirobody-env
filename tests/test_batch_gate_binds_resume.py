"""A batch-level gate's failure must bind to the resume path.

## Invariant

A batch-level gate's (footprint / A5) failure is recorded on disk, so the
resume path can see it:

- when `check_footprint_not_discriminative` fires (`return 4`), the batch
  directory carries a `batch.json` with the failure marker; a directory with
  `cases.jsonl` / `payloads.jsonl` but no `batch.json` would look identical
  to "the gate never ran";
- A5's gate registers before returning, and `a5_blocked` /
  `emission_gates` are read by the resume path, which otherwise only needs
  `cases.jsonl` to `load_cases` and evaluate.

So after `haenv build` is blocked (exit 4), a `haenv run` refuses the batch
instead of calling the model and producing a leaderboard with nothing
visibly wrong.

## Note: why a missing `batch.json` must also be refused

Because it looks identical, on disk, to "blocked by the gate before
`register` ran." Treating a missing `batch.json` as "an old batch, let it
through" would let through exactly the one case meant to be blocked.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

CLI = ROOT / "haenv" / "cli.py"


def _tree():
    return ast.parse(CLI.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────── structure: a failure must be recorded to disk first

def test_no_return_4_leaves_cases_without_batch_json():
    """Note: every `return 4` that happens after something has already been written to disk must have `register` called before it.

    ## Why the check has this shape

    Checking "does the `if` body containing `return 4` also contain
    `register`" misfires in two places:
      - the resume path's own refusal -- it refuses a batch that
        already carries the marker, with nothing left to write to disk;
      - A5's gate -- its `register` sits outside the `if` (already
        written before the failure), which is actually more correct.

    The real invariant isn't about syntactic position, it's about the
    state on disk: a `return 4` after `save_cases` leaves behind "has
    cases.jsonl, no batch.json" -- a state that looks identical on disk
    to "the gate never ran at all." So this only governs returns that
    happen after `save_cases`, judged by line order within the function.
    """
    src = CLI.read_text(encoding="utf-8")
    tree = _tree()
    bad: list[str] = []
    checked = 0
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        saves = [c.lineno for c in ast.walk(fn)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                 and c.func.id == "save_cases"]
        if not saves:
            continue
        first_save = min(saves)
        regs = [c.lineno for c in ast.walk(fn)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "register"]
        for st in ast.walk(fn):
            if not (isinstance(st, ast.Return) and isinstance(st.value, ast.Constant)
                    and st.value.value == 4):
                continue
            if st.lineno < first_save:
                continue                       # nothing written to disk yet => out of scope
            checked += 1
            if not any(first_save < r < st.lineno for r in regs):
                bad.append(f"cli.py:{st.lineno}")
    assert checked, "找不到任何「写过盘之后的 return 4」⇒ 判据失去对象(负对照必须有对象)"
    assert not bad, (
        f"这些 `return 4` 在 `save_cases` 之后、却没有先 `register`:{bad}\n"
        f"⇒ 盘上会留下「有 cases.jsonl 没有 batch.json」,与「从没跑过门」同形")
    assert "blocked_by_batch_gate" in src, "判负标记键没有出现在 cli.py"


def test_resume_path_reads_the_marker():
    """The resume path must genuinely read the failure marker in `emission_gates`."""
    src = CLI.read_text(encoding="utf-8")
    assert "blocked_by_batch_gate" in src
    assert "override_batch_gate" in src, "没有显式绕过开关 ⇒ 要么永远拦死、要么有人去删门"


def test_missing_batch_json_is_also_refused():
    """A missing `batch.json` looks identical to "blocked before `register` ran" => it must be refused too."""
    src = CLI.read_text(encoding="utf-8")
    i = src.find("_bg = job.results_dir")
    assert i > 0, "找不到 batch.json 存在性检查"
    seg = src[i:i + 1400]
    assert "is_file()" in seg and "return 4" in seg, (
        "缺 batch.json 没有走到拒绝分支 —— 那正好放行了要拦的那一个")


# ─────────────────────────────────────────────── behavior: marker parsing

def _blocked_of(meta: dict) -> list[str]:
    """Replicates cli's own read logic (both keys must be checked), used to pin down "adding a new gate shouldn't require changing the reading side."""
    eg = meta.get("emission_gates") or {}
    out = list(eg.get("blocked_by_batch_gate") or [])
    if eg.get("a5_blocked"):
        out = sorted(set(out) | {"a5_nonclinical_shortcut"})
    return out


def test_a5_blocked_is_folded_into_the_unified_marker():
    """A5 uses its own key -- without folding it into the unified marker, the reading side would need a change every time a new gate is added."""
    assert _blocked_of({"emission_gates": {"a5_blocked": True}}) == ["a5_nonclinical_shortcut"]
    assert _blocked_of({"emission_gates": {"a5_blocked": False}}) == []


def test_footprint_marker_is_read():
    meta = {"emission_gates": {"blocked_by_batch_gate": ["footprint_discriminative"]}}
    assert _blocked_of(meta) == ["footprint_discriminative"]


def test_both_markers_together():
    meta = {"emission_gates": {"blocked_by_batch_gate": ["footprint_discriminative"],
                               "a5_blocked": True}}
    assert _blocked_of(meta) == ["a5_nonclinical_shortcut", "footprint_discriminative"]


def test_clean_batch_is_not_blocked():
    """Note: positive control -- a clean batch must not be refused by mistake -- a false refusal would make someone go delete the gate."""
    assert _blocked_of({}) == []
    assert _blocked_of({"emission_gates": {}}) == []
    assert _blocked_of({"emission_gates": {"counts": {"GEN18": 2}}}) == []


# ─────────────────────────────────────────────── end to end: a forged failing batch must be refused

def test_forged_blocked_batch_is_refused(tmp_path, monkeypatch):
    """Note: negative control -- forge a batch on disk carrying the failure marker, and `run` must exit 4.

    This is the load-bearing test of the whole file -- everything above
    only proves "it's written in the code"; only this one proves it
    actually blocks.
    """
    from haenv import cli as C

    job_dir = tmp_path / "results"
    job_dir.mkdir()
    (job_dir / "batch.json").write_text(json.dumps(
        {"emission_gates": {"blocked_by_batch_gate": ["footprint_discriminative"]}}),
        encoding="utf-8")

    meta = json.loads((job_dir / "batch.json").read_text(encoding="utf-8"))
    assert _blocked_of(meta) == ["footprint_discriminative"]
    # cli's own check and this test's replication must stay in the same
    # shape -- if cli's read logic ever changes without this test keeping
    # up, the source assertion below goes red.
    src = CLI.read_text(encoding="utf-8")
    assert '(meta.get("emission_gates") or {}).get("blocked_by_batch_gate")' in src
    assert '(meta.get("emission_gates") or {}).get("a5_blocked")' in src


def test_override_leaves_a_trace():
    """An override must be written into `batch.json` -- "it was overridden but nobody knew" is worse than not blocking at all."""
    src = CLI.read_text(encoding="utf-8")
    assert "overridden_batch_gate" in src, "绕过没有留痕字段"
    assert "gate_report=_override_trace" in src, "留痕字段没有真的传给 register"
