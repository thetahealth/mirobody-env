"""The emission invariant, and the one table that depends on it.

    case in cases.jsonl  <=>  premise_ok and leak_ok and emitted

`report.render`'s emission-gate table -- section 0, the first table in every
report -- prints two per-case gate verdicts, read off the audit under the keys
`premise_ok` and `leak_ok`. A missing key must never render as a verdict.

On the four-command path (`build` -> `verify` -> `run` -> `report`), `report`
reads the cases back out of the batch instead of rebuilding them, and `cli.py`
reconstructs the audit for each row. The two kinds of row are handled
differently:

* Emitted cases: `cli.py` fills both keys in. It may, because they are
  derived: `_build_case_inner` returns `(None, audit)` on either failure, only
  `raw is not None` reaches `built`, and only `built` is persisted. This file
  pins that derivation, so that making emission reachable without the leak
  probe fails here.
* Blocked cases: the two flags are unknown -- `batch.json`'s `emission_gates`
  records why a case was blocked, never its `leak_ok` -- so nothing is filled
  in, and the renderer has a third state (`report._gate_cell`: passed / failed
  / not recorded). Skipped does not read as passed, and absent does not read
  as failed.

## Both halves, with a positive and a negative control

* `test_leak_blocks_emission` -- the probe's verdict gates emission (patch it
  to fail; nothing emits).
* `test_clean_case_emits_and_records_both_flags` -- the negative control: with
  the real probe the same case emits and records both flags true, so the test
  above does not pass merely because the builder is broken in general.
* `test_emitted_readback_carries_the_invariant` /
  `test_blocked_readback_does_not_guess` -- the two read-back stubs `cli.py`
  hands to `report.render`: the emitted one carries both flags, the blocked
  one carries neither.
* `test_renderer_prints_three_states` /
  `test_renderer_row_uses_the_three_state_cell` -- the consumer side: what
  `render` prints for each of true / false / missing key, and that section 0
  routes through it.

SYNTHETIC, evaluation only, not medical advice.
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

from haenv import build as B                                  # noqa: E402
from haenv import job as J                                    # noqa: E402

_JOB = ROOT / "inputs" / "example-ew.job.yaml"


def _first_case():
    job = J.load_job(_JOB)
    return job.cases[0]


@pytest.fixture()
def _case():
    if not _JOB.is_file():
        pytest.skip(f"{_JOB} is not on disk")
    return _first_case()


def test_leak_blocks_emission(_case, monkeypatch):
    """Positive control: a failing leak probe must stop the case, not
    downgrade it."""
    monkeypatch.setattr(B, "leakage_probe",
                        lambda sp, T: (False, ["injected-by-test"]))
    raw, audit = B._build_case_inner(_case, T=_case.index_time_T)          # noqa: SLF001
    assert raw is None, (
        "leak probe failed but the case was still emitted -- the emission invariant is broken, "
        "and the leak_ok=True that cli.py fills in on read-back becomes a lie")
    assert audit.get("leak_ok") is False
    assert not audit.get("emitted")


def test_clean_case_emits_and_records_both_flags(_case):
    """Negative control: with the real probe the same case goes through, so the
    test above is not green merely because the builder refuses everything."""
    raw, audit = B._build_case_inner(_case, T=_case.index_time_T)          # noqa: SLF001
    if raw is None:
        pytest.skip(f"this case was stopped by another gate ({audit.get('post_noise_conflicts')}); "
                    f"rerun this control on a different case")
    assert audit.get("premise_ok") is True
    assert audit.get("leak_ok") is True
    assert audit.get("emitted") is True


def _readback_stubs() -> dict[bool, set[str]]:
    """The two audit literals `cli.py` builds on the read-back path, keyed by
    the literal value of their `emitted`.

    Read off the AST rather than by running it: the branch needs a batch on
    disk to reach. Matched on the **dict literal**, so a comment naming
    `premise_ok` cannot stand in for the key.
    """
    import ast
    out: dict[bool, set[str]] = {}
    for node in ast.walk(ast.parse((ROOT / "haenv" / "cli.py").read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Dict):
            continue
        pairs = {k.value: v for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if not {"case_id", "from_batch"} <= set(pairs):
            continue
        em = pairs.get("emitted")
        if isinstance(em, ast.Constant) and isinstance(em.value, bool):
            out[em.value] = set(pairs)
    return out


def test_emitted_readback_carries_the_invariant():
    """`emitted: True` on read-back must carry both derived flags."""
    stubs = _readback_stubs()
    assert True in stubs, "the 'emitted' read-back literal was not found; cli.py changed shape and this guard lost its object"
    assert {"premise_ok", "leak_ok"} <= stubs[True], (
        "the batch read-back audit lacks premise_ok / leak_ok -- "
        "report section 0 would print a clean batch as 'premise ✗ · leak gate LEAK'")


def test_blocked_readback_does_not_guess():
    """The blocked direction.

    A case the gate stopped is not covered by the emission invariant, and
    `batch.json`'s `emission_gates` records no `leak_ok` for it. Setting either
    flag here would turn "not recorded" into a verdict. The blocked stub
    carries the reason and neither of the two flags.
    """
    stubs = _readback_stubs()
    assert False in stubs, (
        "the 'blocked' read-back literal was not found -- blocked cases vanished from report section 0 again, "
        "and the denominator would read n/n with 0 blocked")
    assert not ({"premise_ok", "leak_ok"} & stubs[False]), (
        "a blocked case's audit carries premise_ok / leak_ok -- neither value exists on disk, "
        "so filling them turns 'not recorded' into a verdict")
    assert "post_noise_conflicts" in stubs[False], (
        "a blocked case carries no reason; its row would show only the blocked mark and not say why")


def test_renderer_prints_three_states():
    """What `render` does with each of the three inputs.

    Both sides are pinned: a test that only checks the missing key would also
    pass on a renderer that prints `not recorded` for everything. All three
    states are pinned, per column:

        value true   -> pass mark / `CLEAN`         (passed)
        value false  -> fail mark / `LEAK`          (failed, and the premise reason rides along)
        key missing  -> `NOT_RECORDED`              (not recorded)

    The three are mutually distinguishable on the page; the property is
    distinguishability, not the exact glyphs.
    """
    from haenv import report as R                                     # noqa: PLC0415

    assert R._gate_cell({"leak_ok": True}, "leak_ok", "CLEAN", "LEAK") == "CLEAN"   # noqa: SLF001
    assert R._gate_cell({"leak_ok": False}, "leak_ok", "CLEAN", "LEAK") == "LEAK"   # noqa: SLF001
    assert R._gate_cell({}, "leak_ok", "CLEAN", "LEAK") == R.NOT_RECORDED, (        # noqa: SLF001
        "a missing leak-gate key still renders as a verdict -- a blocked case has no leak_ok on disk, "
        "so printing LEAK reports a leak that was never probed, and real leaks become unreadable")

    assert R._gate_cell({"premise_ok": True}, "premise_ok", "✓", "✗ x") == "✓"      # noqa: SLF001
    assert R._gate_cell({"premise_ok": False}, "premise_ok", "✓", "✗ out of range") == "✗ out of range"  # noqa: SLF001
    assert R._gate_cell({}, "premise_ok", "✓", "✗ x") == R.NOT_RECORDED, (          # noqa: SLF001
        "a missing premise key still renders as ✗ -- 'not recorded' was written as failed")

    assert len({"CLEAN", "LEAK", R.NOT_RECORDED}) == 3, "the three states must be distinguishable on the page"
    assert R.NOT_RECORDED.strip() not in ("", "—"), (
        "the third state collapsed to a bare `—`, indistinguishable from 'this column has no item'; "
        "it has to say 'not recorded'")


def test_renderer_row_uses_the_three_state_cell():
    """The helper is the thing section 0 actually calls.

    Pinned on the AST, not on a substring: a comment naming `_gate_cell`, or a
    two-state f-string left next to an unused helper, does not pass. Keyed on
    the literal column names so that dropping one column back to the two-state
    form is caught.
    """
    import ast
    src = (ROOT / "haenv" / "report.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_gate_cell"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)):
            keys.add(node.args[1].value)
    assert {"premise_ok", "leak_ok"} <= keys, (
        "the emission-gate table no longer goes through the three-state cell -- check what render's "
        f"section 0 became; columns currently rendered via _gate_cell: {sorted(keys)}")

    two_state = [n for n in ast.walk(tree)
                 if isinstance(n, ast.IfExp)
                 and {getattr(n.body, "value", None), getattr(n.orelse, "value", None)}
                 & {"CLEAN", "LEAK"}]
    assert not two_state, (
        "report.py again has a two-state `'CLEAN' if … else 'LEAK'` render; "
        "a missing key would print as failed again")
