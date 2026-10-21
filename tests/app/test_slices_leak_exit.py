"""The slices geometry's leak exit.

Single-shot and multi-round cells end in `ABORT(leak)` when the solver output leaks.
The slices geometry has the same exit: a slice that leaks keeps only `{"t", "leak"}`
and the loop continues; when every slice leaks, `_out_last` stays `None`, `_gates`
gets `[]`, and the cell must still end in `ABORT(leak)` with no score, not `SCORED`.

## Three tests, two directions

* every slice leaks => `ABORT(leak)`, and the leak reason is named in the output;
* partial leak => no ABORT (the leaked slice is voided, the rest keep running);
* the last slice leaks => `_out_last` is `None`, and the 8 judges on the `last`
  subject go through `mount_subject_missing`, never substituted with an earlier slice.

SYNTHETIC, evaluation only, not medical advice.
"""

from __future__ import annotations


from app._fixtures_slices_leak_exit import *  # explicit shared fixture module

def test_all_slices_leak_aborts(monkeypatch):
    """Every slice leaking => `ABORT(leak)`, with the reason named
    explicitly."""
    _install(monkeypatch, lambda t: _facts(leak=[f"leak_at_{t}"]))
    row = EV._row_slices("C1", "m1", _Raw(), [10, 20, 30], _Solver(), ctx=_CTX)
    assert row["overall"] == "ABORT(leak)", row.get("overall")
    assert row.get("leak_scope") == "all_slices"
    assert set(row.get("leak") or []) == {"leak_at_10", "leak_at_20", "leak_at_30"}, row.get("leak")
    assert row.get("answered_slices") == 0

def test_partial_leak_does_not_abort(monkeypatch):
    """Negative control: a **partial** leak must not ABORT.

    An assertion that any leak fails the whole cell would also condemn the
    intended behavior of voiding that slice and running the rest.
    """
    _install(monkeypatch, lambda t: _facts(leak=["x"]) if t == 10 else _facts())
    row = EV._row_slices("C1", "m1", _Raw(), [10, 20, 30], _Solver(), ctx=_CTX)
    assert row["overall"] != "ABORT(leak)", row.get("overall")
    assert row.get("leak_scope") is None

def test_last_slice_leak_drops_out_last(monkeypatch):
    """The last slice leaking => the `last` subject is absent, **never
    substituted with an earlier slice**."""
    _install(monkeypatch, lambda t: _facts() if t in (10, 20) else _facts(leak=["x"]))
    row = EV._row_slices("C1", "m1", _Raw(), [10, 20, 30], _Solver(), ctx=_CTX)
    assert row["overall"] != "ABORT(leak)", "只有末片泄漏,不该整格 ABORT"
    assert row.get("_saw_last") is False, "末片泄漏了,`last` 仍被喂进判据 ⇒ 拿早片顶替"

def test_healthy_path_still_passes_last(monkeypatch):
    """Reverse: with no leak, `last` **must** still be fed in as usual
    (otherwise the previous test would be vacuously true)."""
    _install(monkeypatch, lambda t: _facts())
    row = EV._row_slices("C1", "m1", _Raw(), [10, 20, 30], _Solver(), ctx=_CTX)
    assert row.get("_saw_last") is True, "正常路径下 `last` 没被喂进判据"
