"""pytest wiring for `tests/`.

1. Puts `tests/` on `sys.path`, so test modules import their shared helpers
   by bare name.
2. Restores the judge registry, the external-task tables and the world-plugin
   state around every test (`_isolate_registries`).

Maintainer-only hooks (test tiering, the runtime ledger, the evaluation-batch
summary) are re-exported from `_maintainer_hooks` when that module is present.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ROOT = _HERE.parent


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Restore judge, external-task and world-plugin state after each test.

    `job.load_job()` ends with `load_plugins(job)`, which writes module-level
    registries and never restores them. On the production path that is
    correct: a process loads one job. In a test session it would make results
    depend on test order -- a test that loads a job declaring `plugins:`
    would change the registries every later test sees. This fixture snapshots
    the tables before each test and restores them afterwards.

    A test that forgets to unregister what it registered still fails when it
    runs on its own; the fixture only removes what an earlier test left
    behind.
    """
    import haenv.judges as J
    import haenv.mount_table as MT
    import haenv.external_gold as EG
    import haenv.job as JOB
    import haenv.world_plugins as WP
    from copy import deepcopy

    # The tables have three different types: `JUDGES` is `list[Judge]`,
    # `MOUNT`/`WHY_NOT` are `dict`, and `SUBJECTS` is a custom `_Subjects`
    # (no `clear`/`update`, not a `MutableMapping`). Each is restored by type.
    targets = [(J, "JUDGES"), (MT, "MOUNT"), (MT, "WHY_NOT")]
    targets += [(EG, name) for name in
                ("BLOCKS", "LATENT_CLASSES", "GATES", "FRAMINGS", "PROBES")]
    targets += [(JOB, "LATENT_REGISTRY"), (JOB, "_LOADED_GROUPS")]
    targets += [(WP, name) for name in
                ("_OVERLAY", "_SOURCES", "_POST_INJECTORS", "_LOADED")]
    # Plugins change the world stamp within a test, but may not survive into the next one.
    # Nested overlay/probe values need copies too; shallow snapshots retain in-place edits.
    nested = {(EG, "PROBES"), (WP, "_OVERLAY")}
    snap = {}
    for mod, name in targets:
        cur = getattr(mod, name, None)
        snap[mod, name] = (list(cur) if isinstance(cur, list)
                           else deepcopy(cur) if (mod, name) in nested
                           else dict(cur) if isinstance(cur, dict)
                           else set(cur) if isinstance(cur, set) else None)
    try:
        yield
    finally:
        for mod, name in targets:
            cur, old = getattr(mod, name, None), snap[mod, name]
            if old is None:
                continue                    # unrecognized type => leave it alone
            if isinstance(cur, list):
                cur[:] = old
            elif isinstance(cur, dict):
                cur.clear()
                cur.update(old)
            elif isinstance(cur, set):
                cur.clear()
                cur.update(old)
        # `SUBJECTS` has its own unregistration entry point: the test that
        # registers an external subject removes it with `unregister_subject`.


_SOLVING_FP_MEMO: dict = {}


@pytest.fixture(autouse=True)
def _stable_solving_fingerprint(monkeypatch):
    """One solving-code fingerprint per worker process and file list.

    Three guards probe fingerprint sensitivity by appending to real tracked files and
    restoring them (`test_freeze_discipline` over every GENERATION file,
    `guards/test_semantic_fingerprint`, `guards/test_backend_limits`). Under xdist a
    batch identity computed in another worker inside that window differs from the one
    computed a moment later, and a resume test reads "identity changed". The identity
    is a pure function of the file contents; within one test session those contents
    are fixed, so the value is computed once per worker. Tests that change what the
    fingerprint sees patch `_fingerprint_of_files` themselves (their patch wins).
    """
    import haenv.anchor as A
    import haenv.solver_accounting as SA
    real = SA._fingerprint_of_files

    def memo(files, root=None):
        key = (str(A.ROOT), tuple(files), root)
        if key not in _SOLVING_FP_MEMO:
            _SOLVING_FP_MEMO[key] = real(files, root)
        return _SOLVING_FP_MEMO[key]
    monkeypatch.setattr(SA, "_fingerprint_of_files", memo)
    yield


try:
    from _maintainer_hooks import (                             # noqa: F401
        pytest_collection_modifyitems, pytest_configure, pytest_report_header,
        pytest_runtest_logreport, pytest_sessionfinish, pytest_terminal_summary)
except ModuleNotFoundError as _e:                               # pragma: no cover
    if _e.name != "_maintainer_hooks":
        raise
