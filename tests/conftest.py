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

pytest_plugins = ["_artifact_cache"]

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ROOT = _HERE.parent

# In-repo judge groups (`plugins/*.dist-info`, e.g. `haenv.m2`) are discovered through
# `importlib.metadata`, which only scans `sys.path`: a shipped job that declares one
# (`inputs/m2-core.job.yaml`) must load in the test session as it does from the command line.
if str(ROOT / "plugins") not in sys.path:
    sys.path.insert(0, str(ROOT / "plugins"))

# Subprocesses that run a script under tools/ must import this tree's `haenv`. A worktree whose
# virtualenv holds an editable install of another checkout would otherwise test that one.
import os as _os
# Child commands must never synchronise the shared interpreter.
_os.environ["UV_NO_SYNC"] = "1"
_os.environ["UV_FROZEN"] = "1"
# xdist owns the CPU parallelism; numerical libraries must not start another
# team in each of its 64 processes (or in their classifier children).
for _thread_limit in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ[_thread_limit] = "1"
_os.environ["PYTHONPATH"] = _os.pathsep.join(
    [str(ROOT / "src"), str(ROOT), str(ROOT / "plugins"),
     *(p for p in _os.environ.get("PYTHONPATH", "").split(_os.pathsep)
       if p and p not in (str(ROOT), str(ROOT / "plugins")))])


def _registry_targets():
    import importlib
    import haenv.judges as J
    import haenv.scoring.mounts.table as MT
    import haenv.contracts.external_gold as EG
    import haenv.app.job as JOB
    import haenv.world.generation.world_plugins as WP

    # The tables have three different types: `JUDGES` is `list[Judge]`,
    # `MOUNT`/`WHY_NOT` are `dict`, and `SUBJECTS` is a custom `_Subjects`
    # (no `clear`/`update`, not a `MutableMapping`). Each is restored by type.
    targets = [(J, "JUDGES"), (MT, "MOUNT"), (MT, "WHY_NOT")]
    targets += [(EG, name) for name in
                ("BLOCKS", "LATENT_CLASSES", "GATES", "FRAMINGS", "PROBES")]
    targets += [(JOB, "LATENT_REGISTRY"), (JOB, "_LOADED_GROUPS")]
    targets += [(WP, name) for name in
                ("_OVERLAY", "_SOURCES", "_POST_INJECTORS", "_LOADED")]
    import haenv.app.pack_hooks as PH
    targets += [(PH, "HOOKS"), (PH, "PACK_EXTENSIONS")]
    # A wrapper and its installation flag form one state. Restoring only the callable
    # leaves install() believing the removed wrapper is still present.
    wrapped = {
        'haenv.world.generation.build': ("observe_labs", "_world_prompt_premise", "_build_case_inner"),
        'haenv.world.history.history_dependency': ("install",),
        'haenv.interaction.qside': ("render_for",),
        'haenv.app.evaluate': ("_row_single", "_row_gated", "build_solvers", "gated_menu"),
        'haenv.scoring.rows': ("gated_menu",),
        'haenv.interaction.gated': ("menu_for", "synth_menu_target", "synth_on_demand", "run_gated"),
    }
    for name, attrs in wrapped.items():
        mod = importlib.import_module(name)
        targets += [(mod, attr) for attr in attrs if hasattr(mod, attr)]
    # Generation wrappers keep their installation sets in closures. Restoring
    # the callable alone would leave the next test unable to install its hook.
    import inspect
    install = importlib.import_module('haenv.world.history.history_dependency').install
    while callable(install):
        state = inspect.getclosurevars(install).nonlocals.get("installed")
        if isinstance(state, set):
            targets.append((install, "_closure:installed"))
        install = getattr(install, "__wrapped__", None)
    for name in ("haenv.p2.hooks", "haenv.p3.hooks", "haenv.p4.hooks", "haenv.m2.hooks",
                 'haenv.app.packs', 'haenv.interaction.query'):
        targets.append((importlib.import_module(name), "_INSTALLED"))
    targets += [(importlib.import_module('haenv.app.packs'), "PACKS"),
                (importlib.import_module('haenv.interaction.query'), "_CFG")]
    # Plugins change the world stamp within a test, but may not survive into the next one.
    # Nested overlay/probe values need copies too; shallow snapshots retain in-place edits.
    nested = {(EG, "PROBES"), (WP, "_OVERLAY")}
    return targets, nested


def _registry_snapshot():
    from copy import deepcopy
    targets, nested = _registry_targets()
    snap = {}
    for mod, name in targets:
        if name.startswith("_closure:"):
            import inspect
            cur = inspect.getclosurevars(mod).nonlocals[name.split(":", 1)[1]]
        else:
            cur = getattr(mod, name, None)
        snap[mod, name] = (list(cur) if isinstance(cur, list)
                           else deepcopy(cur) if (mod, name) in nested
                           else dict(cur) if isinstance(cur, dict)
                           else set(cur) if isinstance(cur, set)
                           else cur if callable(cur) else None)
    return snap


def _registry_restore(snap):
    for (mod, name), old in snap.items():
        if name.startswith("_closure:"):
            import inspect
            cur = inspect.getclosurevars(mod).nonlocals[name.split(":", 1)[1]]
        else:
            cur = getattr(mod, name, None)
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
        elif callable(old):
            setattr(mod, name, old)
    # `SUBJECTS` has its own unregistration entry point: the test that
    # registers an external subject removes it with `unregister_subject`.


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
    snap = _registry_snapshot()
    try:
        yield
    finally:
        _registry_restore(snap)


@pytest.fixture(scope="module", autouse=True)
def _isolate_registries_per_module():
    """The same restore around a whole test module.

    A module-scoped fixture that calls `load_job()` is set up before the
    per-test snapshot above is taken, so its plugin registrations would
    otherwise outlive the module and reach whatever test the worker runs next.
    """
    snap = _registry_snapshot()
    try:
        yield
    finally:
        _registry_restore(snap)


@pytest.fixture(autouse=True)
def _isolate_config_set():
    """`haenv ... --set` / `--pooling` run in-process writes `$HAENV_CONFIG_SET`; restore it."""
    import os
    from haenv.app.settings import SET_ENV
    previous = os.environ.get(SET_ENV)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SET_ENV, None)
        else:
            os.environ[SET_ENV] = previous


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
    import haenv.provenance.anchor as A
    import haenv.runtime.solver_accounting as SA
    real = SA._fingerprint_of_files

    def memo(files, root=None):
        key = (str(A.ROOT), tuple(files), root)
        if key not in _SOLVING_FP_MEMO:
            _SOLVING_FP_MEMO[key] = real(files, root)
        return _SOLVING_FP_MEMO[key]
    monkeypatch.setattr(SA, "_fingerprint_of_files", memo)
    yield


_maintainer_modifyitems = None
try:
    from _maintainer_hooks import (                             # noqa: F401
        pytest_configure, pytest_report_header,
        pytest_runtest_logreport, pytest_sessionfinish, pytest_terminal_summary)
    from _maintainer_hooks import pytest_collection_modifyitems as _maintainer_modifyitems
except ModuleNotFoundError as _e:                               # pragma: no cover
    if _e.name != "_maintainer_hooks":
        raise


# ---- speed tiers: `pytest --fast` is the inner loop; the default tier still runs everything

#: Fixtures built once per session and shared by every worker.
SHARED_BUILD_FIXTURES = frozenset({"offline_batch", "slice_batch", "serial_batch", "artifact_cache"})


def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true",
                     help="deselect `integration` tests (measured slow, or using a shared "
                          "offline build); for the edit-test loop, not for sign-off")


def _slow_ids() -> frozenset:
    import json
    path = _HERE / "fixtures/harness/_speed_tiers.json"
    return frozenset(json.loads(path.read_text(encoding="utf-8"))["slow"]) if path.is_file() else frozenset()


def _runtime_costs() -> dict:
    import json
    path = _HERE / "fixtures/harness/_runtime_costs.json"
    return json.loads(path.read_text(encoding="utf-8"))["seconds"] if path.is_file() else {}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Maintainer tiering first (by file), then the speed tier. Both mark before `-m`
    filters, so `-m "not integration"` selects the same tests as `--fast`."""
    if _maintainer_modifyitems is not None:
        _maintainer_modifyitems(config, items)
    slow = _slow_ids()
    for it in items:
        if it.nodeid in slow or SHARED_BUILD_FIXTURES & set(getattr(it, "fixturenames", ())):
            it.add_marker(pytest.mark.integration)
    if config.getoption("--fast"):
        keep = [it for it in items if it.get_closest_marker("integration") is None]
        config.hook.pytest_deselected(items=[it for it in items if it not in keep])
        items[:] = keep
    # Start independent long contracts first. Small xdist queues leave room for
    # short parameters while those run; this changes no selection or assertion.
    costs = _runtime_costs()
    items.sort(key=lambda item: -costs.get(item.nodeid.split("[", 1)[0], 0.0))
