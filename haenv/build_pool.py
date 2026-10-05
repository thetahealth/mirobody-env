"""Deterministic case generation, and the per-item stages after it, on forked worker processes.

Each case's world is seeded by its own `case_id`, so cases build independently. Results come
back in input order, and the parent re-registers each case's Q-side injection ledger (`wq`),
the one piece of per-case module state that later stages read, so the batch is byte-identical
to a serial build. `map_ordered` runs any pure per-item function the same way: the fork
inherits the parent's state at the call, and results return in input order.
"""
from __future__ import annotations

import multiprocessing as _mp
import os

_CTX: dict = {}


def default_workers() -> int:
    """`min(16, usable cores)`."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:                                   # not Linux
        n = os.cpu_count() or 1
    return max(1, min(16, n))


class _ChildError(Exception):
    """A case's exception carried back from a worker, keeping its type name and message."""


def _child_error(name: str, msg: str) -> Exception:
    return type(name, (_ChildError,), {})(msg)


def _build_one(i: int):
    from . import wq
    cs = _CTX["cases"][i]
    before = dict(wq._QINJ)
    try:
        ra, err = _CTX["build"](cs), None
    except Exception as e:  # one failed case must not sink the batch
        ra, err = None, (type(e).__name__, str(e))
    # register_injection stores a fresh dict, so identity marks this case's registrations.
    inj = [(k, v) for k, v in wq._QINJ.items() if before.get(k) is not v]
    return ra, err, inj


def build_all(cases: list, build, workers: int) -> list[tuple]:
    """`[(case, (raw, audit) | None, exception | None)]` in input order. `build(case)` runs in
    a forked child; the fork inherits the loaded job, plugins and registries."""
    from . import wq
    _CTX.update(cases=list(cases), build=build)
    try:
        with _mp.get_context("fork").Pool(processes=workers) as pool:
            out = list(pool.imap(_build_one, range(len(cases)), chunksize=1))
    finally:
        _CTX.clear()
    results = []
    for cs, (ra, err, inj) in zip(cases, out):
        for k, v in inj:
            wq.register_injection(k, v)
        results.append((cs, ra, None if err is None else _child_error(*err)))
    return results


def _map_one(i: int):
    return _CTX["map_fn"](_CTX["map_items"][i])


def map_ordered(fn, items, workers: int, chunksize: int = 1) -> list:
    """`[fn(x) for x in items]`, on `workers` forked processes when `workers > 1`.

    `fn` runs in a child forked at this call, so it sees the parent's module state as it is
    now; any state it changes stays in the child. Results come back in input order. An
    exception in `fn` is raised in the parent.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    _CTX.update(map_fn=fn, map_items=items)
    try:
        with _mp.get_context("fork").Pool(processes=min(workers, len(items))) as pool:
            return list(pool.imap(_map_one, range(len(items)), chunksize=chunksize))
    finally:
        _CTX.pop("map_fn", None)
        _CTX.pop("map_items", None)
