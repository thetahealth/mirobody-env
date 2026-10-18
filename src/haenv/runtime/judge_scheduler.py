"""Bounded asynchronous orchestration for independent synchronous judge cells."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

#: Most cells one run judges at once. Replayed votes cost nothing, so a resumed or re-sealed
#: run is bounded by local work, not by the provider. `semantic_parallel` re-exports it.
MAX_JUDGE_LANES = 64


class RoutesUnavailable(RuntimeError):
    """No untried, enabled route can serve this request."""


class RouteLanes:
    """Shared request queue admission, with independent route caps and backoff.

    Cell workers use the existing bounded scheduler. Waiting for a cooling route holds
    no route lane. A rejected request can be offered to another route under a new ID.
    """
    def __init__(self, routes):
        self.routes = routes
        self.active = dict.fromkeys(routes, 0)
        self.failures = dict.fromkeys(routes, 0)
        self.ready_at = dict.fromkeys(routes, 0.0)
        self.disabled = set()
        self.condition = threading.Condition()
        self.cursor = 0

    def acquire(self, *, exclude=()):
        names = list(self.routes)
        with self.condition:
            while True:
                enabled = [n for n in names if n not in self.disabled and n not in exclude]
                if not enabled:
                    raise RoutesUnavailable("All eligible judge routes are unavailable")
                now = time.monotonic()
                for offset in range(len(names)):
                    idx = (self.cursor + offset) % len(names)
                    name = names[idx]
                    if (name in enabled and self.active[name] < self.routes[name].concurrency
                            and self.ready_at[name] <= now):
                        self.active[name] += 1
                        self.cursor = (idx + 1) % len(names)
                        return name
                delays = [self.ready_at[n] - now for n in enabled
                          if self.active[n] < self.routes[n].concurrency and self.ready_at[n] > now]
                self.condition.wait(timeout=min(delays) if delays else None)

    def release(self, name, *, failed=False, disable=False):
        with self.condition:
            self.active[name] -= 1
            if disable:
                self.disabled.add(name)
            if failed:
                self.failures[name] += 1
                route = self.routes[name]
                delay = min(route.max_backoff_s,
                            route.failure_backoff_s * 2 ** min(self.failures[name] - 1, 20))
                self.ready_at[name] = time.monotonic() + delay
            else:
                self.failures[name] = 0
            self.condition.notify_all()


def run_bounded(items: list, work, *, concurrency: int, stop: threading.Event | None = None) -> list:
    """At most `concurrency` workers. Never start queued work after a stop.

    Every in-flight worker finishes and persists its own state before returning.
    Output order follows input order; a never-started slot is explicitly None.
    Failures drain in-flight requests before propagating to avoid orphan charges.
    """
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_JUDGE_LANES:
        raise ValueError("Concurrency must be an integer from 1 through %d" % MAX_JUDGE_LANES)
    stop = threading.Event() if stop is None else stop

    async def run():
        semaphore = asyncio.Semaphore(concurrency)
        # asyncio.to_thread would share the loop's default pool (min(32, cpus + 4) workers),
        # which silently caps every lane count above it.
        pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="judge-lane")
        loop = asyncio.get_running_loop()

        async def one(item):
            async with semaphore:
                if stop.is_set():
                    return None
                try:
                    return await loop.run_in_executor(pool, work, item)
                except Exception:
                    stop.set()
                    raise
        try:
            results = await asyncio.gather(*(one(item) for item in items), return_exceptions=True)
        finally:
            pool.shutdown(wait=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    return asyncio.run(run())
