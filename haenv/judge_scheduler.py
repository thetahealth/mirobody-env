"""Bounded asynchronous orchestration for independent synchronous judge cells."""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

#: Most cells one run judges at once. Replayed votes cost nothing, so a resumed or re-sealed
#: run is bounded by local work, not by the provider. `semantic_parallel` re-exports it.
MAX_JUDGE_LANES = 64


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
