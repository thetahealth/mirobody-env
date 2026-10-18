"""The evaluation run's context object, the write lock and the logger.

Split out of `haenv/evaluate.py`; `evaluate` re-exports every name defined here.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading as _threading
from pathlib import Path


log = logging.getLogger("haenv.eval")


@dataclass
class RunContext:
    """Everything one evaluation run reads and writes beyond each call's own arguments.

    `run_eval` makes one per run (or takes the caller's, which has already assigned the
    probes) and passes it down to the row builders as `ctx`; a row builder binds it to the
    cell's solver as `solver.run_ctx`. Nothing here is module-level, so two runs in one process cannot see each
    other's tables.
    """
    resp_path: Path | None = None    # this batch's responses.jsonl
    probes: dict | None = None       # probe registry; None falls back to the module constant
    allow_retired: bool = False      # a retired probe may be used (reproducing history only; --allow-retired)
    replay: bool = False             # a resumed run replays saved slice answers
    replay_idx: dict | None = None   # (case, solver, slice_t) -> first reusable saved row
    workers: int = 1                 # cells in flight; 1 = serial (set by the CLI from --workers / config)
    # Q-side probes, assigned per case before the run (`qside.assign_*`)
    premise_for: dict = field(default_factory=dict)        # case -> false/true premise
    noop_for: dict = field(default_factory=dict)           # case -> no-op probe
    quant_for: dict = field(default_factory=dict)          # case -> computable question
    #: per quant kind `{n, n_distinct, top, top_share, dist}`; `top_share` near 1.0 means
    #: guessing the majority answer scores well
    quant_truth_dist: dict = field(default_factory=dict)
    # read only by the oracle stubs; never rendered into a question
    oracle_gold_tests: dict = field(default_factory=dict)  # case -> gold test list
    oracle_disc_names: dict = field(default_factory=dict)  # case -> discriminator names
    oracle_diagnosis: dict = field(default_factory=dict)   # case -> gold diagnosis name
    oracle_warranted: dict = field(default_factory=dict)   # case -> clinician_action_warranted
    # per-cell accumulators, taken (popped) when the cell's row is written
    usage_acc: dict = field(default_factory=dict)          # (case, solver) -> token usage
    retry_acc: dict = field(default_factory=dict)          # (case, solver) -> retry attribution
    replay_n: dict = field(default_factory=dict)           # (case, solver) -> slices reused
    replay_fresh: dict = field(default_factory=dict)       # (case, solver) -> answers bought during a replay


def run_ctx_of(solver) -> RunContext:
    """The run a solver was bound to by `run_eval`; an unbound solver (a test building one
    directly) gets an empty context."""
    ctx = getattr(solver, "run_ctx", None)
    return ctx if isinstance(ctx, RunContext) else RunContext()


_WRITE_LOCK = _threading.Lock()   # lock for appends: two lines interleaving = one line of broken JSON
