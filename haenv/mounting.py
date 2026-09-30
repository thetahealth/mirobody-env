"""mounting.py -- declares which judges each row builder in `evaluate.py` calls
directly, per execution geometry.

Judges that `run_judges` dispatches are declared in `mount_table.MOUNT`, not here;
this table covers only the probes a row builder orchestrates itself, and the
maintainers' tests check it against the calls in `evaluate.py`.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The four execution geometries, taken from `mount_table`.
from .mount_table import GEOMETRIES  # noqa: E402,F401


@dataclass(frozen=True)
class Mount:
    """Which judges are mounted on one geometry.

    * `geometry`  -- one of the four geometries.
    * `row_fn`    -- the row builder that implements it.
    * `judges`    -- the judge function names the builder calls directly.
    * `profile`   -- which judge table `run_judges` uses (empty = not invoked).
    * `why`       -- why these.
    """
    geometry: str
    row_fn: str
    judges: tuple[str, ...]
    profile: str = ""
    why: str = ""


MOUNTS: tuple[Mount, ...] = (
    Mount("single", "_row_single",
          ("judge_noop_probe", "judge_quant_probe", "run_process_judges"),
          profile="SINGLE",
          why="Single-shot geometry. `run_process_judges` also runs on `gated` "
              "(judging the final answer), `slices` (judging the last slice), and "
              "`multi` (judging the last round) -- all four geometries can render "
              "a question that requires a trace. This table is the declaration "
              "surface for who runs what."),
    Mount("gated", "_row_gated",
          ("judge_noop_probe", "judge_quant_probe", "judge_abstention_calibration",
           "run_process_judges"),
          profile="SINGLE",
          why="On-demand-query geometry. One extra dimension, abstention calibration (it serves "
              "the tier where 'say so when data is insufficient' matters). "
              "`run_process_judges` judges the final answer (`out`, the same subject as "
              "this geometry's `run_judges(SINGLE, out, vp)`): this geometry sets "
              "`solver.prompt_mode = \"ddx\"` => renders `DDX_PROMPT` => it also requires a "
              "trace. This geometry's ledger carries no lab results (by design), so the "
              "'cited unrevealed evidence' judge's coverage is narrower than in slices -- that "
              "is a property of the question face, not a defect."),
    # Judges dispatched by `run_judges` through `mount_table.MOUNT` are not listed here.
    Mount("slices", "_row_slices",
          ("judge_noop_probe", "judge_quant_probe", "run_process_judges"),
          profile="SLICES",
          why="Multi-slice geometry. The tier with the most judges. `run_process_judges` "
              "judges the last slice, following the same convention as this geometry's "
              "`noop` / `quant` / `_premise_row` / the kernel's five tracks: an early slice "
              "has incomplete information, and judging it by the standard for T would be a "
              "smaller version of the same mistake. This table only covers the handful of "
              "probes explicitly orchestrated inside the builder; which judges are "
              "mounted on this geometry through `run_judges` is decided solely by "
              "`mount_table.MOUNT` (the cell value is the subject: `rows` judges across "
              "slices / `last` judges the final answer of the last slice)."),
    Mount("multi", "_row_multi",
          ("judge_noop_probe", "judge_quant_probe", "run_process_judges"),
          profile="MULTI",
          why="Multi-round disease-course loop. `_row_multi` goes through `run_judges` "
              "(`profile=\"MULTI\"` is the declaration surface for this fact); "
              "`multiround_revision` / `premise_repair` are dispatched separately, through "
              "`mount_table.MOUNT`, because they need a trajectory-type subject that only "
              "`RoundRecorder` (below) can provide. This tuple holds the probes explicitly "
              "orchestrated inside the builder: noop / quant / `run_process_judges`, all "
              "judging the last round's answer -- the round with the most complete "
              "information, following the same convention as judging the last slice in the "
              "`slices` geometry."),
)

BY_GEOMETRY: dict[str, Mount] = {m.geometry: m for m in MOUNTS}


class RoundRecorder:
    """Wraps a solver and records every round's output, giving the `multi` geometry
    a `last` subject for `run_judges` (`ctx["round_outputs"]`).

    Contract: `solve()` returns the inner solver's output unchanged; `prompt_mode`
    and every other attribute are forwarded to the inner solver.
    """

    __slots__ = ("inner", "outputs", "facts", "_on_round")

    def __init__(self, inner, on_round=None):
        """`on_round(i, out, facts)` is called after each round (`i` starts at 1); the
        caller uses it to persist the raw response.
        """
        self.inner, self.outputs, self.facts = inner, [], []
        self._on_round = on_round

    @property
    def prompt_mode(self):
        return getattr(self.inner, "prompt_mode", "default")

    @prompt_mode.setter
    def prompt_mode(self, v):
        self.inner.prompt_mode = v

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def solve(self, payload):
        out = self.inner.solve(payload)
        self.outputs.append(out)
        # The leak probe is not repeated here: `runner.run_multiround` probes each round itself.
        from .solve_guard import facts_of_output
        f = facts_of_output(out)
        self.facts.append(f)
        if self._on_round is not None:
            self._on_round(len(self.outputs), out, f)
        return out

    @property
    def bad_rounds(self) -> list[dict]:
        return [{"round": i + 1,
                 "raw_empty": f.raw_empty, "raw_unparseable": f.raw_unparseable}
                for i, f in enumerate(self.facts)
                if f.raw_empty or f.raw_unparseable]

    def abort_reason(self) -> str | None:
        """The cell's disposition; `None` if every round was usable.

        Any bad round aborts the whole cell (unlike slices): later rounds are computed
        against the bad round's answer. Zero recorded rounds => `ABORT(no_response)`.
        """
        if not self.facts:
            return "ABORT(no_response)"
        if any(f.raw_empty for f in self.facts):
            return "ABORT(no_response)"
        if any(f.raw_unparseable for f in self.facts):
            return "ABORT(unparseable)"
        return None


def judges_for(geometry: str) -> tuple[str, ...]:
    """The judges declared for one geometry. An unknown geometry raises."""
    m = BY_GEOMETRY.get(geometry)
    if m is None:
        raise KeyError(
            f"unknown geometry {geometry!r}; only {list(GEOMETRIES)}. "
            f"Does not fall back to an empty set, so \"no judges\" and \"misspelled name\" stay distinguishable.")
    return m.judges


def geometry_of(job) -> str:
    """Which geometry a job runs under, in `evaluate._one`'s dispatch order: `gated` ->
    `slices` -> `multiround` -> `single`. The slice count is not known here, so
    actual dispatch is still decided by `_one`.
    """
    from .mount_table import claim_geometry
    return claim_geometry(job, None)
