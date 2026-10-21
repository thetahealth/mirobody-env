"""solve_guard.py -- the single path for handing a payload to the solver.

Every delivery path runs `probe -> (precheck) -> solve -> persist -> verify`
through `guarded_solve`, unless it is listed in `SOLVE_EXEMPT` with a reason.
The module reports facts (`SolveFacts`); each geometry decides the disposition
(e.g. single-shot aborts the whole cell on a leak, slices void only that slice).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from haenv_kernel.build import leakage_probe                      # kernel
from haenv_kernel.schema import SolverPayload
from haenv_kernel.solver import _extract_json

log = logging.getLogger("haenv")


def _probe_snapshot(payload):
    """Freeze the entire probe view, including dynamic attributes.

    Only plain JSON data has a cache identity. Custom objects and subclasses
    may have equality or access behavior that their attributes cannot capture;
    leave those to the probe. Preserve types and dict order conservatively.
    """
    def freeze(value):
        kind = type(value)
        if kind in (str, int, bool, type(None)):
            return kind, value
        if kind is float:
            return kind, value.hex()
        if kind is list:
            return kind, tuple(freeze(item) for item in value)
        if kind is dict:
            if any(type(key) is not str for key in value):
                raise TypeError("non-string JSON key")
            return kind, tuple((key, freeze(item)) for key, item in value.items())
        raise TypeError("non-JSON value")

    if type(payload) is not SolverPayload:
        return None
    try:
        return freeze(payload.__dict__)
    except (TypeError, RuntimeError):  # Includes recursion or a changing dict.
        return None


class PreparedPayload:
    """A private snapshot and its leakage verdict, reusable within one run.

    Delivery uses a fresh copy. A changed payload or time is probed again, so
    neither a plugin edit nor a different window can inherit a clean verdict.
    """

    def __init__(self, payload, T: int):
        self._payload = deepcopy(payload)
        self._T = int(T)
        ok, violations = leakage_probe(self._payload, self._T)
        self._verdict = ok, tuple(violations)
        self._snapshot = _probe_snapshot(self._payload)

    def copy(self):
        return deepcopy(self._payload)

    def probe(self, payload, T: int):
        if (int(T) == self._T and self._snapshot is not None
                and _probe_snapshot(payload) == self._snapshot):
            ok, violations = self._verdict
            return ok, list(violations)
        return leakage_probe(payload, int(T))

#: `.solve(` call sites that bypass the skeleton, with the reason for each.
SOLVE_EXEMPT: dict[str, str] = {
    "baselines": "Offline constant baseline (`ConstantDdxSolver`), the payload is built fresh "
                 "by the caller, not sourced from `build_instance`, and it never issues a "
                 "network request -- the source of the constant upper bound; the leakage gate "
                 "has no object to check here.",
    "verify.py::_conclusion": "The verifier's offline reference solve that checks gold is "
                              "unchanged by injection; runs inside the emission gate, before "
                              "the case becomes a question.",
    # Accepted key forms: `module`, `file.py::function`, `file.py:line`.
    "mounting.py::solve": "`RoundRecorder.solve` -- a transparent forwarding recorder, not "
                       "a call site: it hands `payload` unchanged to `self.inner.solve`, only "
                       "additionally stores one round's answer, judges facts per round (through "
                       "this module's `facts_of_output`, no second copy written) and leaves "
                       "persistence to the caller's `on_round` callback. "
                       "Isolation and the leakage gate are done by the caller before "
                       "delivery (the multi-round geometry is `evaluate._row_multi` -> kernel "
                       "`runner.run_multiround`); the recorder never touches the payload, never "
                       "issues a request, and never changes the wrapped object's behavior "
                       "(`__getattr__` forwards everything). Wiring status of the multi "
                       "geometry is tracked by `mount_coverage.unwired_geometries()`, not here.",
}


@dataclass(frozen=True)
class SolveFacts:
    """What happened during one "deliver -> answer" step. Facts only, no disposition."""

    out: object | None
    #: Leaked items; `None` means clean. Non-empty => `out` must be `None`
    #: (no delivery, no answer).
    leak: list[str] | None
    #: Raw text present but empty. An absent attribute (offline baselines) means
    #: not applicable and must not count as "no response".
    raw_empty: bool
    #: Raw text non-empty but JSON could not be extracted; the kernel would
    #: otherwise turn this into an abstention.
    raw_unparseable: bool
    failed_attempts: list | None
    n_chars: int
    #: `precheck`'s return value (single-shot uses it to carry `wq.enforce`'s
    #: warning list); `None` if not supplied.
    precheck_result: object = None

    @property
    def usable(self) -> bool:
        """Whether this answer counts toward the denominator."""
        return (self.out is not None and self.leak is None
                and not self.raw_empty and not self.raw_unparseable)

    def abort_reason(self) -> str | None:
        """The `overall` value for whole-cell disposition; `None` if usable. Not used on the slice side (it records facts per slice)."""
        if self.leak:
            return "ABORT(leak)"
        if self.out is None or self.raw_empty:
            return "ABORT(no_response)"
        if self.raw_unparseable:
            return "ABORT(unparseable)"
        return None


def guarded_solve(solver, payload, T: int, *, prompt_mode: str | None = None,
                  on_output: Callable[[object], None] | None = None,
                  precheck: Callable[[], object] | None = None,
                  prepared: PreparedPayload | None = None,
                  tag: str = "") -> SolveFacts:
    """The only path that hands a payload to the solver: probe -> (precheck) -> solve -> persist -> verify raw text.

    `prompt_mode`: phrasing set by the question-generation side; `None` means the
    caller already set it. `on_output`: persistence callback (persistence shape
    belongs to the geometry). `precheck`: runs after the probe and before delivery;
    raise to abort. `tag`: log label only.

    The leakage probe always runs first, so a cell that both leaks and fails
    `precheck` reads `ABORT(leak)`.
    """
    ok, viol = (prepared.probe(payload, int(T)) if prepared is not None
                else leakage_probe(payload, int(T)))
    if not ok:
        log.error("[guard] %s leak -> not delivered: %s", tag or "?", viol[:3])
        return SolveFacts(out=None, leak=list(viol), raw_empty=False,
                          raw_unparseable=False, failed_attempts=None, n_chars=0)

    pre = precheck() if precheck is not None else None

    if prompt_mode is not None:
        solver.prompt_mode = prompt_mode
    out = solver.solve(payload)
    if on_output is not None:
        on_output(out)

    f = facts_of_output(out)
    return SolveFacts(
        out=f.out, leak=None, raw_empty=f.raw_empty,
        raw_unparseable=f.raw_unparseable, failed_attempts=f.failed_attempts,
        n_chars=f.n_chars, precheck_result=pre,
    )


def facts_of_output(out) -> SolveFacts:
    """Judge facts from `out` after an answer has been obtained; no leakage probe, no requests.

    Used where the leakage gate runs elsewhere (multi-round: inside the kernel's
    `runner.run_multiround`). Shares the empty/unparseable judgment with
    `guarded_solve`, so a canned abstention on a bad round is never counted as an
    answer.
    """
    txt = getattr(out, "_raw_text", None)
    empty = txt is not None and not str(txt).strip()
    return SolveFacts(
        out=out, leak=None, raw_empty=empty,
        raw_unparseable=(False if empty else _unextractable(txt)),
        failed_attempts=(getattr(out, "_failed_attempts", None) or None),
        n_chars=len(str(txt)) if txt is not None else 0,
    )


def _unextractable(txt) -> bool:
    """Delegate to `evaluate.json_unextractable`, which uses the kernel's own `_extract_json`."""
    return json_unextractable(txt)


def json_unextractable(raw_text) -> bool:
    """True if the raw text is non-empty but no JSON object can be extracted from it.

    The kernel's `solver._extract_json` falls back to `{}` in that case, and `_to_output({})`
    would turn it into a canned abstention scored as the model's answer. This calls the same
    extractor (which strips `<think>` blocks) so it agrees with the scoring path.
    """
    if raw_text is None:
        return False                       # attribute absent = offline stub, not applicable (same convention as `raw_empty`)
    t = str(raw_text)
    if not t.strip():
        return False                       # genuinely empty -- goes through the existing `raw_empty` path, not judged here
    try:
        data = _extract_json(t)
    except Exception:                      # noqa: BLE001 -- parse error = same as `{}` once retries are exhausted
        return True
    return not (isinstance(data, dict) and data)
