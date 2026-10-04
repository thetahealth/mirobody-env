"""Resume policy: which cells of a batch count as done, and which dispositions a rerun may change.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import json
from pathlib import Path
from .run_state import log


#: Resume policy: a disposition is retryable if a rerun could produce a different result
#: (`ABORT(no_response)`, `ABORT(unparseable)`, `ABORT(no_answer)`, `ERROR`); `ABORT(leak)`
#: and `ABORT(iron_law)` are deterministic verdicts and count as done.
RETRYABLE_OVERALL = frozenset({"ABORT(no_response)", "ABORT(unparseable)",
                               "ABORT(no_answer)", "ERROR"})


TERMINAL_OVERALL = frozenset({"ABORT(leak)", "ABORT(iron_law)"})


def is_retryable(overall) -> bool:
    """Whether this cell should be resent on a resumed run. Unrecognized dispositions count as
    done, so a misspelled name never re-runs (and re-bills) a batch.
    """
    o = str(overall or "")
    if not o:
        return False
    if o.startswith("ABORT(") and ")" in o:
        o = "ABORT(" + o[len("ABORT("):o.index(")")].split(":")[0].strip() + ")"
    return o in RETRYABLE_OVERALL


def _done_keys(path: Path) -> set[str]:
    """The set of cells already run. Retryable dispositions (e.g. transport failures) are not
    counted, so a resumed run fills them in; `ABORT(leak)` is a verdict and counts as done.
    """
    if not path.exists():
        return set()
    keys: set[str] = set()
    retryable: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            r = json.loads(line)
            _ov = str(r.get("overall") or "")
            if is_retryable(_ov):
                retryable[_ov] = retryable.get(_ov, 0) + 1
                continue
            keys.add(f"{r['case']}|{r['solver']}")
        except Exception:
            pass
    if retryable:
        log.warning("[eval] %d cell(s) not counted as done, rerunning: %s",
                    sum(retryable.values()),
                    " · ".join(f"{k} {v}" for k, v in sorted(retryable.items())))
    return keys
