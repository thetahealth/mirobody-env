"""Resume policy: which cells of a batch count as done, and which dispositions a rerun may change.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import json
from pathlib import Path
from .run_state import log


#: Dispositions that hold no judgeable answer. Which of them a resumed run resends depends on who
#: caused the abort (`resend_cause`); `ABORT(leak)` and `ABORT(iron_law)` are deterministic verdicts
#: and always count as done.
RETRYABLE_OVERALL = frozenset({"ABORT(no_response)", "ABORT(unparseable)",
                               "ABORT(no_answer)", "ERROR"})


TERMINAL_OVERALL = frozenset({"ABORT(leak)", "ABORT(iron_law)"})

#: Failure classes of `tools/attribute_failures.py` that originate on our side (provider outage,
#: network, rate limit, key pool). A cell whose final failed attempt is one of them is resent.
#: A reply the model produced (empty body, broken JSON, truncated at `max_tokens`) and an exhausted
#: tool budget are the model's result under the run's settings: a rerun would repeat them, so
#: they stay done.
OUR_SIDE_CLASSES = frozenset({"infra", "auth"})


def _base(overall) -> str:
    o = str(overall or "")
    if o.startswith("ABORT(") and ")" in o:
        o = "ABORT(" + o[len("ABORT("):o.index(")")].split(":")[0].strip() + ")"
    return o


def resend_cause(row) -> str | None:
    """Why a resumed run resends this row's cell, or `None` when the row stands as the result.

    `ERROR` is a harness exception (ours). `ABORT(no_response)` and `ABORT(unparseable)` are resent
    when the final failed attempt recorded in `failed_attempts` is classed as ours; without a
    recorded attempt the cause cannot be told and the row stands as the model's result.
    `ABORT(no_answer)` (tool budget spent before an answer) is the model's result.
    """
    o = _base(row.get("overall") if isinstance(row, dict) else row)
    if o == "ERROR":
        return "ERROR"
    if o not in ("ABORT(no_response)", "ABORT(unparseable)") or not isinstance(row, dict):
        return None
    att = row.get("failed_attempts")
    last = (att if isinstance(att, list) else [att])[-1] if att else None
    if last is None:
        return None
    from .run_ledger import retry_classifier
    m = retry_classifier()
    err = last.get("error") if isinstance(last, dict) else last
    cls = m.classify(str(err or ""))
    return f"{o} / {cls}" if cls in OUR_SIDE_CLASSES else None


def is_retryable(row) -> bool:
    """Whether a resumed run should resend this cell. Takes the eval row; a bare disposition string
    carries no cause, so only `ERROR` is resent. Unrecognized dispositions count as done, so a
    misspelled name never re-runs (and re-bills) a batch.
    """
    return resend_cause(row) is not None


def _done_keys(path: Path) -> set[str]:
    """The set of cells already run. Cells aborted by something on our side (`resend_cause`) are not
    counted, so a resumed run fills them in; every other row, including an abort the model caused,
    counts as done.
    """
    if not path.exists():
        return set()
    keys: set[str] = set()
    resent: dict[str, int] = {}
    kept: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            r = json.loads(line)
            cause = resend_cause(r)
            if cause:
                resent[cause] = resent.get(cause, 0) + 1
                continue
            if _base(r.get("overall")) in RETRYABLE_OVERALL:
                _ov = str(r.get("overall"))
                kept[_ov] = kept.get(_ov, 0) + 1
            keys.add(f"{r['case']}|{r['solver']}")
        except Exception:
            pass
    if resent:
        log.warning("[eval] %d cell(s) not counted as done, rerunning: %s",
                    sum(resent.values()), " · ".join(f"{k} {v}" for k, v in sorted(resent.items())))
    if kept:
        log.warning("[eval] %d aborted cell(s) kept as the model's result, not resent: %s",
                    sum(kept.values()), " · ".join(f"{k} {v}" for k, v in sorted(kept.items())))
    return keys
