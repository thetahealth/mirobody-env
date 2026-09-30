"""What `haenv build` / `verify` print to a console is English.

The README walks a reader through four commands and tells them which line to look
for. This pins the log calls, `print` calls and the relation-check details
(printed after `build`) in `haenv/` to contain no CJK characters, so every line
`build` and `verify` print on that path is English.

Scope: console text only. Case content (symptom and event text in the registry
tables) is a separate question and is not covered here.

Two sides:
* `test_scanner_catches_a_cjk_log_call` -- the scanner flags a synthetic Chinese
  log call and passes an English one, so a green run is not a scanner that
  finds nothing.
* `test_haenv_console_text_has_no_cjk` -- the scan over `haenv/`.

It turns red when someone adds a Chinese log message or relation detail.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
_CJK = re.compile(r"[一-鿿]")
_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
_LOGGERS = {"log", "logger", "_log", "LOG"}


def _console_calls(src: str) -> list[tuple[int, str]]:
    """(line, source) of every log call, `print` call and `Check(...)` whose
    source contains CJK. Comments are not part of a call's source segment."""
    hits = []
    for n in ast.walk(ast.parse(src)):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        is_log = (isinstance(f, ast.Attribute) and f.attr in _LOG_METHODS
                  and isinstance(f.value, ast.Name) and f.value.id in _LOGGERS)
        is_named = isinstance(f, ast.Name) and f.id in ("print", "Check")
        if not (is_log or is_named):
            continue
        seg = ast.get_source_segment(src, n) or ""
        if _CJK.search(seg):
            hits.append((n.lineno, seg.splitlines()[0][:100]))
    return hits


def test_scanner_catches_a_cjk_log_call():
    bad = 'import logging\nlog = logging.getLogger()\nlog.info("[x] %s 发射门拦下", 1)\n'
    good = 'import logging\nlog = logging.getLogger()\nlog.info("[x] %s blocked", 1)\n# 注释不算\n'
    assert _console_calls(bad), "the scanner missed a Chinese log call"
    assert not _console_calls(good), "the scanner flagged English text or a comment"
    assert _console_calls('Check("R8", None, "方差为零")\n'), "the scanner missed a relation detail"


def test_haenv_console_text_has_no_cjk():
    found = []
    for p in sorted((ROOT / "haenv").rglob("*.py")):
        for line, seg in _console_calls(p.read_text(encoding="utf-8")):
            found.append(f"{p.relative_to(ROOT)}:{line}: {seg}")
    assert not found, "console text with CJK characters:\n" + "\n".join(found)
