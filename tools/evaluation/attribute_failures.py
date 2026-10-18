"""CLI and explicit compatibility exports for runtime retry attribution."""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)


from haenv.runtime.retry_classifier import (  # noqa: F401
    BLAME,
    NOT_MODEL,
    BUDGET_CHARS_PER_TOKEN_MIN,
    NO_ANSWER,
    _RULES,
    _REASON_CHARS,
    ROOT,
    _cfg,
    classify,
    _report_classes,
    _attempts,
    scan,
    _budgets_of,
    frozen_batches,
    all_batches,
    selftest,
    render,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main(_tool_sys.argv[1:]))
