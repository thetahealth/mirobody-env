"""Maintainer import for the packaged batch reader."""

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
from haenv.reporting.tools.packread import Batch
from haenv.reporting.tools.packread import Scan
from haenv.reporting.tools.packread import PackReadError
from haenv.reporting.tools.packread import batches
from haenv.reporting.tools.packread import cases
from haenv.reporting.tools.packread import eval_rows
from haenv.reporting.tools.packread import field
from haenv.reporting.tools.packread import is_complete
from haenv.reporting.tools.packread import latest
from haenv.reporting.tools.packread import meta
from haenv.reporting.tools.packread import scan  # noqa: F401
