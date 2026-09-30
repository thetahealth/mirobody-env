"""Generate, evaluate and report on longitudinal health agents.

A job file supplies a raw case and its latent control variables; the pipeline
builds the world, runs the models, scores them and writes a report.

The L0 kernel lives in ``core/`` inside this repository.
"""
import os as _os
import sys as _sys
from pathlib import Path as _Path

__all__ = ["job", "build", "evaluate", "report", "cli", "kernel_path"]

# Opt-in scratch redirect: HAENV_SCRATCH_TMP sets TMPDIR once, process-wide, unless TMPDIR
# is already set. There is no default path, so importing the package creates no directories.
_SCRATCH_TMP = _os.environ.get("HAENV_SCRATCH_TMP", "").strip()
if _SCRATCH_TMP and not _os.environ.get("TMPDIR"):
    try:
        _p = _Path(_SCRATCH_TMP).expanduser()
        _p.mkdir(parents=True, exist_ok=True)
        # `tempfile` reads TMPDIR lazily; setting `tempfile.tempdir` would bypass `gettempdir()`.
        _os.environ["TMPDIR"] = str(_p)
    except OSError:
        pass


#: Kernel location in a wheel: `force-include` copies `core/` (with its LICENSE) to
#: `haenv/_kernel/`, modules directly under it. Absent in a source tree.
_WHEEL_KERNEL = _Path(__file__).resolve().parent / "_kernel"

#: Kernel location in a source tree; fallback when `config.yaml:kernel_path` cannot be read.
_DEFAULT_KERNEL_REL = "core"

#: Data resources in a wheel (`config.yaml`, `registry/`, `inputs/`, `docs/anchor/`,
#: `tools/make_freeze.py`, via `force-include`). Absent in a source tree.
_WHEEL_DATA = _Path(__file__).resolve().parent / "_data"


def data_root() -> _Path:
    """Where read-only resources live.

    Order: ``HAENV_DATA_ROOT`` (used even if it does not exist), then the source tree
    (detected by the presence of ``config.yaml``), then ``haenv/_data`` for a wheel.
    """
    env = _os.environ.get("HAENV_DATA_ROOT")
    if env:
        return _Path(env).expanduser().resolve()
    src = _Path(__file__).resolve().parent.parent
    if (src / "config.yaml").is_file():
        return src
    if (_WHEEL_DATA / "config.yaml").is_file():
        return _WHEEL_DATA
    return src


def output_root() -> _Path:
    """Where artefacts are written; separate from :func:`data_root` so a wheel install
    never writes batches into site-packages.

    Order: ``HAENV_OUTPUT_ROOT``, then ``HAENV_DATA_ROOT``, then the source tree, then the
    current working directory for a wheel install.
    """
    env = _os.environ.get("HAENV_OUTPUT_ROOT") or _os.environ.get("HAENV_DATA_ROOT")
    if env:
        return _Path(env).expanduser().resolve()
    d = data_root()
    if d == _WHEEL_DATA:
        return _Path.cwd()
    return d


ROOT = data_root()


def kernel_path(cfg: dict | None = None) -> _Path | None:
    """Resolve the L0 kernel directory.

    Order: ``HAENV_KERNEL_PATH`` (used even if it does not exist), then
    ``config.yaml:kernel_path``, then ``haenv/_kernel`` for a wheel. Returns ``None``
    rather than raising.
    """
    env = _os.environ.get("HAENV_KERNEL_PATH")
    if env:
        return _Path(env).expanduser().resolve()
    if cfg is None:
        try:
            import yaml as _yaml
            cfg = _yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
        except (OSError, ImportError, ValueError):
            cfg = {}
    p = (ROOT / str(cfg.get("kernel_path") or _DEFAULT_KERNEL_REL)).resolve()
    if p.is_dir():
        return p
    if _WHEEL_KERNEL.is_dir():
        return _WHEEL_KERNEL
    return None


# Mount the kernel before any submodule is imported (some import kernel names at module
# level). insert(0) so kernel top-level names such as "build" are not shadowed by PyPI packages.
def _mount_kernel() -> None:
    kp = kernel_path()
    if kp is not None and str(kp) not in _sys.path:
        _sys.path.insert(0, str(kp))
    if kp is not None:
        # The kernel learns haenv's auxiliary streams from the stream manifest.
        from .streams import register_with_kernel
        register_with_kernel()


_mount_kernel()
