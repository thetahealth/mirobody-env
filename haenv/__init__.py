"""Generate, evaluate and report on longitudinal health agents.

A job file supplies a raw case and its latent control variables; the pipeline
builds the world, runs the models, scores them and writes a report.

The L0 kernel lives in ``haenv_kernel/`` inside this repository.
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


#: The kernel package directory. A source tree and a wheel resolve the same way: `haenv/`
#: and `haenv_kernel/` are siblings at the repository root and in `site-packages`.
_INSTALLED_KERNEL = _Path(__file__).resolve().parent.parent / "haenv_kernel"

#: Kernel location relative to the repository root; fallback when `config.yaml:kernel_path`
#: cannot be read.
_DEFAULT_KERNEL_REL = "haenv_kernel"

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
    """Resolve the directory a tool reads kernel *source* from.

    This does not decide which kernel executes: that is always the `haenv_kernel`
    package on the import path (see `batch.kernel_fingerprint`, which reads the loaded
    module rather than this directory). What this answers is "where is the kernel source
    a read-only tool should parse" -- `analytics` reads `verifier.py` by AST, and the
    maintainers' document-number tooling takes its kernel root from here.

    Order: ``HAENV_KERNEL_PATH`` (a comparison checkout; used even if it does not exist),
    then ``config.yaml:kernel_path``, then the installed package directory for a wheel.
    Returns ``None`` rather than raising.
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
    # Installed wheel only: ROOT is the packaged `_data` directory and the kernel is a
    # sibling package next to `haenv/`. An explicitly overridden ROOT must never fall
    # back to the installed kernel -- that would silently read the wrong tree, the same
    # rule `anchor._fingerprint_path` states for the fingerprint inputs.
    if ROOT == _WHEEL_DATA and _INSTALLED_KERNEL.is_dir():
        return _INSTALLED_KERNEL
    return None


# Hand the kernel the auxiliary stream manifest. The kernel itself is an ordinary package
# (`haenv_kernel`), imported like any other; nothing here puts a directory on `sys.path`.
def _register_kernel_streams() -> None:
    kp = kernel_path()
    if kp is None:
        return
    from .streams import register_with_kernel
    register_with_kernel()


_register_kernel_streams()
