"""Locations for executing source, read-only resources and writable output.

Supported layouts are repository/haenv, repository/src/haenv, and installed
haenv with resources under haenv/_data. No environment variable selects code.
Canonical source labels retain their namespace and omit the src/ prefix.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path, PurePosixPath

NAMESPACES = frozenset({"haenv", "haenv_kernel", "verifier_core"})

# Explicit protocol layouts, not existence-based candidate lists. The new
# clinician location is the exact docs-moves.json path_map destination.
_RESOURCE_DIRECTORIES = {
    "legacy": {
        "anchors": "docs/anchor",
        "extract_replay": "docs/anchor/extract-replay",
        "clinician_placeholder": "docs/anchor/clinician-placeholder",
        "validity": "docs/anchor",
        "references": "docs/anchor",
    },
    "restructured": {
        "anchors": "registry/anchors",
        "extract_replay": "examples/extract-replay",
        "clinician_placeholder": "examples/extract-replay/clinician-placeholder",
        "validity": "registry/validity",
        "references": "registry/references",
    },
}


def _relative(label: str) -> PurePosixPath:
    path = PurePosixPath(label)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a canonical relative path: {label!r}")
    return path


@lru_cache(maxsize=1)
def source_root() -> Path:
    """Parent of the loaded namespaces, regardless of resource/output overrides."""
    return Path(__file__).resolve().parent.parent


def repository_root() -> Path | None:
    """The loaded checkout, or None for an installed distribution."""
    code = source_root()
    if (code / "pyproject.toml").is_file():
        return code
    if code.name == "src" and (code.parent / "pyproject.toml").is_file():
        return code.parent
    if (code / "haenv/_data").is_dir():
        return None
    raise ValueError(f"Unsupported HAEnv source layout: {code}")


def data_root() -> Path:
    """Resource root; an explicit missing root remains missing."""
    override = os.environ.get("HAENV_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    repo = repository_root()
    return repo if repo is not None else source_root() / "haenv/_data"


def output_root() -> Path:
    """Writable run root, separate from code and versioned resources.

    Explicit output overrides win. The legacy data/output redirect remains
    supported; otherwise runs use artifacts/ under the checkout or working
    directory. No directory is created merely by resolving this location.
    """
    override = os.environ.get("HAENV_OUTPUT_ROOT") or os.environ.get("HAENV_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    repo = repository_root()
    return (repo if repo is not None else Path.cwd()) / "artifacts"


def resource_path(label: str) -> Path:
    return data_root() / _relative(label)


def probe_directory(job_root: Path, job_path: str | Path | None) -> Path:
    """Bundled definitions are resources; external jobs keep their own probe root."""
    supplied = resource_path('benchmarks/inputs').resolve()
    if job_path is not None and Path(job_path).resolve().is_relative_to(supplied):
        return resource_path('benchmarks/probes')
    return Path(job_root) / 'probes'


def resource_directory(kind: str, *, layout: str, root: Path | None = None) -> Path:
    """A named read-only resource location under an explicitly selected layout.

    No files are searched and missing new resources never select old ones.
    ``root`` allows a command to name the corresponding output directory.
    Arbitrary canonical resource labels remain available via resource_path.
    """
    if layout not in _RESOURCE_DIRECTORIES or kind not in _RESOURCE_DIRECTORIES[layout]:
        raise ValueError(f"Unknown resource location: {layout!r}/{kind!r}")
    return (data_root() if root is None else Path(root)) / _RESOURCE_DIRECTORIES[layout][kind]


def output_path(label: str) -> Path:
    return output_root() / _relative(label)


def source_path(label: str, *, root: Path | None = None) -> Path:
    """Resolve loaded code and packaged legacy tools independently of data redirects.

    An explicit comparison root is a flat or src checkout, with its own data;
    missing files never get substituted from the executing installation.
    """
    path = _relative(label)
    if root is not None:
        root = Path(root)
        if path.parts[0] in NAMESPACES:
            flat = (root / "haenv/__init__.py").is_file()
            src = (root / "src/haenv/__init__.py").is_file()
            if flat and src:
                raise ValueError(f"Ambiguous HAEnv comparison layout: {root}")
            return (root / "src" if src else root) / path
        return root / path
    if path.parts[0] in NAMESPACES:
        return source_root() / path
    if path.parts[0] == "tools":
        repo = repository_root()
        base = repo if repo is not None else source_root() / "haenv/_data"
        return base / path
    return resource_path(label)


def module_source_root(module_file: str) -> Path:
    """Namespace parent of a known loaded module, including nested modules.

    The namespace in the module path is the boundary; unrelated ancestors and
    redirected resource/output trees never participate.
    """
    path = Path(module_file).resolve().parent
    while path.name not in NAMESPACES:
        if path == path.parent:
            raise ValueError(f"Module is outside the supported namespaces: {module_file}")
        path = path.parent
    return path.parent
