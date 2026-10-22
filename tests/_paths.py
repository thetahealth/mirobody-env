"""Repository locations and the read-only scan surface of evaluation runs."""
from __future__ import annotations

import os
from pathlib import Path

from haenv import locations

ROOT = Path(__file__).resolve().parents[1]


def result_roots(*, legacy_root: Path | None = None) -> tuple[Path, ...]:
    """Use the package resolver with this test tree's legacy checkout base."""
    return locations.result_roots(legacy_root=ROOT if legacy_root is None else legacy_root)


def result_glob(pattern: str, *parts: str | Path) -> list[Path]:
    """Sorted matches under every result root, optionally below a task/pack.

    Callers retain their original pattern (including recursive ``**``); no
    task, pack, stamp or batch is selected implicitly.
    """
    return sorted(p for root in result_roots()
                  for p in root.joinpath(*parts).glob(pattern))


def result_path(*parts: str | Path) -> Path:
    """Locate an explicitly named historical artifact, without changing its ID.

    Prefer the first existing copy in root order. Missing artifacts keep the
    legacy label so existing absence checks still describe the named object.
    This is a reader helper; writers must use their own temporary destinations.
    Use ``result_glob`` for directories whose batches span multiple roots.
    """
    for root in result_roots():
        path = root.joinpath(*parts)
        if path.exists():
            return path
    return (ROOT / "results").joinpath(*parts)


def result_walk():
    """Walk all roots in fixed order, following historical directory symlinks."""
    for root in result_roots():
        for directory, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            yield directory, dirs, sorted(files)


def result_label(path: Path) -> str:
    """Repository-relative provenance where possible, absolute for overrides."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def response_batches(*, include_superseded: bool = False) -> list[Path]:
    """The failure-attribution surface, across both roots (responses required)."""
    paths, seen = [], set()
    for root in result_roots():
        for directory, _dirs, files in os.walk(root, followlinks=True):
            path = Path(directory)
            if not include_superseded and "_superseded" in path.relative_to(root).parts:
                continue
            resolved = path.resolve()
            if "responses.jsonl" in files and resolved not in seen:
                paths.append(path)
                seen.add(resolved)
    return sorted(paths)


def result_relative(path: Path) -> Path:
    """Task/pack/stamp path, independent of which result root stores it."""
    for root in result_roots():
        if path.is_relative_to(root):
            return path.relative_to(root)
    raise ValueError(f"Outside the evaluation scan surface: {path}")


def result_sort_key(path: Path) -> tuple[str, str]:
    """Preserve within-results ordering when selecting batches across roots."""
    return result_relative(path).as_posix(), str(path)
