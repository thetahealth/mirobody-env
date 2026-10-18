"""Locate a maintainer checkout by markers, independently of a command's depth."""
from __future__ import annotations

from pathlib import Path
import sys
import subprocess


def repository_root(anchor=__file__) -> Path:
    """Find the enclosing checkout, or fail explicitly outside a repository."""
    anchor = Path(anchor).resolve()
    for root in (anchor, *anchor.parents):
        if (root / 'pyproject.toml').is_file() and (root / 'tools').is_dir():
            return root
    raise FileNotFoundError(f'Cannot locate repository root from {anchor}')


def bootstrap(anchor=__file__) -> Path:
    """Use this checkout's code and plugins for direct and imported commands."""
    root = repository_root(anchor)
    flat, src = root / 'haenv/__init__.py', root / 'src/haenv/__init__.py'
    if flat.is_file() and src.is_file():
        raise ValueError(f'Ambiguous source layout in {root}')
    if not flat.is_file() and not src.is_file():
        raise ValueError(f'Missing HAEnv source namespace in {root}')
    code = root / 'src' if src.is_file() else root
    for path in (root, code, root / 'plugins'):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


def command_path(label: str, *, root: Path | None = None) -> Path:
    """Locate a current CLI by its exact path or unique basename; never guess a replacement."""
    root = repository_root() if root is None else Path(root)
    relative = Path(label)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Expected a relative tool label: {label!r}')
    if relative.parts[:1] == ('tools',):
        relative = Path(*relative.parts[1:])
    matches = (list((root / 'tools').rglob(relative.name)) if len(relative.parts) == 1
               else [root / 'tools' / relative])
    matches = [p for p in matches if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f'Tool label must resolve to one command: {label!r}: {matches}')
    return matches[0]


def frozen_command_path(root: Path, name: str) -> Path:
    """Use a sealed checkout's own Git tree, retaining its historical command layout."""
    if Path(name).name != name or not name.endswith('.py'):
        raise ValueError(f'Expected an exact command filename: {name!r}')
    root = Path(root)
    tracked = subprocess.check_output(
        ['git', '-C', str(root), 'ls-tree', '-r', '--name-only', 'HEAD', '--', 'tools'],
        text=True).splitlines()
    matches = [root / p for p in tracked if Path(p).name == name]
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f'Sealed source must declare one {name}: {matches}')
    return matches[0]
