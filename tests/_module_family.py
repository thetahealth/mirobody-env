"""The files a module's definitions live in, when it re-exports from modules split out of it.

`haenv/gates.py` re-exports what moved to `gates_case.py`, `gates_outcome.py` and
`gates_shortcut.py`; a check that reads a module's source for its definitions has to read
all of them. The set is read from the module's own imports -- a sibling whose name starts
with the module's name and an underscore -- so a later split needs no edit here.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def module_family(rel: str, root: pathlib.Path = ROOT) -> list[str]:
    """`rel` first, then each `<stem>_*.py` sibling it imports from, in import order."""
    p = root / rel
    stem, here = p.stem, pathlib.PurePosixPath(rel).parent
    out = [rel]
    for n in ast.parse(p.read_text(encoding="utf-8")).body:
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module \
                and n.module.startswith(stem + "_"):
            sib = (here / f"{n.module}.py").as_posix()
            if (root / sib).is_file() and sib not in out:
                out.append(sib)
    return out


def family_source(rel: str, root: pathlib.Path = ROOT) -> str:
    """The concatenated source of `module_family(rel)`."""
    return "\n".join((root / f).read_text(encoding="utf-8") for f in module_family(rel, root))
