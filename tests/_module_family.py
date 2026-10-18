"""The files a module's definitions live in, when it re-exports from modules split out of it.

`haenv/scoring/gates.py` re-exports what moved to `gates_case.py`, `gates_outcome.py` and
`gates_shortcut.py`; a check that reads a module's source for its definitions has to read
all of them. The set is read from the module's own imports -- a sibling whose name starts
with the module's name and an underscore -- so a later split needs no edit here.
"""
from __future__ import annotations

import ast
import pathlib
from haenv.locations import source_path

ROOT = pathlib.Path(__file__).resolve().parents[1]


def module_family(rel: str, root: pathlib.Path = ROOT) -> list[str]:
    """`rel` first, then each `<stem>_*.py` sibling it imports from, in import order."""
    p = source_path(rel, root=root)
    stem, here = p.stem, pathlib.PurePosixPath(rel).parent
    out = [rel]
    for n in ast.parse(p.read_text(encoding="utf-8")).body:
        if not isinstance(n, ast.ImportFrom) or not n.module:
            continue
        if n.level == 1 and n.module.startswith(stem + "_"):
            sib = (here / f"{n.module}.py").as_posix()
        elif n.level == 0 and n.module.startswith('haenv.') \
                and n.module.rsplit('.', 1)[-1].startswith(stem + '_'):
            sib = n.module.replace('.', '/') + '.py'
        else:
            continue
        if source_path(sib, root=root).is_file() and sib not in out:
            out.append(sib)
    return out


def family_source(rel: str, root: pathlib.Path = ROOT) -> str:
    """The concatenated source of `module_family(rel)`."""
    return "\n".join(source_path(f, root=root).read_text(encoding="utf-8")
                     for f in module_family(rel, root))
