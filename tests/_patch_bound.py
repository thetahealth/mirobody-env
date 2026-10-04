"""Patch a name in every `haenv` module that binds the same object.

`haenv.evaluate` re-exports what was split out of it into `qside`, `run_ledger`, `solvers`,
`rows` and `run_state`, and those modules import from one another by name. Patching
`evaluate.X` alone therefore no longer reaches a caller in another module: it looks `X` up
in its own namespace. `patch_bound` replaces `X` wherever it is bound to the object
`evaluate.X` holds, which is what a single `setattr` on `evaluate` did before the split.
"""
from __future__ import annotations

import sys


def patch_bound(monkeypatch, name: str, value, *, source: str = "haenv.evaluate",
                raising: bool = True) -> int:
    """Replace `name` in every loaded `haenv` module bound to the same object as in `source`.

    Returns how many modules were patched; raises if `source` does not define the name, so
    a typo or a renamed target fails instead of patching nothing. `raising=False` keeps
    `monkeypatch.setattr`'s meaning: the name is set on `source` alone.
    """
    import importlib
    origin = importlib.import_module(source)
    if not hasattr(origin, name):
        if not raising:
            monkeypatch.setattr(origin, name, value, raising=False)
            return 1
        raise AttributeError(f"{source} has no attribute {name!r}: nothing to patch")
    target = getattr(origin, name)
    n = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not (mod_name == "haenv" or mod_name.startswith("haenv.")):
            continue
        if hasattr(mod, name) and getattr(mod, name) is target:
            monkeypatch.setattr(mod, name, value)
            n += 1
    return n
