"""Semantic content of a source file: `.py` by AST without docstrings, `.yaml` as canonical JSON.
The freeze fingerprints and the move certifier hash this, so comment edits move neither.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys


#: Cache parsing by actual content: edits can preserve size and mtime, and even
#: ctime has limited resolution on some filesystems. File reads stay live.
#:
#: Semantic content: `.py` via AST with docstrings stripped, `.yaml` via
#: `safe_load` to JSON with key order preserved; a parse failure falls back to
#: raw bytes. Pack fingerprints (`cases.jsonl`, job files) stay byte-exact.
_SEM_CACHE: dict[tuple, bytes] = {}


def semantic_bytes(path: pathlib.Path) -> bytes:
    """A file's semantic content (comments/docstrings/formatting excluded).
    Falls back to the raw bytes if parsing fails."""
    raw = path.read_bytes()
    suf = path.suffix.lower()
    key = (suf, hashlib.sha256(raw).digest())
    hit = _SEM_CACHE.get(key)
    if hit is not None:
        return hit
    out = _semantic_of(raw, suf)
    if len(_SEM_CACHE) > 4096:                                   # simple cap, avoid unbounded growth
        _SEM_CACHE.clear()
    _SEM_CACHE[key] = out
    return out


def _semantic_cached(raw: bytes, suf: str) -> bytes:
    """`_semantic_of`, memoised on content like `semantic_bytes`."""
    key = (suf, hashlib.sha256(raw).digest())
    hit = _SEM_CACHE.get(key)
    if hit is None:
        hit = _semantic_of(raw, suf)
        if len(_SEM_CACHE) > 4096:
            _SEM_CACHE.clear()
        _SEM_CACHE[key] = hit
    return hit


def _semantic_of(raw: bytes, suf: str) -> bytes:
    try:
        if suf == ".py":
            import ast
            tree = ast.parse(raw.decode("utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef,
                                         ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
            return _unparse(ast.fix_missing_locations(tree)).encode("utf-8")
        if suf in (".yaml", ".yml"):
            import json

            import yaml
            # Key order is preserved: some registry tables are first-match-wins.
            return json.dumps(yaml.safe_load(raw.decode("utf-8")),
                              ensure_ascii=False, sort_keys=False,
                              default=str).encode("utf-8")
    except Exception:                                            # noqa: BLE001
        return raw                                               # fail-closed
    return raw


def _unparse(tree) -> str:
    """`ast.unparse` as Python 3.11 and later write it, on every supported version."""
    import ast
    if sys.version_info >= (3, 11):
        return ast.unparse(tree)
    from ._unparse310 import unparse
    return unparse(tree)


def _semantic_source(src: str) -> bytes:
    import textwrap
    return _semantic_of(textwrap.dedent(src).encode("utf-8"), ".py")
