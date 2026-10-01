"""Fingerprints do not depend on the Python version that computes them.

The judging, world and solving fingerprints hash source normalised through `ast.unparse`, whose
output changed in Python 3.11; on 3.10 the same files would otherwise hash differently. The
cases pin the 3.11 form on whichever interpreter runs them, the cases 3.10 already wrote that
way included, and the judging files must hash as the freeze anchor records them.

This is an ordinary unit test (reads the source tree and the freeze anchor only).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib

import pytest

from haenv import anchor

CASES = [
    ("for a, b in pairs:\n    pass\n", "for a, b in pairs:\n    pass"),
    ("a, b = c\n", "a, b = c"),
    ("x = 1, 2\n", "x = (1, 2)"),
    ("y = [k for k, v in d.items()]\n", "y = [k for k, v in d.items()]"),
    ("t = ()\n", "t = ()"),
    ("u = a[(*b, c)]\n", "u = a[*b, c]"),
    ("f = lambda: 0\n", "f = lambda: 0"),
    ("g = lambda x, *, y=1: x\n", "g = lambda x, *, y=1: x"),
    ("if (n := 10) > 5:\n    pass\n", "if (n := 10) > 5:\n    pass"),
    ('def h():\n    """Docstrings do not count."""\n    return x, y\n', "def h():\n    return (x, y)"),
    # characters Unicode 14.0 added are written as themselves; others stay escaped
    ("s = '\\u9fff'\n", "s = '\u9fff'"),
    ("s = '\\U0001fae0'\n", "s = '\U0001fae0'"),
    ("s = '\\uffff'\n", "s = '\\uffff'"),
    ("s = '\\\\u9fff'\n", "s = '\\\\u9fff'"),
    ("z = f\"{'\u9fff'}\"\n", "z = f\"{'\u9fff'}\""),
    # no single quote style fits every part of the f-string
    ("z = f'\\\"\\\"\\\"{x}\\\"\\\"\\\"{y}\\'\\'\\''\n", "z = f'''\"\"\"{x}\"\"\"{y}\\'\\'\\''''"),
]


@pytest.mark.parametrize("source, normalised", CASES, ids=[str(i) for i in range(len(CASES))])
def test_source_normalises_to_the_311_form(source, normalised):
    assert anchor._semantic_of(source.encode("utf-8"), ".py").decode("utf-8") == normalised


def test_judging_files_hash_as_the_anchor_records():
    recorded = anchor.load_freeze()["judging"]
    got = {rel: hashlib.sha256(anchor.semantic_bytes(anchor._fingerprint_path(rel))).hexdigest()[:16]
           for rel in anchor.judging_files()}
    assert {rel for rel in got if got[rel] != recorded.get(rel)} == set()
