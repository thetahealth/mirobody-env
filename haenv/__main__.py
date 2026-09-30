"""Entry point for `python -m haenv`.

Lets tools such as `tools/reliability_passk.py` spawn subprocesses with the current
interpreter rather than an installed `haenv` shim.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
