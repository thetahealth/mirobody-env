"""`$VAR` expansion in `evaluate.load_env_file`.

Two properties are pinned:
1. Expansion uses the value at **assignment time**, not the value at the end
   of the file (`export B=$A` before `A` is later reassigned);
2. An undefined `$VAR` is not silently expanded to an empty string -- that
   would make an environment with **no key configured** look like one with
   **a key configured as empty**, which are different failure modes.

`load_env_file` reads the user env file; when it is wrong the symptom is
"evaluation runs but auth fails", where nobody would suspect expansion
semantics.

This is an ordinary unit test (synthetic fixtures, touches no data on disk,
does not judge whether what's on disk is correct).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from haenv.evaluate import load_env_file                             # noqa: E402


def _write(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False,
                                     dir=str(ROOT / "_scratch")
                                     if (ROOT / "_scratch").is_dir() else None) as f:
        f.write(text)
        return f.name


def test_var_expansion_is_temporal_not_final():
    """`$VAR` expansion must capture the value at assignment time, not the value at end of file."""
    p = _write("export A=first\nexport B=$A\nexport A=second\n")
    try:
        got = load_env_file(p)
    finally:
        pathlib.Path(p).unlink()
    assert got.get("B") == "first", f"B={got.get('B')!r} —— 应为 first,不是文件末尾的 second"
    assert got.get("A") == "second", f"A={got.get('A')!r}"


def test_undefined_var_is_kept_verbatim_not_silently_emptied():
    """An undefined `$VAR` is kept verbatim, not silently emptied.

    An empty string would make "this variable isn't configured" and "this
    variable is configured as empty" look identical in the output, even
    though the former should raise an error and the latter should be
    treated as an empty value.
    """
    p = _write("export C=$NOSUCHVAR\n")
    try:
        got = load_env_file(p)
    finally:
        pathlib.Path(p).unlink()
    assert got.get("C") == "$NOSUCHVAR", f"C={got.get('C')!r} —— 不许静默变空"
