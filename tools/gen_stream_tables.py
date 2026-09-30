"""Write the stream prices that scoring reads, from the stream manifest.

Gated pricing decides scores, so it is written into `registry/gated_pricing_streams.yaml`
(judging segment) instead of being derived at run time from `registry/streams.yaml`.

    uv run python tools/gen_stream_tables.py            # write
    uv run python tools/gen_stream_tables.py --check    # exit 1 if out of sync

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haenv import canary, streams                                   # noqa: E402

OUT = ROOT / "registry" / streams.PRICING_FILENAME

HEADER = """\
# GENERATED from registry/streams.yaml by tools/gen_stream_tables.py. Do not edit.
# Rebuild: uv run python tools/gen_stream_tables.py
#
# Gated-geometry price tier of every stream. Scoring reads this file rather than
# the manifest so that the judging fingerprint moves only when a price changes.
# Tiers are defined in gated_pricing.yaml:basis.
version: 1
streams:
"""


def render() -> str:
    rows = "".join(f"  {n}: {t}\n" for n, t in streams.pricing().items())
    return canary.block("# ") + HEADER + rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the file is out of sync")
    a = ap.parse_args(argv)
    text = render()
    current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
    if a.check:
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is out of sync with registry/streams.yaml; "
                  f"run: uv run python tools/gen_stream_tables.py")
            return 1
        print(f"{OUT.relative_to(ROOT)} in sync ({len(streams.pricing())} streams)")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(streams.pricing())} streams)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
