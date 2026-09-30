#!/usr/bin/env python
"""Assemble `_template.html` + `data.json` + `gen.mjs` into a self-contained `index.html`.

The page must open by double-clicking (`file://`), where `fetch` is blocked, so the data is
inlined into a `<script type="application/json">` block and the generator into a classic
`<script>`. The result is one file with no network requests.

`data.json` is produced by `export_demo_data.py` from evaluation artifacts: one case's
`cases.jsonl` (daily streams, evidence ledger, gold label) and one batch's `eval.jsonl`
(per-dimension scoring rows). The curve in step ① is recomputed in the browser by `gen.mjs`,
a port of `haenv/build.py` that is checked point for point against golden vectors
(`gen_check.mjs`, `check_page_data.py`).

Usage:
    python web/demo/build.py
    # -> web/demo/index.html (generated, do not hand-edit)

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TPL = HERE / "_template.html"
DATA = HERE / "data.json"
#: The browser-side patient generator. Kept as its own file so that
#: `node web/demo/gen_check.mjs` can load it without a browser.
GEN = HERE / "gen.mjs"
OUT = HERE / "index.html"

# The banner is bilingual and must keep the Chinese "勿手改" within the first 400 characters.
BANNER = ("<!-- Generated file — do not hand-edit (生成物,勿手改). Edit `_template.html` or "
          "`data.json`, then rebuild with `python web/demo/build.py`. -->\n")


def main() -> int:
    for p in (TPL, DATA, GEN):
        if not p.is_file():
            print(f"missing {p.relative_to(HERE.parent.parent)}")
            return 2
    tpl = TPL.read_text(encoding="utf-8")
    for slot in ("__DATA__", "__GEN__"):
        if slot not in tpl:
            print(f"the template has no `{slot}` placeholder; the page would be an empty shell")
            return 2
    raw = DATA.read_text(encoding="utf-8")
    json.loads(raw)                      # bad JSON must not produce a half page
    gen = GEN.read_text(encoding="utf-8")
    # `gen.mjs` is both a node `import` and inlined into a classic `<script>`, so it attaches
    # to `globalThis` and must not contain an `export` statement.
    if any(l.startswith("export ") for l in gen.splitlines()):
        print("gen.mjs has an `export` statement, which is a syntax error in a classic "
              "<script>; attach to `globalThis.HaenvGen` instead")
        return 2
    # escape `</` so a `</script>` inside the JSON cannot close the tag early
    html = (BANNER + tpl.replace("__GEN__", gen)
            .replace("__DATA__", raw.replace("</", "<\\/")))
    OUT.write_text(html, encoding="utf-8")
    print(f"ok: {OUT}  {len(html)/1024:.1f} KB"
          f" (data {len(raw)/1024:.1f} KB · generator {len(gen)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
