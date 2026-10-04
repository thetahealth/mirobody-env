#!/usr/bin/env python
"""Check that the data inlined in `index.html` is what production code computes now.

The page recomputes a few patients in the browser (`gen.mjs`) and compares them point for
point with golden vectors carried in its data block. That comparison is only meaningful if
the golden vectors, the kernel constants, the random-layer tables and the weight observation
layer (`wnoise`) are themselves the current output of production code. This script recomputes all three with `page_data.py`,
the same functions the export uses (no second implementation), and compares them with both
`data.json` and the block inlined in `index.html`. The random-layer tables include the
extra case ids of the "parallel futures" fan (`fan_personas`).

    python web/demo/check_page_data.py     # 0 = current · 1 = stale · 2 = missing input

It needs the `haenv` package importable and nothing else: no evaluation results, no models.
`.github/workflows/pages.yml` runs it before publishing; `node web/demo/gen_check.mjs` then
checks the browser generator against the same golden vectors.

On failure, the page data is stale: re-export it and rebuild with `web/demo/build.py`.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data.json"
PAGE = HERE / "index.html"


def _page_data():
    spec = importlib.util.spec_from_file_location("haenv_demo_page_data", HERE / "page_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _inlined() -> dict | None:
    m = re.search(r'<script id="haenv-data"[^>]*>([\s\S]*?)</script>',
                  PAGE.read_text(encoding="utf-8"))
    return json.loads(m.group(1).replace("<\\/", "</")) if m else None


def _golden_diff(want: dict, have: dict) -> list[str]:
    got = have.get("cases") or []
    if len(got) != len(want["cases"]):
        return [f"{len(got)} golden case(s), production computes {len(want['cases'])}"]
    bad = []
    for w, h in zip(want["cases"], got):
        if w["label"] != h.get("label"):
            bad.append(f"case {w['label']!r}: label differs ({h.get('label')!r})")
            continue
        if w["weight"] != h.get("weight"):
            bad.append(f"{w['label']}: weight series differs")
        for part in ("skeleton", "rule_readout", "base", "carried_forward", "observed"):
            if w.get(part) != h.get(part):
                bad.append(f"{w['label']}: {part} differs")
        for sig, c in w["clinical"].items():
            hc = (h.get("clinical") or {}).get(sig) or {}
            if c["pts"] != hc.get("pts"):
                bad.append(f"{w['label']}/{sig}: lab series differs")
            if c.get("abn") != hc.get("abn"):
                bad.append(f"{w['label']}/{sig}: abnormal flags differ")
    if want.get("overlay") != have.get("overlay"):
        bad.append("LLM-path golden cases differ")
    return bad


def main() -> int:
    for p in (DATA, PAGE, HERE / "page_data.py"):
        if not p.is_file():
            print(f"missing {p.name}")
            return 2
    inlined = _inlined()
    if inlined is None:
        print("index.html has no inlined data block")
        return 2
    x = _page_data()
    want = {"golden": x._golden(), "kernel": x._kernel(), "personas": x._personas(),
            "fan_personas": x._fan_personas(), "wnoise": x._wnoise()}
    bad = []
    for name, have in (("data.json", json.loads(DATA.read_text(encoding="utf-8"))),
                       ("index.html", inlined)):
        g = have.get("golden")
        if not g:
            bad.append(f"{name}: no golden section")
        else:
            bad += [f"{name}: {b}" for b in _golden_diff(want["golden"], g)]
        k = have.get("kernel") or {}
        diff = sorted(c for c in set(want["kernel"]) | set(k) if want["kernel"].get(c) != k.get(c))
        if diff:
            bad.append(f"{name}: kernel constants differ: {diff}")
        if have.get("wnoise") != want["wnoise"]:
            bad.append(f"{name}: the weight observation layer (wnoise) differs")
        for sec in ("personas", "fan_personas"):
            p = have.get(sec) or {}
            diff = sorted(c for c in set(want[sec]) | set(p) if want[sec].get(c) != p.get(c))
            if diff:
                bad.append(f"{name}: {sec} random-layer tables differ for {diff}")
    if bad:
        print("The page data is not what production code computes now:\n  " + "\n  ".join(bad[:20]))
        print("The page data is stale: re-export it, then rebuild with web/demo/build.py.")
        return 1
    n = len(want["golden"]["cases"])
    print(f"ok: {n} golden case(s), {len(want['kernel'])} kernel constant(s) and "
          f"{len(want['personas']) + len(want['fan_personas'])} random-layer table(s) and the scale-noise layer match "
          f"production in data.json and index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
