"""The demo freshness check applies its float tolerance only to response coefficients."""
import copy
import json
import math
from types import SimpleNamespace

import pytest

from web.demo import check_page_data as check


@pytest.mark.parametrize("artifact", ["data.json", "index.html"])
@pytest.mark.parametrize("path,steps,expected", [
    (("personas", "demo", "response", "low"), 1, 0),
    (("personas", "demo", "response", "by_drug", "metformin"), 1, 0),
    (("personas", "demo", "response", "low"), 5, 1),
    (("personas", "demo", "response", "by_drug", "metformin"), 5, 1),
    (("personas", "demo", "response", "other"), 1, 1),
    (("personas", "demo", "descent_k"), 1, 1),
    (("fan_personas", "fan", "descent_k"), 1, 1),
    (("kernel", "ar_phi"), 1, 1),
    (("wnoise", "sd_frac"), 1, 1),
    (("golden", "cases", 0, "weight", 0), 1, 1),
    (("golden", "overlay", "weight"), 1, 1),
])
def test_page_data_float_tolerance(monkeypatch, tmp_path, artifact, path, steps, expected):
    want = {
        "personas": {"demo": {"descent_k": 1.0, "response": {
            "low": 1.0, "by_drug": {"metformin": 1.0}, "other": 1.0,
        }}},
        "fan_personas": {"fan": {"descent_k": 1.0}},
        "kernel": {"ar_phi": 1.0},
        "wnoise": {"sd_frac": 1.0},
        "golden": {"cases": [{"label": "demo", "weight": [1.0], "clinical": {}}],
                   "overlay": {"weight": 1.0}},
    }
    release = {"freeze_revision": 120}
    computed = SimpleNamespace(
        **{f"_{key}": (lambda value=value: value) for key, value in want.items()},
        _release=lambda: release,
    )
    original = {**want, "provenance": {"release": release}}
    have = copy.deepcopy(original)
    fields = have
    for key in path[:-1]:
        fields = fields[key]
    for _ in range(steps):
        fields[path[-1]] = math.nextafter(fields[path[-1]], math.inf)

    data = tmp_path / "data.json"
    page = tmp_path / "index.html"
    data.write_text(json.dumps(have if artifact == data.name else original), encoding="utf-8")
    inlined = json.dumps(have if artifact == page.name else original)
    page.write_text(f'<script id="haenv-data">{inlined}</script>', encoding="utf-8")
    monkeypatch.setattr(check, "DATA", data)
    monkeypatch.setattr(check, "PAGE", page)
    monkeypatch.setattr(check, "_page_data", lambda: computed)

    assert check.main() == expected
