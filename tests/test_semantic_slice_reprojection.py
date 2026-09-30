"""A saved slice may be reconstructed only under the identical payload builder."""
from dataclasses import dataclass

import pytest

from haenv.semantic_pipeline import _context_for_cell


@dataclass
class Visible:
    observations: str


@dataclass
class Gold:
    T: int
    diagnosis: str


def case(monkeypatch):
    import build
    calls = []
    def builder(raw, time):
        calls.append(time)
        return (Visible("canonical") if time == 10 else Visible("slice"), Gold(time, "A"))
    monkeypatch.setattr(build, "build_instance", builder)
    row = {"case": "C", "geometry": "slices", "slice_rows": [{"t": 9}], "world_sha": "old"}
    raw, saved = {"C": object()}, {"C": (Visible("canonical"), Gold(10, "A"))}
    metadata = {"world_sha": "old", "kernel_sha256": "kernel-a"}
    return row, raw, saved, metadata, calls


def test_changed_generation_gate_does_not_block_identical_saved_payload_builder(monkeypatch):
    row, raw, saved, metadata, calls = case(monkeypatch)
    checked = set()
    sp, vp = _context_for_cell(row, raw, saved, metadata, "kernel-a", checked)
    assert (sp.observations, vp.T) == ("slice", 9)
    assert calls == [10, 9] and checked == {"C"}
    _context_for_cell(row, raw, saved, metadata, "kernel-a", checked)
    assert calls == [10, 9, 9]


@pytest.mark.parametrize("change", ["kernel", "world", "canonical", "future"])
def test_any_unproven_reprojection_stops_before_using_a_slice(monkeypatch, change):
    row, raw, saved, metadata, calls = case(monkeypatch)
    current = "kernel-a"
    if change == "kernel":current = "kernel-b"
    elif change == "world":row["world_sha"] = "different"
    elif change == "canonical":saved["C"] = (Visible("wrong"), Gold(10, "A"))
    else:row["slice_rows"] = [{"t": 11}]
    with pytest.raises(ValueError):
        _context_for_cell(row, raw, saved, metadata, current, set())
    assert 9 not in calls and 11 not in calls
