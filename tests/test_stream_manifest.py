"""The stream manifest is the one place a stream is declared.

`registry/streams.yaml` feeds every per-stream table; these tests pin that a
single entry reaches all of them and that the scoring-side copy stays in sync.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import copy
import importlib.util
import pathlib
import shutil
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import haenv                                                        # noqa: E402

sys.path.insert(0, str(haenv.kernel_path()))

import synth                                                        # noqa: E402
from latent import DISEASE_SIGNAL_DOMAIN, KNOWN_DEVICES             # noqa: E402

from haenv import events, gated, gates, indicators, streams, wearable   # noqa: E402
from haenv.physio import apply as physio_apply                      # noqa: E402
from haenv.regpath import registry_dir_override                     # noqa: E402


def _gen_tool():
    spec = importlib.util.spec_from_file_location(
        "gen_stream_tables", ROOT / "tools" / "gen_stream_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pricing_file_is_in_sync_with_the_manifest():
    tool = _gen_tool()
    assert tool.OUT.read_text(encoding="utf-8") == tool.render(), (
        "registry/gated_pricing_streams.yaml is stale: run "
        "`uv run python tools/gen_stream_tables.py`")


def test_every_stream_is_priced_through_the_production_path():
    for name in streams.manifest():
        assert gated.kind_of(name) == streams.pricing()[name], name


def test_generation_tables_are_the_manifest():
    m = streams.manifest()
    assert [s.name for s in events.METRICS] == [n for n, e in m.items() if "render" in e]
    assert events.AUX_WHITELIST == set(streams.aux_metrics())
    assert set(synth.AUX_SIGNALS) == set(streams.aux_signals()), (
        "the kernel's own auxiliary signals must also be declared aux in the manifest")
    assert gates.DEVICE_SIGNALS == streams.device_signals()
    assert wearable.DERIVED_BINDINGS == streams.derived_bindings()
    assert wearable.GRID_PARENT == streams.grid_parents()
    reg = physio_apply.load_stream_registry(ROOT / "registry" / "physio_streams.yaml")
    assert reg.excluded == streams.physio_excluded()
    assert set(reg.specs) >= streams.physio_rendered()


def test_devices_match_the_kernel_vocabulary():
    assert set(streams.DEVICES) == set(KNOWN_DEVICES)


def test_clinical_streams_use_kernel_names():
    """A device map spelling `hba1c` where the kernel says `HbA1c` makes the device-idle
    check judge every lab panel idle. A non-auxiliary stream must be a kernel
    disease signal, spelled the kernel's way."""
    kernel = {s for d in DISEASE_SIGNAL_DOMAIN.values() for s in d}
    odd = [n for n, e in streams.manifest().items() if not e["aux"] and n not in kernel]
    assert not odd, odd


def test_rendered_streams_have_a_consistent_dossier():
    """Unit and precision are read from the dossier; its payload range must be the
    range the pipeline enforces (the physiology bound when the layer renders the
    stream, else the render range)."""
    bad = []
    for name, e in streams.manifest().items():
        if "render" not in e:
            continue
        pay = indicators.of(name)["payload_range"]
        if e["physio"] == "render":
            reg = physio_apply.load_stream_registry(ROOT / "registry" / "physio_streams.yaml")
            want = (reg.specs[name].bound.lo, reg.specs[name].bound.hi)
        else:
            want = tuple(e["render"]["hard_range"])
        if (pay.get("low"), pay.get("high")) != tuple(float(x) for x in want):
            bad.append(f"{name}: dossier {pay.get('low')}-{pay.get('high')} vs {want}")
    assert not bad, bad


def _with_manifest(tmp_path, edit):
    reg = tmp_path / "registry"
    shutil.copytree(ROOT / "registry", reg)
    doc = yaml.safe_load((reg / streams.FILENAME).read_text(encoding="utf-8"))
    edit(doc["streams"])
    (reg / streams.FILENAME).write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return registry_dir_override(reg)


def test_one_entry_reaches_every_derived_table(tmp_path):
    """Negative control: declare a new stream once and every
    table that derives from the manifest sees it."""
    def add(s):
        s["probe_stream"] = copy.deepcopy(s["exercise_mets"])
        s["probe_stream"]["pricing"] = "advanced_lab"
    with _with_manifest(tmp_path, add):
        assert "probe_stream" in streams.aux_signals()
        assert "probe_stream" in streams.physio_excluded()
        assert streams.pricing()["probe_stream"] == "advanced_lab"
        assert streams.derived_bindings()["probe_stream"] == ("exercise_avg_hr", "resting_hr", "age")
        assert streams.grid_parents()["probe_stream"] == "resting_hr"
        assert "probe_stream" in _gen_tool().render()


@pytest.mark.parametrize("edit, match", [
    (lambda s: s["steps"].update(pricing="free"), "pricing"),
    (lambda s: s["steps"].update(colour="red"), "unknown fields"),
    (lambda s: s["steps"].pop("physio"), "missing"),
    (lambda s: s["steps"].pop("devices"), "must list its devices"),
    (lambda s: s["active_burn"].update(grid_parent="weight"), "grid_parent"),
    (lambda s: s["active_burn"].update(derived_from=["stepz"]), "derived_from"),
    (lambda s: s["steps"].update(proves_device=["smartwatch"]), "unknown devices"),
])
def test_malformed_entries_are_refused(tmp_path, edit, match):
    with _with_manifest(tmp_path, edit):
        with pytest.raises(streams.ManifestError, match=match):
            streams.manifest()


def test_physio_render_without_a_block_is_refused(tmp_path):
    with _with_manifest(tmp_path, lambda s: s["exercise_mets"].update(physio="render")):
        with pytest.raises(physio_apply.StreamRegistryError, match="no parameter block"):
            physio_apply.load_stream_registry(tmp_path / "registry" / "physio_streams.yaml")
