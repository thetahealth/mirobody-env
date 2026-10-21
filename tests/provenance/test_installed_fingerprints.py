"""Source and installed layouts must fingerprint the same actual code and data."""
from __future__ import annotations
from _paths import ROOT as _TEST_ROOT
from haenv.locations import source_path as _haenv_source_path

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest

import haenv.provenance.anchor as anchor


ROOT = _TEST_ROOT


@pytest.fixture(scope="module")
def installed_tree(tmp_path_factory):
    """Mirror the wheel's package/resource layout without invoking a build backend."""
    site = tmp_path_factory.mktemp("installed-fingerprints")
    # Wheel namespaces are direct children of site-packages, not a src checkout.
    package = site / "haenv"
    shutil.copytree(_haenv_source_path('haenv', root=ROOT), package, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(_haenv_source_path('haenv_kernel', root=ROOT), site / "haenv_kernel",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(_haenv_source_path('verifier_core', root=ROOT), site / "verifier_core",
                    ignore=shutil.ignore_patterns("__pycache__"))
    data = package / "_data"
    shutil.copytree(ROOT / "registry", data / "registry")
    shutil.copy2(ROOT / "config.yaml", data / "config.yaml")
    with (ROOT / "pyproject.toml").open("rb") as source:
        included = tomllib.load(source)["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    tools = {src: dst for src, dst in included.items() if src.startswith("tools/")}
    assert set(tools) == {"tools/generation/p3_gen_job.py", "tools/generation/p4_gen_job.py",
                          "tools/validation/invariant_gates.py"}
    for src, dst in tools.items():
        copied = site / dst
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, copied)
    assert not list((data / "tools").rglob("make_freeze.py"))
    return site


def _installed_readings(site):
    script = '\nimport json, sys\nsys.path.insert(0, sys.argv[1])\nimport haenv.provenance.anchor as anchor\nprint(json.dumps({"judging": anchor.judging_fingerprint(),\n                  "world": anchor.world_fingerprint(),\n                  "world_cached": anchor.world_fp_cached(),\n                  "parts": anchor.judge_parts(),\n                  "module": anchor.__file__}))\n'
    env = {k: v for k, v in os.environ.items()
           if k not in {"HAENV_DATA_ROOT", "HAENV_KERNEL_PATH", "PYTHONPATH"}}
    result = subprocess.run([sys.executable, "-I", "-c", script, str(site)],
                            cwd=site, env=env, capture_output=True, text=True, check=True)
    reading = json.loads(result.stdout)
    assert Path(reading.pop("module")).is_relative_to(site)
    return reading


def test_installed_layout_matches_source(installed_tree):
    assert _installed_readings(installed_tree) == {
        "judging": anchor.judging_fingerprint(),
        "world": anchor.world_fingerprint(),
        "world_cached": anchor.world_fp_cached(),
        "parts": anchor.judge_parts(),
    }


def test_semantic_cache_detects_same_size_edit_with_restored_mtime(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    before = anchor.semantic_bytes(path)
    stat = path.stat()
    path.write_text("VALUE = 2\n", encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert anchor.semantic_bytes(path) != before


@pytest.mark.parametrize("rel,installed_rel", [
    ('haenv/scoring/legacy.py', 'haenv/scoring/legacy.py'),
    ("haenv_kernel/verifier.py", "haenv_kernel/verifier.py"),
    ("verifier_core/gate.py", "verifier_core/gate.py"),
    ("registry/scoring.yaml", "haenv/_data/registry/scoring.yaml"),
])
def test_installed_file_edits_and_removal_change_all_stamps(
        monkeypatch, tmp_path, rel, installed_rel):
    """Mutate each wheel location, not the unrelated source checkout."""
    from haenv import judges

    source = tmp_path / "source"
    site = tmp_path / "site"
    src = source / rel
    dst = site / installed_rel
    text = "VALUE: 1\n" if rel.endswith(".yaml") else "VALUE = 1\n"
    for path in (src, dst):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(anchor, "judging_files", lambda: (rel,))
    monkeypatch.setattr(anchor, "generation_files", lambda: (rel,))
    monkeypatch.setattr(anchor, "_plugin_manifest_shas", lambda: ("", ""))
    monkeypatch.setattr(judges, "_CORE", ())
    monkeypatch.setattr(anchor, "ROOT", source)
    expected = anchor.judging_fingerprint(), anchor.world_fingerprint(), anchor.judge_parts()
    monkeypatch.setattr(anchor, "_PACKAGE_ROOT", site / "haenv")
    monkeypatch.setattr(anchor, "ROOT", site / "haenv/_data")

    def readings():
        return anchor.judging_fingerprint(), anchor.world_fp_cached(), anchor.judge_parts()

    assert readings() == expected
    stat = dst.stat()
    dst.write_text(text.replace("1", "2"), encoding="utf-8")
    os.utime(dst, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = readings()
    assert all(a != b for a, b in zip(changed, expected))
    dst.unlink()
    missing = readings()
    assert all(a != b for a, b in zip(missing, changed))
    assert all(a != b for a, b in zip(missing, expected))
    assert src.read_text(encoding="utf-8") == text
    dst.write_text(text, encoding="utf-8")
    assert readings() == expected


def test_explicit_root_does_not_substitute_installed_files(monkeypatch, tmp_path):
    package = tmp_path / "site" / "haenv"
    installed = package / "scoring/legacy.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(anchor, "_PACKAGE_ROOT", package)
    rel = 'haenv/scoring/legacy.py'
    monkeypatch.setattr(anchor, "ROOT", package / "_data")
    present = anchor._fingerprint_of((rel,))
    monkeypatch.setattr(anchor, "ROOT", tmp_path / "explicit-source-root")
    missing = anchor._fingerprint_of((rel,))
    assert missing != present
    direct = anchor.ROOT / rel
    direct.parent.mkdir(parents=True)
    direct.write_text("VALUE = 1\n", encoding="utf-8")
    assert anchor._fingerprint_of((rel,)) != missing
    direct.unlink()
    assert anchor._fingerprint_of((rel,)) == missing
