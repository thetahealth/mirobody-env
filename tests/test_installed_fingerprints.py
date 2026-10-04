"""Source and installed layouts must fingerprint the same actual code and data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from haenv import anchor


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def installed_tree(tmp_path_factory):
    """Mirror the wheel's package/resource layout without invoking a build backend."""
    site = tmp_path_factory.mktemp("installed-fingerprints")
    package = site / "haenv"
    shutil.copytree(ROOT / "haenv", package, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "haenv_kernel", site / "haenv_kernel",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "verifier_core", site / "verifier_core",
                    ignore=shutil.ignore_patterns("__pycache__"))
    data = package / "_data"
    shutil.copytree(ROOT / "registry", data / "registry")
    shutil.copy2(ROOT / "config.yaml", data / "config.yaml")
    (data / "tools").mkdir()
    shutil.copy2(ROOT / "tools" / "make_freeze.py", data / "tools" / "make_freeze.py")
    return site


def _installed_readings(site):
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
from haenv import anchor
print(json.dumps({"judging": anchor.judging_fingerprint(),
                  "world": anchor.world_fingerprint(),
                  "world_cached": anchor.world_fp_cached(),
                  "parts": anchor.judge_parts(),
                  "module": anchor.__file__}))
"""
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
    ("haenv/scoring.py", "haenv/scoring.py"),
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
    monkeypatch.setattr(anchor, "ROOT", site / "haenv" / "_data")

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
    package.mkdir(parents=True)
    (package / "scoring.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(anchor, "_PACKAGE_ROOT", package)
    monkeypatch.setattr(anchor, "ROOT", tmp_path / "explicit-source-root")
    rel = "haenv/scoring.py"
    missing = anchor._fingerprint_of((rel,))
    direct = anchor.ROOT / rel
    direct.parent.mkdir(parents=True)
    direct.write_text("VALUE = 1\n", encoding="utf-8")
    assert anchor._fingerprint_of((rel,)) != missing
    direct.unlink()
    assert anchor._fingerprint_of((rel,)) == missing
