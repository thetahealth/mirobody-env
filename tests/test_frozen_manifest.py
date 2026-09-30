"""The MANIFEST pack entries of `frozen/` and the pack list `split_wq.py` works from.

    uv run pytest tests/test_frozen_manifest.py -q          # seconds

What is held here:

* every published Q pack has a MANIFEST entry and every entry names a Q file on disk;
* each entry agrees with its Q file (case count, case ids) and with the freeze
  anchor (the anchor records the entry's `stamped_sha256` by prefix);
* `split_wq.py` takes its pack list from MANIFEST, so a stale local `*.cases.jsonl`
  that no entry names is never split or joined.

Every check that can say "clean" has a planted defect next to it that must make
it say "not clean".

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frozen"))

from haenv import canary as C                        # noqa: E402

split_wq = importlib.import_module("split_wq")


def _copy(dst: pathlib.Path) -> pathlib.Path:
    """A throwaway root holding MANIFEST, the Q packs and the job files they name."""
    (dst / "frozen").mkdir(parents=True)
    shutil.copy2(ROOT / "frozen" / "MANIFEST.json", dst / "frozen" / "MANIFEST.json")
    for p in C.jsonl_packs(ROOT):
        shutil.copy2(p, dst / "frozen" / p.name)
    for e in split_wq.pack_entries().values():
        (dst / e["job"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / e["job"], dst / e["job"])
    return dst


def _edit_manifest(root: pathlib.Path, fn) -> None:
    mp = root / "frozen" / "MANIFEST.json"
    man = json.loads(mp.read_text(encoding="utf-8"))
    fn(man)
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")


def test_entries_cover_every_published_pack():
    entries = split_wq.pack_entries()
    assert len(entries) >= 2, entries.keys()
    assert {e["q_file"] for e in entries.values()} == {p.name for p in C.jsonl_packs(ROOT)}


def test_entries_agree_with_q_files_and_anchor():
    """Positive side, on the tree as it is."""
    assert split_wq.entry_problems(ROOT) == []


def test_a_wrong_case_count_is_reported(tmp_path):
    root = _copy(tmp_path)
    job = sorted(split_wq.pack_entries())[0]
    _edit_manifest(root, lambda m: m[job].update(n_cases=m[job]["n_cases"] + 1))
    got = split_wq.entry_problems(root)
    assert any(job in g and "n_cases" in g for g in got), got


def test_a_pack_the_anchor_does_not_record_is_reported(tmp_path):
    root = _copy(tmp_path)
    job = sorted(split_wq.pack_entries())[0]
    _edit_manifest(root, lambda m: m[job].update(stamped_sha256="0" * 64))
    got = split_wq.entry_problems(root)
    assert any(job in g and "anchor" in g for g in got), got


def test_a_q_file_without_an_entry_is_reported(tmp_path):
    root = _copy(tmp_path)
    job = sorted(split_wq.pack_entries())[0]
    _edit_manifest(root, lambda m: m.pop(job))
    got = split_wq.entry_problems(root)
    assert any("on disk" in g for g in got), got


def test_an_edited_job_file_is_reported(tmp_path):
    root = _copy(tmp_path)
    e = sorted(split_wq.pack_entries().items())[0][1]
    jy = root / e["job"]
    jy.write_text(jy.read_text(encoding="utf-8") + "\n# planted\n", encoding="utf-8")
    got = split_wq.entry_problems(root)
    assert any("job_yaml_sha256" in g for g in got), got


def test_a_manifest_without_entries_is_a_problem(tmp_path):
    root = _copy(tmp_path)
    _edit_manifest(root, lambda m: [m.pop(k) for k in list(split_wq.pack_entries(m))])
    assert split_wq.entry_problems(root) == ["frozen/MANIFEST.json names no pack"]


def test_the_pack_list_is_manifest_not_a_glob(tmp_path, monkeypatch):
    """A `*.cases.jsonl` left on disk that no entry names is not a pack."""
    (tmp_path / "MANIFEST.json").write_text(
        (ROOT / "frozen" / "MANIFEST.json").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "stale-local.cases.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(split_wq, "HERE", tmp_path)
    names = {p.name for p in split_wq._packs()}
    assert "stale-local.cases.jsonl" not in names
    assert names == {e["file"] for e in split_wq.pack_entries().values()}
