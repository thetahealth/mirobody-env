"""Sealing a semantic run: file digests, and the code state a pilot is pinned to.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as file:
        h = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def code_state() -> dict:
    """A pilot seals its complete judging segment plus not-yet-mounted semantic modules.

    `files` (raw bytes) and `world_sha` are provenance; `semantic` (comments and
    docstrings excluded, INFRA files left out) with `judging_sha16` is what
    `same_judging_code` compares.
    """
    from .anchor import infra_files, judging_fingerprint, world_fingerprint
    from .semantic_source import semantic_bytes
    from tools.make_freeze import JUDGING
    files = sorted(set(JUDGING) | {
        "haenv/semantic_judge.py",
        "haenv/semantic_inputs.py", "haenv/semantic_rubric.py", "haenv/semantic_pipeline.py",
        "registry/semantic_judging.yaml", "haenv/semantic_lean.py", "registry/semantic_judging_v4.yaml",
        "haenv/semantic_atom_correction.py", "registry/semantic_judging_second.yaml",
        "registry/semantic_judging_why.yaml", "registry/semantic_judging_why_lean.yaml",
        "registry/semantic_judging_source.yaml",
    })
    infra = set(infra_files())
    return {"files": {name: digest(ROOT / name) for name in files},
            "semantic": {name: hashlib.sha256(semantic_bytes(ROOT / name)).hexdigest()
                         for name in files if name not in infra},
            "judging_sha16": judging_fingerprint(), "world_sha": world_fingerprint()}
