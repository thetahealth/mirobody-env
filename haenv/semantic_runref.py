"""Finding the sealed semantic run a manifest or a view points at.

A run reference is a path (absolute, or relative to the file naming it and then to the repo root)
and, optionally, the run's identity: `run_id`, `tasks_sha256`, `manifest_sha256`. When the path is
not there, the run is found again by identity under the search roots; nothing that is not exactly
one run with that identity is accepted. Sealed run files are only ever read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("run_id", "tasks_sha256", "manifest_sha256")
ENV_ROOTS = "HAENV_SEMANTIC_RUN_ROOTS"
SEARCH_DEPTH = 3


def _digest(path: Path) -> str:
    from .semantic_pipeline import digest
    return digest(path)


def identity_problems(run: Path, want: dict) -> list[str]:
    """Why `run` is not the run `want` names (empty when it is)."""
    manifest_path = Path(run) / "manifest.json"
    if not manifest_path.is_file():
        return ["no manifest.json"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError:
        return ["manifest.json unreadable"]
    problems = []
    if want.get("run_id") is not None and manifest.get("run_id") != want["run_id"]:
        problems.append(f"run_id {manifest.get('run_id')} != {want['run_id']}")
    if want.get("manifest_sha256") is not None and _digest(manifest_path) != want["manifest_sha256"]:
        problems.append("manifest.json bytes differ from the recorded digest")
    if want.get("tasks_sha256") is not None:
        tasks = Path(run) / "tasks.jsonl"
        if manifest.get("tasks_sha256") != want["tasks_sha256"]:
            problems.append("manifest tasks_sha256 differs")
        elif not tasks.is_file() or _digest(tasks) != want["tasks_sha256"]:
            problems.append("tasks.jsonl bytes differ from the recorded digest")
    return problems


def _search(roots: list[Path], want: dict) -> list[Path]:
    hits, seen = [], set()
    for top in roots:
        top = Path(top)
        if not top.is_dir():
            continue
        base_depth = len(top.resolve().parts)
        for here, dirs, files in os.walk(top):
            if "manifest.json" in files:
                dirs[:] = []                       # a run: never descend into its samples/results
                real = Path(here).resolve()
                if real not in seen:
                    seen.add(real)
                    if not identity_problems(real, want):
                        hits.append(real)
                continue
            if len(Path(here).resolve().parts) - base_depth >= SEARCH_DEPTH:
                dirs[:] = []
    return hits


def resolve_run(ref, *, base_dirs=(), search_roots=(), want: dict | None = None) -> Path:
    """The run directory `ref` names, verified against `want` (identity fields).

    `ref` is a path string, or a mapping `{"path": ..., "run_id": ..., "tasks_sha256": ...}`
    whose identity fields extend `want`. A relative path is tried against each of `base_dirs`,
    then the repo root. A path that exists must be that run (else refused, no silent
    fall-through to another copy). A path that does not exist is looked up by identity under
    `search_roots` and `$HAENV_SEMANTIC_RUN_ROOTS`: exactly one run may match.
    """
    want = dict(want or {})
    if isinstance(ref, dict):
        want.update({k: ref[k] for k in IDENTITY if ref.get(k) is not None})
        ref = ref.get("path")
    if not ref:
        raise ValueError("A semantic run reference has no path")
    path = Path(str(ref))
    tried = [path] if path.is_absolute() else [Path(b) / path for b in (*base_dirs, ROOT)]
    for candidate in tried:
        if (candidate / "manifest.json").is_file():
            problems = identity_problems(candidate, want)
            if problems:
                raise ValueError(f"Semantic run at {candidate} is not the run that was referenced "
                                 f"({'; '.join(problems)}); refused")
            return candidate.resolve()
    if not any(want.get(k) is not None for k in IDENTITY):
        raise ValueError(f"Semantic run {ref} not found and the reference carries no run_id / "
                         "tasks_sha256 to find it by content")
    roots = [Path(r) for r in search_roots]
    roots += [Path(r) for r in os.environ.get(ENV_ROOTS, "").split(os.pathsep) if r]
    hits = _search(roots, want)
    if len(hits) != 1:
        raise ValueError(f"Semantic run {ref}: {len(hits)} runs with identity "
                         f"{ {k: want[k] for k in IDENTITY if want.get(k) is not None} } "
                         f"under {[str(r) for r in roots]}; "
                         + ("ambiguous, refused" if hits else "not found"))
    return hits[0]


def correction_base(run: Path, correction: dict) -> Path:
    """The base run of a correction manifest's `correction` block, found by path or by content."""
    run = Path(run)
    return resolve_run(correction["base_run"], base_dirs=[run],
                       search_roots=[run.resolve().parent],
                       want={"manifest_sha256": correction.get("base_manifest_sha256"),
                             "tasks_sha256": correction.get("base_tasks_sha256")})


def subset_base(run: Path, manifest: dict) -> Path:
    """The run a subset manifest was cut from."""
    run = Path(run)
    sampling = manifest["sampling"]
    return resolve_run(sampling["base_run"], base_dirs=[run],
                       search_roots=[run.resolve().parent],
                       want={"manifest_sha256": sampling.get("base_manifest_sha256"),
                             "tasks_sha256": sampling.get("base_tasks_sha256")})


def view_run(spec_path: Path, entry) -> Path:
    """A `chain_tip` / `corrections` entry of a declared view, relative to the view file."""
    spec_path = Path(spec_path).resolve()
    return resolve_run(entry, base_dirs=[spec_path.parent], search_roots=[spec_path.parent])
