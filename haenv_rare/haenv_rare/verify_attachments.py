"""haenv-rare-verify -- re-hash every attachment a rare job file points at.

    uv run haenv-rare-verify inputs/rare_coding-p6-zh.job.yaml [more jobs] [--manifest SHA256SUMS]

A pointer is any mapping under a case's `latent.rare_attachments` that carries both `path` and
`sha256` (genome files, PED, DICOM series, EEG recordings, case report). Exit codes: 0 all exist and
match, 1 a file is missing or differs, 2 a job has no pointers at all (nothing checked is not a
pass). `--manifest` writes `sha256  path` lines (each file once, sorted), the format
`sha256sum -c` reads. Run it before a remote run: a copy that is the right size but not the
right bytes (seen 2026-09-23 with DICOM zips) fails here instead of in the middle of a run.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import yaml


def _pointers(node, where: str):
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            yield where, node["path"], node["sha256"]
        for k, v in node.items():
            yield from _pointers(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _pointers(v, f"{where}[{i}]")


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_job(path: str | pathlib.Path) -> dict:
    doc = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    seen: dict[str, str] = {}
    bad: list[tuple[str, str, str]] = []
    checked = 0
    for c in doc.get("cases") or []:
        att = (c.get("latent") or {}).get("rare_attachments") or {}
        for where, p, sha in _pointers(att, f"{c.get('case_id')}.rare_attachments"):
            checked += 1
            if p not in seen:
                f = pathlib.Path(p)
                seen[p] = _sha256(f) if f.is_file() else ""
            if not seen[p]:
                bad.append((where, p, "missing"))
            elif seen[p] != sha:
                bad.append((where, p, "sha_mismatch"))
    return {"job": str(path), "checked": checked, "files": {p: s for p, s in seen.items() if s}, "bad": bad}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="haenv-rare-verify")
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--manifest", default="", help="write `sha256  path` lines for every file checked")
    a = ap.parse_args(argv)
    rc, files = 0, {}
    for j in a.jobs:
        rep = verify_job(j)
        files.update(rep["files"])
        if not rep["checked"]:
            print(f"{j}: no attachment pointers found -- nothing checked", file=sys.stderr)
            rc = max(rc, 2)
            continue
        for where, p, kind in rep["bad"]:
            print(f"{j}: {kind}: {where} -> {p}", file=sys.stderr)
        if rep["bad"]:
            rc = max(rc, 1)
        print(f"{j}: {rep['checked']} pointers, {len(rep['files'])} distinct files, {len(rep['bad'])} bad")
    if a.manifest:
        pathlib.Path(a.manifest).write_text("".join(f"{s}  {p}\n" for p, s in sorted(files.items())), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
