"""packread.py -- the single entry point for reading packs and batches.

- `field()` raises on a missing key (listing the keys the row has) instead of
  returning `None`.
- `batches()` orders batch directories by parsed timestamp; unparseable or
  future-dated names go to `Scan.anomalies`.
- `is_complete()` reports what a batch is missing.

Usage:
    import packread as P
    for b in P.batches(task_type="joint_dx"):       # already in time order, anomalies excluded
        cs = P.cases(b)                             # {case_id: dict}, read line by line
        for r in P.eval_rows(b):
            cid = P.field(r, "case_id")             # raises rather than returning None

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as _dc_field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Batch directory name format.
_STAMP = re.compile(r"^(\d{8})-(\d{6})$")

#: Keys that may hold the case identifier in `eval.jsonl`.
_CASE_KEYS = ("case_id", "case")


class PackReadError(Exception):
    """A pack or batch could not be read."""


@dataclass(frozen=True)
class Batch:
    task_type: str
    job_id: str
    stamp: str
    dir: Path
    when: datetime

    @property
    def cases_file(self) -> Path:
        return self.dir / "cases.jsonl"

    @property
    def eval_file(self) -> Path:
        return self.dir / "eval.jsonl"

    @property
    def meta_file(self) -> Path:
        return self.dir / "batch.json"

    def __str__(self) -> str:
        return f"{self.task_type}/{self.job_id}/{self.stamp}"


@dataclass
class Scan:
    ok: list[Batch] = _dc_field(default_factory=list)
    #: Unparseable or future-dated directories, for the caller to report
    anomalies: list[tuple[Path, str]] = _dc_field(default_factory=list)


def scan(root: Path | None = None, *, task_type: str | None = None,
         job_id: str | None = None, now: datetime | None = None) -> Scan:
    """Scan `results/`, returning (batches in ascending time order, anomalous directories)."""
    root = root or ROOT
    now = now or datetime.now()
    out = Scan()
    for d in sorted((root / "results").glob("*/*/*")):
        if not d.is_dir():
            continue
        tt, jid, stamp = d.parts[-3], d.parts[-2], d.parts[-1]
        if task_type and tt != task_type:
            continue
        if job_id and jid != job_id:
            continue
        m = _STAMP.match(stamp)
        if not m:
            out.anomalies.append((d, f"批次名不是 YYYYmmdd-HHMMSS:{stamp!r}"))
            continue
        try:
            when = datetime.strptime(stamp, "%Y%m%d-%H%M%S")
        except ValueError:
            out.anomalies.append((d, f"批次名解析失败:{stamp!r}"))
            continue
        if when > now:
            out.anomalies.append((d, f"时间在未来({stamp});不计入「最近批次」"))
            continue
        out.ok.append(Batch(tt, jid, stamp, d, when))
    out.ok.sort(key=lambda b: b.when)
    return out


def batches(root: Path | None = None, **kw) -> list[Batch]:
    """Only the normal ones (already in ascending time order). Use `scan()` to get the anomalies."""
    return scan(root, **kw).ok


def latest(root: Path | None = None, **kw) -> Batch | None:
    """The most recent batch -- by timestamp, not the last one in string sort order."""
    bs = batches(root, **kw)
    return bs[-1] if bs else None


def is_complete(b: Batch) -> tuple[bool, list[str]]:
    """(is the batch complete, what is missing)."""
    missing = []
    if not b.meta_file.is_file():
        missing.append("batch.json 缺失")
    if not b.cases_file.is_file():
        missing.append("cases.jsonl 缺失")
    elif b.cases_file.stat().st_size == 0:
        missing.append("cases.jsonl 是 0 字节")
    if b.eval_file.is_file() and b.cases_file.is_file():
        if b.eval_file.stat().st_mtime < b.cases_file.stat().st_mtime:
            # cases rewritten after grading
            missing.append("cases.jsonl 比 eval.jsonl 新(题包可能被就地重出过)")
    return (not missing), missing


def _rows(p: Path) -> list[dict]:
    """Read JSONL line by line. Reads the whole file, not just the first line."""
    if not p.is_file():
        raise PackReadError(f"{p} not found")
    out = []
    for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise PackReadError(f"{p}:{i} is not valid JSON: {e}") from e
    return out


def cases(b: Batch) -> dict[str, dict]:
    """A pack -> `{case_id: case}`. Read line by line, not just the first line."""
    out = {}
    for r in _rows(b.cases_file):
        out[field(r, "case_id")] = r
    return out


def eval_rows(b: Batch) -> list[dict]:
    return _rows(b.eval_file)


def meta(b: Batch) -> dict:
    if not b.meta_file.is_file():
        raise PackReadError(f"{b}: no batch.json")
    return json.loads(b.meta_file.read_text(encoding="utf-8"))


def field(row: dict, name: str):
    """Get a field, raising `PackReadError` (with the row's keys) if absent.
    `case_id` also accepts `case`.
    """
    if name == "case_id":
        for k in _CASE_KEYS:
            if k in row:
                return row[k]
        raise PackReadError(
            f"no case id in row (tried {_CASE_KEYS}); row keys: {sorted(row)[:12]}")
    if name not in row:
        raise PackReadError(f"field {name!r} not in row; row keys: {sorted(row)[:12]}")
    return row[name]
