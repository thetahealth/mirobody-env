"""slicing.py -- per-case slicing is frozen into the batch (`slices.json`).

Offline stubs re-cut slices on every recompute while real models keep the
slicing of their persisted response, so a change in the slicing code would give
them different denominators. The slicing is therefore written to the batch and
read back: what is on disk wins, a case missing from a frozen table fails the
run, and missing cases are appended, never overwritten.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import json
import logging
import pathlib

log = logging.getLogger("haenv.slicing")

FILENAME = "slices.json"


class SlicesMissing(RuntimeError):
    """A slicing table exists on disk, and this case is not in it."""


def path_of(batch_dir) -> pathlib.Path:
    return pathlib.Path(batch_dir) / FILENAME


def recover_from_disk(batch_dir) -> dict[str, list[int]]:
    """Recover per-case slicing from this batch's `eval.jsonl` (`slice_rows[].t`) and `responses.jsonl` (`slice_t`).

    A case whose rows disagree is left out of the result (the caller fails it),
    as is a case that cannot be recovered; neither becomes an empty list.
    """
    d = pathlib.Path(batch_dir)
    seen: dict[str, set[tuple[int, ...]]] = {}

    ev = d / "eval.jsonl"
    if ev.is_file():
        for line in ev.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows = r.get("slice_rows")
            cid = str(r.get("case") or "")
            if not cid or not isinstance(rows, list) or not rows:
                continue
            ts = tuple(int(x["t"]) for x in rows if isinstance(x, dict) and x.get("t") is not None)
            if ts:
                seen.setdefault(cid, set()).add(ts)

    rp = d / "responses.jsonl"
    if rp.is_file():
        per: dict[str, set[int]] = {}
        for line in rp.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            cid, st = str(r.get("case") or ""), r.get("slice_t")
            if cid and st is not None:
                per.setdefault(cid, set()).add(int(st))
        for cid, ts in per.items():
            if cid not in seen and ts:
                seen[cid] = {tuple(sorted(ts))}

    out: dict[str, list[int]] = {}
    for cid, variants in seen.items():
        if len(variants) == 1:
            out[cid] = list(next(iter(variants)))
        else:
            log.error("[slicing] %s disagrees on how it was sliced across this batch's own "
                      "rows (%d variant(s): %s) -- not recovering it, left for the caller to "
                      "fail rather than resolved by majority vote.",
                      cid, len(variants), sorted(variants)[:3])
    return out


def save(batch_dir, by_case: dict[str, list[int]], *, spec: str) -> pathlib.Path:
    """Write per-case slicing into the batch, appending missing cases and never overwriting existing ones.

    Also records `spec` (the value of `job.slices`), checked by `check_spec()`.
    """
    p = path_of(batch_dir)
    cur: dict[str, list[int]] = {}
    cur_spec = ""
    if p.is_file():
        got = load(batch_dir)
        if got is not None:
            cur, cur_spec = got
    added = {str(k): [int(x) for x in v] for k, v in by_case.items() if str(k) not in cur}
    if p.is_file() and not added:
        return p
    merged = {**cur, **added}
    p.write_text(json.dumps({"spec": str(cur_spec or spec), "by_case": dict(sorted(merged.items()))},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    if cur:
        log.info("[slicing] slicing table appended: %d case(s) (%d existing case(s) left untouched) -> %s",
                 len(added), len(cur), p.name)
    else:
        log.info("[slicing] slicing frozen into the batch: %d case(s) · spec=%s -> %s", len(merged), spec, p.name)
    return p


def check_spec(batch_dir, spec: str) -> str | None:
    """Return the on-disk slicing convention if it differs from `spec`, else `None`."""
    got = load(batch_dir)
    if got is None:
        return None
    _by, was = got
    return None if (not was or was == str(spec)) else was


def load(batch_dir) -> tuple[dict[str, list[int]], str] | None:
    """Read back `(by_case, spec)`; `None` when the batch has no slicing table (an empty table means zero sliced cases)."""
    p = path_of(batch_dir)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        # An unreadable file raises rather than being treated as absent.
        raise SlicesMissing(f"{p} could not be read: {type(e).__name__} {e}") from e
    return ({str(k): [int(x) for x in v] for k, v in (d.get("by_case") or {}).items()},
            str(d.get("spec") or ""))


def resolve(batch_dir, case_id: str, compute) -> list[int]:
    """This case's slicing: the on-disk value, or `compute()` when the batch has no slicing table.

    Raises `SlicesMissing` when a table exists but lacks this case.
    """
    got = load(batch_dir)
    if got is None:
        return list(compute())
    by_case, _spec = got
    if case_id not in by_case:
        raise SlicesMissing(
            f"{case_id} is not in {FILENAME} (this batch has slicing frozen, "
            f"{len(by_case)} case(s) total). "
            f"Not falling back to a live computation: a live-computed slice count "
            f"could differ from the rest of the batch, and any cross-slice counting "
            f"dimension would then get two different denominators.")
    return list(by_case[case_id])


def drift(batch_dir, case_id: str, computed: list[int]) -> list[int] | None:
    """The on-disk value when it differs from `computed`, else `None`. Reports only."""
    got = load(batch_dir)
    if got is None:
        return None
    by_case, _ = got
    was = by_case.get(case_id)
    if was is None or list(was) == list(computed):
        return None
    return list(was)


def census(batch_dir) -> dict:
    """This batch's slicing readout, for reports and checks."""
    got = load(batch_dir)
    if got is None:
        return {"frozen": False, "n_cases": 0, "spec": None, "n_slices": {}}
    by_case, spec = got
    return {"frozen": True, "n_cases": len(by_case), "spec": spec,
            "n_slices": {k: len(v) for k, v in sorted(by_case.items())}}
