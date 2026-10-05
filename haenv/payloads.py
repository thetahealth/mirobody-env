"""Persist `build_instance`'s output `(sp, vp)` alongside a batch.

Recompute reads these payloads from disk instead of rebuilding them with the current
`build.py`, so a re-score is judged against the same question and gold that were evaluated.
Payloads are built at the canonical T only (slice geometries are not persisted); a batch
without the file falls back to rebuilding and reports it.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

log = logging.getLogger("haenv.payloads")

FILENAME = "payloads.jsonl"


def canonical_T(raw) -> int:
    """Canonical T, derived exactly as `tools/recompute_judges.py` derives it."""
    return int(raw.prediction_context["prediction_time_T"])


def build_pairs(built: dict) -> dict[str, tuple]:
    from haenv_kernel.build import build_instance                      # kernel
    out: dict[str, tuple] = {}
    for cid, raw in built.items():
        out[cid] = build_instance(raw, canonical_T(raw))
    return out


def _payload_line(item) -> tuple[str, str]:
    """`(payload digest, stamped JSON Lines row)` for one `(case_id, RawCase)`."""
    from haenv_kernel.build import build_instance                      # kernel
    from . import canary
    cid, raw = item
    T = canonical_T(raw)
    sp, vp = build_instance(raw, T)
    rec = {"case_id": cid, "T": T, "sp": asdict(sp), "vp": asdict(vp)}
    blob = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16], canary.stamp_line(rec)


def save_payloads(path: Path, built: dict, workers: int = 1) -> tuple[Path, dict]:
    """Persist `(sp, vp)` and compute a per-case fingerprint.

    The fingerprint covers `{case_id, T, sp, vp}` (unlike `store.digest_of`, which covers `raw`
    only). The canary field is added after hashing, so it moves no digest. Rows are built on
    `workers` processes and written in `built` order.
    """
    from .build_pool import map_ordered
    path.parent.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    lines = map_ordered(_payload_line, built.items(), workers)
    with open(path, "w", encoding="utf-8") as f:
        for cid, (digest, line) in zip(built, lines):
            digests[cid] = digest
            f.write(line + "\n")
    log.info("[payloads] persisted %d case(s) -> %s (%.1f KB)", len(lines), path,
             path.stat().st_size / 1024)
    return path, digests


def load_payloads(path: Path) -> dict[str, tuple]:
    """Read back `{cid: (SolverPayload, VerifierPayload)}`.

    A missing field raises; no default is substituted, since an empty ledger would read as
    "the model cited nothing".
    """
    from haenv_kernel.schema import SolverPayload, VerifierPayload      # kernel
    out: dict[str, tuple] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        try:
            sp = SolverPayload(**rec["sp"])
            vp = VerifierPayload(**rec["vp"])
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"{path.name}: {rec.get('case_id')}'s payload does not match the current schema "
                f"({type(e).__name__}: {e}) -- no default is substituted. "
                f"This usually means the kernel schema changed and this payload is stale; "
                f"the remedy is to regenerate this batch's payloads, not to let a judge run "
                f"on an incomplete input") from e
        out[rec["case_id"]] = (sp, vp)
    return out


def payload_source(batch_dir: Path) -> tuple[str, Path | None]:
    """Where this batch's payloads come from: `("disk", path)` or `("rebuilt", None)`."""
    p = Path(batch_dir) / FILENAME
    return ("disk", p) if p.is_file() else ("rebuilt", None)
