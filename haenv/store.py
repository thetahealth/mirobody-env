"""store.py -- persistence and reload of the emitted case bodies of a batch.

`cases.jsonl` holds the fully assembled cases that cleared the emission gate, including
verifier-only ground truth. `run` and `report` read this copy rather than regenerating, so
a report always shows the case that was evaluated; `--rebuild` regenerates it.
This is separate from `cases/_llm_cache/`, which caches raw model output.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from haenv_kernel.schema import RawCase          # kernel

log = logging.getLogger("haenv.store")


def _case_line(item) -> tuple[str, str]:
    """`(world digest, stamped JSON Lines row)` for one `(case_id, RawCase)`."""
    from . import canary, wq
    cid, raw = item
    case = asdict(raw)
    world = json.dumps({"case_id": cid, "case": case},
                       ensure_ascii=False, separators=(",", ":"))
    return (hashlib.sha256(world.encode()).hexdigest()[:16],
            canary.stamp_line({"case_id": cid, "case": case,
                               "question": {"injected_manifest":
                                            wq.injected_manifest(cid, required=False)}}))


def save_cases(path: Path, built: dict[str, RawCase], workers: int = 1) -> tuple[Path, dict]:
    """Persist the cases and return per-case sha256 digests (recorded in batch.json).

    Each line is `{case_id, case, question}`: `case` is the world W, `question` the Q-side
    injection ledger (see `wq`). The digest covers `{case_id, case}` only, so it answers
    "did the world change". The canary field is added after hashing and moves no digest.
    Rows are serialised on `workers` processes and written in `built` order.
    """
    from .build_pool import map_ordered
    path.parent.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    lines = map_ordered(_case_line, built.items(), workers)
    with open(path, "w", encoding="utf-8") as f:
        for cid, (digest, line) in zip(built, lines):
            digests[cid] = digest
            f.write(line + "\n")
    log.info("[store] persisted %d case(s) -> %s (%.1f KB)", len(built), path,
             path.stat().st_size / 1024)
    return path, digests


#: Build-time audit, one row per case, emitted or not: per-item verdicts, drop reasons,
#: gate hits and synthesis rounds.
AUDIT_FILENAME = "audit.jsonl"


def save_audits(path: Path, audits: list[dict]) -> int:
    """Persist the per-case audits next to `cases.jsonl`. Values that are not JSON
    (sets, dataclasses) are written through `str`, so a row is never dropped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a in audits:
            f.write(json.dumps(a, ensure_ascii=False, default=str) + "\n")
    return len(audits)


def load_audits(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def merge_recorded_audits(stubs: list[dict], recorded: list[dict],
                          order: list[str]) -> list[dict]:
    """Replace read-back stubs with the audit `build` recorded, where one exists.

    A record replaces its stub only when both agree on `emitted`; keys starting with `_` are
    left out. Rows follow `order`.
    """
    rec = {r.get("case_id"): r for r in recorded}
    out = []
    for s in stubs:
        r = rec.get(s["case_id"])
        if r is None or bool(r.get("emitted")) != bool(s.get("emitted")):
            out.append(s)
            continue
        m = {**s, **{k: v for k, v in r.items() if not k.startswith("_")},
             "from_batch": True, "audit_source": AUDIT_FILENAME}
        if "premise_ok" in r and "premise_error" not in r:
            m.pop("premise_error", None)
        out.append(m)
    pos = {cid: i for i, cid in enumerate(order)}
    return sorted(out, key=lambda x: pos.get(x["case_id"], len(pos)))


def digest_of(built: dict[str, RawCase]) -> dict[str, str]:
    """Per-case digests without writing to disk, for checking a reloaded batch."""
    return {cid: hashlib.sha256(json.dumps(
        {"case_id": cid, "case": asdict(raw)}, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()[:16] for cid, raw in built.items()}


def load_cases(path: Path) -> dict[str, RawCase]:
    """Reload W, and register the Q-side injection ledger back into `wq`.

    In pre-layering batches the ledger lives in `case.adjudication.injected_event_manifest`;
    it is registered but left in W, so the world's digest does not change.
    """
    from . import wq
    out: dict[str, RawCase] = {}
    if not path.is_file():
        return out
    legacy = 0
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            raw = RawCase(**rec["case"])
            out[rec["case_id"]] = raw
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.error("[store] %s has a line that could not be read back (%s); that case will be regenerated", path, e)
            continue
        man = ((rec.get("question") or {}).get("injected_manifest")
               or (raw.adjudication or {}).get("injected_event_manifest"))
        if (rec.get("question") or {}).get("injected_manifest") is None and man:
            legacy += 1
        if man:
            wq.register_injection(rec["case_id"], man)
    if legacy:
        log.warning("[store] %d case(s) are from a pre-layering batch (the ledger "
                    "is still inside W.adjudication): registered on the Q side for read-back, "
                    "but its world fingerprint is not comparable with the current pack", legacy)
    if out:
        log.info("[store] read back %d case(s) from the batch (not regenerating) -> %s", len(out), path)
    return out


def check_idempotent(built: dict, recorded: dict | None) -> list[dict]:
    """Idempotency gate: cases rebuilt from the same job must match the recorded per-case sha256.

    Returns the per-case differences; an empty list means idempotent. A difference is not an
    error -- it says the old batch's conclusions cannot be read side by side with a new one.
    Use `drift_scope` to tell question drift (responses void) from ground-truth-only drift
    (recompute scores with `recompute_judges --full`).
    """
    if not recorded:
        return []
    now = digest_of(built)
    out: list[dict] = []
    for cid in sorted(set(now) | set(recorded)):
        a, b = now.get(cid), recorded.get(cid)
        if a != b:
            out.append({"case": cid, "recorded": b, "rebuilt": a,
                        "kind": "missing_in_batch" if b is None else
                                "missing_in_rebuild" if a is None else "drifted"})
    return out


def drift_scope(built: dict, prev_cases_file) -> dict:
    """Classify drift as "question" (responses void) or "truth_only" (recompute scores).

    Needs the previous batch's `cases.jsonl`; without it the result is `undecidable`.
    """
    from pathlib import Path as _P
    p = _P(prev_cases_file)
    if not p.is_file():
        return {"scope": "undecidable", "why": f"the previous batch's cases.jsonl is not available ({p.name})"}
    from haenv_kernel.build import build_instance                    # kernel
    prev = load_cases(p)
    q_changed, t_changed, common = [], [], []
    for cid, raw in built.items():
        old = prev.get(cid)
        if old is None:
            continue
        common.append(cid)
        try:
            t = int(raw.prediction_context["prediction_time_T"])
            spa, vpa = build_instance(old, t)
            spb, vpb = build_instance(raw, t)
        except Exception:  # cannot rebuild -> "cannot decide", not "unchanged"
            return {"scope": "undecidable", "why": f"{cid} failed to rebuild"}
        if _q_fingerprint(spa) != _q_fingerprint(spb):
            q_changed.append(cid)
        elif _t_fingerprint(vpa, old) != _t_fingerprint(vpb, raw):
            t_changed.append(cid)
    return {"scope": ("question" if q_changed else "truth_only" if t_changed else "none"),
            "n_common": len(common), "question_changed": q_changed,
            "truth_only_changed": t_changed,
            "why": ("the question text changed -> persisted responses are void, must re-run"
                    if q_changed else
                    "the question text is byte-identical, only the truth side changed -> "
                    "responses are still valid, just recompute the score"
                    if t_changed else
                    "neither fingerprint changed -- if the per-case sha256 genuinely did "
                    "change, that means this function's fingerprint coverage is "
                    "incomplete (not \"nothing changed\"), fix the coverage before judging "
                    "again")}


def _q_fingerprint(sp) -> str:
    """Fingerprint of the part visible to the solver."""
    import hashlib
    import json as _j
    return hashlib.sha256(_j.dumps(
        {"ld": sp.longitudinal_data, "ev": sp.evidence_ledger,
         "up": sp.user_profile, "pc": sp.prediction_context},
        sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def _t_fingerprint(vp, raw=None) -> str:
    """Fingerprint of the verifier side (ground truth, post-T data, `latent_premise`).

    Every verifier-side field must be covered, or a changed case sha256 would report scope `none`.
    """
    import hashlib
    import json as _j
    return hashlib.sha256(_j.dumps(
        {"fd": getattr(vp, "future_data", None), "adj": getattr(vp, "adjudication", None),
         "ol": getattr(vp, "outcome_label", None), "gd": getattr(vp, "gold_drivers", None),
         "lp": getattr(raw, "latent_premise", None) if raw is not None else None,
         "rp": getattr(vp, "reversal_points", None), "lr": getattr(vp, "label_rule", None)},
        sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
