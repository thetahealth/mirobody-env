"""make_freeze.py -- writes the frozen code snapshot (a build artifact; do not hand-edit).

    uv run python tools/release/make_freeze.py --revision 3 --why "..." \
        --pack <job>=<batch> ...

Segment fingerprints are semantic (see `_sem`); `packs[*].sha256` is the raw sha256 of the
pack's `cases.jsonl`, first 16 hex chars.
"""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from haenv.locations import (
    data_root, output_root, source_root, source_path, resource_path, resource_directory, repository_root,
)

ROOT = source_root()


def _source_file(rel: str) -> Path:
    return source_path(rel, root=ROOT if ROOT != source_root() else None)


def _resource_file(rel: str) -> Path:
    return ROOT / rel if ROOT != source_root() else resource_path(rel)


def _output_base() -> Path:
    return ROOT if ROOT != source_root() else output_root()


def _anchor_output(existing: Path) -> Path:
    return existing if output_root() == data_root() else resource_directory(
        "anchors", layout="restructured", root=output_root()) / existing.name



def _generation_source(batch_dir: Path, _seen: set | None = None) -> dict:
    """Walk `derived_from` back to the batch that actually generated these cases.

    A restamped batch re-registers with the current `generator_sha`, so only the root
    batch's fingerprint is true. A broken chain returns `chain_ok: False`.
    """
    _seen = _seen or set()
    bj = batch_dir / "batch.json"
    if not bj.is_file():
        return {"root_batch": batch_dir.name, "chain_ok": False,
                "why": "This batch has no batch.json -- the generation source is unknowable"}
    try:
        meta = json.loads(bj.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"root_batch": batch_dir.name, "chain_ok": False, "why": f"Could not read batch.json: {e}"}
    parent = meta.get("derived_from")
    if parent and parent not in _seen:
        pdir = batch_dir.parent / str(parent)
        if pdir.is_dir():
            up = _generation_source(pdir, _seen | {parent})
            up["via"] = [batch_dir.name] + list(up.get("via") or [])
            return up
        return {"root_batch": str(parent), "chain_ok": False,
                "why": f"Ancestor batch directory does not exist ({parent}) -- generation-source chain is broken, cannot trace back",
                "via": [batch_dir.name]}
    return {"root_batch": batch_dir.name, "chain_ok": True,
            "generator_sha": meta.get("generator_sha"),
            "generation_git_sha": meta.get("generation_git_sha") or meta.get("haenv_git_sha"),
            "generator": meta.get("generator"),
            **({"job_sha256": meta["job_sha256"]} if meta.get("job_sha256") else {})}

# Revision readers follow this exact entry, without importing this CLI.
PROVENANCE_DECLARATIONS = "src/haenv/provenance/declarations.py"

from haenv.provenance.declarations import (
    GENERATION,
    MANAGEMENT_JUDGING,
    JUDGING,
    M2_JUDGING,
    PACK2_WORLD,
    PACK2_JUDGING,
    P4_JUDGING,
    P4_WORLD,
    P3_JUDGING,
    PROTOCOL_WORLD,
    P3_WORLD,
    P2_JUDGING,
    P2_WORLD,
    INFRA,
    INFRA_PATH_MOVES,
    INFRA_FORBIDDEN_IMPORTS,
    JUDGING_SHA_EQUIVALENCE,
    WORLD_SHA_EQUIVALENCE,
    JUDGING_SHA_MOVES,
    REGISTRY_YAML_OUT_OF_SCOPE,
    PROVENANCE_MODULES,
    JUDGING_IMPORT_ALLOW,
    PENDING_ANCHOR_REVISION,
    PENDING_ANCHOR,
    ALLOW_SYMBOL_PIN,
    PAYLOAD_YAML_ALLOW,
)

def _sha(p: Path) -> str:
    """Raw-byte fingerprint, for "this file is byte-for-byte what was used" (packs, job yaml)."""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _sem(p: Path) -> str:
    """Semantic fingerprint (`haenv.anchor.semantic_bytes`): comments and docstrings don't count."""
    from haenv.scoring.semantic.source import semantic_bytes
    return hashlib.sha256(semantic_bytes(p)).hexdigest()[:16]


def check_segments_disjoint(generation=GENERATION, judging=JUDGING, infra=INFRA) -> None:
    """Hard check before any anchor is written: INFRA ∩ (GENERATION ∪ JUDGING) = ∅."""
    overlap = sorted(set(infra) & (set(generation) | set(judging)))
    if overlap:
        raise SystemExit(f"🔴 INFRA overlaps a frozen segment: {overlap}")


def _equivalence_rows(prev: dict | None = None) -> dict:
    """`JUDGING_SHA_EQUIVALENCE`, each row recomputed from git objects before it is written.

    Neither the rows nor the INFRA membership they rely on are certified by the revision
    that introduces them: a row takes effect only once a previous anchor recorded it (as
    effective or pending), and the files it drops must be INFRA in that previous anchor
    too. A row the previous anchor does not know is verified and written as pending.
    """
    from haenv.provenance.anchor import verify_judging_equivalence
    from haenv.provenance.anchor import verify_judging_move
    prev = prev or {}
    known = {(r["old"], r["rev"], r["new"])
             for key in ("judging_sha_equivalence", "judging_sha_equivalence_pending")
             for r in prev.get(key) or ()}
    trusted_infra = _trusted_infra_labels(prev)
    out = {"effective": [], "pending": []}
    for row in JUDGING_SHA_EQUIVALENCE:
        if (row["old"], row["rev"], row["new"]) in known:
            out["effective"].append(verify_judging_equivalence(row, infra=trusted_infra))
        else:
            out["pending"].append(verify_judging_equivalence(row))
    for row in JUDGING_SHA_MOVES:
        if (row["old"], row["rev"], row["new"]) in known:
            out["effective"].append({**verify_judging_move(row, infra=trusted_infra), "kind": "move"})
        else:
            out["pending"].append({**verify_judging_move(row), "kind": "move"})
    return out


def _trusted_infra_labels(prev: dict) -> tuple[str, ...]:
    """Previous source labels whose declared current role remains infrastructure."""
    previous = set(prev.get("infra") or ())
    agreed = {INFRA_PATH_MOVES.get(f, f) for f in previous} & set(INFRA)
    labels = {f for f in previous if INFRA_PATH_MOVES.get(f, f) in agreed}
    labels.update(old for old, new in INFRA_PATH_MOVES.items() if new in agreed)
    return tuple(sorted(labels))


def _equivalence_fields(prev: dict) -> dict:
    from haenv.provenance.anchor import verify_world_equivalence
    rows = _equivalence_rows(prev)
    return {"judging_sha_equivalence": rows["effective"],
            "judging_sha_equivalence_pending": rows["pending"],
            "world_sha_equivalence": [verify_world_equivalence(row) for row in WORLD_SHA_EQUIVALENCE]}


def _previous_anchor() -> dict:
    try:
        from haenv.provenance.anchor import freeze_path
        return json.loads(freeze_path().read_text(encoding="utf-8"))
    except Exception:                                           # noqa: BLE001 -- no anchor yet
        return {}


def _write_doc_numbers() -> None:
    """Carry the new anchor into the living documents (`doc_numbers_check --write --run`).

    A fresh process, so nothing cached from the previous anchor is read. The anchor is
    already written when this runs; anything still red is printed for a hand fix.
    """
    repo = repository_root()
    if repo is None:
        raise ValueError("Document maintenance requires a source checkout")
    helper = repo / 'tools/validation/doc_numbers_check.py'
    if not helper.is_file():
        print("[freeze] document-maintenance helper is not included; document numbers were not checked")
        return
    r = subprocess.run([sys.executable, str(helper),
                        "--write", "--run"], cwd=repo, capture_output=True, text=True,
                       timeout=1800)
    keep = ("  wrote", "  🔴")
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith(keep) or "rewritten" in line or line.startswith("共 "):
            print("   " + line.strip())
    if r.returncode:
        print(f"   doc_numbers_check exited {r.returncode} after --write: fix the lines above by hand")


def run_publish_gate(packs: dict, pack_types: dict, assert_publishable, not_publishable, root: Path | None = None) -> int:
    """Run the publish gate over every pack's saved rows. 0 = passed, 2 = rejected or not runnable.

    The pack's directory comes from its declared task type (`pack_types`), never from its name. A
    pack whose batch directory is not there is an error; a batch with no `eval.jsonl` (generated,
    not yet run) has nothing to gate and says so on the console.
    """
    root = _output_base() if root is None else root
    for _job, _v in packs.items():
        _dir = root / "results" / pack_types[_job] / _job / _v["batch"]
        if not _dir.is_dir():
            print(f"🔴 Publish gate cannot run: batch directory {_dir} does not exist")
            return 2
        _ev = _dir / "eval.jsonl"
        if not _ev.exists():
            print(f"⚠️ Publish gate not run for {_job}@{_v['batch']}: no eval.jsonl in the batch")
            continue
        _rows = [json.loads(x) for x in _ev.read_text(encoding="utf-8").split("\n") if x.strip()]
        if not _rows:
            print(f"⚠️ Publish gate not run for {_job}@{_v['batch']}: eval.jsonl has no rows")
            continue
        try:
            assert_publishable(_rows, where=f"make_freeze · {_job}@{_v['batch']}")
        except not_publishable as _e:
            print(f"🔴 Freeze rejected: {_job}@{_v['batch']}\n{_e}")
            print("   To freeze an old pack anyway, pass `--allow-stale` explicitly (it gets written into the snapshot's allow_stale field)")
            return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", type=int, required=True)
    ap.add_argument("--date", default="2026-09-02")
    ap.add_argument("--why", default="", help="Reason this run supersedes the previous version")
    ap.add_argument("--supersedes", type=int, default=None)
    ap.add_argument("--pack", action="append", default=[],
                    metavar="JOB=BATCH", help="repeatable; JOB=batch timestamp")
    ap.add_argument("--generated-pack", action="append", default=[],
                    metavar="JOB=BATCH", help="record an audited generation-pipeline output separately from published Q packs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-stale", action="store_true",
                    help="Skip the publish gate (freeze the batch anyway even with mixed/stale/unstamped judge versions); recorded in the snapshot")
    ap.add_argument("--segments-only", action="store_true",
                    help="Recompute only the two segments' source fingerprints; packs are carried over unchanged from the previous version -- "
                         "doesn't need `results/`, so external contributors can run it too. Must be combined with --supersedes and --why")
    args = ap.parse_args()
    check_segments_disjoint()

    if args.segments_only:
        if args.pack or args.generated_pack:
            print("🔴 `--segments-only` does not accept `--pack` -- packs must either all be inherited or all freshly given")
            return 2
        if args.supersedes is None or not str(args.why).strip():
            print("🔴 `--segments-only` must be combined with `--supersedes <N>` and `--why '...'` -- "
                  "if the segments moved but the packs weren't recomputed, that fact must be recorded")
            return 2
        from haenv.provenance.anchor import AnchorError as _AE
        from haenv.provenance.anchor import freeze_path as _fp
        try:
            _prev = json.loads(_fp().read_text(encoding="utf-8"))
        except (_AE, OSError, ValueError) as _e:
            print(f"🔴 Could not read the previous anchor, so there are no packs to inherit: {type(_e).__name__}: {_e}")
            return 2
        _inherited = _prev.get("packs") or {}
        if not _inherited:
            print("🔴 The previous anchor has zero packs -- inheriting would still produce an empty snapshot, refusing")
            return 2
        _snap = {
            "frozen_at": args.date,
            "revision": args.revision,
            "generated_by": "tools/release/make_freeze.py --segments-only (generated, do not edit)",
            "generation": {f: _sem(_source_file(f)) for f in GENERATION},
            "judging": {f: _sem(_source_file(f)) for f in JUDGING},
            "raw_sha": {f: _sha(_source_file(f)) for f in (*GENERATION, *JUDGING)
                        if _source_file(f).is_file()},
            "infra": {f: _sem(_source_file(f)) for f in INFRA},
            **_equivalence_fields(_prev),
            "packs": _inherited,
            "n_total": _prev.get("n_total", sum(v.get("n", 0) for v in _inherited.values())),
            "segments_only": True,
            "packs_inherited_from": _prev.get("revision"),
            "segments_before": {"judging_sha16": _prev.get("judging_sha16"),
                                "world_sha": _prev.get("world_sha")},
            "supersedes": {"revision": args.supersedes, "why": args.why},
        }
        if _prev.get("allow_stale"):
            _snap["allow_stale"] = True         # inherited packs were never verified; the flag carries over
        if _prev.get("generated_packs"):
            _snap["generated_packs"] = _prev["generated_packs"]
            _snap["n_generated"] = sum(v["n"] for v in _prev["generated_packs"].values())
        _dst = Path(args.out) if args.out else _anchor_output(_fp())
        _dst.parent.mkdir(parents=True, exist_ok=True)
        _dst.write_text(json.dumps(_snap, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"rev{args.revision} · segment fingerprints recomputed only · {len(_inherited)} packs "
              f"{_snap['n_total']} cases inherited from rev{_prev.get('revision')} → {_dst}")
        print("   ⚠️ This version's packs were not recomputed under the current code -- the snapshot's `segments_only: true` records this")
        if not args.out:
            _write_doc_numbers()
        return 0

    packs: dict[str, dict] = {}
    generated_packs: dict[str, dict] = {}
    pack_types: dict[str, str] = {}
    missing = []
    from haenv.app.artifacts.jobmeta import pack_task_type
    for spec, generated in [(s, False) for s in args.pack] + [(s, True) for s in args.generated_pack]:
        job, _, batch = spec.partition("=")
        if job in pack_types:
            print(f"🔴 Pack {job!r} is declared more than once")
            return 2
        try:
            pack_types[job] = pack_task_type(_output_base(), job, batch)
        except ValueError as _e:
            print(f"🔴 {_e}")
            return 2
        f = _output_base() / "results" / pack_types[job] / job / batch / "cases.jsonl"
        if not f.exists():
            missing.append(str(f))
            continue
        _jy = _resource_file(f"inputs/{job}.job.yaml")
        target = generated_packs if generated else packs
        source = _generation_source(f.parent)
        target[job] = {"batch": batch,
                      "n": sum(1 for _ in f.open(encoding="utf-8", errors="ignore")),
                      "sha256": _sha(f),
                      "job_yaml_sha256": source.get("job_sha256") if generated else
                                         (_sha(_jy) if _jy.exists() else None),
                      "generation_source": source}
    if missing:
        for m in missing:
            print(f"🔴 Batch does not exist: {m}")
        return 2

    if not packs:
        print("🔴 No `--pack` given at all -- freezing an empty snapshot would overwrite the authoritative anchor, and the publish gate would iterate zero times")
        print("   Usage: --pack JOB=BATCH (repeatable); see `haenv.anchor.frozen_packs()` for the current packs")
        return 2

    snap = {
        "frozen_at": args.date,
        "revision": args.revision,
        "generated_by": "tools/release/make_freeze.py (generated, do not edit)",
        "generation": {f: _sem(_source_file(f)) for f in GENERATION},
        "judging": {f: _sem(_source_file(f)) for f in JUDGING},
        "raw_sha": {f: _sha(_source_file(f)) for f in (*GENERATION, *JUDGING)
                    if _source_file(f).is_file()},
        "infra": {f: _sem(_source_file(f)) for f in INFRA},
        **_equivalence_fields(_previous_anchor()),
        "packs": packs,
        "n_total": sum(v["n"] for v in packs.values()),
    }
    from haenv.provenance.judging import criterion_fingerprints
    snap["judging_criteria"] = criterion_fingerprints()
    if generated_packs:
        snap["generated_packs"] = generated_packs
        snap["n_generated"] = sum(v["n"] for v in generated_packs.values())
    if args.supersedes is not None:
        snap["supersedes"] = {"revision": args.supersedes, "why": args.why}

    if not args.allow_stale:
        from haenv.provenance.judging_provenance import NotPublishable
        from haenv.provenance.judging_provenance import assert_publishable
        _rc = run_publish_gate({**packs, **generated_packs}, pack_types, assert_publishable, NotPublishable)
        if _rc:
            return _rc
    else:
        snap["allow_stale"] = True      # record the override: this snapshot was never verified

    from haenv.provenance.anchor import AnchorError
    from haenv.provenance.anchor import freeze_path
    if args.out:
        dst = Path(args.out)
    else:
        try:
            dst = _anchor_output(freeze_path())
        except AnchorError:
            dst = resource_directory("anchors", layout="restructured", root=_output_base()) / f"freeze-{args.date}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rev{args.revision} · {len(packs)} packs · {snap['n_total']} cases → {dst}")
    for job, v in packs.items():
        print(f"   {job:<20}{v['batch']}  {v['n']:>3} cases  {v['sha256']}")
    if generated_packs:
        print(f"   Generation outputs: {len(generated_packs)} packs · {snap['n_generated']} cases")
        for job, v in generated_packs.items():
            print(f"   {job:<20}{v['batch']}  {v['n']:>3} cases  {v['sha256']}")
    if not args.out:
        _write_doc_numbers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
