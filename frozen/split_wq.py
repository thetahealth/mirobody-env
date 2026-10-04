#!/usr/bin/env python3
"""Splits each frozen case pack into `Q` (the prompt half, distributed) and `W`
(the ground-truth half, not distributed), and checks both.

    # import a frozen pack from the batch that holds it (writes <job>.cases.jsonl and
    # its MANIFEST entry; refuses a batch the freeze anchor does not record)
    uv run python frozen/split_wq.py --import ddx-timeline=results/joint_dx/ddx-timeline/<batch> \
        --revision <freeze revision>

    # split (rebuilds *.Q.jsonl / *.W.jsonl -- both are build artifacts, do not hand-edit)
    uv run python frozen/split_wq.py --split

    # check the published half alone (no W needed): every row stamped, no point after T,
    # MANIFEST entries agree with the Q files and with the freeze anchor
    uv run python frozen/split_wq.py --check

    # join (rebuilds the original *.cases.jsonl from Q+W, and checks it against
    # MANIFEST's file_sha256 and stamped_sha256)
    uv run python frozen/split_wq.py --join --verify

Rules:

1. The pack list is `MANIFEST.json`: every top-level entry with a `file` key is a
   pack. Files on disk that no entry names are ignored, so a stale local
   `*.cases.jsonl` is never split or joined.
2. The W field list is read from `haenv.wq.W_TRUTH_FIELDS` / `Q_TRUTH_FIELDS`, never copied.
3. The split is lossless: `--join` reproduces `<job>.cases.jsonl` byte for byte
   (`file_sha256`), and stamping the result with the canary field reproduces the
   batch file the freeze anchor records (`stamped_sha256`).
4. `*.cases.jsonl` and `*.W.jsonl` stay on disk, excluded from distribution.
5. Each Q row carries the canary strings under `canary.FIELD`, beside `case` (never inside
   it); `*.cases.jsonl` carries none, and `join` strips the field again.
6. Q stops at T: points and evidence after T move to W with their original index, using the
   kernel's own `_filter_le_T`. `--check` verifies this on Q alone.

Serialization: `json.dumps(o, ensure_ascii=False, separators=(",", ":"))` plus a newline.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import yaml                                           # noqa: E402

from haenv import canary as C                       # noqa: E402  <- the strings, defined once
from haenv.wq import Q_TRUTH_FIELDS, W_TRUTH_FIELDS  # noqa: E402  <- single source of truth, see rule 2 above
from haenv_kernel.build import _filter_le_T          # noqa: E402  <- kernel truncation rule, see rule 6 above

#: W-side record of `case`'s original key order.
ORDER_KEY = "_case_key_order"

#: W-side stash of the gold keys stripped from the Q-side ledger, and their key order.
QMAN_KEY = "_q_manifest_truth"
QMAN_ORDER_KEY = "_q_manifest_key_order"

#: Post-T points / evidence rows moved to W as `[[index, item], ...]` (rule 6).
POST_T_SERIES_KEY = "_post_T_series"
POST_T_EVIDENCE_KEY = "_post_T_evidence"

#: A missing `source_timestamp` counts as after T, as in `build.build_instance`.
_EV_TS_DEFAULT = 10**9

#: Keys of one MANIFEST pack entry, in the order they are written.
ENTRY_KEYS = ("job", "freeze_revision", "n_cases", "q_file", "file", "file_sha256",
              "stamped_sha256", "job_yaml_sha256", "per_case_sha256")


def _T(case: dict) -> int:
    return int(case["prediction_context"]["prediction_time_T"])


def post_T_violations(row: dict) -> list[str]:
    """Everything in one Q row that lies after that row's T. Empty = clean.

    Reads the row the way a reader of the published file would, with no W half.
    A row whose T cannot be read is a violation, not a pass.
    """
    case = row.get("case") or {}
    cid = row.get("case_id") or case.get("case_id")
    try:
        T = _T(case)
    except (KeyError, TypeError, ValueError):
        return [f"{cid}: prediction_time_T unreadable"]
    out = []
    for sig, pts in (case.get("longitudinal_data") or {}).items():
        bad = [p.get("ts") if isinstance(p, dict) else p for p in pts
               if not isinstance(p, dict) or p.get("ts") is None or p["ts"] > T]
        if bad:
            out.append(f"{cid}: {sig} has {len(bad)} point(s) after T={T} (e.g. ts={bad[0]})")
    for e in case.get("evidence_ledger") or ():
        if e.get("source_timestamp", _EV_TS_DEFAULT) > T:
            out.append(f"{cid}: evidence {e.get('evidence_id')} after T={T}")
    return out


def check_q(q_path: pathlib.Path) -> tuple[int, list[str]]:
    """(rows read, problems) for one published Q half: every row must carry the
    canary field, keep it out of the case body, and hold nothing after T."""
    n, bad = 0, []
    for line in q_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        row = json.loads(line)
        cid = row.get("case_id") or (row.get("case") or {}).get("case_id")
        if not C.row_has_canary(row):
            bad.append(f"{cid}: no canary field `{C.FIELD}`")
        if not C.canary_outside_the_case(row):
            bad.append(f"{cid}: canary string inside the case body")
        bad.extend(post_T_violations(row))
    return n, bad


def _dump(o: dict) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _manifest(here: pathlib.Path | None = None) -> dict:
    return json.loads(((here or HERE) / "MANIFEST.json").read_text(encoding="utf-8"))


def pack_entries(man: dict | None = None) -> dict[str, dict]:
    """`{job: entry}` for every MANIFEST entry that names a pack file (rule 1)."""
    man = _manifest() if man is None else man
    return {k: v for k, v in man.items() if isinstance(v, dict) and v.get("file")}


def _packs() -> list[pathlib.Path]:
    """The `*.cases.jsonl` paths MANIFEST names, whether or not they are on disk."""
    return [HERE / e["file"] for _job, e in sorted(pack_entries().items())]


def split(pack: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, int, int]:
    """Write `<stem>.Q.jsonl` and `<stem>.W.jsonl`. Returns
    (q, w, row count, number of fields stripped)."""
    stem = pack.name[: -len(".cases.jsonl")]
    q_path = HERE / f"{stem}.Q.jsonl"
    w_path = HERE / f"{stem}.W.jsonl"
    q_lines, w_lines, stripped = [], [], 0
    for line in pack.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = C.strip_row(json.loads(line))
        case = row["case"]
        w_rec = {"case_id": row.get("case_id") or case.get("case_id"),
                 ORDER_KEY: list(case.keys())}
        for f in case.keys():
            if f in W_TRUTH_FIELDS:
                w_rec[f] = case[f]
                stripped += 1
        for f in list(case.keys()):
            if f in W_TRUTH_FIELDS:
                del case[f]
        # Also strip the gold keys from the Q-side ledger (`question.injected_manifest`).
        _man = ((row.get("question") or {}).get("injected_manifest")
                if isinstance(row.get("question"), dict) else None)
        if isinstance(_man, dict):
            _hit = [k for k in _man if k in Q_TRUTH_FIELDS]
            if _hit:
                w_rec[QMAN_ORDER_KEY] = list(_man.keys())      # key order is load-bearing: record it before deleting
                w_rec[QMAN_KEY] = {k: _man[k] for k in _hit}
                stripped += len(_hit)
                for k in _hit:
                    del _man[k]
        T = _T(case)
        series = case.get("longitudinal_data") or {}
        kernel_view = _filter_le_T(series, T)       # what the solver sees at T
        post_s = {}
        for sig, pts in series.items():
            late = [[i, p] for i, p in enumerate(pts) if p["ts"] > T]
            if late:
                post_s[sig] = late
                series[sig] = [p for p in pts if p["ts"] <= T]
        if series != kernel_view:
            raise SystemExit(f"{w_rec['case_id']}: Q-side truncation disagrees with kernel _filter_le_T")
        if post_s:
            w_rec[POST_T_SERIES_KEY] = post_s
        ev = case.get("evidence_ledger")
        if isinstance(ev, list):
            late_ev = [[i, e] for i, e in enumerate(ev)
                       if e.get("source_timestamp", _EV_TS_DEFAULT) > T]
            if late_ev:
                w_rec[POST_T_EVIDENCE_KEY] = late_ev
                case["evidence_ledger"] = [e for e in ev
                                           if e.get("source_timestamp", _EV_TS_DEFAULT) <= T]
        q_lines.append(_dump(C.stamp_row(row)))
        w_lines.append(_dump(w_rec))
    q_path.write_text("".join(l + "\n" for l in q_lines), encoding="utf-8")
    w_path.write_text("".join(l + "\n" for l in w_lines), encoding="utf-8")
    return q_path, w_path, len(q_lines), stripped


def join(pack: pathlib.Path) -> bytes:
    """Reconstruct the original `*.cases.jsonl` bytes from `Q` + `W`."""
    stem = pack.name[: -len(".cases.jsonl")]
    q_path = HERE / f"{stem}.Q.jsonl"
    w_path = HERE / f"{stem}.W.jsonl"
    if not (q_path.is_file() and w_path.is_file()):
        raise SystemExit(f"cannot join, a half is missing: {q_path.name} / {w_path.name}")
    w_by_id = {}
    for line in w_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            w_by_id[r["case_id"]] = r
    out = []
    for line in q_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # The canary field is last on the Q side, so dropping it restores the key order.
        row = C.strip_row(json.loads(line))
        cid = row.get("case_id") or row["case"].get("case_id")
        w = w_by_id[cid]
        for sig, late in (w.get(POST_T_SERIES_KEY) or {}).items():
            pts = row["case"]["longitudinal_data"][sig]
            for i, p in late:
                pts.insert(i, p)
        for i, e in w.get(POST_T_EVIDENCE_KEY) or ():
            row["case"]["evidence_ledger"].insert(i, e)
        merged = dict(row["case"]) | {k: v for k, v in w.items()
                                      if k in W_TRUTH_FIELDS}
        row["case"] = {k: merged[k] for k in w[ORDER_KEY]}
        if QMAN_KEY in w:
            _man = row["question"]["injected_manifest"]
            _man.update(w[QMAN_KEY])
            row["question"]["injected_manifest"] = {k: _man[k] for k in w[QMAN_ORDER_KEY]}
        out.append(_dump(row))
    return "".join(l + "\n" for l in out).encode("utf-8")


def manifest_sha() -> dict[str, str]:
    """`{filename: file_sha256}` -- read live from MANIFEST, never hand-copied."""
    return {e["file"]: e["file_sha256"] for e in pack_entries().values()
            if e.get("file_sha256")}


def _per_case_digests(cases_path: pathlib.Path) -> dict[str, str]:
    """Per-case digests as `batch.json:cases_sha256` records them (`store.digest_of`)."""
    from haenv.store import digest_of, load_cases
    return digest_of(load_cases(cases_path))


def _anchor_packs() -> dict[str, dict]:
    from haenv.anchor import load_freeze
    return load_freeze().get("packs") or {}


def import_pack(job: str, batch_dir: pathlib.Path, revision: int) -> dict:
    """Copy a frozen pack out of the batch that holds it and record its MANIFEST entry.

    The batch's `cases.jsonl` must be the one the freeze anchor records for `job`
    (same sha256 prefix), and its per-case digests must match `batch.json`.
    The canary field is stripped on the way in (rule 5).
    """
    src = batch_dir / "cases.jsonl"
    raw = src.read_bytes()
    want = (_anchor_packs().get(job) or {}).get("sha256")
    if not want:
        raise SystemExit(f"the freeze anchor records no pack `{job}`: refusing to import it")
    if not _sha(raw).startswith(want):
        raise SystemExit(f"{src}: sha256 {_sha(raw)[:16]} is not the anchor's `{job}` pack ({want})")
    body = C.strip_jsonl(raw.decode("utf-8")).encode("utf-8")
    if C.stamp_jsonl(body.decode("utf-8")).encode("utf-8") != raw:
        raise SystemExit(f"{src}: stripping and re-stamping the canary does not give the file back")
    dst = HERE / f"{job}.cases.jsonl"
    dst.write_bytes(body)
    per_case = _per_case_digests(dst)
    meta = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    recorded = meta.get("cases_sha256") or {}
    if recorded and {k: recorded.get(k) for k in per_case} != per_case:
        raise SystemExit(f"{batch_dir}/batch.json: per-case digests disagree with cases.jsonl")
    jy = ROOT / "inputs" / f"{job}.job.yaml"
    entry = {"job": f"inputs/{job}.job.yaml", "freeze_revision": int(revision),
             "n_cases": len(per_case), "q_file": f"{job}.Q.jsonl", "file": dst.name,
             "file_sha256": _sha(body), "stamped_sha256": _sha(raw),
             "job_yaml_sha256": _sha(jy.read_bytes()) if jy.is_file() else None,
             "per_case_sha256": per_case}
    man = _manifest()
    man[job] = {k: entry[k] for k in ENTRY_KEYS}
    (HERE / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    return entry


def entry_problems(root: pathlib.Path = ROOT) -> list[str]:
    """Where the MANIFEST pack entries disagree with the published Q files and the
    freeze anchor. Needs no W half. Empty = agrees; no entry at all is a problem."""
    here = root / "frozen"
    entries = pack_entries(_manifest(here))
    if not entries:
        return ["frozen/MANIFEST.json names no pack"]
    out = []
    named = {e.get("q_file") for e in entries.values()}
    on_disk = {p.name for p in C.jsonl_packs(root)}
    if named != on_disk:
        out.append(f"Q files named {sorted(map(str, named))} != on disk {sorted(on_disk)}")
    anchor = _anchor_packs()
    for job, e in sorted(entries.items()):
        missing = [k for k in ENTRY_KEYS if k not in e]
        if missing:
            out.append(f"{job}: entry lacks {missing}")
            continue
        q = here / e["q_file"]
        if q.is_file():
            ids = [json.loads(x)["case_id"] for x in q.read_text(encoding="utf-8").splitlines()
                   if x.strip()]
            if len(ids) != e["n_cases"]:
                out.append(f"{job}: {len(ids)} Q rows != n_cases {e['n_cases']}")
            if set(ids) != set(e["per_case_sha256"]):
                out.append(f"{job}: Q case ids != per_case_sha256 keys")
        a = anchor.get(job)
        if a is None:
            out.append(f"{job}: the freeze anchor records no such pack")
        elif not str(e["stamped_sha256"]).startswith(str(a.get("sha256"))):
            out.append(f"{job}: stamped_sha256 {str(e['stamped_sha256'])[:16]} != anchor {a.get('sha256')}")
        elif a.get("n") is not None and int(a["n"]) != int(e["n_cases"]):
            out.append(f"{job}: n_cases {e['n_cases']} != anchor n {a['n']}")
        jy = root / e["job"]
        if jy.is_file() and _sha(jy.read_bytes()) != e["job_yaml_sha256"]:
            out.append(f"{job}: {e['job']} sha256 != job_yaml_sha256")
    return out


def check_all(root: pathlib.Path = ROOT) -> int:
    """`--check`: 0 when every published Q pack passes `check_q`, the MANIFEST
    canary block agrees with the files, and the pack entries agree with the Q files
    and the freeze anchor; 1 otherwise. An empty scan is a failure."""
    packs = C.jsonl_packs(root)
    if not packs:
        print("no frozen/*.Q.jsonl: the scan surface is empty, which is not a pass")
        return 1
    rc = 0
    for q in packs:
        n, bad = check_q(q)
        print(f"[check] {q.name}: {n} rows, {len(bad)} problem(s)")
        for b in bad[:10]:
            print(f"         {b}")
        rc |= 1 if (bad or n == 0) else 0
    man = C.manifest_problems(root)
    print(f"[check] MANIFEST `{C.MANIFEST_KEY}` block: {len(man)} problem(s)")
    for b in man:
        print(f"         {b}")
    ent = entry_problems(root)
    print(f"[check] MANIFEST pack entries vs Q files and freeze anchor: {len(ent)} problem(s)")
    for b in ent:
        print(f"         {b}")
    return rc | (1 if man else 0) | (1 if ent else 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imports", action="append", default=[],
                    metavar="JOB=BATCH_DIR",
                    help="copy a frozen pack out of its batch and record it in MANIFEST")
    ap.add_argument("--revision", type=int, help="freeze revision that froze the imported pack")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--join", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="check the published Q half alone: canary on every row, "
                         "nothing after T, MANIFEST agrees with the Q files and the anchor")
    a = ap.parse_args()
    if not (a.imports or a.split or a.join or a.check):
        ap.error("one of --import / --split / --join / --check")
    if a.imports:
        if a.revision is None:
            ap.error("--import needs --revision")
        for spec in a.imports:
            job, _, bdir = spec.partition("=")
            e = import_pack(job, pathlib.Path(bdir), a.revision)
            print(f"[import] {job}: {e['n_cases']} cases -> {e['file']}  "
                  f"sha256={e['file_sha256'][:16]}  stamped={e['stamped_sha256'][:16]}")
    if a.check:
        return check_all()

    if not (a.split or a.join):
        return 0
    packs = _packs()
    if not packs:
        print("MANIFEST names no pack: the scan surface is empty (this is not a pass)")
        return 1
    stamped = {e["file"]: e.get("stamped_sha256") for e in pack_entries().values()}
    man = manifest_sha()
    rc = 0
    for p in packs:
        if not p.is_file():
            print(f"[skip ] {p.name}: not on disk (it is not distributed); nothing to "
                  f"{'split' if a.split else 'join'} -- counted as failure")
            rc |= 1
            continue
        if a.split:
            q, w, n, k = split(p)
            print(f"[split] {p.name}: {n} rows, {k} truth fields stripped")
            print(f"         Q {q.name}  sha256={_sha(q.read_bytes())[:16]}  "
                  f"{q.stat().st_size} B")
            print(f"         W {w.name}  sha256={_sha(w.read_bytes())[:16]}  "
                  f"{w.stat().st_size} B")
        if a.join:
            got = join(p)
            want = p.read_bytes()
            g = _sha(got)
            print(f"[join ] {p.name}: rebuilt {len(got)} B  sha256={g[:16]}…")
            ok = got == want
            print(f"         byte-identical to original: {ok}")
            rc |= 0 if ok else 1
            if a.verify and p.name in man:
                ok = g == man[p.name]
                print(f"         == MANIFEST.file_sha256:{ok}  ({man[p.name][:16]}…)")
                s = _sha(C.stamp_jsonl(got.decode("utf-8")).encode("utf-8"))
                ok2 = s == stamped.get(p.name)
                print(f"         stamped == MANIFEST.stamped_sha256:{ok2}  ({s[:16]}…)")
                rc |= 0 if (ok and ok2) else 1
            elif a.verify:
                print(f"         MANIFEST has no file_sha256 for {p.name}: not verified, counted as failure")
                rc |= 1
    if a.split and C.write_manifest_block(ROOT):
        print(f"         MANIFEST `{C.MANIFEST_KEY}` block updated")
    return rc


if __name__ == "__main__":
    sys.exit(main())
