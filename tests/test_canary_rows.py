"""Row-level canary and the T cut on the published packs.

    uv run pytest tests/test_canary_rows.py -q          # seconds

What is held here:

* every row of every published `frozen/*.Q.jsonl` carries the three canary
  strings in the top-level `canary.FIELD`, never inside `case`;
* no row of a published Q pack holds a point after its index time T;
* `frozen/MANIFEST.json`'s `canary` block agrees with the packs on disk;
* `split_wq.join` still rebuilds the bytes `MANIFEST.file_sha256` records
  (needs the W half, so it skips on a clean clone and says so);
* runtime `cases.jsonl` / `payloads.jsonl` are stamped on write and the
  per-case fingerprints do not move because of it.

Every check that can say "clean" has a planted defect next to it that must make
it say "not clean". A check with no such control is indistinguishable from one
that never looks.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import shutil
import sys
from dataclasses import asdict

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))
sys.path.insert(0, str(ROOT / "frozen"))

from haenv import canary as C                        # noqa: E402

split_wq = importlib.import_module("split_wq")

PACKS = C.jsonl_packs(ROOT)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _rows(p: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_rows(p: pathlib.Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                         for r in rows), encoding="utf-8")


def _copy_frozen(dst: pathlib.Path, with_w: bool = False) -> pathlib.Path:
    """A throwaway repo root holding the published half of `frozen/` (and the
    W half when asked). Planted defects go here, never into the real tree."""
    (dst / "frozen").mkdir(parents=True)
    shutil.copy2(ROOT / "frozen" / "MANIFEST.json", dst / "frozen" / "MANIFEST.json")
    for p in PACKS:
        shutil.copy2(p, dst / "frozen" / p.name)
        if with_w:
            w = p.with_name(p.name.replace(".Q.jsonl", ".W.jsonl"))
            shutil.copy2(w, dst / "frozen" / w.name)
    return dst


# ---------------------------------------------------------------- published Q packs

def test_scan_surface_is_not_empty():
    """Without packs every check below is vacuously green."""
    assert len(PACKS) >= 2, f"only {len(PACKS)} Q pack(s) found"
    assert all(len(_rows(p)) >= 1 for p in PACKS)


@pytest.mark.parametrize("pack", PACKS, ids=lambda p: p.name)
def test_published_pack_is_clean(pack):
    """Positive side: the pack as shipped has canary on every row and nothing after T."""
    n, bad = split_wq.check_q(pack)
    assert n == len(_rows(pack)) and n > 0
    assert bad == [], bad[:5]


def test_check_catches_a_row_without_canary(tmp_path):
    """Negative control: drop the field from one row, and only that row is reported."""
    src = PACKS[0]
    rows = _rows(src)
    victim = rows[5]["case_id"]
    rows[5] = C.strip_row(rows[5])
    p = tmp_path / src.name
    _write_rows(p, rows)
    n, bad = split_wq.check_q(p)
    assert bad == [f"{victim}: no canary field `{C.FIELD}`"], bad


def test_check_catches_a_post_T_point(tmp_path):
    """Negative control: one point one day after T in one stream must be reported."""
    src = PACKS[0]
    rows = _rows(src)
    case = rows[3]["case"]
    T = int(case["prediction_context"]["prediction_time_T"])
    sig = next(iter(case["longitudinal_data"]))
    case["longitudinal_data"][sig].append({"ts": T + 1, "value": 0})
    p = tmp_path / src.name
    _write_rows(p, rows)
    _, bad = split_wq.check_q(p)
    assert len(bad) == 1 and f"{sig} has 1 point(s) after T={T}" in bad[0], bad


def test_check_catches_a_post_T_evidence_row(tmp_path):
    """Negative control: an evidence row dated after T must be reported."""
    src = PACKS[0]
    rows = _rows(src)
    case = rows[1]["case"]
    T = int(case["prediction_context"]["prediction_time_T"])
    case["evidence_ledger"].append({"evidence_id": "EV-PLANTED", "source_timestamp": T + 30})
    p = tmp_path / src.name
    _write_rows(p, rows)
    _, bad = split_wq.check_q(p)
    assert any("EV-PLANTED" in b for b in bad), bad


def test_check_catches_canary_inside_the_case(tmp_path):
    """Negative control: the strings inside `case` are a marker in the prompt."""
    src = PACKS[0]
    rows = _rows(src)
    rows[0]["case"]["user_profile"]["note"] = C.HAENV
    p = tmp_path / src.name
    _write_rows(p, rows)
    _, bad = split_wq.check_q(p)
    assert any("inside the case body" in b for b in bad), bad


def test_q_rows_are_what_the_solver_sees_at_T():
    """The published case body is the kernel's solver view at T, field for field
    on the time series. `build._filter_le_T` is the one truncation rule."""
    from build import _filter_le_T                     # kernel
    for p in PACKS:
        for r in _rows(p):
            c = r["case"]
            T = int(c["prediction_context"]["prediction_time_T"])
            assert c["longitudinal_data"] == _filter_le_T(c["longitudinal_data"], T)


# ---------------------------------------------------------------- MANIFEST canary block

def test_manifest_block_agrees_with_disk():
    assert C.manifest_problems(ROOT) == []
    man = json.loads((ROOT / "frozen" / "MANIFEST.json").read_text(encoding="utf-8"))
    rec = man[C.MANIFEST_KEY]["packs"]
    assert set(rec) == {p.name for p in PACKS}
    for p in PACKS:
        assert rec[p.name]["sha256_with_canary"] != rec[p.name]["sha256_without_canary"]


def test_manifest_block_catches_an_edited_pack(tmp_path):
    """Negative control: one changed value in one row of a copy must be reported."""
    root = _copy_frozen(tmp_path)
    q = root / "frozen" / PACKS[0].name
    rows = _rows(q)
    rows[0]["case"]["user_profile"]["age_range"] = "99-99"
    _write_rows(q, rows)
    got = C.manifest_problems(root)
    assert any(PACKS[0].name in g for g in got), got


def test_manifest_block_catches_a_missing_block(tmp_path):
    """Negative control: no block is a failure, not a pass."""
    root = _copy_frozen(tmp_path)
    mp = root / "frozen" / "MANIFEST.json"
    man = json.loads(mp.read_text(encoding="utf-8"))
    man.pop(C.MANIFEST_KEY)
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    assert C.manifest_problems(root)


def test_rebuild_is_a_no_op_on_disk():
    """`python -m haenv.canary --check` must find nothing to change: the tracked
    files are exactly what the shipped rebuild command renders."""
    assert C.rebuild(ROOT, write=False) == []


# ---------------------------------------------------------------- join back to MANIFEST

_HAVE_W = all(p.with_name(p.name.replace(".Q.jsonl", ".W.jsonl")).is_file() for p in PACKS)
_W_SKIP = ("W half not on disk (normal on a clean clone): join has nothing to rebuild from. "
           "Not a pass.")


@pytest.mark.skipif(not _HAVE_W, reason=_W_SKIP)
def test_join_rebuilds_manifest_bytes(tmp_path, monkeypatch):
    """Positive side, on a copy so the check never writes into `frozen/`."""
    root = _copy_frozen(tmp_path, with_w=True)
    monkeypatch.setattr(split_wq, "HERE", root / "frozen")
    man = split_wq.manifest_sha()
    assert man
    for name, want in man.items():
        assert _sha(split_wq.join(root / "frozen" / name)) == want, name


@pytest.mark.skipif(not _HAVE_W, reason=_W_SKIP)
def test_join_catches_a_lost_post_T_point(tmp_path, monkeypatch):
    """Negative control: drop one post-T point from W and join must not match."""
    root = _copy_frozen(tmp_path, with_w=True)
    monkeypatch.setattr(split_wq, "HERE", root / "frozen")
    name, want = next(iter(split_wq.manifest_sha().items()))
    w = root / "frozen" / name.replace(".cases.jsonl", ".W.jsonl")
    rows = _rows(w)
    r = next(x for x in rows if x.get(split_wq.POST_T_SERIES_KEY))
    sig = next(iter(r[split_wq.POST_T_SERIES_KEY]))
    r[split_wq.POST_T_SERIES_KEY][sig].pop()
    _write_rows(w, rows)
    assert _sha(split_wq.join(root / "frozen" / name)) != want


_HAVE_CASES = all(p.with_name(p.name.replace(".Q.jsonl", ".cases.jsonl")).is_file()
                  for p in PACKS)


@pytest.mark.skipif(not _HAVE_CASES, reason="*.cases.jsonl not on disk; split has no input")
def test_split_reproduces_the_tracked_q_bytes(tmp_path, monkeypatch):
    """The tracked Q half is exactly what `--split` writes today. A hand edit, or a
    split rule that changed without re-splitting, shows up here."""
    (tmp_path / "frozen").mkdir()
    for p in PACKS:
        c = p.with_name(p.name.replace(".Q.jsonl", ".cases.jsonl"))
        shutil.copy2(c, tmp_path / "frozen" / c.name)
    monkeypatch.setattr(split_wq, "HERE", tmp_path / "frozen")
    for p in PACKS:
        c = tmp_path / "frozen" / p.name.replace(".Q.jsonl", ".cases.jsonl")
        q, _w, _n, _k = split_wq.split(c)
        assert q.read_bytes() == p.read_bytes(), p.name


# ---------------------------------------------------------------- runtime batch files

def _raw_case(cid: str = "CAN-01"):
    from schema import RawCase                         # kernel
    return RawCase(
        case_id=cid, user_profile={"age_range": "40-44"},
        prediction_context={"prediction_time_T": 84},
        longitudinal_data={"weight": [{"ts": 0, "value": 88.0}, {"ts": 84, "value": 85.0},
                                      {"ts": 120, "value": 84.0}]},
        evidence_ledger=[{"evidence_id": "EV1", "source_timestamp": 10}],
        outcome_label="event_not_occurred", label_rule={}, gold_drivers=["x"],
        adjudication={}, reversal_points=[])


def test_save_cases_stamps_without_moving_the_digest(tmp_path):
    from haenv.store import digest_of, load_cases, save_cases
    built = {"CAN-01": _raw_case("CAN-01"), "CAN-02": _raw_case("CAN-02")}
    path, digests = save_cases(tmp_path / "cases.jsonl", built)
    rows = _rows(path)
    assert [C.row_has_canary(r) for r in rows] == [True, True]
    assert all(list(r)[-1] == C.FIELD and C.canary_outside_the_case(r) for r in rows)
    assert digests == digest_of(built)
    back = load_cases(path)
    assert {k: asdict(v) for k, v in back.items()} == {k: asdict(v) for k, v in built.items()}
    assert digest_of(back) == digests
    # control: a digest taken over the stamped line would differ, so the equality
    # above is a real statement about where the digest is taken
    line = path.read_text(encoding="utf-8").splitlines()[0]
    assert _sha(line.encode())[:16] != digests["CAN-01"]


def test_save_payloads_stamps_without_moving_the_digest(tmp_path):
    from haenv import payloads as P
    built = {"CAN-01": _raw_case("CAN-01")}
    path, digests = P.save_payloads(tmp_path / P.FILENAME, built)
    line = path.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert C.row_has_canary(row) and C.canary_outside_the_case(row)
    bare = json.dumps(C.strip_row(row), ensure_ascii=False, separators=(",", ":"))
    assert _sha(bare.encode())[:16] == digests["CAN-01"]
    assert _sha(line.encode())[:16] != digests["CAN-01"]
    sp, vp = P.load_payloads(path)["CAN-01"]
    assert all(p["ts"] <= 84 for p in sp.longitudinal_data["weight"])


def test_canary_never_reaches_the_solver_payload(tmp_path):
    """Read back a stamped batch the way `run` does and build the solver view:
    none of the strings may be in it."""
    from build import build_instance                   # kernel
    from haenv.store import load_cases, save_cases
    path, _ = save_cases(tmp_path / "cases.jsonl", {"CAN-01": _raw_case()})
    assert C.HAENV_GUID in path.read_text(encoding="utf-8")
    raw = load_cases(path)["CAN-01"]
    sp, _vp = build_instance(raw, 84)
    text = sp.dumps()
    assert C.HAENV_GUID not in text and not any(ln in text for ln in C.LINES)
