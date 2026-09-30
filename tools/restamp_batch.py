"""restamp_batch.py -- derive a publishable batch (single judge definition) from
an existing one, at zero model cost.

    uv run python tools/restamp_batch.py <batch dir> [--out <new batch dir>] [--dry]
        [--allow-change <field[,field...]> | --allow-change-all]

1. Drop stub rows (`haenv.baselines.BASELINE_NAMES`) into a new batch directory.
2. Recompute real-model judges from persisted responses
   (`recompute_judges --full --tracks`; `--allow-change <fields>` and
   `--allow-change-all` are forwarded only when given). No model is called.
   Without them a field whose recomputed value differs keeps its old value while
   the row gets the new judging stamp (listed in `recompute_conflicts`): after a
   judge change, name the fields that change.
3. Re-run the stubs offline: the tool prints the command, since the job.yaml
   path cannot be derived from the batch.
4. Check `report.assert_publishable`; exit nonzero if it fails.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time as _time

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))


def drop_stubs(src: pathlib.Path, dst: pathlib.Path) -> tuple[int, int]:
    """Copy `src` to `dst` without the stub eval rows. Returns (kept, dropped)."""
    from haenv.baselines import BASELINE_NAMES
    stubs = set(BASELINE_NAMES)
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- won't overwrite an existing batch (old batches are never deleted or overwritten)")
    shutil.copytree(src, dst)
    # Give the copy its own identity; `derived_from` records the source batch.
    _bj = dst / "batch.json"
    if _bj.is_file():
        _m = json.loads(_bj.read_text(encoding="utf-8"))
        # The source directory's own name (a copied batch can carry a stale `batch` field),
        # plus its path as a locator; semantic runs find their judged batch by content.
        _m["derived_from"] = src.name
        _m["derived_from_path"] = str(src.resolve())
        _m["derived_how"] = (f"tools/restamp_batch.py:剔桩 + recompute_judges "
                             f"--full --tracks(零模型开销;冲突字段默认挡下,不自动覆盖)")
        _m["batch"] = dst.name
        _bj.write_text(json.dumps(_m, ensure_ascii=False, indent=2), encoding="utf-8")
    ev = dst / "eval.jsonl"
    kept, dropped = [], 0
    for line in ev.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        if str(json.loads(line).get("solver")) in stubs:
            dropped += 1
            continue
        kept.append(line)
    ev.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    # Filter `trace.jsonl` the same way (step 3 appends the stubs' events again).
    tr = dst / "trace.jsonl"
    if tr.is_file():
        t_keep, t_drop = [], 0
        for line in tr.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            try:
                s = str(json.loads(line).get("solver"))
            except json.JSONDecodeError:
                t_keep.append(line)          # a line that can't be parsed is read_trace's problem, not swallowed here
                continue
            if s in stubs:
                t_drop += 1
                continue
            t_keep.append(line)
        tr.write_text("\n".join(t_keep) + ("\n" if t_keep else ""), encoding="utf-8")
        print(f"   [restamp] Synced stub removal into the trace: kept {len(t_keep)} events · dropped {t_drop}")
    return len(kept), dropped


def check_publishable(batch: pathlib.Path) -> tuple[bool, str]:
    """Run the production leaderboard gate."""
    from haenv.evaluate import load_rows
    from haenv.report import assert_publishable
    try:
        assert_publishable(load_rows(batch / "eval.jsonl"))
        return True, "Pass"
    except Exception as e:                                  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def check_halves(batch: pathlib.Path) -> tuple[bool, str]:
    """Whether the real-model and stub halves agree on judgeable cases, which
    `assert_publishable` does not check: the stub half is re-run under the
    current code, the real-model half is rejudged from copied responses.

    A denominator mismatch fails; a provenance mismatch is only reported. The
    comparison helper ships with this tool and does not require maintainer tests.
    """
    from tools.board_integrity import board_halves_verdict, split_board_halves
    rows = [json.loads(x) for x in
            (batch / "eval.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    sp = split_board_halves(rows)
    den, prov = board_halves_verdict(sp)
    if not sp["real"]["n_rows"] or not sp["stub"]["n_rows"]:
        # Only one half present (stubs not yet re-added): not measured.
        return True, (f"not measured -- only one half exists (real {sp['real']['n_rows']} rows / "
                      f"stub {sp['stub']['n_rows']} rows); judge once both halves are present")
    note = (f"real {sp['real']['solvers']} models {sp['real']['n_rows']} rows / "
            f"stub {sp['stub']['solvers']} stubs {sp['stub']['n_rows']} rows · "
            f"judgeable cases {len(sp['real']['cases'])} vs {len(sp['stub']['cases'])} · "
            f"provenance mismatches {len(prov)}" + (f":{prov}" if prov else ""))
    if den:
        return False, f"denominator mismatch: {den} · {note}"
    return True, note


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    src = pathlib.Path(argv[1]).resolve()
    if not (src / "eval.jsonl").is_file():
        print(f"{src} has no eval.jsonl")
        return 2
    dry = "--dry" in argv
    # `haenv run --batch` (step 3) only accepts `YYYYmmdd-HHMMSS` names.
    dst = (pathlib.Path(argv[argv.index("--out") + 1]).resolve()
           if "--out" in argv else src.parent / _time.strftime("%Y%m%d-%H%M%S"))

    ok0, why0 = check_publishable(src)
    print(f"[restamp] Source batch {src.name}:{'✅ already publishable' if ok0 else '❌ ' + why0[:150]}")
    if ok0:
        # A publishable batch with both halves still needs the halves check.
        okh, whyh = check_halves(src)
        _mark = "❌ " if not okh else ("⚪ " if "NOT MEASURED" in whyh else "✅ ")
        print(f"[restamp] Two-halves provenance check:{_mark}{whyh[:300]}")
        if not okh:
            print("[restamp] ❌ Gate passed, but the two halves don't share the same provenance -- not publishable."
                  "\n           The stub half was run under the current code; the real-model half was copied over and rejudged."
                  "\n           For the cases named above, the current code refuses to judge them, yet the real models are still scoring on them."
                  "\n           Pick one: re-run the real-model half under the current code,"
                  " or drop those cases from both halves and document why.")
            return 1
        print("[restamp] Nothing to do.")
        return 0
    if dry:
        print(f"[restamp] --dry: would produce {dst}")
        return 0

    kept, dropped = drop_stubs(src, dst)
    print(f"[restamp] ① Drop stubs: kept {kept} rows · dropped {dropped} rows -> {dst}")
    if not kept:
        print("Nothing left after dropping stubs -- this batch has only stubs, so it can't become a publishable batch with real models")
        return 1

    # `--allow-change-all` (new value wins on conflicting fields) only when given.
    _rc_argv = [sys.executable, str(ROOT / "tools" / "recompute_judges.py"), str(dst),
                "--full", "--tracks"]
    if "--allow-change-all" in sys.argv:
        _rc_argv.append("--allow-change-all")
        print("[restamp] `--allow-change-all` explicitly authorized -- conflicting fields will be overwritten with the recomputed value")
    elif "--allow-change" in argv:
        _fields = argv[argv.index("--allow-change") + 1]
        _rc_argv += ["--allow-change", _fields]
        print(f"[restamp] `--allow-change {_fields}` authorized -- only these fields take the recomputed value")
    r = subprocess.run(_rc_argv, capture_output=True, text=True, cwd=str(ROOT))
    for ln in (r.stdout or "").splitlines():
        if "[recompute]" in ln:
            print("   " + ln)
    if r.returncode != 0:
        print(f"❌ ② Recompute failed (exit code {r.returncode}): {(r.stderr or '')[-400:]}")
        return 1
    print("[restamp] ② Recompute complete (zero model cost, rejudged from already-persisted responses)")

    ok, why = check_publishable(dst)
    _jid = json.loads((dst / "batch.json").read_text(encoding="utf-8")).get("job_id", "?")
    print(f"\n[restamp] ④ Leaderboard gate:{'✅ Pass' if ok else '❌ ' + why[:300]}")
    _okh, _whyh = check_halves(dst)
    print(f"[restamp] ④b Two-halves provenance:{_whyh[:200]}")
    if ok:
        print(f"[restamp] Publishable batch -> {dst}")
        print(f"[restamp] It has only real models, no stubs. To publish with stubs, first run step ③:\n"
              f"    uv run haenv run inputs/{_jid}.job.yaml --offline "
              f"--batch {dst.name}\n"
              f"  then re-run this tool to re-certify -- once the stubs are backfilled this batch has both halves,"
              f" and in the time between the two halves, the case generation / kernel / judges may all have changed.\n"
              f"  Before step ③, `④b two-halves provenance` can only print 'not measured' (the stubs were just dropped, so there's nothing to compare yet);"
              f"the check applies once the stubs are backfilled.")
        return 0
    print("[restamp] ❌ Still not publishable -- this directory is not publishable material. "
          "The line above says why.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
