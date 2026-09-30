"""reliability_passk.py -- re-run the same case set k times to measure the noise floor.

Do not read a delta of order 0.1 as an effect without a `pass^k` first.

Given a batch's frozen `cases.jsonl`, run k samples (concurrently by default) and report:

1. Mean-level noise, `max|delta|` / `median|delta|` across the k runs -- the number to
   compare when judging whether a change had an effect.
2. Per-cell flip rate -- usually much larger than mean-level noise, because jitter
   cancels in the mean; the two are not interchangeable.
3. `pass^k` (binary dimensions) -- the fraction of cells correct in all of the first k
   runs; `pass^1 >= pass^2 >= ...`, and the rate of decline is the instability.

Cases are reused byte for byte. The online report uses saved semantic judgments,
so its measured jitter includes both solver and adaptive-judge sampling. It is
not an estimate of solver-only variance; judge repeatability is reported apart.
Running the k samples
concurrently lets repeats of a prompt land in the backend's prefix-cache window;
per-run cache readings are recorded in the artifact.

Run:
    uv run python tools/reliability_passk.py <job.yaml> --from-batch <stamp> --k 3 --models A,B
    uv run python tools/reliability_passk.py <job.yaml> --report-only      # aggregate existing samples only
    uv run python tools/reliability_passk.py <job.yaml> --k 3 --serial     # run sequentially
    uv run python tools/reliability_passk.py <job.yaml> --k 3 --workers 6  # global concurrency
    uv run python tools/reliability_passk.py <job.yaml> --k 3 --offline    # dry run with the offline stub

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
import re
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from haenv import kernel_path as _kernel_path       # noqa: E402
# Same kernel resolution as `haenv` itself.
if (_kp := _kernel_path()):
    sys.path.insert(0, str(_kp))

#: Sample batches are named `<stamp>-pk<i>`, so a run's samples are recognizable by name.
SAMPLE_FMT = "%s-pk%d"

#: Global concurrency across all k samples, divided per process, so k processes do not
#: multiply the lanes and trip a backend's rate limit.
PASSK_TOTAL_WORKERS = 4


def _cache_by_sample(sample_dirs, models) -> list[dict]:
    """Per-run input cache rate, read from `billed`. Rows without billing data are recorded as
    a `status`, never as 0.
    """
    out = []
    for d in sample_dirs:
        f = d / "eval.jsonl"
        if not f.is_file():
            out.append({"sample": d.name, "status": "no_eval"})
            continue
        fresh = cached = n = 0
        for ln in f.read_text(encoding="utf-8").split("\n"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if models and r.get("solver") not in models:
                continue
            b = r.get("billed") or {}
            if b.get("status") != "measured":
                continue
            fresh += int(b.get("in_fresh") or 0)
            cached += int(b.get("in_cached") or 0)
            n += 1
        tot = fresh + cached
        out.append({"sample": d.name, "n_rows_measured": n, "in_fresh": fresh,
                    "in_cached": cached,
                    "cache_rate": round(cached / tot, 4) if tot else None,
                    "status": "measured" if n else "no_measured_rows"})
    return out


def pick_source_batch(base, from_batch: str | None):
    """The source batch for the case set: `from_batch` if given (None if it does not exist),
    else the latest complete batch from `tools/packread.batches`, which excludes this
    tool's own `-pk<i>` (and `-rep<i>`) sample directories.
    """
    from pathlib import Path as _P
    base = _P(base)
    if from_batch:
        from haenv.batch import STAMP_RE, SAMPLE_SUFFIX_RE
        m = STAMP_RE.match(from_batch)
        if m and SAMPLE_SUFFIX_RE.match(from_batch[15:]):
            return None                                  # a `-pk<i>` / `-rep<i>` sample is never a prompt source
        d = base / from_batch
        return d if d.is_dir() else None
    import packread as _pr                                           # noqa: PLC0415
    bs = [b for b in _pr.batches(ROOT, job_id=base.name) if _pr.is_complete(b)[0]]
    return bs[-1].dir if bs else None


def _runner_argv() -> list[str]:
    """Command prefix for `haenv run`: the current interpreter (`sys.executable -m haenv`),
    so the samples run in exactly the caller's environment.
    """
    import sys as _s
    return [_s.executable, "-m", "haenv"]


def _run_samples(job_path, sample_dirs, models, *, serial: bool,
                 total_workers: int = PASSK_TOTAL_WORKERS,
                 offline: bool = False, limit: int | None = None,
                 budget_usd: str | None = None, budget_ledger: Path | None = None) -> list[int | None]:
    """Run k samples, concurrently by default; returns each run's exit code in order.

    Concurrent runs send identical prompts within seconds of each other, so runs 2..k can
    hit the backend's prefix cache; prompts are never changed. Total lanes are capped at
    `total_workers` and divided per process (`BACKEND_SEM` is per process). `serial=True`
    runs one at a time, as a fallback and as the control for cache hits.
    """
    if not offline and (budget_usd is None or budget_ledger is None):
        raise ValueError("Paid repetitions require an explicit shared budget and ledger")
    per = max(1, total_workers // max(1, len(sample_dirs))) if not serial else total_workers

    def _cmd(d):
        c = _runner_argv() + ["run", str(job_path), "--batch", d.name,
                              "--workers", str(per)]
        if offline:
            # Dry run with the offline stub: exercises the whole path with no model calls. The stub
            # is deterministic, so the artifact records `stochastic_source: false` and
            # `analytics.noise_floor_for` rejects the floor.
            c.append("--offline")
        else:
            c += ["--judge-budget-usd", str(budget_usd),
                  "--judge-budget-ledger", str(Path(budget_ledger).resolve())]
        if limit:
            c += ["--limit", str(int(limit))]
        return c + (["--models", ",".join(sorted(models))] if models else [])

    if serial:
        out = []
        for i, d in enumerate(sample_dirs, 1):
            print(f"[passk] sample {i}/{len(sample_dirs)} (serial, {per} lanes) -> {d.name}")
            out.append(subprocess.run(_cmd(d), cwd=ROOT).returncode)
            if out[-1] != 0:
                return out + [None] * (len(sample_dirs) - len(out))
        return out

    print(f"[passk] {len(sample_dirs)} samples run concurrently ({per} lanes each, "
          f"{per * len(sample_dirs)} in total) -- the k sends of one prompt land in the same "
          f"prefix-cache window")
    procs = [subprocess.Popen(_cmd(d), cwd=ROOT) for d in sample_dirs]
    return [p.wait() for p in procs]

#: Dimensions aggregated; binary ones also get `pass^k`. `direction_ok` here is raw accuracy,
#: not the per-class macro average that matters on an imbalanced pack.
DIMS = ("dx_hit", "tests_recall", "tests_precision", "disc_recall",
        "wk_tests_recall_last", "wk_tests_precision_last", "wk_disc_recall_last",
        "rev_stability", "rev_responsiveness", "noop_ok", "quant_ok",
        "review_macro", "excl_grounded_rate",
        "direction_ok", "driver_hit", "brier")
BINARY = {"dx_hit", "noop_ok", "quant_ok", "direction_ok", "driver_hit"}

# Stub names come from `baselines.BASELINE_NAMES`, so stubs are never counted as real models.
def _stubs() -> frozenset:
    from haenv.baselines import BASELINE_NAMES
    return frozenset(BASELINE_NAMES)


STUBS_HINT = _stubs()
def _dirty_grids(batch_dir: Path) -> set[tuple]:
    """Cells in this batch that contain an empty response slice; excluded from aggregation,
    since one such cell can move a noise-floor comparison on several dimensions.
    """
    rp = batch_dir / "responses.jsonl"
    if not rp.is_file():
        return set()
    from collections import defaultdict
    per: dict[tuple, list[int]] = defaultdict(list)
    for line in rp.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:                                  # noqa: BLE001
            continue
        n = r.get("n_chars")
        if n is None:
            n = len(str(r.get("raw") or ""))
        per[(r.get("case"), r.get("solver"))].append(int(n))
    return {k for k, ns in per.items() if any(x == 0 for x in ns)}


def _load(batch_dir: Path, models: set[str] | None, *, semantic: bool = False) -> dict:
    """Read one sample run, skipping ABORT cells and cells with an empty slice."""
    out: dict = {}
    p = batch_dir / "eval.jsonl"
    if not p.is_file():
        return out
    dirty = _dirty_grids(batch_dir)
    if dirty:
        print(f"[passk] {batch_dir.name}: dropped {len(dirty)} cells containing an empty "
              f"response slice (a score earned on a broken sequence is not comparable; "
              f"see `_dirty_grids`)")
    for line in p.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("solver") in STUBS_HINT:
            continue
        if models and r.get("solver") not in models:
            continue
        if str(r.get("overall", "")).startswith("ABORT"):
            continue
        if (r.get("case"), r.get("solver")) in dirty:
            continue
        out[(r.get("case"), r.get("solver"))] = r
    if semantic:
        from haenv.semantic_rubric import load_policy
        from haenv.semantic_report import view_for_batch
        semantic_dir = batch_dir / "semantic" / load_policy()["version"]
        if not (semantic_dir / "manifest.json").is_file():
            print(f"[passk] {batch_dir.name}: no saved semantic judgments; code proxies are not substitutes")
            return {}
        viewed = view_for_batch(list(out.values()), batch_dir, semantic_dir)
        if any(row["semantic"]["status"] != "resolved" for row in viewed):
            print(f"[passk] {batch_dir.name}: incomplete semantic judgments; no reliability conclusion")
            return {}
        out = {(row["case"], row["solver"]): row for row in viewed}
    return out


def _assert_samples_distinct(dirs: list[Path]) -> list[Path]:
    """Keep only directories that are genuinely distinct, comparable samples.

    A directory whose `eval.jsonl` is byte-identical to another's is a copy and is dropped.
    If slice depth differs between samples (it is set at solve time, not stored in
    `cases.jsonl`), no sample is returned.
    """
    import hashlib as _h
    keep: list[Path] = []
    seen: dict[str, str] = {}
    depth: dict[str, tuple] = {}
    for d in dirs:
        ep, rp = d / "eval.jsonl", d / "responses.jsonl"
        if not ep.is_file():
            continue
        sig = _h.sha256(ep.read_bytes()).hexdigest()[:16]
        if sig in seen:
            print(f"[passk] dropped {d.name}: its `eval.jsonl` is byte-identical to "
                  f"{seen[sig]} -- a copy, not another sample (it would make k look sufficient)")
            continue
        seen[sig] = d.name
        if rp.is_file():
            from collections import Counter as _C
            per = _C()
            for line in rp.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:                              # noqa: BLE001
                    continue
                per[(r.get("solver"), r.get("case"))] += 1
            if per:
                depth[d.name] = (min(per.values()), max(per.values()))
        keep.append(d)
    if len(set(depth.values())) > 1:
        print(f"[passk] geometry depth differs; these samples are not comparable: {depth}")
        print("[passk]    (the same cases.jsonl can yield different slice counts -- "
              "slice times are fixed at solve time)")
        return []
    return keep


def _aggregate(samples: list[dict]) -> dict:
    """k samples -> the three kinds of numbers, over cells present in every sample."""
    common = sorted(set.intersection(*[set(s) for s in samples])) if samples else []
    rep: dict = {"k": len(samples), "n_grids": len(common), "dims": {}}
    for d in DIMS:
        measured = [k for k in common if all(isinstance(s[k].get(d), (int, float, bool))
                                             for s in samples)]
        vals = []
        for s in samples:
            v = [float(s[k][d]) for k in measured]
            if v:
                vals.append(st.mean(v))
        if len(vals) < 2:
            continue
        # Per-model self-noise: the range of each model's mean across samples. Pooled noise can be
        # dominated by one unstable model. Reported, never scored.
        by_solver: dict[str, dict] = {}
        _solvers = sorted({k[1] for k in measured})
        for sv in _solvers:
            gs = [k for k in measured if k[1] == sv]
            ms = []
            for s in samples:
                v = [float(s[k][d]) for k in gs
                     if isinstance(s[k].get(d), (int, float, bool))]
                if v:
                    ms.append(round(st.mean(v), 4))
            if len(ms) == len(samples) and len(ms) >= 2:
                by_solver[sv] = {"means": ms, "self_noise": round(max(ms) - min(ms), 4),
                                 "n_grids": len(gs)}
        flips = sum(1 for k in measured
                    if len({(s[k].get(d) if isinstance(s[k].get(d), (int, float, bool))
                             else None) for s in samples}) > 1)
        # Number of samples that have this dimension; a dimension missing from some samples
        # is marked not comparable.
        _n_s = len(vals)
        ent = {"means": [round(v, 4) for v in vals],
               "n_grids": len(measured), "n_excluded_missing": len(common) - len(measured),
               "by_solver": by_solver,
               "self_noise_worst": (max(((v["self_noise"], s)
                                         for s, v in by_solver.items()), default=(None, None))),
               "n_samples": _n_s, "k_expected": len(samples),
               "comparable": _n_s == len(samples),
               "noise_max_abs_delta": round(max(vals) - min(vals), 4),
               "flip_grids": flips,
               "flip_rate": round(flips / len(measured), 4)}
        if _n_s != len(samples):
            ent["note"] = (f"Only {_n_s}/{len(samples)} samples have this dimension => "
                           f"this noise figure is unusable (usually means one sample's "
                           f"batch never asked that question -- this measures 'that batch "
                           f"never asked it,' not model jitter)")
        if d in BINARY:
            # `pass^j`: fraction of cells true in all of the first j samples
            ent["pass_k"] = {}
            for j in range(1, len(samples) + 1):
                ok = sum(1 for k in measured
                         if all(bool(samples[i][k].get(d)) for i in range(j)))
                ent["pass_k"][f"pass^{j}"] = round(ok / len(measured), 4)
        rep["dims"][d] = ent
    # The noise floor uses only dimensions present in every sample.
    _incomp = [d for d, v in rep["dims"].items() if not v.get("comparable")]
    if _incomp:
        rep["excluded_from_noise_floor"] = _incomp
    _nz = [v["noise_max_abs_delta"] for v in rep["dims"].values() if v.get("comparable")]
    rep["noise_floor_max"] = round(max(_nz), 4) if _nz else None
    rep["noise_floor_median"] = round(st.median(_nz), 4) if _nz else None
    rep["judge_in_repeat"] = False
    rep["judge_note"] = ("haenv's judges are all deterministic code (verifier.grade / judges.* / "
                         "process never call a model); an LLM appears only at generation time and "
                         "the case set is frozen => the jitter measured here comes entirely from "
                         "the model under test.")
    # Record which solvers the floor was measured on. A floor measured only on deterministic
    # stubs is 0 by construction; `noise_floor_for` rejects it.
    try:
        sys.path.insert(0, str(ROOT))
        from haenv.baselines import BASELINE_NAMES as _BN_P
        _stub = set(_BN_P)
    except Exception:                                            # noqa: BLE001
        _stub = set()
    _measured = sorted({k[1] for k in common})
    rep["solvers_measured"] = _measured
    rep["stochastic_source"] = bool([s for s in _measured if s not in _stub])
    if not rep["stochastic_source"]:
        rep["noise_floor_unusable_why"] = (
            "This floor was measured only on the offline stub. The stub is deterministic "
            "code, and repeating it k times gives byte-identical output, so the noise is "
            "necessarily 0, which is not a model's noise. Rerun on a real model before using "
            "this for an effect judgment.")
    return rep


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    job_path = Path(argv[0])
    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    k = int(opt("--k", "3"))
    from_batch = opt("--from-batch")
    models = {x.strip() for x in (opt("--models") or "").split(",") if x.strip()} or None
    report_only = "--report-only" in argv
    #: Concurrent by default; `--serial` runs sequentially.
    serial = "--serial" in argv
    total_workers = int(opt("--workers", str(PASSK_TOTAL_WORKERS)))
    #: Dry run with the offline stub, no model calls.
    offline = "--offline" in argv
    limit = int(opt("--limit")) if "--limit" in argv else None
    budget_usd = opt("--judge-budget-usd")
    budget_ledger = Path(opt("--judge-budget-ledger")) if "--judge-budget-ledger" in argv else None
    if not report_only and not offline and (budget_usd is None or budget_ledger is None):
        print("[passk] paid repeats require --judge-budget-usd and one shared --judge-budget-ledger")
        return 2
    # k<3 is allowed for a cheap look, but the reading must not be used to decide an effect.

    if k < 3 and not report_only:
        print(f"[passk] k={k} < 3: this reading must not be used to decide whether an "
              f"effect exists. It is fine as a cheap run; no conclusion may rest on it.")

    import yaml
    job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    job_id, task = job["job_id"], job["task_type"]
    base = ROOT / "results" / task / job_id
    if not base.is_dir():
        print(f"[passk] this job has no batch yet: {base}")
        return 2

    # ---- sample batches: all reuse the same `cases.jsonl` ----
    src = pick_source_batch(base, from_batch)
    if src is None:
        print(f"[passk] no real batch under {base} (`YYYYmmdd-HHMMSS` with a cases.jsonl)")
        return 2
    stamp = src.name
    sample_dirs = []
    for i in range(1, k + 1):
        d = base / (SAMPLE_FMT % (stamp, i))
        if not report_only:
            d.mkdir(parents=True, exist_ok=True)
            for f in ("cases.jsonl", "batch.json"):
                if (src / f).is_file() and not (d / f).is_file():
                    shutil.copy(src / f, d / f)
        sample_dirs.append(d)
    print(f"[passk] prompt source {src.name} (reused byte for byte) · k={k} · "
          f"sample directories {[d.name for d in sample_dirs]}")

    if not report_only:
        rcs = _run_samples(job_path, sample_dirs, models, serial=serial,
                           total_workers=total_workers,
                           offline=offline, limit=limit,
                           budget_usd=budget_usd, budget_ledger=budget_ledger)
        for i, rc in enumerate(rcs, 1):
            if rc != 0:
                print(f"[passk] sample {i} status {'not_started' if rc is None else rc} -- stopping; no aggregation "
                      f"over incomplete samples")
                return rc if rc is not None else 2

    sample_dirs = _assert_samples_distinct(sample_dirs)
    if len(sample_dirs) < 3:
        print(f"[passk] only {len(sample_dirs)} comparable samples left (<3) -- "
              f"no noise floor below k=3")
        return 1
    samples = [_load(d, models, semantic=not offline) for d in sample_dirs]
    have = [i for i, s in enumerate(samples, 1) if s]
    if len(have) < 3 or len(have) != len(sample_dirs):
        print(f"[passk] only {len(have)} samples have usable cells -- noise cannot be "
              f"reported unless every planned sample is usable and at least three exist")
        return 1
    rep = _aggregate(samples)
    rep["judgment_mode"] = "offline_code" if offline else "semantic_2plus1"
    rep["noise_components"] = [] if offline else ["solver_sampling", "adaptive_judge_sampling"]
    rep["judge_in_repeat"] = not offline
    if not offline:
        rep["judge_note"] = ("Saved semantic judgments use independent 2+1 votes; variation combines "
                             "solver sampling and judge sampling, not solver-only noise. "
                             "Report judge repeatability separately.")
    rep["job_id"], rep["prompt_source_batch"] = job_id, src.name
    rep["samples"] = [d.name for d in sample_dirs]
    # Per-run prefix-cache readings.
    rep["prefix_cache"] = _cache_by_sample(sample_dirs, models)
    _ms = [c for c in rep["prefix_cache"] if c.get("status") == "measured"]
    _fr = sum(c["in_fresh"] for c in _ms)
    _ca = sum(c["in_cached"] for c in _ms)
    rep["prefix_cache_pooled"] = round(_ca / (_fr + _ca), 4) if (_fr + _ca) else None
    rep["prefix_cache_ceiling"] = round(1 - 1 / len(_ms), 4) if _ms else None
    rep["prefix_cache_note"] = (
        "Per-sample input cache rate (`billed.in_cached / in_total`). "
        "The per-sample number is scheduling noise; do not read it alone -- under "
        "concurrency, which sample reaches a cell first is random, so the leader pays full "
        "price while the others ride the cache, and two samples describing the same "
        "underlying event can show very different per-sample rates. "
        "Read only `prefix_cache_pooled` against `prefix_cache_ceiling` (= 1 - 1/k). "
        "On the google backend this is always 0 because of an accounting gap "
        "(`_google_usage` does not map `cachedContentTokenCount`), not a miss; "
        "do not read it as 0.")

    out = ROOT / "results" / task / job_id / f"reliability-{stamp}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[passk] k={rep['k']} · comparable cells {rep['n_grids']}")
    print(f"{'dimension':26}{'per-sample means':34}{'noise|Δ|':>9}{'flip':>8}  pass^k")
    for d, e in rep["dims"].items():
        pk = (" ".join(f"{v}" for v in e.get("pass_k", {}).values())) if e.get("pass_k") else ""
        print(f"{d:26}{str(e['means']):34}{e['noise_max_abs_delta']:>9}{e['flip_rate']:>8}  {pk}")
    print(f"\nnoise floor: max {rep['noise_floor_max']} · median {rep['noise_floor_median']}")

    # ---- prefix cache: read the pooled rate against the ceiling 1 - 1/k (per-sample
    # rates are scheduling noise under concurrency) ----
    print(f"\ninput prefix cache (pooled; theoretical ceiling for k={len(rep['prefix_cache'])} "
          f"is 1 - 1/k):")
    for c in rep["prefix_cache"]:
        if c.get("status") != "measured":
            print(f"  {c['sample']:28} -- {c.get('status')} (not read as 0)")
            continue
        print(f"  {c['sample']:28} cache rate {c['cache_rate']:.1%} "
              f"(cached {c['in_cached']:,} / fresh {c['in_fresh']:,} · {c['n_rows_measured']} cells)"
              f"  <- per-sample figures are scheduling noise; do not read them alone")
    pooled, ceil = rep.get("prefix_cache_pooled"), rep.get("prefix_cache_ceiling")
    if pooled is not None and ceil:
        verdict = ("at the ceiling; concurrent scheduling has no room left" if pooled >= ceil - 0.05
                   else "below the ceiling -- the windows still miss each other, or this "
                        "backend does no implicit caching")
        print(f"  => pooled {pooled:.1%} / ceiling {ceil:.1%} -- {verdict}")

    # ---- per-model self-noise ----
    print("\nper-model self-noise (the same frozen cases run k times; each model's own "
          "range) -- this column answers 'same patient, same record: does it behave the same "
          "each time'. Reported, never scored:")
    for d, e in rep["dims"].items():
        bs = e.get("by_solver") or {}
        if not bs:
            continue
        worst = e.get("self_noise_worst") or (None, None)
        row = " · ".join(f"{s}={v['self_noise']:.3f}"
                         for s, v in sorted(bs.items(), key=lambda kv: -kv[1]["self_noise"]))
        pooled = e["noise_max_abs_delta"]
        flag = ""
        if worst[0] is not None and pooled and worst[0] > 2 * pooled:
            flag = ("  pooled noise underestimates it -- the least stable model exceeds "
                    f"the pooled value {worst[0] / pooled:.1f}x")
        print(f"  {d:26} pooled {pooled:.4f} | {row}{flag}")
    print("  How to read: the mean reports level, self-noise reports consistency; the two "
          "are not interchangeable. Cases are reused byte for byte; the observed variation "
          "must be read with the declared judging mode.")

    print("Mean-level noise and per-cell flip rate are two numbers, not interchangeable "
          "-- the flip rate is often an order of magnitude larger, because jitter cancels "
          "in the mean.")
    print(f"Judging contribution: {rep['judge_note']}")
    print(f"[passk] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
