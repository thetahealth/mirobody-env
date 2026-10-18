"""pack2_gen_job.py -- compose the pack-2 acute-triage job (`inputs/pack2-triage.job.yaml`): a pool of
POOL_FACTOR x N items, from which the audit picks the published pack of N.

The ddx-workup cases are copied verbatim (raw + latent) and get one latent key `pack2` (class knob)
holding the plan -- disposition (4 classes x the hidden disease H), value band and symptom pool of
the complaint, focal indicator, wording, acute event, distractor density, prior tier. Every assignment is a sha256 of the case id inside its
stratum, independent of every case feature. The composition is checked on the emitted batch by the
shared audit (`tools/validation/pack_audit.py --pack pack2`, SA-3: emitted cells equal planned cells).

Rules:
  * class table: the four dispositions in equal shares of the pool size, each halved over H
    (`haenv.pack2.audit.quotas`); the focal findings, wording, symptom pool and distractor level
    take turns inside each (H, disposition) stratum;
  * pool: the cases whose deterministic build passes the shared realism predicates (SA-7) and shows
    no wearable reading in (T-3, T] against the NEWS2-0 background; when the workup is too small for
    the pool the workup generator adds variants of its conditions (independent conditions 5 per
    round from variant 6, the others 3 per round from variant 4), built here deterministically to
    read the same rules;
  * selection: the first H0 cases by sha256(`pack2-select` + seed, case id); the H1 cases follow
    the chosen H0 cases' index-time T mix (largest remainder), first by the same hash inside each T,
    so the history length carries no trace of H;
  * disposition: dealt in turn inside each (H, registry urgency) stratum, the cases in order of
    their stream count, so neither the urgency prior nor the record size carries class signal.

Zero model calls:
    PYTHONPATH=<repo> python tools/generation/pack2_gen_job.py --ref-batch <deterministic ddx-workup batch> \
        [--n 50] [--seed S] [--out inputs/pack2-triage.job.yaml] [--work <dir>]
SYNTHETIC data, evaluation only, not medical advice.
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
import collections
import functools
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys

import yaml

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "tools"))

import haenv.provenance.canary as CANARY  # noqa: E402
import haenv.world.validation.pack_size as PS             # noqa: E402
from haenv.pack2 import world as W           # noqa: E402
from haenv.pack2.audit import POOL_FACTOR, quotas  # noqa: E402

SOURCE_JOB = ROOT / 'benchmarks' / 'inputs' / 'ddx-workup.job.yaml'
OUT = ROOT / 'benchmarks' / 'inputs' / 'pack2-triage.job.yaml'
#: variants per condition in the workup: independent (H0) conditions v1-v5, the others v1-v3
WORKUP_VARIANTS = (5, 3)
#: parallel deterministic builds (candidate pools)
BUILD_CHUNKS = 2

#: job keys of the workup build and of the pack-2 build
WORKUP_HEAD = {"task_type": "joint_dx", "multiround": False, "include_baseline": True, "models": [], "gated": True}
PACK_HEAD = {"task_type": "joint_dx", "multiround": False, "include_baseline": True, "models": [], "sample_cases": 5,
             "report": "eval-pack2-triage.md", "gated": False, "probe_id": "pack2.triage",
             "plugins": {"judges": ["haenv.pack2"]}}

#: Wearable eligibility: a reading in (T-3, T] that contradicts the NEWS2-0 background.
WEAR_LIMITS = {"systolic_bp": (None, 138), "spo2": (96, None), "body_temp": (36.1, 38.0)}
WEAR_WINDOW_DAYS = 3


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def wearable_eligible(case: dict) -> tuple[bool, str]:
    T = int(case["prediction_context"]["prediction_time_T"])
    for s, (lo, hi) in WEAR_LIMITS.items():
        for p in case["longitudinal_data"].get(s) or []:
            if not (T - WEAR_WINDOW_DAYS < p["ts"] <= T):
                continue
            v = float(p["value"])
            too_low = lo is not None and v < lo
            too_high = hi is not None and (v >= hi if s == "systolic_bp" else v > hi)
            if too_low or too_high:
                return False, f"{s}={v}@{p['ts']}"
    return True, ""


def deal(ids: list[str], counts: dict[str, int], tag: str, stratum=None) -> dict[str, str]:
    """Round-robin over the classes in order, skipping exhausted ones: sizes follow `counts`. With
    `stratum` (case id -> sort key), the cases run in that order, so each stratum gets the classes
    in turn."""
    order = sorted(ids, key=lambda c: (stratum(c) if stratum else (), W.u01(tag, c)))
    left = dict(counts)
    classes = [c for c in W.DISPOSITIONS if left.get(c)]
    out, k = {}, 0
    for cid in order:
        while not left[classes[k % len(classes)]]:
            k += 1
        c = classes[k % len(classes)]
        out[cid] = c
        left[c] -= 1
        k += 1
    return out


def realism_hits(batches: list[pathlib.Path], ids: list[str]) -> dict[str, list[str]]:
    """{case_id: violations} of the shared audit's realism predicates (SA-7) on built pack-2 items;
    a case of `ids` the build did not emit as a pack-2 item counts as one violation."""
    from haenv.world.validation.shared_audit import load_items
    from haenv.world.validation.shared_audit import not_said_by_patient
    from haenv.world.validation.shared_audit import said_sentences
    from haenv.world.validation.shared_audit import side_effect_complaint
    preds = {"not_said_by_patient": not_said_by_patient(said_sentences()),
             "side_effect_complaint": side_effect_complaint}
    out, seen = {}, set()
    for b in batches:
        for it in load_items(b, W.BLOCK, lambda g: g.get("disposition") if g.get("pack") == "pack2" else None):
            seen.add(it["case_id"])
            bad = [f"{k}: {v}" for k, fn in preds.items() for v in fn(it)]
            if bad:
                out[it["case_id"]] = bad
    out.update({cid: ["not emitted as a pack-2 item"] for cid in ids if cid not in seen})
    return out


def is_hidden(lat: dict) -> int:
    return int(lat.get("ddx_join_gold") in ("unified", "comorbidity"))


#: share of candidates expected to survive the realism and wearable checks (about 90% do)
KEEP = 0.8


def topup_cases(need: dict) -> list[dict]:
    """Workup-generator variants beyond the workup, the fewest rounds that cover `need` = {H: pool
    items wanted}: H0 conditions come WORKUP_VARIANTS[0] variants per round, the others
    WORKUP_VARIANTS[1]. The H1 items follow the index-time T mix of the H0 items (`select`), so
    every T of the H0 candidates also needs its share of H1 candidates."""
    from tools.generation.p4_gen_job import _topup
    src = list(yaml.safe_load(SOURCE_JOB.read_bytes())["cases"])

    def made(r0: int, r1: int) -> list[dict]:
        return _topup(WORKUP_VARIANTS[0] * r0, WORKUP_VARIANTS[1] * r1) if (r0 or r1) else []

    def count(cases) -> collections.Counter:
        return collections.Counter((is_hidden(c["latent"]), int(c["latent"].get("index_time_T") or 0)) for c in cases)

    def h0_ok(have) -> bool:
        return sum(v for (h, _t), v in have.items() if h == 0) * KEEP >= need[0]

    def h1_ok(have) -> bool:
        n0 = sum(v for (h, _t), v in have.items() if h == 0)
        return all(have[(1, t)] * KEEP >= need[1] * v / n0 for (h, t), v in have.items() if h == 0)

    r0 = next((r for r in range(60) if h0_ok(count(src + made(r, 0)))), None)
    r1 = r0 is not None and next((r for r in range(60) if h1_ok(count(src + made(r0, r)))), None)
    if r1 is None or r0 is None:
        raise SystemExit(f"[pack2_gen_job] no top-up reaches a pool of {need}")
    return made(r0, r1)


def run_builds(cases: list[dict], work: pathlib.Path, name: str, head: dict) -> list[pathlib.Path]:
    """Deterministic `haenv build`s of `cases` under the job keys `head`, in parallel chunks; returns
    the batch directories (a batch the build's own gates refuse still carries its cases)."""
    work.mkdir(parents=True, exist_ok=True)
    k = max(1, min(BUILD_CHUNKS, len(cases)))
    procs = []
    for i in range(k):
        jid = f"{name}-{i:02d}"
        jp = work / f"{jid}.job.yaml"
        jp.write_text(yaml.safe_dump({"job_id": jid, **head, "cases": cases[i::k]},
                                     allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
        log = open(work / f"{jid}.log", "w", encoding="utf-8")
        procs.append((work / jid, jid, log, subprocess.Popen(
            [sys.executable, "-m", "haenv", "build", str(jp), "--gen", "deterministic", "--fresh",
             "--set", "env_file=/dev/null"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "HAENV_OUTPUT_ROOT": str(work / jid)})))
    out = []
    for root, jid, log, pr in procs:
        rc = pr.wait()
        log.close()
        got = sorted(root.glob(f"results/*/{jid}/*/cases.jsonl"))
        if not got:
            raise SystemExit(f"[pack2_gen_job] build {jid} wrote no cases (rc {rc}); see {log.name}")
        out.append(got[-1].parent)
    return out


def read_batches(batches: list[pathlib.Path], order: list[str]) -> tuple[dict, bytes]:
    """{case_id: built case} and the cases.jsonl lines in `order`."""
    rows = {}
    for b in batches:
        for line in (b / "cases.jsonl").read_text(encoding="utf-8").splitlines():
            rows[json.loads(line)["case_id"]] = line
    cases = {cid: json.loads(v)["case"] for cid, v in rows.items()}
    return cases, "".join(rows[c] + "\n" for c in order if c in rows).encode("utf-8")


def pool_of(by_id: dict, ref_cases: dict, drop: dict) -> tuple[dict, list]:
    """Candidates by H: built in the reference, wearable-consistent, not in `drop`."""
    pool, excluded = {0: [], 1: []}, []
    for cid, c in by_id.items():
        H = is_hidden(c["latent"])
        ok, why = wearable_eligible(ref_cases[cid]) if cid in ref_cases else (False, "not in reference batch")
        if ok and cid in drop:
            ok, why = False, drop[cid]
        if not ok:
            excluded.append({"case_id": cid, "H": H, "why": why})
            continue
        pool[H].append(cid)
    return pool, excluded


def select(pool: dict, need: dict, ref_cases: dict, seed: str) -> dict:
    tag = PS.seeded("pack2-select", seed)
    PS.short({"H0": need[0]}, {"H0": len(pool[0])}, "pack 2 candidate cases")
    chosen = {0: sorted(pool[0], key=lambda c: W.u01(tag, c))[:need[0]]}
    # H1 follows the index-time mix of the chosen H0 cases, so history length does not reveal H.
    T_of = {cid: int(ref_cases[cid]["prediction_context"]["prediction_time_T"]) for h in (0, 1) for cid in pool[h]}
    t_mix = collections.Counter(T_of[c] for c in chosen[0])
    t_need = PS.apportion(need[1], {t: t_mix[t] for t in sorted(t_mix)})
    by_t = collections.defaultdict(list)
    for cid in pool[1]:
        by_t[T_of[cid]].append(cid)
    PS.short({f"H1 T={t}": k for t, k in t_need.items()}, {f"H1 T={t}": len(by_t[t]) for t in t_need},
             "pack 2 candidate cases")
    chosen[1] = [c for t, k in t_need.items() for c in sorted(by_t[t], key=lambda c: W.u01(tag, c))[:k]]
    return chosen


def _draw_event(cid: str, focal: str, rows: list[dict], ev: dict) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cause"]].append(r)
    weight = collections.Counter(r["cause"] for r in ev["events"] if r["focal"] == focal)
    kinds = sorted(by)
    t = W.u01("pack2-cause", cid) * sum(weight[k] for k in kinds)
    for k in kinds:
        t -= weight[k]
        if t < 0:
            break
    return by[k][int(W.u01("pack2-event", cid) * len(by[k])) % len(by[k])]


def plan(by_id: dict, chosen: dict, counts: dict, ref_cases: dict) -> list[dict]:
    """The pack-2 plan of every chosen case; `counts[h]` are the disposition counts of H = h."""
    # Dispositions turn inside each registry urgency, over the cases in stream-count order, so
    # neither the urgency prior nor the record size carries class signal.
    key = {cid: (str(((ref_cases[cid].get("adjudication") or {}).get("ddx") or {}).get("urgency") or ""),
                 len(ref_cases[cid].get("longitudinal_data") or {}))
           for h in (0, 1) for cid in chosen[h]}
    disp = {}
    for h in (0, 1):
        disp.update(deal(chosen[h], counts[h], f"pack2-disp-H{h}", stratum=key.get))
    H = {cid: is_hidden(by_id[cid]["latent"]) for cid in disp}
    ev = W.events_table()
    rows_of = {}                                             # case -> focal -> eligible event rows
    for cid in disp:
        raw, lat = by_id[cid]["raw"], by_id[cid]["latent"]
        conds = {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or []}
        rows_of[cid] = {f: W.eligible_events(f, str(lat.get("ddx_diagnosis") or ""), conds, ev) for f in W.FOCALS}
        if disp[cid] == "watchful_waiting":                  # nothing in an observation item warrants a visit
            rows_of[cid] = {f: [r for r in rs if r["cause"] == "benign" and r["self_limited"]]
                            for f, rs in rows_of[cid].items()}
    cells = collections.defaultdict(list)                   # (H, disposition, focal) -> ids, below
    focal = {}
    for h in (0, 1):                                         # focal findings in turn inside (H, disposition)
        for c in W.DISPOSITIONS:
            ids = sorted([i for i in disp if disp[i] == c and H[i] == h], key=lambda x: W.u01("pack2-focal", x))
            off = int(W.u01("pack2-focal-off", c, h) * len(W.FOCALS))
            turn = [W.FOCALS[(k + off) % len(W.FOCALS)] for k in range(len(W.FOCALS))]
            left = {f: len(ids) // len(turn) + (k < len(ids) % len(turn)) for k, f in enumerate(turn)}
            # a case first takes a focal finding for which the event pool has a row of its own
            # (known condition / hidden line) while that finding's share is open, so those rows
            # are drawn in proportion to what the cases can carry; the rest fill the open shares
            for cid in ids:
                own = [f for f in turn if left[f] and any(r["cause"] != "benign" for r in rows_of[cid][f])]
                if own:
                    f = own[int(W.u01("pack2-focal-own", cid) * len(own)) % len(own)]
                    focal[cid] = f
                    left[f] -= 1
            for cid in ids:
                if cid not in focal:
                    f = next(f for f in turn if left[f])
                    focal[cid] = f
                    left[f] -= 1
    for cid in disp:
        cells[(H[cid], disp[cid], focal[cid])].append(cid)

    def alternate(tag: str, values: tuple) -> dict[str, object]:
        """Inside every (H, disposition, focal) cell the values take turns (start by hash)."""
        out = {}
        for cell, ids in cells.items():
            off = int(W.u01(tag + "-off", *cell) * len(values))
            for k, cid in enumerate(sorted(ids, key=lambda x: W.u01(tag, x))):
                out[cid] = values[(k + off) % len(values)]
        return out

    wording = alternate("pack2-wording", W.WORDINGS)
    high_of = alternate("pack2-dd-level", (True, False))
    # Symptom pool (`marked` | `other`) of the complaint, taking turns inside each cell, and the value
    # band the focal is drawn from. G2 escalates a marked item one step above its numbers' band, so a
    # marked item's numbers come from the band below its disposition; `watchful_waiting` has no band
    # below and carries no marked sentence.
    symptom = alternate("pack2-symptom", ("marked", "other"))
    band = dict(disp)
    for cid in disp:
        if disp[cid] == "watchful_waiting":
            symptom[cid] = "other"
        elif symptom[cid] == "marked":
            band[cid] = W.DISPOSITIONS[W.DISPOSITIONS.index(disp[cid]) + 1]
    # Acute event per case: the cause kind first, in proportion to the pool's rows of that kind for
    # the focal finding among the kinds the case is eligible for (benign | known_condition |
    # hidden_line), then one row of the kind by hash.
    event = {cid: _draw_event(cid, focal[cid], rows_of[cid][focal[cid]], ev) for cid in sorted(disp)}
    plans = []
    for cid in sorted(disp):
        c = by_id[cid]
        raw, lat = c["raw"], c["latent"]
        row = event[cid]
        high = high_of[cid]
        n_dd = (2 + int(W.u01("pack2-dd-n", cid) * 2)) if high else int(W.u01("pack2-dd-n", cid) * 2)
        danger = disp[cid] in ("ed_now", "within_24h")
        cross = (wording[cid] == "alarming" and not danger) or (wording[cid] == "calm" and danger)
        ea = bool(row.get("explains_away"))
        tier = "hard" if (ea or cross) and high else "easy" if (not ea and not cross and not high) else "mid"
        pl = {"disposition": disp[cid], "band": band[cid], "focal": focal[cid], "wording": wording[cid],
              "symptom": symptom[cid], "event": row["id"], "cause": row["cause"], "explains_away": ea,
              "distractors": n_dd, "wording_cross": cross, "prior_tier": tier}
        from tools.generation.p4_gen_job import with_pack_age          # the age band the build prints (`build.pack_age_of`)
        plans.append(with_pack_age({"case_id": cid, "raw": raw, "latent": {**lat, W.LATENT_KEY: pl}}))
    return plans


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref-batch", required=True, help="a deterministic build of inputs/ddx-workup.job.yaml")
    ap.add_argument("--n", type=int, default=PS.DEFAULT_N)
    ap.add_argument("--seed", default=None, help="selection seed (else HAENV_PACK_SEED, else the public seed)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--work", default=None, help="scratch directory of the candidate builds")
    a = ap.parse_args(argv)
    seed = PS.cli_seed(a.seed)
    src_bytes = SOURCE_JOB.read_bytes()
    src = yaml.safe_load(src_bytes)
    ref_cases, _ = read_batches([pathlib.Path(a.ref_batch)], [])
    ref_bytes = (pathlib.Path(a.ref_batch) / "cases.jsonl").read_bytes()
    pool_n = POOL_FACTOR * PS.pool_n(a.n)      # shared by the packs of the nested sizes (`haenv/world/validation/pack_size.py`)
    quota = quotas(pool_n)
    counts = {h: {d: quota[(h, d)] for d in W.DISPOSITIONS} for h in (0, 1)}
    need = {h: sum(counts[h].values()) for h in (0, 1)}
    extra = topup_cases(need)
    clash = sorted({c["case_id"] for c in extra} & {c["case_id"] for c in src["cases"]})
    if clash:
        raise SystemExit(f"[pack2_gen_job] top-up case ids collide with the workup: {clash[:5]}")
    work = pathlib.Path(a.work or (pathlib.Path(os.environ["HAENV_OUTPUT_ROOT"]) / "pack2-gen"
                                   if os.environ.get("HAENV_OUTPUT_ROOT") else ROOT / "derived" / "pack2"))
    topup_bytes = b""
    if extra:
        built, topup_bytes = read_batches(run_builds(extra, work, "pack2-topup", WORKUP_HEAD),
                                          [c["case_id"] for c in extra])
        ref_cases = {**ref_cases, **built}
    by_id = {c["case_id"]: c for c in list(src["cases"]) + extra}
    # Pass 1: every candidate built as a pack-2 item; the shared realism predicates drop the cases
    # whose patient sentences fail them.
    pool, _ = pool_of(by_id, ref_cases, {})
    every = plan(by_id, pool, {h: PS.apportion(len(pool[h]), dict.fromkeys(W.DISPOSITIONS, 1)) for h in (0, 1)},
                 ref_cases)
    hits = realism_hits(run_builds(every, work, "pack2-pass1", PACK_HEAD), [p["case_id"] for p in every])
    drop = {cid: "; ".join(v) for cid, v in hits.items()}
    pool, excluded = pool_of(by_id, ref_cases, drop)
    plans = plan(by_id, select(pool, need, ref_cases, seed), counts, ref_cases)
    print(f"[pack2_gen_job] N {a.n}; pool {pool_n}; top-up {len(extra)}; pool H0 {len(pool[0])} H1 {len(pool[1])}; "
          f"excluded {len(excluded)}")
    for e in excluded:
        print(f"[pack2_gen_job] excluded {e['case_id']} (H{e['H']}): {e['why']}")
    alloc_blob = json.dumps([{"case_id": p["case_id"], **p["latent"][W.LATENT_KEY]} for p in plans],
                            ensure_ascii=False, sort_keys=True).encode("utf-8")
    header = CANARY.block().splitlines()
    prov = {"generator": 'tools/generation/pack2_gen_job.py', "source_job": 'benchmarks/inputs/ddx-workup.job.yaml',
            "source_job_sha16": sha16(src_bytes), "ref_cases_sha16": sha16(ref_bytes),
            "alloc_sha16": sha16(alloc_blob), "tables_sha16": W.tables_sha16()}
    if extra:
        prov["topup"] = {"cases": len(extra), "built_sha16": sha16(topup_bytes)}
    job = {"job_id": "pack2-triage", "pack": PS.header(a.n, seed), **PACK_HEAD, "_provenance": prov, "cases": plans}
    n_h = {h: sum(v for (hh, _d), v in quotas(pool_n).items() if hh == h) for h in (0, 1)}
    text = "\n".join(header) + "\n" + (
        "# Generated file -- do not edit by hand. Rebuild: PYTHONPATH=. python tools/generation/pack2_gen_job.py "
        f"--ref-batch <ddx-workup deterministic batch> --n {a.n}\n"
        f"# Pack 2 · acute triage: pool of {len(plans)} ddx-workup worlds ({n_h[0]} H0 + {n_h[1]} H1) with today's acute visit,\n"
        f"# the published pack of {a.n} is picked from it by the audit; disposition x H by construction.\n"
        "# SYNTHETIC data, evaluation only, not medical advice.\n")
    text += yaml.safe_dump(job, allow_unicode=True, sort_keys=False, width=200)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(f"[pack2_gen_job] wrote {a.out} ({len(plans)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
