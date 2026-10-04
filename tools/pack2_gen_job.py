"""pack2_gen_job.py -- compose the pack-2 acute-triage job (`inputs/pack2-triage.job.yaml`, N items).

The ddx-workup cases are copied verbatim (raw + latent) and get one latent key `pack2` (class knob)
holding the plan -- disposition (4 classes x the hidden disease H), focal indicator, wording, acute
event, distractor density, prior tier. Every assignment is a sha256 of the case id inside its
stratum, independent of every case feature. The composition is checked on the emitted batch by the
shared audit (`tools/pack_audit.py --pack pack2`, SA-3: emitted cells equal planned cells).

Rules:
  * class table: the four dispositions in equal shares of N, each halved over H
    (`haenv.pack2.audit.quotas`);
  * pool: the cases whose deterministic build passes the shared realism predicates (SA-7) and shows
    no wearable reading in (T-3, T] against the NEWS2-0 background; for N > 50 the workup generator
    adds variants of every condition, ceil(N/50) - 1 more rounds (independent conditions 5 per
    round from variant 6, the others 3 per round from variant 4), built here deterministically to
    read the same rules;
  * selection: the first H0 cases by sha256(`pack2-select` + seed, case id); the H1 cases follow
    the chosen H0 cases' index-time T mix (largest remainder), first by the same hash inside each T,
    so the history length carries no trace of H;
  * disposition: dealt in turn inside each (H, registry urgency) stratum, the cases in order of
    their stream count, so neither the urgency prior nor the record size carries class signal.

Zero model calls:
    PYTHONPATH=<repo> python tools/pack2_gen_job.py --ref-batch <deterministic ddx-workup batch> \
        [--n 50] [--seed S] [--out inputs/pack2-triage.job.yaml] [--work <dir>]
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

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

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "tools"))

from haenv import canary as CANARY  # noqa: E402
from haenv import pack_size as PS             # noqa: E402
from haenv.pack2 import world as W           # noqa: E402
from haenv.pack2.audit import quotas         # noqa: E402

SOURCE_JOB = ROOT / "inputs" / "ddx-workup.job.yaml"
OUT = ROOT / "inputs" / "pack2-triage.job.yaml"
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
    from haenv.shared_audit import load_items, not_said_by_patient, said_sentences, side_effect_complaint
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


def topup_cases(n: int) -> list[dict]:
    """Workup-generator variants beyond the workup, ceil(n/50) - 1 rounds of them (none for n <= 50)."""
    rounds = math.ceil(n / PS.DEFAULT_N) - 1
    if rounds <= 0:
        return []
    from p4_gen_job import _topup
    return _topup(WORKUP_VARIANTS[0] * rounds, WORKUP_VARIANTS[1] * rounds)


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
    focal = {}
    for c in W.DISPOSITIONS:
        ids = sorted([i for i, d in disp.items() if d == c], key=lambda x: W.u01("pack2-focal", x))
        off = int(W.u01("pack2-focal-off", c) * len(W.FOCALS))
        for k, cid in enumerate(ids):
            focal[cid] = W.FOCALS[(k + off) % len(W.FOCALS)]
    wording = {}
    for c in W.DISPOSITIONS:
        for f in W.FOCALS:
            ids = sorted([i for i in disp if disp[i] == c and focal[i] == f], key=lambda x: W.u01("pack2-wording", x))
            off = int(W.u01("pack2-wording-off", c, f) * 2)
            for k, cid in enumerate(ids):
                wording[cid] = W.WORDINGS[(k + off) % 2]
    ev = W.events_table()
    # Acute event per case: one eligible row by hash (observation items: benign self-limited rows
    # only). Distractor density (high/low) alternates inside each disposition x focal cell.
    event, high_of = {}, {}
    for cid in sorted(disp):
        raw, lat = by_id[cid]["raw"], by_id[cid]["latent"]
        conds = {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or []}
        rows = W.eligible_events(focal[cid], str(lat.get("ddx_diagnosis") or ""), conds, ev)
        if disp[cid] == "watchful_waiting":                  # nothing in an observation item warrants a visit
            rows = [r for r in rows if r["cause"] == "benign" and r["self_limited"]]
        event[cid] = rows[int(W.u01("pack2-event", cid) * len(rows)) % len(rows)]
    for c in W.DISPOSITIONS:
        for f in W.FOCALS:
            ids = sorted([i for i in disp if disp[i] == c and focal[i] == f], key=lambda x: W.u01("pack2-dd-level", x))
            off = int(W.u01("pack2-dd-off", c, f) * 2)
            for k, cid in enumerate(ids):
                high_of[cid] = (k + off) % 2 == 0
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
        pl = {"disposition": disp[cid], "focal": focal[cid], "wording": wording[cid], "event": row["id"],
              "cause": row["cause"], "explains_away": ea, "distractors": n_dd, "wording_cross": cross,
              "prior_tier": tier}
        plans.append({"case_id": cid, "raw": raw, "latent": {**lat, W.LATENT_KEY: pl}})
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
    extra = topup_cases(a.n)
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
    quota = quotas(a.n)
    counts = {h: {d: quota[(h, d)] for d in W.DISPOSITIONS} for h in (0, 1)}
    need = {h: sum(counts[h].values()) for h in (0, 1)}
    # Pass 1: every candidate built as a pack-2 item; the shared realism predicates drop the cases
    # whose patient sentences fail them.
    pool, _ = pool_of(by_id, ref_cases, {})
    every = plan(by_id, pool, {h: PS.apportion(len(pool[h]), dict.fromkeys(W.DISPOSITIONS, 1)) for h in (0, 1)},
                 ref_cases)
    hits = realism_hits(run_builds(every, work, "pack2-pass1", PACK_HEAD), [p["case_id"] for p in every])
    drop = {cid: "; ".join(v) for cid, v in hits.items()}
    pool, excluded = pool_of(by_id, ref_cases, drop)
    plans = plan(by_id, select(pool, need, ref_cases, seed), counts, ref_cases)
    print(f"[pack2_gen_job] N {a.n}; top-up {len(extra)}; pool H0 {len(pool[0])} H1 {len(pool[1])}; "
          f"excluded {len(excluded)}")
    for e in excluded:
        print(f"[pack2_gen_job] excluded {e['case_id']} (H{e['H']}): {e['why']}")
    alloc_blob = json.dumps([{"case_id": p["case_id"], **p["latent"][W.LATENT_KEY]} for p in plans],
                            ensure_ascii=False, sort_keys=True).encode("utf-8")
    header = CANARY.block().splitlines()
    prov = {"generator": "tools/pack2_gen_job.py", "source_job": "inputs/ddx-workup.job.yaml",
            "source_job_sha16": sha16(src_bytes), "ref_cases_sha16": sha16(ref_bytes),
            "alloc_sha16": sha16(alloc_blob), "tables_sha16": W.tables_sha16()}
    if extra:
        prov["topup"] = {"rounds": math.ceil(a.n / PS.DEFAULT_N) - 1, "cases": len(extra),
                         "built_sha16": sha16(topup_bytes)}
    job = {"job_id": "pack2-triage", "pack": PS.header(a.n, seed), **PACK_HEAD, "_provenance": prov, "cases": plans}
    n_h = {h: sum(v for (hh, _d), v in quotas(a.n).items() if hh == h) for h in (0, 1)}
    text = "\n".join(header) + "\n" + (
        "# Generated file -- do not edit by hand. Rebuild: PYTHONPATH=. python tools/pack2_gen_job.py "
        f"--ref-batch <ddx-workup deterministic batch> --n {a.n}\n"
        f"# Pack 2 · acute triage: {len(plans)} ddx-workup worlds ({n_h[0]} H0 + {n_h[1]} H1) with today's acute visit;\n"
        "# disposition x H by construction.\n"
        "# SYNTHETIC data, evaluation only, not medical advice.\n")
    text += yaml.safe_dump(job, allow_unicode=True, sort_keys=False, width=200)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(f"[pack2_gen_job] wrote {a.out} ({len(plans)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
