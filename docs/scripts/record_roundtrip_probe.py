"""The **three-arm probe** for the case-report round trip -- decides whether Phase 3 happens (2026-09-17).

## The objection this needs to answer

A 2026-09-17 objection: "Does the case-report closed loop actually mean anything (after all, if the
input has no case report, the loop fails trivially; and if it does, the core points in the
case report should not drift, which is the normal expectation anyway)."

**The second objection kills one version outright**: if `render` reads `latent` from
`job.yaml` and writes the fields out as prose, and then `extract` reads it back, that measures
**the LLM's transcription noise**, and the recovery rate would sit near 100% regardless of
synthetic-patient quality. => `render_record`'s signature **only accepts the solver payload**;
the ground truth cannot get in (this is not "we forgot to pass it").

The first objection does not hold: the closed loop's comparison target is **`job.yaml`**, not
the original text. A hand-written job enters the loop just as well.

## Three arms

| arm | what it does | what it measures |
|---|---|---|
| **a - oracle** | takes fields directly from the structured payload (no prose) | upper bound on **expression loss** |
| **b - prose** | payload -> render -> extract | actual recovery rate |
| **c - negative control** | render **someone else's** payload -> extract, still compared against this case | **this is the one that matters most** |

Warning: **arm c is the make-or-break line for this proposal.** If the same fields can be
recovered from another patient's case report, this metric is measuring a **population
prior**, not this patient -- the same family of failure as "a good fit to eight points with
two free parameters is not evidence". **c ≈ b => Phase 3 is off.**

## The denominator is fixed first

`record_render.ROUNDTRIP_FIELDS` -- the class-A patient facts that a case report can
reasonably be expected to yield back. Class-C generation knobs (`index_time_T` /
`event_density` / `noise`) do not describe the patient and should not appear in the case
report at all (`provenance.RARELY_IN_TEXT` already registers this boundary).
**Without fixing the denominator first, this number can be tuned to anything.**

Usage (**consumes LLM tokens, runs on the generation-side flash tier**):

    python docs/scripts/record_roundtrip_probe.py --n 10
    python docs/scripts/record_roundtrip_probe.py --n 3 --offline   # cache only, zero cost

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from haenv import extract as ex          # noqa: E402
from haenv import record_render as rr    # noqa: E402

OUT = ROOT / "docs" / "scripts" / "_record_roundtrip_result.json"


def _cfg() -> dict:
    import yaml

    from haenv import data_root
    return yaml.safe_load((data_root() / "config.yaml").read_text(encoding="utf-8")) or {}


def _job_specs() -> dict[str, dict]:
    """`case_id -> raw` (the copy in job.yaml) -- the round trip's **comparison target**."""
    import yaml
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(str(ROOT / "inputs" / "*.job.yaml"))):
        d = yaml.safe_load(pathlib.Path(f).read_text(encoding="utf-8")) or {}
        for c in (d.get("cases") or []):
            if c.get("case_id") and isinstance(c.get("raw"), dict):
                out[str(c["case_id"])] = c["raw"]
    return out


def _payloads(n: int) -> list[tuple[str, dict]]:
    """`(case_id, sp)` from the most recent batches. **Only takes `sp`** -- `vp` is never looked at."""
    # Warning: **dedup by case_id.** The same case has a payload in multiple batches;
    # without dedup, n=10 contained 3 duplicates -- "10 cases" and "7 cases counted three
    # times each" look identical in the reading, and the latter's variance is fake
    # (duplicated entries are perfectly correlated).
    rows: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for f in sorted(glob.glob(str(ROOT / "results" / "*" / "*" / "*" / "payloads.jsonl")),
                    key=os.path.getmtime, reverse=True):
        for line in pathlib.Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cid = str(r.get("case_id") or "")
            if r.get("sp") and cid and cid not in seen:
                seen.add(cid)
                rows.append((cid, r["sp"]))
            if len(rows) >= n:
                return rows
    return rows


def _oracle_facts(sp: dict) -> dict:
    """Arm a: **no prose involved**, takes those fields directly from the structured payload.

    Anything unavailable is left blank -- blank means "this trace does not carry this fact",
    which is exactly the **expression loss**.
    """
    # Warning: **key names are as measured in the payload, not as named in job.yaml.** The two
    # are not the same vocabulary: job writes `devices`, the payload writes
    # `device_inventory`; job writes `dose_steps`, the payload only has per-point doses under
    # `longitudinal_data.dose_timeline`.
    # The first version used job's naming, and oracle reported 0.571 while prose reported
    # 0.857 -- **oracle was lower**, which cannot be true (oracle is an **upper bound** on
    # expression loss); the extraction path was missing two fields.
    up = sp.get("user_profile") or {}
    ld = sp.get("longitudinal_data") or {}
    w = [float(p["value"]) for p in (ld.get("weight") or []) if isinstance(p, dict)]
    dose = [p["value"] for p in (ld.get("dose_timeline") or []) if isinstance(p, dict)]
    seen, steps = set(), []
    for v in dose:                                   # dedup while preserving order -- the dose steps are ordered
        if v not in seen:
            seen.add(v)
            steps.append(v)
    got = {
        "age_range": up.get("age_range"),
        "sex": up.get("sex"),
        # Warning: **`drug` is deliberately left blank** -- measured: the payload has **no
        # drug name** (a full-text search for `tirzepatide`/`semaglutide`/the Chinese
        # name/`GLP` matches nothing), only the dose-step ladder.
        # This is a **genuine expression loss**, not an extraction-path bug: the world layer
        # never exposes drug identity to the solver.
        "drug": None,
        "dose_steps": steps or None,
        "devices": up.get("device_inventory") or up.get("devices"),
        "start_weight": (w[0] if w else None),
        "nadir_weight": (min(w) if w else None),
    }
    return {k: v for k, v in got.items() if v is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--model", default="gemini-3.8-flash")
    a = ap.parse_args()

    cfg = _cfg()
    specs = _job_specs()
    pays = [(cid, sp) for cid, sp in _payloads(a.n * 3) if cid in specs][:a.n]
    if not pays:
        print("🔴 盘上没有可用 payload(或 case_id 对不上 job.yaml)—— **未量**", file=sys.stderr)
        return 2
    print(f"扫描面:{len(pays)} 例 · 分母 {list(rr.ROUNDTRIP_FIELDS)}\n")

    res = {"a_oracle": [], "b_prose": [], "c_control": [], "leaks": [], "per_case": []}
    for i, (cid, sp) in enumerate(pays):
        want = specs[cid]
        row: dict = {"case_id": cid}

        # -- arm a - oracle (no cost)
        ra = rr.compare_facts(want, _oracle_facts(sp))
        res["a_oracle"].append(ra["rate"])
        row["a"] = ra["rate"]

        # -- arm b - prose (this case's own trace)
        try:
            text, origin = rr.render_record(sp, model=a.model, cfg=cfg, offline=a.offline)
            hits = rr.leak_words(text)
            res["leaks"].append(len(hits))
            ec = ex.text_to_case(text, cid, gold_path="absent", model=a.model,
                                 cfg=cfg, offline=a.offline)
            rb = rr.compare_facts(want, ec.raw)
            res["b_prose"].append(rb["rate"])
            row.update(b=rb["rate"], leak=hits, origin=origin,
                       b_fields=rb["per_field"], chars=len(text))
            # Warning: **keep one copy of the original text** -- the demo page needs to show
            # "what it actually rendered as", and re-rendering would produce a different copy
            # (the cache key is the same => it actually wouldn't, but binding "the copy shown"
            # to "the copy this number was computed from" is what makes it traceable).
            # Only the first case is kept: keeping all of them would bloat the results file,
            # and one is enough to illustrate the shape.
            if "sample" not in res:
                res["sample"] = {"case_id": cid, "text": text, "origin": origin,
                                 "want": {k: want.get(k) for k in rr.ROUNDTRIP_FIELDS},
                                 "got": {k: ec.raw.get(k) for k in rr.ROUNDTRIP_FIELDS},
                                 "per_field": rb["per_field"], "leak": hits}
        except Exception as e:                                # noqa: BLE001
            row["b_error"] = f"{type(e).__name__}: {str(e)[:160]}"

        # -- arm c - negative control (**someone else's** trace, still compared against this case)
        j = (i + 1) % len(pays)
        if j != i:
            try:
                other_text, _ = rr.render_record(pays[j][1], model=a.model, cfg=cfg,
                                                 offline=a.offline)
                ec2 = ex.text_to_case(other_text, cid, gold_path="absent", model=a.model,
                                      cfg=cfg, offline=a.offline)
                rc = rr.compare_facts(want, ec2.raw)
                res["c_control"].append(rc["rate"])
                row.update(c=rc["rate"], c_from=pays[j][0])
            except Exception as e:                            # noqa: BLE001
                row["c_error"] = f"{type(e).__name__}: {str(e)[:160]}"

        res["per_case"].append(row)
        print(f"  {cid}: a={row.get('a')} b={row.get('b')} c={row.get('c')}"
              f"{' · 泄漏 ' + str(row['leak']) if row.get('leak') else ''}"
              f"{' · ' + row['b_error'] if row.get('b_error') else ''}")

    def _m(xs):
        xs = [x for x in xs if x is not None]
        return round(st.mean(xs), 4) if xs else None

    summary = {"n": len(pays), "a_oracle": _m(res["a_oracle"]),
               "b_prose": _m(res["b_prose"]), "c_control": _m(res["c_control"]),
               "n_leaky": sum(1 for x in res["leaks"] if x), "model": a.model}
    print(f"\n{'臂':10s} {'恢复率':>8s}")
    print(f"{'a oracle':10s} {summary['a_oracle']}")
    print(f"{'b 散文':10s} {summary['b_prose']}")
    print(f"{'c 负对照':10s} {summary['c_control']}")
    print(f"\n泄漏词命中的例数:{summary['n_leaky']}/{len(pays)}")

    b, c = summary["b_prose"], summary["c_control"]
    if b is not None and c is not None:
        gap = round(b - c, 4)
        print(f"\n🔴 **判决**:b − c = {gap}")
        print("   c ≈ b ⇒ 这个指标量的是人群先验,不是这个病人 ⇒ **期 3 作废**"
              if gap < 0.15 else
              "   b 显著高于 c ⇒ 恢复率确实在量这个病人 ⇒ 期 3 成立")
    OUT.write_text(json.dumps({"summary": summary, "per_case": res["per_case"],
                               "sample": res.get("sample")},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n逐例读数已存 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
