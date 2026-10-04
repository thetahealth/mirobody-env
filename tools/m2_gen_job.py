"""m2_gen_job.py -- write pack 1 (differential diagnosis, `m2-pack1`) for any pack size N.

  python tools/m2_gen_job.py [--n N] [--seed S] [--out PATH] [--check]

Steps (zero model calls):
  1. pool: the v1.0.1 bridge set (F0, `docs/scripts/m2/alloc_cases.csv`, copied verbatim from
     `inputs/ddx-workup.job.yaml`) plus F1-F3 rows from `docs/scripts/m2/recompose_v3.py`, scaled by
     the pack's non-bridge item count over 34;
  2. compose: F1-F3 cases get one latent key `m2` (provenance class `knob`); F0 cases get nothing, so
     their world prompt, payload and gold stay byte-identical to v1.0.1;
  3. emit: the pool is built with the deterministic generator; a case the build refuses, or an F1-F3
     case with a symptom phrase that has no lay sentences, is not selected;
  4. select: `docs/scripts/m2/pack1_select.py` with the cell sizes of `haenv/m2/pack.py:quotas(N)`,
     ordered by sha256(case_id + seed).

The bridge takes its proportional share up to what the v1.0.1 bridge set supplies; it is a frozen
control set, so a shortfall is filled by new F1-F3 items in proportion and recorded in `_provenance`.
The seed comes from `--seed`, else HAENV_PACK_SEED, else the public seed; the job records only its
sha256 (`pack` header).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
from haenv import canary as CANARY  # noqa: E402
from haenv import pack_size as PS  # noqa: E402
from haenv.m2 import pack as PACK  # noqa: E402
from haenv.m2.core import (TEXT_SOURCE, acute_event_for, explain_away_render, scene_text,  # noqa: E402
                           visible_medication_text)

SRC = ROOT / "inputs" / "ddx-workup.job.yaml"
ALLOC = ROOT / "docs" / "scripts" / "m2" / "alloc_cases.csv"
OUT = ROOT / "inputs" / "m2-pack1.job.yaml"
RECOMPOSE = ROOT / "docs" / "scripts" / "m2" / "recompose_v3.py"
SELECT = ROOT / "docs" / "scripts" / "m2" / "pack1_select.py"

FRAMES = ("F0", "F1", "F2", "F3")
#: K-NT: number of gold tests; (5, 8] is the hard side (spec 4.3, knobs.json).
K_NT_HARD_MIN = 6
#: non-bridge items of a 50-item pack (scale 1 of the F1-F3 pool)
NB_PER_SCALE = 34


def _explain_away_rows() -> list[dict]:
    """The table rows minus retired ones (revision r2-E1: the same filter as `core.ea_rows`)."""
    doc = yaml.safe_load((ROOT / "registry" / "explain_away.yaml").read_text(encoding="utf-8")) or {}
    return [r for r in doc.get("rows") or [] if str(r.get("status") or "") != "retired"]


def k_ea(spec_id: str, known: list[str], med_text: str | None = None) -> str:
    """K-EA reading (profile only, spec 4.4): `by_known_condition` when a table row lists this spec
    and its explainer is one of the case's known conditions, `by_visible_line` when a row lists it
    with a visible-line explainer, else `none`. A row whose mechanism needs a drug counts only when
    that drug is in the visible medication text (`core.drug_visible`, revision r2-E2)."""
    from haenv.m2.core import drug_visible
    hit = "none"
    for r in _explain_away_rows():
        if spec_id not in (r.get("specs") or ()) or not drug_visible(r, med_text):
            continue
        if any(k in known for k in (r.get("explainer_known") or ())):
            return "by_known_condition"
        if r.get("explainer_line"):
            hit = "by_visible_line"
    return hit


#: Explainer base conditions (spec 3.3, `background_comorbidity.yaml:explainer_bases`).
NEW_BASES = ("hypothyroidism", "CAD", "CKD")
#: A base never equals a hidden line (audit BASE-1): the hidden components it would duplicate.
BASE_HIDDEN = {"hypothyroidism": {"JD-HYPO"}, "CAD": {"JD-CAD"}, "CKD": {"JD-CKD", "JD-CKM"}}


def _u(*path) -> float:
    h = hashlib.sha256("|".join(str(x) for x in ("m2-base",) + path).encode("utf-8")).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def base_candidates(spec_id: str) -> list[str]:
    """New bases the explain-away table lets explain a hidden finding of this spec, table order,
    minus any base that equals one of the spec's hidden components."""
    from haenv.m2.core import components
    comps = set(components(spec_id)) | {spec_id}
    out: list[str] = []
    for r in _explain_away_rows():
        if spec_id not in (r.get("specs") or ()):
            continue
        for b in r.get("explainer_known") or ():
            if b in NEW_BASES and b not in out and not (BASE_HIDDEN[b] & comps):
                out.append(b)
    return out


def assign_bases(rows: list[dict], by_id: dict) -> dict[str, str]:
    """case_id -> attached base. Whether an F1-F3 case carries a base, and which, is independent of its
    hidden line and arm: per frame, round(attach rate x frame size) cases in case-keyed order each draw
    a base from the frame's mix, where the attach rate and the mix are the share and the draws of the
    frame's positive-arm cases the explain-away table offers a base (the whole pool's when the frame
    offers none). A base that is a hidden component of the case or already on its record is skipped
    (BASE-1). The explain-away rendering follows only when the drawn base explains a hidden finding.
    F0 (bridge) never gets one."""
    from haenv.m2.core import components
    pos: dict[str, list[str]] = {}
    every: dict[str, list[str]] = {}
    for r in rows:
        if r["fw"] == "F0":
            continue
        every.setdefault(r["fw"], []).append(r["case"])
        if not r["frame"].endswith("-neg"):
            pos.setdefault(r["fw"], []).append(r["case"])

    def known(cid):
        return [by_id[cid]["raw"].get("disease")] + list(by_id[cid]["raw"].get("comorbidities") or [])

    offer: dict[str, list[str]] = {}
    for fw in sorted(pos):
        for cid in sorted(pos[fw]):
            c = [b for b in base_candidates(by_id[cid]["latent"].get("ddx_spec_id", "")) if b not in known(cid)]
            if c:
                offer.setdefault(fw, []).append(c[int(_u(cid, "pick") * len(c))])
    n_pos = sum(len(v) for v in pos.values())
    mix_all = sorted(b for v in offer.values() for b in v)
    out: dict[str, str] = {}
    if not mix_all:
        return out
    for fw in sorted(every):
        p_fw = pos.get(fw) or []
        rate = len(offer.get(fw) or ()) / len(p_fw) if p_fw else len(mix_all) / n_pos
        mix = sorted(offer.get(fw) or ()) or mix_all
        k = int(rate * len(every[fw]) + 0.5)
        for cid in sorted(every[fw], key=lambda c: _u(c, "attach")):
            if sum(1 for c in every[fw] if c in out) >= k:
                break
            spec = by_id[cid]["latent"].get("ddx_spec_id", "")
            comps = set(components(spec)) | {spec}
            pick = [b for b in mix if b not in known(cid) and not (BASE_HIDDEN[b] & comps)]
            if pick:
                out[cid] = pick[int(_u(cid, "which") * len(pick))]
    return out


#: Difficulty prior target (spec 3.4): easy : mid : hard = 25 : 45 : 30 of the whole pack. Only
#: F1-F3 knobs move (F0 is the byte-identical bridge); K-NT is a property of the spec, so the
#: lever is K-DD (`distractor_level`, the benign-distractor density knob).
TIER_TARGET = (("easy", 0.25), ("mid", 0.45), ("hard", 0.30))


def rebalance_kdd(rows: list[dict], by_id: dict) -> dict[str, str]:
    """case_id -> distractor_level for F1-F3 so the pack's prior tiers meet `TIER_TARGET` as
    closely as K-NT allows. Hard needs K-NT 6-8 with high, easy K-NT <= 5 with low; the cases that
    flip are chosen by a case-keyed order within each K-NT group (never by arm or score)."""
    n = len(rows)
    want = {t: round(n * w) for t, w in TIER_TARGET}
    want["mid"] = n - want["easy"] - want["hard"]
    f0 = [r for r in rows if r["fw"] == "F0"]
    have = {"easy": 0, "mid": 0, "hard": 0}
    for r in f0:
        have[prior_of(by_id[r["case"]])["tier"]] += 1
    fr = [r["case"] for r in rows if r["fw"] != "F0"]
    nt_hard = sorted((c for c in fr if len(by_id[c]["latent"].get("ddx_tests") or []) >= K_NT_HARD_MIN),
                     key=lambda c: _u(c, "kdd"))
    nt_easy = sorted((c for c in fr if c not in nt_hard), key=lambda c: _u(c, "kdd"))
    n_hard = min(len(nt_hard), max(0, want["hard"] - have["hard"]))
    n_easy = min(len(nt_easy), max(0, want["easy"] - have["easy"]))
    out = {}
    for i, c in enumerate(nt_hard):
        out[c] = "high" if i < n_hard else "low"          # low on K-NT 6-8 -> mid
    for i, c in enumerate(nt_easy):
        out[c] = "low" if i < n_easy else "high"          # high on K-NT <= 5 -> mid
    return out


#: Benign complaints over the course of an F1-F3 case: one case-keyed draw from 10..18 in every tier.
#: The lower end is the distractor pool (`core/noise._DISTRACTOR_SYMPTOMS`), which a high-tier case
#: injects whole; the planned benign events top the record up to the drawn count.
COMPLAINTS = (10, 18)


def complaint_count(case_id: str) -> int:
    lo, hi = COMPLAINTS
    return lo + int(_u(case_id, "complaints") * (hi - lo + 1))


def declare_complaint_count(latent: dict, n: int) -> dict:
    """Set the declared benign-symptom rate so the case's record carries `n` benign complaints
    (`events.expected_event_counts`). Mutates and returns `latent`."""
    from haenv.events import event_weeks, expected_event_counts
    ed = latent["event_density"] = dict(latent["event_density"])
    T = latent["index_time_T"]
    rate = round(n / event_weeks(T), 4)
    while (got := expected_event_counts({**ed, "symptom_rate": rate}, T)["symptom_rate"]) != n:
        rate = round(rate + (0.0001 if got < n else -0.0001), 4)
    ed["symptom_rate"] = rate
    return latent


def f3_plans(rows: list[dict], by_id: dict) -> dict[str, dict]:
    """case_id -> F3 plan (revision R-25): half of the F3 cases with a live lab stream follow a
    change beyond the RCV on a marker their hidden line declares, the rest a within-RCV repeat."""
    import haenv  # noqa: F401  (kernel path)
    from haenv.build import clinical_plan
    from haenv.m2.core import exceed_streams, f3_plan_for
    f3 = [r for r in rows if r["fw"] == "F3"]
    live = {}
    for r in f3:
        raw = by_id[r["case"]]["raw"]
        live[r["case"]] = list(clinical_plan(raw["disease"], raw["devices"], raw.get("comorbidities") or [], r["case"]))
    have_stream = [c for c in live if any(st in live[c] for st in ("HbA1c", "fasting_glucose", "LDL",
                                                                    "triglycerides", "ALT", "AST"))]
    target = len(have_stream) // 2 + len(have_stream) % 2
    def ok(c):
        return (by_id[c]["latent"].get("ddx_join_gold") != "independent"
                and f3_plan_for(c, by_id[c]["latent"].get("ddx_spec_id", ""), "", live[c], True,
                                by_id[c]["raw"].get("disease", "")) is not None)
    # Declared hidden markers first, then the named disease's own markers; the arm never goes to
    # the negative (benign) arm, whose gold says no clinician action.
    decl = [c for c in have_stream if ok(c) and exceed_streams(by_id[c]["latent"].get("ddx_spec_id", ""), live[c])]
    rest = [c for c in have_stream if ok(c) and c not in decl]
    cand = sorted(decl, key=lambda c: _u(c, "f3-arm")) + sorted(rest, key=lambda c: _u(c, "f3-arm"))
    ex = set(cand[:target])
    out = {}
    for c in have_stream:
        p = f3_plan_for(c, by_id[c]["latent"].get("ddx_spec_id", ""), "", live[c], c in ex,
                        by_id[c]["raw"].get("disease", ""))
        if p:
            out[c] = p
    return out


def prior_of(case: dict, med_text: str | None = None) -> dict:
    lat = case["latent"]
    n_tests = len(lat.get("ddx_tests") or [])
    k_dd = "high" if lat.get("distractor_level") == "high" else "low"
    k_nt = "6-8" if n_tests >= K_NT_HARD_MIN else "<=5"
    tier = "hard" if (k_dd == "high" and k_nt == "6-8") else (
        "easy" if (k_dd == "low" and k_nt == "<=5") else "mid")
    # The conditions the record names (`user_profile.known_conditions`): the primary disease and the
    # M2 explainer bases; other comorbidities are not named, so they explain nothing to a reader.
    known = [case["raw"].get("disease")] + [x for x in case["raw"].get("comorbidities") or [] if x in NEW_BASES]
    return {"k_dd": k_dd, "k_nt": k_nt, "k_ea": k_ea(lat.get("ddx_spec_id", ""), known, med_text),
            "tier": tier, "n_tests": n_tests}


def _load(name: str, path: pathlib.Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recompose():
    return _load("recompose_v3", RECOMPOSE)


def bridge_supply(rows: list[dict], by_id: dict) -> dict:
    """Bridge cell -> sufficient-tier bridge cases of that prior tier."""
    out = {k: 0 for k in PACK.BRIDGE}
    for r in rows:
        c = by_id[r["case"]]
        if r["fw"] == "F0" and not c["latent"].get("ddx_insufficient"):
            k = f"f0_{prior_of(c)['tier']}"
            if k in out:
                out[k] += 1
    return out


def allocation(n: int) -> tuple[list[dict], dict, dict, dict]:
    """(pool rows, source cases, pack cell sizes, bridge shortfall) for a pack of n items."""
    src = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in src["cases"]}
    rc = _recompose()
    f0 = rc.bridge_rows()
    q, short = PACK.quotas(n, bridge_supply(f0, by_id))
    nb = n - sum(q[k] for k in PACK.BRIDGE)
    res = rc.allocate(nb / NB_PER_SCALE)
    v3 = res["rows"]
    gen = rc.generated_cases(v3)
    assert not set(gen) & set(by_id)
    by_id.update(gen)
    rows = f0 + [{"case": r["case"], "spec_id": r["spec_id"], "fw": r["fw"], "frame": r["frame"],
                  "source": r["source"]} for r in v3]
    assert len({r["case"] for r in rows}) == len(rows)
    assert not any(r["case"] in rc.CLINICAL_HOLD for r in rows)
    return rows, by_id, q, short


def compose(rows: list[dict], by_id: dict) -> tuple[list[dict], dict, dict, dict]:
    """(cases, prior table, explainer bases, K-DD changes) of the pool."""
    src_by_id = dict(by_id)
    cases, prior = [], {}
    bases = assign_bases(rows, by_id)
    for cid, b in bases.items():
        raw = dict(by_id[cid]["raw"])
        raw["comorbidities"] = list(raw.get("comorbidities") or []) + [b]
        by_id[cid] = {**by_id[cid], "raw": raw}
    kdd = rebalance_kdd(rows, by_id)
    for cid, lvl in kdd.items():
        lat0 = copy.deepcopy(by_id[cid]["latent"])
        if lat0.get("distractor_level") != lvl:
            lat0["distractor_level"] = lvl
            if lvl == "high":
                from haenv.ddx import declare_high_distractor_density
                declare_high_distractor_density(lat0)
            by_id[cid] = {**by_id[cid], "latent": lat0}
    for r in rows:
        if r["fw"] != "F0":
            lat0 = copy.deepcopy(by_id[r["case"]]["latent"])
            declare_complaint_count(lat0, complaint_count(r["case"]))
            by_id[r["case"]] = {**by_id[r["case"]], "latent": lat0}
    f3 = f3_plans(rows, by_id)
    for r in sorted(rows, key=lambda r: (FRAMES.index(r["fw"]), r["case"])):
        c = by_id[r["case"]]
        lat = dict(c["latent"])
        arm = "negative" if str(r.get("frame") or "").endswith("-neg") else "positive"
        ev = acute_event_for(r["case"], lat.get("ddx_spec_id", ""), arm) if r["fw"] == "F1" else None
        # Visible medication text: the frame scene (F2 names the primary drug) and the symptom texts;
        # a drug-dependent explain-away row licenses only on a drug named here. F0 (bridge) has no
        # scene and no explain-away rendering.
        med = "" if r["fw"] == "F0" else visible_medication_text(
            scene_text(r["fw"], c["raw"], {**c["latent"], "m2": {"frame": r["fw"], **({"acute_event": ev} if ev else {})}},
                       c["case_id"]),
            c["raw"].get("symptoms") or [])
        p = prior_of(c, med)
        prior[r["case"]] = {"frame": r["fw"], **p}
        if r["fw"] != "F0":
            m2 = {"frame": r["fw"], "arm": arm,
                  "prior": {k: p[k] for k in ("k_dd", "k_nt", "k_ea", "tier")}}
            if r["fw"] == "F3" and r["case"] in f3:
                m2["f3"] = f3[r["case"]]
            if r["fw"] == "F1" and ev:
                m2["acute_event"] = ev
            # Explain-away rendering (profile only): the hidden line's symptoms are drawn from findings
            # a visible explainer covers; age tokens leave the templates.
            known = [c["raw"].get("disease")] + [x for x in c["raw"].get("comorbidities") or [] if x in NEW_BASES]
            sy, ea = explain_away_render(lat.get("ddx_spec_id", ""), known, c["raw"].get("symptoms") or [],
                                         T=lat.get("index_time_T"), med_text=med, case_id=c["case_id"])
            if sy != (c["raw"].get("symptoms") or []):
                c = {**c, "raw": {**c["raw"], "symptoms": sy}}
            if ea["lines"] or ea.get("clues"):
                m2["explain_away"] = ea
            m2["scene"] = scene_text(r["fw"], c["raw"], {**c["latent"], "m2": m2}, c["case_id"])
            m2["text_source"] = TEXT_SOURCE
            lat["m2"] = m2
        cases.append({"case_id": c["case_id"], "raw": c["raw"], "latent": lat})
    kdd_changed = {c: v for c, v in sorted(kdd.items()) if v != src_by_id[c]["latent"].get("distractor_level")}
    return cases, prior, bases, kdd_changed


JOB_KEYS = {"task_type": "joint_dx", "multiround": False, "include_baseline": True, "models": [],
            "sample_cases": 5, "gated": True, "plugins": {"judges": ["haenv.m2"]}}


def _dump(job: dict, comment: str) -> str:
    canary = CANARY.block().splitlines()
    buf = io.StringIO()
    buf.write("\n".join(canary) + "\n")
    buf.write(comment)
    yaml.safe_dump(job, buf, allow_unicode=True, sort_keys=False, width=120)
    return buf.getvalue()


def emitted(pool_text: str) -> tuple[set[str], list[str]]:
    """(case ids the deterministic build emits from the pool job, emitted F1-F3 cases with a gold line
    that shows no separating clue on the record). A case of the second kind has no evidence that tells
    its line from the nearest look-alike (the EA-2 check of `tools/m2_item_audit.py`), so it is not
    selected."""
    with tempfile.TemporaryDirectory(prefix="m2-pool-") as td:
        job = pathlib.Path(td) / "m2-pool.job.yaml"
        job.write_text(pool_text, encoding="utf-8")
        env = {**os.environ, "HAENV_OUTPUT_ROOT": td}
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "plugins"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        r = subprocess.run([sys.executable, "-m", "haenv", "build", str(job), "--gen", "deterministic", "--fresh",
                            "--set", "env_file=/dev/null"], cwd=ROOT, env=env, capture_output=True, text=True)
        got = sorted(pathlib.Path(td, "results").glob("*/m2-pool/*/cases.jsonl"))
        # exit 4 = a batch-level gate on the pool; only the per-case emission is read here
        if r.returncode not in (0, 4) or not got:
            sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
            raise SystemExit(f"[m2_gen_job] pool build failed (exit {r.returncode})")
        ok = {json.loads(ln)["case_id"] for ln in got[-1].read_text(encoding="utf-8").splitlines() if ln.strip()}
        import m2_item_audit as IA
        ea2 = IA.check_ea2(IA.collect(got[-1].parent, job))["violations"]
        return ok, sorted({v[0] for v in ea2})


def no_lay_sentences(cases: list[dict]) -> list[str]:
    """F1-F3 cases with a symptom phrase the lay-sentence registry does not cover: the record would
    print that phrase whole (often a diagnosis label), so such a case is not selected."""
    from haenv import events as E
    lay = E.symptom_lay()
    return sorted(c["case_id"] for c in cases if (c.get("latent") or {}).get("m2")
                  and any(sy.get("text") not in lay for sy in c["raw"].get("symptoms") or ()))


def build(n: int = PS.DEFAULT_N, seed: str = PS.PUBLIC_SEED) -> str:
    rows, by_id, q, short = allocation(n)
    cases, prior, bases, kdd = compose(rows, by_id)
    pool = {"job_id": "m2-pool", **JOB_KEYS, "report": "eval-m2-pool.md",
            "_provenance": {"m2_prior": prior}, "cases": cases}
    ok, no_clue = emitted(_dump(pool, "# pool job (intermediate)\n"))
    refused = sorted(c["case_id"] for c in cases if c["case_id"] not in ok)
    unlay = no_lay_sentences(cases)
    S = _load("pack1_select", SELECT)
    prow = [r for r in S.rows_of(pool) if r["case"] in ok and r["case"] not in unlay and r["case"] not in no_clue]
    pick = S.select(prow, q, seed)
    bal = S.balance(pick, prow)
    if not bal["pass"]:
        raise SystemExit(f"[m2_gen_job] balance gate failed: {json.dumps(bal, ensure_ascii=False)}")
    ids = {r["case"] for r in pick}
    plan: dict = {}
    for r in pick:
        if r["frame"] != "F0":
            plan.setdefault(r["frame"], {}).setdefault(r["tier"], 0)
            plan[r["frame"]][r["tier"]] += 1
    rc = _recompose()
    job = {
        "job_id": "m2-pack1", **JOB_KEYS, "report": "eval-m2-pack1.md",
        "pack": PS.header(n, seed),
        "_provenance": {
            "generator": "tools/m2_gen_job.py", "source_job": "inputs/ddx-workup.job.yaml",
            "bridge_set": "docs/scripts/m2/alloc_cases.csv",
            "bridge_set_sha16": hashlib.sha256(ALLOC.read_bytes()).hexdigest()[:16],
            "design": dict(q), "bridge_shortfall": dict(short),
            "plan_cells": {f: dict(sorted(t.items())) for f, t in sorted(plan.items())},
            "pool": {"n": len(cases), "refused_by_build": refused, "no_lay_sentences": unlay, "no_separating_clue": no_clue,
                     "generated_cases": sorted(r["case"] for r in rows if r.get("source") == "generated")},
            "withheld": list(rc.CLINICAL_HOLD),
            "explainer_bases": {c: b for c, b in sorted(bases.items()) if c in ids},
            "kdd_rebalanced": {c: v for c, v in kdd.items() if c in ids},
            "m2_prior": {c: p for c, p in prior.items() if c in ids}},
        "cases": [c for c in cases if c["case_id"] in ids],
    }
    nb = sum(1 for c in job["cases"] if prior[c["case_id"]]["frame"] == "F0")
    return _dump(job, "# Generated file -- do not edit by hand. Rebuild: python tools/m2_gen_job.py --n N [--seed S]\n#\n"
                 f"# Pack 1 (differential diagnosis): {n} cases in four frames -- F0 weight-regain bridge ({nb}, copied\n"
                 "# verbatim from ddx-workup.job.yaml, byte-identical to v1.0.1), F1 acute triage, F2 chronic medication\n"
                 "# adjustment, F3 follow-up interpretation. F1-F3 add one latent key `m2` (knob). The plugin group\n"
                 "# `haenv.m2` (haenv/m2/plugin.py) registers the frame and revision gold blocks.\n"
                 "#\n# SYNTHETIC data, evaluation only, not medical advice.\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=PS.DEFAULT_N)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    import logging
    logging.disable(logging.WARNING)
    text = build(a.n, PS.cli_seed(a.seed))
    out = pathlib.Path(a.out)
    if a.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == text
        print(f"[m2_gen_job] {'up to date' if same else 'STALE'}: {out}")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8")
    print(f"[m2_gen_job] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
