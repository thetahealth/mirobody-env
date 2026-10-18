"""Prepare paired synthetic jobs from actual emitted questions; zero model calls.

Run with the project interpreter and PYTHONPATH=.:plugins. Required inputs are a
source job and its already built/selected batch. Output is scratch material,
never a fifth pack. --build uses the ordinary deterministic build command.
The source may be a pack's emission pool (omit --source-items; --pack-job flags
families whose question is published). After --build verifies every family and
both arms pass the leakage probe, both arms inherit the audited source's marker.
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
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))

import haenv.world.generation.twins as twins


#: no subject (the twin's diagnosis, or the unnamed explainer) may hold more than this share of an arm
CAP_SHARE = 0.25


def family_subject(t, job_case):
    """The disease a family is about: the kept line of an M2 line removal, the unnamed explainer of an M2
    unnaming, otherwise the question's diagnosis spec."""
    if t["kind"] == twins.M2_LINE_KIND:
        return str(t["after"])
    if t["kind"] == "unname-explainer":
        return str(t["operation_details"]["unname"][0])
    return str((job_case.get("latent") or {}).get("ddx_spec_id"))


def _h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def arm_order(trials):
    """One arm's order: strata (kind x subject) in hash order take turns, one trial per turn; inside a
    stratum, questions in hash order, each question once. Independent of case-ID order and of any model."""
    strata = collections.defaultdict(dict)
    for t in sorted(trials, key=lambda t: _h(t["case_id"], t["path"], json.dumps(t["after"], sort_keys=True, ensure_ascii=False))):
        strata[(t["kind"], t["subject"])].setdefault(t["case_id"], t)
    queues = [list(q.values()) for _, q in sorted(strata.items(), key=lambda kv: _h(*kv[0]))]
    out = []
    while any(queues):
        out += [q.pop(0) for q in queues if q]
    return out


def _walk(order, target, cap, used):
    per, picked, seen = collections.Counter(), [], set(used)
    for t in order:
        if len(picked) == target:
            break
        if t["case_id"] in seen or per[t["subject"]] >= cap:
            continue
        picked.append(t)
        seen.add(t["case_id"])
        per[t["subject"]] += 1
    return picked


def _take(order, target, used):
    """Walk the order, skipping questions already used and subjects at the cap; the cap is a quarter of the
    arm actually taken, so a short supply shrinks the arm until no subject holds more than a quarter."""
    while target > 0:
        picked = _walk(order, target, max(1, int(CAP_SHARE * target)), used)
        if len(picked) == target:
            used.update(t["case_id"] for t in picked)
            return picked
        target = len(picked)
    return []


def select_families(trials, count):
    """Gold-changing arm first (half of N, rounded up), then gold-preserving families fill N; one family
    per question; each arm walks its own `arm_order` under the subject cap."""
    if count < 1:
        raise ValueError("family count must be positive")
    if len({t["case_id"] for t in trials}) < count:
        raise ValueError(f"shortage: requested {count} actual questions, only {len({t['case_id'] for t in trials})} eligible")
    used = set()
    change = _take(arm_order([t for t in trials if t["gold_changed"]]), (count + 1) // 2, used)
    keep = _take(arm_order([t for t in trials if not t["gold_changed"]]), count - len(change), used)
    if len(change) + len(keep) < count:
        raise ValueError(f"shortage: {len(change)} gold-changing + {len(keep)} gold-preserving families under the "
                         f"{CAP_SHARE:.0%} subject cap, {count} requested")
    return change + keep


#: packs read by same-question pairs: each question has a gold-changing twin and a matched gold-preserving twin
PAIRED_PACKS = ("p2", "p3", "p4")
#: byte-identical resubmissions of selected questions, per pack
SHAM_N = 20


def _field(path):
    return re.sub(r"/\d+(?=/|$)", "/*", path)


def _record(path):
    m = re.match(r".*/records/\d+", path)
    return m[0] if m else None


def _step(t):
    try:
        return round(abs(float(t["after"]) - float(t["before"])), 6)
    except (TypeError, ValueError):
        return None


def _match(pack, gc, gp):
    """(tier, quality) of a gold-preserving partner for a gold-changing trial on the same question; lower tier is
    closer; None = not a partner. P2: same reading field and step (tier 0 = the same reading moved the other way).
    P3: same path (tier 0 = same magnitude). P4: a removed non-acting perturbation (tier 0 = same record)."""
    q = {"same_path": gc["path"] == gp["path"], "same_field": _field(gc["path"]) == _field(gp["path"]),
         "same_step": _step(gc) is not None and _step(gc) == _step(gp),
         "same_record": _record(gc["path"]) is not None and _record(gc["path"]) == _record(gp["path"])}
    if pack == "p2":
        if gp["kind"] != gc["kind"] or not (q["same_field"] and q["same_step"]):
            return None
        return (0 if q["same_path"] else 1), q
    if pack == "p3":
        if gp["kind"] != gc["kind"] or not q["same_path"]:
            return None
        return (0 if q["same_step"] else 1), q
    if gp["kind"] != "p4-remove-candidate":
        return None
    q["same_step"] = len(gc["before"]) - len(gc["after"]) == len(gp["before"]) - len(gp["after"])
    return (0 if q["same_record"] else 1), q


def _hkey(t):
    return _h(t["case_id"], t["path"], json.dumps(t["after"], sort_keys=True, ensure_ascii=False))


def pair_families(trials, pack):
    """One family per question that has a gold-changing trial and a matched gold-preserving partner: the closest
    tier wins, ties in hash order."""
    by = collections.defaultdict(list)
    for t in trials:
        by[t["case_id"]].append(t)
    out = []
    for cid in sorted(by, key=lambda c: _h(c)):
        best = None
        for gc in sorted((t for t in by[cid] if t["gold_changed"]), key=_hkey):
            for gp in sorted((t for t in by[cid] if not t["gold_changed"]), key=_hkey):
                m = _match(pack, gc, gp)
                if m and (best is None or m[0] < best[0]):
                    best = (m[0], gc, gp, m[1])
        if best:
            tier, gc, gp, q = best
            out.append({"case_id": cid, "pack": pack, "subject": gc["subject"], "kind": gc["kind"], "path": gc["path"],
                        "after": gc["after"], "gc": gc, "gp": gp, "match": {"tier": tier, **q}})
    return out


def select_pairs(families, count):
    """The paired families in `arm_order` under the subject cap; fewer when the supply is short."""
    return _take(arm_order(families), count, set())


def choose_nulls(trials):
    """One valid null per question: condition swap first, then the deterministic medication swap."""
    priority = {kind: i for i, kind in enumerate(twins.NULL_KINDS)}
    out = {}
    for t in sorted((t for t in trials if t["kind"] in priority), key=lambda t: (priority[t["kind"]], _hkey(t))):
        if t["gold_changed"]:
            raise ValueError("semantic null candidate changed gold")
        out.setdefault(t["case_id"], t)
    return out


def null_items(families, nulls, n):
    """The semantic-null arm: selected questions that have a null edit, first `n` in the sham order."""
    have = {f["case_id"] for f in families if f["case_id"] in nulls}
    if len(have) < n:
        raise ValueError(f"shortage: null arm has {len(have)} distinct eligible questions; {n} required")
    return {cid: nulls[cid] for cid in sham_ids(have, n)}


def sham_ids(ids, n):
    """The byte-identical resubmission arm: the first `n` selected questions in sha256 order."""
    return sorted(ids, key=lambda c: _h("sham", c))[:n]


def cluster_counts(rows):
    """Clusters of one arm at three levels: family, base case (case ID without its vNN suffix), condition spec."""
    return {"family": len(rows), "base_case": len({re.sub(r"v\d+$", "", r["case_id"]) for r in rows}),
            "spec": len({r["spec"] for r in rows})}


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def changed_chars(a, b):
    """Characters changed between two JSON values: edit distance on scalars, full length of added/removed items."""
    import difflib
    dump = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True)
    if isinstance(a, dict) and isinstance(b, dict):
        return sum(changed_chars(a[k], b[k]) if k in a and k in b else len(dump(a.get(k, b.get(k))))
                   for k in set(a) | set(b))
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == len(b):
            return sum(changed_chars(x, y) for x, y in zip(a, b))
        n = 0
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [dump(x) for x in a], [dump(y) for y in b],
                                                          autojunk=False).get_opcodes():
            if op == "equal":
                continue
            pairs = min(i2 - i1, j2 - j1)
            n += sum(changed_chars(a[i1 + k], b[j1 + k]) for k in range(pairs))
            n += sum(len(dump(x)) for x in a[i1 + pairs:i2]) + sum(len(dump(y)) for y in b[j1 + pairs:j2])
        return n
    return 0 if a == b else _lev(dump(a), dump(b))


def mark_in_pack(families, pack_job):
    """Flag families whose question is one of the published pack's items; return how many are."""
    import yaml
    pj = yaml.safe_load(Path(pack_job).read_text(encoding="utf-8"))
    ids = set((pj.get("_provenance") or {}).get("pack_ids") or [c["case_id"] for c in pj["cases"]])
    for f in families:
        f["in_pack"] = f["case_id"] in ids
    return sum(f["in_pack"] for f in families)


def arm_checks(pack, batch, job):
    """Per-item checks on one built arm: the kernel leakage probe, plus ①'s item audit (EA-1/EA-2/MED-1)."""
    from haenv.app.artifacts.store import load_cases
    from haenv_kernel.build import build_instance, leakage_probe
    leaks = {}
    for cid, raw in load_cases(Path(batch) / "cases.jsonl").items():
        T = int(raw.prediction_context["prediction_time_T"])
        ok, viol = leakage_probe(build_instance(raw, T)[0], T)
        if not ok:
            leaks[cid] = viol
    out = {"leakage_probe": {"pass": not leaks, "violations": leaks}, "pass": not leaks}
    if pack == "m2":
        sys.path.insert(0, str(ROOT / "tools"))
        import haenv.m2.item_audit as A
        cases = A.collect(Path(batch), Path(job))
        for name, fn in (("EA-1", A.check_ea1), ("EA-2", A.check_ea2), ("MED-1", A.check_med1)):
            r = fn(cases)
            out[name] = {"pass": r.get("pass"), "violations": [str(v) for v in (r.get("violations") or [])]}
            out["pass"] = out["pass"] and r.get("pass") is True
    return out


def carry_family_audit(pack, source, built, summary):
    """Both arms inherit the audit marker of an audited source batch, only when every family verified and
    both arms pass the leakage probe. The marker's first line is the source cases' sha256."""
    from haenv.world.validation.shared_audit import marker_name
    mk = marker_name(pack)
    src = twins.batch_files(Path(source))
    checks = summary.get("arm_checks") or {}
    if (not (src / mk).is_file() or summary.get("problems")
            or any(not (checks.get(arm) or {}).get("pass") for arm in built)):
        return False
    sha = hashlib.sha256((src / "cases.jsonl").read_bytes()).hexdigest()
    for arm, b in built.items():
        (Path(b) / mk).write_text(f"{sha}\n{arm} arm of truth families derived from the audited batch above; "
                                  "checked field by field (family_record.json)\n", encoding="utf-8")
    return True


def _write(path, obj):
    import yaml
    path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False, width=160), encoding="utf-8")


def replay_fingerprints():
    """Only the normal generation/kernel/knob stamps; never require the same git HEAD."""
    from haenv.provenance.anchor import world_fp_cached
    from haenv.app.artifacts.batch import kernel_fingerprint
    from haenv.world.generation.world_knobs import PHYSIO_ENABLED
    return {"world_sha": world_fp_cached(), "kernel_sha256": kernel_fingerprint()[1],
            "world_knobs": f"physio={int(bool(PHYSIO_ENABLED[0]))}"}


def preflight_source(job_path, source, pack, expected_items=None):
    """Read-only mismatch check before selection, output-directory creation or builds."""
    from haenv.app.job import load_job
    from haenv.world.generation.opaque_ids import solver_case_id
    from haenv.world.validation.pack_size import cli_seed
    job_path, source = Path(job_path), Path(source)
    try:
        meta = json.loads((source / "batch.json").read_text())
        cr, pr = twins.rows(source, "cases.jsonl"), twins.rows(source, "payloads.jsonl")
    except (OSError, ValueError) as exc:
        raise ValueError(f"preflight: require the stored batch directory with batch.json, cases.jsonl and payloads.jsonl: {exc}") from exc
    digest = hashlib.sha256(job_path.read_bytes()).hexdigest()[:16]
    if meta.get("job_sha256") != digest:
        raise ValueError("preflight: source job bytes differ from batch.job_sha256")
    if not cr or set(cr) != set(pr):
        raise ValueError("preflight: empty source or case/payload IDs differ")
    if expected_items is not None and len(cr) != expected_items:
        raise ValueError(f"preflight: expected {expected_items} stored questions, got {len(cr)}")
    if meta.get("generator") != "deterministic":
        raise ValueError("preflight: exact deterministic replay requires generator=deterministic")
    seed = cli_seed(None)
    for cid, line in cr.items():
        row = json.loads(line)
        recorded = row["case"].get("latent_premise", {}).get("meta", {}).get("solver_case_id")
        if recorded != solver_case_id(cid, seed):
            raise ValueError("preflight: HAENV_PACK_SEED does not reproduce the stored solver IDs; use the original rendering environment")
        world = json.dumps({"case_id": cid, "case": row["case"]}, ensure_ascii=False, separators=(",", ":"))
        if hashlib.sha256(world.encode()).hexdigest()[:16] != (meta.get("cases_sha256") or {}).get(cid):
            raise ValueError(f"preflight: stored world digest mismatch for {cid}")
    load_job(str(job_path))  # registration only; no output writes or generation
    current = replay_fingerprints()
    for key, value in current.items():
        if not value or value == "unknown" or meta.get(key) != value:
            raise ValueError(f"preflight: source/current {key} differ ({meta.get(key)} vs {value}); rebuild final source after integration")
    import yaml
    job = yaml.safe_load(job_path.read_text())
    if set(cr) - {c["case_id"] for c in job["cases"]}:
        raise ValueError("preflight: source IDs absent from job")
    # Stored payloads must be the normal projections of these exact worlds.
    from haenv.app.artifacts.store import load_cases
    from haenv.app.artifacts.payloads import _payload_line
    worlds = load_cases(source / "cases.jsonl")
    if set(worlds) != set(cr):
        raise ValueError("preflight: not all stored worlds loaded")
    for cid, raw in worlds.items():
        if json.loads(_payload_line((cid, raw))[1]) != json.loads(pr[cid]):
            raise ValueError(f"preflight: stored payload differs from current projection for {cid}")
    return {"world_sha": current["world_sha"], "kernel_sha256": current["kernel_sha256"],
            "source_items": len(cr), "render_seed_matches_stored_ids": True}


def prepare(job_path, source, pack, out, count=20, expected_items=None, pack_job=None):
    import yaml
    from haenv.app.job import load_job
    from haenv.app.artifacts.store import load_cases
    job_path, source, out = Path(job_path), Path(source), Path(out)
    preflight = preflight_source(job_path, source, pack, expected_items)
    job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    load_job(str(job_path))  # production plugin registration and job validation
    cr, pr = twins.rows(source, "cases.jsonl"), twins.rows(source, "payloads.jsonl")
    if set(cr) != set(pr):
        raise ValueError("source case/payload IDs differ")
    cases = load_cases(source / "cases.jsonl")
    if set(cases) != set(cr):
        raise ValueError("source rows were not all loadable")
    jc = {c["case_id"]: c for c in job["cases"]}
    if set(cr) - set(jc):
        raise ValueError("source batch contains cases absent from source job")
    rejected, trials = [], []
    for cid in sorted(cases):
        trials.extend(twins.family_candidates(cases[cid], pack, row=json.loads(cr[cid]), job_case=jc[cid],
                                               pay=json.loads(pr[cid]), rejected=rejected))
    for t in trials:
        t["subject"] = family_subject(t, jc[t["case_id"]])
    valid_ids = {t["case_id"] for t in trials}
    summary = {"pack": pack, "preflight": preflight, "source_questions": len(cases), "requested": count,
               "eligible_questions": len(valid_ids), "valid_trials": len(trials),
               "gold_changing_questions": len({t["case_id"] for t in trials if t["gold_changed"]}),
               "gold_preserving_questions": len({t["case_id"] for t in trials if not t["gold_changed"]}),
               "rejections": dict(collections.Counter(r["reason"] for r in rejected)),
               "rejected_trials": rejected,
               "ineligible": {cid: [r for r in rejected if r["case_id"] == cid]
                              for cid in sorted(set(cases) - valid_ids)},
               "candidate_order": {arm: [{k: t[k] for k in ("case_id", "kind", "subject", "path")}
                                         for t in arm_order([t for t in trials if t["gold_changed"] == (arm == "gold_changing")])]
                                   for arm in ("gold_changing", "gold_preserving")},
               "selection_rule": f"strata (kind x subject) take turns in hash order; one family per question; "
                                 f"gold-changing first (half of N, rounded up), gold-preserving fill N; no subject above "
                                 f"{CAP_SHARE:.0%} of its arm",
               "source_job_sha256": hashlib.sha256(job_path.read_bytes()).hexdigest(),
               "source_cases_sha256": hashlib.sha256((source / "cases.jsonl").read_bytes()).hexdigest(),
               "source_payloads_sha256": hashlib.sha256((source / "payloads.jsonl").read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    # A dependency declaration that misses an actual change is an implementation
    # failure, not evidence that this question lacks legal interventions. Do not
    # let another valid trial on the question silently hide that failure.
    dependency_failures = [r for r in rejected if r["reason"].startswith("undeclared")]
    if dependency_failures:
        cause = f"dependency declaration failed for {len(dependency_failures)} trials; see rejected_trials"
        summary.update(status="failed", cause=cause, problems=[cause], dependency_failures=dependency_failures)
        (out / "family_record.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        raise ValueError(cause)
    for t in trials:
        t["spec"] = str((jc[t["case_id"]].get("latent") or {}).get("ddx_spec_id"))
    if pack in PAIRED_PACKS:
        pairs = pair_families(trials, pack)
        summary.update(design="paired", pair_supply=len(pairs),
                       pair_order=[{k: f[k] for k in ("case_id", "kind", "subject", "path")} for f in arm_order(pairs)],
                       selection_rule=(f"a question enters only with a verified gold-changing twin and a matched "
                                       f"gold-preserving twin (closest match tier, ties in hash order); strata (kind x "
                                       f"subject) take turns in hash order; no subject above {CAP_SHARE:.0%} of the "
                                       f"selection; sham = first {SHAM_N} selected questions in hash order"))
        picked = select_pairs(pairs, count)
        if not picked:
            summary.update(status="shortage", cause="no paired family")
            (out / "family_record.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            raise ValueError("shortage: no paired family")
        for f in picked:
            f["spec"] = f["gc"]["spec"]
        items = {"gc": {f["case_id"]: f["gc"] for f in picked}, "gp": {f["case_id"]: f["gp"] for f in picked}}
        nulls = choose_nulls(trials)
        picked_ids = {f["case_id"] for f in picked}
        summary.update(null_supply=len(picked_ids & nulls.keys()), null_required=SHAM_N,
                       null_kind_supply={kind: len({t["case_id"] for t in trials if t["kind"] == kind
                                                   and t["case_id"] in picked_ids}) for kind in twins.NULL_KINDS},
                       null_assignment_rule="condition swap first per question; otherwise earliest-day/lowest-index "
                                            "adjacent distinct-medication swap; first 20 distinct eligible questions "
                                            "in the existing sham hash order; fewer than 20 is a failure")
        try:
            items["null"] = null_items(picked, nulls, SHAM_N)
        except ValueError as exc:
            summary.update(status="shortage", cause=str(exc), selected=len(picked), families=picked)
            (out / "family_record.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            raise
    else:
        try:
            picked = select_families(trials, count)
        except ValueError as exc:
            summary.update(status="shortage", cause=str(exc))
            (out / "family_record.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            raise
        summary["design"] = "unpaired"
        items = {"twin": {t["case_id"]: t for t in picked}}
    ids = [t["case_id"] for t in picked]
    sham = sham_ids(ids, SHAM_N)
    for f in picked:
        f["sham"] = f["case_id"] in sham
        f["null"] = f["case_id"] in items.get("null", {})
        f["null_kind"] = items.get("null", {}).get(f["case_id"], {}).get("kind")
    arm_ids = {"orig": ids, **{arm: list(v) for arm, v in items.items()}, "sham": sham}
    if pack_job:
        summary["selected_in_pack"] = mark_in_pack(picked, pack_job)
    jobs = {}
    for arm in ("orig", *items, "sham"):
        obj = copy.deepcopy(job)
        obj["job_id"] = f"{job['job_id']}-families-{arm}"
        obj["report"] = f"eval-{obj['job_id']}.md"
        obj["models"] = []
        obj["include_baseline"] = False
        obj["cases"] = [copy.deepcopy(jc[cid]) for cid in arm_ids[arm]]
        obj["_provenance"] = {"family_source_job_sha256": summary["source_job_sha256"],
                               "family_source_cases_sha256": summary["source_cases_sha256"],
                               "family_arm": arm}
        if arm in items:
            obj["_provenance"]["twin"] = {"op": "truth-family", "items": items[arm]}
        # Preserve pack metadata (including hashed seed) verbatim. The arm is a
        # selected intervention subset, not a freshly designed N-item pack.
        jobs[arm] = out / f"{obj['job_id']}.job.yaml"
        _write(jobs[arm], obj)
    rows = {"orig": picked, "sham": [f for f in picked if f["sham"]], "null": [f for f in picked if f["null"]]}
    if summary["design"] == "paired":
        rows.update(gc=picked, gp=picked)
    else:
        rows.update(gc=[t for t in picked if t["gold_changed"]], gp=[t for t in picked if not t["gold_changed"]])
    summary.update(status="prepared", families=picked, sham_ids=sham, selected=len(picked),
                   null_items=items.get("null", {}),
                   null_kind_counts=dict(collections.Counter(t["kind"] for t in items.get("null", {}).values())),
                   arms={arm: {"job": p.name, "cases": len(arm_ids[arm]),
                               "role": {"orig": "original questions, shared by every twin arm",
                                        "gc": "gold-changing twin of every selected question",
                                        "gp": "matched gold-preserving twin of every selected question",
                                        "twin": "one twin per question (gold-changing or gold-preserving per family)",
                                        "null": "semantic null edit (condition order or same-day adjacent distinct-medication order)",
                                        "sham": "byte-identical resubmission of the sham questions"}[arm]}
                         for arm, p in jobs.items()},
                   clusters={arm: cluster_counts(r) for arm, r in rows.items() if r},
                   selected_gold_changing=sum(bool(t.get("gc") or t.get("gold_changed")) for t in picked),
                   selected_gold_preserving=sum(bool(t.get("gp") or not t.get("gold_changed")) for t in picked),
                   paired_jobs={k: p.name for k, p in jobs.items()},
                   seed_policy="source pack metadata and case_id identical in every arm; no RNG redraw during mutation")
    (out / "family_record.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary, jobs


def verify_builds(trials, source, original, twin):
    """Saved worlds and both payloads must equal the prepared intervention exactly."""
    from haenv.app.artifacts.store import load_cases
    source = twins.batch_files(Path(source))
    original, twin = Path(original), Path(twin)
    selected = {t["case_id"]: t for t in trials}
    worlds = load_cases(original / "cases.jsonl")
    problems = []
    for name in ("cases.jsonl", "payloads.jsonl"):
        src, orig, tw = (twins.rows(b, name) for b in (source, original, twin))
        if not set(selected) <= set(orig) or set(tw) != set(selected):
            problems.append(f"{name}: built IDs differ from selected families")
            continue
        for cid, d in selected.items():
            if json.loads(src[cid]) != json.loads(orig[cid]):
                problems.append(f"{cid}: original {name} differs from source")
            changes = twins.changed(orig[cid], tw[cid])
            if not changes:
                problems.append(f"{cid}: empty {name} twin")
            for path in changes:
                parts = path.lstrip("/").split("/")
                if parts[0] not in ("case", "sp", "vp") or not twins._covered("/".join(parts[1:]), d["declared_fields"]):
                    problems.append(f"{cid}: undeclared {name} {path}")
    twins_world = load_cases(twin / "cases.jsonl")
    for cid, d in selected.items():
        if cid not in worlds or cid not in twins_world:
            continue
        problems += twins.check_persisted_twin(worlds[cid], twins_world[cid], d)
    return problems


def _ordinary_build(job, out, arm):
    out = Path(out)
    env = {**os.environ, "HAENV_OUTPUT_ROOT": str((out / "builds").resolve()),
           "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "plugins")))}
    with (out / f"{arm}-build.log").open("w") as log:
        rc = subprocess.run([sys.executable, "-m", "haenv", "build", str(job), "--gen", "deterministic",
                             "--fresh", "--gen-workers", "1"], cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise ValueError(f"ordinary {arm} build failed with rc={rc}; see {arm}-build.log")
    import yaml
    jid = yaml.safe_load(job.read_text())["job_id"]
    matches = sorted((out / "builds" / "results").glob(f"*/{jid}/*/cases.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one isolated build for {arm}, got {len(matches)}")
    return matches[0].parent


def rebuild_public_baseline(job_path, source, out):
    """Explicit public rendering baseline; never pretend it replays the old seed.

    The original allocation and all case IDs stay intact. Only opaque identities
    and the patient wording keyed by them can change under this public rendering
    seed. Its source is recorded separately from the selected source batch.
    """
    import yaml
    from haenv.world.validation.pack_size import PUBLIC_SEED
    if os.environ.get("HAENV_PACK_SEED", PUBLIC_SEED) != PUBLIC_SEED:
        raise ValueError("--public-baseline requires the public default seed; unset HAENV_PACK_SEED")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    obj = yaml.safe_load(Path(job_path).read_text())
    selected = set(twins.rows(Path(source), "cases.jsonl"))
    present = {c["case_id"] for c in obj["cases"]}
    if not selected or selected - present:
        raise ValueError("public baseline source IDs are empty or absent from source job")
    obj["cases"] = [c for c in obj["cases"] if c["case_id"] in selected]
    obj["job_id"] += "-public-family-baseline"
    obj["models"] = []
    obj["include_baseline"] = False
    obj["_provenance"] = {"source_job_sha256": hashlib.sha256(Path(job_path).read_bytes()).hexdigest(),
                           "family_render_seed": "public default", "preserved_allocation_metadata": True}
    path = out / f"{obj['job_id']}.job.yaml"
    _write(path, obj)
    return path, _ordinary_build(path, out, "public-baseline")


def visible_change(summary, built, twin_arms):
    """Per family and twin arm: characters changed in what the solver sees (`sp`, not the judging view)."""
    orig = twins.rows(built["orig"], "payloads.jsonl")
    for arm in twin_arms:
        tw = twins.rows(built[arm], "payloads.jsonl")
        for f in summary["families"]:
            if f["case_id"] not in tw:
                continue
            a, b = json.loads(orig[f["case_id"]])["sp"], json.loads(tw[f["case_id"]])["sp"]
            f.setdefault("visible_chars", {})[arm] = changed_chars(a, b)
            f.setdefault("visible_len_delta", {})[arm] = (len(json.dumps(b, ensure_ascii=False))
                                                          - len(json.dumps(a, ensure_ascii=False)))


def build_arms(summary, jobs, source, out):
    out = Path(out)
    built = {}
    try:
        if summary["design"] == "paired":
            nulls = summary.get("null_items") or {}
            families = {f["case_id"] for f in summary["families"]}
            if "null" not in jobs or len(nulls) != SHAM_N or not set(nulls) <= families:
                raise ValueError(f"null arm requires exactly {SHAM_N} distinct selected questions before building")
        built["orig"] = _ordinary_build(jobs["orig"], out, "orig")
        src, orig = (twins.rows(Path(b), "cases.jsonl") for b in (source, built["orig"]))
        differences = {cid: twins.changed(src[cid], orig[cid]) for cid in orig if cid in src and src[cid] != orig[cid]}
        if differences:
            summary["source_replay_differences"] = differences
            raise ValueError("ordinary original rebuild differs from source; matching source rendering seed/code is required; "
                             "--public-baseline is an explicit alternative reference, not a source replay")
        problems = []
        twin_arms = [a for a in jobs if a not in ("orig", "sham")]
        for arm in twin_arms:
            built[arm] = _ordinary_build(jobs[arm], out, arm)
            trials = (list(summary["null_items"].values()) if arm == "null" else
                      [f[arm] for f in summary["families"]] if summary["design"] == "paired" else summary["families"])
            problems += [f"{arm}: {p}" for p in verify_builds(trials, source, built["orig"], built[arm])]
        built["sham"] = _ordinary_build(jobs["sham"], out, "sham")
        for name in ("cases.jsonl", "payloads.jsonl"):
            o, sh = twins.rows(built["orig"], name), twins.rows(built["sham"], name)
            if set(sh) != set(summary["sham_ids"]) or any(sh[c] != o.get(c) for c in sh):
                problems.append(f"sham: {name} rows are not byte-identical to the original arm")
        summary.update(status="verified" if not problems else "failed", problems=problems)
        if problems:
            raise ValueError("; ".join(problems))
        visible_change(summary, built, twin_arms)
        summary["arm_checks"] = {arm: arm_checks(summary["pack"], built[arm], jobs[arm]) for arm in built}
        failed = [arm for arm, checks in summary["arm_checks"].items() if checks.get("pass") is not True]
        if failed:
            cause = f"arm checks failed: {', '.join(failed)}"
            summary["problems"].append(cause)
            raise ValueError(cause)
        summary["audit_carried"] = carry_family_audit(summary["pack"], source, built, summary)
    except ValueError as exc:
        summary.update(status="failed", cause=str(exc))
        raise
    finally:
        summary["builds"] = {k: str(v.relative_to(out)) for k, v in built.items()}
        (out / "family_record.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", choices=("m2", "p2", "p3", "p4"), required=True)
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--from-batch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--families", type=int, default=20)
    ap.add_argument("--source-items", type=int, help="require this many stored source questions (100 for final N100 packs; "
                                                       "omit for an emission pool)")
    ap.add_argument("--pack-job", type=Path, help="the published pack job: flags families whose question it publishes")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--public-baseline", action="store_true", help="explicitly rebuild the selected source questions under the public rendering seed")
    args = ap.parse_args(argv)
    try:
        reference_job, reference_batch = args.job, args.from_batch
        if args.public_baseline:
            reference_job, reference_batch = rebuild_public_baseline(args.job, args.from_batch, args.out / "public-reference")
        summary, jobs = prepare(reference_job, reference_batch, args.pack, args.out, args.families, args.source_items,
                                args.pack_job)
        if args.public_baseline:
            summary["reference_policy"] = "explicit rebuilt public rendering baseline, not a replay of the selected source batch"
            summary["original_source_cases_sha256"] = hashlib.sha256((twins.batch_files(args.from_batch) / "cases.jsonl").read_bytes()).hexdigest()
        if args.build:
            summary = build_arms(summary, jobs, reference_batch, args.out)
    except ValueError as exc:
        print(f"[families] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: summary[k] for k in ("pack", "status", "source_questions", "eligible_questions",
                                             "selected", "selected_gold_changing", "selected_gold_preserving")}
                      | {k: summary[k] for k in ("selected_in_pack", "audit_carried") if k in summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
