"""pack_balance.py -- two pool-level steps of `tools/generation/make_pack.py` between the pool build and the
shared audit. Zero model calls; deterministic (no seed is read).

1. Pre-selection exclusion (`preselect`). A whole-pool invariant that exists to keep the published
   items unique (`PRESELECT_GATES`: `diversity` -- one independently sampled patient per item, no
   copied benign complaint bundle) used to refuse a pack for a duplicate pair anywhere in the pool,
   even when neither member was published. Now the violating candidates are excluded from the
   selectable pool before the audit samples the published items (`tools/validation/pack_audit.py --exclude`):
   per duplicate group the smallest case id stays, the others are excluded; the exclusion is logged
   in `make_pack.json` and `<out>/preselect.json`. The invariant stays fail-closed: the published
   items must pass it, and so must the selectable pool (pool minus the exclusions); otherwise the
   pack is refused as before. A pool without a violation is untouched.

2. Shortcut balance (`shortcut_balance`). SA-6 (`src/haenv/world/validation/shared_audit.py:surface_classifier`) trains
   a GBM on content-free features of the emitted pool and refuses the pack when its chance-corrected
   balanced accuracy reaches the frozen line `SHORTCUT_CC` (and the permutation null's 0.99 quantile).
   When the audit's own reader would refuse the pool, candidates are removed from the pool job by
   adversarial filtering (AFLite; Le Bras et al., ICML 2020): per round, the reader's out-of-fold
   predictions over its own 6 group orders x 5 seeds rank the items by how often they are classified
   correctly; the most identifiable items are removed (2% of the pool per round), only from cells
   with surplus, never a protected id (ids fixed by the job: `pack_ids`, `paired_pack_ids`), while the
   pool keeps `POOL_FACTOR` x N items and the selection stays feasible (SA-0, SA-3); rounds stop
   when the reader is below the line. The job (and the paired control job) lose those cases and the
   pool is rebuilt; the removal is recorded in `_provenance.shortcut_balance`. A pool the reader
   does not refuse is untouched, so packs that passed SA-6 are byte-identical.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import re

#: whole-pool invariants whose violating candidates are excluded before selection
PRESELECT_GATES = ("diversity",)
#: share of the current pool removed per shortcut-balance round
BALANCE_STEP = 0.02
BALANCE_MAX_ROUNDS = 40


# ------------------------------------------------------------------------- 1. pre-selection exclusion

def _invariant_items(pack: str, batch) -> list[dict]:
    """The pool items as `tools/validation/invariant_gates.py:report` reads them (pack adapter view included)."""
    import haenv.world.validation.invariants as INV
    from haenv.interaction.query.adapter import ADAPTERS, load_pack_adapters
    items = INV.load(pathlib.Path(batch), pack, None)
    load_pack_adapters()
    audit_block = getattr(ADAPTERS.get(pack), "audit_block", None)
    if audit_block is not None:
        blk = ADAPTERS[pack].block
        for it in items:
            pc = it["sp"].get("prediction_context") or {}
            it["sp"] = {**it["sp"], "prediction_context": {**pc, blk: audit_block(pc.get(blk) or {})}}
    return items


def _diversity_groups(items: list[dict], violations: list[str]) -> list[list[str]]:
    """Groups of case ids that break `src/haenv/world/validation/invariants.py:diversity` together."""
    from haenv.world.validation.invariants import skeleton_key
    groups = []
    by = collections.defaultdict(list)
    for it in items:
        by[skeleton_key(it)].append(it["case_id"])
    groups += [sorted(v) for v in by.values() if len(v) > 1]
    if items and items[0]["pack"] in ("p2", "p3", "p4"):
        base = collections.defaultdict(list)
        for it in items:
            base[re.sub(r"v\d+$", "", it["case_id"])].append(it["case_id"])
        groups += [sorted(v) for v in base.values() if len(v) > 1]
    for v in violations:
        m = re.match(r"(\S+)/(\S+): benign Jaccard", v)
        if m:
            groups.append(sorted(m.groups()))
    return groups


def choose_exclusions(groups, already=()) -> set:
    """Per group of candidates that break the gate together, the smallest case id not yet excluded
    stays selectable and the others are excluded (deterministic, seed-free)."""
    excluded = set(already)
    for grp in groups:
        keep = sorted(c for c in grp if c not in excluded)
        excluded |= set(keep[1:])
    return excluded - set(already)


def preselect(pack: str, batch, job_doc: dict) -> dict | None:
    """Exclusions for `PRESELECT_GATES` on the pool `batch`, or None when nothing is excluded (or the
    job fixes the published ids, so there is no selection to exclude from)."""
    prov = job_doc.get("_provenance") or {}
    if prov.get("pack_ids") or prov.get("paired_pack_ids"):
        return None
    import haenv.world.validation.invariants as INV
    items = _invariant_items(pack, batch)
    excluded, detail = set(), {}
    for gid in PRESELECT_GATES:
        fn = getattr(INV, gid)
        bad = fn(items)
        if not bad:
            continue
        excluded |= choose_exclusions(_diversity_groups(items, bad), excluded)
        left = [it for it in items if it["case_id"] not in excluded]
        after = fn(left)
        detail[gid] = {"pool_violations": bad, "excluded": sorted(excluded), "selectable_violations": after,
                       "selectable_pass": not after}
    if not excluded:
        return None
    return {"gates": detail, "excluded": sorted(excluded), "n_pool": len(items),
            "selectable_pass": all(d["selectable_pass"] for d in detail.values())}


def waive(refusals: list[str], gates: dict, pre: dict | None) -> tuple[list[str], list[str]]:
    """Refusals that remain, and those of a `PRESELECT_GATES` gate whose published items pass and
    whose selectable pool passes after the logged exclusion (the pool violation was never selectable)."""
    keep, waived = [], []
    for r in refusals:
        gid = r.split(":", 1)[0]
        g = gates.get(gid) or {}
        if (pre and gid in PRESELECT_GATES and g.get("pack", 1) == 0
                and (pre["gates"].get(gid) or {}).get("selectable_pass")):
            waived.append(r)
        else:
            keep.append(r)
    return keep, waived


# ------------------------------------------------------------------------------ 2. shortcut balance

def _reader_inputs(A, batch, job, ref, keep=None):
    import haenv.world.validation.shared_audit as SA
    ctx = A.context(pathlib.Path(batch), job and pathlib.Path(job), ref and pathlib.Path(ref)) if A.context else {}
    pool = SA.load_items(pathlib.Path(batch), A.block, A.class_of)
    if A.prepare:
        pool = A.prepare(pool, ctx)
    if keep is not None:
        pool = [it for it in pool if it["case_id"] in keep]
    return ctx, pool


def _identifiability(rows, features, label, groups="spec", n_orders=6, n_seeds=5, jobs=32):
    """Mean balanced accuracy and, per row, the share of SA-6's cross-validation runs (same orders,
    seeds and model as `surface_classifier`) that classify it correctly out of fold."""
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from haenv.world.validation.shared_audit import balanced_acc
    X = np.asarray([[float(r[k]) for k in features] for r in rows], float)
    y = np.asarray([r[label] for r in rows])
    g = [r[groups] for r in rows]
    combos = []
    for o in range(n_orders):
        ug = sorted(set(g), key=str)
        np.random.default_rng(o).shuffle(ug)
        idx = {k: i for i, k in enumerate(ug)}
        splits = list(GroupKFold(5).split(X, y, np.asarray([idx[k] for k in g])))
        combos += [(sd, splits) for sd in range(n_seeds)]

    def run(sd, splits):
        pred = np.empty(len(y), dtype=object)
        for tr, te in splits:
            pred[te] = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=sd).fit(X[tr], y[tr]).predict(X[te])
        return balanced_acc(y.tolist(), pred.tolist()), (pred == y)
    out = Parallel(n_jobs=jobs)(delayed(run)(sd, sp) for sd, sp in combos)
    hits = np.mean([h for _, h in out], axis=0)
    return float(np.mean([b for b, _ in out])), hits.tolist()


def _reader_fails(A, rows, jobs) -> dict:
    """{target: cc} of the gate targets the audit's SA-6 would refuse on these rows (exact gate rule)."""
    import haenv.world.validation.shared_audit as SA
    surface = tuple(f for f in A.surface if f not in A.decisive)
    gt = A.surface_targets if A.gate_targets is None else A.gate_targets
    bad = {}
    for t in gt:
        r = SA.surface_classifier(rows, surface, label=t, strata=A.strata, n_perm=0, jobs=jobs)
        if r["cc"] >= SA.SHORTCUT_CC:
            full = SA.surface_classifier(rows, surface, label=t, strata=A.strata, jobs=jobs)
            if not full["pass"]:
                bad[t] = full["cc"]
    return bad


def shortcut_balance(A, configure_select, batch, job, ref, n_items: int, jobs: int = 32) -> dict | None:
    """Ids to remove from the pool job so SA-6 no longer refuses it, or None when it does not refuse.
    `configure_select(A)` installs the job's selection rules on the audit object (as `pack_audit` does)."""
    import haenv.world.validation.shared_audit as SA
    from haenv.world.validation.pack_size import POOL_FACTOR
    import yaml
    configure_select(A)
    ctx, pool = _reader_inputs(A, batch, job, ref)
    rows = A.rows(pool, ctx)
    bad = _reader_fails(A, rows, jobs)
    if not bad:
        return None
    prov = (yaml.safe_load(pathlib.Path(job).read_text(encoding="utf-8")) or {}).get("_provenance") or {}
    protected = set(prov.get("pack_ids") or []) | set(prov.get("paired_pack_ids") or [])
    surface = tuple(f for f in A.surface if f not in A.decisive)
    target = sorted(bad)[0]
    keep = {it["case_id"] for it in pool}
    rounds, removed = [], []
    need_pool = POOL_FACTOR * n_items

    def feasible(ids):
        cur = [it for it in pool if it["case_id"] in ids]
        try:
            pk = A.select(cur) if A.select else cur
        except Exception as exc:                                  # noqa: BLE001  infeasible selection
            return False, f"{type(exc).__name__}: {exc}"
        comp = SA.composition([A.cell_of(it, ctx) for it in pk], A.planned_cells(pk, ctx))
        ok = len(pk) == n_items and comp["pass"] and SA.pool_power(len(cur), len(pk))["pass"]
        return ok, None if ok else f"n={len(pk)} SA-3={comp['pass']}"

    cc = None
    for rnd in range(BALANCE_MAX_ROUNDS):
        cur = [it for it in pool if it["case_id"] in keep]
        rws = A.rows(cur, ctx)
        ba, hits = _identifiability(rws, surface, target, jobs=jobs)
        cc = SA.chance_corrected(ba, len({r[target] for r in rws}))
        if cc < SA.SHORTCUT_CC:
            break
        sel = A.select(cur) if A.select else cur
        need = collections.Counter(A.cell_of(it, ctx) for it in sel)
        have = collections.Counter(A.cell_of(it, ctx) for it in cur)
        order = sorted(range(len(cur)), key=lambda i: (-hits[i], hashlib.sha256(cur[i]["case_id"].encode()).hexdigest()))
        step = max(1, math.ceil(BALANCE_STEP * len(cur)))
        while step >= 1:
            take, have2 = [], collections.Counter(have)
            for i in order:
                cid, cell = cur[i]["case_id"], A.cell_of(cur[i], ctx)
                if cid in protected or have2[cell] - 1 < need.get(cell, 0) or len(keep) - len(take) - 1 < need_pool:
                    continue
                take.append(cid); have2[cell] -= 1
                if len(take) == step:
                    break
            if not take:
                break
            ok, why = feasible(keep - set(take))
            if ok:
                break
            step //= 2
        if not take or step < 1:
            rounds.append({"round": rnd, "cc": round(cc, 4), "stopped": "no feasible removal"})
            return {"target": target, "rounds": rounds, "removed": removed, "pass": False}
        keep -= set(take); removed += take
        rounds.append({"round": rnd, "cc": round(cc, 4), "removed": len(take), "pool": len(keep)})
    rounds.append({"final_cc": round(cc, 4), "pool": len(keep)})
    return {"target": target, "line": SA.SHORTCUT_CC, "reader": "shared_audit.surface_classifier (SA-6)",
            "method": "adversarial filtering, out-of-fold identifiability, surplus cells only",
            "refused_cc": bad, "rounds": rounds, "removed": sorted(removed), "pass": cc is not None and cc < SA.SHORTCUT_CC}


def drop_cases(job_path, ids: set, record: dict) -> None:
    """Remove `ids` from a job's cases (header comments kept) and record the balance in `_provenance`."""
    import yaml
    p = pathlib.Path(job_path)
    src = p.read_text(encoding="utf-8")
    head = "".join(ln for ln in src.splitlines(True) if ln.startswith("#"))
    doc = yaml.safe_load(src)
    doc["cases"] = [c for c in doc["cases"] if c["case_id"] not in ids]
    prov = doc.setdefault("_provenance", {})
    for k in ("eligible_pack_ids",):
        if k in prov:
            prov[k] = [x for x in prov[k] if x not in ids]
    prov.setdefault("shortcut_balance", []).append({k: record[k] for k in ("target", "line", "method", "refused_cc", "rounds", "removed")
                                                    if k in record})
    p.write_text(head + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
