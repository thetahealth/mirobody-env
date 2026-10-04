"""Composition-v2 main-pack generator (maintainer tool, zero model calls).

Builds a `joint_dx` job whose hidden-diagnosis mix is weighted by prevalence instead of "every spec x
k variants" (M2 spec C1/C2/C3). The per-disease cap is set by the "no prior shortcut" rule (a disease
appears in at most `SHARE_CAP` of the cases, a rare one at most `RARE_CAP` times), not by matching the
population rate; the earlier Poisson-at-literature-prevalence cap is still available as `--cap-mode poisson`:

* named single diseases and comorbid pairs are drawn with weight prevalence**alpha (alpha=0.5, the
  square root); a pair weighs (prev_a * prev_b * OR)**alpha;
* a pair is admitted only when it sits in an association cluster with OR >= 1.5 or both diseases have
  prevalence >= 1% and an independent product >= 1e-4 (`registry.association_admissible`);
* independent (no hidden named disease) items take at least 40% of the job;
* a per-spec cap keeps one common disease from filling the job (weights are water-filled above the cap);
* every case carries `latent.composition_v2` so `ddx_case_specs` applies symptom penetrance and synonym
  wording (`registry/symptom_penetrance.yaml`). Jobs without the flag build exactly as before.

Usage:  python tools/gen_composition_v2.py --n 145 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

JOB_ID = "ddx-timeline-v2c"
ALPHA = 0.5
INDEPENDENT_SHARE = 0.40      # >= 0.35 required (PREREG V2); the margin absorbs emission-gate losses
MAX_PER_SPEC = 8
MAX_PER_PAIR = 5
P_MIN = 0.02               # `--cap-mode poisson`: Poisson P(X >= cases) at literature prevalence stays above this
SHARE_CAP = 0.12           # `--cap-mode shortcut`: a disease (alone or as a pair member) in at most this share of the cases
RARE_CAP = 4               # ... and a rare disease (registry.is_rare) in at most this many cases
PAIR_SHARE = 0.45           # of the named items; v1 had ~50% comorbid pairs (PREREG V1: comorbidity 25-35%)
CLUSTER_SHARE = 0.80        # of the pair items; PREREG V7 needs >= 70% inside a registered cluster
LOW_JOINT = 1e-4            # PREREG V6: pairs whose independent joint probability is below this stay under a tenth of the pair items
LOW_JOINT_SHARE = 0.05      # of the pair items


def candidates(alpha: float = ALPHA) -> tuple[dict[str, dict], list[str], dict[str, str]]:
    """`({spec_id: {weight, kind, members, rare, or}}, independent_spec_ids, rejected{spec_id: why})`."""
    from haenv.overlay import HAENV_COMORBID_PAIRS, condition_registry
    from haenv.registry import association_admissible, is_rare, load_association_pairs, load_disease_prevalence
    reg = condition_registry()
    prev, assoc = load_disease_prevalence(), load_association_pairs()
    out: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    for sid, sp in reg.items():
        if sp["join_gold"] == "independent":
            continue
        if sid in HAENV_COMORBID_PAIRS:
            a, b = HAENV_COMORBID_PAIRS[sid]["pair"]
            ok, orv, why = association_admissible(a, b, prev, assoc)
            if not ok:
                rejected[sid] = f"{a}+{b}: {why}"
                continue
            pa, pb = prev[a]["value"], prev[b]["value"]
            out[sid] = {"kind": "pair", "members": [a, b], "or": orv, "admission": why, "joint": pa * pb,
                        "weight": (pa * pb * orv) ** alpha,
                        "rare": bool(is_rare(prev[a]) or is_rare(prev[b]))}
        else:
            if sid not in prev:
                rejected[sid] = "no prevalence registered"
                continue
            out[sid] = {"kind": "single", "members": [sid], "weight": float(prev[sid]["value"]) ** alpha,
                        "rare": bool(is_rare(prev[sid]))}
    ind = sorted(sid for sid, sp in reg.items() if sp["join_gold"] == "independent")
    return out, ind, rejected


def waterfill(weights: dict[str, float], total: int, cap: int, ideal_out: dict | None = None) -> dict[str, int]:
    """Integer allocation proportional to `weights`, no entry above `cap` (excess re-spread over the rest),
    largest-remainder rounding with a content-hash tie-break."""
    fixed: dict[str, int] = {}
    free = dict(weights)
    left = total
    while True:
        s = sum(free.values())
        over = [k for k, w in free.items() if left * w / s > cap]
        if not over:
            break
        for k in over:
            fixed[k] = cap
            left -= cap
            free.pop(k)
        if not free:
            break
    alloc = dict(fixed)
    if ideal_out is not None:
        ideal_out.update({k: float(v) for k, v in fixed.items()})
    if free and left > 0:
        s = sum(free.values())
        ideal = {k: left * w / s for k, w in free.items()}
        if ideal_out is not None:
            ideal_out.update(ideal)
        base = {k: int(math.floor(v)) for k, v in ideal.items()}
        rest = left - sum(base.values())
        order = sorted(free, key=lambda k: (-(ideal[k] - base[k]), hashlib.sha256(("alloc|" + k).encode()).hexdigest()))
        for k in order[:rest]:
            base[k] += 1
        alloc.update({k: v for k, v in base.items() if v})
    return {k: v for k, v in alloc.items() if v}


def _pois_sf(k: int, lam: float) -> float:
    return 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))


def disease_caps(n: int, p_min: float = P_MIN) -> dict[str, int]:
    """Largest number of cases in which a disease may appear (alone or as a pair member) such that a Poisson
    at its literature prevalence (`n * value`) still gives P(X >= k) > `p_min`; at least 1."""
    from haenv.registry import load_disease_prevalence
    caps = {}
    for d, pv in load_disease_prevalence().items():
        lam = n * float(pv["value"])
        k = 1
        while k < n and _pois_sf(k + 1, lam) > p_min:
            k += 1
        caps[d] = k
    return caps


def disease_caps_shortcut(n: int, share: float = SHARE_CAP, rare_cap: int = RARE_CAP) -> dict[str, int]:
    """Per-disease cap from the no-prior-shortcut rule: `floor(share * n)` cases for any disease, `rare_cap` for a rare one."""
    from haenv.registry import is_rare, load_disease_prevalence
    top = max(1, int(math.floor(share * n)))
    return {d: (min(top, rare_cap) if is_rare(pv) else top) for d, pv in load_disease_prevalence().items()}


def enforce_disease_caps(named: dict[str, int], cand: dict, groups: dict[str, str], group_cap: dict[str, int],
                         caps: dict[str, int], ideal: dict[str, float]) -> dict[str, int]:
    """Move slots away from diseases whose total appearances (single + pair membership) exceed `caps`, to the entry of
    the same group with the most unused ideal share whose members still have room. A slot with nowhere to go is dropped."""
    named = dict(named)
    for _ in range(500):
        tot = collections.Counter()
        for e, c in named.items():
            for m in cand[e]["members"]:
                tot[m] += c
        viol = [d for d in tot if tot[d] > caps.get(d, 10 ** 9)]
        if not viol:
            break
        d = max(viol, key=lambda x: tot[x] - caps[x])
        # take the slot from a single-disease entry first: pair slots are scarce and carry the cluster quota
        src = max((e for e in named if d in cand[e]["members"]),
                  key=lambda e: (groups[e] == "single", named[e], e))
        named[src] -= 1
        if not named[src]:
            del named[src]
        g = groups[src]
        tot = collections.Counter()
        for e, c in named.items():
            for m in cand[e]["members"]:
                tot[m] += c
        room = [e for e in cand if groups[e] == g and e != src and named.get(e, 0) < group_cap[g]
                and all(tot[m] + 1 <= caps.get(m, 10 ** 9) for m in cand[e]["members"])]
        if not room:                      # the group is full: hand the slot to single diseases
            room = [e for e in cand if groups[e] == "single" and e != src and named.get(e, 0) < group_cap["single"]
                    and all(tot[m] + 1 <= caps.get(m, 10 ** 9) for m in cand[e]["members"])]
        if room:
            dst = max(room, key=lambda e: (ideal.get(e, 0.0) - named.get(e, 0), cand[e]["weight"], e))
            named[dst] = named.get(dst, 0) + 1
    return named


def top_up(named: dict[str, int], cand: dict, groups: dict[str, str], group_cap: dict[str, int], caps: dict[str, int],
           ideal: dict[str, float], target: int) -> dict[str, int]:
    """`enforce_disease_caps` drops a slot with nowhere to go; give dropped slots to the entries that still have room
    (same-kind pairs first, then single diseases, then any entry), so the job keeps `target` named items."""
    named = dict(named)
    order = ("cluster", "lowjoint", "pair", "single")
    while sum(named.values()) < target:
        tot = collections.Counter()
        for e, c in named.items():
            for m in cand[e]["members"]:
                tot[m] += c
        room = [e for e in cand if named.get(e, 0) < group_cap[groups[e]]
                and all(tot[m] + 1 <= caps.get(m, 10 ** 9) for m in cand[e]["members"])]
        if not room:
            raise RuntimeError(f"cannot place {target - sum(named.values())} more named items under the per-disease caps")
        # singles first: pair slots are scarce and carry the cluster quota
        dst = max(room, key=lambda e: (groups[e] == "single", ideal.get(e, 0.0) - named.get(e, 0), cand[e]["weight"], e))
        named[dst] = named.get(dst, 0) + 1
    return named


def allocate(n: int, *, alpha: float = ALPHA, independent_share: float = INDEPENDENT_SHARE,
             cap: int = MAX_PER_SPEC, pair_cap: int = MAX_PER_PAIR, pair_share: float = PAIR_SHARE,
             cluster_share: float = CLUSTER_SHARE, cap_mode: str = "shortcut",
             low_joint_share: float = LOW_JOINT_SHARE) -> dict:
    """Quota split: independent items first (>= 40%), then of the named items `pair_share` go to comorbid
    pairs (`cluster_share` of those to pairs with a registered OR >= 1.5, the rest to pairs admitted only
    on prevalence), and the remainder to single diseases. Inside each group: weight**alpha, water-filled
    under `cap`."""
    cand, ind, rejected = candidates(alpha)
    n_ind = int(math.ceil(independent_share * n))
    ind_alloc = waterfill({k: 1.0 for k in ind}, n_ind, cap=10 ** 6)
    ideal: dict[str, float] = {}
    n_named = n - n_ind
    n_pair = int(round(pair_share * n_named))
    n_low = int(round(low_joint_share * n_pair))
    n_cl = int(round(cluster_share * (n_pair - n_low)))
    singles = {k: v["weight"] for k, v in cand.items() if v["kind"] == "single"}
    def _grp(v):
        if v["kind"] == "single":
            return "single"
        if v.get("or", 1.0) >= 1.5:
            return "lowjoint" if v["joint"] < LOW_JOINT else "cluster"
        return "pair"
    groups = {k: _grp(v) for k, v in cand.items()}
    cl_pairs = {k: v["weight"] for k, v in cand.items() if groups[k] == "cluster"}
    lo_pairs = {k: v["weight"] for k, v in cand.items() if groups[k] == "lowjoint"}
    pr_pairs = {k: v["weight"] for k, v in cand.items() if groups[k] == "pair"}
    named: dict[str, int] = {}
    for grp, m, cp in ((singles, n_named - n_pair, cap), (cl_pairs, n_cl, pair_cap), (lo_pairs, n_low, pair_cap),
                       (pr_pairs, n_pair - n_cl - n_low, pair_cap)):
        named.update(waterfill(grp, m, cp, ideal))
    gcap = {"single": cap, "cluster": pair_cap, "lowjoint": pair_cap, "pair": pair_cap}
    dcaps = disease_caps_shortcut(n) if cap_mode == "shortcut" else disease_caps(n)
    named = enforce_disease_caps(named, cand, groups, gcap, dcaps, ideal)
    named = top_up(named, cand, groups, gcap, dcaps, ideal, n_named)
    return {"n": n, "alpha": alpha, "cap_mode": cap_mode, "disease_caps": dcaps, "cap": cap, "pair_cap": pair_cap, "pair_share": pair_share, "cluster_share": cluster_share,
            "independent": ind_alloc, "named": named, "ideal": ideal, "candidates": cand, "rejected_pairs": rejected,
            "total_weight": sum(v["weight"] for v in cand.values())}


def _vnum(case_id: str) -> int:
    import re
    m = re.search(r"v(\d+)$", str(case_id))
    return int(m.group(1)) if m else 1


def _header(argv: str, out: list[dict], a) -> str:
    import gen_core_job as g
    jg = collections.Counter(c["latent"]["ddx_join_gold"] for c in out)
    n_ins = sum(1 for c in out if c["latent"].get("ddx_insufficient"))
    return (g._CANARY()
            + f"# Generated file -- do not edit by hand. Rebuild with `tools/gen_composition_v2.py {argv}`\n#\n"
              f"# Composition v2 main pack: {len(out)} cases. Named diseases are drawn with weight prevalence^{a.alpha}\n"
              f"# (per-spec cap {a.max_per_spec}, per-disease cap {a.cap_mode}); comorbid pairs only inside association clusters or between\n"
              f"# common diseases; {jg.get('independent', 0)} cases ({jg.get('independent', 0) / len(out):.0%}) carry no hidden named disease.\n"
              f"# Symptoms are expressed with their literature penetrance and reworded from the registered synonym table\n"
              f"# (`latent.composition_v2`). join_gold: {dict(jg)}. Insufficient tier {n_ins}.\n"
              f"# Variants are not independent questions (they share the hidden diagnosis).\n#\n# SYNTHETIC data, evaluation only, not medical advice.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=145)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--independent-share", type=float, default=INDEPENDENT_SHARE)
    ap.add_argument("--max-per-spec", type=int, default=MAX_PER_SPEC)
    ap.add_argument("--max-per-pair", type=int, default=MAX_PER_PAIR)
    ap.add_argument("--pair-share", type=float, default=PAIR_SHARE)
    ap.add_argument("--cluster-share", type=float, default=CLUSTER_SHARE)
    ap.add_argument("--cap-mode", choices=("shortcut", "poisson"), default="shortcut")
    ap.add_argument("--dist", default="low:1,high:1")
    ap.add_argument("--insufficient-frac", type=float, default=0.6)
    ap.add_argument("--job-id", default=JOB_ID)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--slack", type=int, default=0,
                    help="generate this many extra variants per allocated spec (a candidate pool; some variants are "
                         "blocked by the emission gate). The pool is written as <job-id>-full.job.yaml")
    ap.add_argument("--select", default=None, metavar="CASES_JSONL",
                    help="cases.jsonl of a build of the pool: keep, per spec, the first `count` variants that emitted "
                         "and write the final <job-id>.job.yaml")
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    import gen_core_job as g
    argv = " ".join(sys.argv[1:])
    plan_p = out_dir / f"{a.job_id}.plan.json"
    if a.select:
        plan = json.loads(plan_p.read_text(encoding="utf-8"))
        full = yaml.safe_load((out_dir / f"{a.job_id}-full.job.yaml").read_text(encoding="utf-8"))
        emitted = {json.loads(l)["case_id"] for l in open(a.select, encoding="utf-8")}
        want = {**plan["independent"], **plan["named"]}
        by: dict[str, list[dict]] = collections.defaultdict(list)
        for c in full["cases"]:
            if str(c["case_id"]) in emitted:
                by[str(c["latent"]["ddx_spec_id"])].append(c)
        out, short = [], {}
        for sid, k in sorted(want.items()):
            pool = sorted(by.get(sid, []), key=lambda c: _vnum(c["case_id"]))
            out += pool[:k]
            if len(pool) < k:
                short[sid] = (len(pool), k)
        doc = {**{k: v for k, v in full.items() if k != "cases"}, "job_id": a.job_id, "report": f"eval-{a.job_id}.md", "cases": out}
        p = out_dir / f"{a.job_id}.job.yaml"
        p.write_text(_header(argv, out, a) + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"[v2] selected {len(out)} of {sum(want.values())} wanted; short={short} -> {p}")
        return 0
    from haenv.ddx import ddx_case_specs
    plan = allocate(a.n, alpha=a.alpha, independent_share=a.independent_share, cap=a.max_per_spec, pair_cap=a.max_per_pair,
                    pair_share=a.pair_share, cluster_share=a.cluster_share, cap_mode=a.cap_mode)
    cases: list[dict] = []
    for sid, k in sorted({**plan["independent"], **plan["named"]}.items()):
        cases += ddx_case_specs(only=[sid], variants=k + a.slack, density="mixed",
                                insufficient_frac=a.insufficient_frac, composition_v2=True)
    lvl = g.assign_distractor_levels(cases, a.dist)
    out = []
    for c in cases:
        c = copy.deepcopy(c)
        g.apply_distractor_level(c, lvl[str(c["case_id"])])
        out.append(c)
    jid = a.job_id + ("-full" if a.slack else "")
    doc = {"job_id": jid, "task_type": "joint_dx", "multiround": False, "include_baseline": True,
           "models": [], "sample_cases": 5, "report": f"eval-{jid}.md", **g.MAIN_EXTRA, "cases": out}
    p = out_dir / f"{jid}.job.yaml"
    p.write_text(_header(argv, out, a) + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    plan_p.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    jg = collections.Counter(c["latent"]["ddx_join_gold"] for c in out)
    print(f"[v2] -> {p} ({len(out)} cases) join_gold={dict(jg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
