"""recompose_v3.py -- the hidden-line allocation of the non-bridge M2 pool (frames F1-F3) for a pack of
any size. Zero model calls; imported by `tools/m2_gen_job.py`.

Rule: the pool keeps the four frames and the bridge, and fills the F1-F3 slots so that every v1.0.1
candidate component carries about the same number of gold lines (a frequency prior then has nothing to
rank), under the allocation rules of `tools/gen_composition_v2.py`. Sizes scale with `scale` (the
pack's non-bridge item count over 34): frames, negative arms, the share cap, the rare cap and the
per-spec caps.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "plugins"), str(ROOT / "core"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ALLOC = ROOT / "docs" / "scripts" / "m2" / "alloc_cases.csv"
SRC = ROOT / "inputs" / "ddx-workup.job.yaml"

#: pool per 34 non-bridge pack items (scale 1): frame sizes and negative-arm sizes
FRAME_W = {"F1": 18, "F2": 25, "F3": 18}
NEG_W = {"F1": 8, "F2": 9, "F3": 8}
#: bridge rows in the pool (the v1.0.1 bridge set, `alloc_cases.csv`, fixed size)
N_BRIDGE_POOL = 28
#: the metanephrine reading of JD-58v2's world is negative while its gold carries the
#: pheochromocytoma line; withheld until a clinical review settles the case.
CLINICAL_HOLD = ("JD-58v2",)
SPEC_CAP, PAIR_CAP = 8, 5                          # gen_composition_v2 MAX_PER_SPEC / MAX_PER_PAIR
LAB_STREAMS = ("HbA1c", "fasting_glucose", "LDL", "triglycerides", "ALT", "AST")
#: Workup generator call (`gen_core_job --variants 3 --insufficient-frac 0.6`, density mixed).
GEN_KW = dict(density="mixed", insufficient_frac=0.6)


def sizes(scale: float) -> dict:
    """Pool sizes at `scale`: F1-F3 frame sizes, negative arms, the pool size (bridge included)."""
    from haenv import pack_size as PS
    frame_n = PS.apportion(int(round(sum(FRAME_W.values()) * scale)), FRAME_W)
    neg_n = PS.apportion(int(round(sum(NEG_W.values()) * scale)), NEG_W)
    return {"frame": frame_n, "neg": neg_n, "pool": N_BRIDGE_POOL + sum(frame_n.values())}


def _h(*xs) -> str:
    return hashlib.sha256("|".join(("m2-v3",) + tuple(str(x) for x in xs)).encode("utf-8")).hexdigest()


def _yaml():
    import yaml
    return yaml


def workup() -> dict[str, dict]:
    return {c["case_id"]: c for c in _yaml().safe_load(SRC.read_text(encoding="utf-8"))["cases"]}


def candidates_v101() -> list[str]:
    """The V-3 prior channel's candidate set: every hidden component of the v1.0.1 workup pack."""
    from haenv.m2 import core
    W = workup()
    return sorted({m for c in W.values() if c["latent"].get("ddx_join_gold") in ("unified", "comorbidity")
                   for m in core.components(c["latent"]["ddx_spec_id"])})


def _group(v: dict) -> str:
    if v["kind"] == "single":
        return "single"
    if v.get("or", 1.0) >= 1.5:
        return "lowjoint" if v["joint"] < 1e-4 else "cluster"
    return "pair"


_GEN_CACHE: dict[str, list[dict]] = {}


def _generated(spec: str, upto: int) -> list[dict]:
    """Variants 1..upto of `spec` from the workup's own generator call (variants 1-3 in one call,
    as the workup was built; further variants in calls of 3 from `variant_start=4`)."""
    from haenv.ddx import ddx_case_specs, strip_inert_gap
    from haenv.overlay import condition_registry
    have = _GEN_CACHE.setdefault(spec, [])
    start = len(have) + 1
    # the workup's calls: independent-type specs 5 variants (`--ind-variants 5`), others 3
    chunk = 5 if condition_registry()[spec]["join_gold"] == "independent" else 3
    while len(have) < upto:
        cs = ddx_case_specs(only=[spec], variants=chunk, variant_start=start, **GEN_KW)
        strip_inert_gap(cs, None)
        for c in cs:
            c["latent"].pop("distractor_level", None)
            c["latent"]["distractor_level"] = "low"
        have.extend(cs)
        start += chunk
    return have[:upto]


def demographics_ok(case: dict) -> bool:
    """The production emission gate GEN24 (`gates.check_demographic_plausibility`) run ahead of the
    build on a generated variant, so a variant whose age band the spec excludes is skipped here."""
    from types import SimpleNamespace
    from haenv.gates import check_demographic_plausibility
    raw = SimpleNamespace(user_profile={"age_range": case["raw"].get("age_range")})
    cs = SimpleNamespace(latent=case["latent"], raw=case["raw"])
    return not check_demographic_plausibility(raw, cs)


def pick_cases(spec: str, k: int, used: set[str], W: dict) -> list[tuple[str, str]]:
    """k sufficient-tier cases of `spec`: workup copies first (case order), then generated variants
    in variant order."""
    out: list[tuple[str, str]] = []
    pool = sorted(c for c, v in W.items() if v["latent"].get("ddx_spec_id") == spec)
    for c in pool:
        if len(out) == k:
            return out
        if c in used or c in CLINICAL_HOLD or W[c]["latent"].get("ddx_insufficient"):
            continue
        out.append((c, "workup"))
        used.add(c)
    n = 3
    while len(out) < k:
        for c in _generated(spec, n):
            if len(out) == k:
                break
            cid = c["case_id"]
            if cid in used or cid in W or c["latent"].get("ddx_insufficient"):
                continue
            if not demographics_ok(c):
                continue
            out.append((cid, "generated"))
            used.add(cid)
        n += 3
        if n > MAX_VARIANT:
            raise RuntimeError(f"cannot find {k} sufficient variants of {spec}")
    return out


def n_tests(spec: str) -> int:
    """K-NT of a spec: the number of gold tests it declares (spec-level, every variant the same)."""
    if spec not in _NT:
        from haenv.ddx import ddx_case_specs
        _NT[spec] = len(ddx_case_specs(only=[spec], variants=1, **GEN_KW)[0]["latent"].get("ddx_tests") or [])
    return _NT[spec]


_NT: dict[str, int] = {}
#: highest variant number tried for one spec
MAX_VARIANT = 60
#: hard-tier share of the pool, K-NT hard side (as `tools/m2_gen_job.py`), share of the pairs inside
#: an association cluster (composition v2: >= 70 %).
TIER_HARD = 0.30
K_NT_HARD_MIN = 6
CLUSTER_MIN = 0.70


def bridge_rows() -> list[dict]:
    """The bridge rows of the pool: the v1.0.1 bridge set of `alloc_cases.csv` (per unified non-red
    spec the variant with the lowest v1.0.1 workup score, plus up to 12 insufficient-tier ones)."""
    return [r for r in csv.DictReader(ALLOC.open(encoding="utf-8")) if r["fw"] == "F0"]


def allocate(scale: float = 1.0) -> dict:
    """The F1-F3 pool rows at `scale` (1.0 = 61 rows, the pool of a 50-item pack)."""
    import gen_composition_v2 as G
    from haenv import pack_size as PS
    from haenv.m2 import core
    from haenv.overlay import condition_registry
    from haenv.registry import is_rare, load_disease_prevalence
    W = workup()
    f0 = bridge_rows()
    sz = sizes(scale)
    frame_n, neg_n, n_pool = sz["frame"], sz["neg"], sz["pool"]
    cand, _, _ = G.candidates()
    prev = load_disease_prevalence()
    cands = candidates_v101()
    share = int(math.floor(G.SHARE_CAP * n_pool))
    rare = max(1, int(round(G.RARE_CAP * scale)))
    cap = {d: (min(share, rare) if is_rare(p) else share) for d, p in prev.items()}
    spec_cap = {"single": max(1, int(round(SPEC_CAP * scale))), "pair": max(1, int(round(PAIR_CAP * scale)))}
    # bridge appearances (insufficient tier included) and bridge gold lines (sufficient only)
    appear, lines = collections.Counter(), collections.Counter()
    for r in f0:
        lat = W[r["case"]]["latent"]
        for m in core.components(lat["ddx_spec_id"]):
            appear[m] += 1
            if not lat.get("ddx_insufficient"):
                lines[m] += 1
    used = {r["case"] for r in f0}
    out: list[dict] = []
    spec_n = collections.Counter()

    def add(cid, spec, fw, arm, src, why):
        out.append({"case": cid, "spec_id": spec, "fw": fw, "frame": fw + ("-neg" if arm == "negative" else ""),
                    "arm": arm, "source": src, "rule": why})
        used.add(cid)
        spec_n[spec] += 1

    # negative arms: the sufficient-tier workup independent cases in case order, dealt F1, F2, F3 in
    # turn; a frame still short takes a case of the independent spec with the fewest cases
    have = collections.Counter()
    order = ("F1", "F2", "F3")
    ind_w = sorted(c for c, v in W.items() if v["latent"].get("ddx_join_gold") == "independent"
                   and not v["latent"].get("ddx_insufficient") and c not in CLINICAL_HOLD)
    i = 0
    for cid in ind_w:
        free = [f for f in order if have[f] < neg_n[f]]
        if not free:
            break
        fw = order[i % 3] if order[i % 3] in free else free[0]
        i += 1
        add(cid, W[cid]["latent"]["ddx_spec_id"], fw, "negative", "workup", "workup-negative")
        have[fw] += 1
    ind_specs = sorted(k for k, v in condition_registry().items() if v["join_gold"] == "independent")
    for fw in [fw for fw in order for _ in range(neg_n[fw] - have[fw])]:
        spec = min(ind_specs, key=lambda s: (spec_n[s], _h("neg", s)))
        (cid, src), = pick_cases(spec, 1, used, W)
        add(cid, spec, fw, "negative", src, "added-negative")
    # F1 positives: acute-event components, apportioned over their room under the cap
    acute = sorted((core._acute_events().get("positive") or {}).keys())
    n_f1 = frame_n["F1"] - neg_n["F1"]
    room = {c: max(0, cap[c] - appear[c]) for c in acute}
    PS.short({"F1 positives": n_f1}, {"F1 positives": sum(room.values())}, "m2 acute-event room")
    for spec, k in sorted(PS.apportion(n_f1, room).items()):
        if not k:
            continue
        for m in core.components(spec):
            appear[m] += k
            lines[m] += k
        for cid, src in pick_cases(spec, k, used, W):
            add(cid, spec, "F1", "positive", src, "acute-event")
    # F2/F3 positives: greedy toward equal line counts over the v1.0.1 candidates. The hard tier needs
    # K-NT 6-8, and only comorbid pairs declare >= 6 gold tests, so the hard share fixes the pair count.
    n_slots = frame_n["F2"] + frame_n["F3"] - neg_n["F2"] - neg_n["F3"]
    n_named = sum(1 for r in out if r["arm"] == "positive") + n_slots
    hard_have = sum(1 for r in out if r["arm"] == "positive" and n_tests(r["spec_id"]) >= K_NT_HARD_MIN)
    n_hard = int(round(TIER_HARD * n_pool)) - hard_have
    n_pair = min(n_slots, max(int(round(G.PAIR_SHARE * n_named)), n_hard))
    quota = {"cluster_min": int(math.ceil(CLUSTER_MIN * n_pair)),
             "lowjoint_max": max(0, int(round(G.LOW_JOINT_SHARE * n_pair))),
             "plain_max": n_pair}
    need_k = {"pair": n_pair, "single": n_slots - n_pair}
    got = collections.Counter()
    picks: list[str] = []
    while need_k["pair"] or need_k["single"]:
        best = None
        for s, v in cand.items():
            kind = "single" if v["kind"] == "single" else "pair"
            grp = _group(v)
            if not need_k[kind]:
                continue
            if kind == "pair":
                if n_tests(s) < K_NT_HARD_MIN:
                    continue
                inside_left = quota["cluster_min"] - got["cluster"] - got["lowjoint"]
                if grp == "lowjoint" and got["lowjoint"] >= quota["lowjoint_max"]:
                    continue
                if grp == "pair" and (got["pair"] >= quota["plain_max"] or need_k["pair"] - 1 < inside_left):
                    continue
            if spec_n[s] >= spec_cap[kind]:
                continue
            if any(appear[m] + 1 > cap.get(m, 10 ** 9) for m in v["members"]):
                continue
            u = dict(lines)
            for m in v["members"]:
                u[m] = u.get(m, 0) + 1
            mu = sum(u.get(c, 0) for c in cands) / len(cands)
            score = sum((u.get(c, 0) - mu) ** 2 for c in cands)
            key = (round(score, 9), -v["weight"], _h("pick", s))
            if best is None or key < best[0]:
                best = (key, s, kind, grp)
        if best is None:
            raise SystemExit(f"[m2 pool] no admissible entry left; need {need_k}, got {dict(got)}")
        _, s, kind, grp = best
        need_k[kind] -= 1
        got[grp] += 1
        picks.append(s)
        spec_n[s] += 1
        for m in cand[s]["members"]:
            appear[m] += 1
            lines[m] += 1
    pos23: list[tuple[str, str, str]] = []
    for spec, k in sorted(collections.Counter(picks).items()):
        spec_n[spec] -= k
        for cid, src in pick_cases(spec, k, used, W):
            pos23.append((cid, spec, src))
    # frames: sha order, F3 takes the first ones with a live lab stream
    n_f3 = frame_n["F3"] - neg_n["F3"]
    order23 = sorted(pos23, key=lambda t: _h("frame", t[0]))
    f3 = [t for t in order23 if has_lab_stream(case_of(t[0], t[2], W))][:n_f3]
    for t in order23:
        if len(f3) >= n_f3:
            break
        if t not in f3:
            f3.append(t)
    for cid, spec, src in order23:
        add(cid, spec, "F3" if (cid, spec, src) in f3 else "F2", "positive", src, "uniform-lines-hard-pairs")
    for fw, n in frame_n.items():
        assert sum(1 for r in out if r["fw"] == fw) == n, (fw, n)
    return {"rows": sorted(out, key=lambda r: r["case"]), "lines": dict(lines), "appear": dict(appear),
            "cap": cap, "quota": quota, "groups": dict(got), "picks": collections.Counter(picks), "sizes": sz}


def case_of(cid: str, src: str, W: dict | None = None) -> dict:
    if src == "workup":
        return (W or workup())[cid]
    for lst in _GEN_CACHE.values():
        for c in lst:
            if c["case_id"] == cid:
                return c
    raise KeyError(cid)


def has_lab_stream(case: dict) -> bool:
    import haenv  # noqa: F401
    from haenv.build import clinical_plan
    raw = case["raw"]
    live = clinical_plan(raw["disease"], raw["devices"], raw.get("comorbidities") or [], case["case_id"])
    return any(st in live for st in LAB_STREAMS)


def generated_cases(rows: list[dict]) -> dict[str, dict]:
    """case_id -> generated case (deep copy) for every `source == generated` row."""
    out = {}
    for r in rows:
        if r["source"] != "generated":
            continue
        spec = r["spec_id"]
        n = 3
        while True:
            hit = [c for c in _generated(spec, n) if c["case_id"] == r["case"]]
            if hit:
                out[r["case"]] = copy.deepcopy(hit[0])
                break
            n += 3
            if n > MAX_VARIANT:
                raise KeyError(r["case"])
    return out


def main(argv=None) -> int:
    import logging
    logging.disable(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args(argv)
    res = allocate(a.scale)
    w = csv.DictWriter(sys.stdout, fieldnames=["case", "spec_id", "fw", "frame", "arm", "source", "rule"],
                       lineterminator="\n")
    w.writeheader()
    for r in res["rows"]:
        w.writerow(r)
    print("# sizes", res["sizes"], "quota", res["quota"], "groups", res["groups"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
