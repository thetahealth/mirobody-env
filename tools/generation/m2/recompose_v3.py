"""recompose_v3.py -- the hidden-line allocation of the non-bridge M2 pool (frames F1-F3) for a pack of
any size. Zero model calls; imported by `tools/generation/m2_gen_job.py`.

Rule: the pool keeps the four frames and the bridge, and fills the F1-F3 slots so that every v1.0.1
candidate component carries about the same number of gold lines (a frequency prior then has nothing to
rank), under the allocation rules of `tools/generation/gen_composition_v2.py`. Sizes scale with `scale` (the
pack's non-bridge item count over 34): frames, negative arms, the share cap, the rare cap and the
per-spec caps.

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
import copy
import csv
import hashlib
import math
import pathlib
import sys

ROOT = _repository_root(__file__)
for _p in (str(ROOT), str(ROOT / "plugins"), str(ROOT / "core"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ALLOC = ROOT / 'tools/generation/m2/alloc_cases.csv'
SRC = ROOT / 'benchmarks' / 'inputs' / 'ddx-workup.job.yaml'

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
    import haenv.world.validation.pack_size as PS
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


def expansion_candidates(registry: dict | None = None, metadata: dict | None = None) -> dict:
    """Direct atomic admission, with explicit clinical systems; never read prevalence.

    Metadata is the `conditions` mapping of the integrated clinical_expansion.yaml.
    It may instead be retained on each synthesized spec by the catalogue integrator.
    """
    from haenv.m2.pack import EXPANSION_IDS, EXPANSION_FRAMES
    if registry is None:
        from haenv.world.overlay import condition_registry
        registry = condition_registry()
    metadata = metadata or {}
    missing = sorted(set(EXPANSION_IDS) - set(registry))
    if missing:
        raise ValueError(f"expansion specs not integrated: {missing}")
    out = {}
    for sid in EXPANSION_IDS:
        sp = registry[sid]
        # The public manifest is metadata only; consume IDs and clinical system,
        # never a second condition definition, prevalence or review-based gate.
        system = metadata.get(sid, {}).get("system", sp.get("system"))
        if not isinstance(system, str) or not system.strip():
            raise ValueError(f"{sid}: explicit catalogue system is required")
        frames = sp.get("eligible_frames", EXPANSION_FRAMES[sid])
        if not frames or set(frames) - {"F1", "F2", "F3"}:
            raise ValueError(f"{sid}: invalid eligible_frames: {frames}")
        if sp.get("join_gold") != "unified":
            raise ValueError(f"{sid}: expansion admission requires an atomic unified condition")
        out[sid] = {"system": system, "outside_endocrine": system != "endocrine",
                    "eligible_frames": tuple(frames), "expanded": True}
    return out


def benchmark_candidates() -> dict:
    """Legacy admitted diagnoses plus direct expansion; epidemiology is not a weight.

    Old pair association admission is preserved. Catalogue metadata remains outside
    conditions_unified.yaml, whose clinical schema correctly rejects extra fields.
    """
    from tools.generation import gen_composition_v2 as G
    from haenv.m2 import core
    from haenv.world.overlay import condition_registry
    from haenv.resources.regpath import registry_path
    reg = condition_registry()
    metadata = {}
    p = registry_path("clinical_expansion.yaml")
    if p.is_file():
        metadata = (_yaml().safe_load(p.read_text(encoding="utf-8")) or {}).get("conditions") or {}
    expanded = expansion_candidates(reg, metadata)
    old, independent, _ = G.candidates()
    out = {}
    for sid in sorted(set(old) | set(independent)):
        specialties = {s for c in core.components(sid) for s in reg[c].get("specialty") or ()}
        endocrine = any("内分泌" in s for s in specialties)
        out[sid] = {"system": "endocrine" if endocrine else "/".join(sorted(specialties)),
                    "outside_endocrine": not endocrine, "expanded": False,
                    "eligible_frames": ("F1", "F2", "F3"),
                    "arm": "negative" if sid in independent else "positive"}
    out.update({sid: {**v, "arm": "positive"} for sid, v in expanded.items()})
    return out


def eligible_frames(case: dict, allowed: tuple = ("F1", "F2", "F3")) -> tuple:
    """F3 needs both repeat and changed-result plans on a real live laboratory stream. The F2 negative
    arm needs a primary disease with a control target (`core.f2_negative_eligible`)."""
    from haenv.world.generation.build import clinical_plan
    from haenv.m2.core import f2_negative_eligible, f3_plan_for
    raw, lat = case["raw"], case["latent"]
    arm = "negative" if lat.get("ddx_join_gold") == "independent" else "positive"
    live = list(clinical_plan(raw["disease"], raw["devices"], raw.get("comorbidities") or [], case["case_id"]))
    f3 = all(f3_plan_for(case["case_id"], lat["ddx_spec_id"], arm, live, ex, raw["disease"])
             is not None for ex in (False, True))
    f2 = arm == "positive" or f2_negative_eligible(raw["disease"])
    return tuple(f for f in allowed if (f != "F3" or f3) and (f != "F2" or f2))


def medication_eligible(case: dict) -> bool:
    """Use the clinical producer's actual exposure rule, never infer it from a diagnosis.

    HD-UNI-19 cannot be admitted until the coordinator supplies the hook. Other
    expansion conditions use it when present; legacy cases keep their old policy.
    """
    import haenv.world.clinical.ddx as ddx
    from haenv.m2.pack import EXPANSION_IDS
    sid = (case.get("latent") or {}).get("ddx_spec_id")
    if sid not in EXPANSION_IDS:
        return True
    hook = getattr(ddx, "clinical_medication_eligible", None)
    if hook is None:
        if sid == "HD-UNI-19":
            raise ValueError(f"{sid}: actual statin exposure requires integrated ddx.clinical_medication_eligible")
        return True
    eligible = hook(case)
    if not isinstance(eligible, bool):
        raise ValueError(f"{sid}: clinical_medication_eligible must return bool, got {eligible!r}")
    return eligible


def allocate_expanded(n: int, seed: str = "", *, pool_multiplier: int = 1) -> dict:
    """Bounded seed-generated candidate pool, balanced benchmark strata, no hard quota.

    Every admitted spec receives an equal candidate budget; every expanded spec
    must have an eligible positive patient. Patient identity and frame choice are
    deterministic. Emission/selection later check coverage again.
    """
    import haenv.world.validation.pack_size as PS
    cand = benchmark_candidates()
    if not isinstance(pool_multiplier, int) or isinstance(pool_multiplier, bool) or pool_multiplier < 1:
        raise ValueError("candidate pool multiplier must be a positive integer")
    total = int(math.ceil(1.2 * PS.POOL_FACTOR * PS.pool_n(n) * pool_multiplier))
    # Equal budget inside each clinical stratum, deliberately unrelated to prevalence.
    groups = collections.Counter((v["arm"], v["outside_endocrine"] if v["arm"] == "positive" else None)
                                 for v in cand.values())
    target = {("positive", True): total * 31 / 88, ("positive", False): total * 31 / 88,
              ("negative", None): total * 13 / 44}
    base, rows, counts = PS.variant_base(seed), [], collections.Counter()
    medication_rejected = collections.Counter()
    reserved = set(workup())  # never re-use a frozen patient's identity, including public seed
    for sid, meta in sorted(cand.items()):
        group = (meta["arm"], meta["outside_endocrine"] if meta["arm"] == "positive" else None)
        budget = max(6, int(math.ceil(target[group] / groups[group])))
        free = budget
        frame_counts = collections.Counter()
        top = min(PS.VARIANT_WINDOW, variant_limit(budget))
        try:
            variants = _window(sid, base, top)
        except ValueError as exc:
            raise ValueError(f"expanded variant generation failed {sid}, seed-window base {base}, "
                             f"variant bound {top}: {exc}") from exc
        for c in variants:
            if not free:
                break
            if c["case_id"] in reserved or c["case_id"] in CLINICAL_HOLD:
                continue
            if c["latent"].get("ddx_insufficient") or not demographics_ok(c):
                continue
            if not medication_eligible(c):
                medication_rejected[sid] += 1
                continue
            allowed = eligible_frames(c, meta["eligible_frames"])
            if not allowed:
                continue
            fw = min(allowed, key=lambda f: (frame_counts[f], _h("expanded-frame", c["case_id"], f)))
            _SEEDED[c["case_id"]] = c
            rows.append({"case": c["case_id"], "spec_id": sid, "fw": fw,
                         "frame": fw + ("-neg" if meta["arm"] == "negative" else ""),
                         "source": "generated", "rule": "benchmark strata, equal within stratum",
                         **meta})
            frame_counts[fw] += 1
            counts[sid] += 1
            free -= 1
        if free:
            raise ValueError(f"expanded pool shortfall {sid}: need {budget}, have {budget-free}, variant bound {top}; "
                             f"medication-ineligible {medication_rejected[sid]}")
    return {"rows": sorted(rows, key=lambda r: r["case"]), "candidate_policy": cand,
            "planned": {"basis": "benchmark strata; not disease prevalence", "minimum_pool_n": total},
            "measured": {"pool_n": len(rows), "by_spec": dict(sorted(counts.items())),
                         "medication_ineligible_by_spec": dict(sorted(medication_rejected.items()))}}


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
    from haenv.world.clinical.ddx import ddx_case_specs
    from haenv.world.clinical.ddx import strip_inert_gap
    from haenv.world.overlay import condition_registry
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
    from haenv.scoring.gates import check_demographic_plausibility
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
        # cases of this spec an earlier call took count toward the bound (the negative arm asks one at a time)
        top = variant_limit(k + sum(1 for c in _generated(spec, n - 3) if c["case_id"] in used) - len(out))
        if n > top:
            raise RuntimeError(f"cannot find {k} sufficient variants of {spec} in {top} variants")
    return out


def n_tests(spec: str) -> int:
    """K-NT of a spec: the number of gold tests it declares (spec-level, every variant the same)."""
    if spec not in _NT:
        from haenv.world.clinical.ddx import ddx_case_specs
        _NT[spec] = len(ddx_case_specs(only=[spec], variants=1, **GEN_KW)[0]["latent"].get("ddx_tests") or [])
    return _NT[spec]


_NT: dict[str, int] = {}
#: highest variant number tried for one spec, at least (`variant_limit`)
MAX_VARIANT = 60
#: variant numbers tried per requested case above MAX_VARIANT: about half the variants of the scarcest
#: spec (HD-COM-63) are sufficient and demographically admissible (28 of 55 at N = 100, 113 of 210 at
#: N = 400), so 4 per case leaves room. A flat MAX_VARIANT made every N > 100 pool fail on HD-COM-63.
VARIANTS_PER_CASE = 4


def variant_limit(k: int) -> int:
    """Highest variant number `pick_cases` tries for k cases of one spec (k counts the cases of the
    spec earlier calls already took)."""
    return max(MAX_VARIANT, VARIANTS_PER_CASE * int(k))
#: hard-tier share of the pool, K-NT hard side (as `tools/generation/m2_gen_job.py`), share of the pairs inside
#: an association cluster (composition v2: >= 70 %).
TIER_HARD = 0.30
#: Share of the F1 positives drawn from the components with a registered acute presentation
#: (`registry/acute_events.yaml`, three components); the rest come from the same line-balanced pool as
#: F2/F3 and present with their latest true symptom, so F1 is not five textbook complaints.
F1_ACUTE_SHARE = 0.4
K_NT_HARD_MIN = 6
CLUSTER_MIN = 0.70


def bridge_rows() -> list[dict]:
    """The bridge rows of the pool: the v1.0.1 bridge set of `alloc_cases.csv` (per unified non-red
    spec the variant with the lowest v1.0.1 workup score, plus up to 12 insufficient-tier ones)."""
    return [r for r in csv.DictReader(ALLOC.open(encoding="utf-8")) if r["fw"] == "F0"]


def allocate(scale: float = 1.0) -> dict:
    """The F1-F3 pool rows at `scale` (1.0 = 61 rows, the pool of a 50-item pack)."""
    from tools.generation import gen_composition_v2 as G
    import haenv.world.validation.pack_size as PS
    from haenv.m2 import core
    from haenv.world.overlay import condition_registry
    from haenv.resources.registry import is_rare
    from haenv.resources.registry import load_disease_prevalence
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
    n_f1_all = frame_n["F1"] - neg_n["F1"]
    n_f1 = int(round(F1_ACUTE_SHARE * n_f1_all))
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
    n_f1_pool = n_f1_all - n_f1
    n_slots = frame_n["F2"] + frame_n["F3"] - neg_n["F2"] - neg_n["F3"] + n_f1_pool
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
        if best is None and need_k["pair"]:
            # availability: no pair is admissible any more (caps); the remaining slots take single specs
            need_k["single"] += need_k["pair"]
            need_k["pair"] = 0
            continue
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
    pair = (lambda t: n_tests(t[1]) >= K_NT_HARD_MIN)
    f3 = [t for t in order23 if pair(t) and has_lab_stream(case_of(t[0], t[2], W))][:n_f3]
    f3 += [t for t in order23 if t not in f3 and has_lab_stream(case_of(t[0], t[2], W))][:n_f3 - len(f3)]
    for t in order23:
        if len(f3) >= n_f3:
            break
        if t not in f3:
            f3.append(t)
    # F1 takes the single specs first, so F2/F3 keep the comorbid pairs (the hard-capable specs)
    rest = [t for t in order23 if t not in f3]
    f1 = ([t for t in rest if not pair(t)] + [t for t in rest if pair(t)])[:n_f1_pool]
    for cid, spec, src in order23:
        t = (cid, spec, src)
        add(cid, spec, "F3" if t in f3 else ("F1" if t in f1 else "F2"), "positive", src, "uniform-lines-hard-pairs")
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
    from haenv.world.generation.build import clinical_plan
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
        if r["case"] in _SEEDED:
            out[r["case"]] = copy.deepcopy(_SEEDED[r["case"]])
            continue
        n, top = 3, max(MAX_VARIANT, _variant_no(r["case"]) + 5)
        while True:
            hit = [c for c in _generated(spec, n) if c["case_id"] == r["case"]]
            if hit:
                out[r["case"]] = copy.deepcopy(hit[0])
                break
            n += 3
            if n > top:
                raise KeyError(r["case"])
    return out


def _variant_no(cid: str) -> int:
    """Variant number of a case id (`JD-98v13` -> 13; a frozen id is variant 1)."""
    head, _, v = str(cid).rpartition("v")
    return int(v) if head and v.isdigit() else 1


#: case id -> case, for the variants `instantiate` drew from a seed's window
_SEEDED: dict[str, dict] = {}


def _window(spec: str, base: int, upto: int) -> list[dict]:
    """Variants base+1 .. base+upto of `spec` by the workup's generator call, in calls of the workup's
    chunk size (5 for independent-type specs, else 3)."""
    from haenv.world.clinical.ddx import ddx_case_specs
    from haenv.world.clinical.ddx import strip_inert_gap
    from haenv.world.overlay import condition_registry
    have = _GEN_CACHE.setdefault(f"{spec}@{base}", [])
    chunk = 5 if condition_registry()[spec]["join_gold"] == "independent" else 3
    while len(have) < upto:
        cs = ddx_case_specs(only=[spec], variants=chunk, variant_start=base + 1 + len(have), **GEN_KW)
        strip_inert_gap(cs, None)
        for c in cs:
            c["latent"].pop("distractor_level", None)
            c["latent"]["distractor_level"] = "low"
        have.extend(cs)
    return have[:upto]


def instantiate(rows: list[dict], base: int, W: dict | None = None) -> list[dict]:
    """The F1-F3 pool rows with every case re-drawn from the variant window after `base`
    (`haenv.pack_size.variant_base`): the composition is the table's (spec, frame, arm and rule of
    every row stay), the patient is new -- demographics, base condition, weight, index day, symptom
    days, density and every world draw downstream are keyed by the new case id. A replacement passes
    the allocator's own instance rules: sufficient tier, GEN24 demographics, and a live lab stream
    where the row it replaces (an F3 row) had one. base 0 (the public seed) returns the rows unchanged.
    The drawn cases are kept for `generated_cases`."""
    import haenv.world.validation.pack_size as PS
    if not base:
        return rows
    W = W or workup()
    by_spec: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_spec[r["spec_id"]].append(r)
    new: dict[int, dict] = {}
    for spec, rs in sorted(by_spec.items()):
        need = [r["fw"] == "F3" and has_lab_stream(case_of(r["case"], r["source"], W)) for r in rs]
        free = list(range(len(rs)))
        upto = 0
        while free:
            upto += 6
            if upto > PS.VARIANT_WINDOW:
                raise RuntimeError(f"cannot instantiate {len(rs)} cases of {spec} in the seed's window")
            for c in _window(spec, base, upto)[upto - 6:]:
                if not free:
                    break
                cid = c["case_id"]
                if c["latent"].get("ddx_insufficient") or not demographics_ok(c):
                    continue
                lab = has_lab_stream(c)
                # a case with a lab stream fills a row that needs one first
                j = next((i for i in free if need[i] and lab), None)
                if j is None:
                    j = next((i for i in free if not need[i]), None)
                if j is None:
                    continue
                free.remove(j)
                _SEEDED[cid] = c
                new[id(rs[j])] = {**rs[j], "case": cid, "source": "generated", "instance_of": rs[j]["case"]}
    out = [new[id(r)] for r in rows]
    assert len({r["case"] for r in out}) == len(out)
    return sorted(out, key=lambda r: r["case"])


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
