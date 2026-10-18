"""p4_gen_job.py -- write the pack-4 follow-up job (`inputs/p4-followup.job.yaml`).

Inputs (read only):
  * `inputs/ddx-workup.job.yaml` -- the v1.0.1 tool-track pack (145 cases), copied verbatim
    (`raw` and `latent` unchanged after the YAML round trip) for every case kept here;
  * a variant top-up: independent-type conditions from variant 6 on and the others from variant 4
    on (`--h0-variants` / `--h1-variants`, default ceil(20 N/50) / ceil(8 N/50): creatinine items go
    only to patients with an ACEI indication, so the pool needs more cases), with the workup
    generator's distractor levels (`low:1,high:1`) and insufficient share 0.6.

What is added: one latent key `p4` (provenance class `knob`) per kept case, written by
`haenv.p4.plan.make_plans`: class (balanced inside H), analyte, sign, band position, decoys and the
non-true factor. Kept cases: every eligible H0 case, and the first `--h1` (default ceil(400 N/50))
eligible H1 cases by the seeded `sha256("p4pool", case_id)`. The pool grows with N in proportion
(variant counts and the H1 cap scale by N/50); a cell whose planned supply is below the pack's quota
stops the generator with the gap. The planned comorbidity (CKD, R15) joins the world's comorbidities
(`world_case`), so the labs are drawn for the patient the record describes.

Usage: python tools/generation/p4_gen_job.py [--n N] [--seed S] [--out PATH] [--two-pass] [--check]
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
import math
import copy
import hashlib
import io
import json
import os
import pathlib
import sys
import subprocess
from contextlib import contextmanager

import yaml

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "tools"))

import haenv.provenance.canary as CANARY  # noqa: E402
import haenv.world.validation.pack_size as PS     # noqa: E402

SRC = ROOT / 'benchmarks' / 'inputs' / 'ddx-workup.job.yaml'
OUT = ROOT / 'benchmarks' / 'inputs' / 'p4-followup.job.yaml'
#: pool size of the N = 50 pack: top-up variants (H0, H1) and the H1 cap; scaled by N/50
POOL_50 = {"h0_variants": 31, "h1_variants": 12, "h1": 400}
#: the shared audit's gates read every emitted item: the pool holds at least this many times N
POOL_FACTOR = 4
#: a seed's window adds at most this many variants per condition to reach the pool floor
WINDOW_EXTRA_MAX = 5


def scratch_dir(out: pathlib.Path) -> pathlib.Path:
    """Where the first-pass job and its feasibility cache live (never under inputs/): HAENV_P4_SCRATCH,
    else next to a job written outside inputs/, else derived/p4."""
    if os.environ.get("HAENV_P4_SCRATCH"):
        return pathlib.Path(os.environ["HAENV_P4_SCRATCH"])
    if out.resolve() != OUT.resolve():
        return out.resolve().parent / "p4-gen"
    if os.environ.get("HAENV_OUTPUT_ROOT"):
        return pathlib.Path(os.environ["HAENV_OUTPUT_ROOT"]) / "p4-gen"
    return ROOT / "derived" / "p4"


def pool_size(n: int) -> dict:
    return {k: math.ceil(v * n / PS.DEFAULT_N) for k, v in POOL_50.items()}


def _topup(h0_variants: int, h1_variants: int) -> list[dict]:
    """More variants of the registry's conditions, made by the workup's own generator call:
    independent-type (H0) conditions from variant 6 on, the others (H1) from variant 4 on (the
    workup carries v1-v5 and v1-v3 of them). Distractor levels as the workup (`low:1,high:1`),
    insufficient share 0.6, no rhythm gap (this pack is not sliced)."""
    from haenv.world.clinical.ddx import ddx_case_specs
    from haenv.world.clinical.ddx import strip_inert_gap
    from haenv.world.overlay import condition_registry
    from tools.generation.gen_core_job import apply_distractor_level, assign_distractor_levels
    reg = condition_registry(include_draft=False)
    ind = [k for k in reg if reg[k]["join_gold"] == "independent"]
    oth = [k for k in reg if reg[k]["join_gold"] != "independent"]
    new = []
    if h0_variants:
        new += ddx_case_specs(only=ind, variants=h0_variants, variant_start=6, density="mixed", insufficient_frac=0.6)
    if h1_variants:
        new += ddx_case_specs(only=oth, variants=h1_variants, variant_start=4, density="mixed", insufficient_frac=0.6)
    lvl = assign_distractor_levels(new, "low:1,high:1")
    for c in new:
        apply_distractor_level(c, lvl[str(c["case_id"])])
    strip_inert_gap(new, "")
    return new


def _window_supply(base: int, workup: list[dict], h0_variants: int, h1_variants: int) -> list[dict]:
    """A non-public seed's patients (W6): per registry condition, as many variants as the public pool
    supplies (its ddx-workup copies plus its top-up variants), numbered from the seed's window
    `base + 1` (`haenv.pack_size.variant_base`) by the workup generator's call, with the
    insufficient tier paired inside the window, the top-up's distractor levels (`low:1,high:1`) and
    no rhythm gap. Every per-case world draw is keyed by the case id, so the patients are new while
    the per-condition supply, and so the composition the plans deal, stays the public pool's."""
    from haenv.world.clinical.ddx import ddx_case_specs
    from haenv.world.clinical.ddx import strip_inert_gap
    from haenv.world.overlay import condition_registry
    from tools.generation.gen_core_job import apply_distractor_level, assign_distractor_levels
    reg = condition_registry(include_draft=False)
    have = collections.Counter(str(c["latent"].get("ddx_spec_id")) for c in workup)
    new = []
    for spec, row in reg.items():
        k = have[spec] + (h0_variants if row["join_gold"] == "independent" else h1_variants)
        if k:
            new += ddx_case_specs(only=[spec], variants=k, variant_start=base + 1, window_base=base,
                                  density="mixed", insufficient_frac=0.6)
    lvl = assign_distractor_levels(new, "low:1,high:1")
    for c in new:
        apply_distractor_level(c, lvl[str(c["case_id"])])
    strip_inert_gap(new, "")
    return new


def with_pack_age(case: dict, *, condition_specs: dict | None = None) -> dict:
    """The case with the age band the build prints for a pack patient (`build.pack_age_of`), so a
    generation rule that reads the age (a drug's age range, the pregnancy-capable avoid list) reads it."""
    from haenv.world.generation.build import pack_age_of
    band = pack_age_of(str(case["case_id"]), case.get("raw") or {}, case.get("latent") or {},
                       **({"condition_specs": condition_specs} if condition_specs is not None else {}))
    if band == (case.get("raw") or {}).get("age_range"):
        return case
    return {**case, "raw": {**case["raw"], "age_range": band}}


def world_case(case: dict, plan: dict, *, condition_specs: dict | None = None) -> dict:
    """The case with the comorbidity the item puts on the record (R15, `hooks.observe_followup`) among the
    world's comorbidities and the age band the build prints for them, so the world's labs are drawn for the
    patient the record describes."""
    c = copy.deepcopy(case)
    raw = c["raw"]
    cm = plan.get("comorbidity")
    if cm and cm not in {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or ()}:
        raw["comorbidities"] = list(raw.get("comorbidities") or ()) + [cm]
    c["latent"]["p4"] = plan
    return with_pack_age(c, condition_specs=condition_specs)


def _world_streams(case: dict) -> set[str]:
    from haenv.world.generation.build import clinical_plan
    raw = case.get("raw") or {}
    return set(clinical_plan(str(raw.get("disease") or ""), list(raw.get("devices") or []),
                             list(raw.get("comorbidities") or []), str(case["case_id"])))


def feasibility(job_text: str, scratch: pathlib.Path) -> dict[str, list[list]]:
    """Build the first-pass job in process (deterministic generator, no model call) and record,
    for every case, which (analyte, sign) pairs its world can carry for its dealt class."""
    import logging
    logging.disable(logging.CRITICAL)
    scratch.mkdir(parents=True, exist_ok=True)
    jp = scratch / "p4-followup.pass1.job.yaml"
    jp.write_text(job_text, encoding="utf-8")
    import haenv.world.generation.build as B
    from haenv.app.job import load_job
    from haenv.p4 import hooks
    from haenv.p4.plan import signs_of
    from haenv.p4.realize import Unrealizable, _realize_one
    job = load_job(str(jp), root=ROOT)
    out: dict[str, list[list]] = {}
    orig = hooks.observe_followup

    def probe(raw, cid, truth, audit):
        from haenv.p4.gate import plan_of
        plan = plan_of(raw)
        if plan is not None:
            ok = []
            for an in plan.get("analytes_ok") or ():
                for sg in signs_of(an):
                    try:
                        _realize_one(raw, str(cid), {**plan, "analyte": an, "sign": sg,
                                                     "decoys": (plan.get("decoys") or {}).get(an) or [],
                                                     "factor": (plan.get("factor") or {}).get(an)},
                                     B._LAB_CTX.get(str(cid)))
                        ok.append([an, sg])
                    except Unrealizable:
                        pass
            out[str(cid)] = ok
        return orig(raw, cid, truth, audit)
    hooks.observe_followup = probe
    try:
        for cs in job.cases:
            try:
                B.build_case(cs, max_rounds=6)
            except Exception as e:                      # noqa: BLE001  (a case that fails to build carries nothing)
                out.setdefault(cs.case_id, [])
                print(f"[p4] pass 1: {cs.case_id} failed to build: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        hooks.observe_followup = orig
        logging.disable(logging.NOTSET)
    return out


def _fresh_patient(args):
    from haenv.world.generation.patient_sampling import fresh_patient
    case, seed = args
    return fresh_patient(case, "p4", seed)


def prepare_patients(cases: list[dict], seed: str, workers: int = 1) -> list[dict]:
    """Map the original independent draws in input order, with identical seed keys."""
    inputs = [(case, seed) for case in cases]
    if workers <= 1:
        return [_fresh_patient(args) for args in inputs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_fresh_patient, inputs, chunksize=8))


@contextmanager
def fixed_generation_root():
    """Resolve the fixed CLI resource root once; retain live registry overrides."""
    import haenv
    original = haenv.data_root
    root, declared, cwd = original(), os.environ.get("HAENV_DATA_ROOT"), os.getcwd()

    def resolved():
        if os.environ.get("HAENV_DATA_ROOT") != declared or os.getcwd() != cwd:
            raise RuntimeError("generation resource root or cwd changed during a fixed build")
        return root

    haenv.data_root = resolved
    try:
        yield root
    finally:
        haenv.data_root = original
        resolved()


@fixed_generation_root()
def build(h1: int, two_pass: bool = False, h0_variants: int = 6, h1_variants: int = 2, n: int = PS.DEFAULT_N,
          seed: str = PS.PUBLIC_SEED, scratch: pathlib.Path | None = None, window_extra: int = 0) -> tuple[str, dict]:
    from haenv.world.overlay import condition_registry
    from haenv.p4.audit import planned_cells
    from haenv.p4.plan import eligible_analytes, h01, hidden, make_plans
    src = yaml.load(SRC.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
    cases = [copy.deepcopy(c) for c in src["cases"]]
    base = PS.variant_base(seed)
    if base:
        # W6: a non-public seed draws every patient from its variant window, same per-condition supply
        top = _window_supply(base, cases, h0_variants, h1_variants)
        cases = []
    else:
        top = _topup(h0_variants, h1_variants)
    have = {str(c["case_id"]) for c in cases}
    clash = sorted(str(c["case_id"]) for c in top if str(c["case_id"]) in have)
    if clash:
        raise SystemExit(f"top-up case ids collide with the workup: {clash[:5]}")
    specs = condition_registry(include_draft=True)
    patients = prepare_patients(cases + top, seed, int(os.environ.get("HAENV_GEN_WORKERS", "1")))
    allc = [with_pack_age(c, condition_specs=specs) for c in patients]
    ws = {str(c["case_id"]): _world_streams(c) for c in allc}
    elig = {str(c["case_id"]): eligible_analytes(c, specs.get(str(c["latent"].get("ddx_spec_id"))), ws[str(c["case_id"])])
            for c in allc}
    H0 = [c for c in allc if not hidden(c["latent"]) and elig[str(c["case_id"])]]
    H1 = sorted([c for c in allc if hidden(c["latent"]) and elig[str(c["case_id"])]],
                key=lambda c: h01(PS.seeded("p4pool", seed), c["case_id"]))[:h1]
    pool = H0 + H1
    plans = make_plans(pool, specs, ws)
    feas = None
    if two_pass:
        pass1 = _render(allc, plans, top, H1, h1, None, (h0_variants, h1_variants), n, seed)[0]
        from haenv.p4.hooks import p4_world_fingerprint
        key = hashlib.sha256((pass1 + "|" + p4_world_fingerprint()).encode()).hexdigest()[:16]
        scratch = scratch or scratch_dir(OUT)
        cache = scratch / "feasibility.json"
        cached = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
        if cached.get("key") == key:
            feas = cached["feasible"]
        else:
            feas = feasibility(pass1, scratch)
            cache.write_text(json.dumps({"key": key, "feasible": feas}, ensure_ascii=False, sort_keys=True, indent=0),
                             encoding="utf-8")
        plans = make_plans(pool, specs, ws, feasible=feas)
    PS.short(planned_cells(PS.pool_n(n)), collections.Counter((p["H"], p["class"], p.get("comorbidity")) for p in plans.values()),
             "p4 planned cases")
    txt, kept = _render(allc, plans, top, H1, h1, feas, (h0_variants, h1_variants), n, seed, window=bool(base))
    if len(kept) < POOL_FACTOR * PS.pool_n(n):
        # A seed's window draws new patients, and how many of them can carry a follow-up item
        # varies; one more variant per condition until the floor holds (the public pool never retries).
        if base and window_extra < WINDOW_EXTRA_MAX:
            print(f"[p4] window pool {len(kept)} < {POOL_FACTOR} x {PS.pool_n(n)}: one more variant per condition",
                  file=sys.stderr)
            return build(h1, two_pass, h0_variants + 1, h1_variants + 1, n, seed, scratch, window_extra + 1)
        raise SystemExit(f"[p4] pool of {len(kept)} planned cases is below {POOL_FACTOR} x {PS.pool_n(n)}")
    stats = {"workup": len(cases), "topup": len(top), "window_extra_variants": window_extra, "eligible_H0": len(H0),
             "eligible_H1": sum(1 for c in allc if hidden(c["latent"]) and elig[str(c["case_id"])]),
             "kept": len(kept), "pass1_feasible_none": sorted(k for k, v in (feas or {}).items() if not v),
             "ineligible": sorted(k for k, v in elig.items() if not v),
             "cells": dict(collections.Counter((p["H"], p["class"]) for p in plans.values())),
             "pair_by_class": dict(collections.Counter((p["class"], p["analyte"], p["sign"]) for p in plans.values()))}
    return txt, stats


def _render(allc, plans, top, H1, h1, feas, variants=(6, 2), n=PS.DEFAULT_N, seed=PS.PUBLIC_SEED,
            window: bool = False) -> tuple[str, list]:
    from haenv.world.overlay import condition_registry
    specs = condition_registry(include_draft=True)
    kept = [world_case(c, plans[str(c["case_id"])], condition_specs=specs) for c in allc if str(c["case_id"]) in plans]
    head = CANARY.block().splitlines()
    head += [
        "# Generated file -- do not edit by hand. Rebuild: python tools/generation/p4_gen_job.py",
        "#",
        "# Pack 4 -- follow-up interpretation. Cases: the eligible",
        f"# H0 cases of ddx-workup and of a {len(top)}-case variant top-up, and {len(H1)} eligible H1 cases",
        "# (first by the seeded sha256('p4pool', id)); workup raw/latent copied verbatim (age band as built), plus one",
        "# latent key `p4` (knob) per case: class dealt by hash inside H (sex blocks first); "
        + ("(analyte, sign) dealt inside each class among the pairs a deterministic first-pass build can carry."
           if feas is not None else "sign alternated inside (analyte, class), one pass; a pre-analytic item rises "
           "when today's draw is non-fasting, else falls."),
        "# The plugin group `haenv.p4` (haenv/p4/plugin.py) realizes",
        "# the follow-up item after the world's lab measurement layer and judges it by code.",
        "#",
        "# SYNTHETIC data, evaluation only, not medical advice.",
    ]
    body = {"job_id": "p4-followup", "pack": PS.header(n, seed), "task_type": "joint_dx", "multiround": False,
            "include_baseline": False, "models": [], "sample_cases": 5, "report": "eval-p4-followup.md",
            "gated": False, "probe_id": "p4.followup", "plugins": {"judges": ["haenv.p4"]},
            "_provenance": {"generator": 'tools/generation/p4_gen_job.py', "source_job": 'benchmarks/inputs/ddx-workup.job.yaml',
                            "source_sha16": hashlib.sha256(SRC.read_bytes()).hexdigest()[:16],
                            "topup": f"ddx_case_specs(only=independent, variants={variants[0]}, variant_start=6) + "
                                     f"ddx_case_specs(only=other, variants={variants[1]}, variant_start=4); density=mixed, "
                                     "insufficient_frac=0.6, distractor low:1,high:1",
                            "h1_pool": h1,
                            "feasibility": ("two-pass: (analyte, sign) dealt inside each class among the pairs "
                                            "a deterministic first-pass build can carry" if feas is not None else "one-pass"),
                            "feasibility_sha16": (hashlib.sha256(json.dumps(feas, sort_keys=True).encode()).hexdigest()[:16]
                                                  if feas is not None else None)},
            "cases": kept}
    if window:
        body["_provenance"]["instances"] = (
            "seed window (haenv.pack_size.variant_base): per condition the public pool's supply (workup copies "
            "+ top-up variants) drawn as variants base+1.. by ddx_case_specs(window_base=base); no workup copy")
    buf = io.StringIO()
    yaml.dump(body, buf, Dumper=yaml.CSafeDumper, allow_unicode=True, sort_keys=False, width=120)
    return "\n".join(head) + "\n" + buf.getvalue(), kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PS.DEFAULT_N, help="items in the pack")
    ap.add_argument("--seed", default=None, help="selection seed (else HAENV_PACK_SEED, else the public seed)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--h1", type=int, default=None, help="H1 cases kept (default ceil(400 N/50))")
    ap.add_argument("--h0-variants", type=int, default=None, help="default ceil(20 N/50)")
    ap.add_argument("--h1-variants", type=int, default=None, help="default ceil(8 N/50)")
    ap.add_argument("--two-pass", action="store_true", help="deal (analyte, sign) among the pairs a first-pass build can carry")
    ap.add_argument("--one-pass", action="store_true", help="the default: (analyte, sign) dealt blind")
    a = ap.parse_args()
    seed = PS.cli_seed(a.seed)
    out = pathlib.Path(a.out)
    # the pool is composed for `PS.pool_n(N)`, so the packs of the nested sizes share it (`haenv/world/validation/pack_size.py`)
    size = {k: (getattr(a, k) if getattr(a, k) is not None else v) for k, v in pool_size(PS.pool_n(a.n)).items()}
    def generate(supply):
        return build(supply["h1"], two_pass=a.two_pass and not a.one_pass, h0_variants=supply["h0_variants"],
                     h1_variants=supply["h1_variants"], n=a.n, seed=seed, scratch=scratch_dir(out))
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled():
        from haenv.world.validation.invariants import BRANCHES
        from haenv.p4.supply import cover_branches
        from haenv.p4.audit import planned_directions
        required = ["|".join(branch) for branch in BRANCHES["p4"]]
        txt, st = cover_branches(generate, lambda text, multiplier: branch_preview(text, scratch_dir(out), multiplier, include_cells=True,
                                required_cells={str(k): v for k, v in planned_directions(PS.pool_n(a.n)).items()}),
                                size, required, minimum_emitted=POOL_FACTOR * PS.pool_n(a.n),
                                required_cells={str(k): v for k, v in planned_directions(PS.pool_n(a.n)).items()},
                                max_multiplier=256)
        # The prospective production preview is the candidate admission gate.
        # Keep its successful candidates with their original plans and draws;
        # failed candidates remain in every complete preview on disk.
        attempt = scratch_dir(out) / f"branch-supply-{st['branch_supply'][-1]['multiplier']}"
        ids = set(json.loads((attempt / "emitted_ids.json").read_text()))
        document = yaml.load(txt, Loader=yaml.CSafeLoader)
        document["cases"] = [c for c in document["cases"] if c["case_id"] in ids]
        if len(document["cases"]) != len(ids):
            raise ValueError("admission preview contains unknown candidate identities")
        document["_provenance"]["candidate_admission"] = "actual production emission gates; all failed candidates retained in branch-supply previews"
        st["admitted"] = len(ids)
        txt = CANARY.block() + "\n" + yaml.dump(document, Dumper=yaml.CSafeDumper, allow_unicode=True, sort_keys=False, width=120)
    else:
        txt, st = generate(size)
    if a.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == txt
        print(out.name, "up to date" if same else "DIFFERS")
        return 0 if same else 1
    out.write_text(txt, encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in st.items()},
                     ensure_ascii=False, indent=1))
    return 0


def numerical_rejection(case) -> str | None:
    """Necessary measurement conditions, before the costly full patient build.

    These failures precede rendering and depend only on the original
    CID/plan/time and (for CKD) the already fixed sex/age. No provisional
    medication or clinical rejection is used to exclude a candidate.
    """
    from types import SimpleNamespace
    from haenv.app.job import raw_field
    from haenv.p4.diversity import realize
    from haenv.p4.realize import Unrealizable
    raw = SimpleNamespace(prediction_context={"prediction_time_T": case.index_time_T},
                          user_profile={k: raw_field(case.raw, k) for k in ("sex", "age_range")},
                          latent_premise={"patient_basics": case.raw}, longitudinal_data={}, adjudication={})
    try:
        realize(raw, case.case_id, case.latent["p4"], {})
    except Unrealizable as error:
        if str(error) in (
            "independent observations do not yield the planned single source",
            "non-noise apparent change does not exceed paired RCV",
            "archived comparison does not distinguish the earlier clinical episode",
            "course direction is not supported by the actual comparison interval",
            "clinical event does not fall between the actual comparison reports",
        ):
            return str(error)
    return None


def preview_key(text: str, environment=None) -> str:
    """Exact-input cache for admission counts only; G is the engineering caller.

    All other tracked generation, registry and input dependencies are bound by
    bytes. Worker count and output locations cannot change clinical draws.
    """
    env = os.environ if environment is None else environment
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    digest = hashlib.sha256(text.encode())
    for name in files:
        if name == 'tools/generation/p4_gen_job.py' or name.startswith(("tests/", "docs/", "paper/")) or not name.endswith((".py", ".yaml", ".yml", ".json", ".csv", ".toml")):
            continue
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    ignored = {"HAENV_OUTPUT_ROOT", "HAENV_P4_SCRATCH", "HAENV_GEN_WORKERS", "HAENV_P4_PREVIEW_CACHE"}
    settings = {k: v for k, v in env.items() if k.startswith("HAENV_") and k not in ignored}
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for path in (ROOT / "config.yaml", ROOT / "config.local.yaml", *([pathlib.Path(env["HAENV_CONFIG_OVERLAY"])] if env.get("HAENV_CONFIG_OVERLAY") else [])):
        if path.is_file():
            digest.update(str(path).encode() + path.read_bytes())
    return digest.hexdigest()


def preview_counts(cases_file: pathlib.Path) -> tuple[dict, dict, list]:
    counts, cells, ids = collections.Counter(), collections.Counter(), []
    for line in cases_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gold = row["case"].get("adjudication", {}).get("followup_p4", {})
        if gold.get("next_step"):
            counts[f'{gold["class"]}|{gold["next_step"]}'] += 1
            cell = (int(gold["plan"]["H"]), gold["class"], (gold.get("comorbidity") or {}).get("code"), int(gold["sign"]))
            cells[str(cell)] += 1
            ids.append(row["case_id"])
    return dict(counts), dict(cells), ids


def cache_receipt(cases_file: pathlib.Path) -> dict:
    audit = cases_file.with_name("audit.jsonl")
    return {"cases": str(cases_file.resolve()), "cases_sha256": hashlib.sha256(cases_file.read_bytes()).hexdigest(),
            "audit": str(audit.resolve()), "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest()}


def cached_preview(cache: pathlib.Path, key: str):
    path = cache / (key + ".json")
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    for name in ("cases", "audit"):
        if hashlib.sha256(pathlib.Path(receipt[name]).read_bytes()).hexdigest() != receipt[name + "_sha256"]:
            raise ValueError("candidate preview cache output changed: " + receipt[name])
    # Never trust cached count or identity sidecars: derive them from the
    # original complete production output, preserving its original world stamp.
    return preview_counts(pathlib.Path(receipt["cases"])), receipt


def load_preview_job(path: pathlib.Path, text: str):
    """Use the C-safe parse as input to the unchanged complete job validator."""
    import haenv.app.job as J
    if path.read_text(encoding="utf-8") != text:
        raise ValueError("preview input text does not match its job file")
    document = yaml.load(text, Loader=yaml.CSafeLoader)
    original, target = J._cached_yaml, path.resolve()

    def parsed(requested):
        return copy.deepcopy(document) if pathlib.Path(requested).resolve() == target else original(requested)

    J._cached_yaml = parsed
    try:
        return J.load_job(str(path), root=ROOT), document
    finally:
        J._cached_yaml = original


@fixed_generation_root()
def branch_preview(text: str, scratch: pathlib.Path, multiplier: int, *, include_cells=False, required_cells=None) -> dict:
    """Build the entire prospective pool, before selection or surface audits.

    Only existing clinical gold branch coverage decides whether supply grows.
    Preview jobs, batches and logs remain available, including rejected pools.
    """
    from haenv.app.settings import SET_ENV
    attempt = scratch / f"branch-supply-{multiplier}"
    attempt.mkdir(parents=True, exist_ok=True)
    job = attempt / "p4-followup.job.yaml"
    job.write_text(text, encoding="utf-8")
    cache = pathlib.Path(os.environ["HAENV_P4_PREVIEW_CACHE"]) if os.environ.get("HAENV_P4_PREVIEW_CACHE") else None
    key = preview_key(text) if cache else None
    hit = cached_preview(cache, key) if cache else None
    if hit is not None:
        (counts, cells, ids), receipt = hit
        (attempt / "preview_cache.json").write_text(json.dumps({"key": key, **receipt}, indent=1) + "\n", encoding="utf-8")
        for name, value in (("branches", counts), ("cells", cells), ("emitted_ids", ids)):
            (attempt / (name + ".json")).write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
        return {"branches": counts, "cells": cells} if include_cells else counts
    loaded, document = load_preview_job(job, text)
    candidates = loaded.cases
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=int(os.environ.get("HAENV_GEN_WORKERS", "1"))) as pool:
        reasons = list(pool.map(numerical_rejection, candidates, chunksize=8))
    rejected = {c.case_id: why for c, why in zip(candidates, reasons) if why is not None}
    (attempt / "numerical_rejections.json").write_text(json.dumps(rejected, indent=1) + "\n", encoding="utf-8")
    document["cases"] = [c for c in document["cases"] if c["case_id"] not in rejected]
    # These are upper bounds, never claimed as actual emission counts. A
    # deficient bound proves this whole supply cannot satisfy the original
    # quota even if every remaining candidate passes the full world build.
    potential = collections.Counter(str((int(c["latent"]["p4"]["H"]), c["latent"]["p4"]["class"],
                                         c["latent"]["p4"].get("comorbidity"), int(c["latent"]["p4"]["sign"])))
                                    for c in document["cases"])
    missing = {k: n - potential.get(k, 0) for k, n in (required_cells or {}).items() if potential.get(k, 0) < n}
    if missing:
        (attempt / "necessary_supply.json").write_text(json.dumps({"status": "not_built: necessary upper bound below original quota",
            "potential_cells": potential, "missing_cells": missing}, indent=1) + "\n", encoding="utf-8")
        return {"branches": {}, "cells": {}} if include_cells else {}
    build_job = attempt / "numeric-candidates" / "p4-followup.job.yaml"
    build_job.parent.mkdir(parents=True, exist_ok=True)
    build_job.write_text(CANARY.block() + "\n" + yaml.dump(document, Dumper=yaml.CSafeDumper, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    sets = [x for item in ["env_file=/dev/null", *json.loads(os.environ.get(SET_ENV, "[]"))]
            for x in ("--set", item)]
    cmd = [sys.executable, "-m", "haenv", "build", str(build_job), "--gen", "deterministic", "--fresh",
           "--gen-workers", os.environ.get("HAENV_GEN_WORKERS", "32"), *sets]
    with (attempt / "build.log").open("w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "HAENV_OUTPUT_ROOT": str(attempt)},
                            stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"candidate preview failed ({rc}): {attempt}")
    files = sorted((attempt / "results").glob("*/p4-followup/*/cases.jsonl"))
    if not files:
        raise RuntimeError(f"candidate preview produced no cases: {attempt}")
    counts, cells, ids = preview_counts(files[-1])
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / (key + ".json")).write_text(json.dumps(cache_receipt(files[-1]), indent=1) + "\n", encoding="utf-8")
    (attempt / "branches.json").write_text(json.dumps(counts, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (attempt / "cells.json").write_text(json.dumps(cells, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (attempt / "emitted_ids.json").write_text(json.dumps(ids, indent=1) + "\n", encoding="utf-8")
    return {"branches": dict(counts), "cells": dict(cells)} if include_cells else dict(counts)

if __name__ == "__main__":
    raise SystemExit(main())
