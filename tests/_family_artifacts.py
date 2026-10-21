"""Real component worlds and their production-prepared twin declarations."""
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import pickle

import haenv.world.generation.twins as twins
ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {}

def remember_question(raw):
    import haenv.world.wq as wq
    MANIFESTS[twins.sha16(asdict(raw))] = copy.deepcopy(wq.injected_manifest(raw.case_id))

def restore_question(raw):
    import haenv.world.wq as wq
    from haenv.world.generation.opaque_ids import register_manifest_alias
    from haenv.world.generation.opaque_ids import stored_solver_id
    wq.register_injection(raw.case_id, MANIFESTS[twins.sha16(asdict(raw))])
    register_manifest_alias(raw.case_id, stored_solver_id(raw))
    return None

def _sources():
    from haenv.app.job import load_job
    from haenv.world.generation.build import build_case
    # Loading jobs registers exactly the production plugins and gates.
    result = {}
    for pack, name in (("p3", "p3-meds"), ("p4", "p4-followup")):
        job = load_job(str(ROOT / 'benchmarks' / 'inputs' / f'{name}.job.yaml'))
        found = {}
        for cs in job.cases:
            plan = cs.latent[pack]
            cls = plan["class"]
            if cls in found or cls not in ({"downtitrate", "maintain"} if pack == "p3" else {"preanalytical", "analytic_biological_noise"}):
                continue
            raw, _ = build_case(cs)
            if raw is not None:
                found[cls] = raw
                remember_question(raw)
            if len(found) == 2:
                break
        assert len(found) == 2
        result[pack] = list(found.values())
    # P2 uses the real measurement hook, which installs its production gold. Its emission gate
    # (`p2.gate.record_vs_world`) refuses a record that lists a condition the world did not draw the
    # labs from, as the P2 generator never writes one (`p2_gen_job.world_case`); so the P2 questions
    # are grafted onto a built case whose record and world agree.
    from haenv.p2.hooks import observe_triage
    from haenv.p2.gate import record_vs_world
    from haenv.contracts.external_gold import SLOT
    base = next((r for r in result["p4"] if not record_vs_world({}, r)), None)
    for cs in ([] if base else load_job(str(ROOT / 'benchmarks' / 'inputs' / 'p4-followup.job.yaml')).cases):
        raw, _ = build_case(cs)
        if raw is not None and not record_vs_world({}, raw):
            base = raw
            remember_question(raw)
            break
    assert base is not None
    p2 = []
    for tier in ("inf_n14", "ben_spo2"):
        plan = {"tier": tier, "stratum": "plain", "wording": "neutral", "beta_blocker": False}
        from haenv.p2.rules import table
        plan.update(scenario=table()["tiers"][tier]["scenario"], floor=table()["tiers"][tier]["floor"])
        raw = copy.deepcopy(base)
        raw.latent_premise["meta"][SLOT].pop("p4", None)
        raw.latent_premise["meta"][SLOT]["p2"] = plan
        raw.adjudication.pop("followup_p4", None)
        raw.prediction_context.pop("followup_p4", None)
        observe_triage(raw, raw.case_id, {}, {})
        remember_question(raw)
        p2.append(raw)
    result["p2"] = p2
    return result


def _trials(args):
    raw, pack, manifest = args
    import haenv.world.wq as wq
    from haenv.world.generation.opaque_ids import register_manifest_alias, stored_solver_id
    wq.register_injection(raw.case_id, manifest)
    register_manifest_alias(raw.case_id, stored_solver_id(raw))
    return pack, twins.family_candidates(raw, pack)


def family_records(cache):
    def build(dest):
        worlds = _sources()
        args = [(raw, pack, MANIFESTS[twins.sha16(asdict(raw))]) for pack, rows in worlds.items() for raw in rows]
        trials = {pack: [] for pack in worlds}
        with ProcessPoolExecutor(max_workers=64) as pool:
            for pack, candidates in pool.map(_trials, args):
                trials[pack].extend(candidates)
        (dest / "value.pickle").write_bytes(pickle.dumps((worlds, MANIFESTS, trials), protocol=5))
    dest = cache.directory("component-twin-families", {}, build)
    return pickle.loads((dest / "value.pickle").read_bytes())
