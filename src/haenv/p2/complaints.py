"""P2-local, sourced complaint sampling before the ordinary clinical pipeline.

The context is per build/thread. Other packs retain their pool and draw order.
No gold, timing, event count, genuine diagnostic symptom or scoring rule changes.
"""
from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from dataclasses import replace
from functools import lru_cache, wraps
from types import FunctionType

_ACTIVE = ContextVar("p2_complaint_sampling", default=False)


@lru_cache(maxsize=1)
def extra_pool():
    import haenv.world.observations.events as E
    from haenv.resources.regpath import registry_path
    from haenv.resources.yamlcache import load_yaml
    parents = {e["topic"]: e for e in E.BENIGN_EVENTS}
    data = load_yaml(registry_path("p2_complaint_pool.yaml"))
    rows = []
    for r in data["complaints"]:
        parent = parents[r["profile"]]
        # Borrow only the established relevance tags. These are ordinary
        # adult complaints, not the parent's injury, effort or age restriction;
        # their text carries the stated circumstances, with no invented context.
        rows.append({"topic": r["topic"], "text": r["text"], "context": "",
                     "tags": parent["tags"], "exertion": False, "age_w": (1., 1., 1.), "cond": {}})
    return tuple(rows)


def vocabulary():
    """Canonical topics, including every authored lay form, for clone controls."""
    from haenv.resources.regpath import registry_path
    from haenv.resources.yamlcache import load_yaml
    return {text: r["topic"] for r in load_yaml(registry_path("p2_complaint_pool.yaml"))["complaints"]
            for text in [r["text"], *r["lay"]]}


@lru_cache(maxsize=1)
def prior():
    from haenv.resources.regpath import registry_path
    from haenv.resources.yamlcache import load_yaml
    data = load_yaml(registry_path("p2_complaint_prior.yaml"))
    counts = Counter(data["topics"].values())
    weights = {topic: data["families"][str(family)]["probability"] / counts[family]
               for topic, family in data["topics"].items()}
    if not weights or any(w <= 0 for w in weights.values()):
        raise ValueError("P2 complaint prior needs positive sourced family weights")
    return weights


def install():
    import haenv.world.observations.events as E
    import haenv.world.history.history_dependency as H
    if getattr(E._event_ok, "_p2_complaints", False):
        return
    event_ok, order = E._event_ok, E.weighted_order
    benign_topics = frozenset(e["topic"] for e in E.event_pools()[0])

    @wraps(event_ok)
    def eligible(item, facts, banned, proxy_exempt=False, cond=None):
        if _ACTIVE.get() and item.get("topic") in benign_topics and item["topic"] not in prior():
            return False, "p2_complaint_family_not_selected"
        return event_ok(item, facts, banned, proxy_exempt, cond)

    @wraps(order)
    def weighted(pool, facts, kind, cond=None):
        if _ACTIVE.get():
            weights = prior()
            pool = [{**e, "age_w": [w * weights[e["topic"]] for w in e["age_w"]]}
                    if e.get("topic") in weights else e for e in pool]
        return order(pool, facts, kind, cond)

    eligible._p2_complaints = True
    E._event_ok, E.weighted_order = eligible, weighted
    # Ordinary callers retain the original pool. During a P2 build both the
    # planner and verifier see the same extra complaint profiles. Clone the
    # planner's globals for this call; never mutate the shared pool constants.
    pools, plan, lay, topic = E.event_pools, E.plan_benign_evs, E.symptom_lay, E.symptom_topic
    effects = E.side_effect_topics

    @wraps(pools)
    def source_pools():
        benign, life = pools()
        return (benign + extra_pool(), life) if _ACTIVE.get() else (benign, life)

    @wraps(plan)
    def source_plan(*args, **kwargs):
        if not _ACTIVE.get():
            return plan(*args, **kwargs)
        namespace = {**E.__dict__, "BENIGN_EVENTS": source_pools()[0]}
        local = FunctionType(plan.__code__, namespace, plan.__name__, plan.__defaults__, plan.__closure__)
        local.__kwdefaults__ = plan.__kwdefaults__
        return local(*args, **kwargs)

    @wraps(lay)
    def source_lay(sex=None):
        result = lay(sex)
        if _ACTIVE.get():
            from haenv.resources.regpath import registry_path
            from haenv.resources.yamlcache import load_yaml
            result.update({r["text"]: r["lay"] for r in load_yaml(registry_path("p2_complaint_pool.yaml"))["complaints"]})
        return result

    @wraps(topic)
    def source_topic(text):
        return vocabulary().get(text) or topic(text) if _ACTIVE.get() else topic(text)

    @wraps(effects)
    def source_effects(drugs):
        hit = effects(drugs)
        if not _ACTIVE.get():
            return hit
        from haenv.resources.regpath import registry_path
        from haenv.resources.yamlcache import load_yaml
        return hit | frozenset(r["topic"] for r in load_yaml(registry_path("p2_complaint_pool.yaml"))["complaints"]
                               if r["profile"] in hit)

    E.event_pools, E.plan_benign_evs, E.symptom_lay, E.symptom_topic = source_pools, source_plan, source_lay, source_topic
    E.side_effect_topics = source_effects
    import haenv.scoring.verify as verify
    import haenv.world.validation.invariants as invariants
    import haenv.world.validation.shared_audit as shared_audit
    verify._event_pools = source_pools
    benign_set = invariants.benign_set

    @wraps(benign_set)
    def source_benign_set(item):
        canonical = benign_set(item)
        if item["pack"] != "p2":
            return canonical
        vocab = vocabulary()
        return canonical | frozenset(vocab[e["symptom"]] for e in item["sp"].get("evidence_ledger") or ()
                                      if e.get("symptom") in vocab)

    invariants.benign_set = source_benign_set
    sentences, effect_check = shared_audit.said_sentences, shared_audit.side_effect_complaint

    @wraps(sentences)
    def source_sentences():
        return sentences() | set(vocabulary())

    @wraps(effect_check)
    def source_effect_check(item):
        bad = effect_check(item)
        if "triage_p2" not in (item["sp"].get("prediction_context") or {}):
            return bad
        import json
        from haenv.resources.regpath import registry_path
        from haenv.resources.yamlcache import load_yaml
        blob = json.dumps(item["sp"], ensure_ascii=False, default=str).lower()
        drugs = [d for d in (E._cached_yaml(E._rp("drug_side_effects.yaml")) or {}).get("drugs") or () if d.lower() in blob]
        hit = effects(drugs)
        profiles = {r["topic"]: r["profile"] for r in load_yaml(registry_path("p2_complaint_pool.yaml"))["complaints"]}
        vocab = vocabulary()
        return bad + [s for s in shared_audit._patient_sentences(item["case"]) if profiles.get(vocab.get(s)) in hit]

    shared_audit.said_sentences, shared_audit.side_effect_complaint = source_sentences, source_effect_check
    from haenv.app.pack_hooks import register_pack_extension
    register_pack_extension("p2:complaints", register)


def register(pack):
    if pack.name != "p2":
        return
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    register_hook(Stage.CASE, "p2:complaints", with_complaints)


def with_complaints(next_call, cs, *args, **kwargs):
    if not isinstance(cs.latent.get("p2"), dict):
        return next_call(cs, *args, **kwargs)
    token = _ACTIVE.set(True)
    try:
        from haenv.world.generation.patient_sampling import benign_symptoms
        sampled = benign_symptoms({"case_id": cs.case_id, "raw": cs.raw, "latent": cs.latent})
        return next_call(replace(cs, raw=sampled["raw"]), *args, **kwargs)
    finally:
        _ACTIVE.reset(token)
