"""Load-time validation of the declarative registries (`registry/*.yaml`).

Loading is default-deny: an unregistered field, a reference that resolves to nothing, or a
combination that violates an exclusion constraint raises `RegistryError`, so a bad spec never
reaches generation or scoring.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("haenv.registry")

# Resources resolve through `data_root()`: the repo root in a source tree, the package directory in a wheel.
from haenv import data_root as _data_root
from haenv import data_root as _dr   # resource root: repo in a source tree, package dir in a wheel
ROOT = _data_root()

# Registry paths resolve only through the cache helper, where tests redirect them.

PAIR_FIELDS = frozenset({"pair", "take", "threads", "composition_v2"})

# The independent table is a seed table; the overlay expands it into the kernel's shape.
SEED_FIELDS = frozenset({"weight", "symptoms", "outcome_label"})
SEED_REQUIRED = ("weight", "symptoms", "outcome_label")

# Written out rather than derived from the kernel, so a new kernel field is not admitted silently.
KERNEL_SPEC_FIELDS = frozenset({"aliases", "clinician_warranted", "diagnosis", "join_gold",
                                "outcome_label", "red_flag", "specialty", "symptoms",
                                "tests", "urgency", "weight"})


class RegistryError(ValueError):
    """Load-time validation failed."""


def _load_yaml(name: str) -> dict:
    # Parsed once per process and handed out as deep copies. The path is not built here: the cache
    # helper is the single resolution point, where the plugin overlay attaches.
    from .regpath import overlaid_yaml as _overlaid
    from .regpath import registry_path as _rp
    p = _rp(name)
    if not p.is_file():
        raise RegistryError(f"registry file missing: {p}")
    d = _overlaid(p) or {}
    if not isinstance(d, dict):
        raise RegistryError(f"{name}: top level must be a mapping")
    return d


def load_conditions() -> dict[str, dict]:
    raw = _load_yaml("conditions_independent.yaml").get("conditions") or {}
    out: dict[str, dict] = {}
    for cid, spec in raw.items():
        if not isinstance(spec, dict):
            raise RegistryError(f"{cid}: spec must be a mapping")
        bad = set(spec) - SEED_FIELDS
        if bad:                                             # ⑥ unknown_field
            raise RegistryError(f"{cid}: unregistered fields {sorted(bad)}; "
                                f"a misspelled field would be silently ignored, so unknown fields are rejected"
                                f" (seed table allows: {sorted(SEED_FIELDS)})")
        for f in SEED_REQUIRED:                             # ① missing_field
            if spec.get(f) is None:
                raise RegistryError(f"{cid}: missing required field {f}")
        syms = spec.get("symptoms")
        if syms is not None:
            spec = {**spec, "symptoms": [tuple(s) for s in syms]}
        out[cid] = spec
    return out


def load_unified_conditions() -> dict[str, dict]:
    """Unified condition specs authored in this repository. Every field of the kernel's domain is
    required and has no default.
    """
    raw = _load_yaml("conditions_unified.yaml").get("conditions") or {}
    out: dict[str, dict] = {}
    for cid, spec in raw.items():
        if not isinstance(spec, dict):
            raise RegistryError(f"{cid}: spec must be a mapping")
        # `min_age`/`max_age` are enforced by the emission gate, not the sampler, so demographics stay
        # independent of the diagnosis. `clinical_review` is blocked / pending / done; `draft` is still read.
        bad = set(spec) - KERNEL_SPEC_FIELDS - {"draft", "min_age", "max_age", "clinical_review"}
        if bad:
            raise RegistryError(f"{cid}: unregistered fields {sorted(bad)}"
                                f" (allowed: {sorted(KERNEL_SPEC_FIELDS)} + draft/min_age/max_age/clinical_review)")
        _cr = str(spec.get("clinical_review") or "pending")
        if "clinical_review" in spec and _cr not in ("blocked", "pending", "done"):
            raise RegistryError(f"{cid}: clinical_review={_cr!r} is not recognized"
                                f" (allowed: blocked / pending / done)")
        miss = sorted(KERNEL_SPEC_FIELDS - set(spec))
        if miss:                                    # same domain as the kernel spec: every field required, no defaults
            raise RegistryError(f"{cid}: missing fields {miss}; unified conditions must declare every field, "
                                f"since a default would invent gold")
        if spec.get("join_gold") != "unified":
            raise RegistryError(f"{cid}: this table only accepts join_gold=unified, got {spec.get('join_gold')!r}")
        syms = [tuple(x) for x in (spec.get("symptoms") or ())]
        if len(syms) < 3:
            raise RegistryError(f"{cid}: fewer than 3 symptoms; a multi-system projection over time needs at least 3")
        out[cid] = {**spec, "draft": bool(spec.get("draft", False)), "symptoms": syms,
                    "weight": tuple(spec["weight"]), "aliases": list(spec["aliases"]),
                    "tests": list(spec["tests"]), "specialty": list(spec["specialty"])}
    return out


def load_threads() -> dict[str, tuple[dict, ...]]:
    raw = _load_yaml("threads.yaml").get("threads") or {}
    return {k: tuple(dict(d) for d in v) for k, v in raw.items()}


def load_causal_clusters() -> tuple[tuple[str, ...], ...]:
    raw = _load_yaml("causal_clusters.yaml").get("clusters") or []
    return tuple(tuple(c) for c in raw)


# ---- composition v2: prevalence, association clusters, symptom penetrance + variants -------------
STATUSES = frozenset({"verified", "abstract", "expert_estimate", "design"})
RARE_PREVALENCE = 1 / 2000          # M2 spec C1: prevalence below this counts as a rare disease


def composition_v2_pair_ids() -> frozenset:
    """Pairs added for composition-v2 jobs (`composition_v2: true` in `composition_comorbid.yaml`). Global pools that
    average over the spec catalogue (`findings_render._disease_abnormality_pool`) skip them, so adding a v2 pair never
    shifts a legacy job's draws."""
    raw = _load_yaml("composition_comorbid.yaml").get("pairs") or {}
    return frozenset(str(k) for k, v in raw.items() if isinstance(v, dict) and v.get("composition_v2"))


def load_disease_prevalence() -> dict[str, dict]:
    """`{spec_id: {value, population, source, source_status, ...}}`. Consumed at job-generation time only
    (allocation weights); judging never reads it."""
    raw = _load_yaml("disease_prevalence.yaml").get("prevalence") or {}
    allowed = {"value", "population", "general_value", "source", "source_status", "note"}
    out: dict[str, dict] = {}
    for sid, it in raw.items():
        bad = set(it) - allowed
        if bad:
            raise RegistryError(f"prevalence {sid}: unregistered fields {sorted(bad)}")
        for f in ("value", "population", "source", "source_status"):
            if it.get(f) in (None, ""):
                raise RegistryError(f"prevalence {sid}: missing {f}")
        if not (0.0 < float(it["value"]) < 1.0):
            raise RegistryError(f"prevalence {sid}: value must be in (0, 1)")
        if it["source_status"] not in STATUSES:
            raise RegistryError(f"prevalence {sid}: source_status={it['source_status']!r} not in {sorted(STATUSES)}")
        out[str(sid)] = dict(it)
    return out


def is_rare(prev: dict) -> bool:
    """Rare when either the matched-population value or the general-population value is below 1/2000."""
    vals = [float(prev["value"])] + ([float(prev["general_value"])] if prev.get("general_value") else [])
    return min(vals) < RARE_PREVALENCE


def load_association_pairs() -> dict[frozenset, dict]:
    """Association (shared-susceptibility) pairs, kept apart from the causal clusters: a pair here keeps
    `join_gold: comorbidity` with two separate work-ups. Keyed by `frozenset(pair)`."""
    raw = _load_yaml("association_clusters.yaml")
    allowed = {"pair", "cluster", "or", "measure", "source", "source_status", "note",
               "overrides_causal_cluster", "status"}
    out: dict[frozenset, dict] = {}
    for it in raw.get("pairs") or []:
        bad = set(it) - allowed
        if bad:
            raise RegistryError(f"association pair {it.get('pair')}: unregistered fields {sorted(bad)}")
        pr = tuple(it.get("pair") or ())
        if len(pr) != 2 or pr[0] == pr[1]:
            raise RegistryError(f"association pair {pr}: must name two different specs")
        if it.get("source_status") not in STATUSES:
            raise RegistryError(f"association pair {pr}: source_status={it.get('source_status')!r} not in {sorted(STATUSES)}")
        if it.get("or") is not None and float(it["or"]) <= 0:
            raise RegistryError(f"association pair {pr}: or must be positive")
        if frozenset(pr) in out:
            raise RegistryError(f"association pair {pr}: declared twice")
        out[frozenset(pr)] = dict(it)
    return out


def association_admissible(a: str, b: str, prevalence: dict, assoc: dict) -> tuple[bool, float, str]:
    """M2 spec C2 admission: (1) registered association pair with OR >= 1.5, or (2) both prevalences >= 1%
    and the independent product >= 1e-4. Returns `(ok, or_used, why)`; on path (2) `or_used` is the registered OR if any, else 1.0."""
    e = assoc.get(frozenset((a, b)))
    if e and e.get("or") is not None and float(e["or"]) >= 1.5:
        return True, float(e["or"]), f"association cluster {e.get('cluster')} OR={e['or']}"
    pa, pb = (prevalence.get(a) or {}).get("value"), (prevalence.get(b) or {}).get("value")
    if pa and pb and pa >= 0.01 and pb >= 0.01 and pa * pb >= 1e-4:
        _o = float(e["or"]) if e and e.get("or") else 1.0      # a known OR below 1.5 still weights the draw
        return True, _o, f"both prevalences >= 1% (independent product {pa * pb:.2g})"
    return False, 0.0, "not in an association cluster with OR>=1.5 and not both >=1% prevalence"


def load_symptom_penetrance() -> dict[str, list[dict]]:
    """Per-spec symptom table `{spec_id: [{idx, text, penetrance, source, source_status, variants}]}`.
    `text` must equal the spec's canonical symptom text (checked by `check_symptom_penetrance`)."""
    raw = _load_yaml("symptom_penetrance.yaml").get("symptoms") or {}
    allowed = {"idx", "text", "penetrance", "source", "source_status", "variants"}
    out: dict[str, list[dict]] = {}
    for sid, rows in raw.items():
        seen = set()
        for it in rows:
            bad = set(it) - allowed
            if bad:
                raise RegistryError(f"symptom_penetrance {sid}: unregistered fields {sorted(bad)}")
            if int(it["idx"]) in seen:
                raise RegistryError(f"symptom_penetrance {sid}: idx {it['idx']} declared twice")
            seen.add(int(it["idx"]))
            if not (0.0 < float(it["penetrance"]) <= 1.0):
                raise RegistryError(f"symptom_penetrance {sid}[{it['idx']}]: penetrance must be in (0, 1]")
            if it.get("source_status") not in STATUSES:
                raise RegistryError(f"symptom_penetrance {sid}[{it['idx']}]: source_status={it.get('source_status')!r}")
            v = it.get("variants") or {}
            if set(v) - {"colloquial", "atypical"}:
                raise RegistryError(f"symptom_penetrance {sid}[{it['idx']}]: variant levels must be colloquial/atypical")
            if len(v.get("colloquial") or []) + len(v.get("atypical") or []) < 2:
                raise RegistryError(f"symptom_penetrance {sid}[{it['idx']}]: needs >= 2 variants besides the canonical text")
        out[str(sid)] = [dict(it) for it in rows]
    return out


def check_symptom_penetrance(specs: dict) -> list[str]:
    """Drift gate: each entry's canonical text equals the spec's symptom text at `idx`, and no variant text is
    shared between two symptoms or equal to any canonical text."""
    bad: list[str] = []
    table = load_symptom_penetrance()
    owner: dict[str, str] = {}
    canon = {str(s[1]) for sp in specs.values() for s in sp["symptoms"]}
    for sid, rows in table.items():
        sp = specs.get(sid)
        if sp is None:
            bad.append(f"{sid}: not in the condition registry")
            continue
        for it in rows:
            i = int(it["idx"])
            if i >= len(sp["symptoms"]) or str(sp["symptoms"][i][1]) != str(it["text"]):
                bad.append(f"{sid}[{i}]: canonical text drifted from the spec")
            for lvl, txts in (it.get("variants") or {}).items():
                for t in txts:
                    key = "".join(str(t).split())
                    if key in owner and owner[key] != f"{sid}[{i}]":
                        bad.append(f"{sid}[{i}]: variant {t!r} also used by {owner[key]}")
                    if str(t) in canon:
                        bad.append(f"{sid}[{i}]: variant {t!r} equals a canonical symptom text")
                    owner[key] = f"{sid}[{i}]"
    return sorted(bad)


def load_negated_findings() -> dict[str, tuple[dict, ...]]:
    """Findings the question asserts as normal or absent, mapped to the specs that would make them
    abnormal.
    """
    raw = _load_yaml("negated_findings.yaml") or {}
    allowed = {"finding", "where", "contradicted_by", "item", "direction", "why_not_derived", "idx"}
    out: dict[str, tuple[dict, ...]] = {}
    for sid, items in raw.items():
        if not isinstance(items, list) or not items:
            raise RegistryError(f"{sid}: negated_findings must be a non-empty list")
        rows = []
        for it in items:
            bad = set(it) - allowed
            if bad:
                raise RegistryError(f"{sid}: unregistered fields {sorted(bad)} (allowed: {sorted(allowed)})")
            if not it.get("finding") or not it.get("contradicted_by"):
                raise RegistryError(f"{sid}: finding and contradicted_by are required")
            # Where `item`/`direction` are given, contradicting specs are derived from the findings registry
            # and merged with the hand-written list; otherwise `why_not_derived` is required.
            hand = tuple(it["contradicted_by"])
            item, direction = it.get("item"), it.get("direction")
            if item and direction:
                derived = _declares(str(item), str(direction))
                merged = tuple(sorted((set(hand) | set(derived)) - {str(sid)}))
            else:
                if not str(it.get("why_not_derived") or "").strip():
                    raise RegistryError(
                        f"{sid}/{it['finding']}: without `item`/`direction`, "
                        f"`why_not_derived` is required, so 'not derivable yet' is distinguishable from 'forgotten'")
                merged = hand
            if it.get("idx") is not None and (not isinstance(it["idx"], int) or isinstance(it["idx"], bool) or it["idx"] < 0):
                raise RegistryError(f"{sid}/{it['finding']}: idx must be a non-negative symptom index")
            rows.append({"finding": str(it["finding"]), "where": str(it.get("where", "")),
                         "contradicted_by": merged, "hand_listed": hand,
                         "idx": (int(it["idx"]) if it.get("idx") is not None else None)})
        out[str(sid)] = tuple(rows)
    return out


def _declares(item: str, direction: str) -> tuple[str, ...]:
    out = []
    for cid, prof in (load_condition_findings() or {}).items():
        for f in (prof.get("findings") or ()):
            if str(f.get("id")) == item and str(f.get("gold") or f.get("direction")) == direction:
                out.append(str(cid))
    return tuple(sorted(set(out)))


def load_antagonist_axes() -> tuple[dict, ...]:
    raw = _load_yaml("antagonist_axes.yaml").get("axes") or []
    allowed = {"axis", "high", "low", "why"}
    out = []
    for it in raw:
        bad = set(it) - allowed
        if bad:
            raise RegistryError(f"antagonist_axes: unregistered fields {sorted(bad)}")
        for f in ("axis", "high", "low", "why"):
            if not it.get(f):
                raise RegistryError(f"antagonist_axes: missing required field {f}")
        if it["high"] == it["low"]:
            raise RegistryError(f"antagonist_axes {it['axis']}: high and low are the same spec")
        out.append({k: str(it[k]) for k in allowed})
    return tuple(out)


def pair_conflict(pair, clusters, negated: dict, axes, take=None) -> str | None:
    """Whether this pair may be combined: `None`, or the reason it may not. Shared by the pair
    generator and load-time validation.

    `take` (optional, two index tuples aligned with `pair`): the symptoms of each member that the composed case
    shows. A negated finding registered with `idx` (the position of the symptom that asserts it) only conflicts
    when that symptom is in the member's take; without `take`, or without `idx`, every negated finding applies.
    """
    if any(pair[0] in cl and pair[1] in cl for cl in clusters):   # ③ same_causal_cluster
        cl = next(cl for cl in clusters if pair[0] in cl and pair[1] in cl)
        return (f"{pair[0]} and {pair[1]} are in the same causal cluster {cl}: "
                f"they are one causal chain (two consequences of one process), so the gold is unified, "
                f"not comorbidity")
    for x, y in (tuple(pair), tuple(pair)[::-1]):                 # ⑦ negated_finding_conflict
        for f in negated.get(x, ()):
            if y in f["contradicted_by"]:
                if take is not None and f.get("idx") is not None and int(f["idx"]) not in tuple(take[tuple(pair).index(x)]):
                    continue
                return (f"{x} asserts {f['finding']!r} ({f['where']}) in the case, "
                        f"but a {y} diagnosis makes it abnormal; the record would contradict itself"
                        f" (see registry/negated_findings.yaml)")
    for ax in axes:                                               # ⑧ antagonist_axis
        if {ax["high"], ax["low"]} == set(pair):
            return (f"{ax['high']} and {ax['low']} are opposite diagnoses on {ax['axis']}"
                    f" ({ax['why']}); one patient cannot have both, so the record would contradict itself"
                    f" (see registry/antagonist_axes.yaml)")
    return None


def load_comorbid_pairs(kernel_spec_ids: frozenset | set | None = None,
                        clusters: tuple | None = None) -> dict[str, dict]:
    """Comorbidity combination rules, validated at load. Without `kernel_spec_ids` the
    reference check is skipped.
    """
    raw = _load_yaml("composition_comorbid.yaml").get("pairs") or {}
    clusters = load_causal_clusters() if clusters is None else clusters
    _NEGATED = load_negated_findings()
    _AXES = load_antagonist_axes()
    _OVERRIDES = frozenset(k for k, v in load_association_pairs().items() if v.get("overrides_causal_cluster"))
    out: dict[str, dict] = {}
    for pid, spec in raw.items():
        if not isinstance(spec, dict):
            raise RegistryError(f"{pid}: combination rule must be a mapping")
        bad = set(spec) - PAIR_FIELDS
        if bad:                                             # ⑥ unknown_field
            raise RegistryError(f"{pid}: unregistered fields {sorted(bad)} (allowed: {sorted(PAIR_FIELDS)})")
        for f in ("pair", "take", "threads"):               # ① missing_field
            if not spec.get(f):
                raise RegistryError(f"{pid}: missing required field {f}")
        pair = tuple(spec["pair"])
        if len(pair) != 2:
            # Pairs are binary by design: the overlay interleaves exactly two threads of four evidence
            # points, and `pair_conflict` is written for two specs.
            raise RegistryError(
                f"{pid}: pair must name two specs, got {pair}; "
                f"the item type is fixed at two interleaved threads of 4 evidence points each")
        if pair[0] == pair[1]:                              # ④ self_pair
            raise RegistryError(f"{pid}: both threads are the same spec {pair[0]}; that is not a comorbidity")
        if kernel_spec_ids is not None:                     # ② unknown_spec_ref
            miss = [x for x in pair if x not in kernel_spec_ids]
            if miss:
                raise RegistryError(f"{pid}: references unknown specs {miss}")
        # A pair registered in `association_clusters.yaml` with `overrides_causal_cluster` is exempt from
        # check (3) only (same causal cluster); checks (7) and (8) still apply.
        _cl = () if frozenset(pair) in _OVERRIDES else clusters
        take = tuple(tuple(t) for t in spec["take"])
        if (_why := pair_conflict(pair, _cl, _NEGATED, _AXES, take=take)):
            raise RegistryError(f"{pid}: {_why}")
        if len(take) != 2 or any(len(t) != 2 for t in take):
            raise RegistryError(f"{pid}: take must be two (start, end) pairs, got {take}")
        for t in take:                                      # ⑤ take_out_of_range
            if not (0 <= t[0] < t[1]):
                raise RegistryError(f"{pid}: invalid take range {t} (requires 0 <= start < end)")
        out[pid] = {"pair": pair, "take": take, "threads": tuple(spec["threads"])}
    return out


def check_take_within(pairs: dict, specs: dict) -> list[str]:
    """A take must not run past the number of symptoms the spec has. Returns a list, since a
    failure usually means the kernel spec changed.
    """
    bad = []
    for pid, p in (pairs or {}).items():
        for sid, t in zip(p["pair"], p["take"]):
            n = len((specs.get(sid) or {}).get("symptoms") or ())
            if t[1] > n:
                bad.append(f"{pid}: take {t} exceeds the {n} symptoms of {sid}")
    return sorted(bad)


def check_expanded_domain(expanded: dict) -> list[str]:
    """The expanded spec field domain must match the kernel's. Reports rather than raises: a
    mismatch usually means the kernel gained a field.
    """
    out = []
    for sid, spec in (expanded or {}).items():
        extra = set(spec) - KERNEL_SPEC_FIELDS - {"source"}
        if extra:
            out.append(f"{sid}: unregistered fields after expansion {sorted(extra)}; "
                       f"if the kernel added them, update registry.KERNEL_SPEC_FIELDS")
    return sorted(out)


# ================================================================ vocabulary layer
CONTEXT_FACETS = frozenset({"attr", "course", "gold", "interp", "join",
                            "measure", "neg", "sign", "therapy"})


def load_context_vocab() -> dict[str, tuple[str, str | None]]:
    raw = _load_yaml("vocab_context.yaml").get("context") or {}
    out: dict[str, tuple[str, str | None]] = {}
    for phrase, spec in raw.items():
        if not isinstance(spec, dict) or "facet" not in spec:
            raise RegistryError(f"context entry {phrase!r}: must be a {{facet, note}} mapping")
        bad = set(spec) - {"facet", "note"}
        if bad:
            raise RegistryError(f"context entry {phrase!r}: unregistered fields {sorted(bad)}")
        f = spec["facet"]
        if f not in CONTEXT_FACETS:
            raise RegistryError(
                f"context entry {phrase!r}: facet={f!r} is not registered; a misspelled facet "
                f"would downgrade a whole class of wording (allowed: {sorted(CONTEXT_FACETS)})")
        out[phrase] = (f, spec.get("note"))
    return out


def load_alias_excludes() -> dict[str, tuple[str, ...]]:
    raw = _load_yaml("vocab_alias_excludes.yaml").get("excludes") or {}
    for k, v in raw.items():
        if not isinstance(v, list) or not v:
            raise RegistryError(f"alias_excludes {k!r}: exclusion set must be a non-empty list")
        if any(k == x for x in v):
            raise RegistryError(f"alias_excludes {k!r}: exclusion set must not contain the alias itself"
                                f"; the alias would never match")
    return {k: tuple(v) for k, v in raw.items()}


def load_alias_overlap_ok() -> dict[tuple[str, str], str]:
    raw = _load_yaml("vocab_alias_overlap_ok.yaml").get("allowed_overlaps") or []
    out: dict[tuple[str, str], str] = {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "pair" not in item or "reason" not in item:
            raise RegistryError(f"allowed_overlaps[{i}]: must be {{pair: [a,b], reason: ...}}")
        pair = tuple(item["pair"])
        if len(pair) != 2:
            raise RegistryError(f"allowed_overlaps[{i}]: pair must have two items, got {pair}")
        if not str(item["reason"]).strip():
            raise RegistryError(f"allowed_overlaps[{i}]: reason is required")
        if pair in out:
            raise RegistryError(f"allowed_overlaps: duplicate pair {pair}")
        out[pair] = item["reason"]
    return out


def load_specialty_synonyms() -> dict[str, tuple[str, ...]]:
    raw = _load_yaml("vocab_specialty.yaml").get("specialty") or {}
    for k, v in raw.items():
        if not isinstance(v, list) or not v:
            raise RegistryError(f"specialty {k!r}: synonyms must be a non-empty list; "
                                f"an empty list scores every English answer for this specialty as 0")
        if any(not str(x).strip() or str(x) != str(x).lower() for x in v):
            raise RegistryError(f"specialty {k!r}: synonyms must be non-empty and lowercase (matching is case-folded)")
    return {k: tuple(v) for k, v in raw.items()}


RIVAL_FIELDS = frozenset({"name", "aliases", "discriminator",
                          # Structured forms of the discriminator (mutually exclusive).
                          "discriminator_finding", "discriminator_unresolvable",
                          # Kept when an 'indistinguishable' claim is lifted, so the change stays visible.
                          "discriminator_unresolvable_superseded"})


def load_rivals() -> dict[str, tuple[dict, ...]]:
    """Look-alike rivals: the disease that overlaps the gold's presentation and must be ruled out.

    Every entry must say what distinguishes the two; otherwise it is another name for the gold.
    Alias overlap with the gold is checked in the overlay.
    """
    raw = _load_yaml("rivals.yaml").get("rivals") or {}
    out: dict[str, tuple[dict, ...]] = {}
    for sid, items in raw.items():
        if not isinstance(items, list) or not items:
            raise RegistryError(f"rivals {sid!r}: must be a non-empty list; "
                                f"a spec with no rivals should be removed, not left empty")
        seen: set[str] = set()
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                raise RegistryError(f"rivals {sid}[{i}]: must be a mapping")
            bad = set(it) - RIVAL_FIELDS
            if bad:
                raise RegistryError(f"rivals {sid}[{i}]: unregistered fields {sorted(bad)}"
                                    f" (allowed: {sorted(RIVAL_FIELDS)})")
            for f in ("name", "aliases", "discriminator"):
                if not it.get(f):
                    raise RegistryError(f"rivals {sid}[{i}]: missing required field {f}"
                                        + (" (without a discriminator it is not a rival)"
                                           if f == "discriminator" else ""))
            if not isinstance(it["aliases"], list) or not it["aliases"]:
                raise RegistryError(f"rivals {sid}[{i}]: aliases must be a non-empty list")
            if any(str(a) != str(a).lower() and str(a).isascii() for a in it["aliases"]):
                raise RegistryError(f"rivals {sid}[{i}]: English aliases must be lowercase (matching is case-folded)")
            if it["name"] in seen:
                raise RegistryError(f"rivals {sid}: duplicate rival {it['name']!r}")
            seen.add(it["name"])
        out[sid] = tuple({"name": it["name"], "aliases": tuple(it["aliases"]),
                          "discriminator": it["discriminator"],
                          "discriminator_finding": it.get("discriminator_finding"),
                          "discriminator_unresolvable": it.get("discriminator_unresolvable"),
                          "discriminator_unresolvable_superseded":
                              it.get("discriminator_unresolvable_superseded")}
                         for it in items)
    return out


def load_case_ids() -> dict[str, str]:
    """Frozen public case numbers, spec key -> anonymous id. Assigned explicitly, never by position,
    so adding a spec does not redraw another case's patient.
    """
    raw = _load_yaml("case_ids.yaml").get("case_ids") or {}
    if not raw:
        raise RegistryError("case_ids.yaml is empty; case ids must be frozen explicitly, never assigned by position")
    seen: dict[str, str] = {}
    for sid, anon in raw.items():
        a = str(anon)
        if not a.startswith("JD-") or not a[3:].isdigit():
            raise RegistryError(f"case_ids {sid!r}: id {a!r} must have the form JD-<digits>")
        if a in seen:
            raise RegistryError(f"case_ids: id {a} is used by both {seen[a]} and {sid}; "
                                f"two cases with one id overwrite each other in a batch")
        seen[a] = sid
    return {str(k): str(v) for k, v in raw.items()}


def check_case_ids_cover(spec_ids) -> list[str]:
    """Every spec must have an explicit number; missing ones are reported with the next free number."""
    have = load_case_ids()
    missing = [s for s in spec_ids if s not in have]
    if not missing:
        return []
    nxt = max((int(v[3:]) for v in have.values()), default=0) + 1
    return [f"{len(missing)} specs have no frozen case id: {missing[:6]}; "
            f"add them to registry/case_ids.yaml in order from JD-{nxt:02d} (do not insert between existing ids)"]


#: `recall` is read only by `load_test_recall_vocab()` (the three-valued matcher in
#: `haenv/judges/recall_match.py`); `load_test_vocab()` does not return it.
TEST_VOCAB_SECTIONS = frozenset({"stopwords", "prefixes", "synonyms", "recall"})
TEST_RECALL_SECTIONS = frozenset({"atomic_terms", "prefixes", "qualifiers", "negation", "conditional", "past_result",
                                  "imaging_modalities", "imaging_generic",
                                  "hint_stop_zh", "hint_stop_en", "generic_segments", "weak_forms",
                                  "synonyms", "distinct_from", "panels", "member_panels"})


def load_test_vocab() -> dict:
    """Test-name vocabulary: stopwords, prefixes and synonyms. Without it, recall would measure the
    model's choice of abbreviations.
    """
    raw = _load_yaml("vocab_tests.yaml")
    bad = set(raw) - TEST_VOCAB_SECTIONS
    if bad:
        raise RegistryError(f"vocab_tests.yaml: unregistered sections {sorted(bad)}"
                            f" (allowed: {sorted(TEST_VOCAB_SECTIONS)})")
    stop = [str(x) for x in (raw.get("stopwords") or [])]
    pref = [str(x) for x in (raw.get("prefixes") or [])]
    syn_raw = raw.get("synonyms") or {}
    if not stop or not pref or not syn_raw:
        raise RegistryError("vocab_tests.yaml: all three sections must be non-empty; "
                            "an empty section silently disables that layer")
    syn: dict[str, tuple[str, ...]] = {}
    for k, v in syn_raw.items():
        if not isinstance(v, list) or not v:
            raise RegistryError(f"tests synonyms {k!r}: must be a non-empty list")
        for a in v:
            if str(a).isascii() and str(a) != str(a).lower():
                raise RegistryError(f"tests synonyms {k!r}: English synonyms must be lowercase (matching is case-folded), "
                                    f"got {a!r}")
            if str(a).strip().lower() == str(k).strip().lower():
                raise RegistryError(f"tests synonyms {k!r}: a synonym must not repeat the key itself")
        syn[str(k)] = tuple(str(x) for x in v)
    return {"stopwords": tuple(stop), "prefixes": tuple(pref), "synonyms": syn}



def _lower_ok(where: str, forms) -> tuple[str, ...]:
    out = []
    for a in forms or ():
        a = str(a)
        if not a.strip():
            raise RegistryError(f"{where}: empty form")
        if a.isascii() and a != a.lower():
            raise RegistryError(f"{where}: English forms must be lowercase (matching is case-folded), got {a!r}")
        out.append(a)
    if len(set(out)) != len(out):
        raise RegistryError(f"{where}: duplicate forms {sorted({x for x in out if out.count(x) > 1})}")
    return tuple(out)


def _sourced(where: str, v) -> str:
    src = str((v or {}).get("source") or "").strip() if isinstance(v, dict) else ""
    if not src:
        raise RegistryError(f"{where}: every entry must carry a non-empty `source` "
                            "(standard name / abbreviation / alias / panel composition)")
    return src


def load_test_recall_vocab() -> dict:
    """The `recall` section of `vocab_tests.yaml`, validated default-deny.

    Every synonym, weak form, distinct-from term and panel carries a `source`; a missing source
    is a load error, so an entry cannot be added without saying what kind of equivalence it is.
    """
    raw = _load_yaml("vocab_tests.yaml").get("recall")
    if not isinstance(raw, dict) or not raw:
        raise RegistryError("vocab_tests.yaml: section `recall` missing or empty")
    bad = set(raw) - TEST_RECALL_SECTIONS
    missing = TEST_RECALL_SECTIONS - set(raw)
    if bad or missing:
        raise RegistryError(f"vocab_tests.yaml recall: unregistered {sorted(bad)} / missing {sorted(missing)}")
    out: dict = {}
    for k in ("atomic_terms", "prefixes", "negation", "conditional", "past_result", "imaging_modalities",
              "imaging_generic", "hint_stop_zh", "hint_stop_en"):
        v = raw[k]
        if not isinstance(v, list) or not v:
            raise RegistryError(f"vocab_tests.yaml recall.{k}: must be a non-empty list")
        out[k] = _lower_ok(f"recall.{k}", v) if k not in ("atomic_terms", "prefixes") else tuple(str(x) for x in v)
    for k in ("qualifiers", "generic_segments"):
        v = raw[k]
        if not isinstance(v, dict) or not v or not all(str(x).strip() for x in v.values()):
            raise RegistryError(f"vocab_tests.yaml recall.{k}: mapping fragment -> reason (non-empty)")
        out[k] = {str(a): str(b) for a, b in v.items()}
    for k, field in (("synonyms", "forms"), ("weak_forms", "forms"), ("distinct_from", "terms")):
        v = raw[k]
        if not isinstance(v, dict) or not v:
            raise RegistryError(f"vocab_tests.yaml recall.{k}: must be a non-empty mapping")
        out[k] = {}
        for key, ent in v.items():
            where = f"recall.{k}.{key}"
            src = _sourced(where, ent)
            forms = _lower_ok(where, ent.get(field))
            if not forms:
                raise RegistryError(f"{where}: `{field}` must be non-empty")
            out[k][str(key)] = {field: forms, "source": src}
    out["panels"] = {}
    for key, ent in (raw["panels"] or {}).items():
        where = f"recall.panels.{key}"
        src = _sourced(where, ent)
        if not isinstance(ent.get("definitive"), bool):
            raise RegistryError(f"{where}: `definitive` must be true/false")
        al = _lower_ok(where, ent.get("aliases"))
        mem = tuple(str(x) for x in ent.get("members") or ())
        if not al or not mem:
            raise RegistryError(f"{where}: aliases and members must be non-empty")
        out["panels"][str(key)] = {"aliases": al, "members": mem, "definitive": ent["definitive"], "source": src}
    out["member_panels"] = {}
    for key, ent in (raw["member_panels"] or {}).items():
        where = f"recall.member_panels.{key}"
        src = _sourced(where, ent)
        mins = ent.get("min")
        mem = {str(m): _lower_ok(f"{where}.{m}", f) for m, f in (ent.get("members") or {}).items()}
        if not isinstance(mins, int) or mins < 2 or len(mem) < mins:
            raise RegistryError(f"{where}: `min` must be an int >= 2 and <= number of members")
        out["member_panels"][str(key)] = {"min": mins, "members": mem, "source": src}
    return out

# ---------------------------------------------------------------- findings layer

#: The action threshold is the level that clinically triggers work-up, recorded only where an
#: authoritative figure exists; a reference range's upper bound may not cross it.
FINDING_FIELDS = {"source", "review", "name_cn", "name_en", "aliases", "unit",
                  "action_threshold",
                  # Which test obtains this measurement. Used for scoring only, never rendered.
                  "acquired_by",
                  "ref", "panel", "note", "qualitative", "sex_specific"}
DIRECTIONS = {"high", "low", "normal", "positive", "negative"}
MAGNITUDES = {"mild", "moderate", "marked"}
ROLES = {"screening", "supportive", "confirmatory"}
TRAJECTORIES = {"stable", "progressive", "episodic", "fluctuating", "treatment_responsive"}
PROFILE_FIELDS = {"id", "direction", "magnitude", "role", "trajectory", "n", "penetrance", "penetrance_source"}

# Findings that are only interpretable together.
PAIRED_FINDINGS: tuple[tuple[str, str, str], ...] = (
    ("Aldosterone", "Renin", "醛固酮/肾素比值(ARR)才是筛查量,只给其一无法解读"),
    ("Ca", "PTH", "高钙必须与 PTH 成对:PTH 不被抑制才指向甲旁亢"),
    ("Testosterone", "SHBG", "SHBG 低会把正常总睾酮掩盖成假阴性(FAI)"),
)


def load_findings() -> dict[str, dict]:
    """The findings vocabulary. `source` is required, and self-authored entries must carry a
    `review` status.
    """
    raw = (_load_yaml("findings.yaml") or {}).get("findings") or {}
    if not raw:
        raise RegistryError("findings.yaml is empty or lacks the top-level `findings:` key")
    out: dict[str, dict] = {}
    for fid, v in raw.items():
        if not isinstance(v, dict):
            raise RegistryError(f"{fid}: must be a mapping")
        bad = set(v) - FINDING_FIELDS
        if bad:
            raise RegistryError(f"{fid}: unregistered fields {sorted(bad)} (allowed: {sorted(FINDING_FIELDS)})")
        src = str(v.get("source") or "")
        if not src:
            raise RegistryError(f"{fid}: source is required; every finding must be traceable")
        if not (src.startswith("upstream:") or src == "haenv-authored" or src.startswith("kernel:")):
            raise RegistryError(f"{fid}: source={src!r} is not one of the three allowed source kinds")
        if src == "haenv-authored" and not v.get("review"):
            raise RegistryError(f"{fid}: haenv-authored entries need `review` (use pending if not yet reviewed)")
        if not v.get("name_cn") or not v.get("unit"):
            raise RegistryError(f"{fid}: name_cn and unit are required")
        if not v.get("qualitative") and not isinstance(v.get("ref"), dict):
            raise RegistryError(f"{fid}: a non-qualitative item needs a ref range")
        out[str(fid)] = dict(v)
    return _merge_upstream_findings(out)


def _merge_upstream_findings(fx: dict) -> dict:
    """Merge the upstream indicator table: `enrich` fills only empty fields (aliases are unioned),
    `add` contributes missing entries marked pending review. Upstream never overwrites a
    self-authored value.
    """
    from .regpath import load_registry as _lr
    from .regpath import registry_path as _rp
    if not _rp("findings_upstream.yaml").is_file():
        return fx
    doc = _lr("findings_upstream.yaml") or {}
    for k, add in (doc.get("enrich") or {}).items():
        cur = fx.get(k)
        if not cur:
            continue
        for fld, val in add.items():
            if fld == "aliases":
                have = list(cur.get("aliases") or [])
                low = {str(x).lower() for x in have}
                cur["aliases"] = have + [x for x in val if str(x).lower() not in low]
            elif not cur.get(fld):                 # fill gaps only
                cur[fld] = val
    # Identity is the normalised Chinese name or the id, never aliases or the English name (distinct
    # tests can share an English root); a match merges the upstream key in as an alias.
    _norm = lambda x: __import__("re").sub(r"[\s\-_·]", "", str(x or "")).lower()
    known: dict[str, str] = {}
    for _fid, _v in fx.items():
        for _n in (_v.get("name_cn"), _fid):
            if _n:
                known.setdefault(_norm(_n), _fid)
    for k, ent in (doc.get("add") or {}).items():
        if k in fx:
            continue                               # identical key: the original path
        hit = known.get(_norm(k)) or known.get(_norm(ent.get("name_cn")))
        if hit:
            cur = fx[hit]
            have = list(cur.get("aliases") or [])
            low = {_norm(x) for x in have} | {_norm(hit), _norm(cur.get("name_cn")),
                                              _norm(cur.get("name_en"))}
            for a in ([k, ent.get("name_cn"), ent.get("name_en")]
                      + list(ent.get("aliases") or [])):
                if a and _norm(a) not in low:
                    have.append(a)
                    low.add(_norm(a))
            cur["aliases"] = have
            continue
        fx[k] = dict(ent)
    return fx


def load_condition_findings(findings: dict | None = None) -> dict[str, dict]:
    """Per-condition findings, including trajectories. `all_normal: true` (tested and normal) and a
    missing spectrum (not yet written) are kept distinct.
    """
    fx = findings if findings is not None else load_findings()
    raw = (_load_yaml("condition_findings.yaml") or {}).get("condition_findings") or {}
    out: dict[str, dict] = {}
    for sid, v in raw.items():
        if not isinstance(v, dict):
            raise RegistryError(f"{sid}: must be a mapping")
        rows = v.get("findings")
        if v.get("all_normal"):
            if rows:
                raise RegistryError(f"{sid}: all_normal and findings cannot both be present; "
                                    f"'all normal' and 'these items are abnormal' are different claims")
            out[str(sid)] = {"all_normal": True, "findings": (), "review": v.get("review"),
                             "note": v.get("note", "")}
            continue
        if not isinstance(rows, list) or not rows:
            raise RegistryError(f"{sid}: findings is missing; if everything is normal, write all_normal: true explicitly")
        seen, prof = set(), []
        for it in rows:
            bad = set(it) - PROFILE_FIELDS
            if bad:
                raise RegistryError(f"{sid}: unregistered fields {sorted(bad)}")
            fid = str(it.get("id") or "")
            if fid not in fx:
                raise RegistryError(f"{sid}: finding {fid!r} is not in the vocabulary"
                                    f"; register it in findings.yaml with a source first")
            if fid in seen:
                raise RegistryError(f"{sid}: finding {fid} is declared twice")
            seen.add(fid)
            for key, allowed in (("direction", DIRECTIONS), ("role", ROLES),
                                 ("trajectory", TRAJECTORIES)):
                val = it.get(key)
                if val is None or str(val) not in allowed:
                    raise RegistryError(f"{sid}.{fid}: invalid {key}={val!r} (allowed: {sorted(allowed)})")
            if it.get("magnitude") is not None and str(it["magnitude"]) not in MAGNITUDES:
                raise RegistryError(f"{sid}.{fid}: invalid magnitude={it['magnitude']!r}")
            if str(it["direction"]) in ("normal", "positive", "negative") and it.get("magnitude"):
                raise RegistryError(f"{sid}.{fid}: direction={it['direction']} must not have a magnitude")
            if int(it.get("n", 1)) < 1:
                raise RegistryError(f"{sid}.{fid}: n must be >= 1 (labs are sparse, but measured at least once)")
            if it.get("penetrance") is not None:
                _pn = float(it["penetrance"])
                if not (0.0 < _pn <= 1.0):
                    raise RegistryError(f"{sid}.{fid}: penetrance={_pn} must be in (0, 1]")
                if not str(it.get("penetrance_source") or "").strip():
                    raise RegistryError(f"{sid}.{fid}: penetrance needs a penetrance_source (literature or an explicit estimate note)")
            prof.append({**{k: it.get(k) for k in PROFILE_FIELDS}, "id": fid})
        if not any(p["role"] == "confirmatory" for p in prof):
            raise RegistryError(f"{sid}: at least one confirmatory finding is required; "
                                f"without one the case cannot separate this condition from its rivals")
        for a, b, why in PAIRED_FINDINGS:
            if (a in seen) != (b in seen):
                raise RegistryError(f"{sid}: {a} and {b} must appear together ({why})")
        out[str(sid)] = {"all_normal": False, "findings": tuple(prof),
                         "review": v.get("review"), "note": v.get("note", "")}
    return out


DISC_FIELDS = {"finding", "gold", "rival"}


LOOKALIKE_FIELDS = frozenset({"text", "context", "why_related", "why_looks_benign",
                              "source", "review"})
# A diagnosis name or mechanism term in the wording would write the answer into the question.
_LOOKALIKE_BANNED = ("综合征", "症候", "多囊", "库欣", "甲减", "甲亢", "呼吸暂停",
                     "胰岛素抵抗", "乳糜泻", "脊柱炎", "高雄激素", "皮质醇", "甲状腺")


def load_lookalikes() -> dict[str, tuple[dict, ...]]:
    """Look-alike distractors: they read as something minor but are a manifestation of the gold, so
    they must be joined. They complement benign events, which must not be joined, so both
    "join everything" and "join nothing" lose.

    The wording may not name a diagnosis or mechanism, and both rationale fields, `source` and
    `review` are required.
    """
    doc = _load_yaml("lookalikes.yaml")
    out: dict[str, tuple[dict, ...]] = {}
    for sid, items in (doc.get("lookalikes") or {}).items():
        if not isinstance(items, list) or not items:
            raise RegistryError(f"lookalikes {sid}: must be a non-empty list")
        rows = []
        for i, it in enumerate(items):
            bad = set(it) - LOOKALIKE_FIELDS
            if bad:
                raise RegistryError(f"lookalikes {sid}[{i}]: unregistered fields {sorted(bad)}")
            for k in ("text", "why_related", "why_looks_benign", "source", "review"):
                if not str(it.get(k) or "").strip():
                    raise RegistryError(f"lookalikes {sid}[{i}]: missing `{k}`; "
                                        f"every field is needed for the entry to be reviewable")
            hit = [w for w in _LOOKALIKE_BANNED
                   if w in str(it["text"]) or w in str(it.get("context") or "")]
            if hit:
                raise RegistryError(f"lookalikes {sid}[{i}]: wording contains diagnosis or mechanism terms {hit}; "
                                    f"that writes the answer into the case (a look-alike must read as a minor complaint)")
            rows.append({k: it.get(k) for k in LOOKALIKE_FIELDS})
        out[str(sid)] = tuple(rows)
    return out


DISPUTED_FIELDS = frozenset({"id", "scope", "field", "current", "proposed", "raised",
                             "raised_by", "question", "evidence", "counter", "impact",
                             "who_can_resolve", "status"})


def load_disputed_gold() -> tuple[dict, ...]:
    """Register a gold that may itself be wrong.

    Each entry needs model-side evidence, a counter-argument, the affected dimensions and who can
    resolve it (not the maintainers themselves). Affected cases are still scored but marked in
    the artefacts.
    """
    doc = _load_yaml("disputed_gold.yaml")
    out = []
    for i, it in enumerate(doc.get("disputed_gold") or ()):
        bad = set(it) - DISPUTED_FIELDS
        if bad:
            raise RegistryError(f"disputed_gold[{i}]: unregistered fields {sorted(bad)}")
        for k in ("id", "scope", "question", "evidence", "counter", "impact", "who_can_resolve"):
            if not str(it.get(k) or "").strip():
                raise RegistryError(
                    f"disputed_gold[{i}]: missing `{k}`; "
                    f"`counter` records the other side, `impact` the affected scope, and "
                    f"`who_can_resolve` whether this is a definitional or a clinical question")
        if "我们" in str(it["who_can_resolve"]) and "不是我们" not in str(it["who_can_resolve"]):
            raise RegistryError(f"disputed_gold[{i}]: `who_can_resolve` points at this project's maintainers; "
                                f"that is a definitional question, register it as pending_decision, not as disputed gold")
        out.append(dict(it))
    return tuple(out)


def validate_discriminators(findings: dict, profiles: dict, rivals: dict) -> dict:
    """Check a rival's structured discriminators against the condition's findings spectrum.

    The finding must be in the vocabulary and the spectrum, the gold direction must match the
    spectrum, and gold and rival must differ. Returns per-condition statistics.
    """
    stats: dict[str, dict] = {}
    for sid, items in (rivals or {}).items():
        prof = {p["id"]: p for p in (profiles.get(sid) or {}).get("findings") or ()}
        n_struct = n_unres = n_needs_order = 0
        for it in items:
            df, un = it.get("discriminator_finding"), it.get("discriminator_unresolvable")
            if df and un:
                raise RegistryError(f"{sid}/{it.get('name')}: discriminator_finding and discriminator_unresolvable cannot both be set")
            if un:
                n_unres += 1
                continue
            if not df:
                continue                       # unmarked: reported by the launch gate, not raised here
            bad = set(df) - DISC_FIELDS
            if bad:
                raise RegistryError(f"{sid}/{it.get('name')}: discriminator has unregistered fields {sorted(bad)}")
            fid = str(df.get("finding") or "")
            if fid not in findings:
                raise RegistryError(f"{sid}/{it.get('name')}: discriminator {fid!r} is not in the vocabulary")
            if fid not in prof:
                raise RegistryError(
                    f"{sid}/{it.get('name')}: discriminator {fid} is not in this condition's findings spectrum; "
                    f"the judge would expect a rule-out the case gives no evidence for")
            if str(df.get("gold")) != str(prof[fid]["direction"]):
                raise RegistryError(
                    f"{sid}/{it.get('name')}: gold direction {df.get('gold')!r} does not match the spectrum's "
                    f"{prof[fid]['direction']!r}; the gold contradicts itself")
            if str(df.get("gold")) == str(df.get("rival")):
                raise RegistryError(f"{sid}/{it.get('name')}: gold and rival have the same direction, so it does not discriminate")
            n_struct += 1
            if prof[fid]["role"] != "screening":
                n_needs_order += 1
        stats[sid] = {"n_rivals": len(items), "n_structured": n_struct,
                      "n_unresolvable": n_unres,
                      "n_unlabeled": len(items) - n_struct - n_unres,
                      "n_needs_order": n_needs_order}
    return stats


def derive_tests(spec_id: str, findings: dict, profiles: dict) -> tuple[str, ...]:
    """Derive the work-up (confirmatory and supportive items) from the findings spectrum. Reported
    next to the hand-written `tests`; it does not replace them.
    """
    prof = (profiles.get(spec_id) or {}).get("findings") or ()
    out = []
    for p in prof:
        if p["role"] == "screening":
            continue
        spec = findings.get(p["id"]) or {}
        out.append(str(spec.get("name_cn") or p["id"]))
    return tuple(dict.fromkeys(out))


def compose_condition_findings(profiles: dict, pairs: dict | None = None) -> dict:
    """A comorbid combination's spectrum is the union of its components' spectra.

    Opposite directions for the same item raise; the same direction takes the more extreme
    magnitude and the larger measurement count.
    """
    if pairs is None:
        pairs = (_load_yaml("composition_comorbid.yaml") or {}).get("pairs") or {}
    order = {"mild": 0, "moderate": 1, "marked": 2}
    out = dict(profiles)
    for sid, cfg in pairs.items():
        if sid in out:                       # explicit spectrum wins; never overwritten
            continue
        comps = list((cfg or {}).get("pair") or ())
        parts = [profiles.get(c) for c in comps]
        if not comps or any(p is None for p in parts):
            continue                          # component has no spectrum: do not compose, and do not guess
        merged: dict[str, dict] = {}
        for comp, prof in zip(comps, parts):
            for p in prof.get("findings") or ():
                cur = merged.get(p["id"])
                if cur is None:
                    merged[p["id"]] = {**p, "_from": [comp]}
                    continue
                if str(cur["direction"]) != str(p["direction"]):
                    raise RegistryError(
                        f"{sid}: components {cur['_from'][0]} and {comp} declare opposite directions for {p['id']}"
                        f" ({cur['direction']} vs {p['direction']}); both cannot hold in one patient, "
                        f"so antagonist_axes should block this pair")
                if order.get(str(p.get("magnitude")), -1) > order.get(str(cur.get("magnitude")), -1):
                    cur["magnitude"] = p.get("magnitude")
                cur["n"] = max(int(cur.get("n") or 1), int(p.get("n") or 1))
                if str(p.get("trajectory")) == "episodic":
                    cur["trajectory"] = "episodic"
                if p["role"] == "confirmatory":
                    cur["role"] = "confirmatory"
                cur["_from"].append(comp)
        out[sid] = {"all_normal": False,
                    "findings": tuple({k: v for k, v in m.items() if k != "_from"}
                                      for m in merged.values()),
                    "review": "composed", "note": f"由 {'+'.join(comps)} 合成"}
    return out


def condition_findings_for_case(findings: dict | None = None) -> dict:
    """A case's condition findings: load and compose in one call. Comorbid combinations exist only
    as composed spectra, so every consumer must use this.
    """
    f = load_findings() if findings is None else findings
    return compose_condition_findings(load_condition_findings(f))


# ==========================================================================
# Gold evidence stream specs (world specs), shared by three consumers in different layers.
# ==========================================================================
GOLD_EVIDENCE: dict[str, dict] = {
    # driver -> signal, normal baseline, abnormal level, lead time in days,
    # value range, decimal places
    "medication_intolerance": {"signal": "gi_symptom_score", "normal": 1.2, "abnormal": 6.4,
                               "lead_days": 21, "range": (0.0, 10.0), "ndigits": 1},
    # Must attach to a device stream the patient actually has.
    "calorie_intake_change": {"signal": "diet_carb_pct", "normal": 45.0, "abnormal": 66.0,
                              "lead_days": 21, "range": (10.0, 75.0), "ndigits": 0},
}
