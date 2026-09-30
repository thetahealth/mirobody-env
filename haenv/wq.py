"""wq.py -- the question object `Q` and the checks for the three laws of the
two-layer W x Q ground truth.

`Q` stores a reference to its world (`world_ref`) instead of embedding it, so an
item can drift from its world and that drift can be caught.

1. `Q` embeds none of `W`'s ground-truth fields, storing only `world_ref`.
2. Distractor ground truth (injected entity ids, primitive type) is recorded
   in `Q`, never in `W`.
3. `Q` must not introduce any fact absent from `W` -- `Q` may only open a
   window and ask a question.

SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field

log = logging.getLogger("haenv.wq")

# `W`'s ground-truth half (verifier-only); Law 1 forbids these keys in `Q`.
W_TRUTH_FIELDS: frozenset = frozenset({
    "outcome_label", "label_rule", "gold_drivers", "adjudication",
    "reversal_points", "latent_premise",
})

# `W`'s observable half: what `Q` may expose by opening a window.
W_OBSERVABLE_FIELDS: frozenset = frozenset({
    "case_id", "user_profile", "prediction_context", "longitudinal_data", "evidence_ledger",
})

# Q-ledger keys that decode which EV is a real symptom or a distractor. They are
# gold for the join judges but must be stripped from the distributed `*.Q.jsonl`
# (`frozen/split_wq.py`). Whole keys are stripped, not sub-fields.
Q_TRUTH_FIELDS: frozenset = frozenset({
    "real_symptom_evidence_ids",
    "benign_evidence_ids",
    "lookalike_evidence_ids",
    "id_map",
    "event_schedule",
})


class IronLawViolation(ValueError):
    """One of the three laws was broken. Raises, never just warns."""


class MissingQuestionRecord(IronLawViolation):
    """Requested the Q-side ledger for a world that was never registered.

    Raises rather than returning `{}`, which judges would read as "not
    applicable"; pass `required=False` where absence is acceptable.
    """


class StaleQuestion(IronLawViolation):
    """The world an item references has changed (hash mismatch)."""


def _canonical(obj) -> str:
    """Canonical serialization: sorted keys, no extraneous whitespace."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def world_id(raw) -> str:
    """The world's stable identifier: its `case_id`, unique within a batch only."""
    return str(getattr(raw, "case_id", ""))


def world_truth_hash(raw) -> str:
    """Hash of the whole of `W`'s canonical serialization, observable half included."""
    import dataclasses
    if not dataclasses.is_dataclass(raw):
        raise TypeError(f"world_truth_hash requires a RawCase (dataclass), got {type(raw).__name__}")
    return hashlib.sha256(_canonical(asdict(raw)).encode()).hexdigest()[:16]


def world_ref(raw) -> dict:
    return {"world_id": world_id(raw), "world_truth_hash": world_truth_hash(raw)}


@dataclass
class Question:
    """`Q` -- item ground truth: the world reference and how the question is asked.

    `gold` is empty for item types without a registered deriver.
    """
    question_id: str
    world_ref: dict
    exposed_evidence_ids: tuple[str, ...]
    probe_id: str = ""
    judge_ref: str = ""
    gold: dict = field(default_factory=dict)
    injected_entity_ids: tuple[str, ...] = ()


def build_question(raw, sp, probe_id: str = "", judge_ref: str = "",
                   injected_entity_ids=()) -> Question:
    """Constructs `Q` from `W` and the `SolverPayload` handed to the solver.

    `exposed_evidence_ids` is taken from `sp`, not re-filtered from `raw`, so
    Law 3 can catch a bug in the window-opening logic.
    """
    return Question(
        question_id=f"{world_id(raw)}@T{int((sp.prediction_context or {}).get('prediction_time_T', -1))}",
        world_ref=world_ref(raw),
        exposed_evidence_ids=tuple(str(e.get("evidence_id", ""))
                                   for e in (sp.evidence_ledger or [])),
        probe_id=probe_id, judge_ref=judge_ref,
        gold=_derive_gold_safe(raw),
        injected_entity_ids=tuple(injected_entity_ids or ()),
    )


def _derive_gold_safe(raw) -> dict:
    """Fills `{field: {value, derivation, primitive}}` when derivable, else `{}`."""
    try:
        return derive_gold(raw)
    except (GoldNotDerivable, AttributeError, TypeError):
        return {}


def check_law1_no_embedded_truth(q: Question) -> list[str]:
    """Law 1: `Q` embeds none of `W`'s ground-truth fields.

    Searches dict keys, not serialized text, so a derivation path that names a
    field is allowed. `q.gold` is exempt, provided each entry carries a
    non-empty `derivation` (see `_check_gold_shape`).
    """
    bad: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            bad.update(W_TRUTH_FIELDS & set(node))
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    body = asdict(q)
    gold = body.pop("gold", None)
    walk(body)
    return ([f"Iron law 1: Q embeds W's ground-truth fields {sorted(bad)} -- only world_ref may be stored"] if bad else []) \
        + _check_gold_shape(gold)


def _check_gold_shape(gold) -> list[str]:
    """Every entry of `q.gold` must be `{value, derivation}` with a non-empty
    `derivation`.
    """
    if not gold:
        return []
    if not isinstance(gold, dict):
        return [f"Iron law 1: Q.gold should be {{field: {{value, derivation}}}}, got {type(gold).__name__}"]
    bad = []
    for k, v in gold.items():
        if not isinstance(v, dict) or "value" not in v:
            bad.append(f"Q.gold[{k!r}] is not shaped like {{value, derivation}}")
        elif not (v.get("derivation") or []):
            bad.append(f"Q.gold[{k!r}]'s derivation is empty -- hand-written gold is prohibited")
    return [f"Iron law 1: {b}" for b in bad]


def check_law2_distractor_in_q(q: Question, raw) -> list[str]:
    """Law 2: the injection ledger `injected_event_manifest` must not appear in
    `W.adjudication`; it belongs in the Q-side `_QINJ`.
    """
    adj = getattr(raw, "adjudication", None) or {}
    if "injected_event_manifest" in adj:
        return ["Iron law 2: the distractor ledger injected_event_manifest appears in W.adjudication -- "
                "it records what the item generator injected, and belongs in Q (see wq.register_injection); "
                "a batch with this layout has a world fingerprint that is not comparable"]
    return []


def check_law3_no_new_facts(q: Question, raw) -> list[str]:
    """Law 3: every `evidence_id` referenced by `Q` must be in `W`'s ledger."""
    have = {str(e.get("evidence_id", "")) for e in (getattr(raw, "evidence_ledger", None) or [])}
    ghost = sorted(set(q.exposed_evidence_ids) - have)
    if ghost:
        return [f"Iron law 3: Q references {len(ghost)} pieces of evidence not present in W {ghost[:5]} -- "
                f"Q may only open a window and ask a question, it may not invent facts"]
    return []


def check_fresh(q: Question, raw) -> list[str]:
    now = world_truth_hash(raw)
    want = (q.world_ref or {}).get("world_truth_hash")
    if want and want != now:
        return [f"StaleQuestion: the world this item references has changed (item recorded {want}, now {now})"]
    return []


def check_gold_derivable(q: Question) -> list[str]:
    """Warning tier: gold fields still registered as primitives rather than derived."""
    if not q.gold:
        return ["gold was not derived from W -- no deriver is registered for this item type"]
    prim = sorted(k for k, v in q.gold.items() if (v or {}).get("primitive"))
    return [f"gold's {prim} is still a primitive (registered as a condition property, not derived from W)"] if prim else []


def enforce(q: Question, raw, strict: bool = True) -> list[str]:
    """Runs every check. Under `strict`, Laws 1/3, staleness, gold self-consistency
    and unresolvable derivation steps raise; Law 2, primitive-gold and
    `label_rule` findings are returned as warnings.
    """
    steps = [s for v in (q.gold or {}).values() for s in ((v or {}).get("derivation") or [])]
    fatal = (check_law1_no_embedded_truth(q) + check_law3_no_new_facts(q, raw)
             + check_fresh(q, raw) + check_gold_matches_world(raw)
             + check_derivation_resolvable(raw, steps))
    if fatal and strict:
        raise IronLawViolation("; ".join(fatal))
    return (fatal + check_law2_distractor_in_q(q, raw) + check_gold_derivable(q)
            + check_label_rule_applicable(raw))


# ============================================================ derive_gold
#
# `derive_gold` returns each gold value with a `derivation` whose steps must
# resolve on `W`; missing ground truth is never defaulted.

class GoldNotDerivable(IronLawViolation):
    """`gold` cannot be derived from `W`. No default value is supplied."""


NO_UNIFIED_PATHOLOGY = "无统一病理"      # declared verbatim in the diagnosis of an independent spec


UNRESOLVED = object()      # the path does not resolve (typo / structure changed)
ABSENT = object()          # the path resolves, but the field is absent (absence can be evidence)


def resolve_path(raw, path: str):
    """Resolves one `derivation` step (e.g. `W.adjudication.ddx.threads`).

    Returns `UNRESOLVED` if the path does not resolve and `ABSENT` if only the
    final field is missing; derivations such as `unified` argue from absence.
    """
    node = raw
    segs = [s for i, s in enumerate(path.split(".")) if not (i == 0 and s == "W")]
    for i, seg in enumerate(segs):
        if isinstance(node, dict):
            if seg not in node:
                return ABSENT if i == len(segs) - 1 else UNRESOLVED
            node = node[seg]
        else:
            if not hasattr(node, seg):
                return ABSENT if i == len(segs) - 1 else UNRESOLVED
            node = getattr(node, seg)
        if node is None:
            return ABSENT if i == len(segs) - 1 else UNRESOLVED
    return node


def derive_join_gold(raw) -> tuple[str, list[str]]:
    """Derives `join_gold` from `W` with its derivation path: >=2 threads ->
    comorbidity; "no unified pathology" -> independent; otherwise unified.
    Raises `GoldNotDerivable` on uncovered cases.
    """
    ddx = resolve_path(raw, "W.adjudication.ddx")
    ddx = ddx if isinstance(ddx, dict) else {}
    threads = ddx.get("threads") or []
    diagnosis = str(ddx.get("diagnosis") or "")

    if len(threads) >= 2:
        return "comorbidity", [
            "W.adjudication.ddx.threads",
            f"rule=len(threads)={len(threads)}>=2 -> comorbidity",
        ]
    if NO_UNIFIED_PATHOLOGY in diagnosis:
        return "independent", [
            "W.adjudication.ddx.diagnosis",
            f"rule=diagnosis contains \"{NO_UNIFIED_PATHOLOGY}\" -> independent",
        ]
    if diagnosis:
        return "unified", [
            "W.adjudication.ddx.diagnosis",
            "W.adjudication.ddx.threads",
            "rule=single diagnosis with no multi-thread declaration -> unified",
        ]
    raise GoldNotDerivable(
        f"{world_id(raw)}: cannot derive join_gold from W ("
        f"threads={len(threads)} diagnosis={diagnosis!r}) -- no default value is supplied, a gap is a gap")


# ---------------------------------------------------------------- the remaining gold fields
#
# `urgency` is pinned by two biconditionals: red <=> red_flag_present, and
# green <=> not clinician_action_warranted. Within yellow/orange, severity is a
# primitive property of the condition.
RED_FLAG_URGENCY = "🔴"
NO_ACTION_URGENCY = "🟢"
MID_URGENCY = ("🟡", "🟠")


def derive_urgency(raw) -> tuple[str, list[str], bool]:
    """Derives `urgency`. Returns `(value, derivation, is_primitive)`; primitives
    are reported as such, never counted as derived.
    """
    adj = resolve_path(raw, "W.adjudication")
    adj = adj if isinstance(adj, dict) else {}
    ddx = adj.get("ddx") if isinstance(adj.get("ddx"), dict) else {}
    authored = ddx.get("urgency")

    if adj.get("red_flag_present") is True:
        return RED_FLAG_URGENCY, [
            "W.adjudication.red_flag_present",
            f"rule=red_flag_present -> {RED_FLAG_URGENCY} (🔴 is defined as emergent)",
        ], False
    if adj.get("clinician_action_warranted") is False:
        return NO_ACTION_URGENCY, [
            "W.adjudication.clinician_action_warranted",
            f"rule=no clinical action needed -> {NO_ACTION_URGENCY}",
        ], False
    if authored in MID_URGENCY:
        return authored, [
            "W.adjudication.ddx.urgency",
            "W.adjudication.red_flag_present",
            "W.adjudication.clinician_action_warranted",
            f"primitive=condition_registry.urgency in {MID_URGENCY}"
            "(the two biconditionals pin it to the middle band; severity within that band is decided by the condition)",
        ], True
    raise GoldNotDerivable(
        f"{world_id(raw)}: urgency={authored!r} is incompatible with both biconditionals "
        f"(red_flag={adj.get('red_flag_present')!r} "
        f"clinician_warranted={adj.get('clinician_action_warranted')!r})")


def derive_clinician_warranted(raw) -> tuple[bool, list[str], bool]:
    jg, jd = derive_join_gold(raw)
    if jg == "independent":
        return False, jd + ["rule=join_gold=independent (no unified pathology) -> no clinical action needed"], False
    return True, jd + [f"rule=join_gold={jg} (there is a treatable condition) -> clinical action needed"], False


def derive_outcome_label(raw) -> tuple[str, list[str], bool]:
    """Derives `outcome_label` on ddx items, where it means "is there a treatable
    condition to catch": from the diagnostic structure, not from `label_rule`.
    """
    jg, jd = derive_join_gold(raw)
    if jg == "independent":
        return "event_not_occurred", jd + [
            "rule=join_gold=independent -> no treatable condition -> event_not_occurred"], False
    return "event_occurred", jd + [
        f"rule=join_gold={jg} -> there is a treatable condition -> event_occurred"], False


def is_ddx_world(raw) -> bool:
    """Whether this world is a ddx item (has diagnostic ground truth); the
    derivers apply only there.
    """
    ddx = resolve_path(raw, "W.adjudication.ddx")
    return bool(isinstance(ddx, dict) and ddx.get("diagnosis"))


# field -> (deriver, path in W, applicability predicate). Unregistered fields
# are not covered by the self-consistency gate.
GOLD_DERIVERS: dict[str, tuple] = {
    "join_gold": (lambda raw: (*derive_join_gold(raw), False),
                  "W.adjudication.ddx.join_gold", is_ddx_world),
    "urgency": (derive_urgency, "W.adjudication.ddx.urgency", is_ddx_world),
    "clinician_action_warranted": (derive_clinician_warranted,
                                   "W.adjudication.clinician_action_warranted", is_ddx_world),
    "outcome_label": (derive_outcome_label, "W.outcome_label", is_ddx_world),
}

# Gold fields with no deriver -> their path in `W`. These are clinical
# primitives of the condition that no other field in `W` determines.
GOLD_PRIMITIVE_PATHS: dict[str, str] = {
    "diagnosis": "W.adjudication.ddx.diagnosis",
    "aliases": "W.adjudication.ddx.aliases",
    "threads": "W.adjudication.ddx.threads",
    "spec_id": "W.adjudication.ddx.spec_id",
    "tests": "W.adjudication.ddx.tests",
    "specialty": "W.adjudication.ddx.specialty",
    "red_flag_present": "W.adjudication.red_flag_present",
    "gold_drivers": "W.gold_drivers",
}

# ---------------------------------------------------------------- condition-registry projection
#
# A registry lookup can only catch a per-case copy diverging from its source,
# not a wrong source, so it is counted as a separate class from structural
# derivation. `field -> (key on the spec, path in W)`; `threads` goes through
# overlay instead.
REGISTRY_GOLD: dict[str, tuple[str, str]] = {
    "diagnosis": ("diagnosis", "W.adjudication.ddx.diagnosis"),
    "aliases": ("aliases", "W.adjudication.ddx.aliases"),
    "tests": ("tests", "W.adjudication.ddx.tests"),
    "specialty": ("specialty", "W.adjudication.ddx.specialty"),
    "red_flag_present": ("red_flag", "W.adjudication.red_flag_present"),
}


def _spec_of(w):
    sid = resolve_path(w, "W.adjudication.ddx.spec_id")
    if sid in (UNRESOLVED, ABSENT) or not sid:
        return None
    from .overlay import condition_registry
    return condition_registry().get(str(sid))


def derive_from_registry(w, field: str):
    """`(value, derivation)` from the condition registry; raises
    `GoldNotDerivable` when not found, never falling back to the authored value.
    """
    key, _path = REGISTRY_GOLD[field]
    spec = _spec_of(w)
    if spec is None:
        raise GoldNotDerivable(
            f"{world_id(w)}: {field} not found in the condition registry (spec_id="
            f"{resolve_path(w, 'W.adjudication.ddx.spec_id')!r}) -- a gap is a gap")
    if key not in spec:
        raise GoldNotDerivable(f"{world_id(w)}: {key} is not in the condition registry")
    deriv = ["W.adjudication.ddx.spec_id", f"registry=overlay.condition_registry()[spec_id].{key}"]
    # The insufficient tier places every symptom after T (`ddx.insufficient_triage`), so the
    # condition's red flag is not present at T.
    if field == "red_flag_present" and spec[key] and resolve_path(w, "W.adjudication.ddx.insufficient") is True:
        return False, deriv + ["W.adjudication.ddx.insufficient",
                               "rule=insufficient tier: red-flag presentation after T -> False"]
    return spec[key], deriv


def derive_threads(w):
    """`threads` via `overlay.threads_for` (declared in registry/threads)."""
    from .overlay import threads_for
    sid = resolve_path(w, "W.adjudication.ddx.spec_id")
    if sid in (UNRESOLVED, ABSENT) or not sid:
        raise GoldNotDerivable(f"{world_id(w)}: threads has no spec_id to look up")
    return (threads_for(str(sid), _spec_of(w)),
            ["W.adjudication.ddx.spec_id", "registry=overlay.threads_for(spec_id)"])


# field -> (extractor, authored path in W); the authored path is cross-checked
# by `check_gold_matches_world`.
REGISTRY_DERIVERS: dict[str, tuple] = {
    **{f: ((lambda w, _f=f: derive_from_registry(w, _f)), p)
       for f, (_k, p) in REGISTRY_GOLD.items()},
    "threads": (derive_threads, "W.adjudication.ddx.threads"),
}

# All gold fields the judges consume; coverage, the self-consistency gate and
# default-deny share this one set.
ALL_GOLD_FIELDS: tuple[str, ...] = tuple(
    dict.fromkeys(list(GOLD_DERIVERS) + list(REGISTRY_DERIVERS) + list(GOLD_PRIMITIVE_PATHS)))


def derive_gold(raw) -> dict[str, dict]:
    """Runs every applicable deriver. Returns `{field: {value, derivation, primitive}}`."""
    out: dict[str, dict] = {}
    for name, (fn, _path, applies) in GOLD_DERIVERS.items():
        if not applies(raw):
            continue
        value, derivation, primitive = fn(raw)
        out[name] = {"value": value, "derivation": list(derivation),
                     "primitive": bool(primitive)}
    return out




# ============================================================ gold_of
#
# Judges fetch gold only through `gold_of`: a derivable field is always served
# its derived value, a primitive is read from its registered path. On worlds
# with no authored values, `check_gold_matches_world` has nothing to compare;
# `gold_crosscheck_census` counts how many comparisons actually ran.

class GoldFieldUnregistered(IronLawViolation):
    """A judge fetched a gold field that isn't registered (default-deny)."""


_MISSING = object()


def gold_source(field: str) -> str:
    """Source class of a field: `derived` / `registry` / `primitive`. Raises if
    unregistered. Counts of the three classes are never merged.
    """
    if field in GOLD_DERIVERS:
        return "derived"
    if field in REGISTRY_DERIVERS:
        return "registry"
    if field in GOLD_PRIMITIVE_PATHS:
        return "primitive"
    raise GoldFieldUnregistered(
        f"gold field {field!r} is not registered -- judges may not fetch gold directly from W. "
        f"If it can be structurally derived, register it in GOLD_DERIVERS; if it can be looked up in the condition registry, register it in REGISTRY_GOLD; "
        f"only if neither applies does it go in GOLD_PRIMITIVE_PATHS, honestly labeled as a primitive (registered: "
        f"{sorted(set(GOLD_DERIVERS) | set(REGISTRY_DERIVERS) | set(GOLD_PRIMITIVE_PATHS))})")


def resolved_source(w, field: str) -> tuple[str, object]:
    """The source class of this field on this case, plus its extractor (`None` for
    `primitive`); the class depends on the case, not just the field.
    """
    if field in GOLD_DERIVERS:
        fn, _p, applies = GOLD_DERIVERS[field]
        if applies(w):
            return "derived", fn
    if field in REGISTRY_DERIVERS and _spec_of(w) is not None:
        return "registry", REGISTRY_DERIVERS[field][0]
    return "primitive", None


def _authored_path(field: str) -> str:
    """The path of this field in `W`."""
    if field in GOLD_PRIMITIVE_PATHS:
        return GOLD_PRIMITIVE_PATHS[field]
    if field in REGISTRY_DERIVERS:
        return REGISTRY_DERIVERS[field][1]
    return GOLD_DERIVERS[field][1]


def gold_of(w, field: str, *, default=_MISSING):
    """The entry point through which judges fetch gold. `w` may be a `RawCase` or
    the kernel's `vp`.

    A derivable field returns its derived value; a primitive is read from its
    path on `W`; if absent, returns `default`, or raises when none is given.
    """
    gold_source(field)
    kind, fn = resolved_source(w, field)
    if kind != "primitive":
        try:
            return fn(w)[0]
        except GoldNotDerivable:
            # A failed derivation falls back to the authored value only when a `default`
            # is given; otherwise it raises.
            if default is _MISSING:
                raise
            log.warning("[wq] %s: %s(%s) could not be obtained, falling back to reading authored", world_id(w), field, kind)
    return _primitive_at(w, field, _authored_path(field), default)


def _primitive_at(w, field: str, path: str, default):
    """Reads a primitive from its path on `W`; raises when absent and no `default`."""
    val = resolve_path(w, path)
    if val is UNRESOLVED or val is ABSENT:
        if default is not _MISSING:
            return default
        raise GoldNotDerivable(
            f"{world_id(w)}: primitive {field} is absent on W ({path}) -- a gap is a gap, no default value is supplied")
    return val


def gold_consumption_coverage(raws) -> dict:
    """Coverage per consumption point (field x case), by source class."""
    raws = list(raws)
    per: dict[str, dict] = {}
    for field in ALL_GOLD_FIELDS:
        row = per.setdefault(field, {"derived": 0, "registry": 0, "primitive": 0, "absent": 0,
                                     "registered": gold_source(field)})
        for raw in raws:
            kind, fn = resolved_source(raw, field)
            try:
                if kind == "derived":
                    _v, _d, prim = fn(raw)
                    row["primitive" if prim else "derived"] += 1
                    continue
                if kind == "registry":
                    fn(raw)
                    row["registry"] += 1
                    continue
            except GoldNotDerivable:
                row["absent"] += 1
                continue
            v = resolve_path(raw, _authored_path(field))
            row["absent" if v in (UNRESOLVED, ABSENT) else "primitive"] += 1
    pts = sum(r["derived"] + r["registry"] + r["primitive"] for r in per.values())
    drv = sum(r["derived"] for r in per.values())
    reg = sum(r["registry"] for r in per.values())
    return {"fields": per, "consumption_points": pts,
            "derived_points": drv, "registry_points": reg,
            "derived_share": round(drv / pts, 3) if pts else None,
            "registry_share": round(reg / pts, 3) if pts else None,
            "handwritten_share": round((pts - drv - reg) / pts, 3) if pts else None,
            "n_cases": len(raws)}


def gold_crosscheck_census(raws) -> dict:
    """How many gold comparisons the self-consistency gate actually made, and how
    many it skipped because the authored value was absent.
    """
    raws = list(raws)
    checked = skipped = 0
    for raw in raws:
        for field in ALL_GOLD_FIELDS:
            kind, _fn = resolved_source(raw, field)
            if kind == "primitive":
                continue
            a = resolve_path(raw, _authored_path(field))
            if a is UNRESOLVED or a is ABSENT:
                skipped += 1
            else:
                checked += 1
    return {"crosschecked": checked, "skipped_no_authored": skipped,
            "n_cases": len(raws)}


def check_label_rule_applicable(raw) -> list[str]:
    """Warning tier: a ddx item's `label_rule` states the weight-regain criterion,
    which is not the rule that produces its `outcome_label`. Harmless while the
    forecast judge is not mounted on ddx items.
    """
    ddx = resolve_path(raw, "W.adjudication.ddx")
    if not (isinstance(ddx, dict) and ddx.get("diagnosis")):
        return []
    tgt = (resolve_path(raw, "W.prediction_context") or {})
    tgt = tgt.get("target_event_type") if isinstance(tgt, dict) else None
    lr = resolve_path(raw, "W.label_rule")
    if isinstance(lr, dict) and lr.get("minimum_change_magnitude") and tgt == "weight_regain":
        return [f"{world_id(raw)}: this ddx item's label_rule states the weight-regain criterion "
                f"({lr.get('minimum_change_magnitude')}), while outcome_label is given by the diagnostic structure -- "
                f"the two are not the same rule (known gap; the forecast judge is not mounted on ddx items, so scoring is unaffected)"]
    return []


def check_derivation_resolvable(raw, derivation: list[str]) -> list[str]:
    bad = []
    for step in derivation or []:
        if step.startswith(("rule=", "template=", "primitive=", "probe=")):
            continue
        if not step.startswith("W."):
            bad.append(f"step {step!r} is neither a W path nor a rule/template note")
        elif resolve_path(raw, step) is UNRESOLVED:
            # `ABSENT` is legitimate evidence; only an unresolvable path is an error.
            bad.append(f"step {step!r} does not resolve on W (a typo, or the structure has changed)")
    return bad


def _json_norm(v):
    """Normalizes containers (tuple vs list) before comparison; case, whitespace
    and numeric precision are never normalized.
    """
    if isinstance(v, (list, tuple)):
        return [_json_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_norm(x) for k, x in v.items()}
    return v


def check_gold_matches_world(raw) -> list[str]:
    """Gold self-consistency gate: every gold field written in `W` must equal the
    value derived from `W` (or looked up in the condition registry).

    A change to `overlay.condition_registry()` therefore invalidates the gold of
    batches that depend on it; those cells `ABORT(iron_law)`.
    """
    bad: list[str] = []
    for name in ALL_GOLD_FIELDS:
        kind, fn = resolved_source(raw, name)
        if kind == "primitive":
            continue
        authored = resolve_path(raw, _authored_path(name))
        if authored is UNRESOLVED or authored is ABSENT:
            continue                          # this item type has no such field
        try:
            got = fn(raw)
            derived, deriv = got[0], got[1]
        except GoldNotDerivable as e:
            bad.append(str(e))
            continue
        if _json_norm(derived) != _json_norm(authored):
            bad.append(f"{world_id(raw)}: {name} was written as {authored!r}, "
                       f"but derives from W as {derived!r} (per {deriv[0]}) -- the gold standard contradicts itself")
    return bad


# ============================================================ Law 2 . Q-side injection ledger
#
# The ledger is kept outside `W` in a table keyed by `world_id`, so it never
# enters `asdict(raw)`, `world_truth_hash` or the `case` block of `cases.jsonl`.
_QINJ: dict[str, dict] = {}

#: The ledger's current scope (one job/batch). The same `case_id` can be a
#: different world in another job, so switching jobs requires `enter_scope`,
#: and overwriting a key registered under another scope raises.
_QINJ_SCOPE: list[str] = [""]


class InjectionScopeCollision(RuntimeError):
    """A registered world was overwritten across scopes."""


def enter_scope(scope: str) -> None:
    """Enters a new ledger scope (one job/batch) and clears the previous one."""
    _QINJ_SCOPE[0] = str(scope or "")
    _QINJ.clear()
    _QINJ_META.clear()


def current_scope() -> str:
    return _QINJ_SCOPE[0]


def register_injection(wid: str, manifest: dict) -> None:
    """Registers the Q-side injection ledger for a world. Re-registering within
    the same scope is allowed; across scopes it raises.
    """
    k = str(wid)
    prev = _QINJ_META.get(k)
    cur = _QINJ_SCOPE[0]
    if prev is not None and prev != cur:
        raise InjectionScopeCollision(
            f"{k}: the ledger already belongs to scope {prev!r}, and is about to be overwritten in {cur!r}.\n"
            f"The same case_id can be a different world in a different job -- "
            f"overwriting it would drop the previous job's ledger.\n"
            f"=> call `wq.enter_scope(<job_id or batch stamp>)` before switching jobs.")
    _QINJ[k] = dict(manifest or {})
    _QINJ_META[k] = cur


#: Which scope each key belongs to (for cross-scope overwrite detection).
_QINJ_META: dict[str, str] = {}


def injected_manifest(wid: str, *, required: bool = True) -> dict:
    """Reads the Q-side injection ledger; raises when unregistered unless
    `required=False`.
    """
    rec = _QINJ.get(str(wid))
    if rec is None:
        if required:
            raise MissingQuestionRecord(
                f"{wid}: no Q-side injection ledger registered -- either register_injection was not called during item generation, "
                f"or the question segment was not brought along when the batch was read back (registered so far: {len(_QINJ)})")
        return {}
    return rec


def has_injection(wid: str) -> bool:
    return str(wid) in _QINJ






def check_injection_coverage(world_ids) -> list[str]:
    """Batch-level assertion: every emitted world must have a Q-side ledger."""
    missing = sorted(str(w) for w in world_ids if not has_injection(str(w)))
    return [f"{len(missing)} case(s) missing a Q-side injection ledger: {missing[:5]}"] if missing else []


def manifest_of_row(row: dict) -> dict:
    """The injection ledger of a persisted `cases.jsonl` row:
    `row["question"]["injected_manifest"]`, or
    `row["case"]["adjudication"]["injected_event_manifest"]` in batches that
    store it in `W`.
    """
    q = (row.get("question") or {}).get("injected_manifest")
    if q is not None:
        return q
    return ((row.get("case") or {}).get("adjudication") or {}).get("injected_event_manifest") or {}


