"""overlay.py -- declarations the kernel specs need but do not carry: a closed context
vocabulary, comorbidity threads, added independent and comorbid conditions, alias and
specialty tables, rivals, and the synthesized `condition_registry` that generation and
scoring share.

The context vocabulary is default-deny and delete-only: an unregistered context is dropped,
and a kept form must be a substring of the author's text (`check_vocab`), so haenv never rewrites
facts on the author's behalf.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml
from .regpath import load_registry as _load_registry

import functools as _functools
import logging

log = logging.getLogger("haenv.overlay")

# ============================================================ facet vocabulary
# Emittable facets: describe what was observed
MEASURE = "measure"    # numeric value / lab result / device reading / signal direction
NEG = "neg"            # a negative finding (ruling out a possibility)
SIGN = "sign"          # an observable sign, witnessed event, life fact, history item
COURSE = "course"      # course of illness and response to management: what was done, what happened
THERAPY = "therapy"    # observable facts on the treatment timeline (dose/start/switch), no causal claim

# Blocked facets: describe what it means
ATTR = "attr"          # attribution ("related to X" / "common after X")
INTERP = "interp"      # an interpretive label ("metabolic resistance" / "mechanical" / "nonspecific")
JOIN = "join"          # join structure -- the answer to join_gold
GOLD = "gold"          # directly the gold answer: outcome / driver / diagnosis name / recommended action

EMITTABLE = frozenset({MEASURE, NEG, SIGN, COURSE, THERAPY})
BLOCKED = frozenset({ATTR, INTERP, JOIN, GOLD})

# ============================================================ (1) closed context vocabulary
# original text -> (facet, kept form). Kept form "" = drop the entry; None = keep as-is.
# Covers raw_case symptom contexts only; haenv's own benign/life events do not go through it.
from .registry import load_context_vocab as _load_context_vocab      # noqa: E402
CONTEXT_VOCAB = _load_context_vocab()



def facet_of_emitted_context(ctx: str) -> str:
    """Facet of an already-emitted context, looked up by its emitted form. Shared by
    `events.cap_context_facets` and `gates.case_features` so both bucket it the same way."""
    for src, (facet, keep) in CONTEXT_VOCAB.items():
        if facet in EMITTABLE and (keep if keep is not None else src) == ctx:
            return facet
    return "pool"


def gate_context(ctx: str) -> tuple[str, str, str]:
    """Pass a raw_case context through the closed vocabulary; returns (emitted form, facet,
    reason). Unregistered contexts are dropped."""
    ctx = (ctx or "").strip()
    if not ctx:
        return "", "", "empty"
    hit = CONTEXT_VOCAB.get(ctx)
    if hit is None:
        return "", "undeclared", "未登记于 overlay.CONTEXT_VOCAB(默认拒绝)"
    facet, keep = hit
    if facet in BLOCKED:
        return "", facet, f"facet={facet} 不可发射"
    out = ctx if keep is None else keep
    return out, facet, ("原样" if keep is None else f"按声明保留子串 {out!r}")


def emitted_forms() -> frozenset[str]:
    """All context forms the closed vocabulary allows to appear in solver
    input (as-is forms plus declared kept substrings)."""
    out = set()
    for src, (facet, keep) in CONTEXT_VOCAB.items():
        if facet in EMITTABLE:
            out.add(src if keep is None else keep)
    return frozenset(out)


def check_vocab(contexts) -> list[str]:
    """Self-check of the vocabulary (empty list = pass): kept forms are substrings, facets
    are known, and every context in `contexts` is registered."""
    problems: list[str] = []
    for src, (facet, keep) in CONTEXT_VOCAB.items():
        if facet not in EMITTABLE | BLOCKED:
            problems.append(f"unknown facet {facet!r}: {src!r}")
        if keep and keep not in src:
            problems.append(f"kept form is not a substring of the source (that would rewrite it): {src!r} -> {keep!r}")
        if facet in BLOCKED and keep:
            problems.append(f"facet={facet} is blocked but declares a kept form: {src!r}")
    for c in contexts or ():
        c = (c or "").strip()
        if c and c not in CONTEXT_VOCAB:
            problems.append(f"unregistered context (dropped by default; assign a facet explicitly): {c!r}")
    return problems


# ============================================================ (2) comorbidity thread declarations
# One alias set per disease line, so "were both lines named" is not judged against a single
# flat alias list. Aliases accept only affirmative phrasing.
from .registry import load_threads as _load_threads             # noqa: E402
THREADS: dict[str, tuple[dict, ...]] = _load_threads()


# ============================================================ (3) supplementary independent items
# The anti-shortcut scan needs at least two `independent` items. These items must match
# disease items in context density, clinical register and weight distribution, or those
# dimensions become shortcuts; content lives in `registry/conditions_independent.yaml`.
from .registry import load_conditions as _load_conditions      # noqa: E402
HAENV_DDX_SPECS: dict[str, dict] = _load_conditions()


# ============================================================ (4) composed comorbidity items
# Pairs of kernel unified specs. A pair must be mutually non-causal (enforced by
# `CAUSAL_CLUSTERS`), non-overlapping in presentation, and non-conflicting. Each line
# contributes its 2 most discriminating symptoms, interleaved by day.
from .registry import load_causal_clusters as _load_clusters    # noqa: E402
CAUSAL_CLUSTERS: tuple[tuple[str, ...], ...] = _load_clusters()


def causal_conflict(a: str, b: str) -> str | None:
    """If two specs are in the same cluster, returns that cluster (the
    reason they shouldn't be paired); otherwise None."""
    for cl in CAUSAL_CLUSTERS:
        if a in cl and b in cl:
            return "/".join(cl)
    return None


from .registry import load_comorbid_pairs as _load_pairs        # noqa: E402
HAENV_COMORBID_PAIRS: dict[str, dict] = _load_pairs(clusters=CAUSAL_CLUSTERS)

HAENV_DDX_DAYS = (14, 35, 56, 77)


# Clinical severity ordering, mild to severe. Never `max()` the emoji strings: code-point
# order is the reverse of severity.
URGENCY_ORDER = ("🟢", "🟡", "🟠", "🔴")


def urgency_max(*levels: str) -> str:
    """Returns the more urgent of the given levels. Raises on an unknown
    level -- never silently ordered by code point."""
    idx = []
    for x in levels:
        x = str(x or "").strip()
        if x not in URGENCY_ORDER:
            raise ValueError(f"unknown urgency {x!r}; registered: {URGENCY_ORDER}")
        idx.append(URGENCY_ORDER.index(x))
    return URGENCY_ORDER[max(idx)] if idx else URGENCY_ORDER[0]


class RegistryInvalid(RuntimeError):
    """The active registry is internally inconsistent -- do not emit.
    See `validate_registry`."""


@_functools.lru_cache(maxsize=1)
def validate_registry() -> dict[str, int]:
    """Run the registry validators once per process before item generation; raises
    `RegistryInvalid` on any problem. Alias collisions are checked on the base specs only,
    since a comorbid item's name contains its components' aliases by construction."""
    _k = _kernel_specs()
    base = {**_k, **HAENV_DDX_SPECS}                     # base set: fed to the alias/specialty vocabulary checks
    expanded = {**base, **haenv_comorbid_specs(_k), **haenv_independent_specs()}
    counts: dict[str, int] = {}
    problems: list[str] = []
    from . import demographics as _dg
    from . import registry as _rg0
    for name, hits in (("check_alias_collisions", check_alias_collisions(base)),
                       ("check_specialty_vocab", check_specialty_vocab(base)),
                       ("check_rival_alias_disjoint", check_rival_alias_disjoint(expanded)),
                       ("check_take_within",
                        _rg0.check_take_within(_rg0.load_comorbid_pairs(), expanded)),
                       ("profile_fields_are_answer_neutral",
                        _dg.profile_fields_are_answer_neutral(expanded))):
        counts[name] = len(hits)
        if hits:
            problems.append(f"{name}: {len(hits)} problem(s): {'; '.join(str(h) for h in hits[:2])[:200]}")
    from . import registry as _rg
    try:
        _f = _rg.load_findings()
        import yaml as _y
        _rg.validate_discriminators(
            _f, _rg.condition_findings_for_case(_f),
            _load_registry("rivals.yaml")["rivals"])
        counts["validate_discriminators"] = 0
    except Exception as _e:                              # noqa: BLE001
        counts["validate_discriminators"] = 1
        problems.append(f"validate_discriminators: {type(_e).__name__}: {str(_e)[:200]}")
    # Reported, not enforced: mutual rivals between comorbid items are expected, and the
    # count uses bare `alias_hit` (an upper bound; scoring uses `tracks.alias_hit_asserted`).
    counts["check_rival_not_other_gold"] = len(check_rival_not_other_gold(expanded))
    if problems:
        raise RegistryInvalid(
            "registry is inconsistent, nothing emitted:\n  "
            + "\n  ".join(problems))
    return counts


def haenv_comorbid_specs(kernel_specs: dict) -> dict[str, dict]:
    """Expands `HAENV_COMORBID_PAIRS` into specs shaped like the kernel's
    `DDX_SPECS`."""
    out: dict[str, dict] = {}
    for sid, cfg in HAENV_COMORBID_PAIRS.items():
        a, b = cfg["pair"]
        if a not in kernel_specs or b not in kernel_specs:
            log.warning("[overlay] %s references a spec the kernel does not have %s; skipped", sid, (a, b))
            continue
        sa, sb = kernel_specs[a], kernel_specs[b]
        pick_a = [sa["symptoms"][i] for i in cfg["take"][0]]
        pick_b = [sb["symptoms"][i] for i in cfg["take"][1]]
        seq = [pick_a[0], pick_b[0], pick_a[1], pick_b[1]]
        out[sid] = {
            "diagnosis": f"{sa['diagnosis']} + {sb['diagnosis']}",
            "aliases": list(dict.fromkeys(list(sa.get("aliases") or [])
                                          + list(sb.get("aliases") or []))),
            "join_gold": "comorbidity",
            # Weight follows the first source spec only: a composed item is one patient
            # plus a second disease's symptom text, not a world generated from both.
            "weight": sa["weight"],
            "symptoms": [(d, s[1], s[2]) for d, s in zip(HAENV_DDX_DAYS, seq)],
            "tests": list(sa.get("tests") or []) + list(sb.get("tests") or []),
            "specialty": list(dict.fromkeys(list(sa.get("specialty") or [])
                                            + list(sb.get("specialty") or []))),
            "urgency": urgency_max(sa.get("urgency", "🟢"), sb.get("urgency", "🟢")),
            "red_flag": bool(sa.get("red_flag")) or bool(sb.get("red_flag")),
            "clinician_warranted": True,
            "outcome_label": "event_occurred",             # a diagnosable disease = "occurred" for this item type
            "source": "haenv_overlay",
            "_threads": ({"name": cfg["threads"][0], "aliases": tuple(sa.get("aliases") or ())},
                         {"name": cfg["threads"][1], "aliases": tuple(sb.get("aliases") or ())}),
        }
        # Leak-scan words are inherited like aliases; set only when a source has them.
        if sa.get("leak_aliases") or sb.get("leak_aliases"):
            out[sid]["leak_aliases"] = list(dict.fromkeys(
                list(sa.get("leak_aliases") or sa.get("aliases") or [])
                + list(sb.get("leak_aliases") or sb.get("aliases") or [])))
    return out


def haenv_independent_specs() -> dict[str, dict]:
    """Expands `HAENV_DDX_SPECS` into specs shaped like the kernel's
    `DDX_SPECS` (for merging into ddx.ddx_case_specs)."""
    out: dict[str, dict] = {}
    for sid, spec in HAENV_DDX_SPECS.items():
        out[sid] = {
            "diagnosis": "无统一病理:彼此独立的良性事件",
            "aliases": ["独立", "良性", "无统一", "unknown"],
            "join_gold": "independent",
            "weight": spec["weight"],
            "symptoms": list(spec["symptoms"]),
            "tests": [], "specialty": [], "urgency": "🟢",
            "red_flag": False, "clinician_warranted": False,
            "outcome_label": spec["outcome_label"],
            "source": "haenv_overlay",         # distinguishes from kernel specs: these cases are hand-written by this repo
        }
    return out


def threads_for(spec_id: str | None, spec: dict | None = None) -> tuple[dict, ...] | None:
    """This example's thread declaration, or None when undeclared (scoring then reports
    thread_decl=missing). A composed example's own `_threads` take precedence."""
    own = (spec or {}).get("_threads")
    if own:
        return tuple({"name": t["name"], "aliases": list(t["aliases"])} for t in own)
    return THREADS.get(str(spec_id or "")) or None


def check_threads(specs: dict) -> list[str]:
    """Every spec with join_gold=comorbidity must have a thread declaration
    with >=2 entries. Conversely: every declared spec must actually
    exist."""
    problems: list[str] = []
    for sid, spec in (specs or {}).items():
        if spec.get("join_gold") == "comorbidity" and not threads_for(sid, spec):
            problems.append(f"{sid}: join_gold=comorbidity but no threads are declared; "
                            f"the judge falls back to flat literal alias matching, which gives false negatives")
    for sid, th in THREADS.items():
        if specs and sid not in specs:
            problems.append(f"{sid}: threads are declared but the kernel no longer has this spec")
        if len(th) < 2:
            problems.append(f"{sid}: comorbidity declares only {len(th)} thread(s)")
        for t in th:
            if not t.get("name") or not t.get("aliases"):
                problems.append(f"{sid}: thread declaration lacks name/aliases: {t}")
    return problems


# ============================================================ (4b) alias exclusion set
# A short alias can be a substring of a different disease's name ("polycystic" in
# "polycystic kidney disease"). Scoring side only: the leak scan keeps the raw vocabulary.
from .registry import load_alias_excludes as _load_alias_excludes      # noqa: E402
ALIAS_EXCLUDES = _load_alias_excludes()



def alias_excluded_at(text: str, alias: str, pos: int) -> bool:
    """Whether this occurrence of `alias` at `pos` lies inside an exclusion phrase
    (positional containment, not "the phrase appears somewhere in the text")."""
    low, a = str(text or "").lower(), str(alias or "").lower()
    for bad in ALIAS_EXCLUDES.get(a, ()) :
        b, i = bad.lower(), 0
        while True:
            j = low.find(b, i)
            if j < 0:
                break
            if j <= pos and pos + len(a) <= j + len(b):    # this hit is fully contained within `bad`
                return True
            i = j + 1
    return False


# Cross-example overlaps that share a genuine disease axis; these examples are scored by
# thread coverage, not by dx alias.
from .registry import load_alias_overlap_ok as _load_alias_overlap_ok      # noqa: E402
ALIAS_OVERLAP_OK = _load_alias_overlap_ok()



def check_alias_collisions(specs: dict) -> list[str]:
    """Drift gate: every cross-example alias substring conflict must be declared in
    `ALIAS_EXCLUDES` or `ALIAS_OVERLAP_OK`."""
    problems: list[str] = []
    names = {k: [str(v.get("diagnosis") or "")] + [str(a) for a in (v.get("aliases") or [])]
             for k, v in (specs or {}).items()}
    for ka, la in names.items():
        for a in la[1:]:
            a = a.strip()
            if not a:
                continue
            for kb, lb in names.items():
                if kb == ka:
                    continue
                for b in lb:
                    if a != b and a.lower() in b.lower() \
                       and (ka, kb) not in ALIAS_OVERLAP_OK \
                       and not any(x.lower() in b.lower() for x in ALIAS_EXCLUDES.get(a.lower(), ())):
                        problems.append(f"alias {a!r} of {ka} is a substring of {b!r} of {kb} "
                                        f"and is not declared in ALIAS_EXCLUDES: answering {kb} would score as {ka}")
    return problems

# ============================================================ (5) bilingual specialty synonyms (scoring side)
# `specialty_ok` substring-matches the Chinese gold specialty; these English forms keep a
# correct English answer from scoring 0. Every gold specialty must be registered.
from .registry import load_specialty_synonyms as _load_specialty_synonyms      # noqa: E402
SPECIALTY_SYNONYMS = _load_specialty_synonyms()



def specialty_forms(seg: str) -> tuple[str, ...]:
    """All acceptable spellings of a gold specialty segment (the
    original plus registered English synonyms, all lowercased)."""
    return (seg.lower(),) + tuple(x.lower() for x in SPECIALTY_SYNONYMS.get(seg, ()))


def specialty_hit(said_blob: str, gold_specialties) -> bool:
    """Whether any gold specialty, in Chinese or English, appears in the answer."""
    blob = str(said_blob or "").lower()
    for x in (gold_specialties or []):
        for seg in _spec_segs(str(x)):
            if any(f in blob for f in specialty_forms(seg)):
                return True
    return False


def _spec_segs(s: str) -> list[str]:
    import re
    return [g.strip() for g in re.split(r"[/·、,,()()]", s) if g.strip()]


def check_specialty_vocab(specs: dict) -> list[str]:
    """Drift gate: every gold specialty segment has a registered English form."""
    missing: list[str] = []
    for sid, sp in (specs or {}).items():
        for x in (sp.get("specialty") or []):
            for seg in _spec_segs(str(x)):
                if seg not in SPECIALTY_SYNONYMS:
                    missing.append(f"{sid}: specialty segment {seg!r} has no registered English synonyms")
    return sorted(set(missing))

# ============================================================ (6) red flags and urgency must be self-consistent
# A red flag raises urgency to the highest level; otherwise `URGENCY_ACTION` would mark the
# correct escalation (A5) wrong on specs such as JD-PHEO whose urgency is orange.
RED_FLAG_URGENCY = "🔴"


#: Tests the literature treats as second-line or conditional, marked optional so they leave
#: the recall denominator. {spec_id: {original text: annotated text}}; one citation each.
LITERATURE_OPTIONAL_TESTS: dict[str, dict[str, str]] = {
    # ACG 2019 / EASL 2022: C282Y homozygosity is diagnostic; liver MRI/biopsy is conditional.
    "JD-HEMO": {"肝铁定量MRI/活检": "肝铁定量MRI/活检(必要时:非C282Y同合子或疑肝病)"},
    # Fifth International Workshop on PHPT (JBMR 2022): urine calcium and renal imaging
    # assess complications and FHH, not the diagnosis.
    "JD-PHPT": {"肾脏超声/尿钙": "肾脏超声/尿钙(必要时:评估并发症与与FHH鉴别)"},
}


def apply_literature_optional_tests(specs: dict) -> dict:
    """Apply `LITERATURE_OPTIONAL_TESTS`; returns a new dict. `judges._gold_test_optional`
    excludes marked tests from the recall denominator but not from precision."""
    out = {}
    for k, v in (specs or {}).items():
        rep = LITERATURE_OPTIONAL_TESTS.get(k)
        if rep and v.get("tests"):
            _new = [rep.get(str(t), t) for t in v["tests"]]
            if _new != list(v["tests"]):
                v = {**v, "tests": _new,
                     "tests_annotated_by": "overlay.apply_literature_optional_tests(DG-002)"}
        out[k] = v
    return out


def apply_red_flag_urgency(specs: dict) -> dict:
    """Raises the urgency of any `red_flag=True` spec to the highest level.
    Returns a new dict, does not mutate the argument."""
    out = {}
    for k, v in (specs or {}).items():
        if v.get("red_flag") and v.get("urgency") != RED_FLAG_URGENCY:
            v = {**v, "urgency": RED_FLAG_URGENCY,
                 "urgency_overridden_by": "overlay.apply_red_flag_urgency(red_flag=True)"}
        out[k] = v
    return out


# ============================================================ (6b) synthesized condition registry (single source)
# Generation and scoring both look conditions up here by `spec_id`. Editing it expires the
# gold of batches that depend on it (`wq.check_gold_matches_world` aborts those cells).
_REGISTRY_CACHE: dict | None = None


from .registry import load_unified_conditions as _load_unified      # noqa: E402

HAENV_UNIFIED_SPECS: dict[str, dict] = _load_unified()


def haenv_unified_specs(*, include_draft: bool = False) -> dict[str, dict]:
    """haenv's own unified specs in expanded form. Entries with
    `clinical_review == "blocked"` or `draft` are excluded unless `include_draft`."""
    _META = ("draft", "clinical_review")
    return {k: {kk: vv for kk, vv in v.items() if kk not in _META}
            for k, v in HAENV_UNIFIED_SPECS.items()
            if include_draft
            or (str(v.get("clinical_review") or "") != "blocked" and not v.get("draft"))}


def apply_age_limits(specs: dict) -> dict:
    """Attach `registry/condition_age_limits.yaml` ranges; composed items take the
    intersection of both components, and a spec's own tighter bound wins."""
    from .registry import _load_yaml
    lim = (_load_yaml("condition_age_limits.yaml").get("limits") or {})
    if not lim:
        return specs
    out = dict(specs)
    for sid, spec in list(out.items()):
        src = [lim.get(sid) or {}]
        cfg = HAENV_COMORBID_PAIRS.get(sid)
        if cfg:                                   # composed item: inherit from both components
            src += [lim.get(x) or {} for x in cfg["pair"]]
        los = [int(s["min_age"]) for s in src if s.get("min_age") is not None]
        his = [int(s["max_age"]) for s in src if s.get("max_age") is not None]
        if spec.get("min_age") is not None:
            los.append(int(spec["min_age"]))
        if spec.get("max_age") is not None:
            his.append(int(spec["max_age"]))
        if los or his:
            new = dict(spec)
            if los:
                new["min_age"] = max(los)
            if his:
                new["max_age"] = min(his)
            out[sid] = new
    return out


def strip_author_annotations(specs: dict) -> dict:
    """Strip author annotations (e.g. a mnemonic in parentheses) from symptom text, per
    `registry/symptom_annotations.yaml`. Unregistered parentheticals are kept: deleting
    prompt content by default costs more than missing one."""
    import re as _re
    tbl = (_load_registry("symptom_annotations.yaml") or {}).get("annotations") or {}
    strip = {k for k, v in tbl.items() if str((v or {}).get("action")) == "strip"}
    if not strip:
        return specs
    pats = [_re.compile(r"\s*[(（]" + _re.escape(k) + r"[)）]") for k in sorted(strip)]

    def _clean(s: str) -> str:
        for p in pats:
            s = p.sub("", s)
        return s

    out: dict = {}
    for cid, spec in (specs or {}).items():
        syms = spec.get("symptoms")
        if not syms:
            out[cid] = spec
            continue
        new = []
        for item in syms:
            it = list(item)
            for i in (1, 2):                      # (day, text, context)
                if len(it) > i and isinstance(it[i], str):
                    it[i] = _clean(it[i])
            new.append(tuple(it))
        out[cid] = {**spec, "symptoms": new}
    return out


# ============================================================ alias supplements
# `registry/condition_aliases.yaml`, applied inside `condition_registry` before comorbid
# pairs are composed. `synonyms` extend `aliases` (scoring and leak scan); `leak_only`
# words feed the leak scanner only. A synonym that also names a registered look-alike is
# rejected. Read via `registry._load_yaml` so gold does not vary with installed plugins.

ALIAS_SUPPLEMENTS_FILE = "condition_aliases.yaml"
_ALIAS_ENTRY_KEYS = frozenset({"term", "why"})
_ALIAS_LIST_KEYS = frozenset({"synonyms", "leak_only"})


class AliasSupplementError(ValueError):
    """`registry/condition_aliases.yaml` failed load-time validation."""


def load_alias_supplements() -> dict[str, dict[str, tuple[str, ...]]]:
    """`{spec_id: {"synonyms": (...), "leak_only": (...)}}`, shape-validated; checks that
    need the specs are in `apply_alias_supplements`."""
    from .registry import _load_yaml
    doc = _load_yaml(ALIAS_SUPPLEMENTS_FILE)
    sup = doc.get("supplements")
    if not isinstance(sup, dict):
        raise AliasSupplementError(f"{ALIAS_SUPPLEMENTS_FILE}: top-level `supplements` must be a mapping")
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for sid, body in sup.items():
        if not isinstance(body, dict) or not set(body) <= _ALIAS_LIST_KEYS or not body:
            raise AliasSupplementError(
                f"{ALIAS_SUPPLEMENTS_FILE}:{sid}: allowed keys are {sorted(_ALIAS_LIST_KEYS)}, got {body!r}")
        lists: dict[str, tuple[str, ...]] = {}
        for lk in sorted(_ALIAS_LIST_KEYS):
            terms = []
            for i, e in enumerate(body.get(lk) or ()):
                if not isinstance(e, dict) or set(e) != _ALIAS_ENTRY_KEYS:
                    raise AliasSupplementError(
                        f"{ALIAS_SUPPLEMENTS_FILE}:{sid}.{lk}[{i}]: each entry needs exactly term + why, got {e!r}")
                t, why = str(e["term"]).strip(), str(e["why"]).strip()
                if not t or not why:
                    raise AliasSupplementError(
                        f"{ALIAS_SUPPLEMENTS_FILE}:{sid}.{lk}[{i}]: term and why must be non-empty; "
                        f"every alias needs a stated reason")
                terms.append(t)
            lists[lk] = tuple(terms)
        both = set(lists["synonyms"]) & set(lists["leak_only"])
        if both:
            raise AliasSupplementError(
                f"{ALIAS_SUPPLEMENTS_FILE}:{sid}: {sorted(both)} appear in both synonyms and leak_only; "
                f"a term either names the condition or only points to it, not both")
        out[str(sid)] = lists
    return out


def apply_alias_supplements(specs: dict, supplements: dict | None = None) -> dict:
    """Copy of `specs` with supplements applied: `aliases` gains `synonyms`, and each
    supplemented spec gains `leak_aliases` (= aliases + `leak_only`). Raises on an unknown
    spec id, or a synonym that also names one of the spec's rivals."""
    from .events import alias_hit
    sup = load_alias_supplements() if supplements is None else supplements
    missing = sorted(set(sup) - set(specs))
    if missing:
        raise AliasSupplementError(
            f"{ALIAS_SUPPLEMENTS_FILE}: {missing} are not in the condition registry; supplements for unknown conditions are dead entries")
    out = dict(specs)
    for sid, lists in sup.items():
        spec = dict(specs[sid])
        syn = list(lists.get("synonyms") or ())
        for r in (rivals_for(sid, spec) or ()) if syn else ():
            blob = " ".join([str(r.get("name") or "")] + [str(a) for a in (r.get("aliases") or ())])
            hit = alias_hit(blob, syn)
            if hit:
                raise AliasSupplementError(
                    f"{ALIAS_SUPPLEMENTS_FILE}:{sid}: synonyms {hit} also match the registered rival {r.get('name')!r}"
                    f"; an umbrella term belongs in leak_only")
        al = list(dict.fromkeys(list(spec.get("aliases") or []) + syn))
        spec["aliases"] = al
        spec["leak_aliases"] = list(dict.fromkeys(al + list(lists.get("leak_only") or ())))
        out[sid] = spec
    return out


def _kernel_specs() -> dict:
    """Kernel `DDX_SPECS` with alias supplements: the one starting point for
    `condition_registry` and `validate_registry`."""
    import joint_scenarios as JS                     # kernel (`core/`); read here, not edited
    return apply_alias_supplements(JS.DDX_SPECS)


def leak_aliases_for(spec_id: str | None, aliases) -> list[str]:
    """Aliases the leak scanner uses for `spec_id`: the registry's `leak_aliases` unioned
    with the case's own `aliases`. Scoring never reads this."""
    base = [str(a) for a in (aliases or [])]
    if not spec_id:
        return base
    # The cached registry suffices: draft specs carry no `leak_aliases`.
    spec = condition_registry().get(str(spec_id)) or {}
    extra = [str(a) for a in (spec.get("leak_aliases") or [])]
    return list(dict.fromkeys(base + extra))


def condition_registry(*, fresh: bool = False, include_draft: bool = False) -> dict:
    """The synthesized condition registry `{spec_id: spec}`, shared by item generation and
    scoring."""
    global _REGISTRY_CACHE
    import joint_scenarios as JS                     # kernel (core/; changing it moves world_sha)
    # Draft branch before the cache, so `include_draft=True` never returns the cached set.
    if include_draft:                                # draft conditions never enter the cache, to avoid contaminating later calls
        _k = _kernel_specs()
        return strip_author_annotations(apply_age_limits(
            apply_literature_optional_tests(apply_red_flag_urgency({
                **_k, **haenv_unified_specs(include_draft=True),
                **haenv_comorbid_specs(_k), **haenv_independent_specs()}))))
    if _REGISTRY_CACHE is not None and not fresh:
        return _REGISTRY_CACHE
    _k = _kernel_specs()
    specs = {**_k, **haenv_unified_specs(),
             **haenv_comorbid_specs(_k), **haenv_independent_specs()}
    for p in check_red_flag_urgency(specs):
        log.warning("[overlay] gold contradicts itself (corrected by apply_red_flag_urgency): %s", p)
    _REGISTRY_CACHE = strip_author_annotations(apply_age_limits(
        apply_literature_optional_tests(apply_red_flag_urgency(specs))))
    return _REGISTRY_CACHE


def adjudication_from_condition(spec_id: str) -> dict:
    """The world's `adjudication` block for a condition: the few fields structural
    derivation needs; the rest of gold is looked up by `spec_id` at scoring time."""
    spec = condition_registry().get(str(spec_id))
    if spec is None:
        raise KeyError(f"ddx_condition={spec_id!r} is not in the condition registry; the gold cannot be derived")
    return {"ddx": {"spec_id": str(spec_id), "diagnosis": spec["diagnosis"]},
            "red_flag_present": bool(spec.get("red_flag")),
            "clinician_action_warranted": bool(spec.get("clinician_warranted"))}


def check_red_flag_urgency(specs: dict) -> list[str]:
    """Drift gate: a red flag with urgency not at the highest level = the
    gold label is self-contradictory, and the scorer will mark a correct
    escalation wrong."""
    bad = []
    for k, v in (specs or {}).items():
        if v.get("red_flag") and v.get("urgency") != RED_FLAG_URGENCY:
            bad.append(f"{k}: red_flag=True but urgency={v.get('urgency')} "
                       f"(should be {RED_FLAG_URGENCY}; otherwise a correct A5 answer is marked wrong)")
    return sorted(bad)


# ============================================================ (7) near-miss rivals
# Rivals raise the bar from naming the disease to ruling out a comparable near-miss.
# Declarations live in `registry/rivals.yaml`.
from .registry import load_rivals as _load_rivals                # noqa: E402

RIVALS: dict[str, tuple[dict, ...]] = _load_rivals()


def spec_id_of(obj, *, allow_case_id_lookup: bool = True) -> tuple[str, str]:
    """A case's `spec_id` and the path it came from, tried in order: `adjudication`,
    `latent.ddx_spec_id`, kernel premise, then a registry lookup by `case_id` (logged,
    since it bypasses `cases.jsonl`; disable with `allow_case_id_lookup=False`).
    Returns `("", "none")` rather than guessing."""
    adj = getattr(obj, "adjudication", None) or {}
    if isinstance(adj, dict):
        sid = ((adj.get("ddx") or {}) if isinstance(adj.get("ddx"), dict) else {}).get("spec_id")
        if sid:
            return str(sid), "adjudication.ddx.spec_id"
    lat = getattr(obj, "latent", None) or {}
    if isinstance(lat, dict) and lat.get("ddx_spec_id"):
        return str(lat["ddx_spec_id"]), "latent.ddx_spec_id"
    lp = getattr(obj, "latent_premise", None) or {}
    pb = (lp.get("patient_basics") or {}) if isinstance(lp, dict) else {}
    if pb.get("spec_id"):
        return str(pb["spec_id"]), "latent_premise.patient_basics.spec_id"
    if getattr(obj, "condition", ""):
        return str(obj.condition), "raw.condition"
    cid = getattr(obj, "case_id", "")
    if allow_case_id_lookup and cid:
        from .registry import load_case_ids
        sid = {v: k for k, v in load_case_ids().items()}.get(cid, "")
        if sid:
            log.warning("[overlay] %s: spec_id is only recoverable by looking case_id up in the registry -- "
                        "the case does not carry its own condition, so the batch is not self-contained: "
                        "a registry edit would silently change this case's tool answers while the cases.jsonl fingerprint still matches", cid)
            return str(sid), "case_id_lookup"
    return "", "none"


def rivals_for(spec_id: str | None, spec: dict | None = None) -> tuple[dict, ...]:
    """Near-misses this example must rule out; a composed example takes the union of its
    sources'. Empty when unregistered (scored as not applicable)."""
    if not spec_id:
        return ()
    own = RIVALS.get(spec_id)
    if own:
        return own
    cfg = HAENV_COMORBID_PAIRS.get(spec_id)
    if cfg:
        out, seen = [], set()
        for base in cfg["pair"]:
            for r in RIVALS.get(base, ()):
                if r["name"] not in seen:
                    seen.add(r["name"])
                    out.append(r)
        return tuple(out)
    return ()


def check_rival_alias_disjoint(specs: dict) -> list[str]:
    """Gate: a rival must not match its own spec's gold aliases under the scoring matcher
    (`tracks.alias_hit_asserted`), or "name gold and rule out the rival" is a tautology."""
    from .tracks import alias_hit_asserted
    bad: list[str] = []
    for sid, spec in (specs or {}).items():
        gold = list(spec.get("aliases") or [])
        for r in rivals_for(sid, spec):
            blob = f"{r['name']} {' '.join(r['aliases'])}"
            if alias_hit_asserted(blob, gold):
                bad.append(f"{sid}: rival {r['name']!r} would be scored as a hit on the gold {gold}; "
                           f"naming the rival would equal naming the gold, making this dimension a tautology"
                           f" (register it in vocab_alias_excludes.yaml, or choose another rival)")
    return sorted(bad)


def check_rival_not_other_gold(specs: dict) -> list[str]:
    """Warning-only: rivals that are another example's gold. Mutual rivals are legitimate,
    but reported numbers should account for them."""
    gold_alias = {sid: list(s.get("aliases") or []) for sid, s in (specs or {}).items()}
    from .events import alias_hit
    out: list[str] = []
    for sid, spec in (specs or {}).items():
        for r in rivals_for(sid, spec):
            blob = f"{r['name']} {' '.join(r['aliases'])}"
            others = sorted(o for o, al in gold_alias.items() if o != sid and alias_hit(blob, al))
            if others:
                out.append(f"{sid}: rival {r['name']!r} is also the gold of {others} (mutual rivals, expected)")
    return sorted(out)
