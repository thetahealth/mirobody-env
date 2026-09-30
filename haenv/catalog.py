"""catalog.py -- the single reader of `registry/pack_catalog.yaml`.

The catalog records each pack's role (main / probe / internal), title, what it asks and
whether its readings may be published. `cli._catalog_gate` enforces `publishable: false`
by flagging the report and writing `NOT-FOR-RELEASE.md` next to it.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from typing import Any

#: Allowed roles, default-deny: a misspelled role must not silently become "probe".
ROLES = ("main", "probe", "internal")

#: Values of `relation`; guards pick their verification method by it.
RELATIONS = ("single_factor", "subset_of", "world_delta", "mixture")

_REQUIRED = ("role", "title", "asks", "publishable")


class CatalogError(ValueError):
    """Catalog failed load-time validation (raises rather than warns)."""


def pack_catalog() -> dict[str, dict[str, Any]]:
    """The whole catalog, `{job_id: {...}}`. Every entry is validated at load time; a bad
    entry raises immediately."""
    from .yamlcache import load_yaml as _cached
    from .regpath import registry_path as _rp
    p = _rp("pack_catalog.yaml")
    if not p.is_file():
        raise CatalogError(f"registry is missing the file: {p}")
    doc = _cached(p) or {}
    packs = doc.get("packs") or {}
    if not isinstance(packs, dict) or not packs:
        raise CatalogError(f"{p}: `packs` is empty or not a mapping")
    for job, spec in packs.items():
        if not isinstance(spec, dict):
            raise CatalogError(f"{job}: entry is not a mapping")
        miss = [k for k in _REQUIRED if k not in spec]
        if miss:
            raise CatalogError(f"{job}: missing required field(s) {miss}")
        if spec["role"] not in ROLES:
            raise CatalogError(f"{job}: role `{spec['role']}` is not one of {ROLES}")
        if not spec["publishable"] and not str(spec.get("why") or "").strip():
            raise CatalogError(f"{job}: `publishable: false` but no `why` was written")
        if (spec["role"] == "internal") is bool(spec["publishable"]):
            raise CatalogError(
                f"{job}: role={spec['role']} conflicts with publishable={spec['publishable']}")
        if spec["role"] == "probe" and not spec.get("pairs_with"):
            raise CatalogError(f"{job}: a probe entry must have `pairs_with` (reading it alone is meaningless)")
        rel = spec.get("relation")
        if spec["role"] == "probe":
            if rel not in RELATIONS:
                raise CatalogError(f"{job}: relation `{rel}` is not one of {RELATIONS}")
            if rel == "single_factor" and not spec.get("single_factor"):
                raise CatalogError(f"{job}: a `single_factor` entry must name which key")
            if rel == "world_delta" and not spec.get("world_delta"):
                raise CatalogError(f"{job}: a `world_delta` entry must list the latent key(s) allowed to differ")
    return dict(packs)


def pack_aliases() -> dict[str, str]:
    """Retired job id -> current job id (`aliases:` in `pack_catalog.yaml`). Old batch
    directories keep their old id; this map is how they resolve to today's pack."""
    from .yamlcache import load_yaml as _cached
    from .regpath import registry_path as _rp
    doc = _cached(_rp("pack_catalog.yaml")) or {}
    al = doc.get("aliases") or {}
    if not isinstance(al, dict):
        raise CatalogError("`aliases` is not a mapping")
    packs = doc.get("packs") or {}
    bad = sorted(str(v) for v in al.values() if v not in packs)
    if bad:
        raise CatalogError(f"aliases point at pack(s) not in the catalog: {bad}")
    return {str(k): str(v) for k, v in al.items()}


def current_id(job: str) -> str:
    """The current id for a job id (a retired id maps through `aliases`; anything else is returned as is)."""
    return pack_aliases().get(job, job)


def ids_of(job: str) -> list[str]:
    """A pack's current id followed by its retired ids: every `results/joint_dx/<id>/` directory
    that holds batches of this pack."""
    cur = current_id(job)
    return [cur] + sorted(o for o, n in pack_aliases().items() if n == cur)


def pack_entry(job: str) -> dict[str, Any] | None:
    """A pack's catalog entry, or `None` when it is not in the catalog.

    A retired job id resolves to its current pack through `aliases`.
    Ad hoc jobs outside the catalog are normal; the caller decides what to do.
    """
    cat = pack_catalog()
    return cat.get(job) or cat.get(pack_aliases().get(job, ""))


def pack_role(job: str) -> str | None:
    e = pack_entry(job)
    return str(e["role"]) if e else None


def pack_title(job: str) -> str:
    """The human-readable name; falls back to the `job_id` itself when not in the catalog
    (a report header must not crash over this)."""
    e = pack_entry(job)
    return str(e["title"]) if e else str(job)


def is_publishable(job: str) -> bool | None:
    """Whether the reading may go external; `None` when the pack is not in the catalog
    (not registered is different from registered as not publishable).
    """
    e = pack_entry(job)
    return bool(e["publishable"]) if e else None


def review_pending_conditions() -> list[str]:
    """Conditions whose clinical content is not yet reviewed
    (`clinical_review: pending` in `conditions_unified.yaml`), computed live from the field.
    """
    from .yamlcache import load_yaml as _cached
    from .regpath import registry_path as _rp
    doc = _cached(_rp("conditions_unified.yaml")) or {}
    conds = doc.get("conditions") or {}
    return sorted(k for k, v in conds.items()
                  if isinstance(v, dict) and str(v.get("clinical_review")) == "pending")


#: The condition id's location in an emitted `cases.jsonl` row (there is no top-level `latent`).
_SPEC_ID_PATH = ("case", "latent_premise", "meta", "ddx", "spec_id")


def case_spec_id(case: dict) -> str:
    """Extract the condition id from one emitted `cases.jsonl` row. Use this rather than
    reading `latent` from the row: that key does not exist at the top level, and the lookup
    would fail silently.
    """
    cur = case
    for k in _SPEC_ID_PATH:
        cur = (cur or {}).get(k) if isinstance(cur, dict) else None
    return str(cur or "")


def review_pending_in(cases) -> list[str]:
    """Unreviewed conditions used within this batch of cases (deduplicated, sorted).
    `cases` is the sequence of dicts read from `cases.jsonl`."""
    pend = set(review_pending_conditions())
    return sorted({case_spec_id(c) for c in cases} & pend)


def packs_by_role() -> dict[str, list[str]]:
    """Grouped by role, sorted by job_id within each group. Both the data card and the
    report use this single grouping."""
    out: dict[str, list[str]] = {r: [] for r in ROLES}
    for job, spec in pack_catalog().items():
        out[str(spec["role"])].append(job)
    return {r: sorted(v) for r, v in out.items()}
