"""quantities.py -- the quantity registry: each quantity's canonical name and aliases.

Consumers read values only through `resolve()`. A `derived` quantity is computed from
its definition when absent from a row (e.g. `tool_budget_thrift = 1 - tool_budget_used`).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Quantity:
    """A measurable quantity and all of its names.

    - `rec_key` -- its name in the aggregation layer (`rank_*`'s `rec`); `None` means same name.
    - `row_aliases` / `renamed_from` -- other names it may carry in an `eval.jsonl` row.
    - `derived` -- computes it by definition when the row lacks it.
    - `applies_to` -- geometries it applies to; empty means all.
    """
    canonical: str
    rec_key: str | None = None
    row_aliases: tuple[str, ...] = ()
    renamed_from: tuple[str, ...] = ()
    derived: Callable[[Mapping], object] | None = field(default=None, repr=False)
    applies_to: tuple[str, ...] = ()
    why: str = ""


def _thrift(r: Mapping):
    """`tool_budget_thrift = 1 - tool_budget_used`; older batches record only `tool_budget_used`."""
    u = r.get("tool_budget_used")
    if not isinstance(u, (int, float)) or isinstance(u, bool):
        return None
    # Same clamp as `tracks.tool_track`, so derived and produced thrift agree.
    return min(1.0, max(0.0, round(1.0 - float(u), 3)))


def tool_grounded_joint_of(r: Mapping):
    """`tool_grounded_joint`: grounding rate joined with "was a necessary tool called".

    Not a gated row (no `tool_budget`) => None. The item declares key signals
    (`tool_key_covered` not None): no signal query => 0.0; otherwise grounding rate x
    1[key coverage > 0]. No key signals declared: grounding rate when there was a query,
    else None (nothing was necessary). Design 2026-09-30 §2.1: a model that never queries
    no longer averages 1.000 over the few cells where it did.
    """
    if r.get("tool_budget") is None and r.get("tool_n_calls") is None:
        return None
    n = r.get("tool_n_signal_calls")
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        n = r.get("tool_n_calls") or 0
    g = r.get("tool_target_grounded_rate")
    if g is None:
        g = r.get("tool_grounded_rate")
    kc = r.get("tool_key_covered")
    if kc is None:
        return (round(float(g), 3) if (n and isinstance(g, (int, float))) else None)
    if not n:
        return 0.0
    return round(float(g or 0.0) * (1.0 if float(kc) > 0 else 0.0), 3)


def dx_listed_of(r: Mapping):
    """`dx_listed`: the share of gold lines on the differential and not ruled out, within the
    candidate cap. The one definition, used by the dx judges on their own output and here on
    read.

    Rows written before the field existed carry no live positions; there it falls back to
    `dx_coverage` (comorbidity) or `dx_hit` (unified), without the cap.
    """
    pos = r.get("dx_listed_positions")
    if isinstance(pos, list) and pos:
        cap = r.get("dx_listed_cap")
        n = sum(1 for p in pos if isinstance(p, int) and not isinstance(p, bool)
                and (not isinstance(cap, int) or p <= cap))
        return round(n / len(pos), 3)
    if r.get("dx_kind") == "comorbidity":
        c = r.get("dx_coverage")
        return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else None
    h = resolve(r, "dx_hit")
    return (1.0 if h else 0.0) if isinstance(h, bool) else None


def dx_listed_n0_of(r: Mapping):
    """A6 profile atom: `dx_listed` under the strict cap n + 0 (a line counts only within the first
    `n` live candidates, `n` = number of gold lines). Read from `dx_listed_positions`; `None` when
    the row has no live positions."""
    pos = r.get("dx_listed_positions")
    if not (isinstance(pos, list) and pos):
        return None
    n = sum(1 for p in pos if isinstance(p, int) and not isinstance(p, bool) and p <= len(pos))
    return round(n / len(pos), 3)


#: The only place that defines what a quantity is called; direct reads outside `resolve()` are
#: checked by the development repository's static analysis, separate from runtime resolution.
QUANTITIES: tuple[Quantity, ...] = (
    # ---- Test ordering (paired) ----
    Quantity("tests_recall", row_aliases=("wk_tests_recall_last",),
             why="What fraction of the tests that should have been ordered were ordered"),
    Quantity("tests_precision", rec_key="tests_prec",
             row_aliases=("wk_tests_precision_last",),
             why="What fraction of the ordered tests were ones that should have been ordered. "
                 "The rec key differs from the canonical name -- always read it through "
                 "resolve(), never by looking up the canonical name directly in rec."),
    Quantity("disc_recall", row_aliases=("wk_disc_recall_last",),
             why="Discriminating-point coverage"),

    # ---- Tool track (specific to the gated geometry) ----
    Quantity("tool_target_grounded_rate", renamed_from=("tool_grounded_rate",),
             applies_to=("gated",),
             why="Grounding rate. renamed_from keeps batches written under the former "
                 "key readable."),
    Quantity("tool_dup_rate", applies_to=("gated",), why="Redundancy (lower is better)"),
    Quantity("tool_budget_used", applies_to=("gated",),
             why="Budget used. Non-monotone: zero queries is also 0"),
    Quantity("tool_budget_thrift", derived=_thrift, applies_to=("gated",),
             why="Thrift == 1 - used (monotone). Derived on the read side: older gated "
                 "batches lack this key"),

    # ---- Self-contradiction exclusion (renamed from refuting_evidence) ----
    Quantity("sce_n_excluded", renamed_from=("ref_n_excluded",)),
    Quantity("sce_note", renamed_from=("ref_note",)),
    Quantity("sce_contradiction_rate", renamed_from=("ref_contradiction_rate",)),
    Quantity("sce_n_with_support", renamed_from=("ref_n_with_support",)),
    Quantity("sce_unrevealed_rate", renamed_from=("ref_unrevealed_rate",)),
    Quantity("sce_unrevealed", renamed_from=("ref_unrevealed",)),

    # ---- Remaining scored dimensions (no aliases) ----
    Quantity("noop_ok", why="No-op probe: declaration matches ground truth"),
    Quantity("review_macro", why="Specificity of review declarations (not-warranted class)"),
    Quantity("review_utility", why="Balanced accuracy of review declarations: (sensitivity + specificity) / 2"),
    Quantity("abst_utility", why="Balanced accuracy of abstention: insufficient-tier cells abstained, other cells did not"),
    Quantity("action_consistency", why="Semantic auxiliary atom: the action fits the model's own differential; profile reading"),
    Quantity("abst_utility_cc", why="Chance-corrected abstention utility, max(0, 2 x abst_utility - 1) (A3); profile reading, not scored (A-block 2)"),
    Quantity("tool_grounded_joint", derived=tool_grounded_joint_of, applies_to=("gated",),
             why="Grounding rate joined with whether a necessary tool was called; no query = 0"),
    Quantity("quant_ok", why="Data-check item (triple identity, anchor-exempt)"),
    Quantity("excl_grounded_rate", why="Citation authenticity (continuous version); scored as diagnostic, not as a scored dimension"),
    Quantity("dx_hit", why="Whether the gold diagnosis appears in the model's differential list (matched via the kernel DDX_SPECS aliases); "
                           "on a comorbidity case, whether at least one thread does. Reported, not scored"),
    Quantity("dx_listed_n0", derived=dx_listed_n0_of,
             why="dx_listed under the strict cap n + 0 (A6 profile reading, not scored)"),
    Quantity("dx_listed", derived=dx_listed_of,
             why="Share of gold lines (the diagnosis, or each comorbidity thread) on the differential and not ruled out, "
                 "within the candidate cap. Derived on the read side for rows written before the field existed"),
)

BY_NAME: dict[str, Quantity] = {q.canonical: q for q in QUANTITIES}


class QuantityError(KeyError):
    """Looked up a quantity that is not registered (no fallback)."""


def get(name: str) -> Quantity:
    q = BY_NAME.get(name)
    if q is None:
        raise QuantityError(
            f"{name!r} is not registered in QUANTITIES. Direct reads are not allowed: "
            f"a quantity's name may have only one place of definition; to use it, register it first (add one line, no code changes).")
    return q


def resolve(container: Mapping, name: str, *, rec: bool = False):
    """Read a quantity's value: `rec_key` (when `rec=True`), then `canonical`, then aliases, then `derived`.

    The canonical name wins over aliases. An unregistered quantity raises instead of returning None.
    """
    q = get(name)
    if rec and q.rec_key:
        v = container.get(q.rec_key)
        if v is not None:
            return v
    v = container.get(q.canonical)
    if v is not None:
        return v
    for alias in q.row_aliases + q.renamed_from:
        v = container.get(alias)
        if v is not None:
            return v
    if q.derived is not None:
        return q.derived(container)
    return None


def row_names(name: str) -> tuple[str, ...]:
    """All names this quantity may appear under in one row, canonical name first. Raises if unregistered."""
    q = get(name)
    return (q.canonical,) + q.row_aliases + q.renamed_from


def all_names() -> frozenset[str]:
    """All registered names (canonical names and every alias)."""
    out: set[str] = set()
    for q in QUANTITIES:
        out.add(q.canonical)
        out.update(q.row_aliases)
        out.update(q.renamed_from)
        if q.rec_key:
            out.add(q.rec_key)
    return frozenset(out)
