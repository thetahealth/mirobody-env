"""Registry of gold kinds: what shape the correct answer takes.

A gold kind is the first link of the dispatch chain::

    gold kind -> judges.applicable(geometry, kind, vp) -> mount_table.MOUNT

Rules are tried in registration order and the first match wins. The last built-in
rule maps everything else to :data:`UNDERIVABLE`, which attaches to no judge, so a
case whose join type cannot be derived shows up as unjudged. The rules are in the
frozen scoring segment, since they decide which judges attach.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("haenv.gold_kinds")

#: Sentinel kind: the join type could not be derived. No judge may list it in its `kinds`.
UNDERIVABLE = "ddx:_underivable"


@dataclass(frozen=True)
class KindRule:
    """One kind-derivation rule.

    * `name` -- the rule's name, not a kind name; one rule may produce several
                 kinds (see `join_gold`).
    * `derive` -- `(vp) -> str | None`. `None` means no match; try the next rule.
    * `why` -- why this rule exists, and above all why it sits at this
                 position in the order.
    * `kinds` -- the kinds this rule can produce, used for enumeration and
                 self-checks; it takes no part in dispatch.
    """
    name: str
    derive: Callable
    why: str = ""
    kinds: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------- built-in rules


def _rule_not_ddx(vp) -> str | None:
    from . import wq
    return None if wq.is_ddx_world(vp) else "forecast"


def _rule_insufficient(vp) -> str | None:
    from .wq import _ddx
    return "ddx:insufficient" if bool((_ddx(vp) or {}).get("insufficient")) else None


def _rule_join_gold(vp) -> str | None:
    from . import wq
    try:
        return f"ddx:{wq.gold_of(vp, 'join_gold')}"
    except wq.GoldNotDerivable as e:
        log.error("[gold_kinds] join_gold could not be derived, ddx judges will not attach to this item: %s", e)
        return UNDERIVABLE


#: Built-in derivation rules; order is semantics (first match wins).
_CORE: tuple[KindRule, ...] = (
    KindRule("not_ddx", _rule_not_ddx,
             why="没有诊断真值的世界一律 forecast。必须排第一:`W.outcome_label` 在 "
                 "forecast 题上同样存在,后面几条会经由 `derive_join_gold` 抛 "
                 "`GoldNotDerivable`,forecast 例会被自洽门误判为金标矛盾。",
             kinds=("forecast",)),
    KindRule("insufficient", _rule_insufficient,
             why="信息不足档是独立的第四类金标:它的正解是「现在还不该下结论」,"
                 "归并类型对它没有意义(题面一条真症状都不可见)。"
                 "走独立 kind ⇒ 前三类判据不挂。",
             kinds=("ddx:insufficient",)),
    KindRule("join_gold", _rule_join_gold,
             why="归并类型三选一;推不出时不回落到 unified(否则推不出的例子会被判分)。",
             kinds=("ddx:unified", "ddx:comorbidity", "ddx:independent", UNDERIVABLE)),
)

#: The live registry: built-ins plus plugins.
RULES: list[KindRule] = list(_CORE)

#: Source of each external rule; recorded in the artefacts.
EXTERNAL: dict[str, str] = {}


def register_kind_rule(rule: KindRule, *, source: str = "external",
                       before: str | None = None) -> KindRule:
    """Register an external kind rule.

    `before` inserts ahead of the named rule; the default appends, so an external rule
    is only reached when no built-in rule matched. Inserting ahead of a built-in changes
    the kind of existing questions, which is why it must be explicit. Duplicate names are
    rejected, and built-in rules cannot be replaced or removed at runtime (the scoring
    fingerprint covers source code, not runtime substitutions).
    """
    if not isinstance(rule, KindRule):
        raise TypeError(f"register_kind_rule expects a KindRule, got {type(rule).__name__}")
    if not rule.name or rule.derive is None:
        raise ValueError(f"incomplete rule registration: name={rule.name!r}")
    if rule.name in {r.name for r in _CORE}:
        raise ValueError(
            f"{rule.name!r} is a built-in rule; runtime replacement is not allowed, since "
            f"the judging fingerprint covers source code only. Edit `_CORE` instead.")
    if rule.name in {r.name for r in RULES}:
        raise ValueError(f"{rule.name!r} is already registered (source: {EXTERNAL.get(rule.name, '?')}). "
                         f"Duplicate names are rejected, so the running rule does not depend on registration order.")
    if UNDERIVABLE in rule.kinds:
        raise ValueError(f"{UNDERIVABLE!r} is a sentinel, external rules are not allowed to produce it -- "
                         f"its whole meaning is \"could not be derived\", and only the built-in fallback rule may own it.")
    if before is not None:
        idx = next((i for i, r in enumerate(RULES) if r.name == before), None)
        if idx is None:
            raise KeyError(f"before={before!r} is not in the registry; current rules {[r.name for r in RULES]}. "
                           f"Does not fall back to appending at the end: position is priority.")
        RULES.insert(idx, rule)
    else:
        RULES.append(rule)
    EXTERNAL[rule.name] = source
    return rule


def unregister_kind_rule(name: str) -> KindRule:
    """Remove an external rule. Built-ins cannot be removed."""
    if name in {r.name for r in _CORE}:
        raise ValueError(f"{name!r} is a built-in rule, runtime removal is not allowed; edit `_CORE` instead.")
    for i, r in enumerate(RULES):
        if r.name == name:
            EXTERNAL.pop(name, None)
            return RULES.pop(i)
    raise KeyError(f"{name!r} is not in the registry; current rules {[r.name for r in RULES]}.")


def reset_kind_rules() -> None:
    """Reset to the built-in set. For tests."""
    RULES[:] = list(_CORE)
    EXTERNAL.clear()


def derive(vp) -> str:
    """The gold kind for this case; `judges.gold_kind` delegates here.

    Raises when nothing matches rather than falling back to some kind. Built-in rules
    always match, so a raise means an external rule displaced them.
    """
    for r in RULES:
        got = r.derive(vp)
        if got:
            return got
    raise RuntimeError(
        f"No kind rule matched this item. Current rules {[r.name for r in RULES]}. "
        f"Does not fall back to any kind, so an undecidable case is not judged.")


def known_kinds() -> tuple[str, ...]:
    """Every kind the registry can produce; a judge's `kinds` may not name anything else."""
    seen: list[str] = []
    for r in RULES:
        for k in r.kinds:
            if k not in seen:
                seen.append(k)
    return tuple(seen)


def rules_manifest() -> dict:
    """Built-in and external rules for this batch, reported separately."""
    return {"n_core": len(_CORE), "n_total": len(RULES),
            "core": [r.name for r in _CORE],
            "external": dict(EXTERNAL),
            "order": [r.name for r in RULES],
            "kinds": list(known_kinds())}
