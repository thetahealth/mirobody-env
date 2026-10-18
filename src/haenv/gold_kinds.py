"""Supported public interface; implementation lives in haenv.scoring.gold_kinds."""
from haenv.scoring.gold_kinds import (
    EXTERNAL,
    KindRule,
    RULES,
    UNDERIVABLE,
    derive,
    known_kinds,
    log,
    register_kind_rule,
    reset_kind_rules,
    rules_manifest,
    unregister_kind_rule,
)

__all__ = ['EXTERNAL', 'KindRule', 'RULES', 'UNDERIVABLE', 'derive', 'known_kinds', 'log', 'register_kind_rule', 'reset_kind_rules', 'rules_manifest', 'unregister_kind_rule']
