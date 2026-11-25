"""Render patient wording from final measured state, never tier."""
from __future__ import annotations

from haenv.i18n.text import J, L, Msg
# Keep existing helper imports compatible; judging imports their pure owner.
from haenv.p2.statement_contracts import contracts, permits, raised_temperature, violations


def _leaves(message):
    if isinstance(message, Msg) and message.kind == 'join':
        return [leaf for m in message.slots['items'] for leaf in _leaves(m)]
    return [message]


def measured_message(block, cid):
    from haenv.p2 import render as RD
    cfg = contracts()
    leaves = []
    for m in _leaves(block['message']):
        key = getattr(m, 'key', '')
        if key in cfg['phrases'] and not permits(cfg['phrases'][key], block):
            continue
        leaves.append(m)
    if raised_temperature(block):
        leaves.append(L(RD.B.pick(cfg['required']['raised_temperature'], 'p2-measured-fever', cid)))
    return J('sentences', leaves)
