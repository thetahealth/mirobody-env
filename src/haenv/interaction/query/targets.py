"""Language-independent addresses shared by purchase and record tools."""
from __future__ import annotations

import hashlib

from haenv.i18n.text import Msg, has_cjk


def target_id(target: str) -> str:
    key = target.key if isinstance(target, Msg) and target.kind == 'lex' else str(target)
    return 'test-' + hashlib.sha256(key.encode()).hexdigest()[:16] if has_cjk(key) else key


def canonical_target(target: str, gc: dict) -> str:
    return next((item['target'] for item in gc.get('menu') or ()
                 if target_id(item['target']) == target), target)
