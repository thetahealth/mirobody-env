"""Persist message provenance beside case bodies; decisions consume source IDs.

The sidecar is verifier metadata. It never enters solver payloads or world hashes.
Legacy Chinese batches need no migration; their registry IDs are canonical strings.
"""
from __future__ import annotations

import json
from pathlib import Path

from haenv.i18n.text import Msg, render


def source(value):
    """Canonical value of a structured message tree, independent of its display."""
    if isinstance(value, Msg):
        return render(value, 'zh')
    if isinstance(value, dict):
        return {source(k): source(v) for k, v in value.items()}
    if isinstance(value, list):
        return [source(v) for v in value]
    if isinstance(value, tuple):
        return tuple(source(v) for v in value)
    return value


def encode(value):
    if isinstance(value, Msg):
        return {'$message': [str.__str__(value), value.kind, value.key, encode(value.slots)]}
    if isinstance(value, dict):
        return {'$pairs': [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


class _LegacyMessage(Msg):
    """Compatibility with the first source-sidecar edition, which lacked saved text."""
    __slots__ = ()


def decode(value):
    if isinstance(value, dict) and '$message' in value:
        fields = value['$message']
        if len(fields) == 4:
            saved, kind, key, slots = fields
            return Msg(saved, kind, key, decode(slots))
        kind, key, slots = fields
        slots = decode(slots)
        m = Msg('', kind, key, slots)
        return _LegacyMessage(render(m, 'zh'), kind, key, slots)
    if isinstance(value, dict) and '$pairs' in value:
        return {decode(k): decode(v) for k, v in value['$pairs']}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def display(value, lang):
    """Render while retaining IDs in memory; JSON still contains ordinary text."""
    if isinstance(value, Msg):
        return Msg(render(value, lang), value.kind, value.key, value.slots)
    if isinstance(value, dict):
        return {display(k, lang): display(v, lang) for k, v in value.items()}
    if isinstance(value, list):
        return [display(v, lang) for v in value]
    if isinstance(value, tuple):
        return tuple(display(v, lang) for v in value)
    return render(value, lang)


def save(path, built):
    from dataclasses import asdict
    target = Path(path).with_name('message_sources.jsonl')
    with target.open('w', encoding='utf-8') as f:
        for cid, raw in built.items():
            f.write(json.dumps({'case_id': cid, 'messages': encode(asdict(raw))}, ensure_ascii=False) + '\n')


def load(path):
    target = Path(path).with_name('message_sources.jsonl')
    if not target.is_file():
        return {}
    return {r['case_id']: decode(r['messages']) for line in target.read_text(encoding='utf-8').splitlines()
            if line.strip() for r in [json.loads(line)]}


def restore(actual, original):
    """Attach only matching source descriptors; modified saved text stays modified."""
    if isinstance(original, Msg) and isinstance(actual, str):
        # Validate the artifact binding, not today's translation of an old batch.
        if actual == original:
            return Msg(actual, original.kind, original.key, original.slots)
        if isinstance(original, _LegacyMessage):
            from haenv.i18n.text import MissingText
            for lang in ('zh', 'en'):
                try:
                    if actual == render(original, lang):
                        return Msg(actual, original.kind, original.key, original.slots)
                except MissingText:
                    pass
        return actual
    if isinstance(original, dict) and isinstance(actual, dict):
        # Keys also have provenance (e.g. diagnosis and laboratory names).
        keys = {}
        for key in original:
            keys[str(key)] = key
            if not isinstance(key, _LegacyMessage):
                continue
            for lang in ('zh', 'en'):
                try:
                    keys[render(key, lang)] = key
                except Exception:
                    continue
        return {restore(k, keys.get(k, k)): restore(v, original.get(keys.get(k, k)))
                for k, v in actual.items()}
    if isinstance(original, list) and isinstance(actual, list):
        # Solver payloads remove future evidence from the middle of the ledger.
        # An evidence ID, not its new array position, identifies the source row.
        by_id = {v['evidence_id']: v for v in original
                 if isinstance(v, dict) and v.get('evidence_id')}
        by_time = {v['ts']: v for v in original if isinstance(v, dict) and 'ts' in v}
        if len(by_time) != len(original):
            by_time = {}
        return [restore(v, by_id.get(v['evidence_id']) if isinstance(v, dict) and v.get('evidence_id')
                        else by_time.get(v['ts']) if by_time and isinstance(v, dict) and 'ts' in v
                        else original[i] if i < len(original) else None) for i, v in enumerate(actual)]
    return actual
