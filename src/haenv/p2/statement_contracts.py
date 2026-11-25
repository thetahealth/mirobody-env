"""Patient phrase preconditions on measured state, shared by leak and G checks.

The temperature boundary is reused from NEWS2; wording adds no exact value or
disposition. Identity comparisons precede translation.
"""
from __future__ import annotations

from haenv.i18n.identity import source
from haenv.i18n.text import L, render
from haenv.p2 import rules as R


def contracts():
    from haenv.resources.regpath import registry_path
    from haenv.resources.yamlcache import load_yaml
    return load_yaml(registry_path('p2_statement_contracts.yaml'))


def raised_temperature(block):
    upper = next(hi for lo, hi, points in R.table()['news2']['temp'] if points == 0)
    return block['home_check_today']['temperature_c'] > upper


def permits(predicate, block):
    ck = R.checklist_of(block)
    if predicate == 'raised_temperature':
        return raised_temperature(block)
    if predicate == 'otherwise_fine':
        return not raised_temperature(block) and not any(ck.get(k) for k in R.R0_KEYS)
    return bool(block['intake_checklist'].get(predicate) or ck.get(predicate))


def violations(block, *, required=True):
    """Also validate serialized zh/en text using paired source phrase identities."""
    cfg = contracts()
    message = block.get('message', '')
    canonical = source(message)
    def contains(key):
        phrase = L(key)
        return any(text in str(message) for text in (source(phrase), render(phrase, 'en'))) or source(phrase) in canonical
    out = []
    for key, predicate in cfg['phrases'].items():
        if not required and not key.startswith('p2.say.measured.'):
            continue
        if contains(key) and not permits(predicate, block):
            out.append('measured statement precondition: ' + predicate + ': ' + key)
    for predicate, keys in cfg['required'].items():
        if required and permits(predicate, block) and not any(contains(key) for key in keys):
            out.append('measured statement missing: ' + predicate)
    if required:
        from haenv.p2 import render as RD
        for symptom, variants in RD.SAY.items():
            for phrases in variants.values():
                for phrase in phrases:
                    key = phrase.key
                    predicate = next((p for prefix, p in cfg['symptom_prefixes'].items()
                                      if key.startswith(prefix)), None)
                    if predicate != symptom:
                        out.append('undeclared severity phrase: ' + key)
                    elif contains(key) and not permits(predicate, block):
                        out.append('measured severity precondition: ' + predicate)
    return out
