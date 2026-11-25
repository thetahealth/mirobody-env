"""Read qualified, affirmative symptom reports in ordinary clinical text.

NHS qualifiers are the existing 1.4 p2 SAY contract. No keyword-only inference:
only whole registered affirmative sentences count, in either language.
"""
from functools import lru_cache

from haenv.i18n.identity import source
from haenv.i18n.text import L, render


def records(block):
    """Join dated archive indexes to clinical excerpts imported by the device.

    An excerpt cannot establish a diagnosis without its matching dated source.
    This is a presentation join, not a new clinical rule or an inferred diagnosis.
    """
    rows = block.get('clinical_history_record') or ()
    attachments = [e for device in block.get('device_record') or ()
                   for e in device.get('entries') or () if 'clinical_excerpt' in e]
    out = []
    for row in rows:
        details = next((e['clinical_excerpt'] for e in attachments
                        if e.get('archive_index') == row.get('archive_index')
                        and e.get('day') == row.get('day')), {}) if 'archive_index' in row else {}
        out.append({**details, **row})
    return out


@lru_cache(maxsize=1)
def vocabulary():
    from haenv.p2 import render as RD, rules as R
    phrases = {symptom: RD.say_pool(symptom)
               for symptom in (*R.R0_KEYS, R.HTN_SYMPTOM, R.HYPO_HELP)}
    phrases.update({key: (L(value),) for key, value in (
        ('suspected_infection_today', 'p2.reading.infection'),
        ('head_injury_today', 'p2.reading.head'),
        ('systemically_unwell_today', 'p2.reading.unwell'))})
    return {symptom: tuple(v for phrase in variants
                           for v in (source(phrase), render(phrase, 'en')))
            for symptom, variants in phrases.items()}


def affirmed(text, phrase):
    start = str(text).find(phrase)
    if start < 0:
        return False
    prefix = str(text)[:start].rstrip()
    return not prefix or prefix[-1] in '.!?。！？;；\n'


def reported(block):
    text = [block.get('message', ''), (block.get('intake_checklist') or {}).get('patient_statement', '')]
    for record in block.get('device_record') or ():
        if record.get('day') == block.get('today_day'):
            text.extend(entry.get('text', '') for entry in record.get('entries') or ())
    found = set()
    for symptom, variants in vocabulary().items():
        if any(affirmed(s, v) for s in text for v in variants):
            found.add(symptom)
    return found
