"""WHO-UMC evidence reader; no duration cutoff or probabilistic causal claim.

The clinic action mapping is authored, not a WHO prescribing recommendation.
Legacy records without this evidence retain their original interpretation.
"""
from __future__ import annotations

from haenv.p3.textread import canonical, slots, field


def assessment(onset, start, withdrawal, alternative):
    if onset < start:
        return 'predating'
    if withdrawal is True and alternative is False:
        return 'dose_related'
    if withdrawal is False and alternative is True:
        return 'alternative'
    return 'unattributed'


def visible_evidence(block, symptom, onset):
    """Read observation text, not a directory label, category or hidden cause."""
    found = {}
    for row in block.get('encounter_record') or ():
        text = field(row['text'], 'p3.emr.line.hpi') if 'text' in row else row.get('history')
        for name in ('withdrawal', 'alternative', 'rechallenge'):
            statuses = [('yes', True), ('no', False)]
            if name == 'alternative':
                statuses.append(('unassessed', None))
            for status, value in statuses:
                key = f'p3.causality.{name}.{status}'
                parsed = slots(text, key)
                if parsed and canonical(parsed['symptom']) == canonical(symptom) and int(parsed['onset']) == int(onset):
                    found[name] = value
                    for day in ('hold', 'resume'):
                        if day in parsed:
                            found[day] = int(parsed[day])
    return found or None


def visible_holds(block):
    """Documented hold/resumption spans read from the consult notes' text (the withdrawal note
    dates the hold and the review at which the same dose resumed)."""
    out = set()
    for row in block.get('encounter_record') or ():
        text = field(row['text'], 'p3.emr.line.hpi') if 'text' in row else row.get('history')
        for value in ('yes', 'no'):
            parsed = slots(text, f'p3.causality.withdrawal.{value}')
            if parsed and 'hold' in parsed and 'resume' in parsed and int(parsed['resume']) > int(parsed['hold']):
                out.add((int(parsed['hold']), int(parsed['resume'])))
    return sorted(out)


def visible_cause(block, symptom, onset, start):
    evidence = visible_evidence(block, symptom, onset)
    if evidence is None:
        return None
    return assessment(onset, start, evidence.get('withdrawal'), evidence.get('alternative'))


def hidden_cause(entry, block):
    record = block.get('ae_causality')
    if not record or canonical(entry['symptom']) != canonical(record['symptom']) or entry.get('cause') == 'predating':
        return None
    start = min(int(day) for day, _ in block['dose_line'])
    return assessment(int(entry['onset']), start, record.get('withdrawal'), record.get('alternative'))
