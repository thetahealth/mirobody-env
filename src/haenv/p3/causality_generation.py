"""M4 authored causal-evidence coverage, with the hold in the physical dose line."""
from __future__ import annotations

from copy import deepcopy
from haenv.i18n.text import L, T
from haenv.resources.regpath import load_registry


def applies(plan):
    from haenv.world.truth import coupling_enabled
    from haenv.world.validation.difficulty_r2 import enabled
    cfg = load_registry('p3_causality.yaml')['coverage']
    # R3: a discordant course of an exposure-attributed drug has no course-free equivalent (any onset on
    # the drug is dose-related there), so its plan replays the documented trial in every arm.
    replay = plan.get('causal_replay') == 1
    return (plan.get('ae_causality_version') == 1 and (coupling_enabled() or replay) and enabled(plan)
            and plan['drug'] in cfg['drugs'] and cfg['documented_trial_share_by_class'][plan['class']] == 1.0)


def layout(plan, lay, today):
    if not applies(plan):
        return lay
    lay = deepcopy(lay)
    # Place the trial after the last measured encounter, independently of the
    # response class. This avoids inventing a later titration while an acting
    # symptom is already documented. No measured visit is added or moved.
    start = max(int(day) for day in lay['visits'] if int(day) < int(today))
    end = min(int(day) for day in lay['visits'] if int(day) > start)
    settled_end = int(today) - int(lay['tau'])
    if settled_end > start + 4:
        end = min(end, settled_end)
    onset = start + (end - start) // 4
    hold = start + 2 * (end - start) // 4
    resume = start + 3 * (end - start) // 4
    if not start < onset < hold < resume < end:
        return None
    dose = lay['line'][-1][1]
    lay['line'] = sorted([*lay['line'], [hold, 0], [resume, dose]])
    # These are exposure events, not a new prescribed strength. Retain the
    # original stock and the existing prescription's dose-review clock.
    lay['onset'] = onset
    lay['causality_trial_days'] = [hold, resume]
    return lay


def install(block, plan, lay):
    if 'causality_trial_days' not in lay:
        return
    from haenv.p3.causality import hidden_cause
    symptom = next(e for e in block['diary'] if e.get('kind') == 'drug_symptom' and e['cause'] != 'predating')
    active = plan['subtype'] in ('ae_dose', 'off_ae')
    alternative = not active and plan['subtype'] != 'unattributed'
    withdrawal = rechallenge = active
    hold, resume = lay['causality_trial_days']
    variant = discordant_variant(block, plan)
    if variant:
        # R3: the two observations disagree, or the assessment of other causes is missing; the printed
        # protocol asks for a causality review first (unattributed, `causality.assessment`).
        withdrawal = rechallenge = variant != 'unimproved_without_alternative'
        alternative = {'improved_with_alternative': True, 'unimproved_without_alternative': False}.get(variant)
    block['ae_causality'] = dict(symptom=symptom['symptom'], onset=int(symptom['onset']),
        hold=hold, resume=resume, withdrawal=withdrawal, rechallenge=rechallenge,
        **({} if alternative is None else {'alternative': alternative}))
    symptom['status'] = L('p3.status.ongoing')
    symptom['cause'] = hidden_cause(symptom, block)


def discordant_variant(block, plan):
    """The R3 evidence variant of an unattributed documented course, dealt per patient alike over
    `signal_conflicts.unattributed_check.variants` (None: the legacy concordant-negative course)."""
    if plan.get('subtype') != 'unattributed':
        return None
    from haenv.p3.plan import conflicts, h01
    variants = (conflicts().get('unattributed_check') or {}).get('variants')
    if not variants:
        return None
    return variants[int(h01('p3-discordant-causality', block.get('record_key')) * len(variants))]


def notes(block):
    r = block.get('ae_causality')
    if not r:
        return []
    out = []
    for name, day in [('withdrawal', r['resume']), ('alternative', int(block['T'])), ('rechallenge', int(block['T']))]:
        if name not in r and name != 'alternative':
            continue
        status = 'unassessed' if name not in r else 'yes' if r[name] else 'no'
        out.append(dict(day=day, source=L('p3.source.consult'), symptom=L('p3.emr.complaint.revisit'), onset_day=day,
                        history=T(f'p3.causality.{name}.{status}', hold=r['hold'], resume=r['resume'],
                                  symptom=r['symptom'], onset=r['onset'])))
    return out


def intervention(block, fact):
    """Surgical Tier R control; re-derive the hidden attribution after the edit."""
    from haenv.p3.causality import hidden_cause
    if fact == 'causal_onset':
        from haenv.p3.interactions import _symptom_onset
        _symptom_onset(block)
        block['ae_causality']['onset'] = min(int(d) for d, _ in block['dose_line']) - 30
    else:
        block['ae_causality'].pop(fact.removeprefix('causal_'), None)
    for entry in block['diary']:
        if entry.get('kind') == 'drug_symptom' and entry['cause'] != 'predating':
            entry['cause'] = hidden_cause(entry, block)


def plan_fields(drug):
    from haenv.world.truth import coupling_enabled
    from haenv.world.validation.difficulty_r2 import enabled
    if coupling_enabled() and enabled() and drug in load_registry('p3_causality.yaml')['coverage']['drugs']:
        return {'ae_causality_version': 1}
    return {}


def discordant_capable(drug, known=()) -> bool:
    """R3 (ours): whether this drug can carry a discordant causal course in either arm.
    Independent of coupling, so both arms plan the same subtype set. A timing-attributed
    drug has a course-free discordant symptom; an exposure-attributed one replays the
    documented course (`discordant_fields`)."""
    from haenv.world.validation.difficulty_r2 import enabled
    return bool(enabled() and drug in load_registry('p3_causality.yaml')['coverage']['drugs'])


def discordant_fields(drug, sub, known=()):
    """R3 (ours): the plan fields of a discordant check item in either arm.

    An exposure-attributed drug has no course-free discordant symptom (any onset on
    the drug is dose-related there), so this one subtype replays its documented
    course even when the arm does not couple the world. Every other class keeps
    its arm's own plan fields: a trial changes the visit layout and is not part of
    the coupling-off world.
    """
    from haenv.world.labworld.meds import drug as registered
    if sub != 'unattributed' or not discordant_capable(drug, known):
        return {}
    if registered(drug, known).get('attribution') != 'exposure':
        return {}                       # a timing-attributed drug needs no course replay
    return {'ae_causality_version': 1, 'causal_replay': 1}
