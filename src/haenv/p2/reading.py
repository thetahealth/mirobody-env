"""M5 generation coverage: gold-preserving reading work, never reader filtering.

Clinical bands are the existing sourced P2 policy. Shares, within-disposition
weights and observation times are explicitly authored in p2_reading.yaml.
"""
from copy import deepcopy

from haenv.i18n.text import J, L
from haenv.resources.regpath import load_registry, registry_cached
from haenv.world.history.history_dependency import uniform


@registry_cached('p2_reading.yaml')
def config():
    return load_registry('p2_reading.yaml')


def enabled(plan=None):
    if plan is None:
        # The solver-visible catalog follows the edition a batch was generated for.
        from haenv.world.protocols.library import edition
        current = edition()
        return current["course_coupling"] and current["difficulty_r2"]
    from haenv.world.truth import coupling_enabled
    from haenv.world.validation.difficulty_r2 import enabled as difficulty_enabled
    return coupling_enabled() and difficulty_enabled(plan)


def core(cid, plan=None):
    coverage = config()['coverage']
    rule = tier = None
    if plan and plan.get('tier'):
        from haenv.p2 import rules as R
        tier = plan['tier']
        rule = R.table()['tiers'][tier]['rule']
    share = coverage.get('core_share_by_tier', {}).get(
        tier, coverage.get('core_share_by_rule', {}).get(rule, coverage['core_share']))
    return uniform('p2-m5-composition', cid) < share


def margin(cid, plan=None):
    """Core items whose decisive value is drawn near its sourced threshold (`margin`); a hash of the
    case id, blind to the gold and to any reader outcome."""
    return bool(plan) and enabled(plan) and core(cid, plan) and \
        uniform('p2-m5-margin', cid) < config()['margin']['share']


def cgm_history(cid, plan=None):
    """Core CGM wearers also see nights before the printed 7-night window."""
    return bool(plan) and enabled(plan) and core(cid, plan)


def history_active(block, floor):
    """Use the same floor contrast as the production history-twin contract."""
    from haenv.p2 import rules as R
    cut = {**block, 'clinical_history_record': []}
    return R.floor_of(R.assess(cut)['set']) != floor


def finalize(raw, cid, plan):
    if not enabled(plan):
        return
    from haenv.p2 import rules as R
    from haenv.world.protocols.scoring import FIELDS, _normal, scale_of
    b, g = raw.prediction_context['triage_p2'], raw.adjudication['triage_p2']
    g['composition'] = 'core' if core(cid, plan) else 'anchor'
    if not core(cid, plan):
        return
    # Same note placement for acting and non-acting diagnoses. The source
    # remains present, with the same dates/titles and clinical conclusion.
    archived = {r.get('condition') for r in b.get('clinical_history_record') or ()}
    for profile in (b['background'], raw.user_profile):
        profile['known_conditions'] = [c for c in profile.get('known_conditions', ()) if c not in archived]
    from haenv.p2 import rules as R
    if g['rule'] == 'R11_hypo_level1' or R.table()['tiers'].get(plan.get('tier'), {}).get('render_as') == 'hypo_l1':
        # Reorder the already measured night values, preserving their count and
        # severity. The last night is normal in both AF-history roles.
        nights = b['cgm_nightly']
        window = int(b['today_day']) - 7
        if nights and nights[-1]['min_mmol_L'] < R.table()['thresholds']['hypo_level1_mmol']:
            other = next((r for r in nights[:-1] if r['night_of_day'] >= window and
                          r['min_mmol_L'] >= R.table()['thresholds']['hypo_level1_mmol']), None)
            if other:
                for field in ('min_mmol_L', 'minutes_below_3_9'):
                    nights[-1][field], other[field] = other[field], nights[-1][field]
    # Gold was fixed from the complete structured state; prose already renders
    # those facts. Ticks can miss a fact without denying its narrative report.
    ck = b['intake_checklist']
    ck['tick_note'] = L('p2.reading.tick_note')
    if uniform('p2-m5-narrative', cid) < config()['coverage']['narrative_share']:
        ck['urgent_symptoms'] = []
        for key in R.R0_KEYS:
            if key in ck:
                ck[key] = False
        ck[R.HTN_SYMPTOM] = False
        # The symptom itself is unchanged; the patient's original tick omitted it.
        extra = []
        for key, phrase in (('suspected_infection_today', 'p2.reading.infection'),
                            ('head_injury_today', 'p2.reading.head'),
                            ('systemically_unwell_today', 'p2.reading.unwell')):
            if ck.get(key):
                extra.append(L(phrase))
                ck[key] = False
        b['message'] = J('sentences', [b['message'], *extra])
        from haenv.p2 import render as RD
        from haenv.p2.statements import _leaves
        from haenv.i18n.identity import source
        # The same channel draw moves affirmative and inert symptom wording.
        if uniform('p2-m5-symptom-channel', cid) < config()['coverage']['device_share_of_assistance_reports']:
            phrases = {source(p) for key in (*R.R0_KEYS, R.HTN_SYMPTOM, R.HYPO_HELP)
                       for p in RD.say_pool(key)} | {source(p) for p in extra}
            leaves = _leaves(b['message'])
            moved = [m for m in leaves if source(m) in phrases]
            b['message'] = J('sentences', [m for m in leaves if source(m) not in phrases])
            b['device_record'][0]['entries'].extend({'day': b['today_day'], 'text': m} for m in moved)
        ck[R.HYPO_HELP] = False
    # Import complete clinical excerpts into the device record, retaining their
    # dated archive index. Acting and inert findings use identical containers.
    # Gold already exists: only the location of the supporting detail changes.
    if uniform('p2-m5-clinical-channel', cid) < config()['coverage']['device_clinical_excerpt_share']:
        rows = b.get('clinical_history_record') or []
        for i, row in enumerate(rows):
            index = f"archive-{i}"
            keep = {k: row[k] for k in ('day', 'source', 'test', 'condition') if k in row}
            detail = {k: v for k, v in row.items() if k not in keep}
            b['device_record'][0]['entries'].append(dict(
                day=row['day'], archive_index=index, text=L('p2.reading.excerpt'), clinical_excerpt=detail))
            row.clear()
            row.update(keep, archive_index=index)
    home = b['home_check_today']
    current = {field: home[field] for field in (*FIELDS.values(), 'on_oxygen')}
    first = dict(current)
    changed = uniform('p2-m5-repeat', cid) < config()['coverage']['repeat_share']
    if changed:
        if R.readings(b)['news2']:
            # An earlier normal set: only the parameters that score now differ (deterioration).
            parts = R.news2_parts(home, R.checklist_of(b), 2 if scale_of(b) == 2 else 1)
            first.update({field: _normal(field, b) for key, field in FIELDS.items() if parts[key]})
        else:
            # A five-point Scale-1 first set, without a single three, and a later
            # complete repeat. Interiors use RCP Chart 1, not new thresholds.
            first.update(first_upload_values(cid))
    first_time, repeat_time, later_time = upload_times(cid)
    home['observation_sets'] = [
        dict(day=b['today_day'], time=first_time, observations=first),
        dict(day=b['today_day'], time=repeat_time, observations=current),
    ]
    if changed and R.readings(b)['news2'] and \
            uniform('p2-m5-partial', cid) < config()['observation_times']['partial_later_share']:
        # A later partial re-check that looks normal: incomplete, so the abnormal complete set stands
        # (printed: incomplete later uploads do not supersede it).
        fields = sorted(FIELDS.values())
        missing = fields[int(uniform('p2-m5-partial-field', cid) * len(fields))]
        parts = R.news2_parts(home, R.checklist_of(b), 2 if scale_of(b) == 2 else 1)
        later = {field: (None if field == missing else _normal(field, b) if parts[key] else current[field])
                 for key, field in FIELDS.items()}
        home['observation_sets'].append(dict(day=b['today_day'], time=later_time,
                                             observations={**later, 'on_oxygen': current['on_oxygen']}))
    b['triage_protocol'] = J('lines', [b['triage_protocol'], L(config()['protocol'])])
    a = R.assess(b)
    if a['rule'] != g['rule'] or a['set'] != g['set']:
        raise ValueError('M5 rendering changed the structured-state gold')
    g['readings'] = a['readings']
    # Recompute the existing history annotation after channel placement. No
    # response class, acceptance threshold or gold is changed by this step.
    if 'history_dependency' in g:
        g['history_dependency']['active'] = history_active(b, g['floor'])
        for row in (g.get('look_alike') or {}).get('records', ()):
            row['acts'] = g['history_dependency']['active']


def upload_times(cid):
    """(first, repeat, later partial) same-day upload times; authored, not a clinical interval."""
    times = config()['observation_times']
    starts = times['first_choices']
    h, m = map(int, starts[int(uniform('p2-m5-time', cid) * len(starts))].split(':'))
    gaps = times['repeat_gap_minutes']
    gap = gaps[int(uniform('p2-m5-gap', cid) * len(gaps))]
    stamp = lambda minutes: f"{minutes // 60:02d}:{minutes % 60:02d}"
    return stamp(h * 60 + m), stamp(h * 60 + m + gap), stamp(h * 60 + m + gap + times['partial_gap_minutes'])


def first_upload_values(cid):
    """An aggregate-5 Scale-1 first upload without a single 3 (RCP Chart 1 band interiors),
    drawn per case instead of one constant set."""
    import itertools
    cfg = config()['coverage']['first_upload_bands']
    combos = [c for c in itertools.product(*([0, *sorted(map(int, cfg[k]))] for k in sorted(cfg)))
              if sum(c) == 5]
    pick = combos[int(uniform('p2-m5-first-vector', cid) * len(combos))]
    out = {}
    for key, points in zip(sorted(cfg), pick):
        if points:
            field, (lo, hi) = cfg[key][points]['field'], cfg[key][points]['range']
            u = uniform('p2-m5-first-value', cid, key)
            out[field] = round(lo + (hi - lo) * u, 1) if isinstance(lo, float) else lo + int(u * (hi - lo + 1))
    return out


def first_upload(block):
    """The machine's earlier input set; complete repeats remain visible."""
    b = deepcopy(block)
    home = b['home_check_today']
    sets = home.pop('observation_sets')
    home.update(sets[0]['observations'])
    return b


def measurement_conflicts(block):
    sets = block.get('home_check_today', {}).get('observation_sets')
    if sets is None:
        return []
    from haenv.world.protocols.scoring import FIELDS
    fields = (*FIELDS.values(), 'on_oxygen')
    complete = [s for s in sets if s.get('day') == block['today_day'] and
                all(s.get('observations', {}).get(f) is not None for f in fields)]
    if not complete:
        return ['no complete contemporaneous NEWS2 observation set']
    latest = max(complete, key=lambda s: s['time'])['observations']
    if any(block['home_check_today'].get(f) != latest[f] for f in fields):
        return ['home_check_today differs from latest complete NEWS2 set']
    return []
