"""Saved patient measurements before P2's declared scenario change (P2_WORLD).

Points already include the saved medication/event course. Missing daily home
observations are simulated from the patient's event-free centre plus that course;
they are never interpolated and labelled as recorded world points.
"""
from __future__ import annotations

import copy
from statistics import median

from haenv.world.truth import events as E

STREAMS = ('systolic_bp', 'diastolic_bp', 'resting_hr', 'heart_rate', 'spo2', 'body_temp')


def capture_enabled(raw, plan):
    from haenv.world.truth import coupling_enabled
    return ('patient_truth' in plan or 'event_truth' in plan
            or bool((raw.latent_premise or {}).get('truth')) or coupling_enabled())


def snapshot(raw, T):
    events = copy.deepcopy(E.event_plan_of(raw) or {})
    points = {s: {int(q['ts']): float(q['value']) for q in (raw.longitudinal_data or {}).get(s, ())
                  if int(q['ts']) <= T and isinstance(q.get('value'), (int, float))} for s in STREAMS}
    # A home pulse and a resting wearable rate are separate channels. If only
    # one exists, it supplies a patient centre for a newly simulated channel.
    centres = {}
    for s, series in points.items():
        sig = 'resting_hr' if s == 'heart_rate' else s
        if series:
            centres[s] = median(v - E.increment(events, sig, d) - E.drug_vitals(events, sig, d)
                                for d, v in series.items())
    if 'heart_rate' not in centres and 'resting_hr' in centres:
        centres['heart_rate'] = centres['resting_hr']
    if 'resting_hr' not in centres and 'heart_rate' in centres:
        centres['resting_hr'] = centres['heart_rate']
    if 'body_temp' not in centres and events.get('baseline_temp') is not None:
        centres['body_temp'] = events['baseline_temp']
    # Retain all dated effects, but only the window's physical measurements.
    # Older readings contribute to the fallback centre, not a second history.
    from haenv.p2.render import WINDOW
    points = {s: {d: v for d, v in rows.items() if T - WINDOW < d} for s, rows in points.items()}
    return dict(points=points, centres=centres, events=events)


def value(saved, stream, day):
    points = saved.get('points', {}).get(stream, {})
    if day in points or str(day) in points:
        return points.get(day, points.get(str(day)))
    centre = saved.get('centres', {}).get(stream)
    if centre is None:
        return None
    sig = 'resting_hr' if stream == 'heart_rate' else stream
    return centre + E.increment(saved['events'], sig, day) + E.drug_vitals(saved['events'], sig, day)


def constrain(block, saved, reason):
    """Verifier-only receipt of each explicit target differing from patient state."""
    out = []
    T = block['today_day']
    for stream, field in [('body_temp', 'temperature_c'), ('heart_rate', 'heart_rate_bpm'),
                          ('spo2', 'spo2_pct'), ('systolic_bp', 'systolic_bp_mmHg'),
                          ('diastolic_bp', 'diastolic_bp_mmHg')]:
        before, after = value(saved, stream, T), block['home_check_today'][field]
        digits = 1 if stream == 'body_temp' else 0
        if before is not None and round(before, digits) != after:
            out.append(dict(quantity=stream, day=T, channel='home_check', patient_value=before,
                            realized_value=after, reason=reason))
    for r in block['home_bp_log']:
        for stream, field in [('systolic_bp', 'systolic_mmHg'), ('diastolic_bp', 'diastolic_mmHg')]:
            before = value(saved, stream, r['day'])
            if before is not None and round(before) != r[field]:
                out.append(dict(quantity=stream, day=r['day'], channel=r['time'], patient_value=before,
                                realized_value=r[field], reason=reason))
    return out
