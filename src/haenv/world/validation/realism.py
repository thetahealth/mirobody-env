"""Sampling-scale estimates and descriptive realism threshold comparisons.

A permutation-null SD is a scale estimate, not a confidence interval for a fitted
classifier's population AUC. Existing point readings and thresholds stay intact.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics


def null_uncertainty(draws):
    values = [float(v) for v in draws]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite C2ST permutation reading')
    return {'sd': statistics.stdev(values) if len(values) >= 2 else None,
            'n_resamples': len(values), 'uncertainty_method': 'label-permutation null sample SD',
            'uncertainty_note': 'Sampling-scale estimate under the null; not a fitted-model AUC confidence interval.'}


def paired_max_uncertainty(classifier_draws):
    lengths = {len(v) for v in classifier_draws}
    if len(lengths) != 1 or not classifier_draws:
        raise ValueError('Classifier null draws must have matching permutation identities')
    return null_uncertainty([max(values) for values in zip(*classifier_draws)])


def _margin(reading):
    if reading.get('ci') is not None:
        lo, hi = reading['ci']
        if not all(math.isfinite(float(v)) for v in (lo, hi)) or lo > hi:
            raise ValueError('Invalid C2ST confidence interval')
        return max(reading['value'] - lo, hi - reading['value'])
    sd = reading.get('sd')
    if sd is None:
        return None
    if not math.isfinite(float(sd)) or sd < 0:
        raise ValueError('Invalid C2ST sampling SD')
    return 1.96 * sd


def crossing(current, baseline, threshold):
    threshold = float(threshold)
    value, before = float(current['value']), float(baseline['value']) if baseline else None
    if not math.isfinite(value) or not math.isfinite(threshold) or (before is not None and not math.isfinite(before)):
        raise ValueError('C2ST readings and thresholds must be finite')
    margin = _margin(current)
    result = {'value': value, 'before': before, 'threshold': threshold, 'point_pass': value <= threshold,
              'sampling_margin_95': margin, 'regression': False}
    if value <= threshold:
        status = 'within_threshold'
    elif before is None:
        status = 'crossing_without_baseline'
        result['regression'] = None
    elif before > threshold:
        status = 'inherited_crossing'
    elif margin is None:
        status = 'uncertainty_unmeasured'
        result['regression'] = None
    else:
        previous_margin = _margin(baseline)
        delta_margin = math.hypot(margin, previous_margin) if previous_margin is not None else None
        result['difference_margin_95'] = delta_margin
        # The fixed threshold test uses the new reading's scale. Where both arms have
        # scales, require the increase to exceed their independent combined error too.
        beyond = value - threshold > margin and (delta_margin is None or value - before > delta_margin)
        status = 'regression' if beyond else 'crossing_within_sampling_error'
        result['regression'] = beyond
    result['status'] = status
    result['status_label'] = {'within_threshold': '阈值内', 'inherited_crossing': '继承越线',
                             'crossing_without_baseline': '缺少基线', 'uncertainty_unmeasured': '抽样误差未测',
                             'crossing_within_sampling_error': '误差内越线', 'regression': '回退'}[status]
    return result


def reading_map(document):
    """Named readings in the public C2ST readout shapes (nested pack views supported)."""
    result = {}
    def visit(obj, prefix=()):
        if not isinstance(obj, dict):
            return
        if isinstance(obj.get('value'), (int, float)):
            result['/'.join(prefix)] = obj
            return
        for k, v in obj.items():
            visit(v, (*prefix, k))
    visit(document.get('readings', document))
    return result


def compare(current, baseline, thresholds):
    before = reading_map(baseline) if baseline else {}
    thresholds = thresholds.get('threshold', thresholds)
    result = {}
    for key, reading in reading_map(current).items():
        group = key.rsplit('/', 1)[-1].split('|')[-1]
        threshold = thresholds.get(key, thresholds.get(group))
        if threshold is None:
            result[key] = {'status': 'threshold_unmeasured', 'regression': None}
            continue
        result[key] = crossing(reading, before.get(key), threshold)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(prog='haenv realism')
    ap.add_argument('current', type=Path)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--thresholds', required=True, type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)
    result = compare(json.loads(a.current.read_text()), json.loads(a.baseline.read_text()) if a.baseline else None,
                     json.loads(a.thresholds.read_text()))
    document = {'method': 'new threshold crossing beyond 1.96 sampling SD (or supplied CI); inherited crossings reported separately',
                'readings': result, 'regressions': [k for k, v in result.items() if v['regression'] is True],
                'unmeasured': [k for k, v in result.items() if v['regression'] is None]}
    text = json.dumps(document, ensure_ascii=False, indent=2) + '\n'
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(text, encoding='utf-8')
    print(text, end='')
    return 1 if document['regressions'] else 2 if document['unmeasured'] else 0
