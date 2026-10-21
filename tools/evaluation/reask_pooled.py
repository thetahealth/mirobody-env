"""Recompute released re-ask rates and question-cluster intervals without model calls.

    python tools/evaluation/reask_pooled.py --check
    python tools/evaluation/reask_pooled.py --out artifacts/reask-analysis.json
"""
from __future__ import annotations

import sys
from pathlib import Path
for _root in Path(__file__).resolve().parents:
    if (_root / 'pyproject.toml').is_file() and (_root / 'tools').is_dir():
        sys.path.insert(0, str(_root))
        break
from tools._bootstrap import bootstrap
bootstrap(__file__)

import argparse
import collections
from decimal import Decimal, ROUND_HALF_UP
import json
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'benchmarks/results/1.3.1/reask'
BOOT, SEED = 2000, 20261014
MODELS = {'gpt-6-luna': ('GPT-6 Luna', 'LUNA'), 'glm-5.3-flash': ('GLM 5.3 Flash', 'GLM'),
          'qwen3-32b': ('Qwen3-32B', 'QWEN')}


def number_word(n):
    return ('zero', 'one', 'two', 'three')[n]


def load(source):
    rows = {}
    for line in source.read_text().splitlines():
        r = json.loads(line)
        if r['status'] == 'error':
            continue
        rows.setdefault((r['model'], r['question'], r['repeat'], r['arm'], r['kind']), r)
    return rows


def units(rows, model, kind, arms):
    by = collections.defaultdict(dict)
    for (m, q, rep, arm, kd), r in rows.items():
        if m == model and kd == kind:
            by[(q, rep)][arm] = int(r['D'] or 0)
    return [(q, d) for (q, rep), d in by.items() if all(a in d for a in arms)]


def boot(unit_lists, fn, weights):
    qs = sorted({q for ul in unit_lists for q, _ in ul})
    qi = {q: i for i, q in enumerate(qs)}
    rng = np.random.default_rng(SEED)
    draws = np.stack([np.bincount(rng.integers(0, len(qs), len(qs)), minlength=len(qs)) for _ in range(BOOT)])
    out, pts, bs = [], [], []
    for ul in unit_lists:
        s, c = np.zeros(len(qs)), np.zeros(len(qs))
        for q, d in ul:
            s[qi[q]] += fn(d)
            c[qi[q]] += 1
        pt = s.sum() / c.sum()
        b = (draws @ s) / np.where(draws @ c > 0, draws @ c, np.nan)
        pts.append(pt)
        bs.append(b)
        out.append((pt, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5), int(c.sum())))
    w = np.array(weights, float) / sum(weights)
    pooled_b = sum(w[i] * bs[i] for i in range(len(bs)))
    out.append((float(sum(w[i] * pts[i] for i in range(len(pts)))), np.nanpercentile(pooled_b, 2.5),
                np.nanpercentile(pooled_b, 97.5), sum(o[3] for o in out)))
    return out


def add_half_all(rows, half_rows):
    """Add arm HALL to every target: H for k >= 2 missed results, A's answer for k = 1 (floor(1/2) = 0 added)."""
    k1 = collections.Counter()
    for (m, q, rep, arm, kd), r in list(rows.items()):
        if kd != 'target' or arm != 'B':
            continue
        k = r['n_added']
        if k == 1:
            rows[(m, q, rep, 'HALL', kd)] = rows[(m, q, rep, 'A', kd)]
            k1[m] += 1
            continue
        h = half_rows.get((m, q, rep, 'H', 'half_target'))
        if k < 2 or h is None:
            raise ValueError(f'{m} {q} r{rep}: target with k={k} has no half answer')
        for a in ('A', 'B', 'C'):
            if half_rows[(m, q, rep, a, 'half_target')]['D'] != rows[(m, q, rep, a, 'target')]['D']:
                raise ValueError(f'{m} {q} r{rep}: half file disagrees with the recorded arm {a}')
        rows[(m, q, rep, 'HALL', kd)] = h
    return k1


def analyse(rows):
    models = [m for m in MODELS if units(rows, m, 'target', ['A', 'B', 'C'])]
    tgt = [units(rows, m, 'target', ['A', 'B', 'C', 'HALL']) for m in models]
    if [len(u) for u in tgt] != [len(units(rows, m, 'target', ['A', 'B', 'C'])) for m in models]:
        raise ValueError('a target lacks the half arm')
    weights = [len(u) for u in tgt]
    labels = models + ['pooled']
    res = {'models': models, 'n_targets': dict(zip(models, weights))}
    for name, fn in [('A', lambda d: d['A']), ('B', lambda d: d['B']), ('C', lambda d: d['C']),
                     ('HALL', lambda d: d['HALL']), ('B-C', lambda d: d['B'] - d['C']), ('B-A', lambda d: d['B'] - d['A']),
                     ('C-A', lambda d: d['C'] - d['A'])]:
        res[name] = dict(zip(labels, boot(tgt, fn, weights)))
    ctl_models = [m for m in models if units(rows, m, 'control', ['A'])]
    ctl = [units(rows, m, 'control', ['A']) for m in ctl_models]
    res['control_A'] = dict(zip(ctl_models + ['pooled'], boot(ctl, lambda d: d['A'], [len(u) for u in ctl])))
    return res


def analyse_half(rows):
    """H-A, B-H and H-C on the units with at least two missed decisive results."""
    models = [m for m in MODELS if units(rows, m, 'half_target', ['A', 'B', 'C', 'H'])]
    tgt = [units(rows, m, 'half_target', ['A', 'B', 'C', 'H']) for m in models]
    weights = [len(u) for u in tgt]
    res = {'models': models, 'n_units': dict(zip(models, weights))}
    for name, fn in [('A', lambda d: d['A']), ('H', lambda d: d['H']), ('B', lambda d: d['B']), ('C', lambda d: d['C']),
                     ('H-A', lambda d: d['H'] - d['A']), ('B-H', lambda d: d['B'] - d['H']),
                     ('H-C', lambda d: d['H'] - d['C'])]:
        res[name] = dict(zip(models + ['pooled'], boot(tgt, fn, weights)))
    return res


def whole(x):
    return str(Decimal(repr(100 * float(x))).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def pct(x):
    v = Decimal(repr(100 * float(x))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    return ('$-$' if v < 0 else '') + str(abs(v))


def make_facts(res, half):
    p = 'pooled'
    # The graded-evidence reading: half the missed results help, and all of them help more, pooled.
    if not (half['H-A'][p][1] > 0 and half['B-H'][p][1] > 0):
        raise ValueError('half-evidence arm no longer graded: H-A or B-H interval reaches zero')
    # Figure and caption: in every group the half arm lies between resampling and all of the missed results.
    for lbl in res['models'] + [p]:
        if not res['A'][lbl][0] <= res['HALL'][lbl][0] <= res['B'][lbl][0]:
            raise ValueError(f'{lbl}: half arm no longer between resample and all decisive results')
    # Prose: the decisive results raise accuracy above both other arms for every model and pooled,
    # and the placebo results do not (their difference from resampling has an interval covering zero).
    for lbl in res['models'] + [p]:
        if not (res['B-C'][lbl][1] > 0 and res['B-A'][lbl][1] > 0):
            raise ValueError(f'{lbl}: decisive results no longer clearly above placebo and resampling')
    if not res['C-A'][p][1] <= 0 <= res['C-A'][p][2]:
        raise ValueError('placebo results now differ from resampling')
    ctl_models = [m for m in res['control_A'] if m != p]
    names = [MODELS[m][0] for m in ctl_models]
    facts = {'INJ-N': str(res['B'][p][3]), 'INJ-NMODELS': str(len(res['models'])),
             'INJ-CTRL-N': str(res['control_A'][p][3]), 'INJ-CTRL-A': pct(res['control_A'][p][0]),
             'INJ-CTRL-MODELS': names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]}
    facts['INJ-NMODELS-WORD'] = number_word(len(res['models']))  # prose form of the model count
    for arm in ('A', 'B', 'C', 'HALL'):
        facts[f'INJ-{arm}'] = pct(res[arm][p][0])
    facts['INJ-K1-N'] = str(sum(res['k1'].values()))
    if int(facts['INJ-K1-N']) + half['B'][p][3] != res['B'][p][3]:
        raise ValueError('single-miss and k >= 2 targets do not add up to all targets')
    for c, key in (('B-C', 'BC'), ('B-A', 'BA'), ('C-A', 'CA')):
        est, lo, hi, _ = res[c][p]
        facts.update({f'INJ-{key}-PP': pct(est), f'INJ-{key}-LO': pct(lo), f'INJ-{key}-HI': pct(hi)})
    for m in res['models']:
        tag = MODELS[m][1]
        facts[f'INJ-N-{tag}'] = str(res['n_targets'][m])
        for arm in ('A', 'B', 'C', 'HALL'):
            facts[f'INJ-{arm}-{tag}'] = pct(res[arm][m][0])
        facts[f'INJ-K1-{tag}'] = str(res['k1'][m])
        est, lo, hi, _ = res['B-C'][m]
        facts.update({f'INJ-BC-{tag}-PP': pct(est), f'INJ-BC-{tag}-LO': pct(lo), f'INJ-BC-{tag}-HI': pct(hi)})
    facts['INJ-H-N'] = str(half['B'][p][3])
    for arm in ('A', 'H', 'B', 'C'):
        facts[f'INJ-H-{arm}'] = pct(half[arm][p][0])
    for c, key in (('H-A', 'HA'), ('B-H', 'BH'), ('H-C', 'HC')):
        est, lo, hi, _ = half[c][p]
        facts.update({f'INJ-H-{key}-PP': pct(est), f'INJ-H-{key}-LO': pct(lo), f'INJ-H-{key}-HI': pct(hi)})
    return facts


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def identity(row):
    return tuple(row[k] for k in ('model', 'question', 'repeat', 'kind', 'arm'))


def split_rows(inclusion):
    """Recover the main and k>=2 paired samples from one unique row per unit/arm."""
    rows, half = {}, {}
    eligible = {identity(r)[:4] for r in inclusion
                if r['kind'] == 'target' and r['arm'] == 'B' and r['n_added'] >= 2}
    for row in inclusion:
        key = identity(row)
        if key in rows or row['status'] not in ('answered', 'unanswered', 'empty'):
            raise ValueError(f'duplicate or nonterminal inclusion: {key}')
        if row['D'] not in (0, 1) or row['n_added'] < 0:
            raise ValueError(f'invalid outcome: {key}')
        if row['kind'] not in ('target', 'control') or row['arm'] not in ('A', 'B', 'C', 'H'):
            raise ValueError(f'invalid condition: {key}')
        rows[key] = row
        if key[:4] in eligible:
            m, q, rep, _, arm = key
            half[(m, q, rep, arm, 'half_target')] = {**row, 'kind': 'half_target'}
    # The analysis keys put arm before kind, unlike the row/answer identity.
    primary = {(m, q, rep, arm, kind): r for (m, q, rep, kind, arm), r in rows.items() if arm != 'H'}
    for m, q, rep, kind in {key[:4] for key in rows}:
        arms = {a: rows[(m, q, rep, kind, a)] for a in ('A', 'B', 'C', 'H')
                if (m, q, rep, kind, a) in rows}
        expected = {'A'} if kind == 'control' else {'A', 'B', 'C'} | ({'H'} if (m, q, rep, kind) in eligible else set())
        if set(arms) != expected or arms['A']['n_added'] != 0:
            raise ValueError(f'incomplete unit: {(m, q, rep, kind)}')
        if kind == 'target':
            k = arms['B']['n_added']
            if k < 1 or arms['C']['n_added'] != k or ('H' in arms and arms['H']['n_added'] != k // 2):
                raise ValueError(f'inconsistent added-result counts: {(m, q, rep, kind)}')
    k1 = add_half_all(primary, half)
    return primary, half, k1


def recompute(directory=DATA):
    rows, half, k1 = split_rows(read_jsonl(directory / 'inclusion.jsonl'))
    result = analyse(rows)
    result['k1'] = {m: k1[m] for m in result['models']}
    half_result = analyse_half(half)
    return {'bootstrap': {'draws': BOOT, 'seed': SEED, 'cluster': 'question', 'pooled_weights': 'model unit counts'},
            'main': result, 'half': half_result, 'facts': make_facts(result, half_result)}


def check_answers(directory=DATA):
    """Parse final text again and call the released diagnosis judges on its differential."""
    from haenv_kernel.solver import _extract_json
    from haenv.evaluate import _to_output
    from haenv.judges._helpers import gold_kind
    from haenv.judges.differential import judge_dx_unified, judge_dx_comorbidity

    inclusion = {identity(r): r for r in read_jsonl(directory / 'inclusion.jsonl')}
    answer_rows = read_jsonl(directory / 'answers.jsonl')
    answers = {identity(r): r for r in answer_rows}
    if len(answers) != len(answer_rows) or inclusion.keys() != answers.keys():
        raise ValueError('answers must cover the inclusion list exactly once')
    gold = json.loads((directory / 'gold.json').read_text())
    checked = 0
    for key, row in inclusion.items():
        answer = answers[key]
        try:
            parsed = _extract_json(answer['answer']) if answer['answer'].strip() else {}
        except (ValueError, TypeError):
            parsed = {}
        parsed = parsed if isinstance(parsed, dict) else {}
        if parsed.get('differential') != answer['differential']:
            raise ValueError(f'parsed differential differs: {key}')
        vp = SimpleNamespace(case_id=row['question'], adjudication={'ddx': gold[row['question']]})
        kind = gold_kind(vp)
        judge = {'ddx:unified': judge_dx_unified, 'ddx:comorbidity': judge_dx_comorbidity}.get(kind)
        if judge is None:
            raise ValueError(f'unsupported gold kind: {key}: {kind}')
        out = _to_output(parsed, SimpleNamespace(prediction_context={}))
        atoms = judge(out, vp, {})
        d = int(row['status'] == 'answered' and bool(parsed) and atoms['dx_listed'] == 1.0)
        if d != row['D']:
            raise ValueError(f'stored D differs: {key}: stored={row["D"]}, scored={d}')
        checked += 1
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--check', action='store_true', help='re-score every stored answer and compare all released facts')
    parser.add_argument('--out', type=Path, help='write full per-model and pooled estimates and 95%% intervals as JSON')
    args = parser.parse_args(argv)
    result = recompute(args.data)
    if args.check:
        expected = json.loads((args.data / 'facts.json').read_text())
        if result['facts'] != expected:
            raise ValueError('recomputed facts differ from the released facts')
        count = check_answers(args.data)
        print(f'{len(expected)} facts matched; {count} answers re-scored; 0 D mismatches')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for arm in ('A', 'C', 'HALL', 'B'):
        point, lo, hi, n = result['main'][arm]['pooled']
        print(f'{arm}: {100*point:.1f}% [{100*lo:.1f}, {100*hi:.1f}], n={n}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
