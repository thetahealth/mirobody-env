"""Repeated discrimination gates and saturation diagnostics from saved evaluations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

from haenv.reporting.pack_readout import category
from haenv.reporting.pack_readout import declaration
from haenv.reporting.pack_readout import history_label
from haenv.reporting.pack_readout import seed_readout
from haenv.resources.yamlcache import load_yaml


def jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def raw(pack, row):
    if str(row.get("overall", "")).startswith("ABORT"):
        return 0
    value = pack.raw_correct(row)
    return None if value is None else int(bool(value))


def rate(pack, rows):
    values = [v for r in rows if (v := raw(pack, r)) is not None]
    return sum(values) / len(values) if values else None


def score(pack, rows):
    rows = [r for r in rows if not str(r.get("overall", "")).startswith("ABORT")]
    return pack.main_score(rows) if rows else None


def percentile(values):
    values = sorted(v for v in values if v is not None)
    return [values[int(len(values) * .025)], values[min(len(values) - 1, int(len(values) * .975))]] if len(values) >= 10 else None


def load(directory, models=None):
    data, cases = defaultdict(lambda: defaultdict(dict)), {}
    sources = set()
    for file in sorted(Path(directory).rglob("eval.jsonl")):
        match = re.search(r"-pk(\d+)$", file.parent.name)
        if not match:
            continue
        sources.add(file.parent.name.rsplit("-pk", 1)[0])
        for row in jsonl(file):
            if models is None or row["solver"] in models:
                data[row["solver"]][int(match[1])][row["case"]] = row
    if len(sources) > 1:
        raise ValueError("Multiple source batches in one pack work directory; select one work directory")
    for file in sorted(Path(directory).rglob("cases.jsonl")):
        for row in jsonl(file):
            cid, case = row["case_id"], row["case"]
            if cid in cases and cases[cid] != case:
                raise ValueError(f"Conflicting saved cases: {cid}")
            cases[cid] = case
    return data, cases


def analyze(pack, data, cases, *, n, k, policy, models=None):
    names = sorted(models if models is not None else data)
    ids = sorted(set(cases) | {c for m in data.values() for rs in m.values() for c in rs})
    rounds = range(1, k + 1)
    def rows(model, draw):
        return [data.get(model, {}).get(r, {})[cid] for cid in draw for r in rounds
                if cid in data.get(model, {}).get(r, {})]
    point = {m: {"raw": rate(pack, rows(m, ids)), "main": score(pack, rows(m, ids))} for m in names}
    rng = random.Random(f"{policy['seed']}|real|{pack.name}")
    draws = [{m: score(pack, rows(m, draw)) for m in names}
             for _ in range(policy['bootstrap'])
             for draw in [[rng.choice(ids) for _ in ids]]] if ids else []
    pairs = {}
    for x, y in itertools.combinations(names, 2):
        ci = percentile([d[x] - d[y] for d in draws if d[x] is not None and d[y] is not None])
        pairs[f"{x} - {y}"] = {"main_ci": ci, "separable_main": bool(ci and (ci[0] > 0 or ci[1] < 0))}
    all_right = {}
    for cid in ids:
        cells = [data.get(m, {}).get(r, {}).get(cid) for m in names for r in rounds]
        known = [raw(pack, r) for r in cells if r is not None and raw(pack, r) is not None]
        all_right[cid] = bool(names and all(r is not None for r in cells) and known and all(known))
    for m, stats in point.items():
        rs = rows(m, ids)
        stats.update(n_cells=len(rs), n_abort=sum(str(r.get('overall', '')).startswith('ABORT') for r in rs),
                     n_unknown=sum(raw(pack, r) is None for r in rs),
                     complete=len(ids) == n and all(set(data.get(m, {}).get(r, {})) == set(ids) for r in rounds))
    complete = bool(names and len(ids) == n and all(v['complete'] for v in point.values()))
    order = sorted(names, key=lambda m: (point[m]['main'] if point[m]['main'] is not None else -1,
                                        point[m]['raw'] if point[m]['raw'] is not None else -1), reverse=True)
    strongest = max(names, key=lambda m: point[m]['raw'] if point[m]['raw'] is not None else -1) if names else None
    strongest_raw = point[strongest]['raw'] if strongest else None
    g1 = {"model": strongest, "raw": strongest_raw, "limit": policy['strongest_raw_max'],
          "pass": strongest_raw <= policy['strongest_raw_max'] if strongest_raw is not None else None, "report_only": True}
    g2 = {"all_right": sum(all_right.values()), "of": len(ids), "limit": policy['all_right_max_fraction'] * n,
          "models": names, "pass": sum(all_right.values()) <= policy['all_right_max_fraction'] * n}
    top2 = order[:2]
    pair = f"{min(top2)} - {max(top2)}" if len(top2) == 2 else None
    g3 = {"models": top2, "metric": "main", "pair": pair, "ci": pairs[pair]['main_ci'] if pair else None,
          "pass": pairs[pair]['separable_main'] if pair and pairs[pair]['main_ci'] else None}
    passed = g2['pass'] is True and g3['pass'] is True if complete else None
    gate = {"G1_strongest_raw": g1, "G2_all_right": g2, "G3_top2_separable": g3, "complete": complete,
            "pass": passed, "verdict": "INCOMPLETE" if not complete else "PASS" if passed else "FAIL"}
    dimensions = {**pack.dimensions, "history_dependency": history_label}
    saturation = {}
    for name, callback in dimensions.items():
        groups = defaultdict(list)
        for cid in ids:
            groups[category(callback(cases.get(cid, {})))].append(cid)
        saturation[name] = {value: {"n_questions": len(cs), "all_right": sum(all_right[c] for c in cs),
                                    "all_right_fraction": sum(all_right[c] for c in cs) / len(cs),
                                    "models": {m: {"raw": rate(pack, rows(m, cs)), "n_cells": len(rows(m, cs))} for m in names}}
                            for value, cs in sorted(groups.items())}
    return {"n_questions": len(ids), "expected_n": n, "k": k, "models": point, "pairs": pairs,
            "gate": gate, "saturation": saturation, "difficulty": {"all_right": sum(all_right.values()), "n_questions": len(ids)}}


def find_source(root, pack, n):
    candidates = [root / f"real/{pack}-n{n}/pack", root / f"real/{pack}-n{n}",
                  root / f"{pack}-n{n}/pack", root / f"{pack}-n{n}", root]
    for directory in candidates:
        jobs = sorted(directory.glob('*.job.yaml'))
        batches = [p.parent for p in directory.glob('results/*/*/*/cases.jsonl') if not re.search(r'-pk\d+$', p.parent.name)]
        if len(jobs) == len(batches) == 1:
            return jobs[0], batches[0]
    raise ValueError(f"Expected one job and one frozen batch for pack {pack}")


def find_job_source(job):
    """Resolve a built job through its local or configured output root, as run does."""
    from haenv import output_root
    document = load_yaml(job)
    job_id, task = document['job_id'], document.get('task_type', 'joint_dx')
    roots = dict.fromkeys([job.parent, output_root(), job.parent.parent])
    for root in roots:
        batches = sorted(p.parent for p in (root / 'results' / task / job_id).glob('*/cases.jsonl')
                         if not re.search(r'-(pk|rep)\d+$', p.parent.name))
        if batches:
            return job, batches[-1]
    raise ValueError(f"Job {job.name} has no frozen batch; build and audit it before running a gate")


def prepare(job, batch, destination, n):
    import yaml
    from haenv.reporting.readout_tools import tool
    prepare_sample_dir = tool('reliability_passk').prepare_sample_dir
    rows = jsonl(batch / 'cases.jsonl')
    if len(rows) != n:
        raise ValueError(f"Expected {n} frozen cases; got {len(rows)}")
    document = load_yaml(job)
    document['include_baseline'] = False
    destination.mkdir(parents=True, exist_ok=True)
    target_job = destination / job.name
    text = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    if target_job.exists() and target_job.read_text() != text:
        raise ValueError("Work job differs from source; use a new --out directory")
    target_job.write_text(text, encoding='utf-8')
    target = destination / 'results' / batch.parent.parent.name / batch.parent.name / batch.name
    prepare_sample_dir(batch, target)
    (destination / 'STAMP').write_text(batch.name, encoding='utf-8')
    return target_job, target


def render(result):
    lines = ['# Discrimination gate', '', 'G1 is report only. G2 and G3 determine the verdict; incomplete grids have no verdict.', '',
             '| Pack | G1 raw | G2 all right | G3 main difference CI | Verdict |', '|---|---|---|---|---|']
    for name, reading in result['packs'].items():
        g = reading['gate']; a, b, c = (g[x] for x in ('G1_strongest_raw', 'G2_all_right', 'G3_top2_separable'))
        lines.append(f"| {name} | {a['model']}: {a['raw']} ({a['pass']}) | {b['all_right']}/{b['of']} ({b['pass']}) | {c['ci']} ({c['pass']}) | {g['verdict']} |")
    for name, reading in result['packs'].items():
        seed = reading.get('seed', {})
        lines += ['', f"{name} pack seed: {seed.get('kind', 'unknown')} seed."]
        lines += ['WARNING: ' + warning for warning in seed.get('warnings', [])]
        lines += ['', f'## {name} saturation', '', '| Dimension | Category | Questions | All right fraction | Model raw correctness |', '|---|---|---|---|---|']
        for dimension, categories in reading['saturation'].items():
            for label, r in categories.items():
                rates = ', '.join(f"{m}: {s['raw']}" for m, s in r['models'].items())
                shown = {'true': 'history-dependent', 'false': 'ordinary'}.get(label, label) if dimension == 'history_dependency' else label
                lines.append(f"| {dimension} | {shown} | {r['n_questions']} | {r['all_right_fraction']:.3f} | {rates} |")
    if result.get('comparison'):
        lines += ['', '## Baseline comparison', '', 'Descriptive differences; independent question draws are not paired effects.', '',
                  '| Pack | Same models / k | All right fraction change | Model raw change | Gate before → after |', '|---|---|---|---|---|']
        for p, c in result['comparison'].items():
            lines.append(f"| {p} | {c['same_models']} / {c['same_k']} | {c['all_right_fraction_delta']} | {c['raw_delta']} | {c['before']} → {c['after']} |")
    return '\n'.join(lines) + '\n'


def compare(current, baseline):
    old = baseline.get('packs')
    if old is None:
        # Archived reports store the current and previous panels under revision keys.
        # Select the first panel with gate-shaped entries, in the report's own order.
        old = next((v for v in baseline.values() if isinstance(v, dict) and
                    any(isinstance(x, dict) and 'gate' in x for x in v.values())), {})
    result = {}
    for p, new in current.items():
        if p not in old:
            continue
        prev = old[p]; a, b = new['difficulty'], prev['difficulty']
        previous_k = prev.get('k')
        if previous_k is None:
            recorded_rounds = {v.get('n_rounds') for v in prev.get('models', {}).values()}
            if len(recorded_rounds) == 1:
                previous_k = next(iter(recorded_rounds))
        result[p] = {'all_right_fraction_delta': a['all_right'] / a['n_questions'] - b['all_right'] / b['n_questions'] if a['n_questions'] and b['n_questions'] else None,
                     'raw_delta': {m: v['raw'] - prev['models'][m]['raw'] for m, v in new['models'].items()
                                   if m in prev['models'] and v['raw'] is not None and prev['models'][m]['raw'] is not None},
                     'before': prev['gate']['verdict'], 'after': new['gate']['verdict'],
                     'same_models': sorted(new['models']) == sorted(prev['models']),
                     'same_k': new.get('k') == previous_k}
    return result


def main(argv=None):
    from haenv.app.config import load_cfg
    cfg = load_cfg(); policy = cfg['gate']
    ap = argparse.ArgumentParser(prog='haenv gate')
    ap.add_argument('inputs', nargs='*', type=Path)
    ap.add_argument('--from-work', type=Path)
    ap.add_argument('--packs', default=None)
    ap.add_argument('--models', default=None)
    ap.add_argument('--n', type=int, default=policy['n'])
    ap.add_argument('--k', type=int, default=policy['k'])
    ap.add_argument('--ledger', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--semantic', action='store_true')
    a = ap.parse_args(argv)
    if a.n < 1 or a.k < 1:
        ap.error('--n and --k must be positive')
    if bool(a.from_work) == bool(a.inputs):
        ap.error('Supply either inputs or --from-work')
    if not a.from_work and not a.offline and a.ledger is None:
        ap.error('Paid gates require --ledger')
    packs = a.packs.split(',') if a.packs else None
    if not packs:
        if not a.from_work:
            ap.error('--packs is required for preparing a gate')
        from importlib.metadata import entry_points
        available = sorted({ep.name for ep in entry_points(group='haenv.pack_readouts')})
        packs = [p for p in available if (a.from_work / p).is_dir()]
        if not packs:
            ap.error('Cannot identify packs; specify --packs')
    models = a.models.split(',') if a.models else None
    out = (a.out or (a.from_work / 'gate-report' if a.from_work else Path('gate-work'))).resolve()
    work = a.from_work.resolve() if a.from_work else out / 'work'
    selected = {}
    if not a.from_work:
        from haenv.reporting.readout_tools import tool
        repeat = tool('reliability_passk')
        prepare_sample_dir, _run_samples = repeat.prepare_sample_dir, repeat._run_samples
        if len(a.inputs) not in (1, len(packs)):
            ap.error('Supply one packs root or one job per pack')
        prepared = []
        for i, p in enumerate(packs):
            source = a.inputs[0 if len(a.inputs) == 1 else i].resolve()
            if source.is_file():
                job, batch = find_job_source(source)
            else:
                job, batch = find_source(source, p, a.n)
            selected[p] = models or policy['pack_models'].get(p) or policy['models']
            if not selected[p] and not a.offline:
                ap.error('Specify --models or gate.models / gate.pack_models')
            target_job, target = prepare(job, batch, work / p, a.n)
            samples = [target.with_name(f'{target.name}-pk{r}') for r in range(1, a.k + 1)]
            for sample in samples:
                prepare_sample_dir(target, sample)
            prepared.append((p, target_job, samples))
        identity = {'packs': packs, 'models': selected, 'n': a.n, 'k': a.k, 'semantic': a.semantic, 'offline': a.offline,
                    'jobs': {p: hashlib.sha256(j.read_bytes()).hexdigest() for p, j, _ in prepared}}
        manifest = out / 'gate-work.json'
        if manifest.is_file() and json.loads(manifest.read_text()) != identity:
            ap.error('Gate parameters changed; use a new --out directory')
        manifest.parent.mkdir(parents=True, exist_ok=True); manifest.write_text(json.dumps(identity, indent=2) + '\n')
        def run(item):
            p, job, samples = item
            return _run_samples(job, samples, selected[p], serial=False, total_workers=cfg['eval']['workers'],
                                offline=a.offline, budget_ledger=a.ledger, semantic=a.semantic,
                                output_root=work / p)
        with ThreadPoolExecutor(max_workers=len(prepared)) as executor:
            exits = list(executor.map(run, prepared))
        failed = any(code != 0 for codes in exits for code in codes)
    else:
        failed = False
        manifest = work.parent / 'gate-work.json'
        if manifest.is_file():
            saved = json.loads(manifest.read_text()); selected = saved.get('models', {})
    readings = {}
    for p in packs:
        chosen = models or selected.get(p) or None
        data, cases = load(work / p, chosen)
        readings[p] = analyze(declaration(p), data, cases, n=a.n, k=a.k, policy=policy, models=chosen)
        readings[p]['seed'] = seed_readout([f.parent for f in (work / p).rglob('batch.json')], cfg)
        print(f"{p}: {readings[p]['seed']['kind']} seed")
        for warning in readings[p]['seed']['warnings']:
            print(f"WARNING [{p}] {warning}")
    result = {'schema': 1, 'meta': {'bootstrap': policy['bootstrap'], 'seed': policy['seed'],
                                   'method': 'question-cluster paired bootstrap; percentile 95% CI'}, 'packs': readings}
    if a.baseline:
        result['comparison'] = compare(readings, json.loads(a.baseline.read_text()))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'gate.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'gate.md').write_text(render(result), encoding='utf-8')
    for p, d in readings.items():
        g = d['gate']; print(f"{p}: {g['verdict']} · all right {g['G2_all_right']['all_right']}/{d['n_questions']} · G1={g['G1_strongest_raw']['pass']} G2={g['G2_all_right']['pass']} G3={g['G3_top2_separable']['pass']}")
    print(f"gate.md / gate.json -> {out}")
    return 1 if failed or any(d['gate']['verdict'] != 'PASS' for d in readings.values()) else 0
