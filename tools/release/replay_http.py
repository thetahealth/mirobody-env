#!/usr/bin/env python3
"""Replay released tool transcripts through HTTP and validate every returned FHIR resource.

Use --work with the released board's pool directory (the same input as
board_score.load_rows). No model calls are made. Each response gets two fresh
sessions; purchase groups use FHIR batch requests to preserve gated turn order.
The default requires all four N100 packs. --limit supports a quick rehearsal.
SYNTHETIC data, evaluation only, not medical advice.
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
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import gc
from http.client import HTTPConnection
import json
import os
import multiprocessing
import threading
import time
import tempfile
from types import SimpleNamespace

from tools.evaluation.board_score import PACKS, load_rows
from haenv.serve.fhir_server import DayMap, FHIRServer, from_fhir, request_for
from haenv.serve.session import ToolSession


def jsonl(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def released_responses(work, limit=None):
    """Use the board analysis loader's selected cells, including transport replacements."""
    tasks, questions, missing = [], {}, []
    for pack in PACKS:
        selected = load_rows(work, pack)
        questions[pack] = sorted({q for rounds in selected.values() for rows in rounds.values() for q in rows})
        candidates = {}
        for root in str(work).split(','):
            for f in glob.glob(f'{root}/*/{pack}/results/joint_dx/*/*-pk[0-9]/eval.jsonl'):
                pk = int(f.rsplit('-pk', 1)[1].split('/')[0])
                paths = [Path(f), Path(f.replace(f'-pk{pk}/eval.jsonl', f'-rep{pk}/eval.jsonl'))]
                for path in paths:
                    for row in jsonl(path):
                        chosen = selected.get(row['solver'], {}).get(pk, {}).get(row['case'])
                        if chosen and {k: v for k, v in chosen.items() if k != 'rerun_of_transport_abort'} == row:
                            candidates[(row['solver'], pk, row['case'])] = path.parent
        batches = {}
        for (model, pk, case), batch in candidates.items():
            batches.setdefault(batch, []).append((model, pk, case))
        for batch, keys in batches.items():
            if limit and len(tasks) >= limit:
                break
            responses = {(r['solver'], r['case']): True for r in jsonl(batch / 'responses.jsonl')}
            traces = {(r['solver'], r['case']): r for r in jsonl(batch / 'query_trace.jsonl')}
            payloads = {r['case_id']: {'case_id': r['case_id'], 'sp': r['sp']}
                        for r in jsonl(batch / 'payloads.jsonl')}
            for model, pk, case in keys:
                if limit and len(tasks) >= limit:
                    break
                response, trace = responses.get((model, case)), traces.get((model, case))
                if response is None and not trace:
                    # Transport-abort cells can have neither an answer nor a tool
                    # transcript. They contain no stored calls to replay.
                    if not str(selected[model][pk][case].get('overall', '')).startswith('ABORT'):
                        missing.append(f'{pack}/{model}/{pk}/{case}: missing stored response')
                    continue
                if trace is None and int(selected[model][pk][case].get('q_n_calls') or 0):
                    missing.append(f'{pack}/{model}/{pk}/{case}: missing query trace')
                    continue
                compact = {'calls': [], 'messages': [], 'limits': {}}
                if trace:
                    compact = {k: trace[k] for k in compact if k in trace}
                    compact['messages'] = [{k: m[k] for k in ('role', 'content', 'tool_calls', 'tool_call_id')
                                            if k in m and (k != 'content' or m['role'] == 'tool')}
                                           for m in trace.get('messages', [])]
                tasks.append((str(batch), pack, payloads[case], compact, model, pk, case))
        print(f'Loaded {pack}: {len(questions[pack])} questions; {len(tasks)} responses so far', flush=True)
    if limit:
        tasks = tasks[:limit]
    return tasks, questions, missing


_MODEL_CACHE = {}


def validate_resources(resource, counts, failures):
    """Validate the outer resource and every nested resource using the R4B models."""
    from fhir.resources.R4B import get_fhir_model_class
    rt = resource['resourceType']
    counts[rt] += 1
    try:
        if rt not in _MODEL_CACHE:
            _MODEL_CACHE[rt] = get_fhir_model_class(rt)
        model = _MODEL_CACHE[rt]
        model.parse_obj(resource)
    except Exception as exc:  # report schema failures and continue counting
        failures.append({'resourceType': rt, 'error': str(exc)[:1200]})
    for entry in resource.get('entry', []):
        if 'resource' in entry:
            validate_resources(entry['resource'], counts, failures)
    for child in resource.get('contained', []):
        validate_resources(child, counts, failures)


def _worker_init():
    from haenv.app.job import _load_judge_groups
    _load_judge_groups(['haenv.p2', 'haenv.p3', 'haenv.p4', 'haenv.m2'])


_RAW_CACHE = {}


def purchase_call(call):
    """The gated loop schedules only a string target with no extra arguments."""
    args = call.get('arguments')
    return (call.get('name') == 'order_test' and call.get('error') is None
            and isinstance(args, dict) and set(args) == {'target'}
            and isinstance(args['target'], str))


def replay_response(task):
    """Replay one response in order over a real loopback socket; no state crosses responses."""
    from haenv.app.artifacts.store import load_cases
    batch, pack, payload, trace, model, pk, case = task
    raw = None
    if pack == 'm2':
        if batch not in _RAW_CACHE:
            _RAW_CACHE[batch] = load_cases(Path(batch) / 'cases.jsonl')
        raw = _RAW_CACHE[batch][case]
    limits = trace.get('limits') or {}
    kwargs = {'raw': raw, 'max_points': limits.get('max_points', 400), 'max_calls': limits.get('max_calls')}
    direct = ToolSession(SimpleNamespace(**payload['sp']), **kwargs)
    remote = ToolSession(SimpleNamespace(**payload['sp']), **kwargs)
    counts, failures, diffs = Counter(), [], []
    identical = 0
    stored_diffs = 0
    stored = [m['content'] for m in trace.get('messages', []) if m.get('role') == 'tool']
    calls = list(trace.get('calls', []))
    pending = 0
    if pack == 'm2':
        returned = {m.get('tool_call_id') for m in trace.get('messages', []) if m.get('role') == 'tool'}
        round_number = 0
        for message in trace.get('messages', []):
            if message['role'] != 'assistant':
                continue
            round_number += 1
            for call in message.get('tool_calls', []):
                arguments = call.get('arguments')
                if (call.get('name') == 'order_test' and call.get('id') not in returned
                        and isinstance(arguments, dict) and set(arguments) == {'target'}
                        and isinstance(arguments['target'], str)):
                    # Budget exhaustion can stop the gated runner before the
                    # next delivery persists these purchase results. The orders
                    # were executed; replay them too, in their original group.
                    calls.append({'name': 'order_test', 'arguments': arguments,
                                  'round': round_number, 'error': None})
                    pending += 1
    # Calls is the loop's execution order (purchases are persisted on the next delivery).
    def compare(index, expected, wire):
        nonlocal identical, stored_diffs
        got = from_fhir(wire)
        if got == expected:
            identical += 1
        elif len(diffs) < 5:
            diffs.append({'pack': pack, 'model': model, 'pk': pk, 'case': case,
                          'call': calls[index], 'expected': str(expected)[:1500], 'actual': str(got)[:1500]})
        if index < len(stored) and stored[index] != expected:
            stored_diffs += 1

    with FHIRServer(('127.0.0.1', 0), remote) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        thread.start()
        conn = HTTPConnection('127.0.0.1', server.server_port, timeout=60)

        def request(method, path, body=None):
            conn.request(method, path, body=None if body is None else json.dumps(body).encode(),
                         headers={'Content-Type': 'application/fhir+json'})
            response = conn.getresponse()
            wire = json.loads(response.read())
            validate_resources(wire, counts, failures)
            return wire

        try:
            request('GET', '/metadata')
            i = 0
            while i < len(calls):
                call = calls[i]
                if purchase_call(call):
                    group = []
                    while i + len(group) < len(calls):
                        nxt = calls[i + len(group)]
                        if not purchase_call(nxt) or nxt['round'] != call['round']:
                            break
                        group.append(nxt)
                    entries = []
                    targets = [c['arguments']['target'] for c in group]
                    expected = direct.order_tests(targets)
                    direct.n_calls += len(group)
                    for c in group:
                        method, path, body = request_for('order_test', c['arguments'], remote.patient_id)
                        entries.append({'request': {'method': method, 'url': path.lstrip('/')}, 'resource': body})
                    wire = request('POST', '/', {'resourceType': 'Bundle', 'type': 'batch', 'entry': entries})
                    for j, value in enumerate(expected):
                        compare(i + j, value, wire['entry'][j]['resource'])
                    i += len(group)
                    continue
                args = call['arguments']
                if isinstance(args, str):
                    try:
                        args = json.loads(args or '{}')
                    except ValueError:
                        pass  # The function dispatcher must see malformed JSON verbatim.
                if args is None:
                    args = {}
                expected, _ = direct.call(call['name'], args)
                method, path, body = request_for(call['name'], args, remote.patient_id,
                                               count=remote.max_points, error=bool(call.get('error')))
                wire = request(method, path, body)
                compare(i, expected, wire)
                i += 1
        finally:
            conn.close()
            server.shutdown()
            thread.join()
    return {'calls': len(calls), 'identical': identical, 'differing': len(calls) - identical,
            'pending_purchases_replayed': pending,
            'stored_differing': stored_diffs, 'resources': dict(counts), 'validation_failures': len(failures),
            'failure_examples': failures[:3], 'diffs': diffs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--work', help='Released board pool root(s), comma separated')
    inputs.add_argument('--offline', action='store_true', help='Replay the self-contained stored-session subset shipped with the source')
    parser.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--out', type=Path, default=Path('replay-http-results.json'))
    args = parser.parse_args(argv)
    try:
        import fhir.resources.R4B  # noqa: F401
    except ImportError:
        parser.error('Validation requires haenv[serve] (fhir.resources==7.1.0)')
    start = time.monotonic()
    # JSON transcripts contain no cycles. Avoid repeatedly walking the growing
    # immutable input graph while loading the release.
    gc.disable()
    offline = tempfile.TemporaryDirectory(prefix='haenv-http-replay-') if args.offline else None
    if args.offline:
        fixture = Path(__file__).resolve().parents[2] / 'benchmarks/results/1.3.1/http-replay.json'
        data = json.loads(fixture.read_text())
        batch = Path(offline.name)
        (batch / 'cases.jsonl').write_text(json.dumps(data['case'], ensure_ascii=False) + '\n')
        tasks = [(str(batch), data['pack'], data['payload'], data['trace'], data['model'], data['repeat'], data['question'])]
        questions, missing = {p: [data['question']] if p == data['pack'] else [] for p in PACKS}, []
    else:
        tasks, questions, missing = released_responses(args.work, args.limit)
    if not args.offline and not args.limit and any(len(questions[p]) != 100 for p in PACKS):
        parser.error(f'Expected four N100 packs; found { {p: len(q) for p, q in questions.items()} }')
    summary = {'questions': sum(map(len, questions.values())),
               'questions_per_pack': {p: len(q) for p, q in questions.items()},
               'responses': len(tasks), 'calls': 0, 'identical': 0, 'differing': 0,
               'pending_purchases_replayed': 0,
               'stored_differing': 0, 'validation_failures': 0, 'resources': Counter(),
               'diffs': [], 'failure_examples': [], 'missing': missing, 'worker_errors': []}
    # These immutable transcripts are large. Freeze them before forking so a
    # worker's cyclic GC never scans and copies the parent's entire data graph.
    gc.collect()
    gc.freeze()
    gc.enable()
    context = multiprocessing.get_context('fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn')
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, mp_context=context) as pool:
        pending = {pool.submit(replay_response, task): task for task in tasks}
        for n, future in enumerate(as_completed(pending), 1):
            try:
                result = future.result()
            except Exception as exc:
                task = pending[future]
                summary['worker_errors'].append({'pack': task[1], 'case': task[-1], 'error': repr(exc)})
                continue
            for key in ('calls', 'identical', 'differing', 'stored_differing', 'validation_failures',
                        'pending_purchases_replayed'):
                summary[key] += result[key]
            summary['resources'].update(result['resources'])
            if result['diffs'] and not summary['diffs']:
                print('First difference: ' + json.dumps(result['diffs'][0], ensure_ascii=False), flush=True)
            for key in ('diffs', 'failure_examples'):
                summary[key] = (summary[key] + result[key])[:5]
            if n % 100 == 0:
                print(f"{n}/{len(tasks)} responses; {summary['calls']} calls; "
                      f"{summary['differing']} differing; {summary['validation_failures']} validation failures", flush=True)
    summary['resources'] = dict(sorted(summary['resources'].items()))
    summary['elapsed_seconds'] = round(time.monotonic() - start, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if offline:
        offline.cleanup()
    return int(bool(summary['differing'] or summary['stored_differing'] or summary['validation_failures']
                    or missing or summary['worker_errors']))


if __name__ == '__main__':
    raise SystemExit(main())
