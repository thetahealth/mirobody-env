"""Durable cell starts and recovery receipts, independent of scoring and ABORT policy."""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
import fcntl
import itertools
import json
import os
from pathlib import Path
import threading
import time


def records(path):
    if not path.is_file():
        return []
    # Only a torn final append is ignored. Invalid complete lines are errors.
    data = path.read_bytes()
    end = data.rfind(b'\n') + 1
    return [json.loads(line) for line in data[:end].split(b'\n') if line.strip()]


def cell_of(task, model):
    # Evaluation's scheduler task protocol: case ID is the first tuple member.
    if isinstance(task, (tuple, list)) and task and isinstance(task[0], str):
        return task[0], model
    if isinstance(task, dict) and isinstance(task.get('case'), str):
        return task['case'], model
    return None


class Recovery:
    def __init__(self, batch, tasks, model_of):
        self.batch = Path(batch) if batch is not None else None
        self.lock = threading.Lock()
        self.model_of = model_of
        self.recovered, self.failed, self.started = {}, {}, set()
        self.interrupted, self.attempts = set(), Counter()
        if self.batch is None:
            return
        self.batch.mkdir(parents=True, exist_ok=True)
        latest = {(r['case'], r['solver']): r for r in records(self.batch / 'eval.jsonl')}
        for event in records(self.batch / 'cells.jsonl'):
            key = event['case'], event['solver']
            if event['event'] == 'started':
                self.attempts[key] += 1
                self.interrupted.add(key)
        # Older batches have no cell-start journal; a partial saved response is evidence
        # that a cell had started. A complete eval row always remains authoritative.
        for row in records(self.batch / 'responses.jsonl'):
            if row.get('case') and row.get('solver'):
                self.interrupted.add((row['case'], row['solver']))
        accounting = self.batch / 'solver-accounting'
        for receipt in accounting.glob('receipts/*.json'):
            row = json.loads(receipt.read_text())
            cell = row.get('cell') or {}
            if cell.get('case_id') and cell.get('solver'):
                self.interrupted.add((cell['case_id'], cell['solver']))
        manifest = accounting / 'manifest.json'
        if manifest.is_file():
            from .solver_accounting import cell_v2, _hash
            saved = json.loads(manifest.read_text())
            mapped = {}
            for task in tasks:
                cell = cell_of(task, model_of(task))
                if cell is None:
                    continue
                if saved.get('batch_uid'):
                    mapped[cell_v2(saved['batch_uid'], *cell)] = cell
                if saved.get('run_id'):
                    mapped[_hash([saved['run_id'], *cell])] = cell
            for marker in itertools.chain(accounting.glob('receipts/*.sent'),
                                          accounting.glob('receipts/*.inflight.json')):
                rid = json.loads(marker.read_text()).get('request_id', '').split(':resend:', 1)[0]
                parts = rid.split(':')
                identity = parts[2] if len(parts) == 5 and parts[:2] == ['solver', 'v2'] else parts[1] if len(parts) == 3 and parts[0] == 'solver' else None
                if identity in mapped:
                    self.interrupted.add(mapped[identity])
        self.interrupted -= set(latest)
        planned = {cell_of(t, model_of(t)) for t in tasks}
        self.interrupted &= planned
        for key, row in latest.items():
            if str(row.get('overall', '')).startswith('ABORT') or row.get('overall') == 'ERROR':
                self.failed[key] = {'case': key[0], 'solver': key[1], 'overall': row.get('overall'),
                                    'attempts': self.attempts[key],
                                    'failed_requests': len(row.get('failed_attempts') or [])}

    def start(self, task):
        cell = cell_of(task, self.model_of(task))
        if self.batch is None or cell is None:
            return
        with self.lock:
            self.started.add(cell); self.attempts[cell] += 1
            event = {'event': 'started', 'case': cell[0], 'solver': cell[1], 'pid': os.getpid(), 'at': time.time()}
            with (self.batch / 'cells.jsonl').open('ab') as file:
                file.write((json.dumps(event) + '\n').encode()); file.flush(); os.fsync(file.fileno())

    def emitted(self, row):
        if self.batch is None or not row.get('case') or not row.get('solver'):
            return
        key = row['case'], row['solver']
        entry = {'case': key[0], 'solver': key[1], 'overall': row.get('overall'), 'attempts': self.attempts[key]}
        with self.lock:
            if key in self.interrupted:
                self.recovered[key] = entry
            if str(row.get('overall', '')).startswith('ABORT') or row.get('overall') == 'ERROR':
                entry['failed_requests'] = len(row.get('failed_attempts') or [])
                self.failed[key] = entry
            else:
                self.failed.pop(key, None)

    def finish(self):
        if self.batch is None:
            return
        from .paid_slots import persist_json
        result = {'interrupted': [{'case': c, 'solver': m} for c, m in sorted(self.interrupted)],
                  'recovered': [self.recovered[k] for k in sorted(self.recovered)],
                  'failed_cells': [self.failed[k] for k in sorted(self.failed)],
                  'pending': [{'case': c, 'solver': m} for c, m in sorted(self.interrupted - self.recovered.keys())]}
        persist_json(self.batch / 'resume.json', result)
        if self.interrupted or self.failed:
            print('[haenv] resume receipt: ' + json.dumps(result, ensure_ascii=False), flush=True)


@contextmanager
def recovery(batch, tasks, model_of):
    if batch is None:
        yield Recovery(None, tasks, model_of)
        return
    batch = Path(batch); batch.mkdir(parents=True, exist_ok=True)
    with (batch / 'cells.lock').open('a') as file:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another scheduler is running this batch; resume after it exits') from None
        state = Recovery(batch, tasks, model_of)
        try:
            yield state
        finally:
            state.finish()
