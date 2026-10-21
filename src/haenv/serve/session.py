"""One question's tool session, shared by the HTTP and MCP transports.

Purchases run through the existing gated runner. Replaying the short purchase prefix
keeps its billing, synthesis, refusal and round policy in one implementation.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from haenv.interaction.query.adapter import adapter_for, load_pack_adapters
from haenv.interaction.query.tools import RecordStore, ToolSpec, call_tool


class _Captured(Exception):
    """Stop the gated runner at the next solver delivery without making a model call."""


class ToolSession:
    """A record store and private purchase state; calls are serialized per session."""

    def __init__(self, payload, *, raw=None, max_points=400, max_calls=None,
                 purchase_rounds=None, budget=None):
        load_pack_adapters()
        self.lock = RLock()
        self.payload, self.raw = payload, raw
        self.max_points, self.max_calls, self.n_calls = max_points, max_calls, 0
        self.purchase_rounds, self.budget = purchase_rounds, budget
        self.groups: list[list[str]] = []
        self.gc = None
        self.gated = isinstance(payload.prediction_context.get('m2_frame'), dict)
        if self.gated:
            if raw is None:
                raise ValueError('An M2 session needs cases.jsonl as well as payloads.jsonl')
            self._refresh()
        else:
            self.store = RecordStore(payload, payload.prediction_context['prediction_time_T'],
                                     adapter_for(payload), max_points=max_points)

    @property
    def patient_id(self):
        return str(self.store.case_id or 'patient')

    def _refresh(self):
        from haenv.interaction.gated import DEFAULT_BUDGET, MAX_ROUNDS, run_gated
        from haenv.interaction.query.gated import _specs
        from haenv.m2.query import ADAPTER
        outer = self
        events = []

        class Recorder:
            def append(self, kind, data, **kwargs):
                events.append((kind, data))

        class Solver:
            def solve(self, payload):
                self.payload = payload
                n = int(self.gated_context['round']) - 1
                if n >= len(outer.groups):
                    raise _Captured
                kinds = {i['target']: i['kind'] for i in self.gated_context['menu']}
                return SimpleNamespace(_raw_text=json.dumps({'queries': [
                    {'target': t, 'kind': kinds.get(t, 'test')} for t in outer.groups[n]]}))

        solver = Solver()
        T = int(self.payload.prediction_context['prediction_time_T'])
        rounds = self.purchase_rounds or MAX_ROUNDS
        try:
            _, trace = run_gated(self.raw, T, solver,
                                budget=DEFAULT_BUDGET if self.budget is None else self.budget,
                                max_rounds=rounds, trace=Recorder())
        except _Captured:
            payload, gc = solver.payload, dict(solver.gated_context)
        else:
            if trace.leak:
                raise ValueError('The gated record failed the existing leakage guard')
            # Exhaustion can stop the runner before its next delivery. Recover the
            # same revealed records from its own result events, never from gold.
            import copy
            payload = copy.deepcopy(solver.payload)
            for kind, data in events:
                if kind == 'tool/result' and data.get('observation'):
                    target = next(d['target'] for k, d in events
                                  if k == 'tool/call' and d['call_id'] == data['call_id'])
                    payload.longitudinal_data[target] = data['observation']
            gc = {**solver.gated_context, 'spent': trace.spent,
                  'round': min(len(self.groups) + 1, rounds),
                  'truncated_last_round': trace.truncated_targets,
                  'unavailable_results': trace.unavailable_results,
                  'menu': [i for i in trace.menu if i['target'] not in payload.longitudinal_data
                           and i['target'] not in trace.unavailable_results]}
        self.gc = gc
        self.store = RecordStore(payload, T, ADAPTER, max_points=self.max_points)
        self._gated_specs = _specs(self.store, self.gc,
                                   int(gc['round']) < int(gc['max_rounds']))

    def specs(self):
        if not self.gated:
            if getattr(self, '_spec_store', None) is not self.store:
                self._common_specs = self.store.specs()
                self._spec_store = self.store
            return self._common_specs
        return [ToolSpec(s.name, s.description, s.params,
                         self.order_test if s.name == 'order_test' else s.fn)
                for s in self._gated_specs]

    def order_tests(self, targets):
        """Execute one purchase round in order, including refusals, using run_gated."""
        from haenv.interaction.query.gated import _purchase_result
        if not self.gated:
            raise ValueError('This question has no purchase menu')
        if int(self.gc['round']) >= int(self.gc['max_rounds']):
            return [{'error': 'Purchase rounds exhausted'} for _ in targets]
        self.groups.append(list(targets))
        self._refresh()
        return [_purchase_result(t, self.store, self.gc) for t in targets]

    def order_test(self, target):
        # The gated query loop only schedules string targets. Other values reach
        # its placeholder ToolSpec function, which returns {} without a purchase.
        if not isinstance(target, str):
            return {}
        return self.order_tests([target])[0]

    def call(self, name, arguments):
        """Use the function-call dispatcher verbatim, with the optional loop call limit."""
        with self.lock:
            if self.max_calls is not None and self.n_calls >= self.max_calls:
                return {'error': '调用次数已用完'}, 'over_budget'
            self.n_calls += 1
            return call_tool({s.name: s for s in self.specs()}, name, arguments)


def pack_directory(pack):
    """Accept a saved batch directory or a pack manifest with a pack_batch path."""
    path = Path(pack)
    if path.is_dir():
        return path
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('pack_batch'), str):
        raise ValueError('A pack manifest must contain a pack_batch directory')
    batch = Path(manifest['pack_batch'])
    return batch if batch.is_absolute() or batch.exists() else path.parent / batch


def load_question(pack, question, **kwargs):
    """Load the stored solver payload; only the gated runner receives the raw case."""
    from haenv.app.artifacts.store import load_cases
    from haenv.app.job import _load_judge_groups
    from haenv.app.cli import load_cfg
    query = load_cfg()['query']
    for key in ('max_points', 'max_calls'):
        kwargs.setdefault(key, query[key])
    batch = pack_directory(pack)
    with (batch / 'payloads.jsonl').open(encoding='utf-8') as stream:
        records = (json.loads(line) for line in stream if line.strip())
        row = next((r for r in records if r['case_id'] == question or r['sp'].get('case_id') == question), None)
    if row is None:
        raise ValueError(f'Unknown question: {question}')
    # Register the same pack hooks used by a tool session, including M2 synthesis.
    pc = row['sp'].get('prediction_context') or {}
    groups = [('triage_p2', 'haenv.p2'), ('meds_p3', 'haenv.p3'),
              ('followup_p4', 'haenv.p4'), ('m2_frame', 'haenv.m2')]
    _load_judge_groups([group for key, group in groups if key in pc])
    payload = SimpleNamespace(**row['sp'])
    raw = load_cases(batch / 'cases.jsonl').get(row['case_id']) if 'm2_frame' in pc else None
    return ToolSession(payload, raw=raw, **kwargs)
