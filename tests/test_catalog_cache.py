"""Price preflight shares one locked, expiring catalog without paid requests."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pytest
from haenv import solver_accounting as accounting


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    import haenv.solvers as solvers
    monkeypatch.setattr(solvers, '_ensure_backends_registered', lambda cfg: None)
    monkeypatch.setattr(solvers, 'BACKENDS', {'openrouter': {'url': 'http://local.test/chat/completions', 'key_env': 'KEY'}})
    monkeypatch.setattr(solvers, 'load_env_file', lambda _: {'KEY': 'fake'})
    cfg = {'paid': {'catalog_cache_ttl_s': 60}}
    calls = []
    def fetch(*a, **k):
        calls.append(1)
        time.sleep(.02)
        return [{'id': 'model', 'pricing': {'prompt': '.001'}}]
    monkeypatch.setattr(accounting, '_read_with_retries', fetch)
    return cfg, calls


def test_simultaneous_preflights_fetch_once_and_keep_metadata(tmp_path, catalog):
    cfg, calls = catalog
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(lambda _: accounting._catalog_for_ledger(cfg, tmp_path / 'budget.json'), range(24)))
    assert len(calls) == 1
    assert all(row == rows[0] for row in rows)
    assert rows[0][0]['pricing'] == {'prompt': '.001'}
    # Distinct ledgers in the same directory share the same catalog.
    accounting._catalog_for_ledger(cfg, tmp_path / 'other.json')
    assert len(calls) == 1


@pytest.mark.parametrize('defect', ['expired', 'corrupt', 'missing', 'future'])
def test_unusable_cache_refetches(tmp_path, catalog, defect):
    cfg, calls = catalog
    accounting._catalog_for_ledger(cfg, tmp_path / 'budget.json')
    cache = next((tmp_path / 'price-catalogs').glob('*.json'))
    saved = json.loads(cache.read_text())
    if defect == 'missing':
        cache.unlink()
    elif defect == 'corrupt':
        cache.write_text('broken')
    else:
        saved['fetched_at'] += -1000 if defect == 'expired' else 1000
        cache.write_text(json.dumps(saved))
    accounting._catalog_for_ledger(cfg, tmp_path / 'budget.json')
    assert len(calls) == 2


def test_expired_cache_is_never_used_after_fetch_failure(tmp_path, catalog, monkeypatch):
    cfg, _ = catalog
    accounting._catalog_for_ledger(cfg, tmp_path / 'budget.json')
    cfg['paid']['catalog_cache_ttl_s'] = 0
    monkeypatch.setattr(accounting, '_read_with_retries', lambda *a, **k: None)
    with pytest.raises(ValueError, match='Could not verify'):
        accounting._catalog_for_ledger(cfg, tmp_path / 'budget.json')


def test_changed_endpoint_does_not_share_catalog(tmp_path, catalog, monkeypatch):
    import haenv.solvers as solvers
    cfg, calls = catalog
    accounting._catalog_for_ledger(cfg, tmp_path / 'budget.json')
    solvers.BACKENDS['openrouter']['url'] = 'http://different.test/chat/completions'
    accounting._catalog_for_ledger(cfg, tmp_path / 'budget.json')
    assert len(calls) == 2


def test_processes_share_first_fetch(tmp_path):
    script = '''import sys, time
from pathlib import Path
import haenv.solvers as solvers
import haenv.solver_accounting as accounting
root=Path(sys.argv[1])
solvers._ensure_backends_registered=lambda cfg: None
solvers.BACKENDS={'openrouter':{'url':'http://local.test/chat/completions','key_env':'KEY'}}
solvers.load_env_file=lambda _: {'KEY':'fake'}
def fetch(*a, **k):
 with (root/'fetches').open('a') as f: f.write('fetch\\n')
 time.sleep(.1)
 return [{'id':'model'}]
accounting._read_with_retries=fetch
assert accounting._catalog_for_ledger({},root/'budget.json')==[{'id':'model'}]
'''
    processes = [subprocess.Popen([sys.executable, '-c', script, str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(6)]
    for process in processes:
        out, err = process.communicate(timeout=30)
        assert process.returncode == 0, err.decode()
    assert (tmp_path / 'fetches').read_text().splitlines() == ['fetch']


def test_solver_and_judge_share_the_same_catalog(tmp_path, catalog, monkeypatch):
    from haenv import semantic_pipeline as pipeline
    cfg, calls = catalog
    cfg['models'] = {'judge': {'model': 'model', 'backend': 'openrouter'}}
    policy = {'judge': {'model_key': 'judge', 'model_id': 'model', 'backend': 'openrouter'}}
    endpoints = [{'provider_name': 'synthetic', 'pricing': {'prompt': '.001'}}]
    monkeypatch.setattr(accounting, 'fetch_endpoints', lambda *a: endpoints)
    ledger = tmp_path / 'budget.json'
    solver_metadata = accounting._catalog_for_ledger(cfg, ledger)
    def judge(_):
        with accounting.catalog_scope(ledger, cfg):
            return pipeline.fetch_prices(cfg, policy)
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(judge, range(24)))
    assert len(calls) == 1
    assert all(r == {**solver_metadata[0], 'endpoints': endpoints} for r in rows)
    assert accounting._CATALOG_DIRECTORY.get() is None


def test_scoped_judge_keeps_the_frozen_pipeline_and_resets_context(tmp_path, monkeypatch):
    from haenv import semantic_pipeline as pipeline
    monkeypatch.setattr(pipeline, 'validate_run', lambda _: {
        'policy': {'judge': {'max_concurrency': 3}}})
    class VerifiedScope(Exception):
        pass
    def prices(*a):
        assert accounting._CATALOG_DIRECTORY.get() == tmp_path / 'price-catalogs'
        raise VerifiedScope
    monkeypatch.setattr(pipeline, 'fetch_prices', prices)
    with pytest.raises(VerifiedScope), accounting.catalog_scope(tmp_path / 'budget.json', {}):
        pipeline.execute_run(tmp_path, {}, tmp_path / 'budget.json', limit_usd='1', concurrency=1)
    assert accounting._CATALOG_DIRECTORY.get() is None


def test_catalog_scope_never_intercepts_posts_or_other_gets(tmp_path, catalog):
    import urllib.request
    from haenv.transport import ModelOpener
    cfg, fetches = catalog
    opener = ModelOpener()
    forwarded = []
    class Wire:
        def open(self, request, *a):
            forwarded.append(request)
            return 'original transport'
    opener.env = Wire()
    requests = [
        urllib.request.Request('http://local.test/models', data=b'body', headers={'Authorization': 'Bearer fake'}),
        urllib.request.Request('http://local.test/chat/completions', headers={'Authorization': 'Bearer fake'}),
        urllib.request.Request('http://local.test/models'),
    ]
    with accounting.catalog_scope(tmp_path / 'budget.json', cfg):
        for request in requests:
            assert opener.open(request) == 'original transport'
    assert forwarded == requests and fetches == []


@pytest.mark.parametrize('ttl', [-1, float('inf'), float('nan')])
def test_cache_ttl_must_be_finite_and_nonnegative(ttl):
    from haenv.config import validate
    with pytest.raises(ValueError):
        validate({'paid': {'catalog_cache_ttl_s': ttl}})
