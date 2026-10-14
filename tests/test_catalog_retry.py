"""Free catalog reads retry transient failures and stop before paid dispatch."""
import io
import json
import pytest
from haenv import solver_accounting as accounting, transport


def test_transient_catalog_failures_retry_with_backoff(monkeypatch):
    waits, calls = [], []
    monkeypatch.setattr('time.sleep', waits.append)
    def read(*args, **kwargs):
        calls.append(kwargs['timeout'])
        if len(calls) < 3:
            raise TimeoutError('synthetic catalog timeout')
        return io.BytesIO(json.dumps({'data': [{'id': 'model'}]}).encode())
    monkeypatch.setattr(transport, 'model_urlopen', read)
    assert accounting._read_with_retries(object(), cfg={}) == [{'id': 'model'}]
    assert calls == [60.0] * 3
    assert waits == [5, 10]


def test_catalog_retry_exhaustion_is_bounded(monkeypatch):
    waits, calls = [], []
    monkeypatch.setattr('time.sleep', waits.append)
    def read(*args, **kwargs):
        calls.append(1)
        raise TimeoutError('synthetic catalog timeout')
    monkeypatch.setattr(transport, 'model_urlopen', read)
    assert accounting._read_with_retries(object(), cfg={}, attempts=3) is None
    assert len(calls) == 3 and waits == [5, 10]
