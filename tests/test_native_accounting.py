from datetime import date
from decimal import Decimal
import json

import pytest

from haenv.native_accounting import NativePrices
from haenv.paid_completion import AccountedCompletion
from haenv.semantic_budget import BudgetLedger, BudgetExceeded


@pytest.fixture(autouse=True)
def verified_day(monkeypatch):
    monkeypatch.setattr('haenv.native_accounting.utc_today',lambda:date(2026,9,29))


def prices(model='gemini-3.1-pro-preview',backend='google'):
    return NativePrices.verified(model,backend,today=date(2026,9,29))


def test_google_cache_hit_is_priced_at_the_cached_rate_and_declared():
    p=prices()
    bill=p.cost_receipt({'usageMetadata':{'promptTokenCount':1000,'candidatesTokenCount':100,
        'thoughtsTokenCount':200,'totalTokenCount':1300,'cachedContentTokenCount':900}})
    assert bill['actual_usd'] is None
    # prompt <= 200k => low tier: 100 fresh x $2 + 900 cached x $0.20 + 300 out x $12, per million
    assert Decimal(bill['upper_bound_usd'])==Decimal('.00398')
    assert bill['output_tokens']==300 and bill['cache_discount_applied'] is True
    assert bill['cached_tokens']==900 and bill['cache_input_per_million']=='0.20'
    assert bill['cache_verified_on']=='2026-09-29' and 'ai.google.dev' in bill['cache_source']


def test_google_without_cache_hit_is_full_price_and_says_so():
    p=prices()
    for extra in ({}, {'cachedContentTokenCount':0}):
        bill=p.cost_receipt({'usageMetadata':{'promptTokenCount':1000,'candidatesTokenCount':100,
            'thoughtsTokenCount':200,'totalTokenCount':1300,**extra}})
        assert Decimal(bill['upper_bound_usd'])==Decimal('.0056')   # 1000 x $2 + 300 x $12
        assert bill['cache_discount_applied'] is False


def test_qwen_bound_uses_native_currency_and_never_fake_dollar_receipt():
    p=prices('qwen3.7-flash','dashscope')
    bill=p.cost_receipt({'usage':{'prompt_tokens':1000,'completion_tokens':500,'total_tokens':1500,
                                'completion_tokens_details':{'reasoning_tokens':400}}})
    assert bill['output_tokens']==500  # reasoning is already in completion_tokens
    # input <= 32k tier: 1000 x 0.2 + 500 x 0.8 CNY per million
    assert Decimal(bill['native_currency_upper_bound'])==Decimal('.0006')
    assert Decimal(bill['upper_bound_usd'])==Decimal('.00012')
    assert bill['actual_usd'] is None and bill['currency']=='CNY'
    assert bill['cache_discount_applied'] is False


def test_qwen_implicit_cache_hit_is_priced_at_the_hit_rate():
    p=prices('qwen3.7-flash','dashscope')
    bill=p.cost_receipt({'usage':{'prompt_tokens':1000,'completion_tokens':500,'total_tokens':1500,
                                'prompt_tokens_details':{'cached_tokens':800}}})
    # input <= 32k tier: (200 x 0.2 + 800 x 0.04 + 500 x 0.8) CNY per million = 0.000472
    assert Decimal(bill['native_currency_upper_bound'])==Decimal('.000472')
    assert Decimal(bill['upper_bound_usd'])==Decimal('.0000944')
    assert bill['cache_discount_applied'] is True and bill['cached_tokens']==800
    assert bill['cache_input_per_million']=='0.04'


def test_google_usage_keeps_the_cached_token_field_on_disk_shape():
    from haenv.evaluate import _google_usage, billed_tokens
    u=_google_usage({'usageMetadata':{'promptTokenCount':1000,'candidatesTokenCount':100,
        'totalTokenCount':1100,'cachedContentTokenCount':900}})
    assert u['prompt_tokens_details']['cached_tokens']==900
    assert billed_tokens(u)['in_cached']==900 and billed_tokens(u)['in_fresh']==100


@pytest.mark.parametrize('usage',[{}, {'promptTokenCount':100,'totalTokenCount':90},
    {'promptTokenCount':100,'candidatesTokenCount':50,'totalTokenCount':120},
    {'promptTokenCount':True,'totalTokenCount':110},
    {'promptTokenCount':100,'totalTokenCount':110,'toolUsePromptTokenCount':1}])
def test_incomplete_or_impossible_google_usage_rejected(usage):
    with pytest.raises(ValueError):prices().cost_receipt({'usageMetadata':usage})


def test_unknown_model_and_expired_schedule_fail_before_dispatch():
    with pytest.raises(ValueError):NativePrices.verified('other','google',today=date(2026,9,29))
    with pytest.raises(ValueError):NativePrices.verified('gemini-3.1-pro-preview','google',today=date(2026,10,6))
    with pytest.raises(ValueError):prices().upper_bound('q',72000)


def test_native_adapter_replays_nonactual_bound_without_rebuying(tmp_path,monkeypatch):
    p=prices();ledger=BudgetLedger(tmp_path/'budget.json',limit_usd='1')
    paid=AccountedCompletion(p,ledger,tmp_path/'receipts',backend='google',cost_reader=p.cost_receipt)
    response={'usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':50,'totalTokenCount':150}}
    settings={'backend':'google','model':p.model,'max_tokens':1000}
    paid.request('r','q',settings,lambda _:response)
    assert ledger.snapshot()['settled_usd']=='0'
    assert ledger.snapshot()['requests']['r']['status']=='tariff_capped'
    paid.cost_reader=lambda _:pytest.fail('recalculated receipt during replay')
    assert paid.request('r','q',settings,lambda _:pytest.fail('duplicate'))==response


def test_unknown_native_usage_counts_at_bound_instead_of_free_call(tmp_path):
    p=prices();ledger=BudgetLedger(tmp_path/'budget.json',limit_usd='1')
    paid=AccountedCompletion(p,ledger,tmp_path/'receipts',backend='google',cost_reader=p.cost_receipt)
    assert paid.request('r','q',{'backend':'google','model':p.model,'max_tokens':1000},lambda _: {})=={}
    record=ledger.snapshot()['requests']['r']
    assert record['status']=='unknown_capped' and Decimal(ledger.snapshot()['committed_usd'])==Decimal(record['reserved_usd'])>0


def test_google_actual_solver_http_is_accounted_before_dispatch(tmp_path,monkeypatch):
    from haenv import evaluate as ev
    from haenv.solver_accounting import prepare_accounting
    import haenv.native_accounting as native
    s=ev.GoogleSolver('gemini-3.1-pro','gemini-3.1-pro-preview','secret',10,max_tokens=65536)
    monkeypatch.setattr(native,'preflight_native',lambda *a:prices())
    run=prepare_accounting([('gemini-3.1-pro',ev._const(s))],{},tmp_path,
        ledger_path=tmp_path/'budget.json',limit_usd='10',identity={})
    run.bind(s,'case','gemini-3.1-pro')
    def wire(prompt):
        assert any(r['status']=='reserved' for r in run.ledger.snapshot()['requests'].values())
        return {'usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':10,'totalTokenCount':110}}
    monkeypatch.setattr(s,'_post_wire',wire)
    s._post('q')
    assert run.ledger.snapshot()['settled_usd']=='0'
    assert Decimal(run.ledger.snapshot()['tariff_bound_usd'])>0


def test_dashscope_binding_counts_native_tariff_under_shared_cap(tmp_path,monkeypatch):
    from haenv import evaluate as ev
    from haenv.solver_accounting import prepare_accounting
    import haenv.native_accounting as native
    s=ev.OpenAICompatSolver('qwen3.7-flash','qwen3.7-flash','secret',10,max_tokens=72000,
                            backend='dashscope',url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
    monkeypatch.setattr(native,'preflight_native',lambda *a:prices('qwen3.7-flash','dashscope'))
    run=prepare_accounting([('qwen3.7-flash',ev._const(s))],{},tmp_path,
        ledger_path=tmp_path/'budget.json',limit_usd='10',identity={})
    run.bind(s,'case','qwen3.7-flash')
    monkeypatch.setattr(s,'_post_wire',lambda _: {'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}})
    s._post('q')
    assert run.ledger.snapshot()['settled_usd']=='0'
    assert Decimal(run.ledger.snapshot()['tariff_bound_usd'])>0


def test_native_host_and_live_google_capacity_must_match(monkeypatch):
    from types import SimpleNamespace
    from haenv.native_accounting import preflight_native
    s=SimpleNamespace(model='gemini-3.1-pro-preview',api_key='secret',max_tokens=65536)
    with pytest.raises(ValueError,match='endpoint'):
        preflight_native(s,'google','https://different-provider.invalid')
    class Response:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self):return json.dumps({'name':'models/'+s.model,'outputTokenLimit':1000,'inputTokenLimit':1048576}).encode()
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:Response())
    with pytest.raises(ValueError,match='capacity'):
        preflight_native(s,'google','https://generativelanguage.googleapis.com/v1beta/models')


def test_unverified_google_capacity_says_why_without_the_key(monkeypatch):
    import urllib.error
    from types import SimpleNamespace
    from haenv.native_accounting import preflight_native
    from haenv.transport import PreSendFailure
    s=SimpleNamespace(model='gemini-3.1-pro-preview',api_key='secret-key',max_tokens=65536)
    url='https://generativelanguage.googleapis.com/v1beta/models'
    def refused(*a,**k):raise urllib.error.HTTPError(url,403,'Forbidden',{},None)
    monkeypatch.setattr('urllib.request.urlopen',refused)
    with pytest.raises(ValueError,match='HTTP 403') as error:
        preflight_native(s,'google',url)
    assert 'secret-key' not in str(error.value)
    def unreachable(*a,**k):raise urllib.error.URLError(PreSendFailure(TimeoutError()))
    monkeypatch.setattr('urllib.request.urlopen',unreachable)
    with pytest.raises(ValueError,match=r'not reached \(TimeoutError\)'):
        preflight_native(s,'google',url)


def test_google_capacity_check_is_bounded_as_a_whole(monkeypatch):
    # urlopen's timeout is per connection attempt; a dropped route to a many-address host
    # multiplies it, so the check has one deadline of its own.
    import threading,time
    from types import SimpleNamespace
    from haenv import native_accounting as native
    s=SimpleNamespace(model='gemini-3.1-pro-preview',api_key='k',max_tokens=65536)
    release=threading.Event()
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:release.wait(30))
    monkeypatch.setattr(native,'CAPACITY_DEADLINE_S',0.2)
    started=time.monotonic()
    try:
        with pytest.raises(ValueError,match=r'not reached \(TimeoutError\)'):
            native.preflight_native(s,'google','https://generativelanguage.googleapis.com/v1beta/models')
    finally:
        release.set()
    assert time.monotonic()-started<5


def test_google_normalized_usage_preserves_measured_cache_tokens():
    from haenv.evaluate import _google_usage,billed_tokens
    usage=_google_usage({'usageMetadata':{'promptTokenCount':1000,'candidatesTokenCount':100,
        'thoughtsTokenCount':200,'totalTokenCount':1300,'cachedContentTokenCount':900}})
    assert usage['prompt_tokens_details']['cached_tokens']==900
    assert billed_tokens(usage)['in_cached']==900
    assert billed_tokens(usage)['out']==300


def test_request_after_verified_date_refuses_before_network(tmp_path,monkeypatch):
    p=prices();ledger=BudgetLedger(tmp_path/'budget.json',limit_usd='1')
    paid=AccountedCompletion(p,ledger,tmp_path/'receipts',backend='google',cost_reader=p.cost_receipt)
    monkeypatch.setattr('haenv.native_accounting.utc_today',lambda:date(2026,10,6))
    with pytest.raises(BudgetExceeded,match='expired'):
        paid.request('r','q',{'backend':'google','model':p.model,'max_tokens':1000},lambda _:pytest.fail('expired tariff'))
    assert not ledger.snapshot()['requests']


def test_resume_after_the_verified_date_starts_but_native_dispatch_still_refuses(tmp_path,monkeypatch):
    # A resumed batch whose native models have finished must not be blocked at startup;
    # a new native request after the verified date is still refused before reserving.
    from haenv import evaluate as ev
    from haenv.solver_accounting import prepare_accounting
    s=ev.OpenAICompatSolver('qwen3.7-flash','qwen3.7-flash','secret',10,max_tokens=72000,
                            backend='dashscope',url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
    kw=dict(ledger_path=tmp_path/'budget.json',limit_usd='10',identity={})
    prepare_accounting([('qwen3.7-flash',ev._const(s))],{},tmp_path,**kw)
    monkeypatch.setattr('haenv.native_accounting.utc_today',lambda:date(2026,10,6))
    with pytest.raises(ValueError,match='expired'):   # a new batch still needs a current tariff
        prepare_accounting([('qwen3.7-flash',ev._const(s))],{},tmp_path/'fresh',**kw)
    run=prepare_accounting([('qwen3.7-flash',ev._const(s))],{},tmp_path,**kw)
    run.bind(s,'case','qwen3.7-flash')
    monkeypatch.setattr(s,'_post_wire',lambda _: pytest.fail('expired tariff dispatched'))
    with pytest.raises(BudgetExceeded,match='expired'):
        s._post('q')
    assert not run.ledger.snapshot()['requests']


def test_a_verified_schedule_is_valid_for_seven_days_and_not_before():
    for day in (29, 30):
        NativePrices.verified('gemini-3.1-pro-preview', 'google', today=date(2026, 9, day))
    for day in range(1, 6):
        NativePrices.verified('gemini-3.1-pro-preview', 'google', today=date(2026, 10, day))
    for bad in (date(2026, 9, 28), date(2026, 10, 6), date(2026, 11, 1)):
        with pytest.raises(ValueError, match='expired'):
            NativePrices.verified('gemini-3.1-pro-preview', 'google', today=bad)
