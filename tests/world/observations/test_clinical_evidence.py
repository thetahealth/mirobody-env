"""Critical clinical observations; synthetic designs awaiting physician review."""
from copy import deepcopy
from datetime import date, timedelta
import importlib
import json

import pytest

import haenv.resources.registry as R
from haenv.world.observations.findings_render import _value


def builder():
    # A missing implementation is an assertion failure during the test-first red run.
    from importlib.util import find_spec
    assert find_spec('haenv.world.clinical.clinical_evidence') is not None, 'critical result builder is missing'
    return importlib.import_module('haenv.world.clinical.clinical_evidence')


@pytest.fixture
def inputs():
    return R.load_findings(), R.load_condition_findings()


def result(inputs, sid, fid, T=168, cid='OBS-004', profile=None, user=None):
    findings, profiles = inputs
    return builder().clinical_result_series(cid, sid, fid, T, findings,
                                           profiles[sid] if profile is None else profile,
                                           {} if user is None else user)


@pytest.mark.parametrize('cid', [f'OBS-{i:03}' for i in range(12)])
def test_aki_has_actual_48h_and_7d_creatinine_change(inputs, cid):
    cr = result(inputs, 'HD-UNI-09', 'Cr', T=56, cid=cid)
    composite = result(inputs, 'HD-UNI-09', 'AKICreatinineChange', T=56, cid=cid)[0]
    history = composite['details']['creatinine_history']
    assert cr == [history[-1]]
    assert history[-1]['ts'] - history[-2]['ts'] == 2
    assert history[-1]['value'] - history[-2]['value'] >= 26.5
    assert history[-1]['ts'] - history[0]['ts'] == 7
    assert history[-1]['value'] / history[0]['value'] >= 1.5
    assert 'μmol/L' in composite['value']
    assert '阳性' not in composite['value']
    assert composite['details']['cause_assessment'] == 'not_established'


@pytest.mark.parametrize('cid', [f'OBS-{i:03}' for i in range(12)])
def test_ckd_has_dated_repeated_albuminuria_not_single_late_value(inputs, cid):
    acr = result(inputs, 'HD-UNI-10', 'UACR', cid=cid)
    cr = result(inputs, 'HD-UNI-10', 'Cr', cid=cid)
    comp = result(inputs, 'HD-UNI-10', 'CKDPersistence', cid=cid)[0]
    assert [comp['details']['uacr_history'][-1]] == acr
    assert [comp['details']['creatinine_history'][-1]] == cr
    acr = comp['details']['uacr_history']
    assert len(acr) == 3 and acr[-1]['ts'] - acr[0]['ts'] == 112
    assert all(p['value'] >= 30 and p['unit'] == 'mg/g' for p in acr)
    assert acr[-1]['value'] >= 2 * acr[0]['value']
    assert comp['details']['progression_assessment'] == 'requires_clinical_evaluation'
    assert comp['details']['egfr'] is None
    assert '阳性' not in comp['value']


@pytest.mark.parametrize('sid,fid,T', [('HD-UNI-09','Cr',13), ('HD-UNI-10','Cr',118),
                                     ('HD-UNI-10','CKDPersistence',56)])
def test_too_short_window_is_unavailable_not_fake_normal(inputs, sid, fid, T):
    with pytest.raises(builder().ClinicalEvidenceUnavailable, match='window'):
        result(inputs, sid, fid, T=T)


def test_ckd_calendar_dates_do_not_equate_90_days_with_three_months(inputs):
    comp = result(inputs, 'HD-UNI-10', 'CKDPersistence', user={'index_date':'2026-08-31'})[0]
    acr = comp['details']['uacr_history']
    start = date.fromisoformat(acr[0]['date']); end = date.fromisoformat(acr[-1]['date'])
    assert end == date(2026,8,31) and (end-start).days == 112
    assert start.month + 3 <= end.month
    for p in acr:
        assert date.fromisoformat(p['date']) == end - timedelta(days=168-p['ts'])


@pytest.mark.parametrize('sid,fid', [('HD-UNI-08','ECGAtrialPattern'),
                                     ('HD-UNI-12','PancreaticImaging')])
def test_reports_describe_observations_not_gold_or_boolean_only(inputs, sid, fid):
    point = result(inputs, sid, fid)[0]
    assert isinstance(point['details'],dict) and len(point['details']) >= 3
    assert '阳性' not in point['value'] and '阴性' not in point['value']
    sp = R.load_unified_conditions()[sid]
    blob = json.dumps({'name':point['name'],'value':point['value'],'details':point['details']},ensure_ascii=False).lower()
    assert sp['diagnosis'].lower() not in blob
    assert not any(alias.lower() in blob for alias in sp['aliases'])
    assert sid not in json.dumps(point)
    assert point['details']['clinical_review'] == 'pending'


def test_ecg_intervals_and_p_wave_observations_are_explicit(inputs):
    point = result(inputs, 'HD-UNI-08','ECGAtrialPattern')[0]
    d=point['details']
    assert len(set(d['rr_intervals_ms'])) > 4
    assert d['consistent_p_before_qrs'] is False
    assert d['acquisition'] == '12_lead_ecg'
    assert d['sampling_note'] and 'PPG' not in point['value']


def test_explicit_negative_ecg_returns_regular_record_not_undeclared_normal(inputs):
    f,profiles=inputs;p=deepcopy(profiles['HD-UNI-08'])
    next(x for x in p['findings'] if x['id']=='ECGAtrialPattern')['direction']='negative'
    q=result(inputs,'HD-UNI-08','ECGAtrialPattern',profile=p)[0]
    assert q['details']['consistent_p_before_qrs'] is True
    assert len(set(q['details']['rr_intervals_ms'])) == 1
    assert '未见' in q['value'] or '规律' in q['value']
    assert '阴性' not in q['value']


def test_pancreatic_imaging_has_specific_descriptors_and_shared_lipase(inputs):
    imaging=result(inputs,'HD-UNI-12','PancreaticImaging')[0]
    lip=result(inputs,'HD-UNI-12','Lipase')[0]
    assert imaging['details']['lipase_result'] == lip
    assert imaging['details']['pancreatic_enlargement'] is True
    assert imaging['details']['peripancreatic_stranding'] is True
    assert lip['value'] >= 3 * lip['ref_high']
    assert lip['details']['multiple_of_local_uln'] == lip['value']/lip['ref_high']


@pytest.mark.parametrize('low,high', [(13.,60.),(8.,40.),(0.,180.)])
def test_lipase_criterion_uses_actual_assay_uln_not_invented_global_limit(inputs,low,high):
    findings,profiles=inputs; fx=deepcopy(findings);fx['Lipase']['ref']={'low':low,'high':high}
    lip=builder().clinical_result_series('ASSAY-02','HD-UNI-12','Lipase',168,fx,profiles['HD-UNI-12'],{})[0]
    assert lip['ref_high']==high and lip['ref_low']==low
    assert lip['value'] >= 3*high
    assert lip['unit']==fx['Lipase']['unit']


@pytest.mark.parametrize('sid,fid', [('HD-UNI-09','AKICreatinineChange'),
                                     ('HD-UNI-10','CKDPersistence'),
                                     ('HD-UNI-12','PancreaticImaging')])
def test_explicit_negative_composite_has_observations_but_no_normality_claim(inputs,sid,fid):
    _,profiles=inputs;p=deepcopy(profiles[sid])
    for r in p['findings']:
        if r['id']==fid:r['direction']='negative'
        if r['id'] in {'Cr','UACR','Lipase'}:r['direction']='normal';r['magnitude']=None
    obs=result(inputs,sid,fid,profile=p)[0]
    assert obs['details']['clinical_review']=='pending'
    assert '阴性' not in obs['value']
    if sid=='HD-UNI-09':
        hist=obs['details']['creatinine_history'];assert len({r['value'] for r in hist})==1
    elif sid=='HD-UNI-10':
        assert all(r['value']<30 for r in obs['details']['uacr_history'])
        assert obs['details']['egfr'] is None
    else:
        assert obs['details']['pancreatic_enlargement'] is False
        assert obs['details']['lipase_result']['value'] < 3*obs['details']['lipase_result']['ref_high']


@pytest.mark.parametrize('sid,fid', [('HD-UNI-08','ECGAtrialPattern'),('HD-UNI-09','Cr'),
                                     ('HD-UNI-10','CKDPersistence'),('HD-UNI-12','Lipase')])
def test_missing_or_insufficient_declaration_never_means_normal(inputs,sid,fid):
    for profile in [{},{'findings':[]},{'insufficient':True,'findings':inputs[1][sid]['findings']}]:
        with pytest.raises(builder().ClinicalEvidenceUnavailable):result(inputs,sid,fid,profile=profile)


@pytest.mark.parametrize('sid,fid', [('JD-CKM','Cr'),('HD-UNI-01','Cr'),('HD-UNI-09','NOT_REGISTERED'),
                                     ('JD-BENIGN2','FibrosisAssessment'),('HD-UNI-18','DepressionAssessment'),
                                     ('HD-UNI-19','CK'),('HD-UNI-20','GastricRetention'),('HD-UNI-99','Cr')])
def test_truly_unhandled_cases_return_none_without_reading_or_normalizing_inputs(sid,fid):
    assert builder().clinical_result_series('UNKNOWN',sid,fid,None,{},None,None) is None


def test_inputs_unchanged_order_independent_and_same_values_cross_return(inputs):
    fx,pf=inputs;original=deepcopy((fx,pf))
    expected=result(inputs,'HD-UNI-09','Cr')
    result(inputs,'HD-UNI-12','PancreaticImaging',cid='OTHER')
    assert result(inputs,'HD-UNI-09','Cr')==expected
    assert result(inputs,'HD-UNI-09','AKICreatinineChange')[0]['details']['creatinine_history'][-1:]==expected
    expected[0]['value']=-1
    assert result(inputs,'HD-UNI-09','Cr')[0]['value']>0
    assert (fx,pf)==original


def test_measurement_quantile_is_shared_across_declared_negative_and_positive(inputs):
    _,pf=inputs;p=deepcopy(pf['HD-UNI-09'])
    for r in p['findings']:
        if r['id']=='Cr':r['direction']='normal';r['magnitude']=None
        if r['id']=='AKICreatinineChange':r['direction']='negative'
    a=result(inputs,'HD-UNI-09','AKICreatinineChange')[0]['details']['creatinine_history']
    b=result(inputs,'HD-UNI-09','AKICreatinineChange',profile=p)[0]['details']['creatinine_history']
    assert a[0]['value']==b[0]['value']


def test_existing_value_generation_is_untouched(inputs):
    fx,_=inputs
    before=[_value(fx['Cr'],'high','moderate',f'OLD-{i}','Cr',0) for i in range(10)]
    for i in range(10):result(inputs,'HD-UNI-09','Cr',cid=f'OLD-{i}')
    assert before==[_value(fx['Cr'],'high','moderate',f'OLD-{i}','Cr',0) for i in range(10)]


@pytest.mark.parametrize('sid,fid,bad', [('HD-UNI-09','Cr','unit'),('HD-UNI-10','UACR','range'),
                                         ('HD-UNI-12','Lipase','zero_uln')])
def test_bad_units_ranges_are_explicit_errors_not_default_values(inputs,sid,fid,bad):
    fx,pf=inputs;fx=deepcopy(fx)
    if bad=='unit':fx['Cr']['unit']='mg/dL'
    elif bad=='range':fx['UACR']['ref']={'low':30,'high':20}
    else:fx['Lipase']['ref']={'low':0,'high':0}
    with pytest.raises(ValueError):builder().clinical_result_series('BAD',sid,fid,168,fx,pf[sid],{})


def test_all_measurement_dates_are_bounded_and_json_serializable(inputs):
    for sid,fid,T in [('HD-UNI-08','ECGAtrialPattern',56),('HD-UNI-09','AKICreatinineChange',56),
                      ('HD-UNI-10','CKDPersistence',168),('HD-UNI-12','PancreaticImaging',56)]:
        pts=result(inputs,sid,fid,T=T)
        json.dumps(pts,allow_nan=False)
        def check(obj):
            if isinstance(obj,dict):
                if 'ts' in obj:assert 0<=obj['ts']<=T
                for v in obj.values():check(v)
            elif isinstance(obj,list):
                for v in obj:check(v)
        check(pts)


@pytest.mark.parametrize('sex', ['F','M',None])
@pytest.mark.parametrize('T', [168,336])
def test_renal_companion_and_standalone_use_identical_dated_sex_ref_measurements(inputs,sex,T):
    user={'sex':sex,'index_date':'2026-10-08'}
    for sid,fid,key in [('HD-UNI-09','AKICreatinineChange','creatinine_history'),
                        ('HD-UNI-10','CKDPersistence','creatinine_history')]:
        companion=result(inputs,sid,fid,T=T,user=user)[0]
        direct=result(inputs,sid,'Cr',T=T,user=user)
        assert companion['details'][key][-1:]==direct
        assert all(p['date'] and p['ref_high']>p['ref_low'] for p in direct)


def test_screening_adapter_exposes_only_current_value_matching_the_history_tool(inputs):
    from haenv.world.observations.findings_render import _entry
    fx,pf=inputs
    points=result(inputs,'HD-UNI-09','Cr',T=56)
    row=next(r for r in pf['HD-UNI-09']['findings'] if r['id']=='Cr')
    # Caller adapters should change format only; never redraw measurements.
    screening=[_entry('OBS-004',fx['Cr'],row,p['ts'],p['value'],i) for i,p in enumerate(points)]
    tool=result(inputs,'HD-UNI-09','AKICreatinineChange',T=56)[0]['details']['creatinine_history']
    assert len(screening)==1 and len(tool)==3
    assert [(r['source_timestamp'],r['_val']) for r in screening]==[(tool[-1]['ts'],tool[-1]['value'])]


@pytest.mark.parametrize('T',[None,True,1.5,-1])
def test_invalid_handled_time_does_not_trigger_normal_fallback(inputs,T):
    with pytest.raises(ValueError):result(inputs,'HD-UNI-09','Cr',T=T)


def test_contradictory_renal_declaration_is_rejected(inputs):
    _,profiles=inputs;p=deepcopy(profiles['HD-UNI-10'])
    next(r for r in p['findings'] if r['id']=='UACR')['direction']='normal'
    with pytest.raises(builder().ClinicalEvidenceUnavailable,match='inconsistent'):
        result(inputs,'HD-UNI-10','CKDPersistence',profile=p)


def test_explicit_negative_early_scan_does_not_normalize_high_lipase(inputs):
    _,profiles=inputs;p=deepcopy(profiles['HD-UNI-12'])
    next(r for r in p['findings'] if r['id']=='PancreaticImaging')['direction']='negative'
    imaging=result(inputs,'HD-UNI-12','PancreaticImaging',profile=p)[0]
    assert imaging['details']['pancreatic_enlargement'] is False
    assert imaging['details']['lipase_result']['value'] >= 3*imaging['details']['lipase_result']['ref_high']
    assert '不能排除' in imaging['value']


def test_output_contains_no_source_url_or_gold_id_that_reads_the_answer(inputs):
    # Finding IDs remain the integrated measurement lookup IDs. Descriptive content
    # and details never include spec IDs, disease aliases or disease-specific URLs.
    for sid,fid in [('HD-UNI-09','AKICreatinineChange'),('HD-UNI-10','CKDPersistence'),
                    ('HD-UNI-12','PancreaticImaging'),('HD-UNI-08','ECGAtrialPattern')]:
        p=result(inputs,sid,fid)[0];p=deepcopy(p)
        def remove_ids(obj):
            if isinstance(obj,dict):
                obj.pop('finding_id',None)
                for v in obj.values():remove_ids(v)
            elif isinstance(obj,list):
                for v in obj:remove_ids(v)
        remove_ids(p);blob=json.dumps(p,ensure_ascii=False).lower()
        assert 'http' not in blob and sid.lower() not in blob
        assert all(a.lower() not in blob for a in R.load_unified_conditions()[sid]['aliases'])


@pytest.mark.parametrize('sid,fid,marker,key', [
    ('HD-UNI-09','Cr','AKICreatinineChange','creatinine_history'),
    ('HD-UNI-10','Cr','CKDPersistence','creatinine_history'),
    ('HD-UNI-10','UACR','CKDPersistence','uacr_history')])
def test_current_only_query_keeps_paid_history_in_the_marker(inputs,sid,fid,marker,key):
    current=result(inputs,sid,fid)
    historical=result(inputs,sid,marker)[0]['details'][key]
    assert len(current)==1,'simple numeric query must not provide history for free'
    assert current[0]['ts']==168 and current[0]==historical[-1]
    assert all(p['ts']<=168 for p in historical)
    assert len(historical)==3
    assert not any('history' in k for k in current[0]['details'])


@pytest.mark.parametrize('sid,fid,numeric', [('HD-UNI-09','Cr','Cr'),
    ('HD-UNI-10','UACR','UACR'),('HD-UNI-12','Lipase','Lipase')])
@pytest.mark.parametrize('magnitude',[None,'invented'])
def test_no_default_magnitude_for_missing_or_invalid_critical_numeric_design(inputs,sid,fid,numeric,magnitude):
    _,pf=inputs;p=deepcopy(pf[sid])
    next(r for r in p['findings'] if r['id']==numeric)['magnitude']=magnitude
    with pytest.raises(builder().ClinicalEvidenceUnavailable,match='magnitude'):
        result(inputs,sid,fid,profile=p)


@pytest.mark.parametrize('rows',[None,{},['not a declaration']])
def test_malformed_consumer_declaration_collection_is_explicitly_unavailable(inputs,rows):
    with pytest.raises(builder().ClinicalEvidenceUnavailable,match='declarations'):
        result(inputs,'HD-UNI-08','ECGAtrialPattern',profile={'findings':rows})


@pytest.mark.parametrize('sid,fid', [('HD-UNI-09','Cr'),('HD-UNI-10','UACR'),
    ('HD-UNI-12','Lipase'),('HD-UNI-08','ECGAtrialPattern')])
def test_supplied_consumer_profile_is_authoritative_not_reloaded_from_registry(inputs,sid,fid):
    _,pf=inputs;p=deepcopy(pf[sid])
    p['findings']=[r for r in p['findings'] if r['id']!=fid]
    with pytest.raises(builder().ClinicalEvidenceUnavailable):result(inputs,sid,fid,profile=p)


def test_missing_numeric_vocabulary_is_handled_unavailable_not_empty_or_normal(inputs):
    fx,pf=inputs;fx=deepcopy(fx);del fx['Cr']
    with pytest.raises(builder().ClinicalEvidenceUnavailable):
        builder().clinical_result_series('MISSING','HD-UNI-09','Cr',168,fx,pf['HD-UNI-09'],{})
