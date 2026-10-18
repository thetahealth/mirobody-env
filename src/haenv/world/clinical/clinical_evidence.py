"""Bounded synthetic clinical observations from explicit template declarations.

One pure result API for screening and requested tests. Callers must use this API
for BOTH numeric findings and their companion reports before generic rendering or
stream shortcuts. None means an unhandled (spec, finding) pair. A handled pair
with missing evidence declarations or inadequate time raises explicitly; callers
must not turn that into a generic normal result.

Source thresholds: KDIGO 2012 2.1.1; KDIGO 2024 Tables 1/3 and PP 2.1.5;
ESC 2024 3.1; revised Atlanta 2012, Gut 2013 p.103. Magnitudes, intermediate
values and observation timing are synthesis designs, not validated clinical
reference ranges or patient measurements. Physician review remains pending.
"""
from __future__ import annotations

from datetime import date, timedelta
import math

import haenv.world.generation.rng as rng
from haenv.world.observations.findings_render import _round
from haenv.world.observations.findings_render import _value
from haenv.world.observations.findings_render import with_sex_ref

# Exact new IDs only; never called implicitly on old disease families or findings.
_SUPPORTED = {
    'HD-UNI-08': frozenset({'ECGAtrialPattern'}),
    'HD-UNI-09': frozenset({'Cr', 'AKICreatinineChange'}),
    'HD-UNI-10': frozenset({'Cr', 'UACR', 'CKDPersistence'}),
    'HD-UNI-12': frozenset({'Lipase', 'PancreaticImaging'}),
    'HD-UNI-13': frozenset({'FibrosisAssessment'}),
}
_CHRONIC_SPAN_DAYS = 112  # Template design; exceeds any three calendar months.


class ClinicalEvidenceUnavailable(ValueError):
    """Recognized result cannot be produced honestly from the declared inputs."""


def _decl(profile: dict, fid: str) -> dict:
    declarations = profile.get('findings')
    if not isinstance(declarations, (list, tuple)) or any(
            not isinstance(r, dict) for r in declarations):
        raise ClinicalEvidenceUnavailable('consumer finding declarations must be a sequence of mappings')
    rows = [r for r in declarations if r.get('id') == fid]
    if len(rows) != 1:
        raise ClinicalEvidenceUnavailable(f'{fid}: require exactly one declared finding')
    row = rows[0]
    if row.get('direction') not in {'high', 'normal', 'positive', 'negative'}:
        raise ClinicalEvidenceUnavailable(f'{fid}: unsupported declared direction')
    if row['direction'] == 'high' and row.get('magnitude') not in {'mild', 'moderate', 'marked'}:
        raise ClinicalEvidenceUnavailable(f'{fid}: explicit supported magnitude design required')
    return row


def _spec(findings: dict, fid: str) -> dict:
    spec = findings.get(fid)
    if not isinstance(spec, dict):
        raise ClinicalEvidenceUnavailable(f'{fid}: finding vocabulary is unavailable')
    return spec


def _numeric_spec(findings: dict, fid: str, user_profile: dict) -> dict:
    original = _spec(findings, fid)
    ref = original.get('ref')
    if not isinstance(ref, dict):
        raise ValueError(f'{fid}: explicit assay reference band required')
    try:
        lo, hi = float(ref['low']), float(ref['high'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{fid}: invalid assay reference band') from exc
    if not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo < hi:
        raise ValueError(f'{fid}: invalid assay reference band')
    units = {'Cr': {'μmol/L', 'µmol/L'}, 'UACR': {'mg/g'}, 'Lipase': {'U/L'}}
    if original.get('unit') not in units[fid]:
        raise ValueError(f'{fid}: unsupported units; do not substitute or infer conversion')
    # Reuse established sex-reference handling without inferring an identity.
    return with_sex_ref({fid: original}, user_profile.get('sex'))[fid]


def _draw(spec: dict, row: dict, case_id: str, fid: str, k: int,
          direction: str | None = None, magnitude: str | None = None) -> float:
    direction = row['direction'] if direction is None else direction
    if direction not in {'high', 'normal'}:
        raise ClinicalEvidenceUnavailable(f'{fid}: numeric direction must be high or normal')
    value = _value(spec, direction, row.get('magnitude') if magnitude is None else magnitude,
                   case_id, fid, k)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{fid}: generator produced a nonpositive/nonfinite measurement')
    return value


def _date_at(ts: int, T: int, user_profile: dict) -> str | None:
    # Optional actual timeline anchor. Do not invent a calendar date when absent.
    anchor = user_profile.get('index_date')
    if anchor is None:
        return None
    if isinstance(anchor, date):
        d = anchor
    elif isinstance(anchor, str):
        d = date.fromisoformat(anchor)
    else:
        raise ValueError('index_date must be an ISO date or datetime.date')
    return (d - timedelta(days=T-ts)).isoformat()


def _point(fid: str, name: str, ts: int, value, T: int, user: dict,
           *, spec: dict | None = None, details: dict | None = None,
           flag: str | None = None) -> dict:
    point = {'ts': ts, 'value': value, 'finding_id': fid, 'name': name,
             'details': {'synthetic': True, 'clinical_review': 'pending', **(details or {})}}
    stamp = _date_at(ts, T, user)
    if stamp is not None:
        point['date'] = stamp
    if spec is not None:
        lo, hi = float(spec['ref']['low']), float(spec['ref']['high'])
        point.update(unit=spec['unit'], ref_low=lo, ref_high=hi,
                     flag='H' if value > hi else ('L' if value < lo else 'NORMAL'))
        point['details']['reference_basis'] = 'supplied_local_assay_band'
    elif flag is not None:
        point['flag'] = flag
    return point


def _renal_bundle(case_id: str, spec_id: str, T: int, findings: dict,
                  profile: dict, user: dict) -> tuple[list[dict], list[dict], dict]:
    cr_row = _decl(profile, 'Cr')
    cr_spec = _numeric_spec(findings, 'Cr', user)
    chronic = spec_id == 'HD-UNI-10'
    companion = 'CKDPersistence' if chronic else 'AKICreatinineChange'
    q = _decl(profile, companion)
    if q['direction'] not in {'positive', 'negative'}:
        raise ClinicalEvidenceUnavailable(f'{companion}: explicit positive/negative design required')
    positive = q['direction'] == 'positive'
    if cr_row['direction'] != ('high' if positive else 'normal'):
        raise ClinicalEvidenceUnavailable('inconsistent numeric and companion declarations')
    if T < (_CHRONIC_SPAN_DAYS + 7 if chronic else 14):
        raise ClinicalEvidenceUnavailable('observation window is too short for the requested history')
    if not chronic:
        base = _draw(cr_spec, cr_row, case_id, 'Cr', 0, direction='normal')
        latest = base
        if positive:
            proposal = _draw(cr_spec, cr_row, case_id, 'Cr', 2)
            # Constrained synthesis: intentionally construct both supported criteria.
            # The safety margin survives printed precision. It is not a lab ref limit.
            latest = _round(max(proposal, 1.6*base,
                                base + 30 + 6*rng.unit(case_id, 'clinical-result', 'delta')), 'Cr')
        cr = [_point('Cr', '血清肌酐', day, val, T, user, spec=cr_spec)
              for day, val in [(T-7, base), (T-2, base), (T, latest)]]
        delta = latest - base
        report = _point(companion, '短期血清肌酐记录', T,
                        f"第{T-7}天及第{T-2}天血清肌酐{base} {cr_spec['unit']},"
                        f"第{T}天{latest} {cr_spec['unit']};48小时差值{delta:.2f} {cr_spec['unit']},"
                        f"与7天前比值{latest/base:.3f}", T, user,
                        details={'creatinine_history': cr, 'delta_48h_umol_l': delta,
                                 'ratio_to_7d_baseline': latest/base,
                                 'elapsed_hours_for_delta': 48,
                                 'creatinine_change_observed': delta >= 26.5 or latest/base >= 1.5,
                                 'cause_assessment': 'not_established',
                                 'urine_output': None},
                        flag='CHANGE_OBSERVED' if positive else 'NO_CHANGE_IN_RECORD')
        return cr, [], report
    acr_row = _decl(profile, 'UACR')
    acr_spec = _numeric_spec(findings, 'UACR', user)
    if acr_row['direction'] != ('high' if positive else 'normal'):
        raise ClinicalEvidenceUnavailable('inconsistent albuminuria and persistence declarations')
    if positive:
        first_cr = _draw(cr_spec, cr_row, case_id, 'Cr', 0, magnitude='mild')
        last_cr = _draw(cr_spec, cr_row, case_id, 'Cr', 2)
        if last_cr < first_cr:
            raise ClinicalEvidenceUnavailable('declared renal trend is inconsistent with generated values')
        first_acr = _round(max(31.0, _draw(acr_spec, acr_row, case_id, 'UACR', 0,
                                        magnitude='mild')), 'UACR')
        proposal = _draw(acr_spec, acr_row, case_id, 'UACR', 2)
        last_acr = _round(max(proposal, first_acr*(2.05 + .35*rng.unit(
            case_id, 'clinical-result', 'albuminuria-span'))), 'UACR')
    else:
        first_cr = last_cr = _draw(cr_spec, cr_row, case_id, 'Cr', 0)
        # Construct an explicitly declared below-criterion scenario, not an
        # inference that absent declarations imply no disease. 30 mg/g is sourced.
        first_acr = last_acr = _round(min(29.0, _draw(acr_spec, acr_row, case_id, 'UACR', 0)), 'UACR')
    days = [T-_CHRONIC_SPAN_DAYS, T-_CHRONIC_SPAN_DAYS//2, T]
    cr_values = [first_cr, _round((first_cr+last_cr)/2, 'Cr'), last_cr]
    acr_values = [first_acr, _round((first_acr+last_acr)/2, 'UACR'), last_acr]
    cr = [_point('Cr', '血清肌酐', d, v, T, user, spec=cr_spec) for d,v in zip(days,cr_values)]
    acr = [_point('UACR', '尿白蛋白/肌酐比值', d, v, T, user, spec=acr_spec) for d,v in zip(days,acr_values)]
    text = ';'.join(f"第{d}天血清肌酐{cv} {cr_spec['unit']},尿白蛋白/肌酐比值{av} mg/g"
                    for d,cv,av in zip(days,cr_values,acr_values))
    report = _point(companion, '纵向肾功能及尿白蛋白记录', T,
                    text+';记录跨度112天,未提供eGFR', T, user,
                    details={'creatinine_history': cr, 'uacr_history': acr,
                             'history_span_days': _CHRONIC_SPAN_DAYS,
                             'albuminuria_persistent_in_record': all(v >= 30 for v in acr_values),
                             'uacr_ratio': last_acr/first_acr,
                             'repeat_evaluation_trigger': last_acr >= 2*first_acr,
                             'progression_assessment': 'requires_clinical_evaluation',
                             'egfr': None, 'etiology': 'not_established',
                             'calendar_basis': '112_day_span_exceeds_any_three_calendar_months'},
                    flag='PERSISTENT_CHANGE_IN_RECORD' if positive else 'CRITERION_NOT_OBSERVED')
    return cr, acr, report


def _lipase(case_id: str, T: int, findings: dict, profile: dict, user: dict) -> dict:
    row = _decl(profile, 'Lipase')
    spec = _numeric_spec(findings, 'Lipase', user)
    proposal = _draw(spec, row, case_id, 'Lipase', 0)
    uln = float(spec['ref']['high'])
    if row['direction'] == 'high':
        # Same assay's actual ULN, not an invented universal activity limit.
        val = _round(max(proposal, (3.2+.6*rng.unit(case_id, 'clinical-result', 'lipase'))*uln), 'Lipase')
        if val < 3*uln:
            raise ValueError('assay precision cannot represent the selected relative threshold')
    else:
        val = proposal
    return _point('Lipase', '血清脂肪酶', T, val, T, user, spec=spec,
                  details={'local_uln': uln, 'multiple_of_local_uln': val/uln,
                           'sampling_note': 'activity and upper limit use the same local assay'})


def _ecg(case_id: str, T: int, findings: dict, profile: dict, user: dict) -> dict:
    _spec(findings, 'ECGAtrialPattern')
    row = _decl(profile, 'ECGAtrialPattern')
    if row['direction'] not in {'positive', 'negative'}:
        raise ClinicalEvidenceUnavailable('ECG requires an explicit record design')
    positive = row['direction'] == 'positive'
    rr = [510 + rng.below(430, case_id, 'clinical-result', 'rr', k) + k
          for k in range(8)] if positive else [760 + rng.below(50, case_id, 'clinical-result', 'rr')] * 8
    value = ('本段RR间期不规则,各QRS前未见持续形态一致的P波' if positive else
             '本段RR间期规律,各QRS前可见形态一致的P波;短时记录不排除间歇发作')
    return _point('ECGAtrialPattern', '十二导联心电记录', T, value, T, user,
                  details={'acquisition': '12_lead_ecg', 'recording_duration_seconds': 10,
                           'rr_intervals_ms': rr, 'consistent_p_before_qrs': not positive,
                           'sampling_note': 'RR intervals are an excerpt of this synthetic recording; requires clinical review'},
                  flag='IRREGULAR_RECORD' if positive else 'REGULAR_RECORD')


def _pancreatic(case_id: str, T: int, findings: dict, profile: dict, user: dict) -> dict:
    _spec(findings, 'PancreaticImaging')
    row = _decl(profile, 'PancreaticImaging')
    if row['direction'] not in {'positive', 'negative'}:
        raise ClinicalEvidenceUnavailable('imaging requires an explicit record design')
    positive = row['direction'] == 'positive'
    # These are visible report descriptors for a selected CT design, not a diagnosis.
    value = ('增强CT:胰腺弥漫性增大,实质强化较均匀,胰周脂肪间隙条索,少量胰周液体' if positive else
             '本次CT未见胰腺增大及胰周脂肪间隙条索或液体;单次早期检查不能排除随后变化')
    return _point('PancreaticImaging', '胰腺及胰周影像描述', T, value, T, user,
                  details={'modality': 'contrast_enhanced_ct', 'pancreatic_enlargement': positive,
                           'homogeneous_enhancement': positive, 'peripancreatic_stranding': positive,
                           'peripancreatic_fluid_observed': positive,
                           'lipase_result': _lipase(case_id, T, findings, profile, user),
                           'etiology': 'not_established'},
                  flag='DESCRIPTIVE_CHANGE' if positive else 'CHANGES_NOT_OBSERVED')


def validate_fibrosis_background(profile: dict, conditions) -> None:
    """Only declared compatible backgrounds can use this synthetic exclusion bundle.

    The negative results themselves are independently declared simulated facts,
    never inferred from this list. An unknown or conflicting background requires
    a different explicit design, not erasing the conflicting condition.
    """
    allowed = (profile.get('secondary_assessment') or {}).get('compatible_background_conditions')
    if (not isinstance(allowed, list) or not allowed or not isinstance(conditions, list)
            or not conditions or any(not isinstance(c, str) or c not in allowed for c in conditions)
            or not set(conditions) & {'obesity', 'hypertension', 'dyslipidemia', 'T2D', 'MASLD'}):
        raise ClinicalEvidenceUnavailable('FibrosisAssessment: background missing or incompatible with declared simulation')


def _fibrosis(case_id: str, T: int, findings: dict, profile: dict, user: dict) -> dict:
    """Secondary NIT report, never a FIB-4-derived diagnosis or histological stage.

    AASLD 2023 Table 5: VCTE >=12 / <8 kPa, MRE >=3.63 / <2.55 kPa,
    MRI-PDFF >=5%. EASL 2024 Table 6: VCTE F3 rule-in 12 kPa (MRE
    3.53 kPa); we use the AASLD MRE cut point, not a blended threshold.
    EASL-ALEH 2015 pp.5-6: fasting >=2h, >=10 shots, IQR/median <30%,
    appropriate probe and confounder review. Four-hour fasting is a design.
    Exact magnitudes are synthetic ranges, not guideline-derived distributions.
    """
    _spec(findings, 'FibrosisAssessment')
    declaration = _decl(profile, 'FibrosisAssessment')
    design = profile.get('secondary_assessment')
    if declaration['direction'] not in {'positive', 'negative'} or not isinstance(design, dict):
        raise ClinicalEvidenceUnavailable('FibrosisAssessment requires an explicit secondary assessment design')
    validate_fibrosis_background(profile, user.get('known_conditions'))

    def sample(key):
        band = design.get(key)
        if (not isinstance(band, list) or len(band) != 2
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in band)
                or not 0 < band[0] <= band[1]):
            raise ClinicalEvidenceUnavailable(f'FibrosisAssessment: missing/invalid {key}')
        return round(band[0] + (band[1]-band[0])*rng.unit(case_id, 'secondary-fibrosis', key), 2)

    lsm, mre, pdff, iqr = (sample(k) for k in (
        'vcte_lsm_kpa', 'mre_lsm_kpa', 'mri_pdff_percent', 'vcte_iqr_over_median'))
    try:
        valid, attempts, fasting = (design[k] for k in (
            'valid_measurements', 'attempted_measurements', 'fasting_hours'))
        quality = (isinstance(valid, int) and isinstance(attempts, int)
                   and 10 <= valid <= attempts and fasting >= 2 and iqr < .30
                   and design['probe'] == 'XL' and design['skin_to_capsule_mm'] > 25
                   and design['experienced_operator'] is True
                   and design['mre_quality'] == 'adequate_wave_propagation_no_major_artifact'
                   and all(design['confounders'][k] is False for k in (
                       'acute_hepatitis', 'extrahepatic_cholestasis', 'hepatic_congestion')))
        eti = design['etiology_review']
        etiology_reviewed = (eti['alcohol_g_per_day'] == 0
                            and eti['hepatitis_b_surface_antigen'] == 'negative'
                            and eti['hepatitis_c_antibody'] == 'negative'
                            and eti['competing_drug_or_secondary_cause_identified'] is False)
    except (KeyError, TypeError):
        raise ClinicalEvidenceUnavailable('FibrosisAssessment: missing quality or etiology review') from None
    if not quality or not etiology_reviewed:
        raise ClinicalEvidenceUnavailable('FibrosisAssessment: invalid/confounded secondary assessment')
    high, low = lsm >= 12 and mre >= 3.63, lsm < 8 and mre < 2.55
    positive = declaration['direction'] == 'positive'
    if (positive and (not high or pdff < 5)) or (not positive and not low):
        raise ClinicalEvidenceUnavailable('FibrosisAssessment: readings do not support declared secondary risk')
    risk = 'high' if high else 'low'
    interpretation = ('两项无创评估一致提示进展期纤维化可能性高,需肝病专科进一步管理' if high else
                      '两项无创评估均在进展期纤维化低风险范围,结合临床随访')
    value = (f'空腹{fasting}小时,VCTE采用{design["probe"]}探头(皮肤至肝包膜{design["skin_to_capsule_mm"]} mm),'
             f'有效测量{valid}/{attempts}次,肝硬度中位数{lsm} kPa,IQR/中位数{iqr:.2f};'
             f'为进一步澄清纤维化风险加做MRE:{mre} kPa,波传播充分且无明显伪影;'
             f'MRI-PDFF {pdff}%,{"提示脂肪变" if pdff >= 5 else "未达脂肪变界值"}。'
             '报告复核未见急性肝炎、胆道梗阻或肝淤血干扰;自述不饮酒,HBsAg及抗HCV阴性,'
             '问诊未识别竞争性药物或继发病因。肝病专科综合意见:' + interpretation +
             ';脂肪变的病因判断需结合已知代谢危险因素。FIB-4只能作初筛,不能单凭其确诊;'
             '上述为无创概率判断,未作肝活检,不能据此确定组织学F3/F4或MASH,也不证明短期内进展。')
    return _point('FibrosisAssessment', '肝脏弹性与脂肪定量及专科二级评估', T, value, T, user,
        details={'vcte': {'lsm_kpa': lsm, 'valid_measurements': valid, 'attempted_measurements': attempts,
                          'iqr_over_median': iqr, 'fasting_hours': fasting, 'probe': design['probe'],
                          'skin_to_capsule_mm': design['skin_to_capsule_mm'], 'quality_acceptable': True},
                 'mre': {'lsm_kpa': mre, 'quality': design['mre_quality']}, 'mri_pdff_percent': pdff,
                 'confounder_review': dict(design['confounders']), 'etiology_review': dict(eti),
                 'exclusion_evidence_origin': 'registry_declared_simulated_observations',
                 'background_compatibility_checked': True,
                 'specialist_assessment': {'risk': risk, 'basis': 'concordant_secondary_nits',
                                          'interpretation': interpretation},
                 'histology': None, 'histological_stage': None,
                 'sources': ['https://pmc.ncbi.nlm.nih.gov/articles/PMC10735173/',
                             'https://easlcampus.eu/sites/default/files/2024-08/EASL_CPGs_management_of_MASLD.pdf',
                             'https://www.alehlatam.org/documentos/GUIAS_EASL_ALEH_NON_INVASIVE_TEST.pdf']},
        flag='HIGH_RISK_NIT' if high else 'LOW_RISK_NIT')


def clinical_result_series(case_id: str, spec_id: str, finding_id: str, T: int,
                           findings: dict, profile: dict, user_profile: dict) -> list[dict] | None:
    """Return deterministic visible observations, or None for a genuinely unhandled pair.

    ``profile`` is the authoritative full consumer findings profile, not a single
    requested row. It is never reloaded or filled from the condition registry.
    Missing/invalid declarations raise; handled results never return an empty list.
    ``user_profile`` supplies optional sex and ISO ``index_date``; no demographics
    or dates are guessed. All returned timestamps are relative case days <= T.
    A handled insufficient profile is unavailable, never a default normal report.
    Simple numeric findings expose only the current point. History-check reports
    retain prior/current measurements in details; their current point is identical
    to the standalone numeric return. No prior history is supplied free through
    screening or a simple numeric query.
    """
    if finding_id not in _SUPPORTED.get(spec_id, ()):
        return None
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError('T must be a nonnegative integer case day')
    if not isinstance(case_id, str) or not case_id:
        raise ValueError('case_id must be a nonempty stable sampling key')
    if not isinstance(profile, dict) or not isinstance(user_profile, dict):
        raise ClinicalEvidenceUnavailable('profile and user_profile mappings are required')
    if profile.get('insufficient') or user_profile.get('insufficient'):
        raise ClinicalEvidenceUnavailable('insufficient clinical observations are unavailable')
    if finding_id == 'FibrosisAssessment':
        return [_fibrosis(case_id, T, findings, profile, user_profile)]
    if spec_id in {'HD-UNI-09', 'HD-UNI-10'}:
        cr, acr, report = _renal_bundle(case_id, spec_id, T, findings, profile, user_profile)
        return [cr[-1]] if finding_id == 'Cr' else ([acr[-1]] if finding_id == 'UACR' else [report])
    if spec_id == 'HD-UNI-08':
        return [_ecg(case_id, T, findings, profile, user_profile)]
    if finding_id == 'Lipase':
        return [_lipase(case_id, T, findings, profile, user_profile)]
    return [_pancreatic(case_id, T, findings, profile, user_profile)]
