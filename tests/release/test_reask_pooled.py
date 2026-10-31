"""Released answers and pooled uncertainty are reproducible without model access."""
import json
from pathlib import Path
import shutil

import pytest

from _paths import ROOT
from tools.evaluation import reask_pooled as pooled


def test_released_reask_recomputation_and_diagnosis_scores(capsys):
    with pytest.raises(SystemExit) as help_exit:
        pooled.main(['--help'])
    assert help_exit.value.code == 0
    capsys.readouterr()
    result = pooled.recompute()
    expected = json.loads((pooled.DATA / 'facts.json').read_text())
    assert result['facts'] == expected
    assert len(expected) == 61
    assert result['main']['n_targets'] == {'gpt-6-luna': 44, 'glm-5.3-flash': 50, 'qwen3-32b': 48}
    # Released facts come from the stored D; the 1.4 judge differs only where the generated
    # table registers it, and the table is what --write-drift would write today.
    table = json.loads((pooled.DATA / pooled.DRIFT).read_text())['rows']
    assert table == pooled.matcher_drift()
    assert {r['concept'] for r in table} <= {'aki_volume_depletion', 'ckd_progression', 'judging_synonyms',
                                         'judging_aliases_en', 'english_variants'}
    assert all(r.get('name') for r in table if r['change'] == '1.4 judging names')
    assert all((r['stored_D'], r['current_D']) == (0, 1) for r in table)
    assert pooled.main(['--check']) == 0
    summary = capsys.readouterr().out.splitlines()[0]
    assert summary == (f'61 facts matched; 535 answers re-scored; {len(table)} registered 1.4 '
                       'matcher differences; 0 other D mismatches')
    guide = (ROOT / 'docs/REPRODUCE.md').read_text()
    for claim in ('### Recomputing the released re-ask results',
                  'python tools/evaluation/reask_pooled.py --check',
                  f'`{summary}`',
                  f'accept {len(table)} stored answers that 1.3.1',
                  'benchmarks/results/1.3.1/reask/README.md'):
        assert claim in guide
    readme = (pooled.DATA / 'README.md').read_text()
    assert f'Under it, {len(table)} stored answers that 1.3.1 scored wrong are correct.' in readme


def _write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def test_changed_stored_d_is_rejected(tmp_path):
    for name in ('inclusion.jsonl', 'answers.jsonl', 'gold.json', pooled.DRIFT):
        shutil.copyfile(pooled.DATA / name, tmp_path / name)
    table = json.loads((tmp_path / pooled.DRIFT).read_text())
    registered = [pooled.identity(r) for r in table['rows']]
    rows = pooled.read_jsonl(tmp_path / 'inclusion.jsonl')
    assert pooled.check_answers(tmp_path) == (535, len(registered))
    # A changed stored D outside the registered differences is rejected.
    plain = next(r for r in rows if pooled.identity(r) not in registered)
    plain['D'] = 1 - plain['D']
    _write_jsonl(tmp_path / 'inclusion.jsonl', rows)
    with pytest.raises(ValueError, match='stored D differs'):
        pooled.check_answers(tmp_path)
    plain['D'] = 1 - plain['D']
    # A registered difference that the judge no longer reproduces is rejected.
    drifted = next(r for r in rows if pooled.identity(r) == registered[0])
    drifted['D'] = 1 - drifted['D']
    _write_jsonl(tmp_path / 'inclusion.jsonl', rows)
    with pytest.raises(ValueError, match='no longer reproduces'):
        pooled.check_answers(tmp_path)
    drifted['D'] = 1 - drifted['D']
    _write_jsonl(tmp_path / 'inclusion.jsonl', rows)
    # A difference dropped from the table is rejected.
    (tmp_path / pooled.DRIFT).write_text(json.dumps({**table, 'rows': table['rows'][1:]}))
    with pytest.raises(ValueError, match='stored D differs'):
        pooled.check_answers(tmp_path)


def test_missing_half_answer_is_rejected():
    rows = pooled.read_jsonl(pooled.DATA / 'inclusion.jsonl')
    first_half = next(r for r in rows if r['arm'] == 'H')
    with pytest.raises(ValueError, match='incomplete unit'):
        pooled.split_rows([r for r in rows if r is not first_half])


def test_http_replay_offline_checks_stored_results(tmp_path):
    guide = (ROOT / 'docs/REPRODUCE.md').read_text()
    assert '### Checking HTTP replay' in guide
    assert ('python tools/release/replay_http.py --work leaderboard-pool/ --workers 32 '
            '--out artifacts/replay-http-results.json') in guide
    assert 'python tools/release/replay_http.py --offline' in guide
    released = json.loads((ROOT / 'docs/boards/1.3.1/findings.json').read_text())['replay']
    for summary, key in [('questions', 'questions'), ('responses', 'sessions'), ('calls', 'calls'),
                         ('identical', 'identical'), ('differing', 'differing'),
                         ('stored_differing', 'stored_differing'),
                         ('pending_purchases_replayed', 'pending_purchases'), ('validation_failures', 'invalid')]:
        assert f'`{summary}={str(released[key]).replace(",", "")}`' in guide
    assert '`missing=[]`' in guide and '`worker_errors=[]`' in guide
    pytest.importorskip('fhir.resources.R4B')
    from tools.release.replay_http import main
    output = tmp_path / 'replay.json'
    assert main(['--offline', '--workers', '1', '--out', str(output)]) == 0
    result = json.loads(output.read_text())
    assert (result['questions'], result['responses'], result['calls'], result['identical']) == (1, 1, 31, 31)
    assert result['stored_differing'] == result['differing'] == result['validation_failures'] == 0
    assert result['missing'] == result['worker_errors'] == []
