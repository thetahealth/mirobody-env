"""Released answers and pooled uncertainty are reproducible without model access."""
import json
from pathlib import Path
import shutil

import pytest

from _paths import ROOT
from tools.evaluation import reask_pooled as pooled


def test_released_reask_recomputation_and_diagnosis_scores():
    with pytest.raises(SystemExit) as help_exit:
        pooled.main(['--help'])
    assert help_exit.value.code == 0
    result = pooled.recompute()
    expected = json.loads((pooled.DATA / 'facts.json').read_text())
    assert result['facts'] == expected
    assert len(expected) == 61
    assert result['main']['n_targets'] == {'gpt-6-luna': 44, 'glm-5.3-flash': 50, 'qwen3-32b': 48}
    assert pooled.check_answers() == 535
    guide = (ROOT / 'docs/REPRODUCE.md').read_text()
    for claim in ('### Recomputing the released re-ask results',
                  'python tools/evaluation/reask_pooled.py --check',
                  '61 facts matched; 535 answers re-scored; 0 D mismatches',
                  'benchmarks/results/1.3.1/reask/README.md'):
        assert claim in guide


def test_changed_stored_d_is_rejected(tmp_path):
    for name in ('inclusion.jsonl', 'answers.jsonl', 'gold.json'):
        shutil.copyfile(pooled.DATA / name, tmp_path / name)
    rows = pooled.read_jsonl(tmp_path / 'inclusion.jsonl')
    rows[0]['D'] = 1 - rows[0]['D']
    (tmp_path / 'inclusion.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
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
