"""Released answers and pooled uncertainty are reproducible without model access."""
import json
from pathlib import Path
import shutil

import pytest

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
    pytest.importorskip('fhir.resources.R4B')
    from tools.release.replay_http import main
    output = tmp_path / 'replay.json'
    assert main(['--offline', '--workers', '1', '--out', str(output)]) == 0
    result = json.loads(output.read_text())
    assert (result['questions'], result['responses'], result['calls'], result['identical']) == (1, 1, 31, 31)
    assert result['stored_differing'] == result['differing'] == result['validation_failures'] == 0
    assert result['missing'] == result['worker_errors'] == []
