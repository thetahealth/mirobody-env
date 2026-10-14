"""Run entry reads the durable cap and refuses mismatches before loading a job."""
from pathlib import Path
import pytest
from haenv import cli, semantic_budget as budget


def test_existing_cap_is_read_without_rewriting_ledger(tmp_path):
    path = tmp_path / 'budget.json'
    ledger = budget.BudgetLedger(path, limit_usd='4.25')
    ledger.reserve('pending', '.1')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert budget.run_budget_limit(path) == '4.25'
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


@pytest.mark.parametrize('given', ['4', '5'])
def test_mismatched_explicit_cap_is_refused(tmp_path, given):
    path = tmp_path / 'budget.json'
    budget.BudgetLedger(path, limit_usd='4.25')
    with pytest.raises(ValueError, match='differs from ledger limit'):
        budget.run_budget_limit(path, given)


def test_new_ledger_requires_approval_amount(tmp_path):
    with pytest.raises(ValueError, match='new ledger needs'):
        budget.run_budget_limit(tmp_path / 'budget.json')
    assert budget.run_budget_limit(tmp_path / 'budget.json', '5') == '5'


@pytest.mark.parametrize('given', [None, '4', '5'])
def test_cli_uses_ledger_cap_before_job_load(tmp_path, monkeypatch, capsys, given):
    path = tmp_path / 'budget.json'
    budget.BudgetLedger(path, limit_usd='4.25')
    class JobReached(Exception):
        pass
    def job(*args):
        raise JobReached
    monkeypatch.setattr(cli, '_resolve_input', job)
    args = ['run', 'inputs/example-ew.job.yaml', '--judge-budget-ledger', str(path)]
    if given is not None:
        args += ['--judge-budget-usd', given]
        assert cli.main(args) == 2
        assert 'differs from ledger limit' in capsys.readouterr().out
    else:
        with pytest.raises(JobReached):
            cli.main(args)
        assert 'pinned ledger budget: $4.25' in capsys.readouterr().out
