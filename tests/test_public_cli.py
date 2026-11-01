import json
from pathlib import Path

import pytest

from artifactauditbench.cli import ROOT, demo, main, rescore


@pytest.mark.parametrize('method,count,f1', [
    ('ruleaudit',120,0.8708487084870847),
    ('direct',120,0.3909090909090909),
    ('tool',119,0.6068376068376069),
    ('ruleaudit_scale',360,0.878228782287823),
])
def test_saved_labels_match_manuscript_anchors(method,count,f1):
    result=rescore(method)
    assert result['metrics']['task_count']==count
    assert result['metrics']['multi_label_micro_f1']==pytest.approx(f1)
    assert result['provider_calls']==0
    assert 'cost' not in result['metrics']


def test_failure_policy_keeps_the_missing_tool_task():
    result=rescore('tool','count_as_miss')
    assert result['metrics']['task_count']==120
    assert result['available_predictions']==119
    assert result['missing_task_ids']==['test_seen/v2_449a4b53b64285c4']
    assert result['metrics']['multi_label_micro_f1']==pytest.approx(0.6016949152542372)


def test_dev_demo_has_bounded_inputs_and_known_numeric_error():
    result=demo()
    assert result['scope'].startswith('KNOWN_DEV_CASE')
    assert result['exact_error_set_match']
    assert result['reference_error_types']==['numeric_copy_error']
    assert set(result['input_files_read_by_auditor'])=={'protocol.json','observations.csv','results.csv','statement.txt','visual.png'}


def test_public_cli_rejects_frozen_output_and_overwrite(tmp_path):
    with pytest.raises(SystemExit):
        main(['ruleaudit','--output',str(ROOT/'outputs/v2/rule_audit')])
    with pytest.raises(SystemExit):
        main(['ruleaudit','--output',str(ROOT/'data_v2/new')])
    # An already-existing runs/ directory must not be overwritten.
    runs=ROOT/'runs'/'existing-test-output'
    runs.mkdir(parents=True,exist_ok=True)
    try:
        with pytest.raises(SystemExit):
            main(['ruleaudit','--output',str(runs)])
    finally:
        runs.rmdir()
    with pytest.raises(SystemExit):
        main(['rescore','--method','not-a-method'])


def test_runs_symlink_cannot_redirect_writes(tmp_path, monkeypatch):
    import artifactauditbench.cli as cli
    checkout=tmp_path/'checkout'
    checkout.mkdir()
    elsewhere=tmp_path/'elsewhere'
    elsewhere.mkdir()
    (checkout/'runs').symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.setattr(cli,'ROOT',checkout)
    def must_not_dispatch(*args):
        pytest.fail('output boundary must reject symlinks before executing a runner')
    monkeypatch.setattr(cli,'run_script',must_not_dispatch)
    with pytest.raises(SystemExit):
        cli.main(['ruleaudit','--output',str(checkout/'runs/new-output')])
    assert not list(elsewhere.iterdir())
