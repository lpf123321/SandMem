"""The formal controller must never accept old-memory or incomplete wiring."""
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
control=importlib.import_module('control_spatial_vsibench')


def gate(tmp_path):
    m=dict(source_sha256={})
    control.write(tmp_path/'manifest.json',m)
    r=dict(status='passed',question_ids=[69,1077],manifest_sha256=control.sha(tmp_path/'manifest.json'),
        source_sha256={},results=[dict(question_id=i,status='ok',full_spatial_perception=True,
            old_mllm_cache_used=False,actual_sandbox_in_model=True) for i in [69,1077]])
    return r


@pytest.mark.parametrize('bad', ['old_memory','no_sandbox','wrong_ids','changed_manifest'])
def test_gate_rejects_invalid_acceptance(tmp_path,bad):
    r=gate(tmp_path)
    if bad=='old_memory':r['results'][0]['old_mllm_cache_used']=True
    if bad=='no_sandbox':r['results'][1]['actual_sandbox_in_model']=False
    if bad=='wrong_ids':r['question_ids']=[69,1000]
    if bad=='changed_manifest':control.write(tmp_path/'manifest.json',dict(changed=True))
    control.write(tmp_path/'completion.json',r)
    with pytest.raises(ValueError):control.validate_gate(tmp_path)


def test_gate_accepts_only_complete_fixed_cold_scope(tmp_path):
    control.write(tmp_path/'completion.json',gate(tmp_path))
    _,r=control.validate_gate(tmp_path)
    assert [x['question_id'] for x in r['results']]==[69,1077]
