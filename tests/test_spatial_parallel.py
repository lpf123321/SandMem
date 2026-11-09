import copy
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_spatial_vsibench_parallel as parallel


def test_all_5130_inputs_have_one_worker_and_keep_per_worker_order():
    # Nonconsecutive IDs catch accidental assignment by benchmark ID or task.
    inputs = [{'id': i * 3 + 5, 'question': f'input {i}'} for i in range(5130)]
    original = copy.deepcopy(inputs)
    groups = parallel.assignments(inputs)
    assert [len(g) for g in groups] == [2565, 2565]
    assert len({r['id'] for g in groups for r in g}) == 5130
    for rank, group in enumerate(groups):
        assert group == inputs[rank::2]
    assert inputs == original


def test_workers_cannot_share_a_single_allocated_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    with pytest.raises(RuntimeError, match='two explicitly allocated'):
        parallel.run_workers(tmp_path)


def test_each_worker_receives_a_distinct_allocated_device_and_log(tmp_path, monkeypatch):
    for i in range(2):
        (tmp_path / f'shard_{i}').mkdir()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,3')
    monkeypatch.setenv('SLURM_JOB_ID', 'testjob')
    calls = []

    class Child:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs['env']['CUDA_VISIBLE_DEVICES'], kwargs['stdout'].name))

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(parallel.subprocess, 'Popen', Child)
    monkeypatch.setattr(parallel.signal, 'signal', lambda *a: None)
    parallel.run_workers(tmp_path)
    assert [c[1] for c in calls] == ['1', '3']
    assert [c[0][-1] for c in calls] == ['0', '1']
    assert len({c[2] for c in calls}) == 2
    assert parallel.core.read(tmp_path / 'allocation_testjob.json')['allocated_gpus'] == 2
