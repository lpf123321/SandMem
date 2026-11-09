"""Regression for the observed post-completion SIGUSR1 exit, no model calls."""
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_spatial_vsibench_parallel_v2 as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def parent(tmp_path):
    write(tmp_path / 'failure.json', dict(job_id='42', worker_exit_codes=[-10, 0]))
    write(tmp_path / 'allocation_42.json', dict(worker_exit_codes=[-10, 0]))
    for rank in range(2):
        shard = tmp_path / f'shard_{rank}'
        write(shard / 'manifest.json', dict(inputs=[dict(id=rank)]))
        write(shard / 'worker_42.json', dict(rank=rank, completed=1, total=1))
        write(shard / f'question_{rank}/result.json', dict(question_id=rank, status='ok',
            full_spatial_perception=True, old_mllm_cache_used=False))
    return tmp_path


def test_reconcile_only_persisted_results_after_signal_exit(tmp_path):
    p = parent(tmp_path)
    result = runner.reconcile_completed_signal_shutdown(p)
    assert result['status'] == 'passed'
    assert len(result['completed_results']) == 2
    assert result['new_model_calls'] == result['retries'] == 0
    assert runner.core.read(p / 'failure.json')['worker_exit_codes'] == [-10, 0]


@pytest.mark.parametrize('failure', ['kill', 'incomplete', 'receipt', 'inference', 'qwen'])
def test_real_failure_or_incomplete_work_cannot_be_reconciled(tmp_path, failure):
    p = parent(tmp_path)
    if failure == 'kill':
        write(p / 'failure.json', dict(job_id='42', worker_exit_codes=[-9, 0]))
    elif failure == 'incomplete':
        (p / 'shard_0/question_99').mkdir()
    elif failure == 'receipt':
        write(p / 'shard_0/worker_42.json', dict(rank=0, total=1, completed=99))
    elif failure == 'inference':
        write(p / 'shard_0/failure.json', dict(error='real model failure'))
    else:
        write(p / 'shard_0/question_0/result.json', dict(question_id=0, status='ok', old_mllm_cache_used=True))
    with pytest.raises(RuntimeError):
        runner.reconcile_completed_signal_shutdown(p)


def test_signal_is_sent_once_and_never_to_completed_worker(tmp_path):
    class Child:
        def __init__(self): self.signals = []
        def poll(self): return None
        def send_signal(self, value): self.signals.append(value)
    children = [Child(), Child()]
    write(tmp_path / 'shard_1/worker_42.json', dict(completed=1))
    sent = set()
    runner.checkpoint_children(children, tmp_path, '42', sent)
    runner.checkpoint_children(children, tmp_path, '42', sent)
    assert children[0].signals == [signal.SIGUSR1]
    assert children[1].signals == []


def test_early_request_reaches_core_and_teardown_ignores_late_signal(tmp_path, monkeypatch):
    import torch
    from sandbox_vlm.backbones import spatial_bundle
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    handlers, checkpoints = {}, []
    def register(kind, value):
        previous = handlers.get(kind)
        handlers[kind] = value
        return previous
    monkeypatch.setattr(runner.signal, 'signal', register)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    def props(_):
        handlers[signal.SIGUSR1](signal.SIGUSR1, None)
        return SimpleNamespace(name='CPU fake', total_memory=0, uuid='fake')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', props)
    # Restore the real factory after the worker's local wrapping as well.
    monkeypatch.setattr(spatial_bundle, 'create_spatial_backbone', spatial_bundle.create_spatial_backbone)
    def core_run(_):
        runner.signal.signal(signal.SIGUSR1, lambda *_: checkpoints.append('deferred'))
    monkeypatch.setattr(runner.core, 'run', core_run)
    write(tmp_path / 'shard_0/manifest.json', dict(inputs=[]))
    runner.worker(tmp_path, 0)
    assert checkpoints == ['deferred']
    assert handlers[signal.SIGUSR1] == signal.SIG_IGN
    assert runner.core.read(tmp_path / 'shard_0/worker_42.json')['all_complete'] is True
