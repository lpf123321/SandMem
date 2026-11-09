"""Frozen single-question inference, accelerated parser and two isolated GPUs.

Sharding uses input order only. No model distribution, changed batch shapes,
semantic cache reuse, repeated questions or intermediate score inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import run_spatial_vsibench as core


def assignments(inputs, workers=2):
    groups = [[] for _ in range(workers)]
    for i, item in enumerate(inputs):
        groups[i % workers].append(item)
    return groups


def prepare(parent, out, proof):
    if out.exists():
        raise FileExistsError(out)
    old = core.read(parent / 'manifest.json')
    check = core.read(proof)
    assert check['status'] == 'passed' and check['model_calls'] == 0
    for name, digest in check['source_sha256'].items():
        assert core.sha(ROOT / name) == digest, name
    assert len(old['inputs']) == 5130 and old['scope'] == 'formal'
    for name, digest in old['source_sha256'].items():
        assert core.sha(ROOT / name) == digest, name
    if (parent / 'failure.json').exists():
        raise RuntimeError('parent infrastructure failure requires diagnosis')
    out.mkdir()
    manifest = dict(old)
    source = dict(old['source_sha256'])
    for name in ('sandbox_vlm/backbones/spatial_fast_syntax.py',
                 'scripts/run_spatial_vsibench_parallel.py', 'scripts/run_spatial_vsibench_parallel.sbatch'):
        source[name] = core.sha(ROOT / name)
    for name in source:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    (out / 'native_sam').symlink_to((parent / 'native_sam').resolve(), target_is_directory=True)
    core.write(out / 'parser_equivalence.json', check)
    manifest.update(source_sha256=source, parent_formal=str(parent), parent_manifest_sha256=core.sha(parent / 'manifest.json'),
        execution_optimization=dict(workers=2, device='one unchanged single-GPU model per worker',
            assignment='original input position modulo 2', discovery_parser='exact-token-mask-equivalent traversal',
            equivalence_receipt_sha256=core.sha(proof), changed_method_settings=False,
            cross_question_semantic_cache=False, no_retries=True))
    core.write(out / 'manifest.json', manifest)
    inherited = []
    for rank, items in enumerate(assignments(old['inputs'])):
        shard = out / f'shard_{rank}'
        shard.mkdir()
        sub = dict(manifest, inputs=items, shard_rank=rank, global_questions=5130)
        core.write(shard / 'manifest.json', sub)
        (shard / 'native_sam').symlink_to((out / 'native_sam').resolve(), target_is_directory=True)
        for item in items:
            name = f'question_{item["id"]}'
            previous = parent / name
            if (previous / 'result.json').exists():
                assert core.read(previous / 'result.json')['question_id'] == item['id']
                (shard / name).symlink_to(previous.resolve(), target_is_directory=True)
                (out / name).symlink_to(previous.resolve(), target_is_directory=True)
                inherited.append(dict(question_id=item['id'], source=str(previous.resolve()),
                    result_sha256=core.sha(previous / 'result.json')))
            else:
                if previous.exists():
                    raise RuntimeError('uncompleted parent input must not be retried: ' + name)
                (out / name).symlink_to(shard / name, target_is_directory=True)
    core.write(out / 'inherited_results.json', inherited)
    core.write(out / 'authorization.json', dict(user_request='2026-11-07 可以，请开始优化',
        scope='equivalent parser, fixed input-order sharding, two A30 cards within verified per-job limit',
        manifest_sha256=core.sha(out / 'manifest.json'), old_results_preserved=True, scores_read=False))
    return manifest


def worker(out, rank):
    import torch
    from sandbox_vlm.backbones import spatial_bundle
    from sandbox_vlm.backbones.spatial_fast_syntax import install_fast_discovery_grammar
    original = spatial_bundle.create_spatial_backbone

    def factory(*args, **kwargs):
        bundle = original(*args, **kwargs)
        install_fast_discovery_grammar(bundle.discovery.grammar)
        return bundle

    spatial_bundle.create_spatial_backbone = factory
    shard = out / f'shard_{rank}'
    if torch.cuda.device_count() != 1:
        raise RuntimeError('each worker must see exactly one allocated GPU')
    props = torch.cuda.get_device_properties(0)
    core.write(shard / ('hardware_' + os.environ['SLURM_JOB_ID'] + '.json'), dict(
        rank=rank, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], visible_gpus=1,
        name=props.name, total_memory_bytes=props.total_memory, uuid=str(getattr(props, 'uuid', 'unavailable'))))
    core.run(shard)
    items = core.read(shard / 'manifest.json')['inputs']
    completed = [r['id'] for r in items if (shard / f'question_{r["id"]}' / 'result.json').exists()]
    core.write(shard / ('worker_' + os.environ['SLURM_JOB_ID'] + '.json'),
        dict(rank=rank, completed=len(completed), total=len(items), all_complete=len(completed) == len(items)))


def run_workers(out):
    cards = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(cards) != 2 or len(set(cards)) != 2:
        raise RuntimeError('two explicitly allocated GPUs required')
    children = []
    files = []
    job = os.environ['SLURM_JOB_ID']
    started = time.time()
    for rank, card in enumerate(cards):
        log = (out / f'shard_{rank}' / f'slurm_{job}.log').open('xb')
        files.append(log)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=card)
        children.append(subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
            'worker', '--output', str(out), '--rank', str(rank)], env=env, stdout=log, stderr=subprocess.STDOUT))

    def checkpoint(*_):
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, checkpoint)
    stopping = False
    while any(c.poll() is None for c in children):
        if not stopping and any(c.poll() not in (None, 0) for c in children):
            checkpoint()
            stopping = True
        time.sleep(5)
    exits = [c.wait() for c in children]
    for f in files:
        f.close()
    core.write(out / f'allocation_{job}.json', dict(job_id=job, allocated_gpus=2, device='A30',
        wall_seconds=time.time() - started, worker_exit_codes=exits,
        resource_note='allocated GPU time is twice allocation wall time; per-worker peaks archived separately'))
    if any(exits):
        core.write(out / 'failure.json', dict(job_id=job, worker_exit_codes=exits, no_automatic_retry=True))
        raise RuntimeError('worker failed; preserve all outputs and stop continuation')


def live(job):
    rows = subprocess.check_output(['squeue', '-h', '-u', 'u2024311136', '-o', '%i'], text=True).split()
    return str(job) in rows


def control(parent, out, proof, state_dir):
    state_dir.mkdir(parents=True, exist_ok=True)
    status = state_dir / 'parallel_controller.json'
    # Retire only our verified scheduler process; inference stops at a question
    # boundary through the already-frozen SIGUSR1 handler.
    pidfile = ROOT / 'work/spatial_vsibench_e2e_20261107_04/controller_pid.json'
    pid = core.read(pidfile)['pid']
    cmdline = Path(f'/proc/{pid}/cmdline')
    if cmdline.exists():
        command = cmdline.read_bytes().replace(b'\0', b' ').decode()
        if 'controller_safe.py' not in command or str(parent) not in command:
            raise RuntimeError('scheduler PID identity mismatch')
        os.kill(pid, signal.SIGTERM)
    old_job = core.read(parent / 'active_session.json')['job_id']
    if live(old_job):
        subprocess.run(['scancel', '--signal=USR1', '--batch', str(old_job)], check=True)
    core.write(status, dict(status='waiting_for_parent_question_boundary', parent_job=old_job, pid=os.getpid()))
    while live(old_job):
        time.sleep(20)
    prepare(parent, out, proof)
    while True:
        manifest = core.read(out / 'manifest.json')
        results = []
        for item in manifest['inputs']:
            p = out / f'question_{item["id"]}' / 'result.json'
            if p.exists():
                result = core.read(p)
                assert result['question_id'] == item['id']
                results.append(result)
        core.write(out / 'progress.json', dict(completed=len(results), total=5130, status='continuation_required'))
        if len(results) == 5130:
            (out / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
            core.write(out / 'completion.json', dict(status='completed', questions=5130, manifest_sha256=core.sha(out / 'manifest.json')))
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/baselines_vsibench/score.py'), '--root', manifest['data_root'],
                '--predictions', str(out / 'predictions.jsonl'), '--output', str(out / 'scores.json')], check=True)
            core.write(status, dict(status='completed_and_scored', output=str(out)))
            return
        args = ['sbatch', '--parsable', '--job-name=sandmem-spatial-2gpu', '--partition=gpu-a30',
            '--gres=gpu:a30:2', '--cpus-per-task=2', '--time=12:00:00', '--signal=B:USR1@900',
            '--output=' + str(out / 'slurm_%j.out'), '--error=' + str(out / 'slurm_%j.err'),
            str(ROOT / 'scripts/run_spatial_vsibench_parallel.sbatch'), str(out)]
        submitted = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
        record = dict(argv=args, returncode=submitted.returncode, stdout=submitted.stdout, stderr=submitted.stderr, time=time.time())
        with (out / 'submission_receipts.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if submitted.returncode:
            core.write(status, dict(status='submission_failed', receipt=record))
            raise RuntimeError(submitted.stderr)
        job = submitted.stdout.strip().split(';')[0]
        core.write(out / 'active_session.json', dict(job_id=job, workers=2))
        core.write(out / 'progress.json', dict(status='running_or_queued',
            completed_at_session_start=len(results), total=5130, workers=2, job_id=job))
        core.write(status, dict(status='running_or_queued', job_id=job, completed=len(results), output=str(out)))
        while live(job):
            time.sleep(20)
        if (out / 'failure.json').exists():
            core.write(status, dict(status='stopped_worker_failure', failure=core.read(out / 'failure.json')))
            return
        receipt = out / f'allocation_{job}.json'
        if not receipt.exists():
            core.write(status, dict(status='stopped_missing_allocation_receipt', job_id=job))
            return


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['worker', 'run-workers', 'control'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int)
    p.add_argument('--parent', type=Path)
    p.add_argument('--proof', type=Path)
    p.add_argument('--state-dir', type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    if a.action == 'worker':
        worker(out, a.rank)
    elif a.action == 'run-workers':
        run_workers(out)
    else:
        control(a.parent.resolve(), out, a.proof.resolve(), a.state_dir.resolve())
