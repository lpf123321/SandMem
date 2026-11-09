"""Question-boundary continuation of the same frozen run; never retry a claim."""
import argparse,os,re,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_qwen_half_bench import read,write,sha
import run_qwen_mindcube_reachable as frozen


def verify(out):
    m,original=frozen.verify(out);a=read(out/'mindcube_reachable_runtime/continuation_authorization.json')
    assert a['manifest_sha256']==sha(out/'manifest.json')
    assert a['controller_sha256']==sha(__file__)
    assert a['launcher_sha256']==sha(ROOT/'scripts/continue_mindcube_reachable.sbatch')
    return m,a


def remaining(out,m):
    pending=[]
    for item in m['inputs']:
        q=out/('question_'+item['id'])
        if (q/'result.json').exists():continue
        if (q/'a100_claim').exists() or (q/'claim.json').exists():
            raise RuntimeError('unfinished attempted question; do not retry: '+item['id'])
        pending.append(item['id'])
    return pending


def run(out):
    import torch
    m,a=verify(out);pending=remaining(out,m);runtime=out/'mindcube_reachable_runtime'
    assert read(out/'gate_passed.json')['status']=='passed'
    assert not (runtime/'failure.json').exists()
    prop=torch.cuda.get_device_properties(0);assert 'A100' in prop.name
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    original=read(runtime/'authorization.json')
    required=2*(original['measured_MindCube_peak_reserved_bytes']+4*2**30)+2*2**30
    workers=2 if prop.total_memory>=required else 1
    job=os.environ['SLURM_JOB_ID'];start=time.time();children=[];logs=[];stop=[False]
    def boundary(*_):
        stop[0]=True
        for child in children:
            if child.poll() is None:child.send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1,boundary)
    write(runtime/f'hardware_{job}.json',dict(hardware=prop.name,total_memory_bytes=prop.total_memory,workers=workers,
        allocated_GPUs=1,continuation=True,initial_remaining=len(pending),manifest_sha256=sha(out/'manifest.json')))
    for rank in range(workers):
        log=(runtime/f'worker_{job}_{rank}.log').open('x');logs.append(log)
        children.append(subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/run_qwen_mindcube_reachable.py'),
            'worker','--output',str(out),'--rank',str(rank)],stdout=log,stderr=subprocess.STDOUT))
    while any(c.poll() is None for c in children):
        if (runtime/'failure.json').exists():boundary()
        time.sleep(2)
    codes=[c.wait() for c in children]
    for log in logs:log.close()
    write(runtime/f'allocation_{job}.json',dict(exit_codes=codes,GPUs=1,workers=workers,allocated_wall_seconds=time.time()-start,
        hardware=prop.name,continuation=True))
    if any(codes) or (runtime/'failure.json').exists():raise RuntimeError('continuation failed; no automatic model retry')
    pending=remaining(out,m)
    if pending:
        assert stop[0],'unexpected unfinished continuation'
        write(runtime/f'boundary_{job}.json',dict(status='stopped_at_question_boundary',remaining=len(pending)))
        return
    frozen.component_summary(out);frozen.score(out)


def seconds(value):
    days,clock=(value.split('-',1) if '-' in value else ('0',value))
    parts=list(map(int,clock.split(':')))
    while len(parts)<3:parts.insert(0,0)
    return int(days)*86400+parts[0]*3600+parts[1]*60+parts[2]


def control(out):
    m,a=verify(out);runtime=out/'mindcube_reachable_runtime';job=str(a['initial_job']);signalled=set()
    write(runtime/'continuation_controller.json',dict(pid=os.getpid(),job_id=job,status='watching'))
    while True:
        if (runtime/'completion.json').exists() or (runtime/'failure.json').exists():
            write(runtime/'continuation_controller.json',dict(pid=os.getpid(),job_id=job,status='completed' if (runtime/'completion.json').exists() else 'stopped_internal_failure'));return
        if (runtime/'USER_STOP.json').exists():return
        live=subprocess.check_output(['squeue','-h','-j',job,'-o','%T|%M'],text=True).strip()
        if live:
            state,elapsed=live.split('|',1)
            if state=='RUNNING' and seconds(elapsed)>=a['checkpoint_after_seconds'] and job not in signalled:
                subprocess.run(['scancel','--signal=USR1','--batch',job],check=True);signalled.add(job)
                write(runtime/'continuation_controller.json',dict(pid=os.getpid(),job_id=job,status='finishing_current_questions'))
            time.sleep(20);continue
        state=subprocess.check_output(['sacct','-j',job,'-X','-n','-o','State'],text=True).strip()
        if not state:time.sleep(5);continue
        if state!='COMPLETED':
            write(runtime/'continuation_controller.json',dict(pid=os.getpid(),job_id=job,status='stopped_without_retry',scheduler_state=state));return
        if (runtime/'completion.json').exists():return
        allocation=runtime/f'allocation_{job}.json'
        assert allocation.exists() and not any(read(allocation)['exit_codes'])
        assert job in signalled,'do not resume an unrequested stop'
        assert remaining(out,m),'completed inference did not produce final scoring receipt'
        verify(out)
        job=subprocess.check_output(['sbatch','--parsable','--partition=gpu','--gres=gpu:a100:1','--cpus-per-task=2',
            '--time=08:00:00','--job-name=qwen-mind-resume','--output',str(runtime/'slurm_%j.out'),
            '--error',str(runtime/'slurm_%j.err'),str(ROOT/'scripts/continue_mindcube_reachable.sbatch'),str(out)],text=True).strip().split(';')[0]
        write(runtime/'continuation_controller.json',dict(pid=os.getpid(),job_id=job,status='continuation_submitted'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['control','run']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(control if a.action=='control' else run)(a.output.resolve())
