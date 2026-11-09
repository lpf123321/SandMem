"""Hardware-only handoff; retain all frozen Qwen inputs, code and predictions.

Wait for Vgent's existing normal checkpoint, never interrupt its inference.
Keep A30 working until an A100 submission is secured, then drain at question
boundaries and run the exact frozen Qwen worker on one allocated A100.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_qwen_half_bench import read,write,sha,worker,score


def state(job):
    value=subprocess.check_output(['squeue','-h','-j',str(job),'-o','%T|%r'],text=True).strip()
    return tuple(value.split('|',1)) if value else ('ABSENT','')


def audit_results(out,require_closed=False):
    m=read(out/'manifest.json');results={};incomplete=[]
    for item in m['inputs']:
        p=out/('question_'+str(item['id']))
        if (p/'result.json').exists():
            r=read(p/'result.json');assert r['question_id']==item['id']
            assert r['status'] in ('ok','failed')
            if r['status']=='ok':assert r['full_qwen_perception'] and not r['old_mllm_cache_used'] and r['actual_sandbox_in_model']
            results[str(item['id'])]=sha(p/'result.json')
        elif p.exists():incomplete.append(item['id'])
    if require_closed and incomplete:raise RuntimeError('incomplete question claims cannot be retried: '+str(incomplete))
    return dict(completed=len(results),incomplete=incomplete,result_sha256=results,total=len(m['inputs']))


def validate_vgent_checkpoint(run):
    c=read(run/'completion.json')
    if c['status'] not in ('video_boundary_checkpoint','completed_scored'):raise RuntimeError('Vgent has no successful normal checkpoint')
    incomplete=[p.name for p in (run/'videos').iterdir() if p.is_dir() and not (p/'complete.json').exists()]
    if incomplete:raise RuntimeError('Vgent incomplete videos must not be cancelled: '+str(incomplete))
    return dict(status='passed',completion=c,completion_sha256=sha(run/'completion.json'),
        complete_videos=len(list((run/'videos').glob('*/complete.json'))),no_inference_interrupted=True)


def submit(out,hold=False):
    args=['sbatch','--parsable','--partition=gpu','--gres=gpu:a100:1','--cpus-per-task=1',
        '--time=12:00:00','--job-name=qwen-D116-half-A100',
        '--output',str(out/'a100_slurm_%j.out'),'--error',str(out/'a100_slurm_%j.err')]
    if hold:args+=['--hold']
    args+=[str(ROOT/'scripts/run_qwen_half_a100.sbatch'),str(out)]
    return subprocess.check_output(args,text=True).strip().split(';')[0]


def run(out):
    import torch
    authorization=read(out/'a100_handoff/authorization.json')
    assert authorization['manifest_sha256']==sha(out/'manifest.json')
    assert sha(__file__)==authorization['launcher_sha256']
    assert sha(ROOT/'scripts/run_qwen_half_a100.sbatch')==authorization['sbatch_sha256']
    prop=torch.cuda.get_device_properties(0)
    if 'A100' not in prop.name:raise RuntimeError('A100 requested but different GPU allocated')
    job=os.environ['SLURM_JOB_ID'];started=time.time()
    write(out/f'hardware_{job}.json',dict(device=prop.name,total_memory_bytes=prop.total_memory,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),GPUs=1,manifest_sha256=sha(out/'manifest.json')))
    worker(out,0) # Exact original worker: no prompt/model/input/budget/batch changes.
    audit=audit_results(out,require_closed=True)
    write(out/f'allocation_{job}.json',dict(exit_codes=[0],allocated_wall_seconds=time.time()-started,
        GPUs=1,hardware=prop.name,complete=audit['completed'],total=audit['total']))
    complete=audit['completed']==audit['total']
    write(out/'progress.json',dict(status='completed' if complete else 'continuation_required',
        completed=audit['completed'],total=audit['total'],session_job=job,hardware=prop.name))
    if complete:score(out)


def watch(out):
    folder=out/'a100_handoff';authorization=read(folder/'authorization.json')
    assert authorization['manifest_sha256']==sha(out/'manifest.json')
    assert sha(__file__)==authorization['launcher_sha256']
    baseline=ROOT/'work/baselines_candidates_20261105/vgent_formal_resume_03'
    qwen_job=None
    def save(status,**kw):write(folder/'state.json',dict(status=status,controller_pid=os.getpid(),updated=time.time(),**kw))
    try:
        save('waiting_for_Vgent_normal_checkpoint',vgent_job='17432',A30_continues=True)
        while True:
            if (out/'failure.json').exists():raise RuntimeError('Qwen failure requires review')
            status,reason=state('17432')
            if status=='PENDING' and 'Held' in reason:
                receipt=validate_vgent_checkpoint(baseline)
                write(folder/'vgent_safe_pause.json',receipt)
                subprocess.run(['scancel','--state=PENDING','17432'],check=True)
                break
            if status=='ABSENT':
                receipt=validate_vgent_checkpoint(baseline);write(folder/'vgent_safe_pause.json',receipt);break
            if status not in ('RUNNING','COMPLETING','PENDING','CONFIGURING'):raise RuntimeError('unexpected Vgent state '+status)
            time.sleep(20)
        while state('17432')[0]!='ABSENT':time.sleep(2)
        if (out/'completion.json').exists():
            save('Qwen_completed_before_handoff');resume_vgent(out,baseline);return
        qwen_job=submit(out,hold=True)
        save('A100_submission_secured_draining_A30',a100_job=qwen_job)
        info=read(out/'controller.json');pid=info['controller_pid']
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            assert 'run_qwen_half_bench.py control' in command and str(out) in command
            os.kill(pid,signal.SIGTERM)
        except FileNotFoundError:pass
        oldjob=info['job_id'];checkpoint=folder/('checkpoint_requested_'+oldjob+'.json')
        if state(oldjob)[0] in ('RUNNING','CONFIGURING') and not checkpoint.exists():
            write(checkpoint,dict(job_id=oldjob,requested=time.time(),reason='user requested A100'))
            subprocess.run(['scancel','--signal=USR1','--batch',oldjob],check=True)
        while state(oldjob)[0]!='ABSENT':time.sleep(5)
        oldreceipt=read(out/f'allocation_{oldjob}.json')
        assert not any(oldreceipt['exit_codes']) and not (out/'failure.json').exists()
        closed=audit_results(out,require_closed=True)
        write(folder/'A30_closed.json',dict(old_job=oldjob,allocation=oldreceipt,**closed))
        subprocess.run(['scontrol','release',qwen_job],check=True)
        while True:
            save('A100_running_or_queued',job_id=qwen_job,inherited_completed=closed['completed'])
            while state(qwen_job)[0]!='ABSENT':time.sleep(20)
            if (out/'failure.json').exists():raise RuntimeError('A100 worker failure; no retry')
            receipt=read(out/f'allocation_{qwen_job}.json')
            assert not any(receipt['exit_codes'])
            if (out/'completion.json').exists():
                save('completed_on_A100',last_job=qwen_job);resume_vgent(out,baseline);return
            closed=audit_results(out,require_closed=True)
            qwen_job=submit(out)
    except Exception as error:
        save('stopped_requires_review',error=str(error),traceback=traceback.format_exc(),a100_job=qwen_job)
        if qwen_job and state(qwen_job)[0]=='PENDING':subprocess.run(['scancel','--state=PENDING',qwen_job],check=False)
        raise


def resume_vgent(out,baseline):
    # Fresh allocation, exact unmodified launcher/manifest and all checkpoints.
    if read(baseline/'completion.json')['status']=='completed_scored':return
    job=subprocess.check_output(['sbatch','--parsable','--output',str(ROOT/'work/baselines_candidates_20261105/reports/vgent_after_qwen_%j.out'),
        '--error',str(ROOT/'work/baselines_candidates_20261105/reports/vgent_after_qwen_%j.err'),str(baseline/'launcher.sbatch')],text=True).strip()
    write(out/'a100_handoff/vgent_resume.json',dict(job_id=job,reason='Qwen priority measurement completed',launcher_sha256=sha(baseline/'launcher.sbatch')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','watch']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();{'run':run,'watch':watch}[a.action](a.output.resolve())
