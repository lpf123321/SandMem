"""Durable, monotonic Vgent continuation inside its Slurm allocation.

Only a successful video-boundary checkpoint can request requeue. Interrupted
or failed videos remain protected and require review, never another generation.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    project=Path(os.environ['SANDMEM_ROOT'])
    run=project/'work/baselines_candidates_20261105/vgent_formal_resume_02'
    job=os.environ['SLURM_JOB_ID'];restart=os.environ.get('SLURM_RESTART_COUNT','0')
    attempt=run/'attempts'/('job_'+job+'_restart_'+restart)
    attempt.mkdir(parents=True,exist_ok=False)
    started=time.time()
    state=dict(status='preflight',job_id=job,restart=restart,
               allocation_start=os.environ.get('SLURM_JOB_START_TIME'),
               allocation_end=os.environ.get('SLURM_JOB_END_TIME'),
               prior_formal_model_calls=2310,prior_pilot_model_calls=19)
    def save():
        state['controller_seconds']=time.time()-started
        (attempt/'controller.json').write_text(json.dumps(state,indent=2))
        (run/'controller_status.json').write_text(json.dumps(state,indent=2))
    save()
    try:
        manifest=json.loads((run/'manifest.json').read_text())
        for name,expected in manifest['files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Frozen dependency or reused prediction changed: '+name)
        if json.loads((run/'pixel_equivalence.json').read_text())['status']!='all_checks_passed':
            raise ValueError('CPU pixel equivalence acceptance required')
        incomplete=[path.name for path in (run/'videos').iterdir()
                    if path.is_dir() and not (path/'complete.json').exists()]
        if incomplete:
            raise RuntimeError('Interrupted/failed videos protected; no model retries: '+str(incomplete))
        before=len(list((run/'videos').glob('*/complete.json')))
        state.update(status='running',video_checkpoints_before=before);save()
        result=subprocess.run([sys.executable,'-u',str(run/'runtime/vgent_full.py'),
            '--data-root',str(project/'work/baselines_vsibench_20261105'),'--run',str(run)])
        completion=run/'completion.json'
        if completion.exists():
            (attempt/'model_completion.json').write_bytes(completion.read_bytes())
        if result.returncode:
            raise RuntimeError('Formal process failed; no requeue or failed-input retries')
        complete=json.loads(completion.read_text())
        state.update(status=complete['status'],
            video_checkpoints_after=len(list((run/'videos').glob('*/complete.json'))),
            new_model_calls=complete['new_model_calls'])
        save()
        if complete['status']=='completed_scored':
            # Keep the old run and all old failures immutable. The full score
            # and combined cost receipts belong to this new recovery directory.
            base=run.parent
            summary=dict(method='Vgent default graph-free Spatial135K independent port',
                scores=json.loads((run/'scores.json').read_text()),reused_questions=237,
                old_formal_model_calls=2310,old_pilot_model_calls=19,
                new_formal_model_calls=sum(json.loads(p.read_text())['new_model_calls']
                    for p in (run/'attempts').glob('*/model_completion.json')),
                archived_attempts=[str(p.relative_to(run)) for p in (run/'attempts').glob('*/controller.json')])
            (base/'reports/vgent_recovered_results.json').write_text(json.dumps(summary,indent=2))
            return
        if complete['status']!='video_boundary_checkpoint' or state['video_checkpoints_after']<=before:
            raise RuntimeError('Only a successful checkpoint with forward progress may requeue')
        state['status']='normal_checkpoint_requeue_requested';save()
        print('Normal checkpoint saved; requesting requeue of',job,flush=True)
        result=subprocess.run(['scontrol','requeue',job],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        # Slurm may terminate this allocation before the RPC returns; intent,
        # exact progress and costs are already persisted above.
        (attempt/'requeue_receipt.json').write_text(json.dumps(dict(
            returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2))
        if result.returncode:
            raise RuntimeError('Checkpoint safe, but scheduler rejected requeue: '+result.stderr)
    except Exception:
        state.update(status='stopped_requires_review',failure=traceback.format_exc());save()
        raise


if __name__=='__main__':
    main()
