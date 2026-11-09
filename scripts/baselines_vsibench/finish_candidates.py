"""Serial orchestration only; no prompts/configuration are selected from scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def wait_job(job):
    while True:
        result = subprocess.run(['squeue','--noheader','-j',str(job),'--format=%T'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if result.returncode==0 and not result.stdout.strip():
            return
        if result.returncode and 'Invalid job id' in result.stderr:
            return
        if result.returncode:
            raise RuntimeError(result.stderr)
        time.sleep(20)


def report(root):
    output = dict(benchmark='VSI-Bench original full 5130 questions / 288 videos',
        model='Spatial-MLLM-135K',
        excluded=dict(s_agent='inference code not released',r4dsg='partial cached-observation reference; original video front-end absent'),
        egagent='paused by explicit user instruction',methods={})
    for name, folder in [('SimpleStream','simple_formal_01'),('Vgent','vgent_formal_01')]:
        run=root/folder
        if (run/'scores.json').exists():
            result=json.loads((run/'scores.json').read_text())
            output['methods'][name]=dict(status='completed_scored',**result)
        else:
            output['methods'][name]=dict(status='no_complete_score',run=str(run))
        cost=dict(saved_model_outputs=0,generated_tokens=0,reached_output_limit=0,
                  visual_tokens=0,text_and_special_tokens=0,actual_source_observations=0)
        patterns=['videos/*/query_*/output.json'] if name=='SimpleStream' else ['videos/*/call_*/output.json']
        for pattern in patterns:
            for path in run.glob(pattern):
                item=json.loads(path.read_text());cost['saved_model_outputs']+=1
                cost['generated_tokens']+=item.get('output_tokens',0)
                cost['reached_output_limit']+=int(item.get('reached_output_limit',False))
                receipt=path.parent/('input_receipt.json' if name=='SimpleStream' else 'receipt.json')
                if receipt.exists():
                    data=json.loads(receipt.read_text())
                    cost['visual_tokens']+=data['visual_tokens']
                    cost['text_and_special_tokens']+=data['text_and_special_input_tokens']
                    cost['actual_source_observations']+=sum(len(media['frame_ids']) for media in data['media'])
        if name=='SimpleStream':
            cost['native_spatial_temporal_slots']=cost['actual_source_observations']*2
        cost['scope']='formal only; wiring separately in original receipts; failed forwards may lack an output'
        output['methods'][name]['cost']=cost
    serial=root/'reports/serial_completion.json'
    if serial.exists():
        state=json.loads(serial.read_text());output['pipeline']=state
        jobs=['17319']+[state['wire_job']] if state.get('wire_job') else ['17319']
        jobs+=state.get('formal_jobs',[])
        accounting=subprocess.run(['sacct','-j',','.join(jobs),
            '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES','--parsable2','--noheader'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        (root/'reports/slurm_accounting.txt').write_text(accounting.stdout+accounting.stderr)
    (root/'reports/results.json').write_text(json.dumps(output,indent=2))
    lines=['# Additional Spatial135K VSI-Bench results','',
           '| Method | Status | Overall |','|---|---|---|']
    for name,data in output['methods'].items():
        value=data.get('scores',{}).get('overall')
        lines.append('| %s | %s | %s |'%(name,data['status'],('%.4f%%'%value) if value is not None else 'Unavailable'))
    lines += ['', 'S-Agent: excluded (inference code absent). R4DSG: excluded from faithful end-to-end comparison (partial release only). EGAgent: paused by user.',
              '', 'Independent Spatial ports; no strict equal-budget claim. See per-run configuration and all eight task scores in results.json.']
    (root/'reports/results.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--project',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve();os.chdir(args.project)
    os.environ['SANDMEM_ROOT']=str(args.project.resolve())
    prepared=root/'vgent_formal_prepared_01'
    expected=json.loads((prepared/'manifest.json').read_text())
    for path,digest in expected['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Prepared formal source changed: '+path)
    state=dict(status='waiting_for_simple_and_vgent_wiring',formal_jobs=[])
    def save():
        (root/'reports/serial_completion.json').write_text(json.dumps(state,indent=2))
    save()
    try:
        # A separate immutable submitter queues the two-input Vgent wiring job.
        receipt=root/'reports/vgent_serial_submit_receipt.json'
        while not receipt.exists():
            time.sleep(20)
        submitted=json.loads(receipt.read_text())
        if submitted['returncode']:
            raise RuntimeError('Vgent wiring could not be submitted: '+submitted['stderr'])
        job=submitted['stdout'].strip().split()[-1];state['wire_job']=job;save()
        wait_job(job)
        pilot=root/'vgent_wire_01'
        if not (pilot/'completion.json').exists():
            raise RuntimeError('Vgent wiring ended without a completion receipt')
        acceptance=json.loads((pilot/'completion.json').read_text())
        if acceptance['status']!='wiring_passed_not_scored':
            raise RuntimeError('Vgent wiring failed; no formal run started')
        run=root/'vgent_formal_01'
        cmd=[sys.executable,str(prepared/'runtime/freeze_vgent.py'),
             '--pilot',str(pilot),'--pilot-freeze',str(root/'vgent_wire_frozen_02'),
             '--inputs',str(root/'simple_formal_01/inputs.jsonl'),
             '--repo',str(root/'repos/vgent'),'--run',str(run)]
        frozen=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        (root/'reports/vgent_formal_freeze_receipt.json').write_text(json.dumps(dict(
            returncode=frozen.returncode,stdout=frozen.stdout,stderr=frozen.stderr),indent=2))
        if frozen.returncode:
            raise RuntimeError('Vgent freeze rejected: '+frozen.stderr)
        while True:
            launched=subprocess.run(['sbatch','--output='+str(root/'reports/vgent_full_%j.out'),
                '--error='+str(root/'reports/vgent_full_%j.err'),str(prepared/'launcher.sbatch')],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if launched.returncode:
                raise RuntimeError('Vgent formal submission rejected: '+launched.stderr)
            job=launched.stdout.strip().split()[-1];state['formal_jobs'].append(job)
            state['status']='vgent_formal_running';save();wait_job(job)
            completion=json.loads((run/'completion.json').read_text())
            if completion['status']=='completed_scored':
                state['status']='completed';break
            if completion['status']!='video_boundary_checkpoint':
                raise RuntimeError('Vgent stopped on a recorded failure; no automatic retry')
        save()
    except Exception as error:
        state.update(status='stopped_requires_review',reason=str(error));save()
        raise
    finally:
        report(root)


if __name__=='__main__':
    main()
