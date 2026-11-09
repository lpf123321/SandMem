"""One authorized gate: two fixed cold inputs, then one frozen full run.

No score-dependent decisions, retries, prompt changes, or additional pilot IDs.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_spatial_vsibench import ROOT,DATA,read,write,sha


def validate_gate(wiring):
    m=read(wiring/'manifest.json');receipt=read(wiring/'completion.json')
    if receipt['status']!='passed' or receipt['question_ids']!=[69,1077]:raise ValueError('fixed cold wiring has not passed')
    if receipt['manifest_sha256']!=sha(wiring/'manifest.json'):raise ValueError('wiring receipt manifest changed')
    if receipt['source_sha256']!=m['source_sha256']:raise ValueError('wiring source receipt differs')
    for p,h in m['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('source changed after wiring: '+p)
    if [r['question_id'] for r in receipt['results']]!=[69,1077]:raise ValueError('unexpected wiring input IDs')
    for row in receipt['results']:
        if row.get('status')!='ok' or not row.get('full_spatial_perception') or row.get('old_mllm_cache_used') or not row.get('actual_sandbox_in_model'):
            raise ValueError('fresh Spatial perception and actual sandbox are required')
    return m,receipt


def freeze(wiring,formal):
    m,receipt=validate_gate(wiring)
    if formal.exists():raise FileExistsError('formal freeze exists; preserve it')
    inputs=[json.loads(line) for line in (DATA/'inputs.jsonl').read_text().splitlines()]
    if len(inputs)!=5130 or len({r['id'] for r in inputs})!=5130:raise ValueError('full original denominator required')
    if sha(DATA/'manifest.json')!=m['data_manifest_sha256']:raise ValueError('data preparation changed')
    for p,h in read(DATA/'manifest.json')['files'].items():
        if sha(p)!=h:raise ValueError('data source changed: '+p)
    videos=read(DATA/'videos.json')
    for v in videos:v['source_sha256']=sha(v['source_video'])
    formal.mkdir(parents=True);shutil.copytree(wiring/'source',formal/'source');shutil.copytree(wiring/'native_sam',formal/'native_sam')
    final=deepcopy(m);final.update(scope='formal',status='frozen',inputs=inputs,videos=videos,
        question_ids=[r['id'] for r in inputs],wiring_receipt_sha256=sha(wiring/'completion.json'),
        formal_authorization='user: fixed two-question acceptance only; once passed immediately freeze and run5130',
        scorer='scripts/baselines_vsibench/score.py',data_root=str(ROOT/'work/baselines_vsibench_20261105'),
        all_inputs_previously_exposed=True,new_blind_test_claim=False,strict_equal_budget_claim=False,
        checkpoint_boundary='question; interrupted logical requests preserved as invalid, never retried',
        maximum_discovery_calls_per_question=final['config']['coverage']['discovery']['maximum'],
        maximum_relevance_logical_requests_per_question='2 * sampled video frames',
        SAM_policy='original local box prompts +32 legacy seeds +D102 coverage schedule, at most128 total seeds',
        VGGT_policy='one <=16-view control call; sparse adds <=8 causal <=16-view observation windows',
        scorer_sha256=sha(ROOT/'scripts/baselines_vsibench/score.py'))
    write(formal/'manifest.json',final)
    write(formal/'authorization.json',dict(frozen=True,manifest_sha256=sha(formal/'manifest.json'),
        gate_receipt_sha256=sha(wiring/'completion.json'),questions=5130,weight_training=False))
    # Exact validated cold executions are part of the same fixed full measurement.
    for item in receipt['results']:
        qid=item['question_id'];source=wiring/f'question_{qid}'
        target=formal/f'question_{qid}';target.symlink_to(source.resolve(),target_is_directory=True)
    write(formal/'progress.json',dict(status='frozen',completed=[69,1077],formal_started=False))
    return final


def submit(out,formal=False):
    jobs=subprocess.check_output(['squeue','-h','-u','u2024311136','-o','%P'],text=True).splitlines()
    # Respect per-partition one-submitted-job policy. Prefer A100 when its
    # partition quota is available; A30 is the user's authorized busy fallback.
    partition='gpu' if 'gpu' not in jobs else 'gpu-a30'
    if partition in jobs:return None
    args=['sbatch','--parsable','--partition='+partition,'--gres=gpu:'+('a100' if partition=='gpu' else 'a30')+':1',
        '--time='+('12:00:00' if formal else '02:00:00'),
        '--signal=B:USR1@600' if formal else '--signal=B:USR1@60',
        '--output='+str(out/'slurm_%j.out'),'--error='+str(out/'slurm_%j.err'),
        str(ROOT/'scripts/run_spatial_vsibench.sbatch'),str(out)]
    submitted=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    record=dict(time=time.time(),argv=args,returncode=submitted.returncode,stdout=submitted.stdout,stderr=submitted.stderr)
    with (out/'submission_receipts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if submitted.returncode:raise RuntimeError(submitted.stderr)
    return submitted.stdout.strip().split(';')[0]


def control(wiring,formal):
    state_path=wiring/'controller.json'
    write(state_path,dict(status='waiting_for_wiring',pid=os.getpid()))
    while not (wiring/'completion.json').exists():
        if (wiring/'failure.json').exists():
            write(state_path,dict(status='stopped_wiring_failure',failure=read(wiring/'failure.json')));return
        receipts=wiring/'submission_receipts.jsonl'
        if receipts.exists():
            last=json.loads(receipts.read_text().splitlines()[-1])
            if last['returncode']==0:
                job=last['stdout'].strip().split(';')[0]
                live=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],text=True).strip()
                if not live:
                    write(state_path,dict(status='stopped_wiring_job_ended_without_pass',job_id=job));return
        time.sleep(20)
    if not formal.exists():
        write(state_path,dict(status='freezing_full_5130_no_scores_used'))
        freeze(wiring,formal)
    validate_gate(wiring)
    while not (formal/'completion.json').exists():
        # Only one sequential session; no retry of a failed/interrupted question.
        active=read(formal/'active_session.json') if (formal/'active_session.json').exists() else None
        if active:
            live=subprocess.check_output(['squeue','-h','-j',active['job_id'],'-o','%T'],text=True).strip()
            if live:time.sleep(20);continue
            if (formal/'failure.json').exists():
                write(state_path,dict(status='stopped_formal_infrastructure_failure',failure=read(formal/'failure.json')));return
        job=submit(formal,formal=True)
        if job:
            write(formal/'active_session.json',dict(job_id=job));write(state_path,dict(status='formal_submitted',job_id=job,output=str(formal)))
        time.sleep(20)
    cfg=read(formal/'manifest.json')
    subprocess.run([str(ROOT/'work/envs/spatial135k/bin/python'),'-B',str(ROOT/'scripts/baselines_vsibench/score.py'),
        '--root',cfg['data_root'],'--predictions',str(formal/'predictions.jsonl'),'--output',str(formal/'scores.json')],check=True)
    write(state_path,dict(status='formal_completed_and_scored',output=str(formal)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','control','submit-wiring'])
    p.add_argument('--wiring',type=Path,required=True);p.add_argument('--formal',type=Path)
    a=p.parse_args();w=a.wiring.resolve()
    if a.action=='submit-wiring':print(submit(w),flush=True)
    elif a.action=='freeze':freeze(w,a.formal.resolve())
    else:control(w,a.formal.resolve())
