"""Original frozen Qwen D116: answer-independent 50%-per-type benchmark runs."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/baselines_vsibench'),str(ROOT/'vendor/vggt'),str(ROOT/'vendor/sam2')]
from run_spatial_vsibench import read,write,sha
DATA=ROOT/'work/baselines_vsibench_20261105'
MIND=ROOT/'work/mindcube_data_20261107/extracted/data'
SEED='SandMem-Qwen-D116-half-v1-20261107'


def stratify(rows, inputs, group):
    pools=defaultdict(list)
    for row in rows:
        pools[group(row)].append(row['id'])
    byid={r['id']:r for r in inputs}
    chosen=[];counts={}
    for kind,ids in sorted(pools.items()):
        ids=sorted(ids,key=lambda i:hashlib.sha256((SEED+'\n'+json.dumps(byid[i],sort_keys=True,ensure_ascii=False)).encode()).hexdigest())
        selected=ids[:(len(ids)+1)//2];chosen.extend(selected)
        counts[kind]=dict(full=len(ids),sample=len(selected))
    return [byid[i] for i in chosen], counts


def mind_inputs(rows):
    return [dict(id=r['id'],dataset='mindcube',scene_name=hashlib.sha256(json.dumps(r['images']).encode()).hexdigest()[:24],
        question=r['question'],options=[],images=r['images'],formatted_question=r['question']+
        "\nAnswer with the option's letter from the given choices directly.") for r in rows]


def make_view_videos(items,out):
    # Native image adapter. No video encoding, sampling, duplicate or extra view.
    assets=out/'mindcube_views';assets.mkdir()
    unique={r['scene_name']:r['images'] for r in items};videos=[]
    for scene,names in sorted(unique.items()):
        sources=[]
        for name in names:
            p=(MIND/name).resolve();assert p.is_relative_to(MIND.resolve()) and p.is_file(),p
            sources.append(dict(path=str(p),sha256=sha(p)))
        path=assets/(scene+'.views.json');write(path,dict(source_images=sources))
        videos.append(dict(dataset='mindcube',scene_name=scene,source_video=str(path),source_sha256=sha(path),
            source_fps=1.,source_frames=len(names),sampled_watermark=len(names)-1,source_images=sources,
            frame_mapping=[dict(frame_id=i,source_index=i,t_sec=float(i)) for i in range(len(names))],
            sequence_semantics='native original ordered images; indices are ordinal views, not physical time'))
    return videos


def prepare(out):
    if out.exists():raise FileExistsError(out)
    original=read(ROOT/'provenance/frozen_files.json')['source_sha256']
    for p,h in original.items():assert sha(ROOT/p)==h,p
    rows=[json.loads(x) for x in (DATA/'data/test.jsonl').read_text().splitlines()]
    from prepare_sandmem_vsibench import question_input
    vsi,vc=stratify(rows,[question_input(r) for r in rows],lambda r:r['question_type'])
    mr=[json.loads(x) for x in (MIND/'raw/MindCube_tinybench.jsonl').read_text().splitlines()]
    mind,mc=stratify(mr,mind_inputs(mr),lambda r:r['id'].split('_')[0]+'/'+str(r['type']))
    out.mkdir(parents=True);videos=read(ROOT/'work/sandmem_vsibench_20261106_01/videos.json')
    keys={(r['dataset'],r['scene_name']) for r in vsi};videos=[v for v in videos if (v['dataset'],v['scene_name']) in keys]
    for v in videos:v['source_sha256']=sha(v['source_video'])
    videos+=make_view_videos(mind,out)
    cfg=read(ROOT/'configs/reference_qwen.json')
    # Original Qwen config only: none of the Spatial syntax/length/alignment adapters.
    assert cfg['qa']['mllm']['model_id']=='Qwen/Qwen3-VL-2B-Instruct'
    sources=dict(original)
    for name in ['scripts/run_qwen_half_bench.py','scripts/qwen_d116_cold.py','scripts/run_qwen_half_bench.sbatch',
                 'scripts/run_spatial_vsibench.py','scripts/prepare_sandmem_vsibench.py',
                 'scripts/baselines_vsibench/score.py','scripts/baselines_vsibench/official_components.py',
                 'sandbox_vlm/backbones/qwen_reference_bundle.py','sandbox_vlm/data/view_sequence.py','tests/test_qwen_half_bench.py']:
        sources[name]=sha(ROOT/name)
    for name in sources:
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    native=out/'native_sam';native.mkdir();binary=Path('/home/u2024311136/sandbox/vendor/sam2/sam2/_C.so')
    (native/'_C.so').write_bytes(binary.read_bytes())
    # Selected-only references are held outside the inference inputs.
    selected={r['id'] for r in vsi};write(out/'vsi_references.json',[r for r in rows if r['id'] in selected])
    selected={r['id'] for r in mind};write(out/'mind_references.json',[r for r in mr if r['id'] in selected])
    items=vsi+mind
    # Interleave data sets at the queue level; two workers can help either data set.
    ordered=[]
    for i in range(max(len(vsi),len(mind))):
        if i<len(vsi):ordered.append(vsi[i])
        if i<len(mind):ordered.append(mind[i])
    # Gate on two input-fixed samples per data set, no score inspection.
    gate=[]
    for dataset,pool in [('vsi',vsi),('mind',mind)]:
        gate.extend(pool[:2])
    gate_ids={str(r['id']) for r in gate};ordered=gate+[r for r in ordered if str(r['id']) not in gate_ids]
    official=ROOT/'work/mindcube_official_20261107'
    scorer_files={str(p):sha(p) for p in (official/'src/evaluation').rglob('*.py')}
    write(out/'manifest.json',dict(scope='qwen_half_sample',method='D116 + first_last_then_oldest_observed',
        config=cfg,inputs=ordered,videos=videos,source_sha256=sources,native_extension_sha256=sha(native/'_C.so'),
        sample_seed=SEED,selection='within each declared question type rank SHA256(seed + whitelisted input), take ceil(N/2)',
        stratification={'VSI-Bench':vc,'MindCube-tinybench':mc},question_counts={'VSI-Bench':len(vsi),'MindCube-tinybench':len(mind)},
        reference_sha256={str(out/'vsi_references.json'):sha(out/'vsi_references.json'),str(out/'mind_references.json'):sha(out/'mind_references.json')},
        mindcube_dataset=read(ROOT/'work/mindcube_data_20261107/download.json'),
        mindcube_official_commit=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip(),
        mindcube_scorer_sha256=scorer_files,gate_question_ids=[r['id'] for r in gate],
        answers_or_question_types_in_inference=False,historical_semantic_cache_reuse=False,
        full_benchmark_score_claim=False,training=False,quantization=False,
        maximum_discovery_tokens=512,sparse_QA_tokens=256,dense_QA_tokens=64,
        maximum_joint_images=33,maximum_joint_visual_tokens=8192,maximum_joint_text_tokens=32768,
        maximum_structure_bytes=64*2**20,hardware_policy='A100 preferred; available dual A30 with original BF16 weights',
        score_policy='Only after all selected predictions persist; official scorers, invalid wrong, full selected denominator',
        input_views='MindCube official tinybench; original ordered 2-4 views, no GT poses/maps/meta_info; native image reader; no video conversion'))
    write(out/'freeze.json',dict(manifest_sha256=sha(out/'manifest.json'),source_files=len(sources),scores_used=False,model_calls=0))
    write(out/'progress.json',dict(status='prepared',completed=0,total=len(ordered)))
    print(json.dumps(dict(output=str(out),counts={'VSI-Bench':len(vsi),'MindCube-tinybench':len(mind)},types={'VSI-Bench':vc,'MindCube-tinybench':mc}),indent=2),flush=True)


def expected_failure(error):
    return (isinstance(error,RuntimeError) and str(error).startswith('QA contract failure:')) or (
        isinstance(error,ValueError) and str(error) in ('missing exact final line; no repair','empty answer'))


def worker(out,rank):
    import torch,cv2
    from sandbox_vlm.backbones.qwen_reference_bundle import QwenReferenceBundle
    from qwen_d116_cold import fresh_question
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    m=read(out/'manifest.json');assert read(out/'freeze.json')['manifest_sha256']==sha(out/'manifest.json')
    for p,h in m['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(out/'native_sam/_C.so')==m['native_extension_sha256']
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(0)
    stop=[False];signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True))
    job=os.environ['SLURM_JOB_ID'];start=time.time()
    write(out/f'worker_{job}_{rank}.json',dict(status='loading',device=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),rank=rank,start=start))
    bundle=QwenReferenceBundle(m['config'])
    videos={(v['dataset'],v['scene_name']):v for v in m['videos']}
    claimed=0
    try:
        for item in m['inputs']:
            if stop[0] or time.time()-start>9*3600:break
            if (out/'failure.json').exists():break
            key=str(item['id']);folder=out/('question_'+key)
            if (folder/'result.json').exists():continue
            if item['id'] not in m['gate_question_ids']:
                # Both workers must pass the fixed input gate before the rest.
                while not all((out/('question_'+str(i))/'result.json').exists() for i in m['gate_question_ids']):
                    if (out/'failure.json').exists() or stop[0]:return
                    time.sleep(2)
                if not (out/'gate_passed.json').exists():
                    gates=[read(out/('question_'+str(i))/'result.json') for i in m['gate_question_ids']]
                    if not all(r['status']=='ok' and r['actual_sandbox_in_model'] and r['full_qwen_perception'] for r in gates):
                        raise RuntimeError('fixed Qwen end-to-end gate failed; preserve evidence, no prompt search')
                    write(out/'gate_passed.json',dict(status='passed',ids=m['gate_question_ids'],scores_used=False))
            try:folder.mkdir()
            except FileExistsError:continue # atomic question assignment to one worker only
            write(folder/'claim.json',dict(rank=rank,job_id=job,started=time.time()))
            print('START',key,flush=True);torch.cuda.reset_peak_memory_stats();t=time.time()
            try:
                result=fresh_question(item,videos[(item['dataset'],item['scene_name'])],m['config'],bundle,folder,out/'native_sam')
            except Exception as error:
                write(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
                if not expected_failure(error):raise
                result=dict(question_id=item['id'],status='failed',raw_output='',error=str(error),wall_seconds=time.time()-t)
                write(folder/'result.json',result)
                bundle.runtime.transport_processor.audit_sink=None
                bundle.runtime.model.to(bundle.runtime.device);torch.cuda.empty_cache()
            claimed+=1;print('DONE',key,result['status'],round(time.time()-t,2),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(rank=rank,error=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        signal.signal(signal.SIGUSR1,signal.SIG_IGN)
        write(out/f'worker_{job}_{rank}.json',dict(status='finished',rank=rank,completed_this_session=claimed,
            wall_seconds=time.time()-start,device=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))


def run_workers(out):
    # Distinct one-GPU child processes. Parent forwards checkpoint once per child.
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert len(devices)==2
    processes=[];logs=[];sent=set();start=time.time();job=os.environ['SLURM_JOB_ID']
    def checkpoint(*_):
        for rank,p in enumerate(processes):
            if rank not in sent and p.poll() is None:
                p.send_signal(signal.SIGUSR1);sent.add(rank)
    signal.signal(signal.SIGUSR1,checkpoint)
    for rank,device in enumerate(devices):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
        f=(out/f'worker_{job}_{rank}.log').open('x');logs.append(f)
        processes.append(subprocess.Popen([sys.executable,'-B','-u',__file__,'worker','--output',str(out),'--rank',str(rank)],env=env,stdout=f,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in processes):
        bad=[p.returncode for p in processes if p.poll() is not None and p.returncode!=0]
        if bad and not (out/'failure.json').exists():
            write(out/'failure.json',dict(error='worker failed, including model-load failures',exit_codes=bad))
        if (out/'failure.json').exists():checkpoint()
        time.sleep(2)
    codes=[p.wait() for p in processes]
    for f in logs:f.close()
    m=read(out/'manifest.json');results=[]
    for item in m['inputs']:
        path=out/('question_'+str(item['id']))/'result.json'
        if path.exists():results.append(read(path))
    write(out/f'allocation_{job}.json',dict(exit_codes=codes,allocated_wall_seconds=time.time()-start,GPUs=2,complete=len(results),total=len(m['inputs'])))
    write(out/'progress.json',dict(status='completed' if len(results)==len(m['inputs']) else 'failed' if any(codes) or (out/'failure.json').exists() else 'continuation_required',completed=len(results),total=len(m['inputs'])))
    if any(codes):raise RuntimeError('worker error; preserved without model retry')
    if len(results)==len(m['inputs']):score(out)


def score(out):
    from score import official_scorer
    m=read(out/'manifest.json');refs=read(out/'vsi_references.json');mr=read(out/'mind_references.json')
    results={r['id']:read(out/('question_'+str(r['id']))/'result.json') for r in m['inputs']}
    assert len(results)==len(m['inputs'])
    functions,receipt=official_scorer(DATA/'repos/thinking-in-space')
    scored=[functions['vsibench_process_results'](dict(r),[results[r['id']]['raw_output']])['vsibench_score'] for r in refs]
    vsi=functions['vsibench_aggregate_results'](scored)
    sys.path.insert(0,str(ROOT/'work/mindcube_official_20261107'))
    from src.evaluation.evaluator import BasicEvaluator
    response=out/'mindcube_official_responses.jsonl'
    response.write_text(''.join(json.dumps(dict(id=r['id'],gt_answer=r['gt_answer'],answer=results[r['id']]['raw_output']))+'\n' for r in mr))
    for path,h in m['mindcube_scorer_sha256'].items():assert sha(path)==h,path
    for path,h in m['reference_sha256'].items():assert sha(path)==h,path
    mind=BasicEvaluator().evaluate(str(response))
    costs={}
    for name,reference in [('VSI-Bench',refs),('MindCube-tinybench',mr)]:
        group=[results[r['id']] for r in reference]
        sums=Counter()
        for r in group:sums.update(r.get('costs',{}))
        costs[name]=dict(calls=dict(sums),sum_question_wall_seconds=sum(r.get('wall_seconds',0) for r in group),
            peak_allocated_bytes=max(r.get('peak_allocated_bytes',0) for r in group),
            failures=sum(r['status']!='ok' for r in group))
    write(out/'costs.json',dict(per_dataset=costs,allocations=[read(p) for p in sorted(out.glob('allocation_*.json'))],
        note='Failed/incomplete stage details remain in per-question archives; sum question time differs from allocated GPU time'))
    write(out/'scores.json',dict(label='Qwen3-VL-2B D116 stratified 50% sample; NOT full-benchmark scores',
        vsi_bench=dict(questions=len(refs),scores=vsi,scorer=receipt),mindcube_tinybench=dict(questions=len(mr),official_result=mind),
        status_counts=dict(Counter(r['status'] for r in results.values())),manifest_sha256=sha(out/'manifest.json')))
    write(out/'completion.json',dict(status='completed',questions=len(results),scores=str(out/'scores.json')))


def control(out):
    # Persistently continue at question boundaries. Never repeat an incomplete claim.
    while True:
        if (out/'failure.json').exists():return
        if (out/'completion.json').exists():return
        job=subprocess.check_output(['sbatch','--parsable','--partition=gpu-a30','--gres=gpu:a30:2',
            '--cpus-per-task=2','--time=12:00:00','--job-name=qwen-D116-half',
            '--output',str(out/'slurm_%j.out'),'--error',str(out/'slurm_%j.err'),
            str(ROOT/'scripts/run_qwen_half_bench.sbatch'),str(out)],text=True).strip().split(';')[0]
        write(out/'controller.json',dict(status='submitted',job_id=job,controller_pid=os.getpid()))
        while subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],text=True).strip():time.sleep(20)
        if (out/'failure.json').exists():return
        if not (out/f'allocation_{job}.json').exists():
            write(out/'failure.json',dict(error='allocation ended without boundary receipt',job_id=job));return
        if any(read(out/f'allocation_{job}.json')['exit_codes']):return
        if read(out/'progress.json')['status']=='completed':return


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run-workers','worker','score','control']);p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int)
    a=p.parse_args();out=a.output.resolve()
    {'prepare':prepare,'run-workers':run_workers,'score':score,'control':control,'worker':lambda o:worker(o,a.rank)}[a.action](out)
