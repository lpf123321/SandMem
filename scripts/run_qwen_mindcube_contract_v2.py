"""Prioritize the fixed 527 MindCube questions on A100; exact frozen Qwen D116.

Atomic pre-reservations prevent the unchanged A30 workers taking new MindCube
questions. Two independent one-thread processes may share an allocated A100
only when the measured original MindCube memory use plus headroom permits it.
No model/input budget changes, no question retries and no intermediate scores.
"""
import argparse,collections,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_qwen_half_bench import read,write,sha,expected_failure


def selected(out):return [r for r in read(out/'manifest.json')['inputs'] if r['dataset']=='mindcube']


def claim(folder):
    assert read(folder/'reservation.json')['owner']=='mindcube_contract_v2_runtime'
    try:(folder/'a100_claim').mkdir()
    except FileExistsError:return False
    return True


def verify(out):
    m=read(out/'manifest.json');auth=read(out/'mindcube_contract_v2_runtime/authorization.json')
    assert sha(out/'manifest.json')==auth['manifest_sha256']==read(out/'freeze.json')['manifest_sha256']
    assert sha(out/'component_gate_plan.json')==m['component_gate_plan_sha256']
    assert sha(__file__)==auth['launcher_sha256']
    assert sha(ROOT/'scripts/run_qwen_mindcube_contract_v2.sbatch')==auth['sbatch_sha256']
    for p,h in m['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(out/'native_sam/_C.so')==m['native_extension_sha256']
    assert len(selected(out))==527
    return m,auth


def worker(out,rank):
    import torch,cv2
    from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import QwenClosedContractBundle as QwenReferenceBundle
    from qwen_d116_mindcube_views_v2 import fresh_question
    m,auth=verify(out);priority=out/'mindcube_contract_v2_runtime';job=os.environ['SLURM_JOB_ID'];started=time.time()
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(0)
    stop=[False];signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True))
    prop=torch.cuda.get_device_properties(0);assert 'A100' in prop.name
    write(priority/f'worker_{job}_{rank}.json',dict(status='loading',rank=rank,hardware=prop.name,total_memory_bytes=prop.total_memory))
    bundle=QwenReferenceBundle(m['config']);videos={(v['dataset'],v['scene_name']):v for v in m['videos']};completed=0
    try:
        for item in selected(out):
            if stop[0] or (priority/'failure.json').exists():break
            folder=out/('question_'+str(item['id']))
            if (folder/'result.json').exists():continue
            if item['id'] not in m['gate_question_ids']:
                while not (out/'gate_passed.json').exists():
                    if (priority/'failure.json').exists() or stop[0]:return
                    if all((out/('question_'+str(i))/'result.json').exists() for i in m['gate_question_ids']):
                        try:(out/'gate_acceptance_lock').mkdir()
                        except FileExistsError:time.sleep(1);continue
                        validate_gate(out,m)
                    else:time.sleep(1)
            if not (folder/'reservation.json').exists():continue # Existing in-flight A30 owner, never duplicate.
            if not claim(folder):continue
            write(folder/'claim.json',dict(rank=rank,job_id=job,started=time.time(),hardware=prop.name,owner='mindcube_contract_v2_runtime'))
            torch.cuda.reset_peak_memory_stats();t=time.time();print('START',item['id'],flush=True)
            try:
                result=fresh_question(item,videos[(item['dataset'],item['scene_name'])],m['config'],bundle,folder,out/'native_sam')
            except Exception as error:
                write(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
                # All internal failures stop this frozen run; preserve the
                # attempt and never continue collecting degraded predictions.
                raise
                result=dict(question_id=item['id'],status='failed',raw_output='',error=str(error),wall_seconds=time.time()-t)
                write(folder/'result.json',result)
                bundle.runtime.transport_processor.audit_sink=None;bundle.runtime.model.to(bundle.runtime.device);torch.cuda.empty_cache()
            if result['status']=='ok':
                audit_actual_contract(folder,item)
                validate_question_components(folder)
            completed+=1;print('DONE',item['id'],result['status'],round(time.time()-t,2),flush=True)
    except Exception as error:
        receipt=dict(rank=rank,job_id=job,error=str(error),traceback=traceback.format_exc())
        write(priority/'failure.json',receipt)
        raise
    finally:
        signal.signal(signal.SIGUSR1,signal.SIG_IGN)
        write(priority/f'worker_{job}_{rank}.json',dict(status='finished',rank=rank,completed=completed,
            wall_seconds=time.time()-started,hardware=prop.name,peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def run(out):
    import torch
    m,auth=verify(out);priority=out/'mindcube_contract_v2_runtime';job=os.environ['SLURM_JOB_ID'];start=time.time()
    from accept_qwen_mindcube_contract_v2 import run_component_gate
    run_component_gate(out)
    prop=torch.cuda.get_device_properties(0)
    assert 'A100' in prop.name and len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==1
    # Measured previous full MindCube inference peak, generous independent per-worker headroom.
    # Concurrency only: never shrink an input or fall back to a smaller model.
    required=2*(auth['measured_MindCube_peak_reserved_bytes']+4*2**30)+2*2**30
    workers=2 if prop.total_memory>=required else 1
    write(priority/f'hardware_{job}.json',dict(hardware=prop.name,total_memory_bytes=prop.total_memory,
        workers=workers,allocated_GPUs=1,required_for_two_workers_bytes=required,
        measured_previous_peak_reserved_bytes=auth['measured_MindCube_peak_reserved_bytes'],
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),manifest_sha256=sha(out/'manifest.json')))
    children=[];logs=[];sent=set()
    def checkpoint(*_):
        write(priority/'stop_requested.json',dict(time=time.time(),reason='signal or internal failure'))
        for i,p in enumerate(children):
            if i not in sent and p.poll() is None:p.send_signal(signal.SIGUSR1);sent.add(i)
    signal.signal(signal.SIGUSR1,checkpoint)
    for rank in range(workers):
        log=(priority/f'worker_{job}_{rank}.log').open('x');logs.append(log)
        children.append(subprocess.Popen([sys.executable,'-B','-u',__file__,'worker','--output',str(out),'--rank',str(rank)],stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in children):
        bad=[p.returncode for p in children if p.poll() is not None and p.returncode!=0]
        if bad and not (priority/'failure.json').exists():write(priority/'failure.json',dict(error='worker startup or model failure',exit_codes=bad))
        if (priority/'failure.json').exists():checkpoint()
        time.sleep(2)
    codes=[p.wait() for p in children]
    for log in logs:log.close()
    write(priority/f'allocation_{job}.json',dict(exit_codes=codes,GPUs=1,workers=workers,allocated_wall_seconds=time.time()-start,hardware=prop.name))
    if any(codes) or (priority/'failure.json').exists():raise RuntimeError('MindCube A100 failed; no model retry')
    # A30 may be finishing a claim it owned before the priority reservation.
    while not all((out/('question_'+str(r['id']))/'result.json').exists() for r in selected(out)):
        if (out/'failure.json').exists():raise RuntimeError('existing A30-owned input failed; no retry')
        remaining=[r['id'] for r in selected(out) if not (out/('question_'+str(r['id']))/'result.json').exists()]
        if (priority/'stop_requested.json').exists():
            write(priority/'stopped_at_question_boundary.json',dict(completed=527-len(remaining),remaining=len(remaining)))
            return
        if any((out/('question_'+str(i))/'reservation.json').exists() for i in remaining):
            raise RuntimeError('reserved A100 question incomplete; no automatic retry: '+str(remaining))
        time.sleep(2)
    component_summary(out)
    score(out)


def score(out):
    m,auth=verify(out);priority=out/'mindcube_contract_v2_runtime';refs=read(out/'mind_references.json');items=selected(out)
    assert {r['id'] for r in refs}=={r['id'] for r in items}
    assert sha(out/'mind_references.json')==m['reference_sha256'][str(out/'mind_references.json')]
    for p,h in m['mindcube_scorer_sha256'].items():assert sha(p)==h,p
    records=[];results=[];hardware=collections.Counter();calls=collections.Counter()
    for r in refs:
        folder=out/('question_'+r['id']);prediction=read(folder/'result.json')
        assert prediction['question_id']==r['id'] and prediction['status'] in ('ok','failed')
        if prediction['status']=='ok':assert prediction['full_qwen_perception'] and prediction['actual_sandbox_in_model'] and not prediction['old_mllm_cache_used']
        results.append(prediction);calls.update(prediction.get('costs',{}))
        c=read(folder/'claim.json');hardware[c.get('hardware','NVIDIA A30')]+=1
        records.append(dict(id=r['id'],gt_answer=r['gt_answer'],question=r['question'],answer=prediction['raw_output']))
    response=priority/'official_responses.jsonl';response.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    sys.path.insert(0,str(ROOT/'work/mindcube_official_20261107'))
    from src.evaluation.evaluator import BasicEvaluator
    evaluated=BasicEvaluator().evaluate(str(response));assert evaluated['results']['total']==527
    result=dict(label='Qwen3-VL-2B D116 with isolated output-contract repair v2; MindCube-tinybench 527/1050 fixed sample',
        questions=527,official=evaluated,hardware_question_counts=dict(hardware),
        status_counts=dict(collections.Counter(r['status'] for r in results)),manifest_sha256=sha(out/'manifest.json'),
        sampling=m['stratification']['MindCube-tinybench'],checkpoint=m['config']['qa']['mllm'],
        output_contract=m['output_contract'],component_metrics=read(out/'component_metrics.json'),
        costs=dict(calls=dict(calls),sum_question_wall_seconds=sum(r.get('wall_seconds',0) for r in results),
            A100_allocations=[read(p) for p in priority.glob('allocation_*.json')]),
        original_A30_results_reused_without_retry=False,cold_full_Qwen_perception_rerun=True,blind_test_claim=False)
    write(priority/'scores.json',result);write(out/'mindcube_scores.json',result)
    stats=evaluated['results'];lines=['# Qwen D116：MindCube输出契约修复独立复测','',
        f"总体准确率：{100*stats['gen_cogmap_accuracy']:.4f}%（{stats['gen_cogmap_correct']}/527）。",'',
        '| 类别 | 题数 | 正确数 | 准确率 |','|---|---:|---:|---:|']
    for key in ['among','around','rotation']:
        s=stats['settings'][key];lines.append(f"| {key} | {s['total']} | {s['gen_cogmap_correct']} | {100*s['gen_cogmap_accuracy']:.4f}% |")
    lines+=['','原Qwen3-VL-2B冻结权重、D116完整感知/实例/局部几何/真实沙盒及原帧、既有两阶段规则；每子类ceil(N/2)，527/1050题，非官方全量分数。',
        '本轮527题全部在A100重新执行完整Qwen感知、SAM、VGGT及QA；全部原视图和显式image编号入模。无旧预测/语义缓存复用，无失败重试或按成绩修改。',
        '硬件题数：'+json.dumps(dict(hardware),ensure_ascii=False)+'。','无效答案按官方解析计错。输出契约v2独立结果：严格JSON、固定源帧号、原描述8词、合法框和重复框约束、按剩余额度闭合完整记录；名称不限长；所有QA只允许原题可见选项，候选同样预留FINAL_ANSWER。原始消息及额度不变。']
    (priority/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(priority/'completion.json',dict(status='completed_scored',questions=527,accuracy=stats['gen_cogmap_accuracy'],scores=str(priority/'scores.json')))
    print('MINDCUBE_COMPLETE',json.dumps(dict(questions=527,accuracy=stats['gen_cogmap_accuracy'])),flush=True)


def audit_actual_contract(folder,item):
    r=read(folder/'result.json');expected=list(range(len(item['images'])))
    assert r['actual_frame_ids']==expected and r['dataset_adapter']=='explicit_numbered_views_v1'
    stages=[]
    for p in folder.glob('*_generation.json'):
        stage=p.name.removesuffix('_generation.json');g=read(p);c=read(folder/(stage+'_view_contract.json'))
        messages=read(folder/(stage+'_messages.json'));content=messages[0]['content']
        images=[b for b in content if b['type']=='image'];has_map=stage!='draft'
        assert len(images)==len(expected)+int(has_map)
        assert c['actual_frame_ids']==expected and c['all_views_included']
        texts=[b['text'] for b in content if b['type']=='text']
        assert 'Original question (verbatim):\n'+item['formatted_question'] in texts
        for i in expected:assert f'Original benchmark image {i+1} (actual source frame ID {i}):' in texts
        trace=g['actual_forward_trace'];assert len(trace)==1 and len(trace[0]['image_grid_thw'])==len(images)
        assert g['budget']['status']=='ok'
        stages.append(dict(stage=stage,images=len(images),budget=g['budget'],view_contract=c))
    assert stages
    write(folder/'actual_view_audit.json',dict(status='passed',original_views=len(expected),actual_frame_ids=expected,stages=stages))


def validate_gate(out,m):
    sys.path.insert(0,str(ROOT/'work/mindcube_official_20261107'))
    from src.evaluation.core.extractors import extract_answer
    routes=[]
    for i in m['gate_question_ids']:
        item=next(r for r in m['inputs'] if r['id']==i);folder=out/('question_'+str(i));r=read(folder/'result.json')
        assert r['status']=='ok','fixed input format gate failed; no search/retry'
        audit_actual_contract(folder,item)
        validate_question_components(folder)
        routes.append(bool(r['two_pass']))
        assert extract_answer(r['raw_output']) is not None,'fixed input option parsing gate failed; no search/retry'
    assert any(routes) and not all(routes),'fixed gate must exercise both original D116 input-structure routes'
    write(out/'gate_passed.json',dict(status='passed',ids=m['gate_question_ids'],scores_used=False,all_original_views_and_actual_Qwen_tensor_counts_checked=True))


def validate_question_components(folder):
    from sandbox_vlm.qa.two_pass import candidate_from_generation
    from sandbox_vlm.backbones.qwen_mindcube_contract import visible_choices
    failures=[]
    for p in folder.glob('observation_F*/discovery.json'):
        d=read(p)
        if d['status']!='ok':failures.append(dict(stage='discovery',path=str(p),error=d.get('error')))
    for p in folder.glob('*_generation.json'):
        g=read(p);messages=read(folder/(p.name.removesuffix('_generation.json')+'_messages.json'))
        sparse=g['output_contract']['sparse']
        answer=candidate_from_generation(g)['answer'] if sparse else g.get('raw_output','').strip()
        if g['status']!='generated' or answer not in visible_choices(messages):
            failures.append(dict(stage=p.stem,status=g['status'],error='unusable option field'))
    write(folder/'component_contract_audit.json',dict(status='passed' if not failures else 'failed',failures=failures))
    if failures:raise ValueError('fixed end-to-end component gate failed; no search or automatic retry')


def component_summary(out):
    from sandbox_vlm.qa.two_pass import candidate_from_generation
    counts=collections.Counter();errors=collections.Counter()
    for item in selected(out):
        folder=out/('question_'+item['id']);result=read(folder/'result.json')
        counts['questions']+=1
        for p in folder.glob('observation_F*/discovery.json'):
            d=read(p);counts['discovery_calls']+=1
            if d['status']!='ok':counts['discovery_failures']+=1;errors[d.get('error','unknown')]+=1
        for p in folder.glob('*_generation.json'):
            g=read(p);stage=p.name.removesuffix('_generation.json');counts[stage+'_calls']+=1
            if g.get('reached_output_limit'):counts[stage+'_reached_output_limit']+=1
            if stage=='draft' and candidate_from_generation(g)['status']!='ok':counts['draft_unusable']+=1
        if not result.get('selected_records'):counts['empty_selected_records']+=1
    write(out/'component_metrics.json',dict(counts=dict(counts),discovery_errors=dict(errors),answers_used_for_validation=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker','score']);p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int)
    a=p.parse_args();{'run':run,'worker':lambda out:worker(out,a.rank),'score':score}[a.action](a.output.resolve())
