"""All observed failures, two independent bounded workers, no answer scoring."""
import argparse, gc, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_qwen_half_bench import read,write,sha


def cpu_failure_replay(out):
    from types import SimpleNamespace
    from lmformatenforcer import JsonSchemaParser,CharacterLevelParserConfig
    from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import ClosedStringJSON
    from sandbox_vlm.backbones.qwen_mindcube_contract import visible_choices
    from sandbox_vlm.backbones.spatial_syntax import discovery_schema
    from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext
    from sandbox_vlm.backbones.spatial_fast_syntax_v2 import FastBudgetJSON
    m=read(out/'manifest.json');plan=read(out/'component_gate_plan.json');rows=[]
    for case in plan['cpu_failure_prefixes']:
        raw=JsonSchemaParser(discovery_schema(case['frame_id']),config=CharacterLevelParserConfig(max_json_array_length=512,max_consecutive_whitespaces=512))
        context=BudgetContext(SimpleNamespace(encode=lambda text,**_:list(text)),100000,case['frame_id'])
        parser=FastBudgetJSON(ClosedStringJSON(raw),context);blocked=None
        for i,c in enumerate(case['raw']):
            allowed=parser.get_allowed_characters()
            if c not in allowed:
                base=parser.base
                blocked=dict(path=case['path'],character_index=i,blocked_character=repr(c),role=base.role,
                    words=base.words,inside_word=base.in_word,escape=base.escape,
                    closure_quote_allowed='"' in allowed,allowed_characters=''.join(sorted(allowed))[:100])
                break
            parser=parser.add_character(c)
        assert blocked is not None,('old nonterminating output not blocked',case['path'])
        assert blocked['allowed_characters'],('grammar dead end',case['path'])
        rows.append(blocked)
    # Full input metadata/visible-choice/source checks, never read reference answers.
    videos={v['scene_name']:v for v in m['videos']};unique={}
    for item in m['inputs']:
        visible_choices([dict(content=[dict(type='text',text='Original question (verbatim):\n'+item['formatted_question'])])])
        v=videos[item['scene_name']];assert sha(v['source_video'])==v['source_sha256']
        original=read(v['source_video'])['source_images']
        assert len(original)==len(item['images'])
        assert [r['frame_id'] for r in v['frame_mapping']]==list(range(len(original)))
        assert [r['source_index'] for r in v['frame_mapping']]==list(range(len(original)))
        for row,name in zip(original,item['images']):
            assert row['path'].endswith('/'+name)
            unique[row['path']]=row['sha256']
    for path,h in unique.items():assert sha(path)==h,path
    write(out/'cpu_failure_replay.json',dict(status='passed',model_calls=0,all_observed_failures=len(rows),
        blocked_before_cap=rows,full_question_metadata=527,unique_original_images_verified=len(unique),answers_read=False))


def component_worker(out,rank):
    import torch
    from PIL import Image
    from sandbox_vlm.backbones.qwen_mindcube_contract_fast import QwenFastClosedContractBundle as QwenClosedContractBundle
    from sandbox_vlm.backbones.qwen_mindcube_contract import visible_choices
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.two_pass import candidate_from_generation
    from qwen_d116_mindcube_views import InputArchive,discovery_with_failure_receipt
    m=read(out/'manifest.json');plan=read(out/'component_gate_plan.json')
    assert sha(out/'component_gate_plan.json')==m['component_gate_plan_sha256']
    for path,h in m['source_sha256'].items():assert sha(ROOT/path)==h,path
    torch.set_num_threads(1);torch.manual_seed(0)
    bundle=QwenClosedContractBundle(m['config']);rows=[];start=time.time();dest=out/'component_gate'
    try:
        for i,case in enumerate(plan['cases']):
            if i%2!=rank:continue
            folder=dest/f'{i:03d}_{case["kind"]}';folder.mkdir(exist_ok=False)
            for path,h in case['file_sha256'].items():assert sha(path)==h,path
            archive=InputArchive(folder);archive.stage=case['kind'];bundle.runtime.transport_processor.audit_sink=archive
            print('COMPONENT_START',i,case['question_id'],flush=True)
            if case.get('equivalent_reuse'):
                reuse=case['equivalent_reuse']
                assert sha(reuse['path'])==reuse['sha256']
                result=read(reuse['path']);passed=result['status'] in ('ok','generated')
                write(folder/'equivalent_reuse.json',dict(**reuse,new_model_calls=0,reason='exact same inputs and allowed-token language; completed component validation only'))
            elif case['kind']=='discovery':
                result=discovery_with_failure_receipt(bundle,Path(case['path']),case['frame_id'],case['question'])
                passed=result['status']=='ok' and not result['reached_output_limit']
            else:
                messages=read(case['messages']);images=[]
                for path in case['image_paths']:
                    with Image.open(path) as im:images.append(im.convert('RGB'))
                for message in messages:
                    for block in message['content']:
                        if block['type']=='image':
                            index=block.pop('evidence_image_index');assert block.pop('pixel_sha256')==pixel_sha256(images[index])
                            block['image']=images[index]
                result=bundle.answerer.answer(messages,images,QABudget(**case['limits']))
                candidate=candidate_from_generation(result)
                passed=candidate['status']=='ok' and candidate['answer'] in visible_choices(messages)
                for image in images:image.close()
            write(folder/'generation.json',result)
            row=dict(index=i,kind=case['kind'],question_id=case['question_id'],old_failure=case['old_failure'],
                passed=passed,new_model_calls=0 if case.get('equivalent_reuse') else 1,status=result['status'],output_tokens=result.get('output_tokens'),error=result.get('error'))
            rows.append(row);print('COMPONENT_DONE',i,passed,flush=True)
            if not passed:raise RuntimeError('observed failure regression failed; formal run blocked')
        write(dest/f'worker_{rank}.json',dict(status='passed',cases=rows,seconds=time.time()-start))
    except Exception:
        write(dest/f'worker_{rank}.json',dict(status='failed',cases=rows,error=traceback.format_exc(),retry=False));raise
    finally:
        bundle.runtime.transport_processor.audit_sink=None;del bundle;gc.collect();torch.cuda.empty_cache()


def run_component_gate(out):
    dest=out/'component_gate';dest.mkdir(exist_ok=False);children=[];logs=[];start=time.time()
    for rank in (0,1):
        log=(dest/f'worker_{rank}.log').open('x');logs.append(log)
        children.append(subprocess.Popen([sys.executable,'-B','-u',__file__,'worker','--output',str(out),'--rank',str(rank)],stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in children]
    for log in logs:log.close()
    rows=[]
    for rank in (0,1):
        p=dest/f'worker_{rank}.json'
        if p.exists():rows.extend(read(p)['cases'])
    planned=read(out/'component_gate_plan.json')['logical_calls']
    passed=not any(codes) and len(rows)==planned and all(r['passed'] for r in rows)
    write(dest/'acceptance.json',dict(status='passed' if passed else 'failed',cases=sorted(rows,key=lambda r:r['index']),
        model_calls=sum(r.get('new_model_calls',1) for r in rows),equivalent_reused_cases=sum(not r.get('new_model_calls',1) for r in rows),planned_calls=planned,workers=2,allocated_GPUs=1,exit_codes=codes,seconds=time.time()-start,answers_read=False,retries=0))
    if not passed:raise RuntimeError('complete fixed failure gate failed; no formal inference')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker']);p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,required=True)
    a=p.parse_args();component_worker(a.output.resolve(),a.rank)
