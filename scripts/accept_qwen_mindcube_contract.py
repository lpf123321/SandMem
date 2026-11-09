"""One fixed component contract gate. No answer keys, retries or scores."""
import gc, gzip, time
from pathlib import Path
from run_qwen_half_bench import read, write, sha


def run_component_gate(out):
    import torch
    from PIL import Image
    from sandbox_vlm.backbones.qwen_mindcube_contract import QwenContractBundle, visible_choices
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.two_pass import candidate_from_generation
    from qwen_d116_mindcube_views import InputArchive, discovery_with_failure_receipt
    plan=read(out/'component_gate_plan.json');dest=out/'component_gate';dest.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(0)
    bundle=QwenContractBundle(read(out/'manifest.json')['config'])
    started=time.time();rows=[]
    try:
        for i,case in enumerate(plan['cases']):
            folder=dest/f'{i:02d}_{case["kind"]}';folder.mkdir()
            for path,h in case['file_sha256'].items():assert sha(path)==h,path
            archive=InputArchive(folder);archive.stage=case['kind'];bundle.runtime.transport_processor.audit_sink=archive
            if case['kind']=='discovery':
                result=discovery_with_failure_receipt(bundle,Path(case['path']),case['frame_id'],case['question'])
                write(folder/'generation.json',result)
                passed=result['status']=='ok' and result['output_tokens']<=512
            else:
                messages=read(case['messages']);images=[]
                for path in case['image_paths']:
                    with Image.open(path) as im:images.append(im.convert('RGB'))
                for m in messages:
                    for block in m['content']:
                        if block['type']=='image':
                            idx=block.pop('evidence_image_index');assert block.pop('pixel_sha256')==pixel_sha256(images[idx])
                            block['image']=images[idx]
                result=bundle.answerer.answer(messages,images,QABudget(**case['limits']))
                write(folder/'generation.json',result)
                parsed=candidate_from_generation(result)
                passed=parsed['status']=='ok' and parsed['answer'] in visible_choices(messages)
                for image in images:image.close()
            rows.append(dict(kind=case['kind'],old_failure=case['old_failure'],question_id=case['question_id'],
                passed=passed,status=result['status'],output_tokens=result.get('output_tokens'),error=result.get('error')))
        write(dest/'acceptance.json',dict(status='passed' if all(r['passed'] for r in rows) else 'failed',
            cases=rows,model_calls=len(rows),answers_read=False,retries=0,seconds=time.time()-started))
        if not all(r['passed'] for r in rows):raise RuntimeError('fixed component output-contract gate failed; formal run not started')
    finally:
        bundle.runtime.transport_processor.audit_sink=None
        del bundle
        gc.collect();torch.cuda.empty_cache()
