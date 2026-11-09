"""Freeze one numbered-view repair; exact previously selected 527 inputs."""
import gzip,hashlib,json,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_qwen_half_bench import read,write,sha
PARENT=ROOT/'work/qwen_d116_half_20261107_03'


def input_hash(item):return hashlib.sha256(json.dumps(item,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def prepare(out):
    if out.exists():raise FileExistsError(out)
    parent=read(PARENT/'manifest.json')
    for p,h in parent['source_sha256'].items():assert sha(ROOT/p)==h,p
    original=[r for r in parent['inputs'] if r['dataset']=='mindcube'];assert len(original)==527
    gates=[min([r for r in original if len(r['images'])==n],key=input_hash) for n in (3,4)]
    gate_ids=[r['id'] for r in gates];ordered=gates+[r for r in original if r['id'] not in gate_ids]
    out.mkdir(parents=True);runtime=out/'mindcube_viewfix_runtime';runtime.mkdir()
    m=deepcopy(parent);m.update(scope='MindCube_numbered_view_adapter_correction',inputs=ordered,
        videos=[v for v in parent['videos'] if v['dataset']=='mindcube'],gate_question_ids=gate_ids,
        question_counts={'MindCube-tinybench':527},parent_manifest_sha256=sha(PARENT/'manifest.json'),
        parent_run=str(PARENT),cold_perception_rerun=True,prediction_reuse=False,
        adapter='explicit_numbered_views_v1',repair_scope=['all original numbered input views in QA',
            'explicit image N to frame ID N-1 mapping; supplemental sandbox outside image numbering',
            'supplied view order rather than invented video timestamps'],
        geometry_note='unchanged VGGT/SAM algorithms; cold geometry uses the corrected complete supplied view set where actual masks exist',
        method_core_unchanged=True,config_unchanged_from_parent=True,
        score_policy='One complete 527-question measurement after input-only gate; official scorer, invalid wrong; no tuning or retries',
        gate_selection='minimum whitelisted input SHA256 among 3-view and 4-view inputs; no answer or score criteria',
        hardware_policy='A100 only; up to two independent workers if measured previous peak plus fixed headroom fits')
    assert m['config']==parent['config'] and {r['id'] for r in ordered}=={r['id'] for r in original}
    sources=dict(parent['source_sha256'])
    for p in ['scripts/prepare_qwen_mindcube_viewfix.py','scripts/run_qwen_mindcube_viewfix.py','scripts/run_qwen_mindcube_viewfix.sbatch',
        'scripts/qwen_d116_mindcube_views.py','sandbox_vlm/data/numbered_views.py','tests/test_mindcube_numbered_views.py']:
        sources[p]=sha(ROOT/p)
    m['source_sha256']=sources
    for p,h in sources.items():
        dst=out/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes());assert sha(dst)==h
    (out/'native_sam').mkdir();(out/'native_sam/_C.so').write_bytes((PARENT/'native_sam/_C.so').read_bytes())
    (out/'mind_references.json').write_bytes((PARENT/'mind_references.json').read_bytes())
    m['reference_sha256']={str(out/'mind_references.json'):sha(out/'mind_references.json')}
    for item in ordered:
        folder=out/('question_'+item['id']);folder.mkdir()
        write(folder/'reservation.json',dict(owner='mindcube_viewfix_runtime',question_id=item['id']))
        v=next(v for v in m['videos'] if v['scene_name']==item['scene_name'])
        rows=read(v['source_video'])['source_images']
        assert len(rows)==len(item['images']) and [r['source_index'] for r in v['frame_mapping']]==list(range(len(rows)))
        assert all(Path(row['path']).as_posix().endswith('/'+name) for row,name in zip(rows,item['images']))
    write(out/'manifest.json',m);write(out/'freeze.json',dict(manifest_sha256=sha(out/'manifest.json'),source_files=len(sources),model_calls=0))
    prior=[read(PARENT/('question_'+r['id'])/'result.json') for r in original]
    write(runtime/'authorization.json',dict(manifest_sha256=sha(out/'manifest.json'),launcher_sha256=sha(ROOT/'scripts/run_qwen_mindcube_viewfix.py'),
        sbatch_sha256=sha(ROOT/'scripts/run_qwen_mindcube_viewfix.sbatch'),
        measured_MindCube_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in prior),
        user_authorized='fix complete original views and image numbering, rerun same sample independently',
        cold_full_model=True,no_prompt_search=True,old_predictions_reused=False))
    preflight(out,m,original)
    print(json.dumps(dict(output=str(out),questions=527,gate_ids=gate_ids,source_files=len(sources),model_calls=0)),flush=True)


def preflight(out,m,original):
    from transformers import AutoProcessor,AutoConfig
    from PIL import Image
    import torch
    from sandbox_vlm.data.video import FrameMapping,FrameRef
    from sandbox_vlm.data.view_sequence import VideoFrameReader
    from sandbox_vlm.data.numbered_views import bind_messages,full_view_ids
    from sandbox_vlm.qa.d116_full import joint_messages
    from sandbox_vlm.qa.map_only import indexed_messages
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.budget import check_budget,QABudget
    model=m['config']['qa']['mllm'];processor=AutoProcessor.from_pretrained(model['model_id'],revision=model['revision'],local_files_only=True)
    cfg=AutoConfig.from_pretrained(model['model_id'],revision=model['revision'],local_files_only=True)
    examples=[]
    for n in (2,3,4):
        item=min([r for r in original if len(r['images'])==n],key=input_hash);prior=PARENT/('question_'+item['id'])
        v=next(v for v in m['videos'] if v['scene_name']==item['scene_name'])
        mapping=FrameMapping(video_id='mindcube/'+v['scene_name'],fps=1.,sample_fps=1.,clip_len=16,overlap=4,frames=[FrameRef(**r) for r in v['frame_mapping']])
        ids=full_view_ids(item,mapping);reader=VideoFrameReader(v['source_video'],mapping);reader.allowed_through=n-1
        old=read(prior/'result.json');sparse=old['sparse']
        images=[reader.read(fid,max_side=m['config']['qa']['sparse_image_side'] if sparse else m['config']['qa']['image_side'])[0] for fid in ids]
        with Image.open(prior/'sandbox.png') as im:diagram=im.convert('RGB')
        messages,joint=joint_messages(item['formatted_question'],sparse,ids,read(prior/'payload.json'),images,diagram)
        messages,contract=bind_messages(messages,ids,has_sandbox=True)
        tensors=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
        grid=tensors['image_grid_thw'].tolist();assert len(grid)==n+1
        visual=int((tensors['input_ids']==cfg.image_token_id).sum())
        limits=deepcopy(m['config']['qa']['budget']);limits.update(max_images=33,max_text_tokens=32768,max_image_side=max(768,*[max(i.size) for i in joint]),
            max_output_tokens=m['config']['qa']['sparse_max_output_tokens'] if sparse else limits['max_output_tokens'])
        patch=processor.image_processor.patch_size
        budget=check_budget(QABudget(**limits),image_sizes_hw=[[i.height,i.width] for i in joint],processed_image_sizes_hw=[[r[1]*patch,r[2]*patch] for r in grid],
            input_tokens=tensors['input_ids'].shape[-1],visual_tokens=visual,context_limit=cfg.text_config.max_position_embeddings)
        assert budget['status']=='ok',budget
        dest=out/'cpu_preflight'/f'views_{n}';dest.mkdir(parents=True)
        write(dest/'messages.json',indexed_messages(messages,joint,[pixel_sha256(i) for i in joint]))
        with gzip.open(dest/'tensors.pt.gz','wb',compresslevel=1) as f:torch.save(dict(tensors),f)
        write(dest/'contract.json',dict(contract=contract,image_grid_thw=grid,budget=budget,question_id=item['id'],
            warning='CPU contract illustration uses preserved old sandbox/payload; production regenerates complete perception and geometry',model_calls=0))
        examples.append(dict(original_views=n,question_id=item['id'],model_images=n+1,visual_tokens=visual))
        for i in joint:i.close()
    write(out/'cpu_preflight.json',dict(status='passed',metadata_records=527,original_question_and_view_lists_preserved=True,
        native_processor_examples=examples,model_calls=0,answers_used=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output.resolve())
