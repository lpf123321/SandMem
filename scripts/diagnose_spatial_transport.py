"""Fixed, six-request comparison of official and adapted Spatial transports.

No prompt search, answers, scores, new video selection, or method changes.
The native-video control repeats ONE observation as temporal padding only.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def equal_inputs(a, b):
    import torch
    def same(x, y):
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and torch.equal(x, y)
        if isinstance(x, list):
            return isinstance(y, list) and len(x) == len(y) and all(same(i,j) for i,j in zip(x,y))
        return x == y
    return set(a) == set(b) and all(same(a[k],b[k]) for k in a)


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from sandbox_vlm.backbones.spatial_transport import SpatialBackbone, spatial_config
    from sandbox_vlm.backbones.spatial_inputs import input_record
    from sandbox_vlm.perception.discovery import discovery_prompt, discovery_generation_kwargs, parse_discovery
    from sandbox_vlm.qa.artifact import pixel_sha256
    config = spatial_config(json.loads((ROOT/'configs/reference_qwen.json').read_text()))
    settings = discovery_generation_kwargs(config['perception']['mllm'])
    # All selections and all six calls are fixed before any weights are loaded.
    selected = []
    for label, folder in [('sparse','gpu_sparse_02'),('dense','gpu_dense_01')]:
        source = json.loads((args.reference/folder/'source.json').read_text())
        snapshot_path = Path(source['input_path'])
        if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != source['input_sha256']:
            raise ValueError('source input receipt changed')
        snapshot = json.loads(snapshot_path.read_text())
        index = next(i for i, s in enumerate(source['media']) if s['kind'] == 'source_frame')
        fid = source['media'][index]['frame_ids'][0]
        watermark = int(snapshot_path.parent.name.removeprefix('query_'))
        if fid > watermark:
            raise ValueError('future observation')
        selected.append(dict(label=label,frame_id=fid,watermark=watermark,
            path=snapshot['paths'][index],pixel_sha256=snapshot['pixel_sha256'][index],
            question=source['original_question'],
            prompt=discovery_prompt(fid,question=source['original_question'])))
    write(args.output/'plan.json', dict(request_cap=6,generation_kwargs=settings,
        selection='first actual source frame from each of the two previously fixed structure cases',
        prompt_changed=False,ground_truth_read=False,formal_scores=False,selected=selected,
        native_video_control='one source observation repeated in two temporal slots; diagnostic only'))
    official_file = args.official_root/'src/inference.py'
    tree = ast.parse(official_file.read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_spatial_mllm_inputs')
    namespace = dict(torch=torch,np=np,Image=Image)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(official_file),'exec'),namespace)
    official_prepare = namespace['prepare_spatial_mllm_inputs']
    write(args.output/'official_source.json',dict(path=str(official_file),
        sha256=hashlib.sha256(official_file.read_bytes()).hexdigest(),
        image_rank_fix='CHW -> TCHW with two identical slots; only required static-image compatibility fix'))
    bundle = SpatialBackbone(config,args.official_root)
    runtime, processor = bundle.runtime, bundle.runtime.processor
    write(args.output/'hardware.json',dict(name=torch.cuda.get_device_name(0),
        total_memory=torch.cuda.get_device_properties(0).total_memory))
    outcomes = []
    for selection in selected:
        with Image.open(selection['path']) as opened:
            image = opened.convert('RGB')
        if pixel_sha256(image) != selection['pixel_sha256']:
            raise ValueError('source pixels changed')
        def prepare(message, static_fix=False):
            text = processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
            imgs,vids = process_vision_info(message)
            batch = processor(text=[text],images=imgs or None,videos=vids or None,
                padding=True,padding_side='left',return_tensors='pt')
            batch = official_prepare(batch,vids,imgs)
            if static_fix:
                batch['image_tchw'] = [v.unsqueeze(0).repeat(2,1,1,1) for v in batch['image_tchw']]
            return dict(batch),text
        message = [dict(role='user',content=[dict(type='image',image=image),
                                            dict(type='text',text=selection['prompt'])])]
        direct,text = prepare(message,static_fix=True)
        adapted = bundle.answerer.processor.apply_chat_template(message,tokenize=True,
            add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True)
        if not equal_inputs(direct,adapted):
            raise ValueError('official/direct and adapter input tensors differ')
        # Same physical observation, not two new observations or a new scene.
        video_message = [dict(role='user',content=[dict(type='video',video=[image,image]),
                                                  dict(type='text',text=selection['prompt'])])]
        video,video_text = prepare(video_message)
        case = args.output/selection['label']; case.mkdir()
        write(case/'source.json',selection)
        write(case/'tensor_comparison.json',dict(official_adapter_exact=True,
            image_video_pixels_equal=torch.equal(direct['pixel_values'],video['pixel_values_videos']),
            image_video_spatial_equal=torch.equal(direct['image_tchw'][0],video['video_tchw'][0]),
            image_grid=direct['image_grid_thw'].tolist(),video_grid=video['video_grid_thw'].tolist()))
        raw_first = None
        for name,batch,model in [('official_image',direct,runtime.model),
                                 ('adapter_image',adapted,bundle.answerer.model),
                                 ('official_video_control',video,runtime.model)]:
            folder = case/name; folder.mkdir()
            torch.save(dict(batch),folder/'inputs.pt')
            write(folder/'receipt.json',dict(tensors={k:input_record(v) for k,v in batch.items()},
                prompt=text if name!='official_video_control' else video_text))
            with (args.output/'attempts.jsonl').open('a') as f:
                f.write(json.dumps(dict(case=selection['label'],transport=name,model_request=1))+'\n')
            values = runtime.on_device(batch)
            torch.cuda.synchronize(); started=time.perf_counter()
            with runtime.trace_media() as trace,torch.inference_mode():
                generated = model.generate(**values,**settings,return_dict_in_generate=True,output_scores=True)
            torch.cuda.synchronize()
            tokens = generated.sequences[0,values['input_ids'].shape[-1]:]
            first = generated.scores[0].detach().float().cpu()
            raw = processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            result = dict(raw_output=raw,output_ids=tokens.tolist(),output_tokens=len(tokens),
                generation_seconds=time.perf_counter()-started,actual_forward_trace=trace,
                discovery_status='ok',generation_kwargs=settings)
            try:
                result['detections']=[d.to_dict() for d in parse_discovery(raw,selection['frame_id'],(image.height,image.width))]
            except ValueError as exc:
                result.update(discovery_status='perception_failed',error=str(exc))
            if raw_first is not None and name=='adapter_image':
                result.update(official_first_scores_exact=torch.equal(first,raw_first),
                    official_first_scores_max_abs=float((first-raw_first).abs().max()))
            if name=='official_image':
                raw_first=first
            torch.save(first,folder/'first_scores.pt')
            write(folder/'generation.json',result)
            outcomes.append(dict(case=selection['label'],transport=name,
                status=result['discovery_status'],raw_output=raw,
                scores_exact=result.get('official_first_scores_exact')))
            del generated,values
    write(args.output/'completion.json',dict(status='diagnostic_completed',model_requests=len(outcomes),
        outcomes=outcomes,prompt_or_method_changed=False,formal_scores=None,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--official-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except Exception as exc:
        write(args.output/'failure.json',dict(status='failed',error=repr(exc)))
        raise
