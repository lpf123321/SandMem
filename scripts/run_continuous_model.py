"""Executable corrected model: sampled video -> real 3D + frames -> real QA.

--max-frames bounds a prefix engineering check; omit it to process the entire
sampled video. No benchmark answers, predictions or old identity banks are read.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def execute(args):
    import cv2
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sandbox_vlm.continuous_model import ContinuousSandMem
    from sandbox_vlm.perception.continuous_backend import ContinuousPerception
    from sandbox_vlm.perception.continuous_discovery import ContinuousDiscovery
    from sandbox_vlm.qa.model import QwenAnswerer
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.artifact import pixel_sha256

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        config = json.loads(args.config.read_text())
        from sandbox_vlm.continuous_preflight import sample_plan, validate_start, apply_input_profile
        input_profile = json.loads(args.input_profile.read_text()) if args.input_profile else None
        if input_profile is not None:
            config = apply_input_profile(config, input_profile)
            args.sparse = input_profile['sparse']
        # Metadata and protocol checks happen before any checkpoint load/CUDA.
        metadata_capture = cv2.VideoCapture(str(args.video))
        fps = metadata_capture.get(cv2.CAP_PROP_FPS)
        total = int(metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata_capture.release()
        plan = sample_plan(fps, total, config['sampling']['fps'], args.max_frames)
        preflight = validate_start(config, plan=plan, purpose=args.purpose,
            protocol=args.protocol, answer_format=args.answer_format, input_profile=input_profile)
        write(root / 'preflight.json', preflight)
        if preflight['status'] != 'passed':
            raise ValueError('preflight blocked: ' + '; '.join(preflight['problems']))
        expected_sha = config['geometry'].get('checkpoint_sha256')
        if expected_sha:
            digest=hashlib.sha256()
            with args.checkpoint.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):digest.update(block)
            if digest.hexdigest()!=expected_sha:
                raise ValueError('StreamVGGT checkpoint SHA256 differs from configured official weights')
        backbone = yaml.safe_load(args.backbone_config.read_text())
        if backbone['perception']['mllm']['model_id'] != backbone['qa']['mllm']['model_id']:
            raise ValueError('discovery and QA backbone mismatch')
        # This concrete transport is the existing D116 Qwen reference. Spatial
        # production migration remains explicit separate work, never relabelled.
        if backbone['qa']['mllm']['model_id'] != 'Qwen/Qwen3-VL-2B-Instruct':
            raise ValueError('this reference runner requires an explicit Spatial transport for other backbones')
        upstream = args.upstream.resolve()
        source = json.loads((upstream / 'source_manifest.json').read_text())
        assert source['commit'] == config['geometry']['upstream_commit']
        for row in source['files']:
            assert hashlib.sha256((upstream / 'upstream' / row['path']).read_bytes()).hexdigest() == row['sha256']
        sys.path.insert(0, str(upstream / 'upstream' / 'src'))
        from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.utils.load_fn import load_and_preprocess_images
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise ValueError('cannot open source video')
        fps, total = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps < config['sampling']['fps']:
            raise ValueError('unsupported sampling/source rate')
        ids = plan['frame_ids']
        discovery_cap = config['execution']['max_discovery_frames']
        if 'discovery_schedule' in config['execution']:
            schedule = config['execution']['discovery_schedule']
            discovery_cap = min(schedule['maximum'], max(schedule['minimum'],
                int(np.ceil(len(ids) / config['sampling']['fps'] / 60 * schedule['per_minute']))))
        discovery_indices = set(np.linspace(0, len(ids) - 1,
                                            min(len(ids), discovery_cap)).astype(int).tolist())
        write(root / 'input_plan.json', dict(video=str(args.video.resolve()), source_fps=fps,
            source_frame_count=total, frame_ids=ids, discovery_indices=sorted(discovery_indices),
            question=args.question, sparse=args.sparse, config=config,
            full_video=plan['full_video'], input_profile=input_profile,
            backbone=backbone, backbone_status='historical_Qwen_reference_new_continuous_method'))
        counters = dict(geometry=0, discovery=0, sam_encoder=0, sam_prompt=0, qa=0)
        bounds = dict(geometry=len(ids), discovery=len(discovery_indices), sam_encoder=len(ids),
                      sam_prompt=config['execution']['max_sam_prompts'], qa=2)

        def attempt(kind):
            if counters[kind] >= bounds[kind]:
                raise RuntimeError('sealed call budget exceeded: ' + kind)
            counters[kind] += 1
            with (root / 'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(kind=kind, ordinal=counters[kind])) + '\n')

        model = ContinuousSandMem(str(args.video.resolve()), config)
        geo = StreamVGGT()
        weights = load_file(str(args.checkpoint), device='cpu')
        geo.load_state_dict(weights, strict=True)
        del weights
        geo = geo.cuda().eval().requires_grad_(False)
        sam_config = backbone['perception']['sam']
        sam_path = hf_hub_download(sam_config['checkpoint'], sam_config['checkpoint_file'],
                                   revision=sam_config['revision'], local_files_only=True)
        sam = SAM2ImagePredictor(build_sam2(sam_config['config'], sam_path, device='cuda').eval().requires_grad_(False))
        discovery_config = dict(backbone['perception']['mllm'], max_new_tokens=config['execution']['discovery_output_tokens'])
        discovery = ContinuousDiscovery(discovery_config)
        answerer = QwenAnswerer.from_discovery(backbone['qa']['mllm'], discovery)
        perception = ContinuousPerception(config, ids, discovery=discovery, sam_predictor=sam,
            stream_model=geo, native_loader=load_and_preprocess_images,
            pose_decoder=pose_encoding_to_extri_intri, attempt=attempt)
        frames = root / 'frames'
        frames.mkdir()
        for i, fid in enumerate(ids):
            if time.perf_counter() - start > config['execution']['max_seconds']:
                raise TimeoutError('bounded integration execution timeout')
            capture.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('source decode failed at ' + str(fid))
            path = frames / f'{fid:08d}.png'
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)
            current, receipt = perception.frame(path, fid, i, discover=i in discovery_indices, available_through=fid)
            receipt['memory'] = model.ingest(**current)
            write(root / f'frame_{i:05d}.json', receipt)
            del current
            print(json.dumps(dict(frame=fid, instances=len(model.instances), calls=counters)), flush=True)
        capture.release()
        perception.close()
        del perception, geo, sam
        gc.collect()
        torch.cuda.empty_cache()
        model.write_snapshot(root / 'memory.json')

        def load_image(fid):
            image = Image.open(frames / f'{fid:08d}.png').convert('RGB')
            image.thumbnail((config['qa']['real_image_side'],) * 2)
            return image

        def audit(name, value):
            if name.endswith('_input'):
                images = value.pop('images')
                paths = []
                for i, image in enumerate(images):
                    path = root / f'{name}_image_{i:02d}.png'
                    image.save(path)
                    paths.append(dict(path=str(path), pixel_sha256=pixel_sha256(image)))
                serial = []
                for message in value['messages']:
                    content = []
                    for block in message['content']:
                        if block['type'] == 'image':
                            position = next(i for i, image in enumerate(images) if image is block['image'])
                            content.append(dict(type='image', evidence_image_index=position, **paths[position]))
                        else:
                            content.append(dict(block))
                    serial.append(dict(role=message['role'], content=content))
                value = dict(messages=serial, images=paths, limits=value['limits'])
            write(root / (name + '.json'), value)

        class CountedAnswerer:
            def answer(self, *a, **kw):
                if time.perf_counter() - start > config['execution']['max_seconds']:
                    raise TimeoutError('execution deadline reached before QA')
                attempt('qa')
                return answerer.answer(*a, **kw)

        limits = QABudget(**config['qa']['budget'])
        if not args.sparse:
            from dataclasses import replace
            limits = replace(limits, max_output_tokens=64)
        def validate_answer(value):
            import re, math
            if not isinstance(value,str) or not value.strip():return False
            if args.answer_format=='choice':return bool(re.fullmatch('[A-D]',value.strip()))
            if args.answer_format=='number':
                try:return math.isfinite(float(value))
                except ValueError:return False
            return True
        result = model.answer(args.question, maximum=config['qa']['real_frame_budget'],
            available_through=ids[-1], sparse=args.sparse, load_image=load_image,
            answerer=CountedAnswerer(), limits=limits, audit=audit, answer_validator=validate_answer)
        write(root / 'completion.json', dict(status='completed', answer=result, calls=counters,
            wall_seconds=time.perf_counter() - start, structure_bytes=model.structure_bytes(),
            peak_cuda_allocated=torch.cuda.max_memory_allocated(), peak_cuda_reserved=torch.cuda.max_memory_reserved(),
            backbone_status='Qwen_reference_not_Spatial', prefix_only=not plan['full_video'],
            no_benchmark_score=True, device=torch.cuda.get_device_name(0)))
    except BaseException as exc:
        if hasattr(exc, 'receipt'):
            write(root / 'perception_failure.json', exc.receipt)
        write(root / 'failure.json', dict(type=type(exc).__name__, message=str(exc),
                                         wall_seconds=time.perf_counter() - start))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--backbone-config', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--sparse', action='store_true')
    p.add_argument('--answer-format', choices=['text','choice','number'], required=True)
    p.add_argument('--purpose', choices=['wiring','development','evaluation'], default='development')
    p.add_argument('--protocol', choices=['offline','streaming'], default='offline')
    p.add_argument('--input-profile', type=Path)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError('positive frame prefix limit required')
    if args.preflight_only:
        import cv2
        from sandbox_vlm.continuous_preflight import sample_plan, validate_start, apply_input_profile
        config=json.loads(args.config.read_text())
        profile=json.loads(args.input_profile.read_text()) if args.input_profile else None
        if profile:config=apply_input_profile(config,profile)
        cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        plan=sample_plan(fps,frames,config['sampling']['fps'],args.max_frames)
        result=validate_start(config,plan=plan,purpose=args.purpose,protocol=args.protocol,
                              answer_format=args.answer_format,input_profile=profile)
        args.output.mkdir(parents=True,exist_ok=False)
        write(args.output/'preflight.json',result)
        print(json.dumps(result))
        sys.exit(0 if result['status']=='passed' else 2)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('real-model execution requires an allocated Slurm GPU')
    execute(args)


if __name__ == '__main__':
    main()
