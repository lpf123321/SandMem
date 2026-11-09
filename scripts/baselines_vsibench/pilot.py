"""Input-fixed wiring acceptance. This program never reads benchmark answers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spatial_features import sample_frames, encode_chunk, check_native_equivalence
from official_components import attach_hermes
from causal_adapter import SpatialCausalMem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['hermes', 'causalmem'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from sandbox_vlm.backbones.spatial_model import Spatial135K, MODEL_ID, REVISION
    torch.manual_seed(0)
    torch.set_num_threads(1)
    questions = json.loads((args.root / 'reports/pilot_inputs.json').read_text())
    if any('ground_truth' in question for question in questions):
        raise ValueError('labels must not enter inference manifest')
    started = time.perf_counter()
    result = dict(status='running', model_id=MODEL_ID, revision=REVISION,
        methods=args.methods, purpose='wiring_acceptance_only', model_generations=0,
        hardware=dict(name=torch.cuda.get_device_name(0),
                      total_memory=torch.cuda.get_device_properties(0).total_memory))
    try:
        runtime = Spatial135K(ROOT / 'work/spatial-mllm-official')
        path = ROOT / 'work/spatial-mllm-official/assets/arkitscenes_41069025.mp4'
        video, indices, source_fps = sample_frames(path, 1)
        result.update(video_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      source_frame_ids=indices, source_fps=source_fps, sampling_fps=1)
        # Native-forward equivalence uses the first observed chunk of this fixed video.
        first = indices[:16]
        result['equivalence_input'], result['equivalence'] = check_native_equivalence(
            runtime, video.get_batch(first).asnumpy(), first, sample_fps=1)
        cached = []
        for start in range(0, len(indices), 16):
            source_ids = indices[start:start + 16]
            features, receipt, _ = encode_chunk(runtime, video.get_batch(source_ids).asnumpy(),
                                               source_ids, sample_fps=1)
            feature_path = args.output / f'features_{start:06d}.pt'
            torch.save(features.cpu(), feature_path)
            receipt['feature_file'] = feature_path.name
            receipt['feature_sha256'] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
            (args.output / f'input_{start:06d}.json').write_text(json.dumps(receipt, indent=2))
            cached.append((feature_path, receipt))
            print('encoded', start, '/', len(indices), receipt['fused_shape'], flush=True)
        for method in args.methods:
            method_started = time.perf_counter()
            if method == 'hermes':
                controller, sources = attach_hermes(runtime, args.root / 'repos/HERMES', kv_size=6000, sample_fps=1)
                controller.clear_cache()
                controller.encode_init_prompt()
                for feature_path, receipt in cached:
                    features = torch.load(feature_path, weights_only=True).to(runtime.device)
                    controller.encode_features(features, receipt['video_grid_thw'], len(receipt['source_frame_ids']))
                    controller.predict_and_compress()
                answers = []
                for question in questions:
                    prompt = controller.get_prompt(question['formatted_question'])
                    entry = dict(question_id=question['id'], prompt=prompt)
                    entry['raw_output'] = controller.question_answering(
                        dict(question=question['question'], prompt=prompt),
                        max_new_tokens=16, temperature=0, repetition_penalty=1.0)
                    result['model_generations'] += 1
                    answers.append(entry)
                controller._clear_forward_hooks()
                controller.clear_cache()
            elif method == 'causalmem':
                controller = SpatialCausalMem(runtime, args.root / 'repos/CausalMem', sample_fps=1)
                sources = [controller.source_receipt]
                for feature_path, receipt in cached:
                    features = torch.load(feature_path, weights_only=True).to(runtime.device)
                    controller.add(features, receipt['video_grid_thw'])
                answers = []
                for question in questions:
                    answers.append(dict(question_id=question['id'], **controller.answer(question['formatted_question'])))
                    result['model_generations'] += 1
            else:
                raise ValueError('unimplemented method: ' + method)
            (args.output / (method + '.json')).write_text(json.dumps(dict(
                sources=sources, answers=answers, seconds=time.perf_counter() - method_started), indent=2))
            print(method, 'finished', len(answers), flush=True)
        result['status'] = 'wiring_completed_not_scored'
    except Exception:
        result.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        (args.output / 'completion.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
