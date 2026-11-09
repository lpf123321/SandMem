"""Frozen Spatial baseline evaluation, checkpointed only at video boundaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import subprocess

ROOT = Path(os.environ.get('SANDMEM_ROOT', Path.cwd())).resolve()
sys.path.insert(0, str(ROOT))
from spatial_features import sample_frames, encode_chunk
from official_components import attach_hermes
from causal_adapter import SpatialCausalMem


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--method', choices=['causalmem', 'hermes'], required=True)
    parser.add_argument('--max-seconds', type=int, default=42000)
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    manifest = json.loads((args.run / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(args.run / name) != expected:
            raise ValueError('frozen run file changed: ' + name)
    for name, expected in manifest['external_files'].items():
        if digest(name) != expected:
            raise ValueError('frozen dependency changed: ' + name)
    for repo, expected in manifest['upstream_commits'].items():
        if subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip() != expected:
            raise ValueError('upstream revision changed: ' + repo)
    if args.method != config['method']:
        raise ValueError('method/config mismatch')
    inputs = [json.loads(line) for line in (args.run / 'inputs.jsonl').read_text().splitlines()]
    if any('ground_truth' in row for row in inputs):
        raise ValueError('labels in inference manifest')
    grouped = {}
    for row in inputs:
        grouped.setdefault((row['dataset'], row['scene_name']), []).append(row)
    import torch
    from sandbox_vlm.backbones.spatial_model import Spatial135K
    torch.manual_seed(0)
    torch.set_num_threads(1)
    started = time.perf_counter()
    runtime = Spatial135K(ROOT / 'work/spatial-mllm-official')
    out = args.run / 'videos'
    out.mkdir(exist_ok=True)
    status = dict(method=args.method, status='running', job_id=os.environ.get('SLURM_JOB_ID'),
                  hardware=torch.cuda.get_device_name(0), run_manifest_sha256=digest(args.run / 'manifest.json'))
    new_videos = 0
    consecutive_video_failures = 0
    for (dataset, scene), questions in grouped.items():
        target = out / (dataset + '_' + scene + '.json')
        if target.exists():
            continue  # Includes recorded failures: no automatic failed-input retries.
        if time.perf_counter() - started > args.max_seconds:
            status['status'] = 'time_checkpoint'
            break
        feature_dir = args.run / 'features' / (dataset + '_' + scene)
        feature_dir.mkdir(parents=True, exist_ok=False)
        result = dict(dataset=dataset, scene_name=scene, answers=[], encoding=[],
                      input_question_ids=[question['id'] for question in questions])
        video_seed = int(hashlib.sha256((dataset + '/' + scene).encode()).hexdigest()[:8], 16) % (2**31)
        torch.manual_seed(video_seed)
        result['seed'] = video_seed
        controller = None
        phase = 'video_decode'
        try:
            path = args.root / 'data/videos' / dataset / (scene + '.mp4')
            video, indices, original_fps = sample_frames(path, config['sample_fps'])
            result.update(source_video_sha256=digest(path), source_frame_ids=indices,
                          source_video_fps=original_fps, sample_fps=config['sample_fps'])
            if args.method == 'hermes':
                controller, sources = attach_hermes(runtime, args.root / 'repos/HERMES',
                    kv_size=config['kv_size'], sample_fps=config['sample_fps'])
                controller.clear_cache()
                controller.encode_init_prompt()
            else:
                controller = SpatialCausalMem(runtime, args.root / 'repos/CausalMem',
                    budget=config['feature_budget'], sample_fps=config['sample_fps'])
                sources = [controller.source_receipt]
            result['official_sources'] = sources
            phase = 'visual_encode_and_memory'
            for start in range(0, len(indices), config['encode_chunk_frames']):
                selected = indices[start:start + config['encode_chunk_frames']]
                features, receipt, _ = encode_chunk(runtime, video.get_batch(selected).asnumpy(),
                    selected, sample_fps=config['sample_fps'])
                tensor_path = feature_dir / f'{start:06d}.pt'
                torch.save(features.cpu(), tensor_path)
                receipt.update(feature_path=str(tensor_path), feature_sha256=digest(tensor_path))
                before = time.perf_counter()
                if args.method == 'hermes':
                    controller.encode_features(features, receipt['video_grid_thw'], len(selected))
                    controller.predict_and_compress()
                    receipt['post_memory_kv_lengths'] = controller._get_cache_seq_len_per_layer()
                    receipt['pseudo_attention_queries'] = 3
                else:
                    controller.add(features, receipt['video_grid_thw'])
                    receipt['post_memory_visual_tokens'] = len(controller.cache.mem1_buffer)
                receipt['memory_seconds'] = time.perf_counter() - before
                result['encoding'].append(receipt)
            phase = 'question_answering'
            for question in questions:
                qa_started = time.perf_counter()
                try:
                    if args.method == 'hermes':
                        prompt = controller.get_prompt(question['formatted_question'])
                        generated_ids = []
                        def observe_greedy(module, inputs, output):
                            generated_ids.append(int(torch.topk(output[0, -1], 1).indices[0]))
                        handle = controller.lm_head.register_forward_hook(observe_greedy)
                        try:
                            raw = controller.question_answering(dict(question=question['question'], prompt=prompt),
                                max_new_tokens=config['max_new_tokens'], temperature=0, repetition_penalty=1.0)
                        finally:
                            handle.remove()
                        text_ids = runtime.processor.tokenizer(prompt).input_ids
                        answer = dict(raw_output=raw, actual_text_prompt=prompt,
                                      input_text_token_ids=text_ids, input_text_tokens=len(text_ids),
                                      output_token_ids=generated_ids, output_tokens=len(generated_ids),
                                      kv_lengths=controller._get_cache_seq_len_per_layer())
                    else:
                        answer = controller.answer(question['formatted_question'],
                                                   max_new_tokens=config['max_new_tokens'])
                    answer.update(question_id=question['id'], status='generated',
                                  qa_seconds=time.perf_counter() - qa_started)
                except Exception:
                    answer = dict(question_id=question['id'], status='answer_failed', raw_output='',
                                  failure=traceback.format_exc(), qa_seconds=time.perf_counter() - qa_started)
                    if args.method == 'hermes':
                        result['answers'].append(answer)
                        raise  # A failed mutable KV state must not affect later questions.
                result['answers'].append(answer)
            result['status'] = 'video_completed'
        except Exception:
            result.update(status='video_failed', failure=traceback.format_exc(), failure_phase=phase)
            attempted = {answer['question_id'] for answer in result['answers']}
            result['answers'].extend(dict(question_id=question['id'], status='upstream_failed', raw_output='')
                for question in questions if question['id'] not in attempted)
        finally:
            if args.method == 'hermes' and controller is not None:
                controller._clear_forward_hooks()
                controller.clear_cache()
            del controller
            torch.cuda.empty_cache()
        result.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2))
        temporary.rename(target)
        new_videos += 1
        print(dataset, scene, result['status'], len(result['answers']), flush=True)
        consecutive_video_failures = consecutive_video_failures + 1 if result['status'] == 'video_failed' else 0
        if consecutive_video_failures >= 2:
            status['status'] = 'stopped_repeated_engineering_failure'
            break
    else:
        status['status'] = 'all_videos_attempted'
    status.update(seconds=time.perf_counter() - started, new_videos=new_videos,
                  completed_video_files=len(list(out.glob('*.json'))))
    status_path = args.run / ('job_' + str(os.environ.get('SLURM_JOB_ID', 'local')) + '.json')
    status_path.write_text(json.dumps(status, indent=2))
    if status['status'] == 'all_videos_attempted':
        predictions = [answer for path in sorted(out.glob('*.json'))
                       for answer in json.loads(path.read_text())['answers']]
        (args.run / 'predictions.jsonl').write_text(''.join(json.dumps(answer) + '\n' for answer in predictions))
        from score import score
        (args.run / 'scores.json').write_text(json.dumps(score(args.root, predictions), indent=2))


if __name__ == '__main__':
    main()
