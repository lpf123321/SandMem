"""Independent SimpleStream recent-4 Spatial135K port; no benchmark labels.

Uses the common 1 FPS source pool of the other Spatial baseline runs. The
last four sampled observations are separate images, as in the upstream
Qwen2.5 implementation. Native Spatial processor and both encoder streams
replace the upstream image-only feature injection. No memory or retrieval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ.get('SANDMEM_ROOT', Path.cwd())).resolve()
sys.path.insert(0, str(ROOT))
from spatial_features import sample_frames


def digest(path):
    with Path(path).open('rb') as stream:
        result = hashlib.sha256()
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    manifest = json.loads((args.run / 'manifest.json').read_text())
    for path, expected in manifest['files'].items():
        if digest(path) != expected:
            raise ValueError('Frozen input/dependency changed: ' + path)
    inputs = [json.loads(line) for line in (args.run / 'inputs.jsonl').read_text().splitlines()]
    if any('ground_truth' in row or 'answer' in row for row in inputs):
        raise ValueError('labels in inference inputs')
    grouped = {}
    for row in inputs:
        grouped.setdefault((row['dataset'], row['scene_name']), []).append(row)
    import torch
    from PIL import Image
    from sandbox_vlm.backbones.spatial_model import Spatial135K
    torch.set_num_threads(1)
    started = time.perf_counter()
    completion = dict(status='starting', job_id=os.environ.get('SLURM_JOB_ID'),
                      manifest_sha256=digest(args.run / 'manifest.json'),
                      hardware=torch.cuda.get_device_name(0), generations=0)
    predictions = []
    try:
        runtime = Spatial135K(ROOT / 'work/spatial-mllm-official')
        for (dataset, scene), rows in grouped.items():
            out = args.run / 'videos' / (dataset + '_' + scene)
            if (out / 'complete.json').exists():
                predictions.extend(json.loads((out / 'complete.json').read_text())['answers'])
                continue
            # Never silently retry partially processed videos.
            out.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(int(hashlib.sha256((dataset + '/' + scene).encode()).hexdigest()[:8], 16) % 2**31)
            path = args.data_root / 'data/videos' / dataset / (scene + '.mp4')
            reader, pool, fps = sample_frames(path, config['sample_fps'])
            selected = pool[-config['recent_frames']:]
            frames = reader.get_batch(selected).asnumpy()
            messages_media = []
            provenance = []
            for source_id, frame in zip(selected, frames):
                target = out / ('source_%08d.png' % source_id)
                Image.fromarray(frame).save(target)
                messages_media.append(dict(type='image', image=str(target.resolve())))
                provenance.append(dict(kind='source_frame', frame_ids=[source_id],
                                       source_video=str(path.resolve())))
            (out / 'sampling.json').write_text(json.dumps(dict(
                source_video=str(path.resolve()), source_sha256=digest(path),
                source_fps=fps, source_watermark=len(reader)-1,
                common_1fps_pool=pool, selected_source_frame_ids=selected,
                selected_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                method='last four of common 1 FPS pool; no historical retrieval'), indent=2))
            answers = []
            visual_saved = False
            for row in rows:
                messages = [dict(role='user', content=messages_media + [
                    dict(type='text', text=row['formatted_question'])])]
                batch, receipt = runtime.prepare(messages, provenance, watermark=len(reader)-1)
                if receipt['image_tokens'] <= 0 or receipt['video_tokens']:
                    raise ValueError('Expected separate actual image inputs')
                if len(receipt['image_grid_thw']) != len(selected):
                    raise ValueError('source-frame/image mapping mismatch')
                visual_receipt = {key: value for key, value in receipt['tensors'].items()
                                  if key not in ('input_ids', 'attention_mask')}
                if not visual_saved:
                    visual = {key: value for key, value in batch.items()
                              if key not in ('input_ids', 'attention_mask')}
                    torch.save(visual, out / 'actual_visual_inputs.pt')
                    (out / 'actual_visual_inputs.json').write_text(json.dumps(visual_receipt, indent=2))
                    visual_saved = True
                elif visual_receipt != json.loads((out / 'actual_visual_inputs.json').read_text()):
                    raise ValueError('Question changed the supposedly shared visual input')
                query_dir = out / ('query_' + str(row['id']))
                query_dir.mkdir(exist_ok=False)
                (query_dir / 'messages.json').write_text(json.dumps(messages, indent=2))
                (query_dir / 'input_receipt.json').write_text(json.dumps(receipt, indent=2))
                (query_dir / 'actual_text_inputs.json').write_text(json.dumps({
                    key: batch[key].tolist() for key in ('input_ids', 'attention_mask')}, indent=2))
                result = runtime.generate(batch, generation_kwargs=dict(
                    max_new_tokens=config['max_new_tokens'], do_sample=False,
                    repetition_penalty=1.0))
                actual = [entry for entry in result['actual_forward_trace']
                          if entry['component'] == 'actual_spatial_encoder']
                if not actual or len(actual[0]['shapes']) != len(selected):
                    raise ValueError('Spatial branch did not consume every selected image')
                result.update(question_id=row['id'], actual_images=len(selected),
                              visual_tokens=receipt['visual_tokens'],
                              text_tokens=receipt['text_and_special_input_tokens'])
                (query_dir / 'output.json').write_text(json.dumps(result, indent=2))
                answers.append(result)
                completion['generations'] += 1
                print('answered', row['id'], 'images', len(selected), flush=True)
            (out / 'complete.json').write_text(json.dumps(dict(answers=answers), indent=2))
            predictions.extend(answers)
        (args.run / 'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in predictions))
        if config['purpose'] == 'wiring_only':
            completion['status'] = 'wiring_passed_not_scored'
        else:
            if len(predictions) != 5130:
                raise ValueError('Full denominator required')
            from score import score
            (args.run / 'scores.json').write_text(json.dumps(score(args.data_root, predictions), indent=2))
            completion['status'] = 'completed_scored'
    except Exception:
        completion.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        completion.update(seconds=time.perf_counter()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        (args.run / 'completion.json').write_text(json.dumps(completion, indent=2))


if __name__ == '__main__':
    main()
