"""Frozen default-config Vgent Spatial port; checkpoint at video boundaries."""
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
from vgent_spatial_port import SpatialVgent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    manifest = json.loads((args.run / 'manifest.json').read_text())
    for path, expected in manifest['files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen dependency changed: '+path)
    inputs = [json.loads(line) for line in (args.run / 'inputs.jsonl').read_text().splitlines()]
    if len(inputs) != 5130 or len({row['id'] for row in inputs}) != 5130:
        raise ValueError('full original denominator required')
    if any('ground_truth' in row for row in inputs):
        raise ValueError('labels in inference manifest')
    grouped = {}
    for row in inputs:
        grouped.setdefault((row['dataset'], row['scene_name']), []).append(row)
    import torch
    from sandbox_vlm.backbones.spatial_model import Spatial135K
    torch.set_num_threads(1)
    started = time.perf_counter()
    completion = dict(status='starting', job_id=os.environ.get('SLURM_JOB_ID'),
                      hardware=torch.cuda.get_device_name(0), manifest_sha256=sha(args.run/'manifest.json'))
    runtime = None
    try:
        runtime = Spatial135K(ROOT/'work/spatial-mllm-official')
        predictions = []
        for (dataset, scene), rows in grouped.items():
            folder = args.run/'videos'/(dataset+'_'+scene)
            completed = folder/'complete.json'
            if completed.exists():
                predictions.extend(json.loads(completed.read_text())['answers'])
                continue
            if time.perf_counter()-started >= config['max_seconds_per_job']:
                completion['status'] = 'video_boundary_checkpoint'
                break
            if folder.exists():
                raise RuntimeError('incomplete video retained; no automatic failed-input retries')
            torch.manual_seed(int(hashlib.sha256((dataset+'/'+scene).encode()).hexdigest()[:8],16) % 2**31)
            port = SpatialVgent(runtime, Path(config['repo']), folder)
            path = args.data_root/'data/videos'/dataset/(scene+'.mp4')
            reader, source_ids, fps = sample_frames(path, 1)
            port.set_video(reader.get_batch(source_ids).asnumpy(), source_ids,
                           source_path=path, source_fps=fps, watermark=len(reader)-1)
            (folder/'source.json').write_text(json.dumps(dict(path=str(path.resolve()),
                sha256=sha(path), source_frame_ids=source_ids, fps=fps,
                graph_active=False, graph_threshold=1280),indent=2))
            answers = []
            for row in rows:
                query = folder/('query_'+str(row['id']))
                query.mkdir(exist_ok=False)
                (query/'input.json').write_text(json.dumps(row,indent=2))
                before = port.call_count
                try:
                    result = dict(status='generated', **port.answer(row))
                except torch.cuda.OutOfMemoryError:
                    # Capacity failure is a wiring failure, not a low benchmark
                    # score. Stop without secretly reducing frames/resolution.
                    (query/'failure.txt').write_text(traceback.format_exc())
                    raise
                except Exception:
                    result = dict(status='inference_failed', raw_output='',
                                  failure=traceback.format_exc(), model_calls=port.call_count-before)
                result.update(question_id=row['id'], first_call=before+1,last_call=port.call_count)
                (query/'output.json').write_text(json.dumps(result,indent=2))
                answers.append(result)
                print('Vgent answered',row['id'],result['status'],flush=True)
            completed.write_text(json.dumps(dict(answers=answers),indent=2))
            predictions.extend(answers)
        if len(predictions) == 5130:
            from score import score
            (args.run/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in predictions))
            (args.run/'scores.json').write_text(json.dumps(score(args.data_root,predictions),indent=2))
            completion['status'] = 'completed_scored'
    except Exception:
        completion.update(status='failed',failure=traceback.format_exc())
        raise
    finally:
        completion.update(seconds=time.perf_counter()-started,
            new_model_calls=runtime.generation_attempts if runtime else 0,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        (args.run/('completion_'+str(os.environ.get('SLURM_JOB_ID','local'))+'.json')).write_text(
            json.dumps(completion,indent=2))
        (args.run/'completion.json').write_text(json.dumps(completion,indent=2))


if __name__ == '__main__':
    main()
