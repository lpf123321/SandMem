"""Input-fixed Vgent transport/memory-capacity acceptance; never scores answers."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.inputs.parent / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen Vgent dependency changed: ' + name)
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines()]
    if len(rows) != 2 or any('ground_truth' in row for row in rows):
        raise ValueError('two fixed, answer-free wiring inputs required')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from sandbox_vlm.backbones.spatial_model import Spatial135K
    started = time.perf_counter()
    completion = dict(status='running', job_id=os.environ.get('SLURM_JOB_ID'),
                      hardware=torch.cuda.get_device_name(0), purpose='wiring_only_no_scores')
    runtime = None
    try:
        torch.set_num_threads(1)
        torch.manual_seed(0)
        runtime = Spatial135K(ROOT / 'work/spatial-mllm-official')
        for row in rows:
            out = args.output / ('question_' + str(row['id']))
            port = SpatialVgent(runtime, args.repo, out)
            path = args.data_root / 'data/videos' / row['dataset'] / (row['scene_name']+'.mp4')
            reader, ids, fps = sample_frames(path, 1.0)
            port.set_video(reader.get_batch(ids).asnumpy(), ids, source_path=path,
                           source_fps=fps, watermark=len(reader)-1)
            (out / 'input.json').write_text(json.dumps(row, indent=2))
            result = port.answer(row)
            (out / 'result.json').write_text(json.dumps(result, indent=2))
            print('Vgent wired', row['id'], 'actual MLLM calls', port.call_count, flush=True)
        completion.update(status='wiring_passed_not_scored', model_calls=runtime.generation_attempts)
    except Exception:
        completion.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        completion.update(seconds=time.perf_counter()-started,
            model_calls=runtime.generation_attempts if runtime is not None else 0,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        (args.output / 'completion.json').write_text(json.dumps(completion, indent=2))


if __name__ == '__main__':
    main()
