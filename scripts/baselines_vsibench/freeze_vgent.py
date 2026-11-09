"""Freeze full Vgent only after actual dual-stream wiring passed, without scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', required=True, type=Path)
    parser.add_argument('--pilot-freeze', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    project = Path(os.environ.get('SANDMEM_ROOT',Path.cwd())).resolve()
    completion = json.loads((args.pilot/'completion.json').read_text())
    if completion['status'] != 'wiring_passed_not_scored':
        raise ValueError('Vgent GPU wiring has not passed')
    video_calls = 0
    for receipt in args.pilot.glob('question_*/call_*/receipt.json'):
        data = json.loads(receipt.read_text())
        output = json.loads((receipt.parent/'output.json').read_text())
        if data['video_tokens']:
            if not any(trace['component']=='actual_spatial_encoder'
                       for trace in output['actual_forward_trace']):
                raise ValueError('video evidence did not enter Spatial stream')
            media = data['media'][0]
            if max(media['frame_ids']) > data['watermark']:
                raise ValueError('future media')
            video_calls += 1
    if video_calls < 2 or len(list(args.pilot.glob('question_*/result.json'))) != 2:
        raise ValueError('both wiring inputs must complete with actual video evidence')
    # Pin the exact tested transport (not a newly edited working copy).
    for name in ('vgent_spatial_port.py','spatial_features.py'):
        if sha(Path(__file__).parent/name) != sha(args.pilot_freeze/'runtime'/name):
            raise ValueError('working transport differs from accepted wiring')
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines()]
    if len(rows)!=5130 or any('ground_truth' in row for row in rows):
        raise ValueError('full answer-free benchmark inputs required')
    args.run.mkdir(parents=True,exist_ok=False)
    runtime=args.run/'runtime';runtime.mkdir()
    for name in ('vgent_full.py','vgent_spatial_port.py','spatial_features.py','score.py','official_components.py'):
        shutil.copyfile(Path(__file__).parent/name,runtime/name)
    shutil.copyfile(args.inputs,args.run/'inputs.jsonl')
    config=dict(label='Vgent_default_graph_free_Spatial135K_independent_port',
        model='Diankun/Spatial-MLLM-v1.1-Instruct-135K',
        revision='635b63733ed24f76b2f1a533dd5bc32836ea6c53',dtype='bfloat16',
        repo=str(args.repo.resolve()),sample_fps=1,chunk_size=64,n_retrieval=20,
        n_refine=5,uniform_frame=450,total_pixels=16384,
        graph_activation_threshold=1280,benchmark_max_sampled_frames=434,
        graph_active_videos=0,auxiliary_embedding_calls=0,
        output_caps=dict(keywords=256,sql=512,clip_checks=256,aggregation=128,final=10),
        native_max_json_attempts=5,extra_retries=0,
        text_budget='complete native prompts, original question/options; no truncation',
        visual_budget='native 16384*28*28 final total_pixels and up to450 frames; actual tensors/grids archived',
        max_calls_per_question=15,max_model_calls=15*5130,
        max_generated_tokens=4746*5130,
        max_seconds_per_job=42000,checkpoint_boundary='video',gpu='A100',
        benchmark_revision='3a6a1cf9ac54c403509b15260897cc6ddc41b408',
        scores_visible_before_freeze=False,strict_equal_budget_claim=False,
        pilot=str(args.pilot.resolve()),pilot_sha256=sha(args.pilot/'completion.json'),
        adaptations=['shared1FPS source pool','native Spatial preprocessing/dual streams',
                     'explicit source frame mapping','VSI answer format and raw output for official scoring'],
        engineering_failure_policy='preserve raw/error and score invalid output wrong; stop on GPU capacity failure')
    (args.run/'config.json').write_text(json.dumps(config,indent=2))
    files=list(runtime.glob('*.py'))+[args.run/'config.json',args.run/'inputs.jsonl',args.pilot/'completion.json']
    files += list(args.repo.rglob('*.py'))
    files += [project/'sandbox_vlm/backbones/spatial_model.py',project/'sandbox_vlm/backbones/spatial_inputs.py']
    files += list((project/'work/spatial-mllm-official/src/qwenvl/model').glob('*.py'))
    (args.run/'manifest.json').write_text(json.dumps(dict(files={str(path.resolve()):sha(path) for path in files}),indent=2))
    print('Vgent formal freeze completed',args.run,flush=True)


if __name__=='__main__':
    main()
