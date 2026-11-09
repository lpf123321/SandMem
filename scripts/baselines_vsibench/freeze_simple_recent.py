"""Freeze the input-only recent-four adaptation before any scored inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--pilot', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    # Frozen copies live below work/.../runtime rather than scripts/... .
    import os
    project = Path(os.environ.get('SANDMEM_ROOT', project)).resolve()
    if args.pilot:
        acceptance = json.loads((args.pilot / 'completion.json').read_text())
        if acceptance['status'] != 'wiring_passed_not_scored' or acceptance['generations'] != 2:
            raise ValueError('two input-fixed wiring calls must pass before formal freeze')
    rows = [json.loads(line) for line in (args.data_root / 'data/test.jsonl').read_text().splitlines()]
    if len(rows) != 5130 or len({row['id'] for row in rows}) != 5130:
        raise ValueError('original full benchmark expected')
    inputs = []
    for row in rows:
        item = {key: row[key] for key in ('id', 'dataset', 'scene_name', 'question')}
        item['options'] = row.get('options', [])
        item['formatted_question'] = 'These are frames of a video.\n' + item['question']
        if item['options']:
            item['formatted_question'] += '\nOptions:\n' + '\n'.join(item['options'])
            item['formatted_question'] += "\nAnswer with the option's letter from the given choices directly."
        else:
            item['formatted_question'] += '\nPlease answer the question using a single word or phrase.'
        inputs.append(item)
    if not args.pilot:
        # One numerical and one choice question, chosen only by input hashes.
        selected = []
        for choices in (False, True):
            candidates = [item for item in inputs if bool(item['options']) == choices]
            selected.append(min(candidates, key=lambda item: hashlib.sha256(
                json.dumps(item, sort_keys=True).encode()).hexdigest()))
        inputs = selected
    inputs.sort(key=lambda item: (item['dataset'], item['scene_name'], item['id']))
    args.run.mkdir(parents=True, exist_ok=False)
    runtime = args.run / 'runtime'
    runtime.mkdir()
    for name in ('simple_recent.py', 'freeze_simple_recent.py', 'spatial_features.py',
                 'score.py', 'official_components.py'):
        shutil.copyfile(Path(__file__).parent / name, runtime / name)
    upstream = project / 'work/baselines_candidates_20261105/repos/simplestream'
    config = dict(label='SimpleStream_recent4_Spatial135K_independent_port',
        purpose='formal' if args.pilot else 'wiring_only',
        model='Diankun/Spatial-MLLM-v1.1-Instruct-135K',
        revision='635b63733ed24f76b2f1a533dd5bc32836ea6c53',
        dtype='bfloat16', sample_fps=1, recent_frames=4,
        query_watermark='end of source video; original full offline VSI-Bench',
        frame_selection='last four of shared time-arange 1 FPS pool',
        adaptation_differences=[
            'Shared baseline 1 FPS time-arange source pool replaces upstream qwen linspace sampler.',
            'Native Spatial processor and dual image streams replace upstream Qwen image-only feature injection.',
            'Native checkpoint image delimiters/positions replace manually constructed single image prefix.',
            'Official VSI-Bench question/answer instructions replace dataset-specific OVO/StreamingBench instructions.'],
        max_new_tokens=16, do_sample=False, max_generations=len(inputs),
        text_budget='complete original question/options; no truncation',
        visual_budget='four original RGB observations, native Spatial image preprocessing; actual tokens recorded',
        visual_feature_cache=False, auxiliary_models=[], retries=0,
        benchmark_revision='3a6a1cf9ac54c403509b15260897cc6ddc41b408',
        benchmark_subset='original_full', gpu='A100', strict_equal_budget_claim=False,
        pilot=str(args.pilot.resolve()) if args.pilot else None,
        published_vsi_score=None, scores_visible_before_freeze=False)
    (args.run / 'config.json').write_text(json.dumps(config, indent=2))
    (args.run / 'inputs.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in inputs))
    deps = [project / 'sandbox_vlm/backbones/spatial_model.py',
            project / 'sandbox_vlm/backbones/spatial_inputs.py',
            args.data_root / 'data/test.jsonl']
    deps += list((project / 'work/spatial-mllm-official/src/qwenvl/model').glob('*.py'))
    deps += list(upstream.rglob('*.py')) + [upstream / 'README.md']
    if args.pilot:
        deps.append(args.pilot / 'completion.json')
    files = list(args.run.rglob('*.py')) + [args.run / 'config.json', args.run / 'inputs.jsonl'] + deps
    (args.run / 'manifest.json').write_text(json.dumps(dict(files={
        str(path.resolve()): sha(path) for path in files},
        official_repo='https://github.com/EvolvingLMMs-Lab/SimpleStream',
        official_commit='be227ee8ce90a258b286a34161688c2a59eeeadb'), indent=2))
    print('Frozen', args.run, len(inputs), 'input-only questions', flush=True)


if __name__ == '__main__':
    main()
