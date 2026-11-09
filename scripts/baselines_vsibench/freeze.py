"""Seal a new formal run after wiring acceptance, without inspecting scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--method', choices=['causalmem', 'hermes'], required=True)
    args = parser.parse_args()
    completion = json.loads((args.pilot / 'completion.json').read_text())
    if (completion['status'] != 'wiring_completed_not_scored'
            or not completion['equivalence']['bit_exact_fused_embeddings']
            or args.method not in completion['methods']):
        raise ValueError('matching wiring acceptance required')
    download = json.loads((args.root / 'reports/video_download.json').read_text())
    if len(download) != 3 or any(row['status'] != 'verified_and_extracted' for row in download):
        raise ValueError('official archives not verified')
    rows = [json.loads(line) for line in (args.root / 'data/test.jsonl').read_text().splitlines()]
    if len(rows) != 5130 or len({row['id'] for row in rows}) != 5130:
        raise ValueError('original full dataset required')
    inputs = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in ('ground_truth', 'pruned')}
        item['formatted_question'] = 'These are frames of a video.\n' + item['question']
        if item.get('options'):
            item['formatted_question'] += '\nOptions:\n' + '\n'.join(item['options'])
            item['formatted_question'] += "\nAnswer with the option's letter from the given choices directly."
        else:
            item['formatted_question'] += '\nPlease answer the question using a single word or phrase.'
        path = args.root / 'data/videos' / item['dataset'] / (item['scene_name'] + '.mp4')
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs.append(item)
    # Video grouping and order depend solely on input names. No scoring here.
    inputs.sort(key=lambda item: (item['dataset'], item['scene_name'], item['id']))
    args.run.mkdir(parents=True, exist_ok=False)
    runtime = args.run / 'runtime'
    runtime.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(source, runtime / source.name)
    config = dict(method=args.method, label=args.method + '_Spatial135K_adapted',
        model_id='Diankun/Spatial-MLLM-v1.1-Instruct-135K',
        revision='635b63733ed24f76b2f1a533dd5bc32836ea6c53', dtype='bfloat16',
        sample_fps=1, encode_chunk_frames=16, hard_frame_cap=None,
        preprocessing='pinned native qwen_vl_utils / Spatial processor defaults; no added resize limit',
        kv_size=6000 if args.method == 'hermes' else None,
        feature_budget=12000 if args.method == 'causalmem' else None,
        max_new_tokens=16, do_sample=False, temperature=0, repetition_penalty=1.0,
        max_questions=5130, max_question_generations=5130, max_output_tokens=5130 * 16,
        max_failed_input_retries=0, stop_on_consecutive_video_failures=2,
        text_budget='complete original question/options and fixed official task instructions; no truncation',
        shared_feature_cache=False, checkpoint_boundary='video',
        benchmark_revision='3a6a1cf9ac54c403509b15260897cc6ddc41b408',
        benchmark_subset='original_full', scores_visible_before_freeze=False,
        strict_equal_budget_claim=False, gpu='A100',
        seed_policy='int(sha256(dataset/scene)[:8], 16) % 2**31, reset at each video',
        pilot=str(args.pilot.resolve()), pilot_completion_sha256=sha256(args.pilot / 'completion.json'))
    (args.run / 'config.json').write_text(json.dumps(config, indent=2))
    (args.run / 'inputs.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in inputs))
    (args.run / 'reference_receipt.json').write_text(json.dumps(dict(
        sha256=sha256(args.root / 'data/test.jsonl'), rows=len(inputs),
        videos=len({(item['dataset'], item['scene_name']) for item in inputs}),
        task_counts=dict(Counter(item['question_type'] for item in inputs)),
        archive_receipts=download), indent=2))
    video_metadata = json.loads((args.root / 'reports/full_video_metadata.json').read_text())
    budget = dict(max_questions=5130, max_generated_answer_tokens=5130 * 16,
        sampled_source_frames=sum(item['sampled_frames_1fps'] for item in video_metadata),
        max_visual_encoder_calls=sum(item['chunks16'] for item in video_metadata),
        max_spatial_encoder_calls=sum(item['chunks16'] for item in video_metadata),
        max_connector_calls=sum(item['chunks16'] for item in video_metadata),
        pseudo_attention_query_batches=3 * sum(item['chunks16'] for item in video_metadata)
            if args.method == 'hermes' else 0,
        visual_tokens='actual native grids recorded at each chunk; no hidden additional cap',
        position_metadata='retained coordinates bounded by retained feature count for CausalMem',
        per_video_call_budget=video_metadata)
    (args.run / 'budget.json').write_text(json.dumps(budget, indent=2))
    files = {str(path.relative_to(args.run)): sha256(path)
             for path in args.run.rglob('*') if path.is_file()}
    upstream = {}
    for repo in [args.root / 'repos/thinking-in-space', args.root / 'repos/' /
                 ('HERMES' if args.method == 'hermes' else 'CausalMem')]:
        upstream[str(repo.resolve())] = subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    project = Path(__file__).resolve().parents[2]
    dependencies = [project / 'sandbox_vlm/backbones/spatial_model.py',
                    project / 'sandbox_vlm/backbones/spatial_inputs.py']
    dependencies.extend((args.root / 'repos' / ('HERMES' if args.method == 'hermes' else 'CausalMem')).rglob('*.py'))
    dependencies.extend((project / 'work/spatial-mllm-official/src/qwenvl/model').rglob('*.py'))
    external_files = {str(path.resolve()): sha256(path) for path in dependencies}
    (args.run / 'manifest.json').write_text(json.dumps(dict(
        files=files, upstream_commits=upstream, external_files=external_files), indent=2))
    print('frozen', args.run, len(inputs), 'questions', flush=True)


if __name__ == '__main__':
    main()
