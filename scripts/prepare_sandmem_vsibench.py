"""CPU-only, answer-free VSI-Bench inputs for the full Spatial migration.

This prepares data, a fixed wiring subset and dependency/cost contracts. It is
deliberately not a formal-run authorization or a Qwen-memory replay entrypoint.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def question_input(row):
    item = {key: row[key] for key in ('id', 'dataset', 'scene_name', 'question', 'options')}
    text = 'These are frames of a video.\n' + item['question']
    if item['options']:
        text += '\nOptions:\n' + '\n'.join(item['options'])
        text += "\nAnswer with the option's letter from the given choices directly."
    else:
        text += '\nPlease answer the question using a single word or phrase.'
    item['formatted_question'] = text
    return item


def prepare(data_root, out):
    source = data_root / 'data/test.jsonl'
    # Explicit allowlist: do not propagate question_type, labels or 3D metadata.
    inputs = [question_input(json.loads(line)) for line in source.read_text().splitlines()]
    if len(inputs) != 5130 or len({r['id'] for r in inputs}) != 5130:
        raise ValueError('expected complete 5130-question benchmark')
    metadata_path = data_root / 'reports/full_video_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    meta = {(r['dataset'], r['scene_name']): r for r in metadata}
    keys = {(r['dataset'], r['scene_name']) for r in inputs}
    if len(keys) != 288 or keys != set(meta):
        raise ValueError('video metadata and question manifest disagree')
    videos = []
    for key in sorted(keys):
        m = meta[key]
        path = data_root / 'data/videos' / key[0] / (key[1] + '.mp4')
        if not path.is_file():
            raise FileNotFoundError(path)
        count, fps = m['source_frames'], m['source_fps']
        # Exact source-index list used by the baseline np.arange(time, step=1)
        # pool. Keep internal ordinal frame IDs distinct from source indices.
        indices = sorted(set(min(count - 1, int(t * fps)) for t in range(math.ceil(count / fps))))
        if len(indices) != m['sampled_frames_1fps']:
            raise ValueError('shared sampling count mismatch')
        videos.append(dict(dataset=key[0], scene_name=key[1], source_video=str(path.resolve()),
            source_bytes=path.stat().st_size, source_fps=fps, source_frames=count,
            source_watermark=count-1, sampled_watermark=len(indices)-1,
            frame_mapping=[dict(frame_id=i, source_index=fid, t_sec=fid/fps) for i, fid in enumerate(indices)]))
    wiring = []
    for choices in (False, True):
        pool = [r for r in inputs if bool(r['options']) == choices]
        wiring.append(min(pool, key=lambda r: hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in inputs))
    write(out / 'wiring_inputs.json', dict(selection='minimum SHA256 of whitelisted input JSON in each answer-format group',
        inputs=wiring, scores_used=False, new_blind_test_claim=False))
    write(out / 'videos.json', videos)
    write(out / 'protocol.json', dict(status='data_prepared_not_formal_freeze', method='D116 + first_last_then_oldest_observed',
        model='Diankun/Spatial-MLLM-v1.1-Instruct-135K', revision='635b63733ed24f76b2f1a533dd5bc32836ea6c53',
        dtype='bfloat16', training=False, full_spatial_migration=False,
        dataset_revision='3a6a1cf9ac54c403509b15260897cc6ddc41b408', dataset_scope='original_full_5130',
        visible_prefix='full original video, offline questions; mapping carries explicit watermarks', sample_fps=1,
        question_type_available_to_inference=False, scoring='pinned official VSI-Bench scorer; eight-task macro; numeric official MRA',
        scorer=str((ROOT / 'scripts/baselines_vsibench/score.py').resolve()), scorer_data_root=str(data_root.resolve()),
        scorer_commit='51e089c3ae69b9435e9489058610f5b3964c56a8', invalid_answers='retain full denominator; no fabricated/retried answer',
        strict_equal_budget_claim=False, GPU='A100; no concurrent submission while Vgent holds user job quota',
        native_alignment='1280 raw sandbox width may become 1288 after native processor; approved, disclose actual grids',
        cost_fields=['coverage_tower_calls_and_images', 'relevance_calls_and_images', 'discovery_calls_and_images',
            'external_SAM_prompts_and_propagation', 'external_VGGT_calls_and_frames', 'sandbox_images', 'real_QA_frames',
            'joint_visual_tokens', 'text_tokens', 'output_tokens_and_caps', 'draft_calls', 'verification_calls',
            'decode_CPU_seconds', 'model_GPU_seconds', 'allocated_wall_seconds', 'peak_memory', 'failures'],
        gates_remaining=['approved four-frame discovery GPU contract', 'sparse QA output contract',
            'fresh Spatial coverage/discovery/SAM instances plus local geometry connected to D116',
            'two input-fixed end-to-end wiring samples; actual images/tensors/provenance/budgets audit',
            'freeze exact production configuration/code/call budgets before formal execution']))
    write(out / 'cache_dependencies.json', dict(
        reusable=['original videos with content hash verification', 'answer-free question manifest',
            'source metadata and exact frame mapping', 'decoded raw RGB only with identical source indices/pixels'],
        regenerate=['all Qwen MLLM screening margins, visual descriptors, discovery records and generated text',
            'frame selection, instance memory and association state dependent on those outputs',
            'SAM masks/features for new discovery boxes; local AABBs and renders depending on new regions',
            'all Spatial QA candidates, verification and final outputs'],
        conditional=['external VGGT camera/depth only for identical ordered frame windows, weights and settings; charge provenance and original cost',
            'SAM result only for identical image/box/model/settings, independent of Qwen semantic output; never reuse old instance membership'],
        policy='no Qwen perception caches in the full Spatial run; old-memory QA control must be separately labeled'))
    files = [source, metadata_path, Path(__file__), ROOT / 'scripts/baselines_vsibench/score.py']
    files += list(out.glob('*.json')) + [out / 'inputs.jsonl']
    write(out / 'manifest.json', dict(status='CPU_preparation_only', questions=5130, videos=288,
        wiring_question_ids=[r['id'] for r in wiring], model_calls=0,
        files={str(p.resolve()): sha(p) for p in files},
        video_hash_verification='must bind verified source content hashes at end-to-end/formal freeze; size is not identity'))
    print(json.dumps(dict(output=str(out.resolve()), questions=5130, videos=288,
        wiring_question_ids=[r['id'] for r in wiring], model_calls=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'work/baselines_vsibench_20261105')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_root.resolve(), args.output.resolve())
