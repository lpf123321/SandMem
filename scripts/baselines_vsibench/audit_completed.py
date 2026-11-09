"""Read-only completion/input/budget audit and final tables; no model calls."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from score import official_scorer


ROOT = Path('work/baselines_vsibench_20261105')
RUNS = {'CausalMem': 'formal_causalmem_03', 'HERMES': 'formal_hermes_01'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scorer, _ = official_scorer(ROOT / 'repos/thinking-in-space')
    report = {}
    aligned = {}
    score_rows = []
    for label, name in RUNS.items():
        run = ROOT / name
        manifest = json.loads((run / 'manifest.json').read_text())
        mismatches = [key for key, expected in manifest['files'].items()
                      if digest(run / key) != expected]
        mismatches += [key for key, expected in manifest['external_files'].items()
                       if digest(Path(key)) != expected]
        assert not mismatches, mismatches
        inputs = [json.loads(line) for line in (run / 'inputs.jsonl').read_text().splitlines()]
        assert all('ground_truth' not in row for row in inputs)
        by_id = {row['id']: row for row in inputs}
        predictions = [json.loads(line) for line in (run / 'predictions.jsonl').read_text().splitlines()]
        assert len(predictions) == len({row['question_id'] for row in predictions}) == 5130
        assert set(by_id) == {row['question_id'] for row in predictions}
        videos = [json.loads(path.read_text()) for path in sorted((run / 'videos').glob('*.json'))]
        assert len(videos) == 288
        assert all(row['status'] == 'video_completed' for row in videos)
        detailed = [answer for row in videos for answer in row['answers']]
        assert {row['question_id']: row for row in detailed} == {row['question_id']: row for row in predictions}
        encodings = [entry for video in videos for entry in video['encoding']]
        assert len(encodings) == 1881
        for video in videos:
            actual = [frame for encoding in video['encoding'] for frame in encoding['source_frame_ids']]
            assert actual == video['source_frame_ids']
            assert actual == sorted(set(actual))
        assert sum(len(video['source_frame_ids']) for video in videos) == 27946
        missing_features = [row['feature_path'] for row in encodings if not Path(row['feature_path']).is_file()]
        assert not missing_features, missing_features
        scores = json.loads((run / 'scores.json').read_text())
        assert scores['reference_sha256'] == digest(ROOT / 'data/test.jsonl')
        tasks = {key: value for key, value in scores['scores'].items()
                 if key not in ('overall', 'tabulated_keys', 'tabulated_results')}
        assert len(tasks) == 8
        assert abs(sum(tasks.values()) / 8 - scores['scores']['overall']) < 1e-10
        invalid = []
        lengths, text_lengths, output_limits, lookahead = [], [], [], []
        for answer in predictions:
            question = by_id[answer['question_id']]
            parsed = scorer['fuzzy_matching'](answer['raw_output'])
            if question.get('options'):
                valid = parsed.lower() in [chr(97 + index) for index in range(len(question['options']))]
            else:
                value = scorer['to_float'](parsed)
                valid = value is not None and math.isfinite(value)
            if not valid:
                invalid.append(dict(question_id=question['id'], raw_output=answer['raw_output']))
            # Upstream HERMES performs an unused extra decoder step at its
            # length cap. Preserve its raw head-hook record; disclose the
            # lookahead separately instead of counting it as generated text.
            recorded = len(answer['output_token_ids'])
            lengths.append(min(recorded, 16))
            if recorded > 16:
                lookahead.append(question['id'])
            if answer.get('reached_output_limit') or recorded > 16:
                output_limits.append(question['id'])
            ids = answer['input_text_token_ids']
            text_lengths.append(sum(len(value) for value in ids.values()) if isinstance(ids, dict) else len(ids))
        aligned[label] = {(video['dataset'], video['scene_name']): dict(
            source_video_sha256=video['source_video_sha256'], source_frame_ids=video['source_frame_ids'],
            encodings=[{key: entry[key] for key in ('source_frame_ids', 'padded_source_frame_ids',
                'video_grid_thw', 'video_pixels', 'spatial_input', 'feature_sha256')}
                for entry in video['encoding']]) for video in videos}
        jobs = json.loads((run / 'serial_controller.json').read_text())['jobs']
        receipts = [json.loads((run / ('job_' + str(job) + '.json')).read_text()) for job in jobs]
        result = dict(run=name, jobs=jobs, frozen_file_hashes_checked=len(manifest['files']),
            frozen_dependency_hashes_checked=len(manifest['external_files']),
            frozen_hash_mismatches=mismatches, questions=5130, videos=288,
            status_counts=dict(Counter(row['status'] for row in predictions)),
            parse_invalid_count=len(invalid), parse_invalid=invalid,
            sampled_source_frames=27946, encoding_chunks=len(encodings),
            encoded_visual_tokens=sum(row['visual_tokens'] for row in encodings),
            visual_encoder_calls=sum(row['visual_encoder_calls'] for row in encodings),
            spatial_encoder_calls=sum(row['spatial_encoder_calls'] for row in encodings),
            connector_calls=sum(row['connector_calls'] for row in encodings),
            pseudo_attention_query_batches=sum(row.get('pseudo_attention_queries', 0) for row in encodings),
            qa_requests=len(predictions), output_tokens=sum(lengths), max_output_tokens=max(lengths),
            output_limit_question_ids=output_limits, unused_decoder_lookahead_question_ids=lookahead,
            new_qa_text_token_min=min(text_lengths), new_qa_text_token_max=max(text_lengths),
            new_qa_text_token_sum=sum(text_lengths),
            encoder_seconds=sum(row['encoder_seconds'] for row in encodings),
            memory_seconds=sum(row['memory_seconds'] for row in encodings),
            qa_seconds=sum(row['qa_seconds'] for row in predictions),
            program_seconds=sum(row['seconds'] for row in receipts),
            hardware=[row['hardware'] for row in receipts],
            peak_gpu_allocated_gib=max(row['peak_gpu_allocated_bytes'] for row in videos) / 1024**3,
            peak_gpu_reserved_gib=max(row['peak_gpu_reserved_bytes'] for row in videos) / 1024**3,
            archived_feature_bytes=sum(Path(row['feature_path']).stat().st_size for row in encodings),
            feature_files_exist=True, feature_digest_check='recorded at creation; cross-method receipts compared; no second bulk read',
            scores=scores['scores'])
        if label == 'CausalMem':
            result['max_retained_visual_tokens'] = max(row['retained_visual_tokens'] for row in predictions)
            result['max_retained_position_metadata_bytes'] = max(row['retained_position_metadata_bytes'] for row in predictions)
            assert result['max_retained_visual_tokens'] <= 12000
        else:
            result['max_post_compression_kv_tokens_per_layer_including_system'] = max(
                max(row['post_memory_kv_lengths']) for row in encodings)
            assert result['max_post_compression_kv_tokens_per_layer_including_system'] <= 6014
        report[label] = result
        score_rows.append(dict(method=label + ' + Spatial-135K', overall=scores['scores']['overall'], **tasks))
    fused_differences = []
    for video_key, left in aligned['CausalMem'].items():
        right = aligned['HERMES'][video_key]
        for index, (left_chunk, right_chunk) in enumerate(zip(left['encodings'], right['encodings'])):
            left_hash = left_chunk.pop('feature_sha256')
            right_hash = right_chunk.pop('feature_sha256')
            if left_hash != right_hash:
                fused_differences.append(dict(dataset=video_key[0], scene_name=video_key[1],
                    chunk_index=index, causal_feature_sha256=left_hash, hermes_feature_sha256=right_hash))
    assert aligned['CausalMem'] == aligned['HERMES'], 'shared source/pixel/grid input mismatch'
    report['shared_input_audit'] = dict(all_288_videos_equal=True, all_1881_chunks_equal=True,
        source_frames_pixels_and_grids_equal=True,
        recorded_fused_feature_hashes_equal=not fused_differences,
        matching_fused_feature_hash_count=1881 - len(fused_differences),
        fused_feature_hash_differences=fused_differences,
        limitation='one independently encoded chunk differs numerically; cause not established; no bit-exact claim'
            if fused_differences else None,
        question_manifests_equal=(ROOT / RUNS['CausalMem'] / 'inputs.jsonl').read_bytes()
            == (ROOT / RUNS['HERMES'] / 'inputs.jsonl').read_bytes())
    assert report['shared_input_audit']['question_manifests_equal']
    (ROOT / 'reports/completion_audit.json').write_text(json.dumps(report, indent=2))
    with (ROOT / 'reports/results.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    print(json.dumps({key: {field: value[field] for field in (
        'questions', 'videos', 'parse_invalid_count', 'output_tokens', 'max_output_tokens',
        'output_limit_question_ids', 'encoded_visual_tokens', 'program_seconds',
        'peak_gpu_allocated_gib', 'peak_gpu_reserved_gib', 'new_qa_text_token_min',
        'new_qa_text_token_max', 'archived_feature_bytes')} for key, value in report.items()
        if key in RUNS}, indent=2))
    print('PASS: complete denominator, immutable runs, shared source/pixel/grid inputs, budgets and archived predictions')
    if fused_differences:
        print('DISCLOSE:', len(fused_differences), 'fused feature archive differences; no model rerun or score changes')


if __name__ == '__main__':
    main()
