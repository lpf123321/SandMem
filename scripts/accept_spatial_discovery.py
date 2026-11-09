"""Prepare/run the single approved four-frame discovery compatibility check.

Preparation is CPU-only. GPU execution consumes at most two generation batches,
never reads labels and never repeats a started request. Not a full method run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def prepare(out):
    from sandbox_vlm.backbones.spatial_discovery import POLICY, instruction
    from sandbox_vlm.backbones.spatial_transport import spatial_config
    from sandbox_vlm.perception.discovery import discovery_prompt
    reference = ROOT / 'work/migration_135k_20261103'
    groups, files = [], {}
    for label, folder, expected_ids in [('sparse', 'gpu_sparse_02', [477, 729]),
                                        ('dense', 'gpu_dense_01', [16, 89])]:
        source_path = reference / folder / 'source.json'
        source = json.loads(source_path.read_text())
        path = Path(source['input_path'])
        if sha(path) != source['input_sha256']:
            raise ValueError('original input receipt changed')
        snapshot = json.loads(path.read_text())
        indices = [i for i, media in enumerate(source['media']) if media['kind'] == 'source_frame'][:2]
        ids = [source['media'][i]['frame_ids'][0] for i in indices]
        if ids != expected_ids:
            raise ValueError('fixed source-frame selection changed')
        watermark = int(path.parent.name.split('_')[1])
        frames = []
        for i, fid in zip(indices, ids):
            if fid > watermark:
                raise ValueError('future source frame')
            frame = Path(snapshot['paths'][i])
            original = discovery_prompt(fid, question=source['original_question'])
            frames.append(dict(path=str(frame), frame_id=fid, pixel_sha256=snapshot['pixel_sha256'][i],
                original_prompt=original, actual_prompt=original + '\n' + instruction(fid)))
            files[str(frame)] = sha(frame)
        groups.append(dict(label=label, question=source['original_question'], watermark=watermark, frames=frames))
        files[str(source_path)] = sha(source_path)
        files[str(path)] = sha(path)
    config = spatial_config(json.loads((ROOT / 'configs/reference_qwen.json').read_text()))
    config['spatial_transport'] = dict(discovery_instruction=POLICY)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'config.json', config)
    write(out / 'plan.json', dict(groups=groups, generation_batch_cap=2, logical_discovery_cap=4,
        selection='previously fixed four actual frames; unchanged questions', answers_read=False,
        scoring=False, retries=0, new_SAM_calls=0, new_external_VGGT_calls=0,
        full_spatial_migration=False, authorization='2026-11-06 user approved minimal discovery instruction'))
    frozen = json.loads((ROOT / 'provenance/frozen_files.json').read_text())['source_sha256']
    for name in frozen:
        files[str(ROOT / name)] = sha(ROOT / name)
    for path in list((ROOT / 'sandbox_vlm/backbones').glob('*.py')) + [Path(__file__), out / 'config.json', out / 'plan.json']:
        files[str(path.resolve())] = sha(path)
    write(out / 'manifest.json', dict(files=files, status='prepared_not_run'))


def run(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    for path, expected in manifest['files'].items():
        if sha(path) != expected:
            raise ValueError('prepared dependency changed: ' + path)
    import torch
    from PIL import Image
    from sandbox_vlm.backbones.spatial_transport import SpatialBackbone
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.backbones.spatial_inputs import input_record
    plan = json.loads((out / 'plan.json').read_text())
    gpu = out / 'gpu'
    gpu.mkdir(exist_ok=False)
    write(gpu / 'started.json', dict(manifest_sha256=sha(out / 'manifest.json')))
    outcomes = []
    try:
        torch.set_num_threads(1)
        bundle = SpatialBackbone(json.loads((out / 'config.json').read_text()), ROOT / 'work/spatial-mllm-official')
        write(gpu / 'hardware.json', dict(name=torch.cuda.get_device_name(0),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory))
        for group in plan['groups']:
            folder = gpu / group['label']
            folder.mkdir()
            for frame in group['frames']:
                with Image.open(frame['path']) as opened:
                    if pixel_sha256(opened.convert('RGB')) != frame['pixel_sha256']:
                        raise ValueError('actual source pixels changed')
            def capture(texts, batch):
                torch.save(dict(batch), folder / 'actual_inputs.pt')
                write(folder / 'actual_inputs.json', dict(texts=texts,
                    tensors={k: input_record(v) for k, v in batch.items()}))
                for text, frame in zip(texts, group['frames']):
                    if frame['actual_prompt'] not in text:
                        raise ValueError('approved discovery instruction missing from actual input')
            bundle.runtime.transport_processor.audit_sink = capture
            with (gpu / 'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(label=group['label'], generation_requests=1, logical_requests=2)) + '\n')
            results = bundle.discovery.discover_batch([r['path'] for r in group['frames']],
                [r['frame_id'] for r in group['frames']], question=group['question'])
            write(folder / 'results.json', results)
            for result, frame in zip(results, group['frames']):
                if result['prompt'] != frame['actual_prompt']:
                    raise ValueError('reported prompt differs from actual approved prompt')
                if not any(t['component'] == 'actual_spatial_encoder' for t in result['actual_forward_trace']):
                    raise ValueError('Spatial stream did not execute')
                outcomes.append(dict(label=group['label'], frame_id=result['frame_id'], status=result['status'],
                    detections=len(result['detections']), reached_output_limit=result['reached_output_limit']))
        write(gpu / 'completion.json', dict(status='passed' if all(r['status'] == 'ok' for r in outcomes)
            else 'output_contract_failed', outcomes=outcomes, generation_requests=2, logical_requests=4,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            formal_scores=None, full_spatial_migration=False))
    except Exception:
        import traceback
        write(gpu / 'failure.json', dict(error=traceback.format_exc(), outcomes=outcomes, retry_allowed=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.output.resolve())
