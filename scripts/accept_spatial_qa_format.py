"""One fixed sparse two-pass wiring check; no labels, scores or model retries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out):
    from sandbox_vlm.backbones.spatial_transport import spatial_config
    from sandbox_vlm.backbones.spatial_qa_format import POLICY, INSTRUCTION
    source_root = ROOT / 'work/migration_135k_20261103/gpu_sparse_02'
    source = json.loads((source_root / 'source.json').read_text())
    input_path = Path(source['input_path'])
    if sha(input_path) != source['input_sha256']:
        raise ValueError('original input changed')
    snapshot = json.loads(input_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    config = spatial_config(json.loads((ROOT / 'configs/reference_qwen.json').read_text()))
    write(out / 'config.json', config)
    files = {}
    for name in ('draft_messages.json', 'actual_joint_messages.json'):
        path = source_root / name
        messages = json.loads(path.read_text())
        if name == 'actual_joint_messages.json':
            # Remove only the old generated candidate. Its NEW replacement is
            # produced from the original real-frame inputs during this run.
            from sandbox_vlm.qa.two_pass import VERIFY_INSTRUCTION
            blocks = messages[0]['content']
            removed = [b for b in blocks if b.get('text', '').startswith(VERIFY_INSTRUCTION)]
            if len(removed) != 1:
                raise ValueError('expected exactly one old verifier candidate block')
            messages[0]['content'] = [b for b in blocks if not b.get('text', '').startswith(VERIFY_INSTRUCTION)]
        write(out / name, messages)
        files[str(path)] = sha(path)
    # Original draft limits/pixels were recorded by the old source audit.
    audit = json.loads((source_root / 'source.json').read_text())
    original_folder = input_path.parent
    draft_input_path = original_folder / 'draft/input.json'
    if not draft_input_path.exists():
        # D118 keeps its draft in a named first-pass subdirectory.
        candidates = list(original_folder.glob('*/input.json'))
        matching = [p for p in candidates if json.loads(p.read_text()).get('pixel_sha256') == snapshot['pixel_sha256'][1:]]
        if len(matching) != 1:
            raise ValueError('cannot identify unique original draft input')
        draft_input_path = matching[0]
    draft = json.loads(draft_input_path.read_text())
    if draft['pixel_sha256'] != snapshot['pixel_sha256'][1:]:
        raise ValueError('draft must use identical original real frames')
    write(out / 'plan.json', dict(label='fixed_old_Qwen_memory_QA_format_wiring_only',
        policy=POLICY, instruction=INSTRUCTION, generation_call_cap=2, output_cap=256,
        source=source, snapshot=snapshot, draft=draft, watermark=int(original_folder.name.split('_')[1]),
        scoring=False, answers_read=False, retries=0, full_spatial_migration=False,
        authorization='2026-11-06 user requested fixing discovery and sparse QA format'))
    for p in [input_path, draft_input_path, source_root / 'source.json', Path(__file__)]:
        files[str(p.resolve())] = sha(p)
    for p in list((ROOT / 'sandbox_vlm/backbones').glob('*.py')) + list(out.glob('*.json')):
        files[str(p.resolve())] = sha(p)
    for path in snapshot['paths']:
        files[path] = sha(Path(path))
    write(out / 'manifest.json', dict(files=files, status='prepared_not_run'))


def run(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    for path, expected in manifest['files'].items():
        if sha(Path(path)) != expected:
            raise ValueError('prepared dependency changed: ' + path)
    import torch
    from PIL import Image
    from sandbox_vlm.backbones.spatial_transport import SpatialBackbone
    from sandbox_vlm.backbones.spatial_qa_format import SpatialFormatAnswerer, INSTRUCTION
    from sandbox_vlm.backbones.spatial_inputs import input_record
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.bounded_notes import POLICY
    from sandbox_vlm.qa.two_pass import candidate_from_generation, verification_messages, decide
    plan = json.loads((out / 'plan.json').read_text())
    gpu = out / 'gpu'
    gpu.mkdir(exist_ok=False)
    write(gpu / 'started.json', dict(manifest_sha256=sha(out / 'manifest.json')))
    try:
        torch.set_num_threads(1)
        bundle = SpatialBackbone(json.loads((out / 'config.json').read_text()), ROOT / 'work/spatial-mllm-official')
        answerer = SpatialFormatAnswerer(bundle.answerer.config, bundle.runtime)
        write(gpu / 'hardware.json', dict(name=torch.cuda.get_device_name(0),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory))
        candidate = None
        summaries = []
        for label, spec, filename in [('draft', plan['draft'], 'draft_messages.json'),
                                       ('joint', plan['snapshot'], 'actual_joint_messages.json')]:
            folder = gpu / label
            folder.mkdir()
            messages = json.loads((out / filename).read_text())
            images = [Image.open(path).convert('RGB') for path in spec['paths']]
            if [pixel_sha256(i) for i in images] != spec['pixel_sha256']:
                raise ValueError('source pixels changed')
            if label == 'joint':
                messages = verification_messages(messages, candidate)
            write(folder / 'original_messages.json', messages)
            def capture(texts, batch):
                if not all(INSTRUCTION in text for text in texts):
                    raise ValueError('actual QA format instruction missing')
                torch.save(dict(batch), folder / 'actual_inputs.pt')
                write(folder / 'actual_inputs.json', dict(texts=texts,
                    tensors={k: input_record(v) for k,v in batch.items()}))
            bundle.runtime.transport_processor.audit_sink = capture
            with (gpu / 'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(stage=label, generation_requests=1))+'\n')
            generated = answerer.answer(messages, images, QABudget(**spec['limits']),
                note_boundary=POLICY if label == 'joint' else None)
            write(folder / 'generation.json', generated)
            parsed = candidate_from_generation(generated)
            write(folder / 'parsed.json', parsed)
            if label == 'draft':
                candidate = parsed
            else:
                decision = decide(candidate, generated,
                    [m['frame_ids'][0] for m in plan['source']['media'] if m['kind'] == 'source_frame'])
                write(folder / 'decision.json', decision)
            # Format only: compare against visible option letters, never a key.
            question = plan['source']['original_question']
            import re
            letters = re.findall(r'^([A-Z])\. ', question, flags=re.M)
            format_ok = parsed['status'] == 'ok' and (not letters or parsed['answer'] in letters)
            summaries.append(dict(stage=label, status=parsed['status'], format_ok=format_ok,
                output_tokens=generated.get('output_tokens'), model_calls=generated['model_calls']))
            for image in images:
                image.close()
        write(gpu / 'completion.json', dict(status='passed' if all(s['format_ok'] for s in summaries)
            else 'output_contract_failed', summaries=summaries, generation_requests=2, formal_scores=None,
            full_spatial_migration=False, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    except Exception:
        import traceback
        write(gpu / 'failure.json', dict(error=traceback.format_exc(), retry_allowed=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.output.resolve())
