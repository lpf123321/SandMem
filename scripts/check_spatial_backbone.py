"""Auditable backbone-only wiring, CPU by default; never a formal score run.

Use the already input-hash-selected CPU source audit; no answer/score files are
read. GPU mode explicitly labels QA as fixed-Qwen-memory transport verification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def check(args):
    from PIL import Image
    import torch
    from transformers import Qwen2_5_VLProcessor
    from huggingface_hub import snapshot_download
    from sandbox_vlm.backbones.spatial_model import MODEL_ID, REVISION
    from sandbox_vlm.backbones.spatial_transport import SpatialProcessor, SpatialBackbone, spatial_config, spatial_limits
    from sandbox_vlm.backbones.spatial_inputs import validate_media, input_record
    from sandbox_vlm.qa.relevance import relevance_messages
    from sandbox_vlm.perception.discovery import discovery_prompt
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.budget import check_budget
    from sandbox_vlm.qa.d065_prompt_alignment import extract_d065_answer
    from sandbox_vlm.qa.artifact import pixel_sha256
    source = json.loads((args.source_audit/'source.json').read_text())
    messages = json.loads((args.source_audit/'messages.json').read_text())
    snapshot = json.loads(Path(source['input_path']).read_text())
    # These inputs inherit old Qwen memory; do not advertise a new full method.
    query_folder = Path(source['input_path']).parent.name
    if not query_folder.startswith('query_'):
        raise ValueError('cannot establish the original query watermark')
    watermark = int(query_folder.removeprefix('query_'))
    if hashlib.sha256(Path(source['input_path']).read_bytes()).hexdigest() != source['input_sha256']:
        raise ValueError('reference input receipt changed')
    validate_media(messages, source['media'], watermark=watermark, require_sandbox=True)
    config = spatial_config(json.loads((ROOT/'configs/reference_qwen.json').read_text()))
    checkpoint = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    processor = SpatialProcessor(Qwen2_5_VLProcessor.from_pretrained(checkpoint,
                                local_files_only=True, use_fast=False))
    blocks = [c for m in messages for c in m['content'] if c['type'] == 'image']
    paths = snapshot['paths']
    if len(paths) != len(blocks):
        raise ValueError('image placeholder/source mapping mismatch')
    for index, (block, path) in enumerate(zip(blocks, paths)):
        if block.pop('evidence_image_index') != index or block.pop('pixel_sha256') != snapshot['pixel_sha256'][index]:
            raise ValueError('source image index/hash changed')
        block['image'] = path
    images = [Image.open(p).convert('RGB') for p in paths]
    if [pixel_sha256(i) for i in images] != snapshot['pixel_sha256']:
        raise ValueError('actual source pixels changed')
    real_indices = [i for i, m in enumerate(source['media']) if m['kind'] == 'source_frame']
    # Fixed input order, no inspection of predictions or answers.
    selected = real_indices[:2]
    real = [images[i] for i in selected]
    ids = [source['media'][i]['frame_ids'][0] for i in selected]
    question = source['original_question']
    discovery_messages = [[dict(role='user', content=[dict(type='image', image=image),
        dict(type='text', text=discovery_prompt(fid, question=question))])]
        for image, fid in zip(real, ids)]
    prepared = {}
    for name, value in [('joint', messages),
                        ('relevance', [relevance_messages(question, i) for i in real]),
                        ('discovery', discovery_messages)]:
        batch = processor.apply_chat_template(value, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt', padding=True)
        prepared[name] = batch
        torch.save(dict(batch), args.output/(name+'_inputs.pt'))
        write(args.output/(name+'_tensors.json'), {k: input_record(v) for k,v in batch.items()})
    write(args.output/'messages.json', messages)
    write(args.output/'config.json', config)
    write(args.output/'source.json', source)
    summary = dict(mode='CPU_transport_only_fixed_Qwen_memory', model_id=MODEL_ID, revision=REVISION,
        source_audit=str(args.source_audit.resolve()), watermark=watermark,
        selected_source_frames=ids, sandbox_in_joint=True, model_calls=0,
        formal_evaluation=False, full_spatial_memory=False,
        original_messages_preserved=True, source_pixel_sha256=[pixel_sha256(i) for i in images])
    # Verify single-image preparation equals the same member of the batch.
    for index, conversation in enumerate(discovery_messages):
        single = processor.apply_chat_template(conversation, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt')
        batch = prepared['discovery']
        assert torch.equal(single['input_ids'][0],
                           batch['input_ids'][index][batch['attention_mask'][index].bool()])
        assert torch.equal(single['image_tchw'][0], batch['image_tchw'][index])
    summary['independent_batch_matches_single'] = True
    joint = prepared['joint']
    grid = joint['image_grid_thw'].tolist()
    visual = int((joint['input_ids'] == processor.raw.tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum())
    summary['original_joint_budget'] = check_budget(QABudget(**snapshot['limits']),
        image_sizes_hw=[[i.height, i.width] for i in images],
        processed_image_sizes_hw=[[h*14, w*14] for t,h,w in grid],
        input_tokens=int(joint['input_ids'].shape[-1]), visual_tokens=visual,
        context_limit=128000)
    summary['joint_budget'] = check_budget(spatial_limits(QABudget(**snapshot['limits']), images),
        image_sizes_hw=[[i.height, i.width] for i in images],
        processed_image_sizes_hw=[[h*14, w*14] for t,h,w in grid],
        input_tokens=int(joint['input_ids'].shape[-1]), visual_tokens=visual,
        context_limit=128000)
    if args.gpu:
        if args.official_root is None:
            raise ValueError('GPU verification needs a complete pinned official checkout')
        bundle = SpatialBackbone(config, args.official_root)
        write(args.output/'hardware.json', dict(name=torch.cuda.get_device_name(0),
            total_memory=torch.cuda.get_device_properties(0).total_memory))
        def capture(name):
            def save(texts, batch):
                torch.save(dict(batch), args.output/(name+'_actual_inputs.pt'))
                write(args.output/(name+'_actual_receipt.json'), dict(prompts=texts,
                    tensors={k:input_record(v) for k,v in batch.items()}))
            bundle.answerer.processor.audit_sink = save
        capture('relevance')
        with bundle.runtime.trace_media() as trace:
            coverage = bundle.coverage.encode(bundle.coverage.prepare(real))
            with bundle.coverage.reuse(coverage, list(range(len(real)))):
                margins, stats = bundle.relevance.score(question, real)
        write(args.output/'relevance.json', dict(margins=margins, stats=stats, actual_forward_trace=trace))
        if not any(t['component'] == 'actual_spatial_encoder' for t in trace):
            raise ValueError('relevance did not execute Spatial spatial encoder')
        # Keep the inherited discovery prompt, batch bounds, repeat guard and parser.
        capture('discovery')
        found = bundle.discovery.discover_batch([paths[i] for i in selected], ids, question=question)
        write(args.output/'discovery.json', found)
        if not all(any(t['component'] == 'actual_spatial_encoder' for t in r['actual_forward_trace']) for r in found):
            raise ValueError('discovery did not execute Spatial spatial encoder')
        limits = QABudget(**snapshot['limits'])
        boundary = snapshot.get('note_boundary')
        # Frozen input structure selects the original D116 path. Never read a
        # saved candidate or prediction: the draft below is generated afresh.
        two_pass = Path(source['input_path']).name == 'base_input.json'
        qa_calls = 0
        if two_pass:
            from sandbox_vlm.qa.two_pass import candidate_from_generation, verification_messages, decide
            draft_root = Path(source['input_path']).parent/'draft'
            draft_input = json.loads((draft_root/'input.json').read_text())
            draft_messages = json.loads((draft_root/'messages.json').read_text())
            source_hashes = [snapshot['pixel_sha256'][i] for i in real_indices]
            if draft_input['pixel_sha256'] != source_hashes:
                raise ValueError('draft must use exactly the same causal real frames')
            draft_images = [Image.open(p).convert('RGB') for p in draft_input['paths']]
            if [pixel_sha256(i) for i in draft_images] != draft_input['pixel_sha256']:
                raise ValueError('draft source pixels changed')
            for m in draft_messages:
                for b in m['content']:
                    if b['type'] == 'image':
                        index = b.pop('evidence_image_index')
                        if b.pop('pixel_sha256') != draft_input['pixel_sha256'][index]:
                            raise ValueError('draft image identity changed')
                        b['image'] = draft_input['paths'][index]
            write(args.output/'draft_messages.json', draft_messages)
            capture('draft')
            draft = bundle.answerer.answer(draft_messages, draft_images,
                QABudget(**draft_input['limits']))
            qa_calls += draft['model_calls']
            write(args.output/'draft_generation.json', draft)
            candidate = candidate_from_generation(draft)
            write(args.output/'candidate.json', candidate)
            messages = verification_messages(messages, candidate)
        write(args.output/'actual_joint_messages.json', messages)
        capture('joint')
        generated = bundle.answerer.answer(messages, images, limits, note_boundary=boundary)
        qa_calls += generated['model_calls']
        write(args.output/'joint_generation.json', generated)
        if not generated['model_calls'] or not any(t['component'] == 'actual_spatial_encoder'
                                                 for t in generated['actual_forward_trace']):
            raise ValueError('joint images did not execute Spatial spatial encoder')
        parsed = None
        if two_pass:
            decision = decide(candidate, generated,
                [m['frame_ids'][0] for m in source['media'] if m['kind'] == 'source_frame'])
            write(args.output/'decision.json', decision)
            parsed = decision['prediction']
        elif generated['status'] == 'generated':
            parsed = extract_d065_answer(generated['raw_output'], sparse=snapshot['sparse'])
        write(args.output/'parsed.json', dict(prediction=parsed, scoring_performed=False))
        summary.update(mode='GPU_transport_fixed_Qwen_memory_QA_swap',
            model_calls=2+qa_calls, two_pass=two_pass,
            qa_status=generated['status'], peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved())
    write(args.output/'completion.json', dict(status='passed', **summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--official-root', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        check(args)
    except Exception as exc:
        write(args.output/'failure.json', dict(status='failed', error=repr(exc)))
        raise
