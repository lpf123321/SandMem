"""Fixed four-frame discovery plus original sparse QA syntax acceptance."""
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
    discovery = ROOT / 'work/spatial_discovery_acceptance_20261106_01'
    qa = ROOT / 'work/spatial_qa_format_acceptance_20261106_01'
    out.mkdir(parents=True, exist_ok=False)
    sources = [discovery/'plan.json', discovery/'config.json', qa/'plan.json',
               qa/'draft_messages.json', qa/'actual_joint_messages.json']
    dp = json.loads(sources[0].read_text())
    qp = json.loads(sources[2].read_text())
    write(out/'plan.json', dict(discovery=dp, qa=qp,
        draft_messages=json.loads(sources[3].read_text()), joint_messages=json.loads(sources[4].read_text()),
        generation_request_cap=4, logical_discovery_cap=4, QA_call_cap=2, formal_scores=None,
        authorization='2026-11-06 user explicitly approved original-output-syntax-only constraints',
        visible_choice_domain='original question letters only; no correct choice supplied',
        decoding_change='single greedy generation among grammar-valid tokens; original caps/parser retained',
        full_spatial_migration=False, retries=0))
    write(out/'config.json', json.loads(sources[1].read_text()))
    files = sources + list((ROOT/'sandbox_vlm/backbones').glob('*.py')) + [Path(__file__)]
    files += [ROOT/name for name in json.loads((ROOT/'provenance/frozen_files.json').read_text())['source_sha256']]
    files += [Path(f['path']) for group in dp['groups'] for f in group['frames']]
    files += [Path(p) for p in qp['snapshot']['paths']]
    files += list(out.glob('*.json'))
    write(out/'manifest.json', dict(files={str(p.resolve()):sha(p) for p in files}, status='prepared_not_run'))


def run(out):
    for path, expected in json.loads((out/'manifest.json').read_text())['files'].items():
        if sha(path) != expected:
            raise ValueError('prepared file changed: '+path)
    import torch
    from PIL import Image
    from sandbox_vlm.backbones.spatial_transport import SpatialBackbone
    from sandbox_vlm.backbones.spatial_syntax import SyntaxTokens, SpatialSyntaxDiscovery, SpatialSyntaxAnswerer
    from sandbox_vlm.backbones.spatial_inputs import input_record
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.two_pass import candidate_from_generation, verification_messages, decide
    from sandbox_vlm.qa.bounded_notes import POLICY
    plan = json.loads((out/'plan.json').read_text())
    gpu = out/'gpu'; gpu.mkdir(exist_ok=False)
    write(gpu/'started.json', dict(manifest_sha256=sha(out/'manifest.json')))
    outcomes = []
    try:
        torch.set_num_threads(1)
        config = json.loads((out/'config.json').read_text())
        bundle = SpatialBackbone(config, ROOT/'work/spatial-mllm-official')
        grammar = SyntaxTokens(bundle.runtime.processor.tokenizer)
        grammar.trace_path = gpu/'decode_steps.jsonl'
        discovery = SpatialSyntaxDiscovery(bundle.discovery.config, bundle.runtime, grammar)
        answerer = SpatialSyntaxAnswerer(bundle.answerer.config, bundle.runtime, grammar)
        write(gpu/'hardware.json', dict(name=torch.cuda.get_device_name(0),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory))
        def capture(folder):
            def sink(texts, batch):
                torch.save(dict(batch), folder/'actual_inputs.pt')
                write(folder/'actual_inputs.json', dict(texts=texts,
                    tensors={k:input_record(v) for k,v in batch.items()}))
            bundle.runtime.transport_processor.audit_sink = sink
        def attempt(stage):
            with (gpu/'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(stage=stage, generation_requests=1))+'\n')
        for group in plan['discovery']['groups']:
            folder = gpu/('discovery_'+group['label']); folder.mkdir(); capture(folder)
            for frame in group['frames']:
                with Image.open(frame['path']) as opened:
                    if pixel_sha256(opened.convert('RGB')) != frame['pixel_sha256']:
                        raise ValueError('source pixel mismatch')
            attempt(folder.name)
            results = discovery.discover_batch([f['path'] for f in group['frames']],
                [f['frame_id'] for f in group['frames']], question=group['question'])
            write(folder/'results.json', results)
            for result in results:
                outcomes.append(dict(stage=folder.name, frame_id=result['frame_id'], status=result['status'],
                    detections=len(result['detections']), output_tokens=result['output_tokens']))
        candidate = None
        qp = plan['qa']
        for label,spec in [('draft',qp['draft']),('joint',qp['snapshot'])]:
            folder = gpu/label; folder.mkdir(); capture(folder)
            messages = plan['draft_messages'] if label=='draft' else verification_messages(plan['joint_messages'],candidate)
            write(folder/'messages.json', messages)
            images = [Image.open(p).convert('RGB') for p in spec['paths']]
            if [pixel_sha256(im) for im in images] != spec['pixel_sha256']:
                raise ValueError('QA pixel mismatch')
            attempt(label)
            generated = answerer.answer(messages, images, QABudget(**spec['limits']),
                note_boundary=POLICY if label=='joint' else None)
            write(folder/'generation.json', generated)
            traces = [t for t in generated.get('actual_forward_trace',[]) if t['component']=='actual_spatial_encoder']
            if not traces or len(traces[0]['shapes']) != len(images):
                raise ValueError('not all actual QA images entered Spatial encoder')
            parsed = candidate_from_generation(generated); write(folder/'parsed.json',parsed)
            import re
            choices = re.findall(r'^([A-Z])\. ',qp['source']['original_question'],flags=re.M)
            ok = parsed['status']=='ok' and (not choices or parsed['answer'] in choices)
            outcomes.append(dict(stage=label,status='ok' if ok else 'format_failed',
                output_tokens=generated.get('output_tokens'),actual_images=len(images)))
            if label=='draft':candidate=parsed
            else:write(folder/'decision.json',decide(candidate,generated,
                [m['frame_ids'][0] for m in qp['source']['media'] if m['kind']=='source_frame']))
            for image in images:image.close()
        write(gpu/'completion.json',dict(status='passed' if all(r['status']=='ok' for r in outcomes) else 'output_contract_failed',
            outcomes=outcomes,generation_requests=4,logical_discovery_requests=4,QA_calls=2,
            formal_scores=None,full_spatial_migration=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    except Exception:
        import traceback
        write(gpu/'failure.json',dict(error=traceback.format_exc(),outcomes=outcomes,retry_allowed=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    (prepare if args.action=='prepare' else run)(args.output.resolve())
