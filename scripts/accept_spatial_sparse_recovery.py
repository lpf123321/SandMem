"""Fixed four-frame budget-closure acceptance; no answers or scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))


def prepare(out):
    source=ROOT/'work/spatial_discovery_acceptance_20261106_01'
    plan=json.loads((source/'plan.json').read_text());config=json.loads((source/'config.json').read_text())
    plan.update(authorization='2026-11-06 user approved field bounds and exact-box uniqueness; subsequent continue for format repair',
        policy='budget_closed_discovery_v3',name_word_limit=8,description_word_limit=8,
        exact_box_duplicates='forbidden in same-frame output; no identity correspondence inferred',
        generation_batch_cap=2,logical_discovery_cap=4,retries=0,formal_scores=None)
    proof_path=ROOT/'work/spatial_dense_syntax_replay_20261106_02/completion.json'
    proof=json.loads(proof_path.read_text())
    if proof['status']!='passed':raise ValueError('dense greedy-token equivalence required')
    for p,h in proof['files'].items():
        if sha(p)!=h:raise ValueError('dense equivalence dependency changed: '+p)
    plan.update(generation_batch_cap=1,logical_discovery_cap=2,reused_dense_logical_requests=2,
        dense_reuse_proof=str(proof_path),dense_source_results=str(Path(proof['source'])/'gpu/dense/results.json'))
    out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan);write(out/'config.json',config)
    paths=[source/'plan.json',source/'config.json',Path(__file__)]
    paths+=list((ROOT/'sandbox_vlm/backbones').glob('*.py'))
    paths+=[ROOT/n for n in json.loads((ROOT/'provenance/frozen_files.json').read_text())['source_sha256']]
    paths+=[Path(f['path']) for group in plan['groups'] for f in group['frames']]
    paths+=[proof_path]+[Path(p) for p in proof['files']]
    paths+=list(out.glob('*.json'))
    write(out/'manifest.json',dict(files={str(p.resolve()):sha(p) for p in paths},status='prepared_not_run'))


def run(out):
    for path,h in json.loads((out/'manifest.json').read_text())['files'].items():
        if sha(path)!=h:raise ValueError('prepared file changed: '+path)
    import torch
    from PIL import Image
    from sandbox_vlm.backbones.spatial_transport import SpatialBackbone
    from sandbox_vlm.backbones.spatial_syntax import SyntaxTokens
    from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetGrammar,SpatialBudgetedDiscovery
    from sandbox_vlm.backbones.spatial_inputs import input_record
    from sandbox_vlm.qa.artifact import pixel_sha256
    plan=json.loads((out/'plan.json').read_text());gpu=out/'gpu';gpu.mkdir(exist_ok=False)
    write(gpu/'started.json',dict(manifest_sha256=sha(out/'manifest.json')))
    outcomes=[];started=time.perf_counter()
    try:
        torch.set_num_threads(1)
        bundle=SpatialBackbone(json.loads((out/'config.json').read_text()),ROOT/'work/spatial-mllm-official')
        grammar=BudgetGrammar(SyntaxTokens(bundle.runtime.processor.tokenizer),512,bundle.runtime.processor.tokenizer);grammar.trace_path=gpu/'decode_steps.jsonl'
        discovery=SpatialBudgetedDiscovery(bundle.discovery.config,bundle.runtime,grammar)
        write(gpu/'hardware.json',dict(name=torch.cuda.get_device_name(0),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory))
        for group in plan['groups']:
            folder=gpu/group['label'];folder.mkdir()
            for frame in group['frames']:
                with Image.open(frame['path']) as opened:
                    if pixel_sha256(opened.convert('RGB'))!=frame['pixel_sha256']:raise ValueError('source pixels changed')
            def capture(texts,batch):
                torch.save(dict(batch),folder/'actual_inputs.pt')
                write(folder/'actual_inputs.json',dict(texts=texts,tensors={k:input_record(v) for k,v in batch.items()}))
            if group['label']=='dense':
                proof=json.loads(Path(plan['dense_reuse_proof']).read_text())
                results=json.loads(Path(plan['dense_source_results']).read_text())
                if [r['frame_id'] for r in results]!=[f['frame_id'] for f in group['frames']]:raise ValueError('dense frame identity mismatch')
                write(folder/'results.json',results);write(folder/'reuse.json',proof)
                for result in results:
                    outcomes.append(dict(stage='dense',frame_id=result['frame_id'],status=result['status'],detections=len(result['detections']),
                        output_tokens=result['output_tokens'],reuse='prior GPU output with full stricter-grammar token replay; no new forward'))
                continue
            bundle.runtime.transport_processor.audit_sink=capture
            with (gpu/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(dict(stage=group['label'],generation_requests=1))+'\n')
            results=discovery.discover_batch([f['path'] for f in group['frames']],
                [f['frame_id'] for f in group['frames']],question=group['question'])
            write(folder/'results.json',results)
            for result in results:
                if result['status']=='ok':
                    obj=json.loads(result['raw_output'])['objects'];boxes=[]
                    for row in obj:
                        if len(row['name'].split())>8 or len(row['description'].split())>8:raise ValueError('word bound escaped grammar')
                        box=tuple(row['bbox_2d'])
                        if box in boxes:raise ValueError('exact box duplicate escaped grammar')
                        boxes.append(box)
                outcomes.append(dict(stage=group['label'],frame_id=result['frame_id'],status=result['status'],
                    detections=len(result['detections']),output_tokens=result['output_tokens'],error=result.get('error')))
        write(gpu/'completion.json',dict(status='passed' if all(r['status']=='ok' for r in outcomes) else 'output_contract_failed',
            outcomes=outcomes,generation_requests=1,logical_discovery_requests=2,reused_dense_logical_requests=2,total_checked_frames=4,formal_scores=None,
            full_spatial_migration=False,seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    except Exception:
        import traceback
        write(gpu/'failure.json',dict(error=traceback.format_exc(),outcomes=outcomes,retry_allowed=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.output.resolve())
