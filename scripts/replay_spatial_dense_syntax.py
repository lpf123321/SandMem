"""Verify prior successful greedy tokens survive the stricter grammar.

This reads only the earlier input/output/decoder trace, never answers or scores.
No model forward. Removing only unselected tokens preserves the previous greedy
choice for the same logits; this is not an independent GPU numerical rerun.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replay(out):
    import torch
    from transformers import Qwen2_5_VLProcessor
    from huggingface_hub import snapshot_download
    from lmformatenforcer import JsonSchemaParser
    from sandbox_vlm.backbones.spatial_model import MODEL_ID,REVISION
    from sandbox_vlm.backbones.spatial_syntax import SyntaxTokens,discovery_schema
    from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetGrammar
    source=ROOT/'work/spatial_budget_closed_acceptance_20261106_01'
    rows=json.loads((source/'gpu/dense/results.json').read_text())
    if len(rows)!=2 or any(r['status']!='ok' for r in rows):raise ValueError('previous dense acceptance required')
    prefixes={16:[],89:[]}
    for line in (source/'gpu/decode_steps.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['event']=='start' and row['frame_id'] in prefixes and len(row['output_ids'])>len(prefixes[row['frame_id']]):
            prefixes[row['frame_id']]=row['output_ids']
    checkpoint=snapshot_download(MODEL_ID,revision=REVISION,local_files_only=True)
    tokenizer=Qwen2_5_VLProcessor.from_pretrained(checkpoint,local_files_only=True,use_fast=False).tokenizer
    grammar=BudgetGrammar(SyntaxTokens(tokenizer),512,tokenizer)
    batch=torch.load(source/'gpu/dense/actual_inputs.pt',map_location='cpu',weights_only=True)
    out.mkdir(parents=True,exist_ok=False);verified=[];started=time.perf_counter()
    for index,row in enumerate(rows):
        tokens=prefixes[row['frame_id']]
        eos=tokenizer.eos_token_id
        if eos in tokens:tokens=tokens[:tokens.index(eos)+1]
        elif len(tokens)==row['output_tokens']-1:tokens=[*tokens,eos]
        else:raise ValueError('incomplete actual token trace')
        if len(tokens)!=row['output_tokens']:raise ValueError('output length mismatch')
        raw=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        if raw!=row['raw_output']:raise ValueError('trace does not reconstruct exact raw output')
        function=grammar.prefix(JsonSchemaParser(discovery_schema(row['frame_id']),config=grammar.parser_config(512)))
        ids=batch['input_ids'][index].tolist()
        for n,token in enumerate(tokens):
            allowed=function(0,torch.tensor(ids))
            if token not in allowed:raise ValueError('old selected token disallowed: '+str((row['frame_id'],n,token)))
            ids.append(token)
        verified.append(dict(frame_id=row['frame_id'],tokens=len(tokens),all_selected_tokens_allowed=True,
            raw_sha256=hashlib.sha256(raw.encode()).hexdigest()))
        print('verified dense frame',row['frame_id'],len(tokens),flush=True)
    paths=[source/'gpu/dense/results.json',source/'gpu/dense/actual_inputs.pt',source/'gpu/decode_steps.jsonl',
        ROOT/'sandbox_vlm/backbones/spatial_budgeted_discovery.py',Path(__file__)]
    result=dict(status='passed',verified=verified,model_calls=0,seconds=time.perf_counter()-started,
        source=str(source),reuse_label='prior dense GPU outputs; stricter-mask greedy-token equivalence replay',
        proof='new constraint only removes infeasible unselected tokens; old argmax remains legal for the same logits',
        limitation='not a new GPU forward or cross-run floating-point reproducibility test',
        files={str(p.resolve()):sha(p) for p in paths})
    (out/'completion.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();replay(a.output.resolve())
