"""CPU-only mask replay of the previously fixed four discovery inputs.

Reads actual token traces, never benchmark labels. Saves every allowed-token
mask digest, including exceptions/empty masks, to test implementation equality.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def replay(out, steps=0, optimized=False, limit=0):
    import torch
    from transformers import Qwen2_5_VLProcessor
    from huggingface_hub import snapshot_download
    from lmformatenforcer import JsonSchemaParser
    from sandbox_vlm.backbones.spatial_model import MODEL_ID, REVISION
    from sandbox_vlm.backbones.spatial_syntax import SyntaxTokens, discovery_schema
    from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetGrammar
    torch.set_num_threads(1)
    source = ROOT / 'work/spatial_budget_closed_acceptance_20261106_01'
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((source / 'plan.json').read_text())
    prefixes = {f['frame_id']: [] for g in plan['groups'] for f in g['frames']}
    for line in (source / 'gpu/decode_steps.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['event'] == 'start' and len(row['output_ids']) > len(prefixes[row['frame_id']]):
            prefixes[row['frame_id']] = row['output_ids']
    checkpoint = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    tokenizer = Qwen2_5_VLProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False).tokenizer
    started = time.perf_counter()
    grammar = BudgetGrammar(SyntaxTokens(tokenizer), 512, tokenizer)
    if optimized:
        from sandbox_vlm.backbones.spatial_fast_syntax import install_fast_discovery_grammar
        install_fast_discovery_grammar(grammar)
    init_seconds = time.perf_counter() - started
    profiler = cProfile.Profile()
    profiler.enable()
    receipts = []
    with (out / 'masks.jsonl').open('w') as stream:
        for group in sorted(plan['groups'], key=lambda g: g['label']):
            batch = torch.load(source / 'gpu' / group['label'] / 'actual_inputs.pt', map_location='cpu', weights_only=True)
            results = json.loads((source / 'gpu' / group['label'] / 'results.json').read_text())
            for i, result in enumerate(results):
                if limit and len(receipts) >= limit:
                    break
                fid = result['frame_id']
                tokens = prefixes[fid]
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                elif len(tokens) == result['output_tokens'] - 1 and result['status'] == 'ok':
                    tokens = tokens + [tokenizer.eos_token_id]
                decoded = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                assert result['raw_output'].strip().startswith(decoded.strip())
                function = grammar.prefix(JsonSchemaParser(discovery_schema(fid), config=grammar.parser_config(512)))
                ids = batch['input_ids'][i].tolist()
                started = time.perf_counter()
                for n, token in enumerate(tokens):
                    if steps and n >= steps:
                        break
                    values = function(0, torch.tensor(ids))
                    values = sorted(set(values))
                    digest = hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
                    stream.write(json.dumps(dict(frame_id=fid, step=n, count=len(values), sha256=digest,
                        selected_token_allowed=token in values)) + '\n')
                    ids.append(token)
                stream.flush()
                receipts.append(dict(frame_id=fid, steps=min(steps or len(tokens), len(tokens)),
                    seconds=time.perf_counter() - started))
                print(receipts[-1], flush=True)
    profiler.disable()
    profiler.dump_stats(str(out / 'profile.pstats'))
    with (out / 'profile.txt').open('w') as stream:
        pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(35)
    (out / 'receipt.json').write_text(json.dumps(dict(optimized=optimized, cases=receipts,
        initialization_seconds=init_seconds, model_calls=0, answers_read=False), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--optimized', action='store_true')
    a = p.parse_args()
    replay(a.output.resolve(), a.steps, a.optimized, a.limit)
