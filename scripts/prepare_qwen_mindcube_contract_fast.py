"""Freeze equivalent parser acceleration; preserve completed component receipts."""
import argparse,json,sys,time
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_qwen_half_bench import read,write,sha
PARENT=ROOT/'work/qwen_mindcube_contract_v2_20261108_01'


def prepare(out):
    if out.exists():raise FileExistsError(out)
    m=deepcopy(read(PARENT/'manifest.json'))
    for p,h in m['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert read(PARENT/'cpu_failure_replay.json')['all_observed_failures']==45
    assert (PARENT/'component_gate/acceptance.json').exists(),'wait for all old in-flight calls to finish at marked boundaries'
    assert (PARENT/'intentional_component_boundary_stop.json').exists()
    out.mkdir(parents=True);runtime=out/'mindcube_contract_fast_runtime';runtime.mkdir()
    plan=deepcopy(read(PARENT/'component_gate_plan.json'));reuse=0
    from sandbox_vlm.perception.discovery import parse_discovery
    from lmformatenforcer import JsonSchemaParser,CharacterLevelParserConfig
    from sandbox_vlm.backbones.spatial_syntax import discovery_schema,SyntaxTokens
    from transformers import AutoTokenizer
    checkpoint=m['config']['qa']['mllm']
    syntax=SyntaxTokens(AutoTokenizer.from_pretrained(checkpoint['model_id'],revision=checkpoint['revision'],local_files_only=True))
    from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import ClosedStringJSON
    from sandbox_vlm.backbones.qwen_mindcube_contract_fast import CachedClosedJSON
    for i,case in enumerate(plan['cases']):
        old=PARENT/'component_gate'/f'{i:03d}_{case["kind"]}'/'generation.json'
        if not old.exists():continue
        result=read(old);assert result['status'] in ('ok','generated')
        if case['kind']=='discovery':
            parse_discovery(result['raw_output'],case['frame_id'],tuple(result['image_size_hw']))
            cfg=syntax.parser_config(512)
            a=ClosedStringJSON(JsonSchemaParser(discovery_schema(case['frame_id']),config=cfg))
            b=CachedClosedJSON(JsonSchemaParser(discovery_schema(case['frame_id']),config=cfg))
            for c in result['raw_output']:
                assert a.get_allowed_characters()==b.get_allowed_characters()
                assert c in b.get_allowed_characters()
                a,b=a.add_character(c),b.add_character(c)
            assert a.can_end() and b.can_end()
        case['equivalent_reuse']=dict(path=str(old),sha256=sha(old),original_actual_inputs=str(old.parent/'actual_inputs'),
            original_input_sha256=case['file_sha256'],policy='identical closed-string v2 language; parser traversal only')
        reuse+=1
    write(out/'component_gate_plan.json',plan)
    m.update(parent_run=str(PARENT),parent_manifest_sha256=sha(PARENT/'manifest.json'),
        source_optimisation='cached immutable parser state and bulk plain-character transitions; v2 admissible tokens unchanged',
        component_gate_plan_sha256=sha(out/'component_gate_plan.json'),
        equivalent_component_validation_reuse=reuse,formal_prediction_or_perception_reuse=False)
    sources=dict(m['source_sha256'])
    for p in ['sandbox_vlm/backbones/qwen_mindcube_contract_fast.py','scripts/prepare_qwen_mindcube_contract_fast.py',
        'scripts/accept_qwen_mindcube_contract_fast.py','scripts/run_qwen_mindcube_contract_fast.py',
        'scripts/run_qwen_mindcube_contract_fast.sbatch','tests/test_qwen_mindcube_contract_fast.py']:
        sources[p]=sha(ROOT/p)
    m['source_sha256']=sources
    for p,h in sources.items():
        target=out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes());assert sha(target)==h
    for name in ['cpu_preflight.json','cpu_failure_replay.json']:(out/name).write_bytes((PARENT/name).read_bytes())
    (out/'native_sam').mkdir();(out/'native_sam/_C.so').write_bytes((PARENT/'native_sam/_C.so').read_bytes())
    (out/'mind_references.json').write_bytes((PARENT/'mind_references.json').read_bytes())
    m['reference_sha256']={str(out/'mind_references.json'):sha(out/'mind_references.json')}
    for item in m['inputs']:
        folder=out/('question_'+item['id']);folder.mkdir()
        write(folder/'reservation.json',dict(owner='mindcube_contract_fast_runtime',question_id=item['id']))
    write(out/'manifest.json',m);write(out/'freeze.json',dict(manifest_sha256=sha(out/'manifest.json'),source_files=len(sources),model_calls=0))
    auth=read(PARENT/'mindcube_contract_v2_runtime/authorization.json')
    auth.update(manifest_sha256=sha(out/'manifest.json'),launcher_sha256=sha(ROOT/'scripts/run_qwen_mindcube_contract_fast.py'),
        sbatch_sha256=sha(ROOT/'scripts/run_qwen_mindcube_contract_fast.sbatch'))
    write(runtime/'authorization.json',auth)
    write(out/'equivalence_receipt.json',dict(status='passed',completed_component_transcripts_character_masks_identical=reuse,
        native_token_mask_test='tests/test_qwen_mindcube_contract_fast.py',source_test_sha256=sha(ROOT/'tests/test_qwen_mindcube_contract_fast.py'),
        component_cases=plan['logical_calls'],new_component_calls=plan['logical_calls']-reuse,
        formal_questions=527,formal_reuse=0,old_allocation_receipts=[str(p) for p in (PARENT/'mindcube_contract_v2_runtime').glob('allocation_*.json')]))
    print(json.dumps(dict(output=str(out),source_files=len(sources),component_reuse=reuse,new_component_calls=plan['logical_calls']-reuse,formal_questions=527)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output.resolve())
