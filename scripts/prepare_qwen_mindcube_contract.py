"""Freeze the existing 527 inputs and one output-contract repair, before scores."""
import argparse, hashlib, json, sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_qwen_half_bench import read, write, sha
PARENT=ROOT/'work/qwen_mindcube_viewfix_20261107_01'


def input_hash(item):return hashlib.sha256(json.dumps(item,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def component_plan(m):
    pools={kind:[] for kind in ['wrong_frame','bbox','output_limit','incomplete_json']}
    drafts={n:[] for n in (3,4)}
    for item in m['inputs']:
        folder=PARENT/('question_'+item['id']);key=input_hash(item)
        for p in folder.glob('observation_F*/discovery.json'):
            d=read(p);error=d.get('error','')
            if d['status']=='ok':continue
            category=('wrong_frame' if 'wrong frame identity' in error else 'bbox' if 'four finite' in error else
                      'output_limit' if 'output limit' in error else 'incomplete_json' if 'complete JSON' in error else None)
            if category:
                fid=d['frame_id'];path=p.parent/f'{fid:06d}.jpg'
                pools[category].append((key,str(p),dict(kind='discovery',question_id=item['id'],frame_id=fid,
                    path=str(path),question=item['formatted_question'],old_failure=category,
                    file_sha256={str(path):sha(path),str(p):sha(p)})))
        p=folder/'draft_generation.json'
        if p.exists() and read(p).get('reached_output_limit') and len(item['images']) in drafts:
            source=folder/'draft_messages.json';paths=[folder/f'qa_frame_{i:06d}.png' for i in range(len(item['images']))]
            drafts[len(paths)].append((key,dict(kind='draft',question_id=item['id'],messages=str(source),
                image_paths=list(map(str,paths)),limits=read(p)['budget']['limits'],old_failure='draft_output_limit',
                file_sha256={str(q):sha(q) for q in [source,*paths,p]})))
    cases=[min(pool,key=lambda v:(v[0],v[1]))[2] for pool in pools.values()]
    cases += [min(pool,key=lambda v:v[0])[1] for pool in drafts.values()]
    return dict(selection='minimum unchanged-input SHA256 per observed syntax-failure category, then path tie-break; no correctness criteria',
        cases=cases,logical_calls=6,retries=0,answers_used=False)


def prepare(out):
    if out.exists():raise FileExistsError(out)
    parent=read(PARENT/'manifest.json')
    for p,h in parent['source_sha256'].items():assert sha(ROOT/p)==h,p
    out.mkdir(parents=True);runtime=out/'mindcube_contract_runtime';runtime.mkdir()
    m=deepcopy(parent);m.update(scope='MindCube_Qwen_output_contract_repair_v1',parent_run=str(PARENT),
        parent_manifest_sha256=sha(PARENT/'manifest.json'),cold_perception_rerun=True,prediction_reuse=False,
        method_core_unchanged=True,config_unchanged_from_parent=True,
        output_contract=dict(policy='qwen_mindcube_output_contract_v1',discovery='strict original schema; fixed actual frame ID; original 8-word descriptions; valid nonduplicate boxes; close only complete records within 512 tokens',
            name_word_limit=None,qa='existing 100-word/64-token final reserve also applied to independent candidate; final answer constrained to every visible option letter',
            original_input_messages_and_budgets=True,no_new_evidence_or_identity_rules=True,retries=0),
        score_policy='One complete same-527 measurement after fixed six component calls and original two end-to-end gates; no score-driven changes')
    sources=dict(parent['source_sha256'])
    for p in ['sandbox_vlm/backbones/qwen_mindcube_contract.py','scripts/prepare_qwen_mindcube_contract.py',
              'scripts/accept_qwen_mindcube_contract.py','scripts/run_qwen_mindcube_contract.py',
              'scripts/run_qwen_mindcube_contract.sbatch','tests/test_qwen_mindcube_contract.py']:
        sources[p]=sha(ROOT/p)
    # Grammar implementation and its local import dependencies are frozen too.
    for p in (ROOT/'sandbox_vlm/backbones').glob('*.py'):sources[str(p.relative_to(ROOT))]=sha(p)
    m['source_sha256']=sources
    for p,h in sources.items():
        dst=out/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes());assert sha(dst)==h
    (out/'native_sam').mkdir();(out/'native_sam/_C.so').write_bytes((PARENT/'native_sam/_C.so').read_bytes())
    (out/'mind_references.json').write_bytes((PARENT/'mind_references.json').read_bytes())
    m['reference_sha256']={str(out/'mind_references.json'):sha(out/'mind_references.json')}
    plan=component_plan(m);write(out/'component_gate_plan.json',plan)
    m['component_gate_plan_sha256']=sha(out/'component_gate_plan.json')
    for item in m['inputs']:
        folder=out/('question_'+item['id']);folder.mkdir()
        write(folder/'reservation.json',dict(owner='mindcube_contract_runtime',question_id=item['id']))
    assert len(m['inputs'])==527 and m['inputs']==parent['inputs'] and m['config']==parent['config']
    assert (out/'mind_references.json').read_bytes()==(PARENT/'mind_references.json').read_bytes()
    write(out/'manifest.json',m);write(out/'freeze.json',dict(manifest_sha256=sha(out/'manifest.json'),source_files=len(sources),model_calls=0))
    prior=read(PARENT/'mindcube_viewfix_runtime/hardware_17565.json')
    write(runtime/'authorization.json',dict(manifest_sha256=sha(out/'manifest.json'),
        launcher_sha256=sha(ROOT/'scripts/run_qwen_mindcube_contract.py'),sbatch_sha256=sha(ROOT/'scripts/run_qwen_mindcube_contract.sbatch'),
        measured_MindCube_peak_reserved_bytes=prior['measured_previous_peak_reserved_bytes'],
        user_authorized='2026-11-08: repair internal MindCube evaluation failures and rerun; independent output-contract variant',
        cold_full_model=True,old_predictions_reused=False,no_prompt_search=True))
    import prepare_qwen_mindcube_viewfix as old
    old.PARENT=PARENT
    old.preflight(out,m,m['inputs'])
    print(json.dumps(dict(output=str(out),questions=527,component_calls=6,end_to_end_gates=m['gate_question_ids'],source_files=len(sources),model_calls=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output.resolve())
