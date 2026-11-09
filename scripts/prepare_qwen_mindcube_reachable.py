"""Isolated repair freeze: same 527 inputs, no predictions/perception reused."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_qwen_half_bench import read,write,sha

PARENT=ROOT/'work/qwen_mindcube_contract_total_20261108_01'


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    m=deepcopy(read(PARENT/'manifest.json'))
    for p,h in m['source_sha256'].items():
        assert sha(ROOT/p)==h,p
    assert (PARENT/'mindcube_contract_total_runtime/failure.json').exists()
    plan=deepcopy(read(PARENT/'component_gate_plan.json'))
    for case in plan['cases']:
        case.pop('equivalent_reuse',None)
    q=PARENT/'question_among_group279_q1_1_1'
    image=q/'observation_F000001/000001.jpg'
    failed=q/'observation_F000001/discovery.json'
    assert read(failed)['status']=='perception_failed'
    case=dict(kind='discovery',question_id='among_group279_q1_1_1',frame_id=1,
              question=read(q/'input.json')['formatted_question'],path=str(image),
              old_failure='unbounded punctuation-separated repetition within one whitespace word',
              file_sha256={str(image):sha(image)})
    plan['cases'].insert(0,case)
    plan['logical_calls']=len(plan['cases'])
    plan['selection']='Original fixed 51 failure/format cases plus latest actual punctuation-loop failure; no answer correctness selection; all 52 freshly executed'
    plan['added_failure_raw_output']=dict(path=str(failed),sha256=sha(failed))
    out.mkdir(parents=True)
    runtime=out/'mindcube_reachable_runtime';runtime.mkdir()
    write(out/'component_gate_plan.json',plan)
    sources=dict(m['source_sha256'])
    for p in ['sandbox_vlm/backbones/qwen_mindcube_reachable.py',
              'scripts/prepare_qwen_mindcube_reachable.py','scripts/accept_qwen_mindcube_reachable.py',
              'scripts/run_qwen_mindcube_reachable.py','scripts/run_qwen_mindcube_reachable.sbatch',
              'scripts/continue_mindcube_reachable.py','scripts/continue_mindcube_reachable.sbatch',
              'scripts/qwen_d116_mindcube_reachable.py',
              'tests/test_qwen_mindcube_reachable.py']:
        sources[p]=sha(ROOT/p)
    m.update(parent_run=str(PARENT),parent_manifest_sha256=sha(PARENT/'manifest.json'),
             source_optimisation='original numeric/string contract plus constructive remaining-token feasibility mask',
             component_gate_plan_sha256=sha(out/'component_gate_plan.json'),
             equivalent_component_validation_reuse=0,formal_prediction_or_perception_reuse=False,
             source_sha256=sources)
    m['output_contract'].update(policy='qwen_mindcube_reachable_json_v1',
        output_cap=512,fields_filled=False,raw_output_repaired=False,
        constraint='single-call prefix feasibility including required fields and EOS; ASCII singleton completion witnesses; witnesses never emitted',
        name_word_limit=None,numeric_precision_limit=None,
        old_complete_record_closure_retained=True,per_question_retry=False)
    for p,h in sources.items():
        target=out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/p).read_bytes());assert sha(target)==h
    for name in ['cpu_preflight.json','cpu_failure_replay.json']:
        (out/name).write_bytes((PARENT/name).read_bytes())
    (out/'native_sam').mkdir()
    (out/'native_sam/_C.so').write_bytes((PARENT/'native_sam/_C.so').read_bytes())
    (out/'mind_references.json').write_bytes((PARENT/'mind_references.json').read_bytes())
    m['reference_sha256']={str(out/'mind_references.json'):sha(out/'mind_references.json')}
    for item in m['inputs']:
        folder=out/('question_'+item['id']);folder.mkdir()
        write(folder/'reservation.json',dict(owner='mindcube_reachable_runtime',question_id=item['id']))
    write(out/'manifest.json',m)
    write(out/'freeze.json',dict(manifest_sha256=sha(out/'manifest.json'),source_files=len(sources),model_calls=0))
    auth=read(PARENT/'mindcube_contract_total_runtime/authorization.json')
    auth.update(manifest_sha256=sha(out/'manifest.json'),launcher_sha256=sha(ROOT/'scripts/run_qwen_mindcube_reachable.py'),
                sbatch_sha256=sha(ROOT/'scripts/run_qwen_mindcube_reachable.sbatch'))
    write(runtime/'authorization.json',auth)
    print(dict(output=str(out),source_files=len(sources),component_calls=plan['logical_calls'],formal_questions=527,formal_reuse=0),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.output.resolve())
