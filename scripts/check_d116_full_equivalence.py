"""Replay all existing D116 evidence construction on CPU; no model calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_evidence_binding import read,write,sha


def check(output):
    from sandbox_vlm.qa.d116_full import compose,draw,joint_messages
    from sandbox_vlm.qa.map_only import indexed_messages
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.data.vsi_super import load_super_inputs
    d116=ROOT/'work/bench/d116_two_pass_20261101_01';m=read(d116/'manifest.json')
    d102=ROOT/'work/bench/d102_expanded_development_20261029_01'
    d106=ROOT/'work/bench/d106_observation_repair_20261031_02'
    baseline=Path(m['baseline']);tables={};checks=[];start=time.perf_counter()
    if output.exists():raise FileExistsError(output)
    for job in m['jobs']:
        key=job['variant'],job['duration']
        if key not in tables:tables[key]=load_super_inputs(ROOT/'work/protocol/vsi_super'/key[0]/f'test_{key[1]}.parquet',variant=key[0])
        example=tables[key][int(job['sample_id'].split(':')[1])];assert example.sample_id==job['sample_id']
        src=d116/job['folder'];s=read(src/('base_input.json'if job['new_QA_required']else'input.json'))
        before=read(src/('base_messages.json'if job['new_QA_required']else'messages.json'))
        ids=[int(c['text'].split('ID ')[1].rstrip(':'))for c in before[0]['content']
             if c['type']=='text'and c['text'].startswith('Actual source frame ID ')]
        state=read(baseline/job['folder']/'visual_sandbox.json')
        frozen=read(d102/job['folder']/'control/geometry.json')
        edges=read(d102/job['folder']/'new_unions.json')['edges']
        packets=read(d106/job['folder']/'geometry.json').get('packets',[])if s['sparse']else[]
        payload,geometry,audit=compose(state,frozen,packets,edges,ids,job['query_frame_id'],example.original_question)
        expected=json.loads(before[0]['content'][3]['text'].split('\n',1)[1]);assert payload==expected,(job['record'],'payload')
        image=draw(geometry)
        try:
            digest=pixel_sha256(image);assert digest==s['pixel_sha256'][0],(job['record'],'map pixels')
            images=[object()for _ in ids]
            messages,joint=joint_messages(example.model_question,s['sparse'],ids,payload,images,image)
            serial=indexed_messages(messages,joint,[digest,*s['pixel_sha256'][1:]])
            assert serial==before,(job['record'],'messages')
        finally:image.close()
        checks.append(dict(record=job['record'],query_frame_id=job['query_frame_id'],payload_exact=True,
            map_pixels_exact=True,messages_exact=True))
        if len(checks)%25==0:print('CPU_EQUIVALENCE',len(checks),flush=True)
    write(output,dict(status='passed',queries=len(checks),checks=checks,new_model_calls=0,
        seconds=time.perf_counter()-start,D116_manifest_sha256=sha(d116/'manifest.json')))
    print(json.dumps(dict(status='passed',queries=len(checks),seconds=time.perf_counter()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    check(p.parse_args().output.resolve())
