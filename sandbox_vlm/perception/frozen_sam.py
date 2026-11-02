"""Declared frozen D053 perception inputs; no QA/result/target loading."""
from pathlib import Path
import json
from sandbox_vlm.perception.frozen import file_sha
from sandbox_vlm.memory.coverage import CoverageSandbox
from sandbox_vlm.qa.question_local import select_base_frames

BASE_AUX_FIELDS={'gated_frame_ledger','discovered_frame_bank','last_shortlist_state','prefer_full_next_batch',
                 'current_watermark','frame_mapping','answer_frame_ids','question_key'}


def load_sam_query(record,item,config):
    values={}
    for kind,name in [('base','base_sandbox.json'),('sam_auxiliary','ledger.json')]:
        p=Path(item[kind+'_path'])
        if p.name!=name or file_sha(p)!=item[kind+'_sha256']:
            raise ValueError('frozen SAM/base path or content changed')
        values[kind]=json.loads(p.read_text())
    base=CoverageSandbox.from_dict(values.pop('base'));aux=values.pop('sam_auxiliary')
    if set(aux)!=BASE_AUX_FIELDS|{'answer_working_sandbox','short_track','association_application'}:
        raise ValueError('unexpected frozen SAM fields')
    if aux['association_application']!='question_local_no_base_writeback':raise ValueError('not D052 SAM state')
    sam=CoverageSandbox.from_dict(aux.pop('answer_working_sandbox'))
    q=item['query_frame_id']
    expected=CoverageSandbox(base.video_id,record['original_question']).question_key
    if base.question_key!=expected or sam.question_key!=expected or aux['question_key']!=expected or aux['current_watermark']!=q:
        raise ValueError('frozen question/prefix mismatch')
    if base.video_id!=sam.video_id or any(int(fid)>q for state in [base,sam] for fid in state.viewpoints):
        raise ValueError('future/cross-video observation')
    ledger={int(fid):row for fid,row in aux['gated_frame_ledger'].items()}
    if not ledger or min(ledger)<0 or max(ledger)>q:raise ValueError('future/empty ledger')
    bank=aux['discovered_frame_bank']
    if any(fid not in ledger or fid>q for fid in bank):raise ValueError('invalid discovered bank')
    sparse=sum(v['margin']>0 for v in ledger.values())/len(ledger)<=config['coverage']['discovery']['sparse_fraction']
    chosen=select_base_frames(base,ledger,bank,sparse=sparse,config=config)
    if chosen!=aux['answer_frame_ids']:raise ValueError('frozen selection changed')
    return base,aux,sam,chosen,sparse


def reads_memory(sam,chosen,sparse,config):
    return not(config['coverage'].get('omit_redundant_uncertain_memory',False) and sparse
        and set(map(int,sam.viewpoints))<=set(chosen) and bool(sam.ambiguities))
