"""D065 signatures and charged resident state, independent of old answers."""
from copy import deepcopy
import hashlib,json
from sandbox_vlm.memory.storage import cpu_storage_report


def signature(messages,metadata,config,limits):
    items=[];index=0
    for message in messages:
        row=dict(role=message['role'],content=[])
        for part in message['content']:
            if part['type']=='image':row['content'].append(dict(type='image',source=metadata[index]));index+=1
            else:row['content'].append(deepcopy(part))
        items.append(row)
    if index!=len(metadata):raise ValueError('image/message identity mismatch')
    value=dict(messages=items,processor=config['qa']['mllm'],limits=limits,
        generation=dict(do_sample=False,max_new_tokens=limits['max_output_tokens']))
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def bundle(aux,state,history,audit,catalog,attempted,schedule):
    return dict(frozen_SAM=aux,candidate_state=state.to_dict(),joint_geometry=history,union_audit=audit,
                mask_catalog=catalog,attempted_pair_keys=sorted(attempted),record_schedule=schedule)


def measure(base,aux,sam,state,history,audit,catalog,attempted,schedule):
    encoded=len(json.dumps(dict(visual_sandbox=base.to_dict(),auxiliary_state=bundle(aux,state,history,audit,catalog,attempted,schedule)),
        ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
    host=cpu_storage_report(dict(base=base,sam_state=sam,candidate=state,auxiliary=aux,geometry=history,audit=audit,
        catalog=catalog,attempted=attempted,schedule=schedule))
    return dict(encoded_bytes=encoded,cpu_structural_bytes=host['cpu_structural_bytes'],combined_bytes=encoded+host['cpu_structural_bytes'])
