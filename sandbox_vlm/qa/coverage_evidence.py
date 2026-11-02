"""Question-provenance retrieval with full actual links and budgeted serialization.

All visual hypotheses stay in the input. Description verbosity can decrease;
IDs, names and representative source-frame links are never silently dropped.
No benchmark parser, counting routine or identity decision is implemented here.
"""
import hashlib
import json

import numpy as np

from sandbox_vlm.perception.coverage import unpack_descriptor


MEMORY_NOTICE = (
    '\nThe following records are visual identity hypotheses from actual segmented observations. '
    'They may split one physical object or merge similar ones. Verify against the real frames; '
    'hypothesis IDs are not ground truth physical counts. Geometry is unavailable for this run. '
    'Frames and local references do not establish a shared 3D layout. '
    'Actual frame IDs specify observation order. Background-only observations do not establish '
    'that the main entity in the question appeared in that frame.\nVisual memory (JSON data):\n')


def append_evidence(messages, state, question, tokenizer, *, max_text_tokens, reserved_output_tokens,
                    images, max_notes=16, chat_template=None, notice=MEMORY_NOTICE):
    if hashlib.sha256(question.encode()).hexdigest() != state.question_key:
        raise ValueError('wrong question provenance')
    links={i['visual_instance_id']:[] for i in state.instances}
    for fid,views in state.viewpoints.items():
        for v in views:links[v['visual_instance_id']].append(int(fid))
    for gid in links:links[gid]=sorted(set(links[gid]))
    notes=[dict(name=n['name'],frame_id=n['evidence']['frame_id'],
        observed_visual_id=n['observed_visual_instance_id'],candidate_visual_ids=n['candidate_visual_instance_ids'],
        reason=n['reason']) for n in state.ambiguities[:max_notes]]
    base='\n'.join(c.get('text','') for m in messages for c in m['content'])
    # Conservative allowance for chat/image-boundary special tokens; the model
    # processor's actual budget check remains authoritative, without retries.
    allowance=max_text_tokens-reserved_output_tokens-len(tokenizer.encode(base+notice))-4*images-64
    attempts=[]
    for mode in ('full_records','compact_descriptions','compact_frame_links','columnar_records'):
        if mode=='full_records':
            records=[dict(visual_id=i['visual_instance_id'],name=i['name'],description=i['description'][:64],
                identity_status='visual_hypothesis',actual_frame_ids=links[i['visual_instance_id']],
                geometry_status='not_available') for i in state.instances]
            schema=None
        elif mode!='columnar_records':
            records=[[i['visual_instance_id'],i['name'],i['description'][:32],links[i['visual_instance_id']]]
                if mode=='compact_descriptions' else [i['visual_instance_id'],i['name'],
                links[i['visual_instance_id']][0],links[i['visual_instance_id']][-1]] for i in state.instances]
            schema=(['visual_id','name','short_description','actual_frame_ids'] if mode=='compact_descriptions'
                    else ['visual_id','name','first_actual_frame_id','last_actual_frame_id'])
        else:
            # Preserve the old row-index format for dense IDs. After capacity
            # eviction a small explicit ID table preserves surviving identities.
            row_ids=[i['visual_instance_id']for i in state.instances]
            contiguous=row_ids==list(range(len(state.instances)))
            names=sorted({i['name']for i in state.instances}|{n['name']for n in notes})
            name_index={name:i for i,name in enumerate(names)}
            reasons=sorted({n['reason']for n in notes});reason_index={reason:i for i,reason in enumerate(reasons)}
            records=dict(visual_id_is_row_index=contiguous,name_table=names,
                fields=['name_index','first_actual_frame_id','last_actual_frame_id_if_different'],
                missing_last_frame_equals_first=True,
                rows=[[name_index[i['name']],links[i['visual_instance_id']][0]]+
                      ([links[i['visual_instance_id']][-1]] if links[i['visual_instance_id']][-1]!=links[i['visual_instance_id']][0] else [])
                      for i in state.instances])
            if not contiguous:records['visual_id_table']=row_ids
            compact_notes=dict(fields=['name_index','frame_id','observed_visual_id','candidate_visual_ids','reason_index'],
                name_table=names,reason_table=reasons,
                rows=[[name_index[n['name']],n['frame_id'],n['observed_visual_id'],n['candidate_visual_ids'],reason_index[n['reason']]]for n in notes])
            schema=None
        payload=dict(hypotheses=records,ambiguities=compact_notes if mode=='columnar_records'else notes,record_fields=schema,
            selection_limitations=dict(omitted_relevant_hypotheses=0,
                omitted_ambiguity_notes=max(0,len(state.ambiguities)-len(notes)),
                description_characters=64 if mode=='full_records' else 32 if mode=='compact_descriptions' else 0))
        serialized=json.dumps(payload,separators=(',',':'))
        tokens=len(tokenizer.encode(serialized));attempts.append(dict(format=mode,tokens=tokens))
        exact_fits=False
        if mode=='compact_frame_links' and tokens>allowance and hasattr(tokenizer,'apply_chat_template'):
            # Preserve an already-valid old encoding even when the conservative
            # allowance is pessimistic. Chat image placeholders expand only into
            # visual tokens, so text/special accounting needs no pixel processing.
            original=messages[0]['content'][0]['text']
            messages[0]['content'][0]['text']=original+notice+serialized
            try:
                kwargs=dict(tokenize=True,add_generation_prompt=True)
                if chat_template is not None:kwargs['chat_template']=chat_template
                ids=tokenizer.apply_chat_template(messages,**kwargs)
                image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
                placeholders=sum(i==image_id for i in ids)
                if placeholders==images:
                    actual_text=len(ids)-placeholders
                    attempts[-1]['exact_chat_text_tokens']=actual_text
                    exact_fits=actual_text+reserved_output_tokens<=max_text_tokens
            finally:
                messages[0]['content'][0]['text']=original
        if tokens<=allowance or exact_fits or mode=='columnar_records':
            # An unavoidable oversize still reaches the normal explicit budget
            # failure path. Do not trim IDs, answer partially or call the model.
            messages[0]['content'][0]['text']+=notice+serialized
            return payload,dict(format=mode,format_attempts=attempts,text_allowance=allowance,
                estimated_memory_tokens=tokens,all_hypotheses_included=True)


def temporal_answer_frames(state, ledger, *, maximum=32):
    candidates=sorted(f for f in ledger if ledger[f]['margin']>0 or str(f) in state.viewpoints)
    if not candidates:candidates=sorted(ledger)
    if len(candidates)<=maximum:return candidates
    chosen=[]
    # Half the slots cover the actual time span, rather than only high-cardinality
    # repeated views; remaining slots cover linked hypotheses and complements.
    bins=max(1,maximum//2)
    lo,hi=candidates[0],candidates[-1]
    for b in range(bins):
        available=[f for f in candidates if min(bins-1,(f-lo)*bins//(hi-lo+1))==b]
        if available:chosen.append(max(available,key=lambda f:(ledger[f]['margin'],-f)))
    coverage={f:{v['visual_instance_id'] for v in state.viewpoints.get(str(f),[])} for f in candidates}
    covered=set().union(*(coverage[f] for f in chosen)) if chosen else set()
    vectors=np.stack([unpack_descriptor(ledger[f]['descriptor']).reshape(-1) for f in candidates])
    vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    position={f:i for i,f in enumerate(candidates)}
    similarity=np.max(vectors@vectors[[position[f] for f in chosen]].T,axis=1) if chosen else np.full(len(candidates),-1.)
    while len(chosen)<maximum:
        winner=max((f for f in candidates if f not in chosen),key=lambda f:(len(coverage[f]-covered),
            1.-float(similarity[position[f]]),ledger[f]['margin'],-f))
        chosen.append(winner);covered.update(coverage[winner])
        similarity=np.maximum(similarity,vectors@vectors[position[winner]])
    return sorted(chosen)
