"""D063 augmentation with D053 frame selection and pre-augmentation omission policy."""
from copy import deepcopy
from sandbox_vlm.memory.joint_geometry import merge_correspondences
from sandbox_vlm.memory.short_track import observations
from sandbox_vlm.qa.coverage_evidence import append_evidence


def revise(base, sam_state, sam_auxiliary, windows, *, watermark):
    if any(w['available_through']>watermark for w in windows):
        raise ValueError('future geometry')
    edges=sam_auxiliary['short_track']['accepted_edges']
    baseline,_=merge_correspondences(base,observations(base),edges,watermark=watermark)
    if baseline.to_dict()!=sam_state.to_dict():raise ValueError('frozen SAM state not reproduced')
    combined=list(edges)+[e for w in windows for e in w['edges']]
    return merge_correspondences(base,observations(base),combined,watermark=watermark)


def prepare_messages(record,chosen,sparse,sam_state,answer_state,images,config,processor):
    # Preserve the original memory omission decision before revising identity.
    from sandbox_vlm.qa.prompt import messages_for
    settings=config['coverage'];limits=deepcopy(config['qa']['budget'])
    if sparse:limits['max_output_tokens']=config['qa']['sparse_max_output_tokens']
    msg=messages_for(record['original_question'],images,chosen,'brief_evidence' if sparse else 'question_last')
    omit=settings.get('omit_redundant_uncertain_memory',False) and sparse and set(map(int,sam_state.viewpoints))<=set(chosen) and bool(sam_state.ambiguities)
    if omit:
        payload=None
        serial=dict(format='actual_frames_only',all_actual_observations_displayed=True,
            memory_omission_reason='sparse_visible_evidence_with_unresolved_identity',
            omitted_from_model_visual_ids=[i['visual_instance_id'] for i in sam_state.instances],
            full_memory_retained_and_charged=True)
    else:
        payload,serial=append_evidence(msg,answer_state,record['original_question'],processor.tokenizer,
            max_text_tokens=limits['max_text_tokens'],reserved_output_tokens=limits['max_output_tokens'],
            images=len(images),max_notes=settings['maximum_possible_correspondence_notes'],chat_template=processor.chat_template)
    return msg,payload,serial,limits
