"""Recompose D116's already-tested input construction without development gates.

All geometry and identities are supplied by the frozen perception stages. This
module only preserves D105 geometry, adds D106 missing boxes, retrieves D112
records, and applies D111/D114/D116 input and response policies.
"""
from copy import deepcopy


def compose(state,frozen_geometry,packets,edges,frame_ids,watermark,raw_question):
    from sandbox_vlm.qa.complete_sandbox import complete_records,local_correspondences,display_geometry
    from sandbox_vlm.qa.complete_sandbox_repair import repair_geometry
    from sandbox_vlm.qa.evidence_binding import linked_selection
    from sandbox_vlm.qa.retrieved_sandbox import retrieve
    if 'next_visual_instance_id' in state:
        # An inherited geometry cache cannot restore forgotten observations or
        # objects. The same filter also applies when lexical retrieval keeps all
        # surviving records (so the later retrieval branch would do nothing).
        from sandbox_vlm.memory.coverage import CoverageSandbox
        from sandbox_vlm.memory.retention import _prune_auxiliary
        memory = CoverageSandbox.from_dict(state)
        pairs = {(int(f), l['visual_instance_id']) for f, rows in memory.viewpoints.items() for l in rows}
        frozen_geometry = deepcopy(frozen_geometry)
        frozen_geometry['sandbox'] = _prune_auxiliary(dict(geometry=frozen_geometry['sandbox']), memory)['geometry']
        frozen_geometry['audit'] = [o for o in frozen_geometry['audit'] if (o['frame_id'], o['instance_id']) in pairs]
        packets = deepcopy(packets)
        for packet in packets:
            packet['observations'] = [o for o in packet['observations'] if (o['frame_id'], o['instance_id']) in pairs]
    payload=complete_records(state,frame_ids,watermark)
    relations,excluded=local_correspondences(state,edges,set(frame_ids),watermark)
    geometry,provenance=display_geometry(frozen_geometry,frame_ids)
    plotted=sorted(geometry.instance_ids());ids={r[0]for r in payload['records']}
    payload.update(local_SAM_correspondences=relations,box_observation_sources=provenance,
        plotted_ids=plotted,unplotted_ids=sorted(ids-set(plotted)),
        local_references=[e.coordinate_reference for e in geometry.spatial_extents])
    original=dict(sandbox=geometry.to_dict(),box_sources=deepcopy(provenance))
    payload,repaired=repair_geometry(payload,original,dict(packets=packets))
    geometric=repaired if repaired is not None else original
    selection=linked_selection(raw_question,payload)
    changed=set(selection['relevant_ids'])!={r[0]for r in payload['records']}
    if changed:payload,geometric=retrieve(payload,geometric,selection)
    return payload,geometric,dict(selection=selection,excluded_local_correspondences=excluded,
        geometry_repaired=repaired is not None,selection_changed=changed)


def draw(geometry):
    from sandbox_vlm.qa.retrieved_sandbox import render_retrieved
    return render_retrieved(geometry)


def joint_messages(question,sparse,frame_ids,payload,images,diagram):
    from sandbox_vlm.qa.minimal_d065_prompt import minimal_messages
    return minimal_messages(dict(question=question,sparse=sparse,frame_ids=frame_ids,
        payload=payload,direct_answer=True),images,diagram)
