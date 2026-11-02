"""D065 instances and source selection, with actual mask-supported VGGT input.

Births use the existing D043 actual SAM evidence rule, not successful global
alignment. This module neither splits old IDs nor promotes appearance to truth.
"""
from copy import deepcopy
import hashlib
import numpy as np

from sandbox_vlm.memory.coverage import CoverageSandbox
from sandbox_vlm.memory import Sandbox, SpatialExtent, ObjectInstance, Viewpoint, AppearanceFeature
from sandbox_vlm.memory.geometry import BBox3D
from sandbox_vlm.memory.joint_geometry import observations
from sandbox_vlm.perception.observations import backproject_masked_depth, estimate_local_aabb
from sandbox_vlm.perception.image_geometry import vggt_grid_transforms
from sandbox_vlm.qa.selection import EvidenceSelection
from sandbox_vlm.qa.render import render_sandbox, SPACE_NOTICE
from sandbox_vlm.qa.prompt import messages_for


def instances_only(source):
    """Preserve every admitted D065 ID/link; discard correspondence notes only."""
    source.validate()
    result = CoverageSandbox(source.video_id, '', retain_ambiguities=False)
    result.question_key = source.question_key
    for name in ('instances', 'viewpoints', 'observation_index'):
        setattr(result, name, deepcopy(getattr(source, name)))
    if hasattr(source, 'next_visual_instance_id'):
        result.next_visual_instance_id = source.next_visual_instance_id
    if hasattr(source, 'memory_limit_bytes'):
        result.memory_limit_bytes = source.memory_limit_bytes
    result.validate()
    return result


def joint_frame_plan(state, original_ids, catalog, *, watermark, maximum_images=32, maximum_views=16,
                     allow_scene_only=False):
    if original_ids != sorted(set(original_ids)) or any(f > watermark for f in original_ids):
        raise ValueError('unordered or future source selection')
    obs = observations(state)
    supported = {o['evidence']['frame_id'] for o in obs if o['observation_id'] in catalog
                 and catalog[o['observation_id']]['available_through'] <= watermark}
    if not supported and not (allow_scene_only and not state.instances):
        raise ValueError('no actual SAM mask for spatial reconstruction')
    selected = list(original_ids)
    added = []
    if supported and not (set(selected) & supported):
        added = [min(supported)]
        selected += added
    coverage = {f: {v['visual_instance_id'] for v in state.viewpoints.get(str(f), [])} for f in selected}
    removed = []
    while len(selected) > maximum_images-1:
        eligible = [f for f in selected if f not in added and (f not in supported or len(set(selected)&supported)>1)]
        def loss(f):
            elsewhere = set().union(*(coverage[g] for g in selected if g != f))
            gap = min(abs(f-g) for g in selected if g != f)
            return len(coverage[f]-elsewhere), gap, f
        remove = min(eligible, key=loss)
        selected.remove(remove); removed.append(remove)
    selected.sort()
    frames = sorted(set(selected) & supported) if supported else list(selected)
    if len(frames) > maximum_views:
        # Fixed temporal coverage, independent of answers and object categories.
        frames = [frames[i] for i in np.rint(np.linspace(0, len(frames)-1, maximum_views)).astype(int)]
    return selected, frames, dict(original_frame_ids=original_ids, retained_original_frames=len(set(selected)&set(original_ids)),
        removed_for_diagram=removed, added_for_actual_geometry=added, geometry_frame_ids=frames)


def reconstruct(raw, state, catalog, frame_ids, *, include_scene_extent=False):
    """Build one shared local reference from real masks, depth and cameras."""
    if len(raw['cameras']) != len(frame_ids) or [c.frame_id for c in raw['cameras']] != frame_ids:
        raise ValueError('camera/depth frame order differs')
    ref = raw['cameras'][0].reference
    if any(c.reference != ref for c in raw['cameras']):
        raise ValueError('geometry call mixes references')
    by_frame = {f: [] for f in frame_ids}
    for o in observations(state):
        if o['evidence']['frame_id'] in by_frame and o['observation_id'] in catalog:
            by_frame[o['evidence']['frame_id']].append(o)
    points, actual, audit = {}, {}, []
    for index, fid in enumerate(frame_ids):
        rows = by_frame[fid]
        if not rows:
            continue
        grid = vggt_grid_transforms([tuple(rows[0]['evidence']['source_hw'])], mode='crop')[0]
        for o in rows:
            mask_ref = catalog[o['observation_id']]
            with np.load(mask_ref['path'], allow_pickle=False) as z:
                mask = grid.mask_to_grid(z[mask_ref['key']])
            depth, conf = raw['depth'][index], raw['confidence'][index]
            valid = mask & np.isfinite(conf) & np.isfinite(depth) & (depth > 0)
            if valid.any():
                valid &= conf >= np.quantile(conf[valid], .5, method='linear')
            cloud = backproject_masked_depth(depth, mask, raw['cameras'][index], reliable_pixels=valid)
            gid = o['visual_instance_id']
            estimate = estimate_local_aabb([cloud], coordinate_reference=ref, min_points=30)
            audit.append(dict(observation_id=o['observation_id'], instance_id=gid, frame_id=fid,
                bbox_3d=estimate.bbox_3d.to_dict() if estimate.bbox_3d else None, failure=estimate.failure_reason))
            if estimate.bbox_3d is not None:
                points.setdefault(gid, []).append(cloud)
                actual.setdefault(fid, set()).add(gid)
    objects = []
    for gid, clouds in sorted(points.items()):
        estimate = estimate_local_aabb(clouds, coordinate_reference=ref, min_points=30)
        row = state.instance_by_id(gid)
        objects.append(ObjectInstance(gid, row['name'], row['description'], estimate.bbox_3d,
                                      AppearanceFeature.from_dict(row['appearance'])))
    extents = []
    scene_box = None
    if include_scene_extent:
        scene_points=[]
        for index,camera in enumerate(raw['cameras']):
            depth,conf=raw['depth'][index],raw['confidence'][index]
            valid=np.isfinite(depth)&(depth>0)&np.isfinite(conf)
            if valid.any():valid &= conf>=np.quantile(conf[valid],.5,method='linear')
            sparse=np.zeros_like(valid);sparse[::16,::16]=True;valid &= sparse
            scene_points.append(backproject_masked_depth(depth,np.ones_like(valid),camera,reliable_pixels=valid))
        scene_box=estimate_local_aabb(scene_points,coordinate_reference=ref,min_points=30).bbox_3d
        del scene_points
    if objects:
        corners = np.concatenate([o.bbox_3d.corners() for o in objects]
                                 +([scene_box.corners()] if scene_box is not None else []))
        lo, hi = corners.min(0), corners.max(0)
        bounds = BBox3D((lo+hi)/2, hi-lo, np.eye(3), ref)
        views = [Viewpoint(c.frame_id, c, sorted(actual[c.frame_id])) for c in raw['cameras'] if c.frame_id in actual]
        extents = [SpatialExtent(0, ref, bounds, objects, views)]
    elif scene_box is not None:
        extents=[SpatialExtent(0,ref,scene_box,[],[Viewpoint(c.frame_id,c,[]) for c in raw['cameras']])]
    geometry = Sandbox(extents, video_id=state.video_id, retain_unresolved=False)
    if geometry.validate():
        raise ValueError('invalid reconstructed geometry')
    return geometry, audit


def prepare_messages(state, geometry, question, images, records, *, sparse, diagram_side=768):
    if hashlib.sha256(question.encode()).hexdigest() != state.question_key:
        raise ValueError('question provenance differs')
    ids = [r['frame_id'] for r in records]
    selection = EvidenceSelection(question, sorted(geometry.instance_ids()), [], ids, [], False, {}, [])
    diagram, rendering = render_sandbox(geometry, selection, max_side=diagram_side)
    # Complete compact inventory; no target cap or pending-observation request.
    links = {i['visual_instance_id']: [] for i in state.instances}
    for fid, rows in state.viewpoints.items():
        for r in rows:
            links[r['visual_instance_id']].append(int(fid))
    legend = [[i['visual_instance_id'], i['name'], min(links[i['visual_instance_id']]), max(links[i['visual_instance_id']])]
              for i in state.instances]
    import json
    messages = messages_for(question, images, ids, 'brief_evidence' if sparse else 'question_last')
    content = messages[0]['content']
    content[0]['text'] += ('\n'+SPACE_NOTICE+'\nObject IDs are estimates from actual SAM observations, not known physical counts. '
        'Only plotted boxes have the displayed VGGT reference; the map is an incomplete inventory. '
        'Use all real frames to answer. Inventory fields: [ID,name,first_actual_frame,last_actual_frame].\n'
        +json.dumps(legend, ensure_ascii=False, separators=(',', ':')))
    content[1:1] = [dict(type='text', text='3D sandbox, estimated arbitrary scale:'), dict(type='image', image=diagram)]
    joint = [diagram, *images]
    meta = dict(original_question=question, selection=selection.to_dict(), space_notice=SPACE_NOTICE,
        legend=[dict(instance_id=r[0], name=r[1], first_actual_frame=r[2], last_actual_frame=r[3]) for r in legend],
        render=rendering, plotted_ids=sorted(geometry.instance_ids()), stored_instances=len(state.instances),
        omitted_inventory_ids=[], image_records=[dict(kind='sandbox',input_image_size_hw=[diagram.height,diagram.width])]
        +[dict(kind='source_frame', **r) for r in records], pending_pool_retained=False,
        estimated_identity_is_not_physical_truth=True)
    return joint, meta, messages


def original_inventory_omission(sam_state, chosen, sparse, coverage_settings):
    """Reproduce D065's pre-revision input policy before discarding old notes.

    This reads inherited state only during input preparation. The runtime gets
    one Boolean; old ambiguity records never become a new persistent pool.
    """
    return bool(coverage_settings.get('omit_redundant_uncertain_memory', False)
        and sparse and set(map(int, sam_state.viewpoints)) <= set(chosen)
        and sam_state.ambiguities)


def prepare_comparison_messages(state, geometry, question, images, records, processor, limits, *, sparse,
                                omit_inventory=False):
    """Both D102 arms retain D065 descriptions and its fixed budget serializer."""
    from sandbox_vlm.qa.coverage_evidence import append_evidence
    joint,meta,_=prepare_messages(state,geometry,question,images,records,sparse=sparse)
    del _
    if not geometry.instance_ids() and geometry.spatial_extents:
        selection=EvidenceSelection(question,[],[],[r['frame_id'] for r in records],[],False,{},[])
        diagram,rendering=render_sandbox(geometry,selection,max_side=768,
            include_empty_extents=True,include_unlinked_viewpoints=True)
        joint[0]=diagram;meta['render']=rendering
        meta['scene_only_geometry']=True
    messages=messages_for(question,images,[r['frame_id'] for r in records],
                          'brief_evidence' if sparse else 'question_last')
    content=messages[0]['content']
    content[0]['text']+='\n'+SPACE_NOTICE
    content[1:1]=[dict(type='text',text='3D sandbox, estimated arbitrary scale:'),dict(type='image',image=joint[0])]
    if omit_inventory:
        content[0]['text'] += ('\nOnly plotted boxes share the displayed local VGGT reference. '
            'The map is an incomplete inventory; use the actual video frames.')
        plotted = set(meta['plotted_ids'])
        meta['legend'] = [v for v in meta['legend'] if v['instance_id'] in plotted]
        meta.update(compact_memory=None,
            evidence_serialization=dict(format='actual_frames_and_3D_only',
                original_D065_inventory_omission_preserved=True, all_hypotheses_included=False,
                full_instance_state_retained_and_charged=True),
            omitted_inventory_ids=[i['visual_instance_id'] for i in state.instances
                if i['visual_instance_id'] not in plotted],
            description_policy='preserve original D065 omission; no added full inventory')
        return joint,meta,messages
    notice=('\nThese instance records come from actual SAM observations; IDs are estimates, not physical count truth. '
        'Check them against the real frames. Only plotted boxes share the displayed local VGGT reference. '
        'The map is an incomplete inventory. Record geometry_status describes full-video common geometry, '
        'which is unavailable.\nInstance memory (JSON data):\n')
    payload,serialization=append_evidence(messages,state,question,processor.tokenizer,
        max_text_tokens=limits['max_text_tokens'],reserved_output_tokens=limits['max_output_tokens'],
        images=len(joint),max_notes=0,chat_template=processor.chat_template,notice=notice)
    meta.update(compact_memory=payload,evidence_serialization=serialization,
                description_policy='same D065 deterministic budget serializer; all IDs retained')
    return joint,meta,messages
