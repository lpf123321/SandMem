"""Restore existing geometric evidence without promoting observation identities.

This adapter keeps the original evidence selection and real images. Only frozen
geometric births can bridge a geometric ID to an observation-directory ID.
Local identity relations supply neither a global pose nor a new geometric box.
"""
from copy import deepcopy

from dataclasses import replace

from sandbox_vlm.memory import Sandbox
from sandbox_vlm.qa.input import answer_messages
from sandbox_vlm.qa.render import render_sandbox, SPACE_NOTICE
from sandbox_vlm.qa.selection import EvidenceSelection


def restore_spatial_input(images, metadata, geometry, base, decisions, *, output_schema,
                          structure_observer=None):
    if not isinstance(geometry, Sandbox) or geometry.validate():
        raise ValueError('valid explicit geometric sandbox required')
    if geometry.video_id != base['video_id']:
        raise ValueError('cross-video geometric evidence')
    if len(images) != len(metadata['image_records']) or not images:
        raise ValueError('joint image records differ')
    if metadata['image_records'][0]['kind'] != 'sandbox':
        raise ValueError('joint input must reserve the first diagram slot')
    frames = set(metadata['selection']['frame_ids'])
    legend_ids = {r['instance_id'] for r in metadata['legend']}
    births = {}
    for oid, decision in decisions.items():
        if decision['status'] != 'created':
            continue
        gid = decision['matched_instance_id']
        directory_id = base['source_to_instance'].get(oid)
        if directory_id not in legend_ids:
            continue
        if gid in births:
            raise ValueError('ambiguous geometric birth mapping')
        record = base['records'][directory_id]
        if (record['instance_id'] != directory_id or oid not in record['local_contexts']
                or geometry.get_instance(gid).name != record['name']):
            raise ValueError('geometric birth does not match original observation')
        births[gid] = directory_id

    # A small rendering view shares immutable boxes, features and cameras. Do not
    # serialize and reconstruct every pending observation just to discard it.
    # The full original geometry remains retained and charged by the caller.
    extents = []
    for extent in geometry.spatial_extents:
        instances = [replace(i, global_instance_id=births[i.global_instance_id])
                     for i in extent.instances if i.global_instance_id in births]
        if not instances:
            continue
        views = []
        for view in extent.viewpoints:
            fid = view.frame_id
            actual_ids = {l['instance_id'] for l in base['viewpoints'].get(str(fid), [])}
            ids = [births[g] for g in view.observed_instance_ids
                   if g in births and births[g] in actual_ids]
            if fid in frames and ids:
                views.append(replace(view, observed_instance_ids=ids))
        extents.append(replace(extent, instances=instances, viewpoints=views))
    display = Sandbox(extents, video_id=geometry.video_id)
    if display.validate():
        raise ValueError('invalid geometric rendering view')
    if structure_observer is not None:
        structure_observer('spatial_display_view', [metadata, display, births, extents])
    selection = EvidenceSelection(metadata['original_question'], sorted(births.values()), [],
                                  sorted(frames), [], False, {}, [])
    diagram, render = render_sandbox(display, selection, max_side=max(images[0].size))
    meta = deepcopy(metadata)
    render.update(contains_3D_boxes=bool(births), geometry_source='frozen geometric sandbox',
                  old_geometry_to_directory_id={str(k): v for k, v in births.items()},
                  observation_identities_not_promoted=True,
                  geometry_missing_directory_ids=sorted(legend_ids - set(births.values())))
    meta['render'] = render
    for row in meta['legend']:
        gid = row['instance_id']
        if gid in births.values():
            box = display.get_instance(gid).bbox_3d
            row.update(coordinate_reference=box.frame, bbox_3d=box.to_dict(),
                       geometry_status='existing_geometric_rule_supported_not_ground_truth')
        else:
            row['geometry_status'] = 'no_verified_common_reference_box_in_this_input'
    meta['space_notice'] = SPACE_NOTICE
    meta['image_records'][0] = dict(kind='sandbox', input_image_size_hw=[diagram.height, diagram.width])
    meta['model_selection_limitations']['geometry_notice'] = (
        'Only plotted boxes have the stated shared reference. Other entries, including local identity '
        'links, do not acquire global geometry. Missing geometry does not mean a missing object. '
        'The plotted boxes are an incomplete inventory; do not infer the total from plotted box count.')
    if structure_observer is not None:
        structure_observer('spatial_metadata_copy', [metadata, meta, display, births, selection])
    joint = [diagram, *images[1:]]
    messages = answer_messages(joint, meta, output_schema=output_schema, citation_policy='implicit')
    return joint, meta, messages
