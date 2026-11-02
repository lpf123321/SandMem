"""Verify bounded MLLM match requests using existing reliable memory evidence only."""
from copy import deepcopy
from dataclasses import replace

from sandbox_vlm.alignment.association import match_global
from sandbox_vlm.memory.association import aabb_union
from sandbox_vlm.memory.sandbox import AppearanceFeature
from sandbox_vlm.memory.updates import record_verified_observation
from sandbox_vlm.perception.evidence import PromptEvidence
from sandbox_vlm.perception.image_geometry import vggt_grid_transforms


def verify_match_requests(working, requests, metadata, settings):
    outcomes = []
    for request in requests:
        row = dict(request=deepcopy(request), status='pending')
        outcomes.append(row)
        if (set(request) != {'local_observation_id', 'instance_id'}
                or not isinstance(request['local_observation_id'], str)
                or type(request['instance_id']) is not int):
            row['reason'] = 'unsupported_revision_request'
            continue
        oid, gid = request['local_observation_id'], request['instance_id']
        if oid not in metadata['selection']['pending_ids'] or gid not in metadata['selection']['instance_ids']:
            row['reason'] = 'request_outside_selected_evidence'
            continue
        observation = next((o for o in working.pending_observations if o.local_observation_id == oid), None)
        note = next((n for n in working.ambiguity_notes if n.local_observation_id == oid), None)
        extent = next((e for e in working.spatial_extents if gid in e.instance_ids()), None)
        if observation is None or extent is None:
            row['reason'] = 'pending_observation_or_target_unavailable'
            continue
        if (observation.coordinate_reference != extent.coordinate_reference
                or note is None or note.reason.split(';',1)[0] == 'pending_alignment'):
            row['reason'] = 'reliable_common_reference_unavailable'
            continue
        evidence = [PromptEvidence.from_dict(p) for p in observation.prompt_evidence]
        actual = sorted({p.frame_id for p in evidence})
        if not actual or not set(actual) <= set(metadata['selection']['frame_ids']):
            row['reason'] = 'actual_prompt_images_not_all_provided'
            continue
        cameras = {v.frame_id: v.camera for v in extent.viewpoints}
        if not set(actual) <= cameras.keys():
            row['reason'] = 'camera_not_in_budgeted_memory'
            continue
        grids = {p.frame_id: g for p, g in zip(evidence, vggt_grid_transforms([p.source_hw for p in evidence]))}
        if any(grids[fid].output_hw != cameras[fid].image_size for fid in actual):
            row['reason'] = 'source_camera_grid_mapping_unavailable'
            continue
        match = match_global(observation, extent.instances, evidence, cameras, grids, settings)
        row['verification'] = match
        if match['matched_instance_id'] != gid:
            row['reason'] = 'approved_global_match_gates_not_satisfied'
            continue
        instance = working.get_instance(gid)
        fresh = bool(set(actual) - set(working.frames_for_instance(gid)))
        box = aabb_union([instance.bbox_3d, observation.bbox_3d]) if fresh else instance.bbox_3d
        feature = (AppearanceFeature(instance.appearance.vector.astype(float) + observation.appearance.vector.astype(float),
                                     dtype=instance.appearance.dtype) if fresh else instance.appearance)
        record_verified_observation(working, extent.extent_id, gid, replace(observation, frame_ids=actual), cameras)
        instance.bbox_3d, instance.appearance = box, feature
        if fresh:
            extent.bounds = aabb_union([extent.bounds, box])
        row.update(status='applied', reason='independently_verified_match', fused=fresh)
    return outcomes
