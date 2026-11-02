"""Apply approved conservative association and fusion to the unified sandbox."""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from sandbox_vlm.alignment.association import box_iou, match_global
from sandbox_vlm.memory.geometry import BBox3D
from sandbox_vlm.memory.sandbox import AppearanceFeature, ObjectInstance, SpatialExtent
from sandbox_vlm.memory.updates import _validate_pending_resolution, hold_pending, record_verified_observation
from sandbox_vlm.perception.evidence import PromptEvidence


def aabb_union(boxes):
    if not boxes or len({b.frame for b in boxes}) != 1:
        raise ValueError('AABB union needs a common reference')
    points = np.concatenate([b.corners() for b in boxes])
    lower, upper = points.min(axis=0), points.max(axis=0)
    return BBox3D((lower + upper) / 2, upper - lower, np.eye(3), boxes[0].frame)


def spatial_separation(observation, instances):
    """Strictly disjoint enclosing bounds, in one reliable common reference.

    This tests observed coarse boxes, not complete physical object surfaces.
    Signed axis gaps are positive only for strict separation; touching fails.
    Appearance/name rejection alone can never establish spatial novelty.
    """
    rows = []
    a = observation.bbox_3d.corners()
    for instance in instances:
        if instance.bbox_3d.frame != observation.coordinate_reference:
            raise ValueError('birth separation requires a common reference')
        b = instance.bbox_3d.corners()
        gaps = np.maximum(a.min(axis=0) - b.max(axis=0), b.min(axis=0) - a.max(axis=0))
        rows.append(dict(instance_id=instance.global_instance_id,
                         signed_axis_gaps=gaps.tolist(), disjoint=bool(np.any(gaps > 0))))
    return dict(separated=bool(rows) and all(r['disjoint'] for r in rows), instances=rows)


def separated_from_every_instance(observation, evidence, instances, current_clip_evidence):
    """Sufficient evidence: disjoint 3D bounds AND disjoint real same-frame boxes.

    No semantics, visibility prediction or unmatched status establishes novelty.
    The bounded current-clip evidence is discarded after the clip, not a history cache.
    """
    for instance in instances:
        a, b = observation.bbox_3d.corners(), instance.bbox_3d.corners()
        disjoint_3d = np.any((a.max(axis=0) < b.min(axis=0)) | (b.max(axis=0) < a.min(axis=0)))
        old = current_clip_evidence.get(instance.global_instance_id, [])
        separated = any(p.frame_id == q.frame_id and p.source_hw == q.source_hw
                        and box_iou(p.bbox_xyxy, q.bbox_xyxy) == 0 for p in evidence for q in old)
        if not (disjoint_3d and separated):
            return False
    return bool(instances)


def associate_observation(sandbox, observation, cameras, grids, settings, *,
                          alignment_reliable: bool, current_clip_evidence: dict):
    """Evidence map is temporary, owned by one clip; persistent links live in viewpoints."""
    if sandbox.validate() or not sandbox.video_id or sandbox.video_id != observation.video_id:
        raise ValueError('invalid sandbox or cross-video update')
    observation.__post_init__()
    def unresolved(result):
        if sandbox.retain_unresolved:
            hold_pending(sandbox, observation, result['candidate_instance_ids'], result['reason'])
        else:
            result = dict(result, status='not_stored', observation_retained=False)
        return result
    evidence = [PromptEvidence.from_dict(p) for p in observation.prompt_evidence]
    if not evidence:
        return unresolved(dict(status='pending', reason='missing_actual_prompt_evidence', candidate_instance_ids=[]))
    actual_ids = sorted({p.frame_id for p in evidence})
    if not set(actual_ids) <= set(observation.frame_ids):
        raise ValueError('actual prompt outside local candidate frames')
    observation = replace(observation, frame_ids=actual_ids)
    extents = [e for e in sandbox.spatial_extents if e.coordinate_reference == observation.coordinate_reference]
    if not alignment_reliable or (sandbox.spatial_extents and not extents):
        return unresolved(dict(status='pending', reason='pending_alignment', candidate_instance_ids=[]))
    if len(extents) > 1:
        raise ValueError('explicit extent reconciliation required')
    extent = extents[0] if extents else None
    instances = extent.instances if extent else []
    # Retain the first source-frame camera; never silently reconcile geometry.
    canonical_cameras = dict(cameras)
    if extent:
        canonical_cameras.update({v.frame_id: v.camera for v in extent.viewpoints})
    result = match_global(observation, instances, evidence, canonical_cameras, grids, settings)
    complete = observation.bbox_3d is not None and observation.appearance is not None
    matched = result.get('matched_instance_id')
    separation = None
    if complete and settings.birth_policy == 'spatial_separation':
        separation = spatial_separation(observation, instances)
    new = (complete and not sandbox.instance_ids()) or (complete and not result['candidate_instance_ids']
            and (separation['separated'] if separation is not None else
                 separated_from_every_instance(observation, evidence, instances, current_clip_evidence)))
    if separation is not None:
        result['birth_verification'] = dict(policy=settings.birth_policy, **separation)
    if matched is None and not new:
        return unresolved(result)
    # Prevalidate cameras/features and compute replacements before touching state.
    _validate_pending_resolution(sandbox, observation)
    for p in evidence:
        camera, grid = canonical_cameras.get(p.frame_id), grids.get(p.frame_id)
        if (camera is None or grid is None or camera.frame_id != p.frame_id
                or camera.reference != observation.coordinate_reference
                or grid.source_hw != p.source_hw or grid.output_hw != camera.image_size):
            raise ValueError('missing or inconsistent actual-evidence camera/grid')
    if matched is not None:
        instance = sandbox.get_instance(matched)
        fresh = bool(set(actual_ids) - set(sandbox.frames_for_instance(matched)))
        box = aabb_union([instance.bbox_3d, observation.bbox_3d]) if fresh else instance.bbox_3d
        feature = (AppearanceFeature(instance.appearance.vector.astype(float) + observation.appearance.vector.astype(float),
                                     dtype=instance.appearance.dtype) if fresh else instance.appearance)
        record_verified_observation(sandbox, extent.extent_id, matched, observation, canonical_cameras)
        instance.bbox_3d, instance.appearance = box, feature
        result.update(status='matched', fused=fresh)
    else:
        matched = max(sandbox.instance_ids(), default=-1) + 1
        instance = ObjectInstance(matched, observation.name, observation.short_description,
                                  aabb_union([observation.bbox_3d]), deepcopy(observation.appearance))
        # Prevalidated data make the following compact append/link operation deterministic.
        if extent is None:
            extent = SpatialExtent(max((e.extent_id for e in sandbox.spatial_extents), default=-1) + 1,
                                   observation.coordinate_reference, deepcopy(instance.bbox_3d))
            sandbox.spatial_extents.append(extent)
        extent.instances.append(instance)
        record_verified_observation(sandbox, extent.extent_id, matched, observation, canonical_cameras)
        result.update(status='created', matched_instance_id=matched, fused=False,
                      reason='initial_seed' if len(sandbox.instance_ids()) == 1 else
                      ('reliable_spatial_separation' if separation is not None else 'actual_covisibility_separation'))
    if result["status"] == "created" or result.get("fused"):
        extent.bounds = aabb_union([extent.bounds, observation.bbox_3d])
    current_clip_evidence.setdefault(matched, []).extend(evidence)
    return result
