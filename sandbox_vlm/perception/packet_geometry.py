"""D076: all actual observations share one joint VGGT reference per packet."""
from copy import deepcopy
import numpy as np

from sandbox_vlm.alignment.association import match_global
from sandbox_vlm.memory.geometry import BBox3D
from sandbox_vlm.memory.sandbox import AppearanceFeature, ObjectInstance
from sandbox_vlm.perception.evidence import PromptEvidence
from sandbox_vlm.perception.image_geometry import vggt_grid_transforms
from sandbox_vlm.perception.observations import backproject_masked_depth, estimate_local_aabb, LocalObservation


def build_packet_geometry(raw, packet, masks):
    ids = packet['frame_ids']
    if len(ids) != 16 or ids != sorted(set(ids)) or max(ids) > packet['available_through']:
        raise ValueError('invalid packet frame identity or watermark')
    if [c.frame_id for c in raw['cameras']] != ids:
        raise ValueError('camera order changed')
    sizes = [tuple(packet['source_hw'])] * len(ids)
    grids = vggt_grid_transforms(sizes, mode='crop')
    rows = []
    for obs in packet['observations']:
        fid = obs['evidence']['frame_id']; i = ids.index(fid)
        if list(obs['evidence']['source_hw']) != packet['source_hw']:
            raise ValueError('source dimensions changed')
        mask = grids[i].mask_to_grid(masks[obs['observation_id']])
        depth, conf = raw['depth'][i], raw['confidence'][i]
        values = conf[mask & np.isfinite(conf)]
        threshold = float(np.quantile(values, .5, method='linear')) if len(values) else None
        reliable = (mask & np.isfinite(conf) & (conf >= threshold) & np.isfinite(depth) & (depth > 0)
                    if threshold is not None else np.zeros_like(mask))
        points = backproject_masked_depth(depth, mask, raw['cameras'][i], reliable_pixels=reliable)
        estimated = estimate_local_aabb([points], coordinate_reference=packet['reference'], min_points=30)
        rows.append(dict(deepcopy(obs), bbox_3d=estimated.bbox_3d.to_dict() if estimated.bbox_3d else None,
            point_count=points.selected_pixel_count, failure=estimated.failure_reason))
    return rows, dict(zip(ids, grids))


def associate_packet(rows, cameras, grids, *, reference, frame_ids, available_through,
                     video_id, component_frames, settings):
    """Unique numeric match against every earlier component, then co-view guard.

    Components inherited from D065 remain intact. No position from another
    packet is read. Each component contributes its latest available observation.
    """
    parent = {int(k): int(k) for k in component_frames}
    frames = {int(k): set(v) for k,v in component_frames.items()}
    def root(gid):
        while parent[gid] != gid:
            gid = parent[gid]
        return gid
    prior, edges, decisions = [], [], []
    cameras = {c.frame_id: c for c in cameras}
    for target in sorted(rows, key=lambda o: (o['evidence']['frame_id'], o['observation_id'])):
        tf = target['evidence']['frame_id']; own = root(target['component_id'])
        by_component = {}
        for old in prior:
            gid = root(old['component_id'])
            if old['evidence']['frame_id'] < tf and gid != own and old['bbox_3d'] is not None:
                by_component[gid] = old
        instances = [ObjectInstance(gid, o['name'], '', BBox3D.from_dict(o['bbox_3d']),
                     AppearanceFeature.from_dict(o['appearance'])) for gid,o in sorted(by_component.items())]
        observation = LocalObservation(video_id, target['observation_id'], reference, target['name'], '', [tf],
            BBox3D.from_dict(target['bbox_3d']) if target['bbox_3d'] else None,
            AppearanceFeature.from_dict(target['appearance']), prompt_evidence=[target['evidence']])
        result = match_global(observation, instances, [PromptEvidence.from_dict(target['evidence'])],
                              cameras, grids, settings)
        matched = result['matched_instance_id']
        status = 'pending'
        if matched is not None:
            status = 'same_frame_conflict' if frames[matched] & frames[own] else 'matched'
            if status == 'matched':
                source = by_component[matched]
                score = next(s for s in result['scores'] if s['instance_id'] == matched)
                # Keep the existing typed single-reference evidence contract,
                # plus the complete multi-candidate decision for D076 validation.
                normalized = dict(status='matched', matched_instance_id=0,
                    scores=[dict(score, instance_id=0)])
                packet = dict(reference=reference, frame_ids=frame_ids, available_through=available_through,
                              source=source, target=target)
                edges.append(dict(source=source['observation_id'], target=target['observation_id'],
                    source_frame=source['evidence']['frame_id'], target_frame=tf, would_reuse=True,
                    evidence_type='D076_multi_observation_geometry', reference=reference,
                    geometry_evidence=packet, match_result=normalized, multi_candidate_match=result,
                    candidate_observation_ids={str(k):v['observation_id'] for k,v in by_component.items()}))
                lo, hi = sorted((matched,own));parent[hi] = lo;frames[lo] |= frames[hi]
        decisions.append(dict(target=target['observation_id'], status=status, result=result,
                              candidate_observation_ids={str(k):v['observation_id'] for k,v in by_component.items()}))
        prior.append(target)
    return dict(reference=reference, frame_ids=frame_ids, available_through=available_through,
                observations=rows, edges=edges, decisions=decisions)
