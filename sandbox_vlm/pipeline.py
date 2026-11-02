"""Sequential clip mapping with charged overlap and optional bounded history.

No saved perception output or benchmark target is read. Offline audit outputs
are write-only. In the current instances-only mode, failed associations are
released at the clip boundary without a new-instance claim or pending pool.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import base64
import time

import numpy as np

from sandbox_vlm.alignment.association import MatchSettings
from sandbox_vlm.alignment.similarity import robust_similarity
from sandbox_vlm.alignment.history import HistorySettings, SparseView, sample_view, transform_view, retain_anchors, recover_reference,attach_actual_region
from sandbox_vlm.data.video import Clip, FrameRef, select_discovery_frames
from sandbox_vlm.memory import Sandbox, CameraParams
from sandbox_vlm.memory.association import associate_observation
from sandbox_vlm.memory.geometry import transform_bbox, transform_camera
from sandbox_vlm.memory.storage import cpu_storage_report
from sandbox_vlm.memory.updates import migrate_pending_coordinates
from sandbox_vlm.perception.confidence import select_reliable_depth
from sandbox_vlm.perception.image_geometry import vggt_grid_transforms
from sandbox_vlm.perception.local_geometry import build_local_observation
from sandbox_vlm.perception.observations import backproject_masked_depth


def align_geometry(previous, current, settings):
    """Estimate current-local to previous-local; never compare different pixels."""
    source, target = [], []
    for fid in sorted(set(previous['frames']) & set(current['frames'])):
        a, b = previous['frames'][fid], current['frames'][fid]
        if any(a[k] != b[k] for k in ('source_index', 'sha256', 'image_size_hw')):
            raise ValueError('overlap source identity changed')
        da, db = previous['depth'][fid], current['depth'][fid]
        if da.shape != db.shape:
            return None, dict(status='pending_alignment', reason='different_native_grids')
        reliable = previous['reliable'][fid] & current['reliable'][fid]
        grid = np.zeros(da.shape, dtype=bool)
        grid[::settings['pixel_stride'], ::settings['pixel_stride']] = True
        pa = backproject_masked_depth(da, grid, previous['cameras'][fid], reliable_pixels=reliable)
        pb = backproject_masked_depth(db, grid, current['cameras'][fid], reliable_pixels=reliable)
        if not np.array_equal(pa.pixel_xy, pb.pixel_xy):
            raise ValueError('overlap pixel correspondence mismatch')
        source.append(pb.points_ref); target.append(pa.points_ref)
    src = np.concatenate(source) if source else np.empty((0, 3))
    dst = np.concatenate(target) if target else np.empty((0, 3))
    fit = robust_similarity(src, dst,
        residual_threshold=settings['residual_depth_fraction']*previous['median_depth'],
        iterations=settings['ransac_iterations'], min_inliers=settings['min_inliers'],
        min_inlier_ratio=settings['min_inlier_ratio'], seed=settings['seed'])
    return fit.T_ref_from_local, dict(status=fit.status, reason=fit.reason, **fit.diagnostics)


def encode_overlap(state):
    def array(value):
        a = np.ascontiguousarray(value)
        return dict(dtype=a.dtype.str, shape=list(a.shape), bytes=base64.b64encode(a.tobytes()).decode('ascii'))
    result = {key: {str(k): array(v) if key in ('depth', 'reliable') else
                   v.to_dict() if key == 'cameras' else v for k,v in state[key].items()}
              for key in ('frames', 'depth', 'reliable', 'cameras')}
    result.update(median_depth=state['median_depth'], transform=state['transform'].tolist(),
                  reference=state['reference'], alignment_reliable=state['alignment_reliable'])
    if 'anchors' in state:
        result['anchors']=[a.to_dict() for a in state['anchors']]
    return result


def decode_overlap(state):
    if state is None:
        return None
    def array(value):
        return np.frombuffer(base64.b64decode(value['bytes']), dtype=value['dtype']).reshape(value['shape']).copy()
    result = {key: {int(k): array(v) if key in ('depth','reliable') else
                   CameraParams.from_dict(v) if key=='cameras' else v for k,v in state[key].items()}
              for key in ('frames','depth','reliable','cameras')}
    result.update(median_depth=state['median_depth'], transform=np.asarray(state['transform']),
                  reference=state['reference'], alignment_reliable=state['alignment_reliable'])
    if 'anchors' in state:
        result['anchors']=[SparseView.from_dict(a) for a in state['anchors']]
    return result


def build_video(sample_root, manifest, config, *, backend, discovery, tracker, geometry, audit_sink=None, resume=False, sparse_audit_sink=None):
    """Model transports are injectable for contract checks, never labeled real by this API."""
    retain_unresolved = config['memory'].get('retain_unresolved_observations', True)
    sandbox = backend.read_working_copy() if resume else Sandbox(video_id=manifest['video_id'],
                                                               retain_unresolved=retain_unresolved)
    if sandbox.retain_unresolved != retain_unresolved:
        raise ValueError('memory policy changed across resume; start a separate run')
    previous = decode_overlap(backend.read_auxiliary()) if resume else None
    if sandbox.video_id != manifest['video_id']:
        raise ValueError('cannot resume another video')
    history_config=config['alignment'].get('history',{})
    history_settings=HistorySettings(**{k:v for k,v in history_config.items() if k!='enabled'}) if history_config.get('enabled',False) else None
    rows = []
    for clip in manifest['clips']:
        cid, frames = clip['clip_id'], clip['frames']
        ids = [f['frame_id'] for f in frames]
        refs = [FrameRef.from_dict(f) for f in frames]
        selected = select_discovery_frames(Clip(cid, refs), **dict(
            count=config['perception']['discovery']['frames_per_clip'],
            strategy=config['perception']['discovery']['strategy']))
        if clip['discovery_frame_ids'] != [f.frame_id for f in selected]:
            raise ValueError('prepared discovery frames differ from approved rule')
        paths = [sample_root / f['path'] for f in frames]
        for frame, path in zip(frames, paths):
            if hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
                raise ValueError('prepared frame changed')
        discovery_frames = [f for f in frames if f['frame_id'] in clip['discovery_frame_ids']]
        discovery_batch_size = config['perception']['mllm'].get('batch_size', 1)
        if type(discovery_batch_size) is not int or not 1 <= discovery_batch_size <= 4:
            raise ValueError('discovery batch_size must be 1..4')
        discoveries = []
        for start in range(0, len(discovery_frames), discovery_batch_size):
            group = discovery_frames[start:start+discovery_batch_size]
            if discovery_batch_size == 1:
                discoveries.append(discovery.discover(sample_root / group[0]['path'], group[0]['frame_id']))
            else:
                discoveries.extend(discovery.discover_batch([sample_root / f['path'] for f in group],
                                                            [f['frame_id'] for f in group]))
        detections = [d for r in discoveries if r['status'] == 'ok' for d in r['detections']]
        if history_settings is None:
            tracks, sam_cost = tracker.track_clip(paths[0].parent, ids, detections, clip_id=cid,
                iou_threshold=config['perception']['sam']['local_match_iou'])
        raw_reference = f'clip_{cid}/local'
        raw = geometry.infer(paths, ids, reference=raw_reference)
        depth, confidence, cameras = raw['depth'], raw['confidence'], raw['cameras']
        reliability = select_reliable_depth(depth, confidence,
            keep_fraction=config['perception']['geometry']['depth_conf_keep_fraction']).pixels
        positive = depth[np.isfinite(depth) & (depth > 0)]
        if not len(positive):
            raise ValueError('VGGT returned no finite positive depth')
        current = dict(frames={f['frame_id']: f for f in frames},
            depth=dict(zip(ids, depth)), reliable=dict(zip(ids, reliability)),
            cameras={c.frame_id: c for c in cameras}, median_depth=float(np.median(positive)))
        grids = dict(zip(ids, vggt_grid_transforms([tuple(f['image_size_hw']) for f in frames])))
        current_views=[]
        anchors=previous.get('anchors',[]) if previous is not None else []
        if anchors and history_settings is None:
            raise ValueError('history state cannot be silently discarded by disabling its config')
        if history_settings is not None:
            def collect_features(fid,grid):
                current_views.append(sample_view(grid,video_id=sandbox.video_id,frame=current['frames'][fid],
                    camera=current['cameras'][fid],grid_transform=grids[fid],depth=current['depth'][fid],
                    reliable=current['reliable'][fid],names=[d['name'] for d in detections if d['frame_id']==fid],
                    representative_features=[],settings=history_settings))
            region_events=[]
            def collect_region(detection,mask,appearance,local_id):
                view=next(v for v in current_views if v.frame_id==detection['frame_id'])
                region_events.append(attach_actual_region(view,detection,mask,appearance,local_id,grids[view.frame_id]))
            region_kwargs=dict(prompt_region_sink=collect_region) if history_settings.region_matching else {}
            tracks,sam_cost=tracker.track_clip(paths[0].parent,ids,detections,clip_id=cid,
                iou_threshold=config['perception']['sam']['local_match_iou'],
                sparse_frame_ids=[f.frame_id for f in selected],sparse_feature_sink=collect_features,**region_kwargs)
            if history_settings.region_matching:
                sam_cost['history_region_events']=dict(Counter(e['status'] for e in region_events))
            for view in current_views:
                features=[t.representative_feature.vector for t in tracks if t.seed_detection['frame_id']==view.frame_id and t.representative_feature is not None]
                view.representative_features=np.asarray(features,dtype=np.float32).reshape(-1,view.descriptors.shape[1])
                view.__post_init__()
        reference, transform, reliable = raw_reference, np.eye(4), previous is None
        alignment = dict(status='initial_reference')
        if previous is not None:
            local_transform, alignment = align_geometry(previous, current, config['alignment']['geom'])
            if local_transform is not None:
                transform = previous['transform'] @ local_transform
                reference, reliable = previous['reference'], previous['alignment_reliable']
        history_report=None
        if history_settings is not None:
            started=time.perf_counter()
            if sparse_audit_sink is not None:
                sparse_audit_sink(cid,current_views,anchors)
            if not reliable:
                recovered,recovered_reference,history_report=recover_reference(current_views,anchors,history_settings,config['alignment']['geom'])
                if recovered is not None:
                    overlap_report=alignment
                    migration=None
                    if (local_transform is not None and not previous['alignment_reliable']
                            and previous['reference']!=recovered_reference):
                        migrated_transform=recovered @ np.linalg.inv(previous['transform'] @ local_transform)
                        migration=migrate_pending_coordinates(sandbox,source_reference=previous['reference'],
                            target_reference=recovered_reference,transform=migrated_transform,
                            overlap_proof=overlap_report,history_proof=history_report)
                    history_report['coordinate_migration']=migration
                    transform,reference,reliable=recovered,recovered_reference,True
                    alignment=dict(status='historical_relocalized',overlap_alignment=overlap_report)
            else:
                history_report=dict(status='not_needed',reason='reliable_overlap_or_initial_reference',model_calls=0)
            additions=[transform_view(v,transform,reference) for v in current_views if reliable and len(v.points)>=config['alignment']['geom']['min_inliers']]
            anchors,retention=retain_anchors(anchors,additions,history_settings,watermark=max(ids))
            history_report.update(retention=retention,wall_seconds=time.perf_counter()-started,
                old_pending_structures_migrated=False,settings=vars(history_settings))
        transformed = {c.frame_id: transform_camera(c, transform, dst_reference=reference) for c in cameras}
        frame_map, transient, observations, decisions = current['frames'], {}, [], []
        for track in tracks:
            result = build_local_observation(video_id=manifest['video_id'], observation_id=track.local_track_id,
                name=track.seed_detection['name'], description=track.seed_detection['short_description'],
                frame_ids=ids, source_sizes=[tuple(f['image_size_hw']) for f in frames], masks=track.masks,
                cameras=cameras, depth=depth, confidence=confidence, representative_feature=track.representative_feature,
                keep_fraction=config['perception']['geometry']['depth_conf_keep_fraction'],
                min_points=config['perception']['geometry']['min_points'])
            obs = result.observation
            if obs is None:
                decisions.append(dict(local_observation_id=track.local_track_id, status='perception_failed'))
                continue
            obs.prompt_evidence = [dict(frame_id=p['detection']['frame_id'], bbox_xyxy=p['detection']['bbox_xyxy'],
                source_hw=frame_map[p['detection']['frame_id']]['image_size_hw'], sam_mask_nonempty=p['sam_mask_nonempty'])
                for p in track.prompt_records]
            if obs.bbox_3d is not None:
                obs.bbox_3d = transform_bbox(obs.bbox_3d, transform, dst_reference=reference)
            obs.coordinate_reference = reference
            observations.append(obs.to_dict())
            outcome = associate_observation(sandbox, obs, transformed, grids,
                MatchSettings(**config['alignment']['match']), alignment_reliable=reliable,
                current_clip_evidence=transient)
            decisions.append(dict(local_observation_id=obs.local_observation_id, **outcome))
        # Retained alignment context is charged and stays in the complete GPU
        # snapshot too, rather than becoming hidden CPU history between queries.
        tail = ids[-config['video']['overlap']:] if config['video']['overlap'] else []
        previous = {key: {fid: current[key][fid].copy() if key in ('depth','reliable') else current[key][fid]
                          for fid in tail} for key in ('frames','depth','reliable','cameras')}
        previous.update(median_depth=current['median_depth'], transform=transform,
                        reference=reference, alignment_reliable=reliable)
        if history_settings is not None:
            previous['anchors']=anchors
        live_host_bytes=cpu_storage_report([sandbox,previous])['cpu_structural_bytes']
        admission = backend.commit(sandbox, auxiliary=encode_overlap(previous))
        row = dict(clip_id=cid, discovery=discoveries, sam=sam_cost, vggt=raw['cost'], alignment=alignment,
            alignment_reliable=reliable, T_reference_from_local=transform.tolist(), reference=reference,
            backend=admission, outcomes=dict(Counter(d['status'] for d in decisions)), decisions=decisions)
        row['cpu_candidate_structural_working_bytes']=live_host_bytes
        if history_report is not None:
            row['history']=history_report
        rows.append(dict(clip_id=cid, failed_discovery_frames=[r['frame_id'] for r in discoveries if r['status']!='ok'],
                         sam_prompt_failures=sam_cost.get('prompt_failures'),
                         alignment=alignment, outcomes=row['outcomes'], backend=admission,
                         cpu_candidate_structural_working_bytes=live_host_bytes,
                         discovery_calls=len(discoveries), vggt_cost=raw['cost']))
        rows[-1]['discovery_logical_single_image_requests'] = len(discoveries)
        rows[-1]['discovery_generate_invocations'] = sum(d.get('generation_invocations', 1) for d in discoveries)
        if audit_sink:
            audit_sink(cid, row, observations, [c.to_dict() for c in transformed.values()])
        sandbox = backend.read_working_copy() if admission['complete_state_resident'] else Sandbox(
            video_id=manifest['video_id'], retain_unresolved=retain_unresolved)
        if admission['status'] != 'ok':
            return sandbox, dict(status='capacity_limited', clips=rows, processed_through=ids[-1])
        # Only charged overlap and configured bounded anchors survive. Everything else
        # is released; offline audit archives are never read by the online method.
        if history_settings is not None:
            if current_views:
                del view, features
            del collect_features, collect_region,region_events,region_kwargs,additions
        del tracks, transient, current, raw, depth, confidence, reliability, observations, current_views, anchors
    incomplete = any(c['failed_discovery_frames'] or c['sam_prompt_failures'] for c in rows)
    return sandbox, dict(status='incomplete' if incomplete else 'ok', clips=rows,
                         processed_through=rows and manifest['clips'][-1]['frames'][-1]['frame_id'])
