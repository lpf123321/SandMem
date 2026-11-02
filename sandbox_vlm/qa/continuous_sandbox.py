"""D104: measured local geometry from archived continuous SAM windows.

No discovery, segmentation, tracking-head calls, identity invention, or benchmark
labels. Rejected geometry is an offline audit, never a persistent pending pool.
"""
from copy import deepcopy
from io import BytesIO

import numpy as np

from sandbox_vlm.memory.geometry import (BBox3D, CameraParams, transform_bbox,
    transform_camera, transform_points)
from sandbox_vlm.memory.sandbox import (Sandbox, SpatialExtent, ObjectInstance,
    Viewpoint, AppearanceFeature)
from sandbox_vlm.perception.image_geometry import vggt_grid_transforms
from sandbox_vlm.perception.observations import backproject_masked_depth, estimate_local_aabb


SETTINGS = dict(min_points=30, minimum_target_checks=2, maximum_target_checks=3,
    depth_confidence_quantile=.5, projection_iou=.3, mask_hit_fraction=.5,
    relative_depth_error=.02, maximum_projection_points=512,
    minimum_alignment_points=50, alignment_inlier_fraction=.5,
    alignment_iterations=256, anchor_grid_stride=16, maximum_anchors_per_frame=256)


def open_rgb_images(paths):
    """Retain decoded pixels, never 32 simultaneous filesystem read buffers."""
    from PIL import Image
    result=[]
    for path in paths:
        with Image.open(path) as opened:
            result.append(opened.convert('RGB'))
    return result


def select_windows(windows, state, frame_ids, watermark):
    """Two existing causal windows, chosen before geometry or answer inspection."""
    actual = {}
    for fid, links in state.viewpoints.items():
        if int(fid) in frame_ids:
            for link in links:
                actual.setdefault(link['visual_instance_id'], set()).add(int(fid))
    candidates = []
    for window in windows:
        sf, end = window['source_frame'], window['end_frame']
        if not (sf <= end <= watermark and end-sf < 16 and window['available_through'] <= watermark):
            raise ValueError('noncausal or oversized archived window')
        if window['status'] != 'completed' or end-sf < 2:
            continue
        coverage = set()
        for oid in window['seed_observation_ids']:
            links = state.viewpoints.get(str(sf), [])
            j = int(oid.rsplit('_O', 1)[1])
            if j >= len(links):
                raise ValueError('archived source detached from current state')
            gid = links[j]['visual_instance_id']
            if any(sf <= f <= end for f in actual.get(gid, set())):
                coverage.add(gid)
        candidates.append((window, coverage, sum(sf <= f <= end for f in frame_ids)))
    selected, covered = [], set()
    while candidates and len(selected) < 2:
        def key(item):
            w, ids, hits = item
            spread = min(abs(w['source_frame']-s['source_frame']) for s in selected) if selected else 0
            return len(ids-covered), hits, spread, w['end_frame'], -w['source_frame']
        chosen = max(candidates, key=key)
        candidates.remove(chosen)
        selected.append(deepcopy(chosen[0])); covered |= chosen[1]
    if not selected:
        # A real scene/camera estimate remains possible without inventing objects.
        end = max(frame_ids)
        selected = [dict(source_frame=max(0, end-15), end_frame=end,
            available_through=watermark, status='scene_only', mask_path=None,
            seed_observation_ids=[])]
    return selected


def _bbox(mask):
    y, x = np.nonzero(mask)
    return np.array([x.min(), y.min(), x.max()+1, y.max()+1], float) if len(x) else None


def _iou(a, b):
    lower, upper = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = np.maximum(upper-lower, 0).prod()
    union = np.maximum(a[2:]-a[:2], 0).prod()+np.maximum(b[2:]-b[:2], 0).prod()-inter
    return float(inter/union) if union > 0 else 0.


def projection_check(points, camera, depth, mask, settings=SETTINGS):
    """Check predicted source surface against a separately obtained SAM region.

    No target-mask-based point fitting or selection. All in-bounds projected
    source points participate, so a tiny conveniently aligned subset cannot pass.
    """
    uv, positive = camera.project(points)
    h, w = mask.shape
    inside = positive & np.isfinite(uv).all(1)
    inside &= (uv[:, 0] >= 0) & (uv[:, 0] < w-.5) & (uv[:, 1] >= 0) & (uv[:, 1] < h-.5)
    indices = np.flatnonzero(inside)
    if len(indices) < settings['min_points'] or not mask.any():
        return dict(passed=False, reason='insufficient_projected_support', projected_points=len(indices)), {}
    xy = np.floor(uv[indices]+.5).astype(int)
    cam = transform_points(points[indices], np.linalg.inv(camera.T_ref_from_camera))
    observed_depth = depth[xy[:, 1], xy[:, 0]]
    hit = mask[xy[:, 1], xy[:, 0]]
    scale_values = depth[mask & np.isfinite(depth) & (depth > 0)]
    scale = float(np.median(scale_values)) if len(scale_values) else 0.
    finite = np.isfinite(observed_depth) & (observed_depth > 0)
    residual = np.abs(cam[:, 2]-observed_depth)
    relative = float(np.median(residual[finite])/scale) if finite.any() and scale > 0 else None
    predicted_box = np.r_[uv[indices].min(0), uv[indices].max(0)+1]
    overlap, hit_fraction = _iou(predicted_box, _bbox(mask)), float(hit.mean())
    passed = (len(indices) >= settings['min_points'] and finite.sum() >= settings['min_points']
        and overlap >= settings['projection_iou'] and hit_fraction >= settings['mask_hit_fraction']
        and relative is not None and relative <= settings['relative_depth_error'])
    audit = dict(passed=bool(passed), reason='passed' if passed else 'cross_view_inconsistency',
        projected_points=len(indices), mask_hit_fraction=hit_fraction, bbox_iou=overlap,
        median_relative_depth_error=relative, target_depth_scale=scale)
    evidence = dict(source_point_indices=indices, projected_xy=uv[indices],
        predicted_camera_depth=cam[:, 2], target_depth=observed_depth, target_mask_hits=hit)
    return audit, evidence


def process_window(raw, window, source_hw, *, settings=SETTINGS):
    """The only persistent products are accepted boxes, cameras and sparse anchors."""
    sf, end = window['source_frame'], window['end_frame']
    ids = list(range(sf, end+1))
    if [c.frame_id for c in raw['cameras']] != ids:
        raise ValueError('continuous depth/camera frame identity differs')
    ref = raw['cameras'][0].reference
    if any(c.reference != ref for c in raw['cameras']):
        raise ValueError('mixed camera references')
    grid = vggt_grid_transforms([tuple(source_hw)])[0]
    if grid.output_hw != raw['cameras'][0].image_size:
        raise ValueError('SAM to VGGT grid mismatch')
    cameras = {c.frame_id:c for c in raw['cameras']}
    packet = dict(reference=ref, frame_ids=ids, cameras={f:c.to_dict() for f,c in cameras.items()},
        accepted=[], anchors={}, available_through=window['available_through'])
    scene_clouds = []
    for index, fid in enumerate(ids):
        depth, conf = raw['depth'][index], raw['confidence'][index]
        valid = np.isfinite(depth) & (depth > 0) & np.isfinite(conf)
        if valid.any():
            valid &= conf >= np.quantile(conf[valid], settings['depth_confidence_quantile'])
        sparse = np.zeros_like(valid); sparse[::settings['anchor_grid_stride'], ::settings['anchor_grid_stride']] = True
        cloud = backproject_masked_depth(depth, np.ones_like(valid), cameras[fid], reliable_pixels=valid & sparse)
        scene_clouds.append(cloud)
        take = np.linspace(0, len(cloud.points_ref)-1, min(len(cloud.points_ref), settings['maximum_anchors_per_frame']), dtype=int)
        packet['anchors'][fid] = dict(xy=cloud.pixel_xy[take].astype(np.int16),
            points=cloud.points_ref[take].astype(np.float32),
            depth_scale=float(np.median(depth[valid])) if valid.any() else 0.)
    scene = estimate_local_aabb(scene_clouds, coordinate_reference=ref, min_points=settings['min_points'])
    packet['scene_bounds'] = scene.bbox_3d.to_dict() if scene.bbox_3d else None
    del scene_clouds
    if packet['scene_bounds'] is None:
        raise ValueError('no actual scene geometry')
    audits, evidence = [], {}
    if not window['mask_path']:
        return packet, audits, evidence
    with np.load(window['mask_path'], allow_pickle=False) as masks:
        for oid in window['seed_observation_ids']:
            row = dict(observation_id=oid, source_frame=sf, status='rejected', checks=[])
            audits.append(row)
            required = [f'track__{oid}__F{f}' for f in ids]
            if any(k not in masks or not masks[k].any() for k in required):
                row['reason'] = 'missing_or_empty_continuous_SAM_region'; continue
            seed_mask = grid.mask_to_grid(masks['prompt__'+oid])
            depth, conf = raw['depth'][0], raw['confidence'][0]
            valid = seed_mask & np.isfinite(depth) & (depth > 0) & np.isfinite(conf)
            if valid.any():
                valid &= conf >= np.quantile(conf[valid], settings['depth_confidence_quantile'])
            cloud = backproject_masked_depth(depth, seed_mask, cameras[sf], reliable_pixels=valid)
            estimate = estimate_local_aabb([cloud], coordinate_reference=ref, min_points=settings['min_points'])
            if estimate.bbox_3d is None:
                row['reason'] = 'insufficient_seed_points'; continue
            take = np.linspace(0, len(cloud.points_ref)-1,
                min(len(cloud.points_ref), settings['maximum_projection_points']), dtype=int)
            points = cloud.points_ref[take]
            targets = [ids[1:][i] for i in np.linspace(0, len(ids)-2,
                min(len(ids)-1, settings['maximum_target_checks']), dtype=int)]
            evidence[oid+'__seed_points'] = points.astype(np.float32)
            for fid in targets:
                mask = grid.mask_to_grid(masks[f'track__{oid}__F{fid}'])
                check, support = projection_check(points, cameras[fid], raw['depth'][fid-sf], mask, settings)
                row['checks'].append(dict(frame_id=fid, **check))
                for name, array in support.items():
                    evidence[f'{oid}__F{fid}__{name}'] = array
            if len(targets) < settings['minimum_target_checks'] or not all(c['passed'] for c in row['checks']):
                row['reason'] = 'cross_view_verification_failed'; continue
            row.update(status='accepted', reason='continuous_region_and_geometric_projection')
            packet['accepted'].append(dict(observation_id=oid, source_frame=sf,
                bbox_3d=estimate.bbox_3d.to_dict(), checked_frame_ids=targets))
    return packet, audits, evidence


def align_packets(reference, current, settings=SETTINGS):
    """Fit three shared frames; a fourth source frame is strictly held out."""
    from sandbox_vlm.alignment.similarity import robust_similarity
    shared = sorted(set(reference['anchors']) & set(current['anchors']))
    if len(shared) < 4:
        return None, dict(status='separate_references', reason='fewer_than_four_shared_frames')
    chosen = [shared[i] for i in np.linspace(0, len(shared)-1, 4, dtype=int)]
    pairs = {}
    for fid in chosen:
        a, b = current['anchors'][fid], reference['anchors'][fid]
        lookup = {tuple(x):p for x,p in zip(b['xy'], b['points'])}
        rows = [(p, lookup[tuple(x)]) for x,p in zip(a['xy'], a['points']) if tuple(x) in lookup]
        if len(rows) < settings['minimum_alignment_points']:
            return None, dict(status='separate_references', reason='insufficient_shared_pixels', frame_id=fid)
        pairs[fid] = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
    scale = float(np.median([reference['anchors'][f]['depth_scale'] for f in chosen]))
    fit = robust_similarity(np.concatenate([pairs[f][0] for f in chosen[:3]]),
        np.concatenate([pairs[f][1] for f in chosen[:3]]), residual_threshold=settings['relative_depth_error']*scale,
        iterations=settings['alignment_iterations'], min_inliers=settings['minimum_alignment_points'],
        min_inlier_ratio=settings['alignment_inlier_fraction'], seed=0)
    audit = dict(status='separate_references', training_frame_ids=chosen[:3], held_out_frame_id=chosen[3], fit=fit.diagnostics)
    if fit.T_ref_from_local is None:
        audit['reason'] = fit.reason; return None, audit
    a,b = pairs[chosen[3]]
    residual = np.linalg.norm(transform_points(a, fit.T_ref_from_local)-b, axis=1)
    count = int((residual <= settings['relative_depth_error']*scale).sum())
    audit.update(held_out_points=len(a), held_out_inliers=count, held_out_inlier_fraction=count/len(a))
    if count < settings['minimum_alignment_points'] or count/len(a) < settings['alignment_inlier_fraction']:
        audit['reason'] = 'held_out_frame_failed'; return None, audit
    audit.update(status='aligned', reason='shared_source_pixels_and_held_out_frame')
    return fit.T_ref_from_local, audit


def make_sandbox(packets, state, original_frame_ids):
    """Keep original IDs and actual links; propagation is never an observation link."""
    packets = deepcopy(packets)
    alignment = []
    if len(packets) == 2:
        transform, check = align_packets(packets[0], packets[1]); alignment.append(check)
        if transform is not None:
            ref = packets[0]['reference']; second = packets[1]
            for obj in second['accepted']:
                obj['bbox_3d'] = transform_bbox(BBox3D.from_dict(obj['bbox_3d']), transform, dst_reference=ref).to_dict()
            for fid,c in second['cameras'].items():
                second['cameras'][fid] = transform_camera(CameraParams.from_dict(c), transform, dst_reference=ref).to_dict()
            second['scene_bounds'] = transform_bbox(BBox3D.from_dict(second['scene_bounds']), transform, dst_reference=ref).to_dict()
            second['reference'] = ref
    groups = {}
    used, legend, omitted = set(), [], []
    for packet in packets:
        group = groups.setdefault(packet['reference'], dict(objects=[], cameras={}, links={}, scene=[]))
        group['scene'].append(BBox3D.from_dict(packet['scene_bounds']))
        for fid, camera in packet['cameras'].items():
            group['cameras'].setdefault(fid, CameraParams.from_dict(camera))
        for row in packet['accepted']:
            fid, oid = row['source_frame'], row['observation_id']
            link = state.viewpoints[str(fid)][int(oid.rsplit('_O', 1)[1])]
            gid = link['visual_instance_id']
            actual = sorted(f for f in original_frame_ids if f in packet['frame_ids']
                and any(v['visual_instance_id'] == gid for v in state.viewpoints.get(str(f), [])))
            if not actual or gid in used:
                omitted.append(dict(observation_id=oid, instance_id=gid,
                    reason='no_original_QA_support' if not actual else 'existing_ID_has_one_geometry'))
                continue
            source = state.instances[gid]
            group['objects'].append(ObjectInstance(gid, source['name'], source['description'],
                BBox3D.from_dict(row['bbox_3d']), AppearanceFeature.from_dict(source['appearance'])))
            for f in actual:
                group['links'].setdefault(f, set()).add(gid)
            used.add(gid)
            legend.append(dict(instance_id=gid, name=source['name'], source_frame=fid,
                actual_QA_support_frame_ids=actual, checked_frame_ids=row['checked_frame_ids'],
                coordinate_reference=packet['reference']))
    extents = []
    for ref, group in groups.items():
        # A drawable range exists even where object geometry did not pass.
        boxes = [o.bbox_3d for o in group['objects']] or group['scene']
        points = np.concatenate([b.corners() for b in boxes] +
            [c.T_ref_from_camera[None,:3,3] for c in group['cameras'].values()])
        lo, hi = points.min(0), points.max(0)
        bounds = BBox3D((lo+hi)/2, hi-lo, np.eye(3), ref)
        view_ids = sorted(group['links'])
        if not view_ids:
            view_ids = [min(group['cameras']), max(group['cameras'])]
        views = [Viewpoint(f, group['cameras'][f], sorted(group['links'].get(f, set()))) for f in sorted(set(view_ids))]
        extents.append(SpatialExtent(len(extents), ref, bounds, group['objects'], views))
    sandbox = Sandbox(extents, video_id=state.video_id, retain_unresolved=False)
    if sandbox.validate():
        raise ValueError('invalid validated spatial sandbox')
    return sandbox, dict(legend=legend, omitted_geometry=omitted, alignment=alignment,
        original_instance_count=len(state.instances), plotted_ids=sorted(used),
        no_new_IDs=True, no_pending_pool=True, physical_identity_ground_truth=False)


def render_verified(sandbox, metadata, *, max_side=768):
    """Actual 3D boxes with short in-plot IDs and a complete external image legend."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image
    if sandbox.validate() or not sandbox.spatial_extents:
        raise ValueError('real geometry required, no text-directory replacement')
    fig = plt.figure(figsize=(max_side/100, max_side*.75/100), dpi=100, facecolor='white')
    edges = [(a,b) for a in range(8) for b in range(a+1,8) if (a^b) in (1,2,4)]
    panels = []
    try:
        for index,e in enumerate(sandbox.spatial_extents):
            n=len(sandbox.spatial_extents)
            ax=fig.add_axes([.015+index*.68/n,.12,.68/n,.78],projection='3d')
            def box(b,color,style='solid',alpha=1):
                p=b.corners();ax.add_collection3d(Line3DCollection([(p[a],p[b]) for a,b in edges],
                    colors=color,linestyles=style,alpha=alpha,linewidths=1.1))
            box(e.bounds,'grey','dashed',.45)
            for obj in e.instances:
                color=plt.get_cmap('tab20')(obj.global_instance_id%20)
                box(obj.bbox_3d,color)
                ax.text(*obj.bbox_3d.center,f'I{obj.global_instance_id}',color=color,fontsize=8)
            length=max(np.linalg.norm(e.bounds.extents)*.06,1e-6)
            for v in e.viewpoints:
                p=v.camera.T_ref_from_camera
                ax.scatter(*p[:3,3],s=8,color='black')
                ax.quiver(*p[:3,3],*(p[:3,2]*length),color='black',arrow_length_ratio=.2)
                ax.text(*p[:3,3],f'F{v.frame_id}',fontsize=6)
            radius=max(float(e.bounds.extents.max())/2,1e-6)*1.08
            for setter,x in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),e.bounds.center):setter(x-radius,x+radius)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-60)
            ax.set_title(f'Local reference {index}',fontsize=9)
            ax.set_xlabel('X',fontsize=7);ax.set_ylabel('Y',fontsize=7);ax.set_zlabel('Z',fontsize=7)
            ax.tick_params(labelsize=5)
            panels.append(dict(reference=e.coordinate_reference,instance_ids=sorted(e.instance_ids()),
                frame_ids=[v.frame_id for v in e.viewpoints]))
        lines=[]
        for item in sorted(metadata['legend'],key=lambda r:r['instance_id']):
            frames='/'.join(str(f) for f in item['actual_QA_support_frame_ids'])
            lines.append(f"I{item['instance_id']} {item['name']}  F{frames}")
        import textwrap
        wrapped=[s for line in lines for s in textwrap.wrap(line,width=35,subsequent_indent='  ')]
        ax=fig.add_axes([.71,.08,.285,.82]);ax.axis('off')
        ax.text(0,1,'ID / object / supporting frame',fontsize=8,va='top')
        # Complete labels are never silently cut to the first 24 instances.
        fontsize=min(8, 310/max(len(wrapped),1))
        ax.text(0,.95,'\n'.join(wrapped) if wrapped else 'Observed 3D range and cameras.\nNo object passed verification.',
            fontsize=fontsize,va='top',linespacing=1.25)
        fig.text(.02,.97,'Estimated local 3D | arbitrary scale | incomplete object inventory',fontsize=9,va='top')
        fig.text(.02,.025,'Panels have independent coordinates; layout does not imply physical relationships.',fontsize=7)
        data=BytesIO();fig.savefig(data,format='png',dpi=100);data.seek(0)
        diagram=Image.open(data).convert('RGB').copy()
    finally:
        plt.close(fig)
    return diagram,dict(panels=panels,legend_rows=len(lines),legend_wrapped_rows=len(wrapped),
        legend_fontsize=fontsize,full_legend=True,image_size_hw=[diagram.height,diagram.width])
