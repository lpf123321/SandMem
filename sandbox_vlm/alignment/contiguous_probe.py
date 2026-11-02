"""D081 candidate only: source-frame correspondences across contiguous VGGT clips.

No instance association, answer generation, threshold calibration or state mutation.
"""
import numpy as np
from sandbox_vlm.alignment.similarity import robust_similarity
from sandbox_vlm.memory.geometry import transform_points


def sampled_frame(depth, confidence, camera, *, stride):
    """Native pixel centres, finite positive depth and fixed top-half confidence."""
    depth=np.asarray(depth);confidence=np.asarray(confidence)
    assert depth.shape==confidence.shape==tuple(camera.image_size)
    ys,xs=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride];ys,xs=ys.ravel(),xs.ravel()
    finite=confidence[np.isfinite(confidence)]
    threshold=float(np.quantile(finite,.5))if len(finite)else np.inf
    valid=np.isfinite(depth[ys,xs])&(depth[ys,xs]>0)&np.isfinite(confidence[ys,xs])&(confidence[ys,xs]>=threshold)
    ys,xs=ys[valid],xs[valid];z=depth[ys,xs]
    rays=np.column_stack([xs,ys,np.ones(len(xs))])@np.linalg.inv(camera.intrinsic).T
    points=rays*z[:,None];pose=camera.T_ref_from_camera
    points=points@pose[:3,:3].T+pose[:3,3]
    return dict(frame_id=camera.frame_id,pixel_xy=np.column_stack([xs,ys]),points=points,depth=z,
                grid_hw=list(depth.shape),reference=camera.reference)


def paired_frames(previous,current,frame_ids):
    pairs=[]
    for fid in frame_ids:
        a,b=previous[fid],current[fid]
        if a['grid_hw']!=b['grid_hw']:raise ValueError('shared frame grid changed')
        # Identical source-frame/grid pixels are the correspondence; no image match.
        lookup={tuple(v):i for i,v in enumerate(a['pixel_xy'])}
        indices=[(lookup[tuple(v)],j)for j,v in enumerate(b['pixel_xy'])if tuple(v)in lookup]
        ia=np.asarray([i for i,j in indices],dtype=int);ib=np.asarray([j for i,j in indices],dtype=int)
        pairs.append(dict(frame_id=fid,source=b['points'][ib],reference=a['points'][ia],reference_depth=a['depth'][ia]))
    return pairs


def overlap_fit(previous,current,frame_ids,settings):
    """Four leave-one-source-frame-out checks; final fit uses first three only.

    The fourth fold is fixed in advance. Never select the best fold or refit on
    its held-out frame. All four checks must pass before an edge is accepted.
    """
    if len(frame_ids)!=4 or frame_ids!=sorted(set(frame_ids)):raise ValueError('four exact ordered overlaps required')
    pairs=paired_frames(previous,current,frame_ids);folds=[];production=None
    for held in pairs:
        train=[p for p in pairs if p is not held]
        source=np.concatenate([p['source']for p in train]);reference=np.concatenate([p['reference']for p in train]);depth=np.concatenate([p['reference_depth']for p in train])
        row=dict(held_out_frame_id=held['frame_id'],held_out_points=len(held['source']),passed=False)
        if len(depth)==0 or len(held['reference_depth'])==0:
            row.update(status='pending_alignment',reason='no_reliable_shared_points');folds.append(row);continue
        fit=robust_similarity(source,reference,residual_threshold=settings['residual_depth_fraction']*float(np.median(depth)),
            iterations=settings['ransac_iterations'],min_inliers=settings['min_inliers'],min_inlier_ratio=settings['min_inlier_ratio'],seed=settings['seed'])
        row.update(status=fit.status,reason=fit.reason,training=fit.diagnostics)
        if fit.T_ref_from_local is not None:
            residual=np.linalg.norm(transform_points(held['source'],fit.T_ref_from_local)-held['reference'],axis=1)/float(np.median(held['reference_depth']))
            fraction=float((residual<=settings['residual_depth_fraction']).mean())
            passed=len(residual)>=settings['min_inliers']and fraction>=settings['min_inlier_ratio']
            row.update(passed=passed,held_out_relative_median=float(np.median(residual)),held_out_relative_p90=float(np.quantile(residual,.9)),held_out_fraction=fraction,
                T_previous_from_current=fit.T_ref_from_local.tolist())
            if held['frame_id']==frame_ids[-1]:production=fit.T_ref_from_local
        folds.append(row)
    accepted=all(f['passed']for f in folds)
    return dict(status='aligned'if accepted else'pending_alignment',accepted=accepted,shared_frame_ids=frame_ids,
        production_held_out_frame=frame_ids[-1],T_previous_from_current=production.tolist()if accepted else None,folds=folds)
