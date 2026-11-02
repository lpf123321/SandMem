"""Bounded frozen-SAM/VGGT anchors; appearance proposes, geometry verifies."""
from dataclasses import dataclass, asdict, field
import base64

import numpy as np

from sandbox_vlm.alignment.similarity import robust_similarity
from sandbox_vlm.memory import CameraParams
from sandbox_vlm.memory.geometry import transform_camera, transform_points


def pack_array(value):
    value = np.ascontiguousarray(value)
    return dict(dtype=value.dtype.str, shape=list(value.shape), bytes=base64.b64encode(value.tobytes()).decode('ascii'))


def unpack_array(value):
    return np.frombuffer(base64.b64decode(value['bytes'], validate=True), dtype=value['dtype']).reshape(value['shape']).copy()


@dataclass(frozen=True)
class HistorySettings:
    max_anchors: int = 4
    max_points: int = 512
    cosine_min: float = .80
    distance_ratio_max: float = .80
    min_axis_span: float = .20
    spatial_bins: int = 4
    min_occupied_bins: int = 6
    reprojection_median_max_px: float = 3.
    reprojection_p90_max_px: float = 8.
    consensus_rotation_max_deg: float = 5.
    consensus_scale_relative_max: float = .05
    consensus_translation_depth_max: float = .02
    region_matching: bool = False
    region_appearance_cosine_min: float = .60
    region_require_same_name: bool = True

    def __post_init__(self):
        if type(self.region_matching) is not bool or not -1<=self.region_appearance_cosine_min<=1:
            raise ValueError('invalid region matching settings')
        if type(self.region_require_same_name) is not bool:
            raise ValueError('region name policy must be boolean')
        for key in ('max_anchors', 'max_points', 'spatial_bins', 'min_occupied_bins'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('history counts must be positive integers')
        if self.max_anchors > 4 or self.max_points > 512 or self.min_occupied_bins > self.spatial_bins**2:
            raise ValueError('history exceeds approved bounds')
        for key in ('cosine_min', 'distance_ratio_max', 'min_axis_span'):
            if not np.isfinite(getattr(self, key)) or not 0 < getattr(self, key) < 1:
                raise ValueError('invalid history fraction')
        for key in ('reprojection_median_max_px', 'reprojection_p90_max_px',
                    'consensus_rotation_max_deg', 'consensus_scale_relative_max', 'consensus_translation_depth_max'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError('invalid history tolerance')


@dataclass
class SparseRegion:
    name: str
    local_track_id: str
    prompt_frame_id: int
    point_indices: np.ndarray
    appearance: np.ndarray

    def to_dict(self):
        return dict(name=self.name,local_track_id=self.local_track_id,prompt_frame_id=self.prompt_frame_id,
            point_indices=pack_array(self.point_indices),appearance=pack_array(self.appearance))

    @classmethod
    def from_dict(cls,value):
        return cls(**{**value,'point_indices':unpack_array(value['point_indices']),'appearance':unpack_array(value['appearance'])})


@dataclass
class SparseView:
    video_id: str
    frame_id: int
    source_sha256: str
    camera: CameraParams
    source_hw: tuple[int, int]
    sam_grid_hw: tuple[int, int]
    pixel_xy: np.ndarray  # native VGGT grid, also camera grid
    points: np.ndarray
    descriptors: np.ndarray  # native values/bits, never quantized
    feature_dtype: str
    names: list[str]
    representative_features: np.ndarray
    median_depth: float
    mask_regions: list[SparseRegion] = field(default_factory=list)

    def __post_init__(self):
        if not self.video_id or type(self.frame_id) is not int or self.frame_id < 0 or self.camera.frame_id != self.frame_id:
            raise ValueError('invalid anchor identity')
        if len(self.source_sha256) != 64 or any(x not in '0123456789abcdef' for x in self.source_sha256):
            raise ValueError('anchor requires source SHA256')
        n = len(self.points)
        if (not 0 <= n <= 512 or self.points.shape != (n, 3) or self.pixel_xy.shape != (n, 2)
                or self.descriptors.ndim != 2 or len(self.descriptors) != n or self.descriptors.shape[1] < 1
                or not np.isfinite(self.points).all() or not np.isfinite(self.pixel_xy).all()
                or not np.isfinite(self.median_depth) or self.median_depth <= 0):
            raise ValueError('invalid bounded sparse geometry')
        if self.feature_dtype not in ('float32', 'float16', 'bfloat16'):
            raise ValueError('unsupported feature dtype')
        expected = {'float32': np.float32, 'float16': np.float16, 'bfloat16': np.uint16}[self.feature_dtype]
        if self.descriptors.dtype != expected or not np.isfinite(self.float_descriptors()).all():
            raise ValueError('invalid native descriptors')
        if (len(self.source_hw) != 2 or len(self.sam_grid_hw) != 2 or min(*self.source_hw, *self.sam_grid_hw) < 1
                or np.any(self.pixel_xy < 0) or np.any(self.pixel_xy >= np.array(self.camera.image_size[::-1]))):
            raise ValueError('anchor pixel grid mismatch')
        if self.representative_features.ndim != 2 or not np.isfinite(self.representative_features).all():
            raise ValueError('invalid retrieval features')
        if len(self.mask_regions)>512:
            raise ValueError('unbounded anchor regions')
        for region in self.mask_regions:
            idx=region.point_indices
            if (not region.name.strip() or not region.local_track_id or region.prompt_frame_id!=self.frame_id
                    or idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer) or len(idx)!=len(set(idx.tolist()))
                    or np.any(idx<0) or np.any(idx>=n) or region.appearance.ndim!=1
                    or not np.isfinite(region.appearance).all() or not np.isclose(np.linalg.norm(region.appearance),1,atol=1e-3)):
                raise ValueError('invalid actual-prompt sparse region')

    def float_descriptors(self):
        if self.feature_dtype == 'bfloat16':
            return (self.descriptors.astype(np.uint32) << 16).view(np.float32)
        return self.descriptors.astype(np.float32)

    def to_dict(self):
        return dict(video_id=self.video_id, frame_id=self.frame_id, source_sha256=self.source_sha256,
            camera=self.camera.to_dict(), source_hw=list(self.source_hw), sam_grid_hw=list(self.sam_grid_hw),
            feature_dtype=self.feature_dtype, names=self.names, median_depth=self.median_depth,
            mask_regions=[r.to_dict() for r in self.mask_regions],
            **{key:pack_array(getattr(self,key)) for key in ('pixel_xy','points','descriptors','representative_features')})

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, 'camera':CameraParams.from_dict(value['camera']),
            'mask_regions':[SparseRegion.from_dict(r) for r in value.get('mask_regions',[])],
            'source_hw':tuple(value['source_hw']), 'sam_grid_hw':tuple(value['sam_grid_hw']),
            **{key:unpack_array(value[key]) for key in ('pixel_xy','points','descriptors','representative_features')}})


def attach_actual_region(view, detection, source_mask, appearance, local_track_id, grid_transform):
    """Transport point membership into an actual nonempty prompt mask, no propagation claim."""
    if (detection['frame_id']!=view.frame_id or source_mask.dtype!=np.bool_
            or source_mask.shape!=view.source_hw or not source_mask.any()
            or grid_transform.source_hw!=view.source_hw or grid_transform.output_hw!=view.camera.image_size):
        raise ValueError('actual mask/anchor mapping mismatch')
    if any(r.local_track_id==local_track_id for r in view.mask_regions):
        return dict(status='duplicate_local_prompt_preserved')
    xy=np.column_stack([view.pixel_xy,np.ones(len(view.pixel_xy))]) @ np.linalg.inv(grid_transform.A_grid_from_source).T
    xy=np.floor(xy[:,:2]+.5).astype(int)
    valid=(xy[:,0]>=0)&(xy[:,0]<view.source_hw[1])&(xy[:,1]>=0)&(xy[:,1]<view.source_hw[0])
    indices=np.flatnonzero(valid)
    indices=indices[source_mask[xy[indices,1],xy[indices,0]]]
    if not len(indices):return dict(status='no_sparse_points_in_actual_mask')
    view.mask_regions.append(SparseRegion(detection['name'].strip(),local_track_id,view.frame_id,
        indices.astype(np.uint16),appearance.vector.copy()))
    view.__post_init__()
    return dict(status='actual_region_attached',point_count=len(indices))


def sample_view(grid, *, video_id, frame, camera, grid_transform, depth, reliable, names,
                representative_features, settings):
    """Sample SAM cell centres through explicit source -> VGGT pixel mapping.

    Native bfloat16 is stored as uint16 bits; matching alone widens to float32.
    No interpolated depth, padding support, or non-reliable pixel is invented.
    """
    import torch
    if grid.ndim != 3 or tuple(depth.shape) != camera.image_size or reliable.dtype != np.bool_ or reliable.shape != depth.shape:
        raise ValueError('sparse sampler grid mismatch')
    if tuple(frame['image_size_hw']) != grid_transform.source_hw or grid_transform.output_hw != camera.image_size:
        raise ValueError('source/VGGT mapping mismatch')
    channels, gh, gw = grid.shape
    yy, xx = np.indices((gh, gw))
    # SAM resizes the entire source to square: no crop and no aspect padding.
    xy = np.column_stack([((xx.ravel()+.5)*grid_transform.source_hw[1]/gw)-.5,
                          ((yy.ravel()+.5)*grid_transform.source_hw[0]/gh)-.5])
    mapped = np.column_stack([xy, np.ones(len(xy))]) @ grid_transform.A_grid_from_source.T
    pixels = np.floor(mapped[:,:2]+.5).astype(int)
    h,w = depth.shape
    valid = (pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    idx = np.flatnonzero(valid)
    px,py = pixels[idx].T
    idx = idx[reliable[py,px]&np.isfinite(depth[py,px])&(depth[py,px]>0)]
    # Exclude zero/nonfinite native descriptor cells before deterministic thinning.
    vectors = grid.detach().permute(1,2,0).reshape(-1,channels)
    finite = torch.isfinite(vectors).all(1) & (vectors.float().norm(dim=1)>0)
    idx = idx[finite.cpu().numpy()[idx]]
    # Two feature cells must not duplicate the same geometry pixel.
    _, first = np.unique(pixels[idx], axis=0, return_index=True)
    idx = idx[np.sort(first)]
    if len(idx)>settings.max_points:
        idx = idx[np.linspace(0,len(idx)-1,settings.max_points).round().astype(int)]
    pix = pixels[idx].astype(np.float64)
    rays = np.column_stack([pix,np.ones(len(pix))]) @ np.linalg.inv(camera.intrinsic).T
    points_camera = rays * depth[pixels[idx,1],pixels[idx,0],None]
    points = transform_points(points_camera,camera.T_ref_from_camera)
    selected = vectors[torch.as_tensor(idx,device=vectors.device)].contiguous()
    dtype = str(grid.dtype).removeprefix('torch.')
    descriptors = (selected.view(torch.uint16) if dtype=='bfloat16' else selected).cpu().numpy().copy()
    positive = depth[np.isfinite(depth)&(depth>0)]
    return SparseView(video_id,frame['frame_id'],frame['sha256'],camera,tuple(frame['image_size_hw']),
        (gh,gw),pix,points,descriptors,dtype,sorted(set(n.casefold().strip() for n in names)),
        np.asarray(representative_features,dtype=np.float32).reshape(-1,channels),float(np.median(positive)))


def transform_view(view, transform, reference):
    from dataclasses import replace
    scale = np.cbrt(np.linalg.det(transform[:3,:3]))
    return replace(view,points=transform_points(view.points,transform),
        camera=transform_camera(view.camera,transform,dst_reference=reference),median_depth=view.median_depth*scale)


def retain_anchors(previous, candidates, settings, *, watermark):
    """Pin the first accepted anchor, then greedily retain complementary headings."""
    all_views = {}
    for view in [*previous,*candidates]:
        if view.frame_id>watermark:
            raise ValueError('future history anchor')
        key=(view.video_id,view.frame_id)
        if key in all_views:
            if view.source_sha256!=all_views[key].source_sha256:
                raise ValueError('history source identity changed')
            continue  # Keep the first geometry for repeated source evidence.
        all_views[key]=view
    if len({v.video_id for v in all_views.values()})>1:
        raise ValueError('cross-video anchors')
    if len({v.camera.reference for v in all_views.values()})>1:
        raise ValueError('anchors must share a verified reference')
    pool=sorted(all_views.values(),key=lambda v:v.frame_id)
    if not pool:
        return [],dict(retained_frame_ids=[],removed_frame_ids=[])
    chosen=[previous[0] if previous else pool[0]]
    while len(chosen)<min(settings.max_anchors,len(pool)):
        def diversity(v):
            forward=v.camera.T_ref_from_camera[:3,2]
            return min(float(np.arccos(np.clip(forward @ old.camera.T_ref_from_camera[:3,2],-1,1))) for old in chosen)
        available=[v for v in pool if v.frame_id not in {c.frame_id for c in chosen}]
        chosen.append(max(available,key=lambda v:(diversity(v),-v.frame_id)))
    ids=[v.frame_id for v in chosen]
    return chosen,dict(retained_frame_ids=ids,removed_frame_ids=[v.frame_id for v in pool if v.frame_id not in ids])


def mutual_matches(source, target, settings):
    if len(source)<2 or len(target)<2:
        return np.empty((0,2),dtype=int)
    a,b=source.astype(np.float64),target.astype(np.float64)
    an,bn=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1)
    if np.any(an==0) or np.any(bn==0) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid matching descriptors')
    similarity=np.clip((a/an[:,None]) @ (b/bn[:,None]).T,-1,1)
    top=np.argsort(-similarity,axis=1,kind='stable')[:,:2]
    reverse=np.argsort(-similarity,axis=0,kind='stable')[:2,:]
    pairs=[]
    for i,(j,second) in enumerate(top):
        if reverse[0,j]!=i or similarity[i,j]<settings.cosine_min:
            continue
        d=max(0.,1-similarity[i,j]); second_d=max(0.,1-similarity[i,second])
        reverse_second=max(0.,1-similarity[reverse[1,j],j])
        # Strict uniqueness also rejects identical descriptors with equal distances.
        if second_d>1e-12 and reverse_second>1e-12 and d<=settings.distance_ratio_max**2*min(second_d,reverse_second):
            pairs.append((i,j))
    return np.asarray(pairs,dtype=int).reshape(-1,2)


def region_matches(current,anchor,settings):
    source,target=current.mask_regions,anchor.mask_regions
    eligible=[]
    for i,a in enumerate(source):
        for j,b in enumerate(target):
            if (settings.region_require_same_name and ' '.join(a.name.casefold().split())!=' '.join(b.name.casefold().split())
                    or a.appearance.shape!=b.appearance.shape):continue
            cosine=float(a.appearance.astype(float)@b.appearance.astype(float))
            if cosine>=settings.region_appearance_cosine_min:eligible.append((i,j,cosine))
    a_features,b_features=current.float_descriptors(),anchor.float_descriptors()
    combined,rows=[],[]
    for i,j,cosine in eligible:
        a,b=source[i],target[j]
        unique=sum(k[0]==i for k in eligible)==1 and sum(k[1]==j for k in eligible)==1
        row=dict(current_track_id=a.local_track_id,anchor_track_id=b.local_track_id,name=a.name,
            appearance_cosine=cosine,unique_region_candidate=unique,matched_points=0)
        rows.append(row)
        if not unique:continue  # Similar same-class instances are never forced into one correspondence.
        pairs=mutual_matches(a_features[a.point_indices],b_features[b.point_indices],settings)
        mapped=[(int(a.point_indices[p]),int(b.point_indices[q])) for p,q in pairs]
        row['matched_points']=len(mapped);combined.extend(mapped)
    combined=sorted(set(combined));forward={};reverse={}
    for i,j in combined:
        forward.setdefault(i,set()).add(j);reverse.setdefault(j,set()).add(i)
    pairs=np.asarray([(i,j) for i,j in combined if len(forward[i])==len(reverse[j])==1],dtype=int).reshape(-1,2)
    return pairs,dict(actual_current_regions=len(source),actual_anchor_regions=len(target),
        eligible_region_pairs=rows,point_conflicts_rejected=len(combined)-len(pairs))


def project(points,camera):
    pose=camera.T_ref_from_camera
    local=(points-pose[:3,3]) @ pose[:3,:3]
    if np.any(local[:,2]<=0):
        return None
    pixels=local @ camera.intrinsic.T
    return pixels[:,:2]/pixels[:,2,None]


def verify_pair(current,anchor,settings,geom):
    if current.video_id!=anchor.video_id:
        raise ValueError('cross-video relocalization')
    if current.frame_id==anchor.frame_id and current.source_sha256!=anchor.source_sha256:
        raise ValueError('repeated source identity changed')
    region_details=None
    if settings.region_matching:
        pairs,region_details=region_matches(current,anchor,settings)
    else:
        pairs=mutual_matches(current.float_descriptors(),anchor.float_descriptors(),settings)
    return verify_geometric_pairs(current,anchor,pairs,settings,geom,region_details=region_details)


def verify_geometric_pairs(current,anchor,pairs,settings,geom,*,region_details=None):
    """Shared original geometry gates, independent of correspondence producer.

    Views need only identity, camera, pixels, points and median_depth. This
    function does not infer appearance or assert an observed instance identity.
    Online SAM matching still enters through verify_pair unchanged.
    """
    if current.video_id!=anchor.video_id:
        raise ValueError('cross-video relocalization')
    if current.frame_id==anchor.frame_id and current.source_sha256!=anchor.source_sha256:
        raise ValueError('repeated source identity changed')
    pairs=np.asarray(pairs)
    if (pairs.ndim!=2 or pairs.shape[1]!=2 or not np.issubdtype(pairs.dtype,np.integer)
            or np.any(pairs<0) or np.any(pairs[:,0]>=len(current.points))
            or np.any(pairs[:,1]>=len(anchor.points))
            or len(set(pairs[:,0].tolist()))!=len(pairs)
            or len(set(pairs[:,1].tolist()))!=len(pairs)):
        raise ValueError('unique bounded point correspondences required')
    row=dict(current_frame_id=current.frame_id,anchor_frame_id=anchor.frame_id,matched_points=len(pairs),
             status='pending_alignment',matches=pairs.tolist())
    if region_details is not None:row['region_matching']=region_details
    fit=robust_similarity(current.points[pairs[:,0]],anchor.points[pairs[:,1]],
        residual_threshold=geom['residual_depth_fraction']*anchor.median_depth,
        iterations=geom['ransac_iterations'],min_inliers=geom['min_inliers'],
        min_inlier_ratio=geom['min_inlier_ratio'],seed=geom['seed'])
    row.update(geometry=fit.diagnostics,reason=fit.reason,inlier_match_indices=np.flatnonzero(fit.inliers).tolist())
    if fit.T_ref_from_local is None:
        return None,row
    inliers=pairs[fit.inliers]
    coverage=[]
    for view,indices in ((current,inliers[:,0]),(anchor,inliers[:,1])):
        normalized=view.pixel_xy[indices]/np.asarray(view.camera.image_size[::-1])
        span=np.ptp(normalized,axis=0)
        bins=np.minimum((normalized*settings.spatial_bins).astype(int),settings.spatial_bins-1)
        coverage.append(dict(axis_span=span.tolist(),occupied_bins=len(np.unique(bins,axis=0))))
    row['spatial_coverage']=coverage
    if any(min(c['axis_span'])<settings.min_axis_span or c['occupied_bins']<settings.min_occupied_bins for c in coverage):
        row['reason']='insufficient_spatial_coverage'
        return None,row
    transform=fit.T_ref_from_local
    projected_anchor=project(transform_points(current.points[inliers[:,0]],transform),anchor.camera)
    projected_current=project(transform_points(anchor.points[inliers[:,1]],np.linalg.inv(transform)),current.camera)
    if projected_anchor is None or projected_current is None:
        row['reason']='correspondence_behind_camera'
        return None,row
    errors=[np.linalg.norm(projected_anchor-anchor.pixel_xy[inliers[:,1]],axis=1),
            np.linalg.norm(projected_current-current.pixel_xy[inliers[:,0]],axis=1)]
    row['bidirectional_reprojection']=[dict(median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9))) for e in errors]
    # Cameras can legitimately differ under motion; no small-motion pose gate.
    aligned_camera=transform_camera(current.camera,transform,dst_reference=anchor.camera.reference)
    rotation=aligned_camera.T_ref_from_camera[:3,:3] @ anchor.camera.T_ref_from_camera[:3,:3].T
    row['camera_relative_pose']=dict(rotation_degrees=float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))),
        translation_over_anchor_depth=float(np.linalg.norm(aligned_camera.T_ref_from_camera[:3,3]-anchor.camera.T_ref_from_camera[:3,3])/anchor.median_depth),
        interpretation='observed viewpoint difference, not an error or a small-motion requirement')
    if any(e['median_px']>settings.reprojection_median_max_px or e['p90_px']>settings.reprojection_p90_max_px for e in row['bidirectional_reprojection']):
        row['reason']='bidirectional_reprojection_failed'
        return None,row
    row.update(status='verified_history_pair',reason=None,T_reference_from_local=transform.tolist())
    return transform,row


def recover_reference(current_views,anchors,settings,geom):
    """At most 4x4 pairs, no model/readback loop, no identity or link update."""
    if len(anchors)>settings.max_anchors or len(current_views)>4:
        raise ValueError('unbounded history matching request')
    if anchors and (not current_views or max(v.frame_id for v in anchors)>max(v.frame_id for v in current_views)):
        raise ValueError('future anchor used beyond current watermark')
    if len({v.camera.reference for v in current_views})>1:
        raise ValueError('current views must share their VGGT local reference')
    rows,accepted=[],[]
    for current in current_views:
        def priority(anchor):
            names=len(set(current.names)&set(anchor.names))
            a,b=current.representative_features,anchor.representative_features
            appearance=float(np.max(a@b.T)) if len(a) and len(b) else -1.
            return (-names,-appearance,anchor.frame_id)
        for anchor in sorted(anchors,key=priority):
            transform,row=verify_pair(current,anchor,settings,geom)
            rows.append(row)
            if transform is not None:
                accepted.append((transform,anchor))
    report=dict(status='pending_alignment',reason='no_verified_history_pair',pairs=rows,model_calls=0)
    if not accepted:
        return None,None,report
    reference=accepted[0][1].camera.reference
    # All supporting current frames are in the same VGGT local reference. Reject
    # conflicting accepted anchors instead of picking the prettiest consensus.
    base,anchor=accepted[0]
    for other,view in accepted[1:]:
        relative=other @ np.linalg.inv(base)
        scale=np.cbrt(np.linalg.det(relative[:3,:3]))
        angle=float(np.degrees(np.arccos(np.clip((np.trace(relative[:3,:3]/scale)-1)/2,-1,1))))
        if (view.camera.reference!=reference or abs(scale-1)>settings.consensus_scale_relative_max
                or angle>settings.consensus_rotation_max_deg
                or np.linalg.norm(relative[:3,3])>settings.consensus_translation_depth_max*anchor.median_depth):
            report['reason']='conflicting_verified_history_transforms'
            return None,None,report
    report.update(status='historical_relocalized',reason=None,reference=reference,verified_pairs=len(accepted))
    return base,reference,report
