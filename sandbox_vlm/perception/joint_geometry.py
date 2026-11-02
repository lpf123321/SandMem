"""D063 joint-view geometry; each call remains in an independent local reference."""
import numpy as np

def process_joint_geometry(raw,w,video_id,settings):
    from sandbox_vlm.perception.image_geometry import vggt_grid_transforms
    from sandbox_vlm.perception.observations import backproject_masked_depth,estimate_local_aabb,LocalObservation
    from sandbox_vlm.perception.evidence import PromptEvidence
    from sandbox_vlm.alignment.association import match_global,projected_source_box
    from sandbox_vlm.memory.geometry import BBox3D
    from sandbox_vlm.memory.sandbox import AppearanceFeature,ObjectInstance
    ids=[f['frame_id'] for f in w['frames']];ref=w['reference']
    grids=vggt_grid_transforms([tuple(w['source']['evidence']['source_hw'])]*16,mode='crop')
    rows={}
    for role in ['source','target']:
        o=w[role];fid=o['evidence']['frame_id'];i=ids.index(fid)
        with np.load(w[role+'_mask']['path'],allow_pickle=False) as z:mask=grids[i].mask_to_grid(z[w[role+'_mask']['key']])
        depth,conf=raw['depth'][i],raw['confidence'][i];values=conf[mask & np.isfinite(conf)]
        th=float(np.quantile(values,.5,method='linear')) if len(values) else None
        reliable=mask & np.isfinite(conf) & (conf>=th) & np.isfinite(depth) & (depth>0) if th is not None else np.zeros_like(mask)
        points=backproject_masked_depth(depth,mask,raw['cameras'][i],reliable_pixels=reliable)
        est=estimate_local_aabb([points],coordinate_reference=ref,min_points=30)
        rows[role]=dict(**o,bbox_3d=est.bbox_3d.to_dict() if est.bbox_3d else None,
            point_count=points.selected_pixel_count,confidence_threshold=th,failure=est.failure_reason)
    a,b=rows['source'],rows['target'];edges=[];projected=None
    if a['bbox_3d'] is None or b['bbox_3d'] is None:
        result=dict(status='pending',matched_instance_id=None,reason='missing_endpoint_geometry',scores=[])
    else:
        src=ObjectInstance(0,a['name'],'',BBox3D.from_dict(a['bbox_3d']),AppearanceFeature.from_dict(a['appearance']))
        target=LocalObservation(video_id,b['observation_id'],ref,b['name'],'',[b['evidence']['frame_id']],
            BBox3D.from_dict(b['bbox_3d']),AppearanceFeature.from_dict(b['appearance']),prompt_evidence=[b['evidence']])
        cameras={c.frame_id:c for c in raw['cameras']};gridmap=dict(zip(ids,grids))
        result=match_global(target,[src],[PromptEvidence.from_dict(b['evidence'])],cameras,gridmap,settings)
        projected=projected_source_box(src.bbox_3d,cameras[b['evidence']['frame_id']],gridmap[b['evidence']['frame_id']])
        projected=projected.tolist() if projected is not None else None
        if result['status']=='matched':
            packet=dict(reference=ref,available_through=w['available_through'],frame_ids=ids,source=a,target=b)
            edges.append(dict(source=a['observation_id'],target=b['observation_id'],source_frame=a['evidence']['frame_id'],
                target_frame=b['evidence']['frame_id'],would_reuse=True,evidence_type='D063_joint_ambiguity_geometry',
                reference=ref,match_result=result,geometry_evidence=packet))
    return dict(name=w['name'],sample_id=w['sample_id'],reference=ref,available_through=w['available_through'],
        frame_ids=ids,cameras=[c.to_dict() for c in raw['cameras']],observations=rows,match_result=result,
        source_bbox_projected_to_target=projected,edges=edges)

