"""D106 repairs: chronological evidence and geometry at actual QA observations.

Local render packets refer to existing IDs; they never create or merge identities.
"""
import json

from sandbox_vlm.qa.complete_sandbox import NOTICE
from sandbox_vlm.qa.prompt import messages_for


BACKGROUND_NOTICE = ('Background-only observations do not establish that the main entity '
                     'in the question appeared in that frame.')


def local_window(frame_id, watermark):
    if type(frame_id) is not int or not 0 <= frame_id <= watermark:
        raise ValueError('future or invalid source frame')
    end = min(frame_id + 7, watermark)
    return list(range(max(0, end - 15), end + 1))


def observation_messages(snapshot, images, diagram, payload):
    """Keep the historical density-based answer policy, original question and pixels."""
    sparse = snapshot['sparse']
    base = messages_for(snapshot['question'], images, snapshot['frame_ids'],
                        'brief_evidence' if sparse else 'question_last')
    content = base[0]['content']
    policy = content[0]['text'].replace('The following actual video frames are in chronological order. ', '')
    notice = NOTICE + '\n' + BACKGROUND_NOTICE
    if sparse:
        notice += ('\nEach observation panel has its own local reference and is labelled by its actual '
                   'source frame. I<number>@F<number> means an observation of an existing record, '
                   'not an additional object. Panel placement conveys no physical layout. '
                   'The original question\'s direct-answer requirement applies to the FINAL_ANSWER '
                   'line; the preceding brief evidence notes are separate.')
    names = {r[0]: r[1] for r in payload['records']}
    links = {fid: [] for fid in snapshot['frame_ids']}
    for gid, fid, box, hw in payload['provided_frame_regions']:
        links[fid].append(f'I{gid} ({names[gid]})')
    for block in content[1:]:
        if block['type'] == 'text':
            for fid, labels in links.items():
                if block['text'] == f'Actual source frame ID {fid}:':
                    block['text'] += '\nStored actual region links: ' + (', '.join(labels) or 'none') + '.'
                    break
    header = [dict(type='text', text=policy+'\n'+notice),
              dict(type='text', text='Estimated 3D sandbox diagram:'), dict(type='image', image=diagram),
              dict(type='text', text='Complete sandbox records and observation evidence (JSON data):\n'
                   +json.dumps(payload, ensure_ascii=False, separators=(',', ':'))),
              dict(type='text', text='Actual video frames in chronological order:')]
    return [dict(role='user', content=header+content[1:])], [diagram, *images]


def observation_packet(raw, source_frame, rows, masks):
    """AABBs from each actual source mask; no union across times or references."""
    import numpy as np
    from sandbox_vlm.perception.image_geometry import vggt_grid_transforms
    from sandbox_vlm.perception.observations import backproject_masked_depth, estimate_local_aabb
    index = [c.frame_id for c in raw['cameras']].index(source_frame)
    camera = raw['cameras'][index]
    depth, conf = raw['depth'][index], raw['confidence'][index]
    audit = []
    for row in rows:
        ev = row['evidence']; oid = row['observation_id']
        if ev['frame_id'] != source_frame or not ev['sam_mask_nonempty']:
            raise ValueError('unbacked source observation')
        grid = vggt_grid_transforms([tuple(ev['source_hw'])], mode='crop')[0]
        if grid.output_hw != camera.image_size:
            raise ValueError('source mask/camera grid mismatch')
        mask = grid.mask_to_grid(masks[oid])
        valid = mask & np.isfinite(conf) & np.isfinite(depth) & (depth > 0)
        if valid.any():
            valid &= conf >= np.quantile(conf[valid], .5, method='linear')
        cloud = backproject_masked_depth(depth, mask, camera, reliable_pixels=valid)
        estimate = estimate_local_aabb([cloud], coordinate_reference=camera.reference, min_points=30)
        audit.append(dict(instance_id=row['visual_instance_id'], observation_id=oid, frame_id=source_frame,
                          bbox_3d=estimate.bbox_3d.to_dict() if estimate.bbox_3d else None,
                          reliable_points=estimate.point_count, failure=estimate.failure_reason))
    # Explicitly observed source-frame extent, using the same confidence rule.
    valid = np.isfinite(conf) & np.isfinite(depth) & (depth > 0)
    if valid.any():
        valid &= conf >= np.quantile(conf[valid], .5, method='linear')
    scene = backproject_masked_depth(depth, np.ones(depth.shape, dtype=bool), camera, reliable_pixels=valid)
    extent = estimate_local_aabb([scene], coordinate_reference=camera.reference, min_points=30)
    return dict(source_frame=source_frame, coordinate_reference=camera.reference, camera=camera.to_dict(),
                bounds=extent.bbox_3d.to_dict() if extent.bbox_3d else None,
                extent_failure=extent.failure_reason, observations=audit,
                geometry_kind='estimated_source_mask_depth_AABB_not_cross_view_identity')


def render_observations(packets):
    """One small 3D panel per actual observation frame, chronological display only."""
    from io import BytesIO
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image
    from sandbox_vlm.memory.geometry import BBox3D, CameraParams
    if not 1 <= len(packets) <= 8:
        raise ValueError('fixed sparse observation panel budget')
    if [p['source_frame'] for p in packets] != sorted(set(p['source_frame'] for p in packets)):
        raise ValueError('observation panels must be chronological and unique')
    edges = [(a,b) for a in range(8) for b in range(a+1,8) if (a^b) in (1,2,4)]
    fig = plt.figure(figsize=(12.8,9.6), dpi=100, facecolor='white')
    try:
        cols = min(4, len(packets)); rows = (len(packets)+cols-1)//cols
        for i, packet in enumerate(packets):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            reference = packet['coordinate_reference']; fid = packet['source_frame']; points = []
            def box(data, color, **kwargs):
                bounds = BBox3D.from_dict(data)
                if bounds.frame != reference:
                    raise ValueError('mixed panel coordinates')
                c = bounds.corners(); points.append(c)
                ax.add_collection3d(Line3DCollection([(c[a],c[b]) for a,b in edges], colors=color,
                                                       linewidths=1, **kwargs))
                return bounds
            if packet['bounds'] is not None:
                box(packet['bounds'], 'grey', linestyles='dashed', alpha=.25)
            for obj in packet['observations']:
                if obj['bbox_3d'] is None:
                    continue
                gid = obj['instance_id']; color = plt.get_cmap('tab20')(gid%20)
                bounds = box(obj['bbox_3d'], color)
                ax.text(*bounds.center, f'I{gid}@F{fid}', color=color, fontsize=7)
            camera = CameraParams.from_dict(packet['camera'])
            if camera.reference != reference or camera.frame_id != fid:
                raise ValueError('camera/source reference mismatch')
            p = camera.T_ref_from_camera[:3,3]
            points.append(p[None,:]); cloud = np.concatenate(points)
            span = max(float(np.ptp(cloud,axis=0).max()),1e-5)
            forward = camera.T_ref_from_camera[:3,2]*span*.08
            ax.scatter(*p,c='black',s=9); ax.quiver(*p,*forward,color='black')
            cloud = np.concatenate([cloud,(p+forward)[None,:]])
            center=(cloud.min(0)+cloud.max(0))/2; radius=max(float(np.ptp(cloud,axis=0).max())/2,1e-5)*1.1
            for setter,value in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):
                setter(value-radius,value+radius)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-60)
            ax.set_title(f'F{fid} | independent local reference',fontsize=9)
            ax.set_xlabel('X',fontsize=7);ax.set_ylabel('Y',fontsize=7);ax.set_zlabel('Z',fontsize=7)
            ax.tick_params(labelsize=5)
        fig.suptitle('3D sandbox observations in video order | same I = same stored record',fontsize=12)
        fig.text(.02,.025,'Each panel: its own estimated coordinates, arbitrary scale. Panel layout is not physical layout.\n'
                 'Full names, descriptions and actual frame links are provided in adjacent text.',fontsize=10)
        fig.subplots_adjust(left=.02,right=.98,bottom=.10,top=.92,wspace=.15,hspace=.20)
        buf=BytesIO();fig.savefig(buf,format='png',dpi=100);buf.seek(0)
        with Image.open(buf) as im:image=im.convert('RGB').copy()
    finally:
        plt.close(fig)
    return image
