"""D105: complete readable records beside estimated 3D and original images.

No new association decisions, invented coordinates, pending pool, or solver.
"""
import json
import hashlib
from copy import deepcopy

from sandbox_vlm.qa.prompt import messages_for
from sandbox_vlm.qa.render import SPACE_NOTICE


NOTICE = (
    'The sandbox diagram is a rendered annotation, not a video frame. '
    'The adjacent text contains every instance record with its complete stored description '
    'and actual observation-frame links. IDs are estimated records, not known physical identities. '
    'Only plotted boxes share the stated local estimated VGGT reference; scale is arbitrary. '
    'Unplotted records have no box in this input, not evidence that their objects are absent. '
    'Image-region coordinates refer to the original source image size listed with each region. '
    'Local SAM correspondences are predicted short continuous-region matches between the two '
    'provided actual observations, not independently verified global identities or 3D alignment. '
    'Use the records, region references, local matches and real images together to check identity.\n'
    + SPACE_NOTICE)


def complete_records(state, frame_ids, watermark):
    """Keep all names/descriptions/links; expose regions only in provided images."""
    rows = state['instances']
    ids = [r['visual_instance_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate instance IDs')
    links = {i: set() for i in ids}
    regions = []
    provided = set(frame_ids)
    for fid, observed in state['viewpoints'].items():
        fid = int(fid)
        if fid > watermark:
            raise ValueError('future actual observation')
        for row in observed:
            gid, evidence = row['visual_instance_id'], row['evidence']
            if gid not in links or evidence['frame_id'] != fid or not evidence['sam_mask_nonempty']:
                raise ValueError('unbacked actual region')
            links[gid].add(fid)
            if fid in provided:
                regions.append([gid, fid, deepcopy(evidence['bbox_xyxy']), deepcopy(evidence['source_hw'])])
    if any(not frames for frames in links.values()):
        raise ValueError('instance without actual observation')
    return dict(record_fields=['id', 'name', 'description', 'actual_frame_ids'],
        records=[[r['visual_instance_id'], r['name'], r['description'], sorted(links[r['visual_instance_id']])]
                 for r in rows],
        region_fields=['id', 'frame_id', 'source_bbox_xyxy', 'source_hw'],
        provided_frame_regions=sorted(regions, key=lambda r: (r[1], r[0])))


def assert_original_records_preserved(original, payload):
    """Decode all four historical QA encodings and verify information inclusion."""
    if original is None:
        return
    records = original['hypotheses']
    if isinstance(records, dict):
        names = records['name_table']
        row_ids=records.get('visual_id_table',list(range(len(records['rows']))))
        if len(row_ids)!=len(records['rows']) or len(set(row_ids))!=len(row_ids):
            raise ValueError('invalid columnar instance ID table')
        decoded = [dict(id=row_ids[i], name=names[r[0]], first=r[1], last=r[2] if len(r)>2 else r[1])
                   for i, r in enumerate(records['rows'])]
    elif records and isinstance(records[0], dict):
        decoded = [dict(id=r['visual_id'], name=r['name'], description=r['description'],
                        frames=r['actual_frame_ids']) for r in records]
    else:
        fields = original['record_fields'] or []
        decoded = []
        for r in records:
            if 'short_description' in fields:
                decoded.append(dict(id=r[0], name=r[1], description=r[2], frames=r[3]))
            else:
                decoded.append(dict(id=r[0], name=r[1], first=r[2], last=r[3]))
    full = {r[0]: r for r in payload['records']}
    if set(full) != {r['id'] for r in decoded}:
        raise ValueError('original instance cardinality changed')
    for old in decoded:
        row = full[old['id']]
        if row[1] != old['name'] or not row[2].startswith(old.get('description', '')):
            raise ValueError('original name or description lost')
        if not set(old.get('frames', [old.get('first'), old.get('last')])) <= set(row[3]):
            raise ValueError('original observation link lost')


def local_correspondences(state, edges, frame_ids, watermark):
    """Resolve old IDs from actual endpoint observations, never union the IDs."""
    result, excluded, seen = [], {}, set()
    for edge in edges:
        if not edge.get('would_reuse'):
            continue
        if edge.get('reason') != 'unique_continuous_mask_correspondence':
            continue
        if edge.get('merge_status') not in (None, 'merged', 'already_shared'):
            excluded['inherited_identity_conflict'] = excluded.get('inherited_identity_conflict', 0)+1
            continue
        a, b = edge['source_frame'], edge['target_frame']
        reason = None
        if max(a, b) > watermark:
            raise ValueError('future correspondence')
        if a == b:
            raise ValueError('same-frame correspondence is not identity evidence')
        if a not in frame_ids or b not in frame_ids:
            reason = 'endpoint_not_in_original_QA_frames'
        if not edge.get('continuous_nonempty') or not edge.get('endpoint_prompt_nonempty'):
            raise ValueError('unaccepted local correspondence')
        if reason:
            excluded[reason] = excluded.get(reason, 0)+1
            continue
        ids = []
        for key, fid in [('source', a), ('target', b)]:
            oid = edge[key]
            if not oid.startswith(f'F{fid}_O'):
                raise ValueError('endpoint source identity differs')
            j = int(oid.rsplit('_O', 1)[1])
            rows = state['viewpoints'].get(str(fid), [])
            link = next((v for k, v in enumerate(rows) if v.get('observation_slot', k) == j), None)
            if link is None:
                if 'next_visual_instance_id' not in state:
                    raise ValueError('correspondence endpoint lacks actual observation')
                reason = 'endpoint_removed_by_memory_retention'
                break
            gid = link['visual_instance_id']
            evidence = link['evidence']
            digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
            if key+'_key' in edge and edge[key+'_key'] != digest:
                raise ValueError('endpoint actual-region hash differs')
            ids.append(gid)
        if reason:
            excluded[reason] = excluded.get(reason, 0)+1
            continue
        pair = (edge['source'], edge['target'])
        if pair not in seen:
            seen.add(pair)
            result.append([ids[0], ids[1], a, b, edge['mask_iou']])
    return dict(fields=['source_id', 'target_id', 'source_frame', 'target_frame', 'SAM_mask_iou'],
                rows=sorted(result, key=lambda r: (r[2], r[3], r[0], r[1]))), excluded


def display_geometry(frozen, frame_ids):
    """One existing actual observation box per old ID, no temporal box union."""
    from sandbox_vlm.memory.geometry import BBox3D
    from sandbox_vlm.memory import Sandbox
    data = deepcopy(frozen['sandbox'])
    chosen = {}
    for row in sorted(frozen['audit'], key=lambda r: (r['frame_id'], r['observation_id'])):
        if row['bbox_3d'] is not None and row['frame_id'] in frame_ids:
            chosen.setdefault(row['instance_id'], row)
    provenance = []
    for extent in data['spatial_extents']:
        objects = []
        for obj in extent['instances']:
            gid = obj['global_instance_id']
            if gid not in chosen:
                continue
            source = chosen[gid]
            if source['bbox_3d']['frame'] != extent['coordinate_reference']:
                raise ValueError('box/reference mismatch')
            obj['bbox_3d'] = deepcopy(source['bbox_3d'])
            objects.append(obj)
            provenance.append([gid, source['frame_id'], source['observation_id']])
        extent['instances'] = objects
        ids = {o['global_instance_id'] for o in objects}
        extent['viewpoints'] = [v for v in extent['viewpoints'] if v['frame_id'] in frame_ids]
        for view in extent['viewpoints']:
            view['observed_instance_ids'] = [i for i in view['observed_instance_ids'] if i in ids]
    # Frozen estimated range remains intact; positions from different calls are never mixed.
    sandbox = Sandbox.from_dict(data)
    if sandbox.validate() or not sandbox.spatial_extents:
        raise ValueError('actual local scene geometry required')
    return sandbox, dict(fields=['id', 'box_source_frame', 'source_observation'], rows=provenance)


def render_complete(sandbox):
    """Short ID geometry; complete names/descriptions are adjacent native text."""
    from io import BytesIO
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image
    fig = plt.figure(figsize=(7.68, 5.76), dpi=100, facecolor='white')
    edges = [(a,b) for a in range(8) for b in range(a+1,8) if (a^b) in (1,2,4)]
    panels = []
    try:
        for index, extent in enumerate(sandbox.spatial_extents):
            ax = fig.add_subplot(1, len(sandbox.spatial_extents), index+1, projection='3d')
            def box(bounds, color, **kwargs):
                corners = bounds.corners()
                ax.add_collection3d(Line3DCollection([(corners[a],corners[b]) for a,b in edges],
                    colors=color, linewidths=1.1, **kwargs))
            box(extent.bounds, 'grey', linestyles='dashed', alpha=.3)
            for obj in extent.instances:
                color = plt.get_cmap('tab20')(obj.global_instance_id % 20)
                box(obj.bbox_3d, color)
                ax.text(*obj.bbox_3d.center, f'I{obj.global_instance_id}', color=color, fontsize=8)
            length = max(float(np.linalg.norm(extent.bounds.extents))*.055, 1e-6)
            points = [extent.bounds.corners()]
            for view in extent.viewpoints:
                pose = view.camera.T_ref_from_camera
                p = pose[:3,3]; forward = pose[:3,2]*length
                ax.scatter(*p, color='black', s=7)
                ax.quiver(*p, *forward, color='black', arrow_length_ratio=.2)
                ax.text(*p, f'F{view.frame_id}', fontsize=6)
                points.append(np.array([p,p+forward]))
            cloud = np.concatenate(points); lo, hi = cloud.min(0), cloud.max(0)
            center = (lo+hi)/2; radius = max(float((hi-lo).max())/2, 1e-6)*1.08
            for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim),center):
                setter(value-radius,value+radius)
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=25,azim=-60)
            ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z');ax.tick_params(labelsize=6)
            ax.set_title(f'Estimated local reference {index}',fontsize=10)
            panels.append(dict(reference=extent.coordinate_reference,
                ids=sorted(extent.instance_ids()), frame_ids=[v.frame_id for v in extent.viewpoints]))
        fig.suptitle('3D sandbox | I = instance record ID | F = actual frame ID',fontsize=10)
        fig.text(.02,.025,'Full names, descriptions and observation links are supplied in adjacent text. Arbitrary scale.',fontsize=8)
        fig.subplots_adjust(left=.02,right=.98,bottom=.08,top=.88)
        output=BytesIO();fig.savefig(output,format='png',dpi=100);output.seek(0)
        with Image.open(output) as opened:
            image=opened.convert('RGB').copy()
    finally:
        plt.close(fig)
    return image, dict(panels=panels, image_size_hw=[576,768],
        native_text_legend_complete=True, fixed_elevation=25, fixed_azimuth=-60)


def complete_messages(snapshot, images, diagram, payload):
    """One consistent presentation. Preserve original question/output policy."""
    base = messages_for(snapshot['question'], images, snapshot['frame_ids'],
        'brief_evidence' if snapshot['sparse'] and not snapshot.get('direct_answer',False) else 'question_last')
    content = base[0]['content']
    policy = content[0]['text'].replace('The following actual video frames are in chronological order. ', '')
    header = [dict(type='text',text=policy+'\n'+NOTICE),
              dict(type='text',text='Estimated 3D sandbox diagram:'),dict(type='image',image=diagram),
              dict(type='text',text='Complete sandbox records and observation evidence (JSON data):\n'
                   +json.dumps(payload,ensure_ascii=False,separators=(',',':'))),
              dict(type='text',text='Actual video frames in chronological order:')]
    return [dict(role='user',content=header+content[1:])], [diagram,*images]
