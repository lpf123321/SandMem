"""D076 fixed chronological ambiguity anchors and contextual actual-view packing."""
from copy import deepcopy
import numpy as np
from sandbox_vlm.memory.joint_geometry import observations
from sandbox_vlm.perception.coverage import unpack_descriptor, context_similarity
from sandbox_vlm.alignment.association import names_compatible


def select_packet(state, ledger, *, watermark, attempted, excluded_targets):
    obs = observations(state)
    if any(o['evidence']['frame_id'] > watermark for o in obs):
        raise ValueError('future actual observation')
    byframe = {}; component_frames = {}
    for o in obs:
        gid=o['visual_instance_id'];byframe.setdefault(o['evidence']['frame_id'],[]).append(o)
        component_frames.setdefault(gid,set()).add(o['evidence']['frame_id'])
    anchors=[]
    for note in state.ambiguities:
        target=next((o for o in obs if o['evidence']==note['evidence']),None)
        if target is None or target['observation_id'] in excluded_targets:continue
        tf=target['evidence']['frame_id'];tg=target['visual_instance_id']
        for gid in note['candidate_visual_instance_ids']:
            old=[o for o in obs if o['visual_instance_id']==gid and o['evidence']['frame_id']<tf]
            if not old or component_frames[gid]&component_frames[tg]:continue
            source=max(old,key=lambda o:o['evidence']['frame_id']);sf=source['evidence']['frame_id']
            key=(source['observation_id'],target['observation_id'])
            if key in attempted or not names_compatible(source['name'],target['name'],'exact'):continue
            a=np.asarray(state.instances[gid]['appearance']['vector'],float)
            b=np.asarray(state.instances[tg]['appearance']['vector'],float)
            cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
            if cosine<.6:continue
            anchors.append((tf,sf,-cosine,key,source,target))
    if not anchors:return None
    _,_,_,key,source,target=min(anchors,key=lambda x:x[:4])
    endpoints=[source['evidence']['frame_id'],target['evidence']['frame_id']]
    def descriptor(fid):return unpack_descriptor(ledger[str(fid)]['descriptor'])
    anchor_vectors=[descriptor(f)for f in endpoints]
    ranked=sorted(byframe,key=lambda f:(-min(context_similarity(descriptor(f),v)for v in anchor_vectors),f))
    chosen=set(endpoints)
    for fid in ranked:
        if len(chosen)==16:break
        chosen.add(fid)
    # Early prefixes with fewer actual views keep all of them; fill remaining
    # context slots chronologically near the anchor, without reading future.
    if len(chosen)<16:
        for fid in sorted((int(f)for f in ledger if int(f)<=watermark),
                          key=lambda f:(min(abs(f-a)for a in endpoints),f)):
            chosen.add(fid)
            if len(chosen)==16:break
    if len(chosen)!=16:return None
    rows=[]
    for fid in sorted(chosen):
        for o in byframe.get(fid,[]):
            gid=o['visual_instance_id']
            rows.append(dict(deepcopy(o),component_id=gid,appearance=deepcopy(state.instances[gid]['appearance'])))
    return dict(anchor_pair=list(key),frame_ids=sorted(chosen),observations=rows,
        available_through=watermark,video_id=state.video_id,
        component_frames={str(k):sorted(v)for k,v in component_frames.items()},source_hw=list(rows[0]['evidence']['source_hw']))
