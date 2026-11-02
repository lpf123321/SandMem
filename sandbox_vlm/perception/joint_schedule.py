"""D065 answer-free chronological scheduling, at most two joint-view attempts."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from sandbox_vlm.memory.short_track import observations,observation_key
from sandbox_vlm.alignment.association import names_compatible


def add_masks(catalog,folder,watermark):
    for path in sorted(Path(folder).glob('*_masks.npz')):
        with np.load(path,allow_pickle=False) as archive:
            for key in archive.files:
                if key.startswith('prompt__'):
                    catalog.setdefault(key.removeprefix('prompt__'),dict(path=str(path),key=key,available_through=watermark))


def select_pairs(base,sam,catalog,*,watermark,available_frame_ids,reads_memory,attempted,remaining,appearance_cosine_min=.6):
    if not reads_memory or remaining<=0:return [],dict(reason='memory_not_read_or_quota_exhausted',deferred=0)
    obs=observations(sam);base_byid={o['observation_id']:o for o in observations(base)};byframe={};frames={}
    for o in obs:
        fid=o['evidence']['frame_id'];byframe.setdefault(fid,[]).append(o);frames.setdefault(o['visual_instance_id'],set()).add(fid)
    if any(fid>watermark for fid in byframe):raise ValueError('future scheduler state')
    options=[];deferred=0;seen=set()
    for note in sam.ambiguities:
        gids=note['candidate_visual_instance_ids']
        if len(gids)!=1:continue
        fid=note['evidence']['frame_id'];targets=[o for o in byframe.get(fid,[]) if o['evidence']==note['evidence']]
        if len(targets)!=1:continue
        target=targets[0];gid=gids[0]
        if gid==target['visual_instance_id']:continue
        earlier=[o for o in obs if o['visual_instance_id']==gid and o['evidence']['frame_id']<fid]
        if not earlier:continue
        source=max(earlier,key=lambda o:o['evidence']['frame_id']);sf=source['evidence']['frame_id']
        key=observation_key(source)+'/'+observation_key(target)
        if key in attempted or key in seen:continue
        seen.add(key)
        if (source['observation_id'] not in catalog or target['observation_id'] not in catalog
                or frames[gid]&frames[target['visual_instance_id']] or not names_compatible(source['name'],target['name'],'exact')):continue
        a=np.asarray(sam.instances[gid]['appearance']['vector'],dtype=float);b=np.asarray(sam.instances[target['visual_instance_id']]['appearance']['vector'],dtype=float)
        norm=np.linalg.norm(a)*np.linalg.norm(b);cosine=float(a@b/norm) if norm>0 else -1.
        if cosine<appearance_cosine_min:continue
        ids=np.rint(np.linspace(sf,fid,16)).astype(int).tolist() if fid-sf>=15 else list(range(sf,sf+16))
        if max(ids)>watermark or not set(ids)<=set(available_frame_ids):deferred+=1;continue
        endpoints=[]
        for original in [source,target]:
            item=base_byid[original['observation_id']]
            if item['evidence']!=original['evidence']:raise ValueError('unstable actual observation ID')
            endpoints.append(dict(item,appearance=deepcopy(base.instances[item['visual_instance_id']]['appearance'])))
        masks=[deepcopy(catalog[x['observation_id']]) for x in [source,target]]
        if any(x['available_through']>watermark for x in masks):raise ValueError('future mask')
        options.append(dict(source=endpoints[0],target=endpoints[1],source_mask=masks[0],target_mask=masks[1],
            frame_ids=ids,available_through=watermark,pair_key=key,scheduling_cosine=cosine))
    options.sort(key=lambda x:(x['target']['evidence']['frame_id'],x['source']['evidence']['frame_id']))
    return options[:remaining],dict(deferred=deferred,eligible=len(options),selected=min(remaining,len(options)))
