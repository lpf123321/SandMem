"""D105 geometry repair: keep every original box and add only missing records."""
from copy import deepcopy


def repair_geometry(payload, original_geometry, cached_geometry):
    """Do not replace old coordinates, infer identities, or union local references."""
    result=deepcopy(payload);had=set(payload['plotted_ids']);chosen={}
    for packet in cached_geometry.get('packets',[]):
        for obs in packet['observations']:
            gid=obs['instance_id']
            if obs['bbox_3d']is not None and gid not in had:
                if gid not in chosen or obs['frame_id']<chosen[gid][1]['frame_id']:
                    chosen[gid]=(packet,obs)
    if not chosen:return result,None
    records={r[0]:r for r in payload['records']}
    all_ids=set(records)
    if not set(chosen)<=all_ids:raise ValueError('unknown inherited record')
    panels={}
    sources=[]
    for gid,(packet,obs)in sorted(chosen.items(),key=lambda r:(r[1][1]['frame_id'],r[0])):
        fid=obs['frame_id'];ref=packet['coordinate_reference']
        if fid!=packet['source_frame']or fid not in records[gid][3]:
            raise ValueError('new box lacks actual observation link')
        if obs['bbox_3d']['frame']!=ref or packet['camera']['reference']!=ref:
            raise ValueError('new box and camera reference differ')
        panel=panels.setdefault(ref,dict(coordinate_reference=ref,source_frame=fid,
            bounds=deepcopy(packet['bounds']),camera=deepcopy(packet['camera']),observations=[]))
        panel['observations'].append(deepcopy(obs))
        sources.append([gid,fid,obs['observation_id']])
    result['box_observation_sources']['rows']+=sources
    result['plotted_ids']=sorted(had|set(chosen))
    result['unplotted_ids']=sorted(all_ids-set(result['plotted_ids']))
    result['local_references']+=list(panels)
    geometry=dict(original_D105=deepcopy(original_geometry),added_panels=list(panels.values()),
                  added_ids=sorted(chosen),new_physical_instances=0)
    return result,geometry


def render_repaired(geometry):
    """Keep original D105 geometry intact; append independent estimated panels."""
    from io import BytesIO
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image
    from sandbox_vlm.memory.geometry import BBox3D,CameraParams
    panels=[]
    for e in geometry['original_D105']['sandbox']['spatial_extents']:
        panels.append(dict(reference=e['coordinate_reference'],bounds=e['bounds'],
            objects=[dict(id=o['global_instance_id'],bbox=o['bbox_3d'],label=f'I{o["global_instance_id"]}')for o in e['instances']],
            cameras=[v['camera']for v in e['viewpoints']],title='D105 original local reference'))
    for e in geometry['added_panels']:
        fid=e['source_frame']
        panels.append(dict(reference=e['coordinate_reference'],bounds=e['bounds'],
            objects=[dict(id=o['instance_id'],bbox=o['bbox_3d'],label=f'I{o["instance_id"]}@F{fid}')for o in e['observations']],
            cameras=[e['camera']],title=f'Added F{fid} | independent local reference'))
    if not 1<=len(panels)<=8:raise ValueError('fixed panel budget exceeded')
    edges=[(a,b)for a in range(8)for b in range(a+1,8)if a^b in (1,2,4)]
    fig=plt.figure(figsize=(12.8,9.6),dpi=100,facecolor='white')
    try:
        cols=min(4,len(panels));rows=(len(panels)+cols-1)//cols
        for i,panel in enumerate(panels):
            ax=fig.add_subplot(rows,cols,i+1,projection='3d');points=[]
            def box(data,color,**kwargs):
                b=BBox3D.from_dict(data)
                if b.frame!=panel['reference']:raise ValueError('mixed display references')
                c=b.corners();points.append(c)
                ax.add_collection3d(Line3DCollection([(c[a],c[b])for a,b in edges],colors=color,linewidths=1,**kwargs))
                return b
            if panel['bounds']is not None:box(panel['bounds'],'grey',linestyles='dashed',alpha=.25)
            for obj in panel['objects']:
                color=plt.get_cmap('tab20')(obj['id']%20);b=box(obj['bbox'],color)
                ax.text(*b.center,obj['label'],color=color,fontsize=7)
            cams=[CameraParams.from_dict(c)for c in panel['cameras']]
            for c in cams:
                if c.reference!=panel['reference']:raise ValueError('mixed camera references')
                points.append(c.T_ref_from_camera[:3,3][None,:])
            cloud=np.concatenate(points);span=max(float(np.ptp(cloud,axis=0).max()),1e-5)
            for c in cams:
                p=c.T_ref_from_camera[:3,3];forward=c.T_ref_from_camera[:3,2]*span*.08
                ax.scatter(*p,c='black',s=8);ax.quiver(*p,*forward,color='black')
                ax.text(*p,f'F{c.frame_id}',fontsize=6);points.append((p+forward)[None,:])
            cloud=np.concatenate(points);center=(cloud.min(0)+cloud.max(0))/2;radius=max(float(np.ptp(cloud,axis=0).max())/2,1e-5)*1.1
            for setter,value in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):setter(value-radius,value+radius)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-60);ax.tick_params(labelsize=5)
            ax.set_xlabel('X',fontsize=7);ax.set_ylabel('Y',fontsize=7);ax.set_zlabel('Z',fontsize=7)
            ax.set_title(panel['title'],fontsize=9)
        fig.suptitle('3D sandbox | I = stored instance record ID | F = actual source frame',fontsize=12)
        fig.text(.02,.025,'Different panels use independent coordinates; panel placement is not physical layout. Arbitrary scale.\n'
                 'Full original names, descriptions and frame links are provided in adjacent text.',fontsize=10)
        fig.subplots_adjust(left=.02,right=.98,bottom=.1,top=.92,wspace=.15,hspace=.2)
        buf=BytesIO();fig.savefig(buf,format='png',dpi=100);buf.seek(0)
        with Image.open(buf)as opened:image=opened.convert('RGB').copy()
    finally:plt.close(fig)
    return image
