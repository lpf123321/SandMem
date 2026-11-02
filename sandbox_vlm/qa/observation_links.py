"""D115: visualize existing observation links and actual regions, without new identities."""
from copy import deepcopy


def observation_panels(geometry, payload):
    records={r[0]:r for r in payload['records']}
    base=geometry.get('original_D105',geometry);panels=[]
    for e in base['sandbox']['spatial_extents']:
        panels.append(dict(reference=e['coordinate_reference'],bounds=deepcopy(e['bounds']),
            objects=[dict(id=o['global_instance_id'],bbox=deepcopy(o['bbox_3d']))for o in e['instances']],
            views=[dict(frame_id=v['frame_id'],camera=deepcopy(v['camera']),
                        ids=list(v['observed_instance_ids']))for v in e['viewpoints']]))
    for e in geometry.get('added_panels',[]):
        panels.append(dict(reference=e['coordinate_reference'],bounds=deepcopy(e['bounds']),
            objects=[dict(id=o['instance_id'],bbox=deepcopy(o['bbox_3d']))for o in e['observations']],
            views=[dict(frame_id=e['source_frame'],camera=deepcopy(e['camera']),
                        ids=[o['instance_id']for o in e['observations']])]))
    for p in panels:
        ids={o['id']for o in p['objects']};assert ids<=set(records)
        p['links']=[]
        for v in p['views']:
            for gid in v['ids']:
                assert gid in records and v['frame_id']in records[gid][3]
                if gid in ids:p['links'].append([v['frame_id'],gid])
    return panels


def render_observation_links(geometry,payload,frame_ids,paths,pixel_hashes):
    """One fixed-size canvas; original raw frames remain separate unmodified inputs.

    Use only the existing sparse-input display branch. Empty graphs keep their
    original image. Every edge is a stored actual observation in one reference.
    """
    panels=observation_panels(geometry,payload)
    if not any(p['links']for p in panels):return None,None
    if len(frame_ids)>8 or not 1<=len(panels)<=8:raise ValueError('sparse display budget exceeded')
    from io import BytesIO
    import textwrap
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import findfont
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image,ImageDraw,ImageFont
    from sandbox_vlm.memory.geometry import BBox3D,CameraParams
    from sandbox_vlm.qa.artifact import pixel_sha256
    edges=[(a,b)for a in range(8)for b in range(a+1,8)if a^b in(1,2,4)]
    fig=plt.figure(figsize=(7.68,9.6),dpi=100,facecolor='white')
    palette=plt.get_cmap('tab20')
    try:
        cols=min(2,len(panels));rows=(len(panels)+cols-1)//cols
        for index,p in enumerate(panels):
            ax=fig.add_subplot(rows,cols,index+1,projection='3d');points=[];centers={}
            def box(data,color,**kwargs):
                b=BBox3D.from_dict(data)
                assert b.frame==p['reference']
                c=b.corners();points.append(c)
                ax.add_collection3d(Line3DCollection([(c[a],c[b])for a,b in edges],colors=color,linewidths=.9,**kwargs))
                return b
            if p['bounds']is not None:box(p['bounds'],'grey',linestyles='dashed',alpha=.18)
            for o in p['objects']:
                gid=o['id'];b=box(o['bbox'],palette(gid%20));centers[gid]=b.center
                observed=sorted({f for f,i in p['links']if i==gid})
                label=f'I{gid}\n'+','.join(f'F{f}'for f in observed)
                ax.text(*b.center,'\n'.join(textwrap.wrap(label.split('\n')[0],24)+
                    textwrap.wrap(label.split('\n')[1]if observed else'No linked view',24)),fontsize=6,color=palette(gid%20))
            cams={v['frame_id']:CameraParams.from_dict(v['camera'])for v in p['views']}
            for cam in cams.values():
                assert cam.reference==p['reference'];points.append(cam.T_ref_from_camera[:3,3][None,:])
            cloud=np.concatenate(points);span=max(float(np.ptp(cloud,axis=0).max()),1e-5)
            for fid,cam in cams.items():
                pos=cam.T_ref_from_camera[:3,3];forward=cam.T_ref_from_camera[:3,2]*span*.07
                ax.scatter(*pos,c='black',s=9);ax.quiver(*pos,*forward,color='black')
                ax.text(*pos,f'F{fid}',fontsize=6);points.append((pos+forward)[None,:])
            for fid,gid in p['links']:
                a=cams[fid].T_ref_from_camera[:3,3];b=centers[gid]
                ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],color=palette(gid%20),linestyle=':',linewidth=1.1)
            cloud=np.concatenate(points);mid=(cloud.min(0)+cloud.max(0))/2;radius=max(float(np.ptp(cloud,axis=0).max())/2,1e-5)*1.1
            for setter,value in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),mid):setter(value-radius,value+radius)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-60);ax.tick_params(labelsize=5)
            ax.set_xlabel('X',fontsize=6);ax.set_ylabel('Y',fontsize=6);ax.set_zlabel('Z',fontsize=6)
            ax.set_title(f'Local reference {index+1}',fontsize=9)
        fig.suptitle('Estimated 3D sandbox\nDotted lines = recorded observations',fontsize=12)
        fig.text(.025,.02,'I = stored record; F = actual frame. Arbitrary scale.\nIndependent references; panel placement is not physical layout.',fontsize=9)
        fig.subplots_adjust(left=.02,right=.98,bottom=.075,top=.92,wspace=.05,hspace=.22)
        buf=BytesIO();fig.savefig(buf,format='png',dpi=100);buf.seek(0)
        with Image.open(buf)as im:left=im.convert('RGB').copy()
    finally:
        fig.clear()
        plt.close(fig)
    canvas=Image.new('RGB',(1280,960),'white');canvas.paste(left,(0,0));left.close()
    draw=ImageDraw.Draw(canvas);font_path=findfont('DejaVu Sans');font=ImageFont.truetype(font_path,12)
    draw.line((768,0,768,960),fill='grey',width=1);insets=[]
    for k,(fid,path,digest)in enumerate(zip(frame_ids,paths,pixel_hashes)):
        x=772+(k%2)*254;y=(k//2)*240
        regions=[r for r in payload['provided_frame_regions']if r[1]==fid]
        ids=sorted({r[0]for r in regions})
        draw.text((x,y+2),f'Actual F{fid}',fill='black',font=font)
        with Image.open(path)as source:
            source=source.convert('RGB')
            assert pixel_sha256(source)==digest
            source.thumbnail((246,184),Image.Resampling.LANCZOS)
            thumb=source.copy();source.close()
        td=ImageDraw.Draw(thumb)
        for gid,_,bbox,hw in regions:
            h,w=hw;sx=thumb.width/w;sy=thumb.height/h
            a,b,c,d=[bbox[0]*sx,bbox[1]*sy,bbox[2]*sx,bbox[3]*sy]
            color=tuple(int(v*255)for v in palette(gid%20)[:3]);td.rectangle((a,b,c,d),outline=color,width=2)
            ly=b-14 if b>=14 else min(d+1,thumb.height-14)
            td.text((max(0,a),max(0,ly)),f'I{gid}',font=font,fill=color,stroke_width=1,stroke_fill='white')
        canvas.paste(thumb,(x,y+21));thumb.close()
        caption='Linked records: '+(', '.join(f'I{i}'for i in ids)if ids else'none recorded')
        for z,line in enumerate(textwrap.wrap(caption,35)[:2]):draw.text((x,y+207+z*14),line,fill='black',font=font)
        insets.append(dict(frame_id=fid,pixel_sha256=digest,region_ids=ids,regions=deepcopy(regions)))
    metadata=dict(panels=panels,frame_insets=insets,image_size_hw=[960,1280],
        geometry_and_links_unchanged=True,new_identity_decisions=0,new_geometry_estimates=0,
        links_are_recorded_observations_not_projected_visibility=True,font_path=font_path,
        renderer_working_reserve_bytes=24*2**20)
    return canvas,metadata
