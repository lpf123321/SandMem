from copy import deepcopy
import json
import hashlib

import pytest

from sandbox_vlm.qa.complete_sandbox import (complete_records,assert_original_records_preserved,
    local_correspondences,complete_messages,NOTICE,display_geometry)


def state():
    return dict(instances=[dict(visual_instance_id=i,name='chair',description='完整 description '+str(i)*90)
                          for i in range(2)],viewpoints={
        '1':[dict(visual_instance_id=0,evidence=dict(frame_id=1,bbox_xyxy=[1,2,30,40],
            source_hw=[100,100],sam_mask_nonempty=True))],
        '3':[dict(visual_instance_id=1,evidence=dict(frame_id=3,bbox_xyxy=[4,5,30,40],
            source_hw=[100,100],sam_mask_nonempty=True))]},observation_index={'a':0,'b':1})


def test_full_records_do_not_truncate_descriptions_or_observation_links():
    s=state();s['viewpoints']['2']=deepcopy(s['viewpoints']['1']);s['viewpoints']['2'][0]['evidence']['frame_id']=2
    result=complete_records(s,[1,3],3)
    assert result['records'][0]==[0,'chair',s['instances'][0]['description'],[1,2]]
    assert len(result['records'])==2
    assert [r[1]for r in result['provided_frame_regions']]==[1,3]
    assert 'ambiguities'not in result
    with pytest.raises(ValueError,match='future'):
        complete_records(s,[1],1)


@pytest.mark.parametrize('mode',['full','descriptions','links','columnar'])
def test_original_four_encodings_are_losslessly_included(mode):
    payload=complete_records(state(),[1,3],3)
    if mode=='full':
        rows=[dict(visual_id=i,name='chair',description='完整 description',actual_frame_ids=[f])for i,f in [(0,1),(1,3)]];fields=None
    elif mode=='descriptions':
        rows=[[0,'chair','完整',[1]],[1,'chair','完整',[3]]];fields=['visual_id','name','short_description','actual_frame_ids']
    elif mode=='links':
        rows=[[0,'chair',1,1],[1,'chair',3,3]];fields=['visual_id','name','first_actual_frame_id','last_actual_frame_id']
    else:
        rows=dict(name_table=['chair'],rows=[[0,1],[0,3]]);fields=None
    original=dict(hypotheses=rows,record_fields=fields)
    assert_original_records_preserved(original,payload)
    payload['records'][0][3]=[]
    with pytest.raises(ValueError,match='observation link'):
        assert_original_records_preserved(original,payload)


def test_correspondence_requires_actual_endpoints_and_does_not_merge_IDs():
    s=state();before=deepcopy(s)
    keys=[hashlib.sha256(json.dumps(s['viewpoints'][str(f)][0]['evidence'],sort_keys=True).encode()).hexdigest()for f in [1,3]]
    edge=dict(source='F1_O0',target='F3_O0',source_frame=1,target_frame=3,source_key=keys[0],target_key=keys[1],
        would_reuse=True,reason='unique_continuous_mask_correspondence',continuous_nonempty=True,
        endpoint_prompt_nonempty=True,mask_iou=.9)
    result,excluded=local_correspondences(s,[edge,edge],{1,3},3)
    assert result['rows']==[[0,1,1,3,.9]] and not excluded and s==before
    result,excluded=local_correspondences(s,[edge],{1},3)
    assert not result['rows'] and excluded['endpoint_not_in_original_QA_frames']==1
    with pytest.raises(ValueError,match='future'):
        local_correspondences(s,[edge],{1},1)
    broken=deepcopy(edge);broken['source_key']='wrong'
    with pytest.raises(ValueError,match='actual-region hash'):
        local_correspondences(s,[broken],{1,3},3)
    conflict=deepcopy(edge);conflict['merge_status']='same_frame_conflict'
    assert not local_correspondences(s,[conflict],{1,3},3)[0]['rows']


def test_messages_distinguish_diagram_and_chronological_frames():
    s=state();payload=complete_records(s,[1,3],3);a,b,diagram=object(),object(),object()
    question='Original question?\nA. One\nB. Two'
    msg,images=complete_messages(dict(question=question,sparse=True,frame_ids=[1,3]),[a,b],diagram,payload)
    content=msg[0]['content'];text='\n'.join(c['text']for c in content if c['type']=='text')
    assert images==[diagram,a,b]
    assert [c['image']for c in content if c['type']=='image']==images
    assert 'Geometry is unavailable for this run.'not in text and NOTICE in text
    assert json.dumps(payload,ensure_ascii=False,separators=(',',':')) in text
    assert 'Original question (verbatim):\n'+question in text
    assert next(i for i,c in enumerate(content)if c.get('text')=='Actual video frames in chronological order:')>2
    direct,_=complete_messages(dict(question=question,sparse=True,frame_ids=[1,3],direct_answer=True),[a,b],diagram,payload)
    text='\n'.join(c['text']for c in direct[0]['content']if c['type']=='text')
    assert 'FINAL_ANSWER'not in text and 'evidence notes'not in text
    assert 'without explanation' in text


def test_display_uses_actual_single_observation_box_not_temporal_union():
    import numpy as np
    from sandbox_vlm.memory import Sandbox,SpatialExtent,ObjectInstance,AppearanceFeature,Viewpoint
    from sandbox_vlm.memory.geometry import BBox3D,CameraParams
    box=BBox3D([0,0,2],[10,10,10],np.eye(3),'r')
    actual=BBox3D([0,0,2],[1,1,1],np.eye(3),'r')
    obj=ObjectInstance(0,'chair','desc',box,AppearanceFeature([1.,0.]))
    cam=CameraParams(1,np.eye(4),[[1,0,0],[0,1,0],[0,0,1]],(100,100),'r')
    sandbox=Sandbox([SpatialExtent(0,'r',box,[obj],[Viewpoint(1,cam,[0])])],video_id='v',retain_unresolved=False)
    frozen=dict(sandbox=sandbox.to_dict(),audit=[dict(instance_id=0,frame_id=1,observation_id='F1_O0',bbox_3d=actual.to_dict())])
    before=deepcopy(frozen);display,source=display_geometry(frozen,[1])
    assert display.get_instance(0).bbox_3d.extents.tolist()==[1.,1.,1.]
    assert source['rows']==[[0,1,'F1_O0']] and frozen==before
