from copy import deepcopy
import pytest
from sandbox_vlm.qa.complete_sandbox_repair import repair_geometry
from sandbox_vlm.qa.complete_sandbox import complete_messages
from sandbox_vlm.qa.map_only import indexed_messages


def example():
    box=dict(center=[0,0,2],extents=[1,1,1],axes=[[1,0,0],[0,1,0],[0,0,1]],frame='new')
    payload=dict(records=[[0,'chair','original complete description',[1]],[1,'figure','original figure description',[9]]],
        provided_frame_regions=[[0,1,[1,2,3,4],[20,20]],[1,9,[5,6,7,8],[20,20]]],
        plotted_ids=[0],unplotted_ids=[1],box_observation_sources=dict(rows=[[0,1,'F1_O0']]),local_references=['old'],
        local_SAM_correspondences=dict(rows=[]))
    cache=dict(packets=[dict(source_frame=9,coordinate_reference='new',bounds=box,camera=dict(reference='new'),
        observations=[dict(instance_id=1,frame_id=9,observation_id='F9_O0',bbox_3d=box)])])
    return payload,dict(sandbox=dict(spatial_extents=[])),cache


def test_keep_all_original_coordinates_records_and_append_only_missing_ids():
    payload,original,cache=example();before=deepcopy((payload,original,cache))
    result,g=repair_geometry(payload,original,cache)
    assert (payload,original,cache)==before
    assert g['original_D105']==original and g['added_ids']==[1] and g['new_physical_instances']==0
    assert result['records']==payload['records'] and result['provided_frame_regions']==payload['provided_frame_regions']
    assert result['box_observation_sources']['rows']==[[0,1,'F1_O0'],[1,9,'F9_O0']]
    assert result['plotted_ids']==[0,1] and result['unplotted_ids']==[]
    # A complete input must remain exactly unchanged and needs no new model call.
    unchanged,no_geometry=repair_geometry(result,original,cache)
    assert unchanged==result and no_geometry is None
    cache['packets'][0]['observations'][0]['bbox_3d']['frame']='wrong'
    with pytest.raises(ValueError,match='reference'):repair_geometry(payload,original,cache)


def test_only_map_and_geometry_metadata_can_change_the_D105_messages():
    payload,original,cache=example();updated,_=repair_geometry(payload,original,cache)
    snapshot=dict(question='Unchanged original question.',sparse=True,direct_answer=True,frame_ids=[1,9])
    images=[object(),object()];diagram=object()
    before,joint=complete_messages(snapshot,images,diagram,payload)
    after,joint2=complete_messages(snapshot,images,diagram,updated)
    a=indexed_messages(before,joint,['map','a','b']);b=indexed_messages(after,joint2,['newmap','a','b'])
    changed=[(i,x,y)for i,(x,y)in enumerate(zip(a[0]['content'],b[0]['content']))if x!=y]
    assert [i for i,_,_ in changed]==[2,3]
    assert 'without explanation' in b[0]['content'][0]['text']
    assert 'FINAL_ANSWER'not in str(b) and 'brief evidence notes'not in str(b)
