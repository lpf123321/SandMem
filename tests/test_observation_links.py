from copy import deepcopy

from sandbox_vlm.qa.observation_links import observation_panels


def test_old_camera_does_not_get_future_object_links():
    geometry=dict(sandbox=dict(spatial_extents=[dict(coordinate_reference='ref',bounds=None,
        instances=[dict(global_instance_id=3,bbox_3d={'native':True})],viewpoints=[
            dict(frame_id=10,camera={'native':1},observed_instance_ids=[]),
            dict(frame_id=20,camera={'native':2},observed_instance_ids=[3])])]))
    before=deepcopy(geometry);payload=dict(records=[[3,'subject','description',[20]]])
    panels=observation_panels(geometry,payload)
    assert panels[0]['links']==[[20,3]]
    assert panels[0]['views'][0]['ids']==[]
    assert geometry==before


def test_independent_panels_keep_separate_observations():
    g=dict(original_D105=dict(sandbox=dict(spatial_extents=[])),added_panels=[
        dict(coordinate_reference='a',bounds=None,source_frame=10,camera={'reference':'a'},
            observations=[dict(instance_id=1,bbox_3d={'frame':'a'})]),
        dict(coordinate_reference='b',bounds=None,source_frame=20,camera={'reference':'b'},
            observations=[dict(instance_id=2,bbox_3d={'frame':'b'})])])
    p=observation_panels(g,dict(records=[[1,'one','',[10]],[2,'two','',[20]]]))
    assert [x['links']for x in p]==[[[10,1]],[[20,2]]]
    assert [x['reference']for x in p]==['a','b']
