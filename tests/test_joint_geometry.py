from copy import deepcopy
import numpy as np
import pytest
from sandbox_vlm.memory.joint_geometry import merge_correspondences,observations
from test_short_track_memory import state_for,edge


def geo_edge(obs):
    a,b=obs[:2];sf=a['evidence']['frame_id'];tf=b['evidence']['frame_id'];ref='one_forward_only'
    packet=dict(reference=ref,available_through=tf,frame_ids=list(np.rint(np.linspace(sf,tf,16)).astype(int)),
        source=dict(evidence=a['evidence'],bbox_3d=dict(frame=ref)),target=dict(evidence=b['evidence'],bbox_3d=dict(frame=ref)))
    return dict(source=a['observation_id'],target=b['observation_id'],source_frame=sf,target_frame=tf,
        would_reuse=True,evidence_type='D063_joint_ambiguity_geometry',reference=ref,geometry_evidence=packet,
        match_result=dict(status='matched',matched_instance_id=0,scores=[dict(instance_id=0,passes_numeric_gates=True)]))


def test_joint_geometry_allows_long_gap_preserves_base_and_sam_contract():
    old=state_for([0,65]);before=deepcopy(old.to_dict());obs=observations(old);e=geo_edge(obs)
    new,audit=merge_correspondences(old,obs,[e],watermark=65)
    assert len(new.instances)==1 and old.to_dict()==before and audit['removed_hypotheses']==1
    assert 'continuous_nonempty' not in e and 'mask_iou' not in e
    with pytest.raises(ValueError,match='SAM'):merge_correspondences(old,obs,[edge(obs[0]['observation_id'],obs[1]['observation_id'])],watermark=65)


@pytest.mark.parametrize('fault',['future','missing_endpoint','wrong_ref','not_matched','detached'])
def test_invalid_joint_evidence_cannot_merge(fault):
    old=state_for([0,65]);obs=observations(old);e=geo_edge(obs)
    if fault=='future':e['geometry_evidence']['available_through']=66
    if fault=='missing_endpoint':e['geometry_evidence']['frame_ids'][-1]=64
    if fault=='wrong_ref':e['geometry_evidence']['target']['bbox_3d']['frame']='other'
    if fault=='not_matched':e['match_result']['status']='pending'
    if fault=='detached':e['geometry_evidence']['target']['evidence']=dict(e['geometry_evidence']['target']['evidence'],bbox_xyxy=[0,0,1,1])
    with pytest.raises(ValueError):merge_correspondences(old,obs,[e],watermark=65)


def test_joint_geometry_still_rejects_transitive_covisibility():
    old=state_for([0,65,65]);obs=observations(old);e=geo_edge(obs);other=geo_edge([obs[0],obs[2]])
    new,audit=merge_correspondences(old,obs,[e,other],watermark=65)
    assert len(new.instances)==2 and audit['edges'][1]['merge_status']=='same_frame_conflict'
