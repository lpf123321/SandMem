from copy import deepcopy
import numpy as np
import pytest
from sandbox_vlm.memory.coverage import CoverageSandbox, ContinuitySettings
from sandbox_vlm.memory.short_track import observations, merge_correspondences


def state_for(frames):
    state = CoverageSandbox('v', 'question')
    for i, fid in enumerate(frames):
        state.ingest(video_id='v', name='chair', description=f'actual {i}',
            evidence=dict(frame_id=fid,bbox_xyxy=[i+1.,1.,i+1.5,2.],source_hw=[100,100],sam_mask_nonempty=True),
            appearance=dict(vector=[1.,0.],dtype='float32',dim=2),
            context_descriptor=np.random.default_rng(i).normal(size=(16,64)).astype(np.float32),
            settings=ContinuitySettings(),allowed_through=fid)
    return state


def edge(a,b):
    return dict(source=a,target=b,source_frame=int(a.split('_')[0][1:]),
        target_frame=int(b.split('_')[0][1:]),mask_iou=.9,
        continuous_nonempty=True,endpoint_prompt_nonempty=True,forward_candidates=[b],
        co_visible_reverse_candidates=[a],would_reuse=True)


def test_union_updates_all_links_notes_and_context_without_mutating_original():
    old=state_for([0,1,2]); before=deepcopy(old.to_dict()); obs=observations(old)
    a,b,c=[o['observation_id'] for o in obs]
    new,audit=merge_correspondences(old,obs,[edge(a,b),edge(b,c)],watermark=2)
    assert old.to_dict()==before and len(new.instances)==1
    assert set(new.observation_index.values())=={0} and not new.ambiguities
    assert new.instances[0]['context_frame_id']==2 and new.validate()
    assert all(v['visual_instance_id']==0 for links in new.viewpoints.values() for v in links)
    assert audit['removed_hypotheses']==2


def test_transitive_same_frame_conflict_is_rejected():
    old=state_for([0,1,1]); obs=observations(old);a,b,c=[o['observation_id'] for o in obs]
    new,audit=merge_correspondences(old,obs,[edge(a,b),edge(a,c)],watermark=1)
    assert len(new.instances)==2 and audit['edges'][1]['merge_status']=='same_frame_conflict'
    assert len({v['visual_instance_id'] for v in new.viewpoints['1']})==2


@pytest.mark.parametrize('change',[dict(continuous_nonempty=False),dict(endpoint_prompt_nonempty=False),
    dict(mask_iou=.69),dict(forward_candidates=[]),dict(co_visible_reverse_candidates=[])])
def test_unsupported_accepted_edge_fails_closed(change):
    old=state_for([0,1]); obs=observations(old);a,b=[o['observation_id'] for o in obs]
    e=edge(a,b);e.update(change)
    with pytest.raises(ValueError):merge_correspondences(old,obs,[e],watermark=1)


def test_future_or_detached_actual_endpoint_cannot_merge():
    old=state_for([0,1]);obs=observations(old);a,b=[o['observation_id'] for o in obs]
    with pytest.raises(ValueError):merge_correspondences(old,obs,[edge(a,b)],watermark=0)
    obs[1]['evidence']['bbox_xyxy'][0]=50
    with pytest.raises(ValueError):merge_correspondences(old,obs,[edge(a,b)],watermark=1)
