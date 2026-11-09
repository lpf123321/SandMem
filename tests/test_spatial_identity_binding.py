"""D102's added relation must not silently collapse D116's working IDs."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_spatial_vsibench import d116_identity_and_correspondences
from sandbox_vlm.memory.coverage import CoverageSandbox,ContinuitySettings
from sandbox_vlm.memory.joint_geometry import observations


def test_added_SAM_relation_is_evidence_only_but_legacy_relation_keeps_its_role():
    state=CoverageSandbox('video','question',retain_ambiguities=False,memory_limit_bytes=64*2**20)
    descriptor=np.ones((16,64),dtype=np.float32)/8
    for fid,box in [(0,[1.,1.,20.,20.]),(1,[50.,50.,80.,80.])]:
        state.ingest(video_id='video',name='chair',description='visible object',
            evidence=dict(frame_id=fid,bbox_xyxy=box,source_hw=[100,100],sam_mask_nonempty=True),
            appearance=dict(vector=[1.,0.],dtype='float32',dim=2),context_descriptor=descriptor,
            settings=ContinuitySettings(),allowed_through=fid)
    obs=observations(state);assert len(state.instances)==2
    edge=dict(source=obs[0]['observation_id'],target=obs[1]['observation_id'],
        source_frame=0,target_frame=1,would_reuse=True,mask_iou=.9,continuous_nonempty=True,
        endpoint_prompt_nonempty=True,forward_candidates=[obs[1]['observation_id']],
        co_visible_reverse_candidates=[obs[0]['observation_id']])
    kept,relations=d116_identity_and_correspondences(state,[],[edge],watermark=1)
    assert kept.to_dict()==state.to_dict()
    assert relations['edges'][0]['merge_status']=='merged'
    legacy,_=d116_identity_and_correspondences(state,[edge],[],watermark=1)
    assert len(legacy.instances)==1
