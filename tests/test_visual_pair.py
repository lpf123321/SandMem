from copy import deepcopy
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from sandbox_vlm.memory.coverage import CoverageSandbox, ContinuitySettings
from sandbox_vlm.qa.visual_pair import VisualPairVerifier, observation_crop


def actual(fid, *, feature=(1., 0.), box=(10., 10., 30., 30.)):
    return dict(video_id='video',name='chair',description='black chair',
        evidence=dict(frame_id=fid,bbox_xyxy=list(box),source_hw=[100,100],sam_mask_nonempty=True),
        appearance=dict(vector=list(feature),dtype='float32',dim=2),
        context_descriptor=np.ones((16,64),dtype=np.float32)/8,
        settings=ContinuitySettings(),allowed_through=fid)


class Verdict:
    def __init__(self, accepted):self.accepted=accepted;self.calls=[]
    def verify(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        return dict(accepted=self.accepted,status='verified',forward_calls=1)


def test_far_unique_pair_can_reuse_without_inventing_geometry_or_losing_links():
    state=CoverageSandbox('video','question');yes=Verdict(True)
    state.ingest(**actual(0))
    result=state.ingest(**actual(100),pair_verifier=yes)
    assert result['status']=='matched_pair_verified_visual_hypothesis'
    assert len(state.instances)==1 and state.instances[0]['local_geometry']==[]
    assert state.viewpoints['100'][0]['visual_instance_id']==state.viewpoints['0'][0]['visual_instance_id']==0
    assert len(yes.calls)==1 and state.validate()
    assert CoverageSandbox.from_dict(state.to_dict()).to_dict()==state.to_dict()
    # Exact observation reuse does not spend a second pair forward.
    state.ingest(**actual(100),pair_verifier=yes)
    assert len(yes.calls)==1


def test_rejected_pair_and_competing_candidates_do_not_force_identity():
    state=CoverageSandbox('video','question');no=Verdict(False)
    state.ingest(**actual(0))
    state.ingest(**actual(100),pair_verifier=no)
    assert len(state.instances)==2 and len(no.calls)==1 and len(state.ambiguities)==1
    # Now two identical compatible histories make SAM nonunique.
    state.ingest(**actual(200),pair_verifier=no)
    assert len(state.instances)==3 and len(no.calls)==1 and state.validate()


def test_covisible_objects_and_future_boundaries_are_hard_vetoes():
    state=CoverageSandbox('video','question');yes=Verdict(True)
    state.ingest(**actual(0))
    state.ingest(**actual(0,box=(50.,50.,80.,80.)),pair_verifier=yes)
    assert len(state.instances)==2 and not yes.calls
    future=actual(1);future['allowed_through']=0
    with pytest.raises(ValueError):state.ingest(**future,pair_verifier=yes)
    assert not yes.calls and state.validate()


class Reader:
    def encoded_source(self,fid):
        output=io.BytesIO();Image.new('RGB',(100,100),'white').save(output,format='JPEG')
        return output.getvalue(),SimpleNamespace(source_index=fid*30),[100,100]


def bare_verifier():
    obj=object.__new__(VisualPairVerifier)
    obj.reader=Reader();obj.maximum_pairs=2;obj.margin_min=2.;obj.image_side=256
    obj.records=[];obj.audit_sink=None;obj.forward_calls=0
    return obj


def test_pair_budget_counts_failed_attempts_and_serializes_all_state():
    obj=bare_verifier()
    def fail(*args):raise RuntimeError('processor failed')
    obj.score_images=fail
    kwargs=dict(names=['chair','chair'],descriptions=['black','black'],candidate_id=0,allowed_through=100)
    a,b=actual(0)['evidence'],actual(100)['evidence']
    assert obj.verify(a,b,**kwargs)['status']=='verification_failed'
    assert obj.verify(a,b,**kwargs)['status']=='verification_failed'
    assert obj.verify(a,b,**kwargs)['status']=='pair_budget_exhausted'
    assert len(obj.to_dict()['records'])==2 and all(r['forward_calls']==0 for r in obj.records)
    assert obj.verify(a,a,**kwargs)['status']=='same_frame_forbidden'
    kwargs['allowed_through']=99
    with pytest.raises(ValueError):obj.verify(a,b,**kwargs)


def test_crop_preserves_actual_provenance_and_rejects_changed_source_shape():
    image,audit=observation_crop(Reader(),actual(0)['evidence'])
    assert image.size==(24,24) and audit['crop_xyxy']==[8,8,32,32]
    assert audit['source_index']==0 and audit['input_rgb_sha256']
    ev=deepcopy(actual(0)['evidence']);ev['source_hw']=[101,100]
    with pytest.raises(ValueError):observation_crop(Reader(),ev)
