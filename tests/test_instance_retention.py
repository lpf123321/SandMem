from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from sandbox_vlm.memory.coverage import CoverageSandbox, ContinuitySettings
from sandbox_vlm.memory.visual import BoundedVisualBackend
from sandbox_vlm.memory.retention import fit_instance_memory, observation_link
from sandbox_vlm.memory.joint_geometry import observations
from sandbox_vlm.qa.complete_sandbox import complete_records, local_correspondences
from sandbox_vlm.qa.evidence_binding import linked_selection


def add(state, fid, name='chair', description='actual object', box=(1., 1., 10., 10.)):
    return state.ingest(video_id='v', name=name, description=description,
        evidence=dict(frame_id=fid, bbox_xyxy=list(box), source_hw=[40, 40], sam_mask_nonempty=True),
        appearance=dict(vector=[1., 0.], dim=2, dtype='float32'),
        context_descriptor=np.ones((16, 64), dtype=np.float32),
        settings=ContinuitySettings(), allowed_through=fid)


def history(n=30):
    state=CoverageSandbox('v', 'chair', retain_ambiguities=False)
    for fid in range(n):
        add(state, fid)
    return state


def frames(state, gid):
    return sorted(int(f) for f, rows in state.viewpoints.items()
                  if any(r['visual_instance_id']==gid for r in rows))


def test_below_limit_is_exactly_unchanged_and_never_discards_middle_views():
    state=history(); before=state.to_dict()
    kept,aux,report=fit_instance_memory(state,1_000_000)
    assert report['status']=='ok' and not report['compaction_applied']
    assert aux is None and kept.to_dict()==before==state.to_dict()
    assert frames(kept,0)==list(range(30))


def test_overflow_compacts_to_actual_temporal_endpoints_before_any_eviction():
    state=history(100);before=state.to_dict()
    # Deliberately reverse dict insertion order: frame IDs define recency.
    state.viewpoints=dict(reversed(list(state.viewpoints.items())))
    kept,_,report=fit_instance_memory(state,20_000)
    assert report['status']=='ok' and report['compaction_applied']
    assert report['evicted_instance_ids']==[] and report['removed_observation_links']==98
    assert frames(kept,0)==[0,99] and len(kept.observation_index)==2
    assert state.instances==before['instances'] and len(state.viewpoints)==100
    assert max(report['after'].values())<=20_000
    for f in (0,99):
        old=next(r for r in before['viewpoints'][str(f)] if r['visual_instance_id']==0)
        assert kept.viewpoints[str(f)][0]['evidence']==old['evidence']


def test_single_observation_is_not_duplicated_when_budget_pressure_evicts_it():
    state=CoverageSandbox('v','chair',retain_ambiguities=False)
    add(state,4,description='x'*100_000)
    kept,_,report=fit_instance_memory(state,15_000)
    assert report['removed_observation_links']==0
    assert report['evicted_instance_ids']==[0]
    assert report['status']=='ok' and not kept.instances and not kept.viewpoints and not kept.observation_index


def two_objects():
    state=CoverageSandbox('v','table',retain_ambiguities=False)
    for fid in [0,1,2]:add(state,fid,'chair',description='x'*60_000)
    for fid in [0,3,7]:add(state,fid,'table',box=(20.,20.,30.,30.))
    return state


def test_eviction_keeps_ids_region_slots_and_allocator_stable_across_reload():
    state=two_objects();kept,_,report=fit_instance_memory(state,20_000)
    assert report['evicted_instance_ids']==[0] and report['eviction_last_observed_frames']==[2]
    assert [i['visual_instance_id']for i in kept.instances]==[1] and frames(kept,1)==[0,7]
    assert observation_link(kept,0,1)['visual_instance_id']==1
    assert {o['observation_id']for o in observations(kept)}=={'F0_O1','F7_O0'}
    restored=CoverageSandbox.from_dict(json.loads(json.dumps(kept.to_dict())))
    assert restored.to_dict()==kept.to_dict()
    # A retained actual observation remains deduplicated after restart.
    assert add(restored,7,'table',box=(20.,20.,30.,30.))['status']=='duplicate_actual_observation'
    assert add(restored,20,'lamp')['visual_instance_id']==2
    payload=complete_records(restored.to_dict(),[0,7,20],20)
    assert [r[0]for r in payload['records']]==[1,2]
    assert linked_selection('table',payload)['anchor_ids']==[1]
    assert 0 not in restored.observation_index.values()


def test_ties_use_id_and_never_qa_access_or_list_order():
    state=CoverageSandbox('v','objects',retain_ambiguities=False)
    for name in ['chair','table','lamp']:add(state,9,name,description='x'*12_000)
    state.instances.reverse()
    kept,_,report=fit_instance_memory(state,25_000)
    assert report['evicted_instance_ids'] and report['evicted_instance_ids']==list(range(len(report['evicted_instance_ids'])))
    assert report['eviction_last_observed_frames']==[9]*len(report['evicted_instance_ids'])
    kept.validate()


def test_auxiliary_geometry_masks_and_shared_views_are_cleaned_but_ledger_charged():
    state=two_objects()
    aux=dict(geometry=dict(spatial_extents=[dict(instances=[dict(global_instance_id=0),dict(global_instance_id=1)],
        viewpoints=[dict(frame_id=0,observed_instance_ids=[0,1]),dict(frame_id=3,observed_instance_ids=[1]),
                    dict(frame_id=7,observed_instance_ids=[1])])]),
        mask_catalog={k:dict(path=k)for k in ['F0_O0','F0_O1','F3_O0','F7_O0']},
        accepted_edges=[dict(source='F0_O0',target='F7_O0'),dict(source='F0_O1',target='F7_O0')],
        gated_frame_ledger={'0':{'descriptor':'retained independent coverage'}})
    original=deepcopy(aux)
    kept,clean,report=fit_instance_memory(state,25_000,auxiliary_state=aux)
    assert report['status']=='ok' and report['evicted_instance_ids']==[0]
    assert clean['geometry']['spatial_extents'][0]['instances']==[dict(global_instance_id=1)]
    assert clean['geometry']['spatial_extents'][0]['viewpoints']==[
        dict(frame_id=0,observed_instance_ids=[1]),dict(frame_id=7,observed_instance_ids=[1])]
    assert set(clean['mask_catalog'])=={'F0_O1','F7_O0'}
    assert clean['accepted_edges']==[dict(source='F0_O1',target='F7_O0')]
    assert clean['gated_frame_ledger']==aux['gated_frame_ledger'] and aux==original


def test_removed_sam_endpoints_are_not_rebound_to_the_next_region():
    kept,_,_=fit_instance_memory(two_objects(),20_000)
    def edge(source,target):
        return dict(would_reuse=True,reason='unique_continuous_mask_correspondence',
            source=source,target=target,source_frame=0,target_frame=7,
            continuous_nonempty=True,endpoint_prompt_nonempty=True,mask_iou=.9)
    links,excluded=local_correspondences(kept.to_dict(),[edge('F0_O0','F7_O0'),edge('F0_O1','F7_O0')],{0,7},7)
    assert links['rows']==[[1,1,0,7,.9]]
    assert excluded=={'endpoint_removed_by_memory_retention':1}


def test_backend_publishes_pruned_state_to_caller_and_preserves_previous_on_rejection():
    state=two_objects();aux=dict(mask_catalog={'F0_O0':{},'F0_O1':{},'F7_O0':{}})
    backend=BoundedVisualBackend(25_000,device='cpu')
    report=backend.commit(state,auxiliary_state=aux)
    assert report['status']=='ok' and report['evicted_instance_ids']==[0]
    assert backend.read_working_copy().to_dict()==state.to_dict()
    assert set(aux['mask_catalog'])=={'F0_O1','F7_O0'}
    original=state.to_dict();blob=backend.resident.clone()
    report=backend.commit(state,auxiliary_state={'mandatory_context':'x'*100_000})
    assert report['status']=='capacity_limited' and not report['paging']
    assert state.to_dict()==original and (blob==backend.resident).all()


def test_live_ingest_enforces_the_cap_and_checkpoint_keeps_configuration():
    state=CoverageSandbox('v','chair',retain_ambiguities=False,memory_limit_bytes=20_000)
    triggered=False
    for fid in range(80):
        decision=add(state,fid)
        report=decision['memory_retention'];triggered|=report['compaction_applied']
        assert report['status']=='ok' and max(report['after'].values())<=20_000
    assert triggered and max(frames(state,0))==79 and min(frames(state,0))==0
    restored=CoverageSandbox.from_dict(state.to_dict())
    assert restored.memory_limit_bytes==20_000
    decision=add(restored,80,'oversized',description='x'*100_000)
    assert decision['memory_retention']['status']=='ok'
    assert decision['retained'] is False
    assert all(i['name']!='oversized'for i in restored.instances)


def test_unmanaged_identity_cache_is_rejected_and_tiny_or_pending_policy_is_explicit():
    with pytest.raises(ValueError,match='unmanaged'):
        fit_instance_memory(two_objects(),20_000,auxiliary_state={'unmanaged':{'visual_instance_id':0}})
    with pytest.raises(ValueError,match='pending'):
        fit_instance_memory(CoverageSandbox('v','q'),20_000)
    with pytest.raises(ValueError,match='empty'):
        CoverageSandbox('v','q',retain_ambiguities=False,memory_limit_bytes=1)


def test_alias_observation_keys_survive_compaction_and_restart():
    state=CoverageSandbox('v','chair',retain_ambiguities=False,memory_limit_bytes=20_000)
    for fid in range(90):
        add(state,fid,'chair' if fid%2==0 else 'wooden chair')
    # Force compaction independently of the current allocation boundary.
    kept,_,report=fit_instance_memory(state,12_000)
    assert report['status']=='ok' and kept.instances
    restored=CoverageSandbox.from_dict(kept.to_dict())
    assert add(restored,89,'wooden chair')['status']=='duplicate_actual_observation'
    assert len(restored.observation_index)==sum(map(len,restored.viewpoints.values()))


def test_d116_compose_cannot_restore_evicted_objects_or_intermediate_observations():
    from sandbox_vlm.memory import BBox3D,CameraParams,ObjectInstance,AppearanceFeature,Viewpoint,Sandbox,SpatialExtent
    from sandbox_vlm.qa.d116_full import compose
    source=two_objects();kept,_,_=fit_instance_memory(source,20_000)
    box=BBox3D([0,0,3],[1,1,1],np.eye(3),'ref')
    camera=lambda f,ref='ref':CameraParams(f,np.eye(4),np.eye(3),(40,40),ref)
    objects=[ObjectInstance(i['visual_instance_id'],i['name'],i['description'],box,AppearanceFeature([1,0]))
             for i in source.instances]
    scene=Sandbox([SpatialExtent(0,'ref',box,objects,[Viewpoint(0,camera(0),[0,1]),Viewpoint(7,camera(7),[1])])],video_id='v')
    frozen=dict(sandbox=scene.to_dict(),audit=[
        dict(frame_id=0,instance_id=0,observation_id='F0_O0',bbox_3d=box.to_dict()),
        dict(frame_id=0,instance_id=1,observation_id='F0_O1',bbox_3d=None)])
    newbox=BBox3D([0,0,3],[1,1,1],np.eye(3),'new').to_dict()
    packets=[dict(source_frame=7,coordinate_reference='new',bounds=newbox,camera=camera(7,'new').to_dict(),
        observations=[dict(instance_id=1,frame_id=7,observation_id='F7_O0',bbox_3d=newbox),
                      dict(instance_id=0,frame_id=7,observation_id='stale',bbox_3d=newbox),
                      dict(instance_id=1,frame_id=3,observation_id='F3_O0',bbox_3d=newbox)])]
    payload,geometry,_=compose(kept.to_dict(),frozen,packets,[],[0,7],7,'table')
    assert [r[0]for r in payload['records']]==[1] and payload['records'][0][3]==[0,7]
    assert payload['plotted_ids']==[1] and geometry['added_ids']==[1]
    assert geometry['original_D105']['sandbox']['spatial_extents'][0]['instances']==[]
    assert [o['observation_id']for panel in geometry['added_panels']for o in panel['observations']]==['F7_O0']


def test_compact_qa_serialization_preserves_noncontiguous_ids():
    from sandbox_vlm.qa.coverage_evidence import append_evidence
    from sandbox_vlm.qa.complete_sandbox import assert_original_records_preserved
    kept,_,_=fit_instance_memory(two_objects(),20_000)
    class Tokenizer:
        def encode(self,text):return list(range(len(text)))
    messages=[dict(role='user',content=[dict(type='text',text='table')])]
    payload,audit=append_evidence(messages,kept,'table',Tokenizer(),max_text_tokens=1,
                                 reserved_output_tokens=0,images=0)
    assert audit['format']=='columnar_records'
    assert payload['hypotheses']['visual_id_is_row_index'] is False
    assert payload['hypotheses']['visual_id_table']==[1]
    assert_original_records_preserved(payload,complete_records(kept.to_dict(),[0,7],7))


@pytest.mark.parametrize('module',['sandbox_vlm.memory.short_track','sandbox_vlm.memory.joint_geometry'])
def test_correspondence_helpers_do_not_renumber_surviving_ids(module):
    from importlib import import_module
    implementation=import_module(module)
    kept,_,_=fit_instance_memory(two_objects(),20_000)
    result,audit=implementation.merge_correspondences(kept,implementation.observations(kept),[],watermark=7)
    assert [i['visual_instance_id']for i in result.instances]==[1]
    assert result.next_visual_instance_id==2 and audit['old_to_new']=={1:1}

