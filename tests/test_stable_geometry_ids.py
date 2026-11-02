from copy import deepcopy
import pytest
from sandbox_vlm.qa.stable_ids import geometry_display_ids, remap_payload


def state(assignments):
    return dict(instances=[dict(visual_instance_id=i)for i in sorted(set(assignments))],
        viewpoints={str(n):[dict(visual_instance_id=i,evidence=dict(frame_id=n,bbox_xyxy=[0,0,10,10]))]
                    for n,i in enumerate(assignments)})


def test_merge_preserves_unrelated_SAM_ids_and_does_not_mutate_state():
    sam, revised = state([0,1,2]), state([0,0,1])
    before = deepcopy((sam,revised))
    assert geometry_display_ids(sam,revised) == {0:0,1:2}
    assert (sam,revised) == before


def test_full_payload_renames_records_and_ambiguities_only():
    p=dict(hypotheses=[dict(visual_id=0,name='a',actual_frame_ids=[0,1]),
                       dict(visual_id=1,name='b',actual_frame_ids=[2])],record_fields=None,
           ambiguities=[dict(observed_visual_id=1,candidate_visual_ids=[0],reason='uncertain')],
           selection_limitations=dict(omitted_relevant_hypotheses=0))
    original=deepcopy(p); got=remap_payload(p,{0:0,1:2})
    assert got['hypotheses'][1] == dict(visual_id=2,name='b',actual_frame_ids=[2])
    assert got['ambiguities'][0]['observed_visual_id'] == 2
    assert got['selection_limitations']==p['selection_limitations'] and p==original


def test_compact_payload_preserves_order_and_content():
    p=dict(hypotheses=[[0,'a','description',[0,1]],[1,'b','description',[2]]],
           record_fields=['visual_id','name','short_description','actual_frame_ids'],ambiguities=[])
    assert remap_payload(p,{0:0,1:2})['hypotheses'] == [[0,'a','description',[0,1]],[2,'b','description',[2]]]


def test_no_geometry_change_is_identity():
    s=state([0,1,2]);assert geometry_display_ids(s,s)=={0:0,1:1,2:2}
    p=dict(hypotheses=dict(visual_id_is_row_index=True))
    assert remap_payload(p,{0:0})==p


def test_changed_observations_or_split_components_are_rejected():
    with pytest.raises(ValueError): geometry_display_ids(state([0,0,1]),state([0,1,2]))
    with pytest.raises(ValueError): geometry_display_ids(state([0,1]),state([0]))


def test_noncontiguous_implicit_row_ids_are_rejected():
    with pytest.raises(ValueError):remap_payload(dict(hypotheses=dict(rows=[])),{0:0,1:2})
