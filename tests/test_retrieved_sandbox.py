from copy import deepcopy

from sandbox_vlm.qa.retrieved_sandbox import retrieve


def test_retrieval_preserves_native_fields_and_independent_references():
    payload = dict(records=[[1, 'chair', 'complete description', [2, 9]], [2, 'table', 'other', [2]]],
        provided_frame_regions=[[1, 2, [0, 0, 5, 5], [10, 10]], [2, 2, [5, 5, 9, 9], [10, 10]]],
        plotted_ids=[1, 2], unplotted_ids=[], local_references=['ref1', 'ref2'],
        box_observation_sources=dict(rows=[[1, 2, 'a'], [2, 2, 'b']]),
        local_SAM_correspondences=dict(rows=[[1, 2, 'edge'], [1, 1, 'self']]))
    geometry = dict(original_D105=dict(sandbox=dict(spatial_extents=[
        dict(coordinate_reference='ref1', bounds={'native': 1}, instances=[
            dict(global_instance_id=1, bbox_3d={'native': 2}),
            dict(global_instance_id=2, bbox_3d={'native': 3})],
            viewpoints=[dict(camera={'native': 4}, observed_instance_ids=[1, 2])])])),
        added_panels=[dict(coordinate_reference='ref2', bounds={'native': 5}, camera={'native': 6},
            observations=[dict(instance_id=2, bbox_3d={'native': 7})])], added_ids=[2])
    before = deepcopy((payload, geometry))
    selected, filtered = retrieve(payload, geometry, dict(relevant_ids=[1]))
    assert (payload, geometry) == before
    assert selected['records'] == payload['records'][:1]
    assert selected['provided_frame_regions'] == payload['provided_frame_regions'][:1]
    assert selected['local_SAM_correspondences']['rows'] == [[1, 1, 'self']]
    assert selected['local_references'] == ['ref1', 'ref2']
    extent = filtered['original_D105']['sandbox']['spatial_extents'][0]
    original = geometry['original_D105']['sandbox']['spatial_extents'][0]
    assert extent['instances'] == original['instances'][:1]
    assert extent['bounds'] == original['bounds']
    assert extent['viewpoints'][0]['camera'] == original['viewpoints'][0]['camera']
    assert extent['viewpoints'][0]['observed_instance_ids'] == [1]
    assert filtered['added_panels'][0] == {**geometry['added_panels'][0], 'observations': []}
    assert filtered['added_ids'] == []
