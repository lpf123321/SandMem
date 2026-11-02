import copy

import numpy as np
import pytest

from sandbox_vlm.memory.geometry import BBox3D
from sandbox_vlm.memory.visual import BoundedVisualBackend, VisualMatchSettings, VisualSandbox


def observation(fid, box=(1., 1., 10., 10.), vector=(1., 0.), **kwargs):
    return dict(video_id='video', name='wooden chair', description='wooden back',
        evidence=dict(frame_id=fid, bbox_xyxy=list(box), source_hw=[100, 100], sam_mask_nonempty=True),
        appearance=dict(vector=list(vector), dtype='float32', dim=len(vector)),
        settings=VisualMatchSettings(), allowed_through=fid, **kwargs)


def geometry(reference, x):
    return dict(coordinate_reference=reference,
                bbox_3d=BBox3D(np.array([x, 0, 0]), np.ones(3), np.eye(3), reference).to_dict(),
                reliability='local_estimate')


def test_geometry_missing_still_creates_and_reuses_visual_hypothesis():
    state = VisualSandbox('video')
    assert state.ingest(**observation(0))['status'] == 'created_visual_hypothesis'
    assert state.ingest(**observation(1))['status'] == 'matched_visual_hypothesis'
    assert len(state.instances) == 1 and state.instances[0]['local_geometry'] == []
    assert state.instances[0]['identity_status'] == 'visual_hypothesis'
    baseline = state.to_dict()
    assert state.ingest(**observation(0))['status'] == 'duplicate_actual_observation'
    assert state.to_dict() == baseline
    assert VisualSandbox.from_dict(baseline).to_dict() == baseline


def test_actual_covisible_objects_are_distinct_and_later_alias_is_ambiguous():
    state = VisualSandbox('video')
    state.ingest(**observation(0))
    second = state.ingest(**observation(0, box=(20., 20., 30., 30.)))
    assert second['visual_instance_id'] == 1
    result = state.ingest(**observation(1))
    assert result['status'] == 'ambiguous' and len(state.instances) == 2
    assert state.ambiguities[0]['candidate_visual_instance_ids'] == [0, 1]
    assert '1' not in state.viewpoints
    state.validate()


def test_cross_reference_positions_never_become_common_geometry():
    state = VisualSandbox('video')
    state.ingest(**observation(0, local_geometry=geometry('a', 0)))
    assert state.ingest(**observation(1, local_geometry=geometry('b', 100)))['status'] == 'matched_visual_hypothesis'
    assert [g['coordinate_reference'] for g in state.instances[0]['local_geometry']] == ['a', 'b']
    result = state.ingest(**observation(2, local_geometry=geometry('a', 100)))
    assert result['status'] == 'ambiguous'
    assert result['reason'] == 'common_reference_geometry_disagrees'
    state.validate()


def test_future_and_cross_video_rejected_and_full_backend_budget_preserved():
    state = VisualSandbox('video')
    value = observation(0)
    value['video_id'] = 'another'
    with pytest.raises(ValueError):
        state.ingest(**value)
    value = observation(1)
    value['allowed_through'] = 0
    with pytest.raises(ValueError):
        state.ingest(**value)
    state.ingest(**observation(0))
    backend = BoundedVisualBackend(1_000_000, device='cpu')
    first = backend.commit(state, auxiliary_state=dict(score_frame_ids=[0], scores=[1.]))
    assert first['status'] == 'ok'
    backend.limit_bytes = 2 * first['encoded_bytes'] + 8
    stored = backend.resident.clone()
    candidate = copy.deepcopy(state)
    candidate.ingest(**observation(1, vector=(0., 1.)))
    assert backend.commit(candidate)['status'] == 'capacity_limited'
    assert (stored == backend.resident).all()
