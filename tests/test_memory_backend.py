from copy import deepcopy

import numpy as np

from sandbox_vlm.memory import AppearanceFeature, BBox3D, CameraParams, ObjectInstance, Sandbox, SpatialExtent, Viewpoint
from sandbox_vlm.memory.backend import BoundedSandboxBackend, encode_sandbox
from sandbox_vlm.memory.storage import cpu_storage_report


def scene():
    box = BBox3D([0, 0, 5], [1, 1, 1], np.eye(3), 'ref')
    camera = CameraParams(0, np.eye(4), np.eye(3), (10, 10), 'ref')
    obj = ObjectInstance(0, 'chair', '', box, AppearanceFeature([1, 0]))
    return Sandbox([SpatialExtent(0, 'ref', box, [obj], [Viewpoint(0, camera, [0])])], video_id='v')


def test_cpu_device_contract_preserves_complete_state_and_rejects_core_overflow():
    sb = scene()
    cap = max(len(encode_sandbox(sb)), cpu_storage_report(sb)['cpu_structural_bytes']) * 2
    backend = BoundedSandboxBackend(cap, device='cpu')  # Explicit CPU contract, not GPU measurement.
    assert backend.commit(sb)['status'] == 'ok'
    before = backend.read_working_copy().to_dict()
    oversized = deepcopy(sb)
    oversized.get_instance(0).short_description = 'x' * (cap * 2)
    report = backend.commit(oversized)
    assert report['status'] == 'capacity_limited' and report['gpu_structural_bytes_measured'] is None
    assert not report['paging_used'] and backend.read_working_copy().to_dict() == before


def test_compaction_preserves_first_link_complementary_views_and_pending_evidence():
    from dataclasses import replace
    from sandbox_vlm.memory.backend import compact_redundancy
    from sandbox_vlm.memory.sandbox import AmbiguityNote
    sb = scene()
    original = sb.all_viewpoints()[0]
    views = sb.spatial_extents[0].viewpoints
    views.extend([Viewpoint(1, replace(original.camera, frame_id=1), [0]),
                  Viewpoint(2, replace(original.camera, frame_id=2), [0])])
    pose = np.eye(4);pose[0, 3] = 1
    views.append(Viewpoint(3, replace(original.camera, frame_id=3, T_ref_from_camera=pose), [0]))
    sb.ambiguity_notes.append(AmbiguityNote('evidence', [0], [2], 'key evidence'))
    removed = compact_redundancy(sb)
    assert removed == [1]
    assert sb.frames_for_instance(0) == [0, 2, 3]


def test_required_overlap_context_is_charged_and_preserved_on_rejection():
    sb=scene()
    backend=BoundedSandboxBackend(16384,device='cpu')
    assert backend.commit(sb,auxiliary={'overlap':'retained'})['status']=='ok'
    before=backend.read_working_copy().to_dict()
    report=backend.commit(sb,auxiliary={'overlap':'x'*16384})
    assert report['status']=='capacity_limited' and report['auxiliary_included_in_complete_state']
    assert backend.read_auxiliary()=={'overlap':'retained'}
    assert backend.read_working_copy().to_dict()==before
