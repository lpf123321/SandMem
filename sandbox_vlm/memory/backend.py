"""Bounded complete GPU snapshot, with explicit host working-state accounting.

The complete compact state stays resident. No CPU/disk eviction or paging exists.
A uint8 JSON snapshot is deliberately simple; serialization overhead counts too.
"""
from copy import deepcopy
import json

import numpy as np

from sandbox_vlm.memory.sandbox import Sandbox
from sandbox_vlm.memory.storage import cpu_storage_report

_UNCHANGED = object()


def encode_sandbox(sandbox):
    if sandbox.validate():
        raise ValueError('invalid sandbox cannot enter the backend')
    return json.dumps(sandbox.to_dict(), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def compact_redundancy(sandbox):
    """Drop only exact same-pose/grid/link duplicates, preserving first evidence.

    Actual complementary views and ambiguity evidence cannot be removed by this
    conservative first pass. Features are normalized to the approved float32
    storage, with exact duplicates sharing objects. No lower precision is assumed.
    """
    notes_frames = {fid for n in sandbox.ambiguity_notes for fid in n.frame_ids}
    removed = []
    for extent in sandbox.spatial_extents:
        first = {gid: min(sandbox.frames_for_instance(gid)) for gid in extent.instance_ids()
                 if sandbox.frames_for_instance(gid)}
        retained = []
        for view in sorted(extent.viewpoints, key=lambda v: v.frame_id):
            protected = view.frame_id in notes_frames or view.frame_id in first.values()
            duplicate = any(view.observed_instance_ids == old.observed_instance_ids
                and view.camera.image_size == old.camera.image_size
                and np.array_equal(view.camera.intrinsic, old.camera.intrinsic)
                and np.array_equal(view.camera.T_ref_from_camera, old.camera.T_ref_from_camera)
                for old in retained)
            if duplicate and not protected:
                removed.append(view.frame_id)
            else:
                retained.append(view)
        extent.viewpoints = retained
    # Share exact representative vectors; all instances and pending cores remain.
    shared = {}
    for item in [*sandbox.all_instances(), *sandbox.pending_observations]:
        feature = item.appearance
        if feature is not None:
            key = (feature.dtype, feature.dim, feature.vector.tobytes())
            item.appearance = shared.setdefault(key, feature)
    return removed


class BoundedSandboxBackend:
    def __init__(self, max_bytes, *, device='cuda'):
        import torch
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('explicit positive backend byte limit required')
        self.max_bytes, self.device = max_bytes, torch.device(device)
        self._buffer = None
        self.peak_resident_bytes = 0
        self.last_report = None

    def commit(self, sandbox, *, auxiliary=_UNCHANGED):
        import torch
        if auxiliary is _UNCHANGED:
            auxiliary = self.read_auxiliary() if self._buffer is not None else None
        working = deepcopy(sandbox)
        def serialize():
            core = json.loads(encode_sandbox(working))
            return json.dumps(dict(sandbox=core, auxiliary=auxiliary), ensure_ascii=False,
                              separators=(',', ':'), allow_nan=False).encode('utf-8')
        payload = serialize()
        host = cpu_storage_report([working, auxiliary])['cpu_structural_bytes']
        removed = []
        if len(payload) > self.max_bytes or host > self.max_bytes:
            removed = compact_redundancy(working)
            payload = serialize()
            host = cpu_storage_report([working, auxiliary])['cpu_structural_bytes']
        accepted = len(payload) <= self.max_bytes and host <= self.max_bytes
        row = dict(status='ok' if accepted else 'capacity_limited', limit_bytes=self.max_bytes,
                   encoded_state_bytes=len(payload), cpu_working_state_bytes=host,
                   removed_redundant_viewpoint_frame_ids=removed,
                   complete_state_resident=accepted or self._buffer is not None, paging_used=False,
                   device=str(self.device), gpu_structural_bytes_measured=None,
                   previous_state_preserved_on_rejection=True)
        row['auxiliary_included_in_complete_state'] = auxiliary is not None
        if accepted:
            array = np.frombuffer(payload, dtype=np.uint8).copy()
            new_buffer = torch.from_numpy(array).to(self.device)
            self._buffer = new_buffer
            allocated = new_buffer.untyped_storage().nbytes()
            self.peak_resident_bytes = max(self.peak_resident_bytes, allocated)
            row.update(resident_tensor_bytes=allocated,
                       gpu_structural_bytes_measured=allocated if self.device.type == 'cuda' else None)
        else:
            row['resident_tensor_bytes'] = self._buffer.untyped_storage().nbytes() if self._buffer is not None else 0
        row['peak_resident_tensor_bytes'] = self.peak_resident_bytes
        self.last_report = row
        return row

    def read_working_copy(self):
        if self._buffer is None:
            raise ValueError('no committed sandbox')
        # The complete GPU snapshot remains allocated throughout the bounded read.
        payload = self._buffer.cpu().numpy().tobytes()
        return Sandbox.from_dict(json.loads(payload)['sandbox'])

    def read_auxiliary(self):
        if self._buffer is None:
            return None
        return json.loads(self._buffer.cpu().numpy().tobytes())['auxiliary']
