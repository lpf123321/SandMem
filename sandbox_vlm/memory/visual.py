"""D041 isolated visual identity hypotheses; never fabricated shared 3D geometry.

This candidate does not migrate or mutate the strict Sandbox representation.
Only actual nonempty SAM prompts can create links. Appearance-only identity is
explicitly a hypothesis; geometry remains local to its named reference.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from sandbox_vlm.alignment.association import names_compatible
from sandbox_vlm.memory.sandbox import AppearanceFeature
from sandbox_vlm.memory.storage import cpu_storage_report
from sandbox_vlm.perception.evidence import PromptEvidence


@dataclass(frozen=True)
class VisualMatchSettings:
    appearance_cosine_min: float = 0.90
    uniqueness_margin: float = 0.05
    name_policy: str = 'last_token'
    common_reference_center_distance_diagonal_max: float = 0.25

    def __post_init__(self):
        if (not -1 <= self.appearance_cosine_min <= 1 or not 0 <= self.uniqueness_margin <= 2
                or self.name_policy not in ('exact', 'last_token')
                or not np.isfinite(self.common_reference_center_distance_diagonal_max)
                or self.common_reference_center_distance_diagonal_max < 0):
            raise ValueError('invalid visual hypothesis gates')


class VisualSandbox:
    def __init__(self, video_id):
        if not video_id:
            raise ValueError('explicit video ID required')
        self.video_id = video_id
        self.instances = []
        self.viewpoints = {}  # actual boxes + visual-hypothesis links, no invented camera
        self.ambiguities = []
        self.observation_index = {}  # necessary dedup state, included in serialization/budget

    def to_dict(self):
        result = dict(video_id=self.video_id, representation='D041_visual_identity_hypotheses',
            instances=deepcopy(self.instances), viewpoints=deepcopy(self.viewpoints),
            ambiguities=deepcopy(self.ambiguities), observation_index=dict(self.observation_index),
            geometry_notice='Every local box retains its own reference. Different references have no shared layout. Visual IDs are hypotheses, not geometrically confirmed physical truth.')
        if hasattr(self, 'next_visual_instance_id'):
            result['next_visual_instance_id'] = self.next_visual_instance_id
        return result

    @classmethod
    def from_dict(cls, value):
        if value['representation'] != 'D041_visual_identity_hypotheses':
            raise ValueError('strict or unknown memory cannot be silently migrated')
        state = cls(value['video_id'])
        for name in ('instances', 'viewpoints', 'ambiguities', 'observation_index'):
            setattr(state, name, deepcopy(value[name]))
        if 'next_visual_instance_id' in value:
            state.next_visual_instance_id = value['next_visual_instance_id']
        state.validate()
        return state

    def validate(self):
        ids = {x['visual_instance_id'] for x in self.instances}
        if len(ids) != len(self.instances):
            raise ValueError('duplicate visual instance IDs')
        if hasattr(self, 'next_visual_instance_id'):
            if (type(self.next_visual_instance_id) is not int or self.next_visual_instance_id < 0
                    or any(type(i) is not int or i < 0 or i >= self.next_visual_instance_id for i in ids)):
                raise ValueError('invalid monotonic visual instance ID allocator')
        elif ids != set(range(len(self.instances))):
            raise ValueError('visual instance IDs must be unique and contiguous')
        for instance in self.instances:
            if instance['identity_status'] != 'visual_hypothesis':
                raise ValueError('appearance cannot upgrade to confirmed physical identity')
            AppearanceFeature.from_dict(instance['appearance'])
            if any(g['bbox_3d']['frame'] != g['coordinate_reference'] for g in instance['local_geometry']):
                raise ValueError('local geometry reference mismatch')
        for frame_id, view in self.viewpoints.items():
            slots = [v.get('observation_slot', j) for j, v in enumerate(view)]
            if len(set(slots)) != len(slots) or any(type(s) is not int or s < 0 for s in slots):
                raise ValueError('invalid or duplicate actual observation slots')
            for link in view:
                evidence = PromptEvidence.from_dict(link['evidence'])
                if evidence.frame_id != int(frame_id) or link['visual_instance_id'] not in ids:
                    raise ValueError('dangling or unsupported actual viewpoint link')
                if 'observation_key' in link and self.observation_index.get(link['observation_key']) != link['visual_instance_id']:
                    raise ValueError('actual observation key/index mismatch')
        for note in self.ambiguities:
            PromptEvidence.from_dict(note['evidence'])
            if not set(note['candidate_visual_instance_ids']) <= ids:
                raise ValueError('dangling ambiguity candidate')
        if any(gid is not None and gid not in ids for gid in self.observation_index.values()):
            raise ValueError('dangling observation dedup index')
        return True

    def instance_by_id(self, gid):
        return next(i for i in self.instances if i['visual_instance_id'] == gid)

    def allocate_instance_id(self):
        if not hasattr(self, 'next_visual_instance_id'):
            return len(self.instances)
        gid = self.next_visual_instance_id
        self.next_visual_instance_id += 1
        return gid

    def ingest(self, *, video_id, name, description, evidence, appearance,
               settings, local_geometry=None, allowed_through):
        if video_id != self.video_id:
            raise ValueError('cross-video observation')
        actual = PromptEvidence.from_dict(evidence)
        if actual.frame_id > allowed_through or not name.strip():
            raise ValueError('future or unnamed observation')
        feature = AppearanceFeature.from_dict(appearance)
        if local_geometry is not None:
            from sandbox_vlm.memory.geometry import BBox3D
            box = BBox3D.from_dict(local_geometry['bbox_3d'])
            if box.frame != local_geometry['coordinate_reference']:
                raise ValueError('local geometry reference mismatch')
        # Exact actual observation identity, independent of processing clip ID.
        key_data = dict(frame_id=actual.frame_id, name=name.casefold().strip(), evidence=actual.to_dict())
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        if key in self.observation_index:
            return dict(status='duplicate_actual_observation', visual_instance_id=self.observation_index[key])
        occupied = {link['visual_instance_id'] for link in self.viewpoints.get(str(actual.frame_id), [])}
        scores = []
        for instance in self.instances:
            if not names_compatible(instance['name'], name, settings.name_policy):
                continue
            old = AppearanceFeature.from_dict(instance['appearance'])
            if old.dim != feature.dim:
                continue
            cosine = float(np.dot(old.vector.astype(np.float64), feature.vector.astype(np.float64)))
            geometric_status = 'no_common_reference'
            if local_geometry is not None:
                for old_geometry in instance['local_geometry']:
                    if old_geometry['coordinate_reference'] != local_geometry['coordinate_reference']:
                        continue
                    from sandbox_vlm.memory.geometry import BBox3D
                    a, b = BBox3D.from_dict(old_geometry['bbox_3d']), BBox3D.from_dict(local_geometry['bbox_3d'])
                    diagonal = max(float(np.linalg.norm(a.extents)), float(np.linalg.norm(b.extents)), 1e-12)
                    distance = float(np.linalg.norm(a.center - b.center)) / diagonal
                    geometric_status = ('common_reference_passed' if distance <= settings.common_reference_center_distance_diagonal_max
                                        else 'common_reference_disagrees')
                    if geometric_status == 'common_reference_disagrees':
                        break
            scores.append(dict(visual_instance_id=instance['visual_instance_id'], cosine=cosine,
                already_observed_this_frame=instance['visual_instance_id'] in occupied,
                geometry_status=geometric_status))
        candidates = sorted([s for s in scores if not s['already_observed_this_frame']
                             and s['cosine'] >= settings.appearance_cosine_min],
                            key=lambda s: (-s['cosine'], s['visual_instance_id']))
        if candidates:
            all_ranked = sorted([s for s in scores if not s['already_observed_this_frame']],
                                key=lambda s: (-s['cosine'], s['visual_instance_id']))
            unique = len(all_ranked) == 1 or all_ranked[0]['cosine'] - all_ranked[1]['cosine'] >= settings.uniqueness_margin
            if not unique or candidates[0]['geometry_status'] == 'common_reference_disagrees':
                note = dict(observation_key=key, name=name, description=description,
                    evidence=actual.to_dict(), appearance=feature.to_dict(), local_geometry=deepcopy(local_geometry),
                    candidate_visual_instance_ids=[s['visual_instance_id'] for s in all_ranked
                        if s['cosine'] >= min(settings.appearance_cosine_min,
                                             all_ranked[0]['cosine'] - settings.uniqueness_margin)],
                    reason='appearance_ambiguous' if not unique else 'common_reference_geometry_disagrees')
                self.ambiguities.append(note)
                # Failed/ambiguous observations are also deduplicated and charged.
                self.observation_index[key] = None
                return dict(status='ambiguous', visual_instance_id=None, reason=note['reason'], scores=scores)
            gid = candidates[0]['visual_instance_id']
            status = 'matched_visual_hypothesis'
            instance = self.instance_by_id(gid)
        else:
            gid = self.allocate_instance_id()
            instance = dict(visual_instance_id=gid, identity_status='visual_hypothesis', name=name,
                description=description, appearance=feature.to_dict(), local_geometry=[])
            self.instances.append(instance)
            status = 'created_visual_hypothesis'
        if local_geometry is not None and local_geometry not in instance['local_geometry']:
            instance['local_geometry'].append(deepcopy(local_geometry))
        self.viewpoints.setdefault(str(actual.frame_id), []).append(dict(
            visual_instance_id=gid, evidence=actual.to_dict(), identity_status='visual_hypothesis'))
        self.observation_index[key] = gid
        return dict(status=status, visual_instance_id=gid, scores=scores)

    def encode(self):
        self.validate()
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


class BoundedVisualBackend:
    """Complete serialized GPU state and an explicitly measured host working copy."""
    def __init__(self, limit_bytes, *, device='cuda', overflow_policy=None):
        if overflow_policy not in (None, 'reject', 'first_last_then_oldest_observed'):
            raise ValueError('unknown visual-memory overflow policy')
        self.limit_bytes, self.device = limit_bytes, device
        self.overflow_policy = overflow_policy
        self.resident = None
        self.peak_bytes = 0

    def commit(self, sandbox, *, auxiliary_state=None):
        import torch
        sandbox.validate()
        # The current instances-only path gets the approved retention policy.
        # Historical ambiguity-bearing methods retain their original rejection
        # behavior; frozen runtimes are separate files and remain unchanged.
        policy = self.overflow_policy
        if policy is None:
            policy = ('first_last_then_oldest_observed'
                      if getattr(sandbox, 'retain_ambiguities', True) is False else 'reject')
        if policy != 'reject':
            return self._commit_retained(sandbox, auxiliary_state)
        encoded = json.dumps(dict(visual_sandbox=sandbox.to_dict(), auxiliary_state=auxiliary_state),
                             ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        total = len(encoded)
        # Transactional replacement briefly holds both blobs; that is actual
        # structural GPU storage and must also fit, not hidden workspace.
        replacement_peak = total + (self.resident.numel() if self.resident is not None else 0)
        if replacement_peak > self.limit_bytes:
            return dict(status='capacity_limited', encoded_bytes=len(encoded), complete_bytes=total,
                        replacement_peak_bytes=replacement_peak,
                        previous_state_preserved=True, limit_bytes=self.limit_bytes)
        replacement = torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8).copy()).to(self.device)
        self.resident = replacement
        self.peak_bytes = max(self.peak_bytes, replacement_peak)
        return dict(status='ok', encoded_bytes=len(encoded), complete_bytes=total,
            limit_bytes=self.limit_bytes, peak_complete_bytes=self.peak_bytes,
            replacement_peak_bytes=replacement_peak,
            cpu_working_state=cpu_storage_report(dict(sandbox=sandbox, auxiliary_state=auxiliary_state)),
            complete_state_resident=True, paging=False)

    def _commit_retained(self, sandbox, auxiliary_state):
        import torch
        from sandbox_vlm.memory.retention import fit_instance_memory
        reserve = self.resident.untyped_storage().nbytes() if self.resident is not None else 0
        working, auxiliary, report = fit_instance_memory(sandbox, self.limit_bytes,
            auxiliary_state=auxiliary_state, encoded_reserve_bytes=reserve)
        row = dict(report, limit_bytes=self.limit_bytes, previous_state_preserved=True,
                   complete_state_resident=self.resident is not None, paging=False, retention_applied=False)
        if report['status'] != 'ok':
            return row
        encoded = json.dumps(dict(visual_sandbox=working.to_dict(), auxiliary_state=auxiliary),
                             ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        replacement = torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8).copy()).to(self.device)
        self.resident = replacement
        self.peak_bytes = max(self.peak_bytes, report['after']['replacement_peak_bytes'])
        if report['compaction_applied']:
            # Publishing the bounded copy is essential: QA and the next ingest
            # must not continue using an unbounded caller-side original.
            sandbox.__dict__.clear()
            sandbox.__dict__.update(working.__dict__)
            if isinstance(auxiliary_state, dict):
                auxiliary_state.clear()
                auxiliary_state.update(auxiliary)
        row.update(encoded_bytes=len(encoded), complete_bytes=len(encoded),
            replacement_peak_bytes=report['after']['replacement_peak_bytes'],
            peak_complete_bytes=self.peak_bytes, complete_state_resident=True,
            previous_state_preserved=False,
            retention_applied=report['compaction_applied'],
            cpu_working_state=cpu_storage_report(dict(sandbox=sandbox, auxiliary_state=auxiliary_state)))
        return row

    def read_working_copy(self):
        if self.resident is None:
            raise ValueError('no committed visual memory')
        value = json.loads(self.resident.cpu().numpy().tobytes())['visual_sandbox']
        if value['representation'] == 'D043_coverage_visual_hypotheses':
            from sandbox_vlm.memory.coverage import CoverageSandbox
            return CoverageSandbox.from_dict(value)
        return VisualSandbox.from_dict(value)
