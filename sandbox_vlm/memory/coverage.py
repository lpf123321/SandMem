"""D043 visual hypotheses with actual short-range continuity checks.

No reliable common geometry is invented. Long-range appearance is retained as
possible correspondence, never sufficient to collapse two actual observations.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from sandbox_vlm.alignment.association import names_compatible
from sandbox_vlm.memory.sandbox import AppearanceFeature
from sandbox_vlm.memory.visual import VisualSandbox
from sandbox_vlm.perception.coverage import context_similarity, pack_descriptor, unpack_descriptor
from sandbox_vlm.perception.evidence import PromptEvidence


@dataclass
class ContinuitySettings:
    appearance_cosine_min: float = .90
    uniqueness_margin: float = .05
    name_policy: str = 'last_token'
    maximum_gap_frames: int = 16
    context_cosine_min: float = .95
    actual_box_iou_min: float = .30

    def __post_init__(self):
        if (not 0 <= self.appearance_cosine_min <= 1 or not 0 <= self.uniqueness_margin <= 2
                or self.name_policy not in ('exact', 'last_token') or self.maximum_gap_frames < 1
                or not 0 <= self.context_cosine_min <= 1 or not 0 <= self.actual_box_iou_min <= 1):
            raise ValueError('invalid bounded continuity settings')


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = float(np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod())
    union = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    return intersection / union if union > 0 else 0.


class CoverageSandbox(VisualSandbox):
    def __init__(self, video_id, question, *, retain_ambiguities=True, memory_limit_bytes=None):
        super().__init__(video_id)
        self.question_key = hashlib.sha256(question.encode()).hexdigest()
        self.retain_ambiguities = retain_ambiguities
        if memory_limit_bytes is not None:
            if retain_ambiguities:
                raise ValueError('bounded instance memory must disable pending observations')
            self.memory_limit_bytes = memory_limit_bytes
            self.next_visual_instance_id = 0
            from sandbox_vlm.memory.retention import fit_instance_memory
            _, _, report = fit_instance_memory(self, memory_limit_bytes)
            if report['status'] != 'ok':
                raise ValueError('memory limit cannot hold even the empty instance-memory structure')

    def to_dict(self):
        result = super().to_dict()
        result.update(representation='D043_coverage_visual_hypotheses', question_key=self.question_key)
        if not self.retain_ambiguities:
            result.pop('ambiguities')
            result['retain_ambiguities'] = False
        if hasattr(self, 'memory_limit_bytes'):
            result['memory_limit_bytes'] = self.memory_limit_bytes
        return result

    @classmethod
    def from_dict(cls, value):
        if value['representation'] != 'D043_coverage_visual_hypotheses':
            raise ValueError('cannot migrate frozen or strict identities implicitly')
        result = cls(value['video_id'], '', retain_ambiguities=value.get('retain_ambiguities', True))
        result.question_key = value['question_key']
        for name in ('instances', 'viewpoints', 'ambiguities', 'observation_index'):
            setattr(result, name, deepcopy(value.get(name, [] if name == 'ambiguities' else {})))
        if 'next_visual_instance_id' in value:
            result.next_visual_instance_id = value['next_visual_instance_id']
        if 'memory_limit_bytes' in value:
            result.memory_limit_bytes = value['memory_limit_bytes']
        result.validate()
        return result

    def validate(self):
        if not self.retain_ambiguities and self.ambiguities:
            raise ValueError('instance memory cannot retain ambiguity observations')
        super().validate()
        for instance in self.instances:
            unpack_descriptor(instance['context_descriptor'])
            fid = instance['context_frame_id']
            if not any(v['visual_instance_id'] == instance['visual_instance_id']
                       for v in self.viewpoints.get(str(fid), [])):
                raise ValueError('context lacks an actual observation link')
        return True

    def ingest(self, *, video_id, name, description, evidence, appearance, context_descriptor,
               settings, allowed_through, pair_verifier=None, local_geometry=None):
        if video_id != self.video_id or not name.strip():
            raise ValueError('cross-video or unnamed observation')
        actual = PromptEvidence.from_dict(evidence)
        if actual.frame_id > allowed_through:
            raise ValueError('future actual observation')
        feature = AppearanceFeature.from_dict(appearance)
        if local_geometry is not None:
            from sandbox_vlm.memory.geometry import BBox3D
            if BBox3D.from_dict(local_geometry['bbox_3d']).frame != local_geometry['coordinate_reference']:
                raise ValueError('geometry reference differs from its actual local box')
        packed = pack_descriptor(context_descriptor)
        key = hashlib.sha256(json.dumps(dict(frame_id=actual.frame_id,
            name=name.casefold().strip(), evidence=actual.to_dict()), sort_keys=True).encode()).hexdigest()
        if key in self.observation_index:
            return dict(status='duplicate_actual_observation', visual_instance_id=self.observation_index[key])
        occupied = {v['visual_instance_id'] for v in self.viewpoints.get(str(actual.frame_id), [])}
        scores = []
        for old in self.instances:
            if not names_compatible(name, old['name'], settings.name_policy):
                continue
            old_feature = AppearanceFeature.from_dict(old['appearance'])
            if old_feature.dim != feature.dim:
                continue
            gid = old['visual_instance_id']
            previous = next(v['evidence'] for v in self.viewpoints[str(old['context_frame_id'])]
                            if v['visual_instance_id'] == gid)
            same_size = previous['source_hw'] == evidence['source_hw']
            overlap = box_iou(previous['bbox_xyxy'], actual.bbox_xyxy) if same_size else 0.
            cosine = float(np.dot(old_feature.vector.astype(np.float64), feature.vector.astype(np.float64)))
            context = context_similarity(unpack_descriptor(old['context_descriptor']), context_descriptor)
            gap = actual.frame_id - old['context_frame_id']
            continuous = (0 < gap <= settings.maximum_gap_frames and context >= settings.context_cosine_min
                          and overlap >= settings.actual_box_iou_min)
            scores.append(dict(visual_instance_id=gid, cosine=cosine, context_cosine=context,
                actual_box_iou=overlap, frame_gap=gap, continuity_passed=continuous,
                already_observed_this_frame=gid in occupied))
        possible = sorted([s for s in scores if not s['already_observed_this_frame']
                           and s['cosine'] >= settings.appearance_cosine_min],
                          key=lambda s: (-s['cosine'], s['visual_instance_id']))
        eligible = [s for s in possible if s['continuity_passed']]
        unique = bool(eligible) and (len(eligible) == 1 or eligible[0]['cosine'] - eligible[1]['cosine'] >= settings.uniqueness_margin)
        verification = None
        pair_unique = bool(possible) and (len(possible) == 1 or
            possible[0]['cosine'] - possible[1]['cosine'] >= settings.uniqueness_margin)
        if not unique and pair_verifier is not None and pair_unique and not possible[0]['continuity_passed']:
            candidate = self.instance_by_id(possible[0]['visual_instance_id'])
            previous = next(v['evidence'] for v in self.viewpoints[str(candidate['context_frame_id'])]
                            if v['visual_instance_id'] == candidate['visual_instance_id'])
            verification = pair_verifier.verify(previous, actual.to_dict(),
                names=[candidate['name'], name], descriptions=[candidate['description'], description],
                candidate_id=candidate['visual_instance_id'], allowed_through=allowed_through)
            if verification['accepted']:
                eligible = [possible[0]]
                unique = True
        if unique:
            gid = eligible[0]['visual_instance_id']
            status = ('matched_pair_verified_visual_hypothesis' if verification and verification['accepted']
                      else 'matched_continuous_visual_hypothesis')
            instance = self.instance_by_id(gid)
        else:
            gid = self.allocate_instance_id()
            instance = dict(visual_instance_id=gid, identity_status='visual_hypothesis', name=name,
                description=description, appearance=feature.to_dict(), local_geometry=[])
            self.instances.append(instance)
            status = 'created_visual_hypothesis'
            if possible and self.retain_ambiguities:
                self.ambiguities.append(dict(observation_key=key, name=name, description=description,
                    evidence=actual.to_dict(), candidate_visual_instance_ids=[s['visual_instance_id'] for s in possible[:5]],
                    observed_visual_instance_id=gid,
                    reason='continuity_not_unique' if eligible else 'appearance_without_continuity'))
        instance.update(context_descriptor=packed, context_frame_id=actual.frame_id)
        if local_geometry is not None and local_geometry not in instance['local_geometry']:
            instance['local_geometry'].append(deepcopy(local_geometry))
        link = dict(visual_instance_id=gid, evidence=actual.to_dict(), identity_status='visual_hypothesis')
        if hasattr(self, 'next_visual_instance_id') or not self.retain_ambiguities:
            rows = self.viewpoints.get(str(actual.frame_id), [])
            link.update(observation_key=key,
                        observation_slot=max((r.get('observation_slot', j) for j, r in enumerate(rows)), default=-1)+1)
        self.viewpoints.setdefault(str(actual.frame_id), []).append(link)
        self.observation_index[key] = gid
        decision = dict(status=status, visual_instance_id=gid, possible_revisit=bool(possible and not unique),
                        scores=scores, **({'pair_verification': verification} if verification is not None else {}))
        if hasattr(self, 'memory_limit_bytes'):
            from sandbox_vlm.memory.retention import fit_instance_memory
            bounded, _, report = fit_instance_memory(self, self.memory_limit_bytes)
            if report['status'] != 'ok':
                raise RuntimeError('mandatory instance-memory state exceeds capacity')
            if report['compaction_applied']:
                self.__dict__.clear()
                self.__dict__.update(bounded.__dict__)
            decision.update(memory_retention=report,
                            retained=any(i['visual_instance_id'] == gid for i in self.instances))
        return decision


def compact_memory(state, question, displayed_frames, *, max_notes=16):
    if hashlib.sha256(question.encode()).hexdigest() != state.question_key:
        raise ValueError('question-conditioned memory cannot silently answer a different question')
    groups = defaultdict(list)
    for instance in state.instances:
        groups[instance['name']].append(instance['visual_instance_id'])
    # Every ID is included, regardless of its discovery name or a 64-row limit.
    # Detailed actual links stay resident; displayed links refer only to images
    # actually included in this answer. This is data, not a Count solver.
    shown = {str(fid): [v['visual_instance_id'] for v in state.viewpoints.get(str(fid), [])]
             for fid in displayed_frames}
    notes = [dict(observed_id=n['observed_visual_instance_id'],
                  possible_same_ids=n['candidate_visual_instance_ids'], frame_id=n['evidence']['frame_id'],
                  reason=n['reason']) for n in state.ambiguities[:max_notes]]
    return dict(name_to_visual_hypothesis_ids=dict(groups), displayed_frame_actual_ids=shown,
        possible_correspondences=notes, limitations=dict(omitted_hypotheses=0,
        omitted_possible_correspondences=max(0, len(state.ambiguities)-max_notes),
        hypothesis_ids_are_not_ground_truth_physical_counts=True,
        geometry_available=False, only_displayed_actual_links_included=True))


def select_answer_frames(state, ledger, *, maximum=32):
    """Cover actual linked hypotheses, then complement with visual/temporal views."""
    candidates = sorted(f for f in ledger if ledger[f]['margin'] > 0 or str(f) in state.viewpoints)
    if not candidates:
        candidates = sorted(ledger)
    if len(candidates) <= maximum:
        return candidates
    coverage = {fid: {v['visual_instance_id'] for v in state.viewpoints.get(str(fid), [])} for fid in candidates}
    descriptors = np.stack([unpack_descriptor(ledger[f]['descriptor']).reshape(-1) for f in candidates])
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12)
    max_similarity = np.full(len(candidates), -1., dtype=np.float32)
    position = {fid: i for i, fid in enumerate(candidates)}
    chosen, covered = [], set()
    while len(chosen) < maximum:
        available = [fid for fid in candidates if fid not in chosen]
        def priority(fid):
            diversity = 1.-float(max_similarity[position[fid]])
            return (len(coverage[fid]-covered), ledger[fid]['margin'] > 0, diversity, ledger[fid]['margin'], -fid)
        winner = max(available, key=priority)
        chosen.append(winner)
        covered.update(coverage[winner])
        max_similarity = np.maximum(max_similarity, descriptors @ descriptors[position[winner]])
    return sorted(chosen)
