"""Continuous SandMem candidate: perception -> bounded memory -> rendered QA.

Frozen D116 remains a separate reference. No pending identities, cross-frame
2D IoU gate, geometric identity merges, or full point-cloud paging is used.
"""
from copy import deepcopy
from dataclasses import replace
import json
import re
from collections.abc import Iterator, Mapping

import numpy as np

from sandbox_vlm.memory.continuous_scenes import (
    ContinuousSceneMemory, SceneObservation, VoxelScenePolicy)
from sandbox_vlm.memory.geometry import BBox3D
from sandbox_vlm.memory.storage import cpu_storage_report
from sandbox_vlm.perception.scene_points import reliable_scene_points, scale_fixed_voxel_size
from sandbox_vlm.qa.continuous_selection import select_continuous_frames
from sandbox_vlm.qa.selection import terms
from sandbox_vlm.qa.evidence_binding import PROTOCOL_WORDS


def caption_key(name):
    return ' '.join(name.casefold().split())


def unit_feature(value):
    a = np.asarray(value, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(a)
    if not len(a) or not np.isfinite(a).all() or n <= 0:
        raise ValueError('nonempty finite appearance evidence required')
    return a / n


class ContinuousSandMem:
    """All persistent candidate state is reachable here and charged together.

    Geometry and segmentation adapters provide one chronological sampled frame
    at a time. Segmentation features are frozen; detections are open vocabulary.
    Questions only affect QA retrieval, never the geometry input schedule.
    """

    def __init__(self, video_id, config):
        if not video_id:
            raise ValueError('video identity required')
        self.video_id, self.config = video_id, deepcopy(config)
        self.instances, self.observations, self.frames = {}, {}, {}
        self.next_id = 0
        self.scenes = None
        self.reference = None
        self.last_frame_id = None
        self.capacity_events = []
        self.failed = False

    def _assign(self, detections):
        """Joint one-to-one assignment within caption; dummy columns allow births."""
        from scipy.optimize import linear_sum_assignment
        identities = sorted(self.instances)
        threshold = self.config['identity']['appearance_cosine_min']
        matrix = np.full((len(detections), len(identities) + len(detections)), -1e6)
        matrix[:, len(identities):] = 0
        vectors = [unit_feature(d['appearance']) for d in detections]
        for i, detection in enumerate(detections):
            for j, gid in enumerate(identities):
                old = self.instances[gid]
                if caption_key(old['name']) != caption_key(detection['name']):
                    continue
                feature = np.asarray(old['appearance'])
                if feature.shape != vectors[i].shape:
                    raise ValueError('appearance encoder changed inside one video')
                score = float(np.dot(vectors[i], feature))
                if score >= threshold:
                    matrix[i, j] = score
        assignment = {}
        if detections:
            rows, columns = linear_sum_assignment(matrix, maximize=True)
            for i, j in zip(rows, columns):
                assignment[int(i)] = identities[j] if j < len(identities) and matrix[i, j] >= threshold else None
        return assignment, vectors

    def ingest(self, *, camera, descriptor, world_points, confidence,
               detections, available_through):
        if self.failed:
            raise RuntimeError('memory session failed; cannot resume from partial state')
        fid = camera.frame_id
        if fid > available_through or (self.last_frame_id is not None and fid <= self.last_frame_id):
            raise ValueError('chronological causal sampled frames required')
        if self.reference is not None and camera.reference != self.reference:
            raise ValueError('explicit begin_reference required for independent reconstruction')
        self.reference = camera.reference
        descriptor = unit_feature(descriptor)
        actual = []
        for i, detection in enumerate(detections):
            from sandbox_vlm.perception.evidence import PromptEvidence
            evidence = PromptEvidence.from_dict(detection['evidence'])
            if evidence.frame_id != fid or not detection['name'].strip():
                raise ValueError('unbacked detection or incorrect source frame')
            if detection.get('bbox_3d') is not None:
                if BBox3D.from_dict(detection['bbox_3d']).frame != camera.reference:
                    raise ValueError('object box is outside current geometry reference')
            actual.append(detection)
        # Nothing in matching reads spatial boxes, 2D overlap, scene ID or answers.
        assignment, features = self._assign(actual)
        scene_points, point_report = reliable_scene_points(world_points, confidence, **self.config['points'])
        if self.scenes is None and len(scene_points):
            try:
                size = scale_fixed_voxel_size(scene_points,
                    diagonal_fraction=self.config['scenes']['voxel_size_initial_diagonal_fraction'])
            except ValueError:
                scene_points = np.empty((0, 3))
            else:
                self.scenes = ContinuousSceneMemory(reference=camera.reference, policy=VoxelScenePolicy(
                    size, self.config['scenes']['overlap_threshold'], self.config['scenes']['expansion_factor']))
        links = []
        for i, detection in enumerate(actual):
            gid = assignment[i]
            if gid is None:
                gid = self.next_id
                self.next_id += 1
                self.instances[gid] = dict(id=gid, name=detection['name'],
                    description=detection.get('description', ''), appearance=features[i], boxes={})
            obj = self.instances[gid]
            obj['appearance'] = features[i]
            oid = f'F{fid}_O{i}'
            self.observations[oid] = dict(id=oid, instance_id=gid, frame_id=fid,
                                          evidence=deepcopy(detection['evidence']))
            if detection.get('bbox_3d') is not None and camera.reference not in obj['boxes']:
                obj['boxes'][camera.reference] = dict(bbox=deepcopy(detection['bbox_3d']),
                    source_observation_id=oid, source_frame_id=fid,
                    semantics='static_visible_surface_proxy_at_source_frame')
            links.append(SceneObservation(oid, gid, fid))
        if self.scenes is not None:
            grouped = self.scenes.add(camera=camera, reliable_points=scene_points,
                occupied_voxels=self.scenes.policy.voxelize(scene_points),
                observations=tuple(links), available_through=available_through)
        else:
            grouped = dict(status='no_reliable_geometry', scene_id=None)
        self.frames[fid] = dict(camera=camera, descriptor=descriptor,
                               observation_ids=[link.observation_id for link in links],
                               scene_id=grouped['scene_id'])
        self.last_frame_id = fid
        try:
            capacity = self.enforce_capacity()
        except MemoryError:
            self.failed = True
            raise
        return dict(frame_id=fid, grouping=grouped, points=point_report,
                    assigned_instance_ids=[link.instance_id for link in links], capacity=capacity)

    def begin_reference(self, reference):
        if not reference or reference == self.reference:
            raise ValueError('fresh reference name required')
        raise ValueError('continuous session accepts one upstream reference; independent reconstruction needs an explicit new session')

    def _prune(self, keep):
        self.observations = {oid: row for oid, row in self.observations.items() if oid in keep}
        surviving = {row['instance_id'] for row in self.observations.values()}
        self.instances = {gid: obj for gid, obj in self.instances.items() if gid in surviving}
        for obj in self.instances.values():
            obj['boxes'] = {ref: box for ref, box in obj['boxes'].items()
                            if box['source_observation_id'] in self.observations}
        for frame in self.frames.values():
            frame['observation_ids'] = [oid for oid in frame['observation_ids'] if oid in keep]
        if self.scenes:
            for scene in self.scenes.scenes:
                scene.observations = [obs for obs in scene.observations if obs.observation_id in keep]
            for frame in self.scenes.views.values():
                frame['observations'] = tuple(obs for obs in frame['observations'] if obs.observation_id in keep)

    def export(self):
        def box_data(obj):
            return {**obj, 'appearance': obj['appearance'].tolist()}
        return dict(version='continuous_sandmem_v1', video_id=self.video_id,
            instances=[box_data(self.instances[gid]) for gid in sorted(self.instances)],
            observations=list(self.observations.values()), next_id=self.next_id,
            frames={str(fid): dict(camera=row['camera'].to_dict(), descriptor=row['descriptor'].tolist(),
                                  observation_ids=row['observation_ids'], scene_id=row['scene_id'])
                    for fid, row in self.frames.items()},
            scenes=[dict(scene_id=s.scene_id, reference=s.reference, bounds=s.bounds.tolist(),
                         fixed_limit=s.fixed_limit, closed=s.closed, frame_ids=list(s.cameras),
                         observation_ids=[o.observation_id for o in s.observations])
                    for s in self.scenes.scenes] if self.scenes else [],
            active_voxels=[list(v) for v in sorted(self.scenes.occupied_voxels)] if self.scenes else [],
            voxel_size=self.scenes.policy.voxel_size if self.scenes else None)

    def structure_bytes(self):
        # Actual state plus a bounded streaming serializer workspace. No full
        # float-list expansion or two complete encoded copies exist at runtime.
        return cpu_storage_report(self)['cpu_structural_bytes'] + 262144

    def write_snapshot(self, path):
        """Lossless archival output; never read back to bypass memory eviction.

        Iterate frames/scenes instead of materializing export() for a long video.
        Fixed workspace is charged by structure_bytes; an oversized item fails.
        """
        def checked(row):
            if cpu_storage_report(row)['cpu_structural_bytes'] > 65536:
                raise MemoryError('snapshot row exceeds charged serialization workspace')
            return row

        def frame_rows():
            for fid, row in self.frames.items():
                yield str(fid), checked(dict(camera=row['camera'].to_dict(),
                    descriptor=row['descriptor'].tolist(), observation_ids=row['observation_ids'],
                    scene_id=row['scene_id']))

        def scene_rows():
            for scene in self.scenes.scenes if self.scenes else []:
                yield dict(scene_id=scene.scene_id, reference=scene.reference,
                    bounds=scene.bounds.tolist(), fixed_limit=scene.fixed_limit, closed=scene.closed,
                    frame_ids=iter(scene.cameras),
                    observation_ids=(o.observation_id for o in scene.observations))

        class Pairs:
            def __init__(self, entries): self.entries = entries

        def chunks(value):
            if isinstance(value, (Mapping, Pairs)):
                yield '{'
                entries = value.items() if isinstance(value, Mapping) else value.entries
                for i, (key, item) in enumerate(entries):
                    if i: yield ','
                    yield json.dumps(str(key), ensure_ascii=False)
                    yield ':'
                    yield from chunks(item)
                yield '}'
            elif isinstance(value, (list, tuple, Iterator)):
                yield '['
                for i, item in enumerate(value):
                    if i: yield ','
                    yield from chunks(item)
                yield ']'
            else:
                encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
                if len(encoded.encode()) > 65536:
                    raise MemoryError('snapshot scalar exceeds serialization workspace')
                yield encoded

        if self.structure_bytes() > self.config['memory_limit_bytes']:
            raise MemoryError('cannot serialize a memory already exceeding its budget')
        snapshot = dict(version='continuous_sandmem_v1', video_id=self.video_id,
            instances=(checked({**obj, 'appearance': obj['appearance'].tolist()}) for obj in self.instances.values()),
            observations=iter(self.observations.values()), next_id=self.next_id,
            frames=Pairs(frame_rows()), scenes=scene_rows(),
            active_voxels=(list(v) for v in self.scenes.occupied_voxels) if self.scenes else iter(()),
            voxel_size=self.scenes.policy.voxel_size if self.scenes else None)
        from pathlib import Path
        path = Path(path)
        temporary = path.with_name(path.name + '.partial')
        if path.exists(): raise FileExistsError(path)
        with temporary.open('x', buffering=8192) as stream:
            for chunk in chunks(snapshot): stream.write(chunk)
        temporary.rename(path)

    def enforce_capacity(self):
        limit = self.config['memory_limit_bytes']
        initial = self.structure_bytes()
        if initial <= limit:
            return dict(action='none', bytes=initial)
        keep = set()
        for gid in self.instances:
            rows = sorted((o for o in self.observations.values() if o['instance_id'] == gid),
                          key=lambda o: (o['frame_id'], o['id']))
            keep.update((rows[0]['id'], rows[-1]['id']))
        self._prune(keep)
        removed = []
        while self.structure_bytes() > limit and self.instances:
            gid = min(self.instances, key=lambda g: (max(o['frame_id'] for o in self.observations.values()
                                                       if o['instance_id'] == g), g))
            removed.append(gid)
            self._prune({oid for oid, row in self.observations.items() if row['instance_id'] != gid})
        size = self.structure_bytes()
        if size > limit:
            self.failed = True
            raise MemoryError('required sampled-frame/scene state exceeds structural memory budget')
        return dict(action='first_last_then_oldest_observed', before_bytes=initial,
                    bytes=size, evicted_instance_ids=removed)

    def prepare(self, question, *, maximum, available_through):
        if self.failed:
            raise RuntimeError('memory session failed; QA is prohibited')
        if not question.strip() or self.last_frame_id is None or self.last_frame_id > available_through:
            raise ValueError('nonempty question and causal memory required')
        # Strip only a structural options heading from retrieval, while the
        # original question remains verbatim in actual MLLM input.
        retrieval = re.split(r'\n\s*(?:Options|Choices|选项)\s*[:：]', question, maxsplit=1, flags=re.I)[0]
        query = terms(retrieval) - PROTOCOL_WORDS
        anchors = {gid for gid, obj in self.instances.items() if query & terms(obj['name'])}
        if not anchors:
            anchors = {gid for gid, obj in self.instances.items() if query & terms(obj['description'])}
        fallback = not anchors
        if fallback:
            anchors = set(self.instances)
        anchor_frames = {o['frame_id'] for o in self.observations.values() if o['instance_id'] in anchors}
        relevant = anchors | {o['instance_id'] for o in self.observations.values() if o['frame_id'] in anchor_frames}
        coverage = {fid: {self.observations[oid]['instance_id'] for oid in row['observation_ids']}
                    for fid, row in self.frames.items()}
        selection = select_continuous_frames(tuple(self.frames), available_through=available_through,
            maximum=maximum, temporal_fraction=self.config['qa']['temporal_initial_fraction'],
            relevant_instance_ids=anchors, actual_frame_instances=coverage,
            descriptors={fid: row['descriptor'] for fid, row in self.frames.items()})
        selected_frames = set(selection['frame_ids'])
        payload = dict(record_fields=['id', 'name', 'description', 'actual_frame_ids'],
            records=[[gid, self.instances[gid]['name'], self.instances[gid]['description'],
                      sorted({o['frame_id'] for o in self.observations.values() if o['instance_id'] == gid})]
                     for gid in sorted(relevant)],
            region_fields=['id', 'frame_id', 'source_bbox_xyxy', 'source_hw'],
            provided_frame_regions=[[o['instance_id'], o['frame_id'], o['evidence']['bbox_xyxy'],
                                     o['evidence']['source_hw']] for o in self.observations.values()
                                    if o['instance_id'] in relevant and o['frame_id'] in selected_frames],
            geometry=[dict(instance_id=gid, reference=ref, **deepcopy(box))
                      for gid in sorted(relevant) for ref, box in self.instances[gid]['boxes'].items()],
            frame_scene_links=[[fid, self.frames[fid]['scene_id']] for fid in selection['frame_ids']],
            geometry_notice='Shared estimated coordinates within one continuous reference; arbitrary scale. '
                'Boxes enclose reliable visible surfaces at their stated source frame; not complete object volumes.')
        return dict(question=question, selection=selection, payload=payload,
                    anchor_ids=sorted(anchors), relevant_ids=sorted(relevant), lexical_fallback=fallback)

    def answer(self, question, *, maximum, available_through, sparse, load_image, answerer, limits, audit,
               answer_validator=None):
        """Actual message construction and model calls, with serialized input audit."""
        from sandbox_vlm.qa.continuous_render import render_continuous
        from sandbox_vlm.qa.continuous_messages import build_messages, build_verification_messages
        from sandbox_vlm.qa.prompt import messages_for
        from sandbox_vlm.qa.two_pass import candidate_from_generation, decide
        prepared = self.prepare(question, maximum=maximum, available_through=available_through)
        ids = prepared['selection']['frame_ids']
        images = [load_image(fid) for fid in ids]
        diagram, render_audit = render_continuous(self, prepared, max_side=limits.max_image_side)
        messages, joint = build_messages(question, ids, images, diagram, prepared['payload'], sparse=sparse)
        provided = {(r[0], r[1]) for r in prepared['payload']['provided_frame_regions']}
        has_links = any(gid == box['instance_id'] and self.frames[fid]['camera'].reference == box['reference']
                        for gid, fid in provided for box in prepared['payload']['geometry'])
        two_pass = bool(sparse and has_links)
        audit('prepared', dict(**prepared, rendering=render_audit, two_pass=two_pass))
        try:
            if two_pass:
                draft_messages = messages_for(question, images, ids, 'brief_evidence')
                draft_limits = replace(limits, max_text_tokens=4096)
                audit('draft_input', dict(messages=draft_messages, images=images, limits=vars(draft_limits)))
                draft = answerer.answer(draft_messages, images, draft_limits)
                audit('draft_generation', draft)
                candidate = candidate_from_generation(draft)
                if answer_validator is not None and not answer_validator(candidate.get('answer')):
                    candidate.update(status='failed', answer=None, error='answer_contract_violation')
                messages = build_verification_messages(messages, candidate)
            audit('joint_input', dict(messages=messages, images=joint, limits=vars(limits)))
            from sandbox_vlm.qa.bounded_notes import POLICY
            generation = answerer.answer(messages, joint, limits, note_boundary=POLICY if sparse else None)
            audit('joint_generation', generation)
            if two_pass:
                checked = candidate_from_generation(generation)
                selection_generation = generation
                if answer_validator is not None and not answer_validator(checked.get('answer')):
                    selection_generation = dict(generation, status='answer_failed')
                result = decide(candidate, selection_generation, ids)
            else:
                from sandbox_vlm.qa.d065_prompt_alignment import extract_d065_answer
                try:
                    prediction = extract_d065_answer(generation['raw_output'], sparse=sparse)
                except (KeyError, ValueError):
                    prediction = None
                result = dict(status='ok' if generation['status'] == 'generated' and prediction is not None else 'failed',
                              prediction=prediction)
            if answer_validator is not None and not answer_validator(result.get('prediction')):
                result.update(status='failed', error='answer_contract_violation')
            audit('result', result)
            return result
        finally:
            for image in [*images, diagram]:
                image.close()
