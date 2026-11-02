"""Apply externally verified decisions; this module never infers identity/reliability.

No appearance threshold, geometry fusion rule or automatic new-instance policy is
introduced here. Callers must supply those decisions after evidence verification.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from sandbox_vlm.memory.geometry import CameraParams
from sandbox_vlm.memory.sandbox import AmbiguityNote, ObjectInstance, Sandbox, Viewpoint
from sandbox_vlm.perception.observations import LocalObservation


def migrate_pending_coordinates(sandbox, *, source_reference, target_reference, transform,
                                overlap_proof, history_proof):
    """Apply a verified reference bridge without asserting old geometry/identity.

    Only an unconfirmed local component may move. Old ambiguity keeps its
    pending-alignment prefix, so QA cannot promote it on coordinate metadata
    alone. No historical camera, source link, instance or feature is fabricated.
    """
    from dataclasses import replace
    from sandbox_vlm.memory.geometry import decompose_similarity,transform_bbox
    _check_sandbox(sandbox)
    if (not source_reference or not target_reference or source_reference==target_reference
            or overlap_proof.get('status')!='aligned'
            or history_proof.get('status')!='historical_relocalized'
            or history_proof.get('verified_pairs',0)<1):
        raise ValueError('independently verified overlap/history bridge required')
    decompose_similarity(transform)
    if any(e.coordinate_reference==source_reference for e in sandbox.spatial_extents):
        raise ValueError('confirmed structures require separate full reconciliation')
    moving={o.local_observation_id for o in sandbox.pending_observations if o.coordinate_reference==source_reference}
    replacements=[]
    for obs in sandbox.pending_observations:
        if obs.local_observation_id not in moving:
            replacements.append(obs);continue
        updated=replace(obs,coordinate_reference=target_reference,
            bbox_3d=transform_bbox(obs.bbox_3d,transform,dst_reference=target_reference) if obs.bbox_3d is not None else None)
        updated.__post_init__();replacements.append(updated)
    notes=[replace(n,reason=n.reason+'; coordinates_migrated_geometry_and_identity_unverified')
           if n.local_observation_id in moving else n for n in sandbox.ambiguity_notes]
    # All transformed copies are validated before changing the accepted object.
    candidate=Sandbox(sandbox.spatial_extents,video_id=sandbox.video_id,
                      pending_observations=replacements,ambiguity_notes=notes)
    _check_sandbox(candidate)
    sandbox.pending_observations,sandbox.ambiguity_notes=replacements,notes
    return dict(status='coordinates_migrated_only',source_reference=source_reference,
        target_reference=target_reference,pending_observation_ids=sorted(moving),
        historical_cameras_created=0,instances_created=0,observation_links_created=0,
        historical_geometry_verified=False,identity_confirmed=False)


def _check_sandbox(sandbox: Sandbox):
    problems = sandbox.validate()
    if problems:
        raise ValueError("invalid sandbox: " + "; ".join(problems))
    if not sandbox.video_id:
        raise ValueError("updates require an explicit video_id")


def hold_pending(sandbox: Sandbox, observation: LocalObservation,
                 candidate_instance_ids: list[int], reason: str):
    """Retain compact local state, with no confirmed object or viewpoint links."""
    if not sandbox.retain_unresolved:
        raise ValueError('instances-only memory does not retain unresolved observations')
    _check_sandbox(sandbox)
    observation.__post_init__()
    if observation.video_id != sandbox.video_id:
        raise ValueError("cross-video observation")
    if not reason.strip():
        raise ValueError("pending association requires a reason")
    candidates = sorted(set(candidate_instance_ids))
    if not set(candidates) <= sandbox.instance_ids():
        raise ValueError("pending candidates contain unknown instance IDs")
    existing = next((o for o in sandbox.pending_observations
                     if o.local_observation_id == observation.local_observation_id), None)
    if existing is not None and existing.to_dict() != observation.to_dict():
        raise ValueError("conflicting local observation ID; explicit revision required")
    note = AmbiguityNote(observation.local_observation_id, candidates, list(observation.frame_ids), reason)
    if existing is None:
        sandbox.pending_observations.append(deepcopy(observation))
    sandbox.ambiguity_notes = [n for n in sandbox.ambiguity_notes
                               if n.local_observation_id != observation.local_observation_id] + [note]


def _validate_pending_resolution(sandbox, observation):
    existing = next((o for o in sandbox.pending_observations
                     if o.local_observation_id == observation.local_observation_id), None)
    if existing is None or existing.frame_ids == observation.frame_ids:
        return
    # Old candidate states included propagation frames. Explicit prompt evidence
    # can narrow those to actual frames, but cannot drop already established prompts.
    actual = {p["frame_id"] for p in observation.prompt_evidence}
    if (existing.prompt_evidence or not actual or actual != set(observation.frame_ids)
            or not actual <= set(existing.frame_ids)):
        raise ValueError("pending resolution must preserve actual evidence frame IDs")


def record_verified_observation(sandbox: Sandbox, extent_id: int, instance_id: int,
                                observation: LocalObservation, cameras: dict[int, CameraParams]):
    """Apply an upstream verified match to an EXISTING instance, not a matching rule.

    Only source frames carried by the local observation may acquire links. A
    camera merely looking toward an object never creates an observation link.
    All rejection checks precede mutations; no full-sandbox transaction copy.
    """
    _check_sandbox(sandbox)
    observation.__post_init__()
    if observation.video_id != sandbox.video_id:
        raise ValueError("cross-video observation")
    extent = next((e for e in sandbox.spatial_extents if e.extent_id == extent_id), None)
    if extent is None or instance_id not in extent.instance_ids():
        raise ValueError("match must name an existing instance in the extent")
    if observation.bbox_3d is None or observation.coordinate_reference != extent.coordinate_reference:
        raise ValueError("verified match requires geometry in the common reference")
    _validate_pending_resolution(sandbox, observation)
    viewpoints = {v.frame_id: v for v in extent.viewpoints}
    additions = []
    for frame_id in observation.frame_ids:
        camera = cameras.get(frame_id)
        if camera is None or camera.frame_id != frame_id or camera.reference != extent.coordinate_reference:
            raise ValueError("missing or inconsistent camera for actual observation")
        if frame_id in viewpoints:
            old = viewpoints[frame_id].camera
            if (old.image_size != camera.image_size
                    or not np.allclose(old.intrinsic, camera.intrinsic, atol=1e-8, rtol=1e-8)
                    or not np.allclose(old.T_ref_from_camera, camera.T_ref_from_camera, atol=1e-8, rtol=1e-8)):
                raise ValueError("conflicting camera estimate; explicit geometric reconciliation required")
        else:
            additions.append(Viewpoint(frame_id, deepcopy(camera), [instance_id]))
    for frame_id in observation.frame_ids:
        if frame_id in viewpoints and instance_id not in viewpoints[frame_id].observed_instance_ids:
            viewpoints[frame_id].observed_instance_ids.append(instance_id)
    extent.viewpoints.extend(additions)
    sandbox.pending_observations = [o for o in sandbox.pending_observations
                                     if o.local_observation_id != observation.local_observation_id]
    sandbox.ambiguity_notes = [n for n in sandbox.ambiguity_notes
                               if n.local_observation_id != observation.local_observation_id]


def merge_verified_instances(sandbox: Sandbox, instance_ids: list[int], replacement: ObjectInstance):
    """Apply a verified merge; caller supplies fused geometry/feature explicitly.

    This does not verify that the instances depict the same physical object.
    It implements the proposal's reference-consistency requirement after that
    decision, and prohibits merging across unaligned coordinate references.
    """
    _check_sandbox(sandbox)
    ids = set(instance_ids)
    if len(ids) < 2 or replacement.global_instance_id not in ids:
        raise ValueError("merge needs distinct IDs and must retain one existing ID")
    extent = next((e for e in sandbox.spatial_extents if ids <= e.instance_ids()), None)
    if extent is None:
        raise ValueError("merge requires existing instances in the same aligned extent")
    if replacement.bbox_3d.frame != extent.coordinate_reference:
        raise ValueError("replacement box reference mismatch")
    replacement = deepcopy(replacement)
    extent.instances = [i for i in extent.instances if i.global_instance_id not in ids] + [replacement]
    for viewpoint in extent.viewpoints:
        viewpoint.observed_instance_ids = sorted({replacement.global_instance_id if gid in ids else gid
                                                  for gid in viewpoint.observed_instance_ids})
    for note in sandbox.ambiguity_notes:
        note.candidate_instance_ids = sorted({replacement.global_instance_id if gid in ids else gid
                                              for gid in note.candidate_instance_ids})


def split_verified_instance(sandbox: Sandbox, instance_id: int, replacements: list[ObjectInstance],
                             frame_assignments: dict[int, list[int]]):
    """Apply an upstream verified split with explicit per-frame evidence assignments.

    The caller must resolve every old observation and provide at least one actual
    evidence frame per replacement. Ambiguous candidate references expand to all
    replacement IDs, without becoming confirmed viewpoint links.
    """
    _check_sandbox(sandbox)
    extent = next((e for e in sandbox.spatial_extents if instance_id in e.instance_ids()), None)
    new_ids = {i.global_instance_id for i in replacements}
    if extent is None or len(new_ids) < 2 or len(new_ids) != len(replacements):
        raise ValueError("split requires an existing instance and distinct replacement IDs")
    if new_ids & (sandbox.instance_ids() - {instance_id}):
        raise ValueError("replacement ID already belongs to another instance")
    if any(i.bbox_3d.frame != extent.coordinate_reference for i in replacements):
        raise ValueError("replacement references must match the aligned extent")
    observed = {v.frame_id for v in extent.viewpoints if instance_id in v.observed_instance_ids}
    if not observed or set(frame_assignments) != observed:
        raise ValueError("split must assign exactly the existing observation frames")
    assigned = set()
    for frame_ids in frame_assignments.values():
        if not frame_ids or len(set(frame_ids)) != len(frame_ids) or not set(frame_ids) <= new_ids:
            raise ValueError("each frame assignment must name distinct replacement instances")
        assigned.update(frame_ids)
    if assigned != new_ids:
        raise ValueError("every replacement needs actual observation evidence")
    replacements = deepcopy(replacements)
    extent.instances = [i for i in extent.instances if i.global_instance_id != instance_id] + replacements
    for viewpoint in extent.viewpoints:
        if instance_id in viewpoint.observed_instance_ids:
            viewpoint.observed_instance_ids = sorted(
                (set(viewpoint.observed_instance_ids) - {instance_id}) | set(frame_assignments[viewpoint.frame_id]))
    for note in sandbox.ambiguity_notes:
        if instance_id in note.candidate_instance_ids:
            note.candidate_instance_ids = sorted((set(note.candidate_instance_ids) - {instance_id}) | new_ids)
