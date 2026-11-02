"""Question-based lexical retrieval and deterministic coverage/viewpoint selection."""
from dataclasses import asdict, dataclass, field
import re

import numpy as np


# Function words only, not an object vocabulary or task parser.
_FUNCTION_WORDS = set('a an the is are was were be been being of in on at to from and or with for this that these those it its what which how many much do does did can could would please'.split())


def terms(text):
    # Optional plurals such as object(s) are one lexical item, not an extra 's'.
    text = re.sub(r'(\w+)\(s\)', r'\1s', text, flags=re.IGNORECASE)
    words = set(re.findall(r'\w+', text.casefold())) - _FUNCTION_WORDS
    # One mechanical form per word, equally for questions and records. Expanding
    # both forms used to count frame/frames twice for a single lexical match.
    return {w[:-1] if len(w) > 3 and w.endswith('s') else w for w in words}


@dataclass
class EvidenceSelection:
    question: str
    instance_ids: list[int]
    pending_ids: list[str]
    frame_ids: list[int]
    uncovered_targets: list[str]
    lexical_fallback: bool
    scores: dict
    frame_choices: list[dict]
    omitted_targets: list[str] = field(default_factory=list)
    max_targets: int | None = None

    def to_dict(self):
        return asdict(self)


def select_evidence(sandbox, question: str, *, max_frames: int, max_targets: int | None = None):
    if sandbox.validate() or not sandbox.video_id:
        raise ValueError('selection requires a valid video sandbox')
    if not isinstance(question, str) or not question.strip():
        raise ValueError('original question must be nonempty')
    if type(max_frames) is not int or max_frames < 0:
        raise ValueError('invalid frame budget')
    if max_targets is not None and (type(max_targets) is not int or max_targets < 1):
        raise ValueError('invalid structure target budget')
    query = terms(question)
    instance_scores = {i.global_instance_id: len(query & terms(i.name + ' ' + i.short_description))
                       for i in sandbox.all_instances()}
    pending_scores = {o.local_observation_id: len(query & terms(o.name + ' ' + o.short_description))
                      for o in sandbox.pending_observations}
    selected_ids = {gid for gid, score in instance_scores.items() if score > 0}
    pending_ids = {oid for oid, score in pending_scores.items() if score > 0}
    fallback = not selected_ids and not pending_ids
    if fallback:
        selected_ids = sandbox.instance_ids()
    # Relevant ambiguity candidates may need inspection even when their names differ.
    for note in sandbox.ambiguity_notes:
        if note.local_observation_id in pending_ids:
            selected_ids.update(note.candidate_instance_ids)
    omitted = []
    if max_targets is not None:
        ranked = sorted([(instance_scores[gid], 'instance', gid) for gid in selected_ids]
                        + [(pending_scores[oid], 'pending', oid) for oid in pending_ids],
                        key=lambda row: (-row[0], row[1], row[2]))
        omitted = sorted(f'{kind}:{identity}' for _, kind, identity in ranked[max_targets:])
        selected_ids = {identity for _, kind, identity in ranked[:max_targets] if kind == 'instance'}
        pending_ids = {identity for _, kind, identity in ranked[:max_targets] if kind == 'pending'}
    coverage = {}
    for gid in sorted(selected_ids):
        for fid in sandbox.frames_for_instance(gid):
            coverage.setdefault(fid, set()).add(f'instance:{gid}')
    for obs in sandbox.pending_observations:
        if obs.local_observation_id in pending_ids:
            for evidence in obs.prompt_evidence:
                # Propagated candidate frame_ids alone cannot establish actual support.
                if evidence['sam_mask_nonempty'] is True:
                    coverage.setdefault(evidence['frame_id'], set()).add(f'pending:{obs.local_observation_id}')
    if fallback and not coverage:
        coverage = {v.frame_id: set() for v in sandbox.all_viewpoints() if v.observed_instance_ids}
    all_targets = {f'instance:{gid}' for gid in selected_ids} | {f'pending:{oid}' for oid in pending_ids}
    cameras = {v.frame_id: v.camera for v in sandbox.all_viewpoints()}
    frames, choices, covered = [], [], set()

    def diversity(fid):
        if not frames or fid not in cameras:
            return 0.0  # Missing pose never becomes invented complementary geometry.
        current = cameras[fid]
        distances = []
        for prior in frames:
            previous = cameras.get(prior)
            if previous is not None and previous.reference == current.reference:
                forward = current.T_ref_from_camera[:3, 2]
                old_forward = previous.T_ref_from_camera[:3, 2]
                distances.append(float(np.arccos(np.clip(np.dot(forward, old_forward), -1, 1))))
        return min(distances) if distances else 0.0

    while len(frames) < max_frames and len(frames) < len(coverage):
        remaining = [f for f in coverage if f not in frames]
        fid = max(remaining, key=lambda f: (len(coverage[f] - covered), diversity(f), -f))
        gain = sorted(coverage[fid] - covered)
        choices.append(dict(frame_id=fid, newly_covered=gain, orientation_diversity_radians=diversity(fid)))
        frames.append(fid)
        covered.update(coverage[fid])
    return EvidenceSelection(question, sorted(selected_ids), sorted(pending_ids), frames,
                             sorted(all_targets - covered), fallback,
                             {'instances': instance_scores, 'pending': pending_scores}, choices, omitted, max_targets)
