"""Question-independent temporal seeds, instance coverage, visual complements.

Uses supplied frozen visual descriptors; no new model, task classifier, or
answer input. The candidate set is the full available sampled-video prefix.
"""
import numpy as np


def select_continuous_frames(frame_ids, *, available_through, maximum,
                             temporal_fraction, relevant_instance_ids,
                             actual_frame_instances, descriptors):
    ids = tuple(frame_ids)
    if (not ids or any(type(f) is not int or f < 0 for f in ids)
            or tuple(sorted(set(ids))) != ids or ids[-1] > available_through):
        raise ValueError('complete available sampled prefix required')
    if type(maximum) is not int or maximum < 1 or not 0 < temporal_fraction <= 1:
        raise ValueError('positive existing frame budget and temporal fraction required')
    budget = min(maximum, len(ids))
    seeds = min(budget, max(1, int(maximum * temporal_fraction)))
    chosen, trace = [], []
    for i in range(seeds):
        target = ((ids[0] + ids[-1]) / 2 if seeds == 1
                  else ids[0] + i * (ids[-1] - ids[0]) / (seeds - 1))
        fid = min((f for f in ids if f not in chosen), key=lambda f: (abs(f - target), f))
        chosen.append(fid)
        trace.append(dict(frame_id=fid, stage='temporal'))
    targets = set(relevant_instance_ids)
    coverage = {f: set(actual_frame_instances.get(f, ())) & targets for f in ids}
    covered = set().union(*(coverage[f] for f in chosen))
    if len(chosen) < budget:
        # Missing features cannot masquerade as visually novel intermediates.
        try:
            vectors = np.stack([np.asarray(descriptors[f], dtype=np.float64).reshape(-1) for f in ids])
        except (KeyError, ValueError) as exc:
            raise ValueError('all candidate frames need compatible frozen visual descriptors') from exc
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not vectors.shape[1] or not np.isfinite(vectors).all() or np.any(norms <= 0):
            raise ValueError('invalid frozen visual descriptors')
        vectors /= norms
        position = {f: i for i, f in enumerate(ids)}
        similarity = np.max(vectors @ vectors[[position[f] for f in chosen]].T, axis=1)
        while len(chosen) < budget:
            remaining = [f for f in ids if f not in chosen]
            gains = {f: len(coverage[f] - covered) for f in remaining}
            fid = max(remaining, key=lambda f: (gains[f], 1 - float(similarity[position[f]]), -f))
            trace.append(dict(frame_id=fid,
                              stage='instance_coverage' if gains[fid] else 'visual_diversity',
                              newly_covered=sorted(coverage[fid] - covered)))
            chosen.append(fid)
            covered.update(coverage[fid])
            similarity = np.maximum(similarity, vectors @ vectors[position[fid]])
    return dict(frame_ids=sorted(chosen), choices=trace,
                uncovered_instance_ids=sorted(targets - covered), maximum=maximum)
