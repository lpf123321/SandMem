"""Transactional visual-hypothesis unions supported by actual SAM endpoints."""
from copy import deepcopy
import hashlib
import json

from sandbox_vlm.memory.coverage import CoverageSandbox


def observations(state):
    rows = []
    for fid, links in sorted(state.viewpoints.items(), key=lambda p: int(p[0])):
        for index, link in enumerate(links):
            instance = state.instance_by_id(link['visual_instance_id'])
            slot = link.get('observation_slot', index)
            rows.append(dict(observation_id=f'F{fid}_O{slot}', name=instance['name'],
                visual_instance_id=link['visual_instance_id'], evidence=deepcopy(link['evidence'])))
    return rows


def observation_key(row):
    # The actual box/frame, rather than a mutable hypothesis ID, identifies work.
    return hashlib.sha256(json.dumps(row['evidence'], sort_keys=True).encode()).hexdigest()


def merge_correspondences(state, obs, edges, *, watermark, threshold=.7):
    """Return a validated copy; never merge co-visible components transitively."""
    by_id = {o['observation_id']: o for o in obs}
    instances = {i['visual_instance_id']: i for i in state.instances}
    parent = {i: i for i in instances}
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    frames = {i: {int(f) for f, links in state.viewpoints.items()
                  if any(v['visual_instance_id'] == i for v in links)} for i in parent}
    audit = []
    for edge in edges:
        if not edge['would_reuse']:
            continue
        a, b = by_id[edge['source']], by_id[edge['target']]
        sf, tf = a['evidence']['frame_id'], b['evidence']['frame_id']
        if not (sf < tf <= watermark and tf-sf < 16
                and edge['source_frame'] == sf and edge['target_frame'] == tf
                and edge['mask_iou'] >= threshold
                and edge['continuous_nonempty'] and edge['endpoint_prompt_nonempty']
                and edge['forward_candidates'] == [edge['target']]
                and edge['co_visible_reverse_candidates'] == [edge['source']]):
            raise ValueError('unsupported/future/nonunique short correspondence')
        for endpoint in (a, b):
            if not any(v['visual_instance_id'] == endpoint['visual_instance_id']
                       and v['evidence'] == endpoint['evidence']
                       for v in state.viewpoints.get(str(endpoint['evidence']['frame_id']), [])):
                raise ValueError('correspondence endpoint lacks an actual link')
        x, y = root(a['visual_instance_id']), root(b['visual_instance_id'])
        status = 'already_shared' if x == y else 'same_frame_conflict' if frames[x] & frames[y] else 'merged'
        if status == 'merged':
            lo, hi = sorted((x, y)); parent[hi] = lo; frames[lo] |= frames[hi]
        audit.append(dict(edge, merge_status=status))
    groups = {}
    for i in parent:
        groups.setdefault(root(i), []).append(i)
    labeled_groups = [(min(ids) if hasattr(state, 'next_visual_instance_id') else new, ids)
                      for new, ids in enumerate(groups.values())]
    remap = {old: new for new, ids in labeled_groups for old in ids}
    result = CoverageSandbox.from_dict(state.to_dict())
    result.instances = []
    for new, ids in labeled_groups:
        row = deepcopy(instances[ids[0]])
        if any(instances[i]['local_geometry'] for i in ids) and len(ids) > 1:
            raise ValueError('short visual correspondence cannot fuse local geometry')
        latest = max((instances[i] for i in ids), key=lambda x: x['context_frame_id'])
        row.update(visual_instance_id=new, context_frame_id=latest['context_frame_id'],
                   context_descriptor=deepcopy(latest['context_descriptor']))
        result.instances.append(row)
    for links in result.viewpoints.values():
        for link in links:
            link['visual_instance_id'] = remap[link['visual_instance_id']]
    result.observation_index = {k: remap[v] for k, v in result.observation_index.items()}
    notes = []
    for note in result.ambiguities:
        note['observed_visual_instance_id'] = remap[note['observed_visual_instance_id']]
        note['candidate_visual_instance_ids'] = sorted({remap[i] for i in note['candidate_visual_instance_ids']}
                                                     - {note['observed_visual_instance_id']})
        if note['candidate_visual_instance_ids']:
            notes.append(note)
    result.ambiguities = notes
    result.validate()
    return result, dict(edges=audit, old_to_new=remap,
                        removed_hypotheses=len(state.instances)-len(result.instances))
