"""D061 typed SAM/geometry union, preserving the original visual-state transaction."""
from copy import deepcopy
import hashlib
import json

from sandbox_vlm.memory.coverage import CoverageSandbox


def observations(state):
    rows = []
    for fid, links in sorted(state.viewpoints.items(), key=lambda p: int(p[0])):
        for index, link in enumerate(links):
            instance = state.instances[link['visual_instance_id']]
            rows.append(dict(observation_id=f'F{fid}_O{index}', name=instance['name'],
                visual_instance_id=link['visual_instance_id'], evidence=deepcopy(link['evidence'])))
    return rows


def observation_key(row):
    # The actual box/frame, rather than a mutable hypothesis ID, identifies work.
    return hashlib.sha256(json.dumps(row['evidence'], sort_keys=True).encode()).hexdigest()


def merge_correspondences(state, obs, edges, *, watermark, threshold=.7):
    """Return a validated copy; never merge co-visible components transitively."""
    by_id = {o['observation_id']: o for o in obs}
    parent = list(range(len(state.instances)))
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
        if not (sf < tf <= watermark and tf-sf < 16 and edge['source_frame']==sf and edge['target_frame']==tf):
            raise ValueError('future or nonlocal correspondence')
        if edge.get('evidence_type') == 'D061_original_D014_local_match':
            match=edge['match_result']
            passed=[s for s in match['scores'] if s['passes_numeric_gates']]
            if not (match['status']=='matched' and match['matched_instance_id']==edge['matched_local_group']
                    and len(passed)==1 and passed[0]['instance_id']==edge['matched_local_group'] and edge['reference']):
                raise ValueError('unsupported geometry correspondence')
        elif not (edge['mask_iou'] >= threshold and edge['continuous_nonempty'] and edge['endpoint_prompt_nonempty']
                and edge['forward_candidates']==[edge['target']] and edge['co_visible_reverse_candidates']==[edge['source']]):
            raise ValueError('unsupported SAM correspondence')
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
    for i in range(len(parent)):
        groups.setdefault(root(i), []).append(i)
    remap = {old: new for new, ids in enumerate(groups.values()) for old in ids}
    result = CoverageSandbox.from_dict(state.to_dict())
    result.instances = []
    for new, ids in enumerate(groups.values()):
        row = deepcopy(state.instances[ids[0]])
        if any(state.instances[i]['local_geometry'] for i in ids) and len(ids) > 1:
            raise ValueError('D061 correspondence cannot fuse unrelated local geometry')
        latest = max((state.instances[i] for i in ids), key=lambda x: x['context_frame_id'])
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
