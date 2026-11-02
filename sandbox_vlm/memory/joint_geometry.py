"""D063 typed SAM/joint-geometry union, preserving the original visual-state transaction."""
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
        if not (sf < tf <= watermark and edge['source_frame']==sf and edge['target_frame']==tf):
            raise ValueError('future or detached correspondence')
        if edge.get('evidence_type') in ('D063_joint_ambiguity_geometry', 'D076_multi_observation_geometry'):
            if edge['evidence_type'] == 'D076_multi_observation_geometry':
                multi=edge['multi_candidate_match']
                passed_multi=[s for s in multi['scores'] if s['passes_numeric_gates']]
                selected=multi['matched_instance_id']
                if not (multi['status']=='matched' and len(passed_multi)==1
                        and passed_multi[0]['instance_id']==selected
                        and edge['candidate_observation_ids'].get(str(selected))==edge['source']):
                    raise ValueError('unsupported multi-candidate geometry decision')
            match=edge['match_result']
            passed=[s for s in match['scores'] if s['passes_numeric_gates']]
            packet=edge['geometry_evidence']
            ids=packet['frame_ids']
            if not (len(ids)==16 and len(set(ids))==16 and ids==sorted(ids)
                    and sf in ids and tf in ids and max(ids)<=packet['available_through']<=watermark
                    and edge['reference']==packet['reference'] and edge['reference']
                    and match['status']=='matched' and match['matched_instance_id']==0
                    and len(passed)==1 and passed[0]['instance_id']==0):
                raise ValueError('unsupported joint geometry correspondence')
            for endpoint, key in [(a,'source'),(b,'target')]:
                actual=packet[key]
                if not (actual['evidence']==endpoint['evidence'] and actual['bbox_3d'] is not None
                        and actual['bbox_3d']['frame']==edge['reference']):
                    raise ValueError('joint geometry endpoint/reference mismatch')
        elif edge.get('evidence_type') is not None:
            raise ValueError('unknown correspondence evidence type')
        elif not (tf-sf < 16 and edge['mask_iou'] >= threshold and edge['continuous_nonempty'] and edge['endpoint_prompt_nonempty']
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
            raise ValueError('D063 correspondence cannot fuse unrelated local geometry')
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
