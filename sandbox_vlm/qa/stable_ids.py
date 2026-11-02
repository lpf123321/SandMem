"""D069: preserve frozen SAM display IDs across additional geometry unions.

Only the QA payload is renamed. Internal dense IDs, grouping, observations,
descriptions, and inference images remain unchanged. No counting is performed.
"""
from copy import deepcopy
import json


def geometry_display_ids(sam, candidate):
    def links(state):
        result = {}
        for fid, rows in state['viewpoints'].items():
            for row in rows:
                assert int(fid) == row['evidence']['frame_id']
                key = json.dumps(row['evidence'], sort_keys=True, separators=(',', ':'))
                if key in result:
                    raise ValueError('duplicate actual observation identity')
                result[key] = row['visual_instance_id']
        return result
    original, revised = links(sam), links(candidate)
    if original.keys() != revised.keys():
        raise ValueError('display-only mapping cannot add/drop actual observations')
    first = {}
    for fid, rows in sam['viewpoints'].items():
        for row in rows:
            gid = row['visual_instance_id']
            first[gid] = min(first.get(gid, int(fid)), int(fid))
    groups, destinations = {}, {}
    for key, sid in original.items():
        gid = revised[key]
        groups.setdefault(gid, set()).add(sid)
        destinations.setdefault(sid, set()).add(gid)
    if any(len(gids) != 1 for gids in destinations.values()):
        raise ValueError('additional geometry split an original SAM component')
    if set(groups) != {i['visual_instance_id'] for i in candidate['instances']}:
        raise ValueError('candidate has unsupported display instances')
    mapping = {gid: min(ids, key=lambda sid: (first[sid], sid)) for gid, ids in groups.items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('display mapping collision')
    return mapping


def remap_payload(payload, mapping):
    result = deepcopy(payload)
    if all(k == v for k, v in mapping.items()):
        return result
    hypotheses = result['hypotheses']
    if isinstance(hypotheses, dict):
        raise ValueError('noncontiguous display IDs require explicit-ID format; stop this bounded comparison')
    if result['record_fields']:
        col = result['record_fields'].index('visual_id')
        ids = [row[col] for row in hypotheses]
        for row in hypotheses:
            row[col] = mapping[row[col]]
    else:
        ids = [row['visual_id'] for row in hypotheses]
        for row in hypotheses:
            row['visual_id'] = mapping[row['visual_id']]
    if set(ids) != set(mapping):
        raise ValueError('payload lost candidate instances')
    if not isinstance(result['ambiguities'], list):
        raise ValueError('bounded comparison requires explicit ambiguity IDs')
    for note in result['ambiguities']:
        note['observed_visual_id'] = mapping[note['observed_visual_id']]
        note['candidate_visual_ids'] = [mapping[gid] for gid in note['candidate_visual_ids']]
    return result
