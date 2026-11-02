"""First/last actual observations, then least-recently-observed eviction.

Only instances-only visual memory is supported. Planning is transactional: the
caller publishes the returned state only after accepting the complete budget.
Neither archived states nor model/answer data are consulted.
"""
from copy import deepcopy
import hashlib
import json

from sandbox_vlm.memory.storage import cpu_storage_report

POLICY = 'first_last_then_oldest_observed'


def observation_id(frame_id, link, position):
    """Keep source-region IDs stable when an earlier row in a frame is removed."""
    return f'F{int(frame_id)}_O{link.get("observation_slot", position)}'


def observation_link(state, frame_id, slot):
    for position, link in enumerate(state.viewpoints.get(str(frame_id), [])):
        if link.get('observation_slot', position) == slot:
            return link
    raise KeyError(f'forgotten observation F{frame_id}_O{slot}')


def _prune_auxiliary(auxiliary, state):
    """Clean the existing geometry/mask-catalog envelope; charge other state intact.

Other instance-bearing caches must be rebuilt by their owner rather than being
silently retained or heuristically rewritten with unknown field semantics.
"""
    if auxiliary is None:
        return None
    ids = {i['visual_instance_id'] for i in state.instances}
    pairs = {(int(f), l['visual_instance_id']) for f, rows in state.viewpoints.items() for l in rows}
    obs = {observation_id(f, l, j) for f, rows in state.viewpoints.items() for j, l in enumerate(rows)}
    result = deepcopy(auxiliary)
    geometry = result.get('geometry') if isinstance(result, dict) else None
    if geometry is not None:
        if not isinstance(geometry, dict) or 'spatial_extents' not in geometry:
            raise ValueError('retention requires explicit spatial geometry; rebuild derived QA geometry')
        extents = []
        for extent in geometry['spatial_extents']:
            extent['instances'] = [i for i in extent['instances'] if i['global_instance_id'] in ids]
            local = {i['global_instance_id'] for i in extent['instances']}
            views = []
            for view in extent['viewpoints']:
                view['observed_instance_ids'] = [i for i in view['observed_instance_ids']
                    if i in local and (view['frame_id'], i) in pairs]
                if view['observed_instance_ids']:
                    views.append(view)
            extent['viewpoints'] = views
            # An observed range is independent scene evidence, not an instance
            # tombstone. Keep its bounds/reference even if all objects expired.
            extents.append(extent)
        geometry['spatial_extents'] = extents
    if isinstance(result, dict) and 'mask_catalog' in result:
        result['mask_catalog'] = {k: v for k, v in result['mask_catalog'].items() if k in obs}
    edge_fields = ('accepted_edges', 'new_edges')
    for field in edge_fields:
        if isinstance(result, dict) and field in result:
            result[field] = [e for e in result[field] if e['source'] in obs and e['target'] in obs]
    # The coverage ledger, mapping and discovery schedule have independent roles:
    # keep and charge them, rather than deleting them to manufacture a fit.
    def check_unknown(value):
        if isinstance(value, dict):
            if any(k in value for k in ('visual_instance_id', 'global_instance_id',
                                       'candidate_visual_instance_ids', 'observed_instance_ids',
                                       'records', 'hypotheses', 'payload', 'visual_sandbox',
                                       'accepted_edges', 'new_edges', 'local_SAM_correspondences')):
                raise ValueError('unmanaged instance-bearing auxiliary cache; rebuild it from retained memory')
            for v in value.values():
                check_unknown(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                check_unknown(v)
    if isinstance(result, dict):
        check_unknown({k: v for k, v in result.items() if k not in ('geometry', 'mask_catalog', *edge_fields)})
    else:
        check_unknown(result)
    return result


def fit_instance_memory(state, limit_bytes, *, auxiliary_state=None, encoded_reserve_bytes=0):
    """Return (bounded working copy, auxiliary copy, report); never mutate input.

Both reachable CPU structure and the complete serialized envelope are bounded.
The backend can also reserve bytes for its previous transactional GPU snapshot.
If even mandatory non-instance state does not fit, return capacity_limited.
"""
    if type(limit_bytes) is not int or limit_bytes < 1:
        raise ValueError('positive integer memory limit required')
    if type(encoded_reserve_bytes) is not int or encoded_reserve_bytes < 0:
        raise ValueError('nonnegative encoded reserve required')
    if getattr(state, 'retain_ambiguities', True) or state.ambiguities:
        raise ValueError('first/last retention requires instances-only memory without pending observations')
    state.validate()
    work = deepcopy(state)
    aux = deepcopy(auxiliary_state)
    def measure():
        encoded = json.dumps(dict(visual_sandbox=work.to_dict(), auxiliary_state=aux),
            ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        host = cpu_storage_report(dict(sandbox=work, auxiliary_state=aux))['cpu_structural_bytes']
        return dict(encoded_bytes=len(encoded), cpu_structural_bytes=host,
                    replacement_peak_bytes=len(encoded)+encoded_reserve_bytes)
    def fits(m):
        return max(m['cpu_structural_bytes'], m['replacement_peak_bytes']) <= limit_bytes
    before = measure()
    report = dict(policy=POLICY, limit_bytes=limit_bytes, before=before,
                  compaction_applied=False, removed_observation_links=0,
                  evicted_instance_ids=[], eviction_last_observed_frames=[], paging=False)
    if fits(before):
        report.update(status='ok', after=before)
        return work, aux, report

    report['compaction_applied'] = True
    work.next_visual_instance_id = max(getattr(work, 'next_visual_instance_id', 0),
        max((i['visual_instance_id'] for i in work.instances), default=-1)+1)
    frames = {i['visual_instance_id']: [] for i in work.instances}
    for fid, rows in work.viewpoints.items():
        for link in rows:
            frames[link['visual_instance_id']].append(int(fid))
    if any(not f for f in frames.values()):
        raise ValueError('cannot retain instance without an actual observation')
    endpoints = {i: {min(f), max(f)} for i, f in frames.items()}
    old_links = sum(map(len, work.viewpoints.values()))
    def prune(keep_endpoints):
        views = {}
        for fid, rows in work.viewpoints.items():
            retained = []
            for j, link in enumerate(rows):
                gid = link['visual_instance_id']
                if gid in keep_endpoints and int(fid) in keep_endpoints[gid]:
                    link.setdefault('observation_slot', j)
                    retained.append(link)
            if retained:
                views[fid] = retained
        work.viewpoints = views
        # Retain only keys belonging to retained actual observations. Older
        # snapshots lack per-link keys; rebuild their canonical dedup key.
        index = {}
        by_id = {i['visual_instance_id']: i for i in work.instances}
        for rows in views.values():
            for link in rows:
                gid = link['visual_instance_id']
                key = link.get('observation_key')
                if key is None:
                    key = hashlib.sha256(json.dumps(dict(frame_id=link['evidence']['frame_id'],
                        name=by_id[gid]['name'].casefold().strip(), evidence=link['evidence']),
                        sort_keys=True).encode()).hexdigest()
                    if work.observation_index.get(key) != gid:
                        raise ValueError('legacy observation-key provenance missing; cannot safely compact alias observations')
                link['observation_key'] = key
                index[key] = gid
        work.observation_index = index
    prune(endpoints)
    for instance in work.instances:
        gid = instance['visual_instance_id']
        if instance.get('context_frame_id', max(endpoints[gid])) != max(endpoints[gid]):
            raise ValueError('latest context must refer to the latest actual observation')
    aux = _prune_auxiliary(aux, work)
    report['removed_observation_links'] = old_links-sum(map(len, work.viewpoints.values()))
    measured = measure()
    for gid in sorted(frames, key=lambda i: (max(frames[i]), i)):
        if fits(measured):
            break
        report['evicted_instance_ids'].append(gid)
        report['eviction_last_observed_frames'].append(max(frames[gid]))
        work.instances = [i for i in work.instances if i['visual_instance_id'] != gid]
        del endpoints[gid]
        prune(endpoints)
        aux = _prune_auxiliary(aux, work)
        measured = measure()
    work.validate()
    report.update(status='ok' if fits(measured) else 'capacity_limited', after=measured,
                  retained_instances=len(work.instances), retained_observation_links=sum(map(len, work.viewpoints.values())))
    return work, aux, report
