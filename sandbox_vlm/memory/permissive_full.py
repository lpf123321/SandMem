"""D097: apply D096 births to frozen D053 actual SAM evidence, at a prefix.

The full-run cache contains actual single-frame segments and accepted short SAM
tracks, rather than D083's 16-frame local observations. Only those accepted SAM
edges establish local tracks here. Old appearance/geometry IDs are not unions.
"""
from copy import deepcopy

from sandbox_vlm.memory.coverage import CoverageSandbox
from sandbox_vlm.memory.permissive_birth import PermissiveBirthStore
from sandbox_vlm.memory.short_track import observations, merge_correspondences
from sandbox_vlm.perception.observations import LocalObservation
from sandbox_vlm.memory.sandbox import AppearanceFeature


def admit_frozen_prefix(base, auxiliary, *, watermark):
    obs = observations(base)
    if any(o['evidence']['frame_id'] > watermark for o in obs):
        raise ValueError('future actual evidence')
    # Expand old hypotheses to actual segment endpoints before applying the
    # frozen SAM edges; old numeric/appearance identity cannot leak into births.
    expanded = CoverageSandbox(base.video_id, '')
    expanded.question_key = base.question_key
    for index, o in enumerate(obs):
        old = base.instances[o['visual_instance_id']]
        instance = deepcopy(old)
        instance.update(visual_instance_id=index, local_geometry=[],
                        context_frame_id=o['evidence']['frame_id'])
        expanded.instances.append(instance)
        expanded.viewpoints.setdefault(str(o['evidence']['frame_id']), []).append(
            dict(visual_instance_id=index, evidence=deepcopy(o['evidence']), identity_status='visual_hypothesis'))
    expanded.validate()
    endpoints = observations(expanded)
    if [o['observation_id'] for o in obs] != [o['observation_id'] for o in endpoints]:
        raise ValueError('source endpoint order changed')
    tracks, sam_audit = merge_correspondences(expanded, endpoints,
        auxiliary['short_track']['accepted_edges'], watermark=watermark)
    groups = {}
    for index, o in enumerate(obs):
        groups.setdefault(sam_audit['old_to_new'][index], []).append(o)
    store = PermissiveBirthStore(base.video_id)
    events = []
    endpoint_sources = {}
    for members in groups.values():
        first = members[0]
        source = base.instances[first['visual_instance_id']]
        evidence = [o['evidence'] for o in members]
        oid = 'frozen_SAM_track/' + min(o['observation_id'] for o in members)
        local = LocalObservation(video_id=base.video_id, local_observation_id=oid,
            coordinate_reference='unavailable/' + oid, name=source['name'],
            short_description=source['description'], frame_ids=sorted({p['frame_id'] for p in evidence}),
            appearance=AppearanceFeature.from_dict(source['appearance']),
            failure_reasons=['frozen_full_scope_has_no_object_3d'], prompt_evidence=evidence)
        event = store.admit(local, alignment_reliable=False)
        events.append(event)
        for o in members:
            endpoint_sources[o['observation_id']] = oid
    store.validate()
    endpoint_ids = {endpoint: store.source_to_instance[oid] for endpoint, oid in endpoint_sources.items()}
    # Existing serializer is kept verbatim: all born records have explicit real
    # links, and its notice still states that physical identity is unverified.
    answer = CoverageSandbox(base.video_id, '')
    answer.question_key = base.question_key
    dense = {gid: index for index, gid in enumerate(store.records)}
    for gid, r in store.records.items():
        member = next(o for o in obs if endpoint_ids[o['observation_id']] == gid)
        old = base.instances[member['visual_instance_id']]
        answer.instances.append(dict(visual_instance_id=dense[gid], identity_status='visual_hypothesis',
            name=r['name'], description=r['description'], appearance=r['appearance'], local_geometry=[],
            context_descriptor=deepcopy(old['context_descriptor']), context_frame_id=member['evidence']['frame_id']))
    for fid, links in store.viewpoints.items():
        answer.viewpoints[fid] = [dict(visual_instance_id=dense[v['instance_id']],
            evidence=deepcopy(v['evidence']), identity_status='visual_hypothesis') for v in links]
    # Translate old ambiguity notes through their real endpoint, without
    # expanding one old candidate into an invented all-pairs correspondence.
    by_evidence = {str(o['evidence']): o for o in obs}
    old_to_new = {}
    for o in obs:
        old_to_new.setdefault(o['visual_instance_id'], set()).add(dense[endpoint_ids[o['observation_id']]])
    for note in base.ambiguities:
        o = by_evidence.get(str(note['evidence']))
        if o is None:
            raise ValueError('old ambiguity lacks actual endpoint')
        observed = dense[endpoint_ids[o['observation_id']]]
        candidates = sorted(set().union(*(old_to_new[i] for i in note['candidate_visual_instance_ids'])) - {observed})
        if candidates:
            answer.ambiguities.append(dict(name=note['name'], evidence=deepcopy(note['evidence']),
                observed_visual_instance_id=observed, candidate_visual_instance_ids=candidates,
                reason=note['reason']))
    answer.validate()
    audit = dict(input_actual_segments=len(obs), local_SAM_tracks=len(groups),
        born_records=len(store.records), rejected=len(store.rejected),
        actual_links=sum(map(len, store.viewpoints.values())), physical_identity_verified=0,
        removed_old_identity_unions=True, new_model_calls=0, events=events,
        endpoint_to_instance=endpoint_ids, store_id_to_QA_id=dense, SAM_audit=sam_audit,
        source_ambiguities=deepcopy(base.ambiguities))
    from sandbox_vlm.memory.storage import cpu_storage_report
    audit['adapter_peak_cpu_structural_bytes'] = cpu_storage_report(dict(
        base=base, auxiliary=auxiliary, observations=obs, expanded=expanded, endpoints=endpoints,
        tracks=tracks, store=store, answer=answer, audit=audit, groups=groups,
        endpoint_sources=endpoint_sources, old_to_new=old_to_new))['cpu_structural_bytes']
    return store, answer, audit
