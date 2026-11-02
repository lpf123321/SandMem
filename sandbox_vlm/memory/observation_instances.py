"""D094: actual-evidence births without cross-track numeric identity fusion."""
from copy import deepcopy

from sandbox_vlm.memory import Sandbox
from sandbox_vlm.memory.relaxed import candidate_registry, candidate_frames


def observation_instances(video_id, observations, strict_decisions, reliability):
    """All eligible local SAM tracks become operational, revisable instances.

    Strict decisions supply possible correspondence edges only, never unions.
    Geometry is retained in its original reference, with reliability separate
    from identity. The only cross-track union is an identical actual prompt.
    """
    if any(o.video_id != video_id for o in observations):
        raise ValueError('cross-video instance birth')
    if len({o.local_observation_id for o in observations}) != len(observations):
        raise ValueError('duplicate local track ID')
    state=Sandbox(video_id=video_id, pending_observations=deepcopy(observations))
    registry=candidate_registry(state)
    mapping=registry['source_to_candidate']
    # Translate the original strict IDs to their actual originating tracks.
    anchors={}
    for oid,d in strict_decisions.items():
        if d['status']=='created' and oid in mapping:
            anchors[d['matched_instance_id']]=mapping[oid]
    edges=set()
    for oid,d in strict_decisions.items():
        if oid not in mapping:continue
        targets=list(d.get('candidate_instance_ids',[]))
        if d['status']=='matched':targets.append(d['matched_instance_id'])
        for target in targets:
            if target not in anchors or anchors[target]==mapping[oid]:continue
            edges.add((min(mapping[oid],anchors[target]),max(mapping[oid],anchors[target]),
                       'original_strict_numeric_correspondence_unconfirmed'))
    neighbors={r['instance_id']:set() for r in registry['records']}
    for a,b,_ in edges:neighbors[a].add(b);neighbors[b].add(a)
    for r in registry['records']:
        r['identity_status']='operational_instance_identity_unconfirmed'
        r['possible_instance_ids']=sorted(neighbors[r['instance_id']])
        r.pop('possible_confirmed_ids')
        for g in r['local_geometries']:
            g['reference_alignment_reliable']=bool(reliability[g['observation_id']])
            # Reliable alignment is not verification of object-unit identity.
            g['global_geometry_verified']=False
    registry.update(representation='D094_actual_observation_instances',
        possible_identity_edges=[dict(instance_ids=[a,b],reason=reason) for a,b,reason in sorted(edges)],
        birth_requires_global_alignment=False,birth_requires_3d=False,
        numeric_cross_track_fusions=0)
    converted=set(mapping)
    state.pending_observations=[o for o in state.pending_observations if o.local_observation_id not in converted]
    decisions={o.local_observation_id:dict(status='registered' if o.local_observation_id in converted else 'pending',
                reason='actual_evidence_birth' if o.local_observation_id in converted else 'missing_birth_evidence')
               for o in observations}
    validate_observation_instances(state,registry)
    return state,registry,decisions


def validate_observation_instances(state,registry):
    ids={r['instance_id'] for r in registry['records']}
    if len(ids)!=len(registry['records']) or ids & state.instance_ids():
        raise ValueError('instance ID collision')
    if set(registry['source_to_candidate'].values())!=ids:
        raise ValueError('detached track mapping')
    linked=set()
    from sandbox_vlm.perception.evidence import PromptEvidence
    seen=set()
    for fid,links in registry['viewpoints'].items():
        for link in links:
            p=PromptEvidence.from_dict(link['evidence']);gid=link['instance_id']
            if int(fid)!=p.frame_id or gid not in ids:raise ValueError('invalid actual link')
            key=(gid,str(p.to_dict()))
            if key in seen:raise ValueError('duplicate actual link')
            seen.add(key);linked.add(gid)
    if linked!=ids:raise ValueError('instance without actual evidence')
    for r in registry['records']:
        if r['identity_status']!='operational_instance_identity_unconfirmed':
            raise ValueError('unsupported physical identity confirmation')
        if not candidate_frames(registry,r['instance_id']):raise ValueError('missing actual frame')
        for g in r['local_geometries']:
            if g['bbox_3d']['frame']!=g['coordinate_reference'] or g['global_geometry_verified']:
                raise ValueError('unverified or mixed-reference geometry')
        if not set(r['possible_instance_ids'])<=ids:raise ValueError('dangling possible edge')
    for edge in registry['possible_identity_edges']:
        if len(set(edge['instance_ids']))!=2 or not set(edge['instance_ids'])<=ids:
            raise ValueError('invalid possible edge')
    return True
