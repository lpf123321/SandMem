"""D093 candidate-instance registry; exact actual evidence only, no new geometry."""
from copy import deepcopy
import hashlib
import json

from sandbox_vlm.perception.evidence import PromptEvidence


def actual_key(name, evidence):
    PromptEvidence.from_dict(evidence)
    value=dict(name=' '.join(name.casefold().split()), evidence=evidence)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def candidate_registry(sandbox):
    """Register complete pending SAM tracks and union exact repeated prompts.

    A track is the existing frozen local SAM candidate, not a new physical ID.
    Coordinates remain separate per original reference. Similar features/names
    alone never union tracks. A separate index stores actual observation links.
    """
    pending=sorted(sandbox.pending_observations,key=lambda o:(min(o.frame_ids),o.local_observation_id))
    parent={};owners={};eligible={};rejected=[]
    def root(oid):
        while parent[oid]!=oid:oid=parent[oid]
        return oid
    for obs in pending:
        oid=obs.local_observation_id
        if obs.appearance is None or not obs.prompt_evidence or not obs.name.strip():
            rejected.append(dict(observation_id=oid,reason='missing_feature_or_actual_prompt'));continue
        keys=[actual_key(obs.name,p) for p in obs.prompt_evidence]
        parent[oid]=oid;eligible[oid]=obs
        for key in keys:
            if key in owners:
                a,b=root(oid),root(owners[key]);lo,hi=sorted((a,b));parent[hi]=lo
            else:owners[key]=oid
    groups={}
    for oid in parent:groups.setdefault(root(oid),[]).append(oid)
    records=[];viewpoints={};source_to_candidate={}
    next_id=max(sandbox.instance_ids(),default=-1)+1
    for members in sorted(groups.values(),key=lambda ids:min((min(eligible[o].frame_ids),o) for o in ids)):
        observations=[eligible[o] for o in sorted(members)]
        first=observations[0];gid=next_id;next_id+=1
        record=dict(instance_id=gid,identity_status='candidate_instance',name=first.name,
                    description=first.short_description,appearance=first.appearance.to_dict(),
                    local_geometries=[],possible_confirmed_ids=[])
        seen=set()
        for obs in observations:
            source_to_candidate[obs.local_observation_id]=gid
            if obs.bbox_3d is not None:
                record['local_geometries'].append(dict(observation_id=obs.local_observation_id,
                    bbox_3d=obs.bbox_3d.to_dict(),coordinate_reference=obs.coordinate_reference,
                    global_geometry_verified=False))
            for p in obs.prompt_evidence:
                key=actual_key(obs.name,p)
                if key in seen:continue
                seen.add(key)
                viewpoints.setdefault(str(p['frame_id']),[]).append(dict(instance_id=gid,evidence=deepcopy(p)))
        notes=[n for n in sandbox.ambiguity_notes if n.local_observation_id in members]
        record['possible_confirmed_ids']=sorted({i for n in notes for i in n.candidate_instance_ids})
        records.append(record)
    result=dict(representation='D093_candidates_with_actual_links',records=records,viewpoints=viewpoints,
                source_to_candidate=source_to_candidate,rejected=rejected,
                registration_is_not_physical_identity_confirmation=True)
    validate_registry(sandbox,result)
    return result


def validate_registry(sandbox, registry):
    ids=[r['instance_id'] for r in registry['records']]
    if len(ids)!=len(set(ids)) or set(ids)&sandbox.instance_ids():raise ValueError('candidate ID collision')
    linked=set()
    for fid,links in registry['viewpoints'].items():
        for link in links:
            p=PromptEvidence.from_dict(link['evidence'])
            if p.frame_id!=int(fid) or link['instance_id'] not in ids:raise ValueError('detached candidate link')
            linked.add(link['instance_id'])
    if linked!=set(ids):raise ValueError('candidate without actual link')
    for row in registry['records']:
        if row['identity_status']!='candidate_instance':raise ValueError('candidate promoted without proof')
        for geometry in row['local_geometries']:
            if geometry['bbox_3d']['frame']!=geometry['coordinate_reference']:raise ValueError('mixed reference')
    return True


def candidate_frames(registry,gid):
    return sorted(int(f) for f,links in registry['viewpoints'].items() if any(x['instance_id']==gid for x in links))
