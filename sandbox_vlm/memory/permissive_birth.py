"""D096 incremental evidence-based births; physical identity stays unverified."""
from copy import deepcopy

from sandbox_vlm.memory.relaxed import actual_key
from sandbox_vlm.perception.evidence import PromptEvidence


class PermissiveBirthStore:
    def __init__(self,video_id):
        if not video_id:raise ValueError('explicit video required')
        self.video_id=video_id
        self.records={};self.viewpoints={};self.prompt_index={};self.source_to_instance={}
        self.next_id=0;self.rejected=[];self.possible_identity_edges=[]

    def admit(self,observation,*,alignment_reliable):
        if observation.video_id!=self.video_id:raise ValueError('cross-video observation')
        observation.__post_init__()
        if not observation.name.strip() or not observation.prompt_evidence:
            row=dict(observation_id=observation.local_observation_id,status='rejected',reason='missing_name_or_actual_prompt')
            self.rejected.append(row);return row
        evidence=[PromptEvidence.from_dict(p).to_dict() for p in observation.prompt_evidence]
        if not {p['frame_id'] for p in evidence}<=set(observation.frame_ids):
            raise ValueError('actual prompt outside source track')
        keys=[actual_key(observation.name,p) for p in evidence]
        existing={self.prompt_index[k] for k in keys if k in self.prompt_index}
        oid=observation.local_observation_id
        if oid in self.source_to_instance:existing.add(self.source_to_instance[oid])
        status='updated' if existing else 'created';merged=[]
        if existing:
            gid=min(existing)
            # A bridge consists of identical actual evidence, never a near/appearance match.
            for other in sorted(existing-{gid}):
                target=self.records[gid];old=self.records.pop(other)
                target['local_contexts'].update(old['local_contexts'])
                if target['appearance'] is None:target['appearance']=old['appearance']
                for index in [self.prompt_index,self.source_to_instance]:
                    for key,value in index.items():
                        if value==other:index[key]=gid
                for fid,links in self.viewpoints.items():
                    for link in links:
                        if link['instance_id']==other:link['instance_id']=gid
                self.possible_identity_edges=[dict(edge,instance_ids=sorted({gid if i==other else i for i in edge['instance_ids']})) for edge in self.possible_identity_edges]
                self.possible_identity_edges=[e for e in self.possible_identity_edges if len(e['instance_ids'])>1]
                merged.append(other)
        else:
            gid=self.next_id;self.next_id+=1
            self.records[gid]=dict(instance_id=gid,name=observation.name,description=observation.short_description,
                birth_status='rule_admitted_actual_evidence',physical_identity_status='unverified',
                physical_distinctness_verified=False,unit_status='unverified',
                appearance=None,local_contexts={})
        record=self.records[gid]
        if record['appearance'] is None and observation.appearance is not None:
            record['appearance']=observation.appearance.to_dict()
        record['local_contexts'][oid]=dict(coordinate_reference=observation.coordinate_reference,
            bbox_3d=observation.bbox_3d.to_dict() if observation.bbox_3d is not None else None,
            source_alignment_reliable=bool(alignment_reliable),global_geometry_verified=False,
            appearance_available=observation.appearance is not None)
        self.source_to_instance[oid]=gid;added=0
        for key,p in zip(keys,evidence):
            if key not in self.prompt_index:
                self.viewpoints.setdefault(str(p['frame_id']),[]).append(dict(instance_id=gid,evidence=deepcopy(p),actual_key=key));added+=1
            self.prompt_index[key]=gid
        return dict(observation_id=oid,status=status,instance_id=gid,new_actual_links=added,
                    merged_record_ids=merged,reason='actual_evidence_birth' if status=='created' else 'same_track_or_identical_actual_evidence',
                    birth_confirmed_by_policy=True,physical_identity_verified=False)

    def frames_for_instance(self,gid):
        if gid not in self.records:raise KeyError(gid)
        return sorted(int(f) for f,links in self.viewpoints.items() if any(x['instance_id']==gid for x in links))

    def to_dict(self):
        return dict(video_id=self.video_id,representation='D096_incremental_rule_admitted_instances',
            birth_confirmation_is_not_physical_identity_verification=True,
            records=list(self.records.values()),viewpoints=self.viewpoints,prompt_index=self.prompt_index,
            source_to_instance=self.source_to_instance,next_id=self.next_id,rejected=self.rejected,
            possible_identity_edges=self.possible_identity_edges)

    def validate(self):
        ids=set(self.records);linked=set();keys=set()
        if not set(self.source_to_instance.values())<=ids:raise ValueError('detached source track')
        for fid,links in self.viewpoints.items():
            for link in links:
                p=PromptEvidence.from_dict(link['evidence']);key=link['actual_key'];gid=link['instance_id']
                if p.frame_id!=int(fid) or gid not in ids or key in keys or self.prompt_index.get(key)!=gid:
                    raise ValueError('detached or duplicate actual link')
                keys.add(key);linked.add(gid)
        if keys!=set(self.prompt_index) or linked!=ids:raise ValueError('instance lacks actual support')
        for gid,r in self.records.items():
            if r['birth_status']!='rule_admitted_actual_evidence' or r['physical_identity_status']!='unverified' or r['physical_distinctness_verified']:
                raise ValueError('unsupported physical identity promotion')
            for oid,c in r['local_contexts'].items():
                if self.source_to_instance[oid]!=gid:raise ValueError('detached local geometry')
                if c['bbox_3d'] is not None and c['bbox_3d']['frame']!=c['coordinate_reference']:
                    raise ValueError('mixed local references')
                if c['global_geometry_verified']:raise ValueError('unverified geometry promoted')
        for edge in self.possible_identity_edges:
            if not set(edge['instance_ids'])<=ids or len(set(edge['instance_ids']))!=2:raise ValueError('invalid possible edge')
        return True
