"""D061: reuse the existing D014 matcher once per actual local observation."""
from copy import deepcopy
import numpy as np
from sandbox_vlm.alignment.association import MatchSettings, match_global
from sandbox_vlm.memory.association import aabb_union
from sandbox_vlm.memory.geometry import BBox3D
from sandbox_vlm.memory.sandbox import AppearanceFeature, ObjectInstance
from sandbox_vlm.perception.evidence import PromptEvidence
from sandbox_vlm.perception.observations import LocalObservation


def associate(rows, cameras, grids, *, reference, video_id, settings):
    """No propagation, reverse pass, geometry veto, or cross-reference alignment.

    The ObjectInstance container is only a local matching hypothesis here;
    accepted edges still become visual hypotheses, not ground-truth identities.
    """
    instances=[];members={};edges=[];decisions=[]
    for row in sorted(rows,key=lambda r:(r['evidence']['frame_id'],r['observation_id'])):
        evidence=PromptEvidence.from_dict(row['evidence']);box=row['bbox_3d'];oid=row['observation_id']
        if box is None:
            decisions.append(dict(observation_id=oid,status='unresolved_missing_geometry',scores=[]));continue
        box=BBox3D.from_dict(box)
        if box.frame!=reference:raise ValueError('mixed local references')
        feature=AppearanceFeature.from_dict(row['appearance'])
        current=LocalObservation(video_id,oid,reference,row['name'],'',[evidence.frame_id],box,feature,
                                 prompt_evidence=[row['evidence']])
        # A current frame cannot be merged with an already co-visible component.
        eligible=[i for i in instances if all(m['evidence']['frame_id']!=evidence.frame_id for m in members[i.global_instance_id])]
        result=match_global(current,eligible,[evidence],cameras,grids,settings)
        gid=result['matched_instance_id']
        if gid is None:
            gid=len(instances);instances.append(ObjectInstance(gid,row['name'],'',box,feature));members[gid]=[row]
        else:
            source=members[gid][-1]
            edges.append(dict(source=source['observation_id'],target=oid,source_frame=source['evidence']['frame_id'],
                target_frame=evidence.frame_id,would_reuse=True,evidence_type='D061_original_D014_local_match',
                reference=reference,matched_local_group=gid,match_result=deepcopy(result)))
            item=instances[gid];item.bbox_3d=aabb_union([item.bbox_3d,box])
            item.appearance=AppearanceFeature(item.appearance.vector.astype(float)+feature.vector.astype(float),dtype=item.appearance.dtype)
            members[gid].append(row)
        decisions.append(dict(observation_id=oid,local_group=gid,**result))
    return edges,decisions
