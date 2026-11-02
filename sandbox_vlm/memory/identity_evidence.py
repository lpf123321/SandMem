"""D095 actual-region identity contracts. No whole-track or geometric promotion."""
import hashlib
import json
import math

import numpy as np

UNIT_STATES=('one_object','part','multiple_objects','unclear')
UNIT_PATTERN=r'\{"unit_status":"(?:one_object|part|multiple_objects|unclear)"\}'


def region_key(region):
    box=region['bbox_xyxy'];h,w=region['source_hw'];fid=region['frame_id']
    if (type(fid) is not int or fid<0 or len(box)!=4 or not all(math.isfinite(x) for x in box)
            or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h):
        raise ValueError('invalid source region')
    return hashlib.sha256(json.dumps([fid,list(box),[h,w]],separators=(',',':')).encode()).hexdigest()


def parse_unit(raw):
    value=json.loads(raw)
    if not isinstance(value,dict) or set(value)!={'unit_status'} or value['unit_status'] not in UNIT_STATES:
        raise ValueError('unit schema mismatch')
    return value['unit_status']


def unit_images(image,region):
    from PIL import Image,ImageDraw
    region_key(region)
    if [image.height,image.width]!=region['source_hw']:raise ValueError('native source shape mismatch')
    box=region['bbox_xyxy'];marked=image.copy()
    ImageDraw.Draw(marked).rectangle(tuple(box),outline=(255,0,0),width=2)
    crop=image.crop((math.floor(box[0]),math.floor(box[1]),math.ceil(box[2]),math.ceil(box[3])))
    marked.thumbnail((384,384),Image.Resampling.LANCZOS);crop.thumbnail((256,256),Image.Resampling.LANCZOS)
    return [marked,crop]


def unit_messages(images,prompt):
    return [dict(role='user',content=[*[dict(type='image',image=i) for i in images],dict(type='text',text=prompt)])]


def mask_iou(a,b):
    if a.dtype!=np.bool_ or b.dtype!=np.bool_ or a.shape!=b.shape or a.ndim!=2:
        raise ValueError('native boolean masks on identical source grids required')
    union=np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else 0.


def unique_candidates(mask,references,threshold=.70):
    # Every region participates; names cannot manufacture uniqueness.
    return sorted({key for key,other in references if mask_iou(mask,other)>=threshold})


def verify_pair(a,b,shared,forward,backward,source_key,target_key):
    if not shared or set(a)!=set(shared) or set(b)!=set(shared):
        raise ValueError('complete original shared frame support required')
    rows=[dict(frame_id=f,iou=mask_iou(a[f],b[f]),nonempty=bool(a[f].any() and b[f].any())) for f in shared]
    accepted=(all(r['nonempty'] and r['iou']>=.70 for r in rows)
              and forward==[target_key] and backward==[source_key])
    return dict(status='supported_local_relation' if accepted else 'identity_unresolved',
                shared_frame_checks=rows,forward_candidates=forward,backward_candidates=backward,
                reason='all_shared_masks_and_both_unique_actual_endpoints' if accepted else 'missing_overlap_or_unique_endpoint_evidence')


def supported_components(edges,cannot_links):
    """Reject entire conflicting or non-clique components; no greedy edge pruning."""
    edges={tuple(sorted(e)) for e in edges if len(set(e))==2}
    cannot={tuple(sorted(e)) for e in cannot_links}
    neighbors={}
    for a,b in edges:neighbors.setdefault(a,set()).add(b);neighbors.setdefault(b,set()).add(a)
    seen=set();accepted=[];rejected=[]
    for node in sorted(neighbors):
        if node in seen:continue
        todo=[node];component=set()
        while todo:
            n=todo.pop()
            if n in component:continue
            component.add(n);todo.extend(neighbors[n]-component)
        seen|=component;members=sorted(component)
        pairs={(a,b) for i,a in enumerate(members) for b in members[i+1:]}
        conflict=bool(pairs&cannot);missing=sorted(pairs-edges)
        if conflict or missing:
            rejected.append(dict(members=members,reason='co_visible_conflict' if conflict else 'unproved_transitive_identity',missing_direct_edges=missing))
        else:accepted.append(members)
    return dict(accepted=accepted,rejected=rejected,identity_scope='actual_regions_only_not_parent_tracks')


def pack_mask(mask):
    if mask.dtype!=np.bool_ or mask.ndim!=2:raise ValueError('boolean mask required')
    return dict(shape=list(mask.shape),bits=np.packbits(mask.reshape(-1)))


def unpack_mask(packed):
    return np.unpackbits(packed['bits'],count=math.prod(packed['shape'])).reshape(packed['shape']).astype(bool)
