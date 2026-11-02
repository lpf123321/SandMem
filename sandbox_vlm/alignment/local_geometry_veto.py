"""D060: same-reference geometry may veto a SAM edge, never create one."""
from copy import deepcopy
import numpy as np


def strictly_separated(a, b):
    if a['frame'] != b['frame']:
        raise ValueError('geometry must share a local coordinate reference')
    ac, ae = np.asarray(a['center']), np.asarray(a['extents'])
    bc, be = np.asarray(b['center']), np.asarray(b['extents'])
    if not all(np.isfinite(v).all() for v in [ac, ae, bc, be]) or (ae < 0).any() or (be < 0).any():
        raise ValueError('invalid boxes')
    return bool(np.any((ac+ae/2 < bc-be/2) | (bc+be/2 < ac-ae/2)))


def filter_edges(edges, windows, *, watermark):
    if any(w['available_through'] > watermark for w in windows):
        raise ValueError('future geometry')
    result, audit = [], []
    for original in edges:
        edge = deepcopy(original); checked = []; veto = False
        for w in windows:
            a = w['observations'].get(edge['source']); b = w['observations'].get(edge['target'])
            if a is None or b is None or a['bbox_3d'] is None or b['bbox_3d'] is None:
                continue
            disjoint = strictly_separated(a['bbox_3d'], b['bbox_3d'])
            checked.append(dict(reference=w['reference'], strict_separation=disjoint))
            veto |= disjoint
        if edge['would_reuse'] and veto:
            edge.update(would_reuse=False, reason='same_clip_geometry_separation_veto')
        result.append(edge)
        audit.append(dict(source=edge['source'], target=edge['target'], geometry_checks=checked,
                          vetoed=bool(original['would_reuse'] and veto),
                          verified_geometry=bool(checked)))
    return result, audit
