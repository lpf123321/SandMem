"""Conservative development gates, never a calibrated identity classifier."""
from dataclasses import dataclass

import numpy as np

from sandbox_vlm.perception.image_geometry import ImageGridTransform


@dataclass(frozen=True)
class MatchSettings:
    appearance_cosine_min: float
    center_distance_diagonal_max: float
    projection_iou_min: float
    name_policy: str = 'exact'
    birth_policy: str = 'actual_covisibility'

    def __post_init__(self):
        values = (self.appearance_cosine_min, self.center_distance_diagonal_max, self.projection_iou_min)
        if not np.isfinite(values).all() or not (-1 <= values[0] <= 1 and values[1] >= 0 and 0 <= values[2] <= 1):
            raise ValueError('invalid matching gates')
        if self.name_policy not in ('exact', 'last_token'):
            raise ValueError('unsupported name policy')
        if self.birth_policy not in ('actual_covisibility', 'spatial_separation'):
            raise ValueError('unsupported birth policy')


def names_compatible(first, second, policy='exact'):
    """Approved lexical candidate only; no vocabulary or synonym inference.

    Modifiers may differ, but the final whitespace-delimited token must agree.
    Keep words intact: e.g. table lamp never becomes table. This deliberately
    does not infer irregular plurals, synonyms, or compound noun structure.
    """
    a, b = first.casefold().split(), second.casefold().split()
    if policy == 'exact':
        return bool(a and b and a == b)
    if policy != 'last_token':
        raise ValueError('unsupported name policy')
    def singular(token):
        if len(token) > 4 and token.endswith('ies'):
            return token[:-3] + 'y'
        if len(token) > 4 and token.endswith(('ches', 'shes', 'xes', 'zes', 'sses')):
            return token[:-2]
        if len(token) > 3 and token.endswith('s') and not token.endswith(('ss', 'us', 'is')):
            return token[:-1]
        return token
    return bool(a and b and singular(a[-1]) == singular(b[-1]))


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def projected_source_box(box, camera, grid: ImageGridTransform):
    if box.frame != camera.reference or camera.image_size != grid.output_hw:
        raise ValueError('projection reference/grid mismatch')
    uv, valid = camera.project(box.corners())
    # Do not fabricate near-plane clipping when a corner lies behind the camera.
    if not valid.all():
        return None
    uv = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(grid.A_grid_from_source).T
    lower, upper = uv[:, :2].min(axis=0), uv[:, :2].max(axis=0)
    h, w = grid.source_hw
    lower, upper = np.clip(lower, [0, 0], [w, h]), np.clip(upper, [0, 0], [w, h])
    return np.r_[lower, upper] if np.all(upper > lower) else None


def match_global(observation, instances, evidence, cameras, grids, settings: MatchSettings):
    """Require every actual prompt to pass projection; retain alias/ambiguity candidates."""
    rows, passing, near = [], [], []
    if observation.bbox_3d is None or observation.appearance is None or not evidence:
        return dict(status='pending', matched_instance_id=None, candidate_instance_ids=[],
                    reason='missing_geometry_appearance_or_actual_evidence', scores=[])
    for instance in instances:
        if instance.bbox_3d.frame != observation.coordinate_reference:
            continue
        if instance.appearance.dim != observation.appearance.dim:
            continue
        cosine = float(np.dot(instance.appearance.vector.astype(float), observation.appearance.vector.astype(float)))
        diagonal = max(np.linalg.norm(instance.bbox_3d.extents), np.linalg.norm(observation.bbox_3d.extents))
        distance = float(np.linalg.norm(instance.bbox_3d.center - observation.bbox_3d.center))
        ratio = distance / diagonal if diagonal > 0 else (0.0 if distance == 0 else float('inf'))
        geometry_appearance = bool(cosine >= settings.appearance_cosine_min and ratio <= settings.center_distance_diagonal_max)
        ious = []
        for prompt in evidence:
            camera, grid = cameras.get(prompt.frame_id), grids.get(prompt.frame_id)
            projected = None
            if (camera is not None and grid is not None and camera.reference == observation.coordinate_reference
                    and grid.source_hw == prompt.source_hw):
                projected = projected_source_box(instance.bbox_3d, camera, grid)
            ious.append(dict(frame_id=prompt.frame_id, iou=box_iou(projected, prompt.bbox_xyxy) if projected is not None else None))
        same_name = names_compatible(instance.name, observation.name)
        compatible = names_compatible(instance.name, observation.name, settings.name_policy)
        passes = bool(geometry_appearance and all(r['iou'] is not None and r['iou'] >= settings.projection_iou_min for r in ious))
        if geometry_appearance:
            near.append(instance.global_instance_id)
        if passes:
            passing.append((instance.global_instance_id, compatible))
        rows.append(dict(instance_id=instance.global_instance_id, cosine=cosine,
                         center_distance_diagonal_ratio=ratio if np.isfinite(ratio) else None,
                         same_name=same_name, projection=ious, passes_numeric_gates=passes))
        if settings.name_policy != 'exact':
            rows[-1].update(name_compatible=compatible, name_policy=settings.name_policy)
    matched = passing[0][0] if len(passing) == 1 and passing[0][1] else None
    return dict(status='matched' if matched is not None else 'pending', matched_instance_id=matched,
                candidate_instance_ids=sorted(set(near)), reason=None if matched is not None else 'no_unique_same_name_match', scores=rows)
