"""Pure evidence filtering for the authorized offline frozen-VGGT track probe.

Unused by the online pipeline. Tracking is a candidate source, never an
instance decision. Every rejection is retained per original anchor query.
"""
import numpy as np

from sandbox_vlm.memory.geometry import transform_points


def prepare_tracking_tokens(tokens, required_layers):
    """Official aggregator keeps only head-required layers; other slots are None."""
    if any(i < 0 or i >= len(tokens) or tokens[i] is None for i in required_layers):
        raise ValueError('missing tracking head layer')
    return [value.float() if value is not None else None for value in tokens]


def filter_track_evidence(anchor_xy, forward, backward, visibility, confidence,
                          reverse_visibility, reverse_confidence, depth, reliable,
                          camera, *, score_min=.9, cycle_max_px=3., visibility_min=None, confidence_min=None):
    visibility_min = score_min if visibility_min is None else visibility_min
    confidence_min = score_min if confidence_min is None else confidence_min
    anchor_xy = np.asarray(anchor_xy)
    count = len(anchor_xy)
    if (not 0 < count <= 512 or anchor_xy.shape != (count, 2)
            or np.asarray(forward).shape != (2, count, 2) or np.asarray(backward).shape != (2, count, 2)
            or any(np.asarray(x).shape != (2, count) for x in
                   (visibility, confidence, reverse_visibility, reverse_confidence))
            or depth.shape != camera.image_size or reliable.shape != depth.shape or reliable.dtype != np.bool_
            or not 0 <= score_min <= 1 or not 0 <= visibility_min <= 1 or not 0 <= confidence_min <= 1
            or not np.isfinite(cycle_max_px) or cycle_max_px <= 0):
        raise ValueError('invalid bounded tracking evidence')
    reasons = np.full(count, 'accepted_candidate', dtype=object)
    def reject(mask, reason):
        reasons[(reasons == 'accepted_candidate') & mask] = reason
    finite = (np.isfinite(forward).all(axis=(0, 2)) & np.isfinite(backward).all(axis=(0, 2))
              & np.isfinite(anchor_xy).all(axis=1))
    reject(~finite, 'nonfinite_or_no_reverse_track')
    target = forward[1]
    h, w = depth.shape
    inside = np.isfinite(target).all(1) & (target[:, 0] >= 0) & (target[:, 0] < w) & (target[:, 1] >= 0) & (target[:, 1] < h)
    reject(~inside, 'tracked_pixel_outside_current_image')
    scores = np.stack([visibility, confidence, reverse_visibility, reverse_confidence])
    threshold = np.array([visibility_min, confidence_min, visibility_min, confidence_min])[:, None, None]
    reject(~np.isfinite(scores).all(axis=(0, 1)) | (scores < threshold).any(axis=(0, 1)), 'track_visibility_or_confidence')
    cycle = np.maximum.reduce([np.linalg.norm(forward[0]-anchor_xy, axis=1),
        np.linalg.norm(backward[1]-anchor_xy, axis=1), np.linalg.norm(backward[0]-target, axis=1)])
    reject(~np.isfinite(cycle) | (cycle > cycle_max_px), 'bidirectional_cycle_failed')
    # No interpolated depth. Reject rounded coordinates outside native support.
    integer = np.zeros((count, 2), dtype=int)
    integer[inside] = np.floor(target[inside]+.5).astype(int)
    native = inside & (integer[:, 0] < w) & (integer[:, 1] < h)
    valid_depth = np.zeros(count, dtype=bool)
    ids = np.flatnonzero(native)
    x, y = integer[ids].T
    valid_depth[ids] = reliable[y, x] & np.isfinite(depth[y, x]) & (depth[y, x] > 0)
    reject(~valid_depth, 'no_reliable_current_depth_at_nearest_pixel')
    # One geometry pixel supplies at most one correspondence. Prefer the
    # highest minimum frozen confidence; original query order breaks ties.
    quality = np.where(np.isfinite(scores), scores, -np.inf).min(axis=(0, 1))
    occupied = set()
    for index in sorted(np.flatnonzero(reasons == 'accepted_candidate'), key=lambda i: (-quality[i], i)):
        key = tuple(integer[index])
        if key in occupied:
            reasons[index] = 'duplicate_current_geometry_pixel'
        else:
            occupied.add(key)
    kept = np.flatnonzero(reasons == 'accepted_candidate')
    pixels = integer[kept].astype(float)
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(camera.intrinsic).T
    points_camera = rays * depth[integer[kept, 1], integer[kept, 0], None]
    points = transform_points(points_camera, camera.T_ref_from_camera)
    return dict(anchor_indices=kept, current_pixel_xy=pixels, current_points=points,
        rejection_reasons=reasons.tolist(), cycle_errors_px=[float(v) if np.isfinite(v) else None for v in cycle],
        tracked_subpixel_xy=[[float(v) if np.isfinite(v) else None for v in xy] for xy in target])
